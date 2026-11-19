"""OpenAI-compatible gateway, GGUF inventory, pulls and telemetry dashboard."""
from __future__ import annotations

import asyncio
import faulthandler
import json
import os
import queue
import re
import signal
import tempfile
import threading
import time
import uuid

try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:
    pass
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from engine.accelerator import AcceleratorConfig, discover_hardware, memory_snapshot, resolve_config
from engine.inventory import ModelInventory
from engine.learner import ActiveExpertHandler
from engine.manifest import build_expert_manifest, model_profile_key
from engine.optimizer import optimize_storage_tiers
from engine.planner import make_load_plan
from engine.runtime import NativeRuntimeError, runtime
from moe_boost.cache import ExpertCache
from moe_boost.cache import safe_ram_budget
from moe_boost.pager import pager

def _default_models_dir() -> Path:
    env_dir = os.getenv("MOE_MODELS_DIR")
    if env_dir:
        env_path = Path(env_dir).expanduser().resolve()
        if env_path.is_dir() or (env_path.parent.is_dir() and os.access(str(env_path.parent), os.W_OK)):
            return env_path
    local_dir = (ROOT / "models").resolve()
    if any(local_dir.glob("*.gguf")):
        return local_dir
    ollama_dir = Path("/mnt/ollama-models/OllamaModels/models")
    if ollama_dir.is_dir() and any(ollama_dir.glob("*.gguf")):
        return ollama_dir.resolve()
    media_dir = Path("/media/local-ai/Model")
    try:
        if media_dir.is_dir() and any(media_dir.glob("*.gguf")):
            return media_dir.resolve()
    except (PermissionError, OSError):
        pass
    return local_dir


ROOT = Path(__file__).resolve().parent
MODELS_DIR = _default_models_dir()
EXPERT_DATA_DIR = (Path(os.environ["MOE_EXPERT_DATA_DIR"]).expanduser().resolve()
                   if os.getenv("MOE_EXPERT_DATA_DIR") else None)
EXPERT_PROFILE_INDEX = ROOT / ".expert_profile_index.json"
SETTINGS_FILE = ROOT / ".gui_settings.json"
MANIFEST_PATH = os.getenv("MOE_EXPERT_MANIFEST", str(ROOT / "experts.json"))
TELEMETRY_PATH = os.getenv("MOE_TELEMETRY_PATH", str(ROOT / "expert_telemetry.json"))
inventory = ModelInventory(str(MODELS_DIR))
cache = ExpertCache(MANIFEST_PATH, TELEMETRY_PATH)
runtime.cache = cache
expert_handler = ActiveExpertHandler(cache, warm_fraction=float(os.getenv("MOE_WARM_FRACTION", "0.0")), max_warm_experts=int(os.getenv("MOE_MAX_WARM_EXPERTS", "0")))
prepared_models: dict[str, dict] = {}
pull_jobs: dict[str, dict] = {}
pull_tasks: set[asyncio.Task] = set()
optimization_jobs: dict[str, dict[str, Any]] = {}
last_tier_optimization: dict[str, Any] | None = None


def format_duration_ms(ms: float | int | None) -> str:
    """Auto-adjust time duration across hours, minutes, seconds (h-m-s), and ms."""
    if ms is None:
        return "—"
    value_ms = max(0.0, float(ms))
    if value_ms == 0.0:
        return "0.00s"
    total_seconds = value_ms / 1000.0
    if total_seconds >= 3600.0:
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60.0
        return f"{hours}h {minutes}m {seconds:.1f}s"
    if total_seconds >= 60.0:
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60.0
        return f"{minutes}m {seconds:.1f}s"
    if total_seconds >= 1.0:
        return f"{total_seconds:.2f}s"
    return f"{total_seconds:.2f}s ({value_ms:.1f} ms)"


def format_size_gb_mb(num_bytes: int | float | None) -> str:
    """Auto-adjust byte sizes between GB and MB (including RAM and SSD read totals)."""
    if num_bytes is None:
        return "—"
    value = max(0.0, float(num_bytes))
    gb = value / (1024.0 ** 3)
    mb = value / (1024.0 ** 2)
    if value >= 1024.0 ** 3:
        return f"{gb:.2f} GB"
    if value == 0.0:
        return "0.00 MB"
    if mb < 0.01:
        return f"{mb:.4f} MB"
    return f"{mb:.2f} MB"


def format_speed_mbs(mbs: float | int | None) -> str:
    """Auto-adjust throughput between MB/s and GB/s."""
    if mbs is None:
        return "0.00 MB/s"
    value = max(0.0, float(mbs))
    if value >= 1024.0:
        return f"{value / 1024.0:.2f} GB/s"
    return f"{value:.2f} MB/s"


def build_tier_io(vram_snap: dict | None = None, ssd_metrics: dict | None = None) -> dict[str, dict[str, Any]]:
    vram_data = vram_snap if vram_snap is not None else cache.vram_cache.snapshot()
    ssd_data = ssd_metrics if ssd_metrics is not None else pager.metrics()

    # If queue_engine has active workstation executions, incorporate live VRAM & RAM tool throughput
    qe = getattr(cache, "queue_engine", None)
    vram_ms = float(vram_data.get("read_time_ms") or 0.0)
    vram_bytes = int(vram_data.get("bytes_read") or 0)
    vram_ops = int(vram_data.get("read_ops") or 0)
    vram_speed = float(vram_data.get("read_speed_mbs") or 0.0)
    if qe is not None and qe.tool_jobs_completed > 0:
        exp_sz = 2506752
        v_bytes_work = qe.tool_jobs_completed * exp_sz
        v_ms_work = max(0.1, qe.tool_total_time_ms)
        vram_bytes = max(vram_bytes, v_bytes_work)
        vram_ms = max(vram_ms, v_ms_work)
        vram_ops = max(vram_ops, qe.tool_jobs_completed)
        if vram_speed <= 0.0 and v_ms_work > 0:
            vram_speed = (v_bytes_work / (1024.0 * 1024.0)) / (v_ms_work / 1000.0)

    ram_ms = float(cache.ram_read_time_ms or 0.0)
    ram_bytes = int(cache.ram_bytes_read or 0)
    ram_ops = int(cache.ram_read_ops or 0)
    ram_speed = ((ram_bytes / (1024.0 * 1024.0)) / (ram_ms / 1000.0)) if ram_ms > 0 else 0.0
    if qe is not None and (qe.hand_jobs_completed > 0 or len(qe._ram_workstation_slots) > 0):
        exp_sz = 2506752
        r_jobs = max(qe.hand_jobs_completed, len(qe._ram_workstation_slots))
        r_bytes_work = r_jobs * exp_sz
        r_ms_work = max(0.1, qe.hand_total_time_ms or (r_jobs * 1.5))
        ram_bytes = max(ram_bytes, r_bytes_work)
        ram_ms = max(ram_ms, r_ms_work)
        ram_ops = max(ram_ops, r_jobs)
        if ram_speed <= 0.0 and r_ms_work > 0:
            ram_speed = (r_bytes_work / (1024.0 * 1024.0)) / (r_ms_work / 1000.0)

    ssd_ms = float(ssd_data.get("time_used_for_reading_ms") or 0.0)
    ssd_bytes = int(ssd_data.get("bytes_read") or 0)
    ssd_ops = int(ssd_data.get("read_ops") or 0)
    ssd_speed = float(ssd_data.get("actual_ssd_throughput_mbs") or 0.0)

    return {
        "vram": {
            "read_time_ms": round(vram_ms, 3),
            "read_time_seconds": round(vram_ms / 1000.0, 4),
            "read_time_formatted": format_duration_ms(vram_ms),
            "read_bytes": vram_bytes,
            "read_size_formatted": format_size_gb_mb(vram_bytes),
            "read_ops": vram_ops,
            "read_speed_mbs": round(vram_speed, 2),
            "read_speed_formatted": format_speed_mbs(vram_speed),
        },
        "ram": {
            "read_time_ms": round(ram_ms, 3),
            "read_time_seconds": round(ram_ms / 1000.0, 4),
            "read_time_formatted": format_duration_ms(ram_ms),
            "read_bytes": ram_bytes,
            "read_size_formatted": format_size_gb_mb(ram_bytes),
            "read_ops": ram_ops,
            "read_speed_mbs": round(ram_speed, 2),
            "read_speed_formatted": format_speed_mbs(ram_speed),
        },
        "ssd": {
            "read_time_ms": round(ssd_ms, 3),
            "read_time_seconds": round(ssd_ms / 1000.0, 4),
            "read_time_formatted": format_duration_ms(ssd_ms),
            "read_bytes": ssd_bytes,
            "read_size_formatted": format_size_gb_mb(ssd_bytes),
            "read_ops": ssd_ops,
            "read_speed_mbs": round(ssd_speed, 2),
            "read_speed_formatted": format_speed_mbs(ssd_speed),
        },
    }


runtime_metrics: dict[str, Any] = {
    "generation_tokens_per_second": None, "prefill_tokens_per_second": None,
    "tok_per_sec": {"prefill": None, "generation": None},
    "generation_speed_is_estimated": False, "last_request_seconds": None,
    "last_request_formatted": "—",
    "last_time_to_first_token_seconds": None,
    "last_time_to_first_token_formatted": "—",
    "last_prompt_tokens": None, "last_completion_tokens": None,
    "last_prompt_eval_seconds": None, "last_prompt_eval_formatted": "—",
    "last_generation_eval_seconds": None, "last_generation_eval_formatted": "—",
}
accelerator_settings = AcceleratorConfig()
expert_profile_dirs: dict[str, Path] = {}
expert_profile_names: dict[str, str] = {}
expert_profile_models: dict[str, str] = {}
expert_profile_stems: dict[str, str] = {}
loaded_expert_profiles: dict[str, str] = {}


def expert_profile_directory(details: dict) -> tuple[str, Path]:
    profile_id = model_profile_key(details)
    metadata = details.get("metadata") or {}
    model_name = str(metadata.get("general.name") or details.get("name") or
                     Path(details["path"]).stem)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("._-")[:80] or "model"
    directory = ((EXPERT_DATA_DIR / f"{slug}-{profile_id}") if EXPERT_DATA_DIR else
                 (Path(details["path"]).expanduser().resolve().parent / ".moe_experts" /
                  f"{slug}-{profile_id}"))
    try:
        profile_index = json.loads(EXPERT_PROFILE_INDEX.read_text(encoding="utf-8"))
        if not isinstance(profile_index, dict):
            profile_index = {}
    except (OSError, ValueError, TypeError):
        profile_index = {}
    previous_value = profile_index.get(profile_id)
    previous = Path(str(previous_value)).expanduser() if previous_value else None
    if (not directory.exists() and previous is not None and previous.is_dir()
            and previous.resolve() != directory.resolve()):
        prior_records = (previous / f"{slug}.telemetry.json", previous / "telemetry.json")
        source = next((item for item in prior_records if item.is_file()), None)
        if source:
            try:
                prior_payload = json.loads(source.read_text(encoding="utf-8"))
                _write_json_atomic(directory / f"{slug}.telemetry.json", prior_payload)
            except (OSError, ValueError, TypeError):
                pass
    expert_profile_dirs[profile_id] = directory
    expert_profile_names[profile_id] = model_name
    expert_profile_models[profile_id] = str(Path(details["path"]).expanduser().resolve())
    expert_profile_stems[profile_id] = slug
    profile_index[profile_id] = str(directory)
    _write_json_atomic(EXPERT_PROFILE_INDEX, profile_index)
    return model_name, directory


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


async def load_model_expert_data(details: dict, manifest: dict) -> Path:
    """Restore a model's learned heat profile and refresh ranges after path changes."""
    profile_id = model_profile_key(details)
    model_path = str(Path(details["path"]).expanduser().resolve())
    model_ids = set(manifest)
    if (loaded_expert_profiles.get(profile_id) == model_path
            and model_ids.issubset(cache.manifest)):
        return expert_profile_dirs[profile_id]
    model_name, directory = expert_profile_directory(details)
    directory.mkdir(parents=True, exist_ok=True)
    stem = expert_profile_stems[profile_id]
    telemetry_path = directory / f"{stem}.telemetry.json"
    if not telemetry_path.exists():
        legacy_path = directory / "telemetry.json"
        if legacy_path.is_file():
            telemetry_path = legacy_path
    try:
        profile = json.loads(telemetry_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        profile = {}
    expert_profiles = profile.get("experts", {})
    if not isinstance(expert_profiles, dict):
        expert_profiles = {}
    async with cache.lock:
        cache.manifest.update(manifest)
        for key in model_ids:
            expert_record = expert_profiles.get(key, {})
            legacy_invocations = (profile.get("hits", {}).get(key, 0) +
                                  profile.get("misses", {}).get(key, 0))
            prior_invocations = max(int(profile.get("invocations", {}).get(key, 0)),
                                    int(legacy_invocations),
                                    int(expert_record.get("invocations", 0)))
            cache.invocations[key] = max(cache.invocations.get(key, 0), int(prior_invocations))
            cache.hits[key] = max(cache.hits.get(key, 0), int(profile.get("hits", {}).get(
                key, expert_record.get("cache_hits", 0))))
            cache.misses[key] = max(cache.misses.get(key, 0), int(profile.get("misses", {}).get(key, 0)))
            cache.last_used[key] = max(cache.last_used.get(key, 0.0),
                                       float(profile.get("last_used", {}).get(
                                           key, expert_record.get("last_used", 0.0))))
            cache.load_time_ms[key] = max(cache.load_time_ms.get(key, 0.0),
                                          float(profile.get("load_time_ms", {}).get(
                                              key, expert_record.get("load_time_ms", 0.0))))
            cache.stream_time_ms[key] = max(cache.stream_time_ms.get(key, 0.0),
                                            float(profile.get("stream_time_ms", {}).get(
                                                key, expert_record.get("stream_time_ms", 0.0))))
            cache.stream_bytes[key] = max(cache.stream_bytes.get(key, 0),
                                          int(profile.get("stream_bytes", {}).get(
                                              key, expert_record.get("stream_bytes", 0))))
        for source, targets in profile.get("transitions", {}).items():
            if source not in model_ids or not isinstance(targets, dict):
                continue
            row = cache.transitions.setdefault(source, {})
            for target, count in targets.items():
                if target in model_ids:
                    row[target] = max(row.get(target, 0), int(count))
    _write_json_atomic(directory / f"{stem}.manifest.json", {"model_name": model_name,
                                                       "model_profile_id": profile_id,
                                                       "model_path": model_path,
                                                       "experts": manifest})
    loaded_expert_profiles[profile_id] = model_path
    return directory


async def persist_model_expert_data(profile_id: str) -> None:
    prefix = f"{profile_id}."
    async with cache.lock:
        model_ids = {key for key in cache.manifest if key.startswith(prefix)}
        manifest = {key: cache.manifest[key] for key in model_ids}
        telemetry = {
            "version": 1, "model_profile_id": profile_id, "updated_at": time.time(),
            "hits": {key: value for key, value in cache.hits.items() if key in model_ids},
            "misses": {key: value for key, value in cache.misses.items() if key in model_ids},
            "invocations": {key: value for key, value in cache.invocations.items() if key in model_ids},
            "last_used": {key: value for key, value in cache.last_used.items() if key in model_ids},
            "load_time_ms": {key: value for key, value in cache.load_time_ms.items() if key in model_ids},
            "stream_time_ms": {key: value for key, value in cache.stream_time_ms.items() if key in model_ids},
            "stream_bytes": {key: value for key, value in cache.stream_bytes.items() if key in model_ids},
            "transitions": {source: {target: count for target, count in targets.items()
                                      if target in model_ids}
                            for source, targets in cache.transitions.items() if source in model_ids},
        }
        total_invocations = sum(telemetry["invocations"].values())
        resident = set(cache.entries)
        vram_resident = set(cache.vram_cache.entries)
        ranked_invs = sorted(telemetry["invocations"].values(), reverse=True)
        hot_threshold_inv = ranked_invs[max(0, len(ranked_invs) // 5)] if ranked_invs else 1
        warm_threshold_inv = ranked_invs[max(0, len(ranked_invs) // 2)] if ranked_invs else 1
        telemetry["experts"] = {
            key: {"invocations": telemetry["invocations"].get(key, 0),
                  "invocation_share_percent": (100 * telemetry["invocations"].get(key, 0) /
                                               max(1, total_invocations)),
                  "temperature": "hot" if telemetry["invocations"].get(key, 0) >= hot_threshold_inv and telemetry["invocations"].get(key, 0) > 0 else
                                ("warm" if telemetry["invocations"].get(key, 0) >= warm_threshold_inv and telemetry["invocations"].get(key, 0) > 0 else "cold"),
                  "storage_tier": ("VRAM" if key in vram_resident else
                                   ("RAM" if key in resident else "SSD")),
                  "cache_hits": telemetry["hits"].get(key, 0),
                  "ssd_loads": telemetry["misses"].get(key, 0),
                  "load_time_ms": telemetry["load_time_ms"].get(key, 0.0),
                  "average_load_ms": (telemetry["load_time_ms"].get(key, 0.0) /
                                      max(1, telemetry["misses"].get(key, 0))),
                  "stream_time_ms": telemetry["stream_time_ms"].get(key, 0.0),
                  "stream_bytes": telemetry["stream_bytes"].get(key, 0),
                  "last_used": telemetry["last_used"].get(key)}
            for key in model_ids
        }
    directory = expert_profile_dirs.get(profile_id)
    if directory is None:
        directory = Path(MANIFEST_PATH).expanduser().resolve().parent / "expert_data" / profile_id
    stem = expert_profile_stems.get(profile_id, profile_id)
    _write_json_atomic(directory / f"{stem}.manifest.json", {"model_name": expert_profile_names.get(profile_id),
                                                       "model_profile_id": profile_id,
                                                       "model_path": expert_profile_models.get(profile_id),
                                                       "experts": manifest})
    telemetry["model_name"] = expert_profile_names.get(profile_id)
    _write_json_atomic(directory / f"{stem}.telemetry.json", telemetry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.warm_boot_experts = await expert_handler.warm_boot()
    monitor_task = asyncio.create_task(cache_memory_loop())
    yield
    monitor_task.cancel()
    await asyncio.gather(monitor_task, return_exceptions=True)
    await expert_handler.close()
    await cache.save_telemetry()
    cache.close_handles()
    for task in list(pull_tasks):
        task.cancel()


app = FastAPI(title="MoE Boost by Gravitize.IGP", version="0.2.0", lifespan=lifespan)


async def cache_memory_loop() -> None:
    counter = 0
    while True:
        await cache.rebalance()
        counter += 1
        # Periodically persist hot expert telemetry to disk every 30 seconds
        if counter % 15 == 0:
            try:
                await cache.save_telemetry()
            except Exception:
                pass
        await asyncio.sleep(2.0)


class PullRequest(BaseModel):
    repo_id: str = Field(min_length=3, max_length=200)
    files: list[str] | None = None


class AcceleratorRequest(BaseModel):
    backend: str = "auto"
    gpu_layers: int = -1
    vk_queues: int = 1
    cuda_streams: int = 1
    vram_headroom_mb: int = 400


class LoadCheckRequest(BaseModel):
    model_id: str


class KvCacheConfigRequest(BaseModel):
    type_k: str | None = None
    type_v: str | None = None
    type_kv: str | None = None


class MaxRamConfigRequest(BaseModel):
    max_ram: str | int | float | None = None

class RuntimeLoadRequest(BaseModel):
    model_id: str
    mtp_enabled: bool | None = None
    mtp_mode: str | None = None
    mtp_num_tokens: int | None = None
    mtp_ngram_n: int | None = None
    kv_cache_type_k: str | None = None
    kv_cache_type_v: str | None = None
    max_ram: str | int | float | None = None
    gpu_layers: int | None = None
    moe_staging_mode: bool | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    system_prompt: str | None = None


class RuntimeOptimizeRequest(BaseModel):
    model_id: str
    max_context: int = Field(default=4096, ge=512, le=32768)


class TierOptimizeRequest(BaseModel):
    model_id: str | None = None
    max_prefetch_experts: int = Field(default=32, ge=0, le=256)
    max_prefetch_bytes: int = Field(default=256 * 1024 * 1024, ge=0, le=2 * 1024 * 1024 * 1024)
    max_vram_experts: int | None = Field(default=None, ge=0, le=4096)



class UnloadRequest(BaseModel):
    model_id: str
    timeout_seconds: float = Field(default=10.0, ge=0.0, le=120.0)
    drop_page_cache: bool = True


class CacheDropRequest(BaseModel):
    unload_runtime: bool = False
    drop_page_cache: bool = True


class MtpConfigRequest(BaseModel):
    enabled: bool = True
    mode: str = "auto"
    num_tokens: int = Field(default=3, ge=1, le=16)
    ngram_n: int = Field(default=3, ge=1, le=8)


class ModelsDirectoryRequest(BaseModel):
    path: str = Field(min_length=1, max_length=4096)


class ExpertActivationRequest(BaseModel):
    session_id: str = Field(default="diagnostic", max_length=128)
    expert_ids: list[str] = Field(default_factory=list, max_length=256)
    token_routes: list[list[str]] | None = None


def layer_size_map(details: dict) -> dict[int, int | None]:
    result: dict[int, int | None] = {}
    for layer_id, tensors in details.get("layers", {}).items():
        sizes = [tensor.get("byte_size") for tensor in tensors]
        result[int(layer_id)] = (sum(sizes) if sizes and all(size is not None for size in sizes) else None)
    return result


def runtime_accelerator_plan(details: dict, gpu_layers_override: int | None = None) -> dict[str, Any]:
    config = AcceleratorConfig(**accelerator_settings.__dict__)
    if config.backend == "auto":
        config.backend = runtime.build_backend
    if gpu_layers_override is not None:
        config.gpu_layers = int(gpu_layers_override)
    elif "MOE_GPU_LAYERS" in os.environ and os.environ["MOE_GPU_LAYERS"] != "":
        try:
            config.gpu_layers = int(os.environ["MOE_GPU_LAYERS"])
        except ValueError:
            config.gpu_layers = -1  # auto
    else:
        config.gpu_layers = -1  # Auto-optimize by default
    if "MOE_VRAM_HEADROOM_MB" in os.environ:
        config.vram_headroom_mb = int(os.environ["MOE_VRAM_HEADROOM_MB"])
    sizes = layer_size_map(details)
    plan = resolve_config(config, details.get("block_count"), sizes)
    if plan.get("backend") == "cuda" and "vulkan" in runtime.fallback_backends:
        devices = discover_hardware().get("devices", [])
        cuda_device = plan.get("selected_device") or {}
        matching = next((item for item in devices if item.get("backend") == "vulkan"
                         and item.get("device_type") == "discrete_gpu"
                         and item.get("name") == cuda_device.get("name")), None)
        if matching:
            plan["fallback_vulkan_index"] = matching.get("vulkan_index")
    device = plan.get("selected_device") or {}
    # Account for VRAM currently occupied by our own loaded runtime or staging cache
    # so repeat planning calls (chat turns, auto-optimizer) do not under-count free VRAM.
    if plan.get("backend") in {"cuda", "vulkan"} and device.get("vram_free_mb") is not None:
        reclaimable_bytes = int(cache.vram_cache.snapshot().get("resident_bytes") or 0)
        rt_status = runtime.status()
        if rt_status.get("available") and rt_status.get("backend") == plan.get("backend"):
            active_layers = int(rt_status.get("gpu_layers") or 0)
            if active_layers > 0 and sizes:
                ordered_sizes = [sizes[lid] or 0 for lid in sorted(sizes, reverse=True)[:active_layers]]
                reclaimable_bytes += sum(ordered_sizes) + (150 * 1024 * 1024)
        if reclaimable_bytes > 0:
            reclaimable_mb = reclaimable_bytes // (1024 * 1024)
            total_mb = int(device.get("vram_total_mb") or (int(device["vram_free_mb"]) + reclaimable_mb))
            effective_free_mb = min(max(0, total_mb - 64), int(device["vram_free_mb"]) + reclaimable_mb)
            plan["usable_vram_mb"] = max(0, effective_free_mb - config.vram_headroom_mb)
    if plan.get("backend") == "vulkan" and device.get("device_type") == "integrated_gpu":
        # UMA offload is host RAM, so keep it to a small fraction of the safe
        # cache budget instead of treating the reported heap like discrete VRAM.
        budget = min(int((plan.get("usable_vram_mb") or 0) * 1024 * 1024),
                     int(safe_ram_budget() * 0.15))
        used = layers = 0
        for layer_id in sorted(sizes, reverse=True):
            size = sizes[layer_id]
            if size is None or used + size > budget:
                break
            used += size
            layers += 1
        plan["gpu_layers"] = layers
        plan["uma_layer_budget_bytes"] = budget
        plan["uma_layer_bytes"] = used
        if not layers:
            plan["note"] = "Integrated GPU shares system RAM; no layer fits the conservative UMA budget, so inference uses CPU."
    if details.get("classification") == "MoE":
        # Smart partial GPU layer offload & VRAM chair allocation for MoE models:
        # Auto-optimizes how many layers to offload and how many chairs to allocate
        # based on detected system VRAM.
        if config.gpu_layers == 0 or plan.get("backend") == "cpu":
            plan["gpu_layers"] = 0
            plan["vram_chairs"] = 0
        elif sizes and all(size is not None for size in sizes.values()):
            vram_total_mb = int((plan.get("selected_device") or {}).get("vram_total_mb") or 3072)
            # Minimum chair reservation: 500 chairs (~1,250 MB)
            min_chairs = max(500, int(os.getenv("MOE_MIN_VRAM_CHAIRS", "500")))
            avg_expert_mb = 2.506752  # ~2.5 MB average expert size
            chair_reserve_mb = int(min_chairs * avg_expert_mb)
            # Headroom for compute scratch buffers, Flash-Attention, and baseline KV-cache
            compute_headroom_mb = int(os.getenv("MOE_VRAM_HEADROOM_MB", "400"))
            total_reserve_mb = chair_reserve_mb + compute_headroom_mb

            # Available budget for full MoE layers offload
            layer_budget_bytes = max(0, vram_total_mb - total_reserve_mb) * 1024 * 1024
            used_vram = 0
            safe_moe_layers = 0
            max_allowed = details.get("block_count") or len(sizes)
            if config.gpu_layers > 0:
                max_allowed = min(max_allowed, config.gpu_layers)

            for layer_id in sorted(sizes, reverse=True):
                layer_bytes = sizes[layer_id] or 0
                if used_vram + layer_bytes > layer_budget_bytes or safe_moe_layers >= max_allowed:
                    break
                used_vram += layer_bytes
                safe_moe_layers += 1

            plan["gpu_layers"] = safe_moe_layers
            plan["moe_vram_layer_bytes"] = used_vram

            # Now auto-optimize chairs from remaining VRAM:
            # Remaining VRAM = total VRAM - layer VRAM - compute headroom
            layer_vram_mb = used_vram / (1024 * 1024)
            remaining_for_staging = max(0.0, vram_total_mb - layer_vram_mb - 200.0)
            allocated_chairs = max(min_chairs, int((remaining_for_staging * 0.85) // avg_expert_mb))
            plan["vram_chairs"] = allocated_chairs
            plan["kv_cache_headroom_mb"] = max(0.0, vram_total_mb - layer_vram_mb - (allocated_chairs * avg_expert_mb))

            # Update live queue_engine chairs if running
            if hasattr(cache, "queue_engine") and cache.queue_engine is not None:
                cache.queue_engine.update_allocation(allocated_chairs, plan)
        else:
            plan["gpu_layers"] = 0
            plan["vram_chairs"] = 0
        plan["expert_weights_backend"] = "hybrid-vram-host-mmap" if plan["gpu_layers"] > 0 else "host-mmap"
        plan["note"] = (f"MoE Auto-Optimized: {plan['gpu_layers']} layers on CUDA + {plan.get('vram_chairs', 500)} VRAM chairs "
                        f"+ {round(plan.get('kv_cache_headroom_mb', 0.0), 1)} MB KV-cache headroom.")
    return plan


def heat_summary(limit: int = 1050) -> dict:
    pager_metrics = pager.metrics()
    heat = pager_metrics
    counts = dict(heat.get("expert_heat", {}))
    for key in cache.manifest:
        counts[key] = max(counts.get(key, 0), cache.invocations.get(key, 0))
    all_counts = {key: counts.get(key, 0) for key in set(counts) | set(cache.manifest)}
    ranked = sorted(all_counts.items(), key=lambda item: item[1], reverse=True)
    observed = [(key, count) for key, count in ranked if count > 0]
    total = sum(count for _, count in ranked)
    hot_n = max(1, (len(observed) + 4) // 5) if observed else 0
    warm_n = max(1, (len(observed) + 1) // 2) if observed else 0
    hot_inv_cutoff = observed[hot_n - 1][1] if hot_n <= len(observed) and hot_n > 0 else 1
    warm_inv_cutoff = observed[warm_n - 1][1] if warm_n <= len(observed) and warm_n > 0 else 1
    hot_invocations = sum(count for _, count in observed[:hot_n])
    resident = set(cache.entries)
    vram_resident = set(cache.vram_cache.entries)
    # When runtime offloads layers to GPU (n_gpu_layers > 0), all expert tensors belonging
    # to those layers are resident in GPU VRAM managed by llama.cpp.
    rt_gpu_layers = int(getattr(runtime, "_active_gpu_layers", 0) or getattr(runtime, "gpu_layers", 0) or 0)
    def _layer_of_key(key: str) -> int | None:
        try:
            parts = key.split(".")
            for part in parts:
                if part.startswith("layer") and part[5:].isdigit():
                    return int(part[5:])
        except Exception:
            pass
        return None

    def _resolve_residency(key: str, temp: str | None = None) -> tuple[str, str]:
        layer_num = _layer_of_key(key)
        count = counts.get(key, 0)
        t = temp if temp is not None else ("hot" if count >= hot_inv_cutoff and count > 0 else
                                           ("warm" if count >= warm_inv_cutoff and count > 0 else "cold"))
        # 1. VRAM Priority: Layers offloaded to GPU in runtime, or hot promoted in VRAM cache
        if (layer_num is not None and layer_num < rt_gpu_layers) or key in vram_resident:
            tier_desc = "VRAM · GPU layer offloaded" if (layer_num is not None and layer_num < rt_gpu_layers) else "VRAM · hot cache"
            return "VRAM", tier_desc

        # 2. RAM: Resident in RAM cache (staged hot/warm experts)
        if key in resident:
            if t == "hot":
                return "RAM", "RAM · hot staged"
            elif t == "warm":
                return "RAM", "RAM · warm staged"
            return "RAM", "RAM · cold resident"

        # 3. SSD: Cold experts + overflow (hot/warm un-staged or demoted to disk)
        if t == "hot":
            return "SSD", "SSD · hot overflow"
        elif t == "warm":
            return "SSD", "SSD · warm overflow"
        return "SSD", "SSD · cold overflow"

    active = {row["expert_id"] for row in expert_handler.active_snapshot()}
    total_for_share = total or 1

    def _expert_bytes(key: str) -> int:
        try:
            item = cache.manifest.get(key)
            if item:
                return int(item.get("size") or 0)
        except Exception:
            pass
        return 2506752

    def _calc_streamed_bytes(key: str, count: int, residency: str) -> tuple[int, int, int]:
        exp_sz = _expert_bytes(key)
        ssd_b = cache.stream_bytes.get(key, 0)
        if ssd_b <= 0:
            ssd_loads = cache.misses.get(key, 0)
            ssd_b = ssd_loads * exp_sz
        ram_hits = cache.hits.get(key, 0)
        vram_hits = 0
        if hasattr(cache, "vram_cache") and hasattr(cache.vram_cache, "entries"):
            v_entry = cache.vram_cache.entries.get(key)
            if v_entry is not None:
                vram_hits = getattr(v_entry, "hits", 0)
        layer_num = _layer_of_key(key)
        if layer_num is not None and layer_num < rt_gpu_layers:
            vram_b = count * exp_sz
            ram_b = 0
            ssd_b = 0
        elif residency == "VRAM":
            vram_b = (vram_hits or count) * exp_sz
            ram_b = ram_hits * exp_sz
        elif residency == "RAM":
            vram_b = vram_hits * exp_sz
            ram_b = (ram_hits or count) * exp_sz
        else:
            vram_b = vram_hits * exp_sz
            ram_b = ram_hits * exp_sz
        return vram_b, ram_b, ssd_b

    def _build_expert_entry(key: str, count: int) -> dict:
        temp = "hot" if count >= hot_inv_cutoff and count > 0 else (
            "warm" if count >= warm_inv_cutoff and count > 0 else "cold")
        res, tier = _resolve_residency(key, temp)
        vram_b, ram_b, ssd_b = _calc_streamed_bytes(key, count, res)
        layer_num = _layer_of_key(key)
        in_vram = (layer_num is not None and layer_num < rt_gpu_layers) or (key in vram_resident)
        in_ram = (key in resident) and not in_vram
        in_ssd = (not in_vram) and (not in_ram)

        # Confirm that the physical location exactly matches the designated storage tier
        actual_location = "VRAM" if in_vram else ("RAM" if in_ram else "SSD")
        tier_matched = (res == actual_location)
        is_loaded = in_vram or in_ram

        return {
            "expert_id": key,
            "invocations": count,
            "share_percent": count * 100.0 / total_for_share,
            "temperature": temp,
            "residency": res,
            "tier": tier,
            "actual_location": actual_location,
            "tier_matched": tier_matched,
            "loaded": is_loaded,
            "cache_hits": cache.hits.get(key, 0),
            "ssd_loads": cache.misses.get(key, 0),
            "average_load_ms": round(cache.load_time_ms.get(key, 0.0) /
                                      max(1, cache.misses.get(key, 0)), 2),
            "stream_time_ms": round(cache.stream_time_ms.get(key, 0.0), 2),
            "stream_bytes": ssd_b,
            "ssd_streamed_bytes": ssd_b,
            "ram_streamed_bytes": ram_b,
            "vram_streamed_bytes": vram_b,
            "active": key in active or (_layer_of_key(key) is not None and _layer_of_key(key) < rt_gpu_layers)
        }

    # Group experts across storage tiers so filtering by VRAM, RAM, or SSD always displays
    # representative, active experts rather than getting starved by single-tier truncation.
    vram_ranked: list[tuple[str, int]] = []
    ram_ranked: list[tuple[str, int]] = []
    ssd_ranked: list[tuple[str, int]] = []

    for key, count in ranked:
        res, _ = _resolve_residency(key)
        if res == "VRAM":
            vram_ranked.append((key, count))
        elif res == "RAM":
            ram_ranked.append((key, count))
        else:
            ssd_ranked.append((key, count))

    # Compute total bytes loaded per tier across all experts
    vram_bytes_total = sum(_expert_bytes(k) for k, _ in vram_ranked)
    ram_bytes_total = sum(_expert_bytes(k) for k, _ in ram_ranked)
    ssd_bytes_total = sum(_expert_bytes(k) for k, _ in ssd_ranked)

    tier_sizes = {
        "vram": {
            "bytes": vram_bytes_total,
            "formatted": format_size_gb_mb(vram_bytes_total),
            "expert_count": len(vram_ranked),
        },
        "ram": {
            "bytes": ram_bytes_total,
            "formatted": format_size_gb_mb(ram_bytes_total),
            "expert_count": len(ram_ranked),
        },
        "ssd": {
            "bytes": ssd_bytes_total,
            "formatted": format_size_gb_mb(ssd_bytes_total),
            "expert_count": len(ssd_ranked),
        },
    }

    # Divide limit evenly across tiers (or return all if limit <= 0)
    if limit > 0:
        per_tier = max(1, limit // 3)
        selected_tier_items = vram_ranked[:per_tier] + ram_ranked[:per_tier] + ssd_ranked[:per_tier]
    else:
        selected_tier_items = vram_ranked + ram_ranked + ssd_ranked
    selected_tier_items.sort(key=lambda item: item[1], reverse=True)

    return {"experts": [_build_expert_entry(key, count) for key, count in selected_tier_items],
            "experts_seen": len(observed), "indexed_experts": len(cache.manifest), "total_invocations": total,
            "hot_invocation_percent": (hot_invocations * 100.0 / total) if total else 0.0,
            "cold_invocation_percent": ((total - hot_invocations) * 100.0 / total) if total else 0.0,
            "hot_definition": "invocations attributed to the most-used 20% of observed expert IDs",
            "active_experts": expert_handler.active_snapshot(), "pager": pager_metrics, "tier_sizes": tier_sizes}


def update_request_metrics(started: float, first_byte: float, prompt_tokens: int | None,
                           completion_tokens: int | None, output_chars: int,
                           prompt_seconds: float | None = None,
                           generation_seconds: float | None = None,
                           is_estimated: bool = False) -> None:
    elapsed = max(0.001, time.perf_counter() - started)
    ttft = max(0.001, first_byte - started) if first_byte else elapsed
    count = completion_tokens if completion_tokens is not None else max(1, output_chars // 4)
    p_tok = prompt_tokens if prompt_tokens is not None else 1
    p_sec = max(0.001, prompt_seconds if (prompt_seconds is not None and prompt_seconds > 0) else ttft)
    g_sec = max(0.001, generation_seconds if (generation_seconds is not None and generation_seconds > 0) else (elapsed - ttft))
    gen_tps = round(count / g_sec, 2)
    prefill_tps = round(p_tok / p_sec, 2) if p_tok and p_sec > 0 else None
    runtime_metrics.update({
        "generation_tokens_per_second": gen_tps,
        "prefill_tokens_per_second": prefill_tps,
        "tok_per_sec": {"prefill": prefill_tps, "generation": gen_tps},
        "generation_speed_is_estimated": bool(is_estimated or completion_tokens is None),
        "last_request_seconds": round(elapsed, 4),
        "last_request_formatted": format_duration_ms(elapsed * 1000.0),
        "last_time_to_first_token_seconds": round(ttft, 4),
        "last_time_to_first_token_formatted": format_duration_ms(ttft * 1000.0),
        "last_prompt_tokens": p_tok,
        "last_completion_tokens": count,
        "last_prompt_eval_seconds": round(p_sec, 4),
        "last_prompt_eval_formatted": format_duration_ms(p_sec * 1000.0),
        "last_generation_eval_seconds": round(g_sec, 4),
        "last_generation_eval_formatted": format_duration_ms(g_sec * 1000.0),
        "updated_at": time.time(),
    })


def build_chat_telemetry() -> dict[str, Any]:
    disk_metrics = pager.metrics()
    tier_io = build_tier_io(ssd_metrics=disk_metrics)
    read_ms = float(disk_metrics.get("time_used_for_reading_ms") or 0.0)
    read_bytes = int(disk_metrics.get("bytes_read") or 0)
    return {
        "tok_per_sec": dict(runtime_metrics.get("tok_per_sec") or {
            "prefill": runtime_metrics.get("prefill_tokens_per_second"),
            "generation": runtime_metrics.get("generation_tokens_per_second"),
        }),
        "prefill_tokens_per_second": runtime_metrics.get("prefill_tokens_per_second"),
        "generation_tokens_per_second": runtime_metrics.get("generation_tokens_per_second"),
        "prompt_tokens": runtime_metrics.get("last_prompt_tokens"),
        "completion_tokens": runtime_metrics.get("last_completion_tokens"),
        "prompt_eval_seconds": runtime_metrics.get("last_prompt_eval_seconds"),
        "prompt_eval_formatted": runtime_metrics.get("last_prompt_eval_formatted"),
        "generation_eval_seconds": runtime_metrics.get("last_generation_eval_seconds"),
        "generation_eval_formatted": runtime_metrics.get("last_generation_eval_formatted"),
        "ttft_seconds": runtime_metrics.get("last_time_to_first_token_seconds"),
        "ttft_formatted": runtime_metrics.get("last_time_to_first_token_formatted"),
        "elapsed_seconds": runtime_metrics.get("last_request_seconds"),
        "elapsed_formatted": runtime_metrics.get("last_request_formatted"),
        "ssd_read_time_ms": round(read_ms, 2),
        "ssd_read_time_seconds": round(read_ms / 1000.0, 4),
        "ssd_read_time_formatted": format_duration_ms(read_ms),
        "total_read_bytes": read_bytes,
        "total_read_size_formatted": format_size_gb_mb(read_bytes),
        "ssd_read_ops": int(disk_metrics.get("read_ops") or 0),
        "ssd_throughput_mbs": round(float(disk_metrics.get("actual_ssd_throughput_mbs") or 0.0), 2),
        "ssd_throughput_formatted": format_speed_mbs(disk_metrics.get("actual_ssd_throughput_mbs") or 0.0),
        "tier_io": tier_io,
        "mtp": runtime.mtp_status(),
    }


async def pull_from_hub(job_id: str, repo_id: str, files: list[str] | None) -> None:
    pull_jobs[job_id].update({"status": "running", "started_at": time.time()})
    try:
        from huggingface_hub import HfApi, hf_hub_download

        token = os.getenv("HF_TOKEN") or None
        api = HfApi(token=token)
        repo_files = await asyncio.to_thread(api.list_repo_files, repo_id=repo_id, repo_type="model")
        gguf_files = [name for name in repo_files if name.lower().endswith(".gguf")]
        selected = files if files else gguf_files
        if not selected:
            raise ValueError("repository contains no .gguf files")
        unknown = [name for name in selected if name not in gguf_files]
        if unknown:
            raise ValueError("requested files are not .gguf files in this repository: " + ", ".join(unknown))
        if len(selected) > 128:
            raise ValueError("select at most 128 GGUF files per pull job")
        repo_directory = MODELS_DIR / repo_id.replace("/", "--")
        repo_directory.mkdir(parents=True, exist_ok=True)
        pull_jobs[job_id].update({"files": selected, "total_files": len(selected), "completed_files": 0})
        paths = []
        for filename in selected:
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("unsafe repository filename")
            local_path = await asyncio.to_thread(
                hf_hub_download, repo_id=repo_id, filename=filename, repo_type="model",
                local_dir=str(repo_directory), token=token)
            paths.append(str(local_path))
            pull_jobs[job_id].update({"completed_files": len(paths), "current_file": filename,
                                      "downloaded_paths": paths})
        pull_jobs[job_id].update({"status": "complete", "completed_at": time.time(),
                                  "downloaded_paths": paths})
        # Force fresh inspection on the next inventory request.
        inventory._cache.clear()
    except asyncio.CancelledError:
        pull_jobs[job_id].update({"status": "cancelled", "completed_at": time.time()})
        raise
    except Exception as exc:
        pull_jobs[job_id].update({"status": "error", "error": str(exc), "completed_at": time.time()})


@app.get("/")
@app.get("/gui")
async def gui():
    return FileResponse(ROOT / "static" / "index.html", media_type="text/html")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "inference": runtime.status(), "cache": await cache.snapshot()}


@app.get("/api/models")
async def local_models():
    models = await asyncio.to_thread(inventory.list_models)
    return {"models": models, "models_dir": str(MODELS_DIR),
            "inference": runtime.status()}


@app.put("/api/models/directory")
async def set_models_directory(request: ModelsDirectoryRequest):
    global MODELS_DIR, inventory
    path = Path(request.path).expanduser().resolve()
    if path.exists() and not path.is_dir():
        raise HTTPException(status_code=400, detail="model directory path exists and is not a directory")
    try:
        path.mkdir(parents=True, exist_ok=True)
        MODELS_DIR = path
        inventory = ModelInventory(str(path))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"cannot use model directory: {exc}") from exc
    return {"models_dir": str(MODELS_DIR), "models": await asyncio.to_thread(inventory.list_models)}


@app.post("/api/models/load-check")
async def model_load_check(request: LoadCheckRequest):
    try:
        details = await asyncio.to_thread(inventory.get_details, request.model_id)
        plan = make_load_plan(details, min(cache.budget_bytes, safe_ram_budget()))
        return {**plan, "model_id": request.model_id}
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/load")
async def prepare_model(request: LoadCheckRequest):
    try:
        details = await asyncio.to_thread(inventory.get_details, request.model_id)
        await cache.prepare_for_model_load(details["path"])
        plan = make_load_plan(details, min(cache.budget_bytes, safe_ram_budget()))
        if plan["verdict"] == "cannot-confirm-fit":
            raise HTTPException(status_code=409, detail={"message": "model does not fit the detected safe memory plan",
                                                        "plan": plan})
        manifest = build_expert_manifest(details, request.model_id)
        if details["classification"] == "MoE" and not manifest:
            raise HTTPException(status_code=422, detail="GGUF expert tensor layout could not be mapped safely")
        expert_data_dir = await load_model_expert_data(details, manifest)
        prepared_models[request.model_id] = {"path": details["path"], "indexed_experts": len(manifest),
                                             "strategy": plan["strategy"], "state": "prepared"}
        staged = await expert_handler.warm_boot(set(manifest))
        await persist_model_expert_data(model_profile_key(details))
        return {"state": "prepared", "model_id": request.model_id, "indexed_experts": len(manifest),
                "indexed_expert_ids_sample": list(manifest)[:8],
                "warm_staged_experts": staged, "plan": plan,
                "expert_data_dir": str(expert_data_dir),
                "note": "Expert ranges are indexed and historical hot blocks may be staged in RAM. llama.cpp executes inference in-process; its expert routes are not yet connected to this cache."}
    except HTTPException:
        raise
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/unload")
async def unload_model(request: UnloadRequest):
    try:
        details = await asyncio.to_thread(inventory.get_details, request.model_id)
        if details.get("classification") == "MoE":
            await load_model_expert_data(details, build_expert_manifest(details, request.model_id))
            await persist_model_expert_data(model_profile_key(details))
        result = await cache.unload_file(details["path"], request.timeout_seconds,
                                         drop_page_cache=request.drop_page_cache)
        if result["unloaded"]:
            await asyncio.to_thread(inventory.forget, request.model_id)
            await cache.save_telemetry()
            prepared_models.pop(request.model_id, None)
            result["inspector_index_forgotten"] = True
            result["ram_cache_bytes_remaining"] = 0
        result["freed_bytes_formatted"] = format_size_gb_mb(result.get("freed_bytes") or 0)
        result["freed_vram_formatted"] = format_size_gb_mb(result.get("freed_vram_bytes") or 0)
        result["system_ram_reclaimed_formatted"] = format_size_gb_mb(result.get("system_ram_reclaimed_bytes") or 0)
        return result
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/cache/drop")
async def drop_all_caches(request: CacheDropRequest | None = None):
    req = request or CacheDropRequest()
    runtime_result = None
    if req.unload_runtime:
        runtime_result = await asyncio.to_thread(runtime.unload, req.drop_page_cache)
        runtime_result["ram_reclaimed_formatted"] = format_size_gb_mb(runtime_result.get("ram_reclaimed_bytes") or 0)
    cache_result = await cache.drop_all(drop_page_cache=req.drop_page_cache)
    prepared_models.clear()
    await cache.save_telemetry()
    cache_result["freed_ram_cache_formatted"] = format_size_gb_mb(cache_result.get("freed_ram_cache_bytes") or 0)
    cache_result["freed_vram_formatted"] = format_size_gb_mb(cache_result.get("freed_vram_bytes") or 0)
    cache_result["system_ram_reclaimed_formatted"] = format_size_gb_mb(cache_result.get("system_ram_reclaimed_bytes") or 0)
    if runtime_result is not None:
        cache_result["runtime_unload"] = runtime_result
    return cache_result


@app.get("/api/runtime/kv-cache")
async def get_kv_cache():
    st = runtime.status()
    return {"kv_cache_type_k": st.get("kv_cache_type_k"), "kv_cache_type_v": st.get("kv_cache_type_v")}


@app.put("/api/runtime/kv-cache")
async def set_kv_cache(request: KvCacheConfigRequest):
    k = request.type_kv or request.type_k
    v = request.type_kv or request.type_v
    return await asyncio.to_thread(runtime.configure_kv_cache, type_k=k, type_v=v)


def _read_gui_settings() -> dict:
    try:
        if SETTINGS_FILE.is_file():
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_gui_settings(data: dict) -> None:
    try:
        current = _read_gui_settings()
        current.update(data)
        _write_json_atomic(SETTINGS_FILE, current)
    except Exception:
        pass


@app.get("/api/settings")
async def get_settings():
    return _read_gui_settings()


@app.put("/api/settings")
async def save_settings(request: Request):
    payload = await request.json()
    if isinstance(payload, dict):
        _save_gui_settings(payload)
    return _read_gui_settings()


@app.get("/api/runtime/max-ram")
async def get_max_ram():
    from moe_boost.cache import get_configured_max_ram_bytes
    max_bytes = get_configured_max_ram_bytes()
    return {
        "max_ram_raw": os.getenv("MOE_MAX_RAM") or "",
        "max_ram_bytes": max_bytes,
        "max_ram_formatted": format_size_gb_mb(max_bytes) if max_bytes is not None else None,
        "active_budget_bytes": min(cache.budget_bytes, safe_ram_budget()),
        "active_budget_formatted": format_size_gb_mb(min(cache.budget_bytes, safe_ram_budget())),
    }


@app.put("/api/runtime/max-ram")
async def set_max_ram(request: MaxRamConfigRequest):
    from moe_boost.cache import set_configured_max_ram
    parsed_bytes = set_configured_max_ram(request.max_ram)
    # Rebalance cache immediately to enforce new ceiling
    rebalance_res = await cache.rebalance()
    active_b = min(cache.budget_bytes, safe_ram_budget())
    return {
        "max_ram_raw": os.getenv("MOE_MAX_RAM") or "",
        "max_ram_bytes": parsed_bytes,
        "max_ram_formatted": format_size_gb_mb(parsed_bytes) if parsed_bytes is not None else None,
        "active_budget_bytes": active_b,
        "active_budget_formatted": format_size_gb_mb(active_b),
        "rebalance": rebalance_res,
    }


@app.get("/api/mtp")
async def get_mtp():
    return runtime.mtp_status()


@app.get("/api/queue")
async def get_queue_status():
    """Get status of the VRAM execution queue and RAM overflow spillway."""
    if hasattr(cache, "queue_engine") and cache.queue_engine is not None:
        return cache.queue_engine.snapshot()
    return {"status": "unavailable"}


class QueueDispatchRequest(BaseModel):
    expert_id: str
    tokens: list[float] | None = None
    sync: bool = True
    timeout: float = 5.0
    include_telemetry: bool = False


class QueueBatchDispatchRequest(BaseModel):
    expert_ids: list[str]
    sync: bool = True
    timeout: float = 10.0
    include_telemetry: bool = False


@app.post("/api/queue/dispatch-batch")
async def dispatch_queue_batch(req: QueueBatchDispatchRequest):
    """High-throughput batch dispatch across VRAM Tool Workstation and RAM Hand Workstation."""
    if not hasattr(cache, "queue_engine") or cache.queue_engine is None:
        raise HTTPException(status_code=503, detail="Queue engine not initialized")
    default_tokens = np.ones((1, 64), dtype=np.float32)
    batch = [(eid, default_tokens) for eid in req.expert_ids]
    jobs = await asyncio.to_thread(
        cache.queue_engine.dispatch_batch,
        batch=batch,
        timeout=req.timeout,
    )
    res = {
        "count": len(jobs),
        "results": [
            {
                "expert_id": j.expert_id,
                "executed_on": j.executed_on.value if j.executed_on else None,
                "latency_ms": round((j.completion_time - j.dispatch_time) * 1000.0, 3) if j.completion_time else None,
            }
            for j in jobs
        ],
    }
    if req.include_telemetry:
        res["telemetry"] = cache.queue_engine.snapshot()
    return res


@app.post("/api/queue/dispatch")
async def dispatch_queue_expert(req: QueueDispatchRequest):
    """Dispatch an expert evaluation to VRAM queue or direct RAM overflow spillway."""
    if not hasattr(cache, "queue_engine") or cache.queue_engine is None:
        raise HTTPException(status_code=503, detail="Queue engine not initialized")
    tokens_arr = np.array(req.tokens, dtype=np.float32) if req.tokens else np.ones((1, 64), dtype=np.float32)
    job = await asyncio.to_thread(
        cache.queue_engine.dispatch,
        expert_id=req.expert_id,
        tokens=tokens_arr,
        sync=req.sync,
        timeout=req.timeout,
    )
    res = {
        "expert_id": job.expert_id,
        "executed_on": job.executed_on.value if job.executed_on else None,
        "dispatch_time": job.dispatch_time,
        "completion_time": job.completion_time,
        "latency_ms": round((job.completion_time - job.dispatch_time) * 1000.0, 3) if job.completion_time else None,
        "result_shape": list(job.result.shape) if job.result is not None else None,
    }
    if req.include_telemetry:
        res["telemetry"] = cache.queue_engine.snapshot()
    return res


@app.put("/api/mtp")
async def set_mtp(request: MtpConfigRequest):
    return await asyncio.to_thread(
        runtime.configure_mtp,
        enabled=request.enabled,
        mode=request.mode,
        num_tokens=request.num_tokens,
        ngram_n=request.ngram_n,
    )


@app.post("/api/experts/activate")
async def activate_experts(request: ExpertActivationRequest):
    routes = request.token_routes or ([request.expert_ids] if request.expert_ids else [])
    batches = expert_handler.batch_tokens(routes)
    expert_ids = list(batches)
    if not expert_ids:
        raise HTTPException(status_code=400, detail="provide expert_ids or token_routes")
    try:
        route_sequence = [str(expert_id) for row in routes for expert_id in row]
        async with expert_handler.acquire_for_compute(request.session_id, expert_ids, route_sequence) as blocks:
            response = {"active_experts": list(blocks),
                        "expert_bytes": {key: int(getattr(value, "nbytes", getattr(value, "size_bytes", 0)))
                                         for key, value in blocks.items()},
                        "residency": {key: ("SSD-paged" if hasattr(value, "iter_chunks") else
                                            ("VRAM" if cache.vram_cache.contains(key) else "RAM"))
                                      for key, value in blocks.items()},
                        "token_batches": batches}
        profile_ids = {key.split(".", 1)[0] for key in expert_ids if "." in key}
        for profile_id in profile_ids:
            await persist_model_expert_data(profile_id)
        await cache.save_telemetry()
        return response
    except (ValueError, FileNotFoundError, MemoryError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/models/{model_id:path}")
async def model_details(model_id: str):
    try:
        return await asyncio.to_thread(inventory.get_details, model_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/pull")
async def create_pull(request: PullRequest):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", request.repo_id):
        raise HTTPException(status_code=400, detail="repo_id must use the Hugging Face namespace/repository form")
    if request.files and any(not isinstance(name, str) or not name.lower().endswith(".gguf") for name in request.files):
        raise HTTPException(status_code=400, detail="only .gguf filenames can be selected")
    job_id = uuid.uuid4().hex
    pull_jobs[job_id] = {"job_id": job_id, "repo_id": request.repo_id, "status": "queued",
                         "created_at": time.time(), "files": request.files or []}
    task = asyncio.create_task(pull_from_hub(job_id, request.repo_id, request.files))
    pull_tasks.add(task)
    task.add_done_callback(pull_tasks.discard)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/models/pulls/{job_id}")
async def get_pull(job_id: str):
    if job_id not in pull_jobs:
        raise HTTPException(status_code=404, detail="unknown pull job")
    return pull_jobs[job_id]


@app.get("/api/accelerator")
async def get_accelerator():
    models = await asyncio.to_thread(inventory.list_models, False)
    layers = max((model.get("block_count") or 0 for model in models), default=0)
    layer_sizes = None
    if models:
        try:
            details = await asyncio.to_thread(inventory.get_details, models[0]["id"])
            layer_sizes = layer_size_map(details)
        except (KeyError, ValueError, OSError):
            layer_sizes = None
    return await asyncio.to_thread(resolve_config, accelerator_settings, layers or None, layer_sizes)


@app.put("/api/accelerator")
async def set_accelerator(request: AcceleratorRequest):
    global accelerator_settings
    try:
        accelerator_settings = AcceleratorConfig(**request.model_dump())
        return resolve_config(accelerator_settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/telemetry")
async def telemetry(limit: int = 1050):
    memory = await asyncio.to_thread(memory_snapshot)
    cache_snapshot = await cache.snapshot()
    heat = heat_summary(limit=limit)
    total_ram = memory["ram"].get("total_bytes")
    if total_ram:
        # The cache ceiling represents the policy target; actual system RAM is reported separately.
        try:
            reserve_percent = min(60, max(5, int(os.getenv("MOE_RAM_RESERVE_PERCENT", "35"))))
        except ValueError:
            reserve_percent = 10
        memory["ram"]["reserve_percent"] = reserve_percent
        memory["ram"]["reserve_target_bytes"] = int(total_ram * reserve_percent / 100)
        from moe_boost.cache import get_configured_max_ram_bytes
        max_ram_configured = get_configured_max_ram_bytes()
        moe_ceiling = min(int(total_ram * (100 - reserve_percent) / 100), cache_snapshot["ram_budget_bytes"])
        if max_ram_configured is not None:
            moe_ceiling = min(moe_ceiling, max_ram_configured)
        memory["ram"]["moe_allocatable_ceiling_bytes"] = moe_ceiling
        if max_ram_configured is not None:
            memory["ram"]["max_ram_bytes"] = max_ram_configured
            memory["ram"]["max_ram_formatted"] = format_size_gb_mb(max_ram_configured)
        memory["ram"]["total_formatted"] = format_size_gb_mb(total_ram)
        memory["ram"]["used_formatted"] = format_size_gb_mb(memory["ram"].get("used_bytes"))
        memory["ram"]["available_formatted"] = format_size_gb_mb(memory["ram"].get("available_bytes"))
        memory["ram"]["allocatable_ceiling_formatted"] = format_size_gb_mb(memory["ram"].get("allocatable_ceiling_bytes"))
        memory["ram"]["reserve_target_formatted"] = format_size_gb_mb(memory["ram"].get("reserve_target_bytes"))
        memory["ram"]["moe_allocatable_ceiling_formatted"] = format_size_gb_mb(memory["ram"].get("moe_allocatable_ceiling_bytes"))
    cache_snapshot["resident_formatted"] = format_size_gb_mb(cache_snapshot.get("resident_bytes"))
    cache_snapshot["ram_budget_formatted"] = format_size_gb_mb(cache_snapshot.get("ram_budget_bytes"))
    read_ms = float(heat["pager"].get("time_used_for_reading_ms") or 0.0)
    read_bytes = int(heat["pager"].get("bytes_read") or 0)
    tier_io = build_tier_io(vram_snap=cache_snapshot.get("vram"), ssd_metrics=heat["pager"])
    cache_snapshot["tier_io"] = tier_io
    disk_info = {
        "time_used_for_reading_ms": read_ms,
        "ssd_read_time_ms": round(read_ms, 2),
        "ssd_read_time_seconds": round(read_ms / 1000.0, 4),
        "ssd_read_time_formatted": format_duration_ms(read_ms),
        "bytes_read": read_bytes,
        "total_read_bytes": read_bytes,
        "total_read_size_formatted": format_size_gb_mb(read_bytes),
        "read_ops": int(heat["pager"].get("read_ops") or 0),
        "actual_ssd_throughput_mbs": float(heat["pager"].get("actual_ssd_throughput_mbs") or 0.0),
        "actual_ssd_throughput_formatted": format_speed_mbs(heat["pager"].get("actual_ssd_throughput_mbs") or 0.0),
    }
    return {"timestamp": time.time(), "performance": dict(runtime_metrics), "runtime": runtime.status(),
            "mtp": runtime.mtp_status(), "memory": memory,
            "disk": disk_info, "tier_io": tier_io,
            "tier_optimizer": last_tier_optimization,
            "expert_heat": heat, "cache": cache_snapshot,
            "prepared_models": dict(prepared_models),
            "available_models": await asyncio.to_thread(inventory.list_models)}


@app.get("/v1/models")
async def models():
    local = await asyncio.to_thread(inventory.list_models, False)
    return {"object": "list", "data": [{"id": model["id"], "object": "model"} for model in local]}


@app.get("/api/runtime")
async def runtime_status():
    return runtime.status()


@app.post("/api/runtime/load")
async def runtime_load(request: RuntimeLoadRequest):
    try:
        details = await asyncio.to_thread(inventory.get_details, request.model_id)
        await cache.prepare_for_model_load(details["path"])
        if any(v is not None for v in (request.mtp_enabled, request.mtp_mode, request.mtp_num_tokens, request.mtp_ngram_n)):
            await asyncio.to_thread(
                runtime.configure_mtp,
                enabled=request.mtp_enabled,
                mode=request.mtp_mode,
                num_tokens=request.mtp_num_tokens,
                ngram_n=request.mtp_ngram_n,
            )
        if request.kv_cache_type_k is not None or request.kv_cache_type_v is not None:
            await asyncio.to_thread(
                runtime.configure_kv_cache,
                type_k=request.kv_cache_type_k,
                type_v=request.kv_cache_type_v,
            )
        expert_data_dir = None
        manifest = None
        need_warm_boot = False
        if details.get("classification") == "MoE":
            manifest = build_expert_manifest(details, request.model_id)
            profile_id = model_profile_key(details)
            model_path_resolved = str(Path(details["path"]).resolve())
            was_loaded = loaded_expert_profiles.get(profile_id) == model_path_resolved
            expert_data_dir = await load_model_expert_data(details, manifest)
            need_warm_boot = not was_loaded or cache.resident_bytes == 0
        plan = make_load_plan(details, min(cache.budget_bytes, safe_ram_budget()))
        if plan["verdict"] == "cannot-confirm-fit":
            raise HTTPException(status_code=409, detail={
                "message": "Native runtime could not confirm a safe memory fit",
                "plan": plan,
            })
        model_path = inventory.directory / request.model_id
        # When moe_staging_mode is True, set gpu_layers=0 in llama.cpp to free up ~2.5 GB VRAM
        # specifically for our hot expert cache (tiered 30% VRAM / 70% RAM architecture).
        gpu_layers_override = request.gpu_layers
        if request.moe_staging_mode:
            gpu_layers_override = 0
        accelerator_plan = await asyncio.to_thread(runtime_accelerator_plan, details, gpu_layers_override)
        if accelerator_plan.get("gpu_layers", 0) > 0 and runtime.status().get("model_path") != str(model_path.resolve()):
            await asyncio.to_thread(cache.vram_cache.clear, True)
        result = await asyncio.to_thread(runtime.load, model_path, accelerator_plan["gpu_layers"],
                                         accelerator_plan["backend"],
                                         (accelerator_plan.get("selected_device") or {}).get("vulkan_index")
                                         or accelerator_plan.get("fallback_vulkan_index"))
        if need_warm_boot and manifest:
            await expert_handler.warm_boot(set(manifest))
        # Persist loaded configuration so user does not need to retype settings next time
        saved_data = {
            "last_model_id": request.model_id,
            "kv_cache_type": request.kv_cache_type_k or request.kv_cache_type_v,
            "models": {
                request.model_id: {
                    "kv_cache_type": request.kv_cache_type_k or request.kv_cache_type_v,
                    "mtp_mode": request.mtp_mode,
                    "mtp_tokens": request.mtp_num_tokens,
                    "mtp_ngram": request.mtp_ngram_n,
                    "max_ram": request.max_ram,
                    "moe_staging_mode": request.moe_staging_mode,
                    "gpu_layers": request.gpu_layers,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "system_prompt": request.system_prompt,
                }
            }
        }
        if request.moe_staging_mode is not None:
            saved_data["moe_staging_mode"] = request.moe_staging_mode
        if request.max_ram is not None:
            saved_data["max_ram"] = str(request.max_ram).strip()
        if request.temperature is not None:
            saved_data["temperature"] = request.temperature
        if request.max_tokens is not None:
            saved_data["max_tokens"] = request.max_tokens
        if request.system_prompt is not None:
            saved_data["system_prompt"] = request.system_prompt
        if request.mtp_mode is not None:
            saved_data["mtp_mode"] = request.mtp_mode
        if request.mtp_num_tokens is not None:
            saved_data["mtp_tokens"] = request.mtp_num_tokens
        if request.mtp_ngram_n is not None:
            saved_data["mtp_ngram"] = request.mtp_ngram_n
        _save_gui_settings(saved_data)
        return {**result, "load_plan": plan, "accelerator_plan": accelerator_plan,
                "expert_data_dir": str(expert_data_dir) if expert_data_dir else None}
    except HTTPException:
        raise
    except NativeRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runtime/optimize")
async def runtime_optimize(request: RuntimeOptimizeRequest):
    """Start a bounded, memory-aware real-inference benchmark in the background."""
    job_id = uuid.uuid4().hex[:12]
    optimization_jobs[job_id] = {"job_id": job_id, "status": "queued", "stage": "queued",
                                 "progress": 0, "total": 0, "results": [], "error": None}

    async def run_job():
        job = optimization_jobs[job_id]
        try:
            job.update(status="running", stage="inspecting_model")
            details = await asyncio.to_thread(inventory.get_details, request.model_id)
            await cache.prepare_for_model_load(details["path"])
            plan = make_load_plan(details, min(cache.budget_bytes, safe_ram_budget()))
            if plan["verdict"] == "cannot-confirm-fit":
                raise HTTPException(status_code=409, detail={"message": "Memory planner cannot confirm a safe fit", "plan": plan})
            accelerator_plan = await asyncio.to_thread(runtime_accelerator_plan, details)
            metadata = details.get("metadata") or {}
            trained_context = next((int(value) for key, value in metadata.items()
                                    if str(key).endswith("context_length") and isinstance(value, int)
                                    and value >= 512), request.max_context)
            benchmark_context = min(request.max_context, trained_context, 4096)
            memory = memory_snapshot().get("ram") or {}
            reserve_bytes = int(memory.get("reserve_target_bytes") or 0)
            reserve_percent = int(memory.get("reserve_percent") or 10)

            def update_progress(event: dict):
                stage = event.get("stage", "benchmarking")
                job["stage"] = stage
                job["total"] = int(event.get("total") or job.get("total") or 0)
                if event.get("candidate"):
                    job["progress"] = int(event["candidate"])
                if event.get("result"):
                    job.setdefault("results", []).append(event["result"])

            job["stage"] = "benchmarking_candidates"
            result = await asyncio.to_thread(
                runtime.optimize, inventory.directory / request.model_id,
                max_safe_gpu_layers=accelerator_plan["gpu_layers"],
                backend=accelerator_plan["backend"],
                vulkan_device_index=((accelerator_plan.get("selected_device") or {}).get("vulkan_index")
                                     or accelerator_plan.get("fallback_vulkan_index")),
                reserve_bytes=reserve_bytes, reserve_percent=reserve_percent,
                max_context=benchmark_context, progress_callback=update_progress)
            job.update(status="completed", stage="completed", progress=job.get("total", 0),
                       result={**result, "load_plan": plan, "accelerator_plan": accelerator_plan,
                               "benchmark_context_limit": benchmark_context},
                       results=result.get("results", []))
        except Exception as exc:
            detail = getattr(exc, "detail", None)
            job.update(status="failed", stage="failed", error=detail or str(exc))

    asyncio.create_task(run_job())
    return {"job_id": job_id, "status": "queued", "message": "Memory-aware benchmark queued"}


@app.get("/api/runtime/optimize/{job_id}")
async def runtime_optimize_status(job_id: str):
    job = optimization_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Optimization job not found")
    return job


@app.get("/api/tiers/optimize")
async def get_tier_optimization():
    if last_tier_optimization is not None:
        return last_tier_optimization
    return {
        "status": "not_run",
        "tier_io": build_tier_io(),
        "optimizations_applied": [],
        "active_settings": {},
    }


@app.post("/api/tiers/optimize")
async def post_tier_optimization(request: TierOptimizeRequest | None = None):
    global last_tier_optimization
    req = request or TierOptimizeRequest()
    model_paths: list[str] = []
    for entry in prepared_models.values():
        p = entry.get("path")
        if p:
            model_paths.append(str(p))
    rt_path = runtime.status().get("model_path")
    if rt_path:
        model_paths.append(str(rt_path))
    if req.model_id:
        try:
            details = await asyncio.to_thread(inventory.get_details, req.model_id)
            model_paths.append(str(details["path"]))
            if details.get("classification") == "MoE":
                manifest = build_expert_manifest(details, req.model_id)
                if manifest:
                    await load_model_expert_data(details, manifest)
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = await optimize_storage_tiers(
        cache,
        pager,
        model_paths=model_paths,
        max_prefetch_experts=req.max_prefetch_experts,
        max_prefetch_bytes=req.max_prefetch_bytes,
        max_vram_experts=req.max_vram_experts,
    )
    last_tier_optimization = result
    return result


@app.post("/api/runtime/unload")
async def runtime_unload():
    result = await asyncio.to_thread(runtime.unload, True)
    result["ram_reclaimed_formatted"] = format_size_gb_mb(result.get("ram_reclaimed_bytes") or 0)
    return result


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="request body must be valid JSON") from exc
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise HTTPException(status_code=400, detail="messages must be a JSON array")
    model_id = body.get("model")
    if not isinstance(model_id, str) or not model_id:
        # Fallback to the first available local model or currently active model
        available_models = inventory.list_models(include_errors=False)
        if available_models:
            model_id = available_models[0]["id"]
        elif os.environ.get("MOE_MODEL_FILE"):
            model_id = os.environ.get("MOE_MODEL_FILE")
        else:
            raise HTTPException(status_code=400, detail="model parameter is required (no models found in inventory)")
    try:
        if "mtp" in body and isinstance(body["mtp"], dict):
            mtp_cfg = body["mtp"]
            await asyncio.to_thread(
                runtime.configure_mtp,
                enabled=mtp_cfg.get("enabled"),
                mode=mtp_cfg.get("mode"),
                num_tokens=mtp_cfg.get("num_tokens"),
                ngram_n=mtp_cfg.get("ngram_n"),
            )
        details = await asyncio.to_thread(inventory.get_details, model_id)
        await cache.prepare_for_model_load(details["path"])
        manifest = None
        need_warm_boot = False
        if details.get("classification") == "MoE":
            manifest = build_expert_manifest(details, model_id)
            profile_id = model_profile_key(details)
            model_path_resolved = str(Path(details["path"]).resolve())
            was_loaded = loaded_expert_profiles.get(profile_id) == model_path_resolved
            await load_model_expert_data(details, manifest)
            need_warm_boot = not was_loaded or cache.resident_bytes == 0
        plan = make_load_plan(details, min(cache.budget_bytes, safe_ram_budget()))
        if plan["verdict"] == "cannot-confirm-fit":
            raise HTTPException(status_code=409, detail={
                "message": "Native runtime could not confirm a safe memory fit",
                "plan": plan,
            })
        target_model_path = Path(details["path"]).resolve()
        saved_m = _read_gui_settings().get(model_id) or _read_gui_settings().get("models", {}).get(model_id) or {}
        gpu_layers_override = saved_m.get("gpu_layers")
        # If user explicitly configured MOE_GPU_LAYERS in env (e.g. -1 for auto, or 4), let it take precedence over stale gui_settings
        if "MOE_GPU_LAYERS" in os.environ and os.environ["MOE_GPU_LAYERS"] != "":
            try:
                gpu_layers_override = int(os.environ["MOE_GPU_LAYERS"])
            except ValueError:
                gpu_layers_override = None
        elif saved_m.get("moe_staging_mode"):
            gpu_layers_override = 0
        accelerator_plan = await asyncio.to_thread(runtime_accelerator_plan, details, gpu_layers_override)
        if accelerator_plan.get("gpu_layers", 0) > 0 and runtime.status().get("model_path") != str(target_model_path.resolve()):
            await asyncio.to_thread(cache.vram_cache.clear, True)
        await asyncio.to_thread(runtime.load, target_model_path,
                                 accelerator_plan["gpu_layers"], accelerator_plan["backend"],
                                 (accelerator_plan.get("selected_device") or {}).get("vulkan_index")
                                 or accelerator_plan.get("fallback_vulkan_index"))
        if need_warm_boot and manifest:
            await expert_handler.warm_boot(set(manifest))
    except HTTPException:
        raise
    except NativeRuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=404, detail=f"unknown local GGUF model: {model_id}") from exc

    try:
        temperature = min(2.0, max(0.0, float(body.get("temperature", 0.7))))
        max_tokens = min(32768, max(1, int(body.get("max_tokens", 512))))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="temperature and max_tokens must be numeric") from exc
    messages = body["messages"]
    if not all(isinstance(message, dict) and isinstance(message.get("content"), (str, list))
               for message in messages):
        raise HTTPException(status_code=400, detail="each message needs a string or content-array")

    if body.get("stream", False):
        async def generate():
            events: queue.Queue = queue.Queue()
            sentinel = object()
            cancel_event = threading.Event()
            started = time.perf_counter()

            def produce() -> None:
                try:
                    for item in runtime.stream_completion(messages, temperature=temperature,
                                                          max_tokens=max_tokens,
                                                          cancel_event=cancel_event):
                        events.put(("chunk", item))
                except Exception as exc:
                    events.put(("error", str(exc)))
                finally:
                    events.put(("done", sentinel))

            producer = asyncio.create_task(asyncio.to_thread(produce))
            first_token_at = None
            output_chars = 0
            prompt_tokens = completion_tokens = None
            prompt_ms = predicted_ms = None
            try:
                while True:
                    kind, value = await asyncio.to_thread(events.get)
                    if kind == "chunk":
                        choices = value.get("choices") or []
                        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if first_token_at is None:
                                first_token_at = time.perf_counter()
                            output_chars += len(content)
                        usage = value.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                            completion_tokens = usage.get("completion_tokens", completion_tokens)
                        timings = value.get("timings")
                        if isinstance(timings, dict):
                            prompt_ms = timings.get("prompt_ms", prompt_ms)
                            predicted_ms = timings.get("predicted_ms", predicted_ms)
                        if choices:
                            yield f"data: {json.dumps(value, separators=(',', ':'))}\n\n".encode()
                    elif kind == "error":
                        payload = {"error": {"message": value, "type": "inference_error"}}
                        yield f"data: {json.dumps(payload)}\n\n".encode()
                    else:
                        break
                await producer
                finished = time.perf_counter()
                first = first_token_at or finished
                p_sec = (float(prompt_ms) / 1000.0) if (prompt_ms and float(prompt_ms) > 0) else max(0.001, first - started)
                g_sec = (float(predicted_ms) / 1000.0) if (predicted_ms and float(predicted_ms) > 0) else max(0.001, finished - first)
                update_request_metrics(started, first, prompt_tokens, completion_tokens,
                                       output_chars, prompt_seconds=p_sec,
                                       generation_seconds=g_sec)
                summary_chunk = {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": runtime_metrics["last_prompt_tokens"],
                        "completion_tokens": runtime_metrics["last_completion_tokens"],
                        "total_tokens": (runtime_metrics["last_prompt_tokens"] or 0) + (runtime_metrics["last_completion_tokens"] or 0),
                    },
                    "moe_telemetry": build_chat_telemetry(),
                }
                yield f"data: {json.dumps(summary_chunk, separators=(',', ':'))}\n\n".encode()
                yield b"data: [DONE]\n\n"
            finally:
                cancel_event.set()
                await asyncio.shield(producer)
        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    try:
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        started = time.perf_counter()
        result = await asyncio.to_thread(runtime.completion, messages, temperature=temperature,
                                         max_tokens=max_tokens, stream=False,
                                         tools=tools, tool_choice=tool_choice)
        finished = time.perf_counter()
        usage = result.get("usage") if isinstance(result, dict) else {}
        usage = usage if isinstance(usage, dict) else {}
        timings = result.get("timings") if isinstance(result, dict) else {}
        timings = timings if isinstance(timings, dict) else {}
        choices = result.get("choices") if isinstance(result, dict) else []
        first_msg = (choices[0].get("message") or {}) if choices and isinstance(choices[0], dict) else {}
        content_text = first_msg.get("content") or ""
        p_tok = usage.get("prompt_tokens")
        c_tok = usage.get("completion_tokens")
        elapsed = max(0.001, finished - started)
        p_sec = (float(timings["prompt_ms"]) / 1000.0) if timings.get("prompt_ms") else max(
            0.001, elapsed * ((p_tok or 1) / max(1, (p_tok or 1) + (c_tok or 1) * 4)))
        g_sec = (float(timings["predicted_ms"]) / 1000.0) if timings.get("predicted_ms") else max(
            0.001, elapsed - p_sec)
        update_request_metrics(
            started, started + p_sec,
            p_tok,
            c_tok,
            len(content_text),
            prompt_seconds=p_sec,
            generation_seconds=g_sec,
        )
        if isinstance(result, dict):
            result["moe_telemetry"] = build_chat_telemetry()
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": {"message": str(exc), "type": "inference_error"}}, status_code=500)


@app.get("/v1/moe/cache")
async def cache_status():
    return await cache.snapshot()


@app.post("/v1/moe/experts/{expert_id}/acquire")
async def acquire_expert(expert_id: str):
    try:
        async with cache.acquire(expert_id) as block:
            return {"expert_id": expert_id, "loaded_bytes": len(block)}
    except (ValueError, FileNotFoundError, MemoryError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/moe/telemetry/save")
async def save_telemetry():
    await cache.save_telemetry()
    return {"saved": True}


@app.post("/v1/moe/telemetry/import")
async def import_telemetry(request: Request):
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            raise ValueError("benchmark profile must be a JSON object")
        await cache.import_benchmark(payload)
        raw = payload.get("hits", payload.get("expert_hits", {}))
        return {"imported": True, "experts": len(raw)}
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
