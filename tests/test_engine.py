import json
import asyncio
import struct
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

from engine.inspector import inspect_gguf
from engine.learner import ActiveExpertHandler, PagedExpert
from engine.manifest import build_expert_manifest
from moe_boost.cache import ExpertCache
from moe_boost.pager import Pager


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def make_gguf(path: Path) -> None:
    metadata = bytearray()
    def kv_string(key: str, value: str):
        metadata.extend(gguf_string(key) + struct.pack("<I", 8) + gguf_string(value))
    def kv_u32(key: str, value: int):
        metadata.extend(gguf_string(key) + struct.pack("<II", 4, value))
    kv_string("general.architecture", "llama")
    kv_u32("llama.block_count", 2)
    kv_u32("llama.expert_count", 4)
    kv_u32("llama.expert_used_count", 2)
    kv_u32("general.alignment", 32)
    tensors = bytearray()
    tensor_specs = [
        ("blk.0.ffn_gate_exps.weight", [32, 8, 4], 1, 0),
        ("blk.1.ffn_down_exps.weight", [8, 32, 4], 2, 2048),
    ]
    for name, dims, tensor_type, offset in tensor_specs:
        tensors.extend(gguf_string(name))
        tensors.extend(struct.pack("<I", len(dims)))
        tensors.extend(struct.pack("<" + "Q" * len(dims), *dims))
        tensors.extend(struct.pack("<IQ", tensor_type, offset))
    header = struct.pack("<4sIQQ", b"GGUF", 3, len(tensor_specs), 5)
    data_offset = (len(header) + len(metadata) + len(tensors) + 31) & ~31
    payload = header + metadata + tensors
    payload += b"\0" * (data_offset - len(payload))
    payload += b"\0" * (2048 + 576)
    path.write_bytes(payload)


class GGUFInspectorTests(unittest.TestCase):
    def test_architecture_experts_tensor_offsets_and_sizes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "synthetic.gguf"
            make_gguf(path)
            result = inspect_gguf(path)
            self.assertEqual(result["architecture"], "llama")
            self.assertEqual(result["classification"], "MoE")
            self.assertEqual(result["expert_count"], 4)
            self.assertEqual(result["active_experts"], 2)
            first = result["layers"]["0"][0]
            self.assertEqual(first["byte_size"], 2048)
            self.assertTrue(first["size_exact"])
            self.assertEqual(first["absolute_offset"], result["data_offset"])

    def test_expert_tensor_axis_becomes_per_expert_byte_ranges(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "synthetic.gguf"
            make_gguf(path)
            details = inspect_gguf(path)
            manifest = build_expert_manifest(details, path.name)
            key = next(key for key in manifest if key.endswith("layer0.expert2"))
            entry = manifest[key]
            self.assertEqual(entry["size"], 512)
            self.assertEqual(entry["segments"][0]["offset"], details["data_offset"] + 1024)
            cache = ExpertCache(str(path.with_suffix(".manifest.json")),
                                str(path.with_suffix(".telemetry.json")), budget_bytes=2048)
            cache.manifest = manifest

            async def read_range():
                async with cache.acquire(key) as expert:
                    self.assertEqual(expert.size, 512)
                    self.assertTrue((expert == 0).all())
                result = await cache.unload_file(str(path), timeout_seconds=0)
                self.assertTrue(result["unloaded"])
            asyncio.run(read_range())

    def test_bad_magic_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.gguf"
            path.write_bytes(b"nope")
            with self.assertRaisesRegex(ValueError, "magic"):
                inspect_gguf(path)


class MemoryPlannerTests(unittest.TestCase):
    def test_reports_ram_reserve_and_projected_headroom(self):
        from engine.planner import make_load_plan
        fake_memory = {"backend": "cpu", "gpu": [], "ram": {"available_bytes": 8 * 1024**3}}
        with patch("engine.planner.memory_snapshot", return_value=fake_memory):
            plan = make_load_plan({"path": "tiny.gguf", "classification": "Dense",
                                   "file_size": 2 * 1024**3, "layers": {}},
                                  4 * 1024**3)
        self.assertEqual(plan["estimated_host_working_set_bytes"], 2 * 1024**3)
        self.assertEqual(plan["estimated_ram_available_after_bytes"], 6 * 1024**3)
        self.assertEqual(plan["estimated_vram_model_weights_bytes"], 0)
        self.assertIn("protected by the planner", plan["note"])


class PagerTests(unittest.TestCase):
    def test_native_pread_bounds_and_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.bin"
            path.write_bytes(b"0123456789")
            pager = Pager()
            # Native extension is optional; exercise the portable pread fallback when absent.
            handle = pager.open(str(path))
            try:
                self.assertEqual(pager.file_size(handle), 10)
                self.assertEqual(pager.pread(handle, 2, 4), b"2345")
                with self.assertRaises(Exception):
                    pager.pread(handle, 9, 2)
            finally:
                pager.close(handle)
            pager.record_expert("layer0.expert2")
            metrics = pager.metrics()
            self.assertEqual(metrics["bytes_read"], 4)
            self.assertGreaterEqual(metrics["time_used_for_reading_ms"], 0)
            self.assertEqual(metrics["expert_heat"]["layer0.expert2"], 1)


class NativeRuntimeTests(unittest.TestCase):
    def test_cuda_failure_tries_vulkan_before_cpu(self):
        from engine.runtime import LlamaCppRuntime

        attempts = []

        class FakeLlama:
            def __init__(self, **kwargs):
                attempts.append(kwargs["n_gpu_layers"])
                if len(attempts) == 1:
                    raise RuntimeError("simulated CUDA allocation failure")

            def close(self):
                pass

        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = FakeLlama
        fake_module.llama_supports_gpu_offload = lambda: True
        runtime = LlamaCppRuntime()
        runtime.fallback_backends = ["vulkan", "cpu"]
        runtime._settings["requested_gpu_layers"] = -1
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fixture.gguf"
            model.write_bytes(b"GGUF")
            with patch.dict("sys.modules", {"llama_cpp": fake_module}), \
                 patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0",
                                            "GGML_VK_VISIBLE_DEVICES": "0"}):
                status = runtime.load(model, max_safe_gpu_layers=2, backend="cuda",
                                      vulkan_device_index=1)
                self.assertEqual(attempts, [2, 2])
                self.assertEqual(status["backend"], "vulkan")
                self.assertIn("loaded with Vulkan", status["fallback_reason"])
                self.assertEqual(__import__("os").environ["CUDA_VISIBLE_DEVICES"], "0")
                self.assertEqual(__import__("os").environ["GGML_VK_VISIBLE_DEVICES"], "0")
                runtime.unload()

    def test_cuda_failure_retries_cpu(self):
        from engine.runtime import LlamaCppRuntime

        attempts = []

        class FakeLlama:
            def __init__(self, **kwargs):
                attempts.append(kwargs["n_gpu_layers"])
                if kwargs["n_gpu_layers"]:
                    raise RuntimeError("simulated CUDA allocation failure")

            def close(self):
                pass

        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = FakeLlama
        fake_module.llama_supports_gpu_offload = lambda: True
        runtime = LlamaCppRuntime()
        runtime._settings["requested_gpu_layers"] = -1
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fixture.gguf"
            model.write_bytes(b"GGUF")
            with patch.dict("sys.modules", {"llama_cpp": fake_module}):
                status = runtime.load(model, max_safe_gpu_layers=2, backend="cuda")
                self.assertEqual(attempts, [2, 0])
                self.assertEqual(status["backend"], "cpu")
                self.assertEqual(status["gpu_layers"], 0)
                self.assertIn("retrying on CPU", status["fallback_reason"])
                runtime.unload()

    def test_gpu_offload_respects_safe_layer_limit(self):
        from engine.runtime import LlamaCppRuntime

        attempts = []

        class FakeLlama:
            def __init__(self, **kwargs):
                attempts.append(kwargs["n_gpu_layers"])

            def close(self):
                pass

        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = FakeLlama
        fake_module.llama_supports_gpu_offload = lambda: True
        runtime = LlamaCppRuntime()
        runtime._settings["requested_gpu_layers"] = -1
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fixture.gguf"
            model.write_bytes(b"GGUF")
            with patch.dict("sys.modules", {"llama_cpp": fake_module}):
                status = runtime.load(model, max_safe_gpu_layers=3, backend="cuda")
                self.assertEqual(attempts, [3])
                self.assertEqual(status["backend"], "cuda")
                self.assertEqual(status["gpu_layers"], 3)
                runtime.unload()


class HuggingFacePullTests(unittest.TestCase):
    def test_selected_gguf_is_pulled_to_local_models_directory(self):
        import server
        from engine.inventory import ModelInventory
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir) / "models"
            server.MODELS_DIR = models_dir
            server.inventory = ModelInventory(str(models_dir))

            class FakeApi:
                def __init__(self, **_kwargs):
                    pass

                def list_repo_files(self, **_kwargs):
                    return ["weights/model.gguf", "README.md", "other.gguf"]

            def fake_download(**kwargs):
                result = models_dir / "org--model" / kwargs["filename"]
                result.parent.mkdir(parents=True, exist_ok=True)
                result.write_bytes(b"GGUF-test")
                return str(result)

            server.pull_jobs["unit-pull"] = {"job_id": "unit-pull", "status": "queued"}
            with patch("huggingface_hub.HfApi", FakeApi), patch(
                    "huggingface_hub.hf_hub_download", side_effect=fake_download):
                asyncio.run(server.pull_from_hub("unit-pull", "org/model", ["weights/model.gguf"]))
            self.assertEqual(server.pull_jobs["unit-pull"]["status"], "complete")
            self.assertTrue((models_dir / "org--model" / "weights/model.gguf").is_file())


class CacheUnloadTests(unittest.TestCase):
    def test_unload_waits_for_active_lease_then_frees_cached_ram(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.gguf"
            model.write_bytes(b"0123456789")
            manifest = root / "experts.json"
            manifest.write_text(json.dumps({"experts": {"layer0.expert0": {
                "path": str(model), "offset": 0, "size": 10}}}))
            cache = ExpertCache(str(manifest), str(root / "telemetry.json"), budget_bytes=1024)

            async def exercise():
                async with cache.acquire("layer0.expert0") as block:
                    self.assertEqual(block.tolist(), list(b"0123456789"))
                    blocked = await cache.unload_file(str(model), timeout_seconds=0)
                    self.assertFalse(blocked["unloaded"])
                result = await cache.unload_file(str(model), timeout_seconds=0)
                self.assertTrue(result["unloaded"])
                self.assertEqual(result["ram_cache_bytes_remaining"], 0)
                self.assertEqual((await cache.snapshot())["resident_bytes"], 0)
            asyncio.run(exercise())

    def test_active_set_over_ram_budget_streams_cold_blocks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "medium.gguf"
            model.write_bytes(bytes(range(8)))
            manifest = root / "experts.json"
            manifest.write_text(json.dumps({"experts": {
                "layer0.expert0": {"path": str(model), "offset": 0, "size": 4},
                "layer0.expert1": {"path": str(model), "offset": 4, "size": 4},
            }}))
            cache = ExpertCache(str(manifest), str(root / "telemetry.json"), budget_bytes=6)
            handler = ActiveExpertHandler(cache, prefetch_width=0)

            async def exercise():
                async with handler.acquire_for_compute("set-test", ["layer0.expert0", "layer0.expert1"]) as blocks:
                    paged = [key for key, value in blocks.items() if isinstance(value, PagedExpert)]
                    self.assertEqual(len(paged), 1)
                    self.assertLessEqual((await cache.snapshot())["resident_bytes"], 6)
            asyncio.run(exercise())

    def test_large_expert_uses_bounded_ssd_chunks_and_unload_lock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "large.gguf"
            model.write_bytes(bytes(range(16)))
            manifest = root / "experts.json"
            manifest.write_text(json.dumps({"experts": {"layer1.expert7": {
                "path": str(model), "offset": 0, "size": 16}}}))
            cache = ExpertCache(str(manifest), str(root / "telemetry.json"), budget_bytes=4)
            handler = ActiveExpertHandler(cache, prefetch_width=0)

            async def exercise():
                async with handler.acquire_for_compute("test", ["layer1.expert7"]) as blocks:
                    paged = blocks["layer1.expert7"]
                    self.assertIsInstance(paged, PagedExpert)
                    loaded = []
                    async for offset, block in paged.iter_chunks(chunk_bytes=3):
                        loaded.extend((offset + i, int(value)) for i, value in enumerate(block))
                    self.assertEqual(bytes(value for _, value in sorted(loaded)), bytes(range(16)))
                    blocked = await cache.unload_file(str(model), timeout_seconds=0)
                    self.assertFalse(blocked["unloaded"])
                result = await cache.unload_file(str(model), timeout_seconds=0)
                self.assertTrue(result["unloaded"])
            asyncio.run(exercise())


class AutoOptimizerTests(unittest.TestCase):
    def test_candidates_are_bounded_and_valid(self):
        from engine.optimizer import build_candidates, score_result
        rows = build_candidates({"context_size": 2048, "batch_size": 256, "threads": 4})
        self.assertLessEqual(len(rows), 6)
        self.assertTrue(rows)
        for row in rows:
            self.assertGreaterEqual(row["context_size"], 512)
            self.assertGreaterEqual(row["batch_size"], 32)
            self.assertGreaterEqual(row["threads"], 1)
        self.assertGreater(score_result(20, 100), score_result(10, 100))

    def test_optimizer_score_weights_generation_more_than_prefill(self):
        from engine.optimizer import score_result
        baseline = score_result(20, 50)
        faster_generation = score_result(25, 50)
        faster_prefill = score_result(20, 55)
        self.assertGreater(faster_generation - baseline, faster_prefill - baseline)

    def test_runtime_optimizer_benchmarks_and_leaves_best_loaded(self):
        from engine.runtime import LlamaCppRuntime
        attempts = []

        class FakeLlama:
            def __init__(self, **kwargs):
                attempts.append(kwargs)
            def create_chat_completion(self, **kwargs):
                return iter([
                    {"choices": [{"delta": {"content": "A generated answer."}}]},
                    {"choices": [], "usage": {"prompt_tokens": 14, "completion_tokens": 5}},
                ])
            def tokenize(self, value):
                return list(value[:14])
            def close(self):
                pass

        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = FakeLlama
        fake_module.llama_supports_gpu_offload = lambda: False
        runtime = LlamaCppRuntime()
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "fixture.gguf"
            model.write_bytes(b"GGUF")
            with patch.dict("sys.modules", {"llama_cpp": fake_module}):
                result = runtime.optimize(model, max_safe_gpu_layers=0, backend="cpu",
                                          reserve_bytes=0, max_context=2048)
                self.assertGreater(result["passed"], 0)
                self.assertEqual(runtime.status()["model_path"], str(model.resolve()))
                self.assertEqual(runtime.status()["context_size"], result["best"]["config"]["context_size"])
                self.assertGreaterEqual(len(attempts), result["passed"] + 1)
                runtime.unload()


class VramExpertCacheTests(unittest.TestCase):
    class FakeCuda:
        def __init__(self, total=1000):
            self.total = total
            self.used = 0

        def memory_info(self):
            return self.total - self.used, self.total

        def upload(self, data):
            size = int(getattr(data, "nbytes", len(data)))
            self.used += size
            return {"device": "fake-cuda", "size": size}

        def synchronize(self):
            return None

        def release(self, _tensor, size):
            self.used = max(0, self.used - int(size))

    def test_snapshot_reports_upload_and_device_read_speeds(self):
        from moe_boost.vram_cache import VramExpertCache

        class TimedCuda(self.FakeCuda):
            last_upload_ms = 2.0

            def read_device(self, _tensor):
                return 0.5

        cache = VramExpertCache(headroom_mb=0, max_fraction=0.8, promote_after_hits=1,
                                adapter=TimedCuda(total=10_000_000))
        payload = bytearray(1024 * 1024)
        self.assertTrue(cache.observe("e0", payload, 1))
        cache.acquire_device("e0")
        cache.release_device("e0")
        snap = cache.snapshot()
        self.assertEqual(snap["upload_bytes"], 1024 * 1024)
        self.assertAlmostEqual(snap["upload_speed_mbs"], 500.0, places=1)
        self.assertAlmostEqual(snap["read_speed_mbs"], 2000.0, places=1)
        self.assertGreater(snap["read_speed_mbs"], snap["upload_speed_mbs"])

    def test_promotes_only_after_hot_threshold_and_reports_staging_only(self):
        from moe_boost.vram_cache import VramExpertCache
        adapter = self.FakeCuda()
        cache = VramExpertCache(headroom_mb=0, max_fraction=0.8,
                                promote_after_hits=2, adapter=adapter)
        payload = bytearray(100)
        self.assertFalse(cache.observe("layer0.expert0", payload, 1))
        self.assertTrue(cache.observe("layer0.expert0", payload, 2))
        snapshot = cache.snapshot()
        self.assertEqual(snapshot["cached_experts"], ["layer0.expert0"])
        self.assertEqual(snapshot["resident_bytes"], 100)
        self.assertFalse(snapshot["inference_backend_connected"])
        tensor = cache.acquire_device("layer0.expert0")
        self.assertIsNotNone(tensor)
        cache.release_device("layer0.expert0")

    def test_recently_evicted_expert_obeys_promotion_cooldown(self):
        from moe_boost.vram_cache import VramExpertCache
        adapter = self.FakeCuda(total=1000)
        cache = VramExpertCache(headroom_mb=0, max_fraction=0.2,
                                promote_after_hits=1, adapter=adapter,
                                promotion_cooldown_seconds=60)
        self.assertTrue(cache.observe("hot", bytearray(100), 5))
        self.assertTrue(cache._evict_one())
        self.assertFalse(cache.observe("hot", bytearray(100), 10))
        self.assertNotIn("hot", cache.entries)

    def test_respects_headroom_and_does_not_overcommit(self):
        from moe_boost.vram_cache import VramExpertCache
        adapter = self.FakeCuda(total=1000)
        cache = VramExpertCache(headroom_mb=0, max_fraction=0.15,
                                promote_after_hits=1, adapter=adapter)
        self.assertFalse(cache.observe("too-large", bytearray(300), 1))
        self.assertEqual(cache.snapshot()["resident_bytes"], 0)

    def test_cpu_only_mode_degrades_cleanly(self):
        from moe_boost.vram_cache import VramExpertCache
        cache = VramExpertCache(adapter=None)
        cache.adapter = None
        cache.disabled_reason = "CUDA unavailable"
        self.assertFalse(cache.observe("expert", bytearray(8), 5))
        self.assertFalse(cache.snapshot()["available"])

    def test_pinned_hot_expert_is_not_evicted(self):
        from moe_boost.vram_cache import VramExpertCache
        adapter = self.FakeCuda(total=1000)
        cache = VramExpertCache(headroom_mb=0, max_fraction=0.2,
                                promote_after_hits=1, adapter=adapter)
        self.assertTrue(cache.observe("expert-a", bytearray(100), 1))
        self.assertTrue(cache.observe("expert-b", bytearray(100), 1))
        cache.acquire_device("expert-a")
        # A third promotion must not evict the actively leased expert-a.
        self.assertTrue(cache.observe("expert-c", bytearray(100), 1))
        self.assertIn("expert-a", cache.entries)
        cache.release_device("expert-a")

    def test_ram_eviction_demotes_cold_entry_before_vram_hot_backing(self):
        from collections import OrderedDict
        from moe_boost.cache import Entry
        from moe_boost.vram_cache import VramExpertCache
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = ExpertCache(str(root / "manifest.json"), str(root / "telemetry.json"), budget_bytes=1024)
            adapter = self.FakeCuda(total=10000)
            cache.vram_cache = VramExpertCache(headroom_mb=0, max_fraction=0.8,
                                               promote_after_hits=1, adapter=adapter)
            hot = bytearray(100)
            cache.entries = OrderedDict([
                ("cold", Entry(bytearray(100))),
                ("hot", Entry(hot)),
            ])
            cache.resident_bytes = 200
            cache.hits = {"cold": 0, "hot": 20}
            cache.invocations = {"cold": 1, "hot": 20}
            self.assertTrue(cache.vram_cache.observe("hot", hot, 20))
            self.assertTrue(cache._evict_one())
            self.assertIn("hot", cache.entries)
            self.assertNotIn("cold", cache.entries)
            self.assertEqual(cache.resident_bytes, 100)

    def test_displaced_ssd_promoted_hot_expert_is_demoted_to_ram(self):
        from moe_boost.vram_cache import VramExpertCache
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "weights.bin"
            source.write_bytes(bytes(range(200)))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"experts": {
                "hot-a": {"path": str(source), "offset": 0, "size": 100},
                "hot-b": {"path": str(source), "offset": 100, "size": 100}}}), encoding="utf-8")
            cache = ExpertCache(str(manifest), str(root / "telemetry.json"), budget_bytes=150)
            adapter = self.FakeCuda(total=1000)
            cache.vram_cache = VramExpertCache(headroom_mb=0, max_fraction=0.15,
                                               promote_after_hits=1, adapter=adapter)
            cache.vram_cache.on_demote = cache._demote_vram_expert_to_ram
            # hot-a came directly from SSD and has no RAM backing yet.
            self.assertTrue(cache._try_stage_ssd_hot_expert("hot-a", 100, 1))
            # Promoting hot-b displaces hot-a; the callback must page cold SSD
            # ranges into RAM before dropping the old CUDA copy.
            self.assertTrue(cache._try_stage_ssd_hot_expert("hot-b", 100, 1))
            self.assertIn("hot-a", cache.entries)
            self.assertNotIn("hot-a", cache.vram_cache.entries)
            self.assertEqual(cache.vram_cache.snapshot()["last_demotion_targets"]["hot-a"], "RAM")

    def test_direct_ssd_to_vram_hot_promotion_uses_temporary_buffer(self):
        from moe_boost.vram_cache import VramExpertCache
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "weights.bin"
            source.write_bytes(bytes(range(100)))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"experts": {"ssd-hot": {
                "path": str(source), "offset": 0, "size": 100}}}), encoding="utf-8")
            cache = ExpertCache(str(manifest), str(root / "telemetry.json"), budget_bytes=0)
            cache.vram_cache = VramExpertCache(headroom_mb=0, max_fraction=0.8,
                                               promote_after_hits=2, adapter=self.FakeCuda(total=1000))
            with patch("moe_boost.cache.safe_ram_budget", return_value=1024):
                self.assertTrue(cache._try_stage_ssd_hot_expert("ssd-hot", 100, 2))
            self.assertIn("ssd-hot", cache.vram_cache.entries)
            self.assertEqual(cache.vram_cache.snapshot()["promotion_sources"]["ssd-hot"], "ssd")
            self.assertNotIn("ssd-hot", cache.entries)

    def test_vram_demotion_keeps_hot_expert_in_ram_without_ssd_read(self):
        from moe_boost.vram_cache import VramExpertCache
        adapter = self.FakeCuda(total=1000)
        vram = VramExpertCache(headroom_mb=0, max_fraction=0.8,
                               promote_after_hits=1, adapter=adapter)
        host_copy = bytearray(100)
        self.assertTrue(vram.observe("hot-expert", host_copy, 5))
        # The RAM owner is intentionally independent and remains resident when
        # the VRAM tier drops its device copy.
        self.assertTrue(vram._evict_one())
        self.assertNotIn("hot-expert", vram.entries)
        self.assertEqual(len(host_copy), 100)

    def test_pread_segments_into_and_handle_pooling(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "segmented.bin"
            source.write_bytes(b"AAAABBBBCCCCDDDD")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"experts": {
                "seg-expert": {
                    "path": str(source),
                    "offset": 0,
                    "size": 8,
                    "segments": [
                        {"path": str(source), "offset": 0, "size": 4},
                        {"path": str(source), "offset": 8, "size": 4},
                    ],
                },
            }}), encoding="utf-8")
            cache = ExpertCache(str(manifest), str(root / "telemetry.json"), budget_bytes=64)

            async def exercise():
                async with cache.acquire("seg-expert") as block:
                    self.assertEqual(bytes(block.tolist()), b"AAAACCCC")
                self.assertIn(source.resolve(), cache._open_handles)
                unloaded = await cache.unload_file(str(source), timeout_seconds=0)
                self.assertTrue(unloaded["unloaded"])
                self.assertNotIn(source.resolve(), cache._open_handles)

            asyncio.run(exercise())

    def test_telemetry_time_size_formatting_and_tok_s(self):
        import server
        self.assertEqual(server.format_duration_ms(0), "0.00s")
        self.assertEqual(server.format_duration_ms(250), "0.25s (250.0 ms)")
        self.assertEqual(server.format_duration_ms(4500), "4.50s")
        self.assertEqual(server.format_duration_ms(125400), "2m 5.4s")
        self.assertEqual(server.format_duration_ms(3725000), "1h 2m 5.0s")

        self.assertEqual(server.format_size_gb_mb(0), "0.00 MB")
        self.assertEqual(server.format_size_gb_mb(150 * 1024 * 1024), "150.00 MB")
        self.assertEqual(server.format_size_gb_mb(int(3.5 * (1024 ** 3))), "3.50 GB")

        server.update_request_metrics(
            started=100.0,
            first_byte=100.5,
            prompt_tokens=20,
            completion_tokens=40,
            output_chars=160,
            prompt_seconds=0.5,
            generation_seconds=2.0,
        )
        tel = server.build_chat_telemetry()
        self.assertAlmostEqual(tel["prefill_tokens_per_second"], 40.0)
        self.assertAlmostEqual(tel["generation_tokens_per_second"], 20.0)
        self.assertEqual(tel["tok_per_sec"], {"prefill": 40.0, "generation": 20.0})
        self.assertIn("ssd_read_time_formatted", tel)
        self.assertIn("total_read_size_formatted", tel)
        self.assertIn("tier_io", tel)
        for tier_name in ("vram", "ram", "ssd"):
            self.assertIn(tier_name, tel["tier_io"])
            self.assertIn("read_speed_formatted", tel["tier_io"][tier_name])
            self.assertIn("read_time_formatted", tel["tier_io"][tier_name])
            self.assertIn("read_size_formatted", tel["tier_io"][tier_name])

    def test_mtp_draft_predictor_and_cache_drop_all(self):
        import numpy as np
        from engine.runtime import MtpDraftPredictor
        from moe_boost.vram_cache import VramExpertCache

        mtp = MtpDraftPredictor(enabled=True, mode="auto", num_tokens=3, ngram_n=2)
        mtp.reset_request()
        seq = np.array([10, 20, 30, 40, 50, 10, 20], dtype=np.intc)
        drafted = mtp(seq)
        self.assertEqual(drafted.tolist(), [30, 40, 50])
        # Simulate target model accepting 2 of the 3 drafted tokens (+1 sampled token = +3 length)
        seq2 = np.array([10, 20, 30, 40, 50, 10, 20, 30, 40, 99], dtype=np.intc)
        status = mtp.finalize_request(len(seq2))
        self.assertEqual(status["drafted_tokens"], 3)
        self.assertEqual(status["accepted_tokens"], 2)
        self.assertGreater(status["acceptance_rate_percent"], 60.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "weights.bin"
            source.write_bytes(b"0123456789ABCDEF")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"experts": {
                "exp-a": {"path": str(source), "offset": 0, "size": 8},
                "exp-b": {"path": str(source), "offset": 8, "size": 8},
            }}), encoding="utf-8")
            cache = ExpertCache(str(manifest), str(root / "telemetry.json"), budget_bytes=64)
            cache.vram_cache = VramExpertCache(headroom_mb=0, max_fraction=0.8,
                                               promote_after_hits=1, adapter=self.FakeCuda(total=1000))

            async def exercise_drop():
                async with cache.acquire("exp-a") as _:
                    pass
                async with cache.acquire("exp-a") as _:
                    pass
                snap = await cache.snapshot()
                self.assertGreater(snap["tier_io"]["ram"]["read_bytes"], 0)
                self.assertGreater(snap["tier_io"]["vram"]["read_bytes"], 0)
                dropped = await cache.drop_all(drop_page_cache=True)
                self.assertTrue(dropped["dropped"])
                self.assertEqual(dropped["freed_ram_cache_bytes"], 8)
                self.assertEqual(dropped["freed_vram_bytes"], 8)
                self.assertEqual(cache.resident_bytes, 0)
                self.assertEqual(len(cache.vram_cache.entries), 0)

            asyncio.run(exercise_drop())

    def test_tiers_optimize_endpoints_and_telemetry(self):
        from fastapi.testclient import TestClient
        import server
        from moe_boost.vram_cache import VramExpertCache

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "tier_test.gguf"
            model.write_bytes(b"A" * 4096)
            manifest = root / "experts.json"
            manifest.write_text(json.dumps({"experts": {
                "exp-0": {"path": str(model), "offset": 0, "size": 2048},
                "exp-1": {"path": str(model), "offset": 2048, "size": 2048},
            }}), encoding="utf-8")
            test_cache = ExpertCache(str(manifest), str(root / "telemetry.json"), budget_bytes=65536)

            class TimedFakeCuda(self.FakeCuda):
                last_upload_ms = 0.25

                def warm_up_and_preallocate(self):
                    return {"non_blocking_stream": True, "pinned_dma_bytes": 16 * 1024 * 1024, "vram_slab_bytes": 64 * 1024 * 1024}

                def read_device(self, _tensor):
                    return 0.02

            test_cache.vram_cache = VramExpertCache(headroom_mb=0, max_fraction=0.8,
                                                    promote_after_hits=1, adapter=TimedFakeCuda(total=10_000_000))
            orig_cache = server.cache
            orig_last = server.last_tier_optimization
            try:
                server.cache = test_cache
                server.last_tier_optimization = None
                client = TestClient(server.app)

                before_get = client.get("/api/tiers/optimize")
                self.assertEqual(before_get.status_code, 200)
                self.assertEqual(before_get.json()["status"], "not_run")

                res = client.post("/api/tiers/optimize", json={"max_prefetch_experts": 8})
                self.assertEqual(res.status_code, 200)
                body = res.json()
                self.assertEqual(body["status"], "optimized")
                self.assertIn("tier_io", body["before"])
                self.assertIn("tier_io", body["after"])
                for tier in ("vram", "ram", "ssd"):
                    self.assertIn(tier, body["after"]["tier_io"])
                self.assertTrue(body["optimizations_applied"])
                self.assertEqual(body["active_settings"]["ssd"]["prefetched_experts"], 2)
                self.assertEqual(body["active_settings"]["vram"]["promoted_experts"], 2)
                self.assertGreater(body["active_settings"]["vram"]["calibrated_read_speed_mbs"], 0)
                self.assertGreater(body["active_settings"]["ram"]["calibrated_read_speed_mbs"], 0)

                after_get = client.get("/api/tiers/optimize")
                self.assertEqual(after_get.status_code, 200)
                self.assertEqual(after_get.json()["status"], "optimized")

                tel = client.get("/api/telemetry")
                self.assertEqual(tel.status_code, 200)
                self.assertIsNotNone(tel.json()["tier_optimizer"])
                self.assertEqual(tel.json()["tier_optimizer"]["status"], "optimized")
            finally:
                server.cache = orig_cache
                server.last_tier_optimization = orig_last

    def test_mtp_frequency_chaining_and_runtime_kv_cache_tuning(self):
        import os
        import numpy as np
        from engine.runtime import LlamaCppRuntime, MtpDraftPredictor

        mtp = MtpDraftPredictor(enabled=True, mode="auto", num_tokens=6, ngram_n=3)
        mtp.reset_request()
        # Repeat (10, 20, 30) -> (40, 50, 60, 70, 80, 90) twice and once -> (41, ...)
        seq = np.array([
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 101,
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 102,
            10, 20, 30, 41, 51, 61, 71, 81, 91, 100, 103,
            200, 201, 202,
        ], dtype=np.intc)
        mtp(seq)
        mtp.finalize_request(len(seq))

        # New request with no prompt match should recall the higher-frequency 3-gram continuation
        mtp.reset_request()
        drafted = mtp(np.array([300, 301, 10, 20, 30], dtype=np.intc))
        self.assertEqual(drafted.tolist(), [40, 50, 60, 70, 80, 90])

        # Short prompt lookup match hitting the end of the prompt should chain/extend using learned continuations
        mtp.reset_request()
        chained = mtp(np.array([300, 10, 20, 30, 10, 20, 30], dtype=np.intc))
        self.assertEqual(chained.tolist(), [10, 20, 30, 40, 50, 60])

        # Verify default f16 KV cache (with Q8_0 override), op_offload=False, and physical-core decode threads
        captured = {}

        class DetailedFakeLlama:
            def __init__(self, model_path, n_ctx=512, n_batch=512, n_ubatch=512,
                         n_threads=4, n_threads_batch=8, n_gpu_layers=0,
                         offload_kqv=True, flash_attn=False, op_offload=None,
                         type_k=None, type_v=None, use_mmap=True, use_mlock=False,
                         verbose=False):
                captured.update({
                    "n_threads": n_threads,
                    "n_threads_batch": n_threads_batch,
                    "flash_attn": flash_attn,
                    "op_offload": op_offload,
                    "type_k": type_k,
                    "type_v": type_v,
                })

            def close(self):
                pass

        fake_module = types.ModuleType("llama_cpp")
        fake_module.Llama = DetailedFakeLlama
        fake_module.llama_supports_gpu_offload = lambda: True

        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "kv_test.gguf"
            model.write_bytes(b"GGUF")
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MOE_THREADS", None)
                os.environ.pop("MOE_THREADS_BATCH", None)
                os.environ.pop("MOE_KV_CACHE_TYPE_K", None)
                os.environ.pop("MOE_KV_CACHE_TYPE_V", None)
                os.environ.pop("MOE_OP_OFFLOAD", None)
                runtime = LlamaCppRuntime()
                with patch.dict("sys.modules", {"llama_cpp": fake_module}):
                    status = runtime.load(model, max_safe_gpu_layers=4, backend="cuda")
                    self.assertEqual(captured["type_k"], 1)
                    self.assertEqual(captured["type_v"], 1)
                    self.assertTrue(captured["flash_attn"])
                    self.assertFalse(captured["op_offload"])
                    self.assertLessEqual(captured["n_threads"], captured["n_threads_batch"])
                    self.assertEqual(status["kv_cache_type_k"], "f16")
                    self.assertEqual(status["kv_cache_type_v"], "f16")
                    self.assertFalse(status["op_offload"])
                    runtime.unload()
                os.environ["MOE_KV_CACHE_TYPE_K"] = "q8_0"
                os.environ["MOE_KV_CACHE_TYPE_V"] = "q8_0"
                with patch.dict("sys.modules", {"llama_cpp": fake_module}):
                    status_q8 = runtime.load(model, max_safe_gpu_layers=4, backend="cuda")
                    self.assertEqual(captured["type_k"], 8)
                    self.assertEqual(captured["type_v"], 8)
                    self.assertEqual(status_q8["kv_cache_type_k"], "q8_0")
                    self.assertEqual(status_q8["kv_cache_type_v"], "q8_0")
                    runtime.unload()

    def test_configured_max_ram_bytes_and_safe_ram_budget(self):
        import os
        from moe_boost.cache import _parse_bytes_value, get_configured_max_ram_bytes, safe_ram_budget
        self.assertEqual(_parse_bytes_value("16GB"), 16 * 1024**3)
        self.assertEqual(_parse_bytes_value("8192MB"), 8192 * 1024**2)
        self.assertEqual(_parse_bytes_value("1024"), 1024)

        with patch.dict(os.environ, {"MOE_MAX_RAM": "4GB"}, clear=False):
            self.assertEqual(get_configured_max_ram_bytes(), 4 * 1024**3)
            budget = safe_ram_budget()
            self.assertLessEqual(budget, 4 * 1024**3)

        with patch.dict(os.environ, {"MOE_MAX_RAM_MB": "2048"}, clear=False):
            os.environ.pop("MOE_MAX_RAM", None)
            self.assertEqual(get_configured_max_ram_bytes(), 2048 * 1024**2)
            budget = safe_ram_budget()
            self.assertLessEqual(budget, 2048 * 1024**2)

    def test_api_max_ram_endpoints(self):
        from fastapi.testclient import TestClient
        import server
        client = TestClient(server.app)
        get_res = client.get("/api/runtime/max-ram")
        self.assertEqual(get_res.status_code, 200)

        put_res = client.put("/api/runtime/max-ram", json={"max_ram": "12GB"})
        self.assertEqual(put_res.status_code, 200)
        self.assertEqual(put_res.json()["max_ram_bytes"], 12 * 1024**3)

        tel_res = client.get("/api/telemetry")
        self.assertEqual(tel_res.status_code, 200)
        self.assertEqual(tel_res.json()["memory"]["ram"]["max_ram_formatted"], "12.00 GB")

        clear_res = client.put("/api/runtime/max-ram", json={"max_ram": ""})
        self.assertEqual(clear_res.status_code, 200)
        self.assertIsNone(clear_res.json()["max_ram_bytes"])


if __name__ == "__main__":
    unittest.main()
