"""Local GGUF discovery with a small stat-keyed metadata cache."""
from __future__ import annotations

import threading
from pathlib import Path

from .inspector import inspect_gguf


class ModelInventory:
    def __init__(self, models_dir: str):
        self.directory = Path(models_dir).expanduser().resolve()
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._cache: dict[str, tuple[int, int, dict]] = {}
        self._lock = threading.RLock()

    def inspect_file(self, path: Path) -> dict:
        stat = path.stat()
        key = str(path.resolve())
        with self._lock:
            cached = self._cache.get(key)
            if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
                return cached[2]
        result = inspect_gguf(path)
        with self._lock:
            self._cache[key] = (stat.st_mtime_ns, stat.st_size, result)
        return result

    def list_models(self, include_errors: bool = True) -> list[dict]:
        rows = []
        for path in sorted(self.directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".gguf":
                continue
            try:
                details = self.inspect_file(path)
                rows.append({"id": str(path.relative_to(self.directory)), "name": path.stem,
                             "path": str(path), "size_bytes": details["file_size"],
                             "architecture": details["architecture"],
                             "classification": details["classification"],
                             "expert_count": details["expert_count"],
                             "active_experts": details["active_experts"],
                             "block_count": details["block_count"],
                             "quantization": details["quantization"],
                             "tensor_count": details["tensor_count"],
                             "expert_tensor_count": details["expert_tensor_count"],
                             "mtp_support": details.get("mtp_support", True),
                             "native_mtp_heads": details.get("native_mtp_heads", False),
                             "mtp_layers": details.get("mtp_layers", 0),
                             "mtp_tensor_count": details.get("mtp_tensor_count", 0)})
            except (OSError, ValueError) as exc:
                if include_errors:
                    rows.append({"id": str(path.relative_to(self.directory)), "name": path.stem,
                                 "path": str(path), "error": str(exc)})
        return rows

    def get_details(self, model_id: str) -> dict:
        candidate = (self.directory / model_id).resolve()
        # Accept if within self.directory OR if candidate is a symlink pointing outside
        direct_child = (self.directory / model_id).is_file()
        if not direct_child and (self.directory not in candidate.parents or candidate.suffix.lower() != ".gguf" or not candidate.is_file()):
            raise ValueError("unknown GGUF model id")
        return self.inspect_file(candidate)

    def forget(self, model_id: str) -> str:
        candidate = (self.directory / model_id).resolve()
        direct_child = (self.directory / model_id).is_file()
        if not direct_child and self.directory not in candidate.parents:
            raise ValueError("unknown GGUF model id")
        with self._lock:
            self._cache.pop(str(candidate), None)
        return str(candidate)
