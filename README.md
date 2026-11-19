# MoE Boost by Gravitize.IGP

MoE Boost is a local GGUF inspection, paging, cache, and inference service. It embeds the llama.cpp C++ tensor runtime in the FastAPI process through `llama-cpp-python`; chat does not proxy to another model server. The project also builds its own C++ `pread` pager and keeps GGUF inspection, memory planning, API, and dashboard logic in this repository.

## Install and start

```bash
./install.sh
./start.sh
```

`start.sh` defaults to `/mnt/ollama-models/OllamaModels/models`, binds to `127.0.0.1:4133`, and limits inference threads to four. Override `MOE_MODELS_DIR`, `MOE_HOST`, or `MOE_PORT` when needed.

The installer prefers CUDA when `nvcc` and an NVIDIA GPU are detected, adds Vulkan as a runtime fallback when its SDK is available, then falls back to CPU. Without CUDA it builds Vulkan when available, then CPU. CUDA builds target the local GPU architecture; run the installer on another machine to build for its card. At load time MoE Boost sizes dense-model layer offload from live free VRAM and full layer tensor sizes, keeping the configured headroom. For MoE models, it keeps whole weight layers, including expert FFNs, host-mapped and enables K/Q/V plus KV-cache GPU offload where supported. If CUDA model allocation fails in a dual-backend build, it retries on the matching discrete Vulkan device, then CPU. `MOE_GPU_LAYERS=0` forces CPU; the default `-1` lets the memory planner choose. `MOE_CONTEXT_SIZE` and `MOE_BATCH_SIZE` default to 2048 and 256. K/Q/V offload is enabled, and inference and batch thread counts default to four with a hard cap of four.

Open `http://127.0.0.1:4133/gui` for multi-turn chat, telemetry, the GGUF inventory, and Hugging Face downloads. The first chat request loads its selected GGUF into llama.cpp; the runtime load and unload controls can do this explicitly. Terminal agents can use `POST /v1/chat/completions` with a local model ID returned by `GET /v1/models`. Streaming responses report per-message generation telemetry in the Chat Playground: generated token count, generation tokens/second, time-to-first-token, and prompt tokens/prefill-speed when the installed `llama-cpp-python` exposes streaming usage. Prefill speed is explicitly an approximation based on prompt tokens divided by TTFT because the Python streaming API does not consistently expose isolated prompt-evaluation timing. Older bindings without `stream_options` continue to work, but prompt token counts may be unavailable.

## GGUF inspection and memory planning

The inspector reads GGUF v2/v3 metadata and tensor records without loading tensor payloads. It classifies MoE models, extracts expert/top-k counts, and maps tensor names, layers, quantization types, offsets, and sizes. Before runtime loading, MoE Boost checks the model against current VRAM and safe RAM estimates. llama.cpp memory-maps GGUF weights and executes inference; the separate MoE Boost expert pager/cache is not yet wired into llama.cpp's per-token expert dispatch.

`POST /api/models/load-check` reports estimated fit. `POST /api/runtime/load` loads a local model into llama.cpp after a fit check; `POST /api/runtime/unload` releases the runtime model. Chat loads on demand. `POST /api/models/load` prepares the separate expert slice index/cache, and `POST /api/models/unload` unloads that paging cache and manifest. These are distinct memory paths.


## Automatic performance optimizer

The Chat Playground includes **Auto-optimize (benchmark configurations)**. It runs a bounded sweep of up to six real llama.cpp configurations using the selected GGUF and a fixed deterministic prompt. The candidates vary context size (up to the requested/model-supported limit and 4096 by default), prompt batch size, and CPU thread count. Each candidate is unloaded before the next trial to avoid stacking model allocations. Candidates are skipped if the current available host RAM plus the candidate's scratch estimate would breach the configured physical-RAM reserve (20% by default) plus a 512 MiB safety margin. After testing, the highest-scoring passing candidate is loaded and left active.

The ranking score is `0.70 × generation tokens/s + 0.30 × prefill tokens/s proxy`. Generation speed uses completion token usage when the backend reports it, otherwise tokenizes the collected output. Prefill is a TTFT-based proxy (prompt tokens divided by time-to-first-token), not isolated backend prompt-evaluation timing. Benchmark load time is reported separately and is not included in the speed score. This is an empirical short-run benchmark, not a guarantee that the selected settings are optimal for every prompt length or workload. Run it again after changing the model, backend, GPU build, or workload.

The optimizer runs as a background job: `POST /api/runtime/optimize` starts it and `GET /api/runtime/optimize/{job_id}` reports progress and results. It does not change model weights or quantization. The final selected configuration remains loaded for subsequent chat requests.

## MoE expert pager and cache

`cpp_core/moe_pager.cpp` exposes checked `pread` into caller-owned NumPy buffers and read metrics. `engine/manifest.py` maps GGUF tensor slices to layer/expert IDs. The RAM cache enforces a dynamic budget, reference locks, frequency-aware eviction, warm staging, and per-expert load/stream timing records. The three-tier policy is **VRAM → RAM → SSD**: a hot expert promoted to VRAM retains its RAM backing; if VRAM pressure displaces it, it is immediately available in RAM without an SSD read. A hot expert may also be staged directly SSD → VRAM through a temporary host buffer when safe host-memory headroom and VRAM capacity permit; if that expert is later displaced, the cache first tries to make RAM space by demoting cold/low-use RAM entries to SSD, then restores the hot expert to RAM. RAM eviction explicitly protects experts still resident in VRAM. A per-expert promotion cooldown (`MOE_VRAM_PROMOTION_COOLDOWN_SECONDS`, default 2 seconds) reduces cache thrashing. If all RAM entries are active or protected by VRAM residency, new blocks use bounded SSD paging rather than breaking the protected reserve or evicting a hot VRAM expert's RAM copy. Active RAM and CUDA leases are pinned against eviction. A separate optional CUDA hot-expert staging cache promotes frequently accessed RAM blocks after a configurable hit threshold (`MOE_VRAM_PROMOTE_HITS`, default 2), keeps configurable VRAM headroom (`MOE_VRAM_HEADROOM_MB`, default 500 MiB), and caps allocation to 70% of total VRAM by default (`MOE_VRAM_CACHE_FRACTION`). It reports VRAM residency and safely disables itself when CUDA/PyTorch is unavailable. Cache telemetry separately lists VRAM, RAM, SSD, hot-on-SSD overflow, and hot experts backed in RAM. A hot expert remains SSD-backed only when it cannot be safely retained in RAM under the configured reserve or is too large/in-use for the available working set. `POST /api/experts/activate` is the adapter for a router that supplies expert IDs. llama.cpp's current integration does not expose those routes to this adapter, so chat-driven expert heat/residency is not yet recorded by the MoE Boost cache. The dashboard refreshes its cache telemetry every two seconds while visible. MoE model records are stored as JSON in a `.moe_experts/<model-name>-<fingerprint>/` folder beside each GGUF, with `<model-name>.manifest.json` and `<model-name>.telemetry.json` (set `MOE_EXPERT_DATA_DIR` to use a shared root). Loading the same GGUF from a different folder refreshes its manifest paths while reusing its learned expert heat and transition counts; the top 15% of observed experts can be staged within a 15% RAM budget, RAM entries are normal residents, and SSD entries are cold/idle. Important: this CUDA cache currently stages expert tensors only; the installed llama.cpp adapter does not expose per-token expert-weight hooks, so staged tensors are not consumed by chat generation yet. The dashboard marks this as staging-only and does not claim a generation-speed gain. Actual hot-expert inference acceleration requires an expert-aware backend adapter.

Copy `experts.example.json` to `experts.json` to configure a standalone expert manifest. The pager validates every byte range. Runtime custom model folders can be set with `MOE_MODELS_DIR` or `PUT /api/models/directory`. Hugging Face pulls use `HF_TOKEN` for private repositories and save under the configured model folder.

## Useful endpoints

- `GET /health`, `GET /api/runtime`, `GET /api/telemetry`
- `GET /api/models`, `GET /v1/models`
- `POST /api/models/load-check`, `POST /api/models/load`
- `POST /api/runtime/load`, `POST /api/runtime/unload`, `POST /api/runtime/optimize`, `GET /api/runtime/optimize/{job_id}`
- `POST /api/experts/activate`, `GET /v1/moe/cache`
- `POST /v1/chat/completions` streams per-message generation telemetry in the Chat Playground: generated token count, generation tokens/second, time-to-first-token, and prompt tokens/prefill-speed when the installed llama-cpp-python exposes streaming usage. Prefill speed is explicitly an approximation based on prompt tokens divided by TTFT because the Python streaming API does not consistently expose isolated prompt-evaluation timing. Older bindings without `stream_options` continue to work, but prompt token counts may be unavailable.
- `POST /api/models/pull`, `GET /api/models/pulls/{job_id}`

## Conservative RAM reserve and model-fit estimates

MoE Boost defaults to preserving an OS/application RAM reserve and allows setting a strict maximum RAM budget that the application and cache will not exceed:
- Set `MOE_MAX_RAM=16GB` (or `MOE_MAX_RAM_BYTES`, `MOE_MAX_RAM_MB`) to enforce a hard maximum RAM ceiling (e.g. `12GB`, `16384MB`). The cache and model budget will never exceed this ceiling.
- Set `MOE_RAM_RESERVE_PERCENT=20` (accepted range 5–60) before starting the server to retain an OS reserve percentage (e.g. 10–20%). Restart the app for the cache budget to be recalculated.

The model load check reports estimated VRAM model-weight use, host working-set size, free RAM before loading, and projected free RAM after the estimated working set. These are conservative planning estimates, not exact runtime allocations: GGUF weights are memory-mapped, clean mapped pages can be reclaimed by Linux, and KV cache/compute buffers depend on context and batch settings. The reserve is a planner guardrail, not a hard OS memory reservation.
