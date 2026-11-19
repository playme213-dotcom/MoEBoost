#!/usr/bin/env python3
"""
MoEBoost In-Process Python Model Loader & Interactive Chat.
Loads a GGUF model directly via MoEBoost's LlamaCppRuntime (without needing the HTTP server)
and runs interactive chat with accurate token speed telemetry (prefill & generation tok/s).
"""

import sys
import os
import time
from pathlib import Path

# Ensure MoEBoost source directory is on PYTHONPATH
CURRENT_DIR = Path(__file__).resolve().parent
MOEBOOST_ROOT = CURRENT_DIR / "MoEBoost" if (CURRENT_DIR / "MoEBoost").exists() else CURRENT_DIR
if str(MOEBOOST_ROOT) not in sys.path:
    sys.path.insert(0, str(MOEBOOST_ROOT))

try:
    from engine.runtime import LlamaCppRuntime
except ImportError:
    # Direct relative fallback
    try:
        from Desktop.MoEBoost.engine.runtime import LlamaCppRuntime
    except ImportError:
        raise ImportError(f"Could not import LlamaCppRuntime from {MOEBOOST_ROOT}. Ensure MoEBoost is installed.")

def load_moeboost_model(
    model_path: str,
    gpu_layers: int = -1,
    backend: str = "cuda",
    context_size: int = 4096,
    batch_size: int = 512,
    threads: int = 8
) -> LlamaCppRuntime:
    """
    Initializes MoEBoost engine and loads the specified GGUF model with custom tier offload.
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    print(f"📦 [MoEBoost Loader] Initializing runtime backend '{backend}'...")
    runtime = LlamaCppRuntime()
    
    # Configure context and batch parameters
    runtime._settings["context_size"] = context_size
    runtime._settings["batch_size"] = batch_size
    runtime._settings["threads"] = threads
    runtime._settings["requested_gpu_layers"] = gpu_layers

    print(f"⚡ [MoEBoost Loader] Loading model: {path.name}")
    print(f"   GPU Layers: {gpu_layers} | Context: {context_size} | Threads: {threads}")
    
    t0 = time.perf_counter()
    status = runtime.load(path, max_safe_gpu_layers=gpu_layers if gpu_layers > 0 else 32, backend=backend)
    t1 = time.perf_counter()
    
    print(f"✅ [MoEBoost Loader] Model loaded in {t1 - t0:.2f}s | Active GPU Layers: {status.get('gpu_layers', 'auto')}")
    return runtime

def interactive_chat(runtime: LlamaCppRuntime, system_prompt: str = "You are a helpful and intelligent AI assistant."):
    """
    Runs interactive text-based chat loop using the loaded MoEBoost runtime.
    Displays tokens/s, prefill speed, and TTFT metrics after each response.
    """
    print("\n" + "=" * 65)
    print(" 🚀 MoEBoost Direct Python Chat")
    print(" Type 'exit' or 'quit' to quit.")
    print("=" * 65)

    messages = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("\nSession ended. Unloading model...")
                runtime.unload()
                break

            messages.append({"role": "user", "content": user_input})
            print("\n🤖 Assistant (generating...): ", end="", flush=True)

            t0 = time.perf_counter()
            result = runtime.completion(
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                stream=False
            )
            t1 = time.perf_counter()
            wall_time = max(0.001, t1 - t0)

            choices = result.get("choices") or []
            first_msg = choices[0].get("message") or {} if choices else {}
            reply = first_msg.get("content", "")
            print(reply)

            # Extract accurate engine performance metrics
            perf = result.get("performance") or runtime._read_perf_counters() or {}
            gen_tok_sec = perf.get("generation_tokens_per_second")
            prefill_tok_sec = perf.get("prefill_tokens_per_second")
            ttft = perf.get("last_time_to_first_token_formatted")

            usage = result.get("usage") or {}
            completion_tokens = usage.get("completion_tokens", 0)
            if not gen_tok_sec and completion_tokens:
                gen_tok_sec = round(completion_tokens / wall_time, 2)

            print("\n" + "-" * 65)
            print(f"⚡ [Stats] Generation: {gen_tok_sec or 'N/A'} tok/s | Prefill: {prefill_tok_sec or 'N/A'} tok/s | TTFT: {ttft or 'N/A'} | Wall: {wall_time:.2f}s")
            print("-" * 65)

            messages.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\nInterrupted. Unloading model...")
            runtime.unload()
            break
        except Exception as e:
            print(f"\n❌ Error during generation: {e}")

def main():
    if len(sys.argv) < 2:
        default_model = "/media/local-ai/Model/cerebras_Qwen3-Coder-REAP-25B-A3B-IQ4_XS.gguf"
        if os.path.exists(default_model):
            model_path = default_model
        else:
            print("Usage: python3 moeboost_model_loader.py <path_to_gguf_model> [gpu_layers]")
            sys.exit(1)
    else:
        model_path = sys.argv[1]

    gpu_layers = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    backend = os.getenv("MOE_RUNTIME_BACKEND", "cuda")

    runtime = load_moeboost_model(model_path, gpu_layers=gpu_layers, backend=backend)
    interactive_chat(runtime)

if __name__ == "__main__":
    main()
