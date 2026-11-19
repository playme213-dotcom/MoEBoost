#!/usr/bin/env python3
"""
MoEBoost Text-based Interactive Chat Client with Tok/s Telemetry.
Communicates with MoEBoost (OpenAI-compatible server) and displays real-time:
- Prefill tokens/sec
- Generation tokens/sec
- Time To First Token (TTFT)
- VRAM / RAM consumption & MoE active experts tier routing
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error
import argparse

DEFAULT_URL = "http://127.0.0.1:4133"

def query_telemetry(base_url: str):
    try:
        req = urllib.request.Request(f"{base_url}/api/telemetry", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

def query_models(base_url: str):
    try:
        req = urllib.request.Request(f"{base_url}/v1/models", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("data", [])
    except Exception:
        return []

def chat_completion(base_url: str, messages: list[dict], model: str = None, max_tokens: int = 512, temperature: float = 0.7):
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    if model:
        payload["model"] = model

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            res = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {err_msg}") from e
    t1 = time.perf_counter()
    wall_time = max(0.001, t1 - t0)
    
    # Extract reply
    choices = res.get("choices") or []
    reply = choices[0].get("message", {}).get("content", "") if choices else ""
    usage = res.get("usage") or {}
    
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    
    # Query server telemetry for exact internal kernel timings
    telemetry = query_telemetry(base_url)
    perf = (telemetry or {}).get("performance", {})
    gen_tok_sec = perf.get("generation_tokens_per_second")
    prefill_tok_sec = perf.get("prefill_tokens_per_second")
    ttft = perf.get("last_time_to_first_token_formatted")
    
    if not gen_tok_sec and completion_tokens:
        gen_tok_sec = round(completion_tokens / wall_time, 2)
        
    return {
        "reply": reply,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "wall_time": wall_time,
        "gen_tok_sec": gen_tok_sec,
        "prefill_tok_sec": prefill_tok_sec,
        "ttft": ttft,
        "telemetry": telemetry
    }

def main():
    parser = argparse.ArgumentParser(description="MoEBoost Text-Based Interactive Chat & Telemetry Monitor")
    parser.add_argument("url", nargs="?", default=DEFAULT_URL, help="Base URL of MoEBoost server")
    parser.add_argument("--model", default=None, help="Model ID or filename to use for inference")
    args = parser.parse_args()

    base_url = args.url
    model_id = args.model or os.environ.get("MOE_MODEL_FILE")

    print("=" * 65)
    print(" 🚀 MoEBoost Text-Based Interactive Chat & Telemetry Monitor")
    print(f" 🌐 Target Endpoint: {base_url}")
    print(" Type 'exit' or 'quit' to end session.")
    print("=" * 65)

    # Test connectivity
    telemetry = query_telemetry(base_url)
    if not telemetry:
        print(f"⚠️ Warning: MoEBoost server not detected at {base_url}.")
        print("Please ensure the MoEBoost server is running before sending prompts.")
    else:
        runtime = telemetry.get("runtime", {})
        backend = runtime.get("backend", "cuda")
        gpu_layers = runtime.get("gpu_layers", 0)
        active_model = runtime.get("model_path")
        print(f"✅ MoEBoost Online | Backend: {backend.upper()} | GPU Layers: {gpu_layers}")
        if active_model:
            print(f"📦 Active Model: {os.path.basename(active_model)}")

    # Auto-detect model if not provided
    if not model_id:
        models = query_models(base_url)
        if models:
            model_id = models[0].get("id")
            print(f"💡 Auto-selected Model: {model_id}")

    messages = [
        {"role": "system", "content": "You are a helpful and intelligent AI assistant powered by MoEBoost."}
    ]

    while True:
        try:
            user_input = input("\n👤 User: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("\nSession ended. Goodbye!")
                break

            messages.append({"role": "user", "content": user_input})
            print("\n🤖 Assistant (generating...): ", end="", flush=True)

            result = chat_completion(base_url, messages, model=model_id)
            reply = result["reply"]
            print(reply)

            # Telemetry footer
            gen_speed = f"{result['gen_tok_sec']} tok/s" if result['gen_tok_sec'] else "N/A"
            prefill_speed = f"{result['prefill_tok_sec']} tok/s" if result['prefill_tok_sec'] else "N/A"
            ttft_val = result['ttft'] or "N/A"

            print("\n" + "-" * 65)
            print(f"⚡ [Stats] Generation: {gen_speed} | Prefill: {prefill_speed} | TTFT: {ttft_val} | Wall: {result['wall_time']:.2f}s")
            
            # Show workstation memory status if available
            tel = result.get("telemetry")
            if tel:
                gpu_list = tel.get("memory", {}).get("gpu") or []
                gpu_str = f"GPU: {gpu_list[0].get('vram_used_mb', 0)}MB / {gpu_list[0].get('vram_total_mb', 0)}MB" if gpu_list else ""
                ram_str = f"RAM: {tel.get('memory', {}).get('ram', {}).get('used_formatted', '')}"
                print(f"💾 [Hardware] {gpu_str} | {ram_str}")
            print("-" * 65)

            messages.append({"role": "assistant", "content": reply})

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\n❌ Request error: {e}")

if __name__ == "__main__":
    main()
