#!/usr/bin/env python3
"""Benchmark with real TTFT (streaming) and VRAM measurement."""

import argparse
import json
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    "Explica en detalle qué es speculative decoding y cuándo acelera la inferencia local.",
    "Escribe una función Python que calcule Fibonacci de forma iterativa y explica su complejidad.",
    "Resume en 5 puntos los trade-offs entre cuantización Q3 y Q4 para LLMs en 8GB.",
]


def vram_sample(stop):
    vals = []
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            )
            vals.append(float(out.strip().splitlines()[0]) / 1024.0)
        except Exception:
            pass
        stop.wait(0.5)
    return vals


def run_ollama_stream(host, model, prompt, num_predict, num_ctx):
    payload = {
        "model": model, "prompt": prompt, "stream": True,
        "options": {"num_predict": num_predict, "num_ctx": num_ctx, "temperature": 0.0},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=data, headers={"Content-Type": "application/json"})
    stop = threading.Event()
    samples = []
    t = threading.Thread(target=lambda: samples.extend(vram_sample(stop)), daemon=True)
    t.start()
    t0 = time.perf_counter()
    ttft = None
    eval_count = 0
    eval_duration_ns = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if ttft is None and obj.get("response"):
                ttft = time.perf_counter() - t0
            if obj.get("done"):
                eval_count = obj.get("eval_count", 0)
                eval_duration_ns = obj.get("eval_duration", 0)
    stop.set()
    t.join(timeout=2)
    wall = time.perf_counter() - t0
    toks = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns else 0.0
    vram_peak = max(samples) if samples else None
    return {"toks": round(toks, 2), "wall_s": round(wall, 3), "ttft_s": round(ttft, 3) if ttft else None,
            "eval_count": eval_count, "vram_peak_gb": round(vram_peak, 3) if vram_peak else None}


def run_llamacpp_stream(host, model, prompt, num_predict, num_ctx):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": num_predict,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{host}/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
    stop = threading.Event()
    samples = []
    t = threading.Thread(target=lambda: samples.extend(vram_sample(stop)), daemon=True)
    t.start()
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line or not line.startswith("data:"):
                continue
            if line == "data: [DONE]":
                break
            try:
                obj = json.loads(line[5:])
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    completion_tokens += 1
            except Exception:
                pass
    stop.set()
    t.join(timeout=2)
    wall = time.perf_counter() - t0
    toks = completion_tokens / wall if wall > 0 else 0.0
    vram_peak = max(samples) if samples else None
    return {"toks": round(toks, 2), "wall_s": round(wall, 3), "ttft_s": round(ttft, 3) if ttft else None,
            "eval_count": completion_tokens, "vram_peak_gb": round(vram_peak, 3) if vram_peak else None}


def summarize(runs, slo_toks=100, slo_ttft=0.5, slo_vram=7.5):
    toks = [r["toks"] for r in runs]
    ttfts = [r["ttft_s"] for r in runs if r["ttft_s"] is not None]
    vrams = [r["vram_peak_gb"] for r in runs if r["vram_peak_gb"] is not None]
    median_toks = statistics.median(toks)
    median_ttft = statistics.median(ttfts) if ttfts else None
    max_vram = max(vrams) if vrams else None
    slo_pass = (median_toks >= slo_toks and
                (median_ttft is None or median_ttft <= slo_ttft) and
                (max_vram is None or max_vram <= slo_vram))
    return {
        "toks_median": round(median_toks, 2),
        "toks_min": round(min(toks), 2),
        "toks_max": round(max(toks), 2),
        "ttft_median_s": round(median_ttft, 3) if median_ttft is not None else None,
        "vram_peak_gb": round(max_vram, 3) if max_vram is not None else None,
        "slo_pass": slo_pass,
        "slo_checks": {
            "toks": median_toks >= slo_toks,
            "ttft": (median_ttft is None or median_ttft <= slo_ttft),
            "vram": (max_vram is None or max_vram <= slo_vram),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="ollama", choices=["ollama", "llamacpp"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prompts", nargs="*", default=None)
    args = ap.parse_args()

    host = args.host or ("http://127.0.0.1:11434" if args.backend == "ollama" else "http://127.0.0.1:8080")
    prompts = args.prompts or ([args.prompt] if args.prompt else PROMPTS)

    all_results = []
    for prompt in prompts:
        runs = []
        for _ in range(args.runs):
            if args.backend == "ollama":
                r = run_ollama_stream(host, args.model, prompt, args.num_predict, args.num_ctx)
            else:
                r = run_llamacpp_stream(host, args.model, prompt, args.num_predict, args.num_ctx)
            r["prompt"] = prompt[:60]
            runs.append(r)
        agg = summarize(runs)
        agg["prompt"] = prompt[:60]
        agg["details"] = runs
        all_results.append(agg)
        print(f"{args.backend} | {prompt[:40]}... | toks={agg['toks_median']} | ttft={agg['ttft_median_s']} | vram={agg['vram_peak_gb']} | slo={agg['slo_pass']}")

    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
