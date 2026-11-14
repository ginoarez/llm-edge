#!/usr/bin/env python3
"""Hardware-aware setup wizard (inspired by colibri)."""

import argparse
import json
from pathlib import Path

import yaml

from hardware.profiler import get_profile
from hardware.gpu_backends import select_backend
from recommender.recommend import load_catalog, recommend


def detect():
    p = get_profile()
    # naive GPU detection
    has_cuda = p["gpu_name"] is not None and "NVIDIA" in (p["gpu_name"] or "")
    has_vulkan = False
    integrated = "Integrated" in (p["gpu_name"] or "") or "Iris" in (p["gpu_name"] or "")
    backend = select_backend(has_cuda, has_vulkan, integrated, "default")
    return p, backend


def list_models(profile, backend):
    catalog = load_catalog()
    vram = profile.get("vram_gb") or 0
    ram = profile.get("ram_gb") or 8
    print(f"Hardware: {profile.get('gpu_name')} | VRAM {vram} GB | RAM {ram} GB | backend {backend}")
    print("\nAvailable models:")
    for m in catalog:
        fits = m["min_vram_gb"] <= vram if vram else m["min_vram_gb"] <= ram
        status = "OK" if fits else "NO FIT"
        print(f"  {m['id']:30s} min_vram={m['min_vram_gb']:>4}GB  quality={m['quality_score']:>3}  [{status}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()

    profile, backend = detect()

    if args.list:
        list_models(profile, backend)
        return

    vram = profile.get("vram_gb") or (profile.get("ram_gb") or 8)
    if args.model:
        catalog = load_catalog()
        chosen = next((m for m in catalog if m["id"] == args.model), None)
        if not chosen:
            print(f"Model {args.model} not found in catalog.")
            return
    else:
        rec = recommend(vram, "quality")
        chosen = next((m for m in load_catalog() if m["id"] == rec["model"]), None)

    if not chosen:
        print("No suitable model found.")
        return

    print(f"Recommended model: {chosen['id']} ({chosen['quant']}, min {chosen['min_vram_gb']} GB)")
    print(f"Backend: {backend}")
    print(f"Pull command: ollama pull {chosen['id']}")
    if args.dir:
        print(f"Model dir: {args.dir}")


if __name__ == "__main__":
    main()
