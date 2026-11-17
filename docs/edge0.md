# Edge0 — Extracted Techniques

Source: https://github.com/Edge0-AI/Edge0

## Core idea
Open-source streaming MoE inference framework. Runs 145B/120B latent-MoE on consumer hardware via SSD streaming and trained routing prediction.

## Techniques

### Latent MoE
- Compresses expert inputs (latent width 1792 for 145B, 1280 for 120B).
- Attention stays full width.
- Routes 10 of 512 experts + 1 shared expert.

### Trained routing prediction
- Router predicts next-layer experts for prefetch.
- Paper: arXiv 2609.18063 ("The Other Half of the Memory Wall").

### Memory controls
- `--vram-reserve-mib N`: VRAM left free.
- `--expert-cache N|auto`: expert slots in VRAM.
- `--kv-resident N`: KV cells in VRAM, rest streams from host RAM.
- `--spec N`: MTP verify window.
- `--resident-budget-gib N`: cap on host-RAM expert cache.
- `--mmap-experts`: read experts via mmap (nothing resident).

### Expert residency
- Disjoint VRAM/RAM expert sets; nothing counted twice.
- Residency readings: Q3_K 27.98 GiB → 22.93 VRAM + 18.40 RAM.

### MTP speculation
- Verify window (default 4 tokens).
- ~180 MiB at window 4.

### Quality
- 145B Q3_K: 121.2 tok/s agent decode on RTX 5090 D 24GB.
- 3554.9 tok/s 16K prefill.

### Benchmarks
- Copyable speed data for multiple machines.
- Memory residency measurements.

### Getting started
- edge0-infer: shared serve/chat, selects engine from model folder.
- Platforms: Linux/Windows (CUDA), macOS (Metal), iOS/Android (35B native).

## Modules adopted
- `streaming/trained_routing.py` — predict next experts
- `hardware/memory_controls.py` — Edge0-style controls
- `docs/edge0.md` — this file
