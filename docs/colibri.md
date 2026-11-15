# colibri — Extracted Techniques for llm-edge

Source: https://github.com/JustVugg/colibri (51k stars)

## Core idea
A MoE model is huge on disk but small per token. Keep the dense part in RAM, stream routed experts from disk on demand. Same model, same output, any memory budget.

## Techniques to adopt

### Expert streaming & caching
- LRU cache for routed experts, plus a **pinned hot set** learned from conversation history (`.coli_usage`).
- Experts read in one `pread` (all matrices at once).
- Prefetch pool reads missing experts while resident ones compute.
- Router-lookahead prefetch (GLM-5.2 routing 71.6% predictable one layer ahead).
- O_DIRECT (DIRECT=1) often wins on fast NVMe.

### Multi-disk
- `COLI_MODEL_MIRROR=/second/drive` reads from a copy on a second drive.
- Two NVMe drives on independent controllers: +37.5% decode.
- Partial mirror on smaller drive works.

### GPU backends
- CPU (default), Vulkan (any GPU), CUDA (NVIDIA).
- Expert tier: cache of routed experts in GPU memory.
- Dense chain: whole layer as one GPU submission.
- Integrated GPU: only beneficial for certain models (Qwen3.6, etc.).

### Speculative decoding
- MTP (Multi-Token Prediction) head drafts 2.2-2.8 tokens/forward.
- Prompt lookup: 6-7% on code edits.
- Gate skips drafts that cannot pay.
- Qwen3.6 MTP: 99.7→115.8 tok/s on CUDA.

### KV cache
- Float16 KV caches: 57x smaller for GLM-5.2 MLA (576 floats vs 32768).
- KV prefix reuse: agent turns read only new tokens (Hermes Agent 86s→4.9s).

### Long context
- 256k context at speed of 64k (Qwen3.8-Flash-Next).
- MLA compressed KV state survives restarts.

### GGUF import
- `./coli import` reads published GGUFs as-is (IQ1_S to BF16).
- No conversion needed.

### Hardware detection & recommendation
- Setup wizard detects RAM, disk, CPU, GPUs.
- Recommends model based on fit.
- MCP server for AI assistants.

### Resource planner
- `coli plan` / `coli doctor`: RAM/VRAM planning.
- `coli tune`: measure and save fastest safe settings.

### Benchmarks
- End-to-end measurements on real machines.
- Quality measured alongside speed.
- Benchmark protocol for reproducibility.

### Research directions
- Routing history for expert placement.
- Multi-SSD bandwidth aggregation.
- Hardware-aware planner.
- Lossless/quality-bounded representations.
- Routing-aware speculation.
- CPU/GPU overlap.

## Modules to adopt in llm-edge
- `streaming/expert_cache.py` — learned pins + LRU
- `storage/multi_disk.py` — mirror reading
- `kv/prefix.py` — KV prefix reuse
- `speculative/` — MTP/prompt lookup
- `resource_plan.py` — RAM/VRAM planner
- `mcp_server.py` — MCP server stub
- `gguf/importer.py` — GGUF import concept
- `docs/quickstart.md` — setup guide
