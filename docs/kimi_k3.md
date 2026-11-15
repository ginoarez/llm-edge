# kimi-k3-in-c — Extracted Techniques

Source: https://github.com/FareedKhan-dev/kimi-k3-in-c

## Core idea
A 2.78T-param MoE model runs in 8.24 GB RAM by streaming experts from disk. Same output at every memory budget (byte-identical).

## Techniques

### MoE expert streaming
- Experts never resident; streamed from disk on demand.
- LRU cache + pinned hot set learned from routing history.
- Expert matrices read in one `pread`.
- Prefetch pool reads missing experts while resident ones compute.
- Router-lookahead prefetch (71.6% predictable one layer ahead).

### Quantization
- MXFP4 (~0.5 byte/param) for experts.
- Kernels with floating-point contract.

### Attention
- **KDA (Kimi Delta Attention)**: recurrent state of fixed size; memory never grows with context.
- **MLA (Multi-Head Latent Attention)**: one latent instead of 96 heads (57x smaller KV).
- Attention residuals: layers look back.

### Trunk packing
- Dense layers packed into one file; read by offset in one call.
- Turns memory requirement into a dial.

### Ultra-low-memory mode
- Stream exact embedding rows.
- Reuse one recurrent-state slot.
- Full recompute.

### Invariants
- Byte-identical output across all memory budgets (8 GB → 224 GB).
- Config reader refuses to guess (exit 2 on uncertainty).
- Exit codes for automation.

### Multi-disk
- `COLI_MODEL_MIRROR` reads from second drive.
- Two NVMe: +37.5% decode.

### KV prefix reuse
- Agent turns read only new tokens (park conversation state).

### Speculative decoding
- MTP head drafts 2.2-2.8 tokens/forward.
- Prompt lookup for code edits.
- Gate skips unprofitable drafts.

### Tuning
- `COLI_NUMA=1` for multi-socket.
- `DIRECT=1` (O_DIRECT) for fast NVMe.
- `--ultra-low-memory` for 8 GB floor.

## Modules adopted
- `streaming/offload.py` — ExpertLRUCache
- `streaming/expert_cache.py` — learned pins
- `storage/multi_disk.py` — mirror reading
- `kv/prefix.py` — KV prefix reuse
- `attention/kda.py` — fixed-size state (to add)
- `attention/mla.py` — latent attention (to add)
