# Soup — Extracted Techniques

Source: https://github.com/MakazhanAlpamys/Soup

## Core idea
Fine-tune LLMs from one YAML. Layer streaming trains an 8B model on a 4 GB GPU.

## Techniques

### Exact layer streaming
- Stream frozen base from host RAM one decoder layer at a time.
- Bit-exact vs resident (forward + backward verified separately).
- VRAM peak: 3.32 GB (Llama-3.1-8B NF4, pre-repair), 2.40 GB post-repair.
- Host-to-device transfer is NOT the bottleneck (only 1.4% speedup if removed).
- Largest cost: per-layer NF4 dequantization (9.8%).

### Config validation
- Unknown config keys rejected (v0.75+).
- Single source of truth: config/schema.py.
- `soup doctor` checks GPU/deps/environment.

### Hardware-fit preflight
- `soup profile` answers "will it fit?" without running.
- Prices run from model config.json shaped by seq/batch/method/lora/optimizer.

### Auto-tuning
- Auto batch size, GPU detection, quantization.
- `soup autopilot` zero-config training.

### Export
- GGUF export (`--format gguf --quant q4_k_m`).
- ONNX, TensorRT, BitNet.

### Speculative decoding
- Train + measure your own draft.

### Eval gate
- Eval-gated training, benchmarks, NLG metrics.

### Data engineering
- 100+ recipes, data tools, synthetic generation.

### Compliance
- HIPAA/SOC2/EU-AI-Act templates, provenance, audit.

## Modules adopted
- `streaming/layer_stream.py` — LayerStreamer
- `hardware/fit.py` — will_it_fit
- `hardware/memory_planner.py` — plan
- `recommender/auto_tune.py` — pick_best
