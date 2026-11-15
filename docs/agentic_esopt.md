# Agentic ESOpt — Extracted Techniques

Source: https://github.com/zz1358m/Agentic-ESOpt

## Core idea
Full-parameter, backpropagation-free fine-tuning of long-horizon LLM agents using evolution strategies (ES). Samples perturbations, evaluates full trajectories with scalar rewards, applies reward-weighted update. Inference-level memory only.

## Techniques

### Evolution Strategies
- Sample G seeded full-parameter perturbations.
- Evaluate complete agent trajectories → scalar reward.
- Normalize rewards (z-score), weight perturbations.
- Cosine schedule for perturbation scale.
- Atomic history.json (resume support).

### Memory efficiency
- No activations or optimizer states stored.
- Only inference-level GPU memory.

### Long-horizon scalability
- Trajectory-level attribution (no turn-by-turn credit).
- Better than Agentic RL as horizon grows.

### Flexibility
- Scalar-reward interface composes with skill evolution (Trace2Skill) and test-time search (EoH).

### Seeded replayability
- Same seeds reproduce optimization history on fresh model server.

### Trace2Skill
- Distill skills from completed trajectories.

### Checkpoints
- Released on Hugging Face (Qwen3.5-4B/9B/27B variants).

## Modules adopted
- `adaptation/es_adapter.py` — EvolutionStrategyAdapter
- `experiments/es_recovery.py` — quality recovery harness
