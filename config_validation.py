"""Config validation (Soup-inspired): reject unknown keys."""

KNOWN_KEYS = {"model", "quant", "context", "temperature", "num_predict"}


def validate_config(cfg: dict):
    unknown = set(cfg.keys()) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return True
