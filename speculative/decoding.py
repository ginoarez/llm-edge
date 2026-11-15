"""Speculative decoding concepts (MTP, prompt lookup)."""


def estimate_speedup(acceptance_rate: float, draft_cost: float):
    if acceptance_rate <= 0 or draft_cost <= 0:
        return 1.0
    return 1.0 + (acceptance_rate / draft_cost)
