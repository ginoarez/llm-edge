"""KV prefix reuse (park conversation state between turns)."""


def can_reuse_prefix(prev_ids, new_ids):
    """Return number of reusable prefix tokens."""
    n = 0
    for a, b in zip(prev_ids, new_ids):
        if a == b:
            n += 1
        else:
            break
    return n
