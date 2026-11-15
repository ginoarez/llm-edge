"""Multi-Head Latent Attention: one latent instead of N heads."""


def kv_memory_ratio(num_heads: int, latent_dim: int, head_dim: int):
    full = num_heads * head_dim
    return round(latent_dim / full, 4)
