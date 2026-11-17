"""Edge0-style memory controls."""


def apply_controls(vram_reserve_mib=700, expert_cache=0, kv_resident=0, spec=4, resident_budget_gib=0, mmap_experts=False):
    return {
        "vram_reserve_mib": vram_reserve_mib,
        "expert_cache": expert_cache,
        "kv_resident": kv_resident,
        "spec": spec,
        "resident_budget_gib": resident_budget_gib,
        "mmap_experts": mmap_experts,
    }
