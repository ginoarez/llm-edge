"""Resource planner (colibri-style): RAM/VRAM/disk planning."""


def plan(model_size_gb: float, ram_gb: float, vram_gb: float, disk_gb: float):
    resident = min(model_size_gb, ram_gb * 0.8)
    streamed = max(0, model_size_gb - resident)
    return {
        "model_size_gb": model_size_gb,
        "resident_gb": round(resident, 2),
        "streamed_gb": round(streamed, 2),
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
        "disk_gb": disk_gb,
    }
