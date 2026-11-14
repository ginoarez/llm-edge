"""GPU backend selection (inspired by colibri)."""


def select_backend(has_cuda: bool, has_vulkan: bool, integrated: bool, model: str):
    if has_cuda:
        return "cuda"
    if has_vulkan and not integrated:
        return "vulkan"
    if has_vulkan and integrated and model in {"qwen3.6-35b-a3b", "qwen3-coder-30b"}:
        return "vulkan"
    return "cpu"
