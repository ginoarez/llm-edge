from streaming.trained_routing import predict_next_experts
from hardware.memory_controls import apply_controls


def test_trained_routing():
    r = predict_next_experts([0, 1, 2], 8, 2)
    assert len(r) == 2


def test_memory_controls():
    c = apply_controls()
    assert c["spec"] == 4
    assert c["mmap_experts"] is False
