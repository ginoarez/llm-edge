from streaming.expert_cache import ExpertCache
from kv.prefix import can_reuse_prefix
from resource_plan import plan
from speculative.decoding import estimate_speedup


def test_expert_cache():
    c = ExpertCache(100)
    c.pin("e1", 40)
    assert c.stats()["pinned"] == 1


def test_prefix_reuse():
    assert can_reuse_prefix([1, 2, 3], [1, 2, 4]) == 2


def test_plan():
    p = plan(5.0, 16.0, 8.0, 100.0)
    assert p["resident_gb"] + p["streamed_gb"] == 5.0


def test_spec_speedup():
    assert estimate_speedup(0.5, 0.2) > 1.0
