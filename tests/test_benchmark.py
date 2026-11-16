from benchmark import summarize


def test_slo_pass_when_median_above_100():
    results = [
        {"toks": 110.0, "ttft_s": 0.2, "vram_peak_gb": 6.0},
        {"toks": 105.0, "ttft_s": 0.25, "vram_peak_gb": 6.2},
        {"toks": 99.0, "ttft_s": 0.3, "vram_peak_gb": 6.1},
    ]
    s = summarize(results)
    assert s["slo_pass"] is True
    assert s["toks_median"] == 105.0


def test_slo_fail_when_median_below_100():
    results = [{"toks": 80.0, "ttft_s": 0.4, "vram_peak_gb": 6.0} for _ in range(3)]
    s = summarize(results)
    assert s["slo_pass"] is False
