from hardware.gpu_backends import select_backend


def test_select_backend_cuda():
    assert select_backend(True, True, False, "x") == "cuda"


def test_select_backend_cpu():
    assert select_backend(False, False, False, "x") == "cpu"
