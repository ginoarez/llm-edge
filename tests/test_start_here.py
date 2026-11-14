import importlib.util
from pathlib import Path


def test_start_here_importable():
    p = Path("start_here.py")
    spec = importlib.util.spec_from_file_location("start_here", p)
    assert spec is not None
