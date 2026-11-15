"""Multi-disk mirror reading (colibri-style)."""


def choose_disk(primary_path: str, mirror_path: str | None, primary_load: float, mirror_load: float):
    """Pick the less loaded disk for reading."""
    if mirror_path and mirror_load < primary_load:
        return mirror_path
    return primary_path
