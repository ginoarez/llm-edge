"""Expert cache with learned pins (colibri-style)."""

from collections import OrderedDict


class ExpertCache:
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.cache = OrderedDict()
        self.pinned = set()
        self.used = 0

    def pin(self, expert_id, size_bytes):
        self.pinned.add(expert_id)
        self.used += size_bytes

    def get(self, expert_id):
        if expert_id in self.cache:
            self.cache.move_to_end(expert_id)
            return self.cache[expert_id]
        return None

    def put(self, expert_id, data, size_bytes):
        if expert_id in self.pinned:
            return True
        if size_bytes > self.capacity:
            return False
        while self.used + size_bytes > self.capacity and self.cache:
            k, (s, _) = self.cache.popitem(last=False)
            if k in self.pinned:
                self.cache[k] = (s, _)
                continue
            self.used -= s
        self.cache[expert_id] = (size_bytes, data)
        self.used += size_bytes
        return True

    def stats(self):
        return {"entries": len(self.cache), "pinned": len(self.pinned), "used_bytes": self.used}
