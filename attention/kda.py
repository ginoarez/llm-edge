"""Kimi Delta Attention: fixed-size recurrent state (memory never grows)."""


class KimiDeltaState:
    def __init__(self, state_size: int):
        self.state_size = state_size
        self.state = [0.0] * state_size

    def update(self, x):
        # conceptual: write x into fixed-size state
        for i in range(min(len(x), self.state_size)):
            self.state[i] = x[i]
        return list(self.state)

    def memory_bytes(self):
        return self.state_size * 4  # float32
