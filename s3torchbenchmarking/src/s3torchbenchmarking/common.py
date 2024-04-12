import json
from typing import Optional

import numpy as np


class Distribution:
    def __init__(self, initial_capacity: int = 1024, precision: int = 4):
        self.initial_capacity = initial_capacity
        self._values = np.zeros(shape=initial_capacity, dtype=np.float32)
        self._idx = 0
        self.precision = precision

    def _expand_if_needed(self):
        if self._idx > self._values.size - 1:
            self._values = np.concatenate(
                self._values, np.zeros(self.initial_capacity, dtype=np.float32)
            )

    def add(self, val: float):
        self._expand_if_needed()
        self._values[self._idx] = val
        self._idx += 1

    def is_empty(self) -> bool:
        return self._idx == 0

    def summarize(self) -> Optional[dict]:
        window = self._values[: self._idx]
        if window.size == 0:
            return
        return {
            "n": window.size,
            "mean": round(float(window.mean()), self.precision),
            "min": round(np.percentile(window, 0), self.precision),
            "p50": round(np.percentile(window, 50), self.precision),
            "p75": round(np.percentile(window, 75), self.precision),
            "p90": round(np.percentile(window, 90), self.precision),
            "max": round(np.percentile(window, 100), self.precision),
        }

    def __repr__(self):
        summary_str = json.dumps(self.summarize())
        return "Distribution({0})".format(summary_str)
