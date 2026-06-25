"""Minimal in-process metrics registry (control-plane only).

Counters and gauges plus simple timing. Swap for Prometheus/OTel in Phase 4.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class MetricsRegistry:
    def __init__(self):
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}

    def incr(self, name: str, value: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0.0) + value

    def gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value

    @contextmanager
    def time(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.gauge(name, time.perf_counter() - start)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {"counters": dict(self._counters), "gauges": dict(self._gauges)}
