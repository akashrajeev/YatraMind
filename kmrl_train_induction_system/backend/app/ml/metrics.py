"""Lightweight metrics for ML inference health."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    calls: int = 0
    failures: int = 0
    total_latency_seconds: float = 0.0

    def record(self, latency_seconds: float, failed: bool = False) -> None:
        self.calls += 1
        self.total_latency_seconds += latency_seconds
        if failed:
            self.failures += 1

    @property
    def average_latency_seconds(self) -> float:
        return self.total_latency_seconds / self.calls if self.calls else 0.0


metrics = InferenceMetrics()


class InferenceTimer:
    def __enter__(self) -> "InferenceTimer":
        self.started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        latency = time.perf_counter() - self.started_at
        metrics.record(latency, failed=exc_type is not None)
        logger.info(
            "ML inference completed latency=%.4fs failed=%s calls=%d",
            latency,
            exc_type is not None,
            metrics.calls,
        )
