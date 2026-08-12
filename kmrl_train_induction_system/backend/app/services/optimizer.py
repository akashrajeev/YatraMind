"""Compatibility facade for the canonical optimization engine."""
from __future__ import annotations

import math
from typing import Any, Dict, List

from app.models.trainset import InductionDecision, OptimizationRequest
from app.services.optimization_engine import CanonicalOptimizationEngine
from app.services.fleet_planning import FleetRequirementResult


def compute_trains_needed(required_hours: float, trainsets: List[Dict[str, Any]]) -> int:
    """Legacy hours-to-train helper preserved for older callers/tests."""
    if not trainsets:
        return 0
    if required_hours <= 0:
        return 1
    estimates = []
    for trainset in trainsets:
        value = trainset.get("estimated_service_hours")
        try:
            if value is not None and float(value) > 0:
                estimates.append(float(value))
        except (TypeError, ValueError):
            continue
    average_hours = sum(estimates) / len(estimates) if estimates else 2.0
    return max(1, min(len(trainsets), math.ceil(float(required_hours) / average_hours)))


class TrainInductionOptimizer:
    """Backward-compatible entrypoint delegating to the canonical engine."""

    def __init__(self, engine: CanonicalOptimizationEngine | None = None) -> None:
        self.engine = engine or CanonicalOptimizationEngine()

    async def optimize(
        self,
        trainsets: List[Dict[str, Any]],
        request: OptimizationRequest,
    ) -> tuple[List[InductionDecision], FleetRequirementResult]:
        return await self.engine.optimize(trainsets, request)


__all__ = ["TrainInductionOptimizer", "compute_trains_needed"]
