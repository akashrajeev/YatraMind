"""Compatibility facade for the canonical optimization engine.

The former implementation mixed safety rules, scoring, ML, solver setup and
result construction in one 100KB+ module. Those responsibilities now live in
separate domain/application modules. Existing imports remain valid through
this thin facade while callers migrate to ``OptimizationService``.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.models.trainset import InductionDecision, OptimizationRequest
from app.services.optimization_engine import CanonicalOptimizationEngine
from app.services.fleet_planning import FleetRequirementResult


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


__all__ = ["TrainInductionOptimizer"]
