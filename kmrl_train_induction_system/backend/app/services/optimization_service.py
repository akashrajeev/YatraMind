"""Application service for the canonical optimization workflow."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol
from app.models.trainset import OptimizationRequest, InductionDecision
from app.services.optimization_engine import CanonicalOptimizationEngine
from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.services.fleet_planning import FleetRequirementResult, compute_required_trains

class Optimizer(Protocol):
    async def optimize(self, trainsets: List[Dict[str, Any]], request: OptimizationRequest):
        ...

@dataclass(frozen=True)
class OptimizationResult:
    decisions: List[InductionDecision]
    fleet_requirement: FleetRequirementResult
    stabling_geometry: Dict[str, Any]
    eligible_train_count: int = 0

    @property
    def inducted_count(self) -> int:
        return sum(item.decision == "INDUCT" for item in self.decisions)

    @property
    def eligible_count(self) -> int:
        return self.eligible_train_count

class OptimizationService:
    """Coordinate fleet planning, induction optimization and stabling."""

    def __init__(self, optimizer: Optimizer | None = None, stabling_optimizer: StablingGeometryOptimizer | None = None) -> None:
        self.optimizer = optimizer or CanonicalOptimizationEngine()
        self.stabling_optimizer = stabling_optimizer or StablingGeometryOptimizer()

    async def optimize(self, trainsets: List[Dict[str, Any]], request: OptimizationRequest) -> OptimizationResult:
        decisions, fleet_requirement = await self.optimizer.optimize(trainsets, request)
        serialized = [d.model_dump() if hasattr(d, "model_dump") else d.dict() for d in decisions]
        stabling_geometry = await self.stabling_optimizer.optimize_stabling_geometry(trainsets, serialized)
        # Eligibility describes the input fleet available for optimization, not the
        # number of decisions the optimizer happens to return.
        return OptimizationResult(decisions, fleet_requirement, stabling_geometry, eligible_train_count=len(trainsets))

    @staticmethod
    def compute_fleet_requirement(request: OptimizationRequest) -> FleetRequirementResult:
        return compute_required_trains(
            service_date=request.service_date,
            timetable_config=None,
            override_count=request.required_service_count,
        )
