"""Domain representation of an optimization decision."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable
from app.domain.optimization.types import ConstraintViolation, DecisionStatus

@dataclass(frozen=True)
class OptimizationDecision:
    trainset_id: str
    status: DecisionStatus
    score: float = 0.0
    confidence: float = 0.0
    reasons: tuple[str, ...] = field(default_factory=tuple)
    violations: tuple[ConstraintViolation, ...] = field(default_factory=tuple)

    @property
    def is_service(self) -> bool:
        return self.status is DecisionStatus.INDUCT

    @property
    def is_blocked(self) -> bool:
        return any(v.is_blocking for v in self.violations)

    @classmethod
    def from_values(
        cls,
        trainset_id: str,
        status: DecisionStatus,
        *,
        score: float = 0.0,
        confidence: float = 0.0,
        reasons: Iterable[str] = (),
        violations: Iterable[ConstraintViolation] = (),
    ) -> "OptimizationDecision":
        return cls(trainset_id, status, score, confidence, tuple(reasons), tuple(violations))
