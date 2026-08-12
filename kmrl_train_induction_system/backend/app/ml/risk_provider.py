"""Stable risk-inference boundary for optimization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RiskPrediction:
    trainset_id: str
    risk_probability: float
    health_score: float
    top_features: tuple[str, ...] = field(default_factory=tuple)
    provider: str = "unknown"
    model_version: str | None = None


class RiskProvider(Protocol):
    """Provider contract consumed by optimization, independent of model implementation."""

    async def predict(self, trainsets: Sequence[Mapping[str, Any]]) -> list[RiskPrediction]:
        ...


class HeuristicRiskProvider:
    """Deterministic fallback used when a trained model is unavailable."""

    def __init__(self, provider_name: str = "heuristic-v1") -> None:
        self.provider_name = provider_name

    async def predict(self, trainsets: Sequence[Mapping[str, Any]]) -> list[RiskPrediction]:
        predictions: list[RiskPrediction] = []
        for trainset in trainsets:
            trainset_id = str(trainset.get("trainset_id", "UNKNOWN"))
            mileage = float(trainset.get("current_mileage", 0.0) or 0.0)
            max_mileage = float(trainset.get("max_mileage_before_maintenance", 50000.0) or 50000.0)
            ratio = min(1.0, max(0.0, mileage / max_mileage)) if max_mileage > 0 else 1.0
            critical_cards = int(float(str((trainset.get("job_cards") or {}).get("critical_cards", 0)).strip() or 0))
            risk = min(1.0, 0.1 + (ratio * 0.6) + (0.2 if critical_cards else 0.0))
            predictions.append(
                RiskPrediction(
                    trainset_id=trainset_id,
                    risk_probability=risk,
                    health_score=1.0 - risk,
                    top_features=("mileage", "critical_job_cards") if critical_cards else ("mileage",),
                    provider=self.provider_name,
                    model_version="1",
                )
            )
        return predictions
