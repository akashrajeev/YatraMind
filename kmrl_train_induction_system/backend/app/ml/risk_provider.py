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
    async def predict(self, trainsets: Sequence[Mapping[str, Any]]) -> list[RiskPrediction]:
        ...


class LegacyPredictorRiskProvider:
    """Adapter around the current predictor so optimization no longer imports it directly."""

    async def predict(self, trainsets: Sequence[Mapping[str, Any]]) -> list[RiskPrediction]:
        from app.ml.predictor import batch_predict, predict_maintenance_health

        features = []
        for trainset in trainsets:
            features.append({
                "trainset_id": trainset.get("trainset_id"),
                **{key: value for key, value in trainset.items() if isinstance(value, (int, float))},
            })

        predictions = await batch_predict(features)
        prediction_map = {item["trainset_id"]: item for item in predictions}
        result: list[RiskPrediction] = []
        for trainset in trainsets:
            trainset_id = str(trainset.get("trainset_id", "UNKNOWN"))
            raw = prediction_map.get(trainset_id, {})
            risk = float(raw.get("risk_prob", trainset.get("predicted_failure_risk", 0.2)) or 0.2)
            try:
                health = float(predict_maintenance_health(dict(trainset)))
            except Exception:
                health = float(trainset.get("ml_health_score", 0.85) or 0.85)
            top_features = tuple(str(x) for x in raw.get("top_features", [])[:5])
            result.append(RiskPrediction(
                trainset_id=trainset_id,
                risk_probability=risk,
                health_score=health,
                top_features=top_features,
                provider="legacy-predictor",
            ))
        return result


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
            cards = trainset.get("job_cards") or {}
            critical_cards = int(float(str(cards.get("critical_cards", 0)).strip() or 0)) if isinstance(cards, Mapping) else 0
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
