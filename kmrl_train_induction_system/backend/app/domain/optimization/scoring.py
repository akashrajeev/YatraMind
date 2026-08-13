"""Pure optimization scoring functions.

Kept side-effect free so the scoring model can be tested independently from
FastAPI, database access, ML inference, and OR-Tools.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class OptimizationWeights:
    branding: float = 300.0
    minor_defect_penalty: float = -50.0
    mileage_balance: float = 50.0
    cleaning_due_penalty: float = -30.0
    shunting_complexity_penalty: float = -20.0
    ml_health_weight: float = 100.0


def _as_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0


def tier2_score(trainset: Dict[str, Any], weights: OptimizationWeights = OptimizationWeights()) -> float:
    """Branding obligation minus minor-defect penalties."""
    score = 0.0
    branding = trainset.get("branding", {})
    if isinstance(branding, dict):
        advertiser = branding.get("current_advertiser")
        priority = str(branding.get("priority", "LOW")).upper()
        if advertiser not in (None, "", "None"):
            multiplier = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}.get(priority, 0.0)
            score += weights.branding * multiplier

    job_cards = trainset.get("job_cards", {})
    if isinstance(job_cards, dict):
        open_cards = _as_int(job_cards.get("open_cards"))
        critical_cards = _as_int(job_cards.get("critical_cards"))
        score += weights.minor_defect_penalty * max(0, open_cards - critical_cards)

    return score


def tier3_score(trainset: Dict[str, Any], weights: OptimizationWeights = OptimizationWeights()) -> float:
    """Mileage balancing, cleaning, shunting, and health contribution."""
    score = 0.0
    current_mileage = float(trainset.get("current_mileage", 0.0) or 0.0)
    km_30d = float(trainset.get("km_30d", current_mileage * 0.1) or 0.0)
    normalized = min(1.0, km_30d / 5000.0) if km_30d else 0.5
    score += weights.mileage_balance if normalized < 0.5 else weights.mileage_balance * (1.0 - normalized)

    if bool(trainset.get("requires_cleaning")):
        score += weights.cleaning_due_penalty * (0.3 if not trainset.get("cleaning_due_date") else 0.5)

    shunt_complexity = float(trainset.get("shunt_complexity", 0.0) or 0.0)
    if bool(trainset.get("is_blocked")) or shunt_complexity > 0.5:
        factor = 1.0 if bool(trainset.get("is_blocked")) else shunt_complexity
        score += weights.shunting_complexity_penalty * factor

    health = float(trainset.get("ml_health_score", 0.85) or 0.85)
    score += health * weights.ml_health_weight
    return score


def combined_score(trainset: Dict[str, Any], weights: OptimizationWeights = OptimizationWeights(), tier2_scale: float = 10_000.0) -> float:
    return tier2_scale * tier2_score(trainset, weights) + tier3_score(trainset, weights)
