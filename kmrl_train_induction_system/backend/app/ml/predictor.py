"""Core ML inference primitives.

Model lifecycle is owned by ``ModelRegistry`` and explainability is deliberately
separate so ordinary risk prediction stays lightweight and predictable.
"""
from __future__ import annotations

import hashlib
import logging
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from app.config import settings
from app.ml.metrics import InferenceTimer
from app.ml.model_registry import model_registry

logger = logging.getLogger(__name__)

_ml_seed_initialized = False


def _ensure_deterministic_seeding() -> None:
    global _ml_seed_initialized
    if _ml_seed_initialized:
        return
    seed = settings.ml_deterministic_seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    _ml_seed_initialized = True
    logger.info("ML inference seeded with seed=%s", seed)


async def load_latest_model(*, force_refresh: bool = False) -> Dict[str, Any]:
    """Compatibility wrapper around the centralized model registry."""
    bundle = await model_registry.get_latest(force_refresh=force_refresh)
    return {
        "model": bundle.model,
        "meta": {
            "version": bundle.version,
            "features": list(bundle.features),
            "feature_means": dict(bundle.feature_means),
        },
    }


async def batch_predict(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run deterministic risk inference without SHAP/explainability work."""
    if not features:
        return []

    _ensure_deterministic_seeding()
    sorted_features = sorted(features, key=lambda item: str(item.get("trainset_id", "")))
    input_hash = hashlib.md5(str(sorted_features).encode()).hexdigest()[:8]
    logger.debug("ML batch_predict count=%d input_hash=%s", len(features), input_hash)

    with InferenceTimer():
        bundle = await model_registry.get_latest()
        feature_cols = list(bundle.features)
        means = bundle.feature_means
        df = pd.DataFrame(features)
        centered = df.reindex(columns=feature_cols).fillna(0.0)
        for column in feature_cols:
            if column in means:
                centered[column] = centered[column] - means[column]
        values = centered.astype(np.float32).values

        bundle.model.eval()
        with torch.no_grad():
            probabilities = bundle.model(torch.from_numpy(values)).squeeze(1).numpy()

    output: List[Dict[str, Any]] = []
    for index, row in df.iterrows():
        output.append({
            "trainset_id": row.get("trainset_id"),
            "risk_prob": float(probabilities[index]),
            "model_version": bundle.version,
            "provider": "torch",
        })
    return output


def predict_maintenance_health(trainset: Dict[str, Any]) -> float:
    """Fast deterministic health heuristic used as a fallback signal."""
    try:
        health = 0.85
        cards = trainset.get("job_cards", {})
        if isinstance(cards, dict):
            critical_cards = int(cards.get("critical_cards", 0))
            open_cards = int(cards.get("open_cards", 0))
            if critical_cards > 0:
                health -= 0.5
            if open_cards > 5:
                health -= 0.1 * min(0.5, (open_cards - 5) / 10)

        current_mileage = float(trainset.get("current_mileage", 0) or 0)
        max_mileage = float(trainset.get("max_mileage_before_maintenance", 50000) or 50000)
        if max_mileage > 0:
            ratio = current_mileage / max_mileage
            if ratio > 0.9:
                health -= 0.2
            elif ratio > 0.7:
                health -= 0.1

        certificates = trainset.get("fitness_certificates", {})
        if isinstance(certificates, dict):
            expired = sum(
                1 for cert in certificates.values()
                if isinstance(cert, dict) and str(cert.get("status", "")).upper() == "EXPIRED"
            )
            health -= 0.3 if expired else 0.0

        sensor_health = trainset.get("sensor_health_score")
        if isinstance(sensor_health, (int, float)):
            health = 0.7 * float(sensor_health) + 0.3 * health

        risk = trainset.get("predicted_failure_risk")
        if isinstance(risk, (int, float)):
            health = 0.6 * health + 0.4 * (1.0 - float(risk))

        return max(0.0, min(1.0, health))
    except Exception as exc:
        logger.warning("Maintenance health heuristic failed: %s", exc)
        return 0.85
