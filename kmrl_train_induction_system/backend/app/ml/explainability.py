"""Optional ML explainability operations kept off the inference hot path."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Sequence

import numpy as np
import shap
import torch

logger = logging.getLogger(__name__)


async def top_features(
    model: Any,
    values: np.ndarray,
    feature_names: Sequence[str],
    limit: int = 3,
) -> list[list[str]]:
    """Compute SHAP-based feature rankings only when explicitly requested."""
    return await asyncio.to_thread(
        _calculate_top_features,
        model,
        values,
        list(feature_names),
        limit,
    )


def _calculate_top_features(
    model: Any,
    values: np.ndarray,
    feature_names: list[str],
    limit: int,
) -> list[list[str]]:
    try:
        def predict_fn(data: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                tensor = torch.from_numpy(data.astype(np.float32))
                return model(tensor).detach().numpy()

        explainer = shap.Explainer(predict_fn, feature_names=feature_names)
        shap_values = explainer(values, max_evals=100)
        contributions = np.abs(shap_values.values)
        indexes = np.argsort(-contributions, axis=1)[:, :limit]
        return [[feature_names[index] for index in row] for row in indexes]
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return [[] for _ in range(len(values))]
