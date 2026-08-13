"""Non-critical explainability service.

Prediction/optimization does not need SHAP or narrative generation to produce
a safe decision. This service keeps those expensive operations off the hot
path and allows callers to request explanations after a decision exists.
"""
from __future__ import annotations
from typing import Any, Mapping


class ExplainabilityService:
    def __init__(self, renderer=None):
        self.renderer = renderer

    async def explain(self, trainset: Mapping[str, Any], decision: str) -> dict[str, Any]:
        if self.renderer is not None:
            return await self.renderer(trainset, decision)

        reasons = []
        if decision == "MAINTENANCE":
            reasons.append("Trainset requires maintenance or failed a safety constraint.")
        elif decision == "INDUCT":
            reasons.append("Trainset was selected by the canonical optimization ranking after safety filtering.")
        else:
            reasons.append("Trainset remained eligible but ranked below the induction target.")

        return {
            "trainset_id": trainset.get("trainset_id"),
            "decision": decision,
            "reasons": reasons,
            "explainability_available": False,
        }
