"""Canonical induction optimization engine.

This module is the replacement path for the legacy monolithic optimizer. It
keeps the existing tier hierarchy and OR-Tools formulation, while delegating
safety, scoring, trainset modeling, and risk inference to separate modules.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from ortools.linear_solver import pywraplp

from app.domain.optimization.constraints import validate_trainset_safety
from app.domain.optimization.scoring import OptimizationWeights, combined_score, tier2_score, tier3_score
from app.models.trainset import InductionDecision, OptimizationRequest, OptimizationWeights as RequestWeights
from app.services.fleet_planning import FleetRequirementResult, compute_required_trains
from app.ml.risk_provider import HeuristicRiskProvider, LegacyPredictorRiskProvider, RiskProvider

logger = logging.getLogger(__name__)


class CanonicalOptimizationEngine:
    """Run a deterministic, structured induction optimization."""

    def __init__(self, risk_provider: RiskProvider | None = None) -> None:
        self.risk_provider = risk_provider or LegacyPredictorRiskProvider()

    @staticmethod
    def _effective_weights(request: OptimizationRequest) -> OptimizationWeights:
        if not request.weights:
            return OptimizationWeights()
        base = RequestWeights()
        current = OptimizationWeights()

        def scale(value: float, override: float, default: float) -> float:
            return value if default <= 0 else value * max(0.0, override) / default

        return OptimizationWeights(
            branding=scale(current.branding, request.weights.branding, base.branding),
            minor_defect_penalty=scale(current.minor_defect_penalty, request.weights.readiness, base.readiness),
            mileage_balance=scale(current.mileage_balance, request.weights.mileage_balance, base.mileage_balance),
            cleaning_due_penalty=scale(current.cleaning_due_penalty, request.weights.reliability, base.reliability),
            shunting_complexity_penalty=scale(current.shunting_complexity_penalty, request.weights.shunt, base.shunt),
            ml_health_weight=current.ml_health_weight,
        )

    @staticmethod
    def _normalize(trainset: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(trainset)
        cards = data.get("job_cards")
        if not isinstance(cards, dict):
            cards = {}
        data["job_cards"] = {
            "open_cards": CanonicalOptimizationEngine._as_int(cards.get("open_cards")),
            "critical_cards": CanonicalOptimizationEngine._as_int(cards.get("critical_cards")),
        }
        try:
            data["current_mileage"] = float(data.get("current_mileage", 0.0) or 0.0)
        except (TypeError, ValueError):
            data["current_mileage"] = 0.0
        maximum = data.get("max_mileage_before_maintenance")
        try:
            data["max_mileage_before_maintenance"] = float(maximum) if maximum not in (None, "", 0) else float("inf")
        except (TypeError, ValueError):
            data["max_mileage_before_maintenance"] = float("inf")
        return data

    @staticmethod
    def _as_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _needs_maintenance(trainset: Dict[str, Any]) -> bool:
        cards = trainset.get("job_cards") or {}
        critical = CanonicalOptimizationEngine._as_int(cards.get("critical_cards")) if isinstance(cards, dict) else 0
        mileage = float(trainset.get("current_mileage", 0.0) or 0.0)
        maximum = float(trainset.get("max_mileage_before_maintenance", float("inf")) or float("inf"))
        return critical > 0 or (maximum != float("inf") and mileage >= maximum * 0.95)

    @staticmethod
    def _reasons(trainset: Dict[str, Any]) -> List[str]:
        reasons: List[str] = []
        branding = trainset.get("branding") or {}
        if isinstance(branding, dict) and branding.get("current_advertiser") not in (None, "", "None"):
            priority = str(branding.get("priority", "LOW")).upper()
            reasons.append({
                "HIGH": "High branding obligation - wrapped train must enter service",
                "MEDIUM": "Medium branding obligation - wrapped train",
                "LOW": "Branding obligation - wrapped train",
            }.get(priority, "Branding obligation - wrapped train"))
        cards = trainset.get("job_cards") or {}
        if isinstance(cards, dict) and CanonicalOptimizationEngine._as_int(cards.get("open_cards")) <= CanonicalOptimizationEngine._as_int(cards.get("critical_cards")):
            reasons.append("No minor defects - optimal condition")
        max_mileage = float(trainset.get("max_mileage_before_maintenance", 50000) or 50000)
        ratio = float(trainset.get("current_mileage", 0.0) or 0.0) / max_mileage if max_mileage > 0 else 0.0
        if ratio < 0.5:
            reasons.append("Low mileage - good for fleet balancing")
        elif ratio < 0.7:
            reasons.append("Moderate mileage - suitable for service")
        if not bool(trainset.get("requires_cleaning", False)):
            reasons.append("No cleaning required - ready for service")
        if not bool(trainset.get("is_blocked", False)):
            reasons.append("No shunting complexity - easy deployment")
        health = float(trainset.get("ml_health_score", 0.85) or 0.85)
        if health < 0.5:
            reasons.append("ML Alert: Component fatigue detected")
        return reasons or ["Selected based on tiered optimization criteria"]

    @staticmethod
    def _normalized_score(trainset: Dict[str, Any], tier2: float, tier3: float, decision: str) -> float:
        if decision == "MAINTENANCE":
            return 0.0 if any(v.is_blocking for v in validate_trainset_safety(trainset)) else min(0.49, max(0.0, tier3 / 1000.0))
        branding = trainset.get("branding") or {}
        advertiser = branding.get("current_advertiser") if isinstance(branding, dict) else None
        priority = str(branding.get("priority", "LOW")).upper() if isinstance(branding, dict) else "LOW"
        if advertiser not in (None, "", "None") and tier2 > 0:
            return min(1.0, max(0.90, {"HIGH": 1.0, "MEDIUM": 0.97, "LOW": 0.94}.get(priority, 0.95)))
        if decision == "INDUCT":
            return min(0.89, 0.70 + max(0.0, min(0.19, tier3 / 1000.0)))
        return min(0.69, 0.50 + max(0.0, min(0.19, tier3 / 1000.0)))

    async def optimize(
        self,
        trainsets: Sequence[Dict[str, Any]],
        request: OptimizationRequest,
    ) -> tuple[List[InductionDecision], FleetRequirementResult]:
        normalized = [self._normalize(item) for item in trainsets]
        weights = self._effective_weights(request)

        # Risk inference is isolated behind a provider boundary. If the trained
        # predictor fails, use a deterministic heuristic without changing the
        # optimizer's safety semantics.
        try:
            predictions = await self.risk_provider.predict(normalized)
        except Exception as exc:
            logger.warning("Risk provider failed; using deterministic fallback: %s", exc)
            predictions = await HeuristicRiskProvider().predict(normalized)

        prediction_map = {item.trainset_id: item for item in predictions}
        enriched: List[Dict[str, Any]] = []
        for trainset in normalized:
            prediction = prediction_map.get(str(trainset.get("trainset_id", "UNKNOWN")))
            item = dict(trainset)
            if prediction:
                item["predicted_failure_risk"] = prediction.risk_probability
                item["ml_health_score"] = prediction.health_score
                item["risk_top_features"] = list(prediction.top_features)
            else:
                item["predicted_failure_risk"] = 0.2
                item["ml_health_score"] = 0.85
            enriched.append(item)

        eligible: List[Dict[str, Any]] = []
        blocked: List[tuple[Dict[str, Any], Any]] = []
        for trainset in enriched:
            violations = validate_trainset_safety(trainset)
            blocking = [v for v in violations if v.is_blocking]
            if blocking:
                blocked.append((trainset, violations))
            else:
                eligible.append(trainset)

        fleet_req = compute_required_trains(
            service_date=request.service_date,
            timetable_config=None,
            override_count=request.required_service_count,
        )
        target = min(fleet_req.required_service_trains, len(eligible))

        tier2 = {i: tier2_score(t, weights) for i, t in enumerate(eligible)}
        tier3 = {i: tier3_score(t, weights) for i, t in enumerate(eligible)}
        combined = {i: combined_score(t, weights) for i, t in enumerate(eligible)}

        chosen: set[int]
        solver = pywraplp.Solver.CreateSolver("SCIP")
        if solver is None:
            chosen = {i for i, _ in sorted(combined.items(), key=lambda pair: (-pair[1], str(eligible[pair[0]].get("trainset_id"))))[:target]}
        else:
            variables = {i: solver.BoolVar(f"x_{i}") for i in range(len(eligible))}
            solver.Add(solver.Sum(list(variables.values())) == target) if target > 0 else None
            if target == 0 and variables:
                solver.Add(solver.Sum(list(variables.values())) == 0)
            solver.Maximize(solver.Sum([combined[i] * variables[i] for i in variables]))
            status = solver.Solve()
            if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                chosen = {i for i, _ in sorted(combined.items(), key=lambda pair: (-pair[1], str(eligible[pair[0]].get("trainset_id"))))[:target]}
            else:
                chosen = {i for i, variable in variables.items() if variable.solution_value() > 0.5}

        decisions: List[InductionDecision] = []
        for i, trainset in enumerate(eligible):
            status = "INDUCT" if i in chosen else ("MAINTENANCE" if self._needs_maintenance(trainset) else "STANDBY")
            decisions.append(InductionDecision(
                trainset_id=str(trainset["trainset_id"]),
                decision=status,
                confidence_score=1.0 if status == "INDUCT" else 0.7,
                reasons=self._reasons(trainset) if status == "INDUCT" else (["Maintenance required - not selected for service"] if status == "MAINTENANCE" else ["Standby - lower tiered score than inducted trainsets"]),
                score=self._normalized_score(trainset, tier2[i], tier3[i], status),
                top_reasons=[],
                top_risks=[],
                violations=[],
                shap_values=[],
                summary=None,
            ))

        for trainset, violations in blocked:
            reasons = ["Critical failure detected - requires maintenance"] + [v.message for v in violations if v.is_blocking]
            decisions.append(InductionDecision(
                trainset_id=str(trainset["trainset_id"]),
                decision="MAINTENANCE",
                confidence_score=1.0,
                reasons=reasons,
                score=0.0,
                top_reasons=[],
                top_risks=[],
                violations=[v.code.value for v in violations],
                shap_values=[],
                summary=None,
            ))

        status_order = {"INDUCT": 0, "STANDBY": 1, "MAINTENANCE": 2}
        decisions.sort(key=lambda item: (status_order.get(item.decision, 99), -float(item.score or 0.0), item.trainset_id))

        # Hard safety assertion: a blocking constraint can never produce INDUCT.
        decision_map = {d.trainset_id: d for d in decisions}
        for trainset in enriched:
            if decision_map[trainset["trainset_id"]].decision == "INDUCT":
                assert not any(v.is_blocking for v in validate_trainset_safety(trainset)), (
                    f"Safety violation: {trainset['trainset_id']} selected for induction"
                )

        return decisions, fleet_req
