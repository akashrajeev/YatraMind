# backend/app/services/optimizer.py
from typing import List, Dict, Any
from app.models.trainset import OptimizationRequest, InductionDecision
from ortools.linear_solver import pywraplp
import logging
from app.utils.explainability import (
    top_reasons_and_risks, 
    generate_comprehensive_explanation,
    calculate_composite_score
)

logger = logging.getLogger(__name__)

class TrainInductionOptimizer:
    """AI/ML optimization engine using Google OR-Tools + PyTorch"""
    
    def __init__(self):
        self.optimization_weights = {
            "fitness_score": 0.3,
            "maintenance_priority": 0.25,
            "branding_priority": 0.2,
            "mileage_balance": 0.15,
            "operational_efficiency": 0.1
        }
    
    async def optimize(self, trainsets: List[Dict[str, Any]], request: OptimizationRequest) -> List[InductionDecision]:
        """Run multi-objective optimization for train induction.

        Objectives combined as weighted sum:
        - Service readiness (readiness): prefer trainsets ready to deploy
        - Reliability (reliability): prefer healthy sensors / fewer issues
        - Cost (cost): prefer lower implied operational/maintenance cost
        - Branding exposure (branding): prefer higher branding impact
        """
        try:
            logger.info(f"Starting optimization for {len(trainsets)} trainsets")
            
            # Build an assignment model (0/1) with weighted objective
            solver = pywraplp.Solver.CreateSolver("SCIP")
            if solver is None:
                # Fallback to previous scoring if OR-Tools missing
                return await self._optimize_with_scoring(trainsets, request)

            x_vars = {}
            readiness_scores = {}
            reliability_scores = {}
            cost_scores = {}
            branding_scores = {}

            for idx, t in enumerate(trainsets):
                var = solver.BoolVar(f"x_{idx}")
                x_vars[idx] = var
                readiness_scores[idx] = self._readiness_score(t)
                reliability_scores[idx] = self._reliability_score(t)
                cost_scores[idx] = self._cost_score(t)
                branding_scores[idx] = self._branding_score(t)

            # Capacity: choose up to required_service_hours trainsets (proxy for slots)
            target = max(1, min(request.required_service_hours, len(trainsets)))
            solver.Add(solver.Sum([x_vars[i] for i in x_vars]) <= target)

            # Hard constraints derived from eligibility (filter)
            eligible = self._filter_by_constraints(trainsets)
            eligible_ids = {t["trainset_id"] for t in eligible}
            for idx, t in enumerate(trainsets):
                if t["trainset_id"] not in eligible_ids:
                    solver.Add(x_vars[idx] == 0)

            # Weighted objective
            w_readiness = 0.35
            w_reliability = 0.30
            w_cost = 0.15
            w_branding = 0.20

            objective_terms = []
            for i in x_vars:
                term = (
                    w_readiness * readiness_scores[i]
                    + w_reliability * reliability_scores[i]
                    + w_branding * branding_scores[i]
                    + w_cost * cost_scores[i]
                )
                objective_terms.append(term * x_vars[i])

            solver.Maximize(solver.Sum(objective_terms))
            status = solver.Solve()

            decisions: List[InductionDecision] = []
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                for i, t in enumerate(trainsets):
                    chosen = x_vars[i].solution_value() > 0.5
                    if chosen:
                        score = float(objective_terms[i].Evaluate()) if hasattr(objective_terms[i], 'Evaluate') else 0.85
                        # Generate comprehensive explanation
                        explanation = generate_comprehensive_explanation(t, "INDUCT")
                        decisions.append(
                            InductionDecision(
                                trainset_id=t["trainset_id"],
                                decision="INDUCT",
                                confidence_score=min(1.0, max(0.5, score)),
                                reasons=self._get_induction_reasons(t, score) + explanation["top_reasons"],
                                score=explanation["score"],
                                top_reasons=explanation["top_reasons"],
                                top_risks=explanation["top_risks"],
                                violations=explanation["violations"],
                                shap_values=explanation["shap_values"]
                            )
                        )
                # Non chosen items
                for i, t in enumerate(trainsets):
                    if x_vars[i].solution_value() <= 0.5:
                        if self._needs_maintenance(t):
                            explanation = generate_comprehensive_explanation(t, "MAINTENANCE")
                            decisions.append(
                                InductionDecision(
                                    trainset_id=t["trainset_id"],
                                    decision="MAINTENANCE",
                                    confidence_score=0.9,
                                    reasons=["Maintenance required based on constraints"],
                                    score=explanation["score"],
                                    top_reasons=explanation["top_reasons"],
                                    top_risks=explanation["top_risks"],
                                    violations=explanation["violations"],
                                    shap_values=explanation["shap_values"]
                                )
                            )
                        else:
                            explanation = generate_comprehensive_explanation(t, "STANDBY")
                            decisions.append(
                                InductionDecision(
                                    trainset_id=t["trainset_id"],
                                    decision="STANDBY",
                                    confidence_score=0.7,
                                    reasons=["Standby due to lower composite score"],
                                    score=explanation["score"],
                                    top_reasons=explanation["top_reasons"],
                                    top_risks=explanation["top_risks"],
                                    violations=explanation["violations"],
                                    shap_values=explanation["shap_values"]
                                )
                            )
            else:
                return await self._optimize_with_scoring(trainsets, request)

            logger.info(f"Optimization completed: {len([d for d in decisions if d.decision=='INDUCT'])} inducted")
            return decisions
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _filter_by_constraints(self, trainsets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply rule-based constraints filtering"""
        eligible = []
        
        for trainset in trainsets:
            # Check fitness certificates
            fitness_valid = all(
                cert["status"] == "VALID" 
                for cert in trainset["fitness_certificates"].values()
            )
            
            # Check critical job cards
            no_critical_cards = trainset["job_cards"]["critical_cards"] == 0
            
            # Check mileage limits
            mileage_ok = trainset["current_mileage"] < trainset["max_mileage_before_maintenance"]

            # Cleaning slot hard constraint (if fields provided)
            requires_cleaning = bool(trainset.get("requires_cleaning", False))
            has_cleaning_slot = bool(trainset.get("has_cleaning_slot", False))
            cleaning_ok = (not requires_cleaning) or (requires_cleaning and has_cleaning_slot)
            
            if fitness_valid and no_critical_cards and mileage_ok and cleaning_ok:
                eligible.append(trainset)
        
        return eligible
    
    def _calculate_optimization_score(self, trainset: Dict[str, Any], request: OptimizationRequest) -> float:
        """Calculate multi-objective optimization score"""
        score = 0.0
        
        # Fitness score (0-1)
        fitness_score = self._calculate_fitness_score(trainset)
        score += fitness_score * self.optimization_weights["fitness_score"]
        
        # Maintenance priority (inverse of open cards)
        maintenance_score = max(0, 1 - (trainset["job_cards"]["open_cards"] / 10))
        score += maintenance_score * self.optimization_weights["maintenance_priority"]
        
        # Branding priority
        branding_score = self._calculate_branding_score(trainset)
        score += branding_score * self.optimization_weights["branding_priority"]
        
        # Mileage balance (prefer lower mileage for even distribution)
        mileage_score = 1 - (trainset["current_mileage"] / trainset["max_mileage_before_maintenance"])
        score += mileage_score * self.optimization_weights["mileage_balance"]
        
        # Operational efficiency / reliability via ML risk: higher risk => lower score
        predicted_risk = float(trainset.get("predicted_failure_risk", 0.2))
        operational_score = max(0.0, 1.0 - predicted_risk)
        score += operational_score * self.optimization_weights["operational_efficiency"]
        
        return min(score, 1.0)

    async def _optimize_with_scoring(self, trainsets: List[Dict[str, Any]], request: OptimizationRequest) -> List[InductionDecision]:
        """Fallback scorer-only optimization when OR-Tools is unavailable."""
        eligible_trainsets = self._filter_by_constraints(trainsets)
        scored_trainsets = []
        for trainset in eligible_trainsets:
            score = self._calculate_optimization_score(trainset, request)
            scored_trainsets.append((trainset, score))
        scored_trainsets.sort(key=lambda x: x[1], reverse=True)
        target_inductions = min(14, len(scored_trainsets))
        decisions: List[InductionDecision] = []
        inducted = 0
        for trainset, score in scored_trainsets:
            if inducted < target_inductions and self._can_induct(trainset):
                explanation = generate_comprehensive_explanation(trainset, "INDUCT")
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="INDUCT",
                    confidence_score=min(score, 1.0),
                    reasons=self._get_induction_reasons(trainset, score),
                    score=explanation["score"],
                    top_reasons=explanation["top_reasons"],
                    top_risks=explanation["top_risks"],
                    violations=explanation["violations"],
                    shap_values=explanation["shap_values"]
                ))
                inducted += 1
            elif self._needs_maintenance(trainset):
                explanation = generate_comprehensive_explanation(trainset, "MAINTENANCE")
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="MAINTENANCE",
                    confidence_score=0.9,
                    reasons=["Maintenance required based on constraints"],
                    score=explanation["score"],
                    top_reasons=explanation["top_reasons"],
                    top_risks=explanation["top_risks"],
                    violations=explanation["violations"],
                    shap_values=explanation["shap_values"]
                ))
            else:
                explanation = generate_comprehensive_explanation(trainset, "STANDBY")
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="STANDBY",
                    confidence_score=0.7,
                    reasons=["Standby due to lower priority"],
                    score=explanation["score"],
                    top_reasons=explanation["top_reasons"],
                    top_risks=explanation["top_risks"],
                    violations=explanation["violations"],
                    shap_values=explanation["shap_values"]
                ))
        return decisions

    def _readiness_score(self, trainset: Dict[str, Any]) -> float:
        fitness_valid = all(cert["status"] == "VALID" for cert in trainset["fitness_certificates"].values())
        no_critical = trainset["job_cards"]["critical_cards"] == 0
        not_maintenance = trainset["status"] in ["STANDBY", "ACTIVE"]
        return float(fitness_valid) * 0.5 + float(no_critical) * 0.3 + float(not_maintenance) * 0.2

    def _reliability_score(self, trainset: Dict[str, Any]) -> float:
        # Use sensor health score if available (0..1)
        return float(max(0.0, min(1.0, trainset.get("sensor_health_score", 0.8))))

    def _cost_score(self, trainset: Dict[str, Any]) -> float:
        # Lower mileage and fewer open cards imply lower cost; scale to 0..1
        mileage_ratio = trainset["current_mileage"] / max(1.0, trainset["max_mileage_before_maintenance"])
        open_cards = trainset["job_cards"]["open_cards"]
        return max(0.0, 1.0 - 0.7 * mileage_ratio - 0.3 * min(1.0, open_cards / 10.0))

    def _branding_score(self, trainset: Dict[str, Any]) -> float:
        # Map branding priority (0..1). Support both int priority and dict.
        branding = trainset.get("branding", {})
        if isinstance(branding, dict):
            pri = branding.get("priority")
            if isinstance(pri, str):
                mapping = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
                return mapping.get(pri, 0.0)
        # Fallback to existing numeric field
        return min(1.0, max(0.0, float(trainset.get("branding_priority", 0) / 3.0)))
    
    def _calculate_fitness_score(self, trainset: Dict[str, Any]) -> float:
        """Calculate overall fitness score from certificates"""
        fitness_certs = trainset["fitness_certificates"]
        valid_count = sum(1 for cert in fitness_certs.values() if cert["status"] == "VALID")
        return valid_count / len(fitness_certs)
    
    def _calculate_branding_score(self, trainset: Dict[str, Any]) -> float:
        """Calculate branding priority score"""
        branding = trainset.get("branding", {})
        if not branding or branding.get("current_advertiser") == "None":
            return 0.0
        
        priority_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
        return priority_map.get(branding.get("priority", "LOW"), 0.0)
    
    def _can_induct(self, trainset: Dict[str, Any]) -> bool:
        """Check if trainset can be inducted"""
        return (
            trainset["status"] in ["STANDBY", "ACTIVE"] and
            trainset["job_cards"]["critical_cards"] == 0
        )
    
    def _needs_maintenance(self, trainset: Dict[str, Any]) -> bool:
        """Check if trainset needs maintenance"""
        return (
            trainset["job_cards"]["critical_cards"] > 0 or
            trainset["current_mileage"] >= trainset["max_mileage_before_maintenance"] * 0.95
        )
    
    def _get_induction_reasons(self, trainset: Dict[str, Any], score: float) -> List[str]:
        """Generate human-readable reasons for induction decision"""
        reasons = []
        
        if score > 0.8:
            reasons.append("High optimization score")
        
        if trainset["job_cards"]["open_cards"] == 0:
            reasons.append("No pending maintenance")
        
        if trainset.get("branding", {}).get("priority") == "HIGH":
            reasons.append("High branding priority")
        
        if trainset["current_mileage"] < trainset["max_mileage_before_maintenance"] * 0.5:
            reasons.append("Low mileage - good for service")
        
        return reasons if reasons else ["Selected based on optimization criteria"]
