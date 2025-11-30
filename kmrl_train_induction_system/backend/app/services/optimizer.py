# backend/app/services/optimizer.py
from typing import List, Dict, Any, Optional
from app.models.trainset import OptimizationRequest, InductionDecision, OptimizationWeights
from ortools.linear_solver import pywraplp
import logging
from datetime import datetime
from app.utils.explainability import (
    top_reasons_and_risks, 
    generate_comprehensive_explanation,
    calculate_composite_score
)
from app.services.rule_engine import DurableRulesEngine
from app.utils.normalization import normalize_to_int, normalize_trainset_data

logger = logging.getLogger(__name__)

# Tiered Constraint Hierarchy Weights (Lexicographic Optimization)
WEIGHTS = {
    # Tier 1: Hard Constraints (Binary Filters)
    "CRITICAL_FAILURE": float('-inf'),  # Safety critical/Cert expired -> strict exclusion

    # Tier 2: High Priority Soft Objectives (Revenue)
    "BRANDING_OBLIGATION": 500.0,       # Active wrap adds +500 points (Revenue priority)
    "MINOR_DEFECT_PENALTY_PER_DEFECT": -20.0,  # -20 points per minor defect

    # Tier 3: Optimization Soft Objectives (Health/Ops)
    "MILEAGE_BALANCING": 100.0,         # +100 if below average (helps balance fleet)
    "CLEANING_DUE_PENALTY": -50.0,      # -50 penalty if cleaning due
    "SHUNTING_COMPLEXITY_PENALTY": -10.0 # Minor penalty if train is blocked by others
}

class TrainInductionOptimizer:
    """AI/ML optimization engine using Tiered Constraint Hierarchy (Lexicographic Optimization)"""
    
    def __init__(self):
        self.weights = WEIGHTS.copy()
        self.rule_engine = DurableRulesEngine()
    
    def _apply_final_sorting(self, decisions: List[InductionDecision], trainsets: List[Dict[str, Any]]) -> List[InductionDecision]:
        """Post-processing: Strict sorting
        
        Strict Sorting:
           - Priority 1: Status (INDUCT > STANDBY > MAINTENANCE)
           - Priority 2: Score (descending within status groups)
           - Priority 3: Mileage (ascending) - Balance fleet usage
           - Priority 4: Rank/ID (as final tie-breaker)
        """
        # Priority 1: Status order (INDUCT=0, STANDBY=1, MAINTENANCE=2)
        status_order = {"INDUCT": 0, "STANDBY": 1, "MAINTENANCE": 2}
        
        # Create a map for quick mileage lookup
        mileage_map = {t.get("trainset_id"): t.get("current_mileage", 0.0) for t in trainsets}
        
        def sort_key(decision: InductionDecision) -> tuple:
            status_priority = status_order.get(decision.decision, 99)
            score_value = decision.score if decision.score is not None else 0.0
            
            # Priority 3: Mileage (ascending)
            # We want lower mileage trains to be preferred if scores are tied (to balance usage)
            # But wait, if scores are tied, it means they are equally good candidates.
            # Lower mileage = better for balancing? Yes, usually we want to use under-utilized trains.
            mileage = mileage_map.get(decision.trainset_id, float('inf'))
            
            # Priority 4: ID
            rank_value = decision.trainset_id
            
            return (status_priority, -score_value, mileage, rank_value)
        
        sorted_decisions = sorted(decisions, key=sort_key)
        
        logger.info(f"Final sorting: {len([d for d in sorted_decisions if d.decision=='INDUCT'])} INDUCT, "
                   f"{len([d for d in sorted_decisions if d.decision=='STANDBY'])} STANDBY, "
                   f"{len([d for d in sorted_decisions if d.decision=='MAINTENANCE'])} MAINTENANCE, "
                   f"TOTAL: {len(sorted_decisions)} decisions")
        
        return sorted_decisions
    
    async def optimize(self, trainsets: List[Dict[str, Any]], request: OptimizationRequest, 
                      forced_ids: List[str] = None, excluded_ids: List[str] = None) -> List[InductionDecision]:
        """Run tiered constraint hierarchy optimization (Lexicographic Optimization)."""
        try:
            logger.info(f"Starting tiered optimization for {len(trainsets)} trainsets")
            
            forced_ids = forced_ids or []
            excluded_ids = excluded_ids or []
            
            # Update weights from request if provided
            if request.weights:
                # Map request weights to internal weights keys
                # Assuming request.weights is a dict or Pydantic model
                req_weights = request.weights.dict() if hasattr(request.weights, 'dict') else request.weights
                if req_weights:
                    # Map user-friendly keys to internal keys
                    # Example mapping based on typical user sliders
                    if 'branding' in req_weights:
                        self.weights["BRANDING_OBLIGATION"] = float(req_weights['branding']) * 1000.0
                    if 'reliability' in req_weights:
                        # Reliability usually implies avoiding defects
                        self.weights["MINOR_DEFECT_PENALTY_PER_DEFECT"] = -20.0 * float(req_weights['reliability'])
                    if 'mileage_balance' in req_weights:
                        self.weights["MILEAGE_BALANCING"] = 100.0 * float(req_weights['mileage_balance'])
                    if 'cleaning' in req_weights:
                        self.weights["CLEANING_DUE_PENALTY"] = -50.0 * float(req_weights['cleaning'])
                    if 'shunting' in req_weights:
                        self.weights["SHUNTING_COMPLEXITY_PENALTY"] = -10.0 * float(req_weights['shunting'])
                    
                    logger.info(f"Updated optimization weights based on user request: {self.weights}")
            
            # CRITICAL: Normalize all trainset data to ensure type safety (safety bug fix)
            normalized_trainsets = [normalize_trainset_data(ts) for ts in trainsets]
            trainsets = normalized_trainsets
            logger.info(f"Normalized {len(trainsets)} trainsets for type safety")
            
            # Integrate ML risk prediction for all trainsets
            try:
                from app.ml.predictor import batch_predict, predict_maintenance_health
                logger.info("Calling ML batch_predict for risk assessment")
                
                features_for_pred = []
                for t in trainsets:
                    feature_dict = {"trainset_id": t.get("trainset_id")}
                    for key, value in t.items():
                        if isinstance(value, (int, float)):
                            feature_dict[key] = value
                    features_for_pred.append(feature_dict)
                
                predictions = await batch_predict(features_for_pred)
                prediction_map = {p["trainset_id"]: p for p in predictions}
                
                for trainset in trainsets:
                    trainset_id = trainset.get("trainset_id")
                    if trainset_id in prediction_map:
                        pred = prediction_map[trainset_id]
                        trainset["predicted_failure_risk"] = pred.get("risk_prob", 0.2)
                        if "top_features" in pred:
                            trainset["risk_top_features"] = pred.get("top_features", [])
                    else:
                        trainset["predicted_failure_risk"] = trainset.get("predicted_failure_risk", 0.2)

                    try:
                        trainset["ml_health_score"] = predict_maintenance_health(trainset)
                    except Exception as health_err:
                        logger.warning(f"Health prediction failed for {trainset_id}: {health_err}")
                        trainset.setdefault("ml_health_score", 0.8)
                
                logger.info(f"ML predictions completed for {len(predictions)} trainsets")
            except Exception as ml_error:
                logger.warning(f"ML prediction failed, using existing risk values: {ml_error}")
                for trainset in trainsets:
                    if "predicted_failure_risk" not in trainset:
                        trainset["predicted_failure_risk"] = 0.2
                    if "ml_health_score" not in trainset:
                        trainset["ml_health_score"] = 0.85
            
            # TIER 1: Filter out critical failures (hard constraints) - STRICT FILTER
            eligible_trainsets = []
            critical_failures = []
            
            # Detailed logging for safety audit
            for trainset in trainsets:
                trainset_id = trainset.get("trainset_id", "UNKNOWN")
                
                # Check hard constraints via Rule Engine
                is_critical = await self._has_critical_failure(trainset)
                
                # Also check for "Critical Risk" from ML predictions (moved from Safety Gate)
                risk_features = trainset.get("risk_top_features", [])
                has_critical_risk_ml = any("critical" in str(f).lower() for f in risk_features)
                
                # Override for forced induction (unless strictly unsafe?)
                # Safety first: if it's unsafe, we shouldn't force it unless explicitly overridden with a safety waiver.
                # For now, we assume simulation respects safety unless user explicitly wants to simulate a disaster (not implemented).
                # But if it's just "excluded", we respect that.
                
                if trainset_id in excluded_ids:
                    critical_failures.append(trainset) # Treat as excluded
                    logger.info(f"SIMULATION: {trainset_id} excluded by user request")
                    continue

                if is_critical or has_critical_risk_ml:
                    critical_failures.append(trainset)
                    reason = "Rule Violation" if is_critical else "ML Critical Risk"
                    logger.warning(f"SAFETY EXCLUSION: {trainset_id} removed from eligible pool due to {reason}")
                else:
                    eligible_trainsets.append(trainset)
                    logger.debug(f"SAFETY PASS: {trainset_id} passed Tier 1 hard constraints")
            
            logger.info(f"Tier 1 filtering: {len(critical_failures)} critical failures excluded, {len(eligible_trainsets)} eligible")
            
            # Build optimization model with lexicographic scoring
            solver = pywraplp.Solver.CreateSolver("SCIP")
            if solver is None:
                return await self._optimize_with_tiered_scoring(eligible_trainsets, critical_failures, request, forced_ids)

            # Create decision variables and compute tiered scores
            x_vars = {}
            tier2_scores = {}  # High priority soft objectives
            tier3_scores = {}  # Optimization soft objectives

            for idx, t in enumerate(eligible_trainsets):
                var = solver.BoolVar(f"x_{idx}")
                x_vars[idx] = var
                
                # Simulation Constraints
                if t.get("trainset_id") in forced_ids:
                    solver.Add(var == 1)
                
                # Tier 2: High Priority Soft Objectives
                tier2_scores[idx] = self._calculate_tier2_score(t)
                
                # Tier 3: Optimization Soft Objectives
                tier3_scores[idx] = self._calculate_tier3_score(t)

            # Capacity constraint
            target = max(1, min(request.required_service_hours, len(eligible_trainsets)))
            # If forced count > target, we must increase target to accommodate forced trains
            forced_count = sum(1 for t in eligible_trainsets if t.get("trainset_id") in forced_ids)
            if forced_count > target:
                target = forced_count
                
            solver.Add(solver.Sum([x_vars[i] for i in x_vars]) <= target)

            # Lexicographic objective: Tier 2 dominates Tier 3
            # Reduced from 10000.0 to 10.0 to prevent branding dominance
            tier2_scale = 10.0
            objective_terms = []
            
            for i in x_vars:
                combined_score = (
                    tier2_scale * tier2_scores[i] + tier3_scores[i]
                )
                objective_terms.append(combined_score * x_vars[i])

            solver.Maximize(solver.Sum(objective_terms))
            status = solver.Solve()

            decisions: List[InductionDecision] = []
            
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                chosen_indices = [
                    i for i, var in x_vars.items() if var.solution_value() > 0.5
                ]

                if not chosen_indices:
                    logger.warning(
                        "OR-Tools returned a feasible solution with no inducted trains; "
                        "falling back to scoring-based tiered optimization."
                    )
                    return await self._optimize_with_tiered_scoring(
                        eligible_trainsets, critical_failures, request, forced_ids
                    )

                # Process inducted trainsets
                for i, t in enumerate(eligible_trainsets):
                    chosen = i in chosen_indices
                    if chosen:
                        tier2_val = tier2_scores[i]
                        tier3_val = tier3_scores[i]
                        combined_score = tier2_scale * tier2_val + tier3_val
                        confidence = min(1.0, max(0.5, (combined_score + 5000) / 10000))
                        
                        explanation = generate_comprehensive_explanation(t, "INDUCT")
                        reasons = self._get_tiered_induction_reasons(t, tier2_val, tier3_val)
                        
                        if t.get("trainset_id") in forced_ids:
                            reasons.insert(0, "Forced induction by user simulation")
                        
                        normalized_score = self._calculate_normalized_optimization_score(
                            t, tier2_val, tier3_val, "INDUCT"
                        )
                        
                        decisions.append(
                            InductionDecision(
                                trainset_id=t["trainset_id"],
                                decision="INDUCT",
                                confidence_score=confidence,
                                reasons=reasons + explanation.get("top_reasons", []),
                                score=normalized_score,
                                top_reasons=explanation.get("top_reasons", []),
                                top_risks=explanation.get("top_risks", []),
                                violations=explanation.get("violations", []),
                                shap_values=explanation.get("shap_values", [])
                            )
                        )
                
                # Process non-inducted eligible trainsets
                for i, t in enumerate(eligible_trainsets):
                    if i not in chosen_indices:
                        if self._needs_maintenance(t):
                            explanation = generate_comprehensive_explanation(t, "MAINTENANCE")
                            tier2_val = tier2_scores[i]
                            tier3_val = tier3_scores[i]
                            normalized_score = self._calculate_normalized_optimization_score(
                                t, tier2_val, tier3_val, "MAINTENANCE"
                            )
                            decisions.append(
                                InductionDecision(
                                    trainset_id=t["trainset_id"],
                                    decision="MAINTENANCE",
                                    confidence_score=0.9,
                                    reasons=["Maintenance required - not selected for service"] + explanation.get("top_reasons", []),
                                    score=normalized_score,
                                    top_reasons=explanation.get("top_reasons", []),
                                    top_risks=explanation.get("top_risks", []),
                                    violations=explanation.get("violations", []),
                                    shap_values=explanation.get("shap_values", [])
                                )
                            )
                        else:
                            explanation = generate_comprehensive_explanation(t, "STANDBY")
                            tier2_val = tier2_scores[i]
                            tier3_val = tier3_scores[i]
                            normalized_score = self._calculate_normalized_optimization_score(
                                t, tier2_val, tier3_val, "STANDBY"
                            )
                            decisions.append(
                                InductionDecision(
                                    trainset_id=t["trainset_id"],
                                    decision="STANDBY",
                                    confidence_score=0.7,
                                    reasons=["Standby - lower tiered score than inducted trainsets"] + explanation.get("top_reasons", []),
                                    score=normalized_score,
                                    top_reasons=explanation.get("top_reasons", []),
                                    top_risks=explanation.get("top_risks", []),
                                    violations=explanation.get("violations", []),
                                    shap_values=explanation.get("shap_values", [])
                                )
                            )
                
                # Process critical failures (maintenance required)
                for t in critical_failures:
                    explanation = generate_comprehensive_explanation(t, "MAINTENANCE")
                    
                    failure_reasons = ["Critical failure detected - requires maintenance"]
                    if t.get("trainset_id") in excluded_ids:
                        failure_reasons = ["Excluded by user simulation"]
                    
                    job_cards = t.get("job_cards", {})
                    critical_cards = normalize_to_int(job_cards.get("critical_cards"), 0)
                    if critical_cards > 0:
                        failure_reasons.append(f"{critical_cards} critical job cards open")
                    
                    fitness_certs = t.get("fitness_certificates", {})
                    expired_certs = [k for k, v in fitness_certs.items() if isinstance(v, dict) and str(v.get("status", "")).upper() == "EXPIRED"]
                    if expired_certs:
                        failure_reasons.append(f"Expired certificates: {', '.join(expired_certs)}")
                    
                    decisions.append(
                        InductionDecision(
                            trainset_id=t["trainset_id"],
                            decision="MAINTENANCE",
                            confidence_score=1.0,
                            reasons=failure_reasons,
                            score=0.0,
                            top_reasons=explanation["top_reasons"],
                            top_risks=explanation["top_risks"],
                            violations=explanation["violations"],
                            shap_values=explanation["shap_values"]
                        )
                    )
            else:
                return await self._optimize_with_tiered_scoring(eligible_trainsets, critical_failures, request, forced_ids)

            logger.info(f"Tiered optimization completed: {len([d for d in decisions if d.decision=='INDUCT'])} inducted")
            
            # FINAL SAFETY SANITY CHECK
            for decision in decisions:
                if decision.decision == "INDUCT":
                    trainset_id = decision.trainset_id
                    original_trainset = next((ts for ts in trainsets if ts.get("trainset_id") == trainset_id), None)
                    
                    if original_trainset:
                        # Re-check critical failure conditions via Rule Engine
                        if await self._has_critical_failure(original_trainset):
                            # If forced, we might allow it with a warning? 
                            # For now, strict safety unless we add a specific "unsafe_override" flag.
                            # Simulation "force_induct" implies "I want this train", but if it's unsafe, 
                            # the system should probably still reject it or flag it heavily.
                            # Given the user context "Refactoring Critical Logic Flaws", safety is paramount.
                            # But for "What-if", maybe they want to see what happens?
                            # Let's stick to safety. If it's forced but unsafe, it should fail this check.
                            # However, if we filtered it out in Tier 1, it wouldn't be here.
                            # So this check is for trains that PASSED Tier 1.
                            error_msg = f"SAFETY VIOLATION: Train {trainset_id} with critical failure made it to final INDUCT list!"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
            
            logger.info("Safety sanity check passed - all inducted trains meet Tier 1 constraints")
            
            # POST-PROCESSING: Strict Sorting (Safety Gate logic moved to filtering)
            decisions = self._apply_final_sorting(decisions, trainsets)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            raise
    
    async def _has_critical_failure(self, trainset: Dict[str, Any]) -> bool:
        """TIER 1: Check if trainset has critical failure using Rule Engine."""
        violations = await self.rule_engine.check_constraints(trainset)
        if violations:
            trainset_id = trainset.get("trainset_id", "UNKNOWN")
            logger.warning(f"SAFETY FILTER: {trainset_id} excluded - Violations: {violations}")
            return True
        return False
    
    def _calculate_tier2_score(self, trainset: Dict[str, Any]) -> float:
        """TIER 2: Calculate high priority soft objectives score."""
        score = 0.0
        
        # Branding obligation
        branding = trainset.get("branding", {})
        if isinstance(branding, dict):
            advertiser = branding.get("current_advertiser")
            priority = branding.get("priority", "LOW")
            
            if advertiser and advertiser != "None" and advertiser != "":
                if priority == "HIGH":
                    score += self.weights["BRANDING_OBLIGATION"]
                elif priority == "MEDIUM":
                    score += self.weights["BRANDING_OBLIGATION"] * 0.6
                elif priority == "LOW":
                    score += self.weights["BRANDING_OBLIGATION"] * 0.3
        
        # Minor defect penalty
        job_cards = trainset.get("job_cards", {})
        open_cards = normalize_to_int(job_cards.get("open_cards"), 0)
        critical_cards = normalize_to_int(job_cards.get("critical_cards"), 0)
        minor_cards = max(0, open_cards - critical_cards)
        
        if minor_cards > 0:
            penalty = self.weights["MINOR_DEFECT_PENALTY_PER_DEFECT"] * minor_cards
            score += penalty
        
        return score
    
    def _calculate_tier3_score(self, trainset: Dict[str, Any]) -> float:
        """TIER 3: Calculate optimization soft objectives score."""
        score = 0.0
        
        # Mileage balancing
        current_mileage = trainset.get("current_mileage", 0.0)
        km_30d = trainset.get("km_30d", current_mileage * 0.1)
        km_30d_norm = min(1.0, km_30d / 5000.0) if km_30d else 0.5
        
        if km_30d_norm < 0.5:
            score += self.weights["MILEAGE_BALANCING"]
        else:
            score += self.weights["MILEAGE_BALANCING"] * (1.0 - km_30d_norm)
        
        # Cleaning due penalty
        requires_cleaning = bool(trainset.get("requires_cleaning", False))
        cleaning_due_date = trainset.get("cleaning_due_date")
        
        if requires_cleaning and cleaning_due_date:
            try:
                due_date = self._parse_cleaning_date(cleaning_due_date)
                if due_date:
                    days_until_due = (due_date - datetime.utcnow()).days
                    if days_until_due <= 7:
                        penalty_factor = 1.0 - (days_until_due / 7.0)
                        score += self.weights["CLEANING_DUE_PENALTY"] * penalty_factor
                else:
                    score += self.weights["CLEANING_DUE_PENALTY"] * 0.5
            except Exception:
                score += self.weights["CLEANING_DUE_PENALTY"] * 0.5
        elif requires_cleaning:
            score += self.weights["CLEANING_DUE_PENALTY"] * 0.3
        
        # Shunting complexity penalty
        is_blocked = bool(trainset.get("is_blocked", False))
        shunt_complexity = trainset.get("shunt_complexity", 0.0)
        
        if is_blocked or shunt_complexity > 0.5:
            complexity_factor = 1.0 if is_blocked else shunt_complexity
            score += self.weights["SHUNTING_COMPLEXITY_PENALTY"] * complexity_factor
        
        # ML health contribution
        health_score = float(trainset.get("ml_health_score", 0.85) or 0.85)
        score += health_score * 100.0
        
        return score

    async def _optimize_with_tiered_scoring(self, eligible_trainsets: List[Dict[str, Any]], critical_failures: List[Dict[str, Any]], request: OptimizationRequest, forced_ids: List[str] = None) -> List[InductionDecision]:
        """Fallback tiered scoring-based optimization when OR-Tools is unavailable."""
        forced_ids = forced_ids or []
        scored_trainsets = []
        for trainset in eligible_trainsets:
            tier2_score = self._calculate_tier2_score(trainset)
            tier3_score = self._calculate_tier3_score(trainset)
            
            tier2_scale = 10.0
            combined_score = tier2_scale * tier2_score + tier3_score
            
            # Boost score for forced trains to ensure they are picked
            if trainset.get("trainset_id") in forced_ids:
                combined_score += 1e9
            
            scored_trainsets.append((trainset, tier2_score, tier3_score, combined_score))
        
        scored_trainsets.sort(key=lambda x: x[3], reverse=True)
        
        target_inductions = min(request.required_service_hours, len(scored_trainsets))
        # Adjust target for forced
        forced_count = sum(1 for t in eligible_trainsets if t.get("trainset_id") in forced_ids)
        if forced_count > target_inductions:
            target_inductions = forced_count
            
        decisions: List[InductionDecision] = []
        inducted = 0
        
        for trainset, tier2_score, tier3_score, combined_score in scored_trainsets:
            if inducted < target_inductions:
                confidence = min(1.0, max(0.5, (combined_score + 5000) / 10000))
                explanation = generate_comprehensive_explanation(trainset, "INDUCT")
                reasons = self._get_tiered_induction_reasons(trainset, tier2_score, tier3_score)
                if trainset.get("trainset_id") in forced_ids:
                    reasons.insert(0, "Forced induction by user simulation")
                    
                normalized_score = self._calculate_normalized_optimization_score(
                    trainset, tier2_score, tier3_score, "INDUCT"
                )
                
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="INDUCT",
                    confidence_score=confidence,
                    reasons=reasons + explanation.get("top_reasons", []),
                    score=normalized_score,
                    top_reasons=explanation.get("top_reasons", []),
                    top_risks=explanation.get("top_risks", []),
                    violations=explanation.get("violations", []),
                    shap_values=explanation.get("shap_values", [])
                ))
                inducted += 1
            elif self._needs_maintenance(trainset):
                explanation = generate_comprehensive_explanation(trainset, "MAINTENANCE")
                normalized_score = self._calculate_normalized_optimization_score(
                    trainset, tier2_score, tier3_score, "MAINTENANCE"
                )
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="MAINTENANCE",
                    confidence_score=0.9,
                    reasons=["Maintenance required - not selected for service"] + explanation.get("top_reasons", []),
                    score=normalized_score,
                    top_reasons=explanation.get("top_reasons", []),
                    top_risks=explanation.get("top_risks", []),
                    violations=explanation.get("violations", []),
                    shap_values=explanation.get("shap_values", [])
                ))
            else:
                explanation = generate_comprehensive_explanation(trainset, "STANDBY")
                normalized_score = self._calculate_normalized_optimization_score(
                    trainset, tier2_score, tier3_score, "STANDBY"
                )
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="STANDBY",
                    confidence_score=0.7,
                    reasons=["Standby - lower tiered score than inducted trainsets"] + explanation.get("top_reasons", []),
                    score=normalized_score,
                    top_reasons=explanation.get("top_reasons", []),
                    top_risks=explanation.get("top_risks", []),
                    violations=explanation.get("violations", []),
                    shap_values=explanation.get("shap_values", [])
                ))
        
        for trainset in critical_failures:
            explanation = generate_comprehensive_explanation(trainset, "MAINTENANCE")
            
            failure_reasons = ["Critical failure detected - requires maintenance"]
            job_cards = trainset.get("job_cards", {})
            critical_cards = normalize_to_int(job_cards.get("critical_cards"), 0)
            if critical_cards > 0:
                failure_reasons.append(f"{critical_cards} critical job cards open")
            
            fitness_certs = trainset.get("fitness_certificates", {})
            expired_certs = [k for k, v in fitness_certs.items() if isinstance(v, dict) and str(v.get("status", "")).upper() == "EXPIRED"]
            if expired_certs:
                failure_reasons.append(f"Expired certificates: {', '.join(expired_certs)}")
            
            decisions.append(InductionDecision(
                trainset_id=trainset["trainset_id"],
                decision="MAINTENANCE",
                confidence_score=1.0,
                reasons=failure_reasons,
                score=0.0,
                top_reasons=explanation["top_reasons"],
                top_risks=explanation["top_risks"],
                violations=explanation["violations"],
                shap_values=explanation["shap_values"]
            ))
        
        # FINAL SAFETY SANITY CHECK (fallback)
        for decision in decisions:
            if decision.decision == "INDUCT":
                trainset_id = decision.trainset_id
                original_trainset = next((ts for ts in eligible_trainsets if ts.get("trainset_id") == trainset_id), None)
                if original_trainset:
                     if await self._has_critical_failure(original_trainset):
                        error_msg = f"SAFETY VIOLATION (fallback): Train {trainset_id} with critical failure made it to final INDUCT list!"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
        
        decisions = self._apply_final_sorting(decisions, eligible_trainsets)
        
        return decisions
