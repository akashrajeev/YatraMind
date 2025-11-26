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
        self.weights = WEIGHTS
    
    def _apply_final_safety_gate_and_sort(self, decisions: List[InductionDecision]) -> List[InductionDecision]:
        """Post-processing: Final safety gate + strict sorting
        
        1. Final Safety Gate: Force any train with "Risk: Critical" to MAINTENANCE
        2. Strict Sorting:
           - Priority 1: Status (INDUCT > STANDBY > MAINTENANCE)
           - Priority 2: Score (descending within status groups)
           - Priority 3: Rank (as tie-breaker)
        """
        # Step 1: Final Safety Gate - Check for "Risk: Critical" text
        for decision in decisions:
            # Check top_risks for any critical risk indicators
            has_critical_risk = False
            critical_risk_text = None
            
            # Check top_risks list
            if decision.top_risks:
                for risk in decision.top_risks:
                    risk_lower = str(risk).lower()
                    if "critical" in risk_lower or "risk: critical" in risk_lower:
                        has_critical_risk = True
                        critical_risk_text = risk
                        break
            
            # Check reasons for critical indicators
            if not has_critical_risk and decision.reasons:
                for reason in decision.reasons:
                    if "critical" in str(reason).lower() and "failure" in str(reason).lower():
                        has_critical_risk = True
                        critical_risk_text = reason
                        break
            
            # Force to MAINTENANCE if critical risk detected
            if has_critical_risk and decision.decision != "MAINTENANCE":
                logger.warning(f"FINAL SAFETY GATE: {decision.trainset_id} forced to MAINTENANCE due to critical risk: {critical_risk_text}")
                decision.decision = "MAINTENANCE"
                decision.confidence_score = 1.0
                decision.score = 0.0  # Tier 1: Score = 0 for critical failures
                if critical_risk_text not in decision.reasons:
                    decision.reasons.insert(0, f"Forced to MAINTENANCE: {critical_risk_text}")
        
        # Step 2: Strict Sorting
        # Priority 1: Status order (INDUCT=0, STANDBY=1, MAINTENANCE=2)
        status_order = {"INDUCT": 0, "STANDBY": 1, "MAINTENANCE": 2}
        
        # Priority 2: Score (descending)
        # Priority 3: Rank (use trainset_id as tie-breaker if no explicit rank)
        def sort_key(decision: InductionDecision) -> tuple:
            status_priority = status_order.get(decision.decision, 99)
            score_value = decision.score if decision.score is not None else 0.0
            # Use trainset_id as rank tie-breaker (lexicographic)
            rank_value = decision.trainset_id
            
            return (status_priority, -score_value, rank_value)
        
        sorted_decisions = sorted(decisions, key=sort_key)
        
        logger.info(f"Final sorting: {len([d for d in sorted_decisions if d.decision=='INDUCT'])} INDUCT, "
                   f"{len([d for d in sorted_decisions if d.decision=='STANDBY'])} STANDBY, "
                   f"{len([d for d in sorted_decisions if d.decision=='MAINTENANCE'])} MAINTENANCE, "
                   f"TOTAL: {len(sorted_decisions)} decisions")
        
        return sorted_decisions
    
    def _normalize_trainset_data(self, trainset: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize trainset data to ensure correct types (safety bug fix).
        
        Ensures critical_cards and open_cards are integers, not strings.
        This prevents type comparison bugs that could allow unsafe trains through.
        """
        # Normalize job_cards structure
        job_cards = trainset.get("job_cards", {})
        if not isinstance(job_cards, dict):
            job_cards = {}
        
        normalized_job_cards = {
            "open_cards": self._normalize_to_int(job_cards.get("open_cards"), 0),
            "critical_cards": self._normalize_to_int(job_cards.get("critical_cards"), 0)
        }
        
        # Create normalized copy
        normalized = trainset.copy()
        normalized["job_cards"] = normalized_job_cards
        
        # Normalize other critical fields
        if "current_mileage" in normalized:
            try:
                normalized["current_mileage"] = float(normalized["current_mileage"])
            except (ValueError, TypeError):
                normalized["current_mileage"] = 0.0
        
        if "max_mileage_before_maintenance" in normalized:
            try:
                max_mileage = normalized["max_mileage_before_maintenance"]
                if max_mileage in (None, "", 0):
                    normalized["max_mileage_before_maintenance"] = float('inf')
                else:
                    normalized["max_mileage_before_maintenance"] = float(max_mileage)
            except (ValueError, TypeError):
                normalized["max_mileage_before_maintenance"] = float('inf')
        
        return normalized
    
    async def optimize(self, trainsets: List[Dict[str, Any]], request: OptimizationRequest) -> List[InductionDecision]:
        """Run tiered constraint hierarchy optimization (Lexicographic Optimization).
        
        Tier 1: Hard Constraints - Filter out critical failures (safety/cert expiry)
        Tier 2: High Priority Soft Objectives - Branding obligations, minor defects
        Tier 3: Optimization Soft Objectives - Mileage balancing, cleaning, shunting
        """
        try:
            logger.info(f"Starting tiered optimization for {len(trainsets)} trainsets")
            
            # CRITICAL: Normalize all trainset data to ensure type safety (safety bug fix)
            normalized_trainsets = [self._normalize_trainset_data(ts) for ts in trainsets]
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

                    # Derive a lightweight ML maintenance health signal that feeds
                    # into Tier‑3 scoring. This uses a heuristic predictor so that
                    # the ranking is visibly influenced even without a trained model.
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
                    # Fallback health score when full ML stack is unavailable
                    if "ml_health_score" not in trainset:
                        trainset["ml_health_score"] = 0.85
            
            # TIER 1: Filter out critical failures (hard constraints) - STRICT FILTER
            eligible_trainsets = []
            critical_failures = []
            
            # Detailed logging for safety audit
            for trainset in trainsets:
                trainset_id = trainset.get("trainset_id", "UNKNOWN")
                if self._has_critical_failure(trainset):
                    critical_failures.append(trainset)
                    logger.warning(f"SAFETY EXCLUSION: {trainset_id} removed from eligible pool due to critical failure")
                else:
                    eligible_trainsets.append(trainset)
                    logger.debug(f"SAFETY PASS: {trainset_id} passed Tier 1 hard constraints")
            
            logger.info(f"Tier 1 filtering: {len(critical_failures)} critical failures excluded, {len(eligible_trainsets)} eligible")
            
            # Safety audit: Log detailed breakdown of exclusions
            if critical_failures:
                for fail_trainset in critical_failures:
                    ts_id = fail_trainset.get("trainset_id", "UNKNOWN")
                    job_cards = fail_trainset.get("job_cards", {})
                    crit_count = self._normalize_to_int(job_cards.get("critical_cards"), 0)
                    fitness = fail_trainset.get("fitness_certificates", {})
                    expired_certs = [k for k, v in fitness.items() if isinstance(v, dict) and str(v.get("status", "")).upper() == "EXPIRED"]
                    logger.warning(f"SAFETY AUDIT: {ts_id} - Critical cards: {crit_count}, Expired certs: {expired_certs}")
            
            # Build optimization model with lexicographic scoring
            solver = pywraplp.Solver.CreateSolver("SCIP")
            if solver is None:
                # Fallback to scoring-based optimization
                return await self._optimize_with_tiered_scoring(eligible_trainsets, critical_failures, request)

            # Create decision variables and compute tiered scores
            x_vars = {}
            tier2_scores = {}  # High priority soft objectives
            tier3_scores = {}  # Optimization soft objectives

            for idx, t in enumerate(eligible_trainsets):
                var = solver.BoolVar(f"x_{idx}")
                x_vars[idx] = var
                
                # Tier 2: High Priority Soft Objectives
                tier2_scores[idx] = self._calculate_tier2_score(t)
                
                # Tier 3: Optimization Soft Objectives
                tier3_scores[idx] = self._calculate_tier3_score(t)

            # Capacity constraint
            target = max(1, min(request.required_service_hours, len(eligible_trainsets)))
            solver.Add(solver.Sum([x_vars[i] for i in x_vars]) <= target)

            # Lexicographic objective: Tier 2 dominates Tier 3
            # Scale Tier 2 by large factor to ensure lexicographic ordering
            tier2_scale = 10000.0  # Ensure Tier 2 objectives dominate
            objective_terms = []
            
            for i in x_vars:
                # Combined score: Tier 2 (high priority) dominates Tier 3 (optimization)
                combined_score = (
                    tier2_scale * tier2_scores[i] + tier3_scores[i]
                )
                objective_terms.append(combined_score * x_vars[i])

            solver.Maximize(solver.Sum(objective_terms))
            status = solver.Solve()

            decisions: List[InductionDecision] = []
            
            if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                # Determine which variables the solver actually selected
                chosen_indices = [
                    i for i, var in x_vars.items() if var.solution_value() > 0.5
                ]

                # Safety: if the solver returns a feasible solution but selects
                # no trains at all (all x_i = 0), fall back to the deterministic
                # tiered scoring path so we still get a meaningful ranked split
                # of INDUCT / STANDBY / MAINTENANCE.
                if not chosen_indices:
                    logger.warning(
                        "OR-Tools returned a feasible solution with no inducted trains; "
                        "falling back to scoring-based tiered optimization."
                    )
                    return await self._optimize_with_tiered_scoring(
                        eligible_trainsets, critical_failures, request
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
                        
                        # Use the composite score from explanation (properly normalized 0-1)
                        composite_score = explanation.get("score", 0.0)
                        
                        decisions.append(
                            InductionDecision(
                                trainset_id=t["trainset_id"],
                                decision="INDUCT",
                                confidence_score=confidence,
                                reasons=reasons + explanation.get("top_reasons", []),
                                score=composite_score,  # Use explanation composite score
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
                            composite_score = explanation.get("score", 0.0)
                            decisions.append(
                                InductionDecision(
                                    trainset_id=t["trainset_id"],
                                    decision="MAINTENANCE",
                                    confidence_score=0.9,
                                    reasons=["Maintenance required - not selected for service"] + explanation.get("top_reasons", []),
                                    score=composite_score,
                                    top_reasons=explanation.get("top_reasons", []),
                                    top_risks=explanation.get("top_risks", []),
                                    violations=explanation.get("violations", []),
                                    shap_values=explanation.get("shap_values", [])
                                )
                            )
                        else:
                            explanation = generate_comprehensive_explanation(t, "STANDBY")
                            composite_score = explanation.get("score", 0.0)
                            decisions.append(
                                InductionDecision(
                                    trainset_id=t["trainset_id"],
                                    decision="STANDBY",
                                    confidence_score=0.7,
                                    reasons=["Standby - lower tiered score than inducted trainsets"] + explanation.get("top_reasons", []),
                                    score=composite_score,
                                    top_reasons=explanation.get("top_reasons", []),
                                    top_risks=explanation.get("top_risks", []),
                                    violations=explanation.get("violations", []),
                                    shap_values=explanation.get("shap_values", [])
                                )
                            )
                
                # Process critical failures (maintenance required)
                # TIER 1: Hard Stop - Score = 0, Status = MAINTENANCE
                for t in critical_failures:
                    explanation = generate_comprehensive_explanation(t, "MAINTENANCE")
                    
                    # Build detailed reason for critical failure
                    failure_reasons = ["Critical failure detected - requires maintenance"]
                    job_cards = t.get("job_cards", {})
                    critical_cards = self._normalize_to_int(job_cards.get("critical_cards"), 0)
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
                            score=0.0,  # TIER 1: Hard stop - Score = 0
                            top_reasons=explanation["top_reasons"],
                            top_risks=explanation["top_risks"],
                            violations=explanation["violations"],
                            shap_values=explanation["shap_values"]
                        )
                    )
            else:
                # Fallback if solver fails
                return await self._optimize_with_tiered_scoring(eligible_trainsets, critical_failures, request)

            logger.info(f"Tiered optimization completed: {len([d for d in decisions if d.decision=='INDUCT'])} inducted")
            
            # FINAL SAFETY SANITY CHECK - Critical safety bug fix
            for decision in decisions:
                if decision.decision == "INDUCT":
                    # Double-check that no inducted train has critical failures
                    trainset_id = decision.trainset_id
                    
                    # Find the original trainset data
                    original_trainset = None
                    for ts in trainsets:
                        if ts.get("trainset_id") == trainset_id:
                            original_trainset = ts
                            break
                    
                    if original_trainset:
                        # Re-check critical failure conditions
                        job_cards = original_trainset.get("job_cards", {})
                        critical_cards = self._normalize_to_int(job_cards.get("critical_cards"), 0)
                        
                        # Check for expired certificates
                        fitness_certs = original_trainset.get("fitness_certificates", {})
                        has_expired = False
                        if isinstance(fitness_certs, dict):
                            for cert_type, cert_data in fitness_certs.items():
                                if isinstance(cert_data, dict) and str(cert_data.get("status", "")).upper() == "EXPIRED":
                                    has_expired = True
                                    break
                        
                        # If critical failure detected, raise safety violation
                        if critical_cards > 0:
                            error_msg = f"SAFETY VIOLATION: Train {trainset_id} with {critical_cards} critical job cards made it to final INDUCT list!"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                        
                        if has_expired:
                            error_msg = f"SAFETY VIOLATION: Train {trainset_id} with expired certificate made it to final INDUCT list!"
                            logger.error(error_msg)
                            raise ValueError(error_msg)
            
            logger.info("Safety sanity check passed - all inducted trains meet Tier 1 constraints")
            
            # POST-PROCESSING: Final Safety Gate + Strict Sorting
            decisions = self._apply_final_safety_gate_and_sort(decisions)
            
            return decisions
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}", exc_info=True)
            raise
    
    def _normalize_to_int(self, value: Any, default: int = 0) -> int:
        """Safely normalize value to integer, handling strings and edge cases"""
        if value is None:
            return default
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                # Strip whitespace and convert
                cleaned = value.strip()
                if not cleaned:
                    return default
                return int(float(cleaned))  # Handle "2.0" strings
            except (ValueError, AttributeError):
                logger.warning(f"Failed to convert '{value}' to int, using default {default}")
                return default
        # For other types, try conversion
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Failed to convert {type(value)} value '{value}' to int, using default {default}")
            return default
    
    def _has_critical_failure(self, trainset: Dict[str, Any]) -> bool:
        """TIER 1: Check if trainset has critical failure (hard constraint exclusion).
        
        Returns True if trainset should be excluded due to:
        - Expired fitness certificates
        - Critical job cards
        - Mileage limit exceeded
        - Currently in maintenance
        """
        # Check fitness certificates - strict validation
        fitness_certs = trainset.get("fitness_certificates", {})
        if not isinstance(fitness_certs, dict):
            fitness_certs = {}
        
        trainset_id = trainset.get("trainset_id", "UNKNOWN")
        for cert_type, cert_data in fitness_certs.items():
            if not isinstance(cert_data, dict):
                continue
            cert_status = str(cert_data.get("status", "")).upper()
            if cert_status == "EXPIRED":
                logger.warning(f"SAFETY FILTER: {trainset_id} excluded - EXPIRED certificate: {cert_type}")
                return True  # Critical: expired certificate
        
        # Check critical job cards - STRICT INTEGER COMPARISON (safety bug fix)
        job_cards = trainset.get("job_cards", {})
        if not isinstance(job_cards, dict):
            job_cards = {}
        
        critical_cards = self._normalize_to_int(job_cards.get("critical_cards"), 0)
        trainset_id = trainset.get("trainset_id", "UNKNOWN")
        
        if critical_cards > 0:
            logger.warning(f"SAFETY FILTER: {trainset_id} excluded - {critical_cards} CRITICAL job cards")
            return True  # Critical: open critical job cards
        
        # Check mileage limits
        current_mileage = trainset.get("current_mileage", 0.0)
        max_mileage = trainset.get("max_mileage_before_maintenance", float('inf'))
        if current_mileage >= max_mileage:
            return True  # Critical: mileage limit exceeded
        
        # Check maintenance status
        if trainset.get("status") == "MAINTENANCE":
            return True  # Critical: currently in maintenance
        
        # Check cleaning slot requirement (if required but not available)
        requires_cleaning = bool(trainset.get("requires_cleaning", False))
        has_cleaning_slot = bool(trainset.get("has_cleaning_slot", False))
        if requires_cleaning and not has_cleaning_slot:
            return True  # Critical: cleaning required but slot not available
        
        return False
    
    def _calculate_tier2_score(self, trainset: Dict[str, Any]) -> float:
        """TIER 2: Calculate high priority soft objectives score.
        
        Components:
        - BRANDING_OBLIGATION: Wrapped trains must enter service (high positive score)
        - MINOR_DEFECT_PENALTY: Prefer keeping trains with minor defects in depot (penalty)
        """
        score = 0.0
        
        # Branding obligation: wrapped trains with active advertiser get high priority
        branding = trainset.get("branding", {})
        if isinstance(branding, dict):
            advertiser = branding.get("current_advertiser")
            priority = branding.get("priority", "LOW")
            
            # If train is wrapped (has active advertiser) and priority is HIGH/MEDIUM, strong obligation
            if advertiser and advertiser != "None" and advertiser != "":
                if priority == "HIGH":
                    score += self.weights["BRANDING_OBLIGATION"]  # Full obligation weight
                elif priority == "MEDIUM":
                    score += self.weights["BRANDING_OBLIGATION"] * 0.6  # Partial obligation
                elif priority == "LOW":
                    score += self.weights["BRANDING_OBLIGATION"] * 0.3  # Weak obligation
        
        # Minor defect penalty: -20 points per minor defect (Tier 2)
        job_cards = trainset.get("job_cards", {})
        open_cards = self._normalize_to_int(job_cards.get("open_cards"), 0)
        critical_cards = self._normalize_to_int(job_cards.get("critical_cards"), 0)
        minor_cards = max(0, open_cards - critical_cards)
        
        if minor_cards > 0:
            # Exact weight: -20 per defect
            penalty = self.weights["MINOR_DEFECT_PENALTY_PER_DEFECT"] * minor_cards
            score += penalty
        
        return score
    
    def _calculate_tier3_score(self, trainset: Dict[str, Any]) -> float:
        """TIER 3: Calculate optimization soft objectives score.
        
        Components:
        - MILEAGE_BALANCING: Prefer trains that need mileage to equalize fleet
        - CLEANING_DUE_PENALTY: Penalty if train is due for deep cleaning
        - SHUNTING_COMPLEXITY_PENALTY: Minor penalty if train is blocked by others
        """
        score = 0.0
        
        # Mileage balancing: prefer trains with lower recent mileage (need to equalize fleet)
        current_mileage = trainset.get("current_mileage", 0.0)
        max_mileage = trainset.get("max_mileage_before_maintenance", 50000.0)
        
        # Get 30-day mileage if available, otherwise use current mileage ratio
        km_30d = trainset.get("km_30d", current_mileage * 0.1)  # Approximate if not available
        km_30d_norm = min(1.0, km_30d / 5000.0) if km_30d else 0.5  # Normalize to 0-1
        
        # Mileage balancing: +100 if below average (exact weight from requirements)
        # Lower mileage = higher score (train needs mileage to catch up)
        if km_30d_norm < 0.5:  # Below average mileage
            score += self.weights["MILEAGE_BALANCING"]  # +100 points
        else:
            # Partial credit for moderate mileage
            score += self.weights["MILEAGE_BALANCING"] * (1.0 - km_30d_norm)
        
        # Cleaning due penalty: if train is due for cleaning, apply penalty
        requires_cleaning = bool(trainset.get("requires_cleaning", False))
        cleaning_due_date = trainset.get("cleaning_due_date")
        
        if requires_cleaning and cleaning_due_date:
            try:
                if isinstance(cleaning_due_date, str):
                    due_date = datetime.fromisoformat(cleaning_due_date.replace('Z', '+00:00'))
                else:
                    due_date = cleaning_due_date
                
                days_until_due = (due_date - datetime.utcnow()).days
                if days_until_due <= 7:  # Due within 7 days
                    penalty_factor = 1.0 - (days_until_due / 7.0)
                    score += self.weights["CLEANING_DUE_PENALTY"] * penalty_factor
            except (ValueError, TypeError):
                # If date parsing fails, apply small penalty if cleaning is required
                score += self.weights["CLEANING_DUE_PENALTY"] * 0.5
        elif requires_cleaning:
            # Cleaning required but no date info - apply small penalty
            score += self.weights["CLEANING_DUE_PENALTY"] * 0.3
        
        # Shunting complexity penalty: if train is blocked by others or in complex position
        is_blocked = bool(trainset.get("is_blocked", False))
        shunt_complexity = trainset.get("shunt_complexity", 0.0)  # 0-1 scale
        
        if is_blocked or shunt_complexity > 0.5:
            complexity_factor = 1.0 if is_blocked else shunt_complexity
            score += self.weights["SHUNTING_COMPLEXITY_PENALTY"] * complexity_factor
        
        # ML health contribution: higher health should increase Tier‑3 score so that
        # trains with similar mileage but better ML health rank higher.
        health_score = float(trainset.get("ml_health_score", 0.85) or 0.85)
        score += health_score * 100.0  # explicit +100 * health weight as requested
        
        return score
    
    def _calculate_optimization_score(self, trainset: Dict[str, Any], request: OptimizationRequest, weights: OptimizationWeights) -> float:
        """Calculate multi-objective optimization score"""
        score = 0.0
        
        # Readiness score (fitness + maintenance status)
        readiness_score = self._readiness_score(trainset)
        score += readiness_score * weights.readiness
        
        # Reliability score (sensor health + ML risk)
        reliability_base = self._reliability_score(trainset)
        # Incorporate ML risk prediction
        predicted_risk = float(trainset.get("predicted_failure_risk", 0.2))
        reliability_score = reliability_base * (1.0 - predicted_risk)  # Lower risk = higher reliability
        score += reliability_score * weights.reliability
        
        # Branding priority
        branding_score = self._branding_score(trainset)
        score += branding_score * weights.branding
        
        # Mileage balance (prefer lower mileage for even distribution)
        mileage_score = 1 - (trainset["current_mileage"] / max(1.0, trainset["max_mileage_before_maintenance"]))
        score += mileage_score * weights.mileage_balance
        
        # Shunt cost (lower is better, so we use inverse of cost score)
        cost_score = self._cost_score(trainset)
        score += cost_score * weights.shunt
        
        return min(score, 1.0)

    async def _optimize_with_tiered_scoring(self, eligible_trainsets: List[Dict[str, Any]], critical_failures: List[Dict[str, Any]], request: OptimizationRequest) -> List[InductionDecision]:
        """Fallback tiered scoring-based optimization when OR-Tools is unavailable."""
        # Calculate tiered scores for all eligible trainsets
        scored_trainsets = []
        for trainset in eligible_trainsets:
            tier2_score = self._calculate_tier2_score(trainset)
            tier3_score = self._calculate_tier3_score(trainset)
            
            # Lexicographic: Tier 2 dominates Tier 3
            tier2_scale = 10000.0
            combined_score = tier2_scale * tier2_score + tier3_score
            
            scored_trainsets.append((trainset, tier2_score, tier3_score, combined_score))
        
        # Sort by combined score (Tier 2 dominates)
        scored_trainsets.sort(key=lambda x: x[3], reverse=True)
        
        target_inductions = min(request.required_service_hours, len(scored_trainsets))
        decisions: List[InductionDecision] = []
        inducted = 0
        
        # Process inducted trainsets
        for trainset, tier2_score, tier3_score, combined_score in scored_trainsets:
            if inducted < target_inductions:
                confidence = min(1.0, max(0.5, (combined_score + 5000) / 10000))
                explanation = generate_comprehensive_explanation(trainset, "INDUCT")
                reasons = self._get_tiered_induction_reasons(trainset, tier2_score, tier3_score)
                composite_score = explanation.get("score", 0.0)
                
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="INDUCT",
                    confidence_score=confidence,
                    reasons=reasons + explanation.get("top_reasons", []),
                    score=composite_score,  # Use explanation composite score
                    top_reasons=explanation.get("top_reasons", []),
                    top_risks=explanation.get("top_risks", []),
                    violations=explanation.get("violations", []),
                    shap_values=explanation.get("shap_values", [])
                ))
                inducted += 1
            elif self._needs_maintenance(trainset):
                explanation = generate_comprehensive_explanation(trainset, "MAINTENANCE")
                composite_score = explanation.get("score", 0.0)
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="MAINTENANCE",
                    confidence_score=0.9,
                    reasons=["Maintenance required - not selected for service"] + explanation.get("top_reasons", []),
                    score=composite_score,
                    top_reasons=explanation.get("top_reasons", []),
                    top_risks=explanation.get("top_risks", []),
                    violations=explanation.get("violations", []),
                    shap_values=explanation.get("shap_values", [])
                ))
            else:
                explanation = generate_comprehensive_explanation(trainset, "STANDBY")
                composite_score = explanation.get("score", 0.0)
                decisions.append(InductionDecision(
                    trainset_id=trainset["trainset_id"],
                    decision="STANDBY",
                    confidence_score=0.7,
                    reasons=["Standby - lower tiered score than inducted trainsets"] + explanation.get("top_reasons", []),
                    score=composite_score,
                    top_reasons=explanation.get("top_reasons", []),
                    top_risks=explanation.get("top_risks", []),
                    violations=explanation.get("violations", []),
                    shap_values=explanation.get("shap_values", [])
                ))
        
        # Process critical failures
        # TIER 1: Hard Stop - Score = 0, Status = MAINTENANCE
        for trainset in critical_failures:
            explanation = generate_comprehensive_explanation(trainset, "MAINTENANCE")
            
            # Build detailed reason for critical failure
            failure_reasons = ["Critical failure detected - requires maintenance"]
            job_cards = trainset.get("job_cards", {})
            critical_cards = self._normalize_to_int(job_cards.get("critical_cards"), 0)
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
                score=0.0,  # TIER 1: Hard stop - Score = 0
                top_reasons=explanation["top_reasons"],
                top_risks=explanation["top_risks"],
                violations=explanation["violations"],
                shap_values=explanation["shap_values"]
            ))
        
        # FINAL SAFETY SANITY CHECK (fallback method)
        for decision in decisions:
            if decision.decision == "INDUCT":
                trainset_id = decision.trainset_id
                original_trainset = next((ts for ts in eligible_trainsets if ts.get("trainset_id") == trainset_id), None)
                if original_trainset:
                    job_cards = original_trainset.get("job_cards", {})
                    critical_cards = self._normalize_to_int(job_cards.get("critical_cards"), 0)
                    if critical_cards > 0:
                        error_msg = f"SAFETY VIOLATION (fallback): Train {trainset_id} with {critical_cards} critical job cards made it to final INDUCT list!"
                        logger.error(error_msg)
                        raise ValueError(error_msg)
        
        # POST-PROCESSING: Final Safety Gate + Strict Sorting
        decisions = self._apply_final_safety_gate_and_sort(decisions)
        
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
    
    def _get_tiered_induction_reasons(self, trainset: Dict[str, Any], tier2_score: float, tier3_score: float) -> List[str]:
        """Generate human-readable reasons based on tiered scoring"""
        reasons = []
        
        # Tier 2 reasons (high priority)
        branding = trainset.get("branding", {})
        if isinstance(branding, dict):
            advertiser = branding.get("current_advertiser")
            priority = branding.get("priority", "LOW")
            if advertiser and advertiser != "None" and advertiser != "":
                if priority == "HIGH":
                    reasons.append("High branding obligation - wrapped train must enter service")
                elif priority == "MEDIUM":
                    reasons.append("Medium branding obligation - wrapped train")
                elif priority == "LOW":
                    reasons.append("Branding obligation - wrapped train")
        
        job_cards = trainset.get("job_cards", {})
        minor_cards = max(0, job_cards.get("open_cards", 0) - job_cards.get("critical_cards", 0))
        if minor_cards == 0:
            reasons.append("No minor defects - optimal condition")
        
        # Tier 3 reasons (optimization)
        current_mileage = trainset.get("current_mileage", 0.0)
        max_mileage = trainset.get("max_mileage_before_maintenance", 50000.0)
        mileage_ratio = current_mileage / max_mileage if max_mileage > 0 else 0.0
        
        if mileage_ratio < 0.5:
            reasons.append("Low mileage - good for fleet balancing")
        elif mileage_ratio < 0.7:
            reasons.append("Moderate mileage - suitable for service")
        
        if not trainset.get("requires_cleaning", False):
            reasons.append("No cleaning required - ready for service")
        
        if not trainset.get("is_blocked", False):
            reasons.append("No shunting complexity - easy deployment")
        
        # ML health alert (Tier‑3 influence)
        ml_health = trainset.get("ml_health_score")
        if isinstance(ml_health, (int, float)) and ml_health < 0.5:
            reasons.append("ML Alert: Component fatigue detected")

        return reasons if reasons else ["Selected based on tiered optimization criteria"]
