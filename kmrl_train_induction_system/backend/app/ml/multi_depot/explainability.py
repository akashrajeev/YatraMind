# backend/app/ml/multi_depot/explainability.py
"""
Explainability & Safety for AI Decisions
SHAP/local explainers for all ML/decision outputs
Hard constraints enforced at runtime
"""
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import shap
import logging

logger = logging.getLogger(__name__)


class AIExplainability:
    """Explainability service for AI decisions"""
    
    def __init__(self):
        pass
    
    def explain_service_selection(self, model, features: np.ndarray,
                                 feature_names: List[str], score: float) -> Dict[str, Any]:
        """Explain service selection decision using SHAP"""
        try:
            if model is None:
                return self._fallback_explanation(features, feature_names, score)
            
            # Create SHAP explainer
            def predict_fn(x):
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    outputs = model(x_tensor)
                    return outputs.numpy()
            
            explainer = shap.Explainer(predict_fn, feature_names=feature_names)
            shap_values = explainer(np.array([features]), max_evals=100)
            
            # Top contributing features
            shap_vals = shap_values.values[0, :, 0]
            top_indices = np.argsort(-np.abs(shap_vals))[:5]
            
            top_factors = [
                {
                    "feature": feature_names[i],
                    "value": float(features[i]),
                    "shap_value": float(shap_vals[i]),
                    "impact": "increases_score" if shap_vals[i] > 0 else "decreases_score",
                }
                for i in top_indices
            ]
            
            # Generate explanation text
            positive_factors = [f for f in top_factors if f["impact"] == "increases_score"][:3]
            explanation = f"Selected for service (score: {score:.2f}). "
            if positive_factors:
                factor_names = [f["feature"].replace("_", " ").title() for f in positive_factors]
                explanation += f"Top factors: {', '.join(factor_names)}."
            
            return {
                "score": score,
                "top_factors": top_factors,
                "explanation": explanation,
            }
            
        except Exception as e:
            logger.error(f"Error explaining service selection: {e}")
            return self._fallback_explanation(features, feature_names, score)
    
    def explain_stabling_allocation(self, depot_id: str, location_type: str,
                                   bay_id: Optional[int], factors: Dict[str, float]) -> Dict[str, Any]:
        """Explain stabling allocation decision"""
        location = f"{depot_id} {location_type}"
        if bay_id:
            location += f" Bay {bay_id}"
        
        explanation_parts = []
        
        # Risk-based
        if factors.get("risk_score", 0) > 0.3:
            explanation_parts.append(f"High risk train ({factors['risk_score']:.1%}) requires maintenance-ready location")
        
        # Turnout time
        if factors.get("turnout_time", 0) < 8:
            explanation_parts.append(f"Fast turnout ({factors['turnout_time']:.1f} min) enables rapid deployment")
        
        # Branding
        if factors.get("branding_priority", 0) > 0.5:
            explanation_parts.append(f"High branding priority ({factors['branding_priority']:.0%}) prioritizes service location")
        
        explanation = f"Allocated to {location}. " + "; ".join(explanation_parts[:3])
        
        return {
            "location": location,
            "factors": factors,
            "explanation": explanation,
        }
    
    def explain_transfer_decision(self, transfer_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Explain inter-depot transfer decision"""
        should_transfer = transfer_decision.get("should_transfer", False)
        prob = transfer_decision.get("transfer_probability", 0.0)
        cost = transfer_decision.get("cost_estimate", {})
        
        if should_transfer:
            explanation = f"Transfer recommended (probability: {prob:.1%}). "
            explanation += f"Expected benefit: {cost.get('expected_benefit', 0):.1f} vs cost: {cost.get('net_cost', 0):.1f}."
        else:
            explanation = f"Transfer not recommended (probability: {prob:.1%}). "
            explanation += f"Cost-benefit analysis: net cost {cost.get('net_cost', 0):.1f}."
        
        return {
            "should_transfer": should_transfer,
            "probability": prob,
            "explanation": explanation,
            "cost_estimate": cost,
        }
    
    def _fallback_explanation(self, features: np.ndarray, feature_names: List[str], 
                             score: float) -> Dict[str, Any]:
        """Fallback explanation when SHAP fails"""
        return {
            "score": score,
            "top_factors": [],
            "explanation": f"Service score: {score:.2f}. Based on heuristic analysis.",
        }


class SafetyGuard:
    """Safety guard for AI decisions - enforces hard constraints"""
    
    def __init__(self):
        pass
    
    def validate_service_decision(self, train_features: Dict[str, Any],
                                 decision: str) -> Tuple[bool, Optional[str]]:
        """
        Validate service decision against Tier-1 safety constraints
        
        Returns:
        - (is_valid, rejection_reason)
        """
        # Check fitness certificates
        fc = train_features.get("fitness_certificates", {})
        expired_certs = []
        for cert_name, cert_data in fc.items():
            if isinstance(cert_data, dict) and cert_data.get("status") == "EXPIRED":
                expired_certs.append(cert_name)
        
        if expired_certs:
            return False, f"Expired fitness certificates: {', '.join(expired_certs)}"
        
        # Check critical job cards
        job_cards = train_features.get("job_cards", {})
        if isinstance(job_cards, dict) and job_cards.get("critical_cards", 0) > 0:
            return False, f"Critical job cards open: {job_cards['critical_cards']}"
        
        # Check mileage
        current_mileage = train_features.get("current_mileage", 0)
        max_mileage = train_features.get("max_mileage_before_maintenance", 50000)
        if max_mileage > 0 and current_mileage > max_mileage:
            return False, f"Mileage exceeds maintenance limit: {current_mileage} > {max_mileage}"
        
        # Check maintenance status
        status = train_features.get("status", "")
        if status == "MAINTENANCE":
            return False, "Train currently in maintenance"
        
        return True, None
    
    def validate_stabling_allocation(self, depot_config: Dict[str, Any],
                                   bay_id: int, train_count: int) -> Tuple[bool, Optional[str]]:
        """Validate stabling allocation against capacity constraints"""
        total_bays = depot_config.get("total_bays", 0)
        if bay_id > total_bays:
            return False, f"Bay {bay_id} exceeds depot capacity ({total_bays} bays)"
        
        # Check if bay is already occupied (would need current state)
        # This is a simplified check
        
        return True, None
    
    def validate_transfer_decision(self, from_depot: str, to_depot: str,
                                 train_features: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate transfer decision"""
        # Check if train is in service (shouldn't transfer during service)
        status = train_features.get("status", "")
        if status == "ACTIVE":
            return False, "Cannot transfer train currently in service"
        
        return True, None


