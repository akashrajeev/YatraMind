# backend/app/utils/explainability.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import numpy as np
from datetime import datetime, timedelta
from app.models.trainset import ShapFeature


def _jinja_env() -> Environment:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    return Environment(
        loader=FileSystemLoader(base_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def calculate_composite_score(trainset: Dict[str, Any], decision: str) -> float:
    """Calculate composite score for assignment decision"""
    score = 0.0
    
    # Base fitness score (0-1)
    fitness_score = _calculate_fitness_score(trainset)
    score += fitness_score * 0.3
    
    # Reliability score based on sensor health and predicted failure risk
    reliability_score = _calculate_reliability_score(trainset)
    score += reliability_score * 0.25
    
    # Branding priority score
    branding_score = _calculate_branding_score(trainset)
    score += branding_score * 0.2
    
    # Mileage balance score (prefer trainsets with balanced usage)
    mileage_score = _calculate_mileage_balance_score(trainset)
    score += mileage_score * 0.15
    
    # Operational efficiency score
    efficiency_score = _calculate_operational_efficiency_score(trainset)
    score += efficiency_score * 0.1
    
    # Adjust based on decision type
    if decision == "INDUCT":
        score *= 1.0  # Full score for induction
    elif decision == "STANDBY":
        score *= 0.7  # Reduced score for standby
    elif decision == "MAINTENANCE":
        score *= 0.3  # Lowest score for maintenance
    
    return min(max(score, 0.0), 1.0)


def _calculate_fitness_score(trainset: Dict[str, Any]) -> float:
    """Calculate fitness score based on certificates and job cards"""
    score = 1.0
    
    # Check fitness certificates
    fitness_certs = trainset.get("fitness_certificates", {})
    for cert_type, cert_data in fitness_certs.items():
        if cert_data.get("status") == "EXPIRED":
            score -= 0.3
        elif cert_data.get("status") == "EXPIRING_SOON":
            score -= 0.1
    
    # Check job cards
    job_cards = trainset.get("job_cards", {})
    critical_cards = job_cards.get("critical_cards", 0)
    open_cards = job_cards.get("open_cards", 0)
    
    score -= critical_cards * 0.2
    score -= open_cards * 0.05
    
    return max(score, 0.0)


def _calculate_reliability_score(trainset: Dict[str, Any]) -> float:
    """Calculate reliability score based on sensor health and predicted failure risk"""
    sensor_health = trainset.get("sensor_health_score", 0.8)
    predicted_risk = trainset.get("predicted_failure_risk", 0.2)
    
    # Higher sensor health = higher reliability
    # Lower predicted risk = higher reliability
    reliability = (sensor_health + (1.0 - predicted_risk)) / 2.0
    return max(min(reliability, 1.0), 0.0)


def _calculate_branding_score(trainset: Dict[str, Any]) -> float:
    """Calculate branding priority score"""
    branding = trainset.get("branding", {})
    priority = branding.get("priority", "LOW")
    
    if priority == "HIGH":
        return 1.0
    elif priority == "MEDIUM":
        return 0.6
    else:
        return 0.3


def _calculate_mileage_balance_score(trainset: Dict[str, Any]) -> float:
    """Calculate mileage balance score (prefer balanced usage)"""
    current_mileage = trainset.get("current_mileage", 0.0)
    max_mileage = trainset.get("max_mileage_before_maintenance", 50000.0)
    
    if max_mileage == 0:
        return 0.5
    
    usage_ratio = current_mileage / max_mileage
    
    # Prefer moderate usage (not too low, not too high)
    if usage_ratio < 0.2:
        return 0.3  # Too low usage
    elif usage_ratio > 0.9:
        return 0.1  # Too high usage
    else:
        return 1.0 - abs(usage_ratio - 0.5) * 2  # Peak at 50% usage


def _calculate_operational_efficiency_score(trainset: Dict[str, Any]) -> float:
    """Calculate operational efficiency score"""
    score = 0.5  # Base score
    
    # Check if cleaning slot is available
    if trainset.get("has_cleaning_slot", False):
        score += 0.3
    
    # Check maintenance schedule compliance
    if trainset.get("maintenance_compliant", True):
        score += 0.2
    
    return min(score, 1.0)


def top_reasons_and_risks(trainset: Dict[str, Any]) -> Dict[str, List[str]]:
    reasons: List[str] = []
    risks: List[str] = []

    # Reasons
    if all(c.get("status") == "VALID" for c in trainset.get("fitness_certificates", {}).values()):
        reasons.append("All department certificates valid")
    risk = float(trainset.get("predicted_failure_risk", 0.2))
    if risk < 0.2:
        reasons.append("Low predicted failure probability")
    if trainset.get("has_cleaning_slot", False):
        reasons.append("Available cleaning slot before dawn")

    # Risks
    for dept, cert in (trainset.get("fitness_certificates") or {}).items():
        if cert.get("status") == "EXPIRING_SOON":
            risks.append(f"{dept} certificate expiring soon")
        if cert.get("status") == "EXPIRED":
            risks.append(f"{dept} certificate expired")
    if trainset.get("job_cards", {}).get("critical_cards", 0) > 0:
        risks.append("Open critical job-card")
    if risk >= 0.5:
        risks.append("High predicted failure probability")

    return {
        "top_reasons": reasons[:3],
        "top_risks": risks[:3],
    }


def generate_shap_values(trainset: Dict[str, Any], ml_features: List[str] = None) -> List[ShapFeature]:
    """Generate SHAP values for top features affecting the decision"""
    if not ml_features:
        ml_features = [
            "sensor_health_score", "predicted_failure_risk", "current_mileage",
            "branding_priority", "fitness_score", "maintenance_priority"
        ]
    
    shap_features = []
    
    # Simulate SHAP values based on trainset characteristics
    # In production, this would use actual SHAP explainer
    for feature in ml_features:
        if feature == "sensor_health_score":
            value = trainset.get("sensor_health_score", 0.8)
            impact = "positive" if value > 0.7 else "negative" if value < 0.5 else "neutral"
            shap_features.append(ShapFeature(name="Sensor Health Score", value=value, impact=impact))
        
        elif feature == "predicted_failure_risk":
            value = trainset.get("predicted_failure_risk", 0.2)
            impact = "negative" if value > 0.3 else "positive" if value < 0.1 else "neutral"
            shap_features.append(ShapFeature(name="Predicted Failure Risk", value=value, impact=impact))
        
        elif feature == "current_mileage":
            current = trainset.get("current_mileage", 0.0)
            max_mileage = trainset.get("max_mileage_before_maintenance", 50000.0)
            ratio = current / max_mileage if max_mileage > 0 else 0.0
            impact = "negative" if ratio > 0.8 else "positive" if 0.3 < ratio < 0.7 else "neutral"
            shap_features.append(ShapFeature(name="Mileage Usage Ratio", value=ratio, impact=impact))
        
        elif feature == "branding_priority":
            branding = trainset.get("branding", {})
            priority = branding.get("priority", "LOW")
            value = 1.0 if priority == "HIGH" else 0.6 if priority == "MEDIUM" else 0.3
            impact = "positive" if priority == "HIGH" else "neutral"
            shap_features.append(ShapFeature(name="Branding Priority", value=value, impact=impact))
        
        elif feature == "fitness_score":
            score = _calculate_fitness_score(trainset)
            impact = "positive" if score > 0.8 else "negative" if score < 0.5 else "neutral"
            shap_features.append(ShapFeature(name="Fitness Certificate Score", value=score, impact=impact))
        
        elif feature == "maintenance_priority":
            # Simulate maintenance priority based on job cards and mileage
            job_cards = trainset.get("job_cards", {})
            critical_cards = job_cards.get("critical_cards", 0)
            current_mileage = trainset.get("current_mileage", 0.0)
            max_mileage = trainset.get("max_mileage_before_maintenance", 50000.0)
            
            priority_score = 0.0
            if critical_cards > 0:
                priority_score += 0.5
            if max_mileage > 0 and current_mileage / max_mileage > 0.8:
                priority_score += 0.3
            
            impact = "negative" if priority_score > 0.5 else "positive" if priority_score < 0.2 else "neutral"
            shap_features.append(ShapFeature(name="Maintenance Priority", value=priority_score, impact=impact))
    
    # Sort by absolute value and return top 5
    shap_features.sort(key=lambda x: abs(x.value), reverse=True)
    return shap_features[:5]


def detect_rule_violations(trainset: Dict[str, Any], decision: str) -> List[str]:
    """Detect rule violations for a trainset assignment"""
    violations = []
    
    # Check fitness certificate violations
    fitness_certs = trainset.get("fitness_certificates", {})
    for cert_type, cert_data in fitness_certs.items():
        if cert_data.get("status") == "EXPIRED":
            violations.append(f"{cert_type} certificate expired")
        elif cert_data.get("status") == "EXPIRING_SOON":
            violations.append(f"{cert_type} certificate expiring soon")
    
    # Check job card violations
    job_cards = trainset.get("job_cards", {})
    if job_cards.get("critical_cards", 0) > 0:
        violations.append("Open critical job-card")
    
    # Check mileage violations
    current_mileage = trainset.get("current_mileage", 0.0)
    max_mileage = trainset.get("max_mileage_before_maintenance", 50000.0)
    if current_mileage >= max_mileage:
        violations.append("Mileage limit exceeded")
    
    # Check maintenance status violations
    if trainset.get("status") == "MAINTENANCE" and decision == "INDUCT":
        violations.append("Cannot induct trainset currently in maintenance")
    
    # Check cleaning slot violations
    if trainset.get("requires_cleaning", False) and not trainset.get("has_cleaning_slot", False):
        violations.append("No cleaning slot available before departure")
    
    return violations


def generate_comprehensive_explanation(trainset: Dict[str, Any], decision: str) -> Dict[str, Any]:
    """Generate comprehensive explanation for assignment decision"""
    
    # Calculate composite score
    score = calculate_composite_score(trainset, decision)
    
    # Get top reasons and risks
    reasons_risks = top_reasons_and_risks(trainset)
    
    # Detect rule violations
    violations = detect_rule_violations(trainset, decision)
    
    # Generate SHAP values
    shap_values = generate_shap_values(trainset)
    
    return {
        "score": score,
        "top_reasons": reasons_risks["top_reasons"],
        "top_risks": reasons_risks["top_risks"],
        "violations": violations,
        "shap_values": [feature.dict() for feature in shap_values]
    }


def render_explanation_html(context: Dict[str, Any]) -> str:
    env = _jinja_env()
    tpl = env.get_template("explanation.j2")
    return tpl.render(**context)


def render_explanation_text(context: Dict[str, Any]) -> str:
    """Render explanation as plain text for logging/API responses"""
    lines = []
    lines.append(f"Trainset {context.get('trainset_id', 'Unknown')} - Decision: {context.get('role', 'Unknown')}")
    lines.append(f"Composite Score: {context.get('score', 0.0):.3f}")
    
    if context.get('top_reasons'):
        lines.append("\nTop Reasons:")
        for reason in context['top_reasons']:
            lines.append(f"  • {reason}")
    
    if context.get('top_risks'):
        lines.append("\nTop Risks:")
        for risk in context['top_risks']:
            lines.append(f"  • {risk}")
    
    if context.get('violations'):
        lines.append("\nRule Violations:")
        for violation in context['violations']:
            lines.append(f"  • {violation}")
    
    if context.get('shap_values'):
        lines.append("\nFeature Attributions:")
        for feature in context['shap_values']:
            impact_symbol = "↑" if feature.get('impact') == 'positive' else "↓" if feature.get('impact') == 'negative' else "→"
            lines.append(f"  • {feature.get('name', 'Unknown')}: {feature.get('value', 0.0):.4f} {impact_symbol}")
    
    return "\n".join(lines)


