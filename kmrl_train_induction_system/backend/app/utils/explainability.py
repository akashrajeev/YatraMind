# backend/app/utils/explainability.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import numpy as np
from datetime import datetime, timedelta
from app.models.trainset import ShapFeature
from app.utils.normalization import normalize_to_int

# Default weights matching optimizer.py
WEIGHTS = {
    "BRANDING_OBLIGATION": 250.0,
    "MINOR_DEFECT_PENALTY_PER_DEFECT": -50.0,
    "MILEAGE_BALANCING": 50.0,
    "CLEANING_DUE_PENALTY": -30.0,
    "SHUNTING_COMPLEXITY_PENALTY": -20.0
}

def _jinja_env() -> Environment:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    return Environment(
        loader=FileSystemLoader(base_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

def _parse_cleaning_date(date_str: Any) -> datetime | None:
    """Helper to safely parse cleaning due date."""
    if not date_str:
        return None
    if isinstance(date_str, datetime):
        return date_str
    try:
        return datetime.fromisoformat(str(date_str).replace('Z', '+00:00'))
    except ValueError:
        return None

def _calculate_tier2_score(trainset: Dict[str, Any]) -> float:
    """TIER 2: Calculate high priority soft objectives score (Matches optimizer.py)."""
    score = 0.0
    
    # Branding obligation
    branding = trainset.get("branding", {})
    if isinstance(branding, dict):
        advertiser = branding.get("current_advertiser")
        priority = branding.get("priority", "LOW")
        
        if advertiser and advertiser != "None" and advertiser != "":
            if priority == "HIGH":
                score += WEIGHTS["BRANDING_OBLIGATION"]
            elif priority == "MEDIUM":
                score += WEIGHTS["BRANDING_OBLIGATION"] * 0.6
            elif priority == "LOW":
                score += WEIGHTS["BRANDING_OBLIGATION"] * 0.3
    
    # Minor defect penalty
    job_cards = trainset.get("job_cards", {})
    open_cards = normalize_to_int(job_cards.get("open_cards"), 0)
    critical_cards = normalize_to_int(job_cards.get("critical_cards"), 0)
    minor_cards = max(0, open_cards - critical_cards)
    
    if minor_cards > 0:
        penalty = WEIGHTS["MINOR_DEFECT_PENALTY_PER_DEFECT"] * minor_cards
        score += penalty
    
    return score

def _calculate_tier3_score(trainset: Dict[str, Any]) -> float:
    """TIER 3: Calculate optimization soft objectives score (Matches optimizer.py)."""
    score = 0.0
    
    # Mileage balancing
    current_mileage = trainset.get("current_mileage", 0.0)
    km_30d = trainset.get("km_30d", current_mileage * 0.1)
    km_30d_norm = min(1.0, km_30d / 5000.0) if km_30d else 0.5
    
    if km_30d_norm < 0.5:
        score += WEIGHTS["MILEAGE_BALANCING"]
    else:
        score += WEIGHTS["MILEAGE_BALANCING"] * (1.0 - km_30d_norm)
    
    # Cleaning due penalty
    requires_cleaning = bool(trainset.get("requires_cleaning", False))
    cleaning_due_date = trainset.get("cleaning_due_date")
    
    if requires_cleaning and cleaning_due_date:
        try:
            due_date = _parse_cleaning_date(cleaning_due_date)
            if due_date:
                days_until_due = (due_date - datetime.utcnow()).days
                if days_until_due <= 7:
                    penalty_factor = 1.0 - (days_until_due / 7.0)
                    score += WEIGHTS["CLEANING_DUE_PENALTY"] * penalty_factor
            else:
                score += WEIGHTS["CLEANING_DUE_PENALTY"] * 0.5
        except Exception:
            score += WEIGHTS["CLEANING_DUE_PENALTY"] * 0.5
    elif requires_cleaning:
        score += WEIGHTS["CLEANING_DUE_PENALTY"] * 0.3
    
    # Shunting complexity penalty
    is_blocked = bool(trainset.get("is_blocked", False))
    shunt_complexity = trainset.get("shunt_complexity", 0.0)
    
    if is_blocked or shunt_complexity > 0.5:
        complexity_factor = 1.0 if is_blocked else shunt_complexity
        score += WEIGHTS["SHUNTING_COMPLEXITY_PENALTY"] * complexity_factor
    
    # ML health contribution
    health_score = float(trainset.get("ml_health_score", 0.85) or 0.85)
    score += health_score * 100.0
    
    return score

def calculate_composite_score(trainset: Dict[str, Any], decision: str) -> float:
    """Calculate composite score based on Tiered Constraint Hierarchy (Matches optimizer.py)"""
    
    # Tier 1: Safety Compliance (binary - passed or failed)
    safety_score = _calculate_safety_compliance_score(trainset)
    
    tier2_score = _calculate_tier2_score(trainset)
    tier3_score = _calculate_tier3_score(trainset)
    
    # Normalize to 0-1 range for UI display (Matches optimizer.py formula)
    # Tier 2 is weighted 10x Tier 3
    combined = (tier2_score * 10.0) + tier3_score
    # Expanded normalization range to ensure eligible but low-scoring trains (Standby)
    # have a non-zero score (e.g., ~5-10%) instead of being clamped to 0.
    # Range: -3000 to +3000 maps to 0.0 to 1.0
    normalized = min(1.0, max(0.0, (combined + 3000) / 6000)) 
    
    if safety_score < 1.0:
        return 0.0
    
    return normalized


def _calculate_safety_compliance_score(trainset: Dict[str, Any]) -> float:
    """TIER 1: Calculate Safety Compliance score (perfect 1.0 if passed hard constraints)"""
    # Check fitness certificates
    fitness_certs = trainset.get("fitness_certificates", {})
    if not isinstance(fitness_certs, dict):
        fitness_certs = {}
    
    for cert_data in fitness_certs.values():
        if isinstance(cert_data, dict):
            if str(cert_data.get("status", "")).upper() == "EXPIRED":
                return 0.0
        elif isinstance(cert_data, str):
            if cert_data.upper() == "EXPIRED":
                return 0.0
    
    # Check critical job cards
    job_cards = trainset.get("job_cards", {})
    critical_cards = normalize_to_int(job_cards.get("critical_cards"), 0)
    
    if critical_cards > 0:
        return 0.0
    
    # Check mileage limits
    current_mileage = trainset.get("current_mileage", 0.0)
    max_mileage = trainset.get("max_mileage_before_maintenance", float('inf'))
    if max_mileage > 0 and current_mileage >= max_mileage:
        return 0.0
    
    # Check maintenance status
    if trainset.get("status") == "MAINTENANCE":
        return 0.0
    
    return 1.0


def top_reasons_and_risks(trainset: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate top reasons and risks based on Tiered Constraint Hierarchy"""
    reasons: List[str] = []
    risks: List[str] = []

    # TIER 1: Safety Compliance reasons
    safety_score = _calculate_safety_compliance_score(trainset)
    if safety_score == 1.0:
        reasons.append("Safety compliance verified - all certificates valid")
        
        fitness_certs = trainset.get("fitness_certificates", {})
        if all(c.get("status") == "VALID" for c in fitness_certs.values()):
            reasons.append("All department certificates valid and current")
        
        job_cards = trainset.get("job_cards", {})
        if normalize_to_int(job_cards.get("critical_cards"), 0) == 0:
            reasons.append("No critical job cards pending")
    else:
        fitness_certs = trainset.get("fitness_certificates", {})
        for dept, cert in fitness_certs.items():
            if cert.get("status") == "EXPIRED":
                risks.append(f"{dept} certificate expired - safety critical failure")
            elif cert.get("status") == "EXPIRING_SOON":
                risks.append(f"{dept} certificate expiring soon")
        
        if normalize_to_int(trainset.get("job_cards", {}).get("critical_cards"), 0) > 0:
            risks.append("Open critical job-card - safety compliance failure")

    # TIER 2: Branding Obligation reasons
    branding = trainset.get("branding", {})
    if isinstance(branding, dict):
        advertiser = branding.get("current_advertiser")
        priority = branding.get("priority", "LOW")
        if advertiser and advertiser not in (None, "None", ""):
            if priority == "HIGH":
                reasons.insert(0, "CRITICAL: Mandatory branding commitment - Overrides minor defects and mileage")
            elif priority == "MEDIUM":
                reasons.append("Branding obligation (Medium Priority) - wrapped train")
            elif priority == "LOW":
                reasons.append("Branding commitment (Low Priority) - wrapped train")
    
    # TIER 2: Defect Impact reasons
    job_cards = trainset.get("job_cards", {})
    open_cards = normalize_to_int(job_cards.get("open_cards"), 0)
    critical_cards = normalize_to_int(job_cards.get("critical_cards"), 0)
    minor_cards = max(0, open_cards - critical_cards)
    
    if minor_cards == 0:
        reasons.append("No minor defects - optimal condition for service")
    else:
        # Include ALL pending job cards as risks/warnings as requested
        risks.append(f"{minor_cards} minor job cards pending - requires attention")
        if minor_cards > 5:
            risks.append("High number of minor defects - prefer depot maintenance")

    # TIER 3: Mileage Optimization reasons
    # Use helper to get score for reasoning
    mileage_score = 0.0
    current_mileage = trainset.get("current_mileage", 0.0)
    km_30d = trainset.get("km_30d", current_mileage * 0.1)
    km_30d_norm = min(1.0, km_30d / 5000.0) if km_30d else 0.5
    
    if km_30d_norm < 0.5:
        mileage_score = 1.0 # Good
    else:
        mileage_score = 1.0 - km_30d_norm
        
    if mileage_score > 0.8:
        reasons.append("Low mileage (Secondary Factor) - good for fleet balancing")
    elif mileage_score > 0.6:
        reasons.append("Below-average mileage (Secondary Factor) - helps equalize fleet usage")
    elif mileage_score < 0.4:
        risks.append("High mileage (Secondary Factor) - lower priority but acceptable")

    return {
        "top_reasons": reasons,
        "top_risks": risks,
    }


def generate_shap_values(trainset: Dict[str, Any], ml_features: List[str] = None) -> List[ShapFeature]:
    """Generate SHAP values for top features based on Tiered Constraint Hierarchy"""
    shap_features = []
    
    # TIER 1: Safety Compliance
    safety_score = _calculate_safety_compliance_score(trainset)
    if safety_score < 1.0:
        shap_features.append(ShapFeature(name="Safety Compliance", value=1.0, impact="negative"))
    else:
        shap_features.append(ShapFeature(name="Safety Compliance", value=1.0, impact="positive"))
    
    # TIER 2: Branding
    branding = trainset.get("branding", {})
    if isinstance(branding, dict) and branding.get("current_advertiser") not in (None, "None", ""):
        priority = branding.get("priority", "LOW")
        branding_score = 1.0 if priority == "HIGH" else 0.8 if priority == "MEDIUM" else 0.6
        shap_features.append(ShapFeature(name="Branding Obligation", value=branding_score, impact="positive"))
    else:
        # No branding is neutral/slight negative in a competitive list? 
        # Actually, let's keep it neutral or omitted if 0.
        # But user asked "why are some factors zero". 
        # If we explicitly show it as 0.0, it answers "No branding".
        shap_features.append(ShapFeature(name="Branding Obligation", value=0.0, impact="neutral"))

    # TIER 2: Defect Impact
    job_cards = trainset.get("job_cards", {})
    open_cards = normalize_to_int(job_cards.get("open_cards"), 0)
    critical_cards = normalize_to_int(job_cards.get("critical_cards"), 0)
    minor_cards = max(0, open_cards - critical_cards)
    
    if minor_cards > 0:
        # Normalize impact for display (0 to 1 scale)
        impact = min(1.0, minor_cards / 5.0) # 5 defects = max visual negative impact
        shap_features.append(ShapFeature(name="Defect Impact", value=impact, impact="negative"))
    else:
        # No defects = Positive Impact!
        shap_features.append(ShapFeature(name="Defect Impact", value=1.0, impact="positive"))
    
    # TIER 3: Mileage
    current_mileage = trainset.get("current_mileage", 0.0)
    km_30d = trainset.get("km_30d", current_mileage * 0.1)
    km_30d_norm = min(1.0, km_30d / 5000.0) if km_30d else 0.5
    mileage_val = 1.0 - km_30d_norm
    
    if mileage_val > 0.6:
        # Low mileage = Positive
        shap_features.append(ShapFeature(name="Mileage Optimization", value=mileage_val, impact="positive"))
    elif mileage_val < 0.4:
        # High mileage = Negative
        # Show magnitude of "badness" (1.0 - val)
        shap_features.append(ShapFeature(name="Mileage Optimization", value=(1.0 - mileage_val), impact="negative"))
    else:
        # Average mileage = Neutral
        shap_features.append(ShapFeature(name="Mileage Optimization", value=0.5, impact="neutral"))
    
    # Sort by importance
    def sort_key(f: ShapFeature) -> tuple:
        tier_priority = {
            "Safety Compliance": 0,
            "Branding Obligation": 1,
            "Defect Impact": 2,
            "Mileage Optimization": 3
        }.get(f.name, 99)
        return (tier_priority, -abs(f.value) if f.impact == "negative" else abs(f.value))
    
    shap_features.sort(key=sort_key)
    return shap_features


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
    if normalize_to_int(job_cards.get("critical_cards"), 0) > 0:
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
