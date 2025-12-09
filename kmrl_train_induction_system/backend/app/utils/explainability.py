# backend/app/utils/explainability.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os
import time
import asyncio
import numpy as np
from datetime import datetime, timedelta
from app.models.trainset import ShapFeature
import httpx
import json
import logging

logger = logging.getLogger(__name__)


def _jinja_env() -> Environment:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    return Environment(
        loader=FileSystemLoader(base_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def calculate_composite_score(trainset: Dict[str, Any], decision: str) -> float:
    """Calculate composite score based on Tiered Constraint Hierarchy
    
    Uses tiered scoring aligned with the new optimization model:
    - Tier 1: Safety Compliance (binary - passed or failed)
    - Tier 2: Branding Obligation + Defect Impact
    - Tier 3: Mileage Optimization
    """
    score = 0.0
    
    # Tier 1: Safety Compliance (perfect score if passed hard constraints)
    safety_score = _calculate_safety_compliance_score(trainset)
    if safety_score < 1.0:
        # If safety check failed, return low score immediately
        return 0.0
    
    # Tier 2: Branding Obligation (high priority)
    branding_score = _calculate_branding_obligation_score(trainset)
    score += branding_score * 0.5  # 50% weight for Tier 2
    
    # Tier 2: Defect Impact (penalty for minor defects)
    defect_impact = _calculate_defect_impact_score(trainset)
    score += defect_impact * 0.3  # 30% weight for defect impact
    
    # Tier 3: Mileage Optimization
    mileage_score = _calculate_mileage_optimization_score(trainset)
    score += mileage_score * 0.2  # 20% weight for Tier 3
    
    # Adjust based on decision type
    if decision == "INDUCT":
        score *= 1.0  # Full score for induction
    elif decision == "STANDBY":
        score *= 0.7  # Reduced score for standby
    elif decision == "MAINTENANCE":
        score *= 0.3  # Lowest score for maintenance
    
    return min(max(score, 0.0), 1.0)


def _calculate_safety_compliance_score(trainset: Dict[str, Any]) -> float:
    """TIER 1: Calculate Safety Compliance score (perfect 1.0 if passed hard constraints)
    
    Returns 1.0 if all hard constraints are met:
    - All fitness certificates valid
    - No critical job cards
    - Mileage within limits
    - Not in maintenance status
    Returns 0.0 if any critical failure exists
    """
    # Check fitness certificates - handle both dict and nested dict structures
    fitness_certs = trainset.get("fitness_certificates", {})
    if not isinstance(fitness_certs, dict):
        fitness_certs = {}
    
    for cert_type, cert_data in fitness_certs.items():
        if isinstance(cert_data, dict):
            cert_status = str(cert_data.get("status", "")).upper()
            if cert_status == "EXPIRED":
                return 0.0  # Critical failure
        elif isinstance(cert_data, str):
            if cert_data.upper() == "EXPIRED":
                return 0.0  # Critical failure
    
    # Check critical job cards - handle both dict and direct access
    job_cards = trainset.get("job_cards", {})
    if not isinstance(job_cards, dict):
        job_cards = {}
    
    critical_cards = job_cards.get("critical_cards", 0)
    if isinstance(critical_cards, str):
        try:
            critical_cards = int(float(critical_cards))
        except (ValueError, TypeError):
            critical_cards = 0
    
    if critical_cards > 0:
        return 0.0  # Critical failure
    
    # Check mileage limits
    current_mileage = trainset.get("current_mileage", 0.0)
    max_mileage = trainset.get("max_mileage_before_maintenance", float('inf'))
    if max_mileage > 0 and current_mileage >= max_mileage:
        return 0.0  # Critical failure
    
    # Check maintenance status
    if trainset.get("status") == "MAINTENANCE":
        return 0.0  # Critical failure
    
    # All hard constraints passed
    return 1.0


def _calculate_branding_obligation_score(trainset: Dict[str, Any]) -> float:
    """TIER 2: Calculate Branding Obligation score (high score if has active ad-wrap requirement)
    
    Returns 0.9-1.0 if train has active branding obligation, lower if not
    """
    branding = trainset.get("branding", {})
    if isinstance(branding, dict):
        advertiser = branding.get("current_advertiser")
        priority = branding.get("priority", "LOW")
        
        # If train is wrapped (has active advertiser)
        if advertiser and advertiser != "None" and advertiser != "":
            if priority == "HIGH":
                return 1.0  # Maximum obligation
            elif priority == "MEDIUM":
                return 0.8  # Medium obligation
            elif priority == "LOW":
                return 0.6  # Low obligation
            else:
                return 0.7  # Default for wrapped train
    
    # No active branding obligation
    return 0.1


def _calculate_defect_impact_score(trainset: Dict[str, Any]) -> float:
    """TIER 2: Calculate Defect Impact score (negative impact for minor defects)
    
    Returns negative score if train has minor defects (non-critical job cards)
    Higher absolute value = more defects = more negative impact
    """
    job_cards = trainset.get("job_cards", {})
    open_cards = job_cards.get("open_cards", 0)
    critical_cards = job_cards.get("critical_cards", 0)
    minor_cards = max(0, open_cards - critical_cards)
    
    if minor_cards == 0:
        return 1.0  # No minor defects - perfect score
    elif minor_cards <= 2:
        return 0.7  # Few minor defects - slight penalty
    elif minor_cards <= 5:
        return 0.4  # Moderate minor defects - moderate penalty
    else:
        return 0.1  # Many minor defects - high penalty


def _calculate_mileage_optimization_score(trainset: Dict[str, Any]) -> float:
    """TIER 3: Calculate Mileage Optimization score (helps balance fleet mileage)
    
    Higher score if train is below fleet average mileage (needs mileage to catch up)
    This helps equalize fleet usage over time
    """
    # Get 30-day mileage if available, otherwise use current mileage ratio
    km_30d = trainset.get("km_30d")
    current_mileage = trainset.get("current_mileage", 0.0)
    max_mileage = trainset.get("max_mileage_before_maintenance", 50000.0)
    
    if km_30d is not None:
        # Normalize 30-day mileage (assume 5000km is typical for 30 days)
        km_30d_norm = min(1.0, km_30d / 5000.0) if km_30d > 0 else 0.5
        # Lower 30-day mileage = higher score (needs mileage to balance fleet)
        return 1.0 - km_30d_norm * 0.8  # Range from 1.0 (no mileage) to 0.2 (high mileage)
    else:
        # Fallback to current mileage ratio
        if max_mileage == 0:
            return 0.5
        
        usage_ratio = current_mileage / max_mileage
        # Lower usage ratio = higher score (needs mileage to catch up)
        if usage_ratio < 0.3:
            return 1.0  # Very low mileage - high priority for balancing
        elif usage_ratio < 0.5:
            return 0.8  # Low mileage - good for balancing
        elif usage_ratio < 0.7:
            return 0.5  # Moderate mileage - neutral
        else:
            return 0.3  # High mileage - low priority


def top_reasons_and_risks(trainset: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate top reasons and risks based on Tiered Constraint Hierarchy"""
    reasons: List[str] = []
    risks: List[str] = []

    # TIER 1: Safety Compliance reasons
    safety_score = _calculate_safety_compliance_score(trainset)
    if safety_score == 1.0:
        reasons.append("Safety compliance verified - all certificates valid")
        
        # Check specific certificate statuses
        fitness_certs = trainset.get("fitness_certificates", {})
        if all(c.get("status") == "VALID" for c in fitness_certs.values()):
            reasons.append("All department certificates valid and current")
        
        # Check critical job cards
        job_cards = trainset.get("job_cards", {})
        if job_cards.get("critical_cards", 0) == 0:
            reasons.append("No critical job cards pending")
    else:
        # Safety compliance failed - add to risks
        fitness_certs = trainset.get("fitness_certificates", {})
        for dept, cert in fitness_certs.items():
            if cert.get("status") == "EXPIRED":
                risks.append(f"{dept} certificate expired - safety critical failure")
            elif cert.get("status") == "EXPIRING_SOON":
                risks.append(f"{dept} certificate expiring soon")
        
        if trainset.get("job_cards", {}).get("critical_cards", 0) > 0:
            risks.append("Open critical job-card - safety compliance failure")

    # TIER 2: Branding Obligation reasons
    branding_score = _calculate_branding_obligation_score(trainset)
    branding = trainset.get("branding", {})
    if isinstance(branding, dict):
        advertiser = branding.get("current_advertiser")
        priority = branding.get("priority", "LOW")
        if advertiser and advertiser not in (None, "None", ""):
            if priority == "HIGH":
                reasons.append("Mandatory branding commitment - high priority ad-wrap")
            elif priority == "MEDIUM":
                reasons.append("Branding obligation - medium priority wrapped train")
            elif priority == "LOW":
                reasons.append("Branding commitment - low priority wrapped train")
    
    # TIER 2: Defect Impact reasons
    defect_score = _calculate_defect_impact_score(trainset)
    job_cards = trainset.get("job_cards", {})
    minor_cards = max(0, job_cards.get("open_cards", 0) - job_cards.get("critical_cards", 0))
    if minor_cards == 0:
        reasons.append("No minor defects - optimal condition for service")
    elif minor_cards > 5:
        risks.append(f"{minor_cards} minor defects - prefer depot maintenance if possible")

    # TIER 3: Mileage Optimization reasons
    mileage_score = _calculate_mileage_optimization_score(trainset)
    if mileage_score > 0.8:
        reasons.append("Low mileage - good for fleet balancing")
    elif mileage_score > 0.6:
        reasons.append("Below-average mileage - helps equalize fleet usage")
    elif mileage_score < 0.4:
        risks.append("High mileage - lower priority for service")

    return {
        "top_reasons": reasons[:3],
        "top_risks": risks[:3],
    }


def generate_maintenance_reasons(trainset: Dict[str, Any]) -> Dict[str, List[str]]:
    """Generate maintenance-specific reasons explaining why a train needs maintenance"""
    reasons: List[str] = []
    risks: List[str] = []
    
    # Check critical job cards
    job_cards = trainset.get("job_cards", {})
    if not isinstance(job_cards, dict):
        job_cards = {}
    
    critical_cards = job_cards.get("critical_cards", 0)
    try:
        critical_cards = int(critical_cards) if critical_cards is not None else 0
    except (TypeError, ValueError):
        critical_cards = 0
    
    if critical_cards > 0:
        if critical_cards == 1:
            reasons.append(f"Critical job card open - requires immediate maintenance")
        else:
            reasons.append(f"{critical_cards} critical job cards open - requires immediate maintenance")
    
    # Check mileage
    try:
        current_mileage = float(trainset.get("current_mileage", 0.0))
    except (TypeError, ValueError):
        current_mileage = 0.0
    
    try:
        max_mileage = float(trainset.get("max_mileage_before_maintenance", float('inf')))
    except (TypeError, ValueError):
        max_mileage = float('inf')
    
    if max_mileage > 0 and current_mileage >= max_mileage * 0.95:
        mileage_percent = (current_mileage / max_mileage) * 100
        if mileage_percent >= 100:
            reasons.append(f"Mileage limit exceeded ({current_mileage:.0f} km) - maintenance required")
        else:
            reasons.append(f"Approaching mileage limit ({mileage_percent:.0f}% of {max_mileage:.0f} km) - maintenance due")
    
    # Check expired certificates
    fitness_certs = trainset.get("fitness_certificates", {})
    if isinstance(fitness_certs, dict):
        expired_certs = []
        for dept, cert in fitness_certs.items():
            if isinstance(cert, dict) and cert.get("status") == "EXPIRED":
                expired_certs.append(dept)
        if expired_certs:
            if len(expired_certs) == 1:
                reasons.append(f"{expired_certs[0]} certificate expired - maintenance required")
            else:
                reasons.append(f"Multiple certificates expired ({', '.join(expired_certs)}) - maintenance required")
    
    # Check sensor health if available
    sensor_health = trainset.get("sensor_health_score")
    if sensor_health is not None:
        try:
            sensor_health = float(sensor_health)
            if sensor_health < 0.7:
                risks.append(f"Low sensor health ({sensor_health*100:.0f}%) - may require maintenance")
        except (TypeError, ValueError):
            pass
    
    # Check ML predicted failure risk
    predicted_risk = trainset.get("predicted_failure_risk")
    if predicted_risk is not None:
        try:
            predicted_risk = float(predicted_risk)
            if predicted_risk > 0.7:
                risks.append(f"High failure risk predicted ({predicted_risk*100:.0f}%) - maintenance recommended")
        except (TypeError, ValueError):
            pass
    
    # If no specific reasons found, add generic maintenance reason
    if not reasons:
        reasons.append("Maintenance required based on system analysis")
    
    return {
        "top_reasons": reasons[:3],
        "top_risks": risks[:2],
    }


def generate_shap_values(trainset: Dict[str, Any], ml_features: List[str] = None) -> List[ShapFeature]:
    """Generate SHAP values for top features based on Tiered Constraint Hierarchy
    
    New tiered metrics:
    - Tier 1: Safety Compliance
    - Tier 2: Branding Obligation, Defect Impact
    - Tier 3: Mileage Optimization
    """
    shap_features = []
    
    # TIER 1: Safety Compliance (always shown if relevant)
    safety_score = _calculate_safety_compliance_score(trainset)
    if safety_score < 1.0:
        shap_features.append(ShapFeature(
            name="Safety Compliance",
            value=safety_score,
            impact="negative"  # Failed safety check
        ))
    else:
        shap_features.append(ShapFeature(
            name="Safety Compliance",
            value=1.0,
            impact="positive"  # Passed all safety checks
        ))
    
    # TIER 2: Branding Obligation
    branding_score = _calculate_branding_obligation_score(trainset)
    branding = trainset.get("branding", {})
    if isinstance(branding, dict) and branding.get("current_advertiser") not in (None, "None", ""):
        impact = "positive" if branding_score >= 0.6 else "neutral"
        shap_features.append(ShapFeature(
            name="Branding Obligation",
            value=branding_score,
            impact=impact
        ))
    
    # TIER 2: Defect Impact (negative impact)
    defect_score = _calculate_defect_impact_score(trainset)
    job_cards = trainset.get("job_cards", {})
    minor_cards = max(0, job_cards.get("open_cards", 0) - job_cards.get("critical_cards", 0))
    if minor_cards > 0:
        # Show as negative impact (lower defect_score = more defects)
        impact_value = 1.0 - defect_score  # Invert so higher = more defects
        shap_features.append(ShapFeature(
            name="Defect Impact",
            value=impact_value,
            impact="negative"
        ))
    
    # TIER 3: Mileage Optimization
    mileage_score = _calculate_mileage_optimization_score(trainset)
    # Get 30-day mileage for context
    km_30d = trainset.get("km_30d")
    if km_30d is not None:
        km_30d_norm = min(1.0, km_30d / 5000.0) if km_30d > 0 else 0.5
        impact = "positive" if mileage_score > 0.7 else "neutral" if mileage_score > 0.4 else "negative"
        shap_features.append(ShapFeature(
            name="Mileage Optimization",
            value=mileage_score,
            impact=impact
        ))
    else:
        # Use current mileage ratio
        current_mileage = trainset.get("current_mileage", 0.0)
        max_mileage = trainset.get("max_mileage_before_maintenance", 50000.0)
        if max_mileage > 0:
            usage_ratio = current_mileage / max_mileage
            impact = "positive" if mileage_score > 0.7 else "neutral" if mileage_score > 0.4 else "negative"
            shap_features.append(ShapFeature(
                name="Mileage Optimization",
                value=mileage_score,
                impact=impact
            ))
    
    # Sort by importance (safety first, then by value impact)
    # Safety compliance always first if failed, then by tier priority
    def sort_key(f: ShapFeature) -> tuple:
        tier_priority = {
            "Safety Compliance": 0,
            "Branding Obligation": 1,
            "Defect Impact": 2,
            "Mileage Optimization": 3
        }.get(f.name, 99)
        # Negative values (bad) should come first for visibility
        return (tier_priority, -abs(f.value) if f.impact == "negative" else abs(f.value))
    
    shap_features.sort(key=sort_key)
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


from app.config import settings


# Global rate limiter state
_gemini_lock = asyncio.Lock()
_last_request_time = 0.0
MIN_REQUEST_INTERVAL = 12.0  # Increased to 12s to be safe for Free Tier (approx 5 requests/min)

async def _generate_gemini_explanation(trainset: Dict[str, Any], decision: str, generated_reasons: Dict[str, List[str]]) -> Dict[str, Any]:
    """Call Gemini API to generate human-readable explanation using available context"""
    # SKIP AI for Standby trains to save quota
    if decision == "STANDBY":
        logger.info(f"Skipping Gemini for STANDBY train {trainset.get('trainset_id')} to conserve quota.")
        return {
            "summary": "Trainset is in good health but not selected for immediate service due to lower optimization score or sufficient fleet capacity.",
            "bullets": generated_reasons.get('top_reasons', [])
        }

    api_key = settings.gemini_api_key
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment. Skipping AI explanation.")
        return {}
    
    # logger.info(f"Gemini API Key found: {api_key[:4]}...{api_key[-4:]}") # Reduce log noise
    
    global _last_request_time
    
    async with _gemini_lock:
        # Enforce minimum interval
        now = time.time()
        time_since_last = now - _last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            sleep_time = MIN_REQUEST_INTERVAL - time_since_last
            logger.info(f"Rate limiting: Waiting {sleep_time:.2f}s before next request...")
            await asyncio.sleep(sleep_time)
        
        try:
            # Prepare context for LLM
            # Simplify trainset data to avoid huge payload
            context_data = {
                "id": trainset.get("trainset_id"),
                "status": trainset.get("status"),
                "decision": decision,
                "mileage": trainset.get("current_mileage"),
                "max_mileage": trainset.get("max_mileage_before_maintenance"),
                "certificates": trainset.get("fitness_certificates"),
                "job_cards": trainset.get("job_cards"),
                "branding": trainset.get("branding"),
                "generated_reasons": generated_reasons
            }
            
            prompt = f"""
            You are an expert metro train fleet manager acting as a decision support system. 
            Analyze the following trainset and the system-generated decision to provide a comprehensive, detailed, and professional justification.

            Context:
            - Decision: {decision}
               * INDUCT: Selected for passenger service. High health, low risk.
               * STANDBY: Reserve fleet. Good health but lower priority.
               * MAINTENANCE: Needs repair/inspection. Risk or hard failure.
            
            - Trainset Data (JSON): 
            {json.dumps(context_data, default=str)}
            
            System-Identified Factors:
            - Primary Reasons: {generated_reasons.get('top_reasons')}
            - Potential Risks: {generated_reasons.get('top_risks')}
            
            Task:
            Generate a detailed explanation that will be displayed to fleet operators.
            1. Synthesize the system factors into a coherent narrative.
            2. Highlight specific metrics (mileage, certificate type, job card count) to justify the decision.
            3. If there are risks, explain their operational impact.
            4. Be precise and professional.
            
            Return **STRICT JSON** format:
            {{
                "summary": "A detailed 2-3 sentence narrative explaining exactly why this specific train was assigned this status, referencing key data points.",
                "bullets": [
                    "Detailed point about strongest positive factor (e.g., 'Low mileage of 12,500km makes it ideal for heavy service usage')",
                    "Detailed point about operational status or branding (e.g., 'High priority ad-wrap requires immediate induction')",
                    "Detailed point about health/risks (e.g., 'Zero critical job cards and 98% sensor health score')"
                ]
            }}
            """
            
            async with httpx.AsyncClient() as client:
                logger.info(f"Sending request to Gemini API for trainset {trainset.get('trainset_id')}...")
                
                # Robust retry logic
                max_retries = 1 # 3 - Reduced to fail fast for better UX
                for attempt in range(max_retries):
                    response = await client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
                        json={"contents": [{"parts": [{"text": prompt}]}]},
                        timeout=10.0 # Increased timeout for safety
                    )
                    
                    if response.status_code == 429:
                        # Fail fast strategy: Cap waiting to 2 seconds max
                        retry_after = 2.0 
                        
                        # Don't sleep if this was the last attempt
                        if attempt < max_retries - 1:
                            logger.warning(f"Gemini API rate limited (429). Retrying in {retry_after}s...")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            logger.warning(f"Gemini API rate limited (429). Max retries reached.")
                    
                    break # Exit loop if not 429 or max retries reached
                
                # Update timestamp after request completes (whether success or fail)
                _last_request_time = time.time()
                
                logger.info(f"Gemini API Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    # logger.debug(f"Gemini Raw Response: {json.dumps(result)[:200]}...") 
                    parts = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])
                    if not parts:
                        logger.error(f"Gemini response has no parts: {result}")
                        return {}
                        
                    text_content = parts[0].get('text', '')
                    
                    # Extract JSON from markdown code block if present
                    if "```json" in text_content:
                        text_content = text_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in text_content:
                        text_content = text_content.split("```")[1].split("```")[0].strip()
                    
                    parsed_json = json.loads(text_content)
                    logger.info("Successfully parsed Gemini JSON explanation.")
                    return parsed_json
                else:
                    logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    
                    # Generate elaborated rule-based fallback
                    reasons = generated_reasons.get('top_reasons', [])
                    risks = generated_reasons.get('top_risks', [])
                    
                    # Construct a data-driven narrative
                    narrative = []
                    
                    # 1. Decision Context
                    narrative.append(f"This trainset has been categorized as **{decision}**.")
                    
                    # 2. Operational Health Details
                    job_cards = trainset.get("job_cards", {})
                    critical = job_cards.get("critical_cards", 0)
                    open_cards = job_cards.get("open_cards", 0)
                    
                    if critical > 0:
                        narrative.append(f"Crucially, there are {critical} critical job cards that require immediate attention, necessitating a maintenance stop.")
                    elif open_cards > 0:
                        narrative.append(f"While there are {open_cards} minor open job cards, no critical faults were detected, allowing for operational deployment.")
                    else:
                        narrative.append("The unit reports zero open job cards with all systems functioning optimally.")

                    # 3. Usage & Mileage Context
                    mileage = trainset.get("current_mileage", 0)
                    max_mileage = trainset.get("max_mileage_before_maintenance", 50000)
                    if mileage > 0:
                        usage_percent = (mileage / max_mileage) * 100
                        narrative.append(f"Current operational mileage is {mileage:,.0f} km ({usage_percent:.1f}% of utilization limit).")
                    
                    # 4. Branding/Commercial Context
                    branding = trainset.get("branding", {})
                    advertiser = branding.get("current_advertiser")
                    if advertiser and advertiser != "None":
                         priority = branding.get("priority", "Standard")
                         narrative.append(f"Commercial priority is high due to an active '{advertiser}' ad-wrap ({priority} priority), supporting the induction decision.")

                    # 5. Conclusion from Rules
                    if reasons:
                        narrative.append(f"Key automated selection factors: {'; '.join(reasons)}.")
                    
                    fallback_summary = " ".join(narrative)
                        
                    # Return partial explanation if AI fails
                    return {
                        "summary": fallback_summary,
                        "bullets": reasons
                    }
                    
        except Exception as e:
            logger.error(f"Failed to generate Gemini explanation: {e}")
            # Ensure we return a structuredfallback even on exception
            reasons_text = "; ".join(generated_reasons.get('top_reasons', []))
            return {
                 "summary": f"Automated analysis unavailable. System assigned {decision} based on: {reasons_text}.",
                 "bullets": generated_reasons.get('top_reasons', [])
            }



async def generate_comprehensive_explanation(trainset: Dict[str, Any], decision: str) -> Dict[str, Any]:
    """Generate comprehensive explanation for assignment decision"""
    
    # Calculate composite score
    score = calculate_composite_score(trainset, decision)
    
    # Get top reasons and risks
    reasons_risks = top_reasons_and_risks(trainset)
    
    # Detect rule violations
    violations = detect_rule_violations(trainset, decision)
    
    # Generate SHAP values
    shap_values = generate_shap_values(trainset)
    
    # Try to enhance with Gemini
    ai_explanation = await _generate_gemini_explanation(trainset, decision, reasons_risks)
    
    final_reasons = reasons_risks["top_reasons"]
    # If Gemini provided bullets, merge them or use them?
    # User wants detailed explanations. Let's use Gemini bullets as top reasons if available,
    # but ensure we don't lose critical rule-based info if Gemini hallucinates.
    # Actually, the prompt gives Gemini the rule-based reasons. It should incorporate them.
    if ai_explanation.get("bullets"):
        final_reasons = ai_explanation["bullets"]
        
    return {
        "score": score,
        "top_reasons": final_reasons,
        "top_risks": reasons_risks["top_risks"],
        "violations": violations,
        "shap_values": [feature.dict() for feature in shap_values],
        "summary": ai_explanation.get("summary") # Return summary separately if needed, or put in reasons[0]
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


