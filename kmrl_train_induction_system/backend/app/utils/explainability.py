# backend/app/utils/explainability.py
from __future__ import annotations

from typing import Dict, Any, List
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os


def _jinja_env() -> Environment:
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
    return Environment(
        loader=FileSystemLoader(base_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


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


def render_explanation_html(context: Dict[str, Any]) -> str:
    env = _jinja_env()
    tpl = env.get_template("explanation.j2")
    return tpl.render(**context)


