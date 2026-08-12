"""Structured safety constraints for the optimization domain.

The legacy optimizer currently mixes validation, scoring and human-readable
explanations. This module provides a deterministic, structured safety seam for
migration. It does not alter the legacy pipeline yet.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.domain.optimization.types import ConstraintCode, ConstraintViolation, Severity

REQUIRED_FITNESS_CERTIFICATES = ("rolling_stock", "signalling", "telecom")


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def validate_trainset_safety(trainset: Dict[str, Any]) -> List[ConstraintViolation]:
    """Return structured blocking/non-blocking safety violations for a trainset."""
    trainset_id = str(trainset.get("trainset_id", "UNKNOWN"))
    violations: List[ConstraintViolation] = []

    certificates = trainset.get("fitness_certificates")
    if not isinstance(certificates, dict) or not certificates:
        violations.append(
            ConstraintViolation(
                ConstraintCode.MISSING_FITNESS_CERTIFICATES,
                Severity.CRITICAL,
                f"{trainset_id}: fitness certificates are missing or empty",
                "fitness_certificates",
            )
        )
        # Missing certificates already block induction; continue collecting
        # other diagnostics where the input shape allows it.
    elif isinstance(certificates, dict):
        for required in REQUIRED_FITNESS_CERTIFICATES:
            if required not in certificates:
                violations.append(
                    ConstraintViolation(
                        ConstraintCode.MISSING_REQUIRED_CERTIFICATE,
                        Severity.CRITICAL,
                        f"{trainset_id}: required certificate '{required}' is missing",
                        f"fitness_certificates.{required}",
                    )
                )
        for cert_name, cert_data in certificates.items():
            if isinstance(cert_data, dict) and str(cert_data.get("status", "")).upper() == "EXPIRED":
                violations.append(
                    ConstraintViolation(
                        ConstraintCode.EXPIRED_FITNESS_CERTIFICATE,
                        Severity.CRITICAL,
                        f"{trainset_id}: certificate '{cert_name}' is expired",
                        f"fitness_certificates.{cert_name}.status",
                    )
                )

    job_cards = trainset.get("job_cards")
    if isinstance(job_cards, dict) and _as_int(job_cards.get("critical_cards")) > 0:
        violations.append(
            ConstraintViolation(
                ConstraintCode.CRITICAL_JOB_CARD,
                Severity.CRITICAL,
                f"{trainset_id}: critical job cards are open",
                "job_cards.critical_cards",
            )
        )

    current_mileage = trainset.get("current_mileage", 0.0)
    max_mileage = trainset.get("max_mileage_before_maintenance")
    if max_mileage not in (None, "", 0):
        try:
            if float(current_mileage) >= float(max_mileage):
                violations.append(
                    ConstraintViolation(
                        ConstraintCode.MILEAGE_LIMIT_EXCEEDED,
                        Severity.CRITICAL,
                        f"{trainset_id}: mileage limit has been reached",
                        "current_mileage",
                    )
                )
        except (TypeError, ValueError):
            pass

    if str(trainset.get("status", "")).upper() == "MAINTENANCE":
        violations.append(
            ConstraintViolation(
                ConstraintCode.TRAINSET_IN_MAINTENANCE,
                Severity.CRITICAL,
                f"{trainset_id}: trainset is currently in maintenance",
                "status",
            )
        )

    if bool(trainset.get("requires_cleaning")) and not bool(trainset.get("has_cleaning_slot")):
        violations.append(
            ConstraintViolation(
                ConstraintCode.CLEANING_SLOT_UNAVAILABLE,
                Severity.CRITICAL,
                f"{trainset_id}: required cleaning slot is unavailable",
                "has_cleaning_slot",
            )
        )

    return violations
