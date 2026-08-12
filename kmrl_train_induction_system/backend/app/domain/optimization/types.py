"""Canonical domain types for optimization decisions.

This module is intentionally independent of the legacy optimizer so the
optimization pipeline can be migrated incrementally without changing API
contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DecisionStatus(str, Enum):
    INDUCT = "INDUCT"
    STANDBY = "STANDBY"
    MAINTENANCE = "MAINTENANCE"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class ConstraintCode(str, Enum):
    MISSING_FITNESS_CERTIFICATES = "missing_fitness_certificates"
    MISSING_REQUIRED_CERTIFICATE = "missing_required_certificate"
    EXPIRED_FITNESS_CERTIFICATE = "expired_fitness_certificate"
    CRITICAL_JOB_CARD = "critical_job_card"
    MILEAGE_LIMIT_EXCEEDED = "mileage_limit_exceeded"
    TRAINSET_IN_MAINTENANCE = "trainset_in_maintenance"
    CLEANING_SLOT_UNAVAILABLE = "cleaning_slot_unavailable"


@dataclass(frozen=True)
class ConstraintViolation:
    code: ConstraintCode
    severity: Severity
    message: str
    field: Optional[str] = None

    @property
    def is_blocking(self) -> bool:
        return self.severity == Severity.CRITICAL
