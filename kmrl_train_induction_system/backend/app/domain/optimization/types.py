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
    MISSING_FITNESS_CERTIFICATES = "MISSING_FITNESS_CERTIFICATES"
    MISSING_REQUIRED_CERTIFICATE = "MISSING_REQUIRED_CERTIFICATE"
    EXPIRED_FITNESS_CERTIFICATE = "EXPIRED_FITNESS_CERTIFICATE"
    CRITICAL_JOB_CARD = "CRITICAL_JOB_CARD"
    MILEAGE_LIMIT_EXCEEDED = "MILEAGE_LIMIT_EXCEEDED"
    TRAINSET_IN_MAINTENANCE = "TRAINSET_IN_MAINTENANCE"
    CLEANING_SLOT_UNAVAILABLE = "CLEANING_SLOT_UNAVAILABLE"


@dataclass(frozen=True)
class ConstraintViolation:
    code: ConstraintCode
    severity: Severity
    message: str
    field: Optional[str] = None

    @property
    def is_blocking(self) -> bool:
        return self.severity == Severity.CRITICAL
