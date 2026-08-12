"""Canonical trainset domain models.

These models are deliberately independent of FastAPI and persistence. They
provide a stable vocabulary for the optimization refactor while legacy API
schemas remain unchanged during migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TrainsetStatus(str, Enum):
    ACTIVE = "ACTIVE"
    STANDBY = "STANDBY"
    MAINTENANCE = "MAINTENANCE"


@dataclass(frozen=True)
class FitnessCertificate:
    status: str
    expires_at: str | None = None


@dataclass(frozen=True)
class JobCardSummary:
    open_cards: int = 0
    critical_cards: int = 0


@dataclass(frozen=True)
class BrandingObligation:
    advertiser: str | None = None
    priority: str = "LOW"


@dataclass(frozen=True)
class CleaningRequirement:
    required: bool = False
    available: bool = False
    due_date: str | None = None


@dataclass(frozen=True)
class Trainset:
    trainset_id: str
    status: TrainsetStatus = TrainsetStatus.STANDBY
    current_mileage: float = 0.0
    max_mileage_before_maintenance: float | None = None
    fitness_certificates: Mapping[str, FitnessCertificate] = field(default_factory=dict)
    job_cards: JobCardSummary = field(default_factory=JobCardSummary)
    branding: BrandingObligation = field(default_factory=BrandingObligation)
    cleaning: CleaningRequirement = field(default_factory=CleaningRequirement)

    @classmethod
    def from_legacy_dict(cls, raw: Mapping[str, Any]) -> "Trainset":
        """Build a domain trainset from the current Mongo/API-shaped dict."""
        certs: dict[str, FitnessCertificate] = {}
        raw_certs = raw.get("fitness_certificates") or {}
        if isinstance(raw_certs, Mapping):
            for name, value in raw_certs.items():
                if isinstance(value, Mapping):
                    certs[str(name)] = FitnessCertificate(
                        status=str(value.get("status", "")),
                        expires_at=value.get("expires_at"),
                    )

        raw_cards = raw.get("job_cards") or {}
        if not isinstance(raw_cards, Mapping):
            raw_cards = {}

        raw_branding = raw.get("branding") or {}
        if not isinstance(raw_branding, Mapping):
            raw_branding = {}

        return cls(
            trainset_id=str(raw.get("trainset_id", "UNKNOWN")),
            status=TrainsetStatus(str(raw.get("status", TrainsetStatus.STANDBY.value)).upper()),
            current_mileage=float(raw.get("current_mileage", 0.0) or 0.0),
            max_mileage_before_maintenance=(
                float(raw["max_mileage_before_maintenance"])
                if raw.get("max_mileage_before_maintenance") not in (None, "", 0)
                else None
            ),
            fitness_certificates=certs,
            job_cards=JobCardSummary(
                open_cards=int(float(str(raw_cards.get("open_cards", 0)).strip() or 0)),
                critical_cards=int(float(str(raw_cards.get("critical_cards", 0)).strip() or 0)),
            ),
            branding=BrandingObligation(
                advertiser=raw_branding.get("current_advertiser"),
                priority=str(raw_branding.get("priority", "LOW")).upper(),
            ),
            cleaning=CleaningRequirement(
                required=bool(raw.get("requires_cleaning", False)),
                available=bool(raw.get("has_cleaning_slot", False)),
                due_date=raw.get("cleaning_due_date"),
            ),
        )
