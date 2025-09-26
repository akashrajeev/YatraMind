from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TrainsetStatus(str, Enum):
    ACTIVE = "ACTIVE"
    STANDBY = "STANDBY"
    MAINTENANCE = "MAINTENANCE"


class JobCards(BaseModel):
    open_cards: int = 0
    critical_cards: int = 0


class Trainset(BaseModel):
    trainset_id: str
    status: TrainsetStatus = TrainsetStatus.STANDBY
    current_location: Dict[str, Any] = Field(default_factory=lambda: {"depot": "KALAMASSERY", "bay": ""})
    current_mileage: float = 0.0
    max_mileage_before_maintenance: float = 50000.0
    fitness_certificates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    job_cards: JobCards = Field(default_factory=JobCards)
    branding_priority: int = 0
    sensor_health_score: float = 1.0


class TrainsetUpdate(BaseModel):
    updates: Dict[str, Any] = Field(default_factory=dict)


class OptimizationRequest(BaseModel):
    target_date: datetime = Field(default_factory=datetime.utcnow)
    required_service_hours: int = 14
    override_constraints: Optional[Dict[str, Any]] = None


class ShapFeature(BaseModel):
    name: str
    value: float
    impact: str  # "positive" | "negative" | "neutral"


class InductionDecision(BaseModel):
    trainset_id: str
    decision: str  # INDUCT | STANDBY | MAINTENANCE
    confidence_score: float = 0.8
    reasons: List[str] = Field(default_factory=list)
    # Enhanced explainability fields
    score: float = Field(default=0.0, description="Composite score for this assignment")
    top_reasons: List[str] = Field(default_factory=list, description="Top 3 contributing positive reasons")
    top_risks: List[str] = Field(default_factory=list, description="Top 3 negative reasons")
    violations: List[str] = Field(default_factory=list, description="List of rule violations if assignment chosen despite violation")
    shap_values: List[ShapFeature] = Field(default_factory=list, description="Top 5 features and their impact if ML used")
