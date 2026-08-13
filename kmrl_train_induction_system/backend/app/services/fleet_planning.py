from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import math
import logging

logger = logging.getLogger(__name__)

class ServiceBand(BaseModel):
    name: str
    start_time: str
    end_time: str
    headway_min: int

class LineParameters(BaseModel):
    line_runtime_min: int
    turn_back_min: int

class TimetableConfig(BaseModel):
    service_bands: List[ServiceBand]
    line_params: LineParameters
    reserve_ratio: float = 0.15

class FleetRequirementResult(BaseModel):
    required_service_trains: int
    standby_buffer: int
    total_required_trains: int
    calculation_method: str
    details: Dict[str, Any] = Field(default_factory=dict)

DEFAULT_TIMETABLE = TimetableConfig(
    service_bands=[
        ServiceBand(name="Morning Peak", start_time="08:00", end_time="11:00", headway_min=8),
        ServiceBand(name="Evening Peak", start_time="17:00", end_time="20:00", headway_min=8),
        ServiceBand(name="Off-Peak", start_time="06:00", end_time="22:00", headway_min=15),
    ],
    line_params=LineParameters(line_runtime_min=45, turn_back_min=5),
    reserve_ratio=0.15,
)


def _timetable_requirement(config: TimetableConfig, *, method: str) -> FleetRequirementResult:
    cycle_time = 2 * config.line_params.line_runtime_min + 2 * config.line_params.turn_back_min
    max_trains_needed = 0
    band_details = []
    for band in config.service_bands:
        needed = math.ceil(cycle_time / band.headway_min) if band.headway_min > 0 else 0
        band_details.append({"band": band.name, "headway": band.headway_min, "needed": needed})
        max_trains_needed = max(max_trains_needed, needed)

    standby_buffer = math.ceil(max_trains_needed * config.reserve_ratio)
    return FleetRequirementResult(
        required_service_trains=max_trains_needed,
        standby_buffer=standby_buffer,
        total_required_trains=max_trains_needed + standby_buffer,
        calculation_method=method,
        details={
            "cycle_time_min": cycle_time,
            "bands": band_details,
            "reserve_ratio": config.reserve_ratio,
        },
    )


def compute_required_trains(
    service_date: Optional[str] = None,
    timetable_config: Optional[TimetableConfig] = None,
    override_count: Optional[int] = None,
    legacy_hours: Optional[float] = None,
    avg_hours_per_train: Optional[float] = None,
) -> FleetRequirementResult:
    """Compute required trains with explicit precedence.

    Precedence is manual override > legacy-hours compatibility > explicit
    timetable > default timetable. The legacy path exists only to preserve
    older tests/callers while the timetable model becomes canonical.
    """
    if override_count is not None and override_count > 0:
        reserve_ratio = timetable_config.reserve_ratio if timetable_config else 0.15
        standby = math.ceil(override_count * reserve_ratio)
        return FleetRequirementResult(
            required_service_trains=override_count,
            standby_buffer=standby,
            total_required_trains=override_count + standby,
            calculation_method="override",
            details={"source": "required_service_count"},
        )

    if legacy_hours is not None:
        avg = avg_hours_per_train if avg_hours_per_train and avg_hours_per_train > 0 else 12.0
        required = max(1, math.ceil(legacy_hours / avg)) if legacy_hours > 0 else 1
        reserve_ratio = timetable_config.reserve_ratio if timetable_config else 0.15
        standby = math.ceil(required * reserve_ratio)
        return FleetRequirementResult(
            required_service_trains=required,
            standby_buffer=standby,
            total_required_trains=required + standby,
            calculation_method="legacy_hours",
            details={"hours": legacy_hours, "avg_per_train": avg},
        )

    if timetable_config is not None:
        return _timetable_requirement(timetable_config, method="timetable")

    return _timetable_requirement(DEFAULT_TIMETABLE, method="timetable_default")
