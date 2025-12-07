from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import math
import logging

logger = logging.getLogger(__name__)

# --- Configuration Models ---

class ServiceBand(BaseModel):
    name: str  # e.g., "Peak", "Off-Peak"
    start_time: str  # HH:MM
    end_time: str    # HH:MM
    headway_min: int # Minutes between trains

class LineParameters(BaseModel):
    line_runtime_min: int  # Single trip runtime
    turn_back_min: int     # Dwell/turnaround time at each terminal

class TimetableConfig(BaseModel):
    service_bands: List[ServiceBand]
    line_params: LineParameters
    reserve_ratio: float = 0.15  # 15% operational reserve

# --- Result Model ---

class FleetRequirementResult(BaseModel):
    required_service_trains: int
    standby_buffer: int
    total_required_trains: int
    calculation_method: str  # "timetable", "override", "legacy_hours", "default"
    details: Dict[str, Any] = Field(default_factory=dict)

# --- Default Configuration (can be moved to a config file/DB later) ---

DEFAULT_TIMETABLE = TimetableConfig(
    service_bands=[
        ServiceBand(name="Morning Peak", start_time="08:00", end_time="11:00", headway_min=8),
        ServiceBand(name="Evening Peak", start_time="17:00", end_time="20:00", headway_min=8),
        ServiceBand(name="Off-Peak", start_time="06:00", end_time="22:00", headway_min=15), # Broad coverage, peak overrides this if higher
    ],
    line_params=LineParameters(
        line_runtime_min=45, # Example: Aluva to Petta approx 45-50 mins
        turn_back_min=5      # 5 mins turnaround
    ),
    reserve_ratio=0.15
)

# --- Core Logic ---

def compute_required_trains(
    service_date: Optional[str] = None,
    timetable_config: Optional[TimetableConfig] = None,
    override_count: Optional[int] = None
) -> FleetRequirementResult:
    """
    Compute the number of trains required based on timetable or manual override.
    
    Priority:
    1. override_count (Manual override) - if provided, uses this directly
    2. timetable_config (Timetable-driven) - if provided, uses this
    3. Default timetable fallback - uses DEFAULT_TIMETABLE configuration
    """
    
    # 1. Manual Override
    if override_count is not None and override_count > 0:
        # Even with override, we should calculate standby buffer if possible, 
        # but usually override implies "total trains I want". 
        # However, the prompt says: "If required_service_count is given: treat that as required_service_trains."
        # And "but not the standby buffer logic". 
        # So we apply standby buffer on top of override? 
        # "If required_service_count is given: treat that as required_service_trains. ... standby_buffer = ceil(required_service_trains * reserve_ratio)"
        
        req_service = override_count
        reserve_ratio = timetable_config.reserve_ratio if timetable_config else 0.15
        standby = math.ceil(req_service * reserve_ratio)
        
        return FleetRequirementResult(
            required_service_trains=req_service,
            standby_buffer=standby,
            total_required_trains=req_service + standby,
            calculation_method="override",
            details={"source": "required_service_count"}
        )

    # 2. Timetable Driven (Explicit)
    if timetable_config:
        config = timetable_config
        # Calculate Cycle Time
        cycle_time = 2 * config.line_params.line_runtime_min + 2 * config.line_params.turn_back_min
        
        max_trains_needed = 0
        band_details = []
        
        for band in config.service_bands:
            if band.headway_min > 0:
                needed = math.ceil(cycle_time / band.headway_min)
            else:
                needed = 0
            
            band_details.append({
                "band": band.name,
                "headway": band.headway_min,
                "needed": needed
            })
            max_trains_needed = max(max_trains_needed, needed)
            
        required_service_trains = max_trains_needed
        standby_buffer = math.ceil(required_service_trains * config.reserve_ratio)
        
        return FleetRequirementResult(
            required_service_trains=required_service_trains,
            standby_buffer=standby_buffer,
            total_required_trains=required_service_trains + standby_buffer,
            calculation_method="timetable",
            details={
                "cycle_time_min": cycle_time,
                "bands": band_details,
                "reserve_ratio": config.reserve_ratio
            }
        )

    # 3. Default Fallback (Default Timetable)
    config = DEFAULT_TIMETABLE
    
    # Calculate Cycle Time
    cycle_time = 2 * config.line_params.line_runtime_min + 2 * config.line_params.turn_back_min
    
    max_trains_needed = 0
    band_details = []
    
    for band in config.service_bands:
        if band.headway_min > 0:
            needed = math.ceil(cycle_time / band.headway_min)
        else:
            needed = 0
        
        band_details.append({
            "band": band.name,
            "headway": band.headway_min,
            "needed": needed
        })
        max_trains_needed = max(max_trains_needed, needed)
        
    required_service_trains = max_trains_needed
    standby_buffer = math.ceil(required_service_trains * config.reserve_ratio)
    
    return FleetRequirementResult(
        required_service_trains=required_service_trains,
        standby_buffer=standby_buffer,
        total_required_trains=required_service_trains + standby_buffer,
            calculation_method="timetable",
        details={
            "cycle_time_min": cycle_time,
            "bands": band_details,
            "reserve_ratio": config.reserve_ratio
        }
    )
