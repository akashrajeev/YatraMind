# backend/app/ml/multi_depot/config.py
"""
Multi-Depot Configuration and Data Models
Exposes depot features, fleet features, and simulation run entities to AI
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, date
from enum import Enum
import numpy as np


class LocationType(str, Enum):
    FULL_DEPOT = "FULL_DEPOT"
    TERMINAL_YARD = "TERMINAL_YARD"
    MAINLINE_SIDING = "MAINLINE_SIDING"


class DepotConfig(BaseModel):
    """Depot configuration with features exposed to AI"""
    depot_id: str
    depot_name: str
    location_type: LocationType
    
    # Bay capacities
    service_bay_capacity: int
    maintenance_bay_capacity: int
    standby_bay_capacity: int
    total_bays: int
    
    # Capabilities
    supports_heavy_maintenance: bool = False
    supports_cleaning: bool = False
    can_start_service: bool = True
    
    # Shunting graph (adjacency matrix or edge list)
    shunting_graph: Dict[str, Any] = Field(default_factory=dict)
    
    # Turnout time map: bay_id -> minutes to exit
    turnout_map: Dict[int, float] = Field(default_factory=dict)
    
    # Maintenance capacity (trains per day)
    maintenance_capacity: int = 0
    
    # Geographic coordinates (for distance calculation)
    coordinates: Optional[tuple] = None  # (lat, lon)
    
    # Bay positions (for shunting distance)
    bay_positions: Dict[int, Dict[str, float]] = Field(default_factory=dict)
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert depot config to feature vector for ML"""
        return np.array([
            float(self.service_bay_capacity) / 20.0,  # Normalize
            float(self.maintenance_bay_capacity) / 10.0,
            float(self.standby_bay_capacity) / 10.0,
            float(self.total_bays) / 30.0,
            1.0 if self.supports_heavy_maintenance else 0.0,
            1.0 if self.supports_cleaning else 0.0,
            1.0 if self.can_start_service else 0.0,
            float(self.maintenance_capacity) / 10.0,
        ])


class FleetFeatures(BaseModel):
    """Fleet features exposed to AI models"""
    train_id: str
    mileage: float
    sensor_timeseries: List[Dict[str, Any]] = Field(default_factory=list)
    job_cards: Dict[str, int] = Field(default_factory=dict)
    branding_flag: bool = False
    branding_priority: float = 0.0
    historical_failures: List[Dict[str, Any]] = Field(default_factory=list)
    fitness_certificates: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    current_location: Dict[str, Any] = Field(default_factory=dict)
    sensor_health_score: float = 1.0
    current_mileage: float = 0.0
    max_mileage_before_maintenance: float = 50000.0
    
    def to_feature_vector(self, include_timeseries: bool = False) -> np.ndarray:
        """Convert fleet features to feature vector"""
        features = [
            self.mileage / 100000.0,  # Normalize
            float(self.job_cards.get("critical_cards", 0)) / 10.0,
            float(self.job_cards.get("open_cards", 0)) / 20.0,
            1.0 if self.branding_flag else 0.0,
            self.branding_priority,
            float(len(self.historical_failures)) / 10.0,
            self.sensor_health_score,
            self.current_mileage / 50000.0,
            self.current_mileage / self.max_mileage_before_maintenance if self.max_mileage_before_maintenance > 0 else 0.0,
        ]
        
        # Add fitness certificate features
        fc_valid = sum(1 for fc in self.fitness_certificates.values() 
                      if isinstance(fc, dict) and fc.get("status") == "VALID")
        features.append(fc_valid / 3.0)
        
        if include_timeseries and self.sensor_timeseries:
            # Aggregate timeseries features (last N days)
            recent = self.sensor_timeseries[-7:] if len(self.sensor_timeseries) > 7 else self.sensor_timeseries
            if recent:
                avg_temp = np.mean([s.get("temperature", 25.0) for s in recent if isinstance(s, dict)])
                avg_vibration = np.mean([s.get("vibration", 0.0) for s in recent if isinstance(s, dict)])
                features.extend([avg_temp / 50.0, avg_vibration / 10.0])
            else:
                features.extend([0.5, 0.0])
        else:
            features.extend([0.5, 0.0])
        
        return np.array(features, dtype=np.float32)


class SimulationRun(BaseModel):
    """Simulation run entity"""
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }
    )
    
    run_id: str
    seed: int
    config: Dict[str, Any]
    date_range: tuple  # (start_date, end_date)
    random_seed: Optional[int] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Results
    results: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class MultiDepotState(BaseModel):
    """State representation for multi-depot RL agent"""
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={
            np.ndarray: lambda v: v.tolist(),
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
        }
    )
    
    # Depot states
    depot_bay_occupancy: Dict[str, Dict[int, Optional[str]]]  # depot_id -> bay_id -> train_id
    depot_train_health_scores: Dict[str, Dict[str, float]]  # depot_id -> train_id -> health_score
    depot_turnout_times: Dict[str, Dict[int, float]]  # depot_id -> bay_id -> minutes
    
    # Network-level
    distance_matrix: np.ndarray  # depot-to-depot distances
    predicted_demand: Dict[str, int]  # depot_id -> required_service_trains
    
    # Fleet state
    train_locations: Dict[str, str]  # train_id -> depot_id
    train_decisions: Dict[str, str]  # train_id -> decision (SERVICE/STANDBY/MAINTENANCE)
    
    def to_vector(self, depot_configs: List[DepotConfig], fleet_features: List[FleetFeatures]) -> np.ndarray:
        """Convert state to feature vector for RL agent"""
        # Flatten depot occupancy
        occupancy_vec = []
        for depot_config in depot_configs:
            occupancy = self.depot_bay_occupancy.get(depot_config.depot_id, {})
            for bay_id in range(1, depot_config.total_bays + 1):
                occupancy_vec.append(1.0 if occupancy.get(bay_id) else 0.0)
        
        # Flatten health scores
        health_vec = []
        for depot_config in depot_configs:
            health_scores = self.depot_train_health_scores.get(depot_config.depot_id, {})
            # Average health per depot
            if health_scores:
                health_vec.append(np.mean(list(health_scores.values())))
            else:
                health_vec.append(0.85)  # Default
        
        # Flatten turnout times
        turnout_vec = []
        for depot_config in depot_configs:
            turnout_times = self.depot_turnout_times.get(depot_config.depot_id, {})
            # Average turnout time per depot
            if turnout_times:
                turnout_vec.append(np.mean(list(turnout_times.values())) / 30.0)  # Normalize
            else:
                turnout_vec.append(0.33)  # Default 10 min / 30
        
        # Distance matrix (flattened)
        distance_vec = self.distance_matrix.flatten() / 100.0  # Normalize to km
        
        # Demand vector
        demand_vec = []
        for depot_config in depot_configs:
            demand = self.predicted_demand.get(depot_config.depot_id, 0)
            demand_vec.append(demand / 20.0)  # Normalize
        
        # Combine all
        state_vec = np.concatenate([
            np.array(occupancy_vec),
            np.array(health_vec),
            np.array(turnout_vec),
            distance_vec,
            np.array(demand_vec),
        ])
        
        return state_vec.astype(np.float32)


