# backend/app/services/stabling_optimizer.py
import math
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class StablingGeometryOptimizer:
    """Optimize stabling geometry to minimize nightly shunting and morning turn-out time"""
    
    def __init__(self):
        # Depot layout with bay positions and characteristics
        self.depot_layouts = {
            "Aluva": {
                "total_bays": 8,
                "maintenance_bays": [1, 2, 3],  # Bays 1-3 for maintenance
                "cleaning_bays": [4, 5],       # Bays 4-5 for cleaning
                "service_bays": [6, 7, 8],     # Bays 6-8 for service trains
                "bay_positions": {
                    1: {"x": 0, "y": 0, "type": "maintenance", "turnout_time": 15},
                    2: {"x": 50, "y": 0, "type": "maintenance", "turnout_time": 12},
                    3: {"x": 100, "y": 0, "type": "maintenance", "turnout_time": 18},
                    4: {"x": 0, "y": 50, "type": "cleaning", "turnout_time": 8},
                    5: {"x": 50, "y": 50, "type": "cleaning", "turnout_time": 10},
                    6: {"x": 0, "y": 100, "type": "service", "turnout_time": 5},
                    7: {"x": 50, "y": 100, "type": "service", "turnout_time": 6},
                    8: {"x": 100, "y": 100, "type": "service", "turnout_time": 7}
                },
                "shunting_tracks": ["TRACK_A", "TRACK_B", "TRACK_C"],
                "turnout_points": ["POINT_1", "POINT_2", "POINT_3"]
            },
            "Petta": {
                "total_bays": 6,
                "maintenance_bays": [1, 2],
                "cleaning_bays": [3],
                "service_bays": [4, 5, 6],
                "bay_positions": {
                    1: {"x": 0, "y": 0, "type": "maintenance", "turnout_time": 20},
                    2: {"x": 50, "y": 0, "type": "maintenance", "turnout_time": 15},
                    3: {"x": 0, "y": 50, "type": "cleaning", "turnout_time": 12},
                    4: {"x": 50, "y": 50, "type": "service", "turnout_time": 8},
                    5: {"x": 0, "y": 100, "type": "service", "turnout_time": 6},
                    6: {"x": 50, "y": 100, "type": "service", "turnout_time": 5}
                },
                "shunting_tracks": ["TRACK_A", "TRACK_B"],
                "turnout_points": ["POINT_1", "POINT_2"]
            }
        }
    
    async def optimize_stabling_geometry(self, trainsets: List[Dict[str, Any]], 
                                       induction_decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize stabling geometry for minimal shunting and turn-out time"""
        try:
            logger.info("Starting stabling geometry optimization")
            
            # Group trainsets by depot
            depot_assignments = self._group_trainsets_by_depot(trainsets, induction_decisions)
            
            optimized_layout = {}
            total_shunting_time = 0
            total_turnout_time = 0
            
            for depot_name, depot_trainsets in depot_assignments.items():
                if depot_name not in self.depot_layouts:
                    continue
                
                # Optimize bay assignments for this depot
                depot_optimization = await self._optimize_depot_layout(
                    depot_name, depot_trainsets
                )
                
                optimized_layout[depot_name] = depot_optimization
                total_shunting_time += depot_optimization["total_shunting_time"]
                total_turnout_time += depot_optimization["total_turnout_time"]
            
            # Calculate overall efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(
                optimized_layout, total_shunting_time, total_turnout_time
            )
            
            return {
                "optimized_layout": optimized_layout,
                "efficiency_metrics": efficiency_metrics,
                "total_shunting_time": total_shunting_time,
                "total_turnout_time": total_turnout_time,
                "optimization_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Stabling geometry optimization failed: {e}")
            raise
    
    def _group_trainsets_by_depot(self, trainsets: List[Dict[str, Any]], 
                                decisions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group trainsets by depot based on current location and decisions"""
        depot_groups = {}
        
        for trainset in trainsets:
            depot = trainset.get("current_location", {}).get("depot", "Aluva")
            if depot not in depot_groups:
                depot_groups[depot] = []
            
            # Add decision information
            decision = next((d for d in decisions if d["trainset_id"] == trainset["trainset_id"]), None)
            trainset["induction_decision"] = decision
            depot_groups[depot].append(trainset)
        
        return depot_groups
    
    async def _optimize_depot_layout(self, depot_name: str, trainsets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize bay assignments for a specific depot"""
        depot_layout = self.depot_layouts[depot_name]
        bay_assignments = {}
        shunting_operations = []
        
        # Separate trainsets by decision type
        induct_trainsets = [t for t in trainsets if t.get("induction_decision", {}).get("decision") == "INDUCT"]
        maintenance_trainsets = [t for t in trainsets if t.get("induction_decision", {}).get("decision") == "MAINTENANCE"]
        standby_trainsets = [t for t in trainsets if t.get("induction_decision", {}).get("decision") == "STANDBY"]
        
        # Assign bays based on priority and optimization
        bay_assignments.update(self._assign_service_bays(induct_trainsets, depot_layout))
        bay_assignments.update(self._assign_maintenance_bays(maintenance_trainsets, depot_layout))
        bay_assignments.update(self._assign_standby_bays(standby_trainsets, depot_layout))
        
        # Calculate shunting operations needed
        shunting_operations = self._calculate_shunting_operations(trainsets, bay_assignments, depot_layout)
        
        # Calculate total times
        total_shunting_time = sum(op["estimated_time"] for op in shunting_operations)
        total_turnout_time = sum(
            depot_layout["bay_positions"][bay]["turnout_time"] 
            for bay in bay_assignments.values() 
            if bay in depot_layout["bay_positions"]
        )
        
        return {
            "depot": depot_name,
            "bay_assignments": bay_assignments,
            "shunting_operations": shunting_operations,
            "total_shunting_time": total_shunting_time,
            "total_turnout_time": total_turnout_time,
            "efficiency_score": self._calculate_depot_efficiency(bay_assignments, depot_layout)
        }
    
    def _assign_service_bays(self, induct_trainsets: List[Dict[str, Any]], depot_layout: Dict[str, Any]) -> Dict[str, int]:
        """Assign service bays to trainsets for induction (priority: closest to turnout)"""
        assignments = {}
        service_bays = depot_layout["service_bays"]
        
        # Sort service bays by turnout time (fastest first)
        service_bays_sorted = sorted(service_bays, key=lambda bay: depot_layout["bay_positions"][bay]["turnout_time"])
        
        for i, trainset in enumerate(induct_trainsets):
            if i < len(service_bays_sorted):
                assignments[trainset["trainset_id"]] = service_bays_sorted[i]
        
        return assignments
    
    def _assign_maintenance_bays(self, maintenance_trainsets: List[Dict[str, Any]], depot_layout: Dict[str, Any]) -> Dict[str, int]:
        """Assign maintenance bays to trainsets requiring maintenance"""
        assignments = {}
        maintenance_bays = depot_layout["maintenance_bays"]
        
        for i, trainset in enumerate(maintenance_trainsets):
            if i < len(maintenance_bays):
                assignments[trainset["trainset_id"]] = maintenance_bays[i]
        
        return assignments
    
    def _assign_standby_bays(self, standby_trainsets: List[Dict[str, Any]], depot_layout: Dict[str, Any]) -> Dict[str, int]:
        """Assign standby bays to trainsets (any available bay)"""
        assignments = {}
        all_bays = list(range(1, depot_layout["total_bays"] + 1))
        
        for i, trainset in enumerate(standby_trainsets):
            if i < len(all_bays):
                assignments[trainset["trainset_id"]] = all_bays[i]
        
        return assignments
    
    def _calculate_shunting_operations(self, trainsets: List[Dict[str, Any]], 
                                    bay_assignments: Dict[str, int], 
                                    depot_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate shunting operations needed to move trainsets to assigned bays"""
        operations = []
        
        for trainset in trainsets:
            current_bay = self._extract_bay_number(trainset.get("current_location", {}).get("bay", ""))
            assigned_bay = bay_assignments.get(trainset["trainset_id"])
            
            if current_bay and assigned_bay and current_bay != assigned_bay:
                # Calculate shunting distance and time
                current_pos = depot_layout["bay_positions"].get(current_bay, {"x": 0, "y": 0})
                assigned_pos = depot_layout["bay_positions"].get(assigned_bay, {"x": 0, "y": 0})
                
                distance = math.sqrt(
                    (assigned_pos["x"] - current_pos["x"])**2 + 
                    (assigned_pos["y"] - current_pos["y"])**2
                )
                
                # Estimate shunting time based on distance and complexity
                estimated_time = self._estimate_shunting_time(distance, current_bay, assigned_bay)
                
                operations.append({
                    "trainset_id": trainset["trainset_id"],
                    "from_bay": current_bay,
                    "to_bay": assigned_bay,
                    "distance": distance,
                    "estimated_time": estimated_time,
                    "complexity": "HIGH" if distance > 100 else "MEDIUM" if distance > 50 else "LOW"
                })
        
        return operations
    
    def _extract_bay_number(self, bay_string: str) -> int:
        """Extract bay number from bay string (e.g., 'Aluva_BAY_05' -> 5)"""
        try:
            if "_BAY_" in bay_string:
                return int(bay_string.split("_BAY_")[1])
            return 0
        except (ValueError, IndexError):
            return 0
    
    def _estimate_shunting_time(self, distance: float, from_bay: int, to_bay: int) -> int:
        """Estimate shunting time based on distance and bay complexity"""
        base_time = 5  # Base time in minutes
        distance_factor = distance * 0.1  # 0.1 minutes per unit distance
        complexity_factor = 2 if abs(to_bay - from_bay) > 3 else 1  # More complex if crossing many bays
        
        return int(base_time + distance_factor + complexity_factor)
    
    def _calculate_depot_efficiency(self, bay_assignments: Dict[str, int], depot_layout: Dict[str, Any]) -> float:
        """Calculate depot efficiency score (0-1, higher is better)"""
        if not bay_assignments:
            return 0.0
        
        total_turnout_time = sum(
            depot_layout["bay_positions"][bay]["turnout_time"] 
            for bay in bay_assignments.values() 
            if bay in depot_layout["bay_positions"]
        )
        
        # Normalize efficiency (lower turnout time = higher efficiency)
        max_possible_time = len(bay_assignments) * 20  # Assume max 20 minutes per bay
        efficiency = max(0, 1 - (total_turnout_time / max_possible_time))
        
        return round(efficiency, 3)
    
    def _calculate_efficiency_metrics(self, optimized_layout: Dict[str, Any], 
                                   total_shunting_time: int, total_turnout_time: int) -> Dict[str, Any]:
        """Calculate overall efficiency metrics"""
        total_trainsets = sum(len(depot["bay_assignments"]) for depot in optimized_layout.values())
        
        return {
            "overall_efficiency": round(1 - (total_turnout_time / (total_trainsets * 15)), 3),
            "shunting_efficiency": round(1 - (total_shunting_time / (total_trainsets * 10)), 3),
            "energy_savings": round(total_shunting_time * 0.5, 2),  # kWh saved
            "time_savings": round(total_turnout_time * 0.3, 2),  # Minutes saved
            "depot_scores": {
                depot: layout["efficiency_score"] 
                for depot, layout in optimized_layout.items()
            }
        }
    
    async def get_shunting_schedule(self, optimized_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate detailed shunting schedule for operations team"""
        schedule = []
        
        for depot_name, depot_layout in optimized_layout.items():
            for operation in depot_layout["shunting_operations"]:
                schedule.append({
                    "depot": depot_name,
                    "trainset_id": operation["trainset_id"],
                    "operation": f"Move from Bay {operation['from_bay']} to Bay {operation['to_bay']}",
                    "estimated_duration": f"{operation['estimated_time']} minutes",
                    "complexity": operation["complexity"],
                    "scheduled_time": "21:00-23:00",  # Nightly shunting window
                    "crew_required": "2 operators" if operation["complexity"] == "HIGH" else "1 operator"
                })
        
        return sorted(schedule, key=lambda x: x["estimated_duration"])
