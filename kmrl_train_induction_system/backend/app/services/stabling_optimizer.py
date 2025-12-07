# backend/app/services/stabling_optimizer.py
import math
import re
from typing import List, Dict, Any, Tuple, Set, Optional
from datetime import datetime, timedelta
import logging
from app.config import settings
from app.models.trainset import (
    FleetSummary, DepotAllocation, BayAssignment, OptimizationKPIs, StablingGeometryResponse
)

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
            depot_assignments, unassigned_trainsets = self._group_trainsets_by_depot(trainsets, induction_decisions)
            
            optimized_layout = {}
            total_shunting_time = 0
            total_turnout_time = 0
            
            for depot_name, depot_trainsets in depot_assignments.items():
                if depot_name not in self.depot_layouts:
                    # This should not happen after fix, but keep as safety check
                    logger.warning(f"Unknown depot '{depot_name}' in depot_assignments (should have been filtered)")
                    continue
                
                # Optimize bay assignments for this depot
                depot_optimization = await self._optimize_depot_layout(
                    depot_name, depot_trainsets
                )
                
                optimized_layout[depot_name] = depot_optimization
                total_shunting_time += depot_optimization["total_shunting_time"]
                total_turnout_time += depot_optimization["total_turnout_time"]
                
                # Collect unassigned trainsets from this depot
                if "unassigned" in depot_optimization:
                    unassigned_trainsets.extend(depot_optimization["unassigned"])
            
            # Calculate overall efficiency metrics
            efficiency_metrics = self._calculate_efficiency_metrics(
                optimized_layout, total_shunting_time, total_turnout_time
            )
            
            # Calculate capacity statistics
            total_capacity = sum(
                self.depot_layouts[depot_name]["total_bays"]
                for depot_name in depot_assignments.keys()
                if depot_name in self.depot_layouts
            )
            total_assigned = sum(
                len(layout.get("bay_assignments", {}))
                for layout in optimized_layout.values()
            )
            total_requested = sum(len(trainsets) for trainsets in depot_assignments.values())
            
            result = {
                "optimized_layout": optimized_layout,
                "efficiency_metrics": efficiency_metrics,
                "total_shunting_time": total_shunting_time,
                "total_turnout_time": total_turnout_time,
                "optimization_timestamp": datetime.now().isoformat(),
                "capacity_stats": {
                    "total_bays_available": total_capacity,
                    "total_trainsets_assigned": total_assigned,
                    "total_trainsets_requested": total_requested,
                    "capacity_utilization": round(total_assigned / total_capacity * 100, 2) if total_capacity > 0 else 0.0
                }
            }
            
            # Add unassigned trainsets if any and set capacity warning flag
            if unassigned_trainsets:
                result["unassigned"] = unassigned_trainsets
                result["capacity_warning"] = True
                result["capacity_warning_message"] = (
                    f"Capacity exceeded: {len(unassigned_trainsets)} trainsets could not be assigned bays. "
                    f"Requested: {total_requested}, Available: {total_capacity}, Assigned: {total_assigned}"
                )
                logger.warning(
                    f"Stabling optimization: {len(unassigned_trainsets)} trainsets could not be assigned bays "
                    f"(capacity: {total_capacity}, requested: {total_requested}, assigned: {total_assigned})"
                )
            else:
                result["capacity_warning"] = False
            
            return result
        except Exception as e:
            logger.error(f"Stabling geometry optimization failed: {e}", exc_info=True)
            # Return empty/safe result or re-raise. Re-raising is safer for now.
            raise
    
    def _generate_fleet_summary(self, trainsets: List[Dict[str, Any]], 
                                induction_decisions: List[Dict[str, Any]],
                                fleet_req: Optional[Dict[str, Any]] = None) -> FleetSummary:
        """Generate fleet-level summary statistics"""
        total_trainsets = len(trainsets)
        
        # Count decisions by type
        decision_map = {d.get("trainset_id"): d.get("decision") for d in induction_decisions if isinstance(d, dict)}
        actual_induct = sum(1 for d in induction_decisions if d.get("decision") == "INDUCT")
        actual_standby = sum(1 for d in induction_decisions if d.get("decision") == "STANDBY")
        maintenance_count = sum(1 for d in induction_decisions if d.get("decision") == "MAINTENANCE")
        
        # Count eligible (passed Tier 1 filters)
        eligible_count = actual_induct + actual_standby
        
        # Get fleet requirements
        required_service_trains = fleet_req.get("required_service_trains", 0) if fleet_req else 0
        standby_buffer = fleet_req.get("standby_buffer", 0) if fleet_req else 0
        total_required = fleet_req.get("total_required_trains", 0) if fleet_req else 0
        
        service_shortfall = max(0, required_service_trains - actual_induct)
        compliance_rate = (actual_induct / required_service_trains) if required_service_trains > 0 else 0.0
        
        return FleetSummary(
            total_trainsets=total_trainsets,
            required_service_trains=required_service_trains,
            standby_buffer=standby_buffer,
            total_required_trains=total_required,
            eligible_count=eligible_count,
            actual_induct_count=actual_induct,
            actual_standby_count=actual_standby,
            maintenance_count=maintenance_count,
            service_shortfall=service_shortfall,
            compliance_rate=round(compliance_rate, 3)
        )
    
    def _generate_depot_allocation(self, depot_assignments: Dict[str, List[Dict[str, Any]]],
                                   optimized_layout: Dict[str, Any]) -> List[DepotAllocation]:
        """Generate depot-level allocation breakdown"""
        allocations = []
        
        for depot_name, depot_trainsets in depot_assignments.items():
            if depot_name not in self.depot_layouts:
                continue
            
            depot_layout = self.depot_layouts[depot_name]
            depot_opt = optimized_layout.get(depot_name, {})
            bay_assignments = depot_opt.get("bay_assignments", {})
            
            # Count trains by decision type in this depot
            service_trains = sum(1 for t in depot_trainsets 
                               if t.get("induction_decision", {}).get("decision") == "INDUCT")
            standby_trains = sum(1 for t in depot_trainsets 
                              if t.get("induction_decision", {}).get("decision") == "STANDBY")
            maintenance_trains = sum(1 for t in depot_trainsets 
                                   if t.get("induction_decision", {}).get("decision") == "MAINTENANCE")
            
            # Get capacities
            service_bay_capacity = len(depot_layout.get("service_bays", []))
            maintenance_bay_capacity = len(depot_layout.get("maintenance_bays", []))
            total_bay_capacity = depot_layout.get("total_bays", 0)
            
            # Check for capacity warnings
            capacity_warning = (
                service_trains > service_bay_capacity or
                maintenance_trains > maintenance_bay_capacity or
                (service_trains + standby_trains + maintenance_trains) > total_bay_capacity
            )
            
            allocations.append(DepotAllocation(
                depot_name=depot_name,
                service_trains=service_trains,
                standby_trains=standby_trains,
                maintenance_trains=maintenance_trains,
                total_trains=service_trains + standby_trains + maintenance_trains,
                service_bay_capacity=service_bay_capacity,
                maintenance_bay_capacity=maintenance_bay_capacity,
                total_bay_capacity=total_bay_capacity,
                capacity_warning=capacity_warning
            ))
        
        return allocations
    
    def _generate_bay_layout(self, optimized_layout: Dict[str, Any],
                            trainsets: List[Dict[str, Any]],
                            induction_decisions: List[Dict[str, Any]]) -> Dict[str, List[BayAssignment]]:
        """Generate enhanced bay layout with role and details"""
        bay_layout = {}
        
        # Create decision map for quick lookup
        decision_map = {}
        for d in induction_decisions:
            if isinstance(d, dict):
                decision_map[d.get("trainset_id")] = d
            elif hasattr(d, "trainset_id"):
                decision_map[d.trainset_id] = d
        
        # Create trainset map
        trainset_map = {t.get("trainset_id"): t for t in trainsets}
        
        for depot_name, depot_opt in optimized_layout.items():
            if depot_name not in self.depot_layouts:
                continue
            
            depot_layout = self.depot_layouts[depot_name]
            bay_assignments = depot_opt.get("bay_assignments", {})
            
            # Create reverse map: bay -> trainset_id
            bay_to_trainset = {bay: ts_id for ts_id, bay in bay_assignments.items()}
            
            depot_bays = []
            
            # Process all bays in depot
            for bay_id in range(1, depot_layout["total_bays"] + 1):
                trainset_id = bay_to_trainset.get(bay_id)
                bay_info = depot_layout["bay_positions"].get(bay_id, {})
                
                # Determine role based on bay type and decision
                role = "EMPTY"
                if trainset_id:
                    decision = decision_map.get(trainset_id, {})
                    if isinstance(decision, dict):
                        decision_type = decision.get("decision", "STANDBY")
                    else:
                        decision_type = getattr(decision, "decision", "STANDBY")
                    
                    # Map decision to role
                    if decision_type == "INDUCT":
                        role = "SERVICE"
                    elif decision_type == "MAINTENANCE":
                        role = "MAINTENANCE"
                    else:
                        role = "STANDBY"
                
                # Get turnout time and distance
                turnout_time = bay_info.get("turnout_time") if trainset_id else None
                distance_to_exit = None
                if bay_info.get("x") is not None and bay_info.get("y") is not None:
                    # Calculate distance to exit (assume exit is at y=0, x=0)
                    distance_to_exit = int(math.sqrt(bay_info["x"]**2 + bay_info["y"]**2))
                
                # Generate notes
                notes = None
                if trainset_id:
                    trainset = trainset_map.get(trainset_id, {})
                    note_parts = []
                    
                    # Branding info
                    branding = trainset.get("branding", {})
                    if isinstance(branding, dict) and branding.get("current_advertiser"):
                        note_parts.append(f"Active wrap: {branding.get('current_advertiser')}")
                    
                    # Job cards
                    job_cards = trainset.get("job_cards", {})
                    if isinstance(job_cards, dict):
                        open_cards = job_cards.get("open_cards", 0)
                        if open_cards > 0:
                            note_parts.append(f"{open_cards} open job cards")
                    
                    # Mileage
                    mileage = trainset.get("current_mileage", 0)
                    if mileage > 40000:
                        note_parts.append(f"High mileage: {int(mileage)} km")
                    
                    if note_parts:
                        notes = "; ".join(note_parts)
                
                depot_bays.append(BayAssignment(
                    bay_id=bay_id,
                    role=role,
                    trainset_id=trainset_id,
                    turnout_time_min=turnout_time,
                    distance_to_exit_m=distance_to_exit,
                    notes=notes
                ))
            
            bay_layout[depot_name] = depot_bays
        
        return bay_layout
    
    def _generate_warnings(self, fleet_summary: FleetSummary,
                          depot_allocations: List[DepotAllocation],
                          unassigned_trainsets: List[Dict[str, Any]]) -> List[str]:
        """Generate operational warnings"""
        warnings = []
        
        # Service shortfall warning
        if fleet_summary.service_shortfall > 0:
            warnings.append(
                f"Service shortfall: {fleet_summary.service_shortfall} trains short of required "
                f"{fleet_summary.required_service_trains} service trains."
            )
        
        # Standby buffer warning
        if fleet_summary.actual_standby_count < fleet_summary.standby_buffer:
            warnings.append(
                f"Standby buffer shortfall: Only {fleet_summary.actual_standby_count} standby trains available "
                f"vs {fleet_summary.standby_buffer} requested."
            )
        
        # Depot capacity warnings
        for depot in depot_allocations:
            if depot.capacity_warning:
                if depot.service_trains > depot.service_bay_capacity:
                    warnings.append(
                        f"{depot.depot_name} service bay capacity exceeded by "
                        f"{depot.service_trains - depot.service_bay_capacity} rake(s). "
                        f"One or more SERVICE rakes are parked in non-service bays."
                    )
                if depot.maintenance_trains > depot.maintenance_bay_capacity:
                    warnings.append(
                        f"{depot.depot_name} maintenance bay capacity exceeded by "
                        f"{depot.maintenance_trains - depot.maintenance_bay_capacity} rake(s)."
                    )
        
        # Unassigned trainsets warning
        if unassigned_trainsets:
            warnings.append(
                f"{len(unassigned_trainsets)} trainset(s) could not be assigned bays due to capacity constraints."
            )
        
        return warnings
    
    async def generate_rich_stabling_geometry(self, trainsets: List[Dict[str, Any]],
                                            induction_decisions: List[Dict[str, Any]],
                                            fleet_req: Optional[Dict[str, Any]] = None) -> StablingGeometryResponse:
        """Generate rich, structured stabling geometry response with intelligence"""
        try:
            # First run the standard optimization
            standard_result = await self.optimize_stabling_geometry(trainsets, induction_decisions)
            
            # Extract data from standard result
            optimized_layout = standard_result.get("optimized_layout", {})
            unassigned_trainsets = standard_result.get("unassigned", [])
            total_shunting_time = standard_result.get("total_shunting_time", 0)
            total_turnout_time = standard_result.get("total_turnout_time", 0)
            efficiency_metrics = standard_result.get("efficiency_metrics", {})
            
            # Group trainsets by depot for allocation calculation
            depot_assignments, _ = self._group_trainsets_by_depot(trainsets, induction_decisions)
            
            # Generate structured components
            fleet_summary = self._generate_fleet_summary(trainsets, induction_decisions, fleet_req)
            depot_allocation = self._generate_depot_allocation(depot_assignments, optimized_layout)
            bay_layout = self._generate_bay_layout(optimized_layout, trainsets, induction_decisions)
            warnings = self._generate_warnings(fleet_summary, depot_allocation, unassigned_trainsets)
            
            # Calculate KPIs
            optimized_positions = sum(
                len(depot.get("bay_assignments", {}))
                for depot in optimized_layout.values()
            )
            
            efficiency_improvement = efficiency_metrics.get("overall_efficiency", 0.0) * 100
            energy_savings = efficiency_metrics.get("energy_savings", 0.0)
            
            optimization_kpis = OptimizationKPIs(
                optimized_positions=optimized_positions,
                total_shunting_time_min=total_shunting_time,
                total_turnout_time_min=total_turnout_time,
                efficiency_improvement_pct=round(efficiency_improvement, 2),
                energy_savings_kwh=round(energy_savings, 2) if energy_savings else None,
                night_movements_reduced=None  # Can be calculated if baseline is available
            )
            
            return StablingGeometryResponse(
                fleet_summary=fleet_summary,
                depot_allocation=depot_allocation,
                bay_layout=bay_layout,
                optimization_kpis=optimization_kpis,
                warnings=warnings,
                optimization_timestamp=standard_result.get("optimization_timestamp", datetime.now().isoformat())
            )
            
        except Exception as e:
            logger.error(f"Stabling geometry optimization failed: {e}")
            # Return a safe empty response instead of raising to avoid API 500s
            return {
                "optimized_layout": {},
                "efficiency_metrics": {
                    "overall_efficiency": 0.0,
                    "shunting_efficiency": 0.0,
                    "energy_savings": 0.0,
                    "time_savings": 0.0,
                    "depot_scores": {}
                },
                "total_shunting_time": 0,
                "total_turnout_time": 0,
                "optimization_timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _group_trainsets_by_depot(self, trainsets: List[Dict[str, Any]], 
                                decisions: List[Dict[str, Any]]) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Group trainsets by depot based on current location and decisions.
        
        Returns:
            Tuple of (depot_groups dict, unassigned_trainsets list)
            Unassigned trainsets are those with unknown depots.
        """
        depot_groups = {}
        unassigned_trainsets = []
        
        for trainset in trainsets:
            current_loc = trainset.get("current_location") or {}
            depot = current_loc.get("depot", "Aluva")
            
            # Check if depot is known
            if depot not in self.depot_layouts:
                trainset_id = trainset.get("trainset_id", "UNKNOWN")
                if settings.warn_on_unknown_depot:
                    logger.warning(f"Trainset {trainset_id} has unknown depot '{depot}', marking as unassigned")
                unassigned_trainsets.append({
                    "trainset_id": trainset_id,
                    "reason": "unknown_depot",
                    "depot": depot,
                    "message": f"Depot '{depot}' is not in known depot layouts"
                })
                continue
            
            if depot not in depot_groups:
                depot_groups[depot] = []
            
            # Add decision information
            decision = next((d for d in decisions if d.get("trainset_id") == trainset.get("trainset_id")), None)
            # Ensure dict to avoid NoneType errors downstream
            trainset["induction_decision"] = decision or {}
            depot_groups[depot].append(trainset)
        
        if unassigned_trainsets:
            unknown_depots = set(ts["depot"] for ts in unassigned_trainsets)
            logger.warning(f"Found {len(unassigned_trainsets)} trainsets with unknown depots: {unknown_depots}")
        
        return depot_groups, unassigned_trainsets
    
    async def _optimize_depot_layout(self, depot_name: str, trainsets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize bay assignments for a specific depot with conflict-free assignment"""
        depot_layout = self.depot_layouts[depot_name]
        bay_assignments = {}
        used_bays: Set[int] = set()
        unassigned_trainsets = []
        
        # Separate trainsets by decision type
        induct_trainsets = [t for t in trainsets if t.get("induction_decision", {}).get("decision") == "INDUCT"]
        maintenance_trainsets = [t for t in trainsets if t.get("induction_decision", {}).get("decision") == "MAINTENANCE"]
        standby_trainsets = [t for t in trainsets if t.get("induction_decision", {}).get("decision") == "STANDBY"]
        
        # Assign bays based on priority and optimization (service > maintenance > standby)
        # Each function updates used_bays to prevent conflicts
        service_assignments, service_unassigned = self._assign_service_bays(induct_trainsets, depot_layout, used_bays)
        bay_assignments.update(service_assignments)
        used_bays.update(service_assignments.values())
        unassigned_trainsets.extend(service_unassigned)
        
        maintenance_assignments, maintenance_unassigned = self._assign_maintenance_bays(maintenance_trainsets, depot_layout, used_bays)
        bay_assignments.update(maintenance_assignments)
        used_bays.update(maintenance_assignments.values())
        unassigned_trainsets.extend(maintenance_unassigned)
        
        standby_assignments, standby_unassigned = self._assign_standby_bays(standby_trainsets, depot_layout, used_bays)
        bay_assignments.update(standby_assignments)
        used_bays.update(standby_assignments.values())
        unassigned_trainsets.extend(standby_unassigned)
        
        # Validation: Check for duplicate bay assignments (should never happen now, but safety check)
        bay_to_trainsets = {}
        for trainset_id, bay in bay_assignments.items():
            if bay in bay_to_trainsets:
                bay_to_trainsets[bay].append(trainset_id)
            else:
                bay_to_trainsets[bay] = [trainset_id]
        
        duplicates = {bay: trainsets for bay, trainsets in bay_to_trainsets.items() if len(trainsets) > 1}
        if duplicates:
            error_msg = f"CRITICAL: Duplicate bay assignments detected in depot {depot_name}: {duplicates}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate shunting operations needed
        shunting_operations = self._calculate_shunting_operations(trainsets, bay_assignments, depot_layout)
        
        # Calculate total times (ensure numeric)
        total_shunting_time = sum(
            int(op.get("estimated_time", 0)) if isinstance(op.get("estimated_time"), (int, float)) else 0
            for op in shunting_operations
        )
        total_turnout_time = sum(
            depot_layout["bay_positions"][bay]["turnout_time"] 
            for bay in bay_assignments.values() 
            if bay in depot_layout["bay_positions"]
        )
        
        result = {
            "depot": depot_name,
            "bay_assignments": bay_assignments,
            "shunting_operations": shunting_operations,
            "total_shunting_time": total_shunting_time,
            "total_turnout_time": total_turnout_time,
            "efficiency_score": self._calculate_depot_efficiency(bay_assignments, depot_layout)
        }
    
        if unassigned_trainsets:
            result["unassigned"] = unassigned_trainsets
            if settings.warn_on_capacity_exceeded:
                logger.warning(f"Depot {depot_name}: {len(unassigned_trainsets)} trainsets could not be assigned bays (capacity exceeded)")
        
        return result
    
    def _assign_service_bays(self, induct_trainsets: List[Dict[str, Any]], depot_layout: Dict[str, Any], used_bays: Set[int]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """Assign service bays to trainsets for induction (priority: closest to turnout).
        
        Args:
            induct_trainsets: List of trainsets to assign service bays
            depot_layout: Depot layout configuration
            used_bays: Set of already-used bay numbers (updated in-place)
            
        Returns:
            Tuple of (assignments dict, unassigned_trainsets list)
        """
        assignments = {}
        unassigned = []
        service_bays = depot_layout["service_bays"]
        
        # Filter out already-used bays
        available_bays = [bay for bay in service_bays if bay not in used_bays]
        
        # Sort available service bays by turnout time (fastest first)
        available_bays_sorted = sorted(available_bays, key=lambda bay: depot_layout["bay_positions"][bay]["turnout_time"])
        
        for i, trainset in enumerate(induct_trainsets):
            if i < len(available_bays_sorted):
                bay = available_bays_sorted[i]
                assignments[trainset["trainset_id"]] = bay
            else:
                # Capacity exceeded
                unassigned.append({
                    "trainset_id": trainset.get("trainset_id", "UNKNOWN"),
                    "reason": "no_capacity",
                    "depot": depot_layout.get("depot", "UNKNOWN"),
                    "message": f"No available service bays (capacity: {len(service_bays)}, used: {len(used_bays)}, needed: {len(induct_trainsets)})"
                })
        
        return assignments, unassigned
    
    def _assign_maintenance_bays(self, maintenance_trainsets: List[Dict[str, Any]], depot_layout: Dict[str, Any], used_bays: Set[int]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """Assign maintenance bays to trainsets requiring maintenance.
        
        Args:
            maintenance_trainsets: List of trainsets requiring maintenance
            depot_layout: Depot layout configuration
            used_bays: Set of already-used bay numbers (updated in-place)
            
        Returns:
            Tuple of (assignments dict, unassigned_trainsets list)
        """
        assignments = {}
        unassigned = []
        maintenance_bays = depot_layout["maintenance_bays"]
        
        # Filter out already-used bays
        available_bays = [bay for bay in maintenance_bays if bay not in used_bays]
        
        for i, trainset in enumerate(maintenance_trainsets):
            if i < len(available_bays):
                bay = available_bays[i]
                assignments[trainset["trainset_id"]] = bay
            else:
                # Capacity exceeded
                unassigned.append({
                    "trainset_id": trainset.get("trainset_id", "UNKNOWN"),
                    "reason": "no_capacity",
                    "depot": depot_layout.get("depot", "UNKNOWN"),
                    "message": f"No available maintenance bays (capacity: {len(maintenance_bays)}, used: {len(used_bays)}, needed: {len(maintenance_trainsets)})"
                })
        
        return assignments, unassigned
    
    def _assign_standby_bays(self, standby_trainsets: List[Dict[str, Any]], depot_layout: Dict[str, Any], used_bays: Set[int]) -> Tuple[Dict[str, int], List[Dict[str, Any]]]:
        """Assign standby bays to trainsets (any available bay, excluding already-used bays).
        
        Args:
            standby_trainsets: List of trainsets on standby
            depot_layout: Depot layout configuration
            used_bays: Set of already-used bay numbers (updated in-place)
            
        Returns:
            Tuple of (assignments dict, unassigned_trainsets list)
        """
        assignments = {}
        unassigned = []
        all_bays = list(range(1, depot_layout["total_bays"] + 1))
        
        # Filter out already-used bays
        available_bays = [bay for bay in all_bays if bay not in used_bays]
        
        for i, trainset in enumerate(standby_trainsets):
            if i < len(available_bays):
                bay = available_bays[i]
                assignments[trainset["trainset_id"]] = bay
            else:
                # Capacity exceeded
                unassigned.append({
                    "trainset_id": trainset.get("trainset_id", "UNKNOWN"),
                    "reason": "no_capacity",
                    "depot": depot_layout.get("depot", "UNKNOWN"),
                    "message": f"No available bays for standby (total bays: {depot_layout['total_bays']}, used: {len(used_bays)}, needed: {len(standby_trainsets)})"
                })
        
        return assignments, unassigned
    
    def _calculate_shunting_operations(self, trainsets: List[Dict[str, Any]], 
                                    bay_assignments: Dict[str, int], 
                                    depot_layout: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate shunting operations needed to move trainsets to assigned bays"""
        operations = []
        
        for trainset in trainsets:
            current_loc = trainset.get("current_location") or {}
            current_bay = self._extract_bay_number(current_loc.get("bay", ""))
            assigned_bay = bay_assignments.get(trainset["trainset_id"])
            
            # Skip if current_bay is None (unparseable) or if no assigned bay
            if current_bay is None:
                logger.debug(f"Trainset {trainset.get('trainset_id')} has unparseable current bay, skipping shunting calculation")
                continue
            
            if assigned_bay and current_bay != assigned_bay:
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
    
    def _extract_bay_number(self, bay_string: str) -> Optional[int]:
        """Extract bay number from bay string with robust multi-format support.
        
        Supports formats:
        - "_BAY_5", "_BAY_05" -> 5
        - "Bay 5", "Bay 05" -> 5
        - "B-5", "B-05" -> 5
        - "B5", "B05" -> 5
        - "bay_5", "bay_05" -> 5
        - "5", "05" -> 5
        
        Returns:
            Bay number as int, or None if unparseable
        """
        if not bay_string:
            return None
        
        bay_string = str(bay_string).strip()
        
        # Try regex patterns in order of specificity
        patterns = [
            r'_BAY_(\d+)',           # "_BAY_5", "_BAY_05"
            r'Bay\s+(\d+)',          # "Bay 5", "Bay 05"
            r'B-(\d+)',              # "B-5", "B-05"
            r'B(\d+)',               # "B5", "B05"
            r'bay_(\d+)',            # "bay_5", "bay_05"
            r'^(\d+)$',              # "5", "05"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, bay_string, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        # If no pattern matches, log warning and return None
        logger.warning(f"Could not parse bay number from string: '{bay_string}', returning None")
        return None
    
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
        total_trainsets = sum(len(depot.get("bay_assignments", {})) for depot in optimized_layout.values())

        if total_trainsets == 0:
            return {
                "overall_efficiency": 0.0,
                "shunting_efficiency": 0.0,
                "energy_savings": 0.0,
                "time_savings": 0.0,
                "depot_scores": {depot: layout.get("efficiency_score", 0.0) for depot, layout in optimized_layout.items()}
            }

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
        """Generate detailed shunting schedule for operations team
        
        Ensures every operation has numeric estimated_time field (required for calculations).
        """
        schedule = []
        
        for depot_name, depot_layout in optimized_layout.items():
            for operation in depot_layout.get("shunting_operations", []):
                # Ensure estimated_time is numeric (required field)
                estimated_time = operation.get("estimated_time", 0)
                if not isinstance(estimated_time, (int, float)):
                    logger.warning(f"Shunting operation for {operation.get('trainset_id')} has non-numeric estimated_time: {estimated_time}")
                    try:
                        estimated_time = int(float(estimated_time))
                    except (ValueError, TypeError):
                        estimated_time = 0
                        logger.error(f"Could not convert estimated_time to numeric for {operation.get('trainset_id')}")
                
                schedule.append({
                    "depot": depot_name,
                    "trainset_id": operation.get("trainset_id", "UNKNOWN"),
                    "operation": f"Move from Bay {operation.get('from_bay', '?')} to Bay {operation.get('to_bay', '?')}",
                    "estimated_time": int(estimated_time),  # Numeric field (required)
                    "estimated_duration": f"{estimated_time} minutes",  # String field (for display)
                    "complexity": operation.get("complexity", "MEDIUM"),
                    "scheduled_time": "21:00-23:00",  # Nightly shunting window
                    "crew_required": "2 operators" if operation.get("complexity") == "HIGH" else "1 operator"
                })
        
        # Sort by estimated_time (numeric) not estimated_duration (string)
        return sorted(schedule, key=lambda x: x.get("estimated_time", 0))
