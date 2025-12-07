"""What-If Simulation Service - Runs deterministic scenario comparisons"""
import copy
import json
import uuid
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path

import numpy as np
import torch
import random

from app.services.optimizer import TrainInductionOptimizer
from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.models.trainset import OptimizationRequest, OptimizationWeights
from app.utils.snapshot import capture_snapshot
from app.config import settings

logger = logging.getLogger(__name__)

# Ensure simulation_runs directory exists
SIMULATION_RUNS_DIR = Path(__file__).parent.parent.parent / "simulation_runs"
SIMULATION_RUNS_DIR.mkdir(exist_ok=True)


def _set_nested_value(obj: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested value in a dictionary using dot-separated path (e.g., 'fitness.telecom.valid_until')"""
    keys = path.split('.')
    current = obj
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _apply_overrides(snapshot: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply scenario overrides to snapshot.
    
    Supports:
    - required_service_count: Override train count requirement (manual override)
    - service_date: Override service date for timetable lookup
    - override_train_attributes: Path-based nested setter (e.g., "fitness.telecom.valid_until")
    - depot_layout_override: Override depot layouts
    - cleaning_capacity_override: Override cleaning capacity
    - force_decisions: Force specific trainset decisions
    - inject_delay_events: Inject delay events
    - random_seed: Seed for deterministic randomness
    """
    scenario_snapshot = copy.deepcopy(snapshot)
    
    # Apply random seed if provided
    if "random_seed" in scenario:
        seed = scenario["random_seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Applied random seed: {seed}")
    
    # Override required_service_count (stored in config)
    if "required_service_count" in scenario:
        scenario_snapshot["config"]["required_service_count"] = scenario["required_service_count"]
    
    # Override service_date (stored in config)
    if "service_date" in scenario:
        scenario_snapshot["config"]["service_date"] = scenario["service_date"]
    
    # Override train attributes using path-based setter
    if "override_train_attributes" in scenario:
        for trainset_id, overrides in scenario["override_train_attributes"].items():
            # Find trainset in snapshot
            for trainset in scenario_snapshot["trainsets"]:
                if trainset.get("trainset_id") == trainset_id:
                    for path, value in overrides.items():
                        _set_nested_value(trainset, path, value)
                    logger.debug(f"Applied overrides to trainset {trainset_id}: {overrides}")
                    break
    
    # Override depot layouts
    if "depot_layout_override" in scenario:
        scenario_snapshot["depot_layouts"].update(scenario["depot_layout_override"])
    
    # Override cleaning capacity
    if "cleaning_capacity_override" in scenario:
        scenario_snapshot["cleaning_slots"].update(scenario["cleaning_capacity_override"])
    
    # Force decisions (will be applied during optimization)
    if "force_decisions" in scenario:
        scenario_snapshot["force_decisions"] = scenario["force_decisions"]
    
    # Inject delay events
    if "inject_delay_events" in scenario:
        scenario_snapshot["delay_events"] = scenario["inject_delay_events"]
    
    return scenario_snapshot


def _normalize_decision_explain(decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure decision has an 'explain' field.
    Synthesizes explanation from available fields if missing.
    """
    if "explain" in decision and decision["explain"]:
        return decision
    
    # Build explanation from available fields
    explanation_parts = []
    
    # Add decision type
    decision_type = decision.get("decision", "UNKNOWN")
    explanation_parts.append(f"Decision: {decision_type}")
    
    # Add reasons if available
    if decision.get("reasons"):
        if isinstance(decision["reasons"], list):
            explanation_parts.extend(decision["reasons"])
        elif isinstance(decision["reasons"], str):
            explanation_parts.append(decision["reasons"])
    
    # Add confidence score
    confidence = decision.get("confidence_score", 0)
    if confidence:
        explanation_parts.append(f"Confidence: {confidence:.0%}")
    
    # Add score if available
    score = decision.get("score")
    if score is not None:
        explanation_parts.append(f"Score: {score:.2f}")
    
    # Add violations if present
    if decision.get("violations"):
        violations = decision["violations"]
        if isinstance(violations, list):
            explanation_parts.append(f"Violations: {', '.join(violations)}")
        elif isinstance(violations, str):
            explanation_parts.append(f"Violations: {violations}")
    
    # Synthesize final explanation
    decision["explain"] = ". ".join(explanation_parts) if explanation_parts else "No explanation available"
    
    return decision


def _normalize_decisions_list(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize a list of decisions to ensure each has an explain field"""
    return [_normalize_decision_explain(d) for d in decisions]


def _compute_kpis(decisions: List[Dict[str, Any]], stabling_geometry: Dict[str, Any]) -> Dict[str, Any]:
    """Compute Key Performance Indicators from optimization results"""
    inducted_count = sum(1 for d in decisions if d.get("decision") == "INDUCT")
    standby_count = sum(1 for d in decisions if d.get("decision") == "STANDBY")
    maintenance_count = sum(1 for d in decisions if d.get("decision") == "MAINTENANCE")
    
    total_shunting_time = stabling_geometry.get("total_shunting_time", 0)
    total_turnout_time = stabling_geometry.get("total_turnout_time", 0)
    
    # Count shunting operations
    num_shunt_ops = 0
    optimized_layout = stabling_geometry.get("optimized_layout", {})
    for depot_data in optimized_layout.values():
        num_shunt_ops += len(depot_data.get("shunting_operations", []))
    
    # Count unassigned trainsets
    unassigned = stabling_geometry.get("unassigned", [])
    num_unassigned = len(unassigned)
    
    # Efficiency improvement
    efficiency_metrics = stabling_geometry.get("efficiency_metrics", {})
    overall_efficiency = efficiency_metrics.get("overall_efficiency", 0.0)
    efficiency_improvement = round(float(overall_efficiency) * 100, 2) if overall_efficiency else 0.0
    
    # Ensure all numeric fields are actually numeric (not strings)
    return {
        "num_inducted_trains": int(inducted_count),
        "num_standby_trains": int(standby_count),
        "num_maintenance_trains": int(maintenance_count),
        "total_shunting_time": int(total_shunting_time) if isinstance(total_shunting_time, (int, float)) else 0,
        "total_turnout_time": int(total_turnout_time) if isinstance(total_turnout_time, (int, float)) else 0,
        "num_shunt_ops": int(num_shunt_ops),
        "num_unassigned": int(num_unassigned),
        "efficiency_improvement": float(efficiency_improvement) if isinstance(efficiency_improvement, (int, float)) else 0.0,
        "overall_efficiency": float(overall_efficiency) if overall_efficiency and isinstance(overall_efficiency, (int, float)) else 0.0
    }


def _generate_explain_log(baseline_kpis: Dict[str, Any], scenario_kpis: Dict[str, Any], 
                         scenario: Dict[str, Any]) -> List[str]:
    """Generate human-readable explanation of how scenario changed results compared to baseline"""
    explain_log = []
    
    # Compare inducted trains
    baseline_inducted = baseline_kpis.get("num_inducted_trains", 0)
    scenario_inducted = scenario_kpis.get("num_inducted_trains", 0)
    delta_inducted = scenario_inducted - baseline_inducted
    if delta_inducted != 0:
        explain_log.append(f"Inducted trains changed by {delta_inducted:+d} (baseline: {baseline_inducted}, scenario: {scenario_inducted})")
    
    # Compare shunting time
    baseline_shunt = baseline_kpis.get("total_shunting_time", 0)
    scenario_shunt = scenario_kpis.get("total_shunting_time", 0)
    delta_shunt = scenario_shunt - baseline_shunt
    if delta_shunt != 0:
        explain_log.append(f"Total shunting time changed by {delta_shunt:+d} minutes (baseline: {baseline_shunt}, scenario: {scenario_shunt})")
    
    # Compare shunting operations
    baseline_ops = baseline_kpis.get("num_shunt_ops", 0)
    scenario_ops = scenario_kpis.get("num_shunt_ops", 0)
    delta_ops = scenario_ops - baseline_ops
    if delta_ops != 0:
        explain_log.append(f"Number of shunting operations changed by {delta_ops:+d} (baseline: {baseline_ops}, scenario: {scenario_ops})")
    
    # Compare unassigned
    baseline_unassigned = baseline_kpis.get("num_unassigned", 0)
    scenario_unassigned = scenario_kpis.get("num_unassigned", 0)
    delta_unassigned = scenario_unassigned - baseline_unassigned
    if delta_unassigned != 0:
        explain_log.append(f"Unassigned trainsets changed by {delta_unassigned:+d} (baseline: {baseline_unassigned}, scenario: {scenario_unassigned})")
    
    # Compare efficiency
    baseline_eff = baseline_kpis.get("efficiency_improvement", 0.0)
    scenario_eff = scenario_kpis.get("efficiency_improvement", 0.0)
    delta_eff = scenario_eff - baseline_eff
    if abs(delta_eff) > 0.01:
        explain_log.append(f"Efficiency improvement changed by {delta_eff:+.2f}% (baseline: {baseline_eff:.2f}%, scenario: {scenario_eff:.2f}%)")
    
    # Scenario-specific explanations
    if "override_train_attributes" in scenario:
        explain_log.append(f"Applied attribute overrides to {len(scenario['override_train_attributes'])} trainsets")
    
    if "force_decisions" in scenario:
        explain_log.append(f"Force decisions applied to {len(scenario['force_decisions'])} trainsets")
    
    if "required_service_count" in scenario:
        explain_log.append(f"Service train count requirement changed to {scenario['required_service_count']} trains")
    
    if "service_date" in scenario:
        explain_log.append(f"Service date changed to {scenario['service_date']}")
    
    if not explain_log:
        explain_log.append("No significant changes detected between baseline and scenario")
    
    return explain_log


async def run_whatif(scenario: Dict[str, Any], snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run What-If simulation comparing baseline vs scenario.
    
    Args:
        scenario: Scenario configuration with overrides
        snapshot: Optional snapshot (if None, captures current state)
    
    Returns:
        Dictionary with keys:
        - simulation_id: Unique simulation ID
        - timestamp: Simulation timestamp
        - baseline: Baseline results (KPI dict)
        - scenario: Scenario results (KPI dict)
        - deltas: Delta values (scenario - baseline)
        - explain_log: Human-readable explanation
        - results: ALWAYS an array of two objects [baseline_result, scenario_result]
    """
    try:
        simulation_id = str(uuid.uuid4())
        logger.info(f"Starting What-If simulation {simulation_id}")
        
        # Capture baseline snapshot if not provided
        if snapshot is None:
            snapshot = await capture_snapshot()
        
        # Create scenario snapshot with overrides
        scenario_snapshot = _apply_overrides(snapshot, scenario)
        
        # Prepare optimization request with weights if provided
        required_count = scenario.get("required_service_count", snapshot["config"].get("required_service_count"))
        service_date = scenario.get("service_date", snapshot["config"].get("service_date"))
        
        # Create weights from scenario if provided, otherwise use defaults
        scenario_weights = None
        if "weights" in scenario and scenario["weights"]:
            try:
                scenario_weights = OptimizationWeights(**scenario["weights"])
                logger.info(f"Using custom weights from scenario: {scenario['weights']}")
            except Exception as e:
                logger.warning(f"Invalid weights in scenario, using defaults: {e}")
                scenario_weights = OptimizationWeights()
        else:
            scenario_weights = OptimizationWeights()
        
        request = OptimizationRequest(
            target_date=datetime.utcnow().date(),
            service_date=service_date,
            required_service_count=required_count,
            weights=scenario_weights
        )
        
        # Run baseline optimization (in-memory, no DB writes)
        logger.info("Running baseline optimization")
        baseline_optimizer = TrainInductionOptimizer()
        # Baseline uses default weights (no scenario weights)
        baseline_request = OptimizationRequest(
            target_date=datetime.utcnow().date(),
            service_date=snapshot["config"].get("service_date"),
            required_service_count=snapshot["config"].get("required_service_count"),
            weights=OptimizationWeights()  # Baseline always uses defaults
        )
        baseline_decisions, _ = await baseline_optimizer.optimize(
            snapshot["trainsets"], 
            baseline_request
        )
        baseline_decisions_dict = [d.dict() if hasattr(d, 'dict') else d for d in baseline_decisions]
        # Normalize decisions to ensure explain fields
        baseline_decisions_dict = _normalize_decisions_list(baseline_decisions_dict)
        
        baseline_stabling = StablingGeometryOptimizer()
        baseline_stabling_geometry = await baseline_stabling.optimize_stabling_geometry(
            snapshot["trainsets"],
            baseline_decisions_dict
        )
        baseline_kpis = _compute_kpis(baseline_decisions_dict, baseline_stabling_geometry)
        
        # Run scenario optimization (in-memory, no DB writes)
        logger.info("Running scenario optimization")
        scenario_optimizer = TrainInductionOptimizer()
        scenario_decisions, _ = await scenario_optimizer.optimize(
            scenario_snapshot["trainsets"],
            request
        )
        scenario_decisions_dict = [d.dict() if hasattr(d, 'dict') else d for d in scenario_decisions]
        # Normalize decisions to ensure explain fields
        scenario_decisions_dict = _normalize_decisions_list(scenario_decisions_dict)
        
        # Apply force_decisions if specified (override optimizer decisions)
        if "force_decisions" in scenario:
            for trainset_id, decision in scenario["force_decisions"].items():
                # Find and update decision
                for d in scenario_decisions_dict:
                    if d.get("trainset_id") == trainset_id:
                        d["decision"] = decision
                        logger.info(f"Force decision applied: {trainset_id} -> {decision}")
                        break
                else:
                    # If trainset not in decisions, add it
                    new_decision = {
                        "trainset_id": trainset_id,
                        "decision": decision,
                        "confidence_score": 1.0,
                        "score": 0.0,
                        "reasons": [f"Force decision: {decision}"],
                        "top_reasons": [],
                        "top_risks": [],
                        "violations": [],
                        "shap_values": []
                    }
                    # Normalize to add explain field
                    new_decision = _normalize_decision_explain(new_decision)
                    scenario_decisions_dict.append(new_decision)
        
        scenario_stabling = StablingGeometryOptimizer()
        scenario_stabling_geometry = await scenario_stabling.optimize_stabling_geometry(
            scenario_snapshot["trainsets"],
            scenario_decisions_dict
        )
        scenario_kpis = _compute_kpis(scenario_decisions_dict, scenario_stabling_geometry)
        
        # Compute deltas
        deltas = {
            "num_inducted_trains": scenario_kpis["num_inducted_trains"] - baseline_kpis["num_inducted_trains"],
            "num_standby_trains": scenario_kpis["num_standby_trains"] - baseline_kpis["num_standby_trains"],
            "num_maintenance_trains": scenario_kpis["num_maintenance_trains"] - baseline_kpis["num_maintenance_trains"],
            "total_shunting_time": scenario_kpis["total_shunting_time"] - baseline_kpis["total_shunting_time"],
            "total_turnout_time": scenario_kpis["total_turnout_time"] - baseline_kpis["total_turnout_time"],
            "num_shunt_ops": scenario_kpis["num_shunt_ops"] - baseline_kpis["num_shunt_ops"],
            "num_unassigned": scenario_kpis["num_unassigned"] - baseline_kpis["num_unassigned"],
            "efficiency_improvement": scenario_kpis["efficiency_improvement"] - baseline_kpis["efficiency_improvement"]
        }
        
        # Generate explain log
        explain_log = _generate_explain_log(baseline_kpis, scenario_kpis, scenario)
        
        # Build results array (ALWAYS an array)
        baseline_result = {
            "type": "baseline",
            "kpis": baseline_kpis,
            "decisions": baseline_decisions_dict,
            "stabling_geometry": baseline_stabling_geometry
        }
        
        scenario_result = {
            "type": "scenario",
            "kpis": scenario_kpis,
            "decisions": scenario_decisions_dict,
            "stabling_geometry": scenario_stabling_geometry
        }
        
        results = [baseline_result, scenario_result]
        
        # Build final response
        response = {
            "simulation_id": simulation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "baseline": baseline_kpis,
            "scenario": scenario_kpis,
            "deltas": deltas,
            "explain_log": explain_log,
            "results": results  # ALWAYS an array
        }
        
        # Save to file
        output_file = SIMULATION_RUNS_DIR / f"{simulation_id}.json"
        with open(output_file, 'w') as f:
            json.dump(response, f, indent=2, default=str)
        logger.info(f"Saved simulation results to {output_file}")
        
        return response
        
    except Exception as e:
        logger.error(f"What-If simulation failed: {e}", exc_info=True)
        raise

