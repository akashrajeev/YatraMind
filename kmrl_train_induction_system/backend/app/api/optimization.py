# backend/app/api/optimization.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi import Depends
from typing import List
from datetime import datetime
from app.models.trainset import OptimizationRequest, InductionDecision
from app.services.optimizer import TrainInductionOptimizer
from app.services.solver import RoleAssignmentSolver, SolverWeights
from app.services.rule_engine import DurableRulesEngine
from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.utils.cloud_database import cloud_db_manager
from app.config import settings
from app.security import require_api_key
import asyncio
import json
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/run", response_model=List[InductionDecision])
async def run_optimization(
    background_tasks: BackgroundTasks,
    request: OptimizationRequest,
    _auth=Depends(require_api_key),
):
    """Run AI/ML optimization with rule-based constraints (OR-Tools + Drools)"""
    try:
        # Get all trainsets from MongoDB Atlas
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data = []
        
        async for trainset_doc in cursor:
            trainset_doc.pop('_id', None)
            trainsets_data.append(trainset_doc)
        
        if not trainsets_data:
            raise HTTPException(status_code=404, detail="No trainsets found")
        
        # Rule-based constraint engine (Durable Rules) with safe fallback
        try:
            rule_engine = DurableRulesEngine()
            validated_trainsets = await rule_engine.apply_constraints(trainsets_data)
        except Exception as re_err:
            logger.warning(f"Rule engine unavailable, falling back to basic filter: {re_err}")
            # Basic filter fallback: require all certs VALID and no critical cards
            validated_trainsets = [
                t for t in trainsets_data
                if all(c.get("status") == "VALID" for c in t.get("fitness_certificates", {}).values())
                and t.get("job_cards", {}).get("critical_cards", 0) == 0
            ]
        
        # AI/ML Optimization (Google OR-Tools + PyTorch)
        optimizer = TrainInductionOptimizer()
        optimization_result = await optimizer.optimize(validated_trainsets, request)
        
        # Stabling Geometry Optimization (minimize shunting & turn-out time)
        stabling_optimizer = StablingGeometryOptimizer()
        stabling_geometry = await stabling_optimizer.optimize_stabling_geometry(
            validated_trainsets, [decision.dict() for decision in optimization_result]
        )
        
        # Skip Redis caching for now
        
        # Store optimization history in MongoDB
        background_tasks.add_task(store_optimization_history, request, optimization_result)
        
        # Write metrics to InfluxDB
        background_tasks.add_task(write_optimization_metrics, optimization_result)
        
        logger.info(f"Optimization completed: {len(optimization_result)} decisions")
        return optimization_result
    
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/constraints/check")
async def check_constraints(_auth=Depends(require_api_key)):
    """Real-time constraint validation using rule engine"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        
        violations = []
        valid_trainsets = 0
        
        # Rule-based validation with safe fallback per trainset
        try:
            rule_engine = DurableRulesEngine()
        except Exception as e:
            rule_engine = None

        async for trainset_doc in cursor:
            trainset_id = trainset_doc.get("trainset_id")
            try:
                if rule_engine:
                    constraint_violations = await rule_engine.check_constraints(trainset_doc)
                else:
                    # Fallback: quick checks
                    fc_ok = all(c.get("status") == "VALID" for c in trainset_doc.get("fitness_certificates", {}).values())
                    jc_ok = trainset_doc.get("job_cards", {}).get("critical_cards", 0) == 0
                    constraint_violations = [] if (fc_ok and jc_ok) else ["basic_constraints_failed"]
            except Exception as e:
                constraint_violations = [f"engine_error: {e}"]

            if constraint_violations:
                violations.append({
                    "trainset_id": trainset_id,
                    "violations": constraint_violations,
                    "severity": "CRITICAL" if any(("expired" in v) or ("critical" in v) for v in constraint_violations) else "WARNING"
                })
            else:
                valid_trainsets += 1
        
        return {
            "total_trainsets": valid_trainsets + len(violations),
            "valid_trainsets": valid_trainsets,
            "trainsets_with_violations": len(violations),
            "violations": violations,
            "checked_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Constraint check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Error checking constraints: {str(e)}")

@router.get("/simulate")
async def simulate_what_if(
    exclude_trainsets: str = "",
    force_induct: str = "",
    required_service_count: int = 14,
    w_readiness: float = 0.35,
    w_reliability: float = 0.30,
    w_branding: float = 0.20,
    w_shunt: float = 0.10,
    w_mileage_balance: float = 0.05,
    _auth=Depends(require_api_key),
):
    """What-if simulation with ML models"""
    try:
        # Parse parameters
        excluded = [t.strip() for t in exclude_trainsets.split(",") if t.strip()]
        forced = [t.strip() for t in force_induct.split(",") if t.strip()]
        
        # Get trainsets data
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data = []
        
        async for trainset_doc in cursor:
            trainset_doc.pop('_id', None)
            trainsets_data.append(trainset_doc)
        
        # Apply simulation constraints
        for trainset in trainsets_data:
            if trainset["trainset_id"] in excluded:
                trainset["simulation_constraint"] = "EXCLUDED"
            elif trainset["trainset_id"] in forced:
                trainset["simulation_constraint"] = "FORCED_INDUCT"
        
        # Build features for solver
        features = []
        for t in trainsets_data:
            features.append({
                "trainset_id": t["trainset_id"],
                "allowed_service": True,  # could call DurableRulesEngine here
                "must_ibl": False,
                "cleaning_available": True,
                "readiness": 1.0,  # plug-in ML readiness
                "reliability": t.get("sensor_health_score", 0.8),
                "branding": 1.0 if t.get("branding", {}).get("priority") == "HIGH" else 0.5,
                "shunt_cost_norm": 0.2,
                "km_30d_norm": 0.5,
            })

        weights = SolverWeights(
            readiness=w_readiness,
            reliability=w_reliability,
            branding=w_branding,
            shunt=w_shunt,
            mileage_balance=w_mileage_balance,
        )
        solver = RoleAssignmentSolver(required_service_count=required_service_count, weights=weights)
        solve_out = solver.solve(features)
        
        return {
            "scenario": {
                "excluded_trainsets": excluded,
                "forced_inductions": forced,
                "required_service_count": required_service_count,
                "weights": weights.__dict__,
            },
            "results": solve_out,
            "simulation_timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/latest", response_model=List[InductionDecision])
async def get_latest_ranked_list():
    """Get latest ranked induction list stored in MongoDB"""
    try:
        latest_collection = await cloud_db_manager.get_collection("latest_induction")
        doc = await latest_collection.find_one({})
        if not doc:
            raise HTTPException(status_code=404, detail="No latest induction list available")
        return [InductionDecision(**d) for d in doc.get("decisions", [])]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch latest list: {str(e)}")

@router.get("/stabling-geometry")
async def get_stabling_geometry_optimization():
    """Get optimized stabling geometry to minimize shunting and turn-out time"""
    try:
        # Directly compute optimization (Redis disabled)
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data = []
        
        async for trainset_doc in cursor:
            trainset_doc.pop('_id', None)
            trainsets_data.append(trainset_doc)
        
        if not trainsets_data:
            raise HTTPException(status_code=404, detail="No trainsets found")
        
        # Run stabling geometry optimization
        stabling_optimizer = StablingGeometryOptimizer()
        stabling_geometry = await stabling_optimizer.optimize_stabling_geometry(
            trainsets_data, []
        )
        
        return stabling_geometry
        
    except Exception as e:
        logger.error(f"Stabling geometry optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stabling geometry failed: {str(e)}")

@router.get("/shunting-schedule")
async def get_shunting_schedule():
    """Get detailed shunting schedule for operations team"""
    try:
        # Compute stabling geometry on the fly (Redis disabled)
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data = []
        async for trainset_doc in cursor:
            trainset_doc.pop('_id', None)
            trainsets_data.append(trainset_doc)
        if not trainsets_data:
            raise HTTPException(status_code=404, detail="No trainsets found")
        stabling_optimizer = StablingGeometryOptimizer()
        stabling_data = await stabling_optimizer.optimize_stabling_geometry(trainsets_data, [])
        
        # Generate shunting schedule
        shunting_schedule = await stabling_optimizer.get_shunting_schedule(stabling_data["optimized_layout"])
        
        return {
            "shunting_schedule": shunting_schedule,
            "total_operations": len(shunting_schedule),
            "estimated_total_time": sum(
                int(op["estimated_duration"].split()[0]) for op in shunting_schedule
            ),
            "crew_requirements": {
                "high_complexity": len([op for op in shunting_schedule if op["complexity"] == "HIGH"]),
                "medium_complexity": len([op for op in shunting_schedule if op["complexity"] == "MEDIUM"]),
                "low_complexity": len([op for op in shunting_schedule if op["complexity"] == "LOW"])
            }
        }
        
    except Exception as e:
        logger.error(f"Shunting schedule generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Shunting schedule failed: {str(e)}")

async def store_optimization_history(request: OptimizationRequest, result: List[InductionDecision]):
    """Store optimization results in MongoDB"""
    try:
        collection = await cloud_db_manager.get_collection("optimization_history")
        
        history_record = {
            "timestamp": datetime.now().isoformat(),
            "target_date": request.target_date.isoformat(),
            "required_service_hours": request.required_service_hours,
            "total_decisions": len(result),
            "inducted_count": sum(1 for d in result if d.decision == "INDUCT"),
            "standby_count": sum(1 for d in result if d.decision == "STANDBY"),
            "maintenance_count": sum(1 for d in result if d.decision == "MAINTENANCE"),
            "average_confidence": sum(d.confidence_score for d in result) / len(result) if result else 0,
            "decisions": [d.dict() for d in result]
        }
        
        insert_result = await collection.insert_one(history_record)
        # Also persist latest ranked list separately for quick access
        latest_collection = await cloud_db_manager.get_collection("latest_induction")
        await latest_collection.delete_many({})
        await latest_collection.insert_one({
            "_meta": {"updated_at": datetime.now().isoformat()},
            "decisions": [d.dict() for d in result]
        })
        logger.info("Optimization history stored in MongoDB")
        
    except Exception as e:
        logger.error(f"Error storing optimization history: {e}")

async def write_optimization_metrics(result: List[InductionDecision]):
    """Write metrics to InfluxDB for time-series analysis"""
    try:
        for decision in result:
            metric_data = {
                "trainset_id": decision.trainset_id,
                "sensor_type": "optimization_decision",
                "health_score": decision.confidence_score,
                "temperature": 0.0,
                "timestamp": datetime.now().isoformat()
            }
            await cloud_db_manager.write_sensor_data(metric_data)
        
        logger.info(f"Written {len(result)} optimization metrics to InfluxDB")
        
    except Exception as e:
        logger.error(f"Error writing optimization metrics: {e}")
