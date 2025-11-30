# backend/app/api/optimization.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi import Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import random
from app.models.trainset import OptimizationRequest, InductionDecision, OptimizationWeights
from app.services.optimizer import TrainInductionOptimizer
# Removed RoleAssignmentSolver import
from app.services.rule_engine import DurableRulesEngine
from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.utils.cloud_database import cloud_db_manager
from app.utils.explainability import (
    generate_comprehensive_explanation,
    render_explanation_html,
    render_explanation_text
)
from app.config import settings
from app.security import require_api_key, require_role
from app.models.user import UserRole, User
from pydantic import BaseModel, Field
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


@router.get("/explain/{trainset_id}")
async def explain_assignment(
    trainset_id: str,
    decision: str = "INDUCT",
    format: str = "json",
    _auth=Depends(require_api_key)
):
    """Generate comprehensive explanation for a specific trainset assignment"""
    try:
        # Get trainset data
        collection = await cloud_db_manager.get_collection("trainsets")
        trainset_doc = await collection.find_one({"trainset_id": trainset_id})
        
        if not trainset_doc:
            raise HTTPException(status_code=404, detail=f"Trainset {trainset_id} not found")
        
        trainset_doc.pop('_id', None)
        
        # Generate comprehensive explanation
        explanation = generate_comprehensive_explanation(trainset_doc, decision)
        
        # Add metadata
        explanation.update({
            "trainset_id": trainset_id,
            "role": decision,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        if format == "html":
            html_content = render_explanation_html(explanation)
            return {"html": html_content, "data": explanation}
        elif format == "text":
            text_content = render_explanation_text(explanation)
            return {"text": text_content, "data": explanation}
        else:
            return explanation
            
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")


@router.post("/explain/batch")
async def explain_batch_assignments(
    assignments: List[Dict[str, Any]],
    format: str = "json",
    _auth=Depends(require_api_key)
):
    """Generate explanations for multiple trainset assignments"""
    try:
        explanations = []
        
        for assignment in assignments:
            trainset_id = assignment.get("trainset_id")
            decision = assignment.get("decision", "INDUCT")
            
            if not trainset_id:
                continue
                
            # Get trainset data
            collection = await cloud_db_manager.get_collection("trainsets")
            trainset_doc = await collection.find_one({"trainset_id": trainset_id})
            
            if not trainset_doc:
                explanations.append({
                    "trainset_id": trainset_id,
                    "error": "Trainset not found"
                })
                continue
            
            trainset_doc.pop('_id', None)
            
            # Generate comprehensive explanation
            explanation = generate_comprehensive_explanation(trainset_doc, decision)
            explanation.update({
                "trainset_id": trainset_id,
                "role": decision,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            explanations.append(explanation)
        
        if format == "html":
            html_explanations = []
            for explanation in explanations:
                if "error" not in explanation:
                    html_content = render_explanation_html(explanation)
                    html_explanations.append({
                        "trainset_id": explanation["trainset_id"],
                        "html": html_content
                    })
            return {"explanations": html_explanations, "data": explanations}
        elif format == "text":
            text_explanations = []
            for explanation in explanations:
                if "error" not in explanation:
                    text_content = render_explanation_text(explanation)
                    text_explanations.append({
                        "trainset_id": explanation["trainset_id"],
                        "text": text_content
                    })
            return {"explanations": text_explanations, "data": explanations}
        else:
            return {"explanations": explanations}
            
    except Exception as e:
        logger.error(f"Error generating batch explanations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating batch explanations: {str(e)}")


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
        
        # Create OptimizationRequest with custom weights
        weights = OptimizationWeights(
            readiness=w_readiness,
            reliability=w_reliability,
            branding=w_branding,
            shunt=w_shunt,
            mileage_balance=w_mileage_balance
        )
        
        request = OptimizationRequest(
            required_service_hours=required_service_count,
            weights=weights
        )
        
        # Run optimization with simulation constraints
        optimizer = TrainInductionOptimizer()
        decisions = await optimizer.optimize(trainsets_data, request, forced_ids=forced, excluded_ids=excluded)
        
        # Map decisions to simulation result format
        assignments = []
        for d in decisions:
            role = "service" if d.decision == "INDUCT" else d.decision.lower()
            assignments.append({
                "trainset_id": d.trainset_id,
                "role": role,
                "objective_contrib": d.score,
                "reasons": d.reasons
            })
            
        solve_out = {
            "status": "ok" if any(d.decision == "INDUCT" for d in decisions) else "infeasible",
            "assignments": assignments,
            "objective": sum(d.score for d in decisions if d.decision == "INDUCT")
        }
        
        return {
            "scenario": {
                "excluded_trainsets": excluded,
                "forced_inductions": forced,
                "required_service_count": required_service_count,
                "weights": weights.dict(),
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
        # First, check if there's a manually adjusted list stored
        latest_collection = await cloud_db_manager.get_collection("latest_induction")
        
        # First priority: Get manually adjusted list if it exists
        manually_adjusted_doc = await latest_collection.find_one(
            {"_meta.manually_adjusted": True},
            sort=[("_meta.updated_at", -1)]
        )
        
        if manually_adjusted_doc and "decisions" in manually_adjusted_doc and len(manually_adjusted_doc["decisions"]) > 0:
            logger.info("Returning manually adjusted ranked list")
            return [InductionDecision(**d) for d in manually_adjusted_doc["decisions"]]
        
        # Second priority: Get most recent list (if recent enough)
        latest_doc = await latest_collection.find_one(sort=[("created_at", -1)])
        
        if latest_doc and "decisions" in latest_doc and len(latest_doc["decisions"]) > 0:
            # Check if it's recent (within last hour) - if so, return cached version
            created_at_str = latest_doc.get("created_at") or latest_doc.get("_meta", {}).get("updated_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                    time_diff = datetime.utcnow() - created_at.replace(tzinfo=None)
                    if time_diff.total_seconds() < 3600:  # Less than 1 hour old
                        logger.info("Returning cached ranked list")
                        return [InductionDecision(**d) for d in latest_doc["decisions"]]
                except Exception:
                    pass  # If date parsing fails, regenerate
        
        # If no stored list or it's too old, regenerate
        logger.info("Regenerating ranked list")
        # Get all trainsets from database and create ranked list
        trainsets_collection = await cloud_db_manager.get_collection("trainsets")
        trainsets = []
        async for trainset_doc in trainsets_collection.find({}):
            trainsets.append(trainset_doc)
        
        logger.info(f"Found {len(trainsets)} trainsets in database")
        
        if not trainsets:
                logger.info("No trainsets found, creating sample data")
                # Create sample trainsets if none exist - using proper KMRL trainset IDs
                sample_trainsets = [
                    {
                        "trainset_id": f"T-{i:03d}", 
                        "status": "ACTIVE", 
                        "sensor_health_score": round(random.uniform(0.7, 0.95), 2),
                        "predicted_failure_risk": round(random.uniform(0.05, 0.3), 2),
                        "total_operational_hours": random.randint(100, 2000),
                        "branding": {
                            "priority": random.choice(["HIGH", "MEDIUM", "LOW"]),
                            "runtime_requirements": [random.randint(8, 16)]
                        },
                        "certificates": {
                            "rolling_stock": {
                                "status": random.choice(["valid", "expired", "pending"]),
                                "expiry_days": random.randint(-30, 365)
                            },
                            "signalling": {
                                "status": random.choice(["valid", "expired", "pending"]),
                                "expiry_days": random.randint(-30, 365)
                            },
                            "telecom": {
                                "status": random.choice(["valid", "expired", "pending"]),
                                "expiry_days": random.randint(-30, 365)
                            }
                        },
                        "job_cards": [
                            {
                                "id": f"JC-{i}-{j}",
                                "status": random.choice(["open", "closed", "pending"]),
                                "priority": random.choice(["critical", "high", "medium", "low"]),
                                "description": f"Maintenance task {j}"
                            } for j in range(random.randint(0, 3))
                        ],
                        "cleaning": {
                            "bay_available": random.choice([True, False]),
                            "manpower_available": random.choice([True, False])
                        },
                        "stabling": {
                            "bay_position": random.choice(["near_exit", "middle", "far"])
                        },
                        "maintenance": {
                            "scheduled_dates": [f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"],
                            "types": random.sample(["preventive", "corrective", "emergency"], random.randint(1, 2))
                        }
                    }
                    for i in range(1, 31)  # Create 30 trainsets
                ]
                await trainsets_collection.insert_many(sample_trainsets)
                trainsets = sample_trainsets
                logger.info(f"Created {len(trainsets)} sample trainsets")
        
        # Create ranked decisions using the real optimizer
        try:
            # Use default request parameters
            request = OptimizationRequest(required_service_hours=14)
            
            # Use the real optimizer to ensure consistency
            optimizer = TrainInductionOptimizer()
            
            # Rule-based constraint engine (Durable Rules) with safe fallback
            try:
                rule_engine = DurableRulesEngine()
                validated_trainsets = await rule_engine.apply_constraints(trainsets)
            except Exception as re_err:
                logger.warning(f"Rule engine unavailable, falling back to basic filter: {re_err}")
                validated_trainsets = [
                    t for t in trainsets
                    if all(c.get("status") == "VALID" for c in t.get("fitness_certificates", {}).values())
                    and t.get("job_cards", {}).get("critical_cards", 0) == 0
                ]
            
            # Run optimization
            decisions = await optimizer.optimize(validated_trainsets, request)
            
            logger.info(f"Generated {len(decisions)} ranked decisions using optimizer")
            
            # Store the ranked list
            await latest_collection.insert_one({
                "_meta": {
                    "updated_at": datetime.utcnow().isoformat(),
                    "manually_adjusted": False
                },
                "decisions": [d.dict() for d in decisions],
                "created_at": datetime.utcnow().isoformat(),
                "total_trainsets": len(decisions)
            })
            
            return decisions
            
        except Exception as opt_err:
            logger.error(f"Optimizer failed in latest list generation: {opt_err}")
            # Fallback to simple mock data if optimizer fails completely
            mock_decisions = []
            for i, t in enumerate(trainsets):
                mock_decisions.append(InductionDecision(
                    trainset_id=t.get("trainset_id", f"T-{i}"),
                    decision="STANDBY",
                    confidence_score=0.5,
                    reasons=["Fallback due to optimization error"],
                    score=0.5,
                    top_reasons=[],
                    top_risks=["Optimization service unavailable"],
                    violations=[],
                    shap_values=[]
                ))
            return mock_decisions
            
    except Exception as e:
        logger.error(f"Error fetching latest induction list: {e}")
        return []

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


class RankedListReorderRequest(BaseModel):
    """Request model for reordering ranked induction list"""
    trainset_ids: List[str] = Field(..., description="Ordered list of trainset IDs in new order")
    reason: Optional[str] = Field(None, description="Reason for manual reordering")


@router.post("/latest/reorder", response_model=List[InductionDecision])
async def reorder_ranked_list(
    reorder_request: RankedListReorderRequest,
    current_user: User = Depends(require_role(UserRole.OPERATIONS_MANAGER.value)),
    _auth=Depends(require_api_key),
):
    """Manually reorder the ranked induction list (Admin only)"""
    try:
        # Get current ranked list
        latest_collection = await cloud_db_manager.get_collection("latest_induction")
        latest_doc = await latest_collection.find_one(sort=[("_meta.updated_at", -1), ("created_at", -1)])
        
        if not latest_doc or "decisions" not in latest_doc:
            raise HTTPException(status_code=404, detail="No active ranked list found to reorder")
        
        current_decisions = {d["trainset_id"]: d for d in latest_doc["decisions"]}
        
        # Validate all requested IDs exist
        new_order_decisions = []
        for tid in reorder_request.trainset_ids:
            if tid in current_decisions:
                new_order_decisions.append(current_decisions[tid])
            else:
                logger.warning(f"Reorder request contains unknown trainset ID: {tid}")
        
        # Append any missing trainsets (safety fallback)
        for tid, decision in current_decisions.items():
            if tid not in reorder_request.trainset_ids:
                new_order_decisions.append(decision)
        
        # Update metadata
        new_doc = {
            "_meta": {
                "updated_at": datetime.utcnow().isoformat(),
                "manually_adjusted": True,
                "adjusted_by": current_user.username,
                "reason": reorder_request.reason
            },
            "decisions": new_order_decisions,
            "created_at": latest_doc.get("created_at", datetime.utcnow().isoformat()),
            "total_trainsets": len(new_order_decisions)
        }
        
        await latest_collection.insert_one(new_doc)
        
        logger.info(f"Ranked list manually reordered by {current_user.username}")
        return [InductionDecision(**d) for d in new_order_decisions]
        
    except Exception as e:
        logger.error(f"Error reordering ranked list: {e}")
        raise HTTPException(status_code=500, detail=f"Error reordering list: {str(e)}")
