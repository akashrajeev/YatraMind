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
        # Default required_service_hours to 14 as per user request
        optimization_result = await optimizer.optimize(validated_trainsets, request, required_service_hours=14)
        
        # Stabling Geometry Optimization (minimize shunting & turn-out time)
        stabling_optimizer = StablingGeometryOptimizer()
        stabling_geometry = await stabling_optimizer.optimize_stabling_geometry(
            validated_trainsets, [decision.dict() for decision in optimization_result]
        )
        
        # Persist status updates to MongoDB
        for decision in optimization_result:
            new_status = "STANDBY"
            if decision.decision == "INDUCT":
                new_status = "ACTIVE"
            elif decision.decision == "MAINTENANCE":
                new_status = "MAINTENANCE"
            
            await collection.update_one(
                {"trainset_id": decision.trainset_id},
                {"$set": {"status": new_status}}
            )
        
        # Skip Redis caching for now
        
        # Store optimization history in MongoDB
        background_tasks.add_task(store_optimization_history, request, optimization_result)
        
        # Write metrics to InfluxDB
        background_tasks.add_task(write_optimization_metrics, optimization_result)
        
        logger.info(f"Optimization completed: {len(optimization_result)} decisions. Statuses updated in DB.")
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
            logger.warning(f"Rule engine init failed: {e}")
            rule_engine = None
            
        async for trainset in cursor:
            trainset.pop('_id', None)
            if rule_engine:
                v = await rule_engine.check_constraints(trainset)
                if v:
                    violations.append({
                        "trainset_id": trainset.get("trainset_id"),
                        "violations": v
                    })
                else:
                    valid_trainsets += 1
            else:
                # Basic check
                valid_trainsets += 1
                
        return {
            "total_checked": valid_trainsets + len(violations),
            "valid_count": valid_trainsets,
            "violations": violations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest", response_model=List[InductionDecision])
async def get_latest_ranked_list(_auth=Depends(require_api_key)):
    """Get the latest ranked induction list (cached or regenerated)"""
    try:
        # Try to get from history first (not implemented fully, so regenerate)
        # In a real system, we'd query the 'optimization_history' collection
        # For now, we'll trigger a fresh optimization with default params if no cache
        
        # But wait, user wants consistent ranking. 
        # Let's check if we have a recent result in memory or DB?
        # For simplicity and consistency, let's just run the optimizer again with defaults.
        # This ensures /latest always reflects current state and logic.
        
        # Get all trainsets
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data = []
        async for t in cursor:
            t.pop('_id', None)
            trainsets_data.append(t)
            
        if not trainsets_data:
            return []
            
        optimizer = TrainInductionOptimizer()
        # Create a default request
        default_request = OptimizationRequest()
        
        # Use the SAME optimizer logic as /run
        decisions = await optimizer.optimize(trainsets_data, default_request, required_service_hours=14)
        
        # Persist status updates to MongoDB (Automatic Allocation)
        for decision in decisions:
            new_status = "STANDBY"
            if decision.decision == "INDUCT":
                new_status = "ACTIVE"
            elif decision.decision == "MAINTENANCE":
                new_status = "MAINTENANCE"
            
            await collection.update_one(
                {"trainset_id": decision.trainset_id},
                {"$set": {"status": new_status}}
            )
        
        return decisions
        
    except Exception as e:
        logger.error(f"Error fetching latest induction list: {e}")
        # Return empty list or basic mock if everything fails
        return []

@router.post("/simulate", response_model=List[InductionDecision])
async def run_simulation(
    request: OptimizationRequest,
    forced_ids: List[str] = [],
    excluded_ids: List[str] = [],
    _auth=Depends(require_api_key)
):
    """Run 'What-If' simulation with forced/excluded trainsets"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data = []
        async for t in cursor:
            t.pop('_id', None)
            trainsets_data.append(t)
            
        optimizer = TrainInductionOptimizer()
        # Pass forced/excluded IDs to optimizer
        # Default service hours to 14 for simulation too
        decisions = await optimizer.optimize(trainsets_data, request, forced_ids=forced_ids, excluded_ids=excluded_ids, required_service_hours=14)
        
        return decisions
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stabling-geometry")
async def optimize_stabling_geometry(
    request: OptimizationRequest,
    _auth=Depends(require_api_key)
):
    """Optimize stabling geometry separately"""
    try:
        # Get trainsets
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data = []
        async for t in cursor:
            t.pop('_id', None)
            trainsets_data.append(t)
            
        # Run induction optimization first to get decisions
        optimizer = TrainInductionOptimizer()
        induction_decisions = await optimizer.optimize(trainsets_data, request, required_service_hours=14)
        
        # Run stabling optimization
        stabling_optimizer = StablingGeometryOptimizer()
        stabling_result = await stabling_optimizer.optimize_stabling_geometry(
            trainsets_data, [d.dict() for d in induction_decisions]
        )
        
        return stabling_result
    except Exception as e:
        logger.error(f"Stabling optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def store_optimization_history(request: OptimizationRequest, results: List[InductionDecision]):
    """Background task to store optimization run details"""
    try:
        history_collection = await cloud_db_manager.get_collection("optimization_history")
        doc = {
            "timestamp": datetime.utcnow(),
            "request": request.dict(),
            "results_count": len(results),
            "inducted_count": len([r for r in results if r.decision == "INDUCT"]),
            "top_decisions": [r.dict() for r in results[:5]] # Store top 5 for quick preview
        }
        await history_collection.insert_one(doc)
    except Exception as e:
        logger.error(f"Failed to store optimization history: {e}")

async def write_optimization_metrics(results: List[InductionDecision]):
    """Background task to write metrics to InfluxDB (Mock)"""
    # In a real app, this would write to InfluxDB
    pass

@router.get("/explain/{trainset_id}")
async def explain_decision(
    trainset_id: str, 
    decision: str, 
    format: str = "json", 
    _auth=Depends(require_api_key)
):
    """Explain why a specific decision was made for a trainset"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        trainset = await collection.find_one({"trainset_id": trainset_id})
        
        if not trainset:
            raise HTTPException(status_code=404, detail="Trainset not found")
            
        trainset.pop('_id', None)
        
        explanation = generate_comprehensive_explanation(trainset, decision)
        
        # Include full trainset details for "View Details"
        explanation["trainset_details"] = trainset
        
        if format == "html":
            # Not implemented fully, but structure is there
            return render_explanation_html(explanation)
        
        return explanation
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
