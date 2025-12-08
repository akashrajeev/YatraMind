# backend/app/api/optimization.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
import json
import logging
import uuid
from pathlib import Path
from math import ceil

from app.models.trainset import OptimizationRequest, InductionDecision, StablingGeometryResponse
from app.services.optimizer import TrainInductionOptimizer
from app.services.solver import RoleAssignmentSolver, SolverWeights
from app.services.rule_engine import DurableRulesEngine
from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.services.optimization_store import get_latest_decisions, get_decisions_from_history
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

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/run")
async def run_optimization(
    background_tasks: BackgroundTasks,
    request: OptimizationRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    """Run AI/ML optimization with rule-based constraints (OR-Tools + Drools)"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data: List[Dict[str, Any]] = []

        async for trainset_doc in cursor:
            trainset_doc.pop("_id", None)
            trainsets_data.append(trainset_doc)

        if not trainsets_data:
            raise HTTPException(status_code=404, detail="No trainsets found")

        optimizer = TrainInductionOptimizer()
        optimization_result, fleet_req = await optimizer.optimize(trainsets_data, request)

        granted_train_count = sum(1 for d in optimization_result if d.decision == "INDUCT")
        eligible_train_count = len(optimization_result)

        stabling_optimizer = StablingGeometryOptimizer()
        stabling_geometry = await stabling_optimizer.optimize_stabling_geometry(
            trainsets_data, [decision.dict() for decision in optimization_result]
        )

        efficiency_metrics = stabling_geometry.get("efficiency_metrics", {})
        overall_efficiency = efficiency_metrics.get("overall_efficiency")
        if overall_efficiency is not None:
            stabling_geometry["efficiency_improvement"] = round(float(overall_efficiency) * 100, 2)
        else:
            stabling_geometry["efficiency_improvement"] = 0.0

        optimized_layout = stabling_geometry.get("optimized_layout", {})
        total_positions = sum(
            len(depot_data.get("bay_assignments", {})) for depot_data in optimized_layout.values()
        )
        stabling_geometry["total_optimized_positions"] = total_positions

        # Persist decisions immediately so downstream stabling/shunting calls have fresh data
        await store_optimization_history(request, optimization_result, fleet_req)
        background_tasks.add_task(write_optimization_metrics, optimization_result)

        # Diagnostics block exposed to both API response and snapshot
        diagnostics: Dict[str, Any] = {
            "required_service_trains": fleet_req.required_service_trains,
            "standby_buffer": fleet_req.standby_buffer,
            "calculation_method": fleet_req.calculation_method,
            "eligible_train_count": eligible_train_count,
            "granted_train_count": granted_train_count,
            "fleet_requirement": fleet_req.dict(),
        }

        try:
            sim_dir = Path(cfg.get("SIMULATION_SAVE_DIR", "backend/simulation_runs"))
            sim_dir.mkdir(parents=True, exist_ok=True)
            snapshot_payload = {
                "optimization_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "diagnostics": diagnostics,
                "decisions": [d.dict() for d in optimization_result],
            }
            out_path = sim_dir / f"optimization_{snapshot_payload['optimization_id']}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(snapshot_payload, f, indent=2, default=str)
            logger.info("Saved optimization snapshot to %s", out_path)
        except Exception as e:
            logger.warning("Failed to write optimization snapshot: %s", e)

        note_parts: List[str] = []
        
        if fleet_req.calculation_method == "timetable":
             note_parts.append(
                f"Requirement derived from timetable: {fleet_req.required_service_trains} trains "
                f"(+ {fleet_req.standby_buffer} standby)."
            )

        if fleet_req.required_service_trains and granted_train_count < fleet_req.required_service_trains:
            note_parts.append(
                f"Optimization granted {granted_train_count} trains, "
                f"which is fewer than the required {fleet_req.required_service_trains}."
            )

        note = " ".join(note_parts) if note_parts else None

        logger.info(
            "Optimization diagnostics: method=%s eligible=%d granted=%d required=%d",
            fleet_req.calculation_method,
            eligible_train_count,
            granted_train_count,
            fleet_req.required_service_trains
        )

        response: Dict[str, Any] = {
            "required_service_trains": fleet_req.required_service_trains,
            "standby_buffer": fleet_req.standby_buffer,
            "total_required_trains": fleet_req.total_required_trains,
            "calculation_method": fleet_req.calculation_method,
            "eligible_train_count": eligible_train_count,
            "granted_train_count": granted_train_count,
            "actual_induct_count": granted_train_count,
            "service_shortfall": max(0, fleet_req.required_service_trains - granted_train_count),
            "note": note,
            "diagnostics": diagnostics,
            "decisions": [d.dict() for d in optimization_result],
            "stabling_geometry": stabling_geometry,
        }

        return response

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/constraints/check")
async def check_constraints(current_user: User = Depends(require_role(UserRole.ADMIN))):
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
    current_user: User = Depends(require_role(UserRole.ADMIN))
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
    current_user: User = Depends(require_role(UserRole.ADMIN))
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
    current_user: User = Depends(require_role(UserRole.ADMIN)),
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
        
        # If no stored list or it's too old, return error (deterministic behavior)
        # Do NOT generate random mock data - require explicit optimization run
        logger.warning("No recent optimization found. An optimization must be run first via /api/optimization/run")
        raise HTTPException(
            status_code=404,
            detail={
                "error": "no_optimization_found",
                "message": "No optimization has been run yet. Please run an optimization via POST /api/optimization/run first.",
                "hint": "The /latest endpoint requires a stored optimization result. Run an optimization to generate ranked decisions."
            }
        )
        
        # Create ranked decisions for all trainsets with proper scoring
        mock_decisions = []
        
        for trainset in trainsets:
            # Calculate score based on KMRL priority factors (in order of importance)
            score = 0.0
            reasons = []
            risks = []
            
            try:
                # 1. FITNESS CERTIFICATES (Highest Priority - 35% weight)
                # Check Rolling-Stock, Signalling, and Telecom certificates
                fitness_score = 0.0
                cert_status = trainset.get("fitness_certificates", {})
                
                # Rolling-Stock certificate (most critical)
                rolling_stock = cert_status.get("rolling_stock", {})
                if rolling_stock.get("status") == "VALID":
                    fitness_score += 0.15
                    reasons.append("Valid Rolling-Stock fitness certificate")
                elif rolling_stock.get("status") == "EXPIRED":
                    fitness_score += 0.0
                    risks.append("Rolling-Stock certificate expired")
                else:
                    fitness_score += 0.05
                    risks.append("Rolling-Stock certificate pending")
                
                # Signalling certificate
                signalling = cert_status.get("signalling", {})
                if signalling.get("status") == "VALID":
                    fitness_score += 0.10
                    reasons.append("Valid Signalling fitness certificate")
                elif signalling.get("status") == "EXPIRED":
                    fitness_score += 0.0
                    risks.append("Signalling certificate expired")
                else:
                    fitness_score += 0.05
                    risks.append("Signalling certificate pending")
                
                # Telecom certificate
                telecom = cert_status.get("telecom", {})
                if telecom.get("status") == "VALID":
                    fitness_score += 0.10
                    reasons.append("Valid Telecom fitness certificate")
                elif telecom.get("status") == "EXPIRED":
                    fitness_score += 0.0
                    risks.append("Telecom certificate expired")
                else:
                    fitness_score += 0.05
                    risks.append("Telecom certificate pending")
                
                score += fitness_score
                
                # 2. JOB-CARD STATUS (25% weight)
                # Check for open/pending work orders in Maximo
                job_cards = trainset.get("job_cards", {})
                open_cards = job_cards.get("open_cards", 0)
                critical_cards = job_cards.get("critical_cards", 0)
                
                if critical_cards > 0:
                    score += 0.0
                    risks.append(f"{critical_cards} critical job cards open")
                elif open_cards > 0:
                    score += 0.05
                    risks.append(f"{open_cards} job cards open")
                else:
                    score += 0.25
                    reasons.append("No critical job cards pending")
                
                # 3. CLEANING & DETAILING SLOTS (20% weight)
                # Check availability of manpower and bay occupancy
                # For now, use sensor health as proxy for cleaning readiness
                sensor_health = trainset.get("sensor_health_score", 0.8)
                if sensor_health > 0.9:
                    score += 0.20
                    reasons.append("Excellent sensor health - ready for service")
                elif sensor_health > 0.7:
                    score += 0.10
                    reasons.append("Good sensor health")
                else:
                    score += 0.0
                    risks.append("Poor sensor health - needs attention")
                
                # 4. STABLING GEOMETRY (15% weight)
                # Physical positions of bays affect dispatch efficiency
                # Use current location as proxy for stabling position
                current_location = trainset.get("current_location", {})
                depot = current_location.get("depot", "unknown")
                if depot in ["Petta", "Vytilla"]:  # Main depots
                    score += 0.15
                    reasons.append("Optimal depot location for dispatch")
                else:
                    score += 0.10
                    reasons.append("Good depot location")
                
                # 5. MILEAGE BALANCING (10% weight)
                # Balance wear on bogies, brake pads, and HVAC systems
                current_mileage = trainset.get("current_mileage", 0)
                max_mileage = trainset.get("max_mileage_before_maintenance", 50000)
                mileage_ratio = current_mileage / max_mileage if max_mileage > 0 else 0
                
                if mileage_ratio < 0.5:
                    score += 0.10
                    reasons.append("Low mileage - good for balancing wear")
                elif mileage_ratio > 0.8:
                    score += 0.05
                    risks.append("High mileage - consider maintenance")
                else:
                    score += 0.08
                    reasons.append("Balanced mileage")
                
                # 6. BRANDING PRIORITIES (5% weight - Lowest Priority)
                # Contractual commitments for exterior wrap exposure hours
                branding_priority = trainset.get("branding_priority", 0)
                if branding_priority > 0.7:
                    score += 0.05
                    reasons.append("High branding priority")
                elif branding_priority > 0.3:
                    score += 0.03
                    reasons.append("Medium branding priority")
                else:
                    score += 0.02
                    reasons.append("Low branding priority")
                
                # Normalize score to 0-1 range
                score = min(1.0, max(0.0, score))
                
            except Exception as e:
                logger.error(f"Error calculating score for trainset {trainset.get('trainset_id', 'unknown')}: {e}")
                # Fallback to simple scoring
                score = 0.5
                reasons = ["Default scoring applied"]
                risks = ["Scoring error occurred"]
            
            # Determine decision based on KMRL priority factors
            # INDUCT: All critical factors (fitness, job cards) are good
            # STANDBY: Some issues but not critical
            # MAINTENANCE: Critical issues that need attention
            
            if score >= 0.8 and len([r for r in risks if "certificate" in r.lower() or "critical" in r.lower()]) == 0:
                decision = "INDUCT"
            elif score >= 0.6 and len([r for r in risks if "certificate" in r.lower()]) == 0:
                decision = "STANDBY"
            else:
                decision = "MAINTENANCE"
            
            # Calculate confidence based on score and risk factors
            base_confidence = min(0.95, max(0.6, score))
            risk_penalty = len(risks) * 0.05
            confidence = round(max(0.6, base_confidence - risk_penalty), 2)
            final_score = round(score, 3)
            
            mock_decision = {
                "trainset_id": trainset["trainset_id"],
                "decision": decision,
                "confidence_score": confidence,
                "score": final_score,
                "top_reasons": reasons[:3] if reasons else ["All systems operational"],
                "top_risks": [risk for risk in risks[:2] if risk is not None] if risks else [],
                "violations": [],
                "shap_values": [
                    {"name": "Fitness Certificates", "value": fitness_score, "impact": "positive"},
                    {"name": "Job Card Status", "value": 0.25 if critical_cards == 0 and open_cards == 0 else 0.05, "impact": "positive"},
                    {"name": "Sensor Health", "value": trainset.get("sensor_health_score", 0.85), "impact": "positive"},
                    {"name": "Mileage Balance", "value": 1.0 - mileage_ratio, "impact": "positive"},
                    {"name": "Branding Priority", "value": branding_priority, "impact": "positive"}
                ],
                "reasons": reasons if reasons else ["Default scoring applied"]
            }
            mock_decisions.append(mock_decision)
        
        # Sort by score (highest first) - ensure proper ranking
        mock_decisions.sort(key=lambda x: x["score"], reverse=True)
        
        # Take top 10 trainsets for induction list (not just 3)
        top_decisions = mock_decisions[:10]
        
        logger.info(f"Created {len(mock_decisions)} ranked decisions, returning top {len(top_decisions)}")
        
        # Store the ranked list (only if not manually adjusted)
        # Don't overwrite manually adjusted lists
        existing_doc = await latest_collection.find_one(sort=[("_meta.updated_at", -1), ("created_at", -1)])
        if not existing_doc or not existing_doc.get("_meta", {}).get("manually_adjusted", False):
            await latest_collection.insert_one({
                "_meta": {
                    "updated_at": datetime.utcnow().isoformat(),
                    "manually_adjusted": False
                },
                "decisions": top_decisions,
                "created_at": datetime.utcnow().isoformat(),
                "total_trainsets": len(mock_decisions)
            })
        
        logger.info(f"Stored ranked list with {len(top_decisions)} decisions")
        return top_decisions
        
    except HTTPException:
        # Re-raise HTTP exceptions (like our 404 for no optimization)
        raise
    except Exception as e:
        logger.error(f"Error fetching latest induction list: {e}", exc_info=True)
        # Return deterministic error instead of random mock data
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": f"Failed to fetch latest induction list: {str(e)}",
                "hint": "Please ensure an optimization has been run and try again."
            }
        )

@router.get("/stabling-geometry")
async def get_stabling_geometry_optimization():
    """Get optimized stabling geometry with rich, structured intelligence"""
    try:
        # Load current trainsets
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data: List[Dict[str, Any]] = []
        
        async for trainset_doc in cursor:
            trainset_doc.pop("_id", None)
            trainsets_data.append(trainset_doc)
        
        if not trainsets_data:
            raise HTTPException(status_code=404, detail="No trainsets found")

        # Retrieve latest optimization decisions (DB first, then history)
        decisions: Optional[List[Dict[str, Any]]] = await get_latest_decisions()
        if not decisions:
            decisions = await get_decisions_from_history()

        if not decisions:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No optimization decisions available. Run optimization first.",
                    "code": "no_induction_decisions",
                },
            )

        # Try to get fleet requirements from latest optimization result
        fleet_req = None
        try:
            latest_collection = await cloud_db_manager.get_collection("latest_induction")
            latest_doc = await latest_collection.find_one(sort=[("_meta.updated_at", -1), ("created_at", -1)])
            if latest_doc and "fleet_requirement" in latest_doc:
                fleet_req = latest_doc["fleet_requirement"]
            else:
                # Try optimization history
                history_collection = await cloud_db_manager.get_collection("optimization_history")
                history_doc = await history_collection.find_one(sort=[("timestamp", -1)])
                if history_doc and "fleet_requirement" in history_doc:
                    fleet_req = history_doc["fleet_requirement"]
        except Exception as e:
            logger.warning(f"Could not retrieve fleet requirements: {e}")

        stabling_optimizer = StablingGeometryOptimizer()
        geometry = await stabling_optimizer.optimize_stabling_geometry(
            trainsets_data, decisions, fleet_req
        )

        return geometry
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stabling geometry optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Stabling geometry failed: {str(e)}")

@router.get("/shunting-schedule")
async def get_shunting_schedule():
    """Get detailed shunting schedule with ordered moves and operational intelligence"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find({})
        trainsets_data: List[Dict[str, Any]] = []

        async for trainset_doc in cursor:
            trainset_doc.pop("_id", None)
            trainsets_data.append(trainset_doc)

        if not trainsets_data:
            raise HTTPException(status_code=404, detail="No trainsets found")

        decisions: Optional[List[Dict[str, Any]]] = await get_latest_decisions()
        if not decisions:
            decisions = await get_decisions_from_history()

        if not decisions:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "No optimization decisions available. Run optimization first.",
                    "code": "no_induction_decisions",
                },
            )

        stabling_optimizer = StablingGeometryOptimizer()
        stabling_data = await stabling_optimizer.optimize_stabling_geometry(trainsets_data, decisions)

        operations = stabling_data.get("shunting_operations", [])
        summary = stabling_data.get("shunting_summary", {})

        return {
            "shunting_schedule": operations,
            "schedule_by_depot": {"Muttom Depot": operations},
            "depot_summaries": {"Muttom Depot": summary},
            "total_operations": summary.get("total_operations", 0),
            "estimated_total_time": summary.get("total_time_min", 0),
            "crew_requirements": {
                "high_complexity": len([op for op in operations if op.get("complexity") == "HIGH"]),
                "medium_complexity": len([op for op in operations if op.get("complexity") == "MEDIUM"]),
                "low_complexity": len([op for op in operations if op.get("complexity") == "LOW"]),
            },
            "operational_window": {
                "start_time": stabling_optimizer.operational_window["start"],
                "end_time": stabling_optimizer.operational_window["end"],
                "buffer_minutes": summary.get("buffer_minutes", 0),
            },
            "optimization_timestamp": datetime.now().isoformat(),
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Shunting schedule generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Shunting schedule failed: {str(e)}")

async def store_optimization_history(request: OptimizationRequest, result: List[InductionDecision], fleet_req: Optional[Any] = None):
    """Store optimization results in MongoDB"""
    try:
        collection = await cloud_db_manager.get_collection("optimization_history")
        
        history_record = {
            "timestamp": datetime.now().isoformat(),
            "target_date": request.target_date.isoformat(),
            "required_service_count": request.required_service_count,
            "service_date": request.service_date,
            "total_decisions": len(result),
            "inducted_count": sum(1 for d in result if d.decision == "INDUCT"),
            "standby_count": sum(1 for d in result if d.decision == "STANDBY"),
            "maintenance_count": sum(1 for d in result if d.decision == "MAINTENANCE"),
            "average_confidence": sum(d.confidence_score for d in result) / len(result) if result else 0,
            "decisions": [d.dict() for d in result]
        }
        
        # Store fleet requirement if provided
        if fleet_req:
            history_record["fleet_requirement"] = fleet_req.dict() if hasattr(fleet_req, "dict") else fleet_req
        
        insert_result = await collection.insert_one(history_record)
        # Also persist latest ranked list separately for quick access
        latest_collection = await cloud_db_manager.get_collection("latest_induction")
        await latest_collection.delete_many({})
        latest_doc = {
            "_meta": {"updated_at": datetime.now().isoformat()},
            "decisions": [d.dict() for d in result]
        }
        # Store fleet requirement in latest as well
        if fleet_req:
            latest_doc["fleet_requirement"] = fleet_req.dict() if hasattr(fleet_req, "dict") else fleet_req
        await latest_collection.insert_one(latest_doc)
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
        # Try to get the most recent document, checking both _meta.updated_at and created_at
        latest_doc = await latest_collection.find_one(sort=[("_meta.updated_at", -1), ("created_at", -1)])
        
        if not latest_doc or "decisions" not in latest_doc:
            raise HTTPException(status_code=404, detail="No ranked list found. Please run optimization first.")
        
        current_decisions = latest_doc["decisions"]
        
        # Create a map of trainset_id to decision for quick lookup
        decision_map = {d["trainset_id"]: d for d in current_decisions}
        
        # Reorder decisions based on provided order
        reordered_decisions = []
        for trainset_id in reorder_request.trainset_ids:
            if trainset_id in decision_map:
                decision = decision_map[trainset_id].copy()
                # Mark as manually adjusted
                decision["manually_adjusted"] = True
                decision["adjusted_by"] = current_user.username
                decision["adjusted_at"] = datetime.utcnow().isoformat()
                decision["adjustment_reason"] = reorder_request.reason
                reordered_decisions.append(decision)
            else:
                logger.warning(f"Trainset {trainset_id} not found in current ranked list")
        
        # Add any remaining decisions that weren't in the reorder request (append to end)
        for decision in current_decisions:
            if decision["trainset_id"] not in reorder_request.trainset_ids:
                reordered_decisions.append(decision)
        
        # Update the ranked list in database
        # Delete old documents and insert the new one with manual adjustment flag
        await latest_collection.delete_many({})
        await latest_collection.insert_one({
            "_meta": {
                "updated_at": datetime.utcnow().isoformat(),
                "manually_adjusted": True,
                "adjusted_by": current_user.username,
                "adjustment_reason": reorder_request.reason
            },
            "decisions": reordered_decisions,
            "created_at": datetime.utcnow().isoformat(),
            "total_trainsets": len(reordered_decisions)
        })
        
        logger.info(f"Ranked list manually reordered by {current_user.username} with {len(reordered_decisions)} decisions")
        
        # Return as InductionDecision objects
        return [InductionDecision(**d) for d in reordered_decisions]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reordering ranked list: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reorder ranked list: {str(e)}")
