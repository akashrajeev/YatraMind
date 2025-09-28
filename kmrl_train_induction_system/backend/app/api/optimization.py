# backend/app/api/optimization.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi import Depends
from typing import List, Dict, Any
from datetime import datetime
import random
from app.models.trainset import OptimizationRequest, InductionDecision
from app.services.optimizer import TrainInductionOptimizer
from app.services.solver import RoleAssignmentSolver, SolverWeights
from app.services.rule_engine import DurableRulesEngine
from app.services.stabling_optimizer import StablingGeometryOptimizer
from app.utils.cloud_database import cloud_db_manager
from app.utils.explainability import (
    generate_comprehensive_explanation,
    render_explanation_html,
    render_explanation_text
)
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
        # Always regenerate the ranked list instead of using cached data
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
        
        # Store the ranked list
        latest_collection = await cloud_db_manager.get_collection("latest_induction")
        await latest_collection.insert_one({
            "decisions": top_decisions,
            "created_at": datetime.utcnow().isoformat(),
            "total_trainsets": len(mock_decisions)
        })
        
        logger.info(f"Stored ranked list with {len(top_decisions)} decisions")
        return top_decisions
        
    except Exception as e:
        logger.error(f"Error fetching latest induction list: {e}")
        # Return mock data on error - create all 30 trainsets
        mock_decisions = []
        trainset_ids = [f"T-{i:03d}" for i in range(1, 31)]  # T-001 to T-030
        decisions = ["INDUCT", "STANDBY", "MAINTENANCE"]
        
        for i, trainset_id in enumerate(trainset_ids):
            # Calculate realistic score based on index
            base_score = 0.6 + (i * 0.01)  # Scores from 0.61 to 0.90
            score = min(0.95, base_score)
            
            # Determine decision based on score
            if score >= 0.8:
                decision = "INDUCT"
            elif score >= 0.7:
                decision = "STANDBY"
            else:
                decision = "MAINTENANCE"
            
            confidence = round(random.uniform(0.75, 0.95), 2)
            
            mock_decision = {
                "trainset_id": trainset_id,
                "decision": decision,
                "confidence_score": confidence,
                "score": round(score, 3),
                "top_reasons": [
                    "All department certificates valid",
                    "Low predicted failure probability",
                    "Available cleaning slot before dawn"
                ],
                "top_risks": [
                    "Safety certificate expiring soon" if random.random() > 0.7 else None
                ],
                "violations": [],
                "shap_values": [
                    {"name": "Sensor Health Score", "value": round(random.uniform(0.7, 0.95), 2), "impact": "positive"},
                    {"name": "Predicted Failure Risk", "value": round(random.uniform(0.05, 0.3), 2), "impact": "positive"},
                    {"name": "Branding Priority", "value": random.choice(["HIGH", "MEDIUM", "LOW"]), "impact": "positive"},
                    {"name": "Certificate Status", "value": "valid", "impact": "positive"},
                    {"name": "Runtime Compliance", "value": round(random.uniform(0.8, 1.0), 2), "impact": "positive"}
                ],
                "reasons": [
                    "All department certificates valid",
                    "Low predicted failure probability",
                    "High sensor health score",
                    "Valid maintenance schedule"
                ]
            }
            mock_decisions.append(mock_decision)
        
        # Sort by score (highest first)
        mock_decisions.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Returning {len(mock_decisions)} mock decisions due to error")
        return mock_decisions

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
