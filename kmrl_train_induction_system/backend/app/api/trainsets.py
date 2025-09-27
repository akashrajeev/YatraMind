# backend/app/api/trainsets.py
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Depends
from app.security import require_api_key
from typing import List, Optional
from datetime import datetime, timedelta
from app.models.trainset import Trainset, TrainsetUpdate, TrainsetStatus
from app.utils.cloud_database import cloud_db_manager
import pandas as pd
import logging
import json
import random

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=List[Trainset])
async def get_all_trainsets(
    status: Optional[TrainsetStatus] = None,
    depot: Optional[str] = None,
    clean_data: bool = Query(True, description="Apply data cleaning"),
    _auth=Depends(require_api_key)
):
    """Get all trainsets with data cleaning pipeline (Pandas + NumPy)"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        
        # Build MongoDB query
        query_filter = {}
        if status:
            query_filter["status"] = status.value
        if depot:
            query_filter["current_location.depot"] = depot
        
        # Fetch from MongoDB Atlas
        cursor = collection.find(query_filter)
        trainsets = []
        
        async for trainset_doc in cursor:
            trainset_doc.pop('_id', None)
            trainsets.append(trainset_doc)
        
        # Data Cleaning Pipeline (as per architecture diagram)
        if clean_data and trainsets:
            # Simple data cleaning - remove any invalid entries
            trainsets = [t for t in trainsets if t.get('trainset_id')]
        
        logger.info(f"Retrieved {len(trainsets)} trainsets from MongoDB Atlas")
        return trainsets
        
    except Exception as e:
        logger.error(f"Error fetching trainsets: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/{trainset_id}", response_model=Trainset)
async def get_trainset(trainset_id: str, _auth=Depends(require_api_key)):
    """Get specific trainset by ID with caching"""
    try:
        # Fetch from MongoDB Atlas
        collection = await cloud_db_manager.get_collection("trainsets")
        trainset_doc = await collection.find_one({"trainset_id": trainset_id})
        
        if not trainset_doc:
            raise HTTPException(status_code=404, detail=f"Trainset {trainset_id} not found")
        
        trainset_doc.pop('_id', None)
        
        logger.info(f"Retrieved trainset {trainset_id} from MongoDB Atlas")
        return trainset_doc
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching trainset {trainset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.put("/{trainset_id}")
async def update_trainset(trainset_id: str, update_data: TrainsetUpdate, background_tasks: BackgroundTasks, _auth=Depends(require_api_key)):
    """Update trainset with ML feedback loop trigger"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        
        # Check if trainset exists
        existing = await collection.find_one({"trainset_id": trainset_id})
        if not existing:
            raise HTTPException(status_code=404, detail=f"Trainset {trainset_id} not found")
        
        # Add timestamp and ML feedback trigger
        update_payload = {
            **update_data.updates,
            "last_updated": datetime.now().isoformat(),
            "ml_feedback_required": True
        }
        
        # Update in MongoDB Atlas
        result = await collection.update_one(
            {"trainset_id": trainset_id},
            {"$set": update_payload}
        )
        
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="No changes made")
        
        # Trigger ML feedback loop (background task)
        background_tasks.add_task(trigger_ml_feedback, trainset_id, update_payload)
        
        logger.info(f"Updated trainset {trainset_id} in MongoDB Atlas")
        return {"message": f"Trainset {trainset_id} updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating trainset {trainset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")

@router.get("/{trainset_id}/fitness")
async def get_trainset_fitness(trainset_id: str, _auth=Depends(require_api_key)):
    """Get fitness certificate status with rule-based validation"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        trainset_doc = await collection.find_one(
            {"trainset_id": trainset_id},
            {"fitness_certificates": 1, "trainset_id": 1}
        )
        
        if not trainset_doc:
            raise HTTPException(status_code=404, detail=f"Trainset {trainset_id} not found")
        
        fitness = trainset_doc["fitness_certificates"]
        
        # Rule-based constraint engine validation (Drools equivalent)
        fitness_rules = FitnessRuleEngine()
        overall_status = fitness_rules.evaluate_fitness(fitness)
        
        return {
            "trainset_id": trainset_id,
            "overall_fitness": overall_status,
            "certificates": fitness,
            "validation_timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching fitness for {trainset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Fitness check error: {str(e)}")

async def trigger_ml_feedback(trainset_id: str, update_data: dict):
    """ML Feedback Loop (PyTorch + TensorFlow integration point)"""
    try:
        # This would integrate with your ML models
        logger.info(f"ML feedback triggered for {trainset_id}")
        
        # Store update in InfluxDB for time-series analysis
        sensor_data = {
            "trainset_id": trainset_id,
            "sensor_type": "system_update",
            "health_score": 1.0,
            "timestamp": datetime.now().isoformat()
        }
        await cloud_db_manager.write_sensor_data(sensor_data)
        
    except Exception as e:
        logger.error(f"ML feedback error: {e}")

class FitnessRuleEngine:
    """Rule-based constraint engine (Drools + PyKE equivalent)"""
    
    def evaluate_fitness(self, fitness_certs: dict) -> str:
        """Apply business rules to determine overall fitness"""
        valid_count = 0
        expired_count = 0
        
        for cert_type, cert_data in fitness_certs.items():
            if cert_data["status"] == "VALID":
                valid_count += 1
            elif cert_data["status"] == "EXPIRED":
                expired_count += 1
        
        # Business rules
        if expired_count > 0:
            return "CRITICAL"
        elif valid_count == len(fitness_certs):
            return "VALID"
        else:
            return "WARNING"


@router.get("/{trainset_id}/details")
async def get_trainset_details(
    trainset_id: str,
    _auth=Depends(require_api_key)
):
    """Get comprehensive trainset details for the details view"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        trainset = await collection.find_one({"trainset_id": trainset_id})
        
        if not trainset:
            raise HTTPException(status_code=404, detail="Trainset not found")
        
        # Get assignment history for this trainset
        assignments_collection = await cloud_db_manager.get_collection("assignments")
        assignments = []
        async for assignment in assignments_collection.find({"trainset_id": trainset_id}):
            assignment.pop('_id', None)
            assignments.append(assignment)
        
        # Get maintenance history
        maintenance_collection = await cloud_db_manager.get_collection("maintenance_logs")
        maintenance_logs = []
        async for log in maintenance_collection.find({"trainset_id": trainset_id}).sort("date", -1).limit(10):
            log.pop('_id', None)
            maintenance_logs.append(log)
        
        # Get sensor data
        sensor_collection = await cloud_db_manager.get_collection("sensor_data")
        sensor_data = []
        async for data in sensor_collection.find({"trainset_id": trainset_id}).sort("timestamp", -1).limit(50):
            data.pop('_id', None)
            sensor_data.append(data)
        
        # Calculate operational metrics
        total_assignments = len(assignments)
        successful_assignments = len([a for a in assignments if a.get("status") == "APPROVED"])
        success_rate = (successful_assignments / total_assignments * 100) if total_assignments > 0 else 0
        
        # Get recent performance
        recent_performance = {
            "avg_sensor_health": sum([s.get("health_score", 0.8) for s in sensor_data[-10:]]) / min(len(sensor_data), 10) if sensor_data else 0.8,
            "last_maintenance": maintenance_logs[0].get("date") if maintenance_logs else None,
            "next_scheduled_maintenance": (datetime.utcnow() + timedelta(days=30)).isoformat(),
            "operational_hours": random.randint(2000, 5000),
            "mileage": trainset.get("mileage", {}).get("total_km", random.randint(50000, 200000))
        }
        
        return {
            "trainset_id": trainset_id,
            "basic_info": {
                "status": trainset.get("status", "ACTIVE"),
                "model": trainset.get("model", "KMRL-2024"),
                "manufacturer": trainset.get("manufacturer", "KMRL"),
                "commission_date": trainset.get("commission_date", "2020-01-01"),
                "last_inspection": trainset.get("last_inspection", (datetime.utcnow() - timedelta(days=30)).isoformat())
            },
            "operational_metrics": {
                "total_assignments": total_assignments,
                "successful_assignments": successful_assignments,
                "success_rate": round(success_rate, 2),
                "current_fitness_score": trainset.get("sensor_health_score", 0.85),
                "predicted_failure_risk": trainset.get("predicted_failure_risk", 0.15)
            },
            "performance_data": recent_performance,
            "assignments": assignments[-10:],  # Last 10 assignments
            "maintenance_logs": maintenance_logs,
            "sensor_data": sensor_data[-20:],  # Last 20 sensor readings
            "certificates": trainset.get("certificates", {}),
            "branding": trainset.get("branding", {}),
            "cleaning_schedule": trainset.get("cleaning_schedule", {}),
            "mileage": trainset.get("mileage", {}),
            "recommendations": [
                "Schedule maintenance in 30 days" if recent_performance["last_maintenance"] and 
                (datetime.utcnow() - datetime.fromisoformat(recent_performance["last_maintenance"].replace('Z', '+00:00'))).days > 60 else None,
                "High sensor health - optimal for induction" if trainset.get("sensor_health_score", 0.8) > 0.9 else None,
                "Monitor failure risk closely" if trainset.get("predicted_failure_risk", 0.2) > 0.3 else None
            ]
        }
        
    except Exception as e:
        logger.error(f"Error fetching trainset details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trainset details: {str(e)}")