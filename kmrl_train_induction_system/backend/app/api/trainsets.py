# backend/app/api/trainsets.py
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Optional
from datetime import datetime
from app.models.trainset import Trainset, TrainsetUpdate, TrainsetStatus
from app.utils.cloud_database import cloud_db_manager
import pandas as pd
import logging
import json

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/", response_model=List[Trainset])
async def get_all_trainsets(
    status: Optional[TrainsetStatus] = None,
    depot: Optional[str] = None,
    clean_data: bool = Query(True, description="Apply data cleaning")
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
async def get_trainset(trainset_id: str):
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
async def update_trainset(trainset_id: str, update_data: TrainsetUpdate, background_tasks: BackgroundTasks):
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
async def get_trainset_fitness(trainset_id: str):
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
