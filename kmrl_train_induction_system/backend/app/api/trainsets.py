
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import List, Optional, Dict, Any
from app.models.user import User, UserRole
from app.services.auth_service import require_role, require_permission
from app.api.auth import require_api_key
from app.utils.cloud_database import cloud_db_manager
from app.models.trainset import Trainset, TrainsetReview, ReviewCreate
from datetime import datetime
import logging
from app.utils.explainability import generate_comprehensive_explanation 
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class ExplanationRequest(BaseModel):
    decision: str
    top_reasons: Optional[List[str]] = []
    top_risks: Optional[List[str]] = []

@router.get("/", response_model=List[Trainset])
async def get_all_trainsets(
    status: Optional[str] = None,
    current_user: User = Depends(require_permission("trainsets.view"))
):
    """Get all trainsets with optional status filter"""
    try:
        query = {}
        if status and status != "all":
            query["status"] = status.upper()
            
        collection = await cloud_db_manager.get_collection("trainsets")
        cursor = collection.find(query)
        
        trainsets = []
        async for doc in cursor:
            # Convert _id to string or remove it
            doc.pop('_id', None)
            trainsets.append(doc)
            
        return trainsets
    except Exception as e:
        logger.error(f"Error fetching trainsets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trainsets: {str(e)}")

@router.get("/{trainset_id}")
async def get_trainset_details(
    trainset_id: str,
    _auth = Depends(require_api_key)
):
    """Get detailed information for a specific trainset"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        trainset = await collection.find_one({"trainset_id": trainset_id})
        
        if not trainset:
            raise HTTPException(status_code=404, detail="Trainset not found")
            
        trainset.pop('_id', None)
        
        # Get explanation/decision if available
        assignments_col = await cloud_db_manager.get_collection("assignments")
        # Find latest decision from optimization/assignments
        # This part depends on how you store historical decisions. 
        # For now, let's look at the latest assignment or optimization run.
        # Simple fallback:
        decision = None
        
        return trainset
    except Exception as e:
        logger.error(f"Error fetching trainset details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trainset details: {str(e)}")

@router.get("/{trainset_id}/details")
async def get_full_trainset_details(
    trainset_id: str,
    current_user: User = Depends(require_permission("trainsets.view"))
):
    """Get comprehensive details including history and components"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        trainset = await collection.find_one({"trainset_id": trainset_id})
        
        if not trainset:
            raise HTTPException(status_code=404, detail="Trainset not found")
            
        trainset.pop('_id', None)
        
        # Helper to get decision explanation
        explanation = {}
        
        # Fetch latest induction decision from assignments collection
        assign_col = await cloud_db_manager.get_collection("assignments")
        # Assuming we store decisions there, or likely in a separate 'optimization_history'
        # For this prototype, we'll try to get it from 'induction_decisions' list inside trainset if it exists,
        # OR fetch from the latest optimization result.
        
        # Let's try to fetch the most recent decision from the 'assignments' or 'optimization_results'
        # For now, we will return what's in the trainset document if enriched, or empty.
        
        # NOTE: In a real implementation, you'd join with decision history.
        # Here we mock or extract if available.
        desc_data = trainset.get("last_decision", {})
        
        return {
            "trainset": trainset,
            "maintenance_history": [], # Placeholder
            "component_status": {}, # Placeholder
            "explanation": desc_data # Return stored decision data
        }
        
    except Exception as e:
        logger.error(f"Error fetching full details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{trainset_id}")
async def update_trainset_status(
    trainset_id: str,
    update_data: Dict[str, Any],
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Update trainset status or location"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        
        result = await collection.update_one(
            {"trainset_id": trainset_id},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Trainset not found")
            
        return {"message": "Trainset updated successfully"}
    except Exception as e:
        logger.error(f"Error updating trainset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{trainset_id}/fitness")
async def get_trainset_fitness(
    trainset_id: str,
    current_user: User = Depends(require_role(UserRole.MAINTENANCE_ENGINEER))
):
    """Get calculated fitness score logs"""
    # This would return historical fitness data
    return {"fitness_history": []}


@router.get("/reviews/all")
async def get_all_reviews(
    limit: int = 50,
    current_user: User = Depends(require_permission("trainsets.view"))
):
    try:
        collection = await cloud_db_manager.get_collection("trainset_reviews")
        cursor = collection.find().sort("created_at", -1).limit(limit)
        reviews = []
        async for doc in cursor:
            doc.pop('_id', None)
            reviews.append(doc)
        return reviews
    except Exception as e:
        logger.error(f"Error fetching reviews: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch reviews: {str(e)}")



@router.post("/{trainset_id}/review")
async def submit_review(
    trainset_id: str,
    review: ReviewCreate,
    current_user: User = Depends(require_role(UserRole.PASSENGER))
):
    """Submit a review for a trainset (Passenger only)"""
    if not trainset_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid trainset_id format")
    try:
        collection = await cloud_db_manager.get_collection("trainset_reviews")
        
        review_doc = {
            "trainset_id": trainset_id,
            "user_id": current_user.id,
            "username": current_user.username,
            "rating": review.rating,
            "comment": review.comment,
            "created_at": datetime.now().isoformat()
        }
        
        await collection.insert_one(review_doc)
        
        logger.info(f"Review submitted for {trainset_id} by {current_user.username}")
        
        return {"message": "Review submitted successfully"}
        
    except Exception as e:
        logger.error(f"Error submitting review: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {str(e)}")

@router.post("/{trainset_id}/explain")
async def generate_explanation(
    trainset_id: str,
    request: ExplanationRequest,
    _auth = Depends(require_api_key)
):
    """Generate an on-demand AI explanation for a trainset decision"""
    try:
        collection = await cloud_db_manager.get_collection("trainsets")
        trainset = await collection.find_one({"trainset_id": trainset_id})
        
        if not trainset:
            raise HTTPException(status_code=404, detail="Trainset not found")
            
        trainset.pop('_id', None)
        
        # Use provided decision or default to UNKNOWN
        decision = request.decision
        
        # Call Gemini API
        # We pass empty 'generated_reasons' if not available, or use the provided ones for context
        generated_reasons = {
            "top_reasons": request.top_reasons,
            "top_risks": request.top_risks
        }
        
        explanation = await generate_comprehensive_explanation(trainset, decision)
        
        return explanation
        
    except Exception as e:
        logger.error(f"Error generating explanation for {trainset_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation: {str(e)}")
