from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.models.trainset import ReviewCreate, Trainset, TrainsetReview
from app.models.user import User, UserRole
from app.security import require_api_key
from app.services.auth_service import require_permission, require_role
from app.services.explainability_service import ExplainabilityService
from app.services.trainset_service import TrainsetService

router = APIRouter()
logger = logging.getLogger(__name__)


class ExplanationRequest(BaseModel):
    decision: str
    top_reasons: List[str] = Field(default_factory=list)
    top_risks: List[str] = Field(default_factory=list)


@router.get("/", response_model=List[Trainset])
async def get_all_trainsets(
    status: Optional[str] = None,
    current_user: User = Depends(require_permission("trainsets.view")),
):
    del current_user
    try:
        return await TrainsetService().list(status=status)
    except Exception as exc:
        logger.exception("Error fetching trainsets: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to fetch trainsets: {exc}")


@router.get("/{trainset_id}")
async def get_trainset_details(trainset_id: str, _auth=Depends(require_api_key)):
    del _auth
    trainset = await TrainsetService().get(trainset_id)
    if not trainset:
        raise HTTPException(status_code=404, detail="Trainset not found")
    return trainset


@router.get("/{trainset_id}/details")
async def get_full_trainset_details(
    trainset_id: str,
    current_user: User = Depends(require_permission("trainsets.view")),
):
    del current_user
    trainset = await TrainsetService().get(trainset_id)
    if not trainset:
        raise HTTPException(status_code=404, detail="Trainset not found")
    return {
        "trainset": trainset,
        "maintenance_history": [],
        "component_status": {},
        "explanation": trainset.get("last_decision", {}),
    }


@router.put("/{trainset_id}")
async def update_trainset_status(
    trainset_id: str,
    update_data: Dict[str, Any],
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    del current_user
    try:
        updated = await TrainsetService().update(trainset_id, update_data)
        if not updated:
            raise HTTPException(status_code=404, detail="Trainset not found")
        return {"message": "Trainset updated successfully"}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error updating trainset: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{trainset_id}/fitness")
async def get_trainset_fitness(
    trainset_id: str,
    current_user: User = Depends(require_role(UserRole.MAINTENANCE_ENGINEER)),
):
    del current_user
    if not await TrainsetService().get(trainset_id):
        raise HTTPException(status_code=404, detail="Trainset not found")
    return {"fitness_history": []}


@router.get("/reviews/all")
async def get_all_reviews(
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(require_permission("trainsets.view")),
):
    del current_user
    try:
        return await TrainsetService().reviews_list(limit=limit)
    except Exception as exc:
        logger.exception("Error fetching reviews: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to fetch reviews: {exc}")


@router.post("/{trainset_id}/review")
async def submit_review(
    trainset_id: str,
    review: ReviewCreate,
    current_user: User = Depends(require_role(UserRole.PASSENGER)),
):
    if not trainset_id.replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid trainset_id format")
    if not await TrainsetService().get(trainset_id):
        raise HTTPException(status_code=404, detail="Trainset not found")
    try:
        await TrainsetService().submit_review(
            trainset_id,
            current_user.id,
            current_user.username,
            review.rating,
            review.comment,
        )
        return {"message": "Review submitted successfully"}
    except Exception as exc:
        logger.exception("Error submitting review: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to submit review: {exc}")


@router.post("/{trainset_id}/explain")
async def generate_explanation(
    trainset_id: str,
    request: ExplanationRequest,
    _auth=Depends(require_api_key),
):
    del _auth
    trainset = await TrainsetService().get(trainset_id)
    if not trainset:
        raise HTTPException(status_code=404, detail="Trainset not found")
    return await ExplainabilityService().explain(trainset, request.decision)
