from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.models.user import User, UserRole
from app.repositories.mongo_users import MongoUserRepository
from app.services.auth_service import auth_service, require_role

router = APIRouter()
user_repository = MongoUserRepository()


@router.get("/pending", response_model=List[User])
async def get_pending_users(
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Get users pending approval through the repository boundary."""
    try:
        documents = await user_repository.list_pending()
        return [User(**dict(document)) for document in documents]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching pending users: {exc}",
        )


@router.post("/{user_id}/approve")
async def approve_user(
    user_id: str,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Approve a pending user."""
    success = await auth_service.approve_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to approve user or user not found",
        )
    return {"message": "User approved successfully"}


@router.post("/{user_id}/reject")
async def reject_user(
    user_id: str,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Reject (delete) a pending user."""
    success = await auth_service.reject_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to reject user or user not found",
        )
    return {"message": "User rejected successfully"}
