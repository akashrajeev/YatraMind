from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from app.models.user import User, UserRole
from app.services.auth_service import auth_service, get_current_user, require_role
from app.utils.cloud_database import cloud_db_manager

router = APIRouter()

@router.get("/pending", response_model=List[User])
async def get_pending_users(
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Get list of users pending approval"""
    try:
        collection = await cloud_db_manager.get_collection("users")
        cursor = collection.find({"is_approved": False})
        users = []
        async for doc in cursor:
            doc.pop('_id', None)
            doc.pop('hashed_password', None)
            users.append(User(**doc))
        return users
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching pending users: {str(e)}"
        )

@router.post("/{user_id}/approve")
async def approve_user(
    user_id: str,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Approve a pending user"""
    success = await auth_service.approve_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Failed to approve user or user not found"
        )
    return {"message": "User approved successfully"}

@router.post("/{user_id}/reject")
async def reject_user(
    user_id: str,
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Reject (delete) a pending user"""
    success = await auth_service.reject_user(user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Failed to reject user or user not found"
        )
    return {"message": "User rejected successfully"}
