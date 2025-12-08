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
        # Filter for pending users:
        # 1. Must be unapproved
        # 2. If role is Supervisor/Driver, must also be email_verified
        query = {
            "is_approved": False,
            "$or": [
                # Roles that don't require email verification (or legacy/other roles)
                {"role": {"$nin": [UserRole.STATION_SUPERVISOR, UserRole.SUPERVISOR, UserRole.METRO_DRIVER]}},
                # Roles that DO require email verification
                {
                    "role": {"$in": [UserRole.STATION_SUPERVISOR, UserRole.SUPERVISOR, UserRole.METRO_DRIVER]}, 
                    "email_verified": True
                }
            ]
        }
        cursor = collection.find(query)
        users = []
        async for doc in cursor:
            doc.pop('_id', None)
            doc.pop('hashed_password', None)
            # Ensure safe instantiation even if email_verified missing in old docs
            if "email_verified" not in doc:
                doc["email_verified"] = False 
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
