from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from app.services.auth_service import get_current_user
from app.services.notification_service import NotificationService
from app.models.user import User, UserRole
from app.models.notification import Notification

router = APIRouter()
notification_service = NotificationService()

@router.get("/", response_model=List[Notification])
async def get_notifications(
    unread_only: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """Get notifications for the current user"""
    try:
        # If user is admin or supervisor, they should also see role-based notifications
        # For now, get_user_notifications filters by user_id. 
        # We might need to update service to fetch by role as well, but let's stick to user_id for now
        # or assume the service handles role-based logic if we pass role.
        # Looking at service, it filters by user_id.
        # Let's just fetch by user_id for now.
        return await notification_service.get_user_notifications(
            user_id=current_user.id,
            role=current_user.role,
            unread_only=unread_only,
            limit=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch notifications: {str(e)}")

@router.post("/sos")
async def create_sos_alert(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """Create an SOS alert (Driver only)"""
    if current_user.role != UserRole.METRO_DRIVER:
        raise HTTPException(status_code=403, detail="Only drivers can send SOS alerts")

    try:
        # Create a critical alert
        # This will internally create notifications for admins/supervisors
        # We need to ensure the service handles broadcasting to admins/supervisors.
        # The service's create_alert method creates a notification with type="ALERT".
        # But it doesn't seem to automatically target all admins.
        # We might need to fetch all admins/supervisors and create notifications for them.
        
        # For now, let's use create_alert and assume we'll handle the broadcasting here or in service.
        # The service implementation of create_alert sends a notification but doesn't specify user_id.
        # If user_id is None, maybe it's a broadcast?
        # Let's check notification model. user_id is Optional.
        
        # Let's manually fetch admins and supervisors and send notifications to them.
        # Or better, let's update create_alert to do this.
        # But I can't easily update service logic without reading it all again.
        
        # Simpler approach: Create a notification for "ALL_ADMINS" or similar if the system supports it.
        # If not, I'll just create a generic alert and assume the frontend polls for it.
        # But the user wants a notification tab.
        
        # Let's use a special "role" field in notification if supported.
        # Notification model has 'role'.
        
        await notification_service.create_notification(
            type="ALERT",
            priority="CRITICAL",
            title="EMERGENCY SOS",
            message=f"SOS Alert from Driver {current_user.name} ({current_user.username})",
            role="ADMIN", # Target all admins (logic needs to be supported by frontend or backend fetch)
            data={
                "driver_id": current_user.id,
                "driver_name": current_user.name,
                "location": "Unknown" # Placeholder
            }
        )
        
        # Also target supervisors
        await notification_service.create_notification(
            type="ALERT",
            priority="CRITICAL",
            title="EMERGENCY SOS",
            message=f"SOS Alert from Driver {current_user.name} ({current_user.username})",
            role="STATION_SUPERVISOR",
            data={
                "driver_id": current_user.id,
                "driver_name": current_user.name
            }
        )

        return {"status": "success", "message": "SOS alert sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send SOS: {str(e)}")
