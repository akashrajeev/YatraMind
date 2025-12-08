# backend/app/services/notification_service.py
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import uuid
import asyncio

from app.models.notification import Notification, NotificationCreate, NotificationType, NotificationPriority, NotificationStatus, NotificationChannel
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for managing notifications and alerts"""
    
    def __init__(self):
        self.socket_connections = set()  # Store active socket connections
    
    async def create_notification(
        self,
        type: str,
        priority: str,
        title: str,
        message: str,
        user_id: Optional[str] = None,
        role: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        channels: Optional[List[str]] = None
    ) -> str:
        """Create a new notification"""
        try:
            notification_id = str(uuid.uuid4())
            
            notification = Notification(
                id=notification_id,
                type=NotificationType(type),
                priority=NotificationPriority(priority),
                title=title,
                message=message,
                user_id=user_id,
                role=role,
                data=data or {},
                channels=[NotificationChannel(ch) for ch in (channels or ["IN_APP"])]
            )
            
            # Save to database
            collection = await cloud_db_manager.get_collection("notifications")
            await collection.insert_one(notification.dict())
            
            # Send real-time notification
            await self._send_realtime_notification(notification)
            
            # Send via other channels
            await self._send_notification_channels(notification)
            
            logger.info(f"Created notification {notification_id}: {title}")
            return notification_id
            
        except Exception as e:
            logger.error(f"Error creating notification: {e}")
            raise
    
    async def get_user_notifications(
        self,
        user_id: str,
        role: Optional[str] = None,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Notification]:
        """Get notifications for a specific user or role"""
        try:
            collection = await cloud_db_manager.get_collection("notifications")
            
            # Filter by user_id OR role
            match_conditions = [{"user_id": user_id}]
            if role:
                match_conditions.append({"role": role})
            
            filter_query = {"$or": match_conditions}
            
            if unread_only:
                filter_query["status"] = {"$in": [NotificationStatus.PENDING.value, NotificationStatus.SENT.value]}
            
            cursor = collection.find(filter_query).sort("created_at", -1).limit(limit)
            notifications = []
            
            async for doc in cursor:
                doc.pop('_id', None)
                notifications.append(Notification(**doc))
            
            return notifications
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    async def acknowledge_notification(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Acknowledge a notification"""
        try:
            collection = await cloud_db_manager.get_collection("notifications")
            
            result = await collection.update_one(
                {"id": notification_id},
                {
                    "$set": {
                        "status": NotificationStatus.ACKNOWLEDGED.value,
                        "acknowledged_by": user_id,
                        "acknowledged_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                # Send real-time update
                await self._send_realtime_notification_update(notification_id, "acknowledged")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging notification: {e}")
            return False
    
    async def create_alert(
        self,
        type: str,
        category: str,
        trainset_id: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a critical alert"""
        try:
            alert_id = str(uuid.uuid4())
            
            alert = {
                "id": alert_id,
                "type": type,
                "category": category,
                "trainset_id": trainset_id,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "acknowledged": False,
                "data": data or {}
            }
            
            # Save to database
            collection = await cloud_db_manager.get_collection("alerts")
            await collection.insert_one(alert)
            
            # Create notification for all relevant users
            await self.create_notification(
                type="ALERT",
                priority="CRITICAL" if type == "CRITICAL" else "HIGH",
                title=f"Alert: {category}",
                message=f"{trainset_id}: {message}",
                data={"alert_id": alert_id, "trainset_id": trainset_id, "category": category}
            )
            
            # Send real-time alert
            await self._send_realtime_alert(alert)
            
            logger.warning(f"Created alert {alert_id}: {message}")
            return alert_id
            
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
            raise
    
    async def get_active_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get active alerts"""
        try:
            collection = await cloud_db_manager.get_collection("alerts")
            cursor = collection.find({"acknowledged": False}).sort("timestamp", -1).limit(limit)
            
            alerts = []
            async for doc in cursor:
                doc.pop('_id', None)
                alerts.append(doc)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return []
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str
    ) -> bool:
        """Acknowledge an alert"""
        try:
            collection = await cloud_db_manager.get_collection("alerts")
            
            result = await collection.update_one(
                {"id": alert_id},
                {
                    "$set": {
                        "acknowledged": True,
                        "acknowledged_by": user_id,
                        "acknowledged_at": datetime.utcnow().isoformat()
                    }
                }
            )
            
            if result.modified_count > 0:
                # Send real-time update
                await self._send_realtime_alert_update(alert_id, "acknowledged")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return False
    
    async def cleanup_expired_notifications(self) -> int:
        """Clean up expired notifications"""
        try:
            collection = await cloud_db_manager.get_collection("notifications")
            cutoff_time = datetime.utcnow() - timedelta(days=30)
            
            result = await collection.delete_many({
                "created_at": {"$lt": cutoff_time},
                "status": {"$in": [NotificationStatus.ACKNOWLEDGED.value, NotificationStatus.FAILED.value]}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} expired notifications")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up notifications: {e}")
            return 0
    
    async def get_notification_summary(self) -> Dict[str, Any]:
        """Get notification summary statistics"""
        try:
            collection = await cloud_db_manager.get_collection("notifications")
            
            total = await collection.count_documents({})
            unread = await collection.count_documents({
                "status": {"$in": [NotificationStatus.PENDING.value, NotificationStatus.SENT.value]}
            })
            critical = await collection.count_documents({
                "priority": NotificationPriority.CRITICAL.value
            })
            failed = await collection.count_documents({
                "status": NotificationStatus.FAILED.value
            })
            
            return {
                "total_notifications": total,
                "unread_count": unread,
                "critical_count": critical,
                "failed_count": failed
            }
            
        except Exception as e:
            logger.error(f"Error getting notification summary: {e}")
            return {}
    
    async def _send_realtime_notification(self, notification: Notification):
        """Send real-time notification via socket.io"""
        try:
            # This would integrate with socket.io to send real-time updates
            # For now, we'll just log it
            logger.info(f"Real-time notification: {notification.title}")
            
            # In a real implementation, you would:
            # 1. Get the socket.io instance
            # 2. Emit to specific user or broadcast
            # 3. Handle connection management
            
        except Exception as e:
            logger.error(f"Error sending real-time notification: {e}")
    
    async def _send_realtime_notification_update(self, notification_id: str, status: str):
        """Send real-time notification update"""
        try:
            logger.info(f"Real-time notification update: {notification_id} - {status}")
        except Exception as e:
            logger.error(f"Error sending real-time notification update: {e}")
    
    async def _send_realtime_alert(self, alert: Dict[str, Any]):
        """Send real-time alert"""
        try:
            logger.warning(f"Real-time alert: {alert['message']}")
        except Exception as e:
            logger.error(f"Error sending real-time alert: {e}")
    
    async def _send_realtime_alert_update(self, alert_id: str, status: str):
        """Send real-time alert update"""
        try:
            logger.info(f"Real-time alert update: {alert_id} - {status}")
        except Exception as e:
            logger.error(f"Error sending real-time alert update: {e}")
    
    async def _send_notification_channels(self, notification: Notification):
        """Send notification via configured channels"""
        try:
            for channel in notification.channels:
                if channel == NotificationChannel.EMAIL:
                    await self._send_email_notification(notification)
                elif channel == NotificationChannel.SMS:
                    await self._send_sms_notification(notification)
                elif channel == NotificationChannel.PUSH:
                    await self._send_push_notification(notification)
                elif channel == NotificationChannel.WEBHOOK:
                    await self._send_webhook_notification(notification)
                    
        except Exception as e:
            logger.error(f"Error sending notification channels: {e}")
    
    async def _send_email_notification(self, notification: Notification):
        """Send email notification"""
        try:
            # This would integrate with an email service like SendGrid, SES, etc.
            logger.info(f"Email notification sent: {notification.title}")
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    async def _send_sms_notification(self, notification: Notification):
        """Send SMS notification"""
        try:
            # This would integrate with an SMS service like Twilio, AWS SNS, etc.
            logger.info(f"SMS notification sent: {notification.title}")
        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")
    
    async def _send_push_notification(self, notification: Notification):
        """Send push notification"""
        try:
            # This would integrate with Firebase Cloud Messaging, etc.
            logger.info(f"Push notification sent: {notification.title}")
        except Exception as e:
            logger.error(f"Error sending push notification: {e}")
    
    async def _send_webhook_notification(self, notification: Notification):
        """Send webhook notification"""
        try:
            # This would send HTTP POST to configured webhook URLs
            logger.info(f"Webhook notification sent: {notification.title}")
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
