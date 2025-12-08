# backend/app/services/email_verification_service.py
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional
import logging
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from app.config import settings
from app.utils.cloud_database import cloud_db_manager
from app.models.user import EmailToken

logger = logging.getLogger(__name__)

# Reuse configuration logic from magic_link_service or config
# It's better to centralized this, but for now copying to avoid breaking external dependencies if they change
# Helper to get env var with multiple fallbacks
def get_env(keys, default=None):
    for key in keys:
        if val := os.getenv(key):
            return val
    return default

smtp_user = get_env(["SMTP_USER", "MAIL_USERNAME", "USER"])
smtp_password = get_env(["SMTP_PASSWORD", "SMTP_PASS", "MAIL_PASSWORD", "PASS"])
smtp_from = get_env(["SMTP_FROM_EMAIL", "MAIL_FROM", "USER"]) # Fallback to USER if FROM is missing
smtp_host = get_env(["SMTP_HOST", "MAIL_SERVER", "HOST"], "smtp-relay.brevo.com")
# Determine TLS/SSL settings
# Gmail/Brevo usually use Port 587 with STARTTLS (tls=True, ssl=False)
user_secure_setting = get_env(["SMTP_USE_TLS", "SECURE"], "true").lower() == "true"
smtp_port = int(get_env(["SMTP_PORT", "EMAIL_PORT"], 587))

# Smart defaults: If port is 587, we almost certainly need STARTTLS, even if user said SECURE=false (confusing it with SSL)
if smtp_port == 587:
    smtp_use_tls = True
    smtp_use_ssl = False
elif smtp_port == 465:
    smtp_use_tls = False
    smtp_use_ssl = True
else:
    # Use user preference
    smtp_use_tls = user_secure_setting
    smtp_use_ssl = get_env(["SMTP_USE_SSL"], "false").lower() == "true"

# Initialize FastMail conditionally to prevent startup crash if creds are missing
fm = None
conf = None

if smtp_user and smtp_password and smtp_from:
    try:
        conf = ConnectionConfig(
            MAIL_USERNAME=smtp_user,
            MAIL_PASSWORD=smtp_password,
            MAIL_FROM=smtp_from,
            MAIL_PORT=smtp_port,
            MAIL_SERVER=smtp_host,
            MAIL_STARTTLS=smtp_use_tls,
            MAIL_SSL_TLS=smtp_use_ssl,
            USE_CREDENTIALS=True
        )
        fm = FastMail(conf)
    except Exception as e:
        logger.warning(f"Failed to initialize email configuration: {e}")
else:
    logger.warning("Email credentials missing. Email verification service will be disabled.")

class EmailVerificationService:
    def __init__(self):
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        
    async def create_verification_token(self, user_id: str) -> str:
        """Generate and save a 6-digit verification OTP"""
        # Generate 6-digit OTP
        otp = str(secrets.randbelow(1000000)).zfill(6)
        expires_at = datetime.utcnow() + timedelta(minutes=15) # OTPs usually shorter life
        
        token_doc = EmailToken(
            user_id=user_id,
            token=otp,
            expires_at=expires_at
        )
        
        try:
            collection = await cloud_db_manager.get_collection("email_tokens")
            await collection.insert_one(token_doc.dict())
            return otp
        except Exception as e:
            logger.error(f"Error creating verification token: {e}")
            raise e

    async def verify_token(self, user_id: str, token: str) -> bool:
        """Verify the OTP matches and is not expired"""
        try:
            collection = await cloud_db_manager.get_collection("email_tokens")
            # Find the most recent valid token for this user
            # We assume user might request multiple OTPs, we should technically invalidate old ones or just check if *any* valid one exists matching the input.
            # Simple approach: Match exact token given.
            token_doc = await collection.find_one({
                "user_id": user_id,
                "token": token
            })
            
            if not token_doc:
                return False
                
            if token_doc["expires_at"] < datetime.utcnow():
                await collection.delete_one({"_id": token_doc["_id"]}) # Cleanup
                return False
                
            # Token valid, delete it (consumables)
            await collection.delete_one({"_id": token_doc["_id"]})
            return True
            
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
            return False

    async def send_verification_email(self, email: str, user_id: str, token: str) -> bool:
        """Send verification email with OTP"""
        if not fm:
            logger.warning("Email service not configured. Cannot send verification email.")
            logger.info(f"DEV MODE - Verification OTP: {token}")
            return True 
            
        try:
            message = MessageSchema(
                subject="Your Verification Code",
                recipients=[email],
                body=f"Your verification code is: {token}\n\nThis code expires in 15 minutes.",
                subtype="plain"
            )
            
            await fm.send_message(message)
            return True
        except Exception as e:
            logger.error(f"Error sending verification email: {e}")
            return False

email_verification_service = EmailVerificationService()
