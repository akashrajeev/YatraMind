# backend/app/api/auth.py
from fastapi import APIRouter, HTTPException, Depends, status
from datetime import timedelta
import logging

from app.models.user import User, UserCreate, UserLogin, Token, PasswordChange, UserRole, VerifyOTP
from app.services.auth_service import auth_service, get_current_user
from app.security import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

# Token expiration
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Import email verification service
from app.services.email_verification_service import email_verification_service
from typing import Union

@router.post("/register", response_model=Union[User, dict])
async def register(user_data: UserCreate):
    """Register a new user"""
    try:
        # Check if username already exists handled by DB constraints or create_user
        
        # Determine if email verification is needed
        needs_verification = user_data.role in [
            UserRole.STATION_SUPERVISOR, 
            UserRole.SUPERVISOR, 
            UserRole.METRO_DRIVER,
            UserRole.MAINTENANCE_HEAD,
            UserRole.BRANDING_DEALER
        ]
        
        # Determine initial email_verified status
        # If needs verification -> False
        # Else -> True (maintain existing behavior for others)
        email_verified = not needs_verification
        
        user = await auth_service.create_user(
            username=user_data.username,
            password=user_data.password,
            name=user_data.name,
            role=user_data.role,
            email=user_data.email,
            permissions=user_data.permissions,
            email_verified=email_verified
        )
        
        if needs_verification and user.email:
            # Generate token and send email
            token = await email_verification_service.create_verification_token(user.id)
            await email_verification_service.send_verification_email(user.email, user.id, token)
            return {
                "message": "Account created. A verification code has been sent to your email. Please verify to continue.",
                "id": str(user.id),
                "email": user.email
            }
            
        return user
        
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        # Return the actual error message for debugging
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )


@router.post("/verify-email")
async def verify_email(verification_data: VerifyOTP):
    """Verify user email with OTP"""
    # Verify token
    is_valid = await email_verification_service.verify_token(verification_data.user_id, verification_data.otp)
    
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired OTP"
        )
        
    # Mark email as verified
    success = await auth_service.mark_email_verified(verification_data.user_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user verification status"
        )
        
    return {"message": "Email verified successfully. Your account is now pending admin approval."}


@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin
):
    """Authenticate user and return access token"""
    try:
        user = await auth_service.authenticate_user(
            user_credentials.username, 
            user_credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user account"
            )
            
        # Email Verification Check for specific roles
        if user.role in [UserRole.STATION_SUPERVISOR, UserRole.SUPERVISOR, UserRole.METRO_DRIVER, UserRole.MAINTENANCE_HEAD, UserRole.BRANDING_DEALER]:
            if not user.email_verified:
                 raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Please verify your email before logging in."
                )
                
            if not user.is_approved:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Your account is pending admin approval."
                )
        elif not user.is_approved and not (user.role == UserRole.PASSENGER): 
             # Keep existing check ("Account pending approval") for other roles if they require approval
             # Note: logic in create_user sets is_approved=True for Passenger.
             # Existing logic in login was: if not user.is_approved: raise 403.
             # We should preserve that for others.
             raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account pending approval. Please contact the administrator."
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_service.create_access_token(
            data={"sub": user.id}, 
            expires_delta=access_token_expires
        )
        
        # Update last login
        await auth_service.update_last_login(user.id)
        
        logger.info(f"User {user.username} logged in successfully")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        )


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Logout user (invalidate token on client side)"""
    try:
        # In a real implementation, you would:
        # 1. Add token to a blacklist
        # 2. Store blacklisted tokens in Redis
        # 3. Check blacklist on token validation
        
        logger.info(f"User {current_user.username} logged out")
        return {"message": "Successfully logged out"}
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during logout"
        )


@router.get("/profile", response_model=User)
async def get_profile(
    current_user: User = Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Get current user profile"""
    return current_user


@router.put("/profile", response_model=User)
async def update_profile(
    profile_update: dict,
    current_user: User = Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Update current user profile"""
    try:
        # Update user profile
        updated_user = await auth_service.update_user_profile(
            current_user.id, 
            profile_update
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update profile"
            )
        
        logger.info(f"User {current_user.username} updated profile")
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during profile update"
        )


@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: User = Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Change user password"""
    try:
        # Verify current password
        if not await auth_service.verify_current_password(
            current_user.id, 
            password_change.current_password
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Update password
        success = await auth_service.update_password(
            current_user.id, 
            password_change.new_password
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to update password"
            )
        
        logger.info(f"User {current_user.username} changed password")
        return {"message": "Password updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during password change"
        )


@router.post("/refresh-token", response_model=Token)
async def refresh_token(
    current_user: User = Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Refresh access token"""
    try:
        # Create new access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_service.create_access_token(
            data={"sub": current_user.id}, 
            expires_delta=access_token_expires
        )
        
        logger.info(f"Token refreshed for user {current_user.username}")
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=current_user
        )
        
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during token refresh"
        )
