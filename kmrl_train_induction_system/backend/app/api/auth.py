# backend/app/api/auth.py
from fastapi import APIRouter, HTTPException, Depends, status
from datetime import timedelta
import logging

from app.models.user import User, UserCreate, UserLogin, Token, PasswordChange
from app.services.auth_service import auth_service, get_current_user
from app.security import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

# Token expiration
ACCESS_TOKEN_EXPIRE_MINUTES = 30


@router.post("/login", response_model=Token)
async def login(
    user_credentials: UserLogin,
    _auth=Depends(require_api_key)
):
    """Authenticate user and return access token"""
    try:
        user = await auth_service.authenticate_user(
            user_credentials.email, 
            user_credentials.password
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user account"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth_service.create_access_token(
            data={"sub": user.id}, 
            expires_delta=access_token_expires
        )
        
        # Update last login
        await auth_service.update_last_login(user.id)
        
        logger.info(f"User {user.email} logged in successfully")
        
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
        
        logger.info(f"User {current_user.email} logged out")
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
        
        logger.info(f"User {current_user.email} updated profile")
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
        
        logger.info(f"User {current_user.email} changed password")
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
        
        logger.info(f"Token refreshed for user {current_user.email}")
        
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
