# backend/app/services/auth_service.py
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import uuid
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.user import User
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings - Import from config
from app.config import settings
SECRET_KEY = settings.secret_key or "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Security scheme
security = HTTPBearer()


class AuthService:
    """Service for handling authentication and authorization"""
    
    def __init__(self):
        self.pwd_context = pwd_context
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash with proper encoding handling"""
        # Ensure password is encoded as UTF-8 and truncate to 72 bytes if needed
        # bcrypt has a 72-byte limit for passwords
        password_bytes = plain_password.encode('utf-8')
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
            plain_password = password_bytes.decode('utf-8', errors='ignore')
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hash a password with proper encoding handling"""
        # Ensure password is encoded as UTF-8 and truncate to 72 bytes if needed
        # bcrypt has a 72-byte limit for passwords
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
            password = password_bytes.decode('utf-8', errors='ignore')
        return self.pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user with username and password"""
        try:
            collection = await cloud_db_manager.get_collection("users")
            user_doc = await collection.find_one({"username": username})
            
            if not user_doc:
                return None
            
            if not self.verify_password(password, user_doc["hashed_password"]):
                return None
            
            user_doc.pop('_id', None)
            user_doc.pop('hashed_password', None)
            return User(**user_doc)
            
        except Exception as e:
            logger.error(f"Error authenticating user: {e}")
            return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            collection = await cloud_db_manager.get_collection("users")
            user_doc = await collection.find_one({"id": user_id})
            
            if not user_doc:
                return None
            
            user_doc.pop('_id', None)
            user_doc.pop('hashed_password', None)
            return User(**user_doc)
            
        except Exception as e:
            logger.error(f"Error getting user by ID: {e}")
            return None
    
    async def create_user(
        self,
        username: str,
        password: str,
        name: str,
        role: str,
        email: Optional[str] = None,
        permissions: Optional[list] = None
    ) -> User:
        """Create a new user"""
        try:
            user_id = str(uuid.uuid4())
            hashed_password = self.get_password_hash(password)
            
            user = User(
                id=user_id,
                username=username,
                email=email,
                name=name,
                role=role,
                permissions=permissions or [],
                created_at=datetime.utcnow(),
                is_active=True
            )
            
            collection = await cloud_db_manager.get_collection("users")
            await collection.insert_one({
                **user.dict(),
                "hashed_password": hashed_password
            })
            
            logger.info(f"Created user: {username}")
            return user
            
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise HTTPException(status_code=500, detail="Failed to create user")
    
    async def update_user_permissions(self, user_id: str, permissions: list) -> bool:
        """Update user permissions"""
        try:
            collection = await cloud_db_manager.get_collection("users")
            result = await collection.update_one(
                {"id": user_id},
                {"$set": {"permissions": permissions, "updated_at": datetime.utcnow()}}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating user permissions: {e}")
            return False
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has a specific permission"""
        if user.role == "OPERATIONS_MANAGER":
            return True  # Operations manager has all permissions
        
        return permission in user.permissions
    
    def has_role(self, user: User, role: str) -> bool:
        """Check if user has a specific role"""
        return user.role == role
    
    async def update_last_login(self, user_id: str) -> bool:
        """Update user's last login timestamp"""
        try:
            collection = await cloud_db_manager.get_collection("users")
            result = await collection.update_one(
                {"id": user_id},
                {"$set": {"last_login": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating last login: {e}")
            return False
    
    async def update_user_profile(self, user_id: str, profile_update: dict) -> Optional[User]:
        """Update user profile"""
        try:
            collection = await cloud_db_manager.get_collection("users")
            
            # Remove sensitive fields that shouldn't be updated via profile
            allowed_fields = {"name", "department", "phone", "employee_id"}
            update_data = {k: v for k, v in profile_update.items() if k in allowed_fields}
            update_data["updated_at"] = datetime.utcnow()
            
            result = await collection.update_one(
                {"id": user_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                return await self.get_user_by_id(user_id)
            return None
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return None
    
    async def verify_current_password(self, user_id: str, password: str) -> bool:
        """Verify user's current password"""
        try:
            collection = await cloud_db_manager.get_collection("users")
            user_doc = await collection.find_one({"id": user_id})
            
            if not user_doc:
                return False
            
            return self.verify_password(password, user_doc["hashed_password"])
            
        except Exception as e:
            logger.error(f"Error verifying current password: {e}")
            return False
    
    async def update_password(self, user_id: str, new_password: str) -> bool:
        """Update user password"""
        try:
            hashed_password = self.get_password_hash(new_password)
            collection = await cloud_db_manager.get_collection("users")
            
            result = await collection.update_one(
                {"id": user_id},
                {"$set": {
                    "hashed_password": hashed_password,
                    "updated_at": datetime.utcnow()
                }}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Error updating password: {e}")
            return False


# Global auth service instance
auth_service = AuthService()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        payload = auth_service.verify_token(token)
        
        if payload is None:
            raise credentials_exception
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        user = await auth_service.get_user_by_id(user_id)
        if user is None:
            raise credentials_exception
        
        return user
        
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise credentials_exception


def require_permission(permission: str):
    """Decorator to require a specific permission"""
    def permission_checker(current_user: User = Depends(get_current_user)):
        if not auth_service.has_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return permission_checker


def require_role(role: str):
    """Decorator to require a specific role"""
    def role_checker(current_user: User = Depends(get_current_user)):
        if not auth_service.has_role(current_user, role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role privileges"
            )
        return current_user
    return role_checker
