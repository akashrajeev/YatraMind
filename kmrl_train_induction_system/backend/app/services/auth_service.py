from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import uuid
import secrets
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.models.user import User, UserRole
from app.repositories.mongo_users import MongoUserRepository
from app.repositories.protocols import UserRepository

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

from app.config import settings

if settings.environment.lower() in {"production", "prod"} and not settings.secret_key:
    raise RuntimeError("SECRET_KEY must be configured in production")

# Development/test processes receive an ephemeral secret instead of a predictable
# hard-coded JWT key. Production must provide SECRET_KEY explicitly.
SECRET_KEY = settings.secret_key or secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()


class AuthService:
    """Authentication and authorization application service."""

    def __init__(self, user_repository: UserRepository | None = None):
        self.pwd_context = pwd_context
        self.user_repository = user_repository or MongoUserRepository()

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        password_bytes = plain_password.encode("utf-8")
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
            plain_password = password_bytes.decode("utf-8", errors="ignore")
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        password_bytes = password.encode("utf-8")
        if len(password_bytes) > 72:
            password_bytes = password_bytes[:72]
            password = password_bytes.decode("utf-8", errors="ignore")
        return self.pwd_context.hash(password)

    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        try:
            return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except JWTError:
            return None

    @staticmethod
    def _to_user(document: Dict[str, Any] | None) -> Optional[User]:
        if not document:
            return None
        safe = dict(document)
        safe.pop("_id", None)
        safe.pop("hashed_password", None)
        safe.setdefault("email_verified", False)
        return User(**safe)

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        try:
            user_doc = await self.user_repository.get_by_username(username)
            if not user_doc or not user_doc.get("hashed_password"):
                return None
            if not self.verify_password(password, str(user_doc["hashed_password"])):
                return None
            return self._to_user(dict(user_doc))
        except Exception as exc:
            logger.error("Error authenticating user: %s", exc)
            return None

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        try:
            user_doc = await self.user_repository.get_by_id(user_id)
            return self._to_user(dict(user_doc)) if user_doc else None
        except Exception as exc:
            logger.error("Error getting user by ID: %s", exc)
            return None

    async def create_user(
        self,
        username: str,
        password: str,
        name: str,
        role: str,
        email: Optional[str] = None,
        permissions: Optional[list] = None,
        is_approved: bool = False,
        email_verified: bool = False,
    ) -> User:
        try:
            user_id = str(uuid.uuid4())
            hashed_password = self.get_password_hash(password)
            final_is_approved = True if role == UserRole.PASSENGER else is_approved
            user = User(
                id=user_id,
                username=username,
                email=email,
                name=name,
                role=role,
                permissions=permissions or [],
                created_at=datetime.utcnow(),
                is_active=True,
                is_approved=final_is_approved,
                email_verified=email_verified,
            )
            await self.user_repository.save({**user.dict(), "hashed_password": hashed_password})
            logger.info("Created user: %s", username)
            return user
        except Exception as exc:
            logger.error("Error creating user: %s", exc)
            raise HTTPException(status_code=500, detail="Failed to create user")

    async def approve_user(self, user_id: str) -> bool:
        return await self.user_repository.update(
            user_id,
            {"is_approved": True, "updated_at": datetime.utcnow()},
        )

    async def mark_email_verified(self, user_id: str) -> bool:
        return await self.user_repository.update(
            user_id,
            {"email_verified": True, "updated_at": datetime.utcnow()},
        )

    async def reject_user(self, user_id: str) -> bool:
        return await self.user_repository.delete_pending(user_id)

    async def update_user_permissions(self, user_id: str, permissions: list) -> bool:
        return await self.user_repository.update(
            user_id,
            {"permissions": permissions, "updated_at": datetime.utcnow()},
        )

    def has_permission(self, user: User, permission: str) -> bool:
        if user.role in (UserRole.ADMIN, UserRole.OPERATIONS_MANAGER):
            return True
        return permission in user.permissions

    def has_role(self, user: User, role: str) -> bool:
        return user.role == role

    async def update_last_login(self, user_id: str) -> bool:
        return await self.user_repository.update(
            user_id,
            {"last_login": datetime.utcnow()},
        )

    async def update_user_profile(self, user_id: str, profile_update: dict) -> Optional[User]:
        try:
            allowed_fields = {"name", "department", "phone", "employee_id"}
            update_data = {key: value for key, value in profile_update.items() if key in allowed_fields}
            update_data["updated_at"] = datetime.utcnow()
            if not await self.user_repository.update(user_id, update_data):
                return None
            return await self.get_user_by_id(user_id)
        except Exception as exc:
            logger.error("Error updating user profile: %s", exc)
            return None

    async def verify_current_password(self, user_id: str, password: str) -> bool:
        try:
            user_doc = await self.user_repository.get_by_id(user_id)
            return bool(user_doc and user_doc.get("hashed_password") and self.verify_password(password, str(user_doc["hashed_password"])))
        except Exception as exc:
            logger.error("Error verifying current password: %s", exc)
            return False

    async def update_password(self, user_id: str, new_password: str) -> bool:
        try:
            return await self.user_repository.update(
                user_id,
                {"hashed_password": self.get_password_hash(new_password), "updated_at": datetime.utcnow()},
            )
        except Exception as exc:
            logger.error("Error updating password: %s", exc)
            return False


auth_service = AuthService()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = auth_service.verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        user = await auth_service.get_user_by_id(str(user_id))
        if user is None:
            raise credentials_exception
        return user
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error getting current user: %s", exc)
        raise credentials_exception


def require_permission(permission: str):
    def permission_checker(current_user: User = Depends(get_current_user)):
        if not auth_service.has_permission(current_user, permission):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return current_user
    return permission_checker


def require_role(role: str):
    def role_checker(current_user: User = Depends(get_current_user)):
        if not auth_service.has_role(current_user, role):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role privileges")
        return current_user
    return role_checker
