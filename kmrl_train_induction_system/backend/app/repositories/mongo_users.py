"""Mongo-backed user repository."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from app.models.user import UserRole
from app.repositories.protocols import UserRepository
from app.utils.cloud_database import cloud_db_manager


class MongoUserRepository(UserRepository):
    """Persistence adapter for user reads and writes."""

    def __init__(self, collection_name: str = "users") -> None:
        self.collection_name = collection_name

    async def _collection(self):
        return await cloud_db_manager.get_collection(self.collection_name)

    async def get_by_username(self, username: str) -> Mapping[str, Any] | None:
        document = await (await self._collection()).find_one({"username": username})
        return self._clean(document) if document else None

    async def get_by_id(self, user_id: str) -> Mapping[str, Any] | None:
        document = await (await self._collection()).find_one({"id": user_id})
        return self._clean(document) if document else None

    async def save(self, payload: Mapping[str, Any]) -> str:
        document = dict(payload)
        await (await self._collection()).insert_one(document)
        return str(document["id"])

    async def update(self, user_id: str, updates: Mapping[str, Any]) -> bool:
        result = await (await self._collection()).update_one(
            {"id": user_id},
            {"$set": dict(updates)},
        )
        return result.modified_count > 0

    async def delete_pending(self, user_id: str) -> bool:
        result = await (await self._collection()).delete_one(
            {"id": user_id, "is_approved": False}
        )
        return result.deleted_count > 0

    async def list_pending(self) -> list[Mapping[str, Any]]:
        collection = await self._collection()
        supervisor_roles = [
            UserRole.STATION_SUPERVISOR,
            UserRole.SUPERVISOR,
            UserRole.METRO_DRIVER,
        ]
        query = {
            "is_approved": False,
            "$or": [
                {"role": {"$nin": supervisor_roles}},
                {"role": {"$in": supervisor_roles}, "email_verified": True},
            ],
        }
        users: list[Mapping[str, Any]] = []
        async for document in collection.find(query):
            item = self._clean(document, strip_sensitive=True)
            item.setdefault("email_verified", False)
            users.append(item)
        return users

    @staticmethod
    def _clean(document: Mapping[str, Any] | None, *, strip_sensitive: bool = False) -> dict[str, Any]:
        item = dict(document or {})
        item.pop("_id", None)
        if strip_sensitive:
            item.pop("hashed_password", None)
        return item
