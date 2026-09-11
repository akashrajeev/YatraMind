"""Mongo-backed user repository."""
from __future__ import annotations

from typing import Any, Mapping

from app.models.user import UserRole
from app.utils.cloud_database import cloud_db_manager


class MongoUserRepository:
    """Persistence adapter for user queries used by the application layer."""

    def __init__(self, collection_name: str = "users") -> None:
        self.collection_name = collection_name

    async def list_pending(self) -> list[Mapping[str, Any]]:
        collection = await cloud_db_manager.get_collection(self.collection_name)
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
            item = dict(document)
            item.pop("_id", None)
            item.pop("hashed_password", None)
            item.setdefault("email_verified", False)
            users.append(item)
        return users
