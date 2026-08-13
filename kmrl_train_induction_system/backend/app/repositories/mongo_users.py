"""Mongo-backed user repository adapter."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

from app.repositories.protocols import UserRepository
from app.models.user import UserRole
from app.utils.cloud_database import cloud_db_manager


class MongoUserRepository(UserRepository):
    def __init__(self, collection_name: str = "users") -> None:
        self.collection_name = collection_name

    async def get_by_username(self, username: str) -> Mapping[str, Any] | None:
        collection = await cloud_db_manager.get_collection(self.collection_name)
        document = await collection.find_one({"username": username})
        if document is None:
            return None
        item = dict(document)
        item.pop("_id", None)
        return item

    async def list_pending(self) -> Sequence[Mapping[str, Any]]:
        collection = await cloud_db_manager.get_collection(self.collection_name)
        query = {
            "is_approved": False,
            "$or": [
                {"role": {"$nin": [
                    UserRole.STATION_SUPERVISOR,
                    UserRole.SUPERVISOR,
                    UserRole.METRO_DRIVER,
                ]}},
                {
                    "role": {"$in": [
                        UserRole.STATION_SUPERVISOR,
                        UserRole.SUPERVISOR,
                        UserRole.METRO_DRIVER,
                    ]},
                    "email_verified": True,
                },
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

    async def save(self, payload: Mapping[str, Any]) -> str:
        collection = await cloud_db_manager.get_collection(self.collection_name)
        document = dict(payload)
        result = await collection.insert_one(document)
        return str(result.inserted_id)
