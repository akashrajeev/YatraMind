"""Mongo-backed trainset repository adapter."""
from __future__ import annotations
from typing import Any, Mapping
from app.repositories.protocols import TrainsetRepository
from app.utils.cloud_database import cloud_db_manager


class MongoTrainsetRepository(TrainsetRepository):
    def __init__(self, collection_name: str = "trainsets") -> None:
        self.collection_name = collection_name

    async def list_all(self, status: str | None = None) -> list[Mapping[str, Any]]:
        collection = await cloud_db_manager.get_collection(self.collection_name)
        query = {"status": status.upper()} if status and status != "all" else {}
        documents = []
        async for document in collection.find(query):
            item = dict(document)
            item.pop("_id", None)
            documents.append(item)
        return documents

    async def get(self, trainset_id: str) -> Mapping[str, Any] | None:
        collection = await cloud_db_manager.get_collection(self.collection_name)
        document = await collection.find_one({"trainset_id": trainset_id})
        if document is None:
            return None
        item = dict(document)
        item.pop("_id", None)
        return item

    async def update(self, trainset_id: str, updates: Mapping[str, Any]) -> bool:
        collection = await cloud_db_manager.get_collection(self.collection_name)
        result = await collection.update_one({"trainset_id": trainset_id}, {"$set": dict(updates)})
        return result.matched_count > 0
