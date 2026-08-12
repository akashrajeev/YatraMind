"""Mongo-backed repository adapters using the existing cloud DB manager."""
from __future__ import annotations
from typing import Any, Mapping
from app.repositories.protocols import TrainsetRepository
from app.utils.cloud_database import cloud_db_manager

class MongoTrainsetRepository(TrainsetRepository):
    """Adapter that hides Motor/Mongo collection details from services."""
    def __init__(self, collection_name: str = "trainsets") -> None:
        self.collection_name = collection_name

    async def list_all(self) -> list[Mapping[str, Any]]:
        collection = await cloud_db_manager.get_collection(self.collection_name)
        documents: list[Mapping[str, Any]] = []
        async for document in collection.find({}):
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
