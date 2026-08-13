"""Mongo-backed trainset review repository."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from app.repositories.protocols import ReviewRepository
from app.utils.cloud_database import cloud_db_manager


class MongoReviewRepository(ReviewRepository):
    async def list_reviews(self, limit: int = 50) -> Sequence[Mapping[str, Any]]:
        collection = await cloud_db_manager.get_collection("trainset_reviews")
        cursor = collection.find().sort("created_at", -1).limit(limit)
        reviews = []
        async for document in cursor:
            item = dict(document)
            item.pop("_id", None)
            reviews.append(item)
        return reviews

    async def create_review(self, payload: Mapping[str, Any]) -> str:
        collection = await cloud_db_manager.get_collection("trainset_reviews")
        result = await collection.insert_one(dict(payload))
        return str(result.inserted_id)
