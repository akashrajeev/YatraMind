"""Application service for trainset and review workflows."""
from __future__ import annotations
from datetime import datetime
from typing import Any, Mapping, Sequence
from app.repositories.mongo import MongoTrainsetRepository
from app.repositories.mongo_reviews import MongoReviewRepository
from app.repositories.protocols import TrainsetRepository, ReviewRepository


class TrainsetService:
    def __init__(self, trainsets: TrainsetRepository | None = None, reviews: ReviewRepository | None = None) -> None:
        self.trainsets = trainsets or MongoTrainsetRepository()
        self.reviews = reviews or MongoReviewRepository()

    async def list(self, status: str | None = None) -> list[Mapping[str, Any]]:
        return await self.trainsets.list_all(status=status)

    async def get(self, trainset_id: str) -> Mapping[str, Any] | None:
        return await self.trainsets.get(trainset_id)

    async def update(self, trainset_id: str, updates: Mapping[str, Any]) -> bool:
        return await self.trainsets.update(trainset_id, updates)

    async def reviews_list(self, limit: int = 50) -> Sequence[Mapping[str, Any]]:
        return await self.reviews.list_reviews(limit=limit)

    async def submit_review(self, trainset_id: str, user_id: str, username: str, rating: int, comment: str) -> str:
        return await self.reviews.create_review({
            "trainset_id": trainset_id,
            "user_id": user_id,
            "username": username,
            "rating": rating,
            "comment": comment,
            "created_at": datetime.utcnow().isoformat(),
        })
