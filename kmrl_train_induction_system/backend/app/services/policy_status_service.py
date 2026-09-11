"""Application service for AI policy/model status."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from app.repositories.mongo_policy import MongoPolicyRepository


class PolicyStatusService:
    def __init__(self, repository: MongoPolicyRepository | None = None) -> None:
        self.repository = repository or MongoPolicyRepository()

    async def get_status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "models": await self.repository.latest_model_status(),
            "timestamp": datetime.now().isoformat(),
        }
