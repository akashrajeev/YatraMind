"""In-memory repository implementations for fast deterministic unit tests."""
from __future__ import annotations
from typing import Any, Mapping, Sequence

class MemoryTrainsetRepository:
    def __init__(self, trainsets: Sequence[Mapping[str, Any]] = ()):
        self._trainsets = [dict(item) for item in trainsets]

    async def list_all(self) -> list[Mapping[str, Any]]:
        return [dict(item) for item in self._trainsets]

    async def get(self, trainset_id: str) -> Mapping[str, Any] | None:
        return next((dict(item) for item in self._trainsets if item.get("trainset_id") == trainset_id), None)

class MemoryOptimizationRepository:
    def __init__(self):
        self.runs: list[dict[str, Any]] = []

    async def save_run(self, payload: Mapping[str, Any]) -> str:
        item = dict(payload)
        self.runs.append(item)
        return str(len(self.runs))

    async def get_latest_decisions(self) -> Sequence[Mapping[str, Any]] | None:
        if not self.runs:
            return None
        decisions = self.runs[-1].get("decisions")
        return decisions if isinstance(decisions, list) else None

    async def get_latest_history(self) -> Mapping[str, Any] | None:
        return self.runs[-1] if self.runs else None
