"""Persistence contracts for the application layer.

These protocols isolate business services from the current MongoDB adapter so
storage can be migrated without rewriting optimization logic.
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class TrainsetRepository(Protocol):
    async def list_all(self) -> list[Mapping[str, Any]]:
        ...

    async def get(self, trainset_id: str) -> Mapping[str, Any] | None:
        ...


class OptimizationRepository(Protocol):
    async def save_run(self, payload: Mapping[str, Any]) -> str:
        ...

    async def get_latest_decisions(self) -> Sequence[Mapping[str, Any]] | None:
        ...

    async def get_latest_history(self) -> Mapping[str, Any] | None:
        ...


class UserRepository(Protocol):
    async def get_by_username(self, username: str) -> Mapping[str, Any] | None:
        ...

    async def save(self, payload: Mapping[str, Any]) -> str:
        ...
