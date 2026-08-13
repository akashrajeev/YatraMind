"""Model loading and version lifecycle for ML risk inference."""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import torch

from app.utils.cloud_database import cloud_db_manager


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    version: str
    features: tuple[str, ...]
    feature_means: dict[str, float]
    loaded_at: float


class ModelRegistry:
    """Cache the latest registered model and its metadata in-process."""

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._bundle: ModelBundle | None = None

    async def get_latest(self, *, force_refresh: bool = False) -> ModelBundle:
        now = time.monotonic()
        if (
            not force_refresh
            and self._bundle is not None
            and now - self._bundle.loaded_at < self.ttl_seconds
        ):
            return self._bundle

        collection = await cloud_db_manager.get_collection("models")
        doc = await collection.find_one(sort=[("meta.created_at", -1)])
        if not doc:
            raise RuntimeError("No model registered")

        meta = doc.get("meta", {})
        blob = doc.get("blob")
        if not blob:
            raise RuntimeError("Latest model has no blob")

        buffer = io.BytesIO(blob if isinstance(blob, bytes) else bytes(blob))
        model = torch.jit.load(buffer, map_location="cpu")
        model.eval()

        version = str(meta.get("version", "unknown"))
        features = tuple(str(value) for value in meta.get("features", []))
        feature_means = {
            str(key): float(value)
            for key, value in (meta.get("feature_means") or {}).items()
        }
        bundle = ModelBundle(
            model=model,
            version=version,
            features=features,
            feature_means=feature_means,
            loaded_at=now,
        )
        self._bundle = bundle
        return bundle

    def invalidate(self) -> None:
        self._bundle = None


model_registry = ModelRegistry()
