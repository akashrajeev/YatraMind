"""Mongo/Influx-backed sensor repository adapter."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.repositories.protocols import SensorRepository
from app.utils.cloud_database import cloud_db_manager


class MongoSensorRepository(SensorRepository):
    """Hide telemetry persistence details from ingestion/API modules."""

    async def write_sensor_data(self, payload: Mapping[str, Any]) -> bool:
        return bool(await cloud_db_manager.write_sensor_data(dict(payload)))

    async def write_downsampled(self, records: Sequence[Mapping[str, Any]]) -> int:
        if not records:
            return 0
        collection = await cloud_db_manager.get_collection("timeseries_downsample")
        await collection.insert_many([dict(item) for item in records])
        return len(records)
