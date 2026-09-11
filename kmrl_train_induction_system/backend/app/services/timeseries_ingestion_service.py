"""Application service for CSV time-series ingestion."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from app.repositories.mongo_sensor import MongoSensorRepository


class TimeseriesIngestionService:
    """Validate, normalize, and persist uploaded time-series records."""

    REQUIRED_COLUMNS = {"trainset_id", "sensor_type", "timestamp"}

    def __init__(self, repository: MongoSensorRepository | None = None) -> None:
        self.repository = repository or MongoSensorRepository()

    async def ingest_csv(self, content: bytes) -> dict[str, int]:
        dataframe = pd.read_csv(io.BytesIO(content))
        dataframe.columns = dataframe.columns.str.lower()
        missing = self.REQUIRED_COLUMNS - set(dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns in time-series CSV: {sorted(missing)}")

        written = 0
        for record in dataframe.to_dict(orient="records"):
            metric = {
                "trainset_id": record.get("trainset_id"),
                "sensor_type": record.get("sensor_type", "uploaded"),
                "health_score": float(record.get("health_score", 0.0)),
                "temperature": float(record.get("temperature", 0.0)),
                "timestamp": str(record.get("timestamp")),
            }
            if await self.repository.write_sensor_data(metric):
                written += 1

        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
        grouped = (
            dataframe
            .set_index("timestamp")
            .groupby([pd.Grouper(freq="1h"), "trainset_id", "sensor_type"])
            .mean(numeric_only=True)
            .reset_index()
        )
        documents = grouped.to_dict(orient="records")
        for document in documents:
            document["ingested_at"] = datetime.now(timezone.utc).isoformat()
        downsampled = await self.repository.write_downsampled(documents)
        return {"written": written, "downsampled": downsampled}
