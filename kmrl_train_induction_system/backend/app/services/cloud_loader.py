# backend/app/services/cloud_loader.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

from motor.motor_asyncio import AsyncIOMotorClient
from influxdb_client import InfluxDBClient, Point, WritePrecision
# Redis disabled for now
# from redis.asyncio import Redis
from app.config import settings

logger = logging.getLogger(__name__)

# --------------------------- MongoDB (Motor) --------------------------- #
async def load_to_mongodb(
    mongodb_uri: str,
    database_name: str,
    collections_to_data: Dict[str, List[Dict[str, Any]]],
) -> None:
    client = AsyncIOMotorClient(mongodb_uri)
    db = client[database_name]
    try:
        for collection_name, docs in collections_to_data.items():
            try:
                logger.info(f"MongoDB: clearing {collection_name}")
                await db[collection_name].delete_many({})

                if docs:
                    logger.info(
                        f"MongoDB: inserting {len(docs)} docs into {collection_name}"
                    )
                    await db[collection_name].insert_many(docs)
                else:
                    logger.info(f"MongoDB: {collection_name} empty; skipping insert")
            except Exception as exc:
                logger.exception(
                    f"MongoDB: error while loading {collection_name}: {exc}"
                )
    finally:
        client.close()
        logger.info("MongoDB: done")

# --------------------------- InfluxDB (Cloud) --------------------------- #
def _build_influx_points(sensor_data: List[Dict[str, Any]]) -> List[Point]:
    points: List[Point] = []
    for row in sensor_data:
        ts_raw = row.get("timestamp")
        try:
            ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else ts_raw
        except Exception:
            ts = datetime.utcnow()

        p = (
            Point("sensor_reading")
            .tag("trainset_id", str(row.get("trainset_id", "")))
            .tag("sensor_type", str(row.get("sensor_type", "")))
            .field("health_score", float(row.get("health_score", 0.0)))
            .field("temperature", float(row.get("temperature", 0.0)))
            .time(ts, write_precision=WritePrecision.S)
        )
        points.append(p)
    return points

async def load_to_influxdb(
    url: str,
    token: str,
    org: str,
    bucket: str,
    sensor_data: List[Dict[str, Any]],
    batch_size: int = 5000,
) -> None:
    client = InfluxDBClient(url=url, token=token, org=org)
    try:
        write_api = client.write_api()
        points = _build_influx_points(sensor_data)
        total = len(points)
        logger.info(f"InfluxDB: writing {total} points to {bucket}")

        for start in range(0, total, batch_size):
            batch = points[start : start + batch_size]
            try:
                await asyncio.to_thread(
                    write_api.write, bucket=bucket, org=org, record=batch
                )
                logger.info(f"InfluxDB: wrote {start + len(batch)}/{total}")
            except Exception as exc:
                logger.exception(f"InfluxDB: batch write failed at {start}: {exc}")
    finally:
        client.close()
        logger.info("InfluxDB: done")

# --------------------------- Redis (disabled) --------------------------- #
async def cache_to_redis(
    redis_url: str,
    active_trainsets: List[Dict[str, Any]],
    extra_pairs: List[Tuple[str, str, int]] | None = None,
    active_key: str = "active_trainsets",
    ttl_seconds: int = 3600,
) -> None:
    logger.info("Redis caching skipped (disabled)")

# --------------------------- Orchestrator --------------------------- #
async def load_all_cloud_dbs(
    # Data (required)
    trainsets: List[Dict[str, Any]],
    job_cards: List[Dict[str, Any]],
    branding_contracts: List[Dict[str, Any]],
    cleaning_schedule: List[Dict[str, Any]],
    historical_operations: List[Dict[str, Any]],
    depot_layout: Dict[str, Any] | List[Dict[str, Any]],
    sensor_data: List[Dict[str, Any]],
    # Connection overrides (optional)
    mongodb_uri: str | None = None,
    mongodb_db: str | None = None,
    influx_url: str | None = None,
    influx_token: str | None = None,
    influx_org: str | None = None,
    influx_bucket: str | None = None,
) -> None:
    collections = {
        "trainsets": trainsets,
        "job_cards": job_cards,
        "branding_contracts": branding_contracts,
        "cleaning_schedule": cleaning_schedule,
        "historical_operations": historical_operations,
        "depot_layout": [depot_layout] if isinstance(depot_layout, dict) else depot_layout,
    }

    # Use settings if params are not provided
    mongo_uri = mongodb_uri or settings.mongodb_url
    mongo_db = mongodb_db or settings.database_name
    inf_url = influx_url or settings.influxdb_url
    inf_token = influx_token or settings.influxdb_token
    inf_org = influx_org or settings.influxdb_org
    inf_bucket = influx_bucket or settings.influxdb_bucket

    await load_to_mongodb(mongo_uri, mongo_db, collections)
    await load_to_influxdb(inf_url, inf_token, inf_org, inf_bucket, sensor_data)
    # Skipping Redis caching for now
