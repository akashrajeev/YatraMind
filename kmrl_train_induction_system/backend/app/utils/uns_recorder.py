# backend/app/utils/uns_recorder.py
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


async def record_uns_event(
    source: str,
    target_collection: str,
    raw_payload: Any,
    normalized_docs: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a Unified Namespace (UNS) event in MongoDB.

    Stores an envelope with source, ingested_at, raw payload snapshot, normalized count,
    and optional metadata, and writes normalized documents to the target collection
    if provided.
    """
    try:
        envelope_col = await cloud_db_manager.get_collection("uns_events")
        event_doc: Dict[str, Any] = {
            "source": source,
            "target_collection": target_collection,
            "ingested_at": datetime.now().isoformat(),
            "raw_payload": raw_payload,
            "normalized_count": len(normalized_docs or []),
            "metadata": metadata or {},
        }
        await envelope_col.insert_one(event_doc)

        if normalized_docs:
            target_col = await cloud_db_manager.get_collection(target_collection)
            # Bulk insert without dedupe; upstream should upsert if required
            await target_col.insert_many(normalized_docs)
    except Exception as e:
        logger.exception(f"Failed to record UNS event for {source}->{target_collection}: {e}")

