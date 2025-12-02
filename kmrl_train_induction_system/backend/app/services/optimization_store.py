# backend/app/services/optimization_store.py
"""
Helper functions for retrieving stored optimization decisions
"""
import logging
from typing import List, Dict, Any, Optional
from app.utils.cloud_database import cloud_db_manager

logger = logging.getLogger(__name__)


async def get_latest_decisions() -> Optional[List[Dict[str, Any]]]:
    """
    Retrieve the latest induction decisions from MongoDB.
    
    Returns:
        List of decision dictionaries if found, None otherwise.
        Each decision dict has: trainset_id, decision, confidence_score, etc.
    """
    try:
        latest_collection = await cloud_db_manager.get_collection("latest_induction")
        latest_doc = await latest_collection.find_one(sort=[("_meta.updated_at", -1)])
        
        if latest_doc and "decisions" in latest_doc:
            decisions = latest_doc["decisions"]
            if isinstance(decisions, list) and len(decisions) > 0:
                logger.info(f"Retrieved {len(decisions)} decisions from latest_induction collection")
                return decisions
            else:
                logger.warning("Latest induction document found but decisions list is empty")
                return None
        else:
            logger.warning("No latest induction document found in database")
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving latest decisions: {e}", exc_info=True)
        return None


async def get_decisions_from_history() -> Optional[List[Dict[str, Any]]]:
    """
    Fallback: Retrieve decisions from optimization_history collection.
    
    Returns:
        List of decision dictionaries from most recent history entry, None if not found.
    """
    try:
        history_collection = await cloud_db_manager.get_collection("optimization_history")
        history_doc = await history_collection.find_one(sort=[("timestamp", -1)])
        
        if history_doc and "decisions" in history_doc:
            decisions = history_doc["decisions"]
            if isinstance(decisions, list) and len(decisions) > 0:
                logger.info(f"Retrieved {len(decisions)} decisions from optimization_history (fallback)")
                return decisions
            else:
                logger.warning("History document found but decisions list is empty")
                return None
        else:
            logger.warning("No optimization history found in database")
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving decisions from history: {e}", exc_info=True)
        return None

