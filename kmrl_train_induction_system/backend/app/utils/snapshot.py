"""Snapshot utility for capturing system state for What-If simulations"""
import copy
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from pathlib import Path
from app.utils.cloud_database import cloud_db_manager
from app.services.stabling_optimizer import StablingGeometryOptimizer

logger = logging.getLogger(__name__)


async def capture_snapshot() -> Dict[str, Any]:
    """
    Capture a complete snapshot of the current system state.
    
    Returns a deep-copy JSON with all data needed by optimizer, stabling, and shunting engines.
    Does not mutate the database.
    
    Returns:
        Dictionary with keys:
        - trainsets: List of all trainset data
        - depot_layouts: Depot layout configurations
        - cleaning_slots: Cleaning slot availability
        - certificates: Fitness certificate statuses
        - jobcards: Job card statuses
        - bay_locations: Current bay locations
        - config: Current configuration
        - timestamp: Snapshot capture timestamp
    """
    try:
        logger.info("Capturing system snapshot for What-If simulation")
        
        trainsets_data: List[Dict[str, Any]] = []

        # If Mongo isn't connected (local/dev tests), fall back to bundled mock data
        mongo_connected = cloud_db_manager.connections.get("mongodb", False)

        if mongo_connected:
            collection = await cloud_db_manager.get_collection("trainsets")
            cursor = collection.find({})
            async for trainset_doc in cursor:
                trainset_doc.pop('_id', None)
                trainsets_data.append(trainset_doc)
        else:
            logger.warning("MongoDB not connected; using mock trainset data for snapshot")
            mock_path = Path(__file__).resolve().parents[3] / "data" / "mock" / "trainsets.json"
            with mock_path.open("r", encoding="utf-8") as f:
                trainsets_data = json.load(f)
        
        # Extract depot layouts from stabling optimizer
        stabling_optimizer = StablingGeometryOptimizer()
        depot_layouts = copy.deepcopy(stabling_optimizer.depot_layouts)
        
        # Extract cleaning slots from trainsets
        cleaning_slots = {}
        for trainset in trainsets_data:
            trainset_id = trainset.get("trainset_id")
            cleaning_slots[trainset_id] = {
                "bay_available": trainset.get("has_cleaning_slot", False),
                "manpower_available": trainset.get("has_cleaning_slot", False),
                "requires_cleaning": trainset.get("requires_cleaning", False),
                "cleaning_due_date": trainset.get("cleaning_due_date")
            }
        
        # Extract certificates
        certificates = {}
        for trainset in trainsets_data:
            trainset_id = trainset.get("trainset_id")
            certificates[trainset_id] = copy.deepcopy(trainset.get("fitness_certificates", {}))
        
        # Extract job cards
        jobcards = {}
        for trainset in trainsets_data:
            trainset_id = trainset.get("trainset_id")
            jobcards[trainset_id] = copy.deepcopy(trainset.get("job_cards", {}))
        
        # Extract bay locations
        bay_locations = {}
        for trainset in trainsets_data:
            trainset_id = trainset.get("trainset_id")
            current_loc = trainset.get("current_location", {})
            bay_locations[trainset_id] = {
                "depot": current_loc.get("depot", "Aluva"),
                "bay": current_loc.get("bay", ""),
                "status": trainset.get("status", "STANDBY")
            }
        
        # Get current configuration
        from app.config import settings
        config = {
            "default_hours_per_train": settings.default_hours_per_train,
            "max_hours_warning_threshold_multiplier": settings.max_hours_warning_threshold_multiplier,
            "ml_deterministic_seed": settings.ml_deterministic_seed
        }
        
        snapshot = {
            "trainsets": copy.deepcopy(trainsets_data),
            "depot_layouts": depot_layouts,
            "cleaning_slots": cleaning_slots,
            "certificates": certificates,
            "jobcards": jobcards,
            "bay_locations": bay_locations,
            "config": config,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Snapshot captured: {len(trainsets_data)} trainsets, {len(depot_layouts)} depots")
        return snapshot
        
    except Exception as e:
        logger.error(f"Error capturing snapshot: {e}", exc_info=True)
        raise







