# backend/app/api/ingestion.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
from app.services.data_ingestion import DataIngestionService
from app.services.mqtt_client import iot_streamer
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest/all")
async def ingest_all_sources(background_tasks: BackgroundTasks):
    """Ingest data from all configured sources (Maximo, IoT, Manual, UNS)"""
    try:
        ingestion_service = DataIngestionService()
        
        # Run ingestion in background
        background_tasks.add_task(ingestion_service.ingest_all_sources)
        
        return {
            "message": "Data ingestion started",
            "status": "processing",
            "sources": ["maximo", "iot_sensors", "manual_override", "uns_streams"]
        }
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/ingest/maximo")
async def ingest_maximo_data(background_tasks: BackgroundTasks):
    """Ingest job card data from IBM Maximo"""
    try:
        ingestion_service = DataIngestionService()
        
        # Run Maximo ingestion in background
        background_tasks.add_task(ingestion_service._ingest_maximo_data)
        
        return {
            "message": "Maximo data ingestion started",
            "status": "processing",
            "source": "maximo"
        }
        
    except Exception as e:
        logger.error(f"Maximo ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Maximo ingestion failed: {str(e)}")

@router.post("/ingest/iot")
async def ingest_iot_data(background_tasks: BackgroundTasks):
    """Ingest IoT sensor data"""
    try:
        ingestion_service = DataIngestionService()
        
        # Run IoT ingestion in background
        background_tasks.add_task(ingestion_service._ingest_iot_data)
        
        return {
            "message": "IoT data ingestion started",
            "status": "processing",
            "source": "iot_sensors"
        }
        
    except Exception as e:
        logger.error(f"IoT ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"IoT ingestion failed: {str(e)}")

@router.get("/status")
async def get_ingestion_status():
    """Get current ingestion status"""
    try:
        return {
            "status": "operational",
            "sources": {
                "maximo": "available",
                "iot_sensors": "streaming",
                "manual_override": "available",
                "uns_streams": "available"
            },
            "last_ingestion": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/mqtt/start")
async def start_mqtt_streaming():
    """Start MQTT IoT data streaming"""
    try:
        await iot_streamer.start_streaming()
        
        return {
            "message": "MQTT IoT streaming started",
            "status": "streaming",
            "topics": list(iot_streamer.sensor_topics.values())
        }
        
    except Exception as e:
        logger.error(f"MQTT streaming start failed: {e}")
        raise HTTPException(status_code=500, detail=f"MQTT streaming failed: {str(e)}")

@router.post("/mqtt/stop")
async def stop_mqtt_streaming():
    """Stop MQTT IoT data streaming"""
    try:
        await iot_streamer.stop_streaming()
        
        return {
            "message": "MQTT IoT streaming stopped",
            "status": "stopped"
        }
        
    except Exception as e:
        logger.error(f"MQTT streaming stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"MQTT streaming stop failed: {str(e)}")

@router.get("/mqtt/status")
async def get_mqtt_status():
    """Get MQTT streaming status"""
    try:
        return {
            "status": "connected" if iot_streamer.mqtt_client.connected else "disconnected",
            "topics": list(iot_streamer.sensor_topics.values()),
            "streaming": iot_streamer.mqtt_client.connected
        }
        
    except Exception as e:
        logger.error(f"MQTT status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"MQTT status check failed: {str(e)}")
