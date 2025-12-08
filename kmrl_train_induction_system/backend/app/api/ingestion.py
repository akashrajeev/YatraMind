# backend/app/api/ingestion.py
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from typing import Dict, Any, List, Optional
from app.services.data_ingestion import DataIngestionService
from app.services.mqtt_client import iot_streamer
from app.security import require_api_key
from app.services.auth_service import require_role
from app.models.user import UserRole, User
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/ingest/all")
async def ingest_all_sources(background_tasks: BackgroundTasks, current_user: User = Depends(require_role(UserRole.ADMIN))):
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
async def ingest_maximo_data(background_tasks: BackgroundTasks, current_user: User = Depends(require_role(UserRole.ADMIN))):
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

@router.post("/ingest/timeseries/upload")
async def upload_timeseries(file: UploadFile = File(...)):
    """Upload time-series CSV and persist to InfluxDB; store downsample in MongoDB."""
    try:
        svc = DataIngestionService()
        content = await file.read()
        import pandas as pd
        import io
        df = pd.read_csv(io.BytesIO(content))
        df.columns = df.columns.str.lower()
        required = {"trainset_id", "sensor_type", "timestamp"}
        if not required.issubset(set(df.columns)):
            raise HTTPException(status_code=400, detail="Missing required columns in time-series CSV")

        # Write each row to InfluxDB via cloud_db_manager
        records = df.to_dict(orient="records")
        from app.utils.cloud_database import cloud_db_manager
        written = 0
        for r in records:
            metric = {
                "trainset_id": r.get("trainset_id"),
                "sensor_type": r.get("sensor_type", "uploaded"),
                "health_score": float(r.get("health_score", 0.0)),
                "temperature": float(r.get("temperature", 0.0)),
                "timestamp": str(r.get("timestamp")),
            }
            ok = await cloud_db_manager.write_sensor_data(metric)
            written += 1 if ok else 0

        # Downsampled copy (mean by trainset_id, sensor_type per hour)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            grouped = (
                df.set_index("timestamp")
                .groupby([pd.Grouper(freq="1H"), "trainset_id", "sensor_type"]).mean(numeric_only=True)
                .reset_index()
            )
            col = await cloud_db_manager.get_collection("timeseries_downsample")
            docs = grouped.to_dict(orient="records")
            for d in docs:
                d["ingested_at"] = pd.Timestamp.utcnow().isoformat()
            if docs:
                await col.insert_many(docs)

        return {"status": "ok", "written_influx": written}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Time-series upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/maximo/rest")
async def ingest_maximo_via_rest(background_tasks: BackgroundTasks, current_user: User = Depends(require_role(UserRole.ADMIN))):
    """Ingest job cards from IBM Maximo REST API (uses .env config)."""
    try:
        ingestion_service = DataIngestionService()
        # Force the REST path by calling the internal method directly
        background_tasks.add_task(ingestion_service._ingest_maximo_data)
        return {"message": "Maximo REST ingestion started", "status": "processing"}
    except Exception as e:
        logger.error(f"Maximo REST ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/maximo/google")
async def ingest_maximo_from_google(sheet_url: str = Form(...), current_user: User = Depends(require_role(UserRole.ADMIN))):
    """Upload Google Sheets (published CSV/TSV) as Maximo job cards."""
    try:
        svc = DataIngestionService()
        # Reuse generic tabular path via fitness (columns must match expected schema)
        import requests
        r = requests.get(sheet_url, timeout=30)
        r.raise_for_status()
        # Expect columns similar to simulated job_cards fields
        df_result = await svc.ingest_fitness_file(r.content, "jobcards.csv")
        return {"status": "ok", "records_processed": df_result.get("count", 0)}
    except Exception as e:
        logger.error(f"Maximo Google Sheets ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ingest/iot")
async def ingest_iot_data(background_tasks: BackgroundTasks, current_user: User = Depends(require_role(UserRole.ADMIN))):
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
async def get_ingestion_status(_auth=Depends(require_api_key)):
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
async def start_mqtt_streaming(_auth=Depends(require_api_key)):
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
async def stop_mqtt_streaming(_auth=Depends(require_api_key)):
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
async def get_mqtt_status(_auth=Depends(require_api_key)):
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

# --------------------------- New Upload & Source Endpoints --------------------------- #

@router.post("/fitness/upload")
async def upload_fitness_certificates(file: UploadFile = File(...), _auth=Depends(require_api_key)):
    """Upload fitness certificates (CSV/XLSX)."""
    try:
        svc = DataIngestionService()
        content = await file.read()
        result = await svc.ingest_fitness_file(content, file.filename)
        return {"status": "ok", "records_processed": result.get("count", 0)}
    except Exception as e:
        logger.error(f"Fitness upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/branding/upload")
async def upload_branding_contracts(file: UploadFile = File(...), _auth=Depends(require_api_key)):
    """Upload branding contract records (CSV/XLSX)."""
    try:
        svc = DataIngestionService()
        content = await file.read()
        result = await svc.ingest_branding_file(content, file.filename)
        return {"status": "ok", "records_processed": result.get("count", 0)}
    except Exception as e:
        logger.error(f"Branding upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/depot/upload")
async def upload_depot_layout(file: UploadFile = File(...), _auth=Depends(require_api_key)):
    """Upload depot layout GeoJSON file."""
    try:
        svc = DataIngestionService()
        content = await file.read()
        result = await svc.ingest_depot_geojson(content)
        return {"status": "ok", "objects": result.get("objects", 0)}
    except Exception as e:
        logger.error(f"Depot layout upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/cleaning/google")
async def ingest_cleaning_from_google(sheet_url: str = Form(...), _auth=Depends(require_api_key)):
    """Ingest cleaning schedule from a Google Sheets published CSV/TSV URL."""
    try:
        svc = DataIngestionService()
        result = await svc.ingest_cleaning_google_sheet(sheet_url)
        return {"status": "ok", "records_processed": result.get("count", 0)}
    except Exception as e:
        logger.error(f"Cleaning sheet ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/ingest/n8n/upload")
async def upload_to_n8n(files: List[UploadFile] = File(...)):
    """
    Upload multiple files to be forwarded to n8n webhook.
    """
    try:
        svc = DataIngestionService()
        
        file_data_list = []
        for file in files:
            content = await file.read()
            file_data_list.append((file.filename, content, file.content_type))
            
        result = await svc.send_files_to_n8n(file_data_list)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.error(f"N8N upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"N8N upload failed: {str(e)}")

from typing import List, Dict, Any, Union

@router.post("/ingest/n8n/result")
async def receive_n8n_result(
    data: Union[Dict[str, Any], List[Any]], 
    apply_updates: bool = True,
    _auth=Depends(require_api_key)
):
    """
    Receive processed JSON result from n8n.
    
    - **apply_updates**: If True (default), updates system state (fitness, job cards, etc.).
                         If False, only logs the raw data to n8n_ingested_data collection.
    """
    try:
        svc = DataIngestionService()
        result = await svc.process_n8n_result(data, apply_updates=apply_updates)
        return result
    except Exception as e:
        logger.error(f"N8N result ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"N8N result ingestion failed: {str(e)}")
