from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form, Depends
from typing import Dict, Any, List, Optional, Union
from app.services.data_ingestion import DataIngestionService
from app.services.n8n_ingestion_compat import N8NDataIngestionService
from app.services.timeseries_ingestion_service import TimeseriesIngestionService
from app.services.mqtt_client import iot_streamer
from app.security import require_api_key
from app.services.auth_service import require_role
from app.models.user import UserRole, User
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
timeseries_service = TimeseriesIngestionService()


@router.post("/ingest/all")
async def ingest_all_sources(background_tasks: BackgroundTasks, current_user: User = Depends(require_role(UserRole.ADMIN))):
    try:
        ingestion_service = DataIngestionService()
        background_tasks.add_task(ingestion_service.ingest_all_sources)
        return {"message": "Data ingestion started", "status": "processing", "sources": ["maximo", "iot_sensors", "manual_override", "uns_streams"]}
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/maximo")
async def ingest_maximo_data(background_tasks: BackgroundTasks, current_user: User = Depends(require_role(UserRole.ADMIN))):
    try:
        ingestion_service = DataIngestionService()
        background_tasks.add_task(ingestion_service._ingest_maximo_data)
        return {"message": "Maximo data ingestion started", "status": "processing", "source": "maximo"}
    except Exception as e:
        logger.error(f"Maximo ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Maximo ingestion failed: {str(e)}")


@router.post("/ingest/timeseries/upload")
async def upload_timeseries(
    file: UploadFile = File(...),
    current_user: User = Depends(require_role(UserRole.ADMIN)),
):
    try:
        content = await file.read()
        result = await timeseries_service.ingest_csv(content)
        return {"status": "ok", "written_influx": result["written"], "downsampled": result["downsampled"]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        logger.error(f"Time-series upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/maximo/rest")
async def ingest_maximo_via_rest(background_tasks: BackgroundTasks, current_user: User = Depends(require_role(UserRole.ADMIN))):
    try:
        ingestion_service = DataIngestionService()
        background_tasks.add_task(ingestion_service._ingest_maximo_data)
        return {"message": "Maximo REST ingestion started", "status": "processing"}
    except Exception as e:
        logger.error(f"Maximo REST ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/maximo/google")
async def ingest_maximo_from_google(sheet_url: str = Form(...), current_user: User = Depends(require_role(UserRole.ADMIN))):
    try:
        svc = DataIngestionService()
        import requests
        r = requests.get(sheet_url, timeout=30)
        r.raise_for_status()
        df_result = await svc.ingest_fitness_file(r.content, "jobcards.csv")
        return {"status": "ok", "records_processed": df_result.get("count", 0)}
    except Exception as e:
        logger.error(f"Maximo Google Sheets ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ingest/iot")
async def ingest_iot_data(background_tasks: BackgroundTasks, current_user: User = Depends(require_role(UserRole.ADMIN))):
    try:
        ingestion_service = DataIngestionService()
        background_tasks.add_task(ingestion_service._ingest_iot_data)
        return {"message": "IoT ingestion started", "status": "processing", "source": "iot_sensors"}
    except Exception as e:
        logger.error(f"IoT ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"IoT ingestion failed: {str(e)}")


@router.get("/status")
async def get_ingestion_status(_auth=Depends(require_api_key)):
    return {"status": "operational", "sources": {"maximo": "available", "iot_sensors": "streaming", "manual_override": "available", "uns_streams": "available"}, "last_ingestion": "2024-01-01T00:00:00Z"}


@router.post("/mqtt/start")
async def start_mqtt_streaming(_auth=Depends(require_api_key)):
    try:
        await iot_streamer.start_streaming()
        return {"message": "MQTT IoT streaming started", "status": "streaming", "topics": list(iot_streamer.sensor_topics.values())}
    except Exception as e:
        logger.error(f"MQTT streaming start failed: {e}")
        raise HTTPException(status_code=500, detail=f"MQTT streaming failed: {str(e)}")


@router.post("/mqtt/stop")
async def stop_mqtt_streaming(_auth=Depends(require_api_key)):
    try:
        await iot_streamer.stop_streaming()
        return {"message": "MQTT IoT streaming stopped", "status": "stopped"}
    except Exception as e:
        logger.error(f"MQTT streaming stop failed: {e}")
        raise HTTPException(status_code=500, detail=f"MQTT streaming stop failed: {str(e)}")


@router.get("/mqtt/status")
async def get_mqtt_status(_auth=Depends(require_api_key)):
    try:
        return {"status": "connected" if iot_streamer.mqtt_client.connected else "disconnected", "topics": list(iot_streamer.sensor_topics.values()), "streaming": iot_streamer.mqtt_client.connected}
    except Exception as e:
        logger.error(f"MQTT status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"MQTT status check failed: {str(e)}")


@router.post("/fitness/upload")
async def upload_fitness_certificates(file: UploadFile = File(...), _auth=Depends(require_api_key)):
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
    try:
        svc = DataIngestionService()
        result = await svc.ingest_cleaning_google_sheet(sheet_url)
        return {"status": "ok", "records_processed": result.get("count", 0)}
    except Exception as e:
        logger.error(f"Cleaning sheet ingestion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ingest/n8n/upload")
async def upload_to_n8n(
    files: List[UploadFile] = File(default=[]),
    file: Optional[UploadFile] = File(default=None),
):
    """Inbound n8n webhook upload."""
    try:
        svc = N8NDataIngestionService()
        selected = list(files)
        if file is not None:
            selected.append(file)
        if not selected:
            raise ValueError("N8N upload requires at least one file")
        file_data_list = []
        for upload in selected:
            content = await upload.read()
            file_data_list.append((upload.filename, content, upload.content_type))
        return await svc.send_files_to_n8n(file_data_list)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        logger.error(f"N8N upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"N8N upload failed: {str(e)}")


@router.post("/ingest/n8n/result")
async def receive_n8n_result(
    data: Union[Dict[str, Any], List[Any]],
    apply_updates: bool = True,
):
    """Inbound n8n webhook result; optional updates are applied to system state."""
    try:
        svc = N8NDataIngestionService()
        return await svc.process_n8n_result(data, apply_updates=apply_updates)
    except Exception as e:
        logger.error(f"N8N result ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"N8N result ingestion failed: {str(e)}")
