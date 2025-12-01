# backend/app/services/data_ingestion.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from app.utils.cloud_database import cloud_db_manager
from app.services.data_cleaning import DataCleaningService
from app.config import settings
from app.utils.uns_recorder import record_uns_event
import io
import pandas as pd
import json as _json

logger = logging.getLogger(__name__)

class DataIngestionService:
    """Real-time data ingestion service for heterogeneous inputs"""
    
    def __init__(self):
        self.cleaning_service = DataCleaningService()
        self.ingestion_sources = {
            "maximo": self._ingest_maximo_data,
            "iot_sensors": self._ingest_iot_data,
            "manual_override": self._ingest_manual_data,
            "uns_streams": self._ingest_uns_data
        }
    
    async def ingest_all_sources(self) -> Dict[str, Any]:
        """Ingest data from all configured sources"""
        try:
            logger.info("Starting data ingestion from all sources")
            
            ingestion_results = {}
            
            for source_name, ingest_func in self.ingestion_sources.items():
                try:
                    result = await ingest_func()
                    ingestion_results[source_name] = {
                        "status": "success",
                        "records_processed": result.get("count", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    logger.info(f"Successfully ingested {result.get('count', 0)} records from {source_name}")
                except Exception as e:
                    logger.error(f"Failed to ingest from {source_name}: {e}")
                    ingestion_results[source_name] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
            
            return ingestion_results
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            raise
    
    async def _ingest_maximo_data(self) -> Dict[str, Any]:
        """Ingest job card data from IBM Maximo"""
        try:
            # Check if Maximo API is configured
            is_configured = settings.maximo_base_url and (settings.maximo_api_key or (settings.maximo_username and settings.maximo_password))
            
            if is_configured:
                try:
                    maximo_data = await self._fetch_maximo_job_cards_api()
                except Exception as api_err:
                    # CRITICAL: Do NOT fallback to simulation in production if API fails
                    logger.error(f"Maximo API failed: {api_err}")
                    raise  # Propagate error to prevent split-brain/fake data
            else:
                # Only use simulation if explicitly NOT configured (Dev mode)
                logger.warning("Maximo API not configured - using SIMULATED data")
                maximo_data = await self._fetch_maximo_job_cards()
            
            # Clean and validate data
            cleaned_data = self.cleaning_service.clean_trainset_data(maximo_data)
            
            # Record UNS envelope and normalized docs
            await record_uns_event(
                source="maximo_poller",
                target_collection="job_cards",
                raw_payload={"count": len(maximo_data)},
                normalized_docs=cleaned_data,
                metadata={"mode": "api" if is_configured else "simulated"},
            )
            
            return {"count": len(cleaned_data), "source": "maximo"}
            
        except Exception as e:
            logger.error(f"Maximo ingestion error: {e}")
            raise
    
    async def _ingest_iot_sensor_data_to_influx(self, sensor_data: List[Dict[str, Any]]) -> int:
        count = 0
        for sensor_reading in sensor_data:
            try:
                await cloud_db_manager.write_sensor_data(sensor_reading)
                count += 1
            except Exception as e:
                logger.exception(f"Failed to write sensor reading: {e}")
        return count

    async def _ingest_iot_data(self) -> Dict[str, Any]:
        """Ingest IoT sensor data"""
        try:
            sensor_data = await self._fetch_iot_sensor_data()
            cleaned_sensor_data = self.cleaning_service.clean_sensor_data(sensor_data)
            written = await self._ingest_iot_sensor_data_to_influx(cleaned_sensor_data)
            return {"count": written, "source": "iot_sensors"}
        except Exception as e:
            logger.error(f"IoT ingestion error: {e}")
            raise
    
    async def _ingest_manual_data(self) -> Dict[str, Any]:
        """Ingest manual override data"""
        try:
            manual_data = await self._fetch_manual_overrides()
            collection = await cloud_db_manager.get_collection("manual_overrides")
            await collection.insert_many(manual_data)
            return {"count": len(manual_data), "source": "manual_override"}
        except Exception as e:
            logger.error(f"Manual data ingestion error: {e}")
            raise
    
    async def _ingest_uns_data(self) -> Dict[str, Any]:
        """Ingest UNS (Unified Notification System) streams"""
        try:
            uns_data = await self._fetch_uns_streams()
            collection = await cloud_db_manager.get_collection("uns_notifications")
            await collection.insert_many(uns_data)
            return {"count": len(uns_data), "source": "uns_streams"}
        except Exception as e:
            logger.error(f"UNS ingestion error: {e}")
            raise
    
    async def _fetch_maximo_job_cards(self) -> List[Dict[str, Any]]:
        """Fetch job cards from IBM Maximo (simulated)"""
        # In real implementation, this would connect to Maximo API
        return [
            {
                "job_card_id": f"WO{100000 + i}",
                "trainset_id": f"T-{str(i % 25 + 1).zfill(3)}",
                "work_order_type": "PM",
                "priority": "NORMAL",
                "status": "OPEN",
                "description": f"Preventive maintenance for trainset T-{str(i % 25 + 1).zfill(3)}",
                "created_date": datetime.now().isoformat(),
                "estimated_duration_hours": 4,
                "assigned_technician": f"TECH_{100 + i}",
                "estimated_cost": 5000.0
            }
            for i in range(10)  # Simulate 10 new job cards
        ]

    async def _fetch_maximo_job_cards_api(self) -> List[Dict[str, Any]]:
        """Fetch job cards from IBM Maximo REST API with pagination."""
        import requests
        base = settings.maximo_base_url.rstrip("/")
        url = f"{base}/maximo/oslc/os/wo"  # Example path; adjust to your Maximo object structure
        headers = {"Accept": "application/json"}
        auth = None
        if settings.maximo_api_key:
            headers["apikey"] = settings.maximo_api_key
        elif settings.maximo_username and settings.maximo_password:
            auth = (settings.maximo_username, settings.maximo_password)
        params = {"_limit": 200, "status": "!COMP"}  # open work orders; adjust as needed

        results: List[Dict[str, Any]] = []
        session = requests.Session()
        session.headers.update(headers)
        next_url = url
        while next_url:
            resp = session.get(next_url, params=params if next_url == url else None, auth=auth, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # Map to internal schema
            for item in data.get("member", data.get("rdfs:member", [])):
                results.append({
                    "job_card_id": item.get("wonum") or item.get("wonum").strip() if item.get("wonum") else None,
                    "trainset_id": item.get("assetnum") or "",
                    "work_order_type": item.get("wotype") or item.get("worktype") or "",
                    "priority": item.get("wopriority") or item.get("priority") or "NORMAL",
                    "status": item.get("status") or "OPEN",
                    "description": item.get("description") or "",
                    "created_date": item.get("reportdate") or datetime.now().isoformat(),
                    "estimated_duration_hours": float(item.get("estdur", 0)) or 0,
                    "assigned_technician": item.get("owner") or "",
                    "estimated_cost": float(item.get("estcost", 0)) or 0.0,
                })
            # Discover next page link
            next_url = None
            for link in data.get("link", []):
                if link.get("rel") == "next":
                    next_url = link.get("href")
                    break

            # Stop if no 'member' found (defensive)
            if not data.get("member") and not data.get("rdfs:member"):
                break

        return results
    
    async def _fetch_iot_sensor_data(self) -> List[Dict[str, Any]]:
        """Fetch IoT sensor data (simulated)"""
        import random
        
        sensor_data = []
        for trainset_id in [f"T-{str(i).zfill(3)}" for i in range(1, 26)]:
            for sensor_type in ["bogie_monitoring", "brake_system", "hvac_control"]:
                sensor_data.append({
                    "trainset_id": trainset_id,
                    "sensor_type": sensor_type,
                    "sensor_id": f"{trainset_id}_{sensor_type}_{random.randint(100, 999)}",
                    "health_score": round(random.uniform(0.7, 0.98), 2),
                    "temperature": round(random.uniform(25, 45), 1),
                    "status": "NORMAL",
                    "timestamp": datetime.now().isoformat()
                })
        
        return sensor_data
    
    async def _fetch_manual_overrides(self) -> List[Dict[str, Any]]:
        """Fetch manual override data (simulated)"""
        return [
            {
                "override_id": f"OVR_{i}",
                "trainset_id": f"T-{str(i % 25 + 1).zfill(3)}",
                "override_type": "FORCE_INDUCT",
                "reason": "Special event requirement",
                "authorized_by": "Operations Manager",
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now().timestamp() + 86400).isoformat()
            }
            for i in range(3)  # Simulate 3 manual overrides
        ]
    
    async def _fetch_uns_streams(self) -> List[Dict[str, Any]]:
        """Fetch UNS notification streams (simulated)"""
        return [
            {
                "notification_id": f"UNS_{i}",
                "trainset_id": f"T-{str(i % 25 + 1).zfill(3)}",
                "notification_type": "ALERT",
                "message": f"System alert for trainset T-{str(i % 25 + 1).zfill(3)}",
                "severity": "MEDIUM",
                "timestamp": datetime.now().isoformat(),
                "source": "UNS"
            }
            for i in range(5)  # Simulate 5 UNS notifications
        ]

    # --------------------------- New ingestion helpers --------------------------- #

    async def ingest_fitness_file(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse CSV/XLSX fitness certificates and upsert into MongoDB."""
        df = self._read_tabular(content, filename)
        df.columns = df.columns.str.lower()
        required = {"trainset_id", "dept", "certificate", "status", "valid_from", "valid_to"}
        if not required.issubset(set(df.columns)):
            raise ValueError("Missing required columns for fitness certificates")
        records = df.to_dict(orient="records")
        collection = await cloud_db_manager.get_collection("fitness_certificates")
        ops = 0
        for r in records:
            r["ingested_at"] = datetime.now().isoformat()
            await collection.update_one(
                {"trainset_id": r["trainset_id"], "certificate": r["certificate"]},
                {"$set": r},
                upsert=True,
            )
            ops += 1

        # Denormalize into trainsets collection for direct frontend consumption
        trainsets_col = await cloud_db_manager.get_collection("trainsets")
        for r in records:
            trainset_id = r.get("trainset_id")
            cert_key = str(r.get("certificate", "")).strip().lower()
            update_path = {
                f"fitness_certificates.{cert_key}": {
                    "dept": r.get("dept"),
                    "certificate": r.get("certificate"),
                    "status": r.get("status"),
                    "valid_from": r.get("valid_from"),
                    "valid_to": r.get("valid_to"),
                    "issued_by": r.get("issued_by"),
                    "certificate_id": r.get("certificate_id"),
                    "updated_at": datetime.now().isoformat(),
                },
                "last_updated_sources.fitness": datetime.now().isoformat(),
            }
            await trainsets_col.update_one(
                {"trainset_id": trainset_id},
                {"$set": update_path},
                upsert=True,
            )
        await record_uns_event(
            source="file_upload_service",
            target_collection="fitness_certificates",
            raw_payload={"filename": filename, "count": len(records)},
            normalized_docs=None,
        )
        
        # Trigger optimization refresh after fitness data update
        await self._trigger_optimization_refresh("fitness_upload")
        
        return {"count": ops}

    async def ingest_branding_file(self, content: bytes, filename: str) -> Dict[str, Any]:
        """Parse CSV/XLSX branding contracts and upsert into MongoDB."""
        df = self._read_tabular(content, filename)
        df.columns = df.columns.str.lower()
        required = {"trainset_id", "current_advertiser", "priority", "start_date", "end_date"}
        if not required.issubset(set(df.columns)):
            raise ValueError("Missing required columns for branding records")
        records = df.to_dict(orient="records")
        col = await cloud_db_manager.get_collection("branding_contracts")
        ops = 0
        for r in records:
            r["ingested_at"] = datetime.now().isoformat()
            await col.update_one(
                {"trainset_id": r["trainset_id"], "current_advertiser": r["current_advertiser"]},
                {"$set": r},
                upsert=True,
            )
            ops += 1

        # Denormalize branding summary into trainsets collection
        trainsets_col = await cloud_db_manager.get_collection("trainsets")
        for r in records:
            trainset_id = r.get("trainset_id")
            branding_summary = {
                "current_advertiser": r.get("current_advertiser"),
                "priority": r.get("priority"),
                "start_date": r.get("start_date"),
                "end_date": r.get("end_date"),
                "runtime_requirement_hours": r.get("runtime_requirement_hours"),
                "updated_at": datetime.now().isoformat(),
            }
            await trainsets_col.update_one(
                {"trainset_id": trainset_id},
                {"$set": {"branding": branding_summary, "last_updated_sources.branding": datetime.now().isoformat()}},
                upsert=True,
            )
        await record_uns_event(
            source="branding_contracts_parser",
            target_collection="branding_contracts",
            raw_payload={"filename": filename, "count": len(records)},
            normalized_docs=None,
        )
        
        # Trigger optimization refresh after branding data update
        await self._trigger_optimization_refresh("branding_upload")
        
        return {"count": ops}

    async def ingest_depot_geojson(self, content: bytes) -> Dict[str, Any]:
        """Parse GeoJSON depot layout and store in MongoDB."""
        try:
            data = _json.loads(content.decode("utf-8"))
        except Exception:
            raise ValueError("Invalid GeoJSON")
        col = await cloud_db_manager.get_collection("depot_layout")
        await col.delete_many({})
        version_tag = datetime.now().strftime("%Y%m%d%H%M%S")
        await col.insert_one({
            "layout": data,
            "version": version_tag,
            "ingested_at": datetime.now().isoformat()
        })
        # Propagate depot layout version to all trainsets for reference
        trainsets_col = await cloud_db_manager.get_collection("trainsets")
        await trainsets_col.update_many(
            {},
            {"$set": {"depot_layout_version": version_tag, "last_updated_sources.depot_layout": datetime.now().isoformat()}}
        )
        await record_uns_event(
            source="geojson_ingest",
            target_collection="depot_layout",
            raw_payload={"objects": len(data.get("features", []))},
            normalized_docs=None,
            metadata={"version": version_tag},
        )
        
        # Trigger optimization refresh after depot layout update
        await self._trigger_optimization_refresh("depot_upload")
        
        return {"objects": len(data.get("features", []))}

    async def ingest_cleaning_google_sheet(self, sheet_url: str) -> Dict[str, Any]:
        """Pull cleaning schedule from published Google Sheets CSV/TSV URL and upsert."""
        try:
            df = pd.read_csv(sheet_url)
        except Exception:
            df = pd.read_csv(sheet_url, sep="\t")
        df.columns = df.columns.str.lower()
        required = {"trainset_id", "date", "slot", "bay", "manpower"}
        if not required.issubset(set(df.columns)):
            raise ValueError("Missing required columns for cleaning schedule")
        recs = df.to_dict(orient="records")
        col = await cloud_db_manager.get_collection("cleaning_schedule")
        ops = 0
        for r in recs:
            r["ingested_at"] = datetime.now().isoformat()
            await col.update_one(
                {"trainset_id": r["trainset_id"], "date": r["date"], "slot": r["slot"]},
                {"$set": r},
                upsert=True,
            )
            ops += 1

        # Denormalize latest cleaning info per trainset
        trainsets_col = await cloud_db_manager.get_collection("trainsets")
        # For each trainset, pick the latest by date
        try:
            df_sorted = df.sort_values(by=["trainset_id", "date"]).groupby("trainset_id").tail(1)
            for _, row in df_sorted.iterrows():
                await trainsets_col.update_one(
                    {"trainset_id": row["trainset_id"]},
                    {"$set": {
                        "cleaning_schedule.latest": {
                            "date": str(row.get("date")),
                            "slot": row.get("slot"),
                            "bay": row.get("bay"),
                            "manpower": row.get("manpower"),
                            "updated_at": datetime.now().isoformat(),
                        },
                        "last_updated_sources.cleaning": datetime.now().isoformat(),
                    }},
                    upsert=True,
                )
        except Exception:
            # Best-effort only
            pass
            
        # Trigger optimization refresh after cleaning data update
        await self._trigger_optimization_refresh("cleaning_upload")
        
        return {"count": ops}

    def _read_tabular(self, content: bytes, filename: str) -> pd.DataFrame:
        """Read CSV/XLSX into a DataFrame from bytes."""
        bio = io.BytesIO(content)
        if filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
            return pd.read_excel(bio)
        return pd.read_csv(bio)
    
    async def _trigger_optimization_refresh(self, source: str):
        """Trigger optimization refresh after data uploads"""
        try:
            # Clear any cached optimization results
            optimization_col = await cloud_db_manager.get_collection("optimization_results")
            await optimization_col.delete_many({})
            
            # Log the refresh trigger
            logger.info(f"Optimization refresh triggered by {source}")
            
        except Exception as e:
            logger.error(f"Failed to trigger optimization refresh: {e}")
            # Don't fail the upload if optimization refresh fails

    # --------------------------- N8N Integration --------------------------- #

    async def send_files_to_n8n(self, file_list: List[tuple]) -> Dict[str, Any]:
        """
        Send uploaded files to n8n webhook for processing.
        file_list: List of tuples (filename, content, content_type)
        """
        if not settings.n8n_webhook_url:
            raise ValueError("N8N_WEBHOOK_URL is not configured")
            
        import httpx
        
        try:
            # Prepare files for upload
            # httpx format: files=[('field_name', (filename, content, content_type)), ...]
            # We use 'files' as the field name for all, or 'file' if n8n expects that.
            # Usually 'files' implies an array. Let's use 'files' to be safe for arrays.
            # However, n8n webhook might look for specific field names. 
            # Standard multipart array often uses same key 'files' or 'file[]'.
            # Let's use 'files' as the key.
            
            multipart_files = []
            for fname, fcontent, ftype in file_list:
                multipart_files.append(('files', (fname, fcontent, ftype or 'application/octet-stream')))
            
            # Send to n8n with extended timeout
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(settings.n8n_webhook_url, files=multipart_files)
                response.raise_for_status()
                
                return {
                    "status": "success",
                    "n8n_response": response.json() if response.content else {},
                    "message": f"Successfully sent {len(file_list)} file(s) to n8n"
                }
            
        except Exception as e:
            logger.error(f"Failed to send files to n8n: {e}")
            raise

    async def process_n8n_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and store JSON result received from n8n."""
        try:
            # 1. Store the raw result in a dedicated collection
            collection = await cloud_db_manager.get_collection("n8n_ingested_data")
            
            doc = {
                "data": data,
                "ingested_at": datetime.now().isoformat(),
                "processed": False
            }
            
            result = await collection.insert_one(doc)
            
            # 2. Router Logic: Process 'updates' if present
            updates_processed = 0
            errors = []
            
            if "updates" in data and isinstance(data["updates"], list):
                logger.info(f"Processing {len(data['updates'])} updates from n8n result")
                
                for update_item in data["updates"]:
                    try:
                        source_type = update_item.get("source_type")
                        item_data = update_item.get("data")
                        
                        if not source_type or not item_data:
                            continue
                            
                        if source_type == "fitness":
                            await self._update_fitness_factor(item_data)
                        elif source_type == "job_card":
                            await self._update_job_card_factor(item_data)
                        elif source_type == "branding":
                            await self._update_branding_factor(item_data)
                        elif source_type == "cleaning":
                            await self._update_cleaning_factor(item_data)
                        elif source_type == "iot_sensor":
                            await self._update_iot_factor(item_data)
                        else:
                            logger.warning(f"Unknown source_type in n8n update: {source_type}")
                            continue
                            
                        updates_processed += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process update item: {e}")
                        errors.append(str(e))
                
                # Update the raw doc to mark as processed
                await collection.update_one(
                    {"_id": result.inserted_id},
                    {"$set": {
                        "processed": True, 
                        "updates_processed": updates_processed,
                        "errors": errors
                    }}
                )

            # Record event
            await record_uns_event(
                source="n8n_webhook",
                target_collection="n8n_ingested_data",
                raw_payload={"id": str(result.inserted_id), "updates_count": updates_processed},
                normalized_docs=[doc]
            )
            
            return {
                "status": "stored_and_processed",
                "id": str(result.inserted_id),
                "updates_processed": updates_processed,
                "errors": errors,
                "message": f"N8N result stored. {updates_processed} factors updated."
            }
            
        except Exception as e:
            logger.error(f"Failed to process n8n result: {e}")
            raise

    # --------------------------- Factor Update Helpers --------------------------- #

    async def _update_fitness_factor(self, data: Dict[str, Any]):
        """Update fitness certificate factor."""
        trainset_id = data.get("trainset_id")
        certificate = data.get("certificate")
        if not trainset_id or not certificate:
            raise ValueError("Missing trainset_id or certificate for fitness update")

        # Update specific certificate collection
        col = await cloud_db_manager.get_collection("fitness_certificates")
        data["updated_at"] = datetime.now().isoformat()
        await col.update_one(
            {"trainset_id": trainset_id, "certificate": certificate},
            {"$set": data},
            upsert=True
        )

        # Denormalize to trainsets
        trainsets_col = await cloud_db_manager.get_collection("trainsets")
        cert_key = str(certificate).strip().lower().replace(" ", "_")
        update_path = {
            f"fitness_certificates.{cert_key}": data,
            "last_updated_sources.fitness": datetime.now().isoformat()
        }
        await trainsets_col.update_one(
            {"trainset_id": trainset_id},
            {"$set": update_path},
            upsert=True
        )
        await self._trigger_optimization_refresh("fitness_n8n_update")

    async def _update_job_card_factor(self, data: Dict[str, Any]):
        """Update job card (maintenance) factor."""
        job_card_id = data.get("job_card_id")
        trainset_id = data.get("trainset_id")
        if not job_card_id or not trainset_id:
            raise ValueError("Missing job_card_id or trainset_id for job card update")

        # Update job cards collection
        col = await cloud_db_manager.get_collection("job_cards")
        data["updated_at"] = datetime.now().isoformat()
        await col.update_one(
            {"job_card_id": job_card_id},
            {"$set": data},
            upsert=True
        )
        # Note: Job cards are usually queried dynamically, but we can trigger refresh
        await self._trigger_optimization_refresh("job_card_n8n_update")

    async def _update_branding_factor(self, data: Dict[str, Any]):
        """Update branding factor."""
        trainset_id = data.get("trainset_id")
        advertiser = data.get("current_advertiser")
        if not trainset_id:
            raise ValueError("Missing trainset_id for branding update")

        # Update branding collection
        col = await cloud_db_manager.get_collection("branding_contracts")
        data["updated_at"] = datetime.now().isoformat()
        
        # If advertiser is provided, we treat it as a specific contract update
        query = {"trainset_id": trainset_id}
        if advertiser:
            query["current_advertiser"] = advertiser
            
        await col.update_one(query, {"$set": data}, upsert=True)

        # Denormalize to trainsets
        trainsets_col = await cloud_db_manager.get_collection("trainsets")
        await trainsets_col.update_one(
            {"trainset_id": trainset_id},
            {"$set": {
                "branding": data, 
                "last_updated_sources.branding": datetime.now().isoformat()
            }},
            upsert=True
        )
        await self._trigger_optimization_refresh("branding_n8n_update")

    async def _update_cleaning_factor(self, data: Dict[str, Any]):
        """Update cleaning schedule factor."""
        trainset_id = data.get("trainset_id")
        date = data.get("date")
        if not trainset_id or not date:
            raise ValueError("Missing trainset_id or date for cleaning update")

        col = await cloud_db_manager.get_collection("cleaning_schedule")
        data["updated_at"] = datetime.now().isoformat()
        await col.update_one(
            {"trainset_id": trainset_id, "date": date},
            {"$set": data},
            upsert=True
        )
        await self._trigger_optimization_refresh("cleaning_n8n_update")

    async def _update_iot_factor(self, data: Dict[str, Any]):
        """Update IoT sensor factor."""
        trainset_id = data.get("trainset_id")
        if not trainset_id:
            raise ValueError("Missing trainset_id for IoT update")

        # IoT data typically goes to Influx or a time-series store, 
        # but here we might update the 'current state' in MongoDB for the dashboard
        data["timestamp"] = data.get("timestamp") or datetime.now().isoformat()
        
        # We can reuse the existing helper if it fits, or write directly
        # For now, let's update the trainset's live status
        trainsets_col = await cloud_db_manager.get_collection("trainsets")
        
        # Construct a dynamic update based on sensor type if available
        sensor_type = data.get("sensor_type", "generic_sensor")
        update_path = {
            f"sensors.{sensor_type}": data,
            "last_updated_sources.iot": datetime.now().isoformat()
        }
        
        await trainsets_col.update_one(
            {"trainset_id": trainset_id},
            {"$set": update_path},
            upsert=True
        )
        # No full optimization refresh for high-frequency IoT to avoid thrashing, 
        # unless it's a critical alert (which might be handled by UNS)
