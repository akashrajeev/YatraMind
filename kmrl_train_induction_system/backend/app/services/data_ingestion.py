# backend/app/services/data_ingestion.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from app.utils.cloud_database import cloud_db_manager
from app.services.data_cleaning import DataCleaningService

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
            # Simulate Maximo data ingestion
            maximo_data = await self._fetch_maximo_job_cards()
            
            # Clean and validate data
            cleaned_data = self.cleaning_service.clean_trainset_data(maximo_data)
            
            # Store in MongoDB
            collection = await cloud_db_manager.get_collection("job_cards")
            await collection.delete_many({})  # Clear existing
            await collection.insert_many(cleaned_data)
            
            return {"count": len(cleaned_data), "source": "maximo"}
            
        except Exception as e:
            logger.error(f"Maximo ingestion error: {e}")
            raise
    
    async def _ingest_iot_data(self) -> Dict[str, Any]:
        """Ingest IoT sensor data"""
        try:
            # Simulate IoT sensor data ingestion
            sensor_data = await self._fetch_iot_sensor_data()
            
            # Clean sensor data
            cleaned_sensor_data = self.cleaning_service.clean_sensor_data(sensor_data)
            
            # Write to InfluxDB
            for sensor_reading in cleaned_sensor_data:
                await cloud_db_manager.write_sensor_data(sensor_reading)
            
            return {"count": len(cleaned_sensor_data), "source": "iot_sensors"}
            
        except Exception as e:
            logger.error(f"IoT ingestion error: {e}")
            raise
    
    async def _ingest_manual_data(self) -> Dict[str, Any]:
        """Ingest manual override data"""
        try:
            # Simulate manual data entry
            manual_data = await self._fetch_manual_overrides()
            
            # Process manual overrides
            collection = await cloud_db_manager.get_collection("manual_overrides")
            await collection.insert_many(manual_data)
            
            return {"count": len(manual_data), "source": "manual_override"}
            
        except Exception as e:
            logger.error(f"Manual data ingestion error: {e}")
            raise
    
    async def _ingest_uns_data(self) -> Dict[str, Any]:
        """Ingest UNS (Unified Notification System) streams"""
        try:
            # Simulate UNS data ingestion
            uns_data = await self._fetch_uns_streams()
            
            # Process UNS notifications
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
