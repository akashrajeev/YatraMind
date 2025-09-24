# backend/app/utils/cloud_database_production.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# Production cloud service imports
from motor.motor_asyncio import AsyncIOMotorClient
from influxdb_client import InfluxDBClient, Point, WritePrecision
from redis.asyncio import Redis
import paho.mqtt.client as paho_mqtt

from app.config import settings

logger = logging.getLogger(__name__)

class ProductionCloudDatabaseManager:
    """Production cloud database manager using real cloud services"""
    
    def __init__(self):
        self.mongodb_client: Optional[AsyncIOMotorClient] = None
        self.mongodb_db = None
        self.influxdb_client: Optional[InfluxDBClient] = None
        self.influxdb_write_api = None
        self.redis_client: Optional[Redis] = None
        self.mqtt_client: Optional[paho_mqtt.Client] = None
        
        # Connection status
        self.connections = {
            "mongodb": False,
            "influxdb": False,
            "redis": False,
            "mqtt": False
        }
    
    async def connect_mongodb(self):
        """Connect to MongoDB Atlas"""
        try:
            logger.info("Connecting to MongoDB Atlas...")
            self.mongodb_client = AsyncIOMotorClient(settings.mongodb_url)
            self.mongodb_db = self.mongodb_client[settings.database_name]
            
            # Test connection
            await self.mongodb_client.admin.command('ping')
            self.connections["mongodb"] = True
            logger.info("✅ MongoDB Atlas connected successfully")
            
        except Exception as e:
            logger.error(f"❌ MongoDB Atlas connection failed: {e}")
            self.connections["mongodb"] = False
            raise
    
    async def connect_influxdb(self):
        """Connect to InfluxDB Cloud"""
        try:
            logger.info("Connecting to InfluxDB Cloud...")
            self.influxdb_client = InfluxDBClient(
                url=settings.influxdb_url,
                token=settings.influxdb_token,
                org=settings.influxdb_org
            )
            self.influxdb_write_api = self.influxdb_client.write_api()
            
            # Test connection by writing a test point
            test_point = Point("connection_test").field("test", 1.0)
            self.influxdb_write_api.write(
                bucket=settings.influxdb_bucket,
                org=settings.influxdb_org,
                record=test_point
            )
            self.connections["influxdb"] = True
            logger.info("✅ InfluxDB Cloud connected successfully")
                
        except Exception as e:
            logger.error(f"❌ InfluxDB Cloud connection failed: {e}")
            self.connections["influxdb"] = False
            raise
    
    async def connect_redis(self):
        """Connect to Redis Cloud"""
        try:
            logger.info("Connecting to Redis Cloud...")
            # For rediss:// endpoints, relax cert verification on Windows if needed
            common_kwargs = {
                "decode_responses": True,
                "socket_connect_timeout": 10.0,
                "socket_timeout": 10.0,
                "health_check_interval": 30,
                "retry_on_timeout": True,
            }
            if str(settings.redis_url).startswith("rediss://"):
                self.redis_client = Redis.from_url(
                    settings.redis_url,
                    ssl_cert_reqs=None,
                    **common_kwargs,
                )
            else:
                self.redis_client = Redis.from_url(
                    settings.redis_url,
                    **common_kwargs,
                )
            
            # Test connection
            await self.redis_client.ping()
            self.connections["redis"] = True
            logger.info("✅ Redis Cloud connected successfully")
            
        except Exception as e:
            logger.error(f"❌ Redis Cloud connection failed: {e}")
            self.connections["redis"] = False
            raise
    
    async def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            logger.info("Connecting to MQTT broker...")
            self.mqtt_client = paho_mqtt.Client(client_id="kmrl_system", protocol=paho_mqtt.MQTTv5)

            # Set credentials if provided
            if settings.mqtt_username and settings.mqtt_password:
                self.mqtt_client.username_pw_set(settings.mqtt_username, settings.mqtt_password)

            # Resolve host/port and TLS
            broker_host = getattr(settings, "mqtt_broker_host", None) or settings.mqtt_broker
            broker_port = int(getattr(settings, "mqtt_broker_port", settings.mqtt_port))
            use_tls = str(getattr(settings, "mqtt_use_tls", "")).lower() == "true"
            if use_tls:
                self.mqtt_client.tls_set()

            # Connect to broker
            self.mqtt_client.connect(broker_host, broker_port, 60)
            self.mqtt_client.loop_start()
            
            self.connections["mqtt"] = True
            logger.info("✅ MQTT broker connected successfully")
            
        except Exception as e:
            logger.error(f"❌ MQTT broker connection failed: {e}")
            self.connections["mqtt"] = False
            raise
    
    async def connect_all(self):
        """Connect to all cloud services"""
        try:
            logger.info("Connecting to all cloud services...")
            await self.connect_mongodb()
            await self.connect_influxdb()
            await self.connect_redis()
            await self.connect_mqtt()
            logger.info("✅ All cloud services connected successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to cloud services: {e}")
            raise
    
    async def close_all(self):
        """Close all cloud service connections"""
        try:
            if self.mongodb_client:
                self.mongodb_client.close()
                logger.info("MongoDB connection closed")
            
            if self.influxdb_client:
                self.influxdb_client.close()
                logger.info("InfluxDB connection closed")
            
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            
            if self.mqtt_client:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
                logger.info("MQTT connection closed")
                
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    async def get_collection(self, name: str):
        """Get MongoDB collection"""
        if not self.connections["mongodb"]:
            raise Exception("MongoDB not connected")
        return self.mongodb_db[name]
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get value from Redis cache"""
        if not self.connections["redis"]:
            raise Exception("Redis not connected")
        return await self.redis_client.get(key)
    
    async def cache_set(self, key: str, value: str, expiry: int = 300):
        """Set value in Redis cache"""
        if not self.connections["redis"]:
            raise Exception("Redis not connected")
        await self.redis_client.setex(key, expiry, value)
    
    async def write_sensor_data(self, data: Dict[str, Any]) -> bool:
        """Write sensor data to InfluxDB"""
        if not self.connections["influxdb"]:
            raise Exception("InfluxDB not connected")
        
        try:
            # Parse timestamp
            timestamp = datetime.fromisoformat(data["timestamp"].replace('Z', '+00:00'))
            
            # Create InfluxDB point
            point = (
                Point("sensor_reading")
                .tag("trainset_id", data["trainset_id"])
                .tag("sensor_type", data["sensor_type"])
                .field("health_score", float(data["health_score"]))
                .field("temperature", float(data["temperature"]))
                .time(timestamp, write_precision=WritePrecision.S)
            )
            
            # Add optional fields
            if "vibration_level" in data:
                point.field("vibration_level", float(data["vibration_level"]))
            if "pressure" in data and data["pressure"]:
                point.field("pressure", float(data["pressure"]))
            
            # Write to InfluxDB
            self.influxdb_write_api.write(
                bucket=settings.influxdb_bucket,
                org=settings.influxdb_org,
                record=point
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write sensor data to InfluxDB: {e}")
            return False
    
    async def publish_mqtt_message(self, topic: str, message: Dict[str, Any]):
        """Publish message to MQTT broker"""
        if not self.connections["mqtt"]:
            raise Exception("MQTT not connected")
        
        try:
            payload = json.dumps(message)
            self.mqtt_client.publish(topic, payload)
            logger.debug(f"Published MQTT message to {topic}")
        except Exception as e:
            logger.error(f"Failed to publish MQTT message: {e}")
    
    def get_connection_status(self) -> Dict[str, bool]:
        """Get connection status for all services"""
        return self.connections.copy()
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check for all services"""
        health_status = {
            "overall": True,
            "services": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Check MongoDB
        try:
            if self.connections["mongodb"]:
                await self.mongodb_client.admin.command('ping')
                health_status["services"]["mongodb"] = {"status": "healthy", "details": "Connected"}
            else:
                health_status["services"]["mongodb"] = {"status": "unhealthy", "details": "Not connected"}
                health_status["overall"] = False
        except Exception as e:
            health_status["services"]["mongodb"] = {"status": "unhealthy", "details": str(e)}
            health_status["overall"] = False
        
        # Check InfluxDB
        try:
            if self.connections["influxdb"]:
                health = self.influxdb_client.health()
                if health.status == "pass":
                    health_status["services"]["influxdb"] = {"status": "healthy", "details": "Connected"}
                else:
                    health_status["services"]["influxdb"] = {"status": "unhealthy", "details": health.message}
                    health_status["overall"] = False
            else:
                health_status["services"]["influxdb"] = {"status": "unhealthy", "details": "Not connected"}
                health_status["overall"] = False
        except Exception as e:
            health_status["services"]["influxdb"] = {"status": "unhealthy", "details": str(e)}
            health_status["overall"] = False
        
        # Check Redis
        try:
            if self.connections["redis"]:
                await self.redis_client.ping()
                health_status["services"]["redis"] = {"status": "healthy", "details": "Connected"}
            else:
                health_status["services"]["redis"] = {"status": "unhealthy", "details": "Not connected"}
                health_status["overall"] = False
        except Exception as e:
            health_status["services"]["redis"] = {"status": "unhealthy", "details": str(e)}
            health_status["overall"] = False
        
        # Check MQTT
        try:
            if self.connections["mqtt"] and self.mqtt_client.is_connected():
                health_status["services"]["mqtt"] = {"status": "healthy", "details": "Connected"}
            else:
                health_status["services"]["mqtt"] = {"status": "unhealthy", "details": "Not connected"}
                health_status["overall"] = False
        except Exception as e:
            health_status["services"]["mqtt"] = {"status": "unhealthy", "details": str(e)}
            health_status["overall"] = False
        
        return health_status

# Create production instance
cloud_db_manager = ProductionCloudDatabaseManager()
