# backend/app/utils/database.py
import asyncio
import logging
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis
from influxdb_client import InfluxDBClient
from app.config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Real cloud database manager for production use"""
    
    def __init__(self):
        self.mongodb_client = None
        self.redis_client = None
        self.influxdb_client = None
        self.connected = False
    
    async def connect_all(self):
        """Connect to all cloud databases"""
        try:
            # Connect to MongoDB Atlas
            await self._connect_mongodb()
            
            # Connect to Redis Cloud
            await self._connect_redis()
            
            # Connect to InfluxDB Cloud
            await self._connect_influxdb()
            
            self.connected = True
            logger.info("All cloud databases connected successfully")
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    async def close_all(self):
        """Close all database connections"""
        try:
            if self.mongodb_client:
                self.mongodb_client.close()
            
            if self.redis_client:
                await self.redis_client.close()
            
            if self.influxdb_client:
                self.influxdb_client.close()
            
            self.connected = False
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")
    
    async def _connect_mongodb(self):
        """Connect to MongoDB Atlas"""
        try:
            self.mongodb_client = AsyncIOMotorClient(settings.mongodb_url)
            
            # Test connection
            await self.mongodb_client.admin.command('ping')
            logger.info("MongoDB Atlas connected successfully")
            
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    async def _connect_redis(self):
        """Connect to Redis Cloud"""
        try:
            self.redis_client = Redis.from_url(settings.redis_url, decode_responses=True)
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis Cloud connected successfully")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    async def _connect_influxdb(self):
        """Connect to InfluxDB Cloud"""
        try:
            self.influxdb_client = InfluxDBClient(
                url=settings.influxdb_url,
                token=settings.influxdb_token,
                org=settings.influxdb_org
            )
            
            # Test connection
            health = self.influxdb_client.health()
            if health.status == "pass":
                logger.info("InfluxDB Cloud connected successfully")
            else:
                raise Exception(f"InfluxDB health check failed: {health.message}")
                
        except Exception as e:
            logger.error(f"InfluxDB connection failed: {e}")
            raise
    
    async def get_mongodb_collection(self, collection_name: str):
        """Get MongoDB collection"""
        if not self.mongodb_client:
            raise Exception("MongoDB client not connected")
        
        db = self.mongodb_client[settings.database_name]
        return db[collection_name]
    
    async def get_redis_client(self):
        """Get Redis client"""
        if not self.redis_client:
            raise Exception("Redis client not connected")
        
        return self.redis_client
    
    async def get_influxdb_client(self):
        """Get InfluxDB client"""
        if not self.influxdb_client:
            raise Exception("InfluxDB client not connected")
        
        return self.influxdb_client

# Global database manager instance
db_manager = DatabaseManager()
