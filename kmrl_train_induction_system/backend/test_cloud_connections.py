#!/usr/bin/env python3
"""
Simple test script for cloud service connections
"""

import asyncio
import os
from app.utils.cloud_database import cloud_db_manager

async def test_connections():
    """Test cloud service connections"""
    print("🔍 Testing Cloud Service Connections...")
    
    # Test MongoDB
    try:
        print("\n📊 Testing MongoDB Atlas...")
        await cloud_db_manager.connect_mongodb()
        
        # Test basic operations
        collection = await cloud_db_manager.get_collection("test")
        test_doc = {"test": True, "service": "mongodb"}
        result = await collection.insert_one(test_doc)
        print(f"   ✅ MongoDB: Insert successful - {result.inserted_id}")
        
        # Clean up
        await collection.delete_one({"_id": result.inserted_id})
        print(f"   ✅ MongoDB: Cleanup successful")
        
    except Exception as e:
        print(f"   ❌ MongoDB failed: {e}")
    
    # Test Redis
    try:
        print("\n🔄 Testing Redis Cloud...")
        await cloud_db_manager.connect_redis()
        
        # Test basic operations
        await cloud_db_manager.cache_set("test_key", "test_value", expiry=60)
        value = await cloud_db_manager.cache_get("test_key")
        print(f"   ✅ Redis: Set/Get successful - {value}")
        
    except Exception as e:
        print(f"   ❌ Redis failed: {e}")
    
    # Test InfluxDB
    try:
        print("\n📈 Testing InfluxDB Cloud...")
        await cloud_db_manager.connect_influxdb()
        
        # Test sensor data write
        test_data = {
            "trainset_id": "TEST",
            "sensor_type": "test",
            "health_score": 1.0,
            "temperature": 25.0,
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        success = await cloud_db_manager.write_sensor_data(test_data)
        if success:
            print(f"   ✅ InfluxDB: Write successful")
        else:
            print(f"   ❌ InfluxDB: Write failed")
        
    except Exception as e:
        print(f"   ❌ InfluxDB failed: {e}")
    
    # Print connection status
    print("\n📋 Connection Status:")
    status = cloud_db_manager.get_connection_status()
    for service, connected in status.items():
        icon = "✅" if connected else "❌"
        print(f"   {icon} {service.upper()}: {'Connected' if connected else 'Not Connected'}")

if __name__ == "__main__":
    asyncio.run(test_connections())
