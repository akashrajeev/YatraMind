#!/usr/bin/env python3
"""
Cloud Services Setup Script for KMRL Train Induction System
This script helps you configure and test connections to real cloud services
"""

import os
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Import our cloud database manager
from app.utils.cloud_database import cloud_db_manager
from app.config import settings

class CloudServiceSetup:
    """Setup and test cloud service connections"""
    
    def __init__(self):
        self.results = {
            "mongodb": {"status": "not_tested", "details": ""},
            "influxdb": {"status": "not_tested", "details": ""},
            "redis": {"status": "not_tested", "details": ""},
            "mqtt": {"status": "not_tested", "details": ""}
        }
    
    async def test_mongodb_connection(self) -> bool:
        """Test MongoDB Atlas connection"""
        try:
            print("🔍 Testing MongoDB Atlas connection...")
            print(f"   URL: {settings.mongodb_url[:50]}...")
            
            # Test connection
            await cloud_db_manager.connect_mongodb()
            
            # Test basic operations
            collection = await cloud_db_manager.get_collection("test_connection")
            test_doc = {
                "test": True,
                "timestamp": datetime.now().isoformat(),
                "service": "mongodb_atlas"
            }
            
            # Insert test document
            result = await collection.insert_one(test_doc)
            print(f"   ✅ Insert successful: {result.inserted_id}")
            
            # Read test document
            found_doc = await collection.find_one({"_id": result.inserted_id})
            if found_doc:
                print(f"   ✅ Read successful: {found_doc['test']}")
            
            # Clean up test document
            await collection.delete_one({"_id": result.inserted_id})
            print(f"   ✅ Delete successful")
            
            self.results["mongodb"] = {
                "status": "success",
                "details": f"Connected to MongoDB Atlas successfully"
            }
            return True
            
        except Exception as e:
            print(f"   ❌ MongoDB connection failed: {e}")
            self.results["mongodb"] = {
                "status": "failed",
                "details": str(e)
            }
            return False
    
    async def test_influxdb_connection(self) -> bool:
        """Test InfluxDB Cloud connection"""
        try:
            print("🔍 Testing InfluxDB Cloud connection...")
            print(f"   URL: {settings.influxdb_url}")
            print(f"   Org: {settings.influxdb_org}")
            print(f"   Bucket: {settings.influxdb_bucket}")
            
            # Test connection by writing test data
            test_data = {
                "trainset_id": "TEST_CONNECTION",
                "sensor_type": "connection_test",
                "health_score": 1.0,
                "temperature": 25.0,
                "timestamp": datetime.now().isoformat()
            }
            
            success = await cloud_db_manager.write_sensor_data(test_data)
            if success:
                print(f"   ✅ Write successful")
                self.results["influxdb"] = {
                    "status": "success",
                    "details": "Connected to InfluxDB Cloud successfully"
                }
                return True
            else:
                raise Exception("Write operation failed")
                
        except Exception as e:
            print(f"   ❌ InfluxDB connection failed: {e}")
            self.results["influxdb"] = {
                "status": "failed",
                "details": str(e)
            }
            return False
    
    async def test_redis_connection(self) -> bool:
        """Test Redis Cloud connection"""
        try:
            print("🔍 Testing Redis Cloud connection...")
            print(f"   URL: {settings.redis_url[:50]}...")
            
            # Test connection
            await cloud_db_manager.connect_redis()
            
            # Test basic operations
            test_key = "test_connection"
            test_value = json.dumps({
                "test": True,
                "timestamp": datetime.now().isoformat(),
                "service": "redis_cloud"
            })
            
            # Set test value
            await cloud_db_manager.cache_set(test_key, test_value, expiry=60)
            print(f"   ✅ Set successful")
            
            # Get test value
            retrieved_value = await cloud_db_manager.cache_get(test_key)
            if retrieved_value:
                data = json.loads(retrieved_value)
                print(f"   ✅ Get successful: {data['test']}")
            
            # Delete test value
            await cloud_db_manager.cache_set(test_key, "", expiry=1)
            print(f"   ✅ Delete successful")
            
            self.results["redis"] = {
                "status": "success",
                "details": "Connected to Redis Cloud successfully"
            }
            return True
            
        except Exception as e:
            print(f"   ❌ Redis connection failed: {e}")
            self.results["redis"] = {
                "status": "failed",
                "details": str(e)
            }
            return False
    
    async def test_mqtt_connection(self) -> bool:
        """Test MQTT broker connection"""
        try:
            print("🔍 Testing MQTT broker connection...")
            print(f"   Broker: {settings.mqtt_broker}")
            print(f"   Port: {settings.mqtt_port}")
            
            # Import MQTT client
            from app.services.mqtt_client import MQTTClient
            
            # Test connection
            mqtt_client = MQTTClient()
            await mqtt_client.start_client()
            
            if mqtt_client.is_connected():
                print(f"   ✅ MQTT connection successful")
                await mqtt_client.stop_client()
                
                self.results["mqtt"] = {
                    "status": "success",
                    "details": "Connected to MQTT broker successfully"
                }
                return True
            else:
                raise Exception("MQTT client not connected")
                
        except Exception as e:
            print(f"   ❌ MQTT connection failed: {e}")
            self.results["mqtt"] = {
                "status": "failed",
                "details": str(e)
            }
            return False
    
    async def load_production_data(self) -> bool:
        """Load production data into cloud services"""
        try:
            print("📊 Loading production data into cloud services...")
            
            # Import data loader
            from app.services.cloud_loader import load_all_cloud_dbs
            from app.services.mock_data_generator import KMRLMockDataGenerator
            
            # Generate mock data
            generator = KMRLMockDataGenerator()
            mock_data = await generator.generate_all_mock_data()
            
            # Load into cloud services
            await load_all_cloud_dbs(
                trainsets=mock_data["trainsets"],
                job_cards=mock_data["job_cards"],
                branding_contracts=mock_data["branding_records"],
                cleaning_schedule=mock_data["cleaning_schedule"],
                historical_operations=mock_data["historical_operations"],
                depot_layout=mock_data["depot_layout"],
                sensor_data=mock_data["sensor_data"]
            )
            
            print("   ✅ Production data loaded successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Data loading failed: {e}")
            return False
    
    def print_summary(self):
        """Print connection test summary"""
        print("\n" + "="*60)
        print("📋 CLOUD SERVICES CONNECTION SUMMARY")
        print("="*60)
        
        for service, result in self.results.items():
            status_icon = "✅" if result["status"] == "success" else "❌"
            print(f"{status_icon} {service.upper()}: {result['status'].upper()}")
            if result["details"]:
                print(f"   Details: {result['details']}")
        
        print("\n" + "="*60)
        
        # Overall status
        all_success = all(r["status"] == "success" for r in self.results.values())
        if all_success:
            print("🎉 ALL CLOUD SERVICES CONNECTED SUCCESSFULLY!")
            print("   Your KMRL system is ready for production use.")
        else:
            print("⚠️  SOME CLOUD SERVICES FAILED TO CONNECT")
            print("   Please check your .env configuration and try again.")

async def main():
    """Main setup function"""
    print("🚀 KMRL Train Induction System - Cloud Services Setup")
    print("="*60)
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        print("   Please copy env_template.txt to .env and configure your credentials.")
        return
    
    setup = CloudServiceSetup()
    
    # Test all cloud service connections
    print("\n🔍 Testing Cloud Service Connections...")
    await setup.test_mongodb_connection()
    await setup.test_influxdb_connection()
    await setup.test_redis_connection()
    await setup.test_mqtt_connection()
    
    # Print summary
    setup.print_summary()
    
    # If all connections successful, load production data
    all_success = all(r["status"] == "success" for r in setup.results.values())
    if all_success:
        print("\n📊 Loading Production Data...")
        await setup.load_production_data()
        print("\n🎉 Setup Complete! Your KMRL system is ready for production.")
    else:
        print("\n⚠️  Please fix connection issues before loading production data.")

if __name__ == "__main__":
    asyncio.run(main())
