#!/usr/bin/env python3
"""
Load Production Data - Simplified version for working cloud services
"""

import asyncio
import json
from datetime import datetime
from app.utils.cloud_database import cloud_db_manager
from app.services.mock_data_generator import KMRLMockDataGenerator

async def load_production_data():
    """Load production data into working cloud services"""
    print("🚀 Loading KMRL Production Data")
    print("=" * 50)
    
    try:
        # Connect to working services
        print("🔍 Connecting to cloud services...")
        await cloud_db_manager.connect_mongodb()
        await cloud_db_manager.connect_influxdb()
        print("✅ Connected to MongoDB Atlas and InfluxDB Cloud")
        
        # Generate mock data
        print("\n📊 Generating production data...")
        generator = KMRLMockDataGenerator()
        mock_data = await generator.generate_all_mock_data()
        
        print(f"   • Trainsets: {len(mock_data['trainsets'])}")
        print(f"   • Job Cards: {len(mock_data['job_cards'])}")
        print(f"   • Sensor Data: {len(mock_data['sensor_data'])}")
        print(f"   • Cleaning Schedule: {len(mock_data['cleaning_schedule'])}")
        print(f"   • Historical Operations: {len(mock_data['historical_operations'])}")
        
        # Load into MongoDB Atlas
        print("\n📊 Loading data into MongoDB Atlas...")
        
        # Load trainsets
        trainsets_collection = await cloud_db_manager.get_collection("trainsets")
        await trainsets_collection.delete_many({})
        await trainsets_collection.insert_many(mock_data["trainsets"])
        print(f"   ✅ Loaded {len(mock_data['trainsets'])} trainsets")
        
        # Load job cards
        job_cards_collection = await cloud_db_manager.get_collection("job_cards")
        await job_cards_collection.delete_many({})
        await job_cards_collection.insert_many(mock_data["job_cards"])
        print(f"   ✅ Loaded {len(mock_data['job_cards'])} job cards")
        
        # Load branding contracts
        branding_collection = await cloud_db_manager.get_collection("branding_contracts")
        await branding_collection.delete_many({})
        await branding_collection.insert_many(mock_data["branding_records"])
        print(f"   ✅ Loaded {len(mock_data['branding_records'])} branding contracts")
        
        # Load cleaning schedule
        cleaning_collection = await cloud_db_manager.get_collection("cleaning_schedule")
        await cleaning_collection.delete_many({})
        await cleaning_collection.insert_many(mock_data["cleaning_schedule"])
        print(f"   ✅ Loaded {len(mock_data['cleaning_schedule'])} cleaning slots")
        
        # Load historical operations
        history_collection = await cloud_db_manager.get_collection("historical_operations")
        await history_collection.delete_many({})
        await history_collection.insert_many(mock_data["historical_operations"])
        print(f"   ✅ Loaded {len(mock_data['historical_operations'])} historical records")
        
        # Load depot layout
        depot_collection = await cloud_db_manager.get_collection("depot_layout")
        await depot_collection.delete_many({})
        await depot_collection.insert_one(mock_data["depot_layout"])
        print(f"   ✅ Loaded depot layout")
        
        # Load sensor data into InfluxDB Cloud
        print("\n📈 Loading sensor data into InfluxDB Cloud...")
        sensor_count = 0
        for sensor_reading in mock_data["sensor_data"]:
            success = await cloud_db_manager.write_sensor_data(sensor_reading)
            if success:
                sensor_count += 1
        
        print(f"   ✅ Loaded {sensor_count} sensor readings")
        
        print("\n🎉 Production data loaded successfully!")
        print("\n📋 Summary:")
        print(f"   • MongoDB Atlas: {len(mock_data['trainsets'])} trainsets + other collections")
        print(f"   • InfluxDB Cloud: {sensor_count} sensor readings")
        print(f"   • Total Records: {len(mock_data['trainsets']) + len(mock_data['job_cards']) + len(mock_data['branding_records']) + len(mock_data['cleaning_schedule']) + len(mock_data['historical_operations']) + sensor_count}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading production data: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(load_production_data())
