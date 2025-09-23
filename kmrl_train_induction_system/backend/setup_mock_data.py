"""Setup mock data locally and optionally push to real cloud services using env creds."""
import argparse
import asyncio
import json
from typing import Any, Dict

from app.services.mock_data_generator import load_kmrl_mock_data
from app.services.cloud_loader import load_all_cloud_dbs
from app.config import settings
from app.utils.cloud_database import cloud_db_manager

def _summarize(data: Dict[str, Any]) -> None:
    print("\nMock Data Summary:")
    print(f"   • Trainsets: {len(data['trainsets'])}")
    print(f"   • Job Cards: {len(data['job_cards'])}")
    print(f"   • Sensor Readings: {len(data['sensor_data'])}")
    print(f"   • Cleaning Slots: {len(data['cleaning_schedule'])}")
    print(f"   • Historical Records: {len(data['historical_operations'])}")
    dep = data.get('depot_layout', {})
    print(f"   • Depot Features: {len(dep.get('features', [])) if isinstance(dep, dict) else len(dep)}")

async def main():
    parser = argparse.ArgumentParser(description="KMRL Mock Data Setup")
    parser.add_argument("--to-cloud", action="store_true", help="Write data to MongoDB/InfluxDB/Redis using env credentials")
    args = parser.parse_args()

    print("KMRL Train Induction System - Mock Data Setup")
    print("=" * 60)

    try:
        mock_data = await load_kmrl_mock_data()
        _summarize(mock_data)

        if args.to_cloud:
            print("\nPushing datasets to cloud services using environment credentials...")
            await load_all_cloud_dbs(
                mongodb_uri=settings.mongodb_url,
                mongodb_db=settings.database_name,
                influx_url=settings.influxdb_url,
                influx_token=settings.influxdb_token,
                influx_org=settings.influxdb_org,
                influx_bucket=settings.influxdb_bucket,
                redis_url=settings.redis_url,
                trainsets=mock_data["trainsets"],
                job_cards=mock_data["job_cards"],
                branding_contracts=mock_data["branding_records"],
                cleaning_schedule=mock_data["cleaning_schedule"],
                historical_operations=mock_data["historical_operations"],
                depot_layout=mock_data["depot_layout"],
                sensor_data=mock_data["sensor_data"],
            )
            print("Cloud load completed.")

        print("\nSetup completed successfully!")
        print("\nNext steps:")
        print("1. Run API: set PYTHONPATH=backend and uvicorn app.main:app --reload")
        print("2. Explore http://localhost:8000/docs")

    except Exception as e:
        print(f"Setup failed: {e}")
    finally:
        await cloud_db_manager.close_all()

if __name__ == "__main__":
    asyncio.run(main())
