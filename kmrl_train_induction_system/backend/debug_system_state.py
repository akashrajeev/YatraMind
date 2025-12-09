import asyncio
from app.utils.cloud_database import cloud_db_manager
from datetime import datetime
import pprint
import sys

# Windows requires this event loop policy
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def diagnose_system_state():
    try:
        await cloud_db_manager.connect_mongodb()
        
        print("\n=== 1. LATEST N8N INGESTION ===")
        n8n_col = await cloud_db_manager.get_collection("n8n_ingested_data")
        n8n_doc = await n8n_col.find_one(sort=[("ingested_at", -1)])
        
        if not n8n_doc:
            print("No N8N data found.")
        else:
            print(f"ID: {n8n_doc.get('_id')}")
            print(f"Time: {n8n_doc.get('ingested_at')}")
            print(f"Processed: {n8n_doc.get('processed')}")
            print(f"Updates Count: {n8n_doc.get('updates_processed')}")
            print(f"Errors: {n8n_doc.get('errors')}")
            print(f"Skipped Updates: {n8n_doc.get('skipped_updates')}")
            data_sample = n8n_doc.get("data")
            if isinstance(data_sample, list) and len(data_sample) > 0:
                 print(f"Data Sample (Item 0): {str(data_sample[0])[:200]}...")
            else:
                 print(f"Data Sample: {str(data_sample)[:200]}...")

        print("\n=== 2. TRAINSET STATE (T-009) ===")
        # T-009 was seen in the previous log
        ts_col = await cloud_db_manager.get_collection("trainsets")
        ts_doc = await ts_col.find_one({"trainset_id": "T-009"})
        if ts_doc:
            print(f"Trainset: {ts_doc.get('trainset_id')}")
            print(f"Status: {ts_doc.get('status')}")
            print(f"Job Cards: {ts_doc.get('job_cards')}")
            print(f"Last Updated Sources: {ts_doc.get('last_updated_sources')}")
        else:
            print("T-009 not found.")

        print("\n=== 3. OPTIMIZATION STATUS ===")
        opt_col = await cloud_db_manager.get_collection("optimization_results")
        # Check if there are any results and when they were created
        # Assuming there is a field like 'created_at' or 'optimization_date' or checking the latest object
        # Optimization results structure might vary, let's just grab one
        opt_doc = await opt_col.find_one(sort=[("_id", -1)]) # relying on ObjectId for time
        
        if opt_doc:
             # ObjectId generation time
             print(f"Latest Optimization Result ID: {opt_doc.get('_id')}")
             print(f"Optimization Timestamp (approx): {opt_doc.get('_id').generation_time}")
             print(f"Decisions Count: {len(opt_doc.get('decisions', []))}")
        else:
             print("No optimization results found.")

        print("==============================\n")
        
    except Exception as e:
        print(f"Error reading DB: {e}")
    finally:
        await cloud_db_manager.close_all()

if __name__ == "__main__":
    asyncio.run(diagnose_system_state())
