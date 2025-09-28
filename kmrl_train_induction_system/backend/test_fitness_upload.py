#!/usr/bin/env python3
"""Test fitness certificate upload to debug the issue"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.data_ingestion import DataIngestionService
import asyncio
import io

async def test_fitness_upload():
    """Test the fitness certificate upload process"""
    
    # Create test CSV content
    csv_content = """trainset_id,dept,certificate,status,valid_from,valid_to,issued_by,certificate_id
T-001,ROLLING_STOCK,ROLLING_STOCK,VALID,2024-01-01,2024-12-31,KMRL_CERT_AUTH,RC-2024-001
T-001,SIGNALLING,SIGNALLING,VALID,2024-01-15,2024-12-31,KMRL_SIG_AUTH,SC-2024-001
T-001,TELECOM,TELECOM,VALID,2024-02-01,2024-12-31,KMRL_TEL_AUTH,TC-2024-001
T-002,ROLLING_STOCK,ROLLING_STOCK,VALID,2024-01-01,2024-12-31,KMRL_CERT_AUTH,RC-2024-002
T-002,SIGNALLING,SIGNALLING,EXPIRED,2023-12-31,2023-12-31,KMRL_SIG_AUTH,SC-2023-002"""
    
    print("Testing fitness certificate upload...")
    print("CSV Content:")
    print(csv_content)
    print("\n" + "="*50)
    
    try:
        # Initialize the service
        service = DataIngestionService()
        
        # Test the _read_tabular method
        print("Testing _read_tabular method...")
        df = service._read_tabular(csv_content.encode('utf-8'), 'test_fitness.csv')
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"DataFrame columns (lowercase): {list(df.columns.str.lower())}")
        
        # Check required columns
        required = {"trainset_id", "dept", "certificate", "status", "valid_from", "valid_to"}
        actual_columns = set(df.columns.str.lower())
        print(f"Required columns: {required}")
        print(f"Actual columns: {actual_columns}")
        print(f"Missing columns: {required - actual_columns}")
        print(f"Extra columns: {actual_columns - required}")
        print(f"All required present: {required.issubset(actual_columns)}")
        
        if required.issubset(actual_columns):
            print("✅ All required columns are present!")
            
            # Try the full ingestion process
            print("\nTesting full ingestion process...")
            result = await service.ingest_fitness_file(csv_content.encode('utf-8'), 'test_fitness.csv')
            print(f"✅ Upload successful! Records processed: {result.get('count', 0)}")
        else:
            print("❌ Missing required columns!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fitness_upload())
