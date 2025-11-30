import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Mock settings
os.environ["MAXIMO_BASE_URL"] = "http://mock-maximo"
os.environ["MAXIMO_API_KEY"] = "mock-key"

async def verify():
    print("Verifying modules...")
    
    try:
        print("Importing rule_engine...")
        from app.services.rule_engine import DurableRulesEngine
        re = DurableRulesEngine()
        print("rule_engine instantiated.")
        
        print("Importing optimizer...")
        from app.services.optimizer import TrainInductionOptimizer
        opt = TrainInductionOptimizer()
        print("optimizer instantiated.")
        
        print("Importing stabling_optimizer...")
        from app.services.stabling_optimizer import StablingGeometryOptimizer
        so = StablingGeometryOptimizer()
        print("stabling_optimizer instantiated.")
        
        print("Importing data_ingestion...")
        from app.services.data_ingestion import DataIngestionService
        di = DataIngestionService()
        print("data_ingestion instantiated.")
        
        print("Importing api.optimization...")
        from app.api import optimization
        print("api.optimization imported.")

        print("ALL MODULES VERIFIED SUCCESSFULLY.")
        
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(verify())
