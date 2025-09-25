# app/vercel_main.py - Vercel-optimized version
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.api import trainsets, optimization, dashboard, ingestion
from app.utils.cloud_database import cloud_db_manager
from app.config import settings
from datetime import datetime
import logging
import httpx
import os
from fastapi import Header, HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="KMRL Train Induction System",
    description="AI/ML-driven decision support platform for Kochi Metro train induction planning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# GZip for responses
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Simple API key auth
async def _require_api_key(x_api_key: str | None = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(trainsets.router, prefix="/api/trainsets", tags=["Trainsets"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(ingestion.router, prefix="/api/ingestion", tags=["Data Ingestion"])

# External worker service URL
WORKER_SERVICE_URL = os.getenv("WORKER_SERVICE_URL", "https://your-worker-service.railway.app")

@app.on_event("startup")
async def startup_event():
    """Initialize database connections on startup"""
    try:
        logger.info("Starting KMRL Train Induction System on Vercel...")
        # Connect only to MongoDB and InfluxDB
        await cloud_db_manager.connect_mongodb()
        await cloud_db_manager.connect_influxdb()
        logger.info("MongoDB and InfluxDB connections established")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    try:
        await cloud_db_manager.close_all()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "KMRL Train Induction System API",
        "version": "1.0.0",
        "status": "operational",
        "platform": "vercel",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check MongoDB and InfluxDB connectivity
        await cloud_db_manager.connect_mongodb()
        await cloud_db_manager.connect_influxdb()
        
        return {
            "status": "healthy",
            "database": "mongo+influx connected",
            "platform": "vercel",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "platform": "vercel"
        }

# Vercel-compatible task endpoints (call external worker service)
@app.post("/tasks/optimization/run")
async def trigger_optimization_task():
    """Trigger optimization task via external worker service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{WORKER_SERVICE_URL}/tasks/optimization/run",
                headers={"X-API-Key": os.getenv("WORKER_API_KEY", "")},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"Worker service error: {response.status_code}",
                    "task_id": None
                }
    except Exception as e:
        logger.error(f"Failed to call worker service: {e}")
        return {
            "status": "error",
            "message": "Worker service unavailable",
            "task_id": None
        }

@app.post("/tasks/ingestion/refresh")
async def trigger_ingestion_refresh():
    """Trigger ingestion refresh via external worker service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{WORKER_SERVICE_URL}/tasks/ingestion/refresh",
                headers={"X-API-Key": os.getenv("WORKER_API_KEY", "")},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"Worker service error: {response.status_code}",
                    "task_id": None
                }
    except Exception as e:
        logger.error(f"Failed to call worker service: {e}")
        return {
            "status": "error",
            "message": "Worker service unavailable",
            "task_id": None
        }

@app.post("/tasks/ml/train")
async def trigger_model_training():
    """Trigger model training via external worker service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{WORKER_SERVICE_URL}/tasks/ml/train",
                headers={"X-API-Key": os.getenv("WORKER_API_KEY", "")},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "message": f"Worker service error: {response.status_code}",
                    "task_id": None
                }
    except Exception as e:
        logger.error(f"Failed to call worker service: {e}")
        return {
            "status": "error",
            "message": "Worker service unavailable",
            "task_id": None
        }

@app.get("/tasks/status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status from external worker service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{WORKER_SERVICE_URL}/tasks/status/{task_id}",
                headers={"X-API-Key": os.getenv("WORKER_API_KEY", "")},
                timeout=10.0
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "task_id": task_id,
                    "state": "UNKNOWN",
                    "error": f"Worker service error: {response.status_code}"
                }
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        return {
            "task_id": task_id,
            "state": "UNKNOWN",
            "error": "Worker service unavailable"
        }
