# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import trainsets, optimization, dashboard, ingestion
from app.utils.cloud_database import cloud_db_manager
from app.config import settings
import logging

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(trainsets.router, prefix="/api/trainsets", tags=["Trainsets"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(ingestion.router, prefix="/api/ingestion", tags=["Data Ingestion"])

@app.on_event("startup")
async def startup_event():
    """Initialize cloud database connections on startup"""
    try:
        logger.info("Starting KMRL Train Induction System...")
        # Connect only to MongoDB and InfluxDB for now (skip Redis/MQTT)
        await cloud_db_manager.connect_mongodb()
        await cloud_db_manager.connect_influxdb()
        logger.info("MongoDB and InfluxDB connections established")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close cloud database connections on shutdown"""
    try:
        await cloud_db_manager.close_all()
        logger.info("Cloud database connections closed")
    except Exception as e:
        logger.error(f"Shutdown error: {e}")

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "KMRL Train Induction System API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check MongoDB and InfluxDB connectivity only
        await cloud_db_manager.connect_mongodb()
        await cloud_db_manager.connect_influxdb()
        
        return {
            "status": "healthy",
            "database": "mongo+influx connected",
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }
