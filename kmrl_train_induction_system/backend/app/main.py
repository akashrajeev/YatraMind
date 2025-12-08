# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.api import trainsets, optimization, dashboard, ingestion, assignments, reports, auth, simulation, users
from app.utils.cloud_database import cloud_db_manager
from app.config import settings
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime
import logging
from app.celery_app import celery_app
from app.tasks import nightly_run_optimization, ingestion_refresh_all, train_model
from fastapi import Header, HTTPException
try:
    from fastapi_socketio import SocketManager
    import socketio
    _HAS_SOCKETIO = True
except ImportError:
    _HAS_SOCKETIO = False
    SocketManager = None
    socketio = None
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

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

# Socket.IO for real-time updates (OPTIONAL - currently disabled)
# To enable: Set sio = socketio.AsyncServer() and configure SocketManager
# Note: Frontend may use polling as fallback if Socket.IO is unavailable
sio = None
socket_manager = None

# GZip for responses
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Simple API key auth & rate-limit placeholder
async def _require_api_key(x_api_key: str | None = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics (must be added before startup)
if _HAS_PROM:
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

# Include API routers
app.include_router(trainsets.router, prefix="/api/trainsets", tags=["Trainsets"])
app.include_router(optimization.router, prefix="/api/optimization", tags=["Optimization"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"])
app.include_router(ingestion.router, prefix="/api/ingestion", tags=["Data Ingestion"])
app.include_router(assignments.router, prefix="/api/v1/assignments", tags=["Assignments"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["User Management"])
app.include_router(simulation.router, prefix="/api/simulation", tags=["Simulation"])

scheduler: AsyncIOScheduler | None = None

# Socket.IO event handlers (OPTIONAL - only active if sio is configured)
# These handlers are defined but won't execute unless Socket.IO is enabled above
if _HAS_SOCKETIO and sio:
    @sio.event
    async def connect(sid, environ):
        """Handle client connection"""
        logger.info(f"Client connected: {sid}")
        await sio.emit("status", {"message": "Connected to KMRL Operations Dashboard"}, room=sid)

    @sio.event
    async def disconnect(sid):
        """Handle client disconnection"""
        logger.info(f"Client disconnected: {sid}")

    @sio.event
    async def join_room(sid, data):
        """Handle client joining a room (e.g., user-specific room)"""
        room = data.get("room", "general")
        sio.enter_room(sid, room)
        await sio.emit("joined_room", {"room": room}, room=sid)

    @sio.event
    async def leave_room(sid, data):
        """Handle client leaving a room"""
        room = data.get("room", "general")
        sio.leave_room(sid, room)
        await sio.emit("left_room", {"room": room}, room=sid)

# Global socket instance for use in other modules (returns None if Socket.IO disabled)
def get_socket():
    """Get Socket.IO instance if available, None otherwise"""
    return sio if _HAS_SOCKETIO else None

@app.on_event("startup")
async def startup_event():
    """Initialize cloud database connections on startup"""
    try:
        logger.info("Starting KMRL Train Induction System...")
        # Connect to required databases (MongoDB and InfluxDB)
        # Note: Redis and MQTT are optional components - configure if needed
        await cloud_db_manager.connect_mongodb()
        await cloud_db_manager.connect_influxdb()
        logger.info("MongoDB and InfluxDB connections established")

        # Start APScheduler for periodic ingestions
        global scheduler
        scheduler = AsyncIOScheduler()

        from app.services.data_ingestion import DataIngestionService
        svc = DataIngestionService()

        # Maximo every 15 minutes
        scheduler.add_job(svc._ingest_maximo_data, "interval", minutes=15, id="maximo_ingest", max_instances=1, coalesce=True)
        # Cleaning schedule every 30 minutes (requires CLEANING_SHEET_URL in .env or skip)
        import os
        sheet_url = os.environ.get("CLEANING_SHEET_URL")
        if sheet_url:
            scheduler.add_job(lambda: svc.ingest_cleaning_google_sheet(sheet_url), "interval", minutes=30, id="cleaning_ingest", max_instances=1, coalesce=True)

        # Nightly optimization at 4:30 AM IST (Asia/Kolkata) -> convert to server TZ by cron
        scheduler.add_job(lambda: celery_app.send_task("optimization.nightly_run"), "cron", hour=23, minute=59, id="nightly_opt")
        scheduler.start()

        # Create MongoDB Indexes
        try:
            assignments_col = await cloud_db_manager.get_collection("assignments")
            await assignments_col.create_index("status")
            await assignments_col.create_index("created_at")
            
            trainsets_col = await cloud_db_manager.get_collection("trainsets")
            await trainsets_col.create_index("trainset_id", unique=True)
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")
    except Exception as e:
        logger.error(f"Startup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Close cloud database connections on shutdown"""
    try:
        if scheduler:
            scheduler.shutdown(wait=False)
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

@app.post("/tasks/optimization/run")
async def trigger_optimization_task():
    """Enqueue nightly optimization task to Celery."""
    res = celery_app.send_task("optimization.nightly_run")
    return {"status": "queued", "task_id": res.id}

@app.post("/tasks/ingestion/refresh")
async def trigger_ingestion_refresh():
    """Enqueue ingestion refresh task."""
    res = celery_app.send_task("ingestion.refresh_all")
    return {"status": "queued", "task_id": res.id}

@app.post("/tasks/ml/train")
async def trigger_model_training():
    """Enqueue model training task."""
    res = celery_app.send_task("ml.train_model")
    return {"status": "queued", "task_id": res.id}

@app.get("/tasks/status/{task_id}")
async def get_task_status(task_id: str):
    """Return Celery task status/result."""
    try:
        async_result = celery_app.AsyncResult(task_id)
        payload = {
            "task_id": task_id,
            "state": async_result.state,
        }
        try:
            if async_result.ready():
                payload["result"] = async_result.result
        except Exception:
            # Some backends or states may not allow result access
            payload["result"] = None
        return payload
    except Exception as e:
        return {"task_id": task_id, "state": "UNKNOWN", "error": str(e)}
