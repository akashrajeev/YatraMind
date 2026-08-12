from datetime import datetime
import logging
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from app.api import (
    trainsets,
    optimization,
    dashboard,
    ingestion,
    assignments,
    reports,
    auth,
    simulation,
    users,
    notifications,
    multi_depot_simulation,
)
from app.utils.cloud_database import cloud_db_manager
from app.config import settings
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.celery_app import celery_app
from app.tasks import nightly_run_optimization, ingestion_refresh_all, train_model
from app.models.user import User, UserRole
from app.security import require_role

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KMRL Train Induction System",
    description="AI/ML-driven decision support platform for Kochi Metro train induction planning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

sio = None
socket_manager = None
app.add_middleware(GZipMiddleware, minimum_size=1024)

async def _require_api_key(x_api_key: str | None = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

cors_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)

if _HAS_PROM:
    Instrumentator().instrument(app).expose(app, include_in_schema=False)

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(trainsets.router, prefix="/api/v1/trainsets", tags=["Trainsets"])
app.include_router(optimization.router, prefix="/api/v1/optimization", tags=["Optimization"])
app.include_router(dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
app.include_router(ingestion.router, prefix="/api/v1/ingestion", tags=["Ingestion"])
app.include_router(assignments.router, prefix="/api/v1/assignments", tags=["Assignments"])
app.include_router(reports.router, prefix="/api/v1/reports", tags=["Reports"])
app.include_router(simulation.router, prefix="/api/v1/simulation", tags=["Simulation"])
app.include_router(notifications.router, prefix="/api/v1/notifications", tags=["Notifications"])
app.include_router(multi_depot_simulation.router, prefix="/api/v1/multi-depot", tags=["Multi-Depot Simulation"])

if _HAS_SOCKETIO and sio:
    @sio.event
    async def connect(sid, environ):
        logger.info("Client connected: %s", sid)
        await sio.emit("status", {"message": "Connected to KMRL Operations Dashboard"}, room=sid)

    @sio.event
    async def disconnect(sid):
        logger.info("Client disconnected: %s", sid)

    @sio.event
    async def join_room(sid, data):
        room = data.get("room", "general")
        sio.enter_room(sid, room)
        await sio.emit("joined_room", {"room": room}, room=sid)

    @sio.event
    async def leave_room(sid, data):
        room = data.get("room", "general")
        sio.leave_room(sid, room)
        await sio.emit("left_room", {"room": room}, room=sid)


def get_socket():
    """Get Socket.IO instance if available, otherwise None."""
    return sio if _HAS_SOCKETIO else None

scheduler = None

@app.on_event("startup")
async def startup_event():
    """Initialize cloud database connections and periodic ingestion jobs."""
    try:
        logger.info("Starting KMRL Train Induction System...")
        if settings.mongodb_url:
            await cloud_db_manager.connect_mongodb()
        if settings.influxdb_url:
            await cloud_db_manager.connect_influxdb()

        global scheduler
        scheduler = AsyncIOScheduler()

        from app.services.data_ingestion import DataIngestionService
        svc = DataIngestionService()
        scheduler.add_job(svc._ingest_maximo_data, "interval", minutes=15, id="maximo_ingest", max_instances=1, coalesce=True)

        import os
        sheet_url = os.environ.get("CLEANING_SHEET_URL")
        if sheet_url:
            scheduler.add_job(lambda: svc.ingest_cleaning_google_sheet(sheet_url), "interval", minutes=30, id="cleaning_ingest", max_instances=1, coalesce=True)

        scheduler.add_job(lambda: celery_app.send_task("optimization.nightly_run"), "cron", hour=23, minute=59, id="nightly_opt")
        scheduler.start()

        try:
            assignments_col = await cloud_db_manager.get_collection("assignments")
            await assignments_col.create_index("status")
            await assignments_col.create_index("created_at")
            trainsets_col = await cloud_db_manager.get_collection("trainsets")
            await trainsets_col.create_index("trainset_id", unique=True)
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error("Failed to create MongoDB indexes: %s", e)
    except Exception as e:
        logger.error("Startup failed: %s", e)

@app.on_event("shutdown")
async def shutdown_event():
    """Close cloud database connections on shutdown."""
    try:
        if scheduler:
            scheduler.shutdown(wait=False)
        await cloud_db_manager.close_all()
        logger.info("Cloud database connections closed")
    except Exception as e:
        logger.error("Shutdown error: %s", e)

@app.get("/")
async def root():
    return {
        "message": "KMRL Train Induction System API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Return dependency connectivity status without exposing exception details."""
    try:
        checks = {}
        if settings.mongodb_url:
            await cloud_db_manager.connect_mongodb()
            checks["mongodb"] = "connected"
        else:
            checks["mongodb"] = "not_configured"
        if settings.influxdb_url:
            await cloud_db_manager.connect_influxdb()
            checks["influxdb"] = "connected"
        else:
            checks["influxdb"] = "not_configured"

        healthy = all(value in {"connected", "not_configured"} for value in checks.values())
        return {
            "status": "healthy" if healthy else "unhealthy",
            "dependencies": checks,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {
            "status": "unhealthy",
            "dependencies": {"mongodb": "unavailable", "influxdb": "unavailable"},
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

@app.post("/tasks/optimization/run")
async def trigger_optimization_task(_user: User = Depends(require_role(UserRole.ADMIN))):
    """Enqueue nightly optimization task to Celery."""
    res = celery_app.send_task("optimization.nightly_run")
    return {"status": "queued", "task_id": res.id}

@app.post("/tasks/ingestion/refresh")
async def trigger_ingestion_refresh(_user: User = Depends(require_role(UserRole.ADMIN))):
    """Enqueue ingestion refresh task."""
    res = celery_app.send_task("ingestion.refresh_all")
    return {"status": "queued", "task_id": res.id}

@app.post("/tasks/ml/train")
async def trigger_ml_training(_user: User = Depends(require_role(UserRole.ADMIN))):
    """Enqueue ML model training task."""
    res = celery_app.send_task("ml.train_models")
    return {"status": "queued", "task_id": res.id}
