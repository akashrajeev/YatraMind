from fastapi import APIRouter, HTTPException
import logging

from app.repositories.mongo_dashboard import MongoDashboardRepository
from app.services.dashboard_service import DashboardService

router = APIRouter()
logger = logging.getLogger(__name__)
dashboard_service = DashboardService(MongoDashboardRepository())


@router.get("/overview")
async def get_dashboard_overview():
    try:
        return await dashboard_service.overview()
    except Exception as exc:
        logger.error("Dashboard overview error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {exc}")


@router.get("/alerts")
async def get_active_alerts():
    try:
        return await dashboard_service.alerts()
    except Exception as exc:
        logger.error("Alerts fetch error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {exc}")


@router.get("/performance")
async def get_performance_metrics():
    try:
        return await dashboard_service.performance()
    except Exception as exc:
        logger.error("Performance metrics error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Error fetching performance metrics: {exc}")
