from fastapi import APIRouter, HTTPException
import logging

from app.services.dashboard_service import DashboardService

router = APIRouter()
logger = logging.getLogger(__name__)
dashboard_service = DashboardService()


@router.get("/overview")
async def get_dashboard_overview():
    """Return the fleet overview through the dashboard application service."""
    try:
        return await dashboard_service.overview()
    except Exception as exc:
        logger.exception("Dashboard overview error")
        raise HTTPException(status_code=500, detail=f"Error fetching dashboard data: {exc}")


@router.get("/alerts")
async def get_active_alerts():
    """Return active operational alerts through the dashboard application service."""
    try:
        return await dashboard_service.alerts()
    except Exception as exc:
        logger.exception("Dashboard alerts error")
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {exc}")


@router.get("/performance")
async def get_performance_metrics():
    """Return system performance metrics through the dashboard application service."""
    try:
        return await dashboard_service.performance()
    except Exception as exc:
        logger.exception("Dashboard performance error")
        raise HTTPException(status_code=500, detail=f"Error fetching performance metrics: {exc}")


# Backward-compatible helper names used by older imports/tests.
async def get_sensor_health_summary():
    return {
        "average_health_score": 0.87,
        "sensors_online": 98,
        "sensors_critical": 2,
    }


async def get_sensor_analytics():
    return {
        "temperature_trends": [],
        "vibration_analysis": [],
        "predictive_maintenance_alerts": 3,
    }
