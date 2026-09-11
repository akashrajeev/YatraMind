from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from typing import Optional
from datetime import datetime, timedelta, timezone
import logging
import io

from app.models.assignment import AssignmentStatus
from app.security import require_api_key
from app.services.repository_report_generator import RepositoryReportGenerator
from app.services.assignment_service import AssignmentService
from app.services.audit_service import audit_service

logger = logging.getLogger(__name__)
router = APIRouter()
report_generator = RepositoryReportGenerator()
assignment_service = AssignmentService()


@router.get("/daily-briefing")
async def generate_daily_briefing(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    _auth=Depends(require_api_key),
):
    """Generate daily briefing PDF report."""
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now(timezone.utc)
        pdf_content = await report_generator.generate_daily_briefing(target_date)
        return StreamingResponse(io.BytesIO(pdf_content), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=daily-briefing-{target_date:%Y-%m-%d}.pdf"})
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as exc:
        logger.error("Error generating daily briefing: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate daily briefing: {exc}")


@router.get("/assignments")
async def export_assignments(
    format: str = Query("csv", pattern="^(csv|pdf)$"),
    status: Optional[AssignmentStatus] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    _auth=Depends(require_api_key),
):
    """Export assignments in CSV or PDF format."""
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        assignments = await assignment_service.list_assignments(status=status, created_after=start_dt, created_before=end_dt, limit=10000, offset=0)
        if format == "csv":
            content = await report_generator.generate_assignments_csv(assignments)
            return StreamingResponse(io.StringIO(content), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=assignments-{datetime.now(timezone.utc):%Y-%m-%d}.csv"})
        content = await report_generator.generate_assignments_pdf(assignments)
        return StreamingResponse(io.BytesIO(content), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=assignments-{datetime.now(timezone.utc):%Y-%m-%d}.pdf"})
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as exc:
        logger.error("Error exporting assignments: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to export assignments: {exc}")


@router.get("/audit-logs")
async def export_audit_logs(start_date: Optional[str] = None, end_date: Optional[str] = None, user_id: Optional[str] = None, action: Optional[str] = None, _auth=Depends(require_api_key)):
    """Export audit logs in CSV format."""
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        audit_logs = await audit_service.list_logs(start=start_dt, end=end_dt, user_id=user_id, action=action, limit=10000)
        content = await report_generator.generate_audit_logs_csv(audit_logs)
        return StreamingResponse(io.StringIO(content), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=audit-logs-{datetime.now(timezone.utc):%Y-%m-%d}.csv"})
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as exc:
        logger.error("Error exporting audit logs: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to export audit logs: {exc}")


@router.get("/fleet-status")
async def generate_fleet_status_report(format: str = Query("pdf", pattern="^(pdf|csv)$"), _auth=Depends(require_api_key)):
    """Generate fleet status report."""
    try:
        if format == "pdf":
            content = await report_generator.generate_fleet_status_pdf()
            return StreamingResponse(io.BytesIO(content), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=fleet-status-{datetime.now(timezone.utc):%Y-%m-%d}.pdf"})
        content = await report_generator.generate_fleet_status_csv()
        return StreamingResponse(io.StringIO(content), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename=fleet-status-{datetime.now(timezone.utc):%Y-%m-%d}.csv"})
    except Exception as exc:
        logger.error("Error generating fleet status report: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate fleet status report: {exc}")


@router.get("/performance-analysis")
async def generate_performance_analysis(days: int = Query(30, ge=1, le=365), _auth=Depends(require_api_key)):
    """Generate performance analysis report."""
    try:
        end_date = datetime.now(timezone.utc)
        content = await report_generator.generate_performance_analysis_pdf(end_date - timedelta(days=days), end_date)
        return StreamingResponse(io.BytesIO(content), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=performance-analysis-{days}days.pdf"})
    except Exception as exc:
        logger.error("Error generating performance analysis: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate performance analysis: {exc}")


@router.get("/compliance-report")
async def generate_compliance_report(start_date: Optional[str] = None, end_date: Optional[str] = None, _auth=Depends(require_api_key)):
    """Generate compliance report."""
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now(timezone.utc) - timedelta(days=30)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now(timezone.utc)
        content = await report_generator.generate_compliance_report(start_dt, end_dt)
        return StreamingResponse(io.BytesIO(content), media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=compliance-report-{start_dt:%Y-%m-%d}-to-{end_dt:%Y-%m-%d}.pdf"})
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as exc:
        logger.error("Error generating compliance report: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate compliance report: {exc}")
