# backend/app/api/reports.py
from fastapi import APIRouter, HTTPException, Depends, Query, Response
from fastapi.responses import StreamingResponse
from typing import Optional
from datetime import datetime, timedelta
import logging
import io
import csv
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import json

from app.models.assignment import Assignment, AssignmentStatus
from app.models.audit import AuditLog
from app.utils.cloud_database import cloud_db_manager
from app.security import require_api_key, get_current_user
from app.services.report_generator import ReportGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize report generator
report_generator = ReportGenerator()


@router.get("/daily-briefing")
async def generate_daily_briefing(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Generate daily briefing PDF report"""
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d") if date else datetime.now()
        
        # Generate PDF content
        pdf_content = await report_generator.generate_daily_briefing(target_date)
        
        # Return PDF as streaming response
        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=daily-briefing-{target_date.strftime('%Y-%m-%d')}.pdf"}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error generating daily briefing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate daily briefing: {str(e)}")


@router.get("/assignments")
async def export_assignments(
    format: str = Query("csv", regex="^(csv|pdf)$"),
    status: Optional[AssignmentStatus] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Export assignments in CSV or PDF format"""
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        
        # Build filter
        filter_query = {}
        if status:
            filter_query["status"] = status.value
        if start_dt:
            filter_query["created_at"] = {"$gte": start_dt}
        if end_dt:
            filter_query["created_at"] = {"$lte": end_dt}
        
        # Get assignments
        collection = await cloud_db_manager.get_collection("assignments")
        cursor = collection.find(filter_query).sort("created_at", -1)
        assignments = []
        
        async for doc in cursor:
            doc.pop('_id', None)
            assignments.append(Assignment(**doc))
        
        if format == "csv":
            # Generate CSV
            csv_content = await report_generator.generate_assignments_csv(assignments)
            return StreamingResponse(
                io.StringIO(csv_content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=assignments-{datetime.now().strftime('%Y-%m-%d')}.csv"}
            )
        else:
            # Generate PDF
            pdf_content = await report_generator.generate_assignments_pdf(assignments)
            return StreamingResponse(
                io.BytesIO(pdf_content),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=assignments-{datetime.now().strftime('%Y-%m-%d')}.pdf"}
            )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error exporting assignments: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export assignments: {str(e)}")


@router.get("/audit-logs")
async def export_audit_logs(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Export audit logs in CSV format"""
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
        
        # Build filter
        filter_query = {}
        if start_dt:
            filter_query["timestamp"] = {"$gte": start_dt}
        if end_dt:
            filter_query["timestamp"] = {"$lte": end_dt}
        if user_id:
            filter_query["user_id"] = user_id
        if action:
            filter_query["action"] = action
        
        # Get audit logs
        collection = await cloud_db_manager.get_collection("audit_logs")
        cursor = collection.find(filter_query).sort("timestamp", -1).limit(10000)  # Limit for performance
        audit_logs = []
        
        async for doc in cursor:
            doc.pop('_id', None)
            audit_logs.append(AuditLog(**doc))
        
        # Generate CSV
        csv_content = await report_generator.generate_audit_logs_csv(audit_logs)
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=audit-logs-{datetime.now().strftime('%Y-%m-%d')}.csv"}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error exporting audit logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export audit logs: {str(e)}")


@router.get("/fleet-status")
async def generate_fleet_status_report(
    format: str = Query("pdf", regex="^(pdf|csv)$"),
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Generate fleet status report"""
    try:
        if format == "pdf":
            pdf_content = await report_generator.generate_fleet_status_pdf()
            return StreamingResponse(
                io.BytesIO(pdf_content),
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=fleet-status-{datetime.now().strftime('%Y-%m-%d')}.pdf"}
            )
        else:
            csv_content = await report_generator.generate_fleet_status_csv()
            return StreamingResponse(
                io.StringIO(csv_content),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=fleet-status-{datetime.now().strftime('%Y-%m-%d')}.csv"}
            )
        
    except Exception as e:
        logger.error(f"Error generating fleet status report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate fleet status report: {str(e)}")


@router.get("/performance-analysis")
async def generate_performance_analysis(
    days: int = Query(30, ge=1, le=365),
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Generate performance analysis report"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        pdf_content = await report_generator.generate_performance_analysis_pdf(start_date, end_date)
        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=performance-analysis-{days}days.pdf"}
        )
        
    except Exception as e:
        logger.error(f"Error generating performance analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance analysis: {str(e)}")


@router.get("/compliance-report")
async def generate_compliance_report(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    current_user=Depends(get_current_user),
    _auth=Depends(require_api_key)
):
    """Generate compliance report"""
    try:
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime.now() - timedelta(days=30)
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        
        pdf_content = await report_generator.generate_compliance_report(start_dt, end_dt)
        return StreamingResponse(
            io.BytesIO(pdf_content),
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=compliance-report-{start_dt.strftime('%Y-%m-%d')}-to-{end_dt.strftime('%Y-%m-%d')}.pdf"}
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate compliance report: {str(e)}")
