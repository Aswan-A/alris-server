from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.database.models import Report, ReportUpload, User
from app.config.db import get_db
from app.middleware.auth import get_current_user

router = APIRouter()

@router.get("/")
async def get_user_reports(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    reports = db.query(Report).filter(Report.user_id == current_user.id).offset(skip).limit(limit).all()
    return reports

@router.get("/{report_id}")
async def get_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    report = db.query(Report).filter(Report.id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Check if the report belongs to the current user
    if report.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view this report")
    
    # Get uploads for this report
    uploads = db.query(ReportUpload).filter(ReportUpload.report_id == report_id).all()
    
    return {
        "report": report,
        "uploads": uploads
    }