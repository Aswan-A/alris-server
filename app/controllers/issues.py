from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional
import uuid
from app.database.models import Issue, Report, ReportUpload, Authority, User, HigherAuthority
from app.config.db import get_db
from app.services.ai_service import classify_clip, merge_model
from app.services.storage import upload_file_to_supabase
from app.middleware.auth import get_current_authority, get_current_higher_authority, get_current_user

router = APIRouter()

@router.post("/report")
async def report_issue(
    latitude: float = Form(...),
    longitude: float = Form(...),
    description: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Process the first image for classification and duplicate detection
    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    # Read the first image for AI processing
    first_image_bytes = await files[0].read()
    await files[0].seek(0)  # Reset file pointer
    
    # Classify the issue
    classification = classify_clip(first_image_bytes)
    
    # Check for duplicates
    merge_result = merge_model(first_image_bytes, latitude, longitude)
    
    if merge_result.get("is_duplicate"):
        # This is a duplicate report
        issue_id = merge_result["issue_id"]
        issue = db.query(Issue).filter(Issue.id == issue_id).first()
        
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")
        
        # Create a new report for the existing issue
        report_id = str(uuid.uuid4())
        new_report = Report(
            id=report_id,
            user_id=current_user.id,
            issue_id=issue_id,
            description=description
        )
        db.add(new_report)
        db.commit()
        
        # Upload files
        uploaded_files = []
        for file in files:
            file_url = await upload_file_to_supabase(file, f"reports/{report_id}/{file.filename}")
            if file_url:
                upload_id = str(uuid.uuid4())
                new_upload = ReportUpload(
                    id=upload_id,
                    report_id=report_id,
                    filename=file_url,
                    embedding=merge_result["embedding"]  # Save the embedding
                )
                db.add(new_upload)
                uploaded_files.append(file_url)
        
        db.commit()
        
        return {
            "message": "Duplicate report added to existing issue",
            "issue_id": str(issue_id),
            "report_id": report_id,
            "status": issue.status,
            "classification": classification,
            "uploaded_files": uploaded_files
        }
    else:
        # Create a new issue
        issue_id = str(uuid.uuid4())
        new_issue = Issue(
            id=issue_id,
            latitude=latitude,
            longitude=longitude,
            category=classification["label"],
            status="submitted"
        )
        
        # Create geometry point
        db.execute(
            text("INSERT INTO issues (id, latitude, longitude, location, category, status, created_at, updated_at) VALUES (:id, :latitude, :longitude, ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326), :category, :status, NOW(), NOW())"),
            {
                "id": issue_id,
                "latitude": latitude,
                "longitude": longitude,
                "category": classification["label"],
                "status": "submitted"
            }
        )
        
        # Create a report for this issue
        report_id = str(uuid.uuid4())
        new_report = Report(
            id=report_id,
            user_id=current_user.id,
            issue_id=issue_id,
            description=description
        )
        db.add(new_report)
        
        # Upload files
        uploaded_files = []
        for file in files:
            file_url = await upload_file_to_supabase(file, f"reports/{report_id}/{file.filename}")
            if file_url:
                upload_id = str(uuid.uuid4())
                new_upload = ReportUpload(
                    id=upload_id,
                    report_id=report_id,
                    filename=file_url,
                    embedding=merge_result["embedding"]  # Save the embedding
                )
                db.add(new_upload)
                uploaded_files.append(file_url)
        
        db.commit()
        
        return {
            "message": "New issue reported",
            "issue_id": issue_id,
            "report_id": report_id,
            "status": "submitted",
            "classification": classification,
            "uploaded_files": uploaded_files
        }

@router.get("/")
async def get_issues(
    skip: int = 0,
    limit: int = 100,
    current_authority: Authority = Depends(get_current_authority),
    db: Session = Depends(get_db)
):
    # Get nearby issues based on the authority's location and department
    query = text("""
        SELECT * FROM issues
        WHERE ST_DWithin(
            location::geography, 
            ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326)::geography, 
            10000  -- 10km radius
        )
        AND category IN (
            SELECT category FROM issue_categories WHERE department = :department
        )
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :skip
    """)
    
    result = db.execute(query, {
        "longitude": current_authority.longitude,
        "latitude": current_authority.latitude,
        "department": current_authority.department,
        "limit": limit,
        "skip": skip
    })
    
    issues = result.fetchall()
    return issues

@router.get("/all")
async def get_all_issues(
    skip: int = 0,
    limit: int = 100,
    current_higher_authority: HigherAuthority = Depends(get_current_higher_authority),
    db: Session = Depends(get_db)
):
    # Get all issues in the higher authority's department
    query = text("""
        SELECT issues.* FROM issues
        JOIN issue_categories ON issues.category = issue_categories.category
        WHERE issue_categories.department = :department
        ORDER BY issues.created_at DESC
        LIMIT :limit OFFSET :skip
    """)
    
    result = db.execute(query, {
        "department": current_higher_authority.department,
        "limit": limit,
        "skip": skip
    })
    
    issues = result.fetchall()
    return issues

@router.put("/{issue_id}/status")
async def update_issue_status(
    issue_id: str,
    status: str,
    current_authority: Authority = Depends(get_current_authority),
    db: Session = Depends(get_db)
):
    # Check if the issue exists
    issue = db.query(Issue).filter(Issue.id == issue_id).first()
    if not issue:
        raise HTTPException(status_code=404, detail="Issue not found")
    
    # Check if the issue is assigned to the current authority
    if issue.assigned_to != current_authority.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this issue")
    
    # Update the status
    issue.status = status
    db.commit()
    
    return {"message": "Issue status updated", "issue_id": issue_id, "new_status": status}