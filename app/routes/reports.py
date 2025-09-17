from fastapi import APIRouter
from app.controllers import reports

router = APIRouter()

router.get("/", reports.get_user_reports)
router.get("/{report_id}", reports.get_report)