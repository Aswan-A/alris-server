from fastapi import APIRouter
from app.controllers import issues

router = APIRouter()

router.post("/report", issues.report_issue)
router.get("/", issues.get_issues)
router.get("/all", issues.get_all_issues)
router.put("/{issue_id}/status", issues.update_issue_status)