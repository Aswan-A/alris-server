from fastapi import APIRouter
from app.controllers import auth

router = APIRouter()

router.post("/login/user", auth.login_user)
router.post("/login/authority", auth.login_authority)
router.post("/login/higher-authority", auth.login_higher_authority)
router.post("/refresh", auth.refresh_token)
router.post("/verify", auth.verify_token_endpoint)