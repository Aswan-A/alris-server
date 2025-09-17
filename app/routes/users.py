from fastapi import APIRouter
from app.controllers import users

router = APIRouter()

router.post("/register", users.register_user)
router.get("/me", users.get_user_profile)