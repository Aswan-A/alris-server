from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.models import User
from app.config.db import get_db
from app.utils.jwt import get_password_hash
from app.middleware.auth import get_current_user

router = APIRouter()
from pydantic import BaseModel

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    phone: str | None = None

@router.post("/register")
async def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == request.email).first()
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    hashed_password = get_password_hash(request.password)
    db_user = User(
        name=request.name,
        email=request.email,
        phone=request.phone,
        password_hash=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    return {"message": "User registered successfully", "user_id": str(db_user.id)}

@router.get("/me")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    return {
        "id": str(current_user.id),
        "name": current_user.name,
        "email": current_user.email,
        "phone": current_user.phone,
        "created_at": current_user.created_at
    }