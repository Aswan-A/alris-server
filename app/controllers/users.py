from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database.models import User
from app.config.db import get_db
from app.utils.jwt import get_password_hash
from app.middleware.auth import get_current_user

router = APIRouter()

@router.post("/register")
async def register_user(name: str, email: str, password: str, phone: str = None, db: Session = Depends(get_db)):
    # Check if user already exists
    db_user = db.query(User).filter(User.email == email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    db_user = User(
        name=name,
        email=email,
        phone=phone,
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