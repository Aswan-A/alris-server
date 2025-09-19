from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.database.models import User, Authority, HigherAuthority
from app.config.db import get_db
from app.utils.jwt import create_access_token, create_refresh_token, verify_password, verify_token
from app.config.settings import settings

router = APIRouter()

from pydantic import BaseModel

class LoginRequest(BaseModel):
    email: str
    password: str
import logging
logger = logging.getLogger(__name__)

@router.post("/login/user")
async def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    logger.info(f"Login attempt: email={request.email}")
    # or print(f"Login attempt: {request.email}")

    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        logger.warning(f"Failed login for {request.email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.email}, expires_delta=access_token_expires)
    refresh_token = create_refresh_token(data={"sub": user.email})
    
    logger.info(f"Successful login: {user.email}")
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }

@router.post("/login/higher-authority")
async def login_higher_authority(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    higher_authority = db.query(HigherAuthority).filter(HigherAuthority.email == form_data.username).first()
    if not higher_authority or not verify_password(form_data.password, higher_authority.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": higher_authority.email}, expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={"sub": higher_authority.email})
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@router.post("/refresh")
async def refresh_token(refresh_token: str, db: Session = Depends(get_db)):
    try:
        email = verify_token(refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if the user exists
    user = db.query(User).filter(User.email == email).first()
    if not user:
        authority = db.query(Authority).filter(Authority.email == email).first()
        if not authority:
            higher_authority = db.query(HigherAuthority).filter(HigherAuthority.email == email).first()
            if not higher_authority:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
    
    # Create new access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    new_refresh_token = create_refresh_token(data={"sub": email})
    return {"access_token": access_token, "refresh_token": new_refresh_token, "token_type": "bearer"}

@router.post("/verify")
async def verify_token_endpoint(token: str, db: Session = Depends(get_db)):
    try:
        email = verify_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if the user exists
    user = db.query(User).filter(User.email == email).first()
    if not user:
        authority = db.query(Authority).filter(Authority.email == email).first()
        if not authority:
            higher_authority = db.query(HigherAuthority).filter(HigherAuthority.email == email).first()
            if not higher_authority:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
    
    return {"email": email, "message": "Token is valid"}