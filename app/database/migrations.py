from sqlalchemy import text
from sqlalchemy.engine import Engine
from app.database.models import Base
from app.config.db import engine

def create_tables():
    """Create all tables if they don't exist"""
    Base.metadata.create_all(bind=engine)
    
    # Create spatial index for issues location
    with engine.connect() as conn:
        conn.execute(text('CREATE INDEX IF NOT EXISTS issues_location_gix ON issues USING GIST (location)'))
        conn.commit()

def run_migrations():
    """Run all migrations"""
    # Create tables
    create_tables()
    
    # Add vector extension if not exists
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # Add PostGIS extension if not exists (required for geoalchemy2)
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
        conn.commit()