import os
import psycopg2
from urllib.parse import urlparse
from pgvector.psycopg2 import register_vector
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.event import listen
from geoalchemy2 import Geometry  # Import for event listener
from app.config.settings import settings

# SQLAlchemy setup
engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Register PostgreSQL types for geoalchemy2
def register_postgresql_types(dbapi_connection, connection_record):
    dbapi_connection.autocommit = True
    cursor = dbapi_connection.cursor()
    cursor.execute("SET TIME ZONE UTC")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.close()

listen(engine, 'connect', register_postgresql_types)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Direct psycopg2 connection for pgvector
def get_db_connection():
    db_url = settings.DATABASE_URL
    if not db_url:
        raise ValueError("DATABASE_URL not found in environment.")
    
    parsed = urlparse(db_url)
    conn = psycopg2.connect(
        dbname=parsed.path.lstrip("/"),
        user=parsed.username,
        password=parsed.password,
        host=parsed.hostname,
        port=parsed.port,
        sslmode="require"
    )
    register_vector(conn)
    return conn