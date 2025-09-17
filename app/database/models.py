from sqlalchemy import Column, String, UUID, Double, Text, Boolean, TIMESTAMP, ForeignKey, Enum
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func
from geoalchemy2 import Geometry  # Updated import
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    reports = relationship("Report", back_populates="user")

class Authority(Base):
    __tablename__ = "authorities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    latitude = Column(Double, nullable=False)
    longitude = Column(Double, nullable=False)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)  # Updated
    department = Column(String, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    issues = relationship("Issue", back_populates="assigned_authority")

class HigherAuthority(Base):
    __tablename__ = "Higherauthorities"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String, nullable=True)
    password_hash = Column(String, nullable=False)
    department = Column(String, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())

class Issue(Base):
    __tablename__ = "issues"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    latitude = Column(Double, nullable=False)
    longitude = Column(Double, nullable=False)
    location = Column(Geometry(geometry_type='POINT', srid=4326), nullable=False)  # Updated
    category = Column(String, nullable=True)
    status = Column(Enum("submitted", "ongoing", "resolved", "rejected", name="issue_status"), default="submitted")
    assigned_to = Column(UUID(as_uuid=True), ForeignKey("authorities.id"), nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    assigned_authority = relationship("Authority", back_populates="issues")
    reports = relationship("Report", back_populates="issue")

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    issue_id = Column(UUID(as_uuid=True), ForeignKey("issues.id"), nullable=False)
    description = Column(Text, nullable=True)
    is_spam = Column(Boolean, default=False)
    is_fake = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    user = relationship("User", back_populates="reports")
    issue = relationship("Issue", back_populates="reports")
    uploads = relationship("ReportUpload", back_populates="report")

class ReportUpload(Base):
    __tablename__ = "report_uploads"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("reports.id"), nullable=False)
    filename = Column(String, nullable=False)
    embedding = None  # This would be a vector column in a real implementation
    uploaded_at = Column(TIMESTAMP, server_default=func.now())
    
    report = relationship("Report", back_populates="uploads")