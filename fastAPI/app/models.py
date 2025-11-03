from sqlalchemy import Column, String, Float, JSON, TIMESTAMP, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
import uuid
from db import Base
from sqlalchemy import Column, String, Float, DateTime, Boolean
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

Base = declarative_base()

class Issuer(Base):
    __tablename__ = "issuers"
    issuer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    ticker = Column(String, unique=True, index=True)
    sector = Column(String)
    country = Column(String)


class Company(Base):
    __tablename__ = "companies"

    ticker = Column(String, primary_key=True, index=True)  # e.g., "AAPL"
    name = Column(String, nullable=False)  # e.g., "Apple Inc."

    analyses = relationship("Analysis", back_populates="company", cascade="all, delete-orphan")

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ticker = Column(String, ForeignKey("companies.ticker"), nullable=False)
    report = Column(JSONB, nullable=False)  # Full JSON report
    created_at = Column(DateTime, default=datetime.utcnow)

    company = relationship("Company", back_populates="analyses")
class Score(Base):
    __tablename__ = "scores"
    issuer_id = Column(UUID(as_uuid=True), primary_key=True)
    ts = Column(TIMESTAMP, primary_key=True)
    fused_score = Column(Float, nullable=False)
    band = Column(String)
    model_version = Column(String)
    confidence = Column(Float)

class Explanation(Base):
    __tablename__ = "explanations"
    issuer_id = Column(UUID(as_uuid=True), primary_key=True)
    ts = Column(TIMESTAMP, primary_key=True)
    ebm_feature_json = Column(JSON)
    news_analysis_json = Column(JSON)
    fusion_summary_text = Column(String)
    report_uri = Column(String)

class Alert(Base):
    __tablename__ = "alerts"
    alert_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    issuer_id = Column(UUID(as_uuid=True))
    ts = Column(TIMESTAMP)
    score_prev = Column(Float)
    score_new = Column(Float)
    delta = Column(Float)
    rules_json = Column(JSON)
    severity = Column(String)
    top_drivers = Column(JSON)
    news_factors = Column(JSON)
    fusion_snapshot = Column(JSON)
