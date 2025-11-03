import datetime
import os
from fastapi import Depends, FastAPI, HTTPException, Query
import requests
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select
from db import init_db
from dependencies import get_db
from pipeline.main_pipeline import run_credit_pipeline_with_db_cache
from models import Analysis, Company
from routes import scores
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI(title="Credit Scoring Service", version="1.0.0")

load_dotenv()

origins = ["*"]
env_origins = os.getenv("CORS_ORIGINS", "")
if env_origins:
    origins = env_origins.split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()  # Now sync

@app.get("/")
def root():
    return {"message": "Credit Scoring API is running"}

FRED_API_KEY = os.getenv("FRED_API_KEY", "1a39ebc94e4984ff4091baa2f84c0ba7")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "d2jn8q1r01qj8a5k949gd2jn8q1r01qj8a5k94a0")


class AnalysisRequest(BaseModel):
    company_name: str
    ticker: str


@app.post("/analyze-and-wait")
def analyze_and_wait(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    1. Check if cached analysis exists within 6 hours.
    2. If exists, return it.
    3. Otherwise, run pipeline, save new analysis, return it.
    """

    if not db:
        raise HTTPException(status_code=503, detail="Database service is unavailable.")

    try:
        company = db.query(Company).filter(Company.ticker == request.ticker).first()

        if company:
            latest_analysis = (
                db.query(Analysis)
                .filter(Analysis.ticker == request.ticker)
                .order_by(Analysis.created_at.desc())
                .first()
            )

            if latest_analysis:
                hours_diff = (datetime.datetime.utcnow() - latest_analysis.created_at).total_seconds() / 3600
                if hours_diff < 6:
                    return latest_analysis

        # Generate new report
        report = run_credit_pipeline_with_db_cache(
            company_name=request.company_name,
            ticker=request.ticker,
            fred_api_key=FRED_API_KEY,
        )

        if not company:
            company = Company(ticker=request.ticker, name=request.company_name)
            db.add(company)
            db.commit()

        entry = Analysis(
            ticker=request.ticker,
            report=report,  # Ensure JSON is stored as string
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)

        return entry

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")


@app.get("/search-ticker")
def search_ticker(q: str = Query(..., min_length=1)):
    url = f"https://finnhub.io/api/v1/search?q={q}&token={FINNHUB_API_KEY}"
    r = requests.get(url)
    data = r.json()

    # Finnhub returns { "count": X, "result": [{symbol, description}, ...] }
    results = [
        {"ticker": item["symbol"], "name": item["description"]}
        for item in data.get("result", [])
        if item.get("symbol") and item.get("description")
    ]

    return {"results": results[:10]}  # limit to 10

@app.get("/analyses/latest")
def get_latest_analyses(limit: int = 9, db: Session = Depends(get_db)):
    try:
        analyses = (
            db.query(Analysis)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
            .all()
        )

        response = []
        for a in analyses:
            if isinstance(a.report, str):
                try:
                    report_data = json.loads(a.report)
                except json.JSONDecodeError:
                    report_data = {}
            elif isinstance(a.report, dict):
                report_data = a.report
            else:
                report_data = {}

            response.append({
                "id": str(a.id),
                "ticker": a.ticker,
                "company": {
                    "name": a.company.name if a.company else None,
                    "ticker": a.ticker
                },
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "score": report_data.get("explainability_report", {}).get("final_score"),
                "grade": report_data.get("company_info", {}).get("credit_grade")
            })

        return {"analyses": response}

    except Exception as e:
        return {"error": str(e)}



@app.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    try:
        analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
        
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        # Parse report as dict if stored as JSON string
        report_data = {}
        if analysis.report:
            try:
                report_data = (
                    analysis.report if isinstance(analysis.report, dict) else json.loads(analysis.report)
                )
            except:
                report_data = {}

        return {
            "id": str(analysis.id),
            "ticker": analysis.ticker,
            "company_name": analysis.company.name,
            "created_at": analysis.created_at.isoformat(),
            "report": report_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
