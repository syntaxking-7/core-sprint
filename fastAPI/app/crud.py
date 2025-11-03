from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import Issuer, Score
from datetime import datetime

async def get_issuer_by_ticker(db: AsyncSession, ticker: str):
    result = await db.execute(select(Issuer).where(Issuer.ticker == ticker))
    return result.scalars().first()

async def create_issuer(db: AsyncSession, ticker: str):
    new_issuer = Issuer(name=ticker, ticker=ticker, sector="Unknown", country="Unknown")
    db.add(new_issuer)
    await db.commit()
    await db.refresh(new_issuer)
    return new_issuer

async def get_latest_score(db: AsyncSession, issuer_id):
    result = await db.execute(
        select(Score).where(Score.issuer_id == issuer_id).order_by(Score.ts.desc()).limit(1)
    )
    return result.scalars().first()

async def insert_score(db: AsyncSession, issuer_id, score, band, confidence, model_version="v1.0"):
    new_score = Score(
        issuer_id=issuer_id,
        ts=datetime.utcnow(),
        fused_score=score,
        band=band,
        model_version=model_version,
        confidence=confidence
    )
    db.add(new_score)
    await db.commit()
    await db.refresh(new_score)
    return new_score
