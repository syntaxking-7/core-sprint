from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from dependencies import get_db
import crud

router = APIRouter()

@router.get("/{ticker}")
async def get_score(ticker: str, db: AsyncSession = Depends(get_db)):
    issuer = await crud.get_issuer_by_ticker(db, ticker)
    if not issuer:
        issuer = await crud.create_issuer(db, ticker)

    latest_score = await crud.get_latest_score(db, issuer.issuer_id)
    if latest_score:
        return latest_score

    result = await scoring.run_scoring_pipeline(issuer.issuer_id)
    await crud.insert_score(db, issuer.issuer_id, result["score"], result["band"], result["confidence"])
    return {"status": "computed", **result}
