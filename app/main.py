from fastapi import FastAPI
from sqlalchemy import text

from app.api.documents import router as documents_router
from app.db.session import engine

app = FastAPI(
    title="AI Backend Platform",
    description="Backend services for AI-powered applications",
    version="0.1.0",
)

app.include_router(documents_router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/health/database")
async def database_health_check():
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT 1"))
        return {"database": "connected", "result": result.scalar()}
