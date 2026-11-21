from fastapi import FastAPI

app = FastAPI(
    title="AI Backend Platform",
    description="Backend services for AI-powered applications",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}