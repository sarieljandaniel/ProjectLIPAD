from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routers import analysis, results, telemetry
from backend.services.telemetry_service import telemetry_service

FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await telemetry_service.start()
    yield
    await telemetry_service.stop()


app = FastAPI(title="Project LiPAD API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(telemetry.router)
app.include_router(analysis.router)
app.include_router(results.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Project LiPAD"}


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
