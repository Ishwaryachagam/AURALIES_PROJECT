"""
Auralis backend entrypoint.

Run with:
    uvicorn backend.main:app --reload --port 8000

(Run this from the project ROOT, one level above backend/, so the
`backend.` package imports resolve correctly.)
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.api.routes import router, close_all_connections
from backend.models.model_manager import model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auralis.main")

app = FastAPI(title="Auralis Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def on_startup():
    logger.info("Starting Auralis backend...")
    model_manager.load_all()  # loads Whisper, Silero VAD, emotion model, TTS once


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down - closing WebRTC connections...")
    await close_all_connections()
