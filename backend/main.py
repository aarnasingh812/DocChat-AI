
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.sessions import init_sessions_table
from app.database.chunk_store import init_chunk_store
from app.api.routes import router
from app.api.auth import router as auth_router
from app.database.auth import init_auth_tables
from app.config import FRONTEND_ORIGINS

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB migrations on startup before accepting any requests."""
    init_sessions_table()   # sessions table (retries if DB isn't ready yet)
    init_auth_tables()      # accounts, browser sessions, document ownership
    init_chunk_store()      # session_chunks table (idempotent DDL)
    yield


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocChat AI API",
    version="2.0.0",
    description="RAG-powered document Q&A API. Upload a PDF, then ask questions about it.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(auth_router)
