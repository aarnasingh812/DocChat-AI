
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database.sessions import init_sessions_table
from app.database.chunk_store import init_chunk_store
from app.api.routes import router

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("docchat.main")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB migrations on startup before accepting any requests."""
    init_sessions_table()   # sessions table (retries if DB isn't ready yet)
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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
