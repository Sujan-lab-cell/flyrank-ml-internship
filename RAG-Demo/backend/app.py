"""
KaStack-RAG — FastAPI entry point

Endpoints:
  POST /upload   — upload a PDF, parse + embed into FAISS
  POST /query    — ask a question, retrieve chunks, return answer + topics + persona
  GET  /health   — liveness check
"""

import sys
from pathlib import Path

# Ensure the backend root is on sys.path so relative imports always resolve
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.upload import router as upload_router
from api.query  import router as query_router

app = FastAPI(
    title="KaStack-RAG API",
    description=(
        "RAG-powered study assistant. "
        "Upload a PDF, then query it with natural language."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow the React dev server and any prod origin ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(upload_router, prefix="/upload", tags=["Upload"])
app.include_router(query_router,  prefix="/query",  tags=["Query"])


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe — returns 200 when the server is running."""
    return {"status": "ok", "message": "KaStack-RAG API is running"}
