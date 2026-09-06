"""
POST /query

Accepts a natural-language question, retrieves the most relevant chunks
from FAISS, detects topics in those chunks, resolves the active persona,
and returns a structured JSON response.

Request:
  { "question": "What is the main topic?", "document_id": "optional-uuid" }

Response:
  {
    "answer":   "...",
    "sources":  [{ "chunk_id": "...", "text": "...", "score": 0.42 }],
    "topics":   ["Topic A", "Topic B"],
    "persona":  { "id": "default", "name": "Study Buddy", ... }
  }
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.retriever        import retrieve
from services.topic_detector   import detect_topics
from services.persona_extractor import get_default_persona

router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:    str
    document_id: Optional[str] = None


class SourceItem(BaseModel):
    chunk_id: str
    text:     str
    score:    float


class QueryResponse(BaseModel):
    answer:  str
    sources: List[SourceItem]
    topics:  List[str]
    persona: dict


# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Retrieve relevant chunks and return an answer with metadata."""

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # ── Retrieve relevant chunks from FAISS ───────────────────────────────────
    try:
        chunks = retrieve(request.question, document_id=request.document_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Vector store not found. Upload a PDF first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    if not chunks:
        return QueryResponse(
            answer="No relevant content found. Please upload a PDF first.",
            sources=[],
            topics=[],
            persona=get_default_persona(),
        )

    # ── Detect topics in the retrieved chunks ─────────────────────────────────
    topics = detect_topics(chunks)

    # ── Resolve persona ───────────────────────────────────────────────────────
    persona = get_default_persona()

    # ── Build answer (stub — wire real LLM here) ──────────────────────────────
    context = "\n\n".join(
        f"[{i+1}] {c['text'][:300]}" for i, c in enumerate(chunks)
    )
    answer = (
        f"[{persona['name']}] Based on the uploaded document:\n\n{context}"
    )

    # ── Format sources ────────────────────────────────────────────────────────
    sources = [
        SourceItem(
            chunk_id=c.get("id", ""),
            text=c["text"][:300],
            score=round(c.get("score", 0.0), 4),
        )
        for c in chunks
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        topics=topics,
        persona=persona,
    )
