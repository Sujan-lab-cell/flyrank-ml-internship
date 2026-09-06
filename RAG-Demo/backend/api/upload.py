"""
POST /upload

Accepts a PDF file, extracts text chunks, embeds them with the
sentence-transformer model, and upserts into the FAISS vector store.

Response:
  {
    "document_id": "<uuid>",
    "filename":    "lecture.pdf",
    "chunks":      42,
    "topics":      ["Topic A", "Topic B", ...]
  }
"""

import pickle
import shutil
import uuid
from pathlib import Path
from typing import List

import faiss
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from models.embedding_model import encode
from services.pdf_parser     import extract_chunks
from services.topic_detector import detect_topics
from utils.config            import FAISS_INDEX_PATH, METADATA_PATH, UPLOAD_DIR

router = APIRouter()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── Response schema ───────────────────────────────────────────────────────────
class UploadResponse(BaseModel):
    document_id: str
    filename:    str
    chunks:      int
    topics:      List[str]


# ── Route ─────────────────────────────────────────────────────────────────────
@router.post("", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF and index it for retrieval."""

    # Validate file type
    is_pdf = (
        file.content_type == "application/pdf"
        or (file.filename or "").lower().endswith(".pdf")
    )
    if not is_pdf:
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    doc_id    = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}.pdf"

    # ── Save to disk ──────────────────────────────────────────────────────────
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")
    finally:
        await file.close()

    # ── Parse PDF into text chunks ────────────────────────────────────────────
    try:
        chunks = extract_chunks(save_path)
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"PDF parsing failed: {exc}")

    if not chunks:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="No text could be extracted from this PDF.")

    # Tag every chunk with its document id
    for chunk in chunks:
        chunk["document_id"] = doc_id

    # ── Detect topics ─────────────────────────────────────────────────────────
    topics = detect_topics(chunks)

    # ── Embed chunks ──────────────────────────────────────────────────────────
    texts      = [c["text"] for c in chunks]
    embeddings = np.array(encode(texts), dtype="float32")
    dim        = embeddings.shape[1]

    # ── Upsert into FAISS ─────────────────────────────────────────────────────
    if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists():
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        index    = faiss.IndexFlatL2(dim)
        metadata = []

    index.add(embeddings)
    metadata.extend(chunks)

    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename or "unknown.pdf",
        chunks=len(chunks),
        topics=topics,
    )
