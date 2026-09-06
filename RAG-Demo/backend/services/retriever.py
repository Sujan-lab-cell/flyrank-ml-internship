"""
Retriever — queries the FAISS vector store to find relevant chunks.

Public API:
  retrieve(query, document_id=None, top_k=TOP_K)  -> list[dict]
"""

import pickle
from typing import List, Optional

import faiss
import numpy as np

from models.embedding_model import encode
from utils.config import FAISS_INDEX_PATH, METADATA_PATH, TOP_K

_index    = None
_metadata = None


def _load_index():
    global _index, _metadata
    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {FAISS_INDEX_PATH}. Upload a PDF first."
        )
    if _index is None:
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
    if _metadata is None:
        with open(METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)


def retrieve(
    query: str,
    document_id: Optional[str] = None,
    top_k: int = TOP_K,
) -> List[dict]:
    """
    Embed the query and return the top_k most relevant chunks.

    Parameters
    ----------
    query       : natural-language question
    document_id : optional — filter results to a specific uploaded document
    top_k       : number of chunks to return

    Returns
    -------
    list[dict]: [{ id, page, text, document_id, score }, ...]
    """
    _load_index()

    query_vec            = np.array(encode([query]), dtype="float32")
    distances, indices   = _index.search(query_vec, top_k * 3)   # over-fetch for filtering

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        chunk = _metadata[idx].copy()
        chunk["score"] = round(float(dist), 6)

        # Optional document filter
        if document_id and chunk.get("document_id") != document_id:
            continue

        results.append(chunk)
        if len(results) >= top_k:
            break

    return results
