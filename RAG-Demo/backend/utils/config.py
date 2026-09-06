"""
Central configuration — all tuneable constants in one place.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
UPLOAD_DIR      = BASE_DIR / "uploads"

FAISS_INDEX_PATH = VECTORSTORE_DIR / "faiss_index.bin"
METADATA_PATH    = VECTORSTORE_DIR / "metadata.pkl"

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME  = "all-MiniLM-L6-v2"   # fast, good quality
SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # characters per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5             # number of chunks to retrieve per query

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
