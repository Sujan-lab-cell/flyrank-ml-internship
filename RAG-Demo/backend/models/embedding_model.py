"""
Embedding model wrapper.
Loads a sentence-transformer model and exposes encode().
Swap the model name in config.py to change the embedding model.
"""

from sentence_transformers import SentenceTransformer
from utils.config import EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer instance."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    """Encode a list of strings into embedding vectors."""
    model = get_model()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()
