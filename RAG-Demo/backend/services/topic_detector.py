"""
Topic detector — extracts key topics from a list of text chunks.

Uses simple keyword extraction (no heavy model required).
Swap extract_keywords() for an LLM call when the model is ready.

Public API:
  detect_topics(chunks, top_n=5)  -> list[str]
"""

import re
from collections import Counter
from typing import List

# Common English stop words to ignore during keyword extraction
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "our", "their", "what", "which", "who", "how", "when", "where", "why",
    "not", "no", "so", "as", "if", "then", "than", "up", "out", "about",
    "also", "just", "like", "use", "used", "can", "get", "one", "all",
    "more", "some", "any", "each", "both", "other", "such", "into",
}


def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Return the top_n most frequent meaningful words from text.
    Simple frequency-based extraction — no external model needed.
    """
    words   = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    counts  = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def detect_topics(chunks: List[dict], top_n: int = 5) -> List[str]:
    """
    Detect the main topics across a list of chunk dicts.

    Combines all chunk text, extracts the most frequent meaningful words,
    and returns them as topic labels.

    Parameters
    ----------
    chunks : list of dicts with a "text" key
    top_n  : number of topic keywords to return

    Returns
    -------
    list[str] — e.g. ["machine", "learning", "neural", "data", "model"]
    """
    if not chunks:
        return []

    combined_text = " ".join(c.get("text", "") for c in chunks)
    return _extract_keywords(combined_text, top_n=top_n)
