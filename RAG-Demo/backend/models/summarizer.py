"""
Text summarizer.
Uses a lightweight summarization pipeline (or an LLM call).
Swap the implementation when the LLM is ready.
"""

from transformers import pipeline
from utils.config import SUMMARIZER_MODEL_NAME

_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=SUMMARIZER_MODEL_NAME)
    return _summarizer


def summarize(text: str, max_length: int = 150, min_length: int = 40) -> str:
    """Return a short summary of the given text."""
    summarizer = get_summarizer()
    result = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return result[0]["summary_text"]
