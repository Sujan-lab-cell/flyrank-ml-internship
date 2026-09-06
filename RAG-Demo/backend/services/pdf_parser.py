"""
pdf_parser.py — Extract and chunk text from a PDF file.

Distinct from parser.py (which handles conversations.csv).

Public API:
  extract_chunks(pdf_path)  -> list[dict]
"""

import uuid
from pathlib import Path
from typing import List, Union

import pdfplumber

from utils.config import CHUNK_SIZE, CHUNK_OVERLAP


def _extract_text(pdf_path: Union[str, Path]) -> List[dict]:
    """
    Extract text page by page using pdfplumber.
    Returns [{ page, text }] — one entry per page that has text.
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": page_num, "text": text.strip()})
    return pages


def _chunk_page(page_num: int, text: str) -> List[dict]:
    """
    Split a single page's text into overlapping chunks.
    Returns [{ id, page, text, start_char, end_char }]
    """
    chunks = []
    start  = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append({
            "id":         str(uuid.uuid4()),
            "page":       page_num,
            "text":       text[start:end],
            "start_char": start,
            "end_char":   end,
        })
        if end == len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_chunks(pdf_path: Union[str, Path]) -> List[dict]:
    """
    Parse a PDF and return a flat list of text chunks ready for embedding.

    Parameters
    ----------
    pdf_path : str | Path

    Returns
    -------
    list[dict]:
        [{ id, page, text, start_char, end_char }, ...]
    """
    pages  = _extract_text(pdf_path)
    chunks = []
    for page_data in pages:
        chunks.extend(_chunk_page(page_data["page"], page_data["text"]))
    return chunks
