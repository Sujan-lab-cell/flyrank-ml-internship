"""
build_faiss.py — builds the FAISS index from a processed PDF.

Usage:
    python vectorstore/build_faiss.py --pdf path/to/file.pdf
"""

import argparse
import pickle

import faiss
import numpy as np

from services.parser import parse_pdf
from models.embedding_model import encode
from utils.config import FAISS_INDEX_PATH, METADATA_PATH


def build_index(pdf_path: str) -> None:
    print(f"Parsing PDF: {pdf_path}")
    chunks = parse_pdf(pdf_path)

    print(f"Encoding {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    embeddings = np.array(encode(texts), dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"Saving index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"Saving metadata to {METADATA_PATH}")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    args = parser.parse_args()
    build_index(args.pdf)
