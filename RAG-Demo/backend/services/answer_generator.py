"""
Answer generator — combines retrieved chunks + persona tone to produce a final answer.
Uses a stub LLM call; swap _call_llm() for the real model integration.
"""

from services.retriever import retrieve
from services.persona_extractor import get_default_persona


def _call_llm(prompt: str) -> str:
    """
    Stub — replace with your actual LLM call (OpenAI, Ollama, HuggingFace, etc.)
    """
    return f"[LLM not connected] Prompt received:\n{prompt[:200]}..."


def build_prompt(question: str, chunks: list[dict], persona: dict) -> str:
    context = "\n\n".join(
        f"[Chunk {i+1}]: {c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"You are {persona['name']}, a {persona.get('tone', 'helpful')} assistant.\n"
        f"Answer the following question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def generate_answer(question: str, document_id: str | None = None) -> dict:
    """
    Retrieve relevant chunks and generate an answer.
    Returns: { answer, sources: [{ page, text }] }
    """
    chunks = retrieve(question)
    persona = get_default_persona()
    prompt = build_prompt(question, chunks, persona)
    answer = _call_llm(prompt)

    sources = [
        {"page": c.get("page", 1), "text": c["text"][:200]}
        for c in chunks
    ]
    return {"answer": answer, "sources": sources}
