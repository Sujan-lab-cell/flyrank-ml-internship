"""
Checkpoint generator — creates study checkpoints / key-point summaries from chunks.
"""

from models.summarizer import summarize


def generate_checkpoints(chunks: list[dict], every_n: int = 5) -> list[dict]:
    """
    Every `every_n` chunks, generate a checkpoint summary.
    Returns: [{ chunk_index, summary }]
    """
    checkpoints = []
    for i in range(0, len(chunks), every_n):
        batch_text = " ".join(c["text"] for c in chunks[i: i + every_n])
        summary = summarize(batch_text)
        checkpoints.append({"chunk_index": i, "summary": summary})
    return checkpoints
