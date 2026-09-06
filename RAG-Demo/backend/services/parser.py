"""
parser.py — Load and parse conversations from conversations.csv.

CSV format (no header):
  Each row contains one complete conversation as a raw string.
  Each turn follows the pattern:
      User 1: <message text>
      User 2: <message text>
      ...

Public API:
  load_conversations(csv_path)  -> list[dict]
  parse_conversation(text)      -> dict
"""

import re
from pathlib import Path
from typing import Union

import pandas as pd


# ---------------------------------------------------------------------------
# Regex: matches "User N: " at the start of a line (one or more digits after
# "User "), capturing both the speaker label and everything that follows until
# the next speaker label or end of string.
# ---------------------------------------------------------------------------
_TURN_PATTERN = re.compile(
    r"(User\s+\d+)\s*:\s*",   # speaker  e.g. "User 1:"
    re.IGNORECASE,
)


def parse_conversation(text: str) -> dict:
    """
    Parse a single conversation string into a structured dict.

    Parameters
    ----------
    text : str
        Raw conversation text, e.g.:
            "User 1: Hello\\nUser 2: Hi there!\\n..."

    Returns
    -------
    dict with shape:
        {
            "messages": [
                {"speaker": "User 1", "text": "Hello"},
                {"speaker": "User 2", "text": "Hi there!"},
                ...
            ]
        }
    """
    if not isinstance(text, str):
        return {"messages": []}

    text = text.strip()
    if not text:
        return {"messages": []}

    # Split on speaker markers; keeps the delimiter groups so we can pair them
    # with their following text using zip(speakers, bodies).
    parts = _TURN_PATTERN.split(text)

    # parts layout after split on a capturing group:
    #   [pre_text, speaker_1, body_1, speaker_2, body_2, ...]
    # The first element is any text before the first speaker (usually empty).
    # We start from index 1 and step by 2 to get (speaker, body) pairs.
    speakers = parts[1::2]
    bodies   = parts[2::2]

    messages = []
    for speaker, body in zip(speakers, bodies):
        clean_speaker = " ".join(speaker.strip().split())   # normalise whitespace
        clean_text    = " ".join(body.strip().split())      # collapse newlines + spaces
        if clean_text:                                       # skip empty turns
            messages.append({"speaker": clean_speaker, "text": clean_text})

    return {"messages": messages}


def load_conversations(csv_path: Union[str, Path]) -> list:
    """
    Load every conversation from a CSV file.

    The CSV is expected to have no header row; each cell / row contains
    one complete conversation string (quoted, possibly multi-line).

    Parameters
    ----------
    csv_path : str | Path
        Path to conversations.csv

    Returns
    -------
    list[dict] — one dict per row:
        {
            "conversation_id": 1,          # 1-based integer
            "messages": [
                {"speaker": "User 1", "text": "..."},
                ...
            ]
        }
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    # Read with no header; each row lands in column 0 as a raw string
    df = pd.read_csv(path, header=None, dtype=str)

    conversations = []
    for row_index, row in df.iterrows():
        raw_text = row.iloc[0]                    # first (and only) column
        parsed   = parse_conversation(raw_text)
        conversations.append(
            {
                "conversation_id": int(row_index) + 1,   # 1-based
                **parsed,                                 # injects "messages"
            }
        )

    return conversations
