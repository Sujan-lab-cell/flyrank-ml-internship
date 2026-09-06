"""
Persona extractor — loads and manages personas from data/personas.json.

Public API:
  load_personas()          -> list[dict]
  get_persona(id)          -> dict | None
  get_default_persona()    -> dict
"""

import json
from pathlib import Path
from typing import List, Optional

PERSONAS_FILE = Path(__file__).parent.parent / "data" / "personas.json"


def load_personas() -> List[dict]:
    """Load all personas from personas.json."""
    if not PERSONAS_FILE.exists():
        return []
    with open(PERSONAS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("personas", [])


def get_persona(persona_id: str) -> Optional[dict]:
    """Return a persona by id, or None if not found."""
    return next(
        (p for p in load_personas() if p["id"] == persona_id),
        None,
    )


def get_default_persona() -> dict:
    """Return the first persona, or a safe fallback."""
    personas = load_personas()
    return personas[0] if personas else {
        "id":          "default",
        "name":        "Study Buddy",
        "description": "A helpful AI tutor.",
        "tone":        "friendly and concise",
    }
