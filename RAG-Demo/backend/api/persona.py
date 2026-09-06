"""
GET /persona — returns available personas.
"""

from fastapi import APIRouter, HTTPException
from services.persona_extractor import load_personas, get_persona

router = APIRouter()


@router.get("")
def list_personas():
    return {"personas": load_personas()}


@router.get("/{persona_id}")
def get_persona_by_id(persona_id: str):
    persona = get_persona(persona_id)
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found.")
    return persona
