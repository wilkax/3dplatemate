"""
GET /api/v1/printers — returns the list of known printer profiles.
"""

import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.models.printer import PrinterProfile

router = APIRouter()

_PRINTERS_FILE = Path(__file__).parent.parent.parent.parent / "data" / "printers.json"


def _load_profiles() -> list[PrinterProfile]:
    with open(_PRINTERS_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    return [PrinterProfile(**p) for p in raw]


def get_profile_by_id(printer_id: str) -> PrinterProfile | None:
    """Utility used by the analyze route to look up a profile by ID."""
    for profile in _load_profiles():
        if profile.id == printer_id:
            return profile
    return None


@router.get("/printers", response_model=list[PrinterProfile])
def list_printers():
    """Return all available printer profiles sorted by name."""
    profiles = sorted(_load_profiles(), key=lambda p: p.name)
    return profiles

