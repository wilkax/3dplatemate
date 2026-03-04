"""
Claude Vision integration.

detect_dirty_spots():
    Receives the perspective-corrected top-down plate image.
    Returns dirty-spot polygons in corrected-image pixel coordinates.
"""

import base64
import json

import cv2
import numpy as np
import anthropic

from app.config import settings

# ── Prompt: find dirty spots on the corrected top-down image ─────────────────
_SPOTS_PROMPT = """\
You are analysing a contrast-enhanced, top-down photo of a 3D printer build plate.
The full image IS the plate — every pixel belongs to the plate surface.

Your task: find ALL dirty spots. Be aggressive — false positives are acceptable
because the user will review results in their slicer. Missing a spot is worse
than marking a clean area.

What to look for:
- Raised blobs or chunks of filament from a previous print
- Thin shiny or matte plastic film
- Faint outlines, shadows, or footprints of previous prints
- Discoloration, staining, or areas where the surface texture looks different
- Any region that is not the uniform base colour of the plate

Rules:
- Coordinates are pixel coordinates in this image.
- Draw each polygon tightly around the residue — not a large bounding box.
- Cover the entire dirty area; do not split one connected spot into many polygons.
- If the plate is truly clean, return an empty array for dirty_spots.
"""

_SPOTS_SCHEMA = {
    "type": "object",
    "properties": {
        "dirty_spots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "polygon": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                        },
                    },
                },
                "required": ["description", "polygon"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["dirty_spots"],
    "additionalProperties": False,
}


def enhance_for_detection(image_bytes: bytes) -> bytes:
    """
    Pre-process the corrected plate image to make residue more visible to Claude.

    Steps:
      1. CLAHE on the L channel (LAB) — boosts local contrast without blowing
         out highlights; makes faint film and footprints stand out.
      2. Unsharp mask — sharpens edges of residue blobs.

    Returns JPEG bytes of the enhanced image at high quality.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # CLAHE in LAB colour space (operates only on luminance)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Unsharp mask: sharpened = original * 1.5 - blurred * 0.5
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()


def _detect_media_type(image_bytes: bytes) -> str:
    if image_bytes[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _call_claude(image_bytes: bytes, prompt: str, schema: dict) -> dict:
    """Send one image + prompt to Claude with a JSON schema output config."""
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    media_type = _detect_media_type(image_bytes)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        },
    )

    return json.loads(response.content[0].text)


def detect_dirty_spots(corrected_image_bytes: bytes) -> list[dict]:
    """
    Step 2 — find dirty spots on the perspective-corrected top-down plate image.

    Returns
    -------
    list of {"description": str, "polygon": [[x, y], ...]}
    """
    result = _call_claude(corrected_image_bytes, _SPOTS_PROMPT, _SPOTS_SCHEMA)
    return result.get("dirty_spots", [])

