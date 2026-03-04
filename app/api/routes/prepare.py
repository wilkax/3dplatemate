"""
POST /api/v1/prepare

Step 1 of the manual cleaning workflow:
  1. Detect the 4 plate corners (OpenCV)
  2. Perspective-correct the image to a top-down view
  3. Enhance contrast (CLAHE + unsharp mask) for easier spot spotting
  4. Return the enhanced image as a base64 JPEG plus plate metadata

The frontend displays this image on a canvas and lets the user paint
dirty spots before calling /api/v1/generate-stl.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.routes.printers import get_profile_by_id
from app.core.geometry import PX_PER_MM, compute_homography, warp_image
from app.core.plate_detection import detect_plate_corners
from app.core.training_data import make_session_id, save_plate_images
from app.core.vision import enhance_for_detection

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def _resolve_plate_dimensions(
    printer_id: Optional[str],
    plate_width_mm: Optional[float],
    plate_height_mm: Optional[float],
) -> tuple[float, float]:
    if printer_id:
        profile = get_profile_by_id(printer_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"Printer '{printer_id}' not found.")
        return profile.plate_width_mm, profile.plate_height_mm

    if plate_width_mm and plate_height_mm:
        if plate_width_mm <= 0 or plate_height_mm <= 0:
            raise HTTPException(status_code=422, detail="Plate dimensions must be positive.")
        return plate_width_mm, plate_height_mm

    raise HTTPException(
        status_code=422,
        detail="Provide either 'printer_id' or both 'plate_width_mm' and 'plate_height_mm'.",
    )


@router.post("/prepare")
async def prepare(
    image: UploadFile = File(..., description="Photo of the build plate (JPEG/PNG/WEBP)"),
    printer_id: Optional[str] = Form(None, description="Printer ID from /api/v1/printers"),
    plate_width_mm: Optional[float] = Form(None, description="Manual plate width in mm"),
    plate_height_mm: Optional[float] = Form(None, description="Manual plate height in mm"),
):
    """
    Detect the build plate, perspective-correct and enhance the image.

    Returns JSON:
      image          : base64-encoded JPEG of the enhanced top-down plate view
      plate_width_mm : plate width in mm
      plate_height_mm: plate height in mm
      px_per_mm      : pixel scale of the returned image
    """
    width_mm, height_mm = _resolve_plate_dimensions(printer_id, plate_width_mm, plate_height_mm)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit.")

    # Corner detection
    try:
        corners_px = detect_plate_corners(image_bytes)
    except ValueError as exc:
        logger.error("Corner detection failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"Corner detection failed: {exc}")

    # Perspective correction
    try:
        H = compute_homography(corners_px, width_mm, height_mm)
        corrected_img = warp_image(image_bytes, H, width_mm, height_mm)
    except Exception as exc:
        logger.error("Perspective correction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to correct image perspective.")

    # Encode then enhance
    _, buf = cv2.imencode(".jpg", corrected_img)
    enhanced_bytes = enhance_for_detection(buf.tobytes())

    # Persist for training data — non-blocking, errors are only logged
    session_id = make_session_id()
    save_plate_images(session_id, corrected_img, enhanced_bytes)

    logger.info(
        "Prepared plate image: %.0fx%.0f mm at %.1f px/mm (session %s)",
        width_mm, height_mm, PX_PER_MM, session_id,
    )

    return {
        "image": base64.b64encode(enhanced_bytes).decode(),
        "plate_width_mm": width_mm,
        "plate_height_mm": height_mm,
        "px_per_mm": PX_PER_MM,
        "session_id": session_id,
    }

