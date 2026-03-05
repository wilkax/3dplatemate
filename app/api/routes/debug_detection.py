"""
POST /api/v1/debug-detection

Runs the plate corner detection pipeline in debug mode and returns
annotated intermediate images for every detection step.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.plate_detection import detect_plate_corners_debug

logger = logging.getLogger(__name__)
router = APIRouter()

_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/debug-detection")
async def debug_detection(
    image: UploadFile = File(..., description="Photo of the build plate"),
    plate_width_mm: float = Form(..., description="Plate width in mm"),
    plate_height_mm: float = Form(..., description="Plate height in mm"),
):
    """
    Run the corner detection pipeline and return all intermediate step images.

    Returns JSON:
      steps   : list of {title, image (base64 JPEG), description}
      corners : {top_left, top_right, bottom_right, bottom_left} or null
      error   : error message string or null
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit.")

    logger.info(
        "Debug detection request: %.0fx%.0f mm, image %.1f KB",
        plate_width_mm, plate_height_mm, len(image_bytes) / 1024,
    )

    result = detect_plate_corners_debug(image_bytes, plate_width_mm, plate_height_mm)

    logger.info(
        "Debug detection complete: %d steps, success=%s",
        len(result["steps"]), result["error"] is None,
    )

    return result

