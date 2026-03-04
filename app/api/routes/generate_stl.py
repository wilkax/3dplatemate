"""
POST /api/v1/generate-stl

Step 2 of the manual cleaning workflow:
  1. Receive the PNG mask the user painted in the browser
     (white pixels = dirty spots, black = clean)
  2. Find contours in the mask → extract polygons
  3. Convert pixel coords → mm  (mask_width_px / plate_width_mm)
  4. Apply 2 mm margin and clip to plate bounds
  5. Generate and return binary STL

The mask is at the same resolution as the image returned by /prepare,
so px_per_mm = mask_width / plate_width_mm.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.core.geometry import buffer_polygon
from app.core.stl_gen import generate_stl
from app.core.training_data import save_mask

logger = logging.getLogger(__name__)
router = APIRouter()

_MARGIN_MM = 2.0
_MIN_AREA_PX = 200          # ignore blobs smaller than this (stray brush pixels)
_APPROX_EPSILON = 0.01      # contour simplification factor


@router.post("/generate-stl")
async def generate_stl_from_mask(
    mask: UploadFile = File(
        ..., description="Binary PNG mask — white pixels = dirty spots"
    ),
    plate_width_mm: float = Form(..., description="Plate width in mm"),
    plate_height_mm: float = Form(..., description="Plate height in mm"),
    session_id: Optional[str] = Form(None, description="Session ID from /prepare — links mask to plate images"),
):
    """
    Convert a user-painted mask into a plate_cleaner.stl file.

    The mask must be a PNG at the same resolution as the image returned
    by /prepare (width = plate_width_mm * px_per_mm).
    """
    mask_bytes = await mask.read()

    # Persist for training data — non-blocking, errors are only logged
    if session_id:
        save_mask(session_id, mask_bytes)

    nparr = np.frombuffer(mask_bytes, np.uint8)
    mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise HTTPException(status_code=422, detail="Could not decode mask image.")

    h, w = mask_img.shape
    px_per_mm = w / plate_width_mm

    # Threshold → binary
    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    # Find external contours of painted regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise HTTPException(
            status_code=422,
            detail="No painted areas found in mask. Paint some dirty spots first.",
        )

    raw_polygons_mm: list[list[list[float]]] = []
    buffered_polygons_mm: list[list[list[float]]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < _MIN_AREA_PX:
            continue

        eps = _APPROX_EPSILON * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        polygon_px = approx.reshape(-1, 2).tolist()
        if len(polygon_px) < 3:
            continue

        # Flip Y: image coords have Y=0 at top; 3D printing expects Y=0 at bottom.
        polygon_mm = [[pt[0] / px_per_mm, plate_height_mm - pt[1] / px_per_mm] for pt in polygon_px]
        raw_polygons_mm.append(polygon_mm)

        buffered = buffer_polygon(polygon_mm, _MARGIN_MM, plate_width_mm, plate_height_mm)
        if buffered is not None and len(buffered) >= 3:
            buffered_polygons_mm.append(buffered)

    if not buffered_polygons_mm:
        raise HTTPException(
            status_code=422,
            detail="All painted areas produced degenerate geometry — try painting larger regions.",
        )

    try:
        stl_bytes = generate_stl(buffered_polygons_mm)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    logger.info(
        "Generated STL from mask: %d spot(s), plate %.0fx%.0f mm, %.1f KB",
        len(buffered_polygons_mm), plate_width_mm, plate_height_mm, len(stl_bytes) / 1024,
    )

    return Response(
        content=stl_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": 'attachment; filename="plate_cleaner.stl"'},
    )

