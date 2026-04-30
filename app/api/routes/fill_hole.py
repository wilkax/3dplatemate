"""
"Fill the hole" workflow.

Endpoints
---------
GET  /api/v1/fill-hole/marker.png?size_mm=...    Printable ArUco reference.
POST /api/v1/fill-hole/prepare                   Detect marker, warp top-down,
                                                  auto-detect hole polygon.
POST /api/v1/fill-hole/generate                  Convert traced polygon (mask)
                                                  to a ribbed plug 3MF.
"""

from __future__ import annotations

import base64
import logging

import cv2
import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.core.aruco_detection import (
    DEFAULT_PX_PER_MM,
    DEFAULT_VIEW_MM,
    detect_and_warp,
    render_marker_png,
)
from app.core.hole_detection import detect_hole_polygon
from app.core.plug_gen import generate_plug_3mf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/fill-hole")

_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MIN_DEPTH_MM    = 0.5
_MAX_DEPTH_MM    = 100.0
_MIN_TOLERANCE_MM = 0.0
_MAX_TOLERANCE_MM = 5.0
_MIN_MARKER_MM   = 10.0
_MAX_MARKER_MM   = 300.0


@router.get("/marker.png")
def get_marker_png(size_mm: float = 50.0):
    """Render the ArUco reference marker as a printable PNG (300 DPI)."""
    if not (_MIN_MARKER_MM <= size_mm <= _MAX_MARKER_MM):
        raise HTTPException(
            status_code=422,
            detail=f"size_mm must be between {_MIN_MARKER_MM} and {_MAX_MARKER_MM}.",
        )
    try:
        png = render_marker_png(size_mm=size_mm, px_per_mm=300.0 / 25.4)
    except Exception as exc:
        logger.exception("Marker rendering failed")
        raise HTTPException(status_code=500, detail=f"Marker rendering failed: {exc}")
    return Response(content=png, media_type="image/png")


@router.post("/prepare")
async def prepare(
    image: UploadFile = File(..., description="Photo of marker placed next to the hole"),
    marker_size_mm: float = Form(..., description="Physical edge length of the marker (mm)"),
):
    """Detect the marker, warp around it, and auto-detect the hole polygon.

    Returns JSON:
      image              : base64 JPEG of the perspective-corrected top-down view
      px_per_mm          : pixel scale of that image
      view_size_mm       : extent of that image in mm (square)
      marker_corners_px  : [TL, TR, BR, BL] coords of the marker in the image
      hole_polygon_px    : list of [x, y] auto-detected hole vertices, or null
    """
    if not (_MIN_MARKER_MM <= marker_size_mm <= _MAX_MARKER_MM):
        raise HTTPException(
            status_code=422,
            detail=f"marker_size_mm must be between {_MIN_MARKER_MM} and {_MAX_MARKER_MM}.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image is empty.")
    if len(image_bytes) > _MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 20 MB limit.")

    try:
        result = detect_and_warp(
            image_bytes,
            marker_size_mm=marker_size_mm,
            view_size_mm=DEFAULT_VIEW_MM,
            px_per_mm=DEFAULT_PX_PER_MM,
        )
    except ValueError as exc:
        logger.error("ArUco detection failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))

    warped     = result["warped_bgr"]
    marker_px  = result["marker_corners_px"]
    px_per_mm  = result["px_per_mm"]

    polygon = detect_hole_polygon(warped, marker_px, px_per_mm)

    ok, buf = cv2.imencode(".jpg", warped, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode warped image.")

    return {
        "image":             base64.b64encode(buf.tobytes()).decode(),
        "px_per_mm":         px_per_mm,
        "view_size_mm":      result["view_size_mm"],
        "marker_corners_px": marker_px,
        "marker_size_mm":    result["marker_size_mm"],
        "hole_polygon_px":   polygon,
    }


@router.post("/generate")
async def generate(
    mask: UploadFile = File(..., description="Binary PNG mask \u2014 white pixels = hole"),
    px_per_mm:    float = Form(..., description="Pixel scale of the mask (must match /prepare)"),
    depth_mm:     float = Form(..., description="Hole depth in mm"),
    tolerance_mm: float = Form(0.2,  description="Inward perimeter offset in mm"),
):
    """Convert the painted hole mask into a ribbed plug 3MF."""
    if not (_MIN_DEPTH_MM <= depth_mm <= _MAX_DEPTH_MM):
        raise HTTPException(
            status_code=422,
            detail=f"depth_mm must be between {_MIN_DEPTH_MM} and {_MAX_DEPTH_MM}.",
        )
    if not (_MIN_TOLERANCE_MM <= tolerance_mm <= _MAX_TOLERANCE_MM):
        raise HTTPException(
            status_code=422,
            detail=f"tolerance_mm must be between {_MIN_TOLERANCE_MM} and {_MAX_TOLERANCE_MM}.",
        )
    if px_per_mm <= 0:
        raise HTTPException(status_code=422, detail="px_per_mm must be > 0.")

    mask_bytes = await mask.read()
    if not mask_bytes:
        raise HTTPException(status_code=422, detail="Mask is empty.")

    nparr = np.frombuffer(mask_bytes, np.uint8)
    mask_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise HTTPException(status_code=422, detail="Could not decode mask image.")

    _, binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise HTTPException(
            status_code=422,
            detail="No hole region painted. Mark the hole outline first.",
        )

    largest = max(contours, key=cv2.contourArea)
    eps = max(1.0, 0.003 * cv2.arcLength(largest, True))
    approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        raise HTTPException(status_code=422, detail="Painted region is too small.")

    polygon_mm = [[float(x) / px_per_mm, float(y) / px_per_mm] for x, y in approx]

    try:
        file_bytes = generate_plug_3mf(polygon_mm, depth_mm=depth_mm, tolerance_mm=tolerance_mm)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Plug generation failed")
        raise HTTPException(status_code=500, detail=f"Plug generation failed: {exc}")

    logger.info(
        "Generated plug 3MF: depth %.2f mm, tolerance %.2f mm, %d vertices, %.1f KB",
        depth_mm, tolerance_mm, len(polygon_mm), len(file_bytes) / 1024,
    )

    return Response(
        content=file_bytes,
        media_type="model/3mf",
        headers={"Content-Disposition": 'attachment; filename="hole_plug.3mf"'},
    )
