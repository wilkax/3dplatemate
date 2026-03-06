"""
Debug artifact storage.

For each request a timestamped folder is created under debug_output/:

  debug_output/
  └── 20260304_095512_a3f1/
      ├── 01_original.<ext>          original uploaded image
      ├── 02_corners.json            OpenCV corner detection result
      ├── 03_corrected.jpg           perspective-corrected top-down view
      ├── 04_spots.json              CV spot detection result
      ├── 05_spots_visualization.jpg corrected view with detected polygons drawn
      └── 06_plate_cleaner.3mf       generated 3MF
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.core.geometry import PX_PER_MM

logger = logging.getLogger(__name__)

_DEBUG_ROOT = Path("debug_output")

# Colours (BGR)
_COLOUR_RAW = (0, 165, 255)     # orange  — raw Claude polygons
_COLOUR_BUF = (0, 200, 80)      # green   — buffered (+margin) polygons


def _make_run_dir() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:4]
    run_dir = _DEBUG_ROOT / f"{ts}_{short_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _ext_from_bytes(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if image_bytes[:4] == b"RIFF":
        return "webp"
    return "jpg"


def _draw_polygons_mm(
    canvas: np.ndarray,
    polygons_mm: list[list[list[float]]],
    colour: tuple[int, int, int],
    px_per_mm: float,
    label_prefix: str = "",
) -> None:
    """Draw filled+outlined polygons (mm coords) onto canvas (px coords)."""
    for i, poly in enumerate(polygons_mm):
        pts = (np.array(poly, dtype=np.float32) * px_per_mm).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], colour)
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        cv2.polylines(canvas, [pts], isClosed=True, color=colour, thickness=2)
        if label_prefix:
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(canvas, f"{label_prefix}{i+1}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def save_debug_artifacts(
    image_bytes: bytes,
    corners_result: dict,
    corrected_img: np.ndarray,
    spots_result: list[dict],
    raw_polygons_mm: list[list[list[float]]],
    buffered_polygons_mm: list[list[list[float]]],
    file_bytes: bytes,
) -> Path:
    """
    Save all intermediate artifacts for one request.

    Parameters
    ----------
    image_bytes          : original uploaded image bytes
    corners_result       : OpenCV corner detection result (dict)
    corrected_img        : top-down warped image (numpy BGR array)
    spots_result         : CV spot detection result (list of spot dicts)
    raw_polygons_mm      : spot polygons before buffering (mm coords)
    buffered_polygons_mm : spot polygons after buffering+clipping (mm coords)
    file_bytes           : generated 3MF file bytes

    Returns
    -------
    Path to the run directory.

    Debug folder layout
    -------------------
    01_original.<ext>          original uploaded image
    02_corners.json            OpenCV corner detection result
    03_corrected.jpg           perspective-corrected top-down view
    04_spots.json              CV spot detection result
    05_spots_visualization.jpg corrected view with polygon overlays
    06_plate_cleaner.3mf       generated 3MF
    """
    run_dir = _make_run_dir()

    try:
        ext = _ext_from_bytes(image_bytes)
        (run_dir / f"01_original.{ext}").write_bytes(image_bytes)

        (run_dir / "02_corners.json").write_text(
            json.dumps(corners_result, indent=2), encoding="utf-8"
        )

        cv2.imwrite(str(run_dir / "03_corrected.jpg"), corrected_img)

        (run_dir / "04_spots.json").write_text(
            json.dumps(spots_result, indent=2), encoding="utf-8"
        )

        vis = corrected_img.copy()
        _draw_polygons_mm(vis, raw_polygons_mm, _COLOUR_RAW, PX_PER_MM, label_prefix="raw ")
        _draw_polygons_mm(vis, buffered_polygons_mm, _COLOUR_BUF, PX_PER_MM, label_prefix="buf ")
        cv2.putText(vis, "orange = raw  |  green = buffered",
                    (8, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(run_dir / "05_spots_visualization.jpg"), vis)

        (run_dir / "06_plate_cleaner.3mf").write_bytes(file_bytes)

        logger.info("Debug artifacts saved to %s", run_dir)

    except Exception as exc:
        logger.warning("Failed to save some debug artifacts: %s", exc)

    return run_dir

