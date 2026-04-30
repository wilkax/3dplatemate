"""
Auto-detect the outline of a hole in the perspective-corrected top-down view.

The ArUco-warped image (see app.core.aruco_detection.detect_and_warp) shows
the marker centred on a square mm grid.  Holes typically appear as the darkest
contiguous region next to the marker because their interior is in shadow
relative to the surrounding floor surface.

Strategy:
  1. Build a "ignore-marker" mask covering the marker plus a small safety
     border so the dark marker squares do not contaminate the hole search.
  2. Convert to grayscale and apply Otsu thresholding (inverted): everything
     darker than Otsu's threshold becomes white.
  3. Morphological close to fill small bright speckles inside the hole.
  4. Find external contours, drop those overlapping the marker region or too
     small, return the largest remaining contour as a polygon (in corrected
     pixel coordinates).

If nothing usable is found, returns None — the frontend then expects the user
to draw the polygon manually.
"""

from __future__ import annotations

import cv2
import numpy as np

_MIN_AREA_FRACTION    = 0.005   # >= 0.5% of view area to count as a hole
_MARKER_PADDING_MM    = 4.0     # safety border around marker (mm)
_CONTOUR_EPS_FACTOR   = 0.005   # approxPolyDP epsilon as a fraction of arclen


def _marker_mask(
    shape_hw: tuple[int, int],
    marker_corners_px: list[list[float]],
    px_per_mm: float,
) -> np.ndarray:
    """White (255) inside the marker + padding, black (0) elsewhere."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts  = np.asarray(marker_corners_px, dtype=np.float32)
    centre = pts.mean(axis=0)
    pad_px = _MARKER_PADDING_MM * px_per_mm
    expanded = np.empty_like(pts)
    for i, p in enumerate(pts):
        v = p - centre
        d = np.linalg.norm(v)
        if d < 1e-6:
            expanded[i] = p
        else:
            expanded[i] = p + v / d * pad_px
    cv2.fillConvexPoly(mask, expanded.astype(np.int32), 255)
    return mask


def detect_hole_polygon(
    warped_bgr: np.ndarray,
    marker_corners_px: list[list[float]],
    px_per_mm: float,
) -> list[list[float]] | None:
    """Return the auto-detected hole polygon in warped-image pixel coords.

    Parameters
    ----------
    warped_bgr        : BGR image from aruco_detection.detect_and_warp
    marker_corners_px : 4 [x, y] coords of the marker in this image
    px_per_mm         : scale of the warped image

    Returns
    -------
    list of [x, y] vertices, or None if no plausible hole was found.
    """
    if warped_bgr is None or warped_bgr.size == 0:
        return None

    h, w = warped_bgr.shape[:2]
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, dark = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Suppress the marker (and its surrounding quiet zone) so it cannot win
    marker = _marker_mask((h, w), marker_corners_px, px_per_mm)
    dark[marker > 0] = 0

    # Close to fill small specular bright spots inside the hole
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(3, int(round(2 * px_per_mm)) | 1),
         max(3, int(round(2 * px_per_mm)) | 1)),
    )
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = h * w * _MIN_AREA_FRACTION
    image_area = float(h * w)

    best: tuple[float, np.ndarray] | None = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        # Skip contours that touch the image border by more than half their length;
        # a real hole inside the view should not run off the edge.
        x, y, cw, ch = cv2.boundingRect(cnt)
        touches_border = (x <= 1) or (y <= 1) or (x + cw >= w - 1) or (y + ch >= h - 1)
        if touches_border and area < image_area * 0.5:
            # small border-hugging blob is almost certainly not the hole
            continue
        if best is None or area > best[0]:
            best = (area, cnt)

    if best is None:
        return None

    cnt = best[1]
    eps = max(1.0, _CONTOUR_EPS_FACTOR * cv2.arcLength(cnt, True))
    approx = cv2.approxPolyDP(cnt, eps, True)
    polygon = approx.reshape(-1, 2).astype(float).tolist()
    if len(polygon) < 3:
        return None
    return polygon
