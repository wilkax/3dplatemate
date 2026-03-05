"""
OpenCV-based build plate corner detection.

Strategy:
  1. Sample the dominant colour from the image centre — assuming the build plate
     occupies the middle of the frame, that patch IS the plate surface.
  2. Build a colour-distance mask in LAB space: pixels within N ΔE units of the
     sampled plate colour are marked as foreground. Tried at several thresholds.
  3. Extract every quadrilateral candidate from the mask contours and score each
     one by how close its centroid is to the image centre.  The most-centred
     valid quad wins — not just the first one that passes shape checks.
  4. Fallback: Otsu thresholding (normal then inverted), same centroid scoring.

Validation applied to every candidate:
  - Must be convex
  - All interior angles 50°–130°
  - Must span ≥ 25 % of both image dimensions
  - Aspect ratio within 40 % of known plate ratio (when provided)
"""

from __future__ import annotations

import cv2
import numpy as np

_WORK_MAX_PX    = 1000
_EPSILON_FACTORS = [0.01, 0.02, 0.03, 0.05, 0.08]
_MIN_AREA_RATIO  = 0.05

# Centre patch used to sample the plate colour (fraction of each dimension)
_CENTER_PATCH_F  = 0.08

# ΔE thresholds to try when building the colour-distance mask (LAB units)
_COLOR_THRESHOLDS = [20, 30, 40, 55, 70]


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def _validate_quad(
    quad: np.ndarray,
    work_w: int,
    work_h: int,
    expected_ratio: float | None = None,
) -> bool:
    """
    Return True only if the quad looks like a plausible plate rectangle.

    Checks (in order):
      1. Convexity — non-convex shapes are never a flat plate.
      2. Interior angles — all must be between 50° and 130°.
      3. Spread — bounding box must cover ≥ 25 % of both image dimensions.
      4. Aspect ratio — detected ratio must be within 40 % of the expected
         plate ratio (checked in both landscape and portrait orientations).
         Skipped when expected_ratio is None.
    """
    # 1. Convexity
    contour = quad.reshape(-1, 1, 2).astype(np.int32)
    if not cv2.isContourConvex(contour):
        return False

    # 2. Interior angles
    for i in range(4):
        p1 = quad[(i - 1) % 4]
        p2 = quad[i]
        p3 = quad[(i + 1) % 4]
        v1 = p1 - p2
        v2 = p3 - p2
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
        cos_a = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_a))
        if not (50.0 <= angle <= 130.0):
            return False

    # 3. Spread — use bounding box of the quad corners
    span_x = quad[:, 0].max() - quad[:, 0].min()
    span_y = quad[:, 1].max() - quad[:, 1].min()
    if span_x < 0.25 * work_w or span_y < 0.25 * work_h:
        return False

    # 4. Aspect ratio (optional)
    if expected_ratio is not None:
        # Use average of opposite side lengths for a tilt-robust ratio
        sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        long_side  = (sides[0] + sides[2]) / 2
        short_side = (sides[1] + sides[3]) / 2
        if short_side < 1:
            return False
        detected = long_side / short_side
        # Accept landscape or portrait match within 40 %
        tolerance = 0.40
        landscape_ok = abs(detected - expected_ratio) / expected_ratio <= tolerance
        portrait_ok  = abs(detected - 1.0 / expected_ratio) / (1.0 / expected_ratio) <= tolerance
        if not (landscape_ok or portrait_ok):
            return False

    return True


def _sample_plate_color(img_lab: np.ndarray) -> np.ndarray:
    """Return the mean LAB colour of the centre patch of the image."""
    h, w = img_lab.shape[:2]
    ph = max(1, int(h * _CENTER_PATCH_F))
    pw = max(1, int(w * _CENTER_PATCH_F))
    cy, cx = h // 2, w // 2
    patch = img_lab[cy - ph // 2 : cy + ph // 2, cx - pw // 2 : cx + pw // 2]
    return patch.reshape(-1, 3).mean(axis=0)


def _color_distance_mask(img_lab: np.ndarray, plate_color: np.ndarray, threshold: float) -> np.ndarray:
    """Binary mask of pixels within `threshold` ΔE of plate_color."""
    diff = img_lab.astype(np.float32) - plate_color.astype(np.float32)
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return (dist <= threshold).astype(np.uint8) * 255


def _otsu_mask(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    """Otsu thresholding; optionally invert for light-coloured plates."""
    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    if invert:
        flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    _, mask = cv2.threshold(gray, 0, 255, flags)
    return mask


def _best_centered_quad(
    mask: np.ndarray,
    work_w: int,
    work_h: int,
    min_area: float,
    expected_ratio: float | None = None,
) -> np.ndarray | None:
    """
    Find every quadrilateral in the mask and return the one whose centroid
    is closest to the image centre, provided it passes validation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_center = np.array([work_w / 2.0, work_h / 2.0])
    best_quad  = None
    best_dist  = float("inf")

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # Try polygon approximation at increasing tolerances
        quad = None
        perimeter = cv2.arcLength(contour, True)
        for eps in _EPSILON_FACTORS:
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                break

        if quad is None:
            # minAreaRect always produces 4 corners — use as last resort
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype(np.float32)

        if not _validate_quad(quad, work_w, work_h, expected_ratio):
            continue

        centroid = quad.mean(axis=0)
        dist = np.linalg.norm(centroid - img_center)
        if dist < best_dist:
            best_dist = dist
            best_quad = quad

    return best_quad


def detect_plate_corners(
    image_bytes: bytes,
    plate_width_mm: float | None = None,
    plate_height_mm: float | None = None,
) -> dict:
    """
    Detect the 4 corners of the build plate using color/region segmentation.

    Parameters
    ----------
    image_bytes     : raw image bytes (JPEG / PNG / WEBP)
    plate_width_mm  : known plate width — used to validate aspect ratio
    plate_height_mm : known plate height — used to validate aspect ratio

    Returns
    -------
    dict with keys top_left, top_right, bottom_right, bottom_left — each [x, y]
    in the original image's pixel coordinates.

    Raises
    ------
    ValueError : if no valid plate-shaped region is found.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")

    orig_h, orig_w = img.shape[:2]
    scale = min(_WORK_MAX_PX / orig_w, _WORK_MAX_PX / orig_h, 1.0)
    work = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
    work_h, work_w = work.shape[:2]
    min_area = work_w * work_h * _MIN_AREA_RATIO

    expected_ratio: float | None = None
    if plate_width_mm and plate_height_mm:
        expected_ratio = max(plate_width_mm, plate_height_mm) / min(plate_width_mm, plate_height_mm)

    img_lab  = cv2.cvtColor(work, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray     = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred  = cv2.GaussianBlur(gray, (7, 7), 0)

    plate_color = _sample_plate_color(img_lab)
    quad = None

    # ── 1. Colour-distance mask at increasing ΔE thresholds ──────────────────
    for threshold in _COLOR_THRESHOLDS:
        mask = _color_distance_mask(img_lab, plate_color, threshold)
        quad = _best_centered_quad(mask, work_w, work_h, min_area, expected_ratio)
        if quad is not None:
            break

    # ── 2. Otsu fallback (dark plate on light bg, then inverted) ─────────────
    if quad is None:
        for invert in (False, True):
            mask = _otsu_mask(blurred, invert=invert)
            quad = _best_centered_quad(mask, work_w, work_h, min_area, expected_ratio)
            if quad is not None:
                break

    if quad is None:
        raise ValueError(
            "Could not detect plate corners — no valid plate-shaped region found. "
            "Ensure the full plate with all 4 edges clearly visible, good contrast "
            "against the background, and even lighting with no strong shadows."
        )

    corners = _order_corners(quad / scale)

    return {
        "top_left":     corners[0].tolist(),
        "top_right":    corners[1].tolist(),
        "bottom_right": corners[2].tolist(),
        "bottom_left":  corners[3].tolist(),
    }

