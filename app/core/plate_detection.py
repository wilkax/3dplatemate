"""
OpenCV-based build plate corner detection using color/region segmentation.

Strategy (in order of attempt):
  1. GrabCut — treats image border as background, center as foreground;
     iteratively segments by color distribution. Works well when the plate
     is roughly centered and has a different color from the surroundings.
     Tried with progressively larger border margins.
  2. Otsu thresholding — automatic brightness-based separation; tried on
     both normal and inverted grayscale to handle light and dark plates.
  3. minAreaRect fallback — fits a rectangle to the largest region found
     by any method; always produces 4 corners.

Every candidate quad is validated before being accepted:
  - Must be convex
  - All interior angles between 50° and 130°
  - Must span ≥ 25 % of both image dimensions
  - Aspect ratio must be within 40 % of the known plate ratio (when provided)
"""

from __future__ import annotations

import cv2
import numpy as np

_WORK_MAX_PX = 1000
_EPSILON_FACTORS = [0.01, 0.02, 0.03, 0.05, 0.08]
_MIN_AREA_RATIO = 0.05

# GrabCut border margins to try (fraction of image dimension)
_GRABCUT_MARGINS = [0.08, 0.13, 0.20]


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


def _mask_to_quad(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    """
    Extract the largest quadrilateral from a binary mask.
    Applies morphological cleanup first, then tries multiple epsilon values.
    Returns a (4, 2) float32 array or None.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        if cv2.contourArea(contour) < min_area:
            break
        perimeter = cv2.arcLength(contour, True)
        for eps in _EPSILON_FACTORS:
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

    # minAreaRect fallback: always produces 4 corners
    if contours and cv2.contourArea(contours[0]) >= min_area:
        rect = cv2.minAreaRect(contours[0])
        return cv2.boxPoints(rect).astype(np.float32)

    return None


def _grabcut_mask(img: np.ndarray, margin: float) -> np.ndarray:
    """Run GrabCut and return a binary foreground mask."""
    h, w = img.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    rect = (mx, my, w - 2 * mx, h - 2 * my)

    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgd, fgd, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def _otsu_mask(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    """Otsu thresholding; optionally invert for light-coloured plates."""
    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    if invert:
        flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    _, mask = cv2.threshold(gray, 0, 255, flags)
    return mask


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

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    quad = None

    # ── 1. GrabCut with multiple border margins ───────────────────────────────
    for margin in _GRABCUT_MARGINS:
        try:
            mask = _grabcut_mask(work, margin)
            candidate = _mask_to_quad(mask, min_area)
            if candidate is not None and _validate_quad(candidate, work_w, work_h, expected_ratio):
                quad = candidate
                break
        except Exception:
            continue

    # ── 2. Otsu thresholding (dark plate, then light plate) ───────────────────
    if quad is None:
        for invert in (False, True):
            mask = _otsu_mask(blurred, invert=invert)
            candidate = _mask_to_quad(mask, min_area)
            if candidate is not None and _validate_quad(candidate, work_w, work_h, expected_ratio):
                quad = candidate
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

