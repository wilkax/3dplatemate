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


def detect_plate_corners(image_bytes: bytes) -> dict:
    """
    Detect the 4 corners of the build plate using color/region segmentation.

    Parameters
    ----------
    image_bytes : raw image bytes (JPEG / PNG / WEBP)

    Returns
    -------
    dict with keys top_left, top_right, bottom_right, bottom_left — each [x, y]
    in the original image's pixel coordinates.

    Raises
    ------
    ValueError : if no plate-sized region is found.
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

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    quad = None

    # ── 1. GrabCut with multiple border margins ───────────────────────────────
    for margin in _GRABCUT_MARGINS:
        try:
            mask = _grabcut_mask(work, margin)
            quad = _mask_to_quad(mask, min_area)
            if quad is not None:
                break
        except Exception:
            continue

    # ── 2. Otsu thresholding (dark plate, then light plate) ───────────────────
    if quad is None:
        for invert in (False, True):
            mask = _otsu_mask(blurred, invert=invert)
            quad = _mask_to_quad(mask, min_area)
            if quad is not None:
                break

    if quad is None:
        raise ValueError(
            "Could not detect plate corners — no plate-sized region found. "
            "Ensure the full plate with all 4 edges is visible in the photo."
        )

    corners = _order_corners(quad / scale)

    return {
        "top_left":     corners[0].tolist(),
        "top_right":    corners[1].tolist(),
        "bottom_right": corners[2].tolist(),
        "bottom_left":  corners[3].tolist(),
    }

