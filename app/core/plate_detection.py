"""
OpenCV-based build plate corner detection.

Detection order (first success wins):
  1. Canny on grayscale — fast, works well on high-contrast plates.
  2. Canny on enhanced S-channel — fallback for low-contrast plates such as the
     Bambu Cool Plate SuperTack.  Saturation is boosted ×3 and lightness is
     compressed, making even nearly-gray plates stand out against the background.

Debug mode additionally tries Hough lines, colour-distance masking (LAB / A+B),
HSV histogram backprojection, and Otsu thresholding.

Validation applied to every candidate:
  - Must be convex
  - All interior angles 50°–130°
  - Must span ≥ 25 % of both image dimensions
  - Aspect ratio within 40 % of known plate ratio (when provided)
"""

from __future__ import annotations

import base64

import cv2
import numpy as np

_WORK_MAX_PX    = 1000
_EPSILON_FACTORS = [0.01, 0.02, 0.03, 0.05, 0.08]
_MIN_AREA_RATIO  = 0.05

# Centre patch used to sample the plate colour (fraction of each dimension)
_CENTER_PATCH_F  = 0.08

# ΔE thresholds — full LAB (L+A+B), sensitive to lighting differences
_COLOR_THRESHOLDS = [20, 30, 40, 55, 70]

# ΔE thresholds — chrominance only (A+B), lighting-invariant
_AB_THRESHOLDS = [10, 15, 22, 30, 42]

# Probability thresholds for HSV histogram backprojection (0–255)
_BACKPROJ_THRESHOLDS = [25, 50, 80, 120, 160]


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


def _ab_distance_mask(img_lab: np.ndarray, plate_color: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binary mask using only the A and B chrominance channels of LAB.

    Ignores L (luminance) entirely, so lighting gradients and shadows across
    the plate surface do not affect the match.  Two pixels with identical hue
    but different brightness both pass if their A+B distance is within threshold.
    """
    diff_a = img_lab[:, :, 1].astype(np.float32) - float(plate_color[1])
    diff_b = img_lab[:, :, 2].astype(np.float32) - float(plate_color[2])
    dist = np.sqrt(diff_a ** 2 + diff_b ** 2)
    return (dist <= threshold).astype(np.uint8) * 255


def _hsv_backprojection_mask(
    img_hsv: np.ndarray,
    center_patch_hsv: np.ndarray,
    threshold: int,
) -> np.ndarray:
    """
    Histogram backprojection in H+S space.

    Builds a 2-D histogram of Hue × Saturation from the centre patch (which
    covers the plate surface), then assigns each pixel in the full image a
    probability score based on how well its H+S matches the patch histogram.
    Pixels above `threshold` (0–255) are returned as foreground.

    Advantages over single-point distance:
    - Models the full colour *distribution* of the plate, not just the mean.
    - Inherently handles surface texture, subtle print residue, and uneven
      illumination within the patch.
    - Value (brightness) channel is ignored — lighting-invariant.
    """
    hist = cv2.calcHist(
        [center_patch_hsv], [0, 1], None,
        [36, 32],            # 36 hue bins (5° each), 32 sat bins
        [0, 180, 0, 256],
    )
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    prob = cv2.calcBackProject([img_hsv], [0, 1], hist, [0, 180, 0, 256], scale=1)

    # Smooth with a small disc kernel to fill gaps from specular highlights
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cv2.filter2D(prob, -1, disc, prob)

    _, mask = cv2.threshold(prob, threshold, 255, cv2.THRESH_BINARY)
    return mask


def _enhance_chromatic(bgr: np.ndarray) -> np.ndarray:
    """
    Replicate the paint.net effect that makes gray build plates pop:
      contrast = -100  →  compress Lightness 90 % towards 128 (removes lighting gradients)
      saturation = +200 →  multiply Saturation by 3 (amplifies hue differences)

    Net effect:
    - A neutral-gray plate (S ≈ 0) keeps S ≈ 0 → remains a cool, desaturated region.
    - A warm wood/desk background (orange, S > 0) gets S × 3 → vivid orange.
    - Lighting hotspots and shadows are suppressed by the L compression.

    This creates maximum hue contrast between the plate and any coloured background,
    making HSV histogram backprojection on the result extremely effective.
    """
    hls = cv2.cvtColor(bgr, cv2.COLOR_BGR2HLS).astype(np.float32)
    # Saturation ×3  (paint.net "+200 %")
    hls[:, :, 2] = np.clip(hls[:, :, 2] * 3.0, 0, 255)
    # Contrast compression: pull L 90 % towards mid-gray (paint.net "−100")
    hls[:, :, 1] = 128.0 + (hls[:, :, 1] - 128.0) * 0.1
    return cv2.cvtColor(hls.astype(np.uint8), cv2.COLOR_HLS2BGR)


# (low, high) Canny threshold pairs — permissive → strict
_CANNY_PARAMS = [(20, 60), (30, 100), (50, 150)]


def _line_intersection(
    p1: tuple, p2: tuple, p3: tuple, p4: tuple
) -> np.ndarray | None:
    """Intersection point of the infinite lines through (p1,p2) and (p3,p4)."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    x3, y3 = float(p3[0]), float(p3[1])
    x4, y4 = float(p4[0]), float(p4[1])
    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(det) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / det
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)


def _canny_candidates(
    gray: np.ndarray,
    work_w: int,
    work_h: int,
    min_area: float,
    expected_ratio: float | None,
) -> list[tuple[np.ndarray, float]]:
    """
    Canny edge detection → dilate → findContours → approxPolyDP.

    Tries multiple (low, high) threshold pairs and returns all valid
    quadrilateral candidates sorted by centroid distance to image centre.
    Works directly on a grayscale (or single-channel) image.
    """
    img_center = np.array([work_w / 2.0, work_h / 2.0])
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    all_candidates: list[tuple[np.ndarray, float]] = []
    seen: set[tuple] = set()

    for t_low, t_high in _CANNY_PARAMS:
        edges = cv2.Canny(blurred, t_low, t_high)
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            for eps_f in _EPSILON_FACTORS:
                approx = cv2.approxPolyDP(cnt, eps_f * peri, True)
                if len(approx) == 4:
                    quad = approx.reshape(4, 2).astype(np.float32)
                    key  = tuple(np.round(quad / 5).astype(int).flatten())
                    if key not in seen and _validate_quad(quad, work_w, work_h, expected_ratio):
                        seen.add(key)
                        dist = float(np.linalg.norm(quad.mean(axis=0) - img_center))
                        all_candidates.append((quad, dist))
                    break

    return sorted(all_candidates, key=lambda x: x[1])


def _hough_candidates(
    gray: np.ndarray,
    work_w: int,
    work_h: int,
    min_area: float,
    expected_ratio: float | None,
) -> list[tuple[np.ndarray, float]]:
    """
    HoughLinesP → cluster by angle → two most-separated lines per group
    → four intersection corners → validate quad.

    The plate is a rectangle, so two pairs of roughly-parallel lines should
    dominate the Hough space; intersecting them gives the four corners without
    relying on color at all.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, 30, 100)
    min_len = min(work_w, work_h) * 0.15
    lines   = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=max(30, int(min_len * 0.5)),
        minLineLength=int(min_len),
        maxLineGap=int(min(work_w, work_h) * 0.05),
    )
    if lines is None or len(lines) < 4:
        return []

    # Angle (0–180°), length, endpoints for each detected segment
    line_info = []
    for seg in lines:
        x1, y1, x2, y2 = seg[0]
        angle  = np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))) % 180
        length = np.hypot(x2 - x1, y2 - y1)
        line_info.append((angle, length, (x1, y1), (x2, y2)))

    # Length-weighted circular mean of doubled angles → dominant direction
    a2  = np.array([np.radians(d[0] * 2) for d in line_info])
    w   = np.array([d[1] for d in line_info])
    dom = np.degrees(np.arctan2((np.sin(a2) * w).sum(), (np.cos(a2) * w).sum())) / 2 % 180

    # Split into two groups: parallel and perpendicular to dominant direction
    group_a, group_b = [], []
    for info in line_info:
        diff = abs(((info[0] - dom + 90) % 180) - 90)
        (group_a if diff < 45 else group_b).append(info)
    if len(group_a) < 2 or len(group_b) < 2:
        return []

    def _two_extremes(group, normal_deg):
        nx, ny = np.cos(np.radians(normal_deg)), np.sin(np.radians(normal_deg))
        projs = [
            (((d[2][0] + d[3][0]) / 2 * nx + (d[2][1] + d[3][1]) / 2 * ny), d[2], d[3])
            for d in group
        ]
        projs.sort(key=lambda p: p[0])
        return (projs[0][1], projs[0][2]), (projs[-1][1], projs[-1][2])

    (a1p1, a1p2), (a2p1, a2p2) = _two_extremes(group_a, (dom + 90) % 180)
    (b1p1, b1p2), (b2p1, b2p2) = _two_extremes(group_b, dom)

    corners = []
    for ap1, ap2 in ((a1p1, a1p2), (a2p1, a2p2)):
        for bp1, bp2 in ((b1p1, b1p2), (b2p1, b2p2)):
            pt = _line_intersection(ap1, ap2, bp1, bp2)
            if pt is not None:
                corners.append(pt)
    if len(corners) != 4:
        return []

    quad = np.array(corners, dtype=np.float32)
    if not _validate_quad(quad, work_w, work_h, expected_ratio):
        return []

    img_center = np.array([work_w / 2.0, work_h / 2.0])
    dist = float(np.linalg.norm(quad.mean(axis=0) - img_center))
    return [(quad, dist)]


def _otsu_mask(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    """Otsu thresholding; optionally invert for light-coloured plates."""
    flags = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    if invert:
        flags = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    _, mask = cv2.threshold(gray, 0, 255, flags)
    return mask


def _cleanup_mask(mask: np.ndarray) -> np.ndarray:
    """Morphological close + open to fill holes and remove small blobs."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)


def _find_all_quads(
    clean_mask: np.ndarray,
    work_w: int,
    work_h: int,
    min_area: float,
    expected_ratio: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """
    Return all valid quadrilaterals found in an already-cleaned binary mask,
    sorted by centroid distance to the image centre (closest first).

    Each element is (quad_array_4x2_float32, centroid_dist_px).
    """
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_center = np.array([work_w / 2.0, work_h / 2.0])
    results: list[tuple[np.ndarray, float]] = []

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        quad = None
        perimeter = cv2.arcLength(contour, True)
        for eps in _EPSILON_FACTORS:
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                break

        if quad is None:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype(np.float32)

        if not _validate_quad(quad, work_w, work_h, expected_ratio):
            continue

        dist = float(np.linalg.norm(quad.mean(axis=0) - img_center))
        results.append((quad, dist))

    return sorted(results, key=lambda x: x[1])


def _best_centered_quad(
    mask: np.ndarray,
    work_w: int,
    work_h: int,
    min_area: float,
    expected_ratio: float | None = None,
) -> np.ndarray | None:
    """Return the most-centred valid quad from the mask, or None."""
    clean = _cleanup_mask(mask)
    candidates = _find_all_quads(clean, work_w, work_h, min_area, expected_ratio)
    return candidates[0][0] if candidates else None


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

    # Strategy 1: Canny on grayscale
    candidates = _canny_candidates(gray, work_w, work_h, min_area, expected_ratio)
    quad = candidates[0][0] if candidates else None

    # Strategy 2: Canny on enhanced S-channel (fallback for low-contrast plates
    # such as the Bambu Cool Plate SuperTack where grayscale edges are too faint)
    if quad is None:
        enhanced_sat = cv2.cvtColor(_enhance_chromatic(work), cv2.COLOR_BGR2HLS)[:, :, 2]
        candidates = _canny_candidates(enhanced_sat, work_w, work_h, min_area, expected_ratio)
        quad = candidates[0][0] if candidates else None

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


# ── Debug helpers ─────────────────────────────────────────────────────────────

_DEBUG_MAX_PX = 800


def _make_mask_step(
    title: str,
    mask: np.ndarray,
    work: np.ndarray,
    work_w: int,
    work_h: int,
    min_area: float,
    expected_ratio: float | None,
    final_quad_box: list,          # single-element list used as a mutable reference
) -> dict:
    """
    Build one debug step dict for a given binary mask.

    Visualisation: matched pixels at full brightness, rest darkened to 25%.
    Valid candidate quads are drawn (green = most-centred winner, blue = others).
    `final_quad_box` is a 1-element list; if it is [None] and candidates are
    found, the winner is stored there so callers know detection succeeded.
    """
    clean      = _cleanup_mask(mask)
    candidates = _find_all_quads(clean, work_w, work_h, min_area, expected_ratio)

    darkened = (work * 0.25).astype(np.uint8)
    overlay  = np.where(clean[:, :, np.newaxis] > 0, work, darkened)
    _draw_crosshair(overlay, work_w // 2, work_h // 2)
    _draw_quads(overlay, candidates)

    pct = int((clean > 0).mean() * 100)
    n   = len(candidates)

    if n > 0 and final_quad_box[0] is None:
        final_quad_box[0] = candidates[0][0]
        desc = (f"✅ {pct}% pixels matched · {n} valid quad(s) · "
                f"winner (green) {candidates[0][1]:.0f} px from centre")
    else:
        suffix = " · already solved at earlier step" if final_quad_box[0] is not None else ""
        desc = f"{pct}% pixels matched · {n} valid quad(s){suffix}"

    return {"title": title, "image": _encode_b64(overlay), "description": desc}


def _make_edge_step(
    title: str,
    candidates: list[tuple[np.ndarray, float]],
    edge_img: np.ndarray,            # 1-channel edge/gradient image for background
    work: np.ndarray,                # original colour image (same size)
    work_w: int,
    work_h: int,
    final_quad_box: list,
) -> dict:
    """
    Build one debug step dict for Canny / Hough results.

    Background: edge map converted to BGR (white edges on black), with the
    original image blended in at 30 % so spatial context is preserved.
    Candidate quads drawn on top.
    """
    edge_bgr = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR) if edge_img.ndim == 2 else edge_img
    overlay  = cv2.addWeighted(edge_bgr, 0.7, work, 0.3, 0)
    _draw_crosshair(overlay, work_w // 2, work_h // 2)
    _draw_quads(overlay, candidates)

    n = len(candidates)
    if n > 0 and final_quad_box[0] is None:
        final_quad_box[0] = candidates[0][0]
        desc = (f"✅ {n} valid quad(s) · "
                f"winner (green) {candidates[0][1]:.0f} px from centre")
    else:
        suffix = " · already solved at earlier step" if final_quad_box[0] is not None else ""
        desc = f"{n} valid quad(s){suffix}"

    return {"title": title, "image": _encode_b64(overlay), "description": desc}


def _encode_b64(img: np.ndarray) -> str:
    """Encode a BGR numpy image to a base64 JPEG string, capped at 800 px on the long side."""
    h, w = img.shape[:2]
    if max(h, w) > _DEBUG_MAX_PX:
        s = _DEBUG_MAX_PX / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def _draw_crosshair(img: np.ndarray, cx: int, cy: int, color=(0, 0, 255), size: int = 20) -> None:
    cv2.line(img, (cx - size, cy), (cx + size, cy), color, 2)
    cv2.line(img, (cx, cy - size), (cx, cy + size), color, 2)
    cv2.circle(img, (cx, cy), 6, color, 2)


def _draw_quads(img: np.ndarray, candidates: list[tuple[np.ndarray, float]]) -> None:
    """Draw all candidates; green = winner (index 0), blue = others."""
    for i, (quad, dist) in enumerate(candidates):
        color = (0, 200, 0) if i == 0 else (255, 140, 0)
        thickness = 3 if i == 0 else 1
        pts = quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, thickness)
        cx, cy = quad.mean(axis=0).astype(int)
        cv2.circle(img, (int(cx), int(cy)), 5, color, -1)
        label = f"#{i} d={dist:.0f}px"
        cv2.putText(img, label, (int(cx) + 7, int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def detect_plate_corners_debug(
    image_bytes: bytes,
    plate_width_mm: float,
    plate_height_mm: float,
) -> dict:
    """
    Run the two-strategy corner detection pipeline and return annotated step images.

    Strategies (matching the production detect_plate_corners function):
      A — Canny on original grayscale
      B — Canny on enhanced S-channel (fallback)

    Returns
    -------
    {
        "steps":   [{"title": str, "image": base64_str, "description": str}, ...],
        "corners": {top_left, top_right, bottom_right, bottom_left} | None,
        "error":   str | None,
    }
    """
    steps: list[dict] = []

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"steps": [], "corners": None, "error": "Could not decode image."}

    orig_h, orig_w = img.shape[:2]
    scale    = min(_WORK_MAX_PX / orig_w, _WORK_MAX_PX / orig_h, 1.0)
    work     = cv2.resize(img, (int(orig_w * scale), int(orig_h * scale)))
    work_h, work_w = work.shape[:2]
    min_area = work_w * work_h * _MIN_AREA_RATIO
    expected_ratio = max(plate_width_mm, plate_height_mm) / min(plate_width_mm, plate_height_mm)

    gray         = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    enhanced     = _enhance_chromatic(work)
    enhanced_hls = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HLS)
    enhanced_sat = enhanced_hls[:, :, 2]   # S channel: plate=dark, BG=bright

    # ── Step 1: original image ────────────────────────────────────────────────
    vis1 = work.copy()
    _draw_crosshair(vis1, work_w // 2, work_h // 2)
    steps.append({"title": "Step 1 — Input image",
                  "image": _encode_b64(vis1),
                  "description": "Red crosshair = image centre."})

    # ── Step 2: chromatic-enhanced image ─────────────────────────────────────
    vis2 = enhanced.copy()
    _draw_crosshair(vis2, work_w // 2, work_h // 2)
    steps.append({"title": "Step 2 — Chromatic enhancement (contrast −100, sat ×3)",
                  "image": _encode_b64(vis2),
                  "description": (
                      "Saturation boosted ×3, lightness compressed towards 128. "
                      "Neutral-gray plates become clearly distinct from coloured backgrounds."
                  )})

    # ── Step 3: enhanced saturation channel ──────────────────────────────────
    sat_bgr = cv2.cvtColor(enhanced_sat, cv2.COLOR_GRAY2BGR)
    _draw_crosshair(sat_bgr, work_w // 2, work_h // 2)
    steps.append({"title": "Step 3 — Enhanced saturation channel (S)",
                  "image": _encode_b64(sat_bgr),
                  "description": (
                      "S channel from the enhanced HLS image. "
                      "Plate = dark (S≈0), coloured background = bright. "
                      "Canny on this channel finds the plate boundary as the sharpest edge."
                  )})

    # Mutable box so helpers can record the first winner found
    fq = [None]   # fq[0] = winning quad once found, else None

    # ── Strategy A: Canny on original grayscale ───────────────────────────────
    canny_gray = _canny_candidates(gray, work_w, work_h, min_area, expected_ratio)
    edge_gray  = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    steps.append(_make_edge_step(
        "A — Canny on original grayscale", canny_gray, edge_gray, work, work_w, work_h, fq,
    ))

    # ── Strategy B: Canny on enhanced S-channel (fallback) ───────────────────
    canny_esat   = _canny_candidates(enhanced_sat, work_w, work_h, min_area, expected_ratio)
    edge_esat    = cv2.Canny(cv2.GaussianBlur(enhanced_sat, (5, 5), 0), 30, 100)
    steps.append(_make_edge_step(
        "B — Canny on enhanced S-channel", canny_esat, edge_esat, work, work_w, work_h, fq,
    ))

    final_quad = fq[0]

    # ── Final result ──────────────────────────────────────────────────────────
    if final_quad is not None:
        result = work.copy()
        pts = final_quad.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(result, [pts], True, (0, 220, 0), 3)
        for pt in final_quad:
            cv2.circle(result, tuple(pt.astype(int)), 9, (0, 220, 0), -1)
        _draw_crosshair(result, work_w // 2, work_h // 2)
        steps.append({"title": "Final result — detected plate boundary",
                      "image": _encode_b64(result),
                      "description": "Green polygon = plate corners passed to perspective correction."})

        corners = _order_corners(final_quad / scale)
        return {
            "steps": steps,
            "corners": {
                "top_left":     corners[0].tolist(),
                "top_right":    corners[1].tolist(),
                "bottom_right": corners[2].tolist(),
                "bottom_left":  corners[3].tolist(),
            },
            "error": None,
        }

    return {"steps": steps, "corners": None,
            "error": "No valid plate-shaped region found in any detection step."}

