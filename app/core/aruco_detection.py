"""
ArUco-based reference marker detection for the "fill the hole" workflow.

A single 4x4 ArUco marker (DICT_4X4_50, id 0) of known physical size is placed
flat on the same surface as the hole.  Detecting its 4 corners gives us a
homography from image pixels to mm coordinates, which we use to:

  1. Compute a perspective-correcting warp around the marker so the user sees a
     true top-down view of marker + hole at a known px/mm scale.
  2. Convert any polygon traced on that warped image back into mm.
"""

from __future__ import annotations

import cv2
import numpy as np

ARUCO_DICT_ID    = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_ID  = 0          # we always look for marker id 0
DEFAULT_VIEW_MM  = 200.0      # corrected-view extent (square), marker centred
DEFAULT_PX_PER_MM = 5.0       # corrected-view resolution


def render_marker_png(size_mm: float, px_per_mm: float = 20.0,
                      quiet_zone_mm: float = 5.0) -> bytes:
    """Render marker id 0 of the configured dictionary as a printable PNG.

    A white quiet zone is added around the marker so cutting/cropping does not
    destroy the detection border.  A small text label states the physical size.
    """
    if size_mm <= 0:
        raise ValueError("size_mm must be > 0")

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    marker_px  = max(1, int(round(size_mm * px_per_mm)))
    marker     = cv2.aruco.generateImageMarker(aruco_dict, ARUCO_MARKER_ID, marker_px)

    pad = max(1, int(round(quiet_zone_mm * px_per_mm)))
    canvas_px = marker_px + 2 * pad
    canvas = np.full((canvas_px, canvas_px), 255, dtype=np.uint8)
    canvas[pad:pad + marker_px, pad:pad + marker_px] = marker

    label = f"ArUco id 0  -  {size_mm:.1f} mm"
    cv2.putText(canvas, label,
                (pad, canvas_px - max(4, pad // 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("Failed to encode marker PNG.")
    return buf.tobytes()


def _detect_corners(image_bgr: np.ndarray) -> np.ndarray:
    """Return the 4 image-space corners of marker id 0, ordered TL,TR,BR,BL.

    Raises ValueError if the marker is not detected.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    parameters = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    all_corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        raise ValueError("No ArUco marker detected in the image.")

    ids_flat = ids.flatten().tolist()
    if ARUCO_MARKER_ID not in ids_flat:
        raise ValueError(
            f"ArUco marker id {ARUCO_MARKER_ID} not found "
            f"(detected ids: {ids_flat}).",
        )
    idx = ids_flat.index(ARUCO_MARKER_ID)
    # OpenCV returns (1, 4, 2) per marker, ordered TL, TR, BR, BL.
    return all_corners[idx].reshape(4, 2).astype(np.float32)


def detect_and_warp(
    image_bytes: bytes,
    marker_size_mm: float,
    view_size_mm: float = DEFAULT_VIEW_MM,
    px_per_mm:    float = DEFAULT_PX_PER_MM,
) -> dict:
    """Detect the marker, warp around it, return the corrected top-down view.

    Returns
    -------
    dict with keys:
      warped_bgr         : np.ndarray (BGR) of the perspective-corrected view
      px_per_mm          : float, scale of the warped image
      view_size_mm       : float, side length of the warped square (mm)
      marker_corners_px  : list of 4 [x, y] coords of the marker corners in
                           the warped image (TL, TR, BR, BL)
      marker_size_mm     : echoed back

    Raises ValueError on detection or geometry failure.
    """
    if marker_size_mm <= 0:
        raise ValueError("marker_size_mm must be > 0.")
    if view_size_mm <= marker_size_mm:
        raise ValueError("view_size_mm must be larger than marker_size_mm.")

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image.")

    src_corners = _detect_corners(img)

    # Place the marker centred in the requested view, in mm coordinates.
    half_view   = view_size_mm / 2.0
    half_marker = marker_size_mm / 2.0
    dst_mm = np.float32([
        [half_view - half_marker, half_view - half_marker],   # TL
        [half_view + half_marker, half_view - half_marker],   # TR
        [half_view + half_marker, half_view + half_marker],   # BR
        [half_view - half_marker, half_view + half_marker],   # BL
    ])
    dst_px = dst_mm * px_per_mm

    H = cv2.getPerspectiveTransform(src_corners, dst_px)
    out_size = int(round(view_size_mm * px_per_mm))
    warped = cv2.warpPerspective(img, H, (out_size, out_size))

    return {
        "warped_bgr":        warped,
        "px_per_mm":         px_per_mm,
        "view_size_mm":      view_size_mm,
        "marker_corners_px": dst_px.tolist(),
        "marker_size_mm":    marker_size_mm,
    }
