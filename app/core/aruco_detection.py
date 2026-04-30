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

import io

import cv2
import numpy as np
import trimesh

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



def generate_marker_3mf(
    size_mm: float = 50.0,
    base_height_mm: float = 1.2,
    raise_height_mm: float = 0.8,
    quiet_zone_cells: int = 1,
) -> bytes:
    """Generate a 3D-printable ArUco marker as a 3MF file.

    Geometry
    --------
    The ArUco marker (DICT_4X4_50, id 0) is a 6×6 cell grid
    (4×4 data + 1-cell black border on each side).  An additional white
    quiet zone of ``quiet_zone_cells`` cells is added around the outside
    so the detector always has enough margin.

    Two mesh objects are exported to the scene:

    * **marker_base** — the full white tile (base height).  Print in white
      or light-coloured filament.
    * **marker_pattern** — the raised black cells on top (raise height above
      the base).  Print in black filament after a colour-change at the
      appropriate layer, or in a dual-extrusion setup.

    For single-colour printing the height difference alone creates enough
    shadow contrast for reliable detection at typical photo distances.

    Parameters
    ----------
    size_mm         : edge length of the active marker area (the 6×6 grid), mm
    base_height_mm  : thickness of the base tile, mm
    raise_height_mm : extra height of the black cells above the base, mm
    quiet_zone_cells: number of extra white border cells outside the 6×6 grid
    """
    if size_mm <= 0:
        raise ValueError("size_mm must be > 0.")
    if base_height_mm <= 0 or raise_height_mm <= 0:
        raise ValueError("height values must be > 0.")

    n_marker_cells  = 6                        # 4×4 data + 1-cell black border
    cell_size       = size_mm / n_marker_cells
    qz              = quiet_zone_cells * cell_size
    tile_size       = size_mm + 2.0 * qz      # total tile footprint

    # ── Read the black/white pattern from OpenCV ───────────────────────────────
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    px_per_cell = 16
    img_size    = n_marker_cells * px_per_cell
    marker_img  = cv2.aruco.generateImageMarker(
        aruco_dict, ARUCO_MARKER_ID, img_size,
    )                                          # grayscale, 0 = black, 255 = white

    # ── Base plate (white background + quiet zone) ─────────────────────────────
    base = trimesh.creation.box(extents=[tile_size, tile_size, base_height_mm])
    base.apply_translation([tile_size / 2.0, tile_size / 2.0, base_height_mm / 2.0])

    # ── Raised cells for every dark pixel-block ────────────────────────────────
    raise_meshes: list[trimesh.Trimesh] = []
    z_centre = base_height_mm + raise_height_mm / 2.0

    for row in range(n_marker_cells):
        for col in range(n_marker_cells):
            sample_px = int((col + 0.5) * px_per_cell)
            sample_py = int((row + 0.5) * px_per_cell)
            if marker_img[sample_py, sample_px] < 128:   # black cell
                # col  → x, with quiet-zone offset
                x_centre = qz + col * cell_size + cell_size / 2.0
                # row 0 is image-top; in 3-D we flip Y so row 0 → highest Y
                y_centre = qz + (n_marker_cells - 1 - row) * cell_size + cell_size / 2.0
                cell_box = trimesh.creation.box(
                    extents=[cell_size, cell_size, raise_height_mm],
                )
                cell_box.apply_translation([x_centre, y_centre, z_centre])
                raise_meshes.append(cell_box)

    # ── Assemble scene ─────────────────────────────────────────────────────────
    scene = trimesh.Scene()
    scene.add_geometry(base, geom_name="marker_base")
    if raise_meshes:
        pattern = trimesh.util.concatenate(raise_meshes)
        scene.add_geometry(pattern, geom_name="marker_pattern")

    buf = io.BytesIO()
    scene.export(buf, file_type="3mf")
    return buf.getvalue()
