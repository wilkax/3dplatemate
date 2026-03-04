"""
Geometry utilities.

- compute_homography : maps pixel coordinates → mm coordinates using the 4 detected
                       plate corners and the known plate dimensions.
- transform_polygon  : applies the homography to a list of [x, y] pixel points.
- buffer_polygon     : expands a mm polygon outward by `margin_mm` using Shapely,
                       clips to the plate bounds, and returns the resulting vertices.
"""

from __future__ import annotations

import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.validation import make_valid

# Resolution used for the corrected top-down image (pipeline + debug)
PX_PER_MM: float = 5.0


def compute_homography(
    corners_px: dict,
    plate_width_mm: float,
    plate_height_mm: float,
) -> np.ndarray:
    """
    Compute a homography matrix H such that H @ [px, py, 1]^T → [mm_x, mm_y, w].

    Parameters
    ----------
    corners_px:
        Dict with keys top_left, top_right, bottom_right, bottom_left;
        each value is [x, y] in pixel space.
    plate_width_mm, plate_height_mm:
        Physical build plate dimensions.

    Returns
    -------
    H : (3, 3) float64 array
    """
    src = np.float32([
        corners_px["top_left"],
        corners_px["top_right"],
        corners_px["bottom_right"],
        corners_px["bottom_left"],
    ])
    dst = np.float32([
        [0,               0],
        [plate_width_mm,  0],
        [plate_width_mm,  plate_height_mm],
        [0,               plate_height_mm],
    ])
    H, _ = cv2.findHomography(src, dst)
    return H


def transform_polygon(polygon_px: list[list[float]], H: np.ndarray) -> list[list[float]]:
    """
    Apply homography H to a polygon defined in pixel space.

    Parameters
    ----------
    polygon_px : list of [x, y] pixel coordinates
    H          : (3, 3) homography matrix from compute_homography

    Returns
    -------
    list of [x, y] coordinates in mm space
    """
    pts = np.float32(polygon_px).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2).tolist()


def buffer_polygon(
    polygon_mm: list[list[float]],
    margin_mm: float,
    plate_width_mm: float,
    plate_height_mm: float,
) -> list[list[float]] | None:
    """
    Expand a polygon outward by `margin_mm`, then clip to the plate bounds.

    Parameters
    ----------
    polygon_mm       : list of [x, y] in mm
    margin_mm        : outward offset in mm (typically 2.0)
    plate_width_mm   : used to clip the result to the plate boundary
    plate_height_mm  : used to clip the result to the plate boundary

    Returns
    -------
    list of [x, y] vertices of the buffered, clipped polygon,
    or None if the result is degenerate.
    """
    shape = Polygon(polygon_mm)
    if not shape.is_valid:
        shape = make_valid(shape)

    buffered = shape.buffer(margin_mm, join_style="round", cap_style="round")

    plate_bounds = Polygon([
        [0, 0],
        [plate_width_mm, 0],
        [plate_width_mm, plate_height_mm],
        [0, plate_height_mm],
    ])
    clipped = buffered.intersection(plate_bounds)

    if clipped.is_empty or clipped.geom_type not in ("Polygon", "MultiPolygon"):
        return None

    # For MultiPolygon (rare edge case), take the largest part
    if clipped.geom_type == "MultiPolygon":
        clipped = max(clipped.geoms, key=lambda g: g.area)

    coords = list(clipped.exterior.coords)
    # Remove the closing duplicate vertex that Shapely adds
    return [list(pt) for pt in coords[:-1]]


def warp_image(
    image_bytes: bytes,
    H: np.ndarray,
    plate_width_mm: float,
    plate_height_mm: float,
    px_per_mm: float = PX_PER_MM,
) -> np.ndarray:
    """
    Apply the homography to produce a top-down corrected view of the build plate.

    Parameters
    ----------
    image_bytes    : raw image bytes (JPEG / PNG / WEBP)
    H              : homography from compute_homography (pixels → mm)
    plate_width_mm : physical plate width in mm
    plate_height_mm: physical plate height in mm
    px_per_mm      : output resolution (pixels per mm)

    Returns
    -------
    Warped BGR image as a numpy array.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    out_w = int(round(plate_width_mm * px_per_mm))
    out_h = int(round(plate_height_mm * px_per_mm))

    # H maps pixels → mm; scale to pixels in the output image
    scale = np.diag([px_per_mm, px_per_mm, 1.0])
    H_scaled = scale @ H

    warped = cv2.warpPerspective(img, H_scaled, (out_w, out_h))
    return warped


def corrected_px_to_mm(
    polygon_px: list[list[float]],
    px_per_mm: float = PX_PER_MM,
) -> list[list[float]]:
    """
    Convert polygon vertices from corrected-image pixel space to mm.

    The corrected image has a known, uniform scale (px_per_mm), so this is
    a simple division — no homography needed.

    Parameters
    ----------
    polygon_px : list of [x, y] in corrected-image pixel coordinates
    px_per_mm  : scale of the corrected image (pixels per mm)

    Returns
    -------
    list of [x, y] in mm
    """
    return [[x / px_per_mm, y / px_per_mm] for x, y in polygon_px]

