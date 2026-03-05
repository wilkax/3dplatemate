"""
STL generation.

Takes a list of 2-D polygons (in mm) and extrudes each one to `height_mm` tall,
then combines them all into a single binary STL file ready for slicing.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

CLEANER_HEIGHT_MM = 1.0
_MIN_AREA_MM2 = 0.5   # polygons smaller than this (mm²) cannot be extruded meaningfully


def _polygon_to_mesh(polygon_mm: list[list[float]], height_mm: float) -> trimesh.Trimesh:
    """Extrude a single 2-D polygon into a closed 3-D mesh."""
    shape = Polygon(polygon_mm)

    # Fix any self-touches / ring-order issues that survive the coordinate round-trip
    if not shape.is_valid:
        shape = make_valid(shape)

    # make_valid may promote the result to MultiPolygon / GeometryCollection
    if shape.geom_type != "Polygon":
        if hasattr(shape, "geoms"):
            polys = [g for g in shape.geoms if g.geom_type == "Polygon" and not g.is_empty]
            if not polys:
                raise ValueError(f"make_valid produced no Polygon ({shape.geom_type}).")
            shape = max(polys, key=lambda g: g.area)
        else:
            raise ValueError(f"Unexpected geometry type after make_valid: {shape.geom_type}.")

    if shape.is_empty or shape.area < _MIN_AREA_MM2:
        raise ValueError(f"Polygon area {shape.area:.4f} mm² is too small to extrude.")

    mesh = trimesh.creation.extrude_polygon(shape, height=height_mm)

    if mesh is None or (hasattr(mesh, "is_empty") and mesh.is_empty):
        raise ValueError("trimesh.extrude_polygon returned an empty mesh.")

    return mesh


_FRAME_WIDTH_MM  = 1.0   # width of the plate-boundary reference frame
_FRAME_HEIGHT_MM = 0.2   # height of the frame (thin single layer, ~0.2 mm)


def _make_plate_frame(plate_width_mm: float, plate_height_mm: float) -> trimesh.Trimesh:
    """
    Build a thin rectangular frame the size of the build plate.

    The frame is _FRAME_WIDTH_MM wide and _FRAME_HEIGHT_MM tall.  Its only
    purpose is to tell the slicer the true extent of the plate so it does not
    auto-centre the scraper bumps incorrectly.
    """
    outer = box(0, 0, plate_width_mm, plate_height_mm)
    inner = box(
        _FRAME_WIDTH_MM,
        _FRAME_WIDTH_MM,
        plate_width_mm  - _FRAME_WIDTH_MM,
        plate_height_mm - _FRAME_WIDTH_MM,
    )
    frame_shape = outer.difference(inner)
    return trimesh.creation.extrude_polygon(frame_shape, height=_FRAME_HEIGHT_MM)


def generate_stl(
    polygons_mm: list[list[list[float]]],
    height_mm: float = CLEANER_HEIGHT_MM,
    plate_width_mm: float | None = None,
    plate_height_mm: float | None = None,
) -> bytes:
    """
    Generate a single binary STL containing one extruded object per dirty spot,
    plus an optional thin reference frame around the plate boundary.

    Parameters
    ----------
    polygons_mm     : list of polygons; each polygon is a list of [x, y] mm coords.
    height_mm       : extrusion height for the scraper bumps (default: 1.0 mm).
    plate_width_mm  : if provided (together with plate_height_mm), a thin reference
                      frame the size of the build plate is added so slicers place the
                      scraper at the correct position instead of auto-centring it.
    plate_height_mm : see plate_width_mm.

    Returns
    -------
    Binary STL file contents as bytes.

    Raises
    ------
    ValueError : if polygons_mm is empty or all polygons fail to extrude.
    """
    if not polygons_mm:
        raise ValueError("No polygons provided — nothing to generate.")

    meshes: list[trimesh.Trimesh] = []
    for i, polygon in enumerate(polygons_mm):
        try:
            mesh = _polygon_to_mesh(polygon, height_mm)
            meshes.append(mesh)
        except Exception as exc:
            # Skip degenerate polygons but don't abort the whole request
            logger.warning("Skipping polygon %d due to error: %s", i, exc)

    if not meshes:
        raise ValueError("All polygons were degenerate — no STL generated.")

    # Add a thin plate-boundary frame so the slicer preserves the correct position
    if plate_width_mm and plate_height_mm:
        try:
            frame = _make_plate_frame(plate_width_mm, plate_height_mm)
            meshes.append(frame)
        except Exception as exc:
            logger.warning("Could not add plate reference frame: %s", exc)

    combined = trimesh.util.concatenate(meshes)

    # Export to binary STL in memory
    buffer = io.BytesIO()
    combined.export(buffer, file_type="stl")
    return buffer.getvalue()

