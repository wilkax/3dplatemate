"""
STL generation.

Takes a list of 2-D polygons (in mm) and extrudes each one to `height_mm` tall,
then combines them all into a single binary STL file ready for slicing.
"""

from __future__ import annotations

import io

import numpy as np
import trimesh
from shapely.geometry import Polygon


CLEANER_HEIGHT_MM = 1.0


def _polygon_to_mesh(polygon_mm: list[list[float]], height_mm: float) -> trimesh.Trimesh:
    """Extrude a single 2-D polygon into a closed 3-D mesh."""
    shape = Polygon(polygon_mm)

    if not shape.is_valid or shape.is_empty:
        raise ValueError("Cannot extrude invalid or empty polygon.")

    mesh = trimesh.creation.extrude_polygon(shape, height=height_mm)
    return mesh


def generate_stl(
    polygons_mm: list[list[list[float]]],
    height_mm: float = CLEANER_HEIGHT_MM,
) -> bytes:
    """
    Generate a single binary STL containing one extruded object per dirty spot.

    Parameters
    ----------
    polygons_mm : list of polygons; each polygon is a list of [x, y] mm coordinates.
    height_mm   : extrusion height (default: 1.0 mm).

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
            import logging
            logging.getLogger(__name__).warning(
                "Skipping polygon %d due to error: %s", i, exc
            )

    if not meshes:
        raise ValueError("All polygons were degenerate — no STL generated.")

    combined = trimesh.util.concatenate(meshes)

    # Export to binary STL in memory
    buffer = io.BytesIO()
    combined.export(buffer, file_type="stl")
    return buffer.getvalue()

