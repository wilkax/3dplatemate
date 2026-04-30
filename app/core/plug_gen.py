"""
Generate a ribbed "fill the hole" plug as a 3MF.

Geometry (printed orientation, flat cap on the build plate):

      z = depth  ┌───────────────────┐  ← open top
                 │   walls + ribs    │
      z = cap   ─┼───────────────────┼─
                 │   solid cap       │
      z = 0      └───────────────────┘  ← build-plate side / visible top
                                          after insertion

The user-supplied polygon (mm) is shrunk inward by `tolerance_mm` to give the
actual plug perimeter, then by `WALL_THICKNESS_MM` again to define the cavity.
Two perpendicular ribs through the cavity centroid stiffen the plug while
saving material.  If the polygon is too small to admit walls + ribs the plug
is exported as a simple solid extrusion.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import trimesh
from shapely.geometry import Polygon, box
from shapely.validation import make_valid

logger = logging.getLogger(__name__)

WALL_THICKNESS_MM      = 1.5
CAP_THICKNESS_MM       = 1.0
RIB_THICKNESS_MM       = 1.5
_MIN_PLUG_AREA_MM2     = 4.0    # polygons smaller than this cannot be plugged
_MIN_CAVITY_AREA_MM2   = 4.0    # cavity smaller than this → solid plug
_MIN_CAVITY_HEIGHT_MM  = 1.5    # cavity shorter than this → solid plug


def _largest_polygon(geom):
    """Return the largest Polygon component of a Shapely geometry, or None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if g.geom_type == "Polygon" and not g.is_empty]
        if polys:
            return max(polys, key=lambda g: g.area)
    return None


def _build_plug_mesh(
    polygon_mm: list[list[float]],
    depth_mm:   float,
    tolerance_mm: float,
) -> trimesh.Trimesh:
    if depth_mm <= 0:
        raise ValueError("depth_mm must be > 0.")
    if tolerance_mm < 0:
        raise ValueError("tolerance_mm must be >= 0.")

    raw = Polygon(polygon_mm)
    if not raw.is_valid:
        raw = make_valid(raw)
    raw = _largest_polygon(raw)
    if raw is None or raw.area < _MIN_PLUG_AREA_MM2:
        raise ValueError("Polygon is too small to extrude.")

    plug_outer = _largest_polygon(raw.buffer(-tolerance_mm))
    if plug_outer is None or plug_outer.area < _MIN_PLUG_AREA_MM2:
        raise ValueError(
            f"Tolerance {tolerance_mm} mm leaves no usable plug area "
            f"(input area {raw.area:.1f} mm\u00b2).",
        )

    outer_mesh = trimesh.creation.extrude_polygon(plug_outer, height=depth_mm)

    cavity_shape  = _largest_polygon(plug_outer.buffer(-WALL_THICKNESS_MM))
    cavity_height = depth_mm - CAP_THICKNESS_MM

    if (cavity_shape is None
            or cavity_shape.area < _MIN_CAVITY_AREA_MM2
            or cavity_height < _MIN_CAVITY_HEIGHT_MM):
        logger.info("Plug exported as solid (cavity too small / depth too shallow).")
        return outer_mesh

    cavity_mesh = trimesh.creation.extrude_polygon(cavity_shape, height=cavity_height)
    cavity_mesh.apply_translation([0, 0, CAP_THICKNESS_MM])

    hollow = trimesh.boolean.difference(
        [outer_mesh, cavity_mesh], engine="manifold",
    )

    rib_shapes = _rib_shapes(cavity_shape)
    rib_meshes: list[trimesh.Trimesh] = []
    for shape in rib_shapes:
        m = trimesh.creation.extrude_polygon(shape, height=cavity_height)
        m.apply_translation([0, 0, CAP_THICKNESS_MM])
        rib_meshes.append(m)

    if not rib_meshes:
        return hollow

    return trimesh.boolean.union([hollow, *rib_meshes], engine="manifold")


def _rib_shapes(cavity: Polygon) -> list[Polygon]:
    """Two perpendicular rib strips through the cavity centroid, clipped to it."""
    cx, cy = cavity.centroid.x, cavity.centroid.y
    minx, miny, maxx, maxy = cavity.bounds
    span = max(maxx - minx, maxy - miny) + 2.0
    half = RIB_THICKNESS_MM / 2.0

    raw_strips = [
        box(cx - span, cy - half, cx + span, cy + half),   # horizontal rib
        box(cx - half, cy - span, cx + half, cy + span),   # vertical rib
    ]
    out: list[Polygon] = []
    for s in raw_strips:
        clipped = s.intersection(cavity)
        if clipped.is_empty:
            continue
        if clipped.geom_type == "Polygon":
            if clipped.area >= 0.1:
                out.append(clipped)
        elif hasattr(clipped, "geoms"):
            for g in clipped.geoms:
                if g.geom_type == "Polygon" and g.area >= 0.1:
                    out.append(g)
    return out


def generate_plug_3mf(
    polygon_mm: list[list[float]],
    depth_mm:   float,
    tolerance_mm: float = 0.2,
) -> bytes:
    """Build the plug mesh and export it as a single-object 3MF.

    Raises ValueError if the geometry cannot be produced.
    """
    mesh = _build_plug_mesh(polygon_mm, depth_mm, tolerance_mm)
    if mesh is None or (hasattr(mesh, "is_empty") and mesh.is_empty):
        raise ValueError("Plug mesh generation produced an empty result.")

    scene = trimesh.Scene()
    scene.add_geometry(mesh, geom_name="plug")
    buf = io.BytesIO()
    scene.export(buf, file_type="3mf")
    return buf.getvalue()
