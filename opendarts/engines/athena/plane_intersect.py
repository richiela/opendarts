"""Ray/board-plane intersection -- the small, genuinely universal helper
docs/DESIGN.md's Athena task description asked for: "intersect that ray
with the board's Z=0 plane directly (simple ray-plane intersection --
you'll likely need to write this small helper yourself; a plain
function, not X-camera triangulation)". `opendarts.triangulation.rays`
deliberately has no such function -- it only does multi-ray least-squares
triangulation (`triangulate()`), which needs >=2 rays. Athena's whole
design point is getting a genuinely independent single-camera 2D read,
so it needs single-ray/plane intersection instead, which is a much
simpler, different piece of math (solve one linear equation for the
ray's plane parameter, not a least-squares system).
"""
from __future__ import annotations

import numpy as np

from opendarts.triangulation.rays import Ray

# Same convention as opendarts.triangulation.rays.TriangulationResult:
# "t <= 0" means the solution sits behind the camera -- not a physically
# valid detection, even if the pure linear algebra "succeeds". Mirrors
# the per_ray_depth_mm / all_positive_depth check that was added
# to rays.triangulate() (see that module's docstring) -- the same class
# of bug (a ray and its backward mirror image solve the same equation)
# applies here just as much to a single ray.
_MIN_FORWARD_DEPTH_MM = 1e-6


def ray_plane_intersect(ray: Ray, z_mm: float = 0.0) -> tuple[float, float] | None:
    """Where a single 3D ray crosses the board plane Z = z_mm (board-
    centered world frame, per opendarts.geometry.board's own module
    docstring: origin at the bullseye, board face is Z=0).

    Returns (x_mm, y_mm) on that plane, or None if the ray cannot
    validly reach it: either it runs parallel to the plane (direction_z
    ~ 0, no intersection at all) or the intersection point sits BEHIND
    the camera (t <= 0) -- geometrically "solvable" but not a real
    detection, exactly the blind spot found and fixed in
    opendarts.triangulation.rays.triangulate() for the multi-ray case (see
    that module's TriangulationResult.per_ray_depth_mm docstring); this
    single-ray helper needs the identical guard for the identical reason.
    """
    origin = np.asarray(ray.origin, dtype=np.float64)
    direction = np.asarray(ray.direction, dtype=np.float64)
    dz = direction[2]
    if abs(dz) < 1e-9:
        return None
    t = (z_mm - origin[2]) / dz
    if t <= _MIN_FORWARD_DEPTH_MM:
        return None
    point = origin + t * direction
    return (float(point[0]), float(point[1]))
