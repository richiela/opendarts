"""3D dart-axis geometry: a 2D image line -> a camera-center-containing
world plane -> two-or-three such planes intersected into a 3D line -> ∩
the board's Z=0 plane. This module is pure math on already-detected 2D
lines/pixels (from `opendarts.engines.talos.shaft_line`); it never touches
pixels/images itself.

Reuses `opendarts.triangulation.rays.back_project_ray()` -- the one
genuinely shared, universal piece of camera geometry every engine calls
into unchanged -- for both the two-ray-per-line plane construction below
and the single-pixel ∩ Z=0 helper (`_pixel_board_xy`, also used by
`opendarts.engines.talos.consensus`).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from opendarts.pipeline import CameraCalibration

# Steel-tip entry is in the sisal, ~1.5mm behind the wire-face plane
# (Z=0). Talos's dart axis is the 3-plane line; intersecting it at Z=0
# is the wire-face crossing, not the scoring surface. Cap-ray
# triangulation returns a 3D point in front of the board; sliding that
# point along the dart axis to BOARD_HIT_Z_MM is Talos's mapping of
# "where the needle stuck."
# Measured 2026-08-13:
#   z=0:     new +5/-0, old +0/-1
#   z=-0.75: new +5/-0, old +1/-1
#   z=-1.0:  new +5/-0, old +1/-1
#   z=-1.5:  new +6/-0, old +1/-1  (net 0 on 169; 172/180 on tonight)
#   z=-2.0:  new +6/-1, old +1/-1
# Independently, Athena measured the same sisal plane on its own
# ray∩Z path (BOARD_PLANE_Z_MM = -1.5). This is not a copy of that
# engine: Talos slides along its own reconstructed dart axis, not a
# per-camera ray. Plane-as-primary (replacing cap-rays) was +6/-16
# and is rejected.
BOARD_HIT_Z_MM = -1.5

# Below this |d_z|, treat the reconstructed dart as parallel to the board
# (no unique entry). Geometric degeneracy guard, not a scoring threshold.
_MIN_LINE_DZ = 1e-6
# Below this ||n1 × n2||, two planes are too parallel to define a line.
_MIN_PLANE_CROSS = 1e-8

# 3-plane residual (max |n_i · hit - c_i| in mm at the Z=0 point) above
# which the three planes are not concurrent and one plane is dropped.
# Measured on data/archive/clean, 169 throws, original 3-plane SVD:
#   concurrent cluster: p50=2.00 p90=5.81 p95=8.31, 2 throws in [10, 15)
#   empty bin:          [15, 20)
#   explosion tail:     min 29.2, then 56 / 113 / 228 / 250 / 832
# 17.5 is the midpoint of the empty bin. Prefer 3 planes when the
# residual sits in the concurrent cluster. Not swept against AD%.
CONCURRENCE_GATE_MM = 17.5

# Reconstructed dart tilt (angle from the board normal / +Z) above which
# the 3D axis is not a stuck dart. Measured on data/archive/clean, 169
# current-engine reconstructions:
#   p50=11.0  p90=17.7  p99=23.4  then empty [24, 80)  then one at 81°
# Steel-tip darts in sisal do not lie in the board plane; 40° sits in
# that empty bin. When the all-plane fit exceeds this, drop to a 2-plane
# pair with plausible tilt. Not swept against AD%.
MAX_PLAUSIBLE_TILT_DEG = 40.0


def _pixel_board_xy(pixel, calib: CameraCalibration):
    from opendarts.triangulation.rays import back_project_ray

    ray = back_project_ray(
        (float(pixel[0]), float(pixel[1])),
        calib.camera_matrix, calib.dist_coeffs, calib.rvec, calib.tvec,
    )
    dz = float(ray.direction[2])
    if abs(dz) < 1e-9:
        return None
    t = -float(ray.origin[2]) / dz
    if t <= 0:
        return None
    hit = ray.origin + t * ray.direction
    return (float(hit[0]), float(hit[1]))


def plane_from_image_line(
    p1_px: tuple[float, float],
    p2_px: tuple[float, float],
    calib: CameraCalibration,
) -> tuple[np.ndarray, float] | None:
    """World plane: unit normal n, offset c, with n · X = c.

    Built from two back-projected rays sharing the camera center.
    """
    from opendarts.triangulation.rays import back_project_ray

    ray1 = back_project_ray(
        p1_px, calib.camera_matrix, calib.dist_coeffs, calib.rvec, calib.tvec
    )
    ray2 = back_project_ray(
        p2_px, calib.camera_matrix, calib.dist_coeffs, calib.rvec, calib.tvec
    )
    n = np.cross(ray1.direction, ray2.direction)
    n_norm = float(np.linalg.norm(n))
    if n_norm < _MIN_PLANE_CROSS:
        return None
    n = n / n_norm
    origin = 0.5 * (ray1.origin + ray2.origin)
    c = float(np.dot(n, origin))
    return n, c


def _line_from_planes(
    planes: list[tuple[np.ndarray, float]],
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Least-squares 3D line in the given planes: unit direction + a point.

    Direction D is the vector most orthogonal to every plane normal
    (smallest right-singular vector of the stacked normals). A point P
    on the line is the least-squares solution of n_i · P = c_i.
    For exactly two planes this is equivalent to d = n1 × n2 plus a
    gauge (the SVD nullspace of two normals).
    """
    if len(planes) < 2:
        return None
    normals = np.stack([n for n, _ in planes], axis=0)
    offsets = np.array([c for _, c in planes], dtype=np.float64)

    if len(planes) == 2:
        n1, n2 = normals[0], normals[1]
        direction = np.cross(n1, n2)
        cross_norm = float(np.linalg.norm(direction))
        if cross_norm < _MIN_PLANE_CROSS:
            return None
        direction = direction / cross_norm
        # Point on the line: two plane equations + gauge d · X = 0
        # (closest point to the origin). Equivalent to the min-norm
        # lstsq of the underdetermined 2x3 system.
        a = np.stack([n1, n2, direction], axis=0)
        b = np.array([offsets[0], offsets[1], 0.0], dtype=np.float64)
        try:
            p_ls = np.linalg.solve(a, b)
        except np.linalg.LinAlgError:
            p_ls, *_ = np.linalg.lstsq(normals, offsets, rcond=None)
        svals = np.array([cross_norm], dtype=np.float64)
    else:
        _, svals, vh = np.linalg.svd(normals, full_matrices=True)
        direction = vh[-1]
        d_norm = float(np.linalg.norm(direction))
        if d_norm < 1e-12:
            return None
        direction = direction / d_norm
        p_ls, *_ = np.linalg.lstsq(normals, offsets, rcond=None)

    if abs(float(direction[2])) < _MIN_LINE_DZ:
        return None
    if direction[2] < 0:
        direction = -direction
    return direction, p_ls, {"svd_smallest": float(svals[-1]) if len(svals) else None}


def _hit_z0(direction: np.ndarray, point: np.ndarray) -> np.ndarray | None:
    return _hit_z(direction, point, 0.0)


def _hit_z(direction: np.ndarray, point: np.ndarray, z_mm: float) -> np.ndarray | None:
    dz = float(direction[2])
    if abs(dz) < _MIN_LINE_DZ:
        return None
    t_hit = (float(z_mm) - float(point[2])) / dz
    return point + t_hit * direction


def _residuals_at(hit: np.ndarray, planes: list[tuple[np.ndarray, float]]) -> list[float]:
    return [float(abs(np.dot(n, hit) - c)) for n, c in planes]


def _geom_for(
    hit: np.ndarray,
    direction: np.ndarray,
    planes: list[tuple[np.ndarray, float]],
    extra: dict,
) -> dict:
    plane_residuals = _residuals_at(hit, planes)
    dir_residuals = [float(abs(np.dot(n, direction))) for n, _ in planes]
    tilt_deg = float(math.degrees(math.acos(min(1.0, abs(float(direction[2]))))))
    out = {
        "hit_xyz_mm": [float(hit[0]), float(hit[1]), float(hit[2])],
        "direction": direction.tolist(),
        "plane_residual_mm": plane_residuals,
        "max_plane_residual_mm": max(plane_residuals) if plane_residuals else None,
        "dir_dot_normal": dir_residuals,
        "tilt_deg": tilt_deg,
    }
    out.update(extra)
    return out


def intersect_planes_with_board(
    planes: list[tuple[np.ndarray, float]],
) -> tuple[tuple[float, float], dict] | None:
    """3D dart axis from planes, then ∩ Z=0.

    All planes are fitted first. If the N-plane residual sits in the
    concurrent cluster AND the reconstructed tilt is physically a stuck
    dart (see CONCURRENCE_GATE_MM / MAX_PLAUSIBLE_TILT_DEG), that hit is
    used. Otherwise each 2-plane pair is fitted and the pair whose
    board-hit is least incompatible with the unused plane is kept,
    preferring a plausible tilt over an 80°-in-the-board axis.

    Pair ranking cannot use participating-plane residual: measured on
    clean/ (507 pairs) that quantity is ~1e-14 mm for every pair, as
    two planes always contain their intersection. Ranking uses the
    unused plane's residual instead -- still a geometric consistency
    check, not an AD match.
    """
    if len(planes) < 2:
        return None

    all_fit = _line_from_planes(planes)
    all_hit = None
    all_res = None
    all_geom = None
    if all_fit is not None:
        direction, point, extra = all_fit
        all_hit = _hit_z0(direction, point)
        if all_hit is not None:
            all_res = _residuals_at(all_hit, planes)
            all_geom = _geom_for(
                all_hit, direction, planes,
                {
                    **extra,
                    "hypothesis": "all",
                    "planes_used_indices": list(range(len(planes))),
                    "all_max_plane_residual_mm": max(all_res),
                    "concurrence_gate_mm": CONCURRENCE_GATE_MM,
                    "max_plausible_tilt_deg": MAX_PLAUSIBLE_TILT_DEG,
                },
            )

    all_tilt = None if all_geom is None else all_geom.get("tilt_deg")
    use_all = (
        all_geom is not None
        and all_res is not None
        and max(all_res) <= CONCURRENCE_GATE_MM
        and (all_tilt is None or float(all_tilt) <= MAX_PLAUSIBLE_TILT_DEG)
    )
    # Two planes are always concurrent (participating residual ~0).
    # Prefer them as "all" rather than searching pairs of a pair.
    if len(planes) == 2 and all_geom is not None and all_hit is not None:
        return (float(all_hit[0]), float(all_hit[1])), all_geom
    if use_all and all_hit is not None and all_geom is not None:
        return (float(all_hit[0]), float(all_hit[1])), all_geom

    if len(planes) < 3:
        if all_hit is not None and all_geom is not None:
            return (float(all_hit[0]), float(all_hit[1])), all_geom
        return None

    best: tuple[tuple, np.ndarray, dict] | None = None
    pair_diags = []
    for pair in itertools.combinations(range(len(planes)), 2):
        sub = [planes[i] for i in pair]
        fitted = _line_from_planes(sub)
        if fitted is None:
            continue
        direction, point, extra = fitted
        hit = _hit_z0(direction, point)
        if hit is None:
            continue
        part_res = _residuals_at(hit, sub)
        excluded = [planes[i] for i in range(len(planes)) if i not in pair]
        excl_res = _residuals_at(hit, excluded)
        n1, n2 = sub[0][0], sub[1][0]
        cross = float(np.linalg.norm(np.cross(n1, n2)))
        # Rank by unused-plane residual (participating residual is ~0).
        # Implausible tilt (dart lying in the board) loses to any
        # physically possible pair, even if that pair's unused residual
        # is larger -- the 81° reconstruction on clean/ had 8mm residual
        # (inside CONCURRENCE_GATE_MM) and was still garbage.
        rank = max(excl_res) if excl_res else max(part_res)
        tilt = float(math.degrees(math.acos(min(1.0, abs(float(direction[2]))))))
        implausible = tilt > MAX_PLAUSIBLE_TILT_DEG
        geom = _geom_for(
            hit, direction, sub,
            {
                **extra,
                "hypothesis": "pair",
                "planes_used_indices": list(pair),
                "all_max_plane_residual_mm": max(all_res) if all_res else None,
                "excluded_residual_mm": excl_res,
                "max_excluded_residual_mm": max(excl_res) if excl_res else None,
                "plane_cross_norm": cross,
                "concurrence_gate_mm": CONCURRENCE_GATE_MM,
            },
        )
        pair_diags.append({
            "indices": list(pair),
            "max_excluded_residual_mm": max(excl_res) if excl_res else None,
            "max_participating_residual_mm": max(part_res) if part_res else None,
            "tilt_deg": tilt,
            "implausible_tilt": implausible,
        })
        key = (implausible, rank, tilt)
        if best is None or key < best[0]:
            best = (key, hit, geom)

    if best is None:
        if all_hit is not None and all_geom is not None:
            return (float(all_hit[0]), float(all_hit[1])), all_geom
        return None
    _, hit, geom = best
    geom["pair_hypotheses"] = pair_diags
    return (float(hit[0]), float(hit[1])), geom


def slide_point_along_dart_axis(
    point_xyz,
    planes: list[tuple[np.ndarray, float]],
    z_mm: float = BOARD_HIT_Z_MM,
) -> tuple[float, float] | None:
    """Slide a triangulated 3D point along Talos's dart axis to z=z_mm.

    `point_xyz` is score_dart's closest-point of cap rays (usually in
    front of the wire face). The dart axis is intersect_planes_with_board
    of the shaft planes. Returns (x, y) at the sisal plane, or None if
    the axis cannot be reconstructed.
    """
    if point_xyz is None or len(planes) < 2:
        return None
    hit = intersect_planes_with_board(planes)
    if hit is None:
        return None
    _xy, geom = hit
    direction = np.asarray(geom["direction"], dtype=np.float64)
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    new_hit = _hit_z(direction, point, z_mm)
    if new_hit is None:
        return None
    return (float(new_hit[0]), float(new_hit[1]))


def dart_axis_xy_at_z(
    planes: list[tuple[np.ndarray, float]],
    z_mm: float = BOARD_HIT_Z_MM,
) -> tuple[float, float] | None:
    """Shaft-plane dart axis intersected with the plane z=z_mm.

    Unlike `slide_point_along_dart_axis`, this ignores the cap-ray 3D
    point: it is the axis itself, used as a *gate* (does the shaft
    geometry agree with the centerlines?) not as a primary score.
    """
    if len(planes) < 2:
        return None
    hit = intersect_planes_with_board(planes)
    if hit is None:
        return None
    xy0, geom = hit
    direction = np.asarray(geom["direction"], dtype=np.float64)
    point = np.array([xy0[0], xy0[1], 0.0], dtype=np.float64)
    new_hit = _hit_z(direction, point, z_mm)
    if new_hit is None:
        return None
    return (float(new_hit[0]), float(new_hit[1]))


def snap_xy_to_plane_line(
    xy: tuple[float, float],
    plane: tuple[np.ndarray, float],
    z_mm: float = 0.0,
) -> tuple[float, float] | None:
    """Closest point on (plane ∩ z=z_mm) to xy, in the XY plane.

    The dart lies in the shaft plane, so its board hit must lie on that
    plane's board trace. The cap ray∩z only chooses position along it.
    """
    n, c = plane
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    rhs = float(c) - nz * float(z_mm)
    denom = nx * nx + ny * ny
    if denom < 1e-12:
        return None
    x, y = float(xy[0]), float(xy[1])
    t = (nx * x + ny * y - rhs) / denom
    return (x - t * nx, y - t * ny)
