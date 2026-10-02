"""Cross-view blob selection for Apollo (2026-09-28).

`tip_detection.detect_tip()` picks each camera's dart blob on that one
image alone: the largest diff component (after a 31px dilation) whose
shape is elongated enough. Nothing checks the pick against the other two
views, so a flight, a neighbouring dart, a number-ring or wire-line diff,
or the dart's own upper shaft can win while the real tip sits in a small
detached blob next to it. Measured on the 2026-09 corpus (1,618 darts
the oracle also scored): 347 of 4,854 camera views had their tip more
than 20px from the oracle's, cam2 worst. The 2-of-3 outlier drop rescued
most of those, but by luck rather than on principle -- 047-S5's wire-line
pick was only dropped because it happened to be the outlier, and a 4px
shift of that same wrong tip was enough to bring it back and flip the
call.

Every camera sees the same physical dart, so a blob is only right if it
fits the other two views in 3D. For each camera this module offers
Apollo's top-K blobs, each in both end orientations (a short, end-on
blob as a bare point, since its direction is noise), and scores every
combination of one option per camera:

  plane -- the shaft lines back-project to planes through each camera
           centre; the dart axis lies in all of them, so they must meet
           in one 3D line (MAX_PLANE_RESIDUAL_MM).
  lean  -- that axis must rise out of the board (MAX_AXIS_TILT_DEG).
  dir   -- in every view the projected axis must point the same way as
           the blob, tip to far end: a blob leaning the wrong way against
           the other two views is not the dart.
  tip   -- each tip ray must pass close to the axis (MAX_TIP_AXIS_MM),
           between just below the board and 40mm above it.

The cheapest consistent 3-camera combination wins; taking something
other than Apollo's own pick costs a small prior, so the pick only
changes when the geometry says so. On the corpus the pick changed on
244 darts, and of the 224 changed camera views the oracle could check,
223 moved closer to its tip (191 to within 15px, against 1 before).

Two refinements, each from a failure seen on real throws:

* WIRE TAILS. A diff running along a board wire can merge into the
  dart's component (047-S5 cam0), putting the tip at the end of the wire
  line. When the chosen tip sits on a projected wire and the diff right
  at the tip runs along it (`wire_tail()`), the tip is accepted only if
  it lands on the board entry of the axis the OTHER two views imply
  (WIRE_OK_PX): a needle that really lies along a wire passes, a wire
  line does not. All 44 firings in the corpus were checked by eye.
* TWO CAMERAS. Where no 3-camera combination is consistent (a flight of
  an earlier dart hiding the tip, the neighbouring dart picked), any two
  planes meet in some line, so a pair on its own proves nothing. A pair
  is accepted only if the camera left out actually SEES that axis as
  changed pixels (MIN_AXIS_SUPPORT), both tips sit near the board
  (PAIR_MAX_TIP_HEIGHT_MM -- two short tips pin the point in the air)
  and no different pair is nearly as good. Otherwise nothing is picked
  and Apollo runs exactly as before.

Replayed on the full corpus together with the half-max tip (see
`tip_detection.refine_tip_half_max()`), Apollo went from 1602 to 1613
right calls (12 better, 1 worse: a dart 2.8mm from the 20/5 wire that
the per-camera vote override flips) and Zeus from 1614 to 1616, with no
Zeus call made worse.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from opendarts.engines.apollo import tip_detection as td
from opendarts.triangulation.rays import back_project_ray

# Blobs considered per camera, by dilated area. Apollo's own pick is
# always among the first TOP_K_AREA_CANDIDATES (3); the extra ones are
# where the detached needle blobs and the other ends of merged
# components turned up.
TOP_K_BLOBS = 6
# A blob shorter than this (px) is offered as a bare point: seen end-on
# its direction is noise and would poison the plane test.
SHORT_BLOB_PX = 30.0
# Cost terms, in mm-equivalents added to the geometric residuals.
NON_PRIMARY_COST = 0.75
POINT_ONLY_COST = 0.5
FLIP_COST = 0.5
RANK_COST = 0.1
MISSING_CAMERA_COST = 3.0
# Consistency gates.
MAX_AXIS_TILT_DEG = 55.0
MAX_PLANE_RESIDUAL_MM = 4.0
MAX_TIP_AXIS_MM = 4.0
MIN_TIP_HEIGHT_MM = -6.0
MAX_TIP_HEIGHT_MM = 40.0
MAX_HIT_RADIUS_MM = 185.0
MAX_TIP_RADIUS_MM = 200.0
MIN_DIRECTION_COS = 0.8
# Wire-tail gate (see module docstring).
WIRE_TOL_PX = 2.5
WIRE_ANGLE_DEG = 12.0
WIRE_PATCH_RADIUS_PX = 12
WIRE_OK_PX = 6.0
# Two-camera fallback (see module docstring).
AXIS_SUPPORT_DIFF = 20.0
MIN_AXIS_SUPPORT = 0.5
AXIS_SUPPORT_WEIGHT = 1.0
PAIR_MARGIN = 0.5
PAIR_MAX_TIP_HEIGHT_MM = 8.0
PAIR_DISTINCT_MM = 3.0
# A pick whose far end is within this of Apollo's own detection IS that
# detection, and the engine keeps it whole (alt candidate, diagnostics).
SAME_DETECTION_PX = 3.0

# Radial wires sit on the 9 + 18k degree lines; the ring wires this gate
# cares about are the treble and double pairs.
_RADIAL_WIRE_RANGE_MM = (16.0, 170.0)
_RING_WIRES_MM = (99.0, 107.0, 162.0, 170.0)


@dataclass
class BlobOption:
    """One way to read one camera: a blob and which end is the tip."""

    tip: tuple[float, float]
    far: tuple[float, float]
    origin: np.ndarray
    direction: np.ndarray
    normal: np.ndarray | None
    offset: float | None
    point: bool
    wire: bool
    prior: float
    rank: int
    flip: bool
    primary: bool


@dataclass
class CrossViewPick:
    """The chosen option per camera (`None` = that camera is dropped)."""

    picks: dict[int, BlobOption | None]
    cost: float
    geometry: dict = field(default_factory=dict)

    def diagnostics(self) -> dict:
        return {
            "picks": {
                str(k): (None if o is None else [o.rank, int(o.flip)])
                for k, o in self.picks.items()
            },
            **{k: v for k, v in self.geometry.items()},
        }


def _ray(px, calib) -> tuple[np.ndarray, np.ndarray]:
    r = back_project_ray(
        (float(px[0]), float(px[1])),
        calib.camera_matrix, calib.dist_coeffs, calib.rvec, calib.tvec,
    )
    d = np.asarray(r.direction, float)
    return np.asarray(r.origin, float), d / np.linalg.norm(d)


def _project(xyz, calib) -> np.ndarray:
    import cv2

    p, _ = cv2.projectPoints(
        np.asarray(xyz, float).reshape(1, 3),
        calib.rvec, calib.tvec, calib.camera_matrix, calib.dist_coeffs,
    )
    return p.reshape(2)


def _calib_key(calib) -> tuple:
    return tuple(
        np.asarray(a, np.float64).tobytes()
        for a in (calib.rvec, calib.tvec, calib.camera_matrix, calib.dist_coeffs)
    )


_WIRE_CACHE: dict[tuple, list[np.ndarray]] = {}


def _wire_polylines(calib) -> list[np.ndarray]:
    """The board's radial and treble/double ring wires projected into this
    camera, cached per calibration."""
    import cv2

    key = _calib_key(calib)
    polys = _WIRE_CACHE.get(key)
    if polys is None:
        polys = []
        for k in range(20):
            th = math.radians(9 + 18 * k)
            pts = np.array([
                [r * math.cos(th), r * math.sin(th), 0.0]
                for r in np.linspace(*_RADIAL_WIRE_RANGE_MM, 60)
            ])
            polys.append(cv2.projectPoints(
                pts, calib.rvec, calib.tvec, calib.camera_matrix, calib.dist_coeffs
            )[0].reshape(-1, 2))
        for r in _RING_WIRES_MM:
            pts = np.array([
                [r * math.cos(t), r * math.sin(t), 0.0]
                for t in np.linspace(0, 2 * math.pi, 360)
            ])
            polys.append(cv2.projectPoints(
                pts, calib.rvec, calib.tvec, calib.camera_matrix, calib.dist_coeffs
            )[0].reshape(-1, 2))
        if len(_WIRE_CACHE) > 64:
            _WIRE_CACHE.clear()
        _WIRE_CACHE[key] = polys
    return polys


def wire_tail(bg_bgr: np.ndarray, frame_bgr: np.ndarray, tip, calib) -> bool:
    """True when `tip` sits on a projected board wire and the diff right
    at the tip runs ALONG that wire -- the signature of a wire-line diff
    merged into the dart's component, and also of a needle that really
    lies along a wire, which is why a flagged tip is judged against the
    other views rather than simply dropped."""
    import cv2

    tip = np.asarray(tip, float)
    rad = WIRE_PATCH_RADIUS_PX
    for poly in _wire_polylines(calib):
        dist = np.linalg.norm(poly - tip, axis=1)
        j = int(dist.argmin())
        if dist[j] > WIRE_TOL_PX:
            continue
        w = poly[min(j + 2, len(poly) - 1)] - poly[max(j - 2, 0)]
        w = w / (np.linalg.norm(w) + 1e-9)
        x0, y0 = int(tip[0]) - rad, int(tip[1]) - rad
        if x0 < 0 or y0 < 0 or x0 + 2 * rad >= frame_bgr.shape[1] or y0 + 2 * rad >= frame_bgr.shape[0]:
            return False
        g0 = cv2.cvtColor(bg_bgr[y0:y0 + 2 * rad + 1, x0:x0 + 2 * rad + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        g1 = cv2.cvtColor(frame_bgr[y0:y0 + 2 * rad + 1, x0:x0 + 2 * rad + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        changed = cv2.GaussianBlur(
            cv2.absdiff(g0, g1), (td.GAUSSIAN_BLUR_KSIZE, td.GAUSSIAN_BLUR_KSIZE), 0
        ) >= td.DIFF_THRESHOLD
        yy, xx = np.nonzero(changed)
        if len(xx) < 8:
            return False
        pts = np.stack([xx, yy], 1).astype(float)
        pts -= pts.mean(0)
        evals, evecs = np.linalg.eigh(pts.T @ pts / len(pts))
        if evals[-1] < 4 * max(evals[0], 1e-6):
            continue  # not line-like right at the tip
        if abs(evecs[:, -1] @ w) >= math.cos(math.radians(WIRE_ANGLE_DEG)):
            return True
    return False


def blob_candidates(
    bg_bgr: np.ndarray, frame_bgr: np.ndarray, k: int = TOP_K_BLOBS, precomputed=None,
) -> list[dict]:
    """Apollo's own blob pipeline -- the very components `detect_tip()`
    ranks (`tip_detection.diff_components()`, shared, not recomputed) and
    the same tip location -- but returning the top-`k` instead of
    stopping at the first elongated one. `primary` marks the component
    `detect_tip()` itself would choose."""
    comps = td.diff_components(bg_bgr, frame_bgr, precomputed)
    img_h, img_w = comps.img_h, comps.img_w
    labels, opened, origin = comps.labels, comps.opened, comps.origin
    ox, oy = origin
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    ranked = []
    for i in range(1, comps.n):
        x, y, w, h, area = (int(v) for v in comps.stats[i])
        bboxes[i] = (x, y, w, h)
        if w > 0.6 * img_w or h > 0.6 * img_h:
            continue  # frame-spanning: lighting drift, as in detect_tip()
        ranked.append((area, i))
    ranked.sort(reverse=True)
    out: list[dict] = []
    primary_rank = None
    for rank, (area, i) in enumerate(ranked[:k]):
        res = td._component_stats_in_bbox(labels, i, bboxes[i], opened, origin)
        salvaged = False
        elongated = res is not None and res[3] >= td.MIN_ELONGATION_RATIO
        if not elongated:
            bx, by, bw, bh = bboxes[i]
            footprint = (labels[by:by + bh, bx:bx + bw] == i) & (
                opened[by - oy:by - oy + bh, bx - ox:bx - ox + bw] != 0
            )
            sub = td._largest_elongated_subcomponent(footprint, ((bx, by), (img_h, img_w)))
            if sub is None:
                if res is None:
                    continue
                pts, centered, principal, _elong = res
            else:
                pts, centered, principal, _sub_area, _elong = sub
                salvaged = True
        else:
            pts, centered, principal, _elong = res
        if primary_rank is None and rank < td.TOP_K_AREA_CANDIDATES and (elongated or salvaged):
            primary_rank = rank
        proj = centered @ principal
        others = [(a, j) for a, j in ranked if j != i]
        companion = td._find_companion_end(
            labels, opened, others, pts.mean(axis=0), principal,
            float(proj.min()), float(proj.max()), bboxes, origin,
        )
        tip, diag = td._locate_tip_in_component(
            pts, centered, principal, forced_non_tip_end=companion,
            bg_gray=comps.bg_gray, diff_blur=comps.diff_blur, diff_origin=origin,
        )
        if not np.all(np.isfinite(tip)):
            continue
        out.append({
            "rank": rank,
            "tip": (float(tip[0]), float(tip[1])),
            "far": diag.get("far_end_px"),
            "length": float(proj.max() - proj.min()),
        })
    if primary_rank is None:
        primary_rank = 0  # detect_tip() falls back to the largest component
    for c in out:
        c["primary"] = c["rank"] == primary_rank
    return out


def _board_hit_ok(origin: np.ndarray, direction: np.ndarray) -> bool:
    if abs(direction[2]) <= 1e-9:
        return False
    t = -origin[2] / direction[2]
    return t > 0 and np.hypot(*(origin + t * direction)[:2]) <= MAX_TIP_RADIUS_MM


def camera_options(bg_bgr: np.ndarray, frame_bgr: np.ndarray, calib, precomputed=None) -> list[BlobOption]:
    """Every reading of this camera worth testing against the others."""
    out: list[BlobOption] = []
    for c in blob_candidates(bg_bgr, frame_bgr, precomputed=precomputed):
        base_prior = 0.0 if c["primary"] else NON_PRIMARY_COST + RANK_COST * c["rank"]
        if c["far"] is not None and c["length"] < SHORT_BLOB_PX:
            origin, direction = _ray(c["tip"], calib)
            if _board_hit_ok(origin, direction):
                out.append(BlobOption(
                    tip=c["tip"], far=tuple(map(float, c["far"])), origin=origin, direction=direction,
                    normal=None, offset=None, point=True, wire=False, prior=base_prior + POINT_ONLY_COST,
                    rank=c["rank"], flip=False, primary=c["primary"],
                ))
            continue
        ends = [(c["tip"], c["far"], False)]
        if c["far"] is not None:
            ends.append((c["far"], c["tip"], True))
        for tip, far, flip in ends:
            if far is None or np.linalg.norm(np.subtract(tip, far)) < 8:
                continue
            o1, d1 = _ray(tip, calib)
            _o2, d2 = _ray(far, calib)
            normal = np.cross(d1, d2)
            nn = np.linalg.norm(normal)
            if nn < 1e-6:
                continue
            normal = normal / nn
            if abs(d1[2]) < 1e-9:
                continue
            t = -o1[2] / d1[2]
            if t <= 0 or np.hypot(*(o1 + t * d1)[:2]) > MAX_TIP_RADIUS_MM:
                continue
            prior = 0.0 if (c["primary"] and not flip) else (
                NON_PRIMARY_COST + RANK_COST * c["rank"] + (FLIP_COST if flip else 0.0)
            )
            tip = (float(tip[0]), float(tip[1]))
            far = (float(far[0]), float(far[1]))
            if np.linalg.norm(np.subtract(tip, far)) < SHORT_BLOB_PX:
                out.append(BlobOption(
                    tip=tip, far=far, origin=o1, direction=d1, normal=None, offset=None,
                    point=True, wire=False, prior=prior + POINT_ONLY_COST,
                    rank=c["rank"], flip=flip, primary=c["primary"],
                ))
                continue
            out.append(BlobOption(
                tip=tip, far=far, origin=o1, direction=d1, normal=normal, offset=float(normal @ o1),
                point=False, wire=wire_tail(bg_bgr, frame_bgr, tip, calib), prior=prior,
                rank=c["rank"], flip=flip, primary=c["primary"],
            ))
    return out


def _axis_from_planes(normals: np.ndarray, offsets: np.ndarray):
    if len(normals) == 2:
        d = np.cross(normals[0], normals[1])
        dn = np.linalg.norm(d)
        if dn < 1e-6:
            return None
        d = d / dn
        p = np.linalg.solve(np.stack([normals[0], normals[1], d]), np.array([offsets[0], offsets[1], 0.0]))
    else:
        _, _, vh = np.linalg.svd(normals)
        d = vh[-1]
        p, *_ = np.linalg.lstsq(normals, offsets, rcond=None)
    if d[2] < 0:
        d = -d
    return d, p


def score_combination(combo: list[tuple[int, BlobOption | None]], calibration) -> tuple[float, dict] | None:
    """`(cost, geometry)` for one option per camera, or None when the
    combination fails any consistency gate (see module docstring)."""
    used = [k for k, o in combo if o is not None]
    if len(used) < 2:
        return None
    lines = [o for _, o in combo if o is not None and not o.point]
    if len(lines) < 2:
        return None
    normals = np.stack([o.normal for o in lines])
    offsets = np.array([o.offset for o in lines])
    axis = _axis_from_planes(normals, offsets)
    if axis is None:
        return None
    d, p = axis
    if d[2] < math.cos(math.radians(MAX_AXIS_TILT_DEG)):
        return None
    hit = p + (-p[2] / d[2]) * d
    if np.hypot(hit[0], hit[1]) > MAX_HIT_RADIUS_MM:
        return None
    # A tip flagged as a wire tail must land where the OTHER two views put
    # the dart's board entry; with fewer than two other shaft lines there
    # is nothing to judge it by, so the combination is not trusted.
    for k, o in combo:
        if o is None or not o.wire:
            continue
        others = [q for kk, q in combo if q is not None and kk != k and not q.point]
        if len(others) < 2:
            return None
        no = np.stack([q.normal for q in others])
        co = np.array([q.offset for q in others])
        do = np.cross(no[0], no[1])
        do /= np.linalg.norm(do) + 1e-12
        if do[2] < 0:
            do = -do
        if abs(do[2]) < 1e-6:
            return None
        po = np.linalg.solve(np.stack([no[0], no[1], do]), np.array([co[0], co[1], 0.0]))
        entry = po + (-po[2] / do[2]) * do
        if np.linalg.norm(_project(entry, calibration[k]) - np.asarray(o.tip)) > WIRE_OK_PX:
            return None
    top = hit + 60 * d
    plane = max(max(abs(o.normal @ hit - o.offset), abs(o.normal @ top - o.offset)) for o in lines)
    if plane > MAX_PLANE_RESIDUAL_MM:
        return None
    tip_axis: list[float] = []
    heights: list[float] = []
    for k, o in combo:
        if o is None:
            continue
        a = _project(hit, calibration[k])
        v3 = _project(hit + 40 * d, calibration[k]) - a
        v2 = np.subtract(o.far, o.tip)
        if (
            not o.point
            and np.linalg.norm(v3) > 8
            and (v3 @ v2) / (np.linalg.norm(v3) * np.linalg.norm(v2)) < MIN_DIRECTION_COS
        ):
            return None  # this view's blob leans against the 3D axis
        w = np.cross(o.direction, d)
        wn = np.linalg.norm(w)
        if wn < 1e-6:
            return None
        dist = abs((hit - o.origin) @ w) / wn
        # Where along the axis (mm above the board hit) the tip ray passes
        # closest: a short tip is fine up to the barrel, not in the air.
        a_mat = np.array([
            [o.direction @ o.direction, -(o.direction @ d)],
            [o.direction @ d, -(d @ d)],
        ])
        b_vec = np.array([(hit - o.origin) @ o.direction, (hit - o.origin) @ d])
        try:
            _s, height = np.linalg.solve(a_mat, b_vec)
        except np.linalg.LinAlgError:
            return None
        if not (MIN_TIP_HEIGHT_MM <= height <= MAX_TIP_HEIGHT_MM):
            return None
        tip_axis.append(dist)
        heights.append(float(height))
    if max(tip_axis) > MAX_TIP_AXIS_MM:
        return None
    cost = (
        plane + float(np.mean(tip_axis)) + sum(o.prior for o in lines)
        + MISSING_CAMERA_COST * (len(combo) - len(used))
    )
    return cost, {
        "plane_residual_mm": float(plane),
        "max_tip_axis_mm": float(max(tip_axis)),
        "max_tip_height_mm": max(heights),
        "axis_tilt_deg": math.degrees(math.acos(d[2])),
        "board_hit_mm": [float(v) for v in hit],
        "axis_dir": [float(v) for v in d],
    }


def axis_support(bg_bgr: np.ndarray, frame_bgr: np.ndarray, calib, hit, d) -> float:
    """Fraction of the 3D axis, 5-60mm above the board, that shows up as
    changed pixels in a view that took no part in fitting it."""
    import cv2

    g0 = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g0.shape
    n = seen = 0
    for mm in np.linspace(5, 60, 12):
        q = _project(np.asarray(hit) + mm * np.asarray(d), calib)
        x, y = int(round(q[0])), int(round(q[1]))
        if not (2 <= x < w - 2 and 2 <= y < h - 2):
            continue
        n += 1
        seen += float(cv2.absdiff(g0[y - 2:y + 3, x - 2:x + 3], g1[y - 2:y + 3, x - 2:x + 3]).max()) >= AXIS_SUPPORT_DIFF
    return seen / n if n else 0.0


def select_blobs(
    bg_images: dict, frame_images: dict, calibration: dict, precomputed: dict | None = None,
) -> CrossViewPick | None:
    """The most consistent reading of every camera, or None when no
    combination earns trust (Apollo then runs exactly as before)."""
    cams = sorted(frame_images)
    options = {
        k: camera_options(
            np.ascontiguousarray(bg_images[k]), np.ascontiguousarray(frame_images[k]), calibration[k],
            precomputed=(precomputed or {}).get(k),
        )
        for k in cams
    }
    return select_from_options(options, bg_images, frame_images, calibration)


def select_from_options(
    options: dict[int, list[BlobOption]], bg_images: dict, frame_images: dict, calibration: dict,
) -> CrossViewPick | None:
    """`select_blobs()` on readings an engine built itself -- Talos offers
    its own blobs (`opendarts.engines.talos.cross_view`) and is judged by
    exactly the same 3D consistency, pair fallback and axis support."""
    cams = sorted(options)
    best: tuple[float, list, dict] | None = None
    pairs: list[tuple[float, list, dict]] = []
    for pick in itertools.product(*[[None] + options[k] for k in cams]):
        combo = list(zip(cams, pick))
        scored = score_combination(combo, calibration)
        if scored is None:
            continue
        n_used = sum(o is not None for _, o in combo)
        if n_used >= 3 and (best is None or scored[0] < best[0]):
            best = (scored[0], combo, scored[1])
        elif n_used == 2:
            pairs.append((scored[0], combo, scored[1]))
    if best is not None:
        return CrossViewPick(dict(best[1]), best[0], best[2])
    fallback = []
    for cost, combo, geo in pairs:
        if geo["max_tip_height_mm"] > PAIR_MAX_TIP_HEIGHT_MM:
            continue  # two short tips pin the point above the board
        left_out = [k for k, o in combo if o is None][0]
        support = axis_support(
            np.ascontiguousarray(bg_images[left_out]), np.ascontiguousarray(frame_images[left_out]),
            calibration[left_out], geo["board_hit_mm"], np.array(geo["axis_dir"]),
        )
        if support >= MIN_AXIS_SUPPORT:
            fallback.append((
                cost - AXIS_SUPPORT_WEIGHT * support, combo,
                dict(geo, axis_support=support, dropped_camera=left_out),
            ))
    if not fallback:
        return None
    fallback.sort(key=lambda x: x[0])
    if len(fallback) > 1:
        a, b = fallback[0], fallback[1]
        if b[0] - a[0] < PAIR_MARGIN and np.linalg.norm(
            np.subtract(a[2]["board_hit_mm"][:2], b[2]["board_hit_mm"][:2])
        ) > PAIR_DISTINCT_MM:
            return None  # two different pairs, two different darts: no call
    return CrossViewPick(dict(fallback[0][1]), fallback[0][0], fallback[0][2])
