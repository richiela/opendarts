"""Cross-view blob selection for Talos (2026-09-28).

`shaft_line.fit_shaft_line_px()` picks each camera's dart blob on that
one image alone (the first elongated component of the three largest),
so a flight, the dart's own upper shaft, a neighbouring dart or a
wire-line diff can win while the real tip sits in a smaller blob next to
it. Measured on the 2026-09 corpus (1,618 darts the oracle also scored):
533 Talos tips were more than 20px from the oracle's, and in 188 camera
views the tip Talos used sat at the flight end, 100-150px up the dart,
with that camera then thrown out as the ray outlier.

Every camera sees the same physical dart, so this module offers Talos's
OWN top blobs -- its own frozen diff threshold, components and tip-end
rule (`blob_detection`), not Apollo's -- in both end orientations, and
judges every combination with exactly the 3D consistency Apollo uses
(`opendarts.engines.apollo.cross_view.select_from_options()`: one dart
axis shared by the shaft planes, a plausible lean, each blob pointing the
same way as the projected axis, each tip ray near the axis; a 2-camera
pair only when the camera left out sees the axis and both tips are near
the board). Only the geometry is shared: Talos never sees Apollo's
blobs, tips or calls.

Where the chosen blob is Talos's own pick, its own shaft line is kept
whole. A different blob gets Talos's own shaft line through the chosen
end (`shaft_line._shaft_line_through_tip()`), and is only accepted when
its tip sits within SWAP_MAX_TIP_HEIGHT_MM of the board along the axis:
the consistency test accepts a tip ray anywhere up to 40mm along the
axis, which let a barrel stub stand in for the tip (g1-071-DB).

Where no 3-camera combination fits and a pair is used instead, the
camera left out keeps its own TIP (its shaft plane stays out) when that
tip ray passes through the tip the pair found (`_keeps_own_tip()`):
dropping it only because no blob in that view fits the axis as a WHOLE
shaft -- a flight or a neighbouring dart bending its plane -- threw good
tips away.

Blind-audited against the images on all 422 darts the selection changes
(before the own-tip rule): 203 better, 8 worse (all camera drops), 106
equivalent, 105 undecidable from pixels alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opendarts.engines.apollo.cross_view import (
    FLIP_COST,
    MAX_TIP_AXIS_MM,
    NON_PRIMARY_COST,
    POINT_ONLY_COST,
    RANK_COST,
    SHORT_BLOB_PX,
    TOP_K_BLOBS,
    BlobOption,
    CrossViewPick,
    _board_hit_ok,
    _ray,
    select_from_options,
    wire_tail,
)
from opendarts.engines.talos.shaft_line import (
    MIN_SHAFT_SPAN_PX,
    _SHAFT_TIPWARD_FRACTION,
    _shaft_line_through_tip,
    shaft_components,
)

# A replacement blob (not Talos's own pick) must put its tip within this
# far (mm) of the board along the chosen axis. The rejected replacements
# on the corpus sat a median 13.4mm up the axis and none was within 10px
# of the oracle's tip; 5 and 12mm give the same calls.
SWAP_MAX_TIP_HEIGHT_MM = 8.0
# A pick whose tip is within this of Talos's own detection IS that
# detection, and its own shaft line is kept whole.
SAME_DETECTION_PX = 3.0
# A camera the pick leaves out keeps its own tip when that tip ray passes
# within this (mm) of the board point the other cameras put the tip at --
# the same tolerance the consistency test gives any camera's tip ray
# against the axis, but measured at the tip, not anywhere up the shaft.
# On the corpus the dropped cameras sort cleanly by it: within 4mm their
# own tips sit a median ~3px from the oracle's tip (37 cameras, 37
# darts); beyond 5mm, 8-40px (short tips stopping at the barrel, a
# neighbouring dart), and dropping those is right.
OWN_TIP_MAX_MISS_MM = MAX_TIP_AXIS_MM
# Talos's tip-end rule averages this many extreme pixels, and so does
# the far end offered for the flipped orientation.
N_END_POINTS = 3


@dataclass
class TalosBlobOption(BlobOption):
    """A `BlobOption` that remembers the Talos blob it came from, so the
    chosen reading can be turned into Talos's own shaft line."""

    blob: dict | None = None


def _far_end(pts: np.ndarray, principal: np.ndarray, tip) -> tuple[float, float]:
    proj = (pts - pts.mean(0)) @ principal
    lo = pts[np.argsort(proj)[:N_END_POINTS]].mean(0)
    hi = pts[np.argsort(-proj)[:N_END_POINTS]].mean(0)
    far = hi if np.linalg.norm(lo - tip) < np.linalg.norm(hi - tip) else lo
    return float(far[0]), float(far[1])


def talos_blobs(bg_bgr: np.ndarray, frame_bgr: np.ndarray, precomputed=None, k: int = TOP_K_BLOBS) -> list[dict]:
    """Talos's own blob pipeline -- the very components
    `fit_shaft_line_px()` ranks (`shaft_line.shaft_components()`, shared,
    not recomputed) and the same tip-end rule -- but returning the
    top-`k` instead of stopping at the first elongated one. `primary`
    marks the component `fit_shaft_line_px()` itself would choose."""
    from opendarts.engines.talos.blob_detection import (
        MIN_ELONGATION_RATIO,
        TOP_K_AREA_CANDIDATES,
        _component_stats_in_bbox,
        _find_companion_end,
        _locate_tip_in_component,
    )

    comps = shaft_components(bg_bgr, frame_bgr, precomputed)
    if comps is None:
        return []
    labels, opened, origin, bboxes = comps.labels, comps.opened, comps.origin, comps.bboxes
    ranked = comps.candidates
    primary = None
    for _area, i in ranked[:TOP_K_AREA_CANDIDATES]:
        res = _component_stats_in_bbox(labels, i, bboxes[i], opened, origin)
        if res is not None and res[3] >= MIN_ELONGATION_RATIO:
            primary = i
            break
    if primary is None:
        primary = ranked[0][1]  # fit_shaft_line_px() falls back to the largest
    out: list[dict] = []
    for rank, (area, i) in enumerate(ranked[:k]):
        res = _component_stats_in_bbox(labels, i, bboxes[i], opened, origin)
        if res is None:
            continue
        pts, centered, principal, elong = res
        proj = centered @ principal
        others = [(a, j) for a, j in ranked if j != i]
        companion = _find_companion_end(
            labels, opened, others, pts.mean(0), principal,
            float(proj.min()), float(proj.max()), bboxes, origin,
        )
        tip, _diag = _locate_tip_in_component(
            pts, centered, principal, forced_non_tip_end=companion, bg_gray=comps.bg_gray,
        )
        if not np.all(np.isfinite(tip)):
            continue
        out.append({
            "rank": rank,
            "area": int(area),
            "elongation": float(elong),
            "tip": (float(tip[0]), float(tip[1])),
            "far": _far_end(pts, principal, np.asarray(tip)),
            "pts": pts,
            "principal": principal,
            "primary": i == primary,
            "length": float(proj.max() - proj.min()),
        })
    return out


def camera_options(bg_bgr: np.ndarray, frame_bgr: np.ndarray, calib, precomputed=None) -> list[TalosBlobOption]:
    """Every reading of this camera worth testing against the others."""
    out: list[TalosBlobOption] = []
    for b in talos_blobs(bg_bgr, frame_bgr, precomputed):
        base_prior = 0.0 if b["primary"] else NON_PRIMARY_COST + RANK_COST * b["rank"]
        if b["length"] < SHORT_BLOB_PX:
            # Seen end-on its direction is noise: a bare point.
            origin, direction = _ray(b["tip"], calib)
            if _board_hit_ok(origin, direction):
                out.append(TalosBlobOption(
                    tip=b["tip"], far=b["far"], origin=origin, direction=direction,
                    normal=None, offset=None, point=True, wire=False, prior=base_prior + POINT_ONLY_COST,
                    rank=b["rank"], flip=False, primary=b["primary"], blob=b,
                ))
            continue
        for tip, far, flip in ((b["tip"], b["far"], False), (b["far"], b["tip"], True)):
            if np.linalg.norm(np.subtract(tip, far)) < 8:
                continue
            o1, d1 = _ray(tip, calib)
            _o2, d2 = _ray(far, calib)
            normal = np.cross(d1, d2)
            nn = np.linalg.norm(normal)
            if nn < 1e-6 or not _board_hit_ok(o1, d1):
                continue
            normal = normal / nn
            prior = 0.0 if (b["primary"] and not flip) else (
                NON_PRIMARY_COST + RANK_COST * b["rank"] + (FLIP_COST if flip else 0.0)
            )
            out.append(TalosBlobOption(
                tip=tip, far=far, origin=o1, direction=d1, normal=normal, offset=float(normal @ o1),
                point=False, wire=wire_tail(bg_bgr, frame_bgr, tip, calib), prior=prior,
                rank=b["rank"], flip=flip, primary=b["primary"], blob=b,
            ))
    return out


def select_talos_blobs(
    bg_images: dict, frame_images: dict, calibration: dict, precomputed: dict | None = None,
) -> CrossViewPick | None:
    """The most consistent reading of every camera, or None when no
    combination earns trust (Talos then runs exactly as before)."""
    cams = sorted(set(bg_images) & set(frame_images) & set(calibration))
    options = {
        k: camera_options(
            np.ascontiguousarray(bg_images[k]), np.ascontiguousarray(frame_images[k]), calibration[k],
            precomputed=(precomputed or {}).get(k),
        )
        for k in cams
        if bg_images[k].shape == frame_images[k].shape
    }
    if len(options) < 2:
        return None
    return select_from_options(options, bg_images, frame_images, calibration)


def _closest_on_axis(tip, calib, geometry: dict) -> tuple[float, float] | None:
    """(distance mm, height mm along the axis) of the tip ray's closest
    approach to the chosen dart axis."""
    o, d = _ray(tip, calib)
    hit = np.asarray(geometry["board_hit_mm"], float)
    a = np.asarray(geometry["axis_dir"], float)
    w = np.cross(d, a)
    wn = np.linalg.norm(w)
    if wn < 1e-6:
        return None
    a_mat = np.array([[d @ d, -(d @ a)], [d @ a, -(a @ a)]])
    b_vec = np.array([(hit - o) @ d, (hit - o) @ a])
    try:
        _s, height = np.linalg.solve(a_mat, b_vec)
    except np.linalg.LinAlgError:
        return None
    return float(abs((hit - o) @ w) / wn), float(height)


def _fit_from_option(option: TalosBlobOption):
    """Talos's own shaft line through the chosen blob and end."""
    b = option.blob
    line = _shaft_line_through_tip(b["pts"], b["principal"], option.tip)
    if line is None:
        return None
    p1, p2, span, n_shaft = line
    return p1, p2, {
        "p1_px": p1,
        "p2_px": p2,
        "span_px": float(span),
        "component_area_px": b["area"],
        "elongation_ratio": b["elongation"],
        "tip_px": (float(option.tip[0]), float(option.tip[1])),
        "n_shaft_pixels": int(n_shaft),
        "shaft_tipward_fraction": _SHAFT_TIPWARD_FRACTION,
        "dropped": bool(span < MIN_SHAFT_SPAN_PX),
        "cross_view_pick": [option.rank, int(option.flip)],
    }


def camera_fit(own, calib, pick: CrossViewPick, cam: int):
    """One camera's `fit_shaft_line_px()`-shaped result under a cross-view
    pick: `own` is Talos's own fit for this camera (or None)."""
    option = pick.picks.get(cam)
    if option is None:
        if not _keeps_own_tip(own, calib, pick):
            return None
        # The tip goes in as a ray; the shaft plane, which is what failed
        # the consistency test, stays out of the axis.
        p1, p2, diag = own
        return p1, p2, {**diag, "dropped": True, "kept_own_tip": True}
    if (
        own is not None and not option.flip and not option.wire
        and np.linalg.norm(np.subtract(own[2]["tip_px"], option.tip)) < SAME_DETECTION_PX
    ):
        return own
    fitted = _fit_from_option(option)
    if fitted is None:
        return None
    closest = _closest_on_axis(option.tip, calib, pick.geometry)
    if closest is None or closest[1] > SWAP_MAX_TIP_HEIGHT_MM:
        return None  # the replacement blob is up the shaft, not the tip
    return fitted


def _own_tip_miss_mm(own, calib, pick: CrossViewPick) -> float | None:
    """How far (mm) this camera's own tip ray passes from the board point
    the rest of the pick puts the tip at."""
    if own is None or not own[2].get("tip_px"):
        return None
    origin, direction = _ray(own[2]["tip_px"], calib)
    hit = np.asarray(pick.geometry["board_hit_mm"], float)
    return float(np.linalg.norm(np.cross(hit - origin, direction)))


def _keeps_own_tip(own, calib, pick: CrossViewPick) -> bool:
    miss = _own_tip_miss_mm(own, calib, pick)
    return miss is not None and miss <= OWN_TIP_MAX_MISS_MM
