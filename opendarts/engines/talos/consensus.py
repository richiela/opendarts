"""Talos's own opinion about which 2D pixel is "the observation" and
how to combine multiple cameras' board-plane hits when they disagree --
as opposed to `opendarts.engines.talos.plane_geometry`, which is pure
line/plane math with no opinion about pixel selection, and
`opendarts.engines.talos.shaft_line`, which finds the blob and its
whole-line direction. This module answers eleven narrower questions once a
shaft line already exists:

1. **`_outward_edge_pixel()`** -- NOT the live 2D observation as of
   2026-08-24. Built to avoid picking a silhouette *corner* (argmax
   board-radius) by taking the centroid of near-tip pixels within
   `OUTWARD_CAP_MM` of max board-radius. On the 374-throw opendarts corpus
   that cap is a one-sided outward bias (median Δr vs AD +5.2mm per
   camera, +2.3mm after triangulation); the shaft-line `tip_px`
   (centerline) is unbiased (+0.11mm). Live `engine.score()` uses
   `tip_px` as the ray pixel. This helper stays for tests and as the
   measured record of the rejected observation.
2. **`_pair_if_sector_compromise()`** -- once 3-ray triangulation has
   already run (`opendarts.pipeline.score_dart()`), was its landed sector
   one that no single camera's own ray∩Z=0 voted for, while two cameras
   independently agreed on a different, real sector? That specific
   pattern (a "radial-wire straddle" in the combined 3-ray average) is
   corrected by trusting the agreeing pair's own mean instead -- see the
   function's own docstring for exactly which patterns this does and
   does NOT fire on (same-sector-different-ring is left alone; blind
   always-majority is explicitly rejected as Apollo's correlated-bias
   trap).
3. **`_rescue_outside_from_centerlines()`** -- if the (post-cap, post-
   pair) result scored *outside*, and >=2 cameras' *pre-cap* centerline
   ray∩Z=0 hits are still on the board, take their mean. Caps can walk
   off the board when the silhouette corner is past the double wire;
   the centerline has not. Not a blanket "skip outside caps" rewrite
   (that one broke a throw the 3-ray already had right).
4. **`_lock_unanimous_centerline_sector()`** -- if *all three* on-board
   centerlines share sector S and the result left S, lock to the CL
   mean. 2-of-3 CL lock is Apollo's correlated-bias trap (+2/-6 on the
   old 169); 3/3 is the pattern where a single camera's cap pulled the
   3-ray onto a singleton the centerlines never voted for.
5. **`drop_far_cap_rays()` / `one_cam_z0()`** -- a cap ray whose
   board-plane hit is more than `FAR_CAP_RADIUS_MM` (250mm) from
   bullseye is not a dart on the board; it is a leftover-dart /
   exploded-blob projection. Drop it before triangulation. If only one
   camera remains, that camera's own ray∩Z=0 is the observation.
6. **`_lock_cap_walked_radial()`** -- onboard CLs unanimously in sector
   S, onboard caps unanimously in a *different* sector T, dart axis
   also in S, result left S. The silhouette walked every cap across a
   radial wire; the shaft centerlines and 3D axis did not. Not the
   2-of-3 CL trap: that trap's onboard caps are not unanimous (one cap
   still in the true pie is what saved the 3-ray). Measured 2026-08-13: new 180 +1/-0 (`throw_1786667897769`);
   old 169 +0/-0 including the D-trap `throw_1786593923839`.
7. **`_lock_shaft_snap_consensus()`** -- snap each cap's board-hit onto
   that camera's shaft-plane ∩ Z=0 line; if the snap mean, the onboard
   CL majority bed, and the dart-axis bed all agree on B, and the
   cap-ray result left B, use the snap mean. Ungated snap-as-primary
   is -57 new; snap==CL-maj without the axis gate stole
   `throw_1786580444656` (17→2). Axis agreeing is the independent
   check (that throw's axis was the true 17). Measured 2026-08-13:
   new 180 +1/-0 (`throw_1786666602111`); old 169 +0/-0.
8. **`_lock_centerline_ring()`** -- onboard CLs in the result sector:
   if >=2 share ring R != the result ring and the result is farther
   *out* than those holders' mean (`d_r > 0`), lock to that mean --
   unless an in-sector CL already corroborates a different axis ring
   (026-T7: 2 inner CLs vs axis+cam0 treble, truth treble; 0408695:
   3/3 inner vs a lonely treble axis, truth inner). Leftover other-
   sector CLs are ignored. The 2.7mm `CL_RING_OUTWARD_MM` gate was the
   measured gap between cap-bias steals and true outer walks on the
   old 300 (`4847073` 1.57, `7737057` 2.19 vs remaining-gains min
   3.21). After 2026-08-24 skip-cap, residual dR on this 374 is
   0.03–1.7mm (the 2.7mm quantity was cap-vs-CL); n2 outer-to-treble
   (`7737057`) is net +3/-0 on this corpus, so the live gate is
   `d_r > 0` with the corroboration skip kept. Constant 2.7 stays as
   the historical record. Axis, when present, must still be in
   sector s (`0450821`). A second gate (3 onboard CLs, majority-in-
   sector + dart axis both treble, result `single_outer`) remains for
   d_r <= 0; with skip-cap the first branch usually fires first.
9. **`_lock_axis_ring_lonely_cl()`** -- fewer than 2 onboard CLs land
   in the result sector, dart axis is same sector different ring →
   axis XY. The 3-ray sector came from a walked cap; the shaft planes
   still see the other ring. Measured 2026-08-14: +1/-0
   (`throw_1786666297806`).
10. **`_lock_single_cl_double_to_outer()`** -- exactly one onboard CL
    in the result sector, that CL is `single_outer`, result is
    `double` → that CL. The line-plane fallback (axis ∩ Z=0) walked
    into the double band; the only surviving shaft did not. Measured
    2026-08-14: +1/-0 (`throw_1786689914138`), including without an
    observation==fallback gate.
11. **`_lock_split_centerline_mean()`** -- exactly two onboard CLs in
    the result sector, 1-1 on ring, use their mean. The cap-ray sat
    on one side of a ring wire; the two shafts straddle it. Measured
    2026-08-14: +1/-0 (`throw_1786666443026` 4/outer → 4/double).
    Other 1-1 splits on the 300 already have current == mean bed
    (`6373542`, `7731513`).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from opendarts.engines.talos.plane_geometry import (
    _pixel_board_xy,
    snap_xy_to_plane_line,
)
from opendarts.geometry.board import (
    DOUBLE_OUTER_RADIUS_MM,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    sector_ring_for_point,
)
from opendarts.pipeline import CameraCalibration, ScoreResult

# Neighborhood of the opened-mask tip in which the outward-edge pixel
# is chosen. Same order as tip_detection.END_WINDOW_MIN_PX (6) -- one
# needle-end window, not a radius fudge. 8px and 12px produced the
# same 143/169 on clean/; 12px is the window, not an AD sweep.
OUTWARD_EDGE_WINDOW_PX = 12.0

# Board-mm slice of the outward cap whose image centroid is the 2D
# observation. Argmax-r is a silhouette *corner* (radial + tangential).
# A steel-tip needle is ~2-2.4mm across; 1.0mm is that half-width in
# board plane, so the cap is the outward end-face not the corner.
# Measured on data/archive/clean, bull-gated:
#   argmax-r:     144/169
#   cap 0.5mm:    144  (+2 -2)
#   cap 1.0mm:    146  (+2 -0)
#   cap 1.5mm:    143  (+1 -2)
#   cap 2.0mm:    143  (+2 -3)
#   cap 3.0mm:    141  (+4 -7)
# 1.0 sits in the measured gap; not swept against AD beyond that.
OUTWARD_CAP_MM = 1.0

# Drop a camera's cap ray from triangulation when its board-plane hit
# is this far past the double wire. Real near-outsides live in
# [170, 220); leftover / exploded-blob projections start a gap above
# that. Measured 2026-08-13 on one recorded session
# + old 169: gate 200 was +2/-1 (a real outside pulled on-board);
# 250/300/340 were then +2/-0 and equivalent. Re-measured 2026-08-17
# on living clean/ 885:
#   250: +4/-0 vs 340 (the recorded S4 throw, the recorded D10 throw,
#        the recorded S5 throw, the recorded T11 throw). Junk rays
#        on those throws sit at r≈291–331, inside the old [270, 345)
#        cluster that 340 was chosen to keep.
#   300: +3/-0 (misses 024-S4, junk cam r≈291).
# 250 is 80mm past double-outer -- still well clear of a real outside.
FAR_CAP_RADIUS_MM = 250.0

# Historical cap-vs-CL gap (2026-08-14): 2.7mm sat between cap-bias
# steals (dR 1.57 / 2.19) and real gains (dR 3.21). 2026-08-24: the
# live observation is the centerline tip, not the cap; the live ring-
# lock gate is `d_r > 0` with the axis_corroborated discriminator.
# Constant kept because tests still name the old measurement.
CL_RING_OUTWARD_MM = 2.7



def _outward_edge_pixel(
    pts: np.ndarray,
    tip_px: tuple[float, float],
    calib: CameraCalibration,
    window_px: float = OUTWARD_EDGE_WINDOW_PX,
    cap_mm: float = OUTWARD_CAP_MM,
) -> tuple[float, float] | None:
    """Centroid of near-tip opened pixels within cap_mm of max board-r."""
    tip = np.asarray(tip_px, dtype=np.float64)
    dist = np.hypot(pts[:, 0] - tip[0], pts[:, 1] - tip[1])
    near = pts[dist <= window_px]
    if len(near) < 3:
        near = pts
    scored: list[tuple[float, float, float]] = []
    for p in near:
        xy = _pixel_board_xy(p, calib)
        if xy is None:
            continue
        scored.append((float(p[0]), float(p[1]), math.hypot(xy[0], xy[1])))
    if not scored:
        return None
    max_r = max(s[2] for s in scored)
    cap = [s for s in scored if s[2] >= max_r - cap_mm]
    return (
        sum(s[0] for s in cap) / len(cap),
        sum(s[1] for s in cap) / len(cap),
    )


def _pair_if_sector_compromise(ray_pixels, calibration, ray_scored):
    """If 3-ray triangulation landed in a pie no camera voted for, use the pair.

    Per-cam observation is that camera's ray ∩ Z=0. When two cameras agree
    on a sector+ring and the 3-ray result is a *different sector* that
    neither the pair nor the singleton reported, the combined point is a
    bad average across a radial wire. Trust the pair that already agrees.

    Same-sector ring compromises are left alone: on data/archive/clean
 those include a real 3-ray-correct / pair-wrong
    case (6/single_inner vs pair 6/treble). Sector-only: +3 -0 vs current.
    Majority-always is +8 -6 (Apollo's correlated-bias trap). Not fitted to oracle labels.
    """
    if (
        not ray_scored.ok
        or ray_scored.n_cameras_used < 3
        or len(ray_pixels) < 3
    ):
        return None
    per_bed: dict[int, tuple] = {}
    for cam, pixel in ray_pixels.items():
        if cam not in calibration:
            continue
        xy = _pixel_board_xy(pixel, calibration[cam])
        if xy is None:
            continue
        per_bed[cam] = sector_ring_for_point(xy[0], xy[1])
    if len(per_bed) < 3:
        return None
    (maj, n_maj), = Counter(per_bed.values()).most_common(1)
    if n_maj != 2:
        return None
    three_bed = (ray_scored.sector, ray_scored.ring)
    if three_bed == maj:
        return None
    if maj[0] is None:
        # Pair voted bull/outside — not a radial-wire straddle.
        return None
    if three_bed[0] == maj[0]:
        # Same sector, different ring. Do not switch.
        return None
    if three_bed in per_bed.values():
        # 3-ray matched the singleton — ambiguous 2-wrong-1-right vs
        # 2-right-1-wrong. Not a compromise.
        return None
    holders = [cam for cam, bed in per_bed.items() if bed == maj]
    if len(holders) < 2:
        return None
    # Pair triangulation (closest-point, drop Z) can slide back across a
    # ring wire even when both cameras' ray∩Z=0 already sit in `maj`
    # (measured: throw_1786581538226, 1.04mm into single_inner while both
    # Z=0 hits are 4/treble). The two board-plane hits *are* the
    # observations; their mean stays in the agreed bed.
    xys = []
    for cam in holders:
        xy = _pixel_board_xy(ray_pixels[cam], calibration[cam])
        if xy is not None:
            xys.append(xy)
    if len(xys) < 2:
        return None
    mean = (sum(p[0] for p in xys) / len(xys), sum(p[1] for p in xys) / len(xys))
    sector, ring = sector_ring_for_point(mean[0], mean[1])
    return ScoreResult(
        ok=True,
        sector=sector,
        ring=ring,
        board_xy_mm=mean,
        triangulation=None,
        n_cameras_used=len(xys),
        cameras_used=tuple(holders[:len(xys)]),
        reason=(
            "2-of-3 sector compromise: 3-ray landed in a pie no camera "
            "voted for; using mean of the agreeing cameras' ray∩Z=0"
        ),
    )


def _centerline_board_hits(cl_pixels, calibration):
    """ray∩Z=0 of each camera's pre-cap shaft-tip pixel."""
    hits: dict[int, tuple[float, float]] = {}
    for cam, pixel in cl_pixels.items():
        if cam not in calibration:
            continue
        xy = _pixel_board_xy(pixel, calibration[cam])
        if xy is not None:
            hits[int(cam)] = (float(xy[0]), float(xy[1]))
    return hits


def _mean_xy(xys) -> tuple[float, float]:
    return (
        sum(p[0] for p in xys) / len(xys),
        sum(p[1] for p in xys) / len(xys),
    )


def _score_result_from_xy(xy, cameras, reason: str) -> ScoreResult:
    sector, ring = sector_ring_for_point(xy[0], xy[1])
    return ScoreResult(
        ok=True,
        sector=sector,
        ring=ring,
        board_xy_mm=(float(xy[0]), float(xy[1])),
        triangulation=None,
        n_cameras_used=len(cameras),
        cameras_used=tuple(cameras),
        reason=reason,
    )


def _rescue_outside_from_centerlines(cl_pixels, calibration, scored):
    """If the result scored outside, mean of >=2 on-board centerline Z=0 hits.

    Caps can walk past the double wire when the silhouette corner is the
    outward pick; the pre-cap centerline often has not. Measured
    2026-08-13:

      new 120: +1/-0  (throw_1786665041430, 8/single_outer)
      old 169: +0/-0, zero XY-level fires

    A looser "skip any cap that landed outside while CL is on-board"
    rewrite of the *pixels* was +2/-1 on the same 120 -- the loss was
    throw_1786666628283, which 3-ray already had as 3/double. This
    rescue only fires when the *combined result* is already outside, so
    that throw is untouched.
    """
    if scored is None or not getattr(scored, "ok", False):
        return None
    if getattr(scored, "ring", None) != "outside":
        return None
    hits = _centerline_board_hits(cl_pixels, calibration)
    onboard = [
        (cam, xy) for cam, xy in hits.items()
        if math.hypot(xy[0], xy[1]) <= DOUBLE_OUTER_RADIUS_MM
    ]
    if len(onboard) < 2:
        return None
    mean = _mean_xy([xy for _, xy in onboard])
    return _score_result_from_xy(
        mean,
        [cam for cam, _ in onboard],
        (
            "outside rescue: 3-ray/pair landed outside; using mean of "
            f"{len(onboard)} on-board centerline ray∩Z=0 hits"
        ),
    )


def _lock_unanimous_centerline_sector(cl_pixels, calibration, scored):
    """If all 3 on-board CLs share sector S and the result left S, CL mean.

    The cap pick is a tangential extreme; one camera's cap can pull the
    3-ray across a radial wire onto a sector *no centerline voted for*.
    That is not the 2-of-3 CL majority trap: requiring 3/3 on-board CLs
    in S means throw_1786593923839 (2 CL on 8, truth 16, cap-saved) cannot
    fire. Measured 2026-08-13:

      new 120: +1/-0  (throw_1786666585547, all 3 CL 12, cap pulled to 5)
      old 169: +0/-0, zero XY-level fires
      2-of-3 CL lock (rejected): +3/-0 new, +2/-6 old
    """
    if scored is None or not getattr(scored, "ok", False):
        return None
    hits = _centerline_board_hits(cl_pixels, calibration)
    if len(hits) != 3:
        return None
    beds = []
    for xy in hits.values():
        if math.hypot(xy[0], xy[1]) > DOUBLE_OUTER_RADIUS_MM:
            return None
        beds.append(sector_ring_for_point(xy[0], xy[1]))
    sectors = {b[0] for b in beds}
    if None in sectors or len(sectors) != 1:
        return None
    cl_s = next(iter(sectors))
    if getattr(scored, "sector", None) == cl_s:
        return None
    mean = _mean_xy(list(hits.values()))
    return _score_result_from_xy(
        mean,
        sorted(hits),
        (
            "unanimous centerline sector lock: all 3 on-board centerlines "
            f"in sector {cl_s}; 3-ray/pair left that pie; using CL mean"
        ),
    )


def _onboard_beds(pixels, calibration):
    """On-board ray∩Z=0 hits as (cam, xy, (sector, ring))."""
    hits = _centerline_board_hits(pixels, calibration)
    onboard = []
    for cam, xy in hits.items():
        if math.hypot(xy[0], xy[1]) <= DOUBLE_OUTER_RADIUS_MM:
            onboard.append((cam, xy, sector_ring_for_point(xy[0], xy[1])))
    return onboard


def _lock_pair_unanimous_cl_axis(cl_pixels, calibration, scored, axis_xy=None):
    """Exactly 2 onboard CLs, same sector S, axis in S, result left S -> CL mean.

    The n_onboard==3 unanimous case is `_lock_unanimous_centerline_sector`
    (no axis gate needed there). This is the 2-onboard complement -- the
    third camera's centerline is junk (offboard leftover, typically also
    far-dropped from cap triangulation) or missing entirely, both
    surviving shafts agree on the pie, and the reconstructed 3D dart
    axis independently agrees -- only the cap-ray pair crossed a radial
    wire (the outward cap is a tangential extreme).

    The axis gate is what separates this from the naive ">=2 onboard CLs
    agree -> CL mean" adjacent lock (measured +4/-4 on 885): unanimity
    over ONBOARD CLs blocks the 2-of-3 majority trap
    (throw_1786666300736: 3 onboard CLs 5,5,12; the recorded S4 throw;
    the recorded S5 throw; throw_1786667985388: all 3-onboard splits),
    and the axis gate blocks throw_1786666308551 (2 onboard CLs both 12,
    axis and truth 9). Measured 2026-08-17 on living clean/ 996:
    +3/-0 (the recorded S5 throw, the recorded S8 throw,
    another S8 throw); all four naive-lock canaries unchanged.
    """
    if scored is None or not getattr(scored, "ok", False) or axis_xy is None:
        return None
    onboard = _onboard_beds(cl_pixels, calibration)
    if len(onboard) != 2:
        return None
    sectors = {bed[0] for _, _, bed in onboard}
    if None in sectors or len(sectors) != 1:
        return None
    s = next(iter(sectors))
    if getattr(scored, "sector", None) == s:
        return None
    axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
    if axis_bed[0] != s:
        return None
    mean = _mean_xy([xy for _, xy, _ in onboard])
    return _score_result_from_xy(
        mean,
        [cam for cam, _, _ in onboard],
        (
            "pair unanimous centerline sector lock: both onboard "
            f"centerlines and the dart axis in sector {s}; cap-ray "
            "left that pie; using CL mean"
        ),
    )


def _rescue_outside_lonely_cl_axis(
    cl_pixels, calibration, scored, axis_xy=None, far_cams=None,
):
    """Result outside, exactly 1 onboard CL, a far-dropped junk camera,
    and the dart axis in that CL's bed -> that CL.

    `_rescue_outside_from_centerlines` needs >=2 onboard CLs. After a
    junk camera is far-dropped, a real on-board dart can be left with
    only ONE onboard centerline while the surviving pair's walked cap
    slides the triangulation just past the double wire. Two gates, both
    required, each blocking a real measured loss:

    - The 3D dart axis must land in the SAME bed as the lone centerline
      (true-outside canary throw_1786668048648: 1 onboard CL 17/double
      but axis outside -- does not fire).
    - At least one camera must have been far-dropped (`far_cams`): the
      rescue's whole premise is that the missing corroboration is a
      JUNK view, not a real one. the recorded outside throw (truth
      outside, 10.8mm past the wire) has 1 onboard CL at r=169.5 and
      axis 18/double, but its other two CLs are real near-outside
      observations (r=176/196, nothing far-dropped) -- two real views
      voting outside must not be overridden by the one that sits
      0.5mm inside the wire.

    Measured 2026-08-17 on living clean/ 996: +2/-0
    (the recorded outside throw d_wire 0.94, the recorded outside throw
    d_wire 0.14, both with a genuine r~320 junk camera far-dropped);
    all 84 correct outside results unchanged.
    """
    if scored is None or not getattr(scored, "ok", False) or axis_xy is None:
        return None
    if not far_cams:
        return None
    if getattr(scored, "ring", None) != "outside":
        return None
    onboard = _onboard_beds(cl_pixels, calibration)
    if len(onboard) != 1:
        return None
    cam, xy, bed = onboard[0]
    axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
    if axis_bed != bed:
        return None
    return _score_result_from_xy(
        xy,
        [cam],
        (
            "one-CL outside rescue: result was outside; the only onboard "
            f"centerline and the dart axis agree on {bed[0]}/{bed[1]}; "
            "using that CL"
        ),
    )


def _lock_cap_walked_radial(
    cl_pixels, cap_pixels, calibration, scored, axis_xy=None,
):
    """Caps all in T, CLs all in S, axis in S, result left S -> CL mean.

    Tangential silhouette walked every onboard cap across a radial wire;
    the shaft centerlines and the reconstructed dart axis did not.
    2-of-3 CL lock without this cap-unanimity gate is Apollo's
    correlated-bias trap (`throw_1786593923839`: CLs both 8, one cap
    still in 16, 3-ray correctly 16). Measured 2026-08-13:

      new 180: +1/-0  (throw_1786667897769, 15/single_inner)
      old 169: +0/-0
    """
    if scored is None or not getattr(scored, "ok", False):
        return None
    if not cap_pixels or axis_xy is None:
        return None
    cl_on = _onboard_beds(cl_pixels, calibration)
    cap_on = _onboard_beds(cap_pixels, calibration)
    if len(cl_on) < 2 or len(cap_on) < 2:
        return None
    cl_s = {bed[0] for _, _, bed in cl_on}
    cap_s = {bed[0] for _, _, bed in cap_on}
    if None in cl_s or None in cap_s or len(cl_s) != 1 or len(cap_s) != 1:
        return None
    s = next(iter(cl_s))
    t = next(iter(cap_s))
    if s == t:
        return None
    axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
    if axis_bed[0] != s:
        return None
    if getattr(scored, "sector", None) == s:
        return None
    mean = _mean_xy([xy for _, xy, _ in cl_on])
    return _score_result_from_xy(
        mean,
        [cam for cam, _, _ in cl_on],
        (
            "cap walked radial: onboard caps all in sector "
            f"{t}, onboard centerlines and dart axis in {s}; "
            "using CL mean"
        ),
    )


def _snap_cap_mean(cap_pixels, planes, calibration):
    """Mean of cap ray∩Z=0 hits snapped onto each camera's shaft board-line."""
    if not cap_pixels or not planes:
        return None
    pts = []
    cams = []
    for item in planes:
        cam, n, c = item
        px = cap_pixels.get(cam)
        if px is None or cam not in calibration:
            continue
        xy = _pixel_board_xy(px, calibration[cam])
        if xy is None:
            continue
        snapped = snap_xy_to_plane_line(xy, (n, c), 0.0)
        if snapped is None:
            continue
        pts.append(snapped)
        cams.append(int(cam))
    if len(pts) < 2:
        return None
    return _mean_xy(pts), cams


def _lock_shaft_snap_consensus(
    cl_pixels, cap_pixels, planes, calibration, scored, axis_xy=None,
):
    """Snap mean, CL-majority bed, and dart-axis bed all agree; result left.

    Measured 2026-08-13:
      new 180: +1/-0  (throw_1786666602111, 6/single_outer)
      old 169: +0/-0
    Ungated snap was -57 new. Snap==CL-maj without axis stole
    throw_1786580444656 (axis was the true 17; snap/CL followed 2).
    """
    if scored is None or not getattr(scored, "ok", False):
        return None
    if not cap_pixels or axis_xy is None or not planes:
        return None
    snapped = _snap_cap_mean(cap_pixels, planes, calibration)
    if snapped is None:
        return None
    snap_xy, cams = snapped
    snap_bed = sector_ring_for_point(snap_xy[0], snap_xy[1])
    axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
    cl_on = _onboard_beds(cl_pixels, calibration)
    if len(cl_on) < 2:
        return None
    beds = [bed for _, _, bed in cl_on]
    (maj, n_maj), = Counter(beds).most_common(1)
    if n_maj < 2 or maj is None or maj[0] is None:
        return None
    if snap_bed != maj or axis_bed != maj:
        return None
    cur = (getattr(scored, "sector", None), getattr(scored, "ring", None))
    if cur == maj:
        return None
    return _score_result_from_xy(
        snap_xy,
        cams,
        (
            "shaft snap consensus: cap hits snapped onto shaft-plane "
            f"board-lines, CL majority, and dart axis all in {maj}; "
            "cap-ray result left that bed"
        ),
    )


def _lock_centerline_ring(cl_pixels, calibration, scored, axis_xy=None):
    """Majority of CLs in the result sector, cap-ray farther out -> CL mean.

    Onboard CLs whose sector matches the result: if >=2 share ring R
    != the result ring, and the result is farther out than those
    holders' mean (`d_r > 0`), lock to that mean -- unless an
    in-sector CL already corroborates a different axis ring (026-T7
    vs 0408695). Leftover other-sector CLs are ignored. Axis, when
    present, must still be in sector s (`0450821`).
    `CL_RING_OUTWARD_MM` (2.7) is the historical cap-bias gate; live
    post skip-cap is `d_r > 0` (see module docstring item 8).
    """
    if scored is None or not getattr(scored, "ok", False):
        return None
    xy_s = getattr(scored, "board_xy_mm", None)
    if xy_s is None:
        return None
    s = getattr(scored, "sector", None)
    if s is None:
        return None
    cl_on = _onboard_beds(cl_pixels, calibration)
    same = [(cam, xy, bed) for cam, xy, bed in cl_on if bed[0] == s]
    if len(same) < 2:
        return None
    (maj, n_maj), = Counter(bed[1] for _, _, bed in same).most_common(1)
    if n_maj < 2 or maj == getattr(scored, "ring", None):
        return None
    holders = [(cam, xy) for cam, xy, bed in same if bed[1] == maj]
    mean = _mean_xy([xy for _, xy in holders])
    d_r = math.hypot(xy_s[0], xy_s[1]) - math.hypot(mean[0], mean[1])
    axis_sector = None
    axis_ring = None
    if axis_xy is not None:
        axis_sector, axis_ring = sector_ring_for_point(
            float(axis_xy[0]), float(axis_xy[1])
        )
    # 2026-08-18 -- don't lock against an axis that an in-sector CL
    # already corroborates. 026-T7 is 2 inner CLs vs axis+cam0 treble
    # (truth treble); 0408695 is 3/3 inner vs a lonely treble axis
    # (truth inner). Same "axis ring != CL majority" shape, opposite
    # truth; the discriminator is whether any in-sector shaft agrees
    # with the axis. Not a reason to lower CL_RING_OUTWARD_MM.
    axis_corroborated = (
        axis_xy is not None
        and axis_ring != maj
        and any(bed[1] == axis_ring for _, _, bed in same)
    )
    if (
        d_r > 0.0
        and (axis_xy is None or axis_sector == s)
        and not axis_corroborated
    ):
        return _score_result_from_xy(
            mean,
            [cam for cam, _ in holders],
            (
                "centerline ring lock: "
                f"{n_maj} centerlines in sector {s}/{maj}; cap-ray "
                f"{d_r:.1f}mm farther out; using those CLs' mean"
            ),
        )
    if (
        axis_xy is None
        or d_r <= 0.0
        or len(cl_on) < 3
        or getattr(scored, "ring", None) != "single_outer"
        or maj != "treble"
    ):
        return None
    axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
    if axis_bed != (s, "treble"):
        return None
    return _score_result_from_xy(
        mean,
        [cam for cam, _ in holders],
        (
            "centerline ring lock: 3 onboard CLs, "
            f"{n_maj} in sector {s}/treble and dart axis treble; "
            f"cap-ray {d_r:.1f}mm into single_outer; using those CLs' mean"
        ),
    )


def _lock_axis_ring_lonely_cl(cl_pixels, calibration, scored, axis_xy=None):
    """<2 CLs in the result sector, axis same pie different ring -> axis XY.

    The combined cap-ray result landed in a pie that at most one shaft
    centerline agrees with; the 3D dart axis (shaft planes, not cap
    pixels) still sees a different ring of that pie. Measured 2026-08-14
    on the 300: +1/-0 (throw_1786666297806 20/treble -> 20/single_inner).
    n_same>=2 is the 2-of-3 / split-CL trap (6373542, 7731513).
    """
    if scored is None or not getattr(scored, "ok", False) or axis_xy is None:
        return None
    s = getattr(scored, "sector", None)
    if s is None:
        return None
    axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
    if axis_bed[0] != s or axis_bed[1] == getattr(scored, "ring", None):
        return None
    cl_on = _onboard_beds(cl_pixels, calibration)
    n_same = sum(1 for _, _, bed in cl_on if bed[0] == s)
    if n_same >= 2:
        return None
    # 2026-08-18 -- the result pie is not "lonely" if some OTHER
    # onboard sector already has a CL majority. 0427489: 2 CLs in
    # sector 10 (truth) vs 1 CL + axis in 15; locking to the axis
    # stole the cap-ray. Skip when another sector has strictly more
    # CLs than the result pie.
    other_counts = Counter(
        bed[0] for _, _, bed in cl_on if bed[0] != s
    )
    if other_counts and max(other_counts.values()) > n_same:
        return None
    return _score_result_from_xy(
        (float(axis_xy[0]), float(axis_xy[1])),
        [cam for cam, _, bed in cl_on if bed[0] == s],
        (
            "axis ring lonely centerline: "
            f"{n_same} onboard CL in sector {s}; dart axis "
            f"{axis_bed[1]}; cap-ray {getattr(scored, 'ring', None)}"
        ),
    )


def _lock_single_cl_double_to_outer(cl_pixels, calibration, scored):
    """Exactly one CL in-sector, that CL outer, result double -> that CL.

    line_plane_fallback (axis ∩ Z=0) can sit in the double band while
    the only surviving shaft centerline is still single_outer. Measured
    2026-08-14 on the 300: +1/-0 (throw_1786689914138), and the same
    pattern with no fallback-observation gate is still +1/-0.
    """
    if scored is None or not getattr(scored, "ok", False):
        return None
    if getattr(scored, "ring", None) != "double":
        return None
    s = getattr(scored, "sector", None)
    if s is None:
        return None
    cl_on = _onboard_beds(cl_pixels, calibration)
    same = [(cam, xy, bed) for cam, xy, bed in cl_on if bed[0] == s]
    if len(same) != 1:
        return None
    cam, xy, bed = same[0]
    if bed[1] != "single_outer":
        return None
    return _score_result_from_xy(
        xy,
        [cam],
        (
            "single centerline double-to-outer: only onboard CL in "
            f"sector {s} is single_outer; cap-ray/fallback was double"
        ),
    )


# Two in-sector CLs that straddle a ring wire sit ~8mm apart (treble
# or double width). On the 374-throw opendarts corpus, 2-onboard-CL Δr is
# p50=4.9mm p90=10.8mm; leftover-dart 068-S10 is 46mm (the max). A
# split-mean of two CLs 46mm apart is not a wire straddle -- it is two
# different darts. Above this gate, do not average; if the dart axis
# is in the same sector, use the axis (068: axis r=102.8 treble vs
# mean r=75 inner).
SPLIT_MAX_RADIUS_DISAGREE_MM = 20.0

# One regulation treble/double band (8mm). A 2-1 in-sector ring vote
# whose singleton is closer than this is a wire straddle (013-S1
# Δr=5.5, 030-S5 7.3, g1-055-D2 3.9) -- triangulation already picked
# a side, often the true one. A leftover blob in the same pie sits a
# full band away (014-T18 Δr=12.4, 016-S18 12.7). Not a millimetre
# fudge of 97.5/160.5; this is the board model's own ring width.
INSECTOR_RING_MAJORITY_MIN_LEFTOVER_DR_MM = (
    TREBLE_OUTER_RADIUS_MM - TREBLE_INNER_RADIUS_MM
)


def _lock_split_centerline_mean(cl_pixels, calibration, scored, axis_xy=None):
    """Two in-sector CLs 1-1 on ring -> their mean, unless leftover.

    The cap-ray (or slide) sat on one side of a ring wire; the two
    shaft centerlines straddle it. Do not let the silhouette pick a
    side -- use the shaft mean. Measured 2026-08-14 on the 300:
    +1/-0 (throw_1786666443026 4/single_outer -> 4/double). The other
    1-1 splits already have current == mean bed (6373542, 7731513).

    2026-08-24: if the two CLs disagree in radius by more than
    SPLIT_MAX_RADIUS_DISAGREE_MM they are not a wire straddle. Use the
    dart axis when it sits in the same sector (068-S10, +1/-0).
    """
    if scored is None or not getattr(scored, "ok", False):
        return None
    s = getattr(scored, "sector", None)
    if s is None:
        return None
    cl_on = _onboard_beds(cl_pixels, calibration)
    same = [(cam, xy, bed) for cam, xy, bed in cl_on if bed[0] == s]
    if len(same) != 2:
        return None
    rings = {bed[1] for _, _, bed in same}
    if len(rings) != 2:
        return None
    r0 = math.hypot(*same[0][1])
    r1 = math.hypot(*same[1][1])
    if abs(r0 - r1) > SPLIT_MAX_RADIUS_DISAGREE_MM:
        if axis_xy is None:
            return None
        axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
        if axis_bed[0] != s:
            return None
        cur = (s, getattr(scored, "ring", None))
        if axis_bed == cur:
            return None
        return _score_result_from_xy(
            (float(axis_xy[0]), float(axis_xy[1])),
            [cam for cam, _, _ in same],
            (
                "split centerline leftover: 2 onboard CLs in sector "
                f"{s} split {sorted(rings)} Δr={abs(r0-r1):.1f}mm; "
                f"using dart axis {axis_bed}"
            ),
        )
    mean = _mean_xy([xy for _, xy, _ in same])
    mean_bed = sector_ring_for_point(mean[0], mean[1])
    cur = (s, getattr(scored, "ring", None))
    if mean_bed == cur:
        return None
    return _score_result_from_xy(
        mean,
        [cam for cam, _, _ in same],
        (
            "split centerline mean: 2 onboard CLs in sector "
            f"{s} split {sorted(rings)}; using their mean "
            f"{mean_bed}"
        ),
    )


def _lock_insector_ring_majority_2of3(cl_pixels, calibration, scored):
    """3 onboard CLs, same sector, 2-1 on ring, leftover far -> the 2's mean.

    Not the 2-of-3 *sector* trap (two cameras in pie A vs one in pie B).
    All three already agree on the sector; one leftover/bad tip sits in
    a different *ring* of that pie and pulls triangulation across the
    wire. Ungated 2-1 was +2/-3 on the 374 (014-T18, 016-S18 gained;
    013-S1, 030-S5, g1-055-D2 stolen -- those three are wire-adjacent
    Δr 3.9-7.3mm). Require the singleton farther than one treble/double
    band from the majority mean: leftover blob, not a wire straddle.
    Ring lock cannot fire here -- it requires the result farther *out*
    than the majority, and these leftovers pull the result *in*.
    """
    if scored is None or not getattr(scored, "ok", False):
        return None
    s = getattr(scored, "sector", None)
    if s is None:
        return None
    same = [
        (cam, xy, bed)
        for cam, xy, bed in _onboard_beds(cl_pixels, calibration)
        if bed[0] == s
    ]
    if len(same) != 3:
        return None
    (maj, n_maj), = Counter(bed[1] for _, _, bed in same).most_common(1)
    if n_maj != 2 or maj == getattr(scored, "ring", None):
        return None
    holders = [xy for _, xy, bed in same if bed[1] == maj]
    leftover = [xy for _, xy, bed in same if bed[1] != maj]
    mean = _mean_xy(holders)
    leftover_dr = abs(math.hypot(*leftover[0]) - math.hypot(*mean))
    if leftover_dr <= INSECTOR_RING_MAJORITY_MIN_LEFTOVER_DR_MM:
        return None
    return _score_result_from_xy(
        mean,
        [cam for cam, _, bed in same if bed[1] == maj],
        (
            "2-of-3 in-sector ring majority: 3 onboard CLs in sector "
            f"{s}, 2 in {maj}; leftover Δr={leftover_dr:.1f}mm; "
            f"result {getattr(scored, 'ring', None)}; "
            "using those 2 CLs' mean"
        ),
    )


def apply_centerline_overrides(
    cl_pixels, calibration, scored, cap_pixels=None, axis_xy=None, planes=None,
    far_cams=None,
):
    """Run C, D, cap-walked, shaft-snap, ring lock, lonely-axis, 1-CL double, 2of3 leftover ring, split mean."""
    current = scored
    fired = None
    rescued = _rescue_outside_from_centerlines(cl_pixels, calibration, current)
    if rescued is not None:
        current = rescued
        fired = rescued
    lone_rescued = _rescue_outside_lonely_cl_axis(
        cl_pixels, calibration, current, axis_xy=axis_xy, far_cams=far_cams,
    )
    if lone_rescued is not None:
        current = lone_rescued
        fired = lone_rescued
    locked = _lock_unanimous_centerline_sector(cl_pixels, calibration, current)
    if locked is not None:
        current = locked
        fired = locked
    walked = _lock_cap_walked_radial(
        cl_pixels, cap_pixels, calibration, current, axis_xy=axis_xy,
    )
    if walked is not None:
        current = walked
        fired = walked
    # After cap-walked-radial on purpose: for 2-onboard-CL throws where
    # both fire, both produce the same onboard-CL mean -- the more
    # specific lock (caps unanimous in the other pie) keeps the credit,
    # and this one only adds the cases cap-walked's cap-unanimity gate
    # rejects (one cap still in S, e.g. 137-S8's cam2).
    pair_locked = _lock_pair_unanimous_cl_axis(
        cl_pixels, calibration, current, axis_xy=axis_xy,
    )
    if pair_locked is not None:
        current = pair_locked
        fired = pair_locked
    snapped = _lock_shaft_snap_consensus(
        cl_pixels, cap_pixels, planes, calibration, current, axis_xy=axis_xy,
    )
    if snapped is not None:
        current = snapped
        fired = snapped
    ringed = _lock_centerline_ring(
        cl_pixels, calibration, current, axis_xy=axis_xy,
    )
    if ringed is not None:
        current = ringed
        fired = ringed
    lonely = _lock_axis_ring_lonely_cl(
        cl_pixels, calibration, current, axis_xy=axis_xy,
    )
    if lonely is not None:
        current = lonely
        fired = lonely
    doubled = _lock_single_cl_double_to_outer(cl_pixels, calibration, current)
    if doubled is not None:
        current = doubled
        fired = doubled
    maj2 = _lock_insector_ring_majority_2of3(cl_pixels, calibration, current)
    if maj2 is not None:
        current = maj2
        fired = maj2
    split = _lock_split_centerline_mean(
        cl_pixels, calibration, current, axis_xy=axis_xy,
    )
    if split is not None:
        fired = split
    return fired


def drop_far_cap_rays(ray_pixels, calibration, max_r_mm: float = FAR_CAP_RADIUS_MM):
    """Split cap rays into kept (r <= max_r_mm) and far-dropped.

    A None board hit is treated as far (unusable). Caller decides what
    to do when kept has 0 or 1 cameras -- see engine.score().
    """
    kept: dict[int, tuple[float, float]] = {}
    far: dict[int, float] = {}
    for cam, pixel in ray_pixels.items():
        if cam not in calibration:
            far[int(cam)] = float("inf")
            continue
        xy = _pixel_board_xy(pixel, calibration[cam])
        if xy is None:
            far[int(cam)] = float("inf")
            continue
        r = math.hypot(xy[0], xy[1])
        if r > max_r_mm:
            far[int(cam)] = r
            continue
        kept[int(cam)] = pixel
    return kept, far


def one_cam_cl_pullback(
    cam, cl_pixels, calibration, scored, axis_xy=None,
) -> ScoreResult | None:
    """1-cam Z=0 path: cap crossed a bed the same cam's CL + axis agree on.

    With a single surviving camera there is no triangulation to correct
    the outward cap pick, which is a silhouette extreme and biased
    outward by construction -- on the recorded S8 throw the cap walked
    17.6mm of board radius past its own centerline (8/single_outer ->
    11/double). When the surviving camera's own centerline is onboard,
    disagrees with the cap's bed, AND the shaft-plane dart axis lands in
    the centerline's exact bed, the centerline is the corroborated
    observation. Measured 2026-08-17 on living clean/ 996: +1/-0
    (106-S8); the other 30 single-cam throws (including far-cap-250
    gains 030-D10, 050-S5, 163-T11) unchanged.
    """
    if scored is None or not getattr(scored, "ok", False) or axis_xy is None:
        return None
    px = cl_pixels.get(cam)
    if px is None or cam not in calibration:
        return None
    xy = _pixel_board_xy(px, calibration[cam])
    if xy is None or math.hypot(xy[0], xy[1]) > DOUBLE_OUTER_RADIUS_MM:
        return None
    cl_bed = sector_ring_for_point(xy[0], xy[1])
    cur = (getattr(scored, "sector", None), getattr(scored, "ring", None))
    if cur == cl_bed:
        return None
    axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
    if axis_bed != cl_bed:
        return None
    return _score_result_from_xy(
        (float(xy[0]), float(xy[1])),
        [cam],
        (
            "single-cam centerline pullback: cap-ray Z=0 "
            f"{cur[0]}/{cur[1]} but this camera's own centerline and "
            f"the dart axis agree on {cl_bed[0]}/{cl_bed[1]}; using the CL"
        ),
    )


def one_cam_z0(cam, pixel, calibration) -> ScoreResult | None:
    """Single remaining camera after far-ray drop: that camera's ray∩Z=0."""
    if cam not in calibration:
        return None
    xy = _pixel_board_xy(pixel, calibration[cam])
    if xy is None:
        return None
    return _score_result_from_xy(
        (float(xy[0]), float(xy[1])),
        [cam],
        (
            "1-cam Z=0: the other cap rays landed more than "
            f"{FAR_CAP_RADIUS_MM:.0f}mm from bullseye"
        ),
    )
