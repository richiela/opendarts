"""Calibrated board region-of-interest (ROI) mask -- the CALLER-side
companion to `opendarts.engines.apollo.tip_detection` -- "a calibrated
board region-of-interest mask applied by the CALLER before invoking that
module", named long before it was built (2026-08-12).

**Why this exists, precisely**: `tip_detection.py`'s own docstring
documents a measured, real failure mode -- when this was written,
roughly 1 in 4 real images got a
confidently-wrong tip pixel (p90 118px, worst case 769px) because the
"largest sufficiently-elongated diff blob" heuristic occasionally locks
onto a bright reflection/light-strip artifact instead of the dart, worst
on cam1. (`tip_detection.py` has since been independently improved
concurrently with this module's own work -- current numbers, per that
module's own docstring: p90 77px, worst case 708px, roughly 1 in 6-7 --
better, but the same class of failure, still real and still what this
module targets; see `evaluate_board_roi.py` for the current measured
impact against the CURRENT `tip_detection.py`.) `tip_detection.py` is
deliberately decoupled from
calibration/board-geometry -- it has no way
to know where the board even IS in the image, so it cannot itself reject
a dart-shaped blob just because it's implausibly far from the board.
This module is that missing piece, built OUTSIDE tip_detection.py so
that module's zero-coupling boundary stays intact.

**The actual CV operation**: project the board's known 3D geometry
(`opendarts/geometry/board.py`, board-centered world frame, Z=0 plane)
through a `opendarts.pipeline.CameraCalibration`'s camera matrix + pose
(`cv2.projectPoints`, the exact forward-projection inverse of
`opendarts.triangulation.rays.back_project_ray`'s `cv2.undistortPoints`
back-projection) into that camera's pixel space. A circle of known
board-mm radius, sampled at N points around its circumference and
projected, becomes a polygon in pixel space -- the "where can a real
dart plausibly be" region for that one camera's view.

**Integration approach chosen, and why** (there are two possible
integration points: "a
calibrated board region-of-interest mask applied by the CALLER before
invoking this module" (pre-masking), or the fallback framing "reject
`detect_tip()`'s final answer if it falls outside the ROI, and treat
that as a real detection failure with a clear reason"):

`detect_tip()`'s public API (`opendarts/engines/apollo/tip_detection.py`) is a
single black-box call -- `detect_tip(bg_bgr, frame_bgr) ->
TipDetectionResult` -- with no parameter or hook to inspect/filter the
internal candidate list before its own final elongation-ranked selection
(see that module's `detect_tip()` body: candidates are ranked and the
first passing one is chosen and returned in one pass, nothing
intermediate is exposed). Both integration points named above are
implemented (`detect_tip_in_roi()` for pre-masking, `reject_outside_roi()`
for final-answer rejection) -- **but real measurement (full real
354-pair corpus, real per-camera calibrations) found pre-masking has a
genuine, measured
downside that final-answer rejection does not, which is why
`reject_outside_roi()` is the RECOMMENDED, primary entry point, not
`detect_tip_in_roi()`** -- this was not assumed, it was found by running
both and comparing:

1. **`reject_outside_roi()` -- RECOMMENDED.** Call `detect_tip()`
   completely unmodified on the original, un-masked images; then check
   whether the returned `tip_px` falls inside the projected ROI polygon;
   if not, replace the result with `ok=False` and a clear reason. Pure
   zero-touch black-box use of the existing function -- no pixel of the
   input images is ever altered. **Measured on the real corpus: 0/300
   (0.00%) false rejects on already-good baseline detections (error
   <20px vs the real per-camera label), and 24/27 (88.89%) of the known
   fat-tail failures (error >=100px) correctly rejected, at every margin
   from 0 to 30mm beyond the physical board face** -- see
   `evaluate_board_roi.py`'s printed sweep for the full numbers. (These
   are the current numbers against `tip_detection.py` as it stands
   2026-08-12, after its own concurrent companion-blob/texture-tiebreak
   work landed -- `evaluate_board_roi.py` is the reproducible source of
   truth if that module changes again; the exact good/bad case counts
   will shift with it, the 0% false-reject / ~90% true-reject SHAPE of
   the result has stayed stable across that module's own real revision
   during this session.)
2. **`detect_tip_in_roi()` -- kept, NOT the default recommendation.**
   Zero out every pixel outside the projected ROI polygon in BOTH the
   background and the dart-frame image before calling `detect_tip()` on
   the masked pair. The appeal (before measuring) was that this could
   RECOVER a correct answer, not just detect a wrong one, by preventing
   an off-board reflection blob from ever becoming a candidate component
   in the first place -- and it does, in some real cases (see
   `evaluate_board_roi.py`'s "recovered" column). **But it also measurably
   BREAKS some already-good detections: 15-43 out of 300 (5-14% depending
   on margin) across the sweep, non-monotonically.** Root cause, found by
   direct inspection of a real failing case
   (one recorded throw, cam2): `tip_detection.py`'s own
   `DILATE_KERNEL_PX=31` dilation sometimes bridges the real dart's diff
   blob with an unrelated diff region into ONE connected component that
   already correctly resolves to the true tip end in the original image
   (measured concretely: one real component, area 16254px, bbox
   `y=[0,266]`, correctly picks tip near `y=250`, baseline error 1.4px).
   Masking then truncates that SAME merged component at the ROI boundary
   (measured: the same component shrinks to area 6261px, bbox
   `y=[132,266]`), which changes the component's own width-at-each-end
   comparison enough to flip which end reads as "the tip" -- producing a
   brand-new, large error (here: `(479,250)` correct baseline -> `(522,
   147)` wrong after masking, a mechanism entirely internal to
   `detect_tip()`'s own bridging/width-comparison logic, not a
   coincidence). This is a real, measured, honest finding, not a guess --
   masking is not a strictly-safer operation just because it "only
   removes information."

Neither function reads or modifies a single line of `tip_detection.py`
itself -- both operate purely on its existing public `detect_tip()` /
`TipDetectionResult` surface, respecting the "zero
calibration/board-geometry coupling inside tip_detection.py itself"
boundary. `detect_tip_in_roi()` also applies
`reject_outside_roi()` internally as a defense-in-depth safety net (see
that function's docstring) -- which is exactly what catches its OWN
masking-induced failures in the example above (the masked, wrong
`(522,147)` result gets `ok=False` from the safety net rather than being
returned silently wrong) -- but "caught its own self-inflicted failure"
is not the same as "didn't inflict it"; `reject_outside_roi()` applied
directly to the ORIGINAL baseline result has no such self-inflicted
failures to catch in the first place.

**Measured real impact, full picture**: the evaluation was run against
the full real 354-pair corpus with real per-camera calibrations solved from
the reference case data's `calibration.json` 4-point quads
(see `dev/calibration/real_correspondences.py`).
Headline, stated honestly: `reject_outside_roi()` closes ~89% of the
measured fat-tail gap (24/27 confirmed bad, error>=100px, cases) with
zero measured false rejects, but not all of it -- 3 of the 27 known bad
detections land geometrically WITHIN the board's own projected image
footprint (e.g. a bright reflection or artifact on/near the board face
itself, not off to the side), which no board-shaped ROI can distinguish
from a real dart by position alone. This module does not claim
otherwise. (`tip_detection.py` itself was independently improved
concurrently with this module's own work -- see that module's Progress
log entry -- shifting the exact bad-case count from an earlier measured
31 down to 27; re-run `evaluate_board_roi.py` for the current numbers if
`tip_detection.py` changes again, don't assume these stay pinned
forever.)

**Judgment calls made here, documented rather than silently guessed**
(no real measured distribution for either exists in this project's real
data -- see docs/DESIGN.md's "measure before guessing" discipline; these
values ARE swept/measured for real IMPACT in evaluate_board_roi.py, just
not derived from a first-principles physical measurement):

- `BOARD_FACE_RADIUS_MM = 225.5`: the standard regulation bristle
  dartboard's overall face diameter is 451mm / 17.75in (WDF/PDC
  standard), i.e. 225.5mm radius -- this is the full physical board a
  dart can visibly strike/embed in, NOT `opendarts/geometry/board.py`'s
  `DOUBLE_OUTER_RADIUS_MM=170mm`, which is only the SCORING boundary (a
  dart embedded in the numbers ring between 170mm and 225.5mm is a real,
  visible, in-frame dart that should still be detected even though it
  wouldn't score -- rejecting it via too-tight an ROI would be a real,
  avoidable false reject). `opendarts/geometry/board.py` deliberately only
  models the scoring geometry (see that module's own docstring), so this
  full-face radius is defined HERE, not added there, to avoid scope-
  creeping a module whose whole job is scoring math.
- `ROI_MARGIN_MM = 30.0`: extra slop beyond the physical board face, for
  (a) a dart caught mid-flight/motion-blurred with its diff silhouette
  extending slightly past the true board edge in the captured frame, and
  (b) this ROI's own boundary being only as accurate as the calibration
  it's projected from (real intrinsics for this rig are still
  uncertain) -- a small buffer avoids the ROI itself clipping a
  genuinely-on-board dart due to calibration slop rather than a true
  miss. 30mm was measured, not just guessed, against the real corpus in
  `evaluate_board_roi.py`'s margin sweep (`reject_outside_roi()`, the
  recommended function): false-reject rate stays exactly 0/300 (0.00%)
  and true-reject rate stays exactly 24/27 (88.89%) at every margin from
  0mm through 30mm -- so 30mm costs nothing measured on this corpus while
  providing real headroom against calibration/mid-flight uncertainty
  this corpus's own real (and unusually accurate, <3.3px reprojection
  error) calibration doesn't exercise. Margins beyond that DO start
  measurably losing catch rate (60mm: 66.67%; 100mm: 22.22%), which is
  why 30mm -- not a larger "safer-sounding" number -- is the shipped
  default. See that script's printed sweep for the full numbers.

**Real bug found and fixed 2026-08-12, before wiring this module into
any live call site:** `reject_outside_roi()`
and `detect_tip_in_roi()` both reconstruct a fresh `TipDetectionResult`
at every branch (to attach ROI diagnostics / flip `ok`), and every one of
those reconstructions omitted `alt_tip_px` -- silently dropping it back
to the dataclass default (`None`) even on the common "accepted, inside
ROI" path. This would have quietly broken the cross-camera alt-candidate
mechanism (`tip_detection.py`'s `alt_tip_px`, `opendarts.pipeline.
score_dart`'s `alt_tip_pixels`, see that module's own 2026-08-12 dated
entry -- the real `throw_1786580447119` fix)
for every throw that went through this module, the moment it was wired
into a live call site -- caught here, before that wiring happened, by
reading the reconstruction code directly rather than assuming it was a
transparent passthrough. Fixed by threading `alt_tip_px=result.alt_tip_px`
through every reconstruction; `tests/test_board_roi.py` pins this
directly (a result with a real `alt_tip_px` set survives both
`reject_outside_roi()`'s accept and reject paths, and `detect_tip_in_
roi()`'s masked re-detection, unchanged).

**2026-08-13 -- `alt_tip_px` is gated too, and a plausible alternate is
PROMOTED rather than thrown away with its camera.** Until this date
`reject_outside_roi()` tested only `tip_px`; `alt_tip_px` rode along
completely untested. That was wrong in both directions on real data, and
the cost was measured, not theorised:

* **primary outside / alternate inside** (real count: 19 of 350
  camera-detections on one recorded session, 19 of 507 on
  the seven 2026-08-12 sessions): the whole camera was discarded even
  though this module's own geometry said the OTHER end of the same blob
  was a perfectly plausible on-board dart. With only 1 usable camera left,
  `score_dart()` cannot triangulate at all and the throw scores nothing.
  Three of the four `ok=False` throws in the 2026-08-13 session were
  exactly this. Worked example, `throw_1786666408481` (real, in that
  session): cam0's primary sat 103px from where AD's ground-truth tip
  forward-projects and back-projected to r=319mm on the board plane
  (off-board, correctly rejected) while its alternate was 13px from that
  same projection at r=135mm; cam1 identically, 110px/r=286mm vs
  17px/r=134mm. Both cameras were being binned for having guessed the
  wrong END, with the right end already in hand.
* **primary inside / alternate outside** (real count: 70 of 350 and 95 of
  507 respectively): a candidate this module has geometrically ruled out
  stayed in the pool for `score_dart()`'s primary/alt combination search
  to pick.

The promotion is not a preference or a tie-break. `alt_tip_px` is
populated by `tip_detection` ONLY when that camera's own monocular
tip-vs-fletching end call was explicitly unresolved on that one image
(`_locate_tip_in_component`'s `alt_unresolved`) -- so when exactly one of
the two ends back-projects onto the board, the ROI is real, independent
geometric evidence about which end was right, which is precisely the
question the detector said it could not answer. The rejected end is NOT
carried forward as the new alternate: the same gate has already ruled it
out.

Real measured effect (per-camera, `evaluate_board_roi.py`'s own sweep,
same real 354-pair corpus and same shipped 30mm margin as the numbers
above). **Re-measured 2026-08-13 after `tip_detection.DIFF_THRESHOLD` was
re-tuned 30.0 -> 25.0** -- that change moved the baseline this gate is
measured against (302 good / 23 fat-tail, was 300 good / 27), so these
numbers supersede the first ones taken at the old threshold; both
promotion-on and promotion-off were re-run at the new threshold so the
comparison is like-for-like:

  promotion OFF: 0/302 false rejects | of 23 fat-tail (>=100px) failures:
                 0 recovered, 18 rejected, 5 still confidently wrong
  promotion ON:  0/302 false rejects | of the same 23:
                 9 RECOVERED to <20px, 8 rejected, 6 still confidently wrong

False rejects on good detections stay exactly **0/302 (0.00%)** --
promotion cannot fire on a detection whose primary is already inside the
ROI, so it structurally cannot damage a good one. The honest cost, stated
plainly: ONE case the gate previously rejected correctly now comes back
accepted-and-still-wrong, bought for 9 that come back correct.
Downstream ray-disagreement (`scoring.MAX_RAY_DISAGREEMENT_MM`, the
2-of-3 RANSAC fallback) is the remaining defence for that one. At the
throw level the trade measured clean in every direction (numbers below
are from the original measurement at DIFF_THRESHOLD=30.0, i.e. the
isolated effect of THIS change with nothing else moving):

  one recorded session (120 real throws, package
  calibration -- i.e. what the live run actually used): sector+ring BOTH
  105/120=87.5% -> **107/120=89.2%**, sector 93.3% -> 95.8%, throws
  scoring nothing 4 -> 1.
  The seven 2026-08-12 sessions (169 real throws), package calibration:
  138/169=81.7% -> **139/169=82.2%**; with calibration re-derived through
  the production `oriented_landmarks` path: 163/169=96.4% -> 96.4%
  (unchanged -- zero regression on the best-known historical number).

The alternate-pruning half of this change (primary inside, alternate
outside -> drop the alternate) measured **exactly zero** effect on all
four of those figures, ablated separately: an off-board alternate loses
`score_dart()`'s own lowest-disagreement contest anyway on this corpus.
It is kept because it is correct -- a candidate ruled out geometrically
should not be offered to a downstream search at all -- and reported here
as a measured-zero, not sold as part of the win.

**Wired into both real call sites 2026-08-12** (this was previously
built but not called from either):
`opendarts.live.capture_daemon.handle_ready_to_capture()` (the live capture
path) and `opendarts.capture.replay.replay_throw()` (the offline replay
path, per docs/DESIGN.md's "Replay is the source of truth" -- replaying an old package
must go through the SAME current code, including this gate, not a subset
of it) both now call `reject_outside_roi()` on every camera's raw
`detect_tip()` result before it's used, via the RECOMMENDED entry point
named above -- not `detect_tip_in_roi()`'s pre-masking, per this module's
own measured recommendation. See each call site's own code comment for
the real corpus before/after numbers this produced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opendarts.engines.apollo.tip_detection import TipDetectionResult

# --- Board face / ROI geometry constants ---------------------------------
# See module docstring "Judgment calls made here" for the reasoning behind
# both of these.
BOARD_FACE_RADIUS_MM = 225.5
ROI_MARGIN_MM = 30.0
ROI_RADIUS_MM = BOARD_FACE_RADIUS_MM + ROI_MARGIN_MM  # 255.5mm

# Number of points sampled around the ROI circle's circumference before
# projecting into pixel space. The projected polygon approximates the
# true projected circle/ellipse (a circle in a Z=0 plane projects to a
# conic under a pinhole camera, generally an ellipse-like curve for this
# rig's real oblique viewing angles) -- more samples = closer
# approximation, at negligible extra cost (a handful of matrix ops).
# 144 (every 2.5deg) was chosen so the polygon-vs-true-conic deviation is
# sub-pixel for this project's real camera geometry (checked in
# tests/test_board_roi.py against cv2's own analytic ellipse fit).
N_BOUNDARY_SAMPLES = 144


@dataclass
class BoardRoiResult:
    """Result of projecting the board ROI circle into one camera's pixel
    space. `polygon_px` is `None` whenever `ok=False` -- e.g. the
    calibration placed part of the boundary circle behind the camera
    (a degenerate/wrong pose), or projection produced a non-finite pixel
    (same class of degenerate-geometry guard as
    `opendarts.triangulation.rays.triangulate`'s `well_conditioned` check).
    """

    ok: bool
    polygon_px: np.ndarray | None  # (N, 2) float64, ordered around the circle
    radius_mm: float
    reason: str = ""


def board_roi_world_points_mm(
    radius_mm: float = ROI_RADIUS_MM, n_samples: int = N_BOUNDARY_SAMPLES
) -> np.ndarray:
    """The ROI boundary circle's 3D points (board-centered world frame,
    Z=0 -- see `opendarts/geometry/board.py`'s module docstring for the
    frame convention), evenly spaced around the circumference in the SAME
    clockwise-from-+Y angle convention as `board.polar_to_xy_mm` (not
    reused directly to avoid a hard dependency for what is otherwise a
    two-line formula, but must stay consistent with it -- pinned by
    `tests/test_board_roi.py` against `board.polar_to_xy_mm` directly).
    """
    angles_deg = np.linspace(0.0, 360.0, n_samples, endpoint=False)
    rad = np.radians(angles_deg)
    x = radius_mm * np.sin(rad)
    y = radius_mm * np.cos(rad)
    z = np.zeros_like(x)
    return np.column_stack([x, y, z]).astype(np.float64)


def board_roi_polygon_px(
    calibration,
    *,
    radius_mm: float = ROI_RADIUS_MM,
    n_samples: int = N_BOUNDARY_SAMPLES,
) -> BoardRoiResult:
    """Project the board ROI boundary circle through `calibration`
    (a `opendarts.pipeline.CameraCalibration`, or any object exposing the
    same `camera_matrix`/`dist_coeffs`/`rvec`/`tvec` fields -- e.g.
    `tests.support.synthetic.SyntheticCamera` in tests) into that
    camera's pixel space via `cv2.projectPoints` -- the standard forward-
    projection operation, and the exact inverse of
    `opendarts.triangulation.rays.back_project_ray`'s `cv2.undistortPoints`
    back-projection (same rvec/tvec world-to-camera convention: X_c = R
    @ X_w + t, see that module's docstring).

    Returns points in the SAME cyclic order they were sampled in (see
    `board_roi_world_points_mm`) -- a circle's perspective projection is
    a single conic section (as long as no boundary point crosses behind
    the camera), so this ordering is guaranteed to trace a simple
    (non-self-intersecting) polygon, which `point_in_board_roi` /
    `board_roi_mask` below rely on.
    """
    import cv2

    object_points = board_roi_world_points_mm(radius_mm=radius_mm, n_samples=n_samples)

    rvec = np.asarray(calibration.rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(calibration.tvec, dtype=np.float64).reshape(3, 1)

    # Defensive depth check (same class of guard as
    # opendarts.triangulation.rays.TriangulationResult.all_positive_depth):
    # cv2.projectPoints does not itself distinguish a point in front of
    # the camera from its behind-the-camera mirror image -- a boundary
    # point behind the camera would silently produce a nonsensical
    # projected pixel rather than an error. Not expected for any sane
    # real calibration (the camera looks AT the board), but this is
    # exactly the kind of silent-garbage risk a review pass found in
    # the back-projection direction, so it is checked explicitly here
    # too rather than assumed away.
    R, _ = cv2.Rodrigues(rvec)
    cam_frame_points = (R @ object_points.T + tvec).T
    if not np.all(cam_frame_points[:, 2] > 0):
        return BoardRoiResult(
            ok=False,
            polygon_px=None,
            radius_mm=radius_mm,
            reason=(
                "at least one board-ROI boundary point projects BEHIND the "
                "camera -- degenerate/implausible calibration pose, cannot "
                "build a trustworthy ROI polygon from it"
            ),
        )

    dist_coeffs = np.asarray(calibration.dist_coeffs, dtype=np.float64)
    camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    image_points, _ = cv2.projectPoints(
        object_points, rvec, tvec, camera_matrix, dist_coeffs
    )
    pts = image_points.reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return BoardRoiResult(
            ok=False,
            polygon_px=None,
            radius_mm=radius_mm,
            reason="projection produced non-finite pixel(s) -- degenerate calibration",
        )
    return BoardRoiResult(ok=True, polygon_px=pts, radius_mm=radius_mm)


def point_in_board_roi(pixel_xy: tuple[float, float], polygon_px: np.ndarray) -> bool:
    """Point-in-polygon test (`cv2.pointPolygonTest`, boundary counts as
    inside) against an already-projected ROI polygon from
    `board_roi_polygon_px`."""
    import cv2

    poly = np.asarray(polygon_px, dtype=np.float32).reshape(-1, 1, 2)
    point = (float(pixel_xy[0]), float(pixel_xy[1]))
    return cv2.pointPolygonTest(poly, point, False) >= 0


def board_roi_mask(
    polygon_px: np.ndarray, image_width: int, image_height: int
) -> np.ndarray:
    """Rasterize an ROI polygon (from `board_roi_polygon_px`) into a
    boolean `(image_height, image_width)` mask, `True` inside the ROI.
    `cv2.fillPoly` needs integer vertex coordinates; rounding to the
    nearest pixel is sub-pixel-accurate enough for a mask whose whole
    purpose is excluding gross off-board artifacts, not precise geometry
    (unlike `point_in_board_roi`, which tests the float polygon
    directly)."""
    import cv2

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    poly_int = np.round(np.asarray(polygon_px, dtype=np.float64)).astype(np.int32)
    cv2.fillPoly(mask, [poly_int.reshape(-1, 1, 2)], 255)
    return mask.astype(bool)


def reject_outside_roi(
    result: TipDetectionResult,
    calibration,
    *,
    radius_mm: float = ROI_RADIUS_MM,
    n_samples: int = N_BOUNDARY_SAMPLES,
) -> TipDetectionResult:
    """RECOMMENDED entry point (see module docstring for the real
    measurement that made this the default recommendation over
    `detect_tip_in_roi()`): take an EXISTING `TipDetectionResult` from an
    unmodified `detect_tip()` call and reject it if its `tip_px` falls
    outside the calibrated board ROI -- a real, explicit detection
    failure with a clear reason, not a silent pass-through. Does not
    touch `tip_detection.py`; operates purely on its public
    `TipDetectionResult` dataclass, and never alters a single input pixel
    (unlike `detect_tip_in_roi()`, which measurably can, in a way that
    occasionally hurts).

    Measured on the real 354-pair corpus (`evaluate_board_roi.py`): 0/300
    (0.00%) false rejects on already-good baseline detections, 24/27
    (88.89%) of the known fat-tail (error>=100px) failures correctly
    rejected, at every margin from 0-30mm beyond the physical board face
    (see `ROI_MARGIN_MM`'s comment for the full sweep).

    If the ROI itself cannot be computed (`BoardRoiResult.ok=False` --
    e.g. a degenerate calibration), this FAILS OPEN: the original result
    is returned unchanged (with a diagnostic noting the ROI check was
    skipped), rather than silently rejecting a possibly-good detection
    because of an unrelated calibration problem it had nothing to do
    with. A caller that wants calibration failures to also fail the
    detection should check `CameraCalibration.landmark_spread_ok` /
    `PnpResult.ok` itself -- that is a different, already-existing signal
    (the correlated-bias signal), not this module's job to
    re-implement.
    """
    if not result.ok or result.tip_px is None:
        return result

    roi = board_roi_polygon_px(calibration, radius_mm=radius_mm, n_samples=n_samples)
    if not roi.ok:
        return TipDetectionResult(
            ok=result.ok,
            tip_px=result.tip_px,
            reason=result.reason,
            alt_tip_px=result.alt_tip_px,
            far_end_px=result.far_end_px,
            diagnostics={
                **result.diagnostics,
                "board_roi_checked": False,
                "board_roi_skip_reason": roi.reason,
            },
        )

    inside = point_in_board_roi(result.tip_px, roi.polygon_px)

    # --- alt_tip_px is gated too, 2026-08-13 (see the dated section at the
    # end of this module's docstring for the full real-evidence write-up).
    # Before this, the gate looked ONLY at `tip_px` and passed
    # `alt_tip_px` straight through untested, which was wrong in both
    # directions on real data:
    #   * primary outside / alt inside -> the whole camera was discarded
    #     even though this module's own geometry said the alternate
    #     candidate was a perfectly plausible on-board dart;
    #   * primary inside / alt outside -> a geometrically impossible
    #     candidate stayed in the pool for score_dart()'s combination
    #     search to pick.
    alt_inside = (
        point_in_board_roi(result.alt_tip_px, roi.polygon_px)
        if result.alt_tip_px is not None
        else None
    )

    if inside:
        # Prune an implausible alternate rather than handing it to
        # score_dart(): the alternate is only ever a coin-flip second
        # guess at which END of the same blob is the tip (see
        # tip_detection._locate_tip_in_component's `alt_unresolved`), and
        # an end that back-projects clean off the board is not a
        # candidate this rig's geometry supports.
        return TipDetectionResult(
            ok=result.ok,
            tip_px=result.tip_px,
            reason=result.reason,
            alt_tip_px=result.alt_tip_px if alt_inside is not False else None,
            far_end_px=result.far_end_px,
            diagnostics={
                **result.diagnostics,
                "board_roi_checked": True,
                "board_roi_rejected": False,
                "board_roi_alt_inside": alt_inside,
                "board_roi_alt_pruned": alt_inside is False,
            },
        )

    if alt_inside:
        # The primary is off-board and the alternate is on-board. The two
        # are the two ends of ONE already-selected dart-shaped component,
        # and this camera's own monocular end-choice was explicitly
        # flagged unresolved (that is the only condition under which
        # `alt_tip_px` is populated at all) -- so the ROI is real,
        # independent geometric evidence for which end was right, not a
        # tie-break preference. Promote it, and do NOT keep the rejected
        # end as the new alternate: it has already been ruled out by the
        # same gate.
        return TipDetectionResult(
            ok=True,
            tip_px=result.alt_tip_px,
            reason=result.reason,
            alt_tip_px=None,
            far_end_px=result.far_end_px,
            diagnostics={
                **result.diagnostics,
                "board_roi_checked": True,
                "board_roi_rejected": False,
                "board_roi_alt_promoted": True,
                "board_roi_rejected_primary_px": result.tip_px,
                "pre_roi_reason": result.reason,
            },
        )

    # 2026-08-14: report whether the OPPOSITE end of the same component
    # is on-board. Deliberately reported, not acted on here: promoting it
    # is a last-resort recovery whose safety depends on how many OTHER
    # cameras survived this gate, which is knowledge only the engine has
    # (see ApolloEngine.score()). This function stays a per-camera,
    # context-free plausibility check. Distinct from the `alt_inside`
    # promotion above: that one fires only when tip_detection itself
    # flagged the end-choice unresolved; this covers the confidently-wrong
    # case, where no alternate was ever offered.
    far_end_inside = (
        point_in_board_roi(result.far_end_px, roi.polygon_px)
        if result.far_end_px is not None
        else None
    )
    return TipDetectionResult(
        ok=False,
        tip_px=result.tip_px,
        reason=(
            f"tip pixel {result.tip_px} rejected: outside the calibrated "
            f"board ROI (radius {radius_mm:.1f}mm around board center) -- "
            "not a plausible dart location for this camera's calibration"
        ),
        alt_tip_px=result.alt_tip_px,
        far_end_px=result.far_end_px,
        diagnostics={
            **result.diagnostics,
            "board_roi_checked": True,
            "board_roi_rejected": True,
            "board_roi_alt_inside": alt_inside,
            "board_roi_far_end_inside": far_end_inside,
            "pre_roi_reason": result.reason,
        },
    )


def detect_tip_in_roi(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    calibration,
    *,
    radius_mm: float = ROI_RADIUS_MM,
    n_samples: int = N_BOUNDARY_SAMPLES,
) -> TipDetectionResult:
    """Alternative entry point -- NOT the default recommendation (see
    module docstring "Integration approach chosen" for the real
    measurement that found this has a genuine downside `reject_outside_
    roi()` does not, kept here anyway as a real, tested, documented
    alternative rather than deleted): mask both images to the calibrated
    board ROI BEFORE calling `detect_tip()` -- literally "a calibrated
    board region-of-interest mask applied by the CALLER before invoking
    this module".
    `tip_detection.py` is called completely unmodified through its
    existing public `detect_tip(bg_bgr, frame_bgr)` signature; only the
    two input arrays differ from the raw captured images.

    Pixels outside the ROI are set to 0 (a fixed, identical constant) in
    BOTH images, so their contribution to `detect_tip()`'s first-step
    absolute diff is always exactly zero -- they can never form a
    connected component, let alone win the area/elongation ranking. This
    CAN recover a correct tip (not just detect a wrong one) in cases
    where an off-board reflection blob would otherwise have outranked
    the real dart's own smaller diff blob -- and measurably does, on the
    real corpus (see `evaluate_board_roi.py`'s "recovered" column).

    **But it can also measurably BREAK an already-good detection** (real
    corpus measurement: 15-43/300, 5-14%, depending on margin -- see
    module docstring): `tip_detection.py`'s own dilation step
    (`DILATE_KERNEL_PX=31`) sometimes bridges the real dart's diff blob
    with an unrelated diff region into one connected component whose
    width-at-each-end comparison already correctly picks the true tip in
    the ORIGINAL image; truncating that same merged component at the ROI
    boundary changes its width profile enough to flip which end reads as
    "the tip." This is why `reject_outside_roi()`, not this function, is
    the default recommendation -- use this one deliberately, having read
    the tradeoff above, not as the default choice.

    Also applies `reject_outside_roi()` as a defense-in-depth safety net
    on the result (see that function's docstring) -- masking should make
    an outside-ROI result essentially impossible (nothing outside the
    ROI can produce a nonzero diff anymore), but this catches the
    boundary-precision edge case (the rasterized mask and the float
    polygon test are two independently-computed approximations of the
    same ROI, and could disagree by a pixel or two right at the
    boundary) rather than silently trusting that masking alone is
    airtight.

    Per-image-pair overhead: one `board_roi_polygon_px` projection (cheap
    -- one `cv2.projectPoints` call over `n_samples` points) plus one
    `board_roi_mask` rasterization (one `cv2.fillPoly` over the image's
    resolution) -- negligible next to `detect_tip()`'s own morphology/
    connected-components cost.

    If the ROI cannot be computed (degenerate calibration), fails open:
    calls `detect_tip()` on the ORIGINAL unmasked images and returns that
    result unchanged (diagnostics note the skip), same fail-open
    philosophy as `reject_outside_roi()`.
    """
    from opendarts.engines.apollo.tip_detection import detect_tip

    if bg_bgr.shape != frame_bgr.shape:
        return detect_tip(bg_bgr, frame_bgr)  # let detect_tip's own shape-mismatch path report it

    img_h, img_w = bg_bgr.shape[:2]
    roi = board_roi_polygon_px(calibration, radius_mm=radius_mm, n_samples=n_samples)
    if not roi.ok:
        result = detect_tip(bg_bgr, frame_bgr)
        return TipDetectionResult(
            ok=result.ok,
            tip_px=result.tip_px,
            reason=result.reason,
            alt_tip_px=result.alt_tip_px,
            diagnostics={
                **result.diagnostics,
                "board_roi_masking_applied": False,
                "board_roi_skip_reason": roi.reason,
            },
        )

    mask = board_roi_mask(roi.polygon_px, img_w, img_h)
    bg_masked = bg_bgr.copy()
    frame_masked = frame_bgr.copy()
    bg_masked[~mask] = 0
    frame_masked[~mask] = 0

    result = detect_tip(bg_masked, frame_masked)
    result = TipDetectionResult(
        ok=result.ok,
        tip_px=result.tip_px,
        reason=result.reason,
        alt_tip_px=result.alt_tip_px,
        diagnostics={**result.diagnostics, "board_roi_masking_applied": True},
    )
    # Defense-in-depth safety net -- see docstring.
    return reject_outside_roi(result, calibration, radius_mm=radius_mm, n_samples=n_samples)
