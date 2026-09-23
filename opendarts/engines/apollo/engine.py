"""Apollo -- today's real scoring algorithm (`opendarts.engines.apollo.
tip_detection.detect_tip()` per camera + `opendarts.engines.apollo.
board_roi.reject_outside_roi()`'s ROI gate + `opendarts.engines.apollo.
scoring.score_dart()`'s ray triangulation) wrapped behind the Engine
interface. Per docs/ENGINES.md: "Apollo must wrap this UNCHANGED (call
it, don't reimplement it, don't modify its signature/behavior)."

**2026-08-12 -- this is now the ONLY call path, live and offline.**
Before this date, `opendarts/live/capture_daemon.py`'s
`handle_ready_to_capture()` and `opendarts/capture/replay.py`'s
`replay_throw()` each independently called `detect_tip()`/
`reject_outside_roi()`/`score_dart()` directly (byte-identical, but a
real, separate bypass of this class) out of caution, before
`ApolloEngine.score()` was proven identical to that direct sequence
(real corpus, zero mismatches). That caution is no longer needed: both call sites now
route through `get_engine("Apollo").score()` -- this module -- like
any other engine, with no special-cased bypass left anywhere. `score()`
below is a thin, one-directional adapter: unpack the images-in/
calibration-in interface into exactly the calls those two call sites
used to make inline (detect_tip() per camera, ROI-gate each result, then
score_dart() once), and pack `opendarts.pipeline.ScoreResult` back into an
`EngineResult`. `engine_result_to_score_result()` at the bottom of this
module is the full-fidelity inverse (EngineResult -> ScoreResult) that
makes the bypass removal possible: unlike `opendarts.engines.base.
engine_result_to_score_result()` (the GENERIC lossy adapter used for a
non-Apollo primary, which has no way to recover Apollo-specific
fields like `cameras_used`/`outlier_camera`/`alt_candidates_used` from an
arbitrary engine's diagnostics), THIS module knows exactly how
`score_result_to_engine_result()` below packed those fields into
`diagnostics`, so it can unpack them again losslessly for every field
`opendarts.capture.throw_package.save_throw_package()`'s on-disk schema
actually persists.

**2026-08-25 -- `ApolloEngine.score()` NEVER returns `ok=False`.**
the project's own words, verbatim: "Apollo is the only engine that returns
fails to score... I don't want that happening." This is a hard product
requirement, not a per-throw accuracy target -- `score()` now runs
`result` through SIX progressively-weaker-evidence fallback tiers before
ever handing a caller an answer, the last of which is structurally
guaranteed (by code inspection, not just measurement) to always produce
`ok=True`:

1. Primary triangulation + 2-of-3 RANSAC fallback pair (`score_dart()`
   itself, unchanged since this engine's original build).
2. Zero-genuine-camera classification-majority fallback (2026-08-18) --
   trusts ROI-gate-REJECTED candidates when nothing passed the gate at
   all, if >=2 of them independently agree on a (sector, ring).
3. Unanimous-off-board override (2026-08-25) -- >=2 genuine cameras
   whose FUSED triangulation disagrees too much to trust, but whose
   INDIVIDUAL rays unanimously land outside the double ring.
4. Sparse-camera (<2 genuine) unanimous-off-board fallback (2026-08-25)
   -- tier 3's same unanimity logic, extended to pool ROI-rejected
   candidates too when fewer than 2 genuine cameras exist.
5. Marginal on-board disagreement (<=20mm) low-confidence fallback
   (2026-08-25, confidence 0.30) -- uses the FULL triangulation's own
   already-computed point even though it missed the primary gate by a
   small margin.
6. **Last-resort always-answer fallback** (2026-08-25, confidence 0.15 --
   see `_last_resort_always_answer_fallback()`'s own docstring for the
   full mechanism and the real incident, the recorded outside throw, that
   motivated it). Fires ONLY when tiers 1-5 have ALL already declined.
   Uses a one-vote-per-camera majority across EVERY candidate this
   engine detected (genuine or ROI-rejected, primary or alt, even a
   far-end recovery), falling back through the fused triangulation's own
   point, and finally -- for the genuine zero-evidence case where not
   even that exists -- an explicit, honestly-labeled placeholder. This
   tier's own control flow has no path back to `ok=False`: every branch
   inside it unconditionally returns `ok=True`.

Confirmed, not assumed: a full local-corpus replay (721 throws,
the session corpus, 2026-08-25) shows zero `ok=False`
results and tier 6 firing on exactly the one known throw that needed it.
Whether an `ok=False` path is even THEORETICALLY reachable from a real
`capture_daemon.py` call site (as opposed to a synthetic/test call with
empty images) was explicitly audited, not just measured -- see this
task's own final report for the full writeup; short answer: no live call
site was found that can hand `score()` fewer than a genuine multi-camera
frame set, but tier 6 handles the theoretical zero-camera case anyway
(verified directly, `ApolloEngine().score({}, {}, {})` returns
`ok=True`), so the guarantee does not depend on that audit being
airtight.
"""
from __future__ import annotations

import numpy as np

from opendarts.engines.base import EngineResult
from opendarts.engines.apollo.board_roi import reject_outside_roi
from opendarts.engines.apollo.confidence import compute_confidence
from opendarts.engines.apollo.prior_dart_context import PriorDartLinePx
from opendarts.engines.apollo.scoring import MAX_RAY_DISAGREEMENT_MM, score_dart
from opendarts.engines.apollo.tip_detection import (
    PRECOMPUTE_REQUIREMENTS,
    TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX,
    detect_tip,
)
from opendarts.imageops import DiffCrop
from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM, sector_ring_for_point
from opendarts.pipeline import CameraCalibration, ScoreResult
from opendarts.triangulation.rays import TriangulationResult, back_project_ray

# See ApolloEngine.score()'s own dated comment (2026-08-16, real throw
# the recorded D2 throw) for the full real-incident write-up this
# threshold comes from: only reconsider dropping a prior-dart-line-
# suspected camera when the accepted 3-camera result's OWN ray
# disagreement is already elevated enough to be worth second-guessing.
# Set between the real false-positive throw's own comfortable 3.64mm
# (must NOT trigger) and the real true-positive incident's own 7.55mm
# (must trigger) -- not a knife-edge number, but not yet swept across a
# larger sample either; re-measure if a future corpus pull produces a
# real case landing inside this gap.
PRIOR_DART_DROP_MIN_FULL_DISAGREEMENT_MM = 5.0

# See _per_camera_vote_override()'s own docstring for the mechanics.
# Real, measured gap (2026-08-18, data/archive/clean/, 1107 AD-matched
# throws): a first version of that override with NO disagreement floor
# fixed the 11 real misses it targets but ALSO broke 2 previously-
# correct throws -- both real, both near-zero fused disagreement
# (the recorded D8 throw, 8/double, fused disagreement 0.11mm; 123-S2,
# 2/single_inner, 0.39mm) where a naive single-ray Z=0 vote is simply a
# cruder estimate than the genuine multi-ray triangulation it was about
# to override (055-D8's own cam2 walks slightly past the double wire
# when projected as a single ray alone, but agrees with cam0 to 0.11mm
# once properly triangulated in 3D -- the Z-drop/single-ray-noise gap
# this project's own docs/DESIGN.md has flagged since 2026-08-13). Every one of
# the 11 real, wanted fixes has its own fused disagreement >= 1.04mm; both
# regressions are < 0.39mm -- a real, comfortable gap, not a knife-edge
# number picked in advance. Set at the midpoint (not swept further --
# n=13 total data points here, small; re-measure if a future corpus pull
# produces a real case landing inside this gap).
MIN_DISAGREEMENT_FOR_VOTE_OVERRIDE_MM = 0.7


def _single_ray_board_xy(
    pixel_px: tuple[float, float], calib: CameraCalibration
) -> tuple[float, float] | None:
    """One camera's pixel -> its own ray's Z=0 board-plane hit, with NO
    triangulation/agreement involved -- the explicit, caller-owned
    single-ray fallback deliberately kept out
    of `opendarts.triangulation.rays`/`score_dart()` themselves ("the <=1-
    camera fallback is the CALLER's explicit responsibility"). Used only
    by the last-resort fallback in `ApolloEngine.score()` below, never
    by `score_dart()` itself. `None` when the ray is (numerically)
    parallel to the board plane or points away from it -- an honest "no
    hit", not a guess."""
    ray = back_project_ray(
        pixel_px, calib.camera_matrix, calib.dist_coeffs, calib.rvec, calib.tvec,
    )
    dz = float(ray.direction[2])
    if abs(dz) < 1e-9:
        return None
    t = -float(ray.origin[2]) / dz
    if t <= 0:
        return None
    hit = ray.origin + t * ray.direction
    return (float(hit[0]), float(hit[1]))


def _per_camera_vote_override(
    result: ScoreResult,
    tip_pixels: dict[int, tuple[float, float]],
    alt_tip_pixels: dict[int, tuple[float, float]],
    far_end_recoveries: dict[int, tuple[float, float]],
    calibration: dict[int, CameraCalibration],
) -> ScoreResult:
    """2026-08-18 -- per-camera-vote override, see the dated comment at
    this function's own call site in `ApolloEngine.score()` for the
    real-incident write-up (5 remaining `data/archive/clean/` misses
    that independently motivated this, plus 4 more found while
    generalizing it -- see that comment for the full list and the exact
    reasoning that ruled out a blanket majority vote). This function
    itself is deliberately just the mechanical part: given an already-
    ACCEPTED `result`, ask every camera that produced ANY usable pixel
    candidate on this throw (whichever pixel it actually contributed to
    `result` if it was part of the winning triangulation, else its own
    best raw candidate) what bed ITS OWN single ray∩Z=0 lands in, and
    override to the majority bed only when that majority strictly
    outnumbers however many cameras (0, usually) individually agree with
    the fused triangulation's own bed. Never invoked when `result` is
    not ok (nothing to override), and a no-op whenever the fused bed
    already has at least as much individual support as any rival bed
    (the overwhelming common case -- most throws are not near a wire).
    """
    if not result.ok or not result.cameras_used or len(result.cameras_used) < 2:
        return result
    if (
        result.max_ray_disagreement_mm is None
        or result.max_ray_disagreement_mm < MIN_DISAGREEMENT_FOR_VOTE_OVERRIDE_MM
    ):
        # See MIN_DISAGREEMENT_FOR_VOTE_OVERRIDE_MM's own docstring --
        # a near-zero fused disagreement means the real multi-ray
        # triangulation already agrees with itself far more precisely
        # than any single camera's own crude Z=0 projection can, so a
        # per-camera vote has nothing useful to add and measurably hurts.
        return result

    def _pixel_for(cam: int) -> tuple[float, float] | None:
        if result.alt_candidates_used and cam in result.alt_candidates_used and cam in alt_tip_pixels:
            return alt_tip_pixels[cam]
        if cam in tip_pixels:
            return tip_pixels[cam]
        if cam in far_end_recoveries:
            return far_end_recoveries[cam]
        return None

    candidate_cams = set(result.cameras_used) | set(tip_pixels) | set(far_end_recoveries)
    votes: dict[tuple, list[int]] = {}
    vote_xy: dict[int, tuple[float, float]] = {}
    for cam in candidate_cams:
        if cam not in calibration:
            continue
        px = _pixel_for(cam)
        if px is None:
            continue
        try:
            xy = _single_ray_board_xy(px, calibration[cam])
        except AttributeError:
            # A calibration missing the real CameraCalibration fields
            # (e.g. a test double that stubs out score_dart entirely and
            # never intends real geometry to run) -- no vote from this
            # camera, not a crash. Every other camera's vote still
            # counts; this function already treats "no usable vote" as
            # the safe default via `xy is None` just below.
            continue
        if xy is None:
            continue
        vote_xy[cam] = xy
        bed = sector_ring_for_point(xy[0], xy[1])
        votes.setdefault(bed, []).append(cam)

    fused_bed = (result.sector, result.ring)
    fused_count = len(votes.get(fused_bed, []))
    majority_bed = max(votes, key=lambda b: len(votes[b]), default=None)
    if majority_bed is None or majority_bed == fused_bed:
        return result
    if len(votes[majority_bed]) <= fused_count:
        return result

    agreeing = sorted(votes[majority_bed])
    xs = [vote_xy[c][0] for c in agreeing]
    ys = [vote_xy[c][1] for c in agreeing]
    new_xy = (float(np.mean(xs)), float(np.mean(ys)))
    new_sector, new_ring = majority_bed
    old_sector, old_ring = result.sector, result.ring
    return ScoreResult(
        ok=True,
        sector=new_sector,
        ring=new_ring,
        board_xy_mm=new_xy,
        triangulation=result.triangulation,
        n_cameras_used=len(agreeing),
        cameras_used=tuple(agreeing),
        outlier_camera=result.outlier_camera,
        alt_candidates_used=result.alt_candidates_used,
        max_ray_disagreement_mm=result.max_ray_disagreement_mm,
        reason=(
            f"per-camera vote override: {len(agreeing)} camera(s) {agreeing} "
            f"individually vote (own ray∩Z=0) for (sector={new_sector}, "
            f"ring={new_ring}) -- {len(votes[majority_bed])} vs only "
            f"{fused_count} individually agreeing with the fused triangulation's "
            f"own (sector={old_sector}, ring={old_ring}) -- see the dated "
            "2026-08-18 comment in opendarts/engines/apollo/engine.py "
            f"(original: {result.reason})"
        ),
    )


def _unanimous_off_board_override(
    result: ScoreResult,
    tip_pixels: dict[int, tuple[float, float]],
    alt_tip_pixels: dict[int, tuple[float, float]],
    genuine_cams: set[int],
    calibration: dict[int, CameraCalibration],
) -> ScoreResult:
    """2026-08-25 -- real incident, the recorded outside throw: 3 genuine
    (gate-passing) cameras, but the full 3-ray triangulation AND every
    2-of-3 RANSAC fallback pair disagreed too much to trust
    (`max_ray_disagreement_mm=183.1mm`, no pair inside
    `MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR` either) -- an honest no-score,
    even though AD confirmed a genuinely, unambiguously off-board dart
    (tip r=210mm, DOUBLE_OUTER_RADIUS_MM=170mm) and every other engine
    (Ares/Talos/Athena)
    called it `outside` correctly. Investigated end to end: neither
    existing fallback tier above applies here (`genuine_cams` is 3, not
    the tier-1 lone-camera case's 1 or the tier-2 zero-genuine case's 0)
    -- this is a genuinely new failure shape, not a gap in an existing
    one.

    MAX_RAY_DISAGREEMENT_MM's whole purpose (see scoring.py's own
    docstring) is preventing a CONFIDENTLY WRONG ON-BOARD
    segment/ring call when cameras disagree -- a real, measured incident
    (27% wrong-sector rate on adversarial trials) motivated it. That
    protection has nothing to protect when there is no on-board candidate
    in contention at all: if every genuine camera's OWN individual ray,
    completely independent of any cross-camera triangulation, lands
    outside the double ring, "outside" is a safe answer regardless of how
    much the rays disagree on exactly WHERE outside (same reasoning the
    existing 2026-08-18 tier-2 zero-genuine-camera fallback above already
    uses for its own off-board case: "two-thirds of a board length
    outside the double ring, exact direction stops mattering to the
    SCORE"). Deliberately narrower than that tier-2 fallback in three
    ways, all required by what real corpus data actually showed:

    1. UNANIMITY, not a 2-of-N classification majority. The tier-2
       fallback accepts a majority for ANY bed (on-board included) --
       fine there because it only ever runs when there are zero genuine
       cameras to begin with (nothing more precise was possible). Here,
       genuine per-camera votes on the SAME throw's full/pair
       triangulation attempts land on THREE DIFFERENT on-board sectors
       (single_outer 15, 6, 10, 17 depending which primary/alt
       combination is triangulated) even though every INDIVIDUAL
       camera's own ray∩Z=0 agrees on "outside" -- a majority-vote rule
       here would risk exactly the confidently-wrong on-board call this
       whole gate exists to prevent. Unanimity is required precisely
       because a majority was measured, on this real throw, to not be
       safe.
    2. OFF-BOARD ONLY. This override never fires toward an on-board bed
       -- only ever confirms "outside" when every vote already says so.
       An on-board unanimous-vote case is not what real data motivated
       here and is out of scope.
    3. Every camera's BOTH candidates (primary AND alt, when
       `reject_outside_roi()`/`detect_tip()` flagged one) must
       independently land outside -- not just the primary. Conservative
       by construction: if either tip-pixel interpretation for any
       camera plausibly lands on-board, this does not fire.

    Only reachable when `not result.ok` AND `result.reason` shows the
    specific "rays disagree" rejection from `score_dart()`'s own
    MAX_RAY_DISAGREEMENT_MM/MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR gate --
    deliberately NOT the `<2 camera` or `poorly-spread landmark quad`
    rejection reasons, which are different failure classes (a bad
    calibration, or too few rays to triangulate at all) this override has
    no evidence to speak to. Full corpus validation (494 real throws
    across every session in the session corpus, 2026-08-25):
    fires on exactly 1 throw (`056-OUT`, the real incident above,
    correctly recovered as `outside`), a true no-op on every other throw
    including every other currently-`ok=False` Apollo rejection in the
    corpus -- see this task's own final report for the full per-throw
    accounting.
    """
    if not result.ok and (
        result.reason is not None and "rays disagree" in result.reason
    ) and len(genuine_cams) >= 2:
        votes: list[tuple[int, tuple[float, float]]] = []
        all_outside = True
        for cam in sorted(genuine_cams):
            if cam not in calibration or cam not in tip_pixels:
                all_outside = False
                break
            candidates = [tip_pixels[cam]]
            if cam in alt_tip_pixels:
                candidates.append(alt_tip_pixels[cam])
            for px in candidates:
                xy = _single_ray_board_xy(px, calibration[cam])
                if xy is None:
                    all_outside = False
                    break
                _sector, ring = sector_ring_for_point(xy[0], xy[1])
                if ring != "outside":
                    all_outside = False
                    break
                if px == tip_pixels[cam]:
                    votes.append((cam, xy))
            if not all_outside:
                break
        if all_outside and votes:
            xs = [xy[0] for _c, xy in votes]
            ys = [xy[1] for _c, xy in votes]
            xy = (float(np.mean(xs)), float(np.mean(ys)))
            agreeing = tuple(c for c, _xy in votes)
            return ScoreResult(
                ok=True,
                sector=None,
                ring="outside",
                board_xy_mm=xy,
                triangulation=result.triangulation,
                n_cameras_used=len(agreeing),
                cameras_used=agreeing,
                max_ray_disagreement_mm=result.max_ray_disagreement_mm,
                reason=(
                    f"unanimous off-board override: every genuine camera "
                    f"{agreeing}'s own ray∩Z=0 (primary AND alt candidate, "
                    "where one existed) independently lands outside the "
                    "double ring, even though the full/2-of-3-fallback "
                    "triangulation disagreed too much to trust an on-board "
                    "call -- see the dated 2026-08-25 comment in "
                    "opendarts/engines/apollo/engine.py "
                    f"(original: {result.reason})"
                ),
            )
    return result


# 2026-08-25, tier 4 -- sparse-camera (0 or 1 genuine cameras) unanimous
# off-board fallback. See `_sparse_camera_off_board_fallback()`'s own
# docstring below for the real-incident write-up
# (the recorded outside throw) and mechanics. No dedicated confidence
# override is used for this tier -- see that docstring's own "confidence"
# paragraph for why reusing the existing `MAX_RAY_DISAGREEMENT_MM`
# placeholder (the same convention the 2026-08-18 tier-1 lone-camera
# fallback already established) is the right call here, unlike tier 5
# (`_marginal_disagreement_low_confidence_fallback()`) just below, which
# genuinely does need one.
def _sparse_camera_off_board_fallback(
    result: ScoreResult,
    tip_pixels: dict[int, tuple[float, float]],
    alt_tip_pixels: dict[int, tuple[float, float]],
    ungated_tip_pixels: dict[int, tuple[float, float]],
    ungated_alt_tip_pixels: dict[int, tuple[float, float]],
    genuine_cams: set[int],
    calibration: dict[int, CameraCalibration],
) -> ScoreResult:
    """2026-08-25 -- real incident, the recorded outside throw (AD/operator
    truth: outside, tip r=~260-282mm across every camera's own
    individual candidate, DOUBLE_OUTER_RADIUS_MM=170mm): exactly ONE
    camera (cam2) survived the real board-ROI gate, and its own detection
    carries `tip_off_axis_alt=True` -- one of `_lone_camera_diagnostics_
    clean()`'s existing red flags -- so the 2026-08-18 tier-1 lone-camera
    fallback correctly declines to trust it ALONE for a precise (sector,
    ring) call. But this throw is not actually short on signal: the
    other two cameras each produced a real (ROI-gate-REJECTED) candidate
    of their own, and investigated directly with an ad-hoc
    verification script: EVERY
    candidate pixel any camera detected for this throw -- cam0's primary
    AND alt/far-end, cam1's primary, cam2's primary AND alt -- projects
    (its own ray∩Z=0, no cross-camera triangulation) to `ring="outside"`
    independently, with comfortable margin (r=237-282mm, 67-112mm past
    the double-outer wire). This is the exact same "unanimity make a
    binary off-board classification safe regardless of which end/camera
    is trusted" reasoning `_unanimous_off_board_override()` above
    (2026-08-25, tier 3) already uses for >=2-genuine-camera throws --
    this function is the natural, narrowly-scoped extension to the <2-
    genuine-camera case tier 3 explicitly declines to touch (its own
    `len(genuine_cams) >= 2` gate).

    Deliberately reuses tier 3's own three safety properties, extended
    to a broader candidate pool:

    1. UNANIMITY across EVERY candidate pixel this engine detected for
       this throw -- not just genuine (ROI-gate-passing) cameras' own
       primary+alt (as tier 3 checks), but also every ROI-gate-REJECTED
       camera's own primary+alt (`ungated_tip_pixels`/
       `ungated_alt_tip_pixels`, collected in `ApolloEngine.score()`'s
       own per-camera loop for exactly this purpose). A camera failing
       the ROI gate says something about whether its pixel plausibly
       sits within the board's IMAGE-space projected region -- a
       different question than whether its ray, projected to the board
       PLANE, lands inside the double ring; tier 2's own 2026-08-18
       zero-genuine-camera fallback already established that an
       ROI-rejected candidate is still real, usable signal for exactly
       this kind of off-board classification question.
    2. OFF-BOARD ONLY, never toward an on-board bed -- identical to tier
       3's own scoping, same reasoning (MAX_RAY_DISAGREEMENT_MM's whole
       purpose is preventing a confidently-WRONG ON-BOARD call; nothing
       to protect when no on-board candidate is in contention).
    3. Requires at least 2 DISTINCT cameras' worth of candidates (not
       just 1) -- a single ray, however unambiguous its own off-board
       classification looks, is still only one camera's own noisy
       detection; this fallback exists specifically because that lone
       ray ALONE was judged (by the existing, separately-validated
       `_lone_camera_diagnostics_clean()` gate) not safe enough to carry
       a full (sector, ring) call by itself -- unanimity ACROSS cameras
       is what buys back trust here, not any single camera's own
       precision.

    Only reachable when `not result.ok` (tiers 1-3 and the per-camera
    vote override above all already had their chance and did not
    produce an accepted result -- in particular tier 1's own lone-camera
    fallback already tried and declined exactly the `len(genuine_cams)
    == 1` case this function also covers, so reaching here on that case
    specifically means tier 1's own cleanliness gate rejected it) AND
    `len(genuine_cams) < 2` (tier 3 above already owns the >=2-genuine
    case; this function is a no-op whenever tier 3 could have applied,
    by construction, never overlapping it).

    **Confidence**: no dedicated override -- flows through
    `score_result_to_engine_result()`'s normal `compute_confidence()`
    call exactly like tier 3 above (which received the same treatment,
    2026-08-25, and was accepted as-is). `max_ray_disagreement_mm` is
    stamped at `MAX_RAY_DISAGREEMENT_MM` (the reject threshold itself)
    -- the SAME honest "zero cross-camera corroboration" placeholder the
    2026-08-18 tier-1 lone-camera fallback already established (see that
    tier's own dated comment in `ApolloEngine.score()`), reused here
    rather than inventing a second convention, since this fallback's own
    epistemic position (a set of independent single-ray votes, no real
    multi-ray triangulation) is identical in kind to tier 1's.

    **Honest caveat, same shape as every other exploratory tier in this
    file**: validated against exactly ONE real corpus throw
    (`082-OUT`) -- there is no broader statistical reliability number to
    report the way tier 1's 93.2%/76.2% lone-camera gate measurement
    has. The false-positive spot-check in this task's own final report
    covers ordinary confident on-board throws (this fallback's own
    `not result.ok` gate means it structurally cannot fire on any of
    them), not a second real sparse-camera throw -- there is currently
    only one in the known corpus.
    """
    if result.ok or len(genuine_cams) >= 2:
        return result

    def _candidates_for(cam: int) -> list[tuple[float, float]]:
        if cam in genuine_cams:
            out = [tip_pixels[cam]]
            if cam in alt_tip_pixels:
                out.append(alt_tip_pixels[cam])
            return out
        out = []
        if cam in ungated_tip_pixels:
            out.append(ungated_tip_pixels[cam])
        if cam in ungated_alt_tip_pixels:
            out.append(ungated_alt_tip_pixels[cam])
        return out

    candidate_cams = sorted(genuine_cams | set(ungated_tip_pixels))
    primary_votes: dict[int, tuple[float, float]] = {}
    all_outside = True
    for cam in candidate_cams:
        if cam not in calibration:
            all_outside = False
            break
        pixels = _candidates_for(cam)
        if not pixels:
            continue
        for i, px in enumerate(pixels):
            try:
                xy = _single_ray_board_xy(px, calibration[cam])
            except AttributeError:
                # Same defensive convention `_per_camera_vote_override()`
                # above already established: a calibration missing the
                # real CameraCalibration fields (a test double that
                # stubs out score_dart() entirely and never intends real
                # geometry to run here) means no evidence from this
                # camera, not a crash -- and since this fallback requires
                # UNANIMITY, "no evidence" must be treated the same as
                # "this camera does not confirm off-board", i.e. the
                # whole fallback declines to fire.
                all_outside = False
                break
            if xy is None:
                all_outside = False
                break
            _sector, ring = sector_ring_for_point(xy[0], xy[1])
            if ring != "outside":
                all_outside = False
                break
            if i == 0:
                primary_votes[cam] = xy
        if not all_outside:
            break

    if all_outside and len(primary_votes) >= 2:
        xs = [xy[0] for xy in primary_votes.values()]
        ys = [xy[1] for xy in primary_votes.values()]
        xy = (float(np.mean(xs)), float(np.mean(ys)))
        agreeing = tuple(sorted(primary_votes))
        return ScoreResult(
            ok=True,
            sector=None,
            ring="outside",
            board_xy_mm=xy,
            triangulation=result.triangulation,
            n_cameras_used=len(agreeing),
            cameras_used=agreeing,
            max_ray_disagreement_mm=MAX_RAY_DISAGREEMENT_MM,
            reason=(
                f"sparse-camera unanimous off-board fallback: only "
                f"{len(genuine_cams)} genuine (gate-passing) camera(s) "
                f"survived the board-ROI gate, but {len(agreeing)} cameras "
                f"{agreeing}'s own ray∩Z=0 (primary AND alt candidate, "
                "gate-passing or not, where one existed) independently "
                "land outside the double ring -- see the dated 2026-08-25 "
                "tier-4 comment in opendarts/engines/apollo/engine.py "
                f"(original: {result.reason})"
            ),
        )
    return result


# 2026-08-25, tier 5 -- marginal-disagreement graded-confidence fallback.
# See `_marginal_disagreement_low_confidence_fallback()`'s own docstring
# below for the real-incident write-up (the recorded S16 throw) and
# mechanics.
#
# **Real, measured gap between the one real "should fire" example and
# the one real "must NOT fire" example**: `056-S16`'s own
# `max_ray_disagreement_mm` is 10.4mm (barely over `MAX_RAY_
# DISAGREEMENT_MM`'s own 10.0mm gate) and its fused triangulated point
# lands in the CORRECT sector (16 -- one ring band off from AD's
# `single_inner`, itself explained by sitting right on the treble/single
# wire, see this tier's own docstring). `025-OUT` (a 2026-08-25
# session) disagrees by 72.0mm and its fused triangulated
# point lands in a WRONG on-board sector (12/single_outer, truth is
# `outside`) -- a throw where one genuine camera's own ray individually
# votes ON-BOARD (10/treble-ish territory) while the other two
# independently vote off-board, i.e. genuinely, irreconcilably
# contradictory evidence, not a marginal near-threshold disagreement.
# Only 2 real data points exist for this specific threshold -- not a
# knife-edge number (a >6x gap separates them), but re-measure if a
# future corpus pull produces a real case landing between 10.4mm and
# 72.0mm.
MAX_RAY_DISAGREEMENT_MM_LOW_CONFIDENCE_FALLBACK_MM = 20.0

# See `_marginal_disagreement_low_confidence_fallback()`'s own docstring,
# "Confidence" paragraph, for the full derivation. Deliberately NOT run
# through `opendarts.engines.apollo.confidence.compute_confidence()`'s
# fitted logistic model -- that model's own docstring states its
# coefficients were fit and validated ONLY on throws that already
# cleared `MAX_RAY_DISAGREEMENT_MM` (the model's `disagree_risk` term
# saturates at 1.0 for anything >=10mm, which is exactly the boundary
# this fallback's own throws start at -- feeding it a real 10-20mm
# disagreement value produces a confidence around 0.94, indistinguishable
# from a genuinely accepted, trustworthy throw, which would misrepresent
# a throw that has already failed BOTH the primary gate AND the 2-of-3
# fallback pair gate). `0.30` is deliberately, structurally below every
# real accepted-throw confidence value `compute_confidence()`'s own
# docstring reports across its full 359-throw fitting corpus (the lowest
# real WRONG accepted throw there is 0.839) -- not a number chosen to
# "feel low," a number chosen to sit unambiguously below the floor of
# what this engine has ever reported for a throw it actually trusted.
LOW_CONFIDENCE_FALLBACK_CONFIDENCE = 0.30


def _marginal_disagreement_low_confidence_fallback(
    result: ScoreResult,
) -> tuple[ScoreResult, bool]:
    """2026-08-25 -- real incident, the recorded S16 throw (AD/operator
    truth 16/single_inner): exactly 2 genuine cameras (cam1, cam2) -- too
    few for `score_dart()`'s own 2-of-3 RANSAC fallback to even attempt
    (that fallback structurally requires >=3 cameras, see `scoring.py`'s
    own `len(full_cams) >= 3` gate) -- disagreeing 10.4mm, just over
    `MAX_RAY_DISAGREEMENT_MM`. Every sibling engine on this throw's own
    frames (Zeus/Talos/Athena/Ares, per this task's own evidence table)
    mostly gets throws like this right; the raw signal is present, this
    engine's all-or-nothing gate is what's discarding it.

    Each camera's OWN naive ray∩Z=0 vote, taken alone, is a WORSE
    estimate than the real multi-ray triangulation `score_dart()` already
    computed and discarded: cam1's own vote lands in sector 7, cam2's own
    vote lands in sector 16 -- a straight 1-vs-1 tie with no majority
    (this is exactly why `_per_camera_vote_override()` above is
    deliberately never allowed to see this tier's own output, see this
    function's own call site in `ApolloEngine.score()`), while the REAL
    triangulated point (`tri.board_plane_xy`, already computed by
    `score_dart()`'s own full-ray-set attempt and returned in `result.
    board_xy_mm` even on rejection) lands at sector 16 -- the correct
    sector, one ring band off (`treble` vs AD's `single_inner`,
    consistent with a throw that landed close enough to the treble/single
    wire to also produce an elevated ray disagreement in the first
    place). Using the FULL triangulation's own already-computed point
    (not a per-camera vote, not a synthetic reconstruction) is a strictly
    stronger estimate than anything else available at this point in the
    pipeline.

    Fires only when:
    - `not result.ok`
    - `"rays disagree"` is the specific rejection reason (same scoping
      discipline as tier 3/4 above -- the <2-camera and poorly-spread-
      landmark-quad rejection classes are different failure shapes this
      fallback has no evidence to speak to)
    - `result.max_ray_disagreement_mm` is not None and sits at or below
      `MAX_RAY_DISAGREEMENT_MM_LOW_CONFIDENCE_FALLBACK_MM` -- see that
      constant's own docstring for the real 10.4mm-fires/72.0mm-does-not
      measured gap this threshold sits inside
    - `result.board_xy_mm` is not None (score_dart()'s own "rays
      disagree" rejection branch always populates this from the full
      triangulation's own `tri.board_plane_xy`, whether the triangulation
      used 2 cameras with no pair fallback possible, as in `056-S16`, or
      3+ cameras where no 2-of-3 pair cleared its own stricter threshold
      either -- both cases hand back the SAME already-computed full-set
      point, so this function does not need to distinguish them)

    Never restricted to an on-board (vs `outside`) classification of the
    resulting point -- see this tier's own module-docstring for why: the
    disagreement-magnitude ceiling above is what separates this tier's
    territory from tier 3/4's (which require unanimity of INDIVIDUAL
    per-camera rays specifically for an off-board classification, a
    stronger and different kind of evidence); a marginal throw whose full
    triangulation happens to classify to `outside` is not double-counted
    here, since tier 3/4 already ran first and this function is only
    ever reached when they did not already produce an accepted result.

    **Confidence**: returns `(new_result, True)` when it fires, signaling
    `ApolloEngine.score()` to stamp the returned `EngineResult.
    confidence` (and `diagnostics["confidence"]`) at the fixed
    `LOW_CONFIDENCE_FALLBACK_CONFIDENCE` (0.30) INSTEAD OF running
    `compute_confidence()`'s normal fitted model -- see
    `LOW_CONFIDENCE_FALLBACK_CONFIDENCE`'s own docstring for why the
    fitted model itself is the wrong tool for a throw already past its
    own gate. `(result, False)` (unchanged) whenever this tier does not
    fire, which is the overwhelmingly common case.
    """
    if (
        not result.ok
        and result.reason is not None
        and "rays disagree" in result.reason
        and result.max_ray_disagreement_mm is not None
        and result.max_ray_disagreement_mm <= MAX_RAY_DISAGREEMENT_MM_LOW_CONFIDENCE_FALLBACK_MM
        and result.board_xy_mm is not None
    ):
        sector, ring = sector_ring_for_point(*result.board_xy_mm)
        new_result = ScoreResult(
            ok=True,
            sector=sector,
            ring=ring,
            board_xy_mm=result.board_xy_mm,
            triangulation=result.triangulation,
            n_cameras_used=result.n_cameras_used,
            cameras_used=result.cameras_used,
            outlier_camera=result.outlier_camera,
            alt_candidates_used=result.alt_candidates_used,
            max_ray_disagreement_mm=result.max_ray_disagreement_mm,
            reason=(
                f"marginal-disagreement low-confidence fallback: the full "
                f"ray-set triangulation disagreed by "
                f"{result.max_ray_disagreement_mm:.1f}mm (> "
                f"{MAX_RAY_DISAGREEMENT_MM}mm threshold, but <= "
                f"{MAX_RAY_DISAGREEMENT_MM_LOW_CONFIDENCE_FALLBACK_MM}mm) "
                "with no 2-of-3 pair available/passing either -- using "
                "the full triangulation's own already-computed point at "
                "a deliberately low, non-model-derived confidence rather "
                "than an honest no-score -- see the dated 2026-08-25 "
                "tier-5 comment in opendarts/engines/apollo/engine.py "
                f"(original: {result.reason})"
            ),
        )
        return new_result, True
    return result, False


def _lone_camera_diagnostics_clean(diag: dict) -> bool:
    """Gate for the lone-genuine-camera last-resort fallback below (see
    the 2026-08-18 dated comment in `ApolloEngine.score()`): true only
    when NONE of this engine's own already-established per-camera red
    flags fired on this detection. Measured on the full living
    `data/archive/clean/` corpus
    (2026-08-18): a genuine
    camera's own ray∩Z=0, taken completely alone with no other camera
    to corroborate it, independently matches operator/AD truth 93.2% of
    the time when clean per this gate (n=3116 real per-camera
    detections) vs 76.2% when it is not (n=80) -- a real, measured
    difference (not a certainty either way), which is why this fallback
    requires it rather than trusting any lone survivor unconditionally."""
    perp = diag.get("tip_cluster_perp_px")
    if perp is None or abs(perp) >= TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX:
        return False
    if diag.get("tip_off_axis_alt") or diag.get("tip_island_alt"):
        return False
    if diag.get("prior_dart_contamination_suspected"):
        return False
    return True


# 2026-08-25, tier 6 -- the true last-resort, ALWAYS-ANSWERS fallback.
# the project's own words, verbatim, the whole reason this tier exists: "Apollo
# is the only engine that returns fails to score... I don't want that
# happening." This is not a request to make Apollo more accurate on any
# specific throw -- it is a hard product requirement that `ok=False` never
# leaves this engine, full stop, even on the throws every earlier tier
# above has -- by design -- declined to touch. See
# `_last_resort_always_answer_fallback()`'s own docstring immediately
# below for the mechanism and the real incident (the recorded outside throw)
# that motivated it.
#
# **Confidence derivation.** `LOW_CONFIDENCE_FALLBACK_CONFIDENCE` (0.30,
# tier 5 above) is already the deliberately-lowest value any OTHER tier in
# this engine has ever assigned to an accepted throw -- see that
# constant's own docstring for why 0.30 sits structurally below every real
# accepted-throw confidence `compute_confidence()`'s fitted model has ever
# produced. Tier 6 needs to sit BELOW that, and by a real, reasoned
# margin, not an arbitrary smaller round number: every throw tier 5
# accepts still has a genuine multi-ray triangulation behind it (>=2 rays,
# already computed, merely disagreeing more than the primary gate wants --
# real cross-camera geometry, just noisy). Every throw tier 6 accepts has
# a STRICTLY WEAKER evidentiary basis by construction -- it only ever
# fires when the fused triangulation itself is either unavailable (0 or 1
# usable camera) or has already failed tier 5's own <=20mm ceiling too
# (see `MAX_RAY_DISAGREEMENT_MM_LOW_CONFIDENCE_FALLBACK_MM`'s docstring:
# `025-OUT` itself disagrees by 72.0mm, >6x that ceiling). That is a
# categorical downgrade in kind (no-cross-camera-corroboration-at-all, or
# cross-camera-corroboration-that-actively-disagrees), not a marginal one
# -- halving the already-rock-bottom tier-5 floor is the natural way to
# express "meaningfully less trustworthy than the least trustworthy
# thing this engine has ever shipped," while deliberately stopping short
# of 0.0: that value is reserved elsewhere in this engine
# (`compute_confidence()`'s own docstring: "no answer to have confidence
# IN, not a prediction that a wrong answer is likely") to mean "ok=False,
# there is no result" -- tier 6 always sets ok=True, so it must never
# claim the same "zero" as the thing it exists specifically to replace.
LAST_RESORT_FALLBACK_CONFIDENCE = LOW_CONFIDENCE_FALLBACK_CONFIDENCE / 2.0 # 0.15


def _last_resort_always_answer_fallback(
    result: ScoreResult,
    tip_pixels: dict[int, tuple[float, float]],
    alt_tip_pixels: dict[int, tuple[float, float]],
    ungated_tip_pixels: dict[int, tuple[float, float]],
    ungated_alt_tip_pixels: dict[int, tuple[float, float]],
    far_end_recoveries: dict[int, tuple[float, float]],
    tip_diagnostics: dict[int, dict],
    calibration: dict[int, CameraCalibration],
) -> tuple[ScoreResult, bool]:
    """2026-08-25 -- real incident, the recorded outside throw (AD/operator
    truth: outside, tip_xy_mm=(-202.1, -104.2)mm). Re-verified directly
    against this throw's own stored raw frames before writing this
    function -- the prior
    agent's own diagnosis was confirmed correct, not assumed: exactly 2
    genuine (ROI-gate-passing) cameras (cam0, cam1); cam2 produced a real
    detection but failed the ROI gate on both its primary AND alt
    candidate. `score_dart()`'s own full 2-camera triangulation (cam0,
    cam1) disagrees by 71.98mm (>>`MAX_RAY_DISAGREEMENT_MM_LOW_
    CONFIDENCE_FALLBACK_MM`'s 20.0mm ceiling, so tier 5 above correctly
    declines too) -- an honest, irreducible contradiction at the FUSED
    level. But every individual camera's own solo ray∩Z=0
    (`_single_ray_board_xy`), taken alone, is NOT ambiguous:
        cam0 primary (genuine): xy=(-215.1, -88.2) r=232.5mm outside
        cam1 primary (genuine): xy=( 4.1, 101.6) r=101.7mm 20/treble
        cam1 alt (genuine): xy=( 4.7, 96.0) r= 96.1mm 20/single_inner
        cam2 primary (ROI-rejected): xy=(-300.6, 24.6) r=301.6mm outside
        cam2 alt (ROI-rejected): xy=(-293.6, 27.1) r=294.8mm outside
        cam2 far-end (ROI-rejected): xy=(-261.1, -25.9) r=262.4mm outside
    i.e. 2 of the 3 cameras that produced ANY candidate at all (cam0,
    cam2) independently, individually vote "outside" -- comfortably past
    `DOUBLE_OUTER_RADIUS_MM` (170mm) regardless of exact direction, the
    same "exact direction stops mattering to the SCORE" reasoning tier 2's
    own 2026-08-18 docstring already established -- while only cam1 votes
    on-board. `sector_ring_for_point()` classifies cam0's own vote to
    "outside" and it lands within ~21mm of AD's own confirmed truth point,
    despite being a single, uncorroborated ray. This is a real, genuine
    contradiction (this engine's own multi-ray triangulation disagrees
    with itself by 72mm) resolved cleanly by a plain per-camera majority
    -- exactly the shape the project's own task description anticipated,
    confirmed here by rerunning the real pipeline against this throw's
    stored raw frames rather than assuming it.

    **Mechanism -- deliberately general, not a `025-OUT` special case.**
    Only reachable when tiers 1-5 above have ALL already declined (this
    function's own `if result.ok: return result, False` guard is the
    first line -- an already-accepted result, from ANY earlier tier, is
    never touched). At that point:

    1. **One vote per camera, not one vote per candidate pixel.** Every
       camera that produced ANY candidate at all for this throw --
       genuine (ROI-gate-passing) or not, primary or alt, even a
       ROI-rejected far-end recovery -- gets exactly ONE vote: its own
       single BEST candidate, in trust order (genuine primary > genuine
       alt > ungated/ROI-rejected primary > ungated alt > far-end
       recovery), classified via that candidate's own ray∩Z=0
       (`_single_ray_board_xy` + `sector_ring_for_point`). Deliberately
       one-vote-per-camera, not one-vote-per-pixel: a camera with an
       ambiguous alt candidate (cam1 above) must not get 2 votes against
       a camera with only one candidate (cam0, cam2) -- that would let a
       single noisy detection outvote two independent cameras, exactly
       the failure `_per_camera_vote_override()`'s own docstring already
       warns a blanket majority-of-pixels scheme risks.
    2. **Strict plurality wins.** The bed (sector, ring) with the most
       per-camera votes is used, board_xy_mm averaged across the
       agreeing cameras' own votes -- same "average the agreeing
       cameras' own xy" convention every earlier per-camera-vote tier in
       this file already uses (`_per_camera_vote_override()`, tier 2,
       tier 4). `025-OUT` itself resolves here: 2 votes for "outside"
       (cam0, cam2) strictly beats 1 vote for "20/treble" (cam1) -- no
       tie-break needed on the one real corpus throw this was built and
       verified against.
    3. **Tie-break, for the general case this corpus doesn't currently
       exercise.** An exact N-way tie among per-camera votes (e.g. 1
       genuine camera vs 1 ROI-rejected camera, each voting a different
       on-board bed) has no majority to lean on. Broken, in order:
       (a) if the FUSED triangulation itself still produced a point
       (`result.board_xy_mm` -- populated whenever score_dart() got far
       enough to triangulate at all, even on a "rays disagree" rejection,
       per tier 5's own docstring) and that point's own classification
       matches one of the tied beds, prefer it -- a real multi-ray
       computation, even a rejected one, used more of this throw's actual
       geometry than any single ray did; (b) otherwise, prefer whichever
       tied camera's own per-camera diagnostics are clean per the
       already-validated, already-measured `_lone_camera_diagnostics_
       clean()` gate (tier 1's own 93.2%-vs-76.2% real corpus
       measurement) -- a genuine, previously-established per-camera
       reliability signal, not an arbitrary pick; (c) if still tied
       (nothing clean, or no diagnostics recorded for either candidate --
       true only for a non-genuine camera's vote, since `tip_diagnostics`
       is only ever populated for ROI-gate-passing cameras), fall back to
       the lowest camera index among the tied group -- deterministic, and
       explicitly documented here as the genuinely-arbitrary last resort
       it is, reached only when every other signal this engine has is
       itself tied.
    4. **Exactly one camera voted at all.** The lone vote is used
       directly, with NO diagnostics-clean gate -- unlike tier 1 above
       (which requires a clean gate and only ever replaces a CERTAIN
       no-score), tier 6 is the true final fallback: a dirty lone camera
       is still strictly more informative than declining to answer at
       all, and this is the only tier structurally later than tier 1, so
       reaching here on a 1-camera throw already means tier 1's own
       cleanliness gate declined it.
    5. **Zero cameras produced ANY candidate, but the fused triangulation
       still has a point** (`result.board_xy_mm` not None despite no
       per-camera vote existing in the pooled dicts -- should not
       normally happen since score_dart() needs >=2 tip_pixels to
       triangulate at all and every tip_pixels entry is also pooled here,
       but handled defensively): classify and use that point directly,
       the same mechanism tier 5 already uses, just without tier 5's own
       <=20mm disagreement ceiling -- tier 6 is the last stop, so that
       ceiling no longer applies.
    6. **Zero cameras produced ANY candidate AND no triangulated point
       exists at all -- the genuine "no image data whatsoever" floor.**
       See `ApolloEngine.score()`'s own control-flow audit comment
       (2026-08-25) for whether this is reachable in real operation
       (short answer: not via any live capture_daemon.py call site this
       task could find, but not provable impossible from this module
       alone). Per the project's own "full stop" instruction this still must
       return `ok=True`, not decline -- there is genuinely zero
       geometric evidence to place a point from, so this returns an
       explicit, honestly-labeled PLACEHOLDER: `sector=None`,
       `ring="outside"`, `board_xy_mm` pinned 50mm past
       `DOUBLE_OUTER_RADIUS_MM` along the +x axis (a fixed, clearly
       synthetic point chosen only so the (sector, ring, xy) schema
       tuple stays internally consistent -- `sector_ring_for_point()`
       really does classify it "outside" -- NOT a real position estimate
       of any kind). "Outside" over an arbitrary on-board guess because
       darts play generates real off-board misses regularly (bounce-outs,
       airballs, a throw that misses every camera's field of view) while
       zero per-camera candidates AND zero triangulated point is itself
       already the signature of "nothing board-shaped was detected" --
       claiming a specific on-board segment with literally no pixel
       evidence at all would be a strictly worse fabrication than
       claiming the one classification this total-absence-of-signal
       pattern is already most consistent with.

    **Validation**: fires on exactly one real corpus throw today
    (`025-OUT`) -- the only currently-known `ok=False` throw in
    the session corpus after tiers 1-5 (see this task's own
    final report for the full corpus re-score). Case 6 above (the true
    zero-evidence floor) has NEVER been observed on any real corpus throw
    -- it exists purely to make the "never ok=False" guarantee
    structurally airtight, not because a real throw has needed it.

    Returns `(new_result, True)` whenever this tier fires, signaling
    `ApolloEngine.score()` to stamp confidence at the fixed
    `LAST_RESORT_FALLBACK_CONFIDENCE` (0.15) instead of running
    `compute_confidence()`'s normal fitted model -- same reasoning as
    tier 5's own override (see `LOW_CONFIDENCE_FALLBACK_CONFIDENCE`'s
    docstring: the fitted model was never fit or validated on throws this
    far outside its own training distribution). `(result, False)`
    (unchanged) whenever `result` was already `ok=True` on entry -- this
    tier structurally cannot downgrade an already-accepted answer from
    any earlier tier.
    """
    if result.ok:
        return result, False

    def _best_vote_for(cam: int) -> tuple[float, float] | None:
        if cam not in calibration:
            return None
        for px in (
            tip_pixels.get(cam),
            alt_tip_pixels.get(cam),
            ungated_tip_pixels.get(cam),
            ungated_alt_tip_pixels.get(cam),
            far_end_recoveries.get(cam),
        ):
            if px is None:
                continue
            try:
                xy = _single_ray_board_xy(px, calibration[cam])
            except AttributeError:
                # Same defensive convention every earlier per-camera-vote
                # tier in this file already establishes -- a test double
                # calibration missing real geometry fields means no vote
                # from this camera, not a crash.
                continue
            if xy is not None:
                return xy
        return None

    all_cams = sorted(
        set(tip_pixels) | set(alt_tip_pixels) | set(ungated_tip_pixels)
        | set(ungated_alt_tip_pixels) | set(far_end_recoveries)
    )
    cam_votes: dict[int, tuple[tuple[float, float], tuple]] = {}
    for cam in all_cams:
        xy = _best_vote_for(cam)
        if xy is not None:
            cam_votes[cam] = (xy, sector_ring_for_point(xy[0], xy[1]))

    if cam_votes:
        vote_counts: dict[tuple, list[int]] = {}
        for cam, (_xy, bed) in cam_votes.items():
            vote_counts.setdefault(bed, []).append(cam)
        max_votes = max(len(cams) for cams in vote_counts.values())
        tied_beds = [bed for bed, cams in vote_counts.items() if len(cams) == max_votes]

        if len(tied_beds) == 1:
            winning_bed = tied_beds[0]
        else:
            # Tie-break (a): does the fused (even if rejected)
            # triangulation's own point classify to one of the tied beds?
            fused_bed = (
                sector_ring_for_point(*result.board_xy_mm)
                if result.board_xy_mm is not None
                else None
            )
            if fused_bed is not None and fused_bed in tied_beds:
                winning_bed = fused_bed
            else:
                # Tie-break (b): prefer a tied bed with a clean-diagnostics
                # camera behind it; (c) else lowest camera index overall.
                def _bed_sort_key(bed: tuple) -> tuple:
                    cams_for_bed = sorted(vote_counts[bed])
                    any_clean = any(
                        _lone_camera_diagnostics_clean(tip_diagnostics.get(c, {}))
                        for c in cams_for_bed
                    )
                    return (0 if any_clean else 1, cams_for_bed[0])

                winning_bed = min(tied_beds, key=_bed_sort_key)

        agreeing = sorted(vote_counts[winning_bed])
        xs = [cam_votes[c][0][0] for c in agreeing]
        ys = [cam_votes[c][0][1] for c in agreeing]
        xy = (float(np.mean(xs)), float(np.mean(ys)))
        sector, ring = winning_bed
        new_result = ScoreResult(
            ok=True,
            sector=sector,
            ring=ring,
            board_xy_mm=xy,
            triangulation=result.triangulation,
            n_cameras_used=len(agreeing),
            cameras_used=tuple(agreeing),
            # `result.max_ray_disagreement_mm` can genuinely be None here
            # (e.g. the incoming rejection was score_dart()'s own
            # <2-camera or poorly-spread-landmark-quad reason, neither of
            # which sets a disagreement figure at all -- a per-camera vote
            # is still fully computable in that case, it just has no
            # FUSED disagreement number to carry forward). `compute_
            # confidence()` asserts non-None whenever ok=True (same
            # contract tier 1's own lone-camera fallback above already
            # documents), so fall back to the same honest, most-
            # conservative placeholder that fallback already established
            # rather than leave this None and crash the very next call.
            max_ray_disagreement_mm=(
                result.max_ray_disagreement_mm
                if result.max_ray_disagreement_mm is not None
                else MAX_RAY_DISAGREEMENT_MM
            ),
            reason=(
                f"last-resort always-answer fallback: {len(agreeing)} of "
                f"{len(cam_votes)} camera(s) with any usable candidate "
                f"individually vote (own ray∩Z=0) for (sector={sector}, "
                f"ring={ring}) -- every earlier tier declined this throw "
                "(tiers 1-5 in opendarts/engines/apollo/engine.py all "
                "failed to produce ok=True); see the dated 2026-08-25 "
                "tier-6 comment for the full mechanism "
                f"(original: {result.reason})"
            ),
        )
        return new_result, True

    if result.board_xy_mm is not None:
        sector, ring = sector_ring_for_point(*result.board_xy_mm)
        new_result = ScoreResult(
            ok=True,
            sector=sector,
            ring=ring,
            board_xy_mm=result.board_xy_mm,
            triangulation=result.triangulation,
            n_cameras_used=result.n_cameras_used,
            cameras_used=result.cameras_used,
            outlier_camera=result.outlier_camera,
            alt_candidates_used=result.alt_candidates_used,
            # Same None-guard as the branch above -- score_dart()'s own
            # "rays disagree" rejection can itself carry
            # `max_ray_disagreement_mm=None` (its `max_disagreement is
            # None or max_disagreement > MAX_RAY_DISAGREEMENT_MM` gate,
            # `scoring.py` line ~386, treats a None disagreement as a
            # rejection reason on its own) even while `board_xy_mm` is
            # populated from `tri.board_plane_xy`.
            max_ray_disagreement_mm=(
                result.max_ray_disagreement_mm
                if result.max_ray_disagreement_mm is not None
                else MAX_RAY_DISAGREEMENT_MM
            ),
            reason=(
                "last-resort always-answer fallback: zero individual "
                "per-camera candidates were usable, but the fused "
                "triangulation itself still produced a point -- using it "
                "directly rather than an honest no-score (no <=20mm "
                "ceiling applied, unlike tier 5, since this is the final "
                "fallback) -- see the dated 2026-08-25 tier-6 comment in "
                "opendarts/engines/apollo/engine.py "
                f"(original: {result.reason})"
            ),
        )
        return new_result, True

    # Case 6 -- the genuine zero-evidence floor. See this function's own
    # docstring, point 6, for why "outside" at a fixed synthetic point is
    # the honest placeholder here, not a real estimate.
    placeholder_xy = (DOUBLE_OUTER_RADIUS_MM + 50.0, 0.0)
    sector, ring = sector_ring_for_point(*placeholder_xy)
    new_result = ScoreResult(
        ok=True,
        sector=sector,
        ring=ring,
        board_xy_mm=placeholder_xy,
        triangulation=None,
        n_cameras_used=0,
        cameras_used=None,
        # Same None-guard convention as both branches above (and tier 1's
        # own pre-existing lone-camera fallback) -- `compute_confidence()`
        # asserts non-None whenever ok=True, and there is truly no real
        # disagreement figure to report when zero rays exist at all, so
        # this stamps the same most-conservative placeholder rather than
        # leave it None.
        max_ray_disagreement_mm=MAX_RAY_DISAGREEMENT_MM,
        reason=(
            "last-resort always-answer fallback: ZERO per-camera candidates "
            "and no triangulated point existed at all for this throw -- "
            "genuinely no geometric evidence to place a point from. This is "
            "an explicit, honestly-labeled PLACEHOLDER answer "
            f"(board_xy_mm={placeholder_xy}, pinned outside the double ring "
            "purely to keep the (sector, ring, xy) schema internally "
            "consistent), not a real position estimate -- see the dated "
            "2026-08-25 tier-6 comment (point 6) in "
            "opendarts/engines/apollo/engine.py "
            f"(original: {result.reason})"
        ),
    )
    return new_result, True


def score_result_to_engine_result(result: ScoreResult) -> EngineResult:
    """Pack a opendarts.pipeline.ScoreResult into an EngineResult, carrying
    every field ScoreResult has beyond the shared core four (ok/sector/
    ring/board_xy_mm) into `diagnostics` -- nothing is silently dropped.
    Public (not module-private) so other callers building an EngineResult
    from a real ScoreResult (e.g. a future offline tool) reuse this exact
    mapping instead of re-deriving it."""
    tri = result.triangulation
    # The v2 package schema null-vs-[] pass (2026-08-27, QA's own rule: "a
    # collection with no members -> [] never null, never absent") --
    # `cameras_used`/`alt_candidates_used` mirror
    # `opendarts.capture.throw_package._score_result_to_dict()`'s identical
    # fix (this is the SAME dict shape, one level down: this one becomes
    # `other_engines.Apollo.diagnostics.*`, that one becomes the top-level
    # `result.json` rollup) -- see that function's own comment for the
    # full reasoning, not repeated here. `alt_candidates_used` is the
    # exact field QA's own real-package sweep found null 90/90 packages
    # where Apollo's winning combination never needed an alt candidate.
    diagnostics: dict = {
        "n_cameras_used": result.n_cameras_used,
        "max_ray_disagreement_mm": result.max_ray_disagreement_mm,
        "cameras_used": list(result.cameras_used) if result.cameras_used is not None else [],
        "outlier_camera": result.outlier_camera,
        "alt_candidates_used": (
            list(result.alt_candidates_used) if result.alt_candidates_used is not None else []
        ),
        "triangulation": None
        if tri is None
        else {
            "ok": tri.ok,
            "point_xyz": None if tri.point_xyz is None else tri.point_xyz.tolist(),
            "board_plane_xy": tri.board_plane_xy,
            "plane_discrepancy_mm": tri.plane_discrepancy_mm,
            "per_ray_distance_mm": (
                list(tri.per_ray_distance_mm) if tri.per_ray_distance_mm is not None else []
            ),
            "n_rays": tri.n_rays,
        },
    }
    # 2026-08-14 -- see opendarts/engines/apollo/confidence.py's module
    # docstring for the full derivation/validation (calibrated against
    # the real 360-throw data/archive/clean/ corpus, ECE 0.035 in-sample
    # / 0.0195 leave-one-session-out). Originally diagnostics-only (the
    # shared EngineResult.confidence field didn't exist yet when this was
    # built and two engines were adding a confidence score in parallel --
    # avoiding it sidestepped a possible collision on that shared file).
    # Talos has since added EngineResult.confidence for real (2026-08-14,
    # `d683b40`) and populates it directly, so this now does too --
    # stamped in BOTH places (top-level field is the real, "official"
    # value; diagnostics keeps the same value for anyone already reading
    # it from there) rather than leaving the shared field silently None
    # while Talos's own result carries a real number in the identical
    # spot.
    confidence = compute_confidence(
        ok=result.ok,
        board_xy_mm=result.board_xy_mm,
        max_ray_disagreement_mm=result.max_ray_disagreement_mm,
        fallback_used=result.outlier_camera is not None,
    )
    diagnostics["confidence"] = confidence
    return EngineResult(
        ok=result.ok,
        sector=result.sector,
        ring=result.ring,
        board_xy_mm=result.board_xy_mm,
        reason=result.reason,
        diagnostics=diagnostics,
        confidence=confidence,
    )


class ApolloEngine:
    """The registry entry named "Apollo" -- see opendarts/engines/registry.py."""

    name = "Apollo"
    # What a caller-shared `opendarts.imageops.DiffCrop` must satisfy for
    # this engine to use it -- see `detect_tip()`'s `precomputed`.
    precompute_requirements = PRECOMPUTE_REQUIREMENTS

    def score(
        self,
        bg_images: dict[int, np.ndarray],
        frame_images: dict[int, np.ndarray],
        calibration: dict[int, CameraCalibration],
        *,
        prior_dart_line_px: PriorDartLinePx | None = None,
        precomputed: dict[int, DiffCrop] | None = None,
    ) -> EngineResult:
        """Detect a tip per camera, gate each through the board-ROI
        filter, then triangulate/score once. **2026-08-12 -- this is now
        THE one real call sequence**, not a duplicate of anything: before
        this date, `opendarts/live/capture_daemon.py`'s
        `handle_ready_to_capture()` and `opendarts/capture/replay.py`'s
        `replay_throw()` each independently inlined this exact same
        sequence (a deliberate, documented bypass of this class, kept
        only until this wrapping was proven byte-identical -- see this
        module's own docstring above). Both call sites now call THIS
        method instead of inlining their own copy -- a real, measured
        before/after across the full archived corpus proved this
        wrapping introduces no behavior change, which is what made
        removing the bypass safe.

        `prior_dart_line_px`: added 2026-08-16, keyword-only and
        optional so every existing call site (generic engine dispatch,
        every test, every offline eval script -- none of which pass it)
        is completely unaffected. `{cam_index: (tip_px, far_end_px)}` for
        the immediately-prior throw of the SAME visit, same camera --
        see `opendarts.engines.apollo.prior_dart_context.
        find_prior_dart_line_px()` for the real lookup this is meant to
        be filled from (used by both `opendarts.live.capture_daemon.
        handle_ready_to_capture()` live and `opendarts.capture.replay.
        replay_throw_with_engine()` offline, per docs/DESIGN.md's
        "Replay is the source of truth" -- both must reach the same contamination
        decision for the same package). See `tip_detection.detect_tip()`'s
        own `prior_dart_line_px` parameter for what this actually does.

        `precomputed`: optional per-camera `opendarts.imageops.DiffCrop`
        (2026-09-06 perf pass) -- the shared gray/|diff|/blur front end
        a caller (Zeus) computed once for all its sub-engines. Forwarded
        to `detect_tip()`, which validates and may ignore it; results are
        bit-identical with or without it.
        """
        cams = sorted(set(bg_images) & set(frame_images) & set(calibration))
        tip_pixels: dict[int, tuple[float, float]] = {}
        alt_tip_pixels: dict[int, tuple[float, float]] = {}
        far_end_recoveries: dict[int, tuple[float, float]] = {}
        # 2026-08-16 -- prior-dart-in-visit contamination guard, see
        # tip_detection.py's dated module docstring entry for the full
        # real-incident write-up. Which cameras detect_tip() flagged as
        # suspicious (both candidate ends sitting on the immediately-
        # prior dart's own line in THIS camera) -- gathered here, acted
        # on AFTER the loop (see below this loop for why: a CANDIDATE
        # combination for score_dart()'s own disagreement-based
        # selection to weigh, never a hard per-camera pre-filter -- an
        # earlier version of this guard excluded a flagged camera
        # unconditionally right here and was measured, on the real
        # corpus, to regularly destroy legitimate well-agreeing
        # multi-camera throws whenever a later dart landed close to an
        # earlier one, a normal darts outcome that trips this same
        # geometric signal without any real contamination).
        suspected_cams: set[int] = set()
        # Per-genuine-camera detect_tip() diagnostics, keyed by cam --
        # kept around only for the 2026-08-18 last-resort fallback's own
        # `_lone_camera_diagnostics_clean()` gate below (see its
        # docstring); every existing code path in this method ignores it.
        tip_diagnostics: dict[int, dict] = {}
        # Raw (pre-ROI-gate) tip_px for every camera that produced ANY
        # candidate at all, gate-passing or not -- kept only for the
        # 2026-08-18 zero-genuine-camera fallback further below, which is
        # the one place in this method that deliberately looks at a
        # candidate the real ROI gate already rejected. Every existing
        # code path above still only ever sees the gated `tip_pixels`.
        ungated_tip_pixels: dict[int, tuple[float, float]] = {}
        # Raw (pre-ROI-gate) alt_tip_px, same population as
        # `ungated_tip_pixels` above -- added 2026-08-25 for the
        # sparse-camera off-board fallback (`_sparse_camera_off_board_
        # fallback()` below), which needs a ROI-rejected camera's OWN
        # ambiguous alt candidate too (mirroring what tier 3's
        # `_unanimous_off_board_override()` already does for GENUINE
        # cameras' alt_tip_pixels) -- not read by anything else in this
        # method.
        ungated_alt_tip_pixels: dict[int, tuple[float, float]] = {}
        # own_tip_line_px (2026-09-01, latency task -- see
        # opendarts.engines.apollo.prior_dart_context module docstring's
        # "why call anything off disk at all" section for the full
        # writeup): THIS throw's own (tip_px, far_end_px) per camera,
        # from the SAME raw (pre-ROI-gate) detect_tip() call every other
        # ungated_* dict below already captures -- deliberately the raw
        # form, matching find_prior_dart_line_px()'s own disk-based path
        # exactly (plain detect_tip(bg, frame), no ROI gating), so a
        # FUTURE throw's own prior-dart-contamination lookup reading this
        # back is a byte-accurate stand-in for a fresh recompute, not an
        # approximation biased by whatever this throw's own scoring did
        # or didn't accept. Attached to this EngineResult's own
        # diagnostics below -- like every other per-camera diagnostic
        # this method already writes (tip_diagnostics, prior_dart_
        # contamination_suspected), this DOES get persisted to disk via
        # EngineResult.to_dict()'s wholesale diagnostics serialization
        # (result.json/other_engines' own diagnostics field) -- no
        # special "in-memory only" carve-out, matching this engine's own
        # existing "diagnostics is an open per-engine bucket" convention.
        # A live-loop consumer reads it back from the IN-MEMORY
        # EngineResult moments after scoring (never from a disk re-read),
        # which is the actual latency win -- the fact it's ALSO durably
        # persisted is a side effect of reusing the existing diagnostics
        # channel, not the reason this exists.
        own_tip_line_px: PriorDartLinePx = {}
        for cam in cams:
            prior_line = prior_dart_line_px.get(cam) if prior_dart_line_px else None
            # The `precomputed` keyword is only passed when there is a
            # bundle for this camera -- test doubles for detect_tip()
            # keep the plain signature (same discipline Zeus applies).
            pc_kwargs = {"precomputed": precomputed[cam]} if precomputed and cam in precomputed else {}
            det = detect_tip(
                bg_images[cam], frame_images[cam], prior_dart_line_px=prior_line, **pc_kwargs
            )
            if det.ok and det.tip_px is not None:
                ungated_tip_pixels[cam] = det.tip_px
                if det.alt_tip_px is not None:
                    ungated_alt_tip_pixels[cam] = det.alt_tip_px
                if det.far_end_px is not None:
                    own_tip_line_px[cam] = (det.tip_px, det.far_end_px)
            # 2026-08-12 -- gate every raw detect_tip() result through the
            # board-ROI filter before it can reach score_dart(). This
            # engine was originally built before that gate existed and
            # was missing it -- caught by a real corpus-wide before/after
            # comparison actually failing, not by inspection.
            det = reject_outside_roi(det, calibration[cam])
            if det.ok and det.tip_px is not None:
                tip_pixels[cam] = det.tip_px
                tip_diagnostics[cam] = det.diagnostics
                if det.alt_tip_px is not None:
                    alt_tip_pixels[cam] = det.alt_tip_px
                if det.diagnostics.get("prior_dart_contamination_suspected"):
                    suspected_cams.add(cam)
            elif det.diagnostics.get("board_roi_far_end_inside") and det.far_end_px is not None:
                far_end_recoveries[cam] = det.far_end_px

        # LAST-RESORT END FLIP, added 2026-08-14. A camera the ROI gate
        # rejected whose component's OTHER end IS on-board is, physically,
        # a dart whose flight is angled up out of the board face and whose
        # end-choice went the wrong way (looked at directly on real frames
        # -- e.g. throw_1786666454680 cam0 and cam2, where
        # the correct blob's tip end sits inside the double ring and the
        # detector took the flight end above the board). The existing
        # alt_tip_px promotion inside reject_outside_roi() does not cover
        # it: alt_tip_px only exists when tip_detection judged the end
        # call ambiguous, and these are the cases where it was confidently
        # wrong.
        #
        # Deliberately gated on the throw being otherwise UNSCOREABLE
        # (< 2 surviving cameras), not applied whenever it's available.
        # That gate is the whole difference between a real fix and a
        # regression, measured on the real 300-throw data/archive/clean/
        # corpus, production oriented_landmarks calibration:
        # ungated (flip whenever available): +1 / -2 (289 -> 288)
        # gated on < 2 surviving cameras: +1 / -0 (289 -> 290)
        # The flipped end is a genuinely weaker observation than a
        # gate-passing primary -- it is the right call only when the
        # alternative is no score at all, and adding it to an already-
        # sufficient camera set measurably poisons good triangulations.
        # Robustness (no second real corpus exists to hold out -- see
        # docs/DESIGN.md): re-measured end-to-end at DIFF_THRESHOLD 22/25/
        # 28/31/35, where it is +1/-0, +1/-0, +0/-0, +1/-0, +0/-0 -- never
        # negative at any setting, and positive on a DIFFERENT throw each
        # time, which is what distinguishes a structural rule from a
        # coincidence at one operating point.
        genuine_cams = set(tip_pixels)
        recovery_cams: set[int] = set()
        if len(tip_pixels) < 2:
            tip_pixels.update(far_end_recoveries)
            recovery_cams = set(far_end_recoveries) - genuine_cams

        result = score_dart(tip_pixels, calibration, alt_tip_pixels=alt_tip_pixels or None)

        # 2026-08-17 -- genuine-anchored recovery arbitration. A flipped
        # far end is a WEAKER observation than a gate-passing primary
        # (this file's own end-flip comment above), and two flipped ends
        # are correlated the same way (both biased up their own shafts) --
        # so when the accepted result's cameras_used contains NO genuine
        # gate-passing camera even though one exists, that acceptance is
        # exactly the correlated-garbage pair trap: the two recoveries
        # agree with each other and the RANSAC pair-picker excludes the
        # one real tip as the "outlier." Real incidents, all with the
        # identical structure (2 recoveries injected + 1 genuine tip,
        # recovery-only pair accepted, genuine camera excluded, wrong
        # answer): throw_1786690609789 (T16 scored S16, 23.8mm off),
        # S20 throw (S20 scored outside, 55.8mm off),
        # S1 throw (S1 scored outside, 36.4mm off),
        # S8 throw (S8 scored 16/single_outer, 27.7mm off).
        # Fix: re-arbitrate over {genuine + one recovery} subsets via the
        # exact same score_dart(); accept the best; if none passes, reject
        # honestly rather than return the correlated-recoveries answer.
        # Measured on the full living clean/ corpus (996 AD-matched,
        # 2026-08-17): +1/-0 (094-S8 fixed via its genuine+recovery pair;
        # 057-S20 wrong-"outside" becomes an honest no-score; 065-S1 moves
        # closer but stays a miss; the original end-flip incident
        # throw_1786666454680 and every other throw unchanged).
        if recovery_cams and genuine_cams and result.ok:
            used_cams = set(result.cameras_used or ())
            if used_cams.isdisjoint(genuine_cams):
                best_anchored: ScoreResult | None = None
                for rec_cam in sorted(recovery_cams):
                    subset = {c: tip_pixels[c] for c in genuine_cams}
                    subset[rec_cam] = tip_pixels[rec_cam]
                    subset_alts = {
                        c: px for c, px in alt_tip_pixels.items() if c in subset
                    } or None
                    candidate = score_dart(subset, calibration, alt_tip_pixels=subset_alts)
                    if candidate.ok and (
                        best_anchored is None
                        or (
                            candidate.max_ray_disagreement_mm is not None
                            and best_anchored.max_ray_disagreement_mm is not None
                            and candidate.max_ray_disagreement_mm
                            < best_anchored.max_ray_disagreement_mm
                        )
                    ):
                        best_anchored = candidate
                if best_anchored is not None:
                    result = best_anchored
                else:
                    result = ScoreResult(
                        ok=False,
                        sector=None,
                        ring=None,
                        board_xy_mm=None,
                        triangulation=None,
                        n_cameras_used=len(tip_pixels),
                        reason=(
                            "rejected: only far-end-recovery cameras agreed with "
                            "each other while excluding every genuine gate-passing "
                            "tip -- two flipped far ends are correlated (both "
                            "biased along their own shafts), so their mutual "
                            "agreement is not evidence (see the genuine-anchored "
                            "recovery arbitration comment in "
                            "opendarts/engines/apollo/engine.py)"
                        ),
                    )

        # 2026-08-16 -- prior-dart-in-visit contamination guard, decision
        # point (see suspected_cams' own comment above for why this is a
        # candidate-combination comparison, not a pre-filter). Only
        # considered when at least one camera was flagged AND dropping
        # every flagged camera still leaves score_dart() enough rays to
        # triangulate at all (score_dart()'s own >=2-camera floor) --
        # dropping down to <2 cameras makes `result_dropped` an automatic
        # `ok=False` below, which never wins against a real `result`, so
        # a throw with 2+ flagged cameras (measured on the real corpus:
        # this is common for a later dart legitimately thrown close to
        # an earlier one, NOT evidence every flagged camera is actually
        # bad) safely falls straight through to today's unmodified
        # `result`.
        # Real incident found DURING this guard's own corpus validation
        # (the recorded D2 throw), kept here as the reason for
        # this gate rather than silently tuned around: that throw's own
        # `result` (full 3-camera set, WITH the existing alt_tip_pixels
        # combination search already applied) was ALREADY correct
        # (3.64mm disagreement, using camera 0's alt candidate -- the
        # existing, separately-proven alt-candidate rescue this engine
        # already had). Camera 0 also happened to be flagged suspicious
        # by the prior-dart-line signal (a false positive, same
        # legitimate-close-grouping mechanism as every other false
        # positive measured on this corpus). Comparing purely by "which
        # disagreement number is lower" preferred DROPPING camera 0
        # entirely (0.65mm using only cameras 1+2) -- discarding cam0's
        # already-working alt-rescue for a confidently WRONG answer:
        # cameras 1+2 alone happen to agree tightly while both being
        # biased the same way (the correlated-bias
        # mechanism, exposed here via a NEW path, not the
        # already-rejected batch8 lever of lowering
        # MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR's own trigger). Lower
        # ray disagreement is not, by itself, sufficient evidence of
        # correctness -- this project already knows that. So: only
        # SECOND-GUESS an already-accepted `result` when its own
        # disagreement is itself elevated enough to be worth
        # reconsidering (comfortably above 056-D2's real 3.64mm, at the
        # real incident's own 7.55mm) -- an outright-rejected `result`
        # always qualifies too (nothing to lose by trying).
        result_is_elevated_or_failed = not result.ok or (
            result.max_ray_disagreement_mm is not None
            and result.max_ray_disagreement_mm > PRIOR_DART_DROP_MIN_FULL_DISAGREEMENT_MM
        )
        if result_is_elevated_or_failed and (suspected_cams & tip_pixels.keys()):
            tip_pixels_dropped = {
                c: px for c, px in tip_pixels.items() if c not in suspected_cams
            }
            # 2026-08-17 -- recovery backfill for the drop path. Dropping
            # a prior-dart-suspected camera used to dead-end whenever it
            # left <2 cameras (result_dropped can't exist), even when a
            # THIRD camera was ROI-rejected with its far end on-board --
            # i.e. a weaker-but-real observation was available and unused
            # while the throw no-scored. Real incidents, identical
            # structure (one genuine tip is a prior-dart-suspected tiny
            # fragment, one genuine tip is good, one rejected camera's
            # far end sits within ~5px of the true tip):
            # the recorded S10 throw (no-score, rays 28.8mm) and
            # the recorded D20 throw (no-score, rays 16.5mm). Backfill
            # from far_end_recoveries only INSIDE this drop attempt (the
            # <2-surviving-cameras necessity gate above is untouched),
            # under the exact same dropped_is_better arbitration below.
            # Measured on the full living clean/ corpus (996 AD-matched,
            # 2026-08-17): +2/-0 (both incidents now score their correct
            # bed; no other throw changes at all).
            if len(tip_pixels_dropped) < 2:
                for cam, far_px in far_end_recoveries.items():
                    if cam not in tip_pixels_dropped:
                        tip_pixels_dropped[cam] = far_px
            if len(tip_pixels_dropped) >= 2:
                alt_tip_pixels_dropped = {
                    c: px for c, px in alt_tip_pixels.items() if c in tip_pixels_dropped
                } or None
                result_dropped = score_dart(
                    tip_pixels_dropped, calibration, alt_tip_pixels=alt_tip_pixels_dropped
                )
                # Prefer the dropped-camera combination only when it is
                # BOTH usable and a real, measured improvement in ray
                # agreement over keeping every camera -- reusing
                # score_dart()'s own already-proven, already-tested
                # disagreement metric to arbitrate between two candidate
                # combinations, exactly the same pattern
                # `alt_tip_pixels`'s combination search above already
                # uses (never a new global threshold, which is the exact
                # lever the 2026-08-12 batch8 investigation
                # (opendarts.engines.apollo.scoring's own module
                # docstring) found regresses the correlated-bias probe --
                # this is a per-throw candidate comparison, not a
                # threshold change). The real incident this guards
                # (the recorded T15 throw): keeping cam2 disagrees
                # 7.55mm; dropping it disagrees 1.15mm -- a decisive,
                # unambiguous improvement, not a knife-edge call.
                dropped_is_better = result_dropped.ok and (
                    not result.ok
                    or (
                        result.max_ray_disagreement_mm is not None
                        and result_dropped.max_ray_disagreement_mm is not None
                        and result_dropped.max_ray_disagreement_mm < result.max_ray_disagreement_mm
                    )
                )
                if dropped_is_better:
                    result = result_dropped

        # 2026-08-18 -- last-resort lone-genuine-camera fallback. Real
        # incident that surfaced this: the recorded S20 throw
        # (AD truth 20/single_inner) -- cam2 had no usable candidate,
        # leaving only cam1 (genuine) and cam0's far-end recovery,
        # disagreeing 12.3mm, just over MAX_RAY_DISAGREEMENT_MM, with no
        # 3rd camera for score_dart()'s own RANSAC 2-of-3 pair fallback
        # to even attempt (that fallback structurally requires >=3
        # cameras -- see scoring.py's own `len(full_cams) >= 3` gate).
        # Talos/Athena all scored this throw correctly.
        #
        # Investigated via the REAL production replay path
        # (`opendarts.capture.replay.replay_throw_with_engine`, which
        # supplies `prior_dart_line_px` from stored visit metadata --
        # docs/DESIGN.md's "Replay is the source of truth": a bare `.score()` call with
        # no prior-dart context is NOT faithful to what live capture
        # actually does and undercounts what the existing 2026-08-17
        # prior-dart-drop-path backfill already recovers) on the full
        # living clean/ corpus (1107 packages, 2026-08-18): Apollo had
        # 6 real no-score throws. This fallback fires when exactly ONE
        # genuine (gate-passing) camera survived ROI-gating -- the
        # <=1-camera case this project explicitly leaves to
        # the CALLER (this IS that caller) -- trusting that lone camera's
        # own ray∩Z=0 (`_single_ray_board_xy`), gated on
        # `_lone_camera_diagnostics_clean()`. Recovers 5 of the 6, ZERO
        # wrong:
        #
        # throw_1786665050832 truth outside -> outside MATCH
        # S20 throw truth 20/outer -> 20/outer MATCH
        # S1 throw truth 1/outer -> 1/outer MATCH
        # S15 throw truth 15/outer -> 15/outer MATCH
        # S20 throw truth 20/inner -> 20/inner MATCH
        #
        # (057-S20 is a DIFFERENT existing rejection path than the other
        # four -- the "genuine-anchored recovery arbitration" block above
        # it already rejected honestly because neither genuine+recovery
        # PAIR agreed well enough; this fallback is the first thing in
        # this method to try the lone genuine camera completely BY
        # ITSELF, a different, weaker-but-still-real observation than any
        # pairing attempted above.)
        #
        # The one throw this fallback does NOT touch
        # (the recorded outside throw, truth outside) has ZERO genuine
        # cameras -- literally no signal to use -- and correctly stays an
        # honest no-score, exactly as before this change.
        #
        # **A second strategy was investigated and deliberately NOT
        # shipped**: exactly TWO genuine cameras disagreeing, backfilled
        # with a 3rd ROI-rejected camera's far-end-recovery pixel and
        # re-run through this same unmodified `score_dart()`. The two
        # real throws that originally looked like they needed it
        # (069-S10, 077-D20) turned out, once measured through the real
        # replay path instead of a bare `.score()` call, to ALREADY be
        # recovered by the existing 2026-08-17 prior-dart-drop-path
        # backfill -- so this corpus currently has ZERO real throws to
        # validate a 2-genuine-camera strategy against. Per this task's
        # own "measure the real number, don't ship on reasoning alone"
        # discipline, that is grounds not to ship it, not a reason to
        # ship it anyway on architectural soundness alone -- ship it
        # (and gate/measure it for real) the day a real corpus throw
        # actually exercises this shape.
        #
        # This fallback's own safety gate (`_lone_camera_diagnostics_clean()`):
        # measured separately, corpus-wide, not just on these 5 throws
        # -- a genuine camera's
        # own ray∩Z=0, taken completely alone, independently matches
        # operator/AD truth 93.2% of the time when clean per this gate
        # (n=3116 real per-camera detections across the whole corpus) vs
        # 76.2% when it is not (n=80). All 5 real recoveries above are
        # clean per this gate. Honest caveat: n=5/6 real recoveries is a
        # small sample for a safety-critical fallback -- this is not a
        # claim a clean lone camera is never wrong, only that it is
        # measurably more trustworthy than a flagged one. This block only
        # ever replaces an otherwise-CERTAIN no-score (see the
        # `not result.ok` gate below) -- it can never downgrade an
        # already-accepted `result`, so it carries zero regression risk
        # to any currently-correct throw.
        if not result.ok and len(genuine_cams) == 1:
            cam = next(iter(genuine_cams))
            diag = tip_diagnostics.get(cam, {})
            if _lone_camera_diagnostics_clean(diag):
                xy = _single_ray_board_xy(tip_pixels[cam], calibration[cam])
                if xy is not None:
                    sector, ring = sector_ring_for_point(xy[0], xy[1])
                    result = ScoreResult(
                        ok=True,
                        sector=sector,
                        ring=ring,
                        board_xy_mm=xy,
                        triangulation=None,
                        n_cameras_used=1,
                        cameras_used=(cam,),
                        # A single ray has no second ray to disagree
                        # with -- there is no real "disagreement" figure
                        # to report. Rather than leave this None
                        # (score_result_to_engine_result()'s own
                        # compute_confidence() call asserts non-None
                        # whenever ok=True -- every other accepted
                        # Apollo result always has one) or invent a
                        # falsely-reassuring 0.0mm, stamp it at the
                        # existing reject threshold itself: the most
                        # conservative value confidence.py's existing
                        # [0, DISAGREE_CAP_MM] scale can express, honestly
                        # saying "treat this as having zero cross-camera
                        # corroboration," not "this camera excluded some
                        # other camera as an outlier" (which
                        # outlier_camera would falsely claim --
                        # deliberately left None).
                        max_ray_disagreement_mm=MAX_RAY_DISAGREEMENT_MM,
                        reason=(
                            f"lone-camera Z=0 fallback: camera {cam} was the "
                            "only genuine (gate-passing) tip detection on this "
                            "throw, with a clean per-camera signature (no "
                            "off-axis/island/prior-dart red flags) -- using its "
                            "own ray∩Z=0 rather than an honest no-score (see "
                            "the dated 2026-08-18 comment in "
                            "opendarts/engines/apollo/engine.py)"
                        ),
                    )

        # 2026-08-18, tier 2 -- zero-genuine-camera fallback (EXPLORATORY,
        # validated against exactly ONE real corpus throw so far, not the
        # broad corpus-wide measurement the tier-1 lone-camera fallback
        # above got). the project's own framing after seeing that one remaining
        # no-score throw (the recorded outside throw) recovered correctly
        # by every OTHER engine, via a mechanism Athena already ships
        # (`opendarts/engines/athena/engine.py`'s "ROI-gate fallback": when
        # NOTHING passes the real gate, trust the ungated candidates
        # instead of reporting no score) -- and a real, important
        # correction to this project's own risk calculus: "a no-score is
        # worse than a wrong score since it can't help in the weighted
        # voting" (confirmed against Zeus's own code: `opendarts/engines/
        # zeus/engine.py` requires >=2 of 3 sub-engines ok=True to vote at
        # all; a Apollo no-score doesn't just lose Apollo's own vote,
        # it silently degrades Zeus's real 2-of-3 MAJORITY mechanism into
        # an arbitrary priority-order tie-break the one time Talos/
        # Athena disagree with each other).
        #
        # Real difference from Athena's own version of this idea:
        # Athena blends ALL ungated candidates it has, whatever they
        # say, with no requirement that they agree with each other --
        # measured historically to sometimes be wrong (its own module
        # comment: `throw_1786665050832`, a genuinely on-board AD dart
        # every candidate on every camera missed, ungated fallback
        # confidently said "outside"). This tier is deliberately
        # STRICTER: it requires at least 2 of the ungated (ROI-gate-
        # rejected) candidates' own ray∩Z=0 hits to independently AGREE
        # on the resulting (sector, ring) CLASSIFICATION -- not raw XY
        # distance. Real debugging finding while building this (measured,
        # not assumed): on the one real throw here, cam1's and cam2's
        # ungated ray∩Z=0 hits are 189.5mm APART in raw XY (they disagree
        # sharply on WHICH DIRECTION off-board the dart is) while each
        # individually landing ~267mm from bullseye -- comfortably past
        # DOUBLE_OUTER_RADIUS_MM (170mm) regardless of exact direction, so
        # both independently classify to the identical (None, "outside").
        # A raw-XY-agreement gate (the same MAX_RAY_DISAGREEMENT_MM bar
        # genuine on-board cameras must clear) is the WRONG bar for this
        # tier: it's calibrated for on-board precision, where direction
        # matters; two-thirds of a board length outside the double ring,
        # exact direction stops mattering to the SCORE even though it
        # still moves the raw XY a lot -- classification agreement is the
        # bar that actually reflects that. cam0 had no candidate at all on
        # this throw (detect_tip() itself failed), so this fires on
        # exactly 2 ungated candidates here, not 3; with 3 available this
        # requires a real 2-of-3 classification majority, not unanimity.
        #
        # Honest caveat, stated as plainly as tier 1's own: this fires on
        # exactly ONE known real corpus throw today (n=1, not the n=6
        # broader validation tier 1 got) -- there is no statistical
        # reliability number to report the way tier 1's 93.2%/76.2% gate
        # measurement exists, because the corpus does not currently
        # contain a second real throw with zero genuine cameras to check
        # this design against. Shipped anyway by design's own explicit
        # instruction to build and test it against this specific case
        # ("its only for this one case so we can run it against this case
        # and see what comes up") -- this is deliberately the more
        # exploratory of the two tiers, not a claim of the same
        # confidence tier 1 earned through broader measurement. Never
        # touches an already-scored throw (only reachable when `not
        # result.ok` AND tier 1 above also didn't fire, i.e. zero genuine
        # cameras) and never touches the tier-1 case (`len(genuine_cams)
        # == 1`) -- the two tiers are mutually exclusive by construction.
        if not result.ok and len(genuine_cams) == 0 and len(ungated_tip_pixels) >= 2:
            ungated_votes: dict[int, tuple[tuple[float, float], tuple]] = {}
            for cam, px in ungated_tip_pixels.items():
                xy = _single_ray_board_xy(px, calibration[cam])
                if xy is not None:
                    ungated_votes[cam] = (xy, sector_ring_for_point(xy[0], xy[1]))
            vote_counts: dict[tuple, list[int]] = {}
            for cam, (_xy, bed) in ungated_votes.items():
                vote_counts.setdefault(bed, []).append(cam)
            best_bed = max(vote_counts, key=lambda b: len(vote_counts[b]), default=None)
            agreeing_cams = vote_counts.get(best_bed, []) if best_bed is not None else []
            if best_bed is not None and len(agreeing_cams) >= 2:
                sector, ring = best_bed
                xs = [ungated_votes[c][0][0] for c in agreeing_cams]
                ys = [ungated_votes[c][0][1] for c in agreeing_cams]
                xy = (float(np.mean(xs)), float(np.mean(ys)))
                result = ScoreResult(
                    ok=True,
                    sector=sector,
                    ring=ring,
                    board_xy_mm=xy,
                    triangulation=None,
                    n_cameras_used=len(agreeing_cams),
                    cameras_used=tuple(sorted(agreeing_cams)),
                    max_ray_disagreement_mm=MAX_RAY_DISAGREEMENT_MM,
                    reason=(
                        f"ungated classification-agreement fallback: {len(agreeing_cams)} "
                        f"camera(s) ({sorted(agreeing_cams)}) failed the real board-ROI "
                        f"gate, but their own ray∩Z=0 hits independently classify to the "
                        f"same (sector={sector}, ring={ring}) -- zero genuine (gate-"
                        "passing) cameras existed on this throw, so this is the last "
                        "tier before an honest no-score (see the dated 2026-08-18 tier-2 "
                        "comment in opendarts/engines/apollo/engine.py; exploratory, "
                        "validated against one real corpus throw so far)"
                    ),
                )

        # 2026-08-25 -- tier 3, unanimous off-board override (see
        # _unanimous_off_board_override()'s own docstring for the full
        # real-incident write-up and mechanics: real throw
        # the recorded outside throw, 3 genuine cameras whose full/2-of-3
        # triangulation disagreed too much to trust (183.1mm) even though
        # every camera's own individual ray independently lands outside
        # the double ring). Only reachable when tier 1/tier 2 above did
        # NOT already produce an ok=True result (both are gated on
        # `not result.ok`, so if either already fired, `result.ok` is
        # True here and this is a guaranteed no-op) and `result` is still
        # the specific "rays disagree" rejection from score_dart()'s own
        # gate.
        result = _unanimous_off_board_override(
            result, tip_pixels, alt_tip_pixels, genuine_cams, calibration
        )

        # 2026-08-18 -- per-camera-vote override (see _per_camera_vote_
        # override()'s own docstring for the mechanics). Real incidents,
        # all from the living clean/ corpus's own remaining Apollo
        # misses as of this date, all sharing the identical shape (the
        # fused triangulation crosses a nearby wire even though it has
        # LESS individual-camera support than a rival bed):
        # S1 throw AD 1/inner fused 1/treble (2 vs 0)
        # T16 throw AD 16/treble fused 16/outer (3 vs 0)
        # T15 throw AD 15/inner fused 10/inner (2 vs 1)
        # S7 throw AD 7/inner fused 7/treble (2 vs 1)
        # S10 throw AD 10/outer fused 6/outer (3 vs 0)
        # S2 throw AD 2/outer fused outside (2 vs 0)
        # D4 throw AD outside fused 4/double (2 vs 1)
        # T20 throw AD 20/inner fused 20/treble (2 vs 1)
        # throw_1786690609789 AD 16/treble fused 16/outer (2 vs 0)
        # throw_1786730427489 AD 10/inner fused 15/inner (2 vs 1)
        # throw_1786730562953 AD 12/outer fused 5/outer (2 vs 1)
        # Every one of these has AT LEAST ONE camera individually voting
        # (its own solo ray∩Z=0, no triangulation) for the SAME bed AD
        # confirmed, while the fused 3D triangulation -- despite clearing
        # MAX_RAY_DISAGREEMENT_MM, i.e. not obviously "disagreeing" by the
        # existing metric -- crosses into a neighboring bed with less (in
        # most cases zero) individual support. Deliberately NOT a blanket
        # majority vote (Talos's own "always-majority" experiment,
        # docs/DESIGN.md 2026-08-13 morning, measured net +8/-6 -- a real,
        # documented correlated-bias trap): this only overrides when the
        # rival bed's vote count STRICTLY EXCEEDS however many cameras
        # individually back the fused answer, so a genuine 2-vs-1 (or
        # better) disagreement is required, not merely "some other bed
        # got a vote too" -- a throw where the fused bed already has as
        # much individual support as any rival is left untouched. See
        # this task's own corpus-wide before/after measurement (recorded
        # in docs/DESIGN.md) for the real steal-rate check this
        # required before shipping.
        result = _per_camera_vote_override(
            result, tip_pixels, alt_tip_pixels, far_end_recoveries, calibration
        )

        # 2026-08-25 -- tier 4/5, the two new no-score fallback tiers this
        # task adds (see each function's own docstring above for the full
        # real-incident write-up and mechanics). Deliberately placed
        # AFTER `_per_camera_vote_override()` above, not interleaved with
        # it: that override's own per-camera-vote tie-break has no real
        # evidence to arbitrate a synthetic tier-4/5 answer built from
        # exactly the same kind of per-camera votes (a real, checked risk
        # -- on `056-S16` specifically, letting `_per_camera_vote_
        # override()` see tier 5's own output would hand it a straight
        # 1-vs-1 per-camera tie with zero individual support for the
        # fused bed, which its own tie-break logic would then resolve to
        # WHICHEVER camera happens to iterate first -- arbitrary, not
        # evidence, and would have silently undone the fix). Placing both
        # new tiers after this call means `_per_camera_vote_override()`
        # only ever sees `result` in the exact state it always has --
        # already not ok when these two run, so its own early `if not
        # result.ok: return result` guard means it never even reaches
        # tier 4/5's synthesized output.
        result = _sparse_camera_off_board_fallback(
            result, tip_pixels, alt_tip_pixels, ungated_tip_pixels,
            ungated_alt_tip_pixels, genuine_cams, calibration,
        )
        result, low_confidence_fallback_used = (
            _marginal_disagreement_low_confidence_fallback(result)
        )

        # 2026-08-25 -- tier 6, the true last-resort ALWAYS-ANSWERS
        # fallback (see `_last_resort_always_answer_fallback()`'s own
        # docstring above for the full mechanism and the real-incident
        # write-up, `025-OUT`). Placed last, strictly after every other
        # tier -- its own `if result.ok: return result, False` guard means
        # it is a guaranteed no-op unless every single tier above (1
        # through 5, plus the per-camera vote override) has already had
        # its chance and still left `result.ok` False. This is the
        # concrete mechanism behind the project's own "Apollo is the only
        # engine that returns fails to score... I don't want that
        # happening" -- after this call, `result.ok` is unconditionally
        # True.
        result, last_resort_fallback_used = _last_resort_always_answer_fallback(
            result, tip_pixels, alt_tip_pixels, ungated_tip_pixels,
            ungated_alt_tip_pixels, far_end_recoveries, tip_diagnostics, calibration,
        )

        engine_result = score_result_to_engine_result(result)
        # own_tip_line_px -- see this loop's own comment above for the
        # full reasoning; attached unconditionally (an empty dict when no
        # camera produced a usable line -- absent-vs-empty doesn't matter
        # here since the ONLY consumer, find_prior_dart_line_px()'s
        # cached_frames fast path, already treats "empty dict" and "key
        # missing" identically via a plain .get(cam) per-camera lookup).
        engine_result.diagnostics["own_tip_line_px"] = own_tip_line_px
        if low_confidence_fallback_used:
            # See `LOW_CONFIDENCE_FALLBACK_CONFIDENCE`'s own docstring:
            # `score_result_to_engine_result()`'s normal `compute_
            # confidence()` call is the wrong tool for a throw that
            # already failed both the primary AND 2-of-3-pair gates --
            # override its output with the fixed, deliberately low,
            # non-model-derived value instead, in both places a
            # consumer might read confidence from (the shared
            # `EngineResult.confidence` field and its `diagnostics`
            # mirror -- see `score_result_to_engine_result()`'s own
            # 2026-08-14 comment for why both are kept in sync).
            engine_result.confidence = LOW_CONFIDENCE_FALLBACK_CONFIDENCE
            engine_result.diagnostics["confidence"] = LOW_CONFIDENCE_FALLBACK_CONFIDENCE
            engine_result.diagnostics["low_confidence_fallback_tier"] = (
                "marginal_disagreement"
            )
        if last_resort_fallback_used:
            # Same override mechanism as the tier-5 block above, at the
            # even-lower `LAST_RESORT_FALLBACK_CONFIDENCE` (0.15) -- see
            # that constant's own docstring for the full derivation.
            # Mutually exclusive with `low_confidence_fallback_used`
            # above by construction (tier 6's own `if result.ok: return
            # result, False` guard means it can only fire when tier 5
            # did NOT already produce an accepted result), so this is
            # never reached in the same call as the branch above.
            engine_result.confidence = LAST_RESORT_FALLBACK_CONFIDENCE
            engine_result.diagnostics["confidence"] = LAST_RESORT_FALLBACK_CONFIDENCE
            engine_result.diagnostics["low_confidence_fallback_tier"] = (
                "last_resort_always_answer"
            )
        return engine_result


def engine_result_to_score_result(result: EngineResult) -> ScoreResult:
    """The full-fidelity INVERSE of `score_result_to_engine_result()`
    above -- added 2026-08-12 to make removing the direct-call bypass in
    `opendarts/live/capture_daemon.py` and `opendarts/capture/replay.py`
    possible without losing any field their pre-existing `ScoreResult`-
    shaped callers (`opendarts.capture.throw_package.save_throw_package()`'s
    on-disk schema, `opendarts.capture.rescore_all`'s comparison logic)
    depend on.

    Deliberately NOT the same function as `opendarts.engines.base.
    engine_result_to_score_result()` -- that one is a GENERIC, honestly
    lossy adapter for an ARBITRARY engine's `EngineResult` (no engine
    other than Apollo has a reason to populate `cameras_used`/
    `outlier_camera`/`alt_candidates_used`-shaped diagnostics, so that
    adapter correctly leaves them at their "not applicable" defaults).
    THIS function only ever receives an `EngineResult` that came from
    `ApolloEngine.score()` (i.e. `result.diagnostics` was built by
    `score_result_to_engine_result()` above), so it can unpack every
    field that function packed -- byte-identical round trip for
    everything `save_throw_package()`'s on-disk `result.json` schema
    actually persists (verified directly against `_score_result_to_dict()`
    in `opendarts/capture/throw_package.py`: `ok`/`sector`/`ring`/
    `board_xy_mm`/`n_cameras_used`/`reason`/`max_ray_disagreement_mm`/
    `cameras_used`/`outlier_camera`/`triangulation.{ok,point_xyz,
    board_plane_xy,plane_discrepancy_mm,per_ray_distance_mm,n_rays}` --
    every one of those is recovered here).

    The one HONEST gap, pre-existing and not introduced by this function:
    `score_result_to_engine_result()`'s own `diagnostics["triangulation"]`
    packing already does not carry `TriangulationResult.per_ray_depth_mm`/
    `all_positive_depth`/`condition_number`/`well_conditioned`/`reason` --
    those were never in `diagnostics` to begin with, so this function
    leaves them at honest defaults (`None`/`""`) on the reconstructed
    `TriangulationResult` rather than inventing values. This is a
    pre-existing lossiness one layer up (the forward-direction packing),
    not something introduced here, and it does not affect
    `save_throw_package()`'s on-disk schema, which never reads any of
    those five fields either (confirmed against `_score_result_to_dict()`
    directly, not assumed).
    """
    diag = result.diagnostics
    tri_dict = diag.get("triangulation")
    triangulation: TriangulationResult | None = None
    if tri_dict is not None:
        point_xyz = tri_dict.get("point_xyz")
        triangulation = TriangulationResult(
            ok=tri_dict.get("ok", False),
            point_xyz=None if point_xyz is None else np.array(point_xyz, dtype=np.float64),
            board_plane_xy=tri_dict.get("board_plane_xy"),
            plane_discrepancy_mm=tri_dict.get("plane_discrepancy_mm"),
            per_ray_distance_mm=tri_dict.get("per_ray_distance_mm"),
            per_ray_depth_mm=None, # not carried by diagnostics -- see docstring
            all_positive_depth=None,
            condition_number=None,
            well_conditioned=None,
            n_rays=tri_dict.get("n_rays", 0),
            reason="",
        )
    cameras_used = diag.get("cameras_used")
    alt_candidates_used = diag.get("alt_candidates_used")
    return ScoreResult(
        ok=result.ok,
        sector=result.sector,
        ring=result.ring,
        board_xy_mm=result.board_xy_mm,
        triangulation=triangulation,
        n_cameras_used=diag.get("n_cameras_used") or 0,
        reason=result.reason,
        max_ray_disagreement_mm=diag.get("max_ray_disagreement_mm"),
        cameras_used=tuple(cameras_used) if cameras_used is not None else None,
        outlier_camera=diag.get("outlier_camera"),
        alt_candidates_used=tuple(alt_candidates_used) if alt_candidates_used is not None else None,
    )
