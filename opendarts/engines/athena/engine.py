"""Athena -- a genuinely independent 2D-per-camera scoring design (see
docs/DESIGN.md's Athena task description and docs/ENGINES.md for the
Engine interface this implements). NOT a port of Apollo: no multi-ray
triangulation anywhere in this module. Per camera: find the dart's
board-crossing pixel (`opendarts.engines.athena.crossing_detection`),
back-project it into a 3D ray (the one genuinely universal, reused piece
of geometry -- `opendarts.triangulation.rays.back_project_ray`), and
intersect that ray with the board's Z=0 plane directly
(`opendarts.engines.athena.plane_intersect.ray_plane_intersect`) to get
THAT camera's own, fully independent (sector, ring, x_mm, y_mm) call.
Combine up to 3 such independent reads -- PLUS up to 3 pairwise
shaft-line intersection reads (`opendarts.engines.athena.shaft_lines`,
2026-08-15: each camera's detected shaft AXIS projected onto the board
plane, pairs of those lines intersected -- evidence immune to along-shaft
tip-localization error) -- with a
confidence-weighted consensus (see `_combine_reads()` below) designed
around a signal INDEPENDENT of cross-camera agreement -- naive majority
vote is exactly the mechanism that loses on "two cameras share a correlated bias and
outvote the one correct camera." A blended answer can additionally be
overridden by a mixed-evidence corroborated label
(`_corroborated_label_override()` below -- a point read and a
shaft-line intersection agreeing on one label, gated structurally so it
can never become the twice-rejected plain majority label vote).
"""
from __future__ import annotations

import numpy as np

from opendarts.engines.base import EngineResult
from opendarts.engines.athena.board_gate import point_in_board_roi, project_board_roi_polygon
from opendarts.engines.apollo.prior_dart_context import PriorDartLinePx
from opendarts.engines.athena.confidence import (
    calibrated_confidence,
    raw_confidence_score,
    weighted_spread_mm,
)
from opendarts.engines.athena.crossing_detection import PRECOMPUTE_REQUIREMENTS, detect_crossing
from opendarts.engines.athena.plane_intersect import ray_plane_intersect
from opendarts.engines.athena.shaft_lines import (
    MAX_INTERSECTION_RADIUS_MM,
    MIN_CROSSING_ANGLE_DEG,
    board_plane_line,
    intersect_board_lines,
    intersection_weight,
)
from opendarts.geometry.board import sector_ring_for_point
from opendarts.pipeline import CameraCalibration
from opendarts.triangulation.rays import back_project_ray

# See _camera_weight()'s own comment below for the full real-measurement
# history of what this weight formula was BEFORE this constant/signal --
# elongation_ratio, image-derived -- and why it was replaced.
_RAY_STEEPNESS_FLOOR = 0.01
_RAY_STEEPNESS_EXPONENT = 4.0

# Relative-area plausibility floor for candidate selection
# (`_select_best_candidate` below): among the candidates ADMITTED for one
# camera (ROI-passing on the strict path; everything on the fallback
# path), drop any whose diff-component area is below this fraction of the
# largest admitted candidate's area, BEFORE the steepest-ray pick.
#
# **Why (real miss, 2026-08-17, the recorded outside throw cam1)**:
# the steepest-ray re-ranking chooses AMONG dart hypotheses, but it has
# no notion of blob plausibility at all -- on that throw it picked a
# 19-pixel isolated diff speck sitting on the D4 bed (a reflection
# flicker, visually confirmed well separated from the dart) over the real
# 2528-pixel dart blob whose read was correct, because the speck's ray
# was marginally steeper (0.548 vs 0.517). A component two orders of
# magnitude smaller than a co-admitted component is not an alternative
# dart hypothesis; it is noise that happened to clear the absolute
# MIN_COMPONENT_AREA_PX floor (19 vs 18). Deliberately RELATIVE and
# deliberately restricted to co-ADMITTED candidates: a small blob that is
# the ONLY admitted candidate stays fully trusted (that exact shape --
# big blob ROI-rejected, small tip fragment admitted -- is the CORRECT
# read on other real throws in the same session, e.g.
# one throw's cam0, area 35, and another throw's cam1, area 86).
#
# Measured (full real 885-throw
# data/archive/clean/ corpus, stored per-package calibration,
# operator/AD truth, this floor varied in isolation with the label
# override held at its pre-change rule):
#
# frac: 0.0 0.01 0.02 0.03 0.05 0.08 0.10 0.15 0.20
# total: 846 847 848 849 847 845 844 842 842
#
# Strictly monotone +1 per step up to the 849 peak at 0.03 with ZERO
# broken throws anywhere in [0.01, 0.03]; the first two losses appear at
# 0.05 and it decays from there. The per-throw flips explain the shape
# exactly (every decisive co-passing pair in the
# corpus): the junk specks whose removal wins sit at ratios
# 0.0075-0.021 (areas 19-127px against 900-3900px dart blobs), while the
# real small tip fragments whose removal loses sit at 0.046-0.068
# (areas 370-580px, split off 6700-10300px blobs by occlusion) -- a real
# ~2.2x empirical gap between the two populations. 0.03 shipped from the
# middle of that gap (~1.4x margin to the nearest observed member of
# either side), which is also the measured peak.
MIN_RELATIVE_AREA_FRACTION = 0.03

# See the "Hard steepness floor for CONSENSUS INPUT" comment in
# AthenaEngine.score() below for the real measurement behind this.
#
# **Lowered 0.45 -> 0.35, 2026-08-13.** Re-swept jointly with the
# tip_confidence weight factor (see _camera_weight() point 3) over the
# full real 349-throw data/archive/clean/ corpus: with that factor in
# place, every floor in [0.0, 0.35] scores IDENTICALLY (331/349) and
# 0.40/0.45 score strictly worse (327/326). The floor's original job --
# stop a grazing camera's noisy read from pulling the blend -- is now
# done better, and continuously, by the weight itself, and at 0.45 it was
# additionally throwing away good reads (several real misses had the one
# CORRECT camera excluded at steepness 0.41-0.45 while a 50mm-wrong
# camera survived). Kept as a genuine guard rather than deleted: a truly
# grazing ray (steepness well below 0.35) really is untrustworthy, this
# rig's cameras just never produce one -- their observed range is
# ~0.35-0.68 -- so the value that matters is "below anything real here",
# not a fitted optimum.
MIN_RAY_STEEPNESS_FOR_CONSENSUS = 0.35

# Which Z plane each camera's ray is actually intersected with, in the
# board-centred world frame the calibration solves in (Z=0 is the plane
# through the four cardinal-wire calibration landmarks; negative = further
# INTO the board, away from the cameras).
#
# **This is NOT `crossing_detection.CROSSING_WALK_FRACTION` in disguise**
# -- that hypothesis (a fixed walk of N% of the shaft's own PIXEL span,
# in one camera's 2D image, before back-projecting) was measured and
# rejected four separate times and stays rejected. This is a different,
# strictly-3D quantity: one fixed WORLD-SPACE plane offset, shared by
# every camera, which each camera's own ray geometry then converts into
# its own board-plane displacement (roughly `z / tan(elevation)` along
# that camera's own azimuth). A fixed pixel walk cannot express that --
# it applies the same 2D displacement regardless of where in the frame
# the dart is or how oblique that camera's ray is.
#
# **Why it is nonzero -- measured, not assumed** (rerunnable):
# decomposing every used per-camera read
# minus AD's own `tip_xy_mm` into the component along THAT camera's own
# board-plane azimuth gives a consistently POSITIVE (= displaced toward
# the camera) median for all three cameras on both real corpora
# independently -- +0.71 / +2.12 / +1.39 mm (cams 0/1/2, the 180-throw
# 2026-08-13 session) and +1.73 / +1.72 / +2.01 mm (cams 0/1/2, the 169
# 2026-08-12 throws). Six independent camera x corpus estimates, same
# sign every time. That is the exact signature of intersecting at the
# wrong plane, and the implied plane offset each one back-solves to
# (bias * tan(elevation), elevations ~28-30 degrees here) clusters at
# roughly -1mm. Physically unsurprising: the calibration landmarks sit on
# the wire/board face, while a stuck dart's real crossing point is in the
# sisal a millimetre or two behind it.
#
# **Real measured value, and how it was chosen**: swept jointly with
# RADIAL_CORRECTION_MM over the full real 349-throw data/archive/clean/
# corpus and, critically, LEAVE-ONE-SESSION-OUT cross-validated across
# all 8 real sessions: every one of the 8 folds picked
# z=-1.50 from the other seven sessions, and LOSO held-out total
# (322/349 = 92.3%) sits 1 throw below the in-sample optimum
# (323/349 = 92.6%) -- i.e. essentially no overfit, against a shipped
# baseline of 306/349 = 87.7%. The surrounding grid is a broad flat
# plateau (z in [-1.5, -0.75] all within 3 throws of the peak), not a
# spike.
#
# **Re-swept -1.50 -> -0.75, 2026-08-15, because the CALIBRATION under the
# engine changed, not the engine.** Every number above was fitted against
# per-throw stored calibrations from the pre-wire-junction calibration
# pipeline; the 2026-08-14/15 calibration work (local_refine wire-junction
# landmark refinement + average_correspondences(trim=True)) relocated the
# solved board plane, and this constant silently kept correcting for an
# offset that no longer existed at full size -- measurable as a consistent
# +0.9mm median RADIAL overshoot vs AD across all four sessions of the
# 420-throw corpus under the fresh calibration (residual
# dump), and as this engine alone falling behind Apollo/Talos (which
# bake no such plane constant) in the 2026-08-15 all-engines table.
# Joint grid with RADIAL_CORRECTION_MM under the fresh per-session
# calibration, shaft-line intersections + corroboration override active
# z=-0.75 row is the interior optimum (410 at
# rad 0.0-0.75 vs 406 at the old (-1.5, 1.0) operating point), and
# leave-one-session-out across all 4 sessions picks z=-0.75 in ALL FOUR
# folds independently, held-out total 408/420 >= the old point's
# in-sample total. The physical reading stays the same as before (the
# dart's crossing sits in the sisal behind the landmark plane) -- only
# the size of the residual offset changed because the fresh calibration
# solves the landmark plane more accurately.
BOARD_PLANE_Z_MM = -0.75

# A single global radial correction applied to every per-camera board-
# plane read (outward positive), before sector/ring lookup.
#
# **Two real, separately-measured effects used to add up here.** Only
# the second one is still this constant's job:
#
# 1. `opendarts.geometry.board`'s inner ring boundaries sat ~1.2-2.2mm
# OUTSIDE the ones this rig's real single->treble / single->double
# transition actually falls on. Measured with this engine completely
# out of the loop (AD's OWN `tip_xy_mm` through our own
# `sector_ring_for_point()` vs AD's OWN label: 6/349 real throws
# disagreed, all the same shape). **That is now fixed where it
# belongs**, in `board.INNER_RING_SCORING_OFFSET_MM` (1.5mm, LOSO
# cross-validated, see its own comment) -- shared by every engine
# instead of re-derived inside this one.
# 2. This engine's own reads come in slightly SHORT in radius --
# per-camera median (read radius - AD radius) of -0.4 to -1.8mm across
# all six camera x corpus combinations.
#
# **Re-swept 1.75 -> 0.50 when board.py absorbed effect 1** (2026-08-13),
# rather than left as-is: leaving 1.75 in place on top of the new
# boundary double-counts the same ~1.5mm and measured strictly worse
# (real numbers below). The re-sweep is a full engine re-run per
# candidate over all 349 real throws of `data/archive/clean/`, with the
# board offset held at its shipped 1.5mm:
#
# radial: 0.00 0.25 0.50 0.75 1.75(old)
# production calib 327 328 329 327 325
# 50-frame calib 322 324 323 321 317
# package calib 305 305 305 - 305
#
# LOSO across all 8 real sessions, board offset fixed at 1.5: 7 of 8
# folds pick 0.50 on the production-calibration condition, held-out
# 328/349 vs in-sample 329/349 -- essentially no overfit. Honest note on
# the tie: 0.25 and 0.50 total identically across the two derived
# conditions (652 both) and exactly tie on package calibration, and the
# whole [0.00, 0.75] range is a flat +-2-throw plateau; 0.50 is shipped
# because it is what LOSO picks on the best-scoring condition, not
# because the corpus separates it from 0.25.
#
# Applied per-read rather than post-consensus purely so the per-camera
# diagnostics stay in the same frame as the final answer -- previously
# measured byte-identical accuracy either way. (Worth knowing, measured
# during this same re-sweep: moving ~1.5mm of the correction OUT of the
# per-read step and into the shared post-consensus boundary is itself
# worth ~+3-4 throws -- (radial 0.25, offset 1.5) scores 324 where
# (radial 1.75, offset 0) scores 320 on the 50-frame condition, same
# total displacement. A per-read radial shift moves the weighted
# geometric median differently than shifting the boundary once, after
# the blend.)
#
# Note this replaces nothing: an earlier Athena session tried an
# ADDITIVE post-consensus radial correction and measured every value
# worse than zero. That was against the UNCORRECTED board plane, where
# the dominant error was the azimuthal displacement above, which a purely
# radial term cannot fix; with the plane offset applied first, the
# residual really is radial and this term now measures strictly better.
#
# **Re-swept 0.50 -> 1.00, 2026-08-14, on the corpus that actually exists
# now.** The 0.50 above was fitted against a 349-throw corpus made of 180
# fresh 2026-08-13 throws plus 169 historical 2026-08-12 throws; those 169
# have since been retired out of `data/archive/clean/` (see that
# directory's README) and a fresh 120-throw session added, so the corpus
# the constant was fitted to no longer exists. Re-swept against the real
# 300 throws now on disk (full-engine re-run per candidate,
# everything else held at its shipped value):
#
# radial: 0.50 0.60 0.70 0.80 0.90 1.00 1.10 1.20 1.30 1.50
# BOTH: 276 276 279 279 279 282 281 281 280 280
#
# Broad plateau 1.00-1.30, peak at 1.00, and 0.50 now sits below it.
# **Leave-one-calibration-epoch-out cross-validated over all 4 independent
# epochs** the corpus contains (the 180-throw "session" is really three
# 60-throw sittings, each with its own re-derived pose; plus the 120-throw
# session): all 4 folds independently pick 1.00 from their own training
# folds, and LOSO held-out total (282/300 = 94.0%) EQUALS the in-sample
# total -- zero measurable overfit. No fold regresses against 0.50
# (56/60 vs 55, 55/60 vs 54, 55/60 vs 55, 116/120 vs 112).
#
# Directly corroborated by the residual itself, not just the score: at
# radial=0.50 this engine's combined read still ran a median -0.76mm SHORT
# in radius against AD's own tip_xy_mm over the 300 throws -- i.e. effect
# 2 above was simply under-corrected once the corpus changed. The single
# largest miss family at 0.50 was six throws reading r~105.5-107.0 where
# AD had r~108.4-110.9, all landing just inside the treble outer wire.
#
# The board-geometry alternative was checked FIRST and rejected on the
# evidence: running AD's own tip_xy_mm through
# our own sector_ring_for_point reproduces AD's own ring label on 300/300
# throws, and all four scoring boundaries sit strictly inside their
# empirical label-transition brackets. board.py is right; the residual is
# this engine's own, so it belongs here.
#
# Re-verified at 1.00 that nothing else moved: BOARD_PLANE_Z_MM (-1.5 is
# still the joint 2D optimum), MIN_RAY_STEEPNESS_FOR_CONSENSUS,
# _RAY_STEEPNESS_EXPONENT (4 is still the exact peak),
# _TIP_CONFIDENCE_EXPONENT and board_gate.BOARD_ROI_RADIUS_PAD_FACTOR all
# re-measure at or inside their own plateaus on this corpus.
#
# **Re-swept 1.00 -> 0.25, 2026-08-15, same reason and same joint sweep as
# BOARD_PLANE_Z_MM's own -1.50 -> -0.75 entry directly above (read that
# first): the fresh wire-junction calibration removed most of the radial
# shortfall this constant was compensating, leaving the engine ~+0.9mm
# LONG in radius (median, all four sessions independently) -- the miss
# census showed the signature directly, e.g. three treble throws all read
# at r=107.2, just outside the 107mm treble-outer wire. Joint grid
# at z=-0.75 the radial plateau is 0.0-0.75 (all
# 409-410); LOSO folds pick 0.0/0.25/0.25/0.75; 0.25 shipped from the
# middle of the fold-picked range, with every session at or above BOTH
# its pre-change baseline and the intermediate shaft-lines-only state at
# this exact cell.**
RADIAL_CORRECTION_MM = 0.25


def apply_radial_correction(x_mm: float, y_mm: float) -> tuple[float, float]:
    """Push a board-plane point outward by RADIAL_CORRECTION_MM (see that
    constant's own comment for the two real measured effects it covers).
    A point exactly at the bullseye centre has no defined radial
    direction and is returned unchanged."""
    r = (x_mm * x_mm + y_mm * y_mm) ** 0.5
    if r < 1e-9 or RADIAL_CORRECTION_MM == 0.0:
        return x_mm, y_mm
    k = (r + RADIAL_CORRECTION_MM) / r
    return x_mm * k, y_mm * k


# Prior-dart-in-visit contamination guard, 2026-08-24 -- see
# `AthenaEngine.score()`'s own dated comment (below) for the full real
# incident this addresses (the recorded S10 throw) and the
# corpus-validation numbers behind these two constants.
#
# **`PRIOR_DART_LINE_MAX_PERP_PX` is DELIBERATELY the same 15.0 value
# as Apollo's own `opendarts.engines.apollo.tip_detection.
# PRIOR_DART_LINE_MAX_PERP_PX`, not independently re-derived.** It is the
# same physical quantity -- the perpendicular pixel distance from a
# candidate's own two ends to the infinite line defined by the
# immediately-prior throw's own two ends, on the SAME camera rig, at the
# SAME detection scale -- computed through this engine's own board-
# crossing candidates rather than Apollo's tip detector. This engine's
# own real corpus (374 throws total, the session corpus,
# 2026-08-24) has too few real prior-dart-in-visit throws to
# independently re-sweep a threshold from scratch and get a real,
# non-noisy number out of it (see the dated `score()` comment for the
# exact count) -- reusing the one number that IS real and separately
# cross-validated on this exact rig is more honest than shipping a
# guessed round number with no corpus behind it at all.
PRIOR_DART_LINE_MAX_PERP_PX = 15.0

# Down-weight, never exclude, a candidate flagged as prior-dart-
# suspected -- see `score()`'s own dated comment for why an outright
# admission-time exclusion was considered and rejected (measured to
# replace a wrong on-board answer with a different, not-better, wrong
# off-board one on the one real incident this guards, and carries real
# regression risk on any single-camera throw generally, per Apollo's
# own already-measured ~6.8% false-positive rate on this exact signal --
# a normal, common darts outcome, a new dart landing legitimately close
# to an old one, trips the same geometric proximity test as genuine
# contamination). A multiplicative factor in `_camera_weight()`'s
# existing product -- same shape as the `tip_confidence` factor already
# there -- lets a flagged candidate still be used when it is the only
# read available (this engine's own `_combine_reads()` is weight-
# INVARIANT at n=1, so no penalty value can change that case, only
# diagnostics can flag it -- see `score()`'s per-camera diagnostics).
#
# **Real swept value** (the full living 374-throw
# session corpus, restricted to the 36 throws
# where >=1 USED camera is actually flagged -- geometrically the only
# throws any penalty value CAN affect, confirmed by first running the
# full 374-throw corpus at the shipped value and finding zero throws
# outside this set changed at all): an aggressive first guess (0.05) hit
# a REAL regression -- the recorded S7 throw -- a legitimate,
# high-area (12916px), high-confidence (0.809) cam2 detection matching
# AD to ~2.75mm got flagged (a real false positive of the same kind
# Apollo's own guard already documents) and its weight cut ~20x, which
# let a genuinely tiny 47px noise-speck candidate on cam1 (weight 0.023,
# tip_confidence 0.097) outweigh it -- the exact "noise speck outvotes a
# real detection" failure shape `RULE_2A_MIN_OPPOSING_AREA_PX` and
# `RULE4_LIGHT_CAM_MIN_AREA_PX` above already guard against elsewhere in
# this file, reached here by a different path. Swept 0.0-1.0 in steps of
# 0.1 (plus 0.02/0.05): the regression persists through 0.2, clears at
# 0.3, and the one real improvement this guard produces on the same
# corpus (the recorded S20 throw, a wrong `1/single_inner` corroborated-
# label-override flip to the correct `20/single_inner`) holds for the
# entire [0.3, 0.8] range -- a clean, flat plateau, 35/36 with ZERO
# regressions anywhere in it. Above 0.8 the improvement itself stops
# firing (not enough discount left to change the corroboration weight
# comparison) and by 0.9 the guard is a no-op on this corpus. 0.5
# shipped, the middle of the measured plateau.
PRIOR_DART_WEIGHT_PENALTY = 0.5


def _perp_dist_px(
    pt: tuple[float, float], line_a: tuple[float, float], line_b: tuple[float, float],
) -> float:
    """Perpendicular distance from `pt` to the INFINITE line through
    `line_a`/`line_b` (not the line SEGMENT) -- same geometry Apollo's
    own prior-dart guard uses (see `PRIOR_DART_LINE_MAX_PERP_PX`'s own
    comment for why the two share one physical signal)."""
    ax, ay = line_a
    bx, by = line_b
    px, py = pt
    dx, dy = bx - ax, by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    cross = dx * (py - ay) - dy * (px - ax)
    return abs(cross) / length


def _prior_dart_contamination_suspected(
    picked: dict, prior_line: tuple[tuple[float, float], tuple[float, float]] | None,
) -> bool:
    """True only when BOTH of `picked`'s own two ends (`tip_px` and its
    `opposite_end`'s own `tip_px`) sit within `PRIOR_DART_LINE_MAX_PERP_PX`
    of the prior throw's own line -- deliberately requires both ends
    close, not either alone, mirroring Apollo's own guard: the literal
    signature of a component that is mostly/entirely a reappearance of
    the prior dart's own shaft (both its near AND far extent line up
    with where it already was), which a genuinely new, different dart
    merely landing nearby would not by chance also do. `False` whenever
    there is nothing to compare (no prior line for this camera, or this
    candidate's own opposite end is missing -- see
    `crossing_detection._analyze_component()`'s own note on when that
    can happen)."""
    if prior_line is None:
        return False
    tip = picked.get("tip_px")
    opposite_end = picked.get("opposite_end")
    opp_tip = opposite_end.get("tip_px") if opposite_end else None
    if tip is None or opp_tip is None:
        return False
    line_a, line_b = prior_line
    return (
        _perp_dist_px(tip, line_a, line_b) <= PRIOR_DART_LINE_MAX_PERP_PX
        and _perp_dist_px(opp_tip, line_a, line_b) <= PRIOR_DART_LINE_MAX_PERP_PX
    )


_TIP_CONFIDENCE_EXPONENT = 0.2
_TIP_CONFIDENCE_FLOOR = 0.05

# See the fourth override rule in `_corroborated_label_override()` below
# for the real measurement behind this -- a real absolute-area floor on
# the LIGHTER camera's own detected blob before its corroborated-with-an-
# intersection label is allowed to override the heavier camera's own
# point.
#
# **Real swept value** (full living clean/
# corpus, n=1107, subprocess-per-session, everything else held at its
# shipped value): 0 -> 1081 (baseline), 300/500 -> 1083, then a flat
# plateau 1084 for every threshold in [800, 3000], falling back to 1083
# at 3500 and all the way back down to the 1081 baseline at >=4000 (the
# rule stops firing on any real corpus throw). The two real miss
# populations this floor separates: the three throws it's meant to keep
# firing on have light-camera areas 3377/3692/3766px (real, substantial
# dart blobs); the three it's meant to silence have 24/30/767px (two
# are clear noise-speck detections dwarfed by their own camera's OTHER
# candidate, the third is a real but much smaller/weaker blob that
# happened to be wrong). 2000 shipped from the middle of both the swept
# plateau and this empirical gap.
RULE4_LIGHT_CAM_MIN_AREA_PX = 2000.0

# See gate (a)'s own comment inside `_corroborated_label_override()`
# below for the real regression this floor fixes.
#
# **Real swept value** (same subprocess-per-
# session pattern as RULE4_LIGHT_CAM_MIN_AREA_PX above, full living
# clean/ corpus, n=1107): 0 -> 1085 (the corpus total with rule 4 and
# the outside-only gate2-rule tightening already applied, before this
# floor), then a flat plateau at 1086 for EVERY threshold tried, 100
# through 2000 -- i.e. this specific regression's own noise-speck areas
# (21-90px) sit nowhere near the real validated opposition case's own
# areas (the live outside throw this gate's own test
# covers: opposing areas 700px and 1862px), so there is no cliff to find
# in this range, only a floor to clear. 500 shipped -- comfortably below
# the real 700px validated case, comfortably above the 21-90px noise
# specks that caused the regression.
RULE_2A_MIN_OPPOSING_AREA_PX = 500.0


def _camera_weight(ray_steepness: float, tip_confidence: float | None = None) -> float:
    """A single camera's own trust weight for the consensus blend below
    -- built ONLY from that camera's own self-diagnostics/geometry (never
    from how well it agrees with another camera's read), per this
    module's own docstring for what a safe camera-trust signal needs to
    be.

    **Real measurement history, both real signals tried, honestly kept
    (not silently swapped)**:

    1. **First shipped signal: `elongation_ratio`** (how side-on vs.
       foreshortened/compact this camera's view of the shaft silhouette
       is) -- the physically-motivated starting point docs/DESIGN.md's
       Athena task description itself suggested, plus `tip_confidence`
       blended in (alpha=0.7) -- got the whole pipeline to 82.2% on
       data/archive/clean/. But a targeted follow-up check
       (all 86 real 3-camera-used
       throws, comparing each camera's own weight rank against its real
       AD-referenced accuracy rank) found something the accuracy number
       alone was hiding: **the highest-elongation camera was the MOST
       accurate only 23.3% of the time (n=86) -- WORSE than the 33.3%
       chance baseline** (and the LOWEST-elongation camera was most
       accurate exactly half the time). Elongation of the dart's own
       silhouette measures something real about how well-DETERMINED that
       camera's shaft axis is, but that turns out to be close to
       unrelated to (if anything, mildly anti-correlated with) how
       accurate the resulting BOARD-PLANE point actually is -- a
       different question entirely.
    2. **Replacement, purely geometric, no image analysis needed at
       all**: `ray_steepness` = `abs(ray.direction[2])` (the
       back-projected ray's own direction cosine against the board's Z
       axis) -- how close to PERPENDICULAR-to-the-board (vs. grazing/
       near-parallel) this camera's ray is at the crossing pixel. Basic
       ray-plane intersection sensitivity: a grazing ray (small
       |direction.z|) amplifies ordinary pixel-level detection noise into
       a LARGE board-plane displacement (the ray travels a long distance
       in X/Y for a small change in Z); a steep ray is far less
       sensitive. Checked the identical way as elongation above: the
       steepest-ray camera is the
       most accurate **62.8%** of the time (n=86, vs 33.3% chance) --
       genuinely predictive, not just plausible-sounding -- and a real
       negative Pearson correlation (r=-0.343, n=410 individual used
       reads) between steepness and AD-referenced position error.
       `weight = (floor + steepness) ** exponent`, exponent swept 1-10:
       flat plateau 140/169 (82.8%) for exponent in [3, 10] (vs 135/169
       at exponent=1) -- 4.0 shipped, middle of the plateau. Blending
       tip_confidence back in on top of this (multiplicative, alpha
       0.0-1.0) measured NO further improvement (stayed at 140 or
       dropped to 138) -- left out of the shipped formula, reported
       honestly rather than kept for its own sake.

    Net real, measured, reproducible improvement from switching signals
    (same board ROI gate, same Weiszfeld consensus, only the weight
    SIGNAL changed): 82.2% -> **82.8%** on data/archive/clean/.

    3. **`tip_confidence` added back, 2026-08-13, as a mild multiplicative
       factor -- and this time it measures.** Point 1's "no further
       improvement" verdict was honestly reported and honestly true at
       the time; it was measured against the UNCORRECTED board plane,
       where a systematic ~1mm plane offset (see BOARD_PLANE_Z_MM) was a
       big share of the error budget and no per-camera trust signal could
       do much about it. With that offset removed, what remains is
       dominated by a minority of GROSS reads, and tip_confidence
       separates those sharply: for per-camera reads landing >15mm from
       AD's own tip_xy_mm the median tip_confidence is 0.301 (fresh
       180-throw session) / 0.211 (169 historical throws) against
       0.645 / 0.683 for the good reads -- a large separation, same
       direction on both corpora, measured independently of any
       threshold.
       `weight *= (tip_confidence + 0.05) ** exponent`, exponent swept
       0-1.0 against the full 349-throw corpus with the steepness floor
       removed: 0.0 -> 326, 0.10 -> 330, 0.15/0.20/0.25 -> **331**,
       0.30-0.40 -> 330, 0.50 -> 329, 1.0 -> 327. 0.20 shipped, middle of
       the plateau. Leave-one-session-out cross-validated over all 8 real
       sessions: 7 of 8 folds pick 0.15 (one picks 0.10), LOSO held-out
       330/349 = 94.6% vs in-sample 331/349 = 94.8%. Deliberately MILD --
       a large exponent measures worse, so this is a nudge between
       otherwise-comparable cameras, not a second hard gate.
    """
    weight = (_RAY_STEEPNESS_FLOOR + max(0.0, ray_steepness)) ** _RAY_STEEPNESS_EXPONENT
    if tip_confidence is not None:
        confidence = max(0.0, float(tip_confidence)) + _TIP_CONFIDENCE_FLOOR
        weight *= confidence ** _TIP_CONFIDENCE_EXPONENT
    return weight


WEISZFELD_ITERATIONS = 50
WEISZFELD_MIN_DIST_MM = 1e-6


def _combine_reads(reads: list[dict]) -> dict:
    """Confidence-weighted blend of up to 3 independent per-camera 2D
    reads into one (x_mm, y_mm). Each read is
    {"cam", "x_mm", "y_mm", "sector", "ring", "weight", ...}.

    **Weighted geometric median (Weiszfeld's algorithm)**, not a plain
    weighted mean of (x_mm, y_mm) -- measured
    (real data/archive/clean/ corpus):
    plain weighted-mean 60.4%, unweighted mean 60.9%, single
    highest-weight camera alone 67.5%, plain (unweighted) componentwise
    median 69.2%, weighted geometric median **70.4%** -- the real winner.
    A plain
    mean lets one wrong-but-still-passed-the-ROI-gate camera pull the
    combined point a large, unbounded distance; the geometric median (the
    point minimizing the sum of weighted distances to every read) is
    naturally robust to exactly that the way a spatial median generally
    is, while STILL being continuous and weight-sensitive rather than a
    discrete vote over sector/ring labels -- a 2-of-3 majority vote over
    LABELS is exactly the correlated-bias-losing mechanism this project
    keeps warning about; this blends the continuous board-plane
    point instead, so one high-confidence, correct-but-outnumbered camera
    can still pull the combined point most of the way to its own answer
    rather than being outvoted 1-2.

    **Re-measured 2026-08-13, at the current (much more accurate)
    pipeline state, because the comparison above was made when the engine
    scored 70.4% and the error structure has changed a lot since**
    (full real 349-throw data/archive/clean/
    corpus, sector+ring BOTH-match, everything upstream identical):

        weighted geometric median (shipped) ... 331/349 <-- still wins
        geometric median + outlier trim ....... 329/349
        polar (radius + angle) median ......... 326/349
        componentwise median .................. 324/349
        single highest-weight camera .......... 322/349
        weighted mean ......................... 320/349
        plain mean ............................ 307/349

    Margins are far larger than noise, and the ordering is essentially
    the same one the original comparison found -- so this choice is not
    an artefact of the pipeline state it was originally made in.

    Label-space alternatives were measured here too and rejected on the
    numbers, not on principle: a unanimous-label
    override changes nothing (331), a plain 2-of-3 majority label vote
    gains a single throw (332), and a weight-weighted label vote LOSES
    two (329). A one-throw gain does not justify adopting the exact
    correlated-bias mechanism this blend exists to avoid.

    **EXACTLY TWO reads are a special case, 2026-08-14 -- and every
    comparison above pooled them with the 3-camera ones, which hid it.**
    The weighted geometric median is *degenerate* at n=2: it minimises
    `w1*|p-p1| + w2*|p-p2|`, which over the segment between the two points
    is minimised at whichever endpoint is heavier. So on a 2-camera throw
    the loop below is not a blend at all -- it is a hard pick of the
    steeper camera, and the other camera's read is discarded entirely.

    That is a mathematical property of the estimator, not a tuning
    question, and it was verified against this very function on the real
    corpus rather than argued from theory: on the 111 two-camera throws in
    `data/archive/clean/`, the shipped output lands within 0.01mm of the
    heavier camera's own point on **107 of 111**, while sitting a median
    8.97mm away from the lighter one. And the discarded camera is the
    *more accurate* of the two **34.2% of the time** -- so a third of
    2-camera throws throw away the better read for no gain.

    At n>=3 the geometric median is a genuine robust blend and is exactly
    what makes this engine resistant to one gross read, so it stays. Real
    measured split, same inputs, only the combiner changed:

        combiner n=1 n=2 n=3 total
        weiszfeld (was) 32/34 101/111 149/154 282/300
        weighted mean 32/34 107/111 140/154 279/300 <- better at
                                                                  n=2, much
                                                                  worse at n=3
        weighted mean @ n=2 only, weiszfeld @ n>=3 288/300

    i.e. neither combiner is globally better -- the n=2 and n>=3 cases
    genuinely want different estimators, which is why a global swap
    measures worse and the split does not. **Leave-one-calibration-epoch-
    out cross-validated over all 4 independent epochs in the corpus: all
    4 folds independently pick this hybrid, LOSO held-out total 288/300 =
    96.0% EQUALS the in-sample total (zero measurable overfit), and no
    fold regresses (57/60 vs 56, 56/60 vs 55, 57/60 vs 55, 118/120 vs
    116).** Alternatives for the n=2 slot were measured too, so the plain
    weighted mean is not just the lucky one of several: w^2 287,
    w^0.5 284, w^0.25 283, unweighted mean 281.

    The split is on camera COUNT -- a structural fact about the estimator
    -- not on a fitted threshold, so there is no constant here to go stale
    the way a swept value can.
    """
    if len(reads) == 1:
        return {"x_mm": reads[0]["x_mm"], "y_mm": reads[0]["y_mm"]}

    if len(reads) == 2:
        total = sum(r["weight"] for r in reads)
        if total <= 0:
            return {"x_mm": sum(r["x_mm"] for r in reads) / 2.0,
                    "y_mm": sum(r["y_mm"] for r in reads) / 2.0}
        return {
            "x_mm": sum(r["weight"] * r["x_mm"] for r in reads) / total,
            "y_mm": sum(r["weight"] * r["y_mm"] for r in reads) / total,
        }

    x_mm = sum(r["x_mm"] for r in reads) / len(reads)
    y_mm = sum(r["y_mm"] for r in reads) / len(reads)
    for _ in range(WEISZFELD_ITERATIONS):
        num_x = num_y = den = 0.0
        for r in reads:
            dist = ((r["x_mm"] - x_mm) ** 2 + (r["y_mm"] - y_mm) ** 2) ** 0.5
            dist = max(dist, WEISZFELD_MIN_DIST_MM)
            w = r["weight"] / dist
            num_x += w * r["x_mm"]
            num_y += w * r["y_mm"]
            den += w
        if den <= 0:
            break
        x_mm, y_mm = num_x / den, num_y / den
    return {"x_mm": x_mm, "y_mm": y_mm}


def _corroborated_label_override(
    consensus_reads: list[dict],
    combined: dict,
    sector,
    ring,
):
    """The second use of shaft-line evidence (see `shaft_lines`' module
    docstring for the first): a label corroborated by TWO STRUCTURALLY
    DIFFERENT kinds of evidence -- a camera's own point read AND a
    shaft-line intersection independently landing on the same (sector,
    ring) -- outranks a continuous blend that landed somewhere none of
    that corroboration supports. The check is an intersection's segment
    agreeing with a camera's own tip segment; it runs inside this
    engine's continuous weight machinery (max TOTAL WEIGHT per label
    decides which corroborated label wins, and the returned point is the
    weighted blend of that label's own supporting reads) rather than a
    fixed ordered ladder of rules.

    **Why this is not the twice-rejected majority label vote** (see
    `_combine_reads`' docstring for those real numbers): a 2-of-3 point
    vote lets two cameras sharing one correlated bias outvote the one
    correct camera -- same evidence kind, same failure mode, no new
    information. The mixed-kind gate here requires an INTERSECTION to
    corroborate, and an intersection is immune to along-shaft tip
    localization error by construction, so "a point read and an
    intersection agree" is two different error families agreeing -- real
    corroboration, not an echo.

    **Measured basis, real full-420-throw corpus, fresh wire-junction
    calibration**: among all throws where the max-weight label differed from
    the blend's label, every case where that label was RIGHT (5) had
    multi-read support -- 4 of 5 with mixed point+intersection kinds --
    while every case where it was WRONG (2) was a single lone point read.
    The mixed-kind requirement is a structural fact about which evidence
    kinds agree, not a fitted threshold -- same category as
    `_combine_reads`' n=2/n>=3 split, so there is no constant here to go
    stale. LOSO-validated end-to-end alongside the shaft-line constants
    (see shaft_lines.py + docs/DESIGN.md).

    **Second firing condition, 2026-08-17 -- specific, measured forms of
    "corroboration outranks a LONE read even when the lone read carries
    more weight."** Trigger case: one recorded throw (truth: D11) --
    cam2's lone point read said outside (r=170.15, 0.15mm past the
    double wire, weight 0.164) and out-weighed cam0's point + the
    cam0-cam2 shaft-line intersection BOTH independently reading
    11/double (total 0.071; note the intersection is built from cam2's
    own shaft LINE, i.e. cam2's own along-shaft-immune evidence
    contradicted cam2's own point). The first rule never fires there
    because the lone read also wins the total-weight comparison.

    The BLANKET form of this idea ("whenever the blend label is a lone
    read and any mixed-kind label exists, override") was built first and
    measured on the full real 885-throw corpus: it fires on 13 throws and is
    NON-SEPARABLE -- 6 wins, 6 losses, 1 neutral, including the same
    (10,treble)->(10,single_outer) transition appearing once as a win
    and once as a loss -- the same shape every count-based label-vote
    attempt hit (docs/DESIGN.md 2026-08-14/15). NOT shipped in that form. What
    the full 13-case probe table separates cleanly is two narrower,
    structural sub-gates, shipped here:

    (a) **Unanimous-opposition corroboration**: the corroborated label is
        supported by at least TWO point reads (i.e. every other camera's
        own independent point read -- the blend label's support being a
        single camera means at most two others exist) PLUS intersection
        evidence. In-corpus: fires 2, wins 2, losses 0 (both real: a
        genuinely-outside throw a lone wrong point dragged to 17/double,
        and a treble-wire ring flip). Every one of the 6 blanket-rule
        losses has exactly ONE point read behind the corroborated label,
        so this gate excludes all of them.
    (b) **Lone-OUTSIDE read vs corroborated ON-BOARD label**: the blend's
        lone read says ring="outside" while the corroborated label is a
        real bed. "Outside" is the unbounded no-hit region, not a
        positive bed claim -- and the probe shows a hard empirical
        asymmetry on the double-outer wire: overriding INTO outside on
        single-point corroboration lost every observed time (0 wins, 4
        losses -- all four were real doubles read 0.1-2mm long by one
        camera), while overriding OUT of a lone outside read into a
        corroborated bed won its observed case (076) and lost none.
        Gate (b) permits only the second direction; the first stays with
        the blend unless gate (a)'s stronger unanimous form applies.

    Both sub-gates are evidence-kind/coverage conditions, not fitted
    thresholds. Net measured effect of the two together, full 885-throw
    corpus, stored per-package calibration, operator/AD truth: +3
    (including 076) / -0, no session below its pre-change count -- see
    docs/DESIGN.md's dated entry for the full table.

    Snapped point: weighted blend (`_combine_reads`) of the winning
    label's own supporting reads; in the rare case that blend itself
    lands outside the label (two same-label reads' chord can dip across
    an inner ring boundary -- an annular sector is not convex), fall back
    to the highest-weight supporting read's own point, which by
    construction carries the label.
    """
    if len(consensus_reads) < 2:
        return combined, sector, ring
    label_info: dict[tuple, dict] = {}
    for r in consensus_reads:
        lab = (r["sector"], r["ring"])
        info = label_info.setdefault(lab, {"weight": 0.0, "reads": [], "kinds": set()})
        info["weight"] += r["weight"]
        info["reads"].append(r)
        info["kinds"].add(r.get("kind", "point"))

    def _is_corroborated(lab: tuple) -> bool:
        return {"point", "intersection"} <= label_info[lab]["kinds"]

    def _point_cams(lab: tuple) -> set:
        return {
            r["cam"] for r in label_info[lab]["reads"]
            if r.get("kind", "point") == "point"
        }

    # 2026-08-18 -- real area floor, used by the outside-only branch of
    # the second rule below (NOT by rule 3 -- see that rule's own comment
    # for why it was tried there too and reverted). Topological
    # independence (a different camera PAIR) is not the same thing as
    # REAL independent evidence: a shaft-line intersection built from a
    # noise-speck blob's own "axis" (a reflection/artifact with no real
    # dart shaft at all) is not meaningful evidence just because its
    # camera pair happens to exclude the blend's own point camera. Real
    # regression found (010-S20/083-S20, both AD 20/single_outer): a
    # correct, high-area (11-12k px), high-confidence camera lost to two
    # noise-speck detections (areas 21-90px) plus their own intersection,
    # purely because that intersection's camera pair happened to exclude
    # the correct camera. See `RULE_2A_MIN_OPPOSING_AREA_PX`'s own
    # comment for the real sweep behind the threshold.
    _point_area_by_cam = {
        r["cam"]: r.get("area") for r in consensus_reads
        if r.get("kind", "point") == "point"
    }

    def _cam_area_plausible(cam) -> bool:
        area = _point_area_by_cam.get(cam)
        return area is None or area >= RULE_2A_MIN_OPPOSING_AREA_PX

    def _ix_independent_and_plausible(r: dict, exclude_cams: set) -> bool:
        return (
            r.get("kind", "point") == "intersection"
            and not (set(r["cam"]) & exclude_cams)
            and all(_cam_area_plausible(c) for c in r["cam"])
        )

    blend_lab = (sector, ring)
    best_lab = max(label_info, key=lambda k: label_info[k]["weight"])

    target_lab = None

    # Third rule, 2026-08-17 -- PROVENANCE-INDEPENDENT unanimous
    # opposition. The dominant miss family in the 996-corpus fresh-eyes
    # review (15+ real throws, e.g. one recorded session's S20->S1
    # cluster): the steepness^4 weight concentration lets ONE heavy
    # camera's wrong point drag the blend, while BOTH other cameras'
    # point reads AND the intersection of those two cameras' own shaft
    # lines -- the only read in the whole consensus that shares no
    # provenance with the heavy camera -- agree on the correct label
    # (that independent intersection was also consistently the most
    # accurate single read in every one of these misses, 0.6-1.8mm from
    # AD's own tip). The existing rules never fire there because the
    # heavy camera's own intersections (ix pairs INVOLVING it) hand its
    # label enough fake support/weight to win both the lone-read check
    # and the max-total-weight comparison.
    #
    # Why this is STILL not the twice-rejected majority label vote: the
    # gate requires corroboration that is structurally independent of
    # every camera supporting the blend's label -- >=2 other cameras'
    # point reads PLUS an intersection whose camera pair EXCLUDES all
    # blend-label point cameras, all landing on the same label. Two
    # correlated point reads alone still cannot outvote the heavy camera
    # (that exact shape, e.g. throw_1786667985388's 2-wrong-1-right, and
    # the recorded S20 throw's points-only opposition, does NOT fire
    # this). On a 2-camera throw every intersection involves both
    # cameras, so this gate structurally cannot fire at all there.
    #
    # Checked before the max-weight rule below because the two can
    # disagree (real case the recorded D9 throw: max-weight picks the
    # heavy camera's outside label corroborated only by its own ix;
    # this rule picks 9/double backed by both other cameras + their own
    # independent ix -- the latter matches AD). Measured on the full
    # living clean/ corpus (996 AD-matched, 2026-08-17): see the dated
    # test module for the full gained/lost list.
    blend_point_cams = _point_cams(blend_lab) if blend_lab in label_info else set()
    if len(blend_point_cams) <= 1:
        independent = []
        for lab, info in label_info.items():
            if lab == blend_lab:
                continue
            if len(_point_cams(lab)) < 2:
                continue
            # Deliberately NOT using `_ix_independent_and_plausible`'s area
            # floor here (2026-08-18, measured and reverted): adding it
            # fixed 083-S20 but broke three throws from the original
            # 2026-08-17 gate-3 gains (046-S20, 106-S8, 044-D4) whose
            # opposing point read has a genuinely small area (31-423px --
            # a partially-occluded dart, not a noise speck) yet is still
            # CORRECT. Area alone cannot separate "small but real" from
            # "small and noise" -- net -2 (+1/-3) on the full living
            # clean/ corpus, reverted; the area floor stays scoped to
            # gate (a) and the outside-only branch below only, where it
            # measured cleanly.
            has_independent_ix = any(
                r for r in info["reads"]
                if r.get("kind", "point") == "intersection"
                and not (set(r["cam"]) & blend_point_cams)
            )
            if has_independent_ix:
                independent.append(lab)
        if independent:
            target_lab = max(independent, key=lambda k: label_info[k]["weight"])

    # Fourth rule, 2026-08-18 -- two-camera lone-line-disagreement
    # override. On a throw where only TWO cameras cleared the strict
    # gate, rules 2a/3 above can never fire: both require >=2 point reads
    # opposing the heavy camera's own label, but there is only ONE other
    # camera to begin with (rule 3's own docstring: "on a 2-camera throw
    # every intersection involves both cameras, so this gate structurally
    # cannot fire there at all"). Real miss census (2026-08-18, the 26
    # living-corpus Athena misses): a recurring shape on exactly these
    # 2-camera throws -- the heavier camera's own POINT read disagrees
    # with the LIGHTER camera's own point, and the one available
    # shaft-line intersection (built from BOTH cameras' lines, including
    # the heavy one) sides with the LIGHTER camera, not the heavy one.
    # That is the heavy camera's own shaft LINE contradicting its own
    # POINT -- structurally the same insight already used for gate (b)'s
    # lone-outside case and its own trigger throw (076) above, generalized
    # beyond "outside" specifically and to the 2-camera case rules 2a/3
    # cannot reach at all.
    #
    # Deliberately narrow: fires only when there are EXACTLY two point
    # reads and exactly one intersection (the natural shape of a
    # 2-camera-used throw), the blend already sits at the heavier
    # camera's own label (never touches a throw where the blend already
    # disagrees with its own heaviest camera for some other reason), and
    # the intersection's SECTOR matches the lighter camera's own sector
    # while DIFFERING from the heavy camera's own sector -- i.e. light
    # point + intersection both cross the heavy camera's own SECTOR wire.
    # Target label prefers the EXACT light+ix agreement (both reads
    # combine into the snapped point) when they agree on ring too, else
    # falls back to the intersection's own label alone -- this engine's
    # own most-accurate single evidence class when available (module
    # docstring: pairwise intersections median 2.24mm vs point reads
    # 2.81mm against AD on the full 420-throw corpus).
    #
    # **Deliberately does NOT extend to a same-sector, different-RING
    # disagreement** (light+ix crossing the heavy camera's own RING wire
    # instead of its sector wire) -- measured and NOT shipped in that
    # form (full living
    # clean/ corpus, n=1107, real diagnostics dump for every case): tried
    # first because it also fixes two real misses
    # (throw_1786666471662/throw_1786667731513, both light-camera areas
    # comfortably above the area floor below), but net measured NON-
    # SEPARABLE by area or any other per-read signal found -- 4 gained,
    # 5 lost (007-S2, 044-T10, 061-S15, 048-T20, throw_1786690408695, all
    # with substantial light-camera areas 2600-6300px, ruling out the
    # noise-blob explanation that separates the sector-crossing case
    # below). Physically plausible why: BOARD_PLANE_Z_MM/
    # RADIAL_CORRECTION_MM's own extensive measurement history already
    # establishes this rig's dominant per-camera error as RADIAL
    # (along-shaft), which is exactly the error family a shaft-line
    # intersection is immune to for the AZIMUTHAL/sector call -- but nothing
    # here shows the intersection's own RADIAL precision (needed for a RING
    # call) actually beats a real, confident point read. Kept OUT rather
    # than shipped on a 4-5 coin flip.
    #
    # **Also requires an absolute area floor on the LIGHTER camera's own
    # detected blob** (`RULE4_LIGHT_CAM_MIN_AREA_PX`, see that constant's
    # own comment for the real sweep) -- without it, two real corpus
    # throws (128-S10, throw_1786689883266) regress: the light camera's
    # own "detection" is a 24-30px noise speck (dwarfed by its OWN
    # camera's other candidate, areas 8400-16000px), yet its ray happens
    # to intersect the heavy camera's line at a plausible-looking board
    # point that is, in fact, just noise agreeing with noise.
    if target_lab is None and len(camera_reads_for_override := [
        r for r in consensus_reads if r.get("kind", "point") == "point"
    ]) == 2:
        ix_only = [r for r in consensus_reads if r.get("kind") == "intersection"]
        if len(ix_only) == 1:
            heavy, light = sorted(camera_reads_for_override, key=lambda r: -r["weight"])
            heavy_lab = (heavy["sector"], heavy["ring"])
            light_lab = (light["sector"], light["ring"])
            ix = ix_only[0]
            ix_lab = (ix["sector"], ix["ring"])
            light_area = light.get("area")
            if (
                blend_lab == heavy_lab
                and light["sector"] is not None
                and light["sector"] == ix["sector"]
                and light["sector"] != heavy["sector"]
                and light_area is not None
                and light_area >= RULE4_LIGHT_CAM_MIN_AREA_PX
            ):
                candidate_lab = light_lab if light_lab == ix_lab else ix_lab
                if candidate_lab != heavy_lab:
                    target_lab = candidate_lab

    if target_lab is not None:
        pass
    elif best_lab != blend_lab and _is_corroborated(best_lab):
        # Original rule: the max-total-weight label differs from the
        # blend's and carries mixed-kind corroboration.
        #
        # (2026-08-17, measured and NOT shipped UNIVERSALLY: requiring the
        # corroborating intersection here to be provenance-independent of
        # the label's own point cameras -- the third rule's own gate --
        # measured +2/-3 on the full 996-corpus: it fixed 033-D9 but broke
        # three throws whose correct override is corroborated only by a
        # self-involved intersection. Self-involved ix corroboration is
        # weaker but still real evidence there.)
        #
        # **Narrower form shipped 2026-08-18, scoped to outside only.**
        # 033-D9 itself: the raw blend already sits at the CORRECT
        # (9, double) -- both cam0 and cam2's own points, the two cameras
        # that are NOT the heavy one -- and this rule was overriding that
        # correct blend to "outside" purely because the heavy camera's
        # own point + its own self-involved ix(0,1) out-weighed them.
        # "Outside" is the unbounded no-hit region, not a positive bed
        # claim (the same asymmetry gate (b) above already established:
        # overriding INTO outside on single-point corroboration lost every
        # observed time, overriding OUT of a lone outside read won).
        # Scoping the provenance-independence requirement to ONLY
        # best_lab.ring == "outside" (never to a real bed label, where the
        # previously-measured +2/-3 regression lived) tests that same
        # asymmetry on this rule specifically. Measured on the full living
        # clean/ corpus (n=1107): see this
        # module's own test file for the dated real numbers.
        if best_lab[1] == "outside":
            # Uses the shared independence+plausibility check
            # (`_ix_independent_and_plausible`, see its own comment) --
            # 033-D9 itself is already fixed by topological independence
            # alone (its only candidate intersection shares a camera with
            # the correct blend, so it's excluded either way), so the
            # area floor is not exercised by any of the throws this rule
            # was built against; kept anyway as defense against the same
            # proven noise-speck-intersection failure mode gate (a) and
            # rule 3 both hit, since it measured ZERO difference (neither
            # gain nor loss) on the full living clean/ corpus -- a free
            # guard, not a speculative untested one.
            has_independent_ix = any(
                _ix_independent_and_plausible(r, _point_cams(blend_lab))
                for r in label_info[best_lab]["reads"]
            )
            if has_independent_ix:
                target_lab = best_lab
        else:
            target_lab = best_lab
    else:
        # Second rule, two structural sub-gates (see docstring: the
        # blanket lone-read form was measured non-separable and is NOT
        # what this implements). Preconditions shared by both: the
        # blend's own label is a lone single read (or unsupported by any
        # read), and a mixed-kind corroborated competing label exists.
        blend_info = label_info.get(blend_lab)
        if blend_info is None or len(blend_info["reads"]) <= 1:
            corroborated = [
                lab for lab in label_info
                if lab != blend_lab and _is_corroborated(lab)
            ]

            # Gate (a)'s plausible-point count, 2026-08-18 -- a real area
            # floor on the OPPOSING point reads themselves (not just a
            # raw camera count). Real regression found (010-S20, AD
            # 20/single_outer): a heavy, high-area, high-tip-confidence
            # camera (area 12044px) correctly read 20/single_outer, but
            # got outvoted by "unanimous opposition" from the OTHER two
            # cameras' own point reads at areas 21-86px -- clear noise
            # specks (reflections/artifacts), not real dart detections,
            # that happened to land on "outside" together with their own
            # intersection. Gate (a)'s original design (see its
            # docstring/tests) is legitimate when the opposition is REAL
            # (the validated 049-OUT shape, real dart blobs on both other
            # cameras, areas 700/1862px) -- this only excludes a point
            # read from the count when it HAS a real area value below the
            # floor; missing area data (e.g. every synthetic test read,
            # or the rare candidate with no area at all -- see
            # `_select_best_candidate`'s own note on synthesized
            # single-candidate fallbacks) is trusted unchanged, matching
            # this module's existing `MIN_RELATIVE_AREA_FRACTION` gate's
            # own convention.
            def _n_plausible_point_reads(lab: tuple) -> int:
                return sum(
                    1 for r in label_info[lab]["reads"]
                    if r.get("kind", "point") == "point"
                    and (r.get("area") is None or r["area"] >= RULE_2A_MIN_OPPOSING_AREA_PX)
                )

            blend_is_lone_outside = (
                blend_lab[1] == "outside"
                and blend_info is not None
                and len(blend_info["reads"]) == 1
                and blend_info["reads"][0].get("kind", "point") == "point"
            )
            eligible = [
                lab for lab in corroborated
                if _n_plausible_point_reads(lab) >= 2 # gate (a): unanimous opposition
                or (blend_is_lone_outside and lab[1] != "outside") # gate (b)
            ]
            if eligible:
                target_lab = max(eligible, key=lambda k: label_info[k]["weight"])
    if target_lab is None:
        return combined, sector, ring
    info = label_info[target_lab]

    snapped = _combine_reads(info["reads"])
    snap_sector, snap_ring = sector_ring_for_point(snapped["x_mm"], snapped["y_mm"])
    if (snap_sector, snap_ring) != target_lab:
        top = max(info["reads"], key=lambda r: r["weight"])
        snapped = {"x_mm": top["x_mm"], "y_mm": top["y_mm"]}
        snap_sector, snap_ring = target_lab
    snapped = dict(snapped)
    snapped["label_override"] = True
    return snapped, snap_sector, snap_ring


def _camera_ground_xy(calib: CameraCalibration) -> tuple[float, float]:
    """This camera's own position projected onto the board plane, in the
    same board-centred world frame everything else here uses."""
    import cv2

    rotation, _ = cv2.Rodrigues(np.asarray(calib.rvec, dtype=np.float64).reshape(3))
    translation = np.asarray(calib.tvec, dtype=np.float64).reshape(3, 1)
    centre = (-rotation.T @ translation).reshape(3)
    return float(centre[0]), float(centre[1])


def _select_best_candidate(
    candidates: list[dict],
    calib: CameraCalibration,
    cam: int,
    roi_polygon,
    require_roi: bool,
):
    """Walk one camera's candidate list (already end-picked/re-ranked by
    `_pick_tip_end`), keep the ones that pass the board ROI gate (or, when
    `require_roi=False`, every candidate regardless of the gate -- see
    `AthenaEngine.score()`'s ROI-FALLBACK PASS for when/why that's used),
    and return the steepest-ray one -- same selection rule either way, only
    the admission gate differs. Returns
    `(picked, ray, steepness, n_end_flips)`, with `picked=None` if nothing
    at all qualifies (only possible with `require_roi=True`, since
    `require_roi=False` admits every candidate).

    Between admission and the steepest-ray pick sits the relative-area
    plausibility floor (`MIN_RELATIVE_AREA_FRACTION` -- see its own
    comment for the real miss behind it): a co-admitted candidate dwarfed
    by another admitted candidate's diff-component area is dropped before
    steepness is even consulted. Candidates with no area recorded (the
    synthesized single-candidate fallback in `score()`) are never dropped
    by it -- there is nothing real to compare."""
    admitted: list[dict] = []
    n_end_flips = 0
    for cand in candidates:
        cand = _pick_tip_end(cand, calib, cam, roi_polygon)
        if cand.get("end_flipped"):
            n_end_flips += 1
        if require_roi and not point_in_board_roi(cand["crossing_px"], roi_polygon):
            continue
        admitted.append(cand)

    known_areas = [c["area"] for c in admitted if c.get("area")]
    if len(admitted) > 1 and known_areas:
        max_area = max(known_areas)
        admitted = [
            c for c in admitted
            if not c.get("area") or c["area"] >= MIN_RELATIVE_AREA_FRACTION * max_area
        ]

    best_cand = None
    best_cand_ray = None
    best_cand_steepness = -1.0
    for cand in admitted:
        cand_ray = back_project_ray(
            cand["crossing_px"], calib.camera_matrix, calib.dist_coeffs,
            calib.rvec, calib.tvec, cam=cam,
        )
        cand_direction = np.asarray(cand_ray.direction, dtype=np.float64)
        cand_steepness = float(abs(cand_direction[2]) / (np.linalg.norm(cand_direction) + 1e-12))
        if cand_steepness > best_cand_steepness:
            best_cand, best_cand_ray, best_cand_steepness = cand, cand_ray, cand_steepness
    return best_cand, best_cand_ray, best_cand_steepness, n_end_flips


def _pick_tip_end(cand: dict, calib: CameraCalibration, cam: int, roi_polygon) -> dict:
    """Decide which END of a detected dart blob is the TIP, using camera
    geometry that `crossing_detection.py` deliberately does not have.

    **The rule, and why it is right by construction rather than fitted**:
    a stuck dart's tip is IN the board (at the board plane), while its
    flight sticks out of the board toward the thrower, well above that
    plane. Back-project both ends and run each ray all the way down to
    the board plane: the tip's ray stops essentially where the tip really
    is, but the flight's ray keeps going past the flight and lands
    somewhere further out -- and "further out" is always AWAY from the
    camera, because the ray is travelling away from the camera as it
    descends. So the end whose board-plane intersection is CLOSER to the
    camera's own ground position is the tip. No threshold, no tuned
    constant: it is a comparison between two distances.

    **Real measured accuracy of this rule vs. the image-only extent rule
    it overrides** (every ROI-passing camera-view of
    the full real data/archive/clean/ corpus, restricted to views where
    AD's own tip_xy_mm makes the true end unambiguous -- the two ends'
    board points more than 8mm apart):

        2026-08-13 fresh session (n=488): extent 91.8% -> geometric 97.3%
        2026-08-12 historical (n=459): extent 94.8% -> geometric 96.9%

    Falls back to the extent choice whenever the opposite end is missing,
    either end fails to produce a valid board-plane crossing, or -- see
    below -- either end falls outside the board ROI.

    **Deliberately requires BOTH ends to pass the board ROI gate before
    it will flip anything**, so this never widens what the gate admits.
    Measured, real, all three orderings run end to end over the full
    corpus:

        no flipping at all fresh 161/180 hist 162/169 = 323
        flip only if both ends pass gate fresh 163/180 hist 162/169 = 325
        gate each end, use whichever fresh 163/180 hist 161/169 = 324
        flip first, then gate fresh 163/180 hist 161/169 = 324

    The two looser orderings let a candidate in whose extent-chosen end
    the ROI gate had (correctly) rejected -- they recover the same 2
    fresh throws but give one historical throw back, because the gate was
    doing real work rejecting those views. Strictest ordering shipped.
    """
    alt = cand.get("opposite_end")
    if not alt:
        return cand
    if not (point_in_board_roi(cand["crossing_px"], roi_polygon)
            and point_in_board_roi(alt["crossing_px"], roi_polygon)):
        return cand

    def _plane_point(pixel):
        ray = back_project_ray(
            pixel, calib.camera_matrix, calib.dist_coeffs, calib.rvec, calib.tvec, cam=cam
        )
        return ray_plane_intersect(ray, z_mm=BOARD_PLANE_Z_MM)

    here = _plane_point(cand["crossing_px"])
    there = _plane_point(alt["crossing_px"])
    if here is None or there is None:
        return cand

    ground_x, ground_y = _camera_ground_xy(calib)
    d_here = (here[0] - ground_x) ** 2 + (here[1] - ground_y) ** 2
    d_there = (there[0] - ground_x) ** 2 + (there[1] - ground_y) ** 2
    if d_there >= d_here:
        return cand

    flipped = dict(cand)
    flipped.update(alt)
    flipped["opposite_end"] = {
        "tip_px": cand["tip_px"],
        "crossing_px": cand["crossing_px"],
        "axis_unit": cand.get("axis_unit"),
    }
    flipped["end_flipped"] = True
    return flipped


class AthenaEngine:
    """The registry entry named "Athena" -- see opendarts/engines/registry.py."""

    name = "Athena"
    # What a caller-shared `opendarts.imageops.DiffCrop` must satisfy for
    # this engine to use it -- see `detect_crossing()`'s `precomputed`.
    precompute_requirements = PRECOMPUTE_REQUIREMENTS

    def score(
        self,
        bg_images: dict[int, np.ndarray],
        frame_images: dict[int, np.ndarray],
        calibration: dict[int, CameraCalibration],
        *,
        prior_dart_line_px: "PriorDartLinePx | None" = None,
        precomputed: "dict | None" = None,
    ) -> EngineResult:
        """`prior_dart_line_px`: added 2026-08-24, keyword-only and
        optional so every existing caller (every test, every offline
        eval script, generic engine dispatch) is completely unaffected --
        same shape and same threading discipline as Apollo's own
        parameter of the same name
        (`opendarts.engines.apollo.prior_dart_context.PriorDartLinePx`):
        `{cam_index: (tip_px, far_end_px)}` for the immediately-prior
        throw of the SAME visit, same camera, recomputed fresh from that
        throw's own stored raw images (never a stale stored result -- see
        that module's own docstring for why). Declaring this parameter
        here is what makes `opendarts.capture.replay.replay_throw_with_engine()`,
        `opendarts.live.capture_daemon.handle_ready_to_capture()`, and
        `opendarts.engines.zeus.engine._score_sub_engine()` all thread it in
        automatically via their existing
        `engine_accepts_prior_dart_line_px()` capability check -- no
        further wiring needed at any of those call sites for Athena to
        start receiving real prior-dart context, live or in replay.

        **Real incident this addresses, 2026-08-24**:
        the recorded S10 throw (dart 2 of a 3-dart visit, thrown right after a D15 that landed 60mm away on
        the board but close enough in camera 0's own 2D projection to
        contaminate its board-crossing candidate -- see
        `opendarts.engines.apollo.prior_dart_context`'s own module
        docstring for the general mechanism; this is the SAME root cause
        already fixed for Apollo, see `opendarts/engines/zeus/
        engine.py`'s own 2026-08-24 dated comment). **Real per-camera
        diagnostics on this exact throw** (confirmed via
        `AthenaEngine.score()`'s own diagnostics, not assumed): this is
        NOT a blending failure -- cameras 1 and 2 both failed the board
        ROI gate outright on this throw (their own only candidates land
        143mm/276mm from centre, 200mm+ off the actual board, unrelated
        to prior-dart contamination -- perpendicular distance to the
        prior line is 68-99px on both, nowhere near
        `PRIOR_DART_LINE_MAX_PERP_PX`), so camera 0's contaminated
        candidate (perp distance 10.85px/14.66px on its own two ends,
        both under the 15px threshold) is the ONLY camera-reading this
        throw has at all (`n_cameras_used=1`), landing on a confident-but-
        wrong `single_inner` (50.3, -12.2mm, confidence 0.994) instead of
        AD's `treble` (94.1, -41.4mm).

        **Why this ships as a down-weight signal (see
        `PRIOR_DART_WEIGHT_PENALTY`'s own comment) rather than an
        admission-time exclusion, even though a down-weight cannot, by
        itself, fix THIS throw** -- `_combine_reads()` is proven
        weight-INVARIANT at n=1 (its own docstring: "not a blend at all
        -- it is a hard pick", and at n=1 there is only ever one read to
        pick), so no weight value changes camera 0's candidate being the
        sole input here; a hard exclusion was measured directly instead
        -- with camera 0 excluded, the ONLY remaining evidence is cameras 1/2's
        own genuinely-bad, unrelated-to-contamination candidates (200mm+
        off-board), so the ROI-gate fallback pass would replace a wrong
        ON-board answer (`single_inner`) with a wrong OFF-board one
        (`outside`) -- not a fix, and for a throw where a real,
        un-contaminated alternative HAD existed, the same hard exclusion
        risks destroying it outright with no arbitration step to fall
        back on (Apollo's own sibling guard exists ONLY as a candidate
        for `score_dart()`'s own ray-disagreement arbitration to weigh
        against the unmodified full set, never a direct exclusion --
        Athena's own architecture has no equivalent per-throw arbitration
        step to safely reproduce that pattern with). The down-weight
        formulation is therefore the honest ceiling of what's safely
        available here: it protects any FUTURE throw where a
        prior-dart-contaminated camera competes against a real
        alternative in a blend or in `_corroborated_label_override()`'s
        weight-total comparisons, while this exact throw stays an honest,
        unfixed miss for Athena alone -- see this task's own corpus
        validation for whether the full `Zeus`/Zeus vote still recovers
        via Apollo's own already-shipped fix.

        **Corpus validation** (the session corpus, 374
        throws total, 2026-08-24): every throw's real baseline (no `prior_dart_line_px`)
        vs guarded Athena result was compared against operator/AD truth.
        249/374 throws have a real prior-dart line at all (visit_index
        >= 1 with a locatable prior throw); of those, 36 have >=1 USED
        camera flagged prior-dart-suspected -- every other throw in the
        corpus is BYTE-IDENTICAL before/after (the guard is a true no-op
        there, not just unmeasured). At the shipped `PRIOR_DART_WEIGHT_
        PENALTY=0.5`: **+1 real fix, 0 regressions** on those 36 --
        the recorded S20 throw flips from a wrong `1/single_inner`
        (a corroborated-label-override picking a lone camera's own point
        plus two intersections it's part of over two OTHER cameras' own
        agreeing points) to the correct `20/single_inner`. This exact
        throw's own real incident (the recorded S10 throw) stays an
        HONEST, UNFIXED miss for Athena alone (see above: n=1, weight-
        invariant) -- but Zeus's own vote on it is independently already
        correct (`10/treble`) via Apollo's own already-shipped
        guard (c5f6a6d) agreeing with Talos, 2-of-3, with or
        without this Athena change -- confirmed by direct replay, not
        assumed. See
        `PRIOR_DART_WEIGHT_PENALTY`'s own comment for the real swept
        value behind 0.5 (a naive first guess of 0.05 measured a real
        regression on this same 36-throw set before the sweep found the
        actual safe plateau).
        """
        cams = sorted(set(bg_images) & set(frame_images) & set(calibration))
        per_camera: dict[int, dict] = {}
        reads: list[dict] = []
        # Per-camera inputs for the SHAFT-LINE INTERSECTION pass below --
        # each strict-gate-passing camera's picked candidate (which carries
        # the shaft's own image axis) + its calibration + its point-read
        # weight. Only filled on the normal path; the ROI-fallback pass
        # deliberately stays point-only (it exists to avoid a blank
        # answer, not to compound ungated evidence).
        line_inputs: dict[int, dict] = {}
        # Kept around ONLY for the ROI-FALLBACK PASS below (need each
        # camera's own det/candidates/calib/roi_polygon again without
        # re-running detect_crossing()) -- not read at all on the normal
        # (>=1 camera passed the gate) path.
        detections: dict[int, dict] = {}

        for cam in cams:
            # `precomputed` is only passed when there is a bundle for this
            # camera -- test doubles keep the plain two-argument signature.
            pc_kwargs = {"precomputed": precomputed[cam]} if precomputed and cam in precomputed else {}
            det = detect_crossing(bg_images[cam], frame_images[cam], **pc_kwargs)
            entry: dict = {
                "detected": det.ok,
                "reason": det.reason,
                "tip_px": det.tip_px,
                "crossing_px": det.crossing_px,
                "elongation_ratio": det.elongation_ratio,
                "tip_confidence": det.tip_confidence,
            }
            per_camera[cam] = entry
            if not det.ok or det.crossing_px is None:
                entry["used"] = False
                continue

            calib = calibration[cam]

            # Calibration-aware re-ranking (opendarts.engines.athena.
            # board_gate): crossing_detection.py picked its best candidate
            # by area/elongation alone, with no idea where the board even
            # is in this image, or which candidate's own ray geometry is
            # most trustworthy. Walk its full candidate list, keep every
            # one whose crossing pixel actually falls inside this
            # camera's own projected board ROI (rejects the "wrong blob
            # entirely" failure mode -- a bright reflection etc. -- a real
            # corpus run found this simple blob detector hitting often
            # enough to matter, without needing crossing_detection.py
            # itself to know about calibration, kept decoupled per its own
            # module docstring), then of the ROI-passing candidates pick
            # the one with the STEEPEST ray (see _camera_weight()'s own
            # docstring for the real measurement behind why ray steepness
            # is this engine's trust signal) rather than just the first
            # one in area/elongation order. Measured
            # (real data/archive/clean/
            # corpus): first-by-area 82.8%, most-confidently-inside-ROI
            # 82.2%, steepest-ray-among-passing 83.4% -- steepest wins,
            # shipped below.
            roi_polygon = project_board_roi_polygon(calib)
            candidates = det.diagnostics.get("candidates") or [
                {
                    "crossing_px": det.crossing_px,
                    "elongation_ratio": det.elongation_ratio,
                    "tip_confidence": det.tip_confidence,
                }
            ]
            detections[cam] = {
                "calib": calib, "roi_polygon": roi_polygon, "candidates": candidates,
            }
            picked, ray, best_cand_steepness, n_end_flips = _select_best_candidate(
                candidates, calib, cam, roi_polygon, require_roi=True,
            )
            entry["n_end_flips"] = n_end_flips
            if picked is None:
                entry["used"] = False
                entry["reason"] = "no candidate crossing pixel fell inside the projected board ROI"
                continue

            xy = ray_plane_intersect(ray, z_mm=BOARD_PLANE_Z_MM)
            if xy is None:
                entry["used"] = False
                entry["reason"] = "ray does not validly cross the board plane (parallel or behind camera)"
                continue

            # Applied HERE, per read, rather than once at the end, so this
            # camera's own reported (sector, ring, x_mm, y_mm) diagnostic
            # is in the same frame as the final combined answer. Measured
            # identical accuracy either way, every value tried -- see
            # RADIAL_CORRECTION_MM's own comment.
            x_mm, y_mm = apply_radial_correction(*xy)
            sector, ring = sector_ring_for_point(x_mm, y_mm)
            ray_steepness = best_cand_steepness
            weight = _camera_weight(ray_steepness, picked.get("tip_confidence"))
            # Prior-dart-in-visit contamination guard, 2026-08-24 -- see
            # `score()`'s own dated docstring section and
            # `PRIOR_DART_WEIGHT_PENALTY`'s own comment. `prior_dart_line_px`
            # is `None` by default, so `contaminated` is always False and
            # `weight` unchanged for every existing caller.
            prior_line = prior_dart_line_px.get(cam) if prior_dart_line_px else None
            contaminated = _prior_dart_contamination_suspected(picked, prior_line)
            if contaminated:
                weight *= PRIOR_DART_WEIGHT_PENALTY
            entry.update({
                # NOTE: "used" here means "cleared the board ROI
                # admission gate," NOT "fed the final answer" -- a
                # camera can be used=True and still be excluded from
                # the Weiszfeld blend by the ray-steepness floor below.
                # See the dated 2026-09-02 comment at `camera_reads =
                # qualifying if qualifying else reads` for the full
                # investigation; `diagnostics["n_cameras_used"]` is the
                # field that reflects "reached the final answer."
                "used": True,
                # The candidate the engine ACTUALLY used -- entry["tip_px"]
                # /["crossing_px"] above are the detector's own first
                # choice, which candidate re-ranking may have overridden;
                # the 2026-08-17 miss investigation was actively misled by
                # stored diagnostics that lacked this distinction.
                "used_crossing_px": picked["crossing_px"],
                "used_area_px": picked.get("area"),
                "x_mm": x_mm,
                "y_mm": y_mm,
                "sector": sector,
                "ring": ring,
                "weight": weight,
                "ray_steepness": ray_steepness,
                "elongation_ratio": picked["elongation_ratio"],
                "tip_confidence": picked["tip_confidence"],
                "prior_dart_contamination_suspected": contaminated,
            })
            reads.append({
                "cam": cam, "x_mm": x_mm, "y_mm": y_mm, "sector": sector, "ring": ring,
                "weight": weight, "ray_steepness": ray_steepness,
                "area": picked.get("area"),
            })
            line_inputs[cam] = {"picked": picked, "calib": calib, "weight": weight}

        roi_fallback_used = False
        if not reads and detections:
            # ROI-FALLBACK PASS. 2026-08-14: "I never want to see
            # a no result" -- this engine must always commit to SOME
            # answer, the same way Apollo/Talos always triangulate to
            # some xy and let `sector_ring_for_point()` classify it
            # (which can legitimately return ring="outside", a real,
            # valid answer, not a failure). Before this pass, EVERY
            # camera's candidates failing the ROI gate meant an
            # unconditional `ok=False` -- structurally unable to ever
            # report "outside", even when that is exactly the right call
            # sitting right there in an ungated candidate.
            #
            # Deliberately gated on `reads` being empty ACROSS THE WHOLE
            # THROW (not per-camera) -- this never fires on the other
            # 358/360 real corpus throws, where at least one camera
            # already passes the strict gate, so it cannot regress a
            # single one of them: it only ever replaces "no answer" with
            # "some answer" on throws that would otherwise be a blank
            # no-score. Confirmed by measurement, not just by
            # construction (full 360-throw
            # data/archive/clean/ corpus, before/after byte-identical on
            # every throw except the 2 that were previously no-score):
            # `throw_1786730541658` (AD `None/outside`, a dart that
            # genuinely landed outside the board) -- 3/3 cameras'
            # crossing pixels, ungated, all still project to
            # `ring="outside"`: FIXED, now correctly `ok=True`,
            # `ring="outside"`, matching AD exactly.
            # `throw_1786665050832` (AD `9/single_inner`, a real on-board
            # dart every gated candidate on every camera missed, confirmed
            # by direct investigation to be missed by ALL
            # THREE engines under current code -- a shared/upstream
            # detection failure on this specific throw, not something
            # fixable from inside this one engine) -- all 3 ungated
            # fallback candidates land 200+mm outside the board, combining
            # to `None/outside`: still wrong (AD's dart was genuinely
            # on-board), but a real, honestly-bad answer instead of a
            # blank one -- exactly the project's ask ("never a no result"), not
            # a claim this recovers the correct segment.
            for cam, d in detections.items():
                picked, ray, steepness, n_end_flips = _select_best_candidate(
                    d["candidates"], d["calib"], cam, d["roi_polygon"], require_roi=False,
                )
                entry = per_camera[cam]
                entry["n_end_flips"] = n_end_flips
                if picked is None:
                    continue
                xy = ray_plane_intersect(ray, z_mm=BOARD_PLANE_Z_MM)
                if xy is None:
                    continue
                x_mm, y_mm = apply_radial_correction(*xy)
                sector, ring = sector_ring_for_point(x_mm, y_mm)
                weight = _camera_weight(steepness, picked.get("tip_confidence"))
                # Same prior-dart guard as the strict pass above -- see
                # `score()`'s own dated docstring section. Applied here
                # too so a contaminated candidate stays discounted even
                # when it only surfaces via the ungated fallback path.
                prior_line = prior_dart_line_px.get(cam) if prior_dart_line_px else None
                contaminated = _prior_dart_contamination_suspected(picked, prior_line)
                if contaminated:
                    weight *= PRIOR_DART_WEIGHT_PENALTY
                entry.update({
                    "used": True,
                    "roi_fallback": True,
                    "used_crossing_px": picked["crossing_px"],
                    "used_area_px": picked.get("area"),
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "sector": sector,
                    "ring": ring,
                    "weight": weight,
                    "ray_steepness": steepness,
                    "elongation_ratio": picked["elongation_ratio"],
                    "tip_confidence": picked["tip_confidence"],
                    "prior_dart_contamination_suspected": contaminated,
                    "reason": "ROI-gated candidate unavailable; used ungated fallback candidate",
                })
                reads.append({
                    "cam": cam, "x_mm": x_mm, "y_mm": y_mm, "sector": sector, "ring": ring,
                    "weight": weight, "ray_steepness": steepness,
                })
            roi_fallback_used = bool(reads)

        if not reads:
            return EngineResult(
                ok=False,
                sector=None,
                ring=None,
                board_xy_mm=None,
                reason="no camera produced a usable board-crossing read",
                # confidence=0.0 here is not fabricated -- an ok=False
                # result has, by definition, zero real support (this
                # branch is only reachable when the ROI-FALLBACK PASS
                # above also found nothing, i.e. every camera failed
                # outright), so 0.0 is the only honest value; see
                # opendarts.engines.athena.confidence's own module
                # docstring for how every OTHER (ok=True) confidence
                # value is actually measured/calibrated, not asserted.
                diagnostics={"per_camera": per_camera, "confidence": 0.0},
                confidence=0.0,
            )

        # Hard steepness floor for CONSENSUS INPUT (distinct from the
        # continuous weight formula above, which already downweights a
        # shallow ray but never fully excludes it). Measured
        # (inline sweep, real
        # data/archive/clean/ corpus): dropping any read with
        # ray_steepness below a floor before the Weiszfeld blend, real
        # plateau 84.6% for floor in [0.4, 0.5] (vs 84.0% with no floor)
        # -- a genuinely grazing ray's board-plane estimate is noisy
        # enough that even a low Weiszfeld weight isn't always enough to
        # stop it pulling the blend, so excluding it outright (when at
        # least one other camera still qualifies -- never drops below 1
        # camera) measured better than downweighting alone. 0.45 shipped
        # (middle of the plateau).
        #
        # **2026-09-02 -- this is the exact point where `per_camera[cam]
        # ["used"]` and `diagnostics["n_cameras_used"]` genuinely part
        # ways -- investigated and confirmed real, not a bug, after a
        # corpus-QA peer session found the two disagree (per_camera
        # claiming MORE cameras than the count) on 70/1902 real throws
        # (~3.7%, real opendarts+opendarts corpus), always in the same
        # direction, skewed ~10x toward `ring="outside"` throws (79.7%
        # of disagreements vs a 7.9% baseline OUT rate).** `per_camera
        # [cam]["used"]` is set True for every camera whose candidate
        # cleared the board ROI admission gate above (the strict pass,
        # or the ROI-fallback pass when no camera cleared it) -- i.e.
        # it is exactly `n_cameras_gate_passed`'s own per-camera
        # breakdown, "did this camera's read get admitted at all."
        # `n_cameras_used` (the diagnostics field emitted below, `len(
        # camera_reads)`) is a STRICT SUBSET: only the admitted reads
        # that ALSO cleared this steepness floor and therefore actually
        # fed `_combine_reads()`'s Weiszfeld blend -- i.e. "did this
        # camera's read reach and influence the final (sector, ring,
        # board_xy_mm)." A camera can be `used=True` in `per_camera`
        # while genuinely excluded from the number that decided the
        # final answer; `n_cameras_gate_passed` is the field that
        # actually equals the `per_camera[...]["used"]` count, not
        # `n_cameras_used`. Both fields are real, correctly computed,
        # and describe genuinely DIFFERENT quantities -- "admitted" vs
        # "influenced the final blend" -- not a bug to reconcile; a
        # consumer that wants "cameras whose data reached the final
        # answer" for this engine must read `n_cameras_used`, never
        # count `per_camera[...]["used"]` entries. The 79.7%-OUT skew
        # is consistent with the ROI-fallback pass (fired disproportion-
        # ately on off-board darts) admitting cameras via ungated,
        # geometrically noisier candidates that are then more likely to
        # miss the steepness floor -- observed, not chased further.
        qualifying = [r for r in reads if r["ray_steepness"] >= MIN_RAY_STEEPNESS_FOR_CONSENSUS]
        camera_reads = qualifying if qualifying else reads

        # SHAFT-LINE INTERSECTION PASS -- see opendarts.engines.athena.
        # shaft_lines' module docstring for the concept (fed into this
        # engine's own continuous weighted consensus), the real
        # measurement behind it (pairwise intersections median 2.24mm vs
        # point reads 2.81mm against AD on the full 420-throw corpus),
        # and the conditioning gates. Each valid pair of strict-gate
        # cameras contributes ONE extra weighted read to the same
        # Weiszfeld blend the point reads already feed: the intersection
        # of the two cameras' board-plane shaft lines, which is immune to
        # BOTH cameras' along-shaft tip-localization error -- the error
        # family the per-camera point reads structurally cannot escape,
        # and (measured, tmp/c2d_eval_results dump) the dominant cause of
        # this engine's adjacent-sector misses. Pairs are built from ALL
        # strict-gate cameras (not just steepness-floor survivors): a
        # grazing ray makes a noisy POINT but its shaft LINE is still
        # transversally sound, and the pair weight already carries that
        # camera's own low point weight. A side effect worth naming: a
        # 2-camera throw now has 3 reads, so it goes through the real
        # geometric-median blend instead of _combine_reads()'s documented
        # degenerate-at-2 weighted-mean special case -- the intersection
        # is exactly the third, independent opinion that case was missing.
        intersection_reads: list[dict] = []
        if len(line_inputs) >= 2:
            line_cams = sorted(line_inputs)
            lines = {}
            for cam in line_cams:
                d = line_inputs[cam]
                axis_unit = d["picked"].get("axis_unit")
                if axis_unit is None:
                    continue
                ln = board_plane_line(
                    d["picked"]["crossing_px"], axis_unit, d["calib"], cam,
                    z_mm=BOARD_PLANE_Z_MM,
                )
                if ln is not None:
                    lines[cam] = ln
            for idx_a in range(len(line_cams)):
                for idx_b in range(idx_a + 1, len(line_cams)):
                    cam_a, cam_b = line_cams[idx_a], line_cams[idx_b]
                    if cam_a not in lines or cam_b not in lines:
                        continue
                    hit = intersect_board_lines(lines[cam_a], lines[cam_b])
                    if hit is None:
                        continue
                    (ix, iy), angle_deg = hit
                    if angle_deg < MIN_CROSSING_ANGLE_DEG:
                        continue
                    x_mm, y_mm = apply_radial_correction(ix, iy)
                    if (x_mm * x_mm + y_mm * y_mm) ** 0.5 > MAX_INTERSECTION_RADIUS_MM:
                        continue
                    sector, ring = sector_ring_for_point(x_mm, y_mm)
                    weight = intersection_weight(
                        line_inputs[cam_a]["weight"], line_inputs[cam_b]["weight"], angle_deg,
                    )
                    intersection_reads.append({
                        "cam": (cam_a, cam_b), "x_mm": x_mm, "y_mm": y_mm,
                        "sector": sector, "ring": ring, "weight": weight,
                        "crossing_angle_deg": angle_deg, "kind": "intersection",
                    })

        consensus_reads = camera_reads + intersection_reads

        combined = _combine_reads(consensus_reads)
        sector, ring = sector_ring_for_point(combined["x_mm"], combined["y_mm"])
        combined, sector, ring = _corroborated_label_override(
            consensus_reads, combined, sector, ring,
        )

        # Real, CALIBRATED confidence score -- see
        # opendarts.engines.athena.confidence's own module docstring for
        # the full design, the real per-signal measurements it's built
        # from, and the LOSO-cross-validated calibration curve behind
        # `calibrated_confidence()`. `weighted_spread_mm()` stashes each
        # read's own distance to the blended point onto the read dict
        # itself (`_dist_to_combined_mm`) so `raw_confidence_score()`
        # can reuse it without recomputing.
        weighted_spread_mm(consensus_reads, combined["x_mm"], combined["y_mm"])
        raw_confidence = raw_confidence_score(consensus_reads, sector, ring, roi_fallback_used)
        confidence = calibrated_confidence(raw_confidence)

        reason = (
            f"confidence-weighted blend of {len(camera_reads)}/{len(cams)} camera(s) "
            f"+ {len(intersection_reads)} shaft-line intersection read(s) "
            f"({len(reads)} passed the board ROI gate, "
            f"{len(reads) - len(camera_reads)} excluded by the ray-steepness floor)"
        )
        if roi_fallback_used:
            reason = (
                f"ROI-gate fallback: no camera's candidate passed the board ROI gate, "
                f"used {len(reads)} ungated candidate(s) instead. " + reason
            )
        return EngineResult(
            ok=True,
            sector=sector,
            ring=ring,
            board_xy_mm=(combined["x_mm"], combined["y_mm"]),
            reason=reason,
            diagnostics={
                "per_camera": per_camera,
                # "n_cameras_used" is the "reached the final answer"
                # count -- see the dated 2026-09-02 comment above
                # `camera_reads = qualifying if qualifying else reads`
                # for the full investigation. `per_camera[...]["used"]`
                # tracks a different, wider quantity (admission-gate
                # pass) and equals `n_cameras_gate_passed` below, not
                # this field, whenever the steepness floor excludes at
                # least one admitted camera.
                "n_cameras_used": len(camera_reads),
                "cameras_used": [r["cam"] for r in camera_reads],
                "n_intersection_reads": len(intersection_reads),
                "intersection_reads": [
                    {k: r[k] for k in
                     ("cam", "x_mm", "y_mm", "sector", "ring", "weight",
                      "crossing_angle_deg")}
                    for r in intersection_reads
                ],
                "n_cameras_gate_passed": len(reads),
                "label_override_used": bool(combined.get("label_override", False)),
                "roi_fallback_used": roi_fallback_used,
                "confidence": confidence,
                "raw_confidence_score": raw_confidence,
            },
            # 2026-08-14 -- also stamped on the shared EngineResult.confidence
            # field (Talos added it, `d683b40`, and populates it directly on
            # every result) so this engine's confidence is readable the same
            # way as Talos's, not silently None on the "official" field
            # while a real value sits only in diagnostics.
            confidence=confidence,
        )
