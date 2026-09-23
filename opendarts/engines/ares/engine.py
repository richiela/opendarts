"""Ares -- greenfield 2D-per-camera scoring around board-plane shaft-
line CONCURRENCY. The registry entry named "Ares".

Design (see each module's docstring for the measurements behind it):

1. `detection` -- per camera, motion-diff components + a THIN-RUN
   centerline fit that structurally excludes the flight's occlusion-diff
   mass (the confirmed source of multi-millimetre line tilt on the
   corpus's dominant 20|1-wire misses -- the mask there literally
   contains the board's bright "20" numeral behind the flight).
2. `geometry` -- each camera's shaft centerline back-projected onto the
   board plane at Z=0 exactly. No BOARD_PLANE_Z_MM, no
   RADIAL_CORRECTION_MM, no per-epoch millimetre fudges anywhere.
3. CONSENSUS SELECTION (this module) -- candidate hypotheses are all
   pairwise line intersections across cameras plus all admitted tip
   points; each is scored by how much weighted evidence (best candidate
   per camera) agrees with it. This replaces any per-camera "pick the
   biggest/cleanest blob" rule: junk candidates -- however clean-looking
   in isolation -- do not agree with other cameras' evidence, and the
   real dart does. (Measured: per-camera quality ranking alone picked a
   wrong-but-thin candidate on 6 of the 357-run's 17 misses.)
4. `fusion` -- ONE robust IRLS solve over the consensus candidates'
   lines (strong, measured sigma ~1mm perpendicular) and tip points
   (weak, ~4mm), seeded at the winning hypothesis. Points fill exactly
   the directions lines cannot determine; no degenerate-geometry special
   cases remain.

The engine's tip/base call per candidate is geometric, not image-based:
the end whose board-plane projection lies CLOSER to the camera's own
ground position is the tip (the flight sits above the plane, so its ray
overshoots away from the camera). No threshold; a pure comparison.

Honest builder's log (2026-08-24, the session corpus, 374
throws, replayed over the stored corpus; baseline: current Athena
replay 364/374):

- Whole-blob PCA axis vs thin-run centerline: measured as a probe before
  building -- whole-blob line perp error median 0.727mm / p99 7.4mm; a
  first-draft shaft-only refit was NOT better (0.706mm / p99 9.6mm,
  52% coin flip) until the per-bin largest-contiguous-run + strict
  width-threshold form here landed; the miss-throw cam1 lines (2.3-4.4mm
  perp error, all five 20|1 misses) are what the stricter form fixes.
- Lines-only naive LS (no weights, no robustness, first ROI candidate):
  333/374. Quality-weighted lines + tip-gated admission + robust line
  construction: 357/374. Line-gate-only admission (a camera's line
  admitted even when its tip projects off-board) measured 330/374 --
  withdrawn: junk lines that happen to cross the board poisoned 2-line
  intersections on 31 throws.
- Consensus selection + unified line+point solve: 363 (+10/-4 vs 357).
  Per-camera support cap + along-segment kernel: 369 (+6/-0).
  Residual-outlier line trim: 370 (+1/-0). Line-only admission +
  hypothesis-conditioned solve weights: 371 (+1/-0). Contamination
  penalty restricted to solve weights (not consensus choice): 372
  (+1/-0). FINAL: 372/374 (99.47%) vs Athena's 364/374 -- fixes 8 of
  Athena's 10 misses, introduces 0 new ones; the 2 remaining misses
  (032-S12, 037-T18) are shared with Athena, which produces the
  IDENTICAL wrong answers on both   (independent architectures agreeing
  on the "wrong" side of a wire 1.1mm/2.4mm from AD's own tip -- the
  measured error floor of this rig's 2-camera line geometry, not a
  selection or fusion failure).

Drop-reason diagnostics (2026-08-27): the two ~100mm
single-camera misses (g1-003-S5 / g3-036-S5, both third-dart-of-
visit) were undiagnosable from their own packages -- a dropped camera
recorded only {n_candidates, used:false}, and across both corpora all 46
camera drops HAD detections that were discarded silently. Root cause,
reproduced offline on both packages (behavior identical before/after
this change -- this entry ships LOGGING, not a behavior fix):

- g1-003-S5: cam1 detected the real dart (its point read landed 4.8mm
  from AD's tip) but the thin-run centerline fit failed by ONE bin
  (thin run 5 of 27 bins, 6-bin minimum -- width profile fattened by
  prior-dart overlap; it's dart 3 of the visit), so the read was
  degraded to point-only evidence at 0.5 weight. cam2's fit succeeded
  but its geometric tip pick projected off-gate (r=237mm) with
  line_weight 0.086 < 0.15 -- inadmissible. cam0's winning candidate --
  a 30px stub lying ON the prior throw's own line (contamination
  suspected=True) -- self-supported at the 1.0 camera cap, beating the
  correct camera's 0.5. support() ignoring contamination is a measured
  decision (g1-042, see support()'s docstring); the cap protecting
  two-camera agreement is too (the 363-run) -- but when NO cross-camera
  hypothesis exists, "full-quality junk vs degraded truth" is decided
  1.0 vs 0.5, a real, now-documented weakness this diagnostics change
  makes visible per-package rather than silently fixing (any fix here
  must be corpus-measured first; the contamination/cap levers have
  measured counterexamples in both directions).
- g3-036-S5: same terminal state, different path -- cam1's line missed
  the line-only admission floor by 0.0006 (0.1444 vs 0.15), cam2's tip
  was admitted but sat ~100mm off along its own badly-fit line (rms
  8.1px), cam0's junk won by self-support.

What ships here: every per_camera entry (used or not) now carries a
`candidates` list -- per candidate: gate values against thresholds
(tip_r_mm vs the tip gate, line_weight vs the line-only floor),
admitted/rejection, consensus_support vs the winning hypothesis, and
detection's own `centerline_failure` reason when the thin-run fit
degraded it -- plus `drop_stage` ("no_candidates"/"admission"/
"consensus") and `best_consensus_support` for dropped cameras, and a
result-level `gate_thresholds` dict so packages are self-describing.
Purely additive; all pre-existing keys unchanged.

Related closure (docs/DESIGN.md): the numpy matmul RuntimeWarnings
from detection.py are verified SPURIOUS -- see detection.py's module
docstring (finite inputs, finite outputs, macOS Accelerate FP-status-
flag artifact).

Consensus fix pass (2026-08-27, follow-up to the diagnostics entry
above; the two section-8 miss packages were pulled back from the rig's
quarantine for this, since the corpus freeze deleted them locally).
Three levers were built and measured independently on the FULL frozen v1
corpus (1414 graded throws, operator_marked_wrong convention) plus the two miss packages:

- ASYMMETRIC along kernel (SCORE_SIGMA_ALONG_TIPWARD_MM): measured on
  both misses, every real shaft's board-plane segment ended 94-180mm
  short of AD's true tip, ALWAYS beyond the segment's TIP end (prior-
  dart occlusion eats the tip side of the shaft), never past the base.
  Tipward sigma swept at {30,60,90,120,200,400}: 60 is the measured
  net-best (see the constant's own comment for the numbers).
- Contaminated candidates no longer GENERATE tip hypotheses (they still
  support others' hypotheses, still enter the solve, and are re-admitted
  if no hypothesis forms at all). Corpus-neutral (0 flips either way)
  and the decisive lever for g1-003-S5 (the winning junk was the
  contaminated stub's own tip hypothesis).
- Multi-camera tier (CONSENSUS_MULTI_CAM_MIN_SUPPORT): >= 2 cameras at
  >= 0.1 support outranks any single-camera hypothesis. The strongest
  single lever: +5 fixed / 0 broken on the corpus by itself.

Shipped combination (all three, tipward sigma 60): corpus 98.44% ->
98.87% (+8 fixed / -2 broken, p90 3.37 -> 3.36mm), and g1-003-S5 scores
5/single_outer, 6.2mm from AD, from 2 cameras (was 111.5mm junk).

Honest limits, measured not guessed:
- g3-036-S5 was diagnosed but NOT fixable at this consensus operating
  point: its rescue needs ~100mm+ of tipward allowance (its true
  two-line intersection sits 94-101mm beyond both source tips), and
  every config that admits that much (sigma >= ~100-120) nets WORSE on
  the corpus (98.73% at 120, +10/-6: strong lines start propping junk
  intersections far beyond their own good tips, e.g. 070-S20).
  RESOLVED 2026-08-27 at the DETECTION stage instead -- see the
  detection pass entry below.
- Measured dead ends, do not re-try without new evidence: a tier
  raw-score margin (contradictory: g3-036 needs promotion at 0.55x the
  raw leader, 069-S9 needs suppression at 0.62x); suppressing
  contaminated LINES from intersection generation (replays the g1-042
  trap -- broke it on the corpus, 4.1mm -> 170.8mm); tier floors above
  0.1 (only lose fixes); a visit-index gate on the asym kernel (broken
  070-S20 is itself a third dart, fixed 043-S4 is a first dart).

Detection fix pass (2026-08-27, same session as the consensus pass
above; root-caused by pixel-level comparison against Apollo's detector
on g3-036-S5). The real g3-036 failure was never consensus: on 2 of 3
cameras the thin-run walk TRUNCATED the shaft -- an 11-bin near-board
silhouette widening measured 12.3-12.6px against a 12.2px thin
threshold (0.1-0.4px over) and hard-terminated the walk, amputating a
clean 5-bin taper to 1px sitting dead on AD's truth right past the
streak -- and on that same camera the outer-fifth end-choice mean
picked the FLIGHT end (12.1 vs 11.4px), starting the walk backwards.
Two shipped detection.py changes (see FAT_STREAK_RESUME_MIN_THIN_BINS
and the end-choice comment in _fit_centerline): the walk resumes past
a sustained fat streak when real thin signal continues (the streak's
own shadow-dragged bins stay out of the fit), and the tip end is
chosen by the terminal 3 bins' taper instead of the outer-fifth mean.
Measured on the full frozen corpus against the committed consensus-pass
baseline: 98.87% -> 99.58% (1398 -> 1408 of 1414, +13 fixed / -3
broken, median 1.36 -> 1.32mm, p90 3.36 -> 3.22mm), including both
consensus-pass known misses (070-S20 66.5mm and 032-S15 43.3mm, now
1.3mm/0.3mm) and both original section-8 quarantined misses. The 3
shipped regressions are ALL near-wire, worst 5.4mm, zero howlers: the
documented 015-S1 trap pair (20|1 wire, 2.3/1.2mm -> 3.5/5.4mm on the
wrong side; mechanism diagnosed -- with the flipped walk start the
resumed extension on cam1 bends the fit and shifts the board line
~4mm across the wire) and 024-T5 (T5|T20 wire at 1.8mm). This profile
was chosen over an equal-accuracy alternative (outer-fifth end-choice
+ an rms collinearity guard on the resume, also 1408) whose
regressions include a NEW 141mm howler -- confidently-wrong howlers
are this project's named worst failure mode and exactly what this
pass was dispatched to fix; every guard variant either ships that
howler, gives up 4 net fixes to avoid it (1404), or manufactures
confident junk lines (trim-until-straight: a 229mm fabricated answer
on an off-board dart). Full variant matrix in detection.py's
rejected-guard note above N_TIP_PIXELS_AVERAGED. Other measured dead
ends: extend-only/no-rescue resume (1405); min-thin 3 (1402).

`n_cameras_used` diagnostics field, 2026-09-02 -- this engine was the
only one of the four (Apollo, Talos, Athena
all already had a count field of some kind) with no camera-count
summary at all. Added, matching the semantic a paired investigation of
Athena's own `n_cameras_used` established from that engine's real
code path (see `opendarts.engines.athena.engine.AthenaEngine.score()`'s
own dated 2026-09-02 comment): "cameras whose data reached and
influenced the final combined answer," computed from the exact `lines`/
`points` observation lists that feed the actual `fuse()` call this
function returns (post residual-outlier trim), not from `selected`
(consensus-chosen candidates, which `per_camera[cam]["used"]` already
tracks and is a real, different, WIDER quantity -- a camera can win
consensus and still contribute nothing to the final solve, e.g. a
trimmed line with no admitted point). Purely additive, `ok=True` path
only (mirrors every other engine's convention of not fabricating a
count on a no-answer result); zero behavior change -- confirmed via a
before/after full local corpus replay (the session corpus,
951 throws, all 5 engines): Zeus 945/951, Apollo 938/951, Ares
935/950, Talos 929/951, Athena 928/951, byte-identical before and
after this field was added.
"""
from __future__ import annotations

import math

import numpy as np

from opendarts.engines.base import EngineResult
from opendarts.engines.ares.detection import (
    PRECOMPUTE_REQUIREMENTS,
    DartCandidate,
    detect_candidates,
)
from opendarts.engines.ares.fusion import (
    LineObservation,
    PointObservation,
    fuse,
    line_intersection,
)
from opendarts.engines.ares.geometry import (
    board_plane_line,
    camera_ground_xy,
    perp_distance_mm,
    pixel_board_xy,
)
from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM, sector_ring_for_point
from opendarts.pipeline import CameraCalibration

# Admission gate: the candidate's (geometrically-picked) tip must project
# inside a padded board disc. Same 1.2x pad class every board-aware gate
# in this project uses -- a tip can legitimately land just outside the
# double wire (a real "outside" answer), but a blob whose tip projects
# 100mm+ off the board is a wrong-blob detection, not a dart.
TIP_GATE_RADIUS_MM = 1.2 * DOUBLE_OUTER_RADIUS_MM

# A dart shaft is a thin structure; a fitted "centerline" whose mean
# cross-section width is this wide is a mis-fit through a compact mass
# (flight occlusion diff, merged blobs), not a shaft -- its line is not
# admissible evidence (its tip point may still be). Real corpus cases:
# 036-S1 cam1 (45px), 073-S5 cam1 (34px), g1-075 cam2 (32px), all with
# multi-mm to 100mm+ line errors; real shafts on this rig measure
# 2-16px.
MAX_SHAFT_WIDTH_PX = 25.0

# Line-quality weight shape. Both signals were measured corpus-wide
# (765 gate-passing lines vs AD tip): lines from
# shafts shorter than ~100px carry ~1.4-2.2x the perpendicular error
# (median 1.01mm vs 0.70mm; junk 19-42px runs are far worse still), and
# fit RMS above ~2px marks the mis-fit population (p90 5.35mm vs
# 1.4-2.3mm below it). The weight is 1/sigma^2-shaped, not a hard gate:
# w = (len/(len+LEN_SOFT))^2 * 1/(1+(rms/RMS_SOFT)^2).
LINE_WEIGHT_LEN_SOFT_PX = 60.0
LINE_WEIGHT_RMS_SOFT_PX = 2.5

# Consensus-scoring kernels: a candidate's line supports a hypothesis by
# lw * GAIN * exp(-perp^2/2*sigma_l^2) * exp(-along^2/2*sigma_along^2),
# its tip by pw * exp(-d^2/2*sigma_p^2). Sigmas are the measured
# evidence scales (lines ~1mm class perpendicular with a junk tail, tips
# ~4mm class); GAIN keeps a good line worth several tips. The ALONG term
# is what makes a line a SEGMENT for scoring purposes: an infinite line
# crosses the whole board and would otherwise lend support to junk
# hypotheses tens of mm from the shaft it was fit to (measured: two of
# the 363-run's misses were junk hypotheses propped up by exactly this
# leak). Sigma_along is deliberately loose (measured along-shaft tip
# error: median 3.6mm, biased -2.8mm) so it only kills far-away leaks.
#
# 2026-08-27, ASYMMETRIC along kernel (the g1-003-S5/g3-036-S5 fix; see
# the dated entry at the end of the module docstring): the penalty is
# applied only on the BASE side of the tip anchor. Measured on both
# ~100mm third-dart misses: every real shaft's board-plane segment ended
# 100-180mm SHORT of AD's true tip, always beyond the segment's TIP end
# (prior-dart occlusion eats the tip side of the shaft, so the visible
# segment's tip-end projection lands mid-shaft), never past the base.
# A hypothesis beyond the tip end is exactly where a real tip sits when
# the shaft is occluded -- it stays pinned by the 1.5mm perpendicular
# kernel -- while mid-shaft/base-side hypotheses (where a dart tip
# cannot physically be) keep the full leak-killing penalty.
SCORE_SIGMA_LINE_MM = 1.5
SCORE_SIGMA_ALONG_MM = 10.0
# Tipward (beyond the tip end) along sigma -- looser than the base-side
# 10mm. Sizing is a real, measured corpus trade-off, not free: the two
# root-caused third-dart misses needed 94-111mm of tipward allowance to
# fully rescue their line support, but the full frozen-corpus sweep
# (1414 graded throws x sigma in {30,60,90,120,200,400}) measured 60 as
# the net-best operating point (+8 fixed / -2 broken, 98.87%) -- sigmas
# >= ~100 do rescue the second miss (g3-036-S5) yet net LOSE corpus
# accuracy (98.73% at 120, +10/-6) by letting strong lines prop junk
# intersections far beyond their own good tips. See the module
# docstring's 2026-08-27 entry for every configuration measured.
SCORE_SIGMA_ALONG_TIPWARD_MM = 60.0
SCORE_SIGMA_POINT_MM = 4.0
SCORE_LINE_GAIN = 4.0
# A hypothesis supported by >= 2 cameras at or above this level reflects
# genuine cross-camera agreement and outranks ANY single-camera
# hypothesis (whose self-support is structurally free -- a tip lies on
# its own line by construction). This is the same design intent as
# SCORE_CAMERA_CAP below, made effective when line weights are small:
# measured on g3-036-S5, the true two-line intersection (5.3mm from AD's
# tip) could only muster 0.57 + 0.19 of weak-line support and lost to a
# junk tip's free-standing 1.0. Floor 0.1 measured against 0.15/0.2/0.3
# on the full corpus -- every higher floor only lost fixes (98.87% ->
# 98.66/98.66/98.59%) without recovering either regression.
CONSENSUS_MULTI_CAM_MIN_SUPPORT = 0.1
# Each camera's support is capped at 1.0: a single camera agreeing with
# itself (a candidate's tip lies ON its own line by construction, so a
# tip hypothesis always gets its source's full line+point support "for
# free") must never outvote genuine agreement between two cameras.
# Measured: the 363-run's two worst misses (100mm/73mm off) were both
# single-camera junk hypotheses whose uncapped self-support beat a real
# two-camera consensus.
SCORE_CAMERA_CAP = 1.0
# A camera whose best agreement with the winning hypothesis is below
# this contributes nothing but noise -- excluded from the final solve.
CONSENSUS_MIN_CONTRIB = 0.01

# LINE-ONLY admission: a candidate whose tip projects off-board (failed
# tip gate) may still contribute its shaft LINE when the line's quality
# weight clears this bar -- an occluded/overshot tip does not invalidate
# the line constraint (real case: 025-S1 cam2, line 0.94mm from AD's tip
# while its tip read sat 195mm off along the shaft). Guardrails that
# make this safe where the naive version (admit every board-crossing
# line, measured 330/374) was not: the read's junk tip generates no
# hypothesis, anchors no along-segment kernel, and never enters the
# solve -- the line only matters when it agrees with a consensus built
# from properly-gated evidence.
LINE_ONLY_MIN_WEIGHT = 0.15

# Hypothesis-conditioned solve weights: each evidence part enters the
# final solve scaled by its own agreement with the winning consensus
# hypothesis, judged with kernels LOOSER than the scoring ones (2x/2.5x)
# so genuine near-wire ambiguity is refined by the solve, not pre-
# decided by the hypothesis. What this fixes (measured, 025-S1): a
# strong-weight line whose consensus support was 0.034 still entered
# the solve at full weight, dragged the solution 5mm off the winning
# hypothesis, and made the honest third line look like the residual
# outlier -- the consensus KNEW that read disagreed and the solve never
# heard about it.
SOLVE_COND_SIGMA_LINE_MM = 3.0
SOLVE_COND_SIGMA_POINT_MM = 10.0

# Trimmed refit: when one line's residual at the solved point is both
# large in absolute terms AND several times every other line's residual,
# that line is inconsistent with the consensus of everything else --
# drop it and re-solve. (Huber bounds a bad line's pull but never
# removes it: measured on the 369-run, a residual-3.25mm line vs 0.6/0.7
# for the others dragged the answer 1 degree over the 20|1 wire. A
# support-comparison version of this test was tried first and rejected:
# the per-camera cap saturates the good cameras' support on both sides
# of the comparison, leaving the outlier line itself as the deciding
# vote -- circular.)
TRIM_RESIDUAL_MM = 2.0
TRIM_RATIO = 3.0

# Prior-dart-in-visit contamination: same physical signal Apollo and
# Athena already validate on this exact rig (perpendicular pixel
# distance of BOTH candidate ends to the prior throw's own line) -- the
# 15px threshold is the rig-level constant shared by both, reused rather
# than re-guessed from this corpus's too-few contaminated throws.
PRIOR_DART_LINE_MAX_PERP_PX = 15.0
PRIOR_DART_WEIGHT_PENALTY = 0.5


def _perp_dist_px(pt, line_a, line_b) -> float:
    ax, ay = line_a
    bx, by = line_b
    px, py = pt
    dx, dy = bx - ax, by - ay
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    return abs(dx * (py - ay) - dy * (px - ax)) / length


def _prior_dart_suspected(cand: DartCandidate, prior_line) -> bool:
    if prior_line is None:
        return False
    a, b = prior_line
    return (
        _perp_dist_px(cand.tip_px, a, b) <= PRIOR_DART_LINE_MAX_PERP_PX
        and _perp_dist_px(cand.base_px, a, b) <= PRIOR_DART_LINE_MAX_PERP_PX
    )


class _CamRead:
    """One candidate turned into board-plane evidence.

    Line and tip-point evidence are weighted SEPARATELY: a camera can
    contribute a sub-millimetre shaft line while its own tip
    localization is garbage (occluded tip, foreshortening), and vice
    versa (a degraded head-on blob has a usable tip neighbourhood but no
    trustworthy axis)."""

    def __init__(
        self,
        cam: int,
        cand: DartCandidate,
        calib: CameraCalibration,
        contaminated: bool,
    ):
        self.cam = cam
        self.cand = cand
        self.contaminated = contaminated
        self.tip_xy = None
        self.line = None
        self.flipped = False

        tip_xy = pixel_board_xy(cand.tip_px, calib, cam)
        base_xy = pixel_board_xy(cand.base_px, calib, cam)
        # Geometric tip/base pick: the true tip's plane projection lies
        # closer to the camera's own ground position (the flight is above
        # the plane; its ray lands beyond, away from the camera).
        if tip_xy is not None and base_xy is not None:
            gx, gy = camera_ground_xy(calib)
            d_tip = (tip_xy[0] - gx) ** 2 + (tip_xy[1] - gy) ** 2
            d_base = (base_xy[0] - gx) ** 2 + (base_xy[1] - gy) ** 2
            if d_base < d_tip:
                tip_xy, base_xy = base_xy, tip_xy
                self.flipped = True
        self.tip_xy = tip_xy if tip_xy is not None else base_xy
        # The base end's own plane projection -- kept so support()'s
        # along kernel knows which side of the tip anchor is the base
        # side (full penalty) vs beyond the tip (occlusion direction, no
        # penalty). None when the tip projection failed (base_xy is
        # already serving as the tip read) -- support() then falls back
        # to the symmetric penalty.
        self.base_xy = base_xy if tip_xy is not None else None

        # Board-plane shaft line -- only from a real (non-degraded)
        # centerline fit through a genuinely thin structure. The line
        # needs any TWO distinct on-plane projections of the image shaft
        # line; when one end's ray misses the plane (shallow ray at the
        # flight end -- the failure that silently dropped lines on the
        # first corpus run), step part-way along the shaft instead of
        # discarding the whole constraint.
        if not cand.degraded and cand.mean_width_px <= MAX_SHAFT_WIDTH_PX:
            line = board_plane_line(cand.tip_px, cand.base_px, calib, cam)
            if line is None:
                a = np.asarray(cand.tip_px, dtype=np.float64)
                b = np.asarray(cand.base_px, dtype=np.float64)
                for lo, hi in ((0.0, 0.5), (0.0, 0.25), (0.5, 1.0), (0.75, 1.0)):
                    p1 = (float(a[0] + lo * (b[0] - a[0])), float(a[1] + lo * (b[1] - a[1])))
                    p2 = (float(a[0] + hi * (b[0] - a[0])), float(a[1] + hi * (b[1] - a[1])))
                    line = board_plane_line(p1, p2, calib, cam)
                    if line is not None:
                        break
            self.line = line

    @property
    def tip_gate_ok(self) -> bool:
        if self.tip_xy is None:
            return False
        r = (self.tip_xy[0] ** 2 + self.tip_xy[1] ** 2) ** 0.5
        return r <= TIP_GATE_RADIUS_MM

    def _line_quality(self) -> float:
        if self.line is None:
            return 0.0
        length = float(self.cand.shaft_len_px)
        rms = float(self.cand.fit_rms_px)
        if not np.isfinite(rms) or length <= 0.0:
            return 0.0
        w = (length / (length + LINE_WEIGHT_LEN_SOFT_PX)) ** 2
        w *= 1.0 / (1.0 + (rms / LINE_WEIGHT_RMS_SOFT_PX) ** 2)
        return w

    def _point_quality(self) -> float:
        if self.tip_xy is None:
            return 0.0
        return 0.5 if self.cand.degraded else 1.0

    @property
    def line_weight(self) -> float:
        """1/sigma^2-shaped quality weight for this camera's shaft line
        in the SOLVE (0 = no admissible line). Shape and soft knees
        measured corpus-wide; see LINE_WEIGHT_* above. Carries the
        prior-dart contamination penalty; `support()` deliberately does
        not (see its docstring)."""
        w = self._line_quality()
        if self.contaminated:
            w *= PRIOR_DART_WEIGHT_PENALTY
        return w

    @property
    def point_weight(self) -> float:
        """Quality weight for this camera's tip-point read in the SOLVE
        (admission -- the tip gate -- is the caller's decision, kept
        separate so the board-gate fallback can still use off-gate
        tips). Carries the contamination penalty; `support()` does not."""
        w = self._point_quality()
        if self.contaminated:
            w *= PRIOR_DART_WEIGHT_PENALTY
        return w

    def support(self, hyp: np.ndarray) -> float:
        """How much this candidate's evidence agrees with a hypothesis
        entry point (the consensus-scoring kernel; see SCORE_* above).
        Capped at SCORE_CAMERA_CAP.

        Uses the UNPENALIZED quality weights: the prior-dart penalty
        exists to damp suspect evidence's pull on the solved coordinate,
        not to decide WHICH detection is the dart -- measured (g1-042),
        penalizing support handed the consensus to a clean-looking 172px
        junk blob over the real dart that had landed close to the prior
        throw's line in two cameras at once."""
        s = 0.0
        lw = self._line_quality()
        if lw > 0.0:
            perp = perp_distance_mm((float(hyp[0]), float(hyp[1])), self.line)
            term = lw * SCORE_LINE_GAIN * math.exp(
                -0.5 * (perp / SCORE_SIGMA_LINE_MM) ** 2
            )
            # The along-segment anchor is the candidate's own tip -- only
            # meaningful when that tip is itself plausible (gate-passing).
            # A line-only read's tip is known-junk; anchoring on it would
            # veto the line's genuine support.
            #
            # ASYMMETRIC (2026-08-27, see SCORE_SIGMA_ALONG_MM's comment):
            # full penalty on the base side of the tip anchor (a dart tip
            # cannot be mid-shaft), none beyond the tip end (where the
            # true tip really sits when prior-dart occlusion shortened
            # the visible shaft -- measured 100-180mm beyond on both
            # third-dart misses).
            if self.tip_xy is not None and self.tip_gate_ok:
                d = self.line[1]
                t_hyp = float(d @ (np.asarray(hyp) - self.line[0]))
                t_tip = float(d @ (np.asarray(self.tip_xy) - self.line[0]))
                delta = t_hyp - t_tip
                toward_base = 0.0
                if self.base_xy is not None:
                    t_base = float(
                        d @ (np.asarray(self.base_xy) - self.line[0])
                    )
                    toward_base = t_base - t_tip
                if toward_base == 0.0 or delta * toward_base > 0.0:
                    # Base side (or base direction unknown): full
                    # leak-killing penalty.
                    term *= math.exp(
                        -0.5 * (abs(delta) / SCORE_SIGMA_ALONG_MM) ** 2
                    )
                else:
                    # Beyond the tip end: the occlusion direction --
                    # loose but bounded.
                    term *= math.exp(
                        -0.5
                        * (abs(delta) / SCORE_SIGMA_ALONG_TIPWARD_MM) ** 2
                    )
            s += term
        pw = self._point_quality()
        if pw > 0.0 and self.tip_xy is not None:
            d = math.hypot(hyp[0] - self.tip_xy[0], hyp[1] - self.tip_xy[1])
            s += pw * math.exp(-0.5 * (d / SCORE_SIGMA_POINT_MM) ** 2)
        return min(s, SCORE_CAMERA_CAP)


def _admissible(read: _CamRead, require_gate: bool) -> bool:
    """The admission rule `score()` applies per candidate (factored out
    so the drop-reason diagnostics below report the SAME decision the
    engine actually made, never a reimplementation that can drift)."""
    if read.tip_xy is None and read.line is None:
        return False
    if require_gate and not read.tip_gate_ok and not (
        read.line is not None and read.line_weight >= LINE_ONLY_MIN_WEIGHT
    ):
        return False
    return True


def _candidate_record(
    read: _CamRead, require_gate: bool, hypothesis: np.ndarray | None
) -> dict:
    """Per-candidate drop-reason record ('per rejected
    camera, which gate fired, on what value, against what threshold --
    and where were the candidates it threw away'). Written for EVERY
    candidate on EVERY camera, used or not, so a single-camera answer is
    diagnosable from the package alone."""
    tip_r_mm = (
        float(math.hypot(read.tip_xy[0], read.tip_xy[1]))
        if read.tip_xy is not None else None
    )
    admitted = _admissible(read, require_gate)
    rejection = None
    if not admitted:
        if read.tip_xy is None and read.line is None:
            rejection = "no_board_plane_projection"
        else:
            rejection = (
                f"tip_gate_failed (r {tip_r_mm:.1f}mm > "
                f"{TIP_GATE_RADIUS_MM:.1f}mm) and line "
                + (
                    f"weight {read.line_weight:.4f} < {LINE_ONLY_MIN_WEIGHT}"
                    if read.line is not None else "absent"
                )
            )
    return {
        "tip_px": read.cand.tip_px,
        "base_px": read.cand.base_px,
        "degraded": read.cand.degraded,
        "centerline_failure": read.cand.diagnostics.get("centerline_failure"),
        "area_px": read.cand.area_px,
        "shaft_len_px": read.cand.shaft_len_px,
        "mean_width_px": read.cand.mean_width_px,
        "fit_rms_px": read.cand.fit_rms_px,
        "end_flipped": read.flipped,
        "tip_xy_mm": read.tip_xy,
        "tip_r_mm": tip_r_mm,
        "tip_gate_ok": read.tip_gate_ok,
        "has_line": read.line is not None,
        "line_weight": read.line_weight,
        "point_weight": read.point_weight,
        "prior_dart_contamination_suspected": read.contaminated,
        "admitted": admitted,
        "rejection": rejection,
        "consensus_support": (
            read.support(hypothesis) if hypothesis is not None else None
        ),
    }


def _record_drop_diagnostics(
    per_camera: dict[int, dict],
    all_reads: dict[int, list[_CamRead]],
    reads_by_cam: dict[int, list[_CamRead]],
    hypothesis: np.ndarray | None,
    require_gate: bool,
) -> None:
    """Fill per-camera drop-stage/candidate diagnostics on every return
    path. Symmetric counterpart to the 18-field entry a USED camera has
    always had: a dropped camera now records WHERE it fell out
    (`no_candidates` / `admission` / `consensus`) plus a full gate-value
    record for each candidate it saw. All existing keys are unchanged --
    purely additive (package-schema discipline)."""
    for cam, entry in per_camera.items():
        reads = all_reads.get(cam, [])
        entry["candidates"] = [
            _candidate_record(r, require_gate, hypothesis) for r in reads
        ]
        if entry.get("used"):
            continue
        admitted = reads_by_cam.get(cam, [])
        if not reads:
            entry["drop_stage"] = "no_candidates"
        elif not admitted:
            entry["drop_stage"] = "admission"
        else:
            entry["drop_stage"] = "consensus"
            entry["best_consensus_support"] = (
                max(r.support(hypothesis) for r in admitted)
                if hypothesis is not None else None
            )


# Self-describing gate thresholds, written into every result's
# diagnostics so a package's candidate records can be read against the
# exact bars that applied when it was scored (they have changed before
# and will again).
_GATE_THRESHOLDS = {
    "tip_gate_radius_mm": TIP_GATE_RADIUS_MM,
    "line_only_min_weight": LINE_ONLY_MIN_WEIGHT,
    "consensus_min_contrib": CONSENSUS_MIN_CONTRIB,
    "max_shaft_width_px": MAX_SHAFT_WIDTH_PX,
}


def _consensus_select(
    reads_by_cam: dict[int, list[_CamRead]],
    allow_ungated_tips: bool = False,
) -> tuple[list[_CamRead], np.ndarray | None, int]:
    """Choose ONE candidate per camera by cross-camera consensus.

    Hypotheses: every pairwise intersection of two candidates' lines
    from DIFFERENT cameras (the allowed 2D pairwise evidence), plus every
    admitted tip point. Each is scored by the summed best-candidate
    support per camera; the winner picks each camera's candidate (or
    drops the camera when nothing it saw agrees). Returns (selected
    reads, winning hypothesis, n_hypotheses).

    Two 2026-08-27 changes (the g1-003-S5/g3-036-S5 single-camera-junk
    fix; measured numbers in the module docstring's dated entry):

    - A candidate flagged as prior-dart contamination does not GENERATE
      a tip hypothesis (as a hypothesis generator it is most likely the
      prior dart -- on g1-003-S5 the winning "dart" was a 30px stub on
      the prior throw's own line). It still supports other hypotheses
      and still enters the solve, preserving the measured g1-042
      lesson (contamination must not decide which detection is the
      dart AMONG hypotheses -- here it only stops nominating itself).
      If no hypothesis forms at all without them, they are re-admitted
      rather than going silent.
    - Hypotheses with genuine multi-camera agreement (>= 2 cameras at
      CONSENSUS_MULTI_CAM_MIN_SUPPORT or better) outrank every
      single-camera hypothesis regardless of raw score -- see
      CONSENSUS_MULTI_CAM_MIN_SUPPORT's comment."""
    all_reads = [r for reads in reads_by_cam.values() for r in reads]
    hyps: list[np.ndarray] = []
    cams = sorted(reads_by_cam)
    for i, cam_a in enumerate(cams):
        for cam_b in cams[i + 1:]:
            for ra in reads_by_cam[cam_a]:
                if ra.line is None or ra.line_weight <= 0.0:
                    continue
                # NOTE: contaminated reads' LINES deliberately still
                # generate intersections (unlike their tips, below) --
                # suppressing them was measured to replay the g1-042
                # trap: a real dart contaminated in-camera loses its own
                # intersections and hands the win to junk.
                for rb in reads_by_cam[cam_b]:
                    if rb.line is None or rb.line_weight <= 0.0:
                        continue
                    x = line_intersection(
                        ra.line[0], ra.line[1], rb.line[0], rb.line[1]
                    )
                    if x is None:
                        continue
                    if float(np.hypot(x[0], x[1])) > TIP_GATE_RADIUS_MM:
                        continue
                    hyps.append(x)

    def _tip_hyps(include_contaminated: bool) -> list[np.ndarray]:
        return [
            np.asarray(r.tip_xy, dtype=np.float64)
            for r in all_reads
            if r.tip_xy is not None
            and (r.tip_gate_ok or allow_ungated_tips)
            and (include_contaminated or not r.contaminated)
        ]

    tip_hyps = _tip_hyps(include_contaminated=False)
    if not hyps and not tip_hyps:
        # Nothing left to vote on without the contaminated tips --
        # re-admit them rather than going silent (same never-a-blank
        # posture as the board-gate fallback).
        tip_hyps = _tip_hyps(include_contaminated=True)
    hyps.extend(tip_hyps)
    if not hyps:
        return [], None, 0

    scored = []
    for hyp in hyps:
        cam_supports = [
            max(r.support(hyp) for r in reads)
            for reads in reads_by_cam.values()
            if reads
        ]
        n_strong = sum(
            1 for s in cam_supports if s >= CONSENSUS_MULTI_CAM_MIN_SUPPORT
        )
        scored.append((hyp, n_strong, sum(cam_supports)))
    best_hyp = None
    best_key = (-1, -1.0)
    for hyp, n_strong, raw in scored:
        # Tier first (genuine multi-camera agreement beats any
        # single-camera hypothesis), raw score second. A raw-score
        # margin condition on the promotion was measured and rejected:
        # mathematically contradictory on real throws (g3-036 needs
        # promotion at 0.55x the raw leader, 069-S9 needs suppression
        # at 0.62x).
        key = (1 if n_strong >= 2 else 0, raw)
        if key > best_key:
            best_key = key
            best_hyp = hyp

    selected: list[_CamRead] = []
    for cam in cams:
        reads = reads_by_cam[cam]
        if not reads:
            continue
        best = max(reads, key=lambda r: r.support(best_hyp))
        if best.support(best_hyp) >= CONSENSUS_MIN_CONTRIB:
            selected.append(best)
    return selected, best_hyp, len(hyps)


class AresEngine:
    """The registry entry named "Ares" -- see module docstring."""

    name = "Ares"
    # What a caller-shared `opendarts.imageops.DiffCrop` must satisfy for
    # this engine to use it -- see `detect_candidates()`'s `precomputed`.
    precompute_requirements = PRECOMPUTE_REQUIREMENTS

    def score(
        self,
        bg_images: dict[int, np.ndarray],
        frame_images: dict[int, np.ndarray],
        calibration: dict[int, CameraCalibration],
        *,
        prior_dart_line_px: dict | None = None,
        precomputed: "dict | None" = None,
    ) -> EngineResult:
        cams = sorted(set(bg_images) & set(frame_images) & set(calibration))
        per_camera: dict[int, dict] = {}
        detections: dict[int, list[DartCandidate]] = {}

        for cam in cams:
            # `precomputed` is only passed when there is a bundle for this
            # camera -- test doubles keep the plain two-argument signature.
            pc_kwargs = {"precomputed": precomputed[cam]} if precomputed and cam in precomputed else {}
            cands = detect_candidates(bg_images[cam], frame_images[cam], **pc_kwargs)
            detections[cam] = cands
            per_camera[cam] = {
                "n_candidates": len(cands),
                "used": False,
            }

        # Every candidate becomes a _CamRead exactly once -- construction
        # is gate-independent, so admission (below) is a pure filter over
        # this pool and the drop diagnostics describe the same objects
        # the engine actually voted with.
        all_reads: dict[int, list[_CamRead]] = {}
        for cam in cams:
            prior_line = (
                prior_dart_line_px.get(cam) if prior_dart_line_px else None
            )
            all_reads[cam] = [
                _CamRead(
                    cam, cand, calibration[cam],
                    _prior_dart_suspected(cand, prior_line),
                )
                for cand in detections[cam]
            ]

        def _collect(require_gate: bool) -> dict[int, list[_CamRead]]:
            by_cam: dict[int, list[_CamRead]] = {}
            for cam in cams:
                reads = [
                    r for r in all_reads[cam] if _admissible(r, require_gate)
                ]
                if reads:
                    by_cam[cam] = reads
            return by_cam

        reads_by_cam = _collect(require_gate=True)
        gate_fallback = False
        if not reads_by_cam:
            # Never a silent no-answer: if no camera's candidate passes
            # the board gate, use the ungated candidates -- a real,
            # honestly-bad answer (often correctly "outside") instead of
            # a blank.
            reads_by_cam = _collect(require_gate=False)
            gate_fallback = bool(reads_by_cam)

        if not reads_by_cam:
            _record_drop_diagnostics(
                per_camera, all_reads, reads_by_cam, None, not gate_fallback
            )
            return EngineResult(
                ok=False,
                sector=None,
                ring=None,
                board_xy_mm=None,
                reason="no camera produced a usable dart detection",
                diagnostics={
                    "per_camera": per_camera,
                    "gate_thresholds": _GATE_THRESHOLDS,
                },
                confidence=0.0,
            )

        selected, hypothesis, n_hyps = _consensus_select(
            reads_by_cam, allow_ungated_tips=gate_fallback
        )
        if not selected and not gate_fallback:
            # Possible when only line-only reads were admitted (no
            # gate-passing tip anywhere): no hypothesis could form. Fall
            # back to the ungated pool rather than going silent.
            reads_by_cam = _collect(require_gate=False)
            gate_fallback = bool(reads_by_cam)
            if reads_by_cam:
                selected, hypothesis, n_hyps = _consensus_select(
                    reads_by_cam, allow_ungated_tips=True
                )
        if not selected:
            _record_drop_diagnostics(
                per_camera, all_reads, reads_by_cam, hypothesis,
                not gate_fallback,
            )
            return EngineResult(
                ok=False,
                sector=None,
                ring=None,
                board_xy_mm=None,
                reason="no consensus among camera detections",
                diagnostics={
                    "per_camera": per_camera,
                    "gate_thresholds": _GATE_THRESHOLDS,
                },
                confidence=0.0,
            )

        for read in selected:
            entry = per_camera[read.cam]
            entry.update({
                "used": True,
                "tip_px": read.cand.tip_px,
                "base_px": read.cand.base_px,
                "end_flipped": read.flipped,
                "tip_xy_mm": read.tip_xy,
                "has_line": read.line is not None,
                "degraded": read.cand.degraded,
                "area_px": read.cand.area_px,
                "shaft_len_px": read.cand.shaft_len_px,
                "mean_width_px": read.cand.mean_width_px,
                "fit_rms_px": read.cand.fit_rms_px,
                "tip_gate_ok": read.tip_gate_ok,
                "line_weight": read.line_weight,
                "point_weight": read.point_weight,
                "consensus_support": read.support(hypothesis),
                "prior_dart_contamination_suspected": read.contaminated,
            })
            if read.line is not None:
                entry["line_dir"] = (float(read.line[1][0]), float(read.line[1][1]))

        _record_drop_diagnostics(
            per_camera, all_reads, reads_by_cam, hypothesis, not gate_fallback
        )

        hyp_pt = (float(hypothesis[0]), float(hypothesis[1]))
        lines = []
        for r in selected:
            if r.line is None or r.line_weight <= 0.0:
                continue
            perp = perp_distance_mm(hyp_pt, r.line)
            cond = math.exp(-0.5 * (perp / SOLVE_COND_SIGMA_LINE_MM) ** 2)
            lines.append(LineObservation(
                point=np.asarray(r.line[0], dtype=np.float64),
                direction=np.asarray(r.line[1], dtype=np.float64),
                weight=r.line_weight * cond,
                cam=r.cam,
            ))
        points = []
        for r in selected:
            if (
                r.tip_xy is None
                or r.point_weight <= 0.0
                or not (r.tip_gate_ok or gate_fallback)
            ):
                continue
            d = math.hypot(hyp_pt[0] - r.tip_xy[0], hyp_pt[1] - r.tip_xy[1])
            cond = math.exp(-0.5 * (d / SOLVE_COND_SIGMA_POINT_MM) ** 2)
            points.append(PointObservation(
                xy=np.asarray(r.tip_xy, dtype=np.float64),
                weight=r.point_weight * cond,
                cam=r.cam,
            ))

        fused = fuse(lines, points, init_xy=hypothesis)
        if fused is None:
            return EngineResult(
                ok=False,
                sector=None,
                ring=None,
                board_xy_mm=None,
                reason="fusion produced no estimate (no usable evidence)",
                diagnostics={
                    "per_camera": per_camera,
                    "gate_thresholds": _GATE_THRESHOLDS,
                },
                confidence=0.0,
            )

        trimmed_cam = None
        if fused.n_lines_used >= 2 and fused.per_line_residual_mm:
            worst_cam = max(
                fused.per_line_residual_mm,
                key=lambda c: fused.per_line_residual_mm[c],
            )
            worst_r = fused.per_line_residual_mm[worst_cam]
            others = [
                v for c, v in fused.per_line_residual_mm.items()
                if c != worst_cam
            ]
            if worst_r > TRIM_RESIDUAL_MM and worst_r > TRIM_RATIO * max(others):
                fused_trim = fuse(
                    [ln for ln in lines if ln.cam != worst_cam],
                    points,
                    init_xy=hypothesis,
                )
                if fused_trim is not None:
                    fused = fused_trim
                    trimmed_cam = worst_cam

        # n_cameras_used, 2026-09-02 -- same semantic Athena's own
        # `n_cameras_used` field was investigated and confirmed to mean
        # (see that engine's dated 2026-09-02 comment at `camera_reads =
        # qualifying if qualifying else reads`): "cameras whose data
        # actually reached and influenced the final combined answer,"
        # not merely "consensus-selected" (that weaker sense is what
        # `per_camera[cam]["used"]` already tracks here -- set for every
        # `read.cam` in `selected` above, BEFORE the residual-outlier
        # line trim below can remove a camera's own LINE contribution
        # from the actual solve). Computed from the exact `lines`/
        # `points` observation lists that fed the `fused` result this
        # function is about to return -- i.e. post-trim, mirroring
        # Athena's own post-steepness-floor `camera_reads` -- not
        # re-derived from `selected` or `per_camera`, so it can never
        # drift from what `fuse()` actually solved with. A camera that
        # cleared consensus but contributes neither a line (excluded by
        # the trim, or none was ever admitted) nor a point (ungated/
        # zero-weight/gate-failed) to this final solve is correctly
        # NOT counted -- it was selected, but its data never reached the
        # answer.
        final_line_cams = {
            ln.cam for ln in lines if trimmed_cam is None or ln.cam != trimmed_cam
        }
        final_point_cams = {pt.cam for pt in points}
        n_cameras_used = len(final_line_cams | final_point_cams)

        x_mm, y_mm = fused.xy_mm
        sector, ring = sector_ring_for_point(x_mm, y_mm)

        reason = (
            f"consensus of {fused.n_lines_used} shaft line(s) + "
            f"{fused.n_points_used} tip point(s) from {len(selected)} "
            f"camera(s), {n_hyps} hypotheses"
        )
        if trimmed_cam is not None:
            reason += f"; cam{trimmed_cam} line trimmed as residual outlier"
        if gate_fallback:
            reason = "board-gate fallback (no candidate passed): " + reason

        return EngineResult(
            ok=True,
            sector=sector,
            ring=ring,
            board_xy_mm=(x_mm, y_mm),
            reason=reason,
            diagnostics={
                "per_camera": per_camera,
                "gate_thresholds": _GATE_THRESHOLDS,
                "fusion_method": fused.method,
                "n_cameras_used": n_cameras_used,
                "n_lines_used": fused.n_lines_used,
                "n_points_used": fused.n_points_used,
                "n_hypotheses": n_hyps,
                "trimmed_cam": trimmed_cam,
                "per_line_residual_mm": fused.per_line_residual_mm,
                "condition_ratio": fused.condition_ratio,
                "spread_mm": fused.spread_mm,
                "gate_fallback_used": gate_fallback,
            },
            confidence=None,
        )
