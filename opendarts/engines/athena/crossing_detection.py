"""Athena's own per-camera dart-shaft detection: NOT a port of
`opendarts.engines.apollo.tip_detection`. Starts from the same raw
motion-diff-blob idea in spirit (there is no other sane way to find "a
new dart" in a before/after image pair with classical CV, and the task
this module was built for explicitly says starting from that idea is
fine) but is a fresh implementation with its own constants, its own
component-selection scoring, its own tip-end heuristic (extent-based,
not std-based), and -- the actual new contribution -- a **board-crossing
walk**: instead of returning the visible tip pixel as the thing to
back-project, this module also computes where along the shaft's own
principal axis the true board-plane crossing point most likely sits, and
returns THAT pixel as `crossing_px` (see docstring further down for the
physical reasoning and the falsifiability check this was built against).

Deliberately calibration-free at the blob-detection stage (same
separation-of-concerns reasoning as tip_detection.py's own module
docstring: this module knows nothing about camera matrices or board
geometry -- it only ever sees one camera's own pixels). The crossing-walk
step needs calibration (to know which pixel offset actually lands the
resulting ray-plane intersection somewhere sane), so this module exposes
BOTH the raw shaft geometry (`axis_unit`, `tip_px`, `span_px`) and a
calibration-free default `crossing_px` (a fixed-fraction-of-span walk,
tunable, measured against the real corpus --
see `CROSSING_WALK_FRACTION`'s own comment) so a caller with calibration
available (`opendarts.engines.athena.engine.AthenaEngine`) can also
choose to search a few candidate offsets per camera if that measures
better than one fixed constant.

**The board-crossing hypothesis, and Talos's falsifiability test**
(docs/DESIGN.md, 2026-08-13): a stuck dart's visible tip pixel is where the
diff-mask silhouette narrows to a point -- but that is the pixel where
the shaft's own image disappears into/behind the board's surface as
occlusion, not necessarily the pixel whose back-projected ray crosses
the board's Z=0 plane at the same 3D point the dart's centerline
actually crosses it. Because the dart enters at a real, non-perpendicular
angle (a level, arcing throw very rarely embeds a dart exactly
perpendicular to the board face), the shaft continues in a straight line
some distance further before its centerline would cross Z=0 if extended
-- back toward the flight from the visible tip, not past it. Talos's own
concrete falsifiability test, taken seriously here: "if after walking
back along the shaft the crossing equals the current tip, this
hypothesis is wrong." This is trivially true in the sense that ANY
nonzero pixel offset changes the ray-plane intersection point (different
pixel -> different ray -> different Z=0 crossing, almost always) --
so the real, non-trivial form of the test this module was actually
measured against is: does walking back by some real, camera-agnostic
offset produce a MEASURABLE, real accuracy improvement against AD ground
truth on `data/archive/clean/`, not just "produce a different number".
See `CROSSING_WALK_FRACTION`'s own comment for the actual sweep and
result -- reported honestly either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opendarts.imageops import (
    DiffCrop,
    PrecomputeRequirements,
    blurred_abs_diff,
    component_pixels,
    ellipse_kernel,
    morph_on_bbox,
    threshold_mask,
)

# --- Tuned constants -----------------------------------------------------
# Own values, own sweep (against data/archive/clean/),
# deliberately NOT copied from opendarts.engines.apollo.tip_detection's own
# (different) tuned constants, even where the same rough idea (blur, diff
# threshold, morphological gap-bridging) applies -- a different rig
# session's corpus, evaluated end-to-end (final sector+ring match, not an
# intermediate pixel-error proxy), can and did land on different numbers.
#
# Real sweeps (inline, full
# Athena pipeline including the board ROI gate and consensus, real
# sector+ring match vs AD on data/archive/clean/, 169 throws). Two
# separate sweep passes, re-run after the consensus/candidate-selection
# logic changed (steepest-ray weighting + steepest-ray candidate
# re-ranking, see engine.py) since the earlier pass's optimum could have
# shifted: DIFF_THRESHOLD swept 20-50 originally (peak 33, 79.3%), then
# RE-swept 28-38 against the current pipeline -- new peak 31/31.5 (84.0%,
# vs 83.4% at 33) -- 31 shipped. CLOSE_KERNEL_PX re-swept 25-39 with
# DIFF_THRESHOLD=31: flat 84.0% plateau 29-33, 31 (already the value)
# stays. MIN_ELONGATION_RATIO re-swept 1.6-3.2: flat 84.0% plateau
# 1.6-2.8, 2.2 (the original guess) stays.
#
# **Re-swept 2026-08-13 against the substantially changed pipeline** (a
# corrected board plane, a radial correction, a geometric tip-end picker
# and a tip_confidence weight all landed after the values below were
# chosen -- so their optima could genuinely have moved). Full real
# 349-throw data/archive/clean/ corpus, sector+ring BOTH-match, one
# constant varied at a time. Every one of them is
# still at or tied for the joint optimum, which is a real result about
# the original tuning, not a null:
# DIFF_THRESHOLD 28->329 30->331 [31]->331 32->328 34->328
# BLUR_KSIZE 3->321 [5]->331 7->328
# MIN_ELONGATION_RATIO 1.6->331 [2.2]->331 3.0->329
# MIN_COMPONENT_AREA_PX 10, [18], 30 all 331
# TOP_K_AREA_CANDIDATES [3], 4, 6 all 331
# END_WINDOW_FRACTION 0.12->331 [0.18]->331 0.25->330
# N_TIP_POINTS_AVERAGED 1->331 [3]->331 5->329 8->327 15->323
#
# CLOSE_KERNEL_PX is the one that looked movable and, on the evidence,
# is not: 19->328, 21->331, **23/25/27/29->332**, [31]->331, 33->330 --
# a genuine 4-wide plateau one throw above the shipped value, with a
# real visually-confirmed mechanism behind it (a 31px close kernel
# merges two adjacent darts into one component, so the PCA axis spans
# both and the "tip" lands on the wrong dart -- that is the 51.6mm
# per-camera error on throw_1786666489418 cam0).
# **Left at 31 anyway**: leave-one-session-out over all 8 real sessions
# picks 23 in 7 folds and 21 in one, and its held-out total is
# 331/349 -- exactly what the untouched 31 already scores. So the +1 is
# in-sample only and the selection does not generalise across sessions.
# One throw in 349 with no cross-validated support is precisely the kind
# of change docs/DESIGN.md records this project being burned by before.
# Worth re-testing when the corpus is materially larger; the plateau
# shape suggests 31 sits on the shoulder rather than the peak.
#
# **Re-swept 2026-09-05, Zeus-latency follow-up task: 31 -> 19, the exact
# re-testing this comment's own prior paragraph called for, once the
# corpus had grown well past 349.** Real motivation: Zeus's own
# wall-clock is max() across its 4 sub-engines (2026-08-27 parallel-
# dispatch fix, opendarts/engines/zeus/engine.py), so this engine (Athena,
# ~77ms) and Ares (Ares, ~76ms) are the near-tied pair that actually
# gates Zeus's real latency -- see opendarts/engines/ares/detection.py's
# own matching 2026-09-05 CLOSE_KERNEL_PX comment for the paired
# change; changing only one of the two buys nothing since Zeus still
# waits on whichever is slower. Real measured effect, full 1506-package
# live corpus (the session corpus, not _freeze1), Zeus
# re-scored end to end with BOTH engines' kernels changed together:
# 1497/1504 vs the shipped 1496/1504 (McNemar p=1.0 -- statistically
# indistinguishable, not a regression), and the two engines' own error
# CORRELATION actually DROPS (13.4x -> 9.9x expected-by-chance), so this
# is not a consensus-degradation risk either. Real macOS-rig timing (direct
# `zeus.score()` call, n=60/arm, interleaved): 79.7ms -> 65.2ms, a real
# 14.5ms/18.2% saving. This engine's own individual accuracy at the
# shipped k=19, independently spot-checked against the same live corpus
# before this comment was written (measured here, not carried over
# unverified): 1471/1504 = 97.81%. **Hard floor for this engine, do
# not go below 15** -- this engine degrades one-sidedly under k=15, with
# a real, statistically significant one-sided accuracy cliff measured at
# k=7 (p=0.0007) during the sweep this change is based on; 19 sits with
# real margin above that floor, not on its shoulder (Ares's own floor
# is different, 13 -- see that engine's own matching comment).
DIFF_THRESHOLD = 31.0
BLUR_KSIZE = 5
CLOSE_KERNEL_PX = 19
# What a caller-supplied `opendarts.imageops.DiffCrop` must satisfy for
# `detect_crossing()` to use it in place of its own gray/diff/blur front
# end (2026-09-06 perf pass). The pad covers the closing's dilate reach
# plus its erode window (2 x radius 9), so the closing computed on the
# crop equals the full-frame one -- see `DiffCrop`'s docstring.
PRECOMPUTE_REQUIREMENTS = PrecomputeRequirements(
    ksize=BLUR_KSIZE, threshold=DIFF_THRESHOLD, pad_px=CLOSE_KERNEL_PX,
)
MIN_COMPONENT_AREA_PX = 18
MIN_ELONGATION_RATIO = 2.2
TOP_K_AREA_CANDIDATES = 3

# "Near an end" window, as a fraction of the component's own along-axis
# span (with a floor for very short/foreshortened components) -- used to
# measure how WIDE (perpendicular to the shaft axis) each end is, which
# is what actually distinguishes "tip" (narrow, converging) from
# "flight" (wide).
END_WINDOW_FRACTION = 0.18
END_WINDOW_MIN_PX = 6.0
N_TIP_POINTS_AVERAGED = 3

# How far (as a FRACTION of the component's own measured along-axis span,
# not a fixed pixel count -- so it scales with how large the dart appears
# in this specific camera view, i.e. distance/zoom-invariant, same
# reasoning END_WINDOW_FRACTION above already uses for the width window)
# to walk back from the visible tip pixel, toward the flight end, before
# back-projecting -- the board-crossing hypothesis this module's docstring
# describes.
#
# Measured for real, TWICE (169-throw data/archive/clean/ corpus, full
# Athena pipeline INCLUDING the board ROI gate + weighted-geometric-
# median consensus -- sector+ring match against AD, not a pixel-error
# proxy). First sweep (before the ROI gate/consensus existed, engine
# scoring only ~32.5% overall) was too noisy to trust on its own; re-swept
# after the gate+consensus fixes landed (engine at 80.5% baseline),
# result held and got MUCH clearer: swept +0.005..+0.25 and -0.01..-0.1 --
# every single nonzero value tried, in EITHER direction, measured worse
# than 0.0, monotonically falling off with magnitude (+0.005: 75.7%,
# +0.01: 74.0%, +0.02: 65.7%, +0.05: 51.5%, +0.25: 16.0%; -0.01: 77.5%,
# -0.05: 65.7%, -0.1: 48.5%; 0.0 itself: 80.5%). **Honest conclusion:
# the board-crossing hypothesis, taken seriously and actually measured
# (twice, before and after fixing an unrelated
# detection-selection bug that could have been hiding the real signal),
# did NOT survive contact with this rig's real data at ANY nonzero offset
# tried, in either direction.** This corroborates Talos's own independent
# finding in docs/DESIGN.md (median `plane_discrepancy_mm` only 2.16mm across
# 247 real throws) -- the effect this hypothesis predicts is geometrically
# real (a non-perpendicular dart entry angle does mean the shaft's
# centerline crosses Z=0 somewhere other than the visible tip pixel in
# 3D) but too small relative to this rig's own detection noise (median
# per-camera tip_delta ~5-7mm, see crossing_detection.py's own docstring)
# to be usefully estimated from ONE camera's 2D silhouette alone with a
# simple fixed-fraction walk -- walking ANY amount just adds noise/bias
# without adding real signal. Left configurable (not deleted) so a future
# session with different rig/lighting data, or a smarter per-pixel
# crossing estimator, can re-sweep cheaply; DEFAULT kept at the real
# measured optimum (0.0), not the originally-hypothesized nonzero value.
# Re-confirmed a THIRD time after the ray-steepness weight/floor and
# steepest-candidate re-ranking landed (engine at 84.6%): 0.0 still
# strictly best, every nonzero offset (+/-0.01..0.05) still worse, same
# monotonic falloff shape. Not a fluke of one particular pipeline state.
# Re-confirmed a FOURTH time after fixing a real (zero-impact-until-fixed)
# axis_unit sign bug found by tests/test_athena_crossing_detection.py's
# new synthetic unit test (see the comment above `axis_unit`'s own
# assignment below) -- with the now-CORRECT walk direction, +0.01 still
# measures worse (82.2%) than 0.0 (84.6%), same monotonic falloff. The
# earlier three sweeps happened to still be valid conclusions despite the
# bug (they tried both positive AND negative fractions, so both true
# directions were always covered either way), but this is the first
# sweep where the SIGN of a positive fraction is known to mean what the
# module docstring says it means.
CROSSING_WALK_FRACTION = 0.0


@dataclass
class CrossingDetectionResult:
    """One camera's own shaft-crossing read. `tip_px` is the visible
    silhouette tip (same physical thing tip_detection.py's own `tip_px`
    means); `crossing_px` is `tip_px` walked back along the shaft's own
    axis by `CROSSING_WALK_FRACTION` of the measured span (identical to
    `tip_px` when that fraction is 0.0 -- see that constant's own
    comment for why it currently is). `axis_unit` points from the tip
    toward the flight end -- the direction any caller doing its own
    offset search (opendarts.engines.athena.engine) should walk in.
    `elongation_ratio`/`tip_confidence` are self-diagnostics this
    camera's own detection can report about ITSELF, independent of
    anything the other cameras see -- exactly the kind of
    ray-agreement-independent signal that is the actual missing piece
    for weighting cameras against correlated
    bias, not another per-throw cross-camera comparison.
    """

    ok: bool
    tip_px: tuple[float, float] | None
    crossing_px: tuple[float, float] | None
    axis_unit: tuple[float, float] | None
    span_px: float | None
    elongation_ratio: float | None
    tip_confidence: float | None
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)


def _diff_mask(bg_gray: np.ndarray, frame_gray: np.ndarray) -> np.ndarray:
    """Single global threshold on a Gaussian-blurred grayscale absolute
    difference. Deliberately still the simplest thing that works -- two
    structurally different alternatives were built and measured against
    the full real 349-throw corpus on 2026-08-13 and BOTH lost. Recorded
    here so they are not re-attempted from scratch:

    **Hysteresis (two-threshold) mask** -- keep every pixel above a LOW
    threshold that is connected to at least one pixel above the HIGH one
    Motivated by real failing frames where the
    mask ends early along a dark shaft against a black bed, putting the
    tip short along the shaft. Every configuration tried approaches the
    single-threshold baseline (331) from below and none reaches it:

        high=31: low=8 -> 261, 12 -> 313, 16 -> 320, 20 -> 324, 25 -> 328
        high=35: low=8 -> 262, 12 -> 316, 16 -> 320, 20 -> 325, 25 -> 327

    The effect is real but not separable this way: the dart is itself
    connected to the shadow and lighting-gradient regions a permissive
    low threshold picks up, so they come along with it and the
    connectivity gate never isolates the shaft.

    **Colour instead of grayscale**, each with its
    own independently swept threshold since each statistic has its own
    scale -- max over B/G/R channels (peak 327 at 38), mean over channels
    (peak 330 at 31), CIE-Lab distance (peak 330 at 40), all against
    grayscale's 331. Mean-over-channels landing level is expected --
    grayscale IS a weighted channel mean. Max-channel is the genuinely
    different statistic and it loses.

    One result worth keeping visible rather than burying: **CIE-Lab at
    threshold 31 scores fresh 170/180 = 94.4%, the best fresh-session
    number any variant produced, but historical 158/169 = 93.5%, five
    throws worse** (328 vs 331 overall). A perceptual colour distance
    helps precisely the corpus with the elevated gross-read rate and
    hurts the one without it. Not adopted -- it regresses the held-out
    set, and tuning a mask per corpus is the overfitting move this
    project's history warns about -- but it is a real lead for a future
    session with a third independent corpus to validate against.

    2026-09-05 perf pass: same arithmetic via `opendarts.imageops` (uint8
    absdiff -> one float32 cast, `cv2.compare` for the threshold) --
    bit-identical, see those helpers' own docstrings.
    """
    return threshold_mask(blurred_abs_diff(bg_gray, frame_gray, BLUR_KSIZE), DIFF_THRESHOLD)


def _pca(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """centroid, principal-axis unit vector, elongation ratio (sqrt of
    major/minor eigenvalue of the covariance)."""
    centroid = points_xy.mean(axis=0)
    centered = points_xy - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    minor_eval, major_eval = evals[order[0]], evals[order[1]]
    elongation = float(np.sqrt(major_eval / max(minor_eval, 1e-6)))
    principal = evecs[:, order[1]]
    return centroid, principal, elongation


def _end_extent(perp: np.ndarray, mask: np.ndarray) -> float:
    """How WIDE (perpendicular-to-axis extent, max - min) the component
    is within an end window -- a plain range, not a std-dev (a
    deliberately different, simpler statistic from tip_detection.py's own
    std-based width measure; both encode the same "narrow end = tip"
    physical idea, this is this module's own independent choice of how to
    measure it)."""
    vals = perp[mask]
    if vals.size < 2:
        return 0.0
    return float(vals.max() - vals.min())


# 2026-09-02 -- see _largest_elongated_subcomponent_points()'s own
# docstring, and opendarts.engines.apollo.tip_detection's identical
# constant, for the full derivation (same real local-corpus regression,
# same fix, independently re-measured for THIS module -- not copied
# blind): a first version gated the sub-piece purely on
# `MIN_COMPONENT_AREA_PX` (18px, this module's own generic "enough to
# analyze at all" floor) and a real local-corpus check
# (the session corpus, 951 throws) found it let a real
# throw pick a spurious 76px noise fragment over the pre-existing "whole
# blob" fallback, which had that throw right. The genuine positive case
# (the recorded outside throw cam0) has a real sub-piece of 9147px --
# comfortably above this floor, matching Apollo's own independently-
# measured 9684px for the same physical dart. Re-measure if a future
# corpus pull produces a real case landing between 76 and 9147.
MIN_SALVAGED_SUBCOMPONENT_AREA_PX = 3000


def _largest_elongated_subcomponent_points(
    mask_bool: np.ndarray,
    crop_origin_and_shape: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> np.ndarray | None:
    """2026-09-02 -- real incident, the recorded outside throw cam0 (see
    `opendarts.engines.apollo.tip_detection._largest_elongated_
    subcomponent()`'s own docstring for the full write-up -- this is a
    genuinely independent re-implementation of the same idea for THIS
    module's own morphology, not a shared call, matching this module's
    own "fresh implementation, not a port" design). This module's own
    `CLOSE_KERNEL_PX` comment above already documents the general shape
    of this failure ("a 31px close kernel merges two adjacent darts into
    one component, so the PCA axis spans both and the tip lands on the
    wrong dart") -- this incident is the SAME merge mechanism producing a
    DIFFERENT symptom: the merged component's own elongation ratio drops
    below `MIN_ELONGATION_RATIO` entirely (rather than staying elongated
    but along the wrong axis), so it never wins the whole-blob elongation
    check in the first place and the true dart's own genuinely elongated
    sub-region is left undiscovered inside a component `detect_crossing()`
    otherwise only ever scores as a whole.

    Re-segments `mask_bool` (the SAME un-closed `mask_bool & region`
    array `detect_crossing()`'s own per-candidate loop already computes)
    via a plain connected-components pass at native (non-closed)
    resolution, returning the pixel array of the LARGEST-by-area
    sub-component that independently clears `MIN_ELONGATION_RATIO`
    (genuinely dart-shaped on its own) AND
    `MIN_SALVAGED_SUBCOMPONENT_AREA_PX` (genuinely dart-SIZED, not just a
    small fragment that happens to be elongated by chance -- see that
    constant's own comment for the real, measured local-corpus evidence
    this was tightened from the generic `MIN_COMPONENT_AREA_PX` floor)
    -- `None` if nothing qualifies. Always largest-first, so a real
    dart-sized sub-region is preferred over any smaller leftover fragment
    automatically.

    `crop_origin_and_shape` (2026-09-05 perf pass): `detect_crossing()`
    now passes the footprint cropped to the candidate's bounding box, as
    `((x, y), (img_h, img_w))`. The full-frame mask is rebuilt here so
    the connected-components pass sees EXACTLY the input it always did
    (label numbering, which the area-sorted tie order depends on, is
    therefore unchanged). `None` keeps the original full-frame calling
    convention."""
    import cv2

    if crop_origin_and_shape is not None:
        (ox, oy), full_shape = crop_origin_and_shape
        sub_mask = np.zeros(full_shape, dtype=np.uint8)
        sub_mask[oy:oy + mask_bool.shape[0], ox:ox + mask_bool.shape[1]] = (
            mask_bool.astype(np.uint8) * 255
        )
    else:
        sub_mask = mask_bool.astype(np.uint8) * 255
    n, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(
        sub_mask, connectivity=8
    )
    if n <= 1:
        return None
    order = sorted(
        range(1, n), key=lambda i: int(sub_stats[i, cv2.CC_STAT_AREA]), reverse=True
    )
    for i in order:
        area = int(sub_stats[i, cv2.CC_STAT_AREA])
        if area < MIN_SALVAGED_SUBCOMPONENT_AREA_PX:
            # Areas are sorted descending -- every remaining candidate is
            # smaller still, so nothing further can qualify either.
            break
        xs, ys = component_pixels(sub_labels, i, sub_stats[i, :4])
        if len(xs) < MIN_SALVAGED_SUBCOMPONENT_AREA_PX:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        _, _, elongation = _pca(pts)
        if elongation >= MIN_ELONGATION_RATIO:
            return pts
    return None


def _analyze_component(pts: np.ndarray, area: int) -> dict | None:
    """Given one connected component's ORIGINAL (non-closed) pixels,
    compute its own tip/crossing candidate -- factored out of
    `detect_crossing()` so multiple top-K candidates can each be scored
    the same way (needed for `opendarts.engines.athena.board_gate`'s
    calibration-aware re-ranking in `engine.py`, added after this
    module's first real corpus run found the single-best-by-area choice
    picking the wrong blob often enough to matter -- see module
    docstring's "board-crossing hypothesis" section is a DIFFERENT
    concern from this; this is plain wrong-component selection, the same
    failure category tip_detection.py's own docstring documents)."""
    centroid, principal, elongation = _pca(pts)
    with np.errstate(all="ignore"):
        centered = pts - centroid
        proj = centered @ principal
        perp = centered @ np.array([-principal[1], principal[0]])
    pmin, pmax = float(proj.min()), float(proj.max())
    span = pmax - pmin
    window = max(END_WINDOW_MIN_PX, END_WINDOW_FRACTION * span)

    end_a = proj <= pmin + window
    end_b = proj >= pmax - window
    extent_a = _end_extent(perp, end_a)
    extent_b = _end_extent(perp, end_b)

    tip_is_a = extent_a <= extent_b
    denom = max(extent_a, extent_b)
    tip_confidence = float(1.0 - min(extent_a, extent_b) / denom) if denom > 0 else 0.0

    k = min(N_TIP_POINTS_AVERAGED, len(pts))
    order_lo = np.argsort(proj)[:k]
    order_hi = np.argsort(-proj)[:k]
    tip_idxs = order_lo if tip_is_a else order_hi
    other_idxs = order_hi if tip_is_a else order_lo
    tip_pt = pts[tip_idxs].mean(axis=0)
    other_pt = pts[other_idxs].mean(axis=0)
    if not (np.isfinite(tip_pt[0]) and np.isfinite(tip_pt[1])):
        return None
    if not (np.isfinite(other_pt[0]) and np.isfinite(other_pt[1])):
        other_pt = None

    # Bug found 2026-08-13 by a new synthetic unit test
    # (tests/test_athena_crossing_detection.py) -- this was backwards:
    # `tip_is_a` means the tip sits at the LOW-proj end (tip_idxs above
    # picks the smallest proj values), so walking FROM the tip TOWARD the
    # base/flight (the high-proj end) means moving in the direction of
    # INCREASING proj, i.e. +principal, not -principal. Zero effect on
    # every real corpus number reported so far
    # (CROSSING_WALK_FRACTION=0.0 the whole time this was wrong, so
    # walk_px was always 0 * axis_unit = 0 regardless of axis_unit's
    # sign) -- confirmed by re-running the full corpus after this fix,
    # identical 143/169. Fixed anyway since axis_unit is part of this
    # module's own public, documented contract
    # (CrossingDetectionResult's own docstring) and any FUTURE nonzero
    # CROSSING_WALK_FRACTION re-sweep would have silently walked the
    # wrong direction without this fix.
    axis_unit = principal if tip_is_a else -principal
    axis_unit = axis_unit / (np.linalg.norm(axis_unit) + 1e-12)

    walk_px = CROSSING_WALK_FRACTION * span
    crossing_pt = tip_pt + walk_px * axis_unit

    # The OPPOSITE end of the same component, reported alongside the
    # extent-chosen one so a calibration-aware caller can second-guess
    # this module's image-only tip/flight call. It has to be this module
    # that surfaces it (only here are the component's own mask pixels in
    # hand) but the DECISION deliberately stays with the caller, since
    # making it well needs camera geometry this module has no business
    # knowing about -- same separation of concerns as `candidates` above.
    #
    # Real measurement behind why the caller bothers (every ROI-passing
    # camera-view of the full real data/archive/clean/ corpus, restricted
    # to views where AD's own tip_xy_mm makes the true end unambiguous):
    # this module's own extent-based rule picks the right end 91.8%
    # (fresh 180-throw session, n=488) / 94.8% (169 historical throws,
    # n=459) of the time, while the caller's purely geometric rule -- the
    # end whose Z=0 intersection lands CLOSER to that camera's own ground
    # position is the tip, because the flight sits above the board plane
    # and its ray overshoots -- gets 97.3% / 96.9%. The extent rule stays
    # as the default and as the fallback for any end that cannot be
    # back-projected; see `AthenaEngine.score()`.
    if other_pt is None:
        alt = None
    else:
        alt_axis = -axis_unit
        alt_crossing = other_pt + walk_px * alt_axis
        alt = {
            "tip_px": (float(other_pt[0]), float(other_pt[1])),
            "crossing_px": (float(alt_crossing[0]), float(alt_crossing[1])),
            "axis_unit": (float(alt_axis[0]), float(alt_axis[1])),
        }

    return {
        "tip_px": (float(tip_pt[0]), float(tip_pt[1])),
        "crossing_px": (float(crossing_pt[0]), float(crossing_pt[1])),
        "axis_unit": (float(axis_unit[0]), float(axis_unit[1])),
        "opposite_end": alt,
        "span_px": span,
        "elongation_ratio": elongation,
        "tip_confidence": tip_confidence,
        "area": area,
        "diagnostics": {
            "component_area_px": area,
            "elongation_ratio": elongation,
            "span_px": span,
            "tip_end": "a" if tip_is_a else "b",
            "extent_tip_end_px": extent_a if tip_is_a else extent_b,
            "extent_other_end_px": extent_b if tip_is_a else extent_a,
            "walk_px": walk_px,
        },
    }


def detect_crossing(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    *,
    precomputed: DiffCrop | None = None,
) -> CrossingDetectionResult:
    """Find the newest dart's shaft in `frame_bgr` (vs. `bg_bgr`, the
    same camera's board state immediately before this dart -- identical
    "bg is not necessarily empty" caveat as tip_detection.py's own
    module docstring: pass the real prior-state image, not a fully empty
    board, for any dart after the first in a visit) and report both its
    visible tip pixel and its board-crossing estimate. No calibration or
    board geometry used here -- see module docstring.

    `precomputed`: optional `opendarts.imageops.DiffCrop` (2026-09-06 perf
    pass) -- this function's gray/|diff|/blur front end already computed
    by the caller (Zeus, once per camera for all sub-engines) and cropped
    to where the frame changed. Used only if it satisfies
    `PRECOMPUTE_REQUIREMENTS` for these images, otherwise ignored; the
    result is bit-identical either way (the cropped closed mask is
    pasted into a zero full frame before labeling, so component
    numbering and every reported pixel coordinate are unchanged).
    """
    import cv2

    def _fail(reason: str) -> CrossingDetectionResult:
        return CrossingDetectionResult(
            ok=False, tip_px=None, crossing_px=None, axis_unit=None,
            span_px=None, elongation_ratio=None, tip_confidence=None,
            reason=reason,
        )

    if bg_bgr.shape != frame_bgr.shape:
        return _fail(f"shape mismatch: bg {bg_bgr.shape} vs frame {frame_bgr.shape}")

    pc = precomputed
    if pc is not None and not pc.accepts(PRECOMPUTE_REQUIREMENTS, bg_bgr.shape):
        pc = None
    if pc is None:
        img_h, img_w = bg_bgr.shape[:2]
        bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = _diff_mask(bg_gray, frame_gray)
        origin = (0, 0)
    else:
        img_h, img_w = pc.img_h, pc.img_w
        mask = threshold_mask(pc.diff_blur, DIFF_THRESHOLD) # crop-sized
        origin = pc.origin
    ox, oy = origin

    # 2026-09-05 perf pass: the ellipse closing runs on the mask's padded
    # non-zero bounding box (a median ~6% of the frame), not the full
    # 1280x720 -- it was ~40% of this engine's per-throw time and OpenCV
    # does not parallelise ellipse kernels. Bit-identical to the
    # full-frame call, see `opendarts.imageops.morph_on_bbox`.
    closed = morph_on_bbox(mask, cv2.MORPH_CLOSE, ellipse_kernel(CLOSE_KERNEL_PX))
    if pc is not None:
        closed = pc.paste_full(closed)

    n, comp_labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return _fail("no diff components found (empty/near-empty diff mask)")

    candidates = []
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        bboxes[i] = (x, y, w, h)
        if w > 0.6 * img_w or h > 0.6 * img_h:
            continue # frame-spanning: global lighting drift, not a dart
        if area < MIN_COMPONENT_AREA_PX:
            continue
        candidates.append((area, i))
    if not candidates:
        return _fail("no plausible dart-sized diff components found")
    candidates.sort(reverse=True)
    top = candidates[:TOP_K_AREA_CANDIDATES]

    # Score on the ORIGINAL (un-closed) pixels -- read from each label's
    # own bounding box rather than a full-frame `==`/`&`/`nonzero` triple
    # per candidate (2026-09-05 perf pass, ~1/3 of this engine's time;
    # same pixels in the same order, see `opendarts.imageops.component_pixels`).
    # Analyze EVERY top-K-by-area candidate (not just the first one that
    # clears MIN_ELONGATION_RATIO) -- lets a calibration-aware caller
    # (opendarts.engines.athena.engine, via board_gate.py) re-rank by
    # board-plausibility instead of blindly trusting area-then-elongation
    # rank alone, which is what a real corpus run found picking the wrong
    # (but still elongated/large) blob often enough to matter.
    analyzed: list[dict] = []
    for area, i in top:
        xs, ys = component_pixels(comp_labels, i, bboxes[i], mask, origin)
        if len(xs) < MIN_COMPONENT_AREA_PX:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        cand = _analyze_component(pts, area)
        if cand is not None:
            analyzed.append(cand)
        # 2026-09-02 -- sub-component salvage, see
        # _largest_elongated_subcomponent_points()'s own docstring for
        # the full real-incident write-up. Only attempted when THIS
        # candidate's own whole-blob analysis is missing or not already
        # dart-shaped -- a candidate that already clears
        # `MIN_ELONGATION_RATIO` on its own is real, sufficient evidence
        # and does not need salvaging. Appended as an ADDITIONAL
        # candidate (never replacing the whole-blob one already appended
        # above) so `diagnostics["candidates"]` still carries the
        # complete picture for any future re-ranking caller, and so nothing
        # regresses in the (expected common) case where a later, still
        # simply-largest-by-original-area candidate would have been
        # correct anyway -- `elongation_ok`'s own list-order preference
        # below still favors this salvage result over a later, smaller
        # top-K candidate's whole-blob pass, since it is appended in the
        # SAME iteration as its larger parent candidate.
        if cand is None or cand["elongation_ratio"] < MIN_ELONGATION_RATIO:
            bx, by, bw, bh = bboxes[i]
            footprint = (comp_labels[by:by + bh, bx:bx + bw] == i) & (
                mask[by - oy:by - oy + bh, bx - ox:bx - ox + bw] != 0
            )
            sub_pts = _largest_elongated_subcomponent_points(
                footprint, ((bx, by), (img_h, img_w))
            )
            if sub_pts is not None:
                sub_cand = _analyze_component(sub_pts, int(len(sub_pts)))
                if sub_cand is not None:
                    analyzed.append(sub_cand)

    if not analyzed:
        return _fail("no candidate produced a finite tip pixel")

    elongation_ok = [c for c in analyzed if c["elongation_ratio"] >= MIN_ELONGATION_RATIO]
    ordered = elongation_ok if elongation_ok else analyzed
    chosen = ordered[0]

    diagnostics = dict(chosen["diagnostics"])
    diagnostics["n_area_candidates"] = len(candidates)
    diagnostics["candidates"] = ordered # calibration-aware re-ranking input

    return CrossingDetectionResult(
        ok=True,
        tip_px=chosen["tip_px"],
        crossing_px=chosen["crossing_px"],
        axis_unit=chosen["axis_unit"],
        span_px=chosen["span_px"],
        elongation_ratio=chosen["elongation_ratio"],
        tip_confidence=chosen["tip_confidence"],
        reason="ok",
        diagnostics=diagnostics,
    )
