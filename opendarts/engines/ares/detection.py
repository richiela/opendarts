"""Ares's own per-camera dart detection: motion-diff components plus a
THIN-RUN CENTERLINE fit.

Not a fork of Athena's `crossing_detection` (whole-blob PCA axis) or
Apollo's `tip_detection`. The design responds to a failure mode
confirmed by direct pixel inspection of the 2026-08-24 corpus misses:
a dart's diff component is the
thin shaft PLUS a large mass at the flight end that is mostly the
OCCLUSION DIFF of bright board features behind the flight -- on the
20-sector throws the white "20" numeral is literally legible inside the
mask. That mass's centroid follows the background features, not the dart
axis, so any whole-blob axis fit (PCA or otherwise) tilts by several
pixels and puts multiple millimetres of perpendicular error on the
board-plane shaft line -- measured directly: cam1's line sat 2.3-4.4mm
off AD's tip on every 20|1-wire miss while the other cameras were
sub-millimetre.

The centerline here is therefore fit ONLY to the shaft's thin run:

1. Cross-section binning along the current axis estimate (4px steps).
2. Per bin, the LARGEST CONTIGUOUS pixel run perpendicular to the axis
   (gap tolerance 2px) -- a detached parallel structure (shadow edge,
   neighbouring dart sliver bridged by the morphological close) does not
   drag that bin's centre, unlike a plain centroid.
3. Bins wider than a robust multiple of the shaft's own width (25th
   percentile of bin widths) are excluded -- this is what removes the
   flight/occlusion mass structurally instead of hoping a global weight
   suppresses it.
4. Weighted total-least-squares refit through the surviving bin centres,
   iterated (axis -> bins -> thin run -> axis).

The result is a sub-pixel shaft centerline (tip pixel, base pixel, unit
axis) plus self-diagnostics (shaft length, widths, fit RMS) the engine
turns into per-camera board-plane lines and fusion weights.

A note on numpy RuntimeWarnings from this module (docs/DESIGN.md,
2026-08-27): the matmuls here raise divide-by-zero / overflow / invalid
RuntimeWarnings on routine throws. Verified spurious (tmp probe,
g1-003-S5): a matmul with confirmed all-finite inputs AND a confirmed
all-finite output still raises all three at once -- the signature of the
BLAS backend (macOS Accelerate) leaving FP status flags set on small
gemm/gemv calls, which numpy then reports. No NaN/Inf enters or leaves
these operations. Deliberately NOT suppressed with np.errstate: blanket
suppression here would also hide a REAL future NaN; downstream consumers
already check np.isfinite where it matters (engine._CamRead._line_quality
on fit_rms_px).
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

# Mask constants -- own values (swept against the opendarts corpus;
# see engine.py's module docstring for the sweep results).
DIFF_THRESHOLD = 30.0
BLUR_KSIZE = 5
# CLOSE_KERNEL_PX: 25 -> 15, 2026-09-05, Zeus-latency follow-up (see
# opendarts/engines/zeus/engine.py's own docstring for the 2026-08-27
# Zeus-parallel-dispatch fix this builds on -- Zeus's own wall-clock is
# max() across its 4 sub-engines, so Ares (this engine) and Athena
# (Athena, ~77ms) are the near-tied pair that actually gates Zeus's
# real latency; changing only one of the two buys nothing, see
# opendarts/engines/athena/crossing_detection.py's own matching
# 2026-09-05 CLOSE_KERNEL_PX comment for the other half). Real measured
# effect, full 1506-package live corpus (the session corpus,
# not _freeze1), Zeus re-scored end to end: 1497/1504 vs the shipped
# 1496/1504 (McNemar p=1.0 -- statistically indistinguishable, not a
# regression), and the two engines' own error CORRELATION actually
# DROPS (13.4x -> 9.9x expected-by-chance), so this is not a
# consensus-degradation risk either. Real macOS-rig timing (direct
# `zeus.score()` call, n=60/arm, interleaved): 79.7ms -> 65.2ms, a real
# 14.5ms/18.2% saving. Individual-engine accuracy at the shipped k=15,
# independently spot-checked against the same live corpus before this
# comment was written (measured here, not carried over unverified):
# 1481/1504 = 98.47% (exploratory, not
# shipped). **Hard floor for this engine, do not
# go below 13** (the earlier sweep this change is based on; Athena's
# own floor is different -- see that engine's own matching comment for
# ITS real one-sided accuracy cliff, measured at k=7, p=0.0007). 15
# sits comfortably above this engine's own floor, not on its shoulder.
CLOSE_KERNEL_PX = 15
# What a caller-supplied `opendarts.imageops.DiffCrop` must satisfy for
# `detect_candidates()` to use it in place of its own gray/diff/blur
# front end (2026-09-06 perf pass): the pad covers the closing's dilate
# reach plus its erode window -- see `DiffCrop`'s docstring.
PRECOMPUTE_REQUIREMENTS = PrecomputeRequirements(
    ksize=BLUR_KSIZE, threshold=DIFF_THRESHOLD, pad_px=CLOSE_KERNEL_PX,
)
MIN_COMPONENT_AREA_PX = 40
TOP_K_CANDIDATES = 4
# A component spanning most of the frame is global lighting drift.
MAX_COMPONENT_FRAME_FRACTION = 0.6

# Centerline constants.
BIN_STEP_PX = 4.0
CROSS_SECTION_GAP_PX = 2.0
THIN_RUN_WIDTH_FACTOR = 2.0
THIN_RUN_WIDTH_PAD_PX = 4.0
MIN_SHAFT_BINS = 6
CENTERLINE_ITERATIONS = 3
# Consecutive over-width bins tolerated inside the shaft run (a wire
# crossing or specular glint can fatten one or two bins mid-shaft).
MAX_FAT_STREAK_BINS = 2
# A fat streak LONGER than MAX_FAT_STREAK_BINS no longer hard-terminates
# the walk (2026-08-27, g3-036-S5 root cause): the dart's own silhouette
# widening near the board (shadow/reflection) produced an 11-bin streak
# measuring 12.3-12.6px against a 12.2px thin threshold -- 0.1-0.4px over
# -- which amputated the true tip (a clean 5-bin taper to 1px, dead on AD
# truth, sitting right past the streak) from the run on 2 of 3 cameras.
# The walk now RESUMES past a long streak when at least this many
# consecutive thin bins follow it; the streak's own bins are excluded
# from the fit (their largest-contiguous-run centres are exactly the
# shadow-dragged ones the thin-run design distrusts). Value measured on
# the full frozen corpus (1414 graded throws, 2026-08-27): min-thin 2
# scores 1408, min-thin 3 scores 1402 (g3-036's own cam2 has exactly 2
# consecutive thin bins after its streak before a tolerated wire-glint
# bin, so 3 refuses the very resume this exists for).
FAT_STREAK_RESUME_MIN_THIN_BINS = 2
# NO collinearity guard on the resume -- measured and REJECTED
# (2026-08-27, full frozen corpus, 1414 graded throws, every variant
# against the same committed baseline of 1398):
# - terminal-taper end-choice + unguarded resume (SHIPPED): 1408,
# 3 regressions, all near-wire (1.8-5.4mm), zero howlers.
# - outer-fifth end-choice + rms guard (keep the extension only when
# its weighted TLS rms stays under max(4px, 4x the unextended
# run's rms), refuse when the pre-streak run is itself noisy):
# also 1408, but its regressions include a NEW 141mm howler
# (018-S9: a junk 148px extension of an rms-14 run passed the
# relative bound and replaced a clean tip read 6px from truth).
# Softer refusal variants recover the howler only by giving up 4
# net fixes (1404), two of them ~68mm howler-class misses.
# - trim-until-straight instead of all-or-nothing: MANUFACTURES
# clean high-weight lines out of junk extensions (a 229mm
# fabricated answer on an off-board dart, an 83.8mm no-score).
# - guard + terminal-taper together: 1405 with three howlers.
# Equal accuracy, and the unguarded config's worst regression is
# 5.4mm vs the guard's 141mm -- confidently-wrong howlers are this
# project's named worst failure mode, so the guard lost. The resume
# rms values are still computed and recorded in candidate diagnostics
# (this exact debugging session needed them repeatedly).
N_TIP_PIXELS_AVERAGED = 3


@dataclass
class DartCandidate:
    """One diff component's fitted shaft centerline. `tip_px` is the end
    the detector believes is the tip (narrower end); the ENGINE makes the
    final tip/base call geometrically, so both ends are first-class.

    `degraded=True` marks a recall-fallback candidate whose component was
    too short/compact for the thin-run centerline fit (e.g. a nearly
    head-on dart) -- its ends come from a plain PCA extent, its "line" is
    not trustworthy, and the engine uses it as point evidence only."""

    tip_px: tuple[float, float]
    base_px: tuple[float, float]
    axis_unit: tuple[float, float] # tip -> base
    area_px: float
    shaft_len_px: float
    mean_width_px: float
    fit_rms_px: float
    tip_taper: float # 0..1, how much narrower the tip end is vs the base end
    degraded: bool = False
    diagnostics: dict = field(default_factory=dict)


@dataclass
class _CrossSections:
    """Per-bin cross-section summaries along the current axis estimate,
    in ascending order of `proj` (arrays of equal length `n`)."""

    proj: np.ndarray # (n,) weighted position along the axis
    centre: np.ndarray # (n,2) weighted centre of the kept run
    width: np.ndarray # (n,) perpendicular extent of the kept run
    count: np.ndarray # (n,) pixels in the kept run (float64)

    def __len__(self) -> int:
        return len(self.proj)


def _cross_sections(
    pts: np.ndarray, vals: np.ndarray, mu: np.ndarray, axis: np.ndarray
) -> _CrossSections:
    """Bin pixels along `axis` (BIN_STEP_PX steps); per bin keep only the
    largest contiguous perpendicular run ('contiguous' = consecutive
    perpendicular gaps of at most CROSS_SECTION_GAP_PX; ties -> the run
    nearest the negative-perpendicular side, i.e. the first in sorted
    order) and report its weighted centre / position, width and count.
    Bins with fewer than two pixels are dropped.

    2026-09-06 rewrite (Tier A of the Ares perf review): the previous
    implementation looped over bins in Python -- an `argsort`, a gap
    scan and three weighted sums PER BIN, ~40 bins x 3 iterations x ~5
    candidates per throw, ~11us of interpreter overhead each and ~44% of
    this engine's whole per-throw time (and the single largest GIL
    holder inside Zeus). This version does every bin at once: a stable
    sort by (bin, perp), run boundaries from a gap test, per-run
    weighted sums with `np.add.reduceat`, and per-bin largest-run choice
    with a second stable sort. Same algorithm, same constants, same
    decisions. Measured BIT-IDENTICAL to the loop it replaces on the
    full corpus (17 sessions, 4,995 camera views, 10,054 candidates:
    every tip/base/axis/rms exactly equal; Ares and Zeus
    EngineResults identical on all 1,665 throws) -- `reduceat` runs the
    same pairwise inner loop per segment that `.sum()` runs on a slice,
    and a tie in perpendicular offset never straddles a run boundary, so
    each run's pixel SET and its summation order both survive."""
    rel = pts - mu
    nvec = np.array([-axis[1], axis[0]])
    proj = rel @ axis
    perp = rel @ nvec
    lo, hi = float(proj.min()), float(proj.max())
    nbins = int(max(1, np.ceil((hi - lo) / BIN_STEP_PX)))
    idx = np.clip(((proj - lo) / BIN_STEP_PX).astype(np.int16), 0, nbins - 1)

    # Sort by (bin, perp): each bin is a contiguous slice, perp-ascending.
    # Two stable sorts == np.lexsort((perp, idx)) exactly, at under half
    # the cost (the int16 key gets numpy's radix sort).
    by_perp = np.argsort(perp, kind="stable")
    order = by_perp[np.argsort(idx[by_perp], kind="stable")]
    b_s = idx[order]
    p_s = perp[order]
    n = len(order)

    # Run boundaries: a new bin, or a perpendicular gap wider than allowed.
    seg_start = np.empty(n, dtype=bool)
    seg_start[0] = True
    np.not_equal(b_s[1:], b_s[:-1], out=seg_start[1:])
    seg_start[1:] |= (p_s[1:] - p_s[:-1]) > CROSS_SECTION_GAP_PX
    starts = np.flatnonzero(seg_start)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:] - 1
    ends[-1] = n - 1
    sizes = ends - starts # (count - 1), the same tie-neutral measure as before
    seg_bin = b_s[starts]

    # Largest run per bin, first in sorted order on ties: a stable sort by
    # (bin, -size) puts each bin's winner first in its group.
    by_bin = np.lexsort((-sizes, seg_bin))
    sb = seg_bin[by_bin]
    first = np.empty(len(sb), dtype=bool)
    first[0] = True
    np.not_equal(sb[1:], sb[:-1], out=first[1:])
    best = by_bin[first] # one segment per non-empty bin, ascending bin

    # Weighted sums per segment over the sorted arrays, then pick winners.
    w_s = vals[order]
    wsum = np.add.reduceat(w_s, starts)[best]
    psum = np.add.reduceat(proj[order] * w_s, starts)[best]
    csum = np.add.reduceat(pts[order] * w_s[:, None], starts, axis=0)[best]
    count = (ends - starts + 1)[best]
    width = p_s[ends[best]] - p_s[starts[best]]

    # Keep bins with >= 2 pixels IN THE BIN (not just in the kept run) and
    # positive weight -- the same two skips as before.
    bin_count = np.bincount(idx, minlength=nbins)[seg_bin[best]]
    keep = (bin_count >= 2) & (wsum > 0)
    wsum, psum, csum, count, width = wsum[keep], psum[keep], csum[keep], count[keep], width[keep]

    proj_b = psum / wsum
    centre = csum / wsum[:, None]
    ordr = np.argsort(proj_b, kind="stable")
    return _CrossSections(
        proj=proj_b[ordr], centre=centre[ordr], width=width[ordr],
        count=count[ordr].astype(np.float64),
    )


def _weighted_tls(centres: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted total-least-squares line: returns (mean, unit axis).

    Deliberately still `np.linalg.eigh` (2026-09-06 Tier A review): a
    closed-form 2x2 eigenvector was tried and REJECTED because the axis
    SIGN is load-bearing -- `_cross_sections` lays its 4px bin grid out
    from the minimum projection, so flipping the axis flips which end the
    grid is anchored at and changes bin membership, thin/fat flags and
    the fitted line (measured: different n_shaft_bins on ~10% of
    candidates). LAPACK's sign convention is therefore part of this
    engine's tuned behaviour and must not be re-derived."""
    mu = np.average(centres, axis=0, weights=weights)
    c = centres - mu
    cov = (c * weights[:, None]).T @ c
    evals, evecs = np.linalg.eigh(cov)
    return mu, evecs[:, int(np.argmax(evals))]


def _run_fit_rms(centres: np.ndarray, counts: np.ndarray) -> float:
    """Weighted TLS fit rms (px) of a bin run's centres -- recorded in
    candidate diagnostics when a fat-streak resume fires (see the
    rejected-guard note above N_TIP_PIXELS_AVERAGED)."""
    counts = np.minimum(counts, 3.0 * float(np.median(counts)))
    mu, axis = _weighted_tls(centres, counts)
    nvec = np.array([-axis[1], axis[0]])
    resid = (centres - mu) @ nvec
    return float(np.sqrt(np.average(resid ** 2, weights=counts)))


def _fit_centerline(
    pts: np.ndarray, vals: np.ndarray, area: int
) -> tuple[DartCandidate | None, str | None]:
    """The thin-run centerline fit described in the module docstring.

    Returns (candidate, failure_reason): exactly one is non-None. The
    failure reason is what the degraded fallback records so a package
    can say WHY a camera's line evidence never existed (cam1 on miss
    g1-003-S5 failed here by ONE bin -- thin run 5 of a
    6-bin minimum, width profile fattened by prior-dart overlap -- and
    nothing on disk recorded it)."""
    if len(pts) < MIN_COMPONENT_AREA_PX:
        return None, f"component_pixels_below_min ({len(pts)} < {MIN_COMPONENT_AREA_PX})"
    mu, axis = _weighted_tls(pts, np.ones(len(pts)))
    run = None
    resume_diag: dict | None = None
    for _ in range(CENTERLINE_ITERATIONS):
        bins = _cross_sections(pts, vals, mu, axis)
        if len(bins) < MIN_SHAFT_BINS:
            return None, f"too_few_cross_section_bins ({len(bins)} < {MIN_SHAFT_BINS})"
        widths = bins.width
        ref_w = float(np.percentile(widths, 25))
        thin_thresh = max(THIN_RUN_WIDTH_FACTOR * ref_w, ref_w + THIN_RUN_WIDTH_PAD_PX)

        # Tip end: the outermost 3 bins at whichever end is narrower --
        # a genuine terminal taper (a dart tip narrows to ~1px in its
        # last bins), not the mean of the outer fifth. The wider window
        # dilutes the taper with shaft-width bins: on g3-036-S5's cam2
        # the outer-fifth means were 12.1 vs 11.4px (wrong end, flight
        # chosen) while the terminal-3 means were 11.9 vs 5.1px (tip
        # end, decisively). Measured corpus-wide 2026-08-27 with the
        # fat-streak resume below: terminal-3 1408 vs outer-fifth 1406
        # of 1414, and the outer-fifth config's regressions include a
        # 141mm howler where terminal-3's worst is a 5.4mm wire call
        # (see the resume-guard rejection note above N_TIP_PIXELS_AVERAGED).
        k = min(3, len(bins))
        wa = float(np.mean(widths[:k]))
        wb = float(np.mean(widths[-k:]))
        # Bin indices walked from the tip end.
        ordered = np.arange(len(bins)) if wa <= wb else np.arange(len(bins) - 1, -1, -1)

        # Walk from the tip end collecting thin bins; tolerate short fat
        # streaks inline (wire glints -- their bins stay in the run), and
        # RESUME past a sustained one (near-board silhouette widening,
        # see FAT_STREAK_RESUME_MIN_THIN_BINS's comment) provided a real
        # thin section continues -- a long streak's own bins are excluded
        # from the run. Leading fat bins before the run starts are
        # skipped (mask noise right at the tip).
        flags = (widths[ordered] <= thin_thresh).tolist()
        # thin_ahead[j]: consecutive thin bins starting at j (inclusive).
        thin_ahead = [0] * (len(ordered) + 1)
        for j in range(len(ordered) - 1, -1, -1):
            thin_ahead[j] = thin_ahead[j + 1] + 1 if flags[j] else 0
        this_run: list[int] = [] # bin indices, tip end first
        fat_buffer: list[int] = []
        # Length of this_run at the FIRST long-streak resume -- kept for
        # the resume diagnostics below (rejected-guard debugging aid).
        pre_resume_len: int | None = None
        for j, b in enumerate(ordered.tolist()):
            if flags[j]:
                if fat_buffer:
                    if len(fat_buffer) <= MAX_FAT_STREAK_BINS:
                        # Short streak (wire glint): keep its bins
                        # inline, exactly the pre-resume behavior.
                        this_run.extend(fat_buffer)
                    elif thin_ahead[j] < FAT_STREAK_RESUME_MIN_THIN_BINS:
                        # A long streak followed by too little thin
                        # signal: stop, exactly the pre-resume behavior.
                        # (An "extend-only, never rescue a not-yet-viable
                        # run" variant was measured and REJECTED: corpus
                        # 1405 vs 1408 -- rescuing a one-bin-short fit
                        # into a real line is a measured net win.)
                        break
                    elif pre_resume_len is None:
                        # Resume: the long streak's own bins stay out of
                        # the fit (shadow-dragged centres).
                        pre_resume_len = len(this_run)
                    fat_buffer = []
                this_run.append(b)
            elif this_run:
                fat_buffer.append(b)
        if pre_resume_len is not None and len(this_run) >= 2:
            # Diagnostics only -- the resumed run is ALWAYS kept (a
            # collinearity guard here was measured corpus-wide and
            # rejected; see the note above N_TIP_PIXELS_AVERAGED). The
            # rms of the full run vs the unextended prefix is recorded
            # because a future miss investigation will want exactly
            # these numbers (this session's did, repeatedly).
            r0 = this_run[:pre_resume_len]
            resume_diag = {
                "resume_rms_full_px": round(
                    _run_fit_rms(bins.centre[this_run], bins.count[this_run]), 3
                ),
                "resume_rms_unextended_px": (
                    round(_run_fit_rms(bins.centre[r0], bins.count[r0]), 3)
                    if len(r0) >= 2 else None
                ),
            }
        if len(this_run) < MIN_SHAFT_BINS:
            return None, (
                f"thin_run_too_short ({len(this_run)} of {len(bins)} bins,"
                f" min {MIN_SHAFT_BINS}; thin_thresh {thin_thresh:.1f}px)"
            )
        run = this_run
        centres = bins.centre[run]
        counts = bins.count[run]
        mu, axis = _weighted_tls(centres, np.minimum(counts, 3.0 * float(np.median(counts))))

    assert run is not None
    # `centres`/`counts` are the final iteration's run, tip end first.
    nvec = np.array([-axis[1], axis[0]])
    resid = (centres - mu) @ nvec
    fit_rms = float(np.sqrt(np.average(resid ** 2, weights=counts)))

    # Orient the axis tip -> base and place both ends ON the fitted line.
    tip_c = centres[0]
    base_c = centres[-1]
    d = base_c - tip_c
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None, "degenerate_run_extent (bin centres coincide)"
    d = d / norm

    def _on_line(point: np.ndarray) -> np.ndarray:
        return mu + ((point - mu) @ axis) * axis

    tip_on = _on_line(tip_c)
    base_on = _on_line(base_c)
    d = base_on - tip_on
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None, "degenerate_line_extent (on-line ends coincide)"
    d = d / norm

    # Refine the tip to the extreme mask pixels near the line, projected
    # onto it -- the bin centre sits half a bin short of the real tip.
    perp_all = np.abs((pts - mu) @ np.array([-axis[1], axis[0]]))
    near_line = perp_all <= max(3.0, thin_thresh / 2.0)
    if near_line.sum() >= 1:
        proj_d = (pts[near_line] - tip_on) @ d
        order = np.argsort(proj_d)[:N_TIP_PIXELS_AVERAGED]
        tip_refined = _on_line(pts[near_line][order].mean(axis=0))
    else:
        tip_refined = tip_on

    widths_run = bins.width[run]
    mean_width = float(np.mean(widths_run))
    k = max(2, len(run) // 5)
    w_tip = float(np.mean(widths_run[:k]))
    w_base = float(np.mean(widths_run[-k:]))
    taper = float(1.0 - w_tip / w_base) if w_base > 0 else 0.0

    return DartCandidate(
        tip_px=(float(tip_refined[0]), float(tip_refined[1])),
        base_px=(float(base_on[0]), float(base_on[1])),
        axis_unit=(float(d[0]), float(d[1])),
        area_px=float(area),
        shaft_len_px=float(np.linalg.norm(base_on - tip_refined)),
        mean_width_px=mean_width,
        fit_rms_px=fit_rms,
        tip_taper=taper,
        diagnostics={
            "n_shaft_bins": len(run),
            "component_area_px": int(area),
            **(resume_diag or {}),
        },
    ), None


def _degraded_candidate(pts: np.ndarray, vals: np.ndarray, area: int) -> DartCandidate | None:
    """Recall fallback when the thin-run fit cannot run (component too
    short/compact): plain PCA extent ends, flagged `degraded` so the
    engine treats it as point-only evidence. A nearly head-on dart makes
    exactly this kind of compact blob, and dropping the camera entirely
    was measured to cost real throws (single-camera misses g1-010/g1-075
    in the first corpus run)."""
    if len(pts) < MIN_COMPONENT_AREA_PX:
        return None
    mu, axis = _weighted_tls(pts, vals)
    proj = (pts - mu) @ axis
    perp = (pts - mu) @ np.array([-axis[1], axis[0]])
    order = np.argsort(proj)
    k = min(N_TIP_PIXELS_AVERAGED, len(pts))
    end_a = pts[order[:k]].mean(axis=0)
    end_b = pts[order[-k:]].mean(axis=0)
    span = float(proj.max() - proj.min())
    width = float(np.percentile(np.abs(perp), 90) * 2.0)
    d = end_b - end_a
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return None
    d = d / n
    # Narrower end heuristic: compare perp spread near each end.
    m_a = proj <= proj.min() + max(4.0, 0.25 * span)
    m_b = proj >= proj.max() - max(4.0, 0.25 * span)
    w_a = float(np.ptp(perp[m_a])) if m_a.sum() >= 2 else 0.0
    w_b = float(np.ptp(perp[m_b])) if m_b.sum() >= 2 else 0.0
    if w_b < w_a:
        end_a, end_b = end_b, end_a
        d = -d
    return DartCandidate(
        tip_px=(float(end_a[0]), float(end_a[1])),
        base_px=(float(end_b[0]), float(end_b[1])),
        axis_unit=(float(d[0]), float(d[1])),
        area_px=float(area),
        shaft_len_px=span,
        mean_width_px=width,
        fit_rms_px=float("nan"),
        tip_taper=0.0,
        degraded=True,
        diagnostics={"component_area_px": int(area)},
    )


def detect_candidates(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    *,
    precomputed: DiffCrop | None = None,
) -> list[DartCandidate]:
    """All plausible new-dart candidates in this camera's frame (vs its
    background), largest component first. Empty list when nothing
    dart-like changed. Calibration-free -- the engine applies its own
    board-aware gating and selection.

    `precomputed`: optional `opendarts.imageops.DiffCrop` (2026-09-06 perf
    pass) -- the gray/|diff|/blur front end already computed by the
    caller (Zeus, once per camera for all its sub-engines) and cropped to
    where the frame changed. Used only if it satisfies
    `PRECOMPUTE_REQUIREMENTS` for these images, otherwise ignored;
    bit-identical either way (the cropped closed mask is pasted into a
    zero full frame before labeling)."""
    import cv2

    if bg_bgr.shape != frame_bgr.shape:
        return []
    pc = precomputed
    if pc is not None and not pc.accepts(PRECOMPUTE_REQUIREMENTS, bg_bgr.shape):
        pc = None
    # 2026-09-05 perf pass, all via opendarts.imageops and all bit-identical
    # (see each helper's docstring): uint8 absdiff -> one float32 cast;
    # cv2.compare for the threshold (this mask was 0/1, it is now 0/255 --
    # every consumer below tests non-zero, and connectedComponents /
    # morphology treat any non-zero as foreground); the ellipse closing
    # on the mask's padded non-zero bounding box instead of the full
    # frame (~19% of the engine's time was that full-frame close).
    if pc is None:
        img_h, img_w = bg_bgr.shape[:2]
        bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        fr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff = blurred_abs_diff(bg_gray, fr_gray, BLUR_KSIZE)
        origin = (0, 0)
    else:
        img_h, img_w = pc.img_h, pc.img_w
        diff = pc.diff_blur # crop of the identical full-frame array
        origin = pc.origin
    ox, oy = origin
    mask = threshold_mask(diff, DIFF_THRESHOLD)
    closed = morph_on_bbox(mask, cv2.MORPH_CLOSE, ellipse_kernel(CLOSE_KERNEL_PX))
    if pc is not None:
        closed = pc.paste_full(closed)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return []
    comps = []
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        bboxes[i] = (x, y, w, h)
        if w > MAX_COMPONENT_FRAME_FRACTION * img_w or h > MAX_COMPONENT_FRAME_FRACTION * img_h:
            continue
        if area < MIN_COMPONENT_AREA_PX:
            continue
        comps.append((area, i))
    comps.sort(reverse=True)

    out: list[DartCandidate] = []
    for area, i in comps[:TOP_K_CANDIDATES]:
        # Per-label pixels from the label's own bbox, not a full-frame
        # ==/&/nonzero (~25% of the engine's time) -- same pixels, same
        # order, see `opendarts.imageops.component_pixels`.
        xs, ys = component_pixels(labels, i, bboxes[i], mask, origin)
        if len(xs) < MIN_COMPONENT_AREA_PX:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float64)
        vals = diff[ys - oy, xs - ox].astype(np.float64)
        cand, fit_failure = _fit_centerline(pts, vals, area)
        if cand is None:
            cand = _degraded_candidate(pts, vals, area)
            if cand is not None and fit_failure is not None:
                # Why this candidate is degraded (point-only, no line
                # evidence) -- surfaced through the engine's per-camera
                # drop diagnostics so it lands in every package.
                cand.diagnostics["centerline_failure"] = fit_failure
        if cand is not None:
            out.append(cand)
    return out
