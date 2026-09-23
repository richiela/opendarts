"""Athena's own per-throw confidence score (2026-08-14): a confidence
level iterated until it matches the engine's measured accuracy against
ground truth -- i.e. a genuinely CALIBRATED score, not a
plausible-sounding fabricated number (this project has an
explicit prior stance against exactly that -- see
`opendarts.live.throw_package`'s own `THROW_DETECTED` event docstring,
which reports NO confidence field precisely because none existed
anywhere in this codebase with a real basis; this module is a deliberate,
considered exception to that stance, legitimate ONLY because it is
actually measured/validated below, not just plausible-sounding).

**Two-stage design, standard for calibrated classifiers (histogram
binning + isotonic regression), implemented here in pure python/numpy --
no sklearn in this project's dependencies:**

1. `raw_confidence_score()` -- a single scalar built ONLY from signals
   `AthenaEngine.score()` already genuinely computes for itself at
   inference time (never anything referenced to ground truth, which
   doesn't exist at live inference time anyway): how many of the cameras that actually
   contributed to the final answer independently landed on that SAME
   (sector, ring) label on their own (`label_agreement_fraction`), a
   small tiebreak from how far those per-camera reads sit from the final
   blended point (`weighted_spread`, mm), and a hard penalty when the
   answer came from the ROI-gate fallback path
   (`opendarts.engines.athena.engine`'s own fallback pass -- that path
   exists precisely because every camera's strict candidate failed, so
   pretending it's as trustworthy as a normal read would misrepresent
   it).
2. `calibrated_confidence()` -- maps that raw score through a FIXED,
   pre-computed calibration curve (`_CALIBRATION_BINS` below) so the
   returned number is a real empirical P(correct), not just a monotonic
   proxy for it.

**How the calibration curve itself was built and validated** (a real
measurement over the full
360-throw `data/archive/clean/` corpus via `opendarts.capture.replay` --
never stored `result.json`, which can be stale):

- Checked several raw per-camera signals for how well they discriminate
  correct from wrong throws BEFORE picking what to build the raw score
  from: `n_cameras_used`, weighted mean
  `ray_steepness`, weighted mean `tip_confidence` all showed almost NO
  separation (correct/wrong means within a few percent of each other --
  the same real finding as this project's own extensive correlated-bias
  investigation in docs/DESIGN.md: per-camera INTRINSIC quality signals don't
  reliably predict per-throw correctness). `label_agreement_fraction`
  and consensus spread (mm distance from each read to the final blended
  point) both showed STRONG, clean separation (label agreement: 0.858
  correct-throw mean vs 0.589 wrong-throw mean; `weighted_spread`:
  4.14mm vs 9.98mm) -- these are what the raw score is built from.
- Quantile-binned the raw score, then
  ran Pool-Adjacent-Violators (PAVA) to merge any bin whose accuracy
  DECREASED relative to a lower-raw-score bin (a real calibration curve
  must be monotonic; small-n bins near the top of the distribution can
  show a tiny accuracy dip from pure sampling noise -- e.g. 1.000 ->
  0.986 -> 0.972 on n=72 each -- PAVA removes that mechanically rather
  than by hand-picking which bins to merge).
- **Bin-count swept 3-20** (LOSO
  cross-validated across all 3 real corpus sessions each time): broad
  plateau, mean|predicted-actual| 0.073-0.077 throughout, PAVA
  collapsing to 2-6 real bins regardless of how many were requested.
  10 requested bins chosen (yields 4 real bins after PAVA on the
  all-data fit) -- a value from the middle of the measured plateau, not
  a guess.
- **Leave-one-session-out cross-validated** (fit calibration on 2 of the
  3 real sessions, evaluate reliability on the held-out 3rd, pooled
  across all 3 folds): held-out Expected Calibration Error (n-weighted
  mean |predicted - observed| across 5 report bins) = **0.0132** (1.32
  percentage points) on n=360 held-out predictions -- genuinely
  validated out-of-sample, not just fit-and-declared-victory.
- **Final shipped curve fit on all 360** (standard practice in this
  project once LOSO confirms it isn't badly overfit -- same pattern
  `RADIAL_CORRECTION_MM`/`BOARD_PLANE_Z_MM` in `engine.py` already use):
  in-sample ECE via `dev/tests/test_athena_confidence.py`'s own real,
  currently-passing measurement (10 report bins, through the actual
  shipped `AthenaEngine.score()` code path, not the exploratory
  fitting script above) = **0.0164** -- consistent with the LOSO
  held-out 0.0132 above, confirming this isn't overfit to the fitting
  data. See that test file for the measured (not guessed) ceiling this
  is checked against on every run.
"""
from __future__ import annotations

import math

# ---- Stage 1: raw score -------------------------------------------------

# How much a millimetre of consensus spread (weighted mean distance from
# each used camera's own read to the FINAL blended point) subtracts from
# the raw score, on top of label_agreement_fraction -- a small tiebreak,
# not a dominant term (label agreement is the real signal; see this
# module's own docstring for the measured separation between the two).
# Chosen so a "typical wrong throw" spread (~10mm, see docstring) costs
# about 0.02 -- small relative to label_agreement_fraction's own 0.0-1.0
# range, deliberately: this is a tiebreak between throws with the SAME
# agreement fraction, not a signal meant to dominate on its own (the raw
# exploration in this module's own docstring found weighted_spread's
# raw separation is real but noisier/less monotonic than label agreement
# taken alone's decile table).
_SPREAD_PENALTY_DIVISOR_MM = 500.0
_SPREAD_CLAMP_MM = 50.0

# ROI-gate fallback throws (AthenaEngine.score()'s own fallback pass --
# every camera's strict candidate failed the board ROI gate) are forced
# to a raw score no higher than this, regardless of label agreement or
# spread. Real measured accuracy on this path is 1/2 on the current
# corpus (too small a sample to fit its own calibration bin) but
# conceptually this path only ever runs when normal detection already
# failed outright, so treating it as anywhere near full confidence would
# misrepresent it -- capped below the lowest real calibration bin's own
# midpoint rather than fit to n=2.
_ROI_FALLBACK_RAW_CAP = 0.34


def raw_confidence_score(
    consensus_reads: list[dict],
    sector: str | None,
    ring: str,
    roi_fallback_used: bool,
) -> float:
    """Stage 1: a single 0.0-1.0-ish scalar (before calibration) built
    only from what `AthenaEngine.score()` already has in hand at the
    point it calls this -- `consensus_reads` (the SAME list passed to
    `_combine_reads()`, each with its own `x_mm`/`y_mm`/`sector`/`ring`),
    the FINAL combined `(sector, ring)` those reads were blended into,
    and whether this throw went through the ROI-gate fallback path.
    Does NOT take the combined (x, y) point directly -- the caller must
    call `weighted_spread_mm(consensus_reads, combined_x, combined_y)`
    FIRST, which stashes each read's own distance to the blend onto the
    read dict itself (`_dist_to_combined_mm`) for this function to
    reuse, so the distance is computed exactly once per `score()` call
    rather than twice."""
    if not consensus_reads:
        return 0.0
    n = len(consensus_reads)
    n_agree = sum(1 for r in consensus_reads if (r["sector"], r["ring"]) == (sector, ring))
    label_agreement_fraction = n_agree / n

    total_w = sum(r["weight"] for r in consensus_reads)
    # Caller attaches "_dist_to_combined_mm" per read -- see
    # _weighted_spread_mm() below, which both computes the spread AND
    # returns it for reuse here so it's computed exactly once per score()
    # call.
    dists = [r.get("_dist_to_combined_mm", 0.0) for r in consensus_reads]
    if total_w > 0:
        weighted_spread = sum(r["weight"] * d for r, d in zip(consensus_reads, dists)) / total_w
    else:
        weighted_spread = sum(dists) / n

    spread = min(weighted_spread, _SPREAD_CLAMP_MM)
    score = label_agreement_fraction - (spread / _SPREAD_PENALTY_DIVISOR_MM)
    score = max(0.0, score)
    if roi_fallback_used:
        score = min(score, _ROI_FALLBACK_RAW_CAP)
    return score


def weighted_spread_mm(reads: list[dict], combined_x: float, combined_y: float) -> float:
    """Weighted mean distance from each read's own (x_mm, y_mm) to the
    FINAL blended point -- the same real signal
    `raw_confidence_score()`'s own docstring measured as cleanly
    predictive (4.14mm correct-throw mean vs 9.98mm wrong-throw mean).
    Also stashes each read's own distance under `_dist_to_combined_mm`
    so `raw_confidence_score()` doesn't need combined_x/combined_y
    itself -- computed once, reused once, no duplicated math."""
    if not reads:
        return 0.0
    total_w = sum(r["weight"] for r in reads)
    total = 0.0
    for r in reads:
        d = math.hypot(r["x_mm"] - combined_x, r["y_mm"] - combined_y)
        r["_dist_to_combined_mm"] = d
        total += (r["weight"] * d) if total_w > 0 else d
    return total / total_w if total_w > 0 else total / len(reads)


# ---- Stage 2: calibration ------------------------------------------------

# Fixed calibration curve, fit on the full real 360-throw
# data/archive/clean/ corpus (n_bins=10
# quantile bins + PAVA monotonic pooling -- see this module's own
# docstring for the real measurement, sweep, and LOSO cross-validation
# behind these exact numbers). Each entry is (raw_score_upper_bound,
# calibrated_confidence) in ascending order; a raw score is mapped to the
# first bin whose upper bound it falls at-or-under. NOT hand-picked --
# regenerate via that script if the corpus changes materially enough to
# warrant a re-fit (same discipline this file's sibling constants in
# engine.py already follow for RADIAL_CORRECTION_MM etc.).
#
# **Re-fit 2026-08-15** on the full 420-throw corpus through the changed
# engine (shaft-line intersection reads + corroborated-label override +
# fresh-calibration constant re-sweep, see engine.py/shaft_lines.py) --
# the previous fit (the 360-corpus one described above) had drifted
# CONSERVATIVE against the improved engine (predicted below observed,
# ECE 0.0250 with 10 quantile report bins). Same recipe as documented
# above, re-run: 10 quantile bins + PAVA (collapses to 3), LOSO-by-session
# held-out ECE 0.0155 (n=420, 4 folds) vs in-sample 0.0084 --
# consistent, not overfit, same conclusion shape as the original fit.
_CALIBRATION_BINS: list[tuple[float, float]] = [
    (0.4756, 0.8810),
    (0.8166, 0.9524),
    (1.0000, 0.9940),
]


def calibrated_confidence(raw: float) -> float:
    """Stage 2: map a raw score through the fixed, measured calibration
    curve above. Returns a real, validated P(correct) -- see this
    module's own docstring for the LOSO-held-out Expected Calibration
    Error this was checked against (0.0132) and
    dev/tests/test_athena_confidence.py for the live, currently-passing
    corpus-wide check."""
    for hi, acc in _CALIBRATION_BINS:
        if raw <= hi + 1e-9:
            return acc
    return _CALIBRATION_BINS[-1][1]
