"""Apollo's own per-throw confidence score, iterated until its confidence
level matches the engine's measured accuracy against ground truth. A
genuine, CALIBRATED probability that this throw's `(sector, ring)`
answer is correct, built entirely from signals `ApolloEngine.score()`
already computes -- no new detection/geometry work, no ground-truth
input (this must be computable at real live-inference time, where
ground truth does not exist).

**Scoped engine-local for now**: lives in
`diagnostics["confidence"]` on `ApolloEngine`'s own `EngineResult`, NOT
a new field on the shared `opendarts.engines.base.EngineResult` dataclass --
that's a cross-cutting change, and whether to promote it is decided
after reviewing this alongside Athena's own parallel confidence work on
a different branch; landing it here first would risk exactly the kind of
collision this project's docs/DESIGN.md already documents real incidents of.
`diagnostics` is already a free-form dict (see `opendarts/engines/base.py`),
so adding this key requires no schema change anywhere.

**This project has an explicit standing prior AGAINST a fabricated
confidence number** (`docs/DESIGN.md`) -- a plausible-sounding score with
no real basis is worse than no score at all, because it invites trust
it hasn't earned. What follows is a deliberate, considered exception:
every input is a signal `score_dart()`/`ApolloEngine.score()` already
produces for its own real reasons, and the mapping from those signals to
a 0-1 number is FIT and VALIDATED against the real 360-throw
`data/archive/clean/` corpus (3 sessions, replayed through current code
via `opendarts.engines.registry.get_engine("Apollo").score()` -- not the
stale `result.json` some of that corpus's throws were originally
captured with, see the "Replay is the source of truth" constraint), not
guessed.

## The three signals, and why each one is real evidence

1. **`max_ray_disagreement_mm`** -- how much the accepted camera set's
   triangulated rays disagree at the final point. Already the exact
   quantity `scoring.MAX_RAY_DISAGREEMENT_MM`/
   `MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR` gate accept/reject on -- a
   throw already right at the edge of that gate is a genuinely weaker
   observation than one agreeing to a fraction of a millimeter.
   Measured on the 360-corpus's 359 scored throws: correct throws'
   median disagreement is 1.38mm; wrong throws' median is 3.12mm -- a
   real, if noisy, separation (not every wrong throw disagrees a lot --
   several of the worst misses are wire-adjacent throws every camera
   agreed tightly and confidently wrong on, see `tip_detection.py`'s and
   `docs/DESIGN.md`'s dated 2026-08-14 miss writeups -- which is exactly
   why this is combined with signal 2, not used alone).
2. **Distance from the triangulated `board_xy_mm` to the NEAREST
   scoring wire** (ring boundary OR sector boundary, whichever is
   closer -- see `_dist_to_nearest_wire_mm()` below). This is pure
   geometry on Apollo's own already-triangulated answer, computable
   with zero extra detection work and no ground-truth input. **The
   strongest real, clean finding of this investigation**: on the full
   360-corpus, every throw with nearest-wire distance >=10mm was
   correct (75/75, 100%), while every one of the 11 wrong throws sat
   within 8.28mm of some wire (median 1.93mm). A point that lands
   comfortably inside a scoring region, far from any boundary, is
   basically immune to a few millimeters of detection noise flipping
   its sector/ring -- a point sitting ON a wire is exactly the
   opposite. This directly explains why most of this engine's real
   remaining misses ARE near-wire cases (see `tip_detection.py`'s own
   dated 2026-08-14 comments) -- it is the single most informative
   ground-truth-free signal found in this investigation, and the one thing
   `max_ray_disagreement_mm` alone cannot see (a throw can have
   excellent ray agreement and still be geometrically on the wrong side
   of a wire by a millimeter -- ray agreement measures HOW CONSISTENT
   the observation is, not HOW CLOSE the answer sits to a decision
   boundary; these are different failure modes and need different
   signals).
3. **Whether the 2-of-3 RANSAC fallback pair was used** (`outlier_camera
   is not None` -- i.e. the full 3-camera ray set disagreed enough that
   `score_dart()` fell back to the single best-agreeing pair and
   discarded a camera). A real, if less informative, signal: fallback
   throws are correct 77.8% of the time (7/9) vs 97.4% (341/350) when
   the full set was trusted -- consistent with `scoring.py`'s own
   documented correlated-bias problem (per-ray disagreement magnitude can't
   fully separate "one genuinely bad ray" from "correlated bias", so a
   fallback accept is inherently a weaker claim than a full-set accept
   even when it passes its own, stricter threshold).

`alt_candidates_used` (whether `score_dart()`'s primary/alt combination
search picked an ambiguous camera's alternate) was measured too (90.0%
correct vs 97.1%, n=10 -- some signal) but deliberately left OUT of the
fitted model: with only 11 real "wrong" examples in the whole corpus, a
4th feature risks overfitting a model this data-starved rather than
adding real discriminative power beyond what the wire-distance signal
already captures (an ambiguous alt is disproportionately likely on an
already-near-wire throw). Kept out on evidence, not by assumption --
re-check this on a larger corpus if a future pass wants to.

## The model, and how it was fit

`P(correct)` via logistic regression (fit by hand, IRLS/Newton-Raphson
by hand -- neither `scikit-learn` nor `scipy` is
installed in this project's venv) on the 359 real scored throws, a
light L2 penalty (this data is only 11-positives-for-"wrong" out of 359,
genuinely near-separable, and an unregularized MLE on data this
imbalanced blows up):

    wire_safe = min(dist_to_nearest_wire_mm, WIRE_CAP_MM) / WIRE_CAP_MM (0=on a wire, 1=far from every wire)
    disagree_risk = min(max_ray_disagreement_mm, DISAGREE_CAP_MM) / DISAGREE_CAP_MM (0=perfect agreement, 1=at the reject threshold)
    fallback_risk = 1.0 if the 2-of-3 RANSAC pair was used else 0.0

    logit(confidence) = INTERCEPT + W_WIRE_SAFE*wire_safe
                                   + W_DISAGREE_RISK*disagree_risk
                                   + W_FALLBACK_RISK*fallback_risk

`DISAGREE_CAP_MM = 10.0` is deliberately the same number as
`scoring.MAX_RAY_DISAGREEMENT_MM` (the accept/reject gate itself) --
disagreement beyond that point is already a rejected throw, so nothing
past it should keep pushing this score around. `WIRE_CAP_MM = 20.0` is
the point past which the corpus already shows 100% safety (measured
plateau starts at 10mm; 20mm gives real headroom above the exact
measured edge rather than fitting to it).

## Calibration -- the actual point, measured not eyeballed

"Iterate until confidence matches accuracy" means: bucket throws by
predicted confidence and check whether the OBSERVED accuracy in each
bucket matches the PREDICTED confidence. Expected Calibration Error
(ECE) is the count-weighted mean absolute gap between the two across
buckets.

**In-sample** (fit and measured on the same 359 throws, 8 buckets):
ECE = **0.035** (3.5 percentage points, count-weighted mean gap). Every
bucket's observed accuracy is within a few points of its mean predicted
confidence, and the mapping is monotonic (higher predicted confidence
-> higher or equal observed accuracy in every bucket but one, which sits
within the bucket's own sampling noise at n=44-45).

**Leave-one-session-out** (refit on 2 of the 3 real sessions, predict
the held-out third -- the honest, out-of-sample check; the 3 sessions
are three recorded sessions of 180, 120 and 60 throws,
predicted confidences pooled across all three folds before bucketing, 6
buckets): ECE = **0.0195** (under 2 points) -- calibration holds up
out-of-sample, not just in the fold that produced the coefficients. Both
numbers, plus the full per-bucket reliability tables, are reproduced by
`tests/test_engine_apollo_confidence.py`.

**Honest limitation, stated plainly**: this calibration is good in
aggregate (bucket-level predicted-vs-observed gap is small) but
per-throw DISCRIMINATION is modest -- the 11 real wrong throws'
predicted confidence ranges 0.839-0.960 (median 0.947), which
meaningfully overlaps the 348 correct throws' own range of 0.866-0.990
(median 0.961). This is not a flaw in the fitting, it is what the
underlying geometry actually is: most of this engine's real remaining
misses (see `tip_detection.py`'s dated 2026-08-14 investigation) are
near-wire throws every camera agreed on tightly and confidently -- there
often isn't a strong INTERNAL tell that a specific throw is wrong, only
a real, aggregate pattern that near-wire throws as a POPULATION are
riskier than far-from-any-wire ones. A calibrated score is one where "I
said 95%, I'm right about 95% of the time" -- it does not promise "I can
always tell you WHICH ones are the 5%." Reported honestly rather than
oversold.

Coefficients below are literal numbers copied out of the fit's own
output file -- not retyped/rounded by hand, so they match the fit
exactly.
"""
from __future__ import annotations

import math

from opendarts.geometry import board as _board_geom
from opendarts.geometry.board import (
    BULL_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    SECTOR_NUMBERS_CLOCKWISE,
    TREBLE_OUTER_RADIUS_MM,
    wire_boundary_angle_deg,
)

# --- Fitted model (IRLS logistic regression, 359
# real scored throws from data/archive/clean/, light L2 penalty) --------
INTERCEPT = 2.936600402446311
W_WIRE_SAFE = 1.705957261335808
W_DISAGREE_RISK = -0.7219693347662904
W_FALLBACK_RISK = -1.0897793092201593

# Same number as scoring.MAX_RAY_DISAGREEMENT_MM -- see module docstring.
DISAGREE_CAP_MM = 10.0
# Measured plateau starts at 10mm (100% correct, n=75); this is
# deliberately above that measured edge, not fit to it -- see docstring.
WIRE_CAP_MM = 20.0

# --- Ring/sector wire geometry, self-contained (deliberately NOT reusing
# opendarts/geometry/board.py's own ring-radius list beyond importing the
# plain constants above) -- kept as one small computation local to this
# module rather than adding a new shared helper function to board.py,
# per this task's own explicit engine-local scoping (a new function
# there is a smaller footprint than a new dataclass field, but board.py
# is still genuinely shared/consumed by Talos and Athena too, and
# this module's whole existence is scoped to avoid exactly that kind of
# cross-engine collision while parallel confidence-score work is
# happening elsewhere).
def _ring_radii_mm() -> tuple[float, ...]:
    """Freshly read `TREBLE_INNER_SCORING_RADIUS_MM`/`DOUBLE_INNER_
    SCORING_RADIUS_MM` from the board module ON EVERY CALL, not a
    module-level constant frozen at import time -- those two can be
    live-derived per session/board (`opendarts.geometry.board.
    set_ring_boundary_offsets()`, wired 2026-08-21), and a frozen import
    here would silently go stale the moment that happens, making this
    confidence estimate reason about wire distances that no longer match
    what `sector_ring_for_point()` itself actually scores against."""
    return (
        BULL_RADIUS_MM,
        OUTER_BULL_RADIUS_MM,
        _board_geom.TREBLE_INNER_SCORING_RADIUS_MM,
        TREBLE_OUTER_RADIUS_MM,
        _board_geom.DOUBLE_INNER_SCORING_RADIUS_MM,
        DOUBLE_OUTER_RADIUS_MM,
    )


_WIRE_ANGLES_DEG = tuple(wire_boundary_angle_deg(n) for n in SECTOR_NUMBERS_CLOCKWISE)


def _dist_to_nearest_wire_mm(x_mm: float, y_mm: float) -> float:
    """Distance (mm) from a board-plane point to the NEAREST scoring
    wire -- whichever is closer of (a) the 6 ring-boundary circles
    (bull/outer-bull/treble-inner/treble-outer/double-inner/double-outer,
    using the same measured SCORING radii `opendarts.geometry.board.
    sector_ring_for_point` itself scores against, not the regulation
    ones) or (b) the 20 radial sector wires. See module docstring
    signal 2 for why this is the single most informative real signal
    found in this investigation."""
    r = math.hypot(x_mm, y_mm)
    ring_dist = min(abs(r - rr) for rr in _ring_radii_mm())

    angle = math.degrees(math.atan2(x_mm, y_mm)) % 360.0
    sector_dist = min(
        r * math.radians(abs((angle - wa + 180.0) % 360.0 - 180.0))
        for wa in _WIRE_ANGLES_DEG
    )
    return min(ring_dist, sector_dist)


def compute_confidence(
    *,
    ok: bool,
    board_xy_mm: tuple[float, float] | None,
    max_ray_disagreement_mm: float | None,
    fallback_used: bool,
) -> float:
    """A calibrated P(this throw's sector/ring is correct), in [0, 1].
    See module docstring for the full derivation and validation.

    `ok=False` (no score at all -- `score_dart()` rejected the throw
    outright) always returns 0.0: there is no answer to have confidence
    IN, not a prediction that a wrong answer is likely. Every other
    input is required when `ok=True` (a throw that reached a real
    ScoreResult always has a triangulated point and a disagreement
    figure -- see `scoring.score_dart()`); missing them there would be a
    real upstream bug, not a value this function should silently paper
    over, so it is NOT defensive about that case.
    """
    if not ok:
        return 0.0
    assert board_xy_mm is not None
    assert max_ray_disagreement_mm is not None

    x_mm, y_mm = board_xy_mm
    wire_dist = _dist_to_nearest_wire_mm(x_mm, y_mm)
    wire_safe = min(wire_dist, WIRE_CAP_MM) / WIRE_CAP_MM
    disagree_risk = min(max_ray_disagreement_mm, DISAGREE_CAP_MM) / DISAGREE_CAP_MM
    fb = 1.0 if fallback_used else 0.0

    z = INTERCEPT + W_WIRE_SAFE * wire_safe + W_DISAGREE_RISK * disagree_risk + W_FALLBACK_RISK * fb
    return 1.0 / (1.0 + math.exp(-z))
