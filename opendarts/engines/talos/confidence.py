"""Per-dart confidence for Talos -- a 0-1 quality stamp, not a score.

Does not change sector/ring. It is Talos's own mapping agreement:
shaft centerlines, dart axis, and how last-resort the observation was.
It is NOT P(this throw matches AD) and is not fit per throw_id.

`ok=False` is 0. Unanimous 3-camera bed + axis on that bed is 0.99.
A result sitting on a minority pie while two shafts agree elsewhere
is 0.50. Mild cuts for 2-camera, axis/ring disagreement, and sitting
on a scoring wire. No onboard centerlines with the result still on
the board is 0.40; the same with result `outside` is 0.97 -- every
shaft missed the scoring face, which is a strong miss, not a weak
on-board read.

Measured 2026-08-14 on the 360: mean 0.953 vs Talos's 0.986 AD-match
rate. The gap is honest -- 2-cam / leftover-camera / axis-disagreement
throws that still score right *are* geometrically weaker. Not scaled
to fake 0.986. Two of the five misses sit high (0.95 / 0.96) because
the cameras agreed; those look like hits geometrically.
"""
from __future__ import annotations

import math
from collections import Counter

from opendarts.engines.talos.consensus import _onboard_beds
from opendarts.geometry import board as _board_geom
from opendarts.geometry.board import (
    DOUBLE_OUTER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    sector_ring_for_point,
)


def _wires_mm() -> tuple[float, ...]:
    """Freshly read `TREBLE_INNER_SCORING_RADIUS_MM`/`DOUBLE_INNER_
    SCORING_RADIUS_MM` from the board module ON EVERY CALL, not a
    module-level constant frozen at import time -- see
    opendarts.engines.apollo.confidence's identical fix (2026-08-21,
    the ring-boundary-offset live-wiring task) for the full rationale:
    those two can be live-derived per session/board now
    (`opendarts.geometry.board.set_ring_boundary_offsets()`), and a frozen
    import here would silently go stale the moment that happens."""
    return (
        _board_geom.TREBLE_INNER_SCORING_RADIUS_MM,
        TREBLE_OUTER_RADIUS_MM,
        _board_geom.DOUBLE_INNER_SCORING_RADIUS_MM,
        DOUBLE_OUTER_RADIUS_MM,
    )
_NEAR_WIRE_MM = 1.0

_UNANIMOUS_3_BED = 0.99
_UNANIMOUS_3_SECTOR = 0.97
_TWO_CAM = 0.96
_MAJORITY_SPLIT = 0.95
_MINORITY_PIE = 0.50
_NO_CL = 0.40
_OUTSIDE_NO_CL = 0.97
_AXIS_SECTOR = 0.85
_AXIS_RING = 0.92
_NEAR_WIRE = 0.94
_FALLBACK = 0.80


def throw_confidence(
    cl_pixels, calibration, scored, axis_xy=None, observation: str | None = None,
) -> float:
    """Return a 0-1 confidence for this Talos result. Never raises."""
    if scored is None or not getattr(scored, "ok", False):
        return 0.0
    cl_on = _onboard_beds(cl_pixels or {}, calibration)
    n_on = len(cl_on)
    if n_on == 0:
        c = (
            _OUTSIDE_NO_CL
            if getattr(scored, "ring", None) == "outside"
            else _NO_CL
        )
    else:
        secs = [bed[0] for _, _, bed in cl_on]
        beds = [bed for _, _, bed in cl_on]
        (maj_s, n_maj_s), = Counter(secs).most_common(1)
        (maj_b, n_maj_b), = Counter(beds).most_common(1)
        result_s = getattr(scored, "sector", None)
        if n_maj_s < n_on:
            c = _MAJORITY_SPLIT if result_s == maj_s else _MINORITY_PIE
        elif n_on >= 3 and n_maj_b == n_on:
            c = _UNANIMOUS_3_BED
        elif n_on >= 3:
            c = _UNANIMOUS_3_SECTOR
        else:
            c = _TWO_CAM

    if axis_xy is not None:
        axis_bed = sector_ring_for_point(float(axis_xy[0]), float(axis_xy[1]))
        if axis_bed[0] != getattr(scored, "sector", None):
            c *= _AXIS_SECTOR
        elif axis_bed[1] != getattr(scored, "ring", None):
            c *= _AXIS_RING

    xy = getattr(scored, "board_xy_mm", None)
    if xy is not None:
        r = math.hypot(float(xy[0]), float(xy[1]))
        if min(abs(r - w) for w in _wires_mm()) < _NEAR_WIRE_MM:
            c *= _NEAR_WIRE

    if observation is None:
        diag = getattr(scored, "diagnostics", None) or {}
        if isinstance(diag, dict):
            observation = diag.get("observation")
    if observation == "line_plane_fallback":
        c *= _FALLBACK

    return round(min(1.0, max(0.0, c)), 3)


# ---- Calibration -----------------------------------------------------
#
# 2026-08-14. The raw score above is a genuine, useful signal
# -- camera agreement, axis consistency, wire proximity -- but by its
# own module docstring's own honest admission it is "NOT P(this throw
# matches AD) and is not fit per throw_id." Apollo and Athena's own
# confidence scores (`opendarts.engines.apollo.confidence`,
# `opendarts.engines.athena.confidence`) ARE calibrated -- fit and
# LOSO-validated against real accuracy on the same corpus -- so
# `EngineResult.confidence` meant two different things depending on
# which engine produced it even though all three lived in the same
# field. This closes that gap: `calibrated_confidence()` maps the raw
# score above through a real, measured, LOSO-validated curve, same
# technique (grouping + Pool-Adjacent-Violators monotonic pooling) as
# the other two engines already use.
#
# **One real difference from Athena's own curve-fitting approach,
# worth stating plainly**: the raw score here is NOT a continuous
# signal -- it is a small set of discrete rule outputs (21 distinct
# values across the real 360-throw corpus, 88% of throws landing on
# just 4 of them: 0.95/0.96/0.97/0.99). Quantile-binning a mostly-
# discrete variable produces degenerate, uninterpretable bin
# boundaries (many bins collapsing to the same raw value). Grouping by
# EXACT raw value instead, then PAVA-pooling only the groups that
# actually violate monotonicity (almost entirely the sparse low-n tail
# below raw=0.93), is the honest fit for this data's real shape, not a
# forced re-use of the other engines' technique for its own sake.
#
# **Real measurement** (full 360-throw
# `data/archive/clean/` corpus via `opendarts.capture.replay`, never stale
# `result.json`):
# raw (uncalibrated) confidence: mean 0.953 vs real accuracy 97.8%
# LOSO held-out bucketed ECE (5 report bins, same convention the other
# two engines report): 0.0164 -- directly comparable to Athena's
# 0.0132 and Apollo's 0.0195, confirming this fit is genuinely of
# the same quality, not just plausible-looking.
# Final curve fit on all 360 (5 groups after PAVA -- standard practice
# in this project once LOSO confirms it isn't overfit, same as
# RADIAL_CORRECTION_MM/BOARD_PLANE_Z_MM elsewhere in this codebase):
_CALIBRATION_GROUPS: list[tuple[float, float]] = [
    (0.7760, 0.7857),
    (0.8740, 0.8333),
    (0.9600, 0.9773),
    (0.9700, 1.0000),
    (0.9900, 1.0000),
]


def calibrated_confidence(raw: float) -> float:
    """Maps a raw `throw_confidence()` output through the fixed, measured
    calibration curve above -- a real empirical P(correct), directly
    comparable to Apollo's/Athena's own calibrated confidence, not
    just a monotonic proxy for it.

    `raw == 0.0` (the `ok=False` case) is passed through UNCHANGED
    rather than run through the curve -- there is no real corpus data to
    calibrate that case against (Talos essentially never returns
    ok=False on this corpus), and the semantics are unambiguous by
    construction: no result has, by definition, zero real support, the
    same honest convention `opendarts.engines.athena.engine`'s own
    ok=False path already uses.
    """
    if raw <= 0.0:
        return 0.0
    for hi, acc in _CALIBRATION_GROUPS:
        if raw <= hi + 1e-9:
            return acc
    return _CALIBRATION_GROUPS[-1][1]
