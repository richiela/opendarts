"""Per-installation derivation of ``opendarts.geometry.board_color``'s raw
pixel classification thresholds (``BRIGHTNESS_THRESHOLD_BLACK_CREAM``,
``CHROMA_THRESHOLD``) from real sampled pixels -- an offline, validation-
stage companion to that module, **not wired into its live classifier**
(see "Status" at the bottom of this docstring).

## The idea

``board_color.py``'s own module docstring documents two different kinds
of constant it carries:

1. **The color PHASE** (``_EVEN_PARITY_SINGLE = "black"`` etc,
   ``sector_single_color()`` / ``sector_accent_color()``) -- confirmed
   real WDF/PDC regulation: which of the 20 sectors is black/cream and
   red/green alternates in a fixed, universal pattern. This is a known
   answer key, not something specific to this rig.
2. **The raw pixel thresholds** (``BRIGHTNESS_THRESHOLD_BLACK_CREAM =
   129.0``, ``CHROMA_THRESHOLD = 35.0``) -- measured ONCE from this
   specific rig's own real images/camera color response/lighting, and
   hardcoded. A different camera or lighting setup would need its own
   values.

Since (1) is universal, it can be used as a known-truth label to derive
(2) automatically, per installation: forward-project each of the 20
sectors' known single/treble/double positions (plus bull/outer_bull)
into a camera's real background image (``cv2.projectPoints()``, reusing
``board_color.sample_board_color()``'s own primitives -- no new
projection code here), sample the real pixel, and since we already know
which samples SHOULD be black vs cream (single beds) and which SHOULD be
achromatic vs chromatic (single beds vs treble/double/bull/outer_bull),
take the midpoint of the real measured gap between the two groups --
exactly the methodology ``board_color.py``'s own docstring used to
derive 129.0/35.0 in the first place (see its "Classification
thresholds -- measured, not guessed" section), just automated and rerun
fresh per session/installation instead of once-and-hardcoded.

This is the same "derive from images using a known-universal fact as
ground truth" pattern already established in this codebase by
the offline session pose refit (re-solving a session's camera pose
from stored bg frames through the current landmark stack) --
this module is the color-threshold analogue, built the same way: reads
a session's real packages (bg frames + calibration, preferring a
``calibration_refit.json`` when present via
``opendarts.capture.throw_package.load_throw_package()``'s own existing
precedence), and writes a non-destructive sibling JSON file next to the
session, matching that refit's ``calibration_refit.json`` pattern
exactly (``REFIT_FILENAME`` -> ``BOARD_COLOR_CALIBRATION_FILENAME``
here).

## PATCH_RADIUS_PX is deliberately NOT re-derived here

``board_color.py``'s own docstring already independently swept patch
size 1x1 through 25x25 and found a peak at 7x7 (radius 3) -- accuracy
climbs from 93.9% (1x1) to 95.9% at 7x7, then falls off (94.8% at 11x11,
89.1% at 17x17, 74.6% at 25x25) because too-large a patch starts
spanning into a neighboring band/wire. That falloff shape is a
geometric fact about how wide the real paint bands are relative to a
sampled pixel window, at whatever image resolution the camera captures
at -- not a camera-color-response property the way brightness/chroma
are. ``sweep_patch_radius()`` below DOES let a caller re-run that same
sweep per-session as a diagnostic (it is cheap -- the sweep only
re-samples already-projected pixel patches at different radii, no
re-projection or camera re-solve needed per radius), and this task's own
corpus validation used it to confirm the peak stays near radius 3 across
real sessions -- but the fixed ``PATCH_RADIUS_PX`` import from
``board_color`` is what this module actually classifies with by default;
nothing here promotes a per-session radius into
``derive_session_board_color_calibration()``'s own output as a value to
adopt. See this module's own corpus validation report for the real
per-session sweep numbers.

## Status

**Additive, validation-stage code -- NOT wired into
``board_color.py``'s live ``classify_bgr()``/``BRIGHTNESS_THRESHOLD_
BLACK_CREAM``/``CHROMA_THRESHOLD`` module-level constants, and not
wired into any live scoring path.** Matches this project's own standing
discipline (``board_color.py``'s own ``evaluate_color_sanity()``
docstring: "measure before building any auto-correction") -- this
module produces and validates a fresh derivation; deciding whether/how
to actually swap the live thresholds for a derived-per-installation
value is a separate, deliberate step for whoever owns that decision,
not assumed here.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from opendarts.geometry.board import (
    BULL_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    SECTOR_NUMBERS_CLOCKWISE,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    polar_to_xy_mm,
    sector_center_angle_deg,
)
from opendarts.geometry.board_color import (
    PATCH_RADIUS_PX,
    expected_color,
    project_board_point_px,
    sample_patch_bgr,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Reference sample positions -- board-plane (mm) midpoints of every band
# this task's known-universal color phase can label in advance. Same
# midpoint-of-band convention tests/test_board_color.py's own corpus
# validation test already uses for single/treble/double; extended here
# with a separate single_inner/single_outer midpoint each (rather than
# one shared "single" midpoint) plus bull/outer_bull, for more real
# sample diversity per session.
# ---------------------------------------------------------------------

BULL_SAMPLE_RADIUS_MM = BULL_RADIUS_MM / 2.0
OUTER_BULL_SAMPLE_RADIUS_MM = (BULL_RADIUS_MM + OUTER_BULL_RADIUS_MM) / 2.0
SINGLE_INNER_SAMPLE_RADIUS_MM = (OUTER_BULL_RADIUS_MM + TREBLE_INNER_RADIUS_MM) / 2.0
SINGLE_OUTER_SAMPLE_RADIUS_MM = (TREBLE_OUTER_RADIUS_MM + DOUBLE_INNER_RADIUS_MM) / 2.0
TREBLE_SAMPLE_RADIUS_MM = (TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2.0
DOUBLE_SAMPLE_RADIUS_MM = (DOUBLE_INNER_RADIUS_MM + DOUBLE_OUTER_RADIUS_MM) / 2.0

# Ring -> "achromatic" (single beds, expected black/cream) vs
# "chromatic" (treble/double/bull/outer_bull, expected red/green) -- a
# structural fact about which bands carry paint accent color, same as
# board_color.py's own black/cream vs red/green split; universal, not
# measured per-installation.
_ACHROMATIC_RINGS = ("single_inner", "single_outer")


@dataclass(frozen=True)
class ReferencePoint:
    """One known board-plane position this task can label in advance
    from the universal color phase alone."""

    sector: int | None  # None for bull/outer_bull
    ring: str
    xy_mm: tuple[float, float]
    expected_color: str
    group: str  # "achromatic" or "chromatic"


def reference_points() -> list[ReferencePoint]:
    """All 20 sectors' single_inner/single_outer/treble/double band
    midpoints + bull/outer_bull = 82 known-labeled board-plane
    positions. Pure geometry (no images) -- the same list is reused for
    every camera and every sampled package."""
    points: list[ReferencePoint] = []
    for number in SECTOR_NUMBERS_CLOCKWISE:
        angle = sector_center_angle_deg(number)
        for ring, radius in (
            ("single_inner", SINGLE_INNER_SAMPLE_RADIUS_MM),
            ("single_outer", SINGLE_OUTER_SAMPLE_RADIUS_MM),
            ("treble", TREBLE_SAMPLE_RADIUS_MM),
            ("double", DOUBLE_SAMPLE_RADIUS_MM),
        ):
            xy = polar_to_xy_mm(radius, angle)
            color = expected_color(str(number), ring)
            group = "achromatic" if ring in _ACHROMATIC_RINGS else "chromatic"
            points.append(ReferencePoint(number, ring, xy, color, group))
    # Bull/outer_bull are full circles centered on the board -- angle is
    # arbitrary, sample once at 0deg (camera/package diversity still
    # gives multiple real samples of this same physical position).
    for ring, radius in (("bull", BULL_SAMPLE_RADIUS_MM), ("outer_bull", OUTER_BULL_SAMPLE_RADIUS_MM)):
        xy = polar_to_xy_mm(radius, 0.0)
        color = expected_color(None, ring)
        points.append(ReferencePoint(None, ring, xy, color, "chromatic"))
    return points


# ---------------------------------------------------------------------
# Sampling: project + patch-sample every reference point in one package
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ColorSample:
    """One real (projected, patch-averaged) pixel sample at a known-
    labeled reference position."""

    package_id: str
    camera: int
    sector: int | None
    ring: str
    expected_color: str
    group: str
    b: float
    g: float
    r: float

    @property
    def brightness(self) -> float:
        return (self.b + self.g + self.r) / 3.0

    @property
    def chroma(self) -> float:
        return max(self.b, self.g, self.r) - min(self.b, self.g, self.r)


def project_reference_points_px(
    calib, points: list[ReferencePoint] | None = None
) -> list[tuple[float, float]]:
    """Forward-project every reference point into one camera's pixel
    space -- the exact per-point ``project_board_point_px()`` calls
    ``collect_color_samples()`` makes, hoisted out (2026-09-11
    calibration-speed pass) so a caller sampling MANY frames of the same
    camera against the same calibration (the live calibration burst
    samples its whole ~50-frame raw pool) projects once instead of once
    per frame. The projection depends only on (point, calibration),
    never on the frame, so reuse is exactly equivalent."""
    points = points if points is not None else reference_points()
    return [project_board_point_px(pt.xy_mm, calib) for pt in points]


def collect_color_samples(
    package_id: str,
    calibrations: dict[int, object],
    bg_images: dict[int, np.ndarray],
    patch_radius: int = PATCH_RADIUS_PX,
    points: list[ReferencePoint] | None = None,
    projected_px_by_camera: dict[int, list[tuple[float, float]]] | None = None,
) -> list[ColorSample]:
    """Sample every reference point across every camera that has both a
    calibration and a background image for one throw package. A point
    that projects off-frame for a given camera is simply skipped for
    that camera (matches ``sample_patch_bgr()``'s own "center pixel must
    be in-frame" gate in ``board_color.py``).

    ``projected_px_by_camera``, if given, maps a camera id to
    ``project_reference_points_px(calib, points)`` for that camera's own
    calibration -- a pure caching hook (see that helper's docstring);
    cameras absent from it are projected here exactly as before."""
    points = points if points is not None else reference_points()
    samples: list[ColorSample] = []
    for cam, calib in calibrations.items():
        bg = bg_images.get(cam)
        if bg is None:
            continue
        projected = (projected_px_by_camera or {}).get(cam)
        for pt_i, pt in enumerate(points):
            if projected is not None:
                px, py = projected[pt_i]
            else:
                px, py = project_board_point_px(pt.xy_mm, calib)
            ix, iy = int(round(px)), int(round(py))
            bgr = sample_patch_bgr(bg, ix, iy, patch_radius)
            if bgr is None:
                continue
            b, g, r = bgr
            samples.append(
                ColorSample(
                    package_id=package_id, camera=cam, sector=pt.sector, ring=pt.ring,
                    expected_color=pt.expected_color, group=pt.group, b=b, g=g, r=r,
                )
            )
    return samples


def _classify_with(b: float, g: float, r: float, brightness_threshold: float, chroma_threshold: float) -> str:
    """Mirrors ``board_color.classify_bgr()``'s exact two-stage
    structure, parameterized on the thresholds being derived here --
    can't call ``classify_bgr()`` itself, since that hardcodes the
    module-level constants this function exists to derive fresh
    values for."""
    brightness = (b + g + r) / 3.0
    chroma = max(b, g, r) - min(b, g, r)
    if chroma < chroma_threshold:
        return "black" if brightness < brightness_threshold else "cream"
    return "red" if r > g else "green"


# ---------------------------------------------------------------------
# Threshold derivation: real measured gap, midpoint -- same methodology
# board_color.py's own docstring used, automated
# ---------------------------------------------------------------------

# A gap (in the same 0-255 BGR-mean units as brightness/chroma) below
# this is flagged "low" confidence rather than trusted outright; a gap
# <= 0 (the two groups' robust ranges overlap) is "invalid" -- still
# reported (the midpoint is still computed), never silently substituted
# or hidden, per this project's standing confidence-gate discipline.
# Chosen conservatively relative to the real measured gaps in
# board_color.py's own docstring (28.6 brightness) -- not independently
# re-swept per installation, a reasonable fixed bar rather than a
# re-derived one.
GAP_HIGH_CONFIDENCE_THRESHOLD = 15.0

# **Real finding from this module's own corpus validation, not assumed
# up front**: using literal max()/min() (what board_color.py's own
# original one-time-by-hand derivation effectively did, reporting a
# real range like "45.8-114.7") is NOT robust once derivation is
# automated over hundreds of real samples per session instead of a
# hand-picked few -- a single shadowed/wire-crossing/edge-clipped
# outlier sample inverts the whole group extreme and reports a
# spuriously "invalid" (overlapping) gap even when the bulk of the
# distribution is cleanly separated (measured on one recorded
# session's 4 packages: literal max(black)=115.4 > literal
# min(cream)=41.2 -- both single-sample outliers -- while a robust
# 95th/5th-percentile read of the SAME data gives 96.9/219.0, a clean
# +122 gap). Trimming ROBUST_PERCENTILE from each tail before taking
# the group extreme is a standard robust-statistics choice (matches
# board_color.py's own docstring already reaching for a 99th-percentile
# stat on the chroma side, just applied consistently to both
# thresholds here rather than only one). 5.0 is not independently
# swept per installation -- a fixed, documented choice.
ROBUST_PERCENTILE = 5.0


@dataclass
class ThresholdDerivation:
    """One derived threshold (brightness or chroma): the real measured
    midpoint between two known-labeled groups, plus the robust gap
    statistics a caller needs to judge how much to trust it. ``low_
    group_stat``/``high_group_stat`` are ROBUST edges (the
    ``ROBUST_PERCENTILE``-trimmed extreme of each group, not the
    literal max/min -- see ``ROBUST_PERCENTILE``'s own comment for why
    literal max/min proved too outlier-sensitive once this derivation
    ran against real per-session data)."""

    value: float | None
    low_group_stat: float | None  # robust (percentile-trimmed) upper edge of the "low" group
    high_group_stat: float | None  # robust (percentile-trimmed) lower edge of the "high" group
    low_group_n: int
    high_group_n: int
    gap: float | None  # high_group_stat - low_group_stat; positive = clean separation
    confidence: str  # "high" / "low" / "invalid" / "insufficient_data"

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "low_group_stat": self.low_group_stat,
            "high_group_stat": self.high_group_stat,
            "low_group_n": self.low_group_n,
            "high_group_n": self.high_group_n,
            "gap": self.gap,
            "confidence": self.confidence,
        }


def _derive_gap_threshold(
    low_values: list[float], high_values: list[float], robust_percentile: float = ROBUST_PERCENTILE
) -> ThresholdDerivation:
    if not low_values or not high_values:
        return ThresholdDerivation(
            value=None, low_group_stat=None, high_group_stat=None,
            low_group_n=len(low_values), high_group_n=len(high_values),
            gap=None, confidence="insufficient_data",
        )
    low_stat = float(np.percentile(low_values, 100.0 - robust_percentile))
    high_stat = float(np.percentile(high_values, robust_percentile))
    value = (low_stat + high_stat) / 2.0
    gap = high_stat - low_stat
    if gap >= GAP_HIGH_CONFIDENCE_THRESHOLD:
        confidence = "high"
    elif gap > 0:
        confidence = "low"
    else:
        confidence = "invalid"
    return ThresholdDerivation(
        value=value, low_group_stat=low_stat, high_group_stat=high_stat,
        low_group_n=len(low_values), high_group_n=len(high_values),
        gap=gap, confidence=confidence,
    )


@dataclass
class BoardColorCalibrationResult:
    """Everything a fresh per-installation derivation produces: the two
    derived thresholds, the patch radius they were derived at, and a
    self-consistency accuracy re-check (same discipline as
    board_color.py's own docstring: "compare to the color pattern this
    SAME measurement independently derived") using the DERIVED
    thresholds instead of the hardcoded ones."""

    brightness_threshold: ThresholdDerivation
    chroma_threshold: ThresholdDerivation
    patch_radius: int
    n_samples: int
    n_packages: int  # packages that actually contributed >=1 sample
    n_packages_attempted: int
    accuracy_single_camera: float | None
    accuracy_majority_vote: float | None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "brightness_threshold_black_cream": self.brightness_threshold.to_dict(),
            "chroma_threshold": self.chroma_threshold.to_dict(),
            "patch_radius_px": self.patch_radius,
            "n_samples": self.n_samples,
            "n_packages": self.n_packages,
            "n_packages_attempted": self.n_packages_attempted,
            "accuracy_single_camera": self.accuracy_single_camera,
            "accuracy_majority_vote": self.accuracy_majority_vote,
            "warnings": self.warnings,
        }


def derive_thresholds(
    samples: list[ColorSample],
    patch_radius: int = PATCH_RADIUS_PX,
    robust_percentile: float = ROBUST_PERCENTILE,
) -> BoardColorCalibrationResult:
    """Core derivation: given a pool of real labeled samples (from
    however many packages/cameras a caller collected), derive both
    thresholds and re-check self-consistency accuracy against them."""
    achromatic = [s for s in samples if s.group == "achromatic"]
    chromatic = [s for s in samples if s.group == "chromatic"]

    black_brightness = [s.brightness for s in achromatic if s.expected_color == "black"]
    cream_brightness = [s.brightness for s in achromatic if s.expected_color == "cream"]
    achromatic_chroma = [s.chroma for s in achromatic]
    chromatic_chroma = [s.chroma for s in chromatic]

    brightness_result = _derive_gap_threshold(black_brightness, cream_brightness, robust_percentile)
    chroma_result = _derive_gap_threshold(achromatic_chroma, chromatic_chroma, robust_percentile)

    warnings: list[str] = []
    if brightness_result.confidence not in ("high",):
        warnings.append(
            f"brightness threshold confidence={brightness_result.confidence} "
            f"(gap={brightness_result.gap})"
        )
    if chroma_result.confidence not in ("high",):
        warnings.append(
            f"chroma threshold confidence={chroma_result.confidence} (gap={chroma_result.gap})"
        )

    n_packages = len({s.package_id for s in samples})

    if brightness_result.value is None or chroma_result.value is None:
        # Can't classify anything meaningfully without both thresholds.
        return BoardColorCalibrationResult(
            brightness_threshold=brightness_result,
            chroma_threshold=chroma_result,
            patch_radius=patch_radius,
            n_samples=len(samples),
            n_packages=n_packages,
            n_packages_attempted=n_packages,
            accuracy_single_camera=None,
            accuracy_majority_vote=None,
            warnings=warnings,
        )

    by_position: dict[tuple, list[tuple[int, str, str]]] = {}
    n_correct = 0
    for s in samples:
        predicted = _classify_with(s.b, s.g, s.r, brightness_result.value, chroma_result.value)
        if predicted == s.expected_color:
            n_correct += 1
        key = (s.package_id, s.sector, s.ring)
        by_position.setdefault(key, []).append((s.camera, predicted, s.expected_color))

    accuracy_single_camera = n_correct / len(samples) if samples else None

    n_positions = 0
    n_positions_correct = 0
    for votes in by_position.values():
        counts: dict[str, int] = {}
        for _, pred, _exp in votes:
            counts[pred] = counts.get(pred, 0) + 1
        best = max(counts.values())
        majority = next(pred for _, pred, _ in votes if counts[pred] == best)
        expected = votes[0][2]
        n_positions += 1
        if majority == expected:
            n_positions_correct += 1
    accuracy_majority_vote = n_positions_correct / n_positions if n_positions else None

    return BoardColorCalibrationResult(
        brightness_threshold=brightness_result,
        chroma_threshold=chroma_result,
        patch_radius=patch_radius,
        n_samples=len(samples),
        n_packages=n_packages,
        n_packages_attempted=n_packages,
        accuracy_single_camera=accuracy_single_camera,
        accuracy_majority_vote=accuracy_majority_vote,
        warnings=warnings,
    )


# ---------------------------------------------------------------------
# Storage format, and the session-level READER.
#
# The session-level DERIVE/WRITE/SWEEP entry points are
# dev/calibration/board_color_session.py: re-deriving a session's
# thresholds from its stored packages is a developer operation. The live
# Calibrate button derives thresholds from the frames it just captured,
# through collect_color_samples()/derive_thresholds() above, and writes
# them with result_to_payload() below -- one payload shape, whichever
# path produced the result -- and load_throw_package() reads a session
# file back through load_session_board_color_calibration().
# ---------------------------------------------------------------------

BOARD_COLOR_CALIBRATION_FILENAME = "board_color_calibration.json"
SCHEMA = "board-color-calibration-v1"


def result_to_payload(
    result: BoardColorCalibrationResult, *, solved_by: str | None = None
) -> dict:
    """Serialize a `BoardColorCalibrationResult` to the one JSON shape
    this derivation has ever written -- by
    `opendarts.live.capture_daemon.bootstrap_calibrations()`'s LIVE
    derivation (2026-08-27, the v2 package schema), which persists
    it into a calibration event's own `derived_calibration.json` (see
    `opendarts.capture.calibration_package`), and by the offline
    session-level writer,
    `dev.calibration.board_color_session.write_session_board_color_
    calibration()`. One schema, one payload builder, regardless of which
    caller produced `result`. `solved_by` lets each caller attribute the
    payload honestly; defaults to the offline description."""
    return {
        "schema": SCHEMA,
        "solved_by": solved_by or (
            "dev.calibration.board_color_session "
            "(offline, from stored bg frames + calibration/refit)"
        ),
        **result.to_dict(),
    }

def load_session_board_color_calibration(session_dir: Path) -> dict | None:
    """Load a previously written session derivation as a plain dict, or
    None if the session has no such sibling file OR its schema doesn't
    match this module's current `SCHEMA`.

    **Schema check, added 2026-08-21 (verifier finding on this task's
    own live-wiring pass)**: matches `opendarts.calibration.ring_boundary_
    offset.load_session_ring_boundary_offset()`'s identical guard
    exactly, added there for the identical reason -- a schema mismatch
    is treated exactly like "no file at all" (safe fallback to the
    hardcoded default), not a hard crash inside `load_throw_package()`.
    No such mismatch exists on the real corpus today (this module has
    only ever had one schema version), but the failure mode without this
    guard is real: `opendarts.capture.throw_package.load_throw_package()`
    directly indexes into this payload's keys with no schema check of
    its own, so a hypothetical future schema change (following the same
    real precedent `ring_boundary_offset.py`'s own v1->v2 migration set)
    would otherwise crash every throw load for any session still
    carrying an old-schema file, instead of gracefully falling back.

    **Corrupt-file guard, added 2026-08-21 (OpenDarts pre-hardware-audit
    finding, confirmed to apply here too -- identical to
    `opendarts.calibration.ring_boundary_offset.load_session_ring_boundary_
    offset()`'s own sibling fix, same reasoning)**: `json.loads()` used
    to be unguarded -- a truncated/corrupt file would raise straight
    through `load_throw_package()`, before `set_board_color_thresholds()`
    ever runs for that call, defeating the "never leak a prior session's
    thresholds into a later load" guarantee. Now treated exactly like
    "no file at all," loudly logged rather than silently swallowed."""
    path = Path(session_dir) / BOARD_COLOR_CALIBRATION_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        log.exception(
            "%s: failed to read/parse -- treating as absent (falls back to "
            "the hardcoded default, same as no file at all)", path,
        )
        return None
    if payload.get("schema") != SCHEMA:
        return None
    return payload
