"""Board-paint color sanity check — a companion to
``opendarts.geometry.board``'s ``sector_ring_for_point()``/
``sector_ring_to_token()``, built for the idea described directly:
"can we create something as a final sanity check that can find the tip
and see what color bed it sits in? ... If a dart lands in a sector that
doesn't match that color... we need to do something with that data."

This module is deliberately split into two independent halves, matching
the task's own two steps:

1. **The color-pattern model** (``expected_color()``): given a scored
   ``(sector, ring)``, what color should the board's own paint show
   there on a real board? Pure lookup table, no image data involved.
2. **The color sampler** (``sample_board_color()`` /
   ``sample_board_color_multi_camera()``): given a triangulated
   board-plane point and a camera's solved calibration, forward-project
   into that camera's pixel space (``cv2.projectPoints``, the exact
   primitive already used by
   ``opendarts.engines.athena.board_gate.project_board_roi_polygon()``)
   and classify the real paint color sampled from the camera's
   background image at that pixel.

Comparing the two outputs (``expected_color(sector, ring) != sampled``)
is the actual sanity check -- callers do that comparison themselves via
``color_agrees()``; this module never decides on its own whether a
disagreement should flag or correct a score (see the corpus-wide
measurement this shipped alongside, referenced below, for why: whether
disagreement is even predictive of a real miss had to be MEASURED before
building anything that acts on it, not assumed).

## Real board measured, not the generic description assumed

The commonly-quoted description has sector 20 as "cream single with green
treble/double" and sector 1 as "black with red treble/double". **The
real corpus board (a Winmau Blade 6) measured here is the MIRROR of
that pairing**: sector 20 is BLACK single / RED treble+double, sector 1
is CREAM single / GREEN treble+double -- confirmed by forward-projecting
every one of the 20 sectors' single-bed midpoint into a real camera and
reading the real background pixel, not by trusting that description
literally (exactly the "measure, don't guess" discipline
``docs/DESIGN.md``/``opendarts/geometry/board.py``'s own
``INNER_RING_SCORING_OFFSET_MM`` already established for this project).
The alternation itself (adjacent sectors always opposite color, in sync
between single and treble/double) matches the project's description and is
also confirmed -- only the absolute phase (which specific sectors are
which color) needed correcting from real data.

Real measurement: ``cv2.projectPoints()`` of the single/treble/double
midpoint radius for all 20 sectors (plus bull/outer_bull), sampled from
3 different throw packages (one per corpus session, for lighting
diversity) x 3 cameras each = 558 real background-pixel samples. The
capture script and its raw samples were scratch, not committed, but the
measurement is rerunnable against any corpus package's own
``calibration.json`` + ``cam*_bg.png``.
Every sector showed a perfectly clean, unambiguous alternation (idx 0/
even = black single + red treble/double, idx 1/odd = cream single +
green treble/double, using ``SECTOR_NUMBERS_CLOCKWISE``'s own index
order) with zero exceptions across all 20 sectors x 3 packages x 3
cameras -- see ``SECTOR_PARITY_TO_SINGLE_COLOR``/
``SECTOR_PARITY_TO_ACCENT_COLOR`` below.

## Classification thresholds -- measured, not guessed

Same 558-sample measurement above also produced the real BGR
distributions the classifier's two thresholds are set from:

- ``black`` (single, even-parity sectors): brightness (mean BGR) 45.8-
  114.7, mean 68.4.
- ``cream`` (single, odd-parity sectors): brightness 143.2-252.6, mean
  235.6.
  -> clean brightness gap of 28.6 between the two (114.7 to 143.2);
  ``BRIGHTNESS_THRESHOLD_BLACK_CREAM`` = 129.0, the gap's midpoint.
- ``red``/``green`` (treble/double/bull/outer_bull): separated by
  chroma (``max(B,G,R) - min(B,G,R)``) from the achromatic black/cream
  pair -- black/cream chroma sits at its 99th percentile by ~35 (black
  27.8, cream 34.9), while red/green's MEDIAN chroma is 63-108 (its own
  low tail, down to single digits, is real -- see below).
  ``CHROMA_THRESHOLD`` = 35.0. Within the chromatic branch, plain
  ``R > G`` cleanly separates red (mean R-G=+105, min per-sample -10.9)
  from green (mean R-G=-87, max per-sample +4.6) with no threshold
  tuning needed.

Self-consistency accuracy of this exact classifier (predict color from
sampled BGR, compare to the color pattern this SAME measurement
independently derived): **95.9% single-camera (535/558)**, **100.0%
majority-vote-of-3-cameras (186/186 physical points)** -- every single-
camera error is a real, low-brightness/low-chroma sample (a shadowed
double-ring pixel near a board edge, camera1's own darker corner in two
of the three sampled packages) that a majority vote across the other two
cameras' independent views corrects. This is why
``sample_board_color_multi_camera()`` below is the recommended entry
point, not a single-camera sample.

Patch size (the small-averaging-window-around-the-projected-pixel
question) was independently swept 1x1 through 25x25 against this same
558-point measurement: accuracy climbs from 93.9% (1x1, a single pixel)
to a peak of 95.9% at 7x7 (radius 3), then falls off past that (94.8%
at 11x11, 89.1% at 17x17, 74.6% at 25x25 -- the patch starts spanning
into a neighboring band/wire once it's too large). ``PATCH_RADIUS_PX``
= 3 (7x7) is the measured peak, not a round-number guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opendarts.geometry.board import SECTOR_NUMBERS_CLOCKWISE

# ---------------------------------------------------------------------
# 1. The color-pattern model
# ---------------------------------------------------------------------

# Real board colors this module classifies into. "outside" and any ring
# this project doesn't assign a color to map to None (see
# expected_color() below), not a 5th vocabulary entry -- keeps this a
# closed, physically-real set.
BoardColor = str # one of "black", "cream", "red", "green"

_SECTOR_INDEX = {number: idx for idx, number in enumerate(SECTOR_NUMBERS_CLOCKWISE)}

# Measured (see module docstring): even index in SECTOR_NUMBERS_CLOCKWISE
# (starting with sector 20 itself) is a BLACK single bed / RED treble+
# double; odd index is CREAM / GREEN. This is a real finding, mirrored
# from the commonly-quoted description -- verified against 3 real corpus
# packages' own background images before being hardcoded here, not
# assumed.
_EVEN_PARITY_SINGLE = "black"
_EVEN_PARITY_ACCENT = "red"
_ODD_PARITY_SINGLE = "cream"
_ODD_PARITY_ACCENT = "green"

# Rings sector_ring_for_point() can return that have a real, fixed
# expected color independent of sector.
_BULL_COLOR = "red"
_OUTER_BULL_COLOR = "green"


def sector_single_color(sector_number: int) -> BoardColor:
    """The real single-bed color (``"black"`` or ``"cream"``) for a
    sector NUMBER (int, e.g. ``20``, not the string label
    ``sector_ring_for_point()`` returns) -- see module docstring for the
    real measured alternation this is derived from."""
    idx = _SECTOR_INDEX[sector_number]
    return _EVEN_PARITY_SINGLE if idx % 2 == 0 else _ODD_PARITY_SINGLE


def sector_accent_color(sector_number: int) -> BoardColor:
    """The real treble/double-bed color (``"red"`` or ``"green"``) for a
    sector NUMBER -- same alternation as ``sector_single_color()``, always
    the OTHER pair member."""
    idx = _SECTOR_INDEX[sector_number]
    return _EVEN_PARITY_ACCENT if idx % 2 == 0 else _ODD_PARITY_ACCENT


def expected_color(sector: str | None, ring: str) -> BoardColor | None:
    """The real board paint color a human would see at a scored
    ``(sector, ring)`` -- ``sector``/``ring`` in
    ``opendarts.geometry.board.sector_ring_for_point()``'s own vocabulary
    (``sector`` is the wedge number AS A STRING or None; ``ring`` in
    ``{"bull", "outer_bull", "treble", "double", "single_inner",
    "single_outer", "outside"}``).

    Returns ``None`` when no single, real color applies:
    - ``ring == "outside"`` -- off the board entirely, no paint to check.
    - Any ring that legitimately needs a sector but got ``sector=None``
      (shouldn't happen from a real ``sector_ring_for_point()`` call, but
      handled honestly rather than raising, since this function may see
      hand-constructed/replayed inputs too).
    """
    if ring == "outside":
        return None
    if ring == "bull":
        return _BULL_COLOR
    if ring == "outer_bull":
        return _OUTER_BULL_COLOR
    if sector is None:
        return None
    try:
        number = int(sector)
    except (TypeError, ValueError):
        return None
    if number not in _SECTOR_INDEX:
        return None
    if ring in ("single_inner", "single_outer"):
        return sector_single_color(number)
    if ring in ("treble", "double"):
        return sector_accent_color(number)
    return None


# ---------------------------------------------------------------------
# 2. The color sampler
# ---------------------------------------------------------------------

# Measured (module docstring): the peak of a 1x1..25x25 sweep against
# the real 558-sample corpus measurement. Not a round-number guess.
PATCH_RADIUS_PX = 3

# Measured (module docstring): midpoint of the real black/cream
# brightness gap (114.7 to 143.2) found across 180 real single-bed
# samples spanning all 3 corpus sessions.
BRIGHTNESS_THRESHOLD_BLACK_CREAM = 129.0

# Measured (module docstring): just above black/cream's own 99th-
# percentile chroma (34.9), safely below red/green's median chroma
# (63-108) -- separates "achromatic paint" from "colored paint" before
# red vs. green is decided by simple R-vs-G comparison.
CHROMA_THRESHOLD = 35.0


# LIVE-DERIVED BOARD-COLOR THRESHOLDS, wired 2026-08-21 (this
# integration task, lowest risk of the 5 items -- a downstream sanity
# check, not primary scoring, see opendarts.geometry.board_color_calibration's
# own module docstring for the full image-only derivation).
# Mutable module globals, same propagation mechanism as
# opendarts.geometry.board.set_ring_boundary_offsets() and
# opendarts.capture.board_disc.set_calibrated_board_disc_masks():
# classify_bgr() below resolves these two names from THIS module's own
# global namespace at call time, so a caller never needs to change.
_DEFAULT_BRIGHTNESS_THRESHOLD_BLACK_CREAM = BRIGHTNESS_THRESHOLD_BLACK_CREAM
_DEFAULT_CHROMA_THRESHOLD = CHROMA_THRESHOLD


def set_board_color_thresholds(
    brightness_threshold: float | None = None,
    chroma_threshold: float | None = None,
) -> None:
    """(Re-)set the two live-derivable board-color thresholds in place.
    Pass `None` (either or both, the default) to reset that threshold
    back to its hardcoded default. Real caller:
    `opendarts.capture.throw_package.load_throw_package()`, from a
    session's own `board_color_calibration.json` sibling file (per-
    threshold: only a `confidence == "high"` derived value is ever
    adopted -- see that module's own `ThresholdDerivation.confidence`
    field -- anything lower falls back to the hardcoded default for
    THAT threshold specifically, the same fine-grained fallback posture
    as `dev.calibration.intrinsics_derivation`'s per-camera drift
    decisions)."""
    global BRIGHTNESS_THRESHOLD_BLACK_CREAM, CHROMA_THRESHOLD
    global _APPLIED_BRIGHTNESS_THRESHOLD, _APPLIED_CHROMA_THRESHOLD
    BRIGHTNESS_THRESHOLD_BLACK_CREAM = (
        _DEFAULT_BRIGHTNESS_THRESHOLD_BLACK_CREAM
        if brightness_threshold is None
        else brightness_threshold
    )
    CHROMA_THRESHOLD = (
        _DEFAULT_CHROMA_THRESHOLD if chroma_threshold is None else chroma_threshold
    )
    # Tracked in the caller's own original units, same "store the real
    # applied value rather than reverse-deriving it" reasoning as
    # opendarts.geometry.board.set_ring_boundary_offsets()'s sibling fields
    # (added the same day, for the same reason -- see that function's own
    # comment). `None` means "not accepted this event" for THIS threshold
    # specifically, independent of the other.
    _APPLIED_BRIGHTNESS_THRESHOLD = brightness_threshold
    _APPLIED_CHROMA_THRESHOLD = chroma_threshold


_APPLIED_BRIGHTNESS_THRESHOLD: float | None = None
_APPLIED_CHROMA_THRESHOLD: float | None = None


def get_board_color_thresholds() -> tuple[float | None, float | None]:
    """Returns `(brightness_threshold, chroma_threshold)` exactly as last
    passed to `set_board_color_thresholds()` -- `None` for a threshold
    currently at its hardcoded default. Read-back counterpart added
    2026-08-26 for `opendarts.live.capture_daemon.CalibrationStore`'s
    persisted-calibration snapshot -- see
    `opendarts.geometry.board.get_ring_boundary_offsets()`'s own docstring
    for the full reasoning, mirrored here exactly."""
    return _APPLIED_BRIGHTNESS_THRESHOLD, _APPLIED_CHROMA_THRESHOLD


def classify_bgr(b: float, g: float, r: float) -> BoardColor:
    """Classify one (mean) BGR sample into a real board color. Two-stage,
    matching the real measured structure (module docstring): achromatic
    (black/cream) vs. chromatic (red/green) first, by chroma; then, only
    within each branch, brightness (black vs cream) or R-vs-G (red vs
    green)."""
    brightness = (b + g + r) / 3.0
    chroma = max(b, g, r) - min(b, g, r)
    if chroma < CHROMA_THRESHOLD:
        return "black" if brightness < BRIGHTNESS_THRESHOLD_BLACK_CREAM else "cream"
    return "red" if r > g else "green"


def sample_patch_bgr(
    image: np.ndarray, ix: int, iy: int, patch_radius: int = PATCH_RADIUS_PX
) -> tuple[float, float, float] | None:
    """Mean (B, G, R) of a ``(2*patch_radius+1)``-square patch centered
    at pixel ``(ix, iy)`` in ``image``. Returns ``None`` if the center
    pixel itself is outside the image (a patch that partially overlaps
    the edge is still sampled, clipped to what's actually in-frame --
    only a center pixel fully off-frame is refused, matching how
    ``project_board_roi_polygon()`` callers already treat "projects
    somewhere real" as the actual gate, not "the whole patch must fit")."""
    h, w = image.shape[:2]
    if not (0 <= ix < w and 0 <= iy < h):
        return None
    y0, y1 = max(0, iy - patch_radius), min(h, iy + patch_radius + 1)
    x0, x1 = max(0, ix - patch_radius), min(w, ix + patch_radius + 1)
    patch = image[y0:y1, x0:x1]
    b, g, r = patch.reshape(-1, 3).mean(axis=0)
    return float(b), float(g), float(r)


def project_board_point_px(
    xy_mm: tuple[float, float], calib
) -> tuple[float, float]:
    """Forward-project a board-plane (Z=0) point into a camera's pixel
    space -- the same ``cv2.projectPoints`` primitive
    ``opendarts.engines.athena.board_gate.project_board_roi_polygon()``
    already uses, reused here rather than reinvented (see that module's
    own docstring for why this is the right CV primitive: the forward-
    projection inverse of ``back_project_ray``'s ``cv2.undistortPoints``).
    ``calib`` is a ``opendarts.pipeline.CameraCalibration``.
    """
    import cv2

    pt3d = np.array([[xy_mm[0], xy_mm[1], 0.0]], dtype=np.float64)
    projected, _ = cv2.projectPoints(
        pt3d, calib.rvec, calib.tvec, calib.camera_matrix, calib.dist_coeffs
    )
    px, py = projected.reshape(-1, 2)[0]
    return float(px), float(py)


def sample_board_color(
    xy_mm: tuple[float, float],
    calib,
    bg_image: np.ndarray,
    patch_radius: int = PATCH_RADIUS_PX,
) -> BoardColor | None:
    """Single-camera sample: forward-project ``xy_mm`` into ``calib``'s
    pixel space, sample a patch of ``bg_image`` (the camera's
    background frame -- NOT the dart frame, so the dart itself never
    occludes the sampled point, per this task's own instruction) at that
    pixel, and classify it. Returns ``None`` only if the projected point
    falls outside the image entirely (see ``sample_patch_bgr()``).

    Prefer ``sample_board_color_multi_camera()`` over this for anything
    that isn't itself measuring single- vs. multi-camera behavior --
    module docstring's own measurement found majority-vote-of-3 is
    materially more robust (100% vs 95.9% self-consistency accuracy)."""
    px, py = project_board_point_px(xy_mm, calib)
    ix, iy = int(round(px)), int(round(py))
    bgr = sample_patch_bgr(bg_image, ix, iy, patch_radius)
    if bgr is None:
        return None
    return classify_bgr(*bgr)


@dataclass
class MultiCameraColorSample:
    """Per-camera + aggregated result of sampling one board-plane point
    across every available camera. ``per_camera`` keys are camera ids
    that had BOTH a calibration and a background image passed in;
    a camera whose projection falls off-frame still gets an entry with
    value ``None`` (distinguishable from "camera not available at all",
    which is simply absent from the dict)."""

    per_camera: dict[int, BoardColor | None]
    majority: BoardColor | None
    agreement: float | None # fraction of non-None votes that matched the majority, or None if no votes


def sample_board_color_multi_camera(
    xy_mm: tuple[float, float],
    calibrations: dict[int, object],
    bg_images: dict[int, np.ndarray],
    patch_radius: int = PATCH_RADIUS_PX,
) -> MultiCameraColorSample:
    """Sample every camera that has both a calibration and a background
    image, classify each independently, and majority-vote the result.
    This is the recommended entry point (see module docstring for the
    measured 95.9% -> 100% improvement majority-vote gave on the same
    self-consistency check).

    Tie-break: on an even split (e.g. 1-1 with only 2 cameras available,
    or a genuine 3-way split), returns the FIRST color reaching the max
    vote count in camera-id order -- deterministic, not randomized, but
    honestly not "more correct" than any other tied color; callers that
    care about tie ambiguity can inspect ``per_camera``/``agreement``
    directly rather than trust ``majority`` blindly in that case.
    """
    per_camera: dict[int, BoardColor | None] = {}
    for cam, calib in calibrations.items():
        bg = bg_images.get(cam)
        if bg is None:
            continue
        per_camera[cam] = sample_board_color(xy_mm, calib, bg, patch_radius)

    votes = [v for v in per_camera.values() if v is not None]
    if not votes:
        return MultiCameraColorSample(per_camera=per_camera, majority=None, agreement=None)

    counts: dict[str, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best_count = max(counts.values())
    # Deterministic tie-break: first color (by camera-iteration order
    # above, i.e. dict insertion order) that reaches best_count.
    majority = next(v for v in votes if counts[v] == best_count)
    agreement = counts[majority] / len(votes)
    return MultiCameraColorSample(per_camera=per_camera, majority=majority, agreement=agreement)


def color_agrees(expected: BoardColor | None, sampled: BoardColor | None) -> bool | None:
    """Whether a sampled color matches the expected color for a scored
    (sector, ring). Returns ``None`` (not applicable, not "disagrees")
    when either side is ``None`` -- e.g. ``ring == "outside"`` (no
    expected color) or every camera's projection fell off-frame (no
    sampled color)."""
    if expected is None or sampled is None:
        return None
    return expected == sampled


@dataclass
class ColorSanityResult:
    """The full result of running the color sanity check on one scored
    (sector, ring, board_xy_mm) -- what ``evaluate_color_sanity()``
    returns. Deliberately does not itself decide anything about whether
    the underlying score should change; see that function's own
    docstring for why (measured evidence supports FLAGGING, not
    auto-correcting)."""

    expected: BoardColor | None
    sampled: BoardColor | None
    agrees: bool | None
    per_camera: dict[int, BoardColor | None]
    camera_agreement: float | None

    def to_dict(self) -> dict:
        return {
            "expected_color": self.expected,
            "sampled_color": self.sampled,
            "color_agrees": self.agrees,
            "per_camera_color": self.per_camera,
            "camera_agreement": self.camera_agreement,
        }


def evaluate_color_sanity(
    sector: str | None,
    ring: str | None,
    board_xy_mm: tuple[float, float] | None,
    calibrations: dict[int, object],
    bg_images: dict[int, np.ndarray],
    patch_radius: int = PATCH_RADIUS_PX,
) -> ColorSanityResult:
    """The one entry point this module expects callers (dashboard code,
    offline analysis, or a future live-dispatch hook) to actually use:
    given a scored ``(sector, ring, board_xy_mm)`` and the throw's
    per-camera calibrations + BACKGROUND images (never the dart frame --
    see ``sample_board_color()``), returns everything needed to decide
    what to do about a disagreement, without deciding it here.

    **Deliberately NOT wired into any live scoring path
    (opendarts.engines.dispatch/opendarts.live.capture_daemon) as of this
    writing.** Per the "measure before building any auto-correction"
    rule, a full corpus measurement
    (360 throws x 3 engines) found color disagreement DOES correlate
    with a real elevated miss rate for Apollo (14.8% vs 1.7% baseline)
    and Talos (9.7% vs 0.7%) -- a genuine, non-random signal -- but with
    very low PRECISION (only 4/27, 3/31 disagreements were actually
    real misses; the other 85-90% were throws the engine already scored
    correctly, where the sampled point simply landed near a wire/boundary
    ambiguity). Athena showed no real correlation at all (5.0% vs
    4.1%). This supports shipping the check as an available diagnostic
    a caller can attach/inspect, NOT wiring it to automatically flag or
    correct every live throw's score -- that judgment call (worth the
    false-positive rate for a given use case) belongs to whoever's
    calling this, not baked in here.

    Returns ``ColorSanityResult(agrees=None)`` (not False) whenever
    ``sector``/``ring``/``board_xy_mm`` don't support a check at all
    (e.g. ``ring == "outside"``, ``board_xy_mm is None``) or every
    camera's projection fell outside its frame -- "not applicable" is a
    different, honest answer from "disagrees".
    """
    if ring is None or board_xy_mm is None:
        return ColorSanityResult(
            expected=None, sampled=None, agrees=None, per_camera={}, camera_agreement=None
        )
    expected = expected_color(sector, ring)
    sample = sample_board_color_multi_camera(board_xy_mm, calibrations, bg_images, patch_radius)
    return ColorSanityResult(
        expected=expected,
        sampled=sample.majority,
        agrees=color_agrees(expected, sample.majority),
        per_camera=sample.per_camera,
        camera_agreement=sample.agreement,
    )
