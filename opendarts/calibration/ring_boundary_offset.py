"""Per-board ring-boundary offset measurement -- measuring where this
specific physical board's treble/double ring boundaries ACTUALLY sit,
directly from calibration (background) images.

WHY THIS EXISTS: `opendarts.geometry.board`'s
`INNER_RING_SCORING_OFFSET_MM = 1.5` is a real, measured property of the
rig's current board (its treble-inner / double-inner scoring transitions
sit ~1.5mm INSIDE the regulation 99.0/162.0 radii -- see that constant's
own comment), but it is a flat hardcoded default. A different physical
board (even the same model) could have a genuinely different real
offset, or none, so the offset is derived at calibration time and saved
in the calibration package alongside the images it was measured from.

THE KEY OBSERVATION that makes this measurable from images alone:
today's calibration pipeline precisely measures exactly ONE ring
boundary from the image (the double ring's OUTER edge, via
`landmark_detection.detect_double_ring_quad()`'s ellipse fit) and then
COMPUTES every other ring radius from the solved pose using the
idealized regulation ratios -- never re-checking them against the image.
But the treble and double rings are visually distinct colored bands
(red/green vs the black/cream single beds -- the same physical fact
`opendarts.geometry.board_color` already measured for this rig), so the
real radial position of each band edge is directly observable.

HOW IT WORKS (no ellipse inversion anywhere -- the solved pose makes
projection exact):

1. For each camera, take the already-solved board pose (a
   `CameraCalibration`, e.g. from a session's `calibration_refit.json`
   -- the same "today's pose" source the corpus-replay guardrail in
   docs/DESIGN.md mandates) and FORWARD-project board-plane polar sample
   points (fixed board angle, radius swept in 0.1mm steps across a
   boundary's neighborhood) into pixel space with `cv2.projectPoints`
   -- the exact primitive `board_color.project_board_point_px()` and
   `athena.board_gate` already use.
2. Bilinearly sample the background image (no dart in frame) along each
   radial, at 20 sector-center angles (+/- a small in-sector offset;
   radial wires at sector edges are never sampled) -- the same general
   walk-outward-along-many-angles approach
   `ring_correlation_orientation.py` established for the number ring,
   applied to the ring bands instead.
3. Reduce each sample to a SIGNED opponent-color score,
   `sign * (R - G)`, where `sign` is +1 for red-accent sectors and -1
   for green-accent sectors (`board_color.sector_accent_color()`'s
   already-measured per-sector alternation). This is deliberately not
   raw chroma: the signed opponent signal puts the single bed (black:
   R~G; cream: mildly warm) and the accent band (R-G ~ +105 for red,
   G-R ~ +87 for green, per board_color's real 558-sample measurement)
   at strongly separated levels for BOTH sector parities, and the
   achromatic wire/glare sits between them rather than above them.
4. Normalize each profile to [0, 1] between its own measured
   inside/outside reference levels (a per-profile affine transform of
   the VALUE axis only -- it cannot move any radial feature), then, per
   (camera, boundary), take the PER-RADIUS MEDIAN across all accepted
   (angle, frame) profiles. The profiles share an exact common mm grid
   by construction (step 1 projects the same board radii for every
   angle), so this is a true aligned median: per-profile pixel noise
   and per-sector glare variation collapse, leaving the boundary's
   stable mean transition shape.
5. Localize the wire ONCE on that low-noise median profile (see
   "LOCALIZATION -- why a gradient peak and why bed-adjacent" below),
   giving one radius per (camera, boundary); across cameras, the median
   of per-camera radii (robust to one camera's foreshortening/lighting
   bias). `offset_mm = regulation_radius - measured_radius` -- positive
   means the real boundary sits INSIDE regulation, i.e. directly
   comparable to `INNER_RING_SCORING_OFFSET_MM`'s sign convention.

LOCALIZATION -- why a gradient peak and why bed-adjacent (2026-08-20,
the fix for the treble_outer control anomaly):

The original per-profile 50%-threshold crossing carried a real,
measured bias TOWARD the chromatic band at every boundary (worst at
treble_outer: +0.64mm vs an oracle-label-pinned truth of ~0, with a
+0.52mm black/cream bed-color swing). Mechanism, confirmed on real
pixels: specular glare centered on the ring wire washes out the band's
chroma progressively on the BAND side of the wire (channel clipping --
the red band's R channel is already saturated on this rig, so additive
glare can only close the R-G gap), creating a gradual "washout ramp"
in the opponent signal that starts millimetres before the wire. A 50%
threshold crossing lands inside that ramp, and WHERE it lands depends
on the far side's absolute level -- the diagnosed bed-color swing.

The median transition profile makes the real structure visible: its
derivative is bimodal -- a broad washout-ramp lobe on the band side
and a sharp lobe at the true paint->wire edge adjacent to the bed.
So the localizer is:

  a. Find the median profile's sustained 0.5-crossing inside the
     search zone and expand to the active transition interval (profile
     within (0.08, 0.92) -- bounds only bound the SEARCH REGION; the
     localization below is a local feature, insensitive to them).
  b. Take the smoothed derivative's peaks in that interval; keep the
     contiguous high-slope region around the dominant peak (extend
     toward the bed while the derivative stays >= 20% of max).
  c. Choose the significant (>= 50% of max) peak ADJACENT TO THE
     SINGLE-BED side -- first peak for rising boundaries (bed inside),
     last for falling (bed outside) -- with parabolic sub-sample
     refinement. Physical basis: the washout ramp lives on the
     chromatic-band side (only chroma can be washed out); the bed side
     of the transition ends at the real wire edge.

  EXCEPTION -- double_outer localizes at the median profile's own
  sustained 0.5-crossing instead (`wire_side_is_bed_paint=False`).
  Its far side is not a paint bed but the board surround, a universal
  board-construction fact, not a rig fact: the surround side is dark
  (glare-immune -- the same reason the anomaly never appeared there)
  but carries its own non-paint structure and a brightness-collapse
  chroma tail that generates spurious "bed-adjacent" derivative peaks.
  Measured on the full 13-session corpus: the crossing reads +0.01mm
  there (clean), while a bed-adjacent peak reads -1.2mm (tail junk).

  The selection fractions (0.5 significance, 0.2 contiguity stop) were
  swept, not guessed: results for the three bed-paint boundaries are
  stable across significance 0.5-0.65 and stop 0.2-0.4.

  Why a gradient peak is structurally immune to the two diagnosed
  failure modes: a threshold crossing's physical location depends on
  the far-side reference level (the anomaly's mechanism); a derivative
  PEAK is a purely local feature -- it sits where the signal changes
  fastest and does not move when a plateau's absolute level changes.
  And a gradual glare blend toward near-white has LOW derivative even
  at high absolute brightness, so it cannot fake the wire edge the way
  it faked the 50% level.

All four ring boundaries are measured independently (treble_inner,
treble_outer, double_inner, double_outer) even though only the two inner
ones historically needed a scoring correction -- the outer two are the
method's own built-in control: `board.py` keeps the outer boundaries at
their regulation values (they needed no correction on this board), so a
measurement run that reports a large outer offset is evidence of a
method problem, not a board problem. Likewise treble and double inner
offsets are measured and reported separately, never collapsed into one
number, even though the historical constant treated them as one.

INPUTS. The algorithm consumes: background frames + a solved
`CameraCalibration` + this project's own board geometry and
color-pattern constants. Nothing here reads `ad_ground_truth.json` or
any oracle label. (Validating the RESULT against oracle-labelled real
throws afterwards is a separate, explicitly out-of-band step -- same
discipline as every other calibration-bootstrap piece.)

STORAGE / REPLAY. The offline measurement writes a session-level
`ring_boundary_offset.json` NEXT TO the session's packages -- the exact
non-destructive additive pattern of the session pose refit's own
`calibration_refit.json` (see `opendarts.capture.recalibrate`; this
deliberately mirrors it rather than inventing another convention). Per
the "Replay is the source of truth" constraint the file
stores not just the final offsets but the full raw derivation: which
stored bg images were used (by package-relative path -- the raw pixels
themselves already live in the packages, byte-exact), the calibration
source, every accepted per-(camera, angle, frame) crossing radius, and
the measurement parameters -- enough to recompute or audit the offsets
from scratch later if this detection algorithm improves.

DELIBERATELY NOT WIRED INTO LIVE SCORING. `sector_ring_for_point()`
still uses the hardcoded `INNER_RING_SCORING_OFFSET_MM`; this module is
additive, measurement-only. Wiring a measured per-board offset into the
live scoring path is a separate future step with its own verification
pass (same staged detect-then-validate-then-wire approach as the
orientation-hint work).

Status: v2 (2026-08-20), full-corpus validated. v1 (per-profile
50%-threshold crossings, MAD-trimmed) carried the treble_outer control
anomaly (+0.57mm on a boundary real data independently pins at ~0); v2
replaces the localization with the median-profile bed-adjacent gradient
peak described above. The validated result, measured across all 13
recorded sessions and compared against an independent AD-label
transition census (per-session offsets, mean +- std across sessions;
census bracket in brackets):

    treble_inner  v1 +1.077 -> v2 +1.382 +- 0.352  [+1.18, +2.04]  IN
    treble_outer  v1 +0.569 -> v2 -0.013 +- 0.250  [-0.34, +0.14]  IN
    double_inner  v1 +1.614 -> v2 +2.084 +- 0.343  [+1.74, +2.96]  IN
    double_outer  v1 +0.036 -> v2 +0.056 +- 0.042  [-0.65, +1.54]  IN

i.e. v2 brings all four boundaries inside their independent truth
brackets, fixes the treble_outer control (+0.569 -> ~0) without
breaking the double_outer control, and collapses the +0.52mm
black/cream bed-color swing to -0.08mm.

The v1 per-profile primitives (`_measure_one_profile`,
`_find_sustained_crossing`) are kept intact -- the committed
diagnosis/evidence scripts under dev/calibration/ call them directly
and must stay reproducible.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from opendarts.geometry.board import (
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Measurement parameters
# ---------------------------------------------------------------------

# Radial sampling step along each radial profile (mm in board space).
# 0.1mm is ~5-10x finer than one image pixel at this rig's real scale
# (~0.5-0.9 mm/px at the ring radii), so the localization precision is
# set by the image + interpolation, not by the sampling grid.
RADIAL_STEP_MM = 0.1

# Box-smooth width applied to each radial profile before edge
# localization, in samples (odd). 5 samples = 0.5mm -- narrower than the
# band-edge blur itself at this pixel scale, so it suppresses
# pixel-level noise without materially moving the crossing.
PROFILE_SMOOTH_SAMPLES = 5

# The crossing must stay on the far side of the threshold for this many
# consecutive samples (1.5mm) to count -- rejects glints/specks. Must be
# comfortably smaller than the narrowest zone between a search-window
# edge and a reference zone (see _BOUNDARIES below).
SUSTAIN_SAMPLES = 15

# Minimum inside-vs-outside opponent-signal contrast for an angle to be
# usable at all. The real measured levels (board_color docstring: red
# band R-G ~ +105 / green band G-R ~ +87 vs single beds near 0) sit far
# above this; an angle failing it is occluded/shadowed/foreshortened,
# not a real weak edge.
MIN_CONTRAST = 40.0

# Minimum local radial image scale (px per board-mm, measured from the
# projection itself at the regulation radius) for an angle to be used.
# Below this the edge is spread over too few pixels to localize to
# sub-mm; the angle is skipped, not down-weighted.
MIN_RADIAL_SCALE_PX_PER_MM = 0.35

# In-sector angular offsets (degrees) sampled around each sector's
# center angle. Radial wires sit at +/-9 deg from center; +/-4 deg keeps
# every sample >= 5 deg clear of any wire while tripling the sample
# count per sector.
IN_SECTOR_ANGLE_OFFSETS_DEG = (-4.0, 0.0, 4.0)


# --- v2 median-profile localization parameters (see module docstring,
# "LOCALIZATION" section, for the measured sweep behind these) ---

# Active-transition bounds on the [0,1]-normalized median profile: the
# interval around the sustained 0.5-crossing where the profile is still
# in transit. These bound the peak SEARCH REGION only; the chosen peak
# is a local feature and does not move with them.
TRANSITION_LO = 0.08
TRANSITION_HI = 0.92

# The sustained 0.5-crossing must stay >= 0.5 for this many consecutive
# samples (0.5mm) -- cheap insurance against a residual noise wiggle in
# the median profile.
MEDIAN_CROSSING_SUSTAIN_SAMPLES = 5

# A derivative peak counts as significant if >= this fraction of the
# dominant peak's height. Swept 0.5-0.8 on the full corpus; 0.5-0.65
# give equivalent, census-consistent results on all three bed-paint
# boundaries (0.8 starts re-admitting the washout ramp on
# treble_inner).
PEAK_SIGNIFICANCE_FRACTION = 0.5

# Contiguity: from the dominant peak, extend toward the bed side only
# while the derivative stays >= this fraction of the dominant peak --
# candidate peaks beyond a deeper dip are disconnected structure (e.g.
# tail lobes), not part of the wire transition. Swept 0.2-0.4; results
# stable.
PEAK_CONTIGUITY_STOP_FRACTION = 0.2

# Minimum accepted profiles per (camera, boundary) before the median
# profile is considered meaningful at all.
MIN_PROFILES_PER_CAMERA = 10

# How many packages' bg frames to pool per session in the session-level
# driver. The rig does not move within a session (same rationale as
# recalibrate.DEFAULT_N_SAMPLE_PACKAGES); 8 packages x 3 cameras x 60
# angles already gives hundreds of independent crossings per boundary.
DEFAULT_N_SAMPLE_PACKAGES = 8

OFFSET_FILENAME = "ring_boundary_offset.json"
# v2: median-profile bed-adjacent gradient-peak localization (the v1
# per-profile 50%-crossing schema stored per-crossing "samples"; v2
# stores per-(camera, boundary) "median_profiles" instead).
SCHEMA = "ring-boundary-offset-v2"


@dataclass(frozen=True)
class _BoundarySpec:
    """One ring boundary's sampling geometry (all radii in board mm).

    `rising` is whether the signed opponent signal rises when walking
    OUTWARD across this boundary (True at single->band edges, False at
    band->single/outside edges). The reference zones must sit fully
    inside their own band/bed, clear of both the boundary blur and any
    neighboring boundary.
    """

    name: str
    regulation_radius_mm: float
    r_lo: float
    r_hi: float
    ref_inside: tuple[float, float] # radial zone INSIDE the boundary (smaller r)
    ref_outside: tuple[float, float] # radial zone OUTSIDE the boundary (larger r)
    search: tuple[float, float] # zone the crossing must be found in
    rising: bool
    # True when the non-band side of this boundary is single-bed PAINT
    # (a universal board-construction fact, not a rig fact): the v2
    # localizer then takes the bed-adjacent derivative peak. False only
    # for double_outer, whose far side is the board surround -- there
    # the median profile's own 0.5-crossing is used instead (see module
    # docstring, LOCALIZATION section).
    wire_side_is_bed_paint: bool = True


# The treble band is 8mm wide (99-107) and the double band 8mm (162-170)
# -- reference zones inside the bands stay >= 2mm clear of both band
# edges. The zone outside double_outer is the board surround (no paint
# model applies there; its level is whatever it is, and the contrast
# gate decides per angle whether the edge is usable).
_BOUNDARIES: tuple[_BoundarySpec, ...] = (
    _BoundarySpec(
        "treble_inner", TREBLE_INNER_RADIUS_MM,
        r_lo=90.0, r_hi=105.0,
        ref_inside=(90.0, 94.0), ref_outside=(101.5, 105.0),
        search=(94.5, 101.5), rising=True,
    ),
    _BoundarySpec(
        "treble_outer", TREBLE_OUTER_RADIUS_MM,
        r_lo=101.0, r_hi=116.0,
        ref_inside=(101.0, 104.5), ref_outside=(112.0, 116.0),
        search=(104.5, 111.5), rising=False,
    ),
    _BoundarySpec(
        "double_inner", DOUBLE_INNER_RADIUS_MM,
        r_lo=153.0, r_hi=168.0,
        ref_inside=(153.0, 157.0), ref_outside=(164.5, 168.0),
        search=(157.5, 164.5), rising=True,
    ),
    _BoundarySpec(
        "double_outer", DOUBLE_OUTER_RADIUS_MM,
        r_lo=164.0, r_hi=179.0,
        ref_inside=(164.0, 167.5), ref_outside=(175.0, 179.0),
        search=(167.5, 174.5), rising=False,
        wire_side_is_bed_paint=False,
    ),
)

BOUNDARY_NAMES: tuple[str, ...] = tuple(b.name for b in _BOUNDARIES)


# ---------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------


@dataclass
class MedianProfileRecord:
    """One (camera, boundary)'s aggregated median transition profile and
    its localization -- the raw derivation record the storage format
    persists in full (REPLAY principle: the aggregate offsets can be
    re-derived/audited from these without re-running the detector, and
    re-running it is itself fully reproducible from source_images + the
    packages' stored pixels)."""

    boundary: str
    camera: int
    n_profiles: int # accepted (angle x frame) profiles pooled
    r_lo_mm: float # grid start; step is radial_step_mm
    radial_step_mm: float
    median_profile: list[float] # [0,1]-normalized, rising outward
    localized_radius_mm: float | None
    mode: str # "bed_adjacent_peak" | "median_crossing"
    diagnostics: dict # transition interval, peak table, ...


@dataclass
class BoundaryMeasurement:
    """One ring boundary's aggregated measurement across all cameras,
    frames, and angles."""

    boundary: str
    regulation_radius_mm: float
    measured_radius_mm: float | None # None if too few samples survived
    offset_mm: float | None # regulation - measured; positive = inside regulation
    mad_mm: float | None # MAD of per-camera localized radii about the combined value
    n_samples_used: int # accepted profiles pooled into localized cameras
    n_samples_rejected: int # attempted profiles that failed a gate
    n_angles_attempted: int
    per_camera: dict[int, dict] # cam -> {radius_mm, n_profiles, mode}
    # Honest, documented heuristic in [0, 1] -- NOT a calibrated
    # probability: fraction of attempted (angle x frame) profiles that
    # survived the gates into a localized camera, damped by the
    # across-camera dispersion (zero by 1.5mm MAD). Callers wanting
    # detail should read the components, not just this scalar.
    confidence: float


@dataclass
class RingBoundaryOffsetResult:
    """Full result of one measurement run. `boundaries` is keyed by
    BOUNDARY_NAMES; `median_profiles` holds every (camera, boundary)
    median transition profile + localization (the replay-principle
    derivation records)."""

    boundaries: dict[str, BoundaryMeasurement]
    median_profiles: list[MedianProfileRecord] = field(default_factory=list)
    source_images: list[str] = field(default_factory=list)
    calibration_source: str = ""

    @property
    def treble_inner_offset_mm(self) -> float | None:
        return self.boundaries["treble_inner"].offset_mm

    @property
    def double_inner_offset_mm(self) -> float | None:
        return self.boundaries["double_inner"].offset_mm


# ---------------------------------------------------------------------
# Storage format
# ---------------------------------------------------------------------


def boundary_measurement_to_payload(m: BoundaryMeasurement) -> dict:
    """One `BoundaryMeasurement`'s full JSON-safe payload -- every field
    except the raw per-(camera, angle) profiles (those live separately,
    in `MedianProfileRecord`/`median_profiles`, since they're the
    comparatively large raw-derivation records, not the aggregate). This
    is the SAME shape `result_to_payload()` below has always written
    into the offline session-level `ring_boundary_offset.json` for every
    boundary -- factored out here (2026-08-26) so
    `opendarts.live.capture_daemon.bootstrap_calibrations()` /
    `opendarts.capture.calibration_package.save_calibration_package()` can
    reuse the EXACT same field names/shape for the live per-calibration-
    event package instead of the two representations inventing two
    different vocabularies for the same measurement (see that module's
    own "WHY THIS EXISTS" note added the same day: a live 0.76mm
    divergence between two independent localizers was undiagnosable from
    opendarts's own live package because it only ever kept 2 of the 4
    boundaries' offset/confidence/n_samples_used, discarding
    treble_outer/double_outer entirely and every boundary's mad_mm/
    per_camera breakdown).

    `per_camera` keys are stringified (`{str(c): d for c, d in
    m.per_camera.items()}`) -- JSON object keys are always strings,
    matching `result_to_payload()`'s existing convention.
    """
    return {
        "regulation_radius_mm": m.regulation_radius_mm,
        "measured_radius_mm": m.measured_radius_mm,
        "offset_mm": m.offset_mm,
        "mad_mm": m.mad_mm,
        "n_samples_used": m.n_samples_used,
        "n_samples_rejected": m.n_samples_rejected,
        "n_angles_attempted": m.n_angles_attempted,
        "per_camera": {str(c): d for c, d in m.per_camera.items()},
        "confidence": m.confidence,
    }


def result_to_payload(
    result: RingBoundaryOffsetResult, *, solved_by: str | None = None
) -> dict:
    """Serialize a `RingBoundaryOffsetResult` to the one JSON shape this
    measurement has ever written -- by the offline session-level writer
    (`dev.calibration.ring_boundary_measure.write_session_ring_boundary_
    offset()`), and, while the live derivation existed (2026-08-27 to
    2026-09-13, the v2 package schema), by
    `bootstrap_calibrations()` persisting the same shape into a
    calibration event's own `derived_calibration.json` (see
    `opendarts.capture.calibration_package`). One schema, one payload
    builder, regardless of which caller produced `result`. It stays in
    the package with the reader because real files on disk are written
    in this shape and `load_session_ring_boundary_offset()` below
    dispatches on its `schema`.

    `solved_by` lets each caller attribute the payload honestly; defaults
    to the offline measurement's own description."""
    return {
        "schema": SCHEMA,
        "solved_by": solved_by or (
            "dev.calibration.ring_boundary_measure "
            "(offline, from stored bg frames + session calibration refit)"
        ),
        "calibration_source": result.calibration_source,
        "source_images": result.source_images,
        "parameters": {
            "radial_step_mm": RADIAL_STEP_MM,
            "profile_smooth_samples": PROFILE_SMOOTH_SAMPLES,
            "min_contrast": MIN_CONTRAST,
            "min_radial_scale_px_per_mm": MIN_RADIAL_SCALE_PX_PER_MM,
            "in_sector_angle_offsets_deg": list(IN_SECTOR_ANGLE_OFFSETS_DEG),
            "transition_lo": TRANSITION_LO,
            "transition_hi": TRANSITION_HI,
            "median_crossing_sustain_samples": MEDIAN_CROSSING_SUSTAIN_SAMPLES,
            "peak_significance_fraction": PEAK_SIGNIFICANCE_FRACTION,
            "peak_contiguity_stop_fraction": PEAK_CONTIGUITY_STOP_FRACTION,
            "min_profiles_per_camera": MIN_PROFILES_PER_CAMERA,
        },
        "boundaries": {
            name: boundary_measurement_to_payload(m)
            for name, m in result.boundaries.items()
        },
        # Full raw derivation (REPLAY principle): every (camera,
        # boundary) median transition profile + its localization, so the
        # aggregate can be re-derived/audited without re-running the
        # detector -- and re-running it is itself fully reproducible
        # from source_images + the packages' stored pixels.
        "median_profiles": [
            {
                "boundary": r.boundary,
                "camera": r.camera,
                "n_profiles": r.n_profiles,
                "r_lo_mm": r.r_lo_mm,
                "radial_step_mm": r.radial_step_mm,
                "median_profile": r.median_profile,
                "localized_radius_mm": (
                    round(r.localized_radius_mm, 4)
                    if r.localized_radius_mm is not None else None
                ),
                "mode": r.mode,
                "diagnostics": r.diagnostics,
            }
            for r in result.median_profiles
        ],
    }

def load_session_ring_boundary_offset(session_dir: Path) -> dict | None:
    """Load a previously written session offset file as its raw payload
    dict, or None if the session has none OR its schema doesn't match
    this module's current `SCHEMA`. (Kept as a plain dict, not
    re-hydrated dataclasses: consumers so far only read the aggregate
    numbers, and the payload is its own documented schema.)

    **Schema check, added 2026-08-21 (live-wiring task, item 4's own
    documented gap)**: real v1 files (the pre-gradient-fix, per-profile
    50%-crossing method -- see `SCHEMA`'s own comment above for the real
    bias that fix closed, worst +0.64mm at treble_outer) already sit on
    disk in `data/archive/clean/*/ring_boundary_offset.json` from before
    this schema existed. Silently trusting them once this module goes
    live would mean live scoring quietly inherits that stale, measurably
    biased offset. A schema mismatch is treated exactly like "no file at
    all" -- the caller falls back to `INNER_RING_SCORING_OFFSET_MM`'s
    hardcoded default, the same safe posture as a session with no
    measurement yet, not an error. Regenerating those stale files
    (running `write_session_ring_boundary_offset()` fresh, real corpus
    I/O against `data/archive/clean/`) is deliberately NOT done by this
    check itself -- see this task's own final report for why a schema
    guard was chosen over a live regeneration from within an isolated
    worktree agent.

    **Corrupt-file guard, added 2026-08-21 (OpenDarts pre-hardware-audit
    finding, confirmed to apply here too)**: `json.loads()` used to be
    unguarded -- a truncated/corrupt file on disk (e.g. a crash mid-write)
    would raise straight out of this function, through `opendarts.capture.
    throw_package.load_throw_package()`, BEFORE `set_ring_boundary_
    offsets()` ever runs for that call -- defeating the exact "a prior
    package's session-specific offset can never leak into a later load
    from a different session" guarantee that caller's own docstring
    promises, since the reset call that guarantee depends on would never
    happen. A corrupt file is now treated exactly like "no file at all"
    (safe fallback), loudly logged rather than silently swallowed, same
    posture as the schema-mismatch case just above."""
    path = Path(session_dir) / OFFSET_FILENAME
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
