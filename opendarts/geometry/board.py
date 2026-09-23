"""Standard dartboard geometry — single source of truth for board
dimensions, used by both calibration (landmark 3D coordinates) and
scoring (sector/ring lookup). Per WDF/PDC regulation dimensions —
public, standardized values, not something to guess or duplicate
elsewhere.

All linear dimensions in millimeters. Board-centered world frame:
origin at the bullseye center, board face is the Z=0 plane.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Regulation radii (mm), center to each ring boundary. These are the
# PHYSICAL board model -- they define where `wire_intersection_
# landmarks()` puts its 3D calibration landmarks and what
# `opendarts.calibration.oriented_landmarks` measures the double band
# against. Do not move them to make scoring come out right (see
# INNER_RING_SCORING_OFFSET_MM below for where a measured scoring
# correction belongs instead).
BULL_RADIUS_MM = 6.35
OUTER_BULL_RADIUS_MM = 15.9
TREBLE_INNER_RADIUS_MM = 99.0
TREBLE_OUTER_RADIUS_MM = 107.0
DOUBLE_INNER_RADIUS_MM = 162.0
DOUBLE_OUTER_RADIUS_MM = 170.0

# Default scoring offset for the two INNER ring boundaries (inner treble
# and inner double): how far inside the regulation radius the real
# single->treble and single->double scoring transition sits, in the
# board-mm frame every engine here works in. The outer boundaries keep
# their regulation values.
#
# Applied to SCORING ONLY, not to the regulation radii above: those are
# also the 3D landmark model PnP calibrates against
# (`wire_intersection_landmarks()`) and what `oriented_landmarks`
# measures the double band against, so moving them would silently change
# calibration. Per board, this default is overridden by the image-based
# measurement in `opendarts.calibration.ring_boundary_offset` (see
# `set_ring_boundary_offsets()` below).
INNER_RING_SCORING_OFFSET_MM = 1.5
TREBLE_INNER_SCORING_RADIUS_MM = TREBLE_INNER_RADIUS_MM - INNER_RING_SCORING_OFFSET_MM
DOUBLE_INNER_SCORING_RADIUS_MM = DOUBLE_INNER_RADIUS_MM - INNER_RING_SCORING_OFFSET_MM

# LIVE-DERIVED RING-BOUNDARY OFFSET, wired 2026-08-21 -- see
# opendarts.calibration.ring_boundary_offset's own module docstring for
# the derivation (per-board, measured directly from calibration images). The two SCORING radii above are a MUTABLE module global,
# updated in place by `set_ring_boundary_offsets()` below rather than
# threaded as a parameter through every engine's own score_dart() --
# every real caller (Apollo/Talos/Athena/Zeus's own confidence.py
# and scoring.py files) calls `sector_ring_for_point()`, which resolves
# `TREBLE_INNER_SCORING_RADIUS_MM`/`DOUBLE_INNER_SCORING_RADIUS_MM` from
# THIS module's own global namespace at CALL TIME (a bare name inside a
# function body, not a closure-captured value) -- so reassigning them
# here propagates to every caller without touching any engine's code.
#
# The one place this does NOT auto-propagate is a handful of engines'
# own `confidence.py` files, which import `TREBLE_INNER_SCORING_RADIUS_MM`/
# `DOUBLE_INNER_SCORING_RADIUS_MM` BY VALUE (`from opendarts.geometry.board
# import TREBLE_INNER_SCORING_RADIUS_MM`) for their own boundary-distance
# confidence estimate -- a frozen-at-import-time local name, immune to
# this module's own later reassignment. Fixed alongside this wiring (see
# opendarts/engines/apollo/confidence.py and
# opendarts/engines/talos/confidence.py, now reading the module attribute
# dynamically instead) so the confidence estimate stays consistent with
# whatever `sector_ring_for_point()` itself is actually using.
_DEFAULT_TREBLE_INNER_SCORING_RADIUS_MM = TREBLE_INNER_SCORING_RADIUS_MM
_DEFAULT_DOUBLE_INNER_SCORING_RADIUS_MM = DOUBLE_INNER_SCORING_RADIUS_MM


def set_ring_boundary_offsets(
    treble_inner_offset_mm: float | None = None,
    double_inner_offset_mm: float | None = None,
) -> None:
    """(Re-)derive the two SCORING radii from a pair of live-measured
    per-board offsets, in place. Pass `None` (either or both) to reset
    that boundary back to the hardcoded `INNER_RING_SCORING_OFFSET_MM`
    default -- the exact prior behavior, and what every test/tool that
    never calls this function at all still gets, since these two globals
    start at that default value.

    Callers: `opendarts.capture.throw_package.load_throw_package()` (the
    real live/replay entry point -- every engine loads a throw through
    it) calls this once per load, from that throw's own session-level
    `ring_boundary_offset.json` sibling file when present (mirroring the
    already-established `calibration_refit.json` precedent), else resets
    to the default. NOT thread-safe for concurrent throws from DIFFERENT
    sessions in the same process (a real, accepted limitation -- see
    that function's own docstring: this project's live product scores
    one physical rig/one session at a time, the same posture
    `MEASURED_FOCAL_LENGTH_PX`/`MEASURED_CAMERA_ORIENTATION_HINTS_DEG`
    already have as plain module-level "current rig" constants)."""
    global TREBLE_INNER_SCORING_RADIUS_MM, DOUBLE_INNER_SCORING_RADIUS_MM
    global _APPLIED_TREBLE_INNER_OFFSET_MM, _APPLIED_DOUBLE_INNER_OFFSET_MM
    TREBLE_INNER_SCORING_RADIUS_MM = (
        _DEFAULT_TREBLE_INNER_SCORING_RADIUS_MM
        if treble_inner_offset_mm is None
        else TREBLE_INNER_RADIUS_MM - treble_inner_offset_mm
    )
    DOUBLE_INNER_SCORING_RADIUS_MM = (
        _DEFAULT_DOUBLE_INNER_SCORING_RADIUS_MM
        if double_inner_offset_mm is None
        else DOUBLE_INNER_RADIUS_MM - double_inner_offset_mm
    )
    # Tracked alongside the derived SCORING radii above, in the caller's
    # own original offset_mm units (not re-derived from the radii by
    # reversing the subtraction) -- added 2026-08-26 for
    # opendarts.live.capture_daemon.CalibrationStore's persisted-calibration
    # snapshot, which needs to read back "what offset is actually live
    # right now" at `.set()` time without a second parallel channel
    # threading the original value through every bootstrap call site.
    # Reconstructing this via `TREBLE_INNER_RADIUS_MM -
    # TREBLE_INNER_SCORING_RADIUS_MM` plus an equality check against the
    # default would work almost always, but degrades silently (reports
    # "not accepted" for a real, if astronomically unlikely, live value
    # that happens to exactly equal the hardcoded default) -- storing the
    # real applied value explicitly avoids that edge case entirely.
    _APPLIED_TREBLE_INNER_OFFSET_MM = treble_inner_offset_mm
    _APPLIED_DOUBLE_INNER_OFFSET_MM = double_inner_offset_mm


_APPLIED_TREBLE_INNER_OFFSET_MM: float | None = None
_APPLIED_DOUBLE_INNER_OFFSET_MM: float | None = None


def get_ring_boundary_offsets() -> tuple[float | None, float | None]:
    """Returns `(treble_inner_offset_mm, double_inner_offset_mm)` exactly
    as last passed to `set_ring_boundary_offsets()` -- `None` for a
    boundary currently at its hardcoded default (never live-derived, or
    explicitly reset). Read-back counterpart to that function, added
    2026-08-26 for `opendarts.live.capture_daemon.CalibrationStore`'s
    persisted-calibration snapshot (see that class's own docstring) --
    lets the snapshot writer capture "what's actually live right now"
    without bootstrap_calibrations() having to thread the value through a
    second channel."""
    return _APPLIED_TREBLE_INNER_OFFSET_MM, _APPLIED_DOUBLE_INNER_OFFSET_MM


# Standard sector number ordering, clockwise from 12 o'clock (top).
SECTOR_NUMBERS_CLOCKWISE = [
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
]
SECTOR_ANGLE_DEG = 360.0 / len(SECTOR_NUMBERS_CLOCKWISE) # 18 degrees


def sector_center_angle_deg(number: int) -> float:
    """Angle (degrees, clockwise from 12 o'clock/+Y) of a sector's center."""
    idx = SECTOR_NUMBERS_CLOCKWISE.index(number)
    return idx * SECTOR_ANGLE_DEG


@dataclass(frozen=True)
class BoardPoint3D:
    """A known 3D landmark on the board face, in the board-centered world
    frame (Z=0, X/Y in mm)."""

    label: str
    x_mm: float
    y_mm: float

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x_mm, self.y_mm, 0.0)


def polar_to_xy_mm(radius_mm: float, angle_deg: float) -> tuple[float, float]:
    """Clockwise-from-+Y (12 o'clock) angle convention, matching dartboard
    numbering convention above -- not the usual math counter-clockwise-
    from-+X convention. Keep this the one place that conversion happens.

    Made public 2026-08-12 so
    `dev/calibration/real_correspondences.py` can build the same
    convention's 3D points for the 4-point cardinal-wire quad without
    duplicating this formula -- single source of truth for the
    angle/radius -> board-mm conversion, not two copies that could drift.
    """
    rad = math.radians(angle_deg)
    x = radius_mm * math.sin(rad)
    y = radius_mm * math.cos(rad)
    return (x, y)


def wire_boundary_angle_deg(number: int) -> float:
    """Angle (degrees, clockwise from 12 o'clock/+Y) of the real radial
    WIRE that sits clockwise-adjacent to sector `number`'s center -- i.e.
    the wire shared between `number` and its clockwise neighbor in
    `SECTOR_NUMBERS_CLOCKWISE`. Half a sector width (9deg) clockwise of
    `sector_center_angle_deg(number)`. See `wire_intersection_landmarks()`
    docstring (fixed 2026-08-12) for why this -- not the sector CENTER
    angle -- is the real "wire" position: a wire intersection is "every
    wire intersection
    with a ring boundary," and a radial wire physically runs BETWEEN two
    adjacent sectors, not through a sector's own center.
    """
    idx = SECTOR_NUMBERS_CLOCKWISE.index(number)
    return (idx * SECTOR_ANGLE_DEG + SECTOR_ANGLE_DEG / 2.0) % 360.0


def wire_intersection_landmarks() -> list[BoardPoint3D]:
    """All wire/ring-boundary intersections as known 3D landmarks: bull
    center + 20 sectors x 4 ring boundaries = 81 points, matching the
    count (but NOT the provenance -- these are independently derived from
    pure geometry, not homography-reprojected) of calibration.json's
    `dartboard` field structure.

    **Fixed 2026-08-12**:
    every returned point used to sit at `sector_center_angle_deg(number)`
    (0, 18, 36, ... deg) -- a SECTOR CENTER, not a wire. A real radial
    wire runs BETWEEN two adjacent sectors ("20
    radial sector wires at fixed angular positions... every wire
    intersection with a ring boundary is a known 3D point"), i.e. at
    `wire_boundary_angle_deg(number)` (9, 27, 45, ... deg -- exactly half
    a sector, 9deg, clockwise of each old center value). This was a real
    naming/implementation mismatch, not merely cosmetic: confirmed by
    cross-checking against this project's own independently-derived real
    wire positions in `dev/calibration/real_correspondences.py` -- those real wire points sit at 9/99/189/279deg,
    which are exactly `wire_boundary_angle_deg(20)`/`(6)`/`(3)`/`(11)`
    -- labels `double_outer_20`/`double_outer_6`/`double_outer_3`/
    `double_outer_11` (the exact label-to-angle pairing was checked
    numerically, not assumed) -- under the fixed formula below (uniform
    +9deg rotation from the old, wrong center-angle values), and were
    NOT reachable via the old implementation for any sector number. Every real call site in this
    codebase (tests/test_pnp_calibration.py, tests/test_pipeline_end_to_
    end.py, tests/test_board_geometry.py) selects landmarks purely by
    LABEL and re-derives geometry from this function's own output for a
    self-consistent synthetic project/recover round trip -- none hardcode
    an absolute angle value, so this fix is a uniform 9deg rotation of
    the whole landmark set and does not change any consuming test's
    relative geometry (convex-hull area, angular spread between chosen
    landmarks, etc. are all rotation-invariant) -- confirmed via the full
    suite passing unchanged after this fix, not just reasoned about.
    """
    points = [BoardPoint3D("bull", 0.0, 0.0)]
    ring_radii = {
        "double_outer": DOUBLE_OUTER_RADIUS_MM,
        "double_inner": DOUBLE_INNER_RADIUS_MM,
        "treble_outer": TREBLE_OUTER_RADIUS_MM,
        "treble_inner": TREBLE_INNER_RADIUS_MM,
    }
    for number in SECTOR_NUMBERS_CLOCKWISE:
        angle = wire_boundary_angle_deg(number)
        for ring_name, radius in ring_radii.items():
            x, y = polar_to_xy_mm(radius, angle)
            points.append(BoardPoint3D(f"{ring_name}_{number}", x, y))
    return points


def sector_ring_for_point(x_mm: float, y_mm: float) -> tuple[str | None, str]:
    """Given a board-plane point, return (sector_number_label_or_None, ring).

    ring in {"bull", "outer_bull", "treble", "double", "single_inner",
    "single_outer", "outside"}. sector label is None for bull/outer_bull/
    outside (no sector applies), else the wedge number as a string.

    The two INNER ring boundaries used here are the measured SCORING
    radii (`TREBLE_INNER_SCORING_RADIUS_MM` /
    `DOUBLE_INNER_SCORING_RADIUS_MM`), not the regulation ones -- see
    `INNER_RING_SCORING_OFFSET_MM`'s comment above for the real
    engine-independent measurement behind that, and for why the outer
    boundaries and the bull deliberately keep their regulation values.
    """
    r = math.hypot(x_mm, y_mm)
    if r <= BULL_RADIUS_MM:
        return None, "bull"
    if r <= OUTER_BULL_RADIUS_MM:
        return None, "outer_bull"
    if r > DOUBLE_OUTER_RADIUS_MM:
        return None, "outside"

    angle = math.degrees(math.atan2(x_mm, y_mm)) % 360.0
    idx = int((angle + SECTOR_ANGLE_DEG / 2) // SECTOR_ANGLE_DEG) % len(
        SECTOR_NUMBERS_CLOCKWISE
    )
    number = SECTOR_NUMBERS_CLOCKWISE[idx]

    if TREBLE_INNER_SCORING_RADIUS_MM <= r <= TREBLE_OUTER_RADIUS_MM:
        ring = "treble"
    elif DOUBLE_INNER_SCORING_RADIUS_MM <= r <= DOUBLE_OUTER_RADIUS_MM:
        ring = "double"
    elif r < TREBLE_INNER_SCORING_RADIUS_MM:
        ring = "single_inner"
    else:
        ring = "single_outer"
    return str(number), ring


def sector_ring_to_token(sector: str | None, ring: str | None) -> str:
    """The inverse-ish companion of `sector_ring_for_point()` above: map a
    scored `(sector, ring)` pair to a short, standard-darts-notation token
    for a human-facing quick-glance label -- specifically, the sector
    token that goes into a saved throw package's directory name (e.g.
    `<session>-<throw number>-T8`: session + sequential throw number +
    this token; see
    `opendarts.live.capture_daemon.handle_ready_to_capture()`).

    A filesystem-safe token for case-package naming, in this project's
    own `(sector, ring)` vocabulary.

    `ring` is expected to be one of this project's real vocabulary (see
    `sector_ring_for_point()`'s own docstring): `"bull"`, `"outer_bull"`,
    `"outside"`, `"single_inner"`, `"single_outer"`, `"treble"`,
    `"double"`. `single_inner`/`single_outer` deliberately collapse to
    the same plain `S{sector}` token -- they score identically (a single),
    and this token is a quick-glance label, not an internal precision
    distinction the way the two are for scoring math elsewhere in this
    module.

    Mapping:
    - `"bull"` -> `"DB"` (double bull / 50)
    - `"outer_bull"` -> `"OB"` (outer/single bull / 25)
    - `"outside"` -> `"OUT"`
    - `"single_inner"`/`"single_outer"` -> `f"S{sector}"`, e.g. `"S16"`
    - `"treble"` -> `f"T{sector}"`, e.g. `"T16"`
    - `"double"` -> `f"D{sector}"`, e.g. `"D16"`

    Callers scoring a real dart should route an `ok=False` primary result
    (no score at all) to the literal token `"NR"` (no result) themselves,
    WITHOUT calling this function -- `sector`/`ring` are likely `None` in
    that case, which isn't a real board position, and `"NR"` isn't part
    of the ring vocabulary this function maps. This function itself
    still raises on an unrecognized `ring` value (a real, honest failure
    -- silently falling back to some placeholder token would hide a
    vocabulary drift between this function and `sector_ring_for_point()`)
    rather than silently guessing.
    """
    if ring == "bull":
        return "DB"
    if ring == "outer_bull":
        return "OB"
    if ring == "outside":
        return "OUT"
    if ring in ("single_inner", "single_outer"):
        return f"S{sector}"
    if ring == "treble":
        return f"T{sector}"
    if ring == "double":
        return f"D{sector}"
    raise ValueError(f"unrecognized ring value: {ring!r}")
