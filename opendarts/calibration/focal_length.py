"""Image-only, per-camera focal-length derivation AT CALIBRATION
TIME -- replaces `opendarts.live.capture_daemon.MEASURED_FOCAL_LENGTH_PX`'s
role as the primary source of a camera's intrinsics focal length.

REAL BUG THIS CLOSES: `MEASURED_FOCAL_LENGTH_PX = {0: 800.0, 1: 830.0,
2: 835.0}` is keyed by OS-enumerated camera INDEX, not physical camera
identity. When the team re-seated which USB port each camera plugs into,
"cam0" the OS index started pointing at the physical unit that used to be
"cam2" while the lookup table
kept feeding it index 0's stale 800.0px guess -- a real, measured ~5%
focal-length error (best-fit ~838px vs the stale 800.0px guess, confirmed
by directly sweeping focal length against real detected landmark points
from this rig's own stored calibration frames: reprojection error 3.25px
at the stale guess vs 2.94px at the swept best fit). This module fixes
the actual disease, not just this one incident: a value that is
DERIVED FRESH from THIS calibration event's own images, every time,
tracks whatever physical camera currently sits at a given OS index
automatically -- there is no index-to-physical-camera table left to go
stale after a re-seat.

Same standing bar as the fix for the analogous orientation-hint
constant (`opendarts.calibration.ring_correlation_orientation`): a from-scratch,
camera-images-only derivation, not a corpus-matching exercise against the
old hardcoded numbers (do NOT tune this module to reproduce
`MEASURED_FOCAL_LENGTH_PX`'s old values: a worse-but-genuinely-general
measurement is real progress).

THE METHOD -- Zhang's single-plane-homography orthogonality constraint,
not a naive extra unknown bolted onto the existing 4-point PnP solve.
------------------------------------------------------------------------
`opendarts/calibration/pnp.py`'s own docstring (and
`opendarts.live.capture_daemon._try_solve()`'s inline comment) already
establish why naively adding focal length as a free unknown to the
existing 4-point PnP solve does not work: a single, roughly-fronto-
parallel-ish view of 4 coplanar points cannot jointly disambiguate focal
length from distance/pose (7 unknowns -- f, rvec, tvec -- from 8
equations, 1 DOF of real slack; adding distortion on top of that was
already rejected on the same grounds). This module does NOT try to
extend that solve. It uses a structurally DIFFERENT, decades-old
technique (Zhang, "A Flexible New Technique for Camera Calibration",
2000) that solves for focal length WITHOUT needing pose at all:

  1. `find_oriented_landmarks()` already computes, for every successfully
     oriented frame, ALL 20 of the double-ring's wire-junction landmarks
     (`OrientedLandmarkResult.ring20_px`) -- not just the 4-point
     cardinal-quad subset `average_correspondences()`/PnP actually solve pose
     from. All 20 are coplanar (board Z=0) points of KNOWN board-mm
     position (`ring20_object_points_mm()`, the same formula
     `oriented_landmarks.ad_quad_object_points_mm()` uses, just for every
     k in range(20) instead of the 4 quad indices) -- richer,
     better-spread data than the 4-point quad, already computed by the
     existing detection pass, at zero extra detection cost.
  2. Fit ONE planar homography H (board XY mm -> image px) via DLT from
     these (up to 20) correspondences -- `cv2.findHomography`. This needs
     no camera matrix, no pose, nothing but the correspondences
     themselves.
  3. The dartboard's own board-mm coordinate system is a genuine
     Euclidean (orthonormal, equal-scale) frame by construction
     (`opendarts.geometry.board.polar_to_xy_mm`'s plain polar-to-Cartesian
     mapping) -- exactly the assumption Zhang's method needs. Given a
     known, fixed principal point (image center -- this project's
     existing, unchanged assumption, see `_camera_matrix_for()`), zero
     skew, and square pixels, the image of the absolute conic (IAC) has
     exactly ONE unknown, x = 1/f^2. H's own orthogonality (its columns
     are K.r1 and K.r2, and r1 perp r2 with |r1|=|r2|=1) gives TWO
     independent LINEAR equations in that one unknown from this SINGLE
     homography -- `derive_focal_length_from_ring20()` solves them by
     least squares. This is decoupled from pose entirely: f falls out
     directly from the homography's shape, with none of the "1 DOF of
     slack" fragility the naive joint-PnP-with-unknown-f approach has.
     Pose is still solved exactly as before, downstream, via the
     existing 4-point `calibrate_camera()` -- this module only replaces
     what value it's handed as the INITIAL, in this project's case FINAL
     (K is never refined jointly with pose -- see `pnp.py`'s own
     docstring) camera_matrix.

VALIDATED SYNTHETICALLY FIRST, THEN ON REAL DATA. An exploratory sweep
(not shipped) confirms: noiseless recovery is exact (<1e-3px error) at this rig's real
focal lengths (800.0/830.0/835.0px) and all 3 real camera azimuths at
plausible rig geometry (300-800mm camera-to-board distance, 200-500mm
mount height); realistic per-point pixel noise (0.1-1.0px, in line with
this project's own documented sub-pixel landmark-detection precision)
gives an UNBIASED estimate (mean error ~0) with a few-px standard
deviation per SINGLE frame, further reduced by this module's own
per-camera multi-frame averaging (`derive_focal_length_from_oriented_
results()`) exactly the way `average_correspondences()` already reduces
noise for the existing 4-point pose solve. Only a genuinely unrealistic
geometry (a camera 2m+ from the board -- this rig's cameras are all
within ~1m, see `tests/support/synthetic.py`'s own real-rig-matching
defaults) showed materially degraded conditioning.

REAL DATA VALIDATION: measured against this rig's own real stored
calibration frames (the recorded calibration package this bug was found
from). This module consumes only camera images
+ this project's own board geometry constants (`opendarts.geometry.board`),
same discipline as `opendarts.calibration.ring_boundary_offset` and
`opendarts.calibration.ring_correlation_orientation`.

STORAGE -- JSON, not code. This module derives a value fresh every real calibration event
(cheap -- closed-form linear algebra over <=20 points, no iterative
solve) rather than caching it in a Python constant; the ONLY thing ever
written to Python source for this purpose is
`opendarts.live.capture_daemon.MEASURED_FOCAL_LENGTH_PX`, and even that is
now a last-resort fallback of last resort (see that constant's own
updated comment) -- kept only as
`bootstrap_calibrations()`'s own last-resort fallback: a real safety
net for a degenerate/fresh-rig case where THIS event's own live
derivation genuinely cannot produce anything AND no prior persisted
JSON value exists yet either. `write_focal_length_fallback_entry()`
persists every SUCCESSFUL live derivation to a small JSON file
(`FOCAL_LENGTH_FALLBACK_FILENAME`, living next to a rig's calibration
event packages -- same directory `calibration_package_root` already
points at, mirroring the "small persistent file living alongside a
rig's own calibration events" role `ring_boundary_offset.json` plays at
the session level) so a LATER calibration event whose own live
derivation is degraded/unavailable for one camera can fall back to "the
last real derived value for this INDEX", not the code constant --
self-correcting (the very next successful live event overwrites it) in
a way a hardcoded Python constant structurally cannot be, and always
loudly logged as a degraded, non-fresh value when used (see
`opendarts.live.capture_daemon`'s own call site).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM, SECTOR_ANGLE_DEG, polar_to_xy_mm

log = logging.getLogger(__name__)

# 20 wire-junction landmarks around the double-outer ring, one per
# sector, matching `oriented_landmarks.OrientedLandmarkResult.ring20_px`'s
# own index convention exactly (index k = board angle FIRST_WIRE_ANGLE_DEG
# + SECTOR_ANGLE_DEG*k -- see `oriented_landmarks.ad_quad_object_points_mm()`,
# which this mirrors for every k instead of just the 4 quad indices).
# Not imported from `oriented_landmarks` (FIRST_WIRE_ANGLE_DEG lives
# there, not in `opendarts.geometry.board`) to avoid a needless import-time
# dependency on that much larger module for what is a 2-line formula;
# the two are independently checked to agree by
# `tests/test_focal_length.py`.
N_RING_LANDMARKS = 20
FIRST_WIRE_ANGLE_DEG = SECTOR_ANGLE_DEG / 2.0 # 9.0 -- the 20/1 wire, matches oriented_landmarks.py


def ring20_object_points_mm() -> np.ndarray:
    """The 20 wire-junction board points, Z=0 mm, in `ring20_px` index
    order. See module docstring "THE METHOD" step 1."""
    out = np.zeros((N_RING_LANDMARKS, 3), dtype=np.float64)
    for k in range(N_RING_LANDMARKS):
        angle = FIRST_WIRE_ANGLE_DEG + SECTOR_ANGLE_DEG * k
        x, y = polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, angle)
        out[k] = (x, y, 0.0)
    return out


_RING20_OBJECT_POINTS_MM = ring20_object_points_mm()

# Plausibility bounds on the DERIVED focal length -- a real, physical
# sanity floor/ceiling, not a rig-specific tuning knob. 1280px-wide
# images: 300px would be an implausibly extreme fisheye (~150+ deg HFOV)
# for a webcam-class board camera, 3000px an implausibly narrow/zoomed
# lens (~24 deg HFOV) for a camera meant to see most of a dartboard from
# under a meter away. A derivation landing outside this range is treated
# as a geometric/numerical failure (degenerate homography, bad
# correspondences), not a real measurement, regardless of what the linear
# solve technically returned.
MIN_FOCAL_LENGTH_PX = 300.0
MAX_FOCAL_LENGTH_PX = 3000.0

# Minimum ring20 landmark INDICES with real correspondence data before a
# homography fit is even attempted. A successfully-oriented frame always
# contributes all 20 (see module docstring -- `ring20_px` is all-or-
# nothing per frame, `OrientedLandmarkResult.ok=False` on any single wire
# miss), so this floor only ever matters as defensive code against a
# future change to that invariant; 8 is comfortably above
# cv2.findHomography's own bare minimum (4) while still well below 20.
MIN_RING20_POINTS_FOR_HOMOGRAPHY = 8


@dataclass(frozen=True)
class FocalLengthResult:
    ok: bool
    focal_length_px: float | None
    reason: str
    n_points_used: int
    n_frames_used: int


def derive_focal_length_from_ring20(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    principal_point: tuple[float, float],
    *,
    n_frames_used: int = 0,
) -> FocalLengthResult:
    """THE METHOD, steps 2-3 of the module docstring -- given already-
    averaged (object_points_mm, image_points_px) ring20 correspondences
    (see `derive_focal_length_from_oriented_results()` for the averaging
    step), fit one planar homography and solve Zhang's 2 orthogonality
    equations for f = sqrt(1/x). `principal_point` is `(cx, cy)` in image
    pixels -- this project's existing fixed-principal-point assumption
    (`_camera_matrix_for()`'s own `image_width/2, image_height/2`),
    unchanged by this module.

    Degeneracy is checked explicitly, never silently swallowed into a
    wrong-looking-plausible number: a near-fronto-parallel homography
    (both orthogonality equations' own coefficients near zero relative to
    H's own scale) or a non-physical solution (x <= 0, i.e. an imaginary
    f) is rejected with a real `reason`, not clamped or guessed past.
    """
    object_points_mm = np.asarray(object_points_mm, dtype=np.float64)
    image_points_px = np.asarray(image_points_px, dtype=np.float64)
    n = len(object_points_mm)
    if n < MIN_RING20_POINTS_FOR_HOMOGRAPHY:
        return FocalLengthResult(
            ok=False, focal_length_px=None,
            reason=f"only {n} ring20 correspondence(s), need >= {MIN_RING20_POINTS_FOR_HOMOGRAPHY}",
            n_points_used=n, n_frames_used=n_frames_used,
        )
    if len(image_points_px) != n:
        return FocalLengthResult(
            ok=False, focal_length_px=None,
            reason=f"object/image point count mismatch ({n} vs {len(image_points_px)})",
            n_points_used=n, n_frames_used=n_frames_used,
        )

    import cv2

    obj_xy = object_points_mm[:, :2]
    H, _mask = cv2.findHomography(obj_xy, image_points_px, method=cv2.LMEDS)
    if H is None or not np.isfinite(H).all():
        return FocalLengthResult(
            ok=False, focal_length_px=None,
            reason="cv2.findHomography failed or returned a non-finite homography",
            n_points_used=n, n_frames_used=n_frames_used,
        )

    cx, cy = principal_point
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    Hc = T @ H
    h1 = Hc[:, 0]
    h2 = Hc[:, 1]

    # Zhang's 2 orthogonality equations (r1 perp r2, |r1|=|r2|=1) on the
    # image of the absolute conic omega = diag(x, x, 1), x = 1/f^2, in
    # principal-point-centered coordinates -- see module docstring "THE
    # METHOD" step 3 for the derivation. Both are LINEAR in the single
    # unknown x, so the two rows below form a well-defined (over-
    # determined by exactly 1 row) least-squares system for x.
    rowA_coef = h1[0] * h2[0] + h1[1] * h2[1]
    rowA_rhs = -(h1[2] * h2[2])
    rowB_coef = (h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)
    rowB_rhs = h2[2] ** 2 - h1[2] ** 2

    # Degeneracy guard: scale-relative, not an absolute epsilon, since H
    # itself is only defined up to an arbitrary overall scale (cv2's own
    # normalization is not guaranteed stable across calls/inputs) -- both
    # rowA_coef/rowB_coef scale the same way H's entries do, so comparing
    # them against H's own entry magnitudes is the correct scale-free
    # check, not a fixed pixel-scale constant.
    h_scale = float(max(abs(h1[0]), abs(h1[1]), abs(h2[0]), abs(h2[1]), 1e-12)) ** 2
    if abs(rowA_coef) < 1e-9 * h_scale and abs(rowB_coef) < 1e-9 * h_scale:
        return FocalLengthResult(
            ok=False, focal_length_px=None,
            reason="degenerate homography for this view (both orthogonality "
                   "equations near-zero -- board plane too close to fronto-parallel "
                   "for this technique to disambiguate focal length)",
            n_points_used=n, n_frames_used=n_frames_used,
        )

    M = np.array([[rowA_coef], [rowB_coef]], dtype=np.float64)
    rhs = np.array([rowA_rhs, rowB_rhs], dtype=np.float64)
    x, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    x = float(x[0])
    if not np.isfinite(x) or x <= 0.0:
        return FocalLengthResult(
            ok=False, focal_length_px=None,
            reason=f"non-physical solution (1/f^2={x!r}, expected > 0)",
            n_points_used=n, n_frames_used=n_frames_used,
        )
    f = float(1.0 / np.sqrt(x))
    if not (MIN_FOCAL_LENGTH_PX <= f <= MAX_FOCAL_LENGTH_PX):
        return FocalLengthResult(
            ok=False, focal_length_px=None,
            reason=f"derived f={f:.1f}px outside plausible "
                   f"[{MIN_FOCAL_LENGTH_PX:.0f}, {MAX_FOCAL_LENGTH_PX:.0f}]px range",
            n_points_used=n, n_frames_used=n_frames_used,
        )
    return FocalLengthResult(
        ok=True, focal_length_px=f, reason="ok",
        n_points_used=n, n_frames_used=n_frames_used,
    )


def ring20_image_points_from_result(result: Any) -> np.ndarray | None:
    """Extract a usable (20, 2) `ring20_px` array from one
    `oriented_landmarks.OrientedLandmarkResult`-like object, or None if
    this frame's result isn't usable for focal-length derivation --
    same admission gate `correspond_landmarks_from_pre_orientation()`
    already applies for the 4-point pose correspondence (`ok` and not
    `orientation_ambiguous`), since an ambiguous orientation lock means
    ALL 20 landmark positions, not just the quad's 4, may belong to the
    wrong rotation of the board (see that function's own docstring).
    Duck-typed (no isinstance check) so a test double with the same
    3 attributes works without importing `oriented_landmarks` here.
    """
    if not getattr(result, "ok", False):
        return None
    if getattr(result, "orientation_ambiguous", False):
        return None
    ring20_px = getattr(result, "ring20_px", None)
    if ring20_px is None:
        return None
    arr = np.asarray(ring20_px, dtype=np.float64)
    if arr.shape != (N_RING_LANDMARKS, 2) or not np.isfinite(arr).all():
        return None
    return arr


def average_ring20_image_points(per_frame_points: list[np.ndarray]) -> np.ndarray:
    """Per-landmark-index MEDIAN across `per_frame_points` (each a (20,2)
    array from `ring20_image_points_from_result()`) -- deliberately a
    plain median, not `sector_correspondence._trimmed_mean_image_points()`
    (that helper is hardcoded to exactly 4 landmarks -- `np.zeros((4,2))`,
    `range(4)` -- reusing it here for 20 points would silently drop
    indices 4-19, not raise; see this module's own tests for a guard
    against ever doing that by accident). A per-index median is already
    robust to a minority of noisy/outlier frames without needing this
    module to duplicate that helper's own MAD-threshold tuning."""
    stack = np.stack(per_frame_points, axis=0) # (n, 20, 2)
    return np.median(stack, axis=0)


def derive_focal_length_from_oriented_results(
    results: list[Any],
    principal_point: tuple[float, float],
    *,
    min_frames: int = 1,
) -> FocalLengthResult:
    """Full pipeline: filter `results` (a camera's accumulated
    `OrientedLandmarkResult` list, e.g.
    `opendarts.live.capture_daemon.bootstrap_calibrations()`'s own
    `accumulated_results[cam]`) down to usable ring20 correspondences,
    average them across frames, then derive f.

    `min_frames`: the caller's own trust floor for how many independent
    frames must have contributed before an average is worth solving from
    -- callers should pass the SAME value
    `opendarts.live.capture_daemon._min_calibration_frames_required()`
    already computes for the existing 4-point pose correspondence
    average, so this derivation and the pose solve share one consistent
    "how many real samples do we trust" floor rather than inventing a
    second one.
    """
    per_frame = [
        pts for pts in (ring20_image_points_from_result(r) for r in results) if pts is not None
    ]
    n_frames = len(per_frame)
    if n_frames < max(1, min_frames):
        return FocalLengthResult(
            ok=False, focal_length_px=None,
            reason=f"only {n_frames} usable frame(s) for ring20 correspondence, need >= {min_frames}",
            n_points_used=0, n_frames_used=n_frames,
        )
    averaged = average_ring20_image_points(per_frame)
    return derive_focal_length_from_ring20(
        _RING20_OBJECT_POINTS_MM, averaged, principal_point, n_frames_used=n_frames,
    )


# ---------------------------------------------------------------------
# JSON persistence -- "these are things stored in json files for use...
# they are never meant to be in code". See module
# docstring's "STORAGE" section for the full design.
# ---------------------------------------------------------------------

FOCAL_LENGTH_FALLBACK_FILENAME = "focal_length_fallback.json"
SCHEMA = "focal-length-fallback-v1"


def load_focal_length_fallback(calibration_package_root: Path) -> dict[int, dict]:
    """Load the persisted last-known-good per-camera-INDEX focal-length
    table from `<calibration_package_root>/focal_length_fallback.json`,
    or `{}` if the file doesn't exist, doesn't parse, or doesn't match
    this module's current `SCHEMA` -- same "absent/corrupt/stale-schema
    all degrade to the same safe empty state, loudly logged, never
    raised" posture `ring_boundary_offset.load_session_ring_boundary_
    offset()` already established for this exact kind of small
    persisted-measurement file.

    Returned dict is keyed by camera index (int, not str -- JSON object
    keys are always strings on disk; this function does the int()
    conversion so callers never have to). Each value is the camera's
    full persisted record (`focal_length_px`, `derived_at_utc`,
    `n_frames_used`, `n_points_used`, `package_id`) -- callers that only
    want the float should read `["focal_length_px"]`.

    **Index-staleness is real but bounded, unlike the code constant this
    replaces**: a value in this file was itself live-derived from real
    images at some PAST calibration event, for whatever physical camera
    sat at this index THEN. If a physical re-seat happened since, this
    entry may not describe the CURRENT camera at that index -- exactly
    like `MEASURED_FOCAL_LENGTH_PX`'s own real bug. The difference: this
    file is overwritten by every SUCCESSFUL live derivation
    (`write_focal_length_fallback_entry()`), so the staleness window is
    bounded to "since the last successful live calibration", not
    "forever, until a human edits Python source" -- and every caller that
    falls back to this file logs it loudly as a degraded, non-fresh
    value (see `opendarts.live.capture_daemon`'s own call site), never
    silently.
    """
    path = Path(calibration_package_root) / FOCAL_LENGTH_FALLBACK_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        log.exception(
            "%s: failed to read/parse -- treating as absent (falls back to "
            "the hardcoded MEASURED_FOCAL_LENGTH_PX constant, same as no "
            "file at all)", path,
        )
        return {}
    if payload.get("schema") != SCHEMA:
        log.warning(
            "%s: schema %r does not match current %r -- treating as absent",
            path, payload.get("schema"), SCHEMA,
        )
        return {}
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict):
        return {}
    out: dict[int, dict] = {}
    for key, record in cameras.items():
        try:
            cam = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(record, dict) and isinstance(record.get("focal_length_px"), (int, float)):
            out[cam] = record
    return out


def write_focal_length_fallback_entry(
    calibration_package_root: Path,
    cam: int,
    focal_length_px: float,
    *,
    n_frames_used: int,
    n_points_used: int,
    derived_at_utc: str,
    package_id: str | None = None,
) -> None:
    """Merge-update ONE camera's entry into the persisted fallback table
    -- read-modify-write, preserving every OTHER camera's existing entry
    untouched (a camera that didn't derive live THIS event must keep its
    own last-known-good value, not have it wiped just because a sibling
    camera updated). Called only after a SUCCESSFUL live derivation --
    never writes a fallback-sourced or failed result back into this file
    (that would let a degraded value silently become the new "last
    known good", defeating the self-correcting property this file exists
    for).

    Plain `path.write_text(json.dumps(...))`, matching
    `ring_boundary_offset.write_session_ring_boundary_offset()`'s own
    convention exactly (not a tmp-file-plus-rename atomic write) -- a
    torn write from a crash mid-write degrades to "absent/corrupt", which
    `load_focal_length_fallback()` already treats as a safe, loud, non-
    fatal fallback state, the same posture that existing sibling file
    already relies on.
    """
    calibration_package_root = Path(calibration_package_root)
    calibration_package_root.mkdir(parents=True, exist_ok=True)
    path = calibration_package_root / FOCAL_LENGTH_FALLBACK_FILENAME
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict) and loaded.get("schema") == SCHEMA:
                existing = loaded
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            log.exception(
                "%s: failed to read existing fallback table before updating -- "
                "starting a fresh one (this camera's own new value is still "
                "correctly written; only OTHER cameras' prior entries are lost)",
                path,
            )
    cameras = existing.get("cameras")
    if not isinstance(cameras, dict):
        cameras = {}
    cameras[str(cam)] = {
        "focal_length_px": float(focal_length_px),
        "derived_at_utc": derived_at_utc,
        "n_frames_used": int(n_frames_used),
        "n_points_used": int(n_points_used),
        "package_id": package_id,
    }
    payload = {"schema": SCHEMA, "updated_at_utc": derived_at_utc, "cameras": cameras}
    path.write_text(json.dumps(payload, indent=2))
