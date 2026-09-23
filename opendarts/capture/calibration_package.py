"""Calibration package format -- REPLAY applied to calibration bootstrap
itself, not just scored throws. See docs/DESIGN.md's "Replay is the
source of truth" before touching this file: the same discipline that
governs `opendarts.capture.throw_package` applies here, one level up the
stack -- whatever raw pixels `bootstrap_calibrations()` actually
analyzed to solve a camera's pose must be exactly what's on disk, so a
later replay of a calibration event (not just a scored throw) can
reproduce -- or, once the calibration stack changes, UPDATE -- the same
solved pose.

WHY THIS EXISTS (2026-08-19/20): before this module, this project had
grown several separate, ad-hoc sibling files for different derived
calibration values (`calibration_refit.json` -- the offline session
refit, `save_calibration_snapshot()`'s own
timestamped/`latest.json` pair -- `opendarts.live.capture_daemon`), none of
which shared a single versioned identity, and NONE of which preserved
the raw calibration-bootstrap frames themselves. This module is the real
"calibration package": one self-contained, versioned bundle, written
every time `opendarts.live.capture_daemon.bootstrap_calibrations()` runs a
REAL (not-reused) live calibration event, containing:

  1. A real package id, matching this project's existing timestamp-based
     session-id convention (`calib_<YYYYMMDD-HHMMSS>-<8 hex chars>`,
     mirroring `session_id = time.strftime("%Y%m%d-%H%M%S")` in
     `opendarts.live.capture_daemon.run_capture_loop_body()`, plus a random
     suffix -- see `new_calibration_package_id()`'s own docstring for why
     the suffix is load-bearing, not decorative: it's what keeps two
     overlapping calibration events from ever colliding on one id).
  2. The raw calibration-bootstrap frames themselves (up to
     `opendarts.live.capture_daemon.CALIBRATION_MAX_N_FRAMES` per camera --
     see `save_calibration_package()`'s own docstring for the exact set:
     EVERY raw frame captured for that camera this event, across every
     retry round, not just the frames from the final accepted round --
     UPDATED 2026-08-21, fixing a real REPLAY bug a verifier pass found:
     this used to be only the subset `bootstrap_calibrations()` actually
     ran landmark detection on, which undercounted once that function's
     own DECOUPLED CAPTURE-VS-DETECT TARGET change (same day) started
     capturing more raw frames than it detects -- the package now
     matches `derived_calibration.json`'s own `n_frames_raw_pool`
     diagnostic exactly, not just the detected subset).
  3. Every derived calibration value `bootstrap_calibrations()` actually
     produces today (solved `camera_matrix`/`dist_coeffs`/`rvec`/`tvec`
     per camera, reprojection error, `landmark_spread_ok`, the
     `target_met`/`n_frames_used` diagnostics `diagnostics_out` already
     surfaces). Deliberately does NOT wire in any of this project's other
     still-unmerged derived-value work (orientation hints /
     intrinsics-derivation / ring-boundary-offset / adaptive color
     detection) -- this format captures whatever the live pipeline
     produces TODAY and will start capturing more once those other
     pieces get wired into `bootstrap_calibrations()` itself, without
     this module needing to change.

STORAGE FORMAT FOR RAW FRAMES -- FFV1, confirmed byte-exact, not a
re-litigation of lossy alternatives (see `_encode_raw_video()` /
`_decode_raw_video()` below and this module's own tests):

  - Lossy H.264 was ruled out earlier the same day this module was built
    -- even "lossless" mode (`-qp 0`) was NOT byte-exact against real
    camera frames (measured 0/62 frames exact, off by up to 2, almost
    certainly RGB<->YUV colorspace rounding).
  - FFV1 is written and read through `cv2.VideoWriter` /
    `cv2.VideoCapture` (fourcc `FFV1`), feeding cv2's own BGR uint8
    arrays straight in and out with no PNG round trip. The FFV1 encoder
    is bundled inside the `opencv-python(-headless)` wheel, so this needs
    NO external `ffmpeg` binary on any platform -- verified byte-exact on
    macOS (arm64), Linux (x86_64 + aarch64) and Windows (x64, incl. under
    ARM emulation), 2026-09-20. Byte-exactness is reconfirmed by THIS
    module's own tests against real corpus frames (`data/archive/clean/`)
    and, defensively, re-verified on every encode (`_encode_raw_video()`
    decodes what it just wrote and refuses a non-identical round trip).
  - No lossy fallback exists anywhere in this module, on purpose -- a
    writer that will not open, or a round trip that is not byte-exact,
    raises `RuntimeError` loudly rather than silently degrading to a
    lossy codec. `save_calibration_package()` swallows that (and every
    other) raw-video encoding failure into a logged warning + an honest
    per-camera `"raw_video": null` in `meta.json` instead of ever writing
    a lossy substitute -- "no frames on disk for this camera" is honest;
    a JPEG that LOOKS like the real frames but silently isn't is the one
    thing REPLAY cannot tolerate (see docs/DESIGN.md: "never soften a
    replay gate to accept drift").

Package layout on disk (one directory per calibration event, under
`DEFAULT_CALIBRATION_PACKAGE_ROOT`, i.e. `<repo>/data/calibration_packages/`):

    calib_<YYYYMMDD-HHMMSS>-<8 hex chars>/
        meta.json -- package_id, created_at_utc, schema,
                                     codec/pix_fmt, and per-camera raw
                                     video bookkeeping (n_frames, frame
                                     width/height, filename, or `null`
                                     for a camera whose encode failed --
                                     see `_CalibrationPackageMeta` below).
        derived_calibration.json -- per-camera solved CameraCalibration
                                     (same `calibration_to_dict()` shape
                                     `opendarts.capture.throw_package`
                                     already uses -- one on-disk shape
                                     for "a real CameraCalibration", not
                                     a second one) PLUS
                                     reprojection_error_px/n_frames_used/
                                     target_met/focal_length_px/
                                     focal_length_source/
                                     orientation_hint_deg/
                                     orientation_hint_source/
                                     n_raw_extra_frames_used/
                                     frame_indices_used/
                                     raw_extra_frame_indices/k1/
                                     distortion_source/cx_px/
                                     cx_focal_length_px/cx_k1/
                                     joint_focal_length_px/
                                     principal_point_source per camera (the
                                     `diagnostics_out` shape
                                     `bootstrap_calibrations()` already
                                     produces -- see that function's own
                                     docstring), PLUS a package-wide
                                     `code_version` (best-effort git SHA)
                                     and, when the underlying measurement
                                     ran this event, two full NESTED
                                     blocks -- `ring_boundary_offset`
                                     (schema/solved_by/calibration_source/
                                     source_images/parameters/boundaries/
                                     median_profiles/code_version) and
                                     `board_color_calibration`
                                     (schema/solved_by/
                                     brightness_threshold_black_cream/
                                     chroma_threshold/.../code_version) --
                                     this is the canonical shape
                                     (2026-08-27, the v2 package schema;
                                     see this module's own dated "REPLACE
                                     FOR NEW WRITES" comment below for the
                                     full reasoning, including the ONE
                                     flat legacy key kept,
                                     `ring_boundary_offset_accepted`, and
                                     the package-wide timing fields --
                                     both opendarts-only fields, not
                                     yet removed). A
                                     package saved BEFORE 2026-08-27 still
                                     has the OLD flat scalar keys
                                     (`treble_inner_offset_mm`,
                                     `brightness_threshold`, etc.,
                                     `ring_boundary_measurement`) instead
                                     -- readers (`opendarts.capture.
                                     throw_package.load_throw_package()`'s
                                     `_package_ring_boundary_offsets()`/
                                     `_package_board_color_threshold_
                                     value()`) dispatch on KEY PRESENCE to
                                     handle either real on-disk shape
                                     correctly; this file's own `schema`
                                     was NOT bumped by the reshape (see
                                     those functions' own docstrings for
                                     why a schema-string dispatch would
                                     not be safe here).
        cam{N}_raw.mkv -- FFV1-encoded raw bootstrap-burst
                                     frames for camera N, one file per
                                     camera that had at least one frame
                                     captured. ABSENT (not a placeholder)
                                     for a camera whose encode failed --
                                     `meta.json`'s own per-camera entry
                                     says so explicitly (see above).

LIVE WIRING -- this package's whole point is to run on every real
calibration event, unlike most of this project's other same-day
additive/validation-stage work. See `opendarts.live.capture_daemon.
bootstrap_calibrations()`'s own "CALIBRATION PACKAGE" docstring section
for exactly how it's invoked from there, and the honest constraint this
demanded: raw-frame capture + FFV1 encoding must never slow down or risk
breaking the real calibration/pose-solving result. `save_calibration_
package()` itself is a plain, synchronous, directly-testable function --
the live wiring runs it off the calibration-solving critical path (a
background thread, fire-and-forget from the caller's point of view), and
every failure mode is caught and logged, never re-raised into the live
bootstrap.

**MEASURED FFV1 timing, real number, not a round-number guess**: encoding
`CALIBRATION_MAX_N_FRAMES` (200, the real worst case -- a chronically-bad
camera that never clears its reprojection target) of real 1280x720
camera frames for ONE camera measured 2.68s on the dev machine this was
built/tested on (a healthy camera accepting at N=30-55, this project's
own real measured range, costs proportionally less). Up to 3 cameras'
encodes happen sequentially inside the SAME background thread (not
parallelized across cameras -- see `save_calibration_package()`'s own
per-camera loop), so a worst-case "every camera chronically bad" event
costs roughly 3x that, ~8s, entirely off the critical path (the real
calibration result has already been returned to its caller by then).
Not yet measured on the rig's own hardware.

**CONCURRENT SAVES**, added 2026-08-20 after a verifier pass caught the
real gap: two overlapping live calibration events (e.g. a double-clicked
"Refresh calibration now," or a manual refresh landing while a startup
bootstrap's own background save is still encoding) could otherwise race
against each other's post-save `cleanup_orphaned_calibration_packages()`
pass -- one event's still-encoding, not-yet-throw-referenced package is
neither the OTHER event's `active_package_id` nor referenced by any
throw yet, so a plain cleanup could delete a package while its own
raw-frame encode is still writing to it. `_IN_FLIGHT_PACKAGE_IDS` (module-
level, lock-guarded) tracks every package currently mid-save across the
whole process; `cleanup_orphaned_calibration_packages()` always treats
every in-flight id as kept, regardless of which specific save call
triggered that particular cleanup pass. Registered the moment
`save_calibration_package_background()` is called (synchronously, before
the background thread even starts) and cleared only after that thread's
own save AND cleanup work has both finished -- see that function's own
docstring for the exact window.

A SECOND verifier pass, same day, found this guard alone was not
sufficient: `new_calibration_package_id()` only has 1-second resolution
and nothing serializes overlapping calibration events, so two genuinely
different events (the same double-click scenario above) could resolve
to an IDENTICAL `package_id` -- both targeting the same on-disk
directory, risking one event's `derived_calibration.json` silently
pairing with a DIFFERENT event's raw video, a real REPLAY-integrity
violation. Fixed two ways: (1) `new_calibration_package_id()` now
appends a random hex suffix by default (`secrets.token_hex(4)`), making
two concurrent calls' ids collide only astronomically rarely, which is
the PRIMARY fix (it prevents two writers from ever targeting the same
directory in the first place); (2) `_IN_FLIGHT_PACKAGE_IDS` is a
`Counter`, not a plain `set`, so that even the residual same-id
collision case degrades safely -- two overlapping registrations of one
id are tracked independently and the id stays protected until BOTH
calls have finished, instead of the first call's `finally` block
silently un-protecting the still-running second one. **Known remaining
gap, not closed by any of this**: `_IN_FLIGHT_PACKAGE_IDS` is
process-local, so this module's own standalone cleanup CLI (see the
bottom of this module) run as a separate process gets zero visibility
into a live server process's in-flight saves -- see
`save_calibration_package_background()`'s own docstring for the honest
caveat.

RIG-SIDE CLEANUP -- `cleanup_orphaned_calibration_packages()` below.
"Calibrate a few times while dialing it in, then throw" is real usage
 -- most calibration events on a real rig never get
referenced by any throw. A calibration package is kept iff it is either
the currently-active one OR referenced (`calibration_package_id` in a
throw's `meta.json` -- see `opendarts.capture.throw_package`) by at least
one throw package still present on disk; anything else is a real,
disposable-by-design artifact (raw board photos with zero scored value
once superseded) and this function really deletes it -- NOT a quarantine
into `data/_delete/`, deliberately different from this project's
"never permanently delete" corpus guardrail (docs/DESIGN.md), which is about
historical SCORED throw data, not disposable calibration-dialing-in
exhaust. See that function's own docstring for the full reasoning and
why it is a standalone, explicitly-invoked function rather than
auto-wired into every live bootstrap.

Deliberately NOT touched by this module, matching every other same-day
"still-unwired" piece: intrinsics derivation, ring-boundary offset,
board-color calibration, orientation-hint auto-detection. This format
captures whatever `bootstrap_calibrations()` ACTUALLY solves today
(camera_matrix/dist_coeffs/rvec/tvec via PnP from averaged 4-point
correspondences) -- see that function's own docstring for the real
algorithm.
"""
from __future__ import annotations

import functools
import json
import logging
import secrets
import shutil
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from opendarts.calibration.ring_boundary_offset import SCHEMA as _RING_BOUNDARY_OFFSET_SCHEMA
from opendarts.capture.throw_package import calibration_from_dict, calibration_to_dict
from opendarts.geometry.board_color_calibration import SCHEMA as _BOARD_COLOR_CALIBRATION_SCHEMA
from opendarts.live import capabilities
from opendarts.pipeline import CameraCalibration

log = logging.getLogger(__name__)

# CONCURRENT-SAVE RACE GUARD -- see module docstring's "CONCURRENT SAVES"
# section. Every package id currently mid-save (registered synchronously
# by save_calibration_package_background() before its thread starts,
# cleared only after that thread's own save AND any follow-up cleanup
# pass have both finished) so cleanup_orphaned_calibration_packages()
# never deletes a package another in-flight save call is still writing
# to, regardless of which specific call triggered that cleanup pass.
# Process-global (not per-caller) on purpose -- the race this guards
# against is BETWEEN two overlapping save calls, so the guard has to be
# visible to both, not scoped to either one's own closure.
#
# A `Counter`, not a plain `set` -- REAL BUG, found by a verifier pass
# 2026-08-20 and fixed here, not hypothetical: `new_calibration_package_id()`
# only has 1-second resolution and nothing anywhere serializes overlapping
# calibration events (a fast double-click on "Refresh calibration now",
# the exact scenario this guard exists for, can easily land two
# `bootstrap_calibrations()` calls in the same UTC second). With a plain
# `set`, two concurrent save calls that happen to share one `package_id`
# would both add the same string, and whichever call's `finally` block
# finishes FIRST would `.discard()` it -- silently stripping in-flight
# protection from the OTHER call while it is still mid-encode, even
# though its own registration is still logically active. A `Counter`
# makes each registration/discard independently balanced (increment on
# register, decrement on discard, membership checked via `> 0`) so two
# overlapping registrations of the same id are both honored until BOTH
# have finished -- see `save_calibration_package_background()`'s own
# docstring for the paired half of this fix (a collision-resistant
# `package_id`, which is the primary fix; this Counter is defense in
# depth for the residual, now near-impossible, same-id collision case).
_IN_FLIGHT_PACKAGE_IDS: Counter[str] = Counter()
_IN_FLIGHT_LOCK = threading.Lock()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from opendarts.paths import DATA_DIR

# Same "<repo>/data/<thing>" convention as DEFAULT_PACKAGE_ROOT
# (opendarts.live.capture_daemon) -- gitignored via the existing `/data/`
# entry, never committed. This is the rig's own LIVE write path -- out of
# scope for the 2026-08-22 dev-machine data-layout move (see docs/DESIGN.md),
# unlike DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT below. (The
# lightweight, display-oriented snapshot/`latest.json` pair this
# comment used to describe as a sibling -- `save_calibration_snapshot()`
# / data/calibrations/ -- was removed 2026-08-22 as genuinely dead
# code; this remains the one real calibration-package root, the new,
# much larger (raw video included), fully self-contained REPLAY
# package.)
DEFAULT_CALIBRATION_PACKAGE_ROOT = DATA_DIR / "calibration_packages"

# Second, ARCHIVED root, added 2026-08-22 (REPLAY-persistence live-wiring
# task). Formalizes the archive layout an off-rig pull uses
# (`<archive>/calibration_packages`, i.e. this same path) -- NOT a new
# convention, just the first time it has a Python-side name.
#
# 2026-08-22, later same night -- moved OUT of the repo entirely, per
# the project's own greenfield data-layout call: dev-machine data (pulled
# corpus, pulled calibration packages, quarantine) must never live
# inside a git working directory again -- that's the exact mechanism
# that let a stray `pytest` run leak a real fake throw straight into
# OpenDarts' own git checkout the same night (see docs/DESIGN.md). opendarts's
# own gitignore already meant this specific class of bug couldn't
# happen here, but the OTHER real risks (a `git clean -fdx` treating
# real corpus data as disposable cruft, a fresh isolated worktree not
# carrying gitignored data along) still applied. New home: a
# `calibration_packages/` directory -- a sibling of `sessions/` (the
# throw corpus, was `data/archive/clean/`) and `quarantine/` (was
# `data/_delete/`), all under one project-scoped root, entirely outside
# any git repo. The rig's OWN live write path (`DEFAULT_CALIBRATION_
# PACKAGE_ROOT` above, and `DEFAULT_PACKAGE_ROOT` in capture_daemon.py)
# is deliberately OUT OF SCOPE for this move -- this is about the
# dev-machine side only.
#
# That script pulls the rig's live `data/calibration_packages/` into a
# staging dir, filters it down to packages actually referenced by the
# pulled throws (`cleanup_orphaned_calibration_packages()`), then moves
# what's kept into this flat, shared directory (package ids are globally
# unique timestamps, so merging pulls across many days can never
# collide). A throw package's own `calibration_package_id` doesn't
# record which of these two roots its calibration package ended up in --
# a fresh live throw's package is still under the live root, while an
# archived/pulled throw's package has already been moved here -- so
# `find_calibration_package_dir()` below checks both, live root first.
DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT = Path.home() / "Projects" / "data" / "opendarts" / "calibrations"

CALIBRATION_PACKAGE_SCHEMA = "calibration-package-v1"
CALIBRATION_PACKAGE_DERIVED_SCHEMA = "calibration-package-derived-v1"

# Confirmed byte-exact against both synthetic frames and real corpus
# camera frames (this module's own tests) -- see module docstring.
# cv2's BGR uint8 arrays go straight into cv2.VideoWriter's FFV1 encoder
# (fourcc RAW_VIDEO_CODEC.upper()) and back out of cv2.VideoCapture with
# no colorspace conversion anywhere in the round trip. `bgr24`/pix_fmt is
# recorded in meta.json for provenance (the on-disk layout), no longer a
# pipe argument.
RAW_VIDEO_CODEC = "ffv1"
RAW_VIDEO_PIX_FMT = "bgr24"
RAW_VIDEO_FILENAME_TEMPLATE = "cam{cam}_raw.mkv"
DERIVED_CALIBRATION_FILENAME = "derived_calibration.json"
META_FILENAME = "meta.json"

# Arbitrary container-level framerate -- every frame is an independent,
# separately-captured calibration-burst sample (see
# bootstrap_calibrations()'s own "genuinely independent" framing in
# opendarts.live.capture_daemon), not real motion video; the container
# needs SOME framerate to be valid, and nothing downstream (encode,
# decode, or the byte-exactness check) depends on what it actually is.
RAW_VIDEO_CONTAINER_FPS = 5


@functools.lru_cache(maxsize=1)
def _code_version() -> str | None:
    """Best-effort full `git rev-parse HEAD` (the complete 40-char SHA)
    of THIS repo checkout, or `None` if git isn't available/this isn't a
    git checkout/anything else goes wrong -- the v2 package schema's
    gap #1 (2026-08-27): "no code version anywhere... a replay next month
    runs different code against the same video and quietly returns
    different numbers, with nothing in the record to explain why." Every
    `solved_by` field in a calibration package (the per-camera pose
    solve, plus the `ring_boundary_offset`/`board_color_calibration`
    blocks) names a MODULE, never a commit -- this answers "what code
    state actually produced this" the way `solved_by` alone cannot.

    FULL SHA, not short -- a real follow-up fix (2026-08-27, same day):
    the first version of this function ran `git rev-parse --short HEAD`
    (7 chars). A short SHA can collide as history grows, and cannot be
    meaningfully compared against a full 40-char SHA. Switched to the
    full, unabbreviated SHA.

    Deliberately degrades to `None`, never raises -- same posture as
    every other non-essential provenance field in this module (a missing
    code version must never cost a calibration package its derived
    values). Cached for the lifetime of one process (`lru_cache`) -- the
    running code's own commit cannot change mid-process, so there is no
    reason to re-shell-out on every single calibration event; a fresh
    process (a real deploy) gets a fresh cache. Confirmed runnable from
    this module's real live write path: the rig runs this project from a
    real deployed git checkout (its run loop pulls before relaunching),
    not a detached/sandboxed
    environment -- `git` and a `.git` directory are always present there,
    unlike e.g. a packaged/frozen distribution that might ship without
    either."""
    if not capabilities.has("git"):
        # Capability gate (2026-09-12): a rig with no git (the Windows
        # PC) cannot answer this, and the answer cannot change
        # mid-process -- skip the doomed spawn instead of running it to
        # learn what the probe already knows. lru_cache above makes
        # either path at-most-once anyway.
        return None
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0:
            return None
        sha = proc.stdout.strip()
        return sha or None
    except Exception: # noqa: BLE001 -- never fail a calibration save over this
        return None


def new_calibration_package_id(
    now: datetime | None = None, *, entropy: str | None = None
) -> str:
    """Real timestamp-based package id, matching this project's existing
    session-id convention exactly (`session_id =
    time.strftime("%Y%m%d-%H%M%S")` in
    opendarts.live.capture_daemon.run_capture_loop_body()) -- `calib_`
    prefixed so a package id is never mistaken for a throw-package
    session id even though both share the same timestamp shape. UTC
    (unlike the local-time `time.strftime()` session ids elsewhere in
    this project) -- deliberate: this id also has to sort/compare
    correctly across whatever timezone the rig's own clock happens to be
    set to, and a calibration package is never displayed to an operator
    as a human-read clock time the way a session id sometimes is.

    **Collision-resistant suffix, added 2026-08-20 after a verifier pass
    found a real bug it fixes**: the timestamp alone only has 1-second
    resolution, and nothing serializes overlapping calibration events --
    a fast double-click on "Refresh calibration now" (a real, named
    scenario, not hypothetical) can trigger two independent
    `bootstrap_calibrations()` calls inside the same UTC second, which
    previously meant two genuinely different calibration events (
    different camera captures, different solved poses) could both
    resolve to the IDENTICAL `package_id` and both try to write into the
    SAME on-disk directory -- a real, demonstrated risk of one event's
    `derived_calibration.json` silently pairing with the OTHER event's
    raw video, exactly the "package that looks complete but doesn't
    reflect one real capture" failure docs/DESIGN.md's "Replay is the
    source of truth" calls out as unacceptable. `entropy` (default `None`) appends
    `secrets.token_hex(4)` (8 hex chars, ~4 billion possibilities --
    astronomically unlikely to collide even across a burst of rapid
    real double-clicks) so two concurrent calls essentially never
    resolve to the same id in the first place, regardless of clock
    resolution. Tests that need a deterministic, literal id pass their
    own fixed `entropy=` string instead of relying on the random default.
    """
    now = now or datetime.now(timezone.utc)
    suffix = entropy if entropy is not None else secrets.token_hex(4)
    return f"calib_{now.strftime('%Y%m%d-%H%M%S')}-{suffix}"


def _encode_raw_video(frames: list[np.ndarray], out_path: Path) -> None:
    """Encode `frames` (BGR uint8 arrays, all the same shape) into a
    single lossless FFV1 .mkv at `out_path` using `cv2.VideoWriter` --
    cv2's own BGR arrays go straight in, no PNG round trip, no external
    `ffmpeg` binary (the FFV1 encoder is bundled in the opencv wheel on
    every platform). Raises `ValueError` for a frame-shape mismatch or an
    empty `frames` list, `RuntimeError` if the FFV1 writer will not open
    (an opencv build without the ffmpeg videoio backend) or the written
    file does not decode back byte-identically. Every raise here is
    caught by `save_calibration_package()`'s own per-camera try/except --
    this function has no opinion on graceful degradation, only on doing
    the encode losslessly or failing loudly.

    Byte-exactness is the whole premise (see the module docstring), so it
    is VERIFIED here, not trusted: after writing, the file is decoded and
    compared frame-for-frame, and any difference raises rather than
    keeping a lossy record. Calibration is infrequent, so the extra
    decode is cheap insurance against a platform/wheel that ever silently
    used a lossy pixel format."""
    if not frames:
        raise ValueError("_encode_raw_video: no frames to encode")
    h, w = frames[0].shape[:2]
    for f in frames:
        if f.shape[:2] != (h, w):
            raise ValueError(
                f"_encode_raw_video: frame shape mismatch ({f.shape[:2]} != {(h, w)}) "
                "-- every frame for one camera in one calibration burst must be the "
                "same resolution"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*RAW_VIDEO_CODEC.upper()),
        RAW_VIDEO_CONTAINER_FPS,
        (w, h),
    )
    if not writer.isOpened():
        out_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"_encode_raw_video: cv2.VideoWriter could not open a lossless FFV1 "
            f"writer for {out_path} -- this opencv build lacks the ffmpeg videoio "
            "backend (opencv-python/-headless normally bundles it on every platform)."
        )
    try:
        for f in frames:
            writer.write(np.ascontiguousarray(f, dtype=np.uint8))
    finally:
        writer.release()
    # Byte-exactness is the premise: verify the round trip rather than
    # trust the codec (see this function's docstring).
    decoded = _decode_raw_video(
        out_path, n_frames=len(frames), frame_width=w, frame_height=h
    )
    for i, (original, back) in enumerate(zip(frames, decoded)):
        if not np.array_equal(np.ascontiguousarray(original, dtype=np.uint8), back):
            out_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"_encode_raw_video: FFV1 round trip was not byte-exact at frame "
                f"{i} for {out_path} -- refusing to keep a lossy calibration record"
            )


def _decode_raw_video(
    path: Path, *, n_frames: int, frame_width: int, frame_height: int
) -> list[np.ndarray]:
    """Inverse of `_encode_raw_video()` -- decode `path` back to a list
    of `n_frames` BGR uint8 arrays of shape `(frame_height, frame_width,
    3)` via `cv2.VideoCapture` (the ffmpeg backend bundled in the opencv
    wheel, no external binary). `n_frames`/`frame_width`/`frame_height`
    come from the package's own `meta.json`; a stale/wrong meta.json is
    caught here as a `ValueError` (wrong frame count or shape) rather
    than returning silently-wrong frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(
            f"_decode_raw_video: cv2.VideoCapture could not open {path}"
        )
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) != n_frames:
        raise ValueError(
            f"_decode_raw_video: decoded {len(frames)} frames from {path}, "
            f"expected {n_frames} -- meta.json may not match this file's real "
            "contents"
        )
    for frame in frames:
        if frame.shape != (frame_height, frame_width, 3):
            raise ValueError(
                f"_decode_raw_video: decoded frame shape {frame.shape} != "
                f"{(frame_height, frame_width, 3)} from {path} -- meta.json may not "
                "match this file's real contents"
            )
    # cv2.VideoCapture.read() returns a fresh array per call, so each
    # frame is independently owned (a caller mutating one decoded frame
    # can never affect another) -- matching the previous contract.
    return [np.ascontiguousarray(f, dtype=np.uint8) for f in frames]


@dataclass
class LoadedCalibrationPackage:
    package_id: str
    package_dir: Path
    calibrations: dict[int, CameraCalibration]
    diagnostics: dict[int, dict[str, Any]]
    # None per-camera when that camera's raw video failed to encode
    # (see save_calibration_package()'s own per-camera try/except) or
    # this load intentionally skipped decoding (see
    # load_calibration_package()'s `decode_raw_frames` parameter).
    raw_frames: dict[int, list[np.ndarray] | None]
    created_at_utc: str
    # THE V2 PACKAGE SCHEMA (2026-08-27) -- this calibration
    # event's own LIVE-DERIVED ring-boundary-offset / board-color-
    # threshold full nested payloads
    # (`LoadedCalibrationPackage.ring_boundary_offset` /
    # `board_color_calibration`). The same JSON shape their
    # respective `result_to_payload()` functions produce, or `None` when
    # this package predates the field, the measurement never ran this
    # event, or its stored `schema` no longer matches the current one
    # (see `load_calibration_package()` below). NOT confidence-gated
    # here -- a rejected measurement is still real, reportable data;
    # gating happens wherever a caller decides whether to actually apply
    # it -- the documented convention for these two fields.
    ring_boundary_offset: dict[str, Any] | None = None
    board_color_calibration: dict[str, Any] | None = None


def build_derived_calibration_payload(
    calibrations: dict[int, CameraCalibration],
    diagnostics: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the `derived_calibration.json` payload (package-wide timing,
    `code_version`, per-camera solved calibration + diagnostics, and the
    two nested `ring_boundary_offset`/`board_color_calibration` blocks) --
    factored out of `save_calibration_package()` on 2026-08-27 (calibration
    REPLAY-SOLVE task) so `dev.calibration.calibration_replay.
    replay_calibration_package()` can produce a byte-for-byte-equivalent
    payload from a REPLAYED `calibrations`/`diagnostics` pair without
    duplicating this whitelist/nested-payload logic a second time (a
    second copy would be exactly the kind of two-implementations-of-one-
    on-disk-shape risk this project's own v2 package-schema work has
    spent several rounds eliminating elsewhere -- see docs/DESIGN.md). Pure --
    takes no `package_dir`/`package_id`/raw-frame arguments at all,
    unlike `save_calibration_package()` itself, and never touches disk;
    `save_calibration_package()` below is now a thin wrapper that calls
    this function and writes the result to
    `<package_dir>/derived_calibration.json`, with IDENTICAL output to
    before this refactor (verified by this module's own existing tests,
    which assert on-disk `derived_calibration.json` content and were
    unchanged by this extraction).

    `calibrations`/`diagnostics`: exactly what `bootstrap_calibrations()`
    itself produces -- see `save_calibration_package()`'s own docstring
    for the authoritative shape of both. `diagnostics=None` (the default)
    is treated identically to `{}` -- every per-camera/package-wide field
    below degrades to `None`/absent rather than raising, so a caller with
    no diagnostics collected still gets a valid, if diagnostics-empty,
    payload (matching `save_calibration_package()`'s own long-standing
    contract, unchanged by this refactor).
    """
    diagnostics = diagnostics or {}
    # TIMING. bootstrap_calibrations() puts
    # its own total duration on EVERY camera's diagnostics entry
    # (per_cam_diagnostics is strictly per-camera, no separate top-level
    # slot there) -- pulled out here into this package's own top-level
    # `total_duration_s` instead of staying duplicated per camera, since
    # at the package level there IS a natural top-level home for a
    # package-wide fact. `next(..., None)` rather than indexing
    # `calibrations` cameras directly: this must degrade to None (not
    # raise) for a caller that passed diagnostics=None/{} (e.g. a direct
    # test of this module, per this function's own docstring), same
    # "diagnostics-empty package still writes validly" posture as the
    # per-camera fields below.
    total_duration_s = next(
        (d["calibration_total_duration_s"] for d in diagnostics.values()
         if "calibration_total_duration_s" in d),
        None,
    )
    # SECTION TIMING. Same
    # "pull the one-shared-value up to the package's top level rather
    # than leaving it duplicated on every camera's entry" treatment as
    # total_duration_s above -- capture/motion-thresholds/ring-boundary/
    # board-color each run once for the whole calibration event, not
    # per camera. detect_duration_s/solve_duration_s stay per-camera
    # (genuinely different work per camera), whitelisted below like
    # everything else.
    section_timing = {
        key: next((d[key] for d in diagnostics.values() if key in d), None)
        for key in (
            "capture_duration_s", "motion_threshold_duration_s",
            "ring_boundary_offset_duration_s", "board_color_duration_s",
        )
    }
    # REPLAY PERSISTENCE (docs/DESIGN.md's "Replay is the source of truth"), 2026-08-22 -- the
    # actual measured ring-boundary-offset/board-color VALUES, not just
    # timing. Same "package-wide, duplicated onto every camera's
    # diagnostics entry, pulled up to this package's own top level"
    # treatment as `section_timing` above -- these values describe the
    # whole calibration event, not any one camera. Closes a real,
    # confirmed REPLAY gap: `bootstrap_calibrations()` was applying
    # these as in-memory global state (`set_ring_boundary_offsets()` /
    # `set_board_color_thresholds()`) for LIVE scoring, but never
    # persisting them anywhere a later, fresh-process replay could find
    # them -- so replaying an old package silently fell back to the
    # regulation-default constants instead of reproducing what actually
    # scored the throw live (confirmed on two real throws, an S12 and an
    # S1 throw from one recorded session, which scored `single_inner`
    # live but `treble` on replay before this fix). `None`/`False` means
    # "not accepted this event" (rejected or never measured) -- never a
    # fabricated value; `load_throw_package()` treats an absent/None
    # field here exactly like a package with no calibration-package-level
    # measurement at all and falls back to the existing session-level
    # lookup, then to the regulation-default constants.
    #
    # FULL-BOUNDARY STORAGE GAP FIX, 2026-08-26. Before
    # this date, the 9 keys below were the WHOLE of what a live
    # calibration package ever recorded about the ring-boundary-offset
    # measurement: only the two INNER boundaries' offset_mm/confidence/
    # n_samples_used (3 fields x 2 boundaries), never treble_outer/
    # double_outer at all, and never any boundary's mad_mm,
    # n_samples_rejected, n_angles_attempted, or per-camera radius/
    # n_profiles/mode breakdown -- `bootstrap_calibrations()` computed
    # all of that every single calibration event and then threw it away
    # when the function returned. This was a real, confirmed gap: a QA
    # pass running this project's OWN `ring_boundary_offset.py` against
    # a second implementation's stored background frames found the two
    # localizers disagree by up to 0.76mm on the SAME pixels (same solved
    # extrinsics, same schema, same shared constants) -- a real,
    # live-scoring-affecting divergence (flipped a real dart's score,
    # `double` vs `single_outer`, on session `20260826-1229xx` throw 43)
    # -- and opendarts could not self-diagnose it from its own package data,
    # because the very fields that would show WHERE the two localizers
    # actually differ (treble_outer/double_outer, mad_mm, per_camera)
    # were never stored anywhere. Fixed additively (every key below is
    # unchanged, in name and meaning -- `opendarts.capture.throw_package.
    # load_throw_package()`'s REPLAY lookup and this module's own
    # existing tests read them exactly as before) by ALSO carrying
    # `ring_boundary_measurement`: all 4 boundaries' full data (via
    # `opendarts.calibration.ring_boundary_offset.
    # boundary_measurement_to_payload()`, the SAME field vocabulary the
    # offline session-level `ring_boundary_offset.json` already uses, so
    # the two representations don't invent two different vocabularies
    # for the same measurement), or None if the measurement itself never
    # ran/raised this event. Deliberately does NOT include
    # `median_profiles` (the raw per-angle derivation records) -- those
    # already have a durable home in the offline session-level file for
    # deep replay; this per-event package captures the aggregate
    # per-boundary/per-camera numbers a live diagnostic actually needs,
    # without duplicating potentially large raw arrays into every single
    # calibration package.
    # THE V2 PACKAGE SCHEMA (2026-08-27). `derived_calibration.json`
    # used to be 19 flat top-level scalars; the canonical shape nests
    # into `ring_boundary_offset`/
    # `board_color_calibration` sub-objects, each with its OWN `schema`
    # and a `solved_by` field, plus per-camera `focal_length_px`/
    # `focal_length_source`/`orientation_hint_deg`/`orientation_hint_
    # source`/`n_raw_extra_frames_used`.
    #
    # REPLACE FOR NEW WRITES, WITH A READER SHIM FOR OLD FILES -- a real
    # mid-task correction, not the original plan. First draft of this
    # fix went ADDITIVE (kept every flat key below, alongside the new
    # nested blocks) reasoning that REPLAY only requires OLD on-disk
    # files to stay loadable, not that NEW files match any particular
    # shape. That reasoning was incomplete: verified against a real
    # parity check (an offline tool, run for real against a
    # freshly-regenerated package vs a real recorded calibration
    # package, not assumed) that the additive
    # version left 105 opendarts-only key paths that are PURE DUPLICATES of
    # data the new nested blocks already carry (`treble_inner_offset_mm`
    # duplicates `ring_boundary_offset.boundaries.treble_inner.offset_mm`,
    # etc.) -- and the project's own standing instruction for this round is
    # explicit: the calibration-package bar is "the SAME standard as
    # throw packages -- identical JSON field names, identical structure",
    # which the throw-package
    # rounds satisfied by actually CONFORMING shapes (the `sector`
    # int->str fix, the `Zeus`->`Zeus` reason-string translation, etc.),
    # not merely by adding alongside. Every one of those 105 opendarts-only
    # duplicate paths is now DROPPED from what a NEW save writes -- the
    # nested `ring_boundary_offset`/`board_color_calibration` blocks are
    # the sole source of this data going forward -- the canonical shape
    # was always nested, and matching that OUTPUT is what parity
    # requires.
    #
    # `ring_boundary_offset_accepted` and the timing fields
    # (`total_duration_s`/`section_timing`) are the ONE exception, kept
    # flat -- these are opendarts-only additions, a real opendarts-only
    # diff that is expected and accepted, exactly like
    # `meta.throw_number`/`frame_cameras`/`camera_mode` each were for one
    # round. Re-running the parity checker
    # against this exact package pair after this fix (see this module's
    # own tests below for the concrete
    # numbers): `derived_calibration.json` moved from 105 opendarts-only
    # PURE-DUPLICATE paths down to ONLY this legitimate, spec-sanctioned
    # residual (timing fields, `ring_boundary_offset_accepted`,
    # `code_version`, `n_frames_raw_pool`, `frame_indices_used` -- the
    # last three genuinely NEW this round on both sides) -- zero
    # other-side-only paths, zero type mismatches, at
    # every level of nesting.
    #
    # FOLLOW-UP ROUND (2026-08-27, same day, a fresh QA pull of a
    # recorded calibration package against the real parity checker):
    # `meta.json` reached 23/23 identical, `derived_calibration.json`
    # dropped from 105 opendarts-only paths to 27 -- but 4 real gaps
    # remained, all fixed in this same follow-up: (1) rename
    # `frames_used_indices` -> `frame_indices_used` (a pure naming
    # fix -- not a new
    # concept, see `_frame_indices_used`'s own rename in
    # `opendarts.live.capture_daemon`); (2) add `raw_extra_frame_indices`
    # (the second index list -- the raw pool's complement of
    # `frame_indices_used`, i.e. every raw-pool frame that did NOT feed
    # the accepted solve); (3) 7 real, ALREADY-COMPUTED-BUT-NEVER-
    # PERSISTED k1/cx/distortion provenance fields (`k1`,
    # `distortion_source`, `cx_px`, `cx_focal_length_px`, `cx_k1`,
    # `joint_focal_length_px`, `principal_point_source`) -- the exact
    # same class of gap `focal_length_px`/`orientation_hint_deg` closed
    # in the section-2f round above, just for the k1/cx distortion work
    # (2026-08-26) instead: `bootstrap_calibrations()`'s own
    # `per_cam_diagnostics` has computed every one of these under these
    # exact key names since that work landed, this whitelist just never
    # let them through; (4) `_code_version()` switched from `git
    # rev-parse --short HEAD` to the full `git rev-parse HEAD` (40 chars)
    # -- a short SHA can collide as history
    # grows and a reader can't meaningfully compare a 7-char and a
    # 40-char value.
    #
    # OLD on-disk files (written before this reshape) are UNCHANGED and
    # remain loadable exactly as before -- this is a WRITER-side change
    # only. `opendarts.capture.throw_package.load_throw_package()`'s REPLAY
    # lookup was updated (see that module's own dated comment) to read
    # EITHER shape via key-presence dispatch (the same "resolve via key
    # presence, not schema string" discipline this project already
    # established for `ad_ground_truth.captured_at_utc`'s own interim-
    # format window, 2026-08-27's earlier v2 package-schema round) --
    # an old flat-shaped package still applies its live-measured offset
    # exactly as before; a new nested-shaped package now does too,
    # instead of silently falling through to the session-level lookup
    # (which is what would have happened had the reader been left
    # unchanged after this writer change -- a real regression this task
    # caught and fixed, not shipped).
    #
    # `code_version` -- gap #1 of the three the v2 package schema named
    # ("no code version anywhere... a replay next month runs
    # different code... with nothing in the record to explain why").
    # Computed ONCE per save call (see `_code_version()`'s own
    # docstring for the full reasoning: best-effort full `git rev-parse
    # HEAD` -- the complete 40-char SHA, not the short form --
    # `None` on any failure, never raises) and stamped
    # BOTH at this file's own top level (answers "what code produced
    # this package as a whole," the simplest single home for it) AND
    # inside each of the two nested "solved block" payloads below
    # (`ring_boundary_offset["code_version"]`/
    # `board_color_calibration["code_version"]`) -- those two blocks
    # already carry their own `solved_by` (which module), so pairing a
    # code version at the same locality answers "which module, which
    # commit" together rather than forcing a reader back up to the
    # top-level field to learn what version solved a SPECIFIC block. Not
    # also duplicated onto every per-camera `cameras.N` entry (the third
    # real "solved block," the PnP pose solve) -- a per-camera dict has
    # no existing `solved_by`-shaped slot to pair it with, and the
    # top-level field already answers the same question for the whole
    # package (every solve in one package always ran under the same
    # code state, since it's one calibration event), so a third
    # identical copy would be pure duplication with no new information.
    code_version = _code_version()

    # The ONE flat legacy key kept (see the "REPLACE FOR NEW WRITES"
    # comment above for why this one, specifically, stays flat) --
    # "was the measured offset actually APPLIED to live scoring this
    # event, or measured-then-rejected" is a fact about the EVENT, not
    # solely recoverable from `ring_boundary_offset.boundaries.*` alone
    # (a boundary can have a real `offset_mm` yet still have been
    # rejected, e.g. one inner boundary succeeded while the other
    # didn't -- see `bootstrap_calibrations()`'s own joint accept/reject
    # gate). Same pull-up mechanism as every other package-wide value.
    ring_boundary_offset_accepted = next(
        (d["ring_boundary_offset_accepted"] for d in diagnostics.values()
         if "ring_boundary_offset_accepted" in d),
        None,
    )

    ring_boundary_offset_nested = next(
        (d["ring_boundary_offset_payload"] for d in diagnostics.values()
         if d.get("ring_boundary_offset_payload") is not None),
        None,
    )
    if ring_boundary_offset_nested is not None:
        ring_boundary_offset_nested = {
            **ring_boundary_offset_nested, "code_version": code_version,
        }
    board_color_calibration_nested = next(
        (d["board_color_calibration_payload"] for d in diagnostics.values()
         if d.get("board_color_calibration_payload") is not None),
        None,
    )
    if board_color_calibration_nested is not None:
        board_color_calibration_nested = {
            **board_color_calibration_nested, "code_version": code_version,
        }

    derived = {
        "schema": CALIBRATION_PACKAGE_DERIVED_SCHEMA,
        "code_version": code_version,
        "total_duration_s": total_duration_s,
        **section_timing,
        "ring_boundary_offset_accepted": ring_boundary_offset_accepted,
        "cameras": {
            str(cam): {
                **calibration_to_dict(calib),
                **{
                    k: v
                    for k, v in diagnostics.get(cam, {}).items()
                    if k in (
                        "reprojection_error_px", "n_frames_used", "target_met",
                        "calibration_duration_s", "detect_duration_s", "solve_duration_s",
                        # DECOUPLED CAPTURE-VS-DETECT TARGET, 2026-08-21
                        # (opendarts.live.capture_daemon.CALIBRATION_N_FRAMES_
                        # DETECT) -- n_frames_used above stays "frames
                        # actually detected"; n_frames_raw_pool is the
                        # (usually bigger) total raw pool size this
                        # camera's ring-boundary-offset/board-color
                        # derivations actually ran against. Whitelisted
                        # here for the same reason every other diagnostics
                        # key is -- an unlisted key is silently dropped,
                        # not an error (see this module's own real
                        # gotcha, hit twice before this file's own
                        # history).
                        "n_frames_raw_pool",
                        # THE V2 PACKAGE SCHEMA (2026-08-27) --
                        # `focal_length_px`/`focal_length_source`/
                        # `orientation_hint_deg`/`orientation_hint_
                        # source`/`n_raw_extra_frames_used` (see
                        # the bootstrap's own
                        # `_DIAGNOSTIC_KEYS`). All five are ALREADY
                        # computed by `bootstrap_calibrations()`'s own
                        # `per_cam_diagnostics` -- the first four have
                        # been there since the focal-length/orientation-
                        # hint work landed, `n_raw_extra_frames_used` is
                        # new this round (see that function's own
                        # comment) -- this whitelist simply stopped
                        # silently dropping them.
                        "focal_length_px", "focal_length_source",
                        "orientation_hint_deg", "orientation_hint_source",
                        "n_raw_extra_frames_used",
                        # WHICH raw-pool frames actually fed this
                        # camera's accepted solve (see
                        # `bootstrap_calibrations()`'s own
                        # "FRAME-SELECTION PROVENANCE" comment) --
                        # the v2 package schema's gap #2. `None` when the
                        # invariant it's computed from didn't hold this
                        # event (logged loudly there, never silently
                        # wrong here). Named `frame_indices_used` (NOT
                        # `frames_used_indices`, its original name until
                        # 2026-08-27's follow-up round) -- a pure
                        # rename, the exact class of naming collision
                        # this parity effort exists to eliminate.
                        "frame_indices_used",
                        # The second index list, added in the same
                        # follow-up round: the raw pool's own COMPLEMENT
                        # of `frame_indices_used` -- every raw-pool frame
                        # that did NOT feed the accepted solve, whether
                        # it was never detected at all (a raw-only
                        # extra) or detected-but-rejected. Same
                        # invariant/degrade-to-None posture as
                        # `frame_indices_used` itself -- see
                        # `bootstrap_calibrations()`'s own computation
                        # immediately alongside `frame_indices_used`'s.
                        "raw_extra_frame_indices",
                        # LIVE-DERIVED DISTORTION/PRINCIPAL-POINT
                        # PROVENANCE (2026-08-27 follow-up round) -- 7
                        # fields opendarts was already computing
                        # (in `bootstrap_calibrations()`'s own
                        # `per_cam_diagnostics`, since the 2026-08-26
                        # k1/cx work landed) but never whitelisting into
                        # a saved package, the exact same "computed but
                        # discarded" gap `focal_length_px`/
                        # `orientation_hint_deg` closed in the section-2f
                        # round above. `k1`/`distortion_source` are the
                        # k1-only tier's own resolved values (k1=0.0,
                        # never None, when this event's data didn't
                        # safely support a fit); `cx_px`/
                        # `cx_focal_length_px`/`cx_k1`/
                        # `principal_point_source` are the further +cx
                        # tier's own resolved values (None when that
                        # tier never succeeded this event);
                        # `joint_focal_length_px` is the SELF-CONSISTENT
                        # joint-solve focal length paired with `k1`
                        # (distinct from `focal_length_px` above, which
                        # always reports the homography-only tier's own
                        # value regardless of whether distortion also
                        # succeeded -- see `opendarts.calibration.
                        # distortion`'s own module docstring). See
                        # `opendarts.live.capture_daemon`'s own
                        # `per_cam_diagnostics` construction for the
                        # authoritative source of every one of these 7
                        # key names -- this whitelist simply stopped
                        # silently dropping them.
                        "k1", "distortion_source",
                        "cx_px", "cx_focal_length_px", "cx_k1",
                        "joint_focal_length_px", "principal_point_source",
                    )
                },
            }
            for cam, calib in calibrations.items()
        },
    }
    if ring_boundary_offset_nested is not None:
        derived["ring_boundary_offset"] = ring_boundary_offset_nested
    if board_color_calibration_nested is not None:
        derived["board_color_calibration"] = board_color_calibration_nested
    return derived


def save_calibration_package(
    dest_root: Path,
    package_id: str,
    raw_frames_by_cam: dict[int, list[np.ndarray]],
    calibrations: dict[int, CameraCalibration],
    diagnostics: dict[int, dict[str, Any]] | None = None,
) -> Path:
    """Write one complete calibration package to `dest_root / package_id`.

    `raw_frames_by_cam`: EVERY raw frame `bootstrap_calibrations()`
    actually captured for that camera this calibration event, across
    every retry round (not just the final accepted round's own batch) --
    UPDATED 2026-08-21 (verifier finding Bug 2): this used to mean only
    the subset that went through real landmark detection
    (`_detect_batch()`'s own `raw_frames_accum` accumulator, now
    removed); as of `bootstrap_calibrations()`'s own DECOUPLED
    CAPTURE-VS-DETECT TARGET change (same day), that subset is smaller
    than the full raw pool the live ring-boundary-offset/board-color
    derivations actually consume, so the caller now derives this dict
    from `pre_orientation_pool` instead -- see that function's own
    `package_raw_frames` construction and its comment. Every frame here
    has also been through the same white-balance normalization
    (`oriented_landmarks.normalise_illuminant()`) real detected frames
    get, so "what the calibration pipeline actually analyzed" still
    holds, just for the FULL raw pool rather than only its detected
    subset. Naturally capped at `CALIBRATION_MAX_N_FRAMES` per camera by
    construction (the retry loop stops accumulating once a camera hits
    that cap or its target -- see that module's own docstring). A camera
    absent from this dict, or present with an empty list, is skipped (no
    raw video written, `meta.json`'s own per-camera
    entry says so honestly) -- not an error, since a camera that never
    captured usable frames this event is a real, expected case.

    `calibrations`/`diagnostics`: exactly what `bootstrap_calibrations()`
    itself produces and already exposes today -- `calibrations` is its
    return value, `diagnostics` is what it fills into its own
    `diagnostics_out` parameter (`{cam: {"reprojection_error_px":
    float|None, "n_frames_used": int, "target_met": bool,
    "calibration_duration_s": float, "calibration_total_duration_s":
    float}}`, plus the orientation-hint fields -- see that function's
    own `per_cam_diagnostics` construction for the authoritative shape).
    `None` (the default) writes every camera's diagnostics as `{}` -- a
    package saved by a caller that didn't collect diagnostics (e.g. a
    direct test of this module) still writes valid, if
    diagnostics-empty, derived_calibration.json. `calibration_total_
    duration_s` (identical on every camera's entry, since
    `per_cam_diagnostics` has no separate top-level slot) is pulled out
    into this package's own top-level `total_duration_s` instead of
    staying duplicated per camera below.

    Fails loudly (raises) on a directory-level problem (e.g. `dest_root`
    not writable) -- matches `opendarts.capture.throw_package.
    save_throw_package()`'s own "an incomplete package that LOOKS
    complete is worse than no package" posture for the JSON files this
    writes. Per-camera RAW VIDEO encode failures are the one exception,
    by design (see module docstring's "STORAGE FORMAT" section): each
    camera's `_encode_raw_video()` call is individually wrapped, a
    failure is logged and recorded as `"raw_video": null` in that
    camera's `meta.json` entry, and every OTHER camera's raw video and
    the whole package's JSON files are still written -- because the live
    caller (`bootstrap_calibrations()`'s background wiring) needs a raw
    video encode that goes wrong on one rig to degrade to "no raw frames
    for this calibration event, everything else fine" rather than losing
    the derived calibration record entirely. (FFV1 now encodes via
    cv2.VideoWriter, whose codec ships in the opencv wheel, so the old
    "ffmpeg missing" failure mode no longer exists on any platform.)
    """
    dest_root = Path(dest_root)
    package_dir = dest_root / package_id
    package_dir.mkdir(parents=True, exist_ok=True)

    # NOT normalized here (`diagnostics or {}`) -- `build_derived_
    # calibration_payload()` below does its own normalization of this
    # same parameter, so a second copy here would be dead code (2026-08-27
    # refactor, see that function's own docstring for why it was
    # extracted).
    # No capability gate any more: FFV1 encoding goes through
    # cv2.VideoWriter, whose codec ships inside the opencv wheel on every
    # platform (verified across the fleet 2026-09-20), so there is no
    # external tool that can be "missing" here. A genuine encode failure
    # is still handled per-camera by the try/except below -- degrade this
    # one camera's raw video to an honest null, never lose the derived
    # calibration or any other camera. (The Windows PC used to log a
    # doomed ffmpeg spawn per camera; that whole failure mode is gone.)
    cam_meta: dict[str, Any] = {}
    for cam, frames in sorted(raw_frames_by_cam.items()):
        if not frames:
            cam_meta[str(cam)] = {
                "raw_video": None, "storage": None, "n_frames": 0, "error": None,
            }
            continue
        h, w = frames[0].shape[:2]
        out_path = package_dir / RAW_VIDEO_FILENAME_TEMPLATE.format(cam=cam)
        try:
            _encode_raw_video(frames, out_path)
        except Exception as exc: # noqa: BLE001 -- per-camera degrade, see docstring
            log.warning(
                "calibration package %s: cam%d raw-video encode failed, this "
                "camera's calibration is still saved but has no replayable raw "
                "frames: %s",
                package_id, cam, exc,
            )
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            cam_meta[str(cam)] = {
                "raw_video": None,
                "storage": None,
                "n_frames": len(frames),
                "error": str(exc),
            }
            continue
        cam_meta[str(cam)] = {
            "raw_video": out_path.name,
            # `storage` (v2 schema, 2026-08-27) names how this camera's
            # raw frames are stored, so a reader could DISPATCH its decode
            # on it. In practice there is exactly one real storage kind --
            # FFV1 (`RAW_VIDEO_CODEC`) -- and the only other value ever
            # written is `None` (the no-frames and encode-failure branches
            # above); there is no lossy fallback (see the module
            # docstring). `load_calibration_package()` still decodes off
            # `raw_video` presence, not this field; `storage` is kept for
            # structural parity if a second kind is ever added.
            "storage": RAW_VIDEO_CODEC,
            "n_frames": len(frames),
            "frame_width": w,
            "frame_height": h,
            "error": None,
        }

    created_at_utc = datetime.now(timezone.utc).isoformat()
    meta = {
        "schema": CALIBRATION_PACKAGE_SCHEMA,
        "package_id": package_id,
        "created_at_utc": created_at_utc,
        "codec": RAW_VIDEO_CODEC,
        "pix_fmt": RAW_VIDEO_PIX_FMT,
        "cameras": cam_meta,
    }
    (package_dir / META_FILENAME).write_text(json.dumps(meta, indent=2))

    derived = build_derived_calibration_payload(calibrations, diagnostics)
    (package_dir / DERIVED_CALIBRATION_FILENAME).write_text(json.dumps(derived, indent=2))

    return package_dir


def load_calibration_package(
    package_dir: Path, *, decode_raw_frames: bool = True
) -> LoadedCalibrationPackage:
    """Load a calibration package back into memory -- the counterpart to
    `save_calibration_package()`, and the function a real replay of a
    calibration event (re-run today's `bootstrap_calibrations()`-equivalent
    landmark-detection + PnP solve against the SAME stored raw frames)
    would start from.

    `decode_raw_frames=False` skips the (comparatively expensive) FFV1
    decode and leaves every camera's `raw_frames` entry as `None` --
    for a caller that only wants the derived calibration values (e.g. a
    dashboard listing, or the cleanup function's own reference-scanning,
    neither of which needs raw pixels at all).
    """
    package_dir = Path(package_dir)
    meta = json.loads((package_dir / META_FILENAME).read_text())
    derived = json.loads((package_dir / DERIVED_CALIBRATION_FILENAME).read_text())

    calibrations: dict[int, CameraCalibration] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    for cam_str, entry in derived.get("cameras", {}).items():
        cam = int(cam_str)
        calibrations[cam] = calibration_from_dict(entry)
        diagnostics[cam] = {
            k: entry.get(k)
            for k in ("reprojection_error_px", "n_frames_used", "target_met")
            if k in entry
        }

    raw_frames: dict[int, list[np.ndarray] | None] = {}
    for cam_str, cam_entry in meta.get("cameras", {}).items():
        cam = int(cam_str)
        # NOT dispatching on `cam_entry.get("storage")` -- see
        # `save_calibration_package()`'s own comment on that field
        # (2026-08-27, a v2 package-schema follow-up): opendarts has
        # exactly one real raw-frame storage kind (FFV1), so `storage` is
        # written for structural parity but is a constant
        # here, not yet a real dispatch key -- `raw_video` presence is
        # still the one thing that actually decides "is there a video to
        # decode." If a second storage kind is ever added on this side,
        # THIS is the line that needs to become a real dispatch (`if
        # storage == STORAGE_FFV1: ... elif storage ==
        # STORAGE_PNG_FRAMES: ...`) -- not invented speculatively now.
        if not decode_raw_frames or not cam_entry.get("raw_video"):
            raw_frames[cam] = None
            continue
        video_path = package_dir / cam_entry["raw_video"]
        raw_frames[cam] = _decode_raw_video(
            video_path,
            n_frames=cam_entry["n_frames"],
            frame_width=cam_entry["frame_width"],
            frame_height=cam_entry["frame_height"],
        )

    # THE V2 PACKAGE SCHEMA (2026-08-27) -- schema-checked exactly
    # like the offline session-level siblings
    # (`load_session_ring_boundary_offset()`/
    # `load_session_board_color_calibration()`) -- a mismatch (this measurement
    # method having since moved on to a new schema) is treated as
    # "absent," never as a stale value silently trusted.
    ring_boundary_offset = derived.get("ring_boundary_offset")
    if (
        ring_boundary_offset is not None
        and ring_boundary_offset.get("schema") != _RING_BOUNDARY_OFFSET_SCHEMA
    ):
        ring_boundary_offset = None
    board_color_calibration = derived.get("board_color_calibration")
    if (
        board_color_calibration is not None
        and board_color_calibration.get("schema") != _BOARD_COLOR_CALIBRATION_SCHEMA
    ):
        board_color_calibration = None

    return LoadedCalibrationPackage(
        package_id=meta["package_id"],
        package_dir=package_dir,
        calibrations=calibrations,
        diagnostics=diagnostics,
        raw_frames=raw_frames,
        created_at_utc=meta["created_at_utc"],
        ring_boundary_offset=ring_boundary_offset,
        board_color_calibration=board_color_calibration,
    )


def find_calibration_package_dir(
    package_id: str,
    *,
    search_roots: tuple[Path, ...] | None = None,
) -> Path | None:
    """Locate an on-disk calibration package directory for `package_id`,
    trying the live root (`DEFAULT_CALIBRATION_PACKAGE_ROOT`) first, then
    the archived root (`DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT`) --
    the only two places a real calibration package can genuinely live
    (see those constants' own docstrings). Returns None, not an error,
    if neither has it -- an expected case (e.g. the package was orphan-
    cleaned by `cleanup_orphaned_calibration_packages()` after this
    throw's own `calibration_package_id` was recorded, or this throw
    predates calibration packages' `data/archive/` pull convention) --
    see `load_calibration_package_derived_values()`'s own caller,
    `opendarts.capture.throw_package.load_throw_package()`, for how a miss
    here degrades (falls back to the session-level lookup, same as
    always)."""
    roots = search_roots if search_roots is not None else (
        DEFAULT_CALIBRATION_PACKAGE_ROOT, DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT,
    )
    for root in roots:
        candidate = Path(root) / package_id
        if (candidate / DERIVED_CALIBRATION_FILENAME).exists():
            return candidate
    return None


def load_calibration_package_derived_values(
    package_id: str,
    *,
    search_roots: tuple[Path, ...] | None = None,
) -> dict[str, Any] | None:
    """Read a calibration package's `derived_calibration.json` back as a
    plain dict, for REPLAY (see `opendarts.capture.throw_package.
    load_throw_package()`'s own lookup, which reads the live-derived
    ring-boundary-offset/board-color fields `save_calibration_package()`
    now writes at this dict's top level -- see that function's own
    `ring_and_color_derived` construction). Returns None (never raises)
    if the package can't be found, or its `derived_calibration.json`
    fails to parse -- both treated exactly like "no calibration-
    package-level value" by the caller, which then falls back to the
    existing session-level lookup, same safe-fallback posture as every
    other REPLAY sibling-file loader in this codebase
    (`load_session_ring_boundary_offset()`/
    `load_session_board_color_calibration()`)."""
    pkg_dir = find_calibration_package_dir(package_id, search_roots=search_roots)
    if pkg_dir is None:
        return None
    try:
        return json.loads((pkg_dir / DERIVED_CALIBRATION_FILENAME).read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        log.exception(
            "calibration package %s: derived_calibration.json failed to "
            "read/parse at %s -- treating as absent (falls back to the "
            "session-level lookup, same as no calibration package at all)",
            package_id, pkg_dir,
        )
        return None


def save_calibration_package_background(
    dest_root: Path,
    package_id: str,
    raw_frames_by_cam: dict[int, list[np.ndarray]],
    calibrations: dict[int, CameraCalibration],
    diagnostics: dict[int, dict[str, Any]] | None = None,
    *,
    throw_package_root: Path | None = None,
) -> threading.Thread:
    """Fire-and-forget wrapper around `save_calibration_package()` (plus,
    when `throw_package_root` is given, a follow-up
    `cleanup_orphaned_calibration_packages()` pass) run on a daemon
    thread -- the real live-path entry point
    `opendarts.live.capture_daemon.bootstrap_calibrations()` uses, so that
    neither the (up to a few seconds, see module docstring's measured
    FFV1 timing) raw-frame encode nor the cleanup's own directory walk
    can ever add latency to, or fail, the actual calibration result a
    live caller is waiting on.

    Every exception from either step is caught and logged here -- never
    re-raised (there is no caller left to catch it; this runs on its own
    thread after the function that started it has already returned).
    Returns the `Thread` object (already started) purely so tests can
    `.join()` it deterministically instead of guessing a sleep duration;
    live callers have no reason to keep the reference.

    **CONCURRENT-SAVE RACE GUARD**: `package_id` is registered in the
    module-level `_IN_FLIGHT_PACKAGE_IDS` counter SYNCHRONOUSLY, before
    the background thread even starts (so there is no window where a
    second, overlapping call to this function could run a cleanup pass
    that doesn't yet know about this package) -- and un-registered again
    only in a `finally` covering both the save and the (optional)
    follow-up cleanup step, so this package id is protected for the
    entire time its raw video could still be mid-write, not just during
    the save call itself. See `cleanup_orphaned_calibration_packages()`'s
    own docstring for the other half of this guard.

    **Real, collision-resistant `package_id` is the primary defense.**
    Callers should generate `package_id` via `new_calibration_package_id()`
    (its default `entropy=None` behavior, not a fixed/reused string) --
    that function's own docstring explains why: a plain timestamp alone
    has only 1-second resolution and two genuinely different, overlapping
    calibration events (a fast double-click on "Refresh calibration now"
    is the real, named scenario) can otherwise share an identical id and
    target the same on-disk directory, which risks one event's
    `derived_calibration.json` silently pairing with a DIFFERENT event's
    raw video -- exactly the kind of REPLAY-integrity violation
    docs/DESIGN.md's "Replay is the source of truth" calls out as unacceptable, and something
    this in-flight guard alone cannot fully prevent (it protects a
    package from premature CLEANUP, it does not stop two concurrent
    writers from both targeting the same path if they were ever handed
    the same id). `_IN_FLIGHT_PACKAGE_IDS` is a `Counter`, not a plain
    set, specifically so that even the residual, now near-impossible,
    same-id collision case degrades safely (both registrations are
    tracked independently; the id stays protected until BOTH calls have
    finished) rather than one call's `finally` silently stripping
    protection out from under the other's still-running encode.

    **Known gap, not closed by this guard**: `_IN_FLIGHT_PACKAGE_IDS`
    lives in ONE Python process's memory. It provides zero protection
    against a separate OS process racing a live server's in-flight save
    -- e.g. this module's own standalone CLI (`python -m
    opendarts.capture.calibration_package cleanup`, see the bottom of this
    module) run by an operator/cron job while a live
    `bootstrap_calibrations()` background save is in flight in the real
    server process. A real fix for that would need a cross-process guard
    (e.g. a lockfile under `calibration_package_root`), which is bigger
    scope than this fix and has NOT been built -- an operator invoking
    that CLI manually should avoid doing so while a real live calibration
    event might be in progress, same honest caveat this module's other
    "needs real rig hardware to fully validate" gaps carry.
    """
    with _IN_FLIGHT_LOCK:
        _IN_FLIGHT_PACKAGE_IDS[package_id] += 1

    def _run() -> None:
        try:
            try:
                save_calibration_package(
                    dest_root, package_id, raw_frames_by_cam, calibrations, diagnostics
                )
            except Exception: # noqa: BLE001 -- see docstring, never breaks the live caller
                log.exception(
                    "calibration package %s: background save failed -- the live "
                    "calibration result itself is unaffected, this only means no "
                    "replayable package exists for this event",
                    package_id,
                )
                return
            if throw_package_root is not None:
                try:
                    cleanup_orphaned_calibration_packages(
                        dest_root, throw_package_root, active_package_id=package_id
                    )
                except Exception: # noqa: BLE001 -- same never-break-the-caller rule
                    log.exception(
                        "calibration package %s: post-save cleanup pass failed "
                        "(non-fatal -- orphaned packages may accumulate until the "
                        "next successful cleanup)",
                        package_id,
                    )
        finally:
            # Only now -- after THIS call's own save AND any cleanup it
            # triggered have both finished -- does THIS call's own
            # registration get released. Decrement, not a flat removal --
            # see module-level _IN_FLIGHT_PACKAGE_IDS's own comment: if a
            # second, still-running call happens to share this exact
            # `package_id` (should be near-impossible given
            # new_calibration_package_id()'s entropy, but this is the
            # defense-in-depth path for it), decrementing leaves that
            # OTHER call's own registration intact instead of a flat
            # `discard()` wiping protection out from under it. Only once
            # the count reaches zero (no call anywhere still holds this
            # id in flight) is it safe for a cleanup pass to consider
            # this id orphan-eligible again (it won't actually be
            # orphaned: a real package on disk is still kept via
            # active_package_id/throw-reference the normal way once it's
            # no longer in-flight).
            with _IN_FLIGHT_LOCK:
                _IN_FLIGHT_PACKAGE_IDS[package_id] -= 1
                if _IN_FLIGHT_PACKAGE_IDS[package_id] <= 0:
                    del _IN_FLIGHT_PACKAGE_IDS[package_id]

    thread = threading.Thread(
        target=_run, name="calib-pkg-save", daemon=True
    )
    thread.start()
    return thread


def find_referenced_calibration_package_ids(package_root: Path) -> set[str]:
    """Every `calibration_package_id` referenced by any throw package
    still present under `package_root` (`opendarts.live.capture_daemon.
    DEFAULT_PACKAGE_ROOT` shape: `package_root/<session>/<throw>/
    meta.json`). A throw whose `meta.json` predates this field, or has
    no calibration_package_id for any other reason, simply contributes
    nothing -- same absent-not-malformed convention as every other
    optional throw-package field (see opendarts.capture.throw_package's own
    module docstring). Malformed/unreadable meta.json files are skipped,
    not raised -- a single corrupt throw directory must never abort a
    cleanup scan of everything else."""
    package_root = Path(package_root)
    referenced: set[str] = set()
    if not package_root.is_dir():
        return referenced
    for session_dir in package_root.iterdir():
        if not session_dir.is_dir():
            continue
        for throw_dir in session_dir.iterdir():
            if not throw_dir.is_dir():
                continue
            meta_path = throw_dir / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            pkg_id = meta.get("calibration_package_id")
            if pkg_id:
                referenced.add(pkg_id)
    return referenced


def cleanup_orphaned_calibration_packages(
    calibration_package_root: Path,
    throw_package_root: Path,
    *,
    active_package_id: str | None = None,
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Delete every calibration package under `calibration_package_root`
    that is NEITHER `active_package_id` NOR referenced by any throw
    package still present under `throw_package_root` -- the project's own
    real usage pattern: "calibrate a few times while dialing it in, then
    throw" means most calibration events on a real rig never get
    referenced by any throw at all, and letting them accumulate forever
    on the rig's local disk (raw FFV1 video per camera, real bytes, not
    JSON) is a real, unbounded disk-growth problem this function exists
    to bound.

    **Real deletion, not a `data/_delete/` quarantine** -- deliberately
    different from this project's standing "never permanently delete"
    corpus guardrail (docs/DESIGN.md), which protects historical SCORED throw
    data (every accuracy number in this project traces to it). A
    never-referenced calibration package is, by construction, raw board
    photos from a dialing-in attempt nothing downstream ever used --
    disk space is effectively unlimited once pulled off the rig; only
    the rig's own disk is a concern (any pull-time filtering is a
    separate, off-rig step) -- so these are disposable BY
    DESIGN, unlike corpus data. `active_package_id` is a hard exclusion
    regardless of any throw reference (the currently-live calibration
    must never be deleted out from under a running process, even in the
    narrow window before its first throw has been saved), and this
    function NEVER touches `throw_package_root` itself -- it only reads
    from it (via `find_referenced_calibration_package_ids()`) to decide
    what to keep.

    `dry_run=True` returns exactly what WOULD be deleted/kept without
    touching disk -- used by this module's own tests and available for
    an operator/maintenance script to preview before committing.

    **Why a standalone function, not auto-wired into every live
    `bootstrap_calibrations()` call**: this function's own cost is a
    full directory walk of `throw_package_root` (every session, every
    throw, one `meta.json` parse each) -- unlike the raw-frame FFV1
    encode (bounded, ~1-3s per camera, see module docstring), this scales
    with the TOTAL number of throw packages ever captured this session
    (or ever, if `throw_package_root` isn't itself pruned), which is not
    a cost this project's "must not meaningfully slow down or risk
    breaking the actual calibration" constraint should absorb on every
    single manual "Refresh calibration now" click during dialing-in.
    Instead: `opendarts.live.capture_daemon.bootstrap_calibrations()`'s own
    background-thread package-save wiring calls this automatically
    (see `save_calibration_package_background()` above) -- but only
    AFTER a real (not-reused) calibration successfully saves its own
    package, off the critical path either way, and only when a
    `throw_package_root` was actually given (opt-in per caller, so any
    existing/narrower test double is unaffected). This means cleanup
    runs once per REAL calibration event (the natural "an old
    calibration may have just been superseded" moment the task
    description itself suggests), not on a separate timer and not
    zero times -- while staying fully available as a standalone,
    manually- or cron-invoked maintenance call for an operator who wants
    to force a pass without recalibrating (e.g. `python -m
    opendarts.capture.calibration_package cleanup`, this module's own CLI
    below).

    **CLI caveat**: that standalone CLI runs as a separate OS process
    with its own empty `_IN_FLIGHT_PACKAGE_IDS` -- it has zero visibility
    into a live server process's own in-flight background save, so an
    operator invoking it manually while a real live calibration event
    might be in progress could still delete a package that event is
    actively writing (this module's in-flight guard is process-local by
    construction; see `save_calibration_package_background()`'s own
    docstring for the fuller caveat). Prefer letting the automatic
    post-save wiring run cleanup instead of the manual CLI whenever a
    live calibration might currently be in flight.

    **Concurrent-save race guard**: also keeps every package id currently
    registered (count > 0) in the module-level `_IN_FLIGHT_PACKAGE_IDS`
    counter (see module docstring's "CONCURRENT SAVES" section and
    `save_calibration_package_background()`'s own docstring) --
    regardless of which specific save call's background thread triggered
    THIS particular cleanup pass. Without this, two overlapping live
    calibration events (e.g. a double-clicked "Refresh calibration now")
    could race: one event's still-encoding, not-yet-throw-referenced
    package is neither the OTHER event's `active_package_id` nor
    referenced by any throw yet, so a plain reference-only cleanup could
    delete a package while its own raw-frame encode is still writing to
    it.

    Returns `{"kept": [...], "deleted": [...]}` (package ids, sorted) --
    `dry_run` or not, so a caller/log line can report exactly what
    happened either way.
    """
    calibration_package_root = Path(calibration_package_root)
    referenced = find_referenced_calibration_package_ids(throw_package_root)
    keep_ids = set(referenced)
    if active_package_id:
        keep_ids.add(active_package_id)
    with _IN_FLIGHT_LOCK:
        # `_IN_FLIGHT_PACKAGE_IDS` is a Counter, not a set (see its own
        # module-level comment) -- every key present has a count > 0 by
        # construction (a call's own `finally` block `del`s its entry
        # once decremented to zero), so a plain key-set update is correct
        # here; `set |= Counter` itself isn't supported by `set.__ior__`,
        # hence going through `.update()` on the keys explicitly.
        keep_ids.update(_IN_FLIGHT_PACKAGE_IDS.keys())

    kept: list[str] = []
    deleted: list[str] = []
    if not calibration_package_root.is_dir():
        return {"kept": [], "deleted": []}

    for pkg_dir in sorted(calibration_package_root.iterdir()):
        if not pkg_dir.is_dir():
            continue
        pkg_id = pkg_dir.name
        if pkg_id in keep_ids:
            kept.append(pkg_id)
            continue
        deleted.append(pkg_id)
        if not dry_run:
            shutil.rmtree(pkg_dir, ignore_errors=True)
            log.info(
                "cleanup: deleted orphaned calibration package %s (not the "
                "active calibration, not referenced by any throw package "
                "still present under %s)",
                pkg_id, throw_package_root,
            )

    return {"kept": sorted(kept), "deleted": sorted(deleted)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Calibration package maintenance CLI -- see this module's own "
            "docstring. Currently supports one action: cleanup."
        )
    )
    parser.add_argument("action", choices=["cleanup"])
    parser.add_argument(
        "--calibration-package-root", type=Path, default=DEFAULT_CALIBRATION_PACKAGE_ROOT,
    )
    parser.add_argument("--throw-package-root", type=Path, required=True)
    parser.add_argument("--active-package-id", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.action == "cleanup":
        result = cleanup_orphaned_calibration_packages(
            args.calibration_package_root,
            args.throw_package_root,
            active_package_id=args.active_package_id,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
