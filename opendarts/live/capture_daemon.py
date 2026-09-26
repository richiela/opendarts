"""opendarts/live/capture_daemon.py -- the always-on autonomous capture loop
entrypoint. See docs/DEPLOYMENT.md for the full design this implements
and for how it's meant to run (under run.sh / run.ps1, with no dependency
on an interactive session being alive).

TRIGGER, 2026-09 (READ THIS FIRST): dart throws and takeouts are decided
by ``opendarts.lifecycle`` (docs/LIFECYCLE.md) -- ``_lifecycle_step()``
below is the loop's single decision seam, and
``opendarts.lifecycle.adapter.LifecycleTriggerAdapter`` translates each
lifecycle tick into the ``opendarts.capture.trigger_state.ThrowTriggerState``
this file consumes. The legacy ``opendarts.capture.throw_trigger`` state
machine (``advance()``/``detect_motion()``/``is_settled()``/
``SETTLE_WINDOW_FRAMES``), ``opendarts.capture.discarded_events``,
``opendarts.capture.motion_calibration`` (except ``board_disc_mask``, now
``opendarts.capture.board_disc``), ``refresh_background_after_capture()``,
``_wait_for_stable_frames()`` and the post-capture refractory /
stale-reference self-heal / ambiguous-settle branches are DELETED, not
disabled -- there is no fallback trigger. Every mention of them in the
dated history below is exactly that: history, kept for the reasoning
record, not a description of the current loop.

STATUS (updated 2026-08-12, the turn/takeout state machine landed): the
daemon now runs a
FULL turn end to end without raising -- opendarts.capture.throw_trigger's
turn/takeout logic (dart_count, TAKEOUT_WAITING, the true-baseline-vs-
last-dart-background classifier that disambiguates "another dart landed"
from "a takeout is in progress") is real, and this file's own
refresh_background_after_capture() is no longer a stub. What IS real and
wired here (not faked, not stubbed): fetching live frames -- as of this
revision, DEFAULT frame source is DIRECT LOCAL CAMERA ACCESS
(opendarts.live.local_capture.LocalCameraHub, opened once at startup and
held open for this process's whole lifetime, per that module's own
design point), by design 2026-08-12: "i wanted you to write the camera
pipeline". Also real: bootstrapping per-camera calibration
(opendarts.calibration.sector_correspondence + opendarts.pipeline.
calibrate_camera, feeding the
same downstream calibration code), tip detection + scoring (both now
`opendarts.engines.apollo.ApolloEngine`, called via the engine
framework like any other primary engine as of 2026-08-12's engine-
framework-consolidation -- see this file's own `handle_ready_to_capture()`
docstring), package persistence
(opendarts.capture.throw_package.save_throw_package), and the full
detect_motion()/is_settled() trigger logic itself -- these pieces are
each independently real; by design's explicit priority ordering ("do
the 3 set throw WITHOUT the process scoring [blocking on it]"), a
rejected/low-confidence live score never blocks the loop from continuing
to watch for the next dart or takeout -- handle_ready_to_capture() always
saves a replay package regardless of the score's ok/not-ok status (per
docs/DESIGN.md's "Replay is the source of truth"), unchanged by this revision. What's still
NOT proven: running any of this continuously, unattended, against real
rig hardware for a real multi-dart turn (honest status:
opendarts.live.local_capture itself is still
logic-tested-only, never run against a real camera from any dev
session) -- and the takeout-vs-new-dart classification thresholds
themselves are structurally reasoned, not measured against a real
recorded takeout sequence (none exists in this project's available real
case data) -- see opendarts/capture/throw_trigger.py's module docstring
"HONEST CONFIDENCE NOTE" for the full caveat, same class of gap as
is_settled()'s own settle-duration tuning.

A read-only HTTP retrieval path now EXISTS (opendarts/live/server.py, a
separate FastAPI app/module, not part of this file) -- this daemon still
only WRITES packages to local disk; opendarts/live/server.py is what serves
them back out over HTTP without SSH.

COMBINED PROCESS, added later: opendarts/live/run_product.py runs this
loop's real logic (the "structure is real, math is not" pattern this
file already carries) AND opendarts/live/server.py's dashboard together in
ONE process sharing ONE LocalCameraHub, instead of two separate
processes each opening their own hub. It does NOT duplicate this file's
loop logic -- run_capture_loop_body() below is the actual shared
function both this module's own run_capture_loop() (standalone, opens/
owns/closes its own hub) and run_product.py (hub opened once by the
caller, shared with the web server, closed once by the caller) call.
Use THIS module alone (`-m opendarts.live.capture_daemon`) for a pure
console capture test with no web UI; use opendarts.live.run_product for the
real combined product path -- see that module's own docstring for the
full "when to use which" breakdown.

START/STOP/IDLE-TIMEOUT, added 2026-08-12: Start opens the cameras,
Stop closes them, an idle timeout (configurable) closes them too, and
the program no longer opens cameras on launch. Cameras no longer open
automatically the moment opendarts.live.run_product starts -- an explicit
`POST /api/start` (opendarts/live/server.py) does that now.
`run_capture_loop_body()` below gained an
`also_stop=` parameter so ONE session can end (manual Stop, or a
configurable idle-timeout, default 900s) without tearing down the
whole process -- the caller
(run_product.py's restructured capture thread) loops back and waits for
the next Start. `CaptureLoopController` (below, next to
CalibrationStore/ResetRequest) is the real coordination object -- see
its own docstring for the full design, including the honest
thread-vs-asyncio-task architecture note. Calibration on a
SECOND+ Start this same process lifetime reuses whatever's already in
the shared CalibrationStore instead of re-bootstrapping -- see run_capture_loop_body()'s own "Calibration
bootstrap" docstring section.

AD GROUND TRUTH, added 2026-08-12 (a real live finding, not
theoretical): the
`/api/state/detections` REST list only covers the current visit and is
not durable, which made the original plan (fetch AD's ground truth via a
later REST call/backfill CLI) structurally unable to recover ground truth
once the visit had ended in the meantime -- there is no way to poll fast
enough to reliably win that race. Fixed by
`opendarts.live.ad_ws_listener.AdWsListener` -- a persistent WebSocket
connection to AD's own `/api/events` push stream, opened/started by the caller
(`run_capture_loop()` below, or `run_product.py`'s equivalent) alongside
the camera hub and passed into `run_capture_loop_body()` already-started
-- same lifecycle discipline as `hub` itself: this file never opens,
closes, or owns it, only uses it. `handle_ready_to_capture()` matches
each freshly-saved package against the listener's own in-memory buffer
of recent AD throws (see `_attach_ad_ground_truth_from_ws()` below) --
NO network call at throw time at all (the network cost is paid
continuously, in the background, by the listener itself), still run on a
background thread and wrapped in try/except regardless, so this can
NEVER block or break the capture loop. This standalone daemon's `main()`
enables it by default (see `--no-ad-ground-truth` to opt out);
`run_product` connects only when the config's `ad_enabled` is true. The dashboard's existing
manual REST-based refresh button (`opendarts/live/server.py`, unchanged by
this work) remains as a fallback/backfill path for any package that
missed the inline attach (listener not running, disconnected at the
moment, etc.).

Usage:
    <repo>/.venv/bin/python3 -m opendarts.live.capture_daemon
Running this starts, logs its startup steps (local camera hub open +
calibration bootstrap and the trigger's IDLE/MOTION_DETECTED/SETTLING/
TAKEOUT_WAITING states, all real), and then runs indefinitely -- through
as many full 3-dart-then-takeout turns as actually happen in front of
the cameras, never raising NotImplementedError or exiting on its own --
until stopped with Ctrl-C/SIGTERM.

LATENCY FIX, 2026-08-12 (later the same day as the turn/takeout landing
above) -- see POLL_INTERVAL_SECONDS's own comment below for the dated
technical
detail: real live heartbeat logging on the rig showed this loop's actual
IDLE-state iteration cost at ~165ms, ~5.5x the 30ms design target, even
with zero motion. Root cause, confirmed by real profiling on this dev
machine against real 1280x720 frames (not guessed): fetch_current_frames()'s
local-camera branch was round-tripping every already-in-memory frame
through a throwaway PNG write-then-immediate-read-back on disk that
nothing else ever consumed -- measured at ~91ms/iteration, roughly HALF
the real total. Fixed by reading LocalCameraHub.grab_all()'s in-memory
cache directly instead (see fetch_current_frames()'s own docstring for
the full before/after numbers). HONEST CAVEAT: this closes the ~91ms
piece that was measurable from this dev machine alone; the real rig
iteration rate after this fix has NOT been re-measured live (no SSH/
live-system access for the session that made this change) -- real
confirmation is separate, later work.

N-FRAME-AVERAGED CALIBRATION, 2026-08-12 (later the same day as the
latency fix above). bootstrap_calibrations() below used to capture and
solve calibration from exactly ONE camera frame -- real live evidence
this caused meaningful noise: on the rig, 4 manual Calibrate clicks within
~4s of each other, each an independent single-frame solve on the SAME
physically unmoved cam2, produced reprojection_error_px swinging
0.47->3.80->4.05->3.12->3.80px across the 5 solves that day (real
run_product.log lines, 2026-08-12 19:00:02-19:07:14) -- an 8x spread from
pure per-frame noise, not anything physical changing. Fixed: now captures
CALIBRATION_N_FRAMES (50) independent frames per camera, detects landmarks
on each independently, averages the resulting image_points_px per
landmark index (opendarts.calibration.sector_correspondence.
average_correspondences() -- valid by construction, not assumed, see that
function's own docstring), and solves PnP once on the averaged points.
**Measured, not guessed** -- first against real archived rig camera
frames (a throwaway harness, not
shipped), later against REAL LIVE frames captured directly from the
running rig over HTTP (150
genuinely independent frames per camera pooled across 3 separate live
capture bursts -- not limited by an archived session's own throw count
the way the first harness was): a real, consistent noise-reduction
effect tracking a 1/sqrt(N) pattern (N=50 cuts reprojection-error std to
~12% of the single-frame value, averaged across all 3 cameras). Re-run a
SECOND time live under deliberately changed lighting -- the noise-reduction PERCENTAGE held
nearly identical both times (N=50: ~12.1% vs ~12.3%), confirming the
averaging benefit itself isn't a lighting-condition fluke. (That second
live run also surfaced a real, separate, unresolved finding: cam2's
ABSOLUTE single-frame noise more than doubled under the new lighting
while cam0/cam1 barely moved -- consistent with cam2's own
already-documented glare/reflection sensitivity, see
opendarts/capture/throw_trigger.py's "Measured thresholds" comment for the
real cam2 light-fixture-artifact finding -- and is real evidence toward,
not yet a fix for, docs/DESIGN.md's open per-camera-bias question. Not
addressed by this change, noted here only so the two investigations
cross-reference each other.) See CALIBRATION_N_FRAMES's own dated
comment below for the full numbers, the N=50 reasoning (
2026-08-12, after seeing the N=10 and then N=20/25/30 latency/accuracy
tradeoffs in sequence, calibration time was judged an acceptable price
for accuracy -- N=50 is the highest N with a clean, fully-independent real
measurement behind it, and the explicit, honest
caveat this does NOT fix: this addresses RANDOM per-frame noise only, not
a SYSTEMATIC bias shared by every frame in one quick burst (e.g. a
lighting-condition-dependent detector bias) -- that's the same
"correlated calibration bias" open question
opendarts/engines/apollo/scoring.py's MAX_RAY_DISAGREEMENT_MM docstring already tracks,
unresolved by this change. Both real call sites in this app -- the
Start-triggered auto-calibrate (run_capture_loop_body() below) and the
manual "Refresh calibration now" button (opendarts/live/server.py's
_refresh_calibration_blocking(), which calls this same
bootstrap_calibrations()) -- get this fix, since both go through one
function. NOT validated live on the rig's real hardware (no live-system
access for the session that made this change) -- separate follow-up
work.

STATUS-HONESTY AUDIT, 2026-08-12 -- hit a real live incident where
`opendarts.live.local_capture.LocalCameraHub`'s per-camera status kept
reporting `opened: true` over an hour after the hub had actually been
closed (idle-timeout auto-stop) -- see that module's own docstring and
`opendarts/live/server.py`'s for the full writeup and fix. Prompted an
explicit audit of every status/state surface this module's own
`CaptureLoopController`/`CalibrationStore` expose, for the same "does a
reported value get actively invalidated when reality changes, or could
it go stale and keep claiming an old truth as current" question.
**Both classes checked and found already honest, no fix needed here**:
`CaptureLoopController.meta()` recomputes `seconds_since_activity` fresh
from `time.monotonic()` on every call (never a stored/decrementing
countdown) and `running` is written directly by `request_start()`/
`request_stop()`/`mark_session_ended()` -- there is no code path where
the hub/session state changes without one of those three explicitly
running. `CalibrationStore.meta()`'s `source`/`checked_at_utc` is an
explicit, honestly-timestamped record of when/how the live calibration
was last SET, never presented as "still verified as of right now" --
same "timestamped snapshot, not a live claim" pattern `opendarts/live/
server.py`'s own audit entry documents for its calibration-display
fields. The one real bug found one layer up the stack, in
`opendarts/live/server.py`'s `AppState`: `trigger_state`/`trigger_dart_count`
(this module's own `on_event`/`TRIGGER_STATE` payloads, consumed there)
kept their last real value after a session ended, because
`CaptureLoopController.mark_session_ended()` (called from
`opendarts/live/run_product.py`'s capture thread, right below the
`else:`/`controller.mark_session_ended()` call sites) flips `running`
False but pushes no event of its own to reset the dashboard's copy --
fixed in `opendarts/live/server.py`'s `AppState.stop_capture()`, not here,
since that's where the stale copy actually lives.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import signal
import socket
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from opendarts.calibration.distortion import (
    derive_focal_and_k1_from_oriented_results,
    derive_focal_k1_cx_from_oriented_results,
    load_distortion_fallback,
    load_principal_point_fallback,
    write_distortion_fallback_entry,
    write_principal_point_fallback_entry,
)
from opendarts.calibration.focal_length import (
    derive_focal_length_from_oriented_results,
    load_focal_length_fallback,
    write_focal_length_fallback_entry,
)
from opendarts.calibration.oriented_landmarks import (
    PreOrientationLandmarks,
    correspond_landmarks_from_pre_orientation,
    locate_pre_orientation_landmarks,
    normalise_illuminant,
)
from opendarts.calibration.ring_correlation_orientation import (
    ring_correlation_orientation_for_camera,
)
from opendarts.calibration.rig_ring_geometry import (
    RING_GEOMETRY_DRIFT_THRESHOLD_DEG,
    CameraOrientationCandidate,
    RingGeometry,
    load_ring_geometry,
    resolve_rig_consensus_orientation,
    save_ring_geometry,
    update_ring_geometry,
)
from opendarts.calibration.sector_correspondence import average_correspondences
from opendarts.capture.calibration_package import (
    DEFAULT_CALIBRATION_PACKAGE_ROOT,
    new_calibration_package_id,
    save_calibration_package,
    save_calibration_package_background,
)
from opendarts.capture import clip
from opendarts.capture.lazy_frame import LazyFrame, LazyFrames, handles_of
from opendarts.capture.board_disc import (
    get_calibrated_board_disc_masks,
    set_calibrated_board_disc_masks,
)
from opendarts.capture.throw_package import (
    _rollup_fields_from_winning_sub_engine_diagnostics,
    agreement_string_from_diagnostics,
    calibration_from_dict,
    calibration_to_dict,
    save_ad_ground_truth,
    save_capture_diagnostics,
    save_throw_package,
    write_other_engines_result,
)
from opendarts.capture.trigger_state import MAX_DARTS_PER_TURN, ThrowState, ThrowTriggerState
from opendarts.disk_space import check_free_space
from opendarts.engines.base import EngineResult, engine_result_from_dict, engine_result_to_score_result
from opendarts.engines.apollo import engine_result_to_score_result as apollo_engine_result_to_score_result
from opendarts.engines.apollo.prior_dart_context import (
    CachedPriorThrowFrames,
    PriorDartLinePx,
    engine_accepts_prior_dart_line_px,
    find_prior_dart_line_px,
)
from opendarts.engines.dispatch import DEFAULT_ENGINE_TIMEOUT_S, dispatch_engines
from opendarts.engines.talos.prior_dart import (
    engine_accepts_prior_board_xy_mm,
    find_prior_board_xy_mm,
    find_prior_board_xy_mm_for_package,
)
from opendarts.engines.registry import (
    DEFAULT_ALSO_RUN,
    DEFAULT_PRIMARY_ENGINE,
    engine_names,
    get_engine,
    is_registered,
)
from opendarts.engines.zeus import ZEUS_SUB_ENGINE_NAMES
from opendarts.geometry.board import sector_ring_to_token
from opendarts.live import board_photo, calibration_progress, diagnostics_gate, local_capture
from opendarts.live.ad_ground_truth import DEFAULT_AD_BASE, DEFAULT_MATCH_WINDOW_SEC
from opendarts.live.build_info import build_info
from opendarts.live.heap_trim import release_freed_heap
from opendarts.live.ad_ws_listener import AdWsListener
from opendarts.pipeline import CalibrationAttempt, CameraCalibration, calibrate_camera

if TYPE_CHECKING:
    from opendarts.lifecycle.adapter import LifecycleTriggerAdapter, LiveStep
    from opendarts.lifecycle.driver import LifecycleDriver

log = logging.getLogger("opendarts.capture_daemon")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from opendarts.paths import DATA_DIR

# Where saved throw packages live: DATA_DIR/packages (the checkout's
# data/, or OPENDARTS_DATA_DIR), not under Path.home().
# save_throw_package() creates dest_dir on demand. Override via the
# package_root parameter (e.g. for tests) rather than editing this constant.
DEFAULT_PACKAGE_ROOT = DATA_DIR / "packages"


# Same convention, for EngineConfigStore's own durable snapshot -- added
# 2026-08-14 after a real incident: a process restart (to pick up new
# code -- something that happens often on this project) silently reset
# which engines were checked back to the code default (Apollo only,
# no also-run), with zero warning on the dashboard, which read as "no
# engine scoring is showing up" rather than "your config reset." Unlike
# calibration, engine config has no natural way to re-derive itself on
# startup (it's a pure operator preference, not something recomputed
# from cameras) -- so unlike CalibrationStore, EngineConfigStore actually
# LOADS this file back on construction, not just writes it for audit.

# Same convention again, for CaptureLoopController's own idle-timeout --
# added 2026-09-03, real live operator report. The EXACT same class of bug EngineConfigStore's own
# 2026-08-14 fix above closed for engine config, recurring here because
# CaptureLoopController never got the same treatment: `set_idle_timeout_
# sec()` only ever mutated an in-memory attribute, and `run_product.py`
# constructs a fresh `CaptureLoopController()` (falling back to
# IDLE_TIMEOUT_SEC_DEFAULT, 900s/15min) on every process restart -- which
# happens often on this rig (recalibration, deploys, the standard "kill
# PID, let the while-true loop relaunch" restart) -- silently discarding
# whatever an operator last set via the dashboard. Same reasoning as
# engine config: idle-timeout is a pure operator preference with no
# natural way to re-derive itself on startup, so it LOADS this file back
# on construction, not just writes it for audit.

# Same convention for LifecycleSettingsStore's operator-tunable
# dart_stable_frames (dashboard "Detection time"). Pure operator
# preference, so it loads this file back on construction.

# Scratch dir for fetched frames -- <repo>/tmp/, never /tmp or the
# harness scratchpad, per docs/DESIGN.md's filesystem discipline. This is the
# daemon's OWN scratch space for pulling snapshots before deciding what
# to do with them; it is NOT where finished throw packages are written
# (that's DEFAULT_PACKAGE_ROOT above).
SCRATCH_DIR = REPO_ROOT / "tmp" / "capture_daemon_scratch"

# 2026-08-12: was 0.5s, an UNMEASURED
# placeholder chosen only to avoid a tight busy-loop, that turned into a
# real product problem once actually measured live -- 0.5s x
# throw_trigger.SETTLE_WINDOW_FRAMES(5) = a hard 2.5s minimum stillness
# window on top of poll latency, measured live as ~3s throw-to-visible
# against the project's real sub-0.5s target ("I can throw 2 darts
# almost back to back"). Root cause was this constant, not
# is_settled()'s logic -- a capture loop has to run continuously near
# real camera FPS (measured live: cameraFps ~31), not on a slow fixed-
# interval poll. Dropped toward that same camera-native cadence (this
# rig's real cameras also run ~30-33fps per the same live session) so
# SETTLE_WINDOW_FRAMES consecutive frames now cost ~150ms of poll-sleep
# instead of 2.5s -- see is_settled()'s own docstring
# (opendarts/capture/throw_trigger.py) for why SETTLE_WINDOW_FRAMES's actual
# count is deliberately NOT changed here: 2026-08-12, "that
# doesn't have to be day 1" -- re-tuning the settle WINDOW SIZE (as
# opposed to this poll-rate root cause) needs real recorded throws to do
# honestly, not another blind guess from a dev machine; this fix only
# removes the artificial 0.5s-per-frame floor that was never measured
# against anything.
#
# NOT a busy-loop risk in the real (local-capture) product path: each
# loop iteration's own `fetch_current_frames()` call is real blocking
# I/O -- a `cv2.VideoCapture.read()` per camera (rate-limited by the
# camera's own real frame delivery) PLUS a `cv2.imwrite()` PNG encode/
# write to disk per camera (non-trivial at 1280x720, unmeasured exactly
# but real) -- which already costs meaningfully more than this poll
# interval alone on most hardware; this constant is a FLOOR on top of
# that real cost, not the loop's only pacing mechanism. `stop_event.wait()`
# (not a busy-spin) is what actually paces it, so CPU stays idle between
# iterations regardless of how small this number is. HONEST CAVEAT, not
# hidden: the write-then-immediately-`cv2.imread()`-back-from-disk round
# trip this frame pipeline already does (opendarts.live.local_capture.
# fetch_snapshot() writes, this module's own fetch_current_frames() reads
# it straight back) now happens far more often per second than before --
# real disk I/O this session did NOT attempt to remove or measure the
# cost of, since that's a frame-pipeline design question, not a poll-
# interval one; if throw-to-visible latency is still too slow after this
# fix on real hardware, that disk round-trip is the next thing to
# measure, not another poll-interval guess.
#
# FOLLOW-UP, 2026-08-12 (later the same day) -- the "next thing to
# measure" above WAS measured, and WAS the dominant real cost, confirming
# this comment's own prediction: real live IDLE-state heartbeat log lines
# from the rig (one recorded session, 4 consecutive 10s windows) showed
# ~165ms/iteration actual, ~5.5x this constant's 30ms target, even with
# zero motion (simplest possible per-iteration workload) -- so the excess
# had to be a FIXED per-iteration cost, not something scaling with
# motion-detection CV load. Profiled on this dev machine (no access to
# the rig's real cameras -- see honest split below) using real 1280x720
# frames pulled from an actual archived rig throw package
# (100-200 reps, mean timing):
# - fetch_current_frames() x3 cams, OLD code (imwrite-then-imread PNG
# round trip described in the paragraph above): ~91ms/iteration
# (cv2.imwrite ~20ms/frame + cv2.imread ~10ms/frame, x3 cameras) --
# roughly HALF of the real ~165ms/iteration total, for data that was
# already sitting in memory the whole time (hub.grab_all() already
# returns exactly the dict[int, np.ndarray] this loop needs).
# - advance() IDLE branch x3 cams (detect_motion(), the real CV-math
# cost this task was told NOT to assume was the bottleneck): only
# ~5ms/iteration -- confirms the CV math was never the problem, this
# constant's own reasoning above was right about that.
# FIXED: fetch_current_frames() now calls
# hub.grab_all() directly -- a pure in-memory cache read -- instead of
# round-tripping every frame through a throwaway PNG on disk that nothing
# else ever read back (see fetch_current_frames()'s own docstring for the
# full before/after). This removes the ~91ms/iteration measured above.
# HONEST REMAINING GAP: ~91ms (imwrite/imread) + ~5ms (detect_motion) =
# ~96ms accounts for the pure-compute/disk portion measurable on THIS dev
# machine; the real rig measurement was ~165ms, leaving ~69ms
# unaccounted for by this profiling. That remainder is NOT measured here
# and should not be claimed as explained -- plausible (unconfirmed)
# contributors include this dev machine's disk/CPU being faster than
# the rig's, Python/thread-scheduling overhead this harness's isolated
# per-function timing doesn't capture, or a genuine camera-I/O-adjacent
# cost this profiling approach cannot see without the real hardware. Real
# validation of the fixed loop's actual iteration rate on the rig has NOT
# happened as of this change -- see this module's own docstring for the
# standing "no live-system actions" constraint on the session that made
# this fix.
# REVERTED 0.03 -> 0.05, 2026-09-01, same evening as the frame-freshness
# gate above (MAX_FRAME_FRESHNESS_WAIT_S) -- a deliberate design decision
# not a re-measurement of this constant's own latency reasoning above
# (still believed correct on its own terms): "it may have been
# inadvertently helping us... it kept things more stable." Real, live
# natural experiment prompted this: a sibling system running a port of
# this code took the frame-freshness gate AND independently dropped ITS
# OWN poll interval 0.05->0.03 in the same commit -- 80 seconds later it
# produced this exact failure mode for the first time (a single dart captured twice,
# mid-flight then landed, 414ms apart). The gate stops a STALE DUPLICATE
# read from fooling is_settled() -- it does NOT and structurally cannot
# stop a genuinely STALLED pump (frame_count not advancing at all, not
# just lagging one cycle behind poll cadence); MAX_FRAME_FRESHNESS_WAIT_S's
# own fail-open ceiling still hands a stale frame to advance() once a
# camera has been stuck for a full second, by design (see that
# constant's own docstring -- "never worse than today's pre-fix
# exposure," not "never happens"). A slower poll interval reduces how
# often the gate's OWN ceiling has anything to fail open on in the first
# place, on top of whatever it was already doing for stability before
# the gate existed. The gate is new and unproven as of this revert --
# not yet enough real evidence to treat it as license to run a faster
# poll than what was already known-stable. That system is going back to
# 0.05 too, so both sit at the same known-good poll interval
# while a fresh test set establishes whether the gate needs anything
# more before this can be safely lowered again. HONEST CAVEAT, inherited
# from the same investigation: that system's source still read 0.05
# despite a commit message claiming 0.03 -- if the lower value never
# actually took effect there, that side of this "natural experiment" may not
# implicate poll interval at all. Proceeding on an explicit decision
# regardless of that open question, not concluding the poll
# interval is definitively the cause -- this revert is about restoring a
# previously-known-stable configuration while the gate gets more real
# throws behind it, not a settled root-cause claim on its own.
POLL_INTERVAL_SECONDS = 0.05

# FRAME-FRESHNESS GATE, 2026-09-01 -- real, live phantom-dart incident (a
# genuinely thrown dart captured twice: once mid-flight, once landed,
# ~326ms apart, consuming a turn's 3rd dart slot and losing the real 3rd
# dart entirely -- see local_capture.LocalCameraHub's own module
# docstring "ARCHITECTURE CHANGE" section and this constant's own use
# site in run_capture_loop_body() for the full mechanism). Root cause,
# confirmed both architecturally and against the real incident's own
# persisted diagnostics (per_camera_settle_offset_s identical across all
# 3 cameras, on 2 separate throws, at ~one POLL_INTERVAL_SECONDS -- far
# too fast and far too perfectly-synchronized across independently-
# moving cameras to be 3 real physical views agreeing motion stopped):
# LocalCameraHub.grab()/grab_all() are PURE CACHE READS of whatever its
# dedicated pump thread last wrote -- there is no synchronization between
# "the main loop wants a frame" and "the pump has actually produced a new
# one since the last read." When this loop's own poll cadence
# (POLL_INTERVAL_SECONDS) is close to or faster than a camera's real
# native frame period, the SAME cached frame can be read on two (or, at
# a smaller SETTLE_WINDOW_FRAMES, even just one *pair* of) consecutive
# polls. throw_trigger.is_settled() cannot tell "the scene genuinely
# stopped changing" apart from "we compared a frame to a literal copy of
# itself" -- a duplicate read trivially satisfies it (zero pixel
# difference), regardless of what's actually happening on the board.
# SETTLE_WINDOW_FRAMES=5->2 (12f8dfb, same evening) didn't introduce this
# race -- it only needs ONE duplicate pair to fire at N=2 instead of FOUR
# consecutive duplicate pairs at N=5, which is why this was rare-to-never
# observed before that change and fired within ~2 minutes after it.
#
# FIX: gate entry into the settle window on real per-camera frame
# novelty, using LocalCameraHub.status[cam].frame_count -- already
# incremented by the pump on every genuine cap.read() success (see that
# class's own CameraStatus field), just never consulted by this loop
# before now. A poll whose current_frames came from the SAME pump cycle
# as the previous accepted poll (frame_count unchanged for >=1 camera)
# is not treated as new evidence at all -- skipped entirely (no advance()
# call, no settle-window append), and this loop polls again shortly.
# Deliberately does NOT touch is_settled()'s comparison logic, any
# threshold, or SETTLE_WINDOW_FRAMES itself -- none of those are the
# defect; they were being fed bad input.
#
# THE TIMEOUT (the part most likely to get this wrong if bolted on
# instead of designed in from the start): waiting for EVERY camera to
# show a fresh frame_count blocks on the slowest one -- an actually-
# stalled/dropped camera would otherwise hang this loop forever instead
# of degrading, which is strictly worse than the phantom this fix exists
# to remove. MAX_FRAME_FRESHNESS_WAIT_S bounds that wait in real
# wall-clock time (not poll count -- a camera's own native frame period
# is a real-time fact, independent of how fast this loop happens to be
# polling) per-camera: once a specific camera has shown an unchanged
# frame_count for this long, the gate gives up waiting on THAT camera
# specifically (logged loudly -- see the use site) and proceeds anyway
# with whatever's cached for it, same "never block the live pipeline
# forever, degrade with a loud warning" posture as every other bounded
# wait in this module (_wait_for_stable_frames's own STARTUP_SETTLE_
# MAX_WAIT_S is the direct precedent) -- worst case for that one camera,
# on that one iteration, is exactly today's pre-fix exposure, never
# worse. A camera that later resumes producing fresh frames is trusted
# again immediately, with no lingering penalty.
#
# VALUE, structurally reasoned like STARTUP_SETTLE_MAX_WAIT_S's own
# comment, not independently measured against the rig's real cameras: a
# genuinely healthy camera only ever needs ~one native frame period
# (tens of ms) to catch up, so 1.0s is roughly a 30x margin over that --
# generous enough that ordinary pump/poll jitter never trips it, small
# enough that a truly dead camera degrades within a second rather than
# stalling captures noticeably.
MAX_FRAME_FRESHNESS_WAIT_S = 1.0

# How often the "stalled past the ceiling" WARNING may repeat while a
# stall persists, 2026-09-13 (CPU task). The stalled path is re-entered
# every iteration, so the line was emitting at the loop's full tick rate
# -- 499 lines in 10.66s, measured on the Windows rig. 5s keeps a stall
# plainly visible in the log (and still timestamps its start exactly, via
# the always-logged first iteration) while making the log cost of being
# broken independent of how fast the loop happens to be spinning.
STALL_WARNING_INTERVAL_S = 5.0

# Default idle-timeout, added 2026-08-12 alongside CaptureLoopController
# below (the project's direct request: "the cameras need to time out
# (configurable)"). 900s is a proven real-world default and no
# opendarts-specific reason was found to pick a
# different number -- 15 minutes of zero real activity (no dart captured,
# no manual Start/Stop/Reset/Calibrate click, see CaptureLoopController's
# own docstring for what counts) is the same real-world tradeoff either
# product is making (long enough that a normal, slow-paced casual game
# never trips it; short enough that an operator who genuinely walked away
# doesn't leave 3 cameras running indefinitely). `<= 0` disables the idle
# timeout entirely.
IDLE_TIMEOUT_SEC_DEFAULT = 900

# Absolute floor for the "camera read may be slow/stalling" warning in
# run_capture_loop_body() below -- see that warning's own comment. Chosen
# structurally (half a second is a real, human-noticeable stall on any
# reasonable frame source, local or HTTP), not measured against real
# per-iteration fetch timings on this rig (none exist yet -- same honest
# gap as everything else in this fix that needs real live-testing time to
# actually tune).
_FETCH_SLOW_WARNING_FLOOR_S = 0.5

# FRAME-AGE-AT-ADVANCE() DIAGNOSTIC, 2026-09-04 -- explicitly authorized
# debug instrumentation. Real question this
# answers: when `advance()` consumes `current_frames` (right after
# `fetch_current_frames()` returns, below), how long ago did the pump
# thread actually CAPTURE the frame being handed to it -- as opposed to
# `fetch_elapsed` above, which only measures how long THIS function call
# took, saying nothing about how stale the cached frame it returned
# already was the instant it was grabbed. Directly relevant to the "how
# old was the frame that tripped IDLE -> MOTION_DETECTED, and how long
# had the dart already been visible before that" latency question --
# see the trigger-state-transition block below for where this same
# per-iteration measurement is ALSO surfaced on that one transition
# specifically, so a single log line can answer it without correlating
# two separate lines by timestamp.
#
# Floor: fires an anomaly line whenever ANY camera's frame age exceeds
# this. 50ms is a real, reasoned starting point (not independently
# re-measured against live pump cadence data, which doesn't exist yet
# for this exact quantity -- this diagnostic is what will produce it):
# it's the full configured `POLL_INTERVAL_SECONDS` (0.05s) itself, i.e.
# "the frame being consumed is already at least one whole nominal poll
# cycle stale" -- a natural, self-documenting bar tied to a constant
# this module already defines, rather than an arbitrary round number.
_FRAME_AGE_LOG_FLOOR_S = POLL_INTERVAL_SECONDS

# Sample cadence: independent of the floor above, log the real frame-age
# distribution once every N iterations regardless of whether any camera
# exceeds the floor -- a floor-only mechanism would only ever show the
# tail, never the ordinary/typical case. N=20 at the
# ~0.05s configured poll interval is roughly one sample per real second
# during steady IDLE polling (POLL_INTERVAL_SECONDS itself is the
# CONFIGURED sleep, not the true per-iteration cadence -- see this
# file's own 2026-09-01 "REAL PER-ITERATION CADENCE" comment a few
# hundred lines below for why those differ in practice) -- frequent
# enough to build a real background distribution over a short session
# without materially adding to this module's own log volume (this
# project's own 2026-09-01/070c45c settle-diagnostics incident already
# established DEBUG log volume during an active episode is not free;
# this constant is deliberately independent of that fix's own
# `SETTLE_WINDOW_FRAMES`-scoped concern -- it fires during ordinary IDLE
# polling too, not just an active settle episode, which is exactly the
# "typical case" this sampling exists to characterize).
_FRAME_AGE_SAMPLE_EVERY_N_ITERATIONS = 20

# PER-ITERATION COST BREAKDOWN, 2026-09-04 -- explicitly authorized
# measurement instrumentation, built to reconcile a real, measured contradiction between two existing
# instruments: a real sample of MOTION_DETECTED->READY_TO_CAPTURE
# transitions (n=74) showed a 214ms median wall time
# (ThrowTriggerState.settle_duration_s) against a median loop-ITERATION-
# COUNT of only 1 spanning that same transition, while the pre-existing
# "real wall-clock time since the previous settle-episode iteration"
# cadence line (see `_last_settle_iteration_started`'s own comment in
# run_capture_loop_body()) reported only a 71ms median (n=323) on the
# SAME system. See that comment for the real, structural reason the two
# numbers can't both describe "the real cost of a settle-loop
# iteration," and docs/DESIGN.md's 2026-09-04 entry for the full
# reconciliation this instrument exists to support.
#
# Sample cadence for the ordinary-IDLE-baseline case: same convention
# and same reasoning as _FRAME_AGE_SAMPLE_EVERY_N_ITERATIONS immediately
# above (roughly one sample per real second during steady IDLE polling)
# -- reused directly rather than a second, independently-chosen number,
# since both diagnostics exist to characterize the same "typical
# background distribution, not just the tail" need.
_ITERATION_DIAG_SAMPLE_EVERY_N_ITERATIONS = _FRAME_AGE_SAMPLE_EVERY_N_ITERATIONS

# Anomaly floor for `body` (iteration start -> just before the sleep
# call) -- settle-path iterations (state != IDLE at entry) are ALWAYS
# emitted regardless of this floor (see the emission-volume comment at
# the real call site in run_capture_loop_body()), so this floor only
# matters for catching an abnormally slow IDLE iteration between the
# periodic samples above. Set well above BOTH real baselines already
# measured this session (idle ~76ms, settle-path median 214ms/p90
# 310ms) so it only fires on a genuine outlier -- reused directly from
# _FETCH_SLOW_WARNING_FLOOR_S ("half a second is a real,
# human-noticeable stall on any reasonable frame source") rather than
# inventing a second, arbitrary number. NOT independently re-measured
# against live idle-iteration-tail data on the rig (no live access from
# this worktree) -- a real gap for whoever verifies this live, same
# honest caveat as every other not-yet-tuned threshold in this module.
_ITERATION_DIAG_BODY_LOG_FLOOR_S = _FETCH_SLOW_WARNING_FLOOR_S

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# How often run_capture_loop_body() logs a "still <state>" heartbeat line
# when the trigger state hasn't changed -- found live 2026-08-12: state
# transitions were only ever pushed to the dashboard's WebSocket, never
# logged, so a hung/stuck loop was indistinguishable from a quiet one.
# Promoted to a module-level constant (2026-08-12, alongside the Bug #2
# TAKEOUT_WAITING diagnostic logging fix) so tests can monkeypatch it to
# something small instead of waiting 10 real seconds per test -- was
# previously a local variable inside run_capture_loop_body(), functionally
# identical, just not independently referenceable/patchable.
_HEARTBEAT_EVERY_S = 10.0


_visit_id_lock = threading.Lock()
_last_visit_stamp_ms = 0


# _THROW_NUMBER_LOCK, 2026-09-06/07 -- Zeus-latency follow-up task, Part 2
# ("background the PRIMARY engine's score call, not just the save").
# Guards the ENTIRE throw_number/generation allocation + `_reset_session_
# throw_numbering()` critical section (see both call sites below and
# `handle_ready_to_capture()`'s own naming block) -- a read-increment-
# write on a plain file, with NO lock, exactly the same class of bug
# `new_visit_id()`'s own docstring already documents a real, reproduced
# incident for (two visits sharing one ID on this project's very first
# concurrency-sensitive test run). Before this task, that race was
# UNREACHABLE in practice: `handle_ready_to_capture()`'s own throw-
# numbering block always ran on the SAME single capture-loop thread,
# synchronously, so no two allocations for the SAME session could ever
# be in flight at once (the loop physically could not start a second
# READY_TO_CAPTURE's own handle_ready_to_capture() call until the first
# one's entire body -- naming included -- had already returned). Part 2
# backgrounds the ENTIRE `handle_ready_to_capture()` call (not just its
# own internal save step, which was already backgrounded as of
# 2026-09-01) -- a real, fast follow-up dart can now land while the
# PREVIOUS dart's own background thread is still mid-flight through this
# exact block, on a SEPARATE thread, sharing the SAME `counter_file`/
# `generation_file`. Two overlapping allocations reading the identical
# stale counter value and writing the identical `throw_number` back would
# collide two different throws onto the SAME `dest_dir` -- a real,
# serious REPLAY/data-loss risk (one throw's package silently overwriting
# the other's), not a cosmetic race. A plain (non-reentrant)
# `threading.Lock()`, not `new_visit_id()`'s own module-global counter
# pattern, because this project's throw-numbering state lives in FILES
# (`session_throw_counters/<session_id>.count`/`.generation`), not a
# single in-process integer -- the lock protects the read-modify-write
# of those files, not a Python-level variable. Deliberately module-level
# and GLOBAL (not per-session): `opendarts.live.server.py`'s own
# Delete-packages endpoint can reset MULTIPLE sessions' counters in one
# call, on a THIRD thread (a FastAPI request-handling thread, already
# structurally separate from the capture loop even before this task --
# see that endpoint's own call site) -- one process-wide lock correctly
# serializes all three real callers (handle_ready_to_capture()'s own
# auto-detected-reset branch, run_capture_loop_body()'s manual-Reset
# branch, and server.py's Delete-packages) against each other, at a cost
# (microseconds of file I/O per real throw/reset) too small to matter.
# `_reset_session_throw_numbering()` itself does NOT acquire this lock
# internally (it would deadlock against handle_ready_to_capture()'s own
# call to it from INSIDE an already-held lock, since a plain `Lock()` is
# not reentrant) -- every caller is responsible for holding this lock for
# its own full critical section, including any nested call to that
# function; see each of the three real call sites for how they do this.
_THROW_NUMBER_LOCK = threading.Lock()


# ONCE PER SESSION, NOT ONCE PER THROW. A rig below the disk floor is
# below it for every throw that follows, and a line per throw is how an
# operator learns to scroll past the one message that mattered. So the
# first skipped package of a session says everything -- the numbers, the
# floor, what was skipped and what is unaffected -- at ERROR, and the
# rest of that session's skips go to DEBUG, where a full investigation
# can still count them.
_DISK_FLOOR_LOCK = threading.Lock()
_DISK_FLOOR_REPORTED_SESSIONS: "set[str]" = set()


def _report_package_skipped_for_disk(session_id: str, check, dest_dir: Path) -> None:
    """Say, loudly and exactly once per session, that packages are being
    skipped -- and say what is NOT affected, because the first question a
    line like this raises is "has the rig stopped scoring".
    """
    with _DISK_FLOOR_LOCK:
        first = session_id not in _DISK_FLOOR_REPORTED_SESSIONS
        _DISK_FLOOR_REPORTED_SESSIONS.add(session_id)
        # A process that runs for days accumulates one entry per session
        # start, which is a handful; the cap is hygiene, not a real bound
        # anyone is expected to reach.
        if len(_DISK_FLOOR_REPORTED_SESSIONS) > 256:
            _DISK_FLOOR_REPORTED_SESSIONS.clear()
            _DISK_FLOOR_REPORTED_SESSIONS.add(session_id)
    if first:
        log.error(
            "DISK FLOOR REACHED -- throw packages are NOT being written for "
            "session %s: %s. Scoring, the live board and match history are "
            "unaffected; what is lost is the replay package for %s and for "
            "every later throw this session. Free space on this rig, copy the "
            "existing packages and captures off it, or lower min_free_disk_gb "
            "in data/config.json. This is logged once per session.",
            session_id, check.reason, dest_dir.name,
        )
    else:
        log.debug(
            "disk floor: skipping throw package %s (%s)", dest_dir.name, check.reason
        )


def new_visit_id() -> str:
    """A fresh visit (turn) identifier -- up to MAX_DARTS_PER_TURN darts
    thrown before the board is cleared -- a `visitId` in the retail API.

    **Deliberately NOT a UUID.** This project has no uuid
    usage anywhere; its established ID convention is a millisecond epoch
    stamp -- matches `session_id = time.strftime("%Y%m%d-%H%M%S")` in
    `run_capture_loop_body()`. (`handle_ready_to_capture()`'s own
    `throw_id` was a bare epoch-ms stamp too until 2026-08-14, when it
    became the self-descriptive `f"{session_id}-{throw_number:03d}-
    {sector_token}"` -- see that function's own inline comment -- but the
    epoch-ms convention this visit ID follows is still very much alive
    elsewhere, e.g. `session_id` itself.) Matching that convention keeps
    every ID in this system sortable in the same (chronological) order,
    which a UUID4 would break for no gain here, and nothing about a visit
    ID needs to be unguessable.

    **Strictly increasing, not merely "the current millisecond."** The
    first version of this function was a bare `int(time.time() * 1000)`
    on the reasoning that a visit is physically seconds-to-minutes long
    so a millisecond stamp cannot collide. `tests/test_visit_model.py::
    test_visit_rotates_on_the_real_takeout_complete_transition` then
    produced two IDENTICAL visit IDs on its very first run, and it is
    worth being precise about why that is a real bug and not just a
    fast-test artifact: nothing about the ROTATION points is rate-limited
    by dart-throwing speed. A double Reset click rotates twice back to
    back, and a rotation immediately followed by another is exactly the
    "escape a stuck takeout" sequence an operator actually performs. Two
    visits sharing one ID is not cosmetic either -- `POST /api/visits/
    {visit_id}/throws/{index}/correct` resolves a throw by (visit_id,
    index) alone, so a collision would let a correction land on a dart
    from a different turn.

    So: the stamp is clamped to at least one more than the last one this
    process issued (which also covers a clock that steps backwards, e.g.
    NTP), under a lock because rotations happen on the capture thread
    while nothing stops another thread from asking too. Format and
    sortability are unchanged -- a bumped stamp is still a 13-digit
    millisecond-scale integer, still chronologically ordered.
    """
    global _last_visit_stamp_ms
    with _visit_id_lock:
        stamp = max(int(time.time() * 1000), _last_visit_stamp_ms + 1)
        _last_visit_stamp_ms = stamp
    return f"visit_{stamp}"


def _camera_matrix_for(
    cam: int, *, focal_px: float | None = None,
    image_width: float = IMAGE_WIDTH, image_height: float = IMAGE_HEIGHT,
) -> np.ndarray:
    """Build cam's pinhole intrinsics matrix.

    `focal_px` is REQUIRED in practice: focal length is derived live (or
    read from the persisted live-derived fallback), never assumed from a
    per-rig constant. Passing None raises rather than substituting a
    value.

    `image_width`/`image_height` default to this module's own constants;
    real call sites should pass the ACTUAL negotiated resolution -- see
    `_negotiated_resolution_for()` just below."""
    if focal_px is None:
        raise ValueError(
            f"cam{cam}: no focal length supplied -- focal length must be derived "
            f"live (or read from the persisted live-derived fallback); there is "
            f"no hardcoded per-rig constant to fall back on"
        )
    return build_camera_matrix(
        focal_px, image_width=image_width, image_height=image_height
    )


def build_camera_matrix(
    focal_length_px: float, *, image_width: float, image_height: float,
    cx: float | None = None, cy: float | None = None,
) -> np.ndarray:
    """Pure pinhole-intrinsics-matrix builder, factored out of
    `_camera_matrix_for()` (2026-08-26, the live-focal-length-derivation
    fix) so `_try_solve()` can build a camera matrix from a LIVE-DERIVED
    focal length (`opendarts.calibration.focal_length`) the exact same way
    `_camera_matrix_for()` already builds one from the hardcoded
    `MEASURED_FOCAL_LENGTH_PX` fallback -- one shared formula, not two
    copies that could drift. Zero skew, square pixels.

    `cx`/`cy`: PRINCIPAL POINT, 2026-08-26 (see
    `opendarts.calibration.distortion`'s own module docstring, "PRINCIPAL
    POINT (cx only)" section) -- default `None` means "image center,"
    this project's original, still-default assumption for `cy` and for
    any caller that hasn't derived a `cx` this event. Only
    `_try_solve()`'s own live-derived-cx call site ever passes a non-None
    `cx`; `cy` currently has no live-derived source at all (the real-data
    evidence showed it is NOT safely identifiable, unlike `cx` -- see
    that same docstring section) and is expected to always be passed
    `None` in practice, kept as a parameter only for symmetry/testability
    rather than hardcoding cy's own formula twice.
    """
    px = image_width / 2.0 if cx is None else float(cx)
    py = image_height / 2.0 if cy is None else float(cy)
    return np.array(
        [[focal_length_px, 0, px], [0, focal_length_px, py], [0, 0, 1]],
        dtype=np.float64,
    )


def _negotiated_resolution_for(
    cam: int, hub: "local_capture.LocalCameraHub | None"
) -> tuple[float, float]:
    """THE REAL SAFETY FIX, 2026-08-20 -- see opendarts.live.camera_resolution's
    module docstring for the full bug this closes. Returns the ACTUAL
    negotiated (width, height) for `cam`, read straight from `hub`'s own
    `CameraStatus.actual_width`/`actual_height` (set by
    `LocalCameraHub._open_one_locked()`'s real warm-frame read at open
    time, kept fresh by the pump thereafter -- opendarts/live/
    local_capture.py, not a hardcoded assumption) -- NEVER the hardcoded
    IMAGE_WIDTH/IMAGE_HEIGHT module constants directly, which is exactly
    the bug: this project's own audit found ZERO runtime check anywhere
    that the camera actually opened at that hardcoded resolution before
    this fix.

    Falls back to `(IMAGE_WIDTH, IMAGE_HEIGHT)` -- today's exact prior
    value, so behavior is UNCHANGED when there's nothing better to go on
    -- in two real cases: `hub is None` (a caller with no
    `LocalCameraHub` at all, so there is no negotiated-resolution signal
    to read), or the hub
    has no recorded `actual_width`/`actual_height` for this camera yet
    (never opened, or opened but never produced a successful read) -- in
    that case MEASURED_FOCAL_LENGTH_PX's own derivation resolution is
    still the most defensible number available, not a guess invented
    here.

    **BACKWARD COMPATIBILITY, provably not just assumed**: on real
    hardware that negotiates exactly 1280x720 (this rig's own real,
    current, only-ever-observed behavior -- see IMAGE_WIDTH/IMAGE_HEIGHT's
    own module comment), `hub.status[cam].actual_width/actual_height`
    equal `(IMAGE_WIDTH, IMAGE_HEIGHT)` exactly, so this returns the
    IDENTICAL numbers the old hardcoded-constant code always used --
    `tests/test_camera_resolution.py`/`tests/test_capture_daemon.py`
    assert this explicitly, not just by inspection.

    **Fails LOUD, not silent**, when the hub DOES have real negotiated
    dimensions and they genuinely differ from `(IMAGE_WIDTH,
    IMAGE_HEIGHT)`: logs a clear warning naming both, because
    `MEASURED_FOCAL_LENGTH_PX` (this file's own dated comment) was
    measured AT that specific hardcoded resolution -- a differently-
    negotiated camera makes the principal point right (this function
    fixes that much) but the focal-length ESTIMATE itself is now
    resolution-mismatched and needs re-deriving for this camera at its
    real resolution before the resulting calibration should be trusted --
    exactly the silent-bug class this whole fix exists to close, so this
    is never allowed to pass quietly."""
    if hub is not None:
        # getattr(..., None), not a bare hub.status -- several existing
        # tests (and any future test double) pass a plain sentinel/mock
        # object as `hub` that is not a real LocalCameraHub and has no
        # `.status` at all; that must fall back to the safe historical
        # default below, not raise AttributeError. A real LocalCameraHub
        # always has `.status` (set unconditionally in __init__), so this
        # never masks a real bug on the actual live path.
        hub_status = getattr(hub, "status", None)
        status = hub_status.get(cam) if hub_status is not None else None
        if status is not None and status.actual_width and status.actual_height:
            width, height = status.actual_width, status.actual_height
            if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
                log.warning(
                    "cam%d: camera negotiated %dx%d, NOT the %dx%d "
                    "MEASURED_FOCAL_LENGTH_PX was derived at -- calibration math "
                    "below will use the REAL negotiated %dx%d for the principal "
                    "point (this is the fix), but the focal-length ESTIMATE "
                    "itself is now resolution-mismatched and should be "
                    "re-derived at this camera's real resolution before "
                    "trusting the resulting calibration",
                    cam, width, height, IMAGE_WIDTH, IMAGE_HEIGHT, width, height,
                )
            return float(width), float(height)
    return float(IMAGE_WIDTH), float(IMAGE_HEIGHT)


# N-FRAME-AVERAGED CALIBRATION, added 2026-08-12. Real motivating evidence
# (not theoretical): live on the rig, 4 manual Calibrate clicks within ~4s of
# each other, each an independent single-frame PnP solve on the SAME
# physically unmoved cam2, produced reprojection_error_px swinging
# 0.47 -> 3.80 -> 4.05 -> 3.12 -> 3.80 across the 5 solves that day (real
# log lines, run_product.log, 2026-08-12 19:00:02-19:07:14) -- an 8x
# spread from pure per-frame noise (sensor noise + sub-pixel landmark-
# detection jitter), not anything physical changing. The fix: combine a
# sequence of frames into one calibration. bootstrap_calibrations() below now
# captures CALIBRATION_N_FRAMES independent frames per camera, runs
# landmark detection on each independently, averages the resulting
# image_points_px per landmark index (see
# opendarts.calibration.sector_correspondence.average_correspondences()'s own
# docstring for why averaging by index is valid here, verified not
# assumed), and solves PnP ONCE on the averaged points -- instead of one
# noisy single-frame solve.
#
# **MEASURED, not guessed** -- a throwaway harness (not shipped)
# against REAL data: every real bg_camN.png
# across 3 real archived rig sessions (24-27 real throw-folders each, same physically fixed rig/cameras, frames spaced
# MINUTES apart within a session -- an HONEST, if anything MORE
# adversarial than a tight capture burst, proxy for real per-frame noise;
# see that script's own module docstring for the full "what this data is
# and isn't" caveat). For each of the 9 real (session, camera) pairs:
# independently single-frame-calibrated EVERY real bg frame (the direct
# analogue of the live 4-rapid-click evidence above, n=24-27 per pair
# instead of n=5) and, separately, ran calibrate_camera() ONCE on the
# elementwise average of N randomly-drawn real frames' image_points, 200
# trials per N, for N in {1,2,3,5,8,10}. Real aggregate result (std of
# reprojection_error_px across trials, relative to the single-frame std,
# averaged over all 9 real pairs):
# N=1: 1.05x (sanity check -- matches ~1.0 as expected)
# N=2: 0.62x (1/sqrt(2) predicts 0.71x)
# N=3: 0.47x (1/sqrt(3) predicts 0.58x)
# N=5: 0.35x (1/sqrt(5) predicts 0.45x)
# N=8: 0.27x (1/sqrt(8) predicts 0.35x)
# N=10: 0.23x (1/sqrt(10) predicts 0.32x)
# A real, consistent noise-reduction effect, tracking (slightly beating,
# on this real dataset) the 1/sqrt(N) pattern expected for independent
# per-frame noise. Originally chose N=5 on this archived-data-only
# evidence (real diminishing returns visible: N=1->5 cuts std by ~65
# percentage points, N=5->10 only ~12 more), then bumped to N=10
# (2026-08-12, same day), trading calibration time for accuracy --
# N=10 was, at that point, the highest N
# this project had real measured data for from archived per-throw bg
# frames, which top out at 24-27 per real session (can't measure higher
# N against a pool smaller than N).
#
# SECOND MEASUREMENT ROUND, same day, against REAL LIVE frames instead
# of archived-and-pool-limited ones (tmp/measure_calibration_averaging_
# live.py -- fetches real fresh frames directly from the running rig's
# own /api/cameras/{cam}/snapshot.png over HTTP, no archive-size ceiling):
# 3 separate live capture bursts of 50 frames/camera each (150 genuinely
# independent real frames per camera total), 300 trials per N this time,
# N swept up to 150. Real aggregate result (std ratio vs single-frame,
# averaged across all 3 cameras):
# N=10: 0.30x N=20: 0.20x N=30: 0.17x N=50: 0.12x N=100: 0.06x
# (N=150 is a measurement-methodology degenerate case -- with a pool of
# exactly 150 real frames, N=150 has only ONE possible sample, so its
# std is trivially 0.000, not a real "zero noise" result -- excluded from
# the real curve above for that reason, not omitted by accident.)
# Re-run a THIRD time, same live method, after deliberately changed
# the rig's physical lighting between runs: the N=30/N=50 percentages
# held nearly identical (0.166->0.171x, 0.121->0.123x) -- real evidence
# the averaging benefit itself is not a lighting-condition-specific
# fluke. (That lighting-change re-run also surfaced cam2's absolute noise
# more than doubling while cam0/cam1 barely moved -- a real, separate,
# still-open finding, see this file's module docstring above and
# the per-camera-bias open question; not something this N
# value fixes.)
#
# After seeing the N=10/20/25/30 tradeoffs in sequence, the decision was
# to accept the calibration time for accuracy, set N=50 for now, and dial
# it back later if needed. N=50 chosen as the landing point on a real, still-improving (not yet
# flattened) curve -- N=100 measures even better (0.06x) but 50 is
# the project's explicit call given real diminishing returns are visible and
# it's an easy constant to raise again later if warranted, not a
# structural limit. Latency cost estimate for N=50 (real numbers cited,
# not invented): landmark detection alone measures ~33ms/image
# (opendarts/calibration/sector_correspondence.py's own docstring, real
# measurement across 360 real images) -- N=50 x 3 cameras = 150 detections
# sequentially vs. today's 3 = ~147 extra detections x ~33ms =~ +4.9s;
# capturing 50 genuinely-fresh pump frames instead of 1 adds ~49 more pump
# waits, each bounded by the camera's real negotiated frame interval
# (this file's own POLL_INTERVAL_SECONDS comment cites a real measured
# ~31fps / ~32ms cadence on this rig) =~ +1.6s. Total estimated added
# latency for this one-time Start/manual-Calibrate action: roughly
# +6.4s -- paid once per calibration event, not per scored dart, and
# explicitly an acceptable tradeoff by design's own call above.
#
# **HONEST LIMIT, stated plainly, not implied away**: this fixes RANDOM
# per-frame noise. It does NOT fix a SYSTEMATIC bias shared by every
# frame in one quick N-frame burst (e.g. a lighting-condition-dependent
# detector bias affecting every frame in the burst the same way) --
# averaging N correlated-biased samples reproduces the same bias N times,
# it does not cancel it. This is the SAME "correlated calibration bias"
# gap already tracked in opendarts/engines/apollo/
# scoring.py's MAX_RAY_DISAGREEMENT_MM docstring -- unresolved by this change, and not
# validated live on the rig (no live-system access for the session that made
# this change; see this file's module docstring for the explicit
# follow-up-work note).
CALIBRATION_N_FRAMES = 50

# DECOUPLED RAW-CAPTURE-VS-DETECT TARGETS, added 2026-08-21 (real perf
# scoping task -- measured, not re-derived here). Before this, `CALIBRATION_N_FRAMES` served TWO
# jobs at once: "how many raw frames to grab" AND "how many of those to
# run through the expensive landmark detector before the retry loop's
# first solve attempt" -- and every camera paid full detection cost
# (measured ~110ms/frame, see CALIBRATION_MAX_N_FRAMES's own dated
# comment below) for all 50, even though real A/B testing this session
# (99 real archived bg frames from one session in the
# `data/archive/clean/` corpus) proved N=10 DETECTED frames gives
# IDENTICAL scored segments to N=50 across 10 independent trials replayed
# against all 99 real throws, and the orientation hint locks correctly
# (matches the N=99 reference roll) in 90/90 trials at N>=8. Only
# `measure_ring_boundary_offsets()`'s `treble_inner` boundary specifically
# needs the bigger pool -- real scatter up to ~0.9mm at both N=10 and
# N=20 raw frames, tight (deltas -0.19 to +0.01mm vs a full-99-frame
# reference) at N=50 -- and that function (plus board-color's
# `collect_color_samples()`/`derive_thresholds()`) never touches detected
# landmarks at all, just raw BGR pixels + the already-solved calibration
# (see both post-loop sections further down in
# `_bootstrap_calibrations_unlocked()`).
#
# `CALIBRATION_N_FRAMES` (above) now means ONLY "how many raw frames to
# capture per camera, total" -- still 50, still cheap regardless of count
# (measured ~1.77-2.03s for 50 frames across all 3 cameras; it's DETECT
# that's expensive, not capture). `CALIBRATION_N_FRAMES_DETECT` below is
# the NEW, separate target the round loop's detect/solve/retry logic
# (`_process_camera_round()`/`_try_solve()`) actually chases on round 1
# -- the raw frames beyond this target are still captured (part of the
# same single `_capture(n_frames)` call, no extra capture round needed)
# but appended to `pre_orientation_pool[cam]` as `(frame, None)` pairs,
# bypassing `locate_pre_orientation_landmarks()`/
# `correspond_landmarks_from_pre_orientation()` entirely -- purely raw
# pixels for the two post-loop sections that only ever read the `pb`
# half of that pool's tuples, never the `_pre` half (confirmed by reading
# both consumption sites, `ring_bg_frames`/`color_samples` construction
# below, before making this change). If the target-reprojection-error
# solve doesn't succeed at N=10 detected frames, the EXISTING retry
# mechanics take over unchanged (capture+detect `retry_batch_size` more,
# up to `CALIBRATION_MAX_N_FRAMES`) -- this change is about WHEN detect
# first runs, not any acceptance/retry logic downstream of it.
#
# RAISED 10 -> 50, 2026-08-26. The 2026-08-21 A/B test that
# originally set this to 10 measured a coarse pass/fail bar (does N=10
# change the final SCORED SEGMENT vs N=50 -- no, on that day's corpus) and
# predates BOTH the ring-boundary-offset full-storage fix and, especially,
# the focal-length-derivation rework (RESOLVED 2026-08-26, this same
# session -- see `opendarts.calibration.focal_length`'s own module
# docstring), which now runs on however many detected frames this
# constant hands it. Re-measured on THIS repo's real, local calibration
# packages (the calibration corpus, 7 real calibration
# events x 3 cameras = 21 real camera-events, sweeping N in
# {10,20,30,40,50}) via the REAL, unmodified `bootstrap_calibrations()`
# pipeline (not synthetic data, not a re-derivation from memory):
#
# - For the two historically-easy cameras (cam0/cam1, every package):
# reprojection_error_px and focal_length_px are ALREADY excellent and
# essentially flat from N=10 through N=50 -- median per-event delta
# (N=50 minus N=10) is +0.006px reprojection / +0.125px focal length;
# median per-event std ACROSS N=10..50 is 0.011px reprojection /
# 0.82px focal length (excluding one flagged outlier below). This
# RECONFIRMS the 2026-08-21 finding still holds post-focal-length-fix,
# for these two cameras: N=10 was never the bottleneck for them.
# - cam2, this rig's historically weakest-signal camera (already flagged
# in the focal-length-fix entry above as having the lowest
# orientation-hint confidence of the three), tells a DIFFERENT, real
# story: it FAILED TO CALIBRATE AT ALL at N=10 in 2 of 3 struggling
# real events (not enough frames cleared `average_correspondences()`'s
# own min-samples-to-trust-an-average gate), only becoming reliably
# solvable once N reached 40-50. This is a FEASIBILITY improvement,
# not just a precision one -- more frames were the actual difference
# between "no calibration at all" and "a real one" for this camera on
# those two real events. A third chronic-failure event (cam2 again)
# still failed at every N tested up to 50 -- more frames within this
# range do not rescue a genuinely bad per-event capture (bad
# lighting/framing that day), an honest limit, not evidence against
# raising N.
# - One real, honestly-flagged anomaly, not chased further (same
# "generalization now" posture this file's CALIBRATION BOOTSTRAP FIRST
# PRINCIPLE already applies elsewhere): cam0 in one single calibration
# event showed genuinely non-monotonic
# swings across N (reprojection_error_px 1.21 -> 2.14 -> 3.07 -> 1.40
# -> 2.18px at N=10/20/30/40/50) -- more frames did not reliably
# stabilize this one event. Excluded from the aggregate numbers above
# (stated plainly, not hidden) since it would otherwise dominate a
# small-N mean; kept as a known, real data point.
# - N > 50 could NOT be reliably measured with this local corpus: only
# 2 of 7 real packages captured a raw pool bigger than 50 (both on
# cam2 alone, 75-100 frames) and this measurement's own single-camera
# replay harness hit a real bug at N=60/75/100 (a bootstrap_
# calibrations() code path that assumes camera index 0 is present,
# which a cam2-only replay violates) -- an honest harness limitation,
# not a pipeline finding, and not chased further given time budget.
# `CALIBRATION_N_FRAMES` (the raw-capture ceiling, unchanged at 50)
# is NOT raised by this measurement -- there is no real evidence here
# that it needs to move, only that DETECTING more of what's already
# captured helps.
#
# REVERTED 50 -> 10, 2026-08-26, same day, after a direct LIVE follow-up
# test the corpus-replay measurement above couldn't do: the same test
# already run on another rig (stress-testing a port of this exact
# code): 10
# clean back-to-back live "Refresh calibration now" events at N=10.
# Result: 30/30 camera-events succeeded --
# cam0 3.025-3.104px, cam1 0.930-1.082px, both tight; cam2 0.279-1.466px,
# real run-to-run spread but never failed. Under TODAY's lighting, N=10
# is fully reliable -- the feasibility failures the corpus-replay
# measurement above found were specific to genuinely struggling
# (bad-lighting) events, not a general N=10 weakness.
#
# Then, to settle it: replayed
# ONE of these just-captured real calibration packages at N=10 vs N=50, isolating the frame-count variable alone
# (identical lighting, identical everything else -- see
# `dev/calibration/measure_calibration_frame_count.py`'s own
# `measure_package()`, reused directly for this, not a new harness):
# cam0: 3.029px (N=10) vs 3.052px (N=50) -- N=50 WORSE, noise-level
# cam1: 0.995px (N=10) vs 0.960px (N=50) -- N=50 slightly better
# cam2: 0.341px (N=10) vs 0.645px (N=50) -- N=50 WORSE, +0.30px
# N=50 does NOT reliably improve precision on a package that already
# calibrated fine at N=10 -- it can make it measurably worse. An
# independent same-package replay elsewhere reached the same conclusion.
#
# Reconciling with the earlier corpus measurement's real feasibility
# finding (cam2 failing outright at N=10 on 2 of 3 STRUGGLING events):
# this project already has an ADAPTIVE RETRY LOOP (immediately below)
# that captures+detects MORE frames, up to CALIBRATION_MAX_N_FRAMES,
# specifically when the target isn't met at the starting N -- the
# struggling-camera case is already covered by that existing mechanism,
# automatically, only when actually needed. Starting every single
# calibration event at N=50 paid the extra ~4.4s cost (and, per the
# same-package replay above, sometimes a worse result) even in the
# common case where N=10 was always going to be fine. N=10 restored as
# the starting point; the adaptive retry loop remains the real answer
# for a struggling event, exactly as it was designed to be.
#
# LOWERED 10 -> 5, 2026-08-29, real live N=1/3/5/10 sweep (following an
# earlier N=1 experiment). Method: 10 back-to-back live "Refresh
# calibration now" events on the rig at each of N=1/3/5/10 (each its own
# deployed commit), reading real per-camera reprojection_error_px from
# each event's own derived_calibration.json (script:
# `tmp_calib10x.py`, real numbers, not simulated). Real means across
# the 10 runs at each N:
# N=1: cam0 0.312px cam1 1.054px cam2 0.759px | 9.64s
# N=3: cam0 0.258px cam1 0.728px cam2 1.098px | 11.45s
# N=5: cam0 0.338px cam1 0.626px cam2 0.826px | 13.20s
# N=10: cam0 0.343px cam1 0.567px cam2 0.797px | 17.71s
# cam0 is flat at every N (no real dependence). cam2 is dominated by
# its own run-to-run noise, not N (its worst mean was actually at
# N=3, not N=1 -- no clean trend). cam1 is the one camera that
# genuinely, monotonically improves with more frames, but with sharply
# diminishing returns: N=1->3 buys 0.33px, N=3->5 buys another
# 0.10px, N=5->10 buys only 0.06px more -- 97% of the total available
# benefit is already captured by N=5, and N=5's cam1 mean (0.626px)
# has zero runs above 1.4px the way N=1 repeatedly did. N=10 costs
# 4.5s more per calibration (17.71s vs 13.20s, +34%) for that last
# 0.06px, comfortably inside the noise floor of everything else in
# this table -- not worth it. N=5 restored as the shipped default.
# NOTE, honestly flagged, not chased further: this parameter is the
# MAIN ROUND LOOP's own detect/solve frame count (feeds focal-length/
# k1/cx/PnP/reprojection) and is architecturally independent of the
# own separate, fixed-at-3 `RING_CORRELATION_ORIENTATION_N_FRAMES`
# (orientation resolution runs earlier, in its own phase, before this
# loop even starts) -- so this sweep says nothing about, and should
# not be read as evidence about, orientation-hint quality specifically.
CALIBRATION_N_FRAMES_DETECT = 5

# ADAPTIVE RETRY LOOP, added 2026-08-14. Before this, bootstrap_calibrations() captured a
# FIXED CALIBRATION_N_FRAMES and solved ONCE, unconditionally accepting
# whatever reprojection_error_px came out -- there was NO accept/reject
# threshold anywhere in this path; "ok" only ever meant "PnP converged",
# never "the result is actually good". The rule now: keep trying until
# N reaches 200 or the error drops below the target. Below CALIBRATION_TARGET_REPROJECTION_
# ERROR_PX, a camera is accepted immediately. Above it, bootstrap_
# calibrations() now keeps capturing MORE frames for that camera --
# accumulating into the SAME robust-averaged pool from average_
# correspondences() above, never restarting from zero -- and re-solving,
# until either the target is met or CALIBRATION_MAX_N_FRAMES is reached.
#
# Real historical numbers behind why this is worth doing (from
# docs/DESIGN.md's own logged real calibration runs, post real-intrinsics
# fix): cam0 typically 1.68-1.96px, cam1 typically 2.29-2.36px (never
# once cleared 2px in any logged real run found), cam2 typically
# 0.75-1.04px. **Honest expectation, not a bug if observed**: cam1
# specifically may frequently or even always hit the N=200 cap without
# reaching the 2px target -- this loop makes that VISIBLE (a clear
# logged warning + a `target_met=False` diagnostics flag, see
# `diagnostics_out` below) instead of silently accepting a
# never-actually-verified-good calibration as if it were fine, which is
# the real gap this whole change closes even on a camera that never
# clears the target.
#
# Real latency cost, worth stating plainly and MEASURED against this
# rig's actual oriented-landmark detector (NOT the older ~33ms/image
# figure cited in CALIBRATION_N_FRAMES's own comment above, which was for
# sector_correspondence.py's simpler, now-superseded detector --
# oriented_landmarks.correspond_landmarks_oriented() does real extra work
# per frame, a phase-lock angular search plus white-balance correction,
# and measures meaningfully heavier): a real timing loop against this
# rig's own real cam0/cam1/cam2 bg images
# (20 calls/camera) gives
# **~110ms/call, consistent across all 3 cameras** (109-112ms). At the
# N=200 cap, ONE chronically-bad camera pays 200 x 110ms =~ 22s of
# detection alone, on top of the real pump-wait time for 200 fresh frames
# (~32ms cadence per this file's own POLL_INTERVAL_SECONDS comment) =~
# 6.4s -- roughly 28s worst case for one chronically-bad camera. Per
# the project's own real historical numbers (cam1 typically 2.29-2.36px,
# documented as never once clearing 2px in any logged real run) this is
# a real, not hypothetical, worst case: cam1 alone hitting the cap on
# EVERY calibration event would be a genuinely-expected ~28s addition,
# and if cam0/cam1 both chronically miss simultaneously (plausible per
# those same historical numbers, cam0 also never confirmed below 2px),
# the total could run closer to ~50-56s (their retries happen in
# parallel rounds sharing one capture call, not serially, so it is NOT a
# simple 2x -- rounds are shared across every still-remaining camera,
# only the LAST few rounds where one camera has already dropped out cost
# that camera's detection time with nothing to show for it). Paid once
# per calibration event (startup / manual "Refresh calibration now"),
# same "acceptable tradeoff, not a per-scored-dart cost" category as the
# original N=50 latency estimate above -- but a real, honestly-stated
# cost, not a rounding-error one, and worth the project's awareness given how
# much bigger it is than the original ~13s estimate this comment
# previously (wrongly) cited before being corrected against a real
# timing measurement. Retries capture in CALIBRATION_RETRY_BATCH_SIZE
# -sized batches (not one frame at a time) specifically to bound the
# NUMBER of retry rounds (worst case (200-50)/25 = 6 extra rounds, not
# 150) -- one-at-a-time retries would multiply per-round bookkeeping
# overhead for no real accuracy benefit, since average_correspondences()'s
# own noise-reduction curve is already smooth in N (see that constant's
# own real measured 1/sqrt(N)-tracking numbers above), not something that
# needs single-frame granularity to hit a target.
#
# TARGET RAISED 2.0 -> 2.5, 2026-08-14 evening, live on the rig. Real
# measured cause: a fixed single N=50 batch (no retry) reliably gives
# cam1 ~2.17-2.22px (6 real trials) -- comfortably under 2.5, nowhere
# near 2.0. cam1 was never actually failing to calibrate well; it was
# failing a target that a single good batch structurally cannot clear,
# which meant EVERY calibration event forced it into the retry path.
# That retry path was then measured to make things WORSE, not better:
# real live sweeps that evening (single clean batches, N=5 through
# N=200, 5 trials each) show a smooth, monotonic improvement from N=25
# (~2.42px) through a wide plateau at N~=50-60 (~2.16-2.22px) with no
# further real gain past that -- but the retry-accumulated N=200 path
# (7 separate small batches merged together, each getting its own
# independently-computed, individually-noisier robust-trim threshold --
# see average_correspondences()'s own docstring) measured WORSE than a
# single clean N=50 batch: 2.47-2.51px live, vs 2.17-2.22px for N=50
# alone, across 3 real full runs each. So the old 2.0px target was
# actively self-defeating for this camera -- it manufactured the retry
# path's own accuracy cost by demanding an unreachable number from a
# single good batch. 2.5px is a real, measured, comfortably-clearable
# target for a healthy single N=50 batch (cam0/cam2 clear it even more
# easily, ~1.6-1.7px), not a loosened bar for its own sake -- it exists
# specifically so a healthy camera never has to pay the retry
# mechanism's own real cost. The retry path/accumulation-bias issue
# itself is NOT fixed here -- it still exists, still gets exercised by
# any camera that's genuinely bad (not just missing a too-tight target),
# and is a real follow-up (compute the trim threshold from the full
# accumulated pool each round, not per-round) if it ever needs to fire
# for real.
CALIBRATION_TARGET_REPROJECTION_ERROR_PX = 2.5
CALIBRATION_MAX_N_FRAMES = 200
CALIBRATION_RETRY_BATCH_SIZE = 25

# BEST-OF-N REPROJECTION ATTEMPTS, added 2026-08-30, following the calibration-repeatability findings docs/DESIGN.md documents
# under "Calibration repeatability findings, 2026-08-27": instead of
# accepting the FIRST attempt that clears `CALIBRATION_TARGET_
# REPROJECTION_ERROR_PX`, ALWAYS run a configurable number of genuinely
# independent detect+solve attempts per camera (fresh `_capture()`
# frames each attempt -- never the same pixels an earlier attempt or the
# existing adaptive-retry loop already analyzed) and adopt whichever
# attempt had the LOWEST reprojection_error_px for that camera,
# regardless of which attempt achieved it. Started at N=5.
#
# This constant is the one-line lever for what N should be WHEN this
# feature is turned on. It is deliberately NOT the default value of
# `bootstrap_calibrations()`'s own `n_reprojection_attempts` parameter
# (that stays 1, i.e. today's exact "accept first attempt that clears
# target" behavior, for every existing caller) -- see that function's
# own "BEST-OF-N REPROJECTION ATTEMPTS" docstring section for the full
# design and why this project's own standing "must not silently change
# default production behavior" guardrail means the parameter default and
# this constant are intentionally two different numbers. A caller that
# wants this mode passes `n_reprojection_attempts=
# CALIBRATION_N_REPROJECTION_ATTEMPTS` (or any other explicit N)
# itself. Wired into manual "Refresh calibration now" 2026-08-30 (both
# real bootstrap_calibrations() call sites in opendarts.live.server.
# AppState._refresh_calibration_blocking()) -- Start-time auto-
# calibration remains untouched at the default (1).
#
# 5 -> 10, same day, after seeing 5's real live numbers (10/10 real calibrations, mean reproj cam0 0.289px/cam1
# 0.332px/cam2 0.623px, ~24.3s/event vs ~13.4s at N=1) -- to see
# whether more attempts tightens this further.
#
# 10 -> 5, same day, REVERTED after real N=10 numbers came back WORSE,
# not better: cam0 0.304px (flat/slightly worse), cam1 0.268px
# (improved), cam2 0.707px (worse than N=5's 0.623px -- the one camera
# this feature exists to help), for +13.4s/event (+55%) over N=5.
# With both real tables side by side, N=5 was judged good -- read as
# N=5 already sampling deep enough into this rig's real noise floor that
# more draws stop finding meaningfully better minimums, just adding
# cost -- not a fluke direction expected to reverse with a bigger N.
CALIBRATION_N_REPROJECTION_ATTEMPTS = 5

# Bounded per-frame wait (local-hub mode only -- see
# _capture_calibration_frames_local()) for the pump thread to actually
# produce a NEW frame before this function accepts it as the next of the
# N independent samples. Generous relative to the real measured ~32ms
# pump cadence cited above -- exists only to bound worst case (a camera
# that's stalled/disconnected mid-capture), not because a fresh frame is
# expected to routinely take this long.
CALIBRATION_FRAME_WAIT_TIMEOUT_S = 1.0
CALIBRATION_FRAME_WAIT_POLL_S = 0.01

# MINIMUM CONFIDENCE FLOOR for accepting a LIVE-DERIVED orientation hint --
# MOVED 2026-08-21 to `opendarts.calibration.ring_correlation_orientation.
# MIN_LIVE_HINT_CONFIDENCE` (imported above) so it has ONE canonical home
# shared by this live path AND the offline batch-refit paths, instead of
# being duplicated or re-measured per caller. See that constant's own
# docstring for the full real-measured derivation (verifier-agent
# finding MEDIUM-3, full 13-session corpus, 2026-08-20) -- preserved
# verbatim there, not summarized away.


def _min_calibration_frames_required(n_frames: int) -> int:
    """The "too few of the N succeeded to trust an average" floor --
    opendarts.calibration.sector_correspondence.average_correspondences()'s
    `min_required`. Reasoning: a real average needs a real majority of
    independent samples, not just "more than zero" -- with N=5 (the
    chosen default above), losing 3 of 5 frames to detection failure
    (e.g. a dart landing on the ring boundary mid-burst) means the
    remaining 2 are no longer meaningfully more trustworthy than a single
    frame, so this requires a STRICT MAJORITY (> N/2) when N > 1. N=1 is
    a degenerate case (used by tests exercising single-frame-equivalent
    behavior) where requiring more than 1 success is impossible by
    construction -- min_required is 1 in that case, i.e. today's old
    single-frame behavior exactly.

    As of 2026-09-03, this is used ONLY for `_try_solve_from_detections()`'s
    own genuinely fixed-size, non-accumulating single BEST-OF-N batch --
    see `_min_good_frames_cumulative()` immediately below for the
    ever-growing-pool case (`_try_solve()` and the three `_resolve_*`
    derivation helpers), which this strict-majority rule is NOT correct
    for."""
    if n_frames <= 1:
        return 1
    return (n_frames // 2) + 1


CALIBRATION_MIN_GOOD_FRAMES_CUMULATIVE = 5 # > MIN_DETECTIONS_FOR_TRIMMING (4, see
# opendarts/calibration/sector_correspondence.py) so once this floor is met, robust
# trimming is always active -- never silently degrading to a plain untrimmed mean
# on exactly the marginal cameras this exists to help. Also matches
# CALIBRATION_N_FRAMES_DETECT's own current value (5) -- a camera that gets 100%
# yield on round 1 pays zero extra cost.


def _min_good_frames_cumulative(n_frames_captured_total: int) -> int:
    """The 'too few of this event's ACCUMULATED total to trust an average' floor
    for `_try_solve()`'s own ever-growing per-camera pool (and the three
    `_resolve_*` derivation helpers that share its `frames_captured[cam]` input).
    An ABSOLUTE floor, NOT a fraction of `n_frames_captured_total` -- unlike
    `_min_calibration_frames_required()` above (kept, unchanged, for BEST-OF-N's
    own genuinely fixed-size single-batch case), this number never grows as more
    frames are captured. A fraction-of-a-growing-pool rule is arithmetically
    unwinnable for a low-yield camera: the requirement grows every retry round
    while the achievable count grows much more slowly, so a camera whose round-1
    yield falls short can never catch up no matter how many more frames it
    captures -- see docs/DESIGN.md's DEFECT 2 entry (2026-09-03) for the real incident
    and measured recovery numbers this fix closes.

    Capped at `n_frames_captured_total` itself so a genuinely tiny total (e.g. a
    test double using `n_frames_detect < 5`) degrades to 'need literally all of
    them' rather than demanding an unreachable count."""
    if n_frames_captured_total <= 1:
        return 1
    return min(CALIBRATION_MIN_GOOD_FRAMES_CUMULATIVE, n_frames_captured_total)


def _capture_calibration_frames_local(
    hub: local_capture.LocalCameraHub,
    n_frames: int,
    *,
    poll_interval_s: float = CALIBRATION_FRAME_WAIT_POLL_S,
    per_frame_timeout_s: float = CALIBRATION_FRAME_WAIT_TIMEOUT_S,
) -> dict[int, list[np.ndarray]]:
    """Capture `n_frames` GENUINELY INDEPENDENT frames per camera, straight
    from `hub`'s in-memory pump cache -- no PNG round trip (see
    fetch_current_frames()'s own PERFORMANCE FIX docstring section above
    for why that round trip is measured-wasteful; doing it N times here,
    once per calibration frame, would multiply an already-known-avoidable
    cost for no benefit -- this function reads hub.grab_all() directly,
    same as that fix already does for the hot per-iteration loop).

    "Genuinely independent" is the real, load-bearing part: LocalCameraHub
    (opendarts.live.local_capture, see its own module docstring) reads every
    camera on ONE dedicated pump thread and serves every consumer
    (grab()/grab_all()) from a cache -- back-to-back grab_all() calls with
    no wait between them would almost certainly return the SAME cached
    frame twice (a Python loop iterates in microseconds; the pump's own
    cap.read() cycle is paced by the camera's real frame rate, measured
    elsewhere in this file at ~31fps / ~32ms per cycle -- see
    POLL_INTERVAL_SECONDS's own dated comment), which would silently
    defeat the entire point of this fix (averaging N copies of the
    identical frame reduces zero real noise). This function instead
    blocks (bounded by `per_frame_timeout_s`) between rounds until each
    camera's `status[i].frame_count` has actually incremented past its
    value at the start of that round -- proof a new pump cycle genuinely
    landed for that camera, not a guessed sleep duration. If a camera
    fails to produce a fresh read within the timeout (stalled/disconnected
    mid-capture), this proceeds anyway with whatever grab_all() currently
    has for it rather than blocking the other cameras' capture
    indefinitely -- that round's sample for the lagging camera may be a
    near-duplicate of the previous round (degraded independence, not
    correctness: a duplicate detection just contributes a less-independent
    sample to the average, it does not produce a wrong one).

    Returns dict[cam, list[frame]] -- a camera missing entirely from
    hub.status (never configured) is simply absent; a camera whose
    cap.read() failed on some rounds just has fewer than n_frames entries
    (the partial-failure-among-N handling this whole change is required
    to support -- see bootstrap_calibrations()'s
    own docstring for what happens to a short list downstream).
    """
    n_cams = len(hub.configs)
    frames_by_cam: dict[int, list[np.ndarray]] = {i: [] for i in range(n_cams)}
    last_seen_count = {i: hub.status[i].frame_count for i in range(n_cams)}
    for _round in range(n_frames):
        deadline = time.monotonic() + per_frame_timeout_s
        while True:
            all_fresh = all(
                hub.status[i].frame_count > last_seen_count[i] for i in range(n_cams)
            )
            if all_fresh or time.monotonic() >= deadline:
                break
            time.sleep(poll_interval_s)
        frame_snapshot = hub.grab_all()
        for i in range(n_cams):
            frame = frame_snapshot.get(i)
            if frame is not None:
                frames_by_cam[i].append(frame)
            last_seen_count[i] = hub.status[i].frame_count
    return frames_by_cam




# CONCURRENT-CALIBRATION LOCK, added 2026-08-21 (verifier finding on
# this task's own live-wiring pass, real -- not hypothetical): this
# process has TWO independent real call sites into
# bootstrap_calibrations() -- run_capture_loop_body()'s own Start-time
# auto-calibrate step, and opendarts.live.server.AppState.
# _refresh_calibration_blocking() (the dashboard's manual "Refresh
# calibration now" button, invoked via asyncio.to_thread -- a real OS
# thread, same process). Nothing previously serialized them: only
# single-tab CLIENT-SIDE button disabling protected the manual path, and
# nothing at all protected Start's auto-calibrate racing a manual
# refresh from a second tab. Since this task's own Phase 2 wiring, a
# real bootstrap run mutates several MODULE-LEVEL globals as a side
# effect (opendarts.capture.throw_trigger's derived motion thresholds --
# see this function's own "LIVE-DERIVED MOTION THRESHOLDS" section
# below -- plus the orientation-hint/resolution state already wired
# before this task), so two overlapping bootstrap runs interleaving
# their own captures/detections could race on those globals, each
# overwriting the other's in-progress derivation with a torn or stale
# result. A `threading.Lock` (not `asyncio.Lock` -- both real call sites
# run on real OS threads, not necessarily the same asyncio event loop)
# guards the actual work below.
#
# A DEDICATED exception type, not a bare RuntimeError, because the two
# real call sites need to react to lock contention DIFFERENTLY, and a
# bare RuntimeError would be indistinguishable from every other real
# calibration failure at either site:
# - `AppState._refresh_calibration_blocking()` already wraps its own
# call in a broad `except Exception` and reports any failure as a
# non-fatal `calibration_error` string on the response -- this
# exception type is caught by that broad except exactly like any
# other failure, no change needed there, and correctly non-fatal.
# - `run_capture_loop_body()`'s own call site has NO such broad catch
# -- any exception it lets through is caught by `opendarts.live.
# run_product.py`'s `_capture_thread_target()`, which treats EVERY
# exception from a capture-loop session as fatal and brings down
# the WHOLE SERVER PROCESS (`except Exception: ... break # fatal --
# fall through to whole-process shutdown`, relying on the rig's
# external `while true` restart loop per docs/DESIGN.md). That posture is
# correct and intentional for a real environment failure ("no
# camera calibrated" -- see this function's own next RuntimeError
# just below), but is a wildly disproportionate reaction to a
# benign, transient lock race that resolves itself in a few
# seconds. This dedicated type lets that call site catch ONLY the
# transient case and retry briefly instead of crashing the whole
# process over a UI click's bad timing -- see
# `_bootstrap_calibrations_with_lock_wait()` below, which
# `run_capture_loop_body()` actually calls.
class CalibrationInProgressError(RuntimeError):
    """Raised by `bootstrap_calibrations()` when another calibration is
    already running in this process (see the module comment above this
    class). A transient, expected-to-resolve-itself condition -- NOT the
    same severity as a real calibration failure (no camera in view,
    etc.), which still raises a plain `RuntimeError`/other exception."""


# ---------------------------------------------------------------------
# ORIENTATION
#
# Board orientation is resolved once per calibration event, up front,
# by `opendarts.calibration.ring_correlation_orientation`. A camera
# that cannot be resolved is either predicted from this rig's learned
# ring geometry (`rig_ring_geometry`) or the whole calibration is
# REFUSED -- never silently substituted. See
# `_resolve_orientation_via_ring_correlation()` below.
# ---------------------------------------------------------------------
ORIENTATION_METHOD_RING_CORRELATION = "ring_correlation"

ORIENTATION_METHOD: str = ORIENTATION_METHOD_RING_CORRELATION

# A real, clean, independently-adjustable parameter (not hardcoded into
# the retry loop) so a fixed N=1/3/5/10 configuration can be compared
# against real live behaviour by changing this constant alone.
RING_CORRELATION_ORIENTATION_N_FRAMES = 3

# Every frame in the pass must agree. This is the adjustable parameter a
# looser configuration would lower.
RING_CORRELATION_ORIENTATION_MIN_PASS_FRACTION = 1.0

# Up to 10 retries; after that, fail loudly.
RING_CORRELATION_ORIENTATION_MAX_RETRIES = 10
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# CROSS-CAMERA ELLIPSE-ASPECT VALIDATION, 2026-09-03 -- a DEFENSE-IN-
# DEPTH tripwire, NOT the mechanism that produces the cam0 fix itself
# (see `opendarts.calibration.landmark_detection._mask_for_detection()`'s
# own docstring for the real fix -- a two-pass, ROI-restricted adaptive
# colour segmentation, and its follow-up investigation for the full root-cause
# story). History worth keeping honest, not silently dropped: an earlier
# version of this whole task built a DIFFERENT primary fix (a reference-
# centre correction inside `_trace_outer_boundary()`, keyed off a
# "merged double+treble ring component" hypothesis) plus this same
# tripwire as its stated validation layer. Both the hypothesis and that
# fix were independently investigated further and REFUTED/superseded: a
# genuine double+treble MERGE does not occur (treble is a separate
# connected component in 600/600 real frames tested in that follow-up);
# the real defect is upstream, in `adaptive_ring_color_mask()`'s own
# whole-frame Otsu saturation threshold being dragged up by the dark
# board surround, starving the ring's own dim paint just enough to break
# 8-connectivity and fragment `_outer_ring_component_mask()`'s selection.
# The reference-centre fix was a real, measurable partial mitigation of
# a downstream symptom, not a fix for this actual cause -- REVERTED, not
# shipped, once the real fix was found. This tripwire survives as a
# genuinely complementary safety net for whatever residual bias might
# still survive the real fix on a future frame/camera, not a validation
# layer for the reverted fix.
#
# THE SIGNAL: on this rig's fixed 3-camera ring, every camera images the
# SAME physical double ring, so a genuinely correct final-ellipse aspect
# ratio (major/minor) should be close across cameras in the SAME
# calibration event -- the small remaining spread is real per-camera
# viewing-angle foreshortening, not noise. A camera whose own aspect this
# round is an outlier relative to its currently-confident siblings is a
# real, measured signal of a still-bad seed ellipse.
#
# THRESHOLD -- taken directly from the parallel investigation's own
# stated falsification criterion ("flag if any camera's final-ellipse
# aspect sits more than 0.05 from the median of the present cameras"),
# not re-derived independently, per that investigation's own explicit
# instruction to use exactly this number. Cross-checked, not blindly
# trusted, against this task's own real measurement: with the real ROI
# fix applied, per-camera round-level (N=5 frames/round, matching this
# rig's live production round size) deviation from the same-round group
# median across all 6 real packages available on this machine (`~/
# the calibration corpus, both the 2 packages previously
# known to fail AND 4 known-healthy ones -- all of them now genuinely
# healthy once the real fix is applied, so none of them can reproduce
# the investigation's own quoted 0.09-0.13 failure-floor number on this
# machine) tops out at 0.033 -- consistent with, and independently
# corroborating, that investigation's own quoted healthy ceiling
# (<=0.03). 0.05 gives ~1.5x margin above this real healthy ceiling and
# (trusting the investigation's own quoted failure-floor measurement,
# not independently reproducible here since the real fix already closes
# every locally-available failing example) ~1.8-2.6x margin below the
# quoted 0.09-0.13 failure floor.
ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD = 0.05

# Real, measured: <2 currently-present, currently-confident cameras
# cannot mean anything for a cross-camera comparison -- there is no
# sibling to compare against. Matches this project's own established
# "fewer-than-2-anchors" no-op convention (see `opendarts.calibration.
# rig_ring_geometry.resolve_rig_consensus_orientation()`'s own handling
# of the identical situation). A validated single-camera fallback check
# was investigated and explicitly NOT found -- this is a known, accepted
# gap for the <2-camera case, not solved here.
MIN_CAMERAS_FOR_ELLIPSE_ASPECT_CHECK = 2

# CORROBORATION, 2026-09-13. The cross-camera deviation above is not, by
# itself, evidence of a bad ellipse -- it assumes every camera images the
# ring from a similar enough angle that a correct fit gives a similar
# aspect. Measured on the Windows rig's 3-camera wheel, that assumption
# is false: one camera sits materially more face-on than its siblings and
# reads 1.616 against their 1.675-1.679, a deviation of 0.058 that
# straddles the 0.05 gate and so fires on roughly half of all
# calibrations. Its ellipse fit was checked visually against the actual
# double ring and is CORRECT -- see the aspect/fit probe written for this
# investigation. The gate was firing on good data.
#
# What separates the two cases is STABILITY, not magnitude. The defect
# this gate exists for is a fragmenting ring mask (adaptive_ring_color_
# mask()'s whole-frame Otsu saturation threshold starving the ring's dim
# paint until 8-connectivity breaks); a mask that fragments does so
# frame-by-frame, so the camera's own aspect scatters. A camera that is
# merely mounted differently is perfectly self-consistent. Measured over
# 8 frames per camera on the real rig, EVERY camera -- including the
# 0.058 outlier -- held a within-camera spread of 0.0049-0.0081
# (stdev 0.0014-0.0024). 0.02 is ~2.5x that measured healthy ceiling.
#
# HONEST LIMIT: the healthy side of this number is measured, the failing
# side is not -- no fragmenting camera was available to measure, and the
# mechanism (a per-frame threshold on a static scene) could in principle
# fragment consistently. That is why the certainty band below exists:
# above the original investigation's own quoted failure floor the check
# fires regardless of stability, so a consistent fragmenter is still
# caught on magnitude alone.
ELLIPSE_ASPECT_INSTABILITY_THRESHOLD = 0.02
# The parallel investigation's own quoted failure floor (0.09-0.13). At
# or above it, a deviation is treated as a defect whatever its stability
# -- 3x the measured healthy ceiling of 0.033 and far outside anything
# camera placement on one rig has produced.
ELLIPSE_ASPECT_CERTAIN_DEFECT_DEVIATION = 0.09
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# RING-CORRELATION ORIENTATION -- resolved once per event, before any
# detection runs. Each camera's frames are answered independently and
# must AGREE; a camera that cannot answer is either predicted from the
# rig's learned ring geometry or the whole calibration is refused.
# Hint sources are reported as `"ring_correlation_live"` /
# `"ring_correlation_rig_consensus"`.
# ---------------------------------------------------------------------


def ring_geometry_for_this_event(
    ring_geometry: "RingGeometry | None",
    hints_deg: "dict[int, float]",
    now_utc: str,
) -> "tuple[RingGeometry, float | None, dict[str, Any] | None]":
    """The rig's ring geometry after this calibration event, plus a note
    when the cameras turned out to have MOVED.

    Returns (geometry, drift_deg, relearned_note). The note is None for an
    ordinary update.

    RELEARN RATHER THAN REFUSE, 2026-09-17. Gaps drifting past
    `RING_GEOMETRY_DRIFT_THRESHOLD_DEG` used to refuse the whole
    calibration and tell the operator to delete this rig's learned
    geometry -- later a dashboard button that did exactly that. The answer
    was always yes, because a camera really had been moved, so the drift
    is now reported and the layout relearned from this event.

    Safe to adopt: the caller only reaches this when EVERY camera derived
    its own orientation live from this event's frames, so none of them
    leaned on the stored layout, and the calibration itself is already
    solved. The drift is the only evidence a camera moved, which is why it
    comes back as a note the log and the dashboard both show rather than
    being absorbed silently.
    """
    updated, drift_deg = update_ring_geometry(ring_geometry, hints_deg, now_utc)
    if (
        ring_geometry is None
        or drift_deg is None
        or drift_deg <= RING_GEOMETRY_DRIFT_THRESHOLD_DEG
    ):
        return updated, drift_deg, None

    previous_gaps = [round(x, 2) for x in ring_geometry.gaps_deg]
    reseeded, _ = update_ring_geometry(None, hints_deg, now_utc)
    note = (
        f"gaps shifted {drift_deg:.1f}deg from the learned layout -- relearned it. "
        f"Check the mounts if nobody moved a camera."
    )
    return reseeded, drift_deg, {
        "drift_deg": round(drift_deg, 2),
        "previous_gaps_deg": previous_gaps,
        "gaps_deg": [round(x, 2) for x in reseeded.gaps_deg],
        "at_utc": now_utc,
        "note": note,
    }


class OrientationConsensusRefusedError(RuntimeError):
    """Raised by `bootstrap_calibrations()` -- the ring-consensus
    orientation rule R1, decided 2026-08-29:
    no hardcoded fallback orientation -- make the user act, or refuse to
    calibrate. Raised when a camera's board orientation cannot be
    established -- neither from its own live derivation, nor (when this
    rig's ring geometry is known) from rig-consensus prediction against
    the other cameras (`opendarts.calibration.rig_ring_geometry`), nor
    (when every camera DID derive live) because the observed ring
    geometry drifted past `RING_GEOMETRY_DRIFT_THRESHOLD_DEG` from what
    this rig has learned (a likely camera-moved-on-the-ring event).

    Deliberately NOT caught anywhere inside `_bootstrap_calibrations_
    unlocked()` itself -- it propagates all the way out, exactly like
    every other real calibration failure (a PnP solve error, "no camera
    calibrated at startup") already does: `run_capture_loop_body()`'s
    own Start-time bootstrap treats ANY exception here as fatal to the
    whole process (by design -- see that call site's own comment), and
    `opendarts.live.server.AppState._refresh_calibration_blocking()`'s
    manual "Refresh calibration now" catches it generically and surfaces
    `str(exc)` as `calibration_error` in the dashboard response -- this
    exception's own message is written to BE that operator-facing text
    (naming the camera, its best candidate, its confidence, and the
    floor -- spec section 3's own required shape), not a stack trace for
    a human to translate. No calibration is ever written when this
    fires -- it is always raised before `save_calibration_package()`/
    `CalibrationStore.set()` are ever reached."""


_BOOTSTRAP_CALIBRATIONS_LOCK = threading.Lock()


def bootstrap_calibrations(
    snapshot_dir: Path,
    *,
    hub: local_capture.LocalCameraHub | None = None,
    n_frames: int = CALIBRATION_N_FRAMES,
    n_frames_detect: int = CALIBRATION_N_FRAMES_DETECT,
    target_reprojection_error_px: float | dict[int, float] = CALIBRATION_TARGET_REPROJECTION_ERROR_PX,
    max_frames: int = CALIBRATION_MAX_N_FRAMES,
    retry_batch_size: int = CALIBRATION_RETRY_BATCH_SIZE,
    n_reprojection_attempts: int = 1,
    diagnostics_out: "dict[int, dict[str, Any]] | None" = None,
    calibration_package_root: "Path | None" = None,
    throw_package_root: "Path | None" = None,
    calibration_package_out: "dict[str, Any] | None" = None,
    calibration_package_blocking: bool = False,
) -> dict[int, CameraCalibration]:
    """Thin locking wrapper around `_bootstrap_calibrations_unlocked()`
    (same function this project has always had, renamed) -- see the
    `_BOOTSTRAP_CALIBRATIONS_LOCK` module comment just above for why this
    exists. Raises `CalibrationInProgressError` immediately (never
    blocks) if a calibration is already running in this process --
    every real caller (`run_capture_loop_body()`'s auto-calibrate via
    `_bootstrap_calibrations_with_lock_wait()`, `AppState.
    _refresh_calibration_blocking()`'s manual refresh via its own broad
    `except Exception`) already has its own reasonable place to handle
    that specific, transient case rather than either hanging or treating
    it as a hard failure. Forwards every argument unchanged -- this
    wrapper adds no new behavior beyond the lock itself.

    `n_reprojection_attempts` (added 2026-08-30, default 1 = today's
    exact prior behavior for every existing caller) -- see
    `_bootstrap_calibrations_unlocked()`'s own "BEST-OF-N REPROJECTION
    ATTEMPTS" docstring section and `CALIBRATION_N_REPROJECTION_
    ATTEMPTS`'s own dated module comment above."""
    if not _BOOTSTRAP_CALIBRATIONS_LOCK.acquire(blocking=False):
        raise CalibrationInProgressError(
            "a calibration bootstrap is already running in this process -- "
            "try again once it finishes (this guards against two concurrent "
            "triggers, e.g. two dashboard tabs, or Start's auto-calibrate "
            "racing a manual Refresh, interleaving writes to shared "
            "live-derived state)"
        )
    # How far along it is, for the screens waiting on it
    # (opendarts.live.calibration_progress -- a report only).
    calibration_progress.PROGRESS.start(
        list(range(len(getattr(hub, "configs", None) or []))) if hub is not None else None)
    _progress_ok = False
    _progress_error: str | None = None
    try:
        result = _bootstrap_calibrations_unlocked(
            snapshot_dir,
            hub=hub,
            n_frames=n_frames,
            n_frames_detect=n_frames_detect,
            target_reprojection_error_px=target_reprojection_error_px,
            max_frames=max_frames,
            retry_batch_size=retry_batch_size,
            n_reprojection_attempts=n_reprojection_attempts,
            diagnostics_out=diagnostics_out,
            calibration_package_root=calibration_package_root,
            throw_package_root=throw_package_root,
            calibration_package_out=calibration_package_out,
            calibration_package_blocking=calibration_package_blocking,
        )
        _progress_ok = bool(result)
        if not result:
            _progress_error = "no camera could be calibrated"
        return result
    except Exception as exc:
        _progress_error = str(exc)
        raise
    finally:
        calibration_progress.PROGRESS.finish(_progress_ok, _progress_error)
        _BOOTSTRAP_CALIBRATIONS_LOCK.release()
        # On success everything the calibration allocated is unreferenced
        # by now -- `_bootstrap_calibrations_unlocked()`'s frame is already
        # torn down -- but glibc keeps the freed heap. See
        # opendarts.live.heap_trim for the measurement. (On a RAISE the
        # in-flight traceback still pins that frame, and its nested
        # helpers' closures, so this trim frees little; the caller that
        # swallows the exception trims again -- see AppState.
        # _refresh_calibration_blocking().)
        release_freed_heap("calibration")


# How long run_capture_loop_body()'s own Start-time auto-calibrate will
# wait for a competing calibration (a manual "Refresh calibration now"
# that happened to win the lock race) to finish before giving up and
# letting CalibrationInProgressError propagate as a real (still fatal,
# per this whole file's established posture -- see the class's own
# docstring) failure. 30s is comfortably above a real bootstrap's normal
# duration (CALIBRATION_N_FRAMES-scale capture + detection, seconds) but
# still bounded -- Start must not hang indefinitely if something is
# genuinely stuck holding the lock.
CALIBRATION_LOCK_WAIT_TIMEOUT_S = 30.0
_CALIBRATION_LOCK_POLL_INTERVAL_S = 0.5


def _bootstrap_calibrations_with_lock_wait(
    *args,
    lock_wait_timeout_s: float = CALIBRATION_LOCK_WAIT_TIMEOUT_S,
    **kwargs,
) -> dict[int, CameraCalibration]:
    """`run_capture_loop_body()`'s own entry point into
    `bootstrap_calibrations()` -- retries briefly on
    `CalibrationInProgressError` instead of letting it propagate
    immediately, because THIS call site has no graceful-degrade handling
    of its own (any exception it raises is fatal to the whole process,
    per `CalibrationInProgressError`'s own class docstring) and the lock
    condition it's retrying on is expected to be transient (a few
    seconds, the duration of whatever OTHER calibration currently holds
    it). Still bounded: after `lock_wait_timeout_s` of real waiting, lets
    the final `CalibrationInProgressError` propagate for real -- a lock
    held that long is no longer a benign race, and the existing fatal/
    external-restart posture is the right answer to a genuinely stuck
    process, not something this retry should paper over forever."""
    deadline = time.monotonic() + lock_wait_timeout_s
    while True:
        try:
            return bootstrap_calibrations(*args, **kwargs)
        except CalibrationInProgressError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_CALIBRATION_LOCK_POLL_INTERVAL_S)


def _bootstrap_calibrations_at_start_with_reopen_retry(
    *args,
    hub: "local_capture.LocalCameraHub | None",
    **kwargs,
) -> dict[int, CameraCalibration]:
    """`run_capture_loop_body()`'s own Start-time entry point into
    `_bootstrap_calibrations_with_lock_wait()` -- ONE bounded real camera
    close+reopen (`hub.open_all()`) + a single retry, ONLY here, ONLY at
    Start. Never applied to a manual mid-session "Refresh calibration
    now" (that call site still uses `_bootstrap_calibrations_with_lock_
    wait()` directly, unchanged) -- see this function's own "why Start
    only" section below for why closing real camera handles is safe
    exactly here and nowhere else.

    **The real incident this fixes** (independently confirmed as the
    right shape for a real problem, not copied without scrutiny): a
    calibration event solved 2 of 3 cameras cleanly, but the
    third was silently absent from the result -- not refused, not
    errored, just missing -- DESPITE a real, successful raw capture (a
    full frame pool, no camera-open failure at all). Five independent
    best-of-N landmark-detection attempts against five FRESH frame
    batches, all within the SAME process, all failed to produce usable
    landmarks for that one camera. A manual Stop -> Start -> Recalibrate
    (a REAL camera close+reopen, via a fresh process) fixed it
    immediately, on the very next attempt. That's the actual signature:
    not "this camera's view is bad" (more frames/more attempts already
    ruled that out, 5 times over) but "this camera's OWN in-process
    handle/driver state is stuck," a class of failure purely-optical
    retries structurally cannot fix, because they never touch the thing
    that's actually broken.

    **Why Start only, not a manual mid-session Recalibrate too**: at
    Start, `hub.open_all()`/`close_all()` are the ONLY things touching
    these camera handles so far -- the live capture/pump loop has not
    started consuming them yet, so a close+reopen here is free (no
    in-flight reads to race, no active session to disrupt). A manual
    Recalibrate happens mid-session, with the pump loop ACTIVELY
    depending on those same open handles for live scoring the whole
    time it runs -- closing and reopening them there would be a
    materially different, riskier guarantee to break (a real capture in
    progress could lose frames or crash outright), and is explicitly
    OUT OF SCOPE for this fix.

    **Detection**: after the first attempt, compares the number of
    cameras actually calibrated against `len(hub.configs)` (every
    LOCALLY configured camera is expected to calibrate -- HTTP-fallback
    mode, `hub is None`, has no local handles to reopen at all and
    unconditionally skips this whole mechanism, falling through to the
    plain, unmodified `_bootstrap_calibrations_with_lock_wait()` call).
    Also triggers on ANY exception from the first attempt (not just a
    plain "fewer cameras than expected" result) -- a stuck camera could
    just as easily manifest as this rig's own `OrientationConsensusRefusedError`
    (spec R1's own "refuse rather than silently guess" -- a camera that
    never produces a usable landmark can also never establish an
    orientation hint) as it could a silently-partial result; both are
    real, reachable shapes of "this camera's handle is stuck," and both
    get the identical one-retry treatment.

    **Bounded, not a loop**: exactly ONE retry, ever, per Start attempt.
    If the SECOND attempt (after the real close+reopen) still comes back
    partial or raises, that outcome -- success, partial, or the
    exception -- is returned/propagated as final, unchanged from what
    would have happened with no retry mechanism at all. This is
    deliberately NOT a "keep trying forever" loop: a rig with a
    genuinely bad camera (optical, not driver-state) must still surface
    that clearly (partial calibration, or a real refusal) rather than
    silently retrying into a false sense that it will eventually work.
    """
    # `hasattr` duck-typed capability check, not an isinstance check --
    # matches this project's own established convention elsewhere (e.g.
    # `engine_accepts_prior_dart_line_px()`) for the same reason: several
    # real tests substitute a lightweight fake hub that implements only
    # the surface ITS OWN test actually needs, not the real
    # LocalCameraHub's full interface. `hub is None` (HTTP-fallback) or a
    # hub missing either `.configs` (needed to know the expected camera
    # count) or `.open_all` (needed to actually reopen) both correctly
    # fall through to the plain, unmodified call -- "cannot determine
    # whether this retry would even be meaningful" degrades safely to
    # "don't attempt it," never to a crash on a test double or a future
    # hub implementation this fix wasn't written against.
    if hub is None or not hasattr(hub, "configs") or not hasattr(hub, "open_all"):
        return _bootstrap_calibrations_with_lock_wait(*args, hub=hub, **kwargs)

    expected_n_cameras = len(hub.configs)
    reopen_reason: str | None = None
    try:
        calibrations = _bootstrap_calibrations_with_lock_wait(*args, hub=hub, **kwargs)
    except Exception as exc: # noqa: BLE001 -- real, deliberate catch-and-retry-once; see docstring
        reopen_reason = f"raised {exc!r}"
    else:
        missing = expected_n_cameras - len(calibrations)
        if missing <= 0:
            return calibrations
        reopen_reason = f"{missing}/{expected_n_cameras} camera(s) missing from the result"

    log.warning(
        "Start-time calibration: %s -- this is the real, confirmed 'stuck "
        "camera driver/handle state' incident class (see this function's "
        "own docstring), where more purely-optical retries never help but a "
        "real camera close+reopen does. Trying ONE bounded hub.open_all() "
        "(real close+reopen, ~2.2-2.3s/camera measured, concurrent across "
        "cameras) + a single retry before falling back to today's existing "
        "behavior (partial calibration, or a real refusal).",
        reopen_reason,
    )
    ok_flags = hub.open_all()
    log.info(
        "Start-time calibration retry: reopened %d/%d camera(s) (ok_flags=%s), "
        "retrying bootstrap once.",
        sum(ok_flags), len(ok_flags), ok_flags,
    )
    return _bootstrap_calibrations_with_lock_wait(*args, hub=hub, **kwargs)


def _bootstrap_calibrations_unlocked(
    snapshot_dir: Path,
    *,
    hub: local_capture.LocalCameraHub | None = None,
    n_frames: int = CALIBRATION_N_FRAMES,
    n_frames_detect: int = CALIBRATION_N_FRAMES_DETECT,
    target_reprojection_error_px: float | dict[int, float] = CALIBRATION_TARGET_REPROJECTION_ERROR_PX,
    max_frames: int = CALIBRATION_MAX_N_FRAMES,
    retry_batch_size: int = CALIBRATION_RETRY_BATCH_SIZE,
    n_reprojection_attempts: int = 1,
    diagnostics_out: "dict[int, dict[str, Any]] | None" = None,
    calibration_package_root: "Path | None" = None,
    throw_package_root: "Path | None" = None,
    calibration_package_out: "dict[str, Any] | None" = None,
    calibration_package_blocking: bool = False,
) -> dict[int, CameraCalibration]:
    """Real, working calibration bootstrap -- NOT a stub, and NO LONGER a
    single-frame solve (fixed 2026-08-12 -- see CALIBRATION_N_FRAMES'
    dated comment above for the full real-evidence writeup, real measured
    noise-reduction numbers, the N=5 reasoning, and the honest "random
    noise, not systematic bias" limitation). Captures `n_frames`
    independent frames per camera, runs landmark detection +
    correspondence (opendarts.calibration.oriented_landmarks.
    correspond_landmarks_oriented()) on EACH frame independently, averages
    the resulting image_points_px per landmark index across whichever
    frames succeeded (opendarts.calibration.sector_correspondence.
    average_correspondences() -- see its own docstring for why averaging
    by landmark index is valid here, a real verified fact not an
    assumption), then solves PnP ONCE (opendarts.pipeline.calibrate_camera(),
    genuinely unchanged by this fix) on the averaged points, instead of
    the old single noisy per-frame solve.

    **LANDMARK SOURCE, changed 2026-08-13**: this used to call
    `sector_correspondence.detect_and_correspond()`, which picks its 4
    landmark pixels by looking up a FIXED per-camera reference angle on
    the fitted ring ellipse -- structurally mis-registered, because a
    perspective projection does not preserve the board's equal 18-degree
    wire spacing (see `oriented_landmarks`'s own module docstring for the
    full derivation of why). It now calls
    `oriented_landmarks.correspond_landmarks_oriented()` instead, which
    solves for the board's real rotational phase per image. Shape-
    compatible drop-in: same `(object_points_mm, image_points_px)`
    return, same 4-landmark index order, so `average_correspondences()`
    and everything downstream is untouched. Real measured effect on
    `data/archive/clean/`'s 169 AD-matched throws, through THIS function
    (not a standalone harness): Apollo 81.7% -> 96.4% and Athena
    84.6% -> 89.3% sector+ring BOTH-match. The per-camera orientation
    hints it needs used to be a one-time-per-rig bootstrap CONSTANT
    written into the source, that had to be re-derived by hand if a
    camera ever physically moved. See
    "LIVE-DERIVED ORIENTATION HINT" immediately below for what replaced
    that.

    **LIVE-DERIVED ORIENTATION HINT, changed 2026-08-20 -- applies
    the calibration-bootstrap rule (every calibration input is derived
    from this rig's own frames) to the one
    real call site that still used a hand-maintained per-camera
    orientation-hint constant. This function now derives each
    camera's `orientation_hint_deg` from the SAME frames it is already
    capturing this bootstrap pass, using
    the orientation solve
    -- the regulation number ring's digit-count signature (identical on
    every real board, no template, no external data) -- instead of
    a constant. Real measurement (`dev/calibration/
    validate_number_ring_session_bootstrap.py`, full `data/archive/clean/`
    corpus, all 13 sessions x 3 cameras x 1107 throws): the live-derived
    hint reproduces the EXACT SAME `lock_orientation().roll` decision the
    old constant would have, 39/39 sessions x cameras (100%),
    3321/3321 underlying frames (100%).

    **Two-pass restructuring this required** (least invasive to the
    existing per-round capture/retry loop --
    the alternative, a whole-second-capture-pass design, would have
    doubled real capture latency for no benefit): for each round's newly
    captured frames, `_detect_batch()` below first runs ONLY the
    pre-`lock_orientation()` stages
    (`oriented_landmarks.locate_pre_orientation_landmarks()` -- seed
    ellipse -> bull -> reseat -> phase lock, none of which depends on the
    hint) and buffers that geometry plus this frame's own contribution
    to the orientation solve. The moment a camera has enough buffered
    evidence to produce a confident
    `ring_correlation_orientation_for_camera()` hint (in practice: after
    its first captured batch, `n_frames`-sized, comfortably more than the corpus validation above needed for 100%
    match), that hint is FROZEN for the remainder of this bootstrap call
    -- reused for every already-buffered frame's own (cheap, hint-only)
    finishing stage (`oriented_landmarks.correspond_landmarks_from_pre_orientation()`
    -- `lock_orientation()` + wire refine + quality gates, run ONCE per
    frame, not twice; the pre-orientation detection itself, the real-cost
    part, never re-runs) and for every subsequent retry round's new
    frames alike. This mirrors exactly how the old hardcoded constant was
    used (one fixed hint for the whole bootstrap call), just sourced from
    this call's own frames instead of a file.

    **No hardcoded fallback constant anymore -- RIG-CONSENSUS or REFUSE
    (spec R1/R2, 2026-08-29).** `MEASURED_CAMERA_ORIENTATION_HINTS_DEG`
    is deleted from every live code path. Live
    per-camera derivation is still the FIRST thing tried (unchanged) --
    but when this rig has already learned its own ring geometry from a
    prior all-live calibration (`opendarts.calibration.rig_ring_geometry`),
    a camera that cannot self-derive is predicted from the OTHER
    cameras' own hints + that learned geometry instead (spec R2's
    "rig-consensus", not a fallback branch -- applied on EVERY
    calibration once geometry exists, including cross-checking cameras
    that DID clear the confidence floor, since a marginal live pass is
    not, by itself, evidence of being right on this rig -- see that
    module's own docstring for the D2 alias-detection reasoning). A
    camera that can neither self-derive NOR be predicted (no ring
    geometry yet, or fewer than two mutually-agreeing anchors this
    event) makes the WHOLE bootstrap call REFUSE
    (`OrientationConsensusRefusedError`) rather than silently proceeding
    with fewer cameras -- see that exception's own docstring.
    `diagnostics_out` (see its own docstring section below) records
    `orientation_hint_source` (`"live"` / `"rig_consensus"`) and
    `orientation_hint_deg` per camera so a caller can tell which path a
    given calibration actually took, not just that it succeeded.

    **Real, related behaviour change worth naming explicitly**: the old
    per-camera "no measured orientation hint, skipping" gate (checked
    BEFORE any frame was captured, against the fixed constant's own known
    camera indices) is gone -- it cannot exist anymore, because whether a
    camera's hint is derivable is no longer knowable before its frames
    are captured and processed (that is the whole point of deriving it
    live). Every camera present in the captured frame set is now
    attempted regardless of camera index, which is a genuine
    improvement for docs/DESIGN.md's own stated ambition ("a brand-new rig...
    could run this, ... with no code change") but does mean a camera that
    is not really part of this rig (e.g. an index reported by a
    misconfigured hub) now burns real capture/detection time before
    failing (via the existing "not enough usable frames" path once
    `max_frames` is reached with zero successful correspondences) instead
    of being turned away instantly. Judged an acceptable, honestly-stated
    tradeoff for deriving the hint live, not an oversight.

    Frame source: fetches via a persistent
    `opendarts.live.local_capture.LocalCameraHub` -- `hub` must be passed
    (already opened) in that case; this function never opens/closes a
    hub itself, since opening cameras is expensive and the whole point
    of LocalCameraHub is a hub that stays open across a caller's whole
    lifetime, not one that gets reopened per call. BOTH real call sites in
    this app go through this one function -- the Start-triggered
    auto-calibrate (run_capture_loop_body() below) and the manual
    "Refresh calibration now" button (opendarts/live/server.py's
    _refresh_calibration_blocking()) -- so both get this fix, not just
    one; see this function's own tests for an explicit assertion of that.

    Partial-frame-failure handling: a camera where SOME (not all) of a
    round's captures fail landmark detection still calibrates, from
    whatever succeeded, as long as at least
    `_min_good_frames_cumulative()` of its TOTAL accumulated frames
    (across every round so far, see ADAPTIVE RETRY below) succeeded (an
    ABSOLUTE floor, not a fraction of the growing total -- see that
    function's own docstring for the reasoning, and docs/DESIGN.md's DEFECT 2
    entry, 2026-09-03, for why the earlier strict-majority-of-the-total
    rule was arithmetically unwinnable for a low-yield camera); a camera
    that never clears that floor even after `max_frames` is skipped for
    this pass entirely (same "skip, don't crash the whole calibration"
    posture the old single-frame code already had for a fully-failed
    camera, just with a real minimum-trust floor added for the new
    multi-frame case).

    **DECOUPLED CAPTURE-VS-DETECT TARGET, added 2026-08-21** (see
    `CALIBRATION_N_FRAMES_DETECT`'s own dated module comment above for
    the full real-evidence writeup). The round loop below still captures
    a full `n_frames`-sized raw batch (default 50, cheap regardless of
    count) in ONE `_capture(n_frames)` call, but round 1 of the
    detect/solve/retry logic only runs the expensive landmark detector
    on the first `n_frames_detect` of those raw frames (default 10,
    proven sufficient for identical scored segments and a correctly-
    locked orientation hint). The remaining `n_frames - n_frames_detect`
    raw frames are appended straight into `pre_orientation_pool[cam]` as
    `(frame, None)` pairs -- never run through `locate_pre_orientation_
    landmarks()`/`correspond_landmarks_from_pre_orientation()` at all --
    purely so the two post-loop sections that only ever read raw pixels
    from that pool (ring-boundary-offset, board-color) still get the
    full ~50-frame pool they need for accuracy, without the solve/hint
    path paying detect cost for frames it doesn't need. If the target
    isn't met at `n_frames_detect`, the EXISTING retry mechanics below
    take over completely unchanged (capture+detect `retry_batch_size`
    more real frames, up to `max_frames`) -- this only changes what
    round 1 initially detects, not any retry/acceptance logic.

    **ADAPTIVE RETRY, added 2026-08-14** (see CALIBRATION_TARGET_
    REPROJECTION_ERROR_PX's own dated module comment for the full real-
    evidence writeup): captures an initial `n_frames`-sized batch per
    camera, solves, and ACCEPTS IMMEDIATELY once
    `pnp_result.reprojection_error_px < target_reprojection_error_px`.
    `target_reprojection_error_px` may be a single float (the original,
    still-default uniform behavior) OR a `dict[int, float]` keyed by
    camera index. A dict entry missing for a
    given camera falls back to CALIBRATION_TARGET_REPROJECTION_ERROR_PX,
    same as today's global default -- see `_target_px_for()` just below,
    the one place this resolution happens.
    A camera that solves but doesn't clear the target instead captures
    `retry_batch_size` MORE frames -- accumulated into the SAME robust-
    averaged pool (`average_correspondences()` above), never restarted
    from zero -- and re-solves, repeating until either the target is met
    or the camera's own total captured frame count reaches `max_frames`.
    At the cap without reaching target, the BEST (lowest
    reprojection_error_px) calibration seen across every round for that
    camera is accepted anyway (a degraded-but-present calibration beats
    none) -- but this is ALWAYS logged as a clear warning naming the cap
    and the best error actually reached, never silently accepted as if
    it had met the target. A camera whose PnP solve fails outright
    (`attempt.ok is False`) is NOT retried (more frames of the same
    board scene are very unlikely to fix a degenerate correspondence) --
    same immediate-skip posture as before this change. A camera whose
    `pnp_result` is unavailable (only possible via a caller that bypasses
    the real `calibrate_camera()`, e.g. a test double) is also accepted
    immediately, since there is nothing to evaluate against the target --
    this preserves every existing single-solve caller's behavior exactly.

    **BEST-OF-N REPROJECTION ATTEMPTS, added 2026-08-30** -- a DIFFERENT acceptance
    criterion from ADAPTIVE RETRY above, opt-in via `n_reprojection_
    attempts` (default 1 = today's exact prior behavior, zero change for
    any existing caller). ADAPTIVE RETRY accepts the FIRST attempt that
    clears `target_reprojection_error_px`; `n_reprojection_attempts > 1`
    instead ALWAYS runs that many genuinely independent detect+solve
    attempts per camera and adopts whichever had the LOWEST
    `reprojection_error_px`, regardless of which attempt achieved it or
    whether any of them cleared target. The ADAPTIVE RETRY loop above
    (with its own existing "accept at target, else retry to max_frames
    and keep the best seen" behavior, completely unchanged) always
    produces this camera's own attempt #1 first; this section then runs
    `n_reprojection_attempts - 1` MORE attempts afterward, each on a
    real, fresh `_capture(n_frames_detect)` batch that no earlier
    attempt (ADAPTIVE RETRY's own rounds included) has ever analyzed --
    never the same pixels re-scored, since that would defeat the point
    of an independent noise sample. Orientation, focal length,
    distortion (k1), and principal point (cx) are all read from their
    already-established values (frozen by the time this section runs,
    since it never touches `accumulated_detections`/`accumulated_
    results` -- see `_try_solve_from_detections()`'s own docstring) --
    ONLY the pose fit (from that attempt's own averaged correspondence)
    and its resulting reprojection error vary attempt-to-attempt. This
    keeps orientation resolution and the two post-loop derived-value
    sections (ring-boundary-offset, board-color) each running EXACTLY
    ONCE per event, matching this project's own settled design for both
    (see their own docstring sections elsewhere in this file) --
    genuinely nothing about this section repeats them. Every attempt's
    real captured frames (including a batch that fails to produce a
    usable correspondence average -- see `_try_solve_from_detections()`)
    are still appended into `pre_orientation_pool[cam]`, so the two
    post-loop sections that consume that pool's raw pixels see MORE real
    data than they would without this feature enabled, not less -- a
    genuine (if incidental) accuracy bonus, not a cost.

    Per-camera independence is preserved exactly like ADAPTIVE RETRY's
    own `ThreadPoolExecutor` above -- a separate pool, one worker per
    still-eligible camera, submitted fresh each attempt round; a camera
    that already produced NO calibration at all via ADAPTIVE RETRY (a
    hard PnP failure, or every attempt exhausted without ever clearing
    `_min_good_frames_cumulative()`) is excluded from this section
    entirely -- more independent attempts do not fix a camera ADAPTIVE
    RETRY already gave up on outright, same "more frames of the same
    board scene are very unlikely to fix a degenerate correspondence"
    reasoning that section's own docstring already states. This is
    exactly what keeps ADAPTIVE RETRY's own "give up at max_frames,
    accept the best seen rather than fail outright" safety net intact as
    a fallback here too: attempt #1 (its own output) is always a real
    candidate in the pool this section compares against, so a camera for
    which none of the ADDITIONAL best-of-N attempts manages to beat it
    simply keeps ADAPTIVE RETRY's own best-seen result, unchanged --
    `n_reprojection_attempts > 1` can only ever match or improve a
    camera's adopted reprojection error, never make it worse or turn a
    real result into a failure.

    NOT wired into either live call site (`/api/calibration/refresh`,
    Start-time auto-calibration) as of this feature's own introduction
    -- see `CALIBRATION_N_REPROJECTION_ATTEMPTS`'s own dated module
    comment for why the module-level constant and this parameter's own default (1) are deliberately two
    different values.

    **PARALLELIZED ACROSS CAMERAS, added 2026-08-16** -- calibration used
    to process one camera at a time; it now calibrates all cameras
    simultaneously. Frame CAPTURE itself was already a single shared
    call across every camera per round (`_capture()` above, via
    `LocalCameraHub`'s own pump thread -- see `local_capture.py`'s
    module docstring, `grab()`/`grab_all()` are pure cache reads guarded
    by one `_cache_lock`, already safe to call once per round regardless
    of how many cameras are still retrying). What was genuinely
    sequential was the CPU-bound work AFTER capture for each round --
    landmark detection (`correspond_landmarks_oriented()`), the PnP
    solve (`calibrate_camera()`), and the accept/retry/best-tracking
    bookkeeping -- which used to run one camera at a time inside a plain
    `for cam in list(remaining):` loop. That per-camera round body is
    now `_process_camera_round()` below, submitted to a
    `ThreadPoolExecutor` (one worker per camera still in `remaining`)
    each round instead of looped sequentially.

    **HONEST REAL-SPEEDUP NUMBER, measured not assumed**: the threading
    MECHANISM itself is genuinely concurrent -- proven with a real
    `time.sleep()`-based mock in this function's own tests
    (`tests/test_capture_daemon.py::
    test_bootstrap_calibrations_processes_multiple_cameras_concurrently_not_sequentially`),
    wall time for 3 "cameras" comes back close to ONE camera's sleep
    duration, not the sum of all three. But the REAL landmark-detection +
    PnP-solve work, profiled (`cProfile`) against real archived camera
    frames (this task's own throwaway `tmp/` harness scripts, not
    shipped) and timed old-sequential-vs-new-threaded on the SAME real
    frames through the SAME real detection/solve code, measured only
    ~1.1-1.2x real wall-clock speedup -- nowhere near the ~3x a fully
    GIL-released 3-camera parallel workload would give. Root cause,
    confirmed by profiling one `correspond_landmarks_oriented()` call:
    the bulk of its cost is NOT one or two big cv2 C++ calls that release
    the GIL for a meaningful stretch -- it's thousands of tiny
    numpy/pure-Python operations (`double_colour_score()`'s per-spoke
    scoring, `refine_wire_junction()`'s per-junction search, each making
    many small `numpy.ufunc.reduce`/`.sum()`/`round()` calls), each
    individually fast but each briefly re-acquiring the GIL -- so 3
    threads mostly take turns rather than truly overlapping. This is a
    real, measured finding, not a guess: do not describe this change as
    "3x faster" anywhere -- it is a real but modest (~10-20%) wall-clock
    win, worth keeping because it's free and provably behavior-preserving
    (see the bit-for-bit-identical proof below), not because it delivers
    the ~3x a naive GIL-releases-during-cv2 assumption would predict. A
    genuinely bigger speedup would need either multiprocessing (real
    IPC/pickling cost per camera frame, explicitly the tradeoff this
    change avoids by using threads) or reworking the detection
    algorithm's own hot loops to do less small-array/pure-Python work per
    call -- both out of scope here.

    THREAD SAFETY, worked through explicitly rather than assumed: every
    per-camera accumulator here (`accumulated_detections[cam]`,
    `accumulated_results[cam]`, `frames_captured[cam]`,
    `best_calibration[cam]`, `best_reprojection_px[cam]`) is written ONLY
    by the one worker thread processing that specific `cam` -- different
    threads only ever touch DIFFERENT dict keys, which is safe under
    CPython's GIL for a single `dict[key] = value`/`list.append()`
    without a lock (the exact same reasoning `LocalCameraHub`'s own
    `_caps_lock` docstring in `opendarts/live/local_capture.py` already
    documents for this project). The ONE genuinely shared mutable object,
    `remaining` (a `set`, not a per-camera-keyed dict, so it has no
    "distinct key" safety net), is deliberately NEVER mutated from a
    worker thread -- `_process_camera_round()` only RETURNS whether its
    camera should stay in `remaining`, and the single-threaded round loop
    below applies every worker's decision itself, after every worker for
    that round has finished (`as_completed()`), by discarding straight
    off the main thread. `log`'s own calls are safe to interleave from
    multiple threads (the stdlib `logging` module locks internally per
    handler) -- every per-camera log line already carries `cam%d`, so an
    interleaved log from 3 concurrent cameras is still fully attributable
    per line, just not necessarily printed in camera-index order anymore
    (no test relies on cross-camera log ORDER -- see this function's own
    tests, which are all single-camera fixtures; a real multi-camera
    round-ordering test would be testing an implementation detail this
    function's docstring explicitly does not promise).

    NOT parallelized, and why: `_capture()` itself stays a single
    sequential call per round, never one call per camera. It was already
    effectively batched (one `_capture_calibration_frames_local()` call
    captures every camera's frame for that round together, using
    `LocalCameraHub`'s own internal per-cycle `ThreadPoolExecutor` to
    read every camera's pump in parallel -- see that module), so
    splitting it into 3 separate per-camera capture calls would only
    reintroduce cross-round synchronization complexity (rounds already
    have to wait for the SLOWEST remaining camera's fresh pump frame
    either way) for no real parallelism gain the pump doesn't already
    provide.

    `diagnostics_out`, if given (same optional results-sink convention
    `oriented_landmarks.correspond_landmarks_oriented()`'s own
    `results_out` already uses), is filled with one entry per
    successfully-calibrated camera: `{"reprojection_error_px": float |
    None, "n_frames_used": int, "n_frames_raw_pool": int, "target_met": bool,
    "orientation_hint_source": "live" | "rig_consensus" | None,
    "orientation_hint_deg": float | None}` -- so a caller (the dashboard,
    a future log line) can tell a genuinely-verified-good calibration
    apart from one that hit the frame cap without ever being confirmed
    accurate, which today's return value (a bare CameraCalibration) has
    no way to express, AND (added 2026-08-20, see "LIVE-DERIVED
    ORIENTATION HINT" above; `"rig_consensus"` added 2026-08-29, see
    R2/R1's own docstring section above) whether this camera's own
    live orientation bootstrap actually succeeded or was predicted
    from the other cameras via this rig's own learned ring geometry --
    `orientation_hint_source` is `None` only in the degenerate case
    where a camera calibrated despite never establishing any hint at
    all (a test double bypassing this path, or a camera whose PnP solve
    was fed via some other route). `n_frames_used`
    (unchanged meaning) is how many frames were actually DETECTED and
    fed the PnP solve; `n_frames_raw_pool` (added 2026-08-21, see
    DECOUPLED CAPTURE-VS-DETECT TARGET above) is the total size of this
    camera's raw frame pool -- usually bigger, since it also counts the
    `n_frames - n_frames_detect` raw-only frames the two post-loop
    sections (ring-boundary-offset, board-color) get but the detector
    never saw.

    **CALIBRATION PACKAGE, added 2026-08-20** -- see
    `opendarts.capture.calibration_package`'s own module docstring for the
    full format/rationale (REPLAY applied to calibration bootstrap
    itself, FFV1 raw-frame storage, rig-side cleanup). Opt-in via
    `calibration_package_root` (`None`, the default, means "do nothing
    new here" -- every existing caller/test is completely unaffected):
    when given, and at least one camera calibrated successfully this
    pass, a real `calib_<timestamp>` package is written under it,
    containing every frame this camera's `pre_orientation_pool` actually
    holds (2026-08-21: EVERY raw frame captured this event, across every
    retry round -- both the frames `_detect_batch()` ran landmark
    detection on AND the raw-only "extra" frames from DECOUPLED
    CAPTURE-VS-DETECT TARGET above that were captured but deliberately
    never detected; see `package_raw_frames` further down and its own
    comment for the real bug this closes -- a package used to only ever
    contain the DETECTED subset, silently smaller than the raw pool the
    live ring-boundary-offset/board-color derivations actually consumed)
    for each successfully-calibrated camera, plus that camera's solved
    calibration and the same diagnostics `diagnostics_out` above already
    exposes. `calibration_package_out`
    (same results-sink convention as `diagnostics_out`), if given, is
    filled SYNCHRONOUSLY with `{"package_id": str, "package_dir": Path}`
    before this function returns -- the id itself costs nothing to
    generate (a timestamp string) and callers need it immediately (to
    stamp onto `CalibrationStore` and from there onto every subsequent
    throw package's own `calibration_package_id`), even though the
    actual disk-heavy work (FFV1 encoding, up to ~1-3s per camera -- see
    `opendarts.capture.calibration_package`'s own module docstring's
    "MEASURED FFV1 timing" section for the real measured number, this
    module has no timing measurement of its own) happens afterward.

    **Never on the critical path** (the real constraint this whole
    feature was built under: raw-frame capture/encoding must not slow
    down or risk breaking the live calibration result): by default
    (`calibration_package_blocking=False`) the actual save runs on a
    background daemon thread (`save_calibration_package_background()`),
    fire-and-forget from this function's point of view -- every failure
    mode (missing ffmpeg, a full disk, anything) is caught and logged
    over there, never raised into this function or its caller.
    `calibration_package_blocking=True` (tests only, real live callers
    never pass this) makes the save happen inline instead, so a test can
    assert on the written package without a `Thread.join()` race.
    `throw_package_root`, if also given, is forwarded so the background
    save can run its own post-save cleanup pass (see
    `cleanup_orphaned_calibration_packages()`'s own docstring for why
    that happens here, once per real calibration event, rather than on a
    separate timer or not at all).

    What's genuinely NOT proven here, even though the underlying pieces
    are real: doing this unattended, on a fixed schedule, forever, with
    no operator watching for a silent calibration regression (e.g. the
    rig gets bumped between throws). This function calibrates ONCE, at
    startup (now from N averaged, possibly-retried frames instead of 1,
    still just once), from whatever the cameras see at that moment -- it
    does not itself decide when a re-calibration is warranted later.
    That's a real open question, not resolved by this scaffold (see
    docs/DEPLOYMENT.md Limitations). ALSO NOT PROVEN: whether N-frame
    averaging (or the retry-to-target loop) actually tightens a real live
    recalibration on the rig's real hardware -- the noise-reduction numbers
    cited above are measured against real archived camera frames (not
    synthetic), but live validation on the real rig has not happened (no
    live-system access for the session that built this or the retry loop)
    -- separate, later follow-up work.

    Returns a dict of successfully-calibrated cameras only (0-3 entries,
    matching this rig's 3 known cameras) -- a caller needing "all 3 or
    fail" must check len() explicitly, same "no guessing" discipline as
    the rest of this project.
    """

    def _target_px_for(cam: int) -> float:
        """Resolves `target_reprojection_error_px` (float | dict[int,
        float], see this function's own ADAPTIVE RETRY docstring
        section) for one specific camera -- the one place this
        resolution happens, so every usage below stays a plain float
        comparison. A dict missing an entry for `cam` falls back to the
        module's own uniform default, same as passing a bare float has
        always done."""
        if isinstance(target_reprojection_error_px, dict):
            return target_reprojection_error_px.get(
                cam, CALIBRATION_TARGET_REPROJECTION_ERROR_PX
            )
        return target_reprojection_error_px

    def _capture(batch_n: int) -> dict[int, list[np.ndarray]]:
        if hub is None:
            raise ValueError(
                "bootstrap_calibrations() requires an already-open "
                "hub= LocalCameraHub -- this function never opens/closes "
                "cameras itself (see docstring), and there is no alternative "
                "frame source; the caller must open the hub."
            )
        return _capture_calibration_frames_local(hub, batch_n)

    # TIMING, added 2026-08-21 ("add timings for calibration...
    # both to the packages and to the logging... how long it takes to
    # calibrate (and for each camera)"). `_bootstrap_t0` is this whole
    # call's own start -- everything downstream (capture, per-camera
    # detect/solve rounds, the live-derived-value wiring after the
    # retry loop) is real wall-clock work this measures. Per-camera
    # duration is measured from this SAME t0 (every camera's own first
    # round starts together, at the top of the retry loop below) to the
    # moment THAT camera is discarded from `remaining` -- i.e. genuinely
    # "how long was this camera actively being calibrated", which
    # naturally reflects a camera needing extra retry rounds (see
    # `camera_end_time` below, set in the `as_completed()` loop).
    #
    # SECTION TIMING, added same day. Coarse, function-
    # call-level wrapping deliberately, not surgery inside `_detect_batch`
    # ()'s own two-branch internals (pre-orientation stage vs correspond-
    # ence-finish stage) -- that function has a real history of subtle
    # verifier-found bugs (see its own docstring), and this is a first
    # diagnostic pass, not the optimization itself; the four buckets below
    # (capture / detect / solve, plus the three post-loop derived-value
    # sections further down) are enough to tell whether time is going into
    # camera I/O, CV detection, the PnP solve, or the live-derived-value
    # wiring, without touching that function's insides. `capture_time_s`
    # is a single cumulative float (capture is sequential/shared across
    # cameras, called from the single-threaded round loop, not per-camera
    # workers); `detect_time_s`/`solve_time_s` are per-camera (each
    # camera's own dict entry is only ever touched by that camera's own
    # worker thread within a round -- see `_process_camera_round`'s own
    # docstring on why cross-camera dict mutation is already safe here --
    # so no lock needed for these either).
    _bootstrap_t0 = time.monotonic()
    camera_end_time: dict[int, float] = {}
    capture_time_s = 0.0
    detect_time_s: dict[int, float] = {}
    solve_time_s: dict[int, float] = {}

    _t = time.monotonic()
    calibration_progress.PROGRESS.stage("capture")
    frames_by_cam = _capture(n_frames)
    capture_time_s += time.monotonic() - _t

    # DECOUPLED CAPTURE-VS-DETECT TARGET, see CALIBRATION_N_FRAMES_DETECT's
    # own dated module comment and this function's own docstring section
    # above -- the raw capture above already grabbed `n_frames` (default
    # 50) per camera in one cheap call; round 1 of the detect/solve/retry
    # loop below only gets the first `n_frames_detect` (default 10) of
    # each camera's own frames. The rest are NOT discarded -- see the
    # `pre_orientation_pool[cam]` seeding in the setup loop just below,
    # which appends them straight in as `(frame, None)` pairs so the
    # post-loop ring-boundary-offset/board-color sections (which only
    # ever read the raw `pb` half of that pool's tuples -- confirmed by
    # reading `ring_bg_frames`/`color_samples` construction further down
    # before making this change) still see the full raw pool.
    frames_by_cam_detect = {
        cam: frames[:n_frames_detect] for cam, frames in frames_by_cam.items()
    }
    frames_by_cam_raw_extra = {
        cam: frames[n_frames_detect:] for cam, frames in frames_by_cam.items()
    }

    # Per-camera accumulated state across retry rounds -- see ADAPTIVE
    # RETRY in the docstring above. `remaining` is the set of cameras
    # still being retried this pass; a camera leaves it the moment it
    # either succeeds (target met, or unevaluable and accepted as-is),
    # fails outright (PnP solve error), or hits max_frames.
    accumulated_detections: dict[int, list] = {}
    accumulated_results: dict[int, list] = {}
    frames_captured: dict[int, int] = {}
    best_calibration: dict[int, CameraCalibration] = {}
    best_reprojection_px: dict[int, float | None] = {}
    remaining: set[int] = set()

    # LIVE-DERIVED ORIENTATION HINT state, per camera -- see this
    # function's own "LIVE-DERIVED ORIENTATION HINT" docstring section
    # above for the full design. Orientation evidence accumulates from
    # every round, not just the first, so a camera that needs retries
    # keeps getting MORE evidence to derive a hint
    # from. `pre_orientation_pool[cam]` accumulates EVERY round's own
    # `(processed_bgr, PreOrientationLandmarks)` pairs (ok AND not-ok
    # alike) -- added 2026-08-20 fixing verifier findings HIGH-1/HIGH-2:
    # before this, a frame processed in a round BEFORE a hint existed for
    # its camera was finished once with `orientation_hint_deg=None`
    # (an automatic reject) and never revisited, even after a hint later
    # became available -- silently wasting every pre-hint frame instead
    # of cheaply re-finishing it with `correspond_landmarks_from_pre_orientation()`
    # (see that function's own docstring: it exists specifically so this
    # re-finish is cheap, no detection re-run needed). This pool is what
    # makes that re-finish possible: the moment `established_hint_deg[cam]`
    # is set OR CHANGES (see `_establish_hint_if_possible()` below), the
    # WHOLE pool gets re-finished with the new hint, not just future
    # frames. `established_hint_deg[cam]` / `established_hint_source[cam]`
    # are updated as new, more-confident evidence arrives (see MEDIUM-4
    # fix below -- NOT frozen forever the moment any hint first clears the
    # acceptance floor) and are read fresh every round; `hint_locked[cam]`
    # is set True only once `_resolve_mode_a_orientation_after_round_one()`
    # (Mode A, spec R2.1) resolves this camera's hint -- either its own
    # cross-checked live value, or a rig-consensus prediction -- a real
    # terminal state for this bootstrap pass (nothing further recomputes
    # it). In Mode B (no ring geometry yet), `hint_locked` is never set
    # by this per-camera path at all -- see OrientationConsensusRefusedError's
    # own docstring for what happens instead once `max_frames` is
    # exhausted without every camera going live.
    # DECOUPLED CAPTURE-VS-DETECT TARGET, 2026-08-21 -- see this
    # function's own docstring section above. `PreOrientationLandmarks |
    # None`: a `None` second element marks a raw-only frame that was
    # captured but deliberately never run through
    # `locate_pre_orientation_landmarks()` (one of the `n_frames -
    # n_frames_detect` "extra" frames seeded below) -- `_detect_batch()`'s
    # whole-pool re-finish skips these (`if pre is not None`), and the
    # two post-loop raw-pixel-only sections (ring-boundary-offset,
    # board-color) never look at the `_pre` half at all, so `None` is
    # safe there too.
    pre_orientation_pool: dict[int, list[tuple[np.ndarray, PreOrientationLandmarks | None]]] = {}
    # WHICH POOL SLOT EACH ACCUMULATED DETECTION CAME FROM, recorded where
    # the two are created together rather than re-derived afterwards by
    # re-scanning the pool for `pre is not None`.
    #
    # That re-scan was the old approach and it was wrong from the moment
    # BEST-OF-N REPROJECTION ATTEMPTS landed (2026-08-30): that path also
    # appends to `pre_orientation_pool[cam]`, but keeps its detections in
    # a LOCAL list, so the pool grew `(n_reprojection_attempts - 1) *
    # n_frames_detect` slots the accumulated list never saw. With the
    # shipped 5 attempts and 5 detect-frames that is exactly 20 -- which
    # is why the invariant check failed on every camera of every rig on
    # every calibration, and why every calibration package written since
    # has recorded `frame_indices_used: null`. The check was correct that
    # something was inconsistent; the inconsistency was its own.
    # Last measured seed-ellipse aspect and its deviation from the group
    # median, per camera -- surfaced in diagnostics so the healthy case is
    # observable, not only the failing one (see the aspect check below).
    ellipse_aspect_by_cam: dict[int, float] = {}
    ellipse_aspect_deviation_by_cam: dict[int, float] = {}
    detected_pool_indices_by_det: dict[int, list[int]] = {}
    # The pool indices that fed the calibration actually ADOPTED for this
    # camera. Set at every adoption site, because with best-of-N the
    # winning solve is not always the accumulated one -- an attempt that
    # wins publishes ITS OWN five frames here, which is the honest answer
    # to "which frames produced this calibration".
    adopted_frame_indices: dict[int, list[int] | None] = {}
    established_hint_deg: dict[int, float] = {}
    established_hint_source: dict[int, str] = {}
    hint_locked: dict[int, bool] = {}

    # RIG-CONSENSUS ORIENTATION, 2026-08-29 -- see `opendarts.calibration.
    # rig_ring_geometry`'s own module docstring for the full design and
    # docs/DESIGN.md's dated entry for the real numbers this closes (the
    # ring-consensus orientation rules R1/R2/R3). `ring_geometry
    # is None` (no file yet, or `calibration_package_root is None`) means
    # Mode B (spec R2.1): every camera must derive its own orientation
    # LIVE this event -- there is no hardcoded fallback constant anymore
    # (that was `MEASURED_CAMERA_ORIENTATION_HINTS_DEG`, deleted -- see
    # the frozen fixture copy
    # for the historical values this project no longer ships as a live
    # constant). A camera present but not stored in `ring_geometry` is
    # equally Mode B for THIS event (the stored geometry's own camera
    # count doesn't match -- see the post-round-1 gate below), never a
    # silent partial-Mode-A.
    ring_geometry: RingGeometry | None = load_ring_geometry(calibration_package_root)

    # LIVE-DERIVED FOCAL LENGTH state, per camera -- added 2026-08-26, see
    # opendarts.calibration.focal_length's own module docstring for the full
    # design and docs/DESIGN.md's dated entry for the real bug this closes.
    # Mirrors the orientation-hint state immediately above deliberately
    # (same "(re-)derive every round from the growing accumulated pool,
    # remember the source, only fall back once genuinely exhausted"
    # shape) -- `established_focal_px[cam]`/`established_focal_source[cam]`
    # are read fresh by `_try_solve()` every round (recomputed from
    # `accumulated_results[cam]`'s current, growing contents each time --
    # see `_resolve_focal_length_px()` below), never frozen after a first
    # success, since MORE frames only ever means a better average here
    # (no aliasing/ambiguity failure mode the orientation hint has to
    # guard against with an explicit non-freezing policy -- see that
    # section's own MEDIUM-4 comment). `focal_length_fallback` is loaded
    # ONCE, up front, from `calibration_package_root` (the SAME directory
    # `save_calibration_package()` already writes every event's package
    # into) -- `{}` (not None) when `calibration_package_root` is None
    # (e.g. a caller/test that never wired one up) or the file doesn't
    # exist yet, so `_resolve_focal_length_px()` never needs a None check.
    established_focal_px: dict[int, float] = {}
    established_focal_source: dict[int, str] = {}
    focal_length_fallback: dict[int, dict] = (
        load_focal_length_fallback(calibration_package_root)
        if calibration_package_root is not None else {}
    )

    # LIVE-DERIVED RADIAL DISTORTION (k1) state, per camera -- added
    # 2026-08-26,
    # see opendarts.calibration.distortion's own module docstring for the
    # full conditioning analysis and docs/DESIGN.md's dated entry for the real
    # numbers. Mirrors the focal-length state immediately above
    # deliberately: fits ALONGSIDE it (same accumulated_results[cam]
    # ring20 pool, same per-round re-derivation, never frozen), and on
    # success its own SELF-CONSISTENT (focal_length_px, k1) pair
    # OVERRIDES established_focal_px[cam]/established_focal_source[cam]
    # too (see _resolve_distortion_for()'s own docstring for why fixing
    # the homography-only f and only floating k1 would let k1 silently
    # compensate for that f's own measured distortion-induced bias
    # instead of recovering a self-consistent answer). established_k1
    # stays at 0.0 (equivalent to today's hardcoded zero-distortion
    # assumption) for any camera this event's data doesn't safely
    # support a distortion fit for -- purely additive, never a
    # regression versus not having this at all.
    established_k1: dict[int, float] = {}
    established_distortion_source: dict[int, str] = {}
    # This function's OWN memory of its last successful self-consistent
    # joint-solve focal length -- deliberately separate from
    # established_focal_px (the homography-only tier's own dict, owned
    # by _resolve_focal_length_px()) -- see _resolve_distortion_for()'s
    # own docstring for why sharing that dict would silently pair a
    # fresh homography-only f with a stale k1 on a round where the joint
    # solve doesn't reconverge.
    established_joint_focal_px: dict[int, float] = {}
    distortion_fallback: dict[int, dict] = (
        load_distortion_fallback(calibration_package_root)
        if calibration_package_root is not None else {}
    )

    # LIVE-DERIVED PRINCIPAL POINT (cx only), 2026-08-26 -- see opendarts.calibration.
    # distortion's own module docstring, "PRINCIPAL POINT (cx only)"
    # section, for the full conditioning analysis (cx is safely
    # identifiable on this rig's real data; cy/p1/p2 are not, and are
    # NOT solved for). Mirrors the k1 state immediately above, one tier
    # further: this event's own live (f, pose, k1, cx) joint solve, tried
    # AFTER the k1-only solve above (a superset model, richer but with
    # one more way to fail to converge) -- on success its own SELF-
    # CONSISTENT (focal_length_px, k1, cx) triple supersedes both the
    # homography-only f AND the k1-only (f, k1) pair for this event, same
    # "don't mix a fresh unknown with a stale one from a different fit"
    # reasoning as k1's own override above. On failure, falls back to
    # whatever the k1-only tier already resolved (today's exact prior
    # behavior for this event) -- purely additive, never a regression.
    # `established_cx[cam] is None` (or absent) means "no live-derived cx
    # this event" -- `build_camera_matrix()` then defaults to the image
    # center, exactly as before this addition existed.
    # Deliberately its OWN independent set of dicts, not sharing
    # established_joint_focal_px/established_k1/established_distortion_source
    # (the k1-only tier's own dicts, owned by _resolve_distortion_for()) --
    # if this dict shared those, a SUCCESSFUL cx fit would mark the k1-only
    # tier's own established_distortion_source as something other than
    # "live," which would silently SKIP that tier's own
    # distortion_fallback.json persistence below (it's gated on
    # `== "live"`) every event this richer model also succeeds --
    # starving that already-shipped, independently-useful fallback tier
    # of fresh updates. Keeping this fully separate means both tiers
    # persist their own fallback file on every event they individually
    # succeed, regardless of what the other tier did.
    established_cx_focal_px: dict[int, float] = {}
    established_cx_k1: dict[int, float] = {}
    established_cx: dict[int, float] = {}
    established_principal_point_source: dict[int, str] = {}
    principal_point_fallback: dict[int, dict] = (
        load_principal_point_fallback(calibration_package_root)
        if calibration_package_root is not None else {}
    )

    # No more per-camera-index allowlist gate here -- see this function's
    # own docstring, "Real, related behaviour change worth naming
    # explicitly": whether a camera's hint is derivable is no longer
    # knowable before its frames are captured and processed, so every
    # camera present in the captured frame set is attempted.
    for cam in sorted(frames_by_cam):
        accumulated_detections[cam] = []
        accumulated_results[cam] = []
        frames_captured[cam] = 0
        # Seed the pool with this camera's own raw-only "extra" frames
        # (captured above but deliberately never detected -- see
        # DECOUPLED CAPTURE-VS-DETECT TARGET docstring section) BEFORE
        # round 1 appends its own detected frames -- order doesn't matter
        # for correctness (the whole-pool re-finish in `_detect_batch()`
        # below skips `pre is None` entries regardless of position), just
        # needs to happen once per camera, here, alongside the pool's own
        # initialization.
        #
        # `normalise_illuminant(frame)` here -- REAL BUG fix, verifier
        # pass 2026-08-21 (Bug 1): `locate_pre_orientation_landmarks()`
        # (oriented_landmarks.py) returns the FRAME IT WAS GIVEN after
        # running it through this same grey-world white-balance gain
        # (`oriented_landmarks.normalise_illuminant()`'s own docstring:
        # "so an absolute HSV threshold downstream means the same thing
        # under a camera whose white balance has drifted") -- that's
        # what the detected entries' own `pb` half already is. Before
        # this fix, the raw-only "extra" entries seeded here were the
        # UNTOUCHED `_capture()` output instead -- a silent ~80/20
        # raw/normalized mix landing in `pre_orientation_pool[cam]`,
        # feeding straight into board-color threshold derivation
        # (`collect_color_samples()`/`derive_thresholds()`, an ABSOLUTE
        # HSV computation -- exactly what normalise_illuminant() exists
        # to protect) and ring-boundary-offset. Applying the SAME
        # normalization here makes every pool entry's `pb` half
        # interchangeable again, matching this function's own pre-change
        # invariant (100% of pool frames normalized) instead of silently
        # relying on it.
        pre_orientation_pool[cam] = [
            (normalise_illuminant(frame), None)
            for frame in frames_by_cam_raw_extra.get(cam, [])
        ]
        detected_pool_indices_by_det[cam] = []
        adopted_frame_indices[cam] = None
        remaining.add(cam)
        detect_time_s[cam] = 0.0 # SECTION TIMING, see _bootstrap_t0 above
        solve_time_s[cam] = 0.0



    def _learn_ring_geometry_from_ring_correlation(resolved_cams: "set[int]") -> None:
        """Learn this rig's ring geometry from orientations every camera
        derived LIVE this event.

        Requires at least two resolved cameras and that all of them are
        live-derived (`established_hint_source == "ring_correlation_live"`)
        -- geometry learned from a rig-consensus fallback would just be
        re-learning what it was already told. Feeds the resolved hints to
        `update_ring_geometry()`; if the gaps between cameras have drifted
        past `RING_GEOMETRY_DRIFT_THRESHOLD_DEG` from the learned
        geometry, refuses rather than adopting it -- a camera has probably
        moved on the ring, which is otherwise an invisible physical event.

        (A sibling that learned the same geometry from an aggregate-mark
        method was removed with that method; this no longer mirrors
        anything.)"""
        live_cams = {
            cam for cam in resolved_cams
            if established_hint_source.get(cam) == "ring_correlation_live"
        }
        if not (len(resolved_cams) >= 2 and resolved_cams <= live_cams):
            return
        now_utc = datetime.now(timezone.utc).isoformat()
        hints_for_geometry = {cam: established_hint_deg[cam] for cam in resolved_cams}
        new_geometry, drift_deg, relearn_note = ring_geometry_for_this_event(
            ring_geometry, hints_for_geometry, now_utc)
        relearned = relearn_note is not None
        if relearn_note is not None:
            log.warning("ring geometry RELEARNED: %s (gaps were %s, now %s)",
                        relearn_note["note"], relearn_note["previous_gaps_deg"],
                        relearn_note["gaps_deg"])
            if calibration_package_out is not None:
                calibration_package_out["ring_geometry_relearned"] = relearn_note
        if calibration_package_root is not None:
            try:
                save_ring_geometry(calibration_package_root, new_geometry)
                log.info(
                    "ring geometry %s from this event's %d all-live (ring-correlation "
                    "method) camera(s) -- gaps now %s (n_events=%d, drift this "
                    "update=%s).",
                    "RELEARNED" if relearned else
                    "SEEDED" if ring_geometry is None else "UPDATED", len(resolved_cams),
                    [round(x, 2) for x in new_geometry.gaps_deg], new_geometry.n_events,
                    f"{drift_deg:.3f}deg" if drift_deg is not None else "n/a (first event)",
                )
            except Exception: # noqa: BLE001 -- never break calibration over this
                log.exception(
                    "failed to persist ring_geometry_fallback.json (ring-correlation "
                    "orientation method) -- this event's own calibration is "
                    "unaffected, only future events' Mode A eligibility is."
                )

    def _resolve_orientation_via_ring_correlation() -> None:
        """RING-CORRELATION orientation resolution -- retry, infer,
        refuse. See the module-level "RING-CORRELATION ORIENTATION"
        comment for the full design. Calls
        `opendarts.calibration.ring_correlation_orientation.
        ring_correlation_orientation_for_camera()`.

        Sets `established_hint_deg`/`established_hint_source`/
        `hint_locked` for EVERY camera in `frames_by_cam` on success
        (source `"ring_correlation_live"` or `"ring_correlation_rig_
        consensus"` -- see the module-level comment for why these are
        distinct strings from both other methods' own). Raises
        `OrientationConsensusRefusedError` after `RING_CORRELATION_
        ORIENTATION_MAX_RETRIES` attempts without a resolved set.
        """
        cams = sorted(frames_by_cam)
        if not cams:
            return # nothing to resolve -- the round loop below is a no-op too

        n = RING_CORRELATION_ORIENTATION_N_FRAMES
        window_offset: dict[int, int] = {cam: 0 for cam in cams}
        candidate_hint: dict[int, float | None] = {cam: None for cam in cams}
        candidate_pass_fraction: dict[int, float] = {cam: 0.0 for cam in cams}
        good: set[int] = set()

        def _next_windows_for(needy: list[int]) -> dict[int, list[np.ndarray]]:
            """Reuse the already-captured frame pool front-to-back, then
            fall back to a fresh shared `_capture()` once it is
            exhausted."""
            windows: dict[int, list[np.ndarray]] = {}
            needs_fresh: list[int] = []
            for cam in needy:
                pool = frames_by_cam.get(cam, [])
                start = window_offset[cam]
                w = pool[start : start + n]
                if len(w) == n:
                    windows[cam] = w
                    window_offset[cam] = start + n
                else:
                    needs_fresh.append(cam)
            if needs_fresh:
                fresh = _capture(n)
                for cam in needs_fresh:
                    windows[cam] = fresh.get(cam, [])
            return windows

        def _derive_for(cam: int, frames: list[np.ndarray]) -> None:
            result = ring_correlation_orientation_for_camera(
                frames, min_pass_fraction=RING_CORRELATION_ORIENTATION_MIN_PASS_FRACTION,
            )
            candidate_hint[cam] = result.hint_deg if result.ok else result.majority_hint_deg
            candidate_pass_fraction[cam] = result.pass_fraction
            if result.ok:
                good.add(cam)
            else:
                good.discard(cam)
            # "THE FULL PICTURE" -- a deliberate design decision
            # task's own instructions: surface the richer
            # per-frame diagnostics (correlation margin, any hard-fail
            # flags), not just an aggregate pass/fail
            # line shows, whenever a per-frame result actually exists
            # (a per-frame RuntimeError has none to show).
            last = result.per_frame[-1] if result.per_frame else None
            corr_detail = (
                f", corr_best={last.result.corr_best:.3f}, flags={last.result.flags}"
                if last is not None and last.result is not None
                else f", error={last.error}" if last is not None and last.error is not None
                else ""
            )
            log.info(
                "cam%d: ring-correlation orientation candidate -- %s, "
                "pass_fraction=%.3f (%d/%d frame(s) agreed), hint=%s%s",
                cam, "PASSED" if result.ok else "below floor", result.pass_fraction,
                result.n_agreeing, result.n_frames,
                f"{result.hint_deg:.2f}deg" if result.hint_deg is not None else "n/a",
                corr_detail,
            )

        for attempt in range(1, RING_CORRELATION_ORIENTATION_MAX_RETRIES + 1):
            needs_derivation = [cam for cam in cams if cam not in good]
            if needs_derivation:
                windows = _next_windows_for(needs_derivation)
                for cam in needs_derivation:
                    _derive_for(cam, windows.get(cam, []))

            mode_a_eligible = (
                ring_geometry is not None and len(ring_geometry.gaps_deg) == len(cams)
            )
            if mode_a_eligible:
                candidates = {
                    cam: CameraOrientationCandidate(
                        hint_deg=candidate_hint[cam], confidence=candidate_pass_fraction[cam],
                    )
                    for cam in cams
                }
                consensus = resolve_rig_consensus_orientation(
                    candidates, ring_geometry,
                    min_confidence=RING_CORRELATION_ORIENTATION_MIN_PASS_FRACTION,
                )
                if consensus.ok:
                    for cam, (hint_deg, source) in sorted(consensus.resolved.items()):
                        established_hint_deg[cam] = hint_deg
                        established_hint_source[cam] = (
                            "ring_correlation_live" if source == "live"
                            else "ring_correlation_rig_consensus"
                        )
                        hint_locked[cam] = True
                        log.info(
                            "cam%d: orientation hint RESOLVED (ring-correlation "
                            "method, attempt %d/%d) -- source=%s, %.2f deg. %s",
                            cam, attempt, RING_CORRELATION_ORIENTATION_MAX_RETRIES,
                            established_hint_source[cam], hint_deg,
                            consensus.rejected_cameras.get(cam, ""),
                        )
                    _learn_ring_geometry_from_ring_correlation(set(cams))
                    return
                rejected_this_round = set(consensus.rejected_cameras)
                if rejected_this_round & good:
                    good -= rejected_this_round
                elif not consensus.resolved:
                    good.clear()
                log.warning(
                    "ring-correlation orientation: inference (rig-consensus) did "
                    "not accept attempt %d/%d's candidates -- %s. Retrying with "
                    "fresh frames.",
                    attempt, RING_CORRELATION_ORIENTATION_MAX_RETRIES, consensus.refusal_reason,
                )
                continue

            if len(good) == len(cams):
                for cam in cams:
                    established_hint_deg[cam] = candidate_hint[cam]
                    established_hint_source[cam] = "ring_correlation_live"
                    hint_locked[cam] = True
                log.info(
                    "ring-correlation orientation: all %d camera(s) resolved LIVE "
                    "(Mode B, attempt %d/%d, no ring geometry yet to cross-check "
                    "against).",
                    len(cams), attempt, RING_CORRELATION_ORIENTATION_MAX_RETRIES,
                )
                _learn_ring_geometry_from_ring_correlation(set(cams))
                return
            log.info(
                "ring-correlation orientation: %d/%d camera(s) resolved this "
                "attempt (%d/%d) -- no ring geometry yet, every camera must "
                "derive live. Retrying remaining camera(s) with fresh frames.",
                len(good), len(cams), attempt, RING_CORRELATION_ORIENTATION_MAX_RETRIES,
            )

        missing = sorted(set(cams) - good)
        detail = "; ".join(
            (
                f"cam{cam}: best candidate {candidate_hint[cam]:.2f}deg "
                f"(pass_fraction {candidate_pass_fraction[cam]:.2f})"
                if candidate_hint[cam] is not None
                else f"cam{cam}: no candidate at all"
            )
            for cam in missing
        )
        raise OrientationConsensusRefusedError(
            f"calibration REFUSED (ring-correlation orientation method, spec R1): "
            f"cam(s) {missing} could not establish board orientation after "
            f"{RING_CORRELATION_ORIENTATION_MAX_RETRIES} retries ({detail}, floor "
            f"{RING_CORRELATION_ORIENTATION_MIN_PASS_FRACTION:.2f} pass_fraction). "
            f"Check the listed camera(s)' lighting/framing/focus, then recalibrate."
        )

    if ORIENTATION_METHOD != ORIENTATION_METHOD_RING_CORRELATION:
        raise ValueError(
            f"ORIENTATION_METHOD={ORIENTATION_METHOD!r} is not a recognised orientation "
            f"method -- must be {ORIENTATION_METHOD_RING_CORRELATION!r}"
        )
    # WALL-CLOCK SPANS, 2026-09-12. Everything from _bootstrap_t0 to here
    # is setup + the initial shared capture; this is the orientation phase
    # (docs/CALIBRATION.md phase 3), which until now had no bucket at all.
    # Measured on the rigs, the six existing buckets left 27% of the rig's
    # run and 22% of the PC's unattributed -- and the biggest untimed
    # thing was the phase you would most want to measure before touching
    # it. The spans below are DISJOINT and SEQUENTIAL, so they sum to the
    # total with an explicit `unaccounted` remainder that can never again
    # hide a whole phase.
    setup_wall_s = time.monotonic() - _bootstrap_t0
    _t_orientation = time.monotonic()
    calibration_progress.PROGRESS.stage("orientation")
    if ORIENTATION_METHOD == ORIENTATION_METHOD_RING_CORRELATION:
        _resolve_orientation_via_ring_correlation()
    orientation_wall_s = time.monotonic() - _t_orientation


    def _detect_batch(cam: int, frames: list[np.ndarray]) -> list[Any]:
        """Run landmark detection on ONE round's newly captured frames
        for `cam`. Two passes over just THIS round's frames (see this
        function's own "LIVE-DERIVED ORIENTATION HINT" docstring section
        for why): first the pre-`lock_orientation()` stages ALONE
        (`locate_pre_orientation_landmarks()`, hint-independent), feeding
        this camera's running orientation-evidence pool AND its
        `pre_orientation_pool` (every frame, ok or not -- see that pool's
        own comment above) and a possible hint (re-)establishment; then
        EITHER of two finishing paths, depending on whether this round's
        `_establish_hint_if_possible()` call actually changed the hint
        (fixing verifier findings HIGH-1/HIGH-2):

          * Hint unchanged this round (already established earlier, or
            still not established at all): only THIS round's own new
            frames get the cheap finishing stage
            (`correspond_landmarks_from_pre_orientation()`), same as
            before this fix. A frame processed while no hint exists yet
            for its camera is handed `orientation_hint_deg=None`, which
            `lock_orientation()` already treats as "ambiguous, reject" --
            still buffered in `pre_orientation_pool[cam]` for a future
            re-finish once a hint DOES become available, not lost.

          * Hint newly established OR revised this round: the camera's
            ENTIRE `pre_orientation_pool` (every round's frames, not just
            this one) is re-finished with the new hint, replacing
            `accumulated_detections[cam]`/`accumulated_results[cam]`
            wholesale -- this is what actually fixes HIGH-1 (a fallback
            constant established only on the final round now applies to
            every already-captured frame, not just that round's own) and
            HIGH-2 (frames captured before a live hint existed are no
            longer permanently wasted).

        Returns THIS round's own per-frame OrientedLandmarkResult list
        either way (for this round's own ambiguous-lock log line -- see
        the module docstring's "ADAPTIVE RETRY" section for why
        round-local, not cumulative, numbers are what get logged there
        each round) -- sliced off the tail of `accumulated_results[cam]`
        in the whole-pool-reprocess case, since that list's own order
        matches `pre_orientation_pool[cam]`'s append order and this
        round's frames were appended to it last, just above."""
        pre_stage = [locate_pre_orientation_landmarks(frame) for frame in frames]
        # Indices captured BEFORE the extend, so they name the slots this
        # batch is about to occupy -- see detected_pool_indices_by_det.
        batch_pool_indices = list(
            range(len(pre_orientation_pool[cam]), len(pre_orientation_pool[cam]) + len(pre_stage))
        )
        pre_orientation_pool[cam].extend(pre_stage)
        detected_pool_indices_by_det[cam].extend(batch_pool_indices)

        # REPLAY (docs/DESIGN.md's "Replay is the source of truth"), 2026-08-21 -- calibration
        # packages are now assembled straight from `pre_orientation_pool`
        # (see `package_raw_frames` further down), not a separate
        # `raw_frames_accum` accumulator. Fixing verifier finding Bug 2
        # (2026-08-21): a dedicated `raw_frames_accum` used to be
        # extended HERE, unconditionally on `pre_stage`'s own frames --
        # but that only ever covered frames that went through THIS
        # detect stage, never the `n_frames - n_frames_detect` raw-only
        # "extra" frames seeded straight into `pre_orientation_pool`
        # without detection (see DECOUPLED CAPTURE-VS-DETECT TARGET
        # above). A camera whose target was met at `n_frames_detect`
        # (the common case) would then save a package with only ~10
        # raw frames on disk while `derived_calibration.json`'s own
        # `n_frames_raw_pool` diagnostic correctly reported ~50 --
        # internally self-contradictory, and unable to reproduce what
        # the live ring-boundary-offset/board-color derivations actually
        # computed (both of which consume the FULL pool, raw-extra
        # frames included). `pre_orientation_pool[cam]` already holds
        # every frame (detected AND raw-extra) unconditionally, whether
        # or not a caller opted into package saving -- deriving
        # `package_raw_frames` from it below costs no extra memory and
        # is provably complete by construction, not by keeping two
        # accumulators in sync.

        hint = established_hint_deg.get(cam)


        per_frame_results: list[Any] = []
        detections = [
            correspond_landmarks_from_pre_orientation(
                processed_bgr, pre,
                orientation_hint_deg=hint,
                results_out=per_frame_results,
            )
            for processed_bgr, pre in pre_stage
        ]
        accumulated_detections[cam].extend(detections)
        accumulated_results[cam].extend(per_frame_results)
        frames_captured[cam] += len(frames)
        return per_frame_results

    def _report_round(cam: int, frames: list[np.ndarray], per_frame_results: list[Any]) -> None:
        n_ambiguous = sum(1 for r in per_frame_results if r.orientation_ambiguous)
        if not n_ambiguous:
            return
        # Accurate operator guidance depends on which path actually
        # produced (or failed to produce) this camera's hint -- fixing
        # verifier finding MEDIUM-5, a stale message that always pointed
        # at the old hand-maintained hint constant even under the
        # live-derivation default path, which does not use it at all.
        #
        # 2026-09-03 fix (cam0-calibration-fails-until-restart defect
        # investigation, see docs/DESIGN.md): this check must recognise the
        # real per-method source strings
        # (`"ring_correlation_live"`/`"ring_correlation_rig_consensus"`),
        # not bare `"live"`/`"rig_consensus"` (see the
        # `established_hint_source[cam] = ...`
        # assignments above). Under the live default
        # (`ORIENTATION_METHOD_RING_CORRELATION`), this meant every
        # ambiguous-lock warning ALWAYS fell through to the "no hint
        # established" branch below -- even when a real, confident,
        # correctly-established hint demonstrably existed, confirmed via
        # direct instrumentation. Purely a diagnostic-message bug: this
        # function only builds a log line, `source` is never used for any
        # control-flow decision here or anywhere else that reads its
        # return value (there is none -- `-> None`). Fixed by recognising
        # every real per-method suffix generically, so a future
        # orientation method's own `"<name>_live"`/`"<name>_rig_
        # consensus"` convention keeps working without another silent
        # miss like this one.
        source = established_hint_source.get(cam)
        _live_module_by_source = {
            "ring_correlation_live": "opendarts.calibration.ring_correlation_orientation",
        }
        if source is not None and (source == "rig_consensus" or source.endswith("_rig_consensus")):
            guidance = (
                "This camera's hint was PREDICTED via rig-consensus "
                "(opendarts.calibration.rig_ring_geometry) from the other "
                "camera(s) + this rig's learned ring geometry -- this "
                "camera's own live derivation never cleared its "
                "confidence floor this event. Check this camera's "
                "lighting/framing/focus; an ambiguous lock alongside a "
                "consensus-filled hint is expected until that camera's "
                "own detection improves."
            )
        elif source in _live_module_by_source:
            guidance = (
                "This camera's hint was derived LIVE from its own "
                f"captured frames ({_live_module_by_source[source]}) "
                "-- an ambiguous lock on some frames despite that usually "
                "means per-frame lighting/focus/motion-blur on just those "
                "frames, not a wrong hint; check those specific frames "
                "rather than the hint, which this camera derived live."
            )
        else:
            guidance = (
                "This camera has NO established orientation hint at all "
                "yet this round (neither a confident live derivation nor "
                "a rig-consensus prediction) -- every frame is ambiguous "
                "by construction until one is established."
            )
        log.warning(
            "cam%d: %d/%d calibration frame(s) REJECTED for an ambiguous "
            "orientation lock -- the board's rotation could not be pinned "
            "to a single sector, so those frames' landmarks may be a whole "
            "quarter-turn wrong. %s",
            cam,
            n_ambiguous,
            len(frames),
            guidance,
        )


    def _check_cross_camera_ellipse_aspect_consistency(
        cams_this_round: list[int], frames_by_cam_this_round: dict[int, list[np.ndarray]],
    ) -> None:
        """Part B of the 2026-09-03 cam0 ellipse-merge defect fix (see
        this module's own `ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD`
        comment block for the full design/threshold derivation) -- a
        cross-camera SAFETY NET, run once per round right after this
        round's own futures complete (same synchronization point as
        `_resolve_mode_a_orientation_after_round_one()`, called right
        after it in the round loop below -- deliberately AFTER, not
        before: Mode A can rebuild `accumulated_detections[cam]` wholesale
        for every camera, and this function's own null-out below must
        operate on whatever list is actually live by the time it runs,
        not one Mode A is about to replace out from under it).

        Reads each `cam` in `cams_this_round`'s OWN this-round seed
        ellipses straight from `pre_orientation_pool[cam]`'s tail (the
        same `[-len(frames):]` slicing invariant `_detect_batch()`'s own
        docstring already establishes and trusts for `accumulated_
        results[cam]` -- reused here, not reinvented) -- `pre.ellipse` is
        computed by `locate_pre_orientation_landmarks()` entirely upstream
        of, and unaffected by, orientation-hint resolution, so this is
        safe regardless of whether Mode A has run yet this round.

        A camera with zero valid (`pre.ok and pre.ellipse is not None`)
        detections this round contributes nothing (not "currently
        confident" -- see `MIN_CAMERAS_FOR_ELLIPSE_ASPECT_CHECK`'s own
        comment). With fewer than 2 confident cameras, this is a
        deliberate no-op -- does not fire, does not block that camera's
        own progress, per this task's own explicit instruction.

        An outlier camera (deviation from the SAME-EVENT group's median
        aspect exceeding `ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD`)
        has THIS ROUND's own `accumulated_detections[cam]` entries
        nulled out (the same "not usable" signal an ambiguous-
        orientation frame already produces -- see `correspond_
        landmarks_from_pre_orientation()`) -- feeding straight into the
        EXISTING `_try_solve()`/`average_correspondences(min_required=
        ...)` retry pathway, no new terminal state, no new state
        machine. `_process_camera_round(cam, [])` (the SAME no-new-
        frames re-run pattern Mode A's own `retry_targets` loop below
        already uses) is then called to make that pathway actually
        re-evaluate against the corrected pool this round, updating
        `remaining` exactly as if this round's contaminated frames had
        never arrived."""
        this_round_aspects: dict[int, float] = {}
        this_round_spread: dict[int, float] = {}
        for cam in cams_this_round:
            frames_this_round = frames_by_cam_this_round.get(cam, [])
            if not frames_this_round:
                continue
            pool_tail = pre_orientation_pool[cam][-len(frames_this_round):]
            aspects = []
            for _processed_bgr, pre in pool_tail:
                if pre is None or not pre.ok or pre.ellipse is None:
                    continue
                lo, hi = sorted([pre.ellipse.major_axis_px, pre.ellipse.minor_axis_px])
                if lo > 0:
                    aspects.append(hi / lo)
            if aspects:
                this_round_aspects[cam] = float(np.median(aspects))
                # Spread of THIS camera's own frames -- the corroborating
                # signal (see ELLIPSE_ASPECT_INSTABILITY_THRESHOLD).
                this_round_spread[cam] = float(max(aspects) - min(aspects))

        if len(this_round_aspects) < MIN_CAMERAS_FOR_ELLIPSE_ASPECT_CHECK:
            return

        group_median = float(np.median(list(this_round_aspects.values())))
        # Record EVERY round's measurement, not just the failing ones.
        # Until 2026-09-12 the aspect and its deviation appeared only in
        # the outlier warning, so a healthy rig reported nothing at all --
        # which meant the only way to see how close a camera sat to the
        # 0.05 gate was to trip it. That is exactly backwards for a
        # threshold whose whole question is "how much headroom is there
        # on THIS rig", and it made a real diagnosis (is a camera's
        # deviation a mounting property or a lens one?) unanswerable
        # without swapping hardware and watching for the warning to stop.
        for cam, aspect in this_round_aspects.items():
            ellipse_aspect_by_cam[cam] = float(aspect)
            ellipse_aspect_deviation_by_cam[cam] = float(abs(aspect - group_median))
        log.info(
            "seed-ellipse aspect this round: %s | group median %.3f | max deviation "
            "%.3f (gate %.2f)",
            {cam: round(a, 3) for cam, a in sorted(this_round_aspects.items())},
            group_median,
            max(abs(a - group_median) for a in this_round_aspects.values()),
            ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD,
        )

        # A cross-camera outlier is only treated as a DEFECT when its own
        # frames also disagree with each other, or when it is so far out
        # that placement cannot explain it -- see the two thresholds' own
        # dated comments for the measurements behind this.
        outlier_cams: dict[int, float] = {}
        for cam, aspect in this_round_aspects.items():
            deviation = abs(aspect - group_median)
            if deviation <= ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD:
                continue
            spread = this_round_spread.get(cam, 0.0)
            if deviation >= ELLIPSE_ASPECT_CERTAIN_DEFECT_DEVIATION:
                outlier_cams[cam] = aspect
                continue
            if spread > ELLIPSE_ASPECT_INSTABILITY_THRESHOLD:
                outlier_cams[cam] = aspect
                continue
            # Differs from its siblings but is entirely self-consistent:
            # a camera placed differently, not a fragmenting mask. Said
            # out loud rather than silently skipped -- an operator who
            # expects a symmetric rig should still learn that one camera
            # is not, and this is the only place that measurement exists.
            log.info(
                "cam%d: seed-ellipse aspect %.3f differs from the group median "
                "%.3f by %.3f (over the %.2f gate) but its own %d frame(s) this "
                "round agree to %.4f -- a persistently different VIEW, not a "
                "fragmenting mask, so the frames are kept. Check this camera's "
                "placement if a symmetric rig was intended.",
                cam, aspect, group_median, deviation,
                ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD,
                len(frames_by_cam_this_round.get(cam, [])), spread,
            )
        if not outlier_cams:
            return

        for cam, aspect in sorted(outlier_cams.items()):
            n_this_round = len(frames_by_cam_this_round.get(cam, []))
            if n_this_round == 0:
                continue
            log.warning(
                "cam%d: this round's seed-ellipse aspect ratio (%.3f) is an "
                "outlier vs its %d sibling camera(s) this event (group "
                "median %.3f, deviation %.3f > %.3f threshold) -- likely a "
                "merged double+treble ring component or another "
                "seed-ellipse defect; "
                "marking this round's %d frame(s) not-yet-usable and "
                "retrying.",
                cam, aspect, len(this_round_aspects) - 1, group_median,
                abs(aspect - group_median), ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD,
                n_this_round,
            )
            if accumulated_detections[cam]:
                accumulated_detections[cam][-n_this_round:] = [None] * min(
                    n_this_round, len(accumulated_detections[cam])
                )

        for cam in sorted(outlier_cams):
            stay_in_remaining = _process_camera_round(cam, [])
            if stay_in_remaining:
                remaining.add(cam)
            else:
                remaining.discard(cam)
                camera_end_time[cam] = time.monotonic()

    def _resolve_focal_length_px(cam: int, width: float, height: float) -> tuple[float | None, str | None]:
        """(Re-)derive `cam`'s focal length from its accumulated
        `accumulated_results[cam]` pool -- called every round from
        `_try_solve()` below, mirroring `_establish_hint_if_possible()`'s
        overall shape (see that function's own docstring) but simpler:
        unlike the orientation hint, more accumulated frames here can
        only ever IMPROVE the derivation (no aliasing/ambiguity failure
        mode to guard against -- `derive_focal_length_from_oriented_
        results()`'s own degeneracy checks already reject a genuinely bad
        geometry outright), so this never "freezes" a value and always
        prefers a fresh successful derivation over an older one.

        Priority order, all derived from this rig's own frames (see
        opendarts.calibration.focal_length's own module docstring):
          1. LIVE derivation from THIS event's own accumulated frames.
          2. The persisted `focal_length_fallback.json` entry for this
             camera INDEX (a prior event's own live-derived value --
             self-correcting, since a later successful live derivation
             this event overwrites `established_focal_source[cam]` to
             `"live"` and, post-loop, updates this JSON file too -- see
             this function's own post-loop write). Only consulted once
             `frames_captured[cam]` has reached `max_frames` with live
             derivation still never having succeeded THIS event -- same
             "every real chance to derive it live has already been
             exhausted" gate `_establish_hint_if_possible()` uses.

        **NO third, hardcoded-constant tier, by deliberate design**
. An earlier draft
        of this function kept a hardcoded focal-length constant as a
        true-last-resort tier 3 -- deliberately removed on review: that
        constant was not derived from this rig's frames, and keeping it
        reachable at all, even as a rare degenerate-case safety net, is
        exactly what the calibration-bootstrap rule (every calibration
        input derived from this rig's own frames) prohibits, regardless
        of how good the number is or how
        rarely the path fires. A camera this function
        cannot resolve (no live derivation succeeds within the full
        frame budget, AND no persisted fallback JSON exists yet for its
        index) is loudly, honestly skipped for this bootstrap pass
        instead -- the same "worse-but-genuinely-general beats better-
        but-still-hardcoded" posture this file takes throughout. No code
        path reads a hardcoded focal
        length any more.

        Returns `(focal_length_px, source)` -- `(None, None)` when
        NEITHER tier above has anything to offer: either live derivation
        is not yet confident AND `frames_captured[cam] < max_frames`
        (more rounds may still succeed -- `_try_solve()` treats this like
        "not enough correspondence data yet" and keeps retrying via the
        existing adaptive-retry mechanism), OR the full frame budget is
        exhausted with no persisted fallback available either, in which
        case this camera is not calibrated this pass, loudly logged, no
        new state machine needed either way."""
        result = derive_focal_length_from_oriented_results(
            accumulated_results[cam], (width / 2.0, height / 2.0),
            min_frames=_min_good_frames_cumulative(frames_captured[cam]),
        )
        if result.ok:
            prior = established_focal_px.get(cam)
            changed = prior is None or abs(prior - result.focal_length_px) > 1e-6
            established_focal_px[cam] = result.focal_length_px
            established_focal_source[cam] = "live"
            if changed:
                log.info(
                    "cam%d: focal length (RE)DERIVED LIVE from %d frame(s), %d "
                    "ring20 point(s) -- %.2fpx, derived from this rig's own frames "
                    "(opendarts.calibration.focal_length).",
                    cam, result.n_frames_used, result.n_points_used, result.focal_length_px,
                )
            return result.focal_length_px, "live"

        # Live derivation didn't succeed THIS round -- if an EARLIER
        # round this same event already succeeded, keep using that
        # (strictly more/equal data than any prior round, so a transient
        # dip below the plausibility/degeneracy gates on a fresh noisy
        # frame should not discard an already-good answer).
        if established_focal_source.get(cam) == "live":
            return established_focal_px[cam], "live"

        log.info(
            "cam%d: live focal-length derivation not yet confident (%s, %d frame(s) "
            "so far) -- will keep accumulating more before falling back.",
            cam, result.reason, frames_captured[cam],
        )

        if frames_captured[cam] < max_frames:
            return None, None # more rounds may still succeed -- keep retrying

        fallback_entry = focal_length_fallback.get(cam)
        if fallback_entry is not None:
            fallback_f = float(fallback_entry["focal_length_px"])
            established_focal_px[cam] = fallback_f
            established_focal_source[cam] = "fallback_json"
            log.warning(
                "cam%d: LIVE FOCAL-LENGTH DERIVATION FAILED after the full %d-frame "
                "capture budget (%s) -- falling back to the persisted "
                "focal_length_fallback.json value (%.2fpx, derived %s from a PRIOR "
                "calibration event, n_frames_used=%s). This is a DEGRADED, non-fresh "
                "value -- if this camera was physically moved/re-seated since that "
                "prior event, this number may be stale; expect it to self-correct "
                "the next time live derivation succeeds.",
                cam, frames_captured[cam], result.reason, fallback_f,
                fallback_entry.get("derived_at_utc"), fallback_entry.get("n_frames_used"),
            )
            return fallback_f, "fallback_json"

        # NO tier 3 -- deliberately never falls back to a hardcoded
        # focal-length constant. See this
        # function's own docstring for why. A camera
        # that reaches here has exhausted live derivation for the full
        # frame budget AND has no persisted fallback JSON entry for its
        # index (first-ever calibration on this rig, or a prior event
        # that also never succeeded) -- honestly cannot be calibrated
        # this pass, loudly logged, never silently patched over with a
        # hardcoded number.
        established_focal_source[cam] = "unavailable"
        log.warning(
            "cam%d: LIVE FOCAL-LENGTH DERIVATION FAILED after the full %d-frame "
            "capture budget (%s), and no persisted focal_length_fallback.json entry "
            "exists for this camera index yet -- this camera cannot be calibrated "
            "at all this bootstrap pass. This is intentional: calibration only "
            "uses values derived from this rig's own frames -- check "
            "this camera's lighting/framing/"
            "focus and retry rather than trusting a stale/hardcoded number.",
            cam, frames_captured[cam], result.reason,
        )
        return None, None

    def _resolve_distortion_for(
        cam: int, width: float, height: float, homography_focal_px: float | None
    ) -> tuple[float | None, float, str | None]:
        """(Re-)derive `cam`'s (focal_length_px, k1) pair from its
        accumulated `accumulated_results[cam]` ring20 pool -- mirrors
        `_resolve_focal_length_px()`'s own overall shape (see this
        module's own established_k1 setup comment above), but for the
        joint (f, pose, k1) solve `opendarts.calibration.distortion`
        implements. Returns `(focal_length_px, k1, source)`:
        `focal_length_px` is `None` when this function has nothing to
        offer (caller should keep using `homography_focal_px` as-is);
        `k1` is always a float (0.0 -- today's existing zero-distortion
        behavior -- when distortion isn't available this round).

        Priority, all derived from this rig's own frames (see
        opendarts.calibration.distortion's own module docstring): (1) LIVE joint derivation
        from THIS event's own accumulated frames -- re-attempted every
        round, never frozen (more frames only ever improves this, same
        reasoning `_resolve_focal_length_px()` already uses). (2) the
        persisted `distortion_fallback.json` entry for this camera INDEX
        (a prior event's own live-derived (f, k1) pair, self-correcting)
        -- only consulted once `frames_captured[cam]` has reached
        `max_frames` with live derivation still never having succeeded
        THIS event. NO third, hardcoded-constant tier -- a camera this
        function cannot resolve simply keeps k1=0.0 (today's existing
        default), never a hardcoded distortion guess."""
        result = derive_focal_and_k1_from_oriented_results(
            accumulated_results[cam], (width / 2.0, height / 2.0), width, height,
            min_frames=_min_good_frames_cumulative(frames_captured[cam]),
            initial_focal_px=homography_focal_px,
        )
        if result.ok:
            prior = established_k1.get(cam)
            changed = prior is None or abs(prior - result.k1) > 1e-6
            # Deliberately does NOT touch established_focal_px/
            # established_focal_source (that dict/source pair is
            # `_resolve_focal_length_px()`'s own, homography-only tier --
            # left completely alone so its own focal_length_fallback.json
            # persistence below keeps tracking the homography-only value,
            # independent of whether distortion also succeeded).
            # established_joint_focal_px is THIS function's own separate
            # memory of its last successful SELF-CONSISTENT (f, k1) pair,
            # read back by the "reuse a previous round's result" branch
            # just below -- reading established_focal_px there instead
            # would return whatever `_resolve_focal_length_px()` (called
            # BEFORE this function, every round, in `_try_solve()`) just
            # set THIS round, silently pairing a fresh homography-only f
            # with a stale k1 from a different fit.
            established_joint_focal_px[cam] = result.focal_length_px
            established_k1[cam] = result.k1
            established_distortion_source[cam] = "live"
            if changed:
                log.info(
                    "cam%d: radial distortion (RE)DERIVED LIVE from %d frame(s), %d "
                    "ring20 point(s) -- k1=%.4f, self-consistent f=%.2fpx (reprojection "
                    "RMS %.3fpx on the ring20 fit), derived from this rig's own frames "
                    "(opendarts.calibration.distortion).",
                    cam, result.n_frames_used, result.n_points_used, result.k1,
                    result.focal_length_px, result.reprojection_rms_px or float("nan"),
                )
            return result.focal_length_px, result.k1, "live"

        if established_distortion_source.get(cam) == "live":
            return established_joint_focal_px[cam], established_k1[cam], "live"

        log.info(
            "cam%d: live distortion derivation not yet confident (%s, %d frame(s) "
            "so far) -- will keep accumulating more before falling back.",
            cam, result.reason, frames_captured[cam],
        )

        if frames_captured[cam] < max_frames:
            return None, 0.0, None # more rounds may still succeed -- keep retrying

        fallback_entry = distortion_fallback.get(cam)
        if fallback_entry is not None:
            fb_f = float(fallback_entry["focal_length_px"])
            fb_k1 = float(fallback_entry["k1"])
            established_distortion_source[cam] = "fallback_json"
            log.warning(
                "cam%d: LIVE DISTORTION DERIVATION FAILED after the full %d-frame "
                "capture budget (%s) -- falling back to the persisted "
                "distortion_fallback.json value (f=%.2fpx, k1=%.4f, derived %s from a "
                "PRIOR calibration event). This is a DEGRADED, non-fresh value -- "
                "expect it to self-correct the next time live derivation succeeds.",
                cam, frames_captured[cam], result.reason, fb_f, fb_k1,
                fallback_entry.get("derived_at_utc"),
            )
            return fb_f, fb_k1, "fallback_json"

        # NO fallback available either -- k1 stays 0.0, exactly today's
        # existing hardcoded-zero-distortion behavior. Not an error: this
        # is purely additive, so "distortion unavailable this event" must
        # never block calibration itself (that's still
        # _resolve_focal_length_px()'s job).
        established_distortion_source[cam] = "unavailable"
        log.info(
            "cam%d: live distortion derivation unavailable after the full %d-frame "
            "capture budget (%s), and no persisted distortion_fallback.json entry "
            "exists for this camera index yet -- calibrating with the existing "
            "zero-distortion assumption (k1=0.0) for this camera this pass.",
            cam, frames_captured[cam], result.reason,
        )
        return None, 0.0, None

    def _resolve_principal_point_cx_for(
        cam: int, width: float, height: float, homography_focal_px: float | None
    ) -> tuple[float | None, float | None, float | None, str | None]:
        """(Re-)derive `cam`'s self-consistent (focal_length_px, k1, cx)
        TRIPLE from its accumulated `accumulated_results[cam]` ring20
        pool -- mirrors `_resolve_distortion_for()`'s own shape one tier
        further (see `opendarts.calibration.distortion`'s own module
        docstring, "PRINCIPAL POINT (cx only)" section, and this
        function's own state-setup comment above). Returns
        `(focal_length_px, k1, cx, source)`: `focal_length_px`/`k1`/`cx`
        are always resolved TOGETHER or not at all (never independently
        None) -- `None` means "this function has nothing to offer this
        round, caller keeps using whatever the k1-only tier already
        resolved unchanged" (today's exact prior behavior when this
        function has nothing to add).

        Deliberately calls `derive_focal_k1_cx_from_oriented_results()`
        with `homography_focal_px` (the caller's ALREADY-resolved (f, k1)
        tier's own f, itself possibly the k1-only tier's self-consistent
        f) as the extra multi-init seed -- the same "usually already
        close to the true answer" reasoning `_resolve_distortion_for()`
        itself already uses for its own extra seed."""
        result = derive_focal_k1_cx_from_oriented_results(
            accumulated_results[cam], (width / 2.0, height / 2.0), width, height,
            min_frames=_min_good_frames_cumulative(frames_captured[cam]),
            initial_focal_px=homography_focal_px,
        )
        if result.ok:
            prior = established_cx.get(cam)
            changed = prior is None or abs(prior - result.cx) > 1e-6
            established_cx_focal_px[cam] = result.focal_length_px
            established_cx_k1[cam] = result.k1
            established_cx[cam] = result.cx
            established_principal_point_source[cam] = "live"
            if changed:
                log.info(
                    "cam%d: principal-point-X (RE)DERIVED LIVE from %d frame(s), %d "
                    "ring20 point(s) -- cx=%.2fpx (offset %+.2fpx from center), "
                    "self-consistent f=%.2fpx k1=%.4f (reprojection RMS %.3fpx on "
                    "the ring20 fit), derived from this rig's own frames "
                    "(opendarts.calibration.distortion).",
                    cam, result.n_frames_used, result.n_points_used, result.cx,
                    result.cx - width / 2.0, result.focal_length_px, result.k1,
                    result.reprojection_rms_px or float("nan"),
                )
            return result.focal_length_px, result.k1, result.cx, "live"

        if established_principal_point_source.get(cam) == "live":
            return (
                established_cx_focal_px[cam], established_cx_k1[cam],
                established_cx[cam], "live",
            )

        log.info(
            "cam%d: live principal-point-X derivation not yet confident (%s, %d "
            "frame(s) so far) -- will keep accumulating more before falling back.",
            cam, result.reason, frames_captured[cam],
        )

        if frames_captured[cam] < max_frames:
            return None, None, None, None # more rounds may still succeed -- keep retrying

        fallback_entry = principal_point_fallback.get(cam)
        if fallback_entry is not None:
            fb_f = float(fallback_entry["focal_length_px"])
            fb_k1 = float(fallback_entry["k1"])
            fb_cx = float(fallback_entry["cx"])
            established_principal_point_source[cam] = "fallback_json"
            log.warning(
                "cam%d: LIVE PRINCIPAL-POINT-X DERIVATION FAILED after the full "
                "%d-frame capture budget (%s) -- falling back to the persisted "
                "principal_point_fallback.json value (f=%.2fpx, k1=%.4f, cx=%.2fpx, "
                "derived %s from a PRIOR calibration event). This is a DEGRADED, "
                "non-fresh value -- expect it to self-correct the next time live "
                "derivation succeeds.",
                cam, frames_captured[cam], result.reason, fb_f, fb_k1, fb_cx,
                fallback_entry.get("derived_at_utc"),
            )
            return fb_f, fb_k1, fb_cx, "fallback_json"

        # NO fallback available either -- caller keeps using whatever the
        # k1-only tier already resolved, exactly today's (pre-this-
        # addition) behavior. Not an error: purely additive.
        established_principal_point_source[cam] = "unavailable"
        log.info(
            "cam%d: live principal-point-X derivation unavailable after the full "
            "%d-frame capture budget (%s), and no persisted "
            "principal_point_fallback.json entry exists for this camera index yet "
            "-- calibrating with the existing centered-principal-point assumption "
            "for this camera this pass.",
            cam, frames_captured[cam], result.reason,
        )
        return None, None, None, None

    def _try_solve(cam: int) -> CalibrationAttempt | None:
        """Attempt a PnP solve from `cam`'s full ACCUMULATED detection
        pool so far. Returns None if there still aren't enough usable
        detections to trust an average (caller decides whether to retry
        or give up based on frames_captured[cam] vs max_frames)."""
        detections = accumulated_detections[cam]
        min_required = _min_good_frames_cumulative(frames_captured[cam])
        averaged = average_correspondences(detections, min_required=min_required)
        if averaged is None:
            n_ok = sum(d is not None for d in detections)
            reasons: dict[str, int] = {}
            for r in accumulated_results[cam]:
                if not r.ok:
                    reasons[r.reason] = reasons.get(r.reason, 0) + 1
                elif r.orientation_ambiguous:
                    reasons["orientation ambiguous"] = reasons.get("orientation ambiguous", 0) + 1
            log.warning(
                "cam%d: only %d/%d total captured frame(s) so far have yielded a "
                "usable landmark detection+correspondence (need >= %d to trust an "
                "average)%s; rejection reasons: %s",
                cam,
                n_ok,
                frames_captured[cam],
                min_required,
                "" if frames_captured[cam] < max_frames else " -- giving up, frame cap reached",
                reasons or "(none recorded)",
            )
            return None
        object_points, image_points, n_used = averaged
        # dist_coeffs: WAS np.zeros(5) hardcoded, zero lens distortion
        # ASSUMED, not fit -- investigated 2026-08-12 whether fitting a
        # real, small Brown-Conrady k1/k2 from
        # available real data would be a well-conditioned, safe
        # improvement over this hardcoded zero. Real, honest non-result
        # AT THE TIME: the correspondence budget per camera was only 4
        # REAL independent points (correspond_landmarks() above; averaging
        # N frames reduces per-frame NOISE, it does not add new independent
        # geometric constraints). Solving for just 7 unknowns (f, rvec,
        # tvec) from 4 points (8 equations) leaves only 1 DOF of slack, "a
        # REAL, weakly-constrained problem, not a comfortable overdetermined
        # fit" -- adding even one more unknown (k1) would need 8 unknowns
        # from 8 equations, zero slack, not safely identifiable from this
        # rig's real correspondence budget.
        # SUPERSEDED, 2026-08-26 -- see opendarts.calibration.distortion's own module
        # docstring for the full conditioning analysis and docs/DESIGN.md's
        # dated entry for the real numbers. What changed: the SAME
        # detection pass now also feeds opendarts.calibration.focal_length's
        # ring20 correspondence (20 points, not 4) -- 40 equations for the
        # same 8 unknowns, a completely different, well-conditioned
        # regime, confirmed synthetically before being wired in below.
        # `established_k1[cam]` stays 0.0 (this line's own old hardcoded
        # behavior) for any camera this event's data doesn't safely
        # support a distortion fit for -- purely additive.
        # NEGOTIATED-RESOLUTION FIX, 2026-08-20 -- see
        # _negotiated_resolution_for()'s own docstring for the full bug
        # this closes and the backward-compatibility guarantee (identical
        # numbers on real hardware that negotiates the historical
        # 1280x720, exactly as MEASURED_FOCAL_LENGTH_PX was derived at).
        # `hub` is this function's own enclosing-scope parameter (None
        # when the caller has no hub, real in the default local-camera
        # path) -- not re-fetched or guessed here.
        width, height = _negotiated_resolution_for(cam, hub)
        # LIVE-DERIVED FOCAL LENGTH, 2026-08-26 -- see
        # _resolve_focal_length_px()'s own docstring for the full 3-tier
        # priority (live derivation -> persisted JSON fallback ->
        # hardcoded MEASURED_FOCAL_LENGTH_PX last resort) and
        # opendarts.calibration.focal_length's module docstring for why this
        # replaces `_camera_matrix_for(cam, ...)`'s own index-keyed
        # constant as the PRIMARY source. `focal_px is None` means every
        # tier came up empty for THIS round (live derivation not yet
        # confident and frames_captured[cam] < max_frames) -- treated
        # exactly like "not enough correspondence data yet", reusing the
        # existing adaptive-retry mechanism below rather than a new one.
        focal_px, _focal_source = _resolve_focal_length_px(cam, width, height)
        if focal_px is None:
            log.warning(
                "cam%d: no focal length available yet this round (live derivation "
                "not yet confident) -- treating like insufficient correspondence "
                "data, will keep retrying up to the frame cap.",
                cam,
            )
            return None
        # LIVE-DERIVED RADIAL DISTORTION, 2026-08-26 -- see
        # _resolve_distortion_for()'s own docstring for the full 2-tier
        # priority (live joint derivation -> persisted JSON fallback ->
        # k1=0.0, today's existing behavior) and
        # opendarts.calibration.distortion's module docstring for the
        # conditioning analysis this is built on. `distortion_focal_px`
        # is None when this round has nothing new to offer -- `focal_px`
        # (the existing homography-only resolution above) is used
        # unchanged in that case, exactly today's behavior.
        distortion_focal_px, k1, _distortion_source = _resolve_distortion_for(
            cam, width, height, focal_px
        )
        if distortion_focal_px is not None:
            focal_px = distortion_focal_px
        # LIVE-DERIVED PRINCIPAL POINT (cx only), 2026-08-26 -- see
        # _resolve_principal_point_cx_for()'s own docstring for the full
        # tier priority and opendarts.calibration.distortion's module
        # docstring ("PRINCIPAL POINT (cx only)" section) for the
        # conditioning analysis this is built on. `cx_focal_px` is None
        # when this round has nothing new to offer -- `focal_px`/`k1`
        # (the k1-only tier's own resolution above) are used unchanged in
        # that case, and `cx` stays None (build_camera_matrix() then
        # defaults to the image center, exactly as before this addition
        # existed).
        cx_focal_px, cx_k1, cx, _pp_source = _resolve_principal_point_cx_for(
            cam, width, height, focal_px
        )
        if cx_focal_px is not None:
            focal_px, k1 = cx_focal_px, cx_k1
        return calibrate_camera(
            object_points,
            image_points,
            build_camera_matrix(focal_px, image_width=width, image_height=height, cx=cx),
            np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            image_width=width,
            image_height=height,
        )

    def _frozen_intrinsics_for(cam: int) -> "tuple[float, float, float | None, float, float | None]":
        """One camera's (width, height, focal_px, k1, cx) as the resolver
        chain in `_try_solve_from_detections()` would produce them --
        computed ONCE per camera for the whole BEST-OF-N section
        (2026-09-11 calibration-speed pass). Safe to reuse across that
        section's attempts because nothing there mutates
        `accumulated_detections`/`accumulated_results`/`frames_captured`
        (the resolvers' only inputs besides the fixed resolution) -- the
        exact determinism `_try_solve_from_detections()`'s own docstring
        already documents and relies on. `focal_px is None` means the
        resolver chain has nothing to offer (each attempt would have
        returned None itself, and still does -- see the frozen gate in
        `_try_solve_from_detections()`)."""
        width, height = _negotiated_resolution_for(cam, hub)
        focal_px, _focal_source = _resolve_focal_length_px(cam, width, height)
        if focal_px is None:
            return (width, height, None, 0.0, None)
        distortion_focal_px, k1, _distortion_source = _resolve_distortion_for(
            cam, width, height, focal_px
        )
        if distortion_focal_px is not None:
            focal_px = distortion_focal_px
        cx_focal_px, cx_k1, cx, _pp_source = _resolve_principal_point_cx_for(
            cam, width, height, focal_px
        )
        if cx_focal_px is not None:
            focal_px, k1 = cx_focal_px, cx_k1
        return (width, height, focal_px, k1, cx)

    def _try_solve_from_detections(
        cam: int, detections: list, n_frames_in_batch: int,
        frozen_intrinsics: "tuple[float, float, float | None, float, float | None] | None" = None,
    ) -> CalibrationAttempt | None:
        """BEST-OF-N REPROJECTION ATTEMPTS' own solve -- see this
        function's own docstring section above. Deliberately a SEPARATE
        function from `_try_solve()` immediately above, not that
        function parameterized, to avoid any risk of touching
        `_try_solve()`'s own well-tested ADAPTIVE RETRY behavior: the
        two differ in exactly one respect, which detections list gets
        averaged. `_try_solve()` always reads `accumulated_detections
        [cam]` (the ever-growing, cross-round pool ADAPTIVE RETRY
        accumulates) and derives `min_required` from `frames_captured
        [cam]` (that camera's TOTAL frame count across the whole event
        so far); this function instead takes an explicit `detections`
        list -- ONE independent attempt's own batch, never merged with
        any other attempt's or ADAPTIVE RETRY's own pool -- and derives
        `min_required` from `n_frames_in_batch` (that one batch's own
        size), matching `_min_calibration_frames_required()`'s existing
        "a real average needs a real majority of THIS batch's own
        independent samples" reasoning at the right scale for a single
        small attempt instead of the whole event's cumulative total.

        Everything else -- resolution, focal length, distortion (k1),
        principal point (cx), the final `calibrate_camera()` call -- is
        read via the SAME already-established resolution helpers
        `_try_solve()` itself uses (`_negotiated_resolution_for()`,
        `_resolve_focal_length_px()`, `_resolve_distortion_for()`,
        `_resolve_principal_point_cx_for()`). Because this function never
        mutates `accumulated_detections[cam]`/`accumulated_results[cam]`
        (the pools those three resolvers read from), calling them here
        deterministically returns the SAME frozen values ADAPTIVE RETRY
        already established for this camera by the time BEST-OF-N runs
        -- i.e. orientation/focal/k1/cx are genuinely fixed across every
        best-of-N attempt for a camera, and only the pose fit (driven by
        THIS attempt's own averaged correspondence) varies, exactly the
        design this feature's own docstring section states."""
        min_required = _min_calibration_frames_required(n_frames_in_batch)
        averaged = average_correspondences(detections, min_required=min_required)
        if averaged is None:
            return None
        object_points, image_points, n_used = averaged
        # 2026-09-11 calibration-speed pass: `frozen_intrinsics`, if
        # given, is `_frozen_intrinsics_for(cam)`'s output -- the SAME
        # (width, height, focal_px, k1, cx) the resolver chain below
        # produces, computed once per camera for the whole BEST-OF-N
        # section instead of once per attempt. This is not a behaviour
        # change but this function's own docstring made explicit: it
        # already documents that, with the pools frozen, the resolvers
        # "deterministically return the SAME frozen values" on every
        # attempt -- each repeat call was re-running the full multi-seed
        # focal/k1/cx LM derivation suites just to reproduce a value
        # already in hand. None (the default) resolves inline, exactly
        # as before, for any caller outside that section.
        if frozen_intrinsics is not None:
            width, height, focal_px, k1, cx = frozen_intrinsics
            if focal_px is None:
                return None  # same as the inline chain's own focal gate below
        else:
            width, height = _negotiated_resolution_for(cam, hub)
            focal_px, _focal_source = _resolve_focal_length_px(cam, width, height)
            if focal_px is None:
                return None
            distortion_focal_px, k1, _distortion_source = _resolve_distortion_for(
                cam, width, height, focal_px
            )
            if distortion_focal_px is not None:
                focal_px = distortion_focal_px
            cx_focal_px, cx_k1, cx, _pp_source = _resolve_principal_point_cx_for(
                cam, width, height, focal_px
            )
            if cx_focal_px is not None:
                focal_px, k1 = cx_focal_px, cx_k1
        return calibrate_camera(
            object_points,
            image_points,
            build_camera_matrix(focal_px, image_width=width, image_height=height, cx=cx),
            np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            image_width=width,
            image_height=height,
        )

    def _accumulated_frame_indices(cam: int) -> list[int]:
        """Pool slots behind the accumulated solve: every recorded index
        whose detection actually survived. `accumulated_detections[cam]`
        holds None for a frame that was detected but yielded no usable
        correspondence (including the ellipse-aspect outlier reset, which
        rewrites entries in place and so preserves alignment)."""
        idxs = detected_pool_indices_by_det.get(cam, [])
        dets = accumulated_detections.get(cam, [])
        return [idx for idx, det in zip(idxs, dets) if det is not None]

    def _process_camera_round(cam: int, frames_this_round: list[np.ndarray]) -> bool:
        """The full per-camera round body -- detect, report, try_solve,
        accept/reject/best-tracking -- run on ONE worker thread for `cam`
        (see PARALLELIZED docstring section above for the thread-safety
        argument). Returns True if `cam` should stay in `remaining` for
        another round, False if it's done (target met, unevaluable and
        accepted as-is, PnP failure, or the max_frames cap was hit).
        Deliberately does NOT touch `remaining` itself -- that's the one
        genuinely shared object across cameras, mutated only by the
        single-threaded caller after every worker for this round has
        returned (see as_completed() loop below)."""
        _t_detect = time.monotonic()
        per_frame_results = _detect_batch(cam, frames_this_round)
        detect_time_s[cam] += time.monotonic() - _t_detect # SECTION TIMING, see _bootstrap_t0 above
        _report_round(cam, frames_this_round, per_frame_results)

        _t_solve = time.monotonic()
        attempt = _try_solve(cam)
        solve_time_s[cam] += time.monotonic() - _t_solve # SECTION TIMING, see _bootstrap_t0 above
        if attempt is None:
            # Not enough usable data yet -- keep retrying next round
            # unless the frame cap is already reached (gave up;
            # _try_solve() already logged why).
            return frames_captured[cam] < max_frames

        if not attempt.ok or attempt.calibration is None:
            log.warning("cam%d: calibration failed: %s", cam, attempt.reason)
            return False # a real PnP failure is not fixed by more frames

        reproj_px = attempt.pnp_result.reprojection_error_px if attempt.pnp_result else None
        if reproj_px is None:
            # Can't evaluate against the target (e.g. a test double
            # standing in for calibrate_camera()) -- accept as-is,
            # exactly today's old single-solve behavior.
            best_calibration[cam] = attempt.calibration
            best_reprojection_px[cam] = None
            adopted_frame_indices[cam] = _accumulated_frame_indices(cam)
            log.info(
                "cam%d: calibrated ok from %d total frame(s) (reprojection "
                "error unavailable, accepting as-is)",
                cam, frames_captured[cam],
            )
            return False

        if cam not in best_reprojection_px or best_reprojection_px[cam] is None or reproj_px < best_reprojection_px[cam]:
            best_calibration[cam] = attempt.calibration
            best_reprojection_px[cam] = reproj_px
            adopted_frame_indices[cam] = _accumulated_frame_indices(cam)

        cam_target_px = _target_px_for(cam)
        if reproj_px < cam_target_px:
            log.info(
                "cam%d: calibrated ok from %d total frame(s), "
                "reprojection_error_px=%.2f (met <%.1fpx target)",
                cam, frames_captured[cam], reproj_px, cam_target_px,
            )
            return False
        if frames_captured[cam] >= max_frames:
            log.warning(
                "cam%d: did NOT reach the <%.1fpx reprojection-error target -- "
                "stopped at the N=%d frame cap, best reprojection_error_px=%.2f "
                "(accepting the best result obtained, not failing outright)",
                cam, cam_target_px, frames_captured[cam],
                best_reprojection_px[cam],
            )
            return False
        # else: still above target and under the cap -- stays in
        # `remaining`, retried with more accumulated frames next round.
        return True

    round_num = 0
    # One worker per camera that could ever be `remaining` this pass --
    # `remaining`'s starting size (set in the setup loop above) is its
    # maximum for the whole function; it only ever shrinks. Sized to at
    # least 1 so an empty `remaining` (e.g. `frames_by_cam` itself came
    # back empty -- no cameras configured on `hub`, or the very first
    # capture round returned nothing for any camera -- the ONLY way
    # `remaining` can start empty now that the old per-camera-index
    # "no measured orientation hint, skipping" allowlist gate is gone,
    # see this function's own docstring "Real, related behaviour change
    # worth naming explicitly") doesn't raise on ThreadPoolExecutor's
    # own max_workers>=1 requirement -- the while loop below just never
    # runs in that case, so the pool sits idle and is shut down unused.
    max_workers = max(len(remaining), 1)
    _t_rounds = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="calib-cam") as pool:
        calibration_progress.PROGRESS.stage("solve")
        while remaining:
            round_num += 1
            cams_this_round = sorted(remaining)
            calibration_progress.PROGRESS.stage("solve", f"round {round_num}" if round_num > 1 else None)
            for _cam in cams_this_round:
                calibration_progress.PROGRESS.camera(_cam, "working")
            # round 1 detects only `frames_by_cam_detect` (default the
            # first 10 of each camera's raw frames), not the full
            # `frames_by_cam` raw capture -- see DECOUPLED CAPTURE-VS-
            # DETECT TARGET above; every round after that (retries) is
            # unchanged, still a fresh `new_batch` of `retry_batch_size`
            # captured-AND-detected frames.
            frames_by_cam_this_round = {
                cam: (frames_by_cam_detect.get(cam, []) if round_num == 1 else new_batch.get(cam, []))
                for cam in cams_this_round
            }
            futures = {
                pool.submit(_process_camera_round, cam, frames_by_cam_this_round[cam]): cam
                for cam in cams_this_round
            }
            for future in as_completed(futures):
                cam = futures[future]
                stay_in_remaining = future.result()
                if not stay_in_remaining:
                    remaining.discard(cam) # main-thread-only mutation, see docstring above
                    calibration_progress.PROGRESS.camera(cam, "done" if cam in best_calibration else "failed")
                    camera_end_time[cam] = time.monotonic() # TIMING, see _bootstrap_t0 above

            # MODE A (spec R2.1), 2026-08-29 -- runs exactly once, right
            # here, only when round 1 just finished AND this rig already
            # has ring geometry matching this event's own camera count.
            # See `_resolve_mode_a_orientation_after_round_one()`'s own
            # docstring for the full design; this is its ONLY call site.

            # CROSS-CAMERA ELLIPSE-ASPECT VALIDATION (Part B of the
            # 2026-09-03 cam0 ellipse-merge defect fix), runs every round
            # for every orientation method -- deliberately UNCONDITIONAL,
            # unlike Mode A above (which only runs with
            # matching ring geometry): this is a general seed-ellipse
            # sanity check, not tied to any one orientation method. See
            # `_check_cross_camera_ellipse_aspect_consistency()`'s own
            # docstring for the full design; this is its ONLY call site,
            # placed AFTER Mode A so it operates on whatever `accumulated_
            # detections`/`pre_orientation_pool` state is actually live
            # once this round (including any Mode A whole-pool rebuild)
            # has settled.
            _check_cross_camera_ellipse_aspect_consistency(cams_this_round, frames_by_cam_this_round)

            if remaining:
                _t = time.monotonic()
                new_batch = _capture(retry_batch_size)
                capture_time_s += time.monotonic() - _t # SECTION TIMING, see _bootstrap_t0 above

    rounds_wall_s = time.monotonic() - _t_rounds
    _t_bestof = time.monotonic()

    # BEST-OF-N REPROJECTION ATTEMPTS, added 2026-08-30 -- see this
    # function's own docstring section above for the full design.
    # `n_reprojection_attempts <= 1` (the default, and every existing
    # caller/test today) makes this a complete no-op -- zero behavior
    # change versus before this feature existed. The ADAPTIVE RETRY loop
    # immediately above already produced this camera's own attempt #1
    # (whatever `best_calibration[cam]`/`best_reprojection_px[cam]` are
    # right now); this section runs `n_reprojection_attempts - 1` MORE
    # independent attempts per already-calibrated camera, on genuinely
    # fresh `_capture()` frames, and keeps whichever attempt -- including
    # this one -- had the lowest reprojection_error_px.
    if n_reprojection_attempts > 1:
        best_of_n_eligible = sorted(best_calibration.keys())
        if best_of_n_eligible:
            log.info(
                "best-of-%d reprojection mode: cam(s) %s already calibrated "
                "via the adaptive-retry loop above -- running %d more "
                "independent attempt(s) each on fresh frames (orientation/"
                "focal-length/distortion/cx frozen at their already-"
                "established values for this section)",
                n_reprojection_attempts, best_of_n_eligible,
                n_reprojection_attempts - 1,
            )
            # 2026-09-11 calibration-speed pass: resolve each camera's
            # frozen (f, k1, cx) ONCE for the whole section -- see
            # _frozen_intrinsics_for()'s own docstring. Before this,
            # every attempt re-ran the full multi-seed focal/k1/cx LM
            # derivation suites on the frozen pools just to reproduce
            # the identical values (their determinism here is this
            # section's own documented design invariant).
            frozen_intrinsics_by_cam = {
                cam: _frozen_intrinsics_for(cam) for cam in best_of_n_eligible
            }
            with ThreadPoolExecutor(
                max_workers=max(len(best_of_n_eligible), 1),
                thread_name_prefix="calib-bestn",
            ) as bestn_pool:
                for _attempt_num in range(2, n_reprojection_attempts + 1):
                    calibration_progress.PROGRESS.stage(
                        "refine", f"attempt {_attempt_num - 1} of {n_reprojection_attempts - 1}",
                        (_attempt_num - 2) / max(1, n_reprojection_attempts - 1))
                    _t = time.monotonic()
                    attempt_frames = _capture(n_frames_detect)
                    capture_time_s += time.monotonic() - _t # SECTION TIMING, see _bootstrap_t0 above

                    def _run_one_best_of_n_attempt(cam: int) -> None:
                        """One independent best-of-N attempt for `cam`,
                        run on its own worker thread -- mirrors
                        `_process_camera_round()`'s own per-camera thread-
                        safety argument (this closure only ever touches
                        `pre_orientation_pool[cam]`, `detect_time_s[cam]`,
                        `solve_time_s[cam]`, `best_calibration[cam]`,
                        `best_reprojection_px[cam]` -- distinct dict keys
                        per camera, safe under the GIL for the same
                        reason this file's own PARALLELIZED ACROSS
                        CAMERAS docstring section already documents)."""
                        frames = attempt_frames.get(cam, [])
                        if not frames:
                            return
                        _t_detect = time.monotonic()
                        local_detections: list = []
                        # Pool slots this attempt occupies, recorded in
                        # lockstep with local_detections so an attempt
                        # that WINS can publish the frames behind it (see
                        # adopted_frame_indices). Deliberately NOT added
                        # to detected_pool_indices_by_det: these
                        # detections never enter accumulated_detections,
                        # and pretending otherwise is precisely the
                        # mismatch the old invariant check tripped on.
                        local_pool_indices: list[int] = []
                        for frame in frames:
                            processed_bgr, pre = locate_pre_orientation_landmarks(frame)
                            # Real captured frame -- feed the raw-pixel
                            # pool the two post-loop sections (ring-
                            # boundary-offset, board-color) consume, same
                            # as every other real frame this event
                            # captures (see this function's own docstring
                            # section above, "genuine ... accuracy bonus,
                            # not a cost"). Deliberately does NOT call
                            # `_establish_hint_if_possible()` or append to
                            # the orientation evidence -- this
                            # section must never re-open orientation
                            # resolution, which is already fully settled
                            # by the time BEST-OF-N runs (see this
                            # function's own docstring, requirement that
                            # orientation resolve exactly once per event).
                            frame_pool_index = len(pre_orientation_pool[cam])
                            pre_orientation_pool[cam].append((processed_bgr, pre))
                            if not pre.ok:
                                continue
                            detection = correspond_landmarks_from_pre_orientation(
                                processed_bgr, pre,
                                orientation_hint_deg=established_hint_deg.get(cam),
                                results_out=[],
                            )
                            local_detections.append(detection)
                            local_pool_indices.append(frame_pool_index)
                        detect_time_s[cam] += time.monotonic() - _t_detect # SECTION TIMING, see _bootstrap_t0 above

                        _t_solve = time.monotonic()
                        attempt = _try_solve_from_detections(
                            cam, local_detections, len(frames),
                            frozen_intrinsics=frozen_intrinsics_by_cam[cam],
                        )
                        solve_time_s[cam] += time.monotonic() - _t_solve # SECTION TIMING, see _bootstrap_t0 above
                        if attempt is None or not attempt.ok or attempt.calibration is None:
                            # This attempt yielded no usable candidate --
                            # not a failure of the camera overall, just
                            # this one independent sample; the running
                            # best (attempt #1 at minimum) is untouched.
                            return
                        reproj_px = (
                            attempt.pnp_result.reprojection_error_px
                            if attempt.pnp_result else None
                        )
                        if reproj_px is None:
                            return
                        current_best = best_reprojection_px.get(cam)
                        if current_best is None or reproj_px < current_best:
                            best_calibration[cam] = attempt.calibration
                            best_reprojection_px[cam] = reproj_px
                            # THIS attempt's frames are the ones behind
                            # the adopted calibration now, not the
                            # accumulated pool's.
                            adopted_frame_indices[cam] = [
                                idx for idx, det in zip(local_pool_indices, local_detections)
                                if det is not None
                            ]
                            log.info(
                                "cam%d: best-of-%d attempt improved the adopted "
                                "reprojection_error_px to %.2f",
                                cam, n_reprojection_attempts, reproj_px,
                            )

                    futures = [
                        bestn_pool.submit(_run_one_best_of_n_attempt, cam)
                        for cam in best_of_n_eligible
                    ]
                    for f in futures:
                        f.result()

    # TIMING, continued from _bootstrap_t0 above. `camera_duration_s`
    # covers every camera bootstrap_calibrations() ever attempted (even
    # one that ultimately failed to calibrate at all -- best_calibration
    # won't have it, but knowing THAT camera burned 18s before giving up
    # is exactly the kind of number this was asked for) -- falls back to
    # "now" for the pathological case of a camera that's still in
    # `remaining` after the while loop exits (cannot happen today, the
    # loop only exits once `remaining` is empty, but a duration is
    # always safer than a KeyError if that ever changes). Deliberately
    # measured HERE (right after the retry loop), not at the true end of
    # the function -- this is genuinely "how long was this camera
    # actively being calibrated" (capture + its own detect/solve rounds),
    # a different, smaller quantity than the whole call's own total
    # (see `total_duration_s` below, computed separately and later, for
    # a REAL, honest bug this project's own retrospective caught: an
    # earlier version of this instrumentation computed "TOTAL" here too,
    # before the three post-loop live-derived-value sections had even
    # run -- a real live measurement (2026-08-21) caught a 52.25s
    # "TOTAL" sitting 41 SECONDS before the section-timing log line that
    # revealed ring_boundary_offset alone took another 40.28s -- i.e.
    # the old "TOTAL" undercounted the real end-to-end time by nearly
    # 2x. Fixed by moving the real total's own measurement point to
    # after every section actually finishes, not renaming/removing this
    # per-camera number, which was always correct for what IT claims to
    # measure.
    bestof_wall_s = time.monotonic() - _t_bestof

    _calibration_end_t = time.monotonic()
    camera_duration_s: dict[int, float] = {
        cam: round(camera_end_time.get(cam, _calibration_end_t) - _bootstrap_t0, 3)
        for cam in frames_by_cam
    }

    # ORIENTATION RESOLUTION -- FINAL REFUSAL CHECK + RING GEOMETRY
    # LEARNING (spec R1/R3, 2026-08-29). Mode A's own resolution (or
    # refusal) already ran synchronously right after round 1, above --
    # this section covers two things that can ONLY be decided once the
    # WHOLE retry loop has actually finished:
    #
    # 1. Mode B's refusal (spec R2.1: "no stored geometry yet" --
    # every camera present this event MUST have derived its
    # orientation genuinely LIVE, since there is no geometry to
    # fill a straggler in from). A camera still unresolved here
    # means live derivation never cleared the confidence floor even
    # after the FULL `max_frames` budget -- refuse the WHOLE
    # bootstrap (spec R1) rather than silently proceeding with
    # fewer cameras, exactly like the deleted hardcoded-fallback
    # path used to silently paper over.
    # 2. Ring geometry LEARNING (spec R3) -- "updated whenever a
    # calibration lands with all cameras live and mutually
    # consistent." Applies regardless of which mode ran: the common
    # trigger is Mode B (a virgin rig, or one recovering from a
    # "camera moved" refusal, reaching all-live for the first
    # time), but Mode A can ALSO occasionally hit this (every
    # camera happened to clear the floor AND pass the D2
    # cross-check this event, nothing needed filling) -- a real
    # bonus re-learning opportunity, not required for Mode A to
    # succeed. `established_hint_source[cam] == "live"` is the only
    # condition checked -- a `"rig_consensus"`-sourced camera is
    # deliberately EXCLUDED from this trigger (feeding a
    # consensus-PREDICTED value back into learning the geometry it
    # was predicted FROM would be circular).
    # LIVE-DERIVED FOCAL LENGTH -- PERSIST every SUCCESSFUL live
    # derivation to `focal_length_fallback.json`, 2026-08-26. Only cameras
    # whose FINAL `established_focal_source` is `"live"` are written --
    # never a `"fallback_json"`/`"fallback_constant"` result (that would
    # let a degraded value silently become the new "last known good",
    # defeating the self-correcting property this file exists for -- see
    # `opendarts.calibration.focal_length.write_focal_length_fallback_entry()`'s
    # own docstring). A camera that never even entered `best_calibration`
    # this pass (PnP failed outright, or no focal length was ever
    # resolvable) has nothing worth persisting either. No-op entirely
    # when `calibration_package_root` is None (mirrors this function's
    # own existing `calibration_package_out`/package-saving guard just
    # below) -- there is nowhere durable to write to in that case.
    if calibration_package_root is not None:
        _derived_at_utc = datetime.now(timezone.utc).isoformat()
        for cam in best_calibration:
            if established_focal_source.get(cam) != "live":
                continue
            try:
                # package_id=None: `new_calibration_package_id()` hasn't
                # run yet at this point in the function (it's generated
                # much later, just before `save_calibration_package()`
                # below) -- not worth restructuring that ordering just to
                # backfill a purely diagnostic/traceability field here.
                write_focal_length_fallback_entry(
                    calibration_package_root, cam, established_focal_px[cam],
                    n_frames_used=frames_captured[cam],
                    n_points_used=20,
                    derived_at_utc=_derived_at_utc,
                    package_id=None,
                )
            except Exception: # noqa: BLE001 -- never break calibration over this
                log.exception(
                    "cam%d: failed to persist live-derived focal length to "
                    "focal_length_fallback.json -- this event's own calibration "
                    "is unaffected, only the NEXT event's degraded-fallback tier "
                    "will be missing this update",
                    cam,
                )

        # LIVE-DERIVED RADIAL DISTORTION -- PERSIST every SUCCESSFUL live
        # joint derivation to `distortion_fallback.json`, 2026-08-26 --
        # same "only a genuinely fresh 'live' result, never a degraded
        # fallback-sourced one" gate as the focal-length persistence just
        # above, independent file (see opendarts.calibration.distortion's own
        # module docstring).
        for cam in best_calibration:
            if established_distortion_source.get(cam) != "live":
                continue
            try:
                write_distortion_fallback_entry(
                    calibration_package_root, cam,
                    established_joint_focal_px[cam], established_k1[cam],
                    n_frames_used=frames_captured[cam],
                    n_points_used=20,
                    derived_at_utc=_derived_at_utc,
                    package_id=None,
                )
            except Exception: # noqa: BLE001 -- never break calibration over this
                log.exception(
                    "cam%d: failed to persist live-derived distortion to "
                    "distortion_fallback.json -- this event's own calibration "
                    "is unaffected, only the NEXT event's degraded-fallback tier "
                    "will be missing this update",
                    cam,
                )

        # LIVE-DERIVED PRINCIPAL POINT (cx only) -- PERSIST every
        # SUCCESSFUL live joint derivation to `principal_point_fallback.json`,
        # 2026-08-26 -- same "only a genuinely fresh 'live' result" gate,
        # its OWN independent file (see opendarts.calibration.distortion's
        # own module docstring, "PRINCIPAL POINT (cx only)" section).
        for cam in best_calibration:
            if established_principal_point_source.get(cam) != "live":
                continue
            try:
                write_principal_point_fallback_entry(
                    calibration_package_root, cam,
                    established_cx_focal_px[cam], established_cx_k1[cam], established_cx[cam],
                    n_frames_used=frames_captured[cam],
                    n_points_used=20,
                    derived_at_utc=_derived_at_utc,
                    package_id=None,
                )
            except Exception: # noqa: BLE001 -- never break calibration over this
                log.exception(
                    "cam%d: failed to persist live-derived principal point to "
                    "principal_point_fallback.json -- this event's own calibration "
                    "is unaffected, only the NEXT event's degraded-fallback tier "
                    "will be missing this update",
                    cam,
                )

    # LIVE-DERIVED PER-CAMERA BOARD-DISC DETECTION-REGION MASK -- the one
    # detection-side quantity derived from a calibration. Pure geometry
    # from this event's own just-solved calibration (`opendarts.capture.
    # board_disc.board_disc_mask()`); the lifecycle splits every frame's
    # changed pixels into board/outside with it. Non-fatal on any failure,
    # but never leaves a STALE mask from a PRIOR calibration registered for
    # a camera this event didn't re-derive one for: `set_calibrated_board_
    # disc_masks(None)` clears every camera (a partial dict would keep
    # serving a camera's OLD board position after a recalibration moved
    # it). With no masks the lifecycle judges nothing until the next
    # successful calibration -- and says so. `motion_threshold_time_s`
    # keeps its calibration-package key (`motion_threshold_duration_s`).
    _t_motion = time.monotonic() # SECTION TIMING, see _bootstrap_t0 above
    calibration_progress.PROGRESS.stage("finish")
    try:
        from opendarts.capture.board_disc import board_disc_mask

        board_disc_masks: dict[int, np.ndarray] = {}
        for cam, calib in best_calibration.items():
            cam_width, cam_height = _negotiated_resolution_for(cam, hub)
            mask = board_disc_mask(calib, cam_width, cam_height)
            if mask is None:
                log.warning(
                    "live board-disc mask derivation: camera %d's calibration "
                    "produced a degenerate projection (e.g. a boundary point "
                    "behind the camera) -- no throw detection on this camera "
                    "until the next successful calibration",
                    cam,
                )
                continue
            board_disc_masks[cam] = mask
        set_calibrated_board_disc_masks(board_disc_masks or None)
    except Exception: # noqa: BLE001 -- never break calibration over this
        log.exception(
            "live per-camera board-disc mask derivation failed -- no throw "
            "detection until the next successful calibration"
        )
        set_calibrated_board_disc_masks(None)
    motion_threshold_time_s = time.monotonic() - _t_motion

    # LIVE-DERIVED RING-BOUNDARY OFFSET + BOARD-COLOR THRESHOLDS, wired
    # 2026-08-21 (closing a real gap found and flagged live the same
    # night: opendarts.calibration.ring_boundary_offset and opendarts.geometry.
    # board_color_calibration were correct, tested code that NOTHING
    # ever actually triggered or applied -- every throw scored so far
    # used the generic hardcoded defaults (INNER_RING_SCORING_OFFSET_MM
    # =1.5mm, board-color thresholds 129.0/35.0), never anything measured
    # from THIS rig's own board. "you just invalidated our
    # entire last test session because you used numbers that were hard
    # coded and [not] derived from the board").
    #
    # Both derivations' own session-level wrapper functions
    # (measure_session_ring_boundary_offsets()/derive_session_board_
    # color_calibration()) require a session's worth of ALREADY-SAVED
    # throw packages to read bg frames back from -- but that's a
    # convenience/plumbing choice in those wrappers, not a real
    # constraint of the underlying measurement: the actual pure
    # functions (measure_ring_boundary_offsets()/collect_color_samples()
    # + derive_thresholds()) take a plain {camera: CameraCalibration}
    # dict and a plain list of bg frames -- exactly what THIS calibration
    # burst already has in `best_calibration`/`pre_orientation_pool`, the
    # same frame pool orientation-hint/intrinsics/motion-thresholds
    # above already use. No chicken-and-egg problem, no need to wait for
    # real thrown darts to exist -- called directly here instead of
    # going through the package-reading wrappers.
    #
    # Confidence-gating philosophy, same lesson as tonight's
    # PIXEL_DIFF_THRESHOLD incident above: never adopt a live-derived
    # value with no floor. Ring-boundary-offset: gated on BOTH inner
    # boundaries producing a real (non-None) measurement at all -- the
    # SAME acceptance bar `write_session_ring_boundary_offset()` already
    # uses to decide whether its own offline measurement is worth
    # writing to disk (which itself already enforces
    # MIN_PROFILES_PER_CAMERA=10 internally before a boundary's
    # `measured_radius_mm` comes back non-None) -- not a newly-invented
    # threshold. Board-color: gated per-threshold on `confidence ==
    # "high"`, the SAME categorical gate `load_throw_package()`'s own
    # offline application of this data already uses. Neither gate is a
    # fabricated number -- both reuse bars this codebase already
    # established and shipped elsewhere. Honest caveat: unlike the area
    # motion-thresholds above, this hasn't yet been validated against a
    # real live calibration burst on this rig (no recorded data existed at
    # the time this was wired) -- confidence/sample-count values are
    # logged at INFO on every calibration specifically so that first
    # real run is visible for review, not silently trusted.
    # REPLAY PERSISTENCE (docs/DESIGN.md's "Replay is the source of truth"), 2026-08-22 -- these
    # six variables are the actual measured ring-boundary-offset values,
    # carried out of the try block below (which may raise or reject
    # before ever assigning them) so `per_cam_diagnostics` can always
    # reference them safely and so `save_calibration_package()` ends up
    # writing whatever was REALLY applied to `board.py`'s live globals
    # this calibration event into `derived_calibration.json` -- closing
    # the real gap a verifier pass found the same night this was wired:
    # the live-derived offset was applied in memory (via
    # `set_ring_boundary_offsets()` below) but never once written
    # anywhere a later, fresh-process REPLAY could find it, so replaying
    # an old package silently fell back to the regulation-default
    # INNER_RING_SCORING_OFFSET_MM instead of reproducing the value that
    # actually scored the throw live. `*_confidence`/`*_n_samples_used`
    # are recorded even when the offset itself was REJECTED (absent
    # value, present diagnostics) -- same "log what was measured even
    # when not adopted" posture the INFO log line below already has.
    #
    # STOP APPLYING, 2026-09-02 (see docs/DESIGN.md's dated entry for
    # the full decisive-test data).
    # `treble_inner_offset_mm`/`double_inner_offset_mm` below are STILL
    # the real measured values -- this section keeps measuring and
    # recording them (real diagnostic value; the underlying washout-ramp
    # mechanism is still an open question) -- but neither one is ever
    # passed to `set_ring_boundary_offsets()` as a non-None argument any
    # more, on this or any other live/replay path. The board's own wire
    # position images correctly; the 1.5mm gap between wire position and
    # the real scoring transition is TIP-MEASUREMENT BIAS, not a
    # per-board geometry fact -- a hypothetically perfect wire detector
    # scores exactly as badly (on the brief's own 241-throw decisive
    # test) as this rig's own worst live drift, so no confidence/
    # acceptance gate on the MEASUREMENT can fix this: the measurement
    # itself is answering the wrong question. `INNER_RING_SCORING_
    # OFFSET_MM` (opendarts/geometry/board.py, 1.5mm) stands for both inner
    # wires unconditionally now. `ring_boundary_offset_accepted` is kept
    # as a pure MEASUREMENT-QUALITY diagnostic (did both inner boundaries
    # produce a real, non-None measurement this event -- the same joint
    # gate this project has always used to decide whether a measurement
    # is worth recording/trusting) -- it no longer means "was applied to
    # live scoring," since nothing ever is.
    treble_inner_offset_mm: float | None = None
    treble_inner_confidence: float | None = None
    treble_inner_n_samples_used: int | None = None
    double_inner_offset_mm: float | None = None
    double_inner_confidence: float | None = None
    double_inner_n_samples_used: int | None = None
    ring_boundary_offset_accepted = False
    # FULL-BOUNDARY STORAGE GAP FIX, 2026-08-26. Before this, only
    # the two INNER boundaries' offset_mm/confidence/n_samples_used (3 of
    # ~9 fields, on 2 of 4 boundaries) ever left this function -- everything
    # else `measure_ring_boundary_offsets()` computes (treble_outer/
    # double_outer entirely; every boundary's mad_mm, n_samples_rejected,
    # n_angles_attempted, and per-camera radius/n_profiles/mode breakdown)
    # was computed, logged maybe, then discarded when this function
    # returns -- so a live calibration package could never self-diagnose
    # a real localizer divergence a QA pass found (two independent
    # ring-boundary localizers disagreeing by up to 0.76mm on the
    # SAME pixels/pose, flipping at least one real dart's double/
    # single_outer call). `ring_boundary_measurement` carries ALL 4
    # boundaries' full `BoundaryMeasurement` data (via
    # `boundary_measurement_to_payload()`, the SAME field vocabulary the
    # offline session-level `ring_boundary_offset.json` already uses --
    # deliberately not a second, differently-named shape) into
    # `per_cam_diagnostics` below and from there into
    # `calibration_package.py`'s `derived_calibration.json`. Populated
    # whenever the measurement itself succeeds, REGARDLESS of whether the
    # two inner boundaries were accepted -- same "record what was
    # measured even when not adopted" posture the existing treble/double
    # inner confidence/n_samples_used fields above already have. Does NOT
    # include `median_profiles` (the raw per-angle derivation records) --
    # those are comparatively large and already have a durable home in
    # the offline session-level file for deep replay; this per-event
    # package captures the aggregate per-boundary/per-camera numbers a
    # live diagnostic actually needs.
    ring_boundary_measurement: dict | None = None
    # THE V2 PACKAGE SCHEMA (2026-08-27) -- the FULL nested
    # RingBoundaryOffsetResult payload (schema/solved_by/
    # calibration_source/source_images/parameters/boundaries/
    # median_profiles), the canonical `derived_calibration.json` shape
    # field-for-field -- distinct from `ring_boundary_measurement`
    # above, which stays exactly as it was (the flat per-event
    # `{boundary_name: payload}` dict this project's existing 2026-08-26
    # storage-gap fix already ships and every existing reader/test
    # already depends on). This is purely additive: a NEW top-level
    # nested key (`derived_calibration.json`'s own `ring_boundary_offset`
    # key, see `calibration_package.py`'s pull-up) sitting alongside the
    # untouched flat keys, not a replacement for them -- see this
    # module's own docs/DESIGN.md entry for the full additive-vs-replace
    # reasoning.
    ring_boundary_offset_payload: dict | None = None
    # LIVE RING-BOUNDARY-OFFSET MEASUREMENT REMOVED, 2026-09-13.
    #
    # It was the single largest phase of a calibration -- 10.6-11.5s of a
    # 31s run on the Windows rig, about a third of the whole event -- and
    # its output was never used. `set_ring_boundary_offsets(None, None)`
    # below is unconditional and always was (see the 2026-09-02 decision
    # it implements: "the fix is never apply, not apply more carefully",
    # because confidence did not predict correctness). The measured value
    # went only into diagnostics and derived_calibration.json, under a
    # log line that said so in as many words -- "recorded for research
    # only and is NOT applied to scoring" -- and its own failure warning
    # read "irrelevant to scoring either way".
    #
    # Removed: a shipping product should not spend a third of every
    # calibration deriving a number it discards.
    #
    # WHAT STILL MEASURES IT, checked rather than assumed:
    #   * The offline measurement and the offline session refit that
    #     drives it still write ring_boundary_offset.json, and the
    #     corpus analyses still measure directly. They are developer
    #     tools, not part of the live path.
    # And what does NOT, contrary to the obvious assumption: calibration
    #     replay, because it re-runs THIS function against stored frames
    #     rather than measuring on its own. Replaying a package no longer
    #     yields a ring payload. That is a real consequence of removing
    #     this, stated plainly rather than discovered later.
    # The package schema keeps its fields either way -- they are simply
    # absent from a live calibration now, exactly as they already were
    # whenever the measurement failed.
    from opendarts.geometry.board import set_ring_boundary_offsets

    # Kept, and still unconditional: this is what pins scoring to the
    # hardcoded INNER_RING_SCORING_OFFSET_MM default. A CalibrationStore
    # snapshot restore (see _restore_from_snapshot) calls the same setter
    # with whatever it persisted, so resetting here is what guarantees a
    # fresh calibration cannot inherit a stale offset.
    set_ring_boundary_offsets(None, None)

    # REPLAY PERSISTENCE, same reasoning as the ring-boundary-offset
    # variables above -- carried out of the try block so a rejection or
    # exception still leaves these safely None (absent, never
    # fabricated) for `per_cam_diagnostics` to reference unconditionally.
    # Independent per-threshold (unlike ring-boundary-offset's joint
    # gate) -- matches this codebase's own existing per-threshold
    # confidence discipline (`load_throw_package()`'s session-level
    # application below, and the live gate two lines down).
    brightness_threshold: float | None = None
    brightness_threshold_confidence: str | None = None
    chroma_threshold: float | None = None
    chroma_threshold_confidence: str | None = None
    # THE V2 PACKAGE SCHEMA (2026-08-27) -- the FULL nested
    # BoardColorCalibrationResult payload, same additive treatment as
    # `ring_boundary_offset_payload` immediately above (see that
    # variable's own comment) -- the canonical
    # `board_color_calibration` shape field-for-field, sitting alongside
    # (never replaces) the existing flat `brightness_threshold`/
    # `chroma_threshold`/`*_confidence` keys.
    board_color_calibration_payload: dict | None = None
    _t_color = time.monotonic() # SECTION TIMING, see _bootstrap_t0 above
    try:
        from opendarts.geometry.board_color import set_board_color_thresholds
        from opendarts.geometry.board_color_calibration import (
            collect_color_samples,
            derive_thresholds,
            project_reference_points_px,
            reference_points,
            result_to_payload as board_color_result_to_payload,
        )

        points = reference_points()
        color_samples = []
        for cam, calib in best_calibration.items():
            # 2026-09-11 calibration-speed pass: the 82 reference-point
            # projections depend only on (point, calibration) -- project
            # once per camera instead of once per pooled frame (~50x
            # redundant cv2.projectPoints calls before). Same numbers:
            # see project_reference_points_px()'s own docstring.
            projected = {cam: project_reference_points_px(calib, points)}
            for i, (pb, _pre) in enumerate(pre_orientation_pool.get(cam, [])):
                color_samples.extend(
                    collect_color_samples(
                        f"live_cam{cam}_frame{i}", {cam: calib}, {cam: pb}, points=points,
                        projected_px_by_camera=projected,
                    )
                )
        color_result = derive_thresholds(color_samples)
        board_color_calibration_payload = board_color_result_to_payload(
            color_result,
            solved_by=(
                "opendarts.geometry.board_color_calibration "
                "(live, from this calibration event's own captured frames)"
            ),
        )
        log.info(
            "live board-color measurement: brightness value=%s confidence=%s "
            "| chroma value=%s confidence=%s (from %d sample(s))",
            color_result.brightness_threshold.value, color_result.brightness_threshold.confidence,
            color_result.chroma_threshold.value, color_result.chroma_threshold.confidence,
            len(color_samples),
        )
        brightness_threshold_confidence = color_result.brightness_threshold.confidence
        chroma_threshold_confidence = color_result.chroma_threshold.confidence
        # Loud-on-rejection, added 2026-08-21 (OpenDarts pre-hardware-audit
        # finding, ported back the other direction: their port closed a
        # real asymmetry this side shipped with -- ring-boundary-offset
        # already warns by name when a boundary is rejected; board-color
        # only ever logged the INFO-level measurement line above,
        # regardless of whether a threshold actually got adopted or
        # silently fell back. Same posture now, per-threshold.
        if color_result.brightness_threshold.confidence != "high":
            log.warning(
                "live board-color brightness threshold rejected (confidence=%s, "
                "value=%s) -- keeping the hardcoded BRIGHTNESS_THRESHOLD_BLACK_CREAM "
                "default for this calibration",
                color_result.brightness_threshold.confidence, color_result.brightness_threshold.value,
            )
        else:
            brightness_threshold = color_result.brightness_threshold.value
        if color_result.chroma_threshold.confidence != "high":
            log.warning(
                "live board-color chroma threshold rejected (confidence=%s, "
                "value=%s) -- keeping the hardcoded CHROMA_THRESHOLD default "
                "for this calibration",
                color_result.chroma_threshold.confidence, color_result.chroma_threshold.value,
            )
        else:
            chroma_threshold = color_result.chroma_threshold.value
        set_board_color_thresholds(
            brightness_threshold=brightness_threshold,
            chroma_threshold=chroma_threshold,
        )
    except Exception: # noqa: BLE001 -- never break calibration over this
        log.exception(
            "live board-color threshold derivation failed -- board_color.py "
            "keeps its hardcoded defaults"
        )
        try:
            from opendarts.geometry.board_color import set_board_color_thresholds
            set_board_color_thresholds(None, None)
        except Exception: # noqa: BLE001
            pass
    board_color_time_s = time.monotonic() - _t_color

    # REAL total, measured HERE -- after every section, including the
    # three post-loop live-derived-value ones, has actually finished.
    # See camera_duration_s's own comment above for the honest story of
    # why this moved: a first version of this instrumentation computed
    # "TOTAL" right after the retry loop, before these three sections
    # ever ran -- undercounting the real end-to-end time by whatever
    # they took (a real live measurement caught this at nearly 2x,
    # ring_boundary_offset alone accounting for 40+s the old "TOTAL"
    # never saw).
    total_duration_s = round(time.monotonic() - _bootstrap_t0, 3)
    log.info(
        "bootstrap_calibrations: TOTAL %.2fs | per-camera: %s",
        total_duration_s,
        ", ".join(f"cam{cam}={camera_duration_s[cam]:.2f}s" for cam in sorted(camera_duration_s)),
    )

    # SECTION TIMING summary log, see _bootstrap_t0 above -- the actual
    # answer to "time each section first": how much of the total went
    # into camera I/O vs CV detection vs the PnP solve vs each of the
    # three post-loop live-derived-value sections. capture_time_s is
    # shared/cumulative (not per-camera, capture happens once per round
    # for all cameras together); detect/solve are summed across cameras
    # here for one headline number, with the real per-camera breakdown
    # available in per_cam_diagnostics below for anyone who needs it.
    # PHASE WALL CLOCK -- disjoint sequential spans that sum to the total,
    # with the remainder stated rather than left to subtraction. Read this
    # one for "where did the time go"; the SECTION line below breaks the
    # rounds span down further but its detect/solve figures are SUMS
    # ACROSS CAMERAS running in parallel, so they exceed wall clock and
    # cannot be compared against these.
    phase_wall_s = {
        "setup+capture": round(setup_wall_s, 3),
        "orientation": round(orientation_wall_s, 3),
        "rounds": round(rounds_wall_s, 3),
        "best_of_n": round(bestof_wall_s, 3),
        "motion_thresholds": round(motion_threshold_time_s, 3),
        "board_color": round(board_color_time_s, 3),
    }
    unaccounted_wall_s = round(total_duration_s - sum(phase_wall_s.values()), 3)
    phase_wall_s["unaccounted"] = unaccounted_wall_s
    log.info(
        "bootstrap_calibrations PHASE WALL CLOCK (sums to TOTAL %.2fs): %s",
        total_duration_s,
        " | ".join(f"{name}={secs:.2f}s" for name, secs in phase_wall_s.items()),
    )

    # SECTION TIMING summary log, see _bootstrap_t0 above -- how the
    # rounds span divides into camera I/O vs CV detection vs the PnP
    # solve. capture_time_s is shared/cumulative (capture happens once per
    # round for all cameras together); detect/solve are summed across
    # cameras here for one headline number, with the real per-camera
    # breakdown in per_cam_diagnostics below. These are DELIBERATELY not
    # part of the wall-clock accounting above -- cameras run in parallel,
    # so their sums legitimately exceed the span that contains them.
    log.info(
        "bootstrap_calibrations SECTION TIMING (within rounds; sums across "
        "parallel cameras): capture=%.2fs | detect(sum)=%.2fs "
        "%s | solve(sum)=%.2fs %s",
        capture_time_s,
        sum(detect_time_s.values()),
        {cam: round(t, 2) for cam, t in sorted(detect_time_s.items())},
        sum(solve_time_s.values()),
        {cam: round(t, 2) for cam, t in sorted(solve_time_s.items())},
    )

    # FRAME-SELECTION PROVENANCE (the v2 package schema, 2026-08-27) --
    # "no record of WHICH raw-pool frames actually fed the accepted
    # solve" (only a bare count, `n_frames_used`, was ever stored). Investigated before adding anything (per this
    # task's own instruction to investigate, not force a false claim):
    #
    # CAPTURE ORDER is deterministic -- confirmed by reading, not
    # assumed. Each camera's own `pre_orientation_pool[cam]` is built by
    # ONE thread appending in strict round-received order (the
    # `ThreadPoolExecutor` above parallelizes ACROSS cameras, never
    # within one camera's own sequence); there is no `random`/`shuffle`
    # call anywhere in this retry loop that could reorder a camera's own
    # frames. `pre_orientation_pool[cam]`'s own index IS the raw-pool
    # index -- the exact same list `package_raw_frames` below encodes
    # into `cam{N}_raw.mkv`/PNG frames in this same order, so raw-pool
    # index i in a saved package IS frame i of that file.
    #
    # WHICH raw-pool frames actually CONTRIBUTED to the accepted
    # `average_correspondences()` call is a real, computable subset, not
    # simply "the first n_frames_used" -- a raw-pool entry can be a
    # never-detected raw-extra placeholder (`pre is None`, see DECOUPLED
    # CAPTURE-VS-DETECT TARGET above) or a detected-but-rejected frame
    # (`correspond_landmarks_from_pre_orientation()` returns None for an
    # ambiguous/failed correspondence, still consuming a slot in
    # `accumulated_detections[cam]`). Recomputed HERE, read-only, from
    # state this function already maintains (`pre_orientation_pool`,
    # `accumulated_detections`) -- deliberately NOT threaded through the
    # retry loop's own control flow (that loop already has several
    # documented, previously-verifier-caught subtleties around hint
    # changes/whole-pool reprocessing; adding new mutable accumulator
    # state to it for a provenance-only field is real, avoidable risk to
    # the live calibration result this function's whole docstring already
    # says must never be put at risk). Both accumulators share one
    # invariant by construction (every branch in `_detect_batch()` above
    # keeps `accumulated_detections[cam]` index-aligned with
    # `[e for e in pre_orientation_pool[cam] if e[1] is not None]`, in
    # order, whether built incrementally per round or wholesale on a hint
    # change) -- verified with a defensive length check below rather than
    # trusted blindly: a mismatch degrades to `None` (never a
    # fabricated/misleading list) and is logged loudly, since that would
    # mean the invariant broke somewhere and the safe thing is to say so,
    # not guess.
    # RENAMED 2026-08-27 (a v2 package-schema parity check): this local
    # (and its `diagnostics_out` key, below) used to be
    # `frames_used_indices` -- renamed to `frame_indices_used`, a pure
    # naming fix, not a new concept.
    frame_indices_used: dict[int, list[int] | None] = {}
    # The companion second index list, added in the same follow-up
    # round: the raw pool's own COMPLEMENT of `frame_indices_used` --
    # every raw-pool frame that did NOT feed the accepted solve, whether
    # it was never detected at all (a raw-only extra, see DECOUPLED
    # CAPTURE-VS-DETECT TARGET above) or detected-but-rejected. Computed
    # from the same `pool` (and hence the same length-invariant check)
    # `frame_indices_used` already trusts -- degrades to `None` in
    # lockstep with it, never a fabricated/misleading list on its own.
    raw_extra_frame_indices: dict[int, list[int] | None] = {}
    # Read straight off what the adopting code recorded at the moment it
    # adopted -- see `adopted_frame_indices` / `detected_pool_indices_by_det`.
    #
    # This replaces a re-scan of the pool for `pre is not None` plus a
    # length invariant against `accumulated_detections`, which failed on
    # EVERY camera of EVERY calibration once best-of-N started appending
    # to the pool without accumulating (a fixed 20-slot gap at the shipped
    # settings), and so wrote `frame_indices_used: null` into every
    # calibration package. Nothing infers the mapping any more, so there
    # is no invariant left to violate; `None` here now means only "this
    # camera never adopted a calibration".
    for cam in best_calibration:
        pool = pre_orientation_pool.get(cam, [])
        used = adopted_frame_indices.get(cam)
        if used is None:
            frame_indices_used[cam] = None
            raw_extra_frame_indices[cam] = None
            continue
        frame_indices_used[cam] = used
        used_set = set(used)
        raw_extra_frame_indices[cam] = [i for i in range(len(pool)) if i not in used_set]

    # Computed unconditionally (cheap -- no I/O) so both `diagnostics_out`
    # AND the calibration-package derived-values file below share exactly
    # one place this shape is built, never two independently-maintained
    # copies.
    per_cam_diagnostics: dict[int, dict[str, Any]] = {
        cam: {
            "reprojection_error_px": best_reprojection_px.get(cam),
            "n_frames_used": frames_captured[cam],
            # DECOUPLED CAPTURE-VS-DETECT TARGET, 2026-08-21 -- honest
            # visibility into the new split: `n_frames_used` above is
            # still exactly "how many frames were actually DETECTED and
            # fed the PnP solve" (frames_captured[cam], unchanged
            # meaning); `n_frames_raw_pool` is the (usually bigger) total
            # size of `pre_orientation_pool[cam]` -- detected frames PLUS
            # the raw-only extras -- i.e. how many raw frames the two
            # post-loop sections (ring-boundary-offset, board-color)
            # actually had available for THIS camera.
            "n_frames_raw_pool": len(pre_orientation_pool.get(cam, [])),
            # The v2 package schema (2026-08-27) --
            # `n_raw_extra_frames_used`: the raw-only "extra"
            # frames beyond the ones actually detected -- i.e.
            # `n_frames_raw_pool` (the FULL pool, detected + extras)
            # minus `n_frames_used` (detected only). opendarts never split
            # this out as its own number before now (only the combined
            # pool total existed); computed here from the two values
            # immediately above/below, not a new detection pass.
            "n_raw_extra_frames_used": max(
                0, len(pre_orientation_pool.get(cam, [])) - frames_captured[cam]
            ),
            # See this function's own "FRAME-SELECTION PROVENANCE"
            # comment above `per_cam_diagnostics`'s own definition --
            # raw-pool indices (matching this package's own
            # cam{N}_raw.mkv/PNG frame order) that actually contributed
            # to this camera's accepted PnP solve, or None if the
            # invariant this is computed from didn't hold this event
            # (logged loudly at computation time, never silently wrong).
            # Named `frame_indices_used` (renamed from
            # `frames_used_indices` 2026-08-27).
            "frame_indices_used": frame_indices_used.get(cam),
            # The companion field, added in the same rename round --
            # the raw pool's own complement of `frame_indices_used` (the
            # frames that did NOT feed the accepted solve). Same
            # None-on-invariant-failure posture.
            "raw_extra_frame_indices": raw_extra_frame_indices.get(cam),
            "target_met": (
                best_reprojection_px.get(cam) is not None
                and best_reprojection_px[cam] < _target_px_for(cam)
            ),
            "orientation_hint_source": established_hint_source.get(cam),
            "orientation_hint_deg": established_hint_deg.get(cam),
            # LIVE-DERIVED FOCAL LENGTH, 2026-08-26 -- mirrors the
            # orientation-hint fields immediately above exactly (same
            # "which tier actually produced this event's value" question,
            # see `_resolve_focal_length_px()`'s own docstring for the
            # 3-tier priority: `"live"` / `"fallback_json"` /
            # `"fallback_constant"` / `None` if this camera never
            # resolved a focal length at all this pass).
            "focal_length_px": established_focal_px.get(cam),
            "focal_length_source": established_focal_source.get(cam),
            # LIVE-DERIVED RADIAL DISTORTION, 2026-08-26 -- see
            # _resolve_distortion_for()'s own docstring for the 2-tier
            # priority (`"live"` / `"fallback_json"` / `"unavailable"`).
            # k1=0.0 (this project's own prior hardcoded default) for any
            # camera this event's data didn't safely support a fit for --
            # never None, since `dist_coeffs` is always populated with
            # SOME value (0.0 or fitted) for calibrate_camera(), unlike
            # focal length above which can genuinely be unresolved.
            "k1": established_k1.get(cam, 0.0),
            "distortion_source": established_distortion_source.get(cam),
            # The SELF-CONSISTENT joint-solve focal length paired with
            # the k1 above (used for THIS event's actual
            # calibrate_camera() call whenever distortion_source is
            # "live"/"fallback_json") -- deliberately separate from
            # "focal_length_px" above, which always reports the
            # homography-only tier's own value regardless of whether
            # distortion also succeeded (see established_joint_focal_px's
            # own setup comment for why the two must not be conflated).
            # None when distortion never succeeded this event -- in that
            # case "focal_length_px" above IS what was actually used.
            "joint_focal_length_px": established_joint_focal_px.get(cam),
            # LIVE-DERIVED PRINCIPAL POINT (cx only), 2026-08-26 -- mirrors
            # the k1/joint_focal_length_px fields immediately above
            # exactly, one tier further (see `opendarts.calibration.
            # distortion`'s own module docstring, "PRINCIPAL POINT (cx
            # only)" section). `cx_px`/`cx_focal_length_px`/`cx_k1` are
            # None whenever this tier never succeeded this event -- in
            # that case the k1-only tier's own values above (or the
            # centered-principal-point default) are what was actually
            # used for THIS event's calibrate_camera() call.
            "cx_px": established_cx.get(cam),
            "cx_focal_length_px": established_cx_focal_px.get(cam),
            "cx_k1": established_cx_k1.get(cam),
            "principal_point_source": established_principal_point_source.get(cam),
            # TIMING -- calibration_duration_s is
            # THIS camera's own wall-clock time (capture + detect/solve
            # rounds, from the same _bootstrap_t0 every camera starts
            # from); calibration_total_duration_s is the whole
            # bootstrap_calibrations() call's own total, duplicated onto
            # every camera's entry (rather than a separate top-level
            # slot) since per_cam_diagnostics is strictly per-camera --
            # see this function's own TIMING comment above for the
            # measurement itself.
            "calibration_duration_s": camera_duration_s.get(cam),
            "calibration_total_duration_s": total_duration_s,
            # SECTION TIMING. detect_duration_s/solve_duration_s are THIS
            # camera's own cumulative time in _detect_batch()/_try_solve()
            # across every round. The three post-loop sections
            # (capture/motion-thresholds/ring-boundary/board-color) run
            # once for the whole calibration event, not per camera --
            # duplicated onto every camera's entry for the same reason
            # calibration_total_duration_s is, above.
            "detect_duration_s": round(detect_time_s.get(cam, 0.0), 3),
            "solve_duration_s": round(solve_time_s.get(cam, 0.0), 3),
            "capture_duration_s": round(capture_time_s, 3),
            "motion_threshold_duration_s": round(motion_threshold_time_s, 3),
            # 0.0 and null since 2026-09-13: the live measurement was
            # removed (see its own comment block above). Kept as keys
            # rather than deleted so a package reader can distinguish
            # "this calibration did not measure it" from "this package
            # predates the field", and so the offline tools that DO
            # measure it keep a place to write.
            "ring_boundary_offset_duration_s": 0.0,
            "board_color_duration_s": round(board_color_time_s, 3),
            # Full disjoint wall-clock breakdown, including the remainder
            # -- so a package can answer "where did the time go" without
            # the log, and so a phase can never silently go unmeasured
            # again (see the PHASE WALL CLOCK log above).
            "phase_wall_s": dict(phase_wall_s),
            # Seed-ellipse aspect and how far it sat from the group median
            # (gate: ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD). Recorded
            # always, so headroom against that gate is visible without
            # having to trip it.
            "ellipse_aspect": ellipse_aspect_by_cam.get(cam),
            "ellipse_aspect_deviation": ellipse_aspect_deviation_by_cam.get(cam),
            # REPLAY PERSISTENCE (docs/DESIGN.md's "Replay is the source of truth"), 2026-08-22 --
            # the actual measured ring-boundary-offset/board-color VALUES
            # (not just how long they took), package-wide like every
            # other post-loop section above (duplicated onto every
            # camera's entry, same reason `calibration_total_duration_s`
            # is) -- see `save_calibration_package()`'s own pull-up into
            # `derived_calibration.json`'s top level, and
            # `load_throw_package()`'s new lookup that reads them back
            # for REPLAY. `None`/`False` here (the safe default set
            # above the try blocks) means "not accepted this event" --
            # never a fabricated value.
            "treble_inner_offset_mm": treble_inner_offset_mm,
            "treble_inner_confidence": treble_inner_confidence,
            "treble_inner_n_samples_used": treble_inner_n_samples_used,
            "double_inner_offset_mm": double_inner_offset_mm,
            "double_inner_confidence": double_inner_confidence,
            "double_inner_n_samples_used": double_inner_n_samples_used,
            "ring_boundary_offset_accepted": ring_boundary_offset_accepted,
            # FULL-BOUNDARY STORAGE GAP FIX, 2026-08-26 -- see this
            # function's own `ring_boundary_measurement` comment above.
            # All 4 boundaries' full measurement, or None if the
            # measurement itself never ran/raised this event.
            "ring_boundary_measurement": ring_boundary_measurement,
            "brightness_threshold": brightness_threshold,
            "brightness_threshold_confidence": brightness_threshold_confidence,
            "chroma_threshold": chroma_threshold,
            "chroma_threshold_confidence": chroma_threshold_confidence,
            # THE V2 PACKAGE SCHEMA (2026-08-27) -- the full nested
            # payloads (see `ring_boundary_offset_payload`'s own comment
            # above, where it's built), package-wide like every other
            # post-loop section here, duplicated onto every camera's
            # entry for the same reason `calibration_total_duration_s`
            # is. `calibration_package.py` pulls these up into
            # `derived_calibration.json`'s own top-level
            # `ring_boundary_offset`/`board_color_calibration` keys --
            # this per-camera duplication is purely a storage vehicle
            # inside `per_cam_diagnostics`, matching this file's own
            # established pattern, not a claim these are per-camera
            # values.
            "ring_boundary_offset_payload": ring_boundary_offset_payload,
            "board_color_calibration_payload": board_color_calibration_payload,
        }
        for cam in best_calibration
    }
    if diagnostics_out is not None:
        diagnostics_out.update(per_cam_diagnostics)

    # See this function's own "CALIBRATION PACKAGE" docstring section --
    # opt-in (calibration_package_root is None means do nothing here,
    # every existing caller/test unaffected), never on the critical path
    # (the actual save/cleanup runs on a background thread unless a
    # caller explicitly asks for the blocking/test-only path).
    if calibration_package_root is not None and best_calibration:
        package_id = new_calibration_package_id()
        # REPLAY (docs/DESIGN.md's "Replay is the source of truth"), 2026-08-21 -- see this
        # function's own docstring's "CALIBRATION PACKAGE" section and
        # the comment on `pre_orientation_pool[cam].extend(pre_stage)`
        # in `_detect_batch()` above (verifier finding Bug 2). Every
        # frame this camera's raw pool actually holds -- detected AND
        # the raw-only "extra" frames from DECOUPLED CAPTURE-VS-DETECT
        # TARGET, both consistently white-balance-normalized (Bug 1
        # fix) -- goes into the saved package, matching what the live
        # ring-boundary-offset/board-color derivations actually
        # consumed and what `n_frames_raw_pool` in
        # `derived_calibration.json` reports, instead of only the
        # subset that happened to go through `_detect_batch()`.
        package_raw_frames = {
            cam: [pb for pb, _pre in pre_orientation_pool.get(cam, [])]
            for cam in best_calibration
        }
        if calibration_package_out is not None:
            calibration_package_out["package_id"] = package_id
            calibration_package_out["package_dir"] = calibration_package_root / package_id
        if calibration_package_blocking:
            try:
                save_calibration_package(
                    calibration_package_root,
                    package_id,
                    package_raw_frames,
                    best_calibration,
                    per_cam_diagnostics,
                )
                if throw_package_root is not None:
                    from opendarts.capture.calibration_package import (
                        cleanup_orphaned_calibration_packages,
                    )

                    cleanup_orphaned_calibration_packages(
                        calibration_package_root,
                        throw_package_root,
                        active_package_id=package_id,
                    )
            except Exception: # noqa: BLE001 -- never break a live-shaped caller, see docstring
                log.exception(
                    "calibration package %s: blocking save/cleanup failed", package_id
                )
        else:
            # save_calibration_package_background() itself does its real
            # save/cleanup work inside a daemon thread it starts and
            # never joins -- that inner work is already exception-safe
            # (see that function's own docstring/implementation). What is
            # NOT already covered is the SYNCHRONOUS sliver of work this
            # call still does on THIS thread before returning --
            # `threading.Thread(...)`/`.start()` -- which can, in
            # principle, raise (e.g. a real OS thread-creation failure
            # under resource exhaustion). Wrapped here too so that even
            # this narrow, rare failure mode still can't cascade into
            # "no calibration was returned at all" for a live caller --
            # same never-break-the-real-result guarantee as the blocking
            # branch above.
            try:
                save_calibration_package_background(
                    calibration_package_root,
                    package_id,
                    package_raw_frames,
                    best_calibration,
                    per_cam_diagnostics,
                    throw_package_root=throw_package_root,
                )
            except Exception: # noqa: BLE001 -- never break a live-shaped caller, see docstring
                log.exception(
                    "calibration package %s: failed to start background save", package_id
                )

    return best_calibration


def calibration_status_dict(
    calibrations: dict[int, CameraCalibration], n_cameras: int
) -> dict[int, dict[str, Any]]:
    """The lightweight per-camera DISPLAY projection of a real
    calibrations dict -- ok/reprojection error (px)/landmark-spread note
    for every camera 0..n_cameras-1, "not calibrated" filled in for any
    camera bootstrap_calibrations() didn't return. Factored out of
    opendarts.live.server.AppState._refresh_calibration_blocking (2026-08-12,
    alongside the CalibrationStore work) so the SAME status-shaping logic
    is used whether the calibration came from a startup bootstrap or a
    manual dashboard recalibrate -- one place decides what "ok"/the error
    number/the note mean, not two independently-maintained copies."""
    cam_status: dict[int, dict[str, Any]] = {}
    for cam, calib in calibrations.items():
        reproj_err = (
            calib.pnp_result.reprojection_error_px if calib.pnp_result is not None else None
        )
        cam_status[cam] = {
            "ok": True,
            "reprojection_error_px": reproj_err,
            "landmark_spread_ok": calib.landmark_spread_ok,
        }
    for cam in range(n_cameras):
        if cam not in cam_status:
            cam_status[cam] = {
                "ok": False,
                "reprojection_error_px": None,
                "reason": "not calibrated this pass (landmark detection or PnP failed -- see server log)",
            }
    return cam_status


# CalibrationStore persistence (2026-08-26) -- see that class's own
# docstring's "Durable across process restarts" section. Same directory
# as opendarts.calibration.focal_length.FOCAL_LENGTH_FALLBACK_FILENAME
# (`calibration_package_root`, i.e. DEFAULT_CALIBRATION_PACKAGE_ROOT for
# the real live path) -- one more small, self-correcting, JSON-backed
# "last known good" file living alongside that established sibling, not a
# fourth naming convention.
CURRENT_CALIBRATION_FILENAME = "current_calibration.json"
CURRENT_CALIBRATION_SCHEMA = "current-calibration-snapshot-v1"


class CalibrationStore:
    """Thread-safe holder for the calibrations dict actually used to
    score LIVE throws -- the shared, mutable reference requested,
    2026-08-12: "calib once at beginning and use that to score all
    darts.. if someone manually presses recalibrate, then use it and
    store that calib data." Before this existed, run_capture_loop_body()
    captured `calibrations` as a plain local variable at startup and
    never looked at it again -- opendarts/live/server.py's dashboard could
    recompute a fresh calibration and show new numbers, but the capture
    loop kept scoring every subsequent throw against the STALE startup
    calibration forever. This class is the fix: exactly ONE instance per
    running opendarts.live.run_product combined process, built before the
    capture loop's background thread starts and shared with
    opendarts.live.server.AppState (see that module's `calibration_store`
    param) -- the capture loop thread READS it (`.get()`) on every single
    READY_TO_CAPTURE, the dashboard's manual "Refresh calibration now"
    button WRITES to it (`.set()`) from a different thread (FastAPI's own
    request-handling context) -- a plain dict/attribute swap without a
    lock would be a real (if narrow-window) data race, not just
    theoretical, so this holds one.

    `.get()` returns a shallow copy of the current dict (never the live
    object a concurrent `.set()` could be replacing) -- safe for the
    capture loop to read/iterate without holding the lock a moment longer
    than the copy itself takes; the CameraCalibration objects themselves
    are effectively immutable in normal use (nothing in this codebase
    mutates one in place after calibrate_camera() returns it), so a
    shallow copy of the dict is sufficient, not a deep one.

    Standalone `-m opendarts.live.capture_daemon` (no dashboard sharing this
    process) still gets one internally (see run_capture_loop_body()) even
    though nothing else could ever call `.set()` on it there -- keeps the
    capture loop's own read path IDENTICAL in both modes, one code path,
    not a special-cased "if a store was given" branch sprinkled through
    the loop body.

    **Durable across process restarts, 2026-08-26**. Before this, EVERY process
    restart wiped this object's in-memory contents, so `run_capture_loop_
    body()`'s existing "reuse an already-populated CalibrationStore, skip
    bootstrap_calibrations()" branch (see that function's own "Calibration
    bootstrap" docstring section, unchanged by this) only ever fired for a
    SECOND-OR-LATER Start within one process's lifetime, never across a
    restart -- a code deploy, a crash, or a manual `/api/restart` meant
    the very next Start paid a full fresh bootstrap again, contrary to
    the project's own "calibrate once... store that calib data" spec this
    class's docstring already opens with.

    `snapshot_path`, when given (`None` is the default -- every existing
    caller/test is unaffected, purely in-memory as before): at
    CONSTRUCTION, tries to load a previously-persisted snapshot from that
    path FIRST, before falling back to the `calibrations`/`source`/
    `checked_at_utc`/`package_id` constructor args (same "loaded value
    wins over constructor defaults, absent/corrupt falls through safely"
    contract `EngineConfigStore`'s own `snapshot_path` already established
    -- deliberately the SAME mechanism, not a new one). On every
    successful `.set()` afterward, writes the CURRENT calibrations (plus
    the ring-boundary-offset/board-color-threshold values actually live
    right now, via `opendarts.geometry.board.get_ring_boundary_offsets()`/
    `opendarts.geometry.board_color.get_board_color_thresholds()` -- both
    already applied to those modules' own globals by
    `bootstrap_calibrations()` BEFORE `.set()` is ever called, for both
    real call sites: `run_capture_loop_body()`'s own Start-time bootstrap
    and `opendarts.live.server.AppState.refresh_calibration()`'s manual
    "Refresh calibration now") back to disk, best-effort, same "a disk
    hiccup must never break the live action that triggered it" posture
    `EngineConfigStore._save_snapshot()` already has.

    **Storage format, justified against this project's existing
    conventions rather than invented fresh**: a flat JSON file,
    `current_calibration.json`, living next to `focal_length_fallback.json`
    under `calibration_package_root` (same directory, same "small,
    self-correcting, overwritten-on-every-successful-event" shape that
    file's own docstring already established for exactly this class of
    value). It reuses `opendarts.capture.throw_package.calibration_to_dict()`/
    `calibration_from_dict()` for the per-camera `CameraCalibration` shape
    -- the SAME on-disk representation `derived_calibration.json` (inside
    a real, timestamped `opendarts.capture.calibration_package`) already
    uses, so there is still only one serialization format for "a real
    CameraCalibration on disk" in this project, not a second one.
    Deliberately NOT the calibration-package-plus-pointer-file design that
    was also considered: `save_calibration_package()`'s own raw-frame
    encode + cleanup pass runs on a background thread (see that module's
    docstring), so a pointer written the instant a package_id is decided
    can race a process restart landing before that background save
    finishes, pointing at a package whose `derived_calibration.json`
    doesn't exist yet -- recoverable (this class falls through to a fresh
    bootstrap exactly like "no snapshot at all"), but avoidable. Writing
    this class's own flat snapshot SYNCHRONOUSLY inside `.set()` instead
    sidesteps that race entirely, at the cost of not carrying the raw
    calibration-burst frames (this snapshot is for RESCORING, not
    REPLAYING a calibration event -- the fuller, raw-frame-carrying
    package this event also saves, via `bootstrap_calibrations()`'s
    existing `calibration_package_root` wiring, is untouched by this
    change and remains the real REPLAY artifact per docs/DESIGN.md's
    "Replay is the source of truth"; `package_id` is still recorded in this snapshot purely as a
    cross-reference back to that fuller package when it exists).

    **Known, accepted trade-off, not silently papered over**: a camera physically moved to a different USB
    port after this snapshot was written, followed by a plain process
    restart with no manual "Refresh calibration now," will keep scoring
    against the now-physically-wrong persisted calibration indefinitely --
    there is no automatic re-derivation to catch this, by design (see
    `opendarts.calibration.focal_length`'s own module docstring, RESOLVED
    2026-08-26, for the real re-seat incident this exact risk mirrors).
    The mitigation is visibility, not auto-detection: `.meta()`'s
    `checked_at_utc`/`source` (`"persisted"` for a snapshot loaded at
    construction, distinct from `"startup"`/`"manual"`) are surfaced on
    the dashboard's Cameras tab (`opendarts/live/server.py`'s
    `renderCalibration()`/`live-calib-age` element) so an operator who
    just did a hardware change has a real, concrete "this calibration
    predates your change, go hit Refresh" signal instead of the staleness
    being invisible.
    """

    def __init__(
        self,
        calibrations: dict[int, CameraCalibration] | None = None,
        *,
        source: str = "startup",
        checked_at_utc: str | None = None,
        package_id: str | None = None,
        snapshot_path: "Path | None" = None,
    ) -> None:
        self._lock = threading.Lock()
        self._snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        loaded = self._load_snapshot() if self._snapshot_path is not None else None
        if loaded is not None:
            calibrations, source, checked_at_utc, package_id = loaded
        self._calibrations: dict[int, CameraCalibration] = dict(calibrations or {})
        self._source = source
        self._checked_at_utc = checked_at_utc
        # Which opendarts.capture.calibration_package this store's current
        # contents came from, if any (2026-08-20 -- see that module's own
        # docstring). None for every caller that hasn't opted into
        # calibration packages (bootstrap_calibrations()'s own
        # `calibration_package_root=None` default), or for the one
        # construction path (run_capture_loop_body's standalone-run
        # branch, see that function's docstring) that seeds this from a
        # bootstrap result computed BEFORE this store existed to receive
        # one -- honestly absent, never fabricated.
        self._package_id = package_id

    def get(self) -> dict[int, CameraCalibration]:
        with self._lock:
            return dict(self._calibrations)

    def get_package_id(self) -> str | None:
        """Which calibration package (opendarts.capture.calibration_package)
        the CURRENT calibrations came from, if this store's contents were
        ever set with one. **Prefer `get_with_package_id()` over calling
        this alongside a separate `.get()`** -- see that method's own
        docstring for the real TOCTOU race two independently-locked reads
        would create."""
        with self._lock:
            return self._package_id

    def get_with_package_id(self) -> tuple[dict[int, CameraCalibration], str | None]:
        """Atomic combined read of `.get()` + `.get_package_id()` --
        added 2026-08-20 alongside calibration packages, specifically
        because `handle_ready_to_capture()`'s real call site needs BOTH
        values describing the exact SAME calibration event. Two separate
        lock acquisitions (`store.get()` then `store.get_package_id()`)
        have a real, if narrow, window between them where a concurrent
        `.set()` (a manual "Refresh calibration now" click, on a
        different thread, landing between the two calls) could hand back
        calibrations from one event and a package_id from a DIFFERENT,
        newer one -- silently breaking the very traceability calibration
        packages exist to provide (a throw package whose `calibration.json`
        doesn't actually match the calibration package its own
        `calibration_package_id` points to). One lock acquisition here
        makes that structurally impossible: both values always come from
        the same `.set()` call."""
        with self._lock:
            return dict(self._calibrations), self._package_id

    def set(
        self,
        calibrations: dict[int, CameraCalibration],
        *,
        source: str,
        checked_at_utc: str,
        package_id: str | None = None,
    ) -> None:
        with self._lock:
            self._calibrations = dict(calibrations)
            self._source = source
            self._checked_at_utc = checked_at_utc
            self._package_id = package_id
            if self._snapshot_path is not None:
                self._save_snapshot()

    def meta(self) -> dict[str, Any]:
        """Non-secret bookkeeping about the CURRENT live calibration --
        where it came from (source: "startup"/"manual"/"persisted") and
        when, for the dashboard to display honestly (see docs/DESIGN.md's
        "don't oversell" discipline: distinguishing this from the old,
        now-removed 20s auto-poll is the whole point of this feature).
        `"persisted"` (added 2026-08-26) means this calibration was loaded
        from `snapshot_path` at construction, not solved fresh THIS
        process lifetime -- see this class's own docstring's "Durable
        across process restarts" section, including the real, accepted
        hardware-re-seat staleness trade-off `checked_at_utc` exists to
        make visible."""
        with self._lock:
            return {
                "source": self._source,
                "checked_at_utc": self._checked_at_utc,
                "n_cameras": len(self._calibrations),
                "calibration_package_id": self._package_id,
            }

    # -- persistence (snapshot_path), 2026-08-26 -- see class docstring's
    # "Durable across process restarts" section for the full design and
    # its justification against this project's existing conventions
    # (EngineConfigStore's own snapshot_path, focal_length_fallback.json).
    # ------------------------------------------------------------------

    def _load_snapshot(
        self,
    ) -> "tuple[dict[int, CameraCalibration], str, str, str | None] | None":
        """Best-effort read of a previously-persisted calibration. Returns
        `None` (never raises) on a missing file, corrupt JSON, a schema
        mismatch, or a camera entry that fails to parse -- ANY real
        problem falls back to the constructor's own
        `calibrations`/`source`/`checked_at_utc`/`package_id` arguments
        (an empty dict for every real caller, i.e. "no valid persisted
        calibration exists, calibrate fresh") rather than crashing process
        startup, same posture `EngineConfigStore._load_snapshot()` already
        established for this exact kind of small persisted file.

        On a SUCCESSFUL load, also re-applies the ring-boundary-offset and
        board-color-threshold globals this snapshot carries and re-derives
        the per-camera board-disc masks -- skipping `bootstrap_
        calibrations()` entirely (the whole point of loading a persisted
        calibration) means nothing else in this process would ever apply
        them. Local imports match this module's inline-import convention
        for these modules (avoids a top-level circular import)."""
        assert self._snapshot_path is not None
        try:
            raw = json.loads(self._snapshot_path.read_text())
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            log.warning(
                "%s: failed to read/parse -- treating as absent (falls back to "
                "a fresh bootstrap, same as no persisted calibration at all)",
                self._snapshot_path,
            )
            return None
        if raw.get("schema") != CURRENT_CALIBRATION_SCHEMA:
            log.warning(
                "%s: schema %r does not match current %r -- treating as absent",
                self._snapshot_path, raw.get("schema"), CURRENT_CALIBRATION_SCHEMA,
            )
            return None
        cameras_raw = raw.get("cameras")
        if not isinstance(cameras_raw, dict) or not cameras_raw:
            log.warning("%s: no cameras in persisted snapshot -- treating as absent",
                        self._snapshot_path)
            return None
        calibrations: dict[int, CameraCalibration] = {}
        try:
            for cam_str, entry in cameras_raw.items():
                calibrations[int(cam_str)] = calibration_from_dict(entry)
        except Exception: # noqa: BLE001 -- a bad snapshot must not break startup
            log.exception(
                "%s: failed to parse one or more camera entries -- treating as "
                "absent", self._snapshot_path,
            )
            return None
        checked_at_utc = raw.get("checked_at_utc")
        if not isinstance(checked_at_utc, str):
            log.warning("%s: missing checked_at_utc -- treating as absent",
                        self._snapshot_path)
            return None
        package_id = raw.get("package_id")

        from opendarts.geometry.board import set_ring_boundary_offsets
        from opendarts.geometry.board_color import set_board_color_thresholds

        set_ring_boundary_offsets(
            treble_inner_offset_mm=raw.get("treble_inner_offset_mm"),
            double_inner_offset_mm=raw.get("double_inner_offset_mm"),
        )
        set_board_color_thresholds(
            brightness_threshold=raw.get("brightness_threshold"),
            chroma_threshold=raw.get("chroma_threshold"),
        )
        # BOARD-DISC DETECTION-REGION MASK -- a full-resolution boolean
        # array per camera, so it is RE-DERIVED from `calibrations` rather
        # than persisted (zero risk of drifting stale relative to the
        # calibration it's built from). This path runs before Start
        # negotiates a camera hub, so it uses the rig's fixed IMAGE_WIDTH/
        # IMAGE_HEIGHT; the lifecycle resizes a mask to the frame shape it
        # is actually handed, so a differing negotiated resolution degrades
        # to a slightly imprecise disc boundary until the next "Refresh
        # calibration now" (which uses the real negotiated resolution).
        try:
            from opendarts.capture.board_disc import board_disc_mask

            board_disc_masks: dict[int, np.ndarray] = {}
            for cam, calib in calibrations.items():
                mask = board_disc_mask(calib, IMAGE_WIDTH, IMAGE_HEIGHT)
                if mask is not None:
                    board_disc_masks[cam] = mask
            set_calibrated_board_disc_masks(board_disc_masks or None)
        except Exception: # noqa: BLE001 -- a bad snapshot must not break startup
            log.exception(
                "%s: board-disc mask re-derivation from the persisted calibration "
                "failed -- no throw detection until the next successful calibration",
                self._snapshot_path,
            )
            set_calibrated_board_disc_masks(None)

        log.info(
            "startup calibration: loaded persisted calibration from %s "
            "(%d camera(s), checked_at_utc=%s, package_id=%s) -- skipping "
            "bootstrap_calibrations() entirely; use 'Refresh calibration now' "
            "to force a fresh one",
            self._snapshot_path, len(calibrations), checked_at_utc, package_id,
        )
        return calibrations, "persisted", checked_at_utc, package_id

    def _save_snapshot(self) -> None:
        """Write the CURRENT calibrations (already held under `self._lock`
        by the sole caller, `.set()`) to `self._snapshot_path`, best-effort
        -- a disk hiccup here must never break the live "Refresh
        calibration now" click or Start's own bootstrap that triggered it,
        same posture `EngineConfigStore._save_snapshot()` already has.
        Also captures whatever ring-boundary-offset/board-color-threshold
        values are ACTUALLY live right now (`opendarts.geometry.board.
        get_ring_boundary_offsets()`/`opendarts.geometry.board_color.
        get_board_color_thresholds()`) -- correct at this call site
        specifically because `bootstrap_calibrations()` already applied
        them (or explicitly reset them to None/default) for THIS SAME
        calibration event, before either of `.set()`'s two real callers
        (`run_capture_loop_body()`'s Start-time bootstrap,
        `AppState.refresh_calibration()`'s manual refresh) ever invoke
        `.set()` -- see this class's own docstring for why reading them
        back this way, rather than threading the original values through
        a second parameter, is correct and sufficient."""
        assert self._snapshot_path is not None
        try:
            from opendarts.geometry.board import get_ring_boundary_offsets
            from opendarts.geometry.board_color import get_board_color_thresholds

            treble_offset_mm, double_offset_mm = get_ring_boundary_offsets()
            brightness_threshold, chroma_threshold = get_board_color_thresholds()
            payload = {
                "schema": CURRENT_CALIBRATION_SCHEMA,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                "checked_at_utc": self._checked_at_utc,
                "package_id": self._package_id,
                "cameras": {
                    str(cam): calibration_to_dict(calib)
                    for cam, calib in self._calibrations.items()
                },
                "treble_inner_offset_mm": treble_offset_mm,
                "double_inner_offset_mm": double_offset_mm,
                "brightness_threshold": brightness_threshold,
                "chroma_threshold": chroma_threshold,
            }
            self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            self._snapshot_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc: # noqa: BLE001 -- best-effort, see docstring
            log.warning(
                "failed to persist current-calibration snapshot %s: %s",
                self._snapshot_path, exc,
            )


@dataclass(frozen=True)
class EngineConfig:
    """Which engine is PRIMARY (exactly one -- its result IS `result.json`'s
    top-level fields, unchanged from today), which engines ALSO run
    (zero or more -- each lands under `result.json`'s `other_engines`
    key), and the per-engine timeout. Frozen/immutable -- see
    EngineConfigStore below for the actual thread-safe mutable holder;
    this dataclass is just the snapshot `.get()` hands back."""

    primary: str = DEFAULT_PRIMARY_ENGINE
    also_run: tuple[str, ...] = DEFAULT_ALSO_RUN
    timeout_s: float = DEFAULT_ENGINE_TIMEOUT_S


class EngineConfigStore:
    """Thread-safe holder for the live multi-engine scoring config --
    docs/ENGINES.md's "Config" section: "Same live-mutable pattern
    already used for CalibrationStore and CaptureLoopController (read
    fresh next throw, no restart)." Deliberately the SAME established
    request/signal-holder shape those two classes already use (see
    CalibrationStore's own docstring immediately above for the full
    thread-safety reasoning -- a plain dict/attribute swap without a lock
    would be a real, if narrow-window, data race between the dashboard's
    Config-tab POST (FastAPI request-handling thread) and the capture
    loop reading it fresh on every READY_TO_CAPTURE).

    ONE instance per running opendarts.live.run_product combined process,
    built before the capture loop's background thread starts and shared
    with opendarts.live.server.AppState -- mirrors calibration_store's own
    wiring in opendarts/live/run_product.py exactly, not a new pattern.
    Standalone `-m opendarts.live.capture_daemon` (no dashboard sharing this
    process) still gets a default-constructed one internally (see
    handle_ready_to_capture()'s own `engine_config_store=None` default) --
    same "one code path, not a special-cased branch" reasoning
    CalibrationStore's own docstring gives for its own standalone case.

    **Durable, 2026-08-14** (see DEFAULT_ENGINE_CONFIG_STORE_DIR's own
    comment for the real incident this fixes): pass `snapshot_path` to
    both load an existing `latest.json` at construction (falling back
    silently to the `primary`/`also_run`/`timeout_s` args when none
    exists yet -- a fresh install or a wiped data dir is not an error)
    and write one on every successful `.set()`, best-effort (a disk
    hiccup must never break the live Config-tab save the way
    AppState.refresh_calibration()'s own snapshot write doesn't either).
    `None` (the default) keeps this purely in-memory, matching every
    existing test's own construction and every caller that doesn't pass
    it -- opt-in, not a behavior change for anything not explicitly
    wired to a path.
    """

    def __init__(
        self,
        primary: str = DEFAULT_PRIMARY_ENGINE,
        also_run: tuple[str, ...] = DEFAULT_ALSO_RUN,
        timeout_s: float = DEFAULT_ENGINE_TIMEOUT_S,
        *,
        snapshot_path: "Path | None" = None,
    ) -> None:
        self._lock = threading.Lock()
        self._snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        loaded = self._load_snapshot() if self._snapshot_path is not None else None
        if loaded is not None:
            self._config = loaded
        else:
            self._config = EngineConfig(
                primary=primary, also_run=tuple(also_run), timeout_s=float(timeout_s)
            )

    def _load_snapshot(self) -> "EngineConfig | None":
        """Best-effort read of a previously-saved config. Any real
        problem (missing file, corrupt JSON, an engine name that no
        longer exists because the registry changed) falls back to the
        constructor's own defaults rather than crashing process
        startup -- a stale/bad snapshot must never prevent the whole
        live system from coming up."""
        assert self._snapshot_path is not None
        from opendarts.live.config import read_config_section

        try:
            raw = read_config_section("engine_config", self._snapshot_path)
            if raw is None:
                return None
            primary = raw["primary"]
            also_run = tuple(raw.get("also_run") or ())
            timeout_s = float(raw.get("timeout_s", DEFAULT_ENGINE_TIMEOUT_S))
            if not is_registered(primary) or any(not is_registered(n) for n in also_run):
                log.warning(
                    "engine config snapshot %s names an unregistered engine, ignoring "
                    "(primary=%r also_run=%r)",
                    self._snapshot_path,
                    primary,
                    also_run,
                )
                return None
            return EngineConfig(primary=primary, also_run=also_run, timeout_s=timeout_s)
        except Exception as exc: # noqa: BLE001 -- a bad snapshot must not break startup
            log.warning("failed to load engine config from %s: %s", self._snapshot_path, exc)
            return None

    def get(self) -> EngineConfig:
        with self._lock:
            # EngineConfig is frozen -- safe to hand the same instance out
            # to multiple readers, no copy needed (unlike
            # CalibrationStore.get()'s dict, which IS mutable and does
            # need a fresh shallow copy per caller).
            return self._config

    def meta(self) -> dict[str, Any]:
        """Non-secret bookkeeping for the dashboard's Config tab -- same
        honest-snapshot convention as CalibrationStore.meta()/
        ResetRequest.meta()."""
        with self._lock:
            cfg = self._config
            return {
                "primary": cfg.primary,
                "also_run": list(cfg.also_run),
                "timeout_s": cfg.timeout_s,
                "available_engines": engine_names(),
            }


class ResetRequest:
    """Thread-safe signal for a manual "reset now" request -- the
    dashboard's Reset button (POST /api/reset, opendarts/live/server.py) ->
    this object -> the capture loop's background thread, added
    2026-08-12 by deliberate design decision ("you should wire the 'reset'
    button in controls... take the current image as clean bg (whether
    there was a dart in it or not) and then get ready for a dart throw").

    Mirrors CalibrationStore's own thread-safe holder pattern immediately
    above -- ONE instance per running opendarts.live.run_product combined
    process, built before the capture loop's background thread starts and
    shared with opendarts.live.server.AppState, same "a plain flag swap
    without a lock would be a real, if narrow-window, data race" reasoning
    that class's own docstring already makes. Deliberately the SAME
    established mechanism, not a new one -- see server.py's Reset-wiring
    docstring for why.

    UNLIKE CalibrationStore (which holds actual calibration DATA the loop
    reads once per READY_TO_CAPTURE), this is a bare pending-flag: the
    loop polls `.check_and_clear()` once per iteration, at the TOP of the
    loop body (not gated on any particular trigger state -- a reset must
    be actionable regardless of whatever state the trigger currently sits
    in, unlike CalibrationStore's read which only matters at
    READY_TO_CAPTURE). `.check_and_clear()` atomically reads-and-clears
    the flag under the same lock, so a single click is actioned exactly
    once, not repeatedly on every subsequent iteration until another
    click arrives.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requested = False
        self._requested_at_utc: str | None = None

    def request(self) -> None:
        """Called from the FastAPI request-handling thread (POST
        /api/reset) -- sets the pending flag for the capture loop thread
        to pick up on its next iteration."""
        with self._lock:
            self._requested = True
            self._requested_at_utc = datetime.now(timezone.utc).isoformat()

    def check_and_clear(self) -> bool:
        """Called from the capture loop thread, once per iteration.
        Returns True at most once per `.request()` call -- clears the
        flag under the same lock it reads it with, so a concurrent
        `.request()` racing a `.check_and_clear()` can only ever resolve
        to "this iteration sees it" or "the next one does," never lost
        and never double-actioned."""
        with self._lock:
            was_requested = self._requested
            self._requested = False
            return was_requested

    def meta(self) -> dict[str, Any]:
        """Non-secret bookkeeping for the dashboard -- when the most
        recent reset request was made (None if never), same honest-null
        convention as CalibrationStore.meta()."""
        with self._lock:
            return {"last_requested_at_utc": self._requested_at_utc}


class CaptureLoopController:
    """Thread-safe Start/Stop/idle-timeout controller for the capture
    loop's camera-hub lifecycle -- added 2026-08-12: the program does not
    open cameras on launch; Start opens them, Stop closes them, and a
    configurable idle timeout closes them too. An explicit start/stop pair plus a
    touch/idle-loop timeout (see the two ARCHITECTURE NOTE paragraphs
    below for what's adapted vs. mirrored).

    ARCHITECTURE NOTE 1 (threads, not asyncio tasks): the obvious
    alternative is an asyncio-task-based detector (`self._task =
    asyncio.create_task(...)`, torn down via
    `asyncio.wait_for(...)`/`.cancel()`). opendarts's capture loop
    is a single persistent background `threading.Thread` for this
    process's whole lifetime (opendarts/live/run_product.py's own
    architecture, UNCHANGED by this feature -- the thread itself is not
    created/destroyed per Start/Stop) -- there is no asyncio task to
    create/cancel. This class adapts the same LIFECYCLE SHAPE (a request
    signal the loop-owning thread observes and acts on) via the SAME
    established request/signal pattern CalibrationStore/ResetRequest
    above already use, not a literal port of asyncio cancellation, which
    doesn't exist in this architecture. Concretely: the capture thread's
    own outer loop (opendarts/live/run_product.py's `_capture_thread_target`)
    blocks on `start_requested` (a threading.Event) until `/api/start`
    sets it, then runs exactly ONE "session" of
    `run_capture_loop_body(..., also_stop=controller.session_stop_event)`
    -- a session ends either when `session_stop_event` is set (manual
    Stop or idle-timeout) or the real process `stop_event` is set (whole-
    process shutdown) -- and loops back to waiting for the next Start.

    ARCHITECTURE NOTE 2 (who opens/closes the camera hub): the REQUEST
    HANDLER thread (FastAPI, via `asyncio.to_thread`) does the actual
    `hub.open_all()`/`hub.close_all()` I/O synchronously. This
    class itself never touches the hub directly -- it's a pure
    coordination object. `/api/stop`'s handler waits (bounded, via
    `stopped_ack`) for the capture thread to actually finish its current
    session BEFORE closing the hub -- a bounded-wait-then-
    proceed-anyway shape (since a
    thread cannot be forcibly cancelled, the handler just proceeds and
    closes the hub anyway after logging a loud warning -- an accepted,
    documented, narrow-window deviation, not a silent one).

    Fields, all thread-safe under one lock:
      - `running`: True once a session is active (cameras open, actively
        fetching/scoring). Read by the dashboard (state_dict()'s
        `capture_loop` section) and by `/api/start`/`/api/stop` to skip a
        redundant open/close.
      - `idle_timeout_sec`: configurable (default IDLE_TIMEOUT_SEC_DEFAULT
        above). `<= 0` disables it.
      - `last_activity_monotonic`: updated by `touch()` -- called on every
        real control action (Start/Stop/Reset/Calibrate, see
        opendarts/live/server.py's route handlers) AND on every real capture-
        loop event.

    `session_stop_event`/`stopped_ack`/`start_requested` are plain
    `threading.Event`s, not lock-guarded -- Event objects are already
    internally thread-safe; only the plain-value fields above need the
    lock.

    **Durable, 2026-09-03** (see `DEFAULT_IDLE_TIMEOUT_STORE_DIR`'s own
    comment for the real incident this fixes -- the identical class of
    bug `EngineConfigStore`'s own 2026-08-14 fix already closed for
    engine config, never applied here): pass `snapshot_path` to both
    load a previously-saved value at construction (falling back
    silently to the `idle_timeout_sec` constructor arg when none exists
    yet -- a fresh install or a wiped data dir is not an error) and
    write one on every successful `set_idle_timeout_sec()` call, best-
    effort (a disk hiccup must never break the live dashboard save).
    `None` (the default) keeps this purely in-memory, matching every
    existing test's own construction and every caller that doesn't pass
    it -- opt-in, not a behavior change for anything not explicitly
    wired to a path.
    """

    def __init__(
        self, *, idle_timeout_sec: int = IDLE_TIMEOUT_SEC_DEFAULT,
        snapshot_path: "Path | None" = None,
    ) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        loaded = self._load_snapshot() if self._snapshot_path is not None else None
        self._idle_timeout_sec = (
            loaded if loaded is not None else max(0, int(idle_timeout_sec))
        )
        self._last_activity_monotonic = time.monotonic()
        # The capture thread's own outer loop blocks here (bounded
        # .wait(timeout=...) polls, same shape as _wait_for_stable_frames'
        # own bounded-wait-then-recheck-stop_event pattern above) until
        # request_start() sets it.
        self.start_requested = threading.Event()
        # What run_capture_loop_body()'s new `also_stop=` parameter
        # watches -- set by request_stop() (manual Stop or idle-timeout),
        # cleared by request_start() right before a fresh session begins.
        self.session_stop_event = threading.Event()
        # Set by the capture thread itself the moment a session actually
        # ends (run_capture_loop_body() returned) -- /api/stop's handler
        # waits (bounded) on this before closing the hub.
        self.stopped_ack = threading.Event()

    def touch(self) -> None:
        with self._lock:
            self._last_activity_monotonic = time.monotonic()

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def request_start(self) -> None:
        """Called from the FastAPI request-handling thread (POST
        /api/start), AFTER it has already opened the camera hub -- signals
        the capture thread's outer loop to begin a session."""
        with self._lock:
            self._running = True
            self._last_activity_monotonic = time.monotonic()
        self.stopped_ack.clear()
        self.session_stop_event.clear()
        self.start_requested.set()

    def request_stop(self) -> None:
        """Called from the FastAPI request-handling thread (POST
        /api/stop) or the idle-timeout background task -- signals the
        CURRENT session to end. Does NOT itself close the hub (the
        caller does that, after waiting on `stopped_ack` -- see this
        class's own ARCHITECTURE NOTE 2)."""
        with self._lock:
            self._running = False
        self.session_stop_event.set()

    def mark_session_ended(self) -> None:
        """Called by the capture thread itself, exactly once, right after
        run_capture_loop_body() returns for any reason (manual Stop,
        idle-timeout, or the whole process stopping)."""
        with self._lock:
            self._running = False
        self.start_requested.clear()
        self.stopped_ack.set()

    def _load_snapshot(self) -> "int | None":
        """Best-effort read of the persisted idle-timeout. Lives as the
        flat `idle_timeout_sec` key in the shared live config file, not
        in a directory of its own -- one file an operator edits. Any
        real problem (missing file, corrupt JSON, wrong type) falls back
        to the constructor default rather than crashing startup."""
        assert self._snapshot_path is not None
        from opendarts.live.config import read_config_section

        value = read_config_section("idle_timeout_sec", self._snapshot_path)
        if value is None:
            return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            log.warning(
                "%s: 'idle_timeout_sec' is not a real int (%r) -- ignoring",
                self._snapshot_path, value,
            )
            return None

    def _save_snapshot(self, value: int) -> None:
        if self._snapshot_path is None:
            return
        from opendarts.live.config import write_config_section

        try:
            write_config_section("idle_timeout_sec", value, self._snapshot_path)
        except Exception as exc: # noqa: BLE001 -- best-effort, same as engine config's own save
            log.warning(
                "failed to persist idle-timeout %s: %s", self._snapshot_path, exc
            )

    def get_idle_timeout_sec(self) -> int:
        with self._lock:
            return self._idle_timeout_sec

    def set_idle_timeout_sec(self, value: int) -> None:
        with self._lock:
            self._idle_timeout_sec = max(0, int(value))
            # Inside the lock -- matches EngineConfigStore.set()'s own
            # convention exactly (save call sits inside its own `with
            # self._lock:` block too).
            self._save_snapshot(self._idle_timeout_sec)

    def idle_timeout_due(self) -> bool:
        """True when a session is running, an idle timeout is configured
        (>0), and enough real wall-clock time has passed since the last
        real activity -- polled by the idle-timeout background task (see
        opendarts/live/server.py's AppState._idle_timeout_loop), same 5s
        polling cadence."""
        with self._lock:
            if self._idle_timeout_sec <= 0 or not self._running:
                return False
            return (time.monotonic() - self._last_activity_monotonic) >= self._idle_timeout_sec

    def meta(self) -> dict[str, Any]:
        """Non-secret bookkeeping for the dashboard -- real, honest
        capture-loop lifecycle status, distinct from per-camera
        CameraStatus (opendarts.live.local_capture) which only ever
        describes a hub that HAS been opened at least once."""
        with self._lock:
            return {
                "running": self._running,
                "idle_timeout_sec": self._idle_timeout_sec,
                "seconds_since_activity": round(time.monotonic() - self._last_activity_monotonic, 1),
            }


# save_calibration_snapshot() removed 2026-08-22 -- confirmed genuinely
# dead code before removal, not just unused-looking: grepped every real
# call site in the codebase and found nothing ever reads `latest.json`
# or a timestamped snapshot record back (write-only since it was added
# 2026-08-12). Its own payload (camera pose only, no raw frames) is a
# strict subset of what `opendarts.capture.calibration_package`'s REPLAY
# packages already capture -- and calibration_package_root is hardcoded
# (not optional) at both real dashboard call sites in server.py, so
# every real calibration event already gets the fuller, actually-
# replayable record unconditionally. Historical records that used to
# live in DEFAULT_CALIBRATION_STORE_DIR (data/calibrations/) were
# removed by the project's own hand, per the standing "never rm real data
# unilaterally" rule -- see docs/DESIGN.md.


def fetch_current_frames(
    dest_dir: Path,
    *,
    hub: local_capture.LocalCameraHub | None = None,
) -> dict[int, np.ndarray]:
    """Real, working frame fetch -- NOT a stub. Used both for the
    trigger's rolling "current frame" input and, at startup, to seed the
    initial background.

    Frame source: grabs directly from an
    already-open `opendarts.live.local_capture.LocalCameraHub` (`hub`,
    required in that mode -- see bootstrap_calibrations' docstring for
    why this function never opens/closes the hub itself).

    Local direct camera access is the only frame source. An HTTP
    round-trip path existed once and was removed; there is no opt-in.

    PERFORMANCE FIX, 2026-08-12 (the real ~165ms/iteration finding -- see
    this module's own docstring's dated section for the full writeup):
    this branch used to go
    through `local_capture.fetch_all_snapshots()`, which -- for every
    camera, every single iteration -- takes an already in-memory frame
    from `hub.grab()`'s cache, `cv2.imwrite()`s it to a throwaway PNG in
    `dest_dir`, and hands back a path; this function then immediately
    `cv2.imread()`s that same PNG straight back into memory. Measured on
    this dev machine (real
    1280x720 frames pulled from an actual archived rig throw package,
    200 reps): that round trip alone cost ~91ms for 3 cameras
    (imwrite ~20ms/frame + imread ~10ms/frame, x3) -- against a real
    measured ~165ms/iteration total on the rig, i.e. this ONE avoidable
    round trip was roughly HALF the entire per-iteration cost, for data
    that was already sitting in memory (`hub.grab_all()` returns exactly
    the `dict[int, np.ndarray]` this function needs to return anyway) and
    was never read back from `dest_dir` by anything else (checked: no
    other caller reads `scratch_dir / "current"` -- it was pure,
    unnecessary disk churn, not a debug artifact anyone depended on).
    FIX: the local branch now calls `hub.grab_all()` directly and returns
    it unchanged -- zero PNG encode/decode, zero disk I/O, in the hot
    path. `dest_dir` is accepted but genuinely unused in
    that branch now (kept in the signature for API/call-site
    compatibility, and because `_wait_for_stable_frames()` /
    `run_capture_loop_body()`'s existing call sites already pass one
    positionally/by-keyword).
    """


    if hub is None:
        raise ValueError(
            "fetch_current_frames() requires an already-open "
            "hub= LocalCameraHub -- see docstring. There is no alternative "
            "frame source; the caller must open the hub."
        )
    # DETECTION DECODES SMALL (detect_from_small_decode, see
    # opendarts/capture/lazy_frame.py): the hub's frames come back as a
    # LazyFrames -- the same mapping shape, decoding a frame only when its
    # pixels are read -- so the lifecycle judges each tick from the small
    # grey pictures the pump already made and only a commit decodes.
    if getattr(hub, "small_decode", False):
        return hub.grab_frames()
    # Pure in-memory cache read (see PERFORMANCE FIX docstring section
    # above) -- hub.grab_all() already returns dict[int, np.ndarray],
    # exactly this function's return shape; no PNG write/read round trip.
    return hub.grab_all()


# Startup only needs a first frame set to size the loop's buffers; the
# lifecycle's own WARMUP phase (opendarts.lifecycle.state) refuses to adopt a
# reference until every camera has held still for warmup_stable_frames
# consecutive ticks, so exposure/white-balance convergence after the
# cameras open is handled there, per tick, against the real frames -- not
# by a separate startup settle check with its own thresholds.
STARTUP_FIRST_FRAMES_MAX_WAIT_S = 10.0


def _wait_for_first_frames(
    fetch_fn: Callable[[], dict[int, np.ndarray]],
    *,
    stop_event: threading.Event,
    poll_interval_s: float,
    label: str,
    max_wait_s: float = STARTUP_FIRST_FRAMES_MAX_WAIT_S,
) -> dict[int, np.ndarray]:
    """Fetch until at least one camera delivers a frame, or ``max_wait_s``
    elapses (then return whatever the last fetch gave, possibly empty, and
    say so). Never blocks startup forever on a camera that never produces."""
    deadline = time.monotonic() + max_wait_s
    attempts = 0
    frames: dict[int, np.ndarray] = {}
    while True:
        attempts += 1
        frames = fetch_fn()
        if frames:
            log.info("%s: %d camera(s) delivering after %d fetch(es)", label, len(frames), attempts)
            return frames
        if stop_event.is_set() or time.monotonic() >= deadline:
            log.warning(
                "%s: no camera delivered a frame within %.1fs (%d fetches) -- "
                "proceeding; the lifecycle will warm up when frames arrive",
                label, max_wait_s, attempts,
            )
            return frames
        if poll_interval_s > 0:
            stop_event.wait(poll_interval_s)


#: How long to wait for AD to report the dart we just scored, before
#: recording that it never did. We commit ~100-270ms ahead of AD (measured
#: 2026-09-21: AD's answer for the dart in question landed 218ms after our
#: capture, and 59ms after we looked), so a single look is almost always
#: too early for the last dart of a visit. Generous enough to cover that
#: lag several times over, bounded so a genuinely missed dart is recorded
#: as missed promptly. Costs nothing on the capture path -- the wait
#: happens on the background thread that already writes this file.
AD_ORDINAL_GRACE_SEC = 1.5
AD_ORDINAL_POLL_SEC = 0.05


def _origin_host() -> "str | None":
    """This rig's hostname, for meta.json's `host`. Cached: it cannot
    change within a process, and every throw asks."""
    global _ORIGIN_HOST
    if _ORIGIN_HOST is None:
        try:
            _ORIGIN_HOST = build_info().get("hostname") or socket.gethostname()
        except Exception:  # noqa: BLE001 -- origin metadata must never fail a save
            _ORIGIN_HOST = None
    return _ORIGIN_HOST


def _origin_build() -> "str | None":
    """The code version that scored the throw, for meta.json's `build` --
    the same `code_version` /api/health reports. Cached for the same
    reason, and equally non-fatal: a package missing its origin is worth
    far less than no package at all."""
    global _ORIGIN_BUILD
    if _ORIGIN_BUILD is None:
        try:
            _ORIGIN_BUILD = build_info().get("code_version")
        except Exception:  # noqa: BLE001
            _ORIGIN_BUILD = None
    return _ORIGIN_BUILD


_ORIGIN_HOST: "str | None" = None
_ORIGIN_BUILD: "str | None" = None


def _attach_ad_ground_truth_from_ws(
    package_dir: Path,
    ad_ws_listener: "AdWsListener | None",
    window_sec: float,
    on_event: Any = None,
    session_id: "str | None" = None,
    verdict: "_OracleVerdict | None" = None,
) -> bool:
    """Matches a just-saved package against `ad_ws_listener`'s own in-
    memory buffer of recent AD throws (`AdWsListener.match()`) and writes
    `ad_ground_truth.json` -- see this module's own "AD GROUND TRUTH"
    docstring section above for the full story (the REST detection list
    is not durable across visits; a persistent
    WS listener with a buffer closes that race entirely -- see
    `opendarts.live.ad_ws_listener`'s own module docstring for the design).

    `ad_ws_listener=None` (the `--no-ad-ground-truth` opt-out, or a
    caller that never built one) makes this a no-op, not an error.

    SAFE BY CONSTRUCTION, not merely wrapped in try/except:
    `AdWsListener.match()` performs NO network I/O at all (see that
    method's own docstring) -- it only scans an already-buffered,
    already-received list under a lock, so there is no network stall to
    protect against here the way an inline REST fetch would have needed.
    Still run on a background daemon thread AND wrapped in try/except
    regardless, on this task's own "err toward never block the capture
    loop noticeably" instruction -- a future change to
    match()/save_ad_ground_truth() (e.g. slow disk I/O under load, or the
    matching logic growing a network call it doesn't have today) must
    never be able to reintroduce a stall here just because this
    function's own contract quietly assumed it never would. Mirrors
    _emit()'s own "a bad sink must never take down the capture loop"
    discipline immediately above it in this file.

    Returns whether an attach was started. `verdict`, when given, receives
    the oracle's answer (the AdGroundTruth, or None if the attach failed)
    the moment it is known -- it is what a "mismatch" video-record mode
    waits on to decide this package's clip (see _write_package_clip()).
    It is never set when no attach was started; the caller knows that
    from the return value.
    """
    if ad_ws_listener is None:
        return False
    # AD switched off live: write NOTHING rather than a record saying we
    # looked and found nothing. `matched: false` with a reason like
    # "ws_no_buffered_events" is a statement about a failed lookup, and a
    # package carrying it is indistinguishable from one where AD was
    # running and genuinely missed the throw -- which is the case that
    # actually matters. Absence is the honest encoding of "not asked".
    if getattr(ad_ws_listener, "oracle_base_url", None) is not None:
        if ad_ws_listener.oracle_base_url() is None:
            return False

    def _run() -> None:
        gt = None
        try:
            captured_at = None
            expect_ordinal = None
            meta_path = package_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    captured_at = meta.get("captured_at_utc")
                    # AD's numThrows and our visit_index both reset on
                    # takeout, so n == visit_index + 1 is the same dart on
                    # both sides. Without it match() can only guess by time.
                    vi = meta.get("visit_index")
                    if isinstance(vi, int) and vi >= 0:
                        expect_ordinal = vi + 1
                except (OSError, json.JSONDecodeError):
                    captured_at = None
            # WAIT FOR AD, BRIEFLY. We commit a throw ~100-270ms before AD
            # reports the same dart, so asking once virtually guarantees
            # that the LAST dart of a visit is still missing from AD's
            # buffer -- and the old nearest-in-time fallback then recorded
            # the PREVIOUS dart's answer as this dart's ground truth,
            # inventing disagreements that never happened. Poll instead,
            # for a bounded moment, and accept "AD never said" over "AD
            # said something about a different dart".
            #
            # Free to do here: this whole function already runs on a
            # background daemon thread precisely so it cannot hold up
            # scoring, and match() itself stays non-blocking.
            gt = ad_ws_listener.match(
                captured_at, window_sec=window_sec, expect_ordinal=expect_ordinal)
            if expect_ordinal is not None and not gt.matched:
                deadline = time.monotonic() + AD_ORDINAL_GRACE_SEC
                while not gt.matched and time.monotonic() < deadline:
                    time.sleep(AD_ORDINAL_POLL_SEC)
                    gt = ad_ws_listener.match(
                        captured_at, window_sec=window_sec,
                        expect_ordinal=expect_ordinal)
                if not gt.matched:
                    # The expected dart number never arrived. Either AD
                    # missed this dart (then nothing near our capture will
                    # be found either, and it stays honestly unmatched), or
                    # AD missed an EARLIER one and its numbering is now off
                    # by one for the rest of the visit -- in which case this
                    # dart's answer is sitting right there under a different
                    # n. One last look, on arrival time instead of number.
                    gt = ad_ws_listener.match(
                        captured_at, window_sec=window_sec,
                        expect_ordinal=expect_ordinal, allow_time_fallback=True)
            save_ad_ground_truth(package_dir, gt)
            log.info(
                "AD ground truth (ws) for %s: matched=%s reason=%s sector=%s ring=%s",
                package_dir, gt.matched, gt.match_reason, gt.sector, gt.ring,
            )
            # Tell the dashboard the package CHANGED -- exactly the fix
            # write_other_engines_result() needed on 2026-08-13, for exactly
            # the same reason. This file is written on a background thread,
            # AFTER run_capture_loop_body() already emitted its own
            # PACKAGE_SAVED, and the server only re-reads a package when it
            # sees an event for it (the poll loop reads a package's JSON only
            # when its PATH is new). So without this emit the cached record
            # keeps ad_matched=None until some unrelated later event happens
            # to re-read the file -- the dashboard shows a throw with no AD
            # answer while ad_ground_truth.json on disk plainly has one.
            #
            # Latent all along; the ordinal grace wait above made it the
            # common case by moving this write up to 1.5s later, and it
            # showed first on the slower rig. Same event type and shape the
            # dashboard already handles, so no frontend change.
            _emit(
                on_event,
                {"type": "PACKAGE_SAVED", "path": str(package_dir), "session": session_id},
            )
            # THE ORACLE'S ANSWER, handed to whoever decides this package's
            # clip (the "mismatch" video-record mode, _write_package_clip()).
            # This is the only point in the live path where our answer and
            # the oracle's are both in hand. Deliberately AFTER
            # save_ad_ground_truth(): a recording made on disagreement must
            # never refer to a package that does not yet record why.
        except Exception: # noqa: BLE001 -- enrichment must NEVER affect capture reliability
            log.exception(
                "AD ground-truth (ws) attach failed for %s -- the throw package itself "
                "is unaffected (already saved); the dashboard's manual REST refresh "
                "button can backfill this later if needed",
                package_dir,
            )
        finally:
            if verdict is not None:
                verdict.set(gt)

    threading.Thread(
        target=_run, name=f"opendarts-ad-ground-truth-{package_dir.name}", daemon=True
    ).start()
    return True


class _OracleVerdict:
    """The oracle's answer for one package, handed from the ground-truth
    attach thread to the clip writer (a one-shot slot)."""

    def __init__(self) -> None:
        self._done = threading.Event()
        self._gt: Any = None

    def set(self, gt: Any) -> None:
        self._gt = gt
        self._done.set()

    def wait(self, timeout_s: float) -> Any:
        """The AdGroundTruth, or None if the attach failed or did not
        answer within `timeout_s`."""
        return self._gt if self._done.wait(timeout_s) else None


#: How long a "mismatch" package waits for the oracle before settling for
#: the stills clip. The attach itself polls for up to AD_ORDINAL_GRACE_SEC;
#: the rest is margin. The package's data is already on disk -- only its
#: frames wait (the ring holds far longer than this).
ORACLE_VERDICT_WAIT_S = AD_ORDINAL_GRACE_SEC + 2.0


def _oracle_disagreement(package_dir: Path, gt: Any) -> "str | None":
    """Why this throw should be RECORDED -- the oracle called it
    differently than we did -- or None.

    WHY AUTOMATIC. A misscore that nobody is standing next to is the
    common case on a test rig, and the ring is emptying while nobody
    notices -- by the time a human reads the dashboard the frames are
    gone. The oracle already answered, and comparing two strings costs
    nothing.

    SILENT ON EVERY OTHER OUTCOME, on purpose. No oracle, no match, a
    package we could not score, or an agreement are all ordinary states,
    not events -- recording on "AD did not answer" would fill the disk
    with clips of throws nothing was ever wrong with.

    Never raises: a failure here only means the stills clip."""
    if gt is None or not getattr(gt, "matched", False):
        return None
    try:
        result_path = package_dir / "result.json"
        if not result_path.exists():
            return None
        result = json.loads(result_path.read_text())
        if not result.get("ok"):
            # A throw we could not score is not a MISSCORE -- it is a
            # different failure, with its own evidence already in the
            # package, and treating the two the same would bury the ones
            # where we confidently said the wrong thing.
            return None
        ours = (result.get("sector"), result.get("ring"))
        theirs = (gt.sector, gt.ring)
        if ours == theirs:
            return None
        log.info(
            "oracle disagreement on %s: we called %s/%s, the oracle called %s/%s "
            "-- recording the throw's video clip automatically",
            package_dir.name, ours[0], ours[1], theirs[0], theirs[1],
        )
        return (
            f"oracle disagreement: opendarts called {ours[0]}/{ours[1]}, "
            f"the oracle called {theirs[0]}/{theirs[1]}"
        )
    except Exception: # noqa: BLE001 -- a failed check is just no recording
        log.exception("oracle disagreement check failed for %s", package_dir)
        return None


def _write_package_clip(
    package_dir: Path,
    scored: "clip.ScoredFrames",
    cameras: list[int],
    throw_capture: Any,
    *,
    verdict: "_OracleVerdict | None",
    anchor_wall_s: "float | None",
    on_event: Any = None,
    session_id: "str | None" = None,
) -> None:
    """Write a just-saved package's ONE clip, point its meta.json at it,
    and -- in "mismatch" mode only -- tell the dashboard (PACKAGE_SAVED;
    see the comment at the emit) -- the last step of the package save, run
    once the recording decision is known.

    The decision is the video-record mode's (opendarts.live.config
    video_record_mode, carried on `throw_capture`): "all" records every
    throw; "mismatch" waits for the oracle (`verdict`, filled by the
    ground-truth attach; None when no attach was started) and records
    only a disagreement; "never", no throw-capture service, or no answer
    in time gets the two-frame stills clip. A wanted recording that cannot
    be had also ends in the stills clip -- see
    ThrowCaptureService.write_package_clip.

    Never raises: this runs after THROW_DETECTED and after the package's
    data is on disk, so a failure here is logged loudly and costs this one
    package its frames, nothing else."""
    try:
        mode = getattr(throw_capture, "record_mode", None) if throw_capture is not None else None
        record, reason = False, ""
        if mode == "all":
            record, reason = True, "record all darts"
        elif mode == "mismatch" and verdict is not None:
            disagreement = _oracle_disagreement(package_dir, verdict.wait(ORACLE_VERDICT_WAIT_S))
            if disagreement is not None:
                record, reason = True, disagreement
        if throw_capture is not None:
            throw_capture.write_package_clip(
                package_dir, scored, cameras,
                record=record, anchor_wall_s=anchor_wall_s, reason=reason,
            )
        else:
            video, _ = clip.write_throw_clip(package_dir, scored, cameras)
            clip.point_meta_at_clip(package_dir, video)
        # Announce the clip only when the package list could not have known
        # the outcome in advance (2026-09-26). Every PACKAGE_SAVED makes the
        # server re-read the package and broadcast PACKAGES_UPDATED, and the
        # classic dashboard rebuilds its whole Engines table on each one.
        #   * "mismatch": only now does the list learn that a disputed dart
        #     got its recording -- announced as before.
        #   * "all": every dart is recorded, and no screen gates on the
        #     row's has_video in this mode (the "Save frames" button is
        #     gated on the mode itself, the stills are fetched and retried
        #     by the screens). The server's cached row still has to say
        #     has_video for /api/packages, so it is refreshed quietly:
        #     `refresh_only` re-reads the package and broadcasts nothing.
        #   * "never" / no service: the stills clip; nothing the list shows
        #     changes, so nothing to say.
        if mode == "mismatch":
            _emit(
                on_event,
                {"type": "PACKAGE_SAVED", "path": str(package_dir), "session": session_id},
            )
        elif mode == "all":
            _emit(
                on_event,
                {"type": "PACKAGE_SAVED", "path": str(package_dir), "session": session_id,
                 "refresh_only": True},
            )
    except Exception: # noqa: BLE001 -- see docstring
        log.exception(
            "PACKAGE CLIP WRITE FAILED for %s -- its data files are saved, but it has "
            "NO readable frames. This is a real REPLAY gap for this one throw.",
            package_dir,
        )


def _reuse_zeus_sub_results_for_also_run(
    primary_name: str,
    primary_engine_diagnostics: dict | None,
    also_run_names: tuple[str, ...],
) -> dict[str, EngineResult]:
    """2026-08-27 perf task -- the "duplicate compute" half of the fix.

    **The problem this closes**: when Zeus ("Zeus") is the configured
    live PRIMARY engine, `opendarts.engines.zeus.engine.ZeusEngine.score()`
    already runs all 4 of `ZEUS_SUB_ENGINE_NAMES` (Apollo/Talos/
    Athena/Ares) against the SAME `bg_images`/`frame_images`/
    `calibration` triple this function's own caller has. Before this
    fix, `_dispatch_also_run_engines_in_background()` unconditionally
    re-ran `dispatch_engines()` over whatever `engine_config.also_run`
    names -- which, in the live-configured setup
    (`also_run = ("Apollo", "Talos", "Athena", "Ares")`), is
    EXACTLY the same 4 engines, on the SAME throw, a second time --
    confirmed via direct code read, no dedup existed anywhere before
    this task.

    **The fix**: Zeus's own `EngineResult.diagnostics["sub_results"]`
    already carries each sub-engine's full, already-computed
    `EngineResult.to_dict()` output (see `ZeusEngine.score()`'s own
    `base_diagnostics` construction) -- this function reads it back out
    (via `opendarts.engines.base.engine_result_from_dict()`, a lossless
    round trip) and returns a `dict[str, EngineResult]` for whichever
    `also_run_names` are covered, so the caller can skip dispatching
    them a second time. Scoped NARROWLY and safely:

    - Returns `{}` (nothing to reuse) unless `primary_name == "Zeus"` --
      every OTHER primary-engine configuration (Apollo primary,
      a non-voting primary, etc.) is completely untouched, exactly as
      this task's own scope requires.
    - Only reuses a name that is BOTH in `also_run_names` AND in
      `ZEUS_SUB_ENGINE_NAMES` AND actually present in
      `primary_engine_diagnostics["sub_results"]` -- an `also_run` name
      outside Zeus's own 4 sub-engines
      which `EngineConfig`'s own validation permits since it only checks
      `is_registered()`, not membership in `ZEUS_SUB_ENGINE_NAMES`) is
      correctly left OUT of the returned dict, so the caller still
      dispatches it live, unaffected by this fix -- checked directly:
      that set has zero overlap with
      `ZEUS_SUB_ENGINE_NAMES`, so this is a real, reachable case, not a
      hypothetical one.
    - Same-inputs guarantee, verified not assumed: Zeus's own sub-engine
      calls (`opendarts.engines.zeus.engine._score_all_sub_engines()`) run
      against the exact same `bg_images`/`frame_images`/`calibration`
      expressions this function's caller (`handle_ready_to_capture()`)
      builds for the also-run dispatch -- both sites compute
      `bg_images = {cam: frame for cam, frame in bg_frames.items() if
      cam in current_frames}` and pass the same `calibrations`/
      `dict(current_frames)`, and neither the primary call nor the
      also-run dispatch ever changes the calibration for any of the 4
      Zeus sub-engines -- and
      `prior_dart_line_px` is the SAME `find_prior_dart_line_px(...)`
      lookup, called with the same `(session_dir, visit_id,
      visit_index)` arguments, at both call sites within this one
      `handle_ready_to_capture()` invocation, so it resolves to the same
      value both times.
    - Honest `duration_s`/`timed_out`: see
      `opendarts.engines.zeus.engine._score_sub_engine()`'s own updated
      docstring -- these are now stamped by Zeus's own sub-engine caller
      (real per-sub-engine wall-clock time, `timed_out` always `False`,
      since Zeus has no per-sub-engine timeout of its own), so a reused
      result's `duration_s` is real timing data (from inside Zeus's own
      parallel dispatch), not a silently-wrong `0.0` default -- more
      honest than a fabricated "as if freshly dispatched" number would
      be, and structurally impossible to confuse with a value
      `dispatch_engines()` produced, since it's read straight off
      Zeus's own already-serialized sub-result.
    """
    if primary_name != "Zeus" or not primary_engine_diagnostics:
        return {}
    sub_results = primary_engine_diagnostics.get("sub_results")
    if not sub_results:
        return {}
    reused: dict[str, EngineResult] = {}
    for name in also_run_names:
        if name not in ZEUS_SUB_ENGINE_NAMES:
            continue
        serialized = sub_results.get(name)
        if serialized is None:
            continue
        reused[name] = engine_result_from_dict(serialized)
    return reused


def _dispatch_also_run_engines_in_background(
    dest_dir: Path,
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    calibrations: dict[int, CameraCalibration],
    primary_name: str,
    also_run_names: tuple[str, ...],
    timeout_s: float,
    *,
    session_id: str | None = None,
    on_event: "Callable[[dict[str, Any]], None] | None" = None,
    prior_dart_line_px: "dict[int, tuple[tuple[float, float], tuple[float, float]]] | None" = None,
    reused_sub_results: "dict[str, EngineResult] | None" = None,
) -> None:
    """Runs every "also-run" engine CONCURRENTLY (opendarts.engines.dispatch.
    dispatch_engines(), each own thread, each held to `timeout_s`) on a
    SEPARATE background thread of its own -- see docs/ENGINES.md's
    "Execution model": "the trigger/settle state machine... never waits
    on scoring... regardless of how many engines are enabled or how slow
    one is." This function is fire-and-forget from
    handle_ready_to_capture()'s point of view (mirrors
    `_attach_ad_ground_truth_from_ws()`'s own established pattern
    immediately above it in this file, same "enrichment must never affect
    capture reliability" discipline, same try/except-and-log-only
    failure handling) -- the capture loop has ALREADY moved on to its next
    iteration by the time this thread even starts running, let alone
    finishes.

    Writes `other_engines`/`primary_engine` into the throw's ALREADY-
    SAVED `result.json` via `opendarts.capture.throw_package.
    write_other_engines_result()` once every also-run engine has finished
    or timed out -- ONE additional write, not incremental per-engine (see
    that function's own docstring for why this doesn't reintroduce the
    read-modify-write races docs/ENGINES.md explicitly designed against).

    **Real live bug, fixed 2026-08-13**: this write used to have no
    matching live-push notification of its own -- the ONLY `PACKAGE_SAVED`
    emit happened synchronously in `run_capture_loop_body()` right after
    `handle_ready_to_capture()` returns, which is BEFORE this background
    thread has even started, let alone finished writing `other_engines`.
    The dashboard's WebSocket handler only re-renders on a `PACKAGES_UPDATED`
    push, so a throw's also-run engine rows stayed invisible until
    SOMETHING ELSE happened to trigger a fresh `discover_packages()` re-
    read that incidentally picked up this by-then-complete file too -- in
    practice, the NEXT thrown dart's own `PACKAGE_SAVED` event (which
    rebroadcasts the most recent 20 packages, not just the new one).
    Confirmed live: a real throw's `primary_engine`/`other_engines` were
    genuinely absent from a freshly-saved `result.json` at the moment of
    capture, and `engineSectionsFor()` (server.py's own dashboard JS)
    skips the primary-engine row entirely when `primary_engine` is falsy
    -- so the symptom was "only an AD row shows, and everything about that
    throw stays stuck until the next dart is thrown," exactly matching
    the live report this fix responds to. Fix: emit a SECOND
    `PACKAGE_SAVED` (same event shape and same event type the dashboard
    already handles -- no new frontend code needed) right after
    `write_other_engines_result()` succeeds, so the dashboard picks up the
    now-complete package immediately instead of waiting on an unrelated
    future event. `session_id`/`on_event` are optional (default `None`,
    matching every other optional param this function's caller chain
    already uses) so every existing caller -- tests included -- keeps
    working unchanged; only a caller that actually wants the live push
    needs to pass them.

    `prior_dart_line_px` (2026-08-24 fix, see opendarts.engines.zeus.engine's
    module docstring for the incident): passed straight through to
    `dispatch_engines()`, forwarded only to whichever also-run engine(s)
    actually declare the parameter (Apollo directly, or Zeus wrapping
    it). Optional, defaults to None -- zero behavior change for a caller
    that doesn't pass it.

    `reused_sub_results` (2026-08-27 perf task -- see
    `_reuse_zeus_sub_results_for_also_run()`'s own docstring for the full
    reasoning): an OPTIONAL, already-computed `dict[str, EngineResult]`
    for any `also_run_names` entries the caller already has an answer
    for (today: sub-engines Zeus-as-primary already scored a moment
    earlier, for the SAME throw, against the SAME inputs). Names present
    here are NOT re-dispatched through `dispatch_engines()` -- `results`
    is seeded with these first, then only the REMAINING `also_run_names`
    (if any) are dispatched live and merged in. Default `None` (empty) --
    zero behavior change for every caller that doesn't pass it, which is
    every existing caller/test plus every primary-engine configuration
    other than Zeus.
    """

    def _run() -> None:
        try:
            reused = reused_sub_results or {}
            names_to_dispatch = [n for n in also_run_names if n not in reused]
            priors = find_prior_board_xy_mm_for_package(dest_dir)
            dispatched = (
                dispatch_engines(
                    bg_images, frame_images, calibrations, names_to_dispatch,
                    timeout_s=timeout_s,
                    prior_dart_line_px=prior_dart_line_px,
                    prior_board_xy_mm=priors or None,
                )
                if names_to_dispatch
                else {}
            )
            # Preserve also_run_names' own order (not e.g. reused-then-
            # dispatched) -- purely cosmetic (dict/JSON key order isn't
            # semantically read by anything downstream), kept for a
            # legible log line/diff, matching what a caller would see if
            # every name had gone through dispatch_engines() together.
            results = {
                n: (reused[n] if n in reused else dispatched[n])
                for n in also_run_names
            }
            write_other_engines_result(dest_dir, primary_name, results)
            log.info(
                "also-run engines for %s: %s",
                dest_dir,
                {name: (r.ok, r.timed_out) for name, r in results.items()},
            )
            # The actual fix (see this function's own docstring): tell the
            # dashboard the package changed, same event type/shape it
            # already handles, right after the write that made
            # primary_engine/other_engines real -- don't wait for an
            # unrelated future event to incidentally re-read this file.
            _emit(
                on_event,
                {"type": "PACKAGE_SAVED", "path": str(dest_dir), "session": session_id},
            )
        except Exception: # noqa: BLE001 -- enrichment must NEVER affect capture reliability
            log.exception(
                "also-run engine dispatch failed for %s -- the primary result "
                "(already saved) is unaffected",
                dest_dir,
            )

    threading.Thread(
        target=_run, name=f"opendarts-engines-{dest_dir.name}", daemon=True
    ).start()


def _build_capture_diagnostics(
    trigger: ThrowTriggerState,
    ad_ws_listener: "AdWsListener | None",
    *,
    captured_at_monotonic: float | None = None,
    handle_total_s: float | None = None,
) -> dict[str, Any]:
    """Diagnostic snapshot for ONE just-captured throw, persisted by the
    caller as ``capture_diagnostics.json`` (see opendarts.capture.
    throw_package). Reads straight off ``trigger`` -- the settle timeline
    the lifecycle adapter stamped at the commit tick -- plus
    ``ad_ws_listener.diagnostics_snapshot()`` (None-safe).

    ``settle_duration_s`` is the adapter's own value (pending-dart first
    seen -> commit); ``captured_at_monotonic`` is only a fallback for a trigger built
    without it. The ``ambiguous_settle_*`` and ``board_disc`` keys are
    kept for schema stability -- they belonged to the legacy trigger's
    own classifier and are always ``False``/``None`` now.

    **``timings`` (schema_version 2, 2026-09-08)** -- the per-throw phase
    budget, so a package describes its own latency without needing the
    run log correlated back to it. ``handle_total_s`` is
    handle_ready_to_capture()'s whole synchronous path, passed in by the
    caller (measured there, not here); ``None`` when not supplied --
    absent-or-honest, never a fabricated 0.0.

    Deliberately the ONLY duration here. The primary engine's own
    ``score()`` time is omitted as near-duplicate: the sync path is
    almost entirely that call, and combiner overhead stays recoverable
    as ``handle_total_s`` minus the slowest ``result.json``
    ``other_engines[*].duration_s``. The package-save duration measures work happening after the answer is
    already out the door.

    ``handle_total_s`` also makes the true capture instant recoverable:
    ``meta.captured_at_utc`` is stamped at the END of the sync path, so
    ``captured_at_utc - handle_total_s`` is when the frames were
    actually in hand. That subtraction is why the stamp itself is left
    where it is -- the AD matcher keys off the same value."""
    settle_started = trigger.settle_started_monotonic
    camera_settled_at = trigger.camera_settled_at_monotonic or {}
    settle_duration_s = trigger.settle_duration_s
    per_camera_settle_offset_s: dict[str, float] = {}
    straggler_camera = None
    if settle_started is not None:
        if settle_duration_s is None and captured_at_monotonic is not None:
            settle_duration_s = round(captured_at_monotonic - settle_started, 3)
        per_camera_settle_offset_s = {
            str(cam): round(t - settle_started, 3) for cam, t in camera_settled_at.items()
        }
    if camera_settled_at:
        straggler_camera = max(camera_settled_at, key=camera_settled_at.get)
    return {
        "schema_version": 2,
        "timings": {"handle_total_s": handle_total_s},
        "settle": {
            "settle_duration_s": settle_duration_s,
            "straggler_camera": straggler_camera,
            "per_camera_settle_offset_s": per_camera_settle_offset_s,
            "ambiguous_settle_fired": False,
            "ambiguous_settle_outcome": None,
        },
        "board_disc": {"changed_px_inside": None, "changed_px_outside": None},
        "ad_ws_buffer_at_capture": (
            ad_ws_listener.diagnostics_snapshot() if ad_ws_listener is not None else None
        ),
    }


def _reset_session_throw_numbering(counters_dir: Path, session_id: str) -> None:
    """Bump `session_id`'s counting GENERATION and clear its throw-number
    counter -- the one shared mechanism behind every numbering reset:
    manual Reset, Delete-packages, and handle_ready_to_capture()'s own
    auto-detected "packages got pulled and there's nothing left locally"
    case. Added 2026-08-22 (discussing the package-naming
    convention: "when I hit reset, or packages get pulled and there's no
    packages, we reset to 0 ... add a unique incrementer [to the session
    id]" -- explicitly REVERSING the 2026-08-17 fix's own choice to keep
    numbering continuous across a pull; see handle_ready_to_capture()'s
    own throw_id-construction comment for that history and why it's now
    superseded, not deleted).

    GENERATION, not a bare reset-to-0-in-place, because throw_number
    restarting at 1 under the exact same session_id can collide two
    different ways: (a) Reset doesn't delete any already-saved throw
    packages, so a plain reset-to-0 would try to overwrite session_id-
    001-... which is still sitting right there on disk; (b)
    Delete-packages/an external pull DO clear local disk, but this
    process has no way to know whether some of this session's earlier
    throws were already pulled to the dev machine's archive before the
    reset -- a second session_id-001-... minted later would collide with
    that archived one, invisibly, with zero cross-machine signal
    available to detect it in advance. Bumping generation instead makes
    every reset's throw_id numbering space PROVABLY disjoint from every
    earlier generation's (see handle_ready_to_capture()'s `-g{N}-` infix),
    with no cross-machine knowledge required -- the project's own proposed fix
    for exactly this collision risk.

    Deliberately does NOT touch `session_id` itself or the throw
    package directory on disk -- session stays one stable folder/concept
    for a whole physical sitting, this only affects what NEW throw_ids look like
    from this point forward."""
    counters_dir.mkdir(parents=True, exist_ok=True)
    generation_file = counters_dir / f"{session_id}.generation"
    current_generation = (
        int(generation_file.read_text().strip()) if generation_file.exists() else 0
    )
    generation_file.write_text(str(current_generation + 1))
    (counters_dir / f"{session_id}.count").unlink(missing_ok=True)


def _agreement_string_from_engine_result(engine_result: EngineResult) -> str | None:
    """The v2 package schema (2026-08-27, "3 keys the spec
    missed"): thin wrapper
    around `opendarts.capture.throw_package.agreement_string_from_diagnostics()`
    -- see that function's own docstring for the full writeup,
    including why this is a real, not-fabricated distinction for an
    operator-configurable primary engine. Kept as a thin named wrapper
    here (rather than every call site reaching into `engine_result.
    diagnostics` directly) purely for readability at this function's own
    call site below.
    """
    return agreement_string_from_diagnostics(engine_result.diagnostics)


def _own_tip_line_px_from_engine_result(engine_result: EngineResult) -> "PriorDartLinePx | None":
    """2026-09-01, latency task: pulls Apollo's own `own_tip_line_px` diagnostic (see that
    engine's `score()` docstring section for where it's built) out of
    THIS throw's `engine_result`, regardless of whether Apollo was the
    primary engine directly, or is wrapped by Zeus (the real production
    configuration -- see `opendarts.engines.zeus.engine`'s own module
    docstring: Zeus's `score()` serializes each sub-engine's `EngineResult`
    via `.to_dict()` into `diagnostics["sub_results"]`, so Apollo's own
    diagnostics live nested one level down there instead of at the top).

    Checks the direct (Apollo-as-primary) shape FIRST, then the nested
    (Zeus-as-primary) shape -- mirrors `_reuse_zeus_sub_results_for_
    also_run()`'s own established `sub_results["Apollo"]["diagnostics"]`
    traversal, not a new convention. Returns `None` (never a fabricated
    empty dict) for every other primary engine (Talos, Athena,
    or Zeus without Apollo in its own sub_results) --
    matching this codebase's own absent-not-fabricated discipline, and
    correctly causing the caller to fall back to the frame-cache tier
    (item 8) rather than claim a precomputed tip line that doesn't
    actually exist for this throw."""
    direct = engine_result.diagnostics.get("own_tip_line_px")
    if direct is not None:
        return direct
    sub_results = engine_result.diagnostics.get("sub_results")
    if isinstance(sub_results, dict):
        apollo_sub = sub_results.get("Apollo")
        if isinstance(apollo_sub, dict):
            nested = apollo_sub.get("diagnostics", {}).get("own_tip_line_px")
            if nested is not None:
                return nested
    return None


def handle_ready_to_capture(
    trigger: ThrowTriggerState,
    bg_frames: dict[int, np.ndarray],
    calibrations: dict[int, CameraCalibration],
    package_root: Path,
    session_id: str,
    *,
    ad_ws_listener: "AdWsListener | None" = None,
    ad_match_window_sec: float = DEFAULT_MATCH_WINDOW_SEC,
    engine_config_store: "EngineConfigStore | None" = None,
    on_event: "Callable[[dict[str, Any]], None] | None" = None,
    visit_id: str | None = None,
    visit_index: int | None = None,
    calibration_package_id: str | None = None,
    cached_prior_frames: "CachedPriorThrowFrames | None" = None,
    own_tip_line_px_out: "dict[str, PriorDartLinePx] | None" = None,
    background_save: bool = True,
    store_packages: bool = True,
    # The free-space floor, in GB, below which a throw package is
    # SKIPPED rather than written -- None means the code default (see
    # opendarts.disk_space.DEFAULT_MIN_FREE_DISK_GB, 5 GB), 0 means the
    # same, and a NEGATIVE value disables the guard. Unlike most optional
    # parameters here, None is NOT "no guard": a rig whose disk is full
    # stops being able to write its own log, so the safe direction is the
    # guard being on for every caller that has not thought about it.
    # `opendarts.live.run_product` passes the configured value
    # (`min_free_disk_gb` in data/config.json).
    min_free_disk_gb: "float | None" = None,
    # The throw-capture service (opendarts.capture.throw_capture), or None
    # -- every pre-existing caller and every test. Its video-record mode
    # decides whether the package's one clip is a recording out of its
    # frame ring; None means the two-frame stills clip. See
    # _write_package_clip().
    throw_capture: Any = None,
) -> Path:
    """Real detection + scoring + package save for one completed throw --
    NOT a stub. Each piece this calls (the primary engine's `score()` --
    `opendarts.engines.apollo.ApolloEngine` by default, see below --
    and `opendarts.capture.throw_package.save_throw_package()`) is
    independently real and tested; what's new here is wiring them
    together from a trigger-fired context. Called exactly once per throw,
    when trigger.state == ThrowState.READY_TO_CAPTURE (see run_capture_loop).

    Per docs/DESIGN.md's "Replay is the source of truth", this ALWAYS writes a complete
    replay package for whatever it scored -- including a failed/low-
    confidence score (save_throw_package itself refuses to write a
    PARTIAL package, but a complete package recording an ok=False result
    is exactly what replay-based drift detection needs to see, not
    something to skip).

    ad_ws_listener: optional already-started `AdWsListener` (see module
    docstring's "AD GROUND TRUTH" section) -- when given, kicks off a
    background AD ground-truth attach immediately after the package save
    succeeds (see `_attach_ad_ground_truth_from_ws()` above). `None`
    (the default, and what `--no-ad-ground-truth` resolves to) skips this
    entirely -- the throw package save itself is completely unaffected
    either way.

    A second external oracle's live-scored answer, when wanted, comes
    from running a non-voting registry engine -- this function no longer
    captures a separate ground-truth answer of its own (the older
    "second oracle" mechanism this used to feed was removed once that
    engine existed).

    engine_config_store: added alongside the multi-engine scoring
    framework (docs/ENGINES.md) -- optional `EngineConfigStore` giving the
    live-configured primary/also-run engines + per-engine timeout. `None`
    (the default -- this module's own standalone CLI, and every existing
    caller/test predating this framework) means "no config wired in this
    process," which resolves to `EngineConfig()`'s own defaults:
    `primary="Apollo"`, `also_run=()`.

    cached_prior_frames: added 2026-09-01 (see `opendarts.engines.apollo.
    prior_dart_context.CachedPriorThrowFrames`'s own docstring for the
    full "why call anything off disk at all" writeup) -- the live
    capture loop's own in-memory record of the immediately-prior throw's
    bg/dart frames, threaded straight through to BOTH real
    `find_prior_dart_line_px()` call sites below (the primary engine's
    own lookup, and the also-run dispatch's separate lookup) so neither
    has to re-fetch from disk what the process already has in memory.
    `None` (the default -- every existing caller/test, and the very
    first dart of any visit, which has no prior throw at all) falls
    through to the exact same disk-based lookup this function has always
    used; `find_prior_dart_line_px()` itself is what actually validates
    the cache matches the throw being asked about before trusting it.

    own_tip_line_px_out: added same day (a THIRD, faster tier stacked
    on top of cached_prior_frames above). Optional
    caller-provided mutable dict, same "out-param" idiom `bootstrap_
    calibrations()`'s own `diagnostics_out` already established in this
    module -- NOT a return-type change, so every existing caller/test
    expecting a bare `Path` back is unaffected. When given, this function
    writes `own_tip_line_px_out["own_tip_line_px"] = <value>` right after
    computing THIS throw's own `engine_result` (see `_own_tip_line_px_
    from_engine_result()`'s own docstring for exactly what it extracts
    and from where) -- only when a real value was found; an engine
    combination with nothing to offer here (Talos/Athena
    primary, or Zeus without Apollo among its sub-engines) leaves the
    dict untouched rather than writing a fabricated key. The caller
    (`run_capture_loop_body()`) reads this back immediately after the
    call returns and folds it into the NEXT throw's own `cached_prior_
    frames.precomputed_tip_line` -- entirely in-memory, no disk write or
    read involved at any point in this handoff.

    background_save: added 2026-09-01 ("background the
    throw-package save" -- see this function's own inner
    `_save_and_followups()` docstring section for the full design and
    the honest tradeoff). `True` (the default -- every real live caller)
    is the new behavior: `THROW_DETECTED` fires immediately after
    scoring, then the actual package write (plus capture_diagnostics.json/
    PACKAGE_SAVED/AD-ground-truth-attach/
    also-run dispatch, all as one atomic deferred unit, same relative
    order as before) happens on a daemon thread. `False` runs that exact
    same unit INLINE instead -- byte-identical to this function's own
    pre-2026-09-01 fully-synchronous behavior, return value included
    (this function still always returns `Path`, background or not; the
    only thing `background_save` changes is whether the caller can
    assume the file already exists the moment this call returns). Exists
    for exactly two real callers: this project's own pre-existing test
    suite (which asserts on-disk state immediately after the call --
    correct to keep testing the actual save's own content and shape this
    way, not something worth rewriting into a polling loop just because
    the timing changed) and any future deterministic/offline caller that
    genuinely wants synchronous save semantics for its own reasons.

    **2026-08-12 -- no more primary-engine special-casing.** Every
    primary engine, including the default `Apollo`, is computed the
    SAME way: `get_engine(primary_name).score(...)`. Before this date,
    `primary="Apollo"` had its own inlined `detect_tip()` +
    `reject_outside_roi()` + `score_dart()` call sequence here, bypassing
    `opendarts.engines.apollo.ApolloEngine` entirely -- a deliberate
    caution kept only until that wrapping was proven byte-identical
    (full real corpus, zero mismatches). That proof is why the bypass is
    gone: this
    function now converts the resulting `EngineResult` back into the
    `ScoreResult` shape `save_throw_package()`'s on-disk schema expects,
    via one of two adapters depending on which engine ran --
    `opendarts.engines.apollo.engine_result_to_score_result()` (the
    full-fidelity Apollo-specific inverse, used for `primary="Apollo"`
    -- recovers every field the pre-bypass-removal direct call used to
    populate, including `cameras_used`/`outlier_camera`/
    `alt_candidates_used`) or `opendarts.engines.base.
    engine_result_to_score_result()` (the generic, honestly lossy
    adapter, used for any OTHER primary -- see that adapter's own
    docstring for exactly what's lossy about it; no engine other than
    Apollo has Apollo-shaped diagnostics to recover in the first
    place).

    Also-run engines (`engine_config.also_run`, minus `primary` itself if
    duplicated) are dispatched CONCURRENTLY, each with the configured
    timeout, on a SEPARATE background thread this function does NOT wait
    on -- see `_dispatch_also_run_engines_in_background()` immediately
    above. This is the real, load-bearing "trigger/settle state machine
    must never wait on scoring" guarantee (docs/ENGINES.md's Execution
    model): this function returns `dest_dir` as soon as the PRIMARY
    result is saved, exactly as it always has, regardless of how many
    also-run engines are configured or how slow any of them are.

    visit_id/visit_index (2026-08-14): which turn this dart belongs to
    and which dart of that turn it is (0-based). Passed straight into
    `save_throw_package()`'s meta.json and onto the `THROW_DETECTED`
    event below. Both default None -- every pre-existing caller/test, and
    any path that doesn't track visits, keeps working with the visit
    fields honestly absent rather than fabricated.

    calibration_package_id (2026-08-20): which
    `opendarts.capture.calibration_package` this throw's `calibrations` came
    from, if any -- passed straight into `save_throw_package()`'s
    meta.json (see that module's own docstring for why this reference
    matters: it's what lets the rig-side cleanup and the pull-time
    filtering both tell a still-referenced calibration package apart from
    an orphaned dialing-in attempt). `None` (the default -- every
    pre-existing caller/test, and any live calibration that predates
    calibration packages or opted out via `bootstrap_calibrations()`'s
    own `calibration_package_root=None`) means this throw's meta.json
    simply omits the field, same absent-not-fabricated convention as
    visit_id/visit_index. The real caller
    (`run_capture_loop_body()`) reads this off
    `calibration_store.get_package_id()` fresh on every READY_TO_CAPTURE
    -- same "read fresh, not the stale startup-local value" discipline
    this function's own `calibrations` argument already follows.

    `camera_mode` records which frame-source path produced this package.
    Only one exists now, so it is always "real" -- the field survives
    because packages written before the alternative was removed still
    carry the other value, and a reader must not assume every package on
    disk says "real".

    Emits `THROW_DETECTED` on `on_event` (see docs/LIVE_API.md) the
    moment the PRIMARY result is durably on disk -- before the AD attach
    and before any also-run engine is even dispatched, so a consumer
    learns the score in ONE push instead of watching TRIGGER_STATE reach
    READY_TO_CAPTURE, then catching PACKAGES_UPDATED, then fetching
    /api/packages. Purely additive: `PACKAGE_SAVED` still fires from
    `run_capture_loop_body()` immediately after this function returns,
    unchanged, and so does everything downstream of it.

    (2026-08-27, the v2 package schema, first real v2-session
    QA pass) Two further fixes threaded into the existing `save_throw_
    package()` call below, both real values this function already
    computes and previously let go unused for this purpose: the
    session-sequential `generation` local (already embedded in `throw_id`
    /`dest_dir`'s own `-g{N}-` infix, see that local's own dated comment
    further down) is now ALSO passed through as a real int field, not
    left to a directory-name-parsing consumer; and the primary engine's
    raw `EngineResult.diagnostics` (captured just below, same "before the
    ScoreResult conversion drops it" pattern `agreement`/`camera_mode`
    already use) now backs `result.json`'s top-level rollup
    (`cameras_used`/`triangulation`/`max_ray_disagreement_mm`/
    `n_cameras_used`) with the WINNING sub-engine's own real diagnostics
    whenever the primary is a vote-based consensus engine (Zeus/"Zeus")
    and the narrowed `ScoreResult` itself has nothing of its own to
    report there.
    """
    current_frames = trigger.last_frame
    if current_frames is None:
        raise RuntimeError(
            "READY_TO_CAPTURE with no captured frame set on the trigger -- "
            "should not happen; indicates a bug in opendarts.lifecycle.adapter "
            "(it must set last_frame on every commit tick), not something "
            "this function should paper over."
        )

    # PER-THROW TIMING.
    # Same coarse, log-only pattern as bootstrap_calibrations()'s section
    # timing above -- no control-flow changes, just time.monotonic()
    # around each real boundary. Honest scope: this only covers THIS
    # function's own wall-clock, from the moment trigger.last_frame is
    # confirmed present to right before THROW_DETECTED is emitted -- it
    # does NOT cover whatever runs before handle_ready_to_capture() is
    # even called (frame grab, the lifecycle's observe() in
    # run_capture_loop_body()/_lifecycle_step()) --
    # that's a separate, unmeasured piece of the peer's own "before
    # handle_ready_to_capture()" question, not something this change
    # answers.
    _t_handle_start = time.monotonic()

    engine_config = (
        engine_config_store.get() if engine_config_store is not None else EngineConfig()
    )
    primary_name = engine_config.primary
    also_run = tuple(name for name in engine_config.also_run if name != primary_name)

    # 2026-08-12 -- uniform path for every primary engine, Apollo
    # included; see this function's own docstring above for what changed
    # and why removing the old special-casing is now safe. Every primary
    # engine scores with this rig's own `calibrations`.
    _t_engine = time.monotonic() # PER-THROW TIMING, see _t_handle_start above
    primary_engine = get_engine(primary_name)
    bg_images = {cam: frame for cam, frame in bg_frames.items() if cam in current_frames}
    if engine_accepts_prior_board_xy_mm(primary_engine):
        # 2026-08-17 -- Talos prior-dart erase when Talos is primary.
        # Same find_prior_board_xy_mm() as also-run dispatch and replay.
        # Capability check (not the name "Talos") keeps a stub engine
        # under that name on the 3-arg score() signature working.
        prior_board_xy_mm = find_prior_board_xy_mm(
            package_root / session_id, visit_id, visit_index
        )
        engine_result = primary_engine.score(
            bg_images, dict(current_frames), calibrations,
            prior_board_xy_mm=prior_board_xy_mm or None,
        )
    elif engine_accepts_prior_dart_line_px(primary_engine):
        # 2026-08-16 -- prior-dart-in-visit contamination guard (see
        # opendarts.engines.apollo.tip_detection's dated module docstring
        # entry and opendarts.engines.apollo.prior_dart_context for the
        # full real-incident write-up). Gated PURELY on the
        # `engine_accepts_prior_dart_line_px()` capability check, not a
        # `primary_name == "Apollo"` name check (2026-08-24 fix --
        # removed the name check after it caused a real live miss, see
        # opendarts.engines.zeus.engine's module docstring for the full
        # incident write-up: with Zeus configured as primary, Apollo
        # got ZERO contamination protection anywhere in the live
        # pipeline, even though Zeus's own score() now declares this
        # same parameter and forwards it to its Apollo sub-call --
        # this branch is what makes that forwarding actually happen for
        # the primary-engine call). Still correctly skips every OTHER
        # primary engine that doesn't declare the parameter (Talos is
        # caught by the `elif` immediately above; Athena
        # fall through to the plain `else` below) -- this is a
        # capability check, not a hardcoded engine list, so any FUTURE
        # engine that legitimately declares `prior_dart_line_px` (its
        # own, or by wrapping Apollo the way Zeus does) gets this for
        # free too. Best-effort: any lookup failure already returns None
        # inside find_prior_dart_line_px() itself (never raises), so
        # this can never block a real throw from scoring.
        prior_dart_line_px = find_prior_dart_line_px(
            package_root / session_id, visit_id, visit_index,
            cached_frames=cached_prior_frames,
        )
        engine_result = primary_engine.score(
            bg_images, dict(current_frames), calibrations,
            prior_dart_line_px=prior_dart_line_px,
        )
    else:
        engine_result = primary_engine.score(bg_images, dict(current_frames), calibrations)
    # The v2 package schema (2026-08-27) -- computed from the
    # raw `engine_result` (still has `.diagnostics`), BEFORE the
    # ScoreResult conversion below discards it. See
    # `_agreement_string_from_engine_result()`'s own docstring: None
    # (never fabricated) when `primary_name` isn't a vote-based engine.
    agreement = _agreement_string_from_engine_result(engine_result)
    # Live single-camera-fuse detection. Motivating incident: a D17
    # throw in one recorded session -- three independent per-engine
    # board-plausibility gates each discarded 2 genuine off-board camera
    # detections, leaving a single degenerate ray that the shared
    # "assume dart is on Z=0" fallback then scored as a wrong on-board
    # answer, with no live signal anywhere that only 1 camera actually
    # contributed. Purely additive: `n_cameras_used` (the "reached and
    # influenced the final answer" quantity, same semantic established
    # for all 5 engines' diagnostics this same day) is already computed
    # by the primary engine's own `score()` call above -- this is a
    # single dict `.get()` on data that already exists, no new
    # detection/recompute, same "record what happened, don't change
    # what happens" shape as the ring-offset fix's own effective-radii
    # log line a few lines above this function. Zero added cost to the
    # capture loop: reading a key out of an already-built dict is O(1)
    # and unconditional (not gated behind a DEBUG check the way the
    # settle-diagnostics instrumentation was, which is exactly why that
    # one needed a real before/after timing falsification and this one
    # does not).
    n_cameras_used = (
        engine_result.diagnostics.get("n_cameras_used")
        if engine_result.diagnostics else None
    )
    if n_cameras_used == 1:
        log.warning(
            "single-camera fuse: primary engine %s scored this throw "
            "from only 1 camera's data (n_cameras_used=1) -- depth "
            "along that one camera's own ray is unconstrained, so the "
            "on-board/off-board call is unusually unreliable; "
            "sector=%s ring=%s reason=%s",
            primary_name, engine_result.sector, engine_result.ring,
            engine_result.reason,
        )
    if own_tip_line_px_out is not None:
        # Same "must run before the ScoreResult conversion discards
        # .diagnostics" timing as `agreement` immediately above -- see
        # own_tip_line_px_out's own docstring section for the full
        # reasoning.
        _own_tip_line_px = _own_tip_line_px_from_engine_result(engine_result)
        if _own_tip_line_px is not None:
            own_tip_line_px_out["own_tip_line_px"] = _own_tip_line_px
    camera_mode = "real"
    result = (
        apollo_engine_result_to_score_result(engine_result)
        if primary_name == DEFAULT_PRIMARY_ENGINE
        else engine_result_to_score_result(engine_result)
    )
    engine_duration_s = time.monotonic() - _t_engine # PER-THROW TIMING, see _t_handle_start above

    # Self-descriptive throw naming: a raw epoch-ms
    # `throw_id` told a human nothing at a glance. The format is session
    # id, sequential per-session throw number, and a standard-darts-notation
    # sector token (e.g. `T8`). sector_token comes from THIS (the primary engine's own)
    # result -- this project's own live score, not an external one, names
    # the package. Expressed in this
    # project's own (sector, ring) vocabulary via
    # opendarts.geometry.board.sector_ring_to_token(). An ok=False primary
    # result (no score at all -- a real, honest case this project always
    # still saves a full replay package for, per docs/DESIGN.md's
    # "Replay is the source of truth") gets the literal token "NR" (no result) instead -- never
    # a call into sector_ring_to_token() with a None sector/ring, which
    # isn't a real board position.
    #
    # throw_number bug + fix: originally derived by
    # counting existing throw directories under this session, on the
    # reasoning that handle_ready_to_capture() is called exactly once per
    # throw, sequentially, within one process's life, so disk state and
    # the true count could never diverge. That reasoning missed a real
    # external actor: the standing pull SOP's quarantine step
    # (an off-rig pull that runs `mv packages/* $DEST/`) moves
    # a session's throw directories out from under a STILL-RUNNING
    # daemon mid-session (session_id itself doesn't change -- it's
    # generated once per process start, not per pull, and there's no
    # reason it should: this is still logically one continuous physical
    # sitting). The next throw after a quarantine then recounted 0
    # existing dirs and restarted numbering at 001, producing real
    # `session_id`+throw_number collisions against already-pulled
    # throws of the same session (caught and hand-fixed once already --
    # see data/archive/clean/README.md).
    # Fixed by persisting the counter OUTSIDE `package_root` entirely
    # (a sibling of `data/packages/`, never `data/packages/*` itself) so
    # a `packages/*` quarantine glob structurally cannot sweep it up --
    # the counter survived any number of mid-session quarantines,
    # numbering continuing seamlessly across a pull.
    #
    # SUPERSEDED, 2026-08-22. The 2026-08-17
    # fix's own goal -- numbering survives a pull unchanged -- turned out
    # to be the OPPOSITE of what's actually wanted operationally: a
    # single missed/dropped throw anywhere permanently offsets every
    # later "dart N" from what can see on the board, with no way
    # back in sync. Reversed: a pull that empties this session's local
    # directory (detected below -- no live signal exists for an event
    # that happens over SSH on a different machine, so this is inferred
    # from disk state, not pushed) now RESETS numbering, same as a manual
    # Reset or Delete-packages -- via a new counting GENERATION rather
    # than a bare reset-to-0-in-place, which closes the exact collision
    # risk the 2026-08-17 fix above was originally protecting against
    # (see _reset_session_throw_numbering()'s own docstring for the full
    # collision analysis). session_id itself is UNCHANGED by any of this
    # -- one folder per physical sitting either way.
    session_dir = package_root / session_id
    counters_dir = package_root.parent / "session_throw_counters"
    counters_dir.mkdir(parents=True, exist_ok=True)
    counter_file = counters_dir / f"{session_id}.count"
    generation_file = counters_dir / f"{session_id}.generation"

    # _THROW_NUMBER_LOCK, 2026-09-06/07 -- see that lock's own module-
    # level comment for the full incident/design writeup. Everything from
    # the AUTO-DETECTED RESET check through `dest_dir`'s own final name
    # must be one atomic critical section now that this whole function
    # can run concurrently for two different throws of the SAME session
    # (Part 2, "background the entire handle_ready_to_capture() call") --
    # a read-increment-write on `counter_file` with no lock would let two
    # overlapping allocations collide onto the SAME throw_number/dest_dir.
    with _THROW_NUMBER_LOCK:
        # AUTO-DETECTED RESET: counter_file existing means throws already
        # happened in the CURRENT generation; session_dir missing/empty means
        # nothing from that generation is on disk any more. The only way both
        # are true at once is an external wipe this process was never told
        # about (a pull's quarantine step, `mv packages/* $DEST/`, run over
        # SSH from a different machine) -- Delete-packages and manual Reset
        # both call _reset_session_throw_numbering() directly and
        # immediately instead of relying on this (see their own call sites),
        # so by the time this check runs after either of THOSE, counter_file
        # is already gone and this branch correctly does not fire again.
        if counter_file.exists() and not (session_dir.exists() and any(session_dir.iterdir())):
            _reset_session_throw_numbering(counters_dir, session_id)

        if counter_file.exists():
            throw_number = int(counter_file.read_text().strip()) + 1
        else:
            # BUG, found and fixed 2026-08-22 -- caught by an
            # independent verification pass, not this project's own test
            # suite (real gap: no test here ever exercised "capture a throw
            # after a manual Reset while the OLD generation's packages are
            # still on disk", which is exactly the shape a Reset produces --
            # Reset deliberately does NOT delete existing packages, only
            # Delete-packages/an external pull do). The bootstrap disk-recount
            # below (`sum(1 for p in session_dir.iterdir() ...)`) counts
            # EVERY directory under session_dir, with no notion of
            # generation -- after a Reset, those directories are the PRIOR
            # generation's real throws, still physically present, so this
            # recounted them and inflated the new generation's first
            # throw_number past 1 (e.g. g1-003 instead of g1-001) --
            # confirmed by direct reproduction, not just reasoning about it.
            #
            # Fix: the disk-recount fallback is only trustworthy for a
            # session's ORIGINAL generation (generation_file has never been
            # written) -- that's the one case where "count what's really on
            # disk" and "what generation 0's next throw_number should be"
            # are the same question (a session that started before the
            # 2026-08-17 counter-persistence fix shipped, or whose counter
            # file was lost some other way, but never actually reset).
            # Once ANY reset has happened for this session_id, counter_file
            # being absent can only mean "between the reset and this
            # generation's first throw" -- throw_number must unconditionally
            # be 1, regardless of what's left on disk from an earlier
            # generation.
            throw_number = 1
            if session_dir.exists() and not generation_file.exists():
                throw_number = sum(1 for p in session_dir.iterdir() if p.is_dir()) + 1
        counter_file.write_text(str(throw_number))

        generation = int(generation_file.read_text().strip()) if generation_file.exists() else 0
        # Generation infix, 2026-08-22: OMITTED for generation 0 (the common
        # case -- no reset has ever happened this session, throw_id looks
        # exactly like it always has, zero visible change for anyone who
        # never resets) and only appears as `-g{N}-` once a real reset has
        # occurred -- see _reset_session_throw_numbering()'s own docstring.
        generation_infix = f"-g{generation}" if generation > 0 else ""

        sector_token = "NR" if not result.ok else sector_ring_to_token(result.sector, result.ring)
        throw_id = f"{session_id}{generation_infix}-{throw_number:03d}-{sector_token}"
        dest_dir = package_root / session_id / throw_id
    # captured_at_utc computed HERE, explicitly, ONCE -- 2026-09-01,
    # "background the throw-package save" task. Used
    # for BOTH the THROW_DETECTED event below (fired immediately) AND
    # save_throw_package()'s own meta.json (written moments later, on a
    # background thread) -- a single source of truth is the only way the
    # two can still agree now that the event no longer waits for the
    # write to complete and read the timestamp back. See save_throw_
    # package()'s own `captured_at_utc` parameter docstring for the full
    # reasoning.
    captured_at_utc = datetime.now(timezone.utc).isoformat()
    # PER-THROW TIMING (sync portion) -- measured HERE, right before
    # THROW_DETECTED emits, so it reflects everything a consumer of that
    # event actually waited on (engine compute),
    # now that package save/diagnostics writes/also-run dispatch are all
    # backgrounded below and no longer part of what gates this event.
    # The background thread's own timing (save/diagnostics) logs
    # separately, once it actually completes -- see _save_and_followups()
    # below.
    handle_total_duration_s = time.monotonic() - _t_handle_start
    log.info(
        "handle_ready_to_capture TIMING (sync path): total=%.3fs | "
        "engine=%.3fs | primary_engine=%s",
        handle_total_duration_s, engine_duration_s, primary_name,
    )
    # THE single "a dart was just scored" push (docs/LIVE_API.md) --
    # fired BEFORE the package is written to disk (2026-09-01: publish
    # the throw event before writing the package to disk, taking the
    # file write off the critical path -- real measured numbers elsewhere:
    # save=0.165s of a 0.470s total, confirmed
    # against our own logs too, save=0.153s of 0.421s). Uses the explicit
    # `captured_at_utc` above, never a disk read-back (there is nothing
    # on disk to read yet at this point).
    #
    # `emitted_at_utc` (added 2026-09-07, requested by the QA harness on
    # :8900 -- TRIGGER_STATE already carries this same field, added
    # 2026-09-01, see that emit site's own docstring for the full
    # reasoning): reuses `captured_at_utc` verbatim rather than a second
    # `datetime.now(...)` call -- that variable is already stamped at
    # exactly the right moment (right after the primary engine decided
    # this result, before this emit and well before the backgrounded
    # package write), so a fresh clock read here would only add a
    # second, marginally-later timestamp with no real meaning of its
    # own. Same key name as TRIGGER_STATE's own field so a consumer can
    # measure capture(TRIGGER_STATE)->answer(THROW_DETECTED) using one
    # convention on both sides instead of inferring it from `ts`.
    # What the scoring page's board photo needs to draw this dart the way
    # it looks from the oche: its lean and its flight colour, both read off
    # what scoring already produced (see opendarts.live.board_photo). Each
    # is omitted when it cannot be worked out, and neither may ever cost a
    # throw its score.
    board_view_fields: dict[str, Any] = {}
    if result.ok and result.board_xy_mm is not None:
        try:
            axis = board_photo.dart_axis(engine_result.diagnostics, calibrations)
            if axis is not None:
                board_view_fields["dart_axis"] = axis
                colour = board_photo.flight_color(
                    current_frames, bg_images, calibrations,
                    tuple(result.board_xy_mm), axis,
                )
                if colour is not None:
                    board_view_fields["flight_color"] = colour
        except Exception:  # noqa: BLE001 -- decoration; never the score's problem
            log.warning("board view: dart axis/colour failed", exc_info=True)
    _emit(
        on_event,
        {
            "type": "THROW_DETECTED",
            "emitted_at_utc": captured_at_utc,
            "session": session_id,
            "throw_id": throw_id,
            "path": str(dest_dir),
            "captured_at_utc": captured_at_utc,
            "visit_id": visit_id,
            "visit_index": visit_index,
            "primary_engine": primary_name,
            "ok": result.ok,
            "sector": result.sector,
            "ring": result.ring,
            "board_xy_mm": list(result.board_xy_mm) if result.board_xy_mm is not None else None,
            "n_cameras_used": result.n_cameras_used,
            "reason": result.reason,
            # NOT a confidence score -- this project has none, anywhere
            # (checked 2026-08-14: no `confidence` field exists on
            # ScoreResult, EngineResult, or any engine's diagnostics,
            # unlike some throw cards, which carry one). This is the
            # real quality signal that DOES exist: how far apart the
            # per-camera rays landed, in millimetres, lower = better.
            # Named for what it actually is rather than dressed up as a
            # 0-1 confidence that nothing in this system computes.
            "max_ray_disagreement_mm": result.max_ray_disagreement_mm,
            **board_view_fields,
        },
    )
    # No board photo here (2026-09-26). It used to be rendered from the
    # first dart's bg, but ~0.2 s of CPU inside dart 1's burst is exactly
    # when the Pi has none to spare; run_capture_loop_body() now renders
    # it after each takeout instead, from the same empty board.

    def _save_and_followups() -> None:
        """2026-09-01 -- everything that used to run
        synchronously between the engine scoring above and this
        function's own return, now deferred to a daemon thread so
        THROW_DETECTED (already emitted above) never waits on any of it.
        Same fire-and-forget idiom `save_calibration_package_background()`
        already established for calibration packages (opendarts.capture.
        calibration_package) -- not a new pattern for this codebase, just
        the first time it's applied to a THROW package specifically.

        Runs in the EXACT SAME relative order today's synchronous code
        did (save -> capture_diagnostics.json ->
        PACKAGE_SAVED emit -> AD-ground-truth attach -> also-run
        dispatch; since 2026-09-26 the package's one clip is started
        between the attach and the dispatch, on its own thread, and --
        in "mismatch" video-record mode -- re-emits PACKAGE_SAVED once
        meta.json points at it -- see _write_package_clip()) -- moved as one atomic unit, not individually
        reordered, so every existing ordering assumption downstream code
        already relies on (e.g. `_dispatch_also_run_engines_in_
        background()`'s own docstring: "writes into the throw's
        ALREADY-SAVED result.json") stays true.

        **Honest, real tradeoff, not hidden**: `save_throw_package()`
        "fails loudly (raises) rather than writing a partial package" --
        true, but a raise inside a daemon thread has no caller left to
        propagate to. Before this change, a real save failure was a loud,
        synchronous exception surfacing all the way up through
        run_capture_loop_body()'s own exception handling; now it is a
        caught, logged (log.exception, not swallowed silently) background
        failure -- less visible, structurally, than before. This is the
        SAME tradeoff calibration packages already accept via `save_
        calibration_package_background()`; applying it here for the first
        time to THROW packages (this project's own real REPLAY corpus,
        not just calibration data) is a materially bigger exposure than
        that precedent alone justifies by itself -- flagged plainly, not
        quietly ported on the strength of the calibration precedent
        alone.
        """
        try:
            _t_save = time.monotonic() # PER-THROW TIMING, see _t_handle_start above
            save_throw_package(
                dest_dir=dest_dir,
                session=session_id,
                bg_frames_bgr=bg_frames,
                dart_frames_bgr=current_frames,
                calibrations=calibrations,
                result=result,
                visit_id=visit_id,
                visit_index=visit_index,
                calibration_package_id=calibration_package_id,
                # The v2 package schema (2026-08-26): the SAME
                # session-sequential `throw_number` local already computed
                # above (and already embedded in `throw_id`/`dest_dir`'s
                # own name) -- threaded straight through, never re-derived
                # a second way.
                throw_number=throw_number,
                # The v2 package schema (2026-08-27) -- both
                # computed just above, before `result` (the narrowed
                # ScoreResult) lost the raw `engine_result.diagnostics`
                # this needed.
                camera_mode=camera_mode,
                agreement=agreement,
                # The v2 package schema (2026-08-27, first real
                # v2-session QA pass) -- the SAME session-sequential
                # `generation` local already computed above (and already
                # embedded in `throw_id`/`dest_dir`'s own `-g{N}-` infix),
                # threaded straight through as a real int field, never
                # re-derived a second way. And the raw primary
                # `EngineResult.diagnostics` -- same "captured before the
                # ScoreResult conversion discards it" reasoning as
                # `agreement`/`camera_mode` immediately above -- used by
                # `save_throw_package()` to fill result.json's top-level
                # rollup (`cameras_used`/`triangulation`/
                # `max_ray_disagreement_mm`/`n_cameras_used`) from the
                # winning sub-engine's own diagnostics when the primary is
                # a vote-based consensus engine (Zeus/"Zeus") and `result`
                # itself left those fields at their honest "not
                # populated" defaults.
                generation=generation,
                primary_engine_diagnostics=engine_result.diagnostics,
                # 2026-09-01 -- see this parameter's own docstring on
                # save_throw_package(): a single explicit timestamp,
                # shared with the THROW_DETECTED event already emitted
                # above, instead of this call generating its own.
                captured_at_utc=captured_at_utc,
                # Which rig scored this, and with what code. Read here
                # rather than inside save_throw_package() because
                # opendarts.capture must not import from opendarts.live --
                # see that parameter's own comment for why a package has
                # to be able to answer both questions once it has been
                # pulled somewhere else.
                host=_origin_host(),
                build=_origin_build(),
                # The data files only: the package's ONE clip is written
                # just below, once its recording decision is known -- see
                # _write_package_clip().
                defer_clips=True,
            )
            # Core package-save duration only (matches the peer's own
            # "package save" bucket) -- deliberately measured before the
            # capture_diagnostics.json write just below, which is
            # additional, separate I/O this function also does but that
            # isn't part of save_throw_package() itself.
            save_duration_s = time.monotonic() - _t_save
            # capture_diagnostics.json (2026-08-16, "persist real
            # diagnostics" task -- see docs/DESIGN.md and opendarts.capture.
            # throw_package's own docstring section, and
            # _build_capture_diagnostics()'s own docstring for the real
            # incident this responds to).
            try:
                diagnostics = _build_capture_diagnostics(
                    trigger, ad_ws_listener, captured_at_monotonic=_t_handle_start,
                    handle_total_s=handle_total_duration_s,
                )
                save_capture_diagnostics(dest_dir, diagnostics)
            except Exception: # noqa: BLE001 -- diagnostics must NEVER affect capture reliability
                log.exception(
                    "capture_diagnostics.json build/save failed for %s -- the throw "
                    "package itself (already saved) is unaffected",
                    dest_dir,
                )
            # LOG-LINE n_cameras_used FIX, 2026-09-06 -- real finding from
            # a live ghost-fire investigation, not a design change: this
            # line was logging `result.n_cameras_used` directly, which
            # `opendarts.engines.base.engine_result_to_score_result()` (used
            # whenever the live primary is a vote-based consensus engine
            # like Zeus/"Zeus", not Apollo) DELIBERATELY hardcodes to 0
            # -- "n_cameras_used similarly has no generic engine-level
            # equivalent -- left at 0", per that function's own docstring.
            # That's a correct, honest default for the field ON A
            # ScoreResult -- but this log line was reading it as if it
            # were the real diagnostic value, and it read 0 on EVERY
            # package tonight, including ones that scored correctly with
            # confident, unanimous 4/4 votes -- a real risk of misleading
            # whoever reads this line next (it nearly misled this
            # investigation itself: the real result.json for one such
            # package showed n_cameras_used=2 with max_ray_disagreement_mm
            # =0.029mm, an extremely tight real agreement).
            # `result.json`'s own top-level field is ALREADY correct --
            # `save_throw_package()` promotes it from the winning sub-
            # engine's own diagnostics via `_rollup_fields_from_winning_
            # sub_engine_diagnostics()` a few lines below this log call.
            # This just reads the SAME already-existing rollup for the
            # log line too, instead of the pre-rollup ScoreResult field --
            # no new computation, just stopped discarding data that was
            # already being computed one call away.
            _log_n_cameras_used = _rollup_fields_from_winning_sub_engine_diagnostics(
                engine_result.diagnostics
            ).get("n_cameras_used", result.n_cameras_used)
            log.info(
                "saved throw package %s (ok=%s sector=%s ring=%s n_cameras_used=%s "
                "primary_engine=%s visit=%s/%s)",
                dest_dir,
                result.ok,
                result.sector,
                result.ring,
                _log_n_cameras_used,
                primary_name,
                visit_id,
                visit_index,
            )
            log.info(
                "background throw-package save TIMING: save=%.3fs "
                "| primary_engine=%s",
                save_duration_s, primary_name,
            )
            _emit(
                on_event,
                {"type": "PACKAGE_SAVED", "path": str(dest_dir), "session": session_id},
            )
            # Fire off (background, never blocking) immediately after the
            # package is durably saved -- see
            # _attach_ad_ground_truth_from_ws()'s own docstring for why
            # this is safe regardless of AD's reachability. Now correctly
            # sequenced AFTER the real save above (same thread, so no race
            # with the write the way triggering it from the OLD
            # synchronous call site while THIS save was still backgrounded
            # would have been).
            verdict = _OracleVerdict()
            if not _attach_ad_ground_truth_from_ws(
                dest_dir, ad_ws_listener, ad_match_window_sec,
                on_event=on_event, session_id=session_id, verdict=verdict,
            ):
                verdict = None  # no oracle will answer -- nothing to wait for
            # THE PACKAGE'S ONE CLIP (2026-09-26), after its data: a
            # recording out of the frame ring, or the two frames scoring
            # used, then meta.json pointed at it (and, in "mismatch" mode,
            # PACKAGE_SAVED again -- see _write_package_clip()).
            # The camera JPEGs and ring generations are paired with these
            # exact arrays by identity at the tick they were fetched (the
            # capture loop's _FrameJpegIndex); a camera without them just
            # gets an FFV1 and/or two-frame clip. Its own thread, like the
            # attach above: it may wait on the next ring frame or on the
            # oracle, and nothing below should. Inline when the caller
            # asked for a synchronous save.
            clip_cameras = sorted(set(bg_frames) & set(current_frames) & set(calibrations))
            clip_job = functools.partial(
                _write_package_clip,
                dest_dir,
                clip.ScoredFrames(
                    bg={c: bg_frames[c] for c in clip_cameras},
                    commit={c: current_frames[c] for c in clip_cameras},
                    bg_jpegs=dict(trigger.bg_jpegs or {}),
                    commit_jpegs=dict(trigger.last_frame_jpegs or {}),
                    bg_generations=dict(trigger.bg_generations or {}),
                    commit_generations=dict(trigger.last_frame_generations or {}),
                ),
                clip_cameras,
                throw_capture,
                verdict=verdict,
                # The capture instant, as the misscore anchor defines it
                # (throw_capture.anchor_wall_s_for_package): the stamp minus
                # the sync path that preceded it.
                anchor_wall_s=(
                    datetime.fromisoformat(captured_at_utc).timestamp()
                    - handle_total_duration_s
                ),
                on_event=on_event,
                session_id=session_id,
            )
            if background_save:
                threading.Thread(
                    target=clip_job, name=f"throw-clip-{dest_dir.name}", daemon=True,
                ).start()
            else:
                clip_job()

            if also_run:
                bg_images = {
                    cam: frame for cam, frame in bg_frames.items() if cam in current_frames
                }
                # 2026-08-24 fix (see opendarts.engines.zeus.engine's module
                # docstring for the full incident this closes): the
                # also-run dispatch path never had any way to forward
                # `prior_dart_line_px` at all before this, so Apollo got
                # zero prior-dart contamination protection whenever it ran
                # as an also-run engine rather than the configured primary
                # -- same gap as the primary-engine branch above, same fix
                # shape. Only bothers with the lookup when at least one
                # also-run engine would actually use it.
                # 2026-08-27 perf task -- see _reuse_zeus_sub_results_for_
                # also_run()'s own docstring. When Zeus is primary and
                # also_run overlaps its own ZEUS_SUB_ENGINE_NAMES, this
                # recovers those engines' already-computed answers (from
                # `engine_result.diagnostics`, captured moments ago above,
                # before the ScoreResult conversion) instead of paying to
                # compute them a second time in the background dispatch
                # below -- {} (no-op) for every other primary engine or a
                # non-overlapping also_run.
                reused_sub_results = _reuse_zeus_sub_results_for_also_run(
                    primary_name, engine_result.diagnostics, also_run,
                )
                # Only the engines that will actually RUN need the prior
                # dart's line; one whose answer is reused from Zeus never
                # reads it. Looking it up for those too recomputed
                # detect_tip() on three cameras for nothing (2026-09-17).
                also_run_prior_dart_line_px = None
                if any(
                    engine_accepts_prior_dart_line_px(get_engine(name))
                    for name in also_run if name not in reused_sub_results
                ):
                    also_run_prior_dart_line_px = find_prior_dart_line_px(
                        package_root / session_id, visit_id, visit_index,
                        cached_frames=cached_prior_frames,
                    )
                _dispatch_also_run_engines_in_background(
                    dest_dir, bg_images, dict(current_frames), calibrations,
                    primary_name, also_run, engine_config.timeout_s,
                    session_id=session_id, on_event=on_event,
                    prior_dart_line_px=also_run_prior_dart_line_px,
                    reused_sub_results=reused_sub_results,
                )
        except Exception: # noqa: BLE001 -- see this function's own docstring:
            # a background save failure must never take down the capture
            # loop (there is no caller left to catch it by the time this
            # runs), but it MUST be loud in the log -- this is a real,
            # replay-affecting failure (the throw was announced via
            # THROW_DETECTED but its package never made it to disk), not
            # a routine, ignorable background hiccup.
            log.exception(
                "BACKGROUND THROW-PACKAGE SAVE FAILED for %s -- THROW_DETECTED already "
                "fired for this throw, but no package exists on disk for it. This is a "
                "real REPLAY gap for this one throw, not a routine failure.",
                dest_dir,
            )

    if not store_packages:
        # Package storage disabled (data/config.json's
        # `store_packages`). Everything skipped here is on-disk work:
        # save_throw_package, the capture-diagnostics file, the AD-ground-truth attach, and the also-run dispatch
        # (which exists only to write other engines' answers INTO
        # result.json -- with no result.json it has nothing to write).
        #
        # Scoring is unaffected. THROW_DETECTED has already been emitted
        # above, BEFORE this block, so the live feed and the retail
        # channel behave identically with storage on or off -- which is
        # the whole point: a shipped rig scores darts without
        # accumulating ~6.5 MB of frames per throw. PACKAGE_SAVED simply
        # never fires, because no package was saved.
        log.debug("package storage disabled -- not writing %s", dest_dir)
        return dest_dir

    # THE DISK FLOOR. Checked here, on the same line as `store_packages`
    # above and returning the same way, because the two conditions have
    # the same consequence: no package on disk for this throw, and
    # nothing else changed. THROW_DETECTED has already fired; the live
    # feed, the retail channel and match history never read a package, so
    # scoring continues exactly as it would with storage turned off.
    #
    # `dest_dir` is returned WITHOUT having been created, which is what
    # `store_packages: false` has always done -- so no caller is handed a
    # path it may assume exists (`run_capture_loop_body()` does not read
    # this return value at all on the background path, and the
    # followups that DO write into `dest_dir` all live inside
    # `_save_and_followups()`, which is never reached from here).
    disk = check_free_space(dest_dir, floor_gb=min_free_disk_gb)
    if not disk.ok:
        _report_package_skipped_for_disk(session_id, disk, dest_dir)
        return dest_dir

    if background_save:
        threading.Thread(
            target=_save_and_followups, name="throw-package-save", daemon=True,
        ).start()
    else:
        # background_save=False -- see this parameter's own docstring:
        # runs the exact same unit inline, synchronously, byte-identical
        # to this function's pre-2026-09-01 behavior.
        _save_and_followups()

    return dest_dir


def _dispatch_handle_ready_to_capture_in_background(*args: Any, **kwargs: Any) -> None:
    """2026-09-06/07, Zeus-latency follow-up task, Part 2 -- "background
    the PRIMARY engine's score call, not just the save." Runs the ENTIRE
    `handle_ready_to_capture()` call (args/kwargs forwarded verbatim) on
    its own daemon thread, fire-and-forget from `run_capture_loop_body()`'s
    point of view -- mirrors `_dispatch_also_run_engines_in_background()`'s
    own established shape exactly (same fire-and-forget idiom, same
    "enrichment/scoring must never affect the trigger/settle state
    machine's own progression" discipline), just one level up: THIS
    function backgrounds the PRIMARY engine's own scoring call (plus
    everything `handle_ready_to_capture()` already backgrounds
    internally via `background_save`), not just the also-run engines.

    **Why this was needed, concretely measured, not assumed.** Before
    this task, `handle_ready_to_capture()` (primary-engine `score()` +
    session/throw-number naming + `THROW_DETECTED` emit) ran
    SYNCHRONOUSLY on `run_capture_loop_body()`'s own thread -- only the
    package SAVE (2026-09-01) was backgrounded. While that synchronous
    portion runs, the loop cannot fetch a fresh frame or refresh `motion_
    bg_frames`, which measurably worked against the SAME-DAY two-buffer-
    split absorption window (`POST_CAPTURE_REFRACTORY_WINDOW_S`, see
    `168e8b8`) -- a slow primary engine (Zeus's own real 4-sub-engine
    vote, or Apollo's own RANSAC fallback tiers) could eat a real
    fraction of that window before the loop even resumes watching for
    the next dart.

    **Real, new risk this introduces, closed alongside this function
    (not left as a caveat)**: a fast follow-up dart can now trigger a
    SECOND `handle_ready_to_capture()` call while the FIRST one's own
    background thread is still mid-flight -- structurally impossible
    before this task (the whole function was synchronous, so the loop
    physically could not reach a second READY_TO_CAPTURE handling block
    until the first one's ENTIRE body had already returned). The one
    piece of `handle_ready_to_capture()`'s own internal state that is
    NOT safe under this kind of overlap by construction is its throw-
    number/generation allocation (a plain read-increment-write on a
    shared counter FILE, previously safe only because it could never
    run twice at once for the same session) -- see `_THROW_NUMBER_LOCK`'s
    own module-level comment for the full incident this would otherwise
    cause (two throws colliding onto the SAME `dest_dir`) and the fix
    (a lock around that one critical section, held by all three real
    writers of `session_throw_counters/*`).

    **Traced and confirmed NOT a similar risk, per this task's own
    explicit instruction to check rather than assume**: `CalibrationStore`
    reads (`calibration_store.get_with_package_id()`, already a single
    atomic read per its own docstring, called by the LOOP before
    dispatching -- each background call gets its own independently-
    fetched, immutable snapshot, no shared mutable state to race);
    `EngineConfigStore`/`ad_ws_listener` (already
    exercised under real cross-throw concurrent background execution
    since 2026-09-01's own save-backgrounding -- this task extends an
    ALREADY-accepted concurrency model to cover scoring too, it does not
    introduce a first instance of it); `_dispatch_also_run_engines_in_
    background()`'s own further nested background dispatch (already
    designed to run concurrently with anything else in this process,
    per its own docstring).

    **Exceptions are caught and logged here, never propagated** -- there
    is no caller left to catch them once this thread starts (same
    contract every other fire-and-forget dispatch in this module already
    has). Real, honest tradeoff, not hidden: before this task, an
    exception from `handle_ready_to_capture()` itself (e.g. its own
    `RuntimeError` for "READY_TO_CAPTURE with no captured frame set on
    the trigger... indicates a bug... not something this function should
    paper over") propagated all the way up through `run_capture_loop_
    body()`, crashing the process loudly -- exactly the visibility that
    docstring's own wording wants for a real bug. Backgrounding this call
    means such an exception is now caught-and-logged instead, matching
    the SAME already-accepted tradeoff `_save_and_followups()`'s own
    docstring names for the save step it backgrounds ("less visible,
    structurally, than before... flagged plainly, not quietly ported").
    """
    try:
        handle_ready_to_capture(*args, **kwargs)
    except Exception: # noqa: BLE001 -- see this function's own docstring: no caller is left to catch this
        log.exception(
            "BACKGROUND handle_ready_to_capture() FAILED -- this throw was never "
            "scored/saved at all (unlike a background SAVE failure, which still has "
            "a real primary result on disk; this is an earlier failure, before "
            "scoring even completed). A real, replay-affecting gap for this one "
            "throw, not a routine background hiccup."
        )


def _emit(on_event: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]) -> None:
    """Best-effort delivery of a real state-change event to an optional
    caller-supplied sink -- see run_capture_loop_body()'s on_event param
    and opendarts/live/run_product.py, which is the actual caller that
    passes a non-None one (a thread-safe queue.SimpleQueue.put, pushing
    real trigger-state transitions and saved-package events straight to
    the web server's WebSocket broadcast, instead of the server
    discovering them later via its own polling fallback). A broken/raising
    sink must never take down the capture loop itself -- this is capture,
    not UI, and UI problems are not this loop's problem."""
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception: # noqa: BLE001 -- a bad event sink must not break capture
        log.exception("on_event callback raised -- ignoring (capture loop must keep running)")


def _wait_for_next_iteration(
    stop_event: threading.Event, iteration_started: float, poll_interval_s: float
) -> None:
    """Deadline-compensated replacement for a flat `stop_event.wait(
    poll_interval_s)` in `run_capture_loop_body()`'s own main loop --
    2026-09-05, Zeus-latency follow-up task.

    Real measured problem this fixes: a plain `threading.Event.wait(N)`
    (and, for that matter, `time.sleep(N)`) overshoots its requested
    duration by a real, repeatable amount on this platform -- a direct
    Rig benchmark (n=300 each, clean interpreter, this task) measured
    `Event.wait(0.05)` at 57.45ms and `time.sleep(0.05)` at 57.38ms
    (`asyncio.sleep(0.05)` was faster, 52.11ms, but is NOT used here --
    see below) -- roughly 5ms of pure overshoot per call, on top of
    whatever this iteration's own real work (fetch/advance/save) already
    cost. The OLD code called a flat `stop_event.wait(poll_interval_s)`
    at the end of EVERY iteration regardless of how long that
    iteration's own body took, so a slow iteration's cost was never
    absorbed -- it was added ON TOP of a still-full poll_interval_s
    wait, and this compounded across the ~2.5 waits/settle-episode a
    real dart's settle window typically takes, the real ~10ms/dart cost
    this task measured and was asked to close.

    Fix: track the real intended "next tick" deadline as `iteration_
    started + poll_interval_s` (the SAME `iteration_started =
    time.monotonic()` timestamp this loop already samples at the very
    top of every iteration, per this task's own instruction to reuse
    the existing per-iteration diagnostic timestamp rather than adding a
    new one) and wait only the REMAINING time until that deadline,
    `max(0.0, ...)`. A slow iteration's own overshoot is absorbed into
    the NEXT wait being correspondingly shorter, instead of accumulating
    additively -- and an iteration that ran PAST its own deadline
    (e.g. a real capture + engine score) correctly waits zero rather
    than adding a full extra interval on top of an already-slow
    iteration, which is the actual accumulation bug this closes.

    **Does NOT shorten the configured `POLL_INTERVAL_SECONDS` itself** --
    this only removes accumulated drift/overshoot around the SAME
    nominal interval; on a fast, do-nothing iteration this waits the
    full `poll_interval_s`, same as before.

    **Interrupt semantic preserved, deliberately NOT switched to
    `asyncio.sleep()`** even though it measured marginally faster on the
    same benchmark above: this loop's whole shutdown story depends on
    `stop_event.wait()` returning IMMEDIATELY the instant `stop_event`
    is set, mid-wait, rather than blocking out the full duration first --
    a real, load-bearing behavior (see this function's own docstring,
    "Stopping:" section) that `asyncio.sleep()` cannot provide (it has no
    equivalent externally-settable interrupt) and `time.sleep()` cannot
    either. Trading that away for a few milliseconds of raw sleep-call
    speed would be a regression, not an optimization.

    **Checked for a real "at least poll_interval_s between frames"
    assumption elsewhere in this module before shipping, per this task's
    own explicit instruction -- found none.** `SETTLE_WINDOW_FRAMES`/
    `is_settled()` count discrete ADVANCE() ITERATIONS, not elapsed
    wall-clock time, and never assume a minimum inter-iteration gap;
    `MAX_FRAME_FRESHNESS_WAIT_S`'s own frame-freshness gate is a CEILING
    on how long a stalled camera's stale frame is tolerated, not a floor
    tied to this interval; and the per-iteration diagnostic reconciliation
    check just above this loop's own top (`_recon_gap_s`) already
    expects real inter-iteration timing to vary and only flags a
    genuinely unexplained gap, not a specific minimum. A wait that is
    sometimes slightly SHORTER than poll_interval_s (to correct for a
    previous iteration's own overshoot) does not violate anything this
    module relies on.
    """
    remaining = poll_interval_s - (time.monotonic() - iteration_started)
    stop_event.wait(max(0.0, remaining))


# FRAME-DRIVEN WAKE, 2026-09-05 -- an approved architectural change: go
# frame-driven, so every frame is processed rather than polled and
# missed. See local_capture.LocalCameraHub's own module docstring
# ("FRAME-DRIVEN WAKE PRIMITIVE" section) for the hub-side primitive
# this consumes, and `_wait_for_next_frame()`'s own docstring below for
# the full design.
#
# Bounds how long each individual wait "slice" blocks before this loop
# re-checks `stop_event`, INDEPENDENT of `poll_interval_s` -- see
# `_wait_for_next_frame()`'s own "Interrupt semantic" section for the
# real tradeoff this creates and why it's deliberately kept tight rather
# than reusing `poll_interval_s` itself (which would bound shutdown
# latency to ~50ms instead of ~10ms). NOT measured against real live
# shutdown latency on the rig (no live-system access for the session that
# built this) -- structurally reasoned the same way `MAX_FRAME_
# FRESHNESS_WAIT_S`'s own comment already reasons about a similar
# bound: comfortably smaller than this rig's own real ~32ms pump cadence
# (per `POLL_INTERVAL_SECONDS`'s own dated comment) so it never meaningfully
# delays a genuine frame-arrival wakeup (`Condition.wait(timeout)`
# returns the INSTANT it's notified, regardless of the requested
# timeout -- the timeout only matters when nothing arrives at all), while
# still being small enough that a real `stop_event.set()` is observed
# within roughly one order of magnitude of the OLD flat-timer wait's own
# near-instant `Event.wait()` interrupt, not the ~50ms `poll_interval_s`
# ceiling that would otherwise be a real, if bounded, regression.
_FRAME_WAIT_STOP_EVENT_RECHECK_S = 0.01

# See `_wait_for_next_frame()`'s own "SUSTAINED OVERRUN" section. Chosen
# structurally (5 consecutive dropped-generation iterations at this
# rig's real ~32ms/frame cadence is ~160ms of real, ongoing lag,
# comfortably past a single slow one-off iteration's own expected jitter
# while still escalating to a louder signal within a fraction of a
# second of genuine sustained overrun), not measured against real live
# cadence data -- no live-rig access for the session that built this.
#: How often the drop-to-latest warning may be emitted. Dropping is normal
#: during heavy per-frame work such as a calibration; one line per event
#: buried the output an operator was trying to read.
DROP_WARNING_INTERVAL_SECONDS = 10.0

SUSTAINED_FRAME_DROP_WARNING_ITERATIONS = 5


@dataclass
class FrameWakeState:
    """Loop-local bookkeeping for the frame-driven wake mechanism (see
    `_wait_for_next_frame()` below) -- threaded through
    `run_capture_loop_body()`'s main loop the SAME way every other
    piece of loop-local state already is (see `_last_seen_frame_count`/
    `_frame_stale_since`'s own comment for the identical "reassign the
    local every iteration, never mutated in place across an object
    boundary" convention).

    `last_frame_generation`: the `LocalCameraHub` frame-generation value
    (see that class's own `frame_generation()`/`wait_for_new_frame()`)
    this loop has actually consumed as of its last wake.

    `consecutive_drop_iterations`/`total_frames_dropped`: real drop-to-
    latest accounting -- see `_wait_for_next_frame()`'s own "SUSTAINED
    OVERRUN" section for the full reasoning. Both are cumulative for one
    `run_capture_loop_body()` CALL -- never reset on any state-machine
    transition (Reset, takeout-complete, etc.), since "is this loop
    keeping up with the camera pump" is a fact about loop THROUGHPUT,
    orthogonal to trigger state, matching the identical reasoning
    `_last_seen_frame_count`'s own comment already gives for the
    frame-freshness gate's state.
    """

    last_frame_generation: int = 0
    consecutive_drop_iterations: int = 0
    total_frames_dropped: int = 0

    # Rate-limiting for the drop warning, 2026-09-11. Dropping to the
    # latest frame is NORMAL and correct whenever an iteration's own work
    # outlasts a frame period -- most obviously during a calibration,
    # which does heavy per-frame vision work on purpose. Logging each one
    # produced a wall of warnings that buried the calibration output an
    # operator was actually trying to read.
    #
    # Rate-limited rather than silenced, and never suppressed by phase:
    # the count still has to surface, because the same message during
    # ordinary detection means something quite different. Collapsing to a
    # periodic summary keeps that signal while making the log readable.
    drops_since_last_warning: int = 0
    last_drop_warning_monotonic: float = 0.0


def _wait_for_next_frame(
    hub: "local_capture.LocalCameraHub | None",
    wake_state: FrameWakeState,
    stop_event: threading.Event,
    iteration_started: float,
    poll_interval_s: float,
    iteration: int,
) -> FrameWakeState:
    """Wake as soon as `hub`'s pump publishes a frame generation NEWER
    than this loop last consumed, instead of unconditionally sleeping a
    fixed `poll_interval_s` regardless of whether the camera pump has
    actually produced anything new -- see `LocalCameraHub`'s own module
    docstring ("FRAME-DRIVEN WAKE PRIMITIVE" section) for the hub-side
    counter/condition this calls, and Requirement 1's own real-math
    writeup (`throw_trigger.SETTLE_WINDOW_FRAMES`'s own dated comment)
    for why the settle-window FRAME COUNT changed to hold the real
    wall-clock settle guarantee constant under this faster wake rate.

    Returns a NEW `FrameWakeState` (never mutates the one passed in) --
    the caller reassigns its own loop-local `wake_state = _wait_for_
    next_frame(...)`, mirroring how `_last_seen_frame_count`/
    `_frame_stale_since` are already threaded through this same loop.

    FALLBACK, byte-identical to today's exact behavior: whenever the
    frame-generation primitive genuinely doesn't apply -- `hub` is
    `None`/doesn't
    implement `wait_for_new_frame()` (a bare duck-typed test double, or
    no hub at all) -- this calls the OLD, unchanged
    `_wait_for_next_iteration()` and returns `wake_state` untouched. Same
    `hasattr()`-degrade-safely convention this module's own
    frame-freshness gate already uses for the identical reason, not a
    new pattern.

    DROP-TO-LATEST, BY CONSTRUCTION, never an explicit skip: this
    function never discards a backlog of stale frame sets, because
    there is no backlog TO discard -- `LocalCameraHub` only ever caches
    the single most recent frame per camera (see that class's own
    `grab()`/`grab_all()` docstrings). If this loop's own previous
    iteration (fetch + advance() + a real capture/save/engine-score
    pass) took longer than one real pump cycle, the pump has already
    overwritten the cache with something newer by the time this
    function runs again -- `hub.wait_for_new_frame()` returns
    IMMEDIATELY (its own generation counter is already ahead of what
    was last consumed), and the very next `fetch_current_frames()` call
    in this loop reads whatever is newest. Nothing is ever queued.

    SUSTAINED OVERRUN, measured and logged, never silently absorbed --
    per this task's own explicit instruction that silently falling
    behind is the worst possible failure mode here, worse than an
    occasional visible drop. The one real quantity this function CAN
    measure honestly is HOW MANY pump cycles were skipped between the
    generation last consumed and the one just observed
    (`new_generation - last_frame_generation - 1`, floored at 0) -- a
    positive value means the PREVIOUS iteration took longer than one
    real camera frame period (~32ms on this rig) to complete. A single
    dropped generation logs at WARNING (real, but expected under
    ordinary jitter -- one slow save/engine-score iteration is a known,
    accepted cost here, not itself a bug).
    `SUSTAINED_FRAME_DROP_WARNING_ITERATIONS` CONSECUTIVE drop-carrying
    iterations escalate to a distinctly-worded, louder WARNING naming
    the overrun as SUSTAINED. Both counters live on the returned
    `FrameWakeState` for a caller (or a future live measurement) to
    read/expose directly, not just log lines that could be missed.

    Interrupt semantic -- a real, bounded, disclosed change from the OLD
    flat-timer wait, not a silent regression: each attempt bounds its
    own wait to `_FRAME_WAIT_STOP_EVENT_RECHECK_S` (~10ms, independent of
    `poll_interval_s` -- see that constant's own module-level comment)
    and re-checks `stop_event` between attempts. `stop_event.set()`
    called mid-wait is NOT observed instantly the way the old
    `Event`-based `stop_event.wait(poll_interval_s)` was (a
    `threading.Condition.wait()` only wakes on its OWN condition being
    notified, or its own timeout elapsing -- it cannot observe an
    unrelated `Event` becoming set) -- worst-case shutdown latency is
    therefore now bounded by `_FRAME_WAIT_STOP_EVENT_RECHECK_S`
    (~10ms) instead of being near-instant, a real, small, explicitly
    disclosed tradeoff for gaining frame-driven wakeups, not something
    hidden. This does NOT introduce any path where the loop can block
    INDEFINITELY -- every wait remains bounded by a real timeout
    regardless of whether a frame ever arrives or `hub` ever recovers,
    proven directly in `tests/test_capture_daemon_frame_driven_wake.py`.
    """
    if hub is None or not hasattr(hub, "wait_for_new_frame"):
        _wait_for_next_iteration(stop_event, iteration_started, poll_interval_s)
        return wake_state

    last_generation = wake_state.last_frame_generation
    new_generation = last_generation
    while not stop_event.is_set():
        new_generation = hub.wait_for_new_frame(last_generation, _FRAME_WAIT_STOP_EVENT_RECHECK_S)
        if new_generation != last_generation:
            break
        # Timed out this slice with no new frame published -- re-check
        # stop_event (the bounded-shutdown-latency contract above) and
        # try another slice.

    if new_generation == last_generation:
        # stop_event fired (or was already set) before any new frame
        # generation arrived -- nothing new to report, hand back the
        # state unchanged; the caller's own `while not stop_event.is_
        # set()` loop condition will exit on its own next check.
        return wake_state

    dropped = max(0, new_generation - last_generation - 1)
    if dropped <= 0:
        return FrameWakeState(
            last_frame_generation=new_generation,
            consecutive_drop_iterations=0,
            total_frames_dropped=wake_state.total_frames_dropped,
        )

    consecutive = wake_state.consecutive_drop_iterations + 1
    total_dropped = wake_state.total_frames_dropped + dropped

    pending = wake_state.drops_since_last_warning + dropped
    last_warned = wake_state.last_drop_warning_monotonic
    now_monotonic = time.monotonic()
    sustained = consecutive >= SUSTAINED_FRAME_DROP_WARNING_ITERATIONS
    # The sustained warning is NEVER rate-limited. It is already rare by
    # construction -- it needs N consecutive overrunning iterations -- and
    # it is the one that means the loop is persistently slower than the
    # pump rather than occasionally behind. Silencing it to tidy the log
    # would suppress exactly the message worth keeping.
    quiet = (not sustained) and (now_monotonic - last_warned) < DROP_WARNING_INTERVAL_SECONDS
    if quiet and last_warned:
        # Inside the quiet window: accumulate and stay silent. The counts
        # are carried forward, so nothing is lost -- only deferred.
        return FrameWakeState(
            last_frame_generation=new_generation,
            consecutive_drop_iterations=consecutive,
            total_frames_dropped=total_dropped,
            drops_since_last_warning=pending,
            last_drop_warning_monotonic=last_warned,
        )

    if sustained:
        log.warning(
            "iteration %d: SUSTAINED frame-processing overrun -- %d "
            "consecutive iteration(s) have each dropped >=1 pump cycle "
            "(this one dropped %d, %d total dropped this run) -- this "
            "loop is consistently slower than the camera pump's own "
            "cadence, not just occasionally lagging behind it "
            "(%d dropped since the last of these lines)",
            iteration, consecutive, dropped, total_dropped, pending,
        )
    else:
        # DEBUG, not WARNING. Measured on a real rig 2026-09-12: 19 drops
        # across ~11,000 iterations, every one of them a single cycle, and
        # every one during either throw scoring or a calibration. Neither
        # is a fault -- scoring a dart takes 94-172ms and a calibration
        # takes far longer, so both exceed a 33ms frame period by
        # arithmetic rather than by inefficiency, and skipping to the
        # newest frame is the CORRECT response in both cases.
        #
        # Warning about it trained the eye to ignore warnings, which costs
        # more than the message was ever worth. The escalation above is
        # the real signal and stays at WARNING: consecutive overrunning
        # iterations mean the loop is persistently slower than the pump,
        # which is a fault. This line remains for anyone actually digging,
        # at a level that does not interrupt.
        log.debug(
            "iteration %d: dropped %d pump cycle(s) (skipped straight to "
            "the latest frame instead of queuing, %d total dropped this "
            "run) -- this iteration's own prior work took longer than "
            "one real camera frame period; %d dropped since the last of "
            "these lines, which are rate-limited to one per %.0fs",
            iteration, dropped, total_dropped, pending,
            DROP_WARNING_INTERVAL_SECONDS,
        )
    return FrameWakeState(
        last_frame_generation=new_generation,
        consecutive_drop_iterations=consecutive,
        total_frames_dropped=total_dropped,
        drops_since_last_warning=0,
        last_drop_warning_monotonic=now_monotonic,
    )


def _build_lifecycle_driver(
    *,
    lifecycle_settings_store: "LifecycleSettingsStore | None" = None,
) -> "LifecycleDriver":
    """The one lifecycle (opendarts/lifecycle/) that drives this loop. There is
    no fallback trigger: if this cannot be built the loop must not start."""
    from opendarts.lifecycle.driver import LifecycleDriver
    from opendarts.live.logging_setup import DEFAULT_LOG_DIR

    kwargs: dict = {}
    if lifecycle_settings_store is not None:
        kwargs["config"] = lifecycle_settings_store.get()
        kwargs["config_provider"] = lifecycle_settings_store.get
    return LifecycleDriver(
        log_dir=DEFAULT_LOG_DIR,
        masks_provider=get_calibrated_board_disc_masks,
        save_commit_frames=False,
        **kwargs,
    )


class _FrameJpegIndex:
    """The camera's own JPEG bytes for the frames the loop fetched, keyed
    by ARRAY IDENTITY, so a package can stream-copy them into its clip.

    WHY IDENTITY AND NOT "THE LATEST BYTES". The hub overwrites its
    per-slot JPEG every pump cycle, ~30 times a second. The frames a
    package stores are older than that by the time the save runs: the
    commit frame is this tick's, but the bg is whatever the lifecycle
    last adopted -- often many ticks back. Asking the hub for "the JPEG"
    at save time would pair a scored array with a LATER frame's bytes,
    and the clip would then hold a frame nothing ever scored:
    SCORE==STORE broken, silently, in the one artifact meant to prove it.

    So the bytes are captured the moment the frame is fetched, through
    ``LocalCameraHub.grab_with_jpeg()`` -- which reads a slot's array and
    its bytes together under the hub's cache lock -- and are kept ONLY
    when the array that call returns ``is`` the array ``grab_all()``
    handed the loop. If the pump advanced between the two reads, the
    identity check fails and that frame simply has no bytes: the package
    falls back to an FFV1 encode of the array, which is bigger and
    equally exact. A missing pairing costs disk; a wrong one would cost
    the corpus its meaning.

    The array is held alongside its bytes, not just its id(): a freed
    array's id can be reused by the next allocation, and an id-only map
    would then happily hand a new frame an old frame's bytes.

    Retention is the last ``keep_ticks`` fetches (the commit frame is
    always the current one) plus whatever the lifecycle's reference
    currently holds (``pin()``), which is how a bg adopted many ticks ago
    still has its bytes on the tick that scores against it. Bounded by
    construction: a few ticks of ~100 KB JPEGs, plus one reference set.

    Every hub-less or JPEG-less path (tests' fake hubs, macOS without
    synthetic JPEG) records nothing and every lookup comes back empty --
    exactly the pre-existing FFV1 behaviour.

    LAZY FRAMES (detect_from_small_decode). A fetched LazyFrames already
    carries each frame's bytes and generation, read under the hub's lock
    with the frame, so they are recorded straight from it -- and keyed by
    the LazyFrame, since its pixels do not exist yet. A lookup by array (the
    decoded commit frames and bg) matches the LazyFrame that decoded to
    exactly that array. Nothing here ever decodes a frame.

    THE FRAME'S RING GENERATION RIDES ALONG (2026-09-26). A hub with
    ``grab_paired()`` also says which pump generation published the frame
    -- the frame ring set holding exactly it -- read under the same lock
    as the frame and its bytes, and kept by the same identity rule. That
    number is how a package's clip takes its frames out of the ring by
    NAME instead of searching the ring for pixels that match (see
    opendarts.capture.clip.write_window_clips). A frame whose pairing was
    not proven has no generation either, and its package simply gets the
    two-frame clip.
    """

    def __init__(self, keep_ticks: int = 3) -> None:
        # id(arr) -> (arr, its JPEG or None, its ring generation or None)
        self._ticks: "deque[dict[int, tuple[np.ndarray, bytes | None, int | None]]]" = (
            deque(maxlen=keep_ticks))
        self._pinned: dict[int, tuple[np.ndarray, "bytes | None", "int | None"]] = {}

    def record(self, hub: Any, frames: dict[int, np.ndarray]) -> None:
        entries: dict[int, tuple[np.ndarray, "bytes | None", "int | None"]] = {}
        if isinstance(frames, LazyFrames):
            for cam, handle in frames.handles().items():
                jpeg, generation = frames.jpeg(cam), frames.generation(cam)
                if jpeg is None and generation is None:
                    continue
                entries[id(handle)] = (
                    handle, bytes(jpeg) if jpeg is not None else None,
                    int(generation) if generation is not None else None)
            self._ticks.append(entries)
            return
        paired = getattr(hub, "grab_paired", None)
        grab = getattr(hub, "grab_with_jpeg", None)
        if paired is not None or grab is not None:
            for cam, arr in frames.items():
                try:
                    if paired is not None:
                        got, jpeg, generation = paired(cam)
                    else:
                        (got, jpeg), generation = grab(cam), None
                except Exception:  # noqa: BLE001 -- bytes are an optimisation, never a failure
                    continue
                if got is not arr or (jpeg is None and generation is None):
                    continue
                entries[id(arr)] = (
                    arr, bytes(jpeg) if jpeg is not None else None,
                    int(generation) if generation is not None else None)
        self._ticks.append(entries)

    def _find(self, arr: np.ndarray) -> "tuple[np.ndarray, bytes | None, int | None] | None":
        key = id(arr)
        for entries in (self._pinned, *reversed(self._ticks)):
            hit = entries.get(key)
            if hit is not None and hit[0] is arr:
                return hit
        # A decoded array whose LazyFrame was recorded (a handful of
        # entries, so a scan).
        for entries in (self._pinned, *reversed(self._ticks)):
            for hit in entries.values():
                if isinstance(hit[0], LazyFrame) and hit[0].is_pixels(arr):
                    return hit
        return None

    def lookup(self, frames: "dict[int, np.ndarray] | None") -> dict[int, bytes]:
        """``{cam: bytes}`` for every frame in `frames` whose own bytes are
        known; cameras without a proven pairing are simply absent."""
        out: dict[int, bytes] = {}
        for cam, arr in handles_of(frames).items():
            hit = self._find(arr) if arr is not None else None
            if hit is not None and hit[1] is not None:
                out[cam] = hit[1]
        return out

    def lookup_generations(self, frames: "dict[int, np.ndarray] | None") -> dict[int, int]:
        """``{cam: ring generation}`` for every frame in `frames` whose
        publishing generation is known -- the same identity rule as
        lookup(); cameras without one are simply absent."""
        out: dict[int, int] = {}
        for cam, arr in handles_of(frames).items():
            hit = self._find(arr) if arr is not None else None
            if hit is not None and hit[2] is not None:
                out[cam] = hit[2]
        return out

    def pin(self, *frame_sets: "dict[int, np.ndarray] | None") -> None:
        """Keep the bytes for these arrays (the lifecycle's current
        reference) alive past the tick window. Replaces the previous pin
        set, so a reference the lifecycle has let go of is let go of here
        too."""
        pinned: dict[int, tuple[np.ndarray, "bytes | None", "int | None"]] = {}
        for frames in frame_sets:
            for arr in handles_of(frames).values():
                if arr is None:
                    continue
                hit = self._find(arr)
                if hit is not None:
                    pinned[id(hit[0])] = hit
        self._pinned = pinned


def _lifecycle_step(
    lifecycle: "LifecycleDriver",
    adapter: "LifecycleTriggerAdapter",
    current_frames: dict[int, np.ndarray],
    bg_frames: dict[int, np.ndarray],
    dropped_frames_total: int,
) -> "LiveStep | None":
    """One tick: observe the frames, translate the Tick into the trigger
    state the loop consumes. ``None`` means nothing could be judged (no
    calibrated board masks yet). This is the loop's single decision seam
    -- tests script the trigger by replacing it (``bg_frames`` is passed
    only so a scripted step can hand it back as the reference)."""
    tick = lifecycle.observe(current_frames, dropped_frames_total=dropped_frames_total)
    if tick is None:
        return None
    return adapter.apply(tick, lifecycle.lifecycle, current_frames)


def _calibration_key(calibration_store: Any, calibrations: Any) -> tuple:
    """Cheap identity of the calibration in force: which calibration objects
    the store holds, plus its package id. A recalibration replaces the
    objects, so this changes without hashing any arrays -- it runs every
    tick. Without a store (tests), the loop's own startup calibrations."""
    if calibration_store is None:
        return tuple(sorted((c, id(v)) for c, v in (calibrations or {}).items()))
    cals, package_id = calibration_store.get_with_package_id()
    return (package_id, tuple(sorted((c, id(v)) for c, v in cals.items())))


def _lifecycle_phase_is_idle(lifecycle: Any) -> bool:
    """Whether the lifecycle has settled back into IDLE -- past a clear's
    cooldown or a Reset's warmup, so its reference is a quiet empty
    board. A driver without a live Lifecycle (tests' scripted seam) has
    no phases to wait out and counts as settled."""
    from opendarts.lifecycle.state import Phase

    phase = getattr(getattr(lifecycle, "lifecycle", None), "phase", None)
    return phase is None or phase is Phase.IDLE


def run_capture_loop_body(
    *,
    hub: local_capture.LocalCameraHub | None,
    package_root: Path,
    poll_interval_s: float,
    stop_event: threading.Event,
    scratch_dir: Path = SCRATCH_DIR,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    ad_ws_listener: "AdWsListener | None" = None,
    ad_match_window_sec: float = DEFAULT_MATCH_WINDOW_SEC,
    calibration_store: "CalibrationStore | None" = None,
    reset_request: "ResetRequest | None" = None,
    also_stop: threading.Event | None = None,
    engine_config_store: "EngineConfigStore | None" = None,
    lifecycle_settings_store: "LifecycleSettingsStore | None" = None,
    reprojection_targets_px: dict[int, float] | None = None,
    # background_save: threaded straight through to every real
    # handle_ready_to_capture() call below -- see that function's own
    # docstring for the full design. `True` (the default, every real
    # live caller) is the new 2026-09-01 behavior; `False` restores this
    # loop's own pre-2026-09-01 fully-synchronous save timing, for a
    # test that genuinely needs deterministic on-disk state immediately
    # after this function returns rather than polling for it.
    background_save: bool = True,
    store_packages: bool = True,
    # The free-space floor, in GB, below which a throw package is
    # SKIPPED rather than written -- None means the code default (see
    # opendarts.disk_space.DEFAULT_MIN_FREE_DISK_GB, 5 GB), 0 means the
    # same, and a NEGATIVE value disables the guard. Unlike most optional
    # parameters here, None is NOT "no guard": a rig whose disk is full
    # stops being able to write its own log, so the safe direction is the
    # guard being on for every caller that has not thought about it.
    # `opendarts.live.run_product` passes the configured value
    # (`min_free_disk_gb` in data/config.json).
    min_free_disk_gb: "float | None" = None,
    # throw_capture: the ring-and-writer service, built ONCE by
    # opendarts.live.run_product beside the hub and shared BY REFERENCE
    # with the web server, so the ring a dashboard button dumps from is
    # the ring this loop's own pump is filling. Threaded straight through
    # to handle_ready_to_capture(); None (every existing caller and test)
    # simply means no automatic capture fires.
    throw_capture: Any = None,
) -> None:
    """The actual always-on loop body -- extracted from run_capture_loop()
    below so BOTH that function (standalone `-m opendarts.live.capture_daemon`,
    which opens/owns/closes its own hub and its own stop_event tied to
    signal handlers) and opendarts/live/run_product.py's combined entrypoint
    (which opens ONE hub shared with the web server, and ties stop_event
    to signal handlers registered in run_product's own main thread, since
    this function may run in a background thread there -- Python only
    allows signal.signal() from the main thread) can run the IDENTICAL
    capture logic. This function never opens, closes, or owns `hub` --
    that is always the caller's responsibility; it only uses it. This
    also means this function does NOT register any signal handlers
    itself -- the caller must arrange for `stop_event` to get set
    (directly, or from a signal handler in whichever thread is actually
    allowed to install one).

    Runs a full turn (up to MAX_DARTS_PER_TURN captured darts, then
    TAKEOUT_WAITING, then back to a fresh IDLE once the board genuinely
    returns to the true empty-board baseline) and then keeps going,
    indefinitely, turn after turn, until stop_event is set -- see
    opendarts/capture/throw_trigger.py's module docstring for the real
    state-machine mechanics (dart_count, true_baseline_frames,
    TAKEOUT_WAITING, the takeout-vs-new-dart classifier). Does not raise
    NotImplementedError anymore -- both opendarts.capture.throw_trigger.
    advance() and this file's own refresh_background_after_capture() are
    real now.

    Stopping: checks `stop_event.is_set()` at the top of every loop
    iteration and sleeps via `stop_event.wait(poll_interval_s)` (which
    returns immediately once the event is set, instead of a plain
    `time.sleep()` that would have to fully elapse first) -- so a caller
    setting stop_event sees this loop exit within one in-flight
    fetch/advance/save cycle, not up to a full poll_interval_s late on
    top of that.

    on_event: optional callback invoked with a small dict describing REAL
    state changes as they happen -- `{"type": "TRIGGER_STATE", "state":
    ..., "session": ..., "dart_count": ...}` on every trigger state
    transition (dart_count added 2026-08-12 so the dashboard's status
    pill can show "dart 2 of 3" instead of just the bare state name --
    see opendarts/live/server.py's AppState._handle_live_event), and

    `emitted_at_utc` (added 2026-09-01, latency-instrumentation task):
    a real `datetime.now(timezone.utc).isoformat()` stamped HERE, at the
    exact moment this thread decided the transition -- purely additive,
    pure instrumentation, zero effect on any scoring/detection behavior.
    Added because AppState._handle_live_event()'s own `ts` field (what
    external consumers, e.g. a QA latency harness, were actually reading)
    is stamped at DEQUEUE time on the asyncio event loop -- after this
    event crosses a thread-safe queue, an `asyncio.to_thread()` dispatch,
    and however many prior events this same live_event_loop is still
    working through one at a time (see that loop's own docstring: events
    are handled strictly serially, one full round trip -- including the
    websocket broadcast -- before the next is even dequeued). A single
    dart genuinely produces at least 3 separate TRIGGER_STATE pushes in a
    ~150ms window (IDLE->MOTION_DETECTED, MOTION_DETECTED->SETTLING,
    SETTLING->READY_TO_CAPTURE -- see `still_filling` in throw_trigger.py,
    a real internal state flip, not merely cosmetic), each paying that
    same round trip -- real, measured evidence (this rig's own
    per-camera settle-offset log, `throw_trigger.py`'s own "MOTION_
    DETECTED/SETTLING -> READY_TO_CAPTURE after Xs" line) shows the
    SETTLE WINDOW itself resolves in ~140-150ms median, well under a
    peer QA harness's independently-measured ~278ms for that same phase
    using `ts` -- i.e. the gap lives in event-delivery latency, not
    detection logic. `emitted_at_utc` lets a consumer measure the real
    source-to-source interval directly and compute `ts - emitted_at_utc`
    as the actual dispatch/queue/broadcast overhead, instead of
    conflating it with settle time. Present on every TRIGGER_STATE
    event, all 4 real emit call sites; `_handle_live_event()` passes it
    through verbatim in the broadcast payload alongside `ts` (both kept
    -- this is additive, not a replacement).
    `{"type": "PACKAGE_SAVED", "path": ..., "session": ...}` right after
    `{"type": "PACKAGE_SAVED", "path": ..., "session": ...}` right after
    a throw package is written. Standalone run_capture_loop() passes
    None (nothing to push to -- no web server exists in that process to
    receive it). See _emit() above for the never-breaks-the-loop
    delivery contract.

    Two more event types were ADDED 2026-08-14 (nothing existing was
    changed or removed -- see docs/LIVE_API.md):
      - `THROW_DETECTED` -- the whole scored throw in one payload,
        emitted by handle_ready_to_capture() itself the moment the
        primary engine's result is durably saved. See that function.
      - `VISIT_CLEARED` -- `{"type", "session", "visit_id" (the NEW one),
        "previous_visit_id", "n_darts" (how many were captured in the
        visit that just ended), "reason": "takeout"|"reset"}`, emitted
        at the two real visit rotation points below.
    `TRIGGER_STATE` additionally carries `visit_id` now -- an added key
    on an existing message, no existing key touched.

    ad_ws_listener/ad_match_window_sec: passed straight through to
    handle_ready_to_capture() on every READY_TO_CAPTURE -- see that
    function's own docstring and this module's "AD GROUND TRUTH" section
    above. This function never opens/closes/starts/stops the listener
    itself (same caller-owns-it discipline as `hub`); `None` (the
    default) means AD ground-truth inline capture is off for this run.

    calibration_store: the shared, mutable CalibrationStore this loop
    reads from on EVERY READY_TO_CAPTURE (not a one-time local variable
    captured at startup and never revisited -- see that class's own
    docstring for why this changed, 2026-08-12). This function bootstraps
    calibration once here at startup either way; the difference is what
    happens to the result:
      - `None` (the default -- opendarts/live/capture_daemon.py's own
        standalone `run_capture_loop()`, no dashboard sharing this
        process): a private CalibrationStore is created here, seeded with
        the startup bootstrap, and used for the rest of this run. Nothing
        else can ever call `.set()` on it since nothing else has a
        reference -- functionally identical to the old "plain local
        variable, never updated" behavior, just routed through the same
        read path as the shared case below (one code path, not two).
      - a real CalibrationStore (opendarts/live/run_product.py's combined
        entrypoint, built and shared with opendarts.live.server.AppState
        BEFORE this function is called, typically on a background thread)
        -- seeded here with the startup bootstrap too, but a concurrent
        manual "Refresh calibration now" dashboard action can call
        `.set()` on the SAME object from a different thread at any later
        point; the very next READY_TO_CAPTURE after that call scores
        against the new calibration, not the startup one.

    reset_request: the shared, mutable ResetRequest (see that class's own
    docstring for the full thread-safety reasoning -- same established
    pattern as calibration_store above, deliberately not a new mechanism)
    a manual dashboard "Reset" click (POST /api/reset) signals through.
    Checked at the TOP of every loop iteration, before `advance()` is
    even called (unlike calibration_store, which is only consulted at
    READY_TO_CAPTURE -- a reset must be actionable no matter what state
    the trigger currently sits in, mid-throw or mid-takeout). When a
    request is pending: takes the CURRENT frames (fetched this same
    iteration) as the new `true_baseline_frames` AND the new `bg_frames`,
    unconditionally -- no validation of whether a dart happens to be
    visible: lock-clean semantics (see
    opendarts/capture/throw_trigger.py's module docstring "STALE BASELINE
    NEVER REFRESHED BUG" for the fuller writeup of that same semantics,
    applied there to the AUTOMATIC post-takeout refresh; this is its
    manual, operator-triggered equivalent) -- and resets `dart_count` to 0 and
    `state` to IDLE, discarding whatever turn/takeout progress was
    in-flight. `None` (the default) means no reset signal exists for this
    run -- functionally identical to today's behavior before this feature
    existed.

    also_stop: added 2026-08-12 alongside CaptureLoopController. An
    OPTIONAL second stop condition, checked alongside `stop_event` in the
    main loop's own while condition: `while not stop_event.is_set() and
    not (also_stop is not None and also_stop.is_set())`. `stop_event`
    still means "the whole process is shutting down" (unchanged); `also_
    stop`, when given, means "just THIS session should end" (a manual
    Stop click or an idle-timeout) -- the caller (opendarts/live/
    run_product.py's `_capture_thread_target`) is expected to loop back
    and call this function AGAIN for the next session once a new Start
    arrives, rather than treating a `also_stop`-triggered return as the
    whole capture thread exiting. `None` (the default) means this
    function behaves EXACTLY as before this parameter existed -- every
    existing caller (capture_daemon.py's own standalone
    `run_capture_loop()`, every existing test) passes nothing and is
    unaffected.

    Calibration bootstrap, CHANGED 2026-08-12 (see this same commit's
    Start/Stop work): if `calibration_store` is given AND already holds
    at least one camera's calibration (i.e. a PRIOR session this same
    process lifetime already bootstrapped it, or a manual "Refresh
    calibration now" already ran), this function now REUSES it and skips
    calling `bootstrap_calibrations()` again -- a deliberate design decision:
    calibrate automatically after a Start, with manual recalibration still
    available. A `None` or genuinely EMPTY
    calibration_store (every existing call site/test before this change,
    and any first-ever session in a fresh process) is unaffected -- still
    bootstraps exactly as before. The manual "Refresh calibration now"
    button/`POST /api/calibration/refresh` (opendarts/live/server.py) is
    completely unaffected either way -- it writes directly to the SAME
    CalibrationStore this function reads from, any time, regardless of
    whether a session is currently running.

    EXTENDED 2026-08-26 --
    `calibration_store` can now ALSO already hold at least one camera's
    calibration on the very FIRST Start of a brand-new process, when it
    was constructed with `snapshot_path` and a valid persisted calibration
    existed on disk (see `CalibrationStore`'s own docstring). This
    function needs NO code change to honor that -- the existing "already
    holds at least one camera's calibration" check above already covers
    it, since a persisted load populates the store before this function
    ever runs. The one honest caveat, not papered over: this reuse is now
    "once ever, until a human clicks Refresh," not merely "once per
    process lifetime" -- a camera physically re-seated to a different USB
    port between one process's shutdown and the next Start will NOT be
    caught automatically (see CalibrationStore's own docstring for the
    accepted trade-off and its dashboard-visibility mitigation).

    reprojection_targets_px (added 2026-08-17): passed straight through
    to bootstrap_calibrations()'s own `target_reprojection_error_px` --
    see that function's own docstring. `None` or an empty dict (the
    default) means every camera uses the module's uniform
    CALIBRATION_TARGET_REPROJECTION_ERROR_PX, exactly as before this
    parameter existed. Real source: opendarts.live.config.LiveConfig, loaded
    once at the CLI entrypoint (opendarts/live/run_product.py's main()) --
    never read from disk in here.

    Structure (this function's real contribution, same "structure is
    real, math is not" pattern throw_trigger.py itself already uses):
      1. Bootstrap calibration once per SESSION -- reusing an existing
         CalibrationStore's contents when one is already valid (real,
         see the "Calibration bootstrap" section just above).
      2. Seed an initial background frame set (real).
      3. Loop: fetch current frames (real) -> advance the trigger (real,
         includes the turn/takeout logic) -> on READY_TO_CAPTURE, score +
         save (real) + refresh background (real) + reset to either a
         fresh IDLE (dart_count < MAX_DARTS_PER_TURN) or TAKEOUT_WAITING
         (dart_count == MAX_DARTS_PER_TURN) -> on a detected takeout
         completing, restore the true empty-board baseline as the
         background -> wait for stop_event/poll_interval_s -> repeat,
         until stop_event is set.
    Hub lifecycle (open before calling, close after -- see above) is
    deliberately NOT step 0/step-last here; that's the caller's job.
    """
    # Multi-engine scoring config (docs/ENGINES.md) -- same "own instance
    # if the caller didn't share one" pattern as calibration_store's own
    # standalone-CLI case immediately below: a private default-config
    # EngineConfigStore (Apollo primary, no also-run engines -- byte-
    # identical to pre-framework behavior) so this function's OWN read
    # path (handle_ready_to_capture(engine_config_store=...)) never needs
    # a None-check special case.
    if engine_config_store is None:
        engine_config_store = EngineConfigStore()

    # CHANGED 2026-08-12 -- see this function's own docstring's
    # "Calibration bootstrap" section: reuse an already-populated
    # calibration_store instead of unconditionally re-bootstrapping, so a
    # SECOND (or later) Start this same process lifetime doesn't
    # needlessly recalibrate when a valid calibration already exists.
    #
    # EXTENDED 2026-08-26 -- this branch now ALSO fires on the very FIRST
    # Start of a brand-new process, whenever CalibrationStore was
    # constructed with `snapshot_path` and a valid persisted calibration
    # existed on disk (see that class's own docstring's "Durable across
    # process restarts" section) -- `calibration_store.get()` is non-empty
    # in that case exactly as it would be for a real earlier-this-process
    # bootstrap, so this one check correctly covers both "reused within
    # one process's lifetime" and "reused across a restart" with no
    # separate branch needed. `calibration_store.meta()["source"]` (not
    # this log line) is the honest way to tell the two apart after the
    # fact -- `"startup"`/`"manual"` for a real bootstrap this process
    # lifetime, `"persisted"` for a snapshot loaded at construction.
    if calibration_store is not None and calibration_store.get():
        calibrations = calibration_store.get()
        log.info(
            "startup calibration: reusing existing CalibrationStore contents "
            "(%d camera(s), source=%s) -- skipping auto-recalibration (a valid "
            "calibration already exists, either from an earlier Start/manual "
            "Calibrate this process lifetime, or persisted from a PRIOR process "
            "on disk; manual Calibrate remains available any time regardless)",
            len(calibrations), calibration_store.meta().get("source"),
        )
        # Real, visible confirmation of "did Start's auto-calibrate step
        # run" -- added 2026-08-12. Piggybacks on the
        # SAME CALIBRATION_STATUS message shape the dashboard's existing
        # renderCalibration()/manual-Calibrate button already understand
        # (opendarts/live/server.py) -- not a second, separate UI. `source`
        # distinguishes this (asynchronous, no direct HTTP response to
        # attach it to -- Start's own POST /api/start response returns
        # before this bootstrap section even runs, see CaptureLoopController's
        # ARCHITECTURE NOTE 1) from a manual refresh's own broadcast,
        # which omits `source` entirely (opendarts/live/server.py's
        # api_calibration_refresh route) -- the dashboard only logs an
        # explicit "auto-calibrate" action-log line when `source` is
        # present, so a manual click never double-logs.
        _emit(
            on_event,
            {
                "type": "CALIBRATION_STATUS",
                "source": "startup_reused",
                "calibrations": calibrations,
            },
        )
    else:
        # target_reprojection_error_px is only passed at all when a real
        # per-camera override exists -- omitted (letting
        # bootstrap_calibrations() use its own default) otherwise, so a
        # test double standing in for bootstrap_calibrations() with a
        # narrower signature (no target_reprojection_error_px param at
        # all) is unaffected by this parameter's mere existence, same
        # "don't surprise a caller that never asked for this" posture as
        # every other optional kwarg in this function.
        extra_kwargs: dict[str, Any] = (
            {"target_reprojection_error_px": reprojection_targets_px}
            if reprojection_targets_px
            else {}
        )
        # Calibration-package saving (opendarts.capture.calibration_package,
        # see bootstrap_calibrations()'s own "CALIBRATION PACKAGE"
        # docstring section) -- opted in for BOTH real call sites of
        # bootstrap_calibrations() in this app (this one and
        # opendarts.live.server.AppState._refresh_calibration_blocking's
        # manual refresh), never for a bare/standalone test call that
        # doesn't pass these. throw_package_root=package_root so the
        # background save's own post-save cleanup pass
        # (cleanup_orphaned_calibration_packages()) can see which
        # calibration packages this session's throws actually reference.
        calibration_package_out: dict[str, Any] = {}
        # _bootstrap_calibrations_at_start_with_reopen_retry(), not
        # _bootstrap_calibrations_with_lock_wait() directly -- this call
        # site has no graceful-degrade handling of its own (any exception
        # here is fatal to the whole process, see CalibrationInProgressError's
        # own class docstring), so a benign, transient lock race (a manual
        # "Refresh calibration now" click winning the race against this
        # Start) gets a bounded retry instead of crashing the process
        # over a UI click's bad timing -- handled by the wrapped function,
        # unchanged. 2026-09-01 (see the wrapper's own
        # docstring for the full real incident + design): ALSO retries
        # once, with a real camera close+reopen in between, if a camera
        # comes back missing from the result or the first attempt raises
        # -- scoped to exactly this Start-time call site, never the
        # manual mid-session refresh's own separate call site below.
        calibrations = _bootstrap_calibrations_at_start_with_reopen_retry(
            scratch_dir / "calib",
            hub=hub,
            calibration_package_root=DEFAULT_CALIBRATION_PACKAGE_ROOT,
            throw_package_root=package_root,
            calibration_package_out=calibration_package_out,
            **extra_kwargs,
        )
        if not calibrations:
            raise RuntimeError(
                "no camera calibrated at startup -- cannot run the capture "
                "loop with zero known camera poses. "
                + "Check that the local cameras are in view of the rig "
                "(see the hub status log line above)."
            )
        log.info("startup calibration ok for cameras: %s", sorted(calibrations))
        startup_checked_at_utc = datetime.now(timezone.utc).isoformat()
        startup_package_id = calibration_package_out.get("package_id")
        if calibration_store is None:
            # Standalone run -- see this function's own docstring's
            # `calibration_store` section for why a private one is created
            # here rather than leaving `calibrations` a bare local variable.
            calibration_store = CalibrationStore(
                calibrations,
                source="startup",
                checked_at_utc=startup_checked_at_utc,
                package_id=startup_package_id,
            )
        else:
            calibration_store.set(
                calibrations,
                source="startup",
                checked_at_utc=startup_checked_at_utc,
                package_id=startup_package_id,
            )
        # Same real confirmation as the "reused" branch above -- this is
        # the "it actually ran" case, not the "skipped, already valid"
        # one. See that branch's own comment for the full mechanism.
        _emit(
            on_event,
            {
                "type": "CALIBRATION_STATUS",
                "source": "startup",
                "calibrations": calibrations,
                # Present when this calibration found the cameras had moved
                # on the ring and relearned the rig's layout -- the operator
                # is told on the dashboard, not only in the log.
                "ring_geometry_relearned": calibration_package_out.get(
                    "ring_geometry_relearned"),
            },
        )

    # First frames only size the buffers; the lifecycle's WARMUP phase
    # decides when the cameras have settled enough to adopt a reference.
    bg_frames = _wait_for_first_frames(
        lambda: fetch_current_frames(
            scratch_dir / "bg", hub=hub
        ),
        stop_event=stop_event,
        poll_interval_s=poll_interval_s,
        label="startup background",
    )
    # A board photo straight away, from the first frames after Start, so
    # the Scoring tab's Photo view has a board before anyone has thrown.
    # Each takeout replaces it with the freshly cleared board (see
    # _board_photo_due below); if there are darts in the board at Start,
    # they are in this one only until then.
    if on_event is not None and calibrations:
        board_photo.RENDERER.submit(
            bg_frames, calibrations,
            lambda jpeg: _emit(on_event, {"type": "BOARD_PHOTO", "jpeg": jpeg, "visit_id": None}),
        )
    session_id = time.strftime("%Y%m%d-%H%M%S")
    # `trigger` is rebuilt by the lifecycle adapter every tick; this is
    # only the pre-first-tick value the UI/heartbeat see. `bg_frames` is
    # the scoring-side buffer handle_ready_to_capture() receives -- on a
    # commit tick the adapter sets it to the lifecycle's pre-dart
    # reference (the board right before this dart).
    trigger = ThrowTriggerState(true_baseline_frames=bg_frames)
    # The camera JPEG for each fetched frame, paired by array identity at
    # fetch time, so a package can stream-copy the frames it stores
    # instead of FFV1-encoding them. See _FrameJpegIndex for why the
    # pairing has to happen HERE, at the tick, and nowhere later.
    frame_jpegs = _FrameJpegIndex()
    # THE VISIT MODEL (2026-08-14, docs/LIVE_API.md). A visit is one turn:
    # up to MAX_DARTS_PER_TURN darts, then the board is cleared. It is a
    # thin wrapper around the turn bookkeeping this state machine ALREADY
    # tracks (`ThrowTriggerState.dart_count`, TAKEOUT_WAITING, the
    # true-baseline comparison) -- deliberately NOT a second, parallel
    # state machine that could disagree with the first. The one and only
    # rotation point is the real takeout-complete transition detected
    # below (plus a manual Reset, which explicitly means "abandon this
    # visit"), so a visit ID can never rotate at a moment the trigger
    # state machine doesn't also consider the turn over.
    visit_id = new_visit_id()
    _emit(
        on_event,
        {
            "type": "TRIGGER_STATE",
            "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
            "state": trigger.state.name,
            "session": session_id,
            "dart_count": trigger.dart_count,
            "visit_id": visit_id,
        },
    )
    log.info("trigger state: %s (session %s)", trigger.state.name, session_id)

    # THE BOARD PHOTO AFTER A TAKEOUT (2026-09-26). Set by a takeout or a
    # manual Reset; the photo is then taken on the first tick the
    # lifecycle is back in IDLE -- the end of its post-clear cooldown
    # (clear_cooldown_frames, ~0.3 s), or of the warmup a Reset restarts.
    # The CLEARED tick itself already adopted a fresh reference, but it is
    # the very frame the takeout was accepted on, with the arm possibly
    # still retreating; the cooldown re-adopts every tick precisely to
    # absorb that, so its last reference is the cleaner empty board. The
    # renderer skips the photo when that board matches the last one.
    _board_photo_due = False
    # A recalibration mid-session (the dashboard's Calibrate) also makes the
    # photo due: it is warped with the calibration, so after one the photo on
    # screen no longer matches where the darts are drawn. Scoring re-reads
    # the store on every dart; the photo reads it the same way, and a change
    # of calibration objects (or package id) since the last photo marks it
    # due. The renderer never skips across a calibration change.
    _photo_calibration_key = _calibration_key(calibration_store, calibrations)

    # Heartbeat -- found live 2026-08-12: state transitions were only
    # ever pushed to the dashboard's WebSocket (_emit), never logged, so
    # a run that never leaves IDLE (real threat: detect_motion()'s
    # thresholds were measured against a static case-data corpus, not
    # THIS rig's real live camera noise/lighting -- they may simply not
    # be right here) was indistinguishable in the log from a silently
    # hung loop. This periodic line proves the loop is alive and says
    # what it currently sees, independent of whether the state ever
    # changes. Cadence: module-level _HEARTBEAT_EVERY_S (see its own
    # comment).
    last_heartbeat = time.monotonic()

    # cached_prior_frames (2026-09-01, "why call anything off disk at
    # all when we have the prior dart in memory" finding -- see
    # CachedPriorThrowFrames's own docstring): holds the MOST RECENTLY
    # CAPTURED throw's own bg_images/current_frames across loop
    # iterations, so the NEXT throw's own find_prior_dart_line_px()
    # calls (both real call sites inside handle_ready_to_capture()) can
    # skip their disk round-trip when this throw turns out to be the
    # immediately-prior one of the same visit. Deliberately never
    # explicitly cleared at a turn boundary -- find_prior_dart_line_px()
    # itself validates visit_id/visit_index match before trusting it, so
    # a stale entry from the end of a PRIOR turn simply fails that check
    # and falls through to the disk path exactly like a fresh `None`
    # would, never risking a wrong-throw's frames being used silently.
    cached_prior_frames: "CachedPriorThrowFrames | None" = None

    # FRAME-FRESHNESS GATE state (see MAX_FRAME_FRESHNESS_WAIT_S's own
    # comment for the full incident/mechanism writeup) -- per-camera, and
    # deliberately NOT reset on any state-machine transition (Reset,
    # takeout-complete, etc.): frame novelty is a fact about the camera
    # pump, orthogonal to trigger state, so there's no state-machine
    # moment where "forget what we've already seen" would be correct.
    # _last_seen_frame_count tracks the raw frame_count observed on the
    # PREVIOUS iteration for each camera, updated unconditionally every
    # iteration (whether that iteration's frames were accepted or
    # skipped) -- this is what lets a camera that's been stuck past the
    # ceiling and accepted-anyway immediately resume being trusted the
    # moment it produces a genuinely new frame, with no lingering
    # penalty. _frame_stale_since tracks, per camera, the monotonic
    # instant its frame_count was FIRST observed unchanged from the prior
    # iteration -- cleared the moment that camera advances again.
    _last_seen_frame_count: dict[int, int] = {}
    _frame_stale_since: dict[int, float] = {}

    # STALL-WARNING THROTTLE, 2026-09-13 (CPU task). The "giving up past
    # the ceiling" WARNING below fires on the stalled-camera path, which
    # is re-entered EVERY iteration for as long as the stall lasts -- so
    # it emitted one formatted line plus a disk write per iteration.
    # Measured live on the Windows rig: 499 lines in 10.66s (154KB of log
    # for half a minute of one stall). The condition is worth reporting;
    # reporting it 47 times a second is not. These two track the last
    # instant a line was emitted and how many iterations were suppressed
    # since, so the next line can say what it stood in for rather than
    # silently dropping the count.
    _stall_warned_at: float | None = None
    _stall_warns_suppressed = 0

    # FRAME-DRIVEN WAKE state, 2026-09-05 -- see FrameWakeState's own
    # docstring and _wait_for_next_frame()'s own docstring for the full
    # design. Seeded from the hub's OWN current generation (not 0)
    # before this loop's first wait -- the pump has been running since
    # open_all() returned, potentially for a real, non-trivial number of
    # cycles already, and seeding at 0 would make the very first wait
    # report a spurious, enormous "dropped" count for cycles this loop
    # was never actually supposed to have consumed (there was no prior
    # iteration for them to have been dropped FROM). `hasattr()`-guarded
    # the same way every other consumer of this primitive is, so a bare
    # test-double hub (or a caller passing `hub=None`) degrades safely
    # to the harmless default rather than raising.
    _initial_frame_generation = 0
    if hub is not None and hasattr(hub, "frame_generation"):
        _initial_frame_generation = hub.frame_generation()
    wake_state = FrameWakeState(last_frame_generation=_initial_frame_generation)

    # REAL PER-ITERATION CADENCE, 2026-09-01 -- the project's own instrumentation
    # request. `poll_interval_s`/POLL_INTERVAL_SECONDS is the CONFIGURED
    # sleep between iterations -- it says nothing about how much real
    # wall-clock time elapses between two consecutive current_frames
    # samples, since fetch/gate/advance()/motion-detection CV work all
    # add real time on top of that sleep, every iteration. This matters
    # directly for the "why does opendarts settle in one poll on a dart
    # an identical implementation needed 3-4 polls for" investigation: if opendarts's
    # REAL sampling cadence during an active settle episode runs
    # meaningfully slower than the nominal 50ms, its "first post-seed
    # frame" is sampled later into the dart's real physical flight than a
    # naive 50ms model would suggest -- catching more of an already-
    # mostly-stopped tail, not a duplicate/stale read (that theory is
    # dead, see MAX_FRAME_FRESHNESS_WAIT_S's own comment history) and not
    # a logic bug -- a genuinely different, testable, systematic
    # explanation. Tracked as a plain local (not persisted state) --
    # deliberately reset to None on every IDLE->MOTION_DETECTED entry (see
    # below) so a value never bleeds from one episode into an unrelated
    # one; logged only while a settle episode is actually in progress
    # (MOTION_DETECTED/SETTLING), where it's the number that matters, not
    # every idle iteration where it would just be noise.
    #
    # TWO REAL, UNFIXED GAPS IN THIS INSTRUMENT, found 2026-09-04 while
    # building the PER-ITERATION COST BREAKDOWN below to reconcile a real
    # measured contradiction against this line's own output -- flagged
    # here, NOT fixed (diagnostics-only task; see docs/DESIGN.md's 2026-09-04
    # entry for the full reconciliation and the open decision on
    # whether/how to fix):
    # 1. UNGATED. This `log.debug(...)` call was never wired into
    # opendarts.live.diagnostics_gate (the gate task, commit 2844a1e,
    # landed after this instrumentation and did not touch this
    # specific line) -- like the per-camera settle-status line this
    # project's own diagnostics_gate module docstring already
    # documents finding, `logger.isEnabledFor(logging.DEBUG)` is
    # TRUE unconditionally in real production (the root logger stays
    # at DEBUG for the file handler), so this computation+log call
    # fires on every settle-loop iteration regardless of the gate's
    # state today.
    # 2. STRUCTURALLY BIASED SAMPLE. The check above
    # (`if trigger.state in (MOTION_DETECTED, SETTLING):`) reads
    # `trigger.state` as of the END of the PREVIOUS iteration (this
    # is the very first line of the loop body, before advance() runs
    # again) -- so a gap is only ever logged for iteration K when
    # BOTH iteration K-1 ended in a settle state (state_at_entry(K)
    # settle-eligible) AND iteration K-1 ITSELF also ended in a
    # settle state (so `_last_settle_iteration_started` was actually
    # set entering K). That means this line can NEVER report the
    # cost of the iteration that ENTERS a settle episode (its own
    # entry state was IDLE, not settle-eligible) NOR the cost of the
    # iteration that EXITS one into READY_TO_CAPTURE/TAKEOUT_WAITING/
    # IDLE (the NEXT iteration's own entry check fails, since state
    # already moved on, so no gap is ever computed for THAT
    # iteration's real cost either) -- only "interior, still-waiting,
    # nothing-happened" iterations ever contribute a sample. For the
    # real, common case of a settle episode spanning just 2 total
    # loop iterations (entry + the very next iteration reaching
    # READY_TO_CAPTURE, no interior iteration exists at all), this
    # line NEVER LOGS ANYTHING for that transition -- its whole real
    # cost is invisible here, no matter how expensive. This is the
    # real, confirmed mechanism behind the 71ms-vs-214ms
    # contradiction this same-day investigation was asked to
    # reconcile: this line's own population systematically excludes
    # the two most expensive iterations of every settle episode (the
    # decision-making entry and exit points), so its own median
    # necessarily undersells real settle-path iteration cost.
    _last_settle_iteration_started: float | None = None

    # PER-ITERATION COST BREAKDOWN, 2026-09-04 -- explicitly authorized
    # measurement instrumentation, built specifically to measure every settle-path
    # iteration's real cost DIRECTLY and COMPLETELY -- see the two gaps
    # documented on `_last_settle_iteration_started` immediately above
    # for why that pre-existing instrument cannot be trusted for this.
    # `_prev_iter_window_end` is this instrument's own self-consistency
    # anchor: the real monotonic() timestamp marking the end of the
    # PREVIOUS iteration's own measured `body + sleep` window, carried
    # forward so the NEXT iteration's own `iteration_started` can be
    # compared against it -- see the reconciliation check at the bottom
    # of this loop's own iteration body for the full mechanism. None
    # whenever the gate was off on the previous iteration (nothing to
    # reconcile against) or before the very first diagnostics-on
    # iteration.
    _prev_iter_window_end: float | None = None

    # THE LIFECYCLE (opendarts/lifecycle/). Every tick the driver observes the
    # frames and the adapter translates the resulting Tick into the
    # ThrowTriggerState this loop consumes: READY_TO_CAPTURE on a commit
    # (with the lifecycle's own (bg, frame) pair), a cleared visit on a
    # takeout, MOTION_DETECTED/SETTLING/TAKEOUT_WAITING for the UI. There
    # is no other trigger. If the driver disables itself after repeated
    # errors the loop logs at ERROR and scores nothing -- never a silent
    # fallback.
    from opendarts.lifecycle.adapter import LifecycleTriggerAdapter

    lifecycle = _build_lifecycle_driver(
        lifecycle_settings_store=lifecycle_settings_store,
    )
    lifecycle_adapter = LifecycleTriggerAdapter(max_darts=MAX_DARTS_PER_TURN)
    _lifecycle_waiting_logged_at: float | None = None
    _lifecycle_disabled_logged = False

    iteration = 0
    while not stop_event.is_set() and not (also_stop is not None and also_stop.is_set()):
        iteration += 1
        iteration_started = time.monotonic()
        # MEASUREMENT CLOCK, separate from the PACING clock above,
        # 2026-09-13 (CPU task). `time.monotonic()` is backed by
        # GetTickCount64 on Windows through Python 3.12 -- resolution
        # ~15.6ms, the OS clock tick. Every per-iteration timing below is
        # SMALLER than that, so on the rig this instrument reported every
        # phase as exactly 0.00 / 15.00 / 16.00 / 32.00 ms: a quantised
        # readout that looks like data and is not. Confirmed live before
        # the fix -- `lifecycle=15.00ms` on a phase this project's own
        # benchmarks put near 0.4ms.
        #
        # `time.perf_counter()` is the high-resolution counter
        # (QueryPerformanceCounter on Windows, sub-microsecond) and is
        # what every DURATION below now uses. `iteration_started` stays on
        # monotonic because it drives real PACING
        # (`_wait_for_next_iteration()`), where a coarse clock is harmless
        # and monotonic's guarantees are the ones that matter.
        #
        # Never mix the two in one subtraction: they have unrelated
        # origins, so a perf timestamp minus a monotonic one is garbage.
        _iter_perf = time.perf_counter()
        # Master gate for the WHOLE per-iteration cost-breakdown
        # instrument below -- a single, cheap boolean read, the ONLY
        # thing evaluated when diagnostics are off (per this task's own
        # explicit "no extra time.monotonic() call, no extra dict/string
        # construction, nothing beyond the single boolean check itself"
        # requirement). `_diag_state_at_entry` reuses the SAME
        # `trigger.state` value `prev_state` captures a little further
        # down (before advance() runs) -- captured here too, redundantly
        # but for zero extra cost (a plain enum attribute read), because
        # this diagnostic's own reconciliation/volume-control logic needs
        # it well before `prev_state` is set for the Reset-request/
        # late-camera-backfill code paths that sit between here and
        # there.
        _diag_on = diagnostics_gate.enabled()
        _diag_state_at_entry = trigger.state
        if _diag_on and _prev_iter_window_end is not None:
            # SELF-CONSISTENCY CHECK, required by this task's own spec:
            # the PREVIOUS iteration's own measured `body + sleep` should
            # equal the real wall-clock gap to THIS iteration's own start
            # -- if it doesn't, that gap is itself real information (most
            # plausibly this instrument's own log-line construction/
            # emission cost, which necessarily happens AFTER `sleep` is
            # measured -- see the emission code at the bottom of this
            # loop's own iteration body -- or ordinary Python loop
            # overhead) that the operator should see, not a silent
            # measurement bug papered over by trusting the instrument's
            # own arithmetic blindly. Tolerance NOT independently
            # measured against live cadence data (no rig access from
            # this worktree) -- 10ms is a conservative, documented-as-
            # unverified round number, comfortably above a single
            # log.debug() call's own real cost (this project's own
            # 2026-09-04 diagnostics_gate measurement: ~12 microseconds/
            # call for a SUPPRESSED third-party logger, so plausibly
            # somewhat more here for a real, formatted, file-written
            # line, but nowhere near 10ms) and comfortably below any real
            # settle-path iteration's own cost, so it should only fire on
            # a genuine reconciliation gap, not routine variance.
            _recon_gap_s = _iter_perf - _prev_iter_window_end
            if abs(_recon_gap_s) > 0.010:
                log.debug(
                    "iteration %d: per-iteration diagnostic self-consistency check: "
                    "real gap to this iteration's own start (%.4fs) does not match "
                    "the previous iteration's own measured body+sleep -- likely this "
                    "instrument's own log-line construction/emission cost (which "
                    "necessarily happens AFTER sleep is measured) or ordinary loop "
                    "overhead, not a measurement bug",
                    iteration, _recon_gap_s,
                )
        if trigger.state in (ThrowState.MOTION_DETECTED, ThrowState.SETTLING):
            if _last_settle_iteration_started is not None:
                real_cadence_s = _iter_perf - _last_settle_iteration_started
                log.debug(
                    "iteration %d: real wall-clock time since the previous "
                    "settle-episode iteration: %.3fs (configured poll_interval_s=%.3fs)",
                    iteration, real_cadence_s, poll_interval_s,
                )
            _last_settle_iteration_started = _iter_perf
        else:
            _last_settle_iteration_started = None
        fetch_started = time.monotonic()
        current_frames = fetch_current_frames(
            scratch_dir / "current", hub=hub
        )
        fetch_elapsed = time.monotonic() - fetch_started
        # Same tick as the fetch, before anything else can run: this is the
        # only moment the hub's per-slot JPEG is guaranteed to still be
        # this frame's (the pump overwrites it every cycle).
        frame_jpegs.record(hub, current_frames)
        # Threshold updated 2026-08-12 alongside the POLL_INTERVAL_SECONDS
        # drop: a bare `poll_interval_s *
        # 3` was fine when poll_interval_s was 0.5s (1.5s threshold, well
        # above any real fetch), but at the new ~0.03s local default that
        # same multiplier is only ~90ms -- comfortably BELOW a normal,
        # healthy local fetch's own real cost (a `cv2.VideoCapture.read()`
        # + `cv2.imwrite()` PNG encode/write per camera, sequential across
        # 3 cameras, unmeasured exactly but plausibly in the same
        # ballpark), which would turn this into log spam on every single
        # iteration rather than a genuine stall signal. Floored at
        # _FETCH_SLOW_WARNING_FLOOR_S so it still only fires on something
        # actually abnormal, regardless of how small poll_interval_s is.
        if fetch_elapsed > max(poll_interval_s * 3, _FETCH_SLOW_WARNING_FLOOR_S):
            log.warning(
                "iteration %d: fetching current frames took %.2fs (expected ~%.2fs) -- "
                "camera read may be slow/stalling",
                iteration, fetch_elapsed, poll_interval_s,
            )

        # FRAME-AGE-AT-ADVANCE() DIAGNOSTIC -- see _FRAME_AGE_LOG_FLOOR_S's
        # own module-level comment for the full reasoning.
        #
        # 2026-09-04, opendarts.live.diagnostics_gate task: this computation
        # itself is cheap in isolation (one time.monotonic() call plus a
        # per-camera dict subtraction, no CV work) -- but it ran
        # UNCONDITIONALLY, every single poll-loop iteration, regardless
        # of whether anything ever reads the result. The requirement was
        # explicit: a real switch
        # should make diagnostics cost genuinely ZERO extra allocation
        # when off, not merely "small" -- gated the whole block behind
        # opendarts.live.diagnostics_gate accordingly, same shape as every
        # other per-tick diagnostic-only computation this task gates.
        # `frame_ages_s` stays the empty dict declared below when
        # diagnostics are off, which is also exactly what makes the
        # MOTION_DETECTED-transition line's own reuse of it further down
        # a no-op (no frame-age detail attached) without that call site
        # needing its own separate gate.
        frame_ages_s: dict[int, float] = {}
        if (
            diagnostics_gate.enabled()
            and hub is not None
            and hasattr(hub, "status")
        ):
            now_for_frame_age = time.monotonic()
            for cam in current_frames:
                status = hub.status.get(cam)
                if status is None:
                    continue
                stamped_monotonic = getattr(status, "last_read_at_monotonic", None)
                if stamped_monotonic is None:
                    continue
                frame_ages_s[cam] = now_for_frame_age - stamped_monotonic
        if frame_ages_s:
            _age_exceeds_floor = any(age > _FRAME_AGE_LOG_FLOOR_S for age in frame_ages_s.values())
            _age_sampled_iteration = (iteration % _FRAME_AGE_SAMPLE_EVERY_N_ITERATIONS) == 0
            if _age_exceeds_floor or _age_sampled_iteration:
                log.debug(
                    "iteration %d: frame age at lifecycle consumption: %s%s",
                    iteration,
                    {cam: round(age, 4) for cam, age in sorted(frame_ages_s.items())},
                    (
                        f" (>= {_FRAME_AGE_LOG_FLOOR_S:.3f}s floor)"
                        if _age_exceeds_floor
                        else " (sampled, every "
                        f"{_FRAME_AGE_SAMPLE_EVERY_N_ITERATIONS} iterations)"
                    ),
                )

        # FRAME-FRESHNESS GATE -- see MAX_FRAME_FRESHNESS_WAIT_S's own
        # comment for the full incident/mechanism writeup. Only real for
        # a genuine local hub (LocalCameraHub.status is where frame_count
        # actually lives) -- a bare duck-typed test double without a
        # `.status` attribute, or any frame source with no hub at all (a
        # race that cannot apply there), skip this
        # entirely and behave exactly as before this fix, matching this
        # module's own existing hasattr()-guard convention (see
        # _bootstrap_calibrations_at_start_with_reopen_retry()) for
        # degrading safely against a hub that doesn't implement the full
        # real interface rather than crashing on one that doesn't need
        # to.
        # ITERATION-DIAGNOSTIC TIMING, 2026-09-04 -- covers this whole
        # frame-freshness gate block (including the "stalled past
        # ceiling" WARNING and the "still stale, not yet at ceiling"
        # skip's own bookkeeping, but NOT the possible `continue` itself
        # -- see this diagnostic's own emission code near the bottom of
        # the loop for why an iteration that takes that `continue` path
        # is not currently included in the new per-iteration diagnostic
        # line's emission, a stated, documented scope limit, not an
        # oversight). Set unconditionally (not just inside the `if hub is
        # not None...` check below) so `gate` correctly reports ~0 rather
        # than being silently absent whenever this gate doesn't apply at
        # all (e.g. no hub at all, or a bare test-double hub).
        _t_gate_start = time.perf_counter() if _diag_on else None
        if hub is not None and hasattr(hub, "status"):
            now = time.monotonic()
            for cam in current_frames:
                status = hub.status.get(cam)
                if status is None:
                    continue
                count = status.frame_count
                prev_count = _last_seen_frame_count.get(cam)
                if prev_count is not None and count == prev_count:
                    _frame_stale_since.setdefault(cam, now)
                else:
                    _frame_stale_since.pop(cam, None)
                _last_seen_frame_count[cam] = count

            if not _frame_stale_since:
                # Every camera is delivering again -- arm the throttle so
                # the NEXT stall reports on its own first iteration rather
                # than inheriting the previous stall's interval.
                _stall_warned_at = None
                _stall_warns_suppressed = 0

            if _frame_stale_since:
                stalled_past_ceiling = {
                    cam: now - since
                    for cam, since in _frame_stale_since.items()
                    if now - since >= MAX_FRAME_FRESHNESS_WAIT_S
                }
                if stalled_past_ceiling:
                    # Throttled -- see _stall_warned_at's own comment. The
                    # FIRST iteration of a stall always logs (so entering
                    # the state is never delayed or hidden), then at most
                    # one line per STALL_WARNING_INTERVAL_S, each carrying
                    # the count it stands in for. A camera that recovers
                    # clears _frame_stale_since, which resets the throttle
                    # below, so a NEW stall logs immediately rather than
                    # inheriting the previous one's interval.
                    _suppressed = _stall_warns_suppressed
                    if (
                        _stall_warned_at is None
                        or (now - _stall_warned_at) >= STALL_WARNING_INTERVAL_S
                    ):
                        log.warning(
                            "iteration %d: camera(s) %s produced no new frame past "
                            "the %.2fs ceiling -- frame-freshness gate giving up "
                            "and proceeding anyway with a possibly-stale frame "
                            "rather than blocking capture indefinitely; camera may "
                            "be stalled%s",
                            iteration,
                            {cam: round(age, 2) for cam, age in sorted(stalled_past_ceiling.items())},
                            MAX_FRAME_FRESHNESS_WAIT_S,
                            f" ({_suppressed} further iteration(s) suppressed since "
                            f"the last line)" if _suppressed else "",
                        )
                        _stall_warned_at = now
                        _stall_warns_suppressed = 0
                    else:
                        _stall_warns_suppressed += 1
                    # Proceed anyway (fail open, same posture as every other
                    # bounded wait in this module) -- fall through to the
                    # normal iteration body below.
                else:
                    # At least one camera hasn't produced a genuinely new
                    # frame since the last accepted poll, and none have
                    # blown the ceiling yet -- this poll's current_frames
                    # would feed a duplicate into the settle window (the
                    # exact mechanism behind the 2026-09-01 phantom-dart
                    # incident). Skip it entirely: no Reset check, no
                    # advance() call, no settle-window append -- just poll
                    # again shortly. _last_seen_frame_count/_frame_stale_
                    # since are already updated above regardless, so a
                    # camera that catches up on the very next poll is
                    # detected correctly.
                    #
                    # 2026-09-05: frame-driven wait -- wake as soon as
                    # the pump publishes a new generation, not a flat
                    # poll_interval_s -- see _wait_for_next_frame()'s own
                    # docstring for the full reasoning. This particular
                    # skip path (a STALE-BUT-NOT-YET-AT-CEILING camera)
                    # means at least one camera's own frame_count hasn't
                    # advanced -- waiting on a NEW pump generation here
                    # is exactly the right thing to wait for, since a
                    # fresh generation is what would actually resolve
                    # that staleness.
                    wake_state = _wait_for_next_frame(
                        hub, wake_state,
                        stop_event, iteration_started, poll_interval_s, iteration,
                    )
                    continue
        _gate_elapsed = (time.perf_counter() - _t_gate_start) if _diag_on else 0.0

        # Manual "Reset" (POST /api/reset -> ResetRequest, see that
        # class's own docstring and this function's own reset_request=
        # docstring section) -- checked BEFORE advance() every iteration,
        # unconditionally, regardless of the trigger's current state:
        # whatever turn/takeout progress was in-flight is discarded, and
        # the CURRENT frames (just fetched above) become the new
        # true_baseline_frames AND bg_frames -- lock-clean semantics
        # (whatever's there right now,
        # dart or no dart, no validation). `continue`s past the normal
        # advance() call for this same iteration -- there is nothing left
        # to advance against; the very next iteration resumes normal
        # IDLE-watching against the freshly-reset baseline.
        if reset_request is not None and reset_request.check_and_clear():
            prev_state = trigger.state
            prev_dart_count = trigger.dart_count
            # TWO-BUFFER SPLIT -- a manual reset means "whatever's there
            # right now, dart or no dart, becomes the new normal" for
            # BOTH detection and scoring -- lock-clean semantics.
            bg_frames = current_frames
            trigger = ThrowTriggerState(
                state=ThrowState.IDLE, dart_count=0, true_baseline_frames=current_frames
            )
            lifecycle.reset()
            # THROW NUMBERING RESET, 2026-08-22. A manual Reset does NOT delete
            # any already-saved throw packages -- a bare reset-to-0
            # would try to write the very next throw as session_id-001-
            # ..., colliding with whatever real package is already
            # sitting there. _reset_session_throw_numbering() bumps a
            # counting generation instead, so the next throw_id is
            # provably disjoint from anything already on disk (or
            # already pulled elsewhere) -- see its own docstring and
            # handle_ready_to_capture()'s throw_id-construction comment
            # for the full collision analysis.
            #
            # _THROW_NUMBER_LOCK, 2026-09-06/07 -- see that lock's own
            # module-level comment: this manual Reset can now race a
            # PREVIOUS dart's own still-in-flight background
            # handle_ready_to_capture() call (Part 2), which holds the
            # SAME lock for its own throw_number allocation -- must
            # serialize against it here too, not just inside that
            # function.
            with _THROW_NUMBER_LOCK:
                _reset_session_throw_numbering(
                    package_root.parent / "session_throw_counters", session_id
                )
            # A manual Reset explicitly means "abandon whatever turn was
            # in flight" ("clear visit / escape stuck takeout"), so it
            # rotates the visit unconditionally --
            # including when no dart was captured in it. Rotating an
            # empty, unused visit is harmless (no package ever referenced
            # it); NOT rotating after a Reset would be actively wrong,
            # since the next dart thrown would be filed under the same
            # visit as the darts the operator just discarded.
            previous_visit_id = visit_id
            visit_id = new_visit_id()
            log.info(
                "manual reset (iteration %d): %s -> IDLE -- baseline captured fresh from "
                "current frames, dart count reset to 0, visit %s -> %s",
                iteration, prev_state.name, previous_visit_id, visit_id,
            )
            _emit(
                on_event,
                {
                    "type": "VISIT_CLEARED",
                    "session": session_id,
                    "visit_id": visit_id,
                    "previous_visit_id": previous_visit_id,
                    "n_darts": prev_dart_count,
                    "reason": "reset",
                },
            )
            _board_photo_due = True
            _emit(
                on_event,
                {
                    "type": "TRIGGER_STATE",
                    "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "state": trigger.state.name,
                    "session": session_id,
                    "dart_count": trigger.dart_count,
                    "visit_id": visit_id,
                },
            )
            last_heartbeat = time.monotonic()
            # 2026-09-05: frame-driven wait, see _wait_for_next_frame()'s
            # own docstring.
            wake_state = _wait_for_next_frame(
                hub, wake_state,
                stop_event, iteration_started, poll_interval_s, iteration,
            )
            continue

        prev_state = trigger.state
        prev_dart_count = trigger.dart_count
        _t_lifecycle_start = time.perf_counter() if _diag_on else None
        _live_step = None
        if lifecycle.disabled:
            if not _lifecycle_disabled_logged:
                log.error(
                    "lifecycle: driver disabled itself after repeated errors -- "
                    "NO throw detection for the rest of this run; restart the daemon"
                )
                _lifecycle_disabled_logged = True
        else:
            _live_step = _lifecycle_step(
                lifecycle, lifecycle_adapter, current_frames, bg_frames,
                wake_state.total_frames_dropped,
            )
            if _live_step is None:
                # No calibrated board masks yet: nothing can be judged, and
                # nothing scores without a calibration anyway. Once a minute.
                _now_wait = time.monotonic()
                if (
                    _lifecycle_waiting_logged_at is None
                    or _now_wait - _lifecycle_waiting_logged_at >= 60.0
                ):
                    log.warning(
                        "lifecycle: waiting for calibrated board masks -- "
                        "no throw detection until a calibration is present"
                    )
                    _lifecycle_waiting_logged_at = _now_wait
            else:
                trigger = _live_step.trigger
                # on a commit tick this is the pre-dart board the engines
                # score against; otherwise it just tracks the reference
                bg_frames = _live_step.reference
                if trigger.state is ThrowState.READY_TO_CAPTURE and trigger.last_frame:
                    # Looked up BEFORE re-pinning below: the bg being
                    # scored is the reference as it stood before this
                    # commit, which the lifecycle may just have replaced.
                    trigger.last_frame_jpegs = frame_jpegs.lookup(trigger.last_frame)
                    trigger.bg_jpegs = frame_jpegs.lookup(bg_frames)
                    # Which ring set holds each of those frames -- how the
                    # package's clip takes them out of the ring by number.
                    trigger.last_frame_generations = frame_jpegs.lookup_generations(
                        trigger.last_frame)
                    trigger.bg_generations = frame_jpegs.lookup_generations(bg_frames)
                frame_jpegs.pin(bg_frames, trigger.true_baseline_frames)
        if _live_step is not None and _live_step.cleared_darts:
            previous_visit_id = visit_id
            visit_id = new_visit_id()
            log.info(
                "visit cleared by lifecycle (iteration %d): %s -- visit %s (%d dart(s)) -> %s",
                iteration, _live_step.reason, previous_visit_id, _live_step.cleared_darts, visit_id,
            )
            _emit(
                on_event,
                {
                    "type": "VISIT_CLEARED",
                    "session": session_id,
                    "visit_id": visit_id,
                    "previous_visit_id": previous_visit_id,
                    "n_darts": _live_step.cleared_darts,
                    "reason": "takeout",
                },
            )
            _board_photo_due = True
        if _calibration_key(calibration_store, calibrations) != _photo_calibration_key:
            _board_photo_due = True
        if _board_photo_due and _live_step is not None and _lifecycle_phase_is_idle(lifecycle):
            _board_photo_due = False
            _photo_calibrations = calibration_store.get() if calibration_store is not None else calibrations
            _photo_calibration_key = _calibration_key(calibration_store, calibrations)
            if on_event is not None and _photo_calibrations and bg_frames:
                board_photo.RENDERER.submit(
                    bg_frames, _photo_calibrations,
                    lambda jpeg, _vid=visit_id: _emit(on_event, {
                        "type": "BOARD_PHOTO",
                        "jpeg": jpeg,
                        "visit_id": _vid,
                    }),
                    skip_if_unchanged=True,
                )
        _t_lifecycle_end = time.perf_counter() if _diag_on else None
        _lifecycle_elapsed = (_t_lifecycle_end - _t_lifecycle_start) if _diag_on else 0.0
        _post_elapsed: float | None = None

        if trigger.state != prev_state:
            # FRAME-AGE-AT-TRIP, 2026-09-04 -- answers "when IDLE ->
            # MOTION_DETECTED finally fires, how old was the frame that
            # tripped it" directly on the ONE log line that already
            # narrates this exact transition, rather than making a
            # reader correlate it against a separate periodic line by
            # timestamp. Reuses `frame_ages_s` (computed once, above,
            # right after this iteration's own fetch_current_frames()
            # call) rather than re-measuring -- the age reported here is
            # therefore exactly what advance() itself consumed to reach
            # this verdict, not a later, separately-timed sample. Only
            # attached on the transition INTO MOTION_DETECTED
            # specifically (not every transition) -- that's the one this
            # diagnostic was built to answer; every other transition's
            # own timing (e.g. READY_TO_CAPTURE's settle_duration_s,
            # below) already has its own, separate, purpose-built field.
            frame_age_extra = ""
            if trigger.state == ThrowState.MOTION_DETECTED and frame_ages_s:
                frame_age_extra = " -- frame age at trip: %s" % (
                    {cam: round(age, 4) for cam, age in sorted(frame_ages_s.items())}
                )
            log.info(
                "trigger state: %s -> %s (iteration %d)%s",
                prev_state.name, trigger.state.name, iteration, frame_age_extra,
            )
            # settle_duration_s/straggler_camera (2026-09-01, latency-
            # instrumentation task, purely additive, no behavior change)
            # -- exposes on the wire exactly what throw_trigger.py's own
            # "MOTION_DETECTED/SETTLING -> READY_TO_CAPTURE after Xs" log
            # line already computes internally (see that module's own
            # comment block, same duration/straggler derivation, kept in
            # sync deliberately rather than reading the log line back).
            # Only ever present on the READY_TO_CAPTURE transition itself
            # -- every other state has no settle interval to report, and
            # `.get()`-style absence (never a fabricated 0.0/None-that-
            # looks-like-a-real-zero) is this project's own established
            # convention for "not applicable here," not a KeyError trap
            # for a consumer that only cares about this one transition.
            # Lets an external latency harness read the real internal
            # duration directly instead of deriving it from two separate
            # TRIGGER_STATE receipt timestamps -- which was measuring
            # asyncio dispatch/broadcast latency, not the settle window
            # itself (see docs/DESIGN.md/cross-session latency investigation,
            # 2026-09-01, for the full finding this closes).
            settle_extra: dict[str, Any] = {}
            if trigger.state == ThrowState.READY_TO_CAPTURE and trigger.camera_settled_at_monotonic:
                straggler_camera = max(
                    trigger.camera_settled_at_monotonic, key=trigger.camera_settled_at_monotonic.get
                )
                settle_extra["straggler_camera"] = straggler_camera
                if trigger.settle_started_monotonic is not None:
                    settle_extra["settle_duration_s"] = round(
                        trigger.camera_settled_at_monotonic[straggler_camera]
                        - trigger.settle_started_monotonic,
                        3,
                    )
            _emit(
                on_event,
                {
                    "type": "TRIGGER_STATE",
                    "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "state": trigger.state.name,
                    "session": session_id,
                    "dart_count": trigger.dart_count,
                    "visit_id": visit_id,
                    **settle_extra,
                },
            )
            last_heartbeat = time.monotonic()
            # ITERATION-DIAGNOSTIC TIMING, 2026-09-04 -- `post`: time from
            # advance() returning to this transition event actually being
            # emitted/logged. Only meaningful for a real state-transition
            # iteration -- see `_post_elapsed`'s own initialization above
            # advance().
            #
            # TAKES ITS OWN perf_counter() STAMP, 2026-09-13 (CPU task).
            # This used to reuse `last_heartbeat`'s just-computed
            # time.monotonic() as the end timestamp, to avoid a second
            # clock call. That is no longer sound: `_t_lifecycle_end` is a
            # perf_counter stamp now (monotonic is ~15.6ms-granular on
            # Windows, which quantised every phase here to 0/15/16/32ms),
            # and subtracting across the two clocks yields a meaningless
            # number rather than a coarse one. `last_heartbeat` itself
            # stays on monotonic -- it gates a multi-second interval just
            # below, where 15.6ms granularity is irrelevant. The extra
            # call happens only on a real transition, and only when
            # diagnostics are on.
            if _diag_on:
                _post_elapsed = time.perf_counter() - _t_lifecycle_end
        elif time.monotonic() - last_heartbeat >= _HEARTBEAT_EVERY_S:
            # DEBUG, not INFO
            # default console every _HEARTBEAT_EVERY_S while nothing was
            # happening ("we dont need this constantly spamming"). Still
            # worth keeping (see this loop's own "Heartbeat" comment above
            # for why it exists -- distinguishing a genuinely hung loop
            # from a healthy IDLE one), just not at INFO: the file handler
            # stays at DEBUG regardless of console level (opendarts/live/
            # logging_setup.py), so it's still there if you ever need to
            # prove the loop didn't hang -- just off the default console.
            log.debug(
                "trigger state: still %s after %d iterations (loop is alive)",
                trigger.state.name, iteration,
            )
            last_heartbeat = time.monotonic()

        if trigger.state == ThrowState.READY_TO_CAPTURE:
            # Fresh read EVERY throw, not the stale startup-local
            # `calibrations` -- this is the actual point of
            # CalibrationStore: a manual recalibrate that landed in
            # between two darts must be what scores the NEXT one.
            #
            # get_with_package_id() (not two separate .get()/
            # .get_package_id() calls) -- deliberate: a manual recalibrate
            # landing on a different thread BETWEEN two independently-
            # locked reads could hand back calibrations from one
            # calibration event and a package_id from a newer one, which
            # would silently break calibration-package traceability (see
            # that method's own docstring). One atomic read makes that
            # structurally impossible.
            current_calibrations, current_package_id = calibration_store.get_with_package_id()
            # Fresh {} every throw, not reused across iterations -- a
            # stale value from a PRIOR throw's own successful extraction
            # must never be mistaken for THIS throw's own (a throw whose
            # primary engine legitimately produced nothing here, e.g.
            # Talos primary, must see an empty dict, not last time's).
            own_tip_line_px_out: dict = {}
            _handle_ready_to_capture_kwargs: dict[str, Any] = dict(
                ad_ws_listener=ad_ws_listener, ad_match_window_sec=ad_match_window_sec,
                engine_config_store=engine_config_store, on_event=on_event,
                throw_capture=throw_capture,
                visit_id=visit_id,
                # 0-based index WITHIN the visit, matching the `index`
                # (0/1/2) on the throw-correction route.
                # `trigger.dart_count` was already incremented to N by
                # advance()'s own READY_TO_CAPTURE transition for the Nth
                # dart of this turn, so the 0-based index is one less.
                visit_index=trigger.dart_count - 1,
                calibration_package_id=current_package_id,
                cached_prior_frames=cached_prior_frames,
                own_tip_line_px_out=own_tip_line_px_out,
                background_save=background_save,
                store_packages=store_packages,
                min_free_disk_gb=min_free_disk_gb,
            )
            # PART 2, 2026-09-06/07, Zeus-latency follow-up task --
            # `background_save` now ALSO governs whether THIS LOOP
            # dispatches the ENTIRE handle_ready_to_capture() call (not
            # just its own internal save step, which was already
            # backgrounded as of 2026-09-01) to a background thread.
            # See `_dispatch_handle_ready_to_capture_in_background()`'s
            # own docstring for the full real-measured problem this
            # closes, the new overlapping-calls risk it introduces (and
            # closes, via `_THROW_NUMBER_LOCK`), and why reusing this ONE
            # existing flag (rather than a second, independent parameter)
            # is the right call: `background_save=False` already means
            # "fully synchronous, deterministic, testable" end to end for
            # every pre-existing caller/test that relies on it -- reusing
            # it here means those callers need ZERO changes, while
            # `background_save=True` (the real production default)
            # correctly extends the ALREADY-accepted "the trigger/settle
            # state machine never waits on scoring" guarantee
            # (docs/ENGINES.md) to the PRIMARY engine too, not just
            # also-run engines.
            #
            # `dest_dir` (the function's own return value) is
            # DELIBERATELY not read at all in the background_save=True
            # branch -- confirmed by direct inspection that nothing in
            # this loop's own body reads it after this point (only
            # `own_tip_line_px_out`/the args already captured above are
            # used below, none of which depend on the call having
            # returned yet).
            if background_save:
                threading.Thread(
                    target=_dispatch_handle_ready_to_capture_in_background,
                    args=(trigger, bg_frames, current_calibrations, package_root, session_id),
                    kwargs=_handle_ready_to_capture_kwargs,
                    name=f"handle-ready-to-capture-{session_id}-dart{trigger.dart_count}",
                    daemon=True,
                ).start()
            else:
                # background_save=False -- see handle_ready_to_capture()'s
                # own docstring for this parameter: runs fully inline,
                # synchronously, byte-identical to this loop's
                # pre-2026-09-06/07 behavior (and, before that, this
                # function's own pre-2026-09-01 behavior).
                handle_ready_to_capture(
                    trigger, bg_frames, current_calibrations, package_root, session_id,
                    **_handle_ready_to_capture_kwargs,
                )
            # PACKAGE_SAVED no longer emitted here (2026-09-01, "background
            # the throw-package save" task) -- the actual
            # save is now backgrounded inside handle_ready_to_capture()
            # itself, so emitting PACKAGE_SAVED right here (immediately
            # after the call returns) would claim the file exists before
            # it necessarily does. The real emit now happens from INSIDE
            # that background thread, once the write actually completes --
            # see handle_ready_to_capture()'s own _save_and_followups()
            # docstring.
            # Cache THIS throw's own bg_images/current_frames (2026-09-01,
            # same task as cached_prior_frames above) -- the NEXT throw's
            # own handle_ready_to_capture() call (if it turns out to be
            # this visit's immediately-following one) can then skip a
            # disk round-trip for exactly this data. Built from `bg_
            # frames`/`current_frames` AS THEY WERE for THIS call --
            # deliberately BEFORE the refresh_background_after_capture()
            # reassignment two lines down, and using the exact same
            # `cam in current_frames` filter handle_ready_to_capture()
            # itself applies internally for its own `bg_images`, so this
            # is a byte-accurate stand-in for what got saved to disk as
            # this throw's own bg/commit frames (its clip's two pointer
            # frames) -- never an
            # approximation of it.
            #
            # own_tip_line_px_out was passed in as a FRESH {} right above
            # this call, and handle_ready_to_capture() writes this throw's
            # tip line into it once scoring finishes. Synchronously
            # (background_save=False) that has already happened. Under
            # `background_save=True` (Part 2, 2026-09-06/07, the live
            # default) the call runs on a background thread that has not
            # necessarily started yet, so the value is read here AND the
            # dict itself is carried along: the next dart looks it up when
            # it needs it, about a second later, by which time scoring has
            # normally finished. Until 2026-09-17 only the value was
            # carried, which was always empty in production, so every
            # dart after the first re-ran detect_tip() on all three
            # cameras (~30 ms) before scoring. A lookup that still finds
            # nothing falls through to that recompute, as before.
            cached_prior_frames = CachedPriorThrowFrames(
                visit_id=visit_id,
                visit_index=trigger.dart_count - 1,
                bg_frames={cam: frame for cam, frame in bg_frames.items() if cam in current_frames},
                dart_frames=dict(current_frames),
                precomputed_tip_line=own_tip_line_px_out.get("own_tip_line_px"),
                precomputed_tip_line_out=own_tip_line_px_out,
            )
            # handle_ready_to_capture() ALWAYS saves a package regardless of
            # the live score's ok status (docs/DESIGN.md's "Replay is the source of truth"); nothing
            # below depends on `result.ok`. The lifecycle has already
            # adopted the dart into its reference; `trigger` here only
            # keeps the UI honest until the next tick's adapter output.
            if trigger.dart_count >= MAX_DARTS_PER_TURN:
                trigger = ThrowTriggerState(
                    state=ThrowState.TAKEOUT_WAITING,
                    dart_count=trigger.dart_count,
                    true_baseline_frames=trigger.true_baseline_frames,
                )
                log.info(
                    "turn complete (%d darts captured) -- entering TAKEOUT_WAITING",
                    trigger.dart_count,
                )
            else:
                trigger = ThrowTriggerState(
                    dart_count=trigger.dart_count,
                    true_baseline_frames=trigger.true_baseline_frames,
                )
            _emit(
                on_event,
                {
                    "type": "TRIGGER_STATE",
                    "emitted_at_utc": datetime.now(timezone.utc).isoformat(),
                    "state": trigger.state.name,
                    "session": session_id,
                    "dart_count": trigger.dart_count,
                    "visit_id": visit_id,
                },
            )

        # ITERATION-DIAGNOSTIC TIMING: `body` = iteration_started -> right
        # before the wait below (everything this iteration did, including a
        # synchronous handle_ready_to_capture() when background_save is
        # off); `sleep` = the real measured wait, logged AFTER it so it is
        # never a fabricated field.
        _t_body_end = time.perf_counter() if _diag_on else None
        _body_elapsed = (_t_body_end - _iter_perf) if _diag_on else 0.0
        # 2026-09-05: frame-driven wait -- wake as soon as the pump
        # publishes a new generation, not a flat poll_interval_s -- see
        # _wait_for_next_frame()'s own docstring for the full design
        # (drop-to-latest accounting, sustained-overrun logging, and the
        # no-hub fallback to the OLD deadline-compensated
        # `_wait_for_next_iteration()`, unchanged for that mode). `sleep`
        # below is still the REAL measured wait duration (now frame-
        # driven rather than a fixed sleep on the local-hub path) -- the
        # reconciliation/diagnostic math a few lines below is unaffected,
        # it already treats `sleep` as whatever actually elapsed, never
        # assumed to equal poll_interval_s.
        wake_state = _wait_for_next_frame(
            hub, wake_state,
            stop_event, iteration_started, poll_interval_s, iteration,
        )
        if _diag_on:
            _t_sleep_end = time.perf_counter()
            _sleep_elapsed = _t_sleep_end - _t_body_end
            _emit_diag = (
                _diag_state_at_entry != ThrowState.IDLE
                or (iteration % _ITERATION_DIAG_SAMPLE_EVERY_N_ITERATIONS) == 0
                or _body_elapsed > _ITERATION_DIAG_BODY_LOG_FLOOR_S
            )
            if _emit_diag:

                def _ms(v: float | None) -> str:
                    return f"{v * 1000:.2f}ms" if v is not None else "n/a"

                log.debug(
                    "iteration diagnostic: it=%d state=%s fetch=%s gate=%s lifecycle=%s "
                    "post=%s body=%s sleep=%s",
                    iteration,
                    _diag_state_at_entry.name,
                    _ms(fetch_elapsed),
                    _ms(_gate_elapsed),
                    _ms(_lifecycle_elapsed),
                    _ms(_post_elapsed),
                    _ms(_body_elapsed),
                    _ms(_sleep_elapsed),
                )
            _prev_iter_window_end = _t_sleep_end
        else:
            _prev_iter_window_end = None

    log.info("lifecycle stats: %s", lifecycle.stats.as_dict())
    lifecycle.close()
    log.info(
        "capture loop body stopped cleanly (%s)",
        "stop_event set -- whole process stopping"
        if stop_event.is_set()
        else "also_stop set -- this session ended, caller may start another",
    )


def run_capture_loop(
    package_root: Path = DEFAULT_PACKAGE_ROOT,
    poll_interval_s: float = POLL_INTERVAL_SECONDS,
    ad_base_url: str | None = DEFAULT_AD_BASE,
    ad_window_sec: float = DEFAULT_MATCH_WINDOW_SEC,
) -> None:
    """Standalone entrypoint's own loop runner -- opens/owns/closes its
    OWN `opendarts.live.local_capture.LocalCameraHub` and its own
    stop_event tied to SIGINT/SIGTERM handlers, then delegates the
    actual loop logic to run_capture_loop_body() above (the function
    opendarts/live/run_product.py's combined entrypoint ALSO calls, against
    a hub it shares with the web server instead of one owned solely by
    this function) -- see run_capture_loop_body()'s own docstring for
    the full behavior this wraps.

    Frame source: (the daemon's
    normal mode) opens the hub ONCE here at startup and keeps it open
    for this function's entire lifetime, closing it only on the way out
    (normal stop, exception, or NotImplementedError) -- never reopened
    per poll iteration, matching local_capture.py's own design point
    that opening cameras is expensive and a hub is meant to be a
    long-lived object, not a per-call resource. Pass


    ad_base_url: same lifecycle discipline as `hub` above -- when not
    None (the default), an `AdWsListener` is opened/started here,
    alongside the camera hub, and stopped in the same `finally` block on
    the way out. `ad_base_url=None` (what `--no-ad-ground-truth` resolves
    to, see main()) skips building one entirely -- no listener, no
    background WS connection attempt at all, not merely "disabled".

    A second external oracle's live-scored answer, when wanted, comes
    from running a non-voting registry engine -- this function no longer
    builds a separate ground-truth WebSocket listener for one.
    """
    log.info(
        "capture daemon starting: frame_source=local direct camera package_root=%s",
        package_root,
    )

    stop_event = threading.Event()

    def _handle_stop_signal(signum, _frame) -> None:
        log.info("received signal %d, stopping after current iteration", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_stop_signal)
    signal.signal(signal.SIGINT, _handle_stop_signal)

    hub: local_capture.LocalCameraHub | None = None
    ad_ws_listener: AdWsListener | None = None
    try:
        hub = local_capture.LocalCameraHub()
        ok_flags = hub.open_all()
        if not any(ok_flags):
            raise RuntimeError(
                "no camera opened locally -- cannot run the capture loop with "
                "zero working cameras. See the per-camera log lines above for "
                "why each one failed (hub.status_report()):\n"
                + hub.status_report()
            )
        log.info("local camera hub open:\n%s", hub.status_report())

        if ad_base_url is not None:
            ad_ws_listener = AdWsListener(ad_base_url)
            ad_ws_listener.start()
            log.info(
                "AD ground-truth WebSocket listener starting (ws_url=%s) -- inline "
                "auto-attach ON (--no-ad-ground-truth to disable)",
                ad_ws_listener.ws_url,
            )
        else:
            log.info("AD ground-truth inline auto-attach DISABLED (--no-ad-ground-truth)")

        run_capture_loop_body(
            hub=hub,
            package_root=package_root,
            poll_interval_s=poll_interval_s,
            stop_event=stop_event,
            ad_ws_listener=ad_ws_listener,
            ad_match_window_sec=ad_window_sec,
        )
        log.info("capture daemon stopped cleanly")
    finally:
        if ad_ws_listener is not None:
            log.info("stopping AD ground-truth WebSocket listener")
            ad_ws_listener.stop()
        if hub is not None:
            log.info("closing local camera hub")
            hub.close_all()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="opendarts always-on capture daemon")
    parser.add_argument(
        "--package-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help="Directory finished throw packages are written under (default: %(default)s)",
    )
    parser.add_argument(
        "--ad-base-url",
        type=str,
        default=DEFAULT_AD_BASE,
        help=(
            "Autodarts base URL for the automatic inline AD ground-truth "
            "WebSocket listener (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--ad-window-sec",
        type=float,
        default=DEFAULT_MATCH_WINDOW_SEC,
        help=(
            "How many seconds apart a opendarts capture and an AD-reported throw "
            "may be and still be considered the same physical throw (default: "
            "%(default)s)"
        ),
    )
    parser.add_argument(
        "--no-ad-ground-truth",
        action="store_true",
        default=False,
        help=(
            "Disable the automatic inline AD ground-truth capture (the "
            "persistent WebSocket listener that matches each just-saved throw "
            "package against AD's own live event stream, see "
            "opendarts/live/ad_ws_listener.py). On by default."
        ),
    )
    args = parser.parse_args(argv)

    # See POLL_INTERVAL_SECONDS's own comment for why it is what it is:
    # local capture is real blocking I/O already paced by the camera
    # itself.
    poll_interval_s = POLL_INTERVAL_SECONDS

    # 2026-08-12: it must be obvious whether this is running. Default run mode is a plain foreground console
    # process with verbose, unmissable stdout logging -- NOT a silent
    # launchd background service. a service manager (see
    # docs/DEPLOYMENT.md) is documented as a LATER option, once this loop
    # is actually proven stable and there's a real need to run it
    # unattended -- not the default/near-term way to run it. Explicit
    # stream=sys.stdout (basicConfig defaults to stderr, which is exactly
    # the kind of "can I tell if it's running" ambiguity this is fixing).
    from opendarts.live.logging_setup import configure_console_and_file_logging

    log_path = configure_console_and_file_logging("capture_daemon")
    log.info("=" * 60)
    log.info("opendarts capture daemon -- starting in foreground console mode")
    log.info("frame source: local direct camera (opendarts.live.local_capture)")
    log.info("poll interval: %.3fs", poll_interval_s)
    log.info(
        "AD ground truth: %s",
        f"inline auto-attach ON ({args.ad_base_url})"
        if not args.no_ad_ground_truth
        else "inline auto-attach OFF (--no-ad-ground-truth)",
    )
    log.info("this process logs every real state transition, plus a heartbeat")
    log.info("line at least every ~10s even with no activity -- if you see")
    log.info("nothing for longer than that once running, it has stopped or hung")
    log.info("logging to console AND to %s", log_path)
    log.info("stop with Ctrl-C")
    log.info("=" * 60)
    ad_base_url = None if args.no_ad_ground_truth else args.ad_base_url
    try:
        run_capture_loop(
            package_root=args.package_root,
            poll_interval_s=poll_interval_s,
            ad_base_url=ad_base_url,
            ad_window_sec=args.ad_window_sec,
        )
    except NotImplementedError as exc:
        # No longer an EXPECTED outcome (refresh_background_after_capture()
        # and throw_trigger.ThrowState.TAKEOUT_WAITING are both real as of
        # 2026-08-12) -- kept as a loud,
        # non-silent failure mode in case some OTHER genuinely-unbuilt
        # piece raises it in the future, rather than assumed impossible.
        log.error("capture loop hit an unexpected NotImplementedError: %s", exc)
        log.error(
            "this is NOT an expected/known gap anymore -- the turn/takeout "
            "state machine is implemented; treat this as a real bug and "
            "check the traceback."
        )
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
