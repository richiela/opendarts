"""opendarts/live/server.py -- FastAPI dashboard/API server for this project's
live capture path. See docs/DEPLOYMENT.md's "Remote retrieval without
SSH" section (which this module fulfills/supersedes) and docs/DESIGN.md's
live-system-access guardrail (HTTP only, never SSH) before touching this
file.

**docs/LIVE_API.md is the written contract for everything this module
serves** -- every route, every WebSocket event, and the honest list of
what's still missing. Added 2026-08-14 (this project had no written API
doc at all before then). Keep it updated when a route or event changes
here; a consumer -- a real game/scoring layer eventually -- reads that
file, not this docstring.

STATUS: real, runnable MVP -- this is the "point a browser at it" server
requested, serving this project's own data: throw packages
and live camera snapshots.

Deliberately does NOT run or depend on opendarts/live/capture_daemon.py's
run_capture_loop() -- that loop cannot complete a full autonomous cycle
yet (refresh_background_after_capture() is an unresolved stub, see that
module). This server works standalone:
  - Camera snapshots are fetched live, per-request. As of this revision
    the DEFAULT source is a persistent
    opendarts.live.local_capture.LocalCameraHub -- opened once at process
    startup (see create_app()'s lifespan below) and held open for the
    server's whole lifetime, same "open once, not per-request" design
    point as capture_daemon.py's own hub.
  - Calibration status is computed by this server itself, on demand only
    (see AppState.refresh_calibration, called from the dashboard's
    "Refresh calibration now" button / POST /api/calibration/refresh),
    reusing opendarts.live.capture_daemon.bootstrap_calibrations() (real,
    proven live 2026-08-12 over HTTP; same function now also drives it
    over the local hub) -- NOT by reading state from a live daemon
    process, because no such process/IPC exists to read from yet.
    **CHANGED 2026-08-12**: this used to also run on its own 20s
    background poll timer, purely to keep the displayed numbers fresh --
    REMOVED entirely.
    Real scoring was never driven by this server's calibration poll
    anyway (see opendarts/live/capture_daemon.py's own docstring) -- what
    IS new is that a manual recalibrate now actually replaces the LIVE
    calibration the capture loop scores against (opendarts.live.
    capture_daemon.CalibrationStore, shared via opendarts/live/run_product.py)
    and persists a durable record to disk, not just refreshing this
    server's own display.
  - There is therefore no real "trigger state" (IDLE/MOTION_DETECTED/
    SETTLING/READY_TO_CAPTURE) to report -- /api/state says so plainly
    rather than inventing one. See ThrowTriggerState in
    opendarts/capture/trigger_state.py (produced by opendarts.lifecycle) for
    what gets reported once something is actually running and reachable.

WebSocket /api/events: REAL PUSH when run via opendarts.live.run_product,
POLLING FALLBACK otherwise. When this app is created standalone (this
module's own CLI, `-m opendarts.live.server`, no live capture process
exists), there is no daemon to push real events from, so this server
polls DEFAULT_PACKAGE_ROOT for new/changed throw packages on its own
timer (see AppState._package_poll_loop below) and broadcasts a message
to connected clients when something changes. create_app() also accepts
an optional `live_event_queue` (a thread-safe `queue.SimpleQueue`) --
when given, AppState._live_event_loop consumes real TRIGGER_STATE/
CALIBRATION_STATUS/PACKAGE_SAVED events pushed onto it from another
thread and broadcasts them immediately, no polling delay. This is the
seam opendarts/live/run_product.py's combined entrypoint uses to push real
capture-loop events straight from its background capture thread into
this server's WebSocket clients (see that module). Calibration status is
NOT polled at all anymore (see above) -- it only changes, and only
broadcasts a CALIBRATION_STATUS message, on an explicit manual refresh
**or** when a Start's own auto-calibrate bootstrap runs/reuses an
existing calibration (see "Status pill / button feedback" below --
`source` on that message distinguishes the two).

Status pill / button feedback:
  - **Pill vocabulary is now two-axis**, specified deliberately before
    building rather than guessed from memory --
    a PRIMARY status (No live capture / Connecting / Stopped / Starting /
    Throw / Takeout, driven by `AppState.capture_starting` +
    `CaptureLoopController.meta()['running']` + whether
    `ThrowTriggerState.state == TAKEOUT_WAITING`) and a secondary PHASE
    detail (the real `ThrowState` sub-state -- Motion/Settling/
    Capturing/dart counts) -- a status+event shape adapted
    to opendarts's real signals (see `_render_dashboard_html`'s own
    docstring and the rendered page's `PRIMARY_INFO`/`PHASE_DETAIL`
    JS objects for the exact mapping).
  - **Every Start/Stop/Reset/Calibrate click** gets a persistent,
    timestamped line in a new sidebar "Recent actions" log
    (`#action-log`, JS `logAction()`/`updateActionLine()`) the instant
    it's sent, updated in place once the request resolves -- not just a
    transient button-label change. All four control buttons are disabled
    together (JS `setControlsBusy()`) for the duration of any single
    in-flight action, so a second click cannot register while one is
    still outstanding.
  - **Start's auto-calibrate step is now explicitly confirmed**: the
    capture loop's own "Calibration bootstrap" section
    (opendarts/live/capture_daemon.py's `run_capture_loop_body`) emits a
    real `CALIBRATION_STATUS` event with `source: "startup"` (freshly
    recalibrated) or `"startup_reused"` (skipped, a valid calibration
    already existed) -- `AppState._handle_live_event` broadcasts it
    (reusing the SAME message shape/handler a manual "Calibrate" click's
    own broadcast already uses, just with `source` set), and the
    dashboard both updates the pill's "Starting" detail text and appends
    an action-log line the moment it lands, so there is never a silent
    "not sure it did" gap.

This server opens the cameras itself, directly.

CLI usage:
    .venv/bin/python3 -m opendarts.live.server [--port 8420] [--host 0.0.0.0]

STATUS-HONESTY AUDIT, 2026-08-12 -- real incident: hit a live
`/api/stop` that correctly reported `already_stopped: true` (the
idle-timeout had auto-stopped the capture loop ~88 minutes earlier), but
`GET /api/cameras/status` kept reporting `opened: true, last_read_ok:
true` for all 3 cameras the ENTIRE time -- `last_read_at` frozen over an
hour stale -- while a real live snapshot fetch during that window failed
outright, directly contradicting the status endpoint. Root cause:
`opendarts.live.local_capture.LocalCameraHub.close_all()` released every
camera but never touched `CameraStatus`, so each camera just kept
reporting its last-pumped-before-death state forever (fixed in that
module -- see its own docstring for the full writeup). The rule is that
every status surface presents valid, current data -- not scoped to just
that one field, so every status/state surface this server exposes
(`state_dict()`'s sections, `/api/cameras/status`, every WS broadcast)
was audited for the same pattern: a value that was true when written but
nothing ever invalidates when the underlying reality changes.

Found and fixed here too: `AppState.trigger_state`/`trigger_dart_count`
had the EXACT same bug, one level up the stack. They're only ever written
by a real `TRIGGER_STATE` event pushed from the capture thread
(`_handle_live_event` below) -- when a session ends (manual Stop or
idle-timeout), `CaptureLoopController.mark_session_ended()`
(`opendarts/live/capture_daemon.py`) flips `running` False but pushes no
event of its own, so `trigger_state` just sat at whatever throw-in-
progress state (e.g. `TAKEOUT_WAITING`) it last saw, forever, after the
loop had actually stopped -- `/api/state` would keep claiming a stopped
capture loop was still mid-takeout. The dashboard's own JS pill already
happened to mask this in the RENDERED view (`renderPill()`'s `cl &&
!cl.running` branch takes priority over the trigger's last-known state,
see the "Status pill / button feedback" section above) -- but that's a
client-side rendering workaround, not a fix to the actual data `/api/
state` returns to any caller. Fixed at the source instead, in
`AppState.stop_capture()` (the one shared implementation both `POST
/api/stop` and the idle-timeout path already go through): resets
`trigger_state`/`trigger_dart_count` to honest-null and broadcasts a real
`TRIGGER_STATE` reset the moment a session actually ends, rather than
relying on the client to keep masking it correctly forever. See that
method's own docstring for the full fix.

Every other status/state surface this server exposes was checked
against the same question ("does this get actively invalidated when
reality changes, or could it go stale and keep reporting an old truth as
current?") and found to already be honest, for one of two real reasons
-- not touched, and why, stated plainly rather than silently:
  - **Recomputed fresh on every single read**, so there is no cached
    value that COULD go stale: `cameras_status_dict()`/`/api/cameras/
    status` (reads `hub.status` live), `state_dict()`/`/api/state` (every
    section built fresh per call), the WS `HELLO` payload (calls
    `state_dict()` itself), `CaptureLoopController.meta()`
    (`seconds_since_activity` is `time.monotonic() - last_activity` at
    read time, not a stored countdown; `running` is written directly by
    `request_start()`/`request_stop()`/`mark_session_ended()`, never left
    to imply itself from absence of updates).
  - **Explicitly a timestamped snapshot, never claimed as "live"**:
    `calibration_status`/`calibration_checked_at_utc`
    (`AppState.calibration` in `state_dict()`) is documented and
    displayed as "this dashboard's own last-refreshed DISPLAY data,"
    always paired with `checked_at_utc` so a caller can judge its own
    freshness -- by design, not a bug, since manual-refresh-only
    calibration was itself a deliberate 2026-08-12 change (see above,
    "do we constantly recalibrate?" -- no). Same reasoning covers
    `od_reachable`/`od_last_state`/`od_last_error` (only ever written
    inside the same on-demand calibration-refresh call, in --http-
    fallback mode; not rendered anywhere in the dashboard UI as of this
    writing, and never presented without the same checked_at_utc
    context). `CalibrationStore.meta()`'s `source`/`checked_at_utc` is
    the same pattern one level down -- an explicit record of WHEN/HOW the
    live scoring calibration was last set, not a claim that it's been
    re-verified since.
Every status surface was checked, not just the two fixed.

TWO REAL INCIDENTS FIXED, 2026-08-12 -- read
before touching camera-tile rendering or `AppState.start_capture()`
again.

  1. **Broken-image icon on a stopped/not-yet-started camera.** The
     dashboard's camera-tile JS (`updateCameraFeeds()`; see the cameras
     section's comments for the full writeup) used to only update the small text
     status label (`cam-fetch-status-*`) on a failed
     `/api/cameras/{cam}/snapshot.png` fetch -- it never touched `img.src`
     or hid the `<img>` element itself, so a 502 (cameras not started/
     stopped, a normal, expected state now that cameras don't auto-open
     at process startup) left the browser's OWN native broken-image icon
     sitting in the tile. Fixed with a real sibling placeholder
     `<div id="cam-placeholder-{cam}">` (`.cam-placeholder` CSS below)
     toggled on every single refresh cycle, in BOTH directions -- error
     hides the `<img>` and shows the placeholder, a subsequent successful
     load re-shows the `<img>` and re-hides the placeholder -- so a camera
     that starts mid-session correctly switches back to showing real
     frames, not stuck on the placeholder forever.
  2. **Start button slow to visibly reach "Starting."**
     `AppState.start_capture()` used to set `self.capture_starting = True`
     and broadcast the first `CAPTURE_LOOP_STATUS` message only AFTER
     `await asyncio.to_thread(self.hub.open_all)` had already completed --
     i.e. after all the slow camera-opening work was already done, which
     defeats the point of a distinct "Starting" pill state. Fixed by
     moving that write + broadcast to BEFORE `hub.open_all()` is awaited
     (see that method's own docstring for the full restructure, and
     `self.capture_starting`'s own `__init__` docstring for the updated
     true/false transition list this required). Separately investigated
     (and fixed) in `opendarts/live/local_capture.py`: `LocalCameraHub.
     open_all()` itself used to open all 3 cameras SEQUENTIALLY, each with
     a real measured `open_latency_s` of ~2.2-2.3s -- opening one at a
     time cost ~6.75s all by itself, before calibration even started. Now
     opens all cameras CONCURRENTLY (one thread per camera); see that
     method's own "CONCURRENT OPEN" docstring section for the real safety
     investigation (per-camera lock granularity, the one genuine
     cross-camera dict-write hazard found and how it's guarded, what was
     concluded and why) and `tests/test_local_capture.py` for the mocked-
     delay timing proof (`max()` not `sum()`).

Both fixes are DEV-ONLY / MOCK-TESTED as of this writing -- see
`tests/test_live_server.py`/`tests/test_local_capture.py` for the real
tests. Honestly stated: neither has been validated against the real running
process on the rig (no live-system actions were taken making this
change).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import contextlib
import dataclasses
import hashlib
import html
import json
import logging
import math
import os
import platform
import queue
import shutil
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Body, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from opendarts.capture.throw_package import (
    load_ad_ground_truth,
    mark_operator_ad_wrong,
    record_throw_correction,
)
from opendarts.live import (audio, build_info, camera_names, capabilities,
                           diagnostics_gate, local_capture, vcam)
from opendarts.live import config_document
from opendarts.live import ui as next_ui
from opendarts.live.displays import DisplayError, DisplayStore
from opendarts.live.dashboard_choice import CHOICES as DASHBOARD_CHOICES, DashboardChoice
from opendarts.live import calibration_progress
from opendarts.live.config import (
    DEFAULT_VIDEO_RECORD_MODE,
    always_update,
    normalise_video_record_mode,
    read_config_section,
    set_update_on_next_restart,
    update_on_next_restart,
)
from opendarts.capture import clip as _clip_mod
from opendarts.capture import frame_ring
from opendarts.capture import throw_capture as throw_capture_mod
from opendarts.capture.frame_dump import FLIGHT_EXPECTATION_NOTE
from opendarts import disk_space
from opendarts.live.ad_ground_truth import (
    DEFAULT_AD_BASE,
    DEFAULT_MATCH_WINDOW_SEC,
    DEFAULT_TIMEOUT_SEC,
)
from opendarts.engines.registry import DEFAULT_PRIMARY_ENGINE, engine_names
from opendarts.lifecycle.settings import LifecycleSettingsStore
from opendarts.live.capture_daemon import (
    CALIBRATION_N_REPROJECTION_ATTEMPTS,
    DEFAULT_PACKAGE_ROOT,
    POLL_INTERVAL_SECONDS,
    CalibrationStore,
    CaptureLoopController,
    EngineConfigStore,
    ResetRequest,
    _reset_session_throw_numbering,
    _THROW_NUMBER_LOCK,
    bootstrap_calibrations,
    calibration_status_dict,
)
from opendarts.capture.calibration_package import DEFAULT_CALIBRATION_PACKAGE_ROOT
from opendarts.live.ad_ws_listener import AdWsListener
from opendarts.live.heap_trim import release_freed_heap
from opendarts.live.board_status import BOARD_STATUS_UNKNOWN
from opendarts.capture.trigger_state import MAX_DARTS_PER_TURN
from opendarts.live.logging_setup import DEFAULT_LOG_DIR
from opendarts.geometry.board import (
    BULL_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    SECTOR_NUMBERS_CLOCKWISE,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    sector_ring_for_point,
)
from opendarts.geometry.board_overlay import (
    DEFAULT_HIGHLIGHT_NUMBER,
    draw_calibration_overlay,
    draw_calibration_overlay_rgba,
)

log = logging.getLogger("opendarts.live.server")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# /api/restart's own delay before it actually sends the process a
# SIGTERM -- long enough that the HTTP response has genuinely finished
# flushing over the wire to the caller first (this endpoint's own point
# is to be safely callable programmatically, so the caller should get a
# real "ok" response back, not a dropped connection because the process
# died mid-response). See api_restart()'s own docstring below.
RESTART_SIGTERM_DELAY_S = 0.5

# This server's OWN scratch space for live-fetched frames -- <repo>/tmp/,
# never /tmp or the harness scratchpad, per docs/DESIGN.md's filesystem
# discipline. Distinct from DEFAULT_PACKAGE_ROOT (finished, saved throw
# packages) and from capture_daemon.py's own SCRATCH_DIR (a separate
# process's scratch space, not shared with this one).
DEFAULT_SCRATCH_DIR = REPO_ROOT / "tmp" / "live_server_scratch"

DEFAULT_PORT = 8420
DEFAULT_HOST = "0.0.0.0"

# This rig's known camera count (docs/DESIGN.md: "3-camera-120°-apart
# hardware"). Used only to know which camera ids to show snapshot tiles
# for / report calibration status for -- not load-bearing anywhere else.
N_CAMERAS = 3

# UNMEASURED placeholder, same "don't pretend a guess is a tuned number"
# discipline as capture_daemon.py's own POLL_INTERVAL_SECONDS.
DEFAULT_PACKAGE_POLL_INTERVAL_SECONDS = 4.0

# How often AppState._idle_timeout_loop below polls
# CaptureLoopController.idle_timeout_due() -- a fixed 5s cadence.
# Deliberately not derived from IDLE_TIMEOUT_SEC_DEFAULT -- a fixed poll
# stays correct regardless of the configured timeout, since it can
# be reconfigured live (see /api/idle-timeout below) without needing to
# also resize this poll cadence.
IDLE_CHECK_INTERVAL_SECONDS = 5.0

# REMOVED 2026-08-12 (was DEFAULT_CALIBRATION_POLL_INTERVAL_SECONDS =
# 20.0): this server used to re-run the full calibration pipeline
# (fresh camera frames + OpenCV PnP) on a 20s background timer, purely to
# keep the dashboard's displayed numbers fresh. The rule now: calibrate
# once at the beginning and score every dart with that; a manual
# recalibrate replaces and stores it. That auto-poll was ALSO never
# connected to actual scoring (which always used whatever calibration the
# capture loop bootstrapped once at startup) -- so it was pure overhead
# with a side effect of implying "live" freshness that was misleading
# about what real scoring uses. Calibration now updates at exactly two
# moments, both real: once at process startup (opendarts.live.capture_daemon.
# run_capture_loop_body's own bootstrap, unchanged) and on-demand via the
# "Refresh calibration now" button (AppState.refresh_calibration below),
# which -- new as of this same change -- now actually replaces the LIVE
# calibration the capture loop scores against (see CalibrationStore), not
# just refreshing a display number. (2026-08-22: the separate durable-
# record-to-disk step this comment used to describe here,
# save_calibration_snapshot(), was removed as genuinely dead code --
# nothing ever read it back, and calibration_package_root already
# persists a fuller, actually-replayable record unconditionally on
# every real calibration event.)


# AUDIO CLIENTS -- see AppState.__init__'s own `audio_clients` comment.
# A cap so a long-running process's in-memory dict does not grow across
# many page loads, plus an expiry: a stale audio report is actively
# misleading. "The TV is speaking" from a tab that closed forty minutes
# ago is worse than no line at all, because the thing the panel exists to
# reveal is a screen that has gone quiet.
#
# 45s = three missed heartbeats at the client's 15s interval. One missed
# beat is a Wi-Fi hiccup or a backgrounded tab; three is gone. Long
# enough that a healthy client never flickers off the list, short enough
# that a closed tab does not haunt the panel for a whole leg.
AUDIO_CLIENT_MAX = 20
AUDIO_CLIENT_STALE_S = 45.0

# GET /api/logs/{name} -- read-only tail of one of this project's live
# log files. Must
# match exactly the `log_name` each CLI entrypoint's own
# configure_console_and_file_logging() call uses (opendarts/live/server.py,
# opendarts/live/run_product.py, opendarts/live/capture_daemon.py) -- a
# hardcoded allowlist, not "any filename", so this endpoint can never be
# used to read an arbitrary file off disk.
# GET /api/live/recent -- how many COMPLETED visits the retail catch-up
# returns. The /api/live socket deliberately carries no history (its
# `hello` is a snapshot of now), so a scoreboard reconnecting mid-match
# had the current visit and nothing before it.
#
# An IN-MEMORY ring, deliberately NOT derived from saved packages. A first cut read the package cache and
# was wrong for three reasons: POST /api/packages/delete-all would have
# silently wiped a client's match history, pulling the corpus off the rig
# (routine) would have done the same, and with package storage disabled
# entirely (see `store_packages` in data/config.json) there would be
# no history at all. A scoreboard's replay must not depend on an
# operational decision about disk.
#
# The trade, stated: this dies on a server restart, where a
# package-derived one would survive. That is the right way round --
# clearing the corpus is routine and deliberate, a mid-match restart is
# rare and usually means the match is interrupted anyway. A client that
# genuinely wants archival history should read /api/packages.
RETAIL_RECENT_VISITS_MAX = 12

VALID_LOG_NAMES = ("server", "run_product", "capture_daemon")
DEFAULT_LOG_TAIL_LINES = 200

# How long AppState._live_event_loop's queue.SimpleQueue.get() blocks
# before giving up and looping again (raising queue.Empty, harmless). Real
# fix for a confirmed shutdown hang, not a tuning knob -- see
# _live_event_loop's own docstring for the full mechanism. Short enough
# that shutdown is never meaningfully delayed by it, long enough not to
# busy-loop.
_LIVE_EVENT_POLL_TIMEOUT_S = 0.5

# -- MJPEG camera preview stream (/api/cameras/{cam}/stream.mjpg) ------
#
# Every number here exists to protect the capture loop, not the preview.
# This runs on a 4-core machine where the pump already logs "dropped N
# pump cycles" under load, so the preview's CPU budget is "whatever is
# demonstrably left over": scoring correctness always wins over preview
# smoothness (the project's own priority call for this feature).
#
# MJPEG_MAX_FPS caps how often each connection encodes a JPEG, and the
# stream generator additionally skips any tick where the camera's own
# frame_count has not advanced -- so a camera pumping at 8fps costs 8
# encodes/sec, never 12 re-encodes of the same frame. 12 was chosen as
# "smooth enough to see a hand enter frame" (the actual complaint with
# the old 3-second snapshot poll), well below typical camera rates.
MJPEG_MAX_FPS = 12.0
# Preview frames are downscaled to this width before JPEG encode when the
# camera runs wider. Encode cost scales with pixel count and the Config
# tab renders these tiles a few hundred px wide -- shipping full 1920px
# frames would spend ~4x the CPU on pixels the browser immediately throws
# away. Full resolution stays available on snapshot.png/overlay.png,
# which is where anyone actually inspecting detail already goes.
MJPEG_MAX_WIDTH = 960
# 75 is deliberately mid-grade: this is a live preview, not evidence.
# Lossless (the PNG the snapshot endpoint uses) costs several times the
# encode CPU per frame and an order of magnitude more bandwidth.
MJPEG_JPEG_QUALITY = 75
# A stream whose camera stops producing NEW frames for this long closes
# itself rather than holding the connection open serving a frozen picture
# as if it were live -- the client's placeholder ("no signal") is the
# honest rendering of that state, and its retry tick will reconnect if
# frames come back.
MJPEG_STALL_TIMEOUT_S = 5.0
# Hard ceilings on concurrent MJPEG connections. SEPARATE BUDGETS, because
# the two kinds of stream have opposite priorities and a shared budget lets
# the cosmetic one starve the load-bearing one.
#
# A PREVIEW is a picture in a dashboard. A TRANSPORT stream (?full=1) is
# another machine's CAMERA FEED -- that consumer scores darts from it, and
# refusing it does not degrade a view, it stops a rig working.
#
# WHAT THE OLD SHARED CAP OF 8 ACTUALLY DID, measured on the real rig: the
# ceiling counted CONNECTIONS while every viewer costs one per camera, so
# with three cameras it admitted two viewers and refused the third. A
# consuming machine plus the rig's own dashboard is already six; one more
# browser tab and the next stream 503s. The rig sat refusing every new stream
# with "too many preview streams open (8 max)" while its own dashboard
# showed blank tiles -- and, because a refusal was only a response body and
# never a log line, with nothing in the log to say so.
#
# Sized in VIEWERS rather than connections, so the numbers stay meaningful
# if a rig ever runs more or fewer than three cameras.
MJPEG_MAX_PREVIEW_VIEWERS = 4
MJPEG_MAX_TRANSPORT_CONSUMERS = 4


def _mjpeg_cap(n_cameras: int, viewers: int) -> int:
    """Connection ceiling for `viewers` independent viewers of a rig with
    `n_cameras` cameras, plus one camera-set of slack so a reconnecting
    viewer is never refused by the connections it is itself replacing --
    a browser reopening a tab briefly holds both."""
    return max(1, n_cameras) * (viewers + 1)


# Multipart boundary token for multipart/x-mixed-replace. Any string
# works as long as it never appears in the JPEG payload framing; naming
# it once here keeps the media_type header and the part framing in sync.
MJPEG_BOUNDARY = "opendarts-mjpeg-frame"

# Quality for `?full=1` transport streams. Higher than the preview's,
# because these frames get SCORED: JPEG artefacts around a dart's edge move
# the tip a consumer detects.
#
# MEASURED ON REAL BOARD FRAMES (30 corpus captures at 1280x720, re-encoded
# at each setting; mean pixel 95/255, so a typical lit board rather than a
# bright or a black one):
#
#     q75  119 KB/frame   ->  29.3 Mbps/camera  ->   88 Mbps for three
#     q85  144 KB/frame   ->  35.3 Mbps/camera  ->  106 Mbps for three
#
# So q85 over q75 costs ~18 Mbps for a three-camera set. Still worth it
# against encoding artefacts into a scoring input, but it is a real slice
# of a LAN, not a rounding error -- FOUR consuming machines is ~424 Mbps,
# which is where a gigabit switch starts to matter.
#
# AN EARLIER VERSION OF THIS COMMENT CLAIMED q75 ~59KB / q85 ~69KB AND
# "7 Mbps, which is nothing". Those are almost exactly half the real
# figures, and the q85 number matches this file's own PREVIEW encode at
# 960x540 -- the measurement was taken on downscaled preview frames and
# written down as full resolution. The conclusion survived; the basis for
# it did not. Reproduce with cv2.imencode over data/packages/*/*/cam*_
# frame.png at the two qualities rather than trusting this block.
MJPEG_TRANSPORT_QUALITY = 85

# How often a TRANSPORT stream re-checks for a new frame. NEVER ZERO.
#
# The first version of `?full=1` used 0 to mean "no pacing", which made the
# generator `await asyncio.sleep(0)` -- a yield that reschedules
# immediately. With no new frame to send, that is a busy-wait ON THE EVENT
# LOOP, once per camera per consumer. Measured consequence: POST /api/stop
# took ~45 seconds, because the request had to be served by the same event
# loop three spinning generators were monopolising.
#
# 1/120s polls about four times faster than a 30fps camera produces, so it
# never paces the stream, while costing one attribute read per tick instead
# of a core. Same lesson as the capture loop's own stall backoff (1eeae8f):
# a loop with nothing to do must still wait.
MJPEG_TRANSPORT_POLL_S = 1.0 / 120.0


class MarkAdWrongRequest(BaseModel):
    """Request body for POST /api/packages/{session}/{throw_id}/mark-ad-wrong
    -- see that route and AppState.mark_ad_wrong for the full mechanism.
    ``wrong`` defaults True (the common "mark it" click); the dashboard's
    "Unmark" button sends ``wrong: false`` explicitly to undo a mistaken
    flag."""

    wrong: bool = True
    # Optional[str], not "str | None" -- pydantic v2 evaluates a
    # BaseModel's own annotations at class-definition time (unlike plain
    # function signatures elsewhere in this file, which `from __future__
    # import annotations` defers as strings and Python itself never
    # evaluates), and this repo's .venv is Python 3.9: the PEP 604 `X | Y`
    # syntax isn't valid there without the `eval_type_backport` package,
    # which isn't a dependency here. Confirmed by a real failure
    # (TypeError: unsupported operand type(s) for |) before this fix.
    note: Optional[str] = None

    # "AD was wrong -- and THIS is what was actually right". `confirmed_source` is either an engine name from this
    # throw's own sections or the literal "manual"; `confirmed_sector`/
    # `confirmed_ring` are the segment itself in
    # opendarts.geometry.board.sector_ring_for_point's vocabulary. All
    # Optional (same pydantic-v2-on-Python-3.9 reason as `note` above) and
    # all default None, so the pre-modal request shape
    # ({"wrong": true} / {"wrong": false}) stays valid unchanged -- the
    # dashboard's Unmark button still sends exactly that.
    #
    # confirmed_sector legitimately stays None for a bull/outer_bull/
    # outside confirmation (that IS sector_ring_for_point's own return
    # shape -- no sector applies there), so `confirmed_ring` is the field
    # that actually signals "a confirmation was made", not sector.
    confirmed_source: Optional[str] = None
    confirmed_sector: Optional[str] = None
    confirmed_ring: Optional[str] = None


class CorrectThrowRequest(BaseModel):
    """Request body for
    POST /api/visits/{visit_id}/throws/{index}/correct -- see that route
    and AppState.correct_throw(). Added 2026-08-14 alongside the visit
    model.

    `ring` is REQUIRED and `sector` is optional, deliberately, because
    that is `opendarts.geometry.board.sector_ring_for_point`'s own return
    shape: a bull/outer_bull/outside answer has a real ring and a
    legitimately-None sector. Ring is also what
    `_operator_truth_for()` keys "has a human confirmed this throw?" off,
    so a correction with no ring would silently record nothing.

    All optional fields are `Optional[str]` rather than `str | None` for
    the same real pydantic-v2-on-Python-3.9 reason documented on
    MarkAdWrongRequest above.
    """

    ring: str
    sector: Optional[str] = None
    # Who is asserting this. Defaults to "manual" -- a human typed the
    # segment in -- matching the exact vocabulary
    # `operator_confirmed_source` already uses for the Scoring tab's
    # hand-entered case (an engine NAME is the other real value there).
    source: str = "manual"
    note: Optional[str] = None


@dataclasses.dataclass
class _OperatorTruth:
    """A truth object built from a human's confirmed answer, shaped to be
    a drop-in for `AdGroundTruth` at `_match_fields_for_section()`'s own
    duck-typed interface (`.matched`/`.sector`/`.ring`/`.tip_xy_mm`, the
    only four attributes it reads).

    Deliberately NOT an `AdGroundTruth`: this isn't AD's ground truth and
    must never be mistaken for it on disk or in a log -- it's a human's
    assertion about the same throw, and the only thing it's used for is
    being passed to `_match_fields_for_section()`.

    `tip_xy_mm` is always None: a human confirming "Talos had the right
    segment" asserts a SEGMENT, not a millimetre coordinate.
    `_match_fields_for_section()` already handles that -- its
    `if board_xy and ad_gt.tip_xy_mm` guard leaves tip_distance_mm at
    None -- so an operator-confirmed throw honestly shows no tip delta
    rather than a fabricated one.
    """

    sector: "str | None"
    ring: "str | None"
    matched: bool = True
    tip_xy_mm: "tuple[float, float] | None" = None


def _operator_truth_for(ad_gt: "AdGroundTruth | None") -> "_OperatorTruth | None":
    """The human-confirmed truth for a throw, or None when no human has
    confirmed one (the overwhelmingly common case -- every package that
    predates this feature, and every throw nobody flagged).

    Keys off `operator_confirmed_ring`, not `_sector`: a confirmed
    bull/outer_bull/outside has a real ring and a legitimately-None
    sector (see `opendarts.geometry.board.sector_ring_for_point`), so sector
    can't distinguish "no confirmation" from "confirmed, no sector
    applies". Returning None here is what makes the un-confirmed path
    byte-identical to its pre-2026-08-13 behavior: callers fall straight
    back to the real `ad_gt`.
    """
    if ad_gt is None:
        return None
    ring = getattr(ad_gt, "operator_confirmed_ring", None)
    if not ring:
        return None
    return _OperatorTruth(
        sector=getattr(ad_gt, "operator_confirmed_sector", None),
        ring=ring,
    )


def _ring_probe_radii_mm() -> list[float]:
    """One radius inside each of `sector_ring_for_point()`'s own bands --
    the midpoint of each band, plus one point past the double outer.
    Every value is computed from board.py's own regulation radius
    constants, never a magic number. Factored out 2026-08-14 so
    `board_ring_names()` and `board_sectorless_rings()` below probe the
    IDENTICAL set of points rather than keeping two copies that could
    drift.
    """
    return [
        0.0, # bull
        (BULL_RADIUS_MM + OUTER_BULL_RADIUS_MM) / 2.0, # outer_bull
        (OUTER_BULL_RADIUS_MM + TREBLE_INNER_RADIUS_MM) / 2.0, # single_inner
        (TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2.0, # treble
        (TREBLE_OUTER_RADIUS_MM + DOUBLE_INNER_RADIUS_MM) / 2.0, # single_outer
        (DOUBLE_INNER_RADIUS_MM + DOUBLE_OUTER_RADIUS_MM) / 2.0, # double
        DOUBLE_OUTER_RADIUS_MM * 1.1, # outside
    ]


def board_sectorless_rings() -> set[str]:
    """The rings for which `sector_ring_for_point()` returns NO sector
    (bull / outer_bull / outside, as of today) -- DERIVED by asking that
    function, exactly like `board_ring_names()` derives the ring
    vocabulary itself, rather than hardcoding the three names here.

    Used by POST /api/visits/{visit_id}/throws/{index}/correct to reject
    an impossible correction (a sector on a bull, or a treble with no
    sector) before it's written as a "truth" no engine answer could ever
    equal.
    """
    return {
        ring
        for r in _ring_probe_radii_mm()
        for sector, ring in [sector_ring_for_point(0.0, r)]
        if sector is None
    }


def board_ring_names() -> list[str]:
    """The real ring vocabulary, inner-to-outer, for the Scoring tab's
    "None of these -- enter manually" picker.

    DERIVED by probing `opendarts.geometry.board.sector_ring_for_point()` at
    the midpoint of each of its own radius bands (plus one point past the
    double outer), rather than hardcoding a list of ring-name strings
    here. That module is the single source of truth for this vocabulary
    and it exposes no ring-name constant to import; a hardcoded copy in
    this file would be a second, silently-drifting definition of the same
    thing -- and a human confirming a segment through this dashboard must
    be picking from exactly the strings opendarts's own scorer produces, or
    the confirmed truth could never compare equal to any engine's answer.

    Every probe radius is computed from board.py's own regulation radius
    constants, never a magic number (see `_ring_probe_radii_mm()`).
    """
    names: list[str] = []
    for r in _ring_probe_radii_mm():
        # Straight up the +Y axis (sector 20's own center) -- the ring a
        # point falls in is purely radial, so the angle is irrelevant here.
        _, ring = sector_ring_for_point(0.0, r)
        if ring not in names:
            names.append(ring)
    return names


def _match_fields_for_section(
    section: "dict[str, Any] | None", ad_gt: "AdGroundTruth | _OperatorTruth | None"
) -> tuple[bool | None, float | None]:
    """(sector_match, tip_distance_mm) for ONE engine's result section
    against the throw's TRUTH object -- AD's own ground truth normally,
    or an `_OperatorTruth` when a human has confirmed what was actually
    right (2026-08-13; `discover_packages()` below decides which, this
    function is deliberately agnostic and unchanged by that) -- factored
    out (docs/ENGINES.md: "same
    match/tip-delta comparison against AD, just keyed to that engine's
    section... instead of hardcoding the primary") so
    `discover_packages()` computes this identically for the primary
    (top-level `result.json` fields) and for every entry under
    `other_engines`, instead of two separate copies of the same logic
    that could silently drift apart. `section` is any dict with `sector`/
    `ring`/`board_xy_mm` keys -- the top-level `result` dict and an
    `other_engines[<name>]` dict both qualify, no adaptation needed.
    Returns `(None, None)` whenever a real comparison isn't possible
    (no section, no matched AD ground truth, or no tip_xy_mm) -- same
    honest-null convention `discover_packages()` already used before this
    was factored out.
    """
    if section is None or ad_gt is None or not ad_gt.matched:
        return None, None
    sector_match = section.get("sector") == ad_gt.sector and section.get("ring") == ad_gt.ring
    tip_distance_mm = None
    board_xy = section.get("board_xy_mm")
    if board_xy and ad_gt.tip_xy_mm:
        tip_distance_mm = math.hypot(board_xy[0] - ad_gt.tip_xy_mm[0], board_xy[1] - ad_gt.tip_xy_mm[1])
    return sector_match, tip_distance_mm


# Standardized board-status vocabulary -- see opendarts/live/board_status.py
# for the full design (a separate module, not defined here, specifically
# so ad_ws_listener.py can import it without a circular import back into
# this module).


def disk_usage_for(path: "Path") -> "dict[str, Any] | None":
    """Free/total bytes on the volume holding `path`, or None if it cannot
    be read.

    Reported because throw packages are the one thing here that grows
    without bound -- and a rig that fills its disk does not fail at the
    boundary of the thing that filled it. It fails at the next write on
    the SCORING path, which is the worst possible place to discover a
    capacity problem.

    Measured against the directory rather than the mount point: the
    packages directory is what actually consumes the space, and asking
    about it directly means no assumption about how a rig has laid out
    its volumes.

    None on failure rather than zeros: a real 0 bytes free is an
    emergency, and it must not be indistinguishable from "could not
    read", which is a non-event.
    """
    import shutil

    # NEAREST EXISTING ANCESTOR, not the path itself. The packages directory
    # is only created when the first throw is saved, and disk_usage raises
    # on a path that does not exist -- so every freshly set-up rig reported
    # "unknown" until someone threw a dart. Measured 2026-09-16: three of
    # the fleet's machines showed unknown, and all three simply had no
    # data/packages yet. Walking up to a directory that does exist measures
    # the volume those packages WILL be written to, which is the question
    # being asked.
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        usage = shutil.disk_usage(str(probe))
    except Exception: # noqa: BLE001 -- a diagnostic must never break the page
        return None

    # THE GUARD'S OWN ANSWER, NOT A SECOND ONE (2026-09-17). There is a
    # free-space floor (`opendarts.disk_space`, config key
    # `min_free_disk_gb`) and BOTH big writers already obey it: below the
    # floor the capture daemon skips the throw package and a ring dump is
    # refused outright. Until now its numbers existed only inside the
    # refusal a caller happened to receive, so the dashboard could report
    # "12 GB free" while packages were silently not being written.
    #
    # `check_free_space` is asked here rather than the comparison being
    # redone with a `>` -- a second implementation of the same rule is a
    # second implementation to disagree, and the one that shows on the
    # screen is the one that must match the one that refuses.
    #
    # ONE READING feeds both halves: the `shutil.disk_usage` above is
    # handed straight to the guard via `free_bytes_fn`, so the percentage
    # in the row and the verdict beside it cannot come from two readings
    # taken a moment apart.
    free = int(usage.free)
    check = disk_space.check_free_space(
        probe,
        floor_gb=read_config_section("min_free_disk_gb"),
        free_bytes_fn=lambda _p: free,
    )
    guard = check.as_dict()
    # `ok` from the guard means "a write may proceed", which includes the
    # stand-aside case where the guard is switched off. `below_floor` is
    # the narrower question the screen is asking -- somebody really
    # looked, and the answer was no.
    guard["below_floor"] = bool(check.enabled and not check.ok)
    guard["floor_label"] = disk_space.format_gb(check.floor_bytes)
    guard["free_label"] = disk_space.format_gb(free)

    return {
        "total_bytes": int(usage.total),
        "free_bytes": int(usage.free),
        "used_bytes": int(usage.used),
        "free_pct": round(usage.free / usage.total * 100, 1) if usage.total else None,
        "guard": guard,
    }


def memory_info() -> "dict[str, Any] | None":
    """Physical RAM: total, available, and used, or None if it cannot be
    read.

    Reported for the same reason disk is (`disk_usage_for`): a rig that is
    quietly swapping or near its RAM ceiling does not fail where the memory
    went, it fails as latency on the scoring path -- so the number belongs
    next to disk on the Info tab where someone chasing "why is this rig
    slow" will look first.

    No psutil: the fleet runs a deliberately minimal, headless dependency
    set, so this reads each platform's own source directly -- /proc/meminfo
    on Linux, sysctl+vm_stat on macOS, GlobalMemoryStatusEx on Windows.
    `source` carries WHICH of those answered, so a wrong-looking number can
    be traced to how it was measured rather than guessed at -- and any
    failure returns None (never a fabricated zero: 0 bytes free is an
    emergency and must not be confused with "could not read").

    "Available" is what a new allocation can actually use, which on Linux
    and Windows the OS reports directly (MemAvailable / ullAvailPhys). macOS
    has no single such number, so it is approximated as free + inactive +
    speculative pages; `available_is_estimate` flags that so nobody reads
    the macOS figure with more precision than it has.
    """
    import platform as _platform

    try:
        system = _platform.system()
        if system == "Linux":
            fields: dict[str, int] = {}
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    key, _, rest = line.partition(":")
                    parts = rest.split()
                    if parts and parts[0].isdigit():
                        fields[key] = int(parts[0]) * 1024  # kB -> bytes
            total = fields.get("MemTotal")
            avail = fields.get("MemAvailable")
            if not total or avail is None:
                return None
            return _memory_dict(total, avail, "proc-meminfo", False)

        if system == "Darwin":
            import re
            import subprocess

            # Total from sysconf, not `sysctl hw.memsize`: no subprocess, and
            # it works where a restricted PATH/sandbox leaves sysctl empty.
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            vm = subprocess.run(
                ["vm_stat"], capture_output=True, text=True, timeout=3, check=True,
            ).stdout
            page = 4096
            m = re.search(r"page size of (\d+) bytes", vm)
            if m:
                page = int(m.group(1))
            pages: dict[str, int] = {}
            for line in vm.splitlines():
                key, _, rest = line.partition(":")
                digits = rest.strip().rstrip(".")
                if digits.isdigit():
                    pages[key.strip()] = int(digits)
            free = pages.get("Pages free", 0)
            inactive = pages.get("Pages inactive", 0)
            spec = pages.get("Pages speculative", 0)
            avail = (free + inactive + spec) * page
            if not total:
                return None
            return _memory_dict(total, avail, "sysctl-vm_stat", True)

        if system == "Windows":
            import ctypes

            class _MEMSTATUS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMSTATUS()
            stat.dwLength = ctypes.sizeof(_MEMSTATUS)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return None
            if not stat.ullTotalPhys:
                return None
            return _memory_dict(
                int(stat.ullTotalPhys), int(stat.ullAvailPhys),
                "GlobalMemoryStatusEx", False)
    except Exception:  # noqa: BLE001 -- a diagnostic must never break the page
        return None
    return None


def _memory_dict(total: int, available: int, source: str,
                 estimate: bool) -> "dict[str, Any]":
    available = max(0, min(available, total))
    used = total - available
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": used,
        "used_pct": round(used / total * 100, 1) if total else None,
        "available_is_estimate": estimate,
        "source": source,
    }


# CPU utilisation is a RATE, not a level: it only exists between two readings.
# Rather than block the /api/state handler with a sample-sleep, we keep the
# previous (busy, total) tick counts and diff against them on the next call --
# so the sampling window IS the poll interval (the Info tab re-polls every few
# seconds), and the endpoint never sleeps. `_cpu_prev` is that one-slot memory.
# First call after start has no baseline, so it honestly returns None (the row
# shows "measuring…") rather than a fabricated 0%.
_cpu_prev: "dict[str, int]" = {}


def _cpu_ticks() -> "tuple[int, int] | tuple[None, None]":
    """(busy, total) CPU tick counts on this platform, or (None, None) if
    unreadable. Each platform's own source, no psutil: /proc/stat on Linux,
    Mach host_statistics on macOS, GetSystemTimes on Windows."""
    import platform as _platform

    try:
        system = _platform.system()
        if system == "Linux":
            with open("/proc/stat", encoding="ascii") as fh:
                parts = fh.readline().split()
            if not parts or parts[0] != "cpu":
                return None, None
            vals = [int(x) for x in parts[1:]]
            idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
            total = sum(vals)
            return total - idle, total

        if system == "Darwin":
            import ctypes

            class _CpuLoad(ctypes.Structure):
                _fields_ = [("user", ctypes.c_uint), ("system", ctypes.c_uint),
                            ("idle", ctypes.c_uint), ("nice", ctypes.c_uint)]

            libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
            host = libc.mach_host_self()
            info = _CpuLoad()
            count = ctypes.c_uint(4)  # HOST_CPU_LOAD_INFO_COUNT
            # host_statistics(host, HOST_CPU_LOAD_INFO=3, &info, &count)
            if libc.host_statistics(host, 3, ctypes.byref(info),
                                    ctypes.byref(count)) != 0:
                return None, None
            busy = info.user + info.system + info.nice
            return busy, busy + info.idle

        if system == "Windows":
            import ctypes

            class _FT(ctypes.Structure):
                _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

            idle, kern, user = _FT(), _FT(), _FT()
            if not ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return None, None
            q = lambda ft: (ft.hi << 32) | ft.lo
            # kernel time INCLUDES idle, so total = kernel + user.
            total = q(kern) + q(user)
            return total - q(idle), total
    except Exception:  # noqa: BLE001 -- a diagnostic must never break the page
        return None, None
    return None, None


def cpu_load() -> "dict[str, Any] | None":
    """System-wide CPU utilisation since the previous call, or None until a
    baseline exists (first call) or if it cannot be read.

    Reported beside memory and disk because "is this rig stressed right now"
    is one question with three answers, and CPU is the one that was missing.
    `source` names how it was measured; `busy_pct` is over ALL cores (100%
    means every core saturated). Cross-platform via `_cpu_ticks`; None on
    failure, never a fabricated number.
    """
    import platform as _platform

    busy, total = _cpu_ticks()
    if busy is None or total is None:
        return None
    src = {"Linux": "proc-stat", "Darwin": "mach-host_statistics",
           "Windows": "GetSystemTimes"}.get(_platform.system(), "?")
    prev_busy = _cpu_prev.get("busy")
    prev_total = _cpu_prev.get("total")
    _cpu_prev["busy"], _cpu_prev["total"] = busy, total
    if prev_busy is None or prev_total is None:
        return {"busy_pct": None, "source": src, "cores": os.cpu_count()}
    dt = total - prev_total
    if dt <= 0:
        return {"busy_pct": None, "source": src, "cores": os.cpu_count()}
    pct = round(max(0.0, min(1.0, (busy - prev_busy) / dt)) * 100, 1)
    return {"busy_pct": pct, "source": src, "cores": os.cpu_count()}


def _package_record(meta_path: Path) -> "dict[str, Any] | None":
    """One package's listing record, read from its meta.json, result.json
    and ad_ground_truth.json -- or None when meta.json is missing or not
    readable yet (a package mid-write). Split out of discover_packages() so
    a single saved or edited package can be refreshed without re-reading
    the whole archive (2026-09-17)."""
    pkg_dir = meta_path.parent
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    result: dict[str, Any] | None = None
    result_path = pkg_dir / "result.json"
    if result_path.exists():
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            result = None

    # AD (Autodarts) ground truth -- None if ad_ground_truth.json isn't
    # on disk (most packages, until the backfill CLI or the on-demand
    # refresh button runs against them). See docs/DESIGN.md's "Dashboard
    # wiring spec" section and opendarts/live/ad_ground_truth.py for the
    # fetch + time-window match this file is loading back off disk.
    ad_gt = load_ad_ground_truth(pkg_dir)

    # WHAT EVERY ENGINE ROW IS GRADED AGAINST. When a human has confirmed
    # what was actually right on this throw, THAT is the truth every
    # engine's sector_match/tip_distance_mm is computed against --
    # not AD's own answer, which the same human just said was wrong.
    # An engine that was genuinely right shouldn't keep being scored
    # as a miss against known-bad AD data.
    #
    # `_operator_truth_for()` returns None whenever no confirmation
    # exists, so `truth` is the identical `ad_gt` object this code
    # always passed for every un-confirmed package -- the untouched
    # path is unchanged, by construction rather than by parallel
    # branches (proven directly in tests/test_live_server.py).
    truth = _operator_truth_for(ad_gt) or ad_gt

    primary_match, primary_tip_distance = _match_fields_for_section(result, truth)

    # Multi-engine scoring (docs/ENGINES.md) -- `other_engines` is
    # only present once opendarts.capture.throw_package.
    # write_other_engines_result() has run (a package with no
    # also-run engines configured at capture time, or one saved
    # before this framework existed, simply has no such key -- {}
    # here is the correct, honest default, not an error). Each
    # engine's match/tip-delta REUSES `_match_fields_for_section()`
    # -- the SAME function used for the primary fields two lines
    # above -- so the also-run columns can never compute AD agreement
    # differently than the primary column does.
    other_engines_raw = (result.get("other_engines") or {}) if result is not None else {}
    engines: dict[str, Any] = {}
    for engine_name, section in other_engines_raw.items():
        sector_match, tip_distance_mm = _match_fields_for_section(section, truth)
        engines[engine_name] = {
            "ok": section.get("ok"),
            "sector": section.get("sector"),
            "ring": section.get("ring"),
            "board_xy_mm": section.get("board_xy_mm"),
            "reason": section.get("reason"),
            "timed_out": section.get("timed_out", False),
            "sector_match": sector_match,
            "tip_distance_mm": tip_distance_mm,
        }

    return (
        {
            "primary_engine": (result.get("primary_engine") if result is not None else None),
            "engines": engines,
            "session": meta.get("session"),
            "throw_id": pkg_dir.name,
            "path": str(pkg_dir),
            "captured_at_utc": meta.get("captured_at_utc"),
            # AD answer latency for THIS throw, signed milliseconds:
            # AD's arrival minus our own answer time, straight off
            # ad_ground_truth.json's `staleness_sec`. NEGATIVE means
            # AD's answer arrived before ours; positive means after.
            # Throw-level, not per-engine -- the four
            # engines run concurrently against one capture and share
            # a single answer instant, so there is no honest
            # per-engine value to report here.
            #
            # Deliberately read off `ad_gt`, NOT `truth`: an operator
            # confirmation changes what was CORRECT, never when AD
            # actually replied, and the operator stand-in carries no
            # timing of its own. None whenever AD had no matched
            # event -- absent, never a fabricated 0.
            "ad_latency_ms": (
                round(ad_gt.staleness_sec * 1000.0, 1)
                if ad_gt is not None and ad_gt.staleness_sec is not None
                else None
            ),
            "cameras": meta.get("cameras", []),
            # Whether this throw was RECORDED (a window clip out of the
            # frame ring, opendarts.capture.clip). The dashboard hides the
            # "Save frames" button when it is True -- the frames are
            # already stored -- and the viewer plays the clip.
            #
            # NOT "does meta have a video block" any more: since
            # 2026-09-22 every package has one, because a package that is
            # not a recording stores its two frames as a two-frame stills
            # clip (written just after its data, see
            # opendarts.capture.clip). Asking the
            # old question would call every throw "recorded" and hide the
            # button on exactly the throws that need it. is_recorded_clip
            # reads the block's `kind`, and treats a block without one
            # (every package from before stills clips existed) as the
            # recording it always was.
            "has_video": _clip_mod.is_recorded_clip(meta.get("video")),
            # Which turn this dart belonged to, and which dart of that
            # turn it was (0-based) -- added 2026-08-14 with the visit
            # model. Both honestly None for every package written
            # before it existed (see
            # opendarts.capture.throw_package.save_throw_package, which
            # omits the keys entirely rather than writing nulls).
            # This is also the lookup POST /api/visits/{visit_id}/
            # throws/{index}/correct resolves through, which is why it
            # can correct a throw from an ALREADY-CLOSED visit --
            # unlike a correction API that only works while
            # the visit is still open in memory.
            "visit_id": meta.get("visit_id"),
            "visit_index": meta.get("visit_index"),
            "ok": result.get("ok") if result is not None else None,
            "sector": result.get("sector") if result is not None else None,
            "ring": result.get("ring") if result is not None else None,
            "board_xy_mm": result.get("board_xy_mm") if result is not None else None,
            "reason": result.get("reason") if result is not None else None,
            "n_cameras_used": result.get("n_cameras_used") if result is not None else None,
            "ad_matched": ad_gt.matched if ad_gt else None,
            "ad_match_reason": ad_gt.match_reason if ad_gt else None,
            "ad_sector": ad_gt.sector if ad_gt else None,
            "ad_ring": ad_gt.ring if ad_gt else None,
            "ad_tip_xy_mm": list(ad_gt.tip_xy_mm) if (ad_gt and ad_gt.tip_xy_mm) else None,
            "ad_method": ad_gt.ad_method if ad_gt else None,
            # Human-asserted "AD was wrong on this throw" flag -- see
            # opendarts.capture.throw_package.mark_operator_ad_wrong.
            # False (not None) when ad_gt is None: "no package has ever
            # been flagged" is a real, honest default, unlike
            # ad_matched's None (which means "AD ground truth was
            # never even attempted") -- the operator flag has exactly
            # two states (flagged / not flagged), no third "unknown".
            "ad_operator_marked_wrong": bool(ad_gt.operator_marked_wrong) if ad_gt else False,
            "ad_operator_note": ad_gt.operator_note if ad_gt else None,
            # The human's "...and THIS was actually right" answer (see
            # _operator_truth_for above). None on every package nobody
            # confirmed -- which is what the Scoring tab's AD row keys
            # off to decide whether to show a confirmed-truth badge,
            # and what the modal pre-selects from on a re-open.
            "ad_operator_confirmed_source": (
                ad_gt.operator_confirmed_source if ad_gt else None
            ),
            "ad_operator_confirmed_sector": (
                ad_gt.operator_confirmed_sector if ad_gt else None
            ),
            "ad_operator_confirmed_ring": (
                ad_gt.operator_confirmed_ring if ad_gt else None
            ),
            # Derived comparison fields -- computed HERE (not stored on
            # disk), so they can never drift from the two raw sources.
            # See _match_fields_for_section() below -- SAME function
            # used for every also-run engine's own columns above, so
            # this (the primary/top-level result) and any also-run
            # engine's section are held to identical match logic.
            "sector_match": primary_match,
            "tip_distance_mm": primary_tip_distance,
        }
    )

def discover_packages(package_root: Path) -> list[dict[str, Any]]:
    """Real, working listing of saved throw packages -- reads each
    package's meta.json + result.json (never the raw PNGs; this is a
    lightweight index, not a full package load -- see
    opendarts.capture.throw_package.load_throw_package for that). Package
    layout is <package_root>/<session_id>/<throw_id>/meta.json, matching
    opendarts/capture/throw_package.py's on-disk format.

    Tolerant of a missing/partial package (corrupt JSON, mid-write) --
    skips it rather than crashing the whole listing, since a dashboard
    reading a directory that a concurrent writer (e.g. a live
    capture_daemon.py process) is actively touching is an expected race,
    not a bug.
    """
    packages: list[dict[str, Any]] = []
    if not package_root.exists():
        return packages

    for meta_path in package_root.glob("*/*/meta.json"):
        record = _package_record(meta_path)
        if record is not None:
            packages.append(record)

    _sort_packages(packages)
    return packages


def _sort_packages(packages: "list[dict[str, Any]]") -> None:
    """Newest first. captured_at_utc is an ISO-8601 string (see
    save_throw_package) so lexicographic sort == chronological sort."""
    packages.sort(key=lambda p: p["captured_at_utc"] or "", reverse=True)


def _tail_log_lines(log_path: Path, n: int) -> list[str]:
    """Last `n` lines of `log_path`, or [] if the file doesn't exist yet
    (an entrypoint that's never been run in this environment -- not an
    error, see api_logs' own docstring). Reads the whole file rather than
    seeking from the end -- these are single log files for one long-running
    process, not rotated/huge, so simplicity wins over a partial-read
    optimization that isn't needed yet.
    """
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    lines = text.splitlines()
    return lines[-n:] if n > 0 else lines



# ---------------------------------------------------------------------
# Retail wire vocabulary (WS /api/live)
# ---------------------------------------------------------------------
# The retail channel speaks a FIXED wire vocabulary that is deliberately
# NOT this project's internal ring vocabulary. Three real differences,
# translated here rather than leaked to clients:
#
#   internal            wire         sector      label
#   ------------------  -----------  ----------  -----------------
#   "outside"           "miss"       0           "MISS"
#   "bull"              "bull"       25          "BULL"
#   "outer_bull"        "outer_bull" 25          "25"
#   ok=False (no answer) ""          0           "failed to score"
#
# `""` and `"miss"` are genuinely different and must never be collapsed:
# `"miss"` is a confident "off the board", `""` is an abstention.
# `single_inner`/`single_outer` also stay distinct on the wire.
RETAIL_RING_MISS = "miss"
RETAIL_RING_NO_SCORE = ""
_RETAIL_MULTIPLIER = {"single_inner": 1, "single_outer": 1, "treble": 3, "double": 2}


def retail_score_value(sector: int, ring: str) -> int:
    """Points for one dart, in the retail wire vocabulary."""
    if ring == RETAIL_RING_NO_SCORE:
        return 0
    if ring == "bull":
        return 50
    if ring == "outer_bull":
        return 25
    if ring == RETAIL_RING_MISS:
        return 0
    return sector * _RETAIL_MULTIPLIER.get(ring, 0)


def retail_dart_fields(sector: object, ring: object, ok: object) -> dict[str, Any]:
    """Translate one internal scoring answer into the retail wire shape
    (`label`/`sector`/`ring`/`value`). Never raises on odd input -- an
    unparseable sector degrades to the no-answer shape rather than
    breaking the live feed."""
    if not ok or ring is None:
        return {"label": "failed to score", "sector": 0,
                "ring": RETAIL_RING_NO_SCORE, "value": 0}
    ring_s = str(ring)
    if ring_s == "outside":
        return {"label": "MISS", "sector": 0, "ring": RETAIL_RING_MISS, "value": 0}
    if ring_s == "bull":
        return {"label": "BULL", "sector": 25, "ring": "bull", "value": 50}
    if ring_s == "outer_bull":
        return {"label": "25", "sector": 25, "ring": "outer_bull", "value": 25}
    try:
        sector_i = int(sector)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {"label": "failed to score", "sector": 0,
                "ring": RETAIL_RING_NO_SCORE, "value": 0}
    prefix = {"treble": "T", "double": "D"}.get(ring_s, "S")
    return {
        "label": f"{prefix}{sector_i}",
        "sector": sector_i,
        "ring": ring_s,
        "value": retail_score_value(sector_i, ring_s),
    }

def retail_dart_from_package(pkg: dict[str, Any]) -> dict[str, Any]:
    """One package row -> its retail dart fields, honouring a correction.

    A corrected throw reports the CORRECTION, never the engine's original
    answer -- the same precedence the dashboard's own throwCall() applies.
    Without this a client on /api/live keeps displaying a value the server
    already knows is wrong, with nothing on that channel to contradict it.

    Shared by the live snapshot and the catch-up endpoint on purpose: two
    copies of this precedence would eventually disagree, and the failure
    would be a wrong score shown to a viewer.
    """
    corrected_ring = pkg.get("corrected_ring")
    fields = retail_dart_fields(
        pkg.get("corrected_sector") if corrected_ring else pkg.get("sector"),
        corrected_ring or pkg.get("ring"),
        True if corrected_ring else pkg.get("ok"),
    )
    fields["captured_at_utc"] = pkg.get("captured_at_utc")
    if corrected_ring:
        fields["corrected"] = True
    return fields


class AppState:
    """Shared, mutable server state -- one instance per create_app() call
    (constructor-injected, never a module-level singleton), so tests can
    spin up isolated instances against a temp package root instead of
    fighting DEFAULT_PACKAGE_ROOT. A hub-shaped object (background
    tasks + a WebSocket client set + a _broadcast helper) holding
    opendarts's own calibration/package data.
    """

    def __init__(
        self,
        package_root: Path,
        scratch_dir: Path,
        n_cameras: int,
        package_poll_interval_s: float,
        hub: local_capture.LocalCameraHub | None = None,
        live_event_queue: "queue.SimpleQueue[dict[str, Any]] | None" = None,
        host: str | None = None,
        port: int | None = None,
        ad_base_url: str = DEFAULT_AD_BASE,
        ad_window_sec: float = DEFAULT_MATCH_WINDOW_SEC,
        ad_timeout_s: float = DEFAULT_TIMEOUT_SEC,
        calibration_store: "CalibrationStore | None" = None,
        reset_request: "ResetRequest | None" = None,
        controller: "CaptureLoopController | None" = None,
        engine_config_store: "EngineConfigStore | None" = None,
        lifecycle_settings_store: "LifecycleSettingsStore | None" = None,
        ad_ws_listener: "AdWsListener | None" = None,
        vcam_set: Any = None,
        vcam_set_factory: Any = None,
        reprojection_targets_px: dict[int, float] | None = None,
        throw_capture: Any = None,
        capture_root: "Path | None" = None,
    ) -> None:
        self.package_root = package_root
        # SPOKEN DART CALLS HAVE NO SERVER-SIDE SETTINGS ANY MORE
        # (2026-09-15). There used to be `audio_enabled` and
        # `audio_voice` here, read from and written back to config.json's
        # "audio" section. Both went when playback moved into the
        # browser: on/off, volume and voice are now per-device values in
        # each browser's localStorage, because the whole point of the
        # move is that the iPad beside the board and the TV above it are
        # different listeners with different needs (see
        # opendarts/live/audio.py's docstring). A rig-wide switch would
        # be back to one mute for every screen in the house.
        #
        # What the server still has is `audio_clients` below -- not a
        # setting, a REPORT.
        self.scratch_dir = scratch_dir
        self.n_cameras = n_cameras
        self.package_poll_interval_s = package_poll_interval_s
        # AD (Autodarts) ground-truth config -- the base URL, match
        # window and timeout the WebSocket listener
        # (opendarts/live/ad_ws_listener.py) is built with, reported on
        # /api/state and rewritten when the operator changes it. The
        # fields discover_packages() surfaces are read off each
        # package's own ad_ground_truth.json, never fetched here.
        self.ad_base_url = ad_base_url
        self.ad_window_sec = ad_window_sec
        self.ad_timeout_s = ad_timeout_s
        # The shared, mutable calibration reference the capture loop
        # actually scores against (opendarts.live.capture_daemon.
        # CalibrationStore) -- None in this module's own standalone CLI
        # (no capture loop shares this process at all, see module
        # docstring), a real shared instance when this AppState was built
        # by opendarts/live/run_product.py's combined entrypoint. A manual
        # "Refresh calibration now" writes the freshly-recalibrated data
        # here (see refresh_calibration() below) so the NEXT scored throw
        # actually uses it, not just what this dashboard displays.
        self.calibration_store = calibration_store
        # The shared, mutable reset signal the dashboard's Reset button
        # writes to (POST /api/reset -> api_reset() below) and the capture
        # loop's background thread polls once per iteration -- see
        # opendarts.live.capture_daemon.ResetRequest's own docstring for the
        # full thread-safety reasoning (mirrors calibration_store above
        # exactly, deliberately the same established pattern). None in
        # this module's own standalone CLI (no capture loop shares this
        # process, same as calibration_store's own None case) means a
        # POST here would have nothing to signal -- api_reset() reports
        # that honestly rather than pretending it did something.
        self.reset_request = reset_request
        # The shared Start/Stop/idle-timeout coordinator -- see
        # opendarts.live.capture_daemon.CaptureLoopController's own
        # docstring. None in this module's own standalone CLI (same None
        # case as calibration_store/reset_request above) means
        # /api/start-/api/stop report honestly that no capture loop
        # exists in this process to control.
        self.controller = controller
        # The shared, mutable multi-engine scoring config (docs/ENGINES.md)
        # -- same established pattern/reasoning as calibration_store/
        # reset_request/controller immediately above: None in this
        # module's own standalone CLI (no capture loop shares this
        # process), a real shared opendarts.live.capture_daemon.
        # EngineConfigStore when built by opendarts/live/run_product.py's
        # combined entrypoint. READ-ONLY: the engine set is configured in
        # data/config.json and reported on /api/state; there is no
        # runtime mutation endpoint.
        self.engine_config_store = engine_config_store
        # Operator-tunable lifecycle Detection time (dart_stable_frames) --
        # same established pattern as engine_config_store above. None in
        # this module's own standalone CLI (no capture loop shares this
        # process); a real shared LifecycleSettingsStore when built by
        # opendarts/live/run_product.py's combined entrypoint.
        self.lifecycle_settings_store = lifecycle_settings_store
        # The Autodarts WS listener -- 2026-08-14, the Scoring
        # tab's Autodarts status indicator light. Always running whenever it
        # exists at all (no on/off toggle, no durable enabled flag to
        # load). AppState reads its board_status() live on every
        # state_dict() call rather than caching a copy here -- one source
        # of truth, no staleness to manage. None in this module's own
        # standalone CLI, same as every other shared object above.
        self.ad_ws_listener = ad_ws_listener
        # Windows virtual-camera publishers, when republishing is on.
        # Held only so /api/frame-health can report whether the consumer is
        # collecting what we publish -- the capture path writes to these
        # through the hub's frame_sink, never through here.
        self.vcam_set = vcam_set
        # How to BUILD that set when the Autodarts toggle asks for
        # publishing and there is none yet -- `callable() -> set | None`,
        # supplied by run_product (macOS and an opted-out rig answer
        # None). Before this existed the toggle could only attach a set
        # that startup had already built, so turning the comparison on
        # later did nothing visible and needed a restart.
        self.vcam_set_factory = vcam_set_factory
        # Display-only, for the Config tab -- NOT load-bearing anywhere
        # (uvicorn is actually bound by main()/run_product.py's own CLI
        # args; this is just so the dashboard can honestly show what it
        # was started with instead of leaving the field blank). None when
        # the caller (e.g. tests) didn't pass one.
        self.host = host
        self.port = port
        # Per-camera calibration reprojection-error targets, sourced from opendarts.live.config.LiveConfig at the CLI
        # entrypoint. `None`/empty (the default -- every existing caller,
        # every existing test) means every camera keeps using
        # opendarts.live.capture_daemon.CALIBRATION_TARGET_REPROJECTION_ERROR_PX
        # uniformly, exactly as before this field existed. Consumed by
        # _refresh_calibration_blocking() below (the manual "Refresh
        # calibration now" path); the Start-triggered auto-calibrate path
        # gets the SAME LiveConfig value independently, threaded through
        # opendarts/live/run_product.py's own reprojection_targets_px=
        # parameter straight to run_capture_loop_body() -- not read off
        # this AppState, since that path runs on a different thread with
        # no AppState reference of its own.
        self.reprojection_targets_px = reprojection_targets_px or {}

        # The throw-capture service (opendarts.capture.throw_capture), or
        # None in the standalone-server configuration where no capture
        # loop is running in this process and therefore nothing is filling
        # a ring. Held by REFERENCE and built by run_product before either
        # half starts, so the ring a route triggers a dump from is the
        # same object the pump is filling -- the same discipline
        # calibration_store/controller already follow.
        #
        # None is reported honestly by the routes below rather than
        # papered over: "no ring is attached to this process" and "the
        # ring is empty" are different facts, and collapsing them is how
        # an operator concludes the feature is broken when it was never
        # switched on.
        self.throw_capture = throw_capture

        # Where this rig's frame-ring captures live on disk. Normally
        # nobody passes this: the capture service owns the real root and
        # `capture_root` below reads it off the service, so the directory
        # the Delete control empties is provably the directory the writer
        # fills. The override exists for the standalone server (no
        # service at all, yet still the thing an operator points a
        # browser at) and as the injection seam tests use.
        self._capture_root_override = (
            Path(capture_root) if capture_root is not None else None
        )

        # Frame source: a persistent
        # LocalCameraHub, either injected by the caller (tests; a
        # pre-configured hub) or opened by create_app()'s lifespan at
        # real-server startup and closed at shutdown -- never reopened
        # per request, same "hub is a long-lived object" design point as
        # capture_daemon.py's own hub. `owns_hub` tracks whether THIS
        # AppState is responsible for closing it (false when a caller
        # injected an already-managed hub, e.g. a test's own fixture).
        self.hub: local_capture.LocalCameraHub | None = hub
        self.owns_hub = False

        # Live count of open MJPEG preview connections (see
        # /api/cameras/{cam}/stream.mjpg and _mjpeg_cap()). Plain
        # int, no lock: every increment/decrement happens on the event
        # loop (the endpoint coroutine and the stream generator's
        # finally), never from a worker thread, so there is nothing to
        # race with. Kept on AppState rather than a closure so tests can
        # read and preload it.
        self.mjpeg_client_count: int = 0
        # Transport (?full=1) consumers are counted SEPARATELY from
        # previews -- see MJPEG_MAX_PREVIEW_VIEWERS. A dashboard must not
        # be able to refuse another machine's camera feed.
        self.mjpeg_transport_count: int = 0
        # One encode per frame, shared by every consumer of it.
        self.jpeg_cache = _SharedJpegCache()

        # "Has the server begun shutting down?", supplied by whoever owns
        # the uvicorn Server (see run_product._build_components).
        #
        # WHY NOT THE EXISTING stop_event OR THE LIFESPAN. Neither fires in
        # time. On Ctrl-C, uvicorn's OWN signal handler runs first and
        # starts draining connections; run_product's handler -- the one
        # that sets stop_event -- is not reached until uvicorn re-raises
        # the signal as server.run() returns, and the lifespan shutdown
        # runs later still. Both are AFTER the drain this exists to let
        # finish. uvicorn's own `should_exit` is the earliest honest
        # signal available, so that is what this reads.
        #
        # A callable rather than a flag because the Server is built after
        # the app it serves, so there is no object to hand over yet at
        # construction time.
        self.should_exit_check: "Any" = None

        self.clients: set[WebSocket] = set()

        # Retail feed (WS /api/live) -- see the retail plumbing below.
        self._retail_subscribers: set["asyncio.Queue"] = set()
        self._last_retail_state_fields: "dict[str, Any] | None" = None
        self._last_retail_event_name: "str | None" = None
        self._last_retail_status: "str | None" = None

        self._packages_cache: list[dict[str, Any]] = discover_packages(package_root)
        self._known_package_paths: set[str] = {p["path"] for p in self._packages_cache}

        # WHICH SCREENS IN THE ROOM ARE ACTUALLY SPEAKING, self-reported
        # by each browser (POST /api/audio/clients), keyed by a random
        # per-page-load tab id (the dashboard's `debugTabId`).
        #
        # WHY THE SERVER TRACKS THIS AT ALL, having just given up every
        # other opinion about audio: moving playback into the browser
        # introduced one failure mode the server side never had --
        # autoplay policy. A TV mounted above the board gets its one
        # permitting tap at setup and then silently loses it on any
        # reload (a crash, a Wi-Fi reconnect, an overnight browser
        # update). The TV knows it is blocked; nobody standing at the
        # oche does, and nobody can reach the TV to find out. So each
        # client says how it is doing, and every OTHER dashboard can show
        # it. Someone on an iPad sees that the TV went quiet.
        #
        # A REPORT, never a control: nothing here is read back to decide
        # what to broadcast. The phrase goes to every client regardless,
        # and each one decides for itself whether to make a noise.
        #
        # Bounded and self-expiring the same way, and for the same
        # reason, as the debug snapshots: a tab that closed is not a
        # device that went quiet, it is a device that is not there, and
        # after AUDIO_CLIENT_STALE_S of silence it stops being listed.
        self.audio_clients: dict[str, dict[str, Any]] = {}

        # Calibration status -- None/empty until the
        # first background poll (or a manual refresh_calibration() call)
        # actually completes. /api/state reports this honestly as
        # "checked_at_utc: null" rather than pretending a value exists
        # before it's real.
        self.calibration_error: str | None = None
        #: Set when a calibration found the cameras had moved on the ring and
        #: relearned the rig's layout instead of refusing (see
        #: capture_daemon's _learn_ring_geometry_from_ring_correlation).
        #: Carried to the dashboard so the operator learns a camera moved.
        self.calibration_geometry_relearned: "dict[str, Any] | None" = None
        self.calibration_status: dict[int, dict[str, Any]] = {}
        self.calibration_checked_at_utc: str | None = None

        # Real event push (see module docstring). None (the default --
        # this module's own standalone CLI) means "no live process to
        # push from" -- trigger state stays honestly unavailable/null and
        # this server relies solely on _package_poll_loop below. Non-None
        # (opendarts/live/run_product.py's combined entrypoint) means a
        # background thread elsewhere in the SAME process is pushing
        # real TRIGGER_STATE/PACKAGE_SAVED events onto this queue --
        # _live_event_loop (started only in that case, see
        # start_background_tasks) consumes and broadcasts them.
        self.live_event_queue: "queue.SimpleQueue[dict[str, Any]] | None" = live_event_queue
        self.live_events_enabled = live_event_queue is not None
        self.trigger_state: str | None = None
        self.trigger_last_event_utc: str | None = None
        # How many darts opendarts.capture.trigger_state.ThrowTriggerState has
        # captured so far THIS turn (0-3) -- added 2026-08-12 alongside
        # capture_daemon.py's own on_event payload change, purely so the
        # header status pill can show "dart 2 of 3" instead of just the
        # bare state name. None until the first real TRIGGER_STATE event
        # arrives (or forever, in standalone mode) -- same honest-null
        # convention as trigger_state itself.
        self.trigger_dart_count: int | None = None

        # THE VISIT (turn) MODEL, added 2026-08-14 -- see
        # opendarts.live.capture_daemon.new_visit_id() for where a visit ID
        # is actually minted/rotated (this server never mints one; it
        # only ever mirrors what the capture loop pushed, same honest
        # division of responsibility as trigger_state above). All three
        # stay None/empty forever in standalone mode (no capture loop in
        # this process to push a visit at all), which /api/state reports
        # honestly rather than inventing an empty visit.
        self.visit_id: str | None = None
        # The THROW_DETECTED payloads for the CURRENT visit, in the order
        # they were scored -- index N here is the dart whose
        # `visit_index` is N. Cleared on every VISIT_CLEARED. Bounded by
        # construction (a visit is at most MAX_DARTS_PER_TURN darts, and
        # the list is dropped wholesale when the visit rotates), so it
        # needs no maxlen.
        self.visit_throws: list[dict[str, Any]] = []
        # The scoring page's board photo (opendarts.live.board_photo): the
        # latest empty-board JPEG and a short content hash, which is all a
        # screen needs to know whether it is looking at the current one.
        # None until the first dart of the session.
        self.board_photo_jpeg: bytes | None = None
        self.board_photo_version: str | None = None
        # Completed-visit ring for GET /api/live/recent (see
        # RETAIL_RECENT_VISITS_MAX). Appended when a visit closes, so it
        # is independent of whether throw packages are ever written to
        # disk.
        self._recent_visits: "collections.deque[dict[str, Any]]" = collections.deque(
            maxlen=RETAIL_RECENT_VISITS_MAX
        )

        # Real Start/Stop lifecycle detail -- added 2026-08-12 alongside
        # the expanded status-pill vocabulary (separate ready / starting /
        # takeout / stopped states). `capture_starting` covers the real window between a
        # genuinely-new POST /api/start being ACCEPTED (controller exists,
        # not already running, hub exists -- i.e. start_capture()'s three
        # early guards all pass) and the capture thread's own FIRST
        # TRIGGER_STATE event actually landing -- i.e. "opening cameras +
        # auto-calibrating," which was previously invisible (the pill had
        # no way to distinguish it from either Stopped or an already-live
        # Ready). Set True in start_capture() below the moment those three
        # guards pass -- BEFORE `hub.open_all()` is even awaited, not
        # after.
        # Cleared False on any of THREE real transitions, not two: (1) a
        # real TRIGGER_STATE event arrives (_handle_live_event -- the loop
        # is now definitively live), (2) the session ends for any reason
        # (stop_capture()), or (3), added alongside the early-broadcast
        # fix, the start attempt itself fails (zero cameras opened) --
        # this transition did not need to exist before the fix, since the
        # old code only ever set this field True AFTER confirming
        # open_all() had already succeeded, so a failed start never set it
        # True in the first place; now that it is set True BEFORE knowing
        # the outcome, the failure path must explicitly clear it back to
        # False itself, or a failed Start would leave the pill stuck
        # showing "Starting" forever with no TRIGGER_STATE or stop_capture()
        # ever coming along to clear it. `capture_last_start_error` is the
        # honest failure detail for the Stopped pill state -- e.g. "no cameras opened,"
        # surfaced right on the pill instead of only in the sidebar's
        # status pill.
        self.capture_starting: bool = False
        self.capture_last_start_error: str | None = None
        # CAPTURE-STATUS ORDERING STAMP -- added 2026-09-22 for the
        # "pill stuck on Waiting . Starting until a browser refresh" bug.
        # Read `_capture_status_stamp()` below before removing either
        # field. Every serialized snapshot of the Start/Stop status (the
        # `/api/state` capture_loop section, each CAPTURE_LOOP_STATUS and
        # TRIGGER_STATE broadcast, and the POST /api/start and /api/stop
        # response bodies) carries one of these, so a dashboard can tell
        # an OLDER snapshot from a NEWER one no matter what order they
        # reach it in. `_capture_status_epoch` is per-process: a restart
        # resets the counter to zero, and the client treats a new epoch
        # as "a different server now, start counting again" rather than
        # rejecting everything the new process says as older than what
        # the dead one last said.
        self._capture_status_epoch: str = os.urandom(6).hex()
        self._capture_status_seq: int = 0
        # AD CONNECTION ORDERING STAMP -- same idea, same reasons, for the
        # Config tab's "Compare against Autodarts" note. See
        # `ad_connection_snapshot()`.
        self._ad_connection_epoch: str = os.urandom(6).hex()
        self._ad_connection_seq: int = 0

        self._tasks: list[asyncio.Task] = []

    # -- packages -----------------------------------------------------

    def list_packages(self) -> list[dict[str, Any]]:
        return self._packages_cache

    def _apply_package_changes(
        self, removed: "set[str]", records: "list[dict[str, Any]]",
    ) -> list[dict[str, Any]]:
        """Update the package cache in place of a full rescan: drop
        `removed` paths, replace or add `records`, keep newest-first.
        Runs on the event loop, so the cache is never seen half-updated.
        Returns the new cache."""
        changed = removed | {r["path"] for r in records}
        current = [p for p in self._packages_cache if p["path"] not in changed]
        current.extend(records)
        _sort_packages(current)
        self._packages_cache = current
        self._known_package_paths = {p["path"] for p in current}
        return current

    def _broadcast_packages(self, changed: "list[dict[str, Any]]") -> "list[dict[str, Any]]":
        """The `packages` a PACKAGES_UPDATED carries: the changed packages,
        then the newest 20 as it always has -- from the cache, not disk.
        The changed ones lead because an edit to an older throw fell
        outside the newest 20 and never reached open tabs."""
        seen = {p["path"] for p in changed}
        return list(changed) + [p for p in self._packages_cache[:20] if p["path"] not in seen]

    async def _refresh_package(self, pkg_dir: "Path | str") -> "dict[str, Any] | None":
        """Re-read ONE package and fold it into the cache. None when it is
        gone or not readable yet (it is then dropped from the cache).

        PER PACKAGE, 2026-09-17. Every saved throw used to re-read the whole
        archive -- twice (once at save, once when the also-run engines
        finish) -- on the event loop's queue, so the next dart's events
        waited behind it; ~0.17 ms per package, 1.7 s of CPU a dart at
        5,000 packages."""
        pkg_dir = Path(pkg_dir)
        record = await asyncio.to_thread(_package_record, pkg_dir / "meta.json")
        self._apply_package_changes({str(pkg_dir)}, [record] if record else [])
        return record

    async def _package_poll_loop(self) -> None:
        """The safety net for packages written or removed by anything other
        than this process's own events. Lists paths only; a package's JSON
        is read only when its path is new -- the poll used to parse every
        package's files every few seconds to learn that nothing changed."""
        def _paths() -> "set[str]":
            if not self.package_root.exists():
                return set()
            return {str(m.parent) for m in self.package_root.glob("*/*/meta.json")}

        while True:
            await asyncio.sleep(self.package_poll_interval_s)
            try:
                paths = await asyncio.to_thread(_paths)
                if paths == self._known_package_paths:
                    continue
                added = sorted(paths - self._known_package_paths)
                removed = self._known_package_paths - paths
                records = await asyncio.to_thread(
                    lambda: [r for r in (_package_record(Path(a) / "meta.json") for a in added) if r]
                )
            except Exception as exc: # noqa: BLE001 -- a bad poll must not kill the loop
                log.warning("package poll failed: %s", exc)
                continue

            current = self._apply_package_changes(removed, records)
            log.info(
                "package poll: %d package(s) total, %d new, %d removed",
                len(current), len(records), len(removed),
            )
            await self._broadcast(
                {
                    "type": "PACKAGES_UPDATED",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "count": len(current),
                    "new_count": len(records),
                    "packages": self._broadcast_packages(records),
                }
            )

    # -- operator "AD was wrong" flag (POST /api/packages/{session}/{throw_id}/mark-ad-wrong) --

    async def mark_ad_wrong(
        self,
        session: str,
        throw_id: str,
        wrong: bool,
        note: str | None,
        confirmed_source: str | None = None,
        confirmed_sector: str | None = None,
        confirmed_ring: str | None = None,
    ) -> dict[str, Any]:
        """Backend for the Scoring tab's per-row "AD was wrong" toggle --
        a HUMAN judgment call (the operator watched the real throw; opendarts
        only has two disagreeing automated readings to compare, it cannot
        decide which one is actually right). Added 2026-08-12 after a
        live throw that opendarts scored T1, correctly, while AD's
        ground truth disagreed/showed a miss.

        Durable persistence is opendarts.capture.throw_package.
        mark_operator_ad_wrong(), which extends the throw package's own
        ad_ground_truth.json (see that function's docstring for the full
        mechanism) -- NOT a
        separate parallel file. A toggle, not one-way: wrong=False
        un-marks a mistaken click.

        `confirmed_source`/`confirmed_sector`/`confirmed_ring` (2026-08-13)
        carry the answer the human picked in the Scoring tab's "Which was
        actually right?" modal -- an engine name or "manual", plus the
        segment. They're persisted by the same single
        mark_operator_ad_wrong() call (no second write, no second file),
        and are what discover_packages()/_operator_truth_for() then grade
        every engine row against instead of AD's own answer. wrong=False
        clears them along with the flag and note.

        Broadcasts PACKAGES_UPDATED afterward (same pattern as
        _handle_live_event's PACKAGE_SAVED branch) so every connected dashboard tab sees the flag live, not
        just the one that clicked -- and returns the single freshly-
        updated package dict so the CALLING tab's own click handler can
        apply it immediately via the same ingestPackages() path every
        other live update goes through, without waiting on its own
        broadcast round-trip.
        """
        pkg_dir = self.package_root / session / throw_id
        if not pkg_dir.is_dir():
            return {"ok": False, "reason": f"no such package: {session}/{throw_id}"}

        await asyncio.to_thread(
            mark_operator_ad_wrong,
            pkg_dir,
            wrong,
            note,
            confirmed_source,
            confirmed_sector,
            confirmed_ring,
        )

        updated_pkg = await self._refresh_package(pkg_dir)
        ts = datetime.now(timezone.utc).isoformat()
        await self._broadcast(
            {
                "type": "PACKAGES_UPDATED",
                "ts": ts,
                "count": len(self._packages_cache),
                "new_count": 0,
                "packages": self._broadcast_packages([updated_pkg] if updated_pkg else []),
            }
        )
        return {
            "ok": True,
            "session": session,
            "throw_id": throw_id,
            "operator_marked_wrong": (
                updated_pkg["ad_operator_marked_wrong"] if updated_pkg else wrong
            ),
            "operator_note": updated_pkg["ad_operator_note"] if updated_pkg else note,
            "operator_confirmed_source": (
                updated_pkg["ad_operator_confirmed_source"] if updated_pkg else confirmed_source
            ),
            "operator_confirmed_sector": (
                updated_pkg["ad_operator_confirmed_sector"] if updated_pkg else confirmed_sector
            ),
            "operator_confirmed_ring": (
                updated_pkg["ad_operator_confirmed_ring"] if updated_pkg else confirmed_ring
            ),
            "package": updated_pkg,
        }

    async def correct_throw(
        self,
        visit_id: str,
        index: int,
        sector: str | None,
        ring: str,
        source: str = "manual",
        note: str | None = None,
    ) -> dict[str, Any]:
        """Backend for `POST /api/visits/{visit_id}/throws/{index}/correct`
        -- a GAME DRIVER saying "that dart actually landed in <sector,
        ring>" while a game is being played. Added 2026-08-14 with the visit
        model, which is what makes it addressable at all: without a
        visit ID + an index within it, there is no stable way for a game
        driver to name "the second dart of this turn."

        **Resolved off DISK, not off in-memory visit state.**
        `discover_packages()` surfaces each package's own
        `visit_id`/`visit_index` (written into meta.json at capture
        time), and that's what's matched here. Two real consequences,
        both deliberate:
          - A throw can still be corrected AFTER its visit ended. The
            truth lives on disk, so there's no reason to lose the
            ability to fix it.
          - It works identically whether or not this server process is
            the one that captured the throw.

        **Never destroys the original engine call.** The write goes
        through `opendarts.capture.throw_package.record_throw_correction()`,
        which uses the SAME `ad_ground_truth.json` annotation path the
        Scoring tab's mark-ad-wrong flow already uses (one patching
        mechanism, not two -- see that function's docstring) and never
        touches `result.json`. Per docs/DESIGN.md's "Replay is the source of truth": the
        human's answer is recorded ALONGSIDE what the engine actually
        said, so replaying this package through newer code still
        reproduces and can be graded against the real live call.

        Broadcasts `THROW_CORRECTED` (new, additive) and then
        `PACKAGES_UPDATED` (the existing message every connected
        dashboard tab already re-renders from -- same pattern as
        `mark_ad_wrong()` above, so a correction made by a game driver
        shows up in the Scoring tab live without that tab knowing this
        endpoint exists).
        """
        def _find(packages: "list[dict[str, Any]]") -> "dict[str, Any] | None":
            return next(
                (
                    p
                    for p in packages
                    if p.get("visit_id") == visit_id and p.get("visit_index") == index
                ),
                None,
            )

        # The cache first -- it is kept current per package -- and the
        # disk only when it has not caught up with a throw yet.
        match = _find(self._packages_cache)
        if match is None:
            match = _find(await asyncio.to_thread(discover_packages, self.package_root))
        if match is None:
            return {
                "ok": False,
                "reason": (
                    f"no throw found for visit {visit_id!r} index {index} -- "
                    "either the visit/index is wrong, or that throw was captured "
                    "before the visit model existed (its package has no visit_id)"
                ),
            }

        pkg_dir = Path(match["path"])
        await asyncio.to_thread(
            record_throw_correction, pkg_dir, sector, ring, source, note
        )

        updated_pkg = await self._refresh_package(pkg_dir)

        # Keep the in-memory current-visit view consistent with what was
        # just written, so a client reading /api/state right after its own
        # correction doesn't see the pre-correction answer. Only touches
        # the throw actually corrected, and only when the correction was
        # for the visit that's still open.
        for throw in self.visit_throws:
            if throw.get("visit_id") == visit_id and throw.get("visit_index") == index:
                throw["corrected_sector"] = sector
                throw["corrected_ring"] = ring
                throw["corrected_source"] = source

        ts = datetime.now(timezone.utc).isoformat()
        await self._broadcast(
            {
                "type": "THROW_CORRECTED",
                "ts": ts,
                "visit_id": visit_id,
                "visit_index": index,
                "session": match.get("session"),
                "throw_id": match.get("throw_id"),
                "path": str(pkg_dir),
                # What the engine ACTUALLY said, unchanged on disk --
                # carried here so a consumer can see both sides of the
                # correction in one message.
                "live_sector": match.get("sector"),
                "live_ring": match.get("ring"),
                "corrected_sector": sector,
                "corrected_ring": ring,
                "source": source,
                "note": note,
            }
        )
        # RETAIL: a client that already scored this visit would otherwise
        # keep the old value forever -- nothing on that channel ever
        # contradicted it. Named explicitly rather than left to the
        # generic path: a correction changes a dart without changing
        # status, so the status-transition mapper would name no event and
        # publish nothing.
        await self.publish_retail_state("throw_corrected")
        await self._broadcast(
            {
                "type": "PACKAGES_UPDATED",
                "ts": ts,
                "count": len(self._packages_cache),
                "new_count": 0,
                "packages": self._broadcast_packages([updated_pkg] if updated_pkg else []),
            }
        )
        return {
            "ok": True,
            "visit_id": visit_id,
            "visit_index": index,
            "session": match.get("session"),
            "throw_id": match.get("throw_id"),
            "live_sector": match.get("sector"),
            "live_ring": match.get("ring"),
            "corrected_sector": sector,
            "corrected_ring": ring,
            "source": source,
            "package": updated_pkg,
        }

    # -- calibration ----------------------------------------------------

    def _refresh_calibration_blocking(self, queued_at: float | None = None) -> dict[str, Any]:
        """The actual (blocking: network/hardware + OpenCV) work, run off
        the event loop via asyncio.to_thread by refresh_calibration().
        Never raises -- every failure mode is captured into the returned
        dict so the caller can report it, matching this project's
        "graceful degrade, don't crash the server because the rig is
        unreachable/uncalibratable" requirement.

        Frame source: calibration comes straight from the local hub.

        Only ever called on-demand now (see refresh_calibration() below)
        -- there is no background timer calling this anymore.

        Returns "raw_calibrations" (the real dict[int, CameraCalibration]
        this pass produced, {} on any failure) ALONGSIDE "calibrations"
        (the lightweight display-only projection, same as before) -- the
        raw objects are what refresh_calibration() needs to actually push
        into the shared CalibrationStore and persist to disk; this method
        itself has no opinion on what happens to them.
        """
        if queued_at is not None:
            waited = time.monotonic() - queued_at
            # Only interesting when it is not instant. A thread pool with a
            # free worker hands off in microseconds; anything above a tenth
            # of a second means this request queued behind other work, and
            # that is the number nobody had.
            if waited >= 0.1:
                log.info("calibration refresh: waited %.2fs for a worker "
                         "thread before starting", waited)
        if self.hub is None:
            return {
                "calibrations": {},
                "raw_calibrations": {},
                "calibration_error": "local camera hub not initialized yet",
            }
        # 2026-08-16: refresh returns an error when the cameras are
        # off rather than trying anyway -- this is the
        # ACTUAL live path (run_product.py defaults to
        # the earlier HTTP-fallback-only
        # fix never ran on this rig at all, confirmed by checking
        # the real process). hub.grab_all() is a pure cache read (no
        # I/O, no waiting) -- a brief bounded poll here is nearly
        # free and catches a dead/never-opened camera before
        # bootstrap_calibrations()'s real up-to-200-frame retry loop
        # ever starts. Duck-typed and defensive on purpose: some
        # tests/callers hand this a bare sentinel `hub` (bootstrap
        # itself mocked, hub only needs to be non-None) -- if it
        # doesn't look like a real LocalCameraHub, skip this
        # pre-check entirely rather than crash, same graceful-
        # degrade posture this whole function already has.
        if hasattr(self.hub, "configs") and hasattr(self.hub, "grab_all"):
            n_expected = len(self.hub.configs)
            deadline = time.monotonic() + 2.0
            ready_cams: dict[int, Any] = {}
            while time.monotonic() < deadline:
                ready_cams = self.hub.grab_all()
                if len(ready_cams) >= n_expected:
                    break
                time.sleep(0.1)
            if len(ready_cams) < n_expected:
                missing = sorted(set(range(n_expected)) - set(ready_cams))
                return {
                    "calibrations": {},
                    "raw_calibrations": {},
                    "calibration_error": f"cameras not ready: no frame yet from cam(s) {missing}",
                }
        try:
            # target_reprojection_error_px omitted entirely unless a
            # real per-camera override exists -- same "don't surprise
            # a narrower test double" reasoning as
            # capture_daemon.run_capture_loop_body's identical guard.
            extra_kwargs: dict[str, Any] = (
                {"target_reprojection_error_px": self.reprojection_targets_px}
                if self.reprojection_targets_px
                else {}
            )
            # Calibration-package saving (opendarts.capture.
            # calibration_package) -- see bootstrap_calibrations()'s
            # own "CALIBRATION PACKAGE" docstring section. Opted in
            # for both real bootstrap_calibrations() call sites in
            # this app; throw_package_root=self.package_root so the
            # background save's own post-save cleanup pass can see
            # which calibration packages this app's throws reference.
            calibration_package_out: dict[str, Any] = {}
            calibs = bootstrap_calibrations(
                self.scratch_dir / "calib",
                hub=self.hub,
                    calibration_package_root=DEFAULT_CALIBRATION_PACKAGE_ROOT,
                throw_package_root=self.package_root,
                calibration_package_out=calibration_package_out,
                # BEST-OF-N REPROJECTION ATTEMPTS, wired in 2026-08-30
                # -- manual "Refresh
                # calibration now" only, per the feature's own
                # docstring recommendation (an operator-triggered,
                # infrequent action that already deliberately pays
                # similar one-time latency costs elsewhere, e.g.
                # CALIBRATION_N_FRAMES=50's own history). Start-time
                # auto-calibration (run_capture_loop_body(), a
                # separate call site) deliberately NOT touched --
                # stays at the default (1) unless/until there's real
                # evidence the extra latency is worth paying
                # automatically every Start.
                n_reprojection_attempts=CALIBRATION_N_REPROJECTION_ATTEMPTS,
                **extra_kwargs,
            )
        except Exception as exc: # noqa: BLE001 -- same graceful-degrade rule
            calibration_error = str(exc)
        else:
            calibration_error = None
        if calibration_error is not None:
            # Only now, with `exc` gone, is a FAILED calibration's frame
            # pool actually unreferenced -- its traceback pinned it through
            # bootstrap_calibrations()'s own trim. See opendarts.live.heap_trim.
            release_freed_heap("failed calibration")
            return {
                "calibrations": {},
                "raw_calibrations": {},
                "calibration_error": calibration_error,
            }
        return {
            "calibrations": calibration_status_dict(calibs, self.n_cameras),
            "raw_calibrations": calibs,
            "calibration_error": None,
            "calibration_package_id": calibration_package_out.get("package_id"),
            "ring_geometry_relearned": calibration_package_out.get("ring_geometry_relearned"),
        }

    async def refresh_calibration(self) -> None:
        """The ONLY way calibration ever updates in this server now (see
        DEFAULT_PACKAGE_POLL_INTERVAL_SECONDS's neighboring comment above
        for the removed-auto-poll writeup) -- called from the dashboard's
        "Refresh calibration now" button (POST /api/calibration/refresh)
        exclusively. Manual recalibrate must actually matter, not just
        refresh a display number: when this AppState
        is part of opendarts.live.run_product's combined process
        (self.calibration_store is not None), the freshly-computed
        calibration is pushed into that SAME shared, mutable
        CalibrationStore the capture loop reads from on every dart -- the
        very next scored throw uses it. (2026-08-22: this used to ALSO
        durably persist a standalone snapshot to disk via
        save_calibration_snapshot() -- removed as genuinely dead code,
        nothing ever read it back, and calibration_package_root already
        persists a fuller, actually-replayable record unconditionally on
        every real calibration event.) Best-effort: an unusual store
        failure must never prevent this method from at least updating
        what the dashboard displays.
        """
        if self.controller is not None:
            self.controller.touch() # a manual Calibrate counts as real activity
        # Queue time is measured separately from work time. `to_thread`
        # hands off to the default executor, which is shared with snapshot
        # fetches and anything else that offloads -- so a busy dashboard
        # can leave this request parked before a single frame is grabbed.
        # Without this line that wait is indistinguishable from slow
        # calibration, and the two have completely different fixes.
        t_queued = time.monotonic()
        data = await asyncio.to_thread(self._refresh_calibration_blocking, t_queued)
        self.calibration_status = data.get("calibrations", {})
        self.calibration_checked_at_utc = datetime.now(timezone.utc).isoformat()
        # 2026-08-16: was set on the raw dict every _refresh_calibration_
        # blocking() return path (fail-fast readiness errors included)
        # but never actually read out of it here, so it silently never
        # reached the API response -- caught by a real test proving the
        # fail-fast response body, not just that the function returns
        # early.
        self.calibration_error: str | None = data.get("calibration_error")
        self.calibration_geometry_relearned = data.get("ring_geometry_relearned")

        raw_calibrations = data.get("raw_calibrations") or {}
        if raw_calibrations:
            if self.calibration_store is not None:
                self.calibration_store.set(
                    raw_calibrations,
                    source="manual",
                    checked_at_utc=self.calibration_checked_at_utc,
                    package_id=data.get("calibration_package_id"),
                )

    async def _broadcast_calibration_status(self) -> None:
        await self._broadcast(
            {
                "type": "CALIBRATION_STATUS",
                "ts": self.calibration_checked_at_utc,
                "cameras": self.calibration_status,
                "ring_geometry_relearned": self.calibration_geometry_relearned,
            }
        )

    # -- capture-loop Start/Stop/idle-timeout (opendarts.live.run_product only) ---

    def _capture_status_stamp(self) -> dict[str, Any]:
        """A fresh, strictly increasing ordering stamp for ONE serialized
        snapshot of the capture-loop status -- merge it into every dict
        that tells a dashboard `starting`/`running`.

        WHY THIS EXISTS (2026-09-22, "pill stuck on Waiting . Starting;
        a refresh clears it"). The dashboard learns the Start/Stop status
        over TWO independent channels: the WebSocket broadcasts, and the
        HTTP response body of the tab's own POST /api/start. Nothing
        orders one channel against the other. When a valid calibration
        already exists, the capture thread skips auto-calibration and
        emits its first TRIGGER_STATE within milliseconds of
        `controller.request_start()`. `_live_event_pump` then handles it
        (clearing `capture_starting` and broadcasting) while
        `start_capture()` is still awaiting its own final broadcast, so
        the start response -- which used to hard-code `starting: True` --
        was routinely applied by the clicking tab AFTER the TRIGGER_STATE
        that should have ended Starting. The same interleave can also
        deliver start_capture()'s final CAPTURE_LOOP_STATUS (built with
        starting True before the pump ran) to some tabs after the
        pump's TRIGGER_STATE, because `_broadcast()` awaits each socket
        in turn. And once the board is idle nothing else ever clears the
        client's copy: the capture thread only emits TRIGGER_STATE on a
        state CHANGE, i.e. at the next dart.

        Fixing the order of sends would only move the race (an HTTP
        response and a WebSocket frame can always overtake each other).
        Stamping each snapshot instead makes the order irrelevant: the
        stamp is taken on the event loop in the same synchronous step as
        the state it describes is read, so a higher stamp always
        describes a later server state, and the client just drops any
        snapshot older than the newest one it has applied.

        Taken at SERIALIZATION time, not bumped on state writes: some of
        what a snapshot reports (`controller.meta()`'s `running`) is
        written by the capture thread, which never passes through here,
        so a per-write version could hand two snapshots with different
        contents the same number. A per-snapshot counter cannot."""
        self._capture_status_seq += 1
        return {"status_epoch": self._capture_status_epoch, "status_seq": self._capture_status_seq}

    def ad_connection_snapshot(self) -> dict[str, Any]:
        """`runtime.ad` of the config document: is the Autodarts listener
        wired, and is its WebSocket connected RIGHT NOW -- plus an ordering
        stamp. Every serialization of that fact goes through here: the
        GET/PATCH /api/config bodies and the AD_CONNECTION broadcast.

        WHY THE STAMP (2026-09-22, "On but NOT connected, until a
        refresh"). Switching the toggle to Yes starts the listener and the
        PATCH reply is built at once -- before the socket has had time to
        connect -- so it truthfully says `connected: false`. The listener
        then connects and AD_CONNECTION is broadcast. Those are two
        channels (an HTTP reply and a WebSocket frame) with no order
        between them: on a fast LAN the socket can connect, and the
        broadcast reach the clicking tab, before that tab has processed its
        own PATCH reply. Applied last, the older `false` would then put the
        red note straight back and nothing would correct it until the next
        transition -- the exact shape of the capture-status pill bug
        (`_capture_status_stamp()`). So each snapshot carries a counter
        taken in the SAME synchronous step as the is_connected() read, on
        the event loop, and the dashboard drops any snapshot older than the
        newest it has applied (`acceptAdConnection()` in app.js). Stamped
        per-READ, not per-transition, for the reason given there: the value
        is written by the listener's thread, which never passes through
        here.

        The epoch is per-process, so a restarted server's counter starting
        again from zero is recognised as a new baseline rather than as
        "older" than the dead process's last word.
        """
        listener = self.ad_ws_listener
        self._ad_connection_seq += 1
        stamp = {"connection_epoch": self._ad_connection_epoch,
                 "connection_seq": self._ad_connection_seq}
        if listener is None:
            return {"available": False, "connected": False,
                    "reason": "AD ground truth is not wired in this process",
                    **stamp}
        return {"available": True, "connected": listener.is_connected(), **stamp}

    async def start_capture(self) -> dict[str, Any]:
        """The real backend behind the sidebar's Reset... err, Start
        button (`POST /api/start` below) -- opens the shared camera hub
        off the event loop (`asyncio.to_thread(self.hub.open_all)`) and
        signals `self.controller` to begin a capture-loop session. Honest
        no-ops: no `self.controller` at all (this module's own standalone
        CLI -- no capture loop in this process) or a session already
        running (idempotent, not an error -- `/api/start` is safely
        re-clickable).

        Sets `self.capture_starting = True` (see that field's own
        docstring) the moment a GENUINELY NEW session is requested, and
        broadcasts a real-time `CAPTURE_LOOP_STATUS` message so every
        connected dashboard tab's status pill flips to "Starting"
        immediately -- not just the tab that clicked. A failed start (zero
        cameras opened) records the honest reason in
        `self.capture_last_start_error` so the Stopped pill state can show
        it, instead of the click silently leaving no visible trace.

        EARLY-BROADCAST FIX, 2026-08-12 -- read before reordering
        this method again. Previously, `self.capture_starting = True` and
        the first `CAPTURE_LOOP_STATUS` broadcast both happened AFTER
        `await asyncio.to_thread(self.hub.open_all)` had already
        completed -- i.e. after all the slow camera-opening work was
        already done, which defeats the entire point of a "Starting"
        pill state: it existed to cover exactly this window, but nothing
        ever told a connected dashboard the window had begun until it was
        already over. Fixed by moving the `capture_starting = True` write
        and its `CAPTURE_LOOP_STATUS` broadcast to BEFORE `hub.open_all()`
        is awaited -- immediately after the three real validation guards
        below (controller exists, not already running, hub exists) all
        pass, i.e. the earliest point a request is known to be a
        genuinely-new session worth announcing. A second, FINAL broadcast
        still goes out after `open_all()` resolves either way (success or
        the honest "no cameras opened" failure), unchanged in spirit from
        before -- so callers see two real events for one Start: "starting"
        immediately, then the actual outcome once camera-opening (and, in
        `capture_daemon.py`'s `run_capture_loop_body`, auto-calibration)
        has actually run. `self.capture_starting`'s own docstring promises
        it stays True from "a successful POST /api/start... [until] the
        capture thread's own FIRST TRIGGER_STATE event" -- this change
        does not violate that: it's still set True exactly once per new
        session (never toggled off between the early and final broadcast
        on the success path), and is explicitly cleared False on the
        failure path (mirroring what the OLD code already did on failure,
        just now needing an explicit write since the field was set True
        earlier this time instead of never on that path)."""
        if self.controller is None:
            return {"ok": False, "reason": "no capture loop in this process to start (standalone dashboard mode)"}
        # Real incident, 2026-08-17: a second POST /api/start arriving
        # while a FIRST one is still inside the ~7s `hub.open_all()` below
        # sailed straight past this guard -- `is_running()` doesn't flip
        # True until AFTER open_all() finishes (via request_start() at the
        # bottom of this method), so the window between "capture_starting
        # set True" and "is_running() becomes True" had no idempotency
        # protection at all. Two concurrent open_all() calls on the SAME
        # hub/cameras from the same process is a real, reproducible
        # trigger for a native AVFoundation/OpenCV camera-pipeline crash
        # (`PAC_EXCEPTION`/`EXC_BREAKPOINT`, confirmed via a macOS rig's
        # crash reports + `run_product.log` showing two "Start requested"
        # lines 2-3s apart immediately before every crash that night, and
        # zero crashes on the one occasion only a single Start landed).
        # `capture_starting` already covers exactly this window (see its
        # own docstring and this method's "EARLY-BROADCAST FIX" section
        # above) -- it just wasn't checked here. A second/duplicate Start
        # request (e.g. a client retrying because open_all() takes longer
        # than its own timeout) is now the same safe, idempotent no-op as
        # a genuinely-already-running session, instead of a second real
        # camera-open attempt.
        if self.controller.is_running() or self.capture_starting:
            return {
                "ok": True, "already_running": True, "starting": self.capture_starting,
                **self.controller.meta(), **self._capture_status_stamp(),
            }
        if self.hub is None:
            return {"ok": False, "reason": "no camera hub configured in this process"}

        # EARLY broadcast -- fires the instant this is known to be a
        # genuinely-new session (all three guards above passed), BEFORE
        # the slow hub.open_all()/calibration work below even starts.
        # This is the actual fix for the pill's slow-to-update complaint;
        # see this method's own docstring for the full incident.
        self.capture_last_start_error = None
        self.capture_starting = True
        await self._broadcast(
            {
                "type": "CAPTURE_LOOP_STATUS",
                "ts": datetime.now(timezone.utc).isoformat(),
                "ok": True,
                "starting": True,
                **self.controller.meta(),
                **self._capture_status_stamp(),
            }
        )

        ok_flags = await asyncio.to_thread(self.hub.open_all)
        if not any(ok_flags):
            self.capture_last_start_error = "no cameras opened"
            self.capture_starting = False
            await self._broadcast(
                {
                    "type": "CAPTURE_LOOP_STATUS",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "ok": False,
                    "reason": self.capture_last_start_error,
                    "starting": False,
                    **self.controller.meta(),
                    **self._capture_status_stamp(),
                }
            )
            return {
                "ok": False,
                "reason": "no cameras opened",
                "cameras": ok_flags,
                "hub_status": self.hub.status_report(),
            }
        self.controller.request_start()
        log.info("capture loop: Start requested (cameras opened: %s)", ok_flags)
        await self._broadcast(
            {
                "type": "CAPTURE_LOOP_STATUS",
                "ts": datetime.now(timezone.utc).isoformat(),
                "ok": True,
                "cameras": ok_flags,
                # The live field, not a literal True: nothing can have
                # cleared it between request_start() above and here (no
                # await in between, so `_live_event_pump` cannot run), so
                # today this is always True -- but it is the field that
                # is the truth, and the stamp below promises this dict
                # describes the server state at the moment it was taken.
                "starting": self.capture_starting,
                **self.controller.meta(),
                **self._capture_status_stamp(),
            }
        )
        # Built AFTER the await above, during which the capture thread's
        # first TRIGGER_STATE may well have been handled already (see
        # `_capture_status_stamp()`) -- so this must report the live
        # `capture_starting`, not the `True` it used to hard-code. That
        # literal was one half of the stuck-Starting bug: the response
        # claimed a state the server had already left.
        return {
            "ok": True, "cameras": ok_flags, "starting": self.capture_starting,
            **self.controller.meta(), **self._capture_status_stamp(),
        }

    async def stop_capture(self, *, reason: str = "manual") -> dict[str, Any]:
        """Shared by `POST /api/stop` and `_idle_timeout_loop` below --
        ONE real implementation, not two independently-maintained copies
        (this project's own established discipline, same reasoning as
        calibration_status_dict() being factored out for the analogous
        reason). Signals `self.controller` to end the current session,
        waits (bounded) for the capture thread to actually acknowledge,
        then closes the shared hub -- a bounded-wait-then-
        proceed-anyway shape (see CaptureLoopController's own
        ARCHITECTURE NOTE 2 for the honest deviation: a thread can't be
        force-cancelled the way an asyncio task can, so a timeout here
        just means "proceed and close the hub anyway, loudly logged").

        Also clears `self.capture_starting` unconditionally (a session
        ending mid-bootstrap, e.g. an idle-timeout racing a slow startup,
        must not leave the pill stuck showing "Starting" forever) and
        broadcasts `CAPTURE_LOOP_STATUS` (see start_capture()'s own
        docstring for why -- same "every connected tab sees it live"
        reasoning, not just the tab that clicked).

        STATUS-HONESTY FIX, 2026-08-12 -- also resets `self.trigger_state`/
        `self.trigger_dart_count` to honest-null here (see module's dated
        docstring entry for the full audit this is part of, and
        opendarts.live.local_capture.LocalCameraHub.close_all's own fix for
        the sibling incident this generalizes). Root cause: `trigger_state`
        is ONLY ever written by a real TRIGGER_STATE event pushed from the
        capture thread (`_handle_live_event` below) -- when a session ends
        (manual Stop or idle-timeout), `mark_session_ended()`
        (opendarts/live/capture_daemon.py's CaptureLoopController) flips
        `running` to False but pushes no event of its own, so
        `trigger_state` just sat at whatever throw-in-progress state it
        last saw (e.g. `TAKEOUT_WAITING`) FOREVER after the loop actually
        stopped -- a stopped capture loop claiming, via raw `/api/state`,
        to still be mid-takeout. The dashboard's own JS pill already
        masked this in practice (`renderPill()`'s `cl && !cl.running`
        branch takes priority over the trigger's last-known state, added
        alongside the two-axis pill work) -- but that's a rendering
        workaround, not a fix to the underlying data `/api/state` actually
        returns to any caller, dashboard or otherwise. Fixed at the
        source: every path that stops a session goes through this one
        method, so resetting here (rather than patching the JS further)
        follows the same "closing something must WRITE its new state, not
        rely on a client masking the stale read" principle as the camera
        fix. `trigger_last_event_utc` is stamped with the real stop time
        (not left at the old throw event's timestamp) so a caller
        computing "how fresh is this" gets an honest answer either way."""
        if self.controller is None:
            return {"ok": False, "reason": "no capture loop in this process to stop"}
        if not self.controller.is_running():
            return {
                "ok": True, "already_stopped": True, "starting": self.capture_starting,
                **self.controller.meta(), **self._capture_status_stamp(),
            }
        self.controller.request_stop()
        acked = await asyncio.to_thread(self.controller.stopped_ack.wait, 2.0)
        if not acked:
            log.warning(
                "stop (%s): capture thread did not acknowledge session end within 2.0s -- "
                "closing the camera hub anyway (a frame grab racing hub.close_all() below "
                "is possible in this narrow edge case, same accepted tradeoff shutdown() "
                "makes in opendarts/live/run_product.py)",
                reason,
            )
        if self.hub is not None:
            await asyncio.to_thread(self.hub.close_all)
        self.capture_starting = False
        ts = datetime.now(timezone.utc).isoformat()
        self.trigger_state = None
        self.trigger_dart_count = None
        self.trigger_last_event_utc = ts
        log.info("capture loop: stopped (%s), camera hub closed", reason)
        # CAPTURE_LOOP_STATUS (running: False) is broadcast BEFORE the
        # TRIGGER_STATE reset below, deliberately -- renderPill()'s own
        # `cl && !cl.running` branch takes priority over trig.state once
        # it sees this, so ordering it first means the client never
        # transiently renders the "no TRIGGER_STATE has arrived yet"
        # CONNECTING label in the brief window between the two messages;
        # it goes straight from whatever it was showing to Stopped.
        await self._broadcast(
            {
                "type": "CAPTURE_LOOP_STATUS",
                "ts": ts,
                "ok": True,
                "reason": reason,
                "acknowledged": acked,
                "starting": False,
                **self.controller.meta(),
                **self._capture_status_stamp(),
            }
        )
        await self._broadcast(
            {
                "type": "TRIGGER_STATE",
                "ts": ts,
                "state": None,
                "session": None,
                "dart_count": None,
                "capture_starting": False,
                "reason": f"capture loop stopped ({reason})",
                **self._capture_status_stamp(),
            }
        )
        return {
            "ok": True, "acknowledged": acked, "reason": reason, "starting": False,
            **self.controller.meta(), **self._capture_status_stamp(),
        }

    async def _idle_timeout_loop(self) -> None:
        """Auto-stop after `self.controller.idle_timeout_sec` of no real
        activity -- a 5s poll
        cadence (IDLE_CHECK_INTERVAL_SECONDS), with "only matters while a
        session is actually running, `<= 0` disables it" semantics (see
        CaptureLoopController.idle_timeout_due()). Only started when
        self.controller is not None (see start_background_tasks below) --
        this module's own standalone CLI has no capture loop to time out."""
        while True:
            await asyncio.sleep(IDLE_CHECK_INTERVAL_SECONDS)
            if self.controller is not None and self.controller.idle_timeout_due():
                log.info(
                    "capture loop: idle timeout reached (%ds with no activity) -- auto-stopping",
                    self.controller.get_idle_timeout_sec(),
                )
                await self.stop_capture(reason="idle_timeout")
                # FREE THE RING, 2026-09-17. Stopping capture leaves its
                # frames in memory for as long as the rig sits idle -- up to
                # ~1.3 GB on a rig that holds pixels -- and after this long
                # without activity none of them can be a dart anyone will
                # ask about. A manual Stop keeps them: "capture the dart I
                # just missed" right after Stop is a real use.
                ring = getattr(self.throw_capture, "ring", None)
                if ring is not None:
                    try:
                        ring.clear()
                        log.info("frame ring cleared after the idle timeout")
                    except Exception:  # noqa: BLE001 -- never break the stop
                        log.exception("clearing the frame ring after the idle timeout failed")
                await self._broadcast(
                    {
                        "type": "IDLE_TIMEOUT",
                        "idle_timeout_sec": self.controller.get_idle_timeout_sec(),
                    }
                )

    # -- real live events (opendarts.live.run_product only) -----------------

    async def _live_event_loop(self) -> None:
        """Consumes real events pushed from another thread in this SAME
        process (opendarts/live/run_product.py's capture-loop thread) via
        `self.live_event_queue`, a thread-safe `queue.SimpleQueue` --
        REAL event push, not this server's own polling fallback
        (_package_poll_loop above -- calibration has no poll loop at all
        anymore, see AppState.refresh_calibration's own docstring). Only
        started
        when live_event_queue is not None (see start_background_tasks) --
        this module's own standalone CLI never starts this task, since it
        has no queue and nothing would ever be put on it.

        **BOUNDED read, not an unbounded blocking one -- this is a real,
        confirmed shutdown-hang fix, not defensive styling.** An earlier
        version called `asyncio.to_thread(self.live_event_queue.get)` with
        no timeout. `asyncio.to_thread` runs that on a worker thread from
        the event loop's default ThreadPoolExecutor; cancelling the
        *asyncio* task awaiting it (e.g. via AppState.stop_background_tasks
        on lifespan shutdown) does NOT stop the real underlying OS thread
        -- `queue.SimpleQueue.get()` with no timeout has no way to be
        interrupted, so that thread stays blocked forever once there's
        nothing left to consume (the common case: idle between throws).
        On Python <3.12, `uvicorn.Server.run()`'s own asyncio.run()
        cleanup (`loop.shutdown_default_executor()`) unconditionally
        `ThreadPoolExecutor.shutdown(wait=True)`s that same pool -- joining
        every worker thread, including the orphaned one stuck in
        `queue.get()` -- which hangs `uvicorn.Server.run()` itself
        forever, before control even returns to
        opendarts/live/run_product.py's own `main()`/`shutdown()`. Confirmed
        via a real reproduction (real subprocess, real SIGINT, real
        thread-stack dump via faulthandler) that this hangs the process
        indefinitely -- with or without a live WebSocket client connected
        (see tests/test_run_product.py's
        test_real_signal_with_open_websocket_exits_within_bounded_time for
        the regression test). Using `queue.SimpleQueue.get(timeout=...)`
        instead means the worker thread reliably returns (raising
        `queue.Empty`, harmless, just loop again) at least once every
        `_LIVE_EVENT_POLL_TIMEOUT_S` seconds even with nothing to
        consume, so it's never blocked long enough to matter -- both to
        asyncio cancellation (this task's own `while True` loop simply
        exits next time it checks) and, more importantly, to
        `shutdown_default_executor()`'s real thread join.
        """
        assert self.live_event_queue is not None
        while True:
            try:
                event = await asyncio.to_thread(
                    self.live_event_queue.get, True, _LIVE_EVENT_POLL_TIMEOUT_S
                )
            except queue.Empty:
                continue
            try:
                await self._handle_live_event(event)
            except Exception as exc: # noqa: BLE001 -- one bad event must not kill the loop
                log.warning("live event handling failed (event=%r): %s", event, exc)

    async def _handle_live_event(self, event: dict[str, Any]) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        kind = event.get("type")
        if kind == "TRIGGER_STATE":
            self.trigger_state = event.get("state")
            self.trigger_last_event_utc = ts
            self.trigger_dart_count = event.get("dart_count")
            # Added 2026-08-14 alongside the visit model. `.get()` with a
            # fallback to the current value, not a bare assignment: an
            # emitter that predates this key (or any future one that
            # legitimately doesn't know the visit) must not silently
            # blank out a visit ID this server already learned from a
            # THROW_DETECTED/VISIT_CLEARED event.
            self.visit_id = event.get("visit_id") or self.visit_id
            # The capture thread reaching its first real TRIGGER_STATE
            # this session is the definitive "no longer starting" signal
            # -- see capture_starting's own docstring. Cleared here
            # (not just in stop_capture()) so a session that starts
            # cleanly flips the pill from "Starting" to "Throw"/"Takeout"
            # the moment real data exists, not on some separate timer.
            self.capture_starting = False
            await self._broadcast(
                {
                    "type": "TRIGGER_STATE",
                    "ts": ts,
                    # `emitted_at_utc` (2026-09-01, latency-instrumentation
                    # task, purely additive, no behavior change) --
                    # source-stamped by run_capture_loop_body() at the
                    # moment the transition was actually decided, NOT
                    # here at dequeue time like `ts` above. `.get()` with
                    # no fallback: absent (an emitter that predates this
                    # field, or a test double) means absent here too,
                    # never fabricated as equal to `ts` -- a consumer
                    # computing `ts - emitted_at_utc` needs to be able to
                    # tell "no source stamp available" apart from "zero
                    # dispatch latency this time." See
                    # run_capture_loop_body()'s own on_event docstring for
                    # the full reasoning.
                    "emitted_at_utc": event.get("emitted_at_utc"),
                    "state": self.trigger_state,
                    "session": event.get("session"),
                    "dart_count": self.trigger_dart_count,
                    "capture_starting": False,
                    # Ordering stamp -- this message ENDS Starting on the
                    # client, so it is exactly the one a stale
                    # starting-True snapshot must not be able to undo. See
                    # `_capture_status_stamp()`.
                    **self._capture_status_stamp(),
                    "visit_id": self.visit_id,
                    # settle_duration_s/straggler_camera (2026-09-01,
                    # same latency-instrumentation task) -- only present
                    # on the source event for a READY_TO_CAPTURE
                    # transition (see run_capture_loop_body()'s own
                    # comment), so passed through ONLY when actually
                    # present rather than fabricating them as None on
                    # every other transition -- keeps this payload's key
                    # set meaningful ("this key exists" already tells a
                    # consumer something) instead of padding every
                    # message with two keys that are almost always
                    # absent.
                    **{
                        k: event[k]
                        for k in ("settle_duration_s", "straggler_camera")
                        if k in event
                    },
                }
            )
        elif kind == "THROW_DETECTED":
            # THE single "a dart was just scored" push (docs/LIVE_API.md,
            # added 2026-08-14). Emitted by
            # opendarts.live.capture_daemon.handle_ready_to_capture() the
            # moment the primary engine's result is durably on disk.
            # Rebroadcast VERBATIM (plus a server-side `ts`, the same
            # convention every other branch here follows) -- this server
            # deliberately adds no scoring interpretation of its own; the
            # capture loop is the only thing that actually knows what was
            # scored.
            #
            # PURELY ADDITIVE: the PACKAGE_SAVED branch below still fires
            # for the same throw, milliseconds later, and still drives
            # the existing PACKAGES_UPDATED re-render the dashboard's own
            # JS depends on. Nothing was rewired to go through this
            # instead.
            # Belt-and-braces against a MISSED rotation: VISIT_CLEARED is
            # what normally empties visit_throws, but if one were ever
            # dropped, a throw arriving under a DIFFERENT visit id is
            # itself proof the previous visit ended -- keeping the old
            # turn's darts alongside it would both misreport /api/state's
            # visit section and let this list grow without bound. Only
            # ever compares against a visit id we actually have.
            incoming_visit = event.get("visit_id")
            if incoming_visit is not None and incoming_visit != self.visit_id:
                if self.visit_id is not None and self.visit_throws:
                    log.warning(
                        "throw arrived for visit %s while still tracking %s "
                        "(%d throw(s)) -- no VISIT_CLEARED was seen for the "
                        "previous visit; dropping its accumulated throws",
                        incoming_visit,
                        self.visit_id,
                        len(self.visit_throws),
                    )
                self.visit_throws = []
                self.visit_id = incoming_visit
            self.visit_throws.append(event)
            # SAY IT -- by telling every connected browser what to say,
            # rather than by playing a file into an empty server room.
            #
            # THE PHRASE, NOT THE DART. "treble 20" goes on the wire
            # already resolved, so no JavaScript anywhere has to know
            # that ring "outer_bull" is called "bullseye" or that a
            # single is announced as a bare number. That vocabulary lives
            # in exactly one place (opendarts/live/audio.py) where it is
            # pinned against retail_dart_fields() by test; a second copy
            # in the dashboard would be a second thing to keep correct,
            # and the day they disagreed the board would say one number
            # and the speaker another.
            #
            # A MESSAGE OF ITS OWN rather than a field on THROW_DETECTED
            # below, for two reasons. It is sent FIRST, and being a
            # separate frame is what lets it be: sound is the slowest
            # thing a human notices, so it should leave before the
            # payload that redraws a table. And a client that only wants
            # to be a speaker -- a tab on a TV that shows nothing -- can
            # handle this one type and ignore the scoring feed entirely.
            #
            # SENT UNCONDITIONALLY. There is no server-side "audio
            # enabled" left to gate on; whether a noise is made is each
            # device's own decision, taken from its own localStorage. A
            # muted dashboard drops a ~40-byte message, which is cheaper
            # than the server keeping an opinion it would then have to
            # reconcile with three different screens.
            call = audio.phrase_for(
                event.get("sector"), event.get("ring"), event.get("ok", True)
            )
            if call:
                await self._broadcast({
                    "type": "DART_CALL", "phrase": call, "ts": ts,
                    "visit_id": event.get("visit_id"),
                })
            await self._broadcast({**event, "ts": ts})
            # Retail: the throw itself, fanned out a second time on the
            # retail channel, then an explicit `throw_detected` state
            # message -- the throw carries the dart, the state carries
            # the visit it landed in.
            await self.publish_retail({**event, "ts": ts})
            await self.publish_retail_state("throw_detected")
        elif kind == "VISIT_CLEARED":
            # A turn ended -- the board is confirmed genuinely empty
            # again (or a manual Reset abandoned the turn); `visit_id` is
            # the NEW visit darts from here on belong to. See
            # opendarts.live.capture_daemon.run_capture_loop_body()'s two
            # rotation points.
            # Snapshot the closing visit into the retail catch-up ring
            # BEFORE visit_throws is emptied -- this is the only moment
            # the whole turn exists in one place, and it must not depend
            # on packages having been written.
            self._remember_completed_visit(
                event.get("previous_visit_id") or self.visit_id,
                reason=str(event.get("reason") or ""),
                closed_at_utc=ts,
            )
            self.visit_id = event.get("visit_id") or self.visit_id
            self.visit_throws = []
            log.info(
                "visit cleared (%s -> %s, %s dart(s), reason=%s)",
                event.get("previous_visit_id"),
                event.get("visit_id"),
                event.get("n_darts"),
                event.get("reason"),
            )
            await self._broadcast({**event, "ts": ts})
            # Retail: an explicit, precise name -- a manual reset and a
            # completed visit are genuinely different reasons a visit
            # ended, and a client may want to tell them apart.
            await self.publish_retail_state(
                "manual_reset" if event.get("reason") == "reset" else "visit_complete"
            )
        elif kind == "BOARD_PHOTO":
            # A new empty-board photo, rendered off the capture thread
            # just after Start and after a takeout/Reset that changed the
            # board. The JPEG stays here; screens are told
            # only its version and fetch GET /api/board/photo themselves,
            # so a 200 KB image never rides the event socket.
            jpeg = event.get("jpeg")
            if isinstance(jpeg, (bytes, bytearray)) and jpeg:
                self.board_photo_jpeg = bytes(jpeg)
                self.board_photo_version = hashlib.sha1(self.board_photo_jpeg).hexdigest()[:12]
                await self._broadcast({
                    "type": "BOARD_PHOTO", "ts": ts,
                    "version": self.board_photo_version,
                    "visit_id": event.get("visit_id"),
                })
        elif kind == "AD_BOARD_STATUS":
            # AD's indicator light (Scoring tab header, 2026-08-14).
            # Pushed by AdWsListener's own on_status_change callback --
            # from ITS background thread, via this SAME live_event_queue
            # every other cross-thread event here already uses, not a new
            # mechanism. No AppState field to update: state_dict() reads
            # board_status() live off the listener itself (one source of
            # truth), this is purely a forwarded broadcast so already-
            # connected tabs update immediately instead of only on their
            # next full page load.
            await self._broadcast({**event, "ts": ts})
        elif kind == "AD_CONNECTION":
            # The listener's WebSocket to Autodarts came up or went down
            # (AdWsListener's on_connection_change, 2026-09-22). This is
            # what lets the Config tab's "On and connected / On but NOT
            # connected" note follow the real socket in every open tab
            # instead of only after a reload. AD_BOARD_STATUS cannot do
            # this job -- see on_connection_change's comment in
            # ad_ws_listener.py for why a board status says nothing
            # reliable about the connection.
            #
            # The event's own `connected` is IGNORED and the live value is
            # read instead: transitions can be announced from two threads,
            # so their events can reach this queue out of order, but every
            # transition is followed by an event and each event reports
            # the state at the moment it is handled -- so the LAST
            # broadcast is always the true final state, whatever order the
            # announcements raced in.
            await self._broadcast({
                "type": "AD_CONNECTION", "ts": ts, **self.ad_connection_snapshot(),
            })
        elif kind == "CALIBRATION_STATUS":
            # Real, visible confirmation that Start's auto-calibrate step
            # ran (or was skipped because a valid calibration already
            # existed) -- added 2026-08-12, bug #3 (Start gave no sign
            # of whether it calibrated). Emitted by
            # opendarts.live.capture_daemon.run_capture_loop_body's own
            # "Calibration bootstrap" section (see that module for the
            # two real emit sites, `source` "startup"/"startup_reused")
            # -- reuses the SAME CALIBRATION_STATUS message shape/handler
            # the dashboard's manual "Calibrate" button already produces
            # (POST /api/calibration/refresh -> AppState.refresh_calibration
            # -> _broadcast_calibration_status), just with a `source` the
            # manual path never sets (so the JS can tell them apart and
            # never double-log a manual click's own already-visible
            # result). Never touches self.calibration_store -- that's the
            # SAME object the capture thread already updated directly
            # (or deliberately left alone, for the reused case); this is
            # display-plus-confirmation only, same division of
            # responsibility refresh_calibration() itself follows.
            raw_calibrations = event.get("calibrations") or {}
            self.calibration_status = calibration_status_dict(raw_calibrations, self.n_cameras)
            self.calibration_checked_at_utc = ts
            self.calibration_geometry_relearned = event.get("ring_geometry_relearned")
            await self._broadcast(
                {
                    "type": "CALIBRATION_STATUS",
                    "ts": ts,
                    "cameras": self.calibration_status,
                    "source": event.get("source"),
                    "ring_geometry_relearned": self.calibration_geometry_relearned,
                }
            )
        elif kind == "PACKAGE_SAVED":
            # Eagerly re-discover packages right now instead of waiting up
            # to package_poll_interval_s for _package_poll_loop to notice
            # -- this is the actual point of wiring a live_event_queue in
            # the first place ("real live state, not just polling").
            # _package_poll_loop keeps running
            # independently as a safety net (e.g. a package written by
            # something other than this process's own capture thread).
            path = event.get("path")
            was_known = str(path) in self._known_package_paths
            record = await self._refresh_package(path) if path else None
            current = self._packages_cache
            if event.get("refresh_only") and was_known and record is not None:
                # The daemon's "all"-mode clip landing (see capture_daemon.
                # _write_package_clip): the cached row learns has_video for
                # /api/packages, but screens are not told -- nothing they
                # show depends on it in that mode, and the classic
                # dashboard rebuilds its whole Engines table per broadcast.
                # A package the cache had not seen, or has lost, is
                # announced as usual.
                log.debug("live PACKAGE_SAVED (refresh only): %s", path)
            else:
                log.info(
                    "live PACKAGE_SAVED event: %s (%d package(s) total, %s)",
                    path, len(current),
                    "updated" if was_known else ("new" if record else "not readable yet"),
                )
                await self._broadcast(
                    {
                        "type": "PACKAGES_UPDATED",
                        "ts": ts,
                        "count": len(current),
                        "new_count": 0 if was_known or record is None else 1,
                        "packages": self._broadcast_packages([record] if record else []),
                    }
                )
        else:
            log.debug("unknown live event type %r, ignoring: %r", kind, event)

        # RETAIL FEED (WS /api/live). Generic, diffing-driven pass after
        # every live event: publishes only when the retail snapshot
        # genuinely changed, naming the event from the status
        # transition. The precise names (`throw_detected`,
        # `visit_complete`) come from their own branches above; this is
        # the self-healing catch-all, so a transition without its own
        # call site folds into the next message rather than being lost.
        await self.publish_retail_state()

    # -- cameras (LocalCameraHub.status -- real per-camera debug info) --

    def cameras_status_dict(self) -> dict[str, Any]:
        """Surfaces opendarts.live.local_capture.CameraStatus for every
        camera in the hub -- backend actually negotiated, requested vs.
        actual resolution/fps, open/first-frame latency, frame count,
        last read outcome, last error. This data has existed in
        LocalCameraHub since it was built; this method is what actually
        puts it in front of the dashboard (see /api/cameras/status
        below). Only meaningful when this process owns the hub; any
        other frame source reports honestly rather than inventing a
        per-camera debug surface it does not have.
        """
        if self.hub is None:
            return {
                "frame_source": "local",
                "available": False,
                "cameras": {},
                "reason": "local camera hub not initialized yet",
            }
        return {
            "frame_source": "local",
            "available": True,
            "cameras": {
                str(cam): dataclasses.asdict(status)
                for cam, status in sorted(self.hub.status.items())
            },
            "reason": None,
        }

    # -- state / websocket ----------------------------------------------

    # -- retail subscriber plumbing (WS /api/live) --------------------------
    # A SEPARATE subscriber set from `self.clients` (the /api/events debug
    # feed), not a filtered view of it: the debug stream must never gain
    # message types just because a retail client connected, and the retail
    # stream must never carry debug traffic. Only two message types ever go
    # out here -- "state" and "throw".
    def subscribe_retail(self) -> "asyncio.Queue":
        q: "asyncio.Queue" = asyncio.Queue(maxsize=64)
        self._retail_subscribers.add(q)
        return q

    def unsubscribe_retail(self, q: "asyncio.Queue") -> None:
        self._retail_subscribers.discard(q)

    def _remember_completed_visit(
        self, visit_id: "str | None", *, reason: str, closed_at_utc: str
    ) -> None:
        """Append one just-closed visit to the retail catch-up ring.

        Scores each dart through the SAME retail_dart_from_package() the
        live snapshot uses, so a corrected throw reads identically on the
        socket and in the catch-up -- two copies of that precedence would
        eventually disagree, and the visible failure would be a wrong
        score in front of a viewer.

        An empty visit (a Reset with nothing thrown) is not recorded:
        there is nothing for a client to replay, and it would push a real
        turn out of a bounded ring.
        """
        if not visit_id or not self.visit_throws:
            return
        rows = sorted(
            self.visit_throws,
            key=lambda t: t.get("visit_index") if t.get("visit_index") is not None else 0,
        )
        darts = [retail_dart_from_package(t) for t in rows]
        self._recent_visits.append({
            "visit": visit_id,
            "n_darts": len(darts),
            "darts": darts,
            "total": sum(d.get("value") or 0 for d in darts),
            "completed_at_utc": closed_at_utc,
            "reason": reason or "takeout",
        })

    async def publish_retail(self, event: dict[str, Any]) -> None:
        """Fan one event out to every retail subscriber. A client whose
        queue is full is dropped rather than allowed to block the feed."""
        dead = []
        for q in self._retail_subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._retail_subscribers.discard(q)

    @property
    def retail_status(self) -> str:
        """This project's own live state, expressed in the retail wire
        vocabulary. Derived, not stored -- there is no second source of
        truth to drift."""
        if self.controller is None:
            return "no_live_capture"
        if self.capture_last_start_error:
            return "camera_error"
        if self.capture_starting:
            return "connecting"
        if not self.controller.is_running():
            return "stopped"
        if self.trigger_state == "takeout_waiting":
            return "takeout"
        return "throw"

    def _build_retail_state_fields(self) -> dict[str, Any]:
        """The complete retail snapshot -- every field a `state` message
        carries, present and honest every time (an empty visit is
        `n_darts: 0, darts: []`, never an omitted key).

        Deliberately excludes `type`/`event`/`at`: those are added by the
        caller at publish time and must never enter the diff, since `at`
        differs by construction and would defeat change detection."""
        # Built from `visit_throws` -- the live THROW_DETECTED events --
        # NOT from the package cache. A package-derived snapshot went
        # empty whenever storage was off (`store_packages` in
        # data/config.json), and lagged the throw by however long
        # the background save took even when it was on. Corrections are
        # carried here too: correct_throw() writes `corrected_*` onto the
        # matching visit_throws entry as well as onto the package, so
        # this reports the correction without reading disk.
        darts: list[dict[str, Any]] = []
        if self.visit_id is not None:
            rows = sorted(
                (t for t in self.visit_throws if t.get("visit_id") in (None, self.visit_id)),
                key=lambda t: t.get("visit_index") if t.get("visit_index") is not None else 0,
            )
            darts = [retail_dart_from_package(t) for t in rows]
        return {
            "running": bool(self.controller is not None and self.controller.is_running()),
            "status": self.retail_status,
            "visit": self.visit_id,
            "n_darts": self.trigger_dart_count if self.trigger_dart_count is not None else len(darts),
            "darts": darts,
        }

    def _retail_event_name_for_status_transition(
        self, old_status: "str | None", new_status: str
    ) -> "str | None":
        """Map a (previous, current) status pair onto a purely
        status-driven retail event name, or None when this isn't this
        function's transition to name.

        Returns None when `old_status == new_status` -- no real
        transition happened. This matters: "throw" covers four distinct
        trigger states, so a dart landing is a throw->throw
        non-transition that still shows a real `n_darts` diff; without
        this guard it would be published as a bare "started" and steal
        the diff from the precise `throw_detected` that follows.

        Leaving TAKEOUT also returns None on purpose -- the clear arrives
        moments later from its own call site, as `visit_complete` or
        `manual_reset`, with the visit already cleared."""
        if old_status == new_status:
            return None
        if new_status == "connecting":
            return "connecting"
        if new_status == "camera_error":
            return "camera_error"
        if old_status == "camera_error":
            return "camera_error_cleared"
        if new_status == "takeout":
            return "takeout_started"
        if old_status == "takeout":
            return None
        if new_status == "stopped":
            return "stopped"
        if new_status == "no_live_capture":
            return "no_live_capture"
        if new_status == "throw":
            return "started"
        return "changed"

    async def publish_retail_state(self, event_name: "str | None" = None) -> None:
        """The single gatekeeper for every `state` message.

        DIFFING IS THE TRIGGER. The generic path (`event_name is None`)
        publishes nothing unless the snapshot genuinely differs, and
        derives its name purely from the status transition.

        An EXPLICIT `event_name` always represents a real event the
        caller knows happened, so it publishes even when the snapshot is
        byte-identical -- suppressed only when BOTH the fields and the
        name match the last publish. This is what lets `throw_detected`
        and `visit_complete` both fire for a visit's 3rd dart off the
        same snapshot; a fields-only dedup would swallow the second
        every time."""
        fields = self._build_retail_state_fields()
        old_status = self._last_retail_status
        self._last_retail_status = fields["status"]
        fields_changed = fields != self._last_retail_state_fields
        if event_name is None:
            if not fields_changed:
                return
            event_name = self._retail_event_name_for_status_transition(
                old_status, fields["status"]
            )
            if event_name is None:
                return
        elif not fields_changed and event_name == self._last_retail_event_name:
            return
        self._last_retail_state_fields = fields
        self._last_retail_event_name = event_name
        await self.publish_retail({
            "type": "state", "event": event_name, **fields,
            "at": datetime.now(timezone.utc).isoformat(),
        })

    @property
    def capture_root(self) -> Path:
        """Where frame-ring captures are on THIS rig.

        The running service's own root first, because that is the
        directory the writer is actually filling and a Delete control that
        empties a different one is a Delete control that lies. The
        constructor override next (standalone server, tests), and the
        module default last -- read at call time rather than bound at
        import, so a data directory moved by `OPENDARTS_DATA_DIR` (or by
        the test suite's own path sandbox) is followed rather than
        remembered.
        """
        if self._capture_root_override is not None:
            return self._capture_root_override
        service = self.throw_capture
        root = getattr(service, "capture_root", None) if service is not None else None
        if root is not None:
            return Path(root)
        return Path(throw_capture_mod.DEFAULT_CAPTURE_ROOT)

    def recorded_data_snapshot(self) -> dict[str, Any]:
        """What this rig has recorded, per kind, with the bytes.

        THE NUMBERS THE CONFIRMATION USES. "Delete 312 throw packages
        (2.0 GB) and 3 captures (420 MB)?" is a question somebody can
        answer; "delete everything?" is not, and the difference is
        entirely in these two counts being real rather than estimated.

        WALKS BOTH ROOTS, WHICH IS WHY IT IS NOT ON THE POLL. Sizing
        thousands of package files is milliseconds of real I/O -- fine on
        a button press, wasteful several times a second, and pointless
        the rest of the time since nothing on the page shows it until
        somebody reaches for Delete. `/api/state`'s disk row carries the
        free-space guard instead, which costs one reading.

        A package is a `<session>/<throw>/` directory with a `meta.json`
        -- the same thing `discover_packages()` counts and the same thing
        the delete reports, so the number in the question and the number
        in the answer are the same number.
        """
        package_root = Path(self.package_root)
        packages = 0
        package_bytes = 0
        try:
            sessions = [p for p in package_root.iterdir() if p.is_dir()]
        except OSError:
            sessions = []
        for session_dir in sessions:
            packages += sum(1 for _ in session_dir.glob("*/meta.json"))
            package_bytes += throw_capture_mod.dir_size_bytes(session_dir)
        captures = throw_capture_mod.measure_captures(self.capture_root)
        return {
            "packages": {
                "root": str(package_root),
                "count": packages,
                "bytes": package_bytes,
                "label": frame_ring.format_bytes(package_bytes),
            },
            "captures": {
                "root": captures["root"],
                "count": captures["count"],
                "bytes": captures["bytes"],
                "label": frame_ring.format_bytes(captures["bytes"]),
            },
            "total": {
                "count": packages + captures["count"],
                "bytes": package_bytes + captures["bytes"],
                "label": frame_ring.format_bytes(package_bytes + captures["bytes"]),
            },
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            "package_root": str(self.package_root),
            "frame_source": "local",
            "config": {
                "package_root": str(self.package_root),
                # Rides along with package_root because it is the same
                # question asked one level down: not just where throws go,
                # but whether there is anywhere left to put them.
                "disk": disk_usage_for(self.package_root),
                # RAM sits beside disk: the two capacity questions a slow
                # rig raises, answered from the same place. See memory_info.
                "memory": memory_info(),
                # CPU joins them so the Info tab's "Load" block answers all
                # three "is this rig stressed" numbers from one poll. It is a
                # RATE measured between polls (see cpu_load) -- correct only
                # because /api/state is re-fetched on a timer while the tab
                # is open; the first sample after start reads busy_pct=None.
                "cpu": cpu_load(),
                "frame_source": "local",
                "host": self.host,
                "port": self.port,
                "package_poll_interval_s": self.package_poll_interval_s,
                # This is NOT this server process's own poll interval --
                # it's opendarts.live.capture_daemon.POLL_INTERVAL_SECONDS,
                # the capture LOOP's poll rate, only actually running
                # when this AppState is part of a opendarts.live.run_product
                # combined process. Shown here for honest visibility even
                # when this server is running standalone (no loop at
                # all) -- labeled clearly in the UI, not implied to be
                # live/editable.
                "capture_loop_poll_interval_s": POLL_INTERVAL_SECONDS,
                "live_events_enabled": self.live_events_enabled,
                # True only when this AppState shares a real
                # opendarts.live.capture_daemon.CalibrationStore with an
                # in-process capture loop (opendarts/live/run_product.py) --
                # i.e. whether a manual recalibrate here can actually
                # affect live scoring, not just this dashboard's own
                # display. False in this module's own standalone CLI.
                "calibration_store_wired": self.calibration_store is not None,
            },
            # Multi-engine scoring config (docs/ENGINES.md) -- read fresh
            # every /api/state poll, same honest "None means no capture
            # loop shares this process" convention as calibration/trigger
            # below. `available_engines` always lists the full registry
            # (opendarts.engines.registry.engine_names()) regardless of
            # whether a store is wired, so the Config tab can render its
            # radio/checkboxes even in standalone mode (read-only there,
            # same as every other Config-tab field in that mode).
            "engine_config": (
                self.engine_config_store.meta()
                if self.engine_config_store is not None
                else {
                    "primary": DEFAULT_PRIMARY_ENGINE,
                    "also_run": [],
                    "timeout_s": None,
                    "available_engines": engine_names(),
                }
            ),
            # Detection time (dart_stable_frames) -- None means no capture
            # loop in this process, same convention as other store keys.
            "lifecycle_settings": (
                self.lifecycle_settings_store.meta()
                if self.lifecycle_settings_store is not None
                else None
            ),
            # Live per-dart diagnostics switch (opendarts.live.
            # diagnostics_gate, 2026-09-04) -- a real, process-wide
            # module-level singleton, not per-AppState state, so this is
            # read fresh here (like ad_board_status below) rather than
            # cached; also directly readable/settable via the dedicated
            # GET/POST /api/diagnostics routes for a tool that only wants
            # this one flag.
            "diagnostics": diagnostics_gate.meta(),
            # AD's indicator light (Scoring tab header, 2026-08-14). Read
            # LIVE off the listener's own board_status() every time -- no
            # cached copy on AppState, so there's exactly one place this
            # can ever be wrong. "unknown" (BOARD_STATUS_UNKNOWN) when no
            # listener exists in this process at all.
            "ad_board_status": (
                self.ad_ws_listener.board_status()[0]
                if self.ad_ws_listener is not None
                else BOARD_STATUS_UNKNOWN
            ),
            # How long it has been in that status, so the dashboard can
            # tell "takeout" (normal, a few seconds) from "STUCK in
            # takeout" (AD never saw the darts come out, so it reports no
            # further throws and its buffer keeps serving the finished
            # visit). Read live off the listener, same as the status.
            "ad_board_status_age_sec": (
                self.ad_ws_listener.board_status_age_sec()
                if self.ad_ws_listener is not None
                else None
            ),
            "calibration": {
                # This dashboard's own last-refreshed DISPLAY data --
                # only ever updates on an explicit manual refresh now (no
                # background poll, see refresh_calibration()'s own
                # docstring). checked_at_utc stays honestly null until
                # the first manual refresh in THIS server process.
                "checked_at_utc": self.calibration_checked_at_utc,
                "cameras": self.calibration_status,
                # Set when the last calibration found the cameras had moved
                # and relearned the rig's layout -- the dashboard says so.
                "ring_geometry_relearned": self.calibration_geometry_relearned,
                # What the CAPTURE LOOP is actually scoring throws
                # against right now -- distinct from the two fields
                # above, which are just this dashboard's own display
                # snapshot. None when self.calibration_store is None
                # (this module's own standalone CLI -- no capture loop in
                # this process at all). "source" is "startup" (the
                # capture loop's own once-at-process-start bootstrap,
                # unchanged/untouched by this dashboard) until a manual
                # recalibrate here actually replaces it, at which point
                # it becomes "manual" -- the real answer to "do we
                # constantly recalibrate?" (no).
                "live_source": (
                    self.calibration_store.meta() if self.calibration_store is not None else None
                ),
            },
            "trigger": {
                # Honest either way: "available" is True only when this
                # AppState was actually built with a live_event_queue
                # (opendarts/live/run_product.py's combined entrypoint) AND
                # at least one real event has been pushed onto it yet --
                # "state" stays None honestly until the first one arrives.
                # Standalone `-m opendarts.live.server` (no live capture
                # process/IPC in this process at all -- see
                # opendarts/live/capture_daemon.py's module docstring)
                # reports False/None/this same explanatory reason it
                # always has.
                "available": self.live_events_enabled,
                "state": self.trigger_state,
                "dart_count": self.trigger_dart_count,
                "last_event_utc": self.trigger_last_event_utc,
                "reason": (
                    (
                        "live event push wired (opendarts.live.run_product) -- "
                        "state above is real, pushed straight from the capture "
                        "loop thread, not polled"
                    )
                    if self.live_events_enabled
                    else (
                        "no live capture_daemon process/IPC to read from -- "
                        "this server fetches frames and computes calibration "
                        "itself (see opendarts/live/capture_daemon.py, "
                        "docs/DEPLOYMENT.md); run opendarts.live.run_product "
                        "instead for real live trigger-state push"
                    )
                ),
            },
            # The current VISIT (turn), added 2026-08-14 --
            # `visitId`/`numThrows`/`throws`, grouped into one section
            # here to match the shape every other block in this dict
            # already uses.
            # `available`/None follow the exact same honest-null
            # convention as "trigger" above: standalone `-m
            # opendarts.live.server` has no capture loop pushing visits, and
            # says so rather than reporting an empty visit that doesn't
            # exist. `throws` are the real THROW_DETECTED payloads for
            # this visit, in scored order (index == that dart's
            # visit_index) -- what POST /api/visits/{visit_id}/throws/
            # {index}/correct corrects against.
            "visit": {
                "available": self.live_events_enabled,
                "visit_id": self.visit_id,
                "n_throws": len(self.visit_throws),
                "max_throws": MAX_DARTS_PER_TURN,
                "throws": list(self.visit_throws),
                "board_photo_version": self.board_photo_version,
            },
            "packages": {
                "count": len(self._packages_cache),
                "root_exists": self.package_root.exists(),
            },
            "capture_loop": (
                # Real, honest Start/Stop/idle-timeout status -- see
                # opendarts.live.capture_daemon.CaptureLoopController's own
                # docstring. None (not a fabricated "stopped") when this
                # AppState has no controller at all (this module's own
                # standalone CLI) -- the dashboard must distinguish "no
                # capture loop exists in this process" from "one exists
                # and is currently stopped," same honest-null convention
                # as calibration.live_source/trigger.state above.
                # "starting"/"last_start_error" added 2026-08-12 alongside
                # the expanded status-pill vocabulary -- see
                # self.capture_starting's own docstring for the full
                # mechanism (real window between a successful Start and
                # the capture loop's first real TRIGGER_STATE event).
                {
                    **self.controller.meta(),
                    "starting": self.capture_starting,
                    "last_start_error": self.capture_last_start_error,
                    # See `_capture_status_stamp()` -- lets the dashboard
                    # tell a slow /api/state response from a newer push.
                    **self._capture_status_stamp(),
                }
                if self.controller is not None
                else None
            ),
            "websocket_clients": len(self.clients),
        }

    async def _broadcast(self, msg: dict[str, Any]) -> None:
        if not self.clients:
            return
        data = json.dumps(msg)
        dead: list[WebSocket] = []
        for ws in self.clients:
            try:
                await ws.send_text(data)
            except Exception: # noqa: BLE001 -- a dead socket must not break the others
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    def is_shutting_down(self) -> bool:
        """Whether the server has begun shutting down.

        False when nothing supplied a check -- a TestClient app, or any
        embedding that never owned a uvicorn Server, is never shutting
        down as far as this is concerned, and an endless stream in a test
        is bounded by the test itself.
        """
        check = self.should_exit_check
        if check is None:
            return False
        try:
            return bool(check())
        except Exception: # noqa: BLE001 -- must never break a live stream
            return False

    async def close_all_clients(self) -> None:
        """Proactively closes every tracked WebSocket connection (e.g. a
        browser tab left open on /api/events) as part of this app's own
        shutdown, instead of relying solely on uvicorn's own graceful
        connection-draining. Defense-in-depth, not the fix for the
        confirmed shutdown hang (see _live_event_loop's docstring for the
        actual root cause + fix) -- this closes real live connections
        promptly and cleanly at the ASGI level (a real WebSocket close
        frame, code 1001) rather than leaving them to uvicorn's lower-level
        `connection.shutdown()` (fail_connection(1012) + hard transport
        close), which is also bounded now via `timeout_graceful_shutdown`
        (see opendarts/live/run_product.py's _build_components) but doesn't
        hurt to close cleanly first. Never raises -- a socket that's
        already gone/erroring must not block shutdown.
        """
        for ws in list(self.clients):
            try:
                await ws.close(code=1001)
            except Exception: # noqa: BLE001 -- a dead/half-closed socket must not block shutdown
                pass
        self.clients.clear()

    # -- lifecycle --------------------------------------------------------

    def start_background_tasks(self) -> None:
        # No calibration poll task anymore -- REMOVED 2026-08-12, see
        # DEFAULT_PACKAGE_POLL_INTERVAL_SECONDS's neighboring comment
        # above for the full writeup. Calibration now only ever updates
        # via an explicit refresh_calibration() call (manual "Refresh
        # calibration now" button / POST /api/calibration/refresh).
        self._tasks = [
            asyncio.create_task(self._package_poll_loop(), name="opendarts-package-poll"),
        ]
        if self.live_event_queue is not None:
            self._tasks.append(
                asyncio.create_task(self._live_event_loop(), name="opendarts-live-event-loop")
            )
        if self.controller is not None:
            self._tasks.append(
                asyncio.create_task(self._idle_timeout_loop(), name="opendarts-idle-timeout")
            )
        self._tasks.append(
            asyncio.create_task(self._calibration_progress_loop(), name="opendarts-calibration-progress")
        )

    async def _calibration_progress_loop(self) -> None:
        """Push calibration progress (opendarts.live.calibration_progress)
        to every screen as CALIBRATION_PROGRESS, whenever it moves. The
        calibration records from its own thread; this only looks, five
        times a second, and sends nothing when nothing changed -- so it is
        silent except during a calibration."""
        seen = calibration_progress.PROGRESS.seq
        while True:
            await asyncio.sleep(0.2)
            try:
                seq = calibration_progress.PROGRESS.seq
                if seq == seen:
                    continue
                seen = seq
                await self._broadcast({"type": "CALIBRATION_PROGRESS",
                                       **calibration_progress.PROGRESS.snapshot()})
            except Exception:  # noqa: BLE001 -- a report must never stop
                log.debug("calibration progress push failed", exc_info=True)

    async def stop_background_tasks(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []


# ---------------------------------------------------------------------------
# THE DASHBOARD PAGE: three real files beside this module, not one f-string.
#
# opendarts/live/dashboard/{index.html, app.css, app.js} hold what used to be
# a single ~6,000-line f-string in this file. Inside an f-string every CSS and
# JS brace has to be written doubled (`{{`/`}}`), and three separate times a
# text substitution got that wrong and shipped a page that loaded, rendered,
# and then died at the first JavaScript error -- taking every control with it,
# the tab switcher included. The files carry ordinary braces now, and a
# browser, an editor or `node --check` can read app.js directly.
#
# Nothing else changed: this is still ONE document, assembled here and served
# whole by GET /. No build step, no second request, no framework. See
# opendarts/live/dashboard/__init__.py for the placeholders index.html carries.
# ---------------------------------------------------------------------------
_DASHBOARD_DIR = Path(__file__).resolve().parent / "dashboard"


def _read_dashboard_file(name: str) -> str:
    """One dashboard source file, read from beside THIS module.

    Resolved off ``__file__`` rather than the working directory on purpose:
    run.sh starts the product from the checkout root, a test may run from
    anywhere at all, and the files have to be found in both cases.
    """
    return (_DASHBOARD_DIR / name).read_text(encoding="utf-8")


# Read ONCE at import, not per request. The files cannot change under a
# running process, and GET / should not do disk I/O on a rig whose CPU budget
# belongs to the capture loop.
_DASHBOARD_INDEX_HTML = _read_dashboard_file("index.html")
# The trailing newline on app.css/app.js is a file convention, not page
# content -- stripped so the inlined bytes are exactly what the f-string used
# to emit between <style>/</style> and <script>/</script>.
_DASHBOARD_APP_CSS = _read_dashboard_file("app.css").removesuffix("\n")
_DASHBOARD_APP_JS = _read_dashboard_file("app.js").removesuffix("\n")


def _page_fingerprint(*parts: str) -> str:
    """A short, stable hash of the page this process serves."""
    import hashlib
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


# WHICH PAGE THIS PROCESS SERVES, so an open dashboard can tell it is stale.
#
# A dashboard runs the JS it loaded, forever. After an update the server is
# new but every open screen keeps the old page until someone reloads it --
# fine on a laptop, a real problem on a kiosk nobody can touch. So the page
# is told this fingerprint at load (bootstrap) and again on every WebSocket
# HELLO; a screen that reconnects to a server with a different one reloads
# itself (reloadIfPageIsStale in app.js).
#
# A hash of the ASSETS, not the git commit: a restart that changes nothing
# the page runs must not make every screen reload, a commit that only
# touches the backend should not either, and a checkout with no git (a
# tarball copy) must still work.
_DASHBOARD_PAGE_VERSION = _page_fingerprint(
    _DASHBOARD_INDEX_HTML, _DASHBOARD_APP_CSS, _DASHBOARD_APP_JS,
    # The new dashboard shares the one fingerprint (and the one HELLO
    # field), so a change to either page reloads every open screen.
    *next_ui.FINGERPRINT_PARTS)


def _host_label() -> str:
    # The machine's name leads the tab title, because with several rigs open
    # at once every tab used to read "opendarts live dashboard" and the only
    # way to tell them apart was to click through. `.local` is the mDNS
    # suffix macOS appends; it adds nothing on a tab strip. Escaped because
    # a hostname is operator-controlled text going into HTML.
    return html.escape(
        (platform.node() or "").removesuffix(".local") or "opendarts"
    )


def _dashboard_bootstrap(cam_ids: "list[int]") -> "dict[str, Any]":
    # Every server-computed value on the page, in ONE JSON block index.html
    # carries and app.js reads once (OD_BOOTSTRAP). A new value the page needs
    # is a new key here -- not another interpolation into a template, which is
    # the growth that got the old f-string to 6,000 lines.
    return {
        "cam_ids": cam_ids,
        # Board vocabulary for the "Which was actually right?" modal's
        # manual sector/ring picker, taken from opendarts.geometry.board
        # itself (SECTOR_NUMBERS_CLOCKWISE, and board_ring_names() which
        # derives the ring names by probing sector_ring_for_point) rather
        # than hand-typed: a human's manually-confirmed segment has to be
        # spelled exactly the way opendarts's own scorer spells it or it
        # could never compare equal to any engine's answer.
        "board_sectors": [str(n) for n in SECTOR_NUMBERS_CLOCKWISE],
        "board_rings": board_ring_names(),
        # Drives whether the per-throw "Save frames" control is offered
        # at all. On "all" every throw already records a clip, so the
        # button can only ever duplicate what the rig just did -- and it
        # cannot be decided from the package's own has_video, because
        # the clip is finalised a moment AFTER the package is saved, so
        # a freshly-saved throw reads has_video=false and the useless
        # button flashes up on every dart. The mode is restart-scoped
        # (CONFIG_KEYS marks it restart=True), so reading it once here
        # at page render cannot go stale within a session.
        #
        # EFFECTIVE mode, resolved the same way run_product.py resolves
        # it: read_config_section returns None for an absent or invalid
        # value, and None means "the default applies". Shipping that raw
        # None would make the gate below compare against undefined and
        # show the button on a rig that records every throw -- i.e. the
        # exact bug this key exists to fix.
        "video_record_mode": (
            normalise_video_record_mode(read_config_section("video_record_mode"))
            or DEFAULT_VIDEO_RECORD_MODE
        ),
        # The page's own identity, compared against every HELLO -- see
        # _DASHBOARD_PAGE_VERSION.
        "page_version": _DASHBOARD_PAGE_VERSION,
    }


def _render_dashboard_html(n_cameras: int) -> str:
    """Plain HTML/CSS/vanilla JS -- no build step, no framework
    dependency beyond what's already needed server-side. A tabbed
    layout (Scoring / Engines / Config / Info) rather than one flat
    scrolling page. The camera cards live on the Config tab -- each
    camera gets a single card with its live snapshot AND its calibration
    status together, instead of two separate tables the user had to
    cross-reference by hand.

    **RESTRUCTURED 2026-08-12** to a split-panel layout: a left
    control bar with Start/Stop/Reset/Calibrate, a static header bar,
    and a tabbed right-hand window. An earlier pass only settled the *color convention* (the status pill)
    without settling the page STRUCTURE. This pass fixes the structure:
    a sticky `<header class="top">`
    (`flex: 0 0 auto; position: sticky; top: 0; z-index: 10;`), below it a
    CSS-grid `<main class="layout">`
    (`grid-template-columns: var(--side-w) 1fr; gap: 10px; padding: 10px;`)
    with a fixed-width `<aside class="sidebar">` (a "Controls" block +
    button grid, the `.side-block`/`.btn-grid` pattern) on the left
    and a `<section class="stage">` on the right holding the tab bar +
    tab panels -- the tabs and everything inside
    them (status pill, live WebSocket updates, the Scoring table's row
    numbers/AD comparison/clear-view filter/mark-AD-wrong toggle) simply
    MOVED into the stage column, unchanged in behavior -- this is a
    structural relocation, not a rewrite of any of that logic.

    **Sidebar buttons -- real, wired actions only, audited honestly**
    (this project's "never fake capability that doesn't exist" rule,
    docs/DESIGN.md): the Controls block has Start/Stop/Reset/Calibrate,
    and as of 2026-08-12 all four are real: **Start** (`POST /api/start`), **Stop** (`POST /api/stop`),
    **Reset** (`POST /api/reset`), and **Calibrate** (`POST
    /api/calibration/refresh`, this same `id="btn-refresh-calib"` button
    that used to live in the Cameras tab as "Refresh calibration" --
    moved here, not duplicated, since it's a global action that affects
    every camera, not a per-tab concern).
    - **Start/Stop**: REAL now (were disabled before 2026-08-12 -- see
      this section's own dated history if this comment survives a future
      edit). The program no longer opens the cameras at launch: Start
      opens them, Stop closes them, and an idle session times out
      (configurable). Cameras no longer open automatically the moment
      `opendarts.live.run_product` starts (see that module's own docstring);
      Start opens the shared camera hub and begins a capture-loop
      session (auto-calibrating only if no valid calibration exists yet,
      "skip if geometry is already ok" behavior --
      see opendarts.live.capture_daemon.run_capture_loop_body's "Calibration
      bootstrap" docstring section), Stop ends the session and closes the
      hub. An idle-timeout (default 900s/15min, configurable via
      `POST /api/idle-timeout`) auto-stops a session after real
      inactivity, on a touch/idle-loop pattern -- see
      opendarts.live.capture_daemon.CaptureLoopController's own docstring
      for the full mechanism (adapted for opendarts's thread-based, not
      asyncio-task-based, architecture -- stated plainly there, not
      silently glossed over).
    - **Reset**: REAL (added 2026-08-12). `opendarts.capture.throw_trigger`'s
      turn/takeout state machine already resets its own dart count
      automatically the moment it detects a real takeout, but that's a
      DIFFERENT thing from a manual "force a fresh baseline right now"
      action, which is deliberately exposed separately alongside
      the automatic post-takeout refresh. Reset takes the current
      image as the clean background (whether or not a dart is in it) and
      gets ready for the next throw. Wired via opendarts.live.capture_daemon.ResetRequest (the
      same thread-safe request/signal pattern CalibrationStore already
      established for "FastAPI handler thread writes, capture loop
      thread reads") -- see that class's own docstring and the
      `/api/reset` route below for the full mechanism.
    - No calibration-import button: opendarts's calibration is a
      from-scratch multi-camera OpenCV PnP pipeline (see
      opendarts/live/capture_daemon.py's bootstrap_calibrations), so
      there is nothing to import and no placeholder is shown.

    Loads /api/state + /api/packages + /api/cameras/status once on page
    load, then relies on the WebSocket for live updates (falling back to
    a plain interval poll if the socket ever closes, so the page keeps
    working even if the polling-fallback WS itself hiccups). See
    AppState._render / create_app()'s websocket route for what actually
    drives updates -- real push when running under opendarts.live.run_product,
    a polling fallback otherwise (this module's own docstring has the
    full honest breakdown).

    The text of the page is no longer HERE: it lives in
    opendarts/live/dashboard/{index.html, app.css, app.js}, read once at
    import (see this module's DASHBOARD PAGE section above). This function
    still assembles and returns the same single document -- what it composes
    from changed, not what it produces."""
    cam_ids = list(range(n_cameras))
    # One card per camera: live snapshot + fetch status + calibration
    # badge/metric/note + raw LocalCameraHub detail, all together --
    # replaces the old separate cam-card (Cameras tab) + calib-card
    # (Calibration tab) that required a click to cross-reference. The
    # element ids below (calib-badge-*, calib-err-*, calib-note-*,
    # cam-img-*, cam-fetch-status-*, cam-detail-*) are unchanged from
    # before the merge -- renderCalibration()/renderCameraStatus() below
    # don't care which panel their targets live in. cam-placeholder-*
    # added 2026-08-12 (see BROKEN-IMAGE FALLBACK section of this
    # function's own docstring) -- a real "no signal" box the camera-feed
    # JS (updateCameraFeeds()) swaps in for the <img> on a failed or
    # refused fetch, instead of the browser's native broken-image icon.
    cam_cards = "\n".join(
        f'''<figure class="cam-card">
          <div class="cam-head">
            <span class="cam-label">cam{c}</span>
            <span id="calib-badge-{c}" class="badge unknown">unknown</span>
          </div>
          <div class="cam-media">
            <!-- Starts HIDDEN, with the placeholder below starting VISIBLE
                 (2026-09-13). An <img> with no src is not blank: the
                 browser paints its alt text beside a broken-image icon,
                 so the honest "not started yet" state rendered as "this
                 is broken" for as long as it took the first
                 updateCameraFeeds() to run -- and indefinitely on any
                 path where it does not set a src at all, such as a tab
                 that is still hidden. The empty state is now the DEFAULT
                 rather than something JS has to arrive and install. -->
            <img id="cam-img-{c}" class="cam-img" alt="camera {c} preview"
                 style="display:none;" />
            <!-- The calibration overlay, LAYERED over the live stream
                 rather than replacing it (2026-09-12). Purely decorative
                 to a screen reader -- the badge and reprojection metric
                 beside it carry the same fact as text -- hence alt="" and
                 pointer-events:none. Hidden until a calibration exists
                 AND the stream underneath is actually live, so it can
                 never float over a "No signal" placeholder. -->
            <img id="cam-overlay-{c}" class="cam-overlay" alt="" hidden />
          </div>
          <div id="cam-placeholder-{c}" class="cam-placeholder">
            <span class="cam-placeholder-label">No signal</span>
            <span class="cam-placeholder-sub">camera not started</span>
          </div>
          <!-- Device selector lives ON the preview, 2026-09-11. It was a
               separate section further down the Config tab, which meant
               choosing a device and checking what it shows were in two
               places at once -- the one job this control has is "point
               this slot at the camera showing the right thing", and that
               is only answerable while looking at the picture.
               Directly UNDER the picture, on its own full-width row,
               since 2026-09-12: it started as a third item in the header
               flex, which was fine while every option read "dev 3" and
               fell apart once the options carried real device names --
               a growing control between a label and a badge, squeezing
               both. Under the image it gets the full card width, reads
               as a caption on the thing it selects, and the header goes
               back to being just an identity and a status. -->
          <div class="cam-device">
            <select id="cam-device-{c}" class="cam-device-select" data-slot="{c}"
                    title="Which hardware device feeds this slot. Saves and applies immediately."></select>
          </div>
          <!-- Two UNRELATED facts share this row and used to read as one
               contradictory statement (2026-09-13): a green "calibrated"
               badge directly above "not yet fetched" and a bare em-dash,
               which looks like the badge is lying. It is not -- the left
               half is the live PREVIEW STREAM and the right half is the
               quality of the CALIBRATION SOLVE, and a camera is routinely
               calibrated while its preview is not running. Naming the left
               one "preview" is what makes the right one unambiguous. -->
          <div class="cam-subrow">
            <span class="cam-subrow-label">preview</span>
            <span id="cam-fetch-status-{c}" class="cam-fetch-status">not started</span>
            <span class="calib-metric" id="calib-metric-{c}"
                  title="Reprojection error of this camera's calibration solve -- how far the solved board geometry lands from the landmarks actually seen. Lower is better; under 2.5px is the target. Nothing to do with the preview on the left."><span class="calib-metric-value" id="calib-err-{c}">&mdash;</span><span class="calib-metric-unit">px reproj.</span></span>
          </div>
          <div id="calib-note-{c}" class="calib-note"></div>
          <details class="cam-diag">
            <summary>Diagnostics</summary>
            <div id="cam-detail-{c}" class="cam-detail">no status data yet</div>
          </details>
        </figure>'''
        for c in cam_ids
    )
    host_label = _host_label()
    bootstrap_json = json.dumps(_dashboard_bootstrap(cam_ids))
    # The page-level values first, the two whole files last, so nothing that
    # merely LOOKS like a placeholder inside app.css/app.js is ever rescanned.
    return (
        _DASHBOARD_INDEX_HTML
        .replace("@@OD_HOST_LABEL@@", host_label)
        .replace("@@OD_CAM_CARDS@@", cam_cards)
        .replace("@@OD_BOOTSTRAP_JSON@@", bootstrap_json)
        .replace("@@OD_APP_CSS@@", _DASHBOARD_APP_CSS)
        .replace("@@OD_APP_JS@@", _DASHBOARD_APP_JS)
    )



def _render_next_html(n_cameras: int, role: str = "control") -> str:
    """The greenfield dashboard (opendarts/live/ui, dev/ux/BRIEF.md): the
    same bootstrap as the current page, plus the machine's name for the
    bar, which this page shows rather than only titling the tab with it.

    ``role`` is "control" (the whole app) or "display" (a screen that only
    shows, set up from a controller -- opendarts/live/displays.py). One
    page, two roles, so a fix to what a display shows is a fix to what a
    controller shows."""
    bootstrap = _dashboard_bootstrap(list(range(n_cameras)))
    bootstrap["host_label"] = html.unescape(_host_label())
    bootstrap["role"] = role
    return next_ui.render(host_label=_host_label(), bootstrap_json=json.dumps(bootstrap))


def _frame_sink_attached(hub: "Any") -> "bool | None":
    """Whether a frame sink is wired to the hub's pump, or None if it
    cannot be determined.

    None rather than False for an unknown, so a hub shape this does not
    recognise reports "cannot tell" instead of confidently claiming
    nothing is attached -- a wrong False here would send someone looking
    for a wiring bug that does not exist.
    """
    if hub is None:
        return None
    # Asked of the hub itself, because the hub IS where the pump lives.
    # This used to look through a `_local` child first: the hub was a
    # wrapper composing a local hub and a remote one, and `set_frame_sink`
    # forwarded only to the local child -- so on an all-stream rig the
    # sink was attached to something that never pumped, nothing was ever
    # published, and this function still answered True. That composition
    # is gone (see opendarts/live/remote_capture.py's module docstring):
    # there is one hub, one pump and one sink, so the only thing to ask is
    # the hub.
    if hasattr(hub, "_frame_sink"):
        return getattr(hub, "_frame_sink") is not None
    return None


#: The largest frame-ring window this build will accept from the
#: dashboard. Sixty seconds of three 720p cameras is ~16GB -- the whole
#: memory of a 16GB rig, which is also running other software. A
#: refusal that shows the arithmetic is more useful than accepting a
#: number that will take the process down at the moment it is needed.
#: Not a physical limit and not a guess about anyone's hardware: a rig
#: with the RAM to spare raises it here, in one line, deliberately.
#: Re-exported from the config registry, which is where the validator
#: that enforces it lives. Imported here because several call sites in
#: this module price the ceiling for the dashboard.
MAX_FRAME_RING_SECONDS = config_document.MAX_FRAME_RING_SECONDS


def _apply_frame_ring_seconds(state: "AppState", seconds: float) -> bool:
    """Resize, attach or detach the live ring, and say whether it worked.

    Applied LIVE rather than at the next restart because this setting
    costs gigabytes right now: an operator who turns it down and is told
    to restart to free 4GB has been handed a control that does not do
    what it says. Resizing an existing ring is just a new window -- the
    next append evicts against it -- so a reduction frees memory within
    one pump cycle.

    Returns False (never raises, never silently succeeds) when there is
    nothing live to apply to, so the route can report "saved, but this
    process has no capture loop to apply it to" rather than implying a
    change that did not happen.
    """
    service = getattr(state, "throw_capture", None)
    if service is None:
        return False
    ring = service.ring
    if seconds <= 0:
        if ring is None:
            return False
        # Detach from the pump first, THEN clear: the other order leaves a
        # window in which the pump appends to a ring that has just been
        # emptied, which would look like "the setting did not take".
        if state.hub is not None:
            state.hub.set_frame_ring(None)
        ring.clear()
        service.ring = None
        log.info("frame ring disabled from the dashboard -- retained frames freed")
        return True
    if ring is None:
        ring = frame_ring.FrameRing(seconds)
        service.ring = ring
        if state.hub is not None:
            state.hub.set_frame_ring(ring)
        log.info("frame ring enabled from the dashboard at %.1fs", seconds)
        return True
    ring.seconds = float(seconds)
    log.info("frame ring resized from the dashboard to %.1fs", seconds)
    return True


def _reconfigure_hub(hub: "Any", configs: "Any", urls: "list[str | None] | None") -> None:
    """Reconfigure a hub, passing stream URLs only to a hub that has them.

    Not every hub takes URLs: `LocalCameraHub.reconfigure(configs)` is a
    two-argument method, and so are the stubs that stand in for a hub. So
    the URL argument is offered only where it means something.

    The one case that MUST NOT be quietly swallowed is a real URL going to
    a hub that cannot route one: silently dropping it would leave the
    dashboard showing a stream assignment the process is not honouring --
    the precise failure this whole path exists to avoid. That raises, and
    /api/camera-devices reports the reason.
    """
    if hasattr(hub, "slot_urls"):
        hub.reconfigure(configs, urls)
        return
    if any(u for u in (urls or [])):
        raise RuntimeError(
            "this hub cannot read camera streams -- restart the server to "
            "pick up the saved URLs"
        )
    hub.reconfigure(configs)


def _live_slot_urls(hub: "Any", n_slots: int) -> "list[str | None]":
    """The stream URL actually feeding each slot, or None for local.

    Asks the hub rather than the config, because those disagree exactly
    when someone has saved a change and not restarted -- the case worth
    showing plainly. Duck-typed: a local hub has neither attribute and
    honestly reports all-None.
    """
    # The real hub exposes this directly, one entry per slot. The two
    # fallback branches this used to carry -- one unpicking a wrapper's
    # `_routing`/`_remote._urls`, one reading a remote-only hub's `_urls`
    # -- described hub shapes that no longer exist (see
    # opendarts/live/remote_capture.py's module docstring). What is left
    # is the honest answer for a stub or a fake that has no such notion:
    # every slot is local.
    own = getattr(hub, "slot_urls", None)
    if own is not None:
        return (list(own) + [None] * n_slots)[:n_slots]
    return [None] * n_slots


def _preview_dimensions(
    width: int, height: int, max_width: "int | None" = None
) -> "tuple[int, int]":
    """The (width, height) a preview frame is served at -- unchanged when
    the camera is already at or under MJPEG_MAX_WIDTH, downscaled keeping
    aspect otherwise.

    Its own function purely to keep the stream encoder's downscale rule
    in one readable place.

    NOT for sizing the calibration overlay -- that was the original
    reason this was extracted and it was wrong. The overlay must be
    rendered in the CALIBRATION's pixel space (the native frame size),
    because that is the space its camera_matrix projects into; the
    browser then scales the layer and the stream alike. Sizing the
    overlay with this function is what put the board 1.33x too large on
    a 1280x720 camera.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"frame size must be positive, got {width}x{height}")
    if max_width is None:
        max_width = MJPEG_MAX_WIDTH
    if width <= max_width:
        return (width, height)
    return (max_width, max(1, int(round(height * (max_width / width)))))


class _SharedJpegCache:
    """Encode each camera frame ONCE, however many consumers want it.

    Every MJPEG connection ran its own encode, so the same frame was
    compressed once per open stream: three cameras x three consuming
    machines x 30fps is 270 encodes a second of which 180 are duplicates.
    Measured at 1.4ms per 1280x720 q85 encode, that is ~37% of a core to
    produce ~12% of a core's worth of distinct bytes.

    Keyed by (camera, mode) and validated against the camera's own
    `frame_count`, so a cached entry is reused only for the exact frame it
    was made from -- never a stale one. Two modes (preview and transport)
    are cached separately because they are genuinely different images.

    The per-key lock is what makes this a saving rather than a race: with
    N consumers waking on the same new frame, all N would otherwise miss
    the cache together and start N encodes. The first waiter encodes; the
    rest re-check under the lock and find it done. Serialising per key is
    safe because the encode itself runs in a worker thread -- the event
    loop is never blocked, and different cameras still encode in parallel.

    Bounded by construction: one entry per camera per mode, so a
    three-camera rig holds at most six frames (~1MB), and an entry is
    replaced rather than accumulated.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[int, bool], tuple[int, bytes]] = {}
        self._locks: dict[tuple[int, bool], "asyncio.Lock"] = {}
        self.encodes = 0
        self.hits = 0
        #: Full-frame parts sent as the camera's own JPEG, never encoded.
        self.passthrough = 0

    async def encoded(
        self, cam_id: int, full: bool, frame_count: int, frame: "Any"
    ) -> "bytes | None":
        key = (cam_id, bool(full))
        entry = self._entries.get(key)
        if entry is not None and entry[0] == frame_count:
            self.hits += 1
            return entry[1]
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        async with lock:
            # Re-check: another consumer may have encoded this exact frame
            # while this one waited. This is the line that turns N encodes
            # into one.
            entry = self._entries.get(key)
            if entry is not None and entry[0] == frame_count:
                self.hits += 1
                return entry[1]
            jpg = await asyncio.to_thread(
                _encode_preview_jpeg,
                frame,
                0 if full else None,        # 0 = no downscale
                MJPEG_TRANSPORT_QUALITY if full else None,
            )
            self.encodes += 1
            if jpg is not None:
                self._entries[key] = (frame_count, jpg)
            return jpg


def _grab_with_jpeg(hub: "Any", cam_id: int) -> "tuple[Any, bytes | None]":
    """The hub's frame and camera JPEG for one slot. A hub without
    grab_with_jpeg (a test double, say) has no JPEG to offer.

    The frame may come back UNDECODED -- a LazyFrame, when the hub detects
    from small decodes (opendarts.capture.lazy_frame) -- so a stream that
    forwards the camera's JPEG never pays for a full decode. Whoever needs
    its pixels calls pixels_of() on it."""
    grab_lazy = getattr(hub, "grab_jpeg_lazy", None)
    if grab_lazy is not None:
        return grab_lazy(cam_id)
    grab_pair = getattr(hub, "grab_with_jpeg", None)
    if grab_pair is not None:
        return grab_pair(cam_id)
    return hub.grab(cam_id), None


def _encode_preview_jpeg(
    frame: "Any",
    max_width: "int | None" = None,
    quality: "int | None" = None,
) -> "bytes | None":
    """One preview frame -> JPEG bytes, downscaled to MJPEG_MAX_WIDTH
    first when the source is wider (see that constant's comment for why
    the preview deliberately does not ship full resolution). Runs inside
    asyncio.to_thread from the stream generator -- cv2 releases the GIL
    for both resize and encode, so this is where the stream's real CPU
    cost lives, off the event loop and far away from the pump thread.
    Returns None on encode failure (a caller can only skip the frame;
    there is nothing better to send)."""
    import cv2

    from opendarts.capture.lazy_frame import pixels_of

    # A LazyFrame decodes here, on the encode's worker thread, once per
    # frame however many previews share it (_SharedJpegCache).
    frame = pixels_of(frame)
    # max_width=0 means "do not downscale at all" -- the transport case,
    # where a consumer is SCORING these frames rather than looking at them.
    # Defaults keep the dashboard preview exactly as it was.
    if max_width is None:
        max_width = MJPEG_MAX_WIDTH
    if quality is None:
        quality = MJPEG_JPEG_QUALITY
    h, w = frame.shape[:2]
    target = (w, h) if max_width <= 0 else _preview_dimensions(w, h, max_width)
    if target != (w, h):
        # INTER_AREA: the right interpolator for shrinking, and cheap.
        frame = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


def create_app(
    package_root: Path = DEFAULT_PACKAGE_ROOT,
    scratch_dir: Path = DEFAULT_SCRATCH_DIR,
    n_cameras: int = N_CAMERAS,
    package_poll_interval_s: float = DEFAULT_PACKAGE_POLL_INTERVAL_SECONDS,
    enable_background_poll: bool = True,
    local_hub: local_capture.LocalCameraHub | None = None,
    live_event_queue: "queue.SimpleQueue[dict[str, Any]] | None" = None,
    host: str | None = None,
    port: int | None = None,
    ad_base_url: str = DEFAULT_AD_BASE,
    ad_window_sec: float = DEFAULT_MATCH_WINDOW_SEC,
    ad_timeout_s: float = DEFAULT_TIMEOUT_SEC,
    calibration_store: "CalibrationStore | None" = None,
    reset_request: "ResetRequest | None" = None,
    controller: "CaptureLoopController | None" = None,
    engine_config_store: "EngineConfigStore | None" = None,
    lifecycle_settings_store: "LifecycleSettingsStore | None" = None,
    ad_ws_listener: "AdWsListener | None" = None,
    vcam_set: Any = None,
    vcam_set_factory: Any = None,
    reprojection_targets_px: dict[int, float] | None = None,
    throw_capture: Any = None,
    capture_root: "Path | None" = None,
    display_store: "DisplayStore | None" = None,
    dashboard_choice: "DashboardChoice | None" = None,
) -> FastAPI:
    """FastAPI app factory -- constructor-injected package_root/od_base_url
    (never a hardcoded module-level DEFAULT_PACKAGE_ROOT reference inside
    a route), so tests/test_live_server.py can point this at an isolated
    temp directory instead of the real
    opendarts.live.capture_daemon.DEFAULT_PACKAGE_ROOT. enable_background_poll
    defaults True for real use (the CLI entrypoint below) but tests pass
    False to avoid making real, possibly-slow network/hardware calls on
    every test run.

    Frame source: direct local
    camera access. If `local_hub` is given, it's used AS-IS (the caller
    owns opening/closing it -- this is the test injection seam: build a
    LocalCameraHub against a monkeypatched cv2.VideoCapture, call
    .open_all() yourself, then pass it in here). If `local_hub` is None
    this app's lifespan opens its own
    LocalCameraHub at startup and closes it at shutdown -- the real-server
    path (see main() below).

    live_event_queue: optional real-event-push seam (see module
    docstring's WebSocket /api/events section and AppState._live_event_loop)
    -- opendarts/live/run_product.py's combined entrypoint passes a
    `queue.SimpleQueue` that its own background capture-loop thread pushes
    TRIGGER_STATE/PACKAGE_SAVED events onto; this app's background task
    consumes and broadcasts them immediately instead of waiting for the
    next package poll tick. None (the default -- this module's own
    standalone CLI) means no such task is started and trigger state stays
    honestly reported as unavailable.

    calibration_store: the same opendarts.live.capture_daemon.CalibrationStore
    object opendarts/live/run_product.py's combined entrypoint shares with
    its capture-loop thread (see that module and CalibrationStore's own
    docstring) -- when given, a manual "Refresh calibration now" actually
    replaces what the capture loop scores against, not just this app's
    own display. None (the default -- this module's own standalone CLI,
    which has no capture loop in-process at all) means a manual refresh
    only ever updates this dashboard's display.

    reset_request: the same opendarts.live.capture_daemon.ResetRequest object
    opendarts/live/run_product.py's combined entrypoint shares with its
    capture-loop thread (mirrors calibration_store immediately above --
    same wiring shape, see ResetRequest's own docstring). When given, a
    manual "Reset" click (POST /api/reset) actually signals the capture
    loop to re-baseline itself, not just a no-op. None (the default --
    this module's own standalone CLI, no capture loop in-process) means
    /api/reset reports honestly that no loop is listening.

    controller: the same opendarts.live.capture_daemon.CaptureLoopController
    object opendarts/live/run_product.py's combined entrypoint shares with
    its capture-loop thread (mirrors calibration_store/reset_request
    immediately above -- same wiring shape). When given, POST /api/start
    and /api/stop actually control the real capture loop's camera-hub
    lifecycle, and a background idle-timeout task auto-stops it after
    real inactivity. None (the default -- this module's own standalone
    CLI, no capture loop in-process) means /api/start-/api/stop report
    honestly that no loop is listening, and no idle-timeout task starts.

    engine_config_store: the same opendarts.live.capture_daemon.
    EngineConfigStore object opendarts/live/run_product.py's combined
    entrypoint shares with its capture-loop thread (mirrors
    calibration_store/reset_request/controller immediately above -- same
    wiring shape, docs/ENGINES.md's Config section). READ-ONLY here: it
    reports the live engine set on /api/state. The engine set is
    configured in data/config.json's ``engine_config`` section and
    nowhere else -- there is deliberately no runtime mutation endpoint
    (removed 2026-09-09; the Config tab's controls had already gone when
    the aggregate engine became the permanent primary). None (the
    default -- this module's own standalone CLI, no capture loop
    in-process) means /api/state reports the framework's own defaults.

    lifecycle_settings_store: the same opendarts.lifecycle.settings.
    LifecycleSettingsStore object opendarts/live/run_product.py's combined
    entrypoint shares with its capture-loop thread (mirrors
    engine_config_store immediately above). When given, GET/POST
    /api/detection-time actually change dart_stable_frames for the next
    frame. None (standalone CLI) means those routes report honestly that
    no capture loop is listening.
    """
    package_root = Path(package_root)
    scratch_dir = Path(scratch_dir)
    # In memory unless the caller persists it (run_product passes
    # data/config.json), the same way as the other operator stores.
    displays = display_store if display_store is not None else DisplayStore()
    dashboard = dashboard_choice if dashboard_choice is not None else DashboardChoice()
    state = AppState(
        package_root=package_root,
        scratch_dir=scratch_dir,
        n_cameras=n_cameras,
        package_poll_interval_s=package_poll_interval_s,
        hub=local_hub,
        live_event_queue=live_event_queue,
        host=host,
        port=port,
        ad_base_url=ad_base_url,
        ad_window_sec=ad_window_sec,
        ad_timeout_s=ad_timeout_s,
        calibration_store=calibration_store,
        reset_request=reset_request,
        controller=controller,
        engine_config_store=engine_config_store,
        lifecycle_settings_store=lifecycle_settings_store,
        ad_ws_listener=ad_ws_listener,
        vcam_set=vcam_set,
        vcam_set_factory=vcam_set_factory,
        reprojection_targets_px=reprojection_targets_px,
        throw_capture=throw_capture,
        capture_root=capture_root,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state.hub is None:
            state.hub = local_capture.LocalCameraHub()
            state.hub.open_all()
            state.owns_hub = True
            log.info("local camera hub open:\n%s", state.hub.status_report())
        if enable_background_poll:
            state.start_background_tasks()
        try:
            yield
        finally:
            await state.close_all_clients()
            await state.stop_background_tasks()
            if state.owns_hub and state.hub is not None:
                state.hub.close_all()
            # Publishing outlives no process: the Linux backend runs a
            # worker thread and holds v4l2loopback fds, and a set built
            # lazily by the Autodarts toggle has no other owner to close
            # it. Guarded by getattr because tests inject stand-ins with
            # only the methods they exercise.
            vset, state.vcam_set = state.vcam_set, None
            closer = getattr(vset, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception: # noqa: BLE001 -- shutdown must not raise on a diagnostic
                    log.exception("closing the virtual cameras at shutdown failed")

    app = FastAPI(title="opendarts live dashboard", lifespan=lifespan)
    app.state.opendarts_state = state # exposed for tests / introspection

    def _page_response(request: Request, page: str) -> Response:
        headers = {"Cache-Control": "no-store, max-age=0", "Vary": "Accept-Encoding"}
        # GZIPPED WHEN ACCEPTED, 2026-09-17: ~300 KB of page (much of it
        # comments) is ~96 KB compressed, which matters to a tablet on
        # Wi-Fi. Done here rather than with a compression middleware,
        # which would also try to compress the endless MJPEG streams.
        if "gzip" in request.headers.get("accept-encoding", ""):
            import gzip

            return Response(
                gzip.compress(page.encode("utf-8"), compresslevel=6),
                media_type="text/html; charset=utf-8",
                headers={**headers, "Content-Encoding": "gzip"},
            )
        return HTMLResponse(page, headers=headers)

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Response:
        """The dashboard -- the new one or the classic one, whichever the
        rig's switch says (opendarts/live/dashboard_choice.py). Two query
        options, never paths:

          ``?display`` (``?display=tv&name=Lounge TV``) -- the new page in
            its display role, for a TV nobody touches; always the new page,
            since the classic one has no such role.
          ``?ui=new`` / ``?ui=classic`` -- the other dashboard on this one
            screen, once, without flipping the switch for everyone.
        """
        q = request.query_params
        if "display" in q:
            return _page_response(request, _render_next_html(n_cameras, role="display"))
        ui = q.get("ui") if q.get("ui") in DASHBOARD_CHOICES else dashboard.get()
        if ui == "classic":
            return _page_response(request, _render_dashboard_html(n_cameras))
        return _page_response(request, _render_next_html(n_cameras))

    @app.get("/api/dashboard")
    async def api_dashboard_get() -> dict[str, Any]:
        """Which dashboard ``/`` serves."""
        return {"ok": True, "ui": dashboard.get(), "choices": list(DASHBOARD_CHOICES)}

    @app.put("/api/dashboard")
    async def api_dashboard_put(payload: dict[str, Any] = Body(...)) -> Any:
        """Flip it. Every open dashboard is told (DASHBOARD_SWITCHED) and
        reloads onto the other page; a display is unaffected."""
        try:
            ui = dashboard.set(payload.get("ui"))
        except ValueError as exc:
            return JSONResponse({"ok": False, "reason": str(exc)}, status_code=400)
        await state._broadcast({"type": "DASHBOARD_SWITCHED", "ui": ui})  # noqa: SLF001
        return {"ok": True, "ui": ui}

    # ---- displays: screens that only show, set up from any controller ----
    # opendarts/live/displays.py has the model. Every change is pushed to
    # the displays over the events socket as DISPLAY_UPDATED, so a TV
    # changes the moment a phone saves, with no polling. A display is
    # ``/?display`` (above).

    def _display_error(exc: DisplayError) -> JSONResponse:
        return JSONResponse({"ok": False, "reason": exc.reason}, status_code=exc.status)

    @app.get("/api/displays")
    async def api_displays() -> dict[str, Any]:
        return {"ok": True, "displays": displays.list()}

    @app.post("/api/displays/hello")
    async def api_display_hello(payload: dict[str, Any] = Body(...)) -> Any:
        """A display reporting in (on load, then every ~20 s): registers it
        the first time, refreshes its presence, and answers with the
        settings it should be showing."""
        try:
            rec = displays.hello(payload.get("display_id"), payload.get("info"), payload.get("name"))
        except DisplayError as exc:
            return _display_error(exc)
        return {"ok": True, "display": rec}

    @app.patch("/api/displays/{display_id}")
    async def api_display_update(display_id: str, payload: dict[str, Any] = Body(...)) -> Any:
        try:
            rec = displays.update(display_id, name=payload.get("name"), settings=payload.get("settings"))
        except DisplayError as exc:
            return _display_error(exc)
        await state._broadcast({"type": "DISPLAY_UPDATED", "display": rec})  # noqa: SLF001
        return {"ok": True, "display": rec}

    @app.post("/api/displays/{display_id}/identify")
    async def api_display_identify(display_id: str) -> Any:
        """Flash the display's name on it, so you know which screen a row
        on your phone is."""
        rec = displays.get(display_id)
        if rec is None:
            return _display_error(DisplayError("no such display", 404))
        await state._broadcast({"type": "DISPLAY_IDENTIFY", "display_id": display_id, "name": rec["name"]})  # noqa: SLF001
        return {"ok": True, "online": rec["online"]}

    @app.post("/api/displays/{display_id}/reload")
    async def api_display_reload(display_id: str) -> Any:
        """Reload the page on a display nobody can reach with a keyboard."""
        rec = displays.get(display_id)
        if rec is None:
            return _display_error(DisplayError("no such display", 404))
        await state._broadcast({"type": "DISPLAY_RELOAD", "display_id": display_id})  # noqa: SLF001
        return {"ok": True, "online": rec["online"]}

    @app.delete("/api/displays/{display_id}")
    async def api_display_forget(display_id: str) -> Any:
        """Forget a display. If it is still open it registers again, fresh,
        on its next report -- this is for a screen that is gone."""
        if not displays.forget(display_id):
            return _display_error(DisplayError("no such display", 404))
        await state._broadcast({"type": "DISPLAY_FORGOTTEN", "display_id": display_id})  # noqa: SLF001
        return {"ok": True}

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        # Capabilities ride along on health (2026-09-12): "does that rig
        # have ffmpeg" is answerable with one curl instead of an SSH.
        # Memoized after first use, so this stays cheap to poll.
        # `build` rides along for the same reason capabilities do: until it
        # existed, "what version is that rig running" was answerable only
        # by shelling into the box or probing whether a new route existed
        # yet. It is the first field any bug report needs.
        # `streams` rides along for the same reason: a rig at its
        # connection ceiling turns new streams away, which presents as
        # blank camera tiles with nothing wrong on the rig itself. Before
        # this the only way to tell "refusing everything" from "broken" was
        # to open a stream and read the status code.
        n_cams = len(state.hub.configs) if state.hub is not None else 3
        # `pid` rides along (2026-09-17) for the restart poll: the
        # dashboard has to tell "the rig is back" from "the old process
        # has not died yet", and the two are otherwise identical over
        # HTTP -- /api/restart schedules its SIGTERM half a second out,
        # so the process it is about to kill keeps answering until then.
        # Comparing against the pid POST /api/restart returned makes that
        # exact instead of a guessed delay.
        return {"status": "ok", "service": "opendarts-live-server",
                "pid": os.getpid(),
                "capabilities": capabilities.snapshot(),
                "streams": {
                    "preview_open": state.mjpeg_client_count,
                    "preview_max": _mjpeg_cap(n_cams, MJPEG_MAX_PREVIEW_VIEWERS),
                    "transport_open": state.mjpeg_transport_count,
                    "transport_max": _mjpeg_cap(n_cams, MJPEG_MAX_TRANSPORT_CONSUMERS),
                    "cameras": n_cams,
                },
                "build": build_info.build_info()}

    @app.get("/api/state")
    async def api_state() -> dict[str, Any]:
        return state.state_dict()

    @app.get("/api/board/photo")
    async def api_board_photo() -> Response:
        """The latest empty-board photo for the scoring page, as JPEG. 404
        until the session's first dart has produced one. The version is in
        /api/state (`visit.board_photo_version`) and the BOARD_PHOTO event;
        a screen fetches `?v=<version>`, so the image may be cached for good."""
        if state.board_photo_jpeg is None:
            return JSONResponse({"error": "no board photo yet"}, status_code=404)
        return Response(
            content=state.board_photo_jpeg,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "ETag": f'"{state.board_photo_version}"',
            },
        )

    @app.get("/api/packages")
    async def api_packages() -> list[dict[str, Any]]:
        return state.list_packages()

    @app.get("/api/recorded-data")
    async def api_recorded_data() -> dict[str, Any]:
        """What a Delete would take, per kind, with the bytes.

        EXISTS SO THE QUESTION CAN BE ASKED HONESTLY. The confirmation
        dialog has to name what is going -- "Delete 312 throw packages
        (2.0 GB) and 3 captures (420 MB)?" -- and until this route the
        page knew only how many packages it happened to have loaded and
        nothing at all about captures, so the captures went silently.

        A GET OF ITS OWN rather than more fields on `/api/state`: sizing
        both roots walks every file under them, which is fine on a button
        press and wasteful on a poll that runs whether or not anyone is
        near the button. It is also FRESHER this way -- the numbers in the
        question are read at the moment the question is asked, not at the
        last tick. The cheap half (the free-space guard) does ride on
        `/api/state`, where it is one reading and is wanted continuously.

        `busy` is the delete's own precondition, reported here so the page
        can say why rather than offering a button that will be refused.
        """
        snapshot = await asyncio.to_thread(state.recorded_data_snapshot)
        return {"ok": True, **snapshot,
                "busy": _capture_write_in_flight() is not None,
                "disk": disk_usage_for(state.package_root)}

    def _capture_write_in_flight() -> "str | None":
        """The dump being written right now, or None.

        A dump is the one thing on this rig that is mid-write for seconds
        at a time and lands in the directory a delete is about to remove.
        Deleting under it would leave a half-written capture, or take the
        directory out from under the writer thread and turn an operator's
        tidy-up into a stack trace in the log.
        """
        service = state.throw_capture
        writer = getattr(service, "writer", None) if service is not None else None
        if writer is None:
            return None
        try:
            status = writer.status()
        except Exception: # noqa: BLE001 -- a precondition must never break the page
            return None
        if not status.get("busy"):
            return None
        current = status.get("current") or {}
        return str(current.get("dest_dir") or current.get("kind") or "a capture")

    def _delete_captures() -> "tuple[int, int, list[str]]":
        """Empty the capture root. (captures removed, bytes freed, errors).

        Same properties the package half has had since 2026-08-14 and for
        the same reasons: a real `shutil.rmtree` of the root's CONTENTS
        only, the root itself recreated so the next dump has somewhere to
        land, one bad directory reported rather than aborting the rest,
        and something already gone treated as done rather than as an
        error.

        Bytes are measured per entry immediately before it goes, not from
        one total taken up front: a capture that fails to delete has not
        freed anything, and reporting it as freed would overstate what the
        operator got back.
        """
        root = Path(state.capture_root)
        removed = 0
        freed = 0
        errors: list[str] = []
        try:
            entries = sorted(root.iterdir())
        except FileNotFoundError:
            return 0, 0, errors
        except OSError as exc:
            return 0, 0, [f"{root}: {exc}"]
        for entry in entries:
            try:
                is_dir = entry.is_dir() and not entry.is_symlink()
                size = (throw_capture_mod.dir_size_bytes(entry) if is_dir
                        else entry.lstat().st_size)
                if is_dir:
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            except FileNotFoundError:
                continue # already gone -- not an error here
            except Exception as exc: # noqa: BLE001 -- report, don't abort the rest
                errors.append(f"{entry.name}: {exc}")
                continue
            freed += size
            if is_dir:
                removed += 1
        root.mkdir(parents=True, exist_ok=True)
        return removed, freed, errors

    @app.post("/api/packages/delete-all")
    async def api_packages_delete_all() -> dict[str, Any]:
        """Real, permanent deletion -- the Scoring tab's "Delete recorded
        data" button. Deliberately real `shutil.rmtree`, not a move-to-
        quarantine (unlike this project's own data/archive/ discipline --
        see docs/DESIGN.md's standing guardrail on that) -- this is a live
        operator control over THIS process's own roots, gated on
        the dashboard's own explicit confirm() naming the real counts
        before this route is ever called, not a background/automated
        action an agent could trigger silently.

        BOTH ROOTS, 2026-09-17. It used to take only the throw packages,
        which left frame-ring captures -- the larger of the two, gigabytes
        at a time -- with nothing in the product that removes them, a few
        inches from the button that writes them. The route path is
        unchanged because every caller of it means "clear this rig"; what
        changed is that it now actually does. There is no retention
        policy behind it and deliberately so: if someone takes the time to
        save a capture, they take the time to pull it off the rig, and
        this button is how they clear the rig afterwards.

        Removes every `<package_root>/<session>/` directory (each
        session's full set of throw subdirectories at once, matching
        discover_packages()'s own `<package_root>/<session_id>/<throw_id>/`
        layout) -- package_root itself is never removed, only recreated
        empty if this leaves it missing. `deleted` counts individual
        throw packages (session/throw_id pairs), matching what the
        dashboard's own confirm() dialog counted, not session directories.
        Tolerant of a session partially removed by a concurrent process
        (e.g. a live capture_daemon.py writing a new package at the same
        moment) -- skips what's already gone rather than raising.

        Also resets each deleted session's persisted throw numbering
        (`handle_ready_to_capture()`'s `data/session_throw_counters/
        <session_id>.count`/`.generation`, see that function's own
        2026-08-17 dated comment for why the counter lives outside
        package_root at all). 2026-08-22 fix --
        without this, the counter survives a delete-all untouched (same
        process, same session_id, never re-bootstraps), so the NEXT dart
        after "delete everything" picked up wherever the old count left
        off instead of restarting at 001. Deliberately scoped to
        sessions this call ACTUALLY deleted (mirrors `deleted`/`errors`
        above -- a session that failed to rmtree keeps its numbering
        too, so a still-real throw on disk can never collide with a
        renumbered-from-0 new one).

        UPDATED, 2026-08-22, same conversation: the FIRST version of this
        fix reset the counter to 0 in place, which then pointed
        out (discussing the package-naming convention directly) still
        risks a real collision if some of a session's throws were
        already pulled off the rig before the rest got deleted here -- this
        process has no way to know what's already archived elsewhere.
        Now calls `_reset_session_throw_numbering()` (shared with manual
        Reset's identical need, and with handle_ready_to_capture()'s own
        auto-detected "packages got pulled and there's nothing left
        locally" case) -- bumps a counting GENERATION rather than
        resetting in place, so the next throw_id is provably disjoint
        from anything already on disk or already archived. See that
        function's own docstring for the full collision analysis. The
        earlier "accepted tradeoff" this docstring used to describe is
        CLOSED, not just accepted -- worth recording that this was a
        real, two-step fix, not right on the first attempt.
        """
        # ONE REFUSAL FOR BOTH HALVES, taken before anything is removed.
        # Refusing the whole call rather than "packages yes, captures no"
        # keeps the button's meaning intact: it clears the rig, or it
        # explains why it did not.
        in_flight = _capture_write_in_flight()
        if in_flight is not None:
            reason = (
                f"a capture is being written right now ({in_flight}). Nothing "
                "was deleted -- wait for it to finish and press Delete again."
            )
            log.warning("delete recorded data REFUSED: %s", reason)
            return {"ok": False, "reason": reason, "deleted": 0,
                    "packages": {"deleted": 0, "bytes": 0,
                                 "label": frame_ring.format_bytes(0)},
                    "captures": {"deleted": 0, "bytes": 0,
                                 "label": frame_ring.format_bytes(0)},
                    "bytes": 0, "label": frame_ring.format_bytes(0)}

        root = state.package_root
        deleted = 0
        package_bytes = 0
        errors: list[str] = []
        counters_dir = root.parent / "session_throw_counters"
        for session_dir in sorted(root.iterdir()) if root.exists() else []:
            if not session_dir.is_dir():
                continue
            throw_count = sum(1 for _ in session_dir.glob("*/meta.json"))
            # Measured immediately before the rmtree, per session, for the
            # same reason the capture half does it: a session that fails
            # to delete has freed nothing, and a total taken up front
            # would report bytes the operator never got back.
            session_bytes = throw_capture_mod.dir_size_bytes(session_dir)
            try:
                shutil.rmtree(session_dir)
                deleted += throw_count
                package_bytes += session_bytes
                # _THROW_NUMBER_LOCK, 2026-09-06/07 (opendarts.live.
                # capture_daemon, Part 2 of the Zeus-latency follow-up
                # task -- see that lock's own module-level comment): this
                # HTTP-request-handling thread can now race a live
                # session's own in-flight background handle_ready_to_
                # capture() call, which holds the SAME lock for its own
                # throw_number allocation.
                with _THROW_NUMBER_LOCK:
                    _reset_session_throw_numbering(counters_dir, session_dir.name)
            except FileNotFoundError:
                pass # already gone -- a concurrent process's own cleanup, not an error here
            except Exception as exc: # noqa: BLE001 -- report, don't let one bad dir abort the rest
                errors.append(f"{session_dir.name}: {exc}")
        root.mkdir(parents=True, exist_ok=True)

        # THE OTHER ROOT. Off the event loop: emptying a capture root is
        # real, multi-gigabyte file I/O, and doing it inline would stall
        # every open tab's websocket for the duration.
        captures_deleted, capture_bytes, capture_errors = await asyncio.to_thread(
            _delete_captures
        )
        errors.extend(capture_errors)

        # list_packages() returns a periodically-refreshed CACHE (see
        # _package_poll_loop()), not a fresh read -- broadcasting that
        # directly here would still show the just-deleted packages until
        # the next poll tick. Refresh + update the cache the same way
        # every other package-mutating code path in this class already
        # does (see e.g. _attach_ad_ground_truth_from_ws() immediately
        # above) so GET /api/packages and every connected tab agree with
        # what's actually on disk right now, not stale-until-next-poll.
        current = await asyncio.to_thread(discover_packages, state.package_root)
        state._packages_cache = current
        state._known_package_paths = {p["path"] for p in current}
        # `count` is what makes the OTHER screens drop what was deleted.
        # Every client MERGES `packages` into what it already holds, so an
        # empty list removes nothing on its own; the count mismatch is what
        # triggers their full replace (checkPackageCountAndResync). Without
        # it, only the screen that pressed Delete emptied -- every other
        # dashboard, a kiosk included, kept showing the old darts until
        # someone reloaded it.
        await state._broadcast({  # noqa: SLF001 -- same module, intentional reuse
            "type": "PACKAGES_UPDATED",
            "packages": current,
            "count": len(current),
            "new_count": 0,
        })
        total_bytes = package_bytes + capture_bytes
        log.info(
            "delete recorded data: %d throw package(s) (%s) and %d capture(s) (%s) "
            "removed from %s and %s",
            deleted, frame_ring.format_bytes(package_bytes),
            captures_deleted, frame_ring.format_bytes(capture_bytes),
            root, state.capture_root,
        )
        # `deleted` is still the THROW PACKAGE count and still at the top
        # level: it is what every existing caller of this route reads, and
        # renaming a field to make a shape prettier is how a caller finds
        # out about a change by breaking. The per-kind detail is added
        # beside it, not in place of it.
        result: dict[str, Any] = {
            "ok": not errors,
            "deleted": deleted,
            "packages": {
                "deleted": deleted,
                "bytes": package_bytes,
                "label": frame_ring.format_bytes(package_bytes),
                "root": str(root),
            },
            "captures": {
                "deleted": captures_deleted,
                "bytes": capture_bytes,
                "label": frame_ring.format_bytes(capture_bytes),
                "root": str(state.capture_root),
            },
            "bytes": total_bytes,
            "label": frame_ring.format_bytes(total_bytes),
        }
        if errors:
            result["reason"] = "; ".join(errors)
        return result

    @app.get("/api/cameras/status")
    async def api_cameras_status() -> dict[str, Any]:
        """Per-camera CameraStatus (backend used, negotiated resolution/
        fps, open/first-frame latency, frame count, last error) from
        opendarts.live.local_capture.LocalCameraHub -- real data that
        already existed in the hub but wasn't surfaced to the dashboard
        before this endpoint. See AppState.cameras_status_dict()."""
        return state.cameras_status_dict()

    @app.get("/api/logs/{name}")
    async def api_logs(name: str, n: int = DEFAULT_LOG_TAIL_LINES) -> Response:
        """Read-only tail of one of this project's live-process log files
        (opendarts.live.logging_setup.DEFAULT_LOG_DIR/<name>.log), as plain
        text -- reachable with a bare `curl`/browser fetch and zero
        elevated permissions. `name` must be one of
        VALID_LOG_NAMES (the exact log_name each entrypoint's own
        configure_console_and_file_logging() call uses) -- anything else
        is a 404, not an arbitrary-file-read path. `n` (query param,
        default DEFAULT_LOG_TAIL_LINES) is how many trailing lines to
        return -- a tail, not a full-file dump, since these logs
        accumulate across a long-running process's whole lifetime.
        Missing file (that entrypoint has never run in this environment)
        is NOT an error -- returns 200 with an explanatory body, same
        graceful-degrade convention as the rest of this server.
        """
        if name not in VALID_LOG_NAMES:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": f"unknown log name {name!r} -- must be one of {VALID_LOG_NAMES}",
                },
                status_code=404,
            )
        log_path = DEFAULT_LOG_DIR / f"{name}.log"
        lines = await asyncio.to_thread(_tail_log_lines, log_path, n)
        if not log_path.exists():
            body = f"(no log file yet at {log_path} -- {name} may not have run in this environment)\n"
        else:
            body = "\n".join(lines) + ("\n" if lines else "")
        return Response(
            content=body,
            media_type="text/plain",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    def _azimuth_deg(centre) -> float:
        """Camera azimuth about the board normal, normalised to [0, 360)."""
        import math as _math

        deg = _math.degrees(_math.atan2(float(centre[1]), float(centre[0]))) % 360.0
        return 0.0 if deg >= 360.0 else deg

    @app.get("/api/calibration")
    async def api_calibration_get() -> dict[str, Any]:
        """The FULL adopted calibration -- solved pose included -- for
        every camera the capture loop is currently scoring against.

        `/api/state`'s own `calibration.cameras` is a DISPLAY shape built
        for the dashboard's badges: ok / reprojection_error_px /
        landmark_spread_ok and nothing more. The pose behind it
        (camera_matrix, dist_coeffs, rvec, tvec) has always existed in
        CalibrationStore and been written into every calibration package,
        but had no HTTP surface -- and calibration packages have no route
        either, so on a rig without shell access the numbers were
        unreachable. Answering "how far apart are the cameras" then meant
        re-solving poses from snapshots off the preview endpoint, which
        is a reconstruction of this data rather than this data.

        Read-only, and derived values are marked as such. `position_mm`
        is the camera centre in BOARD coordinates (-R^T t, millimetres,
        origin at the bull, z along the board normal); `azimuth_deg` /
        `elevation_deg` / `distance_mm` are that same point in spherical
        form, because "which angle is this camera at" is the question
        that actually gets asked and deriving it from rvec/tvec is a step
        every caller would otherwise repeat.
        """
        # Local imports, matching this module's own pattern (cv2/numpy are
        # imported per-function here, never at module scope).
        import math

        import cv2
        import numpy as np

        if state.calibration_store is None:
            return {"ok": False, "cameras": {},
                    "reason": "no capture loop in this process to read a calibration from"}

        calibrations = state.calibration_store.get()
        meta = state.calibration_store.meta()
        cameras: dict[str, Any] = {}
        for cam, calib in sorted(calibrations.items()):
            rvec = np.asarray(calib.rvec, dtype=float).reshape(3)
            tvec = np.asarray(calib.tvec, dtype=float).reshape(3)
            matrix = np.asarray(calib.camera_matrix, dtype=float)
            rot, _ = cv2.Rodrigues(rvec.reshape(3, 1))
            centre = (-rot.T @ tvec.reshape(3, 1)).ravel()
            distance = float(np.linalg.norm(centre))
            pnp = calib.pnp_result
            cameras[str(cam)] = {
                "camera_matrix": matrix.tolist(),
                "dist_coeffs": np.asarray(calib.dist_coeffs, dtype=float).ravel().tolist(),
                "rvec": rvec.tolist(),
                "tvec": tvec.tolist(),
                "focal_length_px": float(matrix[0][0]),
                "principal_point_px": [float(matrix[0][2]), float(matrix[1][2])],
                "reprojection_error_px": (
                    float(pnp.reprojection_error_px)
                    if pnp is not None and pnp.reprojection_error_px is not None else None
                ),
                "inlier_fraction": (
                    float(pnp.inlier_fraction)
                    if pnp is not None and pnp.inlier_fraction is not None else None
                ),
                "landmark_spread_ok": calib.landmark_spread_ok,
                # -- derived from rvec/tvec, not stored --
                "position_mm": centre.tolist(),
                "distance_mm": distance,
                # [0, 360), never 360.0: a hair-negative atan2 result
                # (-1e-17 for a camera sitting on the +x axis) rounds UP
                # to exactly 360.0 under `% 360.0`, which is outside the
                # range this field documents and sorts to the wrong end.
                "azimuth_deg": _azimuth_deg(centre),
                "elevation_deg": (
                    math.degrees(math.asin(max(-1.0, min(1.0, centre[2] / distance))))
                    if distance > 0 else None
                ),
            }
        return {"ok": True, "cameras": cameras, **meta}

    @app.get("/api/calibration/progress")
    async def api_calibration_progress() -> dict[str, Any]:
        """How far along the current (or last) calibration is -- the same
        body CALIBRATION_PROGRESS pushes, for a page that opens mid-way."""
        return {"ok": True, **calibration_progress.PROGRESS.snapshot()}

    @app.post("/api/calibration/refresh")
    async def api_calibration_refresh() -> dict[str, Any]:
        """The ONLY way calibration ever updates in this server (no
        background poll anymore, see AppState.refresh_calibration's own
        docstring) -- the Cameras tab's "Refresh calibration" button.
        When this server is part of opendarts.live.run_product's combined
        process, this doesn't just refresh what's displayed here -- it
        replaces the LIVE calibration the capture loop scores every
        subsequent throw against (opendarts.live.capture_daemon.
        CalibrationStore) and persists a durable record to disk."""
        # LOGGED ON ARRIVAL, and this line exists because of a real
        # 48-second mystery on 2026-09-15. An operator clicked Calibrate
        # at 14:44:44 and the first calibration log line appeared at
        # 14:45:32. Nothing in between could say whether the request had
        # arrived and was waiting, or had not arrived at all -- HTTP
        # access logging is off, so the endpoint was invisible until the
        # work itself started talking. Two very different faults (a
        # starved browser connection pool vs. a server sitting on the
        # request) were indistinguishable from the log, which made the
        # measurement impossible rather than merely hard.
        #
        # Three timestamps make it decidable: arrival here, the worker
        # actually starting (see _refresh_calibration_blocking), and
        # bootstrap_calibrations' own TOTAL. The gaps between them say
        # which layer is slow.
        t_arrived = time.monotonic()
        log.info("calibration refresh: request received")
        await state.refresh_calibration()
        log.info("calibration refresh: completed in %.2fs (request to response)",
                 time.monotonic() - t_arrived)
        await state._broadcast_calibration_status() # noqa: SLF001 -- same module, intentional reuse
        return {
            "checked_at_utc": state.calibration_checked_at_utc,
            "cameras": state.calibration_status,
            "live_source": (
                state.calibration_store.meta() if state.calibration_store is not None else None
            ),
            "calibration_error": state.calibration_error,
            "ring_geometry_relearned": state.calibration_geometry_relearned,
        }

    @app.post("/api/calibration/relearn-ring-geometry")
    async def api_relearn_ring_geometry() -> dict[str, Any]:
        """Delete this rig's learned ring geometry so it relearns from
        scratch -- the "the rig moved" button on a drift refusal.

        THE ACTION IS NOT NEW. When the gaps between cameras drift past
        RING_GEOMETRY_DRIFT_THRESHOLD_DEG, calibration is refused
        (capture_daemon's OrientationConsensusRefusedError) and the
        refusal message tells the operator to delete
        ring_geometry_fallback.json by hand. That recovery is proven --
        the rig relearns cleanly in about 10 events -- but it meant
        stopping to find a file on the rig, so the refusal read as a dead
        end. This is the same deletion, reachable.

        STILL HUMAN-CONFIRMED, deliberately. The refusal exists because
        drifted gaps usually mean a camera physically MOVED, and adopting
        the new geometry automatically would silently absorb a reading
        that might instead be a bad detection -- the "confidently wrong,
        not visibly wrong" failure the whole rig-consensus design exists
        to prevent. A button keeps a person in the loop while removing
        the busywork; it does not decide anything on its own.

        Idempotent: clearing when nothing was ever learned reports
        cleared=false rather than failing, so a double-click is harmless.
        """
        from opendarts.calibration.rig_move_reseed import clear_ring_geometry_on_reseed

        try:
            cleared = await asyncio.to_thread(
                clear_ring_geometry_on_reseed, DEFAULT_CALIBRATION_PACKAGE_ROOT
            )
        except Exception as exc: # noqa: BLE001 -- report, never 500 the dashboard
            log.exception("could not clear ring geometry")
            return {"ok": False, "cleared": False, "reason": str(exc)}
        log.warning(
            "ring geometry cleared by operator from the dashboard -- relearning "
            "from scratch on the next calibration (cleared=%s)", cleared,
        )
        return {
            "ok": True,
            "cleared": cleared,
            "reason": None if cleared else "no learned ring geometry was stored",
        }

    @app.post("/api/reset")
    async def api_reset() -> dict[str, Any]:
        """The sidebar's "Reset" button -- added 2026-08-12: takes the
        current image as the clean background (whether or not a dart is
        in it) and gets ready for the next throw -- the
        operator-triggered equivalent of the automatic post-takeout refresh
        (opendarts/capture/throw_trigger.py's module docstring "STALE
        BASELINE NEVER REFRESHED BUG").

        Semantics, unconditional, regardless of current dart_count/state:
        "whatever the board looks like right now, treat that as the clean
        baseline and get ready to score." This handler itself doesn't
        touch any frames -- it can't; frames only exist inside the
        capture loop's own thread/hub. It just writes a request onto the
        shared opendarts.live.capture_daemon.ResetRequest (mirrors the exact
        established CalibrationStore request/signal pattern, see that
        class's own docstring and AppState.reset_request's) -- the
        capture loop picks it up, at the latest, on its very next
        iteration (bounded by that loop's own poll_interval_s, typically
        well under a second) and does the actual re-baseline work there,
        where the frames and trigger state actually live.

        When no ResetRequest is wired (this module's own standalone CLI,
        no capture loop in-process at all) reports that honestly via
        `"loop_listening": false` rather than pretending the click did
        something -- same honest-degrade convention as
        /api/calibration/refresh's own `live_source: null` case."""
        if state.controller is not None:
            state.controller.touch() # a manual Reset counts as real activity
        if state.reset_request is None:
            return {"requested": False, "loop_listening": False}
        state.reset_request.request()
        return {"requested": True, "loop_listening": True, **state.reset_request.meta()}

    @app.post("/api/start")
    async def api_start() -> dict[str, Any]:
        """The sidebar's "Start" button -- added 2026-08-12: opens the
        camera hub, then
        starts a fresh capture-loop session. Auto-calibration follows an
        "only calibrate if geometry isn't already valid" rule
        -- see opendarts.live.capture_daemon.run_capture_loop_body's
        own "Calibration bootstrap" docstring section: a session reuses
        an already-populated CalibrationStore instead of re-bootstrapping,
        so this endpoint itself doesn't need to duplicate that decision --
        it just starts a session, and the session decides.

        Real, honest no-ops (not errors): no `state.controller` at all
        (this module's own standalone CLI); a session already running
        (idempotent -- `/api/start` is safely
        re-clickable); zero cameras opened (returns `ok: false` with the
        real per-camera flags AND `hub.status_report()`, same "no cameras
        opened" reason opendarts/live/run_product.py used to raise at
        process-startup before this feature -- now surfaced here instead,
        since cameras no longer open automatically at all, see this
        module's own docstring)."""
        if state.controller is not None:
            state.controller.touch()
        return await state.start_capture()

    @app.post("/api/stop")
    async def api_stop() -> dict[str, Any]:
        """The sidebar's "Stop" button -- see AppState.stop_capture()'s
        own docstring for the real mechanism (shared with the idle-
        timeout auto-stop below, ONE implementation). Real, honest no-op:
        no `state.controller` at all, or nothing currently running."""
        return await state.stop_capture(reason="manual")

    @app.get("/api/restart")
    async def api_restart_flags() -> dict[str, Any]:
        """What a restart would DO, without doing it -- the two launcher
        update flags out of data/config.json.

        A GET on an action route reports what the action is currently
        configured to do; it restarts nothing. It exists because the
        dashboard has one real decision to make before drawing the
        Maintenance section: an "Update and restart" button is
        meaningless on a rig whose launcher already pulls every time, and
        showing it there would imply the plain Restart button does not.

        Deliberately its own tiny route rather than another field bolted
        onto /api/state, for the same reason GET /api/diagnostics is its
        own route: a script asking "is this rig pinned or does it follow
        main" should not have to pull the whole state payload. It is also
        the obvious thing for the config document (GET /api/config) to
        absorb when that lands.
        """
        return {
            "ok": True,
            "always_update": always_update(),
            "update_on_next_restart": update_on_next_restart(),
        }

    @app.post("/api/restart")
    async def api_restart(body: "dict[str, Any] | None" = None) -> dict[str, Any]:
        """Programmatic replacement for the standing "ssh in, find the
        PID, kill it" restart procedure
        mid-incident: "add a new API /kill or something that stops the
        process so we can programmatically kill it instead of having to
        ssh in."

        Sends THIS process a real SIGTERM after `RESTART_SIGTERM_DELAY_S`
        (long enough for this HTTP response to actually finish flushing
        to the caller first) -- the EXACT SAME signal
        `opendarts.live.run_product._install_signal_handlers()` already
        handles for a real `kill <PID>`, so a call here goes through that
        same already-tested clean-shutdown path (capture loop stopped,
        AD WS listener stopped, server stopped, hub closed -- see
        `run_product.shutdown()`'s own docstring) rather than this
        endpoint inventing a second one. `os.kill` on a POSIX PID this
        process itself owns needs no elevated permission -- no
        subprocess spawn, no sandbox/approval friction, callable from
        anywhere that can reach this HTTP API (no SSH session needed).

        IMPORTANT, same caveat as the manual procedure it replaces: this
        endpoint does NOT itself relaunch the process. It relies on
        the rig's own external `while true` auto-restart loop (already
        running independently, outside this process, outside this
        module's knowledge) to pick the process back up -- calling this
        with no such loop actually running just stops the process, same
        as a manual `kill` would. This module has no way to verify that
        loop exists or is healthy; the honest answer this endpoint gives
        is "SIGTERM sent," not "restarted."

        Works regardless of whether a capture loop is wired in this
        process (unlike /api/start-/api/stop-/api/reset, which act ON
        the capture loop specifically) -- this acts on the WHOLE
        process, so it's available even for this module's own standalone
        CLI with no capture loop at all.

        OPTIONAL BODY, ADDED 2026-09-17: `{"update": true}` sets
        `update_on_next_restart` in data/config.json first, which is what
        the launcher reads to decide whether to `git pull` before
        relaunching (opendarts/live/update_policy.py). NO BODY IS STILL
        THE WHOLE OF THE OLD BEHAVIOUR -- same restart, config file not
        touched, not even read. FlightDeck calls this route with no body
        and must keep working; this endpoint is not the place to discover
        that an optional feature was made mandatory.

        THE FLAG IS WRITTEN BEFORE THE SIGTERM IS SCHEDULED, and a write
        that cannot be verified REFUSES THE RESTART rather than going
        ahead without it. "Update and restart" that restarts without
        updating is worse than an error: the rig comes back looking
        exactly as it should, on the old code, and the next thing anyone
        does is wonder why their fix is not live. An unparseable
        config.json is the real case -- write_config_section() declines
        to overwrite one, and only logs.
        """
        update = False
        if body:
            unknown = sorted(set(body) - {"update"})
            if unknown:
                return {
                    "ok": False,
                    "restarting": False,
                    "reason": f"unknown field(s): {', '.join(unknown)} -- "
                              "the only accepted field is 'update' (bool)",
                }
            raw = body.get("update", False)
            # bool is an int subclass and "true" is a truthy string, so
            # `bool(raw)` would accept both `1` and `"false"`. This
            # endpoint moves a rig onto different code; it does not guess.
            if not isinstance(raw, bool):
                return {
                    "ok": False,
                    "restarting": False,
                    "reason": f"'update' must be true or false, got {raw!r}",
                }
            update = raw

        if update and not set_update_on_next_restart(True):
            return {
                "ok": False,
                "restarting": False,
                "update": False,
                "reason": "could not record the update request in config.json "
                          "-- check that it parses as JSON. NOT restarting, "
                          "because a restart now would come back on the same code.",
            }

        pid = os.getpid()
        log.warning(
            "api_restart: /api/restart called (update=%s) -- sending SIGTERM to self "
            "(pid=%d) in %.1fs via the standard signal-handler shutdown "
            "path; relies on an external auto-restart loop to relaunch "
            "(this endpoint does not itself relaunch anything)",
            update, pid, RESTART_SIGTERM_DELAY_S,
        )
        threading.Timer(
            RESTART_SIGTERM_DELAY_S, os.kill, args=(pid, signal.SIGTERM),
        ).start()
        return {
            "ok": True,
            "restarting": True,
            "update": update,
            "pid": pid,
            "message": (
                f"SIGTERM scheduled in {RESTART_SIGTERM_DELAY_S}s -- this "
                "process will exit via the same clean-shutdown path a "
                "manual `kill` triggers. Relies on an external "
                "auto-restart loop to relaunch it; this endpoint does not "
                "relaunch anything itself."
                + (" The launcher will pull before relaunching." if update else "")
            ),
        }

    @app.get("/api/frame-health")
    async def api_frame_health() -> dict[str, Any]:
        """Are frames keeping up, on both sides of the pipeline?

        Two distinct questions that look alike and fail differently:

        `capture` -- is the pump reading every frame the camera produces?
        A camera configured for 30fps that delivers 21 is degrading
        silently, because the driver drops those frames before this
        process sees them. There is no counter for it; the rate deficit is
        the only evidence, which is why effective_fps exists.

        `publish` -- is the virtual-camera consumer collecting every frame
        we hand it? Counting what we published proves only that we
        published. The consumer reports gaps in frame_index back through
        shared memory, and a gap is direct evidence of a dropped frame
        rather than an inference from rates.
        """
        cameras = []
        if state.hub is not None:
            for idx, st in sorted(state.hub.status.items()):
                requested = st.requested_fps or 0
                effective = st.effective_fps
                # Only judge once a rate has actually been measured --
                # "behind" on a camera that has not completed a window yet
                # would fire on every start.
                keeping_up = None
                if effective is not None and requested:
                    keeping_up = effective >= (requested * 0.9)
                cameras.append({
                    "camera": idx,
                    "device": st.device,
                    "opened": st.opened,
                    "backend": st.backend_used,
                    "requested_fps": requested or None,
                    "effective_fps": effective,
                    "keeping_up": keeping_up,
                    "frame_count": st.frame_count,
                    "last_read_ok": st.last_read_ok,
                })

        publish: dict[str, Any] = {"enabled": False, "slots": []}
        vset = getattr(state, "vcam_set", None)
        if vset is not None:
            publish = {"enabled": True, "slots": vset.stats()}
            # Where publishing runs off the capture thread, `dropped` is
            # the only place a publisher that cannot keep up shows up at
            # all -- the per-slot counts only ever see frames that made it
            # as far as a device.
            worker = getattr(vset, "worker_stats", None)
            if callable(worker):
                try:
                    publish["worker"] = worker()
                except Exception: # noqa: BLE001 -- a diagnostic must not break the diagnostic
                    pass

        return {
            "ok": True,
            "capture": cameras,
            "publish": publish,
            "frame_sink_errors": getattr(state.hub, "frame_sink_errors", 0)
                                 if state.hub is not None else 0,
            # IS A SINK EVEN ATTACHED. frame_sink_errors cannot answer
            # this: 0 means both "never called" and "called and fine", so
            # a rig publishing nothing looks identical to one publishing
            # perfectly. That ambiguity cost a debugging session on a fresh
            # Windows box -- capture running, a publisher set built, and
            # nothing connecting the two.
            "frame_sink_attached": _frame_sink_attached(state.hub),
        }

    def _set_virtual_camera_publishing(enabled: bool) -> None:
        """Start or stop publishing to match the Autodarts toggle, with no restart.

        BUILDS THE SET ON DEMAND, 2026-09-15. This used to return early
        when `state.vcam_set` was None, which is exactly the state a rig
        is in when the comparison was OFF at startup -- so switching it on
        attached nothing, reported nothing, and the operator had to
        restart the process to get publishing. `vcam_set_factory` (from
        run_product) is what makes "on" mean on.

        AND RELEASES THE DEVICES ON THE WAY OUT, not just the sink. On
        Linux that is load-bearing rather than tidy: v4l2loopback ignores
        `VIDIOC_S_FMT` while any consumer holds the device open, silently
        keeping the old format (see `v4l2_publish._set_format`), so a
        writer left holding fds is a format that cannot be re-negotiated
        next time. Closing also stops the publish worker thread, so
        repeated toggling cannot accumulate threads or fds.

        Still a real no-op where publishing was never possible -- macOS
        has no virtual cameras and a rig that opted out has no factory
        answer -- rather than an error.
        """
        if state.hub is None:
            return
        setter = getattr(state.hub, "set_frame_sink", None)
        if setter is None:
            return
        if enabled:
            if state.vcam_set is None:
                factory = getattr(state, "vcam_set_factory", None)
                if factory is None:
                    return
                try:
                    state.vcam_set = factory()
                except Exception: # noqa: BLE001 -- publishing must never break the toggle
                    log.exception("could not build the virtual cameras -- "
                                  "the Autodarts toggle still applied")
                    return
                if state.vcam_set is None:
                    return
            setter(state.vcam_set.publish_all)
        else:
            # Detach FIRST, close second: a pump cycle already in flight
            # gets a set that is closed but still answers publish_all
            # harmlessly (both backends return 0 once closed), where the
            # other order would hand it a half-torn-down one.
            setter(None)
            # ONLY RELEASE WHAT WE KNOW HOW TO REBUILD. Without a factory
            # -- a set handed in by a caller that owns it, which is every
            # test stand-in and any embedder -- closing would make "off"
            # permanent, because switching back on has nothing to build
            # from. Detaching the sink already stops every frame; holding
            # the devices is the lesser wrong of the two.
            if getattr(state, "vcam_set_factory", None) is not None:
                vset, state.vcam_set = state.vcam_set, None
                closer = getattr(vset, "close", None)
                if callable(closer):
                    try:
                        closer()
                    except Exception: # noqa: BLE001
                        log.exception("closing the virtual cameras failed -- "
                                      "publishing is stopped regardless")
        log.info("virtual-camera publishing %s (follows the Autodarts toggle)",
                 "enabled" if enabled else "disabled")

    # -- THE CONFIG DOCUMENT -----------------------------------------------
    #
    # ONE document, two verbs, replacing eight route pairs on 2026-09-17:
    # /api/ad-config, /api/camera-devices, /api/store-packages,
    # /api/frame-ring, /api/port, /api/diagnostics, /api/detection-time and
    # /api/idle-timeout. Each of those was a private little contract with
    # its own body shape, its own idea of a valid value and its own
    # spelling of "saved but not applied"; one of them (idle-timeout) had
    # no GET at all, and half of them were polled by the dashboard on the
    # same tick. Six more settings -- engine_config, cv2_num_threads,
    # v4l2_format, min_free_disk_gb, camera_resolutions,
    # reprojection_targets_px -- had no route whatsoever and could only be
    # reached by hand-editing the file.
    #
    # GET returns the whole EFFECTIVE config: every key the product reads,
    # carrying the value actually in force -- the file's where the file has
    # a usable one, the code default where it does not, and the LIVE value
    # for the handful of settings a running process holds in memory and can
    # legitimately disagree with the file about.
    #
    # PATCH takes a partial document and merges it. It is ALL-OR-NOTHING:
    # every key is validated before any key is written, so a request that
    # fumbles one field changes nothing rather than leaving the rig in a
    # half-configured state no response could describe. Unknown keys are
    # refused by name -- `opendarts.live.config_document.CONFIG_KEYS` is the
    # list of what a setting IS, which is what turns a typo into a refusal
    # instead of a silent write into the operator's own config file.
    #
    # NO COMPATIBILITY ALIASES. The retired routes are gone, deliberately:
    # pre-ship is when a contract gets fixed, and an alias would mean
    # shipping both shapes forever.

    def _config_ctx() -> dict[str, Any]:
        """What the validators need to know about THIS rig.

        Only the frame-ring pricing so far: the cap refusal quotes the
        memory the operator was about to spend, which needs this rig's own
        measured bytes/second and its real slot count.
        """
        n_slots = len(state.hub.configs) if state.hub is not None else state.n_cameras
        per_second, _source = _frame_ring_bytes_per_second(n_slots)
        return {"n_slots": n_slots, "frame_ring_bytes_per_s": per_second}

    def _slot_urls_or_none(urls: "list[str | None] | None") -> "list[str | None] | None":
        """All-local reads as None -- the same "no override" the file uses.

        Without this a rig that has never assigned a stream would report
        `camera_urls: [null, null, null]`, and a PATCH echoing the document
        back would write that list into the config file: three nulls where
        the key had simply been absent.
        """
        if urls is None:
            return None
        return list(urls) if any(u is not None for u in urls) else None

    def _live_config_overrides() -> dict[str, Any]:
        """The keys whose value in force lives in MEMORY, not the file.

        Five settings can legitimately differ from `data/config.json` for
        the life of one process -- the AD listener's own URL and switch,
        the capture controller's idle timeout, the lifecycle store's
        detection speed, the diagnostics gate -- plus the camera assignment,
        which is read off the hub for the same reason `GET
        /api/camera-devices` always did: what the process is ACTUALLY using
        is the thing an operator most needs to see plainly.
        """
        out: dict[str, Any] = {"diagnostics": dict(diagnostics_gate.meta())}
        listener = state.ad_ws_listener
        if listener is not None:
            out["ad_enabled"] = listener.is_enabled()
            out["ad_base_url"] = listener.base_url
        # Duck-typed rather than assumed: a controller stand-in with only
        # the two methods a test needs is a real caller here, and a
        # config document that 500s on one would be a worse contract than
        # one that falls back to the file.
        if hasattr(state.controller, "get_idle_timeout_sec"):
            out["idle_timeout_sec"] = state.controller.get_idle_timeout_sec()
        if state.lifecycle_settings_store is not None:
            out["lifecycle_settings"] = {
                "dart_stable_frames":
                    state.lifecycle_settings_store.get().dart_stable_frames,
            }
        if state.hub is not None:
            out["camera_devices"] = [cfg.device for cfg in state.hub.configs]
            out["camera_urls"] = _slot_urls_or_none(
                _live_slot_urls(state.hub, len(state.hub.configs))
            )
        return out

    def _effective_config() -> dict[str, Any]:
        return config_document.effective_document(
            _config_ctx(), _live_config_overrides()
        )

    def _pending_restart_keys() -> list[str]:
        """Keys whose SAVED value is not the one this process is running.

        Deliberately only the keys where the live value is KNOWABLE. A
        change to `cv2_num_threads` or `v4l2_format` is reported as
        restart-required by the PATCH that made it (the registry says the
        key is read at startup), but nothing can later look at a running
        process and tell whether the file has moved on since -- and a list
        that guessed would be worse than one that is short and true.
        """
        ctx = _config_ctx()
        pending: list[str] = []
        if state.port is not None and config_document.effective_value("port", ctx) != state.port:
            pending.append("port")
        if state.host is not None and config_document.effective_value("host", ctx) != state.host:
            pending.append("host")
        if state.hub is not None:
            live_devices = [cfg.device for cfg in state.hub.configs]
            saved_devices = config_document.stored_value("camera_devices")
            if isinstance(saved_devices, list) and saved_devices != live_devices:
                pending.append("camera_devices")
            saved_urls = config_document.stored_value("camera_urls")
            if isinstance(saved_urls, list) and _slot_urls_or_none(
                [u or None for u in saved_urls]
            ) != _slot_urls_or_none(_live_slot_urls(state.hub, len(live_devices))):
                pending.append("camera_urls")
        return pending

    async def _config_runtime(refresh_camera_names: bool = False) -> dict[str, Any]:
        """The facts each Config-tab panel needs that are NOT settings.

        The port actually bound, whether Autodarts is reachable, what
        hardware exists at each device index, what the ring is holding
        right now, whether a session is running. None of it is writable and
        none of it belongs in the document -- but every one of them was
        carried by a retired route's GET, and a Config tab that had to ask
        five more endpoints for them would have gained nothing from this
        change.
        """
        runtime: dict[str, Any] = {
            "port": {
                "active": state.port,
                "default": DEFAULT_PORT,
                "configured": config_document.stored_value("port") is not None,
            },
            "host": {"active": state.host},
            "store_packages": {
                "running": bool(state.controller is not None
                                and state.controller.is_running()),
            },
            "frame_ring": _frame_ring_payload(),
            "engine_config": {"available_engines": engine_names()},
        }

        # Stamped -- see AppState.ad_connection_snapshot() for the race.
        runtime["ad"] = state.ad_connection_snapshot()

        if state.lifecycle_settings_store is not None:
            meta = state.lifecycle_settings_store.meta()
            runtime["lifecycle_settings"] = {"min": meta["min"], "max": meta["max"]}
        else:
            from opendarts.lifecycle.settings import (
                DART_STABLE_FRAMES_MAX, DART_STABLE_FRAMES_MIN,
            )
            runtime["lifecycle_settings"] = {
                "min": DART_STABLE_FRAMES_MIN, "max": DART_STABLE_FRAMES_MAX,
                "available": False,
            }

        if state.hub is None:
            runtime["cameras"] = {
                "available": False,
                "reason": "no camera hub in this process",
            }
        else:
            # Enumeration runs in a worker thread because the first call
            # spawns ffmpeg / binds COM, and the event loop must not stall
            # behind a cosmetic lookup. Cached after that, which is what
            # makes it safe on a polled route; ?refresh_camera_names=true
            # after plugging or unplugging one.
            naming = await asyncio.to_thread(
                camera_names.enumerate_devices, refresh_camera_names
            )
            live = [cfg.device for cfg in state.hub.configs]
            saved = config_document.stored_value("camera_devices")
            saved_urls = config_document.stored_value("camera_urls")
            runtime["cameras"] = {
                "available": True,
                "devices": live,
                "urls": _live_slot_urls(state.hub, len(live)),
                "saved": saved if isinstance(saved, list) else None,
                "saved_urls": saved_urls if isinstance(saved_urls, list) else None,
                "device_names": naming.names,
                # Offered in no picker: selecting one reads our own output.
                "own_virtual_devices": camera_names.own_virtual_devices(list(naming.names)),
                "device_count": naming.count,
                # Says whether name-at-index provably matches what OpenCV
                # opens at that index (Windows/Linux yes, macOS
                # best-effort) -- the UI must not present a hint as a fact.
                "names_authoritative": naming.authoritative,
                "names_source": naming.source,
            }
        return runtime

    async def _apply_camera_assignment(
        cleaned: dict[str, Any]
    ) -> "tuple[bool, str | None, bool]":
        """Push a new slot -> device/URL assignment onto the RUNNING hub.

        Lifted whole from the retired `POST /api/camera-devices`, rules
        intact. The hub is mutated in place (LocalCameraHub.reconfigure), so
        the capture thread and the app keep the reference they already hold.
        It refuses while cameras are open -- the open captures ARE the old
        assignment -- which is exactly when the previews an operator is
        comparing against exist, so the refusal is answered by cycling the
        loop: stop, reconfigure, start. Never while darts are on the board.
        """
        if state.hub is None:
            return False, "no camera hub in this process", False
        live_devices = [cfg.device for cfg in state.hub.configs]
        devices = cleaned.get("camera_devices", live_devices)

        # WHICH URLS TO APPLY when the request said nothing about streams:
        # the LIVE assignment, not "every slot local". Passing None straight
        # through would silently drop a working stream because someone moved
        # an unrelated device dropdown.
        if "camera_urls" in cleaned:
            apply_urls = cleaned["camera_urls"] or [None] * len(devices)
        else:
            apply_urls = _live_slot_urls(state.hub, len(devices))
        # Pad/trim to the new slot count -- the device list may have grown
        # or shrunk in this same request.
        apply_urls = (list(apply_urls) + [None] * len(devices))[:len(devices)]

        # Re-read rather than reuse the hub's current widths/heights:
        # camera_resolutions is keyed by DEVICE index, and the devices are
        # exactly what may just have changed, so the old per-slot values no
        # longer describe the right hardware.
        resolutions_raw = config_document.effective_value("camera_resolutions",
                                                          _config_ctx())
        resolutions: dict[int, Any] = {}
        try:
            from opendarts.live.camera_resolution import parse_resolution_preference

            for cam_key, pref in (resolutions_raw or {}).items():
                resolutions[int(cam_key)] = parse_resolution_preference(pref)
        except Exception as exc:  # noqa: BLE001 -- a bad config must not block a reassignment
            log.warning("could not read camera_resolutions, using defaults: %s", exc)
            resolutions = {}
        configs = local_capture.camera_configs_from_resolution_preferences(
            resolutions, devices=devices
        )
        try:
            _reconfigure_hub(state.hub, configs, apply_urls)
            return True, None, False
        except RuntimeError:
            pass

        # trigger_dart_count, NOT visit_throws. visit_throws is the RETAIL
        # visit list and survives a visit that never got a clean takeout, so
        # it reports darts on an empty board and would block a reassignment
        # for a game that finished long ago. Also gated on the loop RUNNING:
        # with capture stopped there is no visit to interrupt.
        running = state.controller is not None and state.controller.is_running()
        on_board = state.trigger_dart_count or 0
        if running and on_board > 0:
            return False, (f"{on_board} dart(s) on the board -- take them out, "
                           "or press Reset if the board is already clear"), False
        log.info("reassigning cameras: cycling the capture loop")
        await state.stop_capture(reason="camera reassignment")
        try:
            _reconfigure_hub(state.hub, configs, apply_urls)
        except RuntimeError as exc:
            return False, str(exc), False
        await state.start_capture()
        return True, None, True

    async def _apply_config_changes(
        cleaned: dict[str, Any]
    ) -> "tuple[list[str], dict[str, str], dict[str, Any]]":
        """`(applied_live, notes, extra)` -- what really took effect now.

        A key that has no live half, or whose live half is not wired in
        this process, is reported in `notes` rather than quietly counted as
        applied. That distinction is the whole point of the reply: "saved"
        and "in force" are different facts, and every retired route had its
        own way of saying so.
        """
        applied: list[str] = []
        notes: dict[str, str] = {}
        extra: dict[str, Any] = {}

        listener = state.ad_ws_listener
        if "ad_base_url" in cleaned:
            if listener is None:
                notes["ad_base_url"] = ("Autodarts is not wired in this process -- "
                                        "saved for the next start")
            else:
                try:
                    listener.set_base_url(cleaned["ad_base_url"])
                    applied.append("ad_base_url")
                except ValueError as exc:
                    notes["ad_base_url"] = str(exc)
        if "ad_enabled" in cleaned:
            if listener is None:
                notes["ad_enabled"] = ("Autodarts is not wired in this process -- "
                                       "saved for the next start")
            else:
                enabled = cleaned["ad_enabled"]
                listener.set_enabled(enabled)
                # Virtual-camera publishing follows this switch: publishing
                # to a consumer nobody asked for copies megabytes per frame
                # for nothing, and a toggle that left it running would not
                # really be off. The DEVICES follow it too -- leaving them
                # registered puts three synthetic cameras with nothing
                # behind them in every capture app on the machine. Off-thread
                # because regsvr32 is a subprocess and this is the event loop.
                _set_virtual_camera_publishing(enabled)
                extra["virtual_cameras"] = await asyncio.to_thread(
                    vcam.register.apply, enabled
                )
                applied.append("ad_enabled")

        if "camera_devices" in cleaned or "camera_urls" in cleaned:
            ok_applied, reason, restarted = await _apply_camera_assignment(cleaned)
            for key in ("camera_devices", "camera_urls"):
                if key not in cleaned:
                    continue
                if ok_applied:
                    applied.append(key)
                else:
                    notes[key] = (reason or "not applied") + \
                        " -- saved, and it applies at the next start"
            if restarted:
                extra["restarted_capture"] = True

        if "idle_timeout_sec" in cleaned:
            if state.controller is None:
                notes["idle_timeout_sec"] = ("no capture loop in this process to "
                                             "configure -- saved for the next start")
            else:
                state.controller.set_idle_timeout_sec(cleaned["idle_timeout_sec"])
                applied.append("idle_timeout_sec")

        if "lifecycle_settings" in cleaned:
            if state.lifecycle_settings_store is None:
                notes["lifecycle_settings"] = ("no capture loop in this process to "
                                               "configure -- saved for the next start")
            else:
                state.lifecycle_settings_store.set(
                    dart_stable_frames=cleaned["lifecycle_settings"]["dart_stable_frames"],
                )
                applied.append("lifecycle_settings")

        if "diagnostics" in cleaned:
            # Takes effect on the very next capture-loop iteration -- a
            # plain threading.Event read fresh every time -- and also flips
            # the websockets/uvicorn logger levels in real time.
            diagnostics_gate.set_enabled(cleaned["diagnostics"]["enabled"])
            applied.append("diagnostics")

        if "frame_ring_seconds" in cleaned:
            # Applied live, never at the next restart: this setting is
            # holding gigabytes RIGHT NOW, and an operator who turns it
            # down and is told to restart to free 4GB has been handed a
            # control that does not do what it says.
            if _apply_frame_ring_seconds(state, cleaned["frame_ring_seconds"]):
                applied.append("frame_ring_seconds")
            else:
                notes["frame_ring_seconds"] = ("no frame ring in this process to "
                                               "resize -- saved for the next Start")

        if "min_free_disk_gb" in cleaned:
            # Both on-disk writers re-read this key per write, so persisting
            # it IS applying it.
            applied.append("min_free_disk_gb")

        if "store_packages" in cleaned:
            # No in-memory half at all: run_product reads this key ONCE per
            # session, deliberately, so a session cannot change its storage
            # behaviour halfway through its own package set.
            if state.controller is not None and state.controller.is_running():
                notes["store_packages"] = "applies at the next Start, not to this session"

        # A real operator action counts as activity -- the same touch() the
        # idle-timeout, detection-time and diagnostics routes each did.
        if state.controller is not None and (
            {"idle_timeout_sec", "lifecycle_settings", "diagnostics"} & set(cleaned)
        ):
            state.controller.touch()
        return applied, notes, extra

    @app.get("/api/config")
    async def api_config_get(refresh_camera_names: bool = False) -> dict[str, Any]:
        """The whole effective config, plus the runtime facts that are not
        settings.

        `config` is the document: every key the product reads, valued at
        what is actually in force. `restart_required` names the keys whose
        saved value is not the one this process is running -- the port
        somebody changed an hour ago, a camera assignment that has not been
        restarted into. `runtime` carries the rest of what the Config tab
        shows: the port really bound, whether Autodarts is reachable, what
        hardware exists at each device index, what the ring holds now.

        Cheap enough to poll: the camera enumeration behind
        `runtime.cameras.device_names` is cached (pass
        `?refresh_camera_names=true` after plugging one in), and every
        frame-ring figure is maintained incrementally as frames arrive.
        """
        return {
            "ok": True,
            "config": _effective_config(),
            "restart_required": _pending_restart_keys(),
            "runtime": await _config_runtime(refresh_camera_names),
        }

    @app.patch("/api/config")
    # `Any`, not `dict[str, Any]`: FastAPI would answer a list or a string
    # with its own 422 and a validation blob, where this route already has
    # a per-key error shape a client can read. One refusal format, ours.
    async def api_config_patch(body: Any = Body(default=None)) -> JSONResponse:
        """Merge a partial document into `data/config.json`.

        ALL-OR-NOTHING. Every key named is validated first; if any of them
        fails, NOTHING is written and the reply is a 400 carrying one
        reason per key, so a form with three fields knows which one it got
        wrong. Unknown keys and `capabilities` (written by the startup
        probe) are refused the same way.

        On success the reply carries the new effective document, the keys
        this request actually applied to the running process, and
        `restart_required` -- the keys it changed that cannot reach this
        process at all. `port` and `host` are always in that list when
        touched: uvicorn's listening socket is bound before this app object
        exists, and rebinding it under a live request would drop the very
        response that reports success.

        Persistence is PROVEN, not assumed: `write_config_section()`
        deliberately declines (and only logs) when the existing file cannot
        be parsed, so every key is read back after writing. A refused write
        is reported as the no-op it was.
        """
        ctx = _config_ctx()
        current = _effective_config()
        cleaned, errors = config_document.validate_patch(body, ctx, current)

        # Cross-key rule, checked here because it is the only one that
        # needs two keys at once: a URL list has one entry per slot, and
        # the slot count may itself be changing in this same request.
        if "camera_urls" in cleaned and cleaned["camera_urls"] is not None:
            slots = cleaned.get("camera_devices") or current.get("camera_devices") or []
            if slots and len(cleaned["camera_urls"]) != len(slots):
                errors["camera_urls"] = (
                    f"camera_urls must have one entry per camera slot "
                    f"({len(slots)}), null for a local slot"
                )
        if errors:
            log.info("config PATCH refused: %s", errors)
            return JSONResponse(
                {"ok": False, "errors": errors, "config": current,
                 "restart_required": _pending_restart_keys()},
                status_code=400,
            )

        # Persist BEFORE applying, and prove each write landed. The other
        # order would leave a rig running a setting its own config file
        # does not contain -- which is the state nobody can diagnose later.
        failed: dict[str, str] = {}
        persisted: list[str] = []
        for key in cleaned:
            spec = config_document.KEYS_BY_NAME[key]
            if not spec.persist:
                continue
            if config_document.persist(key, cleaned[key]):
                persisted.append(key)
            else:
                failed[key] = ("config.json could not be updated -- "
                               "check it parses as JSON")
        if failed:
            log.warning("config PATCH could not persist: %s", failed)
            return JSONResponse(
                {"ok": False, "errors": failed, "persisted": persisted,
                 "config": _effective_config(),
                 "restart_required": _pending_restart_keys()},
                status_code=500,
            )

        applied, notes, extra = await _apply_config_changes(cleaned)
        restart_required = sorted(
            {key for key in cleaned
             if config_document.KEYS_BY_NAME[key].restart or key in notes}
        )
        log.info("config PATCH: changed=%s applied_live=%s restart_required=%s",
                 sorted(cleaned), sorted(applied), restart_required)
        return JSONResponse({
            "ok": True,
            "changed": sorted(cleaned),
            "persisted": sorted(persisted),
            "applied_live": sorted(applied),
            "restart_required": restart_required,
            "notes": notes,
            "config": _effective_config(),
            "runtime": await _config_runtime(),
            **extra,
        }, status_code=200)

    # -- Throw capture ring ----------------------------------------------
    # The in-memory buffer of recent raw frames, and the two ways to flush
    # it to disk. See opendarts/capture/frame_ring.py for what it costs
    # and opendarts/capture/throw_capture.py for the two triggers.
    #
    # `frame-ring`, never just `ring`: every other use of the word "ring"
    # in this file means a dartboard ring (`ad_ring`, `BOARD_RINGS`,
    # `sector_ring_for_point`), and an `/api/ring` would read as a board
    # endpoint to everyone including whoever writes the next one.

    def _frame_ring_bytes_per_second(n_slots: int) -> "tuple[float, str]":
        """(bytes/second, "measured" | "pixels") for pricing the ring.

        The ring's own measured rate when it has one. The pixel formula
        overstated a passthrough rig ~36x (a Windows rig, 2026-09-17: 5s priced at
        1.24 GB, held in 34.6 MB). Uncompressed pixels at the ~32.5 sets/s
        the pump really achieves otherwise -- the worst case, which is the
        right thing to quote before anything is known."""
        service = state.throw_capture
        ring = getattr(service, "ring", None) if service is not None else None
        try:
            measured = ring.stats().get("measured_bytes_per_s") if ring is not None else None
        except Exception:  # noqa: BLE001 -- pricing must never break the page
            measured = None
        if measured:
            return float(measured), "measured"
        return frame_ring.estimated_bytes_per_second(n_slots, fps=32.5), "pixels"

    def _frame_ring_payload() -> dict[str, Any]:
        """One shape, used by the GET and by every POST's reply, so the
        dashboard's optimistic update and its next poll cannot disagree
        about what state the ring is in."""
        service = state.throw_capture
        n_slots = len(state.hub.configs) if state.hub is not None else state.n_cameras
        # The config registry is the one interpreter of this key, so the
        # price on the dashboard and the value the ring is built from can
        # never come from two different readings of the same file.
        configured = config_document.effective_value("frame_ring_seconds")
        per_second, per_second_source = _frame_ring_bytes_per_second(n_slots)
        payload: dict[str, Any] = {
            "ok": True,
            # "attached" is not the same question as "enabled": a rig
            # running the standalone server has no capture loop and so no
            # ring at all, which is a different fact from a ring that is
            # switched off, and collapsing the two is how an operator
            # concludes the feature is broken when it was never on.
            "attached": service is not None and service.ring is not None,
            "configured_seconds": float(configured),
            "n_slots": n_slots,
            "estimated_bytes_per_s": per_second,
            # "measured": what this rig's ring really holds per second --
            # camera JPEGs where a slot passes them through, ~40x smaller
            # than pixels. "pixels": the uncompressed worst case, used
            # until the ring has run long enough to measure.
            "estimate_source": per_second_source,
            "estimated_bytes": per_second * float(configured),
            "estimated_label": frame_ring.format_bytes(per_second * float(configured)),
            # What the LONGEST allowed window costs, from the same rate the
            # estimate uses. Replaced a "2GB = Ns" table, which on a
            # passthrough rig quoted windows of up to 20 minutes that the
            # 60-second cap then refused.
            "max_seconds": MAX_FRAME_RING_SECONDS,
            "max_label": frame_ring.format_bytes(per_second * MAX_FRAME_RING_SECONDS),
            "flight_note": FLIGHT_EXPECTATION_NOTE,
        }
        if service is not None:
            payload.update(service.status())
        else:
            payload["ring"] = {"enabled": False}
            payload["writer"] = {"busy": False, "current": None, "last": None}
            payload["captures"] = []
            payload["reason"] = (
                "no capture loop is running in this process, so nothing is "
                "filling a frame ring. Start the product via "
                "opendarts.live.run_product to get one."
            )
        return payload

    @app.post("/api/frame-ring/capture-missed")
    async def api_frame_ring_capture_missed(body: dict[str, Any]) -> dict[str, Any]:
        """The MISSED-DART button: write the whole buffer.

        Nothing can detect a missed dart without a second scorer to
        compare against, so on a rig without one this button is the only
        trigger there can be -- which is why it takes an operator reason: six months
        later "why is there a 6GB dump from Tuesday" has to be answerable
        from the dump itself.
        """
        service = state.throw_capture
        if service is None:
            return {"ok": False, "reason": _frame_ring_payload().get("reason")}
        reason = str(body.get("reason") or "").strip() or "operator pressed the button"
        result = await asyncio.to_thread(
            service.capture_missed_dart, reason=reason, source="manual",
        )
        if state.controller is not None:
            state.controller.touch()   # a real operator action counts as activity
        return {**result, **_frame_ring_payload(), "ok": result.get("ok", False)}

    def _package_dir(session: str, throw_id: str) -> "Path | None":
        """`session`/`throw_id` -> the package directory, or None if either
        is not a safe single path segment or the directory does not exist.
        Same validation as the misscore/mark-ad-wrong routes below."""
        for seg in (session, throw_id):
            if not seg or "/" in seg or "\\" in seg or ".." in seg:
                return None
        pkg = state.package_root / session / throw_id
        return pkg if pkg.is_dir() else None

    @app.get("/api/packages/{session}/{throw_id}/frame/{cam}/{kind}.png")
    async def api_package_frame(
        session: str, throw_id: str, cam: int, kind: str, fmt: str = "png"
    ):
        """One still of a throw: kind=`bg` (baseline) or `after` (the scored
        commit frame). `after` comes from the clip for a recorded package
        (byte-identical, via meta.video) or the dart PNG otherwise; `bg` is
        always the PNG.

        `fmt` picks the wire format: `png` (default) is full-resolution and
        lossless -- what the lightbox opens on click; `jpeg` is a smaller
        display-quality encode for the grid, so opening the viewer does not
        pull ~1MB per still. Both are the same 1280x720 pixels."""
        pkg = _package_dir(session, throw_id)
        if pkg is None:
            return Response("no such package", status_code=404)
        if kind not in ("bg", "after"):
            return Response("kind must be bg or after", status_code=400)
        if fmt not in ("png", "jpeg"):
            return Response("fmt must be png or jpeg", status_code=400)

        def _load() -> "bytes | None":
            import cv2
            # Load the frame as a lossless BGR array first, then encode to
            # the requested wire format.
            meta_p = pkg / "meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            video = meta.get("video")
            has_clip = bool(video) and str(cam) in (video or {}).get("cameras", {})
            arr = None
            if has_clip:
                from opendarts.capture import clip
                # Both ends of the clip are addressable now: bg_index for
                # the baseline, commit_index for the scored frame. bg comes
                # back None on a package whose bg was never de-duplicated
                # into the clip (or a throw-clip/v1 one), which then falls
                # through to the PNG below exactly as before.
                arr = (clip.read_bg_frame(pkg, video, cam) if kind == "bg"
                       else clip.read_commit_frame(pkg, video, cam))
            if arr is None:
                name = "bg" if kind == "bg" else "frame"
                arr = cv2.imread(str(pkg / f"cam{cam}_{name}.png"), cv2.IMREAD_COLOR)
            if arr is None:
                return None
            if fmt == "jpeg":
                ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            else:
                ok, buf = cv2.imencode(".png", arr)
            return buf.tobytes() if ok else None

        data = await asyncio.to_thread(_load)
        if data is None:
            return Response("no such frame", status_code=404)
        media = "image/jpeg" if fmt == "jpeg" else "image/png"
        return Response(content=data, media_type=media)

    @app.get("/api/packages/{session}/{throw_id}/clip/{cam}.mjpg")
    async def api_package_clip(session: str, throw_id: str, cam: int, request: Request):
        """The camera's clip as a looping MJPEG stream an <img> can play
        (browsers do not decode MKV/FFV1). Display quality only -- for
        byte-exact frames use the .png endpoint / replay."""
        pkg = _package_dir(session, throw_id)
        if pkg is None:
            return Response("no such package", status_code=404)

        def _load_jpegs() -> "list[bytes] | None":
            import cv2
            meta_p = pkg / "meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            if not _clip_mod.is_recorded_clip(meta.get("video")):
                # Same rule as clip.json: a stills clip is not footage.
                return None
            video = (meta.get("video") or {}).get("cameras", {})
            if str(cam) not in video:
                return None
            from opendarts.capture import clip
            frames = clip.read_clip_frames(pkg / video[str(cam)]["clip"])
            out = []
            for f in frames:
                ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    out.append(buf.tobytes())
            return out

        jpegs = await asyncio.to_thread(_load_jpegs)
        if not jpegs:
            return Response("no clip for this camera", status_code=404)

        async def _gen():
            # Loops the clip until the viewer window closes. The disconnect
            # check is what makes it terminate -- without it the generator
            # would run forever after the client is gone.
            while not await request.is_disconnected():
                for j in jpegs:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                    await asyncio.sleep(0.12)
                    if await request.is_disconnected():
                        break

        return StreamingResponse(
            _gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/packages/{session}/{throw_id}/clip/{cam}/{index}.png")
    async def api_package_clip_frame(session: str, throw_id: str, cam: int, index: int):
        """One clip frame at full resolution and lossless -- what the
        lightbox opens when a scrubber frame is clicked. The scrubber itself
        keeps using the compressed frames from clip.json for smooth
        scrubbing; this decodes just the one requested frame from the clip on
        demand (no extra storage). PNG re-encode of a frame that is itself
        lossless on disk (FFV1 on macOS, the camera's own MJPEG bytes
        elsewhere)."""
        pkg = _package_dir(session, throw_id)
        if pkg is None:
            return Response("no such package", status_code=404)

        def _load() -> "bytes | None":
            import cv2
            meta_p = pkg / "meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            video = (meta.get("video") or {}).get("cameras", {})
            if str(cam) not in video:
                return None
            from opendarts.capture import clip
            frames = clip.read_clip_frames(pkg / video[str(cam)]["clip"])
            if not (0 <= index < len(frames)):
                return None
            ok, buf = cv2.imencode(".png", frames[index])
            return buf.tobytes() if ok else None

        data = await asyncio.to_thread(_load)
        if data is None:
            return Response("no such clip frame", status_code=404)
        return Response(content=data, media_type="image/png")

    @app.get("/api/packages/{session}/{throw_id}/clip.json")
    async def api_package_clip_json(session: str, throw_id: str):
        """Every clip frame of every camera, as base64 JPEG data URLs, in
        one payload:

            {"fps": 30, "cameras": {"0": {"commit_index": i,
                                          "frames": ["data:image/jpeg;base64,..."]}}}

        The scrubber in the viewer fetches this ONCE and swaps <img> srcs
        client-side, so dragging the slider is instant and each clip is
        decoded only once (decoding per-frame on demand would re-decode the
        whole MKV every drag). Display quality only -- byte-exact frames
        come from the .png / replay path."""
        pkg = _package_dir(session, throw_id)
        if pkg is None:
            return JSONResponse({"reason": "no such package"}, status_code=404)

        def _load() -> "dict | None":
            import base64
            import cv2
            meta_p = pkg / "meta.json"
            meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
            video = meta.get("video") or {}
            cameras = video.get("cameras") or {}
            # Only a RECORDING has a clip worth scrubbing. A stills clip
            # is the bg and the scored frame -- the two frames the viewer's
            # stills grid already shows -- so offering it as "the clip"
            # would put a two-frame scrubber under every ordinary throw.
            # The 404 is what tells the viewer to leave the scrubber out.
            if not cameras or not _clip_mod.is_recorded_clip(video):
                return None
            from opendarts.capture import clip
            out: dict[str, Any] = {}
            for cam_key, entry in sorted(cameras.items(), key=lambda kv: int(kv[0])):
                frames = clip.read_clip_frames(pkg / entry["clip"])
                urls = []
                for f in frames:
                    ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ok:
                        urls.append("data:image/jpeg;base64," +
                                    base64.b64encode(buf.tobytes()).decode("ascii"))
                out[cam_key] = {
                    "commit_index": int(entry.get("commit_index", 0)),
                    "frames": urls,
                }
            return {"fps": int(video.get("fps") or 30), "cameras": out}

        data = await asyncio.to_thread(_load)
        if data is None:
            return JSONResponse({"reason": "no clip for this throw"}, status_code=404)
        return JSONResponse(data)

    @app.get("/packages/{session}/{throw_id}/viewer", response_class=HTMLResponse)
    async def api_package_viewer(session: str, throw_id: str) -> HTMLResponse:
        """A standalone, customer-facing viewer window for one throw:

        - a 2x3 stills grid: 3 cameras across, the reference (before) frame
          on the top row and the scored (after) frame on the bottom;
        - when the throw was recorded, a clip scrubber below it: one frame
          per camera, a slider to drag through the clip, play at
          0.25x / 0.5x / 1x (plays once, does not loop), and left/right
          arrow-key stepping. Frames come from /clip.json (fetched once)
          and are swapped client-side, so scrubbing is instant.

        Opened in a new window from the engine row."""
        pkg = _package_dir(session, throw_id)
        if pkg is None:
            return HTMLResponse("<h1>no such package</h1>", status_code=404)
        meta_p = pkg / "meta.json"
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        cams = meta.get("frame_cameras") or meta.get("cameras") or []
        # A RECORDED clip, not merely a clip: an unrecorded package has a
        # two-frame stills clip, and that is not something to scrub.
        has_clip = _clip_mod.is_recorded_clip(meta.get("video"))
        base = f"/api/packages/{session}/{throw_id}"

        # A short, human label for this throw: the trailing token of the
        # throw id (e.g. "...-001-T20" -> "T20"), and the throw number.
        parts = throw_id.split("-")
        score_label = parts[-1] if parts else throw_id
        throw_no = None
        if len(parts) >= 2 and parts[-2].isdigit():
            throw_no = int(parts[-2])
        ncols = max(1, len(cams))

        def _cells(kind: str) -> str:
            # Grid shows the compressed (jpeg) still; the lightbox opens the
            # full-resolution lossless png via data-full.
            return "".join(
                f'<figure class="still"><img class="zoomable" '
                f'src="{base}/frame/{c}/{kind}.png?fmt=jpeg" '
                f'data-full="{base}/frame/{c}/{kind}.png" '
                f'alt="camera {c} {kind}" loading="lazy">'
                f'<figcaption>Camera {c}</figcaption></figure>' for c in cams)

        # WHAT A PERSON LOOKING AT THIS THROW ACTUALLY WANTS TO KNOW, in place
        # of the package name that used to headline the page: when it was
        # thrown, how big the record is, and the rest of the visit it belonged
        # to. The visit is the navigation: people step through a turn's three
        # darts to see how it progressed, not from one turn to the next, so
        # there is deliberately no older/newer paging. All of it comes from the
        # package cache the dashboard already keeps plus one directory listing.
        def _label(tid: str) -> str:
            return tid.split("-")[-1] if tid else "?"

        def _viewer_href(rec: dict) -> str:
            return f"/packages/{rec['session']}/{rec['throw_id']}/viewer"

        records = state.list_packages()

        captured_iso = meta.get("captured_at_utc") or ""
        size_bytes = sum(f.stat().st_size for f in pkg.iterdir() if f.is_file())
        size_text = (f"{size_bytes / 1e6:.1f} MB" if size_bytes >= 1e6
                     else f"{size_bytes / 1e3:.0f} KB")
        detail_bits = [
            f'<time id="capturedAt" datetime="{html.escape(captured_iso)}">'
            f'{html.escape(captured_iso or "capture time unknown")}</time>',
            size_text,
            f"{len(cams)} camera{'s' if len(cams) != 1 else ''}",
            "recorded clip" if has_clip else "stills only",
        ]
        details_html = '<span class="sep">&middot;</span>'.join(
            f"<span>{b}</span>" for b in detail_bits)

        # The visit: every throw sharing this one's visit_id, in dart order,
        # padded to three so an unfinished visit reads as unfinished.
        visit_id = meta.get("visit_id")
        visit = sorted(
            (r for r in records if visit_id and r.get("session") == session
             and r.get("visit_id") == visit_id),
            key=lambda r: (r.get("visit_index") is None, r.get("visit_index") or 0),
        )
        visit_html = ""
        if visit:
            slots = []
            for r in visit[:3]:
                current = r.get("throw_id") == throw_id
                cls = "dart dart-now" if current else "dart"
                label = html.escape(_label(r.get("throw_id", "")))
                slots.append(f'<span class="{cls}" aria-current="true">{label}</span>' if current
                             else f'<a class="{cls}" href="{html.escape(_viewer_href(r))}">{label}</a>')
            slots += ['<span class="dart dart-empty">&mdash;</span>'] * (3 - len(slots))
            visit_html = ('<nav class="visit" aria-label="Darts in this visit">'
                          '<span class="visit-label">Visit</span>'
                          + "".join(slots) + "</nav>")

        ids_text = f"Session {session}" + (f" &middot; throw #{throw_no}" if throw_no is not None else "")

        no_clip_note = ("" if has_clip else
                        '<section class="panel"><p class="empty">No clip was '
                        "recorded for this throw — the reference and scored "
                        "stills above are the full record.</p></section>")

        scrubber = "" if not has_clip else f"""
<section class="panel" id="scrub" aria-label="Clip playback">
  <div class="panel-head">
    <h2>Clip</h2>
    <span id="scrubStatus" class="muted">Loading…</span>
  </div>
  <div id="scrubCams" class="grid" style="--n:{ncols}"></div>
  <div class="transport">
    <button id="btnPlay" class="btn btn-primary" type="button" aria-label="Play">
      <span class="ico">▶</span><span id="btnPlayLabel">Play</span>
    </button>
    <div class="speeds" role="group" aria-label="Playback speed">
      <button class="btn speed" data-speed="0.25" type="button">0.25×</button>
      <button class="btn speed" data-speed="0.5" type="button">0.5×</button>
      <button class="btn speed is-active" data-speed="1" type="button">1×</button>
    </div>
    <span id="frameLabel" class="frame-label">—</span>
  </div>
  <input id="slider" type="range" min="0" max="0" value="0" step="1"
         aria-label="Clip position" disabled>
  <p class="hint">Drag the slider, press <kbd>Play</kbd>, or use
     <kbd>←</kbd> <kbd>→</kbd> to step frame by frame. The highlighted cell
     marks each camera's scored frame.</p>
</section>
<script>
(function() {{
  const base = {json.dumps(base)};
  const camsRow = document.getElementById('scrubCams');
  const slider = document.getElementById('slider');
  const label = document.getElementById('frameLabel');
  const status = document.getElementById('scrubStatus');
  const btnPlay = document.getElementById('btnPlay');
  const btnPlayLabel = document.getElementById('btnPlayLabel');
  const speedBtns = Array.from(document.querySelectorAll('.speed'));
  let cams = [];      // [{{id, frames:[url], commit, img, cell}}]
  let nFrames = 0, cur = 0, timer = null, speed = 1, fps = 30;

  function show(i) {{
    cur = Math.max(0, Math.min(nFrames - 1, i));
    slider.value = cur;
    for (const c of cams) {{
      const j = Math.min(cur, c.frames.length - 1);
      if (c.frames[j]) c.img.src = c.frames[j];       // compressed, for display
      // full-resolution lossless for the lightbox (fetched only on click)
      c.img.dataset.full = base + '/clip/' + c.id + '/' + j + '.png';
      c.cell.classList.toggle('at-commit', cur === c.commit);
    }}
    label.textContent = 'Frame ' + (cur + 1) + ' / ' + nFrames;
  }}
  function stop() {{
    if (timer) {{ clearInterval(timer); timer = null; }}
    btnPlay.classList.remove('is-playing');
    btnPlay.querySelector('.ico').textContent = '▶';
    btnPlayLabel.textContent = 'Play';
    btnPlay.setAttribute('aria-label', 'Play');
  }}
  function play() {{
    stop();
    // Play from the start if we're already at (or past) the end.
    if (cur >= nFrames - 1) show(0);
    btnPlay.classList.add('is-playing');
    btnPlay.querySelector('.ico').textContent = '❚❚';
    btnPlayLabel.textContent = 'Pause';
    btnPlay.setAttribute('aria-label', 'Pause');
    const dt = Math.max(20, 1000 / (fps * speed));
    timer = setInterval(() => {{
      if (cur >= nFrames - 1) {{ stop(); return; }}  // play once, no loop
      show(cur + 1);
    }}, dt);
  }}
  function setSpeed(s) {{
    speed = s;
    for (const b of speedBtns) b.classList.toggle('is-active',
      parseFloat(b.dataset.speed) === s);
    if (timer) play();  // restart timer at the new rate
  }}

  slider.addEventListener('input', () => {{ stop(); show(parseInt(slider.value, 10)); }});
  btnPlay.addEventListener('click', () => {{ if (timer) stop(); else play(); }});
  for (const b of speedBtns) {{
    b.addEventListener('click', () => setSpeed(parseFloat(b.dataset.speed)));
  }}
  window.addEventListener('keydown', (ev) => {{
    if (ev.target && /^(INPUT|TEXTAREA|SELECT)$/.test(ev.target.tagName)
        && ev.target !== slider) return;
    if (ev.key === 'ArrowLeft')  {{ stop(); show(cur - 1); ev.preventDefault(); }}
    else if (ev.key === 'ArrowRight') {{ stop(); show(cur + 1); ev.preventDefault(); }}
    else if (ev.key === ' ') {{ if (timer) stop(); else play(); ev.preventDefault(); }}
    else if (ev.key === 'Home') {{ stop(); show(0); ev.preventDefault(); }}
    else if (ev.key === 'End')  {{ stop(); show(nFrames - 1); ev.preventDefault(); }}
  }});

  fetch(base + '/clip.json').then(r => r.ok ? r.json() : Promise.reject(r.status))
    .then(data => {{
      fps = data.fps || 30;
      const ids = Object.keys(data.cameras).sort((a, b) => (+a) - (+b));
      for (const id of ids) {{
        const c = data.cameras[id];
        const cell = document.createElement('figure');
        cell.className = 'still';
        const img = document.createElement('img');
        img.className = 'zoomable';
        const cap = document.createElement('figcaption');
        cap.textContent = 'Camera ' + id;
        cell.appendChild(img); cell.appendChild(cap);
        camsRow.appendChild(cell);
        cams.push({{id, frames: c.frames, commit: c.commit_index, img, cell}});
        nFrames = Math.max(nFrames, c.frames.length);
      }}
      if (nFrames === 0) {{ status.textContent = 'No frames'; return; }}
      slider.max = nFrames - 1;
      slider.disabled = false;
      status.textContent = nFrames + ' frames · ' + fps + ' fps';
      show(cams[0] ? cams[0].commit : 0);  // open on the scored frame
    }})
    .catch(err => {{ status.textContent = 'Could not load clip (' + err + ')'; }});
}})();
</script>"""

        page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Throw {score_label}</title>
<style>
  :root{{
    --bg:#0f1115; --panel:#161922; --panel-2:#1c2029; --line:#272c38;
    --text:#e7e9ee; --muted:#8b93a2; --faint:#5f6675;
    --accent:#4ea1ff; --accent-dim:#2b4b6f;
    color-scheme:dark;
  }}
  *{{box-sizing:border-box}}
  body{{
    background:var(--bg); color:var(--text); margin:0;
    font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased;
  }}
  .wrap{{max-width:1160px; margin:0 auto; padding:24px 20px 48px}}
  header.top{{
    display:flex; align-items:baseline; flex-wrap:wrap; gap:10px 14px;
    padding-bottom:16px; margin-bottom:20px; border-bottom:1px solid var(--line);
  }}
  header.top h1{{
    font-size:22px; font-weight:650; letter-spacing:-0.01em; margin:0;
  }}
  header.top{{flex-direction:column; align-items:stretch; gap:8px}}
  .title-row{{display:flex; align-items:center; justify-content:space-between;
    flex-wrap:wrap; gap:10px 12px; min-width:0}}
  .details{{display:flex; flex-wrap:wrap; gap:4px 8px; color:var(--muted);
    font-size:13px; font-variant-numeric:tabular-nums}}
  .details .sep{{color:var(--faint)}}
  .visit{{display:flex; align-items:center; gap:6px; flex-wrap:wrap}}
  .visit-label{{font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.06em; color:var(--faint); margin-right:4px}}
  .dart{{min-width:56px; text-align:center; font-size:15px; font-weight:650;
    font-variant-numeric:tabular-nums; padding:5px 12px; border-radius:999px;
    border:1px solid var(--line); background:var(--panel-2); color:var(--muted);
    text-decoration:none}}
  a.dart:hover, a.dart:focus-visible{{border-color:var(--accent); color:var(--text); outline:none}}
  .dart-now{{color:#dfeaff; background:var(--accent-dim); border-color:transparent}}
  .dart-empty{{color:var(--faint); background:transparent; border-style:dashed}}
  .ids{{font-size:11px; color:var(--faint); font-variant-numeric:tabular-nums}}
  .chips{{display:flex; gap:8px; flex-wrap:wrap}}
  .chip{{
    font-size:12px; font-weight:550; color:var(--muted);
    background:var(--panel-2); border:1px solid var(--line);
    border-radius:999px; padding:3px 10px; white-space:nowrap;
  }}
  .chip-score{{color:#dfeaff; background:var(--accent-dim); border-color:transparent;
    font-variant-numeric:tabular-nums; letter-spacing:0.02em}}
  .chip-muted{{color:var(--faint); font-variant-numeric:tabular-nums}}
  .panel{{
    background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:16px; margin-bottom:20px;
  }}
  .panel-head{{display:flex; align-items:baseline; justify-content:space-between;
    gap:12px; margin-bottom:12px}}
  h2{{font-size:13px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.06em; color:var(--muted); margin:0}}
  .rowlabel{{
    font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:0.06em; color:var(--faint); margin:14px 0 6px;
  }}
  .rowlabel:first-child{{margin-top:0}}
  .grid{{
    display:grid; grid-template-columns:repeat(var(--n,3),1fr); gap:12px;
  }}
  figure.still{{margin:0}}
  figure.still img{{
    width:100%; display:block; aspect-ratio:16/9; object-fit:cover;
    background:#000; border:1px solid var(--line); border-radius:8px;
  }}
  figure.still figcaption{{
    font-size:11px; color:var(--faint); margin-top:5px; text-align:center;
    letter-spacing:0.02em;
  }}
  img.zoomable{{cursor:zoom-in}}
  .lightbox{{
    position:fixed; inset:0; z-index:100; display:none;
    align-items:center; justify-content:center; padding:24px;
    background:rgba(6,8,12,.86); backdrop-filter:blur(2px); cursor:zoom-out;
  }}
  .lightbox.open{{display:flex}}
  .lightbox img{{
    max-width:100%; max-height:100%; display:block; object-fit:contain;
    border:1px solid var(--line); border-radius:8px;
    box-shadow:0 12px 48px rgba(0,0,0,.6); background:#000;
  }}
  .lightbox .lb-cap{{
    position:fixed; left:0; right:0; bottom:16px; text-align:center;
    color:var(--muted); font-size:12px; pointer-events:none;
  }}
  .lightbox .lb-close{{
    position:fixed; top:14px; right:16px; width:36px; height:36px;
    display:flex; align-items:center; justify-content:center;
    background:var(--panel-2); color:var(--text); border:1px solid var(--line);
    border-radius:8px; font-size:18px; cursor:pointer;
  }}
  .lightbox .lb-close:hover{{background:#222735}}
  #scrubCams figure.still{{border-radius:10px; transition:box-shadow .1s}}
  #scrubCams figure.still img{{transition:border-color .1s}}
  #scrubCams figure.still.at-commit img{{
    border-color:var(--accent); box-shadow:0 0 0 2px var(--accent) inset;
  }}
  #scrubCams figure.still.at-commit figcaption{{color:var(--accent)}}
  .transport{{
    display:flex; align-items:center; gap:12px; flex-wrap:wrap;
    margin:16px 0 10px;
  }}
  .btn{{
    background:var(--panel-2); color:var(--text); border:1px solid var(--line);
    border-radius:8px; padding:7px 12px; font:inherit; font-weight:550;
    cursor:pointer; transition:background .1s,border-color .1s;
  }}
  .btn:hover{{background:#222735; border-color:#333a49}}
  .btn:focus-visible{{outline:2px solid var(--accent); outline-offset:2px}}
  .btn-primary{{display:inline-flex; align-items:center; gap:8px; min-width:96px;
    justify-content:center}}
  .btn-primary .ico{{font-size:11px; line-height:1}}
  .btn-primary.is-playing{{background:var(--accent-dim); border-color:transparent;
    color:#eaf2ff}}
  .speeds{{display:inline-flex; gap:2px; background:var(--panel-2);
    border:1px solid var(--line); border-radius:8px; padding:2px}}
  .speeds .btn{{border:none; background:transparent; padding:5px 10px;
    border-radius:6px; font-variant-numeric:tabular-nums}}
  .speeds .btn:hover{{background:#222735}}
  .speeds .btn.is-active{{background:var(--accent); color:#0c1420}}
  .frame-label{{margin-left:auto; color:var(--muted);
    font-variant-numeric:tabular-nums; font-size:13px}}
  #slider{{width:100%; accent-color:var(--accent); margin:4px 0 0; cursor:pointer}}
  #slider:disabled{{opacity:.5; cursor:default}}
  .hint{{color:var(--faint); font-size:12px; margin:12px 0 0}}
  .hint kbd{{
    background:var(--panel-2); border:1px solid var(--line);
    border-bottom-width:2px; border-radius:5px; padding:1px 6px;
    font:inherit; font-size:11px; color:var(--muted);
  }}
  .muted{{color:var(--muted); font-size:12px; font-variant-numeric:tabular-nums}}
  .empty{{color:var(--muted); margin:0; text-align:center; padding:12px}}
  @media (max-width:560px){{
    .grid{{gap:8px}} header.top h1{{font-size:19px}}
    .frame-label{{margin-left:0; width:100%}}
  }}
</style></head><body>
<div class="wrap">
  <header class="top">
    <div class="title-row">
      <h1>{html.escape(score_label)}</h1>
      {visit_html}
    </div>
    <div class="details">{details_html}</div>
    <div class="ids">{ids_text}</div>
  </header>

  <section class="panel">
    <div class="panel-head"><h2>Stills</h2></div>
    <div class="rowlabel">Reference (before)</div>
    <div class="grid" style="--n:{ncols}">{_cells("bg")}</div>
    <div class="rowlabel">Scored (after)</div>
    <div class="grid" style="--n:{ncols}">{_cells("after")}</div>
  </section>

  {scrubber}
  {no_clip_note}
</div>

<div class="lightbox" id="lightbox" role="dialog" aria-modal="true" aria-label="Full-resolution image">
  <button class="lb-close" id="lbClose" type="button" aria-label="Close">✕</button>
  <img id="lbImg" alt="">
  <div class="lb-cap" id="lbCap"></div>
</div>
<script>
(function() {{
  const lb = document.getElementById('lightbox');
  const lbImg = document.getElementById('lbImg');
  const lbCap = document.getElementById('lbCap');
  function open(src, cap) {{
    lbImg.src = src; lbImg.alt = cap || '';
    lbCap.textContent = cap || '';
    lb.classList.add('open');
  }}
  function close() {{ lb.classList.remove('open'); lbImg.removeAttribute('src'); }}
  // Delegated: any zoomable image opens full-resolution in the lightbox.
  // The grid/scrubber show a compressed image; data-full points at the
  // full-resolution lossless version, fetched only on this click.
  document.addEventListener('click', (ev) => {{
    const img = ev.target.closest && ev.target.closest('img.zoomable');
    if (!img || !img.src) return;
    const fig = img.closest('figure');
    const cap = fig && fig.querySelector('figcaption');
    open(img.dataset.full || img.currentSrc || img.src, cap ? cap.textContent : '');
  }});
  lb.addEventListener('click', close);           // click the backdrop or image
  document.getElementById('lbClose').addEventListener('click', close);
  window.addEventListener('keydown', (ev) => {{
    if (ev.key === 'Escape' && lb.classList.contains('open')) {{ close(); ev.stopPropagation(); }}
  }}, true);
}})();
</script>
<script>
// The package stores UTC; show it in the viewer's own time zone. The raw
// ISO text stays as the fallback if this never runs.
(() => {{
  const el = document.getElementById('capturedAt');
  const iso = el && el.getAttribute('datetime');
  if (!iso) return;
  const d = new Date(iso);
  if (isNaN(d)) return;
  el.textContent = d.toLocaleString(undefined, {{
    weekday: 'short', month: 'short', day: 'numeric', year: 'numeric',
    hour: 'numeric', minute: '2-digit', second: '2-digit',
  }});
  el.title = iso;
}})();
</script>
</body></html>"""
        return HTMLResponse(page)

    @app.post("/api/packages/{session}/{throw_id}/capture-misscore")
    async def api_capture_misscore(
        session: str, throw_id: str, body: dict[str, Any]
    ) -> JSONResponse:
        """The MISSCORE button on the engine row: write ~1s around THIS
        throw's own recorded time.

        The throw is identified exactly as the per-throw correction
        control already identifies it -- session + throw_id, the package
        directory name -- because the dashboard already knows which throw
        the operator is looking at, and inventing a second identity for
        the same click is how the two would eventually disagree.

        Path components are validated as single path segments before any
        filesystem access, same discipline as
        POST /api/packages/{session}/{throw_id}/mark-ad-wrong above.
        """
        if not session or "/" in session or "\\" in session or ".." in session:
            return JSONResponse(
                {"ok": False, "reason": f"invalid session: {session!r}"}, status_code=400
            )
        if not throw_id or "/" in throw_id or "\\" in throw_id or ".." in throw_id:
            return JSONResponse(
                {"ok": False, "reason": f"invalid throw_id: {throw_id!r}"}, status_code=400
            )
        service = state.throw_capture
        if service is None:
            return JSONResponse(
                {"ok": False, "reason": _frame_ring_payload().get("reason")},
                status_code=200,
            )
        pkg_dir = state.package_root / session / throw_id
        if not pkg_dir.is_dir():
            return JSONResponse(
                {"ok": False, "reason": f"no such package: {session}/{throw_id}"},
                status_code=404,
            )
        reason = str(body.get("reason") or "").strip() or f"misscore on {throw_id}"
        result = await asyncio.to_thread(
            service.capture_misscore, None,
            reason=reason, source="manual", package_dir=pkg_dir,
        )
        if state.controller is not None:
            state.controller.touch()
        return JSONResponse(
            {**result, **_frame_ring_payload(), "ok": result.get("ok", False)},
            status_code=200,
        )

    # -- spoken dart calls -------------------------------------------------
    #
    # THREE ROUTES, AND NONE OF THEM IS A SETTING. `GET`/`POST
    # /api/audio` (on/off + voice, persisted to config.json) and `POST
    # /api/audio/test` (play one clip on the rig) were both deleted on
    # 2026-09-15 when playback moved into the browser. On/off, volume and
    # voice are per-device localStorage values now, and Test plays in the
    # browser that pressed it, so neither needed a server round-trip at
    # all. What is left is: what sets exist, the bytes of one clip, and a
    # place for each device to say whether it can actually be heard.

    @app.get("/api/audio/voices")
    async def api_audio_voices() -> dict[str, Any]:
        """Everything a browser needs to arm itself, in one request.

        The installed sets with per-set coverage, the default, and the
        whole phrase -> filename map. Deliberately one call rather than
        four: this is fetched when a device first turns sound on and
        again when it changes voice, and a settings panel that fires a
        request per voice to fill a dropdown is a panel that gets slower
        every time someone adds a voice.

        `clips` is the map, not a rule. The browser receives phrases on
        the wire ("treble 20") and needs URLs, and handing it the
        finished mapping is what keeps `clip_name()`'s slugification from
        being reimplemented in JavaScript -- see audio.clip_names().

        COVERAGE PER SET, not for one "current" voice, because there is
        no current voice on this side any more. "61/64 clips" is the
        diagnostic worth surfacing: a set that is three clips short is
        silent on exactly the darts nobody throws while testing.
        """
        return {
            "ok": True,
            "voices": [audio.coverage(name) for name in audio.voices()],
            "default": audio.DEFAULT_VOICE,
            "clips": audio.clip_names(),
        }

    @app.get("/api/audio/clips/{voice}/{name}")
    async def api_audio_clip(voice: str, name: str) -> Response:
        """The bytes of one clip. The only genuinely new job this server
        picked up in the move: being a static file host for ~600 KB of
        MP3.

        TWO WHITELISTS, NOT A SANITISER. `voice` is resolved against the
        directories `voices()` just enumerated (audio.voice_dir()), and
        `name` must appear in `clip_names()` -- a closed, generated set
        of 64 literal filenames. Neither value is ever joined onto a path
        as given, so `..%2f..%2fetc%2fpasswd` is not an attack that has
        to be detected and stripped; it is simply a name nothing matches.
        Path traversal is the classic bug in a route shaped exactly like
        this one, and the way not to have it is to never construct a path
        from user input in the first place.

        Read into memory rather than streamed: the largest clip in either
        shipped set is about 12 KB.
        """
        d = audio.voice_dir(voice)
        if d is None or name not in set(audio.clip_names().values()):
            return JSONResponse(
                {"ok": False, "reason": f"no clip {name!r} in voice {voice!r}"},
                status_code=404,
            )
        p = d / name
        try:
            data = p.read_bytes()
        except OSError as exc:
            # An enumerated set with an unreadable file in it: a real
            # thing (a bad rsync, a permissions mistake), and one the
            # coverage count cannot see because the file EXISTS.
            log.warning("clip %s could not be read: %s", p, exc)
            return JSONResponse({"ok": False, "reason": str(exc)}, status_code=404)
        return Response(
            content=data,
            media_type="audio/mpeg",
            # Immutable for a day. Clip bytes for a given voice+phrase
            # never change without the file being replaced by a
            # maintainer and the rig redeployed, and the browser holds
            # decoded buffers in memory for the session anyway -- this is
            # only about the refetch after a page reload, which is
            # exactly the case a TV hits at 3am after a browser update.
            headers={"Cache-Control": "public, max-age=86400, immutable"},
        )

    @app.post("/api/audio/clients")
    async def api_audio_clients_post(payload: dict[str, Any]) -> dict[str, Any]:
        """One device reporting whether it can actually be heard.

        See AppState.__init__'s `audio_clients` comment for why this
        exists at all. The short version: autoplay policy is the one
        failure the old server-side playback did not have, and a screen
        nobody can reach cannot report its own silence to the person
        standing at the board unless it reports it to something.

        Stored as sent, with a server timestamp. Not validated beyond
        `client_id`, and never interpreted -- this is a mirror and not a
        control surface. A missing id is a 400 rather than a silent no-op: a
        client that has stopped identifying itself is a client-side bug,
        and swallowing it would hide exactly what the endpoint is for.
        """
        client_id = payload.get("client_id")
        if not isinstance(client_id, str) or not client_id:
            return JSONResponse(
                {"ok": False, "reason": "payload.client_id (string) is required"},
                status_code=400,
            )
        state.audio_clients[client_id] = {
            **payload,
            "reported_at_utc": datetime.now(timezone.utc).isoformat(),
            "reported_at_monotonic": time.monotonic(),
        }
        # Move a returning id to the end so the cap evicts genuinely
        # oldest-reporting tabs -- a plain assignment does not reorder an
        # existing key. Same dance, same reason, as the debug snapshots.
        state.audio_clients[client_id] = state.audio_clients.pop(client_id)
        while len(state.audio_clients) > AUDIO_CLIENT_MAX:
            del state.audio_clients[next(iter(state.audio_clients))]
        return {"ok": True}

    @app.get("/api/audio/clients")
    async def api_audio_clients_get() -> dict[str, Any]:
        """Every device that has spoken up recently, with how long ago.

        Expired entries are dropped HERE rather than on a timer: there is
        no background task to justify for a dict that only matters when
        someone looks at it, and reading is the only moment staleness can
        mislead anyone.

        `age_s` is reported rather than a bare boolean, because "the TV
        last checked in 38 seconds ago" and "the TV is fine" are
        different claims and the panel should be able to make the honest
        one.
        """
        now = time.monotonic()
        fresh = {
            cid: rec for cid, rec in state.audio_clients.items()
            if now - float(rec.get("reported_at_monotonic") or 0.0)
            <= AUDIO_CLIENT_STALE_S
        }
        if len(fresh) != len(state.audio_clients):
            state.audio_clients = fresh
        return {
            "ok": True,
            "stale_after_s": AUDIO_CLIENT_STALE_S,
            "clients": [
                {**rec, "client_id": cid,
                 "age_s": round(now - float(rec.get("reported_at_monotonic") or 0.0), 1)}
                for cid, rec in fresh.items()
            ],
        }

    @app.post("/api/packages/{session}/{throw_id}/mark-ad-wrong")
    async def api_mark_ad_wrong(
        session: str, throw_id: str, payload: MarkAdWrongRequest
    ) -> JSONResponse:
        """Operator-triggered "AD was wrong on this throw" toggle -- a
        human judgment call opendarts cannot make algorithmically. The Scoring tab's per-row
        button calls this.

        `session`/`throw_id` come straight off the URL path and are used
        to build a real filesystem path under package_root -- validated
        as a single path component (no "/", no "\\", no "..") first, same
        defensive discipline as GET /api/logs/{name}'s VALID_LOG_NAMES
        allowlist above, just shaped for an arbitrary-but-bounded path
        segment instead of a fixed name set.
        """
        if not session or "/" in session or "\\" in session or ".." in session:
            return JSONResponse({"ok": False, "reason": f"invalid session: {session!r}"}, status_code=400)
        if not throw_id or "/" in throw_id or "\\" in throw_id or ".." in throw_id:
            return JSONResponse({"ok": False, "reason": f"invalid throw_id: {throw_id!r}"}, status_code=400)
        result = await state.mark_ad_wrong(
            session,
            throw_id,
            payload.wrong,
            payload.note,
            payload.confirmed_source,
            payload.confirmed_sector,
            payload.confirmed_ring,
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 404)

    @app.post("/api/visits/{visit_id}/throws/{index}/correct")
    async def api_correct_throw(
        visit_id: str, index: int, payload: CorrectThrowRequest
    ) -> JSONResponse:
        """Correct one throw of a visit -- "that dart actually landed in
        <sector, ring>". The live-game-driver counterpart to the Scoring
        tab's own per-package mark-ad-wrong route above; both write the
        SAME annotation through the same mechanism (see
        AppState.correct_throw() and
        opendarts.capture.throw_package.record_throw_correction()).

        `ring` is validated against `board_ring_names()` -- the real
        vocabulary DERIVED from opendarts.geometry.board itself, not a
        hardcoded list here -- and `sector` against the real wedge
        numbers, because a correction that doesn't use the exact strings
        opendarts's own scorer produces could never compare equal to any
        engine's answer, and would silently record a truth nothing can
        ever match.
        """
        valid_rings = board_ring_names()
        if payload.ring not in valid_rings:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": f"invalid ring {payload.ring!r} -- must be one of {valid_rings}",
                },
                status_code=400,
            )
        valid_sectors = {str(n) for n in SECTOR_NUMBERS_CLOCKWISE}
        if payload.sector is not None and payload.sector not in valid_sectors:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": (
                        f"invalid sector {payload.sector!r} -- must be a wedge number "
                        f"string ({sorted(valid_sectors, key=int)}) or omitted for "
                        "bull/outer_bull/outside"
                    ),
                },
                status_code=400,
            )
        # sector/ring must be a combination sector_ring_for_point() could
        # actually produce -- see board_sectorless_rings(). A "treble with
        # no sector" or a "bull in sector 20" would be recorded as a truth
        # no engine's answer can ever equal, i.e. a permanent silent miss.
        sectorless = board_sectorless_rings()
        if payload.ring in sectorless and payload.sector is not None:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": (
                        f"ring {payload.ring!r} takes no sector -- omit `sector` "
                        f"(sectorless rings: {sorted(sectorless)})"
                    ),
                },
                status_code=400,
            )
        if payload.ring not in sectorless and payload.sector is None:
            return JSONResponse(
                {
                    "ok": False,
                    "reason": f"ring {payload.ring!r} requires a `sector` (wedge number string)",
                },
                status_code=400,
            )
        result = await state.correct_throw(
            visit_id, index, payload.sector, payload.ring, payload.source, payload.note
        )
        return JSONResponse(result, status_code=200 if result.get("ok") else 404)

    @app.get("/api/live/recent")
    async def api_live_recent(limit: int = RETAIL_RECENT_VISITS_MAX) -> dict[str, Any]:
        """Retail catch-up: recently COMPLETED visits, oldest first.

        The /api/live socket carries no history by design -- its `hello`
        is a snapshot of now -- so a client reconnecting mid-match knew
        the current visit and nothing before it. This fills that hole
        without putting history on the socket.

        Served from an IN-MEMORY ring appended as each visit closes, NOT
        from saved packages: deleting the corpus, pulling it off the rig,
        or turning package storage off entirely must never erase a
        client's match history. It does not survive a server restart --
        see RETAIL_RECENT_VISITS_MAX for why that is the right way round.

        The in-progress visit is absent by construction: a visit enters
        this ring only when it closes, and the socket already owns the
        open one.
        """
        limit = max(1, min(int(limit), RETAIL_RECENT_VISITS_MAX))
        visits = list(state._recent_visits)[-limit:] # noqa: SLF001 -- same module
        return {
            "visits": visits,
            "count": len(visits),
            "limit": limit,
            "buffer_max": RETAIL_RECENT_VISITS_MAX,
        }

    @app.get("/api/cameras/{cam_id}/snapshot.png")
    async def camera_snapshot(cam_id: int):
        """Per-camera still -- fetched fresh on every request, not
        cached. No longer the dashboard's raw-preview path (that moved
        to stream.mjpg below, 2026-09-11); kept as the full-resolution,
        lossless single-frame endpoint for anything that wants an
        inspectable still rather than a preview. Default source: state.hub's direct
        cv2.VideoCapture grab (opendarts.live.local_capture.fetch_snapshot).
        Returns 502 with
        a JSON error body (not a broken image) if the frame source is
        unreachable/unopened -- graceful degrade per this server's own
        design brief.
        """
        try:
            if state.hub is None:
                raise RuntimeError(
                    "local camera hub not initialized -- server lifespan "
                    "startup may not have run yet"
                )
            snap = await asyncio.to_thread(
                local_capture.fetch_snapshot,
                cam_id,
                state.scratch_dir / "snapshot",
                state.hub,
            )
        except Exception as exc: # noqa: BLE001 -- any fetch failure -> honest 502
            return JSONResponse(
                {"ok": False, "cam": cam_id, "reason": str(exc)},
                status_code=502,
            )
        data = snap.path.read_bytes()
        return Response(
            content=data,
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/cameras/{cam_id}/overlay.png")
    async def camera_overlay(cam_id: int, highlight: int = DEFAULT_HIGHLIGHT_NUMBER):
        """Calibration confirmation overlay -- a fresh snapshot with
        `opendarts.geometry.board_overlay.draw_calibration_overlay()` drawn
        on top, so an operator can SEE whether calibration got the board
        orientation right (visual companion to the numeric reprojection-
        error metric already shown per camera). See `board_overlay.py`'s
        own module docstring for how the overlay is drawn.

        **Deliberately uncached, on every single request** -- renders
        against whatever `opendarts.live.capture_daemon.CalibrationStore`
        holds RIGHT NOW (`state.calibration_store.get()`), the exact
        object `/api/calibration/refresh` (and the auto-calibrate-on-
        Start path) writes into. This is the actual mechanism satisfying
        the project's "make sure to clear it when re-calibrating" requirement:
        there is no cached overlay image to go stale in the first place,
        so a new calibration is reflected on the very next request with
        no separate invalidation step needed -- same reasoning
        `snapshot.png` above already uses for the raw frame itself.

        Degrades gracefully (same honest-502 pattern as snapshot.png)
        when the frame source itself fails. When the frame source is
        fine but no calibration is available yet for this camera (no
        `calibration_store` wired, or this cam id isn't in it, or its
        calibration is unusable), returns the PLAIN snapshot unchanged
        rather than erroring -- an overlay has nothing honest to draw
        without a calibration, but the camera feed itself is still real
        and worth showing (an overlay renderer falls back to the raw
        frame when there is no usable calibration).
        """
        try:
            if state.hub is None:
                raise RuntimeError(
                    "local camera hub not initialized -- server lifespan "
                    "startup may not have run yet"
                )
            snap = await asyncio.to_thread(
                local_capture.fetch_snapshot,
                cam_id,
                state.scratch_dir / "snapshot",
                state.hub,
            )
        except Exception as exc: # noqa: BLE001 -- any fetch failure -> honest 502
            return JSONResponse(
                {"ok": False, "cam": cam_id, "reason": str(exc)},
                status_code=502,
            )

        calib = None
        if state.calibration_store is not None:
            calib = state.calibration_store.get().get(cam_id)

        if calib is None:
            data = snap.path.read_bytes()
        else:
            import cv2

            frame = cv2.imread(str(snap.path))
            if frame is None:
                data = snap.path.read_bytes()
            else:
                overlay = draw_calibration_overlay(frame, calib, highlight_number=highlight)
                ok, buf = cv2.imencode(".png", overlay)
                data = buf.tobytes() if ok else snap.path.read_bytes()

        return Response(
            content=data,
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/cameras/{cam_id}/overlay-rgba.png")
    async def camera_overlay_rgba(cam_id: int, highlight: int = DEFAULT_HIGHLIGHT_NUMBER):
        """The calibration overlay as a TRANSPARENT LAYER, sized to match
        this camera's MJPEG preview exactly, for compositing on top of it.

        Why this exists next to overlay.png above, which draws the same
        geometry: overlay.png returns a photograph with the overlay baked
        into it. That makes it a REPLACEMENT for the live view, and the
        dashboard used to treat it as one -- the moment a camera's
        calibration came back ok, its tile was dropped off the MJPEG
        stream onto a 3-second still. A calibrated rig got a slideshow
        and an uncalibrated one got video, which is exactly backwards.
        With a transparent layer the tile keeps streaming and the overlay
        sits over it.

        GRABS NO FRAME. It needs only the calibration and the pixel
        dimensions to project into, so it never touches the hub's pump
        and never competes with the stream for frames. Dimensions come
        from the negotiated CameraStatus and through the SAME
        _preview_dimensions() the stream encoder uses -- see that
        function for why one pixel of disagreement matters here.

        And it is fetched once per RECALIBRATION, not on a timer: the
        board does not move between frames, so there was never anything
        for a poll to discover. 404 when this camera has no calibration
        -- there is nothing honest to draw, and the dashboard simply
        shows the stream with no overlay rather than a blank layer.
        """
        calib = None
        if state.calibration_store is not None:
            calib = state.calibration_store.get().get(cam_id)
        if calib is None:
            return JSONResponse(
                {"ok": False, "cam": cam_id, "reason": "no calibration for this camera"},
                status_code=404,
            )
        if state.hub is None:
            return JSONResponse(
                {"ok": False, "cam": cam_id, "reason": "local camera hub not initialized"},
                status_code=503,
            )
        status = state.hub.status.get(cam_id)
        if status is None or not status.actual_width or not status.actual_height:
            # No negotiated size means the camera has never opened, so
            # there is no stream to sit on top of either. Refusing beats
            # guessing a size the overlay would then not line up with.
            return JSONResponse(
                {"ok": False, "cam": cam_id,
                 "reason": "camera has no negotiated frame size yet -- not started?"},
                status_code=503,
            )

        import cv2

        # NATIVE frame size, deliberately NOT the preview size. The
        # calibration's camera_matrix is defined in the camera's own full
        # pixel space, and draw_calibration_overlay_rgba() projects through
        # it without rescaling -- so rendering onto a smaller canvas keeps
        # full-resolution coordinates on it and puts the board 1.33x too
        # large and offset toward the bottom-right. That shipped once; see
        # test_overlay_rgba_is_rendered_in_the_calibrations_own_pixel_space.
        # The browser scales this layer exactly as it scales the stream, and
        # the two share an aspect ratio, so they stay registered.
        size = (status.actual_width, status.actual_height)
        # to_thread for the same reason the stream encoder uses it: the
        # projection plus a PNG encode is real CPU and the event loop is
        # serving live streams alongside this.
        rgba = await asyncio.to_thread(
            draw_calibration_overlay_rgba, size, calib, highlight_number=highlight
        )
        ok, buf = await asyncio.to_thread(cv2.imencode, ".png", rgba)
        if not ok:
            return JSONResponse(
                {"ok": False, "cam": cam_id, "reason": "overlay PNG encode failed"},
                status_code=500,
            )
        return Response(
            content=buf.tobytes(),
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/api/cameras/{cam_id}/stream.mjpg")
    async def camera_stream(cam_id: int, full: int = 0):
        # `full=1` serves the frame untouched, at whatever rate the pump
        # produces -- the TRANSPORT case, where the consumer is scoring
        # these frames rather than looking at them, and a downscaled
        # 12fps preview would be the wrong input. Default 0 keeps the
        # dashboard's behaviour byte for byte: a browser tab wants a
        # small cheap picture, not 41 Mbps.
        """Live MJPEG preview -- multipart/x-mixed-replace, so a plain
        <img src=...> renders it as video with no JavaScript. Replaces the dashboard's old
        3-second snapshot.png polling for the raw-frame preview; the
        snapshot/overlay PNG endpoints above remain for stills and the
        calibration overlay.

        CONTRACT. 503 (JSON body, not a stream) unless the capture loop
        is actually running -- a preview must not hold a connection open
        pretending, and the honest refusal is what lets the dashboard
        keep its "camera not started" placeholder logic. 404 for a camera
        id outside this rig's configured range. Otherwise: JPEG parts at
        most MJPEG_MAX_FPS apart, each emitted only when the pump has
        actually produced a NEW frame for this camera (CameraStatus.
        frame_count advanced -- the generation counter is not used here
        because it bumps even on failed reads). The stream ends, closing
        the connection, when the capture loop stops or the camera goes
        MJPEG_STALL_TIMEOUT_S without a new frame.

        CPU DISCIPLINE (the reason this looks the way it does): frames
        come exclusively from the hub's pump cache via grab() -- never a
        second cv2 read, which is exactly the contention the pump exists
        to prevent. The only real work per frame (resize + JPEG encode,
        _encode_preview_jpeg) happens in asyncio.to_thread, so the event
        loop stays responsive and the pump thread is never touched at
        all. Between frames the generator just sleeps on the event loop
        -- deliberately NOT hub.wait_for_new_frame(), which would park a
        threadpool thread per open stream; at a 12fps cap the extra
        latency of a sleep tick is invisible in a preview, and coroutines
        are free where blocked threads are not. A slow client applies
        backpressure only to its own generator (the yield awaits the
        send), and since every iteration re-reads the LATEST cached
        frame, a stalled client simply misses frames instead of building
        a queue. A dead client cancels its generator at the next
        yield/sleep; the finally releases its preview slot."""
        running = (
            state.hub is not None
            and state.controller is not None
            and state.controller.is_running()
        )
        if not running:
            return JSONResponse(
                {"ok": False, "cam": cam_id, "reason": "capture not running"},
                status_code=503,
            )
        if cam_id < 0 or cam_id >= len(state.hub.configs):
            return JSONResponse(
                {"ok": False, "cam": cam_id, "reason": "no such camera"},
                status_code=404,
            )
        n_cams = len(state.hub.configs)
        if full:
            in_use, cap, kind = (
                state.mjpeg_transport_count,
                _mjpeg_cap(n_cams, MJPEG_MAX_TRANSPORT_CONSUMERS),
                "transport",
            )
        else:
            in_use, cap, kind = (
                state.mjpeg_client_count,
                _mjpeg_cap(n_cams, MJPEG_MAX_PREVIEW_VIEWERS),
                "preview",
            )
        if in_use >= cap:
            # LOGGED, not just returned. This refusal previously existed
            # only as a response body, so a rig at its ceiling showed blank
            # tiles with a completely clean log -- the operator's evidence
            # said "no errors" while every new stream was being turned
            # away. A refusal is worth a line.
            log.warning(
                "cam%d: refusing a %s stream -- %d/%d already open. %s",
                cam_id, kind, in_use, cap,
                "Another machine consuming this rig's cameras holds one "
                "connection per camera, as does each open dashboard.",
            )
            return JSONResponse(
                {
                    "ok": False,
                    "cam": cam_id,
                    "kind": kind,
                    "open": in_use,
                    "max": cap,
                    "reason": f"too many {kind} streams open ({in_use}/{cap})",
                },
                status_code=503,
            )

        async def frame_parts():
            # The slot is claimed HERE, on first iteration, not in the
            # endpoint body above: if the client disconnects between the
            # endpoint returning and Starlette iterating the generator,
            # a generator that never started never runs its finally, and
            # a slot claimed earlier would leak forever. The cost is a
            # tiny admission race past the cap check above (both run on
            # the event loop, so overshoot is bounded by in-flight
            # requests, not unbounded) -- an acceptable trade against a
            # permanent slot leak.
            if full:
                state.mjpeg_transport_count += 1
            else:
                state.mjpeg_client_count += 1
            try:
                # Transport mode is not rate-capped -- the pump's cadence is
                # the rate, so a consumer sees every frame the publisher saw
                # rather than a sampled subset. But the POLL still has to
                # wait: see MJPEG_TRANSPORT_POLL_S for what sleeping 0 did.
                interval = MJPEG_TRANSPORT_POLL_S if full else 1.0 / MJPEG_MAX_FPS
                last_count = -1
                last_new_frame = time.monotonic()
                while True:
                    if state.is_shutting_down():
                        # END THE STREAM RATHER THAN BE CANCELLED. A
                        # multipart/x-mixed-replace response never
                        # finishes on its own, so uvicorn's graceful drain
                        # waits the full GRACEFUL_SHUTDOWN_TIMEOUT_S on
                        # every open stream and then force-cancels the
                        # task, which surfaces as "Cancel N running
                        # task(s), timeout graceful shutdown exceeded"
                        # followed by an ASGI traceback on Ctrl-C.
                        # Returning here closes the connection cleanly and
                        # lets the drain finish immediately.
                        #
                        # Checked FIRST: during shutdown the capture
                        # controller may still report running, so the
                        # condition below would keep the stream alive.
                        return
                    if (
                        state.hub is None
                        or state.controller is None
                        or not state.controller.is_running()
                    ):
                        # Capture stopped mid-stream: end honestly so the
                        # browser's connection dies instead of freezing
                        # on a stale frame that still claims to be live.
                        return
                    status = state.hub.status.get(cam_id)
                    count = status.frame_count if status is not None else 0
                    if count != last_count:
                        frame, camera_jpg = _grab_with_jpeg(state.hub, cam_id)
                        if frame is not None:
                            # A full-frame consumer gets the camera's own
                            # JPEG when the slot kept it (local_capture's
                            # JPEG PASSTHROUGH): no encode, and the bytes
                            # this rig scored rather than a re-encode of
                            # them. Otherwise one encode shared across
                            # every consumer of this frame -- see
                            # _SharedJpegCache.
                            if full and camera_jpg is not None:
                                jpg = camera_jpg
                                state.jpeg_cache.passthrough += 1
                            else:
                                jpg = await state.jpeg_cache.encoded(
                                    cam_id, full, count, frame)
                            if jpg is not None:
                                yield (
                                    b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
                                    b"Content-Type: image/jpeg\r\n"
                                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                                    + jpg
                                    + b"\r\n"
                                )
                        last_count = count
                        last_new_frame = time.monotonic()
                    elif time.monotonic() - last_new_frame > MJPEG_STALL_TIMEOUT_S:
                        return
                    if full:
                        await asyncio.sleep(interval)
                    else:
                        # ON A SHARED CLOCK, 2026-09-17. Each preview used to
                        # sleep a full interval after its own work, so two
                        # tabs sampled the camera at different moments,
                        # asked for different frames, and never shared an
                        # encode (_SharedJpegCache matches the exact frame).
                        # Waking every preview on the same beat makes them
                        # ask for the same frame, so N tabs cost one encode
                        # per camera instead of N.
                        await asyncio.sleep(interval - (time.monotonic() % interval))
            finally:
                if full:
                    state.mjpeg_transport_count -= 1
                else:
                    state.mjpeg_client_count -= 1

        return StreamingResponse(
            frame_parts(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.websocket("/api/live")
    async def live_ws(ws: WebSocket) -> None:
        """The RETAIL feed -- a deliberately smaller channel than
        `/api/events` for a client that only needs "what is the game
        state, and what just happened" (a scoreboard or overlay), not
        the full operational stream.

        Its own subscriber set: an `/api/events` client connecting or
        disconnecting has no effect here, or vice versa. Carries NO
        history -- a client that reconnects mid-session sees only what is
        true right now, never a replay. Completed visits are available
        separately via `GET /api/live/recent`, so history is a deliberate
        fetch rather than something every subscriber pays for.

        Two message types only: `state` (a complete snapshot every time,
        never a delta) and `throw`."""
        await ws.accept()
        queue = state.subscribe_retail()

        async def _drain_queue() -> None:
            while True:
                event = await queue.get()
                await ws.send_json(event)

        async def _watch_for_disconnect() -> None:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(code=message.get("code", 1000))

        try:
            await ws.send_json({
                "type": "state",
                "event": "hello",
                **state._build_retail_state_fields(),
                "at": datetime.now(timezone.utc).isoformat(),
            })
            drain_task = asyncio.ensure_future(_drain_queue())
            watch_task = asyncio.ensure_future(_watch_for_disconnect())
            done, pending = await asyncio.wait(
                {drain_task, watch_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc is not None and not isinstance(exc, WebSocketDisconnect):
                    raise exc
        except WebSocketDisconnect:
            pass
        finally:
            state.unsubscribe_retail(queue)

    @app.websocket("/api/events")
    async def events_ws(ws: WebSocket) -> None:
        """Broadcasts PACKAGES_UPDATED (see AppState._package_poll_loop,
        a POLLING FALLBACK wired to push over the socket, not a real
        event-driven source when no daemon process exists to drive real
        events from -- see this module's docstring) and CALIBRATION_STATUS
        (broadcast only on an explicit manual refresh now, no poll loop
        at all -- see AppState.refresh_calibration).

        `count` on HELLO, added 2026-08-24 -- real incident (flagged by
        the corpus-pull/QA process, caught because the pulled corpus
        disagreed with what the dashboard had shown): one recorded
        session captured 120 real throws, but the dashboard
        displayed 102 and computed every Scoring-tab percentage over
        that undercounted subset, with nothing on the page indicating
        it was incomplete. Root cause: both this HELLO payload and every
        PACKAGES_UPDATED broadcast cap `packages` to the most recent 20
        (harmless at normal pace, since the list is sorted newest-first)
        -- but a WebSocket that drops for longer than it takes >20
        throws to land (confirmed: a real 3m16s socket outage, 38 darts
        landed inside that window) loses everything older than the
        newest 20 on reconnect, silently -- there was no signal in the
        wire protocol a client could even check against. PACKAGES_UPDATED
        already sent a true, uncapped `count` (see its own broadcast
        call sites) that nothing read; HELLO had no `count` field at all
        until now. The client-side self-heal (compare `count` against
        what it actually has, refetch the uncapped `/api/packages` on a
        mismatch) lives in `ingestPackages()`'s own call sites below.
        """
        await ws.accept()
        state.clients.add(ws)
        try:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "HELLO",
                        "ts": datetime.now(timezone.utc).isoformat(),
                        # Lets a screen still running an older page notice
                        # the server moved on and reload itself.
                        "page_version": _DASHBOARD_PAGE_VERSION,
                        "state": state.state_dict(),
                        "count": len(state.list_packages()),
                        "packages": state.list_packages()[:20],
                    }
                )
            )
            while True:
                # This server doesn't act on client->server messages; the
                # receive is only here so a disconnect raises promptly
                # instead of leaking the connection out of `clients`.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            state.clients.discard(ws)

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="opendarts live dashboard/API server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default=DEFAULT_HOST)
    parser.add_argument(
        "--package-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help="Directory saved throw packages live under (default: %(default)s)",
    )
    parser.add_argument(
        "--ad-base-url",
        type=str,
        default=DEFAULT_AD_BASE,
        help=(
            "Autodarts base URL used for AD ground-truth fetches by "
            "opendarts/live/ad_ws_listener.py's automatic real-time listener -- "
            "see opendarts/live/ad_ground_truth.py (default: %(default)s)"
        ),
    )
    args = parser.parse_args(argv)

    from opendarts.live.logging_setup import configure_console_and_file_logging

    log_path = configure_console_and_file_logging("server", level=logging.INFO)
    log.info("logging to console AND to %s", log_path)

    # Same bound as run_product's own main() -- applied here too because
    # this standalone entrypoint serves frames (snapshots, overlays, MJPEG
    # previews) and so does real OpenCV work of its own. See
    # opendarts.live.cv2_threads for the measurement.
    from opendarts.live.config import load_live_config as _load_live_config
    from opendarts.live.cv2_threads import apply_cv2_thread_limit

    apply_cv2_thread_limit(_load_live_config().cv2_num_threads)

    try:
        import uvicorn
    except ImportError:
        log.error(
            "uvicorn is not installed in this environment's Python -- "
            "run `pip install -r requirements.txt` (fastapi + uvicorn "
            "were added to requirements.txt for this server) before "
            "starting opendarts.live.server."
        )
        return 1

    app = create_app(
        package_root=args.package_root,
        host=args.host,
        port=args.port,
        ad_base_url=args.ad_base_url,
    )

    display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    # per capture_daemon.py's own "don't make me guess if it's
    # running" precedent -- one unmissable line, printed before uvicorn
    # takes over stdout with its own (much noisier) access logging.
    print("=" * 60)
    print(f"Listening on http://{display_host}:{args.port} -- open this in a browser")
    print(f"(bound to {args.host}:{args.port}; package root: {args.package_root})")
    print(
        "(frame source: "
        + "local direct camera"
        + ")"
    )
    print("=" * 60)
    sys.stdout.flush()

    # access_log=False: see opendarts/live/run_product.py's identical fix
    # for why -- the dashboard's own snapshot auto-refresh spams uvicorn's
    # per-request access log otherwise.
    # log_config=None: see opendarts/live/run_product.py's identical fix and
    # comment for why -- without it, uvicorn's own colorized stderr-only
    # "uvicorn"/"uvicorn.error" handler (WebSocket accept/close chatter
    # included) bypasses opendarts.live.logging_setup's root-logger config
    # entirely, producing inconsistent console formatting and never
    # reaching the log file.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
