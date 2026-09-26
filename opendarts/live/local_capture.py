"""opendarts/live/local_capture.py -- direct, local camera access via
cv2.VideoCapture. This is the rig's frame source: cameras are opened once
at startup and held open for the process's lifetime. See
docs/DEPLOYMENT.md.

HONESTY, READ BEFORE USING THIS MODULE: this code was written in a dev
session with no access to the rig's physical USB cameras -- it cannot be
tested end-to-end from this environment, full stop. It is written
against a PROVEN capture pattern and its Python control flow is unit-tested with a
mocked cv2.VideoCapture (tests/test_local_capture.py), but that only
proves the LOGIC is sound -- it does NOT prove the real cameras open,
negotiate the requested resolution, or deliver real frames on the real
rig. Real validation requires running this ON THE RIG: either
scripts/test_local_camera_access.py (run directly on the rig, see that
script's own header), or through opendarts/live/capture_daemon.py.

Device contention (RESOLVED, was previously flagged as an untested
risk): an earlier draft treated a second process opening the same 3 USB
camera devices concurrently as an untested risk. Real operating history
has since confirmed that several processes have held
these same 3 cameras open concurrently before without issue -- the
devices are NOT exclusive-open. That specific risk is resolved, not just
assumed or deferred.

Device indices: this rig's 3 cameras are device 0, 1, 2, with no
`data/config.json` override present to contradict those defaults.
DEFAULT_CAMERA_DEVICES below mirrors that.

Proven pattern, followed here: try cv2.CAP_AVFOUNDATION first on
macOS, then cv2.CAP_V4L2, then fall back to cv2.CAP_ANY; set
CAP_PROP_FRAME_WIDTH/HEIGHT/FPS from config; deliberately do NOT set
fourcc or buffer size -- matching the proven capture path's
behavior, which sets neither; there is no evidence either was ever
needed.

What's written fresh here, not copied from CameraHub: this module has
no dedicated pump thread -- CameraHub's exists to protect cap.read()
from being starved by detection/overlay/case-writer load on other
threads, a contention this project doesn't have yet; grabs here are
plain synchronous cap.read() calls on demand. Also fresh: the
CameraStatus rich-debug-info surface below (which backend actually
succeeded, negotiated vs. requested resolution/fps, open/first-frame
latency, per-camera health/last-error) -- loosely modeled on the KIND of
thing CameraHub's own frame_info()/freeze_info() found worth exposing,
but not copied verbatim, adapted to what this simpler, non-threaded
module can actually know about itself.

REAL CONCURRENCY BUG FOUND AND FIXED LIVE, 2026-08-12 -- read before
touching grab()/grab_all()/_open_one()/close_all() again. "a contention
this project doesn't have yet" (above) turned out to be WRONG the moment
opendarts/live/run_product.py combined the capture loop with the dashboard
in one process: `run_product._build_components()`'s capture-loop thread
calls `hub.grab_all()` (via capture_daemon.fetch_current_frames())
continuously, roughly every POLL_INTERVAL_SECONDS (~30ms) -- and
opendarts/live/server.py's camera-preview routes
(`GET /api/cameras/{cam}/stream.mjpg`, which pulls a cached frame per
connected dashboard tile at up to MJPEG_MAX_FPS, and the snapshot/overlay
PNG stills beside it) call `hub.grab()` for the
SAME hub, via `await asyncio.to_thread(local_capture.fetch_snapshot,
...)`. Confirmed via Python's own stdlib source
(`asyncio.to_thread()` is `loop.run_in_executor(None, func_call)`,
i.e. the event loop's default `ThreadPoolExecutor`) that this runs on a
THIRD, separate real OS thread -- neither the capture loop's own
`threading.Thread` nor the main thread running uvicorn's asyncio event
loop. So with a dashboard tab open while the capture loop runs, two independent real OS threads could call
`cap.read()` on the exact SAME `cv2.VideoCapture` object at the same
time, completely unsynchronized -- `cv2.VideoCapture` is not documented
as safe for concurrent multi-thread access on one instance. This is a
real, live-CONFIRMED root cause (not theoretical): matches a REAL
recurring crash found on the macOS rig the same day this was fixed -- macOS
crash reports (`~/Library/Logs/DiagnosticReports/`, one `.ips` per
death, 9/9 matching every observed silent `run_product` death) show
`EXC_BREAKPOINT`/`PAC_EXCEPTION` (Apple Silicon's Pointer Authentication
catching real memory corruption), faulting thread `cameraQueue`, top of
stack `CFRelease` inside `cv2.abi3.so`'s own AVFoundation capture
delegate -- exactly the class of crash an unsynchronized concurrent
read/release race on a shared `cv2.VideoCapture` would produce, and
exactly why it showed no Python exception at all (a hard native crash
bypasses Python's own error handling entirely). Ruled out first (real
measurement, not assumed): a live 5s-interval resource monitor running
straight through an actual death showed FLAT RSS/FDs/thread-count/
scratch-file-count right up to the moment of death -- this is NOT a
slow accumulating leak, consistent with a rare per-iteration race rather
than a buildup. The investigation that produced this (leak theory tested
and ruled out, crash-report evidence, concurrency-model verification,
real search effort on the underlying
OpenCV/AVFoundation bug itself finding no clean upstream fix).

FIX (2026-08-12, per-camera lock -- NOT the final fix, see the dated
section below): `LocalCameraHub` gained one `threading.Lock` PER CAMERA
(not one hub-wide lock) -- see `__init__`'s own comment (at the time)
for why per-camera is the right granularity (the actual non-thread-safe
resource is one camera's `cv2.VideoCapture` instance, not "the hub" as a
whole; a hub-wide lock would also correctly prevent literal-same-instant
overlap but would additionally serialize totally unrelated cameras
against each other for no safety benefit). Every real touch of a given
camera's `cv2.VideoCapture` object (`_open_one()`'s open/configure/
warm-frame sequence, the old `grab()`'s `cap.read()`, `close_all()`'s
`cap.release()`) held that camera's lock for the duration -- this did
NOT fix the underlying OpenCV/AVFoundation bug itself (that's
third-party compiled code this project doesn't control, see the
crash-signature paragraph above), and, as the section below found, it
also did NOT fully close the gap: it prevented two threads from being
inside `cap.read()` at the *exact same instant*, but did nothing to stop
DIFFERENT OS threads from calling into the same camera's native capture
object across successive calls, one after another, un-serialized by
anything other than mutual exclusion in time. The crash kept recurring.

ARCHITECTURE CHANGE, 2026-08-12 (later the same day) -- ONE PUMP THREAD,
NOT JUST A LOCK. The per-camera lock
fix above still didn't stop the recurring crash on the macOS rig, so the
decision was to follow a known crash-free camera hub exactly. A direct
comparison against that proven capture pattern found ONE concrete
remaining structural difference, and it's not about locking at all:

  - The proven pattern: exactly ONE dedicated background thread
    (`_pump_loop`, via its own small `ThreadPoolExecutor` with one
    worker per camera) EVER calls `cap.read()`, for the whole process
    lifetime. Every other consumer -- detection, admin overlays, case
    writer, HTTP snapshot routes -- NEVER touches `cv2.VideoCapture`
    directly; they only read a cached `last_frames` list, guarded by a
    separate cheap lock over the plain Python list.
  - This module, even with the per-camera lock above: `grab()` still
    called `cap.read()` directly, on-demand, from WHICHEVER thread asked
    for a frame -- in practice alternating between
    `opendarts/live/capture_daemon.py`'s capture-loop thread (~30ms
    cadence, via `fetch_current_frames()`) and
    `opendarts/live/server.py`'s `GET /api/cameras/{cam}/snapshot.png`
    route (via `asyncio.to_thread`, a THIRD, separate real OS thread --
    neither the capture loop's thread nor the asyncio event loop's main
    thread). The per-camera lock stopped literal-same-instant overlap on
    one camera, but did NOT stop the underlying native/Objective-C
    `cap.read()` call from being made by a *different OS thread*
    depending on which caller happened to ask first -- a real, plausible
    mechanism for exactly this crash class (Objective-C runtime/
    autorelease-pool assumptions tied to a stable calling thread), and
    the one concrete remaining difference from the crash-free
    reference implementation.

WHAT CHANGED: `LocalCameraHub` now has its own `_pump_loop`/`_pump_once`
running on a single dedicated `threading.Thread` (`"local-cam-pump"`,
daemon, started in `open_all()` after cameras are opened, stopped/joined
in `close_all()` before captures are released), which owns a small
`ThreadPoolExecutor` (one worker per camera, the
`pool.map(read_one, range(n))` pattern) to read all cameras
in parallel each cycle. `grab()`/`grab_all()` are now PURE CACHE READS
-- they never call `cap.read()` themselves, protected by a separate
`_cache_lock` over the plain `_last_frames` dict (not the
`cv2.VideoCapture` object; that per-camera `_locks` dict is now only
used around `_open_one()`'s open/configure/warm-frame sequence and
`close_all()`'s release, which is genuinely different, once-only, code
outside the hot path this fix is about).

SEMANTIC SHIFT in `CameraStatus`, worth being explicit about:
`last_read_ok`/`last_error`/`last_read_at`/`frame_count`/
`actual_width`/`actual_height` used to reflect THIS call's own
`cap.read()` result (grab() updated them itself, per call). They now
reflect the PUMP's most recent read attempt for that camera -- updated
only by `_pump_once()`, never by `grab()`/`grab_all()` (which no longer
read anything themselves, so they have nothing new to report). A caller
asking "did MY grab() call get a fresh frame" is now really asking "was
the pump's last attempt for this camera OK" -- for the polling cadences
this project actually uses (capture loop ~30ms, dashboard snapshot every
~3s) that distinction is not expected to matter in practice, but it is a
real, documented change in what the field means, not silently preserved
behavior.

CACHE FRESHNESS at first-grab time: unlike the reference `CameraHub`
(whose `last_frames[i]` starts as `None` until the pump's first
successful cycle -- its `grab()` "doesn't specially wait"), this
module's existing callers (`fetch_snapshot`, in turn
`capture_daemon.py`'s `bootstrap_calibrations`/`fetch_current_frames`)
already assume a `grab()` right after `open_all()` returns either gets a
real frame or a clear failure reason -- `fetch_snapshot` raises
`RuntimeError` on a `None` frame rather than looping/waiting. To avoid a
caller silently getting `None` for however long it takes the pump to
run its first cycle (normally fast, but not instant, and not bounded
under load), `_open_one_locked()`'s existing synchronous warm-frame read
(needed anyway, to learn the real negotiated resolution before the pump
exists) now ALSO seeds `_last_frames[i]` with that frame, under
`_cache_lock`, before `open_all()` returns -- so a `grab()` immediately
after `open_all()` already has a real cached frame for any camera whose
warm read succeeded, without needing to wait on the pump's first cycle.
The warm-frame read itself deliberately stays a plain synchronous
`cap.read()` inside `_open_one_locked()`, not routed through the pump --
the pump doesn't exist yet at that point in `open_all()` (as of
2026-08-12, DIFFERENT cameras' open+warm-read sequences now run
CONCURRENTLY with each other -- see that method's own "CONCURRENT OPEN"
docstring section -- but no pump thread starts until every camera's own
sequence, each still internally single-threaded/synchronous, has
finished), and routing a single one-time warm read through pump
machinery that isn't running yet would add complexity with no benefit;
this is a genuinely different, once-only code path from the hot-path
contention this fix is about, and matches the reference `_open_one()`
(which also does its own synchronous warm read before any pump exists).

HONESTY ABOUT WHAT THIS DOES AND DOES NOT PROVE: this removes the one
concrete structural difference identified between this module and the
crash-free reference `CameraHub` -- consumer threads (capture loop,
dashboard snapshot route) no longer touch `cv2.VideoCapture` at all,
matching that real implementation, not just its docstring's summary. One
nuance worth stating precisely rather than glossing over: `
ThreadPoolExecutor` does NOT guarantee that camera `i`'s `cap.read()`
runs on the literal same OS thread every pump cycle when more than one
camera (hence more than one pool worker) is configured -- "cap.read()
only runs on the pump thread" is a simplification of what `_pump_once()`
actually does (`self._pool.map(...)`, i.e. reads happen on the pool's
worker threads, not literally the thread running `_pump_loop`); this
module keeps that same non-guarantee. A second, separate exception worth being just
as explicit about: `_open_one_locked()`'s once-only warm-frame read (see
the "CACHE FRESHNESS" section above) also calls `cap.read()`, and it
runs on WHATEVER thread calls `open_all()` -- normally a process's own
startup thread, before any pump exists and before any concurrent
consumer of `grab()` could possibly exist yet, so it does not reintroduce
the cross-thread-contention risk this fix targets, but it does mean "no
thread outside the pump ever calls cap.read()" is only true from the
moment `open_all()` returns onward, not for the hub's entire object
lifetime including construction/opening. What IS guaranteed, and is what
actually matters for the crash theory: (a) once a hub is open, no thread
outside its own pump machinery ever calls `cap.read()` on any configured
camera again, and (b) at most one read of a given SINGLE camera is ever
in flight at a time, for the hub's entire lifetime including the
one-time warm-frame read (the pump waits for the whole batch via
`list(self._pool.map(...))` before starting the next cycle, and doesn't
start until every camera has finished its own open+warm-read step --
as of 2026-08-12 those per-camera steps run CONCURRENTLY across
DIFFERENT cameras, see open_all()'s own "CONCURRENT OPEN" docstring
section, but that was never the hazard (b) is about: (b) is about one
camera's own `cv2.VideoCapture` instance never being touched by two
threads at once, which concurrent opening of DIFFERENT cameras -- each
its own independent instance -- does not violate) -- exactly the
property extended, crash-free live use of that pattern has empirically
validated as sufficient, without requiring literal single-thread-forever
stickiness per camera. This does NOT prove the
crash is fully eliminated here -- it's still third-party AVFoundation/cv2
code underneath that the reference pattern doesn't provably synchronize
against either; that pattern has simply never been observed to hit this
crash in practice, over real extended live use. Proven so far
(mock-tested, see tests/test_local_capture.py): the control-flow/
thread-identity guarantees above hold against a mocked `cv2.VideoCapture`.
NOT proven, and NOT to be claimed as proven: that this actually stops the
real `PAC_EXCEPTION`/`EXC_BREAKPOINT` crash on the real macOS rig -- that
requires live validation on real hardware, which has not happened as of
this change.

FRAME-DRIVEN WAKE PRIMITIVE, 2026-09-05 -- an approved architectural
change: go frame-driven, so every frame is processed rather than polled
and missed. Real problem this closes: `opendarts.live.
capture_daemon.run_capture_loop_body()`'s main loop used to wake on a
FIXED timer (`POLL_INTERVAL_SECONDS`, ~13.3Hz) that is slower than this
hub's own real pump cadence (~31fps/~32ms, see `_pump_once()`'s own
docstring) -- meaning the capture loop only ever actually processed
roughly 43% of the frames the pump already captured, with no signal at
all for "a new frame just became available, go look at it now."

Added `self._frame_generation: int`, a plain counter bumped by
`_pump_once()` once per COMPLETED pump cycle (regardless of whether
every individual camera's own read succeeded that cycle -- see
`_pump_once()`'s own comment for why per-camera staleness is still a
SEPARATE, unchanged concern this counter does not replace) and by
`close_all()` (so a waiter blocked on this counter is woken promptly
rather than left to time out once the hub is torn down, not just
"eventually correct"). `self._cache_lock` -- previously a bare
`threading.Lock`, guarding `_last_frames` -- is now a
`threading.Condition` wrapping that SAME lock: every existing `with
self._cache_lock:` call site (grab()/grab_all()/_pump_once()/
close_all()) is byte-identical in behavior (a `Condition` supports the
context-manager protocol exactly like the `Lock` it wraps), so this is
additive, not a rewrite of the existing cache-protection contract.

`frame_generation()` (read the current counter) and `wait_for_new_frame
(last_generation, timeout)` (block until the counter advances past
`last_generation`, or `timeout` elapses, returning whichever generation
is actually observed) are the only two new public methods. Using the
SAME lock/condition that already guards the counter's own mutation
(rather than a separate `threading.Event` a caller would have to
`.clear()` between calls) is deliberate and load-bearing: a plain Event
has a classic check-then-wait race (the pump can bump+notify in the gap
between a caller's own "is there a new frame yet?" check and its
subsequent `.wait()` call, and that notification is then lost forever,
since `Event.set()` woken with nobody yet waiting is not remembered as
a wakeup a LATER `.wait()` call could pick up if `.clear()` also ran in
between). `Condition.wait()`'s contract has no such gap: the caller's
predicate check (`while self._frame_generation == last_generation`) and
the wait itself happen atomically under the one lock the notifier ALSO
holds while it mutates+notifies -- there is no window in which a pump
cycle can complete "invisibly" between a caller's check and its wait.
See `opendarts.live.capture_daemon._wait_for_next_frame()` for how the
capture loop actually uses this (bounded per-slice waits so
`stop_event` is still re-checked at least every `poll_interval_s`,
never an unbounded/indefinite block) and its own real drop-counting
logic (this hub's counter deliberately says nothing about WHETHER a
caller kept up -- that accounting lives entirely on the consumer side,
where the "was I too slow" question actually belongs).

REAL INCIDENT, STATUS-HONESTY GAP FOUND AND FIXED, 2026-08-12 -- read
before touching `close_all()`/`CameraStatus` again. hit a real,
live `/api/stop` call that reported `already_stopped: true` -- the
900s idle-timeout (built earlier the same day) had already auto-stopped
the capture loop ~88 minutes earlier. But `GET /api/cameras/status` kept
reporting `opened: true, last_read_ok: true` for all 3 cameras the ENTIRE
time, with `last_read_at` frozen at the exact moment the pump died --
over an hour stale. A live snapshot fetch attempted during that window
failed outright (`"failed to grab a frame -- None"`, HTTP 502), directly
contradicting what the status endpoint was still claiming. Root cause:
`close_all()` stopped the pump and released every `cv2.VideoCapture`
correctly, but never touched `self.status` at all -- each camera's
`CameraStatus` just kept reporting whatever the pump last wrote before
being torn down, true at the time, silently false forever after. The
dashboard was, in effect, lying about camera state.
Fixed: `close_all()` now explicitly sets `opened=False`,
`last_read_ok=False`, and a new `closed_at` timestamp for every camera it
actually closes (see that method's own docstring for the full fix and
what's deliberately left alone as genuine history vs. reset); `open_all()`
mirrors this by clearing `closed_at` back to None the moment a fresh open
attempt begins, so a reopened camera doesn't keep showing a stale
"closed at" timestamp next to `opened: true`. This was one instance of a
broader audit requested across the whole dashboard/status surface
(not just this one field) -- see `opendarts/live/server.py`'s module
docstring for the sibling
fix found and made in `AppState.stop_capture()` (`trigger_state` had the
exact same class of bug) and the full list of surfaces checked and found
already honest.
"""
from __future__ import annotations

import contextlib
import logging
import os
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS is set in opendarts/__init__.py,
# NOT here. It has to be set before cv2 is imported ANYWHERE in the process,
# and this module is not first: run_product imports opendarts.lifecycle.settings
# (which pulls cv2 in transitively) one line before it imports this one. Setting
# it here looked right and did nothing -- see that module's docstring.
import cv2
import numpy as np

from opendarts.capture.lazy_frame import LazyFrame, LazyFrames, pixels_of, reduced_gray
from opendarts.live import camera_resolution, jpeg_info

log = logging.getLogger("opendarts.local_capture")

# -- the camera's own JPEG, kept ----------------------------------------------
#
# These cameras send MJPEG. Until 2026-09-17 every slot decoded it, and
# everything downstream -- the full-resolution stream, the v4l2 loopback,
# the Windows virtual cameras -- got a SECOND-generation JPEG encoded from
# those pixels. On one rig that turned an 82 KB q45 camera frame into a
# 122 KB q85 one that carried no more detail and paid for an encode.
#
# A slot in JPEG PASSTHROUGH keeps the bytes the camera sent next to the
# pixels decoded from them, so a consumer that wants JPEG can forward the
# camera's own:
#
#   Linux    V4L2 with CAP_PROP_CONVERT_RGB off hands back the JPEG as a
#            1xN buffer (measured on a Linux rig).
#   Windows  a Media Foundation Source Reader asked for the MJPG native
#            type with converters disabled (win_mf_capture). OpenCV's MSMF
#            ranks native types without looking at the subtype and picked
#            NV12 on a Windows rig, so it cannot be asked through OpenCV.
#   macOS    not available THROUGH OPENCV: AVFoundation is hard-wired to
#            BGRA here. The camera itself DOES send MJPEG (q85, measured
#            at the USB layer -- docs/CAMERAS.md); AVFoundation decodes it
#            and the bytes are gone by the time we see the frame. This is
#            a capture-path gap, not a hardware one -- do not read it as
#            "mac cameras give us raw pixels", which is false and has
#            misled storage decisions before.
#
# Every frame is REPAIRED (jpeg_info.repaired) and decoded before it is
# accepted, so the bytes a consumer forwards are exactly the bytes this
# rig scored. A frame that still will not decode is dropped, as a failed
# read would be.

#: Set to 0 to decode every slot the pre-2026-09-17 way.
RAW_JPEG_ENV = "OPENDARTS_RAW_JPEG"


def _raw_jpeg_wanted() -> bool:
    return os.environ.get(RAW_JPEG_ENV, "1").strip().lower() not in (
        "0", "false", "no", "off")


# SYNTHETIC JPEG, 2026-09-22 -- for slots that have NO camera JPEG to keep.
#
# macOS is the case this exists for. Its cameras DO send MJPEG (q85, measured
# at the USB layer -- docs/CAMERAS.md), but Apple's UVCAssistant decodes it
# before anything we can reach, and there is no userland route to the original
# bytes without root. So a mac slot has pixels and nothing else, and storing a
# throw means re-encoding them losslessly as PNG at ~1,100 KB a frame.
#
# The alternative this implements: encode the decoded frame to JPEG ourselves,
# DECODE IT STRAIGHT BACK, and publish that decode as the slot's pixels. The
# bytes become the slot's `jpegs` entry, so from here down the rig behaves
# exactly like a Linux or Windows passthrough slot -- same ring, same clips,
# same package shape, ~8x less disk.
#
# WHY THE DECODE IS NOT OPTIONAL. The whole point is SCORE==STORE: detection
# and scoring must run on the pixels our stored bytes decode to. Publishing
# the pre-encode frame while storing the JPEG would store something subtly
# different from what was scored, which is the exact property this rig exists
# to guarantee. So the cost is one encode AND one decode per frame (~3.3ms at
# q50 on the macOS rig, ~33% of one core across three cameras at 33fps).
#
# NOT called passthrough, deliberately. These bytes are ours, not the
# camera's, and a status flag claiming otherwise is how the macOS capture
# comment misled storage decisions for months.
#
# ALWAYS ON, for local cameras that hand us no JPEG -- there is no switch.
# A package stored as PNG costs ~10x the disk and the frame ring ~5x the
# memory, for pixels that score identically (measured: 437 corpus darts at
# q100 down to q25, then 18 live darts against Autodarts at q50 -- no score
# moved). Local cameras only: a stream already carries JPEG parts, and a
# replay source must reproduce the pixels it recorded, not a re-encode of them.
#
#: The quality we encode at. q50 is what was validated live; it is below the
#: q85 the macOS cameras send, and that loses nothing the scorer uses.
SYNTHETIC_JPEG_QUALITY = 50


def _synthesise_jpeg(frame: "np.ndarray") -> "tuple[np.ndarray, bytes] | None":
    """(pixels-to-publish, bytes-to-keep) for a slot with no camera JPEG.

    Returns None only if the round trip fails, and the caller then publishes
    the raw pixels with no bytes -- a failure here must degrade to "no
    synthetic JPEG", never to a frame that does not match its bytes.
    """
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), SYNTHETIC_JPEG_QUALITY])
    if not ok:
        return None
    data = buf.tobytes()
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        # Cannot happen for bytes we just encoded, so say so rather than
        # quietly publishing pixels that no longer match `data`.
        log.error("synthetic JPEG: our own %d-byte encode would not decode", len(data))
        return None
    return decoded, data


# DETECTION DECODES SMALL, 2026-09-26 -- a slot with a JPEG (passthrough or
# synthetic) is no longer decoded to full BGR in the pump. The worker decodes
# it straight to 1/4 grey instead (cv2.IMREAD_REDUCED_GRAYSCALE_4, what the
# lifecycle compares) and publishes a LazyFrame: the bytes plus that small
# picture, decoded to full pixels only when a consumer reads them -- in
# practice the commit frame and its reference, the dashboard preview and
# calibration. See opendarts/capture/lazy_frame.py for why scoring is still
# bit-identical. `set_small_decode(False)` (config key
# detect_from_small_decode) restores the full decode in the pump exactly.
#
#: The scale the hub pre-decodes at. It MUST be the lifecycle's
#: SignalConfig.scale for the small picture to be used; at any other scale
#: the lifecycle simply shrinks the full frame itself, as before (tested).
SMALL_DECODE_SCALE = 4


def _lazy_from_jpeg(data: bytes, scale: int,
                    full: "np.ndarray | None" = None) -> "LazyFrame | None":
    """A LazyFrame for `data`, or None if it will not decode even small.
    `full` is the frame's already-decoded pixels, when someone has them."""
    small = reduced_gray(data, scale)
    if small is None:
        return None
    return LazyFrame(data, small, scale, full)


def _synthesise_lazy(frame: "np.ndarray", scale: int) -> "tuple[LazyFrame, bytes] | None":
    """`_synthesise_jpeg()` without its full decode: the same q50 encode,
    then only the small grey decode. The full pixels, when read, are the
    decode of these bytes -- SCORE==STORE holds exactly as before."""
    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), SYNTHETIC_JPEG_QUALITY])
    if not ok:
        return None
    data = buf.tobytes()
    lazy = _lazy_from_jpeg(data, scale)
    if lazy is None:
        log.error("synthetic JPEG: our own %d-byte encode would not decode", len(data))
        return None
    return lazy, data


def _raw_to_lazy(buf: Any, scale: int) -> "tuple[str, Any, bytes | None]":
    """`_raw_to_frame()` for a small-decode slot: same classification, but
    a good JPEG comes back as a LazyFrame, decoded only to small grey. The
    reduced decode still runs libjpeg over every entropy-coded block, so a
    frame that will not decode is rejected here just as before."""
    if buf is None:
        return "bad", None, None
    if getattr(buf, "ndim", 0) == 3:
        return "pixels", buf, None
    data = np.asarray(buf, dtype=np.uint8).reshape(-1).tobytes()
    fixed = jpeg_info.repaired(data)
    if fixed is None:
        return "other", None, None
    lazy = _lazy_from_jpeg(fixed, scale)
    if lazy is None:
        return "bad", None, None
    return "jpeg", lazy, fixed


def _takes_lazy(sink: Any) -> bool:
    """Whether a frame sink handles LazyFrame values itself (reads only
    the geometry when it forwards the JPEG, decodes when it needs pixels).
    Anything else is handed full pixels, decoded on the pump's workers."""
    return bool(getattr(sink, "accepts_lazy_frames", False))


#: The Media Foundation pseudo-backend, tried before CAP_MSMF on Windows.
#: Not a cv2 constant: nothing in OpenCV can open a camera this way.
MF_JPEG_BACKEND = "MF_JPEG"


def _open_mf_jpeg(device: Any) -> Any:
    """A Media Foundation camera that reads like a cv2.VideoCapture.
    Imported lazily and a seam for tests: the module is Windows-only in
    everything but its import."""
    from opendarts.live import win_mf_capture

    return win_mf_capture.MfJpegCapture(device)


def _dshow_index_for(device: Any) -> "int | None":
    """The DirectShow index of Media Foundation camera `device`, matched by
    device path; None when it cannot be matched. Real Windows only.

    This is also what keeps a slot from ever reading one of our own virtual
    cameras: Media Foundation does not list them, so no slot number can
    reach one (a Windows rig read its own output through OpenCV's DirectShow fallback
    on 2026-09-17, when the fallback opened by number)."""
    if sys.platform != "win32" or not isinstance(device, int):
        return None
    try:
        from opendarts.live import camera_names

        return camera_names.dshow_index_for(device)
    except Exception:  # noqa: BLE001 -- unmatched is the safe answer
        return None


def _raw_to_frame(buf: Any) -> "tuple[str, Any, bytes | None]":
    """Classify one read made with conversion off.

    Returns (kind, frame, jpeg):
      "pixels"  the backend ignored the request and sent decoded pixels
      "jpeg"    a JPEG that decoded; `jpeg` is the repaired bytes
      "bad"     a JPEG that would not decode, even repaired
      "other"   not a JPEG at all (an uncompressed format, say)
    """
    if buf is None:
        return "bad", None, None
    if getattr(buf, "ndim", 0) == 3:
        return "pixels", buf, None
    data = np.asarray(buf, dtype=np.uint8).reshape(-1).tobytes()
    fixed = jpeg_info.repaired(data)
    if fixed is None:
        return "other", None, None
    frame = cv2.imdecode(np.frombuffer(fixed, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return "bad", None, None
    return "jpeg", frame, fixed

# This rig's real 3 camera device indices -- see module docstring for
# where this was confirmed. Not a guess.
DEFAULT_CAMERA_DEVICES: list[int] = [0, 1, 2]

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 30


def camera_configs_from_resolution_preferences(
    camera_resolutions: "dict[int, tuple[int, int] | None]",
    *,
    devices: list[int] | None = None,
) -> list["CameraConfig"]:
    """Pure, cv2-free helper: turn `LiveConfig.camera_resolutions` (see
    opendarts.live.config's own module docstring for the field's real JSON
    shape and parsing) into the real `list[CameraConfig]` a caller would
    pass to `LocalCameraHub(configs=...)`. Not called by any real
    LocalCameraHub construction site in this app yet (see
    opendarts.live.camera_resolution's own module docstring for why that
    final wiring is deliberately left for a follow-up with real
    live-hardware validation, not done blind from this dev sandbox) --
    exists so the config-parsing side of this fix is genuinely usable and
    testable end-to-end (config JSON -> LiveConfig -> real CameraConfig
    list) without requiring that last wiring step to prove it works.

    `devices` defaults to `DEFAULT_CAMERA_DEVICES` -- same convention
    `LocalCameraHub.__init__`'s own bare default already uses.

    **Backward compatibility, provably not just assumed**: an empty
    `camera_resolutions` dict (what `load_live_config()` returns for a
    missing file, or one that doesn't mention `camera_resolutions` at
    all -- today's real-world default on every machine that hasn't
    hand-edited this in) produces `[CameraConfig(device=d) for d in
    devices]` -- BYTE-IDENTICAL to `LocalCameraHub.__init__`'s own bare
    `configs or [CameraConfig(device=d) for d in
    DEFAULT_CAMERA_DEVICES]` default, field for field (see
    `tests/test_camera_resolution.py`'s own explicit assertion of this).
    A camera index present in the dict with value None means "auto" --
    resolves to `CameraConfig(device=d, width=None, height=None)` (the
    opt-in probe mode -- see that dataclass's own docstring); a value of
    `(w, h)` resolves to a fixed `CameraConfig(device=d, width=w,
    height=h)`, same as hand-writing that override today.
    """
    devices = devices if devices is not None else DEFAULT_CAMERA_DEVICES
    configs: list[CameraConfig] = []
    for device in devices:
        if device not in camera_resolutions:
            configs.append(CameraConfig(device=device))
            continue
        preference = camera_resolutions[device]
        if preference is None:
            configs.append(CameraConfig(device=device, width=None, height=None))
        else:
            width, height = preference
            configs.append(CameraConfig(device=device, width=width, height=height))
    return configs


def _backend_name(backend: int) -> str:
    """Human-readable name for an OpenCV VideoCapture backend constant --
    OpenCV itself has no reverse lookup for this, so a small explicit
    map is built here from whichever constants this cv2 build exposes."""
    names = {}
    # Every platform's backends, not just this one's: a capture can be
    # served by CAP_ANY resolving to something we never named, and
    # "UNKNOWN_BACKEND(1400)" in a log is strictly worse than "CAP_MSMF"
    # when the backend is exactly what you are trying to diagnose.
    for attr in ("CAP_AVFOUNDATION", "CAP_V4L2", "CAP_MSMF", "CAP_DSHOW",
                 "CAP_GSTREAMER", "CAP_FFMPEG", "CAP_ANY"):
        value = getattr(cv2, attr, None)
        if value is not None:
            names[value] = attr
    return names.get(backend, f"UNKNOWN_BACKEND({backend})")


def _msmf_available() -> "bool | None":
    """Whether this OpenCV build can actually use Media Foundation.

    `videoio_registry.hasBackend` is the honest question. The backend LIST
    is not: getCameraBackends() names MSMF even when its plugin cannot
    load (measured on a Windows VM, 2026-09-17), so a check against the
    list passes on exactly the build this exists to catch.
    """
    try:
        from cv2 import videoio_registry
        return bool(videoio_registry.hasBackend(cv2.CAP_MSMF))
    except Exception:  # noqa: BLE001 -- unknown is not the same as missing
        return None


def _platform_backend() -> int:
    """The one backend this product uses on the OS it is running on.

    OpenCV's side of `_open_one_locked`'s choice, for the probe below. On
    Windows that is DirectShow: cameras are read through our own Media
    Foundation code first, and the headless OpenCV package has no MSMF.
    The probe opens at the number given, in DirectShow's own order.
    """
    return {
        "Darwin": cv2.CAP_AVFOUNDATION,
        "Linux": cv2.CAP_V4L2,
        "Windows": cv2.CAP_DSHOW,
    }.get(platform.system(), cv2.CAP_ANY)


def _fourcc_text(value: float) -> "str | None":
    """Four packed ASCII bytes as text, or a labelled raw number.

    MSMF does not always return a real FOURCC here -- with conversion on it
    returned 22.0 on the Windows rig, an output-format code rather than
    ASCII -- so anything that is not four printable characters is reported
    as the number it was, instead of four bytes of noise.
    """
    v = int(value) if value and value > 0 else 0
    if not v:
        return None
    raw = bytes((v >> (8 * i)) & 0xFF for i in range(4))
    if all(32 <= c < 127 for c in raw):
        return raw.decode("ascii")
    return f"raw:{value:g}"


def _opencv_install_facts() -> "dict[str, Any]":
    """Which OpenCV package is installed, and what the build says it has.

    The installed PACKAGE decides the backends: on Windows the headless
    wheel has no Media Foundation. On a machine with no shell, this is the
    only way to see which one a rig actually ended up with -- and whether
    it has BOTH, which is what an in-place upgrade from headless leaves
    behind. `getBuildInformation()` is OpenCV's own statement of what was
    compiled in, so its Media Foundation line is quoted rather than
    inferred.
    """
    facts: "dict[str, Any]" = {"cv2_file": getattr(cv2, "__file__", None)}
    try:
        from importlib import metadata

        facts["packages"] = {
            name: _dist_version(metadata, name)
            for name in ("opencv-python", "opencv-python-headless",
                         "opencv-contrib-python", "opencv-contrib-python-headless")
        }
    except Exception as exc:  # noqa: BLE001
        facts["packages"] = f"unavailable: {exc}"
    try:
        lines = cv2.getBuildInformation().splitlines()
        facts["build_video_io"] = [ln.strip() for ln in lines
                                   if any(k in ln for k in ("Media Foundation", "DirectShow",
                                                            "AVFoundation", "v4l", "FFMPEG:"))]
    except Exception as exc:  # noqa: BLE001
        facts["build_video_io"] = f"unavailable: {exc}"
    return facts


def _dist_version(metadata: Any, name: str) -> "str | None":
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 -- not installed
        return None


def _windows_camera_context() -> "dict[str, Any]":
    """Why MSMF might refuse a camera that DSHOW opens: three checks.

    Written after a Windows rig's MSMF would open neither of its cameras on
    2026-09-17 while DSHOW opened both. The actual cause turned out to be
    the first field below -- the headless OpenCV wheel has no MSMF
    backend at all -- but the other three were the reasonable suspects
    and stay, because each is a real way for MSMF to refuse a camera:

      * the process's SESSION -- the Frame Server serves the interactive
        desktop session, and a process started as a service or a
        scheduled task without a desktop is session 0
      * whether the FrameServer service is running at all
      * the camera privacy switches -- all apps, and desktop apps
        specifically -- under CapabilityAccessManager

    Every value is best-effort and reported as found: a check that cannot
    run says so rather than guessing.
    """
    info: "dict[str, Any]" = {"msmf_available": _msmf_available()}
    try:
        import ctypes

        session = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.ProcessIdToSessionId(os.getpid(), ctypes.byref(session))
        info["session_id"] = int(session.value) if ok else None
    except Exception as exc:  # noqa: BLE001
        info["session_id"] = f"unavailable: {exc}"
    try:
        import subprocess

        res = subprocess.run(["sc", "query", "FrameServer"], capture_output=True,
                             text=True, timeout=5)
        state = next((ln.split(":", 1)[1].strip() for ln in res.stdout.splitlines()
                      if "STATE" in ln), None)
        info["frameserver"] = state or res.stdout.strip()[-120:] or f"exit {res.returncode}"
    except Exception as exc:  # noqa: BLE001
        info["frameserver"] = f"unavailable: {exc}"
    try:
        import winreg

        base = r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam"
        privacy: "dict[str, Any]" = {}
        for label, hive, sub in (
            ("machine", winreg.HKEY_LOCAL_MACHINE, base),
            ("user", winreg.HKEY_CURRENT_USER, base),
            ("user_desktop_apps", winreg.HKEY_CURRENT_USER, base + r"\NonPackaged"),
        ):
            try:
                with winreg.OpenKey(hive, sub) as key:
                    privacy[label] = winreg.QueryValueEx(key, "Value")[0]
            except OSError:
                privacy[label] = None
        info["camera_privacy"] = privacy
    except Exception as exc:  # noqa: BLE001
        info["camera_privacy"] = f"unavailable: {exc}"
    return info


#: How long a window the achieved-fps figure covers. Long enough that
#: normal jitter averages out, short enough that a stall is visible within
#: a couple of seconds rather than diluted across a whole session.
FPS_WINDOW_SECONDS = 2.0


def _update_effective_fps(status: "CameraStatus", now_monotonic: float) -> None:
    """Fold one frame into the rolling rate.

    Monotonic, not wall clock: a clock adjustment mid-session would
    otherwise produce a nonsense rate, or a negative one.
    """
    if status.fps_window_start is None:
        status.fps_window_start = now_monotonic
        status.fps_window_frames = 0
    status.fps_window_frames += 1
    elapsed = now_monotonic - status.fps_window_start
    if elapsed >= FPS_WINDOW_SECONDS and status.fps_window_frames > 1:
        # (n - 1), not n: `elapsed` spans the INTERVALS between frames, and
        # n timestamps bound n-1 intervals. Dividing by n overstates the
        # rate by one frame per window -- which is the wrong direction for
        # a number whose whole purpose is to reveal a shortfall.
        status.effective_fps = round((status.fps_window_frames - 1) / elapsed, 1)
        # The triggering frame opens the next window rather than being
        # discarded, so no frame is counted twice and none is lost.
        status.fps_window_start = now_monotonic
        status.fps_window_frames = 1


@dataclass
class CameraConfig:
    """Per-camera configuration (device/width/height/fps/enabled).
    Self-contained by design: this shape is owned here, with no runtime
    dependency on anything outside this project.

    RESOLUTION PROBING, added 2026-08-20 (the "hardcoded 1280x720 with no
    runtime check" bug fix -- see opendarts.live.camera_resolution's module
    docstring for the full bug this closes). `width`/`height` default to
    `DEFAULT_WIDTH`/`DEFAULT_HEIGHT` (1280/720) exactly as before this
    change -- a bare `CameraConfig()` (what `LocalCameraHub.__init__`'s
    own default builds for every caller that doesn't pass explicit
    configs, i.e. every real call site in this app today) behaves
    IDENTICALLY to before this change: `_open_one_locked()` sets that
    fixed width/height directly, no probing happens at all. Set BOTH to
    `None` (an explicit, opt-in choice a caller must make -- never the
    default) to request "auto" mode instead: `_open_one_locked()` then
    probes this camera's genuinely-supported resolutions
    (`opendarts.live.camera_resolution.highest_supported_resolution()`) once
    it's opened, and requests the highest one found -- falling back to
    `DEFAULT_WIDTH`/`DEFAULT_HEIGHT` with a logged warning if probing
    finds nothing (a real, if unlikely, possible outcome for an unusual
    camera -- see that function's own docstring). Setting only ONE of
    `width`/`height` to `None` is not a supported half-state; treat it as
    "auto" (both None) if either is None, since a fixed one-axis-only
    request has no clear meaning against a probed pair.
    """

    device: int | str = 0
    width: int | None = DEFAULT_WIDTH
    height: int | None = DEFAULT_HEIGHT
    fps: int = DEFAULT_FPS
    enabled: bool = True

    @property
    def auto_resolution(self) -> bool:
        """True if this config asks `_open_one_locked()` to probe for the
        highest genuinely-supported resolution instead of requesting a
        fixed `width`/`height` -- see this dataclass's own docstring for
        why either field being None is treated as "auto" rather than a
        half-fixed, half-probed state."""
        return self.width is None or self.height is None


@dataclass
class CameraStatus:
    """Rich per-camera debug/health info, kept up to date by
    LocalCameraHub as it opens cameras and grabs frames -- what
    asked this module surface beyond a bare open/closed bool. Loosely
    modeled on the kind of thing a camera hub's frame_info()/
    freeze_info() expose (requested vs. actual size, per-camera state)
    but not copied verbatim -- adapted to this module's own (simpler,
    non-threaded) shape.
    """

    device: int | str
    opened: bool = False
    backend_used: str | None = None # e.g. "CAP_AVFOUNDATION"
    requested_width: int = 0
    requested_height: int = 0
    requested_fps: int = 0
    actual_width: int = 0
    actual_height: int = 0
    actual_fps: float = 0.0
    # The pixel format the driver ACTUALLY negotiated, as the 4-character
    # FOURCC ("MJPG", "YUY2", ...) -- read back, never requested (this
    # module deliberately does not set fourcc; see _open_one()'s own
    # comment). Surfaced 2026-09-13 because it is the difference between
    # a camera set that can share a USB dock and one that cannot, and
    # nothing reported it: uncompressed YUY2 at 1280x720x30 is ~442
    # Mbit/s per camera against roughly 320 Mbit/s of usable USB 2.0
    # bandwidth, so ONE saturates a controller, while MJPEG compresses
    # far enough for three. "Camera set A shares a dock, set B does not"
    # is that fact, and this is where you can now see it.
    # None when the backend does not report it or the camera never opened.
    #
    # MEASURED ON THE WINDOWS RIG, 2026-09-13: CAP_MSMF does NOT report a
    # FOURCC. It returns 22.0 for every camera -- an index into OpenCV's
    # own internal media-type table, not four packed ASCII bytes -- so
    # this reads None there and the format cannot be learned this way.
    # Recorded so the next person does not spend the same restart cycle
    # on it: to read the real format on Windows you need CAP_DSHOW (which
    # does report FOURCC) or a query outside OpenCV. AVFoundation on macOS
    # is untested for this.
    actual_fourcc: str | None = None

    open_latency_s: float | None = None
    first_frame_latency_s: float | None = None
    # frame_count/last_read_ok/last_read_at/last_error/actual_width/
    # actual_height (after open) are now updated by LocalCameraHub's own
    # pump thread (_pump_once()), NOT by grab()/grab_all() -- see module
    # docstring's 2026-08-12 "ARCHITECTURE CHANGE" section for the full
    # semantic-shift writeup. They describe the PUMP's most recent read
    # attempt for this camera, not "did the caller's own grab() call get
    # a fresh frame" (grab() no longer reads anything itself).
    frame_count: int = 0
    last_read_ok: bool = False
    last_read_at: float | None = None
    # FRAME-AGE DIAGNOSTIC companion field, 2026-09-04 -- deliberately a
    # SECOND field, not a change to last_read_at's own meaning. the project's
    # own explicit authorization: "add all the debugs you want," to
    # answer "when IDLE -> MOTION_DETECTED finally fires, how old was
    # the frame that tripped it" -- which needs `last_read_at` and the
    # consuming code's own clock to share ONE clock domain
    # (`time.monotonic()`, never `time.time()` -- NTP can step
    # wall-clock time mid-comparison, silently corrupting an age
    # computed as a plain subtraction; `time.monotonic()` never steps).
    # `last_read_at` itself is deliberately left alone: it's stamped with
    # `time.time()` (wall-clock) and exposed verbatim via `GET
    # /api/cameras/status` (opendarts/live/server.py's `cameras_status_dict()`
    # -> `dataclasses.asdict(status)`) -- a real, external JSON consumer
    # whose field NAME already implies "an epoch timestamp," and nothing
    # in this codebase requires that meaning to change (grepped every
    # reader before adding this field -- none does arithmetic against it
    # as an absolute value, but the dashboard's own raw-JSON surface is a
    # real external contract this diagnostic task has no reason to
    # touch). Stamped alongside `last_read_at` at the open-time warm
    # frame AND `_pump_once()`'s SUCCESSFUL-read branch only --
    # deliberately NOT the pump's failed-attempt branch (which still
    # updates `last_read_at`/`last_read_ok`/`last_error`, matching that
    # field's own documented "last ATTEMPT, not last SUCCESS" semantic)
    # -- see `_pump_once()`'s own comment at its `now_monotonic` local
    # for why: a frame-AGE diagnostic needs "how long ago was the frame
    # actually served by grab()/grab_all() captured," and a failed
    # attempt never replaces that cached frame, so stamping this field
    # on failure would understate the served frame's real age.
    last_read_at_monotonic: float | None = None
    last_error: str | None = None
    # STATUS-HONESTY FIX, 2026-08-12 (see module docstring's dated
    # incident writeup): set by LocalCameraHub.close_all() the moment
    # THIS camera's capture is actually released -- None while the
    # camera has never been closed this hub lifetime (never opened, or
    # currently open). Lets a consumer tell "opened=False because it
    # never opened / failed to open" (closed_at is None, last_error
    # explains why) apart from "opened=False because it WAS open and was
    # then deliberately closed" (closed_at is set) -- two different
    # truths that collapsing onto opened=False alone would blur. Also
    # gives an honest answer to "how stale is last_read_at now" without
    # having to guess: `time.time() - closed_at` is exact, whereas
    # last_read_at alone (left untouched, see close_all()) would read as
    # "just now" right after a close and only reveal its own staleness
    # by comparison to wall-clock time the caller has to compute itself.
    closed_at: float | None = None

    # Rolling achieved frame rate, 2026-09-11. A camera configured for
    # 30fps that is actually delivering 21 is degrading silently: the
    # driver drops those frames before this process ever sees them, so
    # there is no counter to read and the rate deficit is the only
    # evidence. frame_count cannot show it -- a count always rises,
    # however slowly.
    #
    # Measured over a window rather than since-open, so a stall shows up
    # promptly instead of being averaged away by however long the session
    # has already been healthy.
    fps_window_start: float | None = None
    fps_window_frames: int = 0
    effective_fps: float | None = None

    # JPEG PASSTHROUGH, 2026-09-17 -- see the module-level section of that
    # name. True when this slot keeps the camera's own JPEG next to its
    # pixels. `jpeg_rejected` counts camera frames dropped because they
    # would not decode even after repair; a camera that sends them at all
    # is worth knowing about.
    jpeg_passthrough: bool = False
    jpeg_rejected: int = 0
    #: SYNTHETIC JPEG (see that module-level section): this slot had no
    #: camera JPEG, so we encoded one ourselves and published the decode of
    #: it. Distinct from `jpeg_passthrough`, which means the bytes came from
    #: the camera -- these did not, and conflating the two would misreport
    #: provenance in the one place an operator goes to check it.
    jpeg_synthetic: bool = False
    jpeg_synthetic_quality: int | None = None

    def summary(self) -> str:
        """One-line human-readable status -- used for console logging
        both here and in scripts/test_local_camera_access.py."""
        if self.closed_at is not None:
            return (
                f"device={self.device} CLOSED (was opened via {self.backend_used}, "
                f"{self.frame_count} frame(s) read, closed_at={self.closed_at:.3f})"
            )
        if not self.opened:
            return f"device={self.device} FAILED TO OPEN ({self.last_error})"
        first_frame = (
            f"{self.first_frame_latency_s:.3f}s"
            if self.first_frame_latency_s is not None
            else "n/a"
        )
        open_latency = f"{self.open_latency_s:.3f}s" if self.open_latency_s is not None else "n/a"
        return (
            f"device={self.device} backend={self.backend_used} "
            f"requested={self.requested_width}x{self.requested_height}@{self.requested_fps}fps "
            f"actual={self.actual_width}x{self.actual_height}@{self.actual_fps:.1f}fps "
            f"fourcc={self.actual_fourcc or 'n/a'} "
            f"jpeg_passthrough={self.jpeg_passthrough} "
            f"open_latency={open_latency} first_frame_latency={first_frame} "
            f"frames_read={self.frame_count} last_read_ok={self.last_read_ok}"
        )


def _takes_jpegs(sink: Any) -> bool:
    """Whether a frame sink accepts `jpegs=`. Asked once, when the sink is
    attached, not per frame: both publishers take it, but a plain
    callable in a test or a script may not."""
    if sink is None:
        return False
    import inspect

    try:
        params = inspect.signature(sink).parameters
    except (TypeError, ValueError):
        return False
    return "jpegs" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _fourcc_to_str(raw: float | int | None) -> str | None:
    """cv2's CAP_PROP_FOURCC comes back as a float holding four packed
    ASCII bytes -- turn it into "MJPG"/"YUY2"/etc.

    Returns None rather than a garbage string when the backend does not
    report one (0.0 is the common "unknown" answer, and AVFoundation
    reports formats cv2 does not always pack cleanly), so a caller can
    tell "not reported" from a real format.
    """
    try:
        code = int(raw or 0)
    except (TypeError, ValueError):
        return None
    if code <= 0:
        return None
    chars = [chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    text = "".join(chars).strip()
    # Anything non-printable means this was not really a FOURCC.
    if not text or any(not (32 <= ord(c) < 127) for c in text):
        return None
    return text


class CameraHub:
    """The rig's frame source: N slots, each owning its own feed.

    Opens whatever a slot points at once, reads every slot continuously
    from ONE dedicated pump thread, and serves every consumer
    (grab()/grab_all()) from a single cache -- see module docstring's
    2026-08-12 "ARCHITECTURE CHANGE" section for the full incident
    writeup that pump implements. _open_one()'s backend-fallback +
    width/height/fps handling and the _pump_loop/_pump_once +
    ThreadPoolExecutor-per-cycle design are this module's own.

    ONE HUB, N SLOTS, 2026-09-16. A slot reads EITHER a local
    cv2.VideoCapture or an MJPEG stream from another machine
    (`urls[slot]`, see opendarts/live/remote_capture.py). Until today
    those were two separate hub classes with a wrapper mapping
    slot -> (which child, which index inside it), and four bugs in one
    day were all failures of that mapping rather than of either child --
    remote_capture.py's module docstring lists them. There is now exactly
    one numbering scheme, the slot index, and nothing translates between
    schemes because there is nothing to translate:

      * one frame cache (`_last_frames`, slot-keyed)
      * one generation counter, bumped once per pump CYCLE
      * one frame sink, called once per cycle with every slot present
      * one `status`/`configs` list, in slot order

    The pump does not know or care which kind a slot is; `_read_one()` is
    the only place that branches, and all it asks either source for is a
    frame. The class is still named for what it IS -- the hub -- rather
    than for one kind of source; `LocalCameraHub` remains as an alias
    because a lot of call sites and docs spell it that way.

    A STREAM SLOT OPENS NO LOCAL DEVICE. That is structural here rather
    than guarded: `open_all()` only ever calls `_open_one()` for a slot
    with no URL. The wrapper this replaced had to special-case it, got it
    wrong, and cost ~2.6s per phantom camera, serially, before the
    capture loop could run.
    """

    def __init__(self, configs: list[CameraConfig] | None = None,
                 frame_sink=None,
                 urls: "list[str | None] | None" = None,
                 sources: "list[Any] | None" = None,
                 frame_ring=None) -> None:
        if configs:
            self.configs: list[CameraConfig] = list(configs)
        elif sources:
            # A SOURCE LIST SIZES THE HUB TOO, for the same reason a URL
            # list does: without this, a three-slot replay would take the
            # default branch below and open three local devices nobody
            # asked for. That is the phantom-camera bug reached from a
            # third side, and it is worth closing here rather than
            # trusting every replay caller to also pass configs.
            self.configs = [
                CameraConfig(device=(DEFAULT_CAMERA_DEVICES[i]
                                     if i < len(DEFAULT_CAMERA_DEVICES) else i))
                for i in range(len(sources))
            ]
        elif urls:
            # A URL LIST ON ITS OWN SIZES THE HUB. Without this branch,
            # `CameraHub(urls=["http://rig/0"])` would take the default
            # branch below, get three slots, and turn slots 1 and 2 into
            # local devices nobody asked for -- the phantom-camera bug
            # again, reached from the other side. The device index is kept
            # for the status line and for anyone who later swaps that slot
            # back to hardware; nothing opens it while a URL is set.
            self.configs = [
                CameraConfig(device=(DEFAULT_CAMERA_DEVICES[i]
                                     if i < len(DEFAULT_CAMERA_DEVICES) else i))
                for i in range(len(urls))
            ]
        else:
            # `configs or DEFAULT` for thirty commits: BOTH None and []
            # mean "the default three devices", and there is no way to
            # spell "no cameras" here. Preserved deliberately -- a rig with
            # no config keys must build exactly the hub it always did --
            # and the two branches above are what make the cases that
            # needed a different answer expressible instead.
            self.configs = [CameraConfig(device=d) for d in DEFAULT_CAMERA_DEVICES]
        # Which slots read a stream instead of a device. Normalised to
        # exactly len(self.configs) entries here, once, so every later
        # reader can index it without a bounds check -- and LOUDLY, because
        # a silently trimmed URL is a slot reading hardware while the
        # dashboard shows it reading a feed.
        self._slot_urls: list[str | None] = self._normalise_urls(urls)
        # Pre-built source objects, one per slot or None -- the third way a
        # slot can be fed, after "a local device" and "a URL". Unlike a URL
        # (which this hub turns into a StreamSource at open time) these
        # arrive already constructed, because the thing feeding them is a
        # file the CALLER located: a replay source needs a dump directory,
        # a slot mapping and a shared read cursor, none of which this
        # module has any business knowing about. open_all() starts them and
        # close_all() stops them exactly as it does a stream.
        #
        # Takes precedence over a URL for the same slot, and says so -- two
        # sources for one slot is a caller mistake, and silently picking
        # one would present as "the stream I configured is not being read".
        self._preset_sources: list[Any] = self._normalise_sources(sources)
        # THE RING TAP. Optional; see opendarts/capture/frame_ring.py for
        # what it is and why it costs no CPU. Attached here rather than in
        # the capture loop because the hub is the one place every rig has:
        # a camera rig's ring holds what its cameras produced and a
        # stream-fed rig's holds what it decoded off the wire, and each is
        # the authoritative record of what THAT consumer scored.
        self._frame_ring = frame_ring
        self.frame_ring_errors = 0
        # Optional `callable(dict[int, ndarray]) -> None`, invoked once per
        # pump cycle with the frames just cached. Deliberately a plain
        # callback rather than a type this module knows: the only consumer
        # today republishes to Windows virtual cameras, and this file has
        # no business knowing that exists.
        #
        # Called OUTSIDE the cache lock and wrapped, so a slow or broken
        # sink can never stall a frame grab or take the pump down -- the
        # capture loop's correctness must not depend on a diagnostic.
        self._frame_sink = frame_sink
        # (sink, whether it takes jpegs=) as ONE reference, so a sink
        # swapped mid-cycle is never called with the other one's answer.
        self._sink_entry = (frame_sink, _takes_jpegs(frame_sink))
        self.frame_sink_errors = 0
        self._caps: dict[int, cv2.VideoCapture] = {}
        # Local slots reading in JPEG passthrough -- see the module-level
        # section. Written only while a slot opens, before the pump runs.
        self._raw_slots: set[int] = set()
        self.status: dict[int, CameraStatus] = {
            i: CameraStatus(device=cfg.device) for i, cfg in enumerate(self.configs)
        }
        # One SOURCE OBJECT per slot that is not a local device, keyed by
        # SLOT -- the same key as status/configs/_last_frames, because
        # there is only one numbering scheme now.
        #
        # Called `_streams` until 2026-09-16, when a THIRD kind of slot
        # source arrived (dev.capture.frame_replay.ReplaySource,
        # which reads a saved frame dump and paces it at its recorded
        # timestamps). A replay source is not a stream and calling it one
        # would have made `stream.read_failure_reason()` in the pump read
        # as a network question about a file.
        #
        # THE CONTRACT A SOURCE OBJECT SIGNS, which is what lets the pump
        # treat all of them identically and is the reason `_read_one()`
        # is the only method that branches at all:
        #   * `.slot` -- which slot it feeds, for close_all()'s status writes
        #   * `.start()` / `.stop()` / `.join()` -- stop() must wake anyone
        #     blocked in read(), because close_all() signals every source
        #     before joining any of them
        #   * `.read()` -- BLOCKS until it has a frame the pump has not
        #     already been given, returns None when it cannot deliver. It
        #     is what paces the pump for that slot, exactly as a local
        #     camera's own cap.read() does.
        #   * `.read_failure_reason()` -- what to put on CameraStatus when
        #     read() came back empty, in the source's own vocabulary
        #   * and it must hand over a FRESH array it will never write into
        #     again, which is why _pump_once() skips its defensive copy for
        #     these slots and does not for a cv2.VideoCapture.
        self._sources: dict[int, Any] = {}
        # Per-camera lock -- see module docstring's "FIX (2026-08-12,
        # per-camera lock)" section for the original incident this was
        # built for. As of the pump-thread architecture change (same
        # day, later section), this is ONLY still used around
        # _open_one()'s open/configure/warm-frame sequence and
        # close_all()'s release() -- genuinely different, once-only code
        # paths, not the grab()/cap.read() hot path this class used to
        # serialize this way. Built here, once, for every configured
        # camera index -- not lazily -- so there's no window where a
        # lock could be missing for a valid camera index.
        self._locks: dict[int, threading.Lock] = {
            i: threading.Lock() for i in range(len(self.configs))
        }

        # Pump-thread cache -- see module docstring's "ARCHITECTURE
        # CHANGE" section. _last_frames is the ONLY thing grab()/
        # grab_all() ever read; _pump_once() is the ONLY thing that ever
        # writes to it. _cache_lock guards this plain Python dict, NOT
        # any cv2.VideoCapture object -- a different, much cheaper lock
        # than self._locks above -- it guards last_frames, not a
        # camera device.
        self._last_frames: dict[int, np.ndarray | None] = {}
        # The camera's own JPEG for the frame in `_last_frames`, or absent.
        # Written in the same locked step as the frame, so a reader using
        # grab_with_jpeg() never gets bytes from one frame and pixels from
        # another.
        self._last_jpegs: dict[int, bytes] = {}
        # The pump generation that FIRST published the frame in
        # `_last_frames` -- the ring set (opendarts.capture.frame_ring)
        # holding exactly that frame. Written in the same locked step as
        # the frame. A slot re-served unchanged by a failed read keeps its
        # number: the frame, and so the ring set it names, is the same one.
        # Absent for a frame the pump never published (a camera's warm
        # frame at open), which no ring set holds.
        self._slot_generation: dict[int, int] = {}
        # A Condition wrapping a plain Lock -- see module docstring's
        # "FRAME-DRIVEN WAKE PRIMITIVE" section for why this replaced a
        # bare Lock (2026-09-05): every pre-existing `with
        # self._cache_lock:` call site is unaffected (Condition supports
        # the same context-manager protocol), this only ADDS the ability
        # to wait()/notify_all() on frame-generation changes using the
        # exact same lock that already serializes cache reads/writes.
        self._cache_lock: threading.Condition = threading.Condition()
        # Bumped once per completed _pump_once() cycle (see that
        # method's own comment) and once by close_all() -- see
        # wait_for_new_frame()'s own docstring for the full contract.
        self._frame_generation: int = 0
        # DETECTION DECODES SMALL (see the module-level section): the scale
        # JPEG slots are pre-decoded at, or None for the full decode in the
        # pump. Off by default here; run_product turns it on from config.
        self._small_decode_scale: "int | None" = None
        self._pool: ThreadPoolExecutor | None = None
        self._pump_thread: threading.Thread | None = None
        self._pump_stop: threading.Event = threading.Event()

        # Guards structural mutation of self._caps (insert/clear) -- added
        # 2026-08-12 alongside open_all()'s move to CONCURRENT camera
        # opening (see that method's own docstring for the full incident
        # writeup). Each camera's own self._locks[i] already prevents that
        # ONE camera's open sequence from overlapping itself, but it does
        # NOT protect the shared self._caps dict structure across DIFFERENT
        # cameras writing DIFFERENT keys at literally the same instant --
        # that's a different lock object per camera, so it serializes
        # nothing cross-camera. In practice, CPython's GIL makes a single
        # `dict[key] = value` a single atomic bytecode-level operation, so
        # concurrent inserts on DISTINCT keys cannot corrupt the dict even
        # without this lock -- but relying on that as the ONLY reasoning
        # would be exactly the kind of "assume it's fine" this project's own
        # discipline (see _cache_lock above, and docs/DESIGN.md's "measured, not
        # guessed" culture) argues against; this costs nothing (only held
        # once per camera per open_all() call, nowhere near the pump's hot
        # path) and removes any reliance on an implementation detail.
        self._caps_lock: threading.Lock = threading.Lock()

    def _normalise_urls(self, urls: "list[str | None] | None") -> list[str | None]:
        """One entry per slot, padded or trimmed, SAYING SO when it had to.

        Padding is ordinary (the dashboard sends only the slots it knows
        about). Trimming is not: it means a URL was handed in for a slot
        this hub does not have, and dropping that quietly would leave the
        dashboard showing a stream the process is not reading -- the exact
        class of silent refusal docs/DESIGN.md forbids. Neither case is
        worth raising over, because the slot count legitimately changes in
        the same request that changes the URLs.
        """
        n = len(self.configs)
        out: list[str | None] = [u or None for u in (urls or [])]
        if len(out) > n:
            dropped = [u for u in out[n:] if u]
            if dropped:
                log.warning(
                    "camera hub has %d slot(s) but %d URL(s) were given -- "
                    "ignoring %s", n, len(out), dropped,
                )
        return (out + [None] * n)[:n]

    def _normalise_sources(self, sources: "list[Any] | None") -> "list[Any]":
        """One entry per slot, padded or trimmed, SAYING SO when it had to
        -- the same contract `_normalise_urls()` has, and loud for the same
        reason: a silently dropped source is a slot reading hardware while
        the caller believes it is replaying a file.

        Also refuses to let a source and a URL describe the same slot in
        silence. The source wins (it is the more specific instruction, and
        the caller built it deliberately), and the URL it displaces is
        named in the log rather than simply ignored."""
        n = len(self.configs)
        out: list[Any] = list(sources or [])
        if len(out) > n:
            dropped = [s for s in out[n:] if s is not None]
            if dropped:
                log.warning(
                    "camera hub has %d slot(s) but %d source(s) were given -- "
                    "ignoring %d of them", n, len(out), len(dropped),
                )
        out = (out + [None] * n)[:n]
        for i, source in enumerate(out):
            if source is not None and self._slot_urls[i]:
                log.warning(
                    "cam%d: both a source object and a URL (%s) were given -- "
                    "the source wins and the URL is NOT being read",
                    i, self._slot_urls[i],
                )
                self._slot_urls[i] = None
        return out

    @property
    def slot_urls(self) -> "list[str | None]":
        """The stream URL feeding each slot, or None for a local device.

        A property rather than the raw list so a caller cannot mutate this
        hub's routing behind `open_all()`'s back -- the sources are built
        from it at open time, and an edit afterwards would describe a
        rig that does not exist. Read by the dashboard (`_live_slot_urls`)
        to show what this PROCESS is actually reading, which is the thing
        that disagrees with the config file exactly when someone has saved
        a change and not restarted.
        """
        return list(self._slot_urls)

    def reconfigure(self, configs: "list[CameraConfig] | None" = None,
                    urls: "list[str | None] | None" = None,
                    sources: "list[Any] | None" = None) -> None:
        """Replace this hub's camera configs IN PLACE, on the same object.

        Exists so an operator can reassign which hardware device feeds
        which slot -- or point a slot at another machine's stream instead
        of a device -- without restarting the process. Everything derived
        from `configs` -- `status`, `_locks` and the per-slot URLs -- is
        rebuilt here to match; those are the only state that depends on
        them, `_caps`/`_sources` being populated at open time.

        `configs=None` means "keep the slots as they are", which is how a
        pure SOURCE change (local <-> stream, same devices) is spelled.
        `urls=None` likewise means "keep the current routing": a caller
        that means "make every slot local" passes a list of Nones. The
        difference is load-bearing -- the dashboard sends only `devices`
        when a device dropdown moves, and treating that as "make
        everything local" would silently drop a working feed because an
        unrelated slot changed.

        Mutating in place rather than building a new hub is deliberate and
        is what makes a live change possible at all: run_product builds ONE
        hub before either half starts and shares it BY REFERENCE with the
        capture thread and the app, so replacing the object would leave
        both holding the old one. Nothing here replaces the object, so
        every existing reference stays correct.

        REQUIRES A CLOSED HUB, and raises rather than coping if it is not:
        the cameras currently open are the ones `configs` describes, so
        swapping the description out from under them would leave `status`
        and `_caps` disagreeing about what device slot N even is -- the
        exact "status says opened, reads fail" class of bug the 2026-08-12
        status-honesty fix exists to prevent. Callers stop first (POST
        /api/stop) and open again afterwards; open_all() then picks up the
        new devices with no other change.
        """
        with self._caps_lock:
            still_open = sorted(self._caps)
        if still_open:
            raise RuntimeError(
                "cannot reconfigure while cameras are open (slots "
                f"{still_open}) -- stop the capture loop first, then reconfigure"
            )
        # Checked BEFORE the pump, so the refusal names the thing an
        # all-stream rig would actually recognise. A reader thread still
        # running IS the old assignment, exactly as an open capture is,
        # and "the pump thread is still running" would be true but
        # unhelpful on a rig that has no local camera to think about.
        if self._sources:
            raise RuntimeError(
                "cannot reconfigure while stream readers are running (slots "
                f"{sorted(self._sources)}) -- close_all() stops them, call that first"
            )
        if self._pump_thread is not None and self._pump_thread.is_alive():
            raise RuntimeError(
                "cannot reconfigure while the pump thread is still running -- "
                "close_all() joins it, call that first"
            )
        previous = list(self._slot_urls)
        # `sources=None` means "keep them", the same convention `urls=None`
        # already uses and for the same reason: the dashboard sends only
        # `devices` when a device dropdown moves, and treating that as
        # "drop every source" would silently end a replay because an
        # unrelated slot changed. A caller that means "make every slot
        # live again" passes a list of Nones.
        previous_sources = list(self._preset_sources)
        if configs is not None:
            self.configs = list(configs)
        self.status = {
            i: CameraStatus(device=cfg.device) for i, cfg in enumerate(self.configs)
        }
        self._locks = {i: threading.Lock() for i in range(len(self.configs))}
        self._slot_urls = self._normalise_urls(previous if urls is None else urls)
        self._preset_sources = self._normalise_sources(
            previous_sources if sources is None else sources
        )
        log.info(
            "camera hub reconfigured: %d slot(s), devices %s, sources %s",
            len(self.configs), [c.device for c in self.configs],
            [self._slot_source_label(i) for i in range(len(self.configs))],
        )

    def _slot_source_label(self, i: int) -> str:
        """What is actually feeding slot `i`, in one word.

        Used by the reconfigure log line and `status_report()`. Without
        it, a slot replaying a file, a slot reading a stream and a slot
        reading a device are indistinguishable in exactly the log line
        someone reads when one of them is not delivering."""
        if i < len(self._preset_sources) and self._preset_sources[i] is not None:
            return getattr(self._preset_sources[i], "source_label", "source")
        if i < len(self._slot_urls) and self._slot_urls[i]:
            return str(self._slot_urls[i])
        return "local"

    def set_frame_sink(self, frame_sink) -> None:
        """Attach or detach the per-cycle frame callback at runtime.

        Exists so the oracle toggle can start and stop virtual-camera
        publishing live. Publishing copies several megabytes per camera
        per frame, so leaving it running while the oracle is switched off
        is real work for nobody -- and requiring a restart to stop it
        would make the toggle a lie.

        Plain assignment with no lock: the pump reads this once per cycle
        and a torn read is not possible for a single reference. The worst
        case is one cycle using the previous value, which for a diagnostic
        is not worth a lock on the capture path.
        """
        self._sink_entry = (frame_sink, _takes_jpegs(frame_sink))
        self._frame_sink = frame_sink
        self.frame_sink_errors = 0      # a fresh sink deserves a fresh count

    def set_small_decode(self, enabled: bool, scale: int = SMALL_DECODE_SCALE) -> None:
        """Turn DETECTION DECODES SMALL on or off (config key
        detect_from_small_decode). Set before open_all(); a change while the
        pump runs takes effect on its next cycle, and every consumer copes
        with either kind of frame in the cache."""
        self._small_decode_scale = int(scale) if enabled else None

    @property
    def small_decode(self) -> bool:
        """Whether JPEG slots are published as LazyFrames. The capture
        loop then fetches with grab_frames() rather than grab_all()."""
        return self._small_decode_scale is not None

    def set_frame_ring(self, frame_ring) -> None:
        """Attach or detach the throw-capture ring at runtime.

        Exists because the ring's whole cost is memory -- gigabytes of it
        -- and an operator who has decided not to spend that must be able
        to stop spending it without restarting a session. Requiring a
        restart to free 4GB would make the setting a lie in the same way
        the oracle toggle would be if publishing kept running.

        Plain assignment with no lock, matching `set_frame_sink()`: the
        pump reads this once per cycle, a single reference cannot tear,
        and the worst case is one cycle using the previous value. Passing
        None detaches; the ring's own retained frames are NOT dropped here
        (that is `FrameRing.clear()`'s job) because detaching mid-capture
        must not destroy evidence someone is in the middle of writing.
        """
        self._frame_ring = frame_ring
        self.frame_ring_errors = 0      # a fresh ring deserves a fresh count

    @property
    def frame_ring(self):
        """The attached ring, or None. A property so a caller (the
        dashboard's `/api/frame-ring`, a trigger) reaches the SAME object
        the pump is filling rather than a copy -- there is exactly one
        ring per hub, and a second one would buffer nothing."""
        return self._frame_ring

    def open_all(self) -> list[bool]:
        """Open every configured SLOT CONCURRENTLY (one short-lived
        thread per camera), then start the single dedicated pump thread
        that will do all future cap.read() calls for this hub's lifetime
        (see module docstring's "ARCHITECTURE CHANGE" section for the pump
        itself, and its 2026-08-12 "CONCURRENT OPEN" section for why
        opening in parallel here is safe). Returns per-camera opened flags,
        in config order -- `ThreadPoolExecutor.map()` yields results in
        input order regardless of which camera happens to finish opening
        first, so callers can still index this list by camera position.
        Safe to call again after close_all() (e.g. to retry a camera that
        previously failed) -- close_all() at the top tears down any
        previous pump before this one opens fresh captures.

        2026-08-12, CONCURRENT OPEN: this used to open cameras ONE AT A
        TIME (`[self._open_one(i, cfg) for i, cfg in enumerate(...)]`) --
        real measured `open_latency_s` on this rig is ~2.2-2.3s per camera,
        so 3 cameras cost ~6.75s sequentially, before calibration even
        started. Each camera's own `_open_one()`/`_open_one_locked()`
        already only ever touches THAT camera's own `self._locks[i]` slot,
        `self.status[i]` object, and `self._caps[i]`/`self._last_frames[i]`
        dict entries -- opening camera A was never actually dependent on
        camera B's open finishing first, so the sequential loop was
        serializing independent work for no correctness reason. See
        `self._caps_lock`'s own docstring (in `__init__`) for the one real
        cross-camera hazard this introduces (concurrent dict writes) and
        why it's handled. `tests/test_local_capture.py` proves, with a
        mocked per-camera open delay, that N-camera `open_all()` now costs
        roughly `max(delays)` rather than `sum(delays)`.

        A SLOT WITH A URL NEVER TOUCHES cv2 HERE. `_open_one()` is called
        only for the local slots; a stream slot gets a reader thread
        instead. That is what makes an all-stream rig open zero devices --
        structurally, rather than by a guard that has to remember to fire.
        Its flag is True, which is NOT a claim that the stream is live: it
        cannot be, because connecting is asynchronous and the publisher
        may not be started yet. A reader retries quietly forever, and the
        honest liveness signal is the same one a local camera uses --
        `status[i].last_read_ok` and `frame_count` advancing. Reporting a
        speculative False would make a rig that is merely waiting look
        broken, and `any(ok_flags)` is what decides whether the pump runs
        at all."""
        self.close_all()
        n_slots = len(self.configs)
        # Three-way, in precedence order, and each list is built from the
        # one condition that defines it rather than by elimination -- a
        # slot classified by "not either of the other two" is how a fourth
        # kind of source would silently become a local camera.
        preset_slots = [i for i in range(n_slots) if self._preset_sources[i] is not None]
        stream_slots = [i for i in range(n_slots)
                        if self._preset_sources[i] is None and self._slot_urls[i]]
        local_slots = [i for i in range(n_slots)
                       if self._preset_sources[i] is None and not self._slot_urls[i]]
        log.info(
            "opening %d slot(s): %d local camera(s) concurrently (devices=%s), "
            "%d stream(s) (%s), %d pre-built source(s) (%s)",
            n_slots, len(local_slots),
            [self.configs[i].device for i in local_slots],
            len(stream_slots), [self._slot_urls[i] for i in stream_slots] or "none",
            len(preset_slots),
            [self._slot_source_label(i) for i in preset_slots] or "none",
        )
        # Reset before _open_one() runs -- _open_one_locked() seeds this
        # with each camera's warm frame as it opens (see module
        # docstring's "CACHE FRESHNESS at first-grab time" section).
        self._last_frames = {i: None for i in range(len(self.configs))}
        self._last_jpegs = {}
        self._slot_generation = {}
        self._raw_slots = set()
        ok_flags = [False] * len(self.configs)
        for i in preset_slots:
            ok_flags[i] = self._start_preset_source(i)
        for i in stream_slots:
            ok_flags[i] = self._open_stream(i)
        if local_slots:
            # Short-lived, scoped to this call only -- distinct from
            # self._pool, the pump's own persistent one (started below,
            # after every camera has finished opening). One worker per
            # camera, same sizing convention as self._pool.
            with ThreadPoolExecutor(
                max_workers=len(local_slots), thread_name_prefix="local-cam-open"
            ) as open_pool:
                for i, ok in zip(local_slots, open_pool.map(
                    self._open_one, local_slots,
                    [self.configs[i] for i in local_slots],
                )):
                    ok_flags[i] = ok
        n_ok = sum(1 for i in local_slots if ok_flags[i])
        if n_ok < len(local_slots):
            log.warning(
                "opened %d/%d cameras -- see per-camera log lines above for failures",
                n_ok,
                len(local_slots),
            )
        else:
            log.info("opened %d/%d cameras ok", n_ok, len(local_slots))

        # Start the pump AFTER every camera has been opened (concurrently,
        # above) -- the proven open_all() ordering (the
        # pump still only ever starts once ALL cameras have finished their
        # own open attempt, success or failure; only the opening itself is
        # now parallel across cameras, not the pump-start ordering).
        # Only start it if at least one camera actually opened; a hub
        # with zero working cameras has nothing to pump and no caller
        # should be relying on a pump thread existing in that case.
        self._pump_stop.clear()
        if any(ok_flags):
            self._pool = ThreadPoolExecutor(
                max_workers=max(1, len(self.configs)),
                thread_name_prefix="local-cam-pump-worker",
            )
            self._pump_thread = threading.Thread(
                target=self._pump_loop, name="local-cam-pump", daemon=True
            )
            self._pump_thread.start()
        return ok_flags

    def _start_preset_source(self, i: int) -> bool:
        """Start slot `i`'s caller-supplied source. Never opens a local
        device and never touches the network -- what the source does is
        the source's business; this only starts it and files it under the
        one slot key everything else in this class uses.

        Returns True the way a stream slot does, and with the same
        caveat: it is not a claim that frames are flowing, which cannot be
        known synchronously. `status[i].last_read_ok` and `frame_count`
        advancing are the honest liveness signal, exactly as for the other
        two kinds.
        """
        source = self._preset_sources[i]
        status = self.status[i]
        status.closed_at = None
        status.requested_width = self.configs[i].width or 0
        status.requested_height = self.configs[i].height or 0
        status.requested_fps = self.configs[i].fps
        status.backend_used = getattr(source, "source_label", "source")
        # The source is told which slot's status it owns the connection
        # half of, the same split StreamSource documents: the source owns
        # `opened`/`last_error`/the latency fields, the pump owns the
        # frame counters. A source that does not want the seam can ignore
        # the call -- it is a capability check, not a name check.
        attach = getattr(source, "attach_status", None)
        if callable(attach):
            attach(status)
        self._sources[i] = source
        try:
            source.start()
        except Exception as exc: # noqa: BLE001 -- one bad source must never stop the rest opening
            log.exception("cam%d: source failed to start -- %s", i, exc)
            status.opened = False
            status.last_error = f"source failed to start: {type(exc).__name__}: {exc}"
            del self._sources[i]
            return False
        log.info("cam%d: reading a pre-built source -- %s", i, self._slot_source_label(i))
        return True

    def _open_stream(self, i: int) -> bool:
        """Start slot `i`'s MJPEG reader. Never opens a local device.

        The import is deliberately local. `remote_capture` imports this
        module for CameraConfig/CameraStatus, so a module-level import
        here would be a cycle -- and this module is the one that has to
        stay importable on a rig with no networking interest at all.
        """
        from opendarts.live.remote_capture import StreamSource

        status = self.status[i]
        status.closed_at = None
        status.requested_width = self.configs[i].width or 0
        status.requested_height = self.configs[i].height or 0
        status.requested_fps = self.configs[i].fps
        source = StreamSource(i, str(self._slot_urls[i]), status)
        self._sources[i] = source
        source.start()
        log.info("cam%d: reading a stream -- %s", i, self._slot_urls[i])
        return True

    def _open_one(self, i: int, cfg: CameraConfig) -> bool:
        # Held for the whole open/configure/warm-frame sequence -- see
        # __init__'s own comment for why this is a per-camera lock, and
        # the module docstring's "REAL CONCURRENCY BUG" section for why
        # this exists at all. In practice nothing else touches camera `i`
        # this early (open_all() runs before any other thread has a
        # reference to this hub in every real caller -- capture_daemon.py
        # and run_product.py both open the hub before starting any
        # background thread), but locking here too costs nothing and
        # keeps the invariant unconditional rather than "safe only
        # because of caller discipline."
        with self._locks[i]:
            return self._open_one_locked(i, cfg)

    def _open_one_locked(self, i: int, cfg: CameraConfig) -> bool:
        """The real open/configure/warm-frame body -- always called with
        `self._locks[i]` already held by `_open_one()` above; split out
        only so that wrapper stays a trivial, obviously-correct
        one-liner."""
        status = self.status[i]
        # 0 here is a real, honest "not yet known" sentinel for 'auto'
        # mode -- the branch below overwrites both with the real probed
        # values (what was ultimately, actually requested from the
        # driver) the moment probing resolves them, before cap.set() is
        # ever called for width/height. A fixed (non-auto) cfg reports
        # its real requested width/height immediately, exactly as before
        # this change.
        status.requested_width = cfg.width or 0
        status.requested_height = cfg.height or 0
        status.requested_fps = cfg.fps
        # STATUS-HONESTY FIX, 2026-08-12 (see CameraStatus.closed_at's own
        # docstring + close_all()'s): a fresh open attempt -- whether it
        # ends up succeeding or failing below -- means this camera is no
        # longer in the "was open, then deliberately closed" state that
        # field exists to describe. Clearing it here (unconditionally, at
        # the very start of every open attempt) is the mirror-image of
        # close_all()'s own fix: without this, reopening a previously-
        # closed camera would leave a stale closed_at timestamp sitting
        # right next to opened=True, the same class of bug in the other
        # direction -- a status field a caller could reasonably read as
        # "this is currently closed" even while it's actually open again.
        status.closed_at = None

        if not cfg.enabled:
            status.opened = False
            status.last_error = "disabled in config"
            log.info("cam%d (device=%s): disabled in config, skipping", i, cfg.device)
            return False

        # Chosen by the OS we are ACTUALLY on, not by hasattr. Every CAP_*
        # name is a plain integer constant compiled into the cv2 bindings on
        # every platform -- `hasattr(cv2, "CAP_AVFOUNDATION")` is True on
        # Windows and Linux too -- so the old guards were no-ops that made
        # this list identical everywhere. On Windows that meant two
        # guaranteed-failing opens per camera before falling through to
        # CAP_ANY.
        #
        # CAP_ANY is worse than slow there: it picks between MSMF and DSHOW
        # by OpenCV's own priority order (tunable by OPENCV_VIDEOIO_PRIORITY_*
        # env vars we do not set), and the two differ in a way that matters
        # to this product. MSMF goes through the Windows Camera Frame Server,
        # which lets a SECOND process read the same camera; DSHOW takes the
        # device exclusively. Running a second process on the same cameras
        # at once lives or dies on that difference, so the
        # backend must never be left to chance. MSMF is named explicitly and
        # DSHOW is deliberately NOT in the Windows list: falling back to it
        # would "work" while silently making concurrent capture impossible.
        system = platform.system()
        backends: "list[int | str]" = []
        if system == "Darwin":
            backends.append(cv2.CAP_AVFOUNDATION)
        elif system == "Linux":
            backends.append(cv2.CAP_V4L2)
        elif system == "Windows":
            # WINDOWS, 2026-09-17: our own Media Foundation reader, then
            # DirectShow. No OpenCV MSMF (the headless package has none) and
            # no CAP_ANY.
            #
            # Media Foundation first, for the camera's own JPEG (see the
            # module-level JPEG PASSTHROUGH section). A slot's device number
            # counts in Media Foundation's order. It needs a fixed size to
            # pick the native type, so an 'auto' slot starts at DirectShow.
            #
            # DirectShow as the fallback: pixels only, and it takes the
            # camera exclusively. That used to rule it out, back when
            # other software read the physical cameras; it reads our
            # virtual cameras now. DirectShow numbers devices differently (it also
            # lists virtual cameras, ours included), so it is opened at the
            # index whose DEVICE PATH matches -- never at the same number,
            # which would open a different camera while looking fine. CAP_ANY
            # is left out for the same reason: on Windows it is DirectShow at
            # the unmatched number.
            if _raw_jpeg_wanted() and not cfg.auto_resolution:
                backends.append(MF_JPEG_BACKEND)
            backends.append(cv2.CAP_DSHOW)
        if system != "Windows":
            # Last resort elsewhere, including an unrecognised platform: let
            # OpenCV choose rather than refuse to open at all. `backend_used`
            # in CameraStatus records what actually served the camera, so a
            # machine that lands here is visible rather than mysterious.
            backends.append(cv2.CAP_ANY)

        t_start = time.monotonic()
        tried: list[str] = []
        for backend in backends:
            is_mf = backend == MF_JPEG_BACKEND
            backend_name = MF_JPEG_BACKEND if is_mf else _backend_name(backend)
            tried.append(backend_name)
            log.debug("cam%d (device=%s): trying backend %s", i, cfg.device, backend_name)
            if is_mf:
                try:
                    cap = _open_mf_jpeg(cfg.device)
                except Exception as exc:  # noqa: BLE001 -- fall through to DirectShow
                    log.info("cam%d (device=%s): Media Foundation JPEG capture "
                             "unavailable (%s) -- trying DirectShow", i, cfg.device, exc)
                    continue
            elif system == "Windows" and backend == cv2.CAP_DSHOW:
                dshow_index = _dshow_index_for(cfg.device)
                if dshow_index is None:
                    log.warning("cam%d (device=%s): no DirectShow device matches this "
                                "camera's path -- not opening it through DirectShow",
                                i, cfg.device)
                    tried[-1] = f"{backend_name} (no matching device)"
                    continue
                log.info("cam%d (device=%s): DirectShow fallback at DirectShow index %d",
                         i, cfg.device, dshow_index)
                cap = cv2.VideoCapture(dshow_index, backend)
            else:
                cap = cv2.VideoCapture(cfg.device, backend)
            if not cap.isOpened():
                cap.release()
                log.debug(
                    "cam%d (device=%s): backend %s failed to open", i, cfg.device, backend_name
                )
                continue

            # MJPG ON V4L2 ONLY, 2026-09-15. Measured on the Linux rig:
            # without this the driver hands back its default YUYV and caps
            # at 1280x720@10fps, because uncompressed 720p is ~27 MB/s and
            # USB 2.0 will not sustain it faster. Same camera, same call,
            # with MJPG requested: 1280x720@30fps. A third of the frame
            # rate is not a tuning detail, and the 10fps reads as "these
            # cameras are slow" rather than "nobody asked for MJPEG".
            #
            # Asked on EVERY backend, not just V4L2. MJPEG is what these
            # cameras produce natively on all three platforms, and asking
            # explicitly is the difference between a negotiated format and
            # whichever default the backend happens to pick -- V4L2 picks
            # uncompressed, and a default nobody stated is a divergence
            # waiting to be discovered on the next platform.
            #
            # Set BEFORE width/height on purpose: V4L2 applies the pixel
            # format first and silently ignores a fourcc that arrives
            # after the geometry. The other backends do not care about the
            # ordering, so the strictest rule is the one to follow.
            #
            # A refused request is not fatal anywhere -- `set()` returning
            # False leaves the driver default in place, which is exactly
            # the pre-2026-09-15 behaviour.
            try:
                if not cap.set(cv2.CAP_PROP_FOURCC,
                               cv2.VideoWriter_fourcc(*"MJPG")):
                    log.debug("cam%d (device=%s): MJPG not accepted by %s "
                              "-- using the driver default",
                              i, cfg.device, backend_name)
            except Exception:  # noqa: BLE001 -- a refused hint is not fatal
                log.debug("cam%d (device=%s): MJPG request raised on %s",
                          i, cfg.device, backend_name)

            # Request preferred mode, then read back what the driver
            # actually gave us. Deliberately do NOT set buffer size -- the
            # proven capture path does not, and matching that removes a
            # divergence with no evidence it was ever needed.
            #
            # RESOLUTION PROBING, added 2026-08-20 -- see CameraConfig's
            # own docstring for the full opt-in contract. `cfg.width`/
            # `cfg.height` being real ints (today's default, and every
            # real call site's current behavior) takes this exact same
            # branch it always has: one direct cap.set() pair, no probing,
            # byte-identical to before this change. Only an explicit
            # `CameraConfig(width=None, height=None)` (opt-in, never the
            # default) takes the probe branch instead.
            if cfg.auto_resolution:
                # VERIFIER FINDING, 2026-08-20 -- this probe call touches
                # the real cv2.VideoCapture (up to 7x2 set()+get() round
                # trips) with no exception guard, unlike every other
                # camera-touching call in this class (_read_one() guards
                # cap.read(), _release_one_locked() guards cap.release(),
                # _pump_loop() guards the whole pump cycle -- "one bad
                # camera must never kill the rest" is this file's own
                # established discipline). An uncaught exception here
                # would propagate out of _open_one_locked() -> _open_one()
                # -> the list(open_pool.map(...)) call in open_all(),
                # crashing open_all() for EVERY configured camera, not
                # just this one -- a real regression against that
                # discipline the moment 'auto' mode is ever reachable
                # from a live call site (not yet -- see this module's own
                # docstring -- but this guard costs nothing and removes
                # the gap before that wiring lands).
                try:
                    probed = camera_resolution.highest_supported_resolution(cap)
                except Exception: # noqa: BLE001 -- one camera's probing must never kill open_all()
                    log.exception(
                        "cam%d (device=%s): exception while probing resolutions -- "
                        "falling back to the fixed %dx%d default",
                        i, cfg.device, DEFAULT_WIDTH, DEFAULT_HEIGHT,
                    )
                    probed = None
                if probed is None:
                    log.warning(
                        "cam%d (device=%s): 'auto' resolution requested but nothing "
                        "genuinely usable was found (either nothing among the common "
                        "candidates was honored, or probing itself raised -- see any "
                        "preceding log line) -- falling back to the fixed %dx%d default",
                        i, cfg.device, DEFAULT_WIDTH, DEFAULT_HEIGHT,
                    )
                    probed = (DEFAULT_WIDTH, DEFAULT_HEIGHT)
                width, height = probed
                log.info(
                    "cam%d (device=%s): 'auto' resolution probed -- requesting the "
                    "highest genuinely-supported mode found: %dx%d",
                    i, cfg.device, width, height,
                )
                status.requested_width = width
                status.requested_height = height
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            else:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            cap.set(cv2.CAP_PROP_FPS, cfg.fps)
            if backend == cv2.CAP_DSHOW:
                # DSHOW applies the pixel format only when it comes AFTER the
                # geometry (measured on a Windows rig, 2026-09-17) -- the reverse of V4L2.
                # Without MJPG, three 720p cameras exceed USB 2.0 bandwidth.
                try:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                except Exception:  # noqa: BLE001 -- a refused hint is not fatal
                    pass

            open_latency = time.monotonic() - t_start

            # Warm one frame so negotiated size is real (some backends
            # lie until the first read). Deliberately still a plain
            # synchronous cap.read() here, NOT routed through the pump --
            # the pump doesn't exist yet at this point (cameras open
            # concurrently with EACH OTHER as of 2026-08-12, but each
            # camera's own open+configure+warm-read sequence below is
            # still single-threaded/synchronous, and open_all() doesn't
            # start any pump thread until every camera's own sequence has
            # finished), and this is a genuinely different, once-only code
            # path from the hot-path contention the pump exists to fix. See module
            # docstring's "CACHE FRESHNESS at first-grab time" section for
            # why this frame is ALSO seeded into the pump's own cache
            # below -- so a grab() called right after open_all() returns
            # doesn't have to wait for the pump's first cycle.
            #
            # Read BEFORE the slot is registered, because this frame is
            # also what decides JPEG passthrough: a Media Foundation open
            # that cannot deliver a decodable JPEG is abandoned for
            # CAP_MSMF, and a V4L2 one goes back to decoded reads.
            raw = is_mf
            if backend == cv2.CAP_V4L2 and _raw_jpeg_wanted():
                try:
                    raw = bool(cap.set(cv2.CAP_PROP_CONVERT_RGB, 0))
                except Exception:  # noqa: BLE001 -- a refused hint is not fatal
                    raw = False
            t_frame_start = time.monotonic()
            ok, frame = cap.read()
            jpeg: "bytes | None" = None
            if raw:
                kind, decoded, jpeg = _raw_to_frame(frame if ok else None)
                if kind == "jpeg":
                    frame = decoded
                elif is_mf:
                    log.info(
                        "cam%d (device=%s): Media Foundation gave no usable JPEG "
                        "(%s) -- trying CAP_MSMF", i, cfg.device,
                        getattr(cap, "error", None)
                        or (f"{kind}, {getattr(frame, 'size', 0)} bytes" if ok
                            else f"{kind}, read failed"),
                    )
                    cap.release()
                    continue
                else:
                    raw = False
                    try:
                        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    except Exception:  # noqa: BLE001
                        pass
                    if kind == "pixels":
                        # The backend ignored the request: this frame is
                        # already the decoded one.
                        log.debug("cam%d (device=%s): %s ignored CONVERT_RGB=0",
                                  i, cfg.device, backend_name)
                    else:
                        log.info(
                            "cam%d (device=%s): no usable JPEG with conversion "
                            "off (%s) -- decoding in the driver instead",
                            i, cfg.device, kind,
                        )
                        t_frame_start = time.monotonic()
                        ok, frame = cap.read()
            status.jpeg_passthrough = raw
            status.jpeg_rejected = 0
            if raw:
                self._raw_slots.add(i)
            else:
                self._raw_slots.discard(i)

            # self._caps_lock, not self._locks[i] -- see that lock's own
            # docstring in __init__: this guards the SHARED dict structure
            # across concurrently-opening cameras, a different hazard than
            # self._locks[i]'s per-camera cv2.VideoCapture serialization.
            with self._caps_lock:
                self._caps[i] = cap
            status.opened = True
            status.backend_used = backend_name
            status.open_latency_s = open_latency
            status.last_error = None
            # Logged per camera, at INFO, naming the backend that served it.
            # status_report() already carries these numbers, but a slow open
            # is a "why did Start take 45 seconds" question asked from the
            # log after the fact, and answering it should not require
            # knowing to go read a status endpoint at the time. Backend is
            # in the same line because on Windows the answer to "why slow"
            # and the answer to "which backend" are usually the same fact.
            log.info(
                "cam%d (device=%s): opened via %s in %.3fs",
                i, cfg.device, backend_name, open_latency,
            )

            if ok and frame is not None:
                h, w = frame.shape[:2]
                status.actual_width = int(w)
                status.actual_height = int(h)
                status.first_frame_latency_s = time.monotonic() - t_frame_start
                status.frame_count = 1
                status.last_read_ok = True
                status.last_read_at = time.time()
                status.last_read_at_monotonic = time.monotonic()
                # The first frame goes through the SAME synthetic-JPEG step the
                # pump applies to every later one. Skipping it here published
                # the first frame after each open as raw pixels -- scored, but
                # not the decode of any bytes we could store.
                if jpeg is None:
                    synthesised = _synthesise_jpeg(frame)
                    if synthesised is not None:
                        frame, jpeg = synthesised
                        raw = True      # a fresh array from imdecode, no copy needed
                        status.jpeg_synthetic = True
                        status.jpeg_synthetic_quality = SYNTHETIC_JPEG_QUALITY
                if jpeg is not None and self._small_decode_scale is not None:
                    # Same kind of frame the pump will publish next, so the
                    # lifecycle never judges one slot by two different
                    # small pictures; its pixels are already decoded.
                    frame = _lazy_from_jpeg(jpeg, self._small_decode_scale, frame) or frame
                with self._cache_lock:
                    # A decoded JPEG is already a fresh array.
                    self._last_frames[i] = frame if raw else frame.copy()
                    # Not a pump-published frame, so no ring set holds it.
                    self._slot_generation.pop(i, None)
                    if jpeg is not None:
                        self._last_jpegs[i] = jpeg
                    else:
                        # Never leave a previous open's bytes beside this
                        # open's frame: a consumer pairing them by slot would
                        # store bytes that do not decode to what was scored.
                        # The pump already does exactly this on every cycle.
                        self._last_jpegs.pop(i, None)
            else:
                status.actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                status.actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                status.last_read_ok = False
                log.warning(
                    "cam%d (device=%s): opened via %s but first frame read failed",
                    i,
                    cfg.device,
                    backend_name,
                )
            status.actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            _raw_fourcc = cap.get(cv2.CAP_PROP_FOURCC)
            status.actual_fourcc = _fourcc_to_str(_raw_fourcc)
            if status.actual_fourcc is None:
                # Distinguishes "this backend does not implement the
                # property" (raw 0.0, which MSMF is known to do) from "it
                # returned something _fourcc_to_str refused to decode".
                # Without the raw number the two look identical from
                # here, and they call for completely different next
                # steps -- probe via another backend, versus fix the
                # decoder.
                log.debug(
                    "cam%d (device=%s): backend %s reported no usable FOURCC "
                    "(raw=%r)", i, cfg.device, backend_name, _raw_fourcc,
                )

            log.info("cam%d (device=%s): opened ok -- %s", i, cfg.device, status.summary())
            return True

        status.opened = False
        status.last_error = f"all backends failed (tried: {tried})"
        status.open_latency_s = time.monotonic() - t_start
        log.error("cam%d (device=%s): FAILED to open -- %s", i, cfg.device, status.last_error)
        return False

    def grab(self, i: int) -> np.ndarray | None:
        """Return camera i's most recently PUMPED frame -- a pure cache
        read, no `cap.read()` call happens here at all anymore. See
        module docstring's "ARCHITECTURE CHANGE" section: this used to
        call `cap.read()` directly, on whichever OS thread called
        grab() itself; it no longer touches `cv2.VideoCapture` in any
        way.

        Returns None if camera `i` was never opened, or opened but has
        not yet produced any successful read (either the open-time warm
        frame or a pump cycle) -- status[i].last_error explains why, but
        note the SEMANTIC SHIFT documented on CameraStatus/in the module
        docstring: last_error/last_read_ok now describe the PUMP's most
        recent attempt for this camera, not this specific grab() call
        (which reads nothing itself, so it has nothing new to report).

        A returned frame may be slightly stale if the pump's most recent
        attempt for this camera failed transiently -- deliberate
        (`_pump_once()` only overwrites the cache on a
        SUCCESSFUL read, so a momentary hiccup does not null out an
        otherwise-good previous frame). Check status[i].last_read_ok if
        a caller specifically needs to know whether the pump's last
        attempt succeeded, as opposed to merely "is there a cached frame
        at all"."""
        with self._cache_lock:
            frame = self._last_frames.get(i)
        # A LazyFrame decodes here, outside the lock, once per frame.
        return pixels_of(frame)

    def grab_with_jpeg(self, i: int) -> "tuple[np.ndarray | None, bytes | None]":
        """Slot `i`'s cached frame and the camera's own JPEG of that same
        frame, read together under the cache lock. The JPEG is None when
        the slot is not in passthrough (see the module-level JPEG
        PASSTHROUGH section) -- a caller that wants JPEG encodes the
        frame itself then, exactly as before."""
        with self._cache_lock:
            frame, jpeg = self._last_frames.get(i), self._last_jpegs.get(i)
        return pixels_of(frame), jpeg

    def grab_jpeg_lazy(self, i: int) -> "tuple[Any, bytes | None]":
        """grab_with_jpeg() WITHOUT decoding: the cached frame as held (a
        LazyFrame or an array) and its JPEG. For a consumer that forwards
        the bytes when it has them and reads pixels only otherwise."""
        with self._cache_lock:
            return self._last_frames.get(i), self._last_jpegs.get(i)

    def grab_paired(self, i: int) -> "tuple[np.ndarray | None, bytes | None, int | None]":
        """grab_with_jpeg() plus the pump generation that published that
        frame -- the frame ring set holding it (see `_slot_generation`).
        All three read under the one cache lock, so they describe the same
        frame. The generation is None for a frame no ring set holds."""
        with self._cache_lock:
            frame, jpeg, generation = (self._last_frames.get(i), self._last_jpegs.get(i),
                                       self._slot_generation.get(i))
        return pixels_of(frame), jpeg, generation

    def grab_all(self) -> dict[int, np.ndarray]:
        """Return every configured camera's most recently pumped frame.
        Cameras with no cached frame yet (never opened, or opened but no
        successful read yet -- see grab()'s docstring) are simply absent
        from the returned dict, matching
        opendarts.live.capture.fetch_all_snapshots' "skip and continue"
        posture -- caller must check len()/keys() to know which cameras
        actually produced a frame. Pure cache read, like grab() -- no
        `cap.read()` call happens here."""
        frames: dict[int, np.ndarray] = {}
        with self._cache_lock:
            for i in range(len(self.configs)):
                frame = self._last_frames.get(i)
                if frame is not None:
                    frames[i] = frame
        # LazyFrames decode here, outside the lock -- a caller of grab_all()
        # asked for pixels. The capture loop uses grab_frames() instead.
        return {i: pixels_of(f) for i, f in frames.items()}

    def grab_frames(self) -> LazyFrames:
        """grab_all() WITHOUT the full decode: every slot's cached frame as
        held (a LazyFrame carrying its JPEG and small grey picture, or an
        array for a pixels-only slot), plus each slot's JPEG and ring
        generation, all read under one lock hold so they describe the same
        frames. Reading a VALUE of the returned mapping decodes that frame;
        see opendarts/capture/lazy_frame.py."""
        handles: "dict[int, Any]" = {}
        with self._cache_lock:
            for i in range(len(self.configs)):
                frame = self._last_frames.get(i)
                if frame is not None:
                    handles[i] = frame
            jpegs = {i: self._last_jpegs[i] for i in handles if i in self._last_jpegs}
            generations = {i: self._slot_generation[i] for i in handles
                           if i in self._slot_generation}
        return LazyFrames(handles, jpegs=jpegs, generations=generations)

    def _read_one(self, i: int) -> "tuple[np.ndarray | None, bytes | None]":
        """`_read_one_raw()`, plus the SYNTHETIC JPEG round trip for a slot
        that came back with pixels but no bytes.

        WHY HERE AND NOT IN THE PUMP. This runs on the slot's own worker
        thread, so the three cameras encode and decode IN PARALLEL. It used
        to run in `_pump_once()` inside the cache lock -- one camera after
        another, with every `grab()` waiting behind it -- which measured as a
        ~10% frame-rate loss on the macOS rig (27.7 -> 24.9 fps). The
        published pixels are still the decode of the kept bytes; only the
        thread doing the work moved.
        """
        frame, jpeg = self._read_one_raw(i)
        # Local cameras only -- a stream or replay source is left exactly as
        # it delivered (see the SYNTHETIC JPEG section).
        if frame is not None and jpeg is None and self._sources.get(i) is None:
            scale = self._small_decode_scale
            synthesised = (_synthesise_jpeg(frame) if scale is None
                           else self._eager(_synthesise_lazy(frame, scale)))
            if synthesised is None:
                # Detach from OpenCV's reusable buffer: _read_one_raw() skipped
                # the copy on the promise that the round trip would replace it.
                return frame.copy(), None
            else:
                frame, jpeg = synthesised
                status = self.status.get(i)
                if status is not None and not status.jpeg_synthetic:
                    status.jpeg_synthetic = True
                    status.jpeg_synthetic_quality = SYNTHETIC_JPEG_QUALITY
        return frame, jpeg

    def _read_one_raw(self, i: int) -> "tuple[np.ndarray | None, bytes | None]":
        """Runs on one of the pump's own ThreadPoolExecutor worker
        threads (see _pump_once() below) -- this is the ONLY place in
        this class that still calls `cap.read()`. Never call this
        directly from grab()/grab_all() or any other consumer.

        THE ONLY PLACE A SLOT'S SOURCE MATTERS. Everything else in this
        class -- the cache, the generation, the sink, the status writes --
        is written once and works the same either way. Both branches
        block until a frame is ready and answer None when one is not, so
        the cycle is paced by the source in both cases (see
        remote_capture.STREAM_FRAME_WAIT_S for why a stream has to block
        rather than hand back whatever it last decoded).
        Returns (frame, jpeg). `jpeg` is the source's own bytes for that
        frame when it has them: a passthrough camera, or a stream (whose
        parts are already JPEG). A source without `read_pair` gives None.
        """
        stream = self._sources.get(i)
        if stream is not None:
            read_pair = getattr(stream, "read_pair", None)
            if read_pair is not None:
                return read_pair()
            return stream.read(), None
        cap = self._caps.get(i)
        if cap is None or not cap.isOpened():
            return None, None
        try:
            ok, frame = cap.read()
        except Exception: # noqa: BLE001 -- one bad camera must never kill the pump
            return None, None
        if not ok:
            return None, None
        if i not in self._raw_slots:
            # Detached from OpenCV's reusable buffer HERE, on this slot's
            # own worker thread, rather than in _pump_once() under the cache
            # lock -- there it ran one camera after another while every
            # grab() waited. No copy is needed at all now: _read_one()
            # replaces this frame with a fresh imdecode() array (SYNTHETIC
            # JPEG), and only if that round trip fails does it fall back to
            # publishing this one -- which it copies first.
            return frame, None
        scale = self._small_decode_scale
        if scale is None:
            kind, decoded, jpeg = _raw_to_frame(frame)
        else:
            kind, decoded, jpeg = _raw_to_lazy(frame, scale)
        if kind != "jpeg":
            status = self.status.get(i)
            if status is not None:
                status.jpeg_rejected += 1
            return None, None
        if scale is not None:
            decoded, jpeg = self._eager((decoded, jpeg))
        return decoded, jpeg

    def _eager(self, pair: "tuple[Any, bytes] | None") -> "tuple[Any, bytes] | None":
        """Decode a small-decode slot's full pixels HERE, on its worker
        thread in parallel with the other cameras -- as the pump always did
        -- when this cycle's frame sink will read them anyway (one that does
        not take LazyFrames). Otherwise leave the frame lazy."""
        if pair is None:
            return None
        sink = self._sink_entry[0]
        if sink is not None and not _takes_lazy(sink):
            pair[0].pixels()
        return pair

    def _stall_backoff_s(self) -> float:
        """How long `_pump_once()` sleeps after a cycle in which NO camera
        produced a frame -- the nominal period of the fastest configured
        camera, so a stalled cycle is never quicker than a delivering one.

        Uses the FASTEST camera's fps (the shortest real period) rather
        than the slowest, so the backoff stays a floor under busy-spin
        without ever delaying the first camera that recovers by more than
        one of its own frame periods. Falls back to DEFAULT_FPS when no
        camera is configured or an fps is nonsensical -- this runs on the
        pump thread on a failing path and must never raise.
        """
        try:
            rates = [c.fps for c in self.configs if c.enabled and c.fps and c.fps > 0]
        except Exception: # noqa: BLE001 -- the pump must never die on a config read
            rates = []
        return 1.0 / float(max(rates)) if rates else 1.0 / float(DEFAULT_FPS)

    def _pump_loop(self) -> None:
        """The hub's single dedicated cap.read() thread for its whole
        lifetime -- see module docstring's "ARCHITECTURE CHANGE"
        section. Runs _pump_once() back-to-back until close_all() sets
        _pump_stop. An unexpected exception inside _pump_once() is
        logged and backed off from, never allowed to kill this thread
        silently."""
        while not self._pump_stop.is_set():
            try:
                self._pump_once()
            except Exception: # noqa: BLE001 -- the pump must never die silently
                log.exception("local-cam-pump: unexpected exception in _pump_once(), backing off")
                time.sleep(0.05)

    def _pump_once(self) -> None:
        """One pump cycle: read every configured camera IN PARALLEL via
        this hub's own ThreadPoolExecutor (one worker per camera,
        the `pool.map(read_one, range(n))` pattern), then
        update the cache and per-camera CameraStatus fields for whichever
        cameras produced a frame. Cameras that were never opened (failed
        backend probe, or disabled in config) are left alone -- their
        status already reflects why, set once by _open_one_locked(), and
        is not touched here."""
        n = len(self.configs)
        if n == 0 or self._pool is None:
            time.sleep(0.05)
            return
        now = time.time()
        now_monotonic = time.monotonic()
        # Sampled in the SAME instant as `now` above (not a second,
        # independently-timed call a few lines later) so the two clocks
        # describe the identical real moment -- see `last_read_at_
        # monotonic`'s own field comment on CameraStatus for why a
        # consumer needs this monotonic twin, not `now` itself. Used
        # ONLY on the successful-read branch below, deliberately NOT
        # mirrored onto the failure branch the way `now`/`last_read_at`
        # already are -- a genuine, intentional semantic difference from
        # `last_read_at`, not an oversight: `last_read_at` already means
        # "the pump's last ATTEMPT" (see grab()'s own docstring), which
        # is right for that field's own purpose (surfacing whether the
        # pump is currently healthy) but wrong for a FRAME-AGE
        # computation -- on a failing pump cycle, `_last_frames[i]`
        # keeps serving the last frame that WAS successfully read
        # (grab()'s own docstring: "a momentary hiccup does not null out
        # an otherwise-good previous frame"), so stamping this field on
        # a failed attempt would make a served frame look FRESHER than
        # it actually is, exactly backwards for a diagnostic whose whole
        # point is catching a frame that's staler than it looks.
        results = list(self._pool.map(self._read_one, range(n)))

        any_live = False
        with self._cache_lock:
            for i, (frame, jpeg) in enumerate(results):
                stream = self._sources.get(i)
                cap = self._caps.get(i)
                status = self.status.get(i)
                if cap is None and stream is None:
                    # Never opened -- nothing this pump cycle can say
                    # about it that _open_one_locked() didn't already.
                    continue
                if frame is None:
                    if status is not None:
                        status.last_read_at = now
                        status.last_read_ok = False
                        # A stream knows something specific about why it
                        # came back empty ("HTTP 503", "stream ended",
                        # "connection refused"); replacing that with this
                        # layer's generic answer would turn a precise
                        # explanation into a shrug.
                        status.last_error = (
                            stream.read_failure_reason() if stream is not None
                            else "no decodable JPEG from the camera"
                            if i in self._raw_slots
                            else "cap.read() returned no frame"
                        )
                    continue
                any_live = True
                # Every frame here is already detached from anything its
                # source will write into again, so a caller holding a
                # grab()'d reference across the NEXT pump cycle still sees
                # the frame it was given, so consumers can hold refs.
                # A local decoded-in-driver frame was copied
                # by _read_one() on its worker thread; a stream, replay or
                # passthrough frame is a fresh array from cv2.imdecode.
                # A slot with no camera JPEG already had its SYNTHETIC JPEG
                # made in _read_one(), on its own worker thread -- never here,
                # inside the cache lock, where it serialised the cameras.
                copied = frame
                self._last_frames[i] = copied
                # The generation this cycle is about to publish (the bump
                # below, under this same lock hold).
                self._slot_generation[i] = self._frame_generation + 1
                if jpeg is not None:
                    self._last_jpegs[i] = jpeg
                else:
                    self._last_jpegs.pop(i, None)
                if status is not None:
                    status.last_read_at = now
                    status.last_read_at_monotonic = now_monotonic
                    status.last_read_ok = True
                    status.frame_count += 1
                    _update_effective_fps(status, now_monotonic)
                    h, w = copied.shape[:2]
                    status.actual_width = int(w)
                    status.actual_height = int(h)

            # FRAME-DRIVEN WAKE, 2026-09-05 -- bump once per COMPLETED
            # pump cycle, regardless of `any_live` (see module docstring's
            # "FRAME-DRIVEN WAKE PRIMITIVE" section): a waiter cares about
            # "did the pump do another cycle of work" so it can re-check
            # its own per-camera frame_count-based staleness logic sooner
            # rather than sitting on a fixed timer -- whether that cycle
            # happened to succeed on every camera is a SEPARATE, unchanged
            # question (still answered by CameraStatus.frame_count per
            # camera, untouched by this counter). Still inside `with
            # self._cache_lock:` -- the mutation and the notify must be
            # atomic with respect to a waiter's own check-then-wait, or
            # the classic lost-wakeup race this design exists to avoid
            # would simply move here instead.
            self._frame_generation += 1
            self._cache_lock.notify_all()
            # SLOTS WITH A FRAME ONLY -- the same "absent, not None"
            # contract grab_all() has always had. This used to be a plain
            # dict() copy, which on a local rig meant a camera that failed
            # to open was published as None on every cycle; the publishers
            # each grew their own `if frame is None: return False` to cope.
            # Harmless there because a failed camera is rare and permanent.
            # Not harmless once a slot can be a STREAM: a stream is None
            # for its whole connect time and for every second it is down,
            # so an all-stream rig with nothing connected called the sink
            # ~30 times a second with a dict of three Nones, and a test
            # asserting "every slot present" passed on a set that carried
            # no pixels at all. Found by running the real publisher over a
            # real socket, not by the suite.
            #
            # BUILT ONCE FOR BOTH CONSUMERS. The ring taps the same point
            # the sink does -- one cycle, one complete frame set, the
            # generation that was just published -- so building it twice
            # would be two dict comprehensions over the same data and,
            # worse, two chances for the two to disagree about what "this
            # cycle's set" means.
            ring = self._frame_ring
            published = (
                {i: f for i, f in self._last_frames.items() if f is not None}
                if (self._frame_sink is not None or ring is not None) else None
            )
            # The camera JPEGs for exactly those frames. Only slots in
            # `published` can appear, so a consumer never gets bytes for a
            # frame it was not handed.
            published_jpegs = (
                {i: self._last_jpegs[i] for i in published if i in self._last_jpegs}
                if published else {}
            )
            generation = self._frame_generation

        # Outside the lock on purpose: publishing copies several megabytes
        # per camera, and holding the cache lock across that would block
        # every reader for its duration.
        sink, sink_takes_jpegs = self._sink_entry
        if sink is not None and published:
            try:
                # A sink that does not take LazyFrames gets pixels; _eager()
                # already decoded them on the workers, so this costs nothing.
                to_sink = (published if _takes_lazy(sink)
                           else {i: pixels_of(f) for i, f in published.items()})
                if sink_takes_jpegs:
                    sink(to_sink, jpegs=published_jpegs)
                else:
                    sink(to_sink)
            except Exception: # noqa: BLE001 -- a sink must never break capture
                self.frame_sink_errors += 1
                if self.frame_sink_errors == 1:
                    log.exception("frame sink raised -- continuing without it")

        # THE RING TAP. Also outside the lock, for the same reason -- and
        # by REFERENCE, which is the whole reason this feature is
        # affordable: every array in `published` was already allocated by
        # this cycle (a local camera's was `.copy()`d off OpenCV's
        # reusable buffer a few lines above, a source's is a fresh array
        # it will never touch again) and would be freed on the next cycle
        # when `_last_frames[i]` is reassigned. Retaining it defers that
        # free; it does not add an allocation, a copy or a memcpy. See
        # opendarts/capture/frame_ring.py's own docstring for the measured
        # numbers behind that claim and what it costs in memory.
        #
        # Both clock samples are the pump's OWN `now`/`now_monotonic`,
        # taken in the same instant at the top of this method -- not two
        # fresh calls here, which would describe a third instant and make
        # the ring's ordering disagree with the CameraStatus fields
        # written from the same pair.
        if ring is not None and published:
            try:
                ring.append(
                    published,
                    wall_s=now,
                    monotonic_s=now_monotonic,
                    generation=generation,
                    jpegs=published_jpegs,
                )
            except Exception: # noqa: BLE001 -- a diagnostic must never break capture
                self.frame_ring_errors += 1
                if self.frame_ring_errors == 1:
                    log.exception("frame ring raised -- continuing without it")

        if not any_live:
            # Avoid busy-spinning the pump when every camera is
            # mid-reopen/failing -- a real camera's cap.read() otherwise
            # blocks naturally until the next frame is ready, so this
            # sleep only matters in the all-failing case.
            #
            # BACKOFF IS THE CAMERA'S OWN FRAME PERIOD, not a flat 20ms,
            # 2026-09-13 (CPU task). The flat 0.02 cycled a fully-stalled
            # hub at ~50Hz while a HEALTHY hub cycles at the camera's
            # ~30fps -- so a stall made this thread cycle FASTER than
            # success, and because `_frame_generation` bumps once per
            # COMPLETED cycle regardless of `any_live` (see the bump's own
            # comment above), every one of those cycles also woke the
            # frame-driven consumer in capture_daemon. Measured live on
            # the Windows rig with all three cameras stalled: the capture
            # loop ran 499 full iterations in 10.66s (46.8/s) against its
            # configured 20Hz, each doing real lifecycle work and writing
            # a WARNING line. Degradation cost more CPU than health, which
            # is backwards. Sleeping the nominal frame period instead
            # makes a stalled cycle no faster than a delivering one.
            time.sleep(self._stall_backoff_s())

    def close_all(self) -> None:
        """Stop the pump thread FIRST (signal + join with a bounded
        timeout), then shut down its ThreadPoolExecutor, and only THEN
        release every camera's cv2.VideoCapture -- this ordering
        ("retire the readers before the captures they hold") is what
        guarantees no
        release() ever races an in-flight cap.read(): by the time any
        capture is released, nothing can still be reading it. The
        per-camera `self._locks` held around each release() below is a
        defensive residual from the earlier per-camera-lock fix (see
        module docstring) -- with the pump already stopped there should
        be no real contention left to guard against, but it costs
        nothing to keep. Idempotent: safe to call again on an
        already-closed (or never-opened) hub.

        STATUS-HONESTY FIX, 2026-08-12 -- see module docstring's dated
        incident writeup (a real /api/stop confirmed the idle-timeout had
        auto-closed the hub ~88 minutes earlier, but GET
        /api/cameras/status kept reporting `opened: true, last_read_ok:
        true` for all 3 cameras the entire time, with last_read_at frozen
        at the moment the pump died -- a live snapshot fetch during that
        window failed outright, directly contradicting the status
        endpoint). Root cause: this method released every capture and
        stopped the pump but never touched `self.status` at all, so each
        camera's CameraStatus just kept reporting whatever the pump last
        wrote before dying -- true at the time, silently false forever
        after. Fixed by explicitly writing each closed camera's now-true
        state here, the same "closing something must WRITE its new state,
        not rely on the absence of further updates to imply it" principle
        applied to every other status surface audited alongside this fix.
        `opened=False` and `last_read_ok=False` are set for every camera
        this hub actually had a live capture for (i.e. every key in
        `self._caps` -- a camera that never opened in the first place is
        left alone, its status already honestly reflects why via
        `_open_one_locked()`). `closed_at` is stamped with the real close
        time so a caller can tell exactly how stale any leftover
        last_read_at/frame_count/actual_* fields are, rather than having
        to guess -- see CameraStatus.closed_at's own docstring for why
        those particular fields are intentionally left alone (genuinely
        still-meaningful history: what backend/resolution/fps this camera
        negotiated while it WAS open, how many frames it read over its
        lifetime) rather than reset to zero/None, which would destroy
        real information for no honesty benefit."""
        self._pump_stop.set()
        # SIGNAL EVERY STREAM BEFORE JOINING ANY OF THEM, and before the
        # pump is joined: a pump worker may be parked inside
        # StreamSource.read() waiting out its frame budget, and the pump
        # thread cannot finish its cycle until that worker returns. Same
        # ordering principle as "retire the readers before the captures
        # they hold", one layer up -- and signalling all of them first
        # means three dying streams cost one connect timeout between them
        # rather than three in series.
        sources = list(self._sources.values())
        for source in sources:
            source.stop()
        thread = self._pump_thread
        self._pump_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for source in sources:
            source.join()
        self._sources.clear()
        closed_at = time.time()
        for source in sources:
            status = self.status.get(source.slot)
            if status is not None:
                status.opened = False
                status.last_read_ok = False
                status.closed_at = closed_at
        for i, cap in list(self._caps.items()):
            lock = self._locks.get(i)
            with lock if lock is not None else contextlib.nullcontext():
                self._release_one_locked(i, cap)
            status = self.status.get(i)
            if status is not None:
                status.opened = False
                status.last_read_ok = False
                status.closed_at = closed_at
        self._caps.clear()
        self._raw_slots = set()
        with self._cache_lock:
            self._last_frames.clear()
            self._last_jpegs.clear()
            self._slot_generation.clear()
            # FRAME-DRIVEN WAKE, 2026-09-05 -- bump+notify here too, not
            # just in _pump_once(): this is the ONLY notify that can ever
            # fire for a hub whose pump thread never started at all (zero
            # cameras opened -- see open_all()'s own "only start it if at
            # least one camera actually opened" guard) or that is closed
            # before its pump has completed even one real cycle -- in
            # both cases _pump_once()'s own notify_all() may never have
            # fired, or may never fire again, so a waiter blocked on
            # wait_for_new_frame() would otherwise sit out its full
            # timeout rather than noticing "this hub is now closed"
            # promptly. Harmless if nobody is waiting (notify_all() on
            # zero waiters is a no-op) and harmless if the pump's own
            # last-cycle notify already covered it (a second notify_all()
            # on an already-empty waiter set is also a no-op) -- this is
            # a real safety net for the case the pump's own wind-down
            # does NOT provide one, not a redundant no-op call. Proven
            # load-bearing, not just reasoned about: tests/
            # test_local_capture_frame_driven_wake.py's own
            # test_close_all_wakes_a_blocked_waiter_when_the_pump_never_
            # ran_a_single_cycle fails cleanly (a waiter left blocked for
            # its full timeout) if these two lines are removed.
            self._frame_generation += 1
            self._cache_lock.notify_all()

    #: Same meaning, shorter name. Carried over from the deleted
    #: RemoteCameraHub, which had it because a call site spelled it that
    #: way; keeping it means no consumer has to know which hub it holds.
    close = close_all

    def _release_one_locked(self, i: int, cap: "cv2.VideoCapture") -> None:
        try:
            cap.release()
        except Exception: # noqa: BLE001 -- release() must never raise into a caller
            log.debug("cam%d: exception releasing capture (ignored)", i)

    def frame_generation(self) -> int:
        """Current frame-generation counter -- bumped once per completed
        pump cycle (see `_pump_once()`'s own comment) and once by
        `close_all()`. A caller wanting to wait for "a new frame set
        became available since I last looked" should record this value
        BEFORE doing its own work, then pass it to
        `wait_for_new_frame()` afterward -- see that method's own
        docstring, and module docstring's "FRAME-DRIVEN WAKE PRIMITIVE"
        section, for the full contract and why this is race-free."""
        with self._cache_lock:
            return self._frame_generation

    def wait_for_new_frame(self, last_generation: int, timeout: float) -> int:
        """Block until the pump publishes a frame generation newer than
        `last_generation`, or `timeout` seconds elapse -- whichever comes
        first. Returns the generation actually observed when this
        returns: equal to `last_generation` iff this timed out with no
        new generation ever appearing, strictly greater otherwise (never
        LESS -- the counter only ever increases).

        Race-free by construction (see module docstring's "FRAME-DRIVEN
        WAKE PRIMITIVE" section for the full reasoning): the predicate
        check and the wait happen atomically under `self._cache_lock`,
        the SAME lock `_pump_once()`/`close_all()` hold while mutating
        and notifying -- there is no window in which a pump cycle can
        complete invisibly between this method's own check and its wait,
        unlike a bare `threading.Event` a caller would have to `.clear()`
        between calls.

        `timeout <= 0` is a single non-blocking check-and-return (no
        wait attempted at all) -- matches `threading.Condition.wait()`'s
        own documented behavior for a non-positive timeout.
        """
        deadline = time.monotonic() + timeout
        with self._cache_lock:
            while self._frame_generation == last_generation:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cache_lock.wait(remaining)
            return self._frame_generation

    def status_report(self) -> str:
        """Multi-line human-readable status for all slots -- what
        scripts/test_local_camera_access.py prints after opening, and what
        /api/start logs when a camera fails to open.

        Each line is tagged with the slot's SOURCE. Without it, a slot
        reading a stream, a slot replaying a file and a slot reading a
        device are indistinguishable in exactly the log line someone reads
        when one of them is not delivering."""
        def _tag(i: int) -> str:
            if i < len(self._preset_sources) and self._preset_sources[i] is not None:
                return getattr(self._preset_sources[i], "source_label", "source")
            return "stream" if (i < len(self._slot_urls) and self._slot_urls[i]) else "local"

        return "\n".join(
            f"[{_tag(i)}] {self.status[i].summary()}"
            if i < len(self._slot_urls) else self.status[i].summary()
            for i in sorted(self.status)
        )


#: The name this class had while every slot it served was a local device.
#: Kept because it is spelled that way across capture_daemon, run_product,
#: the scripts and the docs, and a mass rename would be churn with no
#: behaviour in it. New code should say `CameraHub`: a hub whose slots can
#: read another machine's stream is not a "local" anything.
LocalCameraHub = CameraHub


@dataclass
class LocalSnapshot:
    """Mirrors opendarts.live.capture.LiveSnapshot's shape (cam/path/width/
    height) closely enough that opendarts/live/capture_daemon.py could swap
    frame sources with a small, clean change -- plus extra debug fields
    this local path can uniquely offer (an HTTP snapshot has no
    meaningful "backend"/capture-latency of its own to report; a direct
    cv2.VideoCapture grab does).
    """

    cam: int
    path: Path
    width: int
    height: int
    backend_used: str | None = None
    capture_latency_s: float | None = None


_default_hub: LocalCameraHub | None = None


def _get_default_hub() -> LocalCameraHub:
    """Lazily open (once) and reuse a module-level default hub -- unlike
    opendarts.live.capture's stateless per-call HTTP fetch, opening cameras
    is expensive and should happen once, not on every fetch_snapshot()
    call. Callers that want explicit control (tests, capture_daemon.py)
    should build and pass their own LocalCameraHub instead of relying on
    this."""
    global _default_hub
    if _default_hub is None:
        _default_hub = LocalCameraHub()
        _default_hub.open_all()
    return _default_hub


def fetch_snapshot(cam: int, dest_dir: Path, hub: LocalCameraHub | None = None) -> LocalSnapshot:
    """Grab camera `cam`'s current frame directly via cv2.VideoCapture
    and save it as a real PNG file -- local-capture counterpart to
    opendarts.live.capture.fetch_snapshot (same cam/dest_dir shape, same
    LiveSnapshot-like return; no base_url/timeout params since there's
    no HTTP round-trip here). Uses (and lazily opens) a shared default
    LocalCameraHub unless one is passed explicitly -- pass your own hub
    to control config/lifecycle (e.g. from capture_daemon.py or tests).
    """
    hub = hub or _get_default_hub()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    frame = hub.grab(cam)
    latency = time.monotonic() - t0
    if frame is None:
        status = hub.status.get(cam)
        reason = status.last_error if status is not None else "unknown camera index"
        raise RuntimeError(f"cam{cam}: failed to grab a frame -- {reason}")

    # ATOMIC WRITE, 2026-08-22 (real live incident, "libpng error:
    # IDAT: CRC error" spotted in the macOS rig's own terminal right at IDLE entry
    # -- traced, not guessed: `dest` below is a FIXED per-camera filename,
    # and TWO separate dashboard routes (server.py's snapshot.png AND
    # overlay.png, the latter cv2.imread()ing this exact file straight
    # back to draw a calibration overlay) both call this function for the
    # same cam_id. Two overlapping requests for the same camera -- e.g.
    # a dashboard reload firing while a prior poll's request is still in
    # flight, exactly what was happening the moment this was seen -- used
    # to race a partial cv2.imwrite() against a concurrent cv2.imread(),
    # producing a torn/truncated PNG (libpng's CRC error on a mid-write
    # read). Non-fatal by luck (imread failure already degrades to raw
    # bytes at both call sites) but a real, reproducible race, not a
    # cosmetic one. FIX: write to a request-unique temp file first, then
    # os.replace() it into `dest` -- POSIX rename is atomic, so any
    # concurrent reader of `dest` always sees either the fully-old or the
    # fully-new file, never a partial one.
    dest = dest_dir / f"local_cam{cam}.png"
    # NOTE: cv2.imwrite() picks its encoder from the filename's
    # extension, so the temp name must still end in `.png` (a `.tmp`
    # suffix on top errors with "could not find a writer for the
    # specified extension") -- the uniqueness comes from the prefix.
    tmp_dest = dest_dir / f".tmp.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.local_cam{cam}.png"
    ok = cv2.imwrite(str(tmp_dest), frame)
    if not ok:
        tmp_dest.unlink(missing_ok=True)
        raise IOError(f"cam{cam}: cv2.imwrite failed writing {tmp_dest}")
    os.replace(tmp_dest, dest)

    h, w = frame.shape[:2]
    status = hub.status.get(cam)
    return LocalSnapshot(
        cam=cam,
        path=dest,
        width=int(w),
        height=int(h),
        backend_used=status.backend_used if status is not None else None,
        capture_latency_s=latency,
    )


def fetch_all_snapshots(
    dest_dir: Path, n_cameras: int = 3, hub: LocalCameraHub | None = None
) -> list[LocalSnapshot]:
    """Local-capture counterpart to opendarts.live.capture.fetch_all_snapshots
    -- same dest_dir/n_cameras shape (no base_url). A camera that fails
    to grab is logged and skipped, not raised, matching the HTTP path's
    caller-must-check-len() posture."""
    hub = hub or _get_default_hub()
    snaps: list[LocalSnapshot] = []
    for i in range(n_cameras):
        try:
            snaps.append(fetch_snapshot(i, dest_dir, hub=hub))
        except (RuntimeError, IOError) as exc:
            log.warning("cam%d: skipped in fetch_all_snapshots -- %s", i, exc)
    return snaps
