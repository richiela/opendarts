"""opendarts/live/run_product.py -- ONE process/CLI command that runs the
capture loop AND the web dashboard/API server TOGETHER, sharing a SINGLE
`opendarts.live.local_capture.LocalCameraHub` instead of each half opening
its own (the status quo before this file existed: `-m
opendarts.live.capture_daemon` and `-m opendarts.live.server` run as two
separate processes in two separate terminals, each independently opening
all 3 cameras -- works, since these cameras are confirmed not
exclusive-open, but wasteful, and means the dashboard only sees the
daemon's real state indirectly, by polling saved packages on a timer,
never real live state).

2026-08-12: "can you combine this into one start script? Or at
least have one parent process that starts all the necessary pieces?"
This module is that one parent process.

WHEN TO USE THIS vs. the two standalone entrypoints
----------------------------------------------------
Both standalone entrypoints keep working EXACTLY as before -- this file
is a strict ADDITION, not a replacement, and does not change their
behavior at all (see opendarts/live/capture_daemon.py and
opendarts/live/server.py's own module docstrings, both updated to point
here).

  - **opendarts.live.run_product (this module)** -- the real day-to-day/
    product way to run this rig: ONE process, ONE camera hub opened
    once and shared by both halves, capture loop + web dashboard running
    together, real live-event push from the capture loop straight to the
    dashboard's WebSocket clients (see "Real event push" below) instead
    of the dashboard only ever discovering new state by polling.
    ``.venv/bin/python3 -m opendarts.live.run_product``

  - **opendarts.live.capture_daemon, run alone** -- still useful for a pure
    console test of the capture loop with NO web UI at all: minimal
    footprint, no fastapi/uvicorn dependency needed, simplest thing to
    reach for when debugging the trigger/scoring loop itself in
    isolation (e.g. over SSH-less console access, or when a browser
    genuinely isn't wanted). Opens/owns/closes its own hub, unaffected
    by anything in this file.
    ``.venv/bin/python3 -m opendarts.live.capture_daemon``

  - **opendarts.live.server, run alone** -- useful when you want ONLY the
    dashboard (browsing saved packages / camera snapshots / calibration
    status) with NO capture loop running at all -- e.g. the capture
    daemon is deliberately stopped, or you're just reviewing history.
    Opens/owns/closes its own hub, unaffected by anything in this file.
    ``.venv/bin/python3 -m opendarts.live.server``

How this is built (reuse, not duplication)
-------------------------------------------
This module does NOT reimplement the capture loop or the dashboard. It
calls:
  - `opendarts.live.capture_daemon.run_capture_loop_body()` -- the exact
    same loop logic `capture_daemon.py`'s own standalone
    `run_capture_loop()` calls, refactored out specifically so both
    callers share one implementation (see that module's docstring for
    why the split exists). Run here in a background thread against the
    ONE shared hub.
  - `opendarts.live.server.create_app()` -- the exact same FastAPI app
    factory `server.py`'s own standalone CLI uses, via its existing
    `local_hub=` (share the caller's already-open hub, don't open
    another) and new `live_event_queue=` (real event push, see below)
    injection seams. Run here via `uvicorn.Server(...).run()` in the
    MAIN thread.

Real event push (task item 2, done -- not left as polling-only)
-----------------------------------------------------------------
The capture loop thread is given an `on_event` callback that puts small
event dicts (`{"type": "TRIGGER_STATE", ...}` on every trigger state
transition, `{"type": "PACKAGE_SAVED", ...}` right after a throw package
is written) onto a thread-safe `queue.SimpleQueue`. The SAME queue is
passed into `create_app(live_event_queue=...)`; the server's own
background task (`AppState._live_event_loop`, see server.py) consumes
that queue and broadcasts each event to connected WebSocket clients
immediately -- real push, not the server discovering it up to
`package_poll_interval_s` (4s) later via its polling fallback.

Calibration cadence, REWORKED 2026-08-12: there is no automatic recalibration
poll anywhere in this process anymore (the old `AppState.
_calibration_poll_loop`, a 20s timer that silently recomputed calibration
purely to keep the dashboard's numbers fresh, has been REMOVED, not just
throttled). Calibration now updates at exactly two moments, both real:
(1) once at startup, when `run_capture_loop_body()` bootstraps it (or
REUSES a persisted one -- see the 2026-08-26 update just below), and
(2) whenever the dashboard's "Refresh calibration now" button is
actually clicked. A single `opendarts.live.capture_daemon.CalibrationStore`
is built HERE, before the capture thread starts, and shared with BOTH
halves (`create_app(calibration_store=...)` and the capture thread's own
`run_capture_loop_body(calibration_store=...)` call below) -- the capture
loop thread reads it fresh on every single READY_TO_CAPTURE, and a manual
recalibrate (running on a request-handling thread) writes the new
calibration into that SAME object, under a lock (see CalibrationStore's
own docstring for the thread-safety argument) -- so a manual recalibrate
genuinely changes what the NEXT scored throw uses, not just what the
dashboard displays. (2026-08-22: a manual recalibrate used to ALSO
persist a standalone durable record via `save_calibration_snapshot()`
-- removed as genuinely dead code, see `AppState.refresh_calibration()`'s
own updated docstring in `opendarts/live/server.py`.)

**Durable across process restarts, 2026-08-26**: the `CalibrationStore` built HERE now passes
`snapshot_path`, so "once at startup" above means "once ever, until a
human clicks Refresh" -- a process restart with a valid persisted
calibration on disk loads it directly and never calls
`bootstrap_calibrations()` at all on the next Start. See
`CalibrationStore`'s own docstring for the full design and the one real,
accepted trade-off (a hardware re-seat between restarts is not
auto-detected -- mitigated by a visible "calibrated at" timestamp on the
dashboard, not by any new auto-detection).

AD ground truth, added 2026-08-12 (a real live finding): the
`/api/state/detections` REST list only covers the
current visit and is not durable, which made the original plan (fetch
AD's ground truth via a later REST call/backfill CLI) structurally unable
to recover ground truth once the visit had ended in the meantime --
there is no way to poll fast enough to reliably win that race. Fixed with
`opendarts.live.ad_ws_listener.AdWsListener`: a persistent WebSocket
connection to AD's own `/api/events` push stream, opened/started here alongside the
camera hub (same caller-owns-the-long-lived-connection pattern) and
closed/stopped in `shutdown()` below. The listener is built by default
but only connects when the config's `ad_enabled` is true (off by default);
`--no-ad-ground-truth` opts out of the INLINE auto-attach entirely -- the dashboard's
existing manual REST-based refresh button stays wired regardless, as a
fallback for any package that missed the inline attach).

Camera lifecycle: Start/Stop/idle-timeout, added 2026-08-12
-----------------------------------------------------------
when the program starts, it should not default to camera start."
`_build_components()` below now only CONSTRUCTS the shared
`LocalCameraHub` -- it no longer opens it, and no longer raises
`RuntimeError` for zero cameras (that real failure mode moved to
`opendarts.live.server`'s `POST /api/start`, see
`AppState.start_capture()`). The capture thread built here
(`_capture_thread_target`) now runs a persistent OUTER loop for this
process's whole lifetime, waiting (bounded polls) on a shared
`opendarts.live.capture_daemon.CaptureLoopController.start_requested` signal
between SESSIONS, rather than running exactly one session and exiting --
see that class's own docstring for the full design (an explicit
start/stop pair plus a touch/idle-loop timeout, adapted for
this file's thread-based, not asyncio-task-based, architecture -- stated
there plainly, not glossed over). A session ending because of a manual
Stop or an idle-timeout (default 900s) does NOT bring the whole process
down -- only a genuine crash
(or the still-defensive legacy `NotImplementedError` case) does, same
"an orphaned dashboard backed by a dead capture loop defeats the whole
point of this file" discipline this module already had before this
feature, just now scoped to real failures instead of every session end.

Clean shutdown
---------------
`main()` installs SIGINT/SIGTERM handlers in the MAIN thread (Python
only allows `signal.signal()` there) that set a shared `threading.Event`
(stopping the capture loop thread) and `uvicorn.Server.should_exit`
(stopping uvicorn's own serve loop). If the capture loop thread instead
dies on its own (crash, or `NotImplementedError` -- see
`capture_daemon.py`'s own honest status on what's not implemented yet)
it also sets the stop event, so an operator isn't left staring at a
dashboard backed by a dead capture loop; `shutdown()` below is the single
place that joins the capture thread (bounded wait, then proceeds anyway
-- the thread is daemon=True so it can never block process exit) and
closes the shared hub EXACTLY once, called from `main()`'s `finally`
after `uvicorn.Server.run()` returns. See tests/test_run_product.py for
this actually being exercised with a real delivered signal, not just
"the code looks right."

HONEST STATUS, same standing caveat as the rest of this project's
camera-access work: nothing in this file has been run against the rig's
real USB cameras -- no dev session for this project ever has. What's
proven here: the Python control flow (exactly one hub opened and shared,
both pieces actually start, signal-triggered shutdown is clean -- no
leaked threads, hub closed exactly once) via
`tests/test_run_product.py`, all mocked, none touching real hardware or
binding a real uvicorn port. Real validation is the same next step as
`local_capture.py`'s own: run this for real on the rig and see what
happens.

CLI usage:
    .venv/bin/python3 -m opendarts.live.run_product
    .venv/bin/python3 -m opendarts.live.run_product --port 8420 --host 0.0.0.0 --package-root ...
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import queue
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Callable

from opendarts.lifecycle.settings import LifecycleSettingsStore
from opendarts.live.displays import DisplayStore
from opendarts.live.dashboard_choice import DashboardChoice
from opendarts.live import (camera_names, capabilities, local_capture,
                           remote_capture, vcam)
from opendarts.live.ad_ground_truth import (
    DEFAULT_AD_BASE,
    DEFAULT_MATCH_WINDOW_SEC,
    DEFAULT_TIMEOUT_SEC as AD_DEFAULT_TIMEOUT_SEC,
)
from opendarts.live.ad_ws_listener import AdWsListener


from opendarts.live.capture_daemon import (
    CURRENT_CALIBRATION_FILENAME,
    DEFAULT_CALIBRATION_PACKAGE_ROOT,
    DEFAULT_PACKAGE_ROOT,
    POLL_INTERVAL_SECONDS,
    CalibrationStore,
    CaptureLoopController,
    EngineConfigStore,
    ResetRequest,
    run_capture_loop_body,
)
# LiveConfig is the DATACLASS, not the file -- importing the type here
# does not read data/config.json, and the module docstring's "loaded
# ONLY at the entrypoint" rule is about load_live_config(), which main()
# still calls exactly once. DEFAULT_CONFIG_PATH was already imported the
# same way.
from opendarts.capture import frame_ring
from opendarts.capture.throw_capture import ThrowCaptureService
from opendarts.live.config import (DEFAULT_CONFIG_PATH,
                                   DEFAULT_FRAME_RING_SECONDS,
                                   DEFAULT_VIDEO_RECORD_MODE, LiveConfig)
from opendarts.live.server import DEFAULT_HOST, DEFAULT_PORT, create_app

log = logging.getLogger("opendarts.run_product")


def _read_store_packages() -> bool:
    """Whether completed throws are written to disk as packages.

    Default TRUE -- a rig that silently discards its own evidence would
    be a bad default, and every existing deployment expects packages.
    Set ``"store_packages": false`` in data/config.json for a
    shipped/retail rig that scores darts without accumulating roughly
    6.5 MB of frames per throw.

    Scoring, the live feed and the retail channel are identical either
    way: THROW_DETECTED is emitted before the save step, and
    GET /api/live/recent is served from an in-memory ring rather than
    from packages, precisely so match history survives this being off.

    What IS lost with it off: the replay corpus, per-throw diagnostics,
    AD ground truth on disk, and the Engines tab (which reads packages).
    A bad value is treated as "store", the safe direction.

    Settable from the dashboard's Config tab (Recording) as well as by
    hand; that control writes this same key. The interpretation itself
    lives in opendarts.live.config.store_packages_enabled(), which this
    delegates to, so this reader and the dashboard cannot drift apart
    and report different states for the same file. Kept as a named
    function here because THIS is the call site that decides a real
    session's behaviour.
    """
    from opendarts.live.config import store_packages_enabled

    return store_packages_enabled()


def _read_min_free_disk_gb() -> float:
    """The free-space floor, in GB, that both on-disk writers stop at.

    ONE read, handed to BOTH writers -- the capture loop's throw-package
    save and the throw-capture service's ring dumps -- so a rig cannot
    end up refusing one and allowing the other on the same disk. The
    interpretation lives in `opendarts.live.config.min_free_disk_gb()`;
    this is simply the call site that decides a real session's behaviour,
    the same way `_read_store_packages()` above is for its key.

    Default 5 GB. `0` or an absent key means the same; a NEGATIVE value
    turns the guard off entirely, which is the documented opt-out for a
    rig with its own disk management.
    """
    from opendarts.live.config import min_free_disk_gb

    return min_free_disk_gb()

# Bounds uvicorn's own graceful-connection-drain wait (see
# uvicorn.Config's timeout_graceful_shutdown, wired in _build_components
# below) -- real defense-in-depth for the originally-hypothesized "a live
# WebSocket client doesn't close promptly" risk, on top of the confirmed
# root-cause fix in opendarts/live/server.py's AppState._live_event_loop.
GRACEFUL_SHUTDOWN_TIMEOUT_S = 3.0

# Absolute last resort, per this session's task: if the process still
# hasn't exited this many seconds after a stop signal was received --
# whether from a hang in uvicorn's own internals (the confirmed root
# cause this session found, now fixed) or any OTHER future bug of similar
# shape -- force the process to exit rather than hang forever. Comfortably
# longer than GRACEFUL_SHUTDOWN_TIMEOUT_S + shutdown()'s own
# join_timeout_s so the clean paths get a real chance to finish first;
# short enough that an operator hitting Ctrl-C never waits long. See
# _install_signal_handlers/_arm_shutdown_watchdog below -- always logs
# loudly if it actually fires, never a silent footgun.
WATCHDOG_FORCE_EXIT_TIMEOUT_S = 10.0


@dataclasses.dataclass
class ProductComponents:
    """Everything main() needs to start, run, and cleanly stop the
    combined process -- built by _build_components() below, kept as a
    plain dataclass (not just local variables in main()) specifically so
    tests/test_run_product.py can build one against a fake hub/fake loop
    body and exercise startup + shutdown without going through main()'s
    own real CLI/uvicorn machinery.
    """

    hub: local_capture.LocalCameraHub
    app: Any # fastapi.FastAPI -- typed Any to avoid a hard fastapi import at module scope
    stop_event: threading.Event
    live_events: "queue.SimpleQueue[dict[str, Any]]"
    capture_thread: threading.Thread
    server: Any # uvicorn.Server -- see hub/app's own Any note above
    # Set by _install_signal_handlers' own handler the moment a stop
    # signal actually arrives (None until then / in tests that never
    # deliver one) -- see _arm_shutdown_watchdog / shutdown() below.
    watchdog_timer: threading.Timer | None = None
    # None when AD ground-truth inline capture is disabled
    # (--no-ad-ground-truth) -- see module docstring's "AD ground truth"
    # section and shutdown()'s own stop() call below. Lifecycle mirrors
    # `hub` exactly: opened/started in _build_components(), closed/
    # stopped exactly once in shutdown().
    ad_ws_listener: AdWsListener | None = None
    # The shared, mutable calibration reference both the capture thread
    # and the dashboard's manual recalibrate action read/write -- see
    # module docstring's "Calibration cadence" section and
    # opendarts.live.capture_daemon.CalibrationStore's own docstring. Kept on
    # this dataclass (not just a closure-captured local in
    # _build_components()) so tests can inspect/drive it directly.
    calibration_store: CalibrationStore | None = None
    # The shared, mutable reset signal both the capture thread and the
    # dashboard's manual "Reset" action read/write -- see module
    # docstring's "Calibration cadence" section's own reasoning (this
    # mirrors it exactly, same established pattern, deliberately not a
    # new mechanism) and opendarts.live.capture_daemon.ResetRequest's own
    # docstring. Kept on this dataclass for the same test-inspection
    # reason as calibration_store above.
    reset_request: ResetRequest | None = None
    # The Start/Stop/idle-timeout coordination object -- see
    # opendarts.live.capture_daemon.CaptureLoopController's own docstring.
    # Same test-inspection reasoning as calibration_store/reset_request
    # above.
    controller: CaptureLoopController | None = None
    # The shared, mutable multi-engine scoring config (docs/ENGINES.md) --
    # same established pattern/reasoning as calibration_store/
    # reset_request/controller above: built once here, shared by
    # reference with BOTH the capture thread (reads fresh every
    # READY_TO_CAPTURE) and the dashboard's Config-tab POST handler
    # (writes a new primary/also-run/timeout). See
    # opendarts.live.capture_daemon.EngineConfigStore's own docstring.
    engine_config_store: EngineConfigStore | None = None
    # Operator-tunable lifecycle Detection time (dart_stable_frames) --
    # same optional-so-manual-ProductComponents-constructions-keep-working
    # default as engine_config_store above.
    lifecycle_settings_store: LifecycleSettingsStore | None = None


def _resolve_camera_urls(
    cli: "list[str] | None", from_config: "list[str | None] | None"
) -> "list[str | None] | None":
    """CLI beats config, and one bare base URL means "that rig's cameras".

    `--camera-url http://rig:8420` is what someone actually types, and
    making them write out three `/api/cameras/N/stream.mjpg?full=1` URLs to
    say the obvious thing would be a papercut on the one command this
    feature exists for. A URL that already names a stream is passed
    through untouched, so an unusual layout stays expressible.

    The literal "local" leaves a slot on its hardware device, which is how
    a mix is expressed on the command line.
    """
    if not cli:
        return from_config
    if len(cli) == 1 and "/api/cameras/" not in cli[0] and cli[0] != "local":
        return list(remote_capture.urls_for(
            cli[0], len(local_capture.DEFAULT_CAMERA_DEVICES)
        ))
    return [None if u == "local" else u for u in cli]


def _camera_configs_from_live_config(live_cfg: Any) -> "list[local_capture.CameraConfig] | None":
    """Turn a loaded LiveConfig into the `configs=` list for the hub, or
    None to mean "use the hub's own default device list".

    Returns None when the config says nothing about EITHER device
    assignment or resolutions, so a rig that has never touched these keys
    constructs a byte-identical hub to before this wiring existed --
    `LocalCameraHub(configs=None)` takes its own
    `[CameraConfig(device=d) for d in DEFAULT_CAMERA_DEVICES]` branch.

    Until 2026-09-10 `camera_configs_from_resolution_preferences()` existed
    but was called by nothing, so `camera_resolutions` was parsed, validated
    and then silently discarded. This is the wiring that was deliberately
    deferred for live-hardware validation; `camera_devices` is what finally
    forced it, since a laptop whose built-in webcam holds index 0 cannot
    otherwise reach its three board cameras at all.
    """
    devices = getattr(live_cfg, "camera_devices", None)
    resolutions = getattr(live_cfg, "camera_resolutions", None) or {}
    if not devices:
        # AUTODETECT, but only where the default is actually wrong. On
        # Linux a UVC camera registers a capture node AND a metadata node,
        # so three cameras occupy /dev/video0..5 and the [0, 1, 2] default
        # selects two cameras and a metadata node -- which fails with
        # "can't open camera by index", a message that points at nothing.
        # Measured on the rig: the right answer there is [0, 2, 4].
        #
        # Returns None on every other platform and whenever it cannot find
        # a full set, so this cannot quietly reconfigure a rig that simply
        # has a camera unplugged -- see camera_names.suggest_camera_devices.
        devices = camera_names.suggest_camera_devices(
            len(local_capture.DEFAULT_CAMERA_DEVICES)
        )
    if not devices and not resolutions:
        return None
    return local_capture.camera_configs_from_resolution_preferences(
        resolutions, devices=devices
    )


def _read_publish_virtual_cameras(ad_enabled: bool) -> bool:
    """Whether to republish frames to Windows virtual cameras.

    Follows the oracle toggle by default: the virtual cameras are how
    other software watches the same board on Windows. Publishing with the
    toggle switched off copies
    several megabytes per camera per frame to a consumer nobody asked for,
    and leaving it on by accident is exactly the kind of cost that hides.

    An explicit `publish_virtual_cameras` key still wins in either
    direction -- someone feeding a different DirectShow consumer, or
    deliberately keeping the cameras fed while the oracle is off, should
    not have that decision overridden by a default.
    """
    from opendarts.live.config import read_config_section

    try:
        value = read_config_section("publish_virtual_cameras")
    except Exception: # noqa: BLE001 -- a bad config must never stop a session
        return False
    if isinstance(value, bool):
        return value
    return bool(ad_enabled)


def _virtual_camera_geometry(camera_configs) -> "tuple[int, int]":
    """Geometry for the virtual cameras.

    Fixed per run because it is baked into the shared mapping size and the
    filter's advertised media type -- a frame of another size is refused
    rather than resized. Takes the first camera's configured size, falling
    back to the capture default when cameras are on "auto" and the real
    size is not known until they open.
    """
    if camera_configs:
        first = camera_configs[0]
        if getattr(first, "width", None) and getattr(first, "height", None):
            return int(first.width), int(first.height)
    return local_capture.DEFAULT_WIDTH, local_capture.DEFAULT_HEIGHT


def _build_components(
    *,
    package_root: Path,
    host: str,
    port: int,
    poll_interval_s: float,
    hub_factory: Any = local_capture.LocalCameraHub,
    ad_base_url: str = DEFAULT_AD_BASE,
    ad_window_sec: float = DEFAULT_MATCH_WINDOW_SEC,
    ad_timeout_s: float = AD_DEFAULT_TIMEOUT_SEC,
    enable_ad_ground_truth_inline: bool = False,
    ad_listener_factory: Any = AdWsListener,
    reprojection_targets_px: dict[int, float] | None = None,
    camera_configs: "list[local_capture.CameraConfig] | None" = None,
    camera_devices: "list[int] | None" = None,
    camera_urls: "list[str | None] | None" = None,
    frame_ring_seconds: "float | None" = None,
    frame_ring_max_gb: "float | None" = None,
    video_record_mode: "str | None" = None,
) -> ProductComponents:
    """Opens exactly ONE camera hub, builds the FastAPI app sharing it
    (via create_app()'s existing local_hub= seam), builds the (not-yet-
    started) capture-loop thread sharing it too, and wires a real
    live_event_queue between them. Does NOT start the capture thread and
    does NOT run the uvicorn server -- that's main()'s job, once signal
    handlers are installed, so a caller can build+start+shutdown these
    pieces independently for testing (see tests/test_run_product.py).

    hub_factory: test injection seam, mirrors tests/test_capture_daemon.py's
    own FakeHub-via-monkeypatch pattern but as an explicit parameter here
    instead, since this function (unlike capture_daemon.run_capture_loop)
    is meant to be called directly by tests, not just monkeypatched into.

    ad_base_url/ad_window_sec/ad_timeout_s: always passed to create_app()
    below (the dashboard's own manual "Fetch AD ground truth now" REST
    button, see opendarts/live/server.py -- unaffected by
    enable_ad_ground_truth_inline, which only controls the NEW automatic
    inline path added this session). ad_listener_factory mirrors
    hub_factory's own test-injection-seam pattern.

    enable_ad_ground_truth_inline defaults to False AT THIS FUNCTION LEVEL
    deliberately (opt-in here, opt-OUT at the CLI level) -- main() always
    passes it explicitly (`not args.no_ad_ground_truth`, i.e. True unless
    --no-ad-ground-truth is given; the config's `ad_enabled`, off by
    default, then decides whether the listener actually connects).
    Keeping the low-level default False means any OTHER caller of this
    function (every existing test in tests/test_run_product.py, and any
    future one that doesn't explicitly ask for AD ground truth) never
    starts a real background WebSocket connection attempt to
    DEFAULT_AD_BASE by surprise -- this project's tests must never make a
    real network call to AD. When True, an AdWsListener is opened/
    started here, alongside the camera hub, and is the caller's
    responsibility to stop exactly once (see shutdown() below) --
    identical lifecycle discipline to `hub`.

    The listener (when built) is threaded straight into
    run_capture_loop_body()'s own `ad_ws_listener=`/`ad_match_window_sec=`
    params below (see opendarts/live/capture_daemon.py's own docstring for
    what that does -- an in-memory buffer match, no network call at
    throw time), so `handle_ready_to_capture()` matches every freshly-
    saved package against it automatically.

    A second external oracle's live-scored answer, when wanted, comes
    from running a non-voting registry engine -- this function no longer
    builds a separate WebSocket listener for one (the older "second
    oracle" dashboard toggle/status-light mechanism this used to feed
    was removed once that engine existed).
    """
    import uvicorn # local import: fastapi/uvicorn are an optional dependency, see main()

    # CHANGED 2026-08-12
    # starts, it should not default to camera start." Cameras used to be
    # opened HERE, unconditionally, the moment this process started (and
    # the whole function raised RuntimeError if zero opened). The process
    # now starts fine with zero cameras open; opening/probing them only
    # happens at an explicit `/api/start` (see that handler's own
    # `start()`, read directly). The hub is constructed here (so it
    # exists to share with create_app()/the capture thread below) but
    # deliberately left CLOSED -- opendarts/live/server.py's new `POST
    # /api/start` endpoint is what actually calls `hub.open_all()`, on an
    # operator's explicit click.
    # Windows only, and only when the operator asked for it: republish
    # every captured frame to virtual cameras so other software can watch
    # the same board. On macOS the two share the hardware natively and on
    # Linux that is v4l2loopback's job, so `available()` is False and this
    # is None -- the hub then behaves exactly as it always has.
    #
    # Fed from the frames the pump ALREADY cached rather than opening the
    # cameras a second time: a second reader is the exact contention this
    # whole mechanism exists to avoid.
    # The oracle toggle, resolved the same way the listener will resolve
    # it, so publishing and the oracle cannot disagree about whether AD is
    # in use.
    ad_enabled = enable_ad_ground_truth_inline
    if ad_enabled:
        from opendarts.live.config import read_config_section
        try:
            configured = read_config_section("ad_enabled")
        except Exception: # noqa: BLE001
            configured = None
        # OFF UNLESS ASKED FOR, changed 2026-09-16. This used to read
        # `if configured is False`, so an absent key meant ON -- a default
        # that stopped making sense the moment anyone else could clone this,
        # and the cost lands on exactly the person least able to explain it:
        #
        #   * the AD WS listener retries forever with backoff, so the first
        #     thing a new user sees is a stream of failures to connect to a
        #     service they have never heard of, on localhost:3180
        #   * it drives virtual-camera publishing too, so a fresh rig
        #     encodes three MJPEG streams into loopback devices for a
        #     consumer nobody has
        #
        # Someone running Autodarts knows what it is and can switch it on.
        # Someone who is not should never have to learn the name to stop
        # paying for it.
        if configured is not True:
            ad_enabled = False

    # Registration is registry state, so it survives the process that
    # wrote it -- a rig that was last run with the oracle on comes back up
    # with the devices still registered, and one that crashed mid-session
    # can be carrying a registration for a DLL path that has since moved.
    # Reconciling it against the persisted flag at startup means the
    # registry always matches what this process is actually doing, rather
    # than whatever the last one left behind. The call is a no-op off
    # Windows and never raises.
    if vcam.register.available():
        result = vcam.register.apply(ad_enabled)
        if not result.get("ok"):
            log.warning("virtual camera %s at startup failed: %s",
                        result.get("action"), result.get("reason"))

    def make_virtual_camera_set():
        """Build the publisher set for this rig, or None if it should not
        publish at all.

        A FACTORY RATHER THAN A PLAIN `if`, 2026-09-15. Publishing used to
        be decided once, here, at startup: switch the oracle on
        later and `server`'s `_set_virtual_camera_publishing(True)` had
        nothing to attach -- `state.vcam_set` was None, so the toggle
        silently did nothing and the rig needed a restart. Hit on the
        Linux rig. Handing `create_app` the means to build one on demand
        is what makes the toggle honest in both directions.

        The explicit `publish_virtual_cameras` key is re-read on every
        call rather than closed over: it must still be able to veto
        publishing when the oracle is switched on hours later, and an
        operator who edits it should not need a restart either. `True` is
        passed for the oracle because the only caller that is not this
        startup path is the toggle turning the oracle ON.
        """
        # SAY WHY, per docs/DESIGN.md's "a refusal, cap, or fallback must
        # say so in the log". Both of these used to return None in
        # silence, so "publishing is switched off" and "publishing is
        # broken" produced identical evidence: virtual cameras that simply
        # never appeared, with a clean log and a healthy-looking rig. That
        # cost a real debugging session on 2026-09-15, chasing a pixel
        # format when the oracle toggle was simply off.
        if not vcam.publish.available():
            log.warning(
                "virtual cameras: not publishing -- the %s backend reports "
                "unavailable. On Linux that means the v4l2loopback module is "
                "not loaded or no loopback devices exist (see docs/LINUX.md); "
                "on macOS there is no backend at all.",
                vcam.backend_name(),
            )
            return None
        if not _read_publish_virtual_cameras(True):
            log.warning(
                "virtual cameras: not publishing -- 'Compare against Autodarts' "
                "is off and no explicit 'publish_virtual_cameras' key overrides "
                "it, so no virtual cameras are being fed. Set that key true to "
                "publish regardless."
            )
            return None
        width, height = _virtual_camera_geometry(camera_configs)
        n_slots = len(camera_configs) if camera_configs else 3
        # CAPABILITY CHECK, NOT A PLATFORM CHECK -- the same rule the engine
        # dispatcher follows (docs/DESIGN.md). The Linux set takes a pixel
        # format; the Windows one has no such concept and always writes
        # BGR24 into shared memory. Asking whether THIS backend declares
        # `fmt` keeps the config key meaningful where it applies and inert
        # where it does not, without this function knowing which OS it is
        # on.
        kwargs: dict[str, Any] = {}
        # read_config_section, NOT load_live_config(): this module's own
        # rule is that the config file is loaded once at the entrypoint,
        # and every deferred read here goes through a targeted section
        # read -- the same way _read_publish_virtual_cameras does.
        fmt = None
        try:
            from opendarts.live.config import (
                normalise_v4l2_format, read_config_section,
            )
            fmt = normalise_v4l2_format(read_config_section("v4l2_format"))
        except Exception: # noqa: BLE001 -- a bad config must not stop publishing
            log.warning("could not read v4l2_format, using the publisher default")
        if fmt:
            import inspect

            params = inspect.signature(vcam.publish.VirtualCameraSet).parameters
            if "fmt" in params:
                kwargs["fmt"] = fmt
            else:
                log.info(
                    "v4l2_format=%s ignored: the %s backend has no pixel-format "
                    "choice (it already publishes an uncompressed copy)",
                    fmt, vcam.backend_name(),
                )
        built = vcam.publish.VirtualCameraSet(n_slots, width, height, **kwargs)
        log.info("virtual cameras: publishing %d slot(s) at %dx%d via %s",
                 len(built.publishers), width, height, vcam.backend_name())
        return built

    vcam_set = None
    frame_sink = None
    if _read_publish_virtual_cameras(ad_enabled):
        vcam_set = make_virtual_camera_set()
        if vcam_set is not None:
            frame_sink = vcam_set.publish_all

    # `configs=None` is the hub's own "use DEFAULT_CAMERA_DEVICES"
    # default, so a rig with no camera_devices key in its config builds
    # exactly the hub it always did.
    # ONE HUB, whatever the slots read. There is no branch here on "local
    # rig" vs "stream rig" because there is no such property of a rig: the
    # assignment is per SLOT in the dashboard, and a mix is an ordinary
    # state rather than one to reject -- a control that offers a state the
    # product refuses is worse than one that works. build_hub just hands
    # the per-slot URLs to the hub; see remote_capture's module docstring
    # for the two-hub composition this replaced and the four bugs that
    # lived in the mapping between them.
    hub = remote_capture.build_hub(
        camera_devices,
        camera_urls,
        hub_factory=hub_factory,
        frame_sink=frame_sink,
        camera_configs=camera_configs,
    )

    # THE THROW-CAPTURE RING, attached to the one hub before either half
    # starts -- the same "built here, shared by reference" reasoning every
    # other cross-thread object in this function uses. The capture thread
    # fills it through the pump and the dashboard triggers dumps off it
    # through `throw_capture`; both must reach the SAME ring, and a second
    # one would buffer nothing while reporting a size.
    #
    # The memory cost is stated in the log at startup, not only in the UI:
    # an operator whose rig is short of RAM reads the log, and "the ring
    # is on and it is 4GB" is the single most useful line this feature can
    # emit. See opendarts/capture/frame_ring.py for the measured
    # arithmetic behind the estimate.
    record_mode = video_record_mode or DEFAULT_VIDEO_RECORD_MODE
    ring_seconds = (
        DEFAULT_FRAME_RING_SECONDS if frame_ring_seconds is None
        else float(frame_ring_seconds)
    )
    # "never" turns the ring OFF regardless of frame_ring_seconds: with no
    # per-throw recording and no manual dumps wanted, buffering frames in
    # RAM would be spent for nothing (up to several GB on macOS). The other
    # modes keep the ring at its configured window.
    if record_mode == "never":
        ring_seconds = 0.0
    # ASKED OF THE HUB, not assumed of it -- docs/DESIGN.md's "capability
    # checks, not name checks". `hub_factory` is a real test-injection
    # seam and the stubs behind it implement only what they need, so a
    # bare `hub.set_frame_ring(...)` would turn "this fake has no ring
    # support" into an AttributeError that reads as a product bug. The
    # slot count degrades the same way: `configs` is the hub's own
    # authority on how many slots it has, and a hub that cannot say falls
    # back to the default camera count purely so the LOG LINE has a
    # number in it.
    # DETECTION DECODES SMALL (opendarts/capture/lazy_frame.py): set before
    # the hub's pump runs, asked of the hub the same way as the ring below.
    from opendarts.live.config import detect_from_small_decode_enabled

    small_decode = detect_from_small_decode_enabled()
    set_small_decode = getattr(hub, "set_small_decode", None)
    if callable(set_small_decode):
        set_small_decode(small_decode)
        log.info(
            "detection: %s (detect_from_small_decode in data/config.json)",
            "each JPEG decoded straight to small grey; full decode only for scored frames"
            if small_decode else "every frame fully decoded in the pump",
        )
    slots = getattr(hub, "configs", None)
    n_slots = len(slots) if slots is not None else len(local_capture.DEFAULT_CAMERA_DEVICES)
    attach_ring = getattr(hub, "set_frame_ring", None)
    if ring_seconds > 0 and callable(attach_ring):
        estimate = frame_ring.estimated_bytes_per_second(n_slots) * ring_seconds
        ring = frame_ring.FrameRing(
            ring_seconds,
            max_bytes=(int(frame_ring_max_gb * 1e9) if frame_ring_max_gb else None),
        )
        attach_ring(ring)
        log.info(
            "throw-capture ring: %.1fs across %d slot(s), about %s of memory "
            "once full%s (frame_ring_seconds in data/config.json; 0 disables it)",
            ring_seconds, n_slots, frame_ring.format_bytes(estimate),
            f", capped at {frame_ring_max_gb:g} GB" if frame_ring_max_gb else "",
        )
    elif ring_seconds > 0:
        # A REFUSAL THAT SAYS SO. This hub cannot hold a ring, so no
        # frames are being retained however the config reads -- and a
        # dashboard reporting a configured 22 seconds beside a hub that
        # buffers nothing is precisely the silent-refusal class
        # docs/DESIGN.md exists to forbid.
        ring = None
        log.warning(
            "throw-capture ring: NOT attached -- this hub (%s) has no "
            "set_frame_ring(), so nothing is being buffered despite "
            "frame_ring_seconds=%.1f", type(hub).__name__, ring_seconds,
        )
    else:
        ring = None
        log.info(
            "throw-capture ring: DISABLED (frame_ring_seconds=0) -- a missed or "
            "misscored dart cannot be captured on this rig"
        )
    # The SAME floor the capture loop's own package save uses (read
    # again per session below, so an operator editing data/config.json
    # does not need a restart for the loop's half). This service is built
    # once for the process, so its floor is the one read at startup.
    throw_capture_service = ThrowCaptureService(
        ring, min_free_disk_gb=_read_min_free_disk_gb(), record_mode=record_mode,
    )

    log.info(
        "camera hub constructed (NOT opened yet -- cameras open on an explicit "
        "Start, see opendarts/live/server.py's POST /api/start): %s",
        hub.status_report(),
    )

    # Built here, BEFORE the AD listener below, specifically so its
    # own on_status_change= callbacks (wired immediately below) can close
    # over this SAME queue object -- there is exactly one live_events
    # queue for the whole process, shared by the capture loop's
    # _on_capture_event() (defined further down) and now these two
    # listener callbacks as well. No separate polling loop, no second
    # queue -- this is the direct fix for the 2026-08-14 correction: don't
    # poll, the WS subscription already carries everything needed.
    stop_event = threading.Event()
    live_events: "queue.SimpleQueue[dict[str, Any]]" = queue.SimpleQueue()

    def _make_board_status_pusher(kind: str) -> "Callable[[str], None]":
        """Returns an AdWsListener on_status_change callback (contract:
        Callable[[str], None], called from ITS background WS-receive
        thread, never the main thread) that pushes a {"type": kind,
        "status": ...} event onto live_events -- the exact same
        cross-thread hand-off mechanism the capture loop's own
        _on_capture_event() below already uses, so
        AppState._live_event_loop's single consumer/broadcaster picks
        these up with no new machinery. `kind` is "AD_BOARD_STATUS" --
        see opendarts/live/server.py's _handle_live_event() dispatch branch
        for the other end of this.
        """

        def _push(status: str) -> None:
            live_events.put({"type": kind, "status": status})

        return _push

    def _push_ad_connection_change(connected: bool) -> None:
        """AdWsListener on_connection_change callback -- same queue, same
        thread contract as the board-status pusher above.

        `connected` is carried for logs/debugging only. The server does NOT
        forward it: AppState._handle_live_event() re-reads the listener's
        live is_connected() when it handles this event (see the
        AD_CONNECTION branch there), because two quick transitions can be
        announced from two different threads (the listener's own, and
        whichever thread called stop()) and so reach this queue in either
        order. An event here only means "the connection changed -- go and
        look"; what is broadcast is always what is true when it is sent.
        """
        live_events.put({"type": "AD_CONNECTION", "connected": bool(connected)})

    ad_ws_listener: AdWsListener | None = None
    if enable_ad_ground_truth_inline:
        ad_ws_listener = ad_listener_factory(
            ad_base_url,
            on_status_change=_make_board_status_pusher("AD_BOARD_STATUS"),
            on_connection_change=_push_ad_connection_change,
        )
        if ad_enabled:
            ad_ws_listener.start()
            log.info(
                "AD ground-truth WebSocket listener starting (ws_url=%s) -- inline auto-attach ON "
                "(--no-ad-ground-truth to disable)",
                ad_ws_listener.ws_url,
            )
        else:
            # Switched off in the config, so do not connect. This was
            # previously read only to decide virtual-camera publishing and
            # never applied to the listener itself, so the oracle came back
            # ON after every restart -- the setting looked like it saved
            # and then silently did not hold.
            #
            # set_enabled(False) rather than simply not starting: it also
            # makes oracle_base_url() return None, which is what every AD
            # path actually gates on. Not starting alone would leave the
            # flag reading as enabled while nothing was connected.
            setter = getattr(ad_ws_listener, "set_enabled", None)
            if setter is not None:
                setter(False)
            log.info(
                "AD ground truth: 'Compare against Autodarts' is off in config -- "
                "not connecting, inline auto-attach OFF until switched on "
                "(ws_url=%s)",
                ad_ws_listener.ws_url,
            )
    else:
        log.info("AD ground-truth inline auto-attach DISABLED (--no-ad-ground-truth)")

    # Built empty here, BEFORE either half starts -- shared by reference
    # (not copied) into both create_app() and the capture thread's own
    # run_capture_loop_body() call below, so a manual dashboard
    # recalibrate and the capture loop's continuous reads are always
    # looking at the SAME object. See module docstring's "Calibration
    # cadence" section and CalibrationStore's own docstring for the full
    # thread-safety argument.
    # snapshot_path (2026-08-26): loads a previously-persisted calibration
    # at construction, before either half of this process starts, so a
    # process restart with no hardware change reuses it directly instead
    # of paying a fresh bootstrap_calibrations() on the next Start -- see
    # CalibrationStore's own docstring's "Durable across process
    # restarts" section.
    calibration_store = CalibrationStore(
        snapshot_path=DEFAULT_CALIBRATION_PACKAGE_ROOT / CURRENT_CALIBRATION_FILENAME
    )
    # Same "built empty here, BEFORE either half starts, shared by
    # reference" reasoning as calibration_store immediately above --
    # opendarts.live.capture_daemon.ResetRequest's own docstring has the full
    # thread-safety argument.
    reset_request = ResetRequest()
    # Same pattern again -- opendarts.live.capture_daemon.CaptureLoopController's
    # own docstring has the full Start/Stop/idle-timeout design this
    # object coordinates between the FastAPI request-handling thread
    # (POST /api/start, /api/stop) and the capture thread built below.
    # **Durable as of 2026-09-03** -- snapshot_path means this loads back
    # whatever an operator last saved via the dashboard's "Camera
    # timeout" dropdown instead of silently reverting to the 15-minute
    # code default on every process restart (a real incident: "it keeps
    # setting to 60 and it keeps resetting to 15... it keeps idling out
    # before anything else" -- the identical class of bug engine_config_
    # store's own 2026-08-14 fix below already closed for engine config).
    controller = CaptureLoopController(
        snapshot_path=DEFAULT_CONFIG_PATH
    )
    # Same "built empty here, BEFORE either half starts, shared by
    # reference" reasoning as calibration_store/reset_request/controller
    # above -- opendarts.live.capture_daemon.EngineConfigStore's own
    # docstring has the full thread-safety argument. **Durable as of
    # 2026-08-14** -- snapshot_path means this loads back whatever an
    # operator last saved via the dashboard's Config tab instead of
    # silently reverting to Apollo-only on every process restart (a
    # real incident: "no engine scoring is showing up" turned out to be
    # exactly this).
    engine_config_store = EngineConfigStore(
        snapshot_path=DEFAULT_CONFIG_PATH
    )
    # Same "built here, BEFORE either half starts, shared by reference"
    # reasoning -- dart_stable_frames is a pure operator preference
    # (dashboard "Detection time"), so it loads back on construction.
    lifecycle_settings_store = LifecycleSettingsStore(
        snapshot_path=DEFAULT_CONFIG_PATH
    )
    # Each display's settings (opendarts/live/displays.py), kept across
    # restarts: a TV comes back looking the way it was set up.
    display_store = DisplayStore(snapshot_path=DEFAULT_CONFIG_PATH)
    # Which dashboard `/` serves, new or classic -- a switch in either one.
    dashboard_choice = DashboardChoice(snapshot_path=DEFAULT_CONFIG_PATH)

    app = create_app(
        package_root=package_root,
        local_hub=hub,
        live_event_queue=live_events,
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
        display_store=display_store,
        dashboard_choice=dashboard_choice,
        ad_ws_listener=ad_ws_listener,
        vcam_set=vcam_set,
        vcam_set_factory=make_virtual_camera_set,
        reprojection_targets_px=reprojection_targets_px,
        throw_capture=throw_capture_service,
    )

    # access_log=False: uvicorn logs one line per HTTP request regardless
    # of log_level (a separate switch) -- the dashboard's own camera-tile
    # auto-refresh hits /api/cameras/{cam}/snapshot.png repeatedly, which
    # drowned out the capture loop's actual state-transition logs in the
    # same console. The capture loop's own events still log normally.
    #
    # timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_S: uvicorn's own
    # default (None) waits FOREVER for existing connections/ASGI tasks
    # (e.g. a browser tab's live /api/events WebSocket) to finish before
    # returning from Server.run() -- see uvicorn/server.py's own
    # Server.shutdown(): `asyncio.wait_for(self._wait_tasks_to_complete(),
    # timeout=self.config.timeout_graceful_shutdown)`. A real client that
    # doesn't close its connection promptly (or never at all) can hang
    # this indefinitely. Bounding it here is real, tested defense-in-depth
    # -- see this module's own docstring ("Clean shutdown") and
    # tests/test_run_product.py for the confirmed root cause this session
    # actually found (AppState._live_event_loop's own blocking queue read,
    # fixed in opendarts/live/server.py) plus this bound closing the
    # originally-hypothesized WebSocket-drain risk too.
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
        # log_config=None: uvicorn's own default LOGGING_CONFIG attaches a
        # separate, colorized, stderr-only handler directly to the
        # "uvicorn"/"uvicorn.error" loggers (uvicorn's WebSocket
        # accept/close chatter logs via "uvicorn.error" --
        # uvicorn/protocols/websockets/websockets_impl.py -- which
        # access_log=False above does NOT touch, that flag only clears
        # "uvicorn.access"). That handler bypasses this project's own
        # opendarts.live.logging_setup root-logger config entirely
        # 2026-08-13: "some of it is green and color coded... others is
        # just a mess -- why". With log_config=None, uvicorn skips
        # installing its own handlers (see uvicorn.config.Config.
        # configure_logging(): the whole dictConfig block is gated on
        # `if self.log_config is not None`), so "uvicorn"/"uvicorn.error"
        # fall back to normal propagate=True and flow into the SAME root
        # handlers opendarts.* loggers use -- one consistent uncolored
        # format, and (a real fix, not just cosmetic) these lines now
        # actually reach the log FILE too, which they never did before.
        log_config=None,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_S,
    )
    server = uvicorn.Server(uvicorn_config)

    # Let long-lived responses end themselves when shutdown starts.
    #
    # An MJPEG stream (/api/cameras/{cam}/stream.mjpg) never finishes on
    # its own, so uvicorn's graceful drain waits the full
    # GRACEFUL_SHUTDOWN_TIMEOUT_S on every open one and then force-cancels
    # the task -- "Cancel N running task(s), timeout graceful shutdown
    # exceeded" plus an ASGI traceback, on every Ctrl-C. Reliable to hit
    # once a second machine consumes the transport streams, because unlike
    # a browser tab those consumers never disconnect on their own.
    #
    # `server.should_exit` rather than components.stop_event: uvicorn
    # installs its own signal handlers for the duration of server.run()
    # (see _install_signal_handlers' own note), so the handler that sets
    # stop_event does not run until uvicorn is already finishing -- long
    # after the drain this needs to shorten.
    app_state = getattr(app.state, "opendarts_state", None)
    if app_state is not None:
        app_state.should_exit_check = lambda: server.should_exit


    def _on_capture_event(event: dict[str, Any]) -> None:
        live_events.put(event)
        # Real capture-loop activity (a trigger-state transition, a
        # package saved) counts as idle-timeout activity too -- the project's
        # own question, answered yes: the activity set includes
        # THROW_DETECTED/TAKEOUT_FINISHED alongside the control
        # actions. See
        # CaptureLoopController's own docstring for the full list of what
        # else counts (Start/Stop/Reset/Calibrate, wired in
        # opendarts/live/server.py's own route handlers).
        controller.touch()

    def _capture_thread_target() -> None:
        """Restructured 2026-08-12 for Start/Stop (see
        CaptureLoopController's own docstring, ARCHITECTURE NOTE 1, for
        the full design this implements): this thread now runs an OUTER
        loop for the process's WHOLE lifetime, blocking (bounded polls)
        on `controller.start_requested` between SESSIONS rather than
        running exactly one session and then exiting. A session ending
        because `also_stop` fired (manual Stop or idle-timeout,
        `stop_event` itself still clear) loops back to waiting for the
        next Start -- does NOT tear down the whole process. A session
        ending because of a genuine crash (or NotImplementedError, the
        one still-defensive legacy case) DOES still bring the whole
        process down, same discipline the pre-Start/Stop version of this
        function always had for any unexpected loop exit."""
        log.info("capture loop thread starting (idle -- waiting for an explicit Start)")
        while not stop_event.is_set():
            if not controller.start_requested.wait(timeout=0.5):
                continue
            if stop_event.is_set():
                break
            log.info("capture loop thread: Start observed -- running a session (shared hub)")
            try:
                run_capture_loop_body(
                    hub=hub,
                    package_root=package_root,
                    poll_interval_s=poll_interval_s,
                                stop_event=stop_event,
                    also_stop=controller.session_stop_event,
                    on_event=_on_capture_event,
                    ad_ws_listener=ad_ws_listener,
                    ad_match_window_sec=ad_window_sec,
                    calibration_store=calibration_store,
                    reset_request=reset_request,
                    engine_config_store=engine_config_store,
                    lifecycle_settings_store=lifecycle_settings_store,
                    reprojection_targets_px=reprojection_targets_px,
                    # The SAME service object the web server holds, so an
                    # automatic oracle-disagreement capture and a
                    # dashboard button dump from one ring and obey one
                    # "only one capture at a time" rule. A second service
                    # here would give the rig two writers racing for the
                    # same disk and two refusal counters neither of which
                    # was the whole truth.
                    throw_capture=throw_capture_service,
                    # Read fresh per session, not once at import: an
                    # operator can change data/config.json and the
                    # next Start picks it up without a restart, the same
                    # way every other setting in that file behaves.
                    store_packages=_read_store_packages(),
                    min_free_disk_gb=_read_min_free_disk_gb(),
                )
            except NotImplementedError as exc:
                log.error(
                    "capture loop stopped -- not fully implemented yet (see "
                    "opendarts/live/capture_daemon.py's own honest status): %s",
                    exc,
                )
                controller.mark_session_ended()
                break # fatal -- fall through to the whole-process shutdown below
            except Exception: # noqa: BLE001 -- a crashed session must not hang the process
                log.exception("capture loop session crashed unexpectedly")
                controller.mark_session_ended()
                break # fatal -- fall through to the whole-process shutdown below
            else:
                controller.mark_session_ended()
                if not stop_event.is_set():
                    log.info(
                        "capture loop thread: session ended (manual Stop or idle-timeout) "
                        "-- idle, waiting for the next Start"
                    )
        log.info("capture loop thread exiting -- signaling shutdown")
        # Whole-process shutdown -- either the outer while loop exited
        # naturally (stop_event was already set) or a session crashed
        # (the `break`s above). An orphaned dashboard backed by a dead
        # capture loop defeats this whole file's point. Safe/idempotent
        # to call even when a signal handler already set these.
        stop_event.set()
        server.should_exit = True

    capture_thread = threading.Thread(
        target=_capture_thread_target, name="opendarts-run-product-capture-loop", daemon=True
    )

    return ProductComponents(
        hub=hub,
        app=app,
        stop_event=stop_event,
        live_events=live_events,
        capture_thread=capture_thread,
        server=server,
        ad_ws_listener=ad_ws_listener,
        calibration_store=calibration_store,
        reset_request=reset_request,
        controller=controller,
        engine_config_store=engine_config_store,
        lifecycle_settings_store=lifecycle_settings_store,
    )


def _force_exit_watchdog_fired(signum: int) -> None:
    """The absolute last resort (see WATCHDOG_FORCE_EXIT_TIMEOUT_S above).
    Deliberately LOUD -- printed AND logged at ERROR, so this is never a
    silent footgun -- an operator (or a log-scraping alert) must be able
    to tell a forced exit happened and why, not just see the process
    quietly vanish. `os._exit()`, not `sys.exit()`: this runs on a daemon
    Timer thread, and the whole point is guaranteeing real process exit
    even if the main thread is stuck inside code that ignores normal
    exception-based unwinding (e.g. blocked inside a C-level thread join,
    the exact confirmed mechanism this session found -- see
    opendarts/live/server.py's AppState._live_event_loop docstring).
    """
    msg = (
        f"opendarts.live.run_product: FORCED EXIT -- shutdown did not complete within "
        f"{WATCHDOG_FORCE_EXIT_TIMEOUT_S:.0f}s of receiving signal {signum}. This is the "
        f"last-resort watchdog (WATCHDOG_FORCE_EXIT_TIMEOUT_S), not normal behavior -- "
        f"something in the clean-shutdown path (uvicorn's own graceful drain, the "
        f"capture-loop thread, or hub.close_all()) is stuck. Forcing process exit now "
        f"so Ctrl-C/SIGTERM always actually works; investigate what was stuck if this "
        f"fires in real use."
    )
    log.error(msg)
    print(msg, file=sys.stderr, flush=True)
    os._exit(1)


def _arm_shutdown_watchdog(components: ProductComponents, signum: int) -> None:
    """Starts the last-resort force-exit timer the moment a stop signal is
    actually handled -- see _force_exit_watchdog_fired's own docstring and
    WATCHDOG_FORCE_EXIT_TIMEOUT_S above. Idempotent: a second SIGINT/
    SIGTERM (or SIGINT-then-SIGTERM) does not restart the clock -- the
    deadline is from the FIRST stop signal, not the most recent one.
    Cancelled by shutdown() once a clean shutdown actually completes (see
    below) so the normal, fast path never triggers this.
    """
    if components.watchdog_timer is not None:
        return
    timer = threading.Timer(WATCHDOG_FORCE_EXIT_TIMEOUT_S, _force_exit_watchdog_fired, args=(signum,))
    timer.daemon = True
    timer.start()
    components.watchdog_timer = timer


def _install_signal_handlers(components: ProductComponents) -> None:
    """Installs SIGINT/SIGTERM handlers in the CURRENT thread -- must be
    called from the main thread (Python's own hard requirement for
    signal.signal()), which is also where main() later calls
    components.server.run(). A separate named function (not a closure
    inlined into main()) specifically so tests/test_run_product.py can
    install these handlers and then deliver a REAL signal
    (os.kill(os.getpid(), signal.SIGINT)) to prove shutdown actually
    works end to end, not just that the code looks right.

    NOTE, confirmed via real reproduction this session: while
    `components.server.run()` (main()'s next call after this) is
    executing, uvicorn's OWN `Server.capture_signals()` installs ITS OWN
    SIGINT/SIGTERM handlers for the duration of that call, silently
    superseding the handler installed here. `_handle_stop_signal` below
    still reliably runs -- uvicorn re-raises whatever signal it captured,
    via `signal.raise_signal()`, in its own `capture_signals()` `finally`
    block, right as `components.server.run()` is about to return -- just
    later than a naive reading of "installs a handler" would suggest.
    This is exactly why the watchdog is armed HERE (inside the handler
    that reacts to the re-raised signal) rather than, say, right before
    `main()` calls `components.server.run()`: this is the real point in
    time closest to where this session's confirmed hang actually begins
    (see opendarts/live/server.py's AppState._live_event_loop docstring).
    """

    def _handle_stop_signal(signum, _frame) -> None:
        log.info("received signal %d -- stopping capture loop + server", signum)
        components.stop_event.set()
        components.server.should_exit = True
        _arm_shutdown_watchdog(components, signum)

    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)


def shutdown(components: ProductComponents, join_timeout_s: float = 5.0) -> None:
    """The one place that stops the capture loop thread, stops uvicorn,
    and closes the shared hub -- EXACTLY once each, in that order. Called
    from main()'s `finally` block after `components.server.run()`
    returns (normal Ctrl-C/SIGTERM handling already set should_exit,
    which is what made server.run() return in the first place; this
    function's job is making sure the OTHER two pieces -- capture
    thread + hub -- are also definitely stopped/closed before the
    process actually exits).
    """
    components.stop_event.set()
    components.server.should_exit = True
    components.capture_thread.join(timeout=join_timeout_s)
    if components.capture_thread.is_alive():
        log.warning(
            "capture loop thread did not stop within %.1fs -- proceeding to close "
            "the camera hub anyway (thread is daemon=True so it cannot block "
            "process exit, but a frame grab racing hub.close_all() below is "
            "possible in this edge case)",
            join_timeout_s,
        )
    if components.ad_ws_listener is not None:
        log.info("stopping AD ground-truth WebSocket listener")
        components.ad_ws_listener.stop()
    log.info("closing shared local camera hub")
    components.hub.close_all()
    if components.watchdog_timer is not None:
        # Clean shutdown actually completed -- disarm the last-resort
        # force-exit timer so it never fires after the fact.
        components.watchdog_timer.cancel()
    log.info("run_product: clean shutdown complete (capture loop stopped, AD "
              "WS listener stopped, server stopped, hub closed -- each exactly once)")


def _build_arg_parser(live_cfg: LiveConfig) -> argparse.ArgumentParser:
    """This entrypoint's CLI, built against an already-loaded config file.

    A separate function from main() purely so the three-layer precedence
    it encodes -- explicit flag beats config file beats module constant --
    is testable without also starting a camera hub, a capture thread and
    a real HTTP server. It takes `live_cfg` as an argument rather than
    loading it: main() already loaded it, and a second load would be a
    second chance for the two to disagree about what the file said.
    """
    parser = argparse.ArgumentParser(
        description=(
            "opendarts combined capture-loop + dashboard process -- ONE shared "
            "camera hub, both pieces in one process. See this module's own "
            "docstring for when to use this vs. the standalone "
            "opendarts.live.capture_daemon / opendarts.live.server entrypoints."
        )
    )
    parser.add_argument(
        "--port",
        type=int,
        default=live_cfg.port if live_cfg.port is not None else DEFAULT_PORT,
        help=(
            "Port to serve the dashboard/API on. Config-file default: "
            "data/config.json's \"port\" key, if present (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default=live_cfg.host if live_cfg.host is not None else DEFAULT_HOST,
        help=(
            "Interface to bind the dashboard/API to. Config-file default: "
            "data/config.json's \"host\" key, if present -- the same "
            "flag-beats-file-beats-constant precedence --port uses "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help="Directory finished throw packages are written under (default: %(default)s)",
    )
    parser.add_argument(
        "--ad-base-url",
        type=str,
        default=live_cfg.ad_base_url if live_cfg.ad_base_url is not None else DEFAULT_AD_BASE,
        help=(
            "Autodarts base URL -- used both by the automatic inline AD "
            "ground-truth WebSocket listener (see --no-ad-ground-truth) and by "
            "the dashboard's manual 'Fetch AD ground truth now' REST button. "
            "Config-file default: data/config.json's \"ad_base_url\" key, "
            "if present (default: %(default)s)"
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
        "--ad-timeout-s",
        type=float,
        default=AD_DEFAULT_TIMEOUT_SEC,
        help=(
            "Timeout for the dashboard's manual AD ground-truth REST refresh "
            "(the inline WebSocket listener has its own separate connect "
            "timeout, unaffected by this flag -- see AdWsListener) (default: "
            "%(default)s)"
        ),
    )
    parser.add_argument(
        "--camera-url",
        action="append",
        default=[],
        metavar="URL",
        help=(
            "Read this camera slot from a stream URL instead of local "
            "hardware. Repeat once per slot, in slot order: --camera-url A "
            "--camera-url B --camera-url C. Pass the literal word 'local' "
            "to leave that slot on its hardware device while giving a URL "
            "to another. Overrides camera_urls in the config file. A bare "
            "base URL such as http://rig:8420 expands to that rig's three "
            "camera streams."
        ),
    )
    parser.add_argument(
        "--no-ad-ground-truth",
        action="store_true",
        default=False,
        help=(
            "Disable the automatic inline AD ground-truth capture (the "
            "persistent WebSocket listener that matches each just-saved throw "
            "package against AD's own live event stream). The dashboard's "
            "manual 'Fetch AD ground truth now' REST button stays available "
            "regardless -- this only disables the automatic path."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    # Real machine-local config file, loaded ONLY here (the CLI
    # entrypoint) -- see opendarts.live.config's own module docstring for
    # the full reasoning. A missing/malformed file degrades to an
    # all-None/empty LiveConfig -- see load_live_config()'s own
    # docstring -- so this never changes behavior for anyone who hasn't
    # created data/config.json. An explicit CLI flag still wins over
    # the config file's value, which still wins over the hardcoded module
    # constant -- three real layers, most-specific wins. See
    # _build_arg_parser() above, which is where those layers are wired.
    from opendarts.live.config import load_live_config

    live_cfg = load_live_config()

    parser = _build_arg_parser(live_cfg)
    args = parser.parse_args(argv)

    from opendarts.live.logging_setup import configure_console_and_file_logging

    log_path = configure_console_and_file_logging("run_product")
    log.info("=" * 60)
    log.info("opendarts run_product -- starting COMBINED capture loop + dashboard")
    log.info("(one process, one shared camera hub -- see module docstring for")
    log.info(" when to use this vs. the standalone capture_daemon/server entrypoints)")
    log.info("logging to console AND to %s", log_path)
    log.info("stop with Ctrl-C")
    log.info("=" * 60)

    # Bound OpenCV's worker pool BEFORE anything opens a camera or touches a
    # frame -- an unbounded pool cost 62.8% of this process's total CPU at
    # idle on the real rig. `live_cfg` was loaded at the top of main(); a
    # missing key means "no override", which still applies the measured
    # default rather than leaving OpenCV's own choice in place. See
    # opendarts.live.cv2_threads for the full before/after.
    from opendarts.live.cv2_threads import apply_cv2_thread_limit

    apply_cv2_thread_limit(live_cfg.cv2_num_threads)

    # Dependency probe at startup (owner request, 2026-09-12): learn once
    # what this rig has, persist it to config's "capabilities"
    # section, and say it in the startup banner -- so "no ffmpeg" is one
    # line here, not a per-calibration failure discovered mid-session.
    caps = capabilities.snapshot()
    log.info("external tools on this rig: %s",
             ", ".join(f"{name}={'yes' if entry.get('present') else 'no'}"
                       for name, entry in sorted(caps.get("tools", {}).items())))

    try:
        import uvicorn # noqa: F401 -- import check only; _build_components does the real import
    except ImportError:
        log.error(
            "uvicorn is not installed in this environment's Python -- "
            "run `pip install -r requirements.txt` (fastapi + uvicorn are "
            "already listed there) before starting opendarts.live.run_product."
        )
        return 1

    try:
        components = _build_components(
            package_root=args.package_root,
            host=args.host,
            port=args.port,
            poll_interval_s=POLL_INTERVAL_SECONDS,
            ad_base_url=args.ad_base_url,
            ad_window_sec=args.ad_window_sec,
            ad_timeout_s=args.ad_timeout_s,
            enable_ad_ground_truth_inline=not args.no_ad_ground_truth,
            reprojection_targets_px=live_cfg.reprojection_targets_px or None,
            camera_configs=_camera_configs_from_live_config(live_cfg),
            camera_devices=getattr(live_cfg, "camera_devices", None),
            # CLI wins over the file, same precedence as every other
            # flag here: a --camera-url given on the command line is a
            # deliberate override of whatever the rig has saved.
            camera_urls=_resolve_camera_urls(
                args.camera_url, getattr(live_cfg, "camera_urls", None)
            ),
            frame_ring_seconds=getattr(live_cfg, "frame_ring_seconds", None),
            frame_ring_max_gb=getattr(live_cfg, "frame_ring_max_gb", None),
            video_record_mode=getattr(live_cfg, "video_record_mode", None),
        )
    except RuntimeError as exc:
        log.error("cannot start: %s", exc)
        return 1

    _install_signal_handlers(components)

    log.info("starting capture loop thread")
    components.capture_thread.start()

    display_host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    # Same "don't make guess if it's running" discipline as
    # capture_daemon.py / server.py's own startup banners.
    print("=" * 60)
    print(f"Listening on http://{display_host}:{args.port} -- open this in a browser")
    print(f"(bound to {args.host}:{args.port}; package root: {args.package_root})")
    print("(frame source: local direct camera -- ONE shared hub for capture loop + dashboard)")
    print("(capture loop + dashboard both running together in this single process)")
    # Reflect the EFFECTIVE state, not just the CLI flag: the listener is
    # built without --no-ad-ground-truth but stays disconnected while the
    # config has the oracle off (oracle_base_url() is None then).
    _listener = getattr(components, "ad_ws_listener", None)
    _oracle_url = getattr(_listener, "oracle_base_url", None)
    _oracle_on = _listener is not None and (
        _oracle_url() is not None if callable(_oracle_url) else True
    )
    if args.no_ad_ground_truth:
        ad_status = "AD ground truth: inline auto-attach OFF (--no-ad-ground-truth) -- manual dashboard refresh still available"
    elif _oracle_on:
        ad_status = f"AD ground truth: inline auto-attach ON ({args.ad_base_url})"
    else:
        ad_status = (
            "AD ground truth: inline auto-attach OFF ('Compare against Autodarts' "
            "is off in config -- switch it on from the Config tab)"
        )
    print(f"({ad_status})")
    print("=" * 60)
    sys.stdout.flush()

    try:
        components.server.run()
    finally:
        shutdown(components)

    return 0


if __name__ == "__main__":
    sys.exit(main())
