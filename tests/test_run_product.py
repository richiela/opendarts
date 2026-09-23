"""Tests for opendarts/live/run_product.py -- the combined capture-loop +
dashboard entrypoint that shares ONE LocalCameraHub between both halves.

HONEST SCOPE for most of this file, same discipline as
tests/test_capture_daemon.py and tests/test_local_capture.py: these are
LOGIC/WIRING tests. Nothing in the bulk of this file opens a real camera
or binds a real uvicorn server to a real port -- `uvicorn.Server` objects
are constructed (cheap, no socket work) but `.run()` is never called
against a real port; instead `ProductComponents.server` is either a
real-but-never-run() `uvicorn.Server` (fine to construct) or a
lightweight fake standing in for one, and
capture_daemon.run_capture_loop_body() itself is monkeypatched to a small
fake loop so no real calibration/frame-fetch/scoring machinery needs to
run either -- that's already covered by tests/test_capture_daemon.py.

What IS verified for real by that bulk of the file:
  - startup opens EXACTLY one camera hub (FakeHub, same pattern
    tests/test_capture_daemon.py's own FakeHub establishes)
  - that hub is the SAME instance shared with the FastAPI app (via
    create_app()'s local_hub= seam) and with the capture-loop thread
  - both pieces (capture thread + app/server wiring) actually start
  - shutdown is clean: capture thread stopped, hub closed EXACTLY once,
    no leaked/hung thread left behind
  - a REAL delivered OS signal (SIGINT and SIGTERM, via
    os.kill(os.getpid(), ...)) actually triggers that same clean
    shutdown path -- not just "the code looks right"
  - a capture-loop crash also brings the server down (should_exit set),
    so an operator never ends up staring at a dashboard backed by a
    dead capture loop

**REAL-PROCESS shutdown-hang regression test, 2026-08-12 -- see the
`TestRealProcessShutdownWithOpenWebsocket` class at the bottom of this
file.** All the tests above this point pass `.server` as either an
un-run `uvicorn.Server` or a fake -- which is EXACTLY why the real
shutdown hang hit live on the rig ("received sig 2.. stopping
capture loop. And never exited") was never caught by this suite before
now: `uvicorn.Server.run()` was never actually called, so nothing here
could see uvicorn's own real signal-handler-override behavior, its real
graceful-shutdown wait, or -- the confirmed actual root cause, see
opendarts/live/server.py's AppState._live_event_loop docstring -- the real
`asyncio.run()` cleanup path (`shutdown_default_executor()`) hanging on
an orphaned worker thread. `TestRealProcessShutdownWithOpenWebsocket`
closes that gap: it launches `opendarts.live.run_product.main()` as a REAL
OS subprocess (via tests/_run_product_subprocess_harness.py, with only
cv2.VideoCapture + run_capture_loop_body swapped for fakes -- everything
else, including the real bound uvicorn port and real signal handling, is
the genuine article), opens a REAL WebSocket connection to /api/events
(the `websockets` library, simulating exactly what a browser tab left
open on the dashboard does), sends the subprocess a REAL SIGINT, and
asserts the process actually exits within a bounded time -- not just
that in-process flags got set. This is the test that would have caught
the original bug (confirmed by literally reverting the
opendarts/live/server.py fix locally and re-running it during this
session's own verification -- it hangs and the test fails/times out, as
expected).
"""
from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import opendarts.live.run_product as run_product


def _cameras(hub):
    """The hub itself. Kept as a named seam, not as an unwrap.

    Until 2026-09-16 `build_components` returned a wrapper composing a
    local hub and a remote one, so these tests had to reach through
    `hub._local` to find the object their `hub_factory` built. There is
    one hub now -- a slot reads a device or a stream, and the hub is the
    same object either way -- so there is nothing to unwrap. Left in place
    because every call site reads better as "the cameras" than as "the
    hub", and because a future composition would have exactly one place to
    teach.
    """
    return hub



REPO_ROOT = Path(__file__).resolve().parent.parent
@pytest.fixture()
def package_root(tmp_path):
    return tmp_path / "pkgroot"


# ---------------------------------------------------------------------------
# FakeHub -- mirrors tests/test_capture_daemon.py's own FakeHub exactly
# (records construction/open/close, never touches cv2) so this file
# doesn't depend on importing test internals across files.
# ---------------------------------------------------------------------------


class FakeHub:
    """Stands in for the real CameraHub.

    `urls=` is part of the signature because it is part of the real hub's:
    a slot reads a device or another machine's stream, and `build_hub`
    passes the per-slot routing to whatever factory it is given. A fake
    that did not accept it would make every test here pass for the wrong
    reason -- it would prove build_components calls SOMETHING, not that it
    calls the hub the product actually builds.
    """

    instances: list["FakeHub"] = []

    def __init__(self, configs=None, frame_sink=None, urls=None) -> None:
        self.frame_sink = frame_sink
        self.urls_seen = urls
        self.open_calls = 0
        self.close_calls = 0
        self._open_ok = True
        FakeHub.instances.append(self)

    def open_all(self):
        self.open_calls += 1
        return [self._open_ok, self._open_ok, self._open_ok]

    def close_all(self):
        self.close_calls += 1

    def status_report(self) -> str:
        return "fake hub status"


class AllFailHub(FakeHub):
    def __init__(self, configs=None, frame_sink=None, urls=None) -> None:
        super().__init__(configs, frame_sink, urls)
        self._open_ok = False

    def open_all(self):
        self.open_calls += 1
        return [False, False, False]


@pytest.fixture(autouse=True)
def _reset_fake_hub_instances():
    FakeHub.instances = []
    AllFailHub.instances = []
    yield
    FakeHub.instances = []
    AllFailHub.instances = []


@contextlib.contextmanager
def _restored_signal_handlers():
    """Any test that installs real signal handlers via
    run_product._install_signal_handlers must restore whatever was there
    before it -- otherwise a later test (or pytest's own SIGINT handling
    for Ctrl-C during a real test run) inherits this module's handler."""
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


def _wait_until(predicate, timeout_s: float = 2.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


# ---------------------------------------------------------------------------
# _build_components -- exactly one hub, shared with app + capture thread
# ---------------------------------------------------------------------------


def test_build_components_constructs_exactly_one_hub_but_does_not_open_it(package_root):
    """CHANGED 2026-08-12 -- _build_components() used to open
    the hub eagerly; now it only CONSTRUCTS it (open_calls == 0) and
    leaves opening to an explicit POST /api/start -- the process does
    not probe cameras at startup."""
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )
    assert len(FakeHub.instances) == 1
    assert FakeHub.instances[0].open_calls == 0
    assert _cameras(components.hub) is FakeHub.instances[0]


def test_build_components_shares_the_same_hub_with_the_app(package_root):
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )
    # create_app()'s local_hub= seam -- the SAME instance, not a second one.
    assert components.app.state.opendarts_state.hub is components.hub


def test_build_components_wires_the_live_event_queue_into_the_app(package_root):
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )
    assert components.app.state.opendarts_state.live_event_queue is components.live_events
    assert components.app.state.opendarts_state.live_events_enabled is True


def test_build_components_shares_one_calibration_store_with_the_app(package_root):
    """The actual correctness requirement: a manual
    dashboard recalibrate must change what the capture loop scores
    against, which is only possible if the app's AppState and the capture
    loop are handed the SAME opendarts.live.capture_daemon.CalibrationStore
    object, not two independently-built ones."""
    from opendarts.live.capture_daemon import CalibrationStore

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )
    assert isinstance(components.calibration_store, CalibrationStore)
    assert components.app.state.opendarts_state.calibration_store is components.calibration_store
    assert components.app.state.opendarts_state.calibration_store is not None


def test_build_components_never_probes_cameras_even_with_an_always_failing_hub(package_root):
    """CHANGED 2026-08-12 -- see the sibling "constructs but does not
    open" test above for the full writeup: _build_components() no longer
    raises RuntimeError for zero cameras at ALL (it never opens anything
    to find out) -- that failure mode moved to AppState.start_capture()
    (POST /api/start), see tests/test_live_server.py's own
    test_api_start_reports_ok_false_when_zero_cameras_open. Proven here
    with a hub whose open_all() would always fail: building succeeds
    anyway, and open_all()/close_all() are never even called."""
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=AllFailHub,
    )
    assert len(FakeHub.instances) == 1 # AllFailHub is a FakeHub subclass
    assert FakeHub.instances[0].open_calls == 0
    assert FakeHub.instances[0].close_calls == 0
    assert _cameras(components.hub) is FakeHub.instances[0]


# ---------------------------------------------------------------------------
# AD ground truth -- inline WebSocket-buffer attach, added 2026-08-12.
#
# Real live finding this exists for: the REST /api/state/detections list
# only covers the current visit and is not durable (2026-08-12) -- see opendarts/live/ad_ws_listener.py's own
# module docstring for the full design and opendarts/live/capture_daemon.py's
# own tests for the actual match/attach logic (this file only proves the
# LIFECYCLE/PLUMBING: the listener is built, started, threaded into the
# capture loop, and stopped exactly once, all without ever touching a
# real socket -- FakeAdWsListener below is a pure in-memory stand-in).
# ---------------------------------------------------------------------------


class FakeAdWsListener:
    instances: list["FakeAdWsListener"] = []

    def __init__(self, base_url, on_status_change=None, on_connection_change=None) -> None:
        self.base_url = base_url
        self.ws_url = f"ws://fake/{base_url}"
        self.on_connection_change = on_connection_change
        self.start_calls = 0
        self.stop_calls = 0
        # 2026-08-14: mirrors AdWsListener's real on_status_change= param
        # (see opendarts/live/run_product.py's _make_board_status_pusher())
        # -- stored so a test can call it directly to simulate a status
        # change, but not exercised by the lifecycle-only tests below.
        self.on_status_change = on_status_change
        FakeAdWsListener.instances.append(self)

    def start(self) -> None:
        self.start_calls += 1

    def stop(self, timeout: float = 3.0) -> None:
        self.stop_calls += 1

    def match(self, opendarts_captured_at_utc, *, window_sec):
        raise AssertionError("match() should never be called by lifecycle-only tests")


@pytest.fixture(autouse=True)
def _reset_fake_ad_listener_instances():
    FakeAdWsListener.instances = []
    yield
    FakeAdWsListener.instances = []


def test_build_components_defaults_to_no_ad_listener_at_all(package_root):
    """Safe-by-default at THIS function level (opt-in here, opt-out at
    the CLI level -- see _build_components()'s own docstring): any caller
    that doesn't explicitly ask for AD ground truth (every OTHER test in
    this file, and any future one) must never start a real background
    WebSocket connection attempt by surprise. Confirms the DEFAULT here
    is off, not merely that it CAN be turned off."""
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        ad_listener_factory=FakeAdWsListener,
    )
    assert components.ad_ws_listener is None
    assert FakeAdWsListener.instances == []


def test_build_components_builds_the_ad_listener_and_starts_it_only_when_configured(
    package_root, monkeypatch,
):
    """Wiring and starting are separate, and only the first is unconditional.

    This test used to pass `enable_ad_ground_truth_inline=True` with no
    config at all and assert start_calls == 1 -- which passed only because
    an absent `ad_enabled` key used to mean ON. It was really pinning the
    default while appearing to pin the wiring, so when the default flipped
    on 2026-09-16 it failed for a reason its name did not describe.

    The listener is still always CONSTRUCTED when the feature is wired in,
    because the dashboard's toggle needs something to talk to in order to
    switch it on later. It is only STARTED when the config asks for it.
    """
    import opendarts.live.config as config_mod

    real = config_mod.read_config_section

    def with_ad_enabled(value):
        def fake(section, path=None):
            if section == "ad_enabled":
                return value
            return real(section) if path is None else real(section, path)
        return fake

    monkeypatch.setattr(config_mod, "read_config_section", with_ad_enabled(None))
    FakeAdWsListener.instances = []
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        ad_base_url="http://fake-ad:3180",
        enable_ad_ground_truth_inline=True,
        ad_listener_factory=FakeAdWsListener,
    )
    assert components.ad_ws_listener is not None, "the toggle needs a listener to talk to"
    assert components.ad_ws_listener.base_url == "http://fake-ad:3180"
    assert components.ad_ws_listener.start_calls == 0, (
        "no ad_enabled key means off -- it must not connect"
    )

    monkeypatch.setattr(config_mod, "read_config_section", with_ad_enabled(True))
    FakeAdWsListener.instances = []
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        ad_base_url="http://fake-ad:3180",
        enable_ad_ground_truth_inline=True,
        ad_listener_factory=FakeAdWsListener,
    )
    assert components.ad_ws_listener.start_calls == 1
    assert len(FakeAdWsListener.instances) == 1


# ---------------------------------------------------------------------------
# Board-status indicator light plumbing -- 2026-08-14 (Scoring tab AD dot;
# the second external board's light was removed once it became a real
# registry engine -- only AD has status now). _build_components() wires an
# on_status_change= callback (built by its own local
# _make_board_status_pusher(kind) closure) into the AD listener it
# constructs -- the callback pushes {"type": kind, "status": ...} onto the
# SAME live_events queue the capture loop's own events already use, no new
# polling and no second queue. This test calls the constructed fake listener's own stored
# .on_status_change directly -- simulating exactly what the real
# AdWsListener would do from its own background WS-receive thread -- and
# asserts the exact dict landed on components.live_events (a
# queue.SimpleQueue, drained here via .get_nowait() so an unexpected SECOND
# queued event, not just the expected one, would also be caught by a
# stray leftover on a later .get_nowait() in the same test).
# ---------------------------------------------------------------------------


def test_ad_status_change_callback_pushes_the_expected_event_onto_live_events(package_root):
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        ad_base_url="http://fake-ad:3180",
        enable_ad_ground_truth_inline=True,
        ad_listener_factory=FakeAdWsListener,
    )
    ad_listener = FakeAdWsListener.instances[0]
    assert ad_listener.on_status_change is not None

    ad_listener.on_status_change("ready")

    assert components.live_events.get_nowait() == {"type": "AD_BOARD_STATUS", "status": "ready"}


def test_ad_connection_change_callback_pushes_an_ad_connection_event(package_root):
    """The listener's connect/disconnect announcements reach the same
    live_events queue as its board-status ones -- the server re-reads the
    live value when it handles the event (see server.py's AD_CONNECTION
    branch), so the queued `connected` is informational."""
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        ad_base_url="http://fake-ad:3180",
        enable_ad_ground_truth_inline=True,
        ad_listener_factory=FakeAdWsListener,
    )
    ad_listener = FakeAdWsListener.instances[0]
    assert ad_listener.on_connection_change is not None

    ad_listener.on_connection_change(True)
    ad_listener.on_connection_change(False)

    assert components.live_events.get_nowait() == {"type": "AD_CONNECTION", "connected": True}
    assert components.live_events.get_nowait() == {"type": "AD_CONNECTION", "connected": False}


def test_ad_listener_is_never_constructed_with_a_status_change_callback_when_disabled(package_root):
    """Default-off at this function level, same as ad_ws_listener itself
    being None -- confirms there is no dangling listener/callback of any
    kind for a caller that never enabled AD ground truth at all (every
    OTHER test in this file, and any future one)."""
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        ad_listener_factory=FakeAdWsListener,
    )
    assert components.ad_ws_listener is None
    assert FakeAdWsListener.instances == []


def test_build_components_always_wires_ad_base_url_into_create_app_for_the_manual_button(
    package_root,
):
    """The dashboard's existing manual REST refresh button
    (opendarts/live/server.py) must get the configured ad_base_url/
    ad_window_sec/ad_timeout_s regardless of enable_ad_ground_truth_inline
    -- this is a pre-existing gap this session's work also fixed (run_product's
    own create_app() call never threaded these through before, so the
    dashboard's manual button silently always used server.py's own
    hardcoded defaults no matter what run_product was configured with)."""
    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        ad_base_url="http://fake-ad:9999",
        ad_window_sec=3.5,
        ad_timeout_s=2.5,
        enable_ad_ground_truth_inline=False, # inline off -- manual button still wired
    )
    state = components.app.state.opendarts_state
    assert state.ad_base_url == "http://fake-ad:9999"
    assert state.ad_window_sec == 3.5
    assert state.ad_timeout_s == 2.5


def test_shutdown_stops_ad_listener_exactly_once(package_root, monkeypatch):
    def fake_loop_body(*, hub, package_root, poll_interval_s, stop_event, on_event=None, **_kwargs):
        while not stop_event.is_set():
            stop_event.wait(0.01)

    monkeypatch.setattr(run_product, "run_capture_loop_body", fake_loop_body)

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        enable_ad_ground_truth_inline=True,
        ad_listener_factory=FakeAdWsListener,
    )
    components.capture_thread.start()
    components.controller.request_start() # a real session must be running for shutdown() to stop
    assert _wait_until(components.capture_thread.is_alive, timeout_s=1.0)

    run_product.shutdown(components, join_timeout_s=2.0)

    assert components.ad_ws_listener.stop_calls == 1
    # Calling shutdown() again must not double-stop in a way that raises
    # (mirrors the existing hub double-close test's own discipline).
    run_product.shutdown(components, join_timeout_s=1.0)
    assert components.ad_ws_listener.stop_calls == 2


def test_shutdown_is_a_noop_for_ad_listener_when_none_was_built(package_root, monkeypatch):
    def fake_loop_body(*, hub, package_root, poll_interval_s, stop_event, on_event=None, **_kwargs):
        while not stop_event.is_set():
            stop_event.wait(0.01)

    monkeypatch.setattr(run_product, "run_capture_loop_body", fake_loop_body)

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )
    assert components.ad_ws_listener is None
    components.capture_thread.start()
    components.controller.request_start()
    run_product.shutdown(components, join_timeout_s=2.0) # must not raise
    assert _cameras(components.hub).close_calls == 1


def test_capture_thread_threads_ad_listener_and_window_into_run_capture_loop_body(
    package_root, monkeypatch
):
    seen: dict = {}

    def fake_loop_body(*, hub, package_root, poll_interval_s, stop_event, on_event=None, **kwargs):
        seen.update(kwargs)
        while not stop_event.is_set():
            stop_event.wait(0.01)

    monkeypatch.setattr(run_product, "run_capture_loop_body", fake_loop_body)

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        ad_window_sec=9.0,
        enable_ad_ground_truth_inline=True,
        ad_listener_factory=FakeAdWsListener,
    )
    components.capture_thread.start()
    # CHANGED 2026-08-12: the capture thread now waits for an explicit
    # Start before ever calling run_capture_loop_body() at all (see
    # CaptureLoopController's own docstring) -- request one, mirroring
    # what a real POST /api/start does.
    components.controller.request_start()
    assert _wait_until(lambda: "ad_ws_listener" in seen, timeout_s=1.0)

    assert seen["ad_ws_listener"] is components.ad_ws_listener
    assert seen["ad_match_window_sec"] == 9.0

    components.stop_event.set()
    components.capture_thread.join(timeout=2.0)


def test_capture_thread_threads_the_same_calibration_store_into_run_capture_loop_body(
    package_root, monkeypatch
):
    """The other half of test_build_components_shares_one_calibration_
    store_with_the_app: not just built and attached to the app, but
    actually PASSED to the real run_capture_loop_body() call the capture
    thread makes -- the object the capture loop reads calibrations from
    at runtime is provably the SAME one a manual dashboard recalibrate
    would write to."""
    seen: dict = {}

    def fake_loop_body(*, hub, package_root, poll_interval_s, stop_event, on_event=None, **kwargs):
        seen.update(kwargs)
        while not stop_event.is_set():
            stop_event.wait(0.01)

    monkeypatch.setattr(run_product, "run_capture_loop_body", fake_loop_body)

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )
    components.capture_thread.start()
    components.controller.request_start() # see the sibling AD-listener test's own comment above
    assert _wait_until(lambda: "calibration_store" in seen, timeout_s=1.0)

    assert seen["calibration_store"] is components.calibration_store
    assert seen["calibration_store"] is components.app.state.opendarts_state.calibration_store

    components.stop_event.set()
    components.capture_thread.join(timeout=2.0)


def test_main_wires_ad_ground_truth_cli_flags_through(package_root, monkeypatch):
    seen: dict = {}

    def fake_build_components(*, package_root, host, port, poll_interval_s, **kwargs):
        seen.update(kwargs)
        hub = FakeHub()
        hub.open_all()
        components = run_product.ProductComponents(
            hub=hub,
            app=object(),
            stop_event=threading.Event(),
            live_events=__import__("queue").SimpleQueue(),
            capture_thread=threading.Thread(target=lambda: None, name="noop"),
            server=_FakeServer(),
        )
        return components

    monkeypatch.setattr(run_product, "_build_components", fake_build_components)
    monkeypatch.setattr(run_product, "_install_signal_handlers", lambda components: None)

    with _restored_signal_handlers():
        rc = run_product.main(
            [
                "--port", "9999",
                "--host", "127.0.0.1",
                "--package-root", str(package_root),
                "--ad-base-url", "http://custom-ad:1234",
                "--ad-window-sec", "5.5",
                "--ad-timeout-s", "2.2",
            ]
        )
    assert rc == 0
    assert seen["ad_base_url"] == "http://custom-ad:1234"
    assert seen["ad_window_sec"] == 5.5
    assert seen["ad_timeout_s"] == 2.2
    assert seen["enable_ad_ground_truth_inline"] is True # on by default


def test_main_no_ad_ground_truth_flag_disables_inline_only(package_root, monkeypatch):
    seen: dict = {}

    def fake_build_components(*, package_root, host, port, poll_interval_s, **kwargs):
        seen.update(kwargs)
        hub = FakeHub()
        hub.open_all()
        components = run_product.ProductComponents(
            hub=hub,
            app=object(),
            stop_event=threading.Event(),
            live_events=__import__("queue").SimpleQueue(),
            capture_thread=threading.Thread(target=lambda: None, name="noop"),
            server=_FakeServer(),
        )
        return components

    monkeypatch.setattr(run_product, "_build_components", fake_build_components)
    monkeypatch.setattr(run_product, "_install_signal_handlers", lambda components: None)

    with _restored_signal_handlers():
        rc = run_product.main(
            ["--package-root", str(package_root), "--no-ad-ground-truth"]
        )
    assert rc == 0
    assert seen["enable_ad_ground_truth_inline"] is False
    # ad_base_url is STILL a real value here -- disabling is scoped to the
    # inline auto-attach only; _build_components() itself is responsible
    # for always wiring a real ad_base_url into create_app() for the
    # dashboard's manual button regardless (see the test for that above).
    assert seen["ad_base_url"] == run_product.DEFAULT_AD_BASE


# ---------------------------------------------------------------------------
# Both pieces actually start; capture-loop crash brings the server down too
# ---------------------------------------------------------------------------


def test_capture_thread_actually_starts_and_stops_on_stop_event(package_root, monkeypatch):
    def fake_loop_body(*, hub, package_root, poll_interval_s, stop_event, on_event=None, **_kwargs):
        assert hub is components.hub # same shared hub reaches the loop body
        while not stop_event.is_set():
            stop_event.wait(0.01)

    monkeypatch.setattr(run_product, "run_capture_loop_body", fake_loop_body)

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )

    components.capture_thread.start()
    components.controller.request_start() # a real session must be running to observe it stop
    assert _wait_until(components.capture_thread.is_alive, timeout_s=1.0)

    components.stop_event.set()
    components.capture_thread.join(timeout=2.0)
    assert not components.capture_thread.is_alive()
    # Well-behaved stop via stop_event -- the loop's own finally still
    # marks should_exit too (idempotent), proving the "bring the other
    # piece down too" wiring runs on every exit path, not just crashes.
    assert components.server.should_exit is True


def test_capture_thread_crash_sets_stop_event_and_server_should_exit(package_root, monkeypatch):
    def crashing_loop_body(**kwargs):
        raise RuntimeError("simulated capture-loop crash")

    monkeypatch.setattr(run_product, "run_capture_loop_body", crashing_loop_body)

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )

    components.capture_thread.start()
    components.controller.request_start() # the crash only happens once a session actually starts
    components.capture_thread.join(timeout=2.0)

    assert not components.capture_thread.is_alive()
    assert components.stop_event.is_set()
    assert components.server.should_exit is True


# ---------------------------------------------------------------------------
# shutdown() -- capture thread stopped, hub closed EXACTLY once
# ---------------------------------------------------------------------------


def test_shutdown_stops_thread_and_closes_hub_exactly_once(package_root, monkeypatch):
    def fake_loop_body(*, hub, package_root, poll_interval_s, stop_event, on_event=None, **_kwargs):
        while not stop_event.is_set():
            stop_event.wait(0.01)

    monkeypatch.setattr(run_product, "run_capture_loop_body", fake_loop_body)

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )
    components.capture_thread.start()
    components.controller.request_start()
    assert _wait_until(components.capture_thread.is_alive, timeout_s=1.0)

    run_product.shutdown(components, join_timeout_s=2.0)

    assert not components.capture_thread.is_alive()
    assert _cameras(components.hub).close_calls == 1
    assert components.server.should_exit is True

    # Calling shutdown() logic a second time must not double-close in a
    # way that raises -- FakeHub.close_all() is idempotent-safe to call
    # again (real LocalCameraHub.close_all() is too, see local_capture.py:
    # close_all() clears self._caps, so a second call is just a no-op
    # loop over an empty dict). Not double-invoked by shutdown() itself
    # in normal operation (main() calls it exactly once), but this
    # confirms a defensive re-call wouldn't corrupt state either.
    run_product.shutdown(components, join_timeout_s=1.0)
    assert _cameras(components.hub).close_calls == 2


# ---------------------------------------------------------------------------
# REAL signal delivery -- os.kill(os.getpid(), SIGINT/SIGTERM), not just
# calling the handler function directly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_real_signal_triggers_clean_shutdown(package_root, monkeypatch, sig):
    def fake_loop_body(*, hub, package_root, poll_interval_s, stop_event, on_event=None, **_kwargs):
        while not stop_event.is_set():
            stop_event.wait(0.01)

    monkeypatch.setattr(run_product, "run_capture_loop_body", fake_loop_body)

    components = run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
    )

    with _restored_signal_handlers():
        run_product._install_signal_handlers(components)
        components.capture_thread.start()
        components.controller.request_start()
        assert _wait_until(components.capture_thread.is_alive, timeout_s=1.0)

        # The actual point of this test: a REAL OS signal, not a direct
        # Python function call to the handler.
        os.kill(os.getpid(), sig)

        assert _wait_until(lambda: components.stop_event.is_set(), timeout_s=2.0)
        assert components.server.should_exit is True

        # Finish the same clean-shutdown path main() would use in its
        # finally block, and confirm it actually completes: thread
        # stopped, hub closed exactly once, nothing hung.
        run_product.shutdown(components, join_timeout_s=2.0)
        assert not components.capture_thread.is_alive()
        assert _cameras(components.hub).close_calls == 1


# ---------------------------------------------------------------------------
# main() -- CLI args thread through, and shutdown() runs even though
# uvicorn.Server.run() is never actually invoked for real here (a fake
# server stands in, matching this suite's "no real bound port" scope).
# ---------------------------------------------------------------------------


class _FakeServer:
    def __init__(self) -> None:
        self.should_exit = False
        self.run_calls = 0

    def run(self) -> None:
        self.run_calls += 1
        # Simulate should_exit already being true by the time run()
        # would normally return (as if a signal had already fired) --
        # main() must still reach its finally: shutdown(components).
        self.should_exit = True


def test_main_wires_cli_args_and_runs_clean_shutdown(package_root, monkeypatch):
    seen: dict = {}

    def fake_build_components(*, package_root, host, port, poll_interval_s, **_kwargs):
        seen.update(
            package_root=package_root,
            host=host,
            port=port,
            poll_interval_s=poll_interval_s,
        )
        seen.update(_kwargs)
        hub = FakeHub()
        hub.open_all()
        components = run_product.ProductComponents(
            hub=hub,
            app=object(),
            stop_event=threading.Event(),
            live_events=__import__("queue").SimpleQueue(),
            capture_thread=threading.Thread(target=lambda: None, name="noop"),
            server=_FakeServer(),
        )
        return components

    monkeypatch.setattr(run_product, "_build_components", fake_build_components)
    monkeypatch.setattr(run_product, "_install_signal_handlers", lambda components: None)

    with _restored_signal_handlers():
        rc = run_product.main(
            [
                "--port",
                "9999",
                "--host",
                "127.0.0.1",
                "--package-root",
                str(package_root),
            ]
        )

    assert rc == 0
    assert seen["port"] == 9999
    assert seen["host"] == "127.0.0.1"
    assert seen["package_root"] == package_root
    assert FakeHub.instances[-1].close_calls == 1 # shutdown() ran in main()'s finally


def test_main_returns_1_when_no_camera_opens(package_root, monkeypatch):
    def fake_build_components(**kwargs):
        raise RuntimeError("no camera opened locally -- simulated for this test")

    monkeypatch.setattr(run_product, "_build_components", fake_build_components)

    rc = run_product.main(["--package-root", str(package_root)])
    assert rc == 1


# ---------------------------------------------------------------------------
# REAL-PROCESS shutdown-hang regression test -- see this module's own
# docstring above for why this exists (the confirmed bug, "received sig 2
# .. never exited", was invisible to every test above this point because
# none of them ever call the real uvicorn.Server.run() against a real
# bound port). Launches tests/_run_product_subprocess_harness.py as a
# genuine OS subprocess, opens a genuine WebSocket connection to it, sends
# a genuine SIGINT, and asserts the process actually exits within a
# bounded time.
# ---------------------------------------------------------------------------

websockets = pytest.importorskip("websockets", reason="real-process shutdown test needs the websockets client library")
import websockets.sync.client # noqa: E402 -- after importorskip, matches this file's existing import-order style

HARNESS_PATH = Path(__file__).resolve().parent / "_run_product_subprocess_harness.py"

# Comfortably above the real, measured shutdown time this session's fix
# achieves (~0.5s against this same harness) and comfortably BELOW
# opendarts.live.run_product.WATCHDOG_FORCE_EXIT_TIMEOUT_S (10s, the
# absolute last-resort backstop) -- a regression back to "hangs forever"
# fails this test via TimeoutExpired well before either of those, rather
# than hanging the test suite itself.
BOUNDED_SHUTDOWN_TIMEOUT_S = 8.0
HEALTH_WAIT_TIMEOUT_S = 20.0


def _free_tcp_port() -> int:
    """Grabs an ephemeral port from the OS, same pattern every stdlib
    "find a free port for a test server" recipe uses -- inherently a
    tiny TOCTOU race (something else could grab it between close() and
    the subprocess's own bind()), acceptable for a test, not for
    anything load-bearing."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/api/health"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.1)
    return False


@pytest.mark.slow # real subprocess lifecycle + real port bind; also the known
# sandbox-only-failure pair (needs a real un-sandboxed process/network) -- see
# tests/conftest.py for the fast/slow split.
class TestRealProcessShutdownWithOpenWebsocket:
    """The actual regression test for the confirmed bug. See this
    module's own docstring for the full context; see
    tests/_run_product_subprocess_harness.py for exactly what's real vs.
    faked in the launched subprocess (short version: everything except
    the camera hardware and the capture-loop body itself, which is the
    same scope every other real-hardware-adjacent test in this project
    fakes)."""

    def _start_subprocess(self, package_root: Path) -> tuple[subprocess.Popen, int]:
        port = _free_tcp_port()
        proc = subprocess.Popen(
            [
                sys.executable,
                str(HARNESS_PATH),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--package-root",
                str(package_root),
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc, port

    def _terminate_forcefully(self, proc: subprocess.Popen) -> None:
        """Belt-and-suspenders cleanup so a failing assertion (or an
        unexpected exception) never leaves a real process running after
        this test function returns -- see this task's own "don't leave
        background processes running" requirement."""
        if proc.poll() is None:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5.0)

    def test_real_sigint_with_open_websocket_exits_within_bounded_time(self, package_root):
        proc, port = self._start_subprocess(package_root)
        try:
            healthy = _wait_for_health(port, HEALTH_WAIT_TIMEOUT_S)
            if not healthy:
                proc.kill()
                out = proc.communicate(timeout=5.0)[0]
                pytest.fail(f"subprocess never became healthy -- output:\n{out}")

            uri = f"ws://127.0.0.1:{port}/api/events"
            # A real WebSocket connection held open exactly like a
            # browser tab left sitting on the dashboard -- the scenario
            # actually hit live ("he had the dashboard open in a
            # browser... when he hit Ctrl-C").
            with websockets.sync.client.connect(uri, open_timeout=10.0) as ws:
                hello_raw = ws.recv(timeout=10.0)
                assert '"type": "HELLO"' in hello_raw or '"type":"HELLO"' in hello_raw

                t0 = time.monotonic()
                proc.send_signal(signal.SIGINT)

                try:
                    proc.wait(timeout=BOUNDED_SHUTDOWN_TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - t0
                    out = proc.stdout.read() if proc.stdout else ""
                    self._terminate_forcefully(proc)
                    pytest.fail(
                        f"process did not exit within {BOUNDED_SHUTDOWN_TIMEOUT_S:.0f}s of "
                        f"SIGINT (still alive at {elapsed:.1f}s) -- this is the exact hang "
                        f"this test exists to catch. subprocess output (tail):\n"
                        + "\n".join(out.splitlines()[-60:])
                    )

            elapsed = time.monotonic() - t0
            assert elapsed < BOUNDED_SHUTDOWN_TIMEOUT_S
            # Clean exit code (0), not killed/crashed -- proves this is
            # the real shutdown() path completing, not just the process
            # dying some other way.
            assert proc.returncode == 0, (
                f"expected clean exit (0), got {proc.returncode} -- "
                f"output:\n" + (proc.stdout.read() if proc.stdout else "")
            )
        finally:
            self._terminate_forcefully(proc)

    def test_real_sigint_with_no_websocket_client_also_exits_within_bounded_time(self, package_root):
        """Baseline companion to the test above -- confirmed during this
        session's own investigation that the hang this test suite is
        guarding against is NOT actually gated on a WebSocket client
        being connected at all (see opendarts/live/server.py's
        AppState._live_event_loop docstring for the real mechanism: an
        always-running background task, independent of any client).
        Keeping both cases covered so a future regression that's
        websocket-specific OR one that isn't are both caught."""
        proc, port = self._start_subprocess(package_root)
        try:
            healthy = _wait_for_health(port, HEALTH_WAIT_TIMEOUT_S)
            if not healthy:
                proc.kill()
                out = proc.communicate(timeout=5.0)[0]
                pytest.fail(f"subprocess never became healthy -- output:\n{out}")

            t0 = time.monotonic()
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=BOUNDED_SHUTDOWN_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                out = proc.stdout.read() if proc.stdout else ""
                self._terminate_forcefully(proc)
                pytest.fail(
                    f"process did not exit within {BOUNDED_SHUTDOWN_TIMEOUT_S:.0f}s of "
                    f"SIGINT (still alive at {elapsed:.1f}s), even with NO WebSocket "
                    f"client connected. subprocess output (tail):\n"
                    + "\n".join(out.splitlines()[-60:])
                )
            assert proc.returncode == 0
        finally:
            self._terminate_forcefully(proc)


# ---------------------------------------------------------------------------
# camera_devices wiring, 2026-09-10.
# camera_configs_from_resolution_preferences() existed for weeks with no
# caller, so camera_resolutions was parsed, validated and then discarded.
# These pin the wiring itself, not just the parsing.
# ---------------------------------------------------------------------------


class _ConfigCapturingHub(FakeHub):
    """FakeHub that remembers what `configs=` it was handed."""

    def __init__(self, configs=None, frame_sink=None, urls=None) -> None:
        super().__init__(configs=configs, frame_sink=frame_sink, urls=urls)
        self.configs_seen = configs


def test_build_components_passes_camera_configs_to_the_hub(package_root):
    from opendarts.live.local_capture import CameraConfig

    FakeHub.instances = []
    wanted = [CameraConfig(device=1), CameraConfig(device=2), CameraConfig(device=3)]
    run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=_ConfigCapturingHub,
        camera_configs=wanted,
    )
    seen = FakeHub.instances[0].configs_seen
    assert [c.device for c in seen] == [1, 2, 3]


def test_build_components_without_camera_configs_passes_none(package_root):
    """None is the hub's own "use DEFAULT_CAMERA_DEVICES" branch, so a rig
    with no config key builds the hub it always did."""
    FakeHub.instances = []
    run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=_ConfigCapturingHub,
    )
    assert FakeHub.instances[0].configs_seen is None


def test_camera_configs_from_live_config_maps_devices_to_slot_order():
    from opendarts.live.config import LiveConfig

    cfg = LiveConfig(camera_devices=[3, 0, 2])
    configs = run_product._camera_configs_from_live_config(cfg)
    assert [c.device for c in configs] == [3, 0, 2]


def test_camera_configs_from_live_config_is_none_when_config_says_nothing():
    """Silence must resolve to None, not to a materialised default list --
    otherwise this layer quietly owns the default device list forever."""
    from opendarts.live.config import LiveConfig

    assert run_product._camera_configs_from_live_config(LiveConfig()) is None


def test_camera_configs_from_live_config_honours_resolutions_too():
    """The wiring this rides in on: camera_resolutions had been parsed and
    then dropped on the floor by every real construction site."""
    from opendarts.live.config import LiveConfig

    cfg = LiveConfig(camera_devices=[1, 2], camera_resolutions={1: None, 2: (1920, 1080)})
    configs = run_product._camera_configs_from_live_config(cfg)
    assert [c.device for c in configs] == [1, 2]
    assert (configs[0].width, configs[0].height) == (None, None)     # "auto"
    assert (configs[1].width, configs[1].height) == (1920, 1080)


# ---------------------------------------------------------------------------
# The AD oracle flag must SURVIVE a restart, 2026-09-11.
# It was read at startup only to decide virtual-camera publishing and never
# applied to the listener, so switching the oracle off and restarting
# brought it back on -- the setting looked saved and silently did not hold.
# ---------------------------------------------------------------------------


class _RecordingListener:
    instances: "list[_RecordingListener]" = []

    def __init__(self, base_url, on_status_change=None, on_connection_change=None):
        self.base_url = base_url
        self.ws_url = base_url.replace("http://", "ws://") + "/api/events"
        self.started = False
        self.enabled = True
        _RecordingListener.instances.append(self)

    def start(self):
        self.started = True

    def stop(self, timeout=3.0):
        self.started = False

    def set_enabled(self, value):
        self.enabled = bool(value)

    def oracle_base_url(self):
        return self.base_url if self.enabled else None

    def is_connected(self):
        return False

    def board_status(self):
        return ("stopped", None)


def _build_with_ad(package_root, monkeypatch, configured):
    """Build components with `ad_enabled` reading back as `configured`."""
    _RecordingListener.instances = []
    import opendarts.live.config as config_mod

    real = config_mod.read_config_section

    def fake(section, path=None):
        if section == "ad_enabled":
            return configured
        return real(section) if path is None else real(section, path)

    monkeypatch.setattr(config_mod, "read_config_section", fake)
    run_product._build_components(
        package_root=package_root,
        host="127.0.0.1",
        port=0,
        poll_interval_s=0.01,
        hub_factory=FakeHub,
        enable_ad_ground_truth_inline=True,
        ad_listener_factory=_RecordingListener,
    )
    return _RecordingListener.instances[0]


def test_ad_disabled_in_config_does_not_connect_on_startup(package_root, monkeypatch):
    lst = _build_with_ad(package_root, monkeypatch, configured=False)
    assert lst.started is False, "it connected despite being switched off"
    assert lst.enabled is False
    assert lst.oracle_base_url() is None, (
        "every AD path gates on oracle_base_url(); leaving it set means the "
        "flag reads as on while nothing is connected"
    )


def test_ad_enabled_in_config_connects_as_before(package_root, monkeypatch):
    lst = _build_with_ad(package_root, monkeypatch, configured=True)
    assert lst.started is True
    assert lst.enabled is True


def test_ad_absent_from_config_defaults_to_off(package_root, monkeypatch):
    """No key means OFF, changed 2026-09-16 (this test used to assert the
    opposite, deliberately).

    The old default came from a time when every rig running this code had
    Autodarts beside it. For anyone else it charged the newest user the
    highest price: the WS listener retries forever, so their first
    impression is a stream of failures to reach a service they have never
    heard of on localhost:3180; the throw path then blocks up to 5s PER
    THROW waiting on it, which presents as "scoring is slow" and points
    nowhere near a toggle named after a different product.

    Someone running Autodarts knows the name and can switch it on.
    """
    lst = _build_with_ad(package_root, monkeypatch, configured=None)
    assert lst.started is False, (
        "an absent ad_enabled key started the AD listener -- a fresh clone "
        "with no Autodarts must not connect to it"
    )
    assert lst.enabled is False

