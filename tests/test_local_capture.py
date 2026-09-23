"""Tests for opendarts/live/local_capture.py -- the direct-cv2.VideoCapture
frame source for the rig.

HONEST SCOPE: these are LOGIC tests only, NOT hardware proof.
cv2.VideoCapture itself is mocked throughout, via the same
monkeypatch-a-real-module-attribute pattern tests/test_capture_replay.py
already uses for cv2.imwrite (see that file's own comment on the
correct monkeypatch seam). Nothing here opens a real camera. Passing
these tests proves the Python control flow -- backend fallback order,
config-driven width/height/fps, graceful handling of a camera that
fails to open -- is correct. It does NOT prove the real hardware on
the rig actually works with this code; that can only be checked by running
scripts/test_local_camera_access.py directly on the rig (see that script's
own header) or opendarts/live/local_capture.py's own module docstring.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from opendarts.live import local_capture as local_capture_module
from opendarts.live.local_capture import (
    CameraConfig,
    LocalCameraHub,
    fetch_all_snapshots,
    fetch_snapshot,
)


@pytest.fixture(autouse=True)
def _stop_pump_threads_after_every_test():
    """Autouse cleanup, added alongside the 2026-08-12 pump-thread
    architecture change (see local_capture.py's own module docstring,
    "ARCHITECTURE CHANGE" section): LocalCameraHub.open_all() now starts
    a real background `threading.Thread` (+ ThreadPoolExecutor) whenever
    at least one camera opens, even against a FakeVideoCapture mock that
    returns instantly (so the pump free-runs as fast as Python allows,
    with no natural per-frame blocking to throttle it). Most tests below
    build a hub and never call close_all() themselves (there was nothing
    to clean up before this change -- grab() didn't start anything).
    Left alone, every such test would leak one busy-spinning daemon
    thread for the rest of the whole pytest process -- harmless for
    correctness (daemon threads die with the process, and each hub's
    state is independent) but wasteful and a real source of potential
    timing flakiness in later tests. This fixture transparently tracks
    every LocalCameraHub constructed during a test and calls
    close_all() on each of them afterward, without requiring every
    individual test body to remember to do so itself."""
    created: list[LocalCameraHub] = []
    original_init = LocalCameraHub.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    LocalCameraHub.__init__ = _tracking_init # type: ignore[method-assign]
    try:
        yield
    finally:
        LocalCameraHub.__init__ = original_init # type: ignore[method-assign]
        for hub in created:
            hub.close_all()


class FakeVideoCapture:
    """Stand-in for cv2.VideoCapture. Behavior (opens or not, produces a
    frame or not, negotiated size) is controlled per-instance by the
    factory that creates it -- see make_fake_capture_factory below."""

    def __init__(
        self,
        device,
        backend,
        *,
        opens: bool,
        frame_ok: bool = True,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
    ) -> None:
        self.device = device
        self.backend = backend
        self._opened = opens
        self._frame_ok = frame_ok
        self.width = width
        self.height = height
        self.fps = fps
        self.set_calls: list[tuple[int, float]] = []
        self.read_calls = 0
        self.released = False

    def isOpened(self) -> bool: # noqa: N802 -- matches cv2's own method name
        return self._opened

    def release(self) -> None:
        self.released = True
        self._opened = False

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def read(self):
        self.read_calls += 1
        if not self._frame_ok:
            return False, None
        frame = np.full((self.height, self.width, 3), 128, dtype=np.uint8)
        return True, frame


def make_fake_capture_factory(
    *, fails_backends: set[int] | None = None, frame_ok: bool = True
):
    """Builds a callable usable as monkeypatch.setattr(cv2, "VideoCapture", ...).
    `fails_backends` names backend constants (e.g. {cv2.CAP_AVFOUNDATION})
    whose FakeVideoCapture reports isOpened() == False, so later backends
    in the fallback list must be tried. Also records every (device,
    backend) pair the module attempted, in call order, on `.calls`.
    """
    calls: list[tuple] = []
    fails_backends = fails_backends or set()

    def factory(device, backend):
        calls.append((device, backend))
        opens = backend not in fails_backends
        return FakeVideoCapture(device, backend, opens=opens, frame_ok=frame_ok)

    factory.calls = calls # type: ignore[attr-defined]
    return factory


# ---------------------------------------------------------------------------
# Backend fallback order
# ---------------------------------------------------------------------------


def test_tries_avfoundation_then_v4l2_then_any_in_order(monkeypatch):
    """The proven open order: AVFoundation first, then V4L2, then
    CAP_ANY as the last resort."""
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    ok_flags = hub.open_all()

    assert ok_flags == [True]
    backends_tried = [backend for _device, backend in factory.calls]
    assert backends_tried[0] == cv2.CAP_AVFOUNDATION
    # Succeeded on the first backend -- V4L2/CAP_ANY should never be tried.
    assert len(backends_tried) == 1
    assert hub.status[0].backend_used == "CAP_AVFOUNDATION"


def test_falls_back_to_cap_any_when_the_platform_backend_fails(monkeypatch):
    """Only the CURRENT platform's backend is tried, then CAP_ANY. The
    suite runs on Darwin, so V4L2/MSMF must not appear -- trying another
    OS's backend is two guaranteed-failing opens per camera."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    factory = make_fake_capture_factory(fails_backends={cv2.CAP_AVFOUNDATION})
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    ok_flags = hub.open_all()

    assert ok_flags == [True]
    backends_tried = [backend for _device, backend in factory.calls]
    assert backends_tried == [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    assert cv2.CAP_V4L2 not in backends_tried
    assert hub.status[0].backend_used == "CAP_ANY"


@pytest.mark.parametrize(
    "system, expected",
    [
        ("Darwin", cv2.CAP_AVFOUNDATION),
        ("Linux", cv2.CAP_V4L2),
        ("Windows", cv2.CAP_DSHOW),
    ],
)
def test_backend_preference_is_chosen_per_platform(monkeypatch, system, expected):
    """The CAP_* names are integer constants present in the bindings on
    every platform, so the backend cannot be picked by hasattr -- it has
    to follow the OS actually running. (Windows tries our own Media
    Foundation reader first; that is not a cv2 open, and off Windows it
    never opens.)"""
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr(local_capture_module, "_dshow_index_for", lambda d: d)
    factory = make_fake_capture_factory(fails_backends=set())
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    assert hub.open_all() == [True]
    assert [b for _d, b in factory.calls][0] == expected


def test_windows_falls_back_to_dshow_and_never_to_cap_any(monkeypatch):
    """Other software reads our virtual cameras now, so DSHOW's exclusive lock
    no longer matters, and the headless OpenCV package has no MSMF. CAP_ANY
    is never tried on Windows: there it is DSHOW at an unmatched number."""
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(local_capture_module, "_dshow_index_for", lambda d: d)
    factory = make_fake_capture_factory(fails_backends={cv2.CAP_DSHOW})
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    assert hub.open_all() == [False]
    backends_tried = [b for _d, b in factory.calls]
    assert backends_tried == [cv2.CAP_DSHOW]
    assert cv2.CAP_MSMF not in backends_tried and cv2.CAP_ANY not in backends_tried


# ---------------------------------------------------------------------------
# Config-driven width/height/fps
# ---------------------------------------------------------------------------


def test_sets_requested_width_height_fps_and_mjpg_but_not_buffersize(monkeypatch):
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    cfg = CameraConfig(device=2, width=640, height=480, fps=15)
    hub = LocalCameraHub(configs=[cfg])
    hub.open_all()

    fake_cap = hub._caps.get(0) # noqa: SLF001 -- test needs the underlying fake
    assert fake_cap is not None
    set_props = {prop for prop, _value in fake_cap.set_calls}
    assert (cv2.CAP_PROP_FRAME_WIDTH, 640) in fake_cap.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 480) in fake_cap.set_calls
    assert (cv2.CAP_PROP_FPS, 15) in fake_cap.set_calls
    # FOURCC IS NOW SET, 2026-09-15 -- this test previously pinned the
    # opposite. Measured on the Linux rig: without an explicit MJPG the
    # V4L2 driver hands back uncompressed YUYV and caps 720p at 10fps,
    # a third of the rate the same camera does in MJPEG. The old pin
    # recorded a real decision, so it is changed rather than deleted.
    fourcc = getattr(cv2, "CAP_PROP_FOURCC", None)
    if fourcc is not None:
        assert fourcc in set_props, "MJPG must be requested explicitly"
        # BEFORE the geometry: V4L2 applies the pixel format first and
        # ignores a fourcc that arrives after width/height, so the
        # ordering is load-bearing rather than incidental.
        order = [prop for prop, _ in fake_cap.set_calls]
        assert order.index(fourcc) < order.index(cv2.CAP_PROP_FRAME_WIDTH), (
            "fourcc must be set before width/height or V4L2 ignores it"
        )

    # Still deliberately NOT set: CAP_PROP_BUFFERSIZE was measured as
    # unsettable on MSMF and buys nothing here.
    buffersize = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffersize is not None:
        assert buffersize not in set_props

    assert hub.status[0].requested_width == 640
    assert hub.status[0].requested_height == 480
    assert hub.status[0].requested_fps == 15


def test_records_negotiated_actual_size_which_may_differ_from_requested(monkeypatch):
    """Drivers often hand back something other than what was requested --
    the module should report what it actually got, not just echo the
    request."""

    def factory(device, backend):
        # Only AVFoundation "succeeds", but negotiates a different size
        # than requested (960x540 instead of the requested 1280x720).
        opens = backend == cv2.CAP_AVFOUNDATION or backend == cv2.CAP_ANY
        return FakeVideoCapture(device, backend, opens=opens, width=960, height=540, fps=24.0)

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0, width=1280, height=720, fps=30)])
    hub.open_all()

    status = hub.status[0]
    assert status.requested_width == 1280
    assert status.requested_height == 720
    assert status.actual_width == 960
    assert status.actual_height == 540
    assert status.actual_fps == 24.0


# ---------------------------------------------------------------------------
# 'auto' resolution probing (2026-08-20 fix -- see
# opendarts.live.camera_resolution's module docstring for the bug this
# closes). CameraConfig(width=None, height=None) is the ONLY way into
# this code path -- every test above this section already proves the
# default (real int width/height) path is completely unaffected, and the
# tests immediately below prove that explicitly too.
# ---------------------------------------------------------------------------


class RespondingFakeVideoCapture(FakeVideoCapture):
    """Like FakeVideoCapture, but `get()` reflects whatever the most
    recent `set()` actually negotiated -- needed to exercise the REAL
    (non-monkeypatched) probing logic end-to-end against something that
    behaves like camera_resolution.py's own tests expect (request vs.
    negotiated can differ, per `honored`)."""

    def __init__(self, device, backend, *, opens: bool, honored: set[tuple[int, int]],
                 fallback: tuple[int, int], frame_ok: bool = True, fps: float = 30.0) -> None:
        super().__init__(device, backend, opens=opens, frame_ok=frame_ok,
                          width=fallback[0], height=fallback[1], fps=fps)
        self.honored = honored
        self._pending_w: int | None = None
        self._pending_h: int | None = None

    def set(self, prop: int, value: float) -> bool:
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self._pending_w = int(value)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self._pending_h = int(value)
        if self._pending_w is not None and self._pending_h is not None:
            requested = (self._pending_w, self._pending_h)
            if requested in self.honored:
                self.width, self.height = requested
            self._pending_w = None
            self._pending_h = None
        return True


def test_default_camera_config_never_probes_unchanged_behavior(monkeypatch):
    """BACKWARD COMPATIBILITY, explicit: a bare default CameraConfig()
    (what every real call site in this app builds today, and what
    LocalCameraHub's own bare `configs=None` default constructs) must
    NEVER even import/call the probing function -- proves the default
    path is byte-identical control flow to before this fix existed, not
    just numerically equal output."""
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("highest_supported_resolution must not be called for a fixed CameraConfig")

    monkeypatch.setattr(
        local_capture_module.camera_resolution, "highest_supported_resolution", _must_not_be_called
    )
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    ok_flags = hub.open_all()

    assert ok_flags == [True]
    fake_cap = hub._caps.get(0) # noqa: SLF001
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1280) in fake_cap.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 720) in fake_cap.set_calls
    assert hub.status[0].requested_width == 1280
    assert hub.status[0].requested_height == 720


def test_auto_resolution_config_property():
    assert CameraConfig(width=None, height=None).auto_resolution is True
    assert CameraConfig(width=None, height=480).auto_resolution is True
    assert CameraConfig(width=640, height=None).auto_resolution is True
    assert CameraConfig(width=640, height=480).auto_resolution is False
    assert CameraConfig().auto_resolution is False # the real default


def test_auto_resolution_requests_the_highest_probed_mode(monkeypatch):
    """CameraConfig(width=None, height=None) must probe and then request
    the highest genuinely-supported resolution found -- verified against
    the REAL (non-monkeypatched) camera_resolution.highest_supported_resolution
    logic, driven by a fake capture whose get() genuinely reflects what
    was set(), not a stubbed-out return value."""

    def factory(device, backend):
        return RespondingFakeVideoCapture(
            device, backend, opens=(backend == cv2.CAP_AVFOUNDATION),
            honored={(640, 480), (1280, 720), (1920, 1080)},
            fallback=(640, 480),
        )

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0, width=None, height=None)])
    ok_flags = hub.open_all()

    assert ok_flags == [True]
    fake_cap = hub._caps.get(0) # noqa: SLF001
    # The LAST width/height set() calls must be the highest honored mode.
    width_sets = [v for p, v in fake_cap.set_calls if p == cv2.CAP_PROP_FRAME_WIDTH]
    height_sets = [v for p, v in fake_cap.set_calls if p == cv2.CAP_PROP_FRAME_HEIGHT]
    assert width_sets[-1] == 1920
    assert height_sets[-1] == 1080
    assert hub.status[0].requested_width == 1920
    assert hub.status[0].requested_height == 1080
    assert hub.status[0].actual_width == 1920
    assert hub.status[0].actual_height == 1080


def test_auto_resolution_falls_back_to_default_when_nothing_honored(monkeypatch, caplog):
    """A real, if unlikely, possible outcome: probing finds nothing
    genuinely supported. Must fall back to DEFAULT_WIDTH/DEFAULT_HEIGHT,
    loudly (a warning, not a silent fallback), never crash or leave the
    camera unconfigured."""

    def factory(device, backend):
        # fallback deliberately NOT one of COMMON_RESOLUTIONS's own
        # candidates -- otherwise get() would coincidentally "honor"
        # that one candidate by always echoing the fallback value back,
        # masking the "nothing genuinely honored" case this test means
        # to exercise.
        return RespondingFakeVideoCapture(
            device, backend, opens=(backend == cv2.CAP_AVFOUNDATION),
            honored=set(), # nothing ever honored
            fallback=(3, 3),
        )

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    with caplog.at_level("WARNING"):
        hub = LocalCameraHub(configs=[CameraConfig(device=0, width=None, height=None)])
        ok_flags = hub.open_all()

    assert ok_flags == [True]
    assert hub.status[0].requested_width == local_capture_module.DEFAULT_WIDTH
    assert hub.status[0].requested_height == local_capture_module.DEFAULT_HEIGHT
    assert any(
        "auto" in rec.getMessage() and "falling back" in rec.getMessage()
        for rec in caplog.records
    )


def test_auto_resolution_survives_an_exception_during_probing(monkeypatch, caplog):
    """VERIFIER FINDING, 2026-08-20: probing touches the real
    cv2.VideoCapture (multiple set()/get() round trips) -- an exception
    partway through must be caught and degrade to the fixed default for
    THIS camera, exactly like every other camera-touching call in this
    class (_read_one(), _release_one_locked(), _pump_loop()), never
    propagate up and take down open_all() for every other camera too."""

    def _raises(*args, **kwargs):
        raise RuntimeError("simulated real cv2.VideoCapture failure mid-probe")

    monkeypatch.setattr(
        local_capture_module.camera_resolution, "highest_supported_resolution", _raises
    )
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    with caplog.at_level("ERROR"):
        hub = LocalCameraHub(configs=[CameraConfig(device=0, width=None, height=None)])
        ok_flags = hub.open_all()

    # The camera itself still opens successfully -- probing failure is
    # not the same as backend-open failure -- just with the safe default
    # resolution instead of a probed one.
    assert ok_flags == [True]
    assert hub.status[0].requested_width == local_capture_module.DEFAULT_WIDTH
    assert hub.status[0].requested_height == local_capture_module.DEFAULT_HEIGHT
    assert any("exception while probing" in rec.getMessage() for rec in caplog.records)


def test_auto_resolution_exception_in_one_camera_does_not_crash_open_all_for_others(monkeypatch):
    """The real regression this guard prevents: WITHOUT it, an exception
    inside one camera's probe call would propagate out of
    ThreadPoolExecutor.map()'s list(...) call in open_all(), crashing the
    open attempt for every OTHER configured camera too, not just the one
    that failed to probe."""

    def _raises(*args, **kwargs):
        raise RuntimeError("simulated real cv2.VideoCapture failure mid-probe")

    monkeypatch.setattr(
        local_capture_module.camera_resolution, "highest_supported_resolution", _raises
    )
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    # cam0 is 'auto' (would raise without the guard); cam1 is a normal
    # fixed-resolution camera that must open fine regardless.
    hub = LocalCameraHub(
        configs=[
            CameraConfig(device=0, width=None, height=None),
            CameraConfig(device=1, width=640, height=480),
        ]
    )
    ok_flags = hub.open_all() # must not raise

    assert ok_flags == [True, True]
    assert hub.status[1].requested_width == 640
    assert hub.status[1].requested_height == 480


def test_auto_resolution_uses_monkeypatched_probe_result(monkeypatch):
    """Integration-boundary test: local_capture.py genuinely calls
    opendarts.live.camera_resolution.highest_supported_resolution() (not a
    hand-rolled duplicate) -- proven by monkeypatching that exact
    function and confirming its return value is what gets requested."""
    calls = []

    def fake_highest(cap, *args, **kwargs):
        calls.append(cap)
        return (1600, 1200)

    monkeypatch.setattr(
        local_capture_module.camera_resolution, "highest_supported_resolution", fake_highest
    )
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0, width=None, height=None)])
    hub.open_all()

    assert len(calls) == 1
    fake_cap = hub._caps.get(0) # noqa: SLF001
    assert (cv2.CAP_PROP_FRAME_WIDTH, 1600) in fake_cap.set_calls
    assert (cv2.CAP_PROP_FRAME_HEIGHT, 1200) in fake_cap.set_calls
    assert hub.status[0].requested_width == 1600
    assert hub.status[0].requested_height == 1200


# ---------------------------------------------------------------------------
# camera_configs_from_resolution_preferences()
# ---------------------------------------------------------------------------


def test_camera_configs_from_resolution_preferences_empty_dict_matches_bare_default():
    """BACKWARD COMPATIBILITY, provably not just assumed: an empty dict
    (real-world default -- missing/absent config.json) must produce
    a CameraConfig list field-for-field identical to
    LocalCameraHub.__init__'s own bare default."""
    from opendarts.live.local_capture import camera_configs_from_resolution_preferences
    from opendarts.live.local_capture import DEFAULT_CAMERA_DEVICES

    resolved = camera_configs_from_resolution_preferences({})
    bare_default = [CameraConfig(device=d) for d in DEFAULT_CAMERA_DEVICES]
    assert resolved == bare_default


def test_camera_configs_from_resolution_preferences_auto_and_explicit_override():
    from opendarts.live.local_capture import camera_configs_from_resolution_preferences

    resolved = camera_configs_from_resolution_preferences(
        {0: None, 1: (1920, 1080)}, devices=[0, 1, 2]
    )

    assert resolved[0] == CameraConfig(device=0, width=None, height=None)
    assert resolved[1] == CameraConfig(device=1, width=1920, height=1080)
    assert resolved[2] == CameraConfig(device=2) # no entry -- untouched default


def test_camera_configs_from_resolution_preferences_full_chain_from_live_config_json(tmp_path):
    """Real end-to-end proof: a hand-written data/config.json ->
    load_live_config() -> camera_configs_from_resolution_preferences()
    produces the exact CameraConfig list an operator would expect from
    the JSON they wrote -- not just each layer tested in isolation."""
    import json

    from opendarts.live.config import load_live_config
    from opendarts.live.local_capture import camera_configs_from_resolution_preferences

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_resolutions": {"0": "auto", "2": "800x600"}}))

    live_cfg = load_live_config(path)
    resolved = camera_configs_from_resolution_preferences(
        live_cfg.camera_resolutions, devices=[0, 1, 2]
    )

    assert resolved[0] == CameraConfig(device=0, width=None, height=None)
    assert resolved[1] == CameraConfig(device=1) # untouched -- no override
    assert resolved[2] == CameraConfig(device=2, width=800, height=600)


# ---------------------------------------------------------------------------
# Graceful handling of a camera that fails to open
# ---------------------------------------------------------------------------


def test_camera_that_fails_all_backends_reports_failure_without_raising(monkeypatch):
    factory = make_fake_capture_factory(
        fails_backends={cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2, cv2.CAP_ANY}
    )
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    ok_flags = hub.open_all() # must not raise

    assert ok_flags == [False]
    status = hub.status[0]
    assert status.opened is False
    assert status.backend_used is None
    assert status.last_error is not None and "all backends failed" in status.last_error

    # grab() on a never-opened camera must also fail gracefully, not raise.
    frame = hub.grab(0)
    assert frame is None


def test_one_failed_camera_does_not_prevent_others_from_opening(monkeypatch):
    def factory(device, backend):
        # Device 1 always fails; devices 0 and 2 succeed on AVFoundation.
        if device == 1:
            return FakeVideoCapture(device, backend, opens=False)
        return FakeVideoCapture(device, backend, opens=(backend == cv2.CAP_AVFOUNDATION))

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0), CameraConfig(device=1), CameraConfig(device=2)])
    ok_flags = hub.open_all()

    assert ok_flags == [True, False, True]
    assert hub.status[0].opened is True
    assert hub.status[1].opened is False
    assert hub.status[2].opened is True

    frames = hub.grab_all()
    assert set(frames.keys()) == {0, 2}


# ---------------------------------------------------------------------------
# Concurrent camera opening, 2026-08-12 -- Bug 2 item 2. open_all() used
# to open every configured camera ONE AT A TIME
# (`[self._open_one(i, cfg) for i, cfg in enumerate(...)]`); real measured
# open_latency_s on the rig is ~2.2-2.3s per camera, so 3 cameras cost
# ~6.75s sequentially, before calibration even started. See open_all()'s
# own "CONCURRENT OPEN" docstring section for the full safety
# investigation this change required (per-camera lock granularity vs. the
# genuinely different self._caps dict-write hazard, and why
# self._caps_lock exists). These tests prove: (1) real timing -- N
# cameras with an artificial per-camera open delay now cost roughly
# max(delays), not sum(delays); (2) correctness under concurrency --
# every camera ends up with its OWN cap/status, not corrupted or
# cross-assigned by the concurrent writes; (3) ok_flags stays in config
# order regardless of which camera's thread happens to finish first
# (ThreadPoolExecutor.map()'s documented ordering guarantee, proven
# here rather than just assumed).
# ---------------------------------------------------------------------------


def test_open_all_opens_cameras_concurrently_not_sequentially(monkeypatch):
    """Real timing proof, mocked delay (not real hardware/wall-clock
    guesswork) -- generous margin so this cannot flake on a loaded CI
    box: sequential would cost ~= n_cameras * per_camera_delay (0.6s for
    3 cameras @ 0.2s each); concurrent should cost close to just
    per_camera_delay (0.2s) plus scheduling overhead. Asserting
    `elapsed < per_camera_delay * 2` gives a wide margin (up to 2x a
    single camera's own delay) while still failing hard if opening
    reverts to sequential (which would take ~3x a single delay, well
    past the threshold)."""
    per_camera_delay = 0.2
    n_cameras = 3

    def factory(device, backend):
        time.sleep(per_camera_delay)
        return FakeVideoCapture(device, backend, opens=True)

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=d) for d in range(n_cameras)])
    t0 = time.monotonic()
    ok_flags = hub.open_all()
    elapsed = time.monotonic() - t0

    assert ok_flags == [True, True, True]
    assert elapsed < per_camera_delay * 2, (
        f"open_all() took {elapsed:.3f}s opening {n_cameras} cameras with a "
        f"{per_camera_delay}s delay each -- expected close to max(delays) "
        f"(~{per_camera_delay:.3f}s), not sum(delays) "
        f"(~{per_camera_delay * n_cameras:.3f}s if still sequential)"
    )


def test_open_all_preserves_config_order_in_ok_flags_regardless_of_completion_order(
    monkeypatch,
):
    """Camera 1 (middle) finishes fastest AND fails to open, camera 0
    finishes slowest and succeeds -- if ok_flags were accidentally
    ordered by completion instead of config position, this would catch
    it (a naive `as_completed()`-based implementation would return
    [False, True, True] or some other out-of-order shape here;
    ThreadPoolExecutor.map(), which open_all() actually uses, is
    documented to preserve input order in its results regardless of
    completion order)."""
    delays = {0: 0.15, 1: 0.02, 2: 0.08}

    def factory(device, backend):
        time.sleep(delays[device])
        opens = device != 1
        return FakeVideoCapture(device, backend, opens=opens)

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=d) for d in range(3)])
    ok_flags = hub.open_all()

    assert ok_flags == [True, False, True]
    assert hub.status[0].opened is True
    assert hub.status[1].opened is False
    assert hub.status[2].opened is True


def test_open_all_concurrent_opens_do_not_cross_assign_caps_or_status(monkeypatch):
    """Correctness under real concurrency, not just timing: every one of
    N cameras opening at (roughly) the same instant must end up with ITS
    OWN cv2.VideoCapture in self._caps and ITS OWN CameraStatus fields --
    a race in the shared self._caps dict write (the one hazard
    self._caps_lock exists for, see its own __init__ docstring) could in
    principle drop an entry or leave the dict in a bad state. Runs with a
    larger camera count and zero artificial delay (maximizes the chance
    of two threads landing on the dict write at literally the same GIL
    time-slice boundary, the actual scenario being guarded against) and
    repeats several times, since a genuine race (if the guard were
    missing/broken) would not necessarily reproduce on every run."""
    n_cameras = 8

    def factory(device, backend):
        return FakeVideoCapture(device, backend, opens=True)

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    for _ in range(10):
        hub = LocalCameraHub(configs=[CameraConfig(device=d) for d in range(n_cameras)])
        ok_flags = hub.open_all()

        assert ok_flags == [True] * n_cameras
        assert set(hub._caps.keys()) == set(range(n_cameras)) # noqa: SLF001
        for i in range(n_cameras):
            assert hub._caps[i].device == i # noqa: SLF001 -- own fake cap
            assert hub.status[i].opened is True
        hub.close_all()


# ---------------------------------------------------------------------------
# Pump-thread architecture, 2026-08-12 -- see local_capture.py's own module
# docstring "ARCHITECTURE CHANGE" section for the full incident writeup:
# the earlier per-camera-lock fix (directly below, in this file's own
# history) stopped two threads from being inside cap.read() at the exact
# same instant, but did NOT stop grab() itself being called -- and hence
# cap.read() being invoked -- from a DIFFERENT OS thread each time,
# depending on which caller (capture loop thread vs. dashboard snapshot
# route's asyncio.to_thread pool) happened to ask first. That approach
# was replaced with a proven camera-hub architecture: exactly one
# dedicated pump thread (+ its
# own small ThreadPoolExecutor, one worker per camera) ever calls
# cap.read(), for the hub's whole lifetime; grab()/grab_all() became pure
# cache reads that never touch cv2.VideoCapture at all.
#
# The tests immediately below (through
# test_close_all_stops_pump_thread_and_releases_captures) REPLACE three
# tests that used to live here: test_grab_serializes_concurrent_reads_on_
# the_same_camera, test_grab_without_the_lock_actually_races, and
# test_grab_on_different_cameras_does_not_share_a_lock. Those tests proved
# that grab() ITSELF serialized concurrent cap.read() calls via a
# per-camera lock -- a premise that no longer holds, since grab() no
# longer calls cap.read() at all. Deleting them without explanation would
# have looked like silently-dropped coverage; this comment (and the tests
# below, which prove the NEW architecture's real guarantees with the same
# "measured, not assumed" discipline) is that explanation, per this
# project's own testing culture (see docs/DESIGN.md).
# ---------------------------------------------------------------------------


class _ThreadRecordingVideoCapture:
    """Fake cv2.VideoCapture that records which real OS thread called its
    own read() every single time, plus the running call count -- the
    actual proof this task cares about: with the pump-thread
    architecture, cap.read() must ONLY ever be called from inside
    LocalCameraHub's own pump machinery (the pump loop thread's
    ThreadPoolExecutor worker(s), or -- once only, before the pump exists
    -- the thread that calls open_all() itself for the warm-frame read;
    see local_capture.py's own module docstring for that one documented
    exception), never from a caller of grab()/grab_all(), no matter how
    many real OS threads hammer those concurrently.
    """

    def __init__(self, device, backend, **_kwargs) -> None:
        self.device = device
        self.backend = backend
        self._opened = True
        self._lock = threading.Lock()
        self.read_calls = 0
        self.read_thread_idents: list[int] = []

    def isOpened(self) -> bool: # noqa: N802
        return self._opened

    def release(self) -> None:
        self._opened = False

    def set(self, prop, value) -> bool:
        return True

    def get(self, prop) -> float:
        return 0.0

    def read(self):
        with self._lock:
            self.read_calls += 1
            self.read_thread_idents.append(threading.get_ident())
        return True, np.full((720, 1280, 3), 128, dtype=np.uint8)


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.005) -> bool:
    """Poll `predicate` until it's true or `timeout_s` elapses -- used
    throughout this section instead of a fixed time.sleep(), since the
    pump thread's real cadence against a fast in-memory fake is not
    something to guess a sleep duration for (this project's own "measure,
    don't guess" discipline)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def test_open_all_starts_exactly_one_pump_thread(monkeypatch):
    """open_all() with 3 configured cameras must start exactly ONE
    threading.Thread for the pump loop -- not one per camera (parallel
    per-camera reads come from the pump's OWN internal
    ThreadPoolExecutor, a separate concern from how many Thread objects
    the hub itself directly owns)."""
    def factory(device, backend, **kwargs):
        return _ThreadRecordingVideoCapture(device, backend)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(
        configs=[CameraConfig(device=0), CameraConfig(device=1), CameraConfig(device=2)]
    )

    threads_before = {t.ident for t in threading.enumerate()}
    hub.open_all()
    threads_after = {t.ident for t in threading.enumerate()}

    assert hub._pump_thread is not None # noqa: SLF001
    assert hub._pump_thread.is_alive() # noqa: SLF001
    assert hub._pump_thread.name == "local-cam-pump" # noqa: SLF001

    new_thread_idents = threads_after - threads_before
    pump_loop_threads = [
        t for t in threading.enumerate()
        if t.ident in new_thread_idents and t.name == "local-cam-pump"
    ]
    assert len(pump_loop_threads) == 1


def test_pump_thread_is_the_only_thread_that_ever_calls_cap_read(monkeypatch):
    """The real, deterministic proof this task asked for: multiple real
    OS threads (the test's own main thread, plus several spawned
    'hammer' threads) call hub.grab(0) concurrently and repeatedly while
    the pump runs many real cycles in the background -- the mock's own
    read() records which OS thread called it every single time. NONE of
    those recorded thread idents may ever be one of the grab-calling
    threads' -- only the hub's own internal pump machinery may appear.
    With exactly one configured camera (so exactly one
    ThreadPoolExecutor worker, max_workers=max(1, len(configs))), that
    machinery is a single, consistent OS thread throughout."""
    def factory(device, backend, **kwargs):
        return _ThreadRecordingVideoCapture(device, backend)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    fake_cap = hub._caps[0] # noqa: SLF001

    # The warm-frame read during open_all() above legitimately runs on
    # THIS test's own thread (see local_capture.py's module docstring --
    # a documented, once-only, pre-pump exception, not part of the
    # guarantee this test is actually about). Clear the fake's recording
    # here so everything captured from this point on reflects ONLY
    # post-open behavior, where the real guarantee applies.
    with fake_cap._lock: # noqa: SLF001 -- test-only introspection into our own fake
        fake_cap.read_thread_idents.clear()
        fake_cap.read_calls = 0

    grab_calling_thread_idents: set[int] = {threading.get_ident()} # this test's own thread

    def _hammer() -> None:
        grab_calling_thread_idents.add(threading.get_ident())
        for _ in range(50):
            hub.grab(0)

    threads = [threading.Thread(target=_hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(50):
        hub.grab(0) # also hammer from the main test thread itself
    for t in threads:
        t.join(timeout=10)
    assert all(not t.is_alive() for t in threads)

    # Let the pump run a good number of real cycles beyond the hammering
    # above, so this isn't just "got lucky before the pump ever ran".
    assert _wait_until(lambda: fake_cap.read_calls >= 30)

    read_thread_idents = set(fake_cap.read_thread_idents)
    assert not (read_thread_idents & grab_calling_thread_idents), (
        "cap.read() was called from a thread that also called grab() -- "
        "grab() (or something it triggered) touched cv2.VideoCapture "
        "directly, which the pump-thread architecture must never allow"
    )
    # Single camera -> single pool worker -> read() should only ever be
    # called from that ONE consistent OS thread post-open (see
    # local_capture.py's own module docstring for why this is NOT
    # guaranteed in general for a multi-camera hub -- only for this
    # max_workers=1 case).
    assert len(read_thread_idents) == 1


def test_grab_and_grab_all_return_cache_without_advancing_read_calls(monkeypatch):
    """grab()/grab_all() must be pure cache reads: calling them, even
    many times in a row, must never itself cause cap.read() to be
    called. Proven by freezing the pump (stopping+joining it directly --
    a documented, deliberate private-attribute access, same "verified by
    controlling the exact state under test" discipline this file already
    uses) once at least one real read has landed (the warm frame and/or
    a pump cycle), snapshotting read_calls at that instant, then calling
    grab()/grab_all() repeatedly and confirming read_calls never moves."""
    def factory(device, backend, **kwargs):
        return _ThreadRecordingVideoCapture(device, backend)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0), CameraConfig(device=1)])
    hub.open_all()
    fake_caps = [hub._caps[0], hub._caps[1]] # noqa: SLF001

    # >= 2 per camera: the warm-frame read (1) plus at least one real
    # pump cycle (1) -- not just "we froze before the pump ever ran".
    assert _wait_until(lambda: all(c.read_calls >= 2 for c in fake_caps))

    hub._pump_stop.set() # noqa: SLF001
    hub._pump_thread.join(timeout=5.0) # noqa: SLF001
    assert not hub._pump_thread.is_alive() # noqa: SLF001

    frozen_counts = [c.read_calls for c in fake_caps]

    for _ in range(20):
        assert hub.grab(0) is not None
        assert hub.grab(1) is not None
        frames = hub.grab_all()
        assert set(frames.keys()) == {0, 1}

    assert [c.read_calls for c in fake_caps] == frozen_counts, (
        "grab()/grab_all() advanced cap.read()'s call count -- they must be "
        "pure cache reads once the pump is stopped, never touching "
        "cv2.VideoCapture themselves"
    )


def test_pump_never_has_two_concurrent_reads_of_the_same_camera(monkeypatch):
    """The pump itself (not grab()) is now what's responsible for never
    racing a single camera's cap.read() against itself -- _pump_once()
    waits for the WHOLE batch (list(self._pool.map(...))) before
    starting its next cycle, so at most one read of a given camera can
    ever be in flight. Proven the same deterministic way the (now
    removed) per-camera-lock tests proved it for grab(): an
    overlap-detecting fake that flags whether a second read() call for
    the same camera ever starts before the first one finished, with an
    artificial delay that widens the race window far beyond what a real
    camera read would ever take."""

    class _OverlapDetectingVideoCapture:
        def __init__(self, device, backend, **_kwargs) -> None:
            self._opened = True
            self._busy = False
            self._busy_lock = threading.Lock()
            self.overlap_detected = threading.Event()
            self.read_calls = 0

        def isOpened(self) -> bool: # noqa: N802
            return self._opened

        def release(self) -> None:
            self._opened = False

        def set(self, prop, value) -> bool:
            return True

        def get(self, prop) -> float:
            return 0.0

        def read(self):
            with self._busy_lock:
                if self._busy:
                    self.overlap_detected.set()
                self._busy = True
            self.read_calls += 1
            time.sleep(0.01) # widen the race window -- real reads are much faster
            frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
            with self._busy_lock:
                self._busy = False
            return True, frame

    def factory(device, backend, **kwargs):
        return _OverlapDetectingVideoCapture(device, backend)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    fake_cap = hub._caps[0] # noqa: SLF001

    assert _wait_until(lambda: fake_cap.read_calls >= 15)
    assert not fake_cap.overlap_detected.is_set(), (
        "two reads of the SAME camera overlapped in time -- the pump's own "
        "batch-then-next-cycle design should make this impossible"
    )


def test_close_all_stops_pump_thread_and_releases_captures(monkeypatch):
    def factory(device, backend, **kwargs):
        return _ThreadRecordingVideoCapture(device, backend)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0), CameraConfig(device=1)])
    hub.open_all()
    fake_caps = [hub._caps[0], hub._caps[1]] # noqa: SLF001
    pump_thread = hub._pump_thread # noqa: SLF001
    assert pump_thread is not None and pump_thread.is_alive()

    hub.close_all()

    assert hub._pump_thread is None # noqa: SLF001
    assert not pump_thread.is_alive(), "close_all() must join the pump thread, not just signal it"
    assert hub._pool is None # noqa: SLF001
    assert hub.grab(0) is None and hub.grab(1) is None, "cache must be cleared on close"
    for cap in fake_caps:
        assert cap._opened is False # noqa: SLF001 -- released


def test_close_all_marks_status_opened_false_not_just_no_longer_updating(monkeypatch):
    """Real regression test for the 2026-08-12 status-honesty incident
    (see local_capture.py's own module docstring dated entry): hit
    a real live `/api/stop` where the idle-timeout had already closed the
    hub ~88 minutes earlier, but `GET /api/cameras/status` kept reporting
    `opened: true, last_read_ok: true` the entire time -- close_all()
    released every capture but never touched CameraStatus, so each
    camera's status just froze at its last-pumped-before-death value
    (true at the time, silently false forever after) instead of the pump
    simply stopping and the fields going quietly out of date. This proves
    the state actually FLIPS on close, not merely that updates stop."""
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0), CameraConfig(device=1)])
    hub.open_all()
    # Let at least one real pump cycle land so last_read_ok/last_read_at
    # reflect genuine pumped activity, not just the open-time warm read --
    # a stronger proof than closing immediately after open_all() returns.
    assert _wait_until(lambda: hub.status[0].frame_count >= 2)
    assert hub.status[0].opened is True
    assert hub.status[0].last_read_ok is True
    assert hub.status[0].closed_at is None

    hub.close_all()

    for cam in (0, 1):
        status = hub.status[cam]
        assert status.opened is False, (
            f"cam{cam}: opened must flip to False on close_all(), not just stop being updated"
        )
        assert status.last_read_ok is False, (
            f"cam{cam}: last_read_ok must flip to False on close_all() -- a closed hub "
            "must not keep claiming its last pumped read is still good"
        )
        assert status.closed_at is not None, (
            f"cam{cam}: closed_at must be stamped so a caller can tell exactly how "
            "stale any leftover last_read_at/frame_count fields are"
        )
    # Genuinely-historical fields (what backend/resolution this camera
    # negotiated while open, how many frames it read over its lifetime)
    # are real information, not reset to zero/None for no honesty benefit.
    assert hub.status[0].backend_used == "CAP_AVFOUNDATION"
    assert hub.status[0].frame_count >= 2


def test_close_all_summary_reports_closed_not_failed_to_open(monkeypatch):
    """summary() must distinguish "was open, then closed" from "never
    opened" -- both read as opened=False, but they're different truths
    (see CameraStatus.summary()'s own docstring)."""
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    hub.close_all()

    summary = hub.status[0].summary()
    assert "CLOSED" in summary
    assert "FAILED TO OPEN" not in summary


def test_reopen_after_close_clears_closed_at(monkeypatch):
    """Mirror-image of the close_all() fix: reopening a previously-closed
    camera must not leave a stale closed_at timestamp sitting next to
    opened=True -- the same status-honesty bug in the other direction."""
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    hub.close_all()
    assert hub.status[0].closed_at is not None

    hub.open_all()

    assert hub.status[0].opened is True
    assert hub.status[0].closed_at is None, (
        "closed_at must be cleared on a fresh open -- otherwise a reopened camera "
        "would show a stale 'closed at' timestamp while opened=True"
    )


def test_disabled_camera_is_skipped_without_attempting_to_open(monkeypatch):
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0, enabled=False)])
    ok_flags = hub.open_all()

    assert ok_flags == [False]
    assert factory.calls == []
    assert hub.status[0].last_error == "disabled in config"


def test_grab_reports_read_failure_without_raising(monkeypatch):
    factory = make_fake_capture_factory(frame_ok=False)
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()

    # First-frame warm read already failed at open time.
    assert hub.status[0].last_read_ok is False
    frame = hub.grab(0)
    assert frame is None
    assert hub.status[0].last_read_ok is False


# ---------------------------------------------------------------------------
# CameraStatus.summary() -- basic sanity, not a hardware claim
# ---------------------------------------------------------------------------


def test_status_summary_reports_failure_reason_for_unopened_camera(monkeypatch):
    factory = make_fake_capture_factory(
        fails_backends={cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2, cv2.CAP_ANY}
    )
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    summary = hub.status[0].summary()
    assert "FAILED TO OPEN" in summary


def test_status_summary_reports_backend_and_resolution_for_opened_camera(monkeypatch):
    factory = make_fake_capture_factory()
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0, width=1280, height=720, fps=30)])
    hub.open_all()
    summary = hub.status[0].summary()
    assert "CAP_AVFOUNDATION" in summary
    assert "1280x720" in summary


# ---------------------------------------------------------------------------
# fetch_snapshot / fetch_all_snapshots -- API-shape parity with
# opendarts.live.capture's HTTP-based functions
# ---------------------------------------------------------------------------


def test_fetch_snapshot_writes_png_and_returns_snapshot_shape(tmp_path, monkeypatch):
    # FakeVideoCapture (like a real driver ignoring an unsupported mode
    # request) negotiates its own fixed 64x48 regardless of what was
    # cap.set() -- this test is about fetch_snapshot()'s own plumbing
    # (PNG written, LocalSnapshot shape reflects the ACTUAL negotiated
    # frame), not config negotiation, which is covered separately above.
    def factory(device, backend):
        opens = backend == cv2.CAP_AVFOUNDATION
        return FakeVideoCapture(device, backend, opens=opens, width=64, height=48)

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0, width=1280, height=720)])
    hub.open_all()

    snap = fetch_snapshot(0, tmp_path, hub=hub)

    assert snap.cam == 0
    assert snap.width == 64
    assert snap.height == 48
    assert snap.path.exists()
    assert snap.backend_used == "CAP_AVFOUNDATION"
    assert snap.capture_latency_s is not None and snap.capture_latency_s >= 0.0

    # Written file should be a real, readable PNG of the right size.
    read_back = cv2.imread(str(snap.path))
    assert read_back is not None
    assert read_back.shape[:2] == (48, 64)


def test_fetch_snapshot_raises_with_reason_when_camera_never_opened(tmp_path, monkeypatch):
    factory = make_fake_capture_factory(
        fails_backends={cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2, cv2.CAP_ANY}
    )
    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()

    with pytest.raises(RuntimeError, match="failed to grab a frame"):
        fetch_snapshot(0, tmp_path, hub=hub)


def test_fetch_all_snapshots_skips_failed_camera_instead_of_raising(tmp_path, monkeypatch):
    def factory(device, backend):
        if device == 1:
            return FakeVideoCapture(device, backend, opens=False)
        return FakeVideoCapture(device, backend, opens=(backend == cv2.CAP_AVFOUNDATION), width=32, height=24)

    monkeypatch.setattr(cv2, "VideoCapture", factory)

    hub = LocalCameraHub(
        configs=[CameraConfig(device=0), CameraConfig(device=1), CameraConfig(device=2)]
    )
    hub.open_all()

    snaps = fetch_all_snapshots(tmp_path, n_cameras=3, hub=hub)

    assert {s.cam for s in snaps} == {0, 2}
    for snap in snaps:
        assert snap.path.exists()


def test_default_camera_devices_matches_od_confirmed_rig_config():
    """This rig's real 3 camera indices, confirmed against
    the documented per-camera device defaults -- see local_capture.py's module docstring.
    A change here should only ever follow a change in the confirmed
    real rig config, never be adjusted to make a test pass."""
    assert local_capture_module.DEFAULT_CAMERA_DEVICES == [0, 1, 2]


# ---------------------------------------------------------------------------
# last_read_at_monotonic -- frame-age-at-advance() diagnostic (2026-09-04)
# ---------------------------------------------------------------------------


def test_last_read_at_monotonic_stamped_on_open_and_on_pump_success(monkeypatch):
    """CameraStatus.last_read_at_monotonic (added alongside the
    frame-age-at-advance() diagnostic) must be stamped -- using
    time.monotonic(), never time.time() -- at both real successful-read
    sites: the open-time warm frame, and a real pump cycle. Bounding it
    between two real time.monotonic() samples taken immediately around
    each event is the actual proof it's the right clock, not merely
    "some float got set"."""
    factory = make_fake_capture_factory(frame_ok=True)
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])

    before_open = time.monotonic()
    hub.open_all()
    after_open = time.monotonic()

    status = hub.status[0]
    assert status.last_read_at is not None
    assert status.last_read_at_monotonic is not None
    assert before_open <= status.last_read_at_monotonic <= after_open

    # Let at least one real pump cycle land beyond the open-time warm
    # frame, and confirm the monotonic stamp genuinely advances with it
    # (not just the pre-existing wall-clock one).
    first_stamp = status.last_read_at_monotonic
    before_pump_cycle = time.monotonic()
    assert _wait_until(lambda: status.last_read_at_monotonic > first_stamp)
    after_pump_cycle = time.monotonic()
    assert before_pump_cycle <= status.last_read_at_monotonic <= after_pump_cycle


def test_last_read_at_monotonic_freezes_on_pump_failure_unlike_last_read_at(monkeypatch):
    """A failed pump read still updates last_read_at/last_read_ok/
    last_error (the pump's own pre-existing "last ATTEMPT, not last
    SUCCESS" convention -- see CameraStatus's own docstring), but must
    NOT move last_read_at_monotonic: grab()/grab_all() keep serving the
    last frame that WAS successfully read (a "momentary hiccup does not
    null out an otherwise-good previous frame" -- grab()'s own
    docstring), so a frame-age diagnostic must not be fooled into
    thinking that served frame just got fresher when it didn't."""
    captures: list = []

    def factory(device, backend):
        cap = FakeVideoCapture(device, backend, opens=True, frame_ok=True)
        captures.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    status = hub.status[0]
    fake_cap = captures[0]

    assert _wait_until(
        lambda: status.last_read_ok is True and status.last_read_at_monotonic is not None
    )

    fake_cap._frame_ok = False # noqa: SLF001 -- deliberately induce read failures
    # Wait for the FIRST observed failure, then snapshot both fields --
    # not before the flip (a real pump cycle can already be mid-flight
    # against the old frame_ok=True value at the instant this test flips
    # it, a genuine race that would make an earlier snapshot describe an
    # arbitrary intermediate success rather than the actual last one).
    # Whatever last_read_at_monotonic reads AT the first observed
    # failure IS the true "last successful read" stamp -- it must not
    # move again from here on, no matter how many more failures follow.
    assert _wait_until(lambda: status.last_read_ok is False)
    frozen_monotonic = status.last_read_at_monotonic
    frozen_wall_clock = status.last_read_at
    assert frozen_monotonic is not None

    # Give a couple more failing pump cycles real time to run, so this
    # isn't just "checked before the pump got a second attempt in".
    assert _wait_until(lambda: status.last_error == "cap.read() returned no frame")
    time.sleep(0.05)

    assert status.last_read_at_monotonic == frozen_monotonic, (
        "last_read_at_monotonic must stay frozen at the last SUCCESSFUL "
        "read -- it moved on a failed pump attempt, which would understate "
        "the real age of the frame grab()/grab_all() is still actually "
        "serving"
    )
    # last_read_at (pre-existing wall-clock "last attempt" semantic,
    # unchanged by this task) is expected to keep moving even through
    # failures -- confirming the freeze above is specific to the new
    # field, not an accidental side effect that also stopped the pump.
    assert status.last_read_at is not None and status.last_read_at != frozen_wall_clock


# ---------------------------------------------------------------------------
# reconfigure() -- live device reassignment, 2026-09-10.
# run_product builds ONE hub and shares it BY REFERENCE with the capture
# thread and the app, so reassignment has to mutate that object rather than
# build a replacement, or both halves keep the old one.
# ---------------------------------------------------------------------------


def test_reconfigure_replaces_devices_on_the_same_object():
    hub = LocalCameraHub(configs=[CameraConfig(device=0), CameraConfig(device=1)])
    before = id(hub)
    hub.reconfigure([CameraConfig(device=3), CameraConfig(device=4)])
    assert id(hub) == before, "must mutate in place -- callers hold this reference"
    assert [c.device for c in hub.configs] == [3, 4]


def test_reconfigure_rebuilds_status_to_match_the_new_devices():
    """status is derived from configs, so leaving it stale would report
    the OLD device for each slot -- the same 'status disagrees with
    reality' class the 2026-08-12 honesty fix exists to prevent."""
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.reconfigure([CameraConfig(device=7)])
    assert hub.status[0].device == 7


def test_reconfigure_handles_a_different_camera_count():
    hub = LocalCameraHub(configs=[CameraConfig(device=d) for d in (0, 1, 2)])
    hub.reconfigure([CameraConfig(device=5), CameraConfig(device=6)])
    assert sorted(hub.status) == [0, 1]
    assert sorted(hub._locks) == [0, 1], "a slot with no lock would KeyError on open"


def test_reconfigure_refuses_while_cameras_are_open(monkeypatch):
    """The open captures ARE the old assignment -- swapping the
    description out from under them leaves status and _caps disagreeing
    about what slot N even is."""
    factory = make_fake_capture_factory(fails_backends=set())
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    try:
        with pytest.raises(RuntimeError, match="cameras are open"):
            hub.reconfigure([CameraConfig(device=3)])
        assert [c.device for c in hub.configs] == [0], "must not partially apply"
    finally:
        hub.close_all()


def test_reconfigure_is_allowed_again_after_close(monkeypatch):
    factory = make_fake_capture_factory(fails_backends=set())
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    hub.close_all()
    hub.reconfigure([CameraConfig(device=3)])
    assert [c.device for c in hub.configs] == [3]


# ---------------------------------------------------------------------------
# MSMF hardware-transform env var, 2026-09-10.
# It only works if it is set BEFORE cv2 is imported anywhere in the process.
# The first attempt put it next to local_capture's own `import cv2`, which
# looked right and did nothing: run_product imports
# opendarts.lifecycle.settings -- which pulls cv2 in transitively -- one line
# before it imports local_capture, so cv2 had already initialised. Windows
# camera opens stayed at ~45s. A setting that is correct but late is
# indistinguishable from no setting, so the ordering is pinned here.
# ---------------------------------------------------------------------------


def test_importing_opendarts_does_not_pull_in_cv2():
    """The env var lives in opendarts/__init__.py and is only effective
    while cv2 is still unimported. If any module reachable from the
    package __init__ starts importing cv2, the variable silently goes
    back to being set too late -- with no error anywhere."""
    import subprocess
    import sys

    code = "import opendarts, sys; print('cv2' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "False", (
        "importing `opendarts` now loads cv2, so "
        "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS is set after cv2 has already "
        "initialised and has no effect"
    )


def test_backend_name_covers_the_windows_backends():
    """A log line reading UNKNOWN_BACKEND(1400) is strictly worse than
    CAP_MSMF when the backend is the thing being diagnosed -- that is
    exactly what the first Windows run produced."""
    assert local_capture_module._backend_name(cv2.CAP_MSMF) == "CAP_MSMF"
    assert local_capture_module._backend_name(cv2.CAP_DSHOW) == "CAP_DSHOW"
    assert local_capture_module._backend_name(cv2.CAP_AVFOUNDATION) == "CAP_AVFOUNDATION"


# ---------------------------------------------------------------------------
# frame_sink -- republishing captured frames, 2026-09-11.
# Added so Windows virtual cameras can be fed from the frames the pump
# already has. The safety properties matter more than the feature: this
# runs on the capture path, and a diagnostic must never be able to affect
# scoring.
# ---------------------------------------------------------------------------


def test_hub_without_a_frame_sink_is_unchanged():
    hub = LocalCameraHub()
    assert hub._frame_sink is None
    assert hub.frame_sink_errors == 0


def test_frame_sink_receives_the_cached_frames(monkeypatch):
    seen = []
    factory = make_fake_capture_factory(fails_backends=set())
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0)], frame_sink=seen.append)
    hub.open_all()
    try:
        hub._pump_once()
    finally:
        hub.close_all()
    assert seen, "sink was never called"
    assert 0 in seen[0], "sink should get a {index: frame} dict"


def test_a_raising_frame_sink_does_not_break_the_pump(monkeypatch):
    """The whole point of the guard. A sink that throws must not stop
    frames being captured or take the pump thread down -- scoring cannot
    depend on a diagnostic succeeding."""
    def boom(_frames):
        raise RuntimeError("sink exploded")

    factory = make_fake_capture_factory(fails_backends=set())
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=0)], frame_sink=boom)
    hub.open_all()
    try:
        hub._pump_once()
        hub._pump_once()
        # >= rather than ==: open_all() starts the background pump thread,
        # which calls the sink too, so the count is not just our two calls.
        assert hub.frame_sink_errors >= 2
        assert hub.status[0].frame_count >= 2, "frames must still be captured"
        assert hub._last_frames.get(0) is not None, "the cache must still fill"
    finally:
        hub.close_all()


def test_a_raising_frame_sink_logs_once_not_per_frame(monkeypatch, caplog):
    """30fps x 3 cameras is 90 tracebacks a second otherwise, which buries
    whatever the real fault was."""
    import logging

    def boom(_frames):
        raise RuntimeError("sink exploded")

    factory = make_fake_capture_factory(fails_backends=set())
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    caplog.set_level(logging.ERROR, logger="opendarts.live.local_capture")
    hub = LocalCameraHub(configs=[CameraConfig(device=0)], frame_sink=boom)
    hub.open_all()
    try:
        for _ in range(5):
            hub._pump_once()
    finally:
        hub.close_all()
    logged = [r for r in caplog.records if "frame sink" in r.getMessage()]
    assert len(logged) == 1, f"logged {len(logged)} times for one repeated fault"


# -- negotiated pixel format -------------------------------------------
#
# Surfaced 2026-09-13 for a real, physical problem: one camera set can
# share a USB dock and another cannot. That is a FORMAT difference, not a
# fault -- uncompressed YUY2 at 1280x720x30 is ~442 Mbit/s per camera
# against roughly 320 Mbit/s of usable USB 2.0 bandwidth, so one
# saturates a controller, while MJPEG compresses far enough for three.
# Nothing reported which format a camera actually got, so the difference
# was invisible from here.


def test_fourcc_decodes_the_formats_that_matter():
    from opendarts.live.local_capture import _fourcc_to_str

    # The two that decide whether three cameras fit on one bus.
    assert _fourcc_to_str(1196444237.0) == "MJPG"
    assert _fourcc_to_str(844715353.0) == "YUY2"


def test_fourcc_reports_none_rather_than_garbage_when_unknown():
    """A backend that does not report a format must read as "not
    reported", never as a plausible-looking string -- this value is used
    to decide whether a camera set can share a dock, and a fabricated
    format would send that diagnosis the wrong way."""
    from opendarts.live.local_capture import _fourcc_to_str

    for unknown in (0.0, None, -1, 3.0):
        assert _fourcc_to_str(unknown) is None, unknown


def test_camera_status_carries_and_prints_the_format():
    from opendarts.live.local_capture import CameraStatus

    status = CameraStatus(device=0)
    assert status.actual_fourcc is None, "must default to not-reported"

    status.opened = True
    status.backend_used = "CAP_MSMF"
    status.actual_width, status.actual_height, status.actual_fps = 1280, 720, 30.0
    status.actual_fourcc = "MJPG"
    # The open-time log line is where an operator actually reads this.
    assert "fourcc=MJPG" in status.summary()

    status.actual_fourcc = None
    assert "fourcc=n/a" in status.summary()


# ---------------------------------------------------------------------------
# PUMP STALL BACKOFF, 2026-09-13 (CPU task). See _stall_backoff_s()'s own
# docstring and the `if not any_live:` comment in _pump_once().
# ---------------------------------------------------------------------------


def test_stall_backoff_is_the_camera_frame_period_not_a_flat_20ms():
    """A fully-stalled hub must not cycle FASTER than a delivering one.

    The old flat `time.sleep(0.02)` cycled a stalled pump at ~50Hz while a
    healthy 30fps hub cycles at ~33ms -- and since `_frame_generation`
    bumps once per completed cycle regardless of success, every one of
    those cycles also woke the frame-driven consumer. Reverting to 0.02
    fails this: 0.02 is not >= the 30fps period.
    """
    hub = local_capture_module.LocalCameraHub(
        [local_capture_module.CameraConfig(device=0, fps=30),
         local_capture_module.CameraConfig(device=1, fps=30)]
    )
    backoff = hub._stall_backoff_s()
    assert backoff == pytest.approx(1.0 / 30.0), (
        f"expected the nominal 30fps frame period, got {backoff}"
    )
    assert backoff >= 1.0 / 30.0, (
        "a stalled cycle must never be quicker than a delivering one"
    )


def test_stall_backoff_uses_the_fastest_camera_and_never_raises():
    """Mixed rates take the FASTEST camera's period (the shortest real
    period), so the backoff stays a floor under busy-spin without
    delaying the first camera to recover. Degenerate configs fall back to
    DEFAULT_FPS rather than raising -- this runs on the pump thread on a
    failing path."""
    mixed = local_capture_module.LocalCameraHub(
        [local_capture_module.CameraConfig(device=0, fps=15),
         local_capture_module.CameraConfig(device=1, fps=60)]
    )
    assert mixed._stall_backoff_s() == pytest.approx(1.0 / 60.0)

    default = 1.0 / float(local_capture_module.DEFAULT_FPS)
    assert local_capture_module.LocalCameraHub([])._stall_backoff_s() == pytest.approx(default)
    assert local_capture_module.LocalCameraHub(
        [local_capture_module.CameraConfig(device=0, fps=0)]
    )._stall_backoff_s() == pytest.approx(default)
    # A disabled camera must not set the pace for a hub that has others.
    half_off = local_capture_module.LocalCameraHub(
        [local_capture_module.CameraConfig(device=0, fps=120, enabled=False),
         local_capture_module.CameraConfig(device=1, fps=30)]
    )
    assert half_off._stall_backoff_s() == pytest.approx(1.0 / 30.0)


# -- SYNTHETIC JPEG (local cameras with no camera JPEG) ------------------
# Always on, no switch: see the SYNTHETIC JPEG section of local_capture.


def test_synthesise_jpeg_uses_the_fixed_quality():
    """The quality is a constant, not configuration. Pinned so a change to it
    is a deliberate, reviewed edit rather than a drift."""
    assert local_capture_module.SYNTHETIC_JPEG_QUALITY == 50


def test_published_pixels_are_the_decode_of_the_published_bytes():
    """THE INVARIANT THIS FEATURE EXISTS FOR.

    We publish pixels and we keep bytes. Scoring runs on the pixels and the
    package stores the bytes, so if they ever disagree the rig is scoring
    something it did not save -- the exact failure SCORE==STORE forbids. The
    returned array must therefore be the decode of the returned bytes, NOT
    the original frame we were handed.
    """
    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, (72, 128, 3), dtype=np.uint8)

    out = local_capture_module._synthesise_jpeg(frame)
    assert out is not None
    published, data = out

    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9", "not a whole JPEG"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(published, decoded), (
        "published pixels must BE the decode of the published bytes"
    )
    # And it must genuinely be the round trip, not a passthrough of the input:
    # noise at q50 cannot survive unchanged, so equality here would mean the
    # encode was skipped and the bytes describe a different image.
    assert not np.array_equal(published, frame), (
        "published the pre-encode frame -- the stored bytes would not match it"
    )


def test_status_reports_synthetic_separately_from_passthrough():
    """`jpeg_passthrough` means the CAMERA's bytes. Synthetic bytes are ours.
    Conflating them would misreport provenance in the one place an operator
    looks to check it -- and a wrong comment about exactly this cost us real
    time (see docs/CAMERAS.md)."""
    status = local_capture_module.CameraStatus(device=0)
    assert status.jpeg_passthrough is False
    assert status.jpeg_synthetic is False
    assert status.jpeg_synthetic_quality is None


def test_pump_publishes_synthetic_pixels_that_decode_from_the_kept_bytes(monkeypatch):
    """A REAL open and pump cycle, not just the helper: a pixels-only local
    camera must come out of the cache with bytes, and the cached pixels must
    be exactly the decode of those bytes -- after the open (first frame) and
    after the pump (every later frame). The round trip runs on each camera's
    worker thread, so this also pins that moving it off the pump kept the
    pairing."""
    monkeypatch.setattr(cv2, "VideoCapture", make_fake_capture_factory(fails_backends=set()))
    hub = LocalCameraHub(configs=[CameraConfig(device=0), CameraConfig(device=1)])
    hub.open_all()
    try:
        def check(when: str) -> None:
            for i in (0, 1):
                frame, data = hub.grab_with_jpeg(i)
                assert frame is not None and data is not None, f"{when}: slot {i} has no bytes"
                decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                assert np.array_equal(frame, decoded), (
                    f"{when}: slot {i}'s cached pixels are not the decode of its bytes")
                assert hub.status[i].jpeg_synthetic is True
                assert hub.status[i].jpeg_synthetic_quality == 50
                assert hub.status[i].jpeg_passthrough is False
        check("after open")
        hub._pump_once()
        check("after a pump cycle")
    finally:
        hub.close_all()


def test_a_failed_round_trip_publishes_a_detached_copy_with_no_bytes(monkeypatch):
    """If the encode ever fails, the camera must still deliver -- raw pixels,
    no bytes (so nothing is stored that does not match), and a COPY, because
    _read_one_raw() skipped the defensive copy expecting the round trip to
    replace OpenCV's reusable buffer."""
    monkeypatch.setattr(cv2, "VideoCapture", make_fake_capture_factory(fails_backends=set()))
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    hub.open_all()
    try:
        monkeypatch.setattr(local_capture_module, "_synthesise_jpeg", lambda frame: None)
        frame_from_cap = {}
        real_raw = hub._read_one_raw
        def spy(i):
            out = real_raw(i)
            frame_from_cap[i] = out[0]
            return out
        monkeypatch.setattr(hub, "_read_one_raw", spy)
        frame, data = hub._read_one(0)
        assert frame is not None and data is None
        assert frame is not frame_from_cap[0], "must be a copy, not OpenCV's buffer"
        assert np.array_equal(frame, frame_from_cap[0])
    finally:
        hub.close_all()


def test_stream_and_replay_sources_are_never_re_encoded():
    """Synthetic JPEG is for LOCAL cameras only. A replay source must deliver
    exactly the pixels it recorded -- re-encoding them would make a replayed
    session score different pixels than it did live -- and a stream already
    carries JPEG parts of its own."""
    class PixelsOnlySource:
        def __init__(self):
            self.frame = np.random.default_rng(3).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        def read(self):
            return self.frame

    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    source = PixelsOnlySource()
    hub._sources[0] = source
    try:
        frame, data = hub._read_one(0)
    finally:
        # A fake, never started -- keep it out of the hub's own teardown.
        hub._sources.pop(0, None)
    assert frame is source.frame, "a source's pixels must pass through untouched"
    assert data is None, "no bytes may be invented for a source"
