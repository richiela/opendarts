"""/api/cameras/{cam}/stream.mjpg -- the live MJPEG preview (2026-09-11).

WHY THIS SUITE LOOKS THE WAY IT DOES. The stream replaced the dashboard's
3-second snapshot.png polling, and every one of its design constraints
exists to protect the capture loop on the 4-core rig (the pump already
drops cycles under load there). So these tests pin the PROTECTIONS, not
just the happy path: refusal when capture isn't running (no connection
held open pretending), a hard cap on concurrent streams with the slot
actually released, no re-encoding when the camera hasn't produced a new
frame, and the stream closing itself rather than serving a frozen frame
as live. The frame data itself must come from the hub's pump cache
(grab()) -- the real-hub test at the bottom proves the whole path
against an actual LocalCameraHub, same FakeVideoCapture pattern
tests/test_live_server.py uses (its own copy kept here deliberately;
this project's test files don't import each other's internals).

TESTCLIENT CANNOT CONSUME AN OPEN-ENDED STREAM -- learned the hard way
writing this file: starlette's TestClient runs the whole request through
a blocking portal call that only returns once the app FINISHES the
response, so even `client.stream(...)` buffers everything up front and
an endless MJPEG response deadlocks the test (observed as a real hang,
stacks confirmed it). Every TestClient test here therefore streams
against a controller/camera arranged so the SERVER ends the response on
its own -- which is no loss, because "the server ends the stream when
capture stops / frames stall" is exactly the behavior worth pinning.
The one thing that genuinely cannot be reached that way, slot release
on abrupt client disconnect, is tested by driving the ASGI app directly
and cancelling it mid-stream (what uvicorn does when a browser drops).
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import opendarts.live.server as server  # noqa: E402
from opendarts.live import local_capture  # noqa: E402
from opendarts.live.server import _encode_preview_jpeg, create_app  # noqa: E402

BOUNDARY = b"--" + server.MJPEG_BOUNDARY.encode()


class FakeCaptureController:
    """Only what the stream endpoint consults: is_running(). `allow`
    bounds how many is_running() checks answer True -- the endpoint
    checks once at admission and once per generator tick, so a finite
    allowance is precisely "capture stops after N ticks", which is what
    lets TestClient consume the stream at all (see module docstring)."""

    def __init__(self, allow: float = float("inf")) -> None:
        self.allow = allow
        self.calls = 0

    def is_running(self) -> bool:
        self.calls += 1
        return self.calls <= self.allow


class _TickingStatus:
    """CameraStatus stand-in whose frame_count advances on every read --
    a camera producing faster than the stream's FPS cap, so every
    generator tick sees a new frame."""

    def __init__(self) -> None:
        self._n = 0

    @property
    def frame_count(self) -> int:
        self._n += 1
        return self._n


class _FrozenStatus:
    """A camera that stops producing: frame_count never advances."""

    frame_count = 7


class FakeHub:
    """Duck-typed stand-in for LocalCameraHub as the stream endpoint
    sees it: configs (range check), status (frame_count), grab (the
    cached frame). No pump thread -- these tests control frame_count
    directly to make skip/stall behavior deterministic instead of
    racing a real pump."""

    def __init__(self, status_obj) -> None:
        self.configs = [object()]
        self.status = {0: status_obj}
        self._frame = np.full((48, 64, 3), 128, dtype=np.uint8)

    def grab(self, i: int):
        return self._frame if i == 0 else None


def _make_app(tmp_path, hub, controller):
    return create_app(
        package_root=tmp_path / "pkgs",
        enable_background_poll=False,
        local_hub=hub,
        controller=controller,
    )


def _split_parts(buf: bytes) -> list[tuple[bytes, bytes]]:
    """(headers, payload) per complete multipart part, payload sliced
    exactly by the part's own Content-Length header."""
    parts = []
    for chunk in buf.split(BOUNDARY + b"\r\n")[1:]:
        if b"\r\n\r\n" not in chunk:
            continue
        head, rest = chunk.split(b"\r\n\r\n", 1)
        length = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        parts.append((head, rest[:length]))
    return parts


def test_stream_refused_when_capture_not_running(tmp_path):
    app = _make_app(tmp_path, FakeHub(_TickingStatus()), FakeCaptureController(allow=0))
    resp = TestClient(app).get("/api/cameras/0/stream.mjpg")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert "not running" in body["reason"]


def test_stream_refused_in_standalone_mode_without_controller(tmp_path):
    # No controller means this dashboard cannot start capture at all
    # (standalone CLI) -- same honest refusal, not a hung connection.
    app = _make_app(tmp_path, FakeHub(_TickingStatus()), controller=None)
    resp = TestClient(app).get("/api/cameras/0/stream.mjpg")
    assert resp.status_code == 503


def test_stream_404_for_camera_outside_configured_range(tmp_path):
    app = _make_app(tmp_path, FakeHub(_TickingStatus()), FakeCaptureController())
    resp = TestClient(app).get("/api/cameras/7/stream.mjpg")
    assert resp.status_code == 404


def test_stream_serves_jpeg_parts_then_ends_when_capture_stops(tmp_path):
    """The core contract in one response: while capture runs, well-formed
    multipart JPEG parts; when it stops, the response ENDS (this request
    completing at all is that assertion -- see module docstring)."""
    controller = FakeCaptureController(allow=5)  # admission + 4 stream ticks
    app = _make_app(tmp_path, FakeHub(_TickingStatus()), controller)

    resp = TestClient(app).get("/api/cameras/0/stream.mjpg")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == (
        f"multipart/x-mixed-replace; boundary={server.MJPEG_BOUNDARY}"
    )
    parts = _split_parts(resp.content)
    assert len(parts) == 4  # one per allowed tick -- the ticking camera never skips
    for head, payload in parts:
        assert b"Content-Type: image/jpeg" in head
        assert payload[:2] == b"\xff\xd8"  # JPEG SOI marker
    # And a part decodes back to the fake camera's own geometry --
    # proving the payload is the grabbed frame, not framing noise.
    decoded = cv2.imdecode(np.frombuffer(parts[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (48, 64, 3)


def test_stream_skips_unchanged_frames_and_closes_on_stall(tmp_path, monkeypatch):
    """Two protections in one observable behavior: a frame_count that
    never advances must produce exactly ONE encoded part (never a
    re-encode of the same cached frame -- that CPU belongs to scoring),
    and the stream must then close itself instead of holding the
    connection open serving a frozen picture as if it were live."""
    monkeypatch.setattr(server, "MJPEG_STALL_TIMEOUT_S", 0.3)
    app = _make_app(tmp_path, FakeHub(_FrozenStatus()), FakeCaptureController())
    state = app.state.opendarts_state

    t0 = time.monotonic()
    resp = TestClient(app).get("/api/cameras/0/stream.mjpg")

    assert resp.status_code == 200
    assert len(_split_parts(resp.content)) == 1
    assert time.monotonic() - t0 < 3.0  # closed by the stall timeout, not luck
    # Normal-completion path of the slot bookkeeping: the generator's
    # finally ran before the response finished.
    assert state.mjpeg_client_count == 0


def test_stream_client_cap_refuses_at_the_limit(tmp_path):
    app = _make_app(tmp_path, FakeHub(_TickingStatus()), FakeCaptureController())
    state = app.state.opendarts_state

    state.mjpeg_client_count = server._mjpeg_cap(3, server.MJPEG_MAX_PREVIEW_VIEWERS)
    resp = TestClient(app).get("/api/cameras/0/stream.mjpg")
    assert resp.status_code == 503
    assert "too many" in resp.json()["reason"]


def test_stream_slot_released_when_client_disconnects_mid_stream(tmp_path):
    """A browser dropping mid-stream cancels the response task (that is
    how uvicorn delivers disconnects) -- the generator's finally must
    still release its preview slot, or dead clients would eat
    the cap until nobody can open a preview at all. TestClient cannot
    exercise this (module docstring), so this drives the ASGI app
    directly and cancels it after the first body chunk."""
    app = _make_app(tmp_path, FakeHub(_TickingStatus()), FakeCaptureController())
    state = app.state.opendarts_state

    async def scenario():
        got_body = asyncio.Event()
        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/cameras/0/stream.mjpg",
            "raw_path": b"/api/cameras/0/stream.mjpg",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
        }

        async def receive():
            await asyncio.Event().wait()  # client never speaks; it just vanishes

        async def send(message):
            if message["type"] == "http.response.body" and message.get("body"):
                got_body.set()

        task = asyncio.ensure_future(app(scope, receive, send))
        await asyncio.wait_for(got_body.wait(), timeout=5.0)
        assert state.mjpeg_client_count == 1
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        # The cancel unwinds the generator at its current await; give the
        # loop a beat in case cleanup is scheduled rather than inline.
        for _ in range(50):
            if state.mjpeg_client_count == 0:
                break
            await asyncio.sleep(0.01)
        assert state.mjpeg_client_count == 0

    asyncio.run(scenario())


def test_encode_preview_jpeg_downscales_wide_frames_only():
    wide = np.zeros((1080, 1920, 3), dtype=np.uint8)
    small = np.zeros((48, 64, 3), dtype=np.uint8)

    decoded_wide = cv2.imdecode(
        np.frombuffer(_encode_preview_jpeg(wide), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded_wide.shape[1] == server.MJPEG_MAX_WIDTH
    assert decoded_wide.shape[0] == round(1080 * server.MJPEG_MAX_WIDTH / 1920)

    decoded_small = cv2.imdecode(
        np.frombuffer(_encode_preview_jpeg(small), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert decoded_small.shape == (48, 64, 3)  # never upscaled


def test_dashboard_streams_the_preview_and_layers_the_overlay_over_it(tmp_path):
    """The division of labour the dashboard settled on, REVISED
    2026-09-12: raw preview = MJPEG stream, calibration overlay = a
    transparent PNG layered on top of that stream.

    This test used to assert the opposite half -- that the overlay stayed
    a baked-in still fetched on the 3-second tick. The CPU reasoning
    behind that (do not re-draw a static board once per streamed frame)
    was right and still holds; what was wrong was the conclusion drawn
    from it. Because the still REPLACED the tile's img, a camera whose
    calibration came back ok lost its live picture entirely, so a working
    rig showed a slideshow and a broken one showed video. A transparent
    layer keeps both properties: the board is drawn once per
    recalibration, and the video never stops.

    Plus the connection hygiene that keeps idle previews free:
    visibilitychange must be wired, and clearing img.src -- the thing
    that actually closes a stream -- must exist.
    """
    app = create_app(package_root=tmp_path / "pkgs", enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "/stream.mjpg?t=" in html
    # The overlay is the TRANSPARENT endpoint now, and it is keyed by the
    # calibration it draws rather than by a clock.
    assert "/overlay-rgba.png?v=" in html
    assert "/overlay.png?t=" not in html, (
        "the baked overlay is a full photograph -- layering it would hide "
        "the very stream it sits on"
    )
    assert "visibilitychange" in html
    assert "function stopCamStream(" in html
    # The old 3-second snapshot poll for the raw preview is gone.
    assert "/snapshot.png?t=" not in html


# -- real-hub integration ----------------------------------------------


class FakeVideoCapture:
    """Minimal cv2.VideoCapture stand-in -- always opens, always returns
    a fixed-size solid-color frame (see tests/test_live_server.py's copy
    and its provenance note)."""

    def __init__(self, device, backend) -> None:
        self.device = device
        self.backend = backend

    def isOpened(self) -> bool:  # noqa: N802 -- matches cv2's own method name
        return True

    def release(self) -> None:
        pass

    def set(self, prop: int, value: float) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 0.0

    def read(self):
        return True, np.full((48, 64, 3), 128, dtype=np.uint8)


@pytest.fixture()
def _fake_local_hub(monkeypatch):
    """Real LocalCameraHub (pump thread and all) against FakeVideoCapture,
    closed after the test -- an orphaned pump thread otherwise busy-spins
    for the rest of the pytest process and spams shutdown tracebacks (see
    tests/test_live_server.py's _stop_pump_threads_after_every_test for
    the incident writeup)."""
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    hub = local_capture.LocalCameraHub(configs=[local_capture.CameraConfig(device=0)])
    hub.open_all()
    yield hub
    hub.close_all()


def test_stream_serves_pumped_frames_from_real_hub(tmp_path, _fake_local_hub):
    """End to end through the real cache path: pump thread fills
    _last_frames, stream grab()s from it (never a second cv2 read --
    the exact contention the pump exists to prevent), and the client
    receives decodable JPEGs of the pumped frame. Finite controller
    allowance ends the stream so TestClient can return at all (module
    docstring); the real pump may or may not land new frames within
    those ticks, so this asserts on the first part, not a count."""
    app = _make_app(tmp_path, _fake_local_hub, FakeCaptureController(allow=6))

    resp = TestClient(app).get("/api/cameras/0/stream.mjpg")

    assert resp.status_code == 200
    parts = _split_parts(resp.content)
    assert len(parts) >= 1  # first part is grabbed immediately at admission
    decoded = cv2.imdecode(np.frombuffer(parts[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (48, 64, 3)

def test_transport_mode_never_sleeps_zero_between_polls():
    """A polling loop that sleeps 0 is a busy-wait, and this one runs ON
    THE EVENT LOOP -- so it starves every other request the server is
    trying to serve.

    Measured on the real rig when `?full=1` first shipped with
    `interval = 0.0`: POST /api/stop took roughly 45 seconds, because it
    had to be handled by the same event loop three spinning stream
    generators were monopolising. One spin per camera per consumer.

    Same lesson as the capture loop's stall backoff in 1eeae8f: a loop
    with nothing to do must still wait. Reverting to 0.0 fails this.
    """
    from opendarts.live import server

    assert server.MJPEG_TRANSPORT_POLL_S > 0, "transport poll interval must not be zero"
    # Fast enough never to pace a 30fps camera...
    assert server.MJPEG_TRANSPORT_POLL_S < 1.0 / 60.0
    # ...and the source must not reintroduce a zero literal.
    src = server._render_dashboard_html.__module__ and __import__("pathlib").Path(
        server.__file__
    ).read_text()
    assert "interval = 0.0 if full" not in src, (
        "transport mode is sleeping 0 again -- that is a busy-wait on the event loop"
    )
    assert "MJPEG_TRANSPORT_POLL_S if full" in src


class _PassthroughHub(FakeHub):
    """A hub whose slot kept the camera's own JPEG (local_capture's JPEG
    PASSTHROUGH section)."""

    def __init__(self, status_obj) -> None:
        super().__init__(status_obj)
        ok, buf = cv2.imencode(".jpg", self._frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        # Marked so it cannot be mistaken for an encode of the same pixels.
        self.camera_jpeg = buf.tobytes()[:-2] + b"\xff\xfe\x00\x02\xff\xd9"

    def grab_with_jpeg(self, i: int):
        return (self._frame, self.camera_jpeg) if i == 0 else (None, None)


def test_full_frame_stream_forwards_the_camera_jpeg(tmp_path):
    hub = _PassthroughHub(_TickingStatus())
    app = _make_app(tmp_path, hub, FakeCaptureController(allow=4))
    resp = TestClient(app).get("/api/cameras/0/stream.mjpg?full=1")
    parts = _split_parts(resp.content)
    assert parts and all(p == hub.camera_jpeg for _h, p in parts)


def test_preview_stream_still_encodes_its_own_jpeg(tmp_path):
    """The preview is downscaled and quality-capped for browsers; the
    camera's full-size bytes are for full-frame consumers only."""
    hub = _PassthroughHub(_TickingStatus())
    app = _make_app(tmp_path, hub, FakeCaptureController(allow=4))
    resp = TestClient(app).get("/api/cameras/0/stream.mjpg")
    parts = _split_parts(resp.content)
    assert parts and all(p != hub.camera_jpeg for _h, p in parts)
