"""A camera slot fed over HTTP, and the one hub that serves both kinds.

Verified end to end against a real publisher before the first version of
these were written: three of the rig's live cameras consumed over HTTP, first
frame in 152ms, grab() at 0.2us.

WHY THREE CAMERAS IN EVERY COUNTING TEST. The generation-units bug
(2026-09-15) shipped past a full suite because EVERY generation test used
ONE camera, where "once per frame" and "once per frame set" are the same
number. With three they differ by 3x, which is the whole bug.

WHY THE FAKES DEFAULT THE WAY THE REAL HUB DOES. The phantom-camera bug
shipped past its tests because the fakes treated `configs=[]` as "no
cameras", while the real hub reads `configs or [CameraConfig(device=d) for
d in DEFAULT_CAMERA_DEVICES]` -- so BOTH None and [] mean "the default
three devices". The hub tests below use the REAL hub against a mocked
`cv2.VideoCapture`, which removes the question entirely: if a local device
is opened, the mock records it.
"""
from __future__ import annotations

import io
import logging
import threading
import time

import cv2
import numpy as np
import pytest

from opendarts.live import local_capture, remote_capture
from opendarts.live.local_capture import CameraConfig, CameraHub
from opendarts.live.remote_capture import (
    StreamSource,
    _MultipartReader,
    build_hub,
    urls_for,
)

BOUNDARY = "opendarts-mjpeg-frame"

LOCAL_GREY = 128
STREAM_COLOUR = (10, 20, 30)


def _jpeg(colour=STREAM_COLOUR, size=(64, 48)) -> bytes:
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    frame[:] = colour
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


def _part(jpg: bytes, *, with_length: bool = True) -> bytes:
    head = b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n"
    if with_length:
        head += f"Content-Length: {len(jpg)}\r\n".encode()
    return head + b"\r\n" + jpg + b"\r\n"


class _FakeResponse:
    """Stands in for urlopen's return -- a streaming body plus headers."""

    def __init__(self, body: bytes, boundary: str = BOUNDARY) -> None:
        self._buf = io.BytesIO(body)
        self.headers = {"Content-Type": f"multipart/x-mixed-replace; boundary={boundary}"}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self): return self
    def __exit__(self, *a): return False


def _serve(monkeypatch, body: bytes) -> None:
    """Every stream URL gets the same finite body, re-served on reconnect."""
    monkeypatch.setattr(remote_capture.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResponse(body))


def _a_few_frames(n: int = 8) -> bytes:
    return b"".join(_part(_jpeg()) for _ in range(n))


class _FakeVideoCapture:
    """cv2.VideoCapture stand-in. Deliberately the same shape as
    tests/test_local_capture.py's, so the two files cannot drift on what a
    camera is."""

    def __init__(self, device, backend, *, width=64, height=48) -> None:
        self.device = device
        self.backend = backend
        self.width = width
        self.height = height
        self._opened = True
        self.read_calls = 0
        self.released = False

    def isOpened(self): # noqa: N802 -- matches cv2's own method name
        return self._opened

    def release(self):
        self.released = True
        self._opened = False

    def set(self, prop, value):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        self.read_calls += 1
        return True, np.full((self.height, self.width, 3), LOCAL_GREY, np.uint8)


def _mock_local_cameras(monkeypatch) -> list:
    """Returns the list of (device, backend) pairs cv2 was asked to open.

    THIS LIST IS THE PHANTOM-CAMERA ASSERTION. A slot reading a stream
    must never appear in it -- on real hardware each entry that should not
    be there costs ~2.6s of failed open, serially, before the capture loop
    can run."""
    opened: list = []

    def factory(device, backend):
        opened.append((device, backend))
        return _FakeVideoCapture(device, backend)

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    return opened


@pytest.fixture(autouse=True)
def _close_every_hub():
    """Every hub built in this file gets closed, whether or not the test
    remembered. open_all() starts a real pump thread and real reader
    threads; against instant fakes the pump free-runs, so a leaked one is
    a busy-spinning daemon thread for the rest of the pytest process and a
    real source of timing flakiness in later tests. Same reasoning as
    tests/test_local_capture.py's own version of this fixture."""
    created: list[CameraHub] = []
    original = CameraHub.__init__

    def _tracking(self, *args, **kwargs):
        original(self, *args, **kwargs)
        created.append(self)

    CameraHub.__init__ = _tracking # type: ignore[method-assign]
    try:
        yield
    finally:
        CameraHub.__init__ = original # type: ignore[method-assign]
        for hub in created:
            hub.close_all()


def _wait_until(predicate, timeout: float = 4.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# The transport
# ---------------------------------------------------------------------------

def test_urls_request_transport_mode_not_the_preview():
    """A consumer SCORES these frames. The publisher's defaults are a
    downscaled, rate-capped dashboard preview -- right for a browser tab,
    wrong as a scoring input."""
    urls = urls_for("http://rig:8420/", 3)
    assert len(urls) == 3
    for i, u in enumerate(urls):
        assert u == f"http://rig:8420/api/cameras/{i}/stream.mjpg?full=1"


def test_multipart_reader_uses_content_length():
    jpgs = [_jpeg((i * 10, 0, 0)) for i in range(3)]
    body = b"".join(_part(j) for j in jpgs)
    reader = _MultipartReader(io.BytesIO(body), BOUNDARY)
    for expected in jpgs:
        assert reader.next_jpeg() == expected
    assert reader.next_jpeg() is None  # stream ended


def test_multipart_reader_falls_back_to_boundary_scanning():
    """Our publisher always sends Content-Length; a third-party one may
    not, and the payload then runs to the next boundary."""
    jpgs = [_jpeg((0, i * 10, 0)) for i in range(2)]
    body = b"".join(_part(j, with_length=False) for j in jpgs)
    body += b"--" + BOUNDARY.encode() + b"--\r\n"
    reader = _MultipartReader(io.BytesIO(body), BOUNDARY)
    assert reader.next_jpeg() == jpgs[0]
    assert reader.next_jpeg() == jpgs[1]


def test_multipart_reader_rejects_an_absurd_content_length():
    """A corrupt length must not become a multi-gigabyte read."""
    body = (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n"
            b"Content-Length: 999999999999\r\n\r\n")
    assert _MultipartReader(io.BytesIO(body), BOUNDARY).next_jpeg() is None


# ---------------------------------------------------------------------------
# One stream slot, on its own
# ---------------------------------------------------------------------------

def _source(monkeypatch, body: bytes, slot: int = 0) -> StreamSource:
    _serve(monkeypatch, body)
    src = StreamSource(slot, f"http://fake:8420/api/cameras/{slot}/stream.mjpg",
                       local_capture.CameraStatus(device=slot))
    src.start()
    return src


def test_a_decoded_frame_reaches_the_pump_and_the_status_is_honest(monkeypatch):
    src = _source(monkeypatch, _a_few_frames())
    try:
        frame = src.read()
        assert frame is not None and frame.shape == (48, 64, 3)
        assert src.status.open_latency_s is not None
        assert src.status.first_frame_latency_s is not None
        assert src.status.backend_used == "mjpeg"
    finally:
        src.stop(); src.join()


def test_read_blocks_for_a_NEW_frame_rather_than_reserving_the_last_one(monkeypatch):
    """This is what paces the pump on a stream slot.

    A source that answered "here is the newest frame I have" immediately
    would leave the pump free-running: it would spin, hand the same frame
    back over and over, and bump the generation many times per real frame
    set -- the frame-counter-units bug in a new costume. A local slot
    needs nothing like this because cap.read() blocks on the driver."""
    src = _source(monkeypatch, _a_few_frames(40))
    try:
        first = src.read()
        assert first is not None
        # Whatever comes back next must be a DIFFERENT delivery, not the
        # same one re-served -- the sequence number is what read() checks.
        served = src._served
        assert src.read() is not None
        assert src._served > served, "read() re-served a frame the pump already had"
    finally:
        src.stop(); src.join()


def test_a_quiet_stream_gives_up_once_and_then_answers_immediately(monkeypatch):
    """A dead slot must not hold up a cycle the other slots could finish.

    The budget is measured from the LAST FRAME, not from the start of the
    wait, so a stream that dies costs exactly one cycle of it. Without
    that, one dead stream would cap a rig with three healthy local cameras
    at 2Hz."""
    monkeypatch.setattr(remote_capture, "RECONNECT_DELAY_S", 60.0)
    src = _source(monkeypatch, _part(_jpeg())) # exactly one frame, then quiet
    try:
        assert src.read() is not None, "the one real frame"
        t0 = time.monotonic()
        assert src.read() is None, "nothing more is coming"
        waited = time.monotonic() - t0
        assert waited >= remote_capture.STREAM_FRAME_WAIT_S * 0.5, (
            f"gave up after {waited:.3f}s -- it must wait out ordinary jitter")
        t1 = time.monotonic()
        assert src.read() is None
        assert (time.monotonic() - t1) < 0.1, (
            "a stream already known to be quiet must answer at once")
    finally:
        src.stop(); src.join()


def test_stop_wakes_a_read_that_is_parked(monkeypatch):
    """close_all() signals every source before joining anything; a read
    still parked on its budget would otherwise gate shutdown."""
    monkeypatch.setattr(remote_capture.urllib.request, "urlopen",
                        lambda *a, **k: _FakeResponse(b""))
    src = StreamSource(0, "http://fake/x", local_capture.CameraStatus(device=0))
    src.start()
    done = threading.Event()

    def reader():
        src.read()
        done.set()

    threading.Thread(target=reader, daemon=True).start()
    time.sleep(0.05)
    src.stop()
    assert done.wait(0.5), "stop() left a read parked on its full budget"
    src.join()


def test_a_refused_stream_is_recorded_without_killing_the_reader(monkeypatch):
    """503 is the publisher honestly saying its capture loop is not
    running. Verified against the real rig in that state: readers stay
    alive and retry, because 'not started yet' is ordinary."""
    import urllib.error

    def boom(*a, **k):
        raise urllib.error.HTTPError("http://fake", 503, "unavailable", {}, None)

    monkeypatch.setattr(remote_capture.urllib.request, "urlopen", boom)
    monkeypatch.setattr(remote_capture, "RECONNECT_DELAY_S", 0.05)
    src = StreamSource(0, "http://fake/x", local_capture.CameraStatus(device=0))
    src.start()
    try:
        assert _wait_until(lambda: "503" in (src.status.last_error or ""))
        assert src.status.opened is False
        assert src.status.last_read_ok is False
        assert src.read() is None
        assert src._thread is not None and src._thread.is_alive(), (
            "the reader died instead of retrying")
    finally:
        src.stop(); src.join()


def test_a_reader_never_dies_on_an_unexpected_error(monkeypatch):
    """A reader dying silently leaves a slot permanently dark with nothing
    to show for it -- the hardest failure to diagnose from the dashboard."""
    def boom(*a, **k):
        raise RuntimeError("something nobody predicted")

    monkeypatch.setattr(remote_capture.urllib.request, "urlopen", boom)
    monkeypatch.setattr(remote_capture, "RECONNECT_DELAY_S", 0.05)
    src = StreamSource(0, "http://fake/x", local_capture.CameraStatus(device=0))
    src.start()
    try:
        assert _wait_until(lambda: "RuntimeError" in (src.status.last_error or ""))
        assert src._thread.is_alive()
    finally:
        src.stop(); src.join()


# ---------------------------------------------------------------------------
# A stream that fails must SAY SO. The reader recorded failures only on the
# CameraStatus and logged nothing, so a camera reconnecting once a second
# was invisible: fps fell with no stated reason. That is the same silent
# failure the publisher's connection cap had -- the operator's evidence
# reads "no errors" while nothing works.
# ---------------------------------------------------------------------------

def _drive_reader(monkeypatch, src: StreamSource, behaviour) -> None:
    monkeypatch.setattr(src, "_stream_once", behaviour)
    monkeypatch.setattr(remote_capture, "RECONNECT_DELAY_S", 0.001)
    src._reader_loop()


def test_a_failing_stream_is_logged_not_only_recorded(monkeypatch, caplog):
    src = StreamSource(0, "http://a/0", local_capture.CameraStatus(device=0))
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] >= 3:
            src._stop.set()
        raise OSError("connection refused")

    with caplog.at_level(logging.WARNING, logger=remote_capture.log.name):
        _drive_reader(monkeypatch, src, boom)
    assert any("stream is down" in r.message for r in caplog.records), (
        "a dead stream logged nothing")


def test_a_stall_is_logged_even_though_it_does_not_raise(monkeypatch, caplog):
    """THE CASE THAT MATTERS FOR BANDWIDTH. _stream_once gives up after
    READ_STALL_TIMEOUT_S by returning cleanly with last_read_ok=False --
    it does not raise, so an exception-only handler stays silent through
    exactly the failure that starvation produces."""
    src = StreamSource(0, "http://a/0", local_capture.CameraStatus(device=0))
    calls = {"n": 0}

    def stall():
        calls["n"] += 1
        src.status.last_read_ok = False
        src.status.last_error = "read stalled"
        if calls["n"] >= 2:
            src._stop.set()
        return # cleanly, like the real stall path

    with caplog.at_level(logging.WARNING, logger=remote_capture.log.name):
        _drive_reader(monkeypatch, src, stall)
    assert any("stream is down" in r.message for r in caplog.records), (
        "a stalled stream logged nothing -- the bandwidth case is silent")


def test_repeated_failures_are_throttled(monkeypatch, caplog):
    """A reader retries every second forever. One line per camera per
    second would bury every other log on the rig -- which is the excuse
    that kept it silent to begin with."""
    src = StreamSource(0, "http://a/0", local_capture.CameraStatus(device=0))
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] >= 25:
            src._stop.set()
        raise OSError("refused")

    with caplog.at_level(logging.WARNING, logger=remote_capture.log.name):
        _drive_reader(monkeypatch, src, boom)
    down = [r for r in caplog.records if "stream is down" in r.message]
    assert len(down) == 1, f"25 failures produced {len(down)} lines, expected 1"


def test_recovery_is_logged_so_a_blip_leaves_a_trace(monkeypatch, caplog):
    src = StreamSource(0, "http://a/0", local_capture.CameraStatus(device=0))
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("refused")
        src.status.last_read_ok = True
        src._stop.set()

    with caplog.at_level(logging.INFO, logger=remote_capture.log.name):
        _drive_reader(monkeypatch, src, flaky)
    assert any("recovered" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# ONE HUB, N SLOTS. Which slots open hardware, and which slot number each
# frame is published under. The wrapper these replace kept two dense 0..n-1
# numbering schemes and a map between them; every bug found on 2026-09-15
# was a failure of that map rather than of either child's own logic.
# ---------------------------------------------------------------------------

def _sink_recorder(hub_box=None):
    """A frame sink that records each published set, and optionally the
    generation in force when it was published.

    The sink runs ON the pump thread, synchronously inside the cycle that
    just bumped the generation, so the pairing is exact rather than
    sampled -- which is what makes "one bump per SET" assertable as an
    identity instead of a rate."""
    calls: list[dict] = []
    gens: list[int] = []

    def sink(frames):
        calls.append(dict(frames))
        if hub_box:
            gens.append(hub_box[0].frame_generation())

    return sink, calls, gens


def test_an_all_local_rig_opens_exactly_its_own_devices_and_no_others(monkeypatch):
    opened = _mock_local_cameras(monkeypatch)
    hub = build_hub([4, 5, 6], None)
    assert hub.open_all() == [True, True, True]
    assert sorted({dev for dev, _backend in opened}) == [4, 5, 6]
    hub.close_all()


def test_an_all_stream_rig_never_opens_a_local_camera(monkeypatch):
    """THE PHANTOM-CAMERA REGRESSION. The hub reads `configs or [default
    devices]`, so both None and [] mean "the default three" -- there is no
    way to spell "no cameras" in its constructor. The wrapper this
    replaced therefore built a local child with configs=None on an
    all-stream rig and opened three devices the machine did not have:
    ~2.6s each, serially, before the capture loop could run.

    Structural now rather than guarded: open_all() only calls _open_one()
    for a slot with no URL."""
    opened = _mock_local_cameras(monkeypatch)
    _serve(monkeypatch, _a_few_frames())
    hub = build_hub([0, 1, 2], ["http://a/0", "http://a/1", "http://a/2"])
    assert hub.open_all() == [True, True, True]
    assert opened == [], f"opened local cameras on an all-stream rig: {opened}"
    hub.close_all()


def test_a_url_list_on_its_own_sizes_the_hub(monkeypatch):
    """`--camera-url http://rig/0` with nothing else said must build ONE
    slot, not three.

    The hub's own `configs or [the default three devices]` would otherwise
    give three, and slots 1 and 2 would become local devices nobody asked
    for -- the phantom-camera bug reached from the other side."""
    opened = _mock_local_cameras(monkeypatch)
    _serve(monkeypatch, _a_few_frames())
    hub = build_hub(None, ["http://a/0"])
    assert len(hub.configs) == 1
    assert hub.slot_urls == ["http://a/0"]
    hub.open_all()
    assert opened == []
    hub.close_all()


def test_saying_nothing_still_builds_the_default_three_local_cameras(monkeypatch):
    """The branch above must not have moved the no-config default. A rig
    with no camera keys in its config builds exactly the hub it always
    did."""
    opened = _mock_local_cameras(monkeypatch)
    hub = build_hub(None, None)
    assert [c.device for c in hub.configs] == local_capture.DEFAULT_CAMERA_DEVICES
    assert hub.slot_urls == [None, None, None]
    hub.open_all()
    assert sorted({d for d, _b in opened}) == local_capture.DEFAULT_CAMERA_DEVICES
    hub.close_all()


def test_a_mixed_rig_opens_only_its_local_slots(monkeypatch):
    opened = _mock_local_cameras(monkeypatch)
    _serve(monkeypatch, _a_few_frames())
    hub = build_hub([7, 8, 9], [None, "http://a/1", None])
    hub.open_all()
    assert sorted({dev for dev, _backend in opened}) == [7, 9], (
        "the stream slot's device index was opened as hardware")
    hub.close_all()


def test_swapping_to_all_streams_stops_opening_local_cameras(monkeypatch):
    """The live-swap path must get this right too: going all-remote at
    runtime must release the cameras, not keep reopening them."""
    opened = _mock_local_cameras(monkeypatch)
    _serve(monkeypatch, _a_few_frames())
    hub = build_hub([0, 1, 2], None)
    hub.open_all()
    assert len(opened) == 3
    hub.close_all()
    opened.clear()
    hub.reconfigure(None, ["http://a/0", "http://a/1", "http://a/2"])
    hub.open_all()
    assert opened == [], "still opening local cameras after the swap"
    hub.close_all()


def test_a_mixed_rig_publishes_the_correct_slot_numbers(monkeypatch):
    """THE CASE THE OLD DENSE INDEXING GOT WRONG. With the stream in the
    MIDDLE, a scheme that numbered each source densely from zero would
    publish the stream as slot 0 and the two cameras as slots 0 and 1 --
    crossing the feeds silently, which on a dartboard means scoring the
    wrong camera's view of the wrong sector."""
    _mock_local_cameras(monkeypatch)
    _serve(monkeypatch, _a_few_frames(200))
    sink, calls, _gens = _sink_recorder()
    hub = build_hub([0, 1, 2], [None, "http://a/1", None], frame_sink=sink)
    hub.open_all()
    try:
        assert _wait_until(lambda: any(set(c) == {0, 1, 2} for c in calls)), (
            f"no full frame set was published; saw {[sorted(c) for c in calls]}")
        full = next(c for c in calls if set(c) == {0, 1, 2})
        assert int(full[0][0, 0, 0]) == LOCAL_GREY, "slot 0 is a local camera"
        assert int(full[2][0, 0, 0]) == LOCAL_GREY, "slot 2 is a local camera"
        # JPEG is lossy, so this is the stream's colour within round-trip
        # error rather than exactly it. Grey would be ~50 away.
        b, g, r = (int(v) for v in full[1][0, 0])
        assert abs(b - STREAM_COLOUR[0]) < 8 and abs(g - STREAM_COLOUR[1]) < 8 \
            and abs(r - STREAM_COLOUR[2]) < 8, f"slot 1 is not the stream: {(b, g, r)}"
        # ...and the same frames are what grab() serves, under the same
        # slot numbers.
        assert int(hub.grab(0)[0, 0, 0]) == LOCAL_GREY
        assert int(hub.grab(1)[0, 0, 0]) != LOCAL_GREY
        assert sorted(hub.grab_all()) == [0, 1, 2]
    finally:
        hub.close_all()


def test_the_sink_fires_once_per_frame_set_with_every_slot_present(monkeypatch):
    """Not once per camera, and not with a partial set.

    On an all-stream rig this used to fire NOT AT ALL: set_frame_sink
    forwarded to a local child that never pumped, so the virtual cameras
    received nothing while /api/frame-health reported frame_sink_attached true.

    PRESENT MEANS A FRAME, NOT A KEY. An earlier version of this test
    asserted `set(call) == {0, 1, 2}` and passed on a set of three Nones:
    the cache is seeded with a None per slot at open, so before any stream
    connected the sink was being handed the right KEYS and no pixels. The
    sink now carries only slots that actually have a frame, matching
    grab_all()'s long-standing "absent, not None" contract."""
    _serve(monkeypatch, _a_few_frames(400))
    sink, calls, _gens = _sink_recorder()
    hub = build_hub(None, ["http://a/0", "http://a/1", "http://a/2"], frame_sink=sink)
    hub.open_all()
    try:
        assert _wait_until(lambda: len(calls) >= 5)
    finally:
        hub.close_all()
    assert not any(f is None for c in calls for f in c.values()), (
        "the sink was handed a slot with no frame in it")
    full = [c for c in calls if set(c) == {0, 1, 2}]
    assert len(full) >= 4, f"only {len(full)} of {len(calls)} calls carried every slot"


def test_the_sink_is_not_called_at_all_while_nothing_has_a_frame(monkeypatch):
    """A rig whose streams are all down must publish NOTHING, rather than
    a dict of Nones once per pump cycle.

    The publishers each reject a None frame, so this was invisible rather
    than broken -- but "invisible" is how ~30 pointless calls a second
    survive, and it made the test above pass on a set carrying no pixels.
    Found by running the real publisher over a real socket."""
    monkeypatch.setattr(remote_capture.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    monkeypatch.setattr(remote_capture, "RECONNECT_DELAY_S", 0.02)
    sink, calls, _gens = _sink_recorder()
    hub = build_hub(None, ["http://a/0", "http://a/1", "http://a/2"], frame_sink=sink)
    hub.open_all()
    try:
        # Long enough that a per-cycle sink call would have fired many
        # times: the pump still cycles, it just has nothing to publish.
        time.sleep(remote_capture.STREAM_FRAME_WAIT_S + 0.4)
        assert hub.frame_generation() > 3, "the pump was not even running"
        assert calls == [], f"published {len(calls)} empty frame set(s)"
    finally:
        hub.close_all()


def test_the_generation_counts_frame_SETS_not_frames(monkeypatch):
    """THREE cameras, because with one the bug is invisible.

    capture_daemon computes `dropped = new_gen - last_gen - 1` and reports
    it as frame sets missed. A counter ticking once per CAMERA makes a
    loop that is keeping up perfectly report ~2 dropped sets on every
    iteration -- that shipped, and a rig comfortably ahead of its cameras
    logged a steady stream of SUSTAINED frame-processing overrun warnings
    while doing 3x the wakeups it needed.

    Asserted as an identity rather than a rate: the sink runs on the pump
    thread inside the cycle that just bumped, so consecutive published
    sets must show the generation advancing by exactly ONE. Per-frame
    counting would show three. (A rate assertion would also have to know
    about open_all()'s and close_all()'s own housekeeping bumps, which is
    noise around the thing being measured.)"""
    _mock_local_cameras(monkeypatch)
    box: list = []
    sink, calls, gens = _sink_recorder(box)
    hub = build_hub([0, 1, 2], None, frame_sink=sink)
    box.append(hub)
    hub.open_all()
    assert _wait_until(lambda: len(calls) >= 10)
    hub.close_all() # stops every thread, so the two records settle
    assert all(set(c) == {0, 1, 2} for c in calls), "a cycle published a partial set"
    steps = {b - a for a, b in zip(gens, gens[1:])}
    assert steps == {1}, (
        f"three cameras advanced the generation by {sorted(steps)} per frame "
        "set -- it must be exactly 1")


def test_a_stalled_stream_does_not_freeze_the_loop(monkeypatch):
    """A mixed rig whose stream is dead must keep running at what its
    LIVE slots can do.

    This is why StreamSource's budget runs from the last frame rather than
    from the start of each wait: with a per-wait timeout, every cycle
    would pay it and three healthy cameras would be capped at 2Hz."""
    _mock_local_cameras(monkeypatch)
    monkeypatch.setattr(remote_capture.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("refused")))
    monkeypatch.setattr(remote_capture, "RECONNECT_DELAY_S", 0.05)
    hub = build_hub([0, 1, 2], [None, None, "http://dead/2"])
    hub.open_all()
    try:
        # Past the one cycle that legitimately waits out the budget.
        time.sleep(remote_capture.STREAM_FRAME_WAIT_S + 0.2)
        before = hub.frame_generation()
        time.sleep(0.4)
        advanced = hub.frame_generation() - before
        assert advanced >= 10, (
            f"the loop advanced {advanced} times in 0.4s with one dead stream "
            "-- a dead slot is pacing the whole hub")
        # And the dead slot says why, rather than reporting a shrug.
        assert "refused" in (hub.status[2].last_error or "").lower()
        assert hub.status[2].last_read_ok is False
    finally:
        hub.close_all()


def test_a_live_swap_keeps_the_same_hub_object(monkeypatch):
    """THE POINT OF ALL OF THIS. The capture thread binds its hub once at
    thread creation and AppState holds its own reference, so handing out a
    new object on a source change leaves one of them stale -- which is the
    restart this design removes."""
    _mock_local_cameras(monkeypatch)
    _serve(monkeypatch, _a_few_frames())
    hub = build_hub([0, 1, 2], None)
    identity = id(hub)
    assert hub.slot_urls == [None, None, None]

    hub.reconfigure(None, ["http://rig:8420/api/cameras/0/stream.mjpg?full=1", None, None])
    assert id(hub) == identity, "the hub object must survive a source change"
    assert hub.slot_urls[0] is not None
    assert hub.slot_urls[1:] == [None, None]

    # ...and back again, which is the case the user actually described:
    # start on local cameras, swap to a feed, swap back.
    hub.reconfigure(None, [None, None, None])
    assert id(hub) == identity
    assert hub.slot_urls == [None, None, None]


def test_reconfigure_without_urls_keeps_the_routing_it_already_had():
    """The dashboard sends only `devices` when a device dropdown moves.
    Treating that as "make everything local" would silently drop a working
    feed because an unrelated slot changed."""
    hub = build_hub([0, 1, 2], [None, "http://a/1", None])
    hub.reconfigure([CameraConfig(device=d) for d in (4, 1, 2)])
    assert hub.slot_urls == [None, "http://a/1", None]
    assert [c.device for c in hub.configs] == [4, 1, 2]


def test_reconfigure_refuses_while_open_so_a_caller_cannot_half_apply(monkeypatch):
    _mock_local_cameras(monkeypatch)
    _serve(monkeypatch, _a_few_frames())
    hub = build_hub([0, 1, 2], None)
    hub.open_all()
    with pytest.raises(RuntimeError):
        hub.reconfigure(None, ["http://a/0", None, None])
    hub.close_all()
    hub.reconfigure(None, ["http://a/0", None, None]) # now fine
    hub.close_all()

    # ...and the same refusal for a hub whose open slots are all streams:
    # a reader thread still running IS the old assignment, exactly as an
    # open capture is.
    streams = build_hub(None, ["http://a/0", "http://a/1", "http://a/2"])
    streams.open_all()
    with pytest.raises(RuntimeError, match="stream readers"):
        streams.reconfigure(None, [None, None, None])
    streams.close_all()


def test_a_url_for_a_slot_that_does_not_exist_is_dropped_LOUDLY(caplog):
    """Silently trimming it would leave the dashboard showing a stream the
    process is not reading -- a refusal that says nothing (docs/DESIGN.md).
    Not an exception, because the slot count legitimately changes in the
    same request that changes the URLs."""
    with caplog.at_level(logging.WARNING, logger=local_capture.log.name):
        hub = build_hub([0, 1], None)
        hub.reconfigure(None, ["http://a/0", "http://a/1", "http://a/2"])
    assert hub.slot_urls == ["http://a/0", "http://a/1"]
    assert any("ignoring" in r.message.lower() or "ignoring" in str(r.args).lower()
               for r in caplog.records), "a dropped URL was dropped in silence"


def test_close_all_stops_the_readers_and_wakes_waiters(monkeypatch):
    _serve(monkeypatch, _a_few_frames(200))
    hub = build_hub(None, ["http://a/0", "http://a/1", "http://a/2"])
    hub.open_all()
    threads = [src._thread for src in hub._sources.values()]
    assert _wait_until(lambda: hub.frame_generation() > 0)
    woke = threading.Event()

    def waiter():
        hub.wait_for_new_frame(hub.frame_generation(), timeout=10.0)
        woke.set()

    threading.Thread(target=waiter, daemon=True).start()
    time.sleep(0.05)
    hub.close_all()
    assert woke.wait(2.0), "close_all() left a waiter parked on its timeout"
    assert not any(t.is_alive() for t in threads), "a reader thread outlived close_all()"
    hub.close_all() # idempotent
    assert hub.grab_all() == {}


def test_grab_never_touches_the_network(monkeypatch):
    """The capture loop calls grab() every tick. A blocking fetch there
    would stall the loop exactly as a blocking cap.read() would -- which is
    why the hub has a pump thread at all."""
    _serve(monkeypatch, _a_few_frames(200))
    hub = build_hub(None, ["http://a/0"])
    hub.open_all()
    try:
        assert _wait_until(lambda: hub.grab(0) is not None)
        # Any network call from here on is a failure.
        monkeypatch.setattr(remote_capture.urllib.request, "urlopen",
                            lambda *a, **k: pytest.fail("grab() hit the network"))
        t0 = time.perf_counter()
        for _ in range(500):
            hub.grab(0)
            hub.grab_all()
        assert (time.perf_counter() - t0) < 0.5
    finally:
        hub.close_all()


def test_the_hub_presents_the_surface_the_capture_loop_actually_uses():
    """Derived from every `hub.<member>` call site in the tree. A missing
    one is an AttributeError at runtime, on the rig, mid-throw.

    hasattr is not enough, and that cost a real crash: `frame_generation`
    shipped once as a @property while the capture loop calls
    `hub.frame_generation()`, and the first real consumer session died
    with "'int' object is not callable". The test that was supposed to
    catch it only checked hasattr, which a property satisfies."""
    hub = build_hub([0, 1, 2], [None, "http://a/1", None])
    methods = ("open_all", "close_all", "close", "grab", "grab_all",
               "status_report", "wait_for_new_frame", "frame_generation",
               "reconfigure", "set_frame_sink")
    for name in methods:
        assert callable(getattr(hub, name)), f"hub.{name} must be callable"
    assert isinstance(hub.frame_generation(), int)
    assert isinstance(hub.status, dict) and sorted(hub.status) == [0, 1, 2]
    assert len(hub.configs) == 3
    assert isinstance(hub.status_report(), str)
    assert hub.frame_sink_errors == 0
    assert hub.slot_urls == [None, "http://a/1", None]
    # The memory probe reads this BY NAME to attribute the frame
    # cache's share of process memory. A hub without it reports a
    # confident 0.00 MB -- the exact silent-zero this project keeps
    # finding.
    assert isinstance(hub._last_frames, dict)
