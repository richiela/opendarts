"""The Linux virtual-camera set -- opendarts.live.v4l2_publish.

These run anywhere: a publisher touches no device until `open()`, which
answers False off Linux, so the set's plumbing (which thread publishes,
what happens after close) is testable on a dev machine. What cannot be
tested here is the kernel side -- format negotiation, what a consumer
actually sees -- which needs the rig.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from opendarts.live import v4l2_publish
from opendarts.live.v4l2_publish import V4L2LoopbackSet


class _StubPublisher:
    """Stands in for a real slot so the set's behaviour is observable
    without a /dev/video node."""

    def __init__(self, slot=0, block=None):
        self.slot = slot
        self.seen = []
        self.threads = []
        self.closed = False
        self._block = block

    def publish(self, frame, jpeg=None):
        self.threads.append(threading.current_thread())
        self.seen.append(frame)
        if self._block is not None:
            self._block.wait(5.0)
        return True

    def reader_stats(self):
        return {"published": len(self.seen)}

    def close(self):
        self.closed = True


def _frames(n=3):
    return {i: np.zeros((8, 8, 3), dtype=np.uint8) for i in range(n)}


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def test_publishing_happens_on_the_calling_thread():
    """Inline, on the pump thread, and measured rather than assumed.

    A worker thread was built for this on 2026-09-15 and removed the same
    day. Three five-minute samples each on the rig, idle, 3 cameras at
    1280x720 MJPEG: inline observe p50 5.8/5.7/5.7ms, worker 6.1/6.1/6.0ms;
    p95 8.1/7.8/7.5 against 8.3/8.1/7.9; both exactly 30.0 fps per camera
    with zero dropped frames. The worker was consistently the SLOWER of the
    two -- GIL contention plus a copy that was redundant anyway, because
    `local_capture._pump_once` reassigns `_last_frames[i]` to a fresh array
    rather than mutating it.

    Pinned so a future reader does not re-derive the "obviously this should
    be off-thread" intuition and rebuild it. The module docstring carries
    the full numbers.
    """
    pubs = [_StubPublisher(i) for i in range(3)]
    s = V4L2LoopbackSet.__new__(V4L2LoopbackSet)
    s.publishers = pubs
    s._closed = False
    landed = s.publish_all(_frames())
    assert landed == 3
    here = threading.current_thread()
    for pub in pubs:
        assert pub.threads == [here], "publishing must stay on the caller's thread"


def test_publish_all_returns_how_many_landed():
    pubs = [_StubPublisher(i) for i in range(3)]
    s = V4L2LoopbackSet.__new__(V4L2LoopbackSet)
    s.publishers = pubs
    s._closed = False
    assert s.publish_all(_frames()) == 3


def test_a_slow_device_does_stall_the_caller():
    """The honest cost of inline publishing, pinned so it is a known
    property rather than a surprise.

    A blocked device blocks the pump cycle. That is the trade accepted
    when the worker was removed: on the rig the encode is ~10ms of a 33ms
    cycle and the pump coordinator sits at 43% of a core, which has
    headroom -- but a device that hangs outright will hold the capture
    thread, and this test says so out loud.
    """
    gate = threading.Event()
    pubs = [_StubPublisher(0, block=gate)]
    s = V4L2LoopbackSet.__new__(V4L2LoopbackSet)
    s.publishers = pubs
    s._closed = False

    done = threading.Event()

    def call():
        s.publish_all({0: np.zeros((8, 8, 3), dtype=np.uint8)})
        done.set()

    t = threading.Thread(target=call, daemon=True)
    t.start()
    assert not done.wait(0.2), "a blocked device should block the caller"
    gate.set()
    assert done.wait(5.0), "the caller must finish once the device unblocks"


def test_close_releases_the_devices():
    pubs = [_StubPublisher(i) for i in range(3)]
    s = V4L2LoopbackSet.__new__(V4L2LoopbackSet)
    s.publishers = pubs
    s._closed = False
    s.close()
    assert all(p.closed for p in pubs)


def test_publishing_creates_no_threads():
    """There is no worker any more, so nothing here should start one --
    the leak this guards against is a worker reintroduced by accident."""
    before = threading.active_count()
    s = V4L2LoopbackSet.__new__(V4L2LoopbackSet)
    s.publishers = [_StubPublisher(i) for i in range(3)]
    s._closed = False
    for _ in range(5):
        s.publish_all(_frames())
    s.close()
    assert threading.active_count() == before


def test_worker_stats_report_that_publishing_is_inline():
    """The key survives the worker it described: /api/frame-health and the
    Windows backend both answer this, and a missing key reads as a broken
    diagnostic rather than an absent mechanism."""
    s = V4L2LoopbackSet.__new__(V4L2LoopbackSet)
    s.publishers = []
    s._closed = False
    assert s.worker_stats() == {"asynchronous": False}


def test_a_slot_the_set_has_no_publisher_for_is_ignored():
    """More frames than devices must not raise -- the hub's dict is sized
    by cameras, the set by loopback nodes, and they can disagree."""
    s = V4L2LoopbackSet.__new__(V4L2LoopbackSet)
    s.publishers = [_StubPublisher(0)]
    s._closed = False
    assert s.publish_all(_frames(3)) == 1


def test_available_is_false_off_linux():
    import platform

    if platform.system() != "Linux":
        assert v4l2_publish.available() is False


@pytest.mark.parametrize("fmt", ["BGR24", "MJPEG"])
def test_a_publisher_accepts_both_documented_formats(fmt):
    p = v4l2_publish.V4L2LoopbackPublisher(0, 64, 48, "/dev/video90", fmt)
    assert p.fmt == fmt


def test_an_unknown_format_is_refused_at_construction():
    with pytest.raises(ValueError):
        v4l2_publish.V4L2LoopbackPublisher(0, 64, 48, "/dev/video90", "YUYV")


# ---------------------------------------------------------------------------
# Choosing the published pixel format from config.
#
# BGR24 hands the consuming app the SAME array OpenDarts scored -- no second JPEG
# generation -- and costs 0.10ms per camera instead of 3.33ms, ~10ms of every
# 33ms pump cycle back on the thread that detects darts. Until this key it
# was a constructor argument no config could reach, so using it meant editing
# code.
# ---------------------------------------------------------------------------

def test_only_the_two_real_formats_are_accepted():
    """A typo must not reach VIDIOC_S_FMT. That call SUCCEEDS without
    honouring an unsupported request, so the failure would surface as a
    black camera in the consuming app with no error logged anywhere."""
    from opendarts.live.config import normalise_v4l2_format

    assert normalise_v4l2_format("BGR24") == "BGR24"
    assert normalise_v4l2_format("MJPEG") == "MJPEG"
    assert normalise_v4l2_format("bgr24") == "BGR24", "case must not matter"
    assert normalise_v4l2_format("  BGR24 ") == "BGR24"
    for bad in ("YUYV", "BGR3", "", "jpeg", 24, True, ["BGR24"]):
        assert normalise_v4l2_format(bad) is None, f"{bad!r} was accepted"
    assert normalise_v4l2_format(None) is None


def test_the_config_key_reaches_a_loaded_config(tmp_path):
    import json

    from opendarts.live.config import load_live_config

    p = tmp_path / "config.json"
    p.write_text(json.dumps({"v4l2_format": "bgr24"}))
    assert load_live_config(p).v4l2_format == "BGR24"


def test_an_invalid_format_falls_back_rather_than_failing(tmp_path):
    """A bad value must leave the publisher on its own default, not stop a
    rig from starting."""
    import json

    from opendarts.live.config import load_live_config

    p = tmp_path / "config.json"
    p.write_text(json.dumps({"v4l2_format": "YUYV"}))
    assert load_live_config(p).v4l2_format is None


def test_the_format_is_offered_only_to_a_backend_that_has_one():
    """CAPABILITY CHECK, NOT A PLATFORM CHECK (docs/DESIGN.md). The Linux
    set takes a pixel format; the Windows one has no such concept and
    always writes BGR24 into shared memory. Passing fmt= blindly would be
    a TypeError on Windows."""
    import inspect

    from opendarts.live import v4l2_publish, vcam_publish

    assert "fmt" in inspect.signature(v4l2_publish.V4L2LoopbackSet).parameters
    assert "fmt" not in inspect.signature(vcam_publish.VirtualCameraSet).parameters


def test_run_product_checks_the_signature_before_passing_fmt():
    from pathlib import Path

    from opendarts.live import run_product

    src = Path(run_product.__file__).read_text()
    block = src[src.index("n_slots = len(camera_configs)"):]
    block = block[:block.index("VirtualCameraSet(n_slots")]
    assert "inspect.signature" in block, "fmt must not be passed blindly"
    assert '"fmt" in params' in block


def test_not_publishing_says_why_rather_than_returning_in_silence():
    """docs/DESIGN.md: a refusal must say so in the log.

    Both gates used to `return None` silently, so "the oracle toggle is
    off" and "v4l2loopback is not loaded" produced identical evidence --
    virtual cameras that never appeared, a clean log, and a rig that looked
    healthy. That cost a real debugging session chasing a pixel format when
    the oracle was simply switched off.
    """
    from pathlib import Path

    from opendarts.live import run_product

    src = Path(run_product.__file__).read_text()
    block = src[src.index("if not vcam.publish.available():"):]
    block = block[:block.index("width, height = _virtual_camera_geometry")]
    # One explanation per gate, so the two causes stay distinguishable.
    assert block.count("log.warning") == 2, (
        "each reason for not publishing needs its own line -- a shared one "
        "cannot say which gate closed")
    assert "v4l2loopback" in block, "the unavailable case must name the likely cause"
    assert "publish_virtual_cameras" in block, "the toggle case must name the override"


# ---------------------------------------------------------------------------
# A publisher that gives up must retry, and must SAY it gave up.
#
# 2026-09-15: a publisher could not claim BGR24 because another process held
# the device, so it latched `_failed = True`. Stopping that process removed the CAUSE
# but not the STATE -- 5000+ frames were captured while every publish returned
# False in silence, and stats() reported a bare {"slot": N}, which reads as
# "no device" rather than "this one gave up".
# ---------------------------------------------------------------------------

def test_a_failed_publisher_retries_instead_of_giving_up_forever(monkeypatch):
    import time as _t

    from opendarts.live import v4l2_publish as V

    p = V.V4L2LoopbackPublisher(0, 64, 48, "/dev/video90", "MJPEG")
    monkeypatch.setattr(V, "available", lambda: False)
    assert p.open() is False
    first = p._retry_at
    assert first > 0, "a failure must schedule a retry, not latch"

    # Inside the backoff window: no retry yet.
    assert p.open() is False
    # Past it: tried again (still fails here, but the attempt happened).
    p._retry_at = _t.monotonic() - 0.01
    assert p.open() is False
    assert p._retry_at > first, "the retry deadline must move forward"


def test_the_reason_is_logged_once_per_failure_run_not_per_frame(monkeypatch, caplog):
    import logging
    import time as _t

    from opendarts.live import v4l2_publish as V

    p = V.V4L2LoopbackPublisher(0, 64, 48, "/dev/video90", "MJPEG")
    monkeypatch.setattr(V, "available", lambda: False)
    with caplog.at_level(logging.WARNING, logger=V.log.name):
        for _ in range(50):
            p._retry_at = _t.monotonic() - 0.01   # force every attempt through
            p.open()
    warn = [r for r in caplog.records if "not publishing" in r.message]
    assert len(warn) == 1, f"50 attempts logged {len(warn)} times, expected 1"


def test_a_publisher_that_gave_up_still_appears_in_stats(monkeypatch):
    """The whole diagnostic failure: an absent entry read as 'no device'
    when it meant 'this one failed and is retrying'."""
    from opendarts.live import v4l2_publish as V

    p = V.V4L2LoopbackPublisher(0, 64, 48, "/dev/video90", "MJPEG")
    monkeypatch.setattr(V, "available", lambda: False)
    p.open()
    st = p.reader_stats()
    assert st is not None, "a failed publisher vanished from the diagnostic"
    assert st["open"] is False
    assert st["retrying"] is True
    assert st["last_error"], "the failure must carry its reason"
    assert st["device"] == "/dev/video90"


