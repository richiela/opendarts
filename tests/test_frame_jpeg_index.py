"""The capture loop's camera-JPEG pairing (capture_daemon._FrameJpegIndex).

The one property that matters: a frame is only ever given the bytes that
were read TOGETHER with it, at the tick it was fetched. Anything else --
bytes the pump published a cycle later -- must come back as "no bytes"
(the package then FFV1-encodes the array), never as the wrong bytes.
"""
from __future__ import annotations

import numpy as np

from opendarts.live.capture_daemon import _FrameJpegIndex


class _Hub:
    """grab_all() + grab_with_jpeg() over a cache the test advances."""

    def __init__(self):
        self.frames: dict[int, np.ndarray] = {}
        self.jpegs: dict[int, bytes] = {}

    def publish(self, tick: int, cams=(0, 1, 2)):
        for c in cams:
            self.frames[c] = np.full((4, 4, 3), tick * 10 + c, np.uint8)
            self.jpegs[c] = f"jpeg-t{tick}-c{c}".encode()

    def grab_all(self):
        return dict(self.frames)

    def grab_with_jpeg(self, i):
        return self.frames.get(i), self.jpegs.get(i)


def test_bytes_are_paired_with_the_frame_fetched_at_the_same_tick():
    hub, idx = _Hub(), _FrameJpegIndex()
    hub.publish(1)
    frames = hub.grab_all()
    idx.record(hub, frames)
    assert idx.lookup(frames) == {c: f"jpeg-t1-c{c}".encode() for c in range(3)}


def test_a_pump_cycle_between_fetch_and_pairing_yields_no_bytes_not_wrong_bytes():
    hub, idx = _Hub(), _FrameJpegIndex()
    hub.publish(1)
    frames = hub.grab_all()
    hub.publish(2, cams=(1,))       # the pump advanced cam1 in between
    idx.record(hub, frames)
    got = idx.lookup(frames)
    assert got == {0: b"jpeg-t1-c0", 2: b"jpeg-t1-c2"}
    assert 1 not in got


def test_a_hub_without_jpegs_records_nothing():
    class _NoJpegHub:
        def grab_all(self):
            return {0: np.zeros((2, 2, 3), np.uint8)}

    idx = _FrameJpegIndex()
    frames = _NoJpegHub().grab_all()
    idx.record(_NoJpegHub(), frames)
    assert idx.lookup(frames) == {}


def test_a_pinned_reference_keeps_its_bytes_after_leaving_the_tick_window():
    """The bg is adopted many ticks before the dart that scores against
    it; pinning is what lets that dart's package still stream-copy it."""
    hub, idx = _Hub(), _FrameJpegIndex(keep_ticks=2)
    hub.publish(1)
    reference = hub.grab_all()
    idx.record(hub, reference)
    idx.pin(reference)
    for tick in range(2, 10):
        hub.publish(tick)
        idx.record(hub, hub.grab_all())
        idx.pin(reference)            # the lifecycle still holds it
    assert idx.lookup(reference) == {c: f"jpeg-t1-c{c}".encode() for c in range(3)}

    idx.pin({})                       # reference replaced
    hub.publish(11)
    idx.record(hub, hub.grab_all())
    assert idx.lookup(reference) == {}


def test_an_equal_but_different_array_gets_no_bytes():
    """Identity, not equality: a copy of a frame is not provably the frame
    the bytes came with, so it gets nothing."""
    hub, idx = _Hub(), _FrameJpegIndex()
    hub.publish(1)
    frames = hub.grab_all()
    idx.record(hub, frames)
    assert idx.lookup({c: a.copy() for c, a in frames.items()}) == {}
