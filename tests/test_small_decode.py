"""DETECTION DECODES SMALL (config key detect_from_small_decode).

See opendarts/capture/lazy_frame.py. These pin:

  * the reduced decode is 320x180 grey for a 1280x720 JPEG, and a
    LazyFrame's full pixels are bit-identical to the old full decode
  * LazyFrames decodes only what is read, and the lifecycle reads nothing
    but small pictures until a commit -- whose scored frames (commit AND
    reference) come out as the full decode of their own JPEGs
  * a pixels-only slot, or a scale the hub did not pre-decode at, falls
    back to shrinking the full frame, exactly as before
  * the switch: off, the hub publishes full arrays and the loop fetches
    with grab_all(); on, LazyFrames through grab_frames()
  * the capture loop's JPEG index pairs bytes with lazily decoded frames
  * a JPEG virtual camera forwards a LazyFrame without decoding it
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from opendarts.capture.lazy_frame import (
    LazyFrame,
    LazyFrames,
    as_frames,
    decode_all,
    handles_of,
    prefetch,
    reduced_gray,
)
from opendarts.lifecycle.adapter import LifecycleTriggerAdapter
from opendarts.lifecycle.signals import DEFAULT_SIGNAL_CONFIG, to_small_gray
from opendarts.lifecycle.state import Action, Lifecycle
from opendarts.live import capture_daemon, local_capture
from opendarts.live.config import detect_from_small_decode_enabled
from opendarts.live.local_capture import CameraConfig, CameraHub


def _board_image(w: int = 1280, h: int = 720, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 90, np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(w, h) // 3, (40, 140, 200), -1)
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)
    return img


def _encode(img: np.ndarray, q: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    return buf.tobytes()


def _lazy(data: bytes) -> LazyFrame:
    return LazyFrame(data, reduced_gray(data, 4), 4)


def _full(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


# -- the frame itself ---------------------------------------------------------


def test_reduced_decode_of_1280x720_is_320x180_grey():
    small = reduced_gray(_encode(_board_image()), 4)
    assert small.shape == (180, 320) and small.dtype == np.uint8


def test_hub_scale_is_the_lifecycle_scale():
    """Otherwise every small picture the pump makes is ignored and the
    lifecycle shrinks the full frame again -- correct, but the saving gone."""
    assert local_capture.SMALL_DECODE_SCALE == DEFAULT_SIGNAL_CONFIG.scale


def test_full_pixels_are_bit_identical_to_the_old_decode_and_decoded_once():
    data = _encode(_board_image())
    lf = _lazy(data)
    assert not lf.decoded
    assert lf.shape == (720, 1280, 3) and not lf.decoded, "shape must come from the header"
    a = lf.pixels()
    assert np.array_equal(a, _full(data))
    assert lf.pixels() is a, "one decode, one array: the loop pairs arrays by identity"
    assert lf.is_pixels(a) and not lf.is_pixels(a.copy())


def test_the_small_picture_is_close_to_the_old_one():
    data = _encode(_board_image())
    old = to_small_gray(_full(data), 4)
    new = reduced_gray(data, 4)
    diff = np.abs(old.astype(int) - new.astype(int))
    assert diff.mean() < 1.0 and diff.max() < DEFAULT_SIGNAL_CONFIG.diff_threshold


def test_lazyframes_decode_only_what_is_read():
    frames = LazyFrames({0: _lazy(_encode(_board_image(seed=0))),
                         1: _lazy(_encode(_board_image(seed=1)))})
    assert sorted(frames) == [0, 1] and 1 in frames and len(frames) == 2
    assert not any(h.decoded for h in frames.handles().values())
    frames[0]
    assert frames.handle(0).decoded and not frames.handle(1).decoded
    assert frames.small_gray(1, 4) is frames.handle(1).small
    assert frames.small_gray(1, 2) is None, "another scale: caller shrinks the full frame"


def test_decode_all_gives_plain_dicts_and_leaves_arrays_alone():
    arrays = {0: _board_image(seed=3)}
    [out] = decode_all(arrays)
    assert out == arrays and out is not arrays and out[0] is arrays[0]
    assert as_frames(arrays) is arrays and handles_of(arrays) is arrays
    lazy = {0: _lazy(_encode(_board_image(seed=4)))}
    assert prefetch(lazy) == 1 and prefetch(lazy) == 0, "one decode in flight per frame"
    [out] = decode_all(LazyFrames(lazy))
    assert type(out) is dict and out[0] is lazy[0].pixels()


# -- the lifecycle --------------------------------------------------------------


def _board_mask(w: int = 1280, h: int = 720) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), min(w, h) // 3, 1, -1)
    return m.astype(bool)


def _with_dart(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    cv2.line(out, (600, 300), (700, 380), (250, 250, 250), 9)
    return out


def _run(frames_for, n_empty=25, n_dart=10):
    """Empty board, then a dart; returns (lifecycle, commit LiveStep)."""
    cams = (0, 1)
    empty = {c: _board_image(seed=10 + c) for c in cams}
    dart = {c: _with_dart(empty[c]) for c in cams}
    jpeg_empty = {c: _encode(empty[c]) for c in cams}
    jpeg_dart = {c: _encode(dart[c]) for c in cams}
    lc = Lifecycle({c: _board_mask() for c in cams})
    ad = LifecycleTriggerAdapter()
    fe, fd = frames_for(jpeg_empty), frames_for(jpeg_dart)
    commit = None
    for t in range(n_empty + n_dart):
        frames = fe if t < n_empty else fd
        tick = lc.observe(frames)
        step = ad.apply(tick, lc, frames, now=t / 30.0)
        if tick.action is Action.COMMIT:
            commit = step
            break
    return lc, commit, jpeg_empty, jpeg_dart, fe, fd


def test_lazy_lifecycle_decodes_nothing_until_a_commit_then_scores_the_full_decodes():
    lc, commit, jpeg_empty, jpeg_dart, fe, fd = _run(
        lambda jp: LazyFrames({c: _lazy(d) for c, d in jp.items()}))
    assert commit is not None, "the dart must commit"
    # scored frames: plain arrays, bit-identical to decoding their own JPEGs
    # the old way -- the commit frame AND the reference it is scored against
    for c in jpeg_dart:
        assert type(commit.trigger.last_frame) is dict
        assert np.array_equal(commit.trigger.last_frame[c], _full(jpeg_dart[c]))
        assert np.array_equal(commit.reference[c], _full(jpeg_empty[c]))
        assert commit.trigger.last_frame[c] is fd.handle(c).pixels()


def test_lazy_idle_ticks_decode_nothing():
    cams = (0, 1)
    lc = Lifecycle({c: _board_mask() for c in cams})
    ad = LifecycleTriggerAdapter()
    seen = []
    for t in range(40):
        frames = LazyFrames({c: _lazy(_encode(_board_image(seed=20 + c))) for c in cams})
        seen.append(frames)
        ad.apply(lc.observe(frames), lc, frames, now=t / 30.0)
    assert not any(h.decoded for f in seen for h in f.handles().values())


def test_switch_off_path_commits_the_same_frames_as_before():
    """Plain arrays in -> plain arrays out, the same objects as went in."""
    lc, commit, jpeg_empty, jpeg_dart, fe, fd = _run(
        lambda jp: {c: _full(d) for c, d in jp.items()})
    assert commit is not None
    for c in jpeg_dart:
        assert commit.trigger.last_frame[c] is fd[c]
        assert commit.reference[c] is fe[c]


def test_a_pixels_only_slot_is_shrunk_from_its_pixels():
    """A slot with no JPEG (stream/replay) keeps today's path inside a
    LazyFrames; the lifecycle's small picture for it is to_small_gray()."""
    arr = _board_image(seed=30)
    data = _encode(_board_image(seed=31))
    frames = LazyFrames({0: arr, 1: _lazy(data)})
    lc = Lifecycle({0: _board_mask(), 1: _board_mask()})
    lc.observe(frames)
    assert np.array_equal(lc.refs[0].small, to_small_gray(arr, 4))
    assert lc.refs[1].small is frames.handle(1).small
    assert lc.refs[0].full is arr and lc.refs[1].full is frames.handle(1)
    assert not frames.handle(1).decoded


# -- the hub and the switch -------------------------------------------------------

W, H = 64, 48


class _RawCapture:
    """V4L2-like: honours CONVERT_RGB=0 and returns the JPEG as a 1xN buffer."""

    def __init__(self, device, backend, payload):
        self.payload = payload
        self.convert = True
        self.released = False

    def isOpened(self):  # noqa: N802
        return not self.released

    def release(self):
        self.released = True

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_CONVERT_RGB:
            self.convert = bool(value)
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: W, cv2.CAP_PROP_FRAME_HEIGHT: H,
                cv2.CAP_PROP_FPS: 30.0}.get(prop, 0.0)

    def read(self):
        if self.convert:
            return True, np.full((H, W, 3), 128, np.uint8)
        return True, np.frombuffer(self.payload, np.uint8).reshape(1, -1).copy()


@pytest.fixture
def raw_hub(monkeypatch):
    data = _encode(_board_image(W, H, seed=40))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(cv2, "VideoCapture", lambda d, b: _RawCapture(d, b, data))
    hubs = []

    def make(small: bool, **kw):
        hub = CameraHub(configs=[CameraConfig(device=0, width=W, height=H)], **kw)
        hub.set_small_decode(small)
        hubs.append(hub)
        hub.open_all()
        assert hub.status[0].jpeg_passthrough is True
        return hub

    yield make, data
    for hub in hubs:
        hub.close_all()


def _cached(hub, i=0):
    return hub.grab_jpeg_lazy(i)[0]


def test_switch_off_hub_publishes_full_arrays_and_the_loop_uses_grab_all(raw_hub):
    make, data = raw_hub
    hub = make(False)
    hub._pump_once()
    assert isinstance(_cached(hub), np.ndarray)
    frames = capture_daemon.fetch_current_frames(None, hub=hub)
    assert type(frames) is dict and np.array_equal(frames[0], _full(data))


def test_switch_on_hub_publishes_lazy_frames_and_the_loop_uses_grab_frames(raw_hub):
    make, data = raw_hub
    hub = make(True)
    hub._pump_once()
    lf = _cached(hub)
    assert isinstance(lf, LazyFrame) and not lf.decoded
    assert lf.jpeg == data and lf.small.shape == (H // 4, W // 4)
    assert hub.status[0].actual_width == W and not lf.decoded
    frames = capture_daemon.fetch_current_frames(None, hub=hub)
    assert isinstance(frames, LazyFrames) and frames.handle(0) is lf
    assert frames.jpeg(0) == data and frames.generation(0) is not None
    # reading a value decodes that frame, once
    assert frames[0] is lf.pixels() and np.array_equal(frames[0], _full(data))
    # grab() and friends still hand out pixels (the pump thread may have
    # moved on to a newer frame of the same bytes by now)
    for got in (hub.grab(0), hub.grab_with_jpeg(0)[0], hub.grab_all()[0],
                hub.grab_paired(0)[0]):
        assert isinstance(got, np.ndarray) and np.array_equal(got, _full(data))


def test_a_sink_that_wants_pixels_gets_them_decoded_on_the_workers(raw_hub):
    make, data = raw_hub
    got = []
    hub = make(True, frame_sink=lambda frames: got.append(frames))
    hub._pump_once()
    assert isinstance(got[-1][0], np.ndarray)
    assert np.array_equal(got[-1][0], _full(data))
    assert _cached(hub).decoded


def test_a_sink_that_takes_lazy_frames_gets_them_undecoded(raw_hub):
    make, _ = raw_hub
    got = []

    def sink(frames, jpegs=None):
        got.append((frames, jpegs))

    sink.accepts_lazy_frames = True
    hub = make(True, frame_sink=sink)
    hub._pump_once()
    frames, jpegs = got[-1]
    assert isinstance(frames[0], LazyFrame) and not frames[0].decoded
    assert jpegs[0] == frames[0].jpeg


def test_v4l2_mjpeg_forwards_a_lazy_frame_without_decoding(monkeypatch, tmp_path):
    from opendarts.live import v4l2_publish

    assert v4l2_publish.V4L2LoopbackSet.publish_all.accepts_lazy_frames is True
    data = _encode(_board_image(W, H, seed=41))
    lf = _lazy(data)
    pub = v4l2_publish.V4L2LoopbackPublisher.__new__(v4l2_publish.V4L2LoopbackPublisher)
    written = []
    monkeypatch.setattr(v4l2_publish.os, "write", lambda fd, b: written.append(bytes(b)))
    pub.__dict__.update(dict(_fd=3, width=W, height=H, fmt="MJPEG", _failed=False,
                             passthrough_frames=0, _frame_index=0, _write_errors=0,
                             slot=0))
    assert pub.publish(lf, data) is True
    assert written == [data] and not lf.decoded


def test_config_switch_defaults_true(tmp_path):
    path = tmp_path / "config.json"
    assert detect_from_small_decode_enabled(path) is True
    path.write_text(json.dumps({"detect_from_small_decode": False}))
    assert detect_from_small_decode_enabled(path) is False
    path.write_text(json.dumps({"detect_from_small_decode": "nope"}))
    assert detect_from_small_decode_enabled(path) is True


# -- the loop's JPEG index ------------------------------------------------------


def test_jpeg_index_pairs_bytes_with_lazily_decoded_frames():
    data = {c: _encode(_board_image(seed=50 + c)) for c in (0, 1)}
    frames = LazyFrames({c: _lazy(d) for c, d in data.items()},
                        generations={0: 7, 1: 8})
    index = capture_daemon._FrameJpegIndex()
    index.record(None, frames)
    index.pin(frames)
    assert not any(h.decoded for h in frames.handles().values()), "recording decodes nothing"
    # the commit decodes; lookups by the decoded arrays find the bytes
    [scored] = decode_all(frames)
    assert index.lookup(scored) == data
    assert index.lookup_generations(scored) == {0: 7, 1: 8}
    # a later tick pushes this one out of the window: the pin still holds
    for _ in range(5):
        index.record(None, LazyFrames({0: _lazy(data[0])}))
    assert index.lookup(scored) == data
    assert index.lookup({0: scored[0].copy()}) == {}, "identity, not equality"


# -- the board photo ------------------------------------------------------------


def test_board_photo_gets_full_frames_decoded_on_its_own_thread(monkeypatch):
    """The takeout photo renders from the lifecycle's reference, which is
    lazy: submit() must not decode on the capture loop, and the render must
    get the full-resolution decode."""
    import threading

    from opendarts.live import board_photo

    data = {c: _encode(_board_image(seed=60 + c)) for c in (0, 1)}
    frames = LazyFrames({c: _lazy(d) for c, d in data.items()})
    release, done, seen = threading.Event(), threading.Event(), {}

    def fake_render(self, fr, calibrations, *, skip_if_unchanged=False):
        release.wait(5)
        seen.update({c: fr[c] for c in fr})
        done.set()
        return None

    monkeypatch.setattr(board_photo.BoardPhotoRenderer, "render_jpeg", fake_render)
    renderer = board_photo.BoardPhotoRenderer()
    renderer.submit(frames, {0: object(), 1: object()}, lambda jpeg: None)
    assert not any(h.decoded for h in frames.handles().values()), "decoded on the caller"
    release.set()
    assert done.wait(5)
    for c, d in data.items():
        assert seen[c].shape == (720, 1280, 3) and np.array_equal(seen[c], _full(d))
