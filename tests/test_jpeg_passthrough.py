"""JPEG passthrough: a camera's own JPEG kept next to its pixels.

See opendarts/live/local_capture.py's JPEG PASSTHROUGH section. These
pin the contract with fake cameras:

  * a V4L2 slot reads with conversion off, repairs and decodes each
    frame, and keeps the repaired bytes
  * a slot whose backend cannot deliver a usable JPEG goes back to
    decoded reads, and says so on its status
  * on Windows, the Media Foundation capture is tried before CAP_MSMF and
    abandoned for it when it gives nothing usable
  * consumers -- the frame sink, the v4l2 publisher, the frame ring and
    its dumps, the stream reader -- get the bytes for exactly the frame
    they were handed
"""
from __future__ import annotations

import io
import json
import os
import time

import cv2
import numpy as np
import pytest

from opendarts.capture.frame_dump import FRAMES_FILENAME, MANIFEST_FILENAME, write_dump
from opendarts.capture.frame_ring import FrameRing
from opendarts.live import jpeg_info, local_capture, remote_capture, v4l2_publish
from opendarts.live.local_capture import CameraConfig, CameraHub

W, H = 64, 48


def _jpeg(value: int = 128, w: int = W, h: int = H) -> bytes:
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = value
    img[:, w // 2:] = 255 - value
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _as_buffer(data: bytes) -> np.ndarray:
    """What OpenCV's V4L2 backend returns with CONVERT_RGB off: 1xN."""
    return np.frombuffer(data, np.uint8).reshape(1, -1).copy()


class RawCapture:
    """A cv2.VideoCapture that honours CONVERT_RGB like V4L2 does.

    `payloads` is what each read returns while conversion is off (cycled);
    with conversion on it returns decoded pixels.
    """

    def __init__(self, device, backend, payloads, *, honours_convert=True):
        self.device = device
        self.backend = backend
        self.payloads = list(payloads)
        self.honours_convert = honours_convert
        self.convert = True
        self.set_calls: list = []
        self.reads = 0
        self.released = False

    def isOpened(self):  # noqa: N802
        return not self.released

    def release(self):
        self.released = True

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        if prop == cv2.CAP_PROP_CONVERT_RGB and self.honours_convert:
            self.convert = bool(value)
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: W, cv2.CAP_PROP_FRAME_HEIGHT: H,
                cv2.CAP_PROP_FPS: 30.0}.get(prop, 0.0)

    def read(self):
        payload = self.payloads[self.reads % len(self.payloads)]
        self.reads += 1
        if self.convert:
            return True, np.full((H, W, 3), 128, np.uint8)
        if isinstance(payload, bytes):
            return True, _as_buffer(payload)
        return True, payload


def _linux_hub(monkeypatch, payloads, **kwargs):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    made: list[RawCapture] = []

    def factory(device, backend):
        cap = RawCapture(device, backend, payloads, **kwargs)
        made.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = CameraHub(configs=[CameraConfig(device=0, width=W, height=H)])
    return hub, made


def _wait(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# -- jpeg_info ---------------------------------------------------------


def test_repaired_appends_a_missing_end_marker_and_trims_padding():
    good = _jpeg()
    assert jpeg_info.repaired(good) == good
    assert jpeg_info.repaired(good[:-2]) == good
    assert jpeg_info.repaired(good + b"\x00" * 40) == good
    assert jpeg_info.repaired(b"not a jpeg") is None


def test_a_frame_with_no_end_marker_decodes_once_repaired():
    """The Scolia case: whole scan, no EOI."""
    fixed = jpeg_info.repaired(_jpeg()[:-2])
    img = cv2.imdecode(np.frombuffer(fixed, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (H, W, 3)


def test_dimensions_reads_the_frame_header():
    assert jpeg_info.dimensions(_jpeg(w=80, h=30)) == (30, 80, 3)
    assert jpeg_info.dimensions(b"\xff\xd8\xff\xda\x00\x02") is None


# -- the hub: Linux ----------------------------------------------------


def test_v4l2_slot_keeps_the_camera_jpeg(monkeypatch):
    jpg = _jpeg()
    hub, made = _linux_hub(monkeypatch, [jpg])
    try:
        assert hub.open_all() == [True]
        assert hub.status[0].jpeg_passthrough is True
        assert (cv2.CAP_PROP_CONVERT_RGB, 0) in made[0].set_calls
        frame, got = hub.grab_with_jpeg(0)
        assert got == jpg
        expected = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(frame, expected)
        assert _wait(lambda: hub.status[0].frame_count > 3)
        frame, got = hub.grab_with_jpeg(0)
        assert got == jpg and frame.shape == (H, W, 3)
    finally:
        hub.close_all()


def test_frames_missing_their_end_marker_are_kept_repaired(monkeypatch):
    jpg = _jpeg()
    hub, _ = _linux_hub(monkeypatch, [jpg[:-2]])
    try:
        hub.open_all()
        assert hub.status[0].jpeg_passthrough is True
        assert hub.grab_with_jpeg(0)[1] == jpg
    finally:
        hub.close_all()


def test_an_undecodable_frame_is_dropped_and_counted(monkeypatch):
    good = _jpeg()
    bad = good[:200]          # header and a sliver of scan: will not decode
    assert cv2.imdecode(np.frombuffer(bad + b"\xff\xd9", np.uint8), 1) is None
    hub, _ = _linux_hub(monkeypatch, [good, bad])
    try:
        hub.open_all()
        assert _wait(lambda: hub.status[0].jpeg_rejected >= 2)
        # Whatever is cached is a good frame with its own bytes.
        frame, got = hub.grab_with_jpeg(0)
        assert got == good and frame is not None
    finally:
        hub.close_all()


def test_a_backend_that_ignores_convert_off_is_not_passthrough(monkeypatch):
    """Pixels came back: use them, and do not read again."""
    hub, made = _linux_hub(monkeypatch, [_jpeg()], honours_convert=False)
    try:
        hub.open_all()
        assert hub.status[0].jpeg_passthrough is False
        # Not the camera's bytes -- but no longer NO bytes either: a local
        # camera without passthrough gets SYNTHETIC JPEG (ours, decoded back
        # to the pixels we publish), reported separately from passthrough.
        frame, data = hub.grab_with_jpeg(0)
        assert data is not None and hub.status[0].jpeg_synthetic is True
        assert np.array_equal(frame, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
        assert made[0].set_calls[-1] == (cv2.CAP_PROP_CONVERT_RGB, 1)
    finally:
        hub.close_all()


def test_a_non_jpeg_camera_goes_back_to_decoded_reads(monkeypatch):
    yuyv = np.zeros((1, W * H * 2), np.uint8)
    hub, made = _linux_hub(monkeypatch, [yuyv])
    try:
        hub.open_all()
        st = hub.status[0]
        assert st.jpeg_passthrough is False
        assert made[0].convert is True
        assert hub.grab(0).shape == (H, W, 3)
        assert _wait(lambda: st.frame_count > 3)
        assert st.last_read_ok
    finally:
        hub.close_all()


def test_the_environment_switch_turns_passthrough_off(monkeypatch):
    monkeypatch.setenv(local_capture.RAW_JPEG_ENV, "0")
    hub, made = _linux_hub(monkeypatch, [_jpeg()])
    try:
        hub.open_all()
        assert hub.status[0].jpeg_passthrough is False
        assert all(p != cv2.CAP_PROP_CONVERT_RGB for p, _ in made[0].set_calls)
    finally:
        hub.close_all()


def test_passthrough_is_on_the_status_json(monkeypatch):
    import dataclasses

    hub, _ = _linux_hub(monkeypatch, [_jpeg()])
    try:
        hub.open_all()
        d = dataclasses.asdict(hub.status[0])
        assert d["jpeg_passthrough"] is True and d["jpeg_rejected"] == 0
    finally:
        hub.close_all()


# -- the hub: Windows --------------------------------------------------


class FakeMf(RawCapture):
    """MfJpegCapture's surface: always JPEG, set() records geometry."""

    def __init__(self, device, payloads):
        super().__init__(device, local_capture.MF_JPEG_BACKEND, payloads)
        self.convert = False

    def set(self, prop, value):
        self.set_calls.append((prop, value))
        return prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                        cv2.CAP_PROP_FPS)


def _windows(monkeypatch, mf_payloads, cfg=None, dshow_index=7):
    """A Windows hub. `dshow_index` is where the path match puts the
    camera in DirectShow's list (None: no match)."""
    monkeypatch.setattr("platform.system", lambda: "Windows")
    monkeypatch.setattr(local_capture, "_dshow_index_for", lambda d: dshow_index)
    mf_made: list = []
    cv_opened: list = []

    def mf(device):
        cap = FakeMf(device, mf_payloads)
        mf_made.append(cap)
        return cap

    def factory(device, backend):
        cv_opened.append((backend, device))
        return RawCapture(device, backend, [_jpeg()], honours_convert=False)

    monkeypatch.setattr(local_capture, "_open_mf_jpeg", mf)
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = CameraHub(configs=[cfg or CameraConfig(device=1, width=W, height=H)])
    return hub, mf_made, cv_opened


def test_windows_reads_the_camera_jpeg_through_media_foundation(monkeypatch):
    jpg = _jpeg()
    hub, mf_made, cv_opened = _windows(monkeypatch, [jpg])
    try:
        assert hub.open_all() == [True]
        st = hub.status[0]
        assert st.backend_used == local_capture.MF_JPEG_BACKEND
        assert st.jpeg_passthrough is True
        assert cv_opened == []
        assert (cv2.CAP_PROP_FRAME_WIDTH, W) in mf_made[0].set_calls
        assert hub.grab_with_jpeg(0)[1] == jpg
    finally:
        hub.close_all()


def test_windows_falls_back_to_directshow_at_the_matched_index(monkeypatch):
    """DirectShow numbers devices its own way; the fallback opens the index
    whose device path matched, not the slot's Media Foundation number."""
    hub, mf_made, cv_opened = _windows(monkeypatch, [b"\x00" * 10], dshow_index=4)
    try:
        assert hub.open_all() == [True]
        assert mf_made[0].released
        assert cv_opened == [(cv2.CAP_DSHOW, 4)]
        assert hub.status[0].backend_used == "CAP_DSHOW"
        assert hub.status[0].jpeg_passthrough is False
    finally:
        hub.close_all()


def test_windows_never_opens_an_unmatched_directshow_number(monkeypatch):
    """No path match: fail, rather than open whatever DirectShow has at that
    number -- on a Windows rig that was one of our own virtual cameras."""
    hub, _, cv_opened = _windows(monkeypatch, [b"\x00" * 10], dshow_index=None)
    try:
        assert hub.open_all() == [False]
        assert cv_opened == []
        assert "no matching device" in hub.status[0].last_error
    finally:
        hub.close_all()


def test_windows_directshow_gets_mjpg_after_the_geometry(monkeypatch):
    """DSHOW applies the pixel format only after the size is set."""
    hub, _, _ = _windows(monkeypatch, [b"\x00" * 10])
    made = []

    def factory(device, backend):
        cap = RawCapture(device, backend, [_jpeg()], honours_convert=False)
        made.append(cap)
        return cap

    monkeypatch.setattr(cv2, "VideoCapture", factory)
    try:
        hub.open_all()
        calls = made[0].set_calls
        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        last_fourcc = max(i for i, (p, v) in enumerate(calls)
                          if p == cv2.CAP_PROP_FOURCC and v == mjpg)
        assert last_fourcc > [p for p, _ in calls].index(cv2.CAP_PROP_FRAME_WIDTH)
    finally:
        hub.close_all()


def test_windows_auto_resolution_skips_media_foundation(monkeypatch):
    hub, mf_made, cv_opened = _windows(
        monkeypatch, [_jpeg()], cfg=CameraConfig(device=1, width=None, height=None))
    monkeypatch.setattr(local_capture.camera_resolution,
                        "highest_supported_resolution", lambda cap: (W, H))
    try:
        hub.open_all()
        assert mf_made == []
        assert cv_opened[0][0] == cv2.CAP_DSHOW
    finally:
        hub.close_all()


def test_media_foundation_capture_is_inert_off_windows():
    from opendarts.live.win_mf_capture import MfJpegCapture

    cap = MfJpegCapture(0)
    assert cap.isOpened() is False
    assert cap.read() == (False, None)
    cap.release()


# -- consumers ---------------------------------------------------------


def test_a_sink_that_takes_jpegs_gets_them_for_the_same_slots(monkeypatch):
    jpg = _jpeg()
    hub, _ = _linux_hub(monkeypatch, [jpg])
    calls: list = []

    def sink(frames, jpegs=None):
        calls.append((dict(frames), dict(jpegs or {})))

    plain: list = []
    try:
        hub.set_frame_sink(sink)
        hub.open_all()
        assert _wait(lambda: len(calls) >= 3)
        frames, jpegs = calls[-1]
        assert set(jpegs) == set(frames) == {0}
        assert jpegs[0] == jpg
        hub.set_frame_sink(plain.append)
        assert _wait(lambda: len(plain) >= 3)
        assert isinstance(plain[-1], dict)
    finally:
        hub.close_all()


def _publisher(tmp_path, fmt="MJPEG"):
    pub = v4l2_publish.V4L2LoopbackPublisher(0, W, H, tmp_path / "video10", fmt)
    path = tmp_path / "out.bin"
    pub._fd = os.open(path, os.O_WRONLY | os.O_CREAT)  # noqa: SLF001
    return pub, path


def test_the_loopback_forwards_the_camera_jpeg_unchanged(tmp_path):
    jpg = _jpeg()
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    pub, path = _publisher(tmp_path)
    try:
        assert pub.publish(frame, jpg)
        assert pub.publish(frame)
    finally:
        pub.close()
    data = path.read_bytes()
    assert data.startswith(jpg)
    assert pub.passthrough_frames == 1 and pub.encoded_frames == 1


def test_the_loopback_says_once_when_the_geometry_is_wrong(tmp_path):
    """Used to raise AttributeError: the flag it checked was never set."""
    pub, _ = _publisher(tmp_path)
    try:
        small = np.zeros((10, 10, 3), np.uint8)
        assert pub.publish(small) is False
        assert pub.publish(small) is False
    finally:
        pub.close()


def test_the_set_hands_each_slot_its_own_bytes(tmp_path):
    jpg = _jpeg()
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    s = v4l2_publish.V4L2LoopbackSet.__new__(v4l2_publish.V4L2LoopbackSet)
    pub, path = _publisher(tmp_path)
    s.publishers = [pub]
    s._closed = False  # noqa: SLF001
    try:
        assert s.publish_all({0: frame}, jpegs={0: jpg}) == 1
    finally:
        pub.close()
    assert path.read_bytes() == jpg


def test_the_ring_keeps_only_the_jpeg_and_decodes_it_back(tmp_path):
    jpg = _jpeg()
    pixels = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    other = np.full((H, W, 3), 7, np.uint8)
    ring = FrameRing(60.0)
    for n in range(3):
        ring.append({0: pixels, 1: other}, wall_s=1000.0 + n,
                    monotonic_s=10.0 + n, generation=n, jpegs={0: jpg})
    sets = ring.snapshot().sets
    assert set(sets[0].pixels) == {1}
    assert sets[0].slots == [0, 1]
    assert sets[0].nbytes == len(jpg) + other.nbytes
    assert np.array_equal(sets[0].frames[0], pixels)
    assert ring.snapshot().n_frames == 6

    dest = tmp_path / "dump"
    write_dump(ring.snapshot(), dest, kind="test")
    manifest = json.loads((dest / MANIFEST_FILENAME).read_text())
    blob = (dest / FRAMES_FILENAME).read_bytes()
    entries = [f for s in manifest["sets"] for f in s["frames"]]
    slot0 = [f for f in entries if f["slot"] == 0]
    slot1 = [f for f in entries if f["slot"] == 1]
    # Stored as the ring holds each frame: the JPEG as sent, pixels as pixels.
    assert {f["encoding"] for f in slot0} == {"jpeg"}
    assert {f["encoding"] for f in slot1} == {"raw"}
    assert slot0[0]["shape"] == [H, W, 3] and slot0[0]["nbytes"] == len(jpg)
    first = slot0[0]
    assert blob[first["offset"]:first["offset"] + first["nbytes"]] == jpg
    # The same bytes object in every set is written once.
    assert [f["repeat_of_earlier_set"] for f in slot0] == [False, True, True]
    assert manifest["bytes_total"] == len(blob) == len(jpg) + other.nbytes

    from opendarts.capture.frame_dump import FrameDumpReader

    with FrameDumpReader(dest) as reader:
        assert np.array_equal(reader.read_frame(first), pixels)
        assert reader.read_jpeg(first) == jpg
        assert np.array_equal(reader.read_frame(slot1[0]), other)
        assert reader.read_jpeg(slot1[0]) is None


def test_a_version_1_dump_still_reads(tmp_path):
    """Pixel-only dumps written before JPEG storage have no `encoding`."""
    from opendarts.capture.frame_dump import FrameDumpReader

    other = np.full((H, W, 3), 7, np.uint8)
    ring = FrameRing(60.0)
    ring.append({1: other}, wall_s=1.0, monotonic_s=1.0, generation=0)
    dest = tmp_path / "dump"
    write_dump(ring.snapshot(), dest, kind="test")
    manifest = json.loads((dest / MANIFEST_FILENAME).read_text())
    manifest["schema"] = "frame-dump/v1"
    for s in manifest["sets"]:
        for f in s["frames"]:
            f.pop("encoding")
    (dest / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with FrameDumpReader(dest) as reader:
        (_, entry, arr), = list(reader.iter_frames())
        assert np.array_equal(arr, other)


def test_the_stream_reader_keeps_the_part_it_decoded():
    status = local_capture.CameraStatus(device=0)
    source = remote_capture.StreamSource(0, "http://x/0", status)
    jpg = _jpeg()
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    source._publish(frame, jpg)  # noqa: SLF001
    got_frame, got_jpg = source.read_pair()
    assert got_frame is frame and got_jpg == jpg
    source._publish(frame, jpg)  # noqa: SLF001
    assert source.read() is frame


def test_a_stream_slot_carries_its_bytes_through_the_hub(monkeypatch):
    jpg = _jpeg()
    boundary = "frame"
    body = b"".join(
        b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\n"
        + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n"
        for _ in range(50))

    class Resp(io.BytesIO):
        headers = {"Content-Type": f"multipart/x-mixed-replace; boundary={boundary}"}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(remote_capture.urllib.request, "urlopen",
                        lambda *a, **k: Resp(body))
    hub = CameraHub(urls=["http://rig/0"])
    try:
        hub.open_all()
        assert _wait(lambda: hub.grab_with_jpeg(0)[1] is not None)
        assert hub.grab_with_jpeg(0)[1] == jpg
    finally:
        hub.close_all()


def test_media_foundation_read_waits_past_empty_samples(monkeypatch):
    """ReadSample can succeed with no sample. A Windows rig's first read did, and the
    hub fell back to CAP_MSMF as if the camera had no JPEG to give."""
    from opendarts.live import win_mf_capture as mf

    cap = mf.MfJpegCapture.__new__(mf.MfJpegCapture)
    cap.device, cap.error = 0, None
    cap._released, cap._opened, cap._configured = False, True, True  # noqa: SLF001
    cap._com = object()  # noqa: SLF001
    cap._reader = mf.ctypes.c_void_p(1)  # noqa: SLF001
    jpg = _jpeg()
    results = iter([(0, 0, None), (0, 0, None), (0, 0, jpg)])
    monkeypatch.setattr(mf, "_ensure_com", lambda com: None)
    monkeypatch.setattr(mf, "_read_sample_bytes", lambda com, reader: next(results))
    ok, buf = cap.read()
    assert ok and buf.tobytes() == jpg

    monkeypatch.setattr(mf, "_read_sample_bytes", lambda com, reader: (0, 0, None))
    assert cap.read() == (False, None)
    assert "empty reads" in cap.error


def test_our_own_virtual_cameras_are_recognised_by_name():
    from opendarts.live import camera_names

    names = ["Surface Camera Front", "USB Camera", "OpenDarts Probe Cam 0",
             "OpenDarts Probe Cam 1"]
    assert camera_names.own_virtual_devices(names) == [2, 3]


def test_the_device_picker_leaves_out_our_own_virtual_cameras():
    from opendarts.live import server

    import inspect
    import pathlib

    src = inspect.getsource(server)
    assert '"own_virtual_devices": camera_names.own_virtual_devices(' in src
    # The picker's own filtering is dashboard JS, which lives in its own file.
    js = (pathlib.Path(server.__file__).parent / "dashboard" / "app.js").read_text()
    assert "if (own.has(d) && !live.includes(d)) continue;" in js


def test_device_paths_match_across_media_foundation_and_directshow():
    """Same camera, two APIs: different interface GUID and case."""
    from opendarts.live.win_mf_capture import device_key

    mf = r"\\?\usb#vid_046d&pid_085b&mi_00#7&1a2b3c&0&0000#{e5323777-f976-4f5b-9b55-b94699c46e44}\global"
    ds = r"\\?\USB#VID_046D&PID_085B&MI_00#7&1A2B3C&0&0000#{65e8773d-8f56-11d0-a3b9-00a0c911ce86}\global"
    assert device_key(mf) == device_key(ds)
    assert device_key(None) is None


def test_the_directshow_index_is_found_by_path_not_by_number(monkeypatch):
    from opendarts.live import camera_names, win_mf_capture

    monkeypatch.setattr(win_mf_capture, "list_devices", lambda: [
        ("USB Camera", r"\\?\usb#a#{mf}\global"),
        ("USB Camera", r"\\?\usb#b#{mf}\global"),
    ])
    monkeypatch.setattr(camera_names, "dshow_devices", lambda: [
        ("Surface Camera Front", r"\\?\display#s#{ds}\global"),
        ("USB Camera", r"\\?\usb#b#{ds}\global"),
        ("OpenDarts Probe Cam 0", None),
        ("USB Camera", r"\\?\usb#a#{ds}\global"),
    ])
    assert camera_names.dshow_index_for(0) == 3
    assert camera_names.dshow_index_for(1) == 1
    assert camera_names.dshow_index_for(2) is None


def test_windows_camera_names_follow_media_foundation_order(monkeypatch):
    from opendarts.live import camera_names, win_mf_capture

    monkeypatch.setattr(win_mf_capture, "list_devices",
                        lambda: [("USB Camera", "p1"), ("Surface Camera Front", "p2")])
    monkeypatch.setattr(camera_names, "dshow_devices",
                        lambda: [("Surface Camera Front", "p2"), ("USB Camera", "p1")])
    got = camera_names._enumerate_windows()
    assert got.names == ["USB Camera", "Surface Camera Front"]
    assert got.source == "media-foundation"
    monkeypatch.setattr(win_mf_capture, "list_devices", lambda: [])
    assert camera_names._enumerate_windows().source == "dshow-com"
