"""Tests for the Windows virtual-camera frame publisher.

Runs on every platform. The Windows-only paths are exercised through
`available()` gating rather than skipped wholesale, because the thing most
likely to break is a caller assuming publishing happened when the platform
cannot support it at all.
"""
from __future__ import annotations

import platform
import struct

import numpy as np
import pytest

from opendarts.live import vcam_publish


def test_header_layout_matches_the_c_contract_sizes():
    """The C reader indexes pixels at a fixed offset, so a header that
    packs to any other size silently shifts every frame. The writer's own
    span is SMALLER than the header: the tail belongs to the reader."""
    assert vcam_publish.HEADER_SIZE == 64           # total, pixels start here
    assert vcam_publish._HEADER.size == 40          # writer-owned prefix
    assert vcam_publish.WRITER_BYTES == 40
    assert vcam_publish.READER_STATS_OFFSET == 40
    assert vcam_publish._READER_STATS.size == 16
    # writer prefix + reader block must fit inside the header
    assert vcam_publish.WRITER_BYTES + vcam_publish._READER_STATS.size <= 64


def test_header_field_order_matches_the_c_contract():
    """Field ORDER is the contract -- same size with two fields swapped
    would pack identically and read as garbage geometry."""
    packed = vcam_publish._HEADER.pack(
        vcam_publish.MAGIC, vcam_publish.VERSION, 1280, 720, 3840,
        vcam_publish.FORMAT_BGR24, 4, 9, 12345,
    )
    magic, version, w, h, stride, fmt, seq, idx, ts = struct.unpack("<8IQ", packed[:40])
    assert magic == 0x4344564F      # 'ODVC'
    assert (version, w, h, stride) == (2, 1280, 720, 3840)
    assert (fmt, seq, idx, ts) == (0, 4, 9, 12345)


def test_mapping_size_accounts_for_header_plus_bgr_pixels():
    p = vcam_publish.VirtualCameraPublisher(0, 1280, 720)
    assert p.size == 64 + 1280 * 720 * 3


def test_available_is_true_only_on_windows():
    assert vcam_publish.available() is (platform.system() == "Windows")


def test_publish_is_a_no_op_when_the_platform_cannot_support_it():
    """Must return False rather than raising. This sits downstream of the
    capture loop and a diagnostic feature must never affect scoring."""
    if vcam_publish.available():
        pytest.skip("this asserts the non-Windows path")
    p = vcam_publish.VirtualCameraPublisher(0, 64, 48)
    assert p.publish(np.zeros((48, 64, 3), dtype=np.uint8)) is False
    assert p.open() is False


def test_publish_rejects_a_frame_of_the_wrong_geometry(caplog):
    """Geometry is baked into the mapping size and the reader's media
    type, so a mismatched frame cannot be published. It must be refused
    rather than resized -- resizing would cost time on the capture path
    and hide a real configuration problem."""
    p = vcam_publish.VirtualCameraPublisher(0, 1280, 720)
    assert p.publish(np.zeros((480, 640, 3), dtype=np.uint8)) is False


def test_publish_rejects_none_without_raising():
    p = vcam_publish.VirtualCameraPublisher(0, 64, 48)
    assert p.publish(None) is False


def test_a_failed_publisher_stays_quiet_rather_than_logging_per_frame(caplog):
    """30fps x 3 cameras is 90 log lines a second if the failure is not
    sticky -- enough to bury whatever the real problem was."""
    import logging

    caplog.set_level(logging.WARNING, logger="opendarts.live.vcam_publish")
    p = vcam_publish.VirtualCameraPublisher(0, 1280, 720)
    bad = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(10):
        p.publish(bad)
    warnings = [r for r in caplog.records if "virtual camera" in r.getMessage()]
    assert len(warnings) <= 1, f"logged {len(warnings)} times for one repeated fault"


def test_set_publishes_the_hub_frame_dict_shape():
    """Takes the dict the capture loop already has, so the call site stays
    one line."""
    s = vcam_publish.VirtualCameraSet(3, 64, 48)
    frames = {i: np.zeros((48, 64, 3), dtype=np.uint8) for i in range(3)}
    landed = s.publish_all(frames)
    assert landed == (3 if vcam_publish.available() else 0)
    s.close()


def test_set_ignores_a_camera_index_it_has_no_slot_for():
    """A hub configured with more cameras than the publisher set must not
    IndexError on the capture path."""
    s = vcam_publish.VirtualCameraSet(2, 64, 48)
    s.publish_all({5: np.zeros((48, 64, 3), dtype=np.uint8)})
    s.close()


def test_close_is_safe_when_nothing_was_ever_opened():
    vcam_publish.VirtualCameraPublisher(0, 64, 48).close()
    vcam_publish.VirtualCameraSet(3, 64, 48).close()


def test_writer_never_touches_the_reader_stats_block():
    """The writer rewrites the header on every frame. If that span
    included the reader's counters it would zero them 30 times a second,
    destroying the one measurement that shows whether frames are being
    dropped on the way out."""
    assert vcam_publish.WRITER_BYTES <= vcam_publish.READER_STATS_OFFSET


def test_reader_stats_says_it_is_closed_rather_than_returning_none():
    """IT USED TO RETURN None, and the diagnostics row then had nothing to
    print but "unknown". A Windows rig showed exactly that on 2026-09-15 --
    capture running at 13,448 frames, publishing enabled, nothing leaving
    the process, and no reason visible anywhere. An absent answer reads as
    "no such thing"; a closed publisher needs to say it is closed."""
    p = vcam_publish.VirtualCameraPublisher(0, 64, 48)
    st = p.reader_stats()
    assert st is not None
    assert st["open"] is False
    assert st["last_error"], "a closed publisher must carry a reason"
    assert st["format"] == "BGR24"
    assert (st["width"], st["height"]) == (64, 48)


def test_a_failed_windows_publisher_retries(monkeypatch):
    """The sticky flag meant one failure killed publishing for the life of
    the process -- the identical bug the v4l2 backend had."""
    import time as _t

    p = vcam_publish.VirtualCameraPublisher(0, 64, 48)
    monkeypatch.setattr(vcam_publish, "available", lambda: False)
    assert p.open() is False
    first = p._retry_at
    assert first > 0, "a failure must schedule a retry, not latch"
    assert p.open() is False, "inside the backoff window it should not retry"
    p._retry_at = _t.monotonic() - 0.01
    assert p.open() is False
    assert p._retry_at > first, "the retry deadline must move forward"


def test_the_windows_reason_is_logged_once_not_per_frame(monkeypatch, caplog):
    import logging
    import time as _t

    p = vcam_publish.VirtualCameraPublisher(0, 64, 48)
    monkeypatch.setattr(vcam_publish, "available", lambda: False)
    with caplog.at_level(logging.WARNING, logger=vcam_publish.log.name):
        for _ in range(50):
            p._retry_at = _t.monotonic() - 0.01
            p.open()
    warn = [r for r in caplog.records if "not publishing" in r.message]
    assert len(warn) == 1, f"50 attempts logged {len(warn)} times"


def test_reader_stats_parses_the_block_the_filter_writes():
    """Decode a block shaped exactly as the C reader writes it."""
    import struct as _s

    raw = _s.pack("<4I", 100, 7, 2, 999999)
    read, missed, torn, tick = vcam_publish._READER_STATS.unpack(raw)
    assert (read, missed, torn, tick) == (100, 7, 2, 999999)


def test_stats_reports_a_slot_even_when_nothing_is_mapped():
    """The diagnostics surface must be able to say 'slot 1 has no
    consumer' rather than omitting the slot entirely."""
    s = vcam_publish.VirtualCameraSet(3, 64, 48)
    rows = s.stats()
    assert [r["slot"] for r in rows] == [0, 1, 2]
    s.close()


def _c_define(name: str) -> int:
    """A #define from the C side of the contract."""
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "tools/winvcam/shared_frame.h").read_text()
    m = re.search(rf"#define\s+{name}\s+(\w+)", src)
    assert m, f"{name} not defined in shared_frame.h"
    return int(m.group(1).rstrip("uU"), 0)


def test_version_and_formats_match_the_c_header():
    """A writer and filter that disagree on the version show the test
    pattern; disagreeing on a format number would show garbage."""
    assert vcam_publish.VERSION == _c_define("ODVCAM_VERSION")
    assert vcam_publish.FORMAT_BGR24 == _c_define("ODVCAM_FORMAT_BGR24")
    assert vcam_publish.FORMAT_MJPEG == _c_define("ODVCAM_FORMAT_MJPEG")


def _mapped(w=64, h=48):
    p = vcam_publish.VirtualCameraPublisher(0, w, h)
    p._mm = bytearray(p.size)       # stands in for the named mapping
    return p


def _header(p):
    return struct.unpack("<8IQ", bytes(p._mm[:40]))


def test_a_camera_jpeg_is_published_as_mjpeg():
    """The filter hands these bytes on as MJPG, the way the real camera
    would, so the consumer does its own decode."""
    p = _mapped()
    jpeg = b"\xff\xd8" + b"x" * 500 + b"\xff\xd9"
    assert p.publish(np.zeros((48, 64, 3), np.uint8), jpeg)
    magic, version, w, h, stride, fmt, seq, idx, _ = _header(p)
    assert (fmt, stride) == (vcam_publish.FORMAT_MJPEG, len(jpeg))
    assert seq % 2 == 0 and idx == 1
    assert bytes(p._mm[64:64 + len(jpeg)]) == jpeg
    st = p.reader_stats()
    assert st["format"] == "MJPEG" and st["camera_jpeg_frames"] == 1


def test_a_slot_without_a_jpeg_is_published_as_pixels():
    p = _mapped()
    frame = np.full((48, 64, 3), 9, np.uint8)
    assert p.publish(frame)
    *_, stride, fmt, _seq, _idx, _ts = _header(p)
    assert (fmt, stride) == (vcam_publish.FORMAT_BGR24, 64 * 3)
    assert bytes(p._mm[64:]) == frame.tobytes()
    assert p.reader_stats()["format"] == "BGR24"


def test_a_jpeg_too_big_for_the_mapping_falls_back_to_pixels():
    p = _mapped()
    frame = np.full((48, 64, 3), 9, np.uint8)
    assert p.publish(frame, b"\xff\xd8" + b"x" * (64 * 48 * 3))
    assert _header(p)[5] == vcam_publish.FORMAT_BGR24


def test_the_set_hands_each_slot_its_own_jpeg():
    s = vcam_publish.VirtualCameraSet(2, 64, 48)
    for p in s.publishers:
        p._mm = bytearray(p.size)
    frames = {0: np.zeros((48, 64, 3), np.uint8), 1: np.zeros((48, 64, 3), np.uint8)}
    assert s.publish_all(frames, jpegs={1: b"\xff\xd8ab\xff\xd9"}) == 2
    assert _header(s.publishers[0])[5] == vcam_publish.FORMAT_BGR24
    assert _header(s.publishers[1])[5] == vcam_publish.FORMAT_MJPEG
    for p in s.publishers:
        p._mm = None


def _c_string(name: str) -> str:
    import re
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "tools/winvcam/shared_frame.h").read_text()
    m = re.search(rf'#define\s+{name}\s+L"([^"]+)"', src)
    assert m, f"{name} not defined in shared_frame.h"
    return m.group(1).replace("\\\\", "\\").replace("%d", "{}")


def test_mapping_and_event_names_match_the_c_header():
    """The filter waits on the event to push each frame as it arrives; a
    name that differs silently drops it back to polling."""
    assert vcam_publish.NAME_FMT == _c_string("ODVCAM_NAME_FMT")
    assert vcam_publish.EVENT_FMT == _c_string("ODVCAM_EVENT_FMT")


def test_each_published_frame_sets_the_event(monkeypatch):
    fired = []
    monkeypatch.setattr(vcam_publish, "_set_event", fired.append)
    p = _mapped()
    p._event = "EVT"
    p.publish(np.zeros((48, 64, 3), np.uint8))
    p.publish(np.zeros((48, 64, 3), np.uint8), b"\xff\xd8x\xff\xd9")
    assert fired == ["EVT", "EVT"]
    p._event = None
