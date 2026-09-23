"""Reading a camera's own JPEG: its compression and its shape."""
from __future__ import annotations

import cv2
import numpy as np

from opendarts.live import jpeg_info


def _jpeg(quality: int) -> bytes:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buf.tobytes()


def test_quality_is_recovered_from_the_tables():
    """OpenCV encodes with the IJG tables, so the estimate should land on
    the quality it was asked for."""
    for q in (45, 75, 85, 95):
        est = jpeg_info.estimated_quality(_jpeg(q))
        assert abs(est["luma"] - q) <= 1, (q, est)
        assert abs(est["chroma"] - q) <= 1, (q, est)


def test_frame_shapes():
    good = _jpeg(80)
    assert jpeg_info.frame_shape(good)["kind"] == "clean"

    padded = good + bytes(37)
    shape = jpeg_info.frame_shape(padded)
    assert shape["kind"] == "padded"
    assert shape["trailing"] == 37 and shape["trailing_all_zero"] is True

    junk = good + b"\x12\x34\x56"
    assert jpeg_info.frame_shape(junk)["trailing_all_zero"] is False

    truncated = good[:-500]
    assert jpeg_info.frame_shape(truncated)["kind"] == "no_eoi"
    assert jpeg_info.frame_shape(b"\x00\x01rest")["kind"] == "not_jpeg"


def test_trimming_restores_a_clean_frame():
    good = _jpeg(80)
    assert jpeg_info.trim_to_eoi(good + bytes(10)) == good
    assert jpeg_info.trim_to_eoi(good) == good
    truncated = good[:-500]
    assert jpeg_info.trim_to_eoi(truncated) == truncated
