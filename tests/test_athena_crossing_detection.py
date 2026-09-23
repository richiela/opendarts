"""Unit tests for opendarts.engines.athena.crossing_detection using
SYNTHETIC diff images (a drawn tapered wedge standing in for a dart's
elongated, narrowing silhouette) -- deliberately not real captured
images (data/archive/clean/'s real corpus already covers real-image
accuracy end-to-end via dev/tests/test_engine_athena_accuracy.py; this
file instead pins the module's own CORE geometric logic -- "the narrow,
converging end is the tip" -- against a fully controlled, known-truth
synthetic shape, the same reasoning tests/test_triangulation.py and the
other tests/test_athena_*.py files use synthetic ground truth for
their own genuinely-testable-in-isolation piece).
"""
from __future__ import annotations

import numpy as np
import pytest

from opendarts.engines.athena.crossing_detection import detect_crossing


def _make_wedge_pair(tip_xy: tuple[int, int], base_xy: tuple[int, int], base_half_width: int = 20):
    """A background + frame pair where `frame` has one extra tapered
    wedge shape drawn on it (base_half_width wide at `base_xy`, a single
    point at `tip_xy`) -- a synthetic stand-in for a dart's silhouette:
    narrow/converging at the true tip end, wide at the flight end."""
    import cv2

    img_h, img_w = 480, 640
    bg = np.full((img_h, img_w, 3), 180, dtype=np.uint8)
    frame = bg.copy()

    bx, by = base_xy
    tx, ty = tip_xy
    # Perpendicular direction to the tip<->base axis, for the base's width.
    dx, dy = tx - bx, ty - by
    length = float(np.hypot(dx, dy))
    perp = np.array([-dy, dx], dtype=np.float64) / length

    p1 = (bx + perp[0] * base_half_width, by + perp[1] * base_half_width)
    p2 = (bx - perp[0] * base_half_width, by - perp[1] * base_half_width)
    pts = np.array([p1, p2, (tx, ty)], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (30, 30, 30))
    return bg, frame


@pytest.mark.parametrize(
    "tip_xy,base_xy",
    [
        ((320, 100), (320, 350)),  # tip at top, base at bottom
        ((320, 380), (320, 130)),  # tip at bottom, base at top
        ((150, 240), (480, 240)),  # tip at left, base at right
        ((500, 240), (170, 240)),  # tip at right, base at left
    ],
)
def test_detect_crossing_finds_the_narrow_end_as_the_tip(tip_xy, base_xy):
    """The synthetic wedge's known apex (`tip_xy`) must be recovered as
    `tip_px`, regardless of orientation -- direct proof of the module's
    core "narrower end = tip" logic (crossing_detection.py's own module
    docstring, step 5's equivalent) against a known-truth shape."""
    bg, frame = _make_wedge_pair(tip_xy, base_xy)
    result = detect_crossing(bg, frame)

    assert result.ok, result.reason
    assert result.tip_px is not None
    # Generous tolerance -- N_TIP_POINTS_AVERAGED means the reported tip
    # is a small mean near the apex, not the exact apex pixel.
    assert result.tip_px[0] == pytest.approx(tip_xy[0], abs=15)
    assert result.tip_px[1] == pytest.approx(tip_xy[1], abs=15)


def test_detect_crossing_axis_unit_points_from_tip_toward_base():
    """axis_unit is documented (CrossingDetectionResult's own docstring)
    to point FROM the tip TOWARD the flight/base end -- verify that's
    actually what comes out, not the reverse."""
    tip_xy, base_xy = (320, 100), (320, 350)
    bg, frame = _make_wedge_pair(tip_xy, base_xy)
    result = detect_crossing(bg, frame)

    assert result.ok
    assert result.axis_unit is not None
    # tip at y=100, base at y=350 -- "toward the base" means +Y.
    assert result.axis_unit[1] > 0.9


def test_detect_crossing_ok_false_on_identical_images():
    """No diff at all (bg == frame) -- must report a clean failure, not
    crash or hallucinate a tip."""
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    result = detect_crossing(img, img.copy())
    assert result.ok is False
    assert result.tip_px is None


def test_detect_crossing_ok_false_on_shape_mismatch():
    bg = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame = np.full((480, 480, 3), 180, dtype=np.uint8)
    result = detect_crossing(bg, frame)
    assert result.ok is False
    assert "shape mismatch" in result.reason
