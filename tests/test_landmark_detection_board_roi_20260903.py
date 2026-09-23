"""Tests for the 2026-09-03 ROI-restricted two-pass adaptive colour
segmentation fix in `opendarts/calibration/landmark_detection.py`
(`_mask_for_detection()`/`_board_roi_from_mask()`) -- see that
function's own docstring for the full root-cause story (a whole-frame
Otsu saturation threshold dominated by the dark board surround, starving
a real camera's own dim ring paint just enough to break 8-connectivity
and fragment `_outer_ring_component_mask()`'s selection).

History worth stating plainly, not hidden: an EARLIER version of this
same task built and shipped a DIFFERENT fix (a reference-centre
correction inside `_trace_outer_boundary()`) against a "double+treble
ring merge" hypothesis -- both the hypothesis and that fix were
investigated further, found real-but-redundant/insufficient, and
REVERTED once this ROI-restriction fix (the actual root cause) was
found. Nothing from that earlier fix survives in this file or in
`landmark_detection.py`.

Two tiers, matching this project's own established
`test_adaptive_ring_color.py` convention:

1. Synthetic, fully isolated tests of `_board_roi_from_mask()`'s own
   pure geometry, and of `_mask_for_detection()`'s own control flow at
   every refusal point (pass 1 refuses, pass 1 produces no usable ROI,
   pass 2 refuses -- this last one flagged explicitly by this task's own
   review as "genuinely untested" on any locally-available real data,
   0/600 real frames exercised it; covered here synthetically instead).
2. Real-data regression tests against the calibration corpus -- they
   need real STORED CALIBRATION-EVENT RAW VIDEO, decoded via
   `opendarts.capture.calibration_package.load_calibration_package()`,
   which is not in the repo, so they live in
   `dev/tests/test_landmark_detection_board_roi_20260903.py` rather
   than here.
"""
from __future__ import annotations

import numpy as np
import pytest

from opendarts.calibration.landmark_detection import (
    BOARD_ROI_MAX_COMPONENTS,
    BOARD_ROI_MIN_COMPONENT_AREA_PX,
    BOARD_ROI_PAD_FRACTION,
    _board_roi_from_mask,
    _mask_for_detection,
)
from opendarts.calibration.oriented_landmarks import locate_pre_orientation_landmarks


# ---------------------------------------------------------------------
# Tier 1a: `_board_roi_from_mask()` pure geometry, fully synthetic
# ---------------------------------------------------------------------


def _blob_mask(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """A uint8 0/255 mask with a filled rectangle at each (x0, y0, x1, y1)
    box -- real ring components are far larger and irregular than a
    plain rectangle, but a filled rectangle's own bounding box is
    trivially known in advance, which is exactly what makes it a clean
    input for testing pure bounding-box arithmetic."""
    mask = np.zeros(shape, dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        mask[y0:y1, x0:x1] = 255
    return mask


def test_board_roi_is_the_union_bbox_of_qualifying_components_padded():
    shape = (600, 600)
    # Two real-sized blobs (well above the area floor), spatially
    # separate so they stay 2 distinct connected components even after
    # the internal morphological close, AND positioned so padding never
    # clamps against the image edge (kept separate from the dedicated
    # clamping test below, so this one isolates pure bbox+pad math).
    boxes = [(150, 150, 250, 250), (350, 350, 450, 450)]
    mask = _blob_mask(shape, boxes)
    roi = _board_roi_from_mask(mask)
    assert roi is not None
    x0, y0, x1, y1 = roi
    # Union bbox before padding: (150, 150, 450, 450) -> 300x300.
    pad = int(round(300 * BOARD_ROI_PAD_FRACTION))
    assert x0 == pytest.approx(150 - pad, abs=2)
    assert y0 == pytest.approx(150 - pad, abs=2)
    assert x1 == pytest.approx(450 + pad, abs=2)
    assert y1 == pytest.approx(450 + pad, abs=2)


def test_board_roi_clamps_to_image_bounds():
    shape = (200, 200)
    # A blob right at the edge -- padding would overflow the image
    # without clamping.
    boxes = [(0, 0, 190, 190)]
    mask = _blob_mask(shape, boxes)
    roi = _board_roi_from_mask(mask)
    assert roi is not None
    x0, y0, x1, y1 = roi
    assert 0 <= x0 < x1 <= 200
    assert 0 <= y0 < y1 <= 200


def test_board_roi_excludes_components_below_the_area_floor():
    """A tiny noise speck (well under `BOARD_ROI_MIN_COMPONENT_AREA_PX`)
    must not pull the ROI toward it -- only the real, larger component
    should contribute to the union box."""
    shape = (400, 400)
    real_component = (100, 100, 300, 300)  # 200x200 = 40000px, real-sized
    noise_speck = (5, 5, 10, 10)  # 25px, far under the floor
    assert 25 < BOARD_ROI_MIN_COMPONENT_AREA_PX
    mask = _blob_mask(shape, [real_component, noise_speck])
    roi = _board_roi_from_mask(mask)
    assert roi is not None
    x0, y0, x1, y1 = roi
    # The noise speck sits at (5,5) -- if it wrongly contributed to the
    # union box, x0/y0 would be near 0 even after padding a 200x200 box
    # (pad ~60px); a real, unpolluted ROI stays well clear of it.
    assert x0 > 20
    assert y0 > 20


def test_board_roi_keeps_only_the_top_n_components_by_area():
    """More than `BOARD_ROI_MAX_COMPONENTS` real-sized components must
    not all be unioned -- only the largest N contribute, per the real
    board having at most a handful of real colour blobs (double ring,
    treble ring, maybe a couple of large branding-text clusters), not
    an unbounded number."""
    shape = (1000, 1000)
    # BOARD_ROI_MAX_COMPONENTS "big" boxes (10000px each), well-separated
    # (150px gaps, far more than CLOSE_KERNEL_PX) so they stay genuinely
    # distinct components, all clearly bigger by area than the excluded
    # box below.
    big_boxes = [
        (50 + i * 250, 50, 150 + i * 250, 150)
        for i in range(BOARD_ROI_MAX_COMPONENTS)
    ]
    # A smaller-but-still-qualifying component, far away (bottom-left),
    # that should be excluded once the top-N-by-area cap is already full
    # of the 4 bigger boxes above.
    excluded_box = (50, 800, 90, 840)  # 1600px, qualifies on area alone
    mask = _blob_mask(shape, big_boxes + [excluded_box])
    roi = _board_roi_from_mask(mask)
    assert roi is not None
    x0, y0, x1, y1 = roi
    # If the far bottom-left box had been included, y1 would extend down
    # near 840 (+ padding) -- the top-band-only union stays well short
    # of that.
    assert y1 < 700, f"excluded_box appears to have leaked into the ROI: {roi}"


def test_board_roi_returns_none_on_an_empty_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    assert _board_roi_from_mask(mask) is None


def test_board_roi_returns_none_when_nothing_clears_the_area_floor():
    shape = (200, 200)
    mask = _blob_mask(shape, [(10, 10, 15, 15)])  # 25px, under the floor
    assert _board_roi_from_mask(mask) is None


# ---------------------------------------------------------------------
# Tier 1b: `_mask_for_detection()`'s control flow, synthetic (mocked
# `adaptive_ring_color_mask()`, no real image processing) -- every
# refusal branch, including the genuinely-untested pass-2-refusal path.
# ---------------------------------------------------------------------


class _FakeQuality:
    def __init__(self, reason: str = "fake"):
        self.reason = reason


class _FakeResult:
    def __init__(self, ok: bool, mask: np.ndarray | None):
        self.ok = ok
        self.mask = mask
        self.quality = _FakeQuality()


def _real_ish_mask(shape=(200, 200)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[50:150, 50:150] = 255
    return mask


def test_mask_for_detection_falls_back_when_pass1_refuses(monkeypatch):
    import opendarts.calibration.adaptive_ring_color as arc

    monkeypatch.setattr(arc, "adaptive_ring_color_mask", lambda img: _FakeResult(False, None))
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask, source = _mask_for_detection(image)
    assert source == "fixed_hsv_fallback"
    assert mask.shape == (200, 200)


def test_mask_for_detection_falls_back_when_pass1_produces_no_usable_roi(monkeypatch):
    import opendarts.calibration.adaptive_ring_color as arc

    # pass 1 "succeeds" but its own mask has nothing above the ROI area
    # floor -- `_board_roi_from_mask()` correctly returns None.
    empty_ish = np.zeros((200, 200), dtype=np.uint8)
    monkeypatch.setattr(arc, "adaptive_ring_color_mask", lambda img: _FakeResult(True, empty_ish))
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask, source = _mask_for_detection(image)
    assert source == "fixed_hsv_fallback"


def test_mask_for_detection_falls_back_when_pass2_refuses(monkeypatch):
    """The genuinely-untested-on-real-data path this task's own review
    flagged explicitly (0/600 real frames tested exercised this) --
    covered synthetically instead: pass 1 succeeds and produces a real
    ROI, but the SECOND (ROI-restricted) call refuses."""
    import opendarts.calibration.adaptive_ring_color as arc

    call_count = {"n": 0}

    def fake_adaptive(img):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeResult(True, _real_ish_mask(img.shape[:2]))
        return _FakeResult(False, None)

    monkeypatch.setattr(arc, "adaptive_ring_color_mask", fake_adaptive)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask, source = _mask_for_detection(image)
    assert call_count["n"] == 2, "pass 2 must genuinely be attempted, not skipped"
    assert source == "fixed_hsv_fallback"
    assert mask.shape == (200, 200)


def test_mask_for_detection_succeeds_end_to_end_both_passes_ok(monkeypatch):
    import opendarts.calibration.adaptive_ring_color as arc

    call_count = {"n": 0}
    shapes_seen = []

    def fake_adaptive(img):
        call_count["n"] += 1
        shapes_seen.append(img.shape[:2])
        if call_count["n"] == 1:
            return _FakeResult(True, _real_ish_mask(img.shape[:2]))
        # pass 2 called on a cropped (smaller) sub-image -- return a
        # mask matching THAT smaller shape.
        return _FakeResult(True, np.full(img.shape[:2], 255, dtype=np.uint8))

    monkeypatch.setattr(arc, "adaptive_ring_color_mask", fake_adaptive)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask, source = _mask_for_detection(image)
    assert call_count["n"] == 2
    assert source == "adaptive_roi"
    assert mask.shape == (200, 200)
    # pass 2's own input shape must genuinely be a CROP, not the same
    # full-frame shape pass 1 saw.
    assert shapes_seen[1] != shapes_seen[0]
    assert shapes_seen[1][0] < shapes_seen[0][0] or shapes_seen[1][1] < shapes_seen[0][1]
    # The full-size returned mask must have real foreground pixels
    # somewhere inside the ROI (pass 2's own all-255 fake mask, embedded
    # at the ROI offset) and stay background everywhere outside it.
    assert mask.sum() > 0
    assert mask.sum() < 255 * 200 * 200  # not the whole frame


def test_mask_for_detection_exception_still_falls_back(monkeypatch):
    """Pre-existing safety net, unchanged by this fix -- a raise from
    the adaptive path must never break calibration's first step."""
    import opendarts.calibration.adaptive_ring_color as arc

    def _raise(img):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(arc, "adaptive_ring_color_mask", _raise)
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask, source = _mask_for_detection(image)
    assert source == "fixed_hsv_fallback"
    assert mask.shape == (200, 200)
