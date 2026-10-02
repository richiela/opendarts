"""Athena's tip continuation down the shaft (2026-09-28) -- see
`opendarts.engines.athena.tip_continuation`'s module docstring."""
from __future__ import annotations

import numpy as np
import pytest

from opendarts.engines.athena import engine as athena_engine
from opendarts.engines.athena.tip_continuation import MAX_CONTINUATION_PX, continue_tip

BG_GREY = 60
# The shaft runs straight down column X from the flight (top) to the tip;
# `axis_unit` points from the tip toward the flight, i.e. up.
X = 200
UP = (0.0, -1.0)


def _images(*stretches):
    """bg and frame (720x1280 BGR) with vertical stretches `(y0, y1, delta)`
    of a 3px-wide shaft added to the frame at column X."""
    bg = np.full((720, 1280, 3), BG_GREY, np.uint8)
    frame = bg.copy()
    for y0, y1, delta in stretches:
        frame[y0:y1, X - 1:X + 2] = BG_GREY + delta
    return bg, frame


def test_continues_past_a_gap_onto_a_faint_needle():
    """Bright upper dart ends at y=160; a 12px gap; then a faint needle
    (under Athena's own 31 cut, over the continuation's 20) to y=195."""
    bg, frame = _images((100, 160, 90), (172, 196, 45))
    (x, y), move = continue_tip(bg, frame, (X, 159.0), UP)
    assert x == pytest.approx(X)
    assert 190 <= y <= 197
    assert move == pytest.approx(y - 159.0)


def test_does_not_jump_a_long_gap_to_a_few_stray_pixels():
    """A 30px gap to a 3px speck: the stretch across the gap must be at
    least a quarter of the gap it jumps, so this is a speck, not shaft."""
    bg, frame = _images((100, 160, 90), (190, 193, 60))
    (x, y), move = continue_tip(bg, frame, (X, 159.0), UP)
    assert move == 0.0
    assert (x, y) == (X, 159.0)
    # the same gap to a 10px stretch of real shaft is crossed
    bg, frame = _images((100, 160, 90), (190, 200, 60))
    (_, y), move = continue_tip(bg, frame, (X, 159.0), UP)
    assert 195 <= y <= 200


def test_never_walks_further_than_the_cap():
    bg, frame = _images((100, 160, 90), (160, 300, 45))
    _, move = continue_tip(bg, frame, (X, 159.0), UP)
    assert 0 < move <= MAX_CONTINUATION_PX


def test_ignores_the_blur_halo_past_a_clean_end():
    """A dart that ends where the detected tip is: the lowered cut reaches
    a pixel or two into the blur, under `MIN_MOVE_PX`, so nothing moves."""
    bg, frame = _images((100, 161, 90))
    assert continue_tip(bg, frame, (X, 159.0), UP) == ((X, 159.0), 0.0)


def test_nothing_to_continue_onto_on_an_unchanged_frame():
    bg, frame = _images()
    assert continue_tip(bg, frame, (X, 159.0), UP) == ((X, 159.0), 0.0)


_CAND = {"crossing_px": (X, 159.0), "tip_px": (X, 159.0), "axis_unit": UP}


def _with_radii(monkeypatch, radius_before, radius_after):
    monkeypatch.setattr(athena_engine, "continue_tip", lambda *a: ((X, 190.0), 31.0))
    radii = {(X, 159.0): radius_before, (X, 190.0): radius_after}
    monkeypatch.setattr(athena_engine, "_board_radius_mm", lambda px, *a: radii[tuple(px)])


@pytest.mark.parametrize("lone_camera", [False, True])
@pytest.mark.parametrize("before, after", [(100.0, 110.0), (160.0, 175.0), (180.0, 190.0)])
def test_continued_on_the_board_or_staying_off_it(monkeypatch, lone_camera, before, after):
    _with_radii(monkeypatch, before, after)
    out = athena_engine._continue_candidate(dict(_CAND), (None, None), None, 0, lone_camera)
    assert out["crossing_px"] == out["tip_px"] == (X, 190.0)
    assert out["tip_continued_px"] == 31.0


@pytest.mark.parametrize("before", [175.0, None])
def test_a_lone_camera_is_not_continued_from_off_the_board_onto_it(monkeypatch, before):
    _with_radii(monkeypatch, before, 160.0)
    assert athena_engine._continue_candidate(dict(_CAND), (None, None), None, 0, True) == _CAND
    # with a second camera seeing the dart, the same read is continued
    out = athena_engine._continue_candidate(dict(_CAND), (None, None), None, 0, False)
    assert out["tip_continued_px"] == 31.0


class _Det:
    ok = True
    reason = "ok"
    tip_px = crossing_px = (X, 159.0)
    elongation_ratio = 5.0
    tip_confidence = 1.0
    diagnostics = {"candidates": [dict(_CAND)]}


@pytest.mark.parametrize("n_seen", [0, 1, 2, 3])
def test_only_the_one_camera_that_sees_the_dart_is_a_lone_camera(monkeypatch, n_seen):
    """Every pass continues; `lone_camera` is set for a camera exactly when
    it is the only one that sees the dart inside the board without
    continuation."""
    calls = []

    def select(candidates, calib, cam, roi_polygon, require_roi, images=None, lone_camera=False):
        calls.append((cam, require_roi, images is not None, lone_camera))
        seen = require_roi and images is None and cam < n_seen
        return (dict(_CAND) if seen else None), None, 1.0, 0

    monkeypatch.setattr(athena_engine, "detect_crossing", lambda *a, **k: _Det())
    monkeypatch.setattr(athena_engine, "project_board_roi_polygon", lambda calib: None)
    monkeypatch.setattr(athena_engine, "_select_best_candidate", select)
    monkeypatch.setattr(athena_engine, "ray_plane_intersect", lambda *a, **k: None)
    images = {cam: np.zeros((4, 4, 3), np.uint8) for cam in range(3)}
    athena_engine.AthenaEngine().score(images, images, {cam: object() for cam in range(3)})
    strict = {cam: lone for cam, require_roi, with_images, lone in calls if require_roi and with_images}
    assert strict == {cam: n_seen == 1 and cam == 0 for cam in range(3)}
    fallback = [with_images for _, require_roi, with_images, _ in calls if not require_roi]
    assert all(fallback)
