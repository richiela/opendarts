"""Cross-view blob selection and the half-max tip (2026-09-28) -- see
`opendarts/engines/apollo/cross_view.py`'s module docstring and the
dated half-max block in `opendarts/engines/apollo/tip_detection.py`.

Everything here is synthetic: three pinhole cameras placed like the real
rig (below, right, left of the board, ~210mm out), a flat grey board and
a dart drawn by projecting its 3D segments, so each case controls exactly
one thing.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from opendarts.engines.apollo import cross_view
from opendarts.engines.apollo.cross_view import (
    camera_options,
    score_combination,
    select_blobs,
    wire_tail,
)
from opendarts.engines.apollo.engine import ApolloEngine
from opendarts.engines.apollo.tip_detection import (
    HALF_MAX_MIN_MOVE_PX,
    detect_tip,
    refine_tip_half_max,
)
from opendarts.pipeline import CameraCalibration

W, H = 1280, 720
BOARD_GREY = 170
DART_GREY = 40
CAMERA_CENTRES = {
    0: (-3.0, -413.0, 215.0),
    1: (351.0, 196.0, 208.0),
    2: (-347.0, 203.0, 213.0),
}
TIP_MM = (40.0, 60.0, 0.0)
LEAN = np.array([0.12, 0.08, 1.0]) / np.linalg.norm([0.12, 0.08, 1.0])


def _look_at(centre) -> CameraCalibration:
    c = np.asarray(centre, float)
    z = -c / np.linalg.norm(c)                      # looking at the bull
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z])
    if rot[1] @ np.array([0.0, 1.0, 0.0]) > 0:      # image y must point down the board
        rot[0], rot[1] = -rot[0], -rot[1]
    rvec, _ = cv2.Rodrigues(rot)
    tvec = -rot @ c
    return CameraCalibration(
        camera_matrix=np.array([[900.0, 0, W / 2], [0, 900.0, H / 2], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
    )


CALS = {k: _look_at(c) for k, c in CAMERA_CENTRES.items()}


def _px(xyz, cal) -> tuple[int, int]:
    p, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(1, 3), cal.rvec, cal.tvec, cal.camera_matrix, cal.dist_coeffs)
    return int(round(p[0, 0, 0])), int(round(p[0, 0, 1]))


def _draw_dart(img, cal, tip=TIP_MM, lean=LEAN, from_mm=0.0):
    tip = np.asarray(tip, float)
    for z0, z1, width in ((0, 25, 2), (25, 70, 8), (70, 110, 4), (110, 150, 18)):
        z0 = max(z0, from_mm)
        if z0 >= z1:
            continue
        cv2.line(img, _px(tip + z0 * lean, cal), _px(tip + z1 * lean, cal), (DART_GREY,) * 3, width)


def _scene(extra=None, skip=()):
    bgs, frames = {}, {}
    for k, cal in CALS.items():
        bg = np.full((H, W, 3), BOARD_GREY, np.uint8)
        fr = bg.copy()
        if k not in skip:
            _draw_dart(fr, cal)
        if extra and k in extra:
            extra[k](fr)
        bgs[k], frames[k] = bg, fr
    return bgs, frames


def _true_tip_px(k):
    return np.array(_px(TIP_MM, CALS[k]), float)


def test_synthetic_rig_sees_the_dart_in_every_camera():
    bgs, frames = _scene()
    for k in CALS:
        det = detect_tip(bgs[k], frames[k])
        assert det.ok
        assert np.linalg.norm(np.subtract(det.tip_px, _true_tip_px(k))) < 6


def test_consistent_blob_wins_over_larger_inconsistent_one():
    # cam0 also sees a big, elongated diff somewhere else on the board --
    # larger than the dart, so detect_tip() alone picks it.
    def decoy(img):
        cv2.line(img, (140, 200), (360, 530), (DART_GREY,) * 3, 45)
    bgs, frames = _scene(extra={0: decoy})
    alone = detect_tip(bgs[0], frames[0])
    assert np.linalg.norm(np.subtract(alone.tip_px, _true_tip_px(0))) > 100

    pick = select_blobs(bgs, frames, CALS)
    assert pick is not None
    chosen = pick.picks[0]
    assert chosen is not None and not chosen.primary
    assert np.linalg.norm(np.subtract(chosen.tip, _true_tip_px(0))) < 6

    result = ApolloEngine().score(bgs, frames, CALS)
    assert result.board_xy_mm is not None
    assert math.dist(result.board_xy_mm, TIP_MM[:2]) < 3.0
    assert result.diagnostics["cross_view"]["picks"]["0"][0] != 0


def _line_options():
    bgs, frames = _scene()
    return {k: next(o for o in camera_options(bgs[k], frames[k], CALS[k]) if o.primary and not o.point)
            for k in CALS}


def test_wire_flagged_tip_on_the_other_views_entry_is_accepted():
    opts = _line_options()
    opts[0].wire = True        # a needle that really lies along a wire
    assert score_combination(list(opts.items()), CALS) is not None


def test_wire_flagged_tip_off_the_other_views_entry_is_rejected():
    opts = _line_options()
    o = opts[0]
    u = np.subtract(o.tip, o.far)
    u /= np.linalg.norm(u)
    o.tip = tuple(np.add(o.tip, 9 * u))    # the diff ran on along a wire
    # Unflagged, 9px is within what the other gates tolerate...
    assert score_combination(list(opts.items()), CALS) is not None
    # ...but a wire-flagged tip must sit on the other views' board entry.
    o.wire = True
    assert score_combination(list(opts.items()), CALS) is None


def test_wire_tail_detects_a_diff_running_along_a_projected_wire():
    cal = CALS[0]
    th = math.radians(9 + 18 * 4)          # a radial wire
    a = _px((60 * math.cos(th), 60 * math.sin(th), 0), cal)
    b = _px((120 * math.cos(th), 120 * math.sin(th), 0), cal)
    bg = np.full((H, W, 3), BOARD_GREY, np.uint8)
    along = bg.copy()
    cv2.line(along, a, b, (DART_GREY,) * 3, 2)
    assert wire_tail(bg, along, a, cal)
    # Same point, but the changed pixels cross the wire instead of following it.
    across = bg.copy()
    d = np.subtract(b, a).astype(float)
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    cv2.line(across, tuple(int(v) for v in np.add(a, 20 * n)), tuple(int(v) for v in np.subtract(a, 20 * n)), (DART_GREY,) * 3, 2)
    assert not wire_tail(bg, across, a, cal)


def _needle_image(faint_len: int, faint_diff: int = 22):
    bg = np.full((120, 200, 3), BOARD_GREY, np.uint8)
    fr = bg.copy()
    fr[57:64, 20:101] = DART_GREY                                     # barrel, square end at x=100
    if faint_len:
        fr[59:61, 101:101 + faint_len] = BOARD_GREY - faint_diff      # faint needle
    return bg, fr


def test_half_max_tip_moves_onto_a_faint_needle():
    bg, fr = _needle_image(9)
    new = refine_tip_half_max(bg, fr, (100.0, 60.0), (20.0, 60.0))
    assert new[0] - 100.0 >= HALF_MAX_MIN_MOVE_PX
    assert new[0] <= 100 + 9 + 1.5


def test_half_max_tip_ignores_moves_under_the_minimum():
    bg, fr = _needle_image(0)
    assert refine_tip_half_max(bg, fr, (100.0, 60.0), (20.0, 60.0)) == (100.0, 60.0)
    bg, fr = _needle_image(2)
    assert refine_tip_half_max(bg, fr, (100.0, 60.0), (20.0, 60.0)) == (100.0, 60.0)


def test_half_max_needs_a_far_end():
    bg, fr = _needle_image(9)
    assert refine_tip_half_max(bg, fr, (100.0, 60.0), None) == (100.0, 60.0)


@pytest.fixture
def cam2_unreadable(monkeypatch):
    """cam2 offers no blob at all, whatever its image shows -- the case
    the two-camera fallback exists for."""
    real = cross_view.camera_options

    def options(bg, fr, calib, precomputed=None):
        if calib is CALS[2]:
            return []
        return real(bg, fr, calib, precomputed=precomputed)

    monkeypatch.setattr(cross_view, "camera_options", options)


def test_two_camera_pair_accepted_when_the_third_view_sees_its_axis(cam2_unreadable):
    bgs, frames = _scene()
    pick = select_blobs(bgs, frames, CALS)
    assert pick is not None
    assert pick.picks[2] is None
    assert pick.geometry["dropped_camera"] == 2
    assert pick.geometry["axis_support"] >= cross_view.MIN_AXIS_SUPPORT


def test_two_camera_pair_rejected_without_third_view_support(cam2_unreadable):
    bgs, frames = _scene(skip=(2,))          # cam2 sees nothing where the pair says the dart is
    assert select_blobs(bgs, frames, CALS) is None


def test_two_camera_pair_rejected_when_both_tips_sit_above_the_board(cam2_unreadable):
    # Both readable cameras only see the dart from 15mm up: their tips
    # meet the axis well above the board, which a pair cannot vouch for.
    bgs, frames = {}, {}
    for k, cal in CALS.items():
        bg = np.full((H, W, 3), BOARD_GREY, np.uint8)
        fr = bg.copy()
        _draw_dart(fr, cal, from_mm=15.0 if k != 2 else 0.0)
        bgs[k], frames[k] = bg, fr
    assert select_blobs(bgs, frames, CALS) is None
