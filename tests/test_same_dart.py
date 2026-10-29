"""The single-camera rule's same-dart check (opendarts/lifecycle/same_dart.py).

Synthetic cameras placed the way a low-camera rig's are (~400 mm out,
~210 mm from the face): a dart is drawn as a real 3-D shaft and flight,
seen whole by one camera; another camera sees only its flight above the
rim and a 1 px shaft -- the case that went unscored. A ghost is a change
in the other camera somewhere the dart cannot be.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from opendarts.capture.board_disc import board_disc_mask
from opendarts.lifecycle import same_dart
from opendarts.lifecycle.signals import DEFAULT_SIGNAL_CONFIG, change_mask, to_small_mask
from opendarts.lifecycle.state import Action, Lifecycle, LifecycleConfig, Phase
from opendarts.pipeline import CameraCalibration

W, H, SCALE = 1280, 720, 4
K = np.array([[740.0, 0, W / 2], [0, 740.0, H / 2], [0, 0, 1]])
LOW_CAMERAS = {0: (348.0, 196.0, 205.0), 1: (-3.0, -413.0, 213.0), 2: (-343.0, 198.0, 206.0)}
KW = dict(dart_len_mm=150.0, lean_deg=35.0, max_rays=400, band_px=2)


def _cal(centre) -> CameraCalibration:
    C = np.array(centre, float)
    fwd = -C / np.linalg.norm(C)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    R = np.array([right, np.cross(fwd, right), fwd])
    rvec, _ = cv2.Rodrigues(R)
    return CameraCalibration(camera_matrix=K.copy(), dist_coeffs=np.zeros((5, 1)), rvec=rvec,
                             tvec=(-R @ C).reshape(3, 1), pnp_result=None, landmark_spread_ok=True)


CALS = {cam: _cal(c) for cam, c in LOW_CAMERAS.items()}


def _px(cam, pts):
    uv, _ = cv2.projectPoints(np.asarray(pts, float), CALS[cam].rvec, CALS[cam].tvec, K, np.zeros(5))
    return uv.reshape(-1, 2) / SCALE


def _blank():
    rng = np.random.default_rng(3)
    return np.clip(120 + rng.normal(0, 1.5, (H // SCALE, W // SCALE)), 0, 255).astype(np.uint8)


def _draw_dart(img, cam, tip=(40.0, 190.0), shaft_px=1, flight=True, shaft=True):
    """A dart standing straight out of the face at ``tip`` (board mm)."""
    x, y = tip
    if shaft:
        a, b = _px(cam, [(x, y, 0.0), (x, y, 110.0)])
        cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), 30, shaft_px)
    if flight:
        # Four fins in an X around the shaft, as a real flight: every camera
        # sees fins at an angle (one flat fin would be edge-on to someone).
        for ang in (45.0, 135.0):
            dx, dy = 18 * np.cos(np.radians(ang)), 18 * np.sin(np.radians(ang))
            poly = _px(cam, [(x - dx, y - dy, 110.0), (x + dx, y + dy, 110.0),
                             (x + dx, y + dy, 150.0), (x - dx, y - dy, 150.0)])
            cv2.fillConvexPoly(img, np.round(poly).astype(np.int32), 30)
    return img


def _support(others_cur, seen_cur):
    ref = _blank()
    others = {cam: (cur, ref) for cam, cur in others_cur.items()}
    return same_dart.support(CALS, 0, change_mask(seen_cur, ref), others, DEFAULT_SIGNAL_CONFIG, **KW)


def test_flight_off_the_board_and_a_thin_shaft_are_the_same_dart():
    seen = _draw_dart(_blank(), 0, shaft_px=3)
    other = _draw_dart(_blank(), 1, shaft_px=1)          # a 1 px shaft the 3x3 open erases
    board1 = to_small_mask(board_disc_mask(CALS[1], W, H), other.shape)
    assert not (change_mask(other, _blank()) & board1).any(), "the live detector sees nothing on camera 1's board"
    on, total = _support({1: other}, seen)[1]
    assert on >= LifecycleConfig().same_dart_min_px and on >= 0.5 * total


def test_a_change_where_the_dart_cannot_be_is_not_the_same_dart():
    seen = _draw_dart(_blank(), 0, shaft_px=3)
    ghost = _blank()
    cv2.rectangle(ghost, (5, 2), (40, 20), 200, -1)       # a glow in the top corner of camera 2
    on, total = _support({2: ghost}, seen)[2]
    assert total > 100 and on < 0.5 * total


def test_band_is_deterministic():
    seen = _draw_dart(_blank(), 0, shaft_px=3)
    other = _draw_dart(_blank(), 2)
    assert _support({2: other}, seen) == _support({2: other}, seen)


# --- the lifecycle uses it only for a one-camera dart --------------------

def _disc(shape=(360, 640)):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return (xx - 320) ** 2 + (yy - 180) ** 2 <= 120 ** 2


def _frame(level=120, dart=False):
    img = np.full((360, 640), level, np.float32)
    if dart:
        img[120:220, 318:334] = 20
    img += np.random.default_rng().normal(0, 2.0, img.shape).astype(np.float32)
    g = np.clip(img, 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _one_camera_dart(lc):
    for _ in range(6):
        lc.observe({0: _frame(), 1: _frame()})
    assert lc.phase is Phase.IDLE
    return [lc.observe({0: _frame(dart=True), 1: _frame()}) for _ in range(15)]


def test_without_calibrations_a_one_camera_dart_is_refused():
    ticks = _one_camera_dart(Lifecycle({0: _disc(), 1: _disc()}, LifecycleConfig(warmup_stable_frames=3)))
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert any(t.reason == "seen by one camera only" for t in ticks)


# (on band, total): enough and mostly on the band commits; mostly elsewhere
# does not; a speck does not, however much of it is on the band (the strip
# switching off confirmed itself with 11 of 11).
@pytest.mark.parametrize("found, commits", [({1: (150, 180)}, True), ({1: (150, 400)}, False), ({1: (11, 11)}, False)])
def test_a_one_camera_dart_commits_only_when_the_same_dart_is_seen(monkeypatch, found, commits):
    monkeypatch.setattr(same_dart, "support", lambda *a, **k: found)
    lc = Lifecycle({0: _disc(), 1: _disc()}, LifecycleConfig(warmup_stable_frames=3), calibrations={0: object(), 1: object()})
    ticks = _one_camera_dart(lc)
    committed = [t for t in ticks if t.action is Action.COMMIT]
    assert bool(committed) is commits
    if commits:
        assert committed[0].reason.startswith("same dart seen by camera 1")
