"""The pair floor (opendarts/lifecycle/state.py, ``Lifecycle._pair_dart_cams``):
two cameras each under ``dart_min_px`` but over ``pair_min_px`` commit as one
dart when they agree on it three ways -- the same_dart.py band, in both
directions, AND (``pair_shape_check``) each shows a dart-shaped (elongated)
change at full resolution, not just a blob that happens to sit on the band.

Synthetic cameras placed the way a low-camera rig's are (same positions as
``tests/test_same_dart.py``), but driven at FULL resolution: the pair rule
decodes real frames and runs Apollo's ``detect_tip`` on them
(``pair_shape_check``), so a small grey picture with no JPEG behind it (as
``test_same_dart.py`` uses) will not do here -- these are plain full BGR
arrays, shrunk by the lifecycle itself exactly as a real pump frame would
be.
"""
from __future__ import annotations

import cv2
import numpy as np

from opendarts.capture.board_disc import board_disc_mask
from opendarts.lifecycle.state import Action, Lifecycle, LifecycleConfig, Phase
from opendarts.pipeline import CameraCalibration

W, H, SCALE = 1280, 720, 4
K = np.array([[740.0, 0, W / 2], [0, 740.0, H / 2], [0, 0, 1]])
LOW_CAMERAS = {0: (348.0, 196.0, 205.0), 1: (-3.0, -413.0, 213.0), 2: (-343.0, 198.0, 206.0)}


def _cal(centre) -> CameraCalibration:
    C = np.array(centre, float)
    fwd = -C / np.linalg.norm(C)
    right = np.cross(fwd, [0.0, 0.0, 1.0]); right /= np.linalg.norm(right)
    R = np.array([right, np.cross(fwd, right), fwd])
    rvec, _ = cv2.Rodrigues(R)
    return CameraCalibration(camera_matrix=K.copy(), dist_coeffs=np.zeros((5, 1)), rvec=rvec,
                             tvec=(-R @ C).reshape(3, 1), pnp_result=None, landmark_spread_ok=True)


CALS = {cam: _cal(c) for cam, c in LOW_CAMERAS.items()}
MASKS = {cam: board_disc_mask(CALS[cam], W, H) for cam in LOW_CAMERAS}


def _px(cam, pts):
    """3-D board mm -> full-resolution pixel coordinates."""
    uv, _ = cv2.projectPoints(np.asarray(pts, float), CALS[cam].rvec, CALS[cam].tvec, K, np.zeros(5))
    return uv.reshape(-1, 2)


def _blank() -> np.ndarray:
    """A full-resolution, featureless BGR board -- same grey on every
    channel, so ``to_small_gray``'s BGR2GRAY reproduces it exactly."""
    rng = np.random.default_rng(3)
    g = np.clip(120 + rng.normal(0, 1.5, (H, W)), 0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


#: Tip position and the flight/shaft geometry tuned so that, at THIS tip,
#: cameras 0 and 2 each read a few dozen changed board pixels -- over
#: ``pair_min_px`` (40), under ``dart_min_px`` (60) -- while camera 1 (on
#: the opposite side of the board) reads none; see the module docstring's
#: cameras and the real-dart tuning this mirrors.
TIP = (0.0, 210.0)
FLIGHT_H = (90.0, 130.0)
FIN_MM = 3.5
SHAFT_PX = 1


def _draw_dart(img, cam, *, tip=TIP, shaft_px=SHAFT_PX, flight_h=FLIGHT_H, fin=FIN_MM):
    """A dart standing straight out of the face at ``tip`` (board mm),
    drawn at full resolution: a thin shaft plus two fins in an X, real 3-D
    geometry projected into ``cam`` -- the same shape every camera gets a
    consistent view of, just from its own angle."""
    x, y = tip
    a, b = _px(cam, [(x, y, 0.0), (x, y, flight_h[0])])
    cv2.line(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), (30, 30, 30), shaft_px)
    for ang in (45.0, 135.0):
        dx, dy = fin * np.cos(np.radians(ang)), fin * np.sin(np.radians(ang))
        poly = _px(cam, [(x - dx, y - dy, flight_h[0]), (x + dx, y + dy, flight_h[0]),
                         (x + dx, y + dy, flight_h[1]), (x - dx, y - dy, flight_h[1])])
        cv2.fillConvexPoly(img, np.round(poly).astype(np.int32), (30, 30, 30))
    return img


def _draw_blob(img, cam, point_mm, radius_px):
    """A compact, round change -- NOT dart-shaped -- centred on the full-
    resolution projection of the same 3-D point ``point_mm`` in ``cam``.
    Drawing the SAME 3-D point in two cameras makes it genuinely
    same_dart-consistent (it is one real point in space, seen twice) while
    staying round (``detect_tip``'s elongation ratio ~1) in both."""
    (u, v) = _px(cam, [point_mm])[0]
    cv2.circle(img, (int(round(u)), int(round(v))), int(radius_px), (30, 30, 30), -1)
    return img


def _ghost(img, cx=900, cy=250, half=12):
    """A compact change on cam 2's board, far from ``TIP``'s own rays --
    same per-camera pixel count as the real dart's, but not the same
    object (test_same_dart.py's ghost, at full resolution)."""
    cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (200, 200, 200), -1)
    return img


def _lifecycle(cfg: LifecycleConfig | None = None) -> Lifecycle:
    """A Lifecycle warmed up on a featureless board, ready to judge a
    would-be dart."""
    lc = Lifecycle(MASKS, cfg or LifecycleConfig(warmup_stable_frames=3), calibrations=CALS)
    blank = _blank()
    for _ in range(5):
        tick = lc.observe({0: blank.copy(), 1: blank.copy(), 2: blank.copy()})
    assert lc.phase is Phase.IDLE, f"did not warm up: {tick.phase}"
    return lc


def _drive(lc: Lifecycle, frames: dict[int, np.ndarray], n: int = 20) -> list:
    """Feed the SAME frames repeatedly (a settled arrival, not a moving
    one) until a commit or ``n`` ticks, whichever comes first."""
    ticks = []
    for _ in range(n):
        tick = lc.observe({cam: f.copy() for cam, f in frames.items()})
        ticks.append(tick)
        if tick.action is Action.COMMIT:
            break
    return ticks


def test_two_near_floor_cameras_agreeing_on_one_dart_commit():
    """Cameras 0 and 2 each see a change between ``pair_min_px`` and
    ``dart_min_px`` -- below the single-camera floor -- but the SAME 3-D
    dart, confirmed in both directions: the pair commits it."""
    cfg = LifecycleConfig(warmup_stable_frames=3)
    lc = _lifecycle(cfg)
    dart0 = _draw_dart(_blank(), 0)
    dart2 = _draw_dart(_blank(), 2)
    frames = {0: dart0, 1: _blank(), 2: dart2}

    ticks = _drive(lc, frames)
    sig = ticks[0].signals
    # preconditions: both pair cameras in the window, the third camera out
    assert cfg.pair_min_px <= sig[0].board_px < cfg.dart_min_px, sig[0].board_px
    assert cfg.pair_min_px <= sig[2].board_px < cfg.dart_min_px, sig[2].board_px
    assert sig[1].board_px < cfg.pair_min_px, sig[1].board_px

    committed = [t for t in ticks if t.action is Action.COMMIT]
    assert committed, "a real pair agreeing on one dart should commit"
    assert committed[0].dart_cams == [0, 2]
    assert committed[0].dart_count == 1


def test_two_near_floor_cameras_not_the_same_dart_does_not_commit():
    """Both cameras read a change in the same pixel-count window, but
    camera 2's is a ghost somewhere the dart in camera 0 cannot be: never
    committed, however long it is held."""
    cfg = LifecycleConfig(warmup_stable_frames=3)
    lc = _lifecycle(cfg)
    dart0 = _draw_dart(_blank(), 0)
    ghost2 = _ghost(_blank())
    frames = {0: dart0, 1: _blank(), 2: ghost2}

    ticks = _drive(lc, frames)
    sig = ticks[0].signals
    assert cfg.pair_min_px <= sig[0].board_px < cfg.dart_min_px, sig[0].board_px
    assert cfg.pair_min_px <= sig[2].board_px < cfg.dart_min_px, sig[2].board_px

    assert not any(t.action is Action.COMMIT for t in ticks)


def test_pair_floor_disabled_the_same_pair_does_not_commit():
    """``pair_min_px`` pushed out of reach reproduces the OLD behaviour on
    the exact pixels that committed in
    ``test_two_near_floor_cameras_agreeing_on_one_dart_commit`` -- proving
    that commit was the pair rule's doing, not the single-camera floor's."""
    cfg = LifecycleConfig(warmup_stable_frames=3, pair_min_px=10_000)
    lc = _lifecycle(cfg)
    dart0 = _draw_dart(_blank(), 0)
    dart2 = _draw_dart(_blank(), 2)
    frames = {0: dart0, 1: _blank(), 2: dart2}

    ticks = _drive(lc, frames)
    assert not any(t.action is Action.COMMIT for t in ticks)


def test_only_one_camera_near_the_floor_does_not_commit():
    """Only camera 0 sees the dart; camera 2 reads nothing -- one camera
    near the floor is still one camera, never a pair."""
    cfg = LifecycleConfig(warmup_stable_frames=3)
    lc = _lifecycle(cfg)
    dart0 = _draw_dart(_blank(), 0)
    frames = {0: dart0, 1: _blank(), 2: _blank()}

    ticks = _drive(lc, frames)
    sig = ticks[0].signals
    assert cfg.pair_min_px <= sig[0].board_px < cfg.dart_min_px, sig[0].board_px
    assert sig[2].board_px < cfg.pair_min_px, sig[2].board_px

    assert not any(t.action is Action.COMMIT for t in ticks)


def test_pair_requires_a_dart_shape_not_just_band_agreement():
    """A round, non-dart-shaped change -- literally the SAME 3-D point in
    both cameras, so same_dart's band agrees in both directions -- commits
    with ``pair_shape_check`` off (the band alone is enough) but not with
    it on (the default): the band says "one object, seen twice"; the
    shape check says whether that object looks like a dart."""
    point = (0.0, 210.0, 110.0)
    radius_px = 24

    lc_checked = _lifecycle(LifecycleConfig(warmup_stable_frames=3))
    blob0 = _draw_blob(_blank(), 0, point, radius_px)
    blob2 = _draw_blob(_blank(), 2, point, radius_px)
    frames = {0: blob0, 1: _blank(), 2: blob2}

    ticks = _drive(lc_checked, frames)
    sig = ticks[0].signals
    assert LifecycleConfig().pair_min_px <= sig[0].board_px < LifecycleConfig().dart_min_px, sig[0].board_px
    assert LifecycleConfig().pair_min_px <= sig[2].board_px < LifecycleConfig().dart_min_px, sig[2].board_px
    assert not any(t.action is Action.COMMIT for t in ticks), "a round blob is not a dart shape"

    lc_unchecked = _lifecycle(LifecycleConfig(warmup_stable_frames=3, pair_shape_check=False))
    ticks = _drive(lc_unchecked, frames)
    committed = [t for t in ticks if t.action is Action.COMMIT]
    assert committed, "with the shape check off, band agreement alone is enough"
    assert committed[0].dart_cams == [0, 2]
