"""Talos's cross-view blob selection and on-board gates (2026-09-28) --
see `opendarts/engines/talos/cross_view.py`'s module docstring and the
dated blocks in `ray_fallback.py` / `plane_geometry.py`.

Same synthetic rig as tests/test_apollo_cross_view.py: three pinhole
cameras placed like the real rig, a flat grey board and a dart drawn by
projecting its 3D segments, so each case controls exactly one thing.
"""
from __future__ import annotations

import dataclasses
import math

import cv2
import numpy as np

from opendarts.engines.apollo import cross_view as apollo_cross_view
from opendarts.engines.apollo.cross_view import CrossViewPick
from opendarts.engines.talos.cross_view import (
    SWAP_MAX_TIP_HEIGHT_MM,
    camera_fit,
    select_talos_blobs,
    talos_blobs,
)
from opendarts.engines.talos.engine import score
from opendarts.engines.talos.plane_geometry import MAX_PLAUSIBLE_TILT_DEG, slide_point_along_dart_axis
from opendarts.engines.talos.ray_fallback import PAIR_MAX_BOARD_Z_MM, prefer_pair_on_board, score_dart
from opendarts.engines.talos.shaft_line import fit_shaft_line_px, shaft_components
from opendarts.pipeline import CameraCalibration, ScoreResult

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
    z = -c / np.linalg.norm(c)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot = np.stack([x, y, z])
    if rot[1] @ np.array([0.0, 1.0, 0.0]) > 0:
        rot[0], rot[1] = -rot[0], -rot[1]
    rvec, _ = cv2.Rodrigues(rot)
    return CameraCalibration(
        camera_matrix=np.array([[900.0, 0, W / 2], [0, 900.0, H / 2], [0, 0, 1]]),
        dist_coeffs=np.zeros(5),
        rvec=rvec.reshape(3, 1),
        tvec=(-rot @ c).reshape(3, 1),
    )


CALS = {k: _look_at(c) for k, c in CAMERA_CENTRES.items()}


def _pxf(xyz, cal) -> np.ndarray:
    p, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(1, 3), cal.rvec, cal.tvec, cal.camera_matrix, cal.dist_coeffs)
    return p.reshape(2)


def _px(xyz, cal) -> tuple[int, int]:
    p = _pxf(xyz, cal)
    return int(round(p[0])), int(round(p[1]))


def _draw_dart(img, cal, tip=TIP_MM, lean=LEAN):
    tip = np.asarray(tip, float)
    for z0, z1, width in ((0, 25, 2), (25, 70, 8), (70, 110, 4), (110, 150, 18)):
        cv2.line(img, _px(tip + z0 * lean, cal), _px(tip + z1 * lean, cal), (DART_GREY,) * 3, width)


def _scene(extra=None):
    bgs, frames = {}, {}
    for k, cal in CALS.items():
        bg = np.full((H, W, 3), BOARD_GREY, np.uint8)
        fr = bg.copy()
        _draw_dart(fr, cal)
        if extra and k in extra:
            extra[k](fr)
        bgs[k], frames[k] = bg, fr
    return bgs, frames


def _true_tip_px(k):
    return _pxf(TIP_MM, CALS[k])


def _decoy(img):
    # a big elongated diff elsewhere on the board, larger than the dart
    cv2.line(img, (140, 200), (360, 530), (DART_GREY,) * 3, 45)


def test_talos_blobs_primary_is_talos_own_pick():
    bgs, frames = _scene()
    for k in CALS:
        own = fit_shaft_line_px(bgs[k], frames[k])
        blobs = talos_blobs(bgs[k], frames[k])
        primary = [b for b in blobs if b["primary"]]
        assert len(primary) == 1
        assert np.allclose(primary[0]["tip"], own[2]["tip_px"])


def test_front_end_is_shared_not_recomputed():
    bgs, frames = _scene()
    first = shaft_components(bgs[0], frames[0])
    fit_shaft_line_px(bgs[0], frames[0])
    talos_blobs(bgs[0], frames[0])
    assert shaft_components(bgs[0], frames[0]) is first


def test_consistent_blob_wins_over_larger_inconsistent_one():
    bgs, frames = _scene(extra={0: _decoy})
    alone = fit_shaft_line_px(bgs[0], frames[0])
    assert np.linalg.norm(np.subtract(alone[2]["tip_px"], _true_tip_px(0))) > 100

    pick = select_talos_blobs(bgs, frames, CALS)
    assert pick is not None
    chosen = pick.picks[0]
    assert chosen is not None and not chosen.primary
    assert np.linalg.norm(np.subtract(chosen.tip, _true_tip_px(0))) < 6

    fitted = camera_fit(alone, CALS[0], pick, 0)
    assert np.linalg.norm(np.subtract(fitted[2]["tip_px"], _true_tip_px(0))) < 6
    assert fitted[2]["cross_view_pick"] == [chosen.rank, 0]


def test_engine_scores_the_dart_not_the_decoy():
    bgs, frames = _scene(extra={0: _decoy})
    r = score(bgs, frames, CALS)
    assert r.ok
    assert math.dist(r.board_xy_mm, TIP_MM[:2]) < 3
    assert r.diagnostics["cross_view"]["picks"]["0"] is not None


def test_selection_is_skipped_with_prior_darts():
    # darts 2/3 with prior-dart erasing run exactly as before
    bgs, frames = _scene(extra={0: _decoy})
    r = score(bgs, frames, CALS, prior_board_xy_mm=[(-120.0, -120.0)])
    assert "cross_view" not in r.diagnostics


def _pick_on_true_axis():
    bgs, frames = _scene()
    pick = select_talos_blobs(bgs, frames, CALS)
    assert pick is not None
    return bgs, frames, pick


def test_replacement_blob_up_the_shaft_is_rejected():
    bgs, frames, pick = _pick_on_true_axis()
    own = fit_shaft_line_px(bgs[1], frames[1])
    option = pick.picks[1]
    tip = np.asarray(TIP_MM, float)
    for height_mm, kept in ((2.0, True), (SWAP_MAX_TIP_HEIGHT_MM + 12.0, False)):
        replacement = dataclasses.replace(
            option, tip=tuple(_pxf(tip + height_mm * LEAN, CALS[1])), primary=False,
        )
        fitted = camera_fit(own, CALS[1], CrossViewPick({**pick.picks, 1: replacement}, pick.cost, pick.geometry), 1)
        assert (fitted is not None) == kept, height_mm


def test_left_out_camera_keeps_its_own_tip_only_when_it_is_the_tip():
    bgs, frames, pick = _pick_on_true_axis()
    own = fit_shaft_line_px(bgs[2], frames[2])
    dropped = CrossViewPick({**pick.picks, 2: None}, pick.cost, pick.geometry)
    kept = camera_fit(own, CALS[2], dropped, 2)
    assert kept is not None
    assert kept[2]["kept_own_tip"] and kept[2]["dropped"]  # tip as a ray, plane out
    # the same camera reading a tip 20mm up the shaft stays dropped
    p1, p2, diag = own
    short = (p1, p2, {**diag, "tip_px": tuple(_pxf(np.asarray(TIP_MM) + 20.0 * LEAN, CALS[2]))})
    assert camera_fit(short, CALS[2], dropped, 2) is None


def _tri(xyz):
    real = score_dart({k: tuple(_true_tip_px(k)) for k in CALS}, CALS).triangulation
    return dataclasses.replace(real, point_xyz=np.asarray(xyz, float))


def test_pair_meeting_off_the_board_is_replaced_by_one_on_it():
    tips = {k: tuple(_true_tip_px(k)) for k in CALS}
    high = ScoreResult(
        ok=True, sector="1", ring="single_inner", board_xy_mm=(0.0, 0.0),
        triangulation=_tri((0.0, 0.0, PAIR_MAX_BOARD_Z_MM + 12.0)),
        n_cameras_used=2, cameras_used=(0, 1), outlier_camera=2, max_ray_disagreement_mm=0.5,
    )
    tips[1] = tuple(_pxf(np.asarray(TIP_MM) + 30.0 * LEAN, CALS[1]))  # cam1 reads up the shaft
    r = prefer_pair_on_board(high, tips, CALS)
    assert set(r.cameras_used) == {0, 2} and r.outlier_camera == 1
    assert math.dist(r.board_xy_mm, TIP_MM[:2]) < 1
    # a pair already on the board is left alone
    low = ScoreResult(**{**high.__dict__, "triangulation": _tri((0.0, 0.0, 1.0))})
    assert prefer_pair_on_board(low, tips, CALS) is low


def test_loose_three_ray_set_gives_way_to_a_tight_pair():
    tips = {k: tuple(_true_tip_px(k)) for k in CALS}
    tips[2] = tuple(np.add(tips[2], (24.0, 0.0)))  # one camera several mm off
    full = score_dart(tips, CALS)
    assert full.ok and full.outlier_camera is None and full.max_ray_disagreement_mm > 5
    r = prefer_pair_on_board(full, tips, CALS)
    assert set(r.cameras_used) == {0, 1} and r.outlier_camera == 2
    assert math.dist(r.board_xy_mm, TIP_MM[:2]) < 1
    # a set that already agrees is kept
    tight = score_dart({k: tuple(_true_tip_px(k)) for k in CALS}, CALS)
    assert prefer_pair_on_board(tight, {k: tuple(_true_tip_px(k)) for k in CALS}, CALS) is tight


def _planes_for_axis(direction):
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    planes = []
    for helper in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)):
        n = np.cross(d, helper)
        n /= np.linalg.norm(n)
        planes.append((n, 0.0))  # both planes contain the axis through the origin
    return planes


def test_slide_refuses_an_implausibly_tilted_axis():
    point = (5.0, 5.0, 3.0)
    upright = _planes_for_axis((math.sin(math.radians(15)), 0.0, math.cos(math.radians(15))))
    lying = _planes_for_axis((math.sin(math.radians(60)), 0.0, math.cos(math.radians(60))))
    assert slide_point_along_dart_axis(point, upright, max_tilt_deg=MAX_PLAUSIBLE_TILT_DEG) is not None
    assert slide_point_along_dart_axis(point, lying) is not None
    assert slide_point_along_dart_axis(point, lying, max_tilt_deg=MAX_PLAUSIBLE_TILT_DEG) is None


def test_apollo_select_blobs_is_select_from_options():
    bgs, frames = _scene(extra={0: _decoy})
    whole = apollo_cross_view.select_blobs(bgs, frames, CALS)
    opts = {k: apollo_cross_view.camera_options(bgs[k], frames[k], CALS[k]) for k in CALS}
    split = apollo_cross_view.select_from_options(opts, bgs, frames, CALS)
    assert whole.cost == split.cost
    assert {k: (o.rank, o.flip) for k, o in whole.picks.items()} == {k: (o.rank, o.flip) for k, o in split.picks.items()}
