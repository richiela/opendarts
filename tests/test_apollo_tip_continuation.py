"""Apollo carrying a camera's tip across a gap to the needle (2026-09-29).

The carried tips are kept only when they land the triangulated point at
least `CARRIED_TIP_MIN_GAIN_MM` closer to the board plane; otherwise every
tier scores the original tips. See the comment above
`CARRIED_TIP_MIN_GAIN_MM` in `opendarts/engines/apollo/engine.py`.

The engine's detection, ROI gate and triangulation are stubbed so each
test drives exactly the per-camera outcome it means to; `continue_tip`
itself (Athena's, reused unchanged) is exercised on a synthetic image.
"""
from __future__ import annotations

import numpy as np

import opendarts.engines.apollo.engine as engine_mod
from opendarts.engines.apollo.engine import ApolloEngine
from opendarts.engines.apollo.tip_detection import TipDetectionResult
from opendarts.pipeline import ScoreResult
from opendarts.triangulation.rays import TriangulationResult


class _FakeCalib:
    landmark_spread_ok = True


def _result(z, xy=(0.0, 100.0)):
    tri = TriangulationResult(
        ok=True, point_xyz=np.array([xy[0], xy[1], z]), board_plane_xy=xy,
        plane_discrepancy_mm=abs(z), per_ray_distance_mm=[0.5, 0.5, 0.5],
        per_ray_depth_mm=[500.0, 500.0, 500.0], all_positive_depth=True,
        condition_number=10.0, well_conditioned=True, n_rays=3,
    )
    return ScoreResult(
        ok=True, sector="20", ring="single_inner", board_xy_mm=xy, triangulation=tri,
        n_cameras_used=3, max_ray_disagreement_mm=0.5, cameras_used=(0, 1, 2),
    )


def _run(monkeypatch, carried_z, original_z, move_px=40.0):
    """Cam 0's tip stops short at (100, 100); `continue_tip` carries it to
    (100, 140). The stubbed triangulation puts the point `carried_z` mm off
    the board when cam 0's tip is carried, `original_z` when it is not."""
    seen: dict = {"override_tips": None}
    tips = {0: (100.0, 100.0), 1: (300.0, 300.0), 2: (500.0, 500.0)}

    def fake_detection(bg, frame, calib, cross_pick, cam, prior_line, pc_kwargs):
        tip = tips[cam]
        return TipDetectionResult(ok=True, tip_px=tip, reason="ok", far_end_px=(tip[0], tip[1] - 80.0))

    def fake_continue(bg, frame, tip_px, axis_unit):
        if tip_px == tips[0]:
            return (100.0, 100.0 + move_px), move_px
        return tip_px, 0.0

    def fake_score(tip_pixels, calibration, alt_tip_pixels=None):
        return _result(carried_z if tip_pixels.get(0) != tips[0] else original_z)

    def spy_override(result, tip_pixels, *a, **k):
        seen["override_tips"] = dict(tip_pixels)
        return result

    monkeypatch.setattr(engine_mod, "_camera_detection", fake_detection)
    monkeypatch.setattr(engine_mod, "continue_tip", fake_continue)
    monkeypatch.setattr(engine_mod, "reject_outside_roi", lambda det, calibration: det)
    monkeypatch.setattr(engine_mod, "score_dart", fake_score)
    monkeypatch.setattr(engine_mod, "_cross_view_pick", lambda *a, **k: None)
    monkeypatch.setattr(engine_mod, "_per_camera_vote_override", spy_override)
    images = {c: np.zeros((8, 8, 3), np.uint8) for c in tips}
    calibration = {c: _FakeCalib() for c in tips}
    er = ApolloEngine().score(images, images, calibration)
    return er, seen


def test_carried_tip_kept_when_it_lands_the_point_on_the_board(monkeypatch):
    """#1292's shape: the original tips meet 23.6mm up; the carried ones on the board."""
    er, seen = _run(monkeypatch, carried_z=-2.1, original_z=23.6)
    tc = er.diagnostics["tip_continuation"]
    assert tc["kept"] is True
    assert tc["moved_px"] == {"0": 40.0}
    assert seen["override_tips"][0] == (100.0, 140.0)


def test_runaway_carry_that_moves_the_point_off_the_board_is_reverted(monkeypatch):
    """A walk up the flight or along a wire pushes the point away from the board."""
    er, seen = _run(monkeypatch, carried_z=9.0, original_z=0.5)
    assert er.diagnostics["tip_continuation"]["kept"] is False
    assert seen["override_tips"][0] == (100.0, 100.0)


def test_carry_that_gains_under_the_minimum_is_reverted(monkeypatch):
    er, _ = _run(monkeypatch, carried_z=1.6, original_z=2.4)
    assert er.diagnostics["tip_continuation"]["kept"] is False


def test_override_scores_the_same_tips_the_gate_chose(monkeypatch):
    """Every later tier must see the gate's choice, never the other set."""
    _, kept = _run(monkeypatch, carried_z=0.0, original_z=15.0)
    _, reverted = _run(monkeypatch, carried_z=15.0, original_z=0.0)
    assert kept["override_tips"][0] == (100.0, 140.0)
    assert reverted["override_tips"][0] == (100.0, 100.0)


def test_nothing_to_carry_leaves_no_trace(monkeypatch):
    er, seen = _run(monkeypatch, carried_z=0.0, original_z=0.0, move_px=0.0)
    assert "tip_continuation" not in er.diagnostics
    assert seen["override_tips"][0] == (100.0, 100.0)


def test_continued_detections_reaches_a_needle_past_a_gap():
    """The real `continue_tip` on a synthetic broken dart: a bright barrel
    ending at y=100, a 15px gap, then a faint needle down to y=150."""
    bg = np.full((240, 200, 3), 40, np.uint8)
    frame = bg.copy()
    frame[20:101, 98:103] = 200  # barrel, ending at the detected tip
    frame[116:151, 99:102] = 65  # faint needle past the gap
    det = TipDetectionResult(ok=True, tip_px=(100.0, 100.0), reason="ok", far_end_px=(100.0, 20.0))
    out = engine_mod._continued_detections({0: bg}, {0: frame}, {0: det})
    assert 0 in out
    carried, move = out[0]
    assert move > 30
    assert 140 <= carried.tip_px[1] <= 152
    assert det.tip_px == (100.0, 100.0), "the original detection must not be mutated"
