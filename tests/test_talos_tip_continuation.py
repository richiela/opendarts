"""Talos carrying a camera's tip across a gap to the needle (2026-09-29),
and the guard that keeps the cross-view pick from leaving Talos without
two cameras.

The carried tips are kept only when the pre-slide ray point they give sits
at least `CARRIED_TIP_MIN_GAIN_MM` closer to the board plane; a carried set
that does not score never replaces one that does. See the comment above the
carry in `opendarts/engines/talos/engine.py`'s `score()`.

The per-camera fit, the cross-view step and the scoring from fits are
stubbed so each test drives exactly the outcome it means to;
`continue_tip` itself (Athena's, reused unchanged) is exercised on a
synthetic image through `_carried_fits`.
"""
from __future__ import annotations

import numpy as np

import opendarts.engines.talos.engine as engine_mod
from opendarts.engines.base import EngineResult

TIPS = {0: (100.0, 100.0), 1: (300.0, 300.0), 2: (500.0, 500.0)}


def _fit(cam):
    tip = TIPS[cam]
    return (tip[0], tip[1] - 80.0), tip, {"tip_px": tip}


def _result(ok=True):
    return EngineResult(
        ok=ok, sector="20" if ok else None, ring="single_inner" if ok else None,
        board_xy_mm=(0.0, 100.0) if ok else None, reason="" if ok else "no score",
        diagnostics={"engine": "Talos"},
    )


def _run(monkeypatch, carried, original, move_px=40.0):
    """Cam 0's tip stops short; `continue_tip` carries it 40px on.
    `carried` / `original` are the (result, pre-slide height) `_score_fits`
    returns for the carried and the original fits."""
    seen = {"scored": []}

    def fake_continue(bg, frame, tip_px, axis_unit):
        if tuple(tip_px) == TIPS[0]:
            return (100.0, 100.0 + move_px), move_px
        return tip_px, 0.0

    def fake_score_fits(fits, common, calibration, cross_pick, prior_extra):
        is_carried = fits[0][2]["tip_px"] != TIPS[0]
        seen["scored"].append("carried" if is_carried else "original")
        return carried() if is_carried else original()

    monkeypatch.setattr(engine_mod, "fit_shaft_line_px", lambda bg, frame, **k: _fit(int(bg[0, 0, 0])))
    monkeypatch.setattr(engine_mod, "select_talos_blobs", lambda *a, **k: None)
    monkeypatch.setattr(engine_mod, "continue_tip", fake_continue)
    monkeypatch.setattr(engine_mod, "_score_fits", fake_score_fits)
    images = {c: np.full((8, 8, 3), c, np.uint8) for c in TIPS}
    er = engine_mod.score(images, images, {c: object() for c in TIPS})
    return er, seen


def test_carried_tip_kept_when_it_lands_the_point_on_the_board(monkeypatch):
    """#1292's shape: the original tips meet 24.7mm up; the carried ones 18.5mm."""
    er, _ = _run(monkeypatch, carried=lambda: (_result(), 18.5), original=lambda: (_result(), 24.7))
    tc = er.diagnostics["tip_continuation"]
    assert tc["kept"] is True
    assert tc["moved_px"] == {"0": 40.0}


def test_runaway_carry_that_moves_the_point_off_the_board_is_reverted(monkeypatch):
    er, _ = _run(monkeypatch, carried=lambda: (_result(), 9.0), original=lambda: (_result(), 0.5))
    assert er.diagnostics["tip_continuation"]["kept"] is False


def test_carry_that_gains_under_the_minimum_is_reverted(monkeypatch):
    er, _ = _run(monkeypatch, carried=lambda: (_result(), 1.6), original=lambda: (_result(), 2.4))
    assert er.diagnostics["tip_continuation"]["kept"] is False


def test_a_carried_set_that_does_not_score_never_replaces_one_that_does(monkeypatch):
    """Every engine must score: the original result stands."""
    er, _ = _run(monkeypatch, carried=lambda: (_result(ok=False), None), original=lambda: (_result(), 12.0))
    assert er.ok
    assert er.diagnostics["tip_continuation"]["kept"] is False


def test_a_carried_set_that_scores_replaces_one_that_does_not(monkeypatch):
    er, _ = _run(monkeypatch, carried=lambda: (_result(), 3.0), original=lambda: (_result(ok=False), None))
    assert er.ok
    assert er.diagnostics["tip_continuation"]["kept"] is True


def test_nothing_to_carry_scores_once_and_leaves_no_trace(monkeypatch):
    er, seen = _run(monkeypatch, carried=lambda: (_result(), 0.0), original=lambda: (_result(), 0.0), move_px=0.0)
    assert "tip_continuation" not in er.diagnostics
    assert seen["scored"] == ["original"]


def test_cross_view_pick_that_leaves_under_two_cameras_yields_to_own_fits(monkeypatch):
    """camera_fit() blanking cameras must never leave Talos without two."""
    passed = {}

    def fake_score_fits(fits, common, calibration, cross_pick, prior_extra):
        passed["fits"] = fits
        passed["extra"] = prior_extra
        return _result(), 0.0

    monkeypatch.setattr(engine_mod, "fit_shaft_line_px", lambda bg, frame, **k: _fit(int(bg[0, 0, 0])))
    monkeypatch.setattr(engine_mod, "select_talos_blobs", lambda *a, **k: object())
    monkeypatch.setattr(engine_mod, "camera_fit", lambda own, calib, pick, cam: own if cam == 0 else None)
    monkeypatch.setattr(engine_mod, "continue_tip", lambda bg, frame, tip, axis: (tip, 0.0))
    monkeypatch.setattr(engine_mod, "_score_fits", fake_score_fits)
    images = {c: np.full((8, 8, 3), c, np.uint8) for c in TIPS}
    engine_mod.score(images, images, {c: object() for c in TIPS})
    assert all(passed["fits"][c] is not None for c in TIPS)
    assert "cross_view_guard" in passed["extra"]


def test_cross_view_pick_keeping_two_cameras_is_left_alone(monkeypatch):
    passed = {}

    def fake_score_fits(fits, common, calibration, cross_pick, prior_extra):
        passed["fits"] = fits
        passed["extra"] = prior_extra
        return _result(), 0.0

    monkeypatch.setattr(engine_mod, "fit_shaft_line_px", lambda bg, frame, **k: _fit(int(bg[0, 0, 0])))
    monkeypatch.setattr(engine_mod, "select_talos_blobs", lambda *a, **k: object())
    monkeypatch.setattr(engine_mod, "camera_fit", lambda own, calib, pick, cam: None if cam == 2 else own)
    monkeypatch.setattr(engine_mod, "continue_tip", lambda bg, frame, tip, axis: (tip, 0.0))
    monkeypatch.setattr(engine_mod, "_score_fits", fake_score_fits)
    images = {c: np.full((8, 8, 3), c, np.uint8) for c in TIPS}
    engine_mod.score(images, images, {c: object() for c in TIPS})
    assert passed["fits"][2] is None
    assert "cross_view_guard" not in passed["extra"]


def test_carried_fits_reach_a_needle_past_a_gap():
    """The real `continue_tip` on a synthetic broken dart: a bright barrel
    ending at y=100, a 15px gap, then a faint needle down to y=150."""
    bg = np.full((240, 200, 3), 40, np.uint8)
    frame = bg.copy()
    frame[20:101, 98:103] = 200  # barrel, ending at the fitted tip
    frame[116:151, 99:102] = 65  # faint needle past the gap
    diag = {"tip_px": (100.0, 100.0)}
    fits = {0: ((100.0, 20.0), (100.0, 100.0), diag)}
    out = engine_mod._carried_fits({0: bg}, {0: frame}, fits)
    assert 0 in out
    (p1, p2, carried_diag), move = out[0]
    assert move > 30
    assert 140 <= carried_diag["tip_px"][1] <= 152
    assert (p1, p2) == ((100.0, 20.0), (100.0, 100.0)), "the shaft line must not change"
    assert diag["tip_px"] == (100.0, 100.0), "the original fit must not be mutated"
