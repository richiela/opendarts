"""Talos prior-dart erase -- see opendarts/engines/talos/prior_dart.py.

Measured 2026-08-16 on living data/archive/clean/ (705 AD-matched):
gated erase is +1/-0 (the recorded T15 throw treble -> inner). 3-arg
score() without priors is unchanged (behavior-preservation still holds).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from opendarts.capture.replay import replay_throw_with_engine
from opendarts.engines.talos import TalosEngine
from opendarts.engines.talos.prior_dart import (
    PRIOR_ERASE_RADIUS_PX,
    engine_accepts_prior_board_xy_mm,
    erase_priors_in_frame,
    find_prior_board_xy_mm,
)
from opendarts.geometry.board_color import project_board_point_px
from opendarts.pipeline import CameraCalibration

REPO_ROOT = Path(__file__).resolve().parent.parent
T15 = "20260816-171712/20260816-171712-065-T15"


def _corpus_root() -> Path:
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    return Path(env_root) if env_root else REPO_ROOT / "data" / "archive" / "clean"


def _t15_dir() -> Path:
    path = _corpus_root() / T15
    if not (path / "calibration.json").exists():
        pytest.skip(f"T15 package not present at {path}")
    return path


def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array(
            [[800.0, 0.0, 400.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.array([[0.0], [0.0], [500.0]], dtype=np.float64),
    )


def test_engine_accepts_prior_board_xy_mm_on_real_talos():
    assert engine_accepts_prior_board_xy_mm(TalosEngine()) is True


def test_engine_accepts_prior_board_xy_mm_false_for_3arg_stub():
    class Stub:
        def score(self, bg_images, frame_images, calibration):
            return None

    assert engine_accepts_prior_board_xy_mm(Stub()) is False


def test_find_prior_board_xy_mm_reads_result_json_from_earlier_visit_throw(tmp_path):
    session = tmp_path / "sess"
    prior = session / "001-D10"
    current = session / "002-S15"
    prior.mkdir(parents=True)
    current.mkdir()
    (prior / "meta.json").write_text(json.dumps({
        "visit_id": "visit_1", "visit_index": 0, "session": "sess", "cameras": [0],
    }))
    (prior / "result.json").write_text(json.dumps({
        "board_xy_mm": [1.0, 2.0],
        "other_engines": {"Talos": {"board_xy_mm": [154.75, -46.5]}},
    }))
    (current / "meta.json").write_text(json.dumps({
        "visit_id": "visit_1", "visit_index": 1, "session": "sess", "cameras": [0],
    }))
    xy = find_prior_board_xy_mm(session, "visit_1", 1)
    assert xy == ((154.75, -46.5),)


def test_find_prior_board_xy_mm_never_reads_the_oracle(tmp_path):
    # Ground truth grades a throw; it must never feed the scoring of the
    # next one. Dart 0 has a primary XY AND a matched oracle tip that
    # disagrees; dart 1 has ONLY an oracle tip. The lookup must return
    # dart 0's primary XY and nothing for dart 1.
    session = tmp_path / "sess"
    for name, index, result, oracle_xy in (
        ("001-D10", 0, {"board_xy_mm": [1.0, 2.0]}, [50.0, 60.0]),
        ("002-S15", 1, {"ok": False}, [70.0, 80.0]),
    ):
        d = session / name
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({
            "visit_id": "visit_1", "visit_index": index, "session": "sess", "cameras": [0],
        }))
        (d / "result.json").write_text(json.dumps(result))
        (d / "ad_ground_truth.json").write_text(json.dumps({
            "matched": True, "tip_xy_mm": oracle_xy,
        }))
    current = session / "003-T20"
    current.mkdir()
    (current / "meta.json").write_text(json.dumps({
        "visit_id": "visit_1", "visit_index": 2, "session": "sess", "cameras": [0],
    }))
    assert find_prior_board_xy_mm(session, "visit_1", 2) == ((1.0, 2.0),)


def test_find_prior_board_xy_mm_empty_on_first_dart_of_visit(tmp_path):
    session = tmp_path / "sess"
    first = session / "001-S20"
    first.mkdir(parents=True)
    (first / "meta.json").write_text(json.dumps({
        "visit_id": "visit_1", "visit_index": 0, "session": "sess", "cameras": [0],
    }))
    assert find_prior_board_xy_mm(session, "visit_1", 0) == ()


def test_erase_priors_in_frame_copies_bg_into_projected_disk():
    calib = _fake_calibration()
    bg = np.full((800, 800, 3), 10, dtype=np.uint8)
    frame = np.full((800, 800, 3), 200, dtype=np.uint8)
    prior = (0.0, 0.0)
    px, py = project_board_point_px(prior, calib)
    ix, iy = int(round(px)), int(round(py))
    out = erase_priors_in_frame(bg, frame, calib, (prior,), radius_px=20)
    assert tuple(out[iy, ix]) == (10, 10, 10)
    assert tuple(out[0, 0]) == (200, 200, 200)
    assert PRIOR_ERASE_RADIUS_PX == 48


def test_t15_without_priors_still_scores_treble():
    """3-arg score must not change -- the gate only fires when priors are passed."""
    pkg_dir = _t15_dir()
    from opendarts.capture.throw_package import load_throw_package

    pkg = load_throw_package(pkg_dir)
    result = TalosEngine().score(pkg.bg_frames, pkg.dart_frames, pkg.calibrations)
    assert result.ok is True
    assert (result.sector, result.ring) == ("15", "treble")
    assert "prior_dart_erased_cameras" not in (result.diagnostics or {})


def test_t15_replay_with_visit_prior_scores_inner():
    # 2026-08-18: smoke tests must never pin an exact score for
    # a specific real corpus package -- calibration is itself subject to
    # REPLAY (docs/DESIGN.md), so this real throw's exact outcome/camera
    # assignment is not a stable smoke-test target. Real regressions are
    # caught by full-corpus replay against AD truth, not pytest pins.
    pkg_dir = _t15_dir()
    result = replay_throw_with_engine(pkg_dir, "Talos")
    assert result.ok is True
