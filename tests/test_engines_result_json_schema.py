"""result.json schema -- docs/ENGINES.md: existing top-level fields are
UNCHANGED (always the primary engine's result), `other_engines` is a
new, additive key, present only when at least one also-run engine
actually ran. Also covers
opendarts.capture.throw_package.write_other_engines_result() directly (the
function that adds it).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from opendarts.capture.throw_package import (
    save_throw_package,
    write_other_engines_result,
)
from opendarts.engines.base import EngineResult
from opendarts.pipeline import CameraCalibration, ScoreResult


def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=np.zeros(3, dtype=np.float64),
        tvec=np.array([0.0, 0.0, 1000.0], dtype=np.float64),
        landmark_spread_ok=True,
    )


def _fake_score_result(ok=True) -> ScoreResult:
    return ScoreResult(
        ok=ok,
        sector="20" if ok else None,
        ring="treble" if ok else None,
        board_xy_mm=(1.0, 2.0) if ok else None,
        triangulation=None,
        n_cameras_used=2,
        reason="fake" if not ok else "",
        max_ray_disagreement_mm=1.5 if ok else None,
    )


def _save_fake_package(tmp_path, ok=True):
    bg = {0: np.zeros((4, 4, 3), dtype=np.uint8)}
    frame = {0: np.ones((4, 4, 3), dtype=np.uint8)}
    calibrations = {0: _fake_calibration()}
    dest_dir = tmp_path / "packages" / "sess1" / "throw1"
    save_throw_package(
        dest_dir=dest_dir, session="sess1", bg_frames_bgr=bg, dart_frames_bgr=frame,
        calibrations=calibrations, result=_fake_score_result(ok=ok),
    )
    return dest_dir


def test_save_throw_package_result_json_has_no_other_engines_key_by_default(tmp_path):
    """The pre-existing, still-unchanged shape -- no also-run engine ever
    ran for this package, so `other_engines`/`primary_engine` must simply
    be absent, not null/empty."""
    dest_dir = _save_fake_package(tmp_path)
    data = json.loads((dest_dir / "result.json").read_text())
    assert "other_engines" not in data
    assert "primary_engine" not in data
    # Existing top-level fields, exactly as save_throw_package() has
    # always written them.
    assert data["ok"] is True
    assert data["sector"] == "20"
    assert data["ring"] == "treble"
    assert data["board_xy_mm"] == [1.0, 2.0]


def test_write_other_engines_result_adds_exactly_two_new_keys_untouched_primary(tmp_path):
    dest_dir = _save_fake_package(tmp_path)
    before = json.loads((dest_dir / "result.json").read_text())

    other = {
        "Talos": EngineResult(
            ok=True, sector=None, ring="outside", board_xy_mm=None,
            reason="stub engine -- always a miss", diagnostics={}, duration_s=0.001, timed_out=False,
        )
    }
    write_other_engines_result(dest_dir, "Apollo", other)

    after = json.loads((dest_dir / "result.json").read_text())
    # Every pre-existing key/value is BYTE-IDENTICAL.
    for key, value in before.items():
        assert after[key] == value, f"primary field {key!r} changed by write_other_engines_result()"

    # Engine names as written into packages:
    # Apollo -> Apollo, Talos -> Talos in what actually lands on disk.
    assert after["primary_engine"] == "Apollo"
    assert after["other_engines"] == {
        "Talos": {
            "ok": True, "sector": None, "ring": "outside", "board_xy_mm": None,
            "reason": "stub engine -- always a miss", "diagnostics": {},
            "duration_s": 0.001, "timed_out": False, "confidence": None,
        }
    }


def test_write_other_engines_result_accepts_plain_dicts_too(tmp_path):
    """Duck-typed like save_ad_ground_truth() -- a plain dict (not an
    EngineResult instance) works too, e.g. a value freshly loaded back
    off disk."""
    dest_dir = _save_fake_package(tmp_path)
    other = {"Talos": {"ok": True, "sector": None, "ring": "outside", "board_xy_mm": None}}
    write_other_engines_result(dest_dir, "Apollo", other)
    data = json.loads((dest_dir / "result.json").read_text())
    assert data["other_engines"]["Talos"]["ring"] == "outside"  # Talos -> Talos in packages


def test_write_other_engines_result_multiple_engines_in_one_write(tmp_path):
    dest_dir = _save_fake_package(tmp_path)
    other = {
        "Talos": EngineResult(ok=True, sector=None, ring="outside", board_xy_mm=None),
        "AnotherEngine": EngineResult(
            ok=False, sector=None, ring=None, board_xy_mm=None,
            reason="timed out", timed_out=True, duration_s=5.0,
        ),
    }
    write_other_engines_result(dest_dir, "Apollo", other)
    data = json.loads((dest_dir / "result.json").read_text())
    # Talos -> Talos via the mapping; "AnotherEngine" has no mapping
    # entry so it passes through unchanged -- confirms the fallback.
    assert set(data["other_engines"]) == {"Talos", "AnotherEngine"}
    assert data["other_engines"]["AnotherEngine"]["timed_out"] is True


def test_write_other_engines_result_raises_on_missing_package():
    with pytest.raises(FileNotFoundError):
        write_other_engines_result("/nonexistent/path/xyz", "Apollo", {})


def test_write_other_engines_result_on_a_rejected_primary_result_still_works(tmp_path):
    """Per docs/DESIGN.md's "Replay is the source of truth", a package recording an ok=False
    primary result is still a complete, real package -- other_engines
    must attach fine regardless of the primary's own ok value."""
    dest_dir = _save_fake_package(tmp_path, ok=False)
    write_other_engines_result(
        dest_dir, "Apollo", {"Talos": EngineResult(ok=True, sector=None, ring="outside", board_xy_mm=None)}
    )
    data = json.loads((dest_dir / "result.json").read_text())
    assert data["ok"] is False  # primary's real (rejected) result, untouched
    assert data["other_engines"]["Talos"]["ok"] is True  # Talos -> Talos
