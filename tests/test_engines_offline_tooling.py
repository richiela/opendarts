"""Engine-awareness of opendarts/capture/replay.py and
opendarts/capture/rescore_all.py -- docs/ENGINES.md's "Offline tooling"
section: "able to re-run an archived package (or the whole archive)
through any registered engine by name, not just the hardcoded
pipeline."
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opendarts.capture.replay import replay_throw_with_engine
from opendarts.capture.rescore_all import rescore_all
from opendarts.capture.throw_package import load_throw_package, save_throw_package
from opendarts.pipeline import CameraCalibration, ScoreResult



def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=np.zeros(3, dtype=np.float64),
        tvec=np.array([0.0, 0.0, 1000.0], dtype=np.float64),
        landmark_spread_ok=True,
    )


@pytest.fixture()
def package_root(tmp_path):
    root = tmp_path / "packages"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_fake_package(root: Path, session="sess1", throw_id="throw1", ok=True) -> Path:
    bg = {0: np.zeros((6, 6, 3), dtype=np.uint8), 1: np.zeros((6, 6, 3), dtype=np.uint8)}
    frame = {0: np.full((6, 6, 3), 255, dtype=np.uint8), 1: np.full((6, 6, 3), 255, dtype=np.uint8)}
    calibrations = {0: _fake_calibration(), 1: _fake_calibration()}
    dest_dir = root / session / throw_id
    result = ScoreResult(
        ok=ok, sector="20" if ok else None, ring="treble" if ok else None,
        board_xy_mm=(1.0, 2.0) if ok else None, triangulation=None, n_cameras_used=2,
    )
    save_throw_package(
        dest_dir=dest_dir, session=session, bg_frames_bgr=bg, dart_frames_bgr=frame,
        calibrations=calibrations, result=result,
    )
    return dest_dir


# ---------------------------------------------------------------------------
# opendarts.capture.replay
# ---------------------------------------------------------------------------


def test_replay_throw_with_engine_requires_an_explicit_engine_name(package_root):
    """2026-09-05 -- `engine_name` is now a REQUIRED positional argument,
    not a `DEFAULT_PRIMARY_ENGINE`-defaulted one (see
    `opendarts/capture/replay.py`'s own docstring for the real incident a
    silent default caused: a day-long offline replay comparison silently
    ran against Apollo while live production actually scores with
    Zeus, producing a false "STORE!=SCORE" bug report). This test used to
    be `test_replay_throw_with_engine_default_matches_replay_throw`,
    proving the now-deleted `replay_throw()` matched `replay_throw_with_
    engine()`'s own then-default -- that comparison is gone along with
    both defaults; this replaces it with a structural proof that calling
    without an engine name is a real `TypeError`, not a silent guess."""
    dest_dir = _save_fake_package(package_root)
    package = load_throw_package(dest_dir)

    with pytest.raises(TypeError):
        replay_throw_with_engine(package)  # missing required engine_name

    # The explicit call still works exactly as before -- just proving it
    # returns a real EngineResult, not asserting a specific accuracy
    # number on this deliberately blank/fake fixture (see
    # _save_fake_package's own docstring: it's a plumbing fixture, not a
    # detection-accuracy one).
    via_engine = replay_throw_with_engine(package, "Apollo")
    assert via_engine.ok in (True, False)


def test_replay_throw_with_engine_talos_fails_honestly_on_blank_tiny_images(package_root):
    """6x6 blank package: Talos cannot reconstruct shaft planes
    (measured ok=False, ring=None). Must actually invoke Talos, not
    Apollo -- Apollo's reason on this fixture is about tip pixels, not
    shaft planes. Original result.json stays untouched."""
    dest_dir = _save_fake_package(package_root)
    before = (dest_dir / "result.json").read_text()
    result = replay_throw_with_engine(dest_dir, "Talos")  # accepts a Path directly too
    assert result.ok is False
    assert result.sector is None
    assert result.ring is None
    assert result.board_xy_mm is None
    assert result.diagnostics.get("engine") == "Talos"
    assert ">=2 shaft planes" in result.reason
    after = (dest_dir / "result.json").read_text()
    assert after == before
    orig = load_throw_package(dest_dir).original_result
    assert orig["ok"] is True
    assert orig["sector"] == "20"


def test_replay_throw_with_engine_unknown_name_raises_key_error(package_root):
    dest_dir = _save_fake_package(package_root)
    with pytest.raises(KeyError):
        replay_throw_with_engine(dest_dir, "NotRegistered")


# ---------------------------------------------------------------------------
# opendarts.capture.rescore_all
# ---------------------------------------------------------------------------


def test_rescore_all_default_engine_matches_pre_existing_behavior(package_root):
    _save_fake_package(package_root, throw_id="throw1", ok=True)
    summary = rescore_all(package_root)  # default engine="Apollo"
    assert summary.total_found == 1
    assert summary.outcomes[0].status == "scored"


def test_rescore_all_talos_engine_does_not_reproduce_stored_apollo_sector(package_root):
    """Fake 6x6 package stores Apollo-shaped sector '20'. Talos on that
    fixture cannot form shaft planes (measured ok=False) -- so the fresh
    result differs, while the ORIGINAL stored result is preserved."""
    dest_dir = _save_fake_package(package_root, throw_id="throw1", ok=True)
    before = (dest_dir / "result.json").read_text()
    summary = rescore_all(package_root, engine="Talos")
    assert summary.total_found == 1
    outcome = summary.outcomes[0]
    assert outcome.status == "scored"
    assert outcome.fresh_ok is False
    assert outcome.fresh_sector is None
    assert outcome.fresh_ring is None
    assert outcome.fresh_reason is not None and ">=2 shaft planes" in outcome.fresh_reason
    # The ORIGINAL result (ok=True, sector=20) is preserved for
    # comparison -- Talos replacing the fresh side doesn't touch it.
    assert outcome.original_ok is True
    assert outcome.original_sector == "20"
    assert (dest_dir / "result.json").read_text() == before
    # Talos did not reproduce Apollo's stored sector "20".
    assert outcome.sector_changed is True


def test_rescore_all_unknown_engine_raises_immediately_before_touching_any_package(
    package_root,
):
    _save_fake_package(package_root, throw_id="throw1", ok=True)
    with pytest.raises(ValueError, match="unknown engine"):
        rescore_all(package_root, engine="NotRegistered")


def test_rescore_all_talos_handles_a_package_with_no_original_result(package_root):
    dest_dir = _save_fake_package(package_root, throw_id="throw1", ok=True)
    (dest_dir / "result.json").unlink()  # simulate "no original_result to diff"
    summary = rescore_all(package_root, engine="Talos")
    outcome = summary.outcomes[0]
    assert outcome.had_original_result is False
    assert outcome.fresh_ok is False
    assert outcome.fresh_reason is not None and ">=2 shaft planes" in outcome.fresh_reason
    assert outcome.sector_changed is None  # no comparison possible, not "unchanged"


def test_cli_has_an_engine_flag():
    from opendarts.capture.rescore_all import main
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    assert "--engine" in buf.getvalue()
