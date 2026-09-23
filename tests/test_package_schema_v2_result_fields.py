"""Tests for the v2 package schema's "3 keys the spec missed"
(2026-08-27): `camera_mode`, `label` and `agreement` at the top level of
`result.json`.

Covers:
  1. `agreement_string_from_diagnostics()` -- the pure extraction logic,
     every real branch (unanimous/majority vote, quorum not reached,
     non-vote-based engine, missing/absent diagnostics).
  2. `save_throw_package()` -- `label` always written (computed
     internally, no caller context needed); `camera_mode`/`agreement`
     omitted-not-fabricated when not supplied, matching `throw_number`'s
     established convention.
  3. `validate_throw_package_result_v2()` -- both positive and negative
     cases.
  4. REPLAY backward compat: an old on-disk result.json (missing all
     three keys) still loads fine through `load_throw_package()` --
     these are archival-only fields (module docstring: "for audit/
     comparison only"), so they ride along inside `original_result`
     rather than needing dedicated `ThrowPackage` fields.
  5. End-to-end at the `opendarts.live.capture_daemon.handle_ready_to_capture()`
     level: `camera_mode` reflects the real frame source,
     `agreement` is recovered from the PRIMARY engine's own vote-tally
     diagnostics when the primary is Zeus, and absent when the primary
     isn't vote-based -- then backfilled from an also-run Zeus ("Zeus")
     entry once its background dispatch completes.

Deliberately does NOT re-test the base save/load round trip or the
throw_number/frame_cameras fields (tests/test_package_schema_v2.py already
owns those) -- only the new result.json surface this task touches.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import opendarts.live.capture_daemon as capture_daemon
from opendarts.capture.throw_package import (
    agreement_string_from_diagnostics,
    load_throw_package,
    save_throw_package,
    validate_throw_package_result_v2,
)
from opendarts.pipeline import CameraCalibration, ScoreResult
from tests.test_capture_daemon import (
    _fake_calibration_attempt,
    _throw_trigger_ready,
    _wait_until,
)

@pytest.fixture()
def pkg_dir(tmp_path):
    return tmp_path / "pkg"


def _marker_image(cam_idx: int, seed: int, h: int = 12, w: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img[0, 0, 0] = cam_idx
    return img


def _calib(seed: int) -> CameraCalibration:
    rng = np.random.default_rng(seed)
    return CameraCalibration(
        camera_matrix=np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=rng.uniform(-0.1, 0.1, size=3).astype(np.float64),
        tvec=np.array([0.0, 0.0, 400.0], dtype=np.float64),
        pnp_result=None,
        landmark_spread_ok=True,
    )


def _ok_result() -> ScoreResult:
    return ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0), triangulation=None,
        n_cameras_used=3, max_ray_disagreement_mm=0.5,
    )


def _no_result() -> ScoreResult:
    return ScoreResult(
        ok=False, sector=None, ring=None, board_xy_mm=None, triangulation=None,
        n_cameras_used=0, max_ray_disagreement_mm=None, reason="no dart detected",
    )


# ---------------------------------------------------------------------------
# 1. agreement_string_from_diagnostics() -- pure logic, every real branch.
# ---------------------------------------------------------------------------

def test_agreement_string_none_when_diagnostics_missing_or_empty():
    assert agreement_string_from_diagnostics(None) is None
    assert agreement_string_from_diagnostics({}) is None


def test_agreement_string_none_when_not_vote_based():
    """A non-vote-based engine's diagnostics (e.g. Apollo's own
    max_ray_disagreement_mm bucket) has no `sub_engine_names` key at all
    -- must return None, never a fabricated ratio."""
    assert agreement_string_from_diagnostics({"max_ray_disagreement_mm": 1.2}) is None


def test_agreement_string_zero_of_n_when_quorum_not_reached():
    """Zeus's own base_diagnostics shape when fewer than
    MIN_SUB_ENGINES_TO_VOTE sub-engines produced a usable result -- no
    `winner`/`vote_tally` keys at all, matching ZeusEngine.score()'s real
    early-return shape."""
    diagnostics = {
        "sub_engine_names": ["Apollo", "Talos", "Athena", "Ares"],
        "votes": {},
        "n_usable": 0,
    }
    assert agreement_string_from_diagnostics(diagnostics) == "0/4"


def test_agreement_string_unanimous_vote():
    diagnostics = {
        "sub_engine_names": ["Apollo", "Talos", "Athena", "Ares"],
        "n_usable": 4,
        "winner": ["6", "single_inner"],
        "vote_tally": [{"sector": "6", "ring": "single_inner", "count": 4}],
    }
    assert agreement_string_from_diagnostics(diagnostics) == "4/4"


def test_agreement_string_majority_vote():
    diagnostics = {
        "sub_engine_names": ["Apollo", "Talos", "Athena", "Ares"],
        "n_usable": 4,
        "winner": ["6", "single_inner"],
        "vote_tally": [
            {"sector": "6", "ring": "single_inner", "count": 3},
            {"sector": "6", "ring": "treble", "count": 1},
        ],
    }
    assert agreement_string_from_diagnostics(diagnostics) == "3/4"


def test_agreement_string_defensive_zero_when_winner_not_in_vote_tally():
    """Should never happen from a real ZeusEngine result (winner is always
    derived FROM vote_tally), but this function must degrade honestly
    (not raise, not fabricate a nonzero count) if it ever did."""
    diagnostics = {
        "sub_engine_names": ["Apollo", "Talos", "Athena"],
        "n_usable": 3,
        "winner": ["6", "single_inner"],
        "vote_tally": [{"sector": "20", "ring": "treble", "count": 3}],
    }
    assert agreement_string_from_diagnostics(diagnostics) == "0/3"


# ---------------------------------------------------------------------------
# 2. save_throw_package() -- label always, camera_mode/agreement optional.
# ---------------------------------------------------------------------------

def test_save_throw_package_writes_label_always_even_with_no_camera_mode_or_agreement(pkg_dir):
    bg = {i: _marker_image(i, 10 + i) for i in range(3)}
    dart = {i: _marker_image(i, 20 + i) for i in range(3)}
    calibs = {i: _calib(30 + i) for i in range(3)}

    save_throw_package(pkg_dir, "sess-label", bg, dart, calibs, _ok_result())

    result = json.loads((pkg_dir / "result.json").read_text())
    # sector="20" ring="treble" -> "T20" via the SAME canonical vocabulary
    # (opendarts.geometry.board.sector_ring_to_token) the throw_id already
    # uses -- not re-derived independently here.
    from opendarts.geometry.board import sector_ring_to_token
    assert result["label"] == sector_ring_to_token("20", "treble")
    assert "camera_mode" not in result
    assert "agreement" not in result
    validate_throw_package_result_v2(result)


def test_save_throw_package_label_is_nr_for_a_no_score_throw(pkg_dir):
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}

    save_throw_package(pkg_dir, "sess-nr", bg, dart, calibs, _no_result())

    result = json.loads((pkg_dir / "result.json").read_text())
    assert result["label"] == "NR"
    validate_throw_package_result_v2(result)


def test_save_throw_package_never_raises_on_an_out_of_vocabulary_ring_value(pkg_dir):
    """Real regression guard: sector_ring_to_token() intentionally raises
    loud on a ring value outside its strict vocabulary (see its own
    docstring) -- a real, deliberate contract for capture_daemon.py's own
    throw_id computation. But save_throw_package() is called far more
    broadly (offline tools, test fixtures using looser shorthand ring
    values like "single") and must NEVER refuse to save an otherwise
    complete, real package just because label-formatting couldn't be
    derived. label is honestly omitted instead of raising."""
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}
    loose_result = ScoreResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0), triangulation=None,
        n_cameras_used=1, max_ray_disagreement_mm=None,
    )

    save_throw_package(pkg_dir, "sess-loose-ring", bg, dart, calibs, loose_result)

    result = json.loads((pkg_dir / "result.json").read_text())
    assert "label" not in result
    assert result["ring"] == "single"
    validate_throw_package_result_v2(result)


def test_save_throw_package_writes_camera_mode_and_agreement_when_supplied(pkg_dir):
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}

    save_throw_package(
        pkg_dir, "sess-supplied", bg, dart, calibs, _ok_result(),
        camera_mode="real", agreement="3/4",
    )

    result = json.loads((pkg_dir / "result.json").read_text())
    assert result["camera_mode"] == "real"
    assert result["agreement"] == "3/4"
    validate_throw_package_result_v2(result)


# ---------------------------------------------------------------------------
# 3. validate_throw_package_result_v2() -- negative cases.
# ---------------------------------------------------------------------------

def test_validate_throw_package_result_v2_accepts_missing_label():
    """label is OPTIONAL -- see _safe_label()'s own docstring: a
    synthetic/test ScoreResult with an out-of-vocabulary ring value
    legitimately has no label, and that must not fail validation."""
    validate_throw_package_result_v2({"ok": True, "sector": "20", "ring": "treble"})


def test_validate_throw_package_result_v2_rejects_empty_label_when_present():
    with pytest.raises(AssertionError, match="label"):
        validate_throw_package_result_v2({"label": ""})


def test_validate_throw_package_result_v2_rejects_empty_camera_mode():
    with pytest.raises(AssertionError, match="camera_mode"):
        validate_throw_package_result_v2({"label": "T20", "camera_mode": ""})


def test_validate_throw_package_result_v2_rejects_non_string_agreement():
    with pytest.raises(AssertionError, match="agreement"):
        validate_throw_package_result_v2({"label": "T20", "agreement": 4})


def test_validate_throw_package_result_v2_accepts_label_only():
    validate_throw_package_result_v2({"label": "NR"})


# ---------------------------------------------------------------------------
# 4. REPLAY backward compat -- an old (pre-this-task) result.json loads fine.
# ---------------------------------------------------------------------------

def test_load_throw_package_old_result_json_missing_all_three_keys_still_loads(pkg_dir):
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}

    save_throw_package(pkg_dir, "sess-old-shape", bg, dart, calibs, _ok_result())

    # Simulate a genuinely pre-existing package: strip the label this
    # save just wrote, so result.json is exactly the v1 shape (this
    # task's own docstring: "these are archival-only fields", never
    # required for a package to remain loadable).
    result_path = pkg_dir / "result.json"
    result = json.loads(result_path.read_text())
    del result["label"]
    result_path.write_text(json.dumps(result))

    pkg = load_throw_package(pkg_dir)
    assert pkg.original_result is not None
    assert "label" not in pkg.original_result
    assert "camera_mode" not in pkg.original_result
    assert "agreement" not in pkg.original_result


# ---------------------------------------------------------------------------
# 5. End-to-end at handle_ready_to_capture() -- real camera_mode/agreement
#    wiring, primary + also-run backfill paths.
# ---------------------------------------------------------------------------



def test_handle_ready_to_capture_camera_mode_defaults_to_real(tmp_path):
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )
    data = json.loads((dest_dir / "result.json").read_text())
    assert data["camera_mode"] == "real"


def test_handle_ready_to_capture_agreement_absent_for_non_vote_based_primary(tmp_path):
    """`agreement` must be honestly absent, never fabricated, when the
    primary is not vote-based and no vote-based engine is also-run.

    Pinned to Apollo explicitly: the DEFAULT primary is Zeus, which IS
    vote-based and legitimately reports agreement -- so the default
    config cannot express the case this test exists to pin."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
        engine_config_store=capture_daemon.EngineConfigStore(
            primary="Apollo", also_run=(),
        ),
    )
    data = json.loads((dest_dir / "result.json").read_text())
    assert "agreement" not in data


def test_handle_ready_to_capture_agreement_from_zeus_primary(tmp_path):
    """Zeus as PRIMARY: fake 4x4 frames give every sub-engine ok=False, so
    Zeus itself never reaches MIN_SUB_ENGINES_TO_VOTE -- the real "0/4"
    branch, sourced from Zeus's own diagnostics synchronously, before the
    ScoreResult conversion would have discarded them."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(primary="Zeus", also_run=(), timeout_s=5.0)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        engine_config_store=store,
    background_save=False,
    )
    data = json.loads((dest_dir / "result.json").read_text())
    assert data["agreement"] == "0/4"
    assert data["label"] == "NR"


def test_handle_ready_to_capture_agreement_backfilled_from_also_run_zeus(tmp_path):
    """Primary is Apollo (not vote-based, agreement absent from the
    synchronous write) but Zeus is configured as an ALSO-RUN engine --
    once its background dispatch completes, write_other_engines_result()
    must backfill `agreement` from the now-available "Zeus" entry, the
    ONLY other place this information ever becomes available."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(primary="Apollo", also_run=("Zeus",), timeout_s=5.0)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        engine_config_store=store,
    background_save=False,
    )

    def _has_other_engines():
        data = json.loads((dest_dir / "result.json").read_text())
        return "other_engines" in data

    assert _wait_until(_has_other_engines), "other_engines never appeared in result.json"
    data = json.loads((dest_dir / "result.json").read_text())
    assert "Zeus" in data["other_engines"]
    assert data["agreement"] == "0/4"
