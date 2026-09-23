"""Tests for the v2 package schema (2026-08-27), the first real
v2-session QA pass: 3 real gaps found by running the actual parity
checker against 90 real production throws.

Covers:
  1. `_rollup_fields_from_winning_sub_engine_diagnostics()` -- the pure
     extraction logic, against diagnostics shapes traced directly from
     the real code (`ZeusEngine.score()`'s own `base_diagnostics`/
     `winning_engine`, `opendarts.engines.apollo.engine.
     score_result_to_engine_result()`'s own diagnostics shape for a
     Apollo winner, and Talos/Athena's own partial shapes).
  2. `save_throw_package()` -- fills the top-level rollup
     (`cameras_used`/`triangulation`/`max_ray_disagreement_mm`/
     `n_cameras_used`) from `primary_engine_diagnostics` only where
     `result` itself left the honest "not populated" default; never
     overwrites a real value (e.g. a Apollo-as-primary throw).
  3. `meta.generation` -- written as a real int when supplied, honestly
     absent (not a default 0) when not, same convention as
     `throw_number`.
  4. REPLAY backward compat: an old on-disk package (no `generation` key,
     null/0 rollup fields) still loads fine.
  5. End-to-end at `opendarts.live.capture_daemon.handle_ready_to_capture()`:
     a real Zeus-primary throw where 3 of 4 sub-engines are stubbed to
     actually WIN a vote with real, richly-populated diagnostics (the
     a realistic multi-engine fixture) -- proves the wiring end to end,
     not just the pure function.

Deliberately does NOT re-test camera_mode/label/agreement
(tests/test_package_schema_v2_result_fields.py already owns those) or the
base save/load round trip / throw_number/frame_cameras (tests/
test_package_schema_v2.py already owns those).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import opendarts.engines.registry as engine_registry
import opendarts.live.capture_daemon as capture_daemon
from opendarts.capture.throw_package import (
    _rollup_fields_from_winning_sub_engine_diagnostics,
    load_throw_package,
    save_throw_package,
    validate_throw_package_meta_v2,
)
from opendarts.engines.base import EngineResult
from opendarts.pipeline import CameraCalibration, ScoreResult
from tests.test_capture_daemon import (
    _fake_calibration_attempt,
    _throw_trigger_ready,
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


def _ok_result_with_full_rollup() -> ScoreResult:
    """A Apollo-as-primary-shaped ScoreResult -- rollup fields already
    real, matching what `apollo_engine_result_to_score_result()`
    always produces. Used to prove `primary_engine_diagnostics` never
    overwrites an already-real value."""
    return ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=3, max_ray_disagreement_mm=0.5,
        cameras_used=(0, 1, 2),
    )


def _ok_result_no_rollup() -> ScoreResult:
    """A Zeus-as-primary-shaped ScoreResult -- exactly what
    `opendarts.engines.base.engine_result_to_score_result()` (the generic
    adapter) produces today for a non-Apollo primary: real sector/ring/
    board_xy_mm, but the rollup fields honestly at their "not applicable"
    defaults."""
    return ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=0, max_ray_disagreement_mm=None,
        cameras_used=None,
    )


# ---------------------------------------------------------------------------
# 1. _rollup_fields_from_winning_sub_engine_diagnostics() -- pure logic.
# ---------------------------------------------------------------------------

def _real_apollo_sub_result_dict() -> dict:
    """Mirrors opendarts.engines.apollo.engine.score_result_to_engine_
    result()'s own diagnostics shape EXACTLY (traced from that function's
    real source, not guessed) -- a Apollo sub-engine's real .to_dict()
    output as it would sit inside Zeus's own `sub_results`."""
    return {
        "ok": True, "sector": "20", "ring": "treble", "board_xy_mm": [1.0, 2.0],
        "reason": "", "duration_s": 0.01, "timed_out": False, "confidence": 0.9,
        "diagnostics": {
            "n_cameras_used": 3,
            "max_ray_disagreement_mm": 0.42,
            "cameras_used": [0, 1, 2],
            "outlier_camera": None,
            "alt_candidates_used": None,
            "triangulation": {
                "ok": True, "point_xyz": [1.0, 2.0, 0.3],
                "board_plane_xy": [1.0, 2.0], "plane_discrepancy_mm": 0.3,
                "per_ray_distance_mm": [0.2, 0.3, 0.15], "n_rays": 3,
            },
            "confidence": 0.9,
        },
    }


def _real_talos_sub_result_dict() -> dict:
    """Talos's own real diagnostics shape -- has cameras_used/
    n_cameras_used but NEVER triangulation/max_ray_disagreement_mm (it is
    not a ray-triangulation engine) -- traced from
    opendarts/engines/talos/engine.py's own diagnostics construction."""
    return {
        "ok": True, "sector": "20", "ring": "treble", "board_xy_mm": [1.1, 1.9],
        "reason": "", "duration_s": 0.02, "timed_out": False, "confidence": 0.7,
        "diagnostics": {
            "engine": "Talos", "observation": "centerline_ray",
            "n_cameras_used": 2, "cameras_used": [0, 1],
        },
    }


def _zeus_diagnostics(winning_engine: str, winner_dict: dict) -> dict:
    return {
        "sub_engine_names": ["Apollo", "Talos", "Athena", "Ares"],
        "sub_results": {
            "Apollo": winner_dict if winning_engine == "Apollo" else _real_apollo_sub_result_dict(),
            "Talos": winner_dict if winning_engine == "Talos" else _real_talos_sub_result_dict(),
        },
        "n_usable": 2,
        "vote_tally": [{"sector": "20", "ring": "treble", "count": 2}],
        "winner": ["20", "treble"],
        "winning_engine": winning_engine,
        "tie_break_applied": False,
        "agreement": "unanimous",
    }


def test_rollup_extraction_full_from_apollo_winner():
    diagnostics = _zeus_diagnostics("Apollo", _real_apollo_sub_result_dict())
    rollup = _rollup_fields_from_winning_sub_engine_diagnostics(diagnostics)
    assert rollup["cameras_used"] == [0, 1, 2]
    assert rollup["n_cameras_used"] == 3
    assert rollup["max_ray_disagreement_mm"] == 0.42
    assert rollup["triangulation"]["ok"] is True
    assert rollup["triangulation"]["n_rays"] == 3


def test_rollup_extraction_partial_from_talos_winner():
    """Talos never populates triangulation/max_ray_disagreement_mm --
    those keys must be genuinely ABSENT from the returned dict, never a
    fabricated None/0 that looks like real "not applicable" data."""
    diagnostics = _zeus_diagnostics("Talos", _real_talos_sub_result_dict())
    rollup = _rollup_fields_from_winning_sub_engine_diagnostics(diagnostics)
    assert rollup["cameras_used"] == [0, 1]
    assert rollup["n_cameras_used"] == 2
    assert "triangulation" not in rollup
    assert "max_ray_disagreement_mm" not in rollup


@pytest.mark.parametrize("diagnostics", [None, {}, {"max_ray_disagreement_mm": 1.2}])
def test_rollup_extraction_empty_when_not_vote_based(diagnostics):
    """None/empty, or a non-vote-based engine's own diagnostics (no
    winning_engine/sub_results shape at all) -- must return {}, never
    raise, never fabricate."""
    assert _rollup_fields_from_winning_sub_engine_diagnostics(diagnostics) == {}


def test_rollup_extraction_empty_when_quorum_not_reached():
    """Zeus's own real 'quorum not reached' diagnostics shape -- no
    winning_engine/vote_tally/winner keys at all (see ZeusEngine.score()'s
    own early-return)."""
    diagnostics = {
        "sub_engine_names": ["Apollo", "Talos", "Athena", "Ares"],
        "sub_results": {},
        "votes": {},
        "n_usable": 0,
    }
    assert _rollup_fields_from_winning_sub_engine_diagnostics(diagnostics) == {}


def test_rollup_extraction_empty_when_winner_not_in_sub_results():
    """Defensive: a malformed/unexpected diagnostics dict where
    winning_engine names an engine not actually present in sub_results
    must not raise -- degrade to {}."""
    diagnostics = {"winning_engine": "Apollo", "sub_results": {}}
    assert _rollup_fields_from_winning_sub_engine_diagnostics(diagnostics) == {}


# ---------------------------------------------------------------------------
# 2. save_throw_package() -- rollup fill-in wiring.
# ---------------------------------------------------------------------------

def test_save_throw_package_fills_rollup_from_primary_engine_diagnostics(pkg_dir):
    bg = {i: _marker_image(i, 10 + i) for i in range(3)}
    dart = {i: _marker_image(i, 20 + i) for i in range(3)}
    calibs = {i: _calib(30 + i) for i in range(3)}
    diagnostics = _zeus_diagnostics("Apollo", _real_apollo_sub_result_dict())

    save_throw_package(
        pkg_dir, "sess-rollup", bg, dart, calibs, _ok_result_no_rollup(),
        primary_engine_diagnostics=diagnostics,
    )

    result = json.loads((pkg_dir / "result.json").read_text())
    assert result["cameras_used"] == [0, 1, 2]
    assert result["n_cameras_used"] == 3
    assert result["max_ray_disagreement_mm"] == 0.42
    assert result["triangulation"]["ok"] is True


def test_save_throw_package_rollup_absent_without_primary_engine_diagnostics(pkg_dir):
    """No `primary_engine_diagnostics` supplied (every pre-existing
    caller/test) -- today's pre-fix "not populated" defaults, EXCEPT
    `cameras_used` (2026-08-27, the v2 package schema null-vs-[] pass:
    a collection-typed field is `[]`, never `null`, regardless of why
    it's empty -- see opendarts.capture.throw_package._score_result_to_dict()'s
    own comment). `triangulation`/`max_ray_disagreement_mm` stay null --
    neither is list-typed."""
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}

    save_throw_package(pkg_dir, "sess-no-rollup-ctx", bg, dart, calibs, _ok_result_no_rollup())

    result = json.loads((pkg_dir / "result.json").read_text())
    assert result["cameras_used"] == []
    assert result["n_cameras_used"] == 0
    assert result["max_ray_disagreement_mm"] is None
    assert result["triangulation"] is None


def test_save_throw_package_never_overwrites_a_real_rollup_value(pkg_dir):
    """A Apollo-as-primary throw already has real rollup values on its
    own ScoreResult -- primary_engine_diagnostics (even if somehow
    supplied) must never clobber them with something else."""
    bg = {i: _marker_image(i, 10 + i) for i in range(3)}
    dart = {i: _marker_image(i, 20 + i) for i in range(3)}
    calibs = {i: _calib(30 + i) for i in range(3)}
    # Deliberately DIFFERENT from the real result, to prove it's ignored.
    diagnostics = _zeus_diagnostics("Talos", _real_talos_sub_result_dict())

    save_throw_package(
        pkg_dir, "sess-no-clobber", bg, dart, calibs, _ok_result_with_full_rollup(),
        primary_engine_diagnostics=diagnostics,
    )

    result = json.loads((pkg_dir / "result.json").read_text())
    assert result["cameras_used"] == [0, 1, 2]
    assert result["n_cameras_used"] == 3
    assert result["max_ray_disagreement_mm"] == 0.5


# ---------------------------------------------------------------------------
# 3. meta.generation -- real int field, absent-not-fabricated when unset.
# ---------------------------------------------------------------------------

def test_save_throw_package_writes_generation_when_supplied(pkg_dir):
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}

    save_throw_package(pkg_dir, "sess-gen", bg, dart, calibs, _ok_result_no_rollup(), generation=2)

    meta = json.loads((pkg_dir / "meta.json").read_text())
    assert meta["generation"] == 2
    assert isinstance(meta["generation"], int)
    validate_throw_package_meta_v2(meta)


def test_save_throw_package_writes_generation_zero_explicitly(pkg_dir):
    """generation=0 (the common/original-generation case) must be written
    as the real int 0, not omitted -- distinct from "not tracked"."""
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}

    save_throw_package(pkg_dir, "sess-gen0", bg, dart, calibs, _ok_result_no_rollup(), generation=0)

    meta = json.loads((pkg_dir / "meta.json").read_text())
    assert "generation" in meta
    assert meta["generation"] == 0


def test_save_throw_package_omits_generation_when_not_supplied(pkg_dir):
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}

    save_throw_package(pkg_dir, "sess-no-gen", bg, dart, calibs, _ok_result_no_rollup())

    meta = json.loads((pkg_dir / "meta.json").read_text())
    assert "generation" not in meta
    validate_throw_package_meta_v2(meta)


def test_validate_throw_package_meta_v2_rejects_non_int_generation():
    meta = {
        "session": "s", "captured_at_utc": "2026-08-27T00:00:00+00:00",
        "cameras": [0], "frame_cameras": [0], "generation": "1",
    }
    with pytest.raises(AssertionError, match="generation"):
        validate_throw_package_meta_v2(meta)


def test_validate_throw_package_meta_v2_rejects_bool_generation():
    """bool is a subclass of int in Python -- must be explicitly excluded,
    same guard throw_number already has."""
    meta = {
        "session": "s", "captured_at_utc": "2026-08-27T00:00:00+00:00",
        "cameras": [0], "frame_cameras": [0], "generation": True,
    }
    with pytest.raises(AssertionError, match="generation"):
        validate_throw_package_meta_v2(meta)


# ---------------------------------------------------------------------------
# 4. REPLAY backward compat.
# ---------------------------------------------------------------------------

def test_load_throw_package_old_meta_json_missing_generation_still_loads(pkg_dir):
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}
    save_throw_package(pkg_dir, "sess-old-meta", bg, dart, calibs, _ok_result_no_rollup())

    meta_path = pkg_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert "generation" not in meta  # not supplied above -- sanity check
    pkg = load_throw_package(pkg_dir)
    assert pkg.generation is None


def test_load_throw_package_old_result_json_null_rollup_still_loads(pkg_dir):
    """`cameras_used` reflects 2026-08-27's null-vs-[] fix (list-typed,
    never null); `triangulation`/`n_cameras_used` are unaffected by that
    fix (neither is list-typed)."""
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}
    save_throw_package(pkg_dir, "sess-old-result", bg, dart, calibs, _ok_result_no_rollup())

    pkg = load_throw_package(pkg_dir)
    assert pkg.original_result["cameras_used"] == []
    assert pkg.original_result["triangulation"] is None
    assert pkg.original_result["n_cameras_used"] == 0


# ---------------------------------------------------------------------------
# 5. End-to-end at handle_ready_to_capture() -- a real Zeus-primary throw
#    where a realistic, richly-populated sub-engine wins the vote.
# ---------------------------------------------------------------------------

class _StubEngine:
    """A registered-engine stand-in with a fixed, realistic EngineResult
    -- used to make Zeus's vote land on a KNOWN winner with KNOWN
    diagnostics, without needing real multi-camera dart detection to
    actually succeed (which trivial synthetic test frames cannot
    reliably produce)."""

    def __init__(self, result: EngineResult):
        self._result = result

    def score(self, bg_images, frame_images, calibration, **kwargs):
        return self._result


def _stub_apollo_winner() -> EngineResult:
    return EngineResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        reason="",
        diagnostics=_real_apollo_sub_result_dict()["diagnostics"],
        confidence=0.9,
    )


def _stub_talos_agree() -> EngineResult:
    return EngineResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.05, 2.05),
        reason="",
        diagnostics=_real_talos_sub_result_dict()["diagnostics"],
        confidence=0.7,
    )


def _stub_athena_agree() -> EngineResult:
    return EngineResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(0.95, 1.95),
        reason="",
        diagnostics={"n_cameras_used": 2, "cameras_used": [0, 1]},
        confidence=0.6,
    )




def test_handle_ready_to_capture_saved_package_log_line_uses_rollup_n_cameras_used(
    tmp_path, monkeypatch, caplog
):
    """Real bug, found live 2026-09-06 during a ghost-fire investigation:
    the "saved throw package" log line read `result.n_cameras_used`
    directly -- which `engine_result_to_score_result()` (used whenever
    the live primary is a vote-based consensus engine, i.e. every real
    Zeus-as-primary package) DELIBERATELY hardcodes to 0, since a generic
    ScoreResult has no per-engine equivalent to report. `result.json`'s
    own top-level field was already correctly promoted from the winning
    sub-engine's diagnostics (see the rollup test immediately above) --
    only this log line was still reading the pre-rollup, always-0 value,
    misleading anyone reading it (it nearly misled a real investigation:
    every package that night logged n_cameras_used=0 regardless of how
    confidently/correctly it actually scored). Same real fixture as the
    rollup test -- confirms the LOG LINE now reads 3, not 0."""
    monkeypatch.setitem(engine_registry.ENGINES, "Apollo", _StubEngine(_stub_apollo_winner()))
    monkeypatch.setitem(engine_registry.ENGINES, "Talos", _StubEngine(_stub_talos_agree()))
    monkeypatch.setitem(engine_registry.ENGINES, "Athena", _StubEngine(_stub_athena_agree()))

    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(primary="Zeus", also_run=(), timeout_s=5.0)

    with caplog.at_level("INFO", logger="opendarts.capture_daemon"):
        capture_daemon.handle_ready_to_capture(
            _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess-e2e-logline",
            engine_config_store=store,
            background_save=False,
        )

    saved_lines = [r.message for r in caplog.records if r.message.startswith("saved throw package")]
    assert len(saved_lines) == 1
    assert "n_cameras_used=3" in saved_lines[0]
    assert "n_cameras_used=0" not in saved_lines[0]


def test_handle_ready_to_capture_zeus_primary_no_winner_leaves_rollup_null(tmp_path):
    """Negative-path E2E confirmation: the existing no-real-sub-engine-
    winner scenario (trivial frames, no stubbing) already used by
    test_package_schema_v2_result_fields.py -- primary_engine_diagnostics
    IS threaded through (Zeus's own real 0/4 diagnostics), but since
    there's no winning_engine at all, the rollup correctly stays at its
    honest "not populated" defaults, exactly as before this task --
    `cameras_used` as `[]` (2026-08-27 null-vs-[] fix, list-typed field),
    `triangulation`/`n_cameras_used` unchanged (neither is list-typed)."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(primary="Zeus", also_run=(), timeout_s=5.0)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess-e2e-null",
        engine_config_store=store,
    background_save=False,
    )

    data = json.loads((dest_dir / "result.json").read_text())
    assert data["agreement"] == "0/4"
    assert data["cameras_used"] == []
    assert data["n_cameras_used"] == 0
    assert data["triangulation"] is None
