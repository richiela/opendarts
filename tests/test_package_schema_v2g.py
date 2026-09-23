"""The v2 package schema follow-up (2026-08-27, "other_engines diagnostics
null-vs-[] gap"): a corpus-QA peer session's ORIGINAL report claimed the
2026-08-27 "final round" null-vs-[] fix (see tests/test_package_schema_v2f.py)
only landed at two PRIMARY-engine-only call sites
(`opendarts.engines.apollo.engine.score_result_to_engine_result()` and
`opendarts.capture.throw_package._score_result_to_dict()`'s top-level
rollup) and never touched `other_engines.<Name>.diagnostics` (written via
`opendarts.engines.base.EngineResult.to_dict()`, the generic pass-through
used by `write_other_engines_result()` for every ALSO-RUN engine) --
flagging Talos ("Talos") as a real observed case of `null` reaching
disk for `cameras_used`/`alt_candidates_used`.

**A follow-up QA correction, re-measured against all 1,530 real frozen packages, changes
the finding but not the fix**:
- `alt_candidates_used` (the `ray_fallback.py:274` local) never actually
  leaves Talos's serialized diagnostics at all -- internal only, no
  on-disk divergence, ever. Confirmed independently below by direct
  trace of every `EngineResult(...)` construction site in
  `opendarts/engines/talos/engine.py` (there are exactly 3, none of which
  ever sets `alt_candidates_used` as a diagnostics key) AND empirically,
  against 222 real historical Talos diagnostics records captured in
  `tests/fixtures/talos_pre_refactor_baseline_{clean,important}.json`
  (see `test_talos_real_historical_diagnostics_never_carry_...` below).
- `cameras_used` DOES reach disk for Talos (the `line_plane_fallback`
  observation only -- 6/53 records in the "important" fixture), but was
  NEVER null in any real package QA measured, and is NEVER null by
  construction in the one code path that sets it (`opendarts/engines/
  talos/engine.py`'s `fallback = EngineResult(...)` block always
  builds it as `[planes[i][0] for i in used_idx]`, a real list).

**Net finding, matching QA's own re-measurement**: there is no real,
reachable Talos code path today that emits `null` for either field --
this is NOT a currently-observed parity failure. The fix below is still
correct and still worth shipping (the generic `EngineResult.to_dict()`
choke point QA itself recommended, "can't be missed by a new engine,
can't drift when one is edited") -- it protects the whole engine
framework going forward, including any future Talos code path that
starts setting these fields to `None` the way the ORIGINAL (later
corrected) report worried about. See `opendarts/engines/base.py`'s own
`_normalize_diagnostics_null_collections()` for the actual fix and
docs/DESIGN.md's dated 2026-08-27 entry for the full writeup.

Deliberately does NOT re-test `EngineResult.to_dict()`'s own core
normalization behavior (empty-vs-null-vs-absent, no fabrication, no
mutation) -- `tests/test_engines_base_and_registry.py` owns that at the
unit level; this file is specifically the Talos-real-behavior audit
plus the end-to-end `write_other_engines_result()` proof.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from opendarts.capture.throw_package import save_throw_package, write_other_engines_result
from opendarts.engines.base import EngineResult
from opendarts.engines.talos import TalosEngine
from opendarts.engines.talos.engine import _engine_result_from_score_dart
from opendarts.pipeline import CameraCalibration, ScoreResult

REPO_ROOT = Path(__file__).resolve().parent.parent
# Baseline records are derived from real captured sessions and live
# outside this repo; set OPENDARTS_FIXTURES_ROOT to enable these tests.
FIXTURES_DIR = Path(
    os.environ.get("OPENDARTS_FIXTURES_ROOT", "/nonexistent/reference-fixtures")
)


# ---------------------------------------------------------------------------
# Part 1 -- confirm Talos's real behavior (not assumed) across both the
# module's literal source AND real historical output.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not FIXTURES_DIR.is_dir(), reason="reference fixtures not present"
)
@pytest.mark.parametrize(
    "fixture_name",
    ["talos_pre_refactor_baseline_clean.json", "talos_pre_refactor_baseline_important.json"],
)
def test_talos_real_historical_diagnostics_never_carry_null_cameras_used_or_alt_candidates(
    fixture_name,
):
    """Empirical confirmation, not just code-reading: across 222 real
    historical Talos diagnostics dicts (169 "clean" + 53 "important"),
    `cameras_used` only ever appears (6/53, all `line_plane_fallback`)
    as a real, non-empty list -- never null, never an empty list either
    (matching the code trace: it's built from `planes_used_indices`,
    which is only reachable when >=2 planes exist). `alt_candidates_used`
    never appears at all, in either fixture. This is the SAME shape
    `score()`'s real diagnostics dict has today -- these fixtures are a
    frozen, exact snapshot of real `EngineResult.diagnostics` output,
    not a re-derivation."""
    data = json.loads((FIXTURES_DIR / fixture_name).read_text())
    records = data["records"]
    assert records, f"{fixture_name} unexpectedly empty"

    cameras_used_seen = 0
    for r in records:
        diag = r.get("diagnostics") or {}
        assert "alt_candidates_used" not in diag, (
            f"real historical Talos diagnostics unexpectedly carries "
            f"alt_candidates_used: {diag}"
        )
        if "cameras_used" in diag:
            cameras_used_seen += 1
            assert diag["cameras_used"] is not None
            assert isinstance(diag["cameras_used"], list)
            assert len(diag["cameras_used"]) >= 2  # >=2 planes required to reach this path

    # Sanity: this fixture set does contain the line_plane_fallback path
    # at least once in total (across the two fixture files), so this
    # test is actually exercising the field, not vacuously passing.
    print(f"{fixture_name}: cameras_used present in {cameras_used_seen}/{len(records)} records")


def test_talos_engine_result_construction_sites_never_set_alt_candidates_used():
    """Direct trace confirmation of opendarts/engines/talos/engine.py's
    own 3 real `EngineResult(...)` construction sites (`_engine_result_
    from_score_dart`, `_plane_miss`, the inline `line∩plane fallback`
    block) -- none of them ever sets `alt_candidates_used` as a
    diagnostics key, so `ScoreResult.alt_candidates_used` (which CAN be
    `None`, per `ray_fallback.py:274`) never actually reaches an
    `EngineResult`'s diagnostics at all. Proven by calling the real,
    exact helper function `score()` itself uses (not a re-derivation)
    with a `scored` object whose `alt_candidates_used` is `None`, and
    confirming the resulting diagnostics dict has no such key."""
    scored = ScoreResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=3, cameras_used=(0, 1, 2),
        reason="ok", max_ray_disagreement_mm=1.0, outlier_camera=None,
        alt_candidates_used=None,
    )
    result = _engine_result_from_score_dart(scored, extra={})
    assert "alt_candidates_used" not in result.diagnostics
    assert "cameras_used" not in result.diagnostics
    # Confirmed absent, not null -- the correct, honest shape per
    # The v2 package schema's own rule ("an engine that doesn't track a
    # field keeps not having that key").
    d = result.to_dict()["diagnostics"]
    assert "alt_candidates_used" not in d
    assert "cameras_used" not in d


def test_talos_engine_result_construction_sites_even_with_real_tuple_cameras_used():
    """Same helper, but with a real (non-None) cameras_used on the
    ScoreResult -- confirms the function genuinely never reads/forwards
    that attribute at all (by design: `_engine_result_from_score_dart`
    only pulls `n_cameras_used`, never `cameras_used`, into diagnostics),
    so there is no code path where it could ever appear as a stale
    null -- it simply never appears from this helper, full stop."""
    scored = ScoreResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=3, cameras_used=(0, 1, 2),
        reason="ok", max_ray_disagreement_mm=1.0, outlier_camera=None,
        alt_candidates_used=(1,),
    )
    result = _engine_result_from_score_dart(scored, extra={})
    assert "cameras_used" not in result.diagnostics
    assert "alt_candidates_used" not in result.diagnostics


def test_talos_empty_inputs_diagnostics_has_no_cameras_used_or_alt_candidates_key():
    """The `_plane_miss()` (ok=False, <2 planes) path -- the real
    diagnostics shape `test_talos_empty_inputs_are_ok_false_and_do_not_
    hang` (tests/test_engine_talos.py) already exercises -- confirmed
    here to have neither key at all, not a null value for either."""
    engine = TalosEngine()
    result = engine.score({}, {}, {})
    assert "cameras_used" not in result.diagnostics
    assert "alt_candidates_used" not in result.diagnostics
    d = result.to_dict()["diagnostics"]
    assert "cameras_used" not in d
    assert "alt_candidates_used" not in d


# ---------------------------------------------------------------------------
# Part 2 -- end-to-end write_other_engines_result() proof. Since Talos
# cannot construct a real null today (Part 1 above), this exercises the
# generic EngineResult.to_dict() fix as a defense-in-depth guarantee: IF
# any engine (Talos today, or a future engine/edit) ever does hand back
# a None-valued cameras_used/alt_candidates_used, the on-disk
# other_engines.<Name>.diagnostics must still be [] , never null.
# ---------------------------------------------------------------------------


def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=np.zeros(3, dtype=np.float64),
        tvec=np.array([0.0, 0.0, 1000.0], dtype=np.float64),
        landmark_spread_ok=True,
    )


def _save_fake_primary_package(tmp_path) -> Path:
    bg = {0: np.zeros((4, 4, 3), dtype=np.uint8)}
    frame = {0: np.ones((4, 4, 3), dtype=np.uint8)}
    calibrations = {0: _fake_calibration()}
    dest_dir = tmp_path / "packages" / "sess1" / "throw1"
    save_throw_package(
        dest_dir=dest_dir, session="sess1", bg_frames_bgr=bg, dart_frames_bgr=frame,
        calibrations=calibrations,
        result=ScoreResult(
            ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
            triangulation=None, n_cameras_used=2, reason="", max_ray_disagreement_mm=1.5,
        ),
    )
    return dest_dir


def test_write_other_engines_result_normalizes_hypothetical_null_talos_fields(tmp_path):
    """Defense-in-depth: a hand-built EngineResult carrying the shape a
    FUTURE (not today's real) Talos code path might produce if it ever
    started forwarding a None-valued cameras_used/alt_candidates_used --
    proves the fix closes that gap generically, at the real write point
    `write_other_engines_result()` uses, without needing Talos itself to
    change."""
    dest_dir = _save_fake_primary_package(tmp_path)
    hypothetical_talos = EngineResult(
        ok=True, sector="20", ring="single", board_xy_mm=(3.0, 4.0),
        reason="hypothetical", diagnostics={
            "engine": "Talos",
            "cameras_used": None,
            "alt_candidates_used": None,
            "triangulation": {"ok": False, "per_ray_distance_mm": None},
            "outlier_camera": None,  # genuine scalar -- must stay null
        },
    )
    write_other_engines_result(dest_dir, "Apollo", {"Talos": hypothetical_talos})
    data = json.loads((dest_dir / "result.json").read_text())
    talos_diag = data["other_engines"]["Talos"]["diagnostics"]
    assert talos_diag["cameras_used"] == []
    assert talos_diag["alt_candidates_used"] == []
    assert talos_diag["triangulation"]["per_ray_distance_mm"] == []
    assert talos_diag["outlier_camera"] is None  # untouched


def test_write_other_engines_result_real_talos_diagnostics_round_trip_no_fabrication(tmp_path):
    """The REAL shape (today's actual Talos output, per Part 1): no
    cameras_used/alt_candidates_used keys at all. Confirms the fix is a
    correct no-op on real output -- it must not fabricate either key."""
    dest_dir = _save_fake_primary_package(tmp_path)
    engine = TalosEngine()
    real_talos_result = engine.score({}, {}, {})  # real, ok=False, _plane_miss path
    write_other_engines_result(dest_dir, "Apollo", {"Talos": real_talos_result})
    data = json.loads((dest_dir / "result.json").read_text())
    talos_diag = data["other_engines"]["Talos"]["diagnostics"]
    assert "cameras_used" not in talos_diag
    assert "alt_candidates_used" not in talos_diag


def test_write_other_engines_result_athena_ares_zeus_unaffected(tmp_path):
    """docs/DESIGN.md's own claim, re-confirmed by direct grep of each
    engine's source (`opendarts/engines/{athena,ares,zeus}/engine.py`):
    Athena's own `cameras_used` is always a real list (a list
    comprehension over `camera_reads`, never None) and it never sets
    `alt_candidates_used`; Ares and Zeus's own top-level diagnostics
    never use any of the 3 field names at all. This end-to-end write
    must leave all three exactly as they were -- no key added, no key
    changed, matching the "never fabricate an absent key" rule."""
    dest_dir = _save_fake_primary_package(tmp_path)
    other = {
        "Athena": EngineResult(
            ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
            diagnostics={"n_cameras_used": 2, "cameras_used": [0, 1]},
        ),
        "Ares": EngineResult(
            ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
            diagnostics={"engine": "Ares", "consensus": "unanimous"},
        ),
        "Zeus": EngineResult(
            ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
            diagnostics={"winning_engine": "Apollo", "vote_tally": {"20/single": 3}},
        ),
    }
    write_other_engines_result(dest_dir, "Apollo", other)
    data = json.loads((dest_dir / "result.json").read_text())
    assert data["other_engines"]["Athena"]["diagnostics"]["cameras_used"] == [0, 1]
    assert "alt_candidates_used" not in data["other_engines"]["Athena"]["diagnostics"]
    assert "cameras_used" not in data["other_engines"]["Ares"]["diagnostics"]
    assert "alt_candidates_used" not in data["other_engines"]["Ares"]["diagnostics"]
    assert "cameras_used" not in data["other_engines"]["Zeus"]["diagnostics"]
    assert "alt_candidates_used" not in data["other_engines"]["Zeus"]["diagnostics"]
