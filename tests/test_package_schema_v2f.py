"""Tests for the v2 package schema's final round.

Covers two parts:

Part 1 -- the null-vs-[] rule ("a collection with no members -> [],
never null, never absent"). QA's own 2 known fields
(`other_engines.Apollo.diagnostics.alt_candidates_used`,
`capture_diagnostics.ad_ws_buffer_at_capture.last_clear_buffer_summary`)
plus every other collection-typed field found by auditing the
UNCOVERED paths QA explicitly named: an abstained engine (`ok=False`),
a null sector (bull/outer_bull/outside), an off-board dart, fewer than
3 cameras in `frame_cameras`, a `generation != 0`, and a non-null
`outlier_camera`. Real additional finds beyond the 2 known fields:
`result.json`'s top-level `cameras_used` (same bug, one level up from
`other_engines.*.diagnostics.cameras_used`) and
`result.json`'s `triangulation.per_ray_distance_mm` (only reachable via
`opendarts.triangulation.rays.triangulate()`'s own `_empty_result()` path
-- a singular/degenerate ray system, real but rare).

Part 2 -- the schema flip, same commit: `meta.schema = "dart-package/v2"`
(new field, both files) and `ad_ground_truth.schema` bumped
`"ad-ground-truth-v1"` -> `"ad-ground-truth-v2"` (replaces the existing
literal, held back through three prior rounds specifically so it could
land only once opendarts's own real gaps were fixed -- see
opendarts.live.ad_ground_truth.AD_GROUND_TRUTH_SCHEMA_V2's own module
comment). Readers must handle old, interim (new key/old schema — a real
on-disk state from the round between the key rename and this flip), and
new packages, all three -- REPLAY, per docs/DESIGN.md's "Replay is the source of truth".

Deliberately does NOT re-test the base save/load round trip, throw_number/
frame_cameras, camera_mode/label/agreement, or the section-2e rollup/
generation/reason fixes -- tests/test_package_schema_v2*.py already own
those.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from opendarts.capture.throw_package import (
    META_SCHEMA_V2,
    load_throw_package,
    save_throw_package,
    validate_throw_package_meta_v2,
)
from opendarts.engines.apollo.engine import score_result_to_engine_result
from opendarts.live.ad_ground_truth import (
    AD_GROUND_TRUTH_SCHEMA_V2,
    AdGroundTruth,
    validate_ad_ground_truth_v2,
    _resolve_ad_ground_truth_captured_at_utc,
)
from opendarts.live.ad_ws_listener import AdWsListener
from opendarts.pipeline import CameraCalibration, ScoreResult
from opendarts.triangulation.rays import triangulate, Ray

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


# ---------------------------------------------------------------------------
# Part 1a. The 2 known null-vs-[] fields.
# ---------------------------------------------------------------------------

def test_alt_candidates_used_is_empty_list_not_null_when_no_alt_used():
    """QA's own named field: other_engines.Apollo.diagnostics.
    alt_candidates_used was null 90/90 real packages -- Apollo's own
    winning combination never needed an alt candidate on any of them."""
    result = ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=3, cameras_used=(0, 1, 2),
        max_ray_disagreement_mm=0.5,
        alt_candidates_used=None, # the real "never needed one" state
    )
    engine_result = score_result_to_engine_result(result)
    assert engine_result.diagnostics["alt_candidates_used"] == []
    assert engine_result.diagnostics["alt_candidates_used"] is not None


def test_alt_candidates_used_carries_real_value_when_present():
    """Not just always-[] -- confirms the fix doesn't clobber a REAL
    alt-candidate list either."""
    result = ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=3, cameras_used=(0, 1, 2),
        max_ray_disagreement_mm=0.5,
        alt_candidates_used=(1,),
    )
    engine_result = score_result_to_engine_result(result)
    assert engine_result.diagnostics["alt_candidates_used"] == [1]


def test_last_clear_buffer_summary_is_empty_list_not_null_before_any_clear():
    """QA's own second named field: capture_diagnostics.
    ad_ws_buffer_at_capture.last_clear_buffer_summary. Before this fix, a
    listener that had never seen a "Takeout finished"/numThrows-reset
    event wrote null here -- now [], and `last_clear_reason` (still None)
    is what actually distinguishes "never cleared" from "cleared, and it
    was empty"."""
    listener = AdWsListener(base_url="http://fake:3180")
    snap = listener.diagnostics_snapshot()
    assert snap["last_clear_buffer_summary"] == []
    assert snap["last_clear_reason"] is None # the real "never cleared" signal
    assert snap["buffered_events"] == [] # already correct before this fix


# ---------------------------------------------------------------------------
# Part 1b. Real additional finds -- same bug class, found by auditing
# QA's own named uncovered paths.
# ---------------------------------------------------------------------------

def test_top_level_cameras_used_is_empty_list_when_engine_abstained(pkg_dir):
    """Uncovered path QA explicitly named: an engine that abstained
    (ok=False). Before this fix, result.json's top-level `cameras_used`
    was null whenever ScoreResult.cameras_used was None (triangulation
    never attempted) -- the exact same bug as alt_candidates_used, one
    level up, just never exercised by the happy-path corpus QA measured
    against."""
    bg = {i: _marker_image(i, 10 + i) for i in range(2)}
    dart = {i: _marker_image(i, 20 + i) for i in range(2)}
    calibs = {i: _calib(30 + i) for i in range(2)}
    abstained = ScoreResult(
        ok=False, sector=None, ring=None, board_xy_mm=None, triangulation=None,
        n_cameras_used=0, cameras_used=None, reason="only 1 camera had a tip detection",
    )
    save_throw_package(pkg_dir, "sess-abstain", bg, dart, calibs, abstained)
    result = json.loads((pkg_dir / "result.json").read_text())
    assert result["ok"] is False
    assert result["cameras_used"] == []
    assert result["cameras_used"] is not None


def test_top_level_cameras_used_still_null_free_when_real_value_present(pkg_dir):
    """Not just always-[] -- a real, non-empty cameras_used still writes
    through unchanged."""
    bg = {i: _marker_image(i, 40 + i) for i in range(3)}
    dart = {i: _marker_image(i, 50 + i) for i in range(3)}
    calibs = {i: _calib(60 + i) for i in range(3)}
    result = ScoreResult(
        ok=True, sector="5", ring="single_outer", board_xy_mm=(3.0, 4.0),
        triangulation=None, n_cameras_used=3, cameras_used=(0, 1, 2),
    )
    save_throw_package(pkg_dir, "sess-real-cameras-used", bg, dart, calibs, result)
    out = json.loads((pkg_dir / "result.json").read_text())
    assert out["cameras_used"] == [0, 1, 2]


def test_triangulation_per_ray_distance_mm_empty_list_on_degenerate_rays(pkg_dir):
    """Real, reachable second-order case: rays.triangulate()'s own
    `_empty_result()` path (singular/degenerate ray system) returns a
    TriangulationResult that is NOT None overall (ok=False) but has every
    inner field, including per_ray_distance_mm, at None -- previously
    produced a null LIST nested inside an already non-null `triangulation`
    object on disk, exactly the "schema parity, no data" pattern QA's
    rule targets. Constructed here via two literally-identical (parallel,
    same origin) rays, which is a real, if rare, degenerate geometry
    opendarts's own camera rig can produce (near-collinear ray pairs)."""
    origin = np.array([0.0, 0.0, 500.0])
    direction = np.array([0.0, 0.0, -1.0])
    tri = triangulate([Ray(origin=origin, direction=direction), Ray(origin=origin, direction=direction)])
    assert tri.ok is False
    assert tri.per_ray_distance_mm is None # confirms the degenerate path really is reached

    bg = {i: _marker_image(i, 70 + i) for i in range(2)}
    dart = {i: _marker_image(i, 80 + i) for i in range(2)}
    calibs = {i: _calib(90 + i) for i in range(2)}
    result = ScoreResult(
        ok=False, sector=None, ring=None, board_xy_mm=None, triangulation=tri,
        n_cameras_used=0, reason="degenerate ray geometry",
    )
    save_throw_package(pkg_dir, "sess-degenerate-tri", bg, dart, calibs, result)
    out = json.loads((pkg_dir / "result.json").read_text())
    assert out["triangulation"] is not None # the outer object IS written
    assert out["triangulation"]["per_ray_distance_mm"] == []
    assert out["triangulation"]["per_ray_distance_mm"] is not None


def test_triangulation_per_ray_distance_mm_real_value_when_present(pkg_dir):
    """Not just always-[] -- a real, successful triangulation's
    per_ray_distance_mm still writes through as a real, non-empty list."""
    origin0 = np.array([-100.0, 0.0, 500.0])
    origin1 = np.array([100.0, 0.0, 500.0])
    direction0 = np.array([0.3, 0.0, -1.0])
    direction1 = np.array([-0.3, 0.0, -1.0])
    tri = triangulate([Ray(origin=origin0, direction=direction0), Ray(origin=origin1, direction=direction1)])
    assert tri.ok is True
    assert tri.per_ray_distance_mm is not None and len(tri.per_ray_distance_mm) == 2

    bg = {i: _marker_image(i, 100 + i) for i in range(2)}
    dart = {i: _marker_image(i, 110 + i) for i in range(2)}
    calibs = {i: _calib(120 + i) for i in range(2)}
    result = ScoreResult(
        ok=True, sector="1", ring="single_inner", board_xy_mm=tri.board_plane_xy,
        triangulation=tri, n_cameras_used=2, cameras_used=(0, 1),
    )
    save_throw_package(pkg_dir, "sess-real-tri", bg, dart, calibs, result)
    out = json.loads((pkg_dir / "result.json").read_text())
    assert len(out["triangulation"]["per_ray_distance_mm"]) == 2


# ---------------------------------------------------------------------------
# Part 1c. Other named uncovered paths -- confirmed already-correct, no
# null-vs-[] regression, as part of the same audit pass.
# ---------------------------------------------------------------------------

def test_null_sector_and_off_board_dart_do_not_disturb_list_fields(pkg_dir):
    """bull/outer_bull/outside all have sector=None -- confirms this
    doesn't somehow leak into any list-typed field, and that `label`
    still resolves ("OUT" for off-board, per the v2 package schema's
    own "already correct" ruling -- re-confirmed here, not re-fixed)."""
    bg = {i: _marker_image(i, 130 + i) for i in range(3)}
    dart = {i: _marker_image(i, 140 + i) for i in range(3)}
    calibs = {i: _calib(150 + i) for i in range(3)}
    off_board = ScoreResult(
        ok=True, sector=None, ring="outside", board_xy_mm=(300.0, 300.0),
        triangulation=None, n_cameras_used=3, cameras_used=(0, 1, 2),
    )
    save_throw_package(pkg_dir, "sess-off-board", bg, dart, calibs, off_board)
    out = json.loads((pkg_dir / "result.json").read_text())
    assert out["sector"] is None
    assert out["label"] == "OUT"
    assert out["cameras_used"] == [0, 1, 2]


def test_generation_nonzero_and_outlier_camera_write_through_correctly(pkg_dir):
    """generation != 0 and a non-null outlier_camera -- both scalars, both
    already handled by pre-existing code (this round doesn't touch
    either), confirmed here as part of this task's own audit rather than
    assumed."""
    bg = {i: _marker_image(i, 160 + i) for i in range(3)}
    dart = {i: _marker_image(i, 170 + i) for i in range(3)}
    calibs = {i: _calib(180 + i) for i in range(3)}
    result = ScoreResult(
        ok=True, sector="7", ring="double", board_xy_mm=(5.0, 6.0),
        triangulation=None, n_cameras_used=2, cameras_used=(0, 1), outlier_camera=2,
    )
    save_throw_package(pkg_dir, "sess-gen5", bg, dart, calibs, result, generation=5)
    meta = json.loads((pkg_dir / "meta.json").read_text())
    out = json.loads((pkg_dir / "result.json").read_text())
    assert meta["generation"] == 5
    assert out["outlier_camera"] == 2


def test_fewer_than_three_cameras_frame_cameras_still_a_real_list(pkg_dir):
    """<3 cameras -- frame_cameras was already correct before this round
    (docs/DESIGN.md's own note), re-confirmed here with a single camera."""
    bg = {0: _marker_image(0, 190)}
    dart = {0: _marker_image(0, 191)}
    calibs = {0: _calib(192)}
    result = ScoreResult(
        ok=True, sector="1", ring="single_inner", board_xy_mm=(1.0, 1.0),
        triangulation=None, n_cameras_used=1, cameras_used=(0,),
    )
    save_throw_package(pkg_dir, "sess-one-cam", bg, dart, calibs, result)
    meta = json.loads((pkg_dir / "meta.json").read_text())
    assert meta["frame_cameras"] == [0]
    assert isinstance(meta["frame_cameras"], list)


# ---------------------------------------------------------------------------
# Part 2a. meta.schema -- new field, both files, written unconditionally.
# ---------------------------------------------------------------------------

def test_save_throw_package_writes_meta_schema(pkg_dir):
    bg = {i: _marker_image(i, 200 + i) for i in range(3)}
    dart = {i: _marker_image(i, 210 + i) for i in range(3)}
    calibs = {i: _calib(220 + i) for i in range(3)}
    result = ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=3,
    )
    save_throw_package(pkg_dir, "sess-schema", bg, dart, calibs, result)
    meta = json.loads((pkg_dir / "meta.json").read_text())
    assert meta["schema"] == META_SCHEMA_V2 == "dart-package/v2"
    validate_throw_package_meta_v2(meta) # the real CI validator, must not raise

    pkg = load_throw_package(pkg_dir)
    assert pkg.schema == META_SCHEMA_V2


def test_validate_throw_package_meta_v2_rejects_missing_schema():
    meta = {
        "session": "s", "cameras": [0], "captured_at_utc": "now",
        "frame_cameras": [0],
    }
    with pytest.raises(AssertionError):
        validate_throw_package_meta_v2(meta)


def test_validate_throw_package_meta_v2_rejects_pre_flip_schema_value():
    """The flip is a real value change, not just presence -- a meta.json
    that (somehow) wrote the wrong literal must still fail the
    validator, not just "has a schema key at all"."""
    meta = {
        "session": "s", "cameras": [0], "captured_at_utc": "now",
        "frame_cameras": [0], "schema": "some-other-version",
    }
    with pytest.raises(AssertionError):
        validate_throw_package_meta_v2(meta)


def test_load_throw_package_old_meta_with_no_schema_key_still_loads(pkg_dir):
    """REPLAY, per docs/DESIGN.md's "Replay is the source of truth": every package saved before
    this field existed -- every real package predating this round --
    has no `schema` key in meta.json at all -- must still load fine,
    schema honestly None, never fabricated."""
    bg = {i: _marker_image(i, 230 + i) for i in range(3)}
    dart = {i: _marker_image(i, 240 + i) for i in range(3)}
    calibs = {i: _calib(250 + i) for i in range(3)}
    result = ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=3,
    )
    save_throw_package(pkg_dir, "sess-pre-flip", bg, dart, calibs, result)
    meta_path = pkg_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["schema"]
    meta_path.write_text(json.dumps(meta))

    pkg = load_throw_package(pkg_dir)
    assert pkg.schema is None
    assert pkg.cameras == [0, 1, 2] # everything else still loads correctly


# ---------------------------------------------------------------------------
# Part 2b. ad_ground_truth.schema -- bump the existing literal.
# ---------------------------------------------------------------------------

def _gt(**overrides) -> AdGroundTruth:
    base = dict(
        matched=True, match_reason="ok", ad_base_url="http://localhost:3180",
        fetched_at_utc="2026-08-27T00:00:00+00:00",
        opendarts_captured_at_utc="2026-08-27T00:00:00.100000+00:00",
        staleness_sec=0.1, window_sec=12.0, sector="20", ring="treble",
    )
    base.update(overrides)
    return AdGroundTruth(**base)


def test_ad_ground_truth_to_dict_writes_bumped_schema():
    d = _gt().to_dict()
    assert d["schema"] == AD_GROUND_TRUTH_SCHEMA_V2 == "ad-ground-truth-v2"
    validate_ad_ground_truth_v2(d) # must not raise


def test_ad_ground_truth_reader_handles_all_three_real_on_disk_states():
    """REPLAY -- three real states this module must keep reading
    forever: (1) old, schema v1 + old key; (2) interim (a real state
    produced by the round between the key rename and this flip) --
    schema STILL v1, key ALREADY renamed; (3) new -- schema v2 + new
    key. All three must resolve captured_at_utc/sector/ring correctly,
    regardless of what the schema string claims."""
    old = {
        "schema": "ad-ground-truth-v1", # pre-flip literal (AD_GROUND_TRUTH_SCHEMA_CURRENT removed 2026-08-27, dead code)
        "matched": True, "match_reason": "ok", "ad_base_url": "http://fake",
        "fetched_at_utc": "now", "opendarts_captured_at_utc": "captured-old",
        "staleness_sec": None, "window_sec": 12.0, "sector": "5", "ring": "double",
    }
    interim = {
        "schema": "ad-ground-truth-v1", # still old schema...
        "matched": True, "match_reason": "ok", "ad_base_url": "http://fake",
        "fetched_at_utc": "now", "captured_at_utc": "captured-interim", # ...new key
        "staleness_sec": None, "window_sec": 12.0, "sector": "6", "ring": "treble",
    }
    new = {
        "schema": AD_GROUND_TRUTH_SCHEMA_V2,
        "matched": True, "match_reason": "ok", "ad_base_url": "http://fake",
        "fetched_at_utc": "now", "captured_at_utc": "captured-new",
        "staleness_sec": None, "window_sec": 12.0, "sector": "7", "ring": "single_outer",
    }

    gt_old = AdGroundTruth.from_dict(old)
    gt_interim = AdGroundTruth.from_dict(interim)
    gt_new = AdGroundTruth.from_dict(new)

    assert gt_old.opendarts_captured_at_utc == "captured-old"
    assert gt_interim.opendarts_captured_at_utc == "captured-interim"
    assert gt_new.opendarts_captured_at_utc == "captured-new"
    assert (gt_old.sector, gt_old.ring) == ("5", "double")
    assert (gt_interim.sector, gt_interim.ring) == ("6", "treble")
    assert (gt_new.sector, gt_new.ring) == ("7", "single_outer")

    # Same guarantee via the resolver function directly, and via a real
    # save_ad_ground_truth()/load_ad_ground_truth() round trip for the
    # state that actually matters going forward (new).
    assert _resolve_ad_ground_truth_captured_at_utc(interim) == "captured-interim"


def test_validate_ad_ground_truth_v2_rejects_pre_flip_schema_even_with_new_key():
    """The interim state is real and must keep LOADING (see test above),
    but the validator -- which only ever runs against a FRESH write, per
    its own docstring -- correctly rejects it: a fresh write is never
    supposed to produce the interim shape any more."""
    interim = {
        "schema": "ad-ground-truth-v1",
        "captured_at_utc": "now",
    }
    with pytest.raises(AssertionError):
        validate_ad_ground_truth_v2(interim)


def test_save_and_load_ad_ground_truth_round_trip_writes_bumped_schema(pkg_dir):
    """End-to-end through the real package-writing path (not just
    to_dict()/from_dict() in isolation)."""
    from opendarts.capture.throw_package import load_ad_ground_truth, save_ad_ground_truth

    pkg_dir.mkdir(parents=True, exist_ok=True)
    save_ad_ground_truth(pkg_dir, _gt(sector="9", ring="treble"))

    on_disk = json.loads((pkg_dir / "ad_ground_truth.json").read_text())
    assert on_disk["schema"] == "ad-ground-truth-v2"
    validate_ad_ground_truth_v2(on_disk)

    loaded = load_ad_ground_truth(pkg_dir)
    assert loaded is not None
    assert loaded.sector == "9"
    assert loaded.ring == "treble"
