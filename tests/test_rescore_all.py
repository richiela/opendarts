"""Tests for opendarts/capture/rescore_all.py -- the batch replay/rescoring
tool built on top of opendarts/capture/replay.py per docs/DESIGN.md's
"Replay is the source of truth".

This file builds a small, realistic package_root/<session>/<throw_id>/
tree by hand (mirroring opendarts.live.server.discover_packages()'s and
opendarts.capture.throw_package.save_throw_package()'s real on-disk layout)
with deliberately varied contents:
  - a package that replays to the SAME result it was saved with
    (unchanged)
  - a package whose ORIGINAL result was ok=False, that replays to
    ok=True under a monkeypatched "fixed" detect_tip
  - a package whose ORIGINAL result was ok=True, that replays to a
    DIFFERENT sector under a monkeypatched detect_tip (a regression /
    sector-changed case)
  - a package with NO result.json at all (never compared, fresh result
    still reported)
  - a deliberately corrupt/partial package (meta.json present, but a
    referenced PNG is missing -- simulates a concurrent live-capture
    process caught mid-write) that must be reported as a per-throw
    failure, not crash the whole batch

Same synthetic-camera approach as tests/test_capture_replay.py (ground-
truth cameras + project_points -- a KNOWN answer, since this file is
about the BATCH PLUMBING, not detection accuracy).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import opendarts.engines.apollo.engine as apollo_engine_module
from tests.support.synthetic import make_camera_matrix, make_ring_camera
from opendarts.capture.rescore_all import rescore_all, write_json_summary
from opendarts.capture.throw_package import save_throw_package
from opendarts.engines.apollo.scoring import score_dart
from opendarts.engines.apollo.tip_detection import TipDetectionResult
from opendarts.geometry.board import (
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    polar_to_xy_mm,
    sector_center_angle_deg,
)
from opendarts.pipeline import CameraCalibration

@pytest.fixture()
def package_root(tmp_path):
    root = tmp_path / "packages"
    root.mkdir(parents=True, exist_ok=True)
    return root


# --------------------------------------------------------------------------
# Synthetic-scene helpers -- same pattern as tests/test_capture_replay.py.
# --------------------------------------------------------------------------

def _synthetic_rig(n_cameras: int = 3, fov_deg: float = 90.0):
    camera_matrix = make_camera_matrix(fov_deg=fov_deg)
    true_cams = [
        make_ring_camera(i, n_cameras=n_cameras, camera_matrix=camera_matrix)
        for i in range(n_cameras)
    ]
    calibrations = {
        i: CameraCalibration(
            camera_matrix=cam.camera_matrix,
            dist_coeffs=cam.dist_coeffs,
            rvec=cam.rvec,
            tvec=cam.tvec,
            pnp_result=None,
            landmark_spread_ok=True,
        )
        for i, cam in enumerate(true_cams)
    }
    return true_cams, calibrations


def _board_point_mm(sector_number: int, ring: str) -> tuple[float, float, float]:
    angle = sector_center_angle_deg(sector_number)
    if ring == "treble":
        radius = (TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2
    elif ring == "double_outer":
        # Mid-band, not edge-of-ring -- keeps a >=4mm margin from either
        # ring boundary regardless of engine-path noise, robust without
        # depending on the exact edge radius (a real double-ring sector
        # is what this test actually verifies, not the precise radius).
        radius = (DOUBLE_INNER_RADIUS_MM + DOUBLE_OUTER_RADIUS_MM) / 2
    else:
        raise ValueError(ring)
    x, y = polar_to_xy_mm(radius, angle)
    return (x, y, 0.0)


def _project_to_pixels(true_cams, point_xyz) -> dict[int, tuple[float, float]]:
    import cv2

    pt = np.asarray(point_xyz, dtype=np.float64).reshape(1, 1, 3)
    out = {}
    for i, cam in enumerate(true_cams):
        px, _ = cv2.projectPoints(pt, cam.rvec, cam.tvec, cam.camera_matrix, cam.dist_coeffs)
        out[i] = tuple(px.reshape(2))
    return out


def _marker_image(cam_idx: int, seed: int, h: int = 24, w: int = 32) -> np.ndarray:
    """Deterministic BGR image whose (0,0) pixel encodes `cam_idx` -- lets
    a monkeypatched detect_tip identify which camera it's being called
    for (real detect_tip signature takes only (bg_bgr, frame_bgr))."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img[0, 0, 0] = cam_idx
    return img


def _save_package(dest_dir: Path, session: str, seed_base: int, n_cameras, calibrations, result) -> None:
    bg_frames = {i: _marker_image(i, seed=seed_base + i) for i in range(n_cameras)}
    dart_frames = {i: _marker_image(i, seed=seed_base + 100 + i) for i in range(n_cameras)}
    save_throw_package(dest_dir, session, bg_frames, dart_frames, calibrations, result)


# --------------------------------------------------------------------------
# Build the full mixed package_root the rest of this file exercises.
# --------------------------------------------------------------------------

def _build_mixed_root(package_root: Path):
    """Returns (true_cams, calibrations, pixels_by_sector) so tests can
    assert against known-correct values, plus the throw_ids used so
    assertions can be keyed by name instead of position."""
    true_cams, calibrations = _synthetic_rig(n_cameras=3)

    pixels_s20_treble = _project_to_pixels(true_cams, _board_point_mm(20, "treble"))
    pixels_s3_double = _project_to_pixels(true_cams, _board_point_mm(3, "double_outer"))

    session = "session-mixed"

    # 1. UNCHANGED: originally ok=True at S20 treble, replay (unpatched,
    # real detect_tip on featureless marker images -- see below) will
    # be monkeypatched globally per-test, so this package's own
    # "unchanged" property is proven by patching detect_tip to return
    # exactly the pixels it was saved with.
    result_unchanged = score_dart(pixels_s20_treble, calibrations)
    assert result_unchanged.ok, result_unchanged.reason
    _save_package(
        package_root / session / "throw-unchanged", session, seed_base=10,
        n_cameras=3, calibrations=calibrations, result=result_unchanged,
    )

    # 2. NEWLY OK: originally ok=False (only 1 camera -- below score_dart's
    # minimum), will replay with all 3 cameras' tips available -> ok=True.
    result_originally_rejected = score_dart({0: pixels_s20_treble[0]}, {0: calibrations[0]})
    assert not result_originally_rejected.ok
    _save_package(
        package_root / session / "throw-newly-ok", session, seed_base=20,
        n_cameras=3, calibrations=calibrations, result=result_originally_rejected,
    )

    # 3. REGRESSED / SECTOR CHANGED: originally ok=True at S20 treble,
    # will replay to S3 double (a monkeypatched detect_tip reports the
    # S3 pixels for this package's marker images specifically).
    result_originally_ok = score_dart(pixels_s20_treble, calibrations)
    assert result_originally_ok.ok and result_originally_ok.sector == "20"
    _save_package(
        package_root / session / "throw-sector-changed", session, seed_base=30,
        n_cameras=3, calibrations=calibrations, result=result_originally_ok,
    )

    # 4. NO ORIGINAL RESULT: save normally, then delete result.json.
    result_for_noresult = score_dart(pixels_s20_treble, calibrations)
    assert result_for_noresult.ok
    _save_package(
        package_root / session / "throw-no-result", session, seed_base=40,
        n_cameras=3, calibrations=calibrations, result=result_for_noresult,
    )
    (package_root / session / "throw-no-result" / "result.json").unlink()

    # 5. CORRUPT/PARTIAL: a real package, then delete one camera's clip
    # afterward -- simulates a concurrent writer caught mid-write
    # (meta.json says 3 cameras, but cam1's frames don't exist).
    result_for_corrupt = score_dart(pixels_s20_treble, calibrations)
    assert result_for_corrupt.ok
    _save_package(
        package_root / session / "throw-corrupt", session, seed_base=50,
        n_cameras=3, calibrations=calibrations, result=result_for_corrupt,
    )
    (package_root / session / "throw-corrupt" / "stills_cam1.mkv").unlink()

    return true_cams, calibrations, pixels_s20_treble, pixels_s3_double


def _install_scenario_detect_tip(monkeypatch, pixels_s20_treble, pixels_s3_double):
    """One monkeypatched detect_tip that behaves differently depending on
    which package's marker images it's given, keyed by the seed baked
    into _marker_image's (0,0) pixel via the frame's cam index AND a
    second marker byte encoding which scenario this frame belongs to.
    Simpler alternative used here: key off the calling package by
    inspecting img[0, 0, 1] (a scenario id we stamp into dart_frames
    below) instead of layering that complexity into _marker_image --
    keeps _save_package/_marker_image reusable as-is.
    """

    def fake_detect_tip(bg_bgr, frame_bgr, prior_dart_line_px=None):
        cam = int(frame_bgr[0, 0, 0])
        scenario = int(frame_bgr[0, 0, 1])
        if scenario == SCENARIO_SECTOR_CHANGED:
            return TipDetectionResult(ok=True, tip_px=pixels_s3_double[cam], reason="scenario-sector-changed")
        # Default (unchanged, newly-ok, no-result, corrupt): report the
        # SAME S20-treble pixels the package was originally scored from.
        return TipDetectionResult(ok=True, tip_px=pixels_s20_treble[cam], reason="scenario-default")

    monkeypatch.setattr(apollo_engine_module, "detect_tip", fake_detect_tip)


SCENARIO_SECTOR_CHANGED = 99


def _stamp_scenario(pkg_dir: Path, scenario: int) -> None:
    """Stamp `scenario` into every dart frame of a saved package, in place.

    The frames live in the package's per-camera stills clip (no PNGs since
    2026-09-22), so this reads both frames back, marks the dart frame, and
    rewrites the clips -- the same writer save_throw_package() uses, so
    the package stays a package that writer could have produced."""
    from opendarts.capture import clip

    video = json.loads((pkg_dir / "meta.json").read_text())["video"]
    bg, dart = {}, {}
    for key in video["cameras"]:
        b, d = clip.read_bg_and_commit_frames(pkg_dir, video, int(key))
        d = d.copy()
        d[0, 0, 1] = scenario
        bg[int(key)], dart[int(key)] = b, d
    assert clip.write_still_clips(pkg_dir, bg, dart)["cameras"] == video["cameras"]


def test_rescore_all_end_to_end_summary_reflects_reality(package_root, monkeypatch):
    true_cams, calibrations, pixels_s20_treble, pixels_s3_double = _build_mixed_root(package_root)

    # Stamp a scenario marker into the sector-changed package's dart
    # frames so the shared fake_detect_tip below can tell it apart from
    # the other packages' otherwise-identical marker images.
    corrupt_and_changed_dir = package_root / "session-mixed" / "throw-sector-changed"
    _stamp_scenario(corrupt_and_changed_dir, SCENARIO_SECTOR_CHANGED)

    _install_scenario_detect_tip(monkeypatch, pixels_s20_treble, pixels_s3_double)

    # Pinned to Apollo deliberately: these tests drive scoring by
    # patching Apollo's own detect_tip, so the engine under replay must
    # be Apollo. The default primary is Zeus (a combiner over all four
    # detectors), which would not be controlled by that patch.
    summary = rescore_all(package_root, engine="Apollo")

    # -- total accounting: 5 packages saved, all 5 discoverable (corrupt
    # one still has a valid meta.json -- only a PNG is missing, so
    # discover_packages() itself still lists it; the load failure
    # happens one layer deeper, inside this module). --------------
    assert summary.total_found == 5

    by_throw = {o.throw_id: o for o in summary.outcomes}
    assert set(by_throw) == {
        "throw-unchanged", "throw-newly-ok", "throw-sector-changed",
        "throw-no-result", "throw-corrupt",
    }

    # -- unchanged --------------------------------------------------
    unchanged = by_throw["throw-unchanged"]
    assert unchanged.status == "scored"
    assert unchanged.had_original_result
    assert unchanged.original_sector == "20" and unchanged.fresh_sector == "20"
    assert unchanged.ok_changed is False
    assert unchanged.sector_changed is False
    assert unchanged.ring_changed is False
    assert unchanged.anything_changed is False
    assert unchanged in summary.unchanged
    assert unchanged not in summary.changed

    # -- newly ok -----------------------------------------------------
    newly_ok = by_throw["throw-newly-ok"]
    assert newly_ok.status == "scored"
    assert newly_ok.original_ok is False
    assert newly_ok.original_sector is None # the rejected live result never had a sector at all
    assert newly_ok.fresh_ok is True
    assert newly_ok.fresh_sector == "20"
    assert newly_ok.newly_ok is True
    assert newly_ok.regressed is False
    # sector/ring both count as "changed" here too: None -> "20"/"treble"
    # is a real, reportable change (rejected live -> now has an actual
    # sector at all), not just the ok-flag flipping in isolation.
    assert newly_ok.sector_changed is True
    assert newly_ok.ring_changed is True
    assert newly_ok in summary.newly_ok
    assert newly_ok in summary.changed
    assert newly_ok in summary.sector_changed
    assert newly_ok in summary.ring_changed

    # -- sector changed / regressed to a different sector -------------
    sector_changed = by_throw["throw-sector-changed"]
    assert sector_changed.status == "scored"
    assert sector_changed.original_sector == "20"
    assert sector_changed.fresh_sector == "3"
    assert sector_changed.sector_changed is True
    assert sector_changed.ring_changed is True # treble -> double_outer
    assert sector_changed.board_xy_changed_mm is not None
    assert sector_changed.board_xy_changed_mm > 10.0 # genuinely far apart, not float noise
    assert sector_changed in summary.sector_changed
    assert sector_changed in summary.ring_changed
    assert sector_changed in summary.changed

    # -- no original result --------------------------------------------
    no_result = by_throw["throw-no-result"]
    assert no_result.status == "scored"
    assert no_result.had_original_result is False
    assert no_result.original_ok is None
    assert no_result.fresh_ok is True # fresh result still reported
    assert no_result.fresh_sector == "20"
    assert no_result.ok_changed is None
    assert no_result.sector_changed is None
    assert no_result.anything_changed is False # can't be "changed" with nothing to compare
    assert no_result in summary.no_original_result
    assert no_result not in summary.changed
    assert no_result not in summary.unchanged # unchanged also requires had_original_result

    # -- corrupt/partial package: reported as a failure, not silently
    # dropped and not a crash of the whole batch -----------------
    corrupt = by_throw["throw-corrupt"]
    assert corrupt.status == "load_failed"
    assert corrupt.error is not None
    assert "camera 1" in corrupt.error # the actual missing camera, not a generic message
    assert corrupt in summary.load_failed

    # -- top-level counts -----------------------------------------------
    assert len(summary.changed) == 2 # newly-ok + sector-changed
    assert len(summary.unchanged) == 1
    assert len(summary.newly_ok) == 1
    assert len(summary.regressed) == 0
    # Both changed throws show a sector/ring change: throw-newly-ok goes
    # None -> "20"/"treble" (rejected live never had a sector at all),
    # throw-sector-changed goes "20"/"treble" -> "3"/"double_outer".
    assert len(summary.sector_changed) == 2
    assert len(summary.ring_changed) == 2
    assert len(summary.no_original_result) == 1
    assert len(summary.load_failed) == 1
    assert len(summary.replay_failed) == 0


def test_rescore_all_empty_root_returns_empty_summary(package_root):
    summary = rescore_all(package_root, engine="Apollo")
    assert summary.total_found == 0
    assert summary.outcomes == []
    assert summary.changed == []
    assert summary.load_failed == []


def test_rescore_all_nonexistent_root_does_not_crash(package_root):
    missing = package_root / "does-not-exist"
    summary = rescore_all(missing)
    assert summary.total_found == 0


def test_rescore_all_replay_failed_is_distinguished_from_load_failed(package_root, monkeypatch):
    """A package that LOADS fine (all files present/valid) but whose
    replay itself raises must be reported as status="replay_failed", not
    conflated with a load failure -- these are different operator-facing
    problems (one is a data/capture issue, the other is a pipeline-code
    bug triggered by real stored inputs)."""
    true_cams, calibrations = _synthetic_rig(n_cameras=3)
    pixels = _project_to_pixels(true_cams, _board_point_mm(20, "treble"))
    result = score_dart(pixels, calibrations)
    assert result.ok

    _save_package(
        package_root / "session-x" / "throw-blows-up", "session-x", seed_base=70,
        n_cameras=3, calibrations=calibrations, result=result,
    )

    def exploding_detect_tip(bg_bgr, frame_bgr, prior_dart_line_px=None):
        raise RuntimeError("simulated pipeline bug during replay")

    monkeypatch.setattr(apollo_engine_module, "detect_tip", exploding_detect_tip)

    summary = rescore_all(package_root, engine="Apollo")
    assert summary.total_found == 1
    outcome = summary.outcomes[0]
    assert outcome.status == "replay_failed"
    assert "simulated pipeline bug" in outcome.error
    assert outcome in summary.replay_failed
    assert outcome not in summary.load_failed


def test_rescore_all_orders_outcomes_by_real_captured_at_utc_not_directory_order(package_root):
    """2026-08-12 real-corpus investigation (the "got worse and worse"
    live-testing report): discover_packages()'s own
    order is filesystem/glob order, not chronological -- confirmed by
    direct inspection before this fix existed. Build three packages whose
    directory names sort OPPOSITE their real capture time, and assert
    rescore_all() returns them in TRUE time order regardless."""
    true_cams, calibrations = _synthetic_rig(n_cameras=3)
    pixels = _project_to_pixels(true_cams, _board_point_mm(20, "treble"))
    result = score_dart(pixels, calibrations)
    assert result.ok

    # Directory names deliberately sort Z, M, A -- the REVERSE of their
    # intended real-time order (A=earliest, Z=latest) -- so a test that
    # accidentally still relies on directory/glob order would fail.
    throw_dirs = ["throw-z-latest", "throw-m-middle", "throw-a-earliest"]
    stamps = {
        "throw-z-latest": "2026-08-12T22:37:49.000000+00:00",
        "throw-m-middle": "2026-08-12T21:10:22.000000+00:00",
        "throw-a-earliest": "2026-08-12T18:50:49.000000+00:00",
    }
    for i, name in enumerate(throw_dirs):
        dest = package_root / "session-chrono" / name
        _save_package(dest, "session-chrono", seed_base=200 + i * 10,
                      n_cameras=3, calibrations=calibrations, result=result)
        meta_path = dest / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["captured_at_utc"] = stamps[name]
        meta_path.write_text(json.dumps(meta))

    summary = rescore_all(package_root, engine="Apollo")
    assert summary.total_found == 3
    ordered_ids = [o.throw_id for o in summary.outcomes]
    assert ordered_ids == ["throw-a-earliest", "throw-m-middle", "throw-z-latest"]
    ordered_times = [o.captured_at_utc for o in summary.outcomes]
    assert ordered_times == sorted(ordered_times)


def test_rescore_all_outcome_carries_fresh_n_cameras_and_ray_disagreement(package_root, monkeypatch):
    """Real 2026-08-12 per-throw CV investigation needed exactly these two
    numbers (n_cameras_used, max_ray_disagreement_mm) per throw to
    diagnose degradation -- score_dart() already computes both, but
    RescoreOutcome silently dropped them on the floor before this fix.
    Asserted against the REAL numbers a direct score_dart() call
    produces, not just "is not None"."""
    true_cams, calibrations = _synthetic_rig(n_cameras=3)
    pixels = _project_to_pixels(true_cams, _board_point_mm(20, "treble"))
    expected = score_dart(pixels, calibrations)
    assert expected.ok

    _save_package(
        package_root / "session-y" / "throw-diag-fields", "session-y", seed_base=300,
        n_cameras=3, calibrations=calibrations, result=expected,
    )

    # detect_tip on featureless random marker images won't reproduce the
    # known pixels on its own (same reason every other scenario test in
    # this file monkeypatches it) -- report exactly the pixels the
    # package was saved from, so the fresh replay is a KNOWN, checkable
    # value, same pattern as _install_scenario_detect_tip above.
    def fake_detect_tip(bg_bgr, frame_bgr, prior_dart_line_px=None):
        cam = int(frame_bgr[0, 0, 0])
        return TipDetectionResult(ok=True, tip_px=pixels[cam], reason="fixed-pixels")

    monkeypatch.setattr(apollo_engine_module, "detect_tip", fake_detect_tip)

    summary = rescore_all(package_root, engine="Apollo")
    assert summary.total_found == 1
    outcome = summary.outcomes[0]
    assert outcome.fresh_n_cameras_used == expected.n_cameras_used == 3
    assert outcome.fresh_max_ray_disagreement_mm == pytest.approx(
        expected.max_ray_disagreement_mm, abs=1e-9
    )
    # Also round-trips through the JSON summary, not just the in-memory
    # dataclass -- this is the shape a downstream analysis script (or a
    # future rescore_all consumer) actually reads.
    d = outcome.to_json_dict()
    assert d["fresh_n_cameras_used"] == 3
    assert d["captured_at_utc"] is not None


def test_write_json_summary_round_trips_the_real_counts(package_root, monkeypatch, tmp_path):
    true_cams, calibrations, pixels_s20_treble, pixels_s3_double = _build_mixed_root(package_root)

    corrupt_and_changed_dir = package_root / "session-mixed" / "throw-sector-changed"
    _stamp_scenario(corrupt_and_changed_dir, SCENARIO_SECTOR_CHANGED)

    _install_scenario_detect_tip(monkeypatch, pixels_s20_treble, pixels_s3_double)

    summary = rescore_all(package_root, engine="Apollo")
    out_path = tmp_path / "rescore_report.json"
    write_json_summary(summary, out_path)

    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert data["totals"]["total_found"] == 5
    assert data["totals"]["changed"] == 2
    assert data["totals"]["newly_ok"] == 1
    assert data["totals"]["load_failed"] == 1
    assert len(data["throws"]) == 5
    # Spot-check one real per-throw record made it through JSON-round-trip
    # with the actual before/after values intact, not just the counts.
    throws_by_id = {t["throw_id"]: t for t in data["throws"]}
    assert throws_by_id["throw-sector-changed"]["original_sector"] == "20"
    assert throws_by_id["throw-sector-changed"]["fresh_sector"] == "3"
