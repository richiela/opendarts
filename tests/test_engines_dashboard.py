"""Dashboard tests for the multi-engine scoring framework: the
Scoring-table columns and /api/state wiring. The Config-tab controls and
the /api/engine-config endpoint were removed once the aggregate engine
became the permanent primary -- the engine set is configured in
data/config.json. See docs/ENGINES.md's "Dashboard" section.
"""
from __future__ import annotations


import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient # noqa: E402

from opendarts.capture.throw_package import save_throw_package
from opendarts.engines.base import EngineResult
from opendarts.live.capture_daemon import EngineConfigStore
from opendarts.live.server import create_app
from opendarts.pipeline import CameraCalibration, ScoreResult

@pytest.fixture()
def package_root(tmp_path):
    return tmp_path / "pkgroot"


# ---------------------------------------------------------------------------
# Dashboard HTML -- Config tab controls.
# ---------------------------------------------------------------------------


def test_config_tab_has_no_engine_config_controls(package_root):
    """The engine set is configured in data/config.json and nowhere
    else. The Config tab's controls went when the aggregate engine became
    the permanent primary ("we always run Zeus with Zeus being the
    primary"), and the runtime mutation endpoint went with them on
    2026-09-09 -- an API audit found the route had outlived its own UI and
    had no caller anywhere. /api/state still REPORTS the live engine set;
    nothing changes it at runtime."""
    html = TestClient(create_app(package_root=package_root, enable_background_poll=False)).get("/?ui=classic").text
    assert 'id="engine-config-list"' not in html
    assert 'id="engine-config-timeout"' not in html
    assert 'id="btn-save-engine-config"' not in html
    assert 'id="engine-config-status"' not in html
    assert "/api/engine-config" not in html




def test_scoring_table_thead_has_a_stable_row_id_for_dynamic_engine_columns(package_root):
    html = TestClient(create_app(package_root=package_root, enable_background_poll=False)).get("/?ui=classic").text
    assert 'id="packages-thead-row"' in html


# ---------------------------------------------------------------------------
# /api/state -- engine_config section.
# ---------------------------------------------------------------------------




def test_api_state_engine_config_reflects_a_wired_store(package_root):
    store = EngineConfigStore(primary="Apollo", also_run=("Talos",), timeout_s=4.0)
    app = create_app(package_root=package_root, enable_background_poll=False, engine_config_store=store)
    body = TestClient(app).get("/api/state").json()
    assert body["engine_config"]["primary"] == "Apollo"
    assert body["engine_config"]["also_run"] == ["Talos"]
    assert body["engine_config"]["timeout_s"] == 4.0


# ---------------------------------------------------------------------------
# /api/engine-config -- live mutation.
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# discover_packages() -- `engines`/`primary_engine` fields, factored
# match-fields helper.
# ---------------------------------------------------------------------------


def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=np.zeros(3, dtype=np.float64),
        tvec=np.array([0.0, 0.0, 1000.0], dtype=np.float64),
        landmark_spread_ok=True,
    )


def test_discover_packages_reports_engines_field_empty_by_default(package_root):
    from opendarts.live.server import discover_packages

    bg = {0: np.zeros((4, 4, 3), dtype=np.uint8)}
    frame = {0: np.ones((4, 4, 3), dtype=np.uint8)}
    result = ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=2,
    )
    save_throw_package(
        dest_dir=package_root / "sess1" / "throw1", session="sess1",
        bg_frames_bgr=bg, dart_frames_bgr=frame,
        calibrations={0: _fake_calibration()}, result=result,
    )
    packages = discover_packages(package_root)
    assert len(packages) == 1
    assert packages[0]["engines"] == {}
    assert packages[0]["primary_engine"] is None


def test_discover_packages_reports_other_engines_with_match_fields(package_root):
    from opendarts.capture.throw_package import write_other_engines_result
    from opendarts.live.ad_ground_truth import AdGroundTruth
    from opendarts.capture.throw_package import save_ad_ground_truth
    from datetime import datetime, timezone
    from opendarts.live.server import discover_packages

    bg = {0: np.zeros((4, 4, 3), dtype=np.uint8)}
    frame = {0: np.ones((4, 4, 3), dtype=np.uint8)}
    result = ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=2,
    )
    dest_dir = package_root / "sess1" / "throw1"
    save_throw_package(
        dest_dir=dest_dir, session="sess1", bg_frames_bgr=bg, dart_frames_bgr=frame,
        calibrations={0: _fake_calibration()}, result=result,
    )
    write_other_engines_result(
        dest_dir, "Apollo",
        {"Talos": EngineResult(ok=True, sector=None, ring="outside", board_xy_mm=None)},
    )
    ad_gt = AdGroundTruth(
        matched=True, match_reason="ok_ws", ad_base_url="unused",
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        opendarts_captured_at_utc=None, staleness_sec=0.1, window_sec=5.0,
        sector="20", ring="treble", tip_xy_mm=(1.0, 2.0),
    )
    save_ad_ground_truth(dest_dir, ad_gt)

    packages = discover_packages(package_root)
    pkg = packages[0]
    # Engine names as written into packages:
    # Apollo -> Apollo, Talos -> Talos in what actually lands on disk.
    assert pkg["primary_engine"] == "Apollo"
    talos = pkg["engines"]["Talos"]
    assert talos["sector"] is None
    assert talos["ring"] == "outside"
    # Talos reported a miss (sector None/ring outside) while AD's ground
    # truth is sector 20/treble -- must NOT match, computed via the SAME
    # _match_fields_for_section() the primary column uses.
    assert talos["sector_match"] is False
    # The PRIMARY's own match (sector 20/treble vs AD's 20/treble) IS a
    # match -- proves the also-run engine's mismatch isn't just a bug
    # that makes everything False.
    assert pkg["sector_match"] is True
