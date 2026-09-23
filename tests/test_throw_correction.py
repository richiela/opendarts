"""Tests for POST /api/visits/{visit_id}/throws/{index}/correct -- the
visit-indexed score correction (added 2026-08-14, docs/LIVE_API.md).

The live-game-driver counterpart to the Scoring tab's own per-package
mark-ad-wrong flow. It only became addressable once the visit model
existed: without a visit ID plus
an index within it, there is no stable way for a game driver to name "the
second dart of this turn."

Three properties are load-bearing and each has a test here:
  * ONE patching mechanism, not two -- the correction writes through the
    same `ad_ground_truth.json` annotation path mark-ad-wrong uses, and
    lands in the same `operator_confirmed_*` fields
    `discover_packages()`/`_operator_truth_for()` already grade every
    engine row against;
  * the ORIGINAL ENGINE CALL SURVIVES (docs/DESIGN.md's "Replay is the source of truth") --
    `result.json` is never read or written, so replaying the package
    through newer code still reproduces and can be graded against what
    the engine actually said live;
  * a correction that could never compare equal to any engine's answer
    (a ring outside the real vocabulary, a bull with a sector, a treble
    without one) is REJECTED rather than silently recorded as an
    unmatchable truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from opendarts.capture.throw_package import (
    load_ad_ground_truth,
    save_ad_ground_truth,
    save_throw_package,
)
from opendarts.live.ad_ground_truth import AdGroundTruth
from opendarts.live.server import board_ring_names, create_app
from opendarts.pipeline import CameraCalibration, ScoreResult

VISIT = "visit_1700000000000"


@pytest.fixture()
def package_root(tmp_path):
    return tmp_path / "pkgroot"


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((5, 1)),
        rvec=np.zeros((3, 1)),
        tvec=np.zeros((3, 1)),
        pnp_result=None,
        landmark_spread_ok=True,
    )


def _write_throw(
    package_root: Path,
    *,
    visit_id: str = VISIT,
    visit_index: int = 0,
    sector: str = "1",
    ring: str = "single_outer",
) -> Path:
    """One real package via the REAL save_throw_package() -- the same
    function the live daemon calls -- so these tests exercise the actual
    on-disk format, not one invented here."""
    return save_throw_package(
        dest_dir=package_root / "session-correct" / f"throw_{visit_index}",
        session="session-correct",
        bg_frames_bgr={0: np.zeros((8, 8, 3), dtype=np.uint8)},
        dart_frames_bgr={0: np.full((8, 8, 3), 255, dtype=np.uint8)},
        calibrations={0: _calibration()},
        result=ScoreResult(
            ok=True,
            sector=sector,
            ring=ring,
            board_xy_mm=(10.0, 20.0),
            triangulation=None,
            n_cameras_used=3,
            reason="",
            max_ray_disagreement_mm=1.5,
        ),
        visit_id=visit_id,
        visit_index=visit_index,
    )


def _attach_ad(pkg_dir: Path, *, sector: str | None, ring: str | None) -> None:
    save_ad_ground_truth(
        pkg_dir,
        AdGroundTruth(
            matched=True,
            match_reason="ok",
            ad_base_url="http://localhost:3180",
            fetched_at_utc="2026-08-14T00:00:00+00:00",
            opendarts_captured_at_utc="2026-08-14T00:00:00+00:00",
            staleness_sec=1.0,
            window_sec=12.0,
            sector=sector,
            ring=ring,
            tip_xy_mm=(10.0, 20.0),
            ad_method="UnanimousCam",
        ),
    )


def _app(package_root: Path):
    return create_app(package_root=package_root, enable_background_poll=False)


def _correct(client: TestClient, index: int = 0, **body):
    payload = {"sector": "20", "ring": "treble"}
    payload.update(body)
    return client.post(f"/api/visits/{VISIT}/throws/{index}/correct", json=payload)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------

def test_correction_records_the_human_answer_as_confirmed_truth(package_root):
    pkg = _write_throw(package_root, sector="1", ring="single_outer")
    client = TestClient(_app(package_root))

    resp = _correct(client)
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["visit_id"] == VISIT
    assert body["visit_index"] == 0
    assert (body["corrected_sector"], body["corrected_ring"]) == ("20", "treble")
    # What the engine actually said is echoed back alongside it.
    assert (body["live_sector"], body["live_ring"]) == ("1", "single_outer")

    # Written through the SAME annotation fields the Scoring tab's
    # mark-ad-wrong flow uses -- one truth record per throw, not two.
    gt = load_ad_ground_truth(pkg)
    assert gt.operator_confirmed_sector == "20"
    assert gt.operator_confirmed_ring == "treble"
    assert gt.operator_confirmed_source == "manual"


def test_the_corrected_truth_is_what_engines_are_then_graded_against(package_root):
    """The actual payoff. `discover_packages()`/`_operator_truth_for()`
    already prefer a human-confirmed answer over AD's own; a correction
    must feed that same path, or it would be recorded and then ignored.

    Engine said single_outer 1, AD agreed with the engine, human says it
    was really T20 -> the throw must flip to sector_match False."""
    pkg = _write_throw(package_root, sector="1", ring="single_outer")
    _attach_ad(pkg, sector="1", ring="single_outer")
    client = TestClient(_app(package_root))

    before = client.get("/api/packages").json()[0]
    assert before["sector_match"] is True, "engine agreed with AD before the correction"

    _correct(client)

    after = client.get("/api/packages").json()[0]
    assert after["sector_match"] is False
    assert after["ad_operator_confirmed_sector"] == "20"
    assert after["ad_operator_confirmed_ring"] == "treble"


def test_original_engine_call_is_never_touched(package_root):
    """docs/DESIGN.md's "Replay is the source of truth": a correction adds the human's
    answer ALONGSIDE what the engine said, never in place of it, so
    replaying this package through newer code still reproduces and can be
    graded against the real live call. Proven byte-for-byte on
    result.json, plus the raw frames the replay actually needs."""
    pkg = _write_throw(package_root, sector="1", ring="single_outer")
    result_before = (pkg / "result.json").read_bytes()
    # The frames live in the package's clip now (no PNGs since
    # 2026-09-22); the clip file is what must stay byte-identical.
    clip_name = json.loads((pkg / "meta.json").read_text())["video"]["cameras"]["0"]["clip"]
    frame_before = (pkg / clip_name).read_bytes()
    calib_before = (pkg / "calibration.json").read_bytes()

    _correct(TestClient(_app(package_root)))

    assert (pkg / "result.json").read_bytes() == result_before
    assert (pkg / clip_name).read_bytes() == frame_before
    assert (pkg / "calibration.json").read_bytes() == calib_before
    # And the live answer is still readable, unchanged, from the listing.
    assert json.loads((pkg / "result.json").read_text())["sector"] == "1"


def test_correcting_a_throw_ad_got_right_still_records_the_correction(package_root):
    """The case that forced `record_throw_correction()` to exist rather
    than just calling `mark_operator_ad_wrong()`: that function's toggle
    contract ties the confirmation to the AD-wrong flag (`wrong=False`
    CLEARS the confirmation), which would silently discard a correction
    on exactly the throws where OUR engine was the wrong one.

    Here AD already agreed with the human, so `operator_marked_wrong`
    must be False -- an honest statement about AD -- while the
    confirmation is still recorded."""
    pkg = _write_throw(package_root, sector="1", ring="single_outer")
    _attach_ad(pkg, sector="20", ring="treble")

    _correct(TestClient(_app(package_root)))

    gt = load_ad_ground_truth(pkg)
    assert gt.operator_marked_wrong is False, "AD was right here -- don't claim otherwise"
    assert gt.operator_confirmed_sector == "20"
    assert gt.operator_confirmed_ring == "treble"


def test_correction_derives_ad_wrong_when_ad_actually_disagrees(package_root):
    pkg = _write_throw(package_root)
    _attach_ad(pkg, sector="5", ring="double")

    _correct(TestClient(_app(package_root)))

    gt = load_ad_ground_truth(pkg)
    assert gt.operator_marked_wrong is True
    # AD's own fetched answer survives untouched next to the correction.
    assert (gt.sector, gt.ring) == ("5", "double")


def test_correction_on_a_throw_with_no_ad_data_at_all(package_root):
    """AD may never have registered the throw. The annotation still needs
    somewhere durable to live -- the same placeholder path
    mark_operator_ad_wrong() already established."""
    pkg = _write_throw(package_root)
    assert load_ad_ground_truth(pkg) is None

    resp = _correct(TestClient(_app(package_root)))
    assert resp.status_code == 200

    gt = load_ad_ground_truth(pkg)
    assert gt.matched is False
    assert gt.match_reason == "operator_marked_no_ad_data"
    assert gt.operator_confirmed_ring == "treble"


def test_correcting_twice_is_idempotent_last_write_wins(package_root):
    """A correction endpoint must be safe to call twice -- a game driver
    retrying a request must not produce a
    half-written or doubled annotation."""
    pkg = _write_throw(package_root)
    client = TestClient(_app(package_root))

    _correct(client, sector="20", ring="treble")
    _correct(client, sector="5", ring="double")

    gt = load_ad_ground_truth(pkg)
    assert (gt.operator_confirmed_sector, gt.operator_confirmed_ring) == ("5", "double")


def test_bull_correction_takes_no_sector(package_root):
    """`sector_ring_for_point()`'s own return shape: bull/outer_bull/
    outside have a real ring and a legitimately-None sector."""
    pkg = _write_throw(package_root)
    client = TestClient(_app(package_root))

    resp = client.post(
        f"/api/visits/{VISIT}/throws/0/correct", json={"ring": "bull"}
    )
    assert resp.status_code == 200
    gt = load_ad_ground_truth(pkg)
    assert gt.operator_confirmed_ring == "bull"
    assert gt.operator_confirmed_sector is None


def test_source_and_note_are_recorded(package_root):
    """`source` reuses the exact vocabulary `operator_confirmed_source`
    already has -- an engine NAME, or the literal "manual" -- so a
    correction made by a game driver is attributable later."""
    pkg = _write_throw(package_root)
    client = TestClient(_app(package_root))

    client.post(
        f"/api/visits/{VISIT}/throws/0/correct",
        json={
            "sector": "20",
            "ring": "treble",
            "source": "Talos",
            "note": "operator watched it land in the treble",
        },
    )

    gt = load_ad_ground_truth(pkg)
    assert gt.operator_confirmed_source == "Talos"
    assert gt.operator_note == "operator watched it land in the treble"


# --------------------------------------------------------------------------
# Addressing
# --------------------------------------------------------------------------

def test_the_right_dart_of_the_visit_is_corrected(package_root):
    """The whole reason this is visit-indexed. Three darts in one visit;
    correcting index 1 must touch only that one."""
    pkgs = [_write_throw(package_root, visit_index=i, sector=str(i + 1)) for i in range(3)]
    client = TestClient(_app(package_root))

    resp = _correct(client, index=1)
    assert resp.status_code == 200
    assert resp.json()["throw_id"] == pkgs[1].name

    assert load_ad_ground_truth(pkgs[0]) is None
    assert load_ad_ground_truth(pkgs[1]).operator_confirmed_ring == "treble"
    assert load_ad_ground_truth(pkgs[2]) is None


def test_a_throw_from_an_already_closed_visit_is_still_correctable(package_root):
    """A real, deliberate improvement over "correct only while the visit
    is open": this resolves
    the throw off DISK (each package carries its own visit_id/visit_index
    in meta.json), so a visit that ended -- or one captured by an
    entirely different process -- is still fixable. There is no in-memory
    visit here at all: this AppState has no capture loop wired."""
    _write_throw(package_root, visit_id="visit_1600000000000", visit_index=2)
    app = _app(package_root)
    assert app.state.opendarts_state.visit_id is None, "no live visit in memory"

    resp = TestClient(app).post(
        "/api/visits/visit_1600000000000/throws/2/correct",
        json={"sector": "20", "ring": "treble"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_unknown_visit_or_index_is_a_clear_404_not_a_silent_success(package_root):
    _write_throw(package_root)
    client = TestClient(_app(package_root))

    missing_visit = client.post(
        "/api/visits/visit_9999999999999/throws/0/correct",
        json={"sector": "20", "ring": "treble"},
    )
    assert missing_visit.status_code == 404
    assert missing_visit.json()["ok"] is False

    missing_index = _correct(client, index=2)
    assert missing_index.status_code == 404
    assert "no throw found" in missing_index.json()["reason"]


# --------------------------------------------------------------------------
# Validation -- a truth nothing can match is worse than an error
# --------------------------------------------------------------------------

def test_ring_outside_the_real_vocabulary_is_rejected(package_root):
    """The correction must use the exact strings opendarts's own scorer
    produces (`board_ring_names()`, DERIVED from
    opendarts.geometry.board itself). An outside spelling -- "T20" -- is
    a good example of what must NOT be silently accepted here: recorded as
    a truth, it could never compare equal to any engine's answer, making
    that throw a permanent silent miss."""
    _write_throw(package_root)
    client = TestClient(_app(package_root))

    resp = client.post(
        f"/api/visits/{VISIT}/throws/0/correct", json={"sector": "20", "ring": "T20"}
    )
    assert resp.status_code == 400
    assert resp.json()["ok"] is False
    # The error names the real vocabulary rather than just saying "bad".
    for ring in board_ring_names():
        assert ring in resp.json()["reason"]


def test_sector_outside_the_real_wedge_numbers_is_rejected(package_root):
    _write_throw(package_root)
    resp = TestClient(_app(package_root)).post(
        f"/api/visits/{VISIT}/throws/0/correct", json={"sector": "21", "ring": "treble"}
    )
    assert resp.status_code == 400


def test_impossible_sector_ring_combinations_are_rejected(package_root):
    """Combinations `sector_ring_for_point()` could never produce: a bull
    WITH a sector, and a treble WITHOUT one."""
    _write_throw(package_root)
    client = TestClient(_app(package_root))

    bull_with_sector = client.post(
        f"/api/visits/{VISIT}/throws/0/correct", json={"sector": "20", "ring": "bull"}
    )
    assert bull_with_sector.status_code == 400

    treble_without_sector = client.post(
        f"/api/visits/{VISIT}/throws/0/correct", json={"ring": "treble"}
    )
    assert treble_without_sector.status_code == 400


def test_a_rejected_correction_writes_nothing(package_root):
    """Validation happens before any write -- a 400 must leave the
    package exactly as it was, not half-annotated."""
    pkg = _write_throw(package_root)
    TestClient(_app(package_root)).post(
        f"/api/visits/{VISIT}/throws/0/correct", json={"sector": "20", "ring": "nonsense"}
    )
    assert load_ad_ground_truth(pkg) is None


# --------------------------------------------------------------------------
# Broadcast
# --------------------------------------------------------------------------

def test_correction_broadcasts_throw_corrected_then_packages_updated(package_root):
    """THROW_CORRECTED is new and additive; PACKAGES_UPDATED is the
    EXISTING message every connected dashboard tab already re-renders
    from -- so a correction made by a game driver shows up in the Scoring
    tab live, without that tab knowing this endpoint exists. Verified
    over a real WebSocket via TestClient, not a spawned process."""
    _write_throw(package_root, sector="1", ring="single_outer")
    app = _app(package_root)
    client = TestClient(app)

    with client.websocket_connect("/api/events") as ws:
        assert ws.receive_json()["type"] == "HELLO"
        _correct(client)
        corrected = ws.receive_json()
        packages = ws.receive_json()

    assert corrected["type"] == "THROW_CORRECTED"
    assert corrected["visit_id"] == VISIT
    assert corrected["visit_index"] == 0
    assert (corrected["live_sector"], corrected["live_ring"]) == ("1", "single_outer")
    assert (corrected["corrected_sector"], corrected["corrected_ring"]) == ("20", "treble")

    assert packages["type"] == "PACKAGES_UPDATED"
    assert packages["packages"][0]["ad_operator_confirmed_sector"] == "20"
