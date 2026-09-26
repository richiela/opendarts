"""Cameras that moved on the ring: relearn the layout, say so, carry on.

When the gaps between cameras drift past RING_GEOMETRY_DRIFT_THRESHOLD_DEG,
a camera has almost certainly been moved or remounted. Until 2026-09-17
that REFUSED the whole calibration and told the operator to delete this
rig's learned geometry -- first by hand, later through a dashboard button.
The answer was always yes, so the calibration now relearns the layout from
its own event and continues, and the drift is reported instead: in the log,
on the action log, and on the calibration panel.

Nothing else about a refusal changed. A camera that cannot establish its
orientation at all still refuses, because relearning would hide a real
camera problem (lighting, framing, focus).

The endpoint that clears the learned geometry stays as a manual control.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opendarts.calibration.rig_ring_geometry import (
    RING_GEOMETRY_FALLBACK_FILENAME,
    RingGeometry,
    update_ring_geometry,
)
from opendarts.live.capture_daemon import (
    RING_GEOMETRY_DRIFT_THRESHOLD_DEG,
    ring_geometry_for_this_event,
)
from opendarts.live.server import create_app


@pytest.fixture
def calib_root(tmp_path, monkeypatch):
    """Point the endpoint at a throwaway calibration package root."""
    root = tmp_path / "calib-packages"
    root.mkdir()
    monkeypatch.setattr(
        "opendarts.live.server.DEFAULT_CALIBRATION_PACKAGE_ROOT", root
    )
    return root


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(package_root=tmp_path / "pkgs",
                                 enable_background_poll=False))


def _write_geometry(root: Path) -> Path:
    path = root / RING_GEOMETRY_FALLBACK_FILENAME
    path.write_text(json.dumps({"schema": "ring-geometry/v1", "gaps_deg": [120.0, 120.0],
                                "n_events": 9, "updated_at_utc": "2026-09-01T00:00:00Z"}))
    return path


# -- the decision itself -------------------------------------------------

_STEADY = {0: 0.0, 1: 120.0, 2: 240.0}


def _learned(hints=_STEADY, events: int = 8) -> RingGeometry:
    geometry = None
    for n in range(events):
        geometry, _ = update_ring_geometry(geometry, hints, f"2026-09-0{n + 1}T00:00:00Z")
    return geometry


def test_an_ordinary_event_just_updates_the_layout():
    geometry = _learned()
    updated, drift, note = ring_geometry_for_this_event(
        geometry, {0: 0.3, 1: 120.2, 2: 239.8}, "2026-09-17T00:00:00Z")
    assert note is None
    assert drift is not None and drift <= RING_GEOMETRY_DRIFT_THRESHOLD_DEG
    assert updated.n_events == geometry.n_events + 1


def test_a_moved_camera_relearns_the_layout_instead_of_refusing():
    """The camera that moved is 30deg round the ring from where it was."""
    geometry = _learned()
    moved = {0: 0.0, 1: 150.0, 2: 240.0}
    relearned, drift, note = ring_geometry_for_this_event(
        geometry, moved, "2026-09-17T00:00:00Z")

    assert drift is not None and drift > RING_GEOMETRY_DRIFT_THRESHOLD_DEG
    assert note is not None
    # Seeded from THIS event alone, not averaged with the old layout --
    # the old one describes where the cameras used to be.
    assert relearned.n_events == 1
    seeded, _ = update_ring_geometry(None, moved, "2026-09-17T00:00:00Z")
    assert relearned.gaps_deg == seeded.gaps_deg
    assert note["drift_deg"] == pytest.approx(drift, abs=0.01)
    assert note["previous_gaps_deg"] == [round(x, 2) for x in geometry.gaps_deg]
    assert note["gaps_deg"] == [round(x, 2) for x in relearned.gaps_deg]
    assert note["at_utc"] == "2026-09-17T00:00:00Z"
    # The note is what the operator reads: it has to say what happened, by
    # how much, and that nobody asked for it.
    assert "moved" in note["note"] and f"{drift:.1f}deg" in note["note"]


def test_a_first_event_has_no_layout_to_have_moved_from():
    seeded, drift, note = ring_geometry_for_this_event(
        None, _STEADY, "2026-09-17T00:00:00Z")
    assert (drift, note) == (None, None)
    assert seeded.n_events == 1


# -- reaching the operator ----------------------------------------------


def test_the_note_travels_from_the_calibration_to_the_dashboard():
    """Every hop between the calibration and the panel, by name: a break
    anywhere here loses the only sign that a camera moved."""
    from opendarts.live import capture_daemon, server

    daemon_src = Path(capture_daemon.__file__).read_text()
    server_src = Path(server.__file__).read_text()
    # The panel and the action log live in the dashboard's own app.js since
    # the page came out of server.py's f-string -- same chain, two files.
    dash_js = (Path(server.__file__).parent / "dashboard" / "app.js").read_text()

    # calibration -> out-dict -> the Start-time event
    assert 'calibration_package_out["ring_geometry_relearned"] = relearn_note' in daemon_src
    assert '"ring_geometry_relearned": calibration_package_out.get(' in daemon_src
    # -> server state -> /api/state, the refresh response and the live push
    assert 'self.calibration_geometry_relearned = data.get("ring_geometry_relearned")' in server_src
    assert 'self.calibration_geometry_relearned = event.get("ring_geometry_relearned")' in server_src
    assert server_src.count('"ring_geometry_relearned": self.calibration_geometry_relearned') >= 3
    # -> the panel and the action log
    assert "function renderCameraMovedNote(relearned)" in dash_js
    assert "renderCameraMovedNote(calibration && calibration.ring_geometry_relearned)" in dash_js
    assert "'Cameras moved', 'warn', movedShort(" in dash_js
    # The action log is a narrow box: the line says the fact and the number,
    # with the full note on its tooltip.
    assert "function movedShort(relearned)" in dash_js


def test_the_state_endpoint_carries_the_note(client):
    calibration = client.get("/api/state").json()["calibration"]
    assert "ring_geometry_relearned" in calibration
    assert calibration["ring_geometry_relearned"] is None    # nothing moved


def test_the_dashboard_has_somewhere_to_show_it(client):
    html = client.get("/?ui=classic").text
    assert 'id="calib-moved-note"' in html
    assert "Cameras moved" in html


# -- the manual control, unchanged --------------------------------------


def test_the_endpoint_deletes_the_learned_geometry(client, calib_root):
    path = _write_geometry(calib_root)
    body = client.post("/api/calibration/relearn-ring-geometry").json()
    assert body["ok"] is True and body["cleared"] is True
    assert not path.exists()


def test_clearing_twice_is_harmless(client, calib_root):
    _write_geometry(calib_root)
    assert client.post("/api/calibration/relearn-ring-geometry").json()["cleared"] is True
    second = client.post("/api/calibration/relearn-ring-geometry").json()
    assert second["ok"] is True and second["cleared"] is False


def test_a_virgin_rig_reports_nothing_to_clear_rather_than_failing(client, calib_root):
    body = client.post("/api/calibration/relearn-ring-geometry").json()
    assert body["ok"] is True and body["cleared"] is False
