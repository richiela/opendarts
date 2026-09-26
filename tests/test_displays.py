"""Displays (opendarts/live/displays.py): screens that only show, whose
settings live on the rig and are changed from any controller."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from opendarts.live.displays import DEFAULT_SETTINGS, DisplayError, DisplayStore
from opendarts.live.server import create_app


def test_first_report_registers_with_the_tv_defaults():
    store = DisplayStore()
    rec = store.hello("tv", {"w": 1920, "h": 1080})
    assert rec["name"] == "Display 1"
    assert rec["settings"] == DEFAULT_SETTINGS
    assert rec["settings"]["layout"] == "split" and rec["settings"]["theme"] == "dark"
    assert rec["online"] is True and rec["info"] == {"w": 1920, "h": 1080}
    assert store.hello("office")["name"] == "Display 2"


def test_a_name_from_the_kiosk_url_is_used_once():
    store = DisplayStore()
    assert store.hello("tv", name="  Lounge   TV ")["name"] == "Lounge TV"
    store.update("tv", name="Board TV")
    assert store.hello("tv", name="Lounge TV")["name"] == "Board TV"


def test_each_display_keeps_its_own_settings():
    store = DisplayStore()
    store.hello("tv")
    store.hello("office")
    store.update("tv", settings={"layout": "scoring", "text_size": "large"})
    assert store.get("tv")["settings"]["layout"] == "scoring"
    assert store.get("office")["settings"]["layout"] == "split"


@pytest.mark.parametrize("bad", [
    {"layout": "tiles"}, {"text_size": 3}, {"theme": "auto"}, {"sound": "yes"},
    {"volume": 1.5}, {"volume": True}, {"nonsense": 1},
])
def test_a_setting_the_display_could_not_show_is_refused(bad):
    store = DisplayStore()
    store.hello("tv")
    with pytest.raises(DisplayError) as err:
        store.update("tv", settings=bad)
    assert err.value.status == 400
    assert store.get("tv")["settings"] == DEFAULT_SETTINGS


def test_a_locked_display_takes_only_its_unlock():
    store = DisplayStore()
    store.hello("tv")
    store.update("tv", settings={"locked": True})
    with pytest.raises(DisplayError) as err:
        store.update("tv", settings={"layout": "engines"})
    assert err.value.status == 409
    with pytest.raises(DisplayError):
        store.update("tv", settings={"locked": False, "layout": "engines"})
    store.update("tv", settings={"locked": False})
    assert store.update("tv", settings={"layout": "engines"})["settings"]["layout"] == "engines"


def test_bad_ids_are_refused():
    store = DisplayStore()
    for bad in ("", "a b", "x" * 65, None, 7, "../tv"):
        with pytest.raises(DisplayError):
            store.hello(bad)


def test_settings_survive_a_restart_and_the_rest_of_config_is_kept(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"port": 8420}))
    store = DisplayStore(snapshot_path=path)
    store.hello("tv")
    store.update("tv", name="Board TV", settings={"layout": "engines", "volume": 0.5})
    again = DisplayStore(snapshot_path=path)
    rec = again.get("tv")
    assert rec["name"] == "Board TV" and rec["settings"]["layout"] == "engines" and rec["settings"]["volume"] == 0.5
    assert rec["online"] is False, "presence is not persisted -- only a report makes a display online"
    assert json.loads(path.read_text())["port"] == 8420


def test_forget():
    store = DisplayStore()
    store.hello("tv")
    assert store.forget("tv") is True
    assert store.get("tv") is None and store.forget("tv") is False


# ---- over HTTP, and pushed to every open page ----

@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(package_root=tmp_path, enable_background_poll=False))


def test_the_display_page_is_the_same_page_in_its_display_role(client):
    resp = client.get("/?display=tv")
    assert resp.status_code == 200
    assert '"role": "display"' in resp.text
    assert '"role": "control"' in client.get("/").text


def test_a_change_from_a_controller_is_pushed_to_the_display(client):
    with client.websocket_connect("/api/events") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "HELLO"
        r = client.post("/api/displays/hello", json={"display_id": "tv", "info": {"w": 1920}})
        assert r.json()["display"]["settings"]["layout"] == "split"
        r = client.patch("/api/displays/tv", json={"settings": {"layout": "engines"}, "name": "Board TV"})
        assert r.status_code == 200
        msg = ws.receive_json()
        while msg["type"] != "DISPLAY_UPDATED":
            msg = ws.receive_json()
        assert msg["display"]["id"] == "tv" and msg["display"]["settings"]["layout"] == "engines"
        assert msg["display"]["name"] == "Board TV"
        client.post("/api/displays/tv/identify")
        msg = ws.receive_json()
        while msg["type"] != "DISPLAY_IDENTIFY":
            msg = ws.receive_json()
        assert msg["display_id"] == "tv" and msg["name"] == "Board TV"
    listed = client.get("/api/displays").json()["displays"]
    assert [d["id"] for d in listed] == ["tv"] and listed[0]["online"] is True


def test_errors_are_answered_as_errors(client):
    assert client.post("/api/displays/hello", json={"display_id": "no spaces"}).status_code == 400
    assert client.patch("/api/displays/ghost", json={"settings": {"layout": "split"}}).status_code == 404
    client.post("/api/displays/hello", json={"display_id": "tv"})
    assert client.patch("/api/displays/tv", json={"settings": {"layout": "tiles"}}).status_code == 400
    client.patch("/api/displays/tv", json={"settings": {"locked": True}})
    assert client.patch("/api/displays/tv", json={"settings": {"layout": "scoring"}}).status_code == 409
    assert client.post("/api/displays/ghost/identify").status_code == 404
    assert client.delete("/api/displays/tv").status_code == 200
    assert client.delete("/api/displays/tv").status_code == 404


# ---- calibration progress (opendarts/live/calibration_progress.py) ----

def test_calibration_progress_walks_the_stages_and_the_cameras():
    from opendarts.live.calibration_progress import CalibrationProgress

    p = CalibrationProgress()
    assert p.snapshot()["active"] is False
    p.start([0, 1, 2])
    snap = p.snapshot()
    assert snap["active"] and snap["stage"] == "capture" and snap["fraction"] == 0.0
    assert snap["cameras"] == {"0": "waiting", "1": "waiting", "2": "waiting"}
    p.stage("orientation")
    p.stage("solve")
    before = p.snapshot()["fraction"]
    p.camera(0, "done")
    p.camera(1, "failed")
    mid = p.snapshot()
    assert mid["fraction"] > before, "each camera finished moves the bar"
    assert [s["state"] for s in mid["stages"]] == ["done", "done", "now", "todo", "todo"]
    p.stage("refine", "attempt 1 of 2", 0.0)
    p.finish(True)
    end = p.snapshot()
    assert end["active"] is False and end["succeeded"] is True and end["fraction"] == 1.0
    assert all(s["state"] == "done" for s in end["stages"])


def test_calibration_progress_ignores_what_it_cannot_record():
    from opendarts.live.calibration_progress import CalibrationProgress

    p = CalibrationProgress()
    p.stage("solve")           # not started: ignored, never raises
    p.camera(0, "done")
    p.finish(False, "x")
    assert p.snapshot()["active"] is False and p.snapshot()["stage"] is None
    p.start([0])
    p.stage("no-such-stage")
    assert p.snapshot()["stage"] == "capture"


def test_calibration_progress_is_served(tmp_path):
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    body = client.get("/api/calibration/progress").json()
    assert body["ok"] is True and "active" in body and "fraction" in body
