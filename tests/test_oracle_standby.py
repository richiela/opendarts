"""Autodarts' camera standby follows OD's idle timeout (2026-10-01).

OD's idle timeout is the one setting: on connect and whenever it changes,
OD sets Autodarts' "Camera standby time" (motion.standby_minutes, one of
5/10/15/30/60) to the smallest choice at least as long, the call
Autodarts' own dashboard makes (PATCH /api/config). HTTP is faked.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from opendarts.live.ad_ws_listener import AdWsListener, standby_minutes_for

URL = "http://board.test:3180"


@pytest.mark.parametrize("sec,expected", [
    (None, None), (0, None), (-5, None),
    (60, 5), (300, 5), (301, 10), (600, 10), (900, 15),
    (1800, 30), (1801, 60), (3600, 60), (7200, 60),
])
def test_the_standby_choice_for_ods_idle_timeout(sec, expected):
    assert standby_minutes_for(sec) == expected


@pytest.fixture
def board(monkeypatch):
    """A stand-in board whose config holds a standby time; records calls."""
    state = {"standby": 60, "fail": None}
    calls: list = []

    class _Resp:
        def __init__(self, body=b"{}", status=200):
            self._body, self.status = body, status
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _urlopen(req, timeout=None):
        if state["fail"]:
            raise urllib.error.URLError(state["fail"])
        method, path = req.get_method(), req.full_url[len(URL):]
        calls.append((method, path))
        if method == "PATCH":
            state["standby"] = json.loads(req.data)["motion"]["standby_minutes"]
            return _Resp()
        return _Resp(json.dumps({"motion": {"standby_minutes": state["standby"]}}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return state, calls


def test_patches_only_when_it_differs(board):
    state, calls = board
    ad = AdWsListener(URL)
    ad.set_standby_source(lambda: 1800)
    assert ad.sync_camera_standby()["changed"] is True
    assert state["standby"] == 30
    assert calls == [("GET", "/api/config"), ("PATCH", "/api/config")]
    calls.clear()
    assert ad.sync_camera_standby()["changed"] is False
    assert calls == [("GET", "/api/config")]


def test_a_changed_idle_timeout_is_matched(board):
    state, _ = board
    ad = AdWsListener(URL)
    ad.sync_camera_standby(900)
    assert state["standby"] == 15
    ad.sync_camera_standby(3600)
    assert state["standby"] == 60


def test_no_idle_timeout_leaves_autodarts_alone(board):
    state, calls = board
    result = AdWsListener(URL).sync_camera_standby(0)
    assert result["changed"] is False and calls == [] and state["standby"] == 60


def test_oracle_off_makes_no_calls(board):
    _, calls = board
    ad = AdWsListener(URL)
    ad._enabled = False  # noqa: SLF001 -- set_enabled() would also stop a socket that never ran
    assert ad.sync_camera_standby(1800)["ok"] is False
    assert calls == []


def test_a_failure_is_dropped_not_raised(board):
    state, _ = board
    state["fail"] = "unreachable"
    result = AdWsListener(URL).sync_camera_standby(1800)
    assert result["ok"] is False and "could not read" in result["reason"]


def test_no_source_means_no_sync(board):
    _, calls = board
    assert AdWsListener(URL).sync_camera_standby()["ok"] is False
    assert calls == []


def test_the_config_route_syncs_a_changed_idle_timeout(monkeypatch, tmp_path):
    """Changing OD's idle timeout through PATCH /api/config asks the
    listener to match it, off-thread (the answer never waits on Autodarts)."""
    from fastapi.testclient import TestClient
    from opendarts.live import config as config_mod
    from opendarts.live.capture_daemon import CaptureLoopController
    from opendarts.live.server import create_app

    config_mod.DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen: list = []
    ad = AdWsListener(URL)
    monkeypatch.setattr(ad, "sync_camera_standby_async", lambda sec=None: seen.append(sec))
    app = create_app(package_root=tmp_path / "packages", port=8420, host="0.0.0.0",
                     enable_background_poll=False,
                     controller=CaptureLoopController(), ad_ws_listener=ad)
    r = TestClient(app).patch("/api/config", json={"idle_timeout_sec": 900})
    assert r.status_code == 200, r.text
    assert seen == [900]
