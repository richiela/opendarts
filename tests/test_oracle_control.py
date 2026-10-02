"""Starting, stopping and resetting Autodarts' board from OD (2026-09-27).

OD makes the same local calls Autodarts' own dashboard makes -- PUT
/api/start, PUT /api/stop, POST /api/reset on the board. The HTTP call is
faked (no real socket: this suite runs where binding ports is refused, see
test_ad_ws_listener's docstring) and records what would have been sent.
"""
from __future__ import annotations

import urllib.error
import urllib.request

import pytest
from fastapi.testclient import TestClient

from opendarts.live.ad_ws_listener import AdWsListener
from opendarts.live.server import create_app

# The timeouts the fake board was called with, per test (see `board`).
calls_timeouts: list = []


@pytest.fixture
def board(monkeypatch):
    """A stand-in Autodarts board: records each request, answers 200."""
    url = "http://board.test:3180"
    calls: list[tuple[str, str]] = []

    class _Resp:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _urlopen(req, timeout=None):
        if not req.full_url.startswith(url):
            raise urllib.error.URLError("unreachable")
        calls.append((req.get_method(), req.full_url[len(url):]))
        timeouts.append(timeout)
        return _Resp()

    timeouts: list = []
    calls_timeouts.clear()
    calls_timeouts.append(timeouts)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    return url, calls


@pytest.mark.parametrize("action,expected", [
    ("start", ("PUT", "/api/start")),
    ("stop", ("PUT", "/api/stop")),
    ("reset", ("POST", "/api/reset")),
    ("calibrate", ("POST", "/api/config/calibration/auto")),
])
def test_each_control_makes_autodarts_own_call(board, action, expected):
    url, calls = board
    result = AdWsListener(url).control(action)
    assert result == {"ok": True, "action": action, "status": 200}
    assert calls == [expected]


def test_calibrate_waits_as_long_as_autodarts_takes_to_calibrate(board):
    """Autodarts answers a calibration only when it is done; a start or a
    stop answers at once. Each gets its own timeout."""
    url, _calls = board
    listener = AdWsListener(url)
    listener.control("start")
    listener.control("calibrate")
    start_t, calibrate_t = calls_timeouts[0]
    assert start_t == 10.0 and calibrate_t >= 60.0


def test_a_control_is_refused_when_autodarts_is_switched_off(board):
    url, calls = board
    listener = AdWsListener(url)
    listener.set_enabled(False)
    result = listener.control("start")
    assert result["ok"] is False and "switched off" in result["error"]
    assert calls == []


def test_an_unknown_action_or_an_unreachable_board_is_an_answer_not_a_crash(board):
    listener = AdWsListener("http://elsewhere.test:3180")  # the fake board is not there
    assert listener.control("explode")["ok"] is False
    result = listener.control("start", timeout=1.0)
    assert result["ok"] is False and "could not reach" in result["error"]


def test_the_route_passes_the_control_through(board, tmp_path):
    url, calls = board
    app = create_app(package_root=tmp_path, enable_background_poll=False,
                     ad_ws_listener=AdWsListener(url))
    client = TestClient(app)
    assert client.post("/api/oracle/stop").json() == {"ok": True, "action": "stop", "status": 200}
    assert calls == [("PUT", "/api/stop")]
    assert client.post("/api/oracle/explode").status_code == 404


def test_the_route_says_so_when_autodarts_is_not_wired(tmp_path):
    app = create_app(package_root=tmp_path, enable_background_poll=False)
    response = TestClient(app).post("/api/oracle/start")
    assert response.status_code == 409 and response.json()["ok"] is False
