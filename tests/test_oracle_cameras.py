"""The oracle's camera check (2026-10-01): a board whose cameras deliver no
frames is not ready, whatever Autodarts' own status says.

Proved on the reference rig: after OD's process was gone for over a minute,
Autodarts' camera grabber gave up on OD's virtual cameras for good, while
/api/state kept saying running/"Throw" -- the oracle light stayed green on
a blind board. Its per-camera frame rate (/api/cams/stats) read [0, 0, 0]
within seconds. These tests drive the real listener with fake readings;
nothing touches the network.
"""
from __future__ import annotations

import threading
import time

import pytest

import opendarts.live.ad_ws_listener as adl
from opendarts.live.ad_ws_listener import AD_CAMS_DOWN_AFTER_SEC, AdWsListener
from opendarts.live.board_status import (
    BOARD_STATUS_CAMERAS_DOWN,
    BOARD_STATUS_READY,
    BOARD_STATUS_STOPPED,
    BOARD_STATUS_TAKEOUT,
)
from opendarts.live.oracle_hold import oracle_hold


def _listener(status=BOARD_STATUS_READY):
    pushed: list[str] = []
    li = AdWsListener("http://ad.invalid:3180", on_status_change=pushed.append)
    li._connected = True  # noqa: SLF001 -- a live connection, without a socket
    running = status != BOARD_STATUS_STOPPED
    event = "Takeout started" if status == BOARD_STATUS_TAKEOUT else None
    li._update_board_status(event, 0, {"running": running})  # noqa: SLF001
    pushed.clear()
    return li, pushed


def test_zero_fps_under_the_threshold_does_not_trip():
    li, pushed = _listener()
    li.note_cams_fps([0, 0, 0], 100.0)
    li.note_cams_fps([0, 0, 0], 100.0 + AD_CAMS_DOWN_AFTER_SEC - 0.1)
    assert li.board_status()[0] == BOARD_STATUS_READY
    assert pushed == []


def test_zero_fps_for_the_threshold_is_cameras_down_and_pushed():
    li, pushed = _listener()
    li.note_cams_fps([30, 0, 29], 100.0)            # one camera is enough
    li.note_cams_fps([30, 0, 29], 100.0 + AD_CAMS_DOWN_AFTER_SEC)
    assert li.board_status()[0] == BOARD_STATUS_CAMERAS_DOWN
    assert pushed == [BOARD_STATUS_CAMERAS_DOWN]
    hold = oracle_hold(li)
    assert hold is not None and hold.as_dict() == {"code": "cameras", "reason": "oracle cameras down"}


def test_a_takeout_with_dead_cameras_is_cameras_down_and_never_graced():
    li, _ = _listener(BOARD_STATUS_TAKEOUT)
    li.note_cams_fps([0, 0, 0], 0.0)
    li.note_cams_fps([0, 0, 0], AD_CAMS_DOWN_AFTER_SEC)
    assert li.board_status()[0] == BOARD_STATUS_CAMERAS_DOWN
    assert oracle_hold(li, takeout_grace_s=60.0).code == "cameras"


def test_it_recovers_as_soon_as_every_camera_has_frames():
    li, pushed = _listener()
    li.note_cams_fps([0, 0, 0], 0.0)
    li.note_cams_fps([0, 0, 0], AD_CAMS_DOWN_AFTER_SEC + 1)
    li.note_cams_fps([30, 29, 30], AD_CAMS_DOWN_AFTER_SEC + 3)
    assert li.board_status()[0] == BOARD_STATUS_READY
    assert pushed == [BOARD_STATUS_CAMERAS_DOWN, BOARD_STATUS_READY]
    assert oracle_hold(li) is None
    # and a new outage starts its own clock
    li.note_cams_fps([0, 0, 0], 50.0)
    assert li.board_status()[0] == BOARD_STATUS_READY


def test_a_stopped_board_stays_stopped():
    li, pushed = _listener(BOARD_STATUS_STOPPED)
    li.note_cams_fps([0, 0, 0], 0.0)
    li.note_cams_fps([0, 0, 0], AD_CAMS_DOWN_AFTER_SEC + 5)
    assert li.board_status()[0] == BOARD_STATUS_STOPPED
    assert pushed == []
    assert oracle_hold(li).code == "stopped"


def test_a_failed_stats_read_is_not_cameras_down(monkeypatch):
    li, pushed = _listener()

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    assert li._read_cams_fps() is None  # noqa: SLF001
    assert li.board_status()[0] == BOARD_STATUS_READY and pushed == []


def test_a_disconnect_clears_the_camera_state():
    li, _ = _listener()
    li.note_cams_fps([0, 0, 0], 0.0)
    li.note_cams_fps([0, 0, 0], AD_CAMS_DOWN_AFTER_SEC)
    li._set_connected(False)  # noqa: SLF001
    li._connected = True  # noqa: SLF001 -- back, before any new reading
    assert li.board_status()[0] == BOARD_STATUS_READY


@pytest.mark.parametrize("enabled,connected", [(False, True), (True, False)])
def test_no_polling_unless_on_and_connected(monkeypatch, enabled, connected):
    li, _ = _listener()
    li._enabled, li._connected = enabled, connected  # noqa: SLF001
    calls: list[int] = []
    monkeypatch.setattr(adl, "AD_CAMS_POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(li, "_read_cams_fps", lambda: calls.append(1) or [30, 30, 30])
    t = threading.Thread(target=li._run_cams_check, daemon=True)  # noqa: SLF001
    t.start()
    time.sleep(0.1)
    li._stop_event.set()  # noqa: SLF001
    t.join(1)
    assert calls == []


def test_polls_while_on_and_connected(monkeypatch):
    li, _ = _listener()
    calls: list[int] = []
    monkeypatch.setattr(adl, "AD_CAMS_POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(li, "_read_cams_fps", lambda: calls.append(1) or [30, 30, 30])
    t = threading.Thread(target=li._run_cams_check, daemon=True)  # noqa: SLF001
    t.start()
    time.sleep(0.1)
    li._stop_event.set()  # noqa: SLF001
    t.join(1)
    assert calls
