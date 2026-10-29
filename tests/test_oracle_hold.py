"""The oracle hold (2026-09-28): with the oracle on, OD is not ready unless
Autodarts is, and a dart that lands while it holds is not scored.

The lifecycle only sees a generic ``hold`` provider; opendarts/live/
oracle_hold.py answers it from the listener; the server reports the same
answer. Scenes come from test_lifecycle_state; the listener is a stand-in
with the four reads the hold uses.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from tests.test_lifecycle_empty_board import DART_A, DART_B
from tests.test_lifecycle_state import Scene, board_mask, run, warm

from opendarts.lifecycle.driver import LifecycleDriver
from opendarts.lifecycle.state import Action, Lifecycle, LifecycleConfig
from opendarts.live.board_status import (
    BOARD_STATUS_READY,
    BOARD_STATUS_STOPPED,
    BOARD_STATUS_TAKEOUT,
    BOARD_STATUS_UNKNOWN,
)
from opendarts.live.oracle_hold import TAKEOUT_GRACE_S, hold_reason_provider, oracle_hold
from opendarts.live.server import AppState

CFG = LifecycleConfig(warmup_stable_frames=3)


class Oracle:
    """A stand-in listener: switched on, connected, ready."""

    def __init__(self, status=BOARD_STATUS_READY, *, enabled=True, connected=True, age=60.0):
        self.status, self.enabled, self.connected, self.age = status, enabled, connected, age
        self.controls: list[tuple[str, float | None]] = []

    def is_enabled(self):
        return self.enabled

    def is_connected(self):
        return self.connected

    def board_status(self):
        return self.status, None

    def board_status_age_sec(self):
        return self.age

    def control(self, action, timeout=None):
        self.controls.append((action, timeout))
        return {"ok": True, "action": action, "status": 200}

    pending_start = False

    def start_when_board_ready(self):
        self.pending_start = True

    def cancel_pending_start(self):
        self.pending_start = False


class Switch:
    """A hold provider the test flips by hand."""

    def __init__(self, reason=None):
        self.reason = reason

    def __call__(self):
        return self.reason


def lifecycle(hold=None) -> Lifecycle:
    return Lifecycle({0: board_mask()}, CFG, hold=hold)


def commits(ticks):
    return sum(t.action is Action.COMMIT for t in ticks)


# ---------------------------------------------------------------- the lifecycle


def test_without_a_hold_provider_a_dart_scores_as_before():
    lc, s = lifecycle(), Scene()
    warm(lc, s)
    s.darts.append(DART_A)
    assert commits(run(lc, s, 15)) == 1 and lc.dart_count == 1


def test_a_provider_that_never_holds_changes_nothing():
    lc, s = lifecycle(Switch(None)), Scene()
    warm(lc, s)
    s.darts.append(DART_A)
    assert commits(run(lc, s, 15)) == 1 and lc.dart_count == 1


def test_a_dart_that_lands_while_held_is_not_scored():
    lc, s = lifecycle(Switch("oracle in takeout")), Scene()
    warm(lc, s)
    s.darts.append(DART_A)
    ticks = run(lc, s, 15)
    assert commits(ticks) == 0 and lc.dart_count == 0
    refused = [t for t in ticks if t.removed is not None]
    assert len(refused) == 1 and refused[0].reason == "not scored: oracle in takeout"


def test_after_a_held_dart_is_pulled_the_next_dart_scores():
    hold = Switch("oracle stopped")
    lc, s = lifecycle(hold), Scene()
    warm(lc, s)
    s.darts.append(DART_A)
    run(lc, s, 15)
    hold.reason = None                  # the oracle is back
    assert commits(run(lc, s, 10)) == 0, "a held dart is never scored late"
    s.darts = []                        # pulled
    ticks = run(lc, s, 20)
    assert commits(ticks) == 0, "pulling it is a removal, not a throw"
    s.darts = [DART_B]
    assert commits(run(lc, s, 15)) == 1 and lc.dart_count == 1


def test_the_oracle_is_judged_as_the_dart_lands_not_when_it_scores():
    # Autodarts often reports a third dart -- and goes into takeout -- a
    # moment before OD commits the same dart. That must not cost the dart.
    hold = Switch(None)
    lc, s = lifecycle(hold), Scene()
    warm(lc, s)
    s.darts.append(DART_A)
    run(lc, s, 1)                       # the dart has landed
    hold.reason = "oracle in takeout"   # ...then the oracle goes into takeout
    assert commits(run(lc, s, 15)) == 1 and lc.dart_count == 1


def test_a_hold_that_lifts_before_the_commit_still_does_not_score_the_dart():
    hold = Switch("oracle offline")
    lc, s = lifecycle(hold), Scene()
    warm(lc, s)
    s.darts.append(DART_A)
    run(lc, s, 1)                       # landed while held
    hold.reason = None
    assert commits(run(lc, s, 15)) == 0 and lc.dart_count == 0


def test_a_hold_mid_visit_and_the_takeout_after_it_score_nothing_extra():
    hold = Switch(None)
    lc, s = lifecycle(hold), Scene()
    warm(lc, s)
    s.darts.append(DART_A)
    assert commits(run(lc, s, 15)) == 1
    hold.reason = "oracle in takeout"
    s.darts.append(DART_B)              # thrown while held
    assert commits(run(lc, s, 15)) == 0 and lc.dart_count == 1
    hold.reason = None
    s.darts = []                        # both pulled
    ticks = run(lc, s, 60)
    assert commits(ticks) == 0
    assert any(t.action is Action.CLEARED for t in ticks) and lc.dart_count == 0
    s.darts = [DART_B]
    assert commits(run(lc, s, 15)) == 1


def test_the_driver_logs_a_held_dart_and_keeps_no_picture(tmp_path):
    s = Scene()
    driver = LifecycleDriver(log_dir=tmp_path, masks_provider=lambda: {0: board_mask()},
                             config=CFG, hold=Switch("oracle in takeout"))
    for _ in range(6):
        driver.observe({0: s.render()})
    s.darts.append(DART_A)
    for _ in range(15):
        driver.observe({0: s.render()})
    driver.close()
    events = [json.loads(line) for line in driver.log_path.read_text().splitlines()]
    held = [e for e in events if e.get("event") == "held"]
    assert len(held) == 1 and held[0]["why"] == "oracle in takeout"
    assert not driver.evidence_dir.exists() or not any(driver.evidence_dir.iterdir())


# ---------------------------------------------------------------- the question


def test_no_hold_with_the_oracle_off_or_absent():
    assert oracle_hold(None) is None
    assert oracle_hold(Oracle(BOARD_STATUS_STOPPED, enabled=False)) is None
    assert oracle_hold(Oracle(BOARD_STATUS_READY)) is None


def test_each_way_the_oracle_can_be_not_ready():
    assert oracle_hold(Oracle(connected=False)).code == "offline"
    assert oracle_hold(Oracle(BOARD_STATUS_STOPPED)).code == "stopped"
    assert oracle_hold(Oracle(BOARD_STATUS_TAKEOUT)).code == "takeout"
    assert oracle_hold(Oracle(BOARD_STATUS_UNKNOWN)).code == "unknown"
    assert oracle_hold(Oracle(BOARD_STATUS_TAKEOUT)).reason == "oracle in takeout"


def test_the_scoring_gate_forgives_a_takeout_that_just_began():
    fresh = Oracle(BOARD_STATUS_TAKEOUT, age=0.4)
    assert oracle_hold(fresh) is not None, "the light shows every takeout"
    assert hold_reason_provider(fresh)() is None, "the gate forgives a fresh one"
    stuck = Oracle(BOARD_STATUS_TAKEOUT, age=TAKEOUT_GRACE_S + 1)
    assert hold_reason_provider(stuck)() == "oracle in takeout"
    assert hold_reason_provider(Oracle(BOARD_STATUS_STOPPED, age=0.1))() == "oracle stopped"


# ---------------------------------------------------------------- the server


class _Running:
    def is_running(self):
        return True


def _state(listener) -> AppState:
    d = Path(tempfile.mkdtemp())
    st = AppState(package_root=d, scratch_dir=d, n_cameras=3, package_poll_interval_s=999,
                  ad_ws_listener=listener)
    st.controller = _Running()
    return st


def test_a_hold_is_takeout_on_the_retail_feed_with_its_reason():
    oracle = Oracle(BOARD_STATUS_TAKEOUT)
    st = _state(oracle)
    fields = st._build_retail_state_fields()
    assert fields["status"] == "takeout" and fields["hold"] == "oracle in takeout"
    assert st.oracle_hold_snapshot() == {"code": "takeout", "reason": "oracle in takeout"}
    oracle.status = BOARD_STATUS_READY
    fields = st._build_retail_state_fields()
    assert fields["status"] == "throw" and fields["hold"] is None


def test_a_hold_starting_and_lifting_are_their_own_retail_events():
    oracle = Oracle(BOARD_STATUS_READY)
    st = _state(oracle)

    async def go():
        q = st.subscribe_retail()
        await st.publish_retail_state()
        oracle.status = BOARD_STATUS_TAKEOUT
        await st.publish_retail_state()
        oracle.status = BOARD_STATUS_STOPPED
        await st.publish_retail_state()
        oracle.status = BOARD_STATUS_READY
        await st.publish_retail_state()
        out = []
        while not q.empty():
            out.append(q.get_nowait())
        return out

    events = [(m["event"], m["status"], m["hold"]) for m in asyncio.run(go())]
    assert events[-3:] == [
        ("hold_started", "takeout", "oracle in takeout"),
        ("hold_changed", "takeout", "oracle stopped"),
        ("hold_released", "throw", None),
    ]


def test_without_an_oracle_the_retail_feed_is_unchanged():
    st = _state(None)
    fields = st._build_retail_state_fields()
    assert fields["status"] == "throw" and fields["hold"] is None


def test_reset_resets_the_oracle_too_while_it_is_on(tmp_path):
    from fastapi.testclient import TestClient
    from opendarts.live.server import create_app

    oracle = Oracle()
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False,
                                   ad_ws_listener=oracle))
    body = client.post("/api/reset").json()
    assert oracle.controls and oracle.controls[0][0] == "reset"
    assert body["oracle_reset"]["ok"] is True
    oracle.enabled = False
    oracle.controls.clear()
    body = client.post("/api/reset").json()
    assert oracle.controls == [] and "oracle_reset" not in body


class _FakeHub:
    def open_all(self):
        return [True, True, True]

    def close_all(self):
        pass

    def status_report(self) -> str:
        return "fake hub"


def test_start_starts_a_stopped_oracle_too(tmp_path):
    """One press gets the board working: Start starts a stopped oracle, as
    Reset resets both -- once OD has warmed up, so Start only asks for it.
    A running, offline or switched-off oracle is left alone."""
    from fastapi.testclient import TestClient
    from opendarts.live.capture_daemon import CaptureLoopController
    from opendarts.live.server import create_app

    def start(oracle):
        c = TestClient(create_app(package_root=tmp_path / str(id(oracle)), enable_background_poll=False,
                                  local_hub=_FakeHub(), controller=CaptureLoopController(),
                                  ad_ws_listener=oracle))
        return c.post("/api/start").json()

    stopped = Oracle(BOARD_STATUS_STOPPED)
    body = start(stopped)
    assert stopped.controls == [] and stopped.pending_start
    assert body["oracle_start"]["ok"] is True and body["oracle_start"]["deferred"] is True
    running = Oracle(BOARD_STATUS_READY)           # checked after warm-up, not started
    body = start(running)
    assert running.controls == [] and running.pending_start and "oracle_start" not in body
    for oracle in (Oracle(BOARD_STATUS_STOPPED, connected=False),
                   Oracle(BOARD_STATUS_STOPPED, enabled=False)):
        body = start(oracle)
        assert oracle.controls == [] and not oracle.pending_start and "oracle_start" not in body


def test_the_oracle_starts_once_the_board_has_settled_not_at_start():
    """Started the moment Start was pressed, Autodarts took its empty board
    before the cabinet's lights came on and sat in a takeout that never
    ended; started at the end of OD's warm-up, it still caught the cameras
    adjusting. It now starts after SETTLE_S of a still, empty board -- any
    change restarts the wait -- once, and only if a Start asked."""
    import time

    from opendarts.live.ad_ws_listener import AdWsListener

    lst = AdWsListener("http://fake-ad:3180")
    calls = []
    lst.control = lambda action, timeout=None: calls.append(action) or {"ok": True}
    lst.is_enabled = lambda: True
    lst.is_connected = lambda: True
    status = [BOARD_STATUS_STOPPED]
    lst.board_status = lambda: (status[0], None)
    S = lst.SETTLE_S

    def run(ticks):
        """ticks: [(t, quiet)] -> the controls sent."""
        for t, quiet in ticks:
            lst.board_tick(quiet, now=t)
        time.sleep(0.05)
        out = list(calls)
        calls.clear()
        return out

    assert run([(0, True), (S + 1, True)]) == []                 # nobody asked
    lst.start_when_board_ready()
    assert run([(0, True), (S - 0.1, True)]) == []               # not settled long enough
    assert run([(S + 0.1, True)]) == ["start"]
    assert run([(10, True), (20, True)]) == []                   # once per Start

    # the 20:23 case: quiet, then the cameras adjust (a scene change) -- the wait restarts
    lst.start_when_board_ready()
    assert run([(0, True), (1.3, True), (1.8, False), (2.5, True), (3.9, True)]) == []
    assert run([(2.5 + S, True)]) == ["start"]

    lst.start_when_board_ready()
    lst.cancel_pending_start()                                   # Stop first
    assert run([(0, True), (S + 1, True)]) == []

    # cameras still counted down a moment after Start: wait, don't give up
    status[0] = BOARD_STATUS_TAKEOUT
    lst._ad_num = 0
    lst._cams_down = True
    lst._board_status = BOARD_STATUS_TAKEOUT
    lst.board_status = lambda: (lst._effective_status_locked(), None)
    lst.start_when_board_ready()
    assert run([(0, True), (S + 0.1, True), (S + 1, True)]) == []
    lst._cams_down = False
    assert run([(S + 1.5, True)]) == ["reset"]
    lst.board_status = lambda: (status[0], None)

    # already running: an empty takeout is reset; a real takeout or a ready board is not
    for st, n, expect in ((BOARD_STATUS_TAKEOUT, 0, ["reset"]), (BOARD_STATUS_TAKEOUT, 3, []),
                          (BOARD_STATUS_READY, 0, [])):
        status[0] = st
        lst._ad_num = n
        lst.start_when_board_ready()
        assert run([(0, True), (S + 0.1, True)]) == expect, (st, n)


def test_the_lifecycle_tells_the_oracle_whether_the_board_is_quiet():
    from types import SimpleNamespace

    from opendarts.lifecycle.state import Action, Phase
    from opendarts.live.capture_daemon import _build_lifecycle_driver

    oracle = Oracle(BOARD_STATUS_STOPPED)
    seen = []
    oracle.board_tick = lambda quiet: seen.append(quiet)
    driver = _build_lifecycle_driver(ad_ws_listener=oracle)
    for phase, action, n in ((Phase.IDLE, Action.NONE, 0), (Phase.WARMUP, Action.NONE, 0),
                             (Phase.SCENE_CHANGE, Action.NONE, 0), (Phase.IDLE, Action.READY, 0),
                             (Phase.IDLE, Action.NONE, 1)):
        driver._on_tick(SimpleNamespace(phase=phase, action=action, dart_count=n))
    assert seen == [True, False, False, False, False]


def test_stop_stops_a_running_oracle_too(tmp_path):
    """Stop stops both, as Start starts both. A stopped, offline or
    switched-off oracle is left alone."""
    from fastapi.testclient import TestClient
    from opendarts.live.capture_daemon import CaptureLoopController
    from opendarts.live.server import create_app

    def start_then_stop(oracle):
        c = TestClient(create_app(package_root=tmp_path / str(id(oracle)), enable_background_poll=False,
                                  local_hub=_FakeHub(), controller=CaptureLoopController(),
                                  ad_ws_listener=oracle))
        assert c.post("/api/start").json()["ok"] is True
        oracle.controls.clear()
        return c.post("/api/stop").json()

    for status in (BOARD_STATUS_READY, BOARD_STATUS_TAKEOUT):
        running = Oracle(status)
        body = start_then_stop(running)
        assert running.controls == [("stop", 10.0)] and body["oracle_stop"]["ok"] is True
    for oracle in (Oracle(BOARD_STATUS_STOPPED), Oracle(BOARD_STATUS_READY, connected=False),
                   Oracle(BOARD_STATUS_READY, enabled=False)):
        body = start_then_stop(oracle)
        assert oracle.controls == [] and "oracle_stop" not in body
