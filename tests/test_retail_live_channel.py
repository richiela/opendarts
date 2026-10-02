"""WS /api/live -- the retail feed's wire contract.

A deliberately smaller channel than `/api/events`, for a client that only
needs game state. Its own subscriber set, no history, and exactly two
message types: `state` and `throw`.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opendarts.live.server import AppState, create_app, retail_dart_fields


@pytest.fixture()
def package_root(tmp_path):
    """A throwaway package root. The retail catch-up ring is deliberately
    independent of packages, so these tests want an empty one."""
    d = tmp_path / "packages"
    d.mkdir()
    return d


def _state() -> AppState:
    d = Path(tempfile.mkdtemp())
    return AppState(package_root=d, scratch_dir=d, n_cameras=3, package_poll_interval_s=999)


@pytest.mark.parametrize(
    "sector,ring,ok,expected",
    [
        (20, "treble", True, {"label": "T20", "sector": 20, "ring": "treble", "value": 60}),
        (20, "double", True, {"label": "D20", "sector": 20, "ring": "double", "value": 40}),
        (16, "single_inner", True,
         {"label": "S16", "sector": 16, "ring": "single_inner", "value": 16}),
        # bull/outer_bull carry sector 25 by convention -- not 0, not null
        (None, "bull", True, {"label": "BULL", "sector": 25, "ring": "bull", "value": 50}),
        (None, "outer_bull", True,
         {"label": "25", "sector": 25, "ring": "outer_bull", "value": 25}),
        # this project's internal "outside" is "miss" on the wire
        (None, "outside", True, {"label": "MISS", "sector": 0, "ring": "miss", "value": 0}),
        # an abstention is NOT a miss -- these must never be collapsed
        (None, None, False,
         {"label": "failed to score", "sector": 0, "ring": "", "value": 0}),
    ],
)
def test_wire_vocabulary_is_exact(sector, ring, ok, expected):
    assert retail_dart_fields(sector, ring, ok) == expected


def test_hello_is_a_flat_complete_snapshot():
    d = Path(tempfile.mkdtemp())
    with TestClient(create_app(package_root=d)) as client:
        with client.websocket_connect("/api/live") as ws:
            hello = ws.receive_json()
    assert hello["type"] == "state"
    assert hello["event"] == "hello"
    for key in ("running", "status", "visit", "n_darts", "darts"):
        assert key in hello, f"hello must always carry {key}"
    # flat object -- never wrapped in envelope/payload/data
    assert "payload" not in hello and "data" not in hello and "envelope" not in hello


def test_diffing_is_the_trigger():
    """Generic publishes only on a real change; an explicit event
    publishes even when the snapshot is identical, and is suppressed only
    when BOTH the fields and the name repeat.

    The last case is the real bug this shape exists to prevent:
    `throw_detected` and `visit_complete` fire back to back off the same
    snapshot for a visit's 3rd dart, and a fields-only dedup swallows the
    second one every single time."""
    st = _state()

    async def run() -> None:
        q = st.subscribe_retail()
        await st.publish_retail_state()
        await st.publish_retail_state()
        baseline = q.qsize()

        await st.publish_retail_state("throw_detected")
        assert q.qsize() == baseline + 1, "explicit event must publish on an unchanged snapshot"

        await st.publish_retail_state("throw_detected")
        assert q.qsize() == baseline + 1, "a redundant repeat must be suppressed"

        await st.publish_retail_state("visit_complete")
        assert q.qsize() == baseline + 2, "a different explicit event must not be swallowed"

    asyncio.run(run())


def test_retail_subscribers_are_separate_from_the_debug_feed():
    """A retail client must never appear in the debug feed's own client
    set, and unsubscribing must actually remove it."""
    st = _state()

    async def run() -> None:
        q = st.subscribe_retail()
        assert q in st._retail_subscribers
        assert q not in st.clients, "retail queues must not leak into the debug feed"
        st.unsubscribe_retail(q)
        assert q not in st._retail_subscribers

    asyncio.run(run())


def test_a_full_retail_queue_drops_that_client_not_the_feed():
    """A client that stops draining must be dropped rather than allowed
    to block the feed -- and a client that keeps draining must survive
    the same burst."""
    st = _state()

    async def run() -> None:
        slow = st.subscribe_retail()
        healthy = st.subscribe_retail()
        for _ in range(slow.maxsize + 5):
            await st.publish_retail({"type": "throw"})
            healthy.get_nowait()          # this one keeps up
        assert slow not in st._retail_subscribers, "a full client should be dropped"
        assert healthy in st._retail_subscribers, "a draining client must survive"

    asyncio.run(run())


# --------------------------------------------------------------------------
# Corrections on the retail channel. A client that already scored a visit
# would otherwise keep the wrong value forever -- nothing on /api/live
# contradicted it, and the snapshot builder read the engine's original
# answer rather than the correction.
# --------------------------------------------------------------------------


def test_retail_dart_reports_the_correction_not_the_original():
    from opendarts.live.server import retail_dart_from_package

    original = retail_dart_from_package(
        {"sector": 20, "ring": "single_inner", "ok": True, "captured_at_utc": "t"}
    )
    assert original["label"] == "S20" and original["value"] == 20
    assert "corrected" not in original

    corrected = retail_dart_from_package(
        {
            "sector": 20, "ring": "single_inner", "ok": True, "captured_at_utc": "t",
            "corrected_sector": "20", "corrected_ring": "treble",
        }
    )
    assert corrected["label"] == "T20"
    assert corrected["value"] == 60, "the correction must drive the value, not the original"
    assert corrected["corrected"] is True


def test_a_correction_on_an_unscored_dart_becomes_scored():
    """ok=False plus a correction is a dart the engine could not place and
    a human then placed -- it must stop reporting as 'failed to score'."""
    from opendarts.live.server import retail_dart_from_package

    d = retail_dart_from_package(
        {"sector": None, "ring": None, "ok": False,
         "corrected_sector": "5", "corrected_ring": "double"}
    )
    assert d["label"] == "D5" and d["value"] == 10


def test_correct_throw_publishes_on_the_retail_channel():
    """Named explicitly, not left to the status-transition mapper: a
    correction changes a dart without changing status, so the generic
    path would name no event and publish nothing at all."""
    import inspect
    from opendarts.live.server import AppState

    src = inspect.getsource(AppState.correct_throw)
    assert 'publish_retail_state("throw_corrected")' in src


# --------------------------------------------------------------------------
# GET /api/live/recent -- the retail catch-up. The socket carries no
# history by design, so a scoreboard reconnecting mid-match knew the
# current visit and nothing before it.
# --------------------------------------------------------------------------






def test_live_recent_is_bounded():
    from opendarts.live.server import RETAIL_RECENT_VISITS_MAX
    assert 1 <= RETAIL_RECENT_VISITS_MAX <= 50








# --------------------------------------------------------------------------
# GET /api/live/recent -- served from an IN-MEMORY ring appended as each
# visit closes, deliberately NOT from saved packages. Clearing the corpus,
# pulling it off the rig, or disabling package storage entirely must never
# erase a client's match history.
# --------------------------------------------------------------------------


def _throw(visit, index, sector, ring, **extra):
    e = {"type": "THROW_DETECTED", "visit_id": visit, "visit_index": index,
         "sector": sector, "ring": ring, "ok": True,
         "captured_at_utc": f"2026-09-09T01:00:0{index}+00:00"}
    e.update(extra)
    return e


def _cleared(previous, new, reason="takeout"):
    return {"type": "VISIT_CLEARED", "previous_visit_id": previous,
            "visit_id": new, "reason": reason, "n_darts": 3}


def _drive(state, events):
    async def go():
        for e in events:
            await state._handle_live_event(e)  # noqa: SLF001 -- matches this suite's pattern
    asyncio.run(go())


def test_live_recent_records_visits_as_they_close(package_root):
    """Driven through the REAL _handle_live_event path, with NO packages
    on disk -- the whole point of the ring."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    _drive(state, [
        _throw("v1", 0, 20, "single_inner"),
        _throw("v1", 1, 20, "treble"),
        _cleared("v1", "v2"),
        _throw("v2", 0, 5, "single_outer"),
        _cleared("v2", "v3"),
    ])

    body = TestClient(app).get("/api/live/recent").json()
    assert [v["visit"] for v in body["visits"]] == ["v1", "v2"], "oldest first"
    assert body["visits"][0]["n_darts"] == 2
    assert [d["label"] for d in body["visits"][0]["darts"]] == ["S20", "T20"]
    assert body["visits"][0]["total"] == 80
    assert body["count"] == 2
    # and none of this needed a single package on disk
    assert TestClient(app).get("/api/packages").json() == []


def test_live_recent_survives_the_corpus_being_deleted(package_root):
    """The failure this ring exists to prevent: a first cut read the
    package cache, so POST /api/packages/delete-all silently wiped a
    client's match history."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    _drive(state, [_throw("v1", 0, 20, "treble"), _cleared("v1", "v2")])
    client = TestClient(app)
    assert client.get("/api/live/recent").json()["count"] == 1

    client.post("/api/packages/delete-all")
    assert client.get("/api/live/recent").json()["count"] == 1, (
        "clearing the corpus must not erase retail match history"
    )


def test_live_recent_never_contains_the_open_visit(package_root):
    """Absent by construction: a visit enters the ring only when it
    closes, and the socket already owns the open one."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    _drive(state, [_throw("v1", 0, 20, "treble"), _cleared("v1", "v_open"),
                   _throw("v_open", 0, 1, "single_inner")])
    body = TestClient(app).get("/api/live/recent").json()
    assert [v["visit"] for v in body["visits"]] == ["v1"]


def test_live_recent_skips_an_empty_visit(package_root):
    """A Reset with nothing thrown has nothing to replay, and recording it
    would push a real turn out of a bounded ring."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    _drive(state, [_cleared("v_empty", "v1", reason="reset")])
    assert TestClient(app).get("/api/live/recent").json()["count"] == 0


def test_live_recent_reports_a_correction(package_root):
    """The catch-up and the socket share retail_dart_from_package(), so a
    corrected throw reads identically in both."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    _drive(state, [
        _throw("v1", 0, 20, "single_inner", corrected_sector="20", corrected_ring="treble"),
        _cleared("v1", "v2"),
    ])
    dart = TestClient(app).get("/api/live/recent").json()["visits"][0]["darts"][0]
    assert dart["label"] == "T20" and dart["value"] == 60 and dart["corrected"] is True


def test_live_recent_ring_is_bounded(package_root):
    from opendarts.live.server import RETAIL_RECENT_VISITS_MAX
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    events = []
    for i in range(RETAIL_RECENT_VISITS_MAX + 5):
        events += [_throw(f"v{i}", 0, 20, "treble"), _cleared(f"v{i}", f"v{i+1}")]
    _drive(state, events)
    body = TestClient(app).get("/api/live/recent").json()
    assert body["count"] == RETAIL_RECENT_VISITS_MAX
    assert body["visits"][0]["visit"] == "v5", "oldest evicted first"


def test_state_darts_come_from_live_throws_not_packages(package_root):
    """The snapshot must not depend on packages: with `store_packages`
    off there are none, and even with it on the save is backgrounded, so
    a package-derived snapshot lagged the throw it was reporting."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    _drive(state, [_throw("v1", 0, 20, "treble")])

    fields = state._build_retail_state_fields()  # noqa: SLF001
    assert fields["visit"] == "v1"
    assert [d["label"] for d in fields["darts"]] == ["T20"]
    assert fields["darts"][0]["value"] == 60
    # ...and nothing was ever written to disk
    assert TestClient(app).get("/api/packages").json() == []


def test_state_darts_report_a_correction(package_root):
    """correct_throw() writes corrected_* onto the visit_throws entry as
    well as the package, so the snapshot reports it without reading
    disk."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    _drive(state, [_throw("v1", 0, 20, "single_inner")])
    assert state._build_retail_state_fields()["darts"][0]["label"] == "S20"  # noqa: SLF001

    state.visit_throws[0]["corrected_sector"] = "20"
    state.visit_throws[0]["corrected_ring"] = "treble"
    d = state._build_retail_state_fields()["darts"][0]  # noqa: SLF001
    assert d["label"] == "T20" and d["value"] == 60 and d["corrected"] is True


class _FakeHub:
    def open_all(self):
        return [True, True, True]

    def close_all(self):
        pass

    def status_report(self) -> str:
        return "fake hub status"


@pytest.mark.parametrize("in_takeout", [False, True])
@pytest.mark.parametrize("reason", ["manual", "idle timeout"])
def test_start_and_stop_reach_the_retail_feed(package_root, reason, in_takeout):
    """A client listening only on the retail feed (FlightDeck, the kiosk)
    must see OD start and stop. The capture thread pushes no event when a
    session ends, so a Stop -- or the idle timeout -- used to leave the
    feed claiming OD was still capturing, and the kiosk never turned the
    cabinet off."""
    from opendarts.live.capture_daemon import CaptureLoopController

    state = create_app(
        package_root=package_root, enable_background_poll=False,
        local_hub=_FakeHub(), controller=CaptureLoopController(),
    ).state.opendarts_state

    async def run() -> list:
        q = state.subscribe_retail()
        await state.start_capture()
        if in_takeout:
            # Stopping from a takeout (or an oracle hold, which also reads
            # takeout) was the case live on the reference rig: the generic path names
            # nothing when the status leaves takeout.
            state.capture_starting = False
            state.trigger_state = "TAKEOUT_WAITING"
            await state.publish_retail_state()
        state.controller.stopped_ack.set()
        await state.stop_capture(reason=reason)
        msgs = []
        while not q.empty():
            msgs.append(q.get_nowait())
        return msgs

    msgs = [m for m in asyncio.run(run()) if m.get("type") == "state"]
    events = [m["event"] for m in msgs]
    assert "connecting" in events
    assert events[-1] == "stopped"
    assert msgs[-1]["running"] is False and msgs[-1]["status"] == "stopped"


def test_no_oracle_hold_during_ods_own_takeout():
    """Pulling the darts puts OD and the oracle in takeout together. That
    is OD's own takeout, not a hold: reporting the oracle's takeout as a
    hold made every visit's takeout read as "held by the oracle" (the
    cabinet strip turned purple at every takeout). Once OD's own takeout
    ends, a still-stuck oracle is a hold again."""
    st = _state()
    st.oracle_hold_snapshot = lambda: {"code": "takeout", "reason": "oracle in takeout"}
    st.visit_id, st.visit_throws = "v1", [{"visit_id": "v1", "visit_index": 0}]
    st.trigger_state = "TAKEOUT_WAITING"
    assert st._build_retail_state_fields()["hold"] is None  # noqa: SLF001
    st.visit_throws = []
    st.trigger_state = "IDLE"
    assert st._build_retail_state_fields()["hold"] == "oracle in takeout"  # noqa: SLF001


def test_ods_own_takeout_reaches_the_retail_feed_from_a_real_trigger_event():
    """The capture loop sends TRIGGER_STATE with the state's name
    (`trigger.state.name`). The retail status compared it against the lower-
    case value, so OD's own takeout never showed: the feed said `throw`, and
    with the oracle also in takeout it read as an oracle hold (purple on the
    cabinet strip at every takeout)."""
    from opendarts.capture.trigger_state import ThrowState

    st = _state()
    st.visit_id, st.visit_throws = "v1", [{"visit_id": "v1", "visit_index": 0}]

    async def run():
        await st._handle_live_event(  # noqa: SLF001
            {"type": "TRIGGER_STATE", "state": ThrowState.TAKEOUT_WAITING.name,
             "session": "s1", "dart_count": 3}
        )

    asyncio.run(run())
    st.oracle_hold_snapshot = lambda: {"code": "takeout", "reason": "oracle in takeout"}
    assert st.trigger_state == "TAKEOUT_WAITING"
    fields = st._build_retail_state_fields()  # noqa: SLF001
    assert fields["hold"] is None


class _Running:
    def is_running(self):
        return True


def test_a_takeout_lasts_until_its_visit_clears_and_the_clear_says_throw(package_root):
    """Driven through the real event path in the order the capture loop
    sends it: a hand mid-pull must not flip the feed to `throw`, and the
    clear itself must say `throw` -- VISIT_CLEARED arrives just before the
    trigger leaves TAKEOUT_WAITING, and the return to throw after it was
    never named, leaving FlightDeck and the kiosk in takeout."""
    from opendarts.capture.trigger_state import ThrowState

    state = create_app(package_root=package_root, enable_background_poll=False).state.opendarts_state
    state.controller = _Running()
    trig = lambda s: {"type": "TRIGGER_STATE", "state": s, "session": "s1", "dart_count": 3}  # noqa: E731
    q = state.subscribe_retail()
    _drive(state, [_throw("v1", i, 20, "single_outer") for i in range(3)])
    _drive(state, [trig(ThrowState.TAKEOUT_WAITING.name)])
    assert state.retail_status == "takeout"
    _drive(state, [trig("SETTLING"), trig("MOTION_DETECTED")])   # hand in mid-pull
    assert state.retail_status == "takeout"
    _drive(state, [_cleared("v1", "v2"), trig("IDLE")])
    msgs = []
    while not q.empty():
        msgs.append(q.get_nowait())
    last = [m for m in msgs if m.get("type") == "state"][-1]
    assert last["status"] == "throw"
    assert state.retail_status == "throw"
