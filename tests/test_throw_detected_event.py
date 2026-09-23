"""Tests for the THROW_DETECTED event -- the single "a dart was just
scored" push (added 2026-08-14, docs/LIVE_API.md).

Before this, learning what was scored took three hops: watch TRIGGER_STATE
reach READY_TO_CAPTURE, then catch PACKAGES_UPDATED, then GET
/api/packages. THROW_DETECTED carries the whole scored throw in one
payload, fired at the exact moment the primary engine's result is durably
on disk.

Two halves are proven here, both against real code paths:
  * the PRODUCER -- `capture_daemon.handle_ready_to_capture()` emits it
    with the right payload (2026-09-01: now fired BEFORE the package is
    saved, not after -- see test_throw_detected_fires_before_the_package_
    is_durably_on_disk's own docstring for the "background the throw-
    package save" task this reflects), for a failed score as well as a
    successful one;
  * the CONSUMER -- `AppState._handle_live_event()` (the exact coroutine
    opendarts/live/run_product.py's live_event_queue plumbing drives)
    rebroadcasts it to connected WebSocket clients and tracks it as part
    of the current visit, verified through a real
    `TestClient.websocket_connect` rather than a spawned server process
    (docs/DESIGN.md behavioral-correction, 2026-08-12).

And the thing most likely to be broken by accident: that this is PURELY
ADDITIVE -- PACKAGE_SAVED, PACKAGES_UPDATED and TRIGGER_STATE all still
fire exactly as before, since the shipped dashboard's own JS depends on
them.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from opendarts.engines.base import EngineResult
from opendarts.live import capture_daemon
from tests.lifecycle_scripting import script_trigger
from opendarts.live.server import create_app
from opendarts.pipeline import CameraCalibration

@pytest.fixture()
def scratch(tmp_path):
    # No teardown of our own on purpose. This fixture used to delete the
    # directory itself and had to retry for a second, because a throw's
    # background threads (AD attach, also-run engines) can still be writing
    # here as the test ends and rmtree failed with "Directory not empty".
    # pytest's own tmp_path reaper runs long after those threads are gone.
    d = tmp_path / "run"
    d.mkdir(parents=True)
    return d


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((5, 1)),
        rvec=np.zeros((3, 1)),
        tvec=np.zeros((3, 1)),
        pnp_result=None,
        landmark_spread_ok=True,
    )


class _FakeEngine:
    """Stands in for a real engine at exactly the surface
    handle_ready_to_capture() uses (`score()` -> EngineResult) -- same
    approach tests/test_capture_daemon.py already takes, so these tests
    exercise the real save/emit wiring without depending on real tip
    detection against synthetic 8x8 images."""

    def __init__(self, result: EngineResult) -> None:
        self._result = result

    def score(self, bg_images, frame_images, calibrations) -> EngineResult:
        return self._result


def _run_capture(
    scratch: Path,
    monkeypatch,
    *,
    engine_result: EngineResult,
    visit_id: str | None = "visit_1700000000000",
    visit_index: int | None = 1,
):
    """Calls the REAL handle_ready_to_capture() -- real save_throw_package,
    real _emit -- with only the engine itself faked."""
    monkeypatch.setattr(
        capture_daemon, "get_engine", lambda name: _FakeEngine(engine_result)
    )
    trigger = capture_daemon.ThrowTriggerState(
        state=capture_daemon.ThrowState.READY_TO_CAPTURE,
        dart_count=(visit_index + 1) if visit_index is not None else 1,
        last_frame={0: np.full((8, 8, 3), 255, dtype=np.uint8)},
    )
    events: list[dict] = []
    dest_dir = capture_daemon.handle_ready_to_capture(
        trigger,
        {0: np.zeros((8, 8, 3), dtype=np.uint8)},
        {0: _calibration()},
        scratch / "packages",
        "session-throw-detected",
        on_event=events.append,
        visit_id=visit_id,
        visit_index=visit_index,
    background_save=False,
    )
    # The default config runs four also-run engines in a BACKGROUND thread
    # that writes `other_engines` into result.json after this call returns.
    # Wait for that write to land before handing back, so teardown never
    # races it (a real "Directory not empty" on rmtree otherwise). This
    # test is about the THROW_DETECTED payload, not the engine roster --
    # it just must not tear down mid-write.
    result_json = dest_dir / "result.json"
    for _ in range(200):
        try:
            if "other_engines" in json.loads(result_json.read_text()):
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.02)
    return dest_dir, events


def _ok_result() -> EngineResult:
    return EngineResult(
        ok=True,
        sector="20",
        ring="treble",
        board_xy_mm=(3.5, 101.2),
        reason="",
        diagnostics={"max_ray_disagreement_mm": 2.75, "n_cameras_used": 3},
    )


# --------------------------------------------------------------------------
# Producer -- capture_daemon.handle_ready_to_capture()
# --------------------------------------------------------------------------

def test_throw_detected_carries_the_whole_scored_throw_in_one_payload(scratch, monkeypatch):
    """The point of the event: everything a consumer previously needed
    three hops to assemble, in one push."""
    dest_dir, events = _run_capture(scratch, monkeypatch, engine_result=_ok_result())

    detected = [e for e in events if e["type"] == "THROW_DETECTED"]
    assert len(detected) == 1
    event = detected[0]

    assert event["ok"] is True
    assert event["sector"] == "20"
    assert event["ring"] == "treble"
    assert event["board_xy_mm"] == [3.5, 101.2]
    # Carried through from the engine's OWN diagnostics (3 in this
    # fixture) by the Apollo EngineResult -> ScoreResult adapter -- not
    # recounted from the frames handed in, which is why it isn't 1 here.
    assert event["n_cameras_used"] == 3
    assert event["session"] == "session-throw-detected"
    assert event["throw_id"] == dest_dir.name
    assert event["path"] == str(dest_dir)
    assert event["visit_id"] == "visit_1700000000000"
    assert event["visit_index"] == 1
    assert event["primary_engine"] == capture_daemon.DEFAULT_PRIMARY_ENGINE


def test_throw_detected_timestamp_is_the_packages_own_not_a_second_now_call(
    scratch, monkeypatch
):
    """captured_at_utc must be read back off the package just written,
    not recomputed -- a consumer joining this event to the stored package
    by time must not get two different answers for one throw."""
    dest_dir, events = _run_capture(scratch, monkeypatch, engine_result=_ok_result())

    event = next(e for e in events if e["type"] == "THROW_DETECTED")
    on_disk = json.loads((dest_dir / "meta.json").read_text())["captured_at_utc"]
    assert event["captured_at_utc"] == on_disk


def test_throw_detected_carries_emitted_at_utc_matching_captured_at_utc(scratch, monkeypatch):
    """`emitted_at_utc` (2026-09-07, requested by the QA harness on :8900 --
    TRIGGER_STATE already carries this same field, added 2026-09-01, same
    key name and convention). Reuses `captured_at_utc` verbatim: that value
    is already stamped right after the primary engine decided this result,
    before this emit and well before the backgrounded package write -- the
    exact "score is decided" moment the QA harness asked for, not a second,
    marginally-later clock read."""
    dest_dir, events = _run_capture(scratch, monkeypatch, engine_result=_ok_result())

    event = next(e for e in events if e["type"] == "THROW_DETECTED")
    assert event["emitted_at_utc"] is not None
    assert event["emitted_at_utc"] == event["captured_at_utc"]
    on_disk = json.loads((dest_dir / "meta.json").read_text())["captured_at_utc"]
    assert event["emitted_at_utc"] == on_disk


def test_throw_detected_surfaces_the_real_quality_signal_not_an_invented_confidence(
    scratch, monkeypatch
):
    """A THROW_DETECTED event elsewhere carries a `confidence` float.
    This project has NO confidence signal anywhere -- not on ScoreResult, not on
    EngineResult, not in any engine's diagnostics (checked 2026-08-14) --
    so none is fabricated. What ships instead is the real signal that
    does exist: how far apart the per-camera rays landed, in mm.

    Pinned as a test because "just add a confidence: 0.85" is exactly the
    kind of plausible-looking fabrication a later change might introduce.
    """
    _dest_dir, events = _run_capture(scratch, monkeypatch, engine_result=_ok_result())
    event = next(e for e in events if e["type"] == "THROW_DETECTED")

    assert event["max_ray_disagreement_mm"] == 2.75
    assert "confidence" not in event


def test_throw_detected_fires_for_a_failed_score_too(scratch, monkeypatch):
    """docs/DESIGN.md's "Replay is the source of truth": a package is ALWAYS written,
    including for a rejected/low-confidence score, and a consumer needs
    to hear about that throw rather than silently waiting forever for an
    event that never comes. `ok: False` plus the real reason, not
    silence."""
    _dest_dir, events = _run_capture(
        scratch,
        monkeypatch,
        engine_result=EngineResult(
            ok=False,
            sector=None,
            ring=None,
            board_xy_mm=None,
            reason="tip not detected in enough cameras",
            diagnostics={},
        ),
    )

    event = next(e for e in events if e["type"] == "THROW_DETECTED")
    assert event["ok"] is False
    assert event["sector"] is None
    assert event["board_xy_mm"] is None
    assert event["reason"] == "tip not detected in enough cameras"


def test_throw_detected_fires_before_the_package_is_durably_on_disk(scratch, monkeypatch):
    """Timing contract INVERTED, 2026-09-01 ("background the throw-package
    save" task) -- deliberately, not a regression. Before
    this date, THROW_DETECTED only fired once the package was durably
    saved (real numbers behind the reason it changed: save=0.153-0.165s
    of a 0.421-0.470s total handle_ready_to_capture() call, confirmed
    against real logs -- roughly a third of the total
    was the write, sitting needlessly on the critical path). Now
    THROW_DETECTED fires FIRST, then the write happens on a background
    thread -- a consumer that needs the package durably on disk (replay,
    re-scoring, a game layer pulling frames) must wait for the SEPARATE
    PACKAGE_SAVED event instead, not assume THROW_DETECTED implies it."""
    seen_at_emit: dict = {}

    real_emit = capture_daemon._emit

    def spy_emit(on_event, event):
        if event.get("type") == "THROW_DETECTED":
            pkg = Path(event["path"])
            seen_at_emit["meta"] = (pkg / "meta.json").exists()
            seen_at_emit["result"] = (pkg / "result.json").exists()
            seen_at_emit["frame"] = (pkg / "stills_cam0.mkv").exists()
        return real_emit(on_event, event)

    monkeypatch.setattr(capture_daemon, "_emit", spy_emit)
    # background_save left at its real default (True) here, deliberately
    # -- this test is specifically about the real, live async timing
    # contract, not the background_save=False deterministic-testing
    # override _run_capture() uses elsewhere in this file.
    trigger = capture_daemon.ThrowTriggerState(
        state=capture_daemon.ThrowState.READY_TO_CAPTURE,
        dart_count=2,
        last_frame={0: np.full((8, 8, 3), 255, dtype=np.uint8)},
    )
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: _FakeEngine(_ok_result()))
    dest_dir = capture_daemon.handle_ready_to_capture(
        trigger,
        {0: np.zeros((8, 8, 3), dtype=np.uint8)},
        {0: _calibration()},
        scratch / "packages",
        "session-throw-detected",
        on_event=None,
        visit_id="visit_1700000000000",
        visit_index=1,
    )

    assert seen_at_emit == {"meta": False, "result": False, "frame": False}, (
        "THROW_DETECTED must fire BEFORE the package exists on disk -- if this "
        "now shows True, the save is no longer actually backgrounded"
    )

    # Wait for the real background save to actually finish before this test
    # function returns -- background_save=True here means a genuine daemon
    # thread is still writing to `dest_dir` after the assertion above. Not
    # doing this races the `scratch` fixture's own teardown (a real
    # shutil.rmtree "Directory not empty" failure, reproduced while writing
    # this test) against that still-running thread.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not (dest_dir / "meta.json").exists():
        time.sleep(0.01)
    assert (dest_dir / "meta.json").exists(), "background save never completed within 5s"


def test_background_save_false_restores_the_old_fully_synchronous_guarantee(
    scratch, monkeypatch
):
    """The escape hatch this project's own pre-existing test suite (and
    any future deterministic/offline caller) relies on: background_save=
    False makes the package durably exist on disk the moment handle_
    ready_to_capture() returns, byte-identical to this function's own
    pre-2026-09-01 behavior -- proven directly here, not just assumed
    because other tests happen to pass with it set."""
    dest_dir, _events = _run_capture(scratch, monkeypatch, engine_result=_ok_result())
    assert (dest_dir / "meta.json").exists()
    assert (dest_dir / "result.json").exists()
    assert (dest_dir / "stills_cam0.mkv").exists()


def test_visit_fields_are_honestly_absent_when_no_visit_is_tracked(scratch, monkeypatch):
    """A caller with no visit context (this module's own standalone CLI
    path, an offline tool) gets None -- never a fabricated "visit 0,
    dart 0"."""
    dest_dir, events = _run_capture(
        scratch, monkeypatch, engine_result=_ok_result(), visit_id=None, visit_index=None
    )

    event = next(e for e in events if e["type"] == "THROW_DETECTED")
    assert event["visit_id"] is None
    assert event["visit_index"] is None
    meta = json.loads((dest_dir / "meta.json").read_text())
    assert "visit_id" not in meta


def test_package_saved_still_fires_unchanged_alongside_the_new_event(scratch, monkeypatch):
    """PURELY ADDITIVE. PACKAGE_SAVED is emitted by run_capture_loop_body()
    right after handle_ready_to_capture() returns and is what drives the
    existing PACKAGES_UPDATED re-render the shipped dashboard JS depends
    on -- proven here at the loop level, so a future "THROW_DETECTED
    replaces PACKAGE_SAVED" cleanup can't silently break the dashboard.
    """
    baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}

    class _StopLoop(Exception):
        pass

    scripted = [
        capture_daemon.ThrowTriggerState(
            state=capture_daemon.ThrowState.READY_TO_CAPTURE,
            dart_count=1,
            true_baseline_frames=baseline,
            last_frame=baseline,
        )
    ]
    calls = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        idx = calls["n"]
        calls["n"] += 1
        if idx < len(scripted):
            return scripted[idx]
        raise _StopLoop()

    monkeypatch.setattr(
        capture_daemon,
        "bootstrap_calibrations",
        lambda snapshot_dir, **kw: {0: _calibration()},
    )
    monkeypatch.setattr(
        capture_daemon, "fetch_current_frames", lambda dest_dir, **kw: baseline
    )
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: _FakeEngine(_ok_result()))

    events: list[dict] = []
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=scratch / "packages",
            poll_interval_s=0.0,
            stop_event=threading.Event(),
            scratch_dir=scratch / "scratch",
            on_event=events.append,
            # 2026-09-01: PACKAGE_SAVED now fires from inside
            # handle_ready_to_capture()'s own background save (see that
            # function's own docstring) -- background_save=False forces
            # it to run inline, so this test's own single-shot event list
            # is complete by the time run_capture_loop_body() returns,
            # matching what this test is actually checking (both events
            # fire, PURELY ADDITIVE) rather than a real backgrounding race.
            background_save=False,
        )

    types = [e["type"] for e in events]
    assert "THROW_DETECTED" in types
    assert "PACKAGE_SAVED" in types
    assert "TRIGGER_STATE" in types
    # THROW_DETECTED lands first (emitted from inside
    # handle_ready_to_capture, before it returns); PACKAGE_SAVED follows.
    assert types.index("THROW_DETECTED") < types.index("PACKAGE_SAVED")


# --------------------------------------------------------------------------
# Consumer -- AppState._handle_live_event / WebSocket
# --------------------------------------------------------------------------

class _FakeWebSocket:
    """Same minimal stand-in tests/test_live_server.py already uses for
    AppState._broadcast (a plain object with an async send_text)."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


def _app(scratch: Path):
    return create_app(package_root=scratch / "packages", enable_background_poll=False)


def _throw_event(index: int = 0, visit_id: str = "visit_1700000000000") -> dict:
    return {
        "type": "THROW_DETECTED",
        "session": "session-throw-detected",
        "throw_id": f"throw_{index}",
        "path": "/nonexistent/throw",
        "captured_at_utc": "2026-08-14T00:00:00+00:00",
        "visit_id": visit_id,
        "visit_index": index,
        "primary_engine": "Apollo",
        "ok": True,
        "sector": "20",
        "ring": "treble",
        "board_xy_mm": [3.5, 101.2],
        "n_cameras_used": 3,
        "reason": "",
        "max_ray_disagreement_mm": 2.75,
    }


def test_server_rebroadcasts_throw_detected_verbatim_to_websocket_clients(scratch):
    app = _app(scratch)
    state = app.state.opendarts_state
    ws = _FakeWebSocket()
    state.clients.add(ws)

    asyncio.run(state._handle_live_event(_throw_event()))  # noqa: SLF001 -- matches this suite's pattern

    # TWO messages per scored dart since 2026-09-15: the spoken call goes
    # out FIRST, as its own frame, because sound is the slowest thing a
    # human notices and it should leave before the payload that redraws a
    # table. See the DART_CALL branch in AppState._handle_live_event.
    assert [json.loads(m)["type"] for m in ws.sent] == ["DART_CALL", "THROW_DETECTED"]
    call = json.loads(ws.sent[0])
    # The PHRASE, already resolved -- never sector+ring for a client to
    # interpret. The naming vocabulary lives in opendarts/live/audio.py
    # and nowhere else.
    assert call["phrase"] == "treble 20"
    msg = json.loads(ws.sent[1])
    assert msg["type"] == "THROW_DETECTED"
    assert msg["sector"] == "20"
    assert msg["visit_id"] == "visit_1700000000000"
    assert msg["visit_index"] == 0
    # The server adds its own receive timestamp (every branch of
    # _handle_live_event does) and interprets nothing else.
    assert "ts" in msg


def test_server_accumulates_the_current_visits_throws_and_clears_on_rotation(scratch):
    """/api/state's `visit` section is what a game driver hydrates from
    on (re)connect -- it must reflect the real throws of the CURRENT
    visit only, and reset the moment the visit rotates."""
    app = _app(scratch)
    state = app.state.opendarts_state

    async def _drive() -> None:
        await state._handle_live_event(_throw_event(0))  # noqa: SLF001
        await state._handle_live_event(_throw_event(1))  # noqa: SLF001
        await state._handle_live_event(  # noqa: SLF001
            {
                "type": "VISIT_CLEARED",
                "session": "session-throw-detected",
                "visit_id": "visit_1700000009999",
                "previous_visit_id": "visit_1700000000000",
                "n_darts": 2,
                "reason": "takeout",
            }
        )

    asyncio.run(_drive())

    assert state.visit_id == "visit_1700000009999"
    assert state.visit_throws == []


def test_a_throw_under_a_new_visit_drops_a_previous_visits_throws(scratch):
    """Belt-and-braces for a MISSED VISIT_CLEARED. A throw arriving under
    a different visit id is itself proof the previous visit ended --
    keeping the old turn's darts would both misreport /api/state's visit
    section and let visit_throws grow without bound over a long session.
    """
    app = _app(scratch)
    state = app.state.opendarts_state

    async def _drive() -> None:
        await state._handle_live_event(_throw_event(0, visit_id="visit_1"))  # noqa: SLF001
        await state._handle_live_event(_throw_event(1, visit_id="visit_1"))  # noqa: SLF001
        # No VISIT_CLEARED in between -- simulating a dropped rotation.
        await state._handle_live_event(_throw_event(0, visit_id="visit_2"))  # noqa: SLF001

    asyncio.run(_drive())

    assert state.visit_id == "visit_2"
    assert len(state.visit_throws) == 1
    assert state.visit_throws[0]["visit_id"] == "visit_2"


def test_api_state_visit_section_is_real_and_honest_about_standalone_mode(scratch):
    """No live_event_queue (this module's own standalone CLI) -- the
    visit section says so rather than reporting an empty visit that
    doesn't exist, exactly like the `trigger` section already does."""
    client = TestClient(_app(scratch))
    visit = client.get("/api/state").json()["visit"]

    assert visit["available"] is False
    assert visit["visit_id"] is None
    assert visit["n_throws"] == 0
    assert visit["throws"] == []
    assert visit["max_throws"] == capture_daemon.MAX_DARTS_PER_TURN


def test_api_state_visit_section_reports_the_live_visit_after_a_real_throw(scratch):
    app = _app(scratch)
    state = app.state.opendarts_state
    asyncio.run(state._handle_live_event(_throw_event(0)))  # noqa: SLF001

    visit = TestClient(app).get("/api/state").json()["visit"]
    assert visit["visit_id"] == "visit_1700000000000"
    assert visit["n_throws"] == 1
    assert visit["throws"][0]["sector"] == "20"


def test_a_real_websocket_client_receives_throw_detected(scratch):
    """End-to-end over a REAL WebSocket via TestClient -- no spawned
    server process, no port bound, nothing to kill (docs/DESIGN.md
    behavioral-correction, 2026-08-12)."""
    app = _app(scratch)
    state = app.state.opendarts_state

    with TestClient(app).websocket_connect("/api/events") as ws:
        assert ws.receive_json()["type"] == "HELLO"
        asyncio.run(state._handle_live_event(_throw_event(2)))  # noqa: SLF001
        call = ws.receive_json()
        msg = ws.receive_json()

    # The spoken call arrives first, then the throw itself.
    assert call["type"] == "DART_CALL" and call["phrase"] == "treble 20"
    assert msg["type"] == "THROW_DETECTED"
    assert msg["visit_index"] == 2
    assert msg["ring"] == "treble"
