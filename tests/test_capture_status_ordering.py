"""The status pill must never stay on "Waiting . Starting" after the server
has left Starting.

REPORTED 2026-09-22: after Start (and/or Calibrate) the pill stayed on
Waiting . Starting until a browser refresh. The server's state was right;
the tab's copy was stale.

Root cause: the dashboard learns the capture-loop status over two channels
nothing orders against each other -- WebSocket pushes and the HTTP body of
its own POST /api/start. With a valid calibration already in place the
capture thread skips auto-calibrate and emits its first TRIGGER_STATE (the
one message that ends Starting) within milliseconds of request_start(),
while start_capture() is still awaiting its final broadcast. The start
response then hard-coded `starting: True` and was applied after that
TRIGGER_STATE, and nothing cleared it again until the next dart.

The fix stamps every capture-loop snapshot with a per-process
{status_epoch, status_seq} taken as the state is read, and the dashboard
drops any snapshot older than the newest one it has applied. These tests
pin both halves: the server's stamps order correctly even when the pump
interleaves with start_capture(), and the client guard, run in a real JS
engine where one is available, rejects exactly the stale snapshots.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from opendarts.live.server import _render_dashboard_html, create_app


@pytest.fixture()
def package_root(tmp_path):
    """Own parent folder per test -- see test_live_server's fixture of the
    same name for why the extra level matters."""
    return tmp_path / "run" / "packages"


class _FakeHub:
    """Minimal local_hub stand-in (same shape as test_live_server's)."""

    def open_all(self):
        return [True, True]

    def close_all(self):
        pass

    def status_report(self) -> str:
        return "fake hub status"


def _app(package_root):
    from opendarts.live.capture_daemon import CaptureLoopController

    return create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=_FakeHub(),
        controller=CaptureLoopController(),
    )


def _recording_broadcast(state, on_msg=None):
    sent: list[dict] = []

    async def _broadcast(msg):
        sent.append(msg)
        if on_msg is not None:
            await on_msg(msg)

    state._broadcast = _broadcast  # noqa: SLF001
    return sent


def test_start_response_is_newer_than_a_trigger_state_that_overtook_it(package_root):
    """The actual race: the capture thread's first TRIGGER_STATE is handled
    by the event pump while start_capture() is still awaiting its FINAL
    CAPTURE_LOOP_STATUS broadcast. Simulated by handling that event from
    inside the broadcast, which is exactly where the await yields."""
    state = _app(package_root).state.opendarts_state
    fired = False

    async def _interleave(msg):
        nonlocal fired
        if msg.get("type") == "CAPTURE_LOOP_STATUS" and msg.get("cameras") and not fired:
            fired = True
            await state._handle_live_event(  # noqa: SLF001
                {"type": "TRIGGER_STATE", "state": "IDLE", "session": "s1", "dart_count": 0}
            )

    sent = _recording_broadcast(state, _interleave)
    body = asyncio.run(state.start_capture())

    assert fired
    early, final, trig = sent[0], sent[1], sent[2]
    assert early["type"] == final["type"] == "CAPTURE_LOOP_STATUS"
    assert trig["type"] == "TRIGGER_STATE"
    # The response must tell the truth the server holds when it is built --
    # it used to hard-code True here, which is half of the bug.
    assert state.capture_starting is False
    assert body["starting"] is False
    # And every snapshot is ordered by when it was taken, so a client can
    # drop the stale starting-True ones whatever order they arrive in.
    assert early["status_seq"] < final["status_seq"] < trig["status_seq"] < body["status_seq"]
    assert len({m["status_epoch"] for m in (early, final, trig, body)}) == 1
    assert final["starting"] is True  # the stale one the client must drop


@pytest.mark.parametrize("reason", ["manual", "idle timeout"])
def test_stop_and_idle_timeout_snapshots_are_stamped_in_order(package_root, reason):
    """Stop and the idle-timeout auto-stop share stop_capture(); both its
    broadcasts and its response must carry increasing stamps, newer than
    anything the Start that preceded it sent."""
    app = _app(package_root)
    state = app.state.opendarts_state
    sent = _recording_broadcast(state)

    started = asyncio.run(state.start_capture())
    state.controller.stopped_ack.set()
    stopped = asyncio.run(state.stop_capture(reason=reason))

    seqs = [m["status_seq"] for m in sent] + [stopped["status_seq"]]
    assert started["status_seq"] < sent[-2]["status_seq"]
    assert seqs[-3:] == sorted(seqs[-3:])
    assert sent[-2]["type"] == "CAPTURE_LOOP_STATUS" and sent[-2]["starting"] is False
    assert sent[-1]["type"] == "TRIGGER_STATE" and "status_seq" in sent[-1]
    assert stopped["starting"] is False


def test_api_state_capture_loop_is_stamped_and_advances(package_root):
    client = TestClient(_app(package_root))
    a = client.get("/api/state").json()["capture_loop"]
    b = client.get("/api/state").json()["capture_loop"]
    assert a["status_epoch"] == b["status_epoch"]
    assert b["status_seq"] > a["status_seq"]


# -- client side --------------------------------------------------------


@pytest.fixture(scope="module")
def dashboard_js() -> str:
    html = _render_dashboard_html(3)
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


def _function_source(js: str, name: str) -> str:
    start = js.index(f"function {name}(")
    depth = 0
    for i in range(js.index("{", start), len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[start : i + 1]
    raise AssertionError(f"unterminated function {name}")


def test_accept_capture_status_rejects_only_stale_snapshots(dashboard_js: str) -> None:
    """Runs the SHIPPED guard in node, not a Python re-implementation of it.
    Skipped where node is absent."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed -- the client guard's behaviour was NOT exercised")
    script = (
        "let captureStatusEpoch = null; let captureStatusSeq = -1;\n"
        + _function_source(dashboard_js, "acceptCaptureStatus")
        + """
const r = [];
r.push(acceptCaptureStatus({status_epoch: 'a', status_seq: 5}));   // first: yes
r.push(acceptCaptureStatus({status_epoch: 'a', status_seq: 3}));   // older: NO
r.push(acceptCaptureStatus({status_epoch: 'a', status_seq: 5}));   // same: yes
r.push(acceptCaptureStatus({status_epoch: 'a', status_seq: 9}));   // newer: yes
r.push(acceptCaptureStatus({status_epoch: 'b', status_seq: 1}));   // restart: yes
r.push(acceptCaptureStatus({status_epoch: 'b', status_seq: 0}));   // older in new epoch: NO
r.push(acceptCaptureStatus({running: true, starting: true}));      // unstamped: yes
r.push(acceptCaptureStatus(null));                                 // standalone: yes
r.push(acceptCaptureStatus({status_epoch: 'b', status_seq: 0}));   // baseline kept: NO
console.log(JSON.stringify(r));
"""
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == [True, False, True, True, True, False, True, True, False]


def test_every_server_snapshot_writer_goes_through_the_guard(dashboard_js: str) -> None:
    """Each place that copies a server snapshot into pillCaptureLoop must be
    guarded -- one unguarded writer is enough to bring the bug back. The
    only bare assignments allowed are the declaration and Start's own
    unstamped optimistic guess."""
    bare = re.findall(r"^\s*(?:let )?pillCaptureLoop = ([^;]+);", dashboard_js, re.M)
    assert sorted(bare) == sorted(["null", "{ running: true, starting: true }", "state.capture_loop", "msg"])
    # The two bare `state.capture_loop` / `msg` hits must sit right after a guard.
    assert "if (!acceptCaptureStatus(state.capture_loop)) return;\n    pillCaptureLoop = state.capture_loop;" in dashboard_js
    assert "if (!acceptCaptureStatus(msg)) return;\n      pillCaptureLoop = msg;" in dashboard_js
    assert "if (acceptCaptureStatus(state.capture_loop)) pillCaptureLoop = state.capture_loop;" in dashboard_js
    assert dashboard_js.count("if (acceptCaptureStatus(body)) pillCaptureLoop = body;") == 2
    assert "if (pillCaptureLoop && acceptCaptureStatus(msg)) pillCaptureLoop.starting = false;" in dashboard_js


def test_optimistic_start_state_actually_renders_as_starting(dashboard_js: str) -> None:
    """renderPill() checks `!cl.running` (Stopped) before `cl.starting`, so
    an optimistic `{running: false, starting: true}` rendered as Stopped."""
    assert "pillCaptureLoop = { running: false, starting: true };" not in dashboard_js


@pytest.mark.parametrize(
    "start_marker,end_marker",
    [
        ("getElementById('btn-refresh-calib').onclick", "getElementById('btn-start').onclick"),
        ("getElementById('btn-start').onclick", "getElementById('btn-stop').onclick"),
        ("getElementById('btn-stop').onclick", "// Reset -- POST /api/reset"),
    ],
)
def test_calibrate_start_stop_resync_status_when_they_finish(
    dashboard_js: str, start_marker: str, end_marker: str
) -> None:
    handler = dashboard_js[dashboard_js.index(start_marker) : dashboard_js.index(end_marker)]
    finally_block = handler[handler.rindex("} finally {") :]
    assert "await resyncCaptureLoop();" in finally_block
