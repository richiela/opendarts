"""The Config tab's Autodarts note must follow the real connection without
a page refresh, in every open tab.

REPORTED 2026-09-22: switching "Compare against Autodarts" to Yes showed
"On but NOT connected ... no comparison is being recorded." in red, and it
stayed red until a reload, which showed "On and connected".

Root cause: the note is drawn from `runtime.ad.connected` of the config
document the tab last received. The PATCH reply is built the instant the
listener is started -- before its socket has connected -- so it
truthfully says `connected: false`. Nothing ever told the page about the
connect that followed: the listener announced board-status changes only
(AD_BOARD_STATUS), never connect/disconnect, and nothing re-reads the
config document on a timer.

The fix: the listener announces every connection transition
(on_connection_change), the server broadcasts a stamped AD_CONNECTION
built from the LIVE is_connected(), and the dashboard applies it to
`runtime.ad` and redraws the note. Because the push and the PATCH reply
travel on different channels, every `runtime.ad` snapshot carries a
{connection_epoch, connection_seq} stamp and the dashboard drops any
snapshot older than the one it holds -- the same guard the status pill got
(tests/test_capture_status_ordering.py).
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
    return tmp_path / "run" / "packages"


class _Listener:
    """AdWsListener stand-in: the surface the config document touches."""

    def __init__(self):
        self.base_url = "http://ad-box:3180"
        self._enabled = False
        self.connected = False

    def is_enabled(self):
        return self._enabled

    def is_connected(self):
        return self.connected

    def oracle_base_url(self):
        return self.base_url if self._enabled else None

    def set_enabled(self, v):
        self._enabled = bool(v)

    def set_base_url(self, url):
        self.base_url = url.strip().rstrip("/")


@pytest.fixture
def no_config_writes(monkeypatch):
    written = {}

    def _fake(section, value, path=None):
        written[section] = value

    monkeypatch.setattr("opendarts.live.config_document.write_config_section", _fake)
    monkeypatch.setattr("opendarts.live.config_document.stored_value", written.get)
    return written


def _app(package_root, listener):
    return create_app(package_root=package_root, enable_background_poll=False,
                      ad_ws_listener=listener)


def _recording_broadcast(state):
    sent: list[dict] = []

    async def _broadcast(msg):
        sent.append(msg)

    state._broadcast = _broadcast  # noqa: SLF001
    return sent


# -- server side --------------------------------------------------------


def test_ad_connection_event_broadcasts_the_live_value_stamped(package_root):
    """The broadcast reports what is true when it is SENT, not what the
    queued event claimed: two transitions announced from two threads can
    reach the queue in either order, and reading live makes the last
    broadcast always the final state."""
    listener = _Listener()
    state = _app(package_root, listener).state.opendarts_state
    sent = _recording_broadcast(state)

    listener.connected = True
    asyncio.run(state._handle_live_event({"type": "AD_CONNECTION", "connected": False}))  # noqa: SLF001
    listener.connected = False
    asyncio.run(state._handle_live_event({"type": "AD_CONNECTION", "connected": True}))  # noqa: SLF001

    assert [m["type"] for m in sent] == ["AD_CONNECTION", "AD_CONNECTION"]
    assert [m["connected"] for m in sent] == [True, False]
    assert all(m["available"] is True for m in sent)
    assert sent[0]["connection_seq"] < sent[1]["connection_seq"]
    assert sent[0]["connection_epoch"] == sent[1]["connection_epoch"]


def test_patch_reply_is_older_than_the_connect_push_that_follows_it(package_root, no_config_writes):
    """The reported sequence: enable -> PATCH reply says not connected ->
    socket connects -> push. The push must carry a HIGHER stamp than the
    reply, so a tab that receives them in the opposite order can still
    tell which one is newer."""
    listener = _Listener()
    app = _app(package_root, listener)
    state = app.state.opendarts_state
    sent = _recording_broadcast(state)
    client = TestClient(app)

    body = client.patch("/api/config", json={"ad_enabled": True}).json()
    assert body["ok"] is True and body["config"]["ad_enabled"] is True
    reply_ad = body["runtime"]["ad"]
    assert reply_ad["connected"] is False

    listener.connected = True  # the listener thread's connect
    asyncio.run(state._handle_live_event({"type": "AD_CONNECTION", "connected": True}))  # noqa: SLF001
    push = [m for m in sent if m.get("type") == "AD_CONNECTION"][-1]
    assert push["connected"] is True
    assert push["connection_epoch"] == reply_ad["connection_epoch"]
    assert push["connection_seq"] > reply_ad["connection_seq"]

    # And a later GET agrees and is newer still.
    got = client.get("/api/config").json()["runtime"]["ad"]
    assert got["connected"] is True and got["connection_seq"] > push["connection_seq"]


def test_no_listener_push_is_still_honest(package_root):
    state = create_app(package_root=package_root,
                       enable_background_poll=False).state.opendarts_state
    sent = _recording_broadcast(state)
    asyncio.run(state._handle_live_event({"type": "AD_CONNECTION", "connected": True}))  # noqa: SLF001
    assert sent[-1]["available"] is False and sent[-1]["connected"] is False


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


def _run_node(dashboard_js: str, scenario: str):
    """Runs the SHIPPED functions in node with every other renderer stubbed.
    Skipped where node is absent."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed -- the client behaviour was NOT exercised")
    stubs = "".join(
        f"function {n}() {{}}\n" for n in (
            "renderPortPanel", "renderStorePackagesPanel", "renderFrameRingPanel",
            "renderCameraDevices", "cameraDevicePayload", "renderIdleTimeoutSelect",
            "renderDetectionTimeSelect", "renderAlwaysUpdate", "renderMaintenanceNote",
        )
    )
    script = (
        "let lastConfig = null; let lastConfigRuntime = null; let configRestartRequired = [];\n"
        "let adConnectionEpoch = null; let adConnectionSeq = -1;\n"
        "let adRenders = 0; function renderAdConfig() { adRenders++; }\n"
        + stubs
        + "\n".join(_function_source(dashboard_js, n) for n in (
            "configRuntime", "acceptAdConnection", "applyAdConnection", "applyConfigDocument",
        ))
        + "\nconst doc = (seq, connected, extra) => ({ok: true, config: Object.assign({ad_enabled: true}, extra || {}),"
          " runtime: {port: {active: 1}, ad: {available: true, connected: connected,"
          " connection_epoch: 'e', connection_seq: seq}}});\n"
        + "const push = (seq, connected, epoch) => ({type: 'AD_CONNECTION', available: true,"
          " connected: connected, connection_epoch: epoch || 'e', connection_seq: seq});\n"
        + "const out = [];\nconst ad = () => configRuntime('ad').connected;\n"
        + scenario
        + "\nconsole.log(JSON.stringify(out));"
    )
    res = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def test_push_after_patch_turns_the_note_green(dashboard_js):
    """The ordinary order: PATCH reply (not yet connected), then the push."""
    out = _run_node(dashboard_js, """
applyConfigDocument(doc(4, false)); out.push(ad());
const before = adRenders;
applyAdConnection(push(5, true)); out.push(ad(), adRenders > before);
""")
    assert out == [False, True, True]


def test_push_that_overtakes_the_patch_reply_is_not_undone_by_it(dashboard_js):
    """The race: the connect push lands first, then this tab's own older
    PATCH reply. The reply's settings still apply; its stale
    `connected: false` must not."""
    out = _run_node(dashboard_js, """
applyConfigDocument(doc(1, false, {ad_base_url: 'http://old'}));
applyAdConnection(push(5, true)); out.push(ad());
applyConfigDocument(doc(4, false, {ad_base_url: 'http://new'}));
out.push(ad(), lastConfig.ad_base_url, lastConfigRuntime.port.active);
""")
    assert out == [True, True, "http://new", 1]


def test_push_before_the_first_document_is_kept(dashboard_js):
    """A push can beat the tab's very first GET, whose runtime.ad was read
    on the server earlier. The push's fact must survive that GET."""
    out = _run_node(dashboard_js, """
applyAdConnection(push(7, true));
applyConfigDocument(doc(6, false)); out.push(ad());
""")
    assert out == [True]


def test_disconnect_push_turns_the_note_back_red_and_restart_resets(dashboard_js):
    out = _run_node(dashboard_js, """
applyConfigDocument(doc(4, false));
applyAdConnection(push(5, true)); out.push(ad());
applyAdConnection(push(6, false)); out.push(ad());
out.push(applyAdConnection(push(5, true)), ad());   // stale push: dropped
applyAdConnection(push(1, true, 'restarted')); out.push(ad());  // new process
""")
    assert out == [True, False, False, False, True]


def test_ad_connection_push_is_wired_into_the_events_socket(dashboard_js):
    handler = dashboard_js[dashboard_js.index("msg.type === 'AD_CONNECTION'"):]
    assert handler[: handler.index("}")].strip().endswith("applyAdConnection(msg);")
