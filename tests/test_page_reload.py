"""Open dashboards reload themselves when the server's page changes.

A page runs the JS it loaded, so after an update every open screen keeps the
old one until someone reloads it -- and a kiosk has nobody to. The server
fingerprints the page it serves and sends that both in the page's bootstrap
and on every WebSocket HELLO; a screen seeing a different one reloads.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opendarts.live import server as server_module
from opendarts.live.server import create_app

APP_JS = Path(__file__).resolve().parents[1] / "opendarts" / "live" / "dashboard" / "app.js"


def _bootstrap(html: str) -> dict:
    m = re.search(r'<script id="bootstrap" type="application/json">(.*?)</script>', html, re.S)
    assert m, "bootstrap block not found"
    return json.loads(m.group(1))


def test_the_page_and_every_hello_carry_the_same_fingerprint(tmp_path):
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    page = _bootstrap(client.get("/?ui=classic").text)["page_version"]
    assert page == server_module._DASHBOARD_PAGE_VERSION
    with client.websocket_connect("/api/events") as ws:
        hello = ws.receive_json()
    assert hello["type"] == "HELLO"
    assert hello["page_version"] == page


def test_the_fingerprint_moves_with_the_page_and_only_the_page():
    fp = server_module._page_fingerprint
    base = fp("<html>", "css", "js")
    assert base == fp("<html>", "css", "js"), "must be stable across restarts"
    assert base != fp("<html>", "css", "js2"), "a JS change must change it"
    assert base != fp("<html>", "css2", "js"), "a CSS change must change it"
    # Part boundaries count: moving text between files is a different page.
    assert fp("ab", "c") != fp("a", "bc")


def _function_source(js: str, name: str) -> str:
    start = js.index(f"function {name}(")
    depth = 0
    for i in range(js.index("{", start), len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[start:i + 1]
    raise AssertionError(f"unterminated function {name}")


def test_reload_rules_including_the_no_loop_guard():
    """Runs the SHIPPED function in node against a fake per-tab store."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed -- the reload rules were NOT exercised")
    script = _function_source(APP_JS.read_text(), "reloadIfPageIsStale") + """
console.warn = () => {};
const mem = {};
const store = {getItem: (k) => (k in mem ? mem[k] : null), setItem: (k, v) => { mem[k] = v; }};
let reloads = 0;
const reload = () => { reloads += 1; };
const r = [];
r.push(reloadIfPageIsStale('A', 'A', store, reload));   // same page: no
r.push(reloadIfPageIsStale(undefined, 'A', store, reload)); // old server, no field: no
r.push(reloadIfPageIsStale('B', undefined, store, reload)); // page cannot tell: no
r.push(reloadIfPageIsStale('B', 'A', store, reload));   // new build: YES
r.push(reloadIfPageIsStale('B', 'A', store, reload));   // stale again after reload: no loop
r.push(reloadIfPageIsStale('C', 'A', store, reload));   // a later build: YES
r.push(reloadIfPageIsStale('D', 'A', null, reload));    // no storage at all: still reloads
console.log(JSON.stringify({r, reloads}));
"""
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    got = json.loads(out.stdout)
    assert got["r"] == [False, False, False, True, False, True, True]
    assert got["reloads"] == 3


def test_hello_checks_the_page_before_rendering_anything():
    """Rendering first would paint new server state through old JS for a
    moment, and could throw on a field the old page does not know."""
    js = APP_JS.read_text()
    hello = js[js.index("if (msg.type === 'HELLO') {"):]
    assert hello.index("reloadIfPageIsStale(") < hello.index("renderState(msg.state)")
