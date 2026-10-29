"""The new dashboard (opendarts/live/ui, dev/ux/BRIEF.md), served at / when
the rig's switch says "new" (opendarts/live/dashboard_choice.py).

It is built on exactly the API the current dashboard uses, so these tests
guard the seams where a page can be wrong while every endpoint still
answers: the page assembles, its script parses, every element the script
looks up exists, a correction speaks the scorer's own vocabulary, and the
picker names the segment the scorer would.

The flows themselves -- start, darts, a correction, review, settings --
run in jsdom against a mock rig: `dev/ux/check_ui.py`.
"""
from __future__ import annotations

import json
import math
import random
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from opendarts.geometry import board as geometry
from opendarts.live import server as server_module
from opendarts.live import ui
from opendarts.live.server import board_ring_names, board_sectorless_rings, create_app


@pytest.fixture
def page(tmp_path) -> str:
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    resp = client.get("/?ui=new")
    assert resp.status_code == 200
    return resp.text


def _bootstrap(html: str) -> dict:
    m = re.search(r'<script id="bootstrap" type="application/json">(.*?)</script>', html, re.S)
    assert m, "bootstrap block not found"
    return json.loads(m.group(1))


def test_the_page_assembles_with_every_placeholder_filled(page):
    assert not re.findall(r"@@OD_[A-Z_]+@@", page)
    boot = _bootstrap(page)
    assert boot["cam_ids"] == [0, 1, 2]
    assert boot["host_label"]
    assert boot["board_sectors"][0] == "20"


def test_both_pages_share_one_fingerprint_that_covers_the_new_one(page, tmp_path):
    """HELLO carries one page_version. If the new page were not part of it,
    an update to it would never reload an open screen -- and if it had its
    own, every such screen would reload on every HELLO."""
    assert _bootstrap(page)["page_version"] == server_module._DASHBOARD_PAGE_VERSION
    expected = server_module._page_fingerprint(
        server_module._DASHBOARD_INDEX_HTML, server_module._DASHBOARD_APP_CSS,
        server_module._DASHBOARD_APP_JS, *ui.FINGERPRINT_PARTS)
    assert server_module._DASHBOARD_PAGE_VERSION == expected


def _is_new(html: str) -> bool:
    return '"role": "control"' in html


def _is_classic(html: str) -> bool:
    return 'id="btn-start"' in html


def test_the_switch_decides_what_slash_serves(tmp_path):
    """One address; the rig's switch picks the page. New by default."""
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    assert _is_new(client.get("/").text)
    assert client.get("/api/dashboard").json()["ui"] == "new"
    assert client.put("/api/dashboard", json={"ui": "classic"}).json() == {"ok": True, "ui": "classic"}
    assert _is_classic(client.get("/").text)
    assert client.put("/api/dashboard", json={"ui": "new"}).status_code == 200
    assert _is_new(client.get("/").text)
    assert client.put("/api/dashboard", json={"ui": "fancy"}).status_code == 400


def test_one_screen_can_peek_at_the_other_without_flipping_it(tmp_path):
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    assert _is_classic(client.get("/?ui=classic").text)
    assert _is_new(client.get("/").text), "a peek never flips the switch"
    client.put("/api/dashboard", json={"ui": "classic"})
    assert _is_new(client.get("/?ui=new").text)


def test_a_display_is_the_new_page_whatever_the_switch_says(tmp_path):
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    client.put("/api/dashboard", json={"ui": "classic"})
    assert '"role": "display"' in client.get("/?display=tv").text
    assert '"role": "display"' in client.get("/?display").text


def test_flipping_it_tells_every_open_screen(tmp_path):
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    with client.websocket_connect("/api/events") as ws:
        assert ws.receive_json()["type"] == "HELLO"
        client.put("/api/dashboard", json={"ui": "classic"})
        msg = ws.receive_json()
        while msg["type"] != "DASHBOARD_SWITCHED":
            msg = ws.receive_json()
        assert msg["ui"] == "classic"


def test_the_switch_survives_a_restart_and_keeps_the_rest_of_config(tmp_path):
    import json as _json

    from opendarts.live.dashboard_choice import DashboardChoice

    path = tmp_path / "config.json"
    path.write_text(_json.dumps({"port": 8420}))
    DashboardChoice(snapshot_path=path).set("classic")
    assert DashboardChoice(snapshot_path=path).get() == "classic"
    assert _json.loads(path.read_text())["port"] == 8420


def test_there_are_no_side_paths_for_the_new_page(tmp_path):
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    assert client.get("/next").status_code == 404
    assert client.get("/next/display").status_code == 404


def test_the_script_parses(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed -- the new dashboard's script was NOT parsed")
    js = tmp_path / "next.js"
    js.write_text(ui.JS, encoding="utf-8")
    res = subprocess.run([node, "--check", str(js)], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_every_element_the_script_looks_up_exists():
    """A renamed id fails silently in the browser -- `$('#x')` is null and
    the handler that used it throws on first click. Literal ids only; ids
    built at runtime (per camera) are created by the script itself."""
    html_ids = set(re.findall(r'\bid="([\w-]+)"', ui.INDEX_HTML))
    # ...plus the elements the script builds itself, e.g. h('canvas', {id: 'x'})
    html_ids |= set(re.findall(r"\bid: '([\w-]+)'", ui.JS))
    wanted = set(re.findall(r"""\$\(\s*'#([\w-]+)'\s*\)""", ui.JS))
    wanted |= set(re.findall(r"""getElementById\(\s*'([\w-]+)'\s*\)""", ui.JS))
    missing = sorted(wanted - html_ids)
    assert not missing, f"the script looks these up but index.html has no such id: {missing}"


def test_a_correction_speaks_the_scorers_vocabulary():
    """A recorded truth spelled differently from what the scorer produces
    could never compare equal to any engine's answer -- it would be a
    correction nothing can ever match."""
    # \b: "ring:" as a key of its own, not the tail of e.g. "scoring:"
    rings_used = set(re.findall(r"\bring: '([a-z_]+)'", ui.JS))
    assert rings_used, "no ring literals found -- this test has rotted"
    assert rings_used <= set(board_ring_names()), sorted(rings_used - set(board_ring_names()))
    m = re.search(r"const SECTORLESS = new Set\(\[([^\]]*)\]\)", ui.JS)
    assert m, "truth.js no longer declares SECTORLESS"
    declared = set(re.findall(r"'([a-z_]+)'", m.group(1)))
    assert declared == board_sectorless_rings()


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


def test_the_picker_names_the_segment_the_scorer_would(tmp_path):
    """Runs the SHIPPED hitTest in node against geometry.sector_ring_for_point.

    The picker hit-tests the inner ring edges at the scorer's own default
    scoring radii (regulation minus INNER_RING_SCORING_OFFSET_MM; the
    page takes the live values from OD.board_scoring_mm). Points within
    2 mm of any edge are still skipped: a tap there is a tap on a wire,
    which the ring chips beside the board settle. Everything else --
    which wedge, which band, every ring name -- must agree exactly."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed -- picker/scorer parity was NOT checked")
    edges = [geometry.BULL_RADIUS_MM, geometry.OUTER_BULL_RADIUS_MM, geometry.TREBLE_INNER_RADIUS_MM,
             geometry.TREBLE_OUTER_RADIUS_MM, geometry.DOUBLE_INNER_RADIUS_MM, geometry.DOUBLE_OUTER_RADIUS_MM,
             geometry.TREBLE_INNER_SCORING_RADIUS_MM, geometry.DOUBLE_INNER_SCORING_RADIUS_MM]
    rng = random.Random(7)
    points = []
    while len(points) < 3000:
        r = rng.uniform(0, 200)
        a = rng.uniform(0, 2 * math.pi)
        if any(abs(r - e) < 2.0 for e in edges):
            continue
        x, y = r * math.cos(a), r * math.sin(a)
        # ...and away from a wedge wire. Wedges are centred on multiples of
        # 18 degrees (clockwise from the top), so the wires sit at 9 mod 18.
        ang = math.degrees(math.atan2(x, y)) % 18.0
        if r > geometry.OUTER_BULL_RADIUS_MM and abs(ang - 9.0) < 0.8:
            continue
        points.append((x, y))
    board_mm = re.search(r"const BOARD_MM = \{.*?\};", ui.JS, re.S).group(0)
    script = (
        "const BOARD_SECTORS = " + json.dumps([str(n) for n in geometry.SECTOR_NUMBERS_CLOCKWISE]) + ";\n"
        + board_mm + "\n" + _function_source(ui.JS, "hitTest") + "\n"
        + "const pts = " + json.dumps(points) + ";\n"
        + "console.log(JSON.stringify(pts.map(([x, y]) => hitTest(x, y))));\n"
    )
    f = tmp_path / "hit.js"
    f.write_text(script, encoding="utf-8")
    res = subprocess.run([node, str(f)], capture_output=True, text=True, check=True)
    got = json.loads(res.stdout)
    mismatches = []
    for (x, y), js in zip(points, got):
        sector, ring = geometry.sector_ring_for_point(x, y)
        if (js["sector"], js["ring"]) != (sector, ring):
            mismatches.append(((round(x, 2), round(y, 2)), (sector, ring), (js["sector"], js["ring"])))
    assert not mismatches, mismatches[:10]


def test_nothing_on_the_page_reaches_off_the_rig(page):
    """One document, no CDN: a rig may have no internet, and every extra
    request competes with the camera streams for the socket budget."""
    assert not re.findall(r'(?:src|href)="https?://', page)
    assert "@import" not in ui.CSS
    assert "fonts.googleapis" not in page


def test_the_rig_has_an_icon_and_every_page_links_it(tmp_path):
    client = TestClient(create_app(package_root=tmp_path, enable_background_poll=False))
    for path, kind in (("/favicon.ico", "image/x-icon"), ("/favicon.svg", "image/svg+xml"),
                       ("/apple-touch-icon.png", "image/png")):
        resp = client.get(path)
        assert resp.status_code == 200 and resp.headers["content-type"].startswith(kind) and resp.content, path
    ico = client.get("/favicon.ico").content
    assert ico[:4] == b"\x00\x00\x01\x00" and ico[4] == 3, "an .ico holding 16, 32 and 48 px"
    for page in ("/", "/?ui=classic", "/?display=tv"):
        assert 'href="/favicon.svg"' in client.get(page).text, page


def test_there_is_no_kiosk_mode_a_tv_is_a_display(tmp_path):
    """?kiosk was the first design's wall-TV switch, replaced by /?display."""
    assert "has('kiosk')" not in ui.JS and "body.kiosk" not in ui.CSS


def test_frame_urls_carry_the_throws_version():
    """A throw id can be reused after delete-all; the dashboard versions its
    frame URLs by capture time so a kept frame is never another dart's."""
    src = re.search(r"function frameUrl\(p, cam, which\) \{.*?\n  \}", ui.JS, re.S).group(0)
    assert "&v=" in src and "captured_at_utc" in src


def test_an_oracle_count_mismatch_is_said_on_the_scoreboard():
    """opendarts/live/oracle_watch.py: a dart the oracle saw and OD did not
    must be obvious, on every screen, without the oracle scoring anything."""
    assert 'id="oracle-miss"' in ui.INDEX_HTML
    assert 'id="oracle-miss-lab"' in ui.INDEX_HTML    # the Engines view, where the TV sits while testing
    assert "case 'ORACLE_MISMATCH'" in ui.JS and "st.oracle_mismatches" in ui.JS
    fn = re.search(r"function oracleMismatch\(m, fromState\) \{.*?\n  \}", ui.JS, re.S).group(0)
    assert "S.visit" not in fn          # a notice, never a change to the visit


def test_status_notices_float_and_never_move_the_page():
    """rla: "I hate when things shift." The status strip and the notices
    lie over the page; none of them pads, pushes or shrinks it."""
    assert "padding-top: var(--conn-h" not in ui.CSS
    assert "has-miss" not in ui.CSS and "has-miss" not in ui.JS
    assert re.search(r"\.throws-notices \{ position: absolute;", ui.CSS)
    assert re.search(r"\.omiss \{ position: absolute;", ui.CSS)
    # the notices live outside the flowing header and scoreboard
    head = ui.INDEX_HTML.split('<header class="throws-head">', 1)[1].split("</header>", 1)[0]
    assert 'id="throws-banner"' not in head and 'id="oracle-miss-lab"' not in head
    turn = ui.INDEX_HTML.split('<div class="turn" id="turn">', 1)[1].split("</aside>", 1)[0]
    assert 'id="oracle-miss"' not in turn


def test_the_board_is_painted_from_the_scoring_edges():
    """A dart the scorer called D1 must be drawn in the green: the bands are
    painted and hit-tested from the inner SCORING radii, and the page is
    told the live ones."""
    from opendarts.live import server as srv
    boot = srv._dashboard_bootstrap([0, 1, 2])
    assert boot["board_scoring_mm"] == {"treble_inner": geometry.TREBLE_INNER_SCORING_RADIUS_MM,
                                        "double_inner": geometry.DOUBLE_INNER_SCORING_RADIUS_MM}
    board_mm = re.search(r"const BOARD_MM = \{.*?\};", ui.JS, re.S).group(0)
    assert f"doubleInnerScore: {geometry.DOUBLE_INNER_SCORING_RADIUS_MM}" in board_mm
    assert f"trebleInnerScore: {geometry.TREBLE_INNER_SCORING_RADIUS_MM}" in board_mm
    paint = _function_source(ui.JS, "drawDiagram")
    assert "mm(BOARD_MM.doubleInnerScore), mm(BOARD_MM.doubleOuter)" in paint
    assert "mm(BOARD_MM.doubleInner)," not in paint
