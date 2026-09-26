"""The Scoring tab's Photo view: the dart drawing and its wiring.

The geometry runs the SHIPPED functions in node, the way the sound and
page-reload tests do; the wiring is pinned by reading app.js, since it is
the one thing a refactor could quietly drop.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "opendarts" / "live" / "dashboard"
APP_JS = ROOT / "app.js"
INDEX = ROOT / "index.html"


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


def _run(script: str) -> object:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed -- the dart drawing was NOT exercised")
    js = APP_JS.read_text()
    src = "\n".join(_function_source(js, n) for n in ("ochePoint", "norm3", "cross3", "dartSilhouette"))
    out = subprocess.run([node, "-e", src + "\n" + script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_the_board_plane_maps_to_itself():
    """The photo is the board plane, so a point ON the board must land on
    itself -- otherwise darts would drift off the spot they scored."""
    got = _run("console.log(JSON.stringify([ochePoint([40, -60, 0], [120, -50, 2370]),"
               " ochePoint([0, 0, 100], [0, 0, 2370])]));")
    assert got[0][0] == pytest.approx(40) and got[0][1] == pytest.approx(-60) and got[0][2] == pytest.approx(1)
    assert got[1][2] > 1, "a point nearer the eye looks larger"


def test_a_dart_starts_at_its_tip_and_leans_the_way_it_went_in():
    got = _run("""
const eye = [120, -50, 2370];
const up = dartSilhouette([30, 80], [0, 0.2, 0.98], 45, eye, 3);
const right = dartSilhouette([30, 80], [0.2, 0, 0.98], 45, eye, 3);
const straight = dartSilhouette([30, 80], null, 45, eye, 3);
const tail = (s) => s.cap.b;
console.log(JSON.stringify({
  tip: up.shaft.find((p) => p.part === 'point').a,
  up: tail(up), right: tail(right), straight: tail(straight),
  vanes: up.vanes.length, depths: up.vanes.map((v) => v.depth),
}));
""")
    assert got["tip"][0] == pytest.approx(30) and got["tip"][1] == pytest.approx(80)
    assert got["up"][1] - 80 > 25, "a dart leaning up draws its tail above its tip"
    assert got["right"][0] - 30 > 25, "a dart leaning right draws its tail to the right"
    # straight in: the tail (112 mm out) is pushed off the tip only by where
    # the eye is -- exactly the perspective shift, nothing more
    k = 2370 / (2370 - 112)
    assert got["straight"][0] == pytest.approx(120 + (30 - 120) * k)
    assert got["straight"][1] == pytest.approx(-50 + (80 + 50) * k)
    assert got["vanes"] == 4
    assert got["depths"] == sorted(got["depths"]), "nearest vane last, so it is drawn on top"


def test_the_photo_view_is_wired_in():
    js = APP_JS.read_text()
    html = INDEX.read_text()
    # both sources of the current version feed the loader
    render_state = js[js.index("function renderState("):]
    assert "setBoardPhotoVersion(state.visit.board_photo_version)" in render_state
    photo_msg = js[js.index("msg.type === 'BOARD_PHOTO'"):]
    assert photo_msg.index("setBoardPhotoVersion(msg.version)") < photo_msg.index("} else if")
    # the drawn board falls back to the diagram until a photo has loaded
    draw = _function_source(js, "drawBoard")
    assert "boardView === 'photo' && boardPhotoImg" in draw
    # the switch exists, and the choice is remembered per screen
    assert 'id="board-view-switch"' in html
    assert 'data-view="photo"' in html and 'data-view="diagram"' in html
    assert "localStorage.setItem(BOARD_VIEW_KEY" in _function_source(js, "setBoardView")
    # darts are drawn in their own flight colour when one was measured
    assert "t.flight_color || DEFAULT_FLIGHT_COLOR" in _function_source(js, "drawPhotoBoard")


def test_the_photo_scale_matches_the_server():
    from opendarts.live import board_photo

    js = APP_JS.read_text()
    assert f"const PHOTO_HALF_MM = {board_photo.HALF_MM};" in js
