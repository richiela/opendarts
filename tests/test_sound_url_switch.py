"""The kiosk sound switch: `?sound=on` / `?sound=off` in the dashboard URL.

Sound is per screen and each screen opts in with its own toggle, which a
kiosk -- a TV run by a browser nobody can touch -- cannot press. The URL is
the one thing whoever sets up a kiosk controls, so it may carry the choice.
These run the SHIPPED parser in node rather than a Python copy of it, and
pin that the choice is applied before the on-load arm that plays sound.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).resolve().parents[1] / "opendarts" / "live" / "dashboard" / "app.js"


@pytest.fixture(scope="module")
def app_js() -> str:
    return APP_JS.read_text()


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


def test_sound_from_url_reads_on_off_and_ignores_the_rest(app_js: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed -- the URL parser's behaviour was NOT exercised")
    cases = [
        "", "?", "#engines",                       # absent -> null
        "?sound=on", "?sound=ON", "?sound=1", "?sound=yes", "?sound=true",
        "?sound=off", "?sound=0", "?sound=no", "?sound=false",
        "?sound=loud", "?sound=",                  # garbled -> null, never a guess
        "?other=1&sound=on",                       # alongside other params
    ]
    script = (
        _function_source(app_js, "soundFromUrl")
        + "\nconst warn = console.warn; console.warn = () => {};"
        + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(soundFromUrl)));"
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == [
        None, None, None,
        True, True, True, True, True,
        False, False, False, False,
        None, None,
        True,
    ]


def test_the_url_choice_is_applied_before_sound_is_armed_on_load(app_js: str) -> None:
    """The on-load `armAudio()` only runs if the setting is already on, so the
    URL must be read, and saved, BEFORE that check -- otherwise a kiosk
    loading `?sound=on` would stay silent until its next reload."""
    apply_at = app_js.index("soundFromUrl(window.location.search)")
    arm = re.search(r"if \(audioSettings\.enabled\) \{\s*armAudio\(\);", app_js)
    assert arm, "on-load arm not found"
    assert apply_at < arm.start()
    block = app_js[apply_at:arm.start()]
    assert "audioSettings.enabled = fromUrl" in block
    assert "writeAudioSettings()" in block, "must be saved like the toggle saves it"


def test_a_screen_that_never_opened_config_still_picks_a_voice_to_load(app_js: str) -> None:
    """The kiosk bug. At page load the voice catalog has not been fetched,
    so the synchronous audioChosenVoice() answers '' and the clips were
    never loaded -- the screen said "sound on, running" and played nothing.
    Runs the shipped functions in node against a catalog that only arrives
    when fetched, the way it does on a screen that loads straight onto the
    Engines tab."""
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed -- voice selection was NOT exercised")
    script = (
        "let audioCatalog = null;\n"
        "const audioSettings = {voice: ''};\n"
        "async function fetchAudioCatalog(force) {\n"
        "  if (audioCatalog && !force) return audioCatalog;\n"
        "  audioCatalog = {default: 'Bella', voices: [{voice: 'Bella'}, {voice: 'Adam'}]};\n"
        "  return audioCatalog;\n"
        "}\n"
        + _function_source(app_js, "audioChosenVoice") + "\n"
        # _function_source starts at `function`, so the `async` in front of
        # this one has to be put back or node rejects its `await`.
        + "async " + _function_source(app_js, "audioChosenVoiceLoaded") + "\n"
        + "const before = audioChosenVoice();\n"
          "audioChosenVoiceLoaded().then((v) => console.log(JSON.stringify([before, v])));\n"
    )
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == ["", "Bella"]


def test_every_path_that_loads_clips_waits_for_the_catalog(app_js: str) -> None:
    """One path left on the old pattern is enough to bring the silence back."""
    assert "loadVoiceSet(audioChosenVoice())" not in app_js
    arm = _function_source(app_js, "armAudio")
    assert "await audioChosenVoiceLoaded()" in arm
