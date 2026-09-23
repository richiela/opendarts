"""The rendered dashboard's JavaScript must be structurally valid.

WHY THIS EXISTS. The dashboard's script (opendarts/live/dashboard/app.js,
one large file the server inlines into the page) is edited by text
substitution. Three separate times a substitution has silently left it
unbalanced -- an orphaned function body, a duplicated block, a stray
brace. Every one produced a page that loaded, rendered, and then died at
the first JavaScript error, taking every interactive control with it
including the tab switcher. Nothing in the suite noticed: the HTML was
still well-formed and every endpoint still answered.

A brace/bracket/paren balance check is approximate -- it cannot prove the
script runs -- but it catches exactly the failure mode that keeps
happening, which is text surgery leaving the structure unclosed.
"""
from __future__ import annotations

import pytest

from opendarts.live.server import _render_dashboard_html


def _extract_script(html: str) -> str:
    assert "<script>" in html, "dashboard has no inline script"
    return html.split("<script>")[-1].split("</script>")[0]


def _strip_literals_and_comments(js: str) -> str:
    """Blank out strings, template literals, regexes and comments.

    Counting delimiters without this is meaningless -- a brace inside a
    string is not structure. Regex literals are detected by what precedes
    them, since `/` is ambiguous between division and a literal.
    """
    out = []
    i, n = 0, len(js)
    prev_significant = ""
    while i < n:
        c = js[i]
        nxt = js[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            while i < n and js[i] != "\n":
                i += 1
            continue
        if c == "/" and nxt == "*":
            i += 2
            while i < n - 1 and not (js[i] == "*" and js[i + 1] == "/"):
                i += 1
            i += 2
            continue
        if c in "\"'`":
            quote = c
            i += 1
            while i < n:
                if js[i] == "\\":
                    i += 2
                    continue
                if js[i] == quote:
                    i += 1
                    break
                i += 1
            prev_significant = "x"          # a literal is a value
            continue
        if c == "/" and prev_significant in ("", "(", ",", "=", ":", "[", "!", "&",
                                             "|", "?", "{", "}", ";", "\n"):
            # Regex literal, not division.
            i += 1
            while i < n:
                if js[i] == "\\":
                    i += 2
                    continue
                if js[i] == "/":
                    i += 1
                    break
                if js[i] == "\n":
                    break
                i += 1
            prev_significant = "x"
            continue
        out.append(c)
        if not c.isspace():
            prev_significant = c
        i += 1
    return "".join(out)


@pytest.mark.parametrize("n_cameras", [1, 3])
def test_dashboard_javascript_delimiters_balance(n_cameras):
    js = _strip_literals_and_comments(_extract_script(_render_dashboard_html(n_cameras)))
    pairs = {"}": "{", ")": "(", "]": "["}
    stack = []
    line = 1
    for ch in js:
        if ch == "\n":
            line += 1
        elif ch in "{([":
            stack.append((ch, line))
        elif ch in pairs:
            assert stack, f"unmatched closing {ch!r} at line ~{line} with nothing open"
            opener, opened_at = stack.pop()
            assert opener == pairs[ch], (
                f"mismatched {ch!r} at line ~{line}: closes {opener!r} opened at ~{opened_at}"
            )
    assert not stack, (
        "unclosed delimiters left open at lines "
        + ", ".join(f"{o!r}@~{ln}" for o, ln in stack[:5])
    )


def test_dashboard_javascript_has_no_unrendered_fstring_braces():
    """The script spent years inside a Python f-string, where every brace
    had to be written doubled. A `{{`/`}}` surviving into the output means
    one of those doublings was carried into a plain file by hand -- the
    browser then sees a literal doubled brace."""
    js = _extract_script(_render_dashboard_html(3))
    assert "{{" not in js and "}}" not in js


def test_every_dashboard_function_is_defined_exactly_once():
    """A duplicated top-level declaration is a hard SyntaxError, and text
    substitution duplicating a block is the way it has happened."""
    import re

    js = _strip_literals_and_comments(_extract_script(_render_dashboard_html(3)))
    for kind in ("function", "const", "let"):
        names = re.findall(rf"^{kind} ([A-Za-z_$][\w$]*)", js, re.M)
        dupes = sorted({x for x in names if names.count(x) > 1})
        assert not dupes, f"top-level {kind} declared more than once: {dupes}"


# The syntax check above cannot catch a DELETED block: removing balanced
# code leaves perfectly valid JavaScript. That has now happened twice --
# a substitution for one feature silently removed another that sat between
# its anchors, and the page kept loading with a control that simply did
# nothing. This pins the handlers that must exist.
EXPECTED_FUNCTIONS = (
    # The config document (2026-09-17): one GET populates the whole tab
    # and one PATCH saves a change, so deleting any of these three leaves
    # EVERY control on the tab silently inert rather than just one.
    "refreshConfig",
    "applyConfigDocument",
    "patchConfig",
    "renderPortPanel",
    "renderAlwaysUpdate",
    # Autodarts comparison -- deleted once by a camera-selector edit.
    "renderAdConfig",
    "submitAdConfig",
    # Camera assignment.
    "renderCameraDevices",
    "cameraDeviceOptionsHtml",
    "setCameraDeviceOptions",
    "cameraDevicePayload",
    "applyCameraDevices",
    # Panels that have been spliced around.
    "renderDetectionTimeSelect",
    "refreshAudioPanel",
    "renderStorePackagesPanel",
    "renderFrameRingPanel",
    "refreshVoicePanel",
    # Spoken dart calls, which became a browser feature on 2026-09-15.
    # Pinned by name because this whole feature is silent when it breaks:
    # deleting any one of these leaves valid JavaScript and a dashboard
    # that scores perfectly and never speaks. `resumeAudioContext` and
    # `unlockAudioSession` are the two that matter most: the first is the
    # autoplay-blocked path, the second is the iOS silent-switch path, and
    # between them they are why a page can make a sound at all.
    "ensureAudioContext",
    "resumeAudioContext",
    "loadVoiceSet",
    "armAudio",
    "speakPhrase",
    "renderAudioPanel",
    "unlockAudioSession",
    "reportAudioState",
    "refreshAudioClients",
    # setRenderProgress / pollRenderProgress were deleted with the
    # render pipeline on 2026-09-15 -- clip sets are shipped, so
    # nothing renders on a rig and there is no progress to poll.
    # Live MJPEG camera previews (2026-09-11) -- the whole feature is
    # these three functions; deleting any one leaves tiles that either
    # never show video, never refresh the overlay, or never release a
    # stream connection.
    "updateCameraFeeds",
    # pollOverlayImages was replaced 2026-09-12 by these two: the overlay
    # is a transparent LAYER over the stream now, refreshed when the
    # calibration changes rather than every 3 seconds. Pinned by name for
    # the same reason its predecessor was -- deleting either one leaves
    # valid JS and a tile that silently never shows an overlay again.
    "refreshCalibrationOverlays",
    "hideCalibrationOverlay",
    "stopCamStream",
)


@pytest.mark.parametrize("name", EXPECTED_FUNCTIONS)
def test_dashboard_defines_expected_handler(name):
    import re

    js = _strip_literals_and_comments(_extract_script(_render_dashboard_html(3)))
    found = re.findall(rf"^(?:async )?function {name}\(", js, re.M)
    assert len(found) == 1, (
        f"{name} is defined {len(found)} times -- 0 means an edit deleted it "
        "(valid JS, dead control); more than 1 is a duplicate declaration"
    )


ELEMENT_HANDLER_PAIRS = (
    # (element id in the markup, function the page must call for it)
    ("ad-enabled-toggle", "submitAdConfig"),
    ("ad-base-url", "submitAdConfig"),
    ("cam-device-0", "applyCameraDevices"),
    ("store-packages-select", "renderStorePackagesPanel"),
    ("server-port", "patchConfig"),
    ("always-update-toggle", "patchConfig"),
    ("audio-voice-select", "refreshVoicePanel"),
    ("audio-enabled-select", "armAudio"),
    ("audio-volume", "writeAudioSettings"),
)


@pytest.mark.parametrize("element_id, handler", ELEMENT_HANDLER_PAIRS)
def test_interactive_element_has_a_handler_that_exists(element_id, handler):
    """An input rendered with nothing wired to it looks completely normal
    and does nothing at all -- the exact failure this suite kept missing."""
    html = _render_dashboard_html(3)
    assert f'id="{element_id}"' in html, f"{element_id} is not in the markup"
    js = _strip_literals_and_comments(_extract_script(html))
    assert handler in js, f"{element_id} exists but {handler} does not"


def test_rendered_script_parses_under_a_real_javascript_engine():
    """The balance check above is approximate and PROVABLY misses things.

    It passed on a script containing `lines.join('` with an unterminated
    string literal -- a `\\n` written into the f-string the script used to
    live in, which Python turned into a REAL newline before the browser ever
    saw it. That particular trap is gone now that app.js is a plain file,
    but the check is not: stripping string literals is exactly what hides an
    unterminated one, however it got there.

    `node --check` is a real parser and caught it immediately. Skipped
    rather than failed where node is absent, so it strengthens CI without
    becoming a new dependency.

    **A SKIP HERE IS A REAL LOSS OF COVERAGE, not a neutral one.** Verified
    independently by a second session: a real newline injected into a JS
    string literal fails THIS test and passes all 28 others in this file.
    Nothing else catches it, because every other check runs on output that
    has had string literals blanked. On a machine without node this file
    goes green while blind to the exact failure mode it exists for.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        # NOT "fine here". This is the ONLY check in this file that can
        # catch an unterminated string literal, confirmed independently by
        # a second session: injecting a real newline into a JS string
        # fails THIS test and passes all 28 others. Skipping it does not
        # degrade coverage gracefully -- it removes the entire guard
        # against the class of bug that has destroyed this dashboard
        # three times. A green run on a box without node means less than
        # a green run on one with it.
        pytest.skip(
            "node not installed -- THE UNTERMINATED-STRING CHECK DID NOT RUN. "
            "The brace-balance checks cannot catch that class of bug (they "
            "blank string literals first). Install node to restore it."
        )

    js = _extract_script(_render_dashboard_html(3))
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "dashboard.js"
        f.write_text(js)
        proc = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert proc.returncode == 0, (
        "the rendered dashboard script is not valid JavaScript:\n"
        + (proc.stderr or proc.stdout)
    )
