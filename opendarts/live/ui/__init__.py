"""The new dashboard, served at ``/`` when the rig's switch says so
(opendarts/live/dashboard_choice.py; the classic one is the other choice).

A redesign from first principles -- three spaces (Play, Throws, Rig), one
status, one concept of truth -- built on exactly the same API as the
current page, so the two can be compared on a real rig before either
replaces the other. The design brief is ``dev/ux/BRIEF.md``.

Delivered the same way as ``opendarts.live.dashboard``: real source files,
read once at import and inlined into ONE document, with no build step and
no second request (the browser's per-origin socket budget is shared with
the camera streams, and a rig may have no internet for a CDN).

The script is several files, concatenated in ``JS_ORDER``. They share one
scope, so a file may use anything defined by a file before it at load
time, and anything at all from inside a handler.

Placeholders in ``index.html``, all filled by ``render``:

  ``@@OD_HOST_LABEL@@``      this machine's name, already HTML-escaped
  ``@@OD_BOOTSTRAP_JSON@@``  server-computed values, as one JSON object
  ``@@OD_UI_CSS@@``          ui.css
  ``@@OD_UI_JS@@``           the concatenated script
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).resolve().parent

JS_ORDER = (
    "core.js",     # bootstrap, DOM helpers, format, API, store, sheets
    "board.js",    # board geometry: live board, agreement plot, picker
    "sound.js",    # spoken calls, per screen
    "truth.js",    # "What actually landed?"
    "play.js",     # the scoreboard, and the one status derivation
    "throws.js",   # review
    "rig.js",      # cameras, settings, system
    "calib.js",    # calibration, on the whole screen; the connection strip
    "display.js",  # the show-only role: a TV set up from a controller
    "shell.js",    # routing, status, sheets, socket, start-up -- runs last
)


def _read(relative: str) -> str:
    return (_DIR / relative).read_text(encoding="utf-8").removesuffix("\n")


# The two type families, embedded rather than linked: a rig may have no
# internet, and a font that silently falls back takes the design with it.
# Latin subsets, SIL Open Font License 1.1 (fonts/OFL-*.txt).
FONTS = (
    ("Barlow Condensed", 600, "normal", "BarlowCondensed-600.woff2"),
    ("Barlow Condensed", 700, "normal", "BarlowCondensed-700.woff2"),
    ("Barlow Condensed", 800, "italic", "BarlowCondensed-800i.woff2"),
    ("Barlow Condensed", 900, "italic", "BarlowCondensed-900i.woff2"),
    ("Barlow", 500, "normal", "Barlow-500.woff2"),
    ("Barlow", 600, "normal", "Barlow-600.woff2"),
    ("IBM Plex Mono", 400, "normal", "IBMPlexMono-400.woff2"),
    ("IBM Plex Mono", 600, "normal", "IBMPlexMono-600.woff2"),
    ("IBM Plex Sans Condensed", 600, "normal", "IBMPlexSansCondensed-600.woff2"),
    ("IBM Plex Sans Condensed", 700, "normal", "IBMPlexSansCondensed-700.woff2"),
)


def _font_faces() -> str:
    import base64

    faces = []
    for family, weight, style, name in FONTS:
        data = base64.b64encode((_DIR / "fonts" / name).read_bytes()).decode("ascii")
        faces.append(
            f"@font-face {{ font-family: '{family}'; font-weight: {weight}; font-style: {style};"
            f" font-display: swap; src: url(data:font/woff2;base64,{data}) format('woff2'); }}"
        )
    return "\n".join(faces)


INDEX_HTML = _read("index.html")
CSS = _read("ui.css").replace("@@OD_UI_FONTS@@", _font_faces(), 1)
JS = "\n".join(_read("js/" + name) for name in JS_ORDER)

# What the page fingerprint must cover, so an open screen reloads itself
# after an update to any of it.
FINGERPRINT_PARTS = (INDEX_HTML, CSS, JS)


def render(*, host_label: str, bootstrap_json: str) -> str:
    """The whole page. ``host_label`` must already be HTML-escaped."""
    # "</" inside the JSON block would end the <script> early.
    safe_json = bootstrap_json.replace("</", "<\\/")
    return (
        INDEX_HTML
        .replace("@@OD_HOST_LABEL@@", host_label)
        .replace("@@OD_BOOTSTRAP_JSON@@", safe_json)
        # The two whole files last, so nothing inside them that merely
        # looks like a placeholder is ever rescanned.
        .replace("@@OD_UI_CSS@@", CSS)
        .replace("@@OD_UI_JS@@", JS)
    )
