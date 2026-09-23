"""The `hidden` attribute must actually hide things.

WHY THIS EXISTS. On 2026-09-15 the audio "tap to enable sound" banner
could not be dismissed. Audio played correctly; the bar just stayed up
forever, warning about a problem that had already been fixed -- which is
worse than no warning, because it trains people to ignore the one control
that matters when sound really is blocked.

The cause was CSS, not logic. `hidden` is styled only by the UA
stylesheet's `[hidden] {{ display: none }}`, whose specificity is (0,1,0).
A rule like `.audio-blocked {{ display: flex }}` scores the same (0,1,0)
and, being later in the cascade, wins. So `el.hidden = true` set the
attribute and changed nothing on screen.

The dashboard already knew this -- `.cam-overlay[hidden]`,
`.cam-placeholder[hidden]`, `.cam-url-all[hidden]`, `.modal-backdrop[hidden]`
and `.modal-manual[hidden]` all carry the guard. The banner was written
without it, and nothing was checking.

So this checks. For every element declared with a bare `hidden` attribute
in the dashboard, if its class has a `display:` rule, there must also be a
`[hidden]` rule for that class. It is a cheap structural check for a bug
that is invisible in Python, invisible in JS, and only appears in a
browser -- which is exactly the kind this suite cannot otherwise reach.
"""
from __future__ import annotations

import re

import pytest

from opendarts.live import server


def dashboard_html() -> str:
    """The rendered dashboard, files already assembled into the page.

    Three cameras, matching tests/test_dashboard_html_structure.py: a
    multi-camera render is the one that exercises every repeated block.
    """
    return server._render_dashboard_html(3)


#: `<tag ... class="a b" ... hidden ...>` -- a bare `hidden` attribute, not
#: `hidden="..."` and not a `data-hidden`. Class and hidden can appear in
#: either order, so both are found independently within one tag.
_TAG = re.compile(r"<(?!/)[a-zA-Z][^>]*>")
_CLASS = re.compile(r"""\bclass\s*=\s*["']([^"']*)["']""")
_HIDDEN = re.compile(r"""\bhidden(?=[\s/>])""")


def hidden_element_classes(html: str) -> "set[str]":
    out: "set[str]" = set()
    for tag in _TAG.findall(html):
        if not _HIDDEN.search(tag):
            continue
        m = _CLASS.search(tag)
        if not m:
            # No class at all: nothing can out-specify the UA rule.
            continue
        out.update(c for c in m.group(1).split() if c)
    return out


def classes_with_a_display_rule(css: str) -> "set[str]":
    """Classes whose own rule sets `display`, which is what defeats
    `[hidden]`."""
    out: "set[str]" = set()
    for sel, body in re.findall(r"\.([A-Za-z0-9_-]+)\s*\{([^}]*)\}", css):
        if re.search(r"(^|[;{\s])display\s*:", body):
            out.add(sel)
    return out


def classes_guarded(css: str) -> "set[str]":
    return set(re.findall(r"\.([A-Za-z0-9_-]+)\[hidden\]", css))


@pytest.fixture(scope="module")
def html() -> str:
    return dashboard_html()


def test_every_hidden_element_can_actually_be_hidden(html: str) -> None:
    """A `display:` rule on a hideable element needs a `[hidden]` guard.

    Without one, setting `el.hidden = true` is a no-op on screen and the
    element is permanently visible -- silently, because the attribute IS
    set and every assertion about state passes.
    """
    css = html
    hidden_classes = hidden_element_classes(html)
    assert hidden_classes, "found no hideable elements -- this test has rotted"

    displayed = classes_with_a_display_rule(css)
    guarded = classes_guarded(css)

    unprotected = sorted((hidden_classes & displayed) - guarded)
    assert not unprotected, (
        "these classes are on elements that start `hidden` and set their own "
        f"`display`, with no `[hidden]` guard: {unprotected}. The UA rule "
        "`[hidden] {{ display: none }}` is specificity (0,1,0) and loses to a "
        "class rule of the same weight later in the file, so `el.hidden = "
        "true` will set the attribute and change nothing on screen. Add "
        "`.<class>[hidden] {{ display: none; }}` beside the rule."
    )
