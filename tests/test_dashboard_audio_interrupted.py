"""Safari's fourth AudioContext state, and the stale-flag bug beside it.

REPORTED FROM AN IPAD, 2026-09-15. Audio was enabled, three phrases had
played, and then it went silent with nothing on screen to say why. The
client reported:

    iPad / Safari  enabled=True  blocked=False  ctx=interrupted  spoken=3

Two defects, and the second is the one that made it invisible.

1. `interrupted` is a Safari-only AudioContext state -- the spec defines
   suspended / running / closed. iOS moves a context into it when the
   audio session is taken away: a call, Siri, another app, the screen
   locking. `resume()` frequently resolves WITHOUT leaving that state,
   because the session is gone and resuming a dead context cannot get one
   back. Recovery needs a NEW context.

2. The statechange handler read
   `if (state === 'running') audioBlocked = false` -- it could only ever
   turn the warning OFF. Going running -> interrupted left the flag stale
   at false, so no banner was rendered. Silent failure is the single
   outcome browser audio was moved off the server to prevent.

These are string-level checks on the rendered dashboard. That is a weak
form of testing and it is what is available: the alternative is a headless
browser and an audio session, and the bug lives in four lines of state
handling that are cheap to pin and expensive to lose.
"""
from __future__ import annotations

import re

import pytest

from opendarts.live.server import _render_dashboard_html


@pytest.fixture(scope="module")
def js() -> str:
    """Just the dashboard's script block."""
    html = _render_dashboard_html(3)
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert blocks, "no script block in the dashboard"
    return "\n".join(blocks)


def test_the_interrupted_state_is_handled(js: str) -> None:
    """iOS's fourth state must be recognised by name.

    Code that only knows running/suspended/closed treats an interrupted
    context as a permanent mystery: not running, never recoverable, and on
    an iPad that is the normal state after a screen lock.
    """
    assert "'interrupted'" in js, (
        "the dashboard no longer mentions the 'interrupted' AudioContext "
        "state. Safari uses it after any audio-session interruption; "
        "without handling, an iPad goes permanently silent after a lock "
        "or a phone call."
    )


def test_an_interrupted_context_is_rebuilt_not_just_resumed(js: str) -> None:
    """resume() alone does not clear `interrupted` -- a new context does."""
    assert "rebuildAudioContext" in js
    # Matched loosely on purpose: the condition gained a `&& !audioRecovering`
    # guard when the re-entry guard was narrowed to the rebuild alone, and a
    # test that pins the exact expression breaks on every such refinement
    # while telling you nothing about the behaviour.
    interrupted_branch = js[js.index("ctx.state === 'interrupted'"):]
    assert "rebuildAudioContext()" in interrupted_branch[:600], (
        "the interrupted branch no longer rebuilds the context. resume() "
        "resolves without clearing that state, so resuming alone leaves "
        "the device silent while reporting success."
    )


def test_statechange_assigns_the_flag_rather_than_only_clearing_it(js: str) -> None:
    """The regression that made the failure invisible.

    A handler that only ever sets `audioBlocked = false` cannot raise the
    banner when a context goes away -- which is the one moment the banner
    is needed.
    """
    handler = js[js.index("onstatechange = ()"):]
    handler = handler[:handler.index("};")]
    assert "audioBlocked = " in handler, "statechange no longer sets audioBlocked"
    assert "!== 'running'" in handler, (
        "statechange sets audioBlocked without testing for NOT running -- "
        "it can only clear the warning, never raise it, so an interrupted "
        "or suspended context renders no banner"
    )


def test_recovery_guard_is_released_in_a_finally(js: str) -> None:
    """A stuck re-entry guard would disable recovery for the page's life.

    Silently, which is the same class of bug as the one being fixed.
    """
    fn = js[js.index("async function resumeAudioContext"):]
    fn = fn[:fn.index("\nasync function fetchAudioCatalog")]
    assert "finally {" in fn and "audioRecovering = false;" in fn, (
        "audioRecovering is no longer released in a finally -- one throw "
        "inside recovery would disable all future recovery"
    )


def test_blocked_is_derived_from_the_live_context_not_cached(js: str) -> None:
    """A cached flag was wrong in the one direction that matters.

    `audioBlocked` initialises to false and is only corrected by a state
    CHANGE or a resume attempt, so a context BORN suspended and never
    transitioning leaves it stale at false. Measured on an iPad,
    2026-09-15: enabled=True, blocked=False, ctx=suspended, 64/64 clips
    loaded, nothing spoken -- and the UI showed no problem, because it
    read the flag rather than the context.

    The banner that once depended on this has been deleted, but the
    predicate still drives the Config panel line and the state reported to
    the rig, which is what an operator reads when they cannot see the
    screen.
    """
    assert "function audioIsBlocked()" in js
    fn = js[js.index("function audioIsBlocked()"):]
    fn = fn[:fn.index("\nfunction ")]
    assert "audioCtx.state" in fn, (
        "audioIsBlocked no longer reads the context's own state -- it is "
        "back to trusting a cached flag that a born-suspended context "
        "never updates"
    )


def test_the_reported_client_state_uses_the_same_source(js: str) -> None:
    """The rig's client list must not disagree with the device's banner.

    Reporting a cached `blocked` while the banner derives a live one gives
    two answers to one question, and the remote diagnostic is the one an
    operator trusts when they cannot see the screen.
    """
    report = js[js.index("blocked:"):]
    assert report.startswith("blocked: audioIsBlocked()"), (
        "reportAudioState sends a cached blocked flag; it must derive from "
        "the same predicate the banner uses"
    )


def test_the_guard_never_blocks_a_plain_resume(js: str) -> None:
    """A tap must always reach resume(), whichever event gets there first.

    iOS fires pointerdown, then touchend, then click, and all three call
    resumeAudioContext. An earlier version wrapped the whole function in
    the re-entry guard, so the first event set it and the other two --
    including the banner's own click handler -- returned without ever
    calling resume(). The banner then could not be dismissed by tapping
    it, which is the one thing a banner exists for. Measured on an iPad,
    2026-09-15.

    resume() is idempotent; only the rebuild can recurse, so only the
    rebuild is guarded.
    """
    fn = js[js.index("async function resumeAudioContext"):]
    fn = fn[:fn.index("\nasync function fetchAudioCatalog")]
    before_resume = fn[:fn.index("await ctx.resume()")]
    assert "audioRecovering" not in before_resume, (
        "resumeAudioContext returns early on the re-entry guard before it "
        "reaches resume(). On iOS a single tap fires three events; the "
        "first would set the guard and suppress the real gesture, leaving "
        "the enable-sound banner impossible to dismiss."
    )
    rebuild_at = fn.index("rebuildAudioContext()")
    assert "audioRecovering" in fn[:rebuild_at], (
        "the rebuild path is no longer guarded -- it changes context "
        "state, which fires onstatechange, which calls back into here"
    )
