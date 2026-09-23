"""opendarts/live/diagnostics_gate.py -- ONE runtime-togglable switch that
gates every PER-TICK / PER-THROW diagnostic COMPUTATION on the LIVE
per-dart detection/scoring path. Default OFF.

2026-09-04: all diagnostics stay in place but only fire when turned on,
so they can be switched off to measure true latency.

SCOPE, narrowed explicitly 2026-09-04: this gate covers the LIVE per-dart detection/
scoring path only -- the capture poll loop (`opendarts.live.capture_daemon.
run_capture_loop_body()`), the throw trigger (originally the since-
deleted `opendarts.capture.throw_trigger.advance()`; today `opendarts.
lifecycle`, which keeps its own per-tick record in a JSONL and reads no
gate), frame fetch, the engines, package save -- anything that fires
per-tick or per-throw while the board is armed.

**CALIBRATION DIAGNOSTICS ARE EXPLICITLY OUT OF SCOPE -- NOTHING under
`bootstrap_calibrations()` (or anything it calls: landmark detection,
per-frame ellipse/orientation/phase-confidence diagnostics, reprojection/
focal-length reporting, calibration package contents, the raw
`cam*_raw.mkv` capture) reads this gate, and nothing should.**
Calibration runs once, takes tens of seconds by design, and that
instrumentation is actively valuable for ongoing calibration
investigations (the cam0 ellipse work, the pose/focal investigation) --
gating or touching it would cost more than it saves. If a future call
site under `bootstrap_calibrations()` is ever tempted to read this gate
because it "technically does real work per round," don't -- it's
calibration-path, not detection-path, and stays untouched regardless.

THE HARD PART, stated explicitly because it's easy to get subtly wrong:
gate the COMPUTATION, not just the log emit. `log.debug(f"...
{expensive()}...")` runs `expensive()` regardless of whether DEBUG is
even enabled -- Python evaluates a function's arguments before the call
happens, so a log-LEVEL change alone does nothing for that class of
cost (this project's own `opendarts.live.logging_setup.
configure_console_and_file_logging()` keeps the ROOT logger at DEBUG
unconditionally, so `logger.isEnabledFor(logging.DEBUG)` is true almost
everywhere in this process by default -- a real, confirmed-live
consequence: the legacy trigger's (`throw_trigger.advance()`, since
deleted) per-camera settle-status DEBUG line was ALREADY firing on every settle-loop
iteration in production before this gate existed, despite looking
level-guarded). Every call site wired to this gate follows the shape:

    if diagnostics_gate.enabled():
        <compute the diagnostic value>
        log.debug(...)

never

    log.debug(<compute the diagnostic value>)

Mechanism: a plain `threading.Event`, matching this project's own
established "cross-thread boolean signal" convention -- see
`opendarts.live.capture_daemon.CaptureLoopController`'s own
`start_requested`/`session_stop_event`/`stopped_ack` ("Event objects are
already internally thread-safe... not lock-guarded"). Read from the
capture loop's background thread on every iteration; written from the
FastAPI request-handling thread via `POST /api/diagnostics`
(`opendarts/live/server.py`) -- takes effect on the very next iteration/log
call, no restart needed.

Deliberately its own small module, not tucked inside `capture_daemon.py`:
it was originally read from `opendarts/capture/` too (the legacy settle
classifier, since deleted), and this project's established import
direction is `opendarts.live` -> `opendarts.capture`, never the reverse at
module level -- a dedicated module here, with zero dependencies of its
own besides `opendarts.live.logging_setup` (itself dependency-free), is
what lets `capture_daemon.py`, `server.py` and `logging_setup.py` (and
any future `opendarts.capture`/`opendarts.lifecycle` reader) share ONE flag
without introducing an import cycle (confirmed: both
`opendarts/live/__init__.py` and `opendarts/capture/__init__.py` are empty, so
nothing eagerly triggers one side while importing the other).

Deliberately NOT persisted to disk (unlike `CaptureLoopController`'s
idle-timeout / `EngineConfigStore`'s engine config, both of which
survive a restart on purpose): the whole point of this switch is a fast
live A/B toggle within one running process, always defaulting back to
OFF (the safe, zero-added-cost state) on every fresh process start.
Persisting this across a restart was never asked for and would add a
real "did I leave this on and forget" footgun for exactly the kind of
true-latency measurement this switch exists to support.
"""
from __future__ import annotations

import logging
import threading

from opendarts.live import logging_setup

log = logging.getLogger(__name__)

# Default OFF -- Event.is_set() starts False, matching this project's own
# CaptureLoopController.start_requested/session_stop_event convention.
_enabled = threading.Event()


def enabled() -> bool:
    """True when live per-dart diagnostics are switched on. Cheap
    (threading.Event.is_set() is a single, already-thread-safe read --
    no lock acquisition of its own) -- safe to call on every poll-loop
    iteration, which is exactly what every gated call site does."""
    return _enabled.is_set()


def set_enabled(value: bool) -> bool:
    """Toggle the switch. Takes effect immediately -- the very next
    poll-loop iteration/log call sees the new state, no restart needed.
    Also flips the third-party (websockets/uvicorn.error) logger levels
    (see `opendarts.live.logging_setup.set_third_party_diagnostics_enabled()`
    for why those specifically) -- that needs a real
    `logging.getLogger(name).setLevel(...)` call happening HERE, at
    toggle time, not just a flag some other code reads later, since a
    logger's own level IS the thing being changed.

    Returns the resulting state (matches
    `CaptureLoopController.set_idle_timeout_sec()`'s own convention of a
    settable value being read back)."""
    if value:
        _enabled.set()
    else:
        _enabled.clear()
    logging_setup.set_third_party_diagnostics_enabled(_enabled.is_set())
    log.info("live diagnostics %s", "ENABLED" if _enabled.is_set() else "disabled")
    return _enabled.is_set()


def meta() -> dict[str, object]:
    """Non-secret bookkeeping for the dashboard -- same honest-snapshot
    convention as `CaptureLoopController.meta()`/`EngineConfigStore.meta()`."""
    return {"enabled": _enabled.is_set()}


# Apply the default (OFF) state's third-party logger levels immediately
# at import time -- so a process that imports this module without ever
# calling set_enabled() at all (every existing test, any script, or a
# live process before the dashboard's first POST /api/diagnostics) still
# gets the quiet, zero-added-file-volume default rather than depending on
# some other code remembering to call set_enabled(False) once at startup.
logging_setup.set_third_party_diagnostics_enabled(False)
