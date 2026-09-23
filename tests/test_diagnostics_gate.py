"""Direct unit tests for opendarts/live/diagnostics_gate.py -- the runtime
switch behind the 2026-09-04 "gate all live per-dart diagnostics, default
OFF" task. Higher-level, real-call-site regression tests (does gating
this specific site actually stop the computation) live in
tests/test_capture_daemon.py, tests/test_throw_trigger_detection.py,
tests/test_live_server.py, and tests/test_logging_setup.py -- this file
is specifically about the switch's own mechanism."""
from __future__ import annotations

import logging
import threading

from opendarts.live import diagnostics_gate


def test_default_is_off():
    assert diagnostics_gate.enabled() is False
    assert diagnostics_gate.meta() == {"enabled": False}


def test_set_enabled_true_then_false():
    assert diagnostics_gate.set_enabled(True) is True
    assert diagnostics_gate.enabled() is True
    assert diagnostics_gate.meta() == {"enabled": True}

    assert diagnostics_gate.set_enabled(False) is False
    assert diagnostics_gate.enabled() is False
    assert diagnostics_gate.meta() == {"enabled": False}


def test_set_enabled_is_idempotent():
    diagnostics_gate.set_enabled(True)
    diagnostics_gate.set_enabled(True)
    assert diagnostics_gate.enabled() is True
    diagnostics_gate.set_enabled(False)
    diagnostics_gate.set_enabled(False)
    assert diagnostics_gate.enabled() is False


def test_set_enabled_accepts_truthy_falsy_not_just_bool():
    """The dashboard/API hands this a JSON bool, but a defensive
    `bool(...)` coercion (see server.py's api_diagnostics_set()) means
    this must also behave sanely given a plain truthy/falsy value."""
    assert diagnostics_gate.set_enabled(1) is True
    assert diagnostics_gate.set_enabled(0) is False


def test_enabled_is_a_cheap_threading_event_read():
    """The actual mechanism -- a threading.Event, not a lock-guarded
    plain bool -- confirmed directly, since every gated call site's own
    real cost claim depends on this being genuinely cheap."""
    assert isinstance(diagnostics_gate._enabled, threading.Event)


def test_set_enabled_flips_third_party_logger_levels_as_a_side_effect():
    """The switch's own real side effect on websockets/uvicorn.error --
    a fuller, dedicated proof of this lives in test_logging_setup.py;
    this confirms diagnostics_gate.set_enabled() is the thing that
    actually calls it, not merely documented to."""
    try:
        diagnostics_gate.set_enabled(False)
        assert logging.getLogger("websockets").level == logging.WARNING
        diagnostics_gate.set_enabled(True)
        assert logging.getLogger("websockets").level == logging.NOTSET
    finally:
        diagnostics_gate.set_enabled(False)


def test_cross_thread_visibility():
    """The real, live use case: the FastAPI request-handling thread
    toggles it, the capture loop's own background thread must see the
    new value on its very next read -- proven with a real second
    thread, not just sequential same-thread calls."""
    diagnostics_gate.set_enabled(False)
    observed: list[bool] = []
    ready = threading.Event()
    proceed = threading.Event()

    def reader():
        ready.set()
        proceed.wait(timeout=2.0)
        observed.append(diagnostics_gate.enabled())

    t = threading.Thread(target=reader)
    t.start()
    ready.wait(timeout=2.0)
    diagnostics_gate.set_enabled(True)
    proceed.set()
    t.join(timeout=2.0)

    assert observed == [True]
