"""Concurrent engine dispatch + per-engine timeout -- docs/ENGINES.md's
"Execution model": "All enabled engines... are dispatched at that same
moment, each in its own thread, each wrapped in a per-engine timeout."
A mocked slow/hanging engine gets cut off; others are unaffected; the
combined dict is still returned promptly either way.
"""
from __future__ import annotations

import threading
import time


from opendarts.engines.base import EngineResult
from opendarts.engines.dispatch import dispatch_engines
from opendarts.engines.registry import ENGINES


class _FastEngine:
    def score(self, bg_images, frame_images, calibration):
        return EngineResult(ok=True, sector="1", ring="single_inner", board_xy_mm=(1.0, 1.0), reason="fast")


class _SlowEngine:
    """Sleeps well past any sane per-engine timeout used in these tests --
    proves dispatch_engines() cuts it off rather than waiting."""

    def __init__(self, sleep_s: float):
        self.sleep_s = sleep_s
        self.finished = threading.Event()

    def score(self, bg_images, frame_images, calibration):
        time.sleep(self.sleep_s)
        self.finished.set()  # only reached if NOT actually abandoned
        return EngineResult(ok=True, sector="2", ring="single_inner", board_xy_mm=(2.0, 2.0), reason="slow")


class _RaisingEngine:
    def score(self, bg_images, frame_images, calibration):
        raise RuntimeError("boom")


def _registry(**engines):
    return dict(engines)


def test_dispatch_engines_empty_names_returns_empty_dict():
    assert dispatch_engines({}, {}, {}, [], timeout_s=1.0) == {}


def test_dispatch_engines_runs_a_fast_engine_and_returns_its_result():
    reg = _registry(Fast=_FastEngine())
    results = dispatch_engines({}, {}, {}, ["Fast"], timeout_s=1.0, registry=reg)
    assert set(results) == {"Fast"}
    r = results["Fast"]
    assert r.ok is True
    assert r.sector == "1"
    assert r.timed_out is False
    assert r.duration_s >= 0.0


def test_dispatch_engines_unknown_name_produces_an_honest_failure_entry():
    reg = _registry(Fast=_FastEngine())
    results = dispatch_engines({}, {}, {}, ["NotRegistered"], timeout_s=1.0, registry=reg)
    r = results["NotRegistered"]
    assert r.ok is False
    assert "unknown engine" in r.reason
    assert r.timed_out is False


def test_dispatch_engines_a_raising_engine_becomes_an_honest_ok_false_result():
    reg = _registry(Bad=_RaisingEngine(), Fast=_FastEngine())
    results = dispatch_engines({}, {}, {}, ["Bad", "Fast"], timeout_s=1.0, registry=reg)
    assert results["Bad"].ok is False
    assert "boom" in results["Bad"].reason
    assert results["Bad"].timed_out is False
    # The raising engine must not affect the other engine's own result.
    assert results["Fast"].ok is True


def test_dispatch_engines_cuts_off_a_hanging_engine_within_the_timeout():
    """THE per-engine-timeout proof: a deliberately slow engine (sleeps
    5s) with a 0.3s timeout must produce a `timed_out=True` result, and
    dispatch_engines() itself must return promptly (well under the
    engine's own 5s sleep) -- not wait for it."""
    slow = _SlowEngine(sleep_s=5.0)
    reg = _registry(Slow=slow, Fast=_FastEngine())

    start = time.monotonic()
    results = dispatch_engines({}, {}, {}, ["Slow", "Fast"], timeout_s=0.3, registry=reg)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"dispatch_engines() took {elapsed:.2f}s -- must not wait for the hung engine"
    assert results["Slow"].ok is False
    assert results["Slow"].timed_out is True
    assert "timed out" in results["Slow"].reason
    # The OTHER (fast) engine's result must be completely unaffected by
    # the slow one timing out.
    assert results["Fast"].ok is True
    assert results["Fast"].timed_out is False


def test_dispatch_engines_multiple_slow_engines_share_one_deadline_not_stacked():
    """Two engines each sleeping longer than the timeout must not cause
    dispatch_engines() to wait ~2x the timeout (one after another) --
    they run CONCURRENTLY under one shared deadline, see
    opendarts.engines.dispatch's own module docstring."""
    reg = _registry(SlowA=_SlowEngine(sleep_s=5.0), SlowB=_SlowEngine(sleep_s=5.0))

    start = time.monotonic()
    results = dispatch_engines({}, {}, {}, ["SlowA", "SlowB"], timeout_s=0.3, registry=reg)
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"dispatch_engines() took {elapsed:.2f}s -- timeouts must be concurrent, not stacked"
    assert results["SlowA"].timed_out is True
    assert results["SlowB"].timed_out is True


def test_dispatch_engines_never_raises_out_of_the_call():
    """No combination of unknown/raising/timing-out engines should ever
    propagate an exception out of dispatch_engines() itself."""
    reg = _registry(
        Bad=_RaisingEngine(), Slow=_SlowEngine(sleep_s=5.0), Fast=_FastEngine(),
    )
    results = dispatch_engines(
        {}, {}, {}, ["Bad", "Slow", "Fast", "NotRegistered"], timeout_s=0.2, registry=reg
    )
    assert set(results) == {"Bad", "Slow", "Fast", "NotRegistered"}


def test_dispatch_engines_real_registry_names_are_dispatchable():
    """Sanity check against the REAL registry (not a fake one) -- Talos
    is a dispatchable engine name. Empty camera inputs: no shaft planes,
    so the real engine returns ok=False (measured), not a stub miss."""
    results = dispatch_engines({}, {}, {}, ["Talos"], timeout_s=1.0, registry=ENGINES)
    r = results["Talos"]
    assert isinstance(r, EngineResult)
    assert r.timed_out is False
    assert r.ok is False


# ---------------------------------------------------------------------------
# Which calibration and priors reach each engine. dispatch.py itself is
# engine-agnostic -- these tests use plain fake engine names.
# ---------------------------------------------------------------------------

class _CalibrationSpyEngine:
    """Records exactly which calibration object it was called with, by
    identity -- proves the shared calibration argument actually reached
    score(), not just that scoring succeeded."""

    def __init__(self):
        self.seen_calibrations: list = []

    def score(self, bg_images, frame_images, calibration):
        self.seen_calibrations.append(calibration)
        return EngineResult(ok=True, sector="1", ring="single_inner", board_xy_mm=(1.0, 1.0))


class _PriorSpyEngine:
    """Records prior_board_xy_mm if it was passed -- proves dispatch
    forwards the kwarg only to engines that declare it."""

    def __init__(self):
        self.seen_priors = []

    def score(self, bg_images, frame_images, calibration, prior_board_xy_mm=None):
        self.seen_priors.append(prior_board_xy_mm)
        return EngineResult(ok=True, sector="1", ring="single_inner", board_xy_mm=(1.0, 1.0))


def test_dispatch_engines_forwards_prior_board_xy_mm_only_to_engines_that_declare_it():
    prior_spy = _PriorSpyEngine()
    plain = _CalibrationSpyEngine()
    priors = ((154.75, -46.5),)
    results = dispatch_engines(
        {}, {}, {}, ["Prior", "Plain"],
        timeout_s=1.0, registry=_registry(Prior=prior_spy, Plain=plain),
        prior_board_xy_mm=priors,
    )
    assert results["Prior"].ok is True
    assert results["Plain"].ok is True
    assert prior_spy.seen_priors == [priors]


def test_dispatch_engines_passes_the_shared_calibration_to_each_engine():
    """Every engine is scored with the one calibration the caller passed."""
    spy = _CalibrationSpyEngine()
    shared_calibration = {0: object()}
    reg = _registry(Spy=spy)

    results = dispatch_engines({}, {}, shared_calibration, ["Spy"], timeout_s=1.0, registry=reg)

    assert results["Spy"].ok is True
    assert spy.seen_calibrations == [shared_calibration]

