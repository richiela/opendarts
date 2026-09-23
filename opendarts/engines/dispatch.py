"""Concurrent engine dispatch -- the actual "run every enabled engine at
once, each with its own timeout, without blocking on the slowest one"
mechanism docs/ENGINES.md's "Execution model" section describes. This is
the ONLY place in the framework that knows about threads/timeouts; every
engine's own `score()` stays a plain, synchronous function (see
opendarts.engines.base).

Design notes (why it's built this way, not the more obvious way):

- All engines are submitted to one `ThreadPoolExecutor` and then waited
  on TOGETHER via a single `concurrent.futures.wait(..., timeout=...)`
  call, not one `future.result(timeout=...)` per engine in a loop. Since
  every engine starts at (approximately) the same moment, a single
  shared deadline achieves the same "per-engine timeout" effect as N
  independent ones would -- and critically, waiting on them ONE AT A TIME
  would let a slow engine 1 eat into engine 2's own budget (each
  `.result(timeout=timeout_s)` call would restart its own clock), which
  is not what "a per-engine timeout" is supposed to mean when engines run
  concurrently.
- The pool is shut down with `wait=False, cancel_futures=True` -- NOT the
  default `pool.shutdown(wait=True)` a `with ThreadPoolExecutor() as
  pool:` block would use. `wait=True` would block the CALLING thread
  until every submitted engine actually finishes, even ones already
  given up on for timing out -- silently defeating the entire point of
  the timeout. Python threads can't be forcibly killed, so a
  genuinely-hung engine's thread keeps running in the background
  (harmless -- it has no side effects per the Engine contract, and its
  result is simply never read), but this function itself returns as soon
  as its own deadline passes, regardless.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time

import numpy as np

from opendarts.engines.base import EngineResult
from opendarts.engines.registry import ENGINES
from opendarts.pipeline import CameraCalibration

log = logging.getLogger(__name__)

DEFAULT_ENGINE_TIMEOUT_S = 5.0


def _call_engine_score(engine, bg_images, frame_images, calibration, extra_kwargs=None):
    """Call engine.score(), forwarding only kwargs that score() declares.

    Used so dispatch can pass `prior_board_xy_mm` to Talos without
    requiring every engine (or every test-double) to grow that
    parameter. Mirrors Apollo's own `engine_accepts_prior_dart_line_px`
    capability check, applied here one kwargs-dict at a time.
    """
    extra_kwargs = extra_kwargs or {}
    if extra_kwargs:
        import inspect

        try:
            params = inspect.signature(engine.score).parameters
        except (TypeError, ValueError):
            params = {}
        extra_kwargs = {
            key: value for key, value in extra_kwargs.items()
            if key in params and value is not None
        }
    return engine.score(bg_images, frame_images, calibration, **extra_kwargs)


def _run_one(engine: object, name: str, bg_images, frame_images, calibration, extra_kwargs=None) -> EngineResult:
    """Runs inside the worker thread. Never raises -- an engine that
    throws gets converted into an honest ok=False EngineResult instead of
    killing this thread silently or corrupting the dispatch batch (the
    per-engine timeout only guards against HANGING; a fast crash needs
    its own handling here, or it would surface as an unhandled exception
    an outer future.result() call would re-raise for THIS engine only,
    but inconsistently vs. the timeout path -- normalizing both into the
    same EngineResult shape is simpler for every caller)."""
    start = time.monotonic()
    try:
        result = _call_engine_score(
            engine, bg_images, frame_images, calibration, extra_kwargs,
        )
    except Exception as exc:  # noqa: BLE001 -- see docstring
        elapsed = time.monotonic() - start
        log.exception("engine %r raised during score()", name)
        return EngineResult(
            ok=False,
            sector=None,
            ring=None,
            board_xy_mm=None,
            reason=f"engine raised {type(exc).__name__}: {exc}",
            diagnostics={},
            duration_s=elapsed,
            timed_out=False,
        )
    elapsed = time.monotonic() - start
    result.duration_s = elapsed
    result.timed_out = False
    return result


def dispatch_engines(
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    calibration: dict[int, CameraCalibration],
    names: list[str],
    *,
    timeout_s: float = DEFAULT_ENGINE_TIMEOUT_S,
    registry: dict[str, object] | None = None,
    prior_board_xy_mm: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None = None,
    prior_dart_line_px: "dict[int, tuple[tuple[float, float], tuple[float, float]]] | None" = None,
) -> dict[str, EngineResult]:
    """Runs every engine in `names` concurrently against the SAME inputs,
    each wrapped in the shared `timeout_s` deadline (see module
    docstring). Always returns one EngineResult per requested name --
    an unknown name, a raising engine, and a timed-out engine all produce
    a real EngineResult entry (never a missing key, never an exception
    propagating out of this function) so a caller (opendarts/live/
    capture_daemon.py, opendarts/capture/replay.py, opendarts/capture/
    rescore_all.py) never needs its own separate error-handling path per
    failure mode.

    `prior_board_xy_mm` (added 2026-08-17, Talos prior-dart erase --
    see `opendarts.engines.talos.prior_dart`): optional board-plane XY of
    earlier darts in this visit. Forwarded only to engines whose
    `score()` declares that parameter; every other engine in the batch
    is called with the original 3-arg signature, unmodified. Default
    None -- zero behavior change for every existing caller.

    `prior_dart_line_px` (added 2026-08-24 -- fixing the real gap found
    investigating the recorded S10 throw, see
    `opendarts.engines.zeus.engine`'s module docstring for the full
    incident): Apollo's own per-camera prior-dart-in-visit
    contamination guard (`opendarts.engines.apollo.prior_dart_context`).
    Before this fix, this function had NO way to forward this parameter
    at all, so Apollo got zero contamination protection whenever it
    ran as an ALSO-RUN engine (as opposed to being configured as the
    literal primary) -- this closes that gap the same way
    `prior_board_xy_mm` already does for Talos: forwarded only to
    engines whose `score()` declares the parameter (via
    `_call_engine_score()`'s existing signature-filtering, unchanged),
    every other engine in the batch unaffected. Default None -- zero
    behavior change for every existing caller.
    """
    registry = ENGINES if registry is None else registry
    results: dict[str, EngineResult] = {}
    if not names:
        return results

    runnable: dict[str, object] = {}
    for name in names:
        engine = registry.get(name)
        if engine is None:
            results[name] = EngineResult(
                ok=False,
                sector=None,
                ring=None,
                board_xy_mm=None,
                reason=f"unknown engine {name!r} -- not in the registry",
                diagnostics={},
            )
            continue
        runnable[name] = engine

    if not runnable:
        return results

    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=len(runnable), thread_name_prefix="engine-dispatch"
    )
    try:
        extra_kwargs = {}
        if prior_board_xy_mm:
            extra_kwargs["prior_board_xy_mm"] = prior_board_xy_mm
        if prior_dart_line_px:
            extra_kwargs["prior_dart_line_px"] = prior_dart_line_px
        extra_kwargs = extra_kwargs or None
        future_to_name = {
            pool.submit(
                _run_one, engine, name, bg_images, frame_images, calibration, extra_kwargs,
            ): name
            for name, engine in runnable.items()
        }
        done, not_done = concurrent.futures.wait(
            list(future_to_name), timeout=timeout_s,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        for future in done:
            name = future_to_name[future]
            results[name] = future.result()
        for future in not_done:
            name = future_to_name[future]
            results[name] = EngineResult(
                ok=False,
                sector=None,
                ring=None,
                board_xy_mm=None,
                reason=f"timed out after {timeout_s}s -- engine did not finish in time",
                diagnostics={},
                duration_s=timeout_s,
                timed_out=True,
            )
    finally:
        # wait=False + cancel_futures=True: see module docstring -- never
        # block this call on a still-running (timed-out) engine thread.
        pool.shutdown(wait=False, cancel_futures=True)

    return results

