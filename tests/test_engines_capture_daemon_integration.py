"""Multi-engine scoring integration at the opendarts.live.capture_daemon
level -- docs/ENGINES.md's "Execution model" and "Config" sections:

- Also-run engines are dispatched concurrently, per-engine timeout, and
  the combined `other_engines`/`primary_engine` write happens on a
  background thread `handle_ready_to_capture()` does NOT wait on -- see
  tests/test_capture_daemon.py's own
  `test_handle_ready_to_capture_never_blocks_on_a_hanging_ad_match` for
  the established pattern this file's timing tests mirror, applied to
  engine dispatch instead of the AD ground-truth attach.
- `EngineConfigStore` is live-mutable (read fresh every READY_TO_CAPTURE,
  no restart) -- mirrors CalibrationStore's own tested behavior.
- `run_capture_loop_body()`'s own iteration timing must be unaffected by
  a slow also-run engine, regardless of engine count/speed -- THE
  load-bearing trigger-loop-never-blocks-on-scoring proof, at the loop
  level (not just handle_ready_to_capture() in isolation).
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

import opendarts.live.capture_daemon as capture_daemon
from tests.lifecycle_scripting import script_trigger
from opendarts.engines.base import EngineResult
from opendarts.engines.athena import AthenaEngine
from opendarts.engines.apollo import ApolloEngine
from opendarts.engines.ares import AresEngine
from opendarts.engines.talos import TalosEngine
from opendarts.engines.registry import DEFAULT_PRIMARY_ENGINE
from opendarts.engines.zeus import ZEUS_SUB_ENGINE_NAMES
from tests.test_capture_daemon import (
    STARTUP_FETCHES,
    ThrowState,
    ThrowTriggerState,
    _fake_calibration_attempt,
    _StopLoop,
    _throw_trigger_ready,
    _wait_until,
)


# ---------------------------------------------------------------------------
# EngineConfigStore -- live mutability (mirrors CalibrationStore's own
# tested shape).
# ---------------------------------------------------------------------------


def test_engine_config_store_defaults_to_the_production_shape():
    """A fresh store IS production config -- Zeus aggregating all four
    detection engines -- not something an operator has to switch on."""
    from opendarts.engines.registry import DEFAULT_ALSO_RUN

    store = capture_daemon.EngineConfigStore()
    cfg = store.get()
    assert cfg.primary == DEFAULT_PRIMARY_ENGINE
    assert cfg.also_run == DEFAULT_ALSO_RUN
    assert cfg.timeout_s > 0


# ---------------------------------------------------------------------------
# EngineConfigStore durability -- 2026-08-14, real incident: a process
# restart silently reverted a checked-all-3-engines config back to
# Apollo-only, with no warning, which read on the dashboard as "no
# engine scoring is showing up" rather than "your config reset."
# ---------------------------------------------------------------------------


def test_engine_config_store_missing_snapshot_file_falls_back_to_constructor_defaults(tmp_path):
    """A fresh install / wiped data dir is not an error -- the
    constructor's own primary/also_run/timeout_s args are the real
    default, exactly as if snapshot_path had never been passed."""
    path = tmp_path / "does" / "not" / "exist.json"
    store = capture_daemon.EngineConfigStore(also_run=("Talos",), snapshot_path=path)
    cfg = store.get()
    assert cfg.primary == DEFAULT_PRIMARY_ENGINE
    assert cfg.also_run == ("Talos",)


def test_engine_config_store_loads_the_engine_config_section_from_the_conf(tmp_path):
    """The conf file is the ONLY way to configure the engine, so a store
    constructed against one must come up with exactly what it names --
    not the code defaults. This is the read half of the 2026-08-14
    incident above: the config that survives a restart is the conf's."""
    from opendarts.live.config import write_config_section

    path = tmp_path / "config.json"
    write_config_section("engine_config", {
        "primary": "Athena",
        "also_run": ["Apollo", "Talos"],
        "timeout_s": 6.0,
    }, path)

    cfg = capture_daemon.EngineConfigStore(snapshot_path=path).get()
    assert cfg.primary == "Athena"
    assert cfg.also_run == ("Apollo", "Talos")
    assert cfg.timeout_s == 6.0


def test_engine_config_store_conf_naming_an_unregistered_engine_falls_back(tmp_path):
    """A conf naming an engine the registry no longer has must not take
    the process down, and must not come up half-applied -- it falls back
    to the constructor defaults whole."""
    from opendarts.live.config import write_config_section

    path = tmp_path / "config.json"
    write_config_section("engine_config", {
        "primary": "NoSuchEngine",
        "also_run": [],
        "timeout_s": 5.0,
    }, path)

    cfg = capture_daemon.EngineConfigStore(also_run=("Talos",), snapshot_path=path).get()
    assert cfg.primary == DEFAULT_PRIMARY_ENGINE
    assert cfg.also_run == ("Talos",)


def test_engine_config_store_corrupt_snapshot_does_not_crash_construction(tmp_path):
    path = tmp_path / "engine_config" / "latest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json")
    store = capture_daemon.EngineConfigStore(also_run=("Talos",), snapshot_path=path)
    cfg = store.get()
    assert cfg.primary == DEFAULT_PRIMARY_ENGINE
    assert cfg.also_run == ("Talos",)


# ---------------------------------------------------------------------------
# handle_ready_to_capture() -- also-run dispatch + combined write, live
# config, non-blocking.
# ---------------------------------------------------------------------------


def test_handle_ready_to_capture_with_no_engine_config_store_is_byte_identical_to_default(tmp_path):
    """engine_config_store=None (every pre-existing caller/test) must
    behave EXACTLY like an EngineConfigStore() default -- Apollo
    primary, no also-run, no other_engines key ever written."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )
    import json

    data = json.loads((dest_dir / "result.json").read_text())
    assert "primary_engine" not in data
    assert "other_engines" not in data


def test_handle_ready_to_capture_dispatches_also_run_engine_and_writes_other_engines(tmp_path):
    """The real end-to-end combined-write proof: Talos configured as an
    also-run engine actually lands under result.json's `other_engines`
    key, with `primary_engine` naming the real primary -- exactly
    docs/ENGINES.md's documented JSON shape."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(primary="Apollo", also_run=("Talos",), timeout_s=5.0)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        engine_config_store=store,
    background_save=False,
    )

    import json

    def _has_other_engines():
        data = json.loads((dest_dir / "result.json").read_text())
        return "other_engines" in data

    assert _wait_until(_has_other_engines), "other_engines never appeared in result.json"
    data = json.loads((dest_dir / "result.json").read_text())
        # a package-only codename swap on the wrapper dict's OWN keys only.
    assert data["primary_engine"] == "Apollo"
    talos = data["other_engines"]["Talos"]
    # Fake 4x4 images have no dart: the real engine cannot form shaft
    # planes (ok=False), not the old stub miss (ok=True / ring=outside).
    # The proof here is that Talos's (Talos') section landed, timed out false.
    assert talos["ok"] is False
    assert talos["sector"] is None
    assert talos["timed_out"] is False
    # diagnostics is the engine's OWN opaque payload, self-reported --
    # NOT touched by the package-only wrapper-key mapping above, so it
    # still says the engine's real internal name.
    assert talos.get("diagnostics", {}).get("engine") == "Talos"
    # Top-level (primary) fields are UNTOUCHED by the also-run write --
    # still exactly what score_dart() produced (an ok=False rejection
    # here, since the fake 4x4 images have no real dart in them -- the
    # POINT is that this field is whatever the primary computed, and
    # adding other_engines doesn't touch it).
    assert "ok" in data and "sector" in data and "ring" in data


def test_handle_ready_to_capture_emits_a_second_package_saved_when_also_run_completes(tmp_path):
    """Real regression test, 2026-08-13: the also-run dispatch background
    thread used to write other_engines/primary_engine into result.json
    with NO matching live-push notification of its own -- confirmed live
    on the real rig, the dashboard's primary-engine row (and every
    also-run row) stayed invisible until an UNRELATED future event (the
    next thrown dart) incidentally triggered a fresh package re-read.
    This asserts the fix: a SECOND on_event('PACKAGE_SAVED', ...) fires
    for THIS throw's own dest_dir once other_engines is actually written.

    **2026-09-01 update**: `background_save=False` below makes the FIRST
    PACKAGE_SAVED (the throw's own main save completing -- see "background
    the throw-package save") fire synchronously, before
    this call even returns -- genuinely a different event than before
    (there was no such emit here at all pre-2026-09-01; the caller emitted
    its own copy immediately instead). The also-run dispatch itself is
    STILL genuinely backgrounded regardless of background_save (its own
    separate thread, unaffected by this parameter) -- so this test now
    explicitly waits for the SECOND PACKAGE_SAVED specifically, matching
    its own name and original intent, rather than accepting the first one
    it sees.

    **2026-09-26 update**: the package's one clip is written after its
    data, inline here too (background_save=False), but it re-announces
    the package only in "mismatch" video-record mode -- there is no
    throw-capture service here, so it says nothing and the also-run
    completion is still the SECOND PACKAGE_SAVED."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(primary="Apollo", also_run=("Talos",), timeout_s=5.0)
    events: list[dict] = []

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        engine_config_store=store, on_event=events.append,
    background_save=False,
    )

    def _second_package_saved_arrived():
        hits = [e for e in events if e.get("type") == "PACKAGE_SAVED"]
        return len(hits) >= 2

    assert _wait_until(_second_package_saved_arrived), (
        f"no PACKAGE_SAVED event arrived for the also-run completion; "
        f"events so far: {events}"
    )
    hits = [e for e in events if e.get("type") == "PACKAGE_SAVED"]
    hit = hits[1]
    assert hit["path"] == str(dest_dir)
    assert hit["session"] == "sess1"
    # And it really does reflect the now-complete file, not a stale race.
    import json

    data = json.loads((dest_dir / "result.json").read_text())
    assert "other_engines" in data


def test_handle_ready_to_capture_never_blocks_on_a_slow_also_run_engine(tmp_path, monkeypatch):
    """THE load-bearing property, mirroring
    test_handle_ready_to_capture_never_blocks_on_a_hanging_ad_match's own
    pattern exactly (tests/test_capture_daemon.py): a deliberately slow
    also-run engine must not delay handle_ready_to_capture()'s return at
    all, and the timeout must actually fire in the background write."""
    HANG_SECONDS = 2.0
    real_score = TalosEngine.score

    def slow_score(self, bg_images, frame_images, calibration):
        time.sleep(HANG_SECONDS)
        return real_score(self, bg_images, frame_images, calibration)

    monkeypatch.setattr(TalosEngine, "score", slow_score)

    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(primary="Apollo", also_run=("Talos",), timeout_s=0.3)

    started = time.monotonic()
    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        engine_config_store=store,
    background_save=False,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, (
        f"handle_ready_to_capture() took {elapsed:.2f}s -- it must return almost "
        f"immediately regardless of a hanging also-run engine (simulated "
        f"{HANG_SECONDS}s hang); this is the exact regression this test exists to catch"
    )
    assert (dest_dir / "result.json").exists()

    import json

    def _has_other_engines():
        data = json.loads((dest_dir / "result.json").read_text())
        return "other_engines" in data

    assert _wait_until(_has_other_engines, timeout_s=HANG_SECONDS + 2.0)
    data = json.loads((dest_dir / "result.json").read_text())
    assert data["other_engines"]["Talos"]["timed_out"] is True  # Talos -> Talos


def test_handle_ready_to_capture_also_run_dispatch_failure_never_breaks_the_primary_save(
    tmp_path, monkeypatch
):
    """Mirrors test_handle_ready_to_capture_survives_a_crashing_ad_match:
    if the background also-run dispatch itself blows up (e.g. a bug in
    write_other_engines_result), the primary result.json (already saved
    synchronously) must be completely unaffected."""

    def raising_write(*args, **kwargs):
        raise RuntimeError("simulated also-run write failure")

    monkeypatch.setattr(capture_daemon, "write_other_engines_result", raising_write)

    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(primary="Apollo", also_run=("Talos",), timeout_s=2.0)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        engine_config_store=store,
    background_save=False,
    )
    assert (dest_dir / "result.json").exists()
    import json

    data = json.loads((dest_dir / "result.json").read_text())
    assert "ok" in data  # primary fields present and intact
    assert "other_engines" not in data  # the (failed) write never landed -- honest, not corrupted


# ---------------------------------------------------------------------------
# run_capture_loop_body() -- THE trigger-loop-never-blocks-on-scoring
# proof, at the loop level (not just handle_ready_to_capture() alone).
# Mirrors test_run_capture_loop_body_forces_takeout_waiting_after_third_
# dart_and_restores_baseline's scripted-advance() harness.
# ---------------------------------------------------------------------------


def test_run_capture_loop_body_iteration_timing_unaffected_by_slow_also_run_engine(
    tmp_path, monkeypatch
):
    """The real proof requested by this task: with a REAL (not
    monkeypatched) handle_ready_to_capture() wired to a deliberately slow
    also-run engine (Talos.score() sleeps 3s), running one READY_TO_CAPTURE
    through the REAL run_capture_loop_body() must complete in well under
    3s -- the trigger/settle state machine's own iteration/state
    progression is not waiting on scoring, regardless of engine speed.

    STALE FRAMING, NOTED NOT REWRITTEN, 2026-09-06/07: at the time this
    was written, `handle_ready_to_capture()` itself still ran
    synchronously on the loop's own thread (only ALSO-RUN engines were
    already backgrounded) -- the assertion below was genuinely testing
    "does the also-run dispatch alone keep the loop unblocked while the
    PRIMARY engine still runs on the loop's own thread." Part 2 of the
    Zeus-latency follow-up task backgrounds the ENTIRE `handle_ready_to_
    capture()` call (see `_dispatch_handle_ready_to_capture_in_
    background()`'s own docstring), which makes this test's own
    assertion trivially, MORE true than it originally was to prove (the
    primary engine no longer runs on the loop's thread here either) --
    left passing and unmodified rather than rewritten, since it's still
    a real, valid regression guard for the also-run case specifically;
    see `tests/test_handle_ready_to_capture_background_timing.py` for
    the newer, narrower proof of the PRIMARY-engine case this test's own
    docstring used to (incidentally, not by design) not cover."""
    HANG_SECONDS = 3.0
    real_score = TalosEngine.score
    talos_entered = threading.Event()

    def slow_score(self, bg_images, frame_images, calibration):
        talos_entered.set()
        time.sleep(HANG_SECONDS)
        return real_score(self, bg_images, frame_images, calibration)

    monkeypatch.setattr(TalosEngine, "score", slow_score)

    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frame = {0: np.full((2, 2, 3), 20, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    fetch_sequence = [true_baseline] * STARTUP_FETCHES + [dart_frame]
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart_frame

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=1,
            true_baseline_frames=true_baseline, last_frame=dart_frame,
        ),
    ]
    advance_calls = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        idx = advance_calls["n"]
        advance_calls["n"] += 1
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after the one scripted READY_TO_CAPTURE")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    # handle_ready_to_capture is DELIBERATELY left real (not monkeypatched)
    # -- this is the whole point of this test, unlike every other loop-body
    # test in tests/test_capture_daemon.py, which fakes it out.

    engine_config_store = capture_daemon.EngineConfigStore(
        primary="Apollo", also_run=("Talos",), timeout_s=0.3,
    )

    stop_event = threading.Event()
    started = time.monotonic()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            engine_config_store=engine_config_store,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 1.5, (
        f"run_capture_loop_body() took {elapsed:.2f}s to process one READY_TO_CAPTURE "
        f"and raise -- a {HANG_SECONDS}s also-run engine must NOT be on the loop's own "
        f"critical path (docs/ENGINES.md's Execution model); this is the exact "
        f"regression this test exists to catch"
    )
    # TEST ISOLATION, not part of the property above. The throw is scored
    # on a background thread that outlives this test; if its also-run
    # dispatch has not reached Talos by the time monkeypatch restores
    # TalosEngine.score, it calls whatever the NEXT test installed there --
    # and a next test that counts Talos calls then counts ours. Waiting for
    # the (patched) slow_score to be entered pins that call to this test,
    # without waiting out its HANG_SECONDS. It used to win the race by
    # luck; the package save got a few ms slower (clips instead of PNGs)
    # and it stopped.
    assert talos_entered.wait(5.0), "the also-run Talos dispatch never happened"


# ---------------------------------------------------------------------------
# 2026-08-27 perf task -- when Zeus is primary and also_run overlaps its
# own ZEUS_SUB_ENGINE_NAMES, reuse Zeus's already-computed sub-results
# instead of re-dispatching them a second time. See
# `_reuse_zeus_sub_results_for_also_run()`'s own docstring for the full
# reasoning; these tests prove (1) the pure extraction logic, (2) real
# duplicate-compute elimination (each sub-engine's score() called once,
# not twice), (3) byte-identical other_engines output (modulo timing
# fields) whether an engine's result came from live dispatch or reuse,
# (4) zero behavior change for every other primary/also_run shape.
# ---------------------------------------------------------------------------


def _read_result_json(path) -> dict:
    """Safe read for a possibly-mid-write result.json --
    `write_other_engines_result()`'s own `Path.write_text()` is not
    atomic (truncates then writes), so a `_wait_until()` poll can
    genuinely observe an empty/partial file mid-write. Returns `{}`
    (never a real "no other_engines yet" answer, so a caller's own
    predicate still correctly retries) instead of letting
    `json.JSONDecodeError` propagate out of `_wait_until()`'s own
    predicate loop -- a real, pre-existing race in this test file's
    already-established poll pattern (several existing tests above use
    the identical unprotected `json.loads(path.read_text())` inside a
    `_wait_until()` predicate), hardened here for this task's own new
    tests specifically rather than touching the shared pattern
    elsewhere in this file."""
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _zeus_diagnostics(sub_results: dict[str, EngineResult]) -> dict:
    """Builds the SAME diagnostics shape ZeusEngine.score() actually
    produces (base_diagnostics["sub_results"] = {n: sub_results[n].to_dict()
    for n in ZEUS_SUB_ENGINE_NAMES}), for pure-function tests of
    _reuse_zeus_sub_results_for_also_run() that don't need to run real
    detection code."""
    return {"sub_results": {n: r.to_dict() for n, r in sub_results.items()}}


def test_reuse_zeus_sub_results_returns_empty_when_primary_is_not_zeus():
    diagnostics = _zeus_diagnostics({
        "Apollo": EngineResult(ok=True, sector="20", ring="single", board_xy_mm=(0.0, 0.0)),
    })
    reused = capture_daemon._reuse_zeus_sub_results_for_also_run(
        "Apollo", diagnostics, ("Apollo", "Talos"),
    )
    assert reused == {}


def test_reuse_zeus_sub_results_returns_empty_when_diagnostics_falsy():
    assert capture_daemon._reuse_zeus_sub_results_for_also_run("Zeus", None, ("Apollo",)) == {}
    assert capture_daemon._reuse_zeus_sub_results_for_also_run("Zeus", {}, ("Apollo",)) == {}


def test_reuse_zeus_sub_results_returns_empty_when_no_sub_results_key():
    reused = capture_daemon._reuse_zeus_sub_results_for_also_run(
        "Zeus", {"n_usable": 4}, ("Apollo",),
    )
    assert reused == {}




def test_reuse_zeus_sub_results_round_trips_via_engine_result_from_dict():
    """The reused values are real EngineResult objects, reconstructed
    losslessly from Zeus's own already-serialized diagnostics -- not
    plain dicts a caller would need to special-case."""
    original = EngineResult(
        ok=False, sector=None, ring=None, board_xy_mm=None,
        reason="no dart found", diagnostics={"engine": "Talos"},
        duration_s=0.033, timed_out=False,
    )
    diagnostics = _zeus_diagnostics({"Talos": original})
    reused = capture_daemon._reuse_zeus_sub_results_for_also_run(
        "Zeus", diagnostics, ("Talos",),
    )
    assert reused["Talos"] == original


def test_handle_ready_to_capture_zeus_primary_reuses_sub_results_no_duplicate_dispatch(
    tmp_path, monkeypatch
):
    """THE real, end-to-end duplicate-compute-elimination proof: with
    Zeus as primary and also_run set to exactly ZEUS_SUB_ENGINE_NAMES
    (the live-configured shape), each real
    sub-engine's own score() must be called EXACTLY ONCE per throw, not
    twice -- confirmed by counting real calls via monkeypatched wrappers
    around each of the 4 real engine classes."""
    call_counts = {"Apollo": 0, "Talos": 0, "Athena": 0, "Ares": 0}

    def _counting(cls, name):
        real_score = cls.score

        def counting_score(self, bg_images, frame_images, calibration, **kwargs):
            call_counts[name] += 1
            return real_score(self, bg_images, frame_images, calibration, **kwargs)

        monkeypatch.setattr(cls, "score", counting_score)

    _counting(ApolloEngine, "Apollo")
    _counting(TalosEngine, "Talos")
    _counting(AthenaEngine, "Athena")
    _counting(AresEngine, "Ares")

    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(
        primary="Zeus", also_run=ZEUS_SUB_ENGINE_NAMES, timeout_s=5.0,
    )

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        engine_config_store=store,
    background_save=False,
    )

    def _has_other_engines():
        return "other_engines" in _read_result_json(dest_dir / "result.json")

    assert _wait_until(_has_other_engines), "other_engines never appeared in result.json"

    # THE proof: exactly once each, not twice (once for the Zeus primary
    # call, once again for the also-run dispatch -- the bug this task
    # fixes) and not zero (still genuinely ran).
    for name, count in call_counts.items():
        assert count == 1, (
            f"{name}.score() was called {count} times for one throw with Zeus "
            f"primary + overlapping also_run -- expected exactly 1 (reused for "
            f"the also-run write, not re-dispatched)"
        )

    data = _read_result_json(dest_dir / "result.json")
    assert data["primary_engine"] == "Zeus"  # Zeus -> Zeus package codename
    # All 4 sub-engines present under other_engines, via their own
    # package display codenames.
    assert set(data["other_engines"]) == {"Apollo", "Talos", "Athena", "Ares"}
    for name in ("Apollo", "Talos", "Athena", "Ares"):
        assert data["other_engines"][name]["timed_out"] is False




def test_handle_ready_to_capture_reused_other_engines_entry_matches_live_dispatch_byte_for_byte(
    tmp_path,
):
    """The exact byte-identical proof this task's own instructions ask
    for: an also-run engine's `other_engines` entry must be identical
    whether it came from live dispatch (non-Zeus-primary path) or reuse
    (Zeus-primary path), MODULO duration_s/timed_out -- a reader of
    result.json must not be able to tell the difference otherwise.
    Fixed 4x4 fake images make Apollo's own real score() deterministic
    (always the same ok=False rejection, same reason, same diagnostics
    shape), so this compares two REAL runs directly, not a mock."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    # Run A: Apollo primary + Talos also-run -- Talos is dispatched
    # LIVE (Apollo isn't Zeus, so no reuse path is even reachable).
    store_live = capture_daemon.EngineConfigStore(
        primary="Apollo", also_run=("Talos",), timeout_s=5.0,
    )
    dest_live = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages_live", "sess1",
        engine_config_store=store_live,
    background_save=False,
    )
    assert _wait_until(
        lambda: "other_engines" in _read_result_json(dest_live / "result.json")
    )
    live_talos = _read_result_json(dest_live / "result.json")["other_engines"]["Talos"]

    # Run B: Zeus primary + Talos also-run -- Talos IS one of
    # ZEUS_SUB_ENGINE_NAMES, so this throw's Talos entry comes from the
    # REUSE path (Zeus's own already-computed sub-result), never
    # dispatch_engines() at all.
    store_reused = capture_daemon.EngineConfigStore(
        primary="Zeus", also_run=("Talos",), timeout_s=5.0,
    )
    dest_reused = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages_reused", "sess1",
        engine_config_store=store_reused,
    background_save=False,
    )
    assert _wait_until(
        lambda: "other_engines" in _read_result_json(dest_reused / "result.json")
    )
    reused_talos = _read_result_json(dest_reused / "result.json")["other_engines"]["Talos"]

    # Byte-identical modulo duration_s/timed_out.
    for key in ("ok", "sector", "ring", "board_xy_mm", "reason", "diagnostics", "confidence"):
        assert live_talos[key] == reused_talos[key], (
            f"{key!r} differs between live-dispatched and reused-from-Zeus "
            f"Talos results: {live_talos[key]!r} vs {reused_talos[key]!r}"
        )
    # Both honestly report timed_out=False (neither actually timed out) --
    # duration_s is real timing data in BOTH cases (dispatch_engines()'s
    # own timing for the live path, Zeus's own per-sub-engine timing --
    # see _score_sub_engine()'s docstring -- for the reused path), so it's
    # legitimately allowed to differ, but both must still be real,
    # non-negative numbers, never a silently-wrong fabricated 0.0.
    assert live_talos["timed_out"] is False
    assert reused_talos["timed_out"] is False
    assert live_talos["duration_s"] >= 0.0
    assert reused_talos["duration_s"] >= 0.0


def test_handle_ready_to_capture_non_zeus_primary_still_dispatches_also_run_live(tmp_path):
    """Zero-behavior-change regression guard: a non-Zeus primary with an
    also_run overlapping ZEUS_SUB_ENGINE_NAMES' own name set must still
    go through the ordinary live dispatch path -- the reuse mechanism is
    scoped strictly to primary_name == "Zeus", per this task's own
    explicit scope."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    store = capture_daemon.EngineConfigStore(
        primary="Apollo", also_run=("Talos", "Athena"), timeout_s=5.0,
    )

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        engine_config_store=store,
    background_save=False,
    )
    assert _wait_until(
        lambda: "other_engines" in _read_result_json(dest_dir / "result.json")
    )
    data = _read_result_json(dest_dir / "result.json")
    assert data["primary_engine"] == "Apollo"
    assert set(data["other_engines"]) == {"Talos", "Athena"}
