"""Tests for THE VISIT MODEL -- "which turn did this dart belong to, and
which dart of that turn was it" (added 2026-08-14, docs/LIVE_API.md).

A visit is a deliberately thin wrapper around turn bookkeeping the
trigger state machine ALREADY does (`ThrowTriggerState.dart_count`,
TAKEOUT_WAITING, the true-baseline comparison), not a second parallel
state machine. So the thing actually worth proving here is that the
wrapper stays welded to the real state machine:

  * a visit rotates at exactly ONE detected moment -- the real
    takeout-complete transition (`advance()` returning IDLE with
    dart_count reset), the same condition `run_capture_loop_body()`
    already uses to restore the true empty-board baseline -- plus a
    manual Reset, which explicitly means "abandon this visit";
  * every dart of one turn carries the SAME visit_id with 0-based
    indices 0..MAX_DARTS_PER_TURN-1, and the next turn's darts carry a
    different one;
  * those two fields survive a real `save_throw_package()` ->
    `load_throw_package()` / `discover_packages()` round trip on disk;
  * a package written BEFORE the visit model existed still loads, with
    both fields honestly None rather than fabricated (proven against the
    real archived corpus when it's present on this machine).

Per docs/DESIGN.md's filesystem discipline, scratch goes in <repo>/tmp/, never
/tmp or the harness scratchpad.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from opendarts.capture.throw_package import (
    load_throw_package,
    save_throw_package,
)
from opendarts.capture.trigger_state import MAX_DARTS_PER_TURN, ThrowState, ThrowTriggerState
from opendarts.live import capture_daemon
from tests.lifecycle_scripting import script_trigger
from opendarts.live.capture_daemon import ResetRequest, new_visit_id
from opendarts.live.server import discover_packages
from opendarts.pipeline import CameraCalibration, ScoreResult

REPO_ROOT = Path(__file__).resolve().parent.parent
@pytest.fixture()
def scratch(tmp_path):
    d = tmp_path / "run"
    d.mkdir(parents=True)
    return d


class _StopLoop(Exception):
    """Escape hatch to end run_capture_loop_body() after a scripted
    sequence -- same mechanism tests/test_capture_daemon.py already uses
    for its own scripted-turn tests, deliberately not a new one."""


def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((5, 1)),
        rvec=np.zeros((3, 1)),
        tvec=np.zeros((3, 1)),
        pnp_result=None,
        landmark_spread_ok=True,
    )


def _run_scripted_loop(scripted_states, *, scratch, reset_request=None, monkeypatch):
    """Drives the REAL run_capture_loop_body() through a scripted
    sequence of advance() return values, capturing every on_event push
    and every (visit_id, visit_index) handed to handle_ready_to_capture().

    Only `advance()` (the pure state machine, exercised on its own real
    frames elsewhere), the calibration bootstrap, the camera fetch, and
    the scoring/save step are faked -- the visit bookkeeping under test
    is the loop's own real code, untouched.
    """
    baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration()}

    def fake_fetch(dest_dir, *, hub=None):
        return baseline

    advance_calls = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        idx = advance_calls["n"]
        advance_calls["n"] += 1
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("scripted sequence exhausted")

    captured: list[dict] = []

    def fake_handle_ready_to_capture(
        trigger, bg_frames, calibrations, package_root, session_id, **kwargs
    ):
        captured.append(
            {"visit_id": kwargs.get("visit_id"), "visit_index": kwargs.get("visit_index")}
        )
        return package_root / f"fake_throw_{len(captured)}"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    events: list[dict] = []
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=scratch / "packages",
            poll_interval_s=0.0,
            stop_event=threading.Event(),
            scratch_dir=scratch / "scratch",
            on_event=events.append,
            reset_request=reset_request,
        )
    return events, captured, baseline


def _ready(dart_count: int, baseline) -> ThrowTriggerState:
    return ThrowTriggerState(
        state=ThrowState.READY_TO_CAPTURE,
        dart_count=dart_count,
        true_baseline_frames=baseline,
        last_frame=baseline,
    )


# --------------------------------------------------------------------------
# ID minting
# --------------------------------------------------------------------------

def test_new_visit_id_follows_this_repos_own_id_convention_not_a_uuid():
    """Deliberately not a UUID: this
    project's established ID shape is a millisecond epoch stamp
    (`throw_<ms>` in handle_ready_to_capture, `%Y%m%d-%H%M%S` for
    sessions), and matching it keeps every ID in the system sortable in
    the same chronological order. Pinned as a test because a future
    "let's just use uuid4" change would silently break that ordering
    property, which nothing else would catch."""
    visit_id = new_visit_id()
    assert visit_id.startswith("visit_")
    stamp = visit_id.split("_", 1)[1]
    assert stamp.isdigit()
    # Millisecond epoch, not seconds -- 13 digits through the year 2286.
    assert len(stamp) == 13
    # Sortable == chronological, the actual property being protected.
    assert new_visit_id() > visit_id


def test_new_visit_id_never_repeats_even_within_one_millisecond():
    """A real bug this suite caught, not a hypothetical: the first
    implementation was a bare `int(time.time() * 1000)`, and two visits
    minted back-to-back (a double Reset, or the takeout-rotate/next-turn
    sequence) produced the SAME ID. Since POST /api/visits/{visit_id}/
    throws/{index}/correct resolves a throw by (visit_id, index) alone,
    a collision would let a correction land on a dart from a different
    turn. 500 IDs in a tight loop covers many milliseconds' worth of
    same-millisecond minting on any real machine."""
    ids = [new_visit_id() for _ in range(500)]
    assert len(set(ids)) == len(ids)
    # Strictly increasing, so the sortable==chronological property above
    # survives the collision fix rather than being traded away for it.
    assert ids == sorted(ids)


# --------------------------------------------------------------------------
# Rotation -- welded to the real trigger-state transition
# --------------------------------------------------------------------------

def test_all_darts_of_one_turn_share_a_visit_and_get_0_based_indices(scratch, monkeypatch):
    """A full 3-dart turn: one visit_id across all three, indices 0/1/2
    (0-based, matching the `index` on the throw-correction route).
    `trigger.dart_count` is 1-based and already incremented by advance()
    when READY_TO_CAPTURE is reached, so an off-by-one here would be very
    easy to introduce and completely invisible without this test."""
    baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    scripted = [_ready(n, baseline) for n in (1, 2, 3)]

    _events, captured, _ = _run_scripted_loop(
        scripted, scratch=scratch, monkeypatch=monkeypatch
    )

    assert len(captured) == MAX_DARTS_PER_TURN
    assert [c["visit_index"] for c in captured] == [0, 1, 2]
    visit_ids = {c["visit_id"] for c in captured}
    assert len(visit_ids) == 1
    assert next(iter(visit_ids)).startswith("visit_")


def test_visit_rotates_on_the_real_takeout_complete_transition(scratch, monkeypatch):
    """THE rotation point. advance() reporting IDLE with dart_count==0
    from a non-IDLE state is the state machine's own "the board is
    confirmed genuinely empty again" verdict -- the same condition the
    loop already uses to restore the true baseline. Proves (a) a
    VISIT_CLEARED event fires there carrying both the old and new visit
    IDs and the real dart count of the turn that ended, and (b) the dart
    thrown AFTER the takeout is filed under the NEW visit at index 0."""
    baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    scripted = [
        _ready(1, baseline),
        _ready(2, baseline),
        _ready(3, baseline),
        # Takeout complete, exactly as advance() reports it from
        # TAKEOUT_WAITING once the board matches the true baseline.
        ThrowTriggerState(state=ThrowState.IDLE, dart_count=0, true_baseline_frames=baseline),
        # First dart of the NEXT turn.
        _ready(1, baseline),
    ]

    events, captured, _ = _run_scripted_loop(
        scripted, scratch=scratch, monkeypatch=monkeypatch
    )

    cleared = [e for e in events if e["type"] == "VISIT_CLEARED"]
    assert len(cleared) == 1, "exactly one rotation for one completed takeout"
    assert cleared[0]["reason"] == "takeout"
    assert cleared[0]["n_darts"] == MAX_DARTS_PER_TURN

    first_turn_visit = captured[0]["visit_id"]
    second_turn_visit = captured[3]["visit_id"]
    assert cleared[0]["previous_visit_id"] == first_turn_visit
    assert cleared[0]["visit_id"] == second_turn_visit
    assert second_turn_visit != first_turn_visit
    # The new turn restarts indexing at 0 -- not a running counter.
    assert captured[3]["visit_index"] == 0


def test_no_visit_rotation_while_a_turn_is_still_in_progress(scratch, monkeypatch):
    """The negative half of the rotation contract: darts 1 and 2 land,
    the trigger passes through the intermediate states, and NOTHING
    rotates. A visit model implemented as its own parallel state machine
    (e.g. rotating on any return to IDLE, which the loop reaches between
    darts) would fail exactly here."""
    baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    scripted = [
        _ready(1, baseline),
        ThrowTriggerState(
            state=ThrowState.MOTION_DETECTED, dart_count=1, true_baseline_frames=baseline
        ),
        ThrowTriggerState(
            state=ThrowState.SETTLING, dart_count=1, true_baseline_frames=baseline
        ),
        _ready(2, baseline),
    ]

    events, captured, _ = _run_scripted_loop(
        scripted, scratch=scratch, monkeypatch=monkeypatch
    )

    assert [e for e in events if e["type"] == "VISIT_CLEARED"] == []
    assert len({c["visit_id"] for c in captured}) == 1


def test_manual_reset_rotates_the_visit_and_says_why(scratch, monkeypatch):
    """A manual Reset means "abandon whatever turn was in flight"
    ("clear visit / escape stuck takeout"), so it
    must rotate too -- otherwise the next dart thrown would be filed
    under the same visit as the darts the operator just discarded. The
    `reason` field is what lets a consumer tell the two rotation causes
    apart."""
    baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    reset = ResetRequest()
    reset.request()
    scripted = [_ready(1, baseline)]

    events, captured, _ = _run_scripted_loop(
        scripted, scratch=scratch, reset_request=reset, monkeypatch=monkeypatch
    )

    cleared = [e for e in events if e["type"] == "VISIT_CLEARED"]
    assert len(cleared) == 1
    assert cleared[0]["reason"] == "reset"
    # The reset is checked BEFORE advance(), so it fires on the very
    # first iteration -- before any dart is captured.
    assert cleared[0]["n_darts"] == 0
    # The dart captured after the reset belongs to the post-reset visit.
    assert captured[0]["visit_id"] == cleared[0]["visit_id"]


def test_trigger_state_events_carry_the_current_visit_id(scratch, monkeypatch):
    """TRIGGER_STATE gained a `visit_id` key (an ADDED key on an existing
    message -- `state`/`session`/`dart_count` are all untouched, which
    the dashboard's own pill JS still depends on). This is what lets a
    client that connects mid-turn learn the current visit without
    waiting for the next throw."""
    baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    scripted = [_ready(1, baseline)]

    events, _captured, _ = _run_scripted_loop(
        scripted, scratch=scratch, monkeypatch=monkeypatch
    )

    trigger_events = [e for e in events if e["type"] == "TRIGGER_STATE"]
    assert trigger_events, "the loop always emits at least the startup TRIGGER_STATE"
    for event in trigger_events:
        # Nothing pre-existing was dropped.
        assert {"state", "session", "dart_count"} <= set(event)
        assert event["visit_id"].startswith("visit_")


# --------------------------------------------------------------------------
# On-disk schema
# --------------------------------------------------------------------------

def _write_package(dest_dir: Path, **visit_kwargs) -> Path:
    """One real package via the REAL save_throw_package(), not a
    hand-rolled meta.json -- so these prove the actual on-disk format."""
    return save_throw_package(
        dest_dir=dest_dir,
        session="session-visit-test",
        bg_frames_bgr={0: np.zeros((8, 8, 3), dtype=np.uint8)},
        dart_frames_bgr={0: np.full((8, 8, 3), 255, dtype=np.uint8)},
        calibrations={0: _fake_calibration()},
        result=ScoreResult(
            ok=True,
            sector="20",
            ring="treble",
            board_xy_mm=(1.0, 2.0),
            triangulation=None,
            n_cameras_used=1,
            reason="",
            max_ray_disagreement_mm=None,
        ),
        **visit_kwargs,
    )


def test_visit_fields_round_trip_through_a_real_package_on_disk(scratch):
    pkg = _write_package(
        scratch / "packages" / "session-visit-test" / "throw_1",
        visit_id="visit_1700000000000",
        visit_index=2,
    )

    meta = json.loads((pkg / "meta.json").read_text())
    assert meta["visit_id"] == "visit_1700000000000"
    assert meta["visit_index"] == 2

    loaded = load_throw_package(pkg)
    assert loaded.visit_id == "visit_1700000000000"
    assert loaded.visit_index == 2

    listed = discover_packages(scratch / "packages")
    assert [(p["visit_id"], p["visit_index"]) for p in listed] == [
        ("visit_1700000000000", 2)
    ]


def test_visit_index_zero_is_written_not_swallowed_as_falsy(scratch):
    """`if visit_index is not None`, never `if visit_index` -- dart 0 of
    every single visit has index 0, so a falsy check would silently drop
    the field for a third of all real throws. Cheap test, real bug
    class."""
    pkg = _write_package(
        scratch / "packages" / "session-visit-test" / "throw_0",
        visit_id="visit_1700000000000",
        visit_index=0,
    )
    assert json.loads((pkg / "meta.json").read_text())["visit_index"] == 0
    assert load_throw_package(pkg).visit_index == 0


def test_a_package_saved_without_a_visit_stays_valid_and_honestly_null(scratch):
    """Backward compatibility by construction: the keys are OMITTED (not
    written as nulls), and every reader degrades to None rather than
    fabricating "visit 0, dart 0" -- same convention a package with no
    ad_ground_truth.json already follows."""
    pkg = _write_package(scratch / "packages" / "session-visit-test" / "throw_old")

    meta = json.loads((pkg / "meta.json").read_text())
    assert "visit_id" not in meta
    assert "visit_index" not in meta

    loaded = load_throw_package(pkg)
    assert loaded.visit_id is None
    assert loaded.visit_index is None

    listed = discover_packages(scratch / "packages")
    assert listed[0]["visit_id"] is None
    assert listed[0]["visit_index"] is None


# --------------------------------------------------------------------------
# Real archived corpus -- the actual backward-compat population
# --------------------------------------------------------------------------

def _corpus_root() -> Path:
    """Same convention as tests/test_engine_apollo_far_end_recovery.py
    -- data/archive/
    is gitignored, so a fresh clone / an isolated worktree has none."""
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    if env_root:
        return Path(env_root) / "clean"
    return REPO_ROOT / "data" / "archive" / "clean"


def test_every_real_archived_package_still_loads_with_no_visit(scratch):
    """The visit model landed AFTER the ORIGINAL data/archive/clean/
    sessions were captured, so those are real backward-compat population
    -- not a hypothetical one. Proves the new fields don't break the real
    on-disk format this project's every accuracy number depends on: each
    LEGACY package still lists cleanly, with visit_id/visit_index
    honestly None.

    **2026-08-14 update**: the corpus has since grown to include a
    session (pulled the same day this test's own
    blanket "every package has visit_id=None" assertion started failing
    for real) captured AFTER the visit model shipped -- that session's
    packages legitimately carry real visit_id/visit_index, which is
    correct forward behavior, not a regression. This test now checks
    each generation against its own honest expectation instead of
    assuming the whole corpus predates the feature forever: KNOWN_LEGACY
    session dirs must still show None (the actual backward-compat
    claim); everything else just has to load without crashing
    (len(listed) == len(pkg_dirs) below already proves that) and is
    allowed a real visit_id.

    Reads only -- never writes into data/archive/: that directory is
    shared across concurrent sessions with no isolation."""
    # Sessions captured BEFORE the visit model shipped (see
    # data/archive/clean/README.md's own session table) -- the actual
    # backward-compat population this test exists to protect. Any
    # FUTURE new session is assumed post-visit-model and intentionally
    # NOT added here; a session added to this corpus after this list was
    # last updated but before the feature that generated it existed
    # would need a human to notice and add it, same as any other corpus-
    # evolution assumption in this test file.
    KNOWN_LEGACY_SESSIONS = {"20260813-164658", "20260813-234015"}

    root = _corpus_root()
    pkg_dirs = sorted(p.parent for p in root.rglob("meta.json")) if root.exists() else []
    if not pkg_dirs:
        pytest.skip(
            "no real archived throw packages under data/archive/clean/ (or "
            "$OPENDARTS_ENGINE_CORPUS_ROOT) -- gitignored, absent in a fresh "
            "clone/worktree; this backward-compat proof only means something "
            "against the real corpus"
        )

    listed = discover_packages(root)
    assert len(listed) == len(pkg_dirs), (
        "every real archived package must still be discoverable after the "
        "meta.json schema gained visit fields"
    )

    legacy = [p for p in listed if p.get("session") in KNOWN_LEGACY_SESSIONS]
    if not legacy:
        # 2026-08-21 fix: the corpus is a living, curated thing that gets
        # deliberately reset (docs/DESIGN.md, and the 2026-08-21 reset itself
        # -- KNOWN_LEGACY_SESSIONS were quarantined out of data/archive/
        # clean/ that same day, replaced with a fresh post-visit-model
        # session). A blanket assert here treated "no legacy session
        # currently in the corpus" as a real regression -- it isn't; it's
        # the expected, documented outcome of a corpus reset removing
        # every pre-visit-model package. Skip cleanly (same posture as
        # the empty-corpus skip above) rather than fail: the ACTUAL
        # backward-compat proof (every package, old or new, still loads
        # -- len(listed) == len(pkg_dirs) above) already ran regardless.
        # This specific "a legacy package's visit_id/visit_index is
        # honestly None" check only means something when real
        # pre-visit-model data is present to check it against.
        pytest.skip(
            "no known-legacy (pre-visit-model) session currently in the "
            "corpus -- expected after a corpus reset (see docs/DESIGN.md); the "
            "backward-compat proof above (every package still loads) "
            "already ran"
        )
    assert all(p["visit_id"] is None for p in legacy)
    assert all(p["visit_index"] is None for p in legacy)
