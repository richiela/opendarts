"""Tests for `opendarts.calibration.rig_move_reseed` -- the absolute-hint
rig-move reseed detector, 2026-08-31.

Real numbers embedded below reproduce the exact live incident this
module exists to catch (see that module's own top docstring):

```
                  before move      after move       shift
cam0 hint_deg:    ~272.6-273.4deg  ~269.9-270.2deg  ~3deg
cam1 hint_deg:    ~14.8-15.0deg    ~15.84-15.96deg  ~1deg
cam2 hint_deg:    ~165.9-166.3deg  ~164.5-164.7deg  ~1.3-1.4deg
```

Normal per-camera noise (BEFORE the move) was stable to within roughly
0.1-0.6deg across dozens of real events -- reproduced here via a fixed
small pseudo-random jitter, not literally re-measured from the corpus
(no data-directory dependency in the fast pytest suite, matching this
project's own `test_rig_ring_geometry.py` precedent one file over)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from opendarts.calibration.rig_move_reseed import (
    ABSOLUTE_HINT_HISTORY_MAX_SAMPLES,
    CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED,
    RESEED_DETECTOR_FALLBACK_FILENAME,
    SCHEMA,
    AbsoluteOrientationHistory,
    check_for_rig_move_and_reseed_if_confirmed,
    clear_ring_geometry_on_reseed,
    load_absolute_orientation_history,
    record_absolute_orientation_event,
    save_absolute_orientation_history,
)
from opendarts.calibration.rig_ring_geometry import (
    RING_GEOMETRY_FALLBACK_FILENAME,
    RingGeometry,
    save_ring_geometry,
)

# Real pre-move / post-move hint centers, from this module's own top
# docstring (the actual live incident).
BEFORE = {0: 273.0, 1: 14.9, 2: 166.1}
AFTER = {0: 270.05, 1: 15.90, 2: 164.60}


def _jitter(base: dict[int, float], seed: int, spread: float = 0.3) -> dict[int, float]:
    """A single event's hints: `base` plus small, bounded per-camera
    noise -- reproducing this rig's own real, documented ~0.1-0.6deg
    per-event stability (spread=0.3 keeps every sample within the real
    observed range with a fixed seed for reproducibility)."""
    rng = random.Random(seed)
    return {cam: v + rng.uniform(-spread, spread) for cam, v in base.items()}


# --- requirement 1: no-move stability (normal noise never reseeds) -----


def test_normal_noise_never_triggers_reseed_across_many_events():
    history = None
    decisions = []
    for i in range(40):
        hints = _jitter(BEFORE, seed=i)
        decision = record_absolute_orientation_event(history, hints, f"evt{i}")
        history = decision.new_history
        decisions.append(decision)
    assert not any(d.reseed_triggered for d in decisions)
    # Baseline should track close to the true center after settling.
    assert history.baselines[0] == pytest.approx(BEFORE[0], abs=0.3)
    assert history.baselines[1] == pytest.approx(BEFORE[1], abs=0.3)
    assert history.baselines[2] == pytest.approx(BEFORE[2], abs=0.3)


def test_normal_noise_streak_never_grows_past_zero_typically():
    """A camera occasionally brushing the threshold in isolation must
    not accumulate a streak (agreement-across-cameras requirement)."""
    history = None
    for i in range(20):
        hints = _jitter(BEFORE, seed=i)
        decision = record_absolute_orientation_event(history, hints, f"evt{i}")
        history = decision.new_history
        # Streak must reset to 0 every single normal event.
        assert history.consecutive_deviating_events == 0


# --- requirement 1: a real move (tonight's own real numbers) MUST fire --


def test_real_move_reconstruction_triggers_reseed_after_required_streak():
    """Reproduce tonight's real incident: stable events at BEFORE, then
    a persistent shift to AFTER held across multiple consecutive
    events -- must trigger a reseed exactly at
    CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED, not before, not never."""
    history = None
    # Establish a real baseline first (several stable events).
    for i in range(10):
        hints = _jitter(BEFORE, seed=100 + i)
        history = record_absolute_orientation_event(history, hints, f"pre{i}").new_history

    # Now feed the real post-move shift, held steady (matching "an
    # entire 10-event batch" in the real incident).
    reseed_event_index = None
    decisions = []
    for i in range(CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED + 2):
        hints = _jitter(AFTER, seed=200 + i, spread=0.1)
        decision = record_absolute_orientation_event(history, hints, f"post{i}")
        history = decision.new_history
        decisions.append(decision)
        if decision.reseed_triggered and reseed_event_index is None:
            reseed_event_index = i
            break  # don't keep feeding events past the reseed -- that's a
            # separate, later history the assertions below aren't about.

    assert reseed_event_index == CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED - 1
    reseed_decision = decisions[reseed_event_index]
    assert reseed_decision.reseed_reason is not None
    assert set(reseed_decision.cameras_deviating) == {0, 1, 2}
    # Real deviation magnitudes should roughly match the real incident.
    assert reseed_decision.deviations_deg[0] == pytest.approx(2.95, abs=0.5)
    assert reseed_decision.deviations_deg[1] == pytest.approx(1.0, abs=0.5)
    assert reseed_decision.deviations_deg[2] == pytest.approx(1.5, abs=0.5)
    # After reseed, the new baseline is the triggering event's own
    # observed hints (fresh start), close to AFTER.
    assert history.baselines[0] == pytest.approx(AFTER[0], abs=0.5)
    assert history.baselines[1] == pytest.approx(AFTER[1], abs=0.5)
    assert history.baselines[2] == pytest.approx(AFTER[2], abs=0.5)
    assert history.consecutive_deviating_events == 0
    assert history.n_events == 1


def test_real_move_does_not_reseed_before_the_required_streak_length():
    history = None
    for i in range(10):
        hints = _jitter(BEFORE, seed=300 + i)
        history = record_absolute_orientation_event(history, hints, f"pre{i}").new_history

    # Only 2 consecutive post-move events -- one short of the required
    # streak (default 3).
    for i in range(CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED - 1):
        hints = _jitter(AFTER, seed=400 + i, spread=0.1)
        decision = record_absolute_orientation_event(history, hints, f"post{i}")
        assert not decision.reseed_triggered
        history = decision.new_history
    assert history.consecutive_deviating_events == CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED - 1


def test_streak_breaks_and_resets_if_move_does_not_persist():
    """A move that shows up for one or two events and then reverts
    (never reaching the required streak) must fully reset, not leave a
    partial streak lingering that a later unrelated blip could complete."""
    history = None
    for i in range(10):
        hints = _jitter(BEFORE, seed=500 + i)
        history = record_absolute_orientation_event(history, hints, f"pre{i}").new_history

    # Two deviating events (builds a streak of 2)...
    for i in range(2):
        hints = _jitter(AFTER, seed=600 + i, spread=0.1)
        history = record_absolute_orientation_event(history, hints, f"dev{i}").new_history
    assert history.consecutive_deviating_events == 2

    # ...then reverts back to normal (streak must reset).
    hints = _jitter(BEFORE, seed=700)
    decision = record_absolute_orientation_event(history, hints, "revert")
    assert not decision.reseed_triggered
    assert decision.new_history.consecutive_deviating_events == 0
    history = decision.new_history

    # A single further deviating event now must NOT trigger (streak
    # restarted from 0, not resumed from 2).
    hints2 = _jitter(AFTER, seed=800, spread=0.1)
    decision2 = record_absolute_orientation_event(history, hints2, "dev-again")
    assert not decision2.reseed_triggered
    assert decision2.new_history.consecutive_deviating_events == 1


# --- requirement 4 / single-camera cases: must NEVER trigger a reseed --


def test_single_noisy_one_off_reading_on_one_camera_does_not_trigger():
    history = None
    for i in range(10):
        hints = _jitter(BEFORE, seed=900 + i)
        history = record_absolute_orientation_event(history, hints, f"pre{i}").new_history

    # One event: cam1 alone jumps by a real, large deviation; cam0/cam2
    # stay normal.
    spike = dict(_jitter(BEFORE, seed=1000))
    spike[1] = BEFORE[1] + 2.0
    decision = record_absolute_orientation_event(history, spike, "spike")
    assert not decision.reseed_triggered
    assert decision.cameras_deviating == [1]
    # Agreement requirement means the streak never even started.
    assert decision.new_history.consecutive_deviating_events == 0
    history = decision.new_history

    # Back to normal next event -- still nothing.
    hints = _jitter(BEFORE, seed=1100)
    decision2 = record_absolute_orientation_event(history, hints, "back-to-normal")
    assert not decision2.reseed_triggered


def test_genuinely_struggling_single_camera_across_many_events_never_triggers():
    """A camera with a real but temporary bad stretch (this rig's own
    historically weakest camera) drifting on ITS OWN for many events in
    a row must never accumulate toward a reseed -- only cam2 deviates,
    every event, for 15 events straight."""
    history = None
    for i in range(10):
        hints = _jitter(BEFORE, seed=1200 + i)
        history = record_absolute_orientation_event(history, hints, f"pre{i}").new_history

    decisions = []
    for i in range(15):
        hints = _jitter(BEFORE, seed=1300 + i, spread=0.15)
        hints[2] = BEFORE[2] + 2.5 + (i * 0.01)  # persistent, cam2-only drift
        decision = record_absolute_orientation_event(history, hints, f"struggle{i}")
        history = decision.new_history
        decisions.append(decision)

    assert not any(d.reseed_triggered for d in decisions)
    # Every one of these events had exactly 1 camera deviating (cam2).
    assert all(d.cameras_deviating == [2] for d in decisions)
    # cam0/cam1's own baselines are unaffected by cam2's own struggle.
    assert history.baselines[0] == pytest.approx(BEFORE[0], abs=0.3)
    assert history.baselines[1] == pytest.approx(BEFORE[1], abs=0.3)


def test_min_cameras_deviating_can_be_overridden_explicitly():
    """Confirms the agreement threshold is real and adjustable, not
    hardcoded to always require 2 of 3 -- with min_cameras_deviating=1
    a single camera's own deviation IS enough (documenting the knob
    exists; production leaves it at the safer default)."""
    history = None
    for i in range(5):
        hints = _jitter(BEFORE, seed=1400 + i)
        history = record_absolute_orientation_event(
            history, hints, f"pre{i}", min_cameras_deviating=1
        ).new_history

    for i in range(CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED):
        spike = dict(_jitter(BEFORE, seed=1500 + i, spread=0.1))
        spike[1] = BEFORE[1] + 2.0
        decision = record_absolute_orientation_event(
            history, spike, f"spike{i}", min_cameras_deviating=1
        )
        history = decision.new_history
    assert decision.reseed_triggered


# --- virgin / edge cases -------------------------------------------------


def test_virgin_history_seeds_without_reseeding():
    decision = record_absolute_orientation_event(None, BEFORE, "t0")
    assert not decision.reseed_triggered
    assert decision.new_history.baselines == BEFORE
    assert decision.new_history.n_events == 1
    assert decision.new_history.consecutive_deviating_events == 0


def test_new_camera_appearing_mid_history_has_no_comparison_this_event():
    history = record_absolute_orientation_event(None, {0: 273.0, 1: 14.9}, "t0").new_history
    # cam2 appears for the first time -- no baseline to compare against.
    decision = record_absolute_orientation_event(history, {0: 273.1, 1: 14.8, 2: 166.0}, "t1")
    assert 2 not in decision.deviations_deg
    assert not decision.reseed_triggered
    assert 2 in decision.new_history.baselines


def test_single_camera_event_never_reseeds_regardless_of_shift():
    history = record_absolute_orientation_event(None, {0: 273.0}, "t0").new_history
    for i in range(10):
        decision = record_absolute_orientation_event(history, {0: 200.0}, f"t{i}")
        history = decision.new_history
        assert not decision.reseed_triggered
        assert decision.new_history.consecutive_deviating_events == 0


def test_max_samples_caps_the_rolling_window():
    history = None
    for i in range(ABSOLUTE_HINT_HISTORY_MAX_SAMPLES + 20):
        hints = _jitter(BEFORE, seed=2000 + i)
        history = record_absolute_orientation_event(history, hints, f"evt{i}").new_history
    assert len(history.stable_samples[0]) == ABSOLUTE_HINT_HISTORY_MAX_SAMPLES


def test_circular_mean_handles_wraparound_correctly():
    """Values straddling 0deg/360deg must average to something near
    0deg/360deg, not near 180deg (the classic circular-mean bug this
    project has hit before -- see this module's own top docstring)."""
    history = None
    wrap_hints = [{0: 359.0}, {0: 1.0}, {0: 359.5}, {0: 0.5}]
    for i, hints in enumerate(wrap_hints):
        history = record_absolute_orientation_event(history, hints, f"t{i}").new_history
    # Should be close to 0/360, not 180.
    assert history.baselines[0] < 5.0 or history.baselines[0] > 355.0


# --- persistence ----------------------------------------------------------


def test_load_absent_file_returns_none(tmp_path: Path):
    assert load_absolute_orientation_history(tmp_path) is None


def test_load_none_root_returns_none():
    assert load_absolute_orientation_history(None) is None


def test_load_corrupt_json_returns_none(tmp_path: Path):
    path = tmp_path / RESEED_DETECTOR_FALLBACK_FILENAME
    path.write_text("{not json")
    assert load_absolute_orientation_history(tmp_path) is None


def test_load_wrong_schema_returns_none(tmp_path: Path):
    path = tmp_path / RESEED_DETECTOR_FALLBACK_FILENAME
    path.write_text(json.dumps({"schema": "some-other-v1"}))
    assert load_absolute_orientation_history(tmp_path) is None


def test_load_malformed_contents_returns_none(tmp_path: Path):
    path = tmp_path / RESEED_DETECTOR_FALLBACK_FILENAME
    path.write_text(json.dumps({"schema": SCHEMA, "baselines": "not a dict"}))
    assert load_absolute_orientation_history(tmp_path) is None


def test_save_load_round_trip(tmp_path: Path):
    history = AbsoluteOrientationHistory(
        baselines={0: 273.0, 1: 14.9, 2: 166.1},
        stable_samples={0: [272.8, 273.2], 1: [14.9], 2: [166.0, 166.2]},
        consecutive_deviating_events=1,
        n_events=5,
        first_learned_utc="2026-08-31T00:00:00Z",
        last_updated_utc="2026-08-31T01:00:00Z",
    )
    save_absolute_orientation_history(tmp_path, history)
    loaded = load_absolute_orientation_history(tmp_path)
    assert loaded is not None
    assert loaded.baselines == history.baselines
    assert loaded.stable_samples == history.stable_samples
    assert loaded.consecutive_deviating_events == 1
    assert loaded.n_events == 5


def test_clear_ring_geometry_on_reseed(tmp_path: Path):
    geometry = RingGeometry(
        gaps_deg=[103.33, 107.78, 148.89],
        n_events=3,
        first_learned_utc="2026-08-30T00:00:00Z",
        last_updated_utc="2026-08-30T00:00:00Z",
        spread_deg=[0.1, 0.1, 0.1],
    )
    save_ring_geometry(tmp_path, geometry)
    assert (tmp_path / RING_GEOMETRY_FALLBACK_FILENAME).exists()
    assert clear_ring_geometry_on_reseed(tmp_path) is True
    assert not (tmp_path / RING_GEOMETRY_FALLBACK_FILENAME).exists()
    assert clear_ring_geometry_on_reseed(tmp_path) is False


# --- end-to-end orchestration ---------------------------------------------


def test_check_and_reseed_end_to_end_clears_both_files(tmp_path: Path):
    """The full real scenario: establish geometry + history, then feed a
    real persistent move through the orchestration entry point, and
    confirm BOTH `ring_geometry_fallback.json` AND this module's own
    file end up reflecting a fresh post-reseed state."""
    geometry = RingGeometry(
        gaps_deg=[103.33, 107.78, 148.89],
        n_events=10,
        first_learned_utc="2026-08-25T00:00:00Z",
        last_updated_utc="2026-08-30T00:00:00Z",
        spread_deg=[0.2, 0.2, 0.2],
    )
    save_ring_geometry(tmp_path, geometry)

    for i in range(10):
        hints = _jitter(BEFORE, seed=3000 + i)
        check_for_rig_move_and_reseed_if_confirmed(tmp_path, hints, f"pre{i}")

    assert (tmp_path / RING_GEOMETRY_FALLBACK_FILENAME).exists()

    reseed_fired = False
    for i in range(CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED + 1):
        hints = _jitter(AFTER, seed=4000 + i, spread=0.1)
        decision = check_for_rig_move_and_reseed_if_confirmed(tmp_path, hints, f"post{i}")
        if decision.reseed_triggered:
            reseed_fired = True
            break

    assert reseed_fired
    # ring_geometry_fallback.json cleared as part of the reseed.
    assert not (tmp_path / RING_GEOMETRY_FALLBACK_FILENAME).exists()
    # This module's own file reflects the fresh, single-event state.
    loaded = load_absolute_orientation_history(tmp_path)
    assert loaded is not None
    assert loaded.n_events == 1
    assert loaded.consecutive_deviating_events == 0


def test_check_and_reseed_normal_events_never_touch_ring_geometry(tmp_path: Path):
    geometry = RingGeometry(
        gaps_deg=[103.33, 107.78, 148.89],
        n_events=5,
        first_learned_utc="2026-08-25T00:00:00Z",
        last_updated_utc="2026-08-30T00:00:00Z",
        spread_deg=[0.2, 0.2, 0.2],
    )
    save_ring_geometry(tmp_path, geometry)
    for i in range(20):
        hints = _jitter(BEFORE, seed=5000 + i)
        check_for_rig_move_and_reseed_if_confirmed(tmp_path, hints, f"evt{i}")
    assert (tmp_path / RING_GEOMETRY_FALLBACK_FILENAME).exists()


def test_check_and_reseed_with_none_root_does_not_raise():
    decision = check_for_rig_move_and_reseed_if_confirmed(None, BEFORE, "t0")
    assert not decision.reseed_triggered
    assert decision.new_history.n_events == 1


def test_check_and_reseed_does_not_block_or_alter_the_triggering_events_own_answer():
    """Requirement 3: the reseed must never be a refusal -- the
    triggering event's own decision object always carries a real,
    usable new_history regardless of reseed_triggered, and the function
    never raises even when a reseed fires."""
    history = None
    for i in range(10):
        hints = _jitter(BEFORE, seed=6000 + i)
        history = record_absolute_orientation_event(history, hints, f"pre{i}").new_history
    for i in range(CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED):
        hints = _jitter(AFTER, seed=7000 + i, spread=0.1)
        decision = record_absolute_orientation_event(history, hints, f"post{i}")
        history = decision.new_history
        # Never raises, always returns a usable object.
        assert decision.new_history is not None
