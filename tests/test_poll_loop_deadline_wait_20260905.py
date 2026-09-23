"""Tests for `opendarts.live.capture_daemon._wait_for_next_iteration()` --
the 2026-09-05 Zeus-latency follow-up task's deadline-compensated
replacement for a flat `stop_event.wait(poll_interval_s)` at the tail of
`run_capture_loop_body()`'s main loop. See that function's own docstring
for the full incident/design.

Two things this file must prove, per the task's own explicit instruction:
  1. A slow iteration's own overshoot is ABSORBED into the next wait
     (shortened), not added on top of a still-full flat interval.
  2. The `stop_event` interrupt semantic is unchanged -- a set stop_event
     still returns the wait immediately, not after the full remaining
     duration.
"""
from __future__ import annotations

import threading
import time

import opendarts.live.capture_daemon as capture_daemon


def test_deadline_compensated_wait_shortens_when_body_already_took_time():
    """A real, measured falsification of the accumulation bug: simulate
    an iteration whose own body already consumed a real chunk of the
    poll interval (a `time.sleep()` between `iteration_started` and the
    wait call, standing in for a slow fetch/advance/save), and confirm
    the ACTUAL wait duration is correspondingly SHORTER than the full
    configured poll_interval_s -- not the full interval piled on top."""
    stop_event = threading.Event()
    poll_interval_s = 0.10
    body_cost_s = 0.04

    iteration_started = time.monotonic()
    time.sleep(body_cost_s)  # stand-in for real per-iteration work

    t0 = time.monotonic()
    capture_daemon._wait_for_next_iteration(stop_event, iteration_started, poll_interval_s)
    actual_wait_s = time.monotonic() - t0

    # Expected wait is ~poll_interval_s - body_cost_s (~0.06s), NOT the
    # full 0.10s a flat `stop_event.wait(poll_interval_s)` would have
    # used regardless of body_cost_s. Real margin, not a knife's edge:
    # the OLD behavior would measure ~0.10s here; this asserts well
    # below that, and comfortably above zero (this iteration did NOT
    # overrun its own deadline).
    assert actual_wait_s < poll_interval_s - body_cost_s + 0.03, (
        f"expected a compensated wait near {poll_interval_s - body_cost_s:.3f}s, "
        f"got {actual_wait_s:.3f}s -- looks like a flat, uncompensated wait"
    )
    assert actual_wait_s > 0.0

    # The TOTAL per-iteration cost (body + wait) should land close to the
    # configured poll_interval_s, not poll_interval_s + body_cost_s --
    # this is the actual real-world claim the fix makes.
    total_s = (time.monotonic() - iteration_started)
    assert total_s < poll_interval_s + 0.03, (
        f"total iteration cost {total_s:.3f}s should stay near the configured "
        f"poll_interval_s={poll_interval_s:.3f}s, not accumulate body_cost_s on top"
    )


def test_deadline_compensated_wait_absorbs_overshoot_across_a_full_two_iteration_sequence():
    """The task's own explicitly-requested scenario: 'construct a
    scenario with a slow iteration followed by normal ones, confirm the
    NEXT wait is shortened to compensate, not a flat interval.' Runs TWO
    consecutive simulated iterations (slow body, then a fast one) and
    measures the real wall-clock cost of each iteration's own
    (body + wait) -- both should land near poll_interval_s, not the
    first one running long while the second pays no compensation at all
    (which is what a flat `stop_event.wait(poll_interval_s)` would
    produce: iteration 1 = body_cost_s + poll_interval_s)."""
    stop_event = threading.Event()
    poll_interval_s = 0.08

    # Iteration 1: slow body (0.05s).
    it1_started = time.monotonic()
    time.sleep(0.05)
    capture_daemon._wait_for_next_iteration(stop_event, it1_started, poll_interval_s)
    it1_total_s = time.monotonic() - it1_started

    # Iteration 2: fast body (essentially none) -- starts its own fresh
    # deadline, same as the real loop's own `iteration_started =
    # time.monotonic()` at the top of every pass.
    it2_started = time.monotonic()
    capture_daemon._wait_for_next_iteration(stop_event, it2_started, poll_interval_s)
    it2_total_s = time.monotonic() - it2_started

    # Both iterations' own total cost should stay close to the
    # configured interval -- the slow iteration's own overshoot must NOT
    # leak into iteration 2's own timing (each iteration computes its
    # OWN deadline from its OWN iteration_started, so there is nothing
    # to "carry over" in the first place -- this is the real proof that
    # overshoot is absorbed per-iteration, not accumulated additively).
    for label, total_s in (("iteration 1 (slow body)", it1_total_s), ("iteration 2 (fast body)", it2_total_s)):
        assert total_s < poll_interval_s + 0.03, (
            f"{label}: total cost {total_s:.3f}s should stay near "
            f"poll_interval_s={poll_interval_s:.3f}s"
        )


def test_deadline_compensated_wait_zero_when_body_already_overran_deadline():
    """An iteration whose own body already ran PAST the configured
    poll_interval_s (a genuinely slow real capture+engine-score
    iteration) must wait ZERO extra time, not add a full interval on top
    of an already-slow iteration -- the actual accumulation bug this fix
    removes, in its most direct form."""
    stop_event = threading.Event()
    poll_interval_s = 0.05

    iteration_started = time.monotonic()
    time.sleep(poll_interval_s * 2)  # body alone already exceeds the interval

    t0 = time.monotonic()
    capture_daemon._wait_for_next_iteration(stop_event, iteration_started, poll_interval_s)
    actual_wait_s = time.monotonic() - t0

    assert actual_wait_s < 0.01, (
        f"expected ~0s wait once the deadline has already passed, got {actual_wait_s:.3f}s"
    )


def test_deadline_compensated_wait_still_honors_flat_interval_on_a_fast_body():
    """Explicitly does NOT shorten the configured POLL_INTERVAL_SECONDS
    itself -- a genuinely fast (near-zero-cost) iteration must still wait
    close to the FULL poll_interval_s, exactly like the old flat
    `stop_event.wait(poll_interval_s)` did."""
    stop_event = threading.Event()
    poll_interval_s = 0.05

    iteration_started = time.monotonic()
    # No sleep -- body cost is ~0.

    t0 = time.monotonic()
    capture_daemon._wait_for_next_iteration(stop_event, iteration_started, poll_interval_s)
    actual_wait_s = time.monotonic() - t0

    assert actual_wait_s > poll_interval_s - 0.02, (
        f"a fast/no-op body should still wait close to the full "
        f"poll_interval_s={poll_interval_s:.3f}s, got {actual_wait_s:.3f}s"
    )


def test_stop_event_interrupt_semantic_preserved():
    """Real, load-bearing behavior this fix must not trade away: a
    `stop_event` set MID-WAIT must return the wait immediately, not
    after the full remaining duration -- confirmed with a real
    background thread setting the event partway through a real wait
    (not a mocked/monkeypatched `Event.wait`)."""
    stop_event = threading.Event()
    poll_interval_s = 2.0  # deliberately large -- if the interrupt didn't
    # work, this test would visibly hang for ~2s instead of failing fast.

    iteration_started = time.monotonic()

    def _set_after_delay():
        time.sleep(0.05)
        stop_event.set()

    setter = threading.Thread(target=_set_after_delay, daemon=True)
    setter.start()

    t0 = time.monotonic()
    capture_daemon._wait_for_next_iteration(stop_event, iteration_started, poll_interval_s)
    elapsed_s = time.monotonic() - t0
    setter.join(timeout=1.0)

    assert stop_event.is_set()
    assert elapsed_s < 0.5, (
        f"expected the wait to return promptly (~0.05s) once stop_event was set, "
        f"got {elapsed_s:.3f}s -- looks like the interrupt semantic was lost"
    )


def test_stop_event_already_set_returns_immediately_even_with_no_overshoot():
    """A stop_event that is ALREADY set before the wait call (the common
    real shutdown-mid-loop case) must return immediately regardless of
    how much of poll_interval_s remains."""
    stop_event = threading.Event()
    stop_event.set()
    poll_interval_s = 1.0

    iteration_started = time.monotonic()  # fresh -- full interval "remains"

    t0 = time.monotonic()
    capture_daemon._wait_for_next_iteration(stop_event, iteration_started, poll_interval_s)
    elapsed_s = time.monotonic() - t0

    assert elapsed_s < 0.05
