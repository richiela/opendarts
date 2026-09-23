"""Script the capture loop's trigger decisions in tests.

The loop has one decision seam, ``capture_daemon._lifecycle_step()``,
which normally runs the lifecycle and returns a ``LiveStep``. Tests that
exercise the loop's own plumbing (visit events, package writing, throw
numbering, reset, heartbeat) don't want real frames judged -- they want to
dictate "this tick is READY_TO_CAPTURE with dart_count=2". ``script_trigger``
installs a fake seam driven by a legacy-style ``fake_advance(trigger,
bg_frames, current_frames) -> ThrowTriggerState`` callable, so the many
existing scripted tests keep their shape, and stubs the lifecycle driver
so no real driver (or its JSONL log) is built.

Takeout semantics: the loop used to infer "visit cleared" from a
dart_count>=1 -> IDLE/0 transition; the lifecycle reports it explicitly
via ``LiveStep.cleared_darts``. The fake seam infers it the old way so
scripted sequences read the same.
"""
from __future__ import annotations

from typing import Callable


from opendarts.capture.trigger_state import ThrowState, ThrowTriggerState
from opendarts.lifecycle.adapter import LiveStep
from opendarts.live import capture_daemon

FakeAdvance = Callable[[ThrowTriggerState, dict, dict], ThrowTriggerState]


class StubLifecycleDriver:
    disabled = False
    lifecycle = None

    class _Stats:
        @staticmethod
        def as_dict() -> dict:
            return {"stub": True}

    stats = _Stats()

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def close(self) -> None:
        pass


def script_trigger(monkeypatch, fake_advance: FakeAdvance) -> StubLifecycleDriver:
    """Replace the loop's lifecycle with ``fake_advance``'s scripted states."""
    stub = StubLifecycleDriver()
    monkeypatch.setattr(capture_daemon, "_build_lifecycle_driver", lambda **kwargs: stub)
    state = {"prev": None}

    def fake_step(lifecycle, adapter, current_frames, bg_frames, dropped_frames_total):
        prev = state["prev"]
        if prev is None:
            prev = ThrowTriggerState(true_baseline_frames=bg_frames)
        new = fake_advance(prev, bg_frames, current_frames)
        cleared = None
        reference = bg_frames
        if (
            prev.dart_count >= 1
            and new.dart_count == 0
            and new.state is ThrowState.IDLE
        ):
            cleared = prev.dart_count
            if new.true_baseline_frames is not None:
                reference = new.true_baseline_frames
        # the loop rebuilds `trigger` after a capture; mirror that so the
        # next scripted call sees what the loop would have held
        if new.state is ThrowState.READY_TO_CAPTURE:
            state["prev"] = ThrowTriggerState(
                state=(
                    ThrowState.TAKEOUT_WAITING
                    if new.dart_count >= capture_daemon.MAX_DARTS_PER_TURN
                    else ThrowState.IDLE
                ),
                dart_count=new.dart_count,
                true_baseline_frames=new.true_baseline_frames,
            )
        else:
            state["prev"] = new
        return LiveStep(trigger=new, reference=reference, cleared_darts=cleared)

    monkeypatch.setattr(capture_daemon, "_lifecycle_step", fake_step)
    return stub


#: How many fetches the loop's startup consumes before its first tick
#: (``_wait_for_first_frames`` returns on the first non-empty fetch). Scripted
#: fetch sequences prepend this many baseline frames.
STARTUP_FETCHES = 1


def idle_advance(trigger: ThrowTriggerState, bg_frames: dict, current_frames: dict) -> ThrowTriggerState:
    """A scripted step that never changes state -- for tests that only
    need to count how often the loop reached its decision point."""
    return trigger
