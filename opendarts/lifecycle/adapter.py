"""Translate lifecycle ticks into what the capture loop already consumes.

The capture loop (``opendarts.live.capture_daemon.run_capture_loop_body``)
consumes ``opendarts.capture.trigger_state.ThrowTriggerState``: it reads
``state``/``dart_count`` for UI events and the READY_TO_CAPTURE branch,
``last_frame`` and the loop's ``bg_frames`` for the engines, and
``true_baseline_frames`` as the current reference.
:class:`LifecycleTriggerAdapter` produces that object from each
:class:`~opendarts.lifecycle.state.Tick`, so ``handle_ready_to_capture()``
and every package/corpus artifact it writes keep their shape.

Mapping::

    COMMIT                         -> READY_TO_CAPTURE  (dart_count already incremented,
                                                          last_frame = the commit frames)
    HAND / SCENE_CHANGE            -> MOTION_DETECTED
    PENDING_DART                   -> SETTLING
    TAKEOUT_PENDING                -> TAKEOUT_WAITING
    WARMUP / IDLE / COOLDOWN       -> TAKEOUT_WAITING if dart_count == max_darts else IDLE
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from opendarts.capture.lazy_frame import as_frames, decode_all, prefetch
from opendarts.capture.trigger_state import MAX_DARTS_PER_TURN, ThrowState, ThrowTriggerState
from opendarts.lifecycle.state import Action, Lifecycle, Phase, Tick

_MOTION_PHASES = frozenset({Phase.HAND, Phase.SCENE_CHANGE})


@dataclass
class LiveStep:
    """What the loop must do with one tick."""

    trigger: ThrowTriggerState
    #: the lifecycle's current per-camera full-res reference. On a COMMIT
    #: tick this is the frame the engines must score against (the board
    #: right before this dart); every other tick it simply keeps the
    #: loop's ``bg_frames``/``motion_bg_frames`` in step with the reference.
    reference: dict[int, np.ndarray]
    #: number of darts removed when this tick cleared the visit, else None
    cleared_darts: int | None = None
    reason: str = ""


class LifecycleTriggerAdapter:
    def __init__(self, max_darts: int = MAX_DARTS_PER_TURN) -> None:
        self.max_darts = max_darts
        self._pending_since: float | None = None
        self._baseline: dict[int, np.ndarray] = {}
        self._baseline_ids: tuple = ()

    # ------------------------------------------------------------------
    def _baseline_for(self, lc: Lifecycle) -> dict[int, np.ndarray]:
        """The reference as ``true_baseline_frames`` -- same dict object as
        long as no camera's reference frame changed, so identity-based
        'did the baseline move' checks in the loop stay meaningful."""
        # Identity of the HANDLES: a lazy reference is not decoded just to
        # ask whether it moved.
        handles = lc.refs.full_handles()
        ids = tuple(sorted((cam, id(f)) for cam, f in handles.items()))
        if ids != self._baseline_ids:
            self._baseline = as_frames(handles)
            self._baseline_ids = ids
        return self._baseline

    def map_state(self, tick: Tick) -> ThrowState:
        if tick.action is Action.COMMIT:
            return ThrowState.READY_TO_CAPTURE
        if tick.phase in _MOTION_PHASES:
            return ThrowState.MOTION_DETECTED
        if tick.phase is Phase.PENDING_DART:
            return ThrowState.SETTLING
        if tick.phase is Phase.TAKEOUT_PENDING:
            return ThrowState.TAKEOUT_WAITING
        if tick.dart_count >= self.max_darts:
            return ThrowState.TAKEOUT_WAITING
        return ThrowState.IDLE

    # ------------------------------------------------------------------
    def apply(
        self,
        tick: Tick,
        lc: Lifecycle,
        current_frames: dict[int, np.ndarray],
        now: float | None = None,
    ) -> LiveStep:
        now = time.monotonic() if now is None else now

        # settle timeline for the package diagnostics: "settling" began when
        # the board change first showed up, and ended at the commit tick
        if tick.phase is Phase.PENDING_DART:
            if self._pending_since is None:
                self._pending_since = now
        elif tick.phase is not Phase.HAND:
            # a hand pauses judgement of a pending change without ending it
            if tick.action is not Action.COMMIT:
                self._pending_since = None

        # A board change is pending: the reference is frozen until it
        # commits or resolves (nothing adopts in the dart branch), so this
        # is the moment to start decoding it -- off this thread, so the
        # commit adds only its own frame's decode. No-op for arrays and for
        # a reference already decoded or in flight.
        if tick.phase is Phase.PENDING_DART:
            prefetch(lc.refs.full_handles())

        baseline = self._baseline_for(lc)
        state = self.map_state(tick)
        trigger = ThrowTriggerState(
            state=state,
            dart_count=tick.dart_count,
            true_baseline_frames=baseline,
        )

        cleared: int | None = None
        if tick.action is Action.COMMIT and tick.commit is not None:
            started = self._pending_since if self._pending_since is not None else now
            # Full pixels for scoring, both sets decoded in parallel when
            # lazy (the reference is normally done already, see prefetch
            # above); plain copies of the dicts otherwise, as before.
            commit_frames, commit_bg = decode_all(tick.commit.frames, tick.commit.bg)
            trigger.last_frame = commit_frames
            trigger.settle_started_monotonic = started
            trigger.settle_duration_s = round(now - started, 3)
            trigger.camera_settled_at_monotonic = {cam: now for cam in tick.commit.dart_cams}
            self._pending_since = None
            reference = commit_bg
        else:
            reference = lc.refs.bg_full()
            if tick.action is Action.CLEARED:
                cleared = tick.cleared

        return LiveStep(trigger=trigger, reference=reference, cleared_darts=cleared, reason=tick.reason)
