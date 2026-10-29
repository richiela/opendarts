"""Frame-set assembly by capture time: which frame of each camera makes a set.

THE PROBLEM, 2026-10-11. A frame set must show one moment from every
camera. The old pump read the cameras in LOCKSTEP -- one blocking read per
camera per cycle -- so any camera faster than the slowest, or any camera
that fell one read behind once, kept a driver queue of old frames that it
then served for the rest of the session (2026-10-07: one camera 4 frames
behind). Commit 1608836 patched that with a drain and a refusal to judge a
set whose capture times spread too far. The refusal blinded a rig whose
camera served frames ~2 behind that the drain's own threshold did not
catch (2026-10-11, Pi, ~6 minutes, a real dart missed): two thresholds that
disagreed about what "behind" means.

THE DESIGN. Two separate jobs, done by separate code:

  1. READING. Each camera has its own reader thread that reads as fast as
     the camera delivers and stamps every frame. A driver queue cannot back
     up, because nothing ever waits on another camera before reading this
     one. (Lockstep reading is what let it back up.)
  2. CHOOSING. This module. Given each camera's recent stamped frames, pick
     one per camera so the set shows one instant -- as a pure function, so
     the same code runs on the rig, in the tests and in the trace simulator.

THE RULE (`Assembler`). Sets follow one REFERENCE camera -- the slowest
live one, so sets come at the slowest camera's rate as the lockstep pump's
did. Each new reference frame is a target instant T (its capture time).
Each camera contributes its frame closest to T, from a short history; the
set goes as soon as no frame still on its way could be closer. So the
choice depends only on capture times, never on which frames happened to
have arrived, and a set never waits for a frame it will not use. So:

  * each camera is within half its period of T (plus jitter), so the
    in-set mismatch is the cameras' phase difference -- under a frame
    period -- never a queue;
  * a camera whose frames are stamped late is matched by the others
    reaching BACK in their history to the same instant, and the set waits
    for it instead of being refused;
  * the choice is made afresh from capture times for every set, so a
    misalignment can never carry from one set to the next. (An earlier
    draft that only let each camera move FORWARD from its last pick locked
    a start-up misalignment in for good -- the same failure as a driver
    queue, one layer up. Tested.)

A camera that stops delivering for STALL_PERIODS of its own period is left
out of the wait: sets go on from the others and that slot re-serves its
last frame, which is what the old pump did on a failed read, so the capture
loop's existing per-camera freshness gate handles it exactly as before.

THE CLOCK. A frame carries the driver's capture time when the platform
gives a usable one (V4L2 on Linux: CLOCK_MONOTONIC, the same clock as
time.monotonic()/perf_counter()) and always its arrival time on the reader
thread (time.perf_counter(), sub-microsecond on every platform -- not
time.monotonic(), which is 15.6 ms granular on Windows before Python 3.13).
A set uses driver times only when EVERY camera in it has them; otherwise
it uses arrival times for all, so two clocks with different offsets are
never compared. Arrival times are honest here only because each camera has
its own reader -- under the lockstep pump a read returned when the OTHER
cameras let it, so its arrival time meant nothing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

#: A camera that has delivered nothing for this many of its own frame
#: periods is stalled: the set stops waiting for it.
STALL_PERIODS = 4.0
#: Bounds on that wait, whatever the measured period says.
STALL_MIN_MS = 100.0
STALL_MAX_MS = 1000.0
#: Frames kept per camera. A frame is dropped once a newer one is no newer
#: than the last target, so a healthy camera holds two or three; this cap
#: only matters while another camera stalls or lags (12 frames = 400 ms at
#: 30 fps, more than STALL_MAX_MS at the stall test's 4 periods).
MAX_DEPTH = 12
#: Nominal period until a camera's own frame intervals have been measured.
DEFAULT_PERIOD_MS = 1000.0 / 30
#: Weight of each new frame interval in the running period/jitter estimates.
PERIOD_EMA = 0.02
#: How early a camera's next frame may come, for the placeholder that lets a
#: set go without waiting for a frame it will not use: a period less this
#: many mean interval deviations (about 2.4 sigma), and never less than
#: NEXT_FRAME_MARGIN_MIN_MS. Too early a guess only makes a set wait for a
#: frame it then does not use; it never changes which frames are chosen.
NEXT_FRAME_MARGIN_JITTERS = 3.0
#: The interval jitter a camera is assumed to have before any has been
#: measured: generous, so the first predictions err towards waiting for a
#: real frame (a guess too early only costs a wait; one too late picks the
#: wrong frame). It decays to the measured value over a few seconds.
INITIAL_JITTER_MS = 8.0
NEXT_FRAME_MARGIN_MIN_MS = 2.0


@dataclass(frozen=True)
class Stamped:
    """One frame as its reader thread delivered it."""

    seq: int                      #: per-camera, strictly increasing
    arrival_ms: float             #: reader's clock when read() returned
    driver_ms: "float | None"     #: the driver's capture time, same clock, or None
    payload: Any = None           #: (frame, jpeg) or whatever the caller keeps

    def key(self, use_driver: bool) -> float:
        return self.driver_ms if (use_driver and self.driver_ms is not None) else self.arrival_ms


@dataclass
class CameraBuffer:
    """One camera's recent frames, newest last, plus its measured period."""

    frames: "deque[Stamped]" = field(default_factory=lambda: deque(maxlen=MAX_DEPTH))
    period_ms: float = DEFAULT_PERIOD_MS
    last_arrival_ms: "float | None" = None
    jitter_ms: float = INITIAL_JITTER_MS  #: mean |frame interval - period|
    dropped: int = 0              #: frames pushed out by MAX_DEPTH before any set saw them
    _next_seq: int = 0
    _last_stamp: "float | None" = None

    def earliest_next_ms(self, newest_ms: float) -> float:
        """The earliest the frame after one stamped `newest_ms` can plausibly
        be stamped: a period later, less NEXT_FRAME_MARGIN_JITTERS of the
        camera's measured interval jitter (at least NEXT_FRAME_MARGIN_MIN_MS)."""
        margin = max(NEXT_FRAME_MARGIN_MIN_MS, NEXT_FRAME_MARGIN_JITTERS * self.jitter_ms)
        return newest_ms + max(0.5 * self.period_ms, self.period_ms - margin)

    def push(self, arrival_ms: float, driver_ms: "float | None", payload: Any = None) -> Stamped:
        # Period and jitter from the best clock this frame has: the driver's
        # stamps are steadier than arrival times when there are any.
        stamp = driver_ms if driver_ms is not None else arrival_ms
        if self._last_stamp is not None:
            gap = stamp - self._last_stamp
            # A gap of several periods is a stall, not the camera's rate --
            # folding it in would make the stall test itself slower.
            if 0 < gap < 3 * self.period_ms:
                self.jitter_ms += PERIOD_EMA * (abs(gap - self.period_ms) - self.jitter_ms)
                self.period_ms += PERIOD_EMA * (gap - self.period_ms)
        self._last_stamp = stamp
        self.last_arrival_ms = arrival_ms
        frame = Stamped(self._next_seq, arrival_ms, driver_ms, payload)
        self._next_seq += 1
        if len(self.frames) == self.frames.maxlen:
            self.dropped += 1
        self.frames.append(frame)
        return frame

    def stall_ms(self) -> float:
        return min(STALL_MAX_MS, max(STALL_MIN_MS, STALL_PERIODS * self.period_ms))

    def stalled(self, now_ms: float) -> bool:
        return self.last_arrival_ms is None or now_ms - self.last_arrival_ms > self.stall_ms()


@dataclass(frozen=True)
class SetChoice:
    target_ms: float                       #: the instant T the set shows
    reference: int                         #: the slot T came from
    picks: "dict[int, Stamped]"            #: slot -> chosen frame (live slots only)
    stalled: "tuple[int, ...]"             #: slots left out (re-serve their last frame)
    repeats: "tuple[int, ...]"             #: live slots whose pick is the one they gave last set
    use_driver: bool                       #: which clock the stamps are on
    spread_ms: float                       #: max - min stamp over the picks

    def stamp(self, slot: int) -> float:
        return self.picks[slot].key(self.use_driver)


#: Cameras whose periods are within this fraction of the slowest one count
#: as the same rate (equal-rate cameras measure within about 1% of each
#: other); the reference is one of them.
SAME_RATE = 0.05
#: A frame up to this fraction of a period AFTER T still counts as "at T":
#: jitter must not push a camera that is level with the reference onto its
#: previous frame. Safe below 1/N of a period for N cameras: the reference
#: closes the round, so every other camera sits at least the largest phase
#: gap (>= P/N) away from the far edge (see `_round_closer`).
AT_TARGET_PERIODS = 0.15
#: The reference stays put until another camera's phase drifts to within
#: this fraction of a period of the far edge, where jitter could flip it
#: between two frames (1 - AT_TARGET_PERIODS, less a margin). Which camera
#: is the reference changes only the set's spread, never its latency -- a
#: set always goes when its last frame arrives -- so it is not worth
#: moving for a tighter round, and moving costs one uneven step.
REFERENCE_UNSAFE_PERIODS = 0.75
#: Weight of each set in the smoothed span that decision uses -- one
#: jittered frame must not move the reference.
SPAN_EMA = 0.1


class Assembler:
    """Per-camera histories and the set-making rule. Pure: no threads and
    no clock of its own -- the caller passes `now_ms` and serialises calls.
    """

    def __init__(self, slots: "Iterable[int]" = ()) -> None:
        self.buffers: "dict[int, CameraBuffer]" = {s: CameraBuffer() for s in slots}
        self.last_target: "float | None" = None
        self.last_seq: "dict[int, int]" = {}
        self.reference: "int | None" = None
        self._span: "float | None" = None
        self._first_ms: "float | None" = None
        self._last_use_driver: "bool | None" = None

    def push(self, slot: int, arrival_ms: float, driver_ms: "float | None",
             payload: Any = None) -> Stamped:
        buf = self.buffers.get(slot)
        if buf is None:
            buf = self.buffers[slot] = CameraBuffer()
        return buf.push(arrival_ms, driver_ms, payload)

    def remove(self, slot: int) -> None:
        self.buffers.pop(slot, None)
        self.last_seq.pop(slot, None)
        if self.reference == slot:
            self.reference = None

    def _round_span(self, ref: int, group: "list[int]", key: Any, period: float) -> float:
        """How far before `ref`'s newest frame, at most, the other cameras of
        `group` last captured -- in phase, modulo one period."""
        x = key(self.buffers[ref].frames[-1])
        return max(((x - key(self.buffers[s].frames[-1])) % period) for s in group)

    def _pick_reference(self, live: "list[int]", key: Any) -> int:
        """The slowest camera -- sets come at its rate. Among cameras at that
        same rate, a new reference is the one that CLOSES the round: the
        camera whose frame comes last in each cycle, just before the largest
        gap in the cameras' phases, so the set spans P minus that gap (at
        most (N-1)/N of a period). It is kept until phases drift far enough
        to make it unsafe (REFERENCE_UNSAFE_PERIODS)."""
        slowest = max(self.buffers[s].period_ms for s in live)
        group = [s for s in live if self.buffers[s].period_ms >= slowest * (1 - SAME_RATE)]
        if len(group) == 1:
            return group[0]
        period = sum(self.buffers[s].period_ms for s in group) / len(group)
        spans = {s: self._round_span(s, group, key, period) for s in group}
        ref = self.reference
        if ref in group:
            self._span = (spans[ref] if self._span is None
                          else self._span + SPAN_EMA * (spans[ref] - self._span))
            if self._span < REFERENCE_UNSAFE_PERIODS * period:
                return ref
        self._span = None
        return min(group, key=lambda s: (spans[s], s))

    def next_set(self, now_ms: float) -> "SetChoice | None":
        """The next frame set, or None if none is ready. Changes nothing;
        call `commit()` with the result once it has been published."""
        live = [s for s, b in self.buffers.items() if b.frames and not b.stalled(now_ms)]
        if not live:
            return None
        # A camera that has not delivered its FIRST frame yet is starting,
        # not stalled: the first sets wait for it (up to the stall time),
        # so a session does not open on sets with a camera missing.
        if self._first_ms is None:
            self._first_ms = now_ms
        if (len(live) < len(self.buffers)
                and any(not b.frames for b in self.buffers.values())
                and now_ms - self._first_ms <= STALL_MIN_MS):
            return None
        stalled = tuple(sorted(s for s in self.buffers if s not in live))
        use_driver = all(f.driver_ms is not None for s in live for f in self.buffers[s].frames)
        if self._last_use_driver is not None and use_driver != self._last_use_driver:
            # The clock changed under us (a camera without capture times
            # joined or left): targets on the old clock mean nothing now.
            self.last_target = None
        key = lambda f: f.key(use_driver)  # noqa: E731
        ref = self._pick_reference(live, key)
        # The oldest reference frame at least half a period past the last
        # target: reference frames are taken in order, none skipped while
        # others catch up, and a change of reference can never make two
        # sets a few milliseconds apart.
        floor = (None if self.last_target is None
                 else self.last_target + 0.5 * self.buffers[ref].period_ms)
        target = next((key(f) for f in self.buffers[ref].frames
                       if floor is None or key(f) > floor), None)
        if target is None:
            return None
        # Each other camera gives its newest frame captured no later than T
        # (give or take AT_TARGET_PERIODS of jitter). That frame is known
        # once the camera has a frame after the edge, or once its next frame
        # could not be stamped before the edge (CameraBuffer.earliest_next_ms)
        # -- so the choice depends only on capture times, never on which
        # frames happened to have arrived, and the set waits only for a
        # frame it will use.
        picks: "dict[int, Stamped]" = {}
        for s in live:
            buf = self.buffers[s]
            if s == ref:
                picks[s] = next(f for f in buf.frames if key(f) == target)
                continue
            edge = target + AT_TARGET_PERIODS * self.buffers[ref].period_ms
            newest = key(buf.frames[-1])
            if newest <= edge and buf.earliest_next_ms(newest) <= edge:
                return None                  # a frame it will use is still on its way
            at_or_before = [f for f in buf.frames if key(f) <= edge]
            # Nothing that old left: the camera is ahead of the whole
            # history (it only just joined) -- its oldest frame is closest.
            picks[s] = at_or_before[-1] if at_or_before else buf.frames[0]
        stamps = [key(f) for f in picks.values()]
        repeats = tuple(s for s, f in picks.items() if self.last_seq.get(s) == f.seq)
        return SetChoice(target, ref, picks, stalled, repeats, use_driver,
                         max(stamps) - min(stamps))

    def commit(self, choice: SetChoice) -> None:
        """Record that `choice` was published, and drop history no future
        target can use: a frame whose successor is already no newer than
        this target can never again be the closest."""
        self.last_target = choice.target_ms
        self.reference = choice.reference
        self._last_use_driver = choice.use_driver
        for s, f in choice.picks.items():
            self.last_seq[s] = f.seq
        for buf in self.buffers.values():
            fr = buf.frames
            while len(fr) >= 2 and fr[1].key(choice.use_driver) <= choice.target_ms:
                fr.popleft()


def spread_of(stamps: Iterable[float]) -> float:
    xs = list(stamps)
    return (max(xs) - min(xs)) if len(xs) > 1 else 0.0
