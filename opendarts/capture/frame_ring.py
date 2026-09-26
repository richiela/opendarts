"""opendarts/capture/frame_ring.py -- an in-memory ring of the last N
seconds of RAW decoded frames, tapped off the camera hub's pump.

WHY THIS EXISTS. Two dart failures are invisible to the corpus today:

* A MISSED dart -- nothing was detected, so no throw package was ever
  written. There is no evidence at all. `opendarts/capture/replay.py` can
  only replay a package, and package replay tests SCORING, because
  detection already happened before a package exists. A missed dart has
  no package, so frame replay is the only way it can ever be debugged.
* A MISSCORED dart -- a package exists, but it holds exactly the two
  frames the lifecycle chose (background + dart). You cannot ask "would a
  different settling threshold have called this right", because the
  neighbouring frames were never kept.

This ring keeps the neighbours. It is not a recording feature; it is the
evidence tier under the package, and the package remains the unit of
record (see docs/DESIGN.md's "Replay is the source of truth").

WHY RAW, AND WHY THAT COSTS NOTHING. Measured on this project's own
corpus at 1280x720: PNG encoding is 49.5ms/frame -- 4.5 cores saturated
at 100 frames/sec, which is simply not available; JPEG q85 is 1.3ms but
lossy, and a lossy record of a misscore is a record of a different
image than the one that was scored. Raw costs nothing to RETAIN: the
pump already allocates a fresh array per frame and already copies a
local camera's frame off OpenCV's reusable buffer (see
`CameraHub._pump_once()`'s own comment), so the ring holds a REFERENCE
to an array that already exists and would otherwise be freed one cycle
later. There is no extra allocation, no extra copy, and no extra CPU --
the ring only defers a free. numpy arrays hold no references to other
Python objects, so numpy opts them out of cyclic-GC tracking entirely
(measured, not assumed) -- so deferring
thousands of them adds no collector pressure either.

WHAT IT COSTS IN MEMORY, which is the whole feasibility argument.
One 1280x720 BGR frame is 1280*720*3 = 2,764,800 bytes. Three cameras
is 8.29 MB per frame SET, and at the ~32.5 sets/sec this hub really
achieves that is ~270 MB/s. A 16GB rig with 14.4GB free (and other software
running beside it) can therefore afford roughly:

    4GB ~ 15s      6GB ~ 22s      8GB ~ 30s

Nobody should have to do that arithmetic to choose a setting, so the
ring is configured in SECONDS and `estimated_bytes_per_second()` below
exists to put the megabytes next to the seconds in the UI. Note the
nominal-vs-real gap: at a nominal 30fps the same three cameras produce
~249 MB/s, not 270 -- the 270 figure comes from the hub's real measured
cadence, which is slightly faster than nominal. `stats()` reports the
ring's OWN measured rate, so the number on screen is the rig's, not an
estimate's.

BOTH CLOCKS ON EVERY SET, AND NEVER SUBTRACTED ACROSS EACH OTHER.
`wall_s` (`time.time()`) is what matches a throw package's
`captured_at_utc`; `monotonic_s` (`time.monotonic()`) is what orders
frames and measures durations. On 2026-09-15 a cross-clock subtraction
produced nonsense timings on the Windows rig and the bug was invisible
until someone read the numbers -- wall and monotonic differ by the
machine's uptime-to-epoch offset, so mixing them yields a plausible
looking float that is wrong by decades. So: WINDOW SELECTION IS
WALL-TO-WALL (the anchor is a package timestamp) and every DURATION,
span and eviction decision is MONOTONIC-TO-MONOTONIC. The two stamps are
sampled in the same instant by the pump and handed in together.

THE RING LIVES AT THE HUB, SO EVERY RIG GETS ONE. A camera rig buffers
what its own cameras produced; a stream-fed rig buffers what it decoded
off the wire. Those are different images and both are correct: a
consumer's own ring is the authoritative record of what THAT consumer
scored, and the source rig's frames are not what it saw.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

log = logging.getLogger("opendarts.capture.frame_ring")

#: The frame rate the memory estimate assumes when nothing better is
#: known. The hub's own pump really cycles at ~31-32.5/s on the rigs
#: measured so far (see `CameraHub._pump_once()`), so a nominal 30
#: UNDER-states the real cost by ~8%. The estimate is a shopping figure,
#: not a measurement: `FrameRing.stats()["measured_bytes_per_s"]` is the
#: real one, and the UI should show that once a ring has been running.
NOMINAL_FPS = 30.0

#: How long between repeated "the ring is capped" lines. A cap that
#: fires does so on every pump cycle, so logging each one would push ~30
#: lines a second and bury the thing being diagnosed. docs/DESIGN.md:
#: throttled, never suppressed.
CAP_LOG_INTERVAL_S = 15.0


def estimated_bytes_per_second(
    n_slots: int,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = NOMINAL_FPS,
    channels: int = 3,
) -> float:
    """What a ring of this shape will accumulate per second of wall time.

    Exists so the dashboard can put "~1.2 GB" next to a seconds control
    as the operator drags it, rather than making them work out that three
    cameras at 720p30 is a quarter of a gigabyte every second. See this
    module's docstring for why the answer is ~270 MB/s on the real rig
    and ~249 MB/s at a nominal 30fps -- this function returns the latter
    unless the caller passes the rate it actually measured.
    """
    return float(n_slots) * float(width) * float(height) * float(channels) * float(fps)


def format_bytes(n: float) -> str:
    """Human-sized bytes, decimal (GB = 1e9), because that is the unit the
    sizing table in this module's docstring and the operator's own
    "16GB machine" are both quoted in. Binary units here would make a
    4GB setting read as 3.7 and invite the wrong comparison."""
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} kB"
    return f"{int(n)} B"


@dataclass(frozen=True)
class FrameSet:
    """One pump cycle's frames, with both clocks as sampled in that cycle.

    `pixels` maps SLOT -> the array the pump cached for that slot, by
    REFERENCE. Nothing here copies: the pump has already handed over an
    array it will never write into again (a local camera's frame is
    `.copy()`d off OpenCV's reusable buffer by the pump itself; a stream
    or replay source allocates a fresh array per frame and never touches
    a published one). Copying again would double the cost of the feature
    to defend against a hazard that does not exist.

    `jpegs` maps SLOT -> the camera's own JPEG, for a slot in JPEG
    passthrough (see local_capture's section of that name). Such a slot
    is held ONLY as its JPEG, not as pixels too: ~82 KB instead of
    ~2.7 MB on 1280x720 cameras, so the same memory holds ~30x the
    window. Nothing is lost -- the pump's pixels were decoded from these
    very bytes, and decoding them again gives the same array. The cost
    moves to whoever reads `frames`, which decodes; that is a dump or a
    replay, never the pump.

    Slots that produced no frame this cycle are ABSENT, never present as
    None -- the same contract `CameraHub.grab_all()` and the frame sink
    already have. A sink test once passed on a set of three Nones because
    it checked keys rather than pixels; absent-not-None is what makes
    "this set carries pixels" checkable at all.
    """

    generation: int
    wall_s: float
    monotonic_s: float
    pixels: Mapping[int, np.ndarray]
    nbytes: int
    jpegs: Mapping[int, bytes] = field(default_factory=dict)

    @property
    def slots(self) -> list[int]:
        return sorted(set(self.pixels) | set(self.jpegs))

    @property
    def frames(self) -> "dict[int, np.ndarray]":
        """Every slot's pixels, decoding the JPEG slots. Decodes on EACH
        access; a caller walking many sets should use frames_with()."""
        return self.frames_with(None)

    def frames_with(self, cache: "dict[int, np.ndarray] | None") -> "dict[int, np.ndarray]":
        """`frames`, decoding each JPEG at most once per `cache`.

        Keyed by the bytes object's id, which is stable because the ring
        holds the object. A camera that stalls is re-served with the SAME
        bytes object, so it decodes to the SAME array -- which is what
        lets a dump recognise the repeat instead of writing it twice.
        """
        out = dict(self.pixels)
        for slot, data in self.jpegs.items():
            key = id(data)
            arr = cache.get(key) if cache is not None else None
            if arr is None:
                arr = _decode_jpeg(data)
                if arr is None:
                    continue
                if cache is not None:
                    cache[key] = arr
            out[slot] = arr
        return out


def _decode_jpeg(data: bytes) -> "np.ndarray | None":
    import cv2

    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        # Cannot happen for bytes the pump accepted (it decoded them
        # first), so say so loudly rather than drop a frame quietly.
        log.error("frame ring: a retained JPEG (%d bytes) would not decode", len(data))
    return arr


@dataclass
class RingSlice:
    """The frames a trigger asked for, and -- when it could not have them
    all -- the honest reason why.

    `aged_out` is the field that exists because of docs/DESIGN.md's "a
    refusal, cap or fallback must say so": a misscore capture asked for a
    throw the ring no longer holds must never write an empty file, and
    must never no-op while looking successful. `reason` carries the
    NUMBERS ("the oldest frame is 22.0s old, the throw is 31.2s old"),
    because "aged out" alone does not tell an operator whether to raise
    the seconds setting or to press the button sooner.
    """

    sets: list[FrameSet]
    #: The wall-clock instant the caller anchored on (a package timestamp
    #: for a misscore; None for a whole-buffer capture, which has no
    #: anchor by definition).
    anchor_wall_s: "float | None" = None
    window_before_s: "float | None" = None
    window_after_s: "float | None" = None
    aged_out: bool = False
    reason: "str | None" = None
    #: Ring extent AT THE MOMENT OF THE SLICE, in wall clock, so a caller
    #: reporting a refusal can quote what was actually available rather
    #: than re-reading a ring that has moved on since.
    ring_oldest_wall_s: "float | None" = None
    ring_newest_wall_s: "float | None" = None

    @property
    def n_frames(self) -> int:
        return sum(len(s.slots) for s in self.sets)

    @property
    def nbytes(self) -> int:
        return sum(s.nbytes for s in self.sets)

    @property
    def span_s(self) -> float:
        """Duration covered, MONOTONIC (never wall) -- see the module
        docstring. Zero for an empty or single-set slice."""
        if len(self.sets) < 2:
            return 0.0
        return self.sets[-1].monotonic_s - self.sets[0].monotonic_s


class FrameRing:
    """The last `seconds` of frame sets, evicted by TIME rather than count.

    By time, because the thing being configured is "how far back can I
    reach", and a count only means that if the frame rate is what you
    assumed. A rig whose cameras degrade to 10fps would silently triple
    its window under a count-based ring, and a rig running two hubs'
    worth of slots would silently third it.

    Thread-safe: the pump appends from its own thread, a trigger snapshots
    from an HTTP handler's thread, and a writer thread reads the snapshot.
    The lock is held only around list surgery and integer arithmetic --
    never around anything that touches a frame's pixels -- so appending
    cannot stall the pump.

    NAMING HAZARD, worth stating once: "ring" in this codebase otherwise
    always means a dartboard ring (`sector_ring_for_point`, `ad_ring`,
    `BOARD_RINGS`). Everything here is spelled `frame_ring`/`FrameRing`
    for that reason, and anything reaching the API or the UI should keep
    the `frame_` prefix rather than shortening it.
    """

    def __init__(
        self,
        seconds: float,
        *,
        max_bytes: "int | None" = None,
        name: str = "frame-ring",
    ) -> None:
        #: Zero or negative means DISABLED, explicitly and checkably --
        #: not "a very small ring". A rig that does not want the memory
        #: must be able to say so, and `enabled` is what the dashboard
        #: reports rather than inferring it from a size.
        self.seconds = float(seconds)
        #: An optional hard ceiling on retained bytes, for a rig that
        #: would rather lose window than lose the process. None means the
        #: time window is the only bound. When this fires it SHORTENS the
        #: window, which is exactly the kind of silent degradation
        #: docs/DESIGN.md forbids going unreported -- so it logs
        #: (throttled) and sets `capped`, which the dashboard shows.
        self.max_bytes = int(max_bytes) if max_bytes else None
        self.name = name

        self._lock = threading.Lock()
        self._sets: list[FrameSet] = []
        self._nbytes = 0
        self._paused = False
        self._dropped_while_paused = 0
        self._appended = 0
        self._evicted = 0
        #: A HEALTH FLAG THAT MOVES BOTH WAYS (docs/DESIGN.md). Set true
        #: by the byte ceiling actually evicting a set that the time
        #: window would have kept; set false again by any append that did
        #: not need the ceiling. Assigned from the condition on every
        #: append rather than only set on the bad branch, so a ring that
        #: recovers stops claiming to be capped.
        self._capped = False
        self._capped_since_mono: "float | None" = None
        self._last_cap_log_mono = 0.0
        #: Real bytes/sec, computed from what was actually retained. The
        #: estimate above is for choosing a setting; this is for checking
        #: the estimate was right on THIS rig, which is the only version
        #: of the number worth trusting.
        self._measured_bytes_per_s: "float | None" = None

    # -- the pump's side ------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.seconds > 0

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def append(
        self,
        frames: Mapping[int, np.ndarray],
        *,
        wall_s: float,
        monotonic_s: float,
        generation: int,
        jpegs: "Mapping[int, bytes] | None" = None,
    ) -> None:
        """Retain one pump cycle. Called from the pump thread, once per
        cycle, with the SAME two clock samples the pump already took for
        its own status fields -- not with two fresh `time.*()` calls,
        which would describe two different instants and make the ring's
        own ordering disagree with `CameraStatus.last_read_at`.

        `frames` is copied at the DICT level only (so a later mutation of
        the pump's own cache dict cannot reach into a retained set) and
        never at the array level -- see `FrameSet`'s docstring.

        Never raises. This runs on the capture path, and the capture
        loop's correctness must not depend on a diagnostic -- the same
        posture the frame sink already has.
        """
        if not self.enabled:
            return
        kept = {i: f for i, f in frames.items() if f is not None}
        if not kept:
            # Nothing to retain. Deliberately not appended as an empty
            # set: an empty set carries no evidence, would still occupy a
            # slot in the window, and would let "the ring has 600 sets"
            # read as healthy on a rig delivering nothing at all.
            return
        # A slot with its camera JPEG is kept as the JPEG alone -- see
        # FrameSet. Its pixel array is then freed on the next cycle, as it
        # would have been without a ring at all.
        kept_jpegs = {i: j for i, j in (jpegs or {}).items() if i in kept and j}
        pixels = {i: f for i, f in kept.items() if i not in kept_jpegs}
        nbytes = (sum(int(f.nbytes) for f in pixels.values())
                  + sum(len(j) for j in kept_jpegs.values()))
        entry = FrameSet(
            generation=int(generation),
            wall_s=float(wall_s),
            monotonic_s=float(monotonic_s),
            pixels=pixels,
            nbytes=nbytes,
            jpegs=kept_jpegs,
        )
        with self._lock:
            if self._paused:
                self._dropped_while_paused += 1
                return
            self._sets.append(entry)
            self._nbytes += nbytes
            self._appended += 1
            self._evict_locked()

    def _evict_locked(self) -> None:
        """Drop everything older than the window, then everything over the
        byte ceiling -- in that order, so the ceiling only ever trims a
        window the time rule already considers current.

        MONOTONIC, never wall: a clock step mid-session (NTP, a VM
        resuming) would otherwise either flush the whole ring or freeze
        eviction entirely, and both look like a ring bug rather than a
        clock one.
        """
        newest = self._sets[-1].monotonic_s
        while len(self._sets) > 1 and (newest - self._sets[0].monotonic_s) > self.seconds:
            self._drop_oldest_locked()

        capped_now = False
        if self.max_bytes is not None:
            while len(self._sets) > 1 and self._nbytes > self.max_bytes:
                self._drop_oldest_locked()
                capped_now = True

        # Both directions, from the condition, every time -- not cleared
        # on the good branch and forgotten on the bad one.
        if capped_now and not self._capped:
            self._capped_since_mono = newest
        elif not capped_now:
            self._capped_since_mono = None
        self._capped = capped_now
        if capped_now:
            now = time.monotonic()
            if now - self._last_cap_log_mono >= CAP_LOG_INTERVAL_S:
                self._last_cap_log_mono = now
                log.warning(
                    "%s: byte ceiling reached -- holding %s of a requested %s "
                    "and %.1fs of a requested %.1fs window. The ring is SHORTER "
                    "than configured; lower the seconds setting or raise the "
                    "ceiling so the number on screen is the number you get.",
                    self.name, format_bytes(self._nbytes),
                    format_bytes(float(self.max_bytes or 0)),
                    self._span_locked(), self.seconds,
                )

        if len(self._sets) >= 2:
            span = self._span_locked()
            if span > 0:
                self._measured_bytes_per_s = self._nbytes / span

    def _drop_oldest_locked(self) -> None:
        dropped = self._sets.pop(0)
        self._nbytes -= dropped.nbytes
        self._evicted += 1

    def _span_locked(self) -> float:
        if len(self._sets) < 2:
            return 0.0
        return self._sets[-1].monotonic_s - self._sets[0].monotonic_s

    # -- the trigger's side ---------------------------------------------

    def pause(self) -> None:
        """Stop admitting frames, without dropping what is held.

        Used by a MISSED-dart capture, which writes the whole buffer:
        pausing keeps peak memory flat at one buffer rather than one
        buffer plus whatever arrives during the write. At a measured
        3.3 GB/s on an NVMe rig a 6GB write is ~2 seconds and only
        ~540MB would have arrived anyway, so this is no longer
        load-bearing there -- but it is free, and a Mac on SATA or a VM on
        shared storage is where a design that needs it will be found.

        MUST BE PAIRED WITH `resume()` IN A `finally`. A writer thread
        that dies with the ring paused leaves the tap off for the rest of
        the session with every other surface looking healthy -- the exact
        "no errors on the rig" failure docs/DESIGN.md is written about.
        """
        with self._lock:
            if not self._paused:
                self._paused = True
                self._dropped_while_paused = 0
                log.info("%s: paused -- arrivals are being dropped until resume()", self.name)

    def resume(self) -> None:
        with self._lock:
            if not self._paused:
                return
            self._paused = False
            dropped, self._dropped_while_paused = self._dropped_while_paused, 0
        log.info("%s: resumed -- %d frame set(s) were dropped while paused", self.name, dropped)

    def snapshot(self) -> RingSlice:
        """EVERY set currently held, as references. Instant: this copies a
        list of N dataclasses, never a pixel, so a trigger returns
        immediately and the UI can say "saving" instead of hanging."""
        with self._lock:
            sets = list(self._sets)
            oldest = sets[0].wall_s if sets else None
            newest = sets[-1].wall_s if sets else None
        return RingSlice(
            sets=sets,
            ring_oldest_wall_s=oldest,
            ring_newest_wall_s=newest,
            reason=None if sets else "the ring is empty -- no frames have been retained yet",
            aged_out=False,
        )

    def sets_since(self, generation: int) -> list[FrameSet]:
        """Every held set whose generation is `generation` or later, oldest
        first, as references -- how a package clip takes its frames out of
        the ring BY NUMBER (opendarts.capture.clip.write_window_clips).
        Walks back from the newest set, so the cost is the few sets asked
        for, not the whole window."""
        with self._lock:
            i = len(self._sets)
            while i > 0 and self._sets[i - 1].generation >= generation:
                i -= 1
            return self._sets[i:]

    def slice_around(
        self, anchor_wall_s: float, *, before_s: float, after_s: float
    ) -> RingSlice:
        """The sets whose WALL stamp falls in [anchor - before, anchor +
        after].

        ANCHORED TO THE RECORDED THROW TIME, not to "the last N frames".
        By the time anyone presses a misscore button they have walked to a
        screen and the board is empty; the last N frames are of an empty
        board.

        WALL-TO-WALL, deliberately. The anchor comes from a throw
        package's own `captured_at_utc`, which is wall clock, and the only
        stamp in this ring that can be compared to it is `wall_s`. A
        monotonic comparison here would be the 2026-09-15 cross-clock bug
        again: it would not raise, it would select nothing, and an empty
        capture would be indistinguishable from "the ring had nothing".

        An anchor older than the whole ring comes back `aged_out=True`
        WITH THE NUMBERS, and with whatever sets did fall in range (often
        none). The caller must refuse to write rather than writing an
        empty file -- see `RingSlice`.
        """
        lo = anchor_wall_s - abs(before_s)
        hi = anchor_wall_s + abs(after_s)
        with self._lock:
            sets = [s for s in self._sets if lo <= s.wall_s <= hi]
            oldest = self._sets[0].wall_s if self._sets else None
            newest = self._sets[-1].wall_s if self._sets else None
            span = self._span_locked()

        aged_out = False
        reason: "str | None" = None
        if oldest is None:
            reason = "the ring is empty -- no frames have been retained yet"
        elif hi < oldest:
            aged_out = True
            reason = (
                f"that throw is {newest - anchor_wall_s:.1f}s older than the newest "
                f"frame in the ring, and the ring only reaches back {span:.1f}s "
                f"(oldest frame is {newest - oldest:.1f}s old). Nothing from that "
                f"moment is still held. Raise the frame-ring seconds setting, or "
                f"capture sooner after the throw."
            )
        elif lo < oldest:
            # PARTIAL is not the same as aged out, and conflating them
            # would throw away a usable capture. Say what was clipped.
            reason = (
                f"the window starts {oldest - lo:.2f}s before the oldest frame the "
                f"ring still holds -- the capture begins at the ring's own edge, "
                f"so the earliest part of the requested window is missing."
            )
        elif not sets:
            reason = (
                f"no frame set falls in the requested window "
                f"[-{abs(before_s):.2f}s, +{abs(after_s):.2f}s] around that throw, "
                f"even though the ring covers it -- the pump may have been stalled "
                f"at that moment."
            )
        if aged_out:
            log.warning("%s: misscore capture REFUSED -- %s", self.name, reason)
        elif reason and sets:
            log.warning("%s: misscore capture is partial -- %s", self.name, reason)

        return RingSlice(
            sets=sets,
            anchor_wall_s=anchor_wall_s,
            window_before_s=abs(before_s),
            window_after_s=abs(after_s),
            aged_out=aged_out,
            reason=reason,
            ring_oldest_wall_s=oldest,
            ring_newest_wall_s=newest,
        )

    def clear(self) -> None:
        """Drop every retained set. The MEASURED rate is kept: it describes
        this rig's cameras, not the contents, and the Config tab prices the
        setting from it -- forgetting it made an idle-cleared ring quote
        uncompressed pixels again."""
        with self._lock:
            self._sets = []
            self._nbytes = 0
            self._capped = False
            self._capped_since_mono = None

    def stats(self) -> dict[str, Any]:
        """Everything the dashboard and `/api/frame-ring` need, cheap
        enough to poll: all of it is maintained incrementally on append,
        so nothing here walks the retained frames."""
        with self._lock:
            n = len(self._sets)
            nbytes = self._nbytes
            span = self._span_locked()
            oldest_wall = self._sets[0].wall_s if self._sets else None
            newest_wall = self._sets[-1].wall_s if self._sets else None
            return {
                "enabled": self.enabled,
                "seconds": self.seconds,
                "paused": self._paused,
                "sets": n,
                "frames": sum(len(s.slots) for s in self._sets),
                "bytes": nbytes,
                "mb": round(nbytes / 1e6, 1),
                "span_s": round(span, 2),
                # How full the configured window actually is. A ring
                # reporting 22s configured and 3.1s held is a rig that
                # only just started -- or one whose pump is stalled, and
                # that difference is what an operator is looking at this
                # number to tell.
                "fill_fraction": round(span / self.seconds, 3) if self.seconds > 0 else 0.0,
                "oldest_wall_s": oldest_wall,
                "newest_wall_s": newest_wall,
                "measured_bytes_per_s": (
                    round(self._measured_bytes_per_s, 1)
                    if self._measured_bytes_per_s is not None else None
                ),
                "capped": self._capped,
                "max_bytes": self.max_bytes,
                "appended": self._appended,
                "evicted": self._evicted,
                "dropped_while_paused": self._dropped_while_paused,
            }


def seconds_for_bytes(
    budget_bytes: float,
    n_slots: int,
    *,
    width: int = 1280,
    height: int = 720,
    fps: float = NOMINAL_FPS,
) -> float:
    """The inverse of `estimated_bytes_per_second()`: how many seconds a
    memory budget buys. Exists for the sizing table in the UI ("8GB ~
    30s") so the two directions cannot drift apart into two formulas."""
    per_s = estimated_bytes_per_second(n_slots, width=width, height=height, fps=fps)
    return 0.0 if per_s <= 0 else budget_bytes / per_s
