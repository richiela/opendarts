"""opendarts.live.set_assembly: which frame of each camera makes a set.

The rule under test: sets follow the slowest camera; each camera gives its
newest frame captured no later than the reference's (give or take jitter);
a set waits only for a frame it will use; a stalled camera is left out.
These drive the real Assembler with hand-built arrival streams."""
from __future__ import annotations

import random

from opendarts.live import set_assembly as sa

P = 1000.0 / 30


def _stream(phases, *, n=120, lag=None, transfer=8.0, jitter=0.0, periods=None,
            driver=True, seed=3, gaps=None):
    """(arrival_ms, slot, capture_ms, driver_ms) in arrival order."""
    rnd = random.Random(seed)
    out = []
    for slot, phase in enumerate(phases):
        period = (periods or {}).get(slot, P)
        for k in range(n):
            cap = phase + k * period + rnd.gauss(0, jitter)
            if gaps and slot in gaps and gaps[slot][0] <= cap < gaps[slot][1]:
                continue
            arr = cap + transfer + (lag or {}).get(slot, 0.0) + abs(rnd.gauss(0, jitter))
            out.append((arr, slot, cap, cap if driver else None))
    out.sort()
    return out


def _run(stream, slots=3, tick_ms=None):
    asm = sa.Assembler(range(slots))
    sets = []
    events = list(stream)
    if tick_ms:                                   # timer ticks, as the hub's pump waits do
        t, end = events[0][0], events[-1][0]
        while t < end:
            events.append((t, None, None, None))
            t += tick_ms
        events.sort(key=lambda e: e[0])
    for arr, slot, cap, drv in events:
        if slot is not None:
            asm.push(slot, arr, drv, payload=cap)
        choice = asm.next_set(arr)
        if choice is not None:
            asm.commit(choice)
            sets.append((arr, choice))
    return sets


def _true_spread(choice):
    caps = [f.payload for f in choice.picks.values()]
    return max(caps) - min(caps)


def test_free_running_cameras_make_one_set_per_frame_within_a_period():
    sets = _run(_stream((0.0, 11.0, 22.0)))
    assert len(sets) >= 118
    assert max(_true_spread(c) for _, c in sets[2:]) < P
    assert sum(len(c.repeats) for _, c in sets) == 0


def test_a_camera_two_frames_late_is_matched_not_refused():
    """The Oct 7 / Oct 11 shape: one camera's pictures reach the host two
    periods after the others'. Its stamps say so, so the others reach back
    to the same instant; no set is refused and none is mismatched."""
    sets = _run(_stream((0.0, 11.0, 22.0), lag={2: 2 * P}))
    assert len(sets) >= 115
    assert max(_true_spread(c) for _, c in sets[3:]) < P


def test_a_late_joining_camera_does_not_lock_in_a_misalignment():
    stream = _stream((0.0, 11.0, 22.0), lag={2: 2 * P})
    stream = [e for e in stream if not (e[1] == 2 and e[0] < 400)]   # cam2 joins late
    sets = _run(stream)
    after = [c for arr, c in sets if arr > 500]
    assert after and max(_true_spread(c) for c in after) < P


def test_jitter_does_not_make_cameras_flip_between_frames():
    sets = _run(_stream((0.0, 11.0, 22.0), jitter=0.5, n=600))
    assert sum(len(c.repeats) for _, c in sets) <= 3
    assert max(_true_spread(c) for _, c in sets[2:]) < P + 4


def test_a_set_goes_when_the_reference_frame_arrives_if_nothing_better_is_coming():
    """Cameras nearly level: the set must not wait a whole period for the
    next frame of a camera that will not use it."""
    sets = _run(_stream((3.0, 2.0, 1.0)))
    for arr, c in sets[5:]:
        assert arr - max(f.arrival_ms for f in c.picks.values()) < 1e-6


def test_a_stalled_camera_is_left_out_and_rejoins():
    stream = _stream((0.0, 11.0, 22.0), n=150, gaps={1: (1500.0, 2500.0)})
    sets = _run(stream, tick_ms=10.0)
    stalled = [arr for arr, c in sets if 1 in c.stalled]
    assert stalled, "cam1 never left out"
    times = [arr for arr, _ in sets]
    gap = max(b - a for a, b in zip(times, times[1:]))
    assert gap < sa.STALL_MIN_MS + 2 * P + 10, f"sets stopped for {gap:.0f} ms"
    assert any(arr > 2600 and 1 in c.picks for arr, c in sets), "cam1 never rejoined"


def test_a_slower_camera_sets_the_pace():
    sets = _run(_stream((0.0, 11.0, 22.0), n=200, periods={1: 50.0}))
    span_s = (sets[-1][0] - sets[10][0]) / 1000.0
    rate = (len(sets) - 11) / span_s
    assert 18.0 < rate < 21.0
    assert all(c.reference == 1 for _, c in sets[10:])
    assert max(_true_spread(c) for _, c in sets[10:]) < P


def test_without_driver_times_every_camera_uses_arrival_times():
    sets = _run(_stream((0.0, 11.0, 22.0), driver=False))
    assert sets and not any(c.use_driver for _, c in sets)
    assert max(_true_spread(c) for _, c in sets[2:]) < P


def test_one_camera_without_driver_times_puts_the_whole_set_on_arrival_times():
    stream = [(a, s, c, (None if s == 1 else d)) for a, s, c, d in _stream((0.0, 11.0, 22.0))]
    sets = _run(stream)
    assert sets and not any(c.use_driver for _, c in sets)
    assert max(_true_spread(c) for _, c in sets) < P


def test_a_reader_hiccup_leaves_no_lasting_lag():
    """All readers late by 150 ms once, then catching up in a burst: the
    sets after the burst are aligned again at once."""
    stream = []
    for arr, slot, cap, drv in _stream((0.0, 11.0, 22.0), n=200):
        if 1000.0 <= arr < 1150.0:
            arr = 1150.0 + slot * 0.1 + cap / 1e4
        stream.append((arr, slot, cap, drv))
    stream.sort()
    sets = _run(stream)
    assert max(_true_spread(c) for arr, c in sets if arr > 1150.0) < P


def test_history_stays_short():
    asm = sa.Assembler(range(3))
    for arr, slot, cap, drv in _stream((0.0, 11.0, 22.0), n=300):
        asm.push(slot, arr, drv)
        choice = asm.next_set(arr)
        if choice:
            asm.commit(choice)
    assert max(len(b.frames) for b in asm.buffers.values()) <= 3


def test_the_first_set_waits_for_a_camera_that_has_not_started_yet():
    sets = _run(_stream((0.0, 11.0, 22.0)))
    assert set(sets[0][1].picks) == {0, 1, 2}
