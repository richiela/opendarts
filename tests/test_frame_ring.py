"""Tests for opendarts/capture/frame_ring.py and the hub tap that fills it.

THREE CAMERAS EVERYWHERE, deliberately. This project has already shipped a
units bug that passed every test because every test used ONE camera, where
"per frame" and "per set" are the same number. A ring counts SETS and
reports FRAMES and BYTES, so one camera would make three different
quantities indistinguishable.

CONTENT, NOT SHAPE. A sink test once passed on a set of three `None`s
because it checked keys rather than pixels. Every assertion about a
retained frame here compares actual pixel values, and the fixtures give
each (slot, tick) pair a distinguishable value so a frame from the wrong
slot or the wrong moment cannot pass.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from opendarts.capture.frame_ring import (
    FrameRing,
    estimated_bytes_per_second,
    format_bytes,
    seconds_for_bytes,
)


def frame(slot: int, tick: int, *, h: int = 4, w: int = 6) -> np.ndarray:
    """A small frame whose PIXELS identify both the slot and the moment, so
    an assertion can tell "slot 2 at tick 7" from anything else."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = slot + 1
    arr[:, :, 1] = tick % 251
    arr[0, 0, 2] = (tick // 251) % 251
    return arr


def frames_at(tick: int, slots=(0, 1, 2)) -> dict:
    return {s: frame(s, tick) for s in slots}


# -- the memory arithmetic, which is the whole feasibility argument -----


def test_estimate_matches_the_hand_arithmetic_for_three_720p_cameras():
    # 1280*720*3 bytes per frame, three cameras, 30 per second.
    per_frame = 1280 * 720 * 3
    assert estimated_bytes_per_second(3) == pytest.approx(per_frame * 3 * 30)
    # And it really is per SLOT, not per set -- one camera is a third.
    assert estimated_bytes_per_second(1) == pytest.approx(per_frame * 30)


def test_seconds_for_bytes_is_the_exact_inverse_of_the_estimate():
    for budget in (4e9, 6e9, 8e9):
        seconds = seconds_for_bytes(budget, 3)
        assert estimated_bytes_per_second(3) * seconds == pytest.approx(budget)


def test_sizing_table_in_the_docstring_is_reproducible_from_the_formula():
    """4GB~15s / 6GB~22s / 8GB~30s, at the ~32.5 sets/sec the real pump
    achieves rather than a nominal 30. Pinned so the table and the code
    cannot drift apart silently."""
    real_fps = 32.5
    assert seconds_for_bytes(4e9, 3, fps=real_fps) == pytest.approx(15, abs=1)
    assert seconds_for_bytes(6e9, 3, fps=real_fps) == pytest.approx(22, abs=1)
    assert seconds_for_bytes(8e9, 3, fps=real_fps) == pytest.approx(30, abs=1)


def test_format_bytes_uses_decimal_units_so_4e9_reads_as_4gb():
    assert format_bytes(4e9) == "4.00 GB"
    assert format_bytes(270e6) == "270 MB"


# -- retention ----------------------------------------------------------


def test_ring_retains_the_actual_arrays_by_reference_not_a_copy():
    ring = FrameRing(10.0)
    originals = frames_at(1)
    ring.append(originals, wall_s=1000.0, monotonic_s=10.0, generation=1)
    held = ring.snapshot().sets[0].frames
    for slot, original in originals.items():
        # Identity, because "the ring only defers the free" is the entire
        # reason this feature costs no extra memory or CPU. A copy here
        # would double the cost and nothing else in the suite would notice.
        assert held[slot] is original


def test_ring_counts_sets_frames_and_bytes_as_three_different_numbers():
    ring = FrameRing(10.0)
    for tick in range(5):
        ring.append(frames_at(tick), wall_s=1000.0 + tick * 0.03,
                    monotonic_s=10.0 + tick * 0.03, generation=tick)
    stats = ring.stats()
    assert stats["sets"] == 5
    assert stats["frames"] == 15                    # 3 cameras x 5 sets
    assert stats["bytes"] == 15 * frame(0, 0).nbytes


def test_evicts_by_elapsed_time_not_by_a_frame_count():
    """A ring configured for 1 second must hold 1 second whether the pump
    ran at 10/s or 100/s -- that is the difference between configuring a
    WINDOW and configuring a buffer size."""
    slow = FrameRing(1.0)
    fast = FrameRing(1.0)
    for i in range(30):
        slow.append(frames_at(i), wall_s=1000.0 + i * 0.1,
                    monotonic_s=100.0 + i * 0.1, generation=i)
    for i in range(300):
        fast.append(frames_at(i), wall_s=1000.0 + i * 0.01,
                    monotonic_s=100.0 + i * 0.01, generation=i)
    assert slow.stats()["sets"] == 11               # 1.0s at 0.1s spacing
    assert fast.stats()["sets"] == 101              # 1.0s at 0.01s spacing
    assert slow.stats()["span_s"] == pytest.approx(1.0, abs=1e-6)
    assert fast.stats()["span_s"] == pytest.approx(1.0, abs=1e-6)


def test_evicted_frames_are_really_gone_and_the_survivors_are_the_newest():
    ring = FrameRing(0.5)
    for i in range(100):
        ring.append(frames_at(i), wall_s=1000.0 + i * 0.1,
                    monotonic_s=100.0 + i * 0.1, generation=i)
    sets = ring.snapshot().sets
    # CONTENT: the last set really carries tick 99's pixels for every slot.
    assert [s.generation for s in sets] == [94, 95, 96, 97, 98, 99]
    for slot in (0, 1, 2):
        assert np.array_equal(sets[-1].frames[slot], frame(slot, 99))
        assert np.array_equal(sets[0].frames[slot], frame(slot, 94))


def test_a_disabled_ring_retains_nothing_and_says_it_is_disabled():
    ring = FrameRing(0)
    assert ring.enabled is False
    for tick in range(10):
        ring.append(frames_at(tick), wall_s=1000.0 + tick,
                    monotonic_s=10.0 + tick, generation=tick)
    assert ring.stats()["sets"] == 0
    assert ring.stats()["enabled"] is False


def test_a_set_with_no_frames_is_not_retained_as_an_empty_set():
    ring = FrameRing(10.0)
    ring.append({}, wall_s=1.0, monotonic_s=1.0, generation=1)
    ring.append({0: None}, wall_s=2.0, monotonic_s=2.0, generation=2)  # type: ignore[dict-item]
    assert ring.stats()["sets"] == 0


def test_the_ring_measures_its_own_byte_rate_rather_than_assuming_one():
    ring = FrameRing(10.0)
    per_set = 3 * frame(0, 0).nbytes
    for i in range(101):
        ring.append(frames_at(i), wall_s=1000.0 + i * 0.01,
                    monotonic_s=100.0 + i * 0.01, generation=i)
    # 101 sets spanning 1.00s: the rate is bytes held over span held.
    assert ring.stats()["measured_bytes_per_s"] == pytest.approx(
        101 * per_set / 1.0, rel=0.01
    )


# -- both clocks, never subtracted across each other --------------------


def test_window_selection_is_wall_to_wall_with_a_realistic_clock_offset():
    """Wall is epoch-based and monotonic is uptime-based, so on a real
    machine they differ by ~1.7e9. A slice implemented against the wrong
    stamp does not raise -- it silently selects nothing, which is
    indistinguishable from "the ring had nothing"."""
    ring = FrameRing(10.0)
    wall0, mono0 = 1_757_000_000.0, 4321.0     # epoch vs. uptime
    for i in range(100):
        ring.append(frames_at(i), wall_s=wall0 + i * 0.03,
                    monotonic_s=mono0 + i * 0.03, generation=i)
    anchor = wall0 + 50 * 0.03
    got = ring.slice_around(anchor, before_s=0.5, after_s=0.2)
    assert got.aged_out is False
    # -0.5s/+0.2s at 0.03s spacing is 17 before + the anchor + 6 after.
    assert [s.generation for s in got.sets] == list(range(34, 57))
    assert np.array_equal(got.sets[0].frames[2], frame(2, 34))


def test_an_anchor_older_than_the_ring_is_refused_with_the_real_numbers():
    ring = FrameRing(22.0)
    wall0, mono0 = 1_757_000_000.0, 4321.0
    for i in range(740):                      # ~22.2s at 30/s
        ring.append(frames_at(i), wall_s=wall0 + i / 30.0,
                    monotonic_s=mono0 + i / 30.0, generation=i)
    got = ring.slice_around(wall0 - 9.0, before_s=0.5, after_s=0.2)
    assert got.aged_out is True
    assert got.sets == []
    assert got.reason is not None
    # THE NUMBERS, not just the word "aged out": how old the throw is and
    # how far back the ring reaches are the two facts that tell an
    # operator whether to raise the setting or press the button sooner.
    assert "22.0s" in got.reason
    assert "s older than the newest frame" in got.reason


def test_a_window_clipped_by_the_ring_edge_is_partial_not_aged_out():
    ring = FrameRing(1.0)
    wall0, mono0 = 1_757_000_000.0, 4321.0
    for i in range(40):
        ring.append(frames_at(i), wall_s=wall0 + i * 0.03,
                    monotonic_s=mono0 + i * 0.03, generation=i)
    # Anchor near the OLD edge: the -0.5s side runs off the start.
    anchor = wall0 + 15 * 0.03
    got = ring.slice_around(anchor, before_s=0.5, after_s=0.2)
    assert got.aged_out is False               # usable evidence, keep it
    assert got.sets                            # and it is not empty
    assert got.reason is not None and "ring's own edge" in got.reason


def test_slice_reports_span_from_monotonic_even_when_wall_has_stepped():
    """An NTP step mid-session moves wall clock and not monotonic. The
    reported span must follow monotonic, or a 1-second capture would read
    as an hour."""
    ring = FrameRing(10.0)
    for i in range(10):
        wall = 1_757_000_000.0 + i * 0.03 + (3600.0 if i >= 5 else 0.0)
        ring.append(frames_at(i), wall_s=wall, monotonic_s=4321.0 + i * 0.03,
                    generation=i)
    assert ring.snapshot().span_s == pytest.approx(9 * 0.03, abs=1e-6)


def test_eviction_follows_monotonic_so_a_wall_clock_step_does_not_flush_it():
    ring = FrameRing(5.0)
    for i in range(20):
        # Wall jumps an hour backwards halfway through.
        wall = 1_757_000_000.0 + i * 0.1 - (3600.0 if i >= 10 else 0.0)
        ring.append(frames_at(i), wall_s=wall, monotonic_s=4321.0 + i * 0.1,
                    generation=i)
    assert ring.stats()["sets"] == 20          # 1.9s of monotonic, all kept


# -- pause / resume -----------------------------------------------------


def test_pause_drops_arrivals_keeps_what_is_held_and_counts_the_drops():
    ring = FrameRing(10.0)
    for i in range(3):
        ring.append(frames_at(i), wall_s=1000.0 + i, monotonic_s=10.0 + i, generation=i)
    ring.pause()
    for i in range(3, 10):
        ring.append(frames_at(i), wall_s=1000.0 + i, monotonic_s=10.0 + i, generation=i)
    assert ring.paused is True
    assert ring.stats()["sets"] == 3
    assert ring.stats()["dropped_while_paused"] == 7
    ring.resume()
    assert ring.paused is False
    ring.append(frames_at(10), wall_s=1010.0, monotonic_s=20.0, generation=10)
    assert ring.stats()["sets"] == 4


def test_pause_and_resume_are_idempotent_when_called_twice():
    """Called more than once on purpose -- a log-spam bug in this project
    passed because no test called the function twice."""
    ring = FrameRing(10.0)
    ring.append(frames_at(0), wall_s=1.0, monotonic_s=1.0, generation=0)
    ring.pause()
    ring.pause()
    assert ring.paused is True
    ring.resume()
    ring.resume()
    assert ring.paused is False
    assert ring.stats()["sets"] == 1


# -- the byte ceiling, and its two-way health flag ----------------------


def test_the_byte_ceiling_shortens_the_window_and_says_so_both_ways(caplog):
    per_set = 3 * frame(0, 0).nbytes
    ring = FrameRing(100.0, max_bytes=per_set * 4)
    with caplog.at_level("WARNING"):
        for i in range(20):
            ring.append(frames_at(i), wall_s=1000.0 + i * 0.01,
                        monotonic_s=10.0 + i * 0.01, generation=i)
    assert ring.stats()["sets"] == 4
    assert ring.stats()["capped"] is True
    assert any("byte ceiling reached" in r.message for r in caplog.records)

    # AND IT MOVES BACK. A flag that can only reach "capped" is the same
    # class of bug as one that can only reach "healthy" -- see
    # docs/DESIGN.md. Raising the ceiling must clear it on the next append.
    ring.max_bytes = per_set * 100
    ring.append(frames_at(99), wall_s=1000.5, monotonic_s=10.5, generation=99)
    assert ring.stats()["capped"] is False


def test_no_ceiling_means_the_time_window_is_the_only_bound():
    ring = FrameRing(100.0)
    for i in range(50):
        ring.append(frames_at(i), wall_s=1000.0 + i * 0.01,
                    monotonic_s=10.0 + i * 0.01, generation=i)
    assert ring.stats()["sets"] == 50
    assert ring.stats()["capped"] is False


# -- concurrency --------------------------------------------------------


def test_appending_and_snapshotting_concurrently_never_tears_a_set():
    """The pump appends from its own thread while a trigger snapshots from
    an HTTP handler's. A snapshot must never see a set missing a slot."""
    ring = FrameRing(2.0)
    stop = threading.Event()
    errors: list[str] = []

    def producer() -> None:
        i = 0
        while not stop.is_set():
            ring.append(frames_at(i), wall_s=time.time(),
                        monotonic_s=time.monotonic(), generation=i)
            i += 1

    def consumer() -> None:
        for _ in range(200):
            for fs in ring.snapshot().sets:
                if sorted(fs.frames) != [0, 1, 2]:
                    errors.append(f"torn set: {sorted(fs.frames)}")

    threads = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    for t in threads:
        t.start()
    threads[1].join(timeout=10)
    stop.set()
    threads[0].join(timeout=5)
    assert errors == []
    assert ring.stats()["sets"] > 0
