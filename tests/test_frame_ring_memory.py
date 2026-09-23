"""MEASURED memory behaviour of the frame ring -- not reasoned about.

The whole feasibility argument for this feature is an arithmetic claim:
three 1280x720 BGR cameras produce ~8.29 MB per frame set, so a ring of N
seconds costs about N x 250-270 MB and nothing more, because the ring
retains references to arrays the pump had already allocated and would
otherwise free. If that claim is wrong -- if the ring copies, or if it
grows without bound, or if the real process footprint is some multiple of
the retained bytes -- then the sizing table an operator chooses from is
fiction and the feature takes the rig down at the moment it is needed.

So these tests allocate REAL 1280x720 frames and measure REAL process
RSS. They are marked slow because they briefly hold hundreds of megabytes,
which is exactly the point: a test that measured 8x12 thumbnails would
prove nothing about a 6GB ring.

RSS IS READ IN-PROCESS, and NOT from `resource.getrusage()`. Two separate
reasons, both learned here rather than assumed:

* `ru_maxrss` is a PEAK that never falls, so it structurally cannot show
  that memory was returned -- it could only ever agree that the ring grew.
  A number that can only move one way is not a measurement; it is the same
  shape of mistake as a health flag with no line that sets it false (see
  docs/DESIGN.md).
* Shelling out to `ps` is not available here. This dev environment's
  sandbox refuses to exec it (`PermissionError: Operation not permitted`),
  and a measurement that silently skips is a measurement nobody makes. So
  the figure comes from the kernel directly: mach `task_info()` on macOS,
  `/proc/self/statm` on Linux.

WHAT THE ALLOCATOR DOES, measured rather than assumed: freeing ~276MB of
720p arrays returned ~97MB of RSS on this machine, not 276MB -- glibc/
libmalloc keep freed pages for reuse. So "the memory came back" is NOT
checked by watching RSS fall. It is checked the way it actually matters:
fill, drop, fill again, and require that the SECOND fill does not grow the
process by a second ring's worth. That is allocator-independent, and it is
the property a long-running rig depends on.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os
import sys

import numpy as np
import pytest

from opendarts.capture.frame_ring import FrameRing, estimated_bytes_per_second

pytestmark = pytest.mark.slow

WIDTH, HEIGHT, SLOTS = 1280, 720, 3
BYTES_PER_FRAME = WIDTH * HEIGHT * 3
BYTES_PER_SET = BYTES_PER_FRAME * SLOTS
FPS = 30.0


class _MachTaskBasicInfo(ctypes.Structure):
    """Apple's `mach_task_basic_info`, field for field. The count passed to
    `task_info()` is in 32-bit words, which is why the struct has to match
    exactly rather than merely be big enough."""

    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time_s", ctypes.c_int32), ("user_time_us", ctypes.c_int32),
        ("system_time_s", ctypes.c_int32), ("system_time_us", ctypes.c_int32),
        ("policy", ctypes.c_int32), ("suspend_count", ctypes.c_int32),
    ]


#: Apple's MACH_TASK_BASIC_INFO flavour.
_MACH_TASK_BASIC_INFO = 20


def current_rss_bytes() -> "int | None":
    """This process's CURRENT resident set size, or None where it cannot
    be read -- never a confident zero. A probe that returns a plausible
    wrong number is worse than one that admits it does not know."""
    if sys.platform == "darwin":
        try:
            lib = ctypes.CDLL(
                ctypes.util.find_library("System") or "libSystem.dylib", use_errno=True
            )
            info = _MachTaskBasicInfo()
            count = ctypes.c_uint(ctypes.sizeof(_MachTaskBasicInfo) // 4)
            rc = lib.task_info(
                ctypes.c_uint(lib.mach_task_self()),
                ctypes.c_int(_MACH_TASK_BASIC_INFO),
                ctypes.byref(info), ctypes.byref(count),
            )
            return int(info.resident_size) if rc == 0 else None
        except Exception:  # noqa: BLE001 -- a probe must never raise
            return None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm") as fh:
                pages = int(fh.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError, IndexError):
            return None
    return None


def real_set(tick: int) -> dict:
    """A freshly allocated 3x720p frame set, as the pump would produce.

    Fresh arrays per set, deliberately: reusing one buffer would make the
    ring appear to cost nothing no matter what it did, which is the exact
    mistake this file exists to avoid."""
    return {
        slot: np.full((HEIGHT, WIDTH, 3), (tick + slot) % 251, dtype=np.uint8)
        for slot in range(SLOTS)
    }


def test_one_frame_set_really_is_the_size_the_sizing_table_assumes():
    frames = real_set(0)
    assert sum(f.nbytes for f in frames.values()) == BYTES_PER_SET
    assert BYTES_PER_SET == pytest.approx(8.29e6, rel=0.01)
    # ~249 MB/s at a nominal 30fps; the ~270 MB/s figure quoted
    # elsewhere is the same arithmetic at the ~32.5 sets/sec the real
    # pump achieves.
    assert estimated_bytes_per_second(SLOTS) == pytest.approx(249e6, rel=0.01)
    assert BYTES_PER_SET * 32.5 == pytest.approx(270e6, rel=0.01)


def test_a_ring_of_n_seconds_holds_n_times_the_per_second_figure():
    """The claim the sizing table is built on, checked against real
    720p allocations rather than against the formula that produced it."""
    for seconds in (0.5, 1.0, 2.0):
        ring = FrameRing(seconds)
        for i in range(int(seconds * FPS * 4)):        # four windows' worth
            ring.append(real_set(i), wall_s=1_757_000_000.0 + i / FPS,
                        monotonic_s=4321.0 + i / FPS, generation=i)
        held = ring.stats()["bytes"]
        expected = estimated_bytes_per_second(SLOTS) * seconds
        # Within one frame set: the window boundary can land either side
        # of a set, and nothing here should be closer than that.
        assert abs(held - expected) <= BYTES_PER_SET, (
            f"{seconds}s held {held / 1e6:.0f} MB, expected ~{expected / 1e6:.0f} MB"
        )
        ring.clear()
        del ring
        gc.collect()


def _fill(ring: FrameRing, n_sets: int, first: int = 0) -> None:
    for i in range(first, first + n_sets):
        ring.append(real_set(i), wall_s=1_757_000_000.0 + i / FPS,
                    monotonic_s=4321.0 + i / FPS, generation=i)


def test_a_second_live_ring_costs_exactly_one_more_ring_of_rss():
    """Does the operating system agree with `stats()["bytes"]`, and is the
    ring really holding the pump's arrays rather than copies of them?

    A DIFFERENTIAL MEASUREMENT, and the first attempt at this test is why.
    The obvious version -- read RSS, fill a ring, read RSS again, expect
    the difference to be the ring's size -- FAILED on a correct
    implementation: a ring reporting 257MB grew the process by 89MB,
    because earlier tests in the same file had already freed that much and
    libmalloc had kept the pages for reuse. The naive measurement was not
    measuring the ring at all; it was measuring how warm the allocator
    happened to be, which depends on test ORDER.

    The differential cancels that completely. Two rings are filled one
    after the other and BOTH ARE HELD ALIVE, so the second cannot reuse
    the first's pages -- whatever the allocator's state, the second ring's
    frames have to come from somewhere new. The growth between the two
    reads is therefore the true cost of one ring, and it discriminates the
    thing that actually matters: holding references costs ~1x the retained
    bytes, and copying would cost ~2x.
    """
    if current_rss_bytes() is None:
        pytest.skip("cannot read RSS on this platform")

    # Warm the allocator first, so the first ring is not the one paying
    # for the process's initial growth.
    warmup = FrameRing(1.0)
    _fill(warmup, int(FPS * 2))
    warmup.clear()
    del warmup
    gc.collect()

    first = FrameRing(1.0)
    _fill(first, int(FPS * 3))
    held = first.stats()["bytes"]
    rss_one = current_rss_bytes()

    second = FrameRing(1.0)
    _fill(second, int(FPS * 3), first=10_000)
    rss_two = current_rss_bytes()
    assert rss_one is not None and rss_two is not None

    grew = rss_two - rss_one
    assert second.stats()["bytes"] == held
    assert grew >= held * 0.85, (
        f"a second live ring holding {held / 1e6:.0f} MB only grew the process "
        f"by {grew / 1e6:.0f} MB -- the accounting and the OS disagree"
    )
    # 1.3x, not 2x. A ring that copied every frame on the way in would
    # land at ~2x and fail here, which is the point of the upper bound.
    assert grew <= held * 1.3, (
        f"a second live ring holding {held / 1e6:.0f} MB grew the process by "
        f"{grew / 1e6:.0f} MB -- something on this path is copying"
    )

    # AND IT COMES BACK -- checked by REUSE, not by watching RSS fall.
    # Measured on this machine: freeing ~276MB of 720p arrays returned
    # ~97MB of RSS, because libmalloc keeps freed pages rather than handing
    # them to the kernel. Asserting RSS falls would therefore fail on a
    # perfectly healthy process. What must be true, and is the property a
    # rig running for days actually depends on, is that a THIRD fill reuses
    # the dropped ring's memory instead of adding to it.
    first.clear()
    del first
    gc.collect()
    third = FrameRing(1.0)
    _fill(third, int(FPS * 3), first=20_000)
    rss_three = current_rss_bytes()
    assert rss_three is not None
    assert rss_three - rss_two <= held * 0.5, (
        f"replacing a dropped ring pushed RSS from {rss_two / 1e6:.0f} MB to "
        f"{rss_three / 1e6:.0f} MB -- the dropped ring's frames were never released"
    )
    second.clear()
    third.clear()


def test_a_long_run_does_not_grow_without_bound():
    """Twenty windows' worth of arrivals through a one-second ring. The
    retained bytes must plateau, and so must the process."""
    ring = FrameRing(1.0)
    samples: list[tuple[int, "int | None"]] = []
    total_sets = int(FPS * 20)
    for i in range(total_sets):
        ring.append(real_set(i), wall_s=1_757_000_000.0 + i / FPS,
                    monotonic_s=4321.0 + i / FPS, generation=i)
        if i and i % int(FPS * 2) == 0:
            samples.append((ring.stats()["bytes"], current_rss_bytes()))

    assert len(samples) >= 5
    held = [s[0] for s in samples]
    # Retained bytes plateau, and every sample after the first window is
    # within one frame set of every other.
    assert max(held) - min(held) <= BYTES_PER_SET, held
    assert ring.stats()["evicted"] > total_sets - int(FPS * 2)

    rss = [s[1] for s in samples if s[1] is not None]
    if len(rss) >= 3:
        # The process must not creep. Twenty windows of arrivals through a
        # one-second ring is ~5GB of allocation churn; if any of it were
        # being retained, the last sample would be far above the second.
        creep = rss[-1] - rss[1]
        assert creep <= BYTES_PER_SET * 4, (
            f"RSS crept {creep / 1e6:.0f} MB across the run -- samples "
            f"{[round(v / 1e6) for v in rss]} MB"
        )
    ring.clear()


def test_pausing_during_a_write_really_keeps_peak_memory_flat():
    """The reason a missed-dart capture pauses the ring: peak stays at one
    buffer rather than one buffer plus whatever arrived during the write."""
    ring = FrameRing(1.0)
    for i in range(int(FPS)):
        ring.append(real_set(i), wall_s=1_757_000_000.0 + i / FPS,
                    monotonic_s=4321.0 + i / FPS, generation=i)
    frozen = ring.stats()["bytes"]
    snapshot = ring.snapshot()                 # references, instant
    ring.pause()
    rss_paused = current_rss_bytes()

    # Two further windows of arrivals during the "write".
    for i in range(int(FPS), int(FPS * 3)):
        ring.append(real_set(i), wall_s=1_757_000_000.0 + i / FPS,
                    monotonic_s=4321.0 + i / FPS, generation=i)
    assert ring.stats()["bytes"] == frozen
    assert ring.stats()["dropped_while_paused"] == int(FPS * 2)

    rss_during = current_rss_bytes()
    if rss_paused is not None and rss_during is not None:
        # The snapshot is still alive here (a writer would be holding it),
        # so the ceiling is one buffer -- not two.
        assert rss_during - rss_paused <= BYTES_PER_SET * 3, (
            f"RSS rose {(rss_during - rss_paused) / 1e6:.0f} MB while paused; "
            "arrivals are being retained despite the pause"
        )
    assert snapshot.nbytes == frozen
    ring.resume()
    ring.clear()
