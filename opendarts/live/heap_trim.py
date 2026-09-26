"""opendarts/live/heap_trim.py -- hand freed heap back to the OS after a
calibration.

WHY THIS EXISTS, measured on the Raspberry Pi 5 rig 2026-09-25. One
calibration took the running process from ~540 MB to ~1,170 MB and it
never came back down -- not after the calibration returned, not after
Stop closed the cameras. A normal evening of Start plus a recalibration or
two reached 3.4 GB.

Nothing is being KEPT. The same calibration driven offline on saved frames
(a fake camera hub feeding ``bootstrap_calibrations()``) leaves ~5 MB of
Python allocations alive afterwards by tracemalloc and no ndarray over
500 KB reachable from the gc -- but its tracemalloc PEAK is ~1.4 GB: the
raw frame pool (every captured frame, ~75 per camera at 1280x720) plus
per-frame detection temporaries, all released when the call returns. On
the Pi the same harness sat at 1,494 / 1,532 / 876 MB after each of three
back-to-back calibrations (from 77 MB at start) with nothing referenced.
With the trims below it sat at 104 / 100 / 112 MB (from 79 MB).

That is glibc, not Python. Frame-sized numpy buffers are big enough to be
``mmap``-ed on first use, but glibc raises its mmap threshold every time
it frees one (up to 32 MB), so from then on they come out of the ordinary
heap -- and freed heap in the middle of an arena is kept for reuse, not
returned. ``MALLOC_ARENA_MAX=2`` was tried on the rig and did not help,
which fits: it caps how many arenas there are, not what each one keeps.
``malloc_trim(0)`` walks every arena and gives back the whole free pages
(``MADV_DONTNEED``), which is exactly the ~1 GB that was sitting there.

WHEN IT RUNS: when ``bootstrap_calibrations()`` returns or raises; when
the background calibration-package save (the last holder of the raw
frames, until they are encoded) finishes; and when the dashboard's refresh
path has swallowed a FAILED calibration's exception, whose traceback kept
the frames alive through the first trim. A trim measured 16-36 ms on the
Pi, after a calibration that takes ~30 s, and it runs after the
calibration lock is released. It is not called from the per-dart path.

Calibration RESULTS are untouched: the same offline harness, run on the
code before and after this change, produced bit-identical rvec, tvec,
camera matrix and distortion for every camera.

Linux/glibc only. macOS's allocator already returns large frees to the
system (the same offline harness settles at ~380 MB there with no trim),
and Windows is a different heap entirely -- on both, and on a non-glibc
Linux libc without ``malloc_trim``, this is a no-op.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys
import threading
import time

log = logging.getLogger("opendarts.live.heap_trim")

_LIBC_LOCK = threading.Lock()
_MALLOC_TRIM = None
_RESOLVED = False


def _malloc_trim():
    """glibc's ``malloc_trim``, or None where there is no such function.
    Resolved once and cached."""
    global _MALLOC_TRIM, _RESOLVED
    with _LIBC_LOCK:
        if not _RESOLVED:
            _RESOLVED = True
            if sys.platform.startswith("linux"):
                try:
                    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
                    fn = getattr(libc, "malloc_trim", None)
                    if fn is not None:
                        fn.argtypes = [ctypes.c_size_t]
                        fn.restype = ctypes.c_int
                        _MALLOC_TRIM = fn
                except OSError:
                    _MALLOC_TRIM = None
        return _MALLOC_TRIM


def release_freed_heap(reason: str) -> bool:
    """Return freed-but-retained heap pages to the OS. ``reason`` only
    labels the log line. Returns True when a trim actually ran.

    Never raises: this follows a calibration, and a trim that fails must
    not turn a good calibration into an error."""
    try:
        fn = _malloc_trim()
        if fn is None:
            return False
        t0 = time.monotonic()
        fn(0)
        log.info("malloc_trim after %s: %.1f ms", reason, (time.monotonic() - t0) * 1000.0)
        return True
    except Exception: # noqa: BLE001 -- see docstring
        log.exception("malloc_trim after %s failed -- continuing", reason)
        return False
