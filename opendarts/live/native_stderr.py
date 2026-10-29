"""opendarts/live/native_stderr.py -- drop known-harmless native warnings from
the process's stderr, so the console stays readable.

Some USB cameras (the Sonix ones on the reference rig) pad every MJPEG frame
with a few bytes before the end-of-image marker. libjpeg decodes the frame
correctly and then says so -- "Corrupt JPEG data: 5 extraneous bytes before
marker 0xd9" -- straight to file descriptor 2, from C, once per decoded
frame. Python's logging never sees it and OpenCV's log level does not
reach it; at 30 frames a second from three cameras it buries the console
log (which is on stdout) completely.

So fd 2 is pointed at a pipe, and one daemon thread copies everything that
arrives on it back to the real stderr -- except lines matching
DROPPED_PREFIXES, which are counted instead. Everything else (tracebacks,
other native warnings) passes through unchanged, line by line. The file log
is not involved at all.

Opt out with OPENDARTS_RAW_STDERR=1 (e.g. to see every native message while
debugging a camera).
"""
from __future__ import annotations

import os
import sys
import threading

#: Lines starting with any of these are dropped. Only messages known to be
#: harmless belong here.
DROPPED_PREFIXES: tuple[bytes, ...] = (
    b"Corrupt JPEG data: ",  # libjpeg: padding before EOI, frame decoded fine
)

_dropped = 0
_installed = False


def dropped_count() -> int:
    """Lines dropped since install (for a summary line, or tests)."""
    return _dropped


def _pump(read_fd: int, out_fd: int) -> None:
    global _dropped
    buf = b""
    while True:
        try:
            chunk = os.read(read_fd, 65536)
        except OSError:
            return
        if not chunk:
            if buf:
                os.write(out_fd, buf)
            return
        buf += chunk
        *lines, buf = buf.split(b"\n")
        keep = []
        for line in lines:
            if line.startswith(DROPPED_PREFIXES):
                _dropped += 1
            else:
                keep.append(line + b"\n")
        if keep:
            try:
                os.write(out_fd, b"".join(keep))
            except OSError:
                pass


def install() -> bool:
    """Route fd 2 through the filter. Idempotent; False if opted out or the
    descriptors can't be set up (the process then runs with raw stderr)."""
    global _installed
    if _installed or os.environ.get("OPENDARTS_RAW_STDERR"):
        return False
    try:
        sys.stderr.flush()
        real = os.dup(2)
        read_fd, write_fd = os.pipe()
        os.dup2(write_fd, 2)
        os.close(write_fd)
    except OSError:
        return False
    threading.Thread(target=_pump, args=(read_fd, real), name="native-stderr-filter", daemon=True).start()
    _installed = True
    return True
