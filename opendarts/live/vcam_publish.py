"""Publish captured frames to the Windows virtual cameras.

This is the writer half of the shared-memory contract in
`tools/winvcam/shared_frame.h`; the DirectShow filter DLL is the reader.
It exists so other software can watch the same board as this product on
Windows, where the two cannot share a physical camera: DirectShow takes a
device exclusively. Giving the other app *different* devices -- virtual
ones fed from our own capture -- is what lets the two run side by side.

Windows only, and a no-op everywhere else. macOS shares cameras natively
through AVFoundation, so nothing is needed there; Linux reaches the same
place through v4l2loopback, which is a different mechanism and not this
module's business.

WHY IT NEVER RAISES. This sits downstream of the capture loop. A virtual
camera is a diagnostic convenience, and a failure to publish must never be
able to affect scoring -- so every operation degrades to "not published"
and says so once, rather than propagating.

WHY THIS PUBLISHES ON THE CAPTURE THREAD, like the Linux backend does.
Both were briefly asynchronous on 2026-09-15 and both are inline again.
The cost here never justified a worker in the first place: there is no
encode at all, since the array OpenCV already holds is exactly what the
reader wants -- 0.10 ms per camera, ~0.3 ms per cycle -- while copying
the frames out of the pump's cache to hand them over costs about the
same, so the handoff would have paid for itself twice and delivered
nothing.

The Linux backend had a real cost to move (3.33 ms per camera to JPEG
encode, ~10 ms of every 33 ms cycle) and moving it STILL did not help:
measured on the rig, the worker was ~0.3 ms slower on the lifecycle's
observe latency, with identical throughput and zero drops either way.
See `opendarts.live.v4l2_publish`'s module docstring for those numbers.

TEARING. One writer, occasional readers, and the writer must never block:
that is a seqlock. `sequence` goes odd before the pixels are touched and
even again afterwards, so a reader that samples mid-write sees an odd
counter and skips the frame. A dropped frame at 30fps is invisible; a torn
one is a visible glitch that would look like a capture fault.
"""
from __future__ import annotations

import functools
import logging
import platform
import struct
import time
from typing import Any

import numpy as np

from opendarts.capture.lazy_frame import pixels_of

log = logging.getLogger("opendarts.live.vcam_publish")

MAGIC = 0x4344564F          # 'ODVC'
# 2 since the mapping can carry JPEG (2026-09-17). A version-1 filter
# refuses the mapping and shows its test pattern -- visibly not live --
# instead of repeating its last BGR frame forever, which a format it did
# not understand would otherwise make it do.
VERSION = 2
HEADER_SIZE = 64
FORMAT_BGR24 = 0
#: The camera's own JPEG. `stride` then holds the JPEG's byte length.
FORMAT_MJPEG = 1
_FORMAT_NAMES = {FORMAT_BGR24: "BGR24", FORMAT_MJPEG: "MJPEG"}
NAME_FMT = "Local\\ODVCam{}"
#: Auto-reset event set after each frame, so the filter pushes on arrival
#: rather than on its own clock. Must match ODVCAM_EVENT_FMT.
EVENT_FMT = "Local\\ODVCamEvt{}"

# Must match ODVCamHeader in tools/winvcam/shared_frame.h.
# The writer touches ONLY the first 40 bytes: the rest is the reader's
# counter block, and rewriting the whole header every frame would clobber
# it -- destroying the one measurement that shows whether the consumer is
# keeping up.
_HEADER = struct.Struct("<8IQ")          # 40 bytes, writer-owned
WRITER_BYTES = 40
_READER_STATS = struct.Struct("<4I")     # 16 bytes at offset 40, reader-owned
READER_STATS_OFFSET = 40


#: How long a failed publisher waits before trying to open again.
RETRY_AFTER_S = 5.0


def available() -> bool:
    """Whether publishing is possible at all on this machine.

    False everywhere but Windows -- reported honestly so a caller can say
    "not applicable here" rather than offering a control that silently
    does nothing.
    """
    return platform.system() == "Windows"


@functools.lru_cache(maxsize=1)
def _kernel32() -> Any:
    import ctypes

    k = ctypes.windll.kernel32  # type: ignore[attr-defined]
    k.CreateEventW.restype = ctypes.c_void_p
    k.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    k.SetEvent.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    return k


def _create_event(name: str) -> Any:
    """The named frame event, or None -- the filter polls without it."""
    try:
        return _kernel32().CreateEventW(None, 0, 0, name) or None
    except Exception:  # noqa: BLE001 -- an optimisation, never required
        return None


def _set_event(handle: Any) -> None:
    try:
        _kernel32().SetEvent(handle)
    except Exception:  # noqa: BLE001
        pass


def _close_handle(handle: Any) -> None:
    try:
        _kernel32().CloseHandle(handle)
    except Exception:  # noqa: BLE001
        pass


class VirtualCameraPublisher:
    """Publishes frames for one camera slot.

    Created per slot rather than one object owning all of them: the slots
    are independent, a failure on one must not stop the others, and the
    capture loop already iterates per camera.
    """

    def __init__(self, slot: int, width: int, height: int) -> None:
        self.slot = int(slot)
        self.width = int(width)
        self.height = int(height)
        self._mm: Any = None
        self._event: Any = None
        self._seq = 0
        self._frame_index = 0
        # Retry deadline, not a sticky flag. A publisher that failed once
        # used to stay dead for the life of the process, so whatever
        # blocked it -- a mapping that could not be created yet, a
        # consumer mid-restart -- kept publishing off long after the cause
        # was gone, silently, at thirty attempts a second. The v4l2
        # backend had the identical bug; see RETRY_AFTER_S there.
        self._retry_at = 0.0
        self._last_error: "str | None" = None
        self._complained = False
        self._format = FORMAT_BGR24
        #: Frames written as the camera's own JPEG vs as pixels.
        self.jpeg_frames = 0
        self.bgr_frames = 0

    @property
    def size(self) -> int:
        return HEADER_SIZE + self.width * self.height * 3

    def open(self) -> bool:
        """Create the mapping. Safe to call repeatedly."""
        if self._mm is not None:
            return True
        import time as _t

        now = _t.monotonic()
        if now < self._retry_at:
            return False
        if not available():
            self._fail(now, "virtual cameras are a Windows-only feature")
            return False
        try:
            import mmap

            # tagname= creates a NAMED mapping backed by the page file,
            # which is what OpenFileMappingW on the reader side looks up.
            # No file on disk is involved.
            self._mm = mmap.mmap(-1, self.size, tagname=NAME_FMT.format(self.slot))
        except Exception as exc:  # noqa: BLE001 -- publishing must never raise
            self._fail(now, f"could not create the shared mapping: {exc}")
            return False
        self._event = _create_event(EVENT_FMT.format(self.slot))
        if self._last_error is not None:
            log.info("virtual camera %d recovered after: %s", self.slot, self._last_error)
        self._last_error = None
        self._complained = False
        log.info("virtual camera %d publishing %dx%d (%s)",
                 self.slot, self.width, self.height, NAME_FMT.format(self.slot))
        return True

    def _fail(self, now: float, reason: str) -> None:
        """Record why publishing is off, complain once per run of
        failures, and schedule a retry. Mirrors the v4l2 backend."""
        self._retry_at = now + RETRY_AFTER_S
        self._last_error = reason
        if not self._complained:
            log.warning(
                "virtual camera %d: not publishing -- %s. Retrying every %.0fs.",
                self.slot, reason, RETRY_AFTER_S,
            )
            self._complained = True

    def publish(self, frame: "np.ndarray", jpeg: "bytes | None" = None) -> bool:
        """Write one frame. Returns whether it was published.

        With `jpeg` -- the camera's own bytes for this frame -- the mapping
        carries the JPEG, and the filter hands it to the consumer as MJPG,
        the way the real camera would. Otherwise it carries the pixels.
        The mapping is sized for pixels, so any JPEG of this geometry fits;
        one that somehow does not is written as pixels instead.
        """
        if self._mm is None and not self.open():
            return False
        if frame is None:
            return False
        h, w = frame.shape[:2]
        if w != self.width or h != self.height or frame.ndim != 3 or frame.shape[2] != 3:
            # Geometry is fixed at construction because it is baked into the
            # mapping size and the reader's media type. A mismatch is a
            # caller error, not something to silently resize -- resizing on
            # the capture path would cost time and hide a real config problem.
            if not self._complained:
                log.warning(
                    "virtual camera %d: frame is %dx%d but the mapping is %dx%d "
                    "-- not publishing (fix camera_resolutions or the slot geometry)",
                    self.slot, w, h, self.width, self.height,
                )
                self._complained = True
            return False

        try:
            if jpeg is not None and len(jpeg) <= self.width * self.height * 3:
                payload = jpeg
                fmt, stride = FORMAT_MJPEG, len(jpeg)
            else:
                # A LazyFrame (small-decode slot) decodes only here.
                payload = np.ascontiguousarray(pixels_of(frame)).tobytes()
                fmt, stride = FORMAT_BGR24, self.width * 3
            self._seq = (self._seq + 1) & 0xFFFFFFFF      # -> odd, write starting
            self._write_header(self._seq, fmt, stride)
            self._mm[HEADER_SIZE:HEADER_SIZE + len(payload)] = payload
            self._seq = (self._seq + 1) & 0xFFFFFFFF      # -> even, complete
            self._frame_index += 1
            self._write_header(self._seq, fmt, stride)
            if self._event:
                _set_event(self._event)
            self._format = fmt
            if fmt == FORMAT_MJPEG:
                self.jpeg_frames += 1
            else:
                self.bgr_frames += 1
        except Exception as exc:  # noqa: BLE001
            if not self._complained:
                log.warning("virtual camera %d write failed: %s -- not publishing",
                            self.slot, exc)
                self._complained = True
            return False
        return True

    def _write_header(self, sequence: int, fmt: int = FORMAT_BGR24,
                      stride: "int | None" = None) -> None:
        self._mm[0:WRITER_BYTES] = _HEADER.pack(
            MAGIC, VERSION, self.width, self.height,
            self.width * 3 if stride is None else stride, fmt, sequence,
            self._frame_index, time.monotonic_ns(),
        )

    def reader_stats(self) -> "dict[str, Any] | None":
        """What the consuming filter reports back, or None if unavailable.

        `frames_missed` is the answer to "are frames being dropped on the
        way out": the reader counts gaps in frame_index, so a non-zero
        value is direct evidence that frames were published and never
        collected, not an inference from rates.
        """
        if self._mm is None:
            # NOT None. Returning None left the diagnostics row with
            # nothing to print but "unknown", which is what a Windows rig
            # showed while publishing was silently off -- capture running,
            # publish enabled, no frames leaving and no reason anywhere.
            return {
                "format": _FORMAT_NAMES[self._format],
                "width": self.width,
                "height": self.height,
                "open": False,
                "published": self._frame_index,
                "last_error": self._last_error or "the shared mapping has not been created yet",
                "retrying": True,
            }
        try:
            read, missed, torn, tick = _READER_STATS.unpack(
                self._mm[READER_STATS_OFFSET:READER_STATS_OFFSET + 16]
            )
        except Exception:  # noqa: BLE001
            return None
        published = self._frame_index
        return {
            # The format of the LAST frame written: MJPEG for a slot with
            # the camera's own JPEG, BGR24 otherwise.
            "format": _FORMAT_NAMES[self._format],
            "camera_jpeg_frames": self.jpeg_frames,
            "bgr_frames": self.bgr_frames,
            "width": self.width,
            "height": self.height,
            "open": True,
            "published": published,
            "read": read,
            "missed": missed,
            "torn": torn,
            "last_read_tick_ms": tick,
            # A consumer that has never read anything is a different
            # problem from one that is merely behind, so say which.
            "consumer_attached": read > 0,
            "drop_rate": (missed / published) if published else 0.0,
        }

    def close(self) -> None:
        if self._mm is not None:
            try:
                # Zero the magic so a reader that still has the view mapped
                # sees a dead mapping instead of the last frame frozen
                # forever -- a stale live-looking picture is worse than none.
                self._mm[0:4] = struct.pack("<I", 0)
                self._mm.close()
            except Exception:  # noqa: BLE001
                pass
            self._mm = None
        if self._event:
            _close_handle(self._event)
            self._event = None


class VirtualCameraSet:
    """All slots together, matching the capture hub's own indexing."""

    def __init__(self, n_cameras: int, width: int, height: int) -> None:
        self.publishers = [
            VirtualCameraPublisher(i, width, height) for i in range(n_cameras)
        ]
        self._closed = False

    def publish_all(self, frames: "dict[int, np.ndarray]",
                    jpegs: "dict[int, bytes] | None" = None) -> int:
        """Publish whatever the hub handed back. Returns how many landed.

        Takes the same dict shape the capture loop already has, so the call
        site stays a single line and no per-camera bookkeeping leaks into it.

        Synchronous on purpose -- see the module docstring's WHY THIS ONE
        STAYS ON THE CAPTURE THREAD section.
        """
        if self._closed:
            # A pump cycle can already be in flight when the oracle toggle
            # closes this set. Publishing now would re-create the mapping
            # close() just killed, leaving a consumer staring at one frozen
            # frame that looks live -- the exact thing close() zeroes the
            # magic to prevent.
            return 0
        jpegs = jpegs or {}
        published = 0
        for idx, frame in frames.items():
            if 0 <= idx < len(self.publishers):
                if self.publishers[idx].publish(frame, jpegs.get(idx)):
                    published += 1
        return published

    #: The hub hands this sink LazyFrames rather than decoding for it:
    #: a slot with its JPEG is forwarded as the JPEG, geometry only.
    publish_all.accepts_lazy_frames = True

    def stats(self) -> "list[dict[str, Any]]":
        """Per-slot publish/consume counters, for the diagnostics surface."""
        out = []
        for p in self.publishers:
            s = p.reader_stats() or {}
            s["slot"] = p.slot
            out.append(s)
        return out

    def worker_stats(self) -> "dict[str, Any]":
        """Mirrors the Linux set's key of the same name.

        Answered rather than omitted so /api/frame-health can say "this
        platform publishes inline" instead of leaving a reader to guess
        whether the field is missing or the worker is dead.
        """
        return {"asynchronous": False}

    def close(self) -> None:
        self._closed = True
        for p in self.publishers:
            p.close()
