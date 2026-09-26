"""Publish captured frames to v4l2loopback virtual cameras on Linux.

The Linux counterpart of `opendarts.live.vcam_publish`, and deliberately
the same shape -- `available()`, a per-slot publisher, and a
`VirtualCameraSet`-alike whose `publish_all` is handed straight to the
hub's `frame_sink`. The call sites do not care which platform they are on.

It exists for the same reason the Windows one does: another app cannot
share a physical camera with this product on Linux. V4L2 permits multiple
*opens* of a device but only one *streamer*; the second gets EBUSY. Giving
the other app different devices -- virtual ones fed from our own capture --
is what lets the two run side by side.

WHY MJPEG BY DEFAULT, THOUGH BGR24 IS CHEAPER AND WORKS.

Both formats are delivered correctly to a consumer: publishing BGR24 and
reading a frame back gave mean B/G/R of 105.5/104.3/104.4 against a source
image of 105.9/104.5/104.4 -- the same picture, right channel order, no
loss.

THE ONLY RULE IS ORDERING, and it is unforgiving because nothing reports
it. A v4l2loopback node's format is fixed when it is first opened. A
consumer that opened the node under one format and never renegotiates
keeps decoding that format: change it underneath and raw BGR bytes go into
a JPEG decoder, which renders black while frames keep arriving. So start
OpenDarts before the consuming app, and restart the consuming app after
changing the format -- restarting OpenDarts alone is not enough.

Black frames seen after changing the format under an already-running
consumer are a test of that ordering, not of BGR24.

COST COMPARISON, measured on a real rig with a consumer reading the
virtual cameras, which rendered both formats correctly. The default is the
simple one, not the fast one, and that is a deliberate trade.

BGR24 costs nothing to publish: the array OpenCV already holds is exactly
what the device wants, so it is a write() and no more. Measured on the
Linux rig at 1280x720 with cv2.setNumThreads(1): 0.10 ms per camera
against 3.33 ms to JPEG-encode, which at three cameras and 30fps is the
difference between ~0.3 ms and ~10 ms of every 33 ms pump cycle, on the
same thread that has to detect darts.

What BGR24 costs instead is choreography, all of it measured:

  * A v4l2loopback node advertises MJPG by default. Publish BGR24 and it
    advertises BGR3, which not every consuming app lists as a camera --
    so the cameras may need to be configured in that app before
    publishing starts.
  * The format is fixed when a device is opened, by either side. A
    consumer that opened while the node was MJPG keeps decoding MJPG no
    matter what the producer does afterwards, and shows black.
  * `S_FMT` does not fail when it cannot honour the request -- see
    `_set_format` -- so getting that order wrong is silent.

MJPEG has none of that: it is what the node already advertises, so
nothing has to be sequenced and the cameras behave like ordinary ones in
every UI that lists them. A rig that wants the CPU back can pass
`fmt="BGR24"` and follow the ordering rules, which docs/LINUX.md sets
out.

WHY IT NEVER RAISES. Identical reasoning to the Windows module: this sits
downstream of the capture loop, publishing is a convenience, and a failure
to publish must never affect scoring. Every operation degrades to "not
published" and says so once.

NO SEQLOCK HERE. The Windows writer hand-rolls a seqlock because it owns
both ends of a shared-memory contract and tearing is its problem to solve.
A v4l2loopback write() is a single buffer handed to the kernel, which does
the frame handoff -- there is no partially-visible frame to defend against,
so the equivalent machinery would be inventing a problem.

PUBLISHED INLINE, ON THE PUMP THREAD -- and measured, because it looks
wrong. The hub calls `publish_all` there, so at 3.33 ms per camera this
spends ~10 ms of every 33 ms cycle encoding on the thread that also has
to detect darts. That seems obviously worth moving, and it was: a worker
thread with a single-slot latest-frame buffer was built on 2026-09-15 and
then removed the same day, because measuring it on the rig with a real
consumer attached showed it bought nothing.

Three five-minute samples each, idle, 3 cameras at 1280x720 MJPEG:

    inline   observe p50 5.8/5.7/5.7 ms   p95 8.1/7.8/7.5 ms   0 drops
    worker   observe p50 6.1/6.1/6.0 ms   p95 8.3/8.1/7.9 ms   0 drops

Both published exactly 30.0 fps per camera with zero dropped frames. The
worker was consistently ~0.3 ms SLOWER on the lifecycle's own observe
latency -- GIL contention with the pump, plus a per-frame copy that was
redundant anyway, since `local_capture._pump_once` reassigns
`_last_frames[i]` to a fresh array rather than mutating it.

What the worker did deliver was headroom: the pump coordinator sat at
8.8% of a core instead of 43.3%. Real, and worth nothing at 43% on any
rig this runs on. It would matter on hardware slow enough for the encode
to saturate that thread, and if that day comes the measurements above are
the place to start -- but carrying a worker thread, copy semantics and
drop accounting against a hypothetical is the wrong trade.

Note the total-CPU figure is useless for judging this: the pump threads
busy-wait, so they simply expand to fill whatever the publish thread is
not using. Whole-process CPU went UP when publishing was disabled
entirely (120% to 139%), which says nothing about publishing.
"""
from __future__ import annotations

import errno
import logging
import os
import platform
import struct
import time
from pathlib import Path

# LINUX ONLY, AND IMPORTED DEFENSIVELY. `fcntl` does not exist on Windows,
# and `opendarts.live.vcam` imports BOTH backends before choosing one by
# platform -- so a bare `import fcntl` here is an ImportError at startup on
# every Windows rig, not just when publishing is used. That took the
# Windows rig down on 2026-09-15: the box had been running code that
# predated the Linux virtual cameras, and the first restart that pulled
# them refused to boot with "No module named fcntl".
#
# Guarded rather than moved inside the one function that uses it, because
# the failure mode to prevent is IMPORT-time, and a guard states the
# platform assumption where a reader of the imports will see it.
try:
    import fcntl
except ModuleNotFoundError: # not Linux
    fcntl = None
from typing import Any

import numpy as np

from opendarts.capture.lazy_frame import pixels_of
from opendarts.live import v4l2_register

log = logging.getLogger("opendarts.live.v4l2_publish")

#: struct v4l2_format is 208 bytes on 64-bit, NOT the 204 that counting
#: its fields suggests. `type` is 4 bytes and the union is 200, but the
#: union contains `struct v4l2_window`, which holds a pointer -- so the
#: union is 8-byte aligned, there are 4 bytes of padding after `type`,
#: and the pixel format starts at offset 8.
#:
#: This matters because the size is encoded INTO the ioctl number, so
#: getting it wrong does not produce a validation error about the
#: contents -- it produces ENOTTY, "Inappropriate ioctl for device",
#: which reads as "this device does not support setting a format" rather
#: than "you asked with the wrong number". Measured against a real
#: v4l2loopback node: 204 gives ENOTTY, 208 succeeds.
_V4L2_FORMAT_SIZE = 208
_V4L2_PIX_OFFSET = 8


def _iowr(letter: str, nr: int, size: int) -> int:
    """_IOWR(letter, nr, size) -- derived rather than hardcoded so the
    request number cannot drift away from the struct size above."""
    return (3 << 30) | (size << 16) | (ord(letter) << 8) | nr


#: VIDIOC_S_FMT == _IOWR('V', 5, struct v4l2_format) == 0xC0D05605
VIDIOC_S_FMT = _iowr("V", 5, _V4L2_FORMAT_SIZE)

#: V4L2_BUF_TYPE_VIDEO_OUTPUT -- we are the producer, so OUTPUT, even
#: though consumers see a CAPTURE device on the other side.
BUF_TYPE_VIDEO_OUTPUT = 2

#: fourcc('M','J','P','G') and fourcc('B','G','R','3')
PIX_FMT_MJPEG = 0x47504A4D
PIX_FMT_BGR24 = 0x33524742
_FOURCC = {"MJPEG": PIX_FMT_MJPEG, "BGR24": PIX_FMT_BGR24}
FIELD_NONE = 1
COLORSPACE_JPEG = 7
COLORSPACE_SRGB = 8

#: What to publish unless told otherwise. MJPEG, for the reasons in the
#: module docstring: it is what an idle loopback node already advertises,
#: so no ordering rules apply and the cameras stay visible to consumers.
#: "BGR24" is the cheap alternative for a rig willing to sequence things.
DEFAULT_FORMAT = "MJPEG"

#: Only consulted when `fmt="MJPEG"`. Below the 95 a camera typically
#: ships: this is a feed for other software, not the
#: frames we score from, and 85 is visually indistinguishable for dart
#: detection while roughly halving encode time.
JPEG_QUALITY = 85

#: How long a failed publisher waits before trying to open again.
#: NOT STICKY-FOREVER, which is what this replaced. A publisher that could
#: not claim its format because a consumer held the device latched
#: `_failed = True` and never retried, so killing the consumer removed the
#: CAUSE but not the STATE -- publishing stayed dead until the set was
#: rebuilt, with 30 silent `return False`s a second in the meantime. The
#: reasons a publisher fails here are mostly transient (a consumer
#: attached, a device briefly busy), so the right behaviour is to back off
#: and try again, not to give up until someone notices.
RETRY_AFTER_S = 5.0

def available() -> bool:
    """Whether publishing is possible at all on this machine.

    False everywhere but Linux, and false on a Linux box without the
    v4l2loopback module loaded -- reported honestly so a caller can say
    "not applicable here" rather than offering a control that silently
    does nothing.
    """
    # `fcntl is not None` is belt-and-braces next to the Linux check: it
    # ties the answer to the thing actually required, so this cannot report
    # True on a machine where _set_format() would fail with a NameError.
    return (
        platform.system() == "Linux"
        and fcntl is not None
        and v4l2_register.available()
    )


class V4L2LoopbackPublisher:
    """Publishes frames for one camera slot.

    Per slot rather than one object owning all of them, matching the
    Windows publisher: the slots are independent, a failure on one must
    not stop the others, and the capture loop already iterates per camera.
    """

    def __init__(self, slot: int, width: int, height: int,
                 device: "str | Path", fmt: str = DEFAULT_FORMAT,
                 quality: int = JPEG_QUALITY) -> None:
        self.slot = int(slot)
        self.width = int(width)
        self.height = int(height)
        self.device = Path(device)
        self.fmt = fmt.upper()
        if self.fmt not in ("BGR24", "MJPEG"):
            raise ValueError(f"fmt must be BGR24 or MJPEG, not {fmt!r}")
        self.quality = int(quality)
        self._fd: "int | None" = None
        self.negotiated: "dict[str, Any] | None" = None
        self._frame_index = 0
        self._write_errors = 0
        #: Frames written as the camera's own JPEG vs encoded here.
        self.passthrough_frames = 0
        self.encoded_frames = 0
        # Set once a bad frame or write has been reported, so a 30fps loop
        # says it once. Was read before it was ever assigned, so the
        # first geometry mismatch raised AttributeError instead.
        self._failed = False
        # Retry deadline rather than a sticky flag -- see RETRY_AFTER_S.
        # `_last_error` is kept so a failed publisher can SAY it failed in
        # stats(), instead of vanishing from the diagnostic entirely: an
        # absent entry read as "no device", not "this one gave up".
        self._retry_at = 0.0
        self._last_error: "str | None" = None
        self._complained = False    # log the reason once per failure run

    def open(self) -> bool:
        """Open the device and fix its format. Safe to call repeatedly."""
        if self._fd is not None:
            return True
        now = time.monotonic()
        if now < self._retry_at:
            return False
        if not available():
            # Stated once per failure run rather than never: a rig with
            # v4l2loopback unloaded published nothing and said nothing,
            # which is indistinguishable from publishing being switched
            # off (docs/DESIGN.md).
            self._fail(now, "no v4l2loopback devices (is the module loaded?)")
            return False
        try:
            fd = os.open(str(self.device), os.O_WRONLY)
        except OSError as exc:
            self._fail(now, f"{exc} (is {os.environ.get('USER', 'this user')} "
                            f"in the 'video' group?)")
            return False
        try:
            self._set_format(fd)
        except OSError as exc:
            os.close(fd)
            self._fail(now, f"could not set {self.width}x{self.height} "
                            f"{self.fmt}: {exc}")
            return False
        self._fd = fd
        if self._last_error is not None:
            log.info("v4l2 loopback %d recovered on %s after: %s",
                     self.slot, self.device, self._last_error)
        self._last_error = None
        self._complained = False
        detail = (f"the camera's own JPEG where the slot has it, else q{self.quality}"
                  if self.fmt == "MJPEG" else "no encode")
        log.info("v4l2 loopback %d publishing %dx%d %s (%s) to %s",
                 self.slot, self.width, self.height, self.fmt, detail, self.device)
        return True

    def _fail(self, now: float, reason: str) -> None:
        """Record a failure, complain ONCE per run of them, and schedule a
        retry. Complaining once keeps a 30fps loop from filling the log;
        retrying means removing the cause is enough on its own."""
        self._retry_at = now + RETRY_AFTER_S
        self._last_error = reason
        if not self._complained:
            log.warning(
                "v4l2 loopback %d (%s): not publishing -- %s. Retrying every "
                "%.0fs.", self.slot, self.device, reason, RETRY_AFTER_S,
            )
            self._complained = True

    @property
    def frame_bytes(self) -> int:
        """Bytes one BGR24 frame occupies. Meaningless for MJPEG."""
        return self.width * self.height * 3

    def _set_format(self, fd: int) -> None:
        """VIDIOC_S_FMT for this slot's geometry and chosen pixel format.

        The two formats differ in what the kernel is told to expect.
        BGR24 is fixed-size, so `bytesperline` and `sizeimage` are exact.
        MJPEG is variable-length, so `bytesperline` is 0 and `sizeimage`
        is an upper bound -- the buffer the kernel allocates, not the size
        of any one frame.
        """
        if self.fmt == "BGR24":
            pixelformat = PIX_FMT_BGR24
            bytesperline = self.width * 3
            sizeimage = self.frame_bytes
            colorspace = COLORSPACE_SRGB
        else:
            pixelformat = PIX_FMT_MJPEG
            bytesperline = 0
            sizeimage = self.width * self.height * 2
            colorspace = COLORSPACE_JPEG
        pix = struct.pack(
            "<12I",
            self.width, self.height, pixelformat, FIELD_NONE,
            bytesperline, sizeimage, colorspace,
            0, 0, 0, 0, 0,                  # priv, flags, ycbcr_enc, quantization, xfer_func
        )
        buf = bytearray(_V4L2_FORMAT_SIZE)
        buf[0:4] = struct.pack("<I", BUF_TYPE_VIDEO_OUTPUT)
        buf[_V4L2_PIX_OFFSET:_V4L2_PIX_OFFSET + len(pix)] = pix
        # S_FMT writes back what the driver actually agreed to, which can
        # differ from the request. Read it rather than assume: a silently
        # substituted format would mean publishing bytes the consumer
        # interprets as something else, which looks like corrupt video.
        fcntl.ioctl(fd, VIDIOC_S_FMT, buf)
        agreed = struct.unpack_from("<12I", buf, _V4L2_PIX_OFFSET)
        got = struct.pack("<I", agreed[2]).decode("ascii", "replace")
        self.negotiated = {
            "width": agreed[0], "height": agreed[1], "fourcc": got,
        }
        # THE CHECK THAT MATTERS, 2026-09-15. S_FMT does NOT fail when it
        # cannot honour the request: if a consumer already has the device
        # open, v4l2loopback keeps the existing format, returns success,
        # and writes the OLD fourcc back into the struct. Believing the
        # return code means publishing raw BGR bytes into a stream still
        # declared MJPEG -- the consumer decodes garbage and shows
        # nothing, which reads as "this format is unsupported" rather
        # than "the format never changed". That cost an afternoon.
        #
        # The ordering rule this implies is real and belongs in the log,
        # not just the docs: set the format BEFORE any consumer attaches.
        want = _FOURCC[self.fmt]
        if agreed[2] != want:
            raise OSError(
                errno.EBUSY,
                f"the device kept {got!r} instead of {self.fmt} -- something "
                f"already has it open, and v4l2loopback will not change "
                f"format under a live consumer. Start publishing before the "
                f"consumer attaches."
            )

    def publish(self, frame: "np.ndarray", jpeg: "bytes | None" = None) -> bool:
        """Write one BGR frame. Returns whether it landed.

        In BGR24 mode this is the whole operation -- the array OpenCV
        already holds is exactly what the device expects, so there is no
        conversion and no encode, matching the Windows path.

        In MJPEG mode, `jpeg` -- the camera's own bytes for this frame,
        from a passthrough slot -- is written as-is. No encode, and the
        consumer gets the camera's frame rather than a copy of it.
        """
        if self._fd is None and not self.open():
            return False
        if frame is None:
            return False
        h, w = frame.shape[:2]
        if w != self.width or h != self.height or frame.ndim != 3 or frame.shape[2] != 3:
            # Geometry is fixed at construction because it is baked into the
            # device's negotiated format. A mismatch is a caller error, not
            # something to silently resize -- resizing on the capture path
            # would cost time and hide a real config problem.
            if not self._failed:
                log.warning(
                    "v4l2 loopback %d: frame is %dx%d but the device is set to %dx%d "
                    "-- not publishing (fix camera_resolutions or the slot geometry)",
                    self.slot, w, h, self.width, self.height,
                )
                self._failed = True
            return False
        try:
            if self.fmt == "BGR24":
                # ascontiguousarray is a no-op on a normal capture frame and
                # a copy on a slice; either way tobytes() below needs it
                # contiguous, and asking is cheaper than assuming. A
                # LazyFrame (small-decode slot) decodes here: this mode
                # needs its pixels.
                os.write(self._fd, np.ascontiguousarray(pixels_of(frame)).tobytes())
            elif jpeg is not None:
                # The geometry check above is what makes this safe: the
                # bytes decoded to this frame, and this frame matches the
                # device's negotiated size.
                os.write(self._fd, jpeg)
                self.passthrough_frames += 1
            else:
                import cv2

                ok, encoded = cv2.imencode(
                    ".jpg", pixels_of(frame), [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                )
                if not ok:
                    self._write_errors += 1
                    return False
                os.write(self._fd, encoded.tobytes())
                self.encoded_frames += 1
            self._frame_index += 1
        except Exception as exc:  # noqa: BLE001 -- publishing must never raise
            self._write_errors += 1
            if not self._failed:
                log.warning("v4l2 loopback %d write failed: %s -- not publishing",
                            self.slot, exc)
                self._failed = True
            return False
        return True

    def reader_stats(self) -> "dict[str, Any] | None":
        """What we can honestly say about the consuming side.

        DELIBERATELY THINNER THAN THE WINDOWS EQUIVALENT. There, the reader
        is our own filter and writes its counters back into the shared
        header, so `missed`/`torn` are measured. v4l2loopback exposes no
        equivalent -- the kernel does not tell a writer how many readers
        there are or whether they kept up. Reporting a fabricated
        `consumer_attached` would be worse than reporting none, so the
        keys that cannot be measured are absent rather than guessed.
        """
        if self._fd is None:
            # NOT None ANY MORE. Returning None made stats() emit a bare
            # {"slot": N}, so a publisher that had given up looked
            # identical to one that had never been asked -- which is
            # exactly the state that cost a debugging session on
            # 2026-09-15 (publishing dead, 5000 frames captured, nothing
            # in the diagnostic to say why).
            return {
                "device": str(self.device),
                "format": self.fmt,
                "open": False,
                "published": self._frame_index,
                "last_error": self._last_error,
                "retrying": self._last_error is not None,
            }
        return {
            "open": True,
            "published": self._frame_index,
            "write_errors": self._write_errors,
            "device": str(self.device),
            "format": self.fmt,
            "negotiated": self.negotiated,
            "quality": self.quality if self.fmt == "MJPEG" else None,
            "camera_jpeg_frames": self.passthrough_frames,
            "encoded_frames": self.encoded_frames,
            # Explicit, so a reader of this dict does not assume the
            # Windows keys are merely missing this time.
            "consumer_stats_available": False,
            "consumer_stats_reason":
                "v4l2loopback does not report reader count or drops to the writer",
        }

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


class V4L2LoopbackSet:
    """All slots together, matching the capture hub's own indexing.

    Same surface as `vcam_publish.VirtualCameraSet` so `run_product` can
    hand either one's `publish_all` to the hub without branching.
    """

    def __init__(self, n_cameras: int, width: int, height: int,
                 numbers: "tuple[int, ...]" = v4l2_register.DEFAULT_DEVICE_NUMBERS,
                 fmt: str = DEFAULT_FORMAT, quality: int = JPEG_QUALITY) -> None:
        devices = [Path(f"/dev/video{n}") for n in numbers]
        self.publishers = [
            V4L2LoopbackPublisher(i, width, height, devices[i], fmt, quality)
            for i in range(min(n_cameras, len(devices)))
        ]
        if n_cameras > len(devices):
            log.warning(
                "v4l2 loopback: %d camera(s) but only %d loopback device(s) "
                "(%s) -- the extra slot(s) will not be published; reload "
                "v4l2loopback with devices=%d",
                n_cameras, len(devices),
                ", ".join(str(d) for d in devices), n_cameras,
            )
        self._closed = False

    def publish_all(self, frames: "dict[int, np.ndarray]",
                    jpegs: "dict[int, bytes] | None" = None) -> int:
        """Take the hub's frame dict. THE HUB CALLS THIS ON THE PUMP THREAD,
        and the encode happens here -- see the module docstring for the
        worker that was measured and removed.

        Values may be LazyFrames (opendarts.capture.lazy_frame): MJPEG mode
        forwards the slot's JPEG and reads only the frame's geometry, so
        the hub need not decode full pixels for this sink.
        """
        if self._closed:
            # A pump cycle can already be in flight when publishing is
            # switched off. Returning quietly beats reopening the devices
            # we just released via the publishers' lazy open().
            return 0
        return self._publish_all_now(frames, jpegs or {})

    def _publish_all_now(self, frames: "dict[int, np.ndarray]",
                         jpegs: "dict[int, bytes] | None" = None) -> int:
        """Encode (or pass through) and write, inline. Returns how many
        landed."""
        jpegs = jpegs or {}
        published = 0
        for idx, frame in frames.items():
            if 0 <= idx < len(self.publishers):
                if self.publishers[idx].publish(frame, jpegs.get(idx)):
                    published += 1
        return published

    #: The hub hands this sink LazyFrames rather than decoding for it.
    publish_all.accepts_lazy_frames = True

    def stats(self) -> "list[dict[str, Any]]":
        """Per-slot publish counters, for the diagnostics surface."""
        out = []
        for p in self.publishers:
            s = p.reader_stats() or {}
            s["slot"] = p.slot
            out.append(s)
        return out

    def worker_stats(self) -> "dict[str, Any]":
        """Reports that publishing is inline.

        Kept, rather than deleted with the worker it described, because
        `/api/frame-health` and the Windows backend both answer this and a
        missing key reads as a broken diagnostic rather than an absent
        mechanism. See the module docstring for the measurements that
        removed the worker.
        """
        return {"asynchronous": False}

    def close(self) -> None:
        self._closed = True
        for p in self.publishers:
            p.close()


#: The Windows backend calls this `VirtualCameraSet`, and `opendarts.live.vcam`
#: selects between them by name. Aliased rather than renamed because
#: "V4L2LoopbackSet" is the honest name for what this is when read on its own.
VirtualCameraSet = V4L2LoopbackSet
