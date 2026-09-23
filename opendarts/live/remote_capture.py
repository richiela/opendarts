"""opendarts/live/remote_capture.py -- one camera SLOT whose frames arrive
over the network instead of off a local device.

WHY THIS EXISTS. One machine holds the cameras and publishes; any number of
machines consume, INCLUDING the publisher itself. That decouples hardware
from compute: a box with no cameras can run the whole product, two rigs can
score the same throws, and comparing them stops meaning "the same moment"
and starts meaning "the same pixels".

The publisher drinking its own stream is the load-bearing part. If it read
locally while everyone else read the stream, it would see better pixels
than its peers and every comparison would quietly favour it. Same source
for everyone, or the comparison is rigged.

WHY MJPEG RATHER THAN RTSP. The transport is this project's own existing
``GET /api/cameras/{cam}/stream.mjpg``, which was built as a dashboard
preview and turns out to have exactly the properties a transport needs:
HTTP fan-out to many clients, and self-contained frames. That last one
matters more than it sounds -- every JPEG decodes independently, so a
consumer that joins late or drops a frame still reconstructs pixels
identical to everyone else's. H.264 with a normal GOP does not give you
that for free, and RTSP fan-out would need a separate media server on the
rig. Measured: ~41 Mbps per consumer at 1280x720, 30fps, three cameras --
comfortable on a LAN, so H.264's efficiency buys nothing here.

WHAT THIS IS NOT FOR. JPEG is lossy, so these are not the exact sensor
pixels. Cross-consumer comparison is unaffected (everyone decodes the same
bytes), but ACCURACY work still belongs on ``replay.py`` against the
corpus, where frames are lossless PNG and nothing is re-encoded.

WHAT CHANGED, 2026-09-16 -- READ THIS BEFORE LOOKING FOR THE OLD CLASSES.
This module used to hold TWO hubs: ``RemoteCameraHub`` (a whole parallel
camera hub whose slots were all streams) and ``SwitchableCameraHub`` (a
wrapper that composed a ``LocalCameraHub`` and a ``RemoteCameraHub`` side
by side and mapped slot -> (which child, which index inside it)). Each
child believed it was the whole world and the wrapper reconciled them.
Every bug found in this area on 2026-09-15 was a failure of that
reconciliation rather than of either child's own logic:

1. ``frame_generation()`` summed the children, but the two counted in
   DIFFERENT UNITS -- the local pump bumped once per CYCLE, the remote
   readers once per decoded FRAME. ``capture_daemon`` reads
   ``dropped = new - last - 1`` as "frame SETS missed", so a healthy
   three-stream rig reported ~2 dropped sets per iteration and woke the
   loop 3x per set. Fixed twice (55fc379, then max() instead of sum()).
2. ``set_frame_sink`` forwarded only to the local child. On an all-stream
   rig the local child never pumps, so the sink was attached to something
   that never produced a frame and NOTHING was ever published -- while
   ``frame_sink_attached`` correctly reported True.
3. ``_last_frames`` had to be re-merged by hand or the memory probe
   reported a confident 0.00 MB of frame cache.
4. The local child was constructed with ``configs=None`` on an all-stream
   rig, and ``LocalCameraHub`` reads ``configs or [CameraConfig(device=d)
   for d in DEFAULT_CAMERA_DEVICES]`` -- so BOTH None and [] mean "the
   default three devices". It opened three phantom cameras, ~2.6s each,
   serially, before the capture loop could run (67276fa).

Four bugs, one root cause: two numbering schemes and a translation layer
between them. There is now exactly ONE hub
(:class:`opendarts.live.local_capture.CameraHub`), one frame cache, one
generation counter and one frame sink, and a slot's SOURCE is the only
thing that differs. What is left in this module is that source: the
MJPEG reader, and :func:`build_hub`, which decides per slot.

THE ONE RULE THIS MODULE MUST NOT BREAK: the capture loop's ``grab()``
never touches the network. It reads the hub's frame cache, and this
module's reader threads are what fill it -- exactly the arrangement
LocalCameraHub's pump thread exists to provide for a physical camera.
"""
from __future__ import annotations

import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

from opendarts.live.local_capture import CameraConfig, CameraStatus

log = logging.getLogger("opendarts.remote_capture")

#: Boundary the publisher uses. Kept as a fallback only -- the real one is
#: read from the response's own Content-Type, because trusting a constant to
#: match a remote server is how a parser silently stops finding frames.
DEFAULT_BOUNDARY = "opendarts-mjpeg-frame"

#: How long a reader waits for its first byte, and between reconnects.
CONNECT_TIMEOUT_S = 5.0

#: Backoff after a failed or ended stream. The publisher may simply not be
#: started yet, which is an ordinary state rather than an error, so this
#: retries quietly and forever rather than giving up.
RECONNECT_DELAY_S = 1.0

#: Minimum gap between repeated "this stream is down" lines for ONE camera.
#: A reader retries every RECONNECT_DELAY_S forever, so logging every
#: attempt would push a line per camera per second and bury everything
#: else -- the reason the reader was silent in the first place. Throttled
#: instead of silent: the first failure logs immediately, repeats are
#: counted and summarised.
STREAM_DOWN_LOG_INTERVAL_S = 15.0

#: A stream that stops delivering does NOT raise -- the socket just goes
#: quiet. This is the only thing that notices, and it mirrors the
#: publisher's own MJPEG_STALL_TIMEOUT_S.
READ_STALL_TIMEOUT_S = 5.0

#: Cap on one JPEG part. A corrupt Content-Length must not turn into a
#: multi-gigabyte read.
MAX_PART_BYTES = 32 * 1024 * 1024

#: How long the PUMP will block waiting for this source to produce a frame
#: it has not already served, measured from the last frame that actually
#: arrived.
#:
#: This is what paces the pump on a stream slot, and it is the successor to
#: the deleted RemoteCameraHub's INCOMPLETE_SET_TIMEOUT_S (same 0.5s, same
#: reasoning, one layer down). A local slot needs nothing like it: its
#: cap.read() blocks until the driver has the next frame, so the camera
#: itself paces the cycle. A stream source that answered "here is the
#: newest frame I have" immediately would leave the pump free-running --
#: it would spin, hand the same frame back repeatedly, and bump the
#: generation many times per real frame set, which is bug (1) above in a
#: new costume. So the source BLOCKS until it has something new, and one
#: pump cycle is one frame set again.
#:
#: Bounded, and bounded RELATIVE TO THE LAST FRAME rather than to the start
#: of the wait, which is the part that matters on a mixed rig. A stream
#: that dies costs exactly one cycle of this timeout; after that its own
#: last-frame stamp is already older than the budget, so every subsequent
#: read returns immediately and the pump runs at whatever the LIVE slots
#: can do. Without the "relative to the last frame" part, one dead stream
#: would cap a rig with three healthy local cameras at 2Hz.
#:
#: 0.5s is ~15 frame periods at 30fps -- long enough that it never fires on
#: ordinary jitter, short enough that a stalled stream degrades to "wake on
#: whatever is arriving" rather than stopping the loop.
STREAM_FRAME_WAIT_S = 0.5


def urls_for(base_url: str, n_cameras: int) -> list[str]:
    """The stream URL per camera slot on a publisher.

    ``full=1`` asks for the untouched frame. Without it the publisher
    serves its dashboard-preview defaults -- downscaled and rate-capped --
    which are right for a browser tab and wrong for a scoring input.
    """
    base = base_url.rstrip("/")
    return [f"{base}/api/cameras/{i}/stream.mjpg?full=1" for i in range(n_cameras)]


class _MultipartReader:
    """Pulls JPEG parts out of a ``multipart/x-mixed-replace`` body.

    Uses ``Content-Length`` when the publisher sends one (ours always
    does), which makes the read exact rather than a scan for the next
    boundary. Falls back to boundary scanning so a third-party publisher
    still works.
    """

    def __init__(self, stream: Any, boundary: str) -> None:
        self._stream = stream
        self._delim = b"--" + boundary.encode()
        self._buf = b""

    def _fill(self, n: int = 65536) -> bool:
        chunk = self._stream.read(n)
        if not chunk:
            return False
        self._buf += chunk
        return True

    def _read_until(self, token: bytes) -> "bytes | None":
        while token not in self._buf:
            if not self._fill():
                return None
        head, _, rest = self._buf.partition(token)
        self._buf = rest
        return head

    def next_jpeg(self) -> "bytes | None":
        """The next complete JPEG payload, or None when the stream ends."""
        if self._read_until(self._delim) is None:
            return None
        headers = self._read_until(b"\r\n\r\n")
        if headers is None:
            return None

        length = None
        for line in headers.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                try:
                    length = int(line.split(b":", 1)[1].strip())
                except ValueError:
                    length = None
                break

        if length is not None:
            if length <= 0 or length > MAX_PART_BYTES:
                return None
            if len(self._buf) >= length:
                payload, self._buf = self._buf[:length], self._buf[length:]
                return payload
            # Read exactly the rest of the part and join once. Growing the
            # buffer 64 KB at a time copied the whole part on every read,
            # and slicing it out copied it twice more (2026-09-17).
            parts = [self._buf]
            need = length - len(self._buf)
            self._buf = b""
            while need > 0:
                chunk = self._stream.read(need)
                if not chunk:
                    return None
                parts.append(chunk)
                need -= len(chunk)
            return b"".join(parts)
        # No Content-Length: the payload runs to the next boundary.
        payload = self._read_until(self._delim)
        if payload is None:
            return None
        self._buf = self._delim + self._buf
        return payload.rstrip(b"\r\n")


class StreamSource:
    """ONE camera slot fed by an MJPEG stream.

    The hub treats this exactly as it treats a ``cv2.VideoCapture``: the
    pump asks for a frame, blocks while there isn't one, and gets None
    when the source cannot deliver. Everything about HTTP, reconnects and
    multipart parsing stops here.

    A dedicated reader THREAD is not an implementation flourish -- reading
    an MJPEG stream is a blocking socket read with no non-blocking form
    that still delivers whole frames, so the read has to happen somewhere
    that is not the pump. That is the same reason LocalCameraHub has a
    pump thread in the first place: the capture loop must never be the
    thing waiting on I/O.

    The reader owns the CONNECTION half of this slot's
    :class:`CameraStatus` (``opened``, ``last_error``, the two latency
    fields); the pump owns the FRAME half (``frame_count``,
    ``last_read_at``, ``effective_fps``, ``actual_*``), exactly as it does
    for a local camera. Split that way so a stream slot's frame counters
    mean the same thing a local slot's do -- frames the pump actually took
    into the cache -- rather than frames that were decoded and dropped.
    """

    def __init__(self, slot: int, url: str, status: CameraStatus) -> None:
        self.slot = slot
        self.url = url
        self.status = status
        status.backend_used = "mjpeg"

        self._cond = threading.Condition()
        self._frame: "np.ndarray | None" = None
        #: The JPEG part `_frame` was decoded from, kept so this rig can
        #: forward the bytes it received instead of re-encoding them.
        self._jpeg: "bytes | None" = None
        #: Bumped on every decoded frame; `_served` is how far the pump has
        #: got. Comparing the two is what makes read() block for a NEW
        #: frame rather than re-serving one the pump already has -- see
        #: STREAM_FRAME_WAIT_S for why that pacing matters.
        self._seq = 0
        self._served = 0
        self._last_frame_mono: "float | None" = None
        self._started_mono = time.monotonic()

        self._stop = threading.Event()
        self._thread: "threading.Thread | None" = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._started_mono = time.monotonic()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"opendarts-mjpeg-reader-{self.slot}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the reader and wake anyone blocked in read().

        Waking the reader is separate from joining it on purpose: the hub
        stops every source before it joins any of them, so three dying
        streams cost one CONNECT_TIMEOUT_S between them rather than three.
        """
        self._stop.set()
        with self._cond:
            self._cond.notify_all()

    def join(self, timeout: float = CONNECT_TIMEOUT_S + 1.0) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)
        with self._cond:
            self._frame = None
            self._jpeg = None

    # -- the pump's side ------------------------------------------------

    def read(self) -> "np.ndarray | None":
        """The next frame; see read_pair(), which this is minus the bytes."""
        return self.read_pair()[0]

    def read_pair(self) -> "tuple[np.ndarray | None, bytes | None]":
        """The next frame this source has not already handed to the pump,
        with the JPEG part it was decoded from -- or (None, None) if none
        arrives within this source's budget.

        The bytes are what this rig forwards when a consumer asks it for
        full frames, so a restreaming rig passes on what it received
        rather than a second-generation encode of it.

        Blocking, like ``cap.read()``, and for the same reason -- see
        STREAM_FRAME_WAIT_S. Returns None promptly (not after a wait) once
        the stream has already been quiet longer than that budget, so a
        dead slot cannot hold up a cycle the other slots could finish.

        The array is handed over WITHOUT a copy. Safe because the reader
        allocates a fresh array per ``cv2.imdecode`` and never writes into
        one it has published -- unlike a cv2.VideoCapture, which reuses
        its buffer and is why the pump copies a local frame.
        """
        with self._cond:
            while True:
                if self._stop.is_set():
                    return None, None
                if self._frame is not None and self._seq != self._served:
                    self._served = self._seq
                    return self._frame, self._jpeg
                base = (self._last_frame_mono if self._last_frame_mono is not None
                        else self._started_mono)
                remaining = (base + STREAM_FRAME_WAIT_S) - time.monotonic()
                if remaining <= 0:
                    return None, None
                self._cond.wait(remaining)

    def read_failure_reason(self) -> str:
        """Why the pump's last read came back empty, for CameraStatus.

        Prefers whatever the reader recorded (``connection refused``,
        ``HTTP 503``, ``stream ended``) over this layer's own generic
        answer: the reader knows something real, and overwriting it with
        "no frame" is how a slot with a precise explanation ends up
        reported as a shrug.
        """
        return self.status.last_error or "no new frame from the stream"

    # -- the reader -----------------------------------------------------

    def _publish(self, frame: "np.ndarray", jpeg: "bytes | None" = None) -> None:
        with self._cond:
            self._frame = frame
            self._jpeg = jpeg
            self._seq += 1
            self._last_frame_mono = time.monotonic()
            self._cond.notify_all()

    def _reader_loop(self) -> None:
        """This stream, reconnecting forever until stopped.

        Never raises out of this thread: a reader dying silently would
        leave a slot permanently dark with nothing to show for it, which
        is the failure mode hardest to diagnose from the dashboard.
        """
        i, url = self.slot, self.url
        # SAY SOMETHING WHEN A STREAM GOES DOWN. This loop used to record
        # the failure only on the CameraStatus and log nothing at all, so a
        # camera reconnecting once a second was invisible in the log -- fps
        # fell with no stated reason, and the evidence for "the network is
        # the problem" had to be inferred. A reconnect is the single most
        # useful line this module can emit while a rig is being pushed.
        down_since: "float | None" = None
        failures = 0
        last_logged = 0.0
        while not self._stop.is_set():
            try:
                self._stream_once()
                # A STALL RETURNS CLEANLY, IT DOES NOT RAISE. _stream_once
                # gives up after READ_STALL_TIMEOUT_S by setting
                # last_read_ok=False and returning, and "stream ended" does
                # the same -- so the exception path below never sees the
                # failure mode that bandwidth starvation actually produces.
                # Treat a quiet return with a failed status as a failure,
                # or the one case worth logging is the one that stays
                # silent.
                if not self.status.last_read_ok:
                    raise RuntimeError(self.status.last_error or "stream stalled")
                if down_since is not None:
                    log.info(
                        "cam%d: stream recovered after %.1fs and %d failed "
                        "attempt(s) -- %s",
                        i, time.monotonic() - down_since, failures, url,
                    )
                    down_since, failures, last_logged = None, 0, 0.0
            except Exception as exc: # noqa: BLE001 -- a reader must never die
                st = self.status
                st.last_read_ok = False
                st.last_error = f"{type(exc).__name__}: {exc}"
                st.opened = False
                failures += 1
                now = time.monotonic()
                if down_since is None:
                    down_since = now
                # First failure logs at once; repeats are throttled so a
                # per-second retry cannot drown the log.
                if last_logged == 0.0 or (now - last_logged) >= STREAM_DOWN_LOG_INTERVAL_S:
                    log.warning(
                        "cam%d: stream is down (%s) -- %d attempt(s) over "
                        "%.0fs, retrying every %.0fs. %s",
                        i, st.last_error, failures, now - down_since,
                        RECONNECT_DELAY_S, url,
                    )
                    last_logged = now
                # WAKE THE PUMP ON A FAILURE TOO, not only on a frame. The
                # pump may be parked in read() on this source's budget;
                # leaving it there would make every failed connection cost
                # a cycle of STREAM_FRAME_WAIT_S on a rig whose other slots
                # are fine.
                with self._cond:
                    self._cond.notify_all()
            if self._stop.wait(RECONNECT_DELAY_S):
                return

    def _stream_once(self) -> None:
        i, url, st = self.slot, self.url, self.status
        opened_at = time.monotonic()
        req = urllib.request.Request(url, headers={"Accept": "multipart/x-mixed-replace"})
        try:
            resp = urllib.request.urlopen(req, timeout=CONNECT_TIMEOUT_S) # noqa: S310
        except urllib.error.HTTPError as exc:
            # 503 is the publisher honestly saying its capture loop is not
            # running -- an ordinary state, not a fault, so it is recorded
            # without the alarm a real error deserves.
            st.opened = False
            st.last_read_ok = False
            st.last_error = f"HTTP {exc.code}"
            return

        with resp:
            ctype = resp.headers.get("Content-Type", "") or ""
            boundary = DEFAULT_BOUNDARY
            if "boundary=" in ctype:
                boundary = ctype.split("boundary=", 1)[1].strip().strip('"')

            st.opened = True
            st.last_error = None
            if st.open_latency_s is None:
                st.open_latency_s = time.monotonic() - opened_at

            reader = _MultipartReader(resp, boundary)
            first = True
            last_frame_at = time.monotonic()

            while not self._stop.is_set():
                jpg = reader.next_jpeg()
                if jpg is None:
                    st.last_read_ok = False
                    st.last_error = "stream ended"
                    st.opened = False
                    return
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    # A corrupt part is not a dead stream; skip it and
                    # keep reading rather than tearing down a connection
                    # that is otherwise fine.
                    st.last_error = "undecodable JPEG part"
                    if time.monotonic() - last_frame_at > READ_STALL_TIMEOUT_S:
                        st.last_read_ok = False
                        return
                    continue

                last_frame_at = time.monotonic()
                if first and st.first_frame_latency_s is None:
                    st.first_frame_latency_s = last_frame_at - opened_at
                    first = False
                # `last_read_ok` is the reader's own "this connection is
                # delivering" bit here, and _reader_loop's stall check
                # reads it back on return. The PUMP overwrites it per
                # cycle with "did I get a frame from this slot", which is
                # the same question one layer out.
                st.last_read_ok = True
                self._publish(frame, jpg)


def build_hub(
    devices: "list[int] | None",
    urls: "list[str | None] | None",
    *,
    hub_factory: Any = None,
    frame_sink: Any = None,
    camera_configs: "list[CameraConfig] | None" = None,
) -> Any:
    """The one hub, with each slot pointed at a device or a stream.

    There is no branch on "is this rig local or remote" because there is
    no such property of a rig: the choice is per SLOT, made in each camera
    card's dropdown, and a mix is an ordinary state rather than one to
    reject. A control that offers a state the product refuses is worse
    than one that works.

    ``hub_factory`` is the test-injection seam, defaulting to
    :class:`opendarts.live.local_capture.CameraHub`. It was called
    ``local_factory`` while it built the LOCAL half of a two-hub
    composition; the name went with the composition.

    ``devices`` is used only when ``camera_configs`` is absent. The
    deleted wrapper took ``devices`` and then used it for nothing but the
    slot COUNT, so a rig that configured devices without resolutions had
    its assignment silently dropped -- a latent bug, not a behaviour worth
    preserving.
    """
    if hub_factory is None:
        from opendarts.live.local_capture import CameraHub

        hub_factory = CameraHub

    configs = camera_configs
    if not configs and devices:
        configs = [CameraConfig(device=d) for d in devices]

    # Slot count: whatever the caller actually said something about. Left
    # as None when nobody said anything, so the hub takes its own
    # DEFAULT_CAMERA_DEVICES branch and a rig with no config keys builds
    # exactly the hub it always did.
    n = len(configs or []) or len(urls or []) or 0
    slot_urls: "list[str | None] | None" = None
    if urls is not None:
        slot_urls = [u or None for u in urls]
        if n:
            slot_urls = (slot_urls + [None] * n)[:n]

    return hub_factory(configs=configs, frame_sink=frame_sink, urls=slot_urls)
