"""opendarts/capture/lazy_frame.py -- a camera frame kept as its JPEG, with
detection's small grey picture decoded up front and the full BGR picture
decoded only when something actually needs it.

WHY. Every camera frame used to be fully decoded to 1280x720 BGR in the
hub's pump (~5.3 ms a frame on the Raspberry Pi 5, about half a core for
three cameras at 30 fps), and detection then threw almost all of it away:
the lifecycle converts to grey and shrinks to 1/4 (320x180) before it looks
at anything. libjpeg can produce that small grey picture directly, by
scaling inside the inverse DCT and skipping the colour planes
(``cv2.IMREAD_REDUCED_GRAYSCALE_4``), for a fraction of the cost. The full
picture is only needed by scoring -- the commit frame and the reference
(bg) it is scored against -- and by a few consumers that genuinely look at
pixels (the dashboard preview, calibration, a pixels-format virtual
camera). Everything else (the frame ring, the clips, a JPEG virtual camera)
wants the JPEG itself.

SCORING IS UNCHANGED, BIT FOR BIT. ``LazyFrame.pixels()`` is the same
``cv2.imdecode(jpeg, IMREAD_COLOR)`` the pump ran before, on the same bytes,
so a scored array is identical to the one it would have been. What changes
is detection's intermediate picture: the reduced decode is not the same
arithmetic as cvtColor + INTER_AREA on the full decode (mean difference
0.29 grey levels over the corpus, against a 16-level change threshold), so
a commit can land one frame earlier or later. Accepted by the owner; see
``detect_from_small_decode`` in config.example.json, which turns all of this
off.

THE MAPPING. ``LazyFrames`` is a read-only ``Mapping[int, np.ndarray]`` over
one tick's frames. Code that reads a VALUE gets full pixels, decoded then
and memoised on the frame, so every existing consumer of a frame dict keeps
working unchanged -- it just pays for the decode it asked for. Code that
only needs keys, the small picture, the JPEG or identity reads
``handles()`` instead, and decodes nothing. A consumer this module's author
missed is therefore slower, never wrong.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator

import numpy as np

# cv2 is imported where it is used, not here: the publishers import this
# module for pixels_of(), and cv2's thread count has to be set before cv2
# is first imported anywhere (see run_product).

log = logging.getLogger("opendarts.capture.lazy_frame")

#: libjpeg's in-decoder scale factors that OpenCV exposes for grey output,
#: by name (IMREAD_GRAYSCALE, IMREAD_REDUCED_GRAYSCALE_2/_4/_8).
_REDUCED_GRAY_FLAGS = {
    1: "IMREAD_GRAYSCALE",
    2: "IMREAD_REDUCED_GRAYSCALE_2",
    4: "IMREAD_REDUCED_GRAYSCALE_4",
    8: "IMREAD_REDUCED_GRAYSCALE_8",
}


def reduced_gray(jpeg: bytes, scale: int) -> "np.ndarray | None":
    """The JPEG decoded straight to grey at 1/``scale`` (1280x720 at 4 ->
    320x180). None when the scale has no in-decoder equivalent or the bytes
    will not decode. libjpeg rounds a non-multiple size UP."""
    import cv2

    flag = _REDUCED_GRAY_FLAGS.get(int(scale))
    if flag is None:
        return None
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), getattr(cv2, flag))


def _dimensions(jpeg: bytes) -> "tuple[int, int] | None":
    from opendarts.live import jpeg_info

    dims = jpeg_info.dimensions(jpeg)
    return (int(dims[0]), int(dims[1])) if dims else None


class LazyFrame:
    """One camera frame: its JPEG, its small grey detection picture, and its
    full BGR pixels once someone asks for them.

    ``pixels()`` decodes at most once, under a per-frame lock, so two
    threads asking together get the SAME array -- identity matters here,
    the capture loop pairs arrays with their bytes by ``is``.
    """

    __slots__ = ("jpeg", "small", "scale", "_full", "_lock", "_shape", "_future")

    def __init__(self, jpeg: bytes, small: np.ndarray, scale: int,
                 full: "np.ndarray | None" = None) -> None:
        self.jpeg = jpeg
        self.small = small
        self.scale = int(scale)
        self._full = full
        self._lock = threading.Lock()
        self._shape: "tuple[int, int, int] | None" = None
        self._future: "Future | None" = None

    @property
    def decoded(self) -> bool:
        return self._full is not None

    def pixels(self) -> np.ndarray:
        """The full BGR frame -- exactly what the pump used to publish."""
        full = self._full
        if full is not None:
            return full
        import cv2

        with self._lock:
            if self._full is None:
                full = cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)
                if full is None:
                    # The reduced decode of these same bytes succeeded, so
                    # this should not happen; say so rather than hand back
                    # something that is not these bytes' pixels.
                    raise ValueError(
                        f"lazy frame: {len(self.jpeg)}-byte JPEG decoded small but not full")
                self._full = full
            return self._full

    def is_pixels(self, arr: Any) -> bool:
        """Whether `arr` is this frame's own decoded array (identity)."""
        return self._full is not None and self._full is arr

    @property
    def shape(self) -> "tuple[int, int, int]":
        """The full frame's shape, from the JPEG header -- no decode."""
        if self._full is not None:
            return self._full.shape
        if self._shape is None:
            dims = _dimensions(self.jpeg)
            if dims is None:
                return self.pixels().shape
            self._shape = (dims[0], dims[1], 3)
        return self._shape

    @property
    def ndim(self) -> int:
        return 3

    def __repr__(self) -> str:
        return (f"LazyFrame({len(self.jpeg)} bytes, small={self.small.shape}, "
                f"decoded={self.decoded})")


def pixels_of(frame: Any) -> Any:
    """Full pixels for a frame handle: a LazyFrame decodes (once), an
    array or None comes back as it is."""
    return frame.pixels() if isinstance(frame, LazyFrame) else frame


class LazyFrames(Mapping):
    """One tick's frames, slot -> full BGR pixels, decoded on access.

    Built by ``CameraHub.grab_frames()`` under the hub's cache lock, so each
    slot's handle, JPEG and ring generation describe the same frame. A slot
    with no JPEG (a pixels-only source) holds a plain array.
    """

    __slots__ = ("_handles", "_jpegs", "_generations")

    def __init__(self, handles: "Mapping[int, Any]",
                 jpegs: "Mapping[int, bytes] | None" = None,
                 generations: "Mapping[int, int] | None" = None) -> None:
        self._handles = dict(handles)
        self._jpegs = dict(jpegs or {})
        for cam, h in self._handles.items():
            if isinstance(h, LazyFrame):
                self._jpegs[cam] = h.jpeg
        self._generations = dict(generations or {})

    def __getitem__(self, cam: int) -> np.ndarray:
        return pixels_of(self._handles[cam])

    def __iter__(self) -> Iterator[int]:
        return iter(self._handles)

    def __len__(self) -> int:
        return len(self._handles)

    def __contains__(self, cam: object) -> bool:
        return cam in self._handles

    def handles(self) -> "dict[int, Any]":
        """slot -> LazyFrame or array, WITHOUT decoding anything."""
        return self._handles

    def handle(self, cam: int) -> Any:
        return self._handles[cam]

    def jpeg(self, cam: int) -> "bytes | None":
        return self._jpegs.get(cam)

    def generation(self, cam: int) -> "int | None":
        return self._generations.get(cam)

    def small_gray(self, cam: int, scale: int) -> "np.ndarray | None":
        """The slot's pre-decoded small grey picture at 1/`scale`, or None
        when it has none (a pixels-only slot, or a different scale) -- the
        caller then shrinks the full frame itself, as before."""
        h = self._handles.get(cam)
        if isinstance(h, LazyFrame) and h.scale == scale:
            return h.small
        return None

    def __repr__(self) -> str:
        return f"LazyFrames({self._handles!r})"


def handles_of(frames: "Mapping[int, Any] | None") -> "Mapping[int, Any]":
    """A frame mapping's values WITHOUT decoding: LazyFrames' handles, or a
    plain dict as it is."""
    if isinstance(frames, LazyFrames):
        return frames.handles()
    return frames if frames is not None else {}


def as_frames(handles: "dict[int, Any]") -> "Mapping[int, Any]":
    """A frame dict for consumers that read pixels: `handles` itself when
    every value is already an array (the switch-off path, unchanged), else a
    LazyFrames over it."""
    if any(isinstance(h, LazyFrame) for h in handles.values()):
        return LazyFrames(handles)
    return handles


# One small pool for the decodes that sit on a latency path: the commit
# frames and the reference they are scored against, three cameras each.
# cv2 releases the GIL inside imdecode, so these really run in parallel
# (cv2_num_threads=1 only limits OpenCV's own internal threading).
_POOL_WORKERS = 6
_pool: "ThreadPoolExecutor | None" = None
_pool_lock = threading.Lock()


def _decode_pool() -> ThreadPoolExecutor:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=_POOL_WORKERS,
                                       thread_name_prefix="lazy-decode")
        return _pool


def prefetch(frames: "Mapping[int, Any] | None") -> int:
    """Start decoding every not-yet-decoded frame in `frames` in the
    background, without waiting. Returns how many were started. Used for
    the reference as soon as a dart appears, so that decode is already done
    when the dart commits."""
    started = 0
    for h in handles_of(frames).values():
        if isinstance(h, LazyFrame) and not h.decoded and h._future is None:
            h._future = _decode_pool().submit(h.pixels)
            started += 1
    return started


def decode_all(*frame_sets: "Mapping[int, Any] | None") -> "list[dict[int, Any]]":
    """Each frame set as a plain ``dict`` of full pixels, every pending
    decode across all of them run in parallel. A set that is already all
    arrays comes back as ``dict(set)`` -- exactly what the caller did before
    this existed."""
    pending: "list[Future]" = []
    for frames in frame_sets:
        for h in handles_of(frames).values():
            if isinstance(h, LazyFrame) and not h.decoded:
                if h._future is None:
                    h._future = _decode_pool().submit(h.pixels)
                pending.append(h._future)
    for fut in pending:
        fut.result()
    return [
        {cam: pixels_of(h) for cam, h in handles_of(frames).items()}
        for frames in frame_sets
    ]
