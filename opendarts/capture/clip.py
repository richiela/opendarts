"""opendarts/capture/clip.py -- per-camera MKV clips for a throw package.

EVERY throw package is JSON + one MKV per camera (2026-09-22). There are
no frame PNGs any more. A package gets ONE clip per camera, written ONCE
(2026-09-26), in one of two shapes:

  * a WINDOW clip (kind "window") holds the whole bg..commit+1 run out
    of the frame ring, taken BY GENERATION: the capture loop knows which
    ring set holds the bg and which holds the scored frame (see
    ThrowTriggerState.bg_generations), so the run is named, not searched
    for. Written when the config's video-record mode wants a recording
    -- see write_window_clips().

  * a STILLS clip (kind "stills") holds exactly two frames -- the bg and
    the scored commit frame -- from the two arrays scoring used, with no
    ring involved. Written when no recording is wanted, and as the
    fallback whenever a window cannot be built or fails its check -- see
    write_still_clips(). write_throw_clip() chooses between the two.

The package's data files are written first and the clip after (its
recording decision can wait on the frame after the commit, or on the
oracle), then meta.json's `video` block is pointed at it atomically --
see point_meta_at_clip(). Until then the package has no `video` block.

Two write paths, one per how the frame was held, and BOTH byte-exact to
what the rig actually scored:

  * JPEG bytes held (every local camera: the camera's own MJPEG on
    Linux/Windows, OpenDarts' own q50 encode on macOS -- see
    docs/CAMERAS.md). The frames are STREAM-COPIED into the MKV
    (av.Packet muxed with no re-encode), and they are the bytes whose
    decode was scored, so decoding a frame back (with the SAME cv2
    decoder scoring used) reproduces the exact pixels that were scored.

  * decoded BGR only (a frame with no JPEG bytes -- rare). The frames are
    FFV1-encoded at pix_fmt `bgr0`, which is LOSSLESS: decode reproduces
    the exact BGR array. (bgr0 is the one RGB-family pixel format this
    ffmpeg build's FFV1 encoder accepts; gbrp/rgb24 are rejected.)

Byte-exactness is the whole point (the clip is the record detection and
scoring are replayed from), so every clip is checked after it is written,
before meta.json points at it. An MJPEG clip gets a BYTE check: its bg and
commit packets are read back from the file, undecoded, and must equal the
JPEG bytes paired with the scored frames (the scored pixels ARE the cv2
decode of those bytes -- local_capture publishes nothing else). An FFV1
clip has no bytes to compare, so its pointer frames are decoded and
compared with the scored arrays. A clip that fails is discarded.

Why per-camera files rather than one 3-stream MKV: the viewer plays one
camera at a time and replay reads one camera's frame by index; separate
files keep both trivial and let a single camera's clip fail without
taking the others.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

log = logging.getLogger(__name__)

#: Nominal container framerate. Every frame is an independent ring sample,
#: not real motion video (same framing as calibration_package's own
#: RAW_VIDEO_CONTAINER_FPS) -- the container needs *some* rate to be valid
#: and nothing downstream depends on the value.
CLIP_CONTAINER_FPS = 30

#: The FFV1 pixel format that is BOTH lossless and RGB-family in this
#: build's encoder. See module docstring.
_FFV1_PIX_FMT = "bgr0"


def write_clip_mjpeg(path: Path, jpegs: list[bytes], width: int, height: int) -> None:
    """Stream-copy the camera's own MJPEG `jpegs` into an MKV at `path`.

    No decode, no re-encode -- the container ends up holding the exact
    bytes the camera produced, which is what makes the clip a faithful
    record on Linux/Windows. `width`/`height` describe the frames (the
    stream needs them; the JPEG bytes carry their own dimensions too)."""
    if not jpegs:
        raise ValueError("write_clip_mjpeg: no frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = av.open(str(path), "w")
    try:
        st = out.add_stream("mjpeg", rate=CLIP_CONTAINER_FPS)
        st.width = int(width)
        st.height = int(height)
        st.pix_fmt = "yuvj420p"
        st.time_base = Fraction(1, CLIP_CONTAINER_FPS)
        for i, jpeg in enumerate(jpegs):
            pkt = av.Packet(jpeg)
            pkt.stream = st
            pkt.pts = i
            pkt.dts = i
            pkt.time_base = st.time_base
            out.mux(pkt)
    finally:
        out.close()


def write_clip_ffv1(path: Path, frames: list[np.ndarray]) -> None:
    """FFV1-encode BGR `frames` into a lossless MKV at `path` (the fallback
    for frames held as decoded pixels with no JPEG bytes)."""
    if not frames:
        raise ValueError("write_clip_ffv1: no frames")
    h, w = frames[0].shape[:2]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = av.open(str(path), "w")
    try:
        st = out.add_stream("ffv1", rate=CLIP_CONTAINER_FPS)
        st.width = int(w)
        st.height = int(h)
        st.pix_fmt = _FFV1_PIX_FMT
        st.time_base = Fraction(1, CLIP_CONTAINER_FPS)
        for f in frames:
            vframe = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(f, dtype=np.uint8), format="bgr24"
            ).reformat(format=_FFV1_PIX_FMT)
            for pkt in st.encode(vframe):
                out.mux(pkt)
        for pkt in st.encode():  # flush
            out.mux(pkt)
    finally:
        out.close()


def read_clip_frames(path: Path) -> list[np.ndarray]:
    """Every frame of `path` as a BGR uint8 array, bit-identical to what
    was written (and therefore to what was scored).

    For an MJPEG clip the JPEG packets are decoded with cv2 -- the SAME
    decoder scoring used -- so the pixels match the scored frame exactly,
    not merely closely (a different JPEG decoder can round IDCT
    differently). For an FFV1 clip the frames are decoded losslessly."""
    path = Path(path)
    inp = av.open(str(path))
    try:
        st = inp.streams.video[0]
        if st.codec_context.name == "mjpeg":
            frames: list[np.ndarray] = []
            for pkt in inp.demux(st):
                if not pkt.size:
                    continue
                arr = cv2.imdecode(
                    np.frombuffer(bytes(pkt), np.uint8), cv2.IMREAD_COLOR
                )
                if arr is None:
                    raise ValueError(
                        f"read_clip_frames: a JPEG packet in {path} would not decode"
                    )
                frames.append(arr)
            return frames
        return [f.to_ndarray(format="bgr24") for f in inp.decode(st)]
    finally:
        inp.close()


def _read_clip_frames_at(path: Path, indices) -> "tuple[int, dict[int, np.ndarray]]":
    """``(n_frames, {index: frame})`` for just the `indices` of `path` that
    are in range -- the pointer reads, without decoding the rest.

    AN MJPEG CLIP IS RANDOM-ACCESS FOR FREE: every packet is a whole JPEG,
    so demuxing all of them is a file read and only the wanted ones are
    decoded, with the same cv2 decoder read_clip_frames() uses -- the
    same bytes through the same decoder, so the same pixels. This used to
    decode the whole clip for every pointer read, and a window write reads
    two pointers back: on the Pi 5 (5.3 ms a 720p decode) that was ~14
    decodes a camera where 2 answer the question. FFV1 is decoded whole,
    as before -- it is the rare fallback and its frames come off a codec
    context, not as independent packets.

    Only the requested packets are decoded, so a clip with some OTHER
    packet that would not decode no longer fails here; one of the
    requested frames that will not decode still raises, as before."""
    path = Path(path)
    wanted = set(indices)
    inp = av.open(str(path))
    try:
        st = inp.streams.video[0]
        if st.codec_context.name != "mjpeg":
            frames = [f.to_ndarray(format="bgr24") for f in inp.decode(st)]
            return len(frames), {i: frames[i] for i in wanted if 0 <= i < len(frames)}
        n = 0
        out: dict[int, np.ndarray] = {}
        for pkt in inp.demux(st):
            if not pkt.size:
                continue
            if n in wanted:
                arr = cv2.imdecode(
                    np.frombuffer(bytes(pkt), np.uint8), cv2.IMREAD_COLOR
                )
                if arr is None:
                    raise ValueError(
                        f"read_clip_frames: a JPEG packet in {path} would not decode"
                    )
                out[n] = arr
            n += 1
        return n, out
    finally:
        inp.close()


def _indexed(path: Path, n: int, frames: "dict[int, np.ndarray]", index: int) -> np.ndarray:
    """Frame `index` of a _read_clip_frames_at() result, with
    read_clip_frame()'s own out-of-range error."""
    if not 0 <= index < n:
        raise IndexError(
            f"read_clip_frame: index {index} out of range for {path} "
            f"({n} frames)"
        )
    return frames[index]


def read_clip_frame(path: Path, index: int) -> np.ndarray:
    """One frame by index (the baseline/commit pointer read). Decodes only
    that frame of an MJPEG clip -- see _read_clip_frames_at()."""
    n, frames = _read_clip_frames_at(path, (index,))
    return _indexed(path, n, frames, index)


def _frame_matches(got: np.ndarray, expected: np.ndarray) -> bool:
    return got.shape == expected.shape and np.array_equal(
        got, np.ascontiguousarray(expected, dtype=np.uint8)
    )


def verify_clip_frame(path: Path, index: int, expected: np.ndarray) -> bool:
    """True iff frame `index` of `path` is byte-identical to `expected`
    (the array that was scored) -- a clip whose commit frame is not
    exactly what was scored is not a faithful record. The write path
    checks its clips with _mjpeg_clip_holds() / _ffv1_clip_holds()."""
    return _frame_matches(read_clip_frame(path, index), expected)


#: Bumped if the pointer/layout below ever changes shape.
#: v2 (2026-09-22): the per-camera pointer gained `bg_index`, and the clip
#: now spans bg..commit+1 rather than a time-derived window. A v1 package
#: has no bg_index and keeps its cam{n}_bg.png, so readers must treat the
#: key as optional rather than assume the newer shape.
#: v3 (2026-09-22, same day): every package has clips, so the block gained
#: `kind` to say which of the two it is. The POINTER shape is unchanged --
#: a v2 reader reads a v3 block correctly and simply does not know the
#: clip is only two frames long, which is why this is a new key and not a
#: new layout.
THROW_CLIP_SCHEMA = "throw-clip/v3"
#: Per-camera RECORDED (window) clip filename inside a package -- the
#: name every recording has had since clips existed.
CLIP_FILENAME_TEMPLATE = "clip_cam{cam}.mkv"
#: Per-camera STILLS clip filename. A different name from a recording so
#: a directory listing tells the two apart without opening meta.json.
#: Readers never build either name themselves -- they follow the
#: pointer's `clip` (a package recorded before 2026-09-26 may also carry
#: a `clip_cam{N}_rec.mkv`, from the old stills-then-upgrade swap).
STILLS_FILENAME_TEMPLATE = "stills_cam{cam}.mkv"

#: The two clip kinds, written into the `video` block's `kind`. See the
#: module docstring. A package saved before `kind` existed is always a
#: WINDOW -- stills clips did not exist then -- which is why
#: is_recorded_clip() defaults the missing key that way rather than
#: guessing from n_frames.
CLIP_KIND_STILLS = "stills"
CLIP_KIND_WINDOW = "window"


def is_recorded_clip(video_meta: "dict | None") -> bool:
    """Whether this package's clips are a RECORDED window from the frame
    ring, as opposed to the two-frame stills clip.

    The distinction exists for the operator, not for the reader: both
    kinds are read through the same pointers, but only a window clip is
    footage of the throw. The dashboard's "Save frames" button and the
    viewer's clip scrubber both hang off this -- before `kind` existed
    they hung off "does meta have a video block at all", which stopped
    meaning anything the moment every package had one."""
    if not video_meta:
        return False
    return (video_meta.get("kind") or CLIP_KIND_WINDOW) == CLIP_KIND_WINDOW

#: THE CLIP IS THE BACKGROUND FRAME THROUGH THE SCORED FRAME, PLUS ONE
#: (2026-09-22). Both ends are real frames the rig scored with, so the
#: window is defined by the frames themselves rather than inferred from a
#: clock:
#:
#:     [ bg ............ commit ] + CLIP_FRAMES_AFTER_COMMIT
#:
#: The bg is the lifecycle's own reference -- re-adopted on EVERY stable
#: idle frame and frozen the instant a dart starts being detected -- so it
#: IS "the last frame before the dart appeared". That is exactly the start
#: the old code spent two tuned constants trying to estimate: take
#: settle_duration_s, convert it to frames, add a lead-in, and hope the bg
#: landed inside. Measured over the corpus it usually did, and it sat right
#: on the edge when it did: bg at clip index 0 or 1 every single time,
#: never 2+ on the MJPEG rigs.
#:
#: Estimating it failed exactly where the clock and the frame rate disagree.
#: On the ~26fps Mac a time-derived window covers fewer FRAMES than on a
#: 30fps rig, so the bg fell outside it 171 times in 942 (18.2%) -- versus
#: 3 in 369 (0.8%) on the rigs. Anchoring on the frame removes the failure
#: mode rather than retuning it, and it is fps-agnostic for free.
#:
#: The payoff is storage: the bg is then GUARANTEED to be in the clip, so
#: the package stops also storing it as a PNG. Same picture, two formats,
#: measured: 1029 KB as PNG vs 75 KB as an MJPEG clip frame (13x), and
#: 1122 KB vs 814 KB even on macOS FFV1. Corpus-wide that is ~1.4 GB of
#: duplicate PNG for roughly zero net frames -- anchoring TRIMS the frames
#: that used to sit before the bg, which very nearly pays for the ones it
#: has to add.
#:
#: The two ends were found by searching the ring for pixels equal to the
#: scored arrays until 2026-09-26; they are NAMED now, by the ring
#: generation the capture loop recorded for each (see
#: write_window_clips()), which costs no decode and cannot stop at an
#: earlier copy of a frozen frame.
#:
#: CLIP_MAX_FRAMES_BEFORE_COMMIT stays as a ceiling, not a target: if the
#: board never settled, the last idle adoption can be far back, and a
#: pathological throw must not write a giant clip. A bg beyond it gets the
#: stills clip instead.
CLIP_FRAMES_AFTER_COMMIT = 1
CLIP_MAX_FRAMES_BEFORE_COMMIT = 12


class CommitFrameNotInClip(RuntimeError):
    """A clip does not read back as the frames that were scored, and there
    is nothing left to fall back to -- see write_still_clips()."""


class ClipWindowUnavailable(RuntimeError):
    """A window clip could not be built or failed its check. Not fatal:
    write_throw_clip() writes the stills clip instead, and says why."""


def read_clip_packets_at(path: Path, indices) -> "tuple[int, dict[int, bytes]]":
    """``(n_packets, {index: raw packet bytes})`` for the `indices` of
    `path` that are in range -- NO decode at all.

    For an MJPEG clip every packet is one whole JPEG, exactly the bytes
    that were muxed in, which is what makes the write-time BYTE check
    possible: the file either holds the paired JPEG at the pointer or it
    does not, and a demux answers that without decoding a pixel."""
    path = Path(path)
    wanted = set(indices)
    inp = av.open(str(path))
    try:
        st = inp.streams.video[0]
        n = 0
        out: dict[int, bytes] = {}
        for pkt in inp.demux(st):
            if not pkt.size:
                continue
            if n in wanted:
                out[n] = bytes(pkt)
            n += 1
        return n, out
    finally:
        inp.close()


def _mjpeg_clip_holds(path: Path, n_frames: int, expected: "dict[int, bytes]") -> bool:
    """The BYTE check: `path` reads back as `n_frames` packets and the
    packet at each index in `expected` is byte-identical to the JPEG given
    for it. See the module docstring for why that is the whole check an
    MJPEG clip needs."""
    try:
        n, got = read_clip_packets_at(path, expected)
    except Exception:  # noqa: BLE001 -- an unreadable clip is a failed check
        return False
    return n == n_frames and all(got.get(i) == data for i, data in expected.items())


def _ffv1_clip_holds(path: Path, n_frames: int, expected: "dict[int, np.ndarray]") -> bool:
    """The PIXEL check, for a clip with no JPEG bytes to compare: `path`
    decodes to `n_frames` frames and the frame at each index in `expected`
    is byte-identical to the array given for it."""
    try:
        n, got = _read_clip_frames_at(path, expected)
    except Exception:  # noqa: BLE001 -- an unreadable clip is a failed check
        return False
    return n == n_frames and all(
        i in got and _frame_matches(got[i], arr) for i, arr in expected.items())


def write_still_clips(
    package_dir: Path,
    bg_frames_by_cam: "dict[int, np.ndarray]",
    commit_frames_by_cam: "dict[int, np.ndarray]",
    *,
    bg_jpegs: "dict[int, bytes] | None" = None,
    commit_jpegs: "dict[int, bytes] | None" = None,
) -> "dict":
    """Write the two-frame STILLS clip for every camera -- bg, then the
    scored commit frame -- and return the pointer block for the package's
    meta.

    Needs no frame ring and no timing: both frames are passed in, so this
    cannot age out and cannot half-succeed. That is why it is the clip a
    package falls back to whenever a recorded window cannot be had.

    `bg_jpegs`/`commit_jpegs` are the camera's OWN JPEG bytes for those
    exact frames, when the rig has them (every local camera --
    opendarts.live.local_capture's `grab_paired`). Given both for a
    camera, the clip is a stream copy of those bytes: no re-encode, and
    ~6x smaller than the lossless alternative (measured on a real
    3-camera package: 465 KB of mjpeg against 2,973 KB of FFV1 for the
    same six frames). Absent -- any frame we could not pair with its
    bytes at the same tick -- the decoded array is FFV1-encoded instead,
    which is bigger and equally exact.

    Every clip is checked after it is written (module docstring): the
    mjpeg clip must read back as exactly those two packets, byte for
    byte, and anything else falls back to FFV1 -- a bigger file, never a
    wrong one. The FFV1 clip is decoded and compared with both arrays,
    and a failure THERE raises: there is nothing left to fall back to,
    and a package must not point at frames that are wrong.

    Returns the same pointer shape write_window_clips() does, with
    ``kind: "stills"``, ``bg_index: 0``, ``commit_index: 1``,
    ``n_frames: 2``.
    """
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    bg_jpegs = bg_jpegs or {}
    commit_jpegs = commit_jpegs or {}

    cameras_out: dict[str, dict] = {}
    for cam in sorted(set(bg_frames_by_cam) & set(commit_frames_by_cam)):
        bg = np.ascontiguousarray(bg_frames_by_cam[cam], dtype=np.uint8)
        commit = np.ascontiguousarray(commit_frames_by_cam[cam], dtype=np.uint8)
        clip_name = STILLS_FILENAME_TEMPLATE.format(cam=cam)
        clip_path = package_dir / clip_name

        encoding: str | None = None
        bg_jpeg, commit_jpeg = bg_jpegs.get(cam), commit_jpegs.get(cam)
        if bg_jpeg and commit_jpeg:
            h, w = commit.shape[:2]
            write_clip_mjpeg(clip_path, [bg_jpeg, commit_jpeg], width=w, height=h)
            if _mjpeg_clip_holds(clip_path, 2, {0: bg_jpeg, 1: commit_jpeg}):
                encoding = "mjpeg"
            else:
                # Loud: the container did not hold the bytes it was given,
                # which is a writer fault, not a storage choice. The package
                # is fine (FFV1 below), but somebody needs to know.
                log.warning(
                    "cam%d: the two-frame clip did NOT read back as the camera JPEGs "
                    "it was written from -- falling back to FFV1.", cam,
                )
        if encoding is None:
            write_clip_ffv1(clip_path, [bg, commit])
            if not _ffv1_clip_holds(clip_path, 2, {0: bg, 1: commit}):
                clip_path.unlink(missing_ok=True)
                raise CommitFrameNotInClip(
                    f"cam{cam}: {clip_name} does not read back as the two frames "
                    "that were scored -- refusing to write a package whose only "
                    "copy of its own frames is wrong"
                )
            encoding = "ffv1"

        cameras_out[str(cam)] = {
            "clip": clip_name,
            "encoding": encoding,
            "n_frames": 2,
            "bg_index": 0,
            "commit_index": 1,
        }

    if not cameras_out:
        raise ValueError(
            "write_still_clips: no camera has BOTH a bg and a commit frame"
        )
    return {
        "schema": THROW_CLIP_SCHEMA,
        "container": "mkv",
        "fps": CLIP_CONTAINER_FPS,
        "kind": CLIP_KIND_STILLS,
        "cameras": cameras_out,
    }


@dataclass(frozen=True)
class ScoredFrames:
    """What one throw was scored from, per camera, and where each frame
    lives in the frame ring.

    `bg`/`commit` are the arrays scoring used. `bg_jpegs`/`commit_jpegs`
    are the camera's own bytes for them and `bg_generations`/
    `commit_generations` the ring generation holding each, all paired by
    array identity at the tick the frame was fetched
    (opendarts.live.capture_daemon._FrameJpegIndex). Any of the four maps
    may lack a camera; only the arrays are required."""

    bg: "dict[int, np.ndarray]"
    commit: "dict[int, np.ndarray]"
    bg_jpegs: "dict[int, bytes]" = field(default_factory=dict)
    commit_jpegs: "dict[int, bytes]" = field(default_factory=dict)
    bg_generations: "dict[int, int]" = field(default_factory=dict)
    commit_generations: "dict[int, int]" = field(default_factory=dict)


def _camera_window(
    cam: int,
    sets: list,
    scored: ScoredFrames,
    *,
    earliest_wall_s: "float | None",
    latest_wall_s: "float | None",
    frames_after_commit: int,
    max_frames_before_commit: int,
) -> "tuple[list, int, int]":
    """This camera's clip, as ``(frame sets, bg_index, commit_index)``:
    the ring sets holding it from the bg's generation through the commit's
    plus `frames_after_commit`. Raises ClipWindowUnavailable with the
    reason when the run cannot be taken.

    A camera's run is the sets that HOLD a frame for it, in generation
    order -- a set it was absent from is skipped, exactly as a camera that
    produced no frame that cycle has no frame to write. The time bounds
    are the capture-instant window the ring slice used to cover
    (opendarts.capture.throw_capture.CLIP_WINDOW_BEFORE_S/AFTER_S): a bg
    older than it gets the stills clip, and an after frame later than it
    is left out."""
    bg_gen = scored.bg_generations.get(cam)
    commit_gen = scored.commit_generations.get(cam)
    if bg_gen is None or commit_gen is None:
        raise ClipWindowUnavailable(
            f"cam{cam}: no ring generation recorded for its "
            f"{'bg' if bg_gen is None else 'commit'} frame")
    run = [fs for fs in sets if cam in fs.jpegs or cam in fs.pixels]
    position = {fs.generation: i for i, fs in enumerate(run)}
    bg_pos, commit_pos = position.get(bg_gen), position.get(commit_gen)
    if bg_pos is None or commit_pos is None:
        raise ClipWindowUnavailable(
            f"cam{cam}: the ring no longer holds its "
            f"{'bg' if bg_pos is None else 'commit'} frame "
            f"(generation {bg_gen if bg_pos is None else commit_gen})")
    if bg_pos > commit_pos:
        raise ClipWindowUnavailable(
            f"cam{cam}: bg generation {bg_gen} is after commit generation {commit_gen}")
    if commit_pos - bg_pos > max_frames_before_commit:
        raise ClipWindowUnavailable(
            f"cam{cam}: bg sits {commit_pos - bg_pos} frames before the "
            f"commit, beyond the {max_frames_before_commit}-frame ceiling")
    if earliest_wall_s is not None and run[bg_pos].wall_s < earliest_wall_s:
        raise ClipWindowUnavailable(
            f"cam{cam}: its bg frame is {earliest_wall_s - run[bg_pos].wall_s:.3f}s "
            "older than the clip window")
    if latest_wall_s is not None and run[commit_pos].wall_s > latest_wall_s:
        raise ClipWindowUnavailable(f"cam{cam}: its commit frame is after the clip window")
    hi = commit_pos + 1
    while (hi - commit_pos - 1 < frames_after_commit and hi < len(run)
           and (latest_wall_s is None or run[hi].wall_s <= latest_wall_s)):
        hi += 1
    return run[bg_pos:hi], 0, commit_pos - bg_pos


def write_window_clips(
    package_dir: Path,
    sets: list,
    scored: ScoredFrames,
    cameras,
    *,
    earliest_wall_s: "float | None" = None,
    latest_wall_s: "float | None" = None,
    frames_after_commit: int = CLIP_FRAMES_AFTER_COMMIT,
    max_frames_before_commit: int = CLIP_MAX_FRAMES_BEFORE_COMMIT,
) -> "dict":
    """Write one WINDOW clip per camera in `cameras` -- the bg..commit+1
    run out of the frame ring -- and return the pointer block for the
    package's meta.

    `sets` are frame-ring sets (opendarts.capture.frame_ring.FrameSet), in
    generation order; each has `pixels` (slot -> BGR array) and `jpegs`
    (slot -> camera JPEG bytes), one or the other per slot. The frames
    are TAKEN BY GENERATION: `scored` says which ring set holds the bg and
    which the scored frame, for each camera, so nothing is decoded or
    compared to find them (see _camera_window()).

    Per camera:
      * every frame of the run a JPEG (every local camera) -> the JPEGs
        are stream-copied into the clip, and the clip then gets the BYTE
        check: its bg and commit packets must equal the JPEG bytes paired
        with the scored frames. A frame whose bytes were not paired cannot
        be checked, so it cannot be stream-copied either;
      * otherwise -> the frames are FFV1-encoded (a JPEG one decoded
        first), and the pointer frames get the pixel check against the
        scored arrays.

    ALL OR NOTHING. Any camera that cannot be taken or fails its check
    raises ClipWindowUnavailable after every window clip this call wrote
    is removed: a package whose cameras disagree about whether they are a
    recording would make every reader ask per camera.

    Returns::

        {"schema": THROW_CLIP_SCHEMA, "container": "mkv", "fps": <int>,
         "kind": "window",
         "cameras": {"0": {"clip": "clip_cam0.mkv", "encoding": "mjpeg"|"ffv1",
                           "n_frames": N, "commit_index": i, "bg_index": 0}, ...}}
    """
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cameras_out: dict[str, dict] = {}
    try:
        for cam in sorted(cameras):
            run, bg_index, commit_index = _camera_window(
                cam, sets, scored,
                earliest_wall_s=earliest_wall_s, latest_wall_s=latest_wall_s,
                frames_after_commit=frames_after_commit,
                max_frames_before_commit=max_frames_before_commit,
            )
            n_frames = len(run)
            clip_name = CLIP_FILENAME_TEMPLATE.format(cam=cam)
            clip_path = package_dir / clip_name
            commit = scored.commit[cam]
            if all(cam in fs.jpegs for fs in run):
                expected = {bg_index: scored.bg_jpegs.get(cam),
                            commit_index: scored.commit_jpegs.get(cam)}
                if any(data is None for data in expected.values()):
                    raise ClipWindowUnavailable(
                        f"cam{cam}: no camera JPEG was paired with its scored "
                        "frames, so a stream copy could not be checked")
                h, w = commit.shape[:2]
                written.append(clip_path)
                write_clip_mjpeg(clip_path, [fs.jpegs[cam] for fs in run], width=w, height=h)
                encoding = "mjpeg"
                ok = _mjpeg_clip_holds(clip_path, n_frames, expected)
            else:
                frames = [fs.pixels[cam] if cam in fs.pixels
                          else _decode_ring_jpeg(fs.jpegs[cam]) for fs in run]
                written.append(clip_path)
                write_clip_ffv1(clip_path, frames)
                encoding = "ffv1"
                ok = _ffv1_clip_holds(clip_path, n_frames, {
                    bg_index: np.ascontiguousarray(scored.bg[cam], dtype=np.uint8),
                    commit_index: np.ascontiguousarray(commit, dtype=np.uint8)})
            if not ok:
                raise ClipWindowUnavailable(
                    f"cam{cam}: {clip_name} failed its {'byte' if encoding == 'mjpeg' else 'pixel'} "
                    "check -- its bg/commit frames are NOT the ones that were scored")
            cameras_out[str(cam)] = {
                "clip": clip_name,
                "encoding": encoding,
                "n_frames": n_frames,
                "commit_index": commit_index,
                "bg_index": bg_index,
            }
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    if not cameras_out:
        raise ClipWindowUnavailable("no camera to write a clip for")
    return {
        "schema": THROW_CLIP_SCHEMA,
        "container": "mkv",
        "fps": CLIP_CONTAINER_FPS,
        "kind": CLIP_KIND_WINDOW,
        "cameras": cameras_out,
    }


def _decode_ring_jpeg(data: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ClipWindowUnavailable("a ring JPEG would not decode")
    return arr


def write_throw_clip(
    package_dir: Path,
    scored: ScoredFrames,
    cameras,
    *,
    sets: "list | None" = None,
    earliest_wall_s: "float | None" = None,
    latest_wall_s: "float | None" = None,
) -> "tuple[dict, str | None]":
    """The ONE clip a package gets: the window clip out of `sets` when a
    recording is wanted (`sets` given), else -- or when the window cannot
    be had -- the two-frame stills clip. Returns ``(video block, reason)``
    where `reason` says why a wanted window was not written (None when it
    was, or was not wanted). Raises only when even the stills clip's
    FFV1 fallback fails (see write_still_clips())."""
    reason: "str | None" = None
    if sets is not None:
        try:
            return write_window_clips(
                package_dir, sets, scored, cameras,
                earliest_wall_s=earliest_wall_s, latest_wall_s=latest_wall_s,
            ), None
        except ClipWindowUnavailable as exc:
            reason = str(exc)
    cams = [c for c in sorted(cameras) if c in scored.bg and c in scored.commit]
    return write_still_clips(
        package_dir,
        {c: scored.bg[c] for c in cams},
        {c: scored.commit[c] for c in cams},
        bg_jpegs={c: scored.bg_jpegs[c] for c in cams if c in scored.bg_jpegs},
        commit_jpegs={c: scored.commit_jpegs[c] for c in cams if c in scored.commit_jpegs},
    ), reason


def point_meta_at_clip(package_dir: Path, video: dict) -> None:
    """Point the package's meta.json `video` block at the clip just
    written -- the moment the package's frames become readable.

    Re-reads meta.json right before writing, so an annotation another
    writer made after the save is kept, and replaces it atomically (temp
    file + os.replace): a reader sees the package without a `video` block
    or with the finished one, never a half-written file."""
    meta_path = Path(package_dir) / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["video"] = video
    tmp = meta_path.with_name(f".{meta_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, meta_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_commit_frame(package_dir: Path, video_meta: dict, cam: int) -> np.ndarray:
    """The scored commit ("after") frame for `cam`, read byte-identically
    from its clip via the pointer in `video_meta` (a package's meta
    ``video`` block). Raises KeyError if this camera has no clip."""
    entry = video_meta["cameras"][str(cam)]
    clip_path = Path(package_dir) / entry["clip"]
    return read_clip_frame(clip_path, int(entry["commit_index"]))


def read_bg_frame(package_dir: Path, video_meta: dict, cam: int) -> "np.ndarray | None":
    """The bg reference for `cam`, read byte-identically from its clip.

    None when this package predates the bg pointer (throw-clip/v1) or the
    bg could not be de-duplicated into the clip -- in both cases the
    package still has its cam{n}_bg.png and the caller should read that.
    Returning None rather than raising keeps the caller's fallback a
    normal branch instead of exception handling."""
    entry = (video_meta.get("cameras") or {}).get(str(cam))
    if not entry or entry.get("bg_index") is None:
        return None
    return read_clip_frame(Path(package_dir) / entry["clip"], int(entry["bg_index"]))


def read_bg_and_commit_frames(
    package_dir: Path, video_meta: dict, cam: int,
) -> "tuple[np.ndarray | None, np.ndarray]":
    """Both of `cam`'s pointer frames in ONE read of its clip.

    read_bg_frame() + read_commit_frame() open the file twice for what is
    always the same file, and every load of a package asks for both --
    which is the hot path for replay and rescore over the whole corpus,
    not a rare one. Only the two pointer frames of an MJPEG clip are
    decoded (see _read_clip_frames_at()): an 8-frame window clip used to
    cost 8 decodes a camera here for the 2 it returns. The bg is None on
    a package whose clip has no bg pointer (throw-clip/v1, or a window
    clip that could not de-duplicate it); the caller reads cam{N}_bg.png
    in that case. Raises KeyError if this camera has no clip at all."""
    entry = video_meta["cameras"][str(cam)]
    commit_index = int(entry["commit_index"])
    bg_index = entry.get("bg_index")
    if bg_index is not None:
        bg_index = int(bg_index)
    n, frames = _read_clip_frames_at(
        Path(package_dir) / entry["clip"],
        (commit_index,) if bg_index is None else (commit_index, bg_index))
    if not 0 <= commit_index < n:
        raise IndexError(
            f"cam{cam}: commit_index {commit_index} out of range for a "
            f"{n}-frame clip"
        )
    bg = None
    if bg_index is not None:
        if not 0 <= bg_index < n:
            raise IndexError(
                f"cam{cam}: bg_index {bg_index} out of range for a "
                f"{n}-frame clip"
            )
        bg = frames[bg_index]
    return bg, frames[commit_index]
