"""opendarts/capture/clip.py -- per-camera MKV clips for a throw package.

EVERY throw package is JSON + one MKV per camera (2026-09-22). There are
no frame PNGs any more, in either shape:

  * a STILLS clip (kind "stills") holds exactly two frames -- the bg
    reference and the scored commit frame -- and is written
    SYNCHRONOUSLY by opendarts.capture.throw_package.save_throw_package()
    from the two arrays it is handed. It needs no frame ring, so it
    cannot age out and cannot be late: the package is complete the
    instant it exists. This is the floor every package gets.

  * a WINDOW clip (kind "window") holds the whole bg..commit+1 run out
    of the frame ring, and REPLACES the stills clip a moment later when
    the config's video-record mode asked for a recording. It is an
    upgrade on top of a package that was already complete, which is what
    lets finalize_throw_clip() fail harmlessly.

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
scoring are replayed from), so the WRITER's caller verifies the round
trip before trusting a clip -- see verify_clip_frame().

Why per-camera files rather than one 3-stream MKV: the viewer plays one
camera at a time and replay reads one camera's frame by index; separate
files keep both trivial and let a single camera's clip fail without
taking the others.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
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


def read_clip_frame(path: Path, index: int) -> np.ndarray:
    """One frame by index (the baseline/commit pointer read). Decodes the
    whole clip -- these are ~13-frame windows, so there is nothing to
    optimise and random-access into MJPEG/FFV1 is not worth the code."""
    frames = read_clip_frames(path)
    if not 0 <= index < len(frames):
        raise IndexError(
            f"read_clip_frame: index {index} out of range for {path} "
            f"({len(frames)} frames)"
        )
    return frames[index]


def verify_clip_frame(path: Path, index: int, expected: np.ndarray) -> bool:
    """True iff frame `index` of `path` is byte-identical to `expected`
    (the array that was scored). The write-time tripwire: a clip whose
    commit frame is not exactly what was scored is not a faithful record,
    and the caller refuses it rather than keep a lie."""
    got = read_clip_frame(path, index)
    return got.shape == expected.shape and np.array_equal(
        got, np.ascontiguousarray(expected, dtype=np.uint8)
    )


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
#: Per-camera STILLS clip filename. A different name from a recording on
#: purpose: the upgrade from stills to window then never has to write
#: over the file meta.json is pointing at (see finalize_throw_clip), and
#: a directory listing tells the two apart without opening meta.json.
#: Readers never build either name themselves -- they follow the
#: pointer's `clip`.
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
    ring, as opposed to the two-frame stills clip every package carries.

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
#: (2026-09-22). Both ends are found by byte-identity against arrays we
#: already hold, so the window is defined by the frames themselves rather
#: than inferred from a clock:
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
#: CLIP_MAX_FRAMES_BEFORE_COMMIT stays as a ceiling, not a target: if the
#: board never settled, the last idle adoption can be far back, and a
#: pathological throw must not write a giant clip. A bg beyond it is the
#: one case that still keeps its PNG.
CLIP_FRAMES_AFTER_COMMIT = 1
CLIP_MAX_FRAMES_BEFORE_COMMIT = 12


class BgFrameNotInClip(RuntimeError):
    """The bg reference is not byte-identical to any frame in the ring
    slice, so the clip cannot stand in for it. Not fatal: the caller keeps
    writing the bg PNG instead. See finalize_throw_clip()."""


class CommitFrameNotInClip(RuntimeError):
    """The scored commit frame is not byte-identical to any frame in the
    camera's clip window. Either the window did not cover the commit or a
    frame was mutated between capture and write -- both make the clip an
    unfaithful record, so the write is refused rather than kept."""


def _still_clip_holds(path: Path, bg: np.ndarray, commit: np.ndarray) -> bool:
    """True iff `path` is a readable 2-frame clip whose frames are
    byte-identical to `bg` then `commit`.

    One decode for both frames, rather than two verify_clip_frame() calls
    that would each decode the whole file -- and it also catches a clip
    that came back with the wrong NUMBER of frames, which a per-index
    check cannot see."""
    try:
        frames = read_clip_frames(path)
    except Exception:  # noqa: BLE001 -- an unreadable clip is a failed verify
        return False
    if len(frames) != 2:
        return False
    return all(
        got.shape == want.shape and np.array_equal(got, want)
        for got, want in zip(frames, (bg, commit))
    )


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
    meta. The synchronous floor every package gets; see the module
    docstring.

    Needs no frame ring and no timing: both frames are passed in, so this
    cannot age out, cannot be late, and cannot half-succeed. That is the
    whole reason it exists. Before this, a package's only frames were the
    ring-sliced clip written ~0.5s later on a background thread, and the
    PNGs were what covered the gap; with the PNGs gone, something has to
    put real frames on disk at save time, and it has to be something that
    cannot fail for a reason outside its own control.

    `bg_jpegs`/`commit_jpegs` are the camera's OWN JPEG bytes for those
    exact frames, when the rig has them (every local camera --
    opendarts.live.local_capture's `grab_with_jpeg`). Given both for a
    camera, the clip is a stream copy of those bytes: no re-encode, and
    ~6x smaller than the lossless alternative (measured on a real
    3-camera package: 465 KB of mjpeg against 2,973 KB of FFV1 for the
    same six frames). Absent -- any frame we could not pair with its
    bytes at the same tick -- the decoded array is FFV1-encoded instead,
    which is bigger and equally exact.

    A MISPAIRED JPEG (bytes from a different frame than the array) would
    be the worst possible outcome: a package that quietly stores a frame
    nothing ever scored, breaking SCORE==STORE invisibly. So the mjpeg
    clip is read straight back and compared against BOTH arrays before it
    is accepted, and anything short of byte-identical falls back to FFV1
    -- a bigger file, never a wrong one. The FFV1 fallback is verified the
    same way, and a failure THERE raises: there is nothing left to fall
    back to, and a package with no frames must not be written at all.

    Returns the same pointer shape write_throw_clips() does, with
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
            if _still_clip_holds(clip_path, bg, commit):
                encoding = "mjpeg"
            else:
                # Loud, because this is not a storage problem -- it means
                # the bytes and the array came from different frames, i.e.
                # a real pairing bug upstream. The package is fine (FFV1
                # below), but somebody needs to know.
                log.warning(
                    "cam%d: the camera JPEGs supplied for this throw do NOT decode "
                    "to the frames that were scored -- falling back to FFV1. The "
                    "package is unaffected, but the JPEG pairing upstream is wrong.",
                    cam,
                )
        if encoding is None:
            write_clip_ffv1(clip_path, [bg, commit])
            if not _still_clip_holds(clip_path, bg, commit):
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


def write_throw_clips(
    package_dir: Path,
    sets: list,
    commit_frames_by_cam: "dict[int, np.ndarray]",
    *,
    bg_frames_by_cam: "dict[int, np.ndarray] | None" = None,
    frames_after_commit: int = CLIP_FRAMES_AFTER_COMMIT,
    max_frames_before_commit: int = CLIP_MAX_FRAMES_BEFORE_COMMIT,
) -> "dict":
    """Write one per-camera MKV clip of the detection/settle window into
    `package_dir`, and return the pointer block for the package's meta.

    `sets` is a frame-ring slice's `.sets` (a list of FrameSet); each has
    `pixels` (slot -> BGR array) and `jpegs` (slot -> camera JPEG bytes),
    one or the other per slot. `commit_frames_by_cam` is the exact array
    the engine scored for each camera (``trigger.last_frame`` /
    ``dart_frames_bgr``).

    Per camera, the frames present across the window are collected in set
    order:
      * every frame a JPEG (every local camera) -> the JPEGs are
        stream-copied into the clip (the exact bytes that were scored);
      * otherwise -> the decoded BGR frames are FFV1-encoded (a slot the
        pump happened to hand over as pixels).

    The commit frame's index in the window is found by byte-identity to
    ``commit_frames_by_cam[cam]`` and then VERIFIED against the written
    clip. A camera whose commit frame is not byte-identical in its clip
    raises ``CommitFrameNotInClip`` -- the clip is not kept as a lie.

    Returns::

        {"schema": THROW_CLIP_SCHEMA, "container": "mkv", "fps": <int>,
         "kind": "window",
         "cameras": {"0": {"clip": "clip_cam0.mkv", "encoding": "mjpeg"|"ffv1",
                           "n_frames": N, "commit_index": i}, ...}}
    """
    package_dir = Path(package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    cams = sorted(c for c in commit_frames_by_cam
                  if any(c in fs.pixels or c in fs.jpegs for fs in sets))
    cameras_out: dict[str, dict] = {}
    decode_cache: dict[int, np.ndarray] = {}

    for cam in cams:
        commit = np.ascontiguousarray(commit_frames_by_cam[cam], dtype=np.uint8)
        # Collect this camera's frames across the window, in order, noting
        # whether the slot was carried as a JPEG in every set it appeared.
        jpegs: list[bytes] = []
        decoded: list[np.ndarray] = []
        walls: list[float | None] = []  # each frame's ring wall_s, if known
        all_jpeg = True
        for fs in sets:
            if cam in fs.jpegs:
                data = fs.jpegs[cam]
                jpegs.append(data)
                key = id(data)
                arr = decode_cache.get(key)
                if arr is None:
                    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    decode_cache[key] = arr
                decoded.append(arr)
                walls.append(getattr(fs, "wall_s", None))
            elif cam in fs.pixels:
                all_jpeg = False
                decoded.append(fs.pixels[cam])
                walls.append(getattr(fs, "wall_s", None))
            # slot absent this cycle -> no frame; skip (keeps the window
            # a contiguous run of frames this camera actually produced).
        if not decoded:
            continue

        # commit_index: byte-identity to the scored array (it is the same
        # object as, or decodes from, one of these frames).
        commit_index = next(
            (i for i, f in enumerate(decoded)
             if f.shape == commit.shape and np.array_equal(
                 np.ascontiguousarray(f, dtype=np.uint8), commit)),
            None,
        )
        if commit_index is None:
            raise CommitFrameNotInClip(
                f"cam{cam}: the scored commit frame is not byte-identical to any "
                f"of the {len(decoded)} frame(s) in the clip window"
            )

        # THE START IS THE BG FRAME. Found the same way the commit frame
        # is -- byte-identity against an array we already hold -- so the
        # window is bounded by two real frames rather than by a clock. The
        # bg is the lifecycle's last idle adoption, i.e. the last frame
        # before the dart was detected, which is precisely the start the
        # old time-derived window was trying to estimate.
        #
        # Not found (bg older than the ring slice, or a pathological
        # unsettled stretch) -> raise, and the caller keeps the bg PNG. The
        # clip is still written; only the de-duplication is given up.
        bg_index = None
        if bg_frames_by_cam is not None and cam in bg_frames_by_cam:
            bg = np.ascontiguousarray(bg_frames_by_cam[cam], dtype=np.uint8)
            bg_index = next(
                (i for i, f in enumerate(decoded)
                 if f.shape == bg.shape and np.array_equal(
                     np.ascontiguousarray(f, dtype=np.uint8), bg)),
                None,
            )
            if bg_index is None or bg_index > commit_index:
                raise BgFrameNotInClip(
                    f"cam{cam}: the bg reference is not byte-identical to any "
                    f"frame at or before the commit in this {len(decoded)}-frame "
                    f"ring slice"
                )

        # bg .. commit (+ after). With no bg supplied, fall back to the cap
        # so a clip is still bounded.
        lo = bg_index if bg_index is not None else max(
            0, commit_index - max_frames_before_commit)
        # The cap is a ceiling, never a target: a bg further back than this
        # would write a giant clip, so refuse the de-duplication rather than
        # silently truncating past the frame we promised to include.
        if bg_index is not None and commit_index - bg_index > max_frames_before_commit:
            raise BgFrameNotInClip(
                f"cam{cam}: bg sits {commit_index - bg_index} frames before the "
                f"commit, beyond the {max_frames_before_commit}-frame ceiling"
            )
        hi = min(len(decoded), commit_index + frames_after_commit + 1)
        decoded = decoded[lo:hi]
        if all_jpeg:
            jpegs = jpegs[lo:hi]
        commit_index -= lo
        if bg_index is not None:
            bg_index -= lo

        clip_name = CLIP_FILENAME_TEMPLATE.format(cam=cam)
        clip_path = package_dir / clip_name
        if all_jpeg:
            h, w = commit.shape[:2]
            write_clip_mjpeg(clip_path, jpegs, width=w, height=h)
            encoding = "mjpeg"
        else:
            write_clip_ffv1(clip_path, decoded)
            encoding = "ffv1"

        # The write-time tripwire (see verify_clip_frame): the clip's
        # commit frame must be exactly what was scored, or refuse it.
        if not verify_clip_frame(clip_path, commit_index, commit):
            clip_path.unlink(missing_ok=True)
            raise CommitFrameNotInClip(
                f"cam{cam}: {clip_name} commit frame (index {commit_index}) is NOT "
                "byte-identical to the scored frame after write -- refusing to keep it"
            )
        # The bg gets the SAME tripwire, for the same reason and a sharper
        # one: the package is about to stop storing cam{n}_bg.png, so this
        # clip becomes the only copy of a SCORING INPUT. Verified after the
        # write, against the real file, exactly like the commit frame --
        # never trusted from the in-memory index alone.
        if bg_index is not None and not verify_clip_frame(
                clip_path, bg_index, np.ascontiguousarray(
                    bg_frames_by_cam[cam], dtype=np.uint8)):
            clip_path.unlink(missing_ok=True)
            raise BgFrameNotInClip(
                f"cam{cam}: {clip_name} bg frame (index {bg_index}) is NOT "
                "byte-identical to the reference after write -- refusing to drop "
                "the bg PNG"
            )
        cameras_out[str(cam)] = {
            "clip": clip_name,
            "encoding": encoding,
            "n_frames": len(decoded),
            "commit_index": commit_index,
            **({"bg_index": bg_index} if bg_index is not None else {}),
        }

    return {
        "schema": THROW_CLIP_SCHEMA,
        "container": "mkv",
        "fps": CLIP_CONTAINER_FPS,
        "kind": CLIP_KIND_WINDOW,
        "cameras": cameras_out,
    }





#: Where an upgrade's clips are built before they replace the ones the
#: package already has. Inside the package, so moving a finished clip
#: into place is a rename on the same filesystem rather than a copy, and
#: dot-prefixed so nothing that scans a package for its frames can ever
#: see a half-written window clip as the package's own.
_UPGRADE_STAGING_DIRNAME = ".clip-upgrade"

#: The name an upgraded (window) clip takes in the one case the plain
#: CLIP_FILENAME_TEMPLATE name is already the package's live clip -- a
#: package that is ALREADY a recording being upgraded again. The two
#: names alternate, so an upgrade never overwrites the file meta.json
#: currently points at; see finalize_throw_clip() for why that matters.
CLIP_ALT_FILENAME_TEMPLATE = "clip_cam{cam}_rec.mkv"


def _unused_clip_name(cam: int, current: "str | None") -> str:
    """The clip filename for `cam` that is NOT `current`."""
    primary = CLIP_FILENAME_TEMPLATE.format(cam=cam)
    return CLIP_ALT_FILENAME_TEMPLATE.format(cam=cam) if current == primary else primary


def _write_meta_atomically(meta_path: Path, meta: dict) -> None:
    """meta.json via a temp file and os.replace, so a reader (or a crash)
    sees the old file or the new one and never a half-written one. It is
    the commit point of a clip upgrade, which is what makes it worth the
    extra file."""
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, meta_path)


def _package_frames_for_upgrade(
    package_dir: Path, video_meta: "dict | None",
) -> "tuple[dict[int, np.ndarray], dict[int, np.ndarray], str, str | None]":
    """The bg and commit frames an upgrade must reproduce, read back out
    of the package itself.

    Two sources, in order: the clips the package already has (every
    package written since 2026-09-22 -- normally the two-frame stills
    clip), else the cam*_bg.png / cam*_frame.png of a package from the
    existing corpus. Both give the byte-exact arrays that were scored,
    which is what lets the new window clip be MATCHED against them.

    Returns ``(commit, bg, source, reason)`` -- `source` is "clip" or
    "png", and `reason` is non-None when nothing usable was found."""
    commit: dict[int, np.ndarray] = {}
    bg: dict[int, np.ndarray] = {}

    if video_meta:
        for key in sorted((video_meta.get("cameras") or {})):
            try:
                cam = int(key)
            except ValueError:
                continue
            try:
                bg_arr, commit_arr = read_bg_and_commit_frames(
                    package_dir, video_meta, cam)
            except Exception as exc:  # noqa: BLE001 -- an unreadable clip is a refusal
                return {}, {}, "clip", f"cam{cam}: its existing clip would not read back ({exc})"
            commit[cam] = commit_arr
            if bg_arr is not None:
                bg[cam] = bg_arr
        if not commit:
            return {}, {}, "clip", "the package's video block names no camera to base a clip on"
        return commit, bg, "clip", None

    for png in sorted(package_dir.glob("cam*_frame.png")):
        try:
            cam = int(png.name[len("cam"):-len("_frame.png")])
        except ValueError:
            continue
        arr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if arr is None:
            return {}, {}, "png", f"cam{cam}: {png.name} could not be decoded"
        commit[cam] = arr
    if not commit:
        return {}, {}, "png", "no dart PNGs in the package to base a clip on"
    for png in sorted(package_dir.glob("cam*_bg.png")):
        try:
            cam = int(png.name[len("cam"):-len("_bg.png")])
        except ValueError:
            continue
        arr = cv2.imread(str(png), cv2.IMREAD_COLOR)
        if arr is not None:
            bg[cam] = arr
    return commit, bg, "png", None


def finalize_throw_clip(package_dir: Path, sets: list) -> "dict":
    """UPGRADE an already-complete throw package to a RECORDED one: write
    per-camera window clips from `sets` (the whole bg..commit+1 run out of
    the frame ring) over whatever clips the package already has, and point
    meta.json's `video` block at them.

    This is now strictly an upgrade, never a rescue. The package it is
    handed already holds every frame it needs -- the two-frame stills clip
    save_throw_package() wrote synchronously, or, for a package from the
    existing corpus, its bg/dart PNGs -- so the frames to match the ring
    slice against are read back out of the package itself, and a failure
    here costs the package nothing at all.

    THE CLIPS ARE REPLACED, NOT DELETED-THEN-WRITTEN. The new clips are
    built in a staging directory and moved into place only once every
    camera has a verified one. That ordering is the whole safety property:
    a package's frames now live ONLY in its clips, so a window clip that
    failed halfway after unlinking the stills clip would have destroyed
    the evidence it was trying to improve on.

    For the same reason the bg is not optional here the way it used to be.
    When the package's frames came from its clips, a bg that cannot be
    found in the ring slice ABANDONS the upgrade -- writing a
    commit-anchored clip with no bg pointer would drop a scoring input
    that has no other copy. A corpus package with real bg PNGs keeps the
    old behaviour: write the clip anyway and leave the PNGs alone.

    Returns ``{"ok": bool, "video"?: dict, "reason"?: str,
    "cameras": [...], "bg_in_clip"?: bool}``."""
    package_dir = Path(package_dir)
    meta_path = package_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    commit, bg, source, reason = _package_frames_for_upgrade(
        package_dir, meta.get("video"))
    if reason is not None:
        return {"ok": False, "reason": reason}

    staging = package_dir / _UPGRADE_STAGING_DIRNAME

    def _cleanup() -> None:
        shutil.rmtree(staging, ignore_errors=True)

    # Only offer the bg for cameras that also have a commit frame, so a
    # stray PNG cannot make write_throw_clips look for a bg in a clip it
    # is not writing.
    bg_for_clip = {c: a for c, a in bg.items() if c in commit} or None

    _cleanup()  # a staging dir left by a crashed earlier attempt
    try:
        video = write_throw_clips(
            staging, sets, commit,
            bg_frames_by_cam=bg_for_clip,
            frames_after_commit=CLIP_FRAMES_AFTER_COMMIT,
            max_frames_before_commit=CLIP_MAX_FRAMES_BEFORE_COMMIT,
        )
        bg_deduped = bg_for_clip is not None
    except BgFrameNotInClip as exc:
        _cleanup()
        if source == "clip":
            # The bg exists ONLY inside the clip this upgrade would
            # overwrite. A commit-anchored replacement would silently lose
            # it, so the upgrade is abandoned and the package keeps the
            # clip it already had -- complete, if less interesting.
            return {"ok": False, "reason":
                    f"{exc} -- the bg has no other copy, so the existing clip is kept"}
        # A corpus package still has its bg PNG, so losing the
        # de-duplication costs nothing but disk: write the clip anyway,
        # anchored on the commit, and keep the PNGs.
        try:
            video = write_throw_clips(
                staging, sets, commit,
                frames_after_commit=CLIP_FRAMES_AFTER_COMMIT,
                max_frames_before_commit=CLIP_MAX_FRAMES_BEFORE_COMMIT,
            )
            bg_deduped = False
        except CommitFrameNotInClip as exc2:
            _cleanup()
            return {"ok": False, "reason": str(exc2)}
    except CommitFrameNotInClip as exc:
        _cleanup()
        return {"ok": False, "reason": str(exc)}

    if set(map(str, commit)) - set(video["cameras"]):
        # A camera had a frame but no clip (absent from the window): a
        # half-recorded package is worse than an un-recorded one, and here
        # it would also mean losing that camera's frames entirely.
        _cleanup()
        return {"ok": False,
                "reason": "not every camera with a dart frame produced a clip"}

    # Every camera verified. Now make the switch, in an order where a
    # crash at ANY point leaves meta.json pointing at clips that exist and
    # match it:
    #
    #   1. move each window clip into the package under a name the
    #      package is NOT currently using -- the stills clips stay exactly
    #      where meta.json says they are;
    #   2. rewrite meta.json atomically (temp file + os.replace) -- the
    #      single commit point: before it the package is the stills
    #      package, after it the recorded one;
    #   3. only then remove the clips nothing points at any more.
    #
    # Renaming the window clips straight over the stills clips (same
    # name) looks simpler and is not safe: a crash between the rename of
    # cam0 and the meta.json write would leave cam0's file a 10-frame
    # window while its pointer still says "commit_index 1 of 2" -- a
    # package that reads back a DIFFERENT frame than was scored, with
    # nothing to flag it. A crash in this ordering costs at most an
    # orphaned file.
    old_video = meta.get("video") or {}
    old_names = {
        str(c): e.get("clip")
        for c, e in (old_video.get("cameras") or {}).items() if isinstance(e, dict)
    }
    for cam in commit:
        entry = video["cameras"][str(cam)]
        final = _unused_clip_name(cam, old_names.get(str(cam)))
        (staging / entry["clip"]).replace(package_dir / final)
        entry["clip"] = final
    _cleanup()

    # Re-read right before writing: other writers annotate meta.json
    # after the save (the oracle's ground truth, operator corrections),
    # and this upgrade runs a moment later on its own thread. Taking
    # their latest version narrows the read-modify-write window to this
    # line, rather than to the whole clip encode above.
    meta = json.loads(meta_path.read_text())
    meta["video"] = video
    _write_meta_atomically(meta_path, meta)

    new_names = {e["clip"] for e in video["cameras"].values()}
    for name in old_names.values():
        if name and name not in new_names:
            (package_dir / name).unlink(missing_ok=True)

    for cam in commit:
        (package_dir / f"cam{cam}_frame.png").unlink(missing_ok=True)
    # Drop the bg PNGs too -- but ONLY when every camera's bg was verified
    # byte-identical inside its own clip (write_throw_clips raises
    # otherwise, and the retry above then leaves bg_deduped False). Same
    # all-or-nothing discipline as the dart PNGs: a package must never end
    # up with some cameras' bg in the clip and others' on disk, because a
    # reader would have to guess per camera.
    bg_ok = bg_deduped and all(
        video["cameras"].get(str(cam), {}).get("bg_index") is not None
        for cam in commit
    )
    if bg_ok:
        for cam in commit:
            (package_dir / f"cam{cam}_bg.png").unlink(missing_ok=True)
    return {"ok": True, "video": video, "cameras": sorted(commit),
            "bg_in_clip": bg_ok}


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
    """Both of `cam`'s pointer frames in ONE decode of its clip.

    read_bg_frame() + read_commit_frame() decode the whole clip twice for
    what is always the same file, and every load of a package asks for
    both -- which is the hot path for replay and rescore over the whole
    corpus, not a rare one. The bg is None on a package whose clip has no
    bg pointer (throw-clip/v1, or a window clip that could not
    de-duplicate it); the caller reads cam{N}_bg.png in that case.
    Raises KeyError if this camera has no clip at all."""
    entry = video_meta["cameras"][str(cam)]
    frames = read_clip_frames(Path(package_dir) / entry["clip"])
    commit_index = int(entry["commit_index"])
    if not 0 <= commit_index < len(frames):
        raise IndexError(
            f"cam{cam}: commit_index {commit_index} out of range for a "
            f"{len(frames)}-frame clip"
        )
    bg_index = entry.get("bg_index")
    bg = None
    if bg_index is not None:
        bg_index = int(bg_index)
        if not 0 <= bg_index < len(frames):
            raise IndexError(
                f"cam{cam}: bg_index {bg_index} out of range for a "
                f"{len(frames)}-frame clip"
            )
        bg = frames[bg_index]
    return bg, frames[commit_index]
