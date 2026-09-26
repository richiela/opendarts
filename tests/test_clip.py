"""opendarts/capture/clip.py -- per-camera MKV clips, byte-exact both ways.

The one property that matters: a frame read back out of the clip is
bit-identical to what was written (and therefore to what was scored), for
both the MJPEG stream-copy path (Linux/Windows) and the FFV1 path (macOS).
Synthetic frames, deterministic, no corpus needed.
"""
from __future__ import annotations

import cv2
import numpy as np

from opendarts.capture import clip


def _frames(n=13, w=320, h=180, seed=0):
    rng = np.random.default_rng(seed)
    # gradient + noise: a non-trivial image JPEG actually has to work on
    base = (np.add.outer(np.arange(h), np.arange(w)) % 256).astype(np.uint8)
    base = np.repeat(base[:, :, None], 3, axis=2)
    return [np.clip(base + rng.integers(0, 30, (h, w, 3)), 0, 255).astype(np.uint8)
            for _ in range(n)]


def test_mjpeg_clip_round_trip_is_byte_exact_to_scored(tmp_path):
    """Linux/Windows: the ring holds the camera's JPEG. We store those
    bytes and decode them back with the SAME cv2 decoder scoring used --
    so the read frame equals the scored frame (decode of that JPEG)."""
    frames = _frames()
    jpegs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
             for f in frames]
    scored = [cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) for j in jpegs]

    out = tmp_path / "cam0.mkv"
    clip.write_clip_mjpeg(out, jpegs, width=320, height=180)
    back = clip.read_clip_frames(out)

    assert len(back) == len(frames)
    for i, (a, b) in enumerate(zip(scored, back)):
        assert np.array_equal(a, b), f"mjpeg frame {i} not byte-identical to scored"


def test_ffv1_clip_round_trip_is_byte_exact(tmp_path):
    """macOS: the ring holds decoded BGR. FFV1 (bgr0) is lossless, so the
    read frame equals the written pixel array exactly."""
    frames = _frames(seed=1)
    out = tmp_path / "cam0.mkv"
    clip.write_clip_ffv1(out, frames)
    back = clip.read_clip_frames(out)

    assert len(back) == len(frames)
    for i, (a, b) in enumerate(zip(frames, back)):
        assert np.array_equal(a, b), f"ffv1 frame {i} not byte-identical"


def test_verify_clip_frame_tripwire(tmp_path):
    frames = _frames(seed=2)
    out = tmp_path / "cam0.mkv"
    clip.write_clip_ffv1(out, frames)
    # the real commit frame verifies
    assert clip.verify_clip_frame(out, 5, frames[5])
    # a mutated frame does not -- the tripwire fires
    wrong = frames[5].copy()
    wrong[0, 0, 0] = wrong[0, 0, 0] ^ 0xFF
    assert not clip.verify_clip_frame(out, 5, wrong)


def test_read_clip_frame_out_of_range(tmp_path):
    out = tmp_path / "cam0.mkv"
    clip.write_clip_ffv1(out, _frames(n=4))
    import pytest
    with pytest.raises(IndexError):
        clip.read_clip_frame(out, 99)


def test_empty_frames_raise(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        clip.write_clip_ffv1(tmp_path / "x.mkv", [])
    with pytest.raises(ValueError):
        clip.write_clip_mjpeg(tmp_path / "y.mkv", [], 320, 180)


# --- write_window_clips: per-camera clips taken from the ring by generation ---

import types
import numpy as _np
import cv2 as _cv2
import pytest as _pytest
from opendarts.capture import clip as _clip


def _fs(i, pixels=None, jpegs=None):
    """A ring set at generation `i`, one pump period after set i-1."""
    return types.SimpleNamespace(generation=i, wall_s=1000.0 + i / 30.0,
                                 pixels=pixels or {}, jpegs=jpegs or {})


def _scored(bg, commit, bg_gen, commit_gen, **jpegs):
    return _clip.ScoredFrames(bg=bg, commit=commit,
                              bg_generations={c: bg_gen for c in bg},
                              commit_generations={c: commit_gen for c in commit},
                              **jpegs)


def test_write_window_clips_mjpeg_path(tmp_path):
    """Linux/Win: every slot a JPEG -> MJPEG clips, commit frame recovered
    byte-identical to the scored (decoded) frame."""
    fr = _frames(n=9)  # 9 sets
    cams = (0, 1, 2)
    # each camera's per-set JPEG, and the scored commit = decode of set 4
    jpegs_by_cam = {c: [_cv2.imencode(".jpg", f, [_cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                        for f in fr] for c in cams}
    sets = [_fs(i, jpegs={c: jpegs_by_cam[c][i] for c in cams}) for i in range(len(fr))]
    dec = lambda c, i: _cv2.imdecode(_np.frombuffer(jpegs_by_cam[c][i], _np.uint8), _cv2.IMREAD_COLOR)
    commit = {c: dec(c, 4) for c in cams}
    bg = {c: dec(c, 1) for c in cams}

    ptr = _clip.write_window_clips(
        tmp_path, sets,
        _scored(bg, commit, 1, 4,
                bg_jpegs={c: jpegs_by_cam[c][1] for c in cams},
                commit_jpegs={c: jpegs_by_cam[c][4] for c in cams}),
        cams)
    assert ptr["schema"] == _clip.THROW_CLIP_SCHEMA
    for c in cams:
        e = ptr["cameras"][str(c)]
        assert e["encoding"] == "mjpeg" and e["commit_index"] == 3 and e["bg_index"] == 0
        # bounded both ends: the bg, and one frame after the commit
        assert e["n_frames"] == 3 + 1 + _clip.CLIP_FRAMES_AFTER_COMMIT
        assert (tmp_path / e["clip"]).exists()
        back = _clip.read_commit_frame(tmp_path, ptr, c)
        assert _np.array_equal(back, commit[c])


def test_write_window_clips_ffv1_path(tmp_path):
    """Pixel slots -> FFV1 clips, commit byte-identical."""
    fr = _frames(n=7, seed=3)
    cams = (0, 1)
    sets = [_fs(i, pixels={c: fr[i] for c in cams}) for i in range(len(fr))]
    ptr = _clip.write_window_clips(
        tmp_path, sets, _scored({c: fr[0] for c in cams}, {c: fr[3] for c in cams}, 0, 3), cams)
    for c in cams:
        e = ptr["cameras"][str(c)]
        assert e["encoding"] == "ffv1" and e["commit_index"] == 3
        assert _np.array_equal(_clip.read_commit_frame(tmp_path, ptr, c), fr[3])


def test_write_window_clips_refuses_a_commit_not_in_the_ring(tmp_path):
    fr = _frames(n=4, seed=5)
    sets = [_fs(i, pixels={0: fr[i]}) for i in range(len(fr))]
    with _pytest.raises(_clip.ClipWindowUnavailable):
        _clip.write_window_clips(tmp_path, sets, _scored({0: fr[0]}, {0: fr[3]}, 0, 7), [0])
    assert not list(tmp_path.glob("*.mkv"))


def test_clip_spans_bg_through_commit_plus_one(tmp_path):
    """THE WINDOW IS TWO REAL FRAMES, NOT A CLOCK.

    The bg is the lifecycle's last idle adoption -- the last frame before
    the dart was detected -- so it defines the clip's start exactly, and
    the scored frame defines the end. Everything before the bg is dropped;
    one frame is kept after the commit.
    """
    fr = _frames(n=25, seed=7)
    sets = [_fs(i, pixels={0: fr[i]}) for i in range(len(fr))]
    bg_pos, commit_pos = 12, 17
    ptr = _clip.write_window_clips(
        tmp_path, sets, _scored({0: fr[bg_pos]}, {0: fr[commit_pos]}, bg_pos, commit_pos), [0])
    e = ptr["cameras"]["0"]
    assert e["bg_index"] == 0, "the bg IS the first frame"
    assert e["commit_index"] == commit_pos - bg_pos
    assert e["n_frames"] == (commit_pos - bg_pos) + 1 + _clip.CLIP_FRAMES_AFTER_COMMIT
    back = _clip.read_clip_frames(tmp_path / e["clip"])
    assert _np.array_equal(back[0], fr[bg_pos])                  # starts at bg
    assert _np.array_equal(back[e["commit_index"]], fr[commit_pos])
    assert _np.array_equal(back[-1], fr[commit_pos + 1])         # one after


def test_a_camera_absent_from_a_set_is_skipped_not_padded(tmp_path):
    """A camera's run is the sets that HOLD a frame for it -- a cycle it
    produced nothing in has no frame to write."""
    fr = _frames(n=9, seed=13)
    sets = [_fs(i, pixels={0: fr[i]} if i != 3 else {}) for i in range(len(fr))]
    ptr = _clip.write_window_clips(tmp_path, sets, _scored({0: fr[1]}, {0: fr[5]}, 1, 5), [0])
    e = ptr["cameras"]["0"]
    assert (e["commit_index"], e["n_frames"]) == (3, 5)   # gens 1, 2, 4, 5, 6


def test_bg_and_commit_are_both_readable_back_byte_exact(tmp_path):
    """Both ends are pointers, and both must survive the round trip -- the
    clip is the ONLY copy of a scoring input (and of what recalibrate.py
    re-solves a session's calibration from)."""
    fr = _frames(n=20, seed=8)
    sets = [_fs(i, pixels={0: fr[i]}) for i in range(len(fr))]
    ptr = _clip.write_window_clips(tmp_path, sets, _scored({0: fr[9]}, {0: fr[14]}, 9, 14), [0])
    assert _np.array_equal(_clip.read_bg_frame(tmp_path, ptr, 0), fr[9])
    assert _np.array_equal(_clip.read_commit_frame(tmp_path, ptr, 0), fr[14])
    assert ptr["schema"] == _clip.THROW_CLIP_SCHEMA


def test_bg_not_in_the_ring_is_refused_not_faked(tmp_path):
    """A bg the ring no longer holds must raise, so the caller writes the
    stills clip. Silently starting the clip somewhere else would leave the
    package pointing at a frame that is not the reference."""
    fr = _frames(n=10, seed=9)
    sets = [_fs(i, pixels={0: fr[i]}) for i in range(2, len(fr))]
    with _pytest.raises(_clip.ClipWindowUnavailable, match="bg"):
        _clip.write_window_clips(tmp_path, sets, _scored({0: fr[1]}, {0: fr[6]}, 1, 6), [0])


def test_bg_beyond_the_ceiling_is_refused(tmp_path):
    """The cap is a ceiling, not a target: a board that never settled can
    leave the last idle adoption far back, and that must not write a giant
    clip. Refuse the window rather than truncate past the frame we
    promised to include."""
    n = _clip.CLIP_MAX_FRAMES_BEFORE_COMMIT + 6
    fr = _frames(n=n + 4, seed=10)
    sets = [_fs(i, pixels={0: fr[i]}) for i in range(len(fr))]
    with _pytest.raises(_clip.ClipWindowUnavailable, match="ceiling"):
        _clip.write_window_clips(tmp_path, sets, _scored({0: fr[0]}, {0: fr[n]}, 0, n), [0])


def test_a_pointer_without_bg_index_reads_no_bg(tmp_path):
    """A throw-clip/v1 pointer (or a pre-2026-09-26 window that could not
    de-duplicate its bg) has no bg_index: the reader gets None and falls
    back to the package's cam{N}_bg.png."""
    fr = _frames(n=5, seed=12)
    _clip.write_clip_ffv1(tmp_path / "clip_cam0.mkv", fr)
    ptr = {"cameras": {"0": {"clip": "clip_cam0.mkv", "commit_index": 2}}}
    assert _clip.read_bg_frame(tmp_path, ptr, 0) is None
    assert _np.array_equal(_clip.read_commit_frame(tmp_path, ptr, 0), fr[2])
