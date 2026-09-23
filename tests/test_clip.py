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


# --- write_throw_clips: per-camera clips + verified commit pointer ---------

import types
import numpy as _np
import cv2 as _cv2
import pytest as _pytest
from opendarts.capture import clip as _clip


def _fs(pixels=None, jpegs=None, wall_s=None):
    return types.SimpleNamespace(pixels=pixels or {}, jpegs=jpegs or {}, wall_s=wall_s)


def test_write_throw_clips_mjpeg_path(tmp_path):
    """Linux/Win: every slot a JPEG -> MJPEG clips, commit frame recovered
    byte-identical to the scored (decoded) frame."""
    fr = _frames(n=9)  # 9 sets
    cams = (0, 1, 2)
    # each camera's per-set JPEG, and the scored commit = decode of set 4
    jpegs_by_cam = {c: [_cv2.imencode(".jpg", f, [_cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
                        for f in fr] for c in cams}
    sets = [_fs(jpegs={c: jpegs_by_cam[c][i] for c in cams}) for i in range(len(fr))]
    commit = {c: _cv2.imdecode(_np.frombuffer(jpegs_by_cam[c][4], _np.uint8), _cv2.IMREAD_COLOR)
              for c in cams}

    ptr = _clip.write_throw_clips(tmp_path, sets, commit)
    assert ptr["schema"] == _clip.THROW_CLIP_SCHEMA
    for c in cams:
        e = ptr["cameras"][str(c)]
        assert e["encoding"] == "mjpeg" and e["commit_index"] == 4
        # No bg supplied -> the clip is still BOUNDED: it ends one frame
        # after the commit rather than running to the end of the ring
        # slice, and the before side is held by the ceiling. Only the
        # de-duplication is given up, never the bound.
        assert e["n_frames"] == 4 + 1 + _clip.CLIP_FRAMES_AFTER_COMMIT
        assert e.get("bg_index") is None
        assert (tmp_path / e["clip"]).exists()
        back = _clip.read_commit_frame(tmp_path, ptr, c)
        assert _np.array_equal(back, commit[c])


def test_write_throw_clips_ffv1_path(tmp_path):
    """Mac: slots carried as pixels -> FFV1 clips, commit byte-identical."""
    fr = _frames(n=7, seed=3)
    cams = (0, 1)
    sets = [_fs(pixels={c: fr[i] for c in cams}) for i in range(len(fr))]
    commit = {c: fr[3] for c in cams}
    ptr = _clip.write_throw_clips(tmp_path, sets, commit)
    for c in cams:
        e = ptr["cameras"][str(c)]
        assert e["encoding"] == "ffv1" and e["commit_index"] == 3
        assert _np.array_equal(_clip.read_commit_frame(tmp_path, ptr, c), commit[c])


def test_write_throw_clips_refuses_commit_not_in_window(tmp_path):
    fr = _frames(n=4, seed=5)
    sets = [_fs(pixels={0: fr[i]}) for i in range(len(fr))]
    stranger = _frames(n=1, seed=99)[0]  # not in the window
    with _pytest.raises(_clip.CommitFrameNotInClip):
        _clip.write_throw_clips(tmp_path, sets, {0: stranger})


def test_clip_spans_bg_through_commit_plus_one(tmp_path):
    """THE WINDOW IS TWO REAL FRAMES, NOT A CLOCK.

    The bg is the lifecycle's last idle adoption -- the last frame before
    the dart was detected -- so it defines the clip's start exactly, and
    the scored frame defines the end. Everything before the bg is dropped;
    one frame is kept after the commit.
    """
    fr = _frames(n=25, seed=7)
    sets = [_fs(pixels={0: fr[i]}) for i in range(len(fr))]
    bg_pos, commit_pos = 12, 17
    ptr = _clip.write_throw_clips(
        tmp_path, sets, {0: fr[commit_pos]}, bg_frames_by_cam={0: fr[bg_pos]})
    e = ptr["cameras"]["0"]
    assert e["bg_index"] == 0, "the bg IS the first frame"
    assert e["commit_index"] == commit_pos - bg_pos
    assert e["n_frames"] == (commit_pos - bg_pos) + 1 + _clip.CLIP_FRAMES_AFTER_COMMIT
    back = _clip.read_clip_frames(tmp_path / e["clip"])
    assert _np.array_equal(back[0], fr[bg_pos])                  # starts at bg
    assert _np.array_equal(back[e["commit_index"]], fr[commit_pos])
    assert _np.array_equal(back[-1], fr[commit_pos + 1])         # one after


def test_bg_and_commit_are_both_readable_back_byte_exact(tmp_path):
    """Both ends are pointers now, and both must survive the round trip --
    the package is about to stop storing the bg PNG, so the clip becomes
    the ONLY copy of a scoring input (and of what recalibrate.py re-solves
    a session's calibration from)."""
    fr = _frames(n=20, seed=8)
    sets = [_fs(pixels={0: fr[i]}) for i in range(len(fr))]
    ptr = _clip.write_throw_clips(
        tmp_path, sets, {0: fr[14]}, bg_frames_by_cam={0: fr[9]})
    assert _np.array_equal(_clip.read_bg_frame(tmp_path, ptr, 0), fr[9])
    assert _np.array_equal(_clip.read_commit_frame(tmp_path, ptr, 0), fr[14])
    assert ptr["schema"] == _clip.THROW_CLIP_SCHEMA


def test_bg_not_in_the_slice_is_refused_not_faked(tmp_path):
    """A bg that is not in the ring slice must raise, so the caller keeps
    the PNG. Silently starting the clip somewhere else would leave the
    package pointing at a frame that is not the reference."""
    fr = _frames(n=10, seed=9)
    sets = [_fs(pixels={0: fr[i]}) for i in range(len(fr))]
    stranger = _frames(n=1, seed=99)[0]
    with _pytest.raises(_clip.BgFrameNotInClip):
        _clip.write_throw_clips(tmp_path, sets, {0: fr[6]},
                                bg_frames_by_cam={0: stranger})


def test_bg_beyond_the_ceiling_is_refused(tmp_path):
    """The cap is a ceiling, not a target: a board that never settled can
    leave the last idle adoption far back, and that must not write a giant
    clip. Refuse the de-duplication rather than truncate past the frame we
    promised to include."""
    n = _clip.CLIP_MAX_FRAMES_BEFORE_COMMIT + 6
    fr = _frames(n=n + 4, seed=10)
    sets = [_fs(pixels={0: fr[i]}) for i in range(len(fr))]
    with _pytest.raises(_clip.BgFrameNotInClip):
        _clip.write_throw_clips(tmp_path, sets, {0: fr[n]}, bg_frames_by_cam={0: fr[0]})


def _pkg(tmp_path, n=20, bg_pos=9, commit_pos=14, seed=11):
    """A minimal saved package (bg + dart PNGs + meta) ready to finalize."""
    import json as _json
    fr = _frames(n=n, seed=seed)
    sets = [_fs(pixels={0: fr[i]}) for i in range(len(fr))]
    _cv2.imwrite(str(tmp_path / "cam0_bg.png"), fr[bg_pos])
    _cv2.imwrite(str(tmp_path / "cam0_frame.png"), fr[commit_pos])
    (tmp_path / "meta.json").write_text(_json.dumps({"cameras": [0]}))
    return sets, fr


def test_finalize_drops_the_bg_png_once_it_is_in_the_clip(tmp_path):
    """The whole point: the same frame stops being stored twice. Measured
    on the corpus it is ~1029 KB as PNG against ~75 KB as an MJPEG clip
    frame."""
    sets, fr = _pkg(tmp_path)
    res = _clip.finalize_throw_clip(tmp_path, sets)
    assert res["ok"] and res["bg_in_clip"] is True
    assert not (tmp_path / "cam0_bg.png").exists(), "bg PNG should be gone"
    assert not (tmp_path / "cam0_frame.png").exists()
    # and it is still readable, byte-exact, through the pointer
    assert _np.array_equal(_clip.read_bg_frame(tmp_path, res["video"], 0), fr[9])


def test_finalize_keeps_the_bg_png_when_it_is_not_in_the_slice(tmp_path):
    """Losing the de-duplication must never cost the clip. The clip is
    still written (anchored on the commit) and the bg PNG stays."""
    import json as _json
    fr = _frames(n=20, seed=12)
    sets = [_fs(pixels={0: fr[i]}) for i in range(len(fr))]
    _cv2.imwrite(str(tmp_path / "cam0_bg.png"), _frames(n=1, seed=98)[0])  # not in slice
    _cv2.imwrite(str(tmp_path / "cam0_frame.png"), fr[14])
    (tmp_path / "meta.json").write_text(_json.dumps({"cameras": [0]}))
    res = _clip.finalize_throw_clip(tmp_path, sets)
    assert res["ok"] is True, res
    assert res["bg_in_clip"] is False
    assert (tmp_path / "cam0_bg.png").exists(), "bg PNG must be kept"
    assert res["video"]["cameras"]["0"].get("bg_index") is None
    assert _clip.read_bg_frame(tmp_path, res["video"], 0) is None
