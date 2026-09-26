"""Every throw package is JSON + one MKV per camera -- no frame PNGs.

End-to-end over save_throw_package() (the package's data, and -- outside
the live path -- its two-frame STILLS clip), opendarts.capture.clip.
write_throw_clip() (the ONE clip the live path writes after the data: a
RECORDED window taken from the ring by generation, byte-checked, or the
stills clip when that cannot be had), point_meta_at_clip(), and
load_throw_package() (which must read all of those AND the PNG-only
packages of the existing corpus).
"""
from __future__ import annotations

import json
import types

import av
import cv2
import numpy as np
import pytest

from opendarts.capture import clip
from opendarts.capture.throw_package import load_throw_package, save_throw_package
from opendarts.pipeline import CameraCalibration, ScoreResult


def _frames(n, w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    base = np.repeat((np.add.outer(np.arange(h), np.arange(w)) % 256).astype(np.uint8)[:, :, None], 3, 2)
    return [np.clip(base + rng.integers(0, 30, (h, w, 3)), 0, 255).astype(np.uint8) for _ in range(n)]


def _jpeg_frames(n, seed=0):
    """(jpeg_bytes, decoded_array) pairs -- what a passthrough camera
    hands the rig: the bytes, and the pixels scoring sees, which are by
    definition the cv2 decode of those bytes."""
    out = []
    for f in _frames(n, seed=seed):
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
        assert ok
        data = buf.tobytes()
        out.append((data, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)))
    return out


def _calib(seed):
    rng = np.random.default_rng(seed)
    return CameraCalibration(
        camera_matrix=np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]], np.float64),
        dist_coeffs=np.zeros(5, np.float64),
        rvec=rng.uniform(-0.1, 0.1, 3).astype(np.float64),
        tvec=np.array([0.0, 0.0, 400.0], np.float64),
        pnp_result=None, landmark_spread_ok=True,
    )


def _result():
    return ScoreResult(ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
                       triangulation=None, n_cameras_used=3, max_ray_disagreement_mm=0.5)


def _clip_packets(path):
    with av.open(str(path)) as inp:
        return [bytes(p) for p in inp.demux(inp.streams.video[0]) if p.size]


# -- the stills clip -------------------------------------------------------


def test_unrecorded_package_is_json_plus_a_two_frame_clip_per_camera(tmp_path):
    cams = range(3)
    bg = {c: _frames(1, seed=c)[0] for c in cams}
    dart = {c: _frames(1, seed=50 + c)[0] for c in cams}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, dart, {c: _calib(c) for c in cams}, _result())

    assert not list(pkg.glob("*.png")), "no frame PNGs, ever"
    meta = json.loads((pkg / "meta.json").read_text())
    video = meta["video"]
    assert video["kind"] == clip.CLIP_KIND_STILLS
    assert not clip.is_recorded_clip(video), "a stills clip is not a recording"
    for c in cams:
        entry = video["cameras"][str(c)]
        assert entry == {"clip": f"stills_cam{c}.mkv", "encoding": "ffv1",
                         "n_frames": 2, "bg_index": 0, "commit_index": 1}
        frames = clip.read_clip_frames(pkg / entry["clip"])
        assert len(frames) == 2
        # byte-exact, both ends
        assert np.array_equal(frames[0], bg[c])
        assert np.array_equal(frames[1], dart[c])

    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], dart[c])


def test_camera_jpegs_are_stream_copied_not_reencoded(tmp_path):
    """Given the camera's own bytes for the exact frames, the stills clip
    holds those bytes verbatim -- and still decodes to the scored arrays."""
    cams = range(2)
    pairs = {c: _jpeg_frames(2, seed=10 + c) for c in cams}
    bg = {c: pairs[c][0][1] for c in cams}
    dart = {c: pairs[c][1][1] for c in cams}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, dart, {c: _calib(c) for c in cams}, _result(),
                       bg_jpegs={c: pairs[c][0][0] for c in cams},
                       dart_jpegs={c: pairs[c][1][0] for c in cams})

    video = json.loads((pkg / "meta.json").read_text())["video"]
    for c in cams:
        entry = video["cameras"][str(c)]
        assert entry["encoding"] == "mjpeg"
        assert _clip_packets(pkg / entry["clip"]) == [pairs[c][0][0], pairs[c][1][0]]
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], dart[c])


def test_a_stills_clip_that_does_not_hold_its_bytes_falls_back_to_ffv1(tmp_path, monkeypatch):
    """The stills clip's BYTE check: if the file does not read back as
    exactly the JPEG bytes it was written from, it is discarded for a
    lossless FFV1 clip of the scored arrays -- a bigger file, never a
    wrong one."""
    pairs = _jpeg_frames(3, seed=21)
    bg = {0: pairs[0][1]}
    dart = {0: pairs[1][1]}
    real_write = clip.write_clip_mjpeg

    def _tampered(path, jpegs, width, height):
        # A corrupted mux: the commit packet is some OTHER frame's bytes.
        real_write(path, [jpegs[0], pairs[2][0]], width, height)

    monkeypatch.setattr(clip, "write_clip_mjpeg", _tampered)
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, dart, {0: _calib(0)}, _result(),
                       bg_jpegs={0: pairs[0][0]}, dart_jpegs={0: pairs[1][0]})
    entry = json.loads((pkg / "meta.json").read_text())["video"]["cameras"]["0"]
    assert entry["encoding"] == "ffv1"
    loaded = load_throw_package(pkg)
    assert np.array_equal(loaded.dart_frames[0], dart[0])
    assert np.array_equal(loaded.bg_frames[0], bg[0])


def test_jpegs_for_only_some_cameras_is_fine(tmp_path):
    pairs = {c: _jpeg_frames(2, seed=30 + c) for c in range(2)}
    bg = {c: pairs[c][0][1] for c in range(2)}
    dart = {c: pairs[c][1][1] for c in range(2)}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, dart, {c: _calib(c) for c in range(2)}, _result(),
                       bg_jpegs={0: pairs[0][0][0]}, dart_jpegs={0: pairs[0][1][0]})
    video = json.loads((pkg / "meta.json").read_text())["video"]
    assert video["cameras"]["0"]["encoding"] == "mjpeg"
    assert video["cameras"]["1"]["encoding"] == "ffv1"


def test_deferred_save_writes_the_data_and_no_clip(tmp_path):
    """The live path's first step: the package's data, at once, with no
    clip and no `video` block -- the one clip comes after."""
    cams = range(2)
    bg = {c: _frames(1, seed=c)[0] for c in cams}
    dart = {c: _frames(1, seed=50 + c)[0] for c in cams}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, dart, {c: _calib(c) for c in cams}, _result(),
                       defer_clips=True)
    assert "video" not in json.loads((pkg / "meta.json").read_text())
    assert (pkg / "result.json").exists() and (pkg / "calibration.json").exists()
    assert not list(pkg.glob("*.mkv"))


# -- the one clip: a recorded window, or the stills -----------------------


def _ring_sets(fr, cams, jpegs=None):
    """Ring sets holding frame i at generation i, one pump period apart --
    as pixels, or as the given per-frame JPEG bytes."""
    return [types.SimpleNamespace(
        generation=i, wall_s=1000.0 + i / 30.0,
        pixels={} if jpegs else {c: fr[i] for c in cams},
        jpegs={c: jpegs[i] for c in cams} if jpegs else {})
        for i in range(len(fr))]


def _scored(bg, commit, bg_gen, commit_gen, *, bg_jpegs=None, commit_jpegs=None):
    return clip.ScoredFrames(
        bg=bg, commit=commit,
        bg_jpegs=bg_jpegs or {}, commit_jpegs=commit_jpegs or {},
        bg_generations={c: bg_gen for c in bg},
        commit_generations={c: commit_gen for c in commit})


def _deferred(tmp_path, bg, commit):
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, commit, {c: _calib(c) for c in bg}, _result(),
                       defer_clips=True)
    return pkg


def test_recorded_package_holds_only_the_window_clip(tmp_path):
    """Pixels-only ring slots: an FFV1 window, pixel-checked, bg..commit+1
    taken by generation, and no stills clip beside it."""
    cams = range(3)
    fr = _frames(9, seed=7)
    commit = {c: fr[5] for c in cams}
    bg = {c: fr[2] for c in cams}
    pkg = _deferred(tmp_path, bg, commit)

    video, why_not = clip.write_throw_clip(
        pkg, _scored(bg, commit, 2, 5), cams, sets=_ring_sets(fr, cams))
    assert why_not is None
    clip.point_meta_at_clip(pkg, video)

    video = json.loads((pkg / "meta.json").read_text())["video"]
    assert clip.is_recorded_clip(video)
    assert not list(pkg.glob("*.png"))
    assert not list(pkg.glob("stills_cam*.mkv")), "one clip, never both"
    for c in cams:
        entry = video["cameras"][str(c)]
        assert entry["clip"] == f"clip_cam{c}.mkv" and entry["encoding"] == "ffv1"
        assert (entry["bg_index"], entry["commit_index"], entry["n_frames"]) == (0, 3, 5)
        frames = clip.read_clip_frames(pkg / entry["clip"])
        assert all(np.array_equal(a, b) for a, b in zip(frames, fr[2:7]))
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], commit[c])


def test_mjpeg_window_is_a_stream_copy_of_the_ring_bytes(tmp_path):
    cams = range(2)
    pairs = _jpeg_frames(9, seed=17)
    data = [p[0] for p in pairs]
    commit = {c: pairs[6][1] for c in cams}
    bg = {c: pairs[3][1] for c in cams}
    pkg = _deferred(tmp_path, bg, commit)
    video, why_not = clip.write_throw_clip(
        pkg, _scored(bg, commit, 3, 6,
                     bg_jpegs={c: data[3] for c in cams},
                     commit_jpegs={c: data[6] for c in cams}),
        cams, sets=_ring_sets([None] * 9, cams, jpegs=data))
    assert why_not is None and video["kind"] == clip.CLIP_KIND_WINDOW
    for c in cams:
        entry = video["cameras"][str(c)]
        assert entry["encoding"] == "mjpeg"
        assert (entry["bg_index"], entry["commit_index"], entry["n_frames"]) == (0, 3, 5)
        assert _clip_packets(pkg / entry["clip"]) == data[3:8]


def _tampered_ring(data, gen, cams):
    """The ring's JPEG at `gen` replaced by different bytes -- what a ring
    holding some OTHER frame under that generation looks like."""
    sets = _ring_sets([None] * len(data), cams, jpegs=data)
    other = _jpeg_frames(1, seed=999)[0][0]
    sets[gen].jpegs[0] = other
    return sets


@pytest.mark.parametrize("bad_gen", [3, 6], ids=["bg", "commit"])
def test_window_byte_check_rejects_a_tampered_packet_and_falls_back(tmp_path, bad_gen):
    """THE BYTE CHECK: the window clip's bg and commit packets must equal
    the JPEG bytes paired with the scored frames. A ring that holds other
    bytes at the named generation fails it -- and the WHOLE package falls
    back to the (byte-checked) stills clip, never a window that lies."""
    cams = range(2)
    pairs = _jpeg_frames(9, seed=19)
    data = [p[0] for p in pairs]
    commit = {c: pairs[6][1] for c in cams}
    bg = {c: pairs[3][1] for c in cams}
    pkg = _deferred(tmp_path, bg, commit)
    video, why_not = clip.write_throw_clip(
        pkg, _scored(bg, commit, 3, 6,
                     bg_jpegs={c: data[3] for c in cams},
                     commit_jpegs={c: data[6] for c in cams}),
        cams, sets=_tampered_ring(data, bad_gen, cams))
    assert "byte check" in why_not
    assert video["kind"] == clip.CLIP_KIND_STILLS
    assert not list(pkg.glob("clip_cam*.mkv")), "a failed window leaves nothing behind"
    for c in cams:
        entry = video["cameras"][str(c)]
        assert entry["encoding"] == "mjpeg"
        assert _clip_packets(pkg / entry["clip"]) == [data[3], data[6]]


def test_window_rejects_a_wrong_generation_and_falls_back(tmp_path):
    """A generation that names the WRONG ring frame (here: the one before
    the scored frame) is caught by the same byte check."""
    cams = range(2)
    pairs = _jpeg_frames(9, seed=23)
    data = [p[0] for p in pairs]
    commit = {c: pairs[6][1] for c in cams}
    bg = {c: pairs[3][1] for c in cams}
    pkg = _deferred(tmp_path, bg, commit)
    video, why_not = clip.write_throw_clip(
        pkg, _scored(bg, commit, 3, 5,          # commit is really generation 6
                     bg_jpegs={c: data[3] for c in cams},
                     commit_jpegs={c: data[6] for c in cams}),
        cams, sets=_ring_sets([None] * 9, cams, jpegs=data))
    assert "byte check" in why_not and video["kind"] == clip.CLIP_KIND_STILLS


def test_window_pixel_check_rejects_a_wrong_frame_for_a_pixels_only_camera(tmp_path):
    """No bytes to compare for a pixels-only slot: its pointer frames are
    decoded and compared with the scored arrays, as before."""
    cams = range(2)
    fr = _frames(9, seed=29)
    commit = {c: fr[5] for c in cams}
    bg = {c: fr[2] for c in cams}
    pkg = _deferred(tmp_path, bg, commit)
    video, why_not = clip.write_throw_clip(
        pkg, _scored(bg, commit, 2, 4), cams, sets=_ring_sets(fr, cams))
    assert "pixel check" in why_not and video["kind"] == clip.CLIP_KIND_STILLS


@pytest.mark.parametrize("case", ["bg_gone", "commit_gone", "no_generation", "over_ceiling"])
def test_a_window_that_cannot_be_taken_gets_the_stills_clip(tmp_path, case):
    """Frames the ring no longer holds, a frame with no recorded
    generation, or a bg beyond the ceiling: the package gets the stills
    clip of exactly what was scored, and no window file is left behind."""
    cams = range(2)
    fr = _frames(20, seed=31)
    bg_gen, commit_gen = (2, 5) if case != "over_ceiling" else (2, 15)
    commit = {c: fr[commit_gen] for c in cams}
    bg = {c: fr[bg_gen] for c in cams}
    sets = _ring_sets(fr, cams)
    scored = _scored(bg, commit, bg_gen, commit_gen)
    if case == "bg_gone":
        sets = sets[3:]
    elif case == "commit_gone":
        del sets[commit_gen]
    elif case == "no_generation":
        scored = clip.ScoredFrames(bg=bg, commit=commit,
                                   commit_generations={c: commit_gen for c in cams})
    pkg = _deferred(tmp_path, bg, commit)
    video, why_not = clip.write_throw_clip(pkg, scored, cams, sets=sets)
    assert why_not and video["kind"] == clip.CLIP_KIND_STILLS
    assert not list(pkg.glob("clip_cam*.mkv"))
    clip.point_meta_at_clip(pkg, video)
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], commit[c])


def test_the_window_is_bounded_in_time_around_the_capture_instant(tmp_path):
    """A bg older than the window gets the stills clip; an after frame
    later than it is left out (the clip then ends at the commit)."""
    cams = range(1)
    fr = _frames(9, seed=37)
    commit = {0: fr[5]}
    bg = {0: fr[2]}
    sets = _ring_sets(fr, cams)
    anchor = sets[5].wall_s
    pkg = _deferred(tmp_path, bg, commit)
    video, why_not = clip.write_throw_clip(
        pkg, _scored(bg, commit, 2, 5), cams, sets=sets,
        earliest_wall_s=sets[3].wall_s, latest_wall_s=anchor + 0.2)
    assert "older than the clip window" in why_not
    video, why_not = clip.write_throw_clip(
        pkg, _scored(bg, commit, 2, 5), cams, sets=sets,
        earliest_wall_s=anchor - 0.6, latest_wall_s=sets[5].wall_s + 0.01)
    assert why_not is None
    assert (video["cameras"]["0"]["commit_index"], video["cameras"]["0"]["n_frames"]) == (3, 4)


def test_one_camera_failing_means_no_camera_is_recorded(tmp_path):
    """All or nothing: a package whose cameras disagree about being a
    recording would make every reader ask per camera."""
    cams = range(2)
    fr = _frames(9, seed=41)
    commit = {c: fr[5] for c in cams}
    bg = {c: fr[2] for c in cams}
    scored = clip.ScoredFrames(bg=bg, commit=commit,
                               bg_generations={0: 2, 1: 2}, commit_generations={0: 5})
    pkg = _deferred(tmp_path, bg, commit)
    video, why_not = clip.write_throw_clip(pkg, scored, cams, sets=_ring_sets(fr, cams))
    assert "cam1" in why_not and video["kind"] == clip.CLIP_KIND_STILLS
    assert not list(pkg.glob("clip_cam*.mkv"))


def test_pointing_meta_at_the_clip_is_atomic_and_keeps_later_annotations(tmp_path, monkeypatch):
    """point_meta_at_clip() is the moment a package's frames become
    readable: it keeps what other writers added since the save, and a
    crash inside it leaves the previous meta.json whole, with no temp file."""
    cams = range(1)
    bg = {0: _frames(1, seed=1)[0]}
    dart = {0: _frames(1, seed=2)[0]}
    pkg = _deferred(tmp_path, bg, dart)
    meta = json.loads((pkg / "meta.json").read_text())
    meta["operator_note"] = "added after the save"
    (pkg / "meta.json").write_text(json.dumps(meta))
    before = (pkg / "meta.json").read_bytes()
    video, _ = clip.write_throw_clip(pkg, clip.ScoredFrames(bg=bg, commit=dart), cams)

    import os as _os

    def _die(*_a, **_k):
        raise KeyboardInterrupt("simulated crash at the commit point")

    monkeypatch.setattr(_os, "replace", _die)
    with pytest.raises(KeyboardInterrupt):
        clip.point_meta_at_clip(pkg, video)
    monkeypatch.undo()
    assert (pkg / "meta.json").read_bytes() == before
    assert sorted(p.name for p in pkg.iterdir() if p.name.endswith(".tmp")) == []

    clip.point_meta_at_clip(pkg, video)
    meta = json.loads((pkg / "meta.json").read_text())
    assert meta["operator_note"] == "added after the save" and meta["video"] == video
    loaded = load_throw_package(pkg)
    assert np.array_equal(loaded.dart_frames[0], dart[0])


def test_a_package_recorded_by_the_old_upgrade_still_loads(tmp_path):
    """Packages recorded before 2026-09-26 can point at `clip_cam{N}_rec.mkv`
    (the old stills-then-upgrade swap alternated names). Readers follow the
    pointer, so they load unchanged."""
    cams = range(2)
    fr = _frames(9, seed=43)
    commit = {c: fr[5] for c in cams}
    bg = {c: fr[2] for c in cams}
    pkg = _deferred(tmp_path, bg, commit)
    video, _ = clip.write_throw_clip(pkg, _scored(bg, commit, 2, 5), cams,
                                     sets=_ring_sets(fr, cams))
    for c in cams:
        (pkg / f"clip_cam{c}.mkv").rename(pkg / f"clip_cam{c}_rec.mkv")
        video["cameras"][str(c)]["clip"] = f"clip_cam{c}_rec.mkv"
    clip.point_meta_at_clip(pkg, video)
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], commit[c])


# -- the existing corpus ------------------------------------------------


def _png_only_package(pkg, bg, dart, cams):
    """A package exactly as the pre-2026-09-22 writer left it: PNGs, and
    a meta.json with no video block."""
    pkg.mkdir(parents=True)
    for c in cams:
        assert cv2.imwrite(str(pkg / f"cam{c}_bg.png"), bg[c])
        assert cv2.imwrite(str(pkg / f"cam{c}_frame.png"), dart[c])
    save_throw_package(pkg / "_donor", "sess", bg, dart, {c: _calib(c) for c in cams}, _result())
    meta = json.loads((pkg / "_donor" / "meta.json").read_text())
    meta.pop("video")
    (pkg / "meta.json").write_text(json.dumps(meta))
    for name in ("calibration.json", "result.json"):
        (pkg / name).write_bytes((pkg / "_donor" / name).read_bytes())
    for p in (pkg / "_donor").iterdir():
        p.unlink()
    (pkg / "_donor").rmdir()


def test_old_png_only_package_still_loads(tmp_path):
    cams = range(3)
    bg = {c: _frames(1, seed=60 + c)[0] for c in cams}
    dart = {c: _frames(1, seed=70 + c)[0] for c in cams}
    pkg = tmp_path / "old"
    _png_only_package(pkg, bg, dart, cams)
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], dart[c])
