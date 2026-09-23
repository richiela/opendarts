"""Every throw package is JSON + one MKV per camera -- no frame PNGs.

End-to-end over save_throw_package() (which writes the two-frame STILLS
clip synchronously), opendarts.capture.clip.finalize_throw_clip() (which
upgrades it to a RECORDED window), and load_throw_package() (which must
read both of those AND the PNG-only packages of the existing corpus).
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


# -- the stills clip every package gets ---------------------------------


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


def test_mispaired_jpeg_falls_back_to_ffv1_never_a_wrong_frame(tmp_path):
    """Bytes from a DIFFERENT frame than the array (the pairing bug the
    capture loop exists to prevent) must never reach disk as the record."""
    pairs = _jpeg_frames(3, seed=21)
    bg = {0: pairs[0][1]}
    dart = {0: pairs[1][1]}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, dart, {0: _calib(0)}, _result(),
                       bg_jpegs={0: pairs[0][0]},
                       dart_jpegs={0: pairs[2][0]})   # the NEXT frame's bytes
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


# -- the upgrade to a recording -----------------------------------------


def _ring_sets(fr, cams):
    return [types.SimpleNamespace(pixels={c: fr[i] for c in cams}, jpegs={})
            for i in range(len(fr))]


def test_video_upgrade_replaces_the_stills_clip(tmp_path):
    cams = range(3)
    fr = _frames(9, seed=7)
    commit = {c: fr[5] for c in cams}
    bg = {c: fr[2] for c in cams}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, commit, {c: _calib(c) for c in cams}, _result())

    out = clip.finalize_throw_clip(pkg, _ring_sets(fr, cams))
    assert out["ok"] and out["cameras"] == [0, 1, 2] and out["bg_in_clip"]

    video = json.loads((pkg / "meta.json").read_text())["video"]
    assert clip.is_recorded_clip(video)
    assert not list(pkg.glob("*.png"))
    assert not list(pkg.glob("stills_cam*.mkv")), "the stills clips are gone"
    assert not (pkg / ".clip-upgrade").exists()
    for c in cams:
        entry = video["cameras"][str(c)]
        assert entry["clip"] == f"clip_cam{c}.mkv"
        assert (entry["bg_index"], entry["commit_index"], entry["n_frames"]) == (0, 3, 5)
        frames = clip.read_clip_frames(pkg / entry["clip"])
        assert all(np.array_equal(a, b) for a, b in zip(frames, fr[2:7]))
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], commit[c])


def test_failed_upgrade_leaves_the_stills_clip_valid_and_untouched(tmp_path):
    """Commit frame not in the window -> nothing about the package changes."""
    cams = range(2)
    bg = {c: _frames(1, seed=c)[0] for c in cams}
    dart = {c: _frames(1, seed=40 + c)[0] for c in cams}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, dart, {c: _calib(c) for c in cams}, _result())
    before = {p.name: p.read_bytes() for p in pkg.iterdir()}

    out = clip.finalize_throw_clip(pkg, _ring_sets(_frames(4, seed=999), cams))

    assert not out["ok"]
    assert {p.name: p.read_bytes() for p in pkg.iterdir()} == before
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], dart[c])


def test_crash_before_the_meta_commit_point_leaves_the_stills_package_valid(tmp_path, monkeypatch):
    """The window clips are fully written and moved in, then the process
    'dies' before meta.json is rewritten. meta.json must still describe
    files that exist and hold the scored frames -- which is exactly what
    moving the window clips in under a DIFFERENT name than the stills
    clips buys."""
    cams = range(3)
    fr = _frames(9, seed=8)
    commit = {c: fr[5] for c in cams}
    bg = {c: fr[2] for c in cams}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, commit, {c: _calib(c) for c in cams}, _result())
    meta_before = (pkg / "meta.json").read_bytes()

    def _die(*_a, **_k):
        raise KeyboardInterrupt("simulated crash at the commit point")

    monkeypatch.setattr(clip, "_write_meta_atomically", _die)
    with pytest.raises(KeyboardInterrupt):
        clip.finalize_throw_clip(pkg, _ring_sets(fr, cams))

    assert (pkg / "meta.json").read_bytes() == meta_before
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], commit[c])

    # ...and the upgrade can simply be run again afterwards.
    monkeypatch.undo()
    assert clip.finalize_throw_clip(pkg, _ring_sets(fr, cams))["ok"]
    assert not list(pkg.glob("stills_cam*.mkv"))


def test_upgrading_an_already_recorded_package_never_overwrites_its_live_clip(tmp_path):
    cams = range(2)
    fr = _frames(9, seed=9)
    commit = {c: fr[5] for c in cams}
    bg = {c: fr[2] for c in cams}
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, commit, {c: _calib(c) for c in cams}, _result())
    assert clip.finalize_throw_clip(pkg, _ring_sets(fr, cams))["ok"]
    assert clip.finalize_throw_clip(pkg, _ring_sets(fr, cams))["ok"]
    video = json.loads((pkg / "meta.json").read_text())["video"]
    assert {e["clip"] for e in video["cameras"].values()} == {"clip_cam0_rec.mkv", "clip_cam1_rec.mkv"}
    assert sorted(p.name for p in pkg.glob("*.mkv")) == ["clip_cam0_rec.mkv", "clip_cam1_rec.mkv"]
    loaded = load_throw_package(pkg)
    for c in cams:
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


def test_old_png_only_package_can_still_be_upgraded(tmp_path):
    """The corpus path through finalize_throw_clip: frames come from the
    PNGs, which are dropped once the clip holds them byte-exact."""
    cams = range(2)
    fr = _frames(9, seed=12)
    bg = {c: fr[2] for c in cams}
    dart = {c: fr[5] for c in cams}
    pkg = tmp_path / "old"
    _png_only_package(pkg, bg, dart, cams)
    out = clip.finalize_throw_clip(pkg, _ring_sets(fr, cams))
    assert out["ok"] and out["bg_in_clip"]
    assert not list(pkg.glob("*.png"))
    loaded = load_throw_package(pkg)
    for c in cams:
        assert np.array_equal(loaded.bg_frames[c], bg[c])
        assert np.array_equal(loaded.dart_frames[c], dart[c])
