"""A package's ONE clip, end to end through handle_ready_to_capture(), per
video-record mode (2026-09-26).

The live path writes the package's data first, then its one clip: a
window taken out of the frame ring by the generations the capture loop
paired with the scored frames, or the two-frame stills clip -- then points
meta.json at it, and says so only when the package list could not have
known the outcome ("mismatch"). These run that whole path with
a real FrameRing and ThrowCaptureService, and the frames' JPEG bytes and
generations on the trigger exactly as the capture loop sets them.

Plus the invariant the clip's BYTE check rests on, on real package frames:
the pixels the rig scores for a JPEG slot ARE the cv2 decode of the bytes
paired with them.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

import opendarts.live.capture_daemon as capture_daemon
from opendarts.capture import clip
from opendarts.capture.frame_ring import FrameRing
from opendarts.capture.throw_capture import ThrowCaptureService
from opendarts.capture.throw_package import load_throw_package
from opendarts.capture.trigger_state import ThrowState, ThrowTriggerState
from opendarts.live import local_capture
from opendarts.live.ad_ground_truth import AdGroundTruth
from opendarts.pipeline import CameraCalibration

CAMS = (0, 1, 2)
BG_GEN, COMMIT_GEN, N_SETS = 4, 7, 10


def _calib():
    return CameraCalibration(
        camera_matrix=np.array([[900.0, 0, 32], [0, 900.0, 24], [0, 0, 1]]),
        dist_coeffs=np.zeros(5), rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 400.0]),
        pnp_result=None, landmark_spread_ok=True)


def _jpeg(gen: int, cam: int) -> "tuple[bytes, np.ndarray]":
    """A camera JPEG and the pixels the hub publishes for it (its decode)."""
    rng = np.random.default_rng(gen * 10 + cam)
    img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    return data, cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


class _Rig:
    """A ring filled the way the pump fills it, and the trigger the capture
    loop builds on the commit tick -- arrays, their bytes, their
    generations -- for a dart whose bg is generation BG_GEN and whose
    scored frame is COMMIT_GEN."""

    def __init__(self, *, pixels_only_cam=None, n_sets=N_SETS):
        self.ring = FrameRing(20.0)
        self.frames = {g: {c: _jpeg(g, c) for c in CAMS} for g in range(n_sets)}
        now = time.time()
        for g in range(n_sets):
            wall = now - (COMMIT_GEN - g) / 30.0 - 0.02
            pixels = {c: arr for c, (_, arr) in self.frames[g].items()}
            jpegs = {c: data for c, (data, _) in self.frames[g].items() if c != pixels_only_cam}
            self.ring.append(pixels, wall_s=wall, monotonic_s=1000.0 + g / 30.0,
                             generation=g, jpegs=jpegs)
        self.pixels_only_cam = pixels_only_cam

    def arrays(self, gen):
        return {c: self.frames[gen][c][1] for c in CAMS}

    def trigger(self) -> ThrowTriggerState:
        paired = [c for c in CAMS if c != self.pixels_only_cam]
        return ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=1,
            last_frame=self.arrays(COMMIT_GEN),
            last_frame_jpegs={c: self.frames[COMMIT_GEN][c][0] for c in paired},
            bg_jpegs={c: self.frames[BG_GEN][c][0] for c in paired},
            last_frame_generations={c: COMMIT_GEN for c in CAMS},
            bg_generations={c: BG_GEN for c in CAMS},
        )


class _FakeAd:
    """Just the AdWsListener surface the ground-truth attach uses."""

    def match(self, captured_at, *, window_sec, expect_ordinal=None, allow_time_fallback=False):
        return AdGroundTruth(
            matched=True, match_reason="ok_ws", ad_base_url="http://fake-ad:0",
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            opendarts_captured_at_utc=captured_at, staleness_sec=0.1,
            window_sec=window_sec, sector="20", ring="treble", tip_xy_mm=(1.0, 2.0),
            ad_method="UnanimousCam", ad_bouncer=False)

    def diagnostics_snapshot(self):
        return None


def _throw(tmp_path, rig, mode, *, ad=None):
    events: list[dict] = []
    svc = ThrowCaptureService(rig.ring, capture_root=tmp_path / "captures", record_mode=mode)
    dest = capture_daemon.handle_ready_to_capture(
        rig.trigger(), rig.arrays(BG_GEN), {c: _calib() for c in CAMS},
        tmp_path / "packages", "sess",
        on_event=events.append, background_save=False, throw_capture=svc,
        ad_ws_listener=ad,
    )
    return dest, json.loads((dest / "meta.json").read_text())["video"], events


def _packets(path):
    with av.open(str(path)) as inp:
        return [bytes(p) for p in inp.demux(inp.streams.video[0]) if p.size]


def _assert_window(dest, video, rig):
    assert video["kind"] == clip.CLIP_KIND_WINDOW
    assert sorted(p.name for p in dest.glob("*.mkv")) == [f"clip_cam{c}.mkv" for c in CAMS]
    for c in CAMS:
        e = video["cameras"][str(c)]
        assert (e["bg_index"], e["commit_index"], e["n_frames"]) == (
            0, COMMIT_GEN - BG_GEN, COMMIT_GEN - BG_GEN + 2)
        if c != rig.pixels_only_cam:
            assert e["encoding"] == "mjpeg"
            assert _packets(dest / e["clip"]) == [
                rig.frames[g][c][0] for g in range(BG_GEN, COMMIT_GEN + 2)]
        else:
            assert e["encoding"] == "ffv1"
    _assert_scored_frames_load(dest, rig)


def _assert_stills(dest, video, rig):
    assert video["kind"] == clip.CLIP_KIND_STILLS
    assert sorted(p.name for p in dest.glob("*.mkv")) == [f"stills_cam{c}.mkv" for c in CAMS]
    for c in CAMS:
        e = video["cameras"][str(c)]
        assert (e["bg_index"], e["commit_index"], e["n_frames"]) == (0, 1, 2)
    _assert_scored_frames_load(dest, rig)


def _assert_scored_frames_load(dest, rig):
    loaded = load_throw_package(dest)
    for c in CAMS:
        assert np.array_equal(loaded.bg_frames[c], rig.frames[BG_GEN][c][1])
        assert np.array_equal(loaded.dart_frames[c], rig.frames[COMMIT_GEN][c][1])


def _package_saved(events, dest):
    return [e for e in events if e.get("type") == "PACKAGE_SAVED" and e["path"] == str(dest)]


def test_all_records_the_window_by_generation(tmp_path):
    rig = _Rig()
    dest, video, _ = _throw(tmp_path, rig, "all")
    _assert_window(dest, video, rig)


def test_all_does_not_re_announce_the_package_when_its_clip_lands(tmp_path):
    """Every dart is recorded in "all": the list knew that in advance, so
    the clip's landing only refreshes the server's cached row (has_video)
    -- `refresh_only`, which broadcasts nothing."""
    dest, video, events = _throw(tmp_path, _Rig(), "all")
    saved = _package_saved(events, dest)
    assert [bool(e.get("refresh_only")) for e in saved] == [False, True]


def test_never_does_not_re_announce_the_package_when_its_clip_lands(tmp_path):
    dest, video, events = _throw(tmp_path, _Rig(), "never")
    assert [bool(e.get("refresh_only")) for e in _package_saved(events, dest)] == [False]


@pytest.mark.parametrize("disagrees", [True, False])
def test_mismatch_re_announces_the_package_once_its_clip_lands(tmp_path, monkeypatch, disagrees):
    """Only in "mismatch" can the list not know in advance whether a dart
    gets its recording -- so the clip is announced, after the data save
    and the oracle's own attach."""
    monkeypatch.setattr(capture_daemon, "_oracle_disagreement",
                        lambda package_dir, gt: "oracle disagreement: test" if disagrees else None)
    dest, video, events = _throw(tmp_path, _Rig(), "mismatch", ad=_FakeAd())
    saved = _package_saved(events, dest)
    assert len(saved) == 3 and not any(e.get("refresh_only") for e in saved)
    assert video["kind"] == (clip.CLIP_KIND_WINDOW if disagrees else clip.CLIP_KIND_STILLS)


def test_the_server_refreshes_its_row_quietly_on_refresh_only(tmp_path):
    """The server side of "all": the cached row learns has_video, and no
    PACKAGES_UPDATED goes out -- unless the package was never announced,
    in which case it is announced as usual."""
    import asyncio

    from opendarts.live.server import create_app

    dest, _, _ = _throw(tmp_path, _Rig(), "all")
    meta_path = dest / "meta.json"
    recorded = json.loads(meta_path.read_text())
    meta_path.write_text(json.dumps({k: v for k, v in recorded.items() if k != "video"}))

    app = create_app(package_root=tmp_path / "packages", enable_background_poll=False)
    state = app.state.opendarts_state
    sent = []

    async def _broadcast(msg):
        sent.append(msg)

    state._broadcast = _broadcast  # noqa: SLF001
    assert [p["has_video"] for p in state.list_packages()] == [False]

    meta_path.write_text(json.dumps(recorded))  # the clip lands
    event = {"type": "PACKAGE_SAVED", "path": str(dest), "session": "sess", "refresh_only": True}
    asyncio.run(state._handle_live_event(event))  # noqa: SLF001
    assert [m for m in sent if m["type"] == "PACKAGES_UPDATED"] == []
    assert [p["has_video"] for p in state.list_packages()] == [True]

    # a package the server never saw is announced even so
    state._apply_package_changes({str(dest)}, [])  # noqa: SLF001
    asyncio.run(state._handle_live_event(event))  # noqa: SLF001
    assert [m["new_count"] for m in sent if m["type"] == "PACKAGES_UPDATED"] == [1]


def test_never_writes_the_stills_clip_only(tmp_path):
    rig = _Rig()
    dest, video, _ = _throw(tmp_path, rig, "never")
    _assert_stills(dest, video, rig)


def test_mismatch_disagreement_records_the_window(tmp_path, monkeypatch):
    """The oracle's answer reaches the clip decision: a disagreement gets
    the recording."""
    seen = []

    def _disagrees(package_dir, gt):
        seen.append(gt)
        return "oracle disagreement: test"

    monkeypatch.setattr(capture_daemon, "_oracle_disagreement", _disagrees)
    rig = _Rig()
    dest, video, _ = _throw(tmp_path, rig, "mismatch", ad=_FakeAd())
    assert len(seen) == 1 and seen[0].matched and seen[0].sector == "20"
    _assert_window(dest, video, rig)


def test_mismatch_agreement_writes_the_stills_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_daemon, "_oracle_disagreement", lambda package_dir, gt: None)
    rig = _Rig()
    dest, video, _ = _throw(tmp_path, rig, "mismatch", ad=_FakeAd())
    _assert_stills(dest, video, rig)


def test_mismatch_without_an_oracle_writes_the_stills_clip_without_waiting(tmp_path):
    t0 = time.monotonic()
    rig = _Rig()
    dest, video, _ = _throw(tmp_path, rig, "mismatch", ad=None)
    assert time.monotonic() - t0 < capture_daemon.ORACLE_VERDICT_WAIT_S
    _assert_stills(dest, video, rig)


def test_frames_missing_from_the_ring_fall_back_to_the_stills_clip(tmp_path):
    """The ring was cleared (a missed-dart dump, a restart of the ring):
    the recording cannot be had, so the package gets the stills clip."""
    rig = _Rig()
    rig.ring.clear()
    dest, video, _ = _throw(tmp_path, rig, "all")
    _assert_stills(dest, video, rig)


def test_ring_disabled_falls_back_to_the_stills_clip(tmp_path):
    rig = _Rig()
    rig.ring = FrameRing(0.0)
    dest, video, _ = _throw(tmp_path, rig, "all")
    _assert_stills(dest, video, rig)


def test_a_pixels_only_camera_gets_an_ffv1_window_with_the_pixel_check(tmp_path):
    rig = _Rig(pixels_only_cam=1)
    dest, video, _ = _throw(tmp_path, rig, "all")
    _assert_window(dest, video, rig)


def test_no_frame_after_the_commit_yet_waits_then_ends_the_clip_at_the_commit(tmp_path, monkeypatch):
    """The ring stops at the commit (a stalled pump): the recording waits
    its bounded moment for the next frame, then ends at the commit."""
    from opendarts.capture import throw_capture

    monkeypatch.setattr(throw_capture, "CLIP_AFTER_FRAME_WAIT_S", 0.05)
    rig = _Rig(n_sets=COMMIT_GEN + 1)
    dest, video, _ = _throw(tmp_path, rig, "all")
    assert video["kind"] == clip.CLIP_KIND_WINDOW
    for c in CAMS:
        e = video["cameras"][str(c)]
        assert (e["commit_index"], e["n_frames"]) == (COMMIT_GEN - BG_GEN, COMMIT_GEN - BG_GEN + 1)
    _assert_scored_frames_load(dest, rig)


def test_a_trigger_without_generations_gets_the_stills_clip(tmp_path):
    """A hub that cannot name its frames' ring sets (no grab_paired)."""
    rig = _Rig()
    trigger = rig.trigger()
    trigger.last_frame_generations = trigger.bg_generations = None
    svc = ThrowCaptureService(rig.ring, capture_root=tmp_path / "c", record_mode="all")
    dest = capture_daemon.handle_ready_to_capture(
        trigger, rig.arrays(BG_GEN), {c: _calib() for c in CAMS}, tmp_path / "packages",
        "sess", background_save=False, throw_capture=svc)
    _assert_stills(dest, json.loads((dest / "meta.json").read_text())["video"], rig)


# -- the invariant the byte check rests on, on REAL package frames ----------


def _real_packages_root() -> "Path | None":
    """A directory of real throw packages: OPENDARTS_REAL_PACKAGES, else this
    checkout's data/packages. None when neither has any."""
    env = os.environ.get("OPENDARTS_REAL_PACKAGES")
    root = Path(env) if env else Path(__file__).resolve().parent.parent / "data" / "packages"
    return root if any(root.glob("*/*/meta.json")) else None


REAL_ROOT = _real_packages_root()


def _real_packages_by_session():
    """{session: [package dir, ...]} for every session holding packages."""
    out = {}
    for session in sorted(p for p in REAL_ROOT.iterdir() if p.is_dir()):
        throws = sorted(p for p in session.iterdir() if (p / "meta.json").is_file())
        if throws:
            out[session.name] = throws
    return out


def _real_jpeg_packets(limit_per_session: "int | None"):
    """(package, cam, packet index, bytes) for the MJPEG clips of the real
    corpus -- the camera bytes exactly as the rig stored them."""
    for throws in _real_packages_by_session().values():
        for pkg in throws[:limit_per_session]:
            video = json.loads((pkg / "meta.json").read_text()).get("video") or {}
            for key, entry in sorted((video.get("cameras") or {}).items()):
                if entry.get("encoding") != "mjpeg":
                    continue
                for i, data in enumerate(_packets(pkg / entry["clip"])):
                    yield pkg, int(key), i, data


def _check_invariant(packets) -> int:
    checked = 0
    for pkg, cam, i, data in packets:
        where = f"{pkg.name} cam{cam} packet {i}"
        decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None, where
        # The hub's own ingest of a camera JPEG (passthrough / stream):
        # the bytes it keeps are these bytes, and the pixels it publishes
        # are their cv2 decode.
        kind, published, kept = local_capture._raw_to_frame(np.frombuffer(data, np.uint8))
        assert kind == "jpeg" and kept == data, where
        assert np.array_equal(published, decoded), where
        # The same bytes decode to the same pixels every time.
        again = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(again, decoded), where
        checked += 1
    return checked


@pytest.mark.skipif(REAL_ROOT is None, reason="no real packages (set OPENDARTS_REAL_PACKAGES)")
def test_real_frames_pixels_are_the_decode_of_their_paired_bytes():
    """THE INVARIANT THE BYTE CHECK RELIES ON. A clip is accepted when its
    pointer packets equal the JPEG bytes paired with the scored frames --
    which proves it holds the scored PIXELS only because the rig's scored
    pixels are, by construction, cv2.imdecode of exactly those bytes.
    Checked on real camera bytes (every packet of two packages per
    session), through the hub's own ingest; and the replay read path
    returns the same pixels for them."""
    assert _check_invariant(_real_jpeg_packets(2)) > 0
    for throws in _real_packages_by_session().values():
        pkg = throws[0]
        video = json.loads((pkg / "meta.json").read_text())["video"]
        for key, entry in video["cameras"].items():
            packets = _packets(pkg / entry["clip"])
            bg, commit = clip.read_bg_and_commit_frames(pkg, video, int(key))
            for arr, idx in ((bg, entry["bg_index"]), (commit, entry["commit_index"])):
                want = cv2.imdecode(np.frombuffer(packets[idx], np.uint8), cv2.IMREAD_COLOR)
                assert np.array_equal(arr, want)


@pytest.mark.slow
@pytest.mark.skipif(REAL_ROOT is None, reason="no real packages (set OPENDARTS_REAL_PACKAGES)")
def test_real_frames_pixels_are_the_decode_of_their_paired_bytes_whole_corpus():
    assert _check_invariant(_real_jpeg_packets(None)) > 0


@pytest.mark.skipif(REAL_ROOT is None, reason="no real packages (set OPENDARTS_REAL_PACKAGES)")
def test_real_frames_synthetic_jpeg_publishes_the_decode_of_its_bytes():
    """macOS: the hub encodes the camera's pixels itself (SYNTHETIC JPEG)
    and publishes the DECODE of that encode, never the raw pixels -- so the
    same invariant holds for the bytes it keeps. On real frames."""
    for pkg, cam, i, data in _real_jpeg_packets(1):
        if i:
            continue
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        published, kept = local_capture._synthesise_jpeg(raw)
        assert np.array_equal(
            published, cv2.imdecode(np.frombuffer(kept, np.uint8), cv2.IMREAD_COLOR))
