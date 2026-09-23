"""Tests for opendarts/capture/throw_capture.py -- the two triggers, the
anchor, and the two honesty requirements (aged out, integrity).

Three cameras throughout. The window arithmetic is per SET and the byte
arithmetic is per FRAME, and one camera makes those the same number.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

from opendarts.capture.frame_dump import FRAMES_FILENAME, MANIFEST_FILENAME
from opendarts.capture.frame_ring import FrameRing
from opendarts.capture.throw_capture import (
    CAPTURE_LIST_LIMIT,
    INTEGRITY_FILENAME,
    MISSCORE_WINDOW_AFTER_S,
    MISSCORE_WINDOW_BEFORE_S,
    ThrowCaptureService,
    anchor_wall_s_for_package,
    list_captures_on_disk,
    measure_captures,
)


def frame(slot: int, tick: int) -> np.ndarray:
    rng = np.random.default_rng(seed=slot * 100_000 + tick)
    arr = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    arr[0, 0] = (slot + 1, tick % 251, 3)
    return arr


def fill(ring: FrameRing, *, n: int, wall0: float, mono0: float = 4321.0,
         spacing: float = 1 / 30.0) -> None:
    for i in range(n):
        ring.append(
            {s: frame(s, i) for s in (0, 1, 2)},
            wall_s=wall0 + i * spacing,
            monotonic_s=mono0 + i * spacing,
            generation=i,
        )


def service(tmp_path, ring: "FrameRing | None") -> ThrowCaptureService:
    return ThrowCaptureService(ring, capture_root=tmp_path / "captures")


# -- the window, as a documented decision -------------------------------


def test_the_misscore_window_is_asymmetric_with_more_before_than_after():
    """Detection fires AFTER the landing (dart_stable_frames plus
    cooldowns) and the package's timestamp is stamped later still, after
    the engine has scored -- so the useful frames are almost all before
    the anchor. If this ever becomes symmetric, the capture is centred on
    an already-settled board."""
    assert MISSCORE_WINDOW_BEFORE_S > MISSCORE_WINDOW_AFTER_S
    assert MISSCORE_WINDOW_BEFORE_S == pytest.approx(0.5)
    assert MISSCORE_WINDOW_AFTER_S == pytest.approx(0.2)


# -- the anchor ---------------------------------------------------------


def _package(tmp_path, *, captured_at: str, handle_total_s=None, frames=None) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "meta.json").write_text(json.dumps({
        "session": "s", "cameras": [0, 1, 2], "captured_at_utc": captured_at,
    }))
    if handle_total_s is not None:
        (pkg / "capture_diagnostics.json").write_text(json.dumps({
            "schema_version": 2, "timings": {"handle_total_s": handle_total_s},
        }))
    for cam, arr in (frames or {}).items():
        assert cv2.imwrite(str(pkg / f"cam{cam}_frame.png"), arr)
    return pkg


def test_the_anchor_subtracts_handle_total_s_to_reach_the_real_capture_instant(tmp_path):
    """meta.captured_at_utc is stamped at the END of the synchronous path,
    after scoring -- the capture daemon documents the recovery itself."""
    stamped = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)
    pkg = _package(tmp_path, captured_at=stamped.isoformat(), handle_total_s=0.42)
    anchor = anchor_wall_s_for_package(pkg)
    assert anchor.basis == "capture_instant"
    assert anchor.wall_s == pytest.approx(stamped.timestamp() - 0.42)
    assert "handle_total_s" in (anchor.detail or "")


def test_a_package_without_handle_total_s_anchors_on_the_timestamp_and_says_so(tmp_path):
    stamped = datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)
    pkg = _package(tmp_path, captured_at=stamped.isoformat())
    anchor = anchor_wall_s_for_package(pkg)
    assert anchor.basis == "package_timestamp"
    assert anchor.wall_s == pytest.approx(stamped.timestamp())
    # Not silently equivalent: the caller is told the centre is late.
    assert "LATER than the landing" in (anchor.detail or "")


def test_an_unparseable_timestamp_yields_no_anchor_rather_than_1970(tmp_path):
    pkg = _package(tmp_path, captured_at="not a timestamp")
    anchor = anchor_wall_s_for_package(pkg)
    assert anchor.wall_s is None
    assert "nothing to anchor a window on" in (anchor.detail or "")


def test_a_trailing_z_timestamp_still_parses(tmp_path):
    pkg = _package(tmp_path, captured_at="2026-09-16T12:00:00Z")
    anchor = anchor_wall_s_for_package(pkg)
    assert anchor.wall_s == pytest.approx(
        datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc).timestamp()
    )


# -- the missed-dart trigger --------------------------------------------


def test_a_missed_dart_capture_writes_the_whole_buffer_and_resumes_the_ring(tmp_path):
    ring = FrameRing(30.0)
    fill(ring, n=60, wall0=1_757_000_000.0)
    svc = service(tmp_path, ring)
    result = svc.capture_missed_dart(reason="I threw and nothing registered")
    assert result["ok"] is True
    svc.writer.join()

    dest = Path(result["job"]["dest_dir"])
    manifest = json.loads((dest / MANIFEST_FILENAME).read_text())
    assert manifest["kind"] == "missed_dart"
    assert manifest["n_sets"] == 60                     # the WHOLE buffer
    assert manifest["n_frames"] == 180                  # three per set
    assert manifest["extra"]["trigger"] == "manual"
    assert manifest["extra"]["operator_reason"].startswith("I threw")
    # THE RING IS RUNNING AGAIN. Left paused, the rig would be unable to
    # capture anything for the rest of the session while looking healthy.
    assert ring.paused is False
    assert ring.stats()["dropped_while_paused"] == 0


def test_a_missed_dart_capture_pauses_the_ring_for_the_duration(tmp_path):
    """Peak memory stays at one buffer rather than one buffer plus
    whatever arrives during the write."""
    ring = FrameRing(30.0)
    fill(ring, n=10, wall0=1_757_000_000.0)
    svc = service(tmp_path, ring)

    paused_during_write: list[bool] = []
    original = ring.resume

    def _watch() -> None:
        paused_during_write.append(ring.paused)
        original()

    ring.resume = _watch  # type: ignore[method-assign]
    svc.capture_missed_dart(reason="x")
    svc.writer.join()
    assert paused_during_write == [True]
    assert ring.paused is False


def test_a_missed_dart_capture_on_an_empty_ring_refuses_and_unpauses(tmp_path):
    ring = FrameRing(30.0)
    svc = service(tmp_path, ring)
    result = svc.capture_missed_dart(reason="x")
    assert result["ok"] is False
    assert "holds nothing to write" in result["reason"]
    assert ring.paused is False                         # not left stuck
    assert not (tmp_path / "captures").exists()


def test_a_missed_dart_capture_with_no_ring_at_all_says_what_to_do(tmp_path):
    svc = service(tmp_path, None)
    result = svc.capture_missed_dart(reason="x")
    assert result["ok"] is False
    assert "frame_ring_seconds" in result["reason"]


def test_a_disabled_ring_refuses_with_the_setting_that_disabled_it(tmp_path):
    svc = service(tmp_path, FrameRing(0))
    result = svc.capture_missed_dart(reason="x")
    assert result["ok"] is False
    assert "frame_ring_seconds=0" in result["reason"]
    # Called TWICE: a refusal that only fires once is a refusal that
    # stops being reported.
    assert svc.capture_missed_dart(reason="x")["ok"] is False


# -- the misscore trigger -----------------------------------------------


def test_a_misscore_capture_takes_the_window_around_the_throw_not_the_latest_frames(
    tmp_path,
):
    """By the time anyone presses the button they have walked to a screen
    and the board is empty -- the last N frames are of an empty board."""
    wall0 = 1_757_000_000.0
    ring = FrameRing(30.0)
    fill(ring, n=600, wall0=wall0)                      # 20s at 30/s
    svc = service(tmp_path, ring)
    anchor = wall0 + 300 / 30.0                         # the throw, 10s in

    result = svc.capture_misscore(anchor, reason="called T20, was S20")
    assert result["ok"] is True
    svc.writer.join()

    manifest = json.loads(
        (Path(result["job"]["dest_dir"]) / MANIFEST_FILENAME).read_text()
    )
    assert manifest["kind"] == "misscore"
    assert manifest["anchor_wall_s"] == pytest.approx(anchor)
    # -0.5s/+0.2s at 30/s is 15 before + the anchor + 6 after = 22 sets.
    assert manifest["n_sets"] == 22
    assert manifest["n_frames"] == 66
    # Centred on the THROW: the newest retained set is 10s later and is
    # deliberately not in this capture.
    newest = max(fs["wall_s"] for fs in manifest["sets"])
    assert newest < wall0 + 600 / 30.0 - 9


def test_a_misscore_capture_leaves_the_ring_running(tmp_path):
    ring = FrameRing(30.0)
    wall0 = 1_757_000_000.0
    fill(ring, n=60, wall0=wall0)
    svc = service(tmp_path, ring)
    svc.capture_misscore(wall0 + 1.0, reason="x")
    # Not paused at any point -- a misscore is usually the first of
    # several, and stopping the ring would lose the next one.
    assert ring.paused is False
    svc.writer.join()
    assert ring.paused is False
    assert ring.stats()["dropped_while_paused"] == 0


def test_an_aged_out_throw_is_refused_with_the_numbers_and_writes_nothing(tmp_path):
    ring = FrameRing(22.0)
    wall0 = 1_757_000_000.0
    fill(ring, n=700, wall0=wall0)
    svc = service(tmp_path, ring)

    result = svc.capture_misscore(wall0 - 30.0, reason="that one a minute ago")
    assert result["ok"] is False
    assert result["aged_out"] is True
    assert "22.0s" in result["reason"]              # how far back it reaches
    assert "s older than the newest frame" in result["reason"]
    # NEVER AN EMPTY FILE and never a silent success.
    assert not (tmp_path / "captures").exists()
    assert svc.writer.status()["busy"] is False


def test_a_misscore_with_no_anchor_and_no_package_is_refused(tmp_path):
    ring = FrameRing(22.0)
    fill(ring, n=100, wall0=1_757_000_000.0)
    svc = service(tmp_path, ring)
    result = svc.capture_misscore(None, reason="x")
    assert result["ok"] is False
    assert "no anchor" in result["reason"]


def test_a_misscore_can_resolve_its_anchor_from_a_package(tmp_path):
    now = datetime.now(timezone.utc)
    wall0 = now.timestamp() - 5.0
    ring = FrameRing(30.0)
    fill(ring, n=300, wall0=wall0)
    pkg = _package(tmp_path, captured_at=(now - timedelta(seconds=2)).isoformat(),
                   handle_total_s=0.25)
    svc = service(tmp_path, ring)
    result = svc.capture_misscore(None, reason="x", package_dir=pkg)
    assert result["ok"] is True
    svc.writer.join()
    manifest = json.loads(
        (Path(result["job"]["dest_dir"]) / MANIFEST_FILENAME).read_text()
    )
    assert manifest["extra"]["anchor_basis"] == "capture_instant"
    assert manifest["anchor_wall_s"] == pytest.approx(
        (now - timedelta(seconds=2)).timestamp() - 0.25
    )


# -- the integrity tripwire, end to end ---------------------------------


def test_a_misscore_naming_a_package_writes_an_integrity_verdict_beside_the_dump(
    tmp_path,
):
    now = datetime.now(timezone.utc)
    wall0 = now.timestamp() - 5.0
    ring = FrameRing(30.0)
    fill(ring, n=300, wall0=wall0)
    # The package stores the SAME arrays the ring holds, which is what the
    # live path really does -- trigger.last_frame is the array grab_all()
    # handed the loop, which is the array the pump cached and the ring
    # retained.
    chosen = ring.snapshot().sets[150].frames
    pkg = _package(
        tmp_path,
        captured_at=(now - timedelta(seconds=0.1)).isoformat(),
        handle_total_s=0.0,
        frames=chosen,
    )
    anchor_wall = wall0 + 150 / 30.0
    svc = service(tmp_path, ring)
    result = svc.capture_misscore(anchor_wall, reason="x", package_dir=pkg)
    assert result["ok"] is True
    svc.writer.join()

    dest = Path(result["job"]["dest_dir"])
    verdict = json.loads((dest / INTEGRITY_FILENAME).read_text())
    assert verdict["ok"] is True
    assert verdict["checked"] == 3 and verdict["matched"] == 3
    assert "re-encoded or mutated" in verdict["what_this_proves"]
    assert svc.status()["captures"][-1]["integrity"]["ok"] is True


def test_the_integrity_verdict_records_a_mismatch_rather_than_hiding_it(tmp_path):
    now = datetime.now(timezone.utc)
    wall0 = now.timestamp() - 5.0
    ring = FrameRing(30.0)
    fill(ring, n=300, wall0=wall0)
    chosen = dict(ring.snapshot().sets[150].frames)
    tampered = chosen[2].copy()
    tampered[3, 4, 1] = (int(tampered[3, 4, 1]) + 1) % 256
    chosen[2] = tampered
    pkg = _package(tmp_path, captured_at=now.isoformat(), handle_total_s=0.0,
                   frames=chosen)
    svc = service(tmp_path, ring)
    result = svc.capture_misscore(wall0 + 150 / 30.0, reason="x", package_dir=pkg)
    svc.writer.join()

    dest = Path(result["job"]["dest_dir"])
    verdict = json.loads((dest / INTEGRITY_FILENAME).read_text())
    assert verdict["ok"] is False
    assert verdict["matched"] == 2
    assert any("cam2" in d for d in verdict["details"])
    # The dump itself is KEPT: it is still evidence about a throw that
    # cannot be retaken.
    assert (dest / FRAMES_FILENAME).exists()


# -- status -------------------------------------------------------------


def test_status_reports_the_ring_the_writer_and_the_window(tmp_path):
    ring = FrameRing(22.0)
    fill(ring, n=30, wall0=1_757_000_000.0)
    svc = service(tmp_path, ring)
    status = svc.status()
    assert status["ring"]["seconds"] == 22.0
    assert status["ring"]["sets"] == 30
    assert status["writer"]["busy"] is False
    assert status["window"]["misscore_before_s"] == MISSCORE_WINDOW_BEFORE_S
    assert status["captures"] == []

    svc.capture_missed_dart(reason="x", source="oracle")
    svc.writer.join()
    after = svc.status()
    assert len(after["captures"]) == 1
    assert after["captures"][0]["source"] == "oracle"
    assert after["captures"][0]["kind"] == "missed_dart"


# -- the list comes off the disk ----------------------------------------
#
# The panel used to list what THIS process had written, which empties on
# restart while every file stays where it was -- a list saying "no
# captures" beside gigabytes of them. These pin the fix: the directory is
# the list, and what the session remembers is merged onto it.


def test_captures_on_disk_are_listed_by_a_process_that_never_wrote_them(tmp_path):
    """The restart case, directly: one service writes a real dump, a
    SECOND service with the same root and an empty memory lists it."""
    ring = FrameRing(22.0)
    fill(ring, n=30, wall0=1_757_000_000.0)
    writer_svc = service(tmp_path, ring)
    writer_svc.capture_missed_dart(reason="a dart landed and nothing scored")
    writer_svc.writer.join()

    fresh = service(tmp_path, FrameRing(22.0))
    listed = fresh.status()["captures"]
    assert len(listed) == 1, "a fresh process listed no captures beside real files"
    record = listed[0]
    assert record["kind"] == "missed_dart"
    assert record["on_disk"] is True
    assert record["at_utc"] is not None
    # Three cameras' frames really are in there, so the size is real.
    dest = Path(record["dest"])
    on_disk_bytes = sum(p.stat().st_size for p in dest.iterdir() if p.is_file())
    assert record["bytes"] == on_disk_bytes > 0


def test_a_remembered_capture_whose_files_are_gone_is_kept_and_flagged(tmp_path):
    """Something removed the dump out from under this process. Dropping it
    from the list would quietly agree with the removal; the flag is what
    makes it visible."""
    ring = FrameRing(22.0)
    fill(ring, n=30, wall0=1_757_000_000.0)
    svc = service(tmp_path, ring)
    svc.capture_missed_dart(reason="x")
    svc.writer.join()
    dest = Path(svc.status()["captures"][0]["dest"])
    shutil.rmtree(dest)

    listed = svc.status()["captures"]
    assert len(listed) == 1
    assert listed[0]["on_disk"] is False
    assert listed[0]["dest"] == str(dest)


def test_measure_captures_counts_directories_and_weighs_every_byte(tmp_path):
    root = tmp_path / "captures"
    ring = FrameRing(22.0)
    fill(ring, n=30, wall0=1_757_000_000.0)
    svc = ThrowCaptureService(ring, capture_root=root)
    svc.capture_missed_dart(reason="one")
    svc.writer.join()
    svc.capture_missed_dart(reason="two")
    svc.writer.join()

    measured = measure_captures(root)
    assert measured["count"] == 2
    real = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    assert measured["bytes"] == real > 0


def test_measuring_a_capture_root_that_does_not_exist_is_zero_not_an_error(tmp_path):
    measured = measure_captures(tmp_path / "never-created")
    assert measured["count"] == 0
    assert measured["bytes"] == 0
    assert not (tmp_path / "never-created").exists(), "measuring must not create it"


def test_a_directory_this_module_did_not_write_is_still_reported(tmp_path):
    """An unparseable name is still a real directory holding real bytes.
    Reporting it as an unknown capture is more honest than pretending it
    is not there -- a delete is going to take it either way."""
    root = tmp_path / "captures"
    (root / "something-else").mkdir(parents=True)
    (root / "something-else" / "data.bin").write_bytes(b"x" * 1234)

    listed = list_captures_on_disk(root)
    assert len(listed) == 1
    assert listed[0]["kind"] is None
    assert listed[0]["at_utc"] is None
    assert listed[0]["bytes"] == 1234
    assert measure_captures(root) == {"root": str(root), "count": 1, "bytes": 1234}


def test_the_list_is_bounded_while_the_measurement_is_not(tmp_path):
    """The list is what a poll renders, so it is capped. The count is what
    a confirmation dialog quotes, so it is the truth."""
    root = tmp_path / "captures"
    for i in range(CAPTURE_LIST_LIMIT + 5):
        d = root / f"2026091{i // 10}T12{i:04d}-000001-misscore"
        d.mkdir(parents=True)
        (d / "frames.bin").write_bytes(b"y" * 10)

    assert len(list_captures_on_disk(root)) == CAPTURE_LIST_LIMIT
    assert measure_captures(root)["count"] == CAPTURE_LIST_LIMIT + 5


# -- record_throw_clip: the per-throw video-record path ---------------------

def test_record_throw_clip_writes_clip_into_package(tmp_path):
    """A real ring + a saved package -> record_throw_clip UPGRADES the
    package's two-frame stills clips to verified bg..commit+1 recordings
    and repoints meta.video at them. No PNGs at any point."""
    import json
    from opendarts.capture import clip
    from opendarts.capture.throw_package import save_throw_package
    from opendarts.pipeline import CameraCalibration, ScoreResult

    ring = FrameRing(20.0)
    wall0 = 100_000.0
    fill(ring, n=60, wall0=wall0)
    commit_tick = 30
    bg_tick = commit_tick - 5
    commit = {s: frame(s, commit_tick) for s in (0, 1, 2)}   # == the ring's tick-30 frame
    # The bg must be a frame the ring holds: its only copy is the stills
    # clip, so an upgrade that cannot find it is abandoned (see
    # test_record_throw_clip_keeps_stills_when_bg_not_in_ring).
    bg = {s: frame(s, bg_tick) for s in (0, 1, 2)}
    calib = lambda: CameraCalibration(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros(5),
        rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 400.0]),
        pnp_result=None, landmark_spread_ok=True)
    calibs = {s: calib() for s in (0, 1, 2)}
    result = ScoreResult(ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
                         triangulation=None, n_cameras_used=3, max_ray_disagreement_mm=0.5)

    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, commit, calibs, result)
    assert not clip.is_recorded_clip(json.loads((pkg / "meta.json").read_text())["video"])

    svc = service(tmp_path, ring)
    anchor = wall0 + commit_tick * (1 / 30.0)
    out = svc.record_throw_clip(pkg, anchor, reason="test", source="all")

    assert out["ok"], out
    assert not list(pkg.glob("*.png"))
    meta = json.loads((pkg / "meta.json").read_text())
    assert clip.is_recorded_clip(meta["video"])
    for s in (0, 1, 2):
        assert not (pkg / f"stills_cam{s}.mkv").exists(), "the stills clip is replaced"
        assert (pkg / f"clip_cam{s}.mkv").exists()
        entry = meta["video"]["cameras"][str(s)]
        assert entry["n_frames"] == 5 + 1 + 1   # bg..commit, plus one after
        bg_back, commit_back = clip.read_bg_and_commit_frames(pkg, meta["video"], s)
        assert np.array_equal(commit_back, commit[s])
        assert np.array_equal(bg_back, bg[s])


def test_record_throw_clip_keeps_stills_when_bg_not_in_ring(tmp_path):
    """A FAILED upgrade costs the package nothing: its stills clip and
    meta.video are left exactly as save_throw_package wrote them."""
    import json
    from opendarts.capture import clip
    from opendarts.capture.throw_package import load_throw_package, save_throw_package
    from opendarts.pipeline import CameraCalibration, ScoreResult

    ring = FrameRing(20.0)
    wall0 = 100_000.0
    fill(ring, n=60, wall0=wall0)
    commit = {s: frame(s, 30) for s in (0, 1, 2)}
    bg = {s: frame(s, 0) for s in (0, 1, 2)}   # long gone from a slice around tick 30
    calibs = {s: CameraCalibration(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros(5),
        rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 400.0]),
        pnp_result=None, landmark_spread_ok=True) for s in (0, 1, 2)}
    result = ScoreResult(ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
                         triangulation=None, n_cameras_used=3, max_ray_disagreement_mm=0.5)
    pkg = tmp_path / "pkg"
    save_throw_package(pkg, "sess", bg, commit, calibs, result)
    before = {p.name: p.read_bytes() for p in pkg.iterdir()}

    out = service(tmp_path, ring).record_throw_clip(
        pkg, wall0 + 30 / 30.0, reason="test", source="all")

    assert not out["ok"]
    after = {p.name: p.read_bytes() for p in pkg.iterdir()}
    assert after == before, "a failed upgrade must not touch the package"
    loaded = load_throw_package(pkg)
    for s in (0, 1, 2):
        assert np.array_equal(loaded.bg_frames[s], bg[s])
        assert np.array_equal(loaded.dart_frames[s], commit[s])
    assert not clip.is_recorded_clip(json.loads((pkg / "meta.json").read_text())["video"])


def test_record_throw_clip_refused_without_ring(tmp_path):
    svc = service(tmp_path, None)
    out = svc.record_throw_clip(tmp_path / "pkg", 1.0, reason="x", source="all")
    assert out["ok"] is False
