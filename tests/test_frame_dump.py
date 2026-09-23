"""Tests for opendarts/capture/frame_dump.py -- writing a ring slice, the
one-at-a-time rule, the `finally` that resumes the ring, and the integrity
tripwire.

Three slots throughout: a dump's manifest indexes by slot, and a
one-camera test cannot tell a correct index from a hardcoded 0.
"""
from __future__ import annotations

import json
import threading

import cv2
import numpy as np
import pytest

from opendarts.capture.frame_dump import (
    DUMP_SCHEMA,
    FRAMES_FILENAME,
    KIND_MISSCORE,
    KIND_MISSED_DART,
    MANIFEST_FILENAME,
    DumpProgress,
    FrameDumpReader,
    FrameDumpWriter,
    encode_preview,
    verify_package_frames_in_dump,
    write_dump,
)
from opendarts.capture.frame_ring import FrameRing


def frame(slot: int, tick: int, *, h: int = 8, w: int = 12) -> np.ndarray:
    """Pixels that identify slot and tick, so a frame read back from the
    wrong offset fails rather than passing a shape check."""
    rng = np.random.default_rng(seed=slot * 100_000 + tick)
    arr = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    arr[0, 0] = (slot + 1, tick % 251, 7)
    return arr


def filled_ring(n_sets: int = 12, seconds: float = 30.0) -> FrameRing:
    ring = FrameRing(seconds)
    for i in range(n_sets):
        ring.append(
            {s: frame(s, i) for s in (0, 1, 2)},
            wall_s=1_757_000_000.0 + i * 0.03,
            monotonic_s=4321.0 + i * 0.03,
            generation=i,
        )
    return ring


# -- the file on disk ---------------------------------------------------


def test_a_dump_writes_a_manifest_and_one_frames_file(tmp_path):
    ring = filled_ring()
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSED_DART)
    manifest = json.loads((dest / MANIFEST_FILENAME).read_text())
    assert manifest["schema"] == DUMP_SCHEMA
    assert manifest["kind"] == KIND_MISSED_DART
    assert manifest["slots"] == [0, 1, 2]
    assert manifest["n_sets"] == 12
    assert manifest["n_frames"] == 36              # three per set, not one
    assert (dest / FRAMES_FILENAME).stat().st_size == manifest["bytes_total"]


def test_every_frame_reads_back_byte_identical(tmp_path):
    ring = filled_ring()
    expected = {
        (fs.generation, slot): arr
        for fs in ring.snapshot().sets for slot, arr in fs.frames.items()
    }
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    with FrameDumpReader(dest) as reader:
        seen = 0
        for fs, entry, arr in reader.iter_frames():
            original = expected[(fs["generation"], entry["slot"])]
            assert arr.dtype == original.dtype
            assert arr.shape == original.shape
            # BYTES, not a checksum and not a shape -- fidelity is the
            # entire premise of the feature.
            assert np.array_equal(arr, original)
            seen += 1
    assert seen == 36


def test_both_clocks_survive_the_round_trip_and_stay_separate(tmp_path):
    ring = filled_ring()
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    manifest = json.loads((dest / MANIFEST_FILENAME).read_text())
    first, last = manifest["sets"][0], manifest["sets"][-1]
    assert first["wall_s"] == pytest.approx(1_757_000_000.0)
    assert first["monotonic_s"] == pytest.approx(4321.0)
    # A build that collapsed them to one clock would make these equal.
    assert abs(first["wall_s"] - first["monotonic_s"]) > 1_000_000
    assert last["monotonic_s"] - first["monotonic_s"] == pytest.approx(11 * 0.03)


def test_a_repeated_array_is_stored_once_and_still_reads_back_for_both_sets(tmp_path):
    """When a slot's read fails the pump re-publishes the SAME array next
    cycle, so a stalled camera would otherwise store identical megabytes
    over and over. Both manifest entries must still read back correctly."""
    stalled = frame(1, 999)
    ring = FrameRing(30.0)
    for i in range(6):
        ring.append(
            {0: frame(0, i), 1: stalled, 2: frame(2, i)},
            wall_s=1_757_000_000.0 + i * 0.03,
            monotonic_s=4321.0 + i * 0.03,
            generation=i,
        )
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSED_DART)
    manifest = json.loads((dest / MANIFEST_FILENAME).read_text())

    slot1 = [e for fs in manifest["sets"] for e in fs["frames"] if e["slot"] == 1]
    assert len(slot1) == 6                              # every cycle recorded
    assert {e["offset"] for e in slot1} == {slot1[0]["offset"]}   # stored once
    assert [e["repeat_of_earlier_set"] for e in slot1] == [False] + [True] * 5
    # 18 frames indexed, 13 actually stored (6 slot-0 + 1 slot-1 + 6 slot-2).
    assert manifest["n_frames"] == 18
    assert (dest / FRAMES_FILENAME).stat().st_size == 13 * stalled.nbytes

    with FrameDumpReader(dest) as reader:
        for entry in slot1:
            assert np.array_equal(reader.read_frame(entry), stalled)


def test_an_empty_slice_is_refused_rather_than_written_as_an_empty_dump(tmp_path):
    ring = FrameRing(10.0)
    with pytest.raises(ValueError, match="refusing to write an empty frame dump"):
        write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    assert not (tmp_path / "d" / FRAMES_FILENAME).exists()


def test_the_reason_travels_onto_disk_with_the_data(tmp_path):
    """A capture whose first half was clipped must carry that on disk, not
    only in a log line on a rig nobody is watching."""
    ring = filled_ring()
    dest = write_dump(
        ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE,
        reason="the window starts before the oldest frame the ring still holds",
        extra={"trigger": "oracle"},
    )
    manifest = json.loads((dest / MANIFEST_FILENAME).read_text())
    assert "oldest frame" in manifest["reason"]
    assert manifest["extra"]["trigger"] == "oracle"


def test_progress_reports_bytes_written_against_bytes_total(tmp_path):
    ring = filled_ring()
    progress = DumpProgress(job_id="j", kind=KIND_MISSCORE, dest_dir=str(tmp_path))
    write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE, progress=progress)
    assert progress.state == "done"
    assert progress.sets_written == 12
    assert progress.bytes_written == progress.bytes_total == 36 * frame(0, 0).nbytes
    d = progress.as_dict()
    # A real rate, measured on THIS disk -- not the reference 3.3 GB/s
    # repeated as though it were a property of the software.
    assert d["bytes_per_s"] is not None and d["bytes_per_s"] > 0


def test_a_reader_refuses_a_schema_it_does_not_understand(tmp_path):
    ring = filled_ring()
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    manifest = json.loads((dest / MANIFEST_FILENAME).read_text())
    manifest["schema"] = "frame-dump/v99"
    (dest / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="refusing to guess at the layout"):
        FrameDumpReader(dest)


def test_a_truncated_frames_file_raises_rather_than_serving_short_data(tmp_path):
    ring = filled_ring()
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    frames = dest / FRAMES_FILENAME
    frames.write_bytes(frames.read_bytes()[: frames.stat().st_size // 2])
    with FrameDumpReader(dest) as reader:
        with pytest.raises(IOError, match="short read"):
            for _ in reader.iter_frames():
                pass


# -- the writer thread --------------------------------------------------


def test_a_second_capture_while_one_is_writing_is_refused_with_a_reason(tmp_path):
    ring = filled_ring(n_sets=4)
    writer = FrameDumpWriter()
    gate = threading.Event()

    original = write_dump

    import opendarts.capture.frame_dump as frame_dump_module

    def _blocking(*args, **kwargs):
        gate.wait(timeout=5)
        return original(*args, **kwargs)

    frame_dump_module.write_dump = _blocking  # type: ignore[assignment]
    try:
        first = writer.submit(ring.snapshot(), tmp_path / "a", kind=KIND_MISSED_DART)
        assert first["ok"] is True
        second = writer.submit(ring.snapshot(), tmp_path / "b", kind=KIND_MISSCORE)
        assert second["ok"] is False
        assert "one capture at a time" in second["reason"]
        assert str(tmp_path / "a") in second["reason"]
        assert second["running"]["kind"] == KIND_MISSED_DART
    finally:
        gate.set()
        writer.join()
        frame_dump_module.write_dump = original  # type: ignore[assignment]

    assert (tmp_path / "a" / FRAMES_FILENAME).exists()
    assert not (tmp_path / "b").exists()
    assert writer.status()["refusals"] == 1
    # A THIRD submit once the first has finished must SUCCEED -- a refusal
    # that never clears would leave the rig unable to capture anything for
    # the rest of the session.
    third = writer.submit(ring.snapshot(), tmp_path / "c", kind=KIND_MISSCORE)
    assert third["ok"] is True
    writer.join()
    assert (tmp_path / "c" / FRAMES_FILENAME).exists()


def test_the_ring_is_resumed_even_when_the_write_dies(tmp_path):
    """The `finally` that matters. A writer thread that dies with the ring
    paused leaves the tap off for the rest of the session while every
    other surface reports health."""
    import opendarts.capture.frame_dump as frame_dump_module

    ring = filled_ring()
    writer = FrameDumpWriter()
    original = frame_dump_module.write_dump

    def _explode(*args, **kwargs):
        raise OSError("no space left on device")

    frame_dump_module.write_dump = _explode  # type: ignore[assignment]
    try:
        ring.pause()
        assert ring.paused is True
        writer.submit(ring.snapshot(), tmp_path / "d", kind=KIND_MISSED_DART,
                      pause_ring=ring)
        writer.join()
    finally:
        frame_dump_module.write_dump = original  # type: ignore[assignment]

    assert ring.paused is False
    status = writer.status()
    assert status["busy"] is False
    assert status["last"]["state"] == "failed"
    assert "no space left" in status["last"]["error"]
    # And the ring really works again afterwards, not just reports so.
    ring.append({0: frame(0, 99)}, wall_s=1_757_000_001.0, monotonic_s=4322.0,
                generation=99)
    assert ring.stats()["sets"] == 13


def test_writer_status_moves_through_busy_and_back_to_idle(tmp_path):
    ring = filled_ring()
    writer = FrameDumpWriter()
    assert writer.status()["busy"] is False
    assert writer.status()["last"] is None
    writer.submit(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    writer.join()
    status = writer.status()
    assert status["busy"] is False
    assert status["last"]["state"] == "done"
    assert status["last"]["mb_total"] == status["last"]["mb_written"]


# -- the integrity tripwire ---------------------------------------------


def _write_package(package_dir, frames: dict) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    for cam, arr in frames.items():
        assert cv2.imwrite(str(package_dir / f"cam{cam}_frame.png"), arr)


def test_integrity_passes_when_the_package_frame_is_in_the_dump(tmp_path):
    ring = filled_ring()
    sets = ring.snapshot().sets
    chosen = sets[5].frames
    _write_package(tmp_path / "pkg", chosen)
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    result = verify_package_frames_in_dump(tmp_path / "pkg", dest)
    assert result.ok is True
    assert result.checked == 3 and result.matched == 3


def test_integrity_fails_loudly_when_a_frame_was_mutated_in_between(tmp_path, caplog):
    """The one thing that would quietly invalidate the feature: something
    re-encoding or mutating a frame between capture and save. The check
    must call that out rather than passing on "close enough"."""
    ring = filled_ring()
    chosen = dict(ring.snapshot().sets[5].frames)
    tampered = chosen[1].copy()
    tampered[0, 0, 0] = (int(tampered[0, 0, 0]) + 1) % 256   # ONE byte
    chosen[1] = tampered
    _write_package(tmp_path / "pkg", chosen)
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    with caplog.at_level("ERROR"):
        result = verify_package_frames_in_dump(tmp_path / "pkg", dest)
    assert result.ok is False
    assert result.matched == 2 and result.checked == 3
    assert any("cam1" in d and "byte-identical" in d for d in result.details)
    assert any("integrity check FAILED" in r.message for r in caplog.records)


def test_integrity_says_so_when_the_package_has_no_frames_at_all(tmp_path):
    ring = filled_ring()
    (tmp_path / "pkg").mkdir()
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSCORE)
    result = verify_package_frames_in_dump(tmp_path / "pkg", dest)
    assert result.ok is False
    assert "nothing to check against" in " ".join(result.details)


# -- the watchable version ----------------------------------------------


def test_encode_preview_produces_a_playable_file_at_the_measured_rate(tmp_path):
    ring = filled_ring(n_sets=20)
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSED_DART)
    out = encode_preview(dest, 1, tmp_path / "preview.avi")
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        # Recorded at 0.03s spacing, so ~33.3fps -- measured from the
        # dump's own monotonic stamps, not a nominal 30 that would stretch
        # the settle.
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(33.3, abs=1.0)
    finally:
        cap.release()


def test_encode_preview_refuses_a_slot_the_dump_does_not_hold(tmp_path):
    ring = filled_ring()
    dest = write_dump(ring.snapshot(), tmp_path / "d", kind=KIND_MISSED_DART)
    with pytest.raises(ValueError, match="no frames for slot 7"):
        encode_preview(dest, 7, tmp_path / "x.avi")
