"""The free-space floor, proven without filling a real disk.

Two writers grow a rig's disk without bound -- throw packages (~6.5 MB
each, one per scored throw) and frame-ring captures (one file per missed
dart or misscore, sized by the ring window). Both now ask
`opendarts.disk_space` first, and the free-space READING is injected --
`opendarts.disk_space.free_bytes` is looked up at call time, and the
throw-capture service takes its own `free_bytes_fn` -- so every number
below is a number this test chose.

THREE CAMERAS THROUGHOUT. The dump's estimated size is per frame and per
camera, and one camera would make the estimate and the per-set size the
same number -- which is exactly the arithmetic these tests are about.

What is asserted is CONTENT, not shape: the refusal must carry the free
space, the floor and (for a dump) the size of the write, because "aged
out"-style refusals that say only that they refused are what this project
keeps having to fix.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from opendarts import disk_space
from opendarts.capture.frame_dump import MANIFEST_FILENAME
from opendarts.capture.frame_ring import FrameRing
from opendarts.capture.throw_capture import ThrowCaptureService
from opendarts.engines.base import EngineResult
from opendarts.live import capture_daemon
from opendarts.live import config as live_config
from opendarts.pipeline import CameraCalibration

GB = disk_space.BYTES_PER_GB
CAMERAS = (0, 1, 2)


# -- the floor itself ---------------------------------------------------


def test_an_absent_or_zero_floor_means_the_default_of_five_gb():
    """`0` is what an operator types when they have not thought about
    this. A literal zero floor would be a guard that never fires while
    still looking configured."""
    assert disk_space.DEFAULT_MIN_FREE_DISK_GB == 5.0
    assert disk_space.resolve_floor_gb(None) == 5.0
    assert disk_space.resolve_floor_gb(0) == 5.0
    assert disk_space.resolve_floor_gb(0.0) == 5.0
    assert disk_space.resolve_floor_gb(12.5) == 12.5
    # Negative passes straight through -- that is the opt-out, not a typo
    # to be corrected into the default.
    assert disk_space.resolve_floor_gb(-1) == -1.0


def test_a_negative_floor_disables_the_guard_and_says_nothing_was_checked():
    """The documented opt-out. Not merely "ok": a caller reporting "space
    is fine" must be able to say whether anyone actually looked."""
    calls: list[Path] = []

    def _never(path: Path) -> int:
        calls.append(path)
        return 0

    check = disk_space.check_free_space(
        "/anywhere", floor_gb=-1, needed_bytes=999 * GB, free_bytes_fn=_never,
    )
    assert check.ok is True
    assert check.enabled is False
    assert calls == [], "a disabled guard must not even read the disk"
    assert "disabled" in (check.reason or "")
    assert "min_free_disk_gb=-1" in (check.reason or "")


def test_below_the_floor_the_check_quotes_the_free_space_and_the_floor():
    check = disk_space.check_free_space(
        "/data", floor_gb=5, free_bytes_fn=lambda _p: int(1.25 * GB),
    )
    assert check.ok is False
    assert check.free_bytes == int(1.25 * GB)
    assert check.floor_bytes == 5 * GB
    assert "1.25 GB" in check.reason
    assert "5.00 GB" in check.reason
    assert "/data" in check.reason


def test_a_write_that_would_cross_the_floor_is_refused_while_space_looks_fine():
    """The question is not "is there room now" but "is there room after".
    A 4.7 GB dump onto 5.1 GB of disk passes the first and ends the
    session with nothing."""
    check = disk_space.check_free_space(
        "/data", floor_gb=5, needed_bytes=int(4.7 * GB),
        free_bytes_fn=lambda _p: int(5.1 * GB),
    )
    assert check.ok is False
    assert check.free_after_bytes == int(5.1 * GB) - int(4.7 * GB)
    assert "5.10 GB" in check.reason      # what is free now
    assert "4.70 GB" in check.reason      # what this write costs
    assert "0.40 GB" in check.reason      # what would be left
    assert "5.00 GB" in check.reason      # the floor


def test_an_unreadable_disk_lets_the_write_through_and_says_it_could_not_check():
    """A guard that cannot read the disk must not be the reason a rig
    stops recording evidence."""
    check = disk_space.check_free_space(
        "/data", floor_gb=5, free_bytes_fn=lambda _p: None,
    )
    assert check.ok is True
    assert check.enabled is True
    assert check.free_bytes is None
    assert "could not be read" in check.reason
    assert check.as_dict()["free_gb"] is None


def test_free_bytes_walks_up_to_a_directory_that_exists(tmp_path):
    """Every caller asks about somewhere it has not created yet -- a
    throw's `<root>/<session>/<throw>`, a dump's timestamped directory."""
    deep = tmp_path / "packages" / "session-1" / "throw-007"
    assert not deep.exists()
    answer = disk_space.free_bytes(deep)
    assert isinstance(answer, int) and answer > 0


# -- the config key -----------------------------------------------------


def _config(tmp_path, payload) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload) + "\n")
    return path


def test_the_config_key_is_min_free_disk_gb_and_defaults_to_five(tmp_path):
    assert live_config.min_free_disk_gb(tmp_path / "absent.json") == 5.0
    assert live_config.min_free_disk_gb(_config(tmp_path, {})) == 5.0
    assert live_config.min_free_disk_gb(_config(tmp_path, {"min_free_disk_gb": 0})) == 5.0
    assert live_config.min_free_disk_gb(_config(tmp_path, {"min_free_disk_gb": 20})) == 20.0
    assert live_config.min_free_disk_gb(_config(tmp_path, {"min_free_disk_gb": -1})) == -1.0


def test_a_malformed_floor_reads_as_the_default_not_as_off(tmp_path):
    """The failure direction that matters is a rig filling its disk, not
    a rig refusing to write one package. `true` is rejected explicitly --
    bool is an int subclass, so it would otherwise mean a 1 GB floor."""
    assert live_config.min_free_disk_gb(_config(tmp_path, {"min_free_disk_gb": "lots"})) == 5.0
    assert live_config.min_free_disk_gb(_config(tmp_path, {"min_free_disk_gb": True})) == 5.0
    bad = tmp_path / "broken.json"
    bad.write_text("{ not json")
    assert live_config.min_free_disk_gb(bad) == 5.0


def test_the_example_config_documents_the_key():
    root = Path(__file__).resolve().parent.parent
    example = json.loads((root / "config.example.json").read_text())
    assert example["min_free_disk_gb"] == 5
    readme = " ".join(example["_readme"])
    assert "min_free_disk_gb" in readme
    assert "NEGATIVE value disables the guard" in readme


def test_both_writers_are_handed_the_same_floor_from_one_reader():
    """A rig must never refuse one writer and allow the other on the same
    disk, which is what two independent readers would eventually do."""
    from opendarts.live import run_product

    source = Path(run_product.__file__).read_text()
    # One call site for the throw-capture service (built once for the
    # process), one per capture-loop session (re-read so an operator
    # editing data/config.json does not need a restart).
    assert source.count("min_free_disk_gb=_read_min_free_disk_gb()") == 2
    assert "ring, min_free_disk_gb=_read_min_free_disk_gb()" in source
    # And both resolve through the ONE config interpreter.
    import inspect as _inspect
    assert "min_free_disk_gb" in _inspect.getsource(run_product._read_min_free_disk_gb)
    assert run_product._read_min_free_disk_gb() == live_config.min_free_disk_gb()


# -- throw packages -----------------------------------------------------


@pytest.fixture(autouse=True)
def _forget_which_sessions_have_been_reported():
    capture_daemon._DISK_FLOOR_REPORTED_SESSIONS.clear()
    yield
    capture_daemon._DISK_FLOOR_REPORTED_SESSIONS.clear()


def _calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros((5, 1)),
        rvec=np.zeros((3, 1)),
        tvec=np.zeros((3, 1)),
        pnp_result=None,
        landmark_spread_ok=True,
    )


class _FakeEngine:
    """The engine surface handle_ready_to_capture() uses, and no more --
    the same stand-in tests/test_capture_daemon.py uses, so the real
    save/emit wiring is exercised against synthetic frames."""

    def score(self, bg_images, frame_images, calibrations) -> EngineResult:
        return EngineResult(
            ok=True, sector="20", ring="treble", board_xy_mm=(3.5, 101.2),
            reason="", diagnostics={"max_ray_disagreement_mm": 2.75,
                                    "n_cameras_used": 3},
        )


def _throw(
    tmp_path, monkeypatch, *, session: str, free_gb: float | None,
    floor_gb: "float | None" = None,
) -> tuple[Path, list[dict]]:
    """One REAL handle_ready_to_capture() call, three cameras, with only
    the engine and the free-space reading faked."""
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: _FakeEngine())
    if free_gb is not None:
        monkeypatch.setattr(
            disk_space, "free_bytes", lambda _path: int(free_gb * GB)
        )
    trigger = capture_daemon.ThrowTriggerState(
        state=capture_daemon.ThrowState.READY_TO_CAPTURE,
        dart_count=1,
        last_frame={c: np.full((8, 8, 3), 255, dtype=np.uint8) for c in CAMERAS},
    )
    events: list[dict] = []
    dest_dir = capture_daemon.handle_ready_to_capture(
        trigger,
        {c: np.zeros((8, 8, 3), dtype=np.uint8) for c in CAMERAS},
        {c: _calibration() for c in CAMERAS},
        tmp_path / "packages",
        session,
        on_event=events.append,
        visit_id="visit_1700000000000",
        visit_index=0,
        background_save=False,
        min_free_disk_gb=floor_gb,
    )
    return dest_dir, events


def _wait_for_other_engines(dest_dir: Path) -> None:
    """The also-run roster writes into result.json on a background thread.
    Let it land so teardown never races it."""
    import time

    for _ in range(200):
        try:
            if "other_engines" in json.loads((dest_dir / "result.json").read_text()):
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.02)


def test_above_the_floor_the_package_is_written_exactly_as_today(tmp_path, monkeypatch):
    dest_dir, events = _throw(
        tmp_path, monkeypatch, session="session-roomy", free_gb=500.0
    )
    _wait_for_other_engines(dest_dir)

    assert dest_dir.is_dir()
    meta = json.loads((dest_dir / "meta.json").read_text())
    assert meta["session"] == "session-roomy"
    assert meta["frame_cameras"] == [0, 1, 2]
    result = json.loads((dest_dir / "result.json").read_text())
    assert result["sector"] == "20"
    assert result["ring"] == "treble"
    # The frames themselves, one clip per camera -- the whole reason a
    # package costs real disk and the whole reason this guard exists.
    for cam in CAMERAS:
        assert (dest_dir / meta["video"]["cameras"][str(cam)]["clip"]).is_file()
    saved = [e for e in events if e["type"] == "PACKAGE_SAVED"]
    assert saved and saved[0]["path"] == str(dest_dir)


def test_below_the_floor_the_package_is_skipped_but_the_throw_still_scores(
    tmp_path, monkeypatch,
):
    """Scoring must not stop. THROW_DETECTED carries the whole scored
    throw and it is what the live board and match history are fed from --
    neither reads a package."""
    dest_dir, events = _throw(
        tmp_path, monkeypatch, session="session-full", free_gb=1.25
    )

    assert not dest_dir.exists(), "no half-made package directory left behind"
    assert not (tmp_path / "packages" / "session-full").exists()

    detected = [e for e in events if e["type"] == "THROW_DETECTED"]
    assert len(detected) == 1
    assert detected[0]["ok"] is True
    assert detected[0]["sector"] == "20"
    assert detected[0]["ring"] == "treble"
    assert detected[0]["board_xy_mm"] == [3.5, 101.2]
    assert detected[0]["visit_id"] == "visit_1700000000000"
    # Nothing was saved, so nothing may claim it was.
    assert [e for e in events if e["type"] == "PACKAGE_SAVED"] == []


def test_the_skip_is_logged_once_per_session_not_once_per_throw(
    tmp_path, monkeypatch, caplog,
):
    """A line per dart is how the one message that mattered gets scrolled
    past. Three throws, one loud line -- and it carries the numbers."""
    caplog.set_level(logging.DEBUG, logger="opendarts.capture_daemon")
    for _ in range(3):
        _throw(tmp_path, monkeypatch, session="session-noisy", free_gb=1.25)

    loud = [
        r for r in caplog.records
        if r.levelno >= logging.ERROR and "DISK FLOOR REACHED" in r.getMessage()
    ]
    assert len(loud) == 1, f"expected one loud line, got {len(loud)}"
    message = loud[0].getMessage()
    assert "session-noisy" in message
    assert "1.25 GB" in message           # what is free
    assert "5.00 GB" in message           # the floor
    assert "min_free_disk_gb" in message  # how to change it
    assert "unaffected" in message        # and that scoring did not stop

    # The later ones are still countable, just not at ERROR.
    quiet = [r for r in caplog.records if r.levelno == logging.DEBUG
             and "disk floor: skipping throw package" in r.getMessage()]
    assert len(quiet) == 2

    # A DIFFERENT session gets its own loud line -- "once" is per session,
    # not once for the life of the process.
    _throw(tmp_path, monkeypatch, session="session-later", free_gb=1.25)
    loud = [r for r in caplog.records
            if r.levelno >= logging.ERROR and "DISK FLOOR REACHED" in r.getMessage()]
    assert len(loud) == 2


def test_a_negative_floor_writes_the_package_on_a_disk_that_is_nearly_full(
    tmp_path, monkeypatch,
):
    """The opt-out reaches the package writer, not only the check."""
    dest_dir, events = _throw(
        tmp_path, monkeypatch, session="session-optout", free_gb=0.01, floor_gb=-1,
    )
    _wait_for_other_engines(dest_dir)
    assert dest_dir.is_dir()
    assert json.loads((dest_dir / "result.json").read_text())["sector"] == "20"
    saved = [e for e in events if e["type"] == "PACKAGE_SAVED"]
    assert saved and saved[0]["path"] == str(dest_dir)


# -- frame-ring dumps ---------------------------------------------------


def _frame(slot: int, tick: int) -> np.ndarray:
    rng = np.random.default_rng(seed=slot * 100_000 + tick)
    arr = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    arr[0, 0] = (slot + 1, tick % 251, 3)
    return arr


def _filled_ring(n: int = 30, wall0: float = 1_757_000_000.0) -> FrameRing:
    ring = FrameRing(30.0)
    for i in range(n):
        ring.append(
            {s: _frame(s, i) for s in CAMERAS},
            wall_s=wall0 + i / 30.0,
            monotonic_s=4321.0 + i / 30.0,
            generation=i,
        )
    return ring


def _service(tmp_path, ring, *, free_bytes: "int | None", floor_gb=None):
    return ThrowCaptureService(
        ring,
        capture_root=tmp_path / "captures",
        min_free_disk_gb=floor_gb,
        free_bytes_fn=lambda _path: free_bytes,
    )


def test_above_the_floor_a_dump_proceeds_exactly_as_today(tmp_path):
    ring = _filled_ring(n=30)
    svc = _service(tmp_path, ring, free_bytes=500 * GB)
    result = svc.capture_missed_dart(reason="nothing registered")
    assert result["ok"] is True
    svc.writer.join()

    manifest = json.loads((Path(result["job"]["dest_dir"]) / MANIFEST_FILENAME).read_text())
    assert manifest["kind"] == "missed_dart"
    assert manifest["n_sets"] == 30
    assert manifest["n_frames"] == 90            # three cameras per set
    assert ring.paused is False


def test_below_the_floor_a_ring_dump_is_refused_with_the_numbers(tmp_path):
    ring = _filled_ring(n=30)
    svc = _service(tmp_path, ring, free_bytes=int(1.25 * GB))
    result = svc.capture_misscore(1_757_000_000.5, reason="called T20, was S20")

    assert result["ok"] is False
    assert "1.25 GB" in result["reason"]
    assert "5.00 GB" in result["reason"]
    assert "min_free_disk_gb" in result["reason"]
    assert result["disk"]["free_bytes"] == int(1.25 * GB)
    assert result["disk"]["floor_bytes"] == 5 * GB
    # The estimate is the slice's own size: three cameras of 8x12x3.
    assert result["estimated_bytes"] == result["disk"]["needed_bytes"] > 0
    # Refused, therefore nothing on disk and no empty directory.
    assert not (tmp_path / "captures").exists()
    # The ring is untouched and still usable for the next attempt.
    assert ring.paused is False
    assert result["ring"]["sets"] == 30


def test_a_dump_whose_own_size_would_cross_the_floor_is_refused(tmp_path):
    """Current free space is ABOVE the floor; the dump is refused anyway,
    because the ring knows how many bytes it is about to write."""
    ring = _filled_ring(n=30)
    dump_bytes = ring.snapshot().nbytes
    assert dump_bytes == 30 * 3 * 8 * 12 * 3, "three cameras, 30 sets, 8x12x3"

    just_above = 5 * GB + dump_bytes // 2
    svc = _service(tmp_path, ring, free_bytes=just_above)
    refused = svc.capture_missed_dart(reason="nothing registered")
    assert refused["ok"] is False
    assert refused["estimated_bytes"] == dump_bytes
    assert "would leave" in refused["reason"]
    assert not (tmp_path / "captures").exists()
    # AND THE RING IS RUNNING AGAIN. The missed-dart path pauses before
    # it snapshots; a refusal that left the tap off would turn a disk
    # problem into a rig that silently records nothing.
    assert ring.paused is False

    # The same ring, the same dump, with room for it: allowed.
    ok_svc = _service(tmp_path, ring, free_bytes=5 * GB + dump_bytes * 2)
    allowed = ok_svc.capture_missed_dart(reason="nothing registered")
    assert allowed["ok"] is True
    ok_svc.writer.join()


def test_a_negative_floor_dumps_onto_a_disk_that_is_nearly_full(tmp_path):
    ring = _filled_ring(n=30)
    svc = _service(tmp_path, ring, free_bytes=1, floor_gb=-1)
    result = svc.capture_missed_dart(reason="nothing registered")
    assert result["ok"] is True
    svc.writer.join()
    manifest = json.loads((Path(result["job"]["dest_dir"]) / MANIFEST_FILENAME).read_text())
    assert manifest["n_frames"] == 90


def test_a_refused_dump_is_refused_every_time_it_is_asked(tmp_path):
    """A refusal that only fires once is a refusal that stops being
    reported -- the same rule the disabled-ring refusal already follows."""
    ring = _filled_ring(n=30)
    svc = _service(tmp_path, ring, free_bytes=int(0.5 * GB))
    for _ in range(3):
        assert svc.capture_missed_dart(reason="x")["ok"] is False
    assert svc.capture_misscore(1_757_000_000.5, reason="x")["ok"] is False
    assert not (tmp_path / "captures").exists()
