"""LifecycleDriver: JSONL log, commit evidence, self-disable on errors."""
from __future__ import annotations

import json

import numpy as np

from opendarts.lifecycle.driver import LifecycleDriver
from opendarts.lifecycle.state import Action, LifecycleConfig
from tests.test_lifecycle_state import CX, CY, Scene, board_mask


def test_driver_logs_and_saves_commit_evidence(tmp_path):
    masks = {0: board_mask()}
    drv = LifecycleDriver(
        log_dir=tmp_path,
        masks_provider=lambda: masks,
        config=LifecycleConfig(warmup_stable_frames=3),
        run_id="test",
        save_commit_frames=True,  # off by default: the throw package has the frames
    )
    s = Scene()
    for _ in range(6):
        drv.observe({0: s.render()})
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = [drv.observe({0: s.render()}) for _ in range(10)]
    assert sum(t.action is Action.COMMIT for t in ticks if t) == 1
    drv.close()

    lines = [json.loads(l) for l in (tmp_path / "lifecycle-test.jsonl").read_text().splitlines()]
    assert lines[0]["event"] == "lifecycle_built"
    commits = [l for l in lines if l.get("action") == "commit"]
    assert len(commits) == 1
    assert "legacy" not in commits[0]  # the shadow-mode side-by-side field is gone
    ev = tmp_path / "lifecycle-test" / "commit-001"
    assert (ev / "cam0_bg.png").exists() and (ev / "cam0_frame.png").exists()
    meta = json.loads((ev / "meta.json").read_text())
    assert meta["dart_index"] == 0 and meta["dart_cams"] == [0]
    assert drv.stats.commits == 1 and drv.stats.ticks == 16


def test_driver_waits_for_masks_and_rebuilds_on_recalibration(tmp_path):
    masks: dict = {}
    drv = LifecycleDriver(log_dir=tmp_path, masks_provider=lambda: masks, run_id="t2",
                          config=LifecycleConfig(warmup_stable_frames=2))
    s = Scene()
    assert drv.observe({0: s.render()}) is None
    masks[0] = board_mask()
    assert drv.observe({0: s.render()}) is not None
    lc1 = drv.lifecycle
    masks[0] = board_mask().copy()  # fresh array, identical content: not a recalibration
    drv.observe({0: s.render()})
    assert drv.lifecycle is lc1
    shifted = np.roll(board_mask(), 20, axis=1)  # different geometry: recalibration
    masks[0] = shifted
    drv.observe({0: s.render()})
    assert drv.lifecycle is not lc1
    lines = (tmp_path / "lifecycle-t2.jsonl").read_text().splitlines()
    assert sum('"lifecycle_built"' in l for l in lines) == 2


def test_driver_disables_itself_after_repeated_errors(tmp_path):
    drv = LifecycleDriver(log_dir=tmp_path, masks_provider=lambda: {0: board_mask()}, run_id="t3")
    drv.MAX_ERRORS = 3
    bad = {0: np.zeros((3,), dtype=np.uint8)}  # not an image
    for _ in range(5):
        assert drv.observe(bad) is None
    assert drv._disabled
    assert drv.stats.errors >= 3


def test_driver_periodic_summary_reports_rate_cost_actions_and_dropped_frames(tmp_path, caplog):
    import logging

    masks = {0: board_mask()}
    drv = LifecycleDriver(
        log_dir=tmp_path,
        masks_provider=lambda: masks,
        config=LifecycleConfig(warmup_stable_frames=3),
        run_id="test",
        save_commit_frames=False,
        summary_interval_s=0.0,  # summarise on every tick after the first
    )
    s = Scene()
    with caplog.at_level(logging.INFO, logger="opendarts.lifecycle.driver"):
        for i in range(6):
            drv.observe({0: s.render()}, dropped_frames_total=i)  # loop dropped one cycle per tick
        s.darts.append((CX - 2, CY - 60, 16, 100))
        for i in range(10):
            drv.observe({0: s.render()}, dropped_frames_total=5)
    drv.close()

    summaries = [r for r in caplog.records if "summary:" in r.getMessage()]
    assert summaries, [r.getMessage() for r in caplog.records]
    msg = summaries[-1].getMessage()
    assert msg.startswith("lifecycle ") and "summary:" in msg and "observe p50/p95/max=" in msg and "dropped_pump_cycles=" in msg
    assert "legacy=" not in msg

    lines = [json.loads(l) for l in (tmp_path / "lifecycle-test.jsonl").read_text().splitlines()]
    sums = [l for l in lines if l.get("event") == "summary"]
    assert sums
    assert sum(x["commits"] for x in sums) == 1
    assert sum(x["dropped_pump_cycles"] for x in sums) == 5  # 0 -> 5 across the warm ticks
    assert all(x["observe_ms"]["max"] >= x["observe_ms"]["p50"] for x in sums)


def test_old_lifecycle_logs_are_pruned_to_the_limits(tmp_path, monkeypatch):
    import os

    from opendarts.lifecycle import driver as driver_module

    monkeypatch.setattr(driver_module, "MAX_LOG_FILES", 3)
    monkeypatch.setattr(driver_module, "MAX_LOG_BYTES", 10_000)
    paths = []
    for n in range(6):
        p = tmp_path / f"lifecycle-2026091{n}-000000.jsonl"
        p.write_text("x" * 100)
        os.utime(p, (1_000 + n, 1_000 + n))
        paths.append(p)
    (tmp_path / "lifecycle-20260910-000000").mkdir()      # evidence of the oldest
    (tmp_path / "run_product.log").write_text("untouched")
    driver_module._prune_old_logs(tmp_path)
    assert sorted(p.name for p in tmp_path.glob("lifecycle-*.jsonl")) == [
        p.name for p in paths[3:]]
    assert not (tmp_path / "lifecycle-20260910-000000").exists()
    assert (tmp_path / "run_product.log").exists()

    monkeypatch.setattr(driver_module, "MAX_LOG_FILES", 100)
    monkeypatch.setattr(driver_module, "MAX_LOG_BYTES", 150)   # room for one
    driver_module._prune_old_logs(tmp_path, keep_path=paths[3])
    assert sorted(p.name for p in tmp_path.glob("lifecycle-*.jsonl")) == [
        paths[3].name, paths[5].name]
