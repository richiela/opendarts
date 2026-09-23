"""LifecycleSettingsStore, driver config hot-swap, and `lifecycle_settings`
on the config document.

The Detection-speed control had a route pair of its own
(`GET`/`POST /api/detection-time`) until 2026-09-17. Its coverage moved
onto `GET`/`PATCH /api/config`, where the setting is one nested key.
"""
from __future__ import annotations

import json

import pytest

from opendarts.lifecycle.driver import LifecycleDriver
from opendarts.lifecycle.settings import (
    DART_STABLE_FRAMES_MAX,
    DART_STABLE_FRAMES_MIN,
    LifecycleSettingsStore,
)
from opendarts.lifecycle.state import DEFAULT_CONFIG
from tests.test_lifecycle_state import Scene, board_mask


def test_store_defaults_and_identity():
    store = LifecycleSettingsStore()
    cfg = store.get()
    assert cfg is DEFAULT_CONFIG
    assert cfg.dart_stable_frames == DEFAULT_CONFIG.dart_stable_frames
    assert store.get() is cfg
    assert store.meta() == {
        "dart_stable_frames": DEFAULT_CONFIG.dart_stable_frames,
        "min": DART_STABLE_FRAMES_MIN,
        "max": DART_STABLE_FRAMES_MAX,
    }


def test_set_replaces_only_dart_stable_frames():
    store = LifecycleSettingsStore()
    before = store.get()
    after = store.set(dart_stable_frames=4)
    assert after is not before
    assert after is store.get()
    assert after.dart_stable_frames == 4
    for k, v in vars(before).items():
        if k == "dart_stable_frames":
            continue
        assert getattr(after, k) == v


def test_set_none_only_is_noop():
    store = LifecycleSettingsStore()
    cfg = store.get()
    assert store.set() is cfg
    assert store.set(dart_stable_frames=None) is cfg


@pytest.mark.parametrize("bad", [0, 6, "3"])
def test_set_rejects_out_of_range_and_non_int(bad):
    store = LifecycleSettingsStore()
    before = store.get()
    with pytest.raises(ValueError):
        store.set(dart_stable_frames=bad)
    assert store.get() is before


def test_snapshot_round_trip(tmp_path):
    path = tmp_path / "latest.json"
    store = LifecycleSettingsStore(snapshot_path=path)
    store.set(dart_stable_frames=4)
    raw = json.loads(path.read_text())
    assert raw == {"lifecycle_settings": {"dart_stable_frames": 4}}

    store2 = LifecycleSettingsStore(snapshot_path=path)
    assert store2.get().dart_stable_frames == 4
    assert store2.get() is not DEFAULT_CONFIG


def test_corrupt_snapshot_falls_back_without_raising(tmp_path, caplog):
    path = tmp_path / "latest.json"
    path.write_text("{not json")
    store = LifecycleSettingsStore(snapshot_path=path)
    assert store.get() is DEFAULT_CONFIG
    assert store.get().dart_stable_frames == DEFAULT_CONFIG.dart_stable_frames
    assert any("unreadable" in r.message for r in caplog.records)


def test_out_of_range_snapshot_falls_back_without_raising(tmp_path, caplog):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps({"lifecycle_settings": {"dart_stable_frames": 9}}))
    store = LifecycleSettingsStore(snapshot_path=path)
    assert store.get() is DEFAULT_CONFIG
    assert any("out-of-range" in r.message for r in caplog.records)


def test_driver_hot_swaps_config_without_rebuild(tmp_path):
    store = LifecycleSettingsStore()
    masks = {0: board_mask()}
    driver = LifecycleDriver(
        log_dir=tmp_path,
        masks_provider=lambda: masks,
        config_provider=store.get,
        run_id="hotswap",
        save_commit_frames=False,
    )
    driver.observe({0: Scene().render()})
    lc = driver.lifecycle
    assert lc is not None
    assert lc.cfg.dart_stable_frames == DEFAULT_CONFIG.dart_stable_frames

    store.set(dart_stable_frames=5)
    driver.observe({0: Scene().render()})
    assert driver.lifecycle is lc
    assert driver.lifecycle.cfg.dart_stable_frames == 5
    assert driver.cfg is store.get()
    assert driver.cfg.dart_stable_frames == 5

    lines = [json.loads(l) for l in (tmp_path / "lifecycle-hotswap.jsonl").read_text().splitlines()]
    changed = [l for l in lines if l.get("event") == "config_changed"]
    assert len(changed) == 1
    assert changed[0]["config"]["dart_stable_frames"] == 5
    driver.close()


pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from opendarts.live.server import create_app


def test_config_get_reflects_the_live_store(tmp_path):
    """The document reports the value IN FORCE, which for this key lives
    in the store, not the file -- the two differ for the whole life of a
    process the moment anyone changes it."""
    store = LifecycleSettingsStore()
    app = create_app(
        package_root=tmp_path, enable_background_poll=False, lifecycle_settings_store=store
    )
    body = TestClient(app).get("/api/config").json()
    assert body["ok"] is True
    assert body["config"]["lifecycle_settings"] == {
        "dart_stable_frames": DEFAULT_CONFIG.dart_stable_frames
    }
    # The bounds are not settings, so they ride in the runtime block.
    assert body["runtime"]["lifecycle_settings"]["min"] == 1
    assert body["runtime"]["lifecycle_settings"]["max"] == 5


def test_patching_it_updates_the_store_and_state(tmp_path):
    store = LifecycleSettingsStore()
    app = create_app(
        package_root=tmp_path, enable_background_poll=False, lifecycle_settings_store=store
    )
    client = TestClient(app)
    body = client.patch(
        "/api/config", json={"lifecycle_settings": {"dart_stable_frames": 4}}
    ).json()
    assert body["ok"] is True
    assert body["config"]["lifecycle_settings"]["dart_stable_frames"] == 4
    # Applies on the next frame via the driver's config_provider hot-swap,
    # so it is live, not restart-required.
    assert body["applied_live"] == ["lifecycle_settings"]
    assert body["restart_required"] == []
    assert store.get().dart_stable_frames == 4

    # /api/state keeps carrying what it always carried -- the dashboard
    # polls it, and the config rewrite moved no state onto the document.
    state = client.get("/api/state").json()
    assert state["lifecycle_settings"]["dart_stable_frames"] == 4
    assert state["lifecycle_settings"]["min"] == 1
    assert state["lifecycle_settings"]["max"] == 5


def test_patching_out_of_range_is_refused_without_500(tmp_path):
    store = LifecycleSettingsStore()
    app = create_app(
        package_root=tmp_path, enable_background_poll=False, lifecycle_settings_store=store
    )
    client = TestClient(app)
    resp = client.patch(
        "/api/config", json={"lifecycle_settings": {"dart_stable_frames": 9}}
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert body["errors"]["lifecycle_settings"] == (
        "dart_stable_frames must be an int in [1, 5], got 9"
    )
    assert store.get().dart_stable_frames == DEFAULT_CONFIG.dart_stable_frames


def test_an_unknown_sub_key_is_refused_rather_than_written(tmp_path):
    """A nested group is merged, so a typo inside it would otherwise be
    persisted forever as a key nothing reads."""
    store = LifecycleSettingsStore()
    app = create_app(
        package_root=tmp_path, enable_background_poll=False, lifecycle_settings_store=store
    )
    resp = TestClient(app).patch(
        "/api/config", json={"lifecycle_settings": {"dart_stabel_frames": 3}}
    )
    assert resp.status_code == 400
    assert resp.json()["errors"]["lifecycle_settings"] == (
        "lifecycle_settings has no key 'dart_stabel_frames'"
    )


def test_without_a_store_it_is_saved_but_reported_as_not_applied(tmp_path):
    """A process with no capture loop still owns the config file, so the
    value persists -- but nothing in THIS process read it, and the reply
    says which of the two happened rather than implying both."""
    app = create_app(package_root=tmp_path, enable_background_poll=False)
    client = TestClient(app)
    body = client.patch(
        "/api/config", json={"lifecycle_settings": {"dart_stable_frames": 3}}
    ).json()
    assert body["ok"] is True
    assert body["persisted"] == ["lifecycle_settings"]
    assert body["applied_live"] == []
    assert "no capture loop in this process" in body["notes"]["lifecycle_settings"]
    assert body["restart_required"] == ["lifecycle_settings"]
    # The document still reports the persisted value, which IS what the
    # next start will use.
    assert body["config"]["lifecycle_settings"] == {"dart_stable_frames": 3}
    assert client.get("/api/state").json()["lifecycle_settings"] is None


def test_dashboard_html_has_detection_time_select(tmp_path):
    html = TestClient(create_app(package_root=tmp_path, enable_background_poll=False)).get("/").text
    assert 'id="idle-timeout-select"' in html
    assert 'id="detection-time-select"' in html
    assert "patchConfig({lifecycle_settings: settings}" in html
    assert "const settings = {dart_stable_frames: frames};" in html
    assert "renderDetectionTimeSelect" in html
