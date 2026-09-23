"""The config document itself -- `GET`/`PATCH /api/config`.

The per-key behaviour ported from each retired route lives beside that
key's other tests (the port in test_port_control_and_host_config.py,
Recording in test_store_packages_control.py, the ring in
test_frame_ring_api.py, the camera assignment / Autodarts / diagnostics /
idle timeout in test_live_server.py). What is here is the CONTRACT that
is new: one document, all-or-nothing writes, a registry that says what a
key is, and a reply that distinguishes "saved" from "in force".
"""
from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from opendarts.live import config as config_mod  # noqa: E402
from opendarts.live.config_document import (  # noqa: E402
    CONFIG_KEYS,
    KEYS_BY_NAME,
    WRITABLE_KEYS,
    ConfigValueError,
    effective_document,
    validate_patch,
)
from opendarts.live.server import create_app  # noqa: E402


def cfg_path():
    """Read through the MODULE, never a `from ... import` binding.

    tests/conftest.py's autouse `_isolate_repo_paths` repoints this
    attribute per test; a name bound at test-module import time would
    still hold the session sandbox's path and every assertion here would
    be reading a file nothing wrote.
    """
    return config_mod.DEFAULT_CONFIG_PATH


@pytest.fixture
def client(tmp_path):
    """A plain app on this test's own config file."""
    cfg_path().parent.mkdir(parents=True, exist_ok=True)
    app = create_app(package_root=tmp_path / "packages", port=8420,
                     host="0.0.0.0", enable_background_poll=False)
    return TestClient(app)


def saved() -> dict:
    path = cfg_path()
    return json.loads(path.read_text()) if path.exists() else {}


# ---------------------------------------------------------------------------
# GET: the whole document, valued at what is in force
# ---------------------------------------------------------------------------

def test_get_carries_every_registry_key_with_its_code_default(client):
    """"Every key the product reads" is the promise. A key the product
    reads but the document omits is a setting nobody can discover."""
    body = client.get("/api/config").json()
    assert body["ok"] is True
    assert list(body["config"]) == [k.name for k in CONFIG_KEYS]
    # Real values, not placeholders: these are the code defaults an empty
    # config file resolves to.
    assert body["config"]["port"] == 8420
    assert body["config"]["host"] == "0.0.0.0"
    assert body["config"]["ad_enabled"] is False
    assert body["config"]["store_packages"] is True
    assert body["config"]["frame_ring_seconds"] == 20.0
    assert body["config"]["video_record_mode"] == "all"
    assert body["config"]["min_free_disk_gb"] == 5.0
    assert body["config"]["cv2_num_threads"] == 1
    assert body["config"]["idle_timeout_sec"] == 900
    assert body["config"]["lifecycle_settings"] == {"dart_stable_frames": 2}
    assert body["config"]["engine_config"]["primary"] == "Zeus"
    assert body["config"]["diagnostics"] == {"enabled": False}
    assert body["config"]["always_update"] is False
    assert body["config"]["update_on_next_restart"] is False
    assert body["config"]["capabilities"] is None


def test_get_reports_the_files_value_where_the_file_has_a_usable_one(client):
    cfg_path().write_text(json.dumps({
        "port": 9000, "store_packages": False, "cv2_num_threads": -1,
        "camera_resolutions": {"0": "auto", "1": "1920x1080"},
    }))
    config = client.get("/api/config").json()["config"]
    assert config["port"] == 9000
    assert config["store_packages"] is False
    assert config["cv2_num_threads"] == -1
    assert config["camera_resolutions"] == {"0": "auto", "1": "1920x1080"}


def test_a_file_value_the_loader_would_ignore_reads_as_the_default(client):
    """The document must never report a value the next launch will not
    use. `load_live_config()` drops an unusable key with a warning and
    carries on, so the effective value is the code default -- and this is
    the same validator, run in the same direction."""
    cfg_path().write_text(json.dumps({"port": "not a port", "host": "  "}))
    config = client.get("/api/config").json()["config"]
    assert config["port"] == 8420
    assert config["host"] == "0.0.0.0"


def test_the_document_round_trips_through_a_patch_unchanged(client):
    """PATCH(GET()) must be a no-op. It is the cheapest proof that the
    two halves agree about every key's shape -- a value the document
    reports that the document then refuses would be a contradiction, and
    an editor or a script will do exactly this."""
    document = client.get("/api/config").json()["config"]
    document.pop("capabilities")          # read-only, and refused by name
    resp = client.patch("/api/config", json=document)
    assert resp.status_code == 200, resp.json()
    assert resp.json()["config"] == client.get("/api/config").json()["config"]


# ---------------------------------------------------------------------------
# PATCH: refusals
# ---------------------------------------------------------------------------

def test_an_unknown_key_is_refused_by_name(client):
    """A typo must not become a key nobody reads, sitting in the
    operator's file forever looking like a setting."""
    cfg_path().write_text(json.dumps({"port": 8420}))
    resp = client.patch("/api/config", json={"prot": 9000})
    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert body["errors"] == {
        "prot": "unknown config key 'prot' -- this rig has no such setting"
    }
    assert saved() == {"port": 8420}


def test_an_invalid_value_is_refused_and_nothing_at_all_is_written(client):
    """ALL-OR-NOTHING. The request names three good keys and one bad one;
    a per-key loop that wrote as it went would leave the rig in a state
    nobody asked for and no reply could describe."""
    cfg_path().write_text(json.dumps({"port": 8420}))
    resp = client.patch("/api/config", json={
        "store_packages": False,
        "idle_timeout_sec": 60,
        "frame_ring_seconds": 12,
        "min_free_disk_gb": "lots",
    })
    assert resp.status_code == 400
    body = resp.json()
    assert body["errors"] == {
        "min_free_disk_gb": "min_free_disk_gb must be a number, got 'lots'"
    }
    # Not one of the three valid keys landed.
    assert saved() == {"port": 8420}
    # And the document reported back is the UNCHANGED one.
    assert body["config"]["store_packages"] is True
    assert body["config"]["idle_timeout_sec"] == 900


def test_every_bad_key_is_reported_not_just_the_first(client):
    """A form with three fields has to know which of them it got wrong."""
    resp = client.patch("/api/config", json={
        "port": 70000, "ad_base_url": "box:3180", "cv2_num_threads": 1.5,
    })
    assert resp.status_code == 400
    assert sorted(resp.json()["errors"]) == ["ad_base_url", "cv2_num_threads", "port"]


def test_capabilities_cannot_be_written(client):
    """It is written by the startup capability probe. Anything typed into
    it is overwritten at the next launch, so accepting a write here would
    be a control that silently undoes itself."""
    cfg_path().write_text(json.dumps({
        "capabilities": {"platform": "Darwin", "tools": {"git": {"present": True}}}
    }))
    resp = client.patch("/api/config", json={
        "capabilities": {"platform": "Windows", "tools": {}}
    })
    assert resp.status_code == 400
    assert resp.json()["errors"]["capabilities"] == (
        "capabilities is read-only: it is written by the startup probe, not by hand"
    )
    assert saved()["capabilities"]["platform"] == "Darwin"
    assert KEYS_BY_NAME["capabilities"].writable is False
    assert "capabilities" not in WRITABLE_KEYS
    # It is still REPORTED -- read-only is not the same as hidden.
    assert client.get("/api/config").json()["config"]["capabilities"]["platform"] == "Darwin"


def test_a_capabilities_write_is_refused_even_beside_valid_keys(client):
    """All-or-nothing applies to the read-only refusal too."""
    resp = client.patch("/api/config", json={"port": 9100, "capabilities": {}})
    assert resp.status_code == 400
    assert "port" not in saved()


def test_a_body_that_is_not_an_object_is_refused(client):
    resp = client.patch("/api/config", json=[{"port": 9000}])
    assert resp.status_code == 400
    assert "JSON object" in resp.json()["errors"]["_body"]


# ---------------------------------------------------------------------------
# PATCH: merging
# ---------------------------------------------------------------------------

def test_a_partial_patch_leaves_every_other_key_untouched(client):
    cfg_path().write_text(json.dumps({
        "port": 9000,
        "ad_base_url": "http://example.local:3180",
        "camera_devices": [4, 5, 6],
        "engine_config": {"primary": "Zeus", "also_run": ["Apollo"], "timeout_s": 7.0},
    }))
    body = client.patch("/api/config", json={"store_packages": False}).json()
    assert body["ok"] is True
    assert body["changed"] == ["store_packages"]

    after = saved()
    assert after["port"] == 9000
    assert after["ad_base_url"] == "http://example.local:3180"
    assert after["camera_devices"] == [4, 5, 6]
    assert after["engine_config"]["timeout_s"] == 7.0
    assert after["store_packages"] is False
    # ...and the document agrees with the file it just wrote.
    assert body["config"]["port"] == 9000
    assert body["config"]["engine_config"]["also_run"] == ["Apollo"]


def test_a_nested_group_is_merged_not_replaced(client):
    """`{"engine_config": {"timeout_s": 9}}` must not drop `primary`,
    which the loader requires whenever the section exists at all."""
    cfg_path().write_text(json.dumps({
        "engine_config": {"primary": "Zeus", "also_run": ["Talos"], "timeout_s": 5.0}
    }))
    body = client.patch("/api/config", json={"engine_config": {"timeout_s": 9.0}}).json()
    assert body["ok"] is True
    assert saved()["engine_config"] == {
        "primary": "Zeus", "also_run": ["Talos"], "timeout_s": 9.0
    }


def test_an_unregistered_engine_is_refused_by_name(client):
    resp = client.patch("/api/config", json={"engine_config": {"primary": "Nonesuch"}})
    assert resp.status_code == 400
    reason = resp.json()["errors"]["engine_config"]
    assert "'Nonesuch'" in reason and "Zeus" in reason
    assert "engine_config" not in saved()


# ---------------------------------------------------------------------------
# PATCH: saved vs in force
# ---------------------------------------------------------------------------

def test_a_restart_required_key_is_reported_as_such_and_does_not_apply_live(client):
    """`cv2_num_threads` is read while the process comes up. Saying
    "saved" and leaving it there would be a control that appears to have
    changed the running rig."""
    import cv2

    before = cv2.getNumThreads()
    body = client.patch("/api/config", json={"cv2_num_threads": 3}).json()
    assert body["ok"] is True
    assert body["persisted"] == ["cv2_num_threads"]
    assert body["applied_live"] == []
    assert body["restart_required"] == ["cv2_num_threads"]
    assert saved()["cv2_num_threads"] == 3
    assert cv2.getNumThreads() == before, "a startup-only key must not touch the process"


def test_a_live_apply_key_applies_immediately(client, tmp_path):
    """The other half of the same distinction, proven against the live
    object rather than the reply: the lifecycle store really holds the
    new value the moment the PATCH returns."""
    from opendarts.lifecycle.settings import LifecycleSettingsStore

    store = LifecycleSettingsStore()
    app = create_app(package_root=tmp_path / "pkgs", enable_background_poll=False,
                     lifecycle_settings_store=store)
    c = TestClient(app)

    body = c.patch("/api/config",
                   json={"lifecycle_settings": {"dart_stable_frames": 5}}).json()
    assert body["applied_live"] == ["lifecycle_settings"]
    assert body["restart_required"] == []
    assert store.get().dart_stable_frames == 5, (
        "the object the capture loop reads must already hold the new value"
    )


def test_one_patch_can_mix_a_live_key_and_a_restart_key(client):
    """The reply has to split them rather than give the request one
    verdict -- which is exactly what eight separate routes could never do
    for a change that spans two settings."""
    from opendarts.live import diagnostics_gate

    body = client.patch("/api/config", json={
        "port": 9400, "diagnostics": {"enabled": True},
    }).json()
    assert body["ok"] is True
    assert body["changed"] == ["diagnostics", "port"]
    assert body["applied_live"] == ["diagnostics"]
    assert body["restart_required"] == ["port"]
    assert diagnostics_gate.enabled() is True
    assert saved()["port"] == 9400


def test_the_new_update_flags_validate_and_persist_like_any_other_key(client):
    """`always_update` / `update_on_next_restart` are read by the LAUNCHER
    (run.sh / run.ps1), so nothing in this process applies them -- but
    they are keys of the same file and go through the same door."""
    body = client.patch("/api/config", json={"always_update": True}).json()
    assert body["ok"] is True
    assert body["config"]["always_update"] is True
    assert saved()["always_update"] is True
    assert body["restart_required"] == []

    resp = client.patch("/api/config", json={"update_on_next_restart": "yes"})
    assert resp.status_code == 400
    assert resp.json()["errors"]["update_on_next_restart"] == (
        "update_on_next_restart must be true or false, got 'yes'"
    )
    assert "update_on_next_restart" not in saved()


def test_state_still_carries_what_it_carried_before(client):
    """The rewrite moved SETTINGS onto the document. It moved no state:
    the dashboard polls /api/state for the live picture and would go
    blank if any of this had followed the settings out."""
    state = client.get("/api/state").json()
    for key in ("capture_loop", "calibration", "diagnostics", "lifecycle_settings",
                "engine_config", "config"):
        assert key in state, f"/api/state stopped carrying {key}"
    assert state["diagnostics"] == {"enabled": False}


# ---------------------------------------------------------------------------
# The registry itself
# ---------------------------------------------------------------------------

def test_every_key_says_what_it_is():
    """The one-line doc is what the API table and the template are
    written from; a blank one is a key nobody can explain."""
    for spec in CONFIG_KEYS:
        assert spec.doc.strip(), f"{spec.name} has no description"
        assert spec.name == spec.name.lower()


def test_exactly_one_key_is_not_persisted_and_exactly_one_is_read_only():
    not_persisted = [k.name for k in CONFIG_KEYS if not k.persist]
    read_only = [k.name for k in CONFIG_KEYS if not k.writable]
    assert not_persisted == ["diagnostics"], (
        "a key that applies live but is never remembered is a deliberate, "
        "documented exception -- adding another one is a decision"
    )
    assert read_only == ["capabilities"]


def test_validate_patch_is_all_or_nothing_without_a_server():
    cleaned, errors = validate_patch({"port": 9000, "store_packages": "yes"})
    assert errors == {"store_packages": "store_packages must be true or false, got 'yes'"}
    # `cleaned` still holds the good key -- it is the CALLER that must not
    # apply anything while `errors` is non-empty, and that is what the
    # route does.
    assert cleaned == {"port": 9000}


def test_every_documented_key_appears_in_the_live_api_doc():
    """`docs/LIVE_API.md` is the published surface. A key the document
    returns but the doc never names cannot be deprecated, because it was
    never announced."""
    from pathlib import Path

    doc = (Path(__file__).resolve().parent.parent / "docs" / "LIVE_API.md").read_text()
    missing = [k.name for k in CONFIG_KEYS if f"`{k.name}`" not in doc
               and f"`{k.name}." not in doc and f"{k.name}`" not in doc]
    assert not missing, f"undocumented config keys: {missing}"


def test_the_effective_document_needs_no_running_server():
    """It is an ordinary function over the config file, so a script or a
    test can ask what a rig is configured to do without building an app."""
    cfg_path().parent.mkdir(parents=True, exist_ok=True)
    cfg_path().write_text(json.dumps({"idle_timeout_sec": 30}))
    doc = effective_document()
    assert doc["idle_timeout_sec"] == 30
    assert doc["store_packages"] is True


def test_a_validator_raises_the_error_type_the_route_catches():
    """A validator raising anything else would reach FastAPI as a 500 and
    the operator would see no reason at all."""
    with pytest.raises(ConfigValueError):
        KEYS_BY_NAME["port"].validate(0, {})
