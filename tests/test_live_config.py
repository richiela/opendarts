"""tests/test_live_config.py -- opendarts.live.config, the machine-local
config file for port / per-camera reprojection targets / AD base URL
. Real file
I/O against tmp_path, never the real data/config.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

from opendarts.live.config import LiveConfig, load_live_config


def test_load_live_config_missing_file_returns_all_none_empty(tmp_path):
    cfg = load_live_config(tmp_path / "does_not_exist.json")
    assert cfg.port is None
    assert cfg.ad_base_url is None
    assert cfg.reprojection_targets_px == {}
    assert cfg.camera_resolutions == {}


def test_load_live_config_reads_real_values(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "port": 9999,
        "ad_base_url": "http://example.local:3180",
        "reprojection_targets_px": {"0": 1.0, "1": 3.1, "2": 1.0},
    }))

    cfg = load_live_config(path)

    assert cfg.port == 9999
    assert cfg.ad_base_url == "http://example.local:3180"
    assert cfg.reprojection_targets_px == {0: 1.0, 1: 3.1, 2: 1.0}


def test_load_live_config_partial_file_leaves_other_fields_none(tmp_path):
    """Real use case: only override reprojection targets, leave port/AD
    URL at their code defaults."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"reprojection_targets_px": {"1": 3.1}}))

    cfg = load_live_config(path)

    assert cfg.port is None
    assert cfg.ad_base_url is None
    assert cfg.reprojection_targets_px == {1: 3.1}


def test_load_live_config_malformed_json_degrades_to_defaults_not_a_raise(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not valid json")

    cfg = load_live_config(path)

    assert cfg == LiveConfig()


def test_load_live_config_non_object_json_degrades_to_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps([1, 2, 3]))

    cfg = load_live_config(path)

    assert cfg == LiveConfig()


def test_load_live_config_bad_field_types_are_ignored_not_fatal(tmp_path):
    """A real config file with a typo'd value (string where an int is
    expected) must degrade that ONE field, not crash the whole load --
    same graceful-degrade posture as every other calibration/AD-
    reachability failure this project already has."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "port": "not-a-number",
        "ad_base_url": 12345, # not a string
        "reprojection_targets_px": {"not-a-cam": 1.0, "1": "not-a-float"},
    }))

    cfg = load_live_config(path)

    assert cfg.port is None
    assert cfg.ad_base_url is None
    assert cfg.reprojection_targets_px == {}


def test_load_live_config_reads_camera_resolutions_auto_and_explicit(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "camera_resolutions": {"0": "auto", "1": "1920x1080"},
    }))

    cfg = load_live_config(path)

    assert cfg.camera_resolutions == {0: None, 1: (1920, 1080)}


def test_load_live_config_camera_resolutions_bad_index_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "camera_resolutions": {"not-a-cam": "auto", "1": "1920x1080"},
    }))

    cfg = load_live_config(path)

    assert cfg.camera_resolutions == {1: (1920, 1080)}


def test_load_live_config_camera_resolutions_bad_value_type_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "camera_resolutions": {"0": 1920, "1": "1920x1080"}, # 0: not a string
    }))

    cfg = load_live_config(path)

    assert cfg.camera_resolutions == {1: (1920, 1080)}


def test_load_live_config_camera_resolutions_malformed_string_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "camera_resolutions": {"0": "not-a-resolution", "1": "1920x1080"},
    }))

    cfg = load_live_config(path)

    assert cfg.camera_resolutions == {1: (1920, 1080)}


def test_load_live_config_camera_resolutions_not_an_object_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_resolutions": ["auto"]}))

    cfg = load_live_config(path)

    assert cfg.camera_resolutions == {}



# ---------------------------------------------------------------------------
# camera_devices -- which hardware device feeds each camera slot.
# Added 2026-09-10: a laptop with four cameras puts its built-in webcam on
# index 0, so the hardcoded [0, 1, 2] device list made one board camera
# unreachable with no way to say otherwise.
# ---------------------------------------------------------------------------


def test_load_live_config_camera_devices_assigns_slots_in_order(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_devices": [1, 2, 3]}))
    assert load_live_config(path).camera_devices == [1, 2, 3]


def test_load_live_config_absent_camera_devices_is_none_not_a_default_list(tmp_path):
    """None means "no override" -- the hub then takes its OWN default
    branch. Returning [0, 1, 2] here instead would look identical today
    and silently pin the default at this layer forever."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"port": 8420}))
    assert load_live_config(path).camera_devices is None


def test_load_live_config_camera_devices_need_not_be_contiguous_or_sorted(tmp_path):
    """Slot order is the meaning, not device order: a rig may want slot 0
    on device 3. Nothing here should reorder or renumber."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_devices": [3, 0, 2]}))
    assert load_live_config(path).camera_devices == [3, 0, 2]


def test_load_live_config_camera_devices_duplicate_device_rejected(tmp_path):
    """One camera cannot be two views of the board. Accepting this would
    present live as "cam1 and cam2 see the same thing", which reads as a
    mounting problem rather than a config one."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_devices": [1, 1, 2]}))
    assert load_live_config(path).camera_devices is None


def test_load_live_config_camera_devices_rejects_bool_entries(tmp_path):
    """bool is an int subclass, so a bare isinstance(int) check would
    accept True as device index 1."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_devices": [True, 2, 3]}))
    assert load_live_config(path).camera_devices is None


def test_load_live_config_camera_devices_rejects_negative_and_non_int(tmp_path):
    for bad in ([-1, 2, 3], ["0", 1, 2], [1.5, 2, 3]):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"camera_devices": bad}))
        assert load_live_config(path).camera_devices is None, bad


def test_load_live_config_empty_camera_devices_ignored(tmp_path):
    """An empty list is not "zero cameras" -- it is a mistake. Omitting
    the key is how you ask for the default."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_devices": []}))
    assert load_live_config(path).camera_devices is None


def test_load_live_config_camera_devices_not_a_list_ignored(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"camera_devices": {"0": 1}}))
    assert load_live_config(path).camera_devices is None


# ---------------------------------------------------------------------------
# The two launcher update flags, and the decision built out of them.
#
# NOTHING IN THE PRODUCT READS THESE. run.sh/run.ps1 do, between the old
# process exiting and the new one starting, and what they decide is which
# code the rig comes back on -- so the rules are pinned here rather than
# left to two shell scripts nobody can run on this machine.


def test_update_flags_default_to_false_with_no_file(tmp_path):
    """The default is the feature. Both launchers pulled unconditionally
    until 2026-09-17; a rig whose config says nothing must now come back
    on the same commit it was running."""
    from opendarts.live.config import always_update, update_on_next_restart

    missing = tmp_path / "does_not_exist.json"
    assert always_update(missing) is False
    assert update_on_next_restart(missing) is False


def test_update_flags_default_to_false_with_the_keys_absent(tmp_path):
    from opendarts.live.config import always_update, update_on_next_restart

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"port": 8420}))
    assert always_update(path) is False
    assert update_on_next_restart(path) is False


def test_update_flags_read_a_real_true(tmp_path):
    from opendarts.live.config import always_update, update_on_next_restart

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"always_update": True, "update_on_next_restart": True}))
    assert always_update(path) is True
    assert update_on_next_restart(path) is True


def test_a_malformed_update_flag_reads_as_false(tmp_path):
    """OPPOSITE failure direction to store_packages_enabled(), on purpose.

    There, a typo must not cost the rig its evidence, so anything odd
    reads as ON. Here, anything odd reading as ON would mean a mistyped
    config file silently restored pull-on-every-restart -- the exact
    behaviour these keys exist to end. `"true"` (a string) and `1` are the
    realistic typos and both must read false.
    """
    from opendarts.live.config import always_update, update_on_next_restart

    for bad in ("true", "yes", 1, [], {}):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"always_update": bad, "update_on_next_restart": bad}))
        assert always_update(path) is False, bad
        assert update_on_next_restart(path) is False, bad


def test_an_unparseable_config_reads_as_false_rather_than_raising(tmp_path):
    from opendarts.live.config import always_update, update_on_next_restart

    path = tmp_path / "config.json"
    path.write_text("{not json at all")
    assert always_update(path) is False
    assert update_on_next_restart(path) is False


def test_set_update_on_next_restart_persists_and_keeps_its_neighbours(tmp_path):
    from opendarts.live.config import set_update_on_next_restart, update_on_next_restart

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"port": 9999, "store_packages": False}))

    assert set_update_on_next_restart(True, path) is True
    assert update_on_next_restart(path) is True
    raw = json.loads(path.read_text())
    assert raw["update_on_next_restart"] is True
    assert raw["port"] == 9999, "the operator's other settings were not preserved"
    assert raw["store_packages"] is False


def test_set_update_on_next_restart_reports_failure_on_an_unparseable_file(tmp_path):
    """write_config_section() REFUSES to overwrite a config it cannot
    parse and only logs -- so a bare try/except would report success for a
    write that never happened, and the dashboard would tell an operator
    their rig is about to update when it is not."""
    from opendarts.live.config import set_update_on_next_restart

    path = tmp_path / "config.json"
    path.write_text("{ oops")
    assert set_update_on_next_restart(True, path) is False
    assert path.read_text() == "{ oops", "the hand-edited file was overwritten"


def test_decide_skips_when_both_flags_are_off(tmp_path):
    from opendarts.live.update_policy import SKIP, decide

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"port": 8420}))
    assert decide(path) == SKIP


def test_decide_consumes_the_one_shot_flag(tmp_path):
    """Asked once, pulled once. The launcher clears the flag as it reads
    it, so a rig that crash-loops does not pull on every pass -- which is
    the behaviour this whole feature replaced."""
    from opendarts.live.update_policy import PULL_ONCE, SKIP, decide

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"update_on_next_restart": True}))

    assert decide(path) == PULL_ONCE
    assert json.loads(path.read_text())["update_on_next_restart"] is False
    assert decide(path) == SKIP


def test_decide_pulls_every_time_under_always_update(tmp_path):
    from opendarts.live.update_policy import PULL_ALWAYS, decide

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"always_update": True}))
    assert decide(path) == PULL_ALWAYS
    assert decide(path) == PULL_ALWAYS


def test_decide_clears_the_one_shot_even_on_an_always_update_rig(tmp_path):
    """Otherwise a flag left set here comes back to life the day someone
    turns always_update off."""
    from opendarts.live.update_policy import PULL_ALWAYS, decide

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"always_update": True, "update_on_next_restart": True}))
    assert decide(path) == PULL_ALWAYS
    assert json.loads(path.read_text())["update_on_next_restart"] is False


def test_decide_does_not_create_a_config_file_it_had_no_reason_to_touch(tmp_path):
    """A fresh clone has no data/config.json, and the launcher calling
    this on every relaunch must not conjure one."""
    from opendarts.live.update_policy import SKIP, decide

    path = tmp_path / "config.json"
    assert decide(path) == SKIP
    assert not path.exists()


def test_the_cli_prints_one_line_and_clears_the_flag(tmp_path):
    """What the launchers actually invoke, run as they invoke it.

    A subprocess rather than a call to decide(): the shell reads STDOUT,
    so an accidental print(), a stray log line on stdout or a non-zero
    exit would break both launchers while every in-process test still
    passed.
    """
    import subprocess
    import sys

    data = tmp_path / "data"
    data.mkdir()
    (data / "config.json").write_text(json.dumps({"update_on_next_restart": True}))
    env = dict(os.environ, OPENDARTS_DATA_DIR=str(data), PYTHONDONTWRITEBYTECODE="1")

    done = subprocess.run(
        [sys.executable, "-m", "opendarts.live.update_policy"],
        capture_output=True, text=True, env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "pull update_on_next_restart"
    assert len(done.stdout.strip().splitlines()) == 1
    assert json.loads((data / "config.json").read_text())["update_on_next_restart"] is False
