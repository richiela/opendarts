"""tests/test_port_control_and_host_config.py -- the serve address:
the `host` config key and its flag-beats-file-beats-constant
precedence, the dashboard's port control (now `port` in the config
document, GET/PATCH /api/config), the read-only bind-address row that
deliberately has no control on the page, and the checked-in
`config.example.json` template that documents every key.

The template tests compare against the REAL code constants, imported --
never against a transcribed literal. A shipped template that looks
authoritative and is wrong is worse than no template, and the only way a
test can catch that drift is to read the same source the code does.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from opendarts.live.config import (
    DEFAULT_CONFIG_PATH,
    LiveConfig,
    load_live_config,
)

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from opendarts.live.server import DEFAULT_HOST, DEFAULT_PORT, create_app  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "config.example.json"


@pytest.fixture
def package_root(tmp_path):
    root = tmp_path / "packages"
    root.mkdir()
    return root


@pytest.fixture
def isolated_config(tmp_path, monkeypatch):
    """Point every DEFAULT-path config read/write at tmp_path.

    `read_config_section`/`write_config_section` bind their default path
    as a DEFAULT ARGUMENT VALUE, evaluated at import time, so patching
    the module constant is not enough -- the already-bound default has to
    be replaced too. Without this, the config document would read and
    WRITE this developer's real data/config.json.
    """
    from opendarts.live import config as config_mod

    path = tmp_path / "config.json"
    monkeypatch.setattr(config_mod, "DEFAULT_CONFIG_PATH", path)
    monkeypatch.setattr(
        config_mod.read_config_section, "__defaults__", (path,), raising=False
    )
    monkeypatch.setattr(
        config_mod.write_config_section, "__defaults__", (path,), raising=False
    )
    return path


# --------------------------------------------------------------------------
# `host` as a real config key, and its three-layer precedence
# --------------------------------------------------------------------------

def test_live_config_reads_host(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"host": "127.0.0.1"}))

    assert load_live_config(path).host == "127.0.0.1"


def test_live_config_absent_host_is_none_meaning_no_override(tmp_path):
    """The dataclass never invents a value -- "no override" is None, and
    the DEFAULT_HOST fallback lives at the CLI layer where every other
    key's fallback lives."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"port": 9000}))

    assert load_live_config(path).host is None


@pytest.mark.parametrize("bad", [123, True, ["0.0.0.0"], "", "   "])
def test_live_config_rejects_a_host_that_is_not_a_real_address_string(tmp_path, bad):
    """Including the blank string: argparse would hand "" straight to
    uvicorn, which binds it as every interface on some stacks and refuses
    it on others. A blank key is a mistake, so it reads as absent."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"host": bad}))

    assert load_live_config(path).host is None


def test_host_precedence_code_default_when_nothing_is_configured():
    from opendarts.live.run_product import _build_arg_parser

    args = _build_arg_parser(LiveConfig()).parse_args([])

    assert args.host == DEFAULT_HOST == "0.0.0.0"
    assert args.port == DEFAULT_PORT


def test_host_precedence_config_file_beats_the_code_default():
    from opendarts.live.run_product import _build_arg_parser

    args = _build_arg_parser(LiveConfig(host="192.0.2.7", port=9100)).parse_args([])

    assert args.host == "192.0.2.7"
    assert args.port == 9100


def test_host_precedence_cli_flag_beats_the_config_file():
    """The layer that actually matters on a rig locked out by its own
    config: `--host` on the launch line must override the file, so a bad
    file value is recoverable from the command that starts the process."""
    from opendarts.live.run_product import _build_arg_parser

    parser = _build_arg_parser(LiveConfig(host="192.0.2.7", port=9100))
    args = parser.parse_args(["--host", "127.0.0.1", "--port", "9200"])

    assert args.host == "127.0.0.1"
    assert args.port == 9200


# --------------------------------------------------------------------------
# `port` and `host` on the config document (GET/PATCH /api/config)
# --------------------------------------------------------------------------

def test_config_get_reports_the_default_port_when_nothing_is_configured(
    package_root, isolated_config
):
    app = create_app(
        package_root=package_root, host=DEFAULT_HOST, enable_background_poll=False
    )
    body = TestClient(app).get("/api/config").json()

    assert body["ok"] is True
    assert body["config"]["port"] == DEFAULT_PORT
    assert body["runtime"]["port"]["configured"] is False
    # The bind address this process is actually on, reported beside the
    # document so the Config tab needs no second request for it.
    assert body["config"]["host"] == DEFAULT_HOST
    assert body["runtime"]["host"]["active"] == DEFAULT_HOST

    # And honestly null, not invented, when nothing bound one -- the
    # standalone create_app() case, where AppState.host is None.
    unbound = create_app(package_root=package_root, enable_background_poll=False)
    assert TestClient(unbound).get("/api/config").json()["runtime"]["host"]["active"] is None


def test_config_get_reports_the_configured_port_and_the_live_one_separately(
    package_root, isolated_config
):
    """The two legitimately disagree the moment someone saves a new port,
    and the panel has to be able to say so rather than implying the
    server already moved."""
    isolated_config.write_text(json.dumps({"port": 9321}))
    app = create_app(package_root=package_root, port=8420, enable_background_poll=False)
    body = TestClient(app).get("/api/config").json()

    assert body["config"]["port"] == 9321
    assert body["runtime"]["port"]["configured"] is True
    assert body["runtime"]["port"]["active"] == 8420
    # And the disagreement is NAMED, not left for the client to spot.
    assert body["restart_required"] == ["port"]


def test_patching_the_port_persists_to_the_same_key_the_launcher_reads(
    package_root, isolated_config
):
    app = create_app(package_root=package_root, port=8420, enable_background_poll=False)
    body = TestClient(app).patch("/api/config", json={"port": 9500}).json()

    assert body["ok"] is True
    assert body["persisted"] == ["port"]
    assert body["restart_required"] == ["port"]
    # It is NOT applied live: uvicorn's socket was bound before this app
    # object existed, so rebinding under a live request would drop the
    # response reporting success.
    assert body["applied_live"] == []
    assert body["runtime"]["port"]["active"] == 8420
    # Not just "the endpoint said ok" -- the value must land in the real
    # file, under the key load_live_config() reads, as a real int.
    assert json.loads(isolated_config.read_text())["port"] == 9500
    assert load_live_config(isolated_config).port == 9500


def test_patching_the_port_preserves_every_other_key(package_root, isolated_config):
    """One setting must never cost an operator their camera mapping."""
    isolated_config.write_text(json.dumps({
        "ad_base_url": "http://example.local:3180",
        "camera_devices": [1, 2, 3],
    }))
    app = create_app(package_root=package_root, enable_background_poll=False)

    TestClient(app).patch("/api/config", json={"port": 9500})

    saved = json.loads(isolated_config.read_text())
    assert saved["ad_base_url"] == "http://example.local:3180"
    assert saved["camera_devices"] == [1, 2, 3]
    assert saved["port"] == 9500


def test_patching_the_port_accepts_a_numeric_string(package_root, isolated_config):
    """The browser control is an <input>, so the value arrives as text."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).patch("/api/config", json={"port": " 9600 "}).json()

    assert body["ok"] is True
    assert body["config"]["port"] == 9600
    assert json.loads(isolated_config.read_text())["port"] == 9600


@pytest.mark.parametrize("bad", [0, -1, 65536, 99999, "abc", "80a", 12.5, True, None, [8420]])
def test_patching_a_garbage_port_is_refused_with_a_reason_and_writes_nothing(
    package_root, isolated_config, bad
):
    """Rejected with an honest reason rather than coerced. `int(12.5)`
    truncates and `bool` is an int subclass -- either would put a port
    the operator did not choose into the one file that decides whether
    this dashboard comes back after a restart."""
    isolated_config.write_text(json.dumps({"port": 8420}))
    app = create_app(package_root=package_root, enable_background_poll=False)

    resp = TestClient(app).patch("/api/config", json={"port": bad})

    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert "port" in body["errors"]["port"]
    # The existing value is untouched -- a rejection is a no-op, not a
    # half-write.
    assert json.loads(isolated_config.read_text())["port"] == 8420


def test_an_empty_patch_body_is_refused(package_root, isolated_config):
    """`{}` names no key, so there is nothing to do and nothing to
    report. Saying so beats a cheerful "saved" for a no-op."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).patch("/api/config", json={}).json()

    assert body["ok"] is False
    assert "at least one config key" in body["errors"]["_body"]


def test_patching_the_port_reports_the_no_op_when_the_config_cannot_be_parsed(
    package_root, isolated_config
):
    """write_config_section() REFUSES to overwrite a file it cannot
    parse, and only logs. Trusting the write would report success for
    something that never happened, so every key is read back."""
    isolated_config.write_text("{ this is not json")
    app = create_app(package_root=package_root, enable_background_poll=False)

    resp = TestClient(app).patch("/api/config", json={"port": 9700})

    assert resp.status_code == 500
    body = resp.json()
    assert body["ok"] is False
    assert body["persisted"] == []
    assert "json" in body["errors"]["port"].lower()
    # The operator's broken file is still theirs to fix, not silently replaced.
    assert isolated_config.read_text() == "{ this is not json"


def test_the_bind_host_is_a_config_key_with_no_control_on_the_page(
    package_root, isolated_config
):
    """A deliberate asymmetry, asserted so neither half drifts.

    `host` IS a key of the config document -- it is a key of
    data/config.json, and an API that refused to write one key of its own
    file would be lying about being the door to it. What has no control
    is the PAGE: a wrong bind address makes this dashboard unreachable at
    the next restart and the Windows rig has no shell to fix it from, so
    the field there is a span. The API reports the change as
    restart-required rather than pretending it moved.
    """
    app = create_app(package_root=package_root, host=DEFAULT_HOST,
                     enable_background_poll=False)
    client = TestClient(app)

    body = client.patch("/api/config", json={"host": "127.0.0.1"}).json()
    assert body["ok"] is True
    assert body["restart_required"] == ["host"]
    assert body["applied_live"] == []
    assert json.loads(isolated_config.read_text())["host"] == "127.0.0.1"
    # Still bound where it was; the document says so too.
    assert body["runtime"]["host"]["active"] == DEFAULT_HOST

    routes = {getattr(r, "path", "") for r in app.routes}
    assert "/api/host" not in routes
    assert "/api/port" not in routes, "the per-setting route was retired"

    html = client.get("/").text
    assert '<input type="text" id="server-host"' not in html


# --------------------------------------------------------------------------
# The dashboard: port is a control, bind address is a row
# --------------------------------------------------------------------------

def test_config_tab_has_a_real_port_control_wired_to_the_endpoint(package_root):
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/").text

    assert 'id="server-port"' in html
    assert "'/api/config'" in html
    # Wired, not merely present: a change handler that really PATCHes.
    assert "document.getElementById('server-port').addEventListener('change'" in html
    assert "patchConfig({port: ev.target.value}" in html
    assert "method: 'PATCH'" in html


def test_port_field_says_plainly_that_it_applies_at_restart(package_root):
    """Same honesty the Recording control already carries: the note must
    name WHEN the value lands, because it is not now."""
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/").text

    assert "applies at the next restart, not now" in html


def test_bind_address_is_displayed_but_has_no_input(package_root):
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/").text

    assert 'id="server-host"' in html
    # Displayed as text, never as a form control -- not even a disabled
    # one (see the CSS comment on .config-field-value).
    assert '<input type="text" id="server-host"' not in html
    assert "getElementById('server-host').value" not in html


def test_the_old_info_tabs_config_duplication_did_not_come_back(package_root):
    """The Info tab exists again (restored 2026-09-13 the same day it was
    deleted), but what killed the FIRST one must not return with it.

    That tab was not bad because it was a tab. It was bad because it
    printed read-only copies of CONFIG VALUES -- host and port among them
    -- so one subject lived in two places, and its standing note claimed
    all eight "are set in data/config.json" when host, package root
    and frame source never were. What the tab carries now is build,
    machine, versions and wiring: facts about the rig that duplicate no
    setting.

    So this no longer asserts the tab's absence. It asserts the
    DUPLICATION's absence, which is what actually mattered."""
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/").text

    # The old renderer and its table id are gone -- renamed, not left
    # rendering into an element that no longer exists. (The name survives
    # in a comment recording the rename; the function must not.)
    assert "function renderConfig(" not in html
    assert "renderConfig(state.config)" not in html
    assert 'id="config-tbody"' not in html
    assert "function renderDiagnostics(" in html
    assert "renderDiagnostics(state.config)" in html
    assert 'id="diagnostics-tbody"' in html

    # The wrong note must not have come back with the tab.
    assert "set in <code>data/config.json</code>" not in html

    # And host/port stay CONTROLS in Config rather than becoming read-only
    # rows in Info again -- that duplication is the original sin here.
    info = html[html.index('id="tab-info"'):html.index('id="tab-engines"')]
    assert 'id="server-port"' not in info, "the port control belongs in Config"


# --------------------------------------------------------------------------
# The checked-in config.example.json template
# --------------------------------------------------------------------------

def test_template_config_exists_and_is_valid_json():
    raw = json.loads(TEMPLATE_PATH.read_text())
    assert isinstance(raw, dict)


def test_template_config_is_not_the_file_the_code_reads():
    """It is documentation, not configuration. If this ever became the
    live path, a checked-in file would start overriding every rig."""
    assert DEFAULT_CONFIG_PATH != TEMPLATE_PATH
    assert DEFAULT_CONFIG_PATH.name == "config.json"
    assert DEFAULT_CONFIG_PATH.parent.name == "data"


def test_template_config_covers_every_key_and_omits_capabilities():
    raw = json.loads(TEMPLATE_PATH.read_text())

    expected = {
        "host", "port", "ad_base_url", "ad_enabled", "camera_devices",
        "camera_resolutions", "reprojection_targets_px", "store_packages",
        "publish_virtual_cameras", "cv2_num_threads", "idle_timeout_sec",
        "lifecycle_settings", "engine_config",
        # 2026-09-16, the throw-capture ring. Listed in the template
        # because its default is the one in this project that costs
        # gigabytes of RAM, and "every key and its real default can be
        # seen without reading source" is exactly what the template is
        # for. `frame_ring_max_gb` is deliberately NOT listed: it has no
        # default (absent means "no ceiling"), and a template value would
        # invite copying a cap nobody chose.
        "frame_ring_seconds",
        # 2026-09-17, the free-space floor both on-disk writers stop at.
        # Listed for the same reason: the default (5) is the number that
        # decides whether a rig keeps recording, and a negative value here
        # is the documented way to turn the guard off -- neither is
        # discoverable from an absent key.
        "min_free_disk_gb",
        # 2026-09-17, the two keys that decide whether run.sh/run.ps1 pull
        # before relaunching -- read by the LAUNCHER on each loop, not by
        # the product. Listed for the strongest version of the template's
        # own reason: both default to FALSE, both used to be unconditionally
        # true in effect (the launchers pulled every time), and an absent
        # key here would leave "does this machine follow main?" answerable
        # only by reading a shell script.
        "always_update",
        "update_on_next_restart",
    }
    assert set(raw) - {"_readme"} == expected

    # Every listed key is a real key of the config document, and a
    # WRITABLE one -- the template must not invite editing something the
    # API would refuse.
    from opendarts.live.config_document import WRITABLE_KEYS

    assert expected <= set(WRITABLE_KEYS)

    # `capabilities` is written by the startup probe, not by hand.
    # Shipping it in a template invites editing something that is
    # overwritten at the next launch.
    from opendarts.live.capabilities import CONFIG_SECTION

    assert CONFIG_SECTION == "capabilities"
    assert CONFIG_SECTION not in raw
    # ...and the omission is explained rather than merely done.
    assert any(CONFIG_SECTION in line for line in raw["_readme"])


def test_template_config_says_what_it_is():
    raw = json.loads(TEMPLATE_PATH.read_text())
    readme = "\n".join(raw["_readme"]).lower()

    assert "template" in readme
    assert "data/config.json" in readme
    assert "optional" in readme
    # "absent key means use the code default" -- the one fact that makes
    # copying the whole file pointless.
    assert "absent" in readme


def test_template_config_values_are_the_real_code_defaults():
    """Every scalar checked against the constant the code actually reads,
    imported here rather than transcribed. This is the test that catches
    the template drifting into a confident lie."""
    from opendarts.engines.dispatch import DEFAULT_ENGINE_TIMEOUT_S
    from opendarts.engines.registry import DEFAULT_ALSO_RUN, DEFAULT_PRIMARY_ENGINE
    from opendarts.lifecycle.state import DEFAULT_CONFIG as LIFECYCLE_DEFAULT_CONFIG
    from opendarts.live.ad_ground_truth import DEFAULT_AD_BASE
    from opendarts.live.capture_daemon import IDLE_TIMEOUT_SEC_DEFAULT
    from opendarts.live.cv2_threads import DEFAULT_CV2_NUM_THREADS
    from opendarts.live.local_capture import DEFAULT_CAMERA_DEVICES

    raw = json.loads(TEMPLATE_PATH.read_text())

    assert raw["host"] == DEFAULT_HOST
    assert raw["port"] == DEFAULT_PORT
    assert raw["ad_base_url"] == DEFAULT_AD_BASE
    assert raw["camera_devices"] == DEFAULT_CAMERA_DEVICES
    assert raw["cv2_num_threads"] == DEFAULT_CV2_NUM_THREADS
    assert raw["idle_timeout_sec"] == IDLE_TIMEOUT_SEC_DEFAULT
    assert raw["lifecycle_settings"] == {
        "dart_stable_frames": LIFECYCLE_DEFAULT_CONFIG.dart_stable_frames
    }
    # No "audio" section, deliberately: on/off, volume and voice became
    # per-device browser settings on 2026-09-15 when spoken calls moved
    # out of the rig. A template key for a setting the product no longer
    # reads would be a confident lie, which is exactly what this file
    # exists to catch.
    assert "audio" not in raw
    assert raw["engine_config"] == {
        "primary": DEFAULT_PRIMARY_ENGINE,
        "also_run": list(DEFAULT_ALSO_RUN),
        "timeout_s": DEFAULT_ENGINE_TIMEOUT_S,
    }
    # The two update flags default OFF: a rig that pulled new code on
    # every restart without anyone asking is the opposite of a shipped
    # appliance.
    assert raw["always_update"] is False
    assert raw["update_on_next_restart"] is False


def test_template_config_booleans_match_the_functions_that_interpret_them(tmp_path):
    """store_packages and ad_enabled have no module constant -- their
    default lives inside the function that reads them, so the template is
    checked against that function's answer to an EMPTY config rather than
    against a literal someone believed."""
    from opendarts.live.config import store_packages_enabled

    raw = json.loads(TEMPLATE_PATH.read_text())

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    assert raw["store_packages"] == store_packages_enabled(empty)

    # ad_enabled: absent means off (the comparison is opt-in since
    # 2026-09-16), so the template shows false.
    assert load_live_config(empty).ad_enabled is None
    assert raw["ad_enabled"] is False

    # publish_virtual_cameras has no fixed default at all -- absent, it
    # follows ad_enabled -- so the template's value must equal what that
    # function returns for the ad_enabled default above, and the _readme
    # has to say the key is not a constant.
    from opendarts.live.run_product import _read_publish_virtual_cameras

    assert raw["publish_virtual_cameras"] == _read_publish_virtual_cameras(
        raw["ad_enabled"]
    )
    assert any("publish_virtual_cameras" in line for line in raw["_readme"])


def test_template_config_dict_valued_defaults_are_empty_not_illustrative(tmp_path):
    """camera_resolutions / reprojection_targets_px default to NO
    entries, and an example dropped in as if it were the default would be
    a shipped behaviour change for anyone who copied the file. The
    examples live in _readme, where they cannot be pasted by accident."""
    raw = json.loads(TEMPLATE_PATH.read_text())

    assert raw["camera_resolutions"] == {}
    assert raw["reprojection_targets_px"] == {}

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    cfg = load_live_config(empty)
    assert cfg.camera_resolutions == {}
    assert cfg.reprojection_targets_px == {}

    readme = "\n".join(raw["_readme"])
    assert "camera_resolutions" in readme
    assert "reprojection_targets_px" in readme


def test_template_config_loads_cleanly_through_the_real_loader(tmp_path, caplog):
    """The strongest single check: hand the template to load_live_config()
    and it must produce exactly the values the code would have used
    anyway, with no "ignoring" warning -- i.e. every key is spelled the
    way the loader spells it and typed the way the loader types it.
    `_readme` is an unknown key, which the loader ignores silently by
    design (it only reads keys it knows)."""
    import logging

    path = tmp_path / "config.json"
    path.write_text(TEMPLATE_PATH.read_text())

    with caplog.at_level(logging.WARNING, logger="opendarts.live.config"):
        cfg = load_live_config(path)

    assert caplog.text == ""
    assert cfg.host == DEFAULT_HOST
    assert cfg.port == DEFAULT_PORT
    assert cfg.ad_enabled is False
    assert cfg.cv2_num_threads == 1


def test_bind_address_and_port_are_one_field(package_root):
    """They are a single fact -- the address this dashboard answers on.
    Splitting them across two boxes made a reader assemble it. The port
    is editable and the host is not, which is a difference in what you
    may DO to them, not a reason to file them under separate headings."""
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/").text
    server = _enclosing_config_field(html, "server-port")

    import re as _re
    containers = _re.findall(r'class="config-field(?:\s[^"]*)?"', server)
    assert len(containers) == 1, f"host and port belong in one box, found {containers}"
    assert 'id="server-port"' in server
    # Still not editable, even sharing a box with something that is: the
    # host is a span. Asserted on the element itself rather than on what
    # happens to sit near it.
    assert '<span id="server-host"' in server, (
        "the bind address must remain a span, never an input"
    )
    assert '<input type="number" id="server-port"' in server


def _enclosing_config_field(html: str, control_id: str) -> str:
    """The single `.config-field` box that contains `control_id`.

    Slicing by section heading (the old approach) coupled these tests to
    where a control is FILED rather than to the control itself: the 2026-09-14
    config regroup removed the "Server" heading and both tests broke while
    the markup they actually check was untouched.
    """
    import re as _re
    at = html.index(f'id="{control_id}"')
    starts = [m.start() for m in _re.finditer(r'<(?:label|div) class="config-field(?:\s[^"]*)?"', html)
              if m.start() < at]
    assert starts, f"no config-field box opens before {control_id}"
    start = starts[-1]
    # Walk to the matching close of that box.
    depth, i = 0, start
    for m in _re.finditer(r'<(?:label|div)\b[^>]*>|</(?:label|div)>', html[start:]):
        depth += -1 if m.group(0).startswith("</") else 1
        if depth == 0:
            return html[start:start + m.end()]
    raise AssertionError(f"box holding {control_id} never closes")
