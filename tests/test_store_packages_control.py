"""The dashboard's Recording control -- `store_packages` in the config
document (`GET`/`PATCH /api/config`).

`store_packages` was reachable only by hand-editing data/config.json on
the rig, which on the Windows rig means no shell at all. These cover the
control that exposes it, and specifically the two things about it that
are easy to get wrong in a way that still looks fine:

  - it is NOT live. The value is read once per session
    (run_product._read_store_packages), so a change made while a session
    runs does not apply to that session, and the reply has to say so.
  - persisting IS the whole action. write_config_section() deliberately
    REFUSES, and only logs, when the config file cannot be parsed -- so a
    plain try/except reports success for a write that never happened.

It had a route pair of its own (`GET`/`POST /api/store-packages`) until
2026-09-17; the coverage moved onto the config document with it.
"""

import json

import pytest
from fastapi.testclient import TestClient

from opendarts.live import config as config


@pytest.fixture
def client(tmp_path):
    """A client plus the config file it really reads and writes.

    No path patching here: tests/conftest.py's autouse `_isolate_repo_paths`
    already points every default config path at this test's own tmp
    directory, module attribute and bound function default together.
    """
    from opendarts.live import server

    cfg = config.DEFAULT_CONFIG_PATH
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text("{}\n")
    app = server.create_app(package_root=tmp_path / "packages",
                            enable_background_poll=False)
    yield TestClient(app), cfg


def test_defaults_to_on_when_the_key_is_absent(client):
    c, _ = client
    body = c.get("/api/config").json()
    assert body["ok"] is True
    assert body["config"]["store_packages"] is True, "an absent key must read as storing"
    assert body["runtime"]["store_packages"]["running"] is False


def test_disabling_persists_the_shared_key(client):
    c, cfg = client
    body = c.patch("/api/config", json={"store_packages": False}).json()
    assert body["ok"] is True
    assert body["changed"] == ["store_packages"]
    assert body["persisted"] == ["store_packages"]
    assert body["config"]["store_packages"] is False
    # The SAME top-level key an operator would hand-edit, not a parallel one.
    assert json.loads(cfg.read_text())["store_packages"] is False
    assert c.get("/api/config").json()["config"]["store_packages"] is False


def test_re_enabling_round_trips(client):
    c, cfg = client
    c.patch("/api/config", json={"store_packages": False})
    body = c.patch("/api/config", json={"store_packages": True}).json()
    assert body["config"]["store_packages"] is True
    assert body["persisted"] == ["store_packages"]
    assert json.loads(cfg.read_text())["store_packages"] is True


def test_a_value_that_is_not_a_boolean_is_refused(client):
    """The old route took `{"enabled": <anything>}` through `bool()`, so
    the string "no" saved as True. The document validates the key."""
    c, cfg = client
    resp = c.patch("/api/config", json={"store_packages": "no"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert body["errors"] == {
        "store_packages": "store_packages must be true or false, got 'no'"
    }
    assert "store_packages" not in json.loads(cfg.read_text())


def test_an_unparseable_config_reports_failure_not_success(client):
    """The regression this exists for: write_config_section() REFUSES to
    overwrite a config it cannot parse and merely logs it, raising
    nothing. Trusting the write would report a green 'saved' for a change
    that is not on disk and will never take effect."""
    c, cfg = client
    cfg.write_text("{ this is not json")
    resp = c.patch("/api/config", json={"store_packages": False})
    assert resp.status_code == 500, "a declined write must not report success"
    body = resp.json()
    assert body["ok"] is False
    assert body["persisted"] == []
    assert "JSON" in body["errors"]["store_packages"]


def test_other_keys_survive_the_write(client):
    c, cfg = client
    cfg.write_text(json.dumps({"port": 8420, "ad_enabled": True}) + "\n")
    c.patch("/api/config", json={"store_packages": False})
    raw = json.loads(cfg.read_text())
    assert raw["port"] == 8420 and raw["ad_enabled"] is True, (
        "persisting one toggle must not destroy the operator's own settings"
    )
    assert raw["store_packages"] is False


def test_the_reader_and_the_endpoint_share_one_interpreter():
    """run_product decides a session's behaviour; the document reports it.
    Two separate readings of the same key would let the dashboard show a
    state the capture loop is not in."""
    import inspect

    from opendarts.live import run_product

    src = inspect.getsource(run_product._read_store_packages)
    assert "store_packages_enabled" in src, (
        "the capture-loop reader must delegate to the shared interpreter"
    )


def test_a_malformed_value_reads_as_storing(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"store_packages": "yes please"}) + "\n")
    assert config.store_packages_enabled(cfg) is True, (
        "a typo must never silently stop a rig recording its own evidence"
    )


def test_the_control_is_in_the_markup_and_wired():
    """An input rendered with nothing listening looks entirely normal."""
    from opendarts.live.server import _render_dashboard_html

    html = _render_dashboard_html(3)
    # Same two-line extraction tests/test_dashboard_js_syntax.py uses,
    # inlined rather than imported: pulling a helper out of a sibling TEST
    # module needs tests/ on sys.path, and that re-imports test modules
    # under a second name -- which broke an unrelated test's fixture
    # teardown when this file first did it.
    assert "<script>" in html, "dashboard has no inline script"
    js = html.split("<script>")[-1].split("</script>")[0]
    assert 'id="store-packages-select"' in html
    assert 'id="store-packages-note"' in html
    assert "getElementById('store-packages-select').addEventListener" in js
    assert "patchConfig({store_packages: on}" in js
    # Redrawn from the config document every tick, like every sibling panel
    # -- otherwise the control shows "Checking..." until the page reloads.
    assert "renderStorePackagesPanel();" in js
