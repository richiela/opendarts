"""The oracle toggle owns virtual-camera publishing, at runtime.

The bug these exist for, hit on the Linux rig 2026-09-15: publishing was
decided once at startup, so a rig launched with the oracle OFF had
`state.vcam_set is None` forever. Switching the oracle on afterwards
attached nothing -- the oracle toggle reported success, the listener
connected, and no frames were ever published until someone restarted the
process.
"""
from __future__ import annotations

import threading

import pytest

from opendarts.live.server import create_app

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

@pytest.fixture()
def package_root(tmp_path):
    return tmp_path / "pkgroot"


@pytest.fixture()
def no_config_writes(monkeypatch):
    """Never touch the developer's real data/config.json.

    tests/conftest.py already redirects every default config path into
    this test's tmp directory, so this is now about seeing WHAT was
    written rather than about protection: the config document persists
    through opendarts.live.config_document.persist(), so that is the
    binding to replace. Reported as written, so persist()'s own readback
    still sees the value.
    """
    written = {}
    monkeypatch.setattr(
        "opendarts.live.config_document.write_config_section",
        lambda section, value, path=None: written.__setitem__(section, value),
    )
    monkeypatch.setattr(
        "opendarts.live.config_document.stored_value", written.get,
    )
    return written


class _Listener:
    def __init__(self, enabled=False):
        self.base_url = "http://localhost:3180"
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled

    def is_connected(self):
        return False

    def set_enabled(self, v):
        self._enabled = bool(v)

    def set_base_url(self, url):
        self.base_url = url


class _Hub:
    def __init__(self):
        self.configs = []
        self.status = {}
        self.sink = None
        self.frame_sink_errors = 0

    def set_frame_sink(self, fn):
        self.sink = fn


class _Set:
    """A publisher set that records its own lifecycle."""

    def __init__(self):
        self.closed = 0

    def publish_all(self, frames):
        return 0

    def stats(self):
        return []

    def worker_stats(self):
        return {"asynchronous": True, "dropped": 3}

    def close(self):
        self.closed += 1


def _client(package_root, hub, *, factory=None, vcam_set=None, listener=None):
    listener = listener or _Listener()
    app = create_app(package_root=package_root, enable_background_poll=False,
                     local_hub=hub, vcam_set=vcam_set,
                     vcam_set_factory=factory, ad_ws_listener=listener)
    return TestClient(app), listener


def test_enabling_the_oracle_builds_the_publishers_it_needs(package_root, no_config_writes):
    """The rig bug: no set at startup meant the toggle had nothing to
    attach, and the fix is to build one rather than give up."""
    built = []

    def factory():
        s = _Set()
        built.append(s)
        return s

    hub = _Hub()
    client, _ = _client(package_root, hub, factory=factory)

    client.patch("/api/config", json={"ad_enabled": True})

    assert len(built) == 1, "the set has to be created on demand"
    assert hub.sink == built[0].publish_all, "and actually attached"


def test_disabling_the_oracle_stops_publishing_and_releases_the_devices(
        package_root, no_config_writes):
    """Not just the sink: on Linux v4l2loopback ignores VIDIOC_S_FMT while
    any consumer holds the device open, so a writer that keeps its fds is
    a format that cannot be set correctly next time."""
    vset = _Set()
    hub = _Hub()
    hub.sink = vset.publish_all
    client, _ = _client(package_root, hub, vcam_set=vset, factory=_Set)

    client.patch("/api/config", json={"ad_enabled": False})

    assert hub.sink is None
    assert vset.closed == 1


def test_a_set_the_caller_owns_is_detached_but_not_closed(package_root, no_config_writes):
    """No factory means no way back: closing a set this process cannot
    rebuild would turn the off switch into a one-way door. Detaching the
    sink already stops every frame."""
    vset, hub = _Set(), _Hub()
    hub.sink = vset.publish_all
    client, _ = _client(package_root, hub, vcam_set=vset)

    client.patch("/api/config", json={"ad_enabled": False})
    assert hub.sink is None
    assert vset.closed == 0

    client.patch("/api/config", json={"ad_enabled": True})
    assert hub.sink == vset.publish_all, "and the same set comes back"


def test_toggling_repeatedly_needs_no_restart_and_leaks_nothing(
        package_root, no_config_writes):
    built = []

    def factory():
        s = _Set()
        built.append(s)
        return s

    hub = _Hub()
    client, _ = _client(package_root, hub, factory=factory)
    before = threading.active_count()

    for _ in range(3):
        client.patch("/api/config", json={"ad_enabled": True})
        assert hub.sink is not None
        client.patch("/api/config", json={"ad_enabled": False})
        assert hub.sink is None

    assert len(built) == 3, "a fresh set per on, since off closes the last"
    assert all(s.closed == 1 for s in built), "and every one of them closed"
    assert threading.active_count() <= before


def test_a_platform_without_virtual_cameras_stays_a_quiet_no_op(
        package_root, no_config_writes):
    """macOS shares cameras natively and an opted-out rig said no: the
    factory answers None, and the toggle must still apply to the oracle."""
    hub = _Hub()
    client, listener = _client(package_root, hub, factory=lambda: None)

    body = client.patch("/api/config", json={"ad_enabled": True}).json()

    assert body["ok"] is True
    assert listener.is_enabled() is True
    assert hub.sink is None


def test_a_factory_that_blows_up_does_not_take_the_toggle_with_it(
        package_root, no_config_writes):
    """Publishing is a convenience; failing to set it up must not stop the
    oracle being switched on."""
    def boom():
        raise RuntimeError("no such device")

    hub = _Hub()
    client, listener = _client(package_root, hub, factory=boom)

    body = client.patch("/api/config", json={"ad_enabled": True}).json()

    assert body["ok"] is True
    assert listener.is_enabled() is True
    assert hub.sink is None


def test_frame_health_stops_reporting_publishing_once_it_is_off(
        package_root, no_config_writes):
    vset = _Set()
    hub = _Hub()
    client, _ = _client(package_root, hub, vcam_set=vset, factory=_Set)
    assert client.get("/api/frame-health").json()["publish"]["enabled"] is True

    client.patch("/api/config", json={"ad_enabled": False})

    assert client.get("/api/frame-health").json()["publish"]["enabled"] is False


def test_frame_health_surfaces_how_many_frames_the_publisher_dropped(package_root):
    """Dropped frames never reach a device, so no per-slot counter can
    show them -- which makes the worker's own count the only evidence."""
    app = create_app(package_root=package_root, enable_background_poll=False,
                     vcam_set=_Set())
    body = TestClient(app).get("/api/frame-health").json()
    assert body["publish"]["worker"]["dropped"] == 3


def test_shutting_the_server_down_closes_the_publishers(package_root):
    """A set built lazily by the toggle has no other owner: nothing in
    run_product holds it, so without this the worker thread and the
    loopback fds would live until the process died."""
    vset, hub = _Set(), _Hub()
    app = create_app(package_root=package_root, enable_background_poll=False,
                     local_hub=hub, vcam_set=vset)
    with TestClient(app):
        pass
    assert vset.closed == 1


def test_a_publisher_set_without_a_worker_is_reported_without_one(package_root):
    """The Windows backend publishes inline on purpose (0.10ms per camera
    against the handoff copy's ~0.3ms), and older stand-ins have no such
    method at all -- neither is an error."""
    class _NoWorker:
        def stats(self):
            return [{"slot": 0, "published": 5}]

    app = create_app(package_root=package_root, enable_background_poll=False,
                     vcam_set=_NoWorker())
    body = TestClient(app).get("/api/frame-health").json()
    assert body["publish"]["enabled"] is True
    assert "worker" not in body["publish"]
