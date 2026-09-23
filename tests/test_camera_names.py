"""opendarts/live/camera_names.py -- the pure logic, with every platform
call faked.

WHY EVERYTHING HERE IS FAKED: this suite runs on macOS (dev), Linux (rig
tooling) and CI, but the module's platform paths read Windows COM, Linux
sysfs and macOS subprocess output. Real hardware answers would make the
tests describe whatever machine they happen to run on. What actually
needs pinning is the logic that is the same everywhere: index alignment
(the one property that matters -- an entry may go blank but must NEVER be
dropped, or every later name shifts onto the wrong camera), ordering,
fallback order, the authoritative flag, caching, and the never-raise
guarantee. The genuinely platform-bound parts (the COM vtable walk, the
QUERYCAP ioctl, real ffmpeg output) can only be verified on their own
platforms and are called out in the module docstring as such.
"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from opendarts.live import camera_names
from opendarts.live.camera_names import (
    DeviceEnumeration,
    _enumerate_linux,
    _enumerate_macos,
    _guid_fields,
    _parse_system_profiler_cameras,
)


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    """Every test starts with an empty module cache.

    The cache is module-global on purpose (the dashboard polls), which
    without this would make test outcomes depend on execution order.
    """
    monkeypatch.setattr(camera_names, "_cached", None)
    monkeypatch.setattr(camera_names, "_warned", False)


# ----------------------------------------------------------- system_profiler


def test_system_profiler_parse():
    text = ('{"SPCameraDataType": [{"_name": "FaceTime HD Camera"},'
            ' {"_name": "USB Camera"}]}')
    assert _parse_system_profiler_cameras(text) == [
        "FaceTime HD Camera", "USB Camera"]


def test_system_profiler_parse_rejects_garbage():
    assert _parse_system_profiler_cameras("not json") is None
    assert _parse_system_profiler_cameras('{"SPCameraDataType": null}') is None


# ------------------------------------------------------- macOS orchestration


def _completed(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def test_macos_uses_system_profiler_and_reports_nonauthoritative():
    def run(cmd, **kwargs):
        # The ONLY external tool consulted -- no ffmpeg, ever.
        assert cmd[0] == "system_profiler"
        return _completed(
            cmd,
            stdout='{"SPCameraDataType": [{"_name": "FaceTime HD Camera"},'
                   ' {"_name": "USB Camera #1"}, {"_name": "OBS Virtual Camera"}]}',
        )

    enum = _enumerate_macos(run=run)
    assert enum.names == ["FaceTime HD Camera", "USB Camera #1",
                          "OBS Virtual Camera"]
    assert enum.count == 3
    # system_profiler's order matches CAP_AVFOUNDATION in practice but
    # Apple guarantees nothing, so the UI presents these as hints.
    assert enum.authoritative is False
    assert enum.source == "system-profiler"


def test_macos_returns_empty_when_system_profiler_fails():
    def run(cmd, **kwargs):
        raise FileNotFoundError(cmd[0])

    enum = _enumerate_macos(run=run)
    assert enum.names == []
    assert enum.count is None
    assert enum.source == "none"


def test_macos_returns_empty_on_nonzero_exit():
    def run(cmd, **kwargs):
        return _completed(cmd, returncode=1, stderr="boom")

    assert _enumerate_macos(run=run).source == "none"


# ------------------------------------------------------------------- Linux


def _fake_sysfs(tmp_path: Path, nodes: dict[str, str]) -> Path:
    root = tmp_path / "video4linux"
    for node, name in nodes.items():
        d = root / node
        d.mkdir(parents=True)
        (d / "name").write_text(name + "\n")
    return root


def test_linux_orders_by_node_number_not_lexicographically(tmp_path):
    # video10 sorts before video2 as a string; as an OpenCV index it is
    # eight devices later. Getting this wrong swaps names between cameras.
    root = _fake_sysfs(tmp_path, {
        "video0": "Board Cam A", "video2": "Board Cam B",
        "video10": "Board Cam C",
    })
    enum = _enumerate_linux(sysfs_root=root, is_capture=lambda p: True)
    assert enum.names[0] == "Board Cam A"
    assert enum.names[2] == "Board Cam B"
    assert enum.names[10] == "Board Cam C"
    assert enum.count == 11
    assert enum.authoritative is True
    assert enum.source == "v4l2-sysfs"


def test_linux_gap_nodes_become_empty_entries_not_omissions(tmp_path):
    root = _fake_sysfs(tmp_path, {"video0": "Cam A", "video2": "Cam B"})
    enum = _enumerate_linux(sysfs_root=root, is_capture=lambda p: True)
    # /dev/video1 does not exist, but OpenCV index 1 still means
    # "/dev/video1" -- the slot must stay so Cam B stays at index 2.
    assert enum.names == ["Cam A", "", "Cam B"]


def test_linux_non_capture_node_is_blanked_but_keeps_its_index(tmp_path):
    # A UVC camera exposes a metadata node with the SAME name one index
    # later. Dropping it would shift every later camera; keeping the name
    # would lure the operator to an index that cannot stream. Blank it.
    root = _fake_sysfs(tmp_path, {
        "video0": "HD Webcam", "video1": "HD Webcam",
        "video2": "Board Cam",
    })
    enum = _enumerate_linux(
        sysfs_root=root,
        is_capture=lambda p: p != "/dev/video1")
    assert enum.names == ["HD Webcam", "", "Board Cam"]
    assert enum.count == 3


def test_linux_unknown_capability_keeps_the_name(tmp_path):
    # The QUERYCAP probe can fail (permissions, containers). A possibly-
    # unopenable label degrades better than blanking real cameras.
    root = _fake_sysfs(tmp_path, {"video0": "Board Cam"})
    enum = _enumerate_linux(sysfs_root=root, is_capture=lambda p: None)
    assert enum.names == ["Board Cam"]


def test_linux_missing_sysfs_is_unknown_not_zero(tmp_path):
    enum = _enumerate_linux(sysfs_root=tmp_path / "nope",
                            is_capture=lambda p: True)
    assert enum.names == []
    assert enum.count is None


def test_linux_empty_sysfs_is_authoritative_zero(tmp_path):
    root = tmp_path / "video4linux"
    root.mkdir()
    enum = _enumerate_linux(sysfs_root=root, is_capture=lambda p: True)
    assert enum.count == 0
    assert enum.authoritative is True


# ------------------------------------------------------------------ Windows


def test_guid_fields_round_trip():
    d1, d2, d3, d4 = _guid_fields("{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}")
    assert (d1, d2, d3) == (0x62BE5D10, 0x60EB, 0x11D0)
    assert d4 == (0xBD, 0x3B, 0x00, 0xA0, 0xC9, 0x11, 0xCE, 0x86)


def test_guid_fields_rejects_malformed():
    with pytest.raises(ValueError):
        _guid_fields("62BE5D10-60EB-11D0")


# ------------------------------------------------- dispatch, cache, façade


def test_dispatch_selects_platform_impl(monkeypatch):
    calls = []
    for system, impl in (("Windows", "_enumerate_windows"),
                         ("Linux", "_enumerate_linux"),
                         ("Darwin", "_enumerate_macos")):
        monkeypatch.setattr(camera_names.platform, "system", lambda s=system: s)
        monkeypatch.setattr(
            camera_names, impl,
            lambda s=system: calls.append(s) or camera_names._EMPTY)
        camera_names.enumerate_devices(refresh=True)
    assert calls == ["Windows", "Linux", "Darwin"]


def test_unknown_platform_returns_empty(monkeypatch):
    monkeypatch.setattr(camera_names.platform, "system", lambda: "Plan9")
    enum = camera_names.enumerate_devices(refresh=True)
    assert enum == camera_names._EMPTY


def test_result_is_cached_until_explicit_refresh(monkeypatch):
    calls = []

    def fake():
        calls.append(1)
        return DeviceEnumeration(["Cam"], 1, True, "v4l2-sysfs")

    monkeypatch.setattr(camera_names, "_enumerate_uncached", fake)
    camera_names.enumerate_devices()
    camera_names.enumerate_devices()
    camera_names.device_names()
    camera_names.device_count()
    assert len(calls) == 1, "enumeration must not run on every poll"
    camera_names.enumerate_devices(refresh=True)
    assert len(calls) == 2


def test_enumeration_failure_never_raises_and_logs_once(monkeypatch, caplog):
    def boom():
        raise RuntimeError("platform exploded")

    monkeypatch.setattr(camera_names, "_enumerate_uncached", boom)
    with caplog.at_level("WARNING", logger="opendarts.live.camera_names"):
        assert camera_names.enumerate_devices(refresh=True) == camera_names._EMPTY
        assert camera_names.enumerate_devices(refresh=True) == camera_names._EMPTY
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, "a broken platform is one line, not a stream"


def test_device_names_returns_a_copy(monkeypatch):
    monkeypatch.setattr(
        camera_names, "_cached",
        DeviceEnumeration(["Cam A"], 1, True, "v4l2-sysfs"))
    names = camera_names.device_names()
    names.append("mutation")
    assert camera_names.device_names() == ["Cam A"]


def test_label_for_always_includes_the_number(monkeypatch):
    monkeypatch.setattr(
        camera_names, "_cached",
        DeviceEnumeration(["Board Cam A", "", "Board Cam C"], 3, True,
                          "v4l2-sysfs"))
    assert camera_names.label_for(0) == "dev 0 — Board Cam A"
    # Unnamed and out-of-range fall back to the bare number -- never to
    # nothing, because the number is what config stores and logs report.
    assert camera_names.label_for(1) == "dev 1"
    assert camera_names.label_for(7) == "dev 7"


def test_label_for_does_not_mark_nonauthoritative_names(monkeypatch):
    """A name we cannot prove is still the name we show, unannotated.

    This used to assert the opposite -- a trailing '?' on every
    non-authoritative (macOS) name. Inverted 2026-09-12 on the owner's
    call: "we did the best we can, no point calling it out". The marker
    asked the operator to resolve a doubt they have no way to resolve,
    while the device NUMBER beside it was exact the whole time. The
    `authoritative` flag itself is unchanged and still reported -- the
    point of this test is that it must not leak into the label.
    """
    monkeypatch.setattr(
        camera_names, "_cached",
        DeviceEnumeration(["FaceTime HD"], 1, False, "ffmpeg-avfoundation"))
    label = camera_names.label_for(0)
    assert label == "dev 0 — FaceTime HD"
    assert "?" not in label
    # Identical to the authoritative rendering of the same name: the
    # source of the name changes nothing about how it is presented.
    monkeypatch.setattr(
        camera_names, "_cached",
        DeviceEnumeration(["FaceTime HD"], 1, True, "v4l2-sysfs"))
    assert camera_names.label_for(0) == label


# ------------------------------------------------------------- endpoint


def test_api_camera_devices_reports_names(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from opendarts.live import server

    monkeypatch.setattr(
        camera_names, "_cached",
        DeviceEnumeration(["Board Cam A", "", "Board Cam C"], 3, True,
                          "v4l2-sysfs"))

    class _Cfg:
        def __init__(self, device):
            self.device = device

    class _HubStub:
        configs = [_Cfg(0), _Cfg(2)]

    app = server.create_app(package_root=tmp_path,
                            enable_background_poll=False,
                            local_hub=_HubStub())
    body = TestClient(app).get("/api/config").json()["runtime"]["cameras"]
    assert body["available"] is True
    assert body["devices"] == [0, 2]
    assert body["device_names"] == ["Board Cam A", "", "Board Cam C"]
    assert body["device_count"] == 3
    assert body["names_authoritative"] is True
    assert body["names_source"] == "v4l2-sysfs"


def test_api_camera_devices_refresh_names_reenumerates(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from opendarts.live import server

    calls = []

    def fake():
        calls.append(1)
        return DeviceEnumeration(["Cam"], 1, True, "v4l2-sysfs")

    monkeypatch.setattr(camera_names, "_enumerate_uncached", fake)

    class _Cfg:
        def __init__(self, device):
            self.device = device

    class _HubStub:
        configs = [_Cfg(0)]

    app = server.create_app(package_root=tmp_path,
                            enable_background_poll=False,
                            local_hub=_HubStub())
    client = TestClient(app)
    client.get("/api/config")
    client.get("/api/config")
    assert len(calls) == 1, "polling must serve the cache"
    client.get("/api/config?refresh_camera_names=true")
    assert len(calls) == 2


def test_enumerate_never_waits_on_an_in_flight_enumeration(monkeypatch):
    """A second caller mid-enumeration takes the stale answer, not the lock.

    The first enumeration can take seconds (ffmpeg spawn / COM bind) and
    its caller is the dashboard polling `/api/config` through
    `asyncio.to_thread` -- the same default executor `fetch_snapshot`
    uses. If this blocked, every poll during that window would park
    another worker thread, spending the capture path's pool on a label
    for a dropdown. Asserted here because the cost only shows up under
    contention, which no other test creates.
    """
    import threading

    camera_names.enumerate_devices.__globals__["_cached"] = None

    entered = threading.Event()
    release = threading.Event()

    def slow():
        entered.set()
        release.wait(5)
        return camera_names.DeviceEnumeration(
            names=["cam"], count=1, authoritative=True, source="test")

    monkeypatch.setattr(camera_names, "_enumerate_uncached", slow)

    slow_caller = threading.Thread(target=camera_names.enumerate_devices)
    slow_caller.start()
    try:
        assert entered.wait(5), "the slow enumeration never started"
        # The contended call must return immediately rather than block
        # until `release` is set.
        t0 = time.monotonic()
        result = camera_names.enumerate_devices()
        assert time.monotonic() - t0 < 1.0, "contended caller blocked on the lock"
        assert result.source == "none", "should be the empty stale answer"
    finally:
        release.set()
        slow_caller.join(5)

    # And the enumeration that DID run still populates the cache.
    assert camera_names.enumerate_devices().source == "test"


# -- an empty list is "unknown", not "zero cameras" --------------------
#
# system_profiler returns an empty SPCameraDataType for a process without
# a camera grant exactly as it does for a Mac with no cameras attached.
# Those cannot be told apart, so an empty list must report count=None
# (leave the picker on bare numbers), never count=0 (which would shrink a
# working rig's dropdown to nothing). This replaced the old ffmpeg
# "Capture screen N slides onto index 0" hazard, which system_profiler
# does not have -- it never lists screen pseudo-devices at all.


def test_empty_system_profiler_is_unknown_not_zero_cameras():
    def run(cmd, **kw):
        return _completed(cmd, stdout='{"SPCameraDataType": []}')

    result = _enumerate_macos(run=run)
    assert result.count is None, "an empty list must not shrink the picker"
    assert result.names == []
