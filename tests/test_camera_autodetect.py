"""Picking device indexes on a rig whose config names none.

`DEFAULT_CAMERA_DEVICES` is [0, 1, 2], which is correct on macOS and
Windows and wrong on Linux: a UVC camera registers a capture node AND a
metadata node, so three cameras occupy /dev/video0..5 and only the even
ones deliver frames. The default therefore selects two cameras and a
metadata node on every Linux rig's first start.

The subtle half is the exclusion. Metadata nodes fail a capture check, so
they are easy. Our own v4l2loopback devices PASS one -- measured on the
rig, /dev/video10-12 report capture=True -- so a capture test alone would
select a virtual camera and feed it its own output. That failure would
present as a frozen or duplicated image, not as a configuration error,
which is why it is pinned here rather than left to the obvious test.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opendarts.live import camera_names

#: The rig, exactly as measured on 2026-09-15: three UVC cameras with
#: their metadata nodes, plus the three loopbacks the setup script makes.
RIG = {
    "/dev/video0": ("uvcvideo", "USB Camera: USB Camera", True),
    "/dev/video1": ("uvcvideo", "USB Camera: USB Camera", False),
    "/dev/video2": ("uvcvideo", "USB Camera: USB Camera", True),
    "/dev/video3": ("uvcvideo", "USB Camera: USB Camera", False),
    "/dev/video4": ("uvcvideo", "USB Camera: USB Camera", True),
    "/dev/video5": ("uvcvideo", "USB Camera: USB Camera", False),
    "/dev/video10": ("v4l2 loopback", "OpenDarts Cam 0", True),
    "/dev/video11": ("v4l2 loopback", "OpenDarts Cam 1", True),
    "/dev/video12": ("v4l2 loopback", "OpenDarts Cam 2", True),
}


def fake_sysfs(tmp_path: Path, nodes) -> Path:
    root = tmp_path / "video4linux"
    root.mkdir()
    for path in nodes:
        (root / Path(path).name).mkdir()
    return root


def querier(nodes):
    def query(dev_path: str):
        entry = nodes.get(dev_path)
        if entry is None:
            return None
        driver, card, is_capture = entry
        return camera_names._NodeInfo(is_capture=is_capture, driver=driver,
                                      card=card)
    return query


def _pretend(monkeypatch, system: str) -> None:
    """Swap the module REFERENCE, never an attribute on `platform` itself.

    `monkeypatch.setattr(camera_names.platform, "system", ...)` reaches
    into the real, shared platform module, so for the duration of the test
    every other module in the process -- and any background thread -- sees
    the fake too. That produced a teardown error in an unrelated test file
    on the first run of this suite. Replacing the name `camera_names`
    binds to is scoped to this module and cannot leak.
    """
    monkeypatch.setattr(camera_names, "platform",
                        SimpleNamespace(system=lambda: system))


@pytest.fixture(autouse=True)
def _on_linux(monkeypatch):
    """Every case here is about Linux; the function is a no-op elsewhere."""
    _pretend(monkeypatch, "Linux")


def test_skips_metadata_nodes(tmp_path: Path) -> None:
    """The actual rig answer: [0, 2, 4], not [0, 1, 2]."""
    got = camera_names.suggest_camera_devices(
        3, sysfs_root=fake_sysfs(tmp_path, RIG), query=querier(RIG))
    assert got == [0, 2, 4]


def test_never_selects_our_own_loopback(tmp_path: Path) -> None:
    """A virtual camera passes a capture check and must still be refused.

    Constructed so capability alone is not enough to get the right answer:
    the loopbacks sit at LOWER device numbers than the real cameras, so a
    correct-by-accident implementation that merely sorted capture-capable
    nodes would pick all three virtual ones and publish each camera its
    own output.
    """
    nodes = {
        "/dev/video0": ("v4l2 loopback", "OpenDarts Cam 0", True),
        "/dev/video1": ("v4l2 loopback", "OpenDarts Cam 1", True),
        "/dev/video2": ("v4l2 loopback", "OpenDarts Cam 2", True),
        "/dev/video3": ("uvcvideo", "USB Camera", True),
        "/dev/video4": ("uvcvideo", "USB Camera", False),
        "/dev/video5": ("uvcvideo", "USB Camera", True),
        "/dev/video6": ("uvcvideo", "USB Camera", False),
        "/dev/video7": ("uvcvideo", "USB Camera", True),
    }
    got = camera_names.suggest_camera_devices(
        3, sysfs_root=fake_sysfs(tmp_path, nodes), query=querier(nodes))
    assert got == [3, 5, 7]


def test_falls_back_rather_than_running_short(tmp_path: Path) -> None:
    """One camera unplugged must not silently reconfigure the rig.

    Returning [0, 2] here would start a two-camera rig that looks
    healthy. The operator needs the missing camera reported, which is what
    the existing default plus an open failure does.
    """
    nodes = {k: v for k, v in RIG.items() if k not in ("/dev/video4",)}
    got = camera_names.suggest_camera_devices(
        3, sysfs_root=fake_sysfs(tmp_path, nodes), query=querier(nodes))
    assert got is None


def test_unreadable_nodes_are_skipped_not_guessed(tmp_path: Path) -> None:
    """A node QUERYCAP could not answer for is not assumed to be a camera.

    This is the no-'video'-group case: every open fails, so nothing is
    known, and inventing a device list from nothing would be worse than
    falling back.
    """
    got = camera_names.suggest_camera_devices(
        3, sysfs_root=fake_sysfs(tmp_path, RIG), query=lambda _p: None)
    assert got is None


def test_absent_sysfs_is_not_an_error(tmp_path: Path) -> None:
    """A kernel or container with no v4l2 class at all just declines."""
    got = camera_names.suggest_camera_devices(
        3, sysfs_root=tmp_path / "nope", query=querier(RIG))
    assert got is None


def test_no_op_off_linux(tmp_path: Path, monkeypatch) -> None:
    """macOS and Windows do not enumerate metadata nodes.

    0..N-1 is already right there, and probing would only add a way to be
    wrong.
    """
    _pretend(monkeypatch, "Darwin")
    got = camera_names.suggest_camera_devices(
        3, sysfs_root=fake_sysfs(tmp_path, RIG), query=querier(RIG))
    assert got is None


def test_wired_into_the_config_default(tmp_path: Path, monkeypatch) -> None:
    """A config naming no devices gets the detected list, not [0, 1, 2].

    The unit above can be perfect and still never run -- this is the wire
    that makes a first start actually benefit from it.
    """
    from opendarts.live import run_product

    monkeypatch.setattr(camera_names, "suggest_camera_devices",
                        lambda count: [0, 2, 4])

    class Cfg:
        camera_devices = None
        camera_resolutions = None

    configs = run_product._camera_configs_from_live_config(Cfg())
    assert configs is not None
    assert [c.device for c in configs] == [0, 2, 4]


def test_an_explicit_config_is_never_overridden(monkeypatch) -> None:
    """Autodetect fills a gap; it does not second-guess an operator.

    Someone who set [1, 3, 5] by hand meant it, and a rig that quietly
    replaced their choice at each boot would be impossible to configure.
    """
    from opendarts.live import run_product

    monkeypatch.setattr(camera_names, "suggest_camera_devices",
                        lambda count: pytest.fail("probed despite a config"))

    class Cfg:
        camera_devices = [1, 3, 5]
        camera_resolutions = None

    configs = run_product._camera_configs_from_live_config(Cfg())
    assert [c.device for c in configs] == [1, 3, 5]
