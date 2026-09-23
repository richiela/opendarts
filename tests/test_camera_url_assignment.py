"""Assigning a camera slot to a stream URL from the dashboard.

A slot reads EITHER a local device or a network stream. The choice is per
slot, made in each camera card's dropdown, so a machine with no cameras --
a VM, a second rig, a laptop -- can score from another machine's published
streams.

The hub-internals half of this used to assert on a wrapper's `_routing`
and `_sub_index` -- the slot -> (which child, which index inside it) map
that no longer exists. What replaced those assertions is the behaviour
they were standing in for: which slot gets which frame, and which slots
open hardware. See tests/test_remote_capture.py for the source-level
tests.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opendarts.live.config import load_live_config
from opendarts.live.local_capture import CameraHub
from opendarts.live.remote_capture import build_hub
from opendarts.live.run_product import _resolve_camera_urls
from opendarts.live.server import _live_slot_urls, create_app


# -- which hub a rig gets ----------------------------------------------

def test_every_rig_gets_the_same_hub_whatever_its_slots_read():
    """Uniform ON PURPOSE, and now structurally rather than by wrapping.

    Pointing a slot at a stream must not hand out a DIFFERENT object: the
    capture thread binds its hub once at thread creation and AppState
    holds its own reference, so a replacement would leave one of them
    reading the old one until a restart."""
    for urls in (None, ["http://a/0", "http://a/1", "http://a/2"],
                 [None, "http://a/1", None]):
        hub = build_hub([0, 1, 2], urls, frame_sink="SINK")
        assert isinstance(hub, CameraHub), f"urls={urls}"
        assert hub._frame_sink == "SINK", "must keep publishing to virtual cameras"


def test_the_slot_urls_are_exactly_what_was_asked_for():
    assert build_hub([0, 1, 2], None).slot_urls == [None, None, None]
    assert build_hub([0, 1, 2], ["http://a/0", "http://a/1", "http://a/2"]).slot_urls == [
        "http://a/0", "http://a/1", "http://a/2"]
    assert build_hub([5, 6, 7], [None, "http://a/1", None]).slot_urls == [
        None, "http://a/1", None]


def test_a_mix_keeps_every_slots_own_device_index():
    """The dropdown sets slots independently, so a mix is reachable from
    the UI. A control that offers a state the product refuses is worse
    than one that works.

    And the DEVICE indices survive it. The wrapper this replaced rebuilt
    `configs` from two children, and a stream slot's config came back
    carrying the remote child's sub-index instead of the device the
    operator chose -- so the camera panel compared [5, 0, 7] against a
    saved [5, 6, 7] and reported a restart as pending forever."""
    hub = build_hub([5, 6, 7], [None, "http://a/1", None])
    assert [c.device for c in hub.configs] == [5, 6, 7]


def test_a_remote_slot_publishes_to_the_virtual_cameras_too():
    """The old rule was "a remote slot never republishes", on the grounds
    that a consumer must not re-serve frames it borrowed. It was wrong in
    both directions. A virtual camera is a LOCAL device (a v4l2loopback
    node, a DirectShow filter) -- nothing goes on the network, so there is
    no re-broadcast to prevent -- while the MJPEG endpoint DOES re-serve
    remote frames over HTTP, via state.hub.grab(), and was never covered
    by the rule. So it blocked the harmless case and permitted the one it
    was written for.

    Its real cost: on an all-stream rig the sink was attached to a local
    child that never pumped, so the virtual cameras received nothing at all."""
    hub = build_hub([0, 1, 2], [None, "http://a/1", None], frame_sink="SINK")
    assert hub._frame_sink == "SINK"
    assert hub.slot_urls == [None, "http://a/1", None]


# -- what the UI reads -------------------------------------------------

def test_live_slot_urls_reports_what_the_process_actually_reads():
    """Asked of the HUB, not the config, because those disagree exactly
    when a change is saved but not restarted into."""
    assert _live_slot_urls(build_hub([0, 1, 2], None), 3) == [None, None, None]
    remote = build_hub(None, ["http://a/0", "http://a/1"])
    assert _live_slot_urls(remote, 2) == ["http://a/0", "http://a/1"]
    mixed = build_hub([0, 1, 2], [None, "http://a/1", None])
    assert _live_slot_urls(mixed, 3) == [None, "http://a/1", None]


def test_live_slot_urls_is_all_none_for_a_hub_that_has_no_such_notion():
    """A stub or a fake hub honestly reports every slot local rather than
    raising -- the camera panel must still answer in a process whose hub
    predates stream support."""
    class _Stub:
        configs: list = []

    assert _live_slot_urls(_Stub(), 3) == [None, None, None]
    assert _live_slot_urls(None, 2) == [None, None]


def test_the_config_document_carries_urls(tmp_path):
    """`camera_urls` is a setting, so it is a key of the document; what
    each slot is ACTUALLY reading is not, so it rides in the runtime
    block beside the saved value it may disagree with."""
    app = create_app(package_root=tmp_path, enable_background_poll=False)
    body = TestClient(app).get("/api/config").json()
    assert "camera_urls" in body["config"]
    cams = body["runtime"]["cameras"]
    if not cams.get("available"):
        pytest.skip("no camera hub in this process")
    for key in ("urls", "saved_urls", "devices", "saved"):
        assert key in cams, f"runtime.cameras is missing {key}"


@pytest.mark.parametrize("bad,why", [
    ("rig:8420", "no scheme"),
    ("ftp://rig/x", "wrong scheme"),
    (42, "not a string"),
])
def test_patching_a_bad_url_is_refused_before_anything_is_written(tmp_path, bad, why):
    """A scheme-less URL becomes an unreachable host much later and
    presents as 'the camera stopped working', a long way from the edit
    that caused it."""
    from opendarts.live.config import DEFAULT_CONFIG_PATH

    app = create_app(package_root=tmp_path, enable_background_poll=False)
    resp = TestClient(app).patch("/api/config", json={
        "camera_devices": [0, 1, 2], "camera_urls": [bad, None, None],
    })
    assert resp.status_code == 400, why
    assert "url" in resp.json()["errors"]["camera_urls"].lower()
    # All-or-nothing: the devices named in the same request are not
    # written either.
    assert not DEFAULT_CONFIG_PATH.exists() or (
        "camera_devices" not in json.loads(DEFAULT_CONFIG_PATH.read_text())
    )


def test_urls_must_match_the_slot_count(tmp_path):
    app = create_app(package_root=tmp_path, enable_background_poll=False)
    resp = TestClient(app).patch("/api/config", json={
        "camera_devices": [0, 1, 2], "camera_urls": ["http://a/0"],
    })
    assert resp.status_code == 400
    assert resp.json()["errors"]["camera_urls"] == (
        "camera_urls must have one entry per camera slot (3), null for a local slot"
    )


# -- config + CLI ------------------------------------------------------

def test_config_round_trips_and_rejects_junk():
    d = Path(tempfile.mkdtemp()) / "c.json"
    d.write_text(json.dumps({"camera_urls": ["http://a/0", None, "http://a/2"]}))
    assert load_live_config(d).camera_urls == ["http://a/0", None, "http://a/2"]
    # One bad entry must not take the whole key down silently...
    d.write_text(json.dumps({"camera_urls": ["nope", "http://a/1", None]}))
    assert load_live_config(d).camera_urls == [None, "http://a/1", None]
    # ...and an all-bad list resolves to "no URLs" rather than a list of None.
    d.write_text(json.dumps({"camera_urls": ["nope", 7, ""]}))
    assert load_live_config(d).camera_urls is None
    d.write_text("{}")
    assert load_live_config(d).camera_urls is None


def test_a_bare_base_url_expands_to_that_rigs_cameras():
    """`--camera-url http://rig:8420` is what someone actually types.
    Making them write three stream URLs to say the obvious thing would be
    a papercut on the one command this feature exists for."""
    urls = _resolve_camera_urls(["http://rig:8420"], None)
    assert len(urls) == 3
    assert urls[1] == "http://rig:8420/api/cameras/1/stream.mjpg?full=1"


def test_explicit_urls_and_local_slots_pass_through():
    assert _resolve_camera_urls(["http://a/s", "local", "http://c/s"], None) == [
        "http://a/s", None, "http://c/s"]


def test_cli_beats_the_config_file():
    assert _resolve_camera_urls([], ["http://cfg/0", None, None]) == ["http://cfg/0", None, None]
    assert _resolve_camera_urls(["http://cli/s", "local", "local"],
                                ["http://cfg/0", None, None])[0] == "http://cli/s"
