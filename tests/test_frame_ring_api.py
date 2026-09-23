"""The dashboard surface of the throw-capture ring: the routes, the
memory figure they put on screen, and the automatic oracle trigger.

THREE SLOTS everywhere the cost is computed. The whole point of the note
under the seconds box is that it prices THIS rig, so a test with one
camera would pass against a hardcoded three-camera constant.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendarts.capture.frame_ring import FrameRing
from opendarts.capture.throw_capture import ThrowCaptureService
from opendarts.live import capture_daemon
from opendarts.live.server import (
    MAX_FRAME_RING_SECONDS,
    _apply_frame_ring_seconds,
    create_app,
)


def frame(slot: int, tick: int) -> np.ndarray:
    arr = np.zeros((8, 12, 3), dtype=np.uint8)
    arr[:, :, 0] = slot + 1
    arr[:, :, 1] = tick % 251
    return arr


def fill(ring: FrameRing, *, n: int, wall0: float) -> None:
    for i in range(n):
        ring.append({s: frame(s, i) for s in (0, 1, 2)},
                    wall_s=wall0 + i / 30.0, monotonic_s=4321.0 + i / 30.0,
                    generation=i)


@pytest.fixture
def client(tmp_path):
    """A real app with a real service over a real ring.

    No config-path patching: tests/conftest.py's autouse
    `_isolate_repo_paths` already points every default config path at this
    test's own tmp directory, so a test can never edit the developer's own
    data/config.json.
    """
    from opendarts.live.config import DEFAULT_CONFIG_PATH as cfg

    cfg.parent.mkdir(parents=True, exist_ok=True)
    ring = FrameRing(22.0)
    service = ThrowCaptureService(ring, capture_root=tmp_path / "captures")
    app = create_app(
        package_root=tmp_path / "packages",
        scratch_dir=tmp_path / "scratch",
        enable_background_poll=False,
        throw_capture=service,
    )
    with TestClient(app) as c:
        c.opendarts_service = service       # type: ignore[attr-defined]
        c.opendarts_ring = ring             # type: ignore[attr-defined]
        c.opendarts_cfg = cfg               # type: ignore[attr-defined]
        yield c


# -- GET /api/config's frame_ring runtime block -------------------------
#
# The ring had its own route pair until 2026-09-17. `frame_ring_seconds`
# is now a key of the config document like any other, and everything the
# GET used to carry that is NOT a setting -- what the ring holds right
# now, the writer's progress, what a window costs on THIS rig -- rides
# under `runtime.frame_ring`.


def ring_runtime(client):
    return client.get("/api/config").json()["runtime"]["frame_ring"]


def test_get_prices_the_current_setting_for_this_rigs_camera_count(client):
    body = ring_runtime(client)
    assert body["ok"] is True
    assert body["n_slots"] == 3
    # Nothing measured yet: 1280*720*3 bytes, three cameras, at the ~32.5
    # sets a second the pump really achieves -- the same rate the sizing
    # table uses, so the two agree.
    assert body["estimate_source"] == "pixels"
    assert body["estimated_bytes_per_s"] == pytest.approx(1280 * 720 * 3 * 3 * 32.5)
    assert body["estimated_bytes"] == pytest.approx(
        body["estimated_bytes_per_s"] * body["configured_seconds"]
    )
    assert body["estimated_label"].endswith(("GB", "MB"))


def test_get_reports_the_ceiling_and_what_it_costs(client):
    body = ring_runtime(client)
    assert body["max_seconds"] == 60
    # Nothing measured yet: three uncompressed 720p cameras at ~32.5 sets/s.
    assert body["max_label"] == "16.17 GB"

def test_get_reports_what_the_ring_is_actually_holding_not_just_its_setting(client):
    fill(client.opendarts_ring, n=90, wall0=1_757_000_000.0)
    ring = ring_runtime(client)["ring"]
    assert ring["enabled"] is True
    assert ring["sets"] == 90
    assert ring["frames"] == 270                  # three per set, not one
    assert ring["span_s"] == pytest.approx(89 / 30.0, abs=0.01)
    # Configured 22s, holding ~3s: a session that just started, or a
    # stalled pump. That distinction is why both numbers are reported.
    assert ring["fill_fraction"] < 0.2


def test_get_says_so_when_no_ring_is_attached_at_all(tmp_path):
    """"No capture loop in this process" and "the ring is empty" are
    different facts; collapsing them is how an operator concludes the
    feature is broken when it was never switched on."""
    app = create_app(package_root=tmp_path, scratch_dir=tmp_path / "s",
                     enable_background_poll=False)
    with TestClient(app) as c:
        body = c.get("/api/config").json()["runtime"]["frame_ring"]
    assert body["attached"] is False
    assert "no capture loop is running in this process" in body["reason"]


def test_the_flight_expectation_note_reaches_the_dashboard(client):
    note = ring_runtime(client)["flight_note"]
    assert "30ms" in note and "SETTLE SEQUENCE" in note
    # Anyone expecting slow motion should be disappointed by physics
    # BEFORE they press the button, not after.
    assert "not slow-motion flight" in note


# -- PATCH /api/config {"frame_ring_seconds": ...} ----------------------


def test_setting_seconds_persists_and_applies_to_the_live_ring(client):
    body = client.patch("/api/config", json={"frame_ring_seconds": 12}).json()
    assert body["ok"] is True
    assert body["persisted"] == ["frame_ring_seconds"]
    # THE key that must not wait for a restart: it is holding gigabytes
    # right now, so "saved" alone would be a control that lies.
    assert body["applied_live"] == ["frame_ring_seconds"]
    assert body["restart_required"] == []
    assert client.opendarts_ring.seconds == 12.0
    assert json.loads(client.opendarts_cfg.read_text())["frame_ring_seconds"] == 12
    # And the reply already carries the new price, so the UI does not need
    # a second round trip to update the note.
    assert body["config"]["frame_ring_seconds"] == 12.0
    assert body["runtime"]["frame_ring"]["configured_seconds"] == 12.0


def test_setting_zero_detaches_the_ring_and_frees_what_it_held(client):
    fill(client.opendarts_ring, n=60, wall0=1_757_000_000.0)
    body = client.patch("/api/config", json={"frame_ring_seconds": 0}).json()
    assert body["ok"] is True and body["applied_live"] == ["frame_ring_seconds"]
    assert client.opendarts_service.ring is None
    assert body["runtime"]["frame_ring"]["ring"]["enabled"] is False
    # Called TWICE: turning off an already-off ring must not error, and
    # the reply says plainly that there was nothing live left to apply to.
    again = client.patch("/api/config", json={"frame_ring_seconds": 0}).json()
    assert again["ok"] is True
    assert again["applied_live"] == []
    assert "no frame ring in this process" in again["notes"]["frame_ring_seconds"]


def test_turning_it_back_on_after_zero_really_buffers_again(client):
    client.patch("/api/config", json={"frame_ring_seconds": 0})
    assert client.opendarts_service.ring is None
    body = client.patch("/api/config", json={"frame_ring_seconds": 8}).json()
    assert body["ok"] is True and body["applied_live"] == ["frame_ring_seconds"]
    ring = client.opendarts_service.ring
    assert ring is not None and ring.seconds == 8.0
    fill(ring, n=5, wall0=1_757_000_000.0)
    assert ring_runtime(client)["ring"]["sets"] == 5


def test_an_absurd_window_is_refused_with_the_arithmetic(client):
    resp = client.patch("/api/config", json={"frame_ring_seconds": 600})
    assert resp.status_code == 400
    reason = resp.json()["errors"]["frame_ring_seconds"]
    assert "GB of memory" in reason
    assert f"{MAX_FRAME_RING_SECONDS:g}s" in reason
    assert client.opendarts_ring.seconds == 22.0          # unchanged


@pytest.mark.parametrize("bad", [None, True, "twelve", -1, [5]])
def test_a_bad_seconds_value_is_refused_without_touching_the_ring(client, bad):
    resp = client.patch("/api/config", json={"frame_ring_seconds": bad})
    assert resp.status_code == 400
    assert resp.json()["errors"]["frame_ring_seconds"]
    assert client.opendarts_ring.seconds == 22.0
    # Nothing written at all: a refused value must not create or touch the
    # operator's config file.
    saved = json.loads(client.opendarts_cfg.read_text()) if client.opendarts_cfg.exists() else {}
    assert "frame_ring_seconds" not in saved


def test_apply_returns_false_when_there_is_nothing_live_to_apply_to():
    """A saved-but-not-applied change must report itself as such, so the
    dashboard can say "saved for the next Start" rather than implying
    gigabytes were just freed."""
    class _State:
        throw_capture = None
        hub = None
    assert _apply_frame_ring_seconds(_State(), 10.0) is False


# -- the two triggers over HTTP -----------------------------------------


def test_the_missed_dart_route_writes_the_whole_buffer(client):
    fill(client.opendarts_ring, n=45, wall0=1_757_000_000.0)
    body = client.post("/api/frame-ring/capture-missed",
                       json={"reason": "nothing registered"}).json()
    assert body["ok"] is True
    client.opendarts_service.writer.join()
    dest = Path(body["job"]["dest_dir"])
    manifest = json.loads((dest / "manifest.json").read_text())
    assert manifest["n_sets"] == 45
    assert manifest["extra"]["operator_reason"] == "nothing registered"
    assert manifest["extra"]["trigger"] == "manual"


def test_the_missed_dart_route_reports_a_refusal_rather_than_a_bare_failure(client):
    body = client.post("/api/frame-ring/capture-missed", json={}).json()
    assert body["ok"] is False
    assert "holds nothing to write" in body["reason"]


def test_the_misscore_route_captures_around_the_package_timestamp(client, tmp_path):
    now = datetime.now(timezone.utc)
    ring = client.opendarts_ring
    fill(ring, n=300, wall0=now.timestamp() - 10.0)
    pkg = tmp_path / "packages" / "sess" / "throw-001"
    pkg.mkdir(parents=True)
    anchor = now.timestamp() - 5.0
    (pkg / "meta.json").write_text(json.dumps({
        "session": "sess", "cameras": [0, 1, 2],
        "captured_at_utc": datetime.fromtimestamp(anchor, timezone.utc).isoformat(),
    }))

    body = client.post("/api/packages/sess/throw-001/capture-misscore",
                       json={"reason": "called T20"}).json()
    assert body["ok"] is True
    client.opendarts_service.writer.join()
    manifest = json.loads(
        (Path(body["job"]["dest_dir"]) / "manifest.json").read_text()
    )
    assert manifest["kind"] == "misscore"
    assert manifest["anchor_wall_s"] == pytest.approx(anchor, abs=0.001)
    assert manifest["n_sets"] == 22                # -0.5s/+0.2s at 30/s


def test_the_misscore_route_refuses_an_unknown_package_with_404(client):
    resp = client.post("/api/packages/nope/nope/capture-misscore", json={})
    assert resp.status_code == 404
    assert "no such package" in resp.json()["reason"]


@pytest.mark.parametrize("bad", ["../etc", "a/b", "a\\b"])
def test_the_misscore_route_rejects_a_path_traversal_attempt(client, bad):
    resp = client.post(f"/api/packages/{bad}/t/capture-misscore", json={})
    assert resp.status_code in (400, 404)


def test_an_aged_out_misscore_over_http_returns_the_numbers(client, tmp_path):
    ring = client.opendarts_ring
    now = datetime.now(timezone.utc)
    fill(ring, n=300, wall0=now.timestamp() - 10.0)
    pkg = tmp_path / "packages" / "sess" / "old-throw"
    pkg.mkdir(parents=True)
    (pkg / "meta.json").write_text(json.dumps({
        "session": "sess", "cameras": [0, 1, 2],
        "captured_at_utc": datetime.fromtimestamp(
            now.timestamp() - 600.0, timezone.utc
        ).isoformat(),
    }))
    body = client.post("/api/packages/sess/old-throw/capture-misscore", json={}).json()
    assert body["ok"] is False
    assert body["aged_out"] is True
    assert "older than the newest frame" in body["reason"]
    # Never an empty file.
    assert not any((tmp_path / "captures").glob("*")) if (tmp_path / "captures").exists() else True


# -- the automatic oracle trigger ---------------------------------------


class _Gt:
    def __init__(self, sector, ring, matched=True):
        self.sector, self.ring, self.matched = sector, ring, matched


def _package_with_result(tmp_path, *, sector, ring_name, ok=True) -> Path:
    pkg = tmp_path / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "result.json").write_text(json.dumps(
        {"ok": ok, "sector": sector, "ring": ring_name}
    ))
    (pkg / "meta.json").write_text(json.dumps({
        "session": "s", "cameras": [0, 1, 2],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }))
    return pkg


class _RecordingService:
    #: The mismatch trigger only fires in "mismatch" mode (in "all" every
    #: throw is already recorded, in "never" there is no ring).
    def __init__(self, ok=True, record_mode="mismatch"):
        self.calls: list[dict] = []
        self._ok = ok
        self.record_mode = record_mode

    def record_throw_clip(self, package_dir, anchor=None, *, reason, source):
        self.calls.append({"package_dir": package_dir, "anchor": anchor,
                           "reason": reason, "source": source})
        return {"ok": self._ok, "reason": None if self._ok else "aged out: 22.0s"}


def test_a_disagreement_with_the_oracle_fires_a_capture_automatically(tmp_path, caplog):
    pkg = _package_with_result(tmp_path, sector=20, ring_name="single_outer")
    service = _RecordingService()
    with caplog.at_level("INFO"):
        capture_daemon._capture_misscore_on_oracle_disagreement(
            pkg, _Gt(20, "treble"), service
        )
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["source"] == "oracle"
    assert call["package_dir"] == pkg
    # The reason carries BOTH answers, so a clip found later explains
    # itself without a second lookup.
    assert "opendarts called 20/single_outer" in call["reason"]
    assert "oracle called 20/treble" in call["reason"]
    assert any("oracle disagreement" in r.getMessage() for r in caplog.records)


def test_disagreement_does_not_fire_a_clip_in_all_or_never_mode(tmp_path):
    """"all" already recorded this throw at commit; "never" has no ring --
    so the oracle-disagreement trigger is a no-op in both."""
    pkg = _package_with_result(tmp_path, sector=20, ring_name="single_outer")
    for mode in ("all", "never"):
        service = _RecordingService(record_mode=mode)
        capture_daemon._capture_misscore_on_oracle_disagreement(
            pkg, _Gt(20, "treble"), service
        )
        assert service.calls == [], mode


def test_agreement_fires_nothing(tmp_path):
    pkg = _package_with_result(tmp_path, sector=20, ring_name="treble")
    service = _RecordingService()
    capture_daemon._capture_misscore_on_oracle_disagreement(
        pkg, _Gt(20, "treble"), service
    )
    # Called TWICE to be sure it is not a first-call quirk.
    capture_daemon._capture_misscore_on_oracle_disagreement(
        pkg, _Gt(20, "treble"), service
    )
    assert service.calls == []


def test_an_unmatched_oracle_event_fires_nothing(tmp_path):
    """AD not answering is an ordinary state, not a misscore. Firing here
    would fill the disk with dumps of throws nothing was wrong with."""
    pkg = _package_with_result(tmp_path, sector=20, ring_name="treble")
    service = _RecordingService()
    capture_daemon._capture_misscore_on_oracle_disagreement(
        pkg, _Gt(None, None, matched=False), service
    )
    assert service.calls == []


def test_a_throw_we_could_not_score_is_not_treated_as_a_misscore(tmp_path):
    pkg = _package_with_result(tmp_path, sector=None, ring_name=None, ok=False)
    service = _RecordingService()
    capture_daemon._capture_misscore_on_oracle_disagreement(
        pkg, _Gt(20, "treble"), service
    )
    assert service.calls == []


def test_an_automatic_capture_that_is_refused_is_logged_loudly(tmp_path, caplog):
    """Nobody is watching a dashboard line for this one, so the log is the
    only place the refusal can be seen at all."""
    pkg = _package_with_result(tmp_path, sector=20, ring_name="single_outer")
    service = _RecordingService(ok=False)
    with caplog.at_level("WARNING"):
        capture_daemon._capture_misscore_on_oracle_disagreement(
            pkg, _Gt(20, "treble"), service
        )
    messages = [r.getMessage() for r in caplog.records]
    assert any("did NOT happen" in m and "aged out" in m for m in messages)


def test_no_service_and_no_package_are_both_silent_no_ops(tmp_path):
    pkg = _package_with_result(tmp_path, sector=1, ring_name="bull")
    capture_daemon._capture_misscore_on_oracle_disagreement(pkg, _Gt(2, "bull"), None)
    service = _RecordingService()
    capture_daemon._capture_misscore_on_oracle_disagreement(
        tmp_path / "nothing-here", _Gt(2, "bull"), service
    )
    assert service.calls == []


def test_a_ring_that_has_measured_itself_is_priced_by_that(client, monkeypatch):
    """A passthrough rig holds camera JPEGs: a Windows rig held 5s in 34.6 MB while the
    pixel formula quoted 1.24 GB."""
    from opendarts.capture.frame_ring import FrameRing

    monkeypatch.setattr(FrameRing, "stats",
                        lambda self: {"enabled": True, "measured_bytes_per_s": 7_000_000.0})
    body = ring_runtime(client)
    if not body["attached"]:
        pytest.skip("no ring attached in this fixture")
    assert body["estimate_source"] == "measured"
    assert body["estimated_bytes_per_s"] == pytest.approx(7_000_000.0)
    assert body["max_label"] == "420 MB"
