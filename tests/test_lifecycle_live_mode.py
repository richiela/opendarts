"""The capture loop driven end to end by opendarts.lifecycle.

Drives the REAL run_capture_loop_body() through a scripted synthetic scene
(same canvas as tests/test_lifecycle_state.py) and checks the loop's own
bookkeeping: handle_ready_to_capture() gets called once per dart with the
lifecycle's (bg, frame) pair and the right visit ids, the takeout clears
the visit, and the adapter's ThrowTriggerState mapping is what the UI
events see.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from opendarts.capture.trigger_state import ThrowState
from opendarts.lifecycle.adapter import LifecycleTriggerAdapter
from opendarts.lifecycle.state import Action, Lifecycle, LifecycleConfig, Phase
from opendarts.live import capture_daemon
from opendarts.pipeline import CameraCalibration


H, W = 360, 640
CX, CY, R = 320, 180, 120
RNG = np.random.default_rng(11)


def board_mask() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - CX) ** 2 + (yy - CY) ** 2 <= R**2


class Scene:
    def __init__(self):
        self.darts: list[tuple[int, int, int, int]] = []
        self.blobs: list[tuple[int, int, int, int]] = []

    def render(self) -> np.ndarray:
        img = np.full((H, W), 120, dtype=np.float32)
        for x, y, w, h in self.darts:
            img[y : y + h, x : x + w] = 20
        for x, y, w, h in self.blobs:
            img[max(0, y) : y + h, max(0, x) : x + w] = 230
        img += RNG.normal(0, 2.0, img.shape).astype(np.float32)
        gray = np.clip(img, 0, 255).astype(np.uint8)
        return np.dstack([gray, gray, gray])


@pytest.fixture()
def scratch(tmp_path):
    d = tmp_path / "run"
    d.mkdir(parents=True)
    return d


class _StopLoop(Exception):
    pass


def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros((5, 1)), rvec=np.zeros((3, 1)),
        tvec=np.zeros((3, 1)), pnp_result=None, landmark_spread_ok=True,
    )


def _script() -> list[np.ndarray]:
    """empty board -> 3 darts -> arm pulls them -> empty board."""
    s = Scene()
    frames = [s.render() for _ in range(20)]
    for x in (CX - 50, CX, CX + 50):
        s.darts.append((x, CY - 60, 16, 100))
        frames += [s.render() for _ in range(10)]
    for i in range(8):
        s.blobs = [(0, CY - 40, 60 + i * 40, 80)]
        frames.append(s.render())
    s.darts.clear()
    frames += [s.render() for _ in range(3)]
    for i in range(8):
        s.blobs = [(0, CY - 40, 340 - i * 45, 80)] if 340 - i * 45 > 0 else []
        frames.append(s.render())
    s.blobs = []
    frames += [s.render() for _ in range(25)]
    return frames


def _run_live_loop(frames, *, scratch, monkeypatch):
    monkeypatch.setattr(
        capture_daemon, "bootstrap_calibrations",
        lambda snapshot_dir, *, hub=None, **_: {0: _fake_calibration()},
    )
    monkeypatch.setattr(capture_daemon, "get_calibrated_board_disc_masks", lambda: {0: board_mask()})
    # a short warmup so the script stays small
    monkeypatch.setattr(capture_daemon, "_build_lifecycle_driver", lambda **kwargs: _driver(scratch))

    it = iter(frames)

    def fake_fetch(dest_dir, *, hub=None):
        try:
            return {0: next(it)}
        except StopIteration:
            raise _StopLoop()

    captured: list[dict] = []

    def fake_handle_ready_to_capture(trigger, bg_frames, calibrations, package_root, session_id, **kwargs):
        captured.append(
            {
                "visit_id": kwargs.get("visit_id"),
                "visit_index": kwargs.get("visit_index"),
                "dart_count": trigger.dart_count,
                "state": trigger.state,
                "bg": {c: f.copy() for c, f in bg_frames.items()},
                "frame": {c: f.copy() for c, f in trigger.last_frame.items()},
                "settle_duration_s": trigger.settle_duration_s,
            }
        )
        return package_root / f"fake_throw_{len(captured)}"

    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    events: list[dict] = []
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=scratch / "packages",
            poll_interval_s=0.0,
            stop_event=threading.Event(),
            scratch_dir=scratch / "scratch",
            on_event=events.append,
            background_save=False,
        )
    return events, captured


def _driver(scratch):
    from opendarts.lifecycle.driver import LifecycleDriver

    return LifecycleDriver(
        log_dir=scratch / "logs",
        masks_provider=capture_daemon.get_calibrated_board_disc_masks,
        config=LifecycleConfig(warmup_stable_frames=3),
        save_commit_frames=False,
    )


def _dart_present(img: np.ndarray, x: int) -> bool:
    return bool((img[CY - 60 : CY + 20, x : x + 12, 0] < 40).mean() > 0.8)


def test_live_mode_scores_three_darts_then_clears_the_visit(scratch, monkeypatch):
    events, captured = _run_live_loop(_script(), scratch=scratch, monkeypatch=monkeypatch)

    assert [c["visit_index"] for c in captured] == [0, 1, 2]
    assert len({c["visit_id"] for c in captured}) == 1
    assert [c["dart_count"] for c in captured] == [1, 2, 3]
    assert all(c["state"] is ThrowState.READY_TO_CAPTURE for c in captured)
    assert all(c["settle_duration_s"] is not None for c in captured)

    # engines get the board right before each dart, and the board with it
    xs = (CX - 50, CX, CX + 50)
    for i, c in enumerate(captured):
        for j, x in enumerate(xs):
            assert _dart_present(c["frame"][0], x) == (j <= i), (i, j, "frame")
            assert _dart_present(c["bg"][0], x) == (j < i), (i, j, "bg")

    cleared = [e for e in events if e["type"] == "VISIT_CLEARED"]
    assert len(cleared) == 1
    assert cleared[0]["n_darts"] == 3 and cleared[0]["reason"] == "takeout"
    assert cleared[0]["previous_visit_id"] == captured[0]["visit_id"]

    states = [(e["state"], e["dart_count"]) for e in events if e["type"] == "TRIGGER_STATE"]
    assert ("READY_TO_CAPTURE", 3) in states
    assert ("TAKEOUT_WAITING", 3) in states
    assert states[-1] == ("IDLE", 0)


def test_live_mode_bounce_out_and_hand_hover_produce_no_capture(scratch, monkeypatch):
    s = Scene()
    frames = [s.render() for _ in range(20)]
    # bounce-out: in motion for both visible frames (a static frame would
    # count as commit-ready evidence under the evidence-counter semantics)
    s.darts.append((CX, CY - 60, 16, 100))
    frames += [s.render()]
    s.darts = [(CX + 30, CY - 55, 16, 100)]
    frames += [s.render()]
    s.darts.clear()
    frames += [s.render() for _ in range(15)]
    for i in range(8):  # hand hovers, touches nothing
        s.blobs = [(0, CY - 40, 60 + i * 40, 80)]
        frames.append(s.render())
    for i in range(8):
        s.blobs = [(0, CY - 40, 340 - i * 45, 80)] if 340 - i * 45 > 0 else []
        frames.append(s.render())
    s.blobs = []
    frames += [s.render() for _ in range(20)]
    events, captured = _run_live_loop(frames, scratch=scratch, monkeypatch=monkeypatch)
    assert captured == []
    assert not [e for e in events if e["type"] == "VISIT_CLEARED"]
    states = [e["state"] for e in events if e["type"] == "TRIGGER_STATE"]
    assert "MOTION_DETECTED" in states  # the hand was visible to the UI
    assert states[-1] == "IDLE"


# --------------------------------------------------------------------------
# adapter mapping


def _lc_with_tick(action=Action.NONE, phase=Phase.IDLE, dart_count=0, reason="", cleared=0):
    from opendarts.lifecycle.state import Tick

    lc = Lifecycle({0: board_mask()}, LifecycleConfig(warmup_stable_frames=3))
    frame = Scene().render()
    for _ in range(5):
        lc.observe({0: frame})
    tick = Tick(n=1, phase=phase, action=action, dart_count=dart_count, signals={}, stable=True,
                hand=False, dart_cams=[], reason=reason, cleared=cleared)
    return lc, tick, frame


@pytest.mark.parametrize(
    "phase,dart_count,expected",
    [
        (Phase.IDLE, 0, ThrowState.IDLE),
        (Phase.COOLDOWN, 2, ThrowState.IDLE),
        (Phase.IDLE, 3, ThrowState.TAKEOUT_WAITING),
        (Phase.HAND, 1, ThrowState.MOTION_DETECTED),
        (Phase.SCENE_CHANGE, 0, ThrowState.MOTION_DETECTED),
        (Phase.PENDING_DART, 0, ThrowState.SETTLING),
        (Phase.TAKEOUT_PENDING, 3, ThrowState.TAKEOUT_WAITING),
    ],
)
def test_adapter_phase_mapping(phase, dart_count, expected):
    lc, tick, frame = _lc_with_tick(phase=phase, dart_count=dart_count)
    step = LifecycleTriggerAdapter().apply(tick, lc, {0: frame}, now=100.0)
    assert step.trigger.state is expected
    assert step.trigger.dart_count == dart_count
    assert step.cleared_darts is None
    assert step.trigger.last_frame is None


def test_adapter_baseline_dict_is_stable_until_the_reference_moves():
    lc, tick, frame = _lc_with_tick()
    ad = LifecycleTriggerAdapter()
    a = ad.apply(tick, lc, {0: frame}, now=1.0).trigger.true_baseline_frames
    b = ad.apply(tick, lc, {0: frame}, now=2.0).trigger.true_baseline_frames
    assert a is b
    lc.refs.adopt_all({0: Scene().render()}, {0: lc.refs[0].small})
    c = ad.apply(tick, lc, {0: frame}, now=3.0).trigger.true_baseline_frames
    assert c is not a


def test_adapter_cleared_tick_reports_dart_count():
    lc, tick, frame = _lc_with_tick(action=Action.CLEARED, phase=Phase.COOLDOWN, dart_count=0,
                                    reason="2 darts removed", cleared=2)
    step = LifecycleTriggerAdapter().apply(tick, lc, {0: frame}, now=5.0)
    assert step.cleared_darts == 2
    assert step.trigger.state is ThrowState.IDLE and step.trigger.dart_count == 0


def test_adapter_settle_duration_spans_pending_to_commit():
    from opendarts.lifecycle.state import Commit, Tick

    lc, _, frame = _lc_with_tick()
    ad = LifecycleTriggerAdapter()
    pend = Tick(n=1, phase=Phase.PENDING_DART, action=Action.NONE, dart_count=0, signals={},
                stable=False, hand=False, dart_cams=[0])
    ad.apply(pend, lc, {0: frame}, now=10.0)
    ad.apply(pend, lc, {0: frame}, now=10.2)
    commit = Commit(dart_index=0, bg={0: lc.refs[0].full}, frames={0: frame}, dart_cams=[0])
    ct = Tick(n=3, phase=Phase.COOLDOWN, action=Action.COMMIT, dart_count=1, signals={},
              stable=True, hand=False, dart_cams=[0], commit=commit)
    step = ad.apply(ct, lc, {0: frame}, now=10.5)
    t = step.trigger
    assert t.state is ThrowState.READY_TO_CAPTURE and t.dart_count == 1
    assert t.last_frame[0] is frame
    assert step.reference[0] is lc.refs[0].full
    assert t.settle_duration_s == pytest.approx(0.5)
    assert t.settle_started_monotonic == 10.0
    assert t.camera_settled_at_monotonic == {0: 10.5}
