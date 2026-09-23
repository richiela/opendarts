"""Synthetic-scene tests for opendarts.lifecycle (signals + state machine).

A scene is a 640x360 gray canvas with a circular "board" region; darts
are thin dark bars inside it, hands are big bright blobs that move in
from the edge. Everything is driven at full resolution through the real
``Lifecycle.observe`` path (downscale, change detection, aggregation).
"""
from __future__ import annotations

import numpy as np
import pytest

from opendarts.lifecycle.signals import (
    change_mask,
    compute_signals,
    to_small_gray,
    to_small_mask,
)
from opendarts.lifecycle.state import Action, Lifecycle, LifecycleConfig, Phase

H, W = 360, 640
CX, CY, R = 320, 180, 120
RNG = np.random.default_rng(7)


def board_mask() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W]
    return (xx - CX) ** 2 + (yy - CY) ** 2 <= R**2


class Scene:
    """Mutable synthetic scene; ``render()`` returns a BGR frame with noise."""

    def __init__(self, level: int = 120, noise: float = 2.0):
        self.level = level
        self.noise = noise
        self.darts: list[tuple[int, int, int, int]] = []  # x, y, w, h
        self.blobs: list[tuple[int, int, int, int]] = []  # hand-like

    def render(self) -> np.ndarray:
        img = np.full((H, W), self.level, dtype=np.float32)
        for x, y, w, h in self.darts:
            img[y : y + h, x : x + w] = 20
        for x, y, w, h in self.blobs:
            img[max(0, y) : y + h, max(0, x) : x + w] = 230
        img += RNG.normal(0, self.noise, img.shape).astype(np.float32)
        gray = np.clip(img, 0, 255).astype(np.uint8)
        return np.dstack([gray, gray, gray])


def new_lifecycle(cfg: LifecycleConfig | None = None) -> Lifecycle:
    cfg = cfg or LifecycleConfig(warmup_stable_frames=3)
    return Lifecycle({0: board_mask()}, cfg)


def run(lc: Lifecycle, scene: Scene, n: int):
    ticks = []
    for _ in range(n):
        ticks.append(lc.observe({0: scene.render()}))
    return ticks


def warm(lc: Lifecycle, scene: Scene):
    ticks = run(lc, scene, 6)
    assert lc.phase is Phase.IDLE, [t.phase for t in ticks]
    assert any(t.action is Action.READY for t in ticks)


# --------------------------------------------------------------------------
# signals


def test_change_mask_ignores_sensor_noise():
    s = Scene()
    a, b = to_small_gray(s.render()), to_small_gray(s.render())
    assert change_mask(a, b).sum() == 0


def test_signals_split_board_and_outside():
    s = Scene()
    ref = to_small_gray(s.render())
    s.darts.append((CX - 2, CY - 60, 16, 100))   # inside board
    s.blobs.append((10, 10, 60, 60))           # outside board
    cur = to_small_gray(s.render())
    bm = to_small_mask(board_mask(), cur.shape)
    sig, changed = compute_signals(cur, ref, ref, bm)
    assert sig.board_px > 0 and sig.outside_px > 0
    assert sig.board_px == pytest.approx(100 * 16 / 16, rel=0.6)
    assert sig.outside_px == pytest.approx(60 * 60 / 16, rel=0.3)
    assert sig.board_delta_px + sig.outside_delta_px > 0
    assert sig.union_px == 0 and sig.board_new_px == sig.board_px
    assert changed.shape == cur.shape


# --------------------------------------------------------------------------
# state machine


def test_warmup_then_idle_with_no_actions_on_quiet_scene():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    ticks = run(lc, s, 30)
    assert all(t.action in (Action.NONE, Action.READOPT) for t in ticks)
    assert lc.dart_count == 0


def test_dart_commits_after_stable_frames_and_hands_pre_dart_bg():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    pre = s.render()
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = run(lc, s, 10)
    commits = [t for t in ticks if t.action is Action.COMMIT]
    assert len(commits) == 1
    t = commits[0]
    # default dart_stable_frames=2 (evidence semantics): the arrival
    # frame counts, so evidence=2 on the first stable frame -> commit
    assert ticks.index(t) == 1
    assert t.commit is not None and t.commit.dart_index == 0
    assert t.commit.dart_cams == [0]
    # bg is the scene before the dart, frame is the scene with it
    assert np.abs(t.commit.bg[0].astype(int) - pre.astype(int)).mean() < 3
    assert (t.commit.frames[0][CY - 60 : CY + 20, CX - 2 : CX + 10, 0] < 40).all()
    assert lc.dart_count == 1
    assert ticks[4].phase is Phase.COOLDOWN
    assert lc.phase is Phase.IDLE


def test_three_darts_then_takeout_clears_visit():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    for i, x in enumerate((CX - 50, CX, CX + 50)):
        s.darts.append((x, CY - 60, 16, 100))
        ticks = run(lc, s, 15)
        assert sum(t.action is Action.COMMIT for t in ticks) == 1, i
        assert lc.dart_count == i + 1
    # a 4th change is refused
    s.darts.append((CX + 80, CY - 40, 16, 100))
    ticks = run(lc, s, 10)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert lc.dart_count == 3
    s.darts.pop()
    run(lc, s, 100)  # let the stuck change get absorbed / return to idle
    # takeout: all darts vanish
    s.darts.clear()
    ticks = run(lc, s, 12)
    cleared = [t for t in ticks if t.action is Action.CLEARED]
    assert len(cleared) == 1
    assert lc.dart_count == 0
    assert not lc.refs.has_darts()
    run(lc, s, 15)
    assert lc.phase is Phase.IDLE
    # next visit works against the fresh reference
    s.darts.append((CX, CY - 60, 16, 100))
    ticks = run(lc, s, 10)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1


def test_wobbling_dart_waits_for_stability():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    ticks = []
    for i in range(8):
        # the dart moves every frame: never stable
        s.darts = [(CX - 2 + (i % 2) * 6, CY - 60, 16, 100)]
        ticks.append(lc.observe({0: s.render()}))
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert lc.phase is Phase.PENDING_DART
    ticks = run(lc, s, 6)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1


def test_bounce_out_never_commits():
    # A bounce-out is moving for every frame it is visible, so no frame
    # is stable and the evidence counter never converts to a commit.
    # (Honest limit: a bounce-out that rested perfectly still for one
    # full frame after one frame of evidence WOULD commit at
    # dart_stable_frames=2 -- the Detection time slider trades this off.)
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = run(lc, s, 1)
    s.darts = [(CX + 30, CY - 55, 16, 100)]   # still bouncing
    ticks += run(lc, s, 1)
    s.darts.clear()
    ticks += run(lc, s, 20)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert lc.dart_count == 0
    assert lc.phase is Phase.IDLE


def test_hand_moving_in_suppresses_and_no_commit_from_hand_over_board():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    ticks = []
    # arm reaches in from the left edge to the board over 8 frames
    for i in range(8):
        s.blobs = [(0, CY - 40, 60 + i * 40, 80)]
        ticks.append(lc.observe({0: s.render()}))
    assert any(t.action is Action.HAND_ENTER for t in ticks)
    assert lc.phase is Phase.HAND
    assert not any(t.action is Action.COMMIT for t in ticks)
    # arm holds still over the board for a moment, then withdraws
    ticks = run(lc, s, 3)
    assert not any(t.action is Action.COMMIT for t in ticks)
    for i in range(8):
        s.blobs = [(0, CY - 40, 340 - i * 45, 80)] if 340 - i * 45 > 0 else []
        ticks.append(lc.observe({0: s.render()}))
    s.blobs = []
    ticks += run(lc, s, 12)
    assert any(t.action is Action.HAND_EXIT for t in ticks)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert lc.dart_count == 0
    assert lc.phase is Phase.IDLE


def _arm_in(lc: Lifecycle, s: Scene) -> list:
    ticks = []
    for i in range(8):
        s.blobs = [(0, CY - 40, 60 + i * 40, 80)]
        ticks.append(lc.observe({0: s.render()}))
    assert lc.phase is Phase.HAND
    return ticks


def _arm_out(lc: Lifecycle, s: Scene) -> list:
    ticks = []
    for i in range(8):
        s.blobs = [(0, CY - 40, 340 - i * 45, 80)] if 340 - i * 45 > 0 else []
        ticks.append(lc.observe({0: s.render()}))
    s.blobs = []
    return ticks


def test_dart_placed_by_hand_is_absorbed_not_committed():
    # A board change that exists when the hand leaves was made by the
    # hand. With no darts committed it becomes background (a hand event
    # re-takes the clean image).
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    _arm_in(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = run(lc, s, 3)
    ticks += _arm_out(lc, s)
    ticks += run(lc, s, 15)
    assert any(t.action is Action.HAND_EXIT for t in ticks)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert any(t.action is Action.FORCED_ADOPT and "under hand" in t.reason for t in ticks)
    assert lc.dart_count == 0
    assert lc.phase is Phase.IDLE
    # the placed dart is background now: a thrown one still commits
    s.darts.append((CX + 50, CY - 60, 16, 100))
    ticks = run(lc, s, 10)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1


def test_startup_dart_pulled_by_hand_is_not_a_dart_and_visit_still_counts_three():
    # Shadow session 07:46:43: a dart left in the board at daemon start
    # was in the reference; pulling it by hand read as a new dart, and
    # the real visit's third dart was then refused as a 4th.
    s = Scene()
    s.darts.append((CX + 30, CY - 70, 16, 100))  # already in the board
    lc = new_lifecycle()
    warm(lc, s)
    _arm_in(lc, s)
    s.darts.clear()  # hand pulls it
    ticks = run(lc, s, 3)
    ticks += _arm_out(lc, s)
    ticks += run(lc, s, 15)
    assert not any(t.action is Action.COMMIT for t in ticks), [t.reason for t in ticks if t.action is not Action.NONE]
    assert lc.dart_count == 0
    # a full visit against the now-empty board
    for i, x in enumerate((CX - 50, CX, CX + 50)):
        s.darts.append((x, CY - 60, 16, 100))
        ticks = run(lc, s, 15)
        assert sum(t.action is Action.COMMIT for t in ticks) == 1, i
    assert lc.dart_count == 3


def test_landed_dart_survives_a_hand_that_comes_near_without_touching():
    # The exception to the hand-caused rule: a change pending well before
    # the hand arrived is a landed dart and commits once the hand is gone.
    lc, s = new_lifecycle(LifecycleConfig(warmup_stable_frames=3, dart_stable_frames=100)), Scene()
    warm(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = run(lc, s, 12)  # pending, never stable long enough to commit yet
    assert not any(t.action is Action.COMMIT for t in ticks)
    ticks = _arm_in(lc, s)
    ticks += run(lc, s, 3)
    ticks += _arm_out(lc, s)
    ticks += run(lc, s, 120)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1
    assert not any(t.action is Action.FORCED_ADOPT for t in ticks)
    assert lc.dart_count == 1


def test_hand_nudging_a_committed_dart_clears_the_visit():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    run(lc, s, 15)
    assert lc.dart_count == 1
    _arm_in(lc, s)
    s.darts = [(CX + 40, CY - 60, 16, 100)]  # moved by the hand
    ticks = run(lc, s, 3)
    ticks += _arm_out(lc, s)
    ticks += run(lc, s, 15)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert sum(t.action is Action.CLEARED for t in ticks) == 1
    assert lc.dart_count == 0 and not lc.refs.has_darts()


def test_hand_adding_a_dart_to_a_visit_clears_instead_of_scoring():
    # Not a removal (the committed dart is untouched) and not a throw:
    # the hand did it, so the visit resets and the board is re-based.
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    run(lc, s, 15)
    assert lc.dart_count == 1
    _arm_in(lc, s)
    s.darts.append((CX + 40, CY - 60, 16, 100))
    ticks = run(lc, s, 3)
    ticks += _arm_out(lc, s)
    ticks += run(lc, s, 15)
    assert not any(t.action is Action.COMMIT for t in ticks)
    cleared = [t for t in ticks if t.action is Action.CLEARED]
    assert len(cleared) == 1 and "under hand" in cleared[0].reason
    assert lc.dart_count == 0 and not lc.refs.has_darts()


def test_takeout_count_pauses_on_borderline_frames_instead_of_resetting():
    # Shadow session 07:51:24: one camera flickered 0.697/0.712 around the
    # 0.70 coverage threshold and every dip restarted the count, turning a
    # 0.2 s clear into 4 s. Script the aggregate directly: a non-majority
    # stable frame must pause the count, not zero it.
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    run(lc, s, 15)
    assert lc.dart_count == 1
    frame = s.render()
    frames, smalls = {0: frame}, {0: to_small_gray(frame)}
    sig = lc.observe(frames).signals

    def agg(majority: bool):
        return Lifecycle._Agg(
            stable=True, hand_now=False, scene=False, dart_cams=[],
            union_cams=[0, 1, 2], removal_cams=[0, 1] if majority else [0],
            partial_cams=[0, 1, 2], over_darts=True,
        )

    actions = []
    for majority in (True, True, False, True, False, True, True):
        actions.append(lc._step(agg(majority), sig, frames, smalls).action)
    assert actions[-1] is Action.CLEARED, actions
    assert lc.dart_count == 0


def test_one_removal_camera_is_enough_right_after_a_hand():
    # Shadow session 20260907 g4-001-T3 takeout: cam1 0.694, cam2 0.484 --
    # only one camera ever crossed 0.7. With no hand that is not a
    # takeout; right after a hand it is.
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    run(lc, s, 15)
    frame = s.render()
    frames, smalls = {0: frame}, {0: to_small_gray(frame)}
    sig = lc.observe(frames).signals
    one_cam = Lifecycle._Agg(
        stable=True, hand_now=False, scene=False, dart_cams=[],
        union_cams=[0, 1, 2], removal_cams=[0], partial_cams=[0, 1, 2], over_darts=True,
    )
    actions = [lc._step(one_cam, sig, frames, smalls).action for _ in range(10)]
    assert Action.CLEARED not in actions
    assert lc.dart_count == 1
    lc._c.since_hand = 3  # a hand just left
    actions = [lc._step(one_cam, sig, frames, smalls).action for _ in range(10)]
    assert Action.CLEARED in actions
    assert lc.dart_count == 0


def test_hand_over_committed_darts_holds_then_removal_clears():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    run(lc, s, 15)
    assert lc.dart_count == 1
    # fingers appear on the dart without a big moving arm (soft removal)
    s.blobs = [(CX - 12, CY - 30, 24, 30)]
    ticks = run(lc, s, 8)
    assert not any(t.action is Action.COMMIT for t in ticks), [t.reason for t in ticks]
    # dart pulled, fingers gone
    s.blobs = []
    s.darts.clear()
    ticks = run(lc, s, 12)
    assert any(t.action is Action.CLEARED for t in ticks)
    assert lc.dart_count == 0


def test_slow_exposure_drift_is_absorbed_not_committed():
    # Under continuous refresh the reference is at most one frame old on
    # a quiet board, so a 0.5 gray/frame ramp never shows more than 0.5
    # gray of difference -- board_px stays at zero throughout instead of
    # accumulating toward the dart floor.
    lc, s = new_lifecycle(LifecycleConfig(warmup_stable_frames=3)), Scene()
    warm(lc, s)
    ticks = []
    for i in range(120):
        s.level = 120 + i * 0.5  # +60 gray levels over 4 s
        ticks.append(lc.observe({0: s.render()}))
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert max(t.signals[0].board_px for t in ticks) < 80
    assert lc.dart_count == 0
    # and a real dart is still detected afterwards
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = run(lc, s, 10)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1


def test_lighting_step_is_scene_change_not_dart():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.level = 200
    ticks = run(lc, s, 20)
    assert any(t.action is Action.SCENE_CHANGE for t in ticks)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert lc.phase is Phase.IDLE
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = run(lc, s, 10)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1


def test_permanent_outside_change_does_not_wedge_hand_and_dart_still_commits():
    """A large static change outside the board (an object placed in view,
    a light) must not keep the machine in HAND forever; after the static
    timeout the outside is absorbed and a dart on the board still commits
    against the pre-change board reference."""
    lc, s = new_lifecycle(LifecycleConfig(warmup_stable_frames=3, hand_static_max_frames=10)), Scene()
    warm(lc, s)
    s.blobs = [(0, 0, 190, 260)]  # a bright block left of the board, ~3000 small px
    ticks = run(lc, s, 30)
    assert any(t.action is Action.HAND_ENTER for t in ticks)
    exits = [t for t in ticks if t.action is Action.HAND_EXIT]
    assert exits and "absorbed" in exits[0].reason
    assert lc.phase is Phase.IDLE
    assert not any(t.action is Action.COMMIT for t in ticks)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = run(lc, s, 12)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1


def test_moderate_static_outside_change_is_absorbed_by_refresh():
    lc, s = new_lifecycle(LifecycleConfig(warmup_stable_frames=3, idle_readopt_frames=5)), Scene()
    warm(lc, s)
    s.blobs = [(0, 0, 640, 40)]  # ~1600 small px: below the static-arm rule, not moving
    ticks = run(lc, s, 12)
    assert not any(t.action in (Action.HAND_ENTER, Action.COMMIT) for t in ticks)
    assert any(t.action is Action.READOPT for t in ticks)
    assert ticks[-1].signals[0].outside_px == 0
    s.darts.append((CX - 2, CY - 60, 16, 100))
    ticks = run(lc, s, 12)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1


def test_reference_refreshes_on_stable_non_dart_frame():
    """The reference refreshes on EVERY stable frame with no dart camera,
    not on a cadence: a static sub-threshold change (~60 small px, below
    dart_min_px) is absorbed within a couple of frames even with the
    fallback cadence configured absurdly high."""
    lc, s = new_lifecycle(LifecycleConfig(warmup_stable_frames=3, idle_readopt_frames=1000)), Scene()
    warm(lc, s)
    s.darts.append((CX - 4, CY - 40, 12, 80))  # ~60 small px: artifact-sized
    ticks = run(lc, s, 4)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert any(t.action is Action.READOPT for t in ticks[:3])
    assert ticks[-1].signals[0].board_px == 0  # absorbed, not pending
    assert lc.dart_count == 0
    # a real dart still commits against the refreshed reference
    s.darts.append((CX + 30, CY - 60, 16, 100))
    ticks = run(lc, s, 10)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1


def test_reference_still_refreshes_under_outside_churn():
    """Fallback cadence: a small moving change outside the board (below
    the hand thresholds) keeps ``stable`` false indefinitely, so the
    per-stable-frame refresh never fires -- but the board itself is
    delta-quiet, and the fallback still absorbs board residue so drift
    cannot stack underneath the churn."""
    lc, s = new_lifecycle(LifecycleConfig(warmup_stable_frames=3, idle_readopt_frames=5)), Scene()
    warm(lc, s)
    s.darts.append((CX - 4, CY - 40, 12, 80))  # ~60 px board residue, below dart_min_px
    ticks = []
    for i in range(15):
        # 32x32 blob hopping around far outside the board: outside_px ~64
        # (< hand_min_outside_px), outside_delta ~128 (> stability floor)
        s.blobs = [(40 + (i % 3) * 60, 30, 32, 32)]
        ticks.append(lc.observe({0: s.render()}))
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert any(
        t.action is Action.READOPT and "churn" in t.reason for t in ticks
    ), [(t.action, t.stable, t.reason) for t in ticks]
    assert ticks[-1].signals[0].board_px == 0


def test_single_camera_sub_threshold_change_never_commits():
    """Ghost shape (room light switched off): ~67 changed board px in ONE
    camera, literal zero in the others, no hand, fast settle. Below
    dart_min_px it must not commit, and refresh absorbs it instead."""
    masks = {0: board_mask(), 1: board_mask()}
    lc = Lifecycle(masks, LifecycleConfig(warmup_stable_frames=3))
    s0, s1 = Scene(), Scene()
    for _ in range(6):
        lc.observe({0: s0.render(), 1: s1.render()})
    assert lc.phase is Phase.IDLE
    s0.darts.append((CX - 4, CY - 40, 12, 80))  # ~60 px, cam 0 only
    ticks = [lc.observe({0: s0.render(), 1: s1.render()}) for _ in range(10)]
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert lc.dart_count == 0
    assert ticks[-1].signals[0].board_px == 0


def test_adjacent_dart_landing_against_committed_darts_commits():
    """Missed-dart shape: a third dart lands touching the committed ones,
    so most of its pixels fall inside the committed-dart grown union.
    The union-subtracted remainder is small and fragmented -- the old
    board_new gate (50 px total / 30 px blob) rejected exactly this dart
    and it was absorbed as background, never scored. The full board
    change is still dart-sized and must commit."""
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    for x in (CX - 32, CX + 16):
        s.darts.append((x, CY - 60, 16, 100))
        ticks = run(lc, s, 15)
        assert sum(t.action is Action.COMMIT for t in ticks) == 1
    assert lc.dart_count == 2
    s.darts.append((CX - 60, CY - 8, 120, 16))  # crosses both darts
    ticks = run(lc, s, 10)
    commits = [t for t in ticks if t.action is Action.COMMIT]
    assert len(commits) == 1, [t.reason for t in ticks if t.action is not Action.NONE]
    sig = commits[0].signals[0]
    # the shape that killed the old gate: full change dart-sized, union-
    # subtracted remainder fragmented below the old 30-px blob floor
    assert sig.board_px >= lc.cfg.dart_min_px
    assert sig.board_new_blob_px < 30
    assert lc.dart_count == 3


def test_partial_removal_does_not_commit_a_phantom():
    """With the union no longer subtracted from the dart test, a
    departing dart's own pixels are a large board change. A camera
    mid-removal (union coverage 0.25-0.7, no hand seen) must read as
    removal in progress -- its change is dominated by the committed union
    (board_new_px < in_union_px) -- never as a new dart."""
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    for x in (CX - 40, CX + 24):
        s.darts.append((x, CY - 60, 16, 100))
        ticks = run(lc, s, 15)
        assert sum(t.action is Action.COMMIT for t in ticks) == 1
    assert lc.dart_count == 2
    s.darts.pop()  # one of two darts silently gone: coverage ~0.5
    ticks = run(lc, s, 55)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert any(t.action is Action.PARTIAL_ADOPT for t in ticks)
    assert lc.dart_count == 2  # a partial removal keeps the visit


def test_tick_as_dict_is_json_friendly():
    import json

    lc, s = new_lifecycle(), Scene()
    t = run(lc, s, 1)[0]
    json.dumps(t.as_dict())


def test_multi_camera_any_camera_dart_and_all_camera_stability():
    masks = {0: board_mask(), 1: board_mask()}
    lc = Lifecycle(masks, LifecycleConfig(warmup_stable_frames=3))
    s0, s1 = Scene(), Scene()
    for _ in range(6):
        lc.observe({0: s0.render(), 1: s1.render()})
    assert lc.phase is Phase.IDLE
    s1.darts.append((CX - 2, CY - 60, 16, 100))  # visible in cam 1 only
    ticks = [lc.observe({0: s0.render(), 1: s1.render()}) for _ in range(8)]
    commits = [t for t in ticks if t.action is Action.COMMIT]
    assert len(commits) == 1 and commits[0].commit.dart_cams == [1]
    assert set(commits[0].commit.bg) == {0, 1}


def test_reset_forgets_darts():
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.darts.append((CX - 2, CY - 60, 16, 100))
    run(lc, s, 15)
    assert lc.dart_count == 1
    lc.reset()
    assert lc.dart_count == 0 and lc.phase is Phase.WARMUP
    run(lc, s, 6)
    assert lc.phase is Phase.IDLE


def _two_cam_idle() -> tuple[Lifecycle, Scene, Scene]:
    masks = {0: board_mask(), 1: board_mask()}
    lc = Lifecycle(masks, LifecycleConfig(warmup_stable_frames=3))
    s0, s1 = Scene(), Scene()
    for _ in range(6):
        lc.observe({0: s0.render(), 1: s1.render()})
    assert lc.phase is Phase.IDLE
    return lc, s0, s1


def _commit_dart1_two_cam(lc: Lifecycle, s0: Scene, s1: Scene) -> tuple[int, int, int, int]:
    dart1 = (CX - 2, CY - 60, 16, 100)
    s0.darts.append(dart1)
    s1.darts.append(dart1)
    ticks = [lc.observe({0: s0.render(), 1: s1.render()}) for _ in range(15)]
    assert sum(t.action is Action.COMMIT for t in ticks) == 1
    assert lc.dart_count == 1
    assert lc.phase is Phase.IDLE
    return dart1


def test_grouped_dart_over_union_commits_after_hold():
    lc, s0, s1 = _two_cam_idle()
    _commit_dart1_two_cam(lc, s0, s1)
    # cam 0: bright blob overlapping dart 1's mask so union_coverage >= 0.5
    s0.blobs = [(CX - 6, CY - 50, 28, 55)]
    # cam 1: a normal new dart, well away from dart 1
    s1.darts.append((CX + 50, CY - 60, 16, 100))
    budget = lc.cfg.dart_stable_frames + lc.cfg.over_darts_hold_frames + 2
    ticks = [lc.observe({0: s0.render(), 1: s1.render()}) for _ in range(budget + 8)]
    commits = [t for t in ticks if t.action is Action.COMMIT]
    assert len(commits) == 1, [(t.action, t.reason, t.signals[0].union_coverage) for t in ticks]
    assert ticks.index(commits[0]) < budget
    assert ticks.index(commits[0]) < lc.cfg.dart_force_frames
    assert commits[0].dart_count == 2
    assert commits[0].reason == "over darts, stable"
    assert lc.dart_count == 2


def test_hand_over_committed_darts_still_holds():
    lc, s0, s1 = _two_cam_idle()
    _commit_dart1_two_cam(lc, s0, s1)
    s1.darts.append((CX + 50, CY - 60, 16, 100))
    ticks = []
    for i in range(10):
        arm = (0, CY - 40, 60 + i * 40, 80)
        s0.blobs = [(CX - 6, CY - 50, 28, 55), arm]
        s1.blobs = [arm]
        ticks.append(lc.observe({0: s0.render(), 1: s1.render()}))
    assert not any(t.action is Action.COMMIT for t in ticks), [t.reason for t in ticks]
    assert (
        lc.phase is Phase.HAND
        or any(t.action is Action.HAND_ENTER for t in ticks)
        or any(t.reason == "held: change over committed darts" for t in ticks)
    )


def test_over_darts_hold_resets_on_motion():
    masks = {0: board_mask(), 1: board_mask()}
    cfg = LifecycleConfig(warmup_stable_frames=3, dart_force_frames=12)
    lc = Lifecycle(masks, cfg)
    s0, s1 = Scene(), Scene()
    for _ in range(6):
        lc.observe({0: s0.render(), 1: s1.render()})
    dart1 = (CX - 2, CY - 60, 16, 100)
    s0.darts.append(dart1)
    s1.darts.append(dart1)
    ticks = [lc.observe({0: s0.render(), 1: s1.render()}) for _ in range(15)]
    assert sum(t.action is Action.COMMIT for t in ticks) == 1
    assert lc.phase is Phase.IDLE
    s0.blobs = [(CX - 6, CY - 50, 28, 55)]
    ticks = []
    for i in range(cfg.dart_force_frames + 6):
        xoff = (i // 2) % 2 * 8
        s1.darts = [dart1, (CX + 50 + xoff, CY - 60, 16, 100)]
        ticks.append(lc.observe({0: s0.render(), 1: s1.render()}))
    commits = [t for t in ticks if t.action is Action.COMMIT]
    assert len(commits) == 1, [(i, t.action, t.reason) for i, t in enumerate(ticks)]
    assert commits[0].reason == "forced"
    assert commits[0].commit is not None and commits[0].commit.forced
    assert not any(t.reason == "over darts, stable" for t in ticks)
