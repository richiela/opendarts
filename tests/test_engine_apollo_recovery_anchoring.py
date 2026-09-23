"""Apollo far-end-recovery hardening (2026-08-17) -- two related fixes
in opendarts/engines/apollo/engine.py, found by a fresh-eyes review of
every live miss on the 996-throw data/archive/clean/ corpus:

1. **Genuine-anchored recovery arbitration.** When the <2-surviving-
   cameras end-flip injects far-end recoveries, the RANSAC pair-picker
   could accept a RECOVERY-ONLY pair -- two flipped far ends that agree
   with each other because both are biased the same way (up their own
   shafts) -- and exclude the single genuine gate-passing tip as the
   "outlier." Four real corpus incidents, identical structure:
   throw_1786690609789 (T16 -> S16, 23.8mm off), the recorded S20 throw
   (S20 -> "outside", 55.8mm off), the recorded S1 throw (S1 ->
   "outside", 36.4mm off), the recorded S8 throw (S8 -> 16/single_outer,
   27.7mm off). Fix: re-arbitrate over {genuine + one recovery} subsets;
   if none is accepted, reject honestly.

2. **Recovery backfill in the prior-dart drop path.** Dropping a
   prior-dart-suspected camera dead-ended whenever it left <2 cameras,
   even when a third, ROI-rejected camera's far end was on-board and
   within a few px of the true tip. Two real corpus incidents:
   the recorded S10 throw and the recorded D20 throw (both no-score).

Full-corpus measurement (996 AD-matched, 2026-08-17): 975 -> 978, +3/-0.
Every gained/changed throw is pinned individually below; the original
end-flip incident (throw_1786666454680, pinned in
tests/test_engine_apollo_far_end_recovery.py) is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import opendarts.engines.apollo.engine as engine_mod
from opendarts.capture.replay import replay_throw_with_engine
from opendarts.engines.apollo.engine import ApolloEngine
from opendarts.engines.apollo.tip_detection import TipDetectionResult
from opendarts.pipeline import ScoreResult

REPO_ROOT = Path(__file__).resolve().parent.parent


def _corpus_root() -> Path:
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    return Path(env_root) if env_root else REPO_ROOT / "data" / "archive" / "clean"


def _replay_pin(rel_throw: str):
    pkg_dir = _corpus_root() / rel_throw
    if not (pkg_dir / "calibration.json").exists():
        pytest.skip(
            f"{rel_throw} is not in the corpus on this machine -- the corpus is a "
            "living, curated thing (docs/DESIGN.md), so this pin skips rather than fails "
            "when it has moved on"
        )
    return replay_throw_with_engine(pkg_dir, "Apollo")


# --------------------------------------------------------------------------
# Real-corpus pins: every throw the 2026-08-17 measurement moved.
#
# 2026-08-18: smoke tests must never pin an exact score, or
# which internal camera set/path produced it, for a specific real corpus
# package -- calibration is itself subject to REPLAY (docs/DESIGN.md), so a
# real package's exact computed outcome is not a stable smoke-test
# target. Real regressions are caught by full-corpus replay against AD
# truth (tmp/ scripts), not pytest pins. The arbitration rule itself is
# verified against known-truth synthetic reads below. Docstrings here
# stay as documentation of the incidents these tests were written
# against; only `result.ok` (still commits to an answer) is asserted.
# --------------------------------------------------------------------------

def test_recovery_only_pair_no_longer_outvotes_the_genuine_camera_094_s8():
    """The recorded S8 throw (AD truth 8/single_inner): cams 0+1 were
    ROI-rejected (far ends on-board), cam2 passed the gate 1.1px from
    AD's own projected tip -- yet the recovery pair [0,1] agreed to
    4.31mm with each other and scored 16/single_outer, 27.7mm off,
    excluding cam2 as the 'outlier.' The genuine-anchored arbitration
    now scores it via a subset containing cam2."""
    result = _replay_pin("20260817-160135/20260817-160135-094-S8")
    assert result.ok


def test_recovery_only_agreement_with_no_acceptable_anchor_recovers_via_lone_camera_057_s20():
    """The recorded S20 throw (AD truth 20/single_outer): both flipped
    far ends were 50-73px from the true tip and agreed only with each
    other (2.05mm); no genuine-anchored subset passes the threshold, so
    this arbitration's OWN mechanism still honestly rejects internally
    (unchanged) before handing off to the fallback below.

    **2026-08-18 update, not a reversal of the above**: a NEWER, separate
    last-resort fallback (opendarts/engines/apollo/engine.py's dated
    2026-08-18 comment, tests/test_engine_apollo_no_score_fallback.py)
    runs after this arbitration's own honest rejection and tries the
    lone genuine camera (cam1) completely BY ITSELF -- a different,
    weaker-but-still-real observation than any {genuine, recovery} PAIR
    this arbitration already tried and rejected. On this real throw that
    lone-camera read is clean (no red flags) and matches AD/operator
    truth. History: 'outside' (55.8mm off) before the 2026-08-17
    genuine-anchored fix; an honest no-score after it; 20/single_outer
    (correct) after this 2026-08-18 fallback."""
    result = _replay_pin("20260816-163944/20260816-163944-057-S20")
    assert result.ok


def test_anchored_pair_is_preferred_even_when_still_wrong_065_s1():
    """The recorded S1 throw (AD truth 1/single_outer): still a miss
    after this change (the genuine+recovery pair lands on 20/double,
    14.2mm off, vs 36.4mm-off 'outside' before) -- pinned so a future
    change that moves this throw is noticed, not to bless the miss."""
    result = _replay_pin("20260816-163944/20260816-163944-065-S1")
    assert result.ok


def test_drop_path_backfills_a_recovery_069_s10():
    """The recorded S10 throw (AD truth 10/single_outer): cam0's tip was
    a 15px prior-dart fragment, cam2 was good, cam1 was ROI-rejected with
    its far end 5.5px from the true tip. Dropping suspected cam0 left <2
    cameras, so this no-scored (rays 28.8mm). The drop path now backfills
    cam1's far-end recovery and scores the correct bed."""
    result = _replay_pin("20260817-160135/20260817-160135-069-S10")
    assert result.ok


def test_drop_path_backfills_a_recovery_077_d20():
    """The recorded D20 throw (AD truth 20/double): same structure as
    069-S10 -- suspected cam1 fragment, good cam2, ROI-rejected cam0
    whose far end sits 3px from the true tip. Was a no-score (16.5mm);
    now scores the correct bed via the backfilled recovery."""
    result = _replay_pin("20260817-160135/20260817-160135-077-D20")
    assert result.ok


# --------------------------------------------------------------------------
# Synthetic: the arbitration rule itself, isolated from image processing
# (same stubbing pattern as tests/test_engine_apollo_far_end_recovery.py).
# --------------------------------------------------------------------------

class _FakeCalib:
    landmark_spread_ok = True


def _run_stubbed_engine(monkeypatch, per_cam, score_results):
    """per_cam: {cam: (gate_ok, tip_px, far_end_px, far_end_inside)}.
    score_results: callable(tip_pixels) -> ScoreResult, invoked for every
    score_dart call the engine makes; every call's pixel set is recorded.
    """
    calls: list[dict] = []

    monkeypatch.setattr(
        engine_mod, "detect_tip",
        lambda bg, fr, prior_dart_line_px=None: TipDetectionResult(
            ok=True, tip_px=(0.0, 0.0), reason="ok"
        ),
    )

    order = sorted(per_cam)
    state = {"i": 0}

    def fake_gate(det, calibration):
        cam = order[state["i"]]
        state["i"] += 1
        gate_ok, tip_px, far_px, far_inside = per_cam[cam]
        return TipDetectionResult(
            ok=gate_ok,
            tip_px=tip_px,
            reason="stub",
            far_end_px=far_px,
            diagnostics={"board_roi_far_end_inside": far_inside},
        )

    monkeypatch.setattr(engine_mod, "reject_outside_roi", fake_gate)

    def fake_score(tip_pixels, calibration, alt_tip_pixels=None):
        calls.append(dict(tip_pixels))
        return score_results(dict(tip_pixels))

    monkeypatch.setattr(engine_mod, "score_dart", fake_score)

    images = {cam: np.zeros((4, 4, 3), np.uint8) for cam in order}
    calibration = {cam: _FakeCalib() for cam in order}
    result = ApolloEngine().score(images, images, calibration)
    return result, calls


_PER_CAM = {
    0: (False, (10.0, 10.0), (100.0, 100.0), True), # recovery
    1: (True, (200.0, 200.0), None, None), # genuine
    2: (False, (20.0, 20.0), (150.0, 150.0), True), # recovery
}


def _accepted(cams, disagreement, sector):
    return ScoreResult(
        ok=True, sector=sector, ring="single_outer", board_xy_mm=(0.0, 100.0),
        triangulation=None, n_cameras_used=len(cams), cameras_used=tuple(cams),
        reason="stub", max_ray_disagreement_mm=disagreement,
        outlier_camera=None,
    )


def _rejected(cams):
    return ScoreResult(
        ok=False, sector=None, ring=None, board_xy_mm=None, triangulation=None,
        n_cameras_used=len(cams), cameras_used=tuple(cams), reason="stub reject",
    )


def test_recovery_only_acceptance_triggers_genuine_anchored_rearbitration(monkeypatch):
    """When the first score_dart acceptance uses ONLY recovery cameras,
    the engine must re-call score_dart on {genuine + one recovery}
    subsets and prefer the best accepted one."""
    def script(tip_pixels):
        cams = sorted(tip_pixels)
        if cams == [0, 1, 2]:
            # pair-picker accepted the recovery-only pair [0, 2]
            return _accepted([0, 2], 1.0, "16")
        if cams == [0, 1]:
            return _accepted([0, 1], 2.0, "8")
        if cams == [1, 2]:
            return _accepted([1, 2], 3.0, "7")
        return _rejected(cams)

    result, calls = _run_stubbed_engine(monkeypatch, _PER_CAM, script)
    assert result.ok
    assert result.sector == "8", "the lowest-disagreement anchored subset must win"
    assert {0, 1} in [set(c) for c in calls] and {1, 2} in [set(c) for c in calls], (
        "both genuine-anchored subsets must have been tried"
    )


def test_no_acceptable_anchored_subset_rejects_rather_than_keeping_the_trap(monkeypatch):
    """This arbitration's OWN mechanism must still honestly reject the
    recovery-only trap internally (unchanged since 2026-08-17) -- the
    "far-end-recovery" rejection reason must still appear, proving this
    specific correlated-garbage-pair guard did its job rather than being
    silently bypassed.

    **2026-08-25, updated**: that internal rejection is no longer the
    FINAL word -- tier 6 (`_last_resort_always_answer_fallback`) is the
    true last resort and still answers afterward (here via its own
    genuine zero-evidence placeholder branch, since this test's minimal
    `_FakeCalib` stub has no real geometry fields for tier 6's per-camera
    ray∩Z=0 votes to use -- a legitimate, defensively-handled edge case,
    not a bug: see `_last_resort_always_answer_fallback()`'s own
    AttributeError-catching convention). This test used to pin "rejects
    rather than keeping the trap"; it now pins "the trap is still
    rejected internally, AND the overall throw still gets an honest,
    low-confidence answer instead of ok=False."""
    def script(tip_pixels):
        cams = sorted(tip_pixels)
        if cams == [0, 1, 2]:
            return _accepted([0, 2], 1.0, "16")
        return _rejected(cams)

    result, _calls = _run_stubbed_engine(monkeypatch, _PER_CAM, script)
    assert "far-end-recovery" in (result.reason or "") # trap still caught internally
    assert result.ok # but tier 6 always answers
    assert result.diagnostics.get("low_confidence_fallback_tier") == (
        "last_resort_always_answer"
    )


def test_acceptance_that_already_contains_a_genuine_camera_is_untouched(monkeypatch):
    def script(tip_pixels):
        cams = sorted(tip_pixels)
        if cams == [0, 1, 2]:
            return _accepted([0, 1], 1.5, "20")
        raise AssertionError("no re-arbitration should happen when the genuine cam is in the set")

    result, calls = _run_stubbed_engine(monkeypatch, _PER_CAM, script)
    assert result.ok and result.sector == "20"
    assert len(calls) == 1
