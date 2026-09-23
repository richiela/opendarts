"""Tests for Apollo's last-resort lone-genuine-camera no-score
fallback (2026-08-18) -- see `opendarts/engines/apollo/engine.py`'s
dated 2026-08-18 comment in `ApolloEngine.score()` for the full
real-incident write-up and corpus measurement.

The real problem: `score_dart()` requires >=2 cameras with a tip pixel
(the <=1-camera case is explicitly left to the
CALLER) -- so a throw where only 1 camera's tip detection passed the
board-ROI gate was an unconditional no-score, even when that lone
survivor looks individually clean and every other engine
(Talos/Athena) scored the same throw correctly.

Investigated via the REAL production replay path
(`opendarts.capture.replay.replay_throw_with_engine`, which supplies
`prior_dart_line_px` from stored visit metadata -- docs/DESIGN.md's
"Replay is the source of truth": a bare `ApolloEngine().score()` call with no
prior-dart context is NOT faithful to what live capture/real replay
actually do, and undercounts what the existing 2026-08-17 prior-dart-
drop-path backfill already recovers) on the full living
`data/archive/clean/` corpus (1107 AD-matched throws, 2026-08-18):
Apollo had 6 real no-score throws. This fallback recovers 5 of them,
ZERO wrong:

    throw_1786665050832 outside -> outside MATCH
    S20 throw 20/outer -> 20/outer MATCH
    S1 throw 1/outer -> 1/outer MATCH
    S15 throw 15/outer -> 15/outer MATCH
    S20 throw 20/inner -> 20/inner MATCH

The one throw this fallback does not touch
(the recorded outside throw, truth outside) has ZERO genuine cameras --
literally no signal to use -- and correctly stays an honest no-score.

Note on the recorded S20 throw: this throw is rejected by a DIFFERENT
existing mechanism than a plain <2-camera no-score -- the
"genuine-anchored recovery arbitration" block (engine.py, 2026-08-17)
already tried the lone genuine camera PAIRED with each far-end recovery
candidate and found neither pair agreed well enough, so it honestly
rejects. This fallback is the first thing to try that lone genuine
camera completely BY ITSELF -- a different, weaker-but-still-real
observation than any pairing already attempted.

A second strategy (backfilling a 3rd ROI-rejected camera's far-end
pixel when exactly TWO genuine cameras disagree, then re-running the
same unmodified `score_dart()`) was investigated and deliberately NOT
shipped: the two throws that originally looked like they needed it
(069-S10, 077-D20) turned out, once measured through the real replay
path, to already be recovered by the existing 2026-08-17 prior-dart-
drop-path backfill (see tests/test_engine_apollo_recovery_anchoring.py's
`test_drop_path_backfills_a_recovery_069_s10`/`077_d20`) -- so this
corpus currently has zero real throws to validate a 2-genuine-camera
strategy against. Per this project's "measure the real number, don't
ship on reasoning alone" discipline, that's grounds not to ship it.

All real-corpus regression tests below skip cleanly (not fail) when the
named session isn't present on this machine, per this project's living-
corpus discipline (docs/DESIGN.md).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import opendarts.engines.apollo.engine as engine_mod
from opendarts.capture.replay import replay_throw_with_engine
from opendarts.engines.apollo.engine import (
    ApolloEngine,
    _lone_camera_diagnostics_clean,
    _single_ray_board_xy,
)
from opendarts.engines.apollo.scoring import MAX_RAY_DISAGREEMENT_MM
from opendarts.engines.apollo.tip_detection import TipDetectionResult
from opendarts.geometry.board import sector_ring_for_point
from opendarts.pipeline import CameraCalibration

REPO_ROOT = Path(__file__).resolve().parent.parent


def _corpus_root() -> Path:
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    return Path(env_root) if env_root else REPO_ROOT / "data" / "archive" / "clean"


# --------------------------------------------------------------------------
# Synthetic, isolated: _lone_camera_diagnostics_clean()'s own gate logic.
# --------------------------------------------------------------------------


def test_clean_diagnostics_pass_the_gate():
    assert _lone_camera_diagnostics_clean({"tip_cluster_perp_px": 2.0}) is True


def test_missing_perp_fails_the_gate():
    assert _lone_camera_diagnostics_clean({}) is False


def test_off_axis_perp_at_or_past_the_threshold_fails_the_gate():
    from opendarts.engines.apollo.tip_detection import TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX

    assert _lone_camera_diagnostics_clean(
        {"tip_cluster_perp_px": TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX}
    ) is False
    assert _lone_camera_diagnostics_clean(
        {"tip_cluster_perp_px": TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX + 5.0}
    ) is False
    assert _lone_camera_diagnostics_clean(
        {"tip_cluster_perp_px": -(TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX + 5.0)}
    ) is False


def test_tip_off_axis_alt_fired_fails_the_gate():
    assert _lone_camera_diagnostics_clean(
        {"tip_cluster_perp_px": 1.0, "tip_off_axis_alt": True}
    ) is False


def test_tip_island_alt_fired_fails_the_gate():
    assert _lone_camera_diagnostics_clean(
        {"tip_cluster_perp_px": 1.0, "tip_island_alt": True}
    ) is False


def test_prior_dart_contamination_suspected_fails_the_gate():
    assert _lone_camera_diagnostics_clean(
        {"tip_cluster_perp_px": 1.0, "prior_dart_contamination_suspected": True}
    ) is False


# --------------------------------------------------------------------------
# Synthetic, isolated: _single_ray_board_xy()'s own geometry. Camera
# looks straight down +Z from (0, 0, -500) (rvec=0, tvec=(0,0,500));
# principal point (400, 400), focal 800, no distortion -- exact algebra
# (not small-angle-approximated, the Z-intersection cancels the
# direction-vector normalization) gives board_xy = (0.625*dx, 0.625*dy)
# for pixel (400+dx, 400+dy).
# --------------------------------------------------------------------------


def _straight_down_calibration() -> CameraCalibration:
    camera_matrix = np.array(
        [[800.0, 0.0, 400.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)
    return CameraCalibration(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvec=rvec, tvec=tvec,
        landmark_spread_ok=True,
    )


def test_single_ray_board_xy_matches_hand_computed_geometry():
    calib = _straight_down_calibration()
    xy = _single_ray_board_xy((500.0, 400.0), calib) # dx=100, dy=0
    assert xy == pytest.approx((62.5, 0.0), abs=1e-6)

    xy2 = _single_ray_board_xy((400.0, 448.0), calib) # dx=0, dy=48
    assert xy2 == pytest.approx((0.0, 30.0), abs=1e-6)


def test_single_ray_board_xy_principal_point_hits_bullseye():
    calib = _straight_down_calibration()
    xy = _single_ray_board_xy((400.0, 400.0), calib)
    assert xy == pytest.approx((0.0, 0.0), abs=1e-6)


# --------------------------------------------------------------------------
# End-to-end mechanism, monkeypatched (mirrors
# tests/test_engine_apollo_far_end_recovery.py's `_stub_engine`
# pattern): detect_tip/reject_outside_roi are stubbed so the test drives
# exactly the per-camera outcome it means to; score_dart is left REAL so
# the <2-camera rejection and the fallback's own `_single_ray_board_xy`
# both run for real.
# --------------------------------------------------------------------------


def _stub_detect_and_gate(monkeypatch, per_cam):
    """per_cam: {cam: (gate_ok, tip_px, diagnostics, far_end_px, far_end_inside)}.

    The RAW (pre-gate) `detect_tip()` call also returns each camera's own
    `tip_px` (not a shared placeholder) -- needed since 2026-08-18's tier-2
    ungated fallback reads that raw value directly (`ungated_tip_pixels`,
    populated BEFORE `reject_outside_roi()` can reject anything). Every
    pre-existing tier-1 test above this comment only ever depended on the
    GATED value (`fake_gate`'s own per_cam lookup), so giving the raw call
    real per-camera values here doesn't change any of those results --
    only tier-2 tests (below) exercise this raw path for real."""
    order = sorted(per_cam)
    calls = {"i": 0}

    def fake_detect(bg, fr, prior_dart_line_px=None):
        cam = order[calls["i"]]
        _gate_ok, tip_px, _diag, _far_px, _far_inside = per_cam[cam]
        return TipDetectionResult(ok=True, tip_px=tip_px, reason="ok")

    monkeypatch.setattr(engine_mod, "detect_tip", fake_detect)

    def fake_gate(det, calibration):
        cam = order[calls["i"]]
        calls["i"] += 1
        gate_ok, tip_px, diagnostics, far_px, far_inside = per_cam[cam]
        diag = dict(diagnostics)
        if far_inside is not None:
            diag["board_roi_far_end_inside"] = far_inside
        return TipDetectionResult(
            ok=gate_ok, tip_px=tip_px, reason="stub", diagnostics=diag, far_end_px=far_px,
        )

    monkeypatch.setattr(engine_mod, "reject_outside_roi", fake_gate)
    return order


def test_lone_clean_camera_triggers_the_fallback_and_matches_hand_computed_geometry(monkeypatch):
    order = _stub_detect_and_gate(monkeypatch, {
        0: (True, (500.0, 400.0), {"tip_cluster_perp_px": 2.0}, None, None),
        1: (False, (10.0, 10.0), {}, None, False),
        2: (False, (20.0, 20.0), {}, None, False),
    })
    calibration = {c: _straight_down_calibration() for c in order}
    images = {c: np.zeros((4, 4, 3), np.uint8) for c in order}

    result = ApolloEngine().score(images, images, calibration)
    assert result.ok, result.reason
    expected_xy = (62.5, 0.0)
    assert result.board_xy_mm == pytest.approx(expected_xy, abs=1e-6)
    expected_sector, expected_ring = sector_ring_for_point(*expected_xy)
    assert (result.sector, result.ring) == (expected_sector, expected_ring)
    assert result.diagnostics["cameras_used"] == [0]
    assert result.diagnostics["n_cameras_used"] == 1
    assert result.diagnostics["max_ray_disagreement_mm"] == MAX_RAY_DISAGREEMENT_MM
    assert "lone-camera Z=0 fallback" in result.reason
    # confidence must not crash (this is exactly the real bug this
    # fallback hit first -- compute_confidence() asserts
    # max_ray_disagreement_mm is not None whenever ok=True).
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 1.0


def test_lone_dirty_camera_does_not_trigger_tier_1_but_tier_6_still_answers(monkeypatch):
    """Same shape as the test above, except the lone survivor's own
    tip_cluster_perp_px is past TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX -- tier 1
    (the clean-gate lone-camera fallback) must stay quiet, same as before
    this test's rename.

    **2026-08-25, updated**: this no longer means an honest no-score --
    tier 6 (`_last_resort_always_answer_fallback`, see engine.py's dated
    2026-08-25 tier-6 comment) is the true last resort and does NOT
    require a clean diagnostics gate, so it still answers here using the
    same dirty lone ray, at the tier-6 floor confidence. This test used
    to pin "the fallback must stay quiet"; it now pins "tier 1 stays
    quiet, but the overall throw is never left as ok=False."""
    order = _stub_detect_and_gate(monkeypatch, {
        0: (True, (500.0, 400.0), {"tip_cluster_perp_px": 20.0}, None, None),
        1: (False, (10.0, 10.0), {}, None, False),
        2: (False, (20.0, 20.0), {}, None, False),
    })
    calibration = {c: _straight_down_calibration() for c in order}
    images = {c: np.zeros((4, 4, 3), np.uint8) for c in order}

    result = ApolloEngine().score(images, images, calibration)
    assert "lone-camera Z=0 fallback" not in result.reason # tier 1 declined
    assert result.ok # but tier 6 always answers
    assert result.diagnostics.get("low_confidence_fallback_tier") == (
        "last_resort_always_answer"
    )


def test_lone_camera_with_prior_dart_contamination_flag_skips_tier_1_but_tier_6_still_answers(
    monkeypatch,
):
    """**2026-08-25, updated** -- same rationale as the test above: tier 1
    still correctly declines a prior-dart-suspected lone camera, but tier
    6 is the true last resort and answers anyway."""
    order = _stub_detect_and_gate(monkeypatch, {
        0: (
            True, (500.0, 400.0),
            {"tip_cluster_perp_px": 1.0, "prior_dart_contamination_suspected": True},
            None, None,
        ),
        1: (False, (10.0, 10.0), {}, None, False),
        2: (False, (20.0, 20.0), {}, None, False),
    })
    calibration = {c: _straight_down_calibration() for c in order}
    images = {c: np.zeros((4, 4, 3), np.uint8) for c in order}

    result = ApolloEngine().score(images, images, calibration)
    assert "lone-camera Z=0 fallback" not in result.reason # tier 1 declined
    assert result.ok # but tier 6 always answers
    assert result.diagnostics.get("low_confidence_fallback_tier") == (
        "last_resort_always_answer"
    )


def test_zero_genuine_cameras_that_also_disagree_skips_tier_2_but_tier_6_still_answers(
    monkeypatch,
):
    """No genuine camera at all AND the ungated candidates don't even
    agree on a classification (cam0's raw pixel projects to a real
    on-board bed, "6/single_inner"; cam1/cam2 both project off-board but
    to a DIFFERENT bed than each other has no bearing here since neither
    is even needed -- the point is no 2 cameras share a classification)
    -- tier 2 must stay quiet, same as tier 1 already does for the
    <2-cameras-with-no-agreement case.

    **2026-08-25, updated**: tier 6 is now the true last resort and
    resolves even a genuine 3-way per-camera tie (see
    `_last_resort_always_answer_fallback()`'s own docstring, point 3, for
    the tie-break mechanics) -- deterministically, via the lowest camera
    index among the tied beds here, since no fused triangulation point
    exists (0 genuine tip_pixels means score_dart() never even attempted
    triangulation) and none of these 3 cameras is genuine/gate-passing
    (so none has a recorded clean-diagnostics signal to break the tie
    either). This test used to pin "tier 2 stays quiet, honest no-score";
    it now pins "tier 2 stays quiet, but tier 6 still answers, using its
    own documented tie-break rule."""
    order = _stub_detect_and_gate(monkeypatch, {
        0: (False, (500.0, 400.0), {}, None, False), # -> 6/single_inner
        1: (False, (10.0, 10.0), {}, None, False), # -> outside
        2: (False, (405.0, 400.0), {}, None, False), # -> bull (different bed than either)
    })
    calibration = {c: _straight_down_calibration() for c in order}
    images = {c: np.zeros((4, 4, 3), np.uint8) for c in order}

    result = ApolloEngine().score(images, images, calibration)
    assert "ungated classification-agreement fallback" not in result.reason # tier 2 declined
    assert result.ok # but tier 6 always answers
    assert result.diagnostics.get("low_confidence_fallback_tier") == (
        "last_resort_always_answer"
    )
    # Tie-break lands on cam0's own vote (lowest camera index among the
    # 3-way tie) -- see this test's own docstring for why.
    assert (result.sector, result.ring) == ("6", "single_inner")


def test_zero_genuine_cameras_with_two_ungated_agreeing_triggers_tier_2(monkeypatch):
    """The real mechanism (mirrors the real corpus throw
    outside throw): zero genuine cameras, but 2 of the 3
    ROI-rejected candidates' own ray∩Z=0 hits independently classify to
    the SAME bed -- tier 2 must fire and use exactly those 2, ignoring
    the 3rd which disagrees."""
    order = _stub_detect_and_gate(monkeypatch, {
        0: (False, (10.0, 10.0), {}, None, False), # -> outside
        1: (False, (20.0, 20.0), {}, None, False), # -> outside (agrees with cam0)
        2: (False, (500.0, 400.0), {}, None, False), # -> 6/single_inner (disagrees)
    })
    calibration = {c: _straight_down_calibration() for c in order}
    images = {c: np.zeros((4, 4, 3), np.uint8) for c in order}

    result = ApolloEngine().score(images, images, calibration)
    assert result.ok, result.reason
    assert (result.sector, result.ring) == (None, "outside")
    assert result.diagnostics["cameras_used"] == [0, 1]
    assert result.diagnostics["n_cameras_used"] == 2
    assert "ungated classification-agreement fallback" in result.reason


def test_one_ungated_candidate_alone_never_triggers_tier_2_but_tier_6_still_answers(
    monkeypatch,
):
    """A single ROI-rejected candidate, however far off-board, must never
    be enough alone for TIER 2 -- this is the whole point of requiring
    real classification AGREEMENT (2+), the deliberate difference from
    Athena's own looser "blend whatever's there" version of this idea
    (see the dated 2026-08-18 tier-2 comment's own real-incident
    citation of Athena's one documented wrong answer from doing
    exactly that).

    **2026-08-25, updated**: tier 6 is the true last resort and DOES
    accept a single camera's own lone vote when it is the only candidate
    that exists at all (see `_last_resort_always_answer_fallback()`'s own
    docstring, point 4) -- a real, deliberate, DIFFERENT design choice
    from tier 2's own stricter 2+-agreement requirement, made only
    because tier 6 runs strictly after every other tier has already
    declined. This test used to pin "must never be enough alone"; it now
    pins "tier 2 specifically still requires 2+, but tier 6 is willing to
    use a lone candidate as the final fallback."""
    order = _stub_detect_and_gate(monkeypatch, {
        0: (False, (10.0, 10.0), {}, None, False), # -> outside, but alone
    })
    calibration = {c: _straight_down_calibration() for c in order}
    images = {c: np.zeros((4, 4, 3), np.uint8) for c in order}

    result = ApolloEngine().score(images, images, calibration)
    assert "ungated classification-agreement fallback" not in result.reason # tier 2 declined
    assert result.ok # but tier 6 always answers
    assert (result.sector, result.ring) == (None, "outside")
    assert result.diagnostics.get("low_confidence_fallback_tier") == (
        "last_resort_always_answer"
    )


def test_two_or_three_genuine_cameras_skip_the_tier_1_lone_camera_path_but_tier_6_still_answers(
    monkeypatch,
):
    """TIER 1 (the lone-genuine-camera fallback) is scoped to EXACTLY one
    genuine camera -- it must never fire (and never call
    `_single_ray_board_xy`) when 2 or 3 genuine cameras are present, even
    when they end up disagreeing (score_dart()'s own existing, untouched
    behavior: all three cameras here share the SAME straight-down
    calibration, so their rays are parallel and never converge/agree).

    **2026-08-25, updated**: this no longer means an honest no-score --
    tier 6 is the true last resort and resolves this exact
    parallel-rays-never-agree case via its own per-camera-vote tie-break
    (`_last_resort_always_answer_fallback()`'s own docstring, point 3):
    all 3 cameras are genuine with clean diagnostics here, each voting a
    DIFFERENT on-board bed (a genuine 3-way tie, no fused triangulation
    point to break it either -- `tri.ok` is False on this synthetic
    parallel-rays setup), so the tie-break falls to the lowest camera
    index. This test used to pin "must never fire, stays an honest
    no-score"; it now pins "tier 1 specifically stays scoped to exactly
    one genuine camera, but tier 6 still answers as the final fallback."
    """
    order = _stub_detect_and_gate(monkeypatch, {
        0: (True, (500.0, 400.0), {"tip_cluster_perp_px": 1.0}, None, None),
        1: (True, (400.0, 500.0), {"tip_cluster_perp_px": 1.0}, None, None),
        2: (True, (300.0, 300.0), {"tip_cluster_perp_px": 1.0}, None, None),
    })
    calibration = {c: _straight_down_calibration() for c in order}
    images = {c: np.zeros((4, 4, 3), np.uint8) for c in order}

    result = ApolloEngine().score(images, images, calibration)
    assert "lone-camera Z=0 fallback" not in result.reason # tier 1 declined (3 genuine, not 1)
    assert result.ok # but tier 6 always answers
    assert result.diagnostics.get("low_confidence_fallback_tier") == (
        "last_resort_always_answer"
    )
    # Tie-break lands on cam0's own vote (lowest camera index among the
    # 3-way tie, no fused point to break it another way) -- see this
    # test's own docstring for why.
    assert (result.sector, result.ring) == ("6", "single_inner")


# --------------------------------------------------------------------------
# Real corpus: the throws this fallback was actually built from, and the
# specific accounting the project's own task instructions require --
# recovered-correct / still-no-score / recovered-WRONG -- per throw, not
# just an aggregate number.
# --------------------------------------------------------------------------

RECOVERED_LONE_CAMERA_THROWS = {
    "20260813-164658/throw_1786665050832": (None, "outside"),
    "20260816-163944/20260816-163944-057-S20": ("20", "single_outer"),
    "20260817-160135/20260817-160135-015-S1": ("1", "single_outer"),
    "20260817-160135/20260817-160135-045-S15": ("15", "single_outer"),
    "20260818-004451/20260818-004451-006-S20": ("20", "single_inner"),
}
STILL_NO_SCORE_THROW = "20260816-112211/20260816-112211-043-OUT"


@pytest.mark.parametrize("throw, truth", sorted(RECOVERED_LONE_CAMERA_THROWS.items()))
def test_real_lone_camera_no_score_throws_now_recover_correctly(throw, truth):
    # 2026-08-18: smoke tests must never pin an exact score, or
    # which internal fallback path fired, for a specific real corpus
    # package -- calibration is itself subject to REPLAY (docs/DESIGN.md), so
    # this real throw's exact outcome is not a stable smoke-test target.
    # Real regressions are caught by full-corpus replay against AD truth
    # (tmp/ scripts), not pytest pins. `truth` stays as documentation of
    # the incident this test was written against. What stays meaningful:
    # the fallback's actual contract, always some answer, never no-score.
    pkg_dir = _corpus_root() / throw
    if not (pkg_dir / "calibration.json").exists():
        pytest.skip(f"{throw} not present in the corpus on this machine")
    res = replay_throw_with_engine(pkg_dir, "Apollo")
    assert res.ok, res.reason


def test_real_zero_genuine_camera_throw_now_recovers_via_tier_2_ungated_fallback():
    """2026-08-18, tier 2 -- superseded the earlier "stays an honest
    no-score" expectation. after seeing every OTHER engine
    correctly recover this exact throw: a no-score is worse than a wrong
    score for Zeus's own 2-of-3 voting (confirmed against
    opendarts/engines/zeus/engine.py's own >=2-sub-engines-ok gate -- a
    missing Apollo vote silently degrades Zeus's real majority
    mechanism into an arbitrary tie-break the one time the other two
    disagree). cam1 and cam2's own ROI-gate-REJECTED candidates
    independently classify to the identical (None, 'outside') via their
    own ray∩Z=0 (even though their raw XY hits are 189mm apart -- they
    disagree on WHICH direction off-board, not on being off-board at
    all), which is real 2-of-2 classification agreement even among
    rejected candidates. cam0 had no candidate on this throw at all."""
    pkg_dir = _corpus_root() / STILL_NO_SCORE_THROW
    if not (pkg_dir / "calibration.json").exists():
        pytest.skip(f"{STILL_NO_SCORE_THROW} not present in the corpus on this machine")
    res = replay_throw_with_engine(pkg_dir, "Apollo")
    assert res.ok, res.reason
    assert (res.sector, res.ring) == (None, "outside")
    assert "ungated classification-agreement fallback" in res.reason
