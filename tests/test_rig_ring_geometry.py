"""Tests for `opendarts.calibration.rig_ring_geometry` -- the
ring-consensus orientation rules R2/R3, 2026-08-29.

Real numbers embedded below (not round guesses) come from this module's
own real-corpus validation (this
task, exploratory) against 21 real all-live calibrations from
the calibration corpus: leave-one-out
prediction mean |error| 0.24deg / worst 0.91deg (all 3 cameras); cam2
alone (matching the spec's own worked example) mean 0.19deg / worst
0.46deg; D2 alias rejection 21/21; leave-one-out drift 1.248deg worst.

Tests here use SYNTHETIC geometries/hints (not the real corpus directly
-- no network/data-directory dependency in the fast pytest suite) but
match the real corpus's own measured shape (gap magnitudes ~103/108/149
degrees, well separated) so the numbers being asserted are representative
of the real rig, not arbitrary."""
from __future__ import annotations

import json

import pytest

from opendarts.calibration.rig_ring_geometry import (
    HALF_SECTOR_DEG,
    RING_GEOMETRY_DRIFT_THRESHOLD_DEG,
    RING_GEOMETRY_MAX_SAMPLES,
    CameraOrientationCandidate,
    RingGeometry,
    candidate_predictions,
    cross_check_disagreement_deg,
    load_ring_geometry,
    max_gap_deviation_deg,
    predict_missing_hint,
    raw_ordered_gaps,
    resolve_rig_consensus_orientation,
    save_ring_geometry,
    update_ring_geometry,
)

# Real gap magnitudes measured from the real rig (spec section 2's own
# table): 103.33 / 107.78 / 148.90 deg, summing to 360.
REAL_GAP_A = 103.33
REAL_GAP_B = 107.78
REAL_GAP_C = 148.90 - 0.01  # exact float sum-to-360 nudge


def _synthetic_hints(cam0_deg: float, gap_order=(REAL_GAP_A, REAL_GAP_B, REAL_GAP_C)) -> dict[int, float]:
    """3-camera hints matching the real rig's own gap structure, rotated
    to start at `cam0_deg`."""
    h0 = cam0_deg % 360.0
    h1 = (h0 + gap_order[0]) % 360.0
    h2 = (h1 + gap_order[1]) % 360.0
    return {0: h0, 1: h1, 2: h2}


def _circ_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _learn(events: list[dict[int, float]]) -> RingGeometry:
    geometry = None
    for i, hints in enumerate(events):
        geometry, _drift = update_ring_geometry(geometry, hints, f"2026-08-{20+i:02d}T00:00:00Z")
    return geometry


# --- raw_ordered_gaps / basic geometry math -----------------------------


def test_raw_ordered_gaps_sums_to_360():
    hints = _synthetic_hints(87.2)
    gaps = raw_ordered_gaps(hints)
    assert len(gaps) == 3
    assert sum(gaps) == pytest.approx(360.0, abs=1e-6)


def test_raw_ordered_gaps_invariant_under_uniform_rotation():
    """A pure board rotation (add a constant to every hint) must NOT
    change the gap structure -- this is the whole physical premise this
    module rests on (spec section 2)."""
    gaps_a = sorted(raw_ordered_gaps(_synthetic_hints(87.2)))
    gaps_b = sorted(raw_ordered_gaps(_synthetic_hints(310.9)))  # rotated by ~223.7deg, crosses the 0/360 wrap
    for a, b in zip(gaps_a, gaps_b):
        assert a == pytest.approx(b, abs=1e-6)


# --- update_ring_geometry / learning -------------------------------------


def test_update_ring_geometry_seeds_fresh_on_first_event():
    hints = _synthetic_hints(87.2)
    geometry, drift = update_ring_geometry(None, hints, "2026-08-20T00:00:00Z")
    assert drift is None
    assert geometry.n_events == 1
    assert geometry.n_cameras == 3
    assert sum(geometry.gaps_deg) == pytest.approx(360.0, abs=1e-6)


def test_update_ring_geometry_converges_across_noisy_events():
    """21 noisy events (matching the real corpus's own measured spread,
    sub-2deg) should converge the learned geometry close to the true
    underlying gaps."""
    import random
    rng = random.Random(20260829)
    events = []
    for i in range(21):
        noisy_gaps = (
            REAL_GAP_A + rng.uniform(-1.5, 1.5),
            REAL_GAP_B + rng.uniform(-1.5, 1.5),
        )
        cam0 = rng.uniform(0, 360)
        events.append(_synthetic_hints(cam0, gap_order=(noisy_gaps[0], noisy_gaps[1], 360 - sum(noisy_gaps))))
    geometry = _learn(events)
    assert geometry.n_events == 21
    true_gaps = sorted([REAL_GAP_A, REAL_GAP_B, REAL_GAP_C])
    learned_gaps = sorted(geometry.gaps_deg)
    for t, l in zip(true_gaps, learned_gaps):
        assert abs(t - l) < 1.0  # converges well inside the real per-event spread


def test_update_ring_geometry_survives_cross_era_reflection():
    """The real, measured phenomenon this module's own docstring
    documents: some real events' raw gap sequence only aligns to
    canonical geometry in reflection. Learning must still converge
    (not diverge into garbage) when a later batch of events needs
    reflection to align."""
    era1 = [_synthetic_hints(cam0, gap_order=(REAL_GAP_A, REAL_GAP_B, REAL_GAP_C)) for cam0 in (10, 50, 130, 220)]
    # era2: same physical gaps, but the raw reading direction is reversed
    era2 = [
        _synthetic_hints(cam0, gap_order=(REAL_GAP_C, REAL_GAP_B, REAL_GAP_A))
        for cam0 in (300, 15, 190)
    ]
    geometry = _learn(era1 + era2)
    learned = sorted(geometry.gaps_deg)
    true_gaps = sorted([REAL_GAP_A, REAL_GAP_B, REAL_GAP_C])
    for t, l in zip(true_gaps, learned):
        assert abs(t - l) < 0.5


def test_update_ring_geometry_wrong_camera_count_is_a_noop():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    unchanged, drift = update_ring_geometry(geometry, {0: 1.0, 1: 2.0}, "t1")
    assert drift is None
    assert unchanged is geometry


def test_update_ring_geometry_max_samples_window_is_capped():
    events = [_synthetic_hints(cam0) for cam0 in range(0, 360, 5)]  # 72 events
    assert len(events) > RING_GEOMETRY_MAX_SAMPLES
    geometry = _learn(events)
    assert len(geometry.recent_samples) == RING_GEOMETRY_MAX_SAMPLES
    assert geometry.n_events == len(events)  # lifetime count, never capped


# --- drift detection (spec R3) -------------------------------------------


def test_max_gap_deviation_deg_zero_for_identical_event():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    dev = max_gap_deviation_deg(_synthetic_hints(200.0), geometry)  # same gaps, different rotation
    assert dev == pytest.approx(0.0, abs=1e-6)


def test_max_gap_deviation_deg_detects_a_real_gap_change():
    """A camera that's physically moved on the ring changes the gap
    STRUCTURE, not just the absolute rotation -- this must be
    detectable via a large max_gap_deviation_deg."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    moved_hints = _synthetic_hints(87.2, gap_order=(REAL_GAP_A + 20.0, REAL_GAP_B - 10.0, REAL_GAP_C - 10.0))
    dev = max_gap_deviation_deg(moved_hints, geometry)
    assert dev > RING_GEOMETRY_DRIFT_THRESHOLD_DEG


def test_update_ring_geometry_refuses_to_absorb_a_real_drift():
    """A genuine gap-structure change must NOT get silently blended into
    the running average -- update_ring_geometry() returns the geometry
    UNCHANGED and reports the real drift, leaving the refuse decision to
    the caller (spec R3)."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    moved_hints = _synthetic_hints(87.2, gap_order=(REAL_GAP_A + 20.0, REAL_GAP_B - 10.0, REAL_GAP_C - 10.0))
    unchanged, drift = update_ring_geometry(geometry, moved_hints, "t1")
    assert drift is not None and drift > RING_GEOMETRY_DRIFT_THRESHOLD_DEG
    assert unchanged.gaps_deg == geometry.gaps_deg
    assert unchanged.n_events == geometry.n_events  # not incremented -- rejected, not learned


def test_drift_threshold_has_real_margin_over_measured_same_rig_noise():
    """Real leave-one-out measurement (this task, 21 real events):
    worst same-rig
    drift observed was 1.248deg. The shipped threshold must have real
    margin above that, and real margin below the 9deg half-sector
    identification boundary (an actual gap-identity confusion should
    never look like ordinary noise)."""
    MEASURED_WORST_SAME_RIG_DRIFT_DEG = 1.248
    assert RING_GEOMETRY_DRIFT_THRESHOLD_DEG > 2.0 * MEASURED_WORST_SAME_RIG_DRIFT_DEG
    assert RING_GEOMETRY_DRIFT_THRESHOLD_DEG < HALF_SECTOR_DEG


# --- candidate_predictions / predict_missing_hint (spec R2.2 steps 1/3) --


def test_candidate_predictions_finds_the_true_answer_among_candidates():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(87.2)
    known = {0: hints[0], 1: hints[1]}
    cands = candidate_predictions(known, geometry, 2)
    assert cands  # at least one candidate
    best_err = min(_circ_diff(pred, hints[2]) for _err, pred in cands)
    assert best_err < 1.0  # the true answer IS among the returned candidates


def test_predict_missing_hint_uses_tiebreak_to_resolve_real_ambiguity():
    """The real, provable structural ambiguity this module's own top
    docstring documents: with exactly 2 known cameras (the common real
    case), there can be a genuine tie between two very different
    predictions (spec: a bad tiebreak choice is 40deg+ off, not a few
    degrees of noise) -- the sub-floor tiebreak must resolve it
    correctly when available."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(200.0)  # different absolute rotation than the geometry was learned from
    known = {0: hints[0], 2: hints[2]}
    pred = predict_missing_hint(known, geometry, 1, tiebreak_hint=hints[1])
    assert _circ_diff(pred, hints[1]) < 1.0


def test_predict_missing_hint_leave_one_out_real_corpus_shape():
    """Reproduces this task's own real leave-one-out validation
    methodology (not the literal real corpus, to keep this test fast/
    offline, but the same synthetic-with-real-measured-noise shape) --
    real number embedded: mean error should land well under the 9deg
    half-sector margin, matching the real corpus's own 0.24deg mean."""
    import random
    rng = random.Random(1)
    events = []
    for _ in range(21):
        cam0 = rng.uniform(0, 360)
        noisy = (REAL_GAP_A + rng.uniform(-1.5, 1.5), REAL_GAP_B + rng.uniform(-1.5, 1.5))
        events.append(_synthetic_hints(cam0, gap_order=(noisy[0], noisy[1], 360 - sum(noisy))))

    errors = []
    for i, hints in enumerate(events):
        others = events[:i] + events[i + 1:]
        geometry = _learn(others)
        for missing_cam in (0, 1, 2):
            known = {c: v for c, v in hints.items() if c != missing_cam}
            pred = predict_missing_hint(known, geometry, missing_cam, tiebreak_hint=hints[missing_cam])
            errors.append(_circ_diff(pred, hints[missing_cam]))
    mean_err = sum(errors) / len(errors)
    assert mean_err < 1.0  # real corpus measured 0.24deg mean -- generous margin for synthetic noise


# --- cross_check_disagreement_deg / D2 (spec R2.2 step 2) ----------------


def test_cross_check_accepts_a_genuinely_correct_camera():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    d = cross_check_disagreement_deg(2, hints[2], {0: hints[0], 1: hints[1]}, geometry)
    assert d is not None and d < 1.0


def test_cross_check_rejects_a_real_alias_shift():
    """D2, the actual finding this whole spec exists to close: an alias
    at +162deg (spec section 1's own measured runner-up alias margin)
    clears any confidence floor with a normal reprojection error but
    disagrees wildly with consensus -- must be rejected, not silently
    trusted."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    aliased = (hints[2] + 162.0) % 360.0
    d = cross_check_disagreement_deg(2, aliased, {0: hints[0], 1: hints[1]}, geometry)
    assert d is not None and d > HALF_SECTOR_DEG


def test_cross_check_none_when_not_enough_other_evidence():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    d = cross_check_disagreement_deg(2, 165.0, {0: 87.2}, geometry)  # only 1 of 2 needed others
    assert d is None


# --- resolve_rig_consensus_orientation (the full R2.2 orchestration) -----


def test_resolve_all_three_confident_and_agreeing_stays_live():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    candidates = {c: CameraOrientationCandidate(hint_deg=h, confidence=2.5) for c, h in hints.items()}
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    assert result.ok
    for c, h in hints.items():
        pred, source = result.resolved[c]
        assert source == "live"
        assert pred == h


def test_resolve_fills_in_a_sub_floor_camera_via_consensus():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    candidates = {
        0: CameraOrientationCandidate(hint_deg=hints[0], confidence=2.2),
        1: CameraOrientationCandidate(hint_deg=hints[1], confidence=2.0),
        2: CameraOrientationCandidate(hint_deg=hints[2], confidence=1.55),  # real sub-floor, matches docs/DESIGN.md cam2 range
    }
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    assert result.ok
    pred, source = result.resolved[2]
    assert source == "rig_consensus"
    assert _circ_diff(pred, hints[2]) < 1.0
    assert 2 in result.rejected_cameras


def test_resolve_D2_regression_rejects_confident_alias_and_refuses_ambiguous_refill():
    """The literal spec done-when #4 regression, UPDATED 2026-08-31 per
    the spec section 9 post-ship fix: a camera fed a deliberately
    alias-shifted (+162deg) hint that CLEARS the confidence floor is
    still rejected by D2 cross-check and NEVER accepted as 'live' -- but
    for a 3-camera rig with only 2 anchors, cam2's own candidate was the
    ONLY signal available to fill it back in, and that signal is exactly
    the one D2 just proved untrustworthy. `candidate_predictions()` is
    structurally always tied for N=3 with 2 known + 1 missing (see this
    module's own top docstring) -- with no trustworthy tie-break left,
    the correct behaviour is now REFUSAL, not a silent guess. This test
    used to assert a successful (but not provably correct) refill before
    the post-ship fix landed -- it now asserts the corrected, safer
    outcome directly, matching spec section 9's own recommended
    direction ("refuse rather than pick")."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    aliased = (hints[2] + 162.0) % 360.0
    candidates = {
        0: CameraOrientationCandidate(hint_deg=hints[0], confidence=2.2),
        1: CameraOrientationCandidate(hint_deg=hints[1], confidence=2.0),
        2: CameraOrientationCandidate(hint_deg=aliased, confidence=2.5),  # confidently WRONG
    }
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    # The alias must still be rejected by D2 -- confirmed via rejected_cameras
    # -- but with no other signal to fill cam2 back in, this must now refuse
    # rather than silently accept a guess (NEVER "live", and per the fix,
    # not a silently-guessed "rig_consensus" either).
    assert not result.ok
    assert 2 in result.rejected_cameras
    assert "D2 cross-check" in result.rejected_cameras[2]
    assert 2 not in result.resolved  # never filled in with an unverifiable guess
    assert "mirror ambiguity" in result.refusal_reason.lower() or "tie" in result.refusal_reason.lower()


def test_resolve_refuses_with_fewer_than_two_anchors():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    candidates = {
        0: CameraOrientationCandidate(hint_deg=hints[0], confidence=2.2),
        1: CameraOrientationCandidate(hint_deg=hints[1], confidence=1.0),  # sub-floor
        2: CameraOrientationCandidate(hint_deg=None, confidence=0.0),  # no candidate at all
    }
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    assert not result.ok
    assert "fewer than" in result.refusal_reason.lower() or "2" in result.refusal_reason


def test_resolve_refuses_when_camera_count_mismatches_geometry():
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")  # 3 cameras
    candidates = {0: CameraOrientationCandidate(hint_deg=1.0, confidence=2.5)}
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    assert not result.ok


def test_resolve_two_anchors_disagreeing_with_geometry_refuses():
    """Two confident cameras whose own mutual offset doesn't match ANY
    stored gap combination -- a genuine "this doesn't fit the learned
    ring at all" case, not resolvable by trusting either blindly."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    candidates = {
        0: CameraOrientationCandidate(hint_deg=0.0, confidence=2.2),
        1: CameraOrientationCandidate(hint_deg=45.0, confidence=2.0),  # nowhere near any real gap
        2: CameraOrientationCandidate(hint_deg=None, confidence=0.0),
    }
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    assert not result.ok


def test_resolve_sub_floor_own_value_never_used_as_tiebreak_after_rejection():
    """Real bug this task's own validation caught and fixed: using a
    D2-rejected anchor's own
    (proven wrong) value as the fill-in tie-break defeats the whole
    point of rejecting it. With NO separate tiebreak signal available,
    the fill-in must not silently re-derive one from the rejected value
    itself.

    UPDATED 2026-08-31 per spec section 9's post-ship fix: this used to
    assert the resolver still produced SOME answer (just not one close
    to the rejected alias). That was itself a real, latent instance of
    the exact mirror-ambiguity bug the post-ship fix closes -- with the
    rejected value excluded and nothing else to break the tie, the
    correct behaviour is refusal, not "any answer other than the
    alias." See test_resolve_D2_regression_rejects_confident_alias_and_
    refuses_ambiguous_refill for the direct regression on this same
    scenario -- kept here as a second, differently-worded assertion of
    the same real fix for redundancy/clarity of intent."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    aliased = (hints[2] + 162.0) % 360.0
    candidates = {
        0: CameraOrientationCandidate(hint_deg=hints[0], confidence=2.2),
        1: CameraOrientationCandidate(hint_deg=hints[1], confidence=2.0),
        2: CameraOrientationCandidate(hint_deg=aliased, confidence=2.5),
    }
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    # The D2-rejected value must never be used as its own tie-break --
    # with nothing else to distinguish the tied candidates, the correct
    # outcome is now a refusal, not a guess (of any kind, correct or not).
    assert not result.ok
    assert 2 not in result.resolved


# --- post-ship mirror-ambiguity tie-break fix, 2026-08-31 --
#
# candidate_predictions() can return two candidates with IDENTICAL
# error -- a real, structural mirror ambiguity for a 3-camera ring (the
# gap sequence read forward vs backward both fit equally well) -- and
# nothing in the evidence used to separate them. Two real, distinct
# bugs: (1) the tie was resolved by enumeration order, not evidence,
# when no trustworthy tiebreak existed; (2) a D2-rejected candidate
# could still be used AS that tiebreak, if the camera had never been an
# anchor to begin with (min_confidence never cleared).


def test_predict_missing_hint_refuses_on_genuine_tie_with_no_tiebreak():
    """Direct unit-level regression: a genuine tie (multiple candidates
    within tolerance) with no trustworthy tiebreak_hint must return
    None (refuse), never silently pick the first-enumerated candidate.
    Real numbers reproduced from spec section 9's own worked example --
    a blended pre-/post-board-rotation sample window, gaps
    [107.77, 148.94, 103.29] -- the spec's own quoted `err 0.76` pair
    (123.84deg / 165.01deg predictions) is reproduced here almost
    exactly (this test's own geometry/known-hints combination isn't
    byte-identical to the spec's, so the exact predicted values differ
    slightly, but the SAME real tie shape is confirmed: two candidates,
    essentially equal match error, 40+deg apart)."""
    geometry = RingGeometry(
        gaps_deg=[107.77, 148.94, 103.29],
        n_events=20,
        first_learned_utc="2026-08-01T00:00:00Z",
        last_updated_utc="2026-08-29T00:00:00Z",
        spread_deg=[1.0, 1.0, 1.0],
        recent_samples=[],
    )
    known = {0: 272.95, 1: 15.58}  # spec's own real ground-truth cam0/cam1
    cands = candidate_predictions(known, geometry, 2)
    assert len(cands) == 2  # the real tie: exactly two candidates
    assert abs(cands[0][0] - cands[1][0]) < 0.01  # tied within essentially the same match error
    assert _circ_diff(cands[0][1], cands[1][1]) > 30.0  # genuinely far apart, not noise
    pred = predict_missing_hint(known, geometry, 2, tiebreak_hint=None)
    assert pred is None  # refuse -- NOT the old silent enumeration-order guess


def test_resolve_refuses_on_blended_era_geometry_zero_candidate_real_spec_reproduction():
    """The literal real incident from spec section 9: blended-era
    geometry (gaps [107.77, 148.94, 103.29], real ground truth
    cam0=272.95/cam1=15.58/cam2=165.57) + a camera with zero candidate
    this event used to resolve to `ok=True, source="rig_consensus"`, a
    value 41.7deg (2.3 wedges) off truth, decided purely by enumeration
    order. Must now refuse rather than produce any answer at all."""
    geometry = RingGeometry(
        gaps_deg=[107.77, 148.94, 103.29],
        n_events=20,
        first_learned_utc="2026-08-01T00:00:00Z",
        last_updated_utc="2026-08-29T00:00:00Z",
        spread_deg=[1.0, 1.0, 1.0],
        recent_samples=[],
    )
    candidates = {
        0: CameraOrientationCandidate(hint_deg=272.95, confidence=2.2),
        1: CameraOrientationCandidate(hint_deg=15.58, confidence=2.0),
        2: CameraOrientationCandidate(hint_deg=None, confidence=0.0),
    }
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    assert not result.ok
    assert 2 not in result.resolved
    assert result.refusal_reason is not None
    assert "cam2" in result.refusal_reason


def test_resolve_d2_rejected_sub_floor_never_anchor_candidate_excluded_from_tiebreak():
    """The second real bug from spec section 9: a camera whose own
    candidate is BOTH sub-floor (never an anchor, confidence below
    min_confidence) AND fails D2 cross-check used to still have its own
    (proven-wrong) value used as the fill-in tie-break, because the
    earlier `d2_rejected_own_value` bookkeeping only tracked cameras
    that had actually been anchors (`if cam in anchors:`). Confirmed
    real, reachable, and now fixed: a sub-floor aliased (+162deg)
    candidate that D2 correctly flags as ~120deg off consensus must no
    longer be usable to pick between two tied fill-in candidates. For
    this 3-camera geometry no OTHER signal exists for cam2 once its own
    value is (correctly) excluded, so the safe, correct outcome is
    refusal -- not the old silently-guessed answer (previously measured
    at 41.1deg off truth on this exact scenario) this bug used to
    produce."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    aliased = (hints[2] + 162.0) % 360.0
    candidates = {
        0: CameraOrientationCandidate(hint_deg=hints[0], confidence=2.2),
        1: CameraOrientationCandidate(hint_deg=hints[1], confidence=2.0),
        2: CameraOrientationCandidate(hint_deg=aliased, confidence=1.0),  # SUB-FLOOR, never an anchor
    }
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    assert not result.ok
    assert 2 in result.rejected_cameras
    assert "D2 cross-check" in result.rejected_cameras[2]
    assert 2 not in result.resolved


def test_resolve_sub_floor_never_anchor_candidate_that_passes_d2_still_used_as_tiebreak():
    """Confirms the `d2_rejected_own_value` fix is correctly SCOPED --
    it must exclude only cameras D2 has actively disproven, not every
    never-anchor (sub-floor) camera. A sub-floor camera whose own
    candidate is genuinely correct and PASSES the D2 cross-check must
    still be usable as its own fill-in tie-break, exactly as spec R2.2
    step 3's own wording intends. This is the real, already-shipped,
    working case that must not regress from either fix above."""
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    hints = _synthetic_hints(310.0)
    candidates = {
        0: CameraOrientationCandidate(hint_deg=hints[0], confidence=2.2),
        1: CameraOrientationCandidate(hint_deg=hints[1], confidence=2.0),
        2: CameraOrientationCandidate(hint_deg=hints[2], confidence=1.0),  # sub-floor, genuinely correct
    }
    result = resolve_rig_consensus_orientation(candidates, geometry, min_confidence=1.8)
    assert result.ok
    pred, source = result.resolved[2]
    assert source == "rig_consensus"
    assert _circ_diff(pred, hints[2]) < 1.0


# --- persistence -----------------------------------------------------------


def test_load_ring_geometry_none_root_returns_none():
    assert load_ring_geometry(None) is None


def test_load_ring_geometry_missing_file_returns_none(tmp_path):
    assert load_ring_geometry(tmp_path) is None


def test_save_and_load_round_trip(tmp_path):
    geometry, _ = update_ring_geometry(None, _synthetic_hints(87.2), "t0")
    geometry, _ = update_ring_geometry(geometry, _synthetic_hints(200.0), "t1")
    save_ring_geometry(tmp_path, geometry)
    reloaded = load_ring_geometry(tmp_path)
    assert reloaded is not None
    assert reloaded.gaps_deg == geometry.gaps_deg
    assert reloaded.n_events == geometry.n_events
    assert reloaded.recent_samples == geometry.recent_samples


def test_load_ring_geometry_corrupt_json_degrades_safely(tmp_path):
    (tmp_path / "ring_geometry_fallback.json").write_text("{not valid json")
    assert load_ring_geometry(tmp_path) is None


def test_load_ring_geometry_wrong_schema_degrades_safely(tmp_path):
    path = tmp_path / "ring_geometry_fallback.json"
    path.write_text(json.dumps({"schema": "ring-geometry-v0"}))
    assert load_ring_geometry(tmp_path) is None


def test_load_ring_geometry_missing_fields_degrades_safely(tmp_path):
    path = tmp_path / "ring_geometry_fallback.json"
    path.write_text(json.dumps({"schema": "ring-geometry-v1", "gaps_deg": [1.0, 2.0]}))
    assert load_ring_geometry(tmp_path) is None
