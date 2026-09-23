"""Tests for `opendarts.live.capture_daemon._min_good_frames_cumulative()`
(DEFECT 2, 2026-09-03 -- see docs/DESIGN.md's dated entry for the full real
incident and the "arithmetically self-defeating" root cause this fix
closes).

`_min_calibration_frames_required(n_frames)` returns a STRICT MAJORITY of
`n_frames` -- correct for `_try_solve_from_detections()`'s own genuinely
fixed-size single BEST-OF-N batch, but wrong when reused (as it used to be)
against `_try_solve()`'s own ever-growing per-camera `frames_captured[cam]`
pool: the requirement grows every retry round while a low-yield camera's
achievable count grows much more slowly, so it can never catch up no
matter how many more frames it captures. `_min_good_frames_cumulative()`
replaces that reuse with an ABSOLUTE floor (5) that never grows with the
pool.

Two tiers, matching this project's established fast/slow split
(`tests/conftest.py`):

  1. Fast, fully synthetic (default `pytest tests/` tier) -- the pure
     `_min_good_frames_cumulative()` function in isolation, and a
     confirmation that `_min_calibration_frames_required()` itself (and
     its own existing test) are untouched.
  2. Real, unmocked end-to-end passes against the two real local
     calibration packages this defect was found from -- they need the
     calibration corpus, which is not in the repo, and the replay
     harness that reads it, so they live in
     `dev/tests/test_calibration_min_good_frames_20260903.py` rather
     than here. They prove cam2 (which never solved under the old rule
     on either real event) recovers under today's fixed code within the
     real `max_frames=200` production budget, and that a healthy package
     (every camera solving on round 1) is byte-identical before and
     after this fix.
"""
from __future__ import annotations

import opendarts.live.capture_daemon as capture_daemon

# ---------------------------------------------------------------------------
# Fast tier -- the pure function, in complete isolation.
# ---------------------------------------------------------------------------


def test_min_good_frames_cumulative_is_an_absolute_floor_not_a_fraction():
    # Never grows past CALIBRATION_MIN_GOOD_FRAMES_CUMULATIVE (5) no
    # matter how large the accumulated total gets -- this is the whole
    # point of the fix (the old strict-majority rule DID grow, which is
    # what made it arithmetically unwinnable for a low-yield camera).
    assert capture_daemon._min_good_frames_cumulative(1) == 1
    assert capture_daemon._min_good_frames_cumulative(2) == 2
    assert capture_daemon._min_good_frames_cumulative(3) == 3
    assert capture_daemon._min_good_frames_cumulative(4) == 4
    assert capture_daemon._min_good_frames_cumulative(5) == 5
    assert capture_daemon._min_good_frames_cumulative(6) == 5
    assert capture_daemon._min_good_frames_cumulative(10) == 5
    assert capture_daemon._min_good_frames_cumulative(50) == 5
    assert capture_daemon._min_good_frames_cumulative(205) == 5


def test_min_good_frames_cumulative_capped_at_the_total_when_total_is_tiny():
    # A genuinely tiny total (e.g. a test double using n_frames_detect < 5)
    # degrades to "need literally all of them," never an unreachable
    # count above the total itself.
    assert capture_daemon._min_good_frames_cumulative(0) == 1
    assert capture_daemon._min_good_frames_cumulative(1) == 1


def test_old_strict_majority_formula_would_have_been_arithmetically_unwinnable():
    """Direct, real-number demonstration of the defect this fix closes --
    the old `_min_calibration_frames_required()` reused against a growing
    total requires MORE good frames than a real event (cam2's own
    documented ~24% real yield, see docs/DESIGN.md's DEFECT 2 entry) could
    ever produce, at every single total from round 1 through the full
    200-frame budget. The new floor is satisfiable at every one of the
    same totals once yield * total >= 5."""
    old_formula = capture_daemon._min_calibration_frames_required
    new_formula = capture_daemon._min_good_frames_cumulative
    real_yield = 49 / 205  # cam2's own real measured yield on this exact incident

    for total in (5, 30, 55, 80, 105, 130, 155, 180, 205):
        achievable = int(total * real_yield)
        assert achievable < old_formula(total), (
            "the old formula should be unreachable at every total on this real yield"
        )
    # The new floor is reachable once the accumulated total is large
    # enough for the real yield to produce >= 5 good frames -- exactly
    # what let cam2 recover under the fix (see the real replay tests
    # below).
    assert int(205 * real_yield) >= new_formula(205)


def test_min_calibration_frames_required_untouched_by_this_fix():
    """`_min_calibration_frames_required()` itself is unchanged --
    verbatim re-assertion of its own existing test
    (`tests/test_capture_daemon.py::test_min_calibration_frames_required_
    is_a_strict_majority_above_one`), kept here too as an explicit,
    load-bearing regression guard for THIS task specifically: this fix
    must redirect the 4 `_try_solve()`/`_resolve_*` call sites without
    touching this function or its one remaining real caller
    (`_try_solve_from_detections()`, BEST-OF-N's own genuinely
    fixed-size single batch)."""
    assert capture_daemon._min_calibration_frames_required(1) == 1
    assert capture_daemon._min_calibration_frames_required(2) == 2
    assert capture_daemon._min_calibration_frames_required(3) == 2
    assert capture_daemon._min_calibration_frames_required(5) == 3
    assert capture_daemon._min_calibration_frames_required(8) == 5
    assert capture_daemon._min_calibration_frames_required(10) == 6
