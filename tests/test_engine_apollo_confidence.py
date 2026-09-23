"""Tests for opendarts.engines.apollo.confidence -- see that module's own
docstring for the full derivation/validation writeup (signals used, how
the logistic-regression coefficients were fit, in-sample and
leave-one-session-out reliability tables).

Two kinds of test here, deliberately separate:

1. **Unit tests on `compute_confidence()` directly** -- no real data
   needed, pin the function's own documented contract (ok=False -> 0.0,
   monotonicity in each signal, a hand-checked numeric value) so a
   future refactor can't silently change its math.
2. **Real-corpus calibration test** -- the actual point of a CALIBRATED
   confidence score, by design's own framing ("iterate until confidence
   matches accuracy"): replay the real `data/archive/clean/` corpus
   through the CURRENT `ApolloEngine` (per this project's REPLAY first
   principle -- never read stale stored `result.json`), bucket by
   predicted confidence, and assert the count-weighted mean gap between
   predicted confidence and observed accuracy (Expected Calibration
   Error) stays under a REAL MEASURED threshold with real margin, not a
   round-number guess (opendarts/engines/apollo/confidence.py's own
   docstring: measured 0.035 in-sample; this test's threshold is set
   above that with headroom for normal pipeline drift, not equal to it
   -- a test that fails on every trivial future change is as useless as
   one that can never fail). Skips cleanly (not a failure) when no real
   corpus is present, the same convention every real-corpus test here
   uses.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from opendarts.engines.apollo.confidence import (
    DISAGREE_CAP_MM,
    WIRE_CAP_MM,
    _dist_to_nearest_wire_mm,
    compute_confidence,
)
from opendarts.engines.registry import get_engine

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Unit tests -------------------------------------------------------


def test_no_score_is_zero_confidence():
    assert compute_confidence(
        ok=False, board_xy_mm=None, max_ray_disagreement_mm=None, fallback_used=False
    ) == 0.0


def test_confidence_is_a_probability():
    conf = compute_confidence(
        ok=True, board_xy_mm=(50.0, 0.0), max_ray_disagreement_mm=2.0, fallback_used=False
    )
    assert 0.0 <= conf <= 1.0


def test_more_ray_disagreement_means_lower_confidence():
    """Holding position and fallback fixed, more disagreement must never
    increase confidence -- this is the whole point of using it as a risk
    signal (see module docstring signal 1)."""
    low = compute_confidence(
        ok=True, board_xy_mm=(50.0, 0.0), max_ray_disagreement_mm=0.5, fallback_used=False
    )
    high = compute_confidence(
        ok=True, board_xy_mm=(50.0, 0.0), max_ray_disagreement_mm=8.0, fallback_used=False
    )
    assert high < low


def test_closer_to_a_wire_means_lower_confidence():
    """Holding disagreement and fallback fixed, a point closer to a
    ring/sector boundary must never have higher confidence than one
    sitting safely inside a region -- see module docstring signal 2 (the
    strongest real signal found: every throw >=10mm from any wire in the
    real corpus was correct)."""
    # r=97.5 sits exactly on the treble-inner scoring wire (near-zero
    # distance); r=50 sits deep inside single_inner (far from every wire).
    on_wire = compute_confidence(
        ok=True, board_xy_mm=(0.0, 97.5), max_ray_disagreement_mm=2.0, fallback_used=False
    )
    safe = compute_confidence(
        ok=True, board_xy_mm=(0.0, 50.0), max_ray_disagreement_mm=2.0, fallback_used=False
    )
    assert on_wire < safe


def test_fallback_used_means_lower_confidence():
    """Holding position/disagreement fixed, using the 2-of-3 RANSAC
    fallback pair must never score higher confidence than trusting the
    full 3-camera set -- see module docstring signal 3 (measured 77.8%
    vs 97.4% correct on the real corpus)."""
    full_set = compute_confidence(
        ok=True, board_xy_mm=(50.0, 0.0), max_ray_disagreement_mm=2.0, fallback_used=False
    )
    fallback = compute_confidence(
        ok=True, board_xy_mm=(50.0, 0.0), max_ray_disagreement_mm=2.0, fallback_used=True
    )
    assert fallback < full_set


def test_confidence_matches_hand_computed_sigmoid():
    """Pins the exact math, not just the direction -- a future change to
    the formula (not just the coefficients) should have to update this
    deliberately, not accidentally."""
    from opendarts.engines.apollo import confidence as conf_mod

    board_xy = (0.0, 50.0) # deep inside single_inner -- far from any wire
    disagreement = 2.0
    wire_dist = _dist_to_nearest_wire_mm(*board_xy)
    wire_safe = min(wire_dist, WIRE_CAP_MM) / WIRE_CAP_MM
    disagree_risk = min(disagreement, DISAGREE_CAP_MM) / DISAGREE_CAP_MM
    z = (
        conf_mod.INTERCEPT
        + conf_mod.W_WIRE_SAFE * wire_safe
        + conf_mod.W_DISAGREE_RISK * disagree_risk
        + conf_mod.W_FALLBACK_RISK * 0.0
    )
    expected = 1.0 / (1.0 + math.exp(-z))

    actual = compute_confidence(
        ok=True, board_xy_mm=board_xy, max_ray_disagreement_mm=disagreement, fallback_used=False
    )
    assert actual == pytest.approx(expected, abs=1e-12)


def test_dist_to_nearest_wire_is_zero_on_a_ring_boundary():
    # (0, 97.5) sits exactly on the treble-inner SCORING radius (see
    # opendarts.geometry.board.TREBLE_INNER_SCORING_RADIUS_MM = 99.0 - 1.5).
    assert _dist_to_nearest_wire_mm(0.0, 97.5) == pytest.approx(0.0, abs=1e-9)


def test_dist_to_nearest_wire_is_positive_deep_inside_a_region():
    # (0, 150) is dead-center of sector 20's wedge (angle=0deg, 9deg --
    # the maximum possible -- from either neighboring radial wire) AND
    # sits at r=150mm, comfortably between the treble-outer (107mm) and
    # double-inner-scoring (160.5mm) ring boundaries. Note r=50 (used in
    # the other tests above for its simple round number) is NOT a good
    # "far from everything" example: even dead-center of a wedge, sector
    # wires are only 9deg apart, so their LINEAR distance shrinks toward
    # the bull (arc = r*sin(9deg) is just 7.85mm at r=50) -- real
    # geometry, not a bug in `_dist_to_nearest_wire_mm`.
    assert _dist_to_nearest_wire_mm(0.0, 150.0) > 10.0


# --- Real-corpus calibration test --------------------------------------


def _find_corpus_throw_dirs() -> list[Path]:
    """The shared corpus-discovery convention:
    OPENDARTS_ENGINE_CORPUS_ROOT env var, else this repo's own data/archive/
    (gitignored, empty in a fresh clone/isolated worktree)."""
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    roots = [Path(env_root)] if env_root else [REPO_ROOT / "data" / "archive"]
    dirs: list[Path] = []
    # os.walk(followlinks=True), NOT Path.rglob() -- an isolated agent
    # worktree symlinks data/archive/<session>/ back to the main
    # checkout's real files (avoids duplicating gigabytes of images per
    # worktree) rather than copying them, and Python 3.9's pathlib.rglob()
    # does not follow symlinks (fixed only in 3.13's recurse_symlinks=) --
    # this silently found zero packages and SKIPPED inside a worktree,
    # never verifying anything, even though the real corpus was genuinely
    # right there and readable via a plain ls/cd. Found 2026-08-14
    # in review -- same bug,
    # same fix, as the parallel Athena confidence work.
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
            if "ad_ground_truth.json" in filenames:
                pkg_dir = Path(dirpath)
                if (pkg_dir / "calibration.json").exists() and (pkg_dir / "meta.json").exists():
                    dirs.append(pkg_dir)
    return sorted(dirs)


@pytest.fixture(scope="module")
def corpus_throw_dirs() -> list[Path]:
    dirs = _find_corpus_throw_dirs()
    if not dirs:
        pytest.skip(
            "no real archived throw packages with ad_ground_truth.json found "
            "under data/archive/ (or $OPENDARTS_ENGINE_CORPUS_ROOT) -- this "
            "calibration proof only means something against real AD-matched "
            "throws"
        )
    return dirs


# Real measured in-sample ECE (opendarts/engines/apollo/confidence.py's
# own docstring): 0.035. This threshold is set well above that with real
# margin -- it exists to catch a genuine calibration regression (a future
# change to the coefficients, the signals, or the formula that breaks the
# fit), not to pin the exact current number, which would make this test
# fail on any benign pipeline drift (a slightly different corpus size, a
# tip_detection re-tune that shifts a handful of near-wire throws) with
# no real loss of calibration.
MAX_ACCEPTABLE_ECE = 0.10


@pytest.mark.slow # full-corpus replay, ~41s measured 2026-08-16 -- see tests/conftest.py
def test_confidence_is_calibrated_against_real_ad_ground_truth(corpus_throw_dirs):
    """The actual point of this whole module, by design's own framing:
    bucket real throws by PREDICTED confidence and check the OBSERVED
    accuracy in each bucket is close -- Expected Calibration Error,
    count-weighted mean |predicted - observed| across buckets, must stay
    under MAX_ACCEPTABLE_ECE. Replays every real AD-matched throw through
    the CURRENT ApolloEngine (never reads stale stored result.json --
    the "Replay is the source of truth" constraint)."""
    from opendarts.capture.throw_package import load_throw_package

    engine = get_engine("Apollo")
    confidences: list[float] = []
    corrects: list[bool] = []

    for pkg_dir in corpus_throw_dirs:
        ad = json.loads((pkg_dir / "ad_ground_truth.json").read_text())
        if not ad.get("matched"):
            continue
        package = load_throw_package(pkg_dir)
        result = engine.score(package.bg_frames, package.dart_frames, package.calibrations)
        if not result.ok:
            # No-score throws are always confidence=0.0 by construction
            # (see compute_confidence's docstring) -- excluded from the
            # reliability table itself (there is no "accuracy" to check a
            # non-answer against), same convention this project's other
            # accuracy metrics use for no-score throws.
            continue
        conf = result.diagnostics.get("confidence")
        assert conf is not None, f"{pkg_dir}: ApolloEngine result missing diagnostics['confidence']"
        confidences.append(conf)
        corrects.append(result.sector == ad.get("sector") and result.ring == ad.get("ring"))

    n = len(confidences)
    assert n > 0, "no AD-matched, successfully-scored throws found in the real corpus"

    import numpy as np

    order = np.argsort(confidences)
    conf_sorted = np.array(confidences)[order]
    correct_sorted = np.array(corrects)[order]

    n_bins = min(8, n)
    bin_edges = np.linspace(0, n, n_bins + 1).astype(int)
    ece = 0.0
    table_lines = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        c = conf_sorted[lo:hi]
        y = correct_sorted[lo:hi]
        mean_conf = float(c.mean())
        mean_acc = float(y.mean())
        ece += (len(c) / n) * abs(mean_conf - mean_acc)
        table_lines.append(f" predicted={mean_conf:.3f} observed={mean_acc:.3f} n={len(c)}")

    print(f"\nReliability table ({n} real AD-matched throws, {n_bins} buckets):")
    print("\n".join(table_lines))
    print(f"Expected Calibration Error (count-weighted mean gap): {ece:.4f}")

    assert ece < MAX_ACCEPTABLE_ECE, (
        f"confidence score is not well-calibrated against real AD ground "
        f"truth: ECE={ece:.4f} >= {MAX_ACCEPTABLE_ECE} -- see the printed "
        f"reliability table above for which bucket(s) diverge"
    )
