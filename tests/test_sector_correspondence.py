"""Tests for opendarts/calibration/sector_correspondence.py -- the robust
N-frame correspondence-averaging utility (see that module's docstring;
it used to also own a per-camera orientation-constant mechanism, removed
2026-08-20. This test file was trimmed to match: only the averaging tests,
which never depended on the removed mechanism, remain.)
"""
from __future__ import annotations

import numpy as np
import pytest

from opendarts.calibration.sector_correspondence import average_correspondences


# --- average_correspondences() -- N-frame-averaging core, added 2026-08-12 -


def _obj_pts() -> np.ndarray:
    return np.array(
        [[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [-100.0, 0.0, 0.0], [0.0, -100.0, 0.0]]
    )


def _img_pts_with_jitter(base: np.ndarray, jitter: np.ndarray) -> np.ndarray:
    return base + jitter


def test_average_correspondences_closer_to_ground_truth_than_any_single_noisy_frame():
    """The core empirical claim of this whole change, proven directly at
    the unit level (not just measured against real data separately):
    given N synthetic detections with KNOWN per-frame jitter around a
    known ground-truth image_points, the averaged result's distance from
    ground truth is smaller than every individual single frame's own
    distance from ground truth."""
    rng = np.random.default_rng(20260812)
    obj = _obj_pts()
    ground_truth_img = np.array([[640.0, 200.0], [900.0, 360.0], [640.0, 520.0], [380.0, 360.0]])

    n = 6
    jitters = rng.normal(scale=8.0, size=(n, 4, 2))
    detections = [(obj, _img_pts_with_jitter(ground_truth_img, j)) for j in jitters]

    averaged = average_correspondences(detections, min_required=1)
    assert averaged is not None
    avg_obj, avg_img, n_used = averaged
    assert n_used == n
    assert np.allclose(avg_obj, obj)

    avg_dist = np.linalg.norm(avg_img - ground_truth_img)
    single_frame_dists = [
        np.linalg.norm(img - ground_truth_img) for _obj, img in detections
    ]
    assert avg_dist < min(single_frame_dists), (
        f"averaged distance {avg_dist} was not smaller than the best single "
        f"frame's own distance {min(single_frame_dists)}"
    )
    # Stronger, expected claim: averaging should usually beat MOST single
    # frames, not just the very worst one -- guards against a degenerate
    # "beats the worst by luck" pass.
    n_beaten = sum(avg_dist < d for d in single_frame_dists)
    assert n_beaten >= n - 1, (
        f"averaged result only beat {n_beaten}/{n} single frames -- expected "
        f"it to beat nearly all of them for independent zero-mean jitter"
    )


def test_average_correspondences_matches_hand_computed_mean():
    obj = _obj_pts()
    img_a = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    img_b = np.array([[12.0, 8.0], [18.0, 14.0], [22.0, 18.0], [8.0, 22.0]])
    img_c = np.array([[11.0, 9.0], [19.0, 12.0], [21.0, 19.0], [9.0, 21.0]])
    detections = [(obj, img_a), (obj, img_b), (obj, img_c)]

    result = average_correspondences(detections, min_required=1)
    assert result is not None
    avg_obj, avg_img, n_used = result
    assert n_used == 3
    expected = np.mean(np.stack([img_a, img_b, img_c]), axis=0)
    assert np.allclose(avg_img, expected)


def test_average_correspondences_skips_none_entries_and_averages_the_rest():
    """Partial-failure-among-N: None entries (a frame where detection
    itself failed) must be excluded from the average, not treated as
    zeros or otherwise corrupting it."""
    obj = _obj_pts()
    img_a = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    img_b = np.array([[30.0, 30.0], [40.0, 30.0], [40.0, 40.0], [30.0, 40.0]])
    detections = [(obj, img_a), None, (obj, img_b), None]

    result = average_correspondences(detections, min_required=1)
    assert result is not None
    _avg_obj, avg_img, n_used = result
    assert n_used == 2
    assert np.allclose(avg_img, (img_a + img_b) / 2.0)


def test_average_correspondences_returns_none_below_min_required():
    obj = _obj_pts()
    img = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    detections = [(obj, img), None, None] # only 1 success

    assert average_correspondences(detections, min_required=2) is None
    # But the same input with a lower floor succeeds:
    assert average_correspondences(detections, min_required=1) is not None


def test_average_correspondences_all_none_returns_none():
    assert average_correspondences([None, None, None], min_required=1) is None


def test_average_correspondences_empty_list_returns_none():
    assert average_correspondences([], min_required=1) is None


def test_average_correspondences_asserts_on_mismatched_object_points():
    """object_points_mm must be identical across every non-None detection
    for the same camera (see this function's own docstring for why that's
    guaranteed by construction upstream) -- a mismatch is a hard failure,
    not a silently-wrong average."""
    obj_a = _obj_pts()
    obj_b = _obj_pts() + 5.0 # deliberately different -- simulates a broken contract
    img = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    detections = [(obj_a, img), (obj_b, img)]

    with pytest.raises(AssertionError):
        average_correspondences(detections, min_required=1)


def test_average_correspondences_single_detection_is_unchanged():
    """N=1 (degenerate case) must reproduce today's old single-frame
    behavior exactly -- the average of one thing is that thing."""
    obj = _obj_pts()
    img = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    result = average_correspondences([(obj, img)], min_required=1)
    assert result is not None
    _avg_obj, avg_img, n_used = result
    assert n_used == 1
    assert np.allclose(avg_img, img)


# ---------------------------------------------------------------------
# Robust/trimmed averaging, added 2026-08-14 -- see average_correspondences()'s
# own module-level TRIM_* constants for the real measured basis
# (against data/archive/clean/).
# ---------------------------------------------------------------------


def test_average_correspondences_rejects_a_real_observed_outlier():
    """The real thing this whole change exists to catch, using the ACTUAL
    numbers found sitting in the real
    corpus: one session's camera 0 has one real frame whose
    detected landmark 0 sits 237.893px from that index's own median
    across an otherwise-tight (median 0.705px) group of 29 real
    detections. Reproduced here as a synthetic-but-realistically-shaped
    fixture (real magnitude, not a contrived huge number) so this test
    doesn't depend on data/archive/ being present on every machine.

    Plain (untrimmed) averaging lets a single outlier this large drag the
    mean by 237.893/29 ≈ 8.2px even on a 29-frame batch -- a huge error
    for a calibration pipeline whose whole point is sub-pixel/few-px
    precision. Trimmed averaging must reject it and land within a couple
    px of the tight group's own median instead.
    """
    obj = _obj_pts()
    rng = np.random.default_rng(20260814)
    n_good = 28
    base = np.array([[640.0, 200.0], [900.0, 360.0], [640.0, 520.0], [380.0, 360.0]])
    # Tight real-shaped jitter (~0.7px median deviation, matching the
    # real non-outlier group this fixture reproduces) on every point.
    good_jitter = rng.normal(scale=0.7, size=(n_good, 4, 2))
    good_imgs = base[None, :, :] + good_jitter

    # One real-magnitude outlier: landmark index 0 (only) is 237.893px
    # off from the rest -- landmarks 1-3 in that same bad frame are fine,
    # matching a plausible single-point occlusion/glare, not a
    # whole-frame detection catastrophe.
    outlier_img = base.copy()
    outlier_img[0] = outlier_img[0] + np.array([237.893, 0.0])

    detections = [(obj, img) for img in good_imgs] + [(obj, outlier_img)]

    untrimmed = average_correspondences(detections, min_required=1, trim=False)
    trimmed_diag: dict = {}
    trimmed = average_correspondences(
        detections, min_required=1, trim=True, trim_diagnostics_out=trimmed_diag
    )
    assert untrimmed is not None and trimmed is not None
    _obj_u, img_u, _n_u = untrimmed
    _obj_t, img_t, n_used = trimmed

    # n_used (how many detections contributed at all) is unchanged by
    # trimming -- trimming operates on individual points WITHIN the
    # average, not on which frames "succeeded".
    assert n_used == n_good + 1

    dist_from_truth_untrimmed = float(np.linalg.norm(img_u[0] - base[0]))
    dist_from_truth_trimmed = float(np.linalg.norm(img_t[0] - base[0]))
    # The plain mean is dragged measurably off by the single outlier
    # (real math: 237.893 / 29 ≈ 8.2px pull on index 0 alone).
    assert dist_from_truth_untrimmed > 5.0, dist_from_truth_untrimmed
    # The trimmed mean must land close to the tight group's own true
    # value -- within a small multiple of the injected good-jitter scale
    # (0.7px), nowhere near the outlier's pull.
    assert dist_from_truth_trimmed < 1.5, dist_from_truth_trimmed
    assert dist_from_truth_trimmed < dist_from_truth_untrimmed / 3.0

    # The other 3 landmark indices (never touched by the outlier) must
    # come out essentially identical whether trimmed or not -- trimming
    # must not perturb indices that had nothing to trim.
    assert np.allclose(img_t[1:], img_u[1:], atol=0.5)

    assert trimmed_diag["trimmed"] is True
    assert trimmed_diag["n_trimmed_per_index"][0] == 1 # exactly the injected outlier
    assert trimmed_diag["n_trimmed_per_index"][1:] == [0, 0, 0] # nothing else touched


def test_average_correspondences_trimming_still_averages_down_normal_noise():
    """The other half explicitly asked to be proven, not assumed:
    trimming must NOT regress the already-measured "averaging beats any
    single noisy frame" property for genuinely normal (non-outlier)
    per-frame noise. Reruns this file's own pre-existing
    test_average_correspondences_closer_to_ground_truth_than_any_single_noisy_frame
    fixture (same seed, same isotropic Gaussian jitter, std=8px/axis)
    with trim=True (now the default) and checks the identical assertions
    that test already makes with trim implicitly on."""
    rng = np.random.default_rng(20260812)
    obj = _obj_pts()
    ground_truth_img = np.array([[640.0, 200.0], [900.0, 360.0], [640.0, 520.0], [380.0, 360.0]])

    n = 6
    jitters = rng.normal(scale=8.0, size=(n, 4, 2))
    detections = [(obj, ground_truth_img + j) for j in jitters]

    diag: dict = {}
    averaged = average_correspondences(detections, min_required=1, trim=True, trim_diagnostics_out=diag)
    assert averaged is not None
    _avg_obj, avg_img, n_used = averaged
    assert n_used == n

    avg_dist = np.linalg.norm(avg_img - ground_truth_img)
    single_frame_dists = [np.linalg.norm(img - ground_truth_img) for _obj, img in detections]
    assert avg_dist < min(single_frame_dists), (
        f"trimmed-averaged distance {avg_dist} was not smaller than the best single "
        f"frame's own distance {min(single_frame_dists)}"
    )
    n_beaten = sum(avg_dist < d for d in single_frame_dists)
    assert n_beaten >= n - 1

    # Real normal Gaussian tail (nothing genuinely wrong with any of
    # these 6 frames) must survive trimming entirely -- this fixture's
    # own largest per-point deviation is real 8px-std jitter tail, not an
    # outlier, and discarding it would be exactly the "too aggressive"
    # failure mode this test guards against.
    assert diag["n_trimmed_per_index"] == [0, 0, 0, 0]


def test_average_correspondences_trim_below_min_detections_falls_back_to_plain_mean():
    """Below MIN_DETECTIONS_FOR_TRIMMING, a per-batch median/MAD estimate
    is too unstable to trust (e.g. n=2 or n=3) -- trimming must silently
    fall back to the exact old plain-mean behavior rather than make an
    overconfident call from too few samples."""
    obj = _obj_pts()
    img_a = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    img_b = np.array([[12.0, 8.0], [18.0, 14.0], [22.0, 18.0], [8.0, 22.0]])
    img_c = np.array([[11.0, 9.0], [19.0, 12.0], [21.0, 19.0], [9.0, 21.0]])
    detections = [(obj, img_a), (obj, img_b), (obj, img_c)]

    diag: dict = {}
    result = average_correspondences(detections, min_required=1, trim=True, trim_diagnostics_out=diag)
    assert result is not None
    _avg_obj, avg_img, n_used = result
    assert n_used == 3
    expected = np.mean(np.stack([img_a, img_b, img_c]), axis=0)
    assert np.allclose(avg_img, expected)
    assert diag["trimmed"] is False
