"""Tests for `opendarts/calibration/adaptive_ring_color.py` -- the
additive, NOT-live-wired adaptive 2-color (red/green ring) clustering
built as a from-scratch, image-only alternative to
`landmark_detection.py`'s fixed HSV thresholds.

Two tiers:

1. Synthetic tests of the CIRCULAR hue-clustering math itself (the part
   that's genuinely novel vs. the rest of this project's landmark-
   detection code) -- deliberately including cases that straddle the
   OpenCV hue 0/179 wraparound seam, since that's exactly the case naive
   (non-circular) clustering gets wrong -- plus the escalating-k /
   band-labeling / junk-isolation behavior (synthetic reproductions of
   the real corpus-measured failure shapes: a small off-hue patch, a
   few scattered off-hue pixels, and a WALL-DOMINANT non-ring
   population bigger than the real green ring). Synthetic is
   appropriate here because this is pure geometry/math, not real-image
   accuracy (compare to `test_landmark_detection.py`'s own reasoning
   for why IT uses only real images).
2. Real-image tests against `data/archive/clean/` (this project's real
   corpus), skipped cleanly when that data isn't present on a given
   machine/worktree. Confirms: (a) the module runs end-to-end on
   real captures, (b) its own confidence gates (separation, compactness)
   behave as measured in
   dev/calibration/validate_adaptive_ring_color.py's full-corpus run,
   (c) feeding its output through the SAME downstream boundary-trace +
   ellipse-fit code `landmark_detection.py`'s fixed path uses produces a
   plausible ellipse when the confidence gate says ok.

Tolerances/thresholds referenced below are the MEASURED real numbers
from the full-corpus validation run (13 sessions x 3 cams x 3 sampled
throws/session = 117 images), not guessed round numbers -- see
`adaptive_ring_color.py`'s own `MAX_CLUSTER_COMPACTNESS_DEG` docstring
for the full measurement this threshold came from.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from opendarts.calibration.adaptive_ring_color import (
    MAX_CLUSTER_COMPACTNESS_DEG,
    MAX_PIXEL_DIST_TO_CENTER_DEG,
    MIN_CLUSTER_SEPARATION_DEG,
    _circular_deg_distance,
    _circular_deg_distance_arr,
    _circular_mean_hue,
    _hue_to_unit_circle,
    _label_band,
    _unit_circle_to_hue,
    adaptive_ring_color_mask,
)

CLEAN_ROOT = Path(__file__).resolve().parents[1] / "data" / "archive" / "clean"


# --------------------------------------------------------------------
# Tier 1: synthetic tests of the circular hue math
# --------------------------------------------------------------------

def test_hue_unit_circle_roundtrip_is_exact_away_from_seam():
    hues = np.array([0.0, 10.0, 45.0, 90.0, 120.0, 150.0, 178.0])
    xy = _hue_to_unit_circle(hues)
    recovered = np.array([_unit_circle_to_hue(p) for p in xy])
    # Roundtrip through radians/degrees has float error, not exact zero
    # -- measure the real error rather than asserting exact equality.
    err = np.abs(((recovered - hues + 90) % 180) - 90)
    assert err.max() < 1e-6, f"roundtrip error {err.max()} too large"


def test_circular_distance_correctly_handles_0_179_wraparound():
    """The whole point of embedding on the unit circle: hue=2 and
    hue=178 are only 4 REAL degrees apart (both near the red seam,
    2*2=4 and 2*178=356 -> 4 apart mod 360), not the ~176*2=352-degrees-
    the-long-way a naive linear |a-b| on raw hue would compute."""
    d = _circular_deg_distance(2.0, 178.0)
    assert d < 10.0, f"expected a short wraparound distance, got {d}"

    naive_linear = abs(2.0 - 178.0)  # what a non-circular method would compute
    assert naive_linear > 170.0  # confirms the naive approach would be badly wrong
    assert d < naive_linear / 10


def test_circular_distance_is_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.uniform(0, 179, size=2)
        d_ab = _circular_deg_distance(a, b)
        d_ba = _circular_deg_distance(b, a)
        assert d_ab == pytest.approx(d_ba)
        assert 0.0 <= d_ab <= 180.0


def _synthetic_board_image(
    red_hue_cv2: float = 5.0,
    green_hue_cv2: float = 60.0,
    size: int = 200,
    noise_hue_std: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """A minimal synthetic BGR image: half red-ring-colored pixels, half
    green-ring-colored pixels (both high S/V), rest low-saturation
    "board face" gray -- enough to exercise the clustering end-to-end
    without depending on any real captured image."""
    import cv2

    rng = np.random.default_rng(seed)
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :, 1] = 40  # low-saturation background (below any real S/V filter)
    hsv[:, :, 2] = 200

    # Left half: red ring pixels (hue near red_hue_cv2, wrapping-safe
    # since red_hue_cv2 may itself be near 0/179).
    left = hsv[:, : size // 2]
    left_hue = (red_hue_cv2 + rng.normal(0, noise_hue_std, left.shape[:2])) % 180
    left[:, :, 0] = left_hue.astype(np.uint8)
    left[:, :, 1] = 200
    left[:, :, 2] = 180

    # Right half: green ring pixels.
    right = hsv[:, size // 2 :]
    right_hue = (green_hue_cv2 + rng.normal(0, noise_hue_std, right.shape[:2])) % 180
    right[:, :, 0] = right_hue.astype(np.uint8)
    right[:, :, 1] = 200
    right[:, :, 2] = 180

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_recovers_red_and_green_clusters_on_synthetic_seam_straddling_image():
    """Red placed AT the 0/179 seam (hue=2, i.e. real 4 degrees) --
    exactly the case naive 1-D clustering on raw hue mishandles. Circular
    embedding should still correctly separate it from green (hue=60)."""
    img = _synthetic_board_image(red_hue_cv2=2.0, green_hue_cv2=60.0)
    result = adaptive_ring_color_mask(img)
    assert result.ok, result.quality.reason
    q = result.quality
    # red center should land near true red (allow real clustering noise)
    assert _circular_deg_distance(q.red_hue_cv2, 2.0) < 15.0
    assert _circular_deg_distance(q.green_hue_cv2, 60.0) < 15.0
    assert q.cluster_separation_deg > MIN_CLUSTER_SEPARATION_DEG


def test_recovers_red_and_green_clusters_when_red_is_on_the_other_seam_side():
    """Same as above but red placed at hue=177 (the OTHER side of the
    seam from a raw-hue perspective) -- confirms the circular embedding
    doesn't have a directional bias."""
    img = _synthetic_board_image(red_hue_cv2=177.0, green_hue_cv2=60.0)
    result = adaptive_ring_color_mask(img)
    assert result.ok, result.quality.reason
    q = result.quality
    assert _circular_deg_distance(q.red_hue_cv2, 177.0) < 15.0
    assert _circular_deg_distance(q.green_hue_cv2, 60.0) < 15.0


def test_two_greenish_clusters_with_no_red_is_flagged_not_ok():
    """Both hue populations sit in the GREEN label band (50 and 60
    cv2-units = 100 and 120 real degrees) -- no red ring color exists in
    the image at all, so every k must fail with a missing-red-band
    refusal rather than silently labeling one of the greens 'red'."""
    img = _synthetic_board_image(red_hue_cv2=50.0, green_hue_cv2=60.0, noise_hue_std=1.0)
    result = adaptive_ring_color_mask(img)
    assert not result.ok
    assert result.mask is None
    assert "red" in result.quality.reason and "label band" in result.quality.reason


def test_low_separation_between_red_and_green_band_clusters_is_flagged_not_ok():
    """One population at the red band's inner edge (17 cv2 = 34 real
    degrees from 0) and one at the green band's inner edge (43 cv2 = 86
    real) -- both correctly LABELED red/green, but only ~52 real degrees
    apart, under MIN_CLUSTER_SEPARATION_DEG=60. No real board has its
    two ring colors this close; must refuse via the separation gate."""
    img = _synthetic_board_image(red_hue_cv2=17.0, green_hue_cv2=43.0, noise_hue_std=1.0)
    result = adaptive_ring_color_mask(img)
    assert not result.ok
    assert result.mask is None
    assert "separation" in result.quality.reason


def test_too_few_colorful_pixels_is_flagged_not_ok():
    """An all-gray (no saturated pixels at all) image -- the degenerate
    case this module's own MIN_COLORFUL_PIXELS gate exists for."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)  # flat gray, BGR
    result = adaptive_ring_color_mask(img)
    assert not result.ok
    assert result.mask is None
    assert "colorful pixels" in result.quality.reason


def test_injected_off_hue_patch_is_isolated_as_junk_and_kept_out_of_mask():
    """Reproduces the REAL failure mode found on the full corpus (one
    camera's field of view contains a large saturated non-ring
    population -- a cyan-blue wall/backdrop region): inject a third,
    off-hue "blue wall" patch. Under the original k=2-only design this
    patch was absorbed into the green cluster (inflating its compactness
    and forcing a refusal at best, silently corrupting the mask at
    worst); under the escalating-k design it must instead be ISOLATED
    into its own junk-band cluster at k=3, with the result accepted and
    the patch's pixels excluded from the returned mask."""
    import cv2

    img = _synthetic_board_image(red_hue_cv2=5.0, green_hue_cv2=60.0, size=200)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Strongly saturated "blue-ish" patch (~110 cv2 units = 220 real
    # degrees -- squarely in the junk gap between the green band's end
    # at 180 and the red band's start at 320). S/V matched to the
    # synthetic ring pixels' own S=200/V=180 (NOT higher/lower) so the
    # adaptive S/V "colorful" filter treats this patch exactly like
    # ring paint -- the clustering, not the S/V filter, must handle it.
    hsv[20:80, 150:190, 0] = 110
    hsv[20:80, 150:190, 1] = 200
    hsv[20:80, 150:190, 2] = 180
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = adaptive_ring_color_mask(img2)
    assert result.ok, result.quality.reason
    q = result.quality
    assert q.k_used >= 3, f"expected escalation past k=2, got k={q.k_used}"
    # The patch is 60x40 = 2400 px; allow HSV<->BGR roundtrip edge loss.
    assert q.n_junk_px >= 2000, f"expected the patch isolated as junk, n_junk_px={q.n_junk_px}"
    assert any(
        _circular_deg_distance(h, 110.0) < 10.0 for h in q.junk_center_hues_cv2
    ), f"expected a junk cluster near cv2-hue 110, got {q.junk_center_hues_cv2}"
    # Ring centers must be CLEAN (not dragged by the patch).
    assert _circular_deg_distance(q.red_hue_cv2, 5.0) < 10.0
    assert _circular_deg_distance(q.green_hue_cv2, 60.0) < 10.0
    assert max(q.red_compactness_deg, q.green_compactness_deg) <= MAX_CLUSTER_COMPACTNESS_DEG
    # And the patch region itself must be excluded from the mask.
    patch_fg = int((result.mask[20:80, 150:190] > 0).sum())
    assert patch_fg == 0, f"{patch_fg} junk-patch pixels leaked into the mask"


def test_scattered_off_hue_pixels_are_gated_out_of_mask_at_k2():
    """The per-pixel distance gate (MAX_PIXEL_DIST_TO_CENTER_DEG): a
    SMALL number of off-hue pixels too few to form their own cluster
    (unlike the patch test above) get absorbed into a ring cluster by
    k-means hard assignment -- the gate must still keep them out of the
    returned mask. Pixels planted at cv2-hue 40 (80 real degrees): 40
    real degrees from the green center (120 real), i.e. twice the gate,
    while barely moving the green center or its mean compactness."""
    import cv2

    img = _synthetic_board_image(red_hue_cv2=5.0, green_hue_cv2=60.0, size=200)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[90:105, 150:170, 0] = 40  # 15x20 = 300 px, in the green half
    hsv[90:105, 150:170, 1] = 200
    hsv[90:105, 150:170, 2] = 180
    img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Sanity of the test's own construction: the planted hue must be
    # beyond the gate from the true green center (40 real deg > gate).
    assert _circular_deg_distance(40.0, 60.0) > MAX_PIXEL_DIST_TO_CENTER_DEG

    result = adaptive_ring_color_mask(img2)
    assert result.ok, result.quality.reason
    q = result.quality
    # 300 px among ~20000 green members shifts the circular-mean center
    # by ~0.6 real degrees and mean compactness by ~0.6 -- k=2 passes.
    assert q.k_used == 2, f"expected no escalation for 300 stray px, got k={q.k_used}"
    assert q.n_gated_out_px >= 250, (
        f"expected the ~300 planted off-hue pixels gated out, n_gated_out_px={q.n_gated_out_px}"
    )
    planted_fg = int((result.mask[90:105, 150:170] > 0).sum())
    assert planted_fg == 0, f"{planted_fg} gated-out pixels leaked into the mask"


def test_wall_dominant_blue_population_recovered_by_escalating_k():
    """The measured real cam1 failure shape: the non-ring blue
    population is LARGER than the real green ring population (4-6x on
    the real corpus), so at k=2 it doesn't just inflate a cluster -- it
    TAKES OVER, dragging the merged center out of the green band
    entirely. Escalating k must recover the true ring pair."""
    import cv2

    size = 200
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    rng = np.random.default_rng(1)
    # rows 0-99: "wall" -- blue, hue 105, twice the area of either ring color
    hsv[:100, :, 0] = (105 + rng.normal(0, 2.0, (100, size))).astype(np.uint8)
    hsv[:100, :, 1] = 200
    hsv[:100, :, 2] = 180
    # rows 100-149: red ring pixels (hue 2, wraps)
    hsv[100:150, :, 0] = ((2 + rng.normal(0, 2.0, (50, size))) % 180).astype(np.uint8)
    hsv[100:150, :, 1] = 200
    hsv[100:150, :, 2] = 180
    # rows 150-199: green ring pixels
    hsv[150:, :, 0] = (60 + rng.normal(0, 2.0, (50, size))).astype(np.uint8)
    hsv[150:, :, 1] = 200
    hsv[150:, :, 2] = 180
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    result = adaptive_ring_color_mask(img)
    assert result.ok, result.quality.reason
    q = result.quality
    assert q.k_used >= 3
    assert _circular_deg_distance(q.red_hue_cv2, 2.0) < 10.0
    assert _circular_deg_distance(q.green_hue_cv2, 60.0) < 10.0
    assert q.n_junk_px >= 15000  # the 100x200 wall region
    # No wall pixel may enter the mask.
    wall_fg = int((result.mask[:100, :] > 0).sum())
    assert wall_fg == 0, f"{wall_fg} wall pixels leaked into the mask"


def test_label_band_boundaries():
    """_label_band encodes the universal color-NAME bands (red within
    40 real degrees of 0; green 80-180 real; junk elsewhere) -- pin the
    boundaries in cv2-hue units (half the real angle)."""
    assert _label_band(0.0) == "red"
    assert _label_band(20.0) == "red"      # 40 real deg -- inclusive edge
    assert _label_band(160.0) == "red"     # 320 real deg (i.e. -40)
    assert _label_band(179.0) == "red"
    assert _label_band(40.0) == "green"    # 80 real deg -- inclusive edge
    assert _label_band(65.0) == "green"    # real green ring paint (~130 real)
    assert _label_band(90.0) == "green"    # 180 real deg -- inclusive edge
    assert _label_band(21.0) == "junk"     # 42 real deg: orange gap
    assert _label_band(30.0) == "junk"     # 60 real deg: orange gap
    assert _label_band(105.0) == "junk"    # 210 real deg: the measured wall hue
    assert _label_band(130.0) == "junk"    # 260 real deg: blue


def test_circular_mean_hue_handles_seam():
    hues = np.array([178.0, 179.0, 1.0, 2.0])
    m = _circular_mean_hue(hues)
    assert m is not None
    assert _circular_deg_distance(m, 0.0) < 2.5


def test_circular_mean_hue_of_empty_input_is_none_not_nan():
    """An empty cluster's circular mean must be None, never NaN --
    np.cos([]).mean() is NaN, and NaN comparisons are all False, so a
    NaN leaking out of here would silently pass BOTH the `is None`
    check and the separation/compactness gates in _try_k (a verifier
    pass caught exactly this gap). Pin the contract."""
    m = _circular_mean_hue(np.array([]))
    assert m is None


def test_try_k_merges_multiple_same_band_clusters_across_the_seam():
    """Direct unit test of the same-band merge in _try_k: hand it
    points whose red population genuinely splits into TWO k-means
    clusters at k=4 (sub-modes at cv2-hue 174 and 6 -- opposite sides
    of the 0/179 seam, both inside the red label band), plus a green
    mode and a blue junk mode. The merge must (a) put BOTH red
    sub-modes' pixels into red_members, (b) compute the merged center
    across the seam (near 0, NOT the naive linear mean of 174 and 6 =
    90), and (c) pass the gates. This path is dormant on the current
    real corpus (no accepted image needs a multi-cluster same-band
    merge yet -- a verifier pass measured that breaking it changes no
    test and barely moves corpus numbers), so it's pinned here at the
    unit level rather than via a contrived end-to-end image."""
    from opendarts.calibration.adaptive_ring_color import _try_k

    rng = np.random.default_rng(3)
    red_a = (174.0 + rng.normal(0, 1.0, 4000)) % 180
    red_b = (6.0 + rng.normal(0, 1.0, 4000)) % 180
    green = 63.0 + rng.normal(0, 1.0, 6000)
    blue = 105.0 + rng.normal(0, 1.0, 6000)
    hue_vals = np.concatenate([red_a, red_b, green, blue])
    points = _hue_to_unit_circle(hue_vals).astype(np.float32)

    att = _try_k(points, hue_vals, 4)
    assert att.ok, att.fail_reason
    # Both red sub-modes merged: red_members must cover ~8000 points.
    assert att.red_members is not None
    assert int(att.red_members.sum()) >= 7500, int(att.red_members.sum())
    # Merged center across the seam: near 0 (i.e. within a few real
    # degrees of hue 0), NOT near the naive linear mean 90.
    assert _circular_deg_distance(att.red_center_hue, 0.0) < 8.0, att.red_center_hue
    assert _circular_deg_distance(att.red_center_hue, 90.0) > 90.0
    # 174 and 6 are 24 real degrees apart -> merged mean member
    # distance ~12 deg, under the 14-deg gate but well above a single
    # sub-mode's ~1.6 -- i.e. this asserts the merge REALLY happened.
    assert 8.0 < att.red_compactness <= MAX_CLUSTER_COMPACTNESS_DEG
    assert _circular_deg_distance(att.green_center_hue, 63.0) < 5.0
    assert len(att.junk_center_hues) == 1
    assert _circular_deg_distance(att.junk_center_hues[0], 105.0) < 5.0


def test_circular_deg_distance_arr_matches_scalar():
    rng = np.random.default_rng(2)
    hues = rng.uniform(0, 179, size=100)
    center = 63.0
    arr = _circular_deg_distance_arr(hues, center)
    for h, d in zip(hues, arr):
        assert d == pytest.approx(_circular_deg_distance(float(h), center))


def test_both_labels_get_foreground_in_returned_mask():
    """The returned mask should mark BOTH ring clusters' (post-gate)
    pixels as foreground (red|green, matching
    landmark_detection._color_mask's own red|green OR), not just one --
    and the quality's n_red_px/n_green_px must count exactly the mask's
    own foreground."""
    img = _synthetic_board_image(red_hue_cv2=5.0, green_hue_cv2=60.0)
    result = adaptive_ring_color_mask(img)
    assert result.ok
    n_fg = int((result.mask > 0).sum())
    assert n_fg == result.quality.n_red_px + result.quality.n_green_px
    assert result.quality.n_red_px > 0
    assert result.quality.n_green_px > 0


# --------------------------------------------------------------------
# Tier 2: real-image tests against data/archive/clean/ (skips cleanly
# when not present, same pattern as test_landmark_detection.py)
# --------------------------------------------------------------------

def _sampled_real_images() -> list[tuple[str, int, Path]]:
    if not CLEAN_ROOT.is_dir():
        return []
    out = []
    sessions = sorted(p for p in CLEAN_ROOT.iterdir() if p.is_dir())
    for session_dir in sessions[:3]:  # keep this test tier fast -- 3 sessions
        throw_dirs = sorted(
            p for p in session_dir.iterdir() if p.is_dir() and (p / "cam0_bg.png").is_file()
        )
        if not throw_dirs:
            continue
        first = throw_dirs[0]
        for cam in range(3):
            img_path = first / f"cam{cam}_bg.png"
            if img_path.is_file():
                out.append((session_dir.name, cam, img_path))
    return out


REAL_IMAGES = _sampled_real_images()

pytestmark_real = pytest.mark.skipif(
    not REAL_IMAGES, reason="data/archive/clean/ corpus not present on this machine"
)


@pytestmark_real
def test_real_corpus_images_produce_a_confidence_signal_without_crashing():
    """Every real (session, cam) sampled must return a well-formed
    result -- ok or not-ok, but never crash, and always carry a
    non-empty `reason` explaining itself."""
    import cv2

    assert REAL_IMAGES, "sampler found nothing -- fixture bug"
    for session, cam, img_path in REAL_IMAGES:
        img = cv2.imread(str(img_path))
        assert img is not None, img_path
        result = adaptive_ring_color_mask(img)
        assert result.quality.reason, f"{session} cam{cam}: empty reason"
        if result.ok:
            assert result.mask is not None
            assert result.mask.shape == img.shape[:2]
        else:
            assert result.mask is None


@pytestmark_real
def test_real_corpus_confident_results_agree_with_fixed_hsv_downstream_ellipse():
    """For real images where adaptive_ring_color_mask is confident
    (ok=True, i.e. passed both the separation AND compactness gates),
    feeding its mask through landmark_detection's SAME downstream
    boundary-trace + ellipse-fit code should land close to the existing
    fixed-HSV pipeline's own ellipse.

    Bound is the MEASURED real number from the full 117-image corpus
    validation (dev/calibration/validate_adaptive_ring_color.py),
    accepted subset (114/117): center delta median 3.3px, p90 18.4px,
    max 141px -- and that max was overlaid and inspected: it's the
    FIXED method's fit that's wrong there (minor axis 1126px on a
    720px-tall image), not the adaptive one; see
    adaptive_ring_color.py's module docstring for the honest
    breakdown. Margin added on top of the measured p90, not
    guessed."""
    import cv2

    from opendarts.calibration import landmark_detection as ld

    n_confident = 0
    center_deltas = []
    for session, cam, img_path in REAL_IMAGES:
        img = cv2.imread(str(img_path))
        adaptive = adaptive_ring_color_mask(img)
        if not adaptive.ok:
            continue  # this test only checks the CONFIDENT subset
        fixed_result = ld.detect_double_ring_quad(img)
        if not fixed_result.ok:
            continue

        component_mask = ld._outer_ring_component_mask(adaptive.mask)
        if component_mask is None:
            continue
        boundary_pts = ld._trace_outer_boundary(adaptive.mask, component_mask)
        if boundary_pts is None:
            continue
        ellipse_tuple, _kept = ld._robust_fit_ellipse(boundary_pts)
        if ellipse_tuple is None:
            continue
        adaptive_ellipse = ld.Ellipse.from_cv2(ellipse_tuple)

        n_confident += 1
        fe = fixed_result.ellipse
        center_delta = float(np.hypot(fe.cx - adaptive_ellipse.cx, fe.cy - adaptive_ellipse.cy))
        center_deltas.append(center_delta)

    assert n_confident >= 3, f"too few confident+comparable real images ({n_confident}) to assert anything"
    # Measured p90 on the full corpus was ~19px; generous margin on top
    # (this small 3-session sample can land anywhere in that
    # distribution by chance) rather than a tight bound that would flake.
    median_delta = float(np.median(center_deltas))
    assert median_delta < 60.0, (
        f"median ellipse-center delta {median_delta:.1f}px on the confident "
        f"subset is far outside the measured full-corpus range (p90 ~19px)"
    )
