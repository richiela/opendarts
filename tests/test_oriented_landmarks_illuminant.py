"""Tests for `oriented_landmarks.normalise_illuminant()` and the
illuminant-invariance it buys the whole detector.

WHY THIS FILE EXISTS (real incident, 2026-08-13). Camera 1's white
balance sat green-shifted for the first 60 of the 2026-08-13 session's
180 real throws and then returned to normal, with nothing physical
moving. `landmark_detection`'s colour mask thresholds HSV against
ABSOLUTE cut points (`GREEN_HSV_LOW = (35, 60, 40)`), so under that cast
dark sisal bristle crossed `S >= 60` with a green hue and was masked as
ring paint: the seed ellipse and the bull-anchored re-seat both came out
~20% too big (minor axis 806-909 px against a true ~671), which pushed
`angular_edge_profile`'s radial band off the sector wires and collapsed
the phase lock. On the real corpus the correction took that block's
usable-frame rate from 18.3% to 100.0% and its ellipse from
804.0 px +-86.33 to 672.4 px +-2.93.

Those real frames are gitignored (`/data/archive/clean/*`), so the
regression is reproduced here SYNTHETICALLY -- a rendered board plus
sensor noise plus a deliberate per-channel gain -- which is also the
stronger test: it can assert the exact property the correction is
supposed to have (invariance to a per-channel gain), which a single real
frame cannot.

Every tolerance below was measured first and is quoted with the observed
value, per this project's standing rule.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from opendarts.calibration.oriented_landmarks import (
    MAX_ILLUMINANT_GAIN,
    apply_homography,
    find_oriented_landmarks,
    normalise_illuminant,
)
from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM, polar_to_xy_mm

from tests.test_oriented_landmarks import (
    render_synthetic_board,
    synthetic_board_homography,
)

# The cast used below. Chosen to be the same SHAPE as the real measured
# one (green gained, red and blue slightly cut) but stronger, so the
# failure is unambiguous rather than marginal: the real 2026-08-13 cam1
# frames needed correction gains of only (1.024, 0.941, 1.041) BGR.
CAST_BGR = (0.97, 1.14, 0.97)
NOISE_SD = 6.0


def _sensor_frame(base, gains_bgr=(1.0, 1.0, 1.0), noise_sd=NOISE_SD, seed=0):
    """`base` with additive sensor noise and a per-channel gain.

    The noise is not decoration: a NOISELESS rendered board has perfectly
    flat, perfectly neutral dark beds, whose saturation a gain of this
    size cannot push past 60 -- so a noiseless synthetic board is immune
    to the very failure this file is about, and would make the test pass
    vacuously. Real bristle is textured, which spreads per-pixel
    saturation and is exactly what lets the cast tip a large fraction of
    dark pixels over an absolute threshold.
    """
    rng = np.random.default_rng(seed)
    f = base.astype(np.float32) + rng.normal(0.0, noise_sd, base.shape).astype(np.float32)
    f = f * np.asarray(gains_bgr, dtype=np.float32)[None, None, :]
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def _quad_error_px(img, H_mm, *, white_balance):
    """Worst of the 4 quad landmarks' distance from synthetic truth.

    `local_refine=False`: this file tests the ILLUMINANT correction, on
    the shared synthetic render whose wires are painted DARK -- the
    opposite polarity to the real rig's bright metal wires that the
    local wire-junction refinement stage (`opendarts.calibration.
    wire_junction`, 2026-08-15) is built for and measured on. On this
    render the refinement correctly rejects most landmarks, but the
    sensor noise this file deliberately adds creates thin bright rims
    beside the dark wire lines that can pass its gates on a few points
    (measured: 4/20 accepted, up to 8.7px moved) -- an artefact of the
    render's inverted wire polarity, not the failure mode under test
    here, so the stage is disabled to keep this file measuring what it
    is about.
    """
    truth = apply_homography(
        H_mm, [polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, 9.0 + 90.0 * i) for i in range(4)]
    )
    hint_pt = apply_homography(H_mm, [polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, 0.0)])[0]

    first = find_oriented_landmarks(
        img, min_phase_confidence=0.0, white_balance=white_balance,
        local_refine=False,
    )
    if first.bull_px is None:
        return float("inf"), first
    hint = math.degrees(
        math.atan2(hint_pt[1] - first.bull_px[1], hint_pt[0] - first.bull_px[0])
    ) % 360.0
    res = find_oriented_landmarks(
        img,
        orientation_hint_deg=hint,
        min_phase_confidence=0.0,
        white_balance=white_balance,
        local_refine=False,
    )
    if res.quad_px is None:
        return float("inf"), res
    err = np.hypot(res.quad_px[:, 0] - truth[:, 0], res.quad_px[:, 1] - truth[:, 1])
    return float(err.max()), res


# ---------------------------------------------------------------------
# normalise_illuminant() itself
# ---------------------------------------------------------------------


def test_equalises_the_channel_means():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 200, (64, 96, 3), dtype=np.uint8)
    img = (img.astype(np.float32) * np.array([0.8, 1.3, 1.0], np.float32)).astype(np.uint8)
    before = img.reshape(-1, 3).mean(axis=0)
    after = normalise_illuminant(img).reshape(-1, 3).mean(axis=0)
    # Measured: channel means (78.52, 126.78, 100.46), spread 48.26 grey
    # levels before -> (101.43, 101.41, 101.44), spread 0.03 after.
    assert before.max() - before.min() > 20.0, before
    assert after.max() - after.min() < 0.5, after


def test_leaves_an_already_neutral_image_alone():
    rng = np.random.default_rng(2)
    img = rng.integers(20, 220, (64, 96, 3), dtype=np.uint8)
    out = normalise_illuminant(img)
    # Measured max per-pixel change on this already-grey-world input: 1
    # grey level, i.e. output rounding only.
    assert int(np.abs(out.astype(np.int16) - img.astype(np.int16)).max()) <= 2


@pytest.mark.parametrize("seed", [3, 4, 5])
@pytest.mark.parametrize("gains", [(0.9, 1.15, 1.0), (0.97, 1.14, 0.97), (1.05, 0.95, 1.0)])
def test_inverts_a_known_per_channel_gain(seed, gains):
    """Correcting a cast image and correcting the uncast original land on
    the same image -- a gain is exactly what grey-world undoes.

    The residual is uint8 re-quantisation, not correction error: the cast
    is rounded to 8 bits BEFORE being corrected, and a channel gained by
    0.9 loses ~10% of its levels on the way in and cannot get them back.
    That is a property of the synthetic test setup, not of the corrector
    -- a real camera never round-trips through a second quantisation this
    way. Measured over 5 seeds x 3 gain triples: worst mean |difference|
    2.61 grey levels, worst max 6, and the mildest cast (1.05, 0.95, 1.0)
    -- which loses almost no levels -- comes back at mean 0.04-0.58, max
    1. Tolerances set from the worst measured, not from the mildest.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(20, 200, (64, 96, 3), dtype=np.uint8)
    cast = np.clip(base.astype(np.float32) * np.asarray(gains, np.float32),
                   0, 255).astype(np.uint8)
    a = normalise_illuminant(base).astype(np.int16)
    b = normalise_illuminant(cast).astype(np.int16)
    raw = np.abs(base.astype(np.int16) - cast.astype(np.int16))
    # The correction has to actually do something: the uncorrected pair
    # differs by far more than the corrected pair does.
    assert float(np.abs(a - b).mean()) < 4.0
    assert int(np.abs(a - b).max()) <= 8
    if float(raw.mean()) > 4.0:
        assert float(np.abs(a - b).mean()) < 0.5 * float(raw.mean())


def test_gains_are_clamped():
    img = np.zeros((32, 32, 3), np.uint8)
    img[:, :, 0] = 200   # B
    img[:, :, 1] = 4     # G -- would need a gain of ~34x without the clamp
    img[:, :, 2] = 200   # R
    out = normalise_illuminant(img)
    assert int(out[:, :, 1].max()) <= int(math.ceil(4 * MAX_ILLUMINANT_GAIN))


@pytest.mark.parametrize(
    "img",
    [
        np.zeros((8, 8, 3), np.uint8),                       # all-black
        np.zeros((8, 8), np.uint8),                          # not 3-channel
        np.dstack([np.zeros((8, 8), np.uint8),               # one dead channel
                   np.full((8, 8), 100, np.uint8),
                   np.full((8, 8), 100, np.uint8)]),
    ],
)
def test_degenerate_input_is_returned_unchanged_not_divided_by_zero(img):
    out = normalise_illuminant(img)
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_does_not_modify_its_input():
    rng = np.random.default_rng(4)
    img = (rng.integers(20, 200, (32, 32, 3)) * np.array([0.8, 1.2, 1.0])).astype(np.uint8)
    before = img.copy()
    normalise_illuminant(img)
    assert np.array_equal(img, before)


# ---------------------------------------------------------------------
# The detector-level regression this exists for
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_a_colour_cast_no_longer_wrecks_detection_even_without_the_correction(cam_index):
    """UPDATED 2026-08-21 (adaptive-ring-color live-wiring task) -- this
    test used to be named `test_a_colour_cast_wrecks_detection_without_
    the_correction` and demonstrated the ORIGINAL failure: without
    illuminant normalisation, this cast flooded the fixed-HSV colour
    mask (`landmark_detection._color_mask()`'s absolute hue cutoffs), the
    ring component swallowed the beds and background, and the fitted
    ellipse blew up (measured then: ellipse minor axis 383.2px ->
    1699-1722px, worst quad landmark 1.58-2.56px -> 631-860px).

    `landmark_detection.detect_double_ring_quad()` now tries
    `adaptive_ring_color_mask()` first (per-image hue clustering, not an
    absolute cutoff a per-channel gain can slide out from under -- see
    `_mask_for_detection()`'s own docstring). Re-measured with the SAME
    cast, correction still OFF: cast_err is now 2.10-2.36px, essentially
    indistinguishable from the clean (uncast) 1.96-2.08px baseline on
    all 3 cameras -- the failure this test used to pin no longer
    reproduces via this specific synthetic cast.

    **This does NOT mean illuminant normalisation is now redundant.**
    It stays on by default (see `test_white_balance_is_on_by_default`,
    updated the same day) and remains the correct fix for: (a) a
    stronger/different real cast than this one, and (b) the fixed-HSV
    FALLBACK path, which still runs unmodified whenever the adaptive
    method's own confidence gate refuses (~3/117 real images, see
    `opendarts.calibration.adaptive_ring_color`'s own docstring) -- that
    fallback has the exact same absolute-hue-cutoff vulnerability this
    test originally demonstrated. Kept as a real regression guard on the
    NEW behavior (adaptive segmentation's own cast robustness), not
    deleted, so a future change that breaks adaptive's illuminant
    handling is caught here rather than silently assumed away.
    """
    H_mm, _ = synthetic_board_homography(cam_index)
    base = render_synthetic_board(H_mm)

    clean_err, clean = _quad_error_px(_sensor_frame(base), H_mm, white_balance=False)
    cast_err, cast = _quad_error_px(
        _sensor_frame(base, CAST_BGR), H_mm, white_balance=False
    )

    assert clean_err < 5.0, clean_err
    assert cast_err < 5.0, cast_err
    assert cast.ellipse is not None and clean.ellipse is not None
    assert cast.ellipse.minor_axis_px < 1.5 * clean.ellipse.minor_axis_px


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_the_correction_makes_detection_invariant_to_the_cast(cam_index):
    """The fix, stated as the property it actually has: with the
    correction on, a cast frame gives the SAME answer as the uncast one,
    not merely a better one.

    Measured worst-quad-landmark error, uncast (no correction needed)
    vs cast + correction: 2.37 / 1.58 / 2.56 px vs 2.38 / 1.58 / 2.57 px
    on cams 0/1/2 -- agreeing to 0.01 px, and both sitting on the same
    ~1.6-2.6 px floor `test_end_to_end_recovers_the_true_quad_on_a_
    rendered_board` already records for the rendered wire's own width.
    """
    H_mm, _ = synthetic_board_homography(cam_index)
    base = render_synthetic_board(H_mm)

    clean_err, _ = _quad_error_px(_sensor_frame(base), H_mm, white_balance=False)
    fixed_err, fixed = _quad_error_px(
        _sensor_frame(base, CAST_BGR), H_mm, white_balance=True
    )

    assert fixed_err < 5.0, fixed_err
    assert abs(fixed_err - clean_err) < 0.5, (fixed_err, clean_err)
    assert fixed.ok
    assert not fixed.orientation_ambiguous
    # And the phase lock is back above the shipped quality gate rather
    # than merely finite. Measured on the cast frame: 1.62-2.32 sigma
    # without the correction, 2.63-2.70 with it.
    assert fixed.phase_confidence > 2.4, fixed.phase_confidence


def test_white_balance_is_on_by_default():
    """The production callers pass nothing, so the default is what ships.

    Asserted through the real entry point rather than by reading the
    signature. UPDATED 2026-08-21 (adaptive-ring-color live-wiring
    task): this test used to compare `default` against
    `white_balance=False` and require a >=2x ellipse-size gap under a
    colour cast -- that gap has genuinely closed (see
    `test_a_colour_cast_no_longer_wrecks_detection_even_without_the_
    correction`'s own docstring: adaptive ring-color segmentation is now
    independently robust to this cast, so `white_balance=False` no
    longer produces a visibly-worse result to compare against). Proving
    "on by default" no longer works via a behavioral SIDE EFFECT that
    has stopped being visible -- compare directly against the explicit
    `white_balance=True` value instead: if the default really is True,
    calling with no keyword must produce a NUMERICALLY IDENTICAL result
    to calling with `white_balance=True` explicitly, real values, not
    just "both succeeded."
    """
    H_mm, _ = synthetic_board_homography(1)
    img = _sensor_frame(render_synthetic_board(H_mm), CAST_BGR)
    default = find_oriented_landmarks(img, min_phase_confidence=0.0)
    explicit_on = find_oriented_landmarks(
        img, min_phase_confidence=0.0, white_balance=True
    )
    assert default.ellipse is not None and explicit_on.ellipse is not None
    assert default.ellipse.minor_axis_px == pytest.approx(explicit_on.ellipse.minor_axis_px)
    assert default.ellipse.major_axis_px == pytest.approx(explicit_on.ellipse.major_axis_px)
    assert default.ellipse.cx == pytest.approx(explicit_on.ellipse.cx)
    assert default.ellipse.cy == pytest.approx(explicit_on.ellipse.cy)
