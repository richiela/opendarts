"""Tests for `opendarts.calibration.wire_junction` -- the local per-landmark
wire-junction refinement stage -- and its integration through
`find_oriented_landmarks(local_refine=...)`.

Synthetic ground truth throughout, per the project's standing rule:
a bright two-wire junction is DRAWN at a known displacement off a known
ellipse (the exact geometry the stage exists to recover -- the real
measured failure was the true junction sitting up to ~10px radially off
the globally-fitted ellipse), and the refinement must find it. Every
tolerance below records the value actually measured on this fixture
(run of 2026-08-15) rather than a round
guess:

    full-mode error vs drawn truth:   0.66-0.80 px   (assert < 1.5)
    radial-only radial residual:      ~0.3 px        (assert < 1.5)
    seed errors these recover from:   5.6-8.5 px
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from opendarts.calibration.landmark_detection import Ellipse
from opendarts.calibration.oriented_landmarks import ellipse_ray_intersection
from opendarts.calibration.wire_junction import (
    JunctionRefinement,
    _fit_arm_line,
    refine_ring_junctions,
    refine_wire_junction,
    ridge_map,
    suggested_tophat_kernel_px,
)

BED_WIDTH_PX = 16.0
SEED_ANGLE_DEG = 205.0
WIRE_VAL = 230
BG_VAL = 60


def make_junction_image(
    dr_true: float = -8.0,
    dt_true_deg: float = 1.0,
    *,
    draw_spoke: bool = True,
    w: int = 800,
    h: int = 600,
):
    """A bright ring-wire arc drawn along the ellipse radially displaced
    by `dr_true`, and (optionally) a bright spoke wire through the true
    junction at `SEED_ANGLE_DEG + dt_true_deg` -- exactly the situation
    the refinement exists for: the SEED sits on the undisplaced ellipse
    at the undisplaced angle, and the truth is elsewhere.

    Returns (image, ellipse, bull, seed_xy, true_junction_xy).
    """
    ell = Ellipse(cx=400, cy=300, major_axis_px=460, minor_axis_px=420, angle_deg=15.0)
    bull = (380.0, 320.0)
    img = np.full((h, w, 3), BG_VAL, np.uint8)

    ang_t = SEED_ANGLE_DEG + dt_true_deg
    u = np.array([math.cos(math.radians(ang_t)), math.sin(math.radians(ang_t))])
    true_j = np.asarray(ellipse_ray_intersection(ell, bull, ang_t)) + dr_true * u

    arc = []
    for da in np.arange(-20.0, 20.0, 0.1):
        uu = np.array([math.cos(math.radians(ang_t + da)),
                       math.sin(math.radians(ang_t + da))])
        arc.append(np.asarray(ellipse_ray_intersection(ell, bull, ang_t + da))
                   + dr_true * uu)
    cv2.polylines(img, [np.asarray(arc, np.int32)], False, (WIRE_VAL,) * 3, 2, cv2.LINE_AA)
    if draw_spoke:
        p_in = true_j - 60.0 * u
        p_out = true_j + 7.0 * u   # the short outward overshoot real spiders have
        cv2.line(img, tuple(np.round(p_in).astype(int)),
                 tuple(np.round(p_out).astype(int)), (WIRE_VAL,) * 3, 2, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    seed = np.asarray(ellipse_ray_intersection(ell, bull, SEED_ANGLE_DEG))
    return img, ell, bull, seed, true_j


def refine_on(img, ell, bull, seed) -> JunctionRefinement:
    return refine_wire_junction(
        ridge_map(img, 7), ell, bull, SEED_ANGLE_DEG, tuple(seed),
        bed_width_px=BED_WIDTH_PX,
    )


# ---------------------------------------------------------------------
# ridge_map
# ---------------------------------------------------------------------


def test_ridge_map_keeps_thin_bright_lines_and_suppresses_wide_blocks():
    """The whole stage keys on this property. Measured on this exact
    fixture: line response 155.6 at its centre, wide-block interior
    response 0.0 (the block is wider than the kernel, so the top-hat
    removes it entirely)."""
    img = np.full((200, 200, 3), BG_VAL, np.uint8)
    cv2.line(img, (20, 100), (180, 100), (WIRE_VAL,) * 3, 2)       # thin: survives
    cv2.rectangle(img, (20, 20), (180, 60), (WIRE_VAL,) * 3, -1)   # wide: suppressed
    r = ridge_map(img, 7)
    assert r.shape == (200, 200)
    assert r[100, 100] > 100.0
    assert abs(r[40, 100]) < 1.0


def test_suggested_tophat_kernel_is_odd_and_clamped():
    ks = [suggested_tophat_kernel_px(w) for w in (0.0, 4.0, 12.0, 18.0, 40.0, 400.0)]
    assert all(k % 2 == 1 for k in ks)
    assert min(ks) >= 5 and max(ks) <= 13
    # monotone-ish with bed width inside the clamp range
    assert suggested_tophat_kernel_px(12.0) <= suggested_tophat_kernel_px(20.0)


# ---------------------------------------------------------------------
# _fit_arm_line
# ---------------------------------------------------------------------


def test_fit_arm_line_recovers_a_clean_line():
    s = np.array([-10.0, -6.0, -2.0, 2.0, 6.0, 10.0])
    u = 0.5 + 0.1 * s
    c0, c1, rms = _fit_arm_line(s, u)
    assert abs(c0 - 0.5) < 1e-9 and abs(c1 - 0.1) < 1e-9 and rms < 1e-9


def test_fit_arm_line_trims_a_gross_outlier():
    """One station centred on the wrong structure must not drag the
    line. Measured: a plain (or RMS-multiple-trimmed) fit of this data
    lands c0 = 1.06 / rms = 1.49; the MAD-based trim recovers the clean
    line to machine precision."""
    s = np.array([-10.0, -6.0, -2.0, 2.0, 6.0, 10.0, 4.0])
    u = np.concatenate([0.5 + 0.1 * s[:-1], [5.0]])   # last one is junk
    c0, c1, rms = _fit_arm_line(s, u)
    assert abs(c0 - 0.5) < 0.05 and abs(c1 - 0.1) < 0.02 and rms < 0.1


def test_fit_arm_line_degenerate_returns_none():
    assert _fit_arm_line(np.array([1.0, 1.0]), np.array([0.0, 0.0])) is None


# ---------------------------------------------------------------------
# refine_wire_junction on the drawn fixture
# ---------------------------------------------------------------------


@pytest.mark.parametrize("dr_true,dt_true", [(-8.0, 1.0), (5.0, -0.8)])
def test_full_mode_recovers_a_displaced_junction(dr_true, dt_true):
    """The headline behaviour: a junction drawn 5.6-8.5px away from the
    ellipse seed (the real rig's measured bias was up to ~10px) comes
    back to within 0.66-0.80px measured; assert < 1.5px."""
    img, ell, bull, seed, true_j = make_junction_image(dr_true, dt_true)
    r = refine_on(img, ell, bull, seed)
    seed_err = float(np.linalg.norm(seed - true_j))
    assert r.ok and r.mode == "full"
    assert seed_err > 4.0            # the fixture really displaced the seed
    assert float(np.linalg.norm(np.asarray(r.xy) - true_j)) < 1.5


def test_radial_only_mode_when_the_spoke_is_missing():
    """No spoke wire drawn: the ring evidence alone must still fix the
    RADIAL error and leave the tangential position seeded. Measured:
    radial residual 0.3px (of a drawn 8.0), total residual 3.7px (the
    tangential part of the fixture's 1.0deg offset, untouched by
    construction)."""
    img, ell, bull, seed, true_j = make_junction_image(-8.0, 1.0, draw_spoke=False)
    r = refine_on(img, ell, bull, seed)
    assert r.ok and r.mode == "radial_only"
    u = np.array([math.cos(math.radians(SEED_ANGLE_DEG)),
                  math.sin(math.radians(SEED_ANGLE_DEG))])
    radial_resid = abs(float((np.asarray(r.xy) - true_j) @ u))
    assert radial_resid < 1.5
    # and it must beat the seed overall, not just radially
    assert (np.linalg.norm(np.asarray(r.xy) - true_j)
            < np.linalg.norm(seed - true_j))


def test_blank_image_rejects_and_returns_the_seed():
    img, ell, bull, seed, _ = make_junction_image()
    blank = np.full_like(img, BG_VAL)
    r = refine_on(blank, ell, bull, seed)
    assert not r.ok and r.mode == "rejected"
    assert np.allclose(r.xy, seed)


def test_junction_outside_the_bounded_window_rejects():
    """A junction drawn 25px inward -- past the radial window (max 14px)
    -- must NOT be chased: bounded search means bounded, and the seed
    comes back untouched."""
    img, ell, bull, seed, _ = make_junction_image(-25.0, 0.0)
    r = refine_on(img, ell, bull, seed)
    assert not r.ok and np.allclose(r.xy, seed)


def test_degenerate_bed_width_rejects():
    img, ell, bull, seed, _ = make_junction_image()
    r = refine_wire_junction(ridge_map(img, 7), ell, bull, SEED_ANGLE_DEG,
                             tuple(seed), bed_width_px=0.0)
    assert not r.ok and "bed width" in r.reason


# ---------------------------------------------------------------------
# refine_ring_junctions + find_oriented_landmarks integration
# ---------------------------------------------------------------------


def _rendered_board_case():
    from tests.test_oriented_landmarks import (
        render_synthetic_board, synthetic_board_homography,
    )

    H_mm, _ = synthetic_board_homography(1)
    return render_synthetic_board(H_mm), H_mm


def test_find_oriented_landmarks_flag_wiring():
    """`local_refine=False` must reproduce the pure ellipse path
    (modes None, no refine note); `True` must populate the per-landmark
    modes, and every REJECTED landmark's coordinates must be identical
    to the unrefined ones -- the graceful per-point fallback exercised
    end to end.

    (This render's wires are dark -- the opposite polarity to the real
    rig's bright wires -- so most landmarks reject here; measured on
    this frame 16-20 of 20 reject, the remainder latching onto thin
    bright bed-edge artefacts of the render. The per-point fallback
    contract, not a specific mode count, is what this asserts.)"""
    from opendarts.calibration.oriented_landmarks import find_oriented_landmarks

    img, _H = _rendered_board_case()
    off = find_oriented_landmarks(img, min_phase_confidence=0.0, local_refine=False)
    on = find_oriented_landmarks(img, min_phase_confidence=0.0, local_refine=True)
    assert off.ring20_refine_modes is None
    assert not any("local refine" in n for n in off.notes)
    assert on.ring20_refine_modes is not None and len(on.ring20_refine_modes) == 20
    assert any("local refine" in n for n in on.notes)
    rejected = [i for i, m in enumerate(on.ring20_refine_modes) if m == "rejected"]
    assert rejected, "expected at least some rejections on a dark-wire render"
    assert np.allclose(on.ring20_px[rejected], off.ring20_px[rejected])


def test_quad_stays_consistent_with_ring20():
    """The 4-point quad must always be ring20's rows 0/5/10/15, refined or
    not -- the index contract every downstream consumer relies on."""
    from opendarts.calibration.oriented_landmarks import (
        AD_QUAD_RING_INDICES, find_oriented_landmarks,
    )

    img, _H = _rendered_board_case()
    res = find_oriented_landmarks(img, min_phase_confidence=0.0)
    assert res.ok
    assert np.array_equal(res.quad_px, res.ring20_px[list(AD_QUAD_RING_INDICES)])


def test_refine_ring_junctions_is_index_aligned_and_always_usable():
    """One drawn junction among 20 requested landmarks: the output list
    must be index-aligned, every entry's xy usable (seed on failure),
    and only physically-plausible entries refined."""
    img, ell, bull, seed, true_j = make_junction_image(-8.0, 1.0)
    from opendarts.calibration.oriented_landmarks import board_to_image_homography

    # A homography consistent with the fixture ellipse/bull (phase is
    # irrelevant here -- only the bed-width prediction uses it).
    H = board_to_image_homography(ell, bull, 0.0)
    angles = [SEED_ANGLE_DEG + 18.0 * k for k in range(20)]
    ring = np.asarray([ellipse_ray_intersection(ell, bull, a) or (np.nan, np.nan)
                       for a in angles])
    refs = refine_ring_junctions(img, ell, bull, angles, ring, H, 162.0 / 170.0)
    assert len(refs) == 20
    for i, r in enumerate(refs):
        assert np.isfinite(r.xy).all()
        if not r.ok:
            assert np.allclose(r.xy, ring[i], equal_nan=True) or np.isnan(ring[i]).any()
    # the drawn junction is at index 0's angle; it must have refined
    assert refs[0].ok
    assert float(np.linalg.norm(np.asarray(refs[0].xy) - true_j)) < 1.5
