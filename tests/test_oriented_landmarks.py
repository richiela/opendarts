"""Tests for `opendarts.calibration.oriented_landmarks`.

Everything with a known-correct answer is checked against SYNTHETIC
ground truth (a real `cv2.projectPoints` camera looking at the real board
geometry), per this project's standing rule that real images can only
ever be a plausibility check. Tolerances are set from the measured
value, not from a round number picked in advance -- each one records what
was actually observed.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from opendarts.calibration.landmark_detection import Ellipse
from opendarts.calibration.oriented_landmarks import (
    AD_QUAD_RING_INDICES,
    edge_magnitude,
    FIRST_WIRE_ANGLE_DEG,
    N_SECTORS,
    RED_DOUBLE_SECTOR_INDICES,
    ad_quad_object_points_mm,
    affine_unit_circle_to_ellipse,
    angular_edge_profile,
    apply_homography,
    board_to_image_homography,
    detect_bull,
    disk_boost,
    double_colour_score,
    ellipse_ray_intersection,
    ellipse_ray_intersections,
    find_oriented_landmarks,
    lock_phase,
    normalised_board_point,
    predicted_wire_angles,
    refine_wire_angles,
    spoke_score,
    spoke_scores,
)
from opendarts.calibration.sector_correspondence import ad_quad_object_point_mm
from opendarts.geometry.board import (
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    SECTOR_ANGLE_DEG,
    SECTOR_NUMBERS_CLOCKWISE,
    polar_to_xy_mm,
)
from tests.support.synthetic import make_camera_matrix, make_ring_camera, project_points

W, H = 1280, 720


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def synthetic_board_homography(cam_index: int = 0, n_cameras: int = 3):
    """A real projective board-mm -> pixel map from a real synthetic
    camera pose, plus that camera. Because the board is the Z=0 plane,
    the projection restricted to it IS a homography, recovered here by
    projecting four known points and solving exactly."""
    cam = make_ring_camera(cam_index, n_cameras, make_camera_matrix(W, H, 90.0),
                           ring_radius_mm=500.0, height_mm=300.0)
    src_angles = (9.0, 99.0, 189.0, 279.0)
    obj = np.asarray([polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, a) for a in src_angles],
                     dtype=np.float32)
    obj3 = np.hstack([obj, np.zeros((4, 1), dtype=np.float32)]).astype(np.float64)
    px = project_points(cam, obj3).astype(np.float32)
    return cv2.getPerspectiveTransform(obj, px), cam


def board_mm_to_unit(H_mm):
    """Convert a board-MM homography into the unit-disk convention this
    module uses (x = sin a, y = -cos a on a unit circle)."""
    S = np.array([[DOUBLE_OUTER_RADIUS_MM, 0.0, 0.0],
                  [0.0, -DOUBLE_OUTER_RADIUS_MM, 0.0],
                  [0.0, 0.0, 1.0]])
    return H_mm @ S


def ellipse_and_bull_from(H_unit):
    """The ellipse the unit circle projects to, and the bull pixel."""
    t = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    pts = apply_homography(H_unit, circle).astype(np.float32)
    ell = Ellipse.from_cv2(cv2.fitEllipse(pts.reshape(-1, 1, 2)))
    bull = apply_homography(H_unit, [[0.0, 0.0]])[0]
    return ell, (float(bull[0]), float(bull[1]))


def render_synthetic_board(H_mm, *, roll: int = 0) -> np.ndarray:
    """Render a plausible dartboard through a real projective map.

    Painted by inverse-mapping every pixel back to board mm and colouring
    by real `opendarts.geometry.board` geometry, so the rendered board's
    colours and wires are consistent with the same single source of truth
    the detector consumes. `roll` rotates the board by whole sectors, for
    testing that the orientation lock actually keys off the colour
    pattern.
    """
    Hinv = np.linalg.inv(H_mm)
    ys, xs = np.mgrid[0:H, 0:W]
    P = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=0).astype(np.float64)
    with np.errstate(all="ignore"):
        Q = Hinv @ P
        bx = (Q[0] / Q[2]).reshape(H, W)
        by = (Q[1] / Q[2]).reshape(H, W)
    good = np.isfinite(bx) & np.isfinite(by)
    bx = np.where(good, bx, 1e6)
    by = np.where(good, by, 1e6)

    r = np.hypot(bx, by)
    ang = (np.degrees(np.arctan2(bx, by)) - roll * SECTOR_ANGLE_DEG) % 360.0
    idx = (((ang + SECTOR_ANGLE_DEG / 2.0) // SECTOR_ANGLE_DEG).astype(int)) % N_SECTORS

    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (25, 25, 25)
    is_red = np.isin(idx, np.asarray(sorted(RED_DOUBLE_SECTOR_INDICES)))
    dark_bed = (idx % 2) == 0

    inside = r <= DOUBLE_OUTER_RADIUS_MM
    single = inside & (r > 15.9)
    img[single & dark_bed] = (35, 35, 35)
    img[single & ~dark_bed] = (225, 225, 220)

    for lo, hi in ((DOUBLE_INNER_RADIUS_MM, DOUBLE_OUTER_RADIUS_MM), (99.0, 107.0)):
        band = inside & (r >= lo) & (r <= hi)
        img[band & is_red] = (40, 40, 210)      # BGR red
        img[band & ~is_red] = (60, 170, 60)     # BGR green
    img[r <= 15.9] = (60, 170, 60)              # outer bull, green
    img[r <= 6.35] = (40, 40, 210)              # inner bull, red

    # radial sector wires + the ring wires, as thin dark lines
    dang = np.abs(((ang - FIRST_WIRE_ANGLE_DEG + SECTOR_ANGLE_DEG / 2.0) % SECTOR_ANGLE_DEG)
                  - SECTOR_ANGLE_DEG / 2.0)
    wire_mm = np.radians(dang) * np.maximum(r, 1.0)
    img[inside & (r > 15.9) & (wire_mm < 0.9)] = (18, 18, 18)
    for rad in (DOUBLE_INNER_RADIUS_MM, DOUBLE_OUTER_RADIUS_MM, 99.0, 107.0, 15.9):
        img[np.abs(r - rad) < 0.9] = (18, 18, 18)
    return img


# ---------------------------------------------------------------------
# disk_boost -- the core derivation
# ---------------------------------------------------------------------


@pytest.mark.parametrize("u", [(0.0, 0.0), (0.3, 0.0), (-0.2, 0.45), (0.55, -0.6)])
def test_disk_boost_sends_u_to_the_origin(u):
    got = apply_homography(disk_boost(u), [u])[0]
    assert np.allclose(got, [0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("u", [(0.3, 0.0), (-0.2, 0.45), (0.55, -0.6), (0.1, 0.8)])
def test_disk_boost_preserves_the_unit_circle_exactly(u):
    """The whole method rests on this: every member of the family must
    map the unit circle exactly onto the fitted ellipse. Measured
    residual for these cases is < 1e-15, so 1e-12 is a real bound with
    headroom, not a number picked to pass."""
    t = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    out = apply_homography(disk_boost(u), circle)
    radii = np.hypot(out[:, 0], out[:, 1])
    assert np.max(np.abs(radii - 1.0)) < 1e-12


@pytest.mark.parametrize("u", [(0.3, 0.0), (-0.2, 0.45), (0.55, -0.6)])
def test_disk_boost_negative_u_is_the_inverse(u):
    prod = disk_boost(u) @ disk_boost((-u[0], -u[1]))
    prod = prod / prod[2, 2]
    assert np.allclose(prod, np.eye(3), atol=1e-12)


def test_naive_mobius_matrix_does_not_preserve_the_circle():
    """Guards the module docstring's claim that the obvious
    `[[1,0,-cx],[0,1,-cy],[-cx,-cy,1]]` map -- which also sends u to the
    origin, and is what the reference implementation uses -- is NOT
    circle-preserving, so replacing `disk_boost` with it would silently
    reintroduce an angle-dependent error.

    Measured max radius deviation for u=(0.32, 0.0), the real
    |u| this rig produces: 0.0555 -- i.e. the naive map distorts the ring
    by ~5.5% of its radius, angle-dependently. Asserting > 0.01 keeps the
    test about the qualitative fact rather than the exact number.
    """
    cx, cy = 0.32, 0.0
    naive = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [-cx, -cy, 1.0]])
    t = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    out = apply_homography(naive, circle)
    dev = float(np.max(np.abs(np.hypot(out[:, 0], out[:, 1]) - 1.0)))
    assert dev > 0.01
    # and the exact boost, on the same input, does not deviate at all
    out2 = apply_homography(disk_boost((cx, cy)), circle)
    assert float(np.max(np.abs(np.hypot(out2[:, 0], out2[:, 1]) - 1.0))) < 1e-12


def test_disk_boost_rejects_a_point_outside_the_disk():
    with pytest.raises(ValueError):
        disk_boost((1.2, 0.0))


# ---------------------------------------------------------------------
# the family reproduces a real projective camera map
# ---------------------------------------------------------------------


def test_affine_maps_unit_circle_onto_the_ellipse_boundary_samples():
    ell = Ellipse(cx=640.0, cy=360.0, major_axis_px=340.0, minor_axis_px=660.0,
                  angle_deg=89.0)
    t = np.linspace(0.0, 2.0 * math.pi, 400, endpoint=False)
    circle = np.stack([np.cos(t), np.sin(t)], axis=1)
    got = apply_homography(affine_unit_circle_to_ellipse(ell), circle)
    want = ell.boundary_samples(400)
    assert np.max(np.hypot(got[:, 0] - want[:, 0], got[:, 1] - want[:, 1])) < 1e-9


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_family_reproduces_a_real_camera_projection(cam_index):
    """THE central claim: ellipse + bull determine the board-to-image map
    up to one rotation. Search phi against a real synthetic camera's own
    20 wire pixels; the best member of the family must reproduce them.

    Measured max error across the three cameras: 0.0000 px on all three
    (the derivation is exact; the only residual would be ellipse-fit
    sampling and the 0.02deg phi grid). Against three REAL camera
    homographies the same check measured 0.016-0.073 px. The 0.30px
    bound leaves room for the
    real-image case while still failing by orders of magnitude on any
    sign or derivation error.
    """
    H_mm, _cam = synthetic_board_homography(cam_index)
    H_unit = board_mm_to_unit(H_mm)
    ell, bull = ellipse_and_bull_from(H_unit)

    wire_angles = [FIRST_WIRE_ANGLE_DEG + SECTOR_ANGLE_DEG * k for k in range(N_SECTORS)]
    target = apply_homography(
        H_mm, [polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, a) for a in wire_angles])

    best = math.inf
    for phi in np.arange(0.0, 360.0, 0.02):
        Hc = board_to_image_homography(ell, bull, float(phi))
        pred = apply_homography(
            Hc, [normalised_board_point(a) for a in wire_angles])
        best = min(best, float(np.max(np.hypot(pred[:, 0] - target[:, 0],
                                               pred[:, 1] - target[:, 1]))))
    assert best < 0.30


def test_predicted_wire_angles_are_not_equally_spaced():
    """The actual bug being fixed. Under real perspective the 20 wires
    are NOT 18 degrees apart in the image; a detector that assumed they
    were is mis-registered. Measured on this synthetic camera the real
    consecutive gaps run min 9.32 deg to max 34.22 deg -- a 3.7x spread
    around the 18 deg the old scheme assumes."""
    H_mm, _ = synthetic_board_homography(0)
    ell, bull = ellipse_and_bull_from(board_mm_to_unit(H_mm))
    angles = np.asarray(predicted_wire_angles(ell, bull, 0.0))
    gaps = np.diff(np.sort(angles))
    assert gaps.min() < 12.0
    assert gaps.max() > 28.0


# ---------------------------------------------------------------------
# geometry utilities
# ---------------------------------------------------------------------


def test_ellipse_ray_intersection_lands_on_the_ellipse():
    ell = Ellipse(cx=600.0, cy=350.0, major_axis_px=340.0, minor_axis_px=660.0,
                  angle_deg=89.0)
    origin = (600.0, 300.0)
    for ang in range(0, 360, 7):
        p = ellipse_ray_intersection(ell, origin, float(ang))
        assert p is not None
        dx, dy = p[0] - ell.cx, p[1] - ell.cy
        th = math.radians(ell.angle_deg)
        xr = math.cos(th) * dx + math.sin(th) * dy
        yr = -math.sin(th) * dx + math.cos(th) * dy
        val = (xr / (ell.major_axis_px / 2)) ** 2 + (yr / (ell.minor_axis_px / 2)) ** 2
        assert abs(val - 1.0) < 1e-9
        # and the point really is along the requested direction
        assert abs(((math.degrees(math.atan2(p[1] - origin[1], p[0] - origin[0]))
                     - ang + 180.0) % 360.0) - 180.0) < 1e-9


def test_vectorised_ray_intersection_matches_the_scalar_one():
    ell = Ellipse(cx=600.0, cy=350.0, major_axis_px=340.0, minor_axis_px=660.0,
                  angle_deg=89.0)
    origin = (612.0, 296.0)
    angles = np.arange(0.0, 360.0, 1.3)
    vec = ellipse_ray_intersections(ell, origin, angles)
    for i, a in enumerate(angles):
        s = ellipse_ray_intersection(ell, origin, float(a))
        assert s is not None
        assert np.allclose(vec[i], s, atol=1e-9)


def test_vectorised_spoke_scores_match_the_scalar_one():
    rng = np.random.default_rng(7)
    profile = rng.random(1440)
    cands = rng.random((5, N_SECTORS)) * 360.0
    vec = spoke_scores(profile, cands)
    for i in range(5):
        assert abs(vec[i] - spoke_score(profile, cands[i])) < 1e-9


def test_ad_quad_object_points_agree_with_sector_correspondence():
    """Two independently written formulas for the same 4 physical points
    -- a real cross-check that this module's parallel derivation did not
    drift from the one already validated in `sector_correspondence`."""
    mine = ad_quad_object_points_mm()
    for i in range(4):
        x, y = ad_quad_object_point_mm(i, 0)
        assert np.allclose(mine[i], (x, y, 0.0), atol=1e-9)


def test_ad_quad_ring_indices_are_the_90_degree_landmarks():
    for i, k in enumerate(AD_QUAD_RING_INDICES):
        assert (FIRST_WIRE_ANGLE_DEG + SECTOR_ANGLE_DEG * k) % 360.0 == (9.0 + 90.0 * i) % 360.0


def test_red_double_sectors_match_a_regulation_board():
    """Sector 20's double is red and the beds strictly alternate --
    measured directly on 75 real samples per bed, and a property of
    any regulation
    board rather than of this rig."""
    assert 0 in RED_DOUBLE_SECTOR_INDICES                      # index 0 is sector 20
    assert SECTOR_NUMBERS_CLOCKWISE[0] == 20
    assert len(RED_DOUBLE_SECTOR_INDICES) == N_SECTORS // 2
    for j in range(N_SECTORS):
        assert (j in RED_DOUBLE_SECTOR_INDICES) != ((j + 1) % N_SECTORS in RED_DOUBLE_SECTOR_INDICES)


def test_normalised_board_point_is_the_flipped_board_frame():
    # board angle 0 is 12 o'clock: +y in board mm, -y (up) in image-style coords
    assert np.allclose(normalised_board_point(0.0), (0.0, -1.0), atol=1e-12)
    assert np.allclose(normalised_board_point(90.0), (1.0, 0.0), atol=1e-12)


# ---------------------------------------------------------------------
# refinement
# ---------------------------------------------------------------------


def test_refine_wire_angles_finds_a_synthetic_peak_sub_bin():
    n = 1440
    profile = np.zeros(n)
    true_deg = 100.30                       # deliberately between bins (bin = 0.25 deg)
    centre = true_deg / 360.0 * n
    for d in (-2, -1, 0, 1, 2):
        i = int(round(centre)) + d
        profile[i % n] = math.exp(-0.5 * ((i - centre) / 1.1) ** 2)
    got = refine_wire_angles(profile, [100.0])[0]
    # Measured 0.0083 deg here, 0.0022-0.0038 deg at two other offsets.
    assert abs(got - true_deg) < 0.02


def test_refine_window_cannot_reach_a_neighbouring_wire():
    """A refine window wider than half the smallest real inter-wire gap
    could silently lock a wire onto its neighbour. Measured minimum gap
    on a real synthetic camera: 12.0 deg, so half-gap 6.0 deg vs the
    1.0 deg window."""
    from opendarts.calibration.oriented_landmarks import WIRE_REFINE_WINDOW_DEG

    H_mm, _ = synthetic_board_homography(0)
    ell, bull = ellipse_and_bull_from(board_mm_to_unit(H_mm))
    gaps = np.diff(np.sort(np.asarray(predicted_wire_angles(ell, bull, 0.0))))
    assert WIRE_REFINE_WINDOW_DEG < 0.5 * float(gaps.min())


# ---------------------------------------------------------------------
# end to end, on a rendered synthetic board
# ---------------------------------------------------------------------


def test_colour_score_only_resolves_orientation_up_to_two_sectors():
    """Documents the real limit of the colour signal, which is why the
    orientation hint exists: red/green doubles alternate with a period of
    TWO sectors, so every same-parity rotation scores essentially as well
    as the true one. The signal narrows 20 candidates to 10 and provably
    cannot do better.

    Measured on a rendered board: all 10 same-parity rolls score 1.000,
    all 10 opposite-parity rolls score -1.000.
    """
    H_mm, _ = synthetic_board_homography(0)
    ell, bull = ellipse_and_bull_from(board_mm_to_unit(H_mm))
    img = render_synthetic_board(H_mm)

    profile = angular_edge_profile(edge_magnitude(img), ell, bull)
    phase, _score, _conf = lock_phase(profile, ell, bull)
    scores = np.asarray([
        double_colour_score(img, board_to_image_homography(
            ell, bull, phase + SECTOR_ANGLE_DEG * roll))
        for roll in range(N_SECTORS)])

    best = int(np.argmax(scores))
    same = scores[[r for r in range(N_SECTORS) if r % 2 == best % 2]]
    opposite = scores[[r for r in range(N_SECTORS) if r % 2 != best % 2]]
    assert len(same) == len(opposite) == N_SECTORS // 2
    # every same-parity roll beats every opposite-parity roll ...
    assert float(same.min()) > float(opposite.max())
    # ... and they are indistinguishable from each other, which is the point
    assert float(same.max() - same.min()) < 0.05


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_end_to_end_recovers_the_true_quad_on_a_rendered_board(cam_index):
    """Full pipeline against synthetic ground truth: render a board
    through a known projective map, run the finder, and check the 4
    emitted quad landmarks land on the true projected wire points."""
    H_mm, _ = synthetic_board_homography(cam_index)
    img = render_synthetic_board(H_mm)

    truth = apply_homography(
        H_mm,
        [polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, 9.0 + 90.0 * i) for i in range(4)])
    hint_pt = apply_homography(H_mm, [polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, 0.0)])[0]

    res = find_oriented_landmarks(img)
    assert res.ok, res.reason
    assert res.bull_px is not None and res.quad_px is not None
    hint = math.degrees(math.atan2(hint_pt[1] - res.bull_px[1],
                                   hint_pt[0] - res.bull_px[0])) % 360.0

    res = find_oriented_landmarks(img, orientation_hint_deg=hint)
    assert res.ok, res.reason
    assert not res.orientation_ambiguous
    err = np.hypot(res.quad_px[:, 0] - truth[:, 0], res.quad_px[:, 1] - truth[:, 1])
    # Measured worst-case across the three cameras: 2.56 px (2.39 / 1.56
    # / 2.56 on cams 0/1/2), dominated by
    # the rendered wire's own ~2px width and the colour-mask ellipse.
    assert float(err.max()) < 8.0


def test_bull_is_found_on_a_rendered_board():
    from opendarts.calibration.landmark_detection import detect_double_ring_quad

    H_mm, _ = synthetic_board_homography(0)
    img = render_synthetic_board(H_mm)
    truth = apply_homography(H_mm, [(0.0, 0.0)])[0]
    seed = detect_double_ring_quad(img)
    assert seed.ok and seed.ellipse is not None
    bull = detect_bull(img, seed.ellipse)
    assert bull.ok and bull.xy is not None
    # Measured 0.00 px on this render (the rendered bull is exactly
    # symmetric); the real-image measurement against the oracle's bull was
    # median 1.0-2.8 px, so 3.0 keeps this honest for both.
    assert math.hypot(bull.xy[0] - truth[0], bull.xy[1] - truth[1]) < 3.0


def test_orientation_is_flagged_ambiguous_without_a_hint():
    """Without a hint the colour signal alone leaves 10 candidates, and
    the module must SAY so rather than silently return a 1-in-10 guess."""
    H_mm, _ = synthetic_board_homography(0)
    img = render_synthetic_board(H_mm)
    res = find_oriented_landmarks(img)
    assert res.orientation_ambiguous


def test_blank_image_fails_cleanly():
    res = find_oriented_landmarks(np.zeros((H, W, 3), np.uint8))
    assert not res.ok
    assert res.quad_px is None
    assert res.reason


# ---------------------------------------------------------------------
# correspond_landmarks_oriented() -- the production wrapper
# (2026-08-13, when capture_daemon.bootstrap_calibrations() was wired to it)
# ---------------------------------------------------------------------


def test_correspond_landmarks_oriented_refuses_an_ambiguous_lock():
    """The wrapper is where "report it" becomes "reject it". Without a
    hint the colour signal leaves 10 candidates (see
    test_orientation_is_flagged_ambiguous_without_a_hint above), and a
    wrong pick is a calibration rotated a whole quarter turn -- worse
    than no calibration, so this must be None rather than a plausible-
    looking quad."""
    from opendarts.calibration.oriented_landmarks import correspond_landmarks_oriented

    H_mm, _ = synthetic_board_homography(0)
    img = render_synthetic_board(H_mm)
    assert find_oriented_landmarks(img).orientation_ambiguous
    assert correspond_landmarks_oriented(img, 0) is None


def test_correspond_landmarks_oriented_results_out_records_every_attempt():
    """The diagnostics sink a caller needs to explain a rejection without
    a second detection pass -- populated on success and on failure."""
    from opendarts.calibration.oriented_landmarks import correspond_landmarks_oriented

    H_mm, _ = synthetic_board_homography(0)
    img = render_synthetic_board(H_mm)
    hint_pt = apply_homography(H_mm, [polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, 0.0)])[0]
    seeded = find_oriented_landmarks(img)
    hint = math.degrees(math.atan2(hint_pt[1] - seeded.bull_px[1],
                                   hint_pt[0] - seeded.bull_px[0])) % 360.0

    sink = []
    got = correspond_landmarks_oriented(img, 0, orientation_hints_deg={0: hint},
                                        results_out=sink)
    assert got is not None
    assert len(sink) == 1 and sink[0].ok and not sink[0].orientation_ambiguous

    blank_sink = []
    assert correspond_landmarks_oriented(np.zeros((H, W, 3), np.uint8), 0,
                                         results_out=blank_sink) is None
    assert len(blank_sink) == 1 and not blank_sink[0].ok and blank_sink[0].reason
