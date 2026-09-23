"""Synthetic known-truth for Talos's 3D math: 2D shaft line -> plane
(camera center + line) -> planes intersect in a 3D dart axis -> axis ∩ Z=0.

No oracle labels. Cameras from tests.support.synthetic
(make_ring_camera / make_camera_matrix), wrapped as CameraCalibration.
Tolerances are from a printed measurement on this exact scene, not a
round-number guess.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.support.synthetic import (
    make_camera_matrix,
    make_ring_camera,
    project_points,
)
from opendarts.engines.talos.consensus import (
    FAR_CAP_RADIUS_MM,
    OUTWARD_CAP_MM,
    OUTWARD_EDGE_WINDOW_PX,
    _lock_unanimous_centerline_sector,
    _lock_cap_walked_radial,
    _lock_centerline_ring,
    _lock_axis_ring_lonely_cl,
    _lock_single_cl_double_to_outer,
    _lock_split_centerline_mean,
    _lock_insector_ring_majority_2of3,
    CL_RING_OUTWARD_MM,
    SPLIT_MAX_RADIUS_DISAGREE_MM,
    INSECTOR_RING_MAJORITY_MIN_LEFTOVER_DR_MM,
    _outward_edge_pixel,
    _pair_if_sector_compromise,
    _rescue_outside_from_centerlines,
    apply_centerline_overrides,
    drop_far_cap_rays,
    one_cam_z0,
)
from opendarts.engines.talos.plane_geometry import (
    BOARD_HIT_Z_MM,
    _pixel_board_xy,
    intersect_planes_with_board,
    plane_from_image_line,
    slide_point_along_dart_axis,
    snap_xy_to_plane_line,
)
from opendarts.geometry.board import (
    polar_to_xy_mm,
    sector_center_angle_deg,
    sector_ring_for_point,
)
from opendarts.pipeline import CameraCalibration, ScoreResult


# Measured on this file's two noiseless scenes:
#   dart (40, 25) 17.35deg tilt: error_xy_mm = 1.065814103640e-13
#   dart (-70, 40) 30deg tilt:   error_xy_mm = 4.973799150321e-14
# 1e-11 mm is ~100x the larger measured value -- tight enough that a
# millimetre-scale (or even micron-scale) regression fails, loose enough
# for float64 projectPoints / SVD noise across machines.
_MAX_RECOVERY_ERROR_MM = 1e-11

# Measured on this file's noiseless cap-centroid scene:
# centroid_error_px = 0.0
# (exact mean of the same three pixel coords). 1e-12 px is machine-eps
# scale, ~2e8x smaller than the 0.207px gap to the max-r corner.
_MAX_CAP_CENTROID_ERROR_PX = 1e-12


def _as_calibration(cam) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=cam.camera_matrix,
        dist_coeffs=cam.dist_coeffs,
        rvec=cam.rvec,
        tvec=cam.tvec,
        landmark_spread_ok=True,
    )


def _project_line_px(cam, p_a, p_b):
    pts = project_points(cam, np.stack([p_a, p_b]))
    p1 = (float(pts[0, 0]), float(pts[0, 1]))
    p2 = (float(pts[1, 0]), float(pts[1, 1]))
    return p1, p2


def _three_ring_cameras():
    K = make_camera_matrix(fov_deg=90.0)
    syn = [make_ring_camera(i, n_cameras=3, camera_matrix=K) for i in range(3)]
    calibrations = {i: _as_calibration(cam) for i, cam in enumerate(syn)}
    return syn, calibrations


def _project_board_pixel(cam, calib, xy_mm):
    pts = project_points(
        cam, np.array([[xy_mm[0], xy_mm[1], 0.0]], dtype=np.float64)
    )
    pixel = (float(pts[0, 0]), float(pts[0, 1]))
    recovered = _pixel_board_xy(pixel, calib)
    assert recovered is not None
    return pixel, recovered


def _fake_ray_scored(sector, ring, n_cameras_used=3, ok=True) -> ScoreResult:
    return ScoreResult(
        ok=ok,
        sector=sector,
        ring=ring,
        board_xy_mm=(0.0, 0.0),
        triangulation=None,
        n_cameras_used=n_cameras_used,
    )


@pytest.mark.parametrize(
    "true_hit_xy, p_shaft",
    [
        # Tilted ~17.35 deg from vertical, board entry in +X+Y.
        ((40.0, 25.0), np.array([20.0, 10.0, 80.0], dtype=np.float64)),
        # 30 deg from vertical, board entry in -X+Y.
        (
            (-70.0, 40.0),
            np.array([-70.0, 40.0, 0.0], dtype=np.float64)
            + 100.0 * np.array([0.5, 0.0, np.sqrt(3.0) / 2.0]),
        ),
    ],
)
def test_intersect_planes_recovers_known_tilted_dart_axis(true_hit_xy, p_shaft):
    true_hit = np.array([true_hit_xy[0], true_hit_xy[1], 0.0], dtype=np.float64)
    p_shaft = np.asarray(p_shaft, dtype=np.float64)
    axis = p_shaft - true_hit
    axis = axis / np.linalg.norm(axis)

    K = make_camera_matrix(fov_deg=90.0)
    syn_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=K) for i in range(3)]
    calibrations = {i: _as_calibration(cam) for i, cam in enumerate(syn_cams)}

    planes = []
    for i, cam in enumerate(syn_cams):
        p1, p2 = _project_line_px(cam, true_hit, p_shaft)
        plane = plane_from_image_line(p1, p2, calibrations[i])
        assert plane is not None, f"cam{i} plane_from_image_line returned None"
        n, c = plane
        # True board-entry must already lie in this camera's plane.
        plane_res = abs(float(np.dot(n, true_hit) - c))
        print(f"cam{i} true_hit plane residual mm={plane_res:.12e}")
        assert plane_res < _MAX_RECOVERY_ERROR_MM
        planes.append((n, c))

    hit = intersect_planes_with_board(planes)
    assert hit is not None
    xy, geom = hit
    err = float(np.linalg.norm(np.array([xy[0], xy[1]], dtype=np.float64) - true_hit[:2]))
    print(f"recovered xy={xy} true={true_hit_xy} error_xy_mm={err:.12e}")
    print(
        f"tilt_deg recovered={geom.get('tilt_deg')} "
        f"direction={geom.get('direction')}"
    )
    assert err < _MAX_RECOVERY_ERROR_MM, (
        f"board-entry recovery error {err:.12e} mm exceeds measured-bound "
        f"{_MAX_RECOVERY_ERROR_MM} mm (measured noiseless ~1e-13 mm)"
    )
    assert xy[0] == pytest.approx(true_hit_xy[0], abs=_MAX_RECOVERY_ERROR_MM)
    assert xy[1] == pytest.approx(true_hit_xy[1], abs=_MAX_RECOVERY_ERROR_MM)
    recovered_dir = np.array(geom["direction"], dtype=np.float64)
    dir_err = float(np.linalg.norm(recovered_dir - axis))
    print(f"direction error={dir_err:.12e}")
    # Measured direction error ~6e-16; 1e-12 is still ~machine-eps scale.
    assert dir_err < 1e-12
    assert geom["hypothesis"] == "all"
    assert geom["planes_used_indices"] == [0, 1, 2]


def test_two_planes_also_recover_the_known_board_entry():
    """2-plane path (n1 × n2) is the engine's minimum; must recover the
    same known hit, not only the 3-plane SVD path."""
    true_hit = np.array([40.0, 25.0, 0.0], dtype=np.float64)
    p_shaft = np.array([20.0, 10.0, 80.0], dtype=np.float64)
    K = make_camera_matrix(fov_deg=90.0)
    syn_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=K) for i in range(2)]
    planes = []
    for i, cam in enumerate(syn_cams):
        p1, p2 = _project_line_px(cam, true_hit, p_shaft)
        plane = plane_from_image_line(p1, p2, _as_calibration(cam))
        assert plane is not None
        planes.append(plane)
    hit = intersect_planes_with_board(planes)
    assert hit is not None
    xy, geom = hit
    err = float(np.linalg.norm(np.array([xy[0], xy[1]]) - true_hit[:2]))
    print(f"2-plane recovered xy={xy} error_xy_mm={err:.12e}")
    assert err < _MAX_RECOVERY_ERROR_MM, f"2-plane error {err:.12e} mm"
    assert geom["hypothesis"] == "all"
    assert geom["planes_used_indices"] == [0, 1]


def test_outward_edge_pixel_picks_max_board_radius_within_tip_window():
    """Singleton cap: these three collinear board points are 5mm apart, so
    OUTWARD_CAP_MM=1.0 contains only the max-r pixel and the centroid of
    that one-pixel cap is the argmax-r corner."""
    K = make_camera_matrix(fov_deg=90.0)
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=K)
    calib = _as_calibration(cam)

    # These are deliberately collinear board-plane candidates: their
    # projected pixel distances are all inside the 12px tip neighborhood,
    # while their board radii are distinct and known.
    board_points = np.array(
        [
            [90.0, 0.0, 0.0],
            [95.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pixels = project_points(cam, board_points)
    tip_px = (float(pixels[0, 0]), float(pixels[0, 1]))

    radii_mm = []
    distances_px = []
    for pixel in pixels:
        xy = _pixel_board_xy(pixel, calib)
        assert xy is not None
        radii_mm.append(float(np.hypot(xy[0], xy[1])))
        distances_px.append(float(np.linalg.norm(pixel - pixels[0])))

    print(f"outward-edge candidate radii_mm={radii_mm}")
    print(f"outward-edge tip distances_px={distances_px}")
    assert max(distances_px) < OUTWARD_EDGE_WINDOW_PX
    assert radii_mm[0] < radii_mm[-1]

    selected = _outward_edge_pixel(pixels, tip_px, calib)
    assert selected is not None
    expected = (float(pixels[-1, 0]), float(pixels[-1, 1]))
    selected_xy = _pixel_board_xy(selected, calib)
    assert selected_xy is not None
    selected_radius_mm = float(np.hypot(selected_xy[0], selected_xy[1]))
    measured_radius_gap_mm = radii_mm[-1] - radii_mm[0]
    measured_selection_error_mm = abs(selected_radius_mm - max(radii_mm))
    print(
        f"outward-edge selected_radius_mm={selected_radius_mm:.12e} "
        f"max_minus_min_gap_mm={measured_radius_gap_mm:.12e} "
        f"selection_error_mm={measured_selection_error_mm:.12e}"
    )

    # The noiseless synthetic projection returned the exact max-radius
    # candidate (measured selection error printed above), so this discrete
    # choice needs no broad geometric tolerance.
    assert selected == expected


def test_outward_edge_pixel_returns_cap_centroid_not_max_r_corner():
    """When several near-tip pixels sit within OUTWARD_CAP_MM of max
    board-r, the 2D observation is their image-space centroid, not the
    single max-r corner."""
    K = make_camera_matrix(fov_deg=90.0)
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=K)
    calib = _as_calibration(cam)

    # Inward point plus three high-r points: two split tangentially so
    # the cap centroid cannot collapse to the max-r pixel. Recovered
    # radii: 90.0, 99.680,
    # 99.680, 100.0 -- the last three are inside the 1.0mm cap.
    board_points = np.array(
        [
            [90.0, 0.0, 0.0],
            [99.6, -4.0, 0.0],
            [99.6, 4.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    pixels = project_points(cam, board_points)
    tip_px = (float(pixels[0, 0]), float(pixels[0, 1]))

    radii_mm = []
    distances_px = []
    for pixel in pixels:
        xy = _pixel_board_xy(pixel, calib)
        assert xy is not None
        radii_mm.append(float(np.hypot(xy[0], xy[1])))
        distances_px.append(float(np.linalg.norm(pixel - pixels[0])))

    print(f"cap-centroid candidate radii_mm={radii_mm}")
    print(f"cap-centroid tip distances_px={distances_px}")
    assert max(distances_px) < OUTWARD_EDGE_WINDOW_PX
    max_r = max(radii_mm)
    cap_indices = [
        i for i, r in enumerate(radii_mm) if r >= max_r - OUTWARD_CAP_MM
    ]
    print(f"cap-centroid in_cap_indices={cap_indices} max_r_mm={max_r:.12e}")
    assert 0 not in cap_indices, "inward point must sit outside the 1mm cap"
    assert len(cap_indices) >= 2

    cap_px = pixels[cap_indices]
    expected_centroid = (
        float(np.mean(cap_px[:, 0])),
        float(np.mean(cap_px[:, 1])),
    )
    max_r_px = (
        float(pixels[int(np.argmax(radii_mm)), 0]),
        float(pixels[int(np.argmax(radii_mm)), 1]),
    )

    selected = _outward_edge_pixel(pixels, tip_px, calib)
    assert selected is not None
    centroid_error_px = float(
        np.hypot(
            selected[0] - expected_centroid[0],
            selected[1] - expected_centroid[1],
        )
    )
    dist_to_max_r_px = float(
        np.hypot(selected[0] - max_r_px[0], selected[1] - max_r_px[1])
    )
    print(f"cap-centroid selected={selected}")
    print(f"cap-centroid expected={expected_centroid}")
    print(f"cap-centroid max_r_px={max_r_px}")
    print(f"cap-centroid error_px={centroid_error_px:.12e}")
    print(f"cap-centroid dist_to_max_r_px={dist_to_max_r_px:.12e}")

    assert centroid_error_px < _MAX_CAP_CENTROID_ERROR_PX, (
        f"cap centroid error {centroid_error_px:.12e} px exceeds "
        f"{_MAX_CAP_CENTROID_ERROR_PX} px (measured noiseless 0.0 px)"
    )
    # Measured dist_to_max_r_px = 2.068490220775e-01; a regression to
    # argmax-r would drive this to 0.
    assert dist_to_max_r_px > 0.1
    assert selected != max_r_px


def test_pair_if_sector_compromise_uses_mean_of_agreeing_cameras_ray_z0():
    """True 2-of-3 sector compromise: 3-ray bed is a pie no camera voted
    for. Returned board_xy_mm is the mean of the agreeing cameras'
    ray∩Z=0, not a pair triangulation."""
    syn, calibrations = _three_ring_cameras()
    p20_a = polar_to_xy_mm(50.0, 0.0)
    p20_b = polar_to_xy_mm(70.0, 0.0)
    p1 = polar_to_xy_mm(50.0, 18.0)
    print(
        f"compromise true beds a={sector_ring_for_point(*p20_a)} "
        f"b={sector_ring_for_point(*p20_b)} "
        f"singleton={sector_ring_for_point(*p1)}"
    )
    assert sector_ring_for_point(*p20_a) == ("20", "single_inner")
    assert sector_ring_for_point(*p20_b) == ("20", "single_inner")
    assert sector_ring_for_point(*p1) == ("1", "single_inner")

    ray_pixels = {}
    recovered = {}
    beds = {}
    for cam, xy in ((0, p20_a), (1, p20_b), (2, p1)):
        pixel, rec = _project_board_pixel(syn[cam], calibrations[cam], xy)
        ray_pixels[cam] = pixel
        recovered[cam] = rec
        beds[cam] = sector_ring_for_point(rec[0], rec[1])
        rec_err = float(np.hypot(rec[0] - xy[0], rec[1] - xy[1]))
        print(
            f"compromise cam{cam} rec={rec} bed={beds[cam]} "
            f"project_unproject_err_mm={rec_err:.12e}"
        )
        assert rec_err < _MAX_RECOVERY_ERROR_MM
    assert beds[0] == beds[1] == ("20", "single_inner")
    assert beds[2] == ("1", "single_inner")

    holders = [cam for cam, bed in beds.items() if bed == ("20", "single_inner")]
    xys = [recovered[cam] for cam in holders]
    expected_mean = (
        sum(p[0] for p in xys) / len(xys),
        sum(p[1] for p in xys) / len(xys),
    )
    print(f"compromise holders={holders} expected_mean={expected_mean}")

    # 3-ray bed is a different sector that neither the pair nor the
    # singleton reported -- true compromise, not "followed the singleton".
    ray_scored = _fake_ray_scored("18", "treble")
    result = _pair_if_sector_compromise(ray_pixels, calibrations, ray_scored)
    assert result is not None
    mean_error_mm = float(
        np.hypot(
            result.board_xy_mm[0] - expected_mean[0],
            result.board_xy_mm[1] - expected_mean[1],
        )
    )
    print(
        f"compromise result_xy={result.board_xy_mm} "
        f"result_bed={(result.sector, result.ring)} "
        f"mean_error_mm={mean_error_mm:.12e} "
        f"n_cameras_used={result.n_cameras_used} "
        f"cameras_used={result.cameras_used}"
    )
    assert mean_error_mm < _MAX_RECOVERY_ERROR_MM, (
        f"agreeing-pair mean error {mean_error_mm:.12e} mm exceeds "
        f"{_MAX_RECOVERY_ERROR_MM} mm (measured noiseless 0.0 mm)"
    )
    assert result.ok is True
    assert (result.sector, result.ring) == ("20", "single_inner")
    assert result.triangulation is None
    assert result.n_cameras_used == 2
    assert set(result.cameras_used) == {0, 1}
    assert "sector compromise" in result.reason
    # Mean of the pair, not the singleton's Z=0 and not the fake 3-ray
    # (0, 0) we stuffed into ScoreResult.
    singleton_err_mm = float(
        np.hypot(
            result.board_xy_mm[0] - recovered[2][0],
            result.board_xy_mm[1] - recovered[2][1],
        )
    )
    print(f"compromise dist_to_singleton_mm={singleton_err_mm:.12e}")
    assert singleton_err_mm > 10.0
    assert result.board_xy_mm != (0.0, 0.0)


@pytest.mark.parametrize(
    "case",
    [
        "same_sector_different_ring",
        "majority_bull",
        "majority_outside",
        "three_ray_equals_majority",
        "three_ray_equals_singleton",
    ],
)
def test_pair_if_sector_compromise_does_not_fire(case):
    syn, calibrations = _three_ring_cameras()
    p20_inner_a = polar_to_xy_mm(50.0, 0.0)
    p20_inner_b = polar_to_xy_mm(70.0, 0.0)
    p20_treble = polar_to_xy_mm(103.0, 0.0)
    p1_inner = polar_to_xy_mm(50.0, 18.0)
    p_bull = (0.0, 3.0)
    p_outside = (0.0, 200.0)

    def project_map(xy_by_cam):
        pixels = {}
        beds = {}
        for cam, xy in xy_by_cam.items():
            pixel, rec = _project_board_pixel(
                syn[cam], calibrations[cam], xy
            )
            pixels[cam] = pixel
            beds[cam] = sector_ring_for_point(rec[0], rec[1])
        return pixels, beds

    if case == "same_sector_different_ring":
        ray_pixels, beds = project_map(
            {0: p20_treble, 1: p20_treble, 2: p20_inner_a}
        )
        print(f"{case} beds={beds}")
        assert beds[0] == beds[1] == ("20", "treble")
        assert beds[2] == ("20", "single_inner")
        ray_scored = _fake_ray_scored("20", "single_inner")
    elif case == "majority_bull":
        ray_pixels, beds = project_map({0: p_bull, 1: p_bull, 2: p20_inner_a})
        print(f"{case} beds={beds}")
        assert beds[0] == beds[1] == (None, "bull")
        ray_scored = _fake_ray_scored("20", "single_inner")
    elif case == "majority_outside":
        ray_pixels, beds = project_map(
            {0: p_outside, 1: p_outside, 2: p20_inner_a}
        )
        print(f"{case} beds={beds}")
        assert beds[0] == beds[1] == (None, "outside")
        ray_scored = _fake_ray_scored("20", "single_inner")
    elif case == "three_ray_equals_majority":
        ray_pixels, beds = project_map(
            {0: p20_inner_a, 1: p20_inner_b, 2: p1_inner}
        )
        print(f"{case} beds={beds}")
        assert beds[0] == beds[1] == ("20", "single_inner")
        assert beds[2] == ("1", "single_inner")
        ray_scored = _fake_ray_scored("20", "single_inner")
    elif case == "three_ray_equals_singleton":
        ray_pixels, beds = project_map(
            {0: p20_inner_a, 1: p20_inner_b, 2: p1_inner}
        )
        print(f"{case} beds={beds}")
        assert beds[0] == beds[1] == ("20", "single_inner")
        assert beds[2] == ("1", "single_inner")
        ray_scored = _fake_ray_scored("1", "single_inner")
    else:
        raise AssertionError(f"unknown case {case}")

    result = _pair_if_sector_compromise(ray_pixels, calibrations, ray_scored)
    print(f"{case} result={result}")
    assert result is None


# Measured on the C/D noiseless scenes:
#   C mean_error_mm = 0.000000000000e+00
#     (exact mean of the two recovered on-board XYs)
#   D mean_error_mm = 0.000000000000e+00
#     (exact mean of the three recovered CL XYs)
# 1e-11 mm is the same bound as _MAX_RECOVERY_ERROR_MM (~100x the
# ~1e-13 mm project/unproject noise). The consensus mean itself was
# bitwise-identical to the expected mean.
_MAX_CL_MEAN_ERROR_MM = 1e-11


def _mean_xy(xys):
    return (
        sum(p[0] for p in xys) / len(xys),
        sum(p[1] for p in xys) / len(xys),
    )


def _project_cl(syn, calibrations, xy_by_cam):
    pixels = {}
    recovered = {}
    beds = {}
    for cam, xy in xy_by_cam.items():
        pixel, rec = _project_board_pixel(syn[cam], calibrations[cam], xy)
        rec_err = float(np.hypot(rec[0] - xy[0], rec[1] - xy[1]))
        bed = sector_ring_for_point(rec[0], rec[1])
        print(
            f"cam{cam} rec={rec} bed={bed} "
            f"project_unproject_err_mm={rec_err:.12e}"
        )
        assert rec_err < _MAX_RECOVERY_ERROR_MM
        pixels[cam] = pixel
        recovered[cam] = rec
        beds[cam] = bed
    return pixels, recovered, beds


def test_rescue_outside_from_centerlines_mean_of_two_onboard():
    """C: result is outside, 2 on-board CLs + 1 off-board -> mean of the two."""
    syn, calibrations = _three_ring_cameras()
    angle_8 = sector_center_angle_deg(8)
    p_on_a = polar_to_xy_mm(140.0, angle_8)
    p_on_b = polar_to_xy_mm(150.0, angle_8)
    p_off = polar_to_xy_mm(200.0, angle_8)
    print(
        f"C true beds a={sector_ring_for_point(*p_on_a)} "
        f"b={sector_ring_for_point(*p_on_b)} "
        f"off={sector_ring_for_point(*p_off)}"
    )
    assert sector_ring_for_point(*p_on_a) == ("8", "single_outer")
    assert sector_ring_for_point(*p_on_b) == ("8", "single_outer")
    assert sector_ring_for_point(*p_off) == (None, "outside")

    cl_pixels, recovered, beds = _project_cl(
        syn, calibrations, {0: p_on_a, 1: p_on_b, 2: p_off}
    )
    assert beds[0] == beds[1] == ("8", "single_outer")
    assert beds[2] == (None, "outside")

    expected_mean = _mean_xy([recovered[0], recovered[1]])
    print(f"C expected_mean={expected_mean}")

    scored = _fake_ray_scored(None, "outside")
    result = _rescue_outside_from_centerlines(cl_pixels, calibrations, scored)
    assert result is not None
    mean_error_mm = float(
        np.hypot(
            result.board_xy_mm[0] - expected_mean[0],
            result.board_xy_mm[1] - expected_mean[1],
        )
    )
    print(
        f"C result_xy={result.board_xy_mm} "
        f"result_bed={(result.sector, result.ring)} "
        f"mean_error_mm={mean_error_mm:.12e} "
        f"n_cameras_used={result.n_cameras_used} "
        f"cameras_used={result.cameras_used}"
    )
    assert mean_error_mm < _MAX_CL_MEAN_ERROR_MM, (
        f"C on-board CL mean error {mean_error_mm:.12e} mm exceeds "
        f"{_MAX_CL_MEAN_ERROR_MM} mm (measured noiseless 0.0 mm)"
    )
    assert result.ok is True
    assert result.ring != "outside"
    assert (result.sector, result.ring) == ("8", "single_outer")
    assert result.n_cameras_used == 2
    assert set(result.cameras_used) == {0, 1}
    assert "outside rescue" in result.reason

    composed = apply_centerline_overrides(cl_pixels, calibrations, scored)
    assert composed is not None
    composed_err = float(
        np.hypot(
            composed.board_xy_mm[0] - expected_mean[0],
            composed.board_xy_mm[1] - expected_mean[1],
        )
    )
    print(f"C apply_centerline_overrides mean_error_mm={composed_err:.12e}")
    assert composed_err < _MAX_CL_MEAN_ERROR_MM


def test_rescue_outside_does_not_fire_when_result_already_onboard():
    """C must not rewrite an already-on-board result, even with 2 on-board CLs."""
    syn, calibrations = _three_ring_cameras()
    angle_8 = sector_center_angle_deg(8)
    cl_pixels, _, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(140.0, angle_8),
            1: polar_to_xy_mm(150.0, angle_8),
            2: polar_to_xy_mm(200.0, angle_8),
        },
    )
    assert beds[0][1] != "outside"
    assert beds[1][1] != "outside"
    scored = _fake_ray_scored("8", "single_outer")
    result = _rescue_outside_from_centerlines(cl_pixels, calibrations, scored)
    print(f"C already-onboard result={result}")
    assert result is None
    assert apply_centerline_overrides(cl_pixels, calibrations, scored) is None


def test_rescue_outside_does_not_fire_with_only_one_onboard_cl():
    """C requires >=2 on-board centerline ray∩Z=0 hits."""
    syn, calibrations = _three_ring_cameras()
    angle_8 = sector_center_angle_deg(8)
    cl_pixels, _, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(140.0, angle_8),
            1: polar_to_xy_mm(200.0, angle_8),
            2: polar_to_xy_mm(210.0, angle_8),
        },
    )
    print(f"C one-onboard beds={beds}")
    assert beds[0][1] != "outside"
    assert beds[1] == (None, "outside")
    assert beds[2] == (None, "outside")
    scored = _fake_ray_scored(None, "outside")
    result = _rescue_outside_from_centerlines(cl_pixels, calibrations, scored)
    print(f"C only-1-onboard result={result}")
    assert result is None
    assert apply_centerline_overrides(cl_pixels, calibrations, scored) is None


def test_lock_unanimous_centerline_sector_mean_of_three():
    """D: all 3 CLs on-board in sector 12, scored sector 5 -> CL mean."""
    syn, calibrations = _three_ring_cameras()
    angle_12 = sector_center_angle_deg(12)
    p12_a = polar_to_xy_mm(50.0, angle_12)
    p12_b = polar_to_xy_mm(60.0, angle_12)
    p12_c = polar_to_xy_mm(70.0, angle_12)
    print(
        f"D true beds a={sector_ring_for_point(*p12_a)} "
        f"b={sector_ring_for_point(*p12_b)} "
        f"c={sector_ring_for_point(*p12_c)}"
    )
    assert sector_ring_for_point(*p12_a) == ("12", "single_inner")
    assert sector_ring_for_point(*p12_b) == ("12", "single_inner")
    assert sector_ring_for_point(*p12_c) == ("12", "single_inner")

    cl_pixels, recovered, beds = _project_cl(
        syn, calibrations, {0: p12_a, 1: p12_b, 2: p12_c}
    )
    assert beds[0] == beds[1] == beds[2] == ("12", "single_inner")

    expected_mean = _mean_xy([recovered[0], recovered[1], recovered[2]])
    print(f"D expected_mean={expected_mean}")

    scored = _fake_ray_scored("5", "single_inner")
    result = _lock_unanimous_centerline_sector(cl_pixels, calibrations, scored)
    assert result is not None
    mean_error_mm = float(
        np.hypot(
            result.board_xy_mm[0] - expected_mean[0],
            result.board_xy_mm[1] - expected_mean[1],
        )
    )
    print(
        f"D result_xy={result.board_xy_mm} "
        f"result_bed={(result.sector, result.ring)} "
        f"mean_error_mm={mean_error_mm:.12e} "
        f"n_cameras_used={result.n_cameras_used} "
        f"cameras_used={result.cameras_used}"
    )
    assert mean_error_mm < _MAX_CL_MEAN_ERROR_MM, (
        f"D unanimous CL mean error {mean_error_mm:.12e} mm exceeds "
        f"{_MAX_CL_MEAN_ERROR_MM} mm (measured noiseless 0.0 mm)"
    )
    assert result.ok is True
    assert result.sector == "12"
    assert (result.sector, result.ring) == ("12", "single_inner")
    assert result.n_cameras_used == 3
    assert set(result.cameras_used) == {0, 1, 2}
    assert "unanimous centerline" in result.reason

    composed = apply_centerline_overrides(cl_pixels, calibrations, scored)
    assert composed is not None
    composed_err = float(
        np.hypot(
            composed.board_xy_mm[0] - expected_mean[0],
            composed.board_xy_mm[1] - expected_mean[1],
        )
    )
    print(f"D apply_centerline_overrides mean_error_mm={composed_err:.12e}")
    assert composed_err < _MAX_CL_MEAN_ERROR_MM
    assert composed.sector == "12"


def test_lock_unanimous_centerline_does_not_fire_on_two_of_three():
    """D requires 3/3. 2 CL on 12 + 1 on 5 is the correlated-bias trap
    (throw_1786593923839: 2 CL on 8, truth 16, cap saved it)."""
    syn, calibrations = _three_ring_cameras()
    angle_12 = sector_center_angle_deg(12)
    angle_5 = sector_center_angle_deg(5)
    cl_pixels, _, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(50.0, angle_12),
            1: polar_to_xy_mm(60.0, angle_12),
            2: polar_to_xy_mm(55.0, angle_5),
        },
    )
    print(f"D 2-of-3 beds={beds}")
    assert beds[0][0] == beds[1][0] == "12"
    assert beds[2][0] == "5"
    scored = _fake_ray_scored("5", "single_inner")
    result = _lock_unanimous_centerline_sector(cl_pixels, calibrations, scored)
    print(f"D 2-of-3 result={result}")
    assert result is None
    assert apply_centerline_overrides(cl_pixels, calibrations, scored) is None


def test_lock_cap_walked_radial_cl_15_caps_10():
    """Onboard CLs all 15, onboard caps all 10, axis 15, result 10 -> CL mean.

    Pattern of throw_1786667897769. cam1 is off-board leftover.
    """
    syn, calibrations = _three_ring_cameras()
    a15 = sector_center_angle_deg(15)
    a10 = sector_center_angle_deg(10)
    cl_pixels, rec_cl, beds_cl = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(90.0, a15),
            1: polar_to_xy_mm(200.0, a15),
            2: polar_to_xy_mm(85.0, a15),
        },
    )
    cap_pixels, _rec_cap, beds_cap = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(90.0, a10),
            1: polar_to_xy_mm(200.0, a10),
            2: polar_to_xy_mm(95.0, a10),
        },
    )
    print(f"cap-walked CL beds={beds_cl} cap beds={beds_cap}")
    assert beds_cl[0][0] == beds_cl[2][0] == "15"
    assert beds_cl[1][0] is None
    assert beds_cap[0][0] == beds_cap[2][0] == "10"
    axis_xy = polar_to_xy_mm(90.0, a15)
    assert sector_ring_for_point(*axis_xy)[0] == "15"
    scored = _fake_ray_scored("10", "single_inner")
    result = _lock_cap_walked_radial(
        cl_pixels, cap_pixels, calibrations, scored, axis_xy=axis_xy,
    )
    assert result is not None
    expected = _mean_xy([rec_cl[0], rec_cl[2]])
    err = float(np.hypot(
        result.board_xy_mm[0] - expected[0],
        result.board_xy_mm[1] - expected[1],
    ))
    print(f"cap-walked result={result.sector, result.ring} err={err:.12e}")
    assert err < _MAX_CL_MEAN_ERROR_MM
    assert result.sector == "15"
    assert "cap walked" in result.reason
    composed = apply_centerline_overrides(
        cl_pixels, calibrations, scored,
        cap_pixels=cap_pixels, axis_xy=axis_xy,
    )
    assert composed is not None
    assert composed.sector == "15"


def test_lock_cap_walked_radial_does_not_fire_when_caps_disagree():
    """D-trap shape: CLs both 8, caps 16 and 8 (not unanimous) -> no lock."""
    syn, calibrations = _three_ring_cameras()
    a8 = sector_center_angle_deg(8)
    a16 = sector_center_angle_deg(16)
    cl_pixels, _, beds_cl = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(200.0, a8),
            1: polar_to_xy_mm(85.0, a8),
            2: polar_to_xy_mm(90.0, a8),
        },
    )
    cap_pixels, _, beds_cap = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(200.0, a16),
            1: polar_to_xy_mm(90.0, a16),
            2: polar_to_xy_mm(90.0, a8),
        },
    )
    print(f"trap CL beds={beds_cl} cap beds={beds_cap}")
    assert beds_cl[1][0] == beds_cl[2][0] == "8"
    assert beds_cap[1][0] == "16"
    assert beds_cap[2][0] == "8"
    axis_xy = polar_to_xy_mm(90.0, a8)
    scored = _fake_ray_scored("16", "single_inner")
    result = _lock_cap_walked_radial(
        cl_pixels, cap_pixels, calibrations, scored, axis_xy=axis_xy,
    )
    print(f"trap cap-walked result={result}")
    assert result is None
    # Chain-level, axis in the SCORED pie (16): nothing fires -- the
    # pair-unanimous lock's axis gate refuses, cap-walked's cap gate
    # refuses.
    assert apply_centerline_overrides(
        cl_pixels, calibrations, scored,
        cap_pixels=cap_pixels, axis_xy=polar_to_xy_mm(90.0, a16),
    ) is None
    # Chain-level, axis agreeing with the two unanimous onboard CLs
    # (8): since 2026-08-17 `_lock_pair_unanimous_cl_axis` DOES fire on
    # this variant -- measured +3/-0 on the living clean/ 996
    # (047-S5, 137-S8, 170-S8) with every documented naive-lock canary
    # (056-S4, 011-S5, throw_1786666308551, throw_1786667985388)
    # unchanged. See that function's docstring; the cap-walked lock
    # itself still refuses (asserted directly above).
    chain = apply_centerline_overrides(
        cl_pixels, calibrations, scored,
        cap_pixels=cap_pixels, axis_xy=axis_xy,
    )
    print(f"trap chain (axis in CL pie) result={chain}")
    assert chain is not None
    assert chain.sector == "8"
    assert "pair unanimous centerline" in chain.reason


def test_lock_centerline_ring_fires_when_cap_is_farther_out():
    """2+ onboard CLs unanimous inner, scored treble 8mm farther out."""
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a20 = sector_center_angle_deg(20)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(90.0, a20),
            1: polar_to_xy_mm(88.0, a20),
            2: polar_to_xy_mm(200.0, a20),
        },
    )
    print(f"ring-lock CL beds={beds} CL_RING_OUTWARD_MM={CL_RING_OUTWARD_MM}")
    assert beds[0][1] == beds[1][1] == "single_inner"
    scored_xy = polar_to_xy_mm(100.0, a20)
    scored = ScoreResult(
        ok=True,
        sector="20",
        ring="treble",
        board_xy_mm=scored_xy,
        triangulation=None,
        n_cameras_used=3,
    )
    d_r = float(np.hypot(*scored_xy) - np.hypot(*_mean_xy([rec[0], rec[1]])))
    print(f"ring-lock d_r={d_r:.3f}")
    assert d_r > CL_RING_OUTWARD_MM
    result = _lock_centerline_ring(cl_pixels, calibrations, scored)
    assert result is not None
    assert result.ring == "single_inner"
    assert "centerline ring lock" in result.reason


def test_lock_centerline_ring_fires_when_dr_below_old_cap_gate():
    """2026-08-24 skip-cap: live gate is d_r > 0, not CL_RING_OUTWARD_MM.

    Residual post-cap dR on the 374 is 0.03–1.7mm (022-T20 is 0.03mm
    past the treble wire). The 2.7mm constant was a cap-vs-CL quantity.
    """
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a20 = sector_center_angle_deg(20)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(96.0, a20),
            1: polar_to_xy_mm(96.2, a20),
            2: polar_to_xy_mm(200.0, a20),
        },
    )
    scored_xy = polar_to_xy_mm(97.8, a20)
    scored = ScoreResult(
        ok=True,
        sector="20",
        ring="treble",
        board_xy_mm=scored_xy,
        triangulation=None,
        n_cameras_used=3,
    )
    d_r = float(np.hypot(*scored_xy) - np.hypot(*_mean_xy([rec[0], rec[1]])))
    print(f"ring-lock small d_r={d_r:.3f} beds={beds}")
    assert 0.0 < d_r < CL_RING_OUTWARD_MM
    result = _lock_centerline_ring(cl_pixels, calibrations, scored)
    assert result is not None
    assert result.ring == "single_inner"
    assert "centerline ring lock" in result.reason


def test_lock_centerline_ring_does_not_fire_when_axis_is_cl_corroborated():
    """026-T7: 2 inner CLs vs axis+one treble CL -- do not lock to inner."""
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a7 = sector_center_angle_deg(7)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(99.0, a7),   # treble, agrees with axis
            1: polar_to_xy_mm(96.0, a7),   # inner majority
            2: polar_to_xy_mm(96.5, a7),
        },
    )
    print(f"corroborated-axis CL beds={beds}")
    scored_xy = polar_to_xy_mm(101.0, a7)
    scored = ScoreResult(
        ok=True,
        sector="7",
        ring="treble",
        board_xy_mm=scored_xy,
        triangulation=None,
        n_cameras_used=3,
    )
    axis_xy = polar_to_xy_mm(99.5, a7)
    assert sector_ring_for_point(*axis_xy) == ("7", "treble")
    assert _lock_centerline_ring(
        cl_pixels, calibrations, scored, axis_xy=axis_xy,
    ) is None
    # Majority inner with no dissenting CL still locks (0408695).
    cl_unanimous, _, _ = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(96.0, a7),
            1: polar_to_xy_mm(96.5, a7),
            2: polar_to_xy_mm(95.5, a7),
        },
    )
    fired = _lock_centerline_ring(
        cl_unanimous, calibrations, scored, axis_xy=axis_xy,
    )
    assert fired is not None
    assert fired.ring == "single_inner"


def test_lock_centerline_ring_skips_when_axis_is_outside():
    """dR>=2.7 but dart axis off the board: do not steal a true double."""
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a14 = sector_center_angle_deg(14)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(159.6, a14),
            1: polar_to_xy_mm(159.8, a14),
            2: polar_to_xy_mm(200.0, a14),
        },
    )
    print(f"axis-outside CL beds={beds}")
    assert beds[0][1] == beds[1][1] == "single_outer"
    scored_xy = polar_to_xy_mm(163.0, a14)
    scored = ScoreResult(
        ok=True,
        sector="14",
        ring="double",
        board_xy_mm=scored_xy,
        triangulation=None,
        n_cameras_used=2,
    )
    d_r = float(np.hypot(*scored_xy) - np.hypot(*_mean_xy([rec[0], rec[1]])))
    print(f"axis-outside d_r={d_r:.3f}")
    assert d_r > CL_RING_OUTWARD_MM
    axis_xy = polar_to_xy_mm(172.0, a14)
    assert sector_ring_for_point(*axis_xy)[1] == "outside"
    assert _lock_centerline_ring(
        cl_pixels, calibrations, scored, axis_xy=axis_xy,
    ) is None
    # Without an axis, the dR gate still fires (unit-test default).
    fired = _lock_centerline_ring(cl_pixels, calibrations, scored)
    assert fired is not None
    assert fired.ring == "single_outer"


def test_lock_centerline_ring_n3_outer_to_treble_below_dr_gate():
    """3 onboard CLs + axis treble, scored just into single_outer."""
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a20 = sector_center_angle_deg(20)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(105.4, a20),
            1: polar_to_xy_mm(105.6, a20),
            2: polar_to_xy_mm(105.5, a20),
        },
    )
    print(f"n3 outer-to-treble CL beds={beds}")
    assert all(b == ("20", "treble") for b in beds.values())
    scored_xy = polar_to_xy_mm(107.4, a20)
    scored = ScoreResult(
        ok=True,
        sector="20",
        ring="single_outer",
        board_xy_mm=scored_xy,
        triangulation=None,
        n_cameras_used=3,
    )
    d_r = float(np.hypot(*scored_xy) - np.hypot(*_mean_xy(list(rec.values()))))
    print(f"n3 outer-to-treble d_r={d_r:.3f}")
    assert 0.0 < d_r < CL_RING_OUTWARD_MM
    axis_xy = polar_to_xy_mm(106.2, a20)
    assert sector_ring_for_point(*axis_xy) == ("20", "treble")
    result = _lock_centerline_ring(
        cl_pixels, calibrations, scored, axis_xy=axis_xy,
    )
    assert result is not None
    assert result.ring == "treble"
    assert "centerline ring lock" in result.reason


def test_lock_centerline_ring_n2_outer_to_treble_fires_after_skip_cap():
    """After skip-cap, 7737057's 2.7mm n2 skip is net +3/-0 on this 374.

    2 onboard CLs + axis treble, result just into single_outer, 3rd
    off-board. Live gate is d_r > 0; the old cap-bias trap does not
    apply to centerline rays.
    """
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a20 = sector_center_angle_deg(20)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(105.4, a20),
            1: polar_to_xy_mm(105.6, a20),
            2: polar_to_xy_mm(200.0, a20),
        },
    )
    print(f"n2 lock CL beds={beds}")
    scored_xy = polar_to_xy_mm(107.4, a20)
    scored = ScoreResult(
        ok=True,
        sector="20",
        ring="single_outer",
        board_xy_mm=scored_xy,
        triangulation=None,
        n_cameras_used=3,
    )
    d_r = float(np.hypot(*scored_xy) - np.hypot(*_mean_xy([rec[0], rec[1]])))
    print(f"n2 lock d_r={d_r:.3f}")
    assert 0.0 < d_r < CL_RING_OUTWARD_MM
    axis_xy = polar_to_xy_mm(106.2, a20)
    result = _lock_centerline_ring(
        cl_pixels, calibrations, scored, axis_xy=axis_xy,
    )
    assert result is not None
    assert result.ring == "treble"
    assert "centerline ring lock" in result.reason


def test_lock_axis_ring_lonely_cl_fires_when_one_cl_in_sector():
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a20 = sector_center_angle_deg(20)
    a1 = sector_center_angle_deg(1)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(90.0, a1),
            1: polar_to_xy_mm(90.0, a20),
            2: polar_to_xy_mm(200.0, a20),
        },
    )
    print(f"lonely CL beds={beds}")
    assert beds[0][0] == "1"
    assert beds[1] == ("20", "single_inner")
    scored_xy = polar_to_xy_mm(100.0, a20)
    scored = ScoreResult(
        ok=True,
        sector="20",
        ring="treble",
        board_xy_mm=scored_xy,
        triangulation=None,
        n_cameras_used=3,
    )
    axis_xy = polar_to_xy_mm(90.0, a20)
    result = _lock_axis_ring_lonely_cl(
        cl_pixels, calibrations, scored, axis_xy=axis_xy,
    )
    print(f"lonely result ring={None if result is None else result.ring}")
    assert result is not None
    assert result.ring == "single_inner"
    assert "lonely centerline" in result.reason


def test_lock_axis_ring_lonely_cl_does_not_fire_with_two_in_sector():
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a20 = sector_center_angle_deg(20)
    cl_pixels, _, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(90.0, a20),
            1: polar_to_xy_mm(100.0, a20),
            2: polar_to_xy_mm(200.0, a20),
        },
    )
    print(f"lonely skip beds={beds}")
    scored = ScoreResult(
        ok=True,
        sector="20",
        ring="treble",
        board_xy_mm=polar_to_xy_mm(100.0, a20),
        triangulation=None,
        n_cameras_used=3,
    )
    axis_xy = polar_to_xy_mm(90.0, a20)
    assert _lock_axis_ring_lonely_cl(
        cl_pixels, calibrations, scored, axis_xy=axis_xy,
    ) is None


def test_lock_single_cl_double_to_outer_fires():
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a16 = sector_center_angle_deg(16)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(158.0, a16),
            1: polar_to_xy_mm(200.0, a16),
            2: polar_to_xy_mm(210.0, a16),
        },
    )
    print(f"double-to-outer CL beds={beds} rec0={rec[0]}")
    assert beds[0] == ("16", "single_outer")
    scored = ScoreResult(
        ok=True,
        sector="16",
        ring="double",
        board_xy_mm=polar_to_xy_mm(165.0, a16),
        triangulation=None,
        n_cameras_used=1,
    )
    result = _lock_single_cl_double_to_outer(cl_pixels, calibrations, scored)
    assert result is not None
    assert result.ring == "single_outer"
    assert "double-to-outer" in result.reason


def test_lock_single_cl_double_to_outer_does_not_fire_with_two():
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a16 = sector_center_angle_deg(16)
    cl_pixels, _, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(158.0, a16),
            1: polar_to_xy_mm(157.0, a16),
            2: polar_to_xy_mm(200.0, a16),
        },
    )
    print(f"double-to-outer skip beds={beds}")
    scored = ScoreResult(
        ok=True,
        sector="16",
        ring="double",
        board_xy_mm=polar_to_xy_mm(165.0, a16),
        triangulation=None,
        n_cameras_used=2,
    )
    assert _lock_single_cl_double_to_outer(cl_pixels, calibrations, scored) is None


def test_lock_split_centerline_mean_fires_on_outer_double():
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a4 = sector_center_angle_deg(4)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(157.0, a4),
            1: polar_to_xy_mm(164.5, a4),
            2: polar_to_xy_mm(200.0, a4),
        },
    )
    print(f"split CL beds={beds}")
    assert beds[0][1] == "single_outer"
    assert beds[1][1] == "double"
    mean = _mean_xy([rec[0], rec[1]])
    mean_bed = sector_ring_for_point(*mean)
    print(f"split mean r={np.hypot(*mean):.3f} bed={mean_bed}")
    assert mean_bed == ("4", "double")
    scored = ScoreResult(
        ok=True,
        sector="4",
        ring="single_outer",
        board_xy_mm=polar_to_xy_mm(158.0, a4),
        triangulation=None,
        n_cameras_used=2,
    )
    result = _lock_split_centerline_mean(cl_pixels, calibrations, scored)
    assert result is not None
    assert result.ring == "double"
    assert "split centerline" in result.reason


def test_lock_split_centerline_mean_skips_when_mean_matches():
    """6373542 shape: 1-1 outer/double, result already double."""
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a16 = sector_center_angle_deg(16)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(157.4, a16),
            1: polar_to_xy_mm(164.5, a16),
            2: polar_to_xy_mm(200.0, a16),
        },
    )
    print(f"split skip beds={beds}")
    mean = _mean_xy([rec[0], rec[1]])
    mean_bed = sector_ring_for_point(*mean)
    scored_xy = polar_to_xy_mm(163.8, a16)
    print(f"split skip mean={mean_bed} scored={sector_ring_for_point(*scored_xy)}")
    scored = ScoreResult(
        ok=True,
        sector="16",
        ring="double",
        board_xy_mm=scored_xy,
        triangulation=None,
        n_cameras_used=2,
    )
    assert mean_bed == ("16", "double")
    assert _lock_split_centerline_mean(cl_pixels, calibrations, scored) is None


def test_lock_split_leftover_uses_axis_when_delta_r_is_huge():
    """068-S10: leftover CL at r=52 vs real at r=99 is not a wire straddle."""
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a10 = sector_center_angle_deg(10)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(52.6, a10),
            1: polar_to_xy_mm(98.97, a10),
            2: polar_to_xy_mm(200.0, a10),
        },
    )
    print(f"leftover split beds={beds}")
    d_r = abs(float(np.hypot(*rec[0]) - np.hypot(*rec[1])))
    print(f"leftover split Δr={d_r:.3f} gate={SPLIT_MAX_RADIUS_DISAGREE_MM}")
    assert d_r > SPLIT_MAX_RADIUS_DISAGREE_MM
    scored = ScoreResult(
        ok=True,
        sector="10",
        ring="single_inner",
        board_xy_mm=polar_to_xy_mm(75.0, a10),
        triangulation=None,
        n_cameras_used=2,
    )
    axis_xy = polar_to_xy_mm(102.8, a10)
    assert sector_ring_for_point(*axis_xy) == ("10", "treble")
    result = _lock_split_centerline_mean(
        cl_pixels, calibrations, scored, axis_xy=axis_xy,
    )
    assert result is not None
    assert result.ring == "treble"
    assert "leftover" in result.reason
    # Without an axis, do not invent a mean of two different darts.
    assert _lock_split_centerline_mean(cl_pixels, calibrations, scored) is None


def test_lock_insector_ring_majority_2of3_pulls_inward_leftover():
    """014-T18: 2 treble + 1 leftover inner, result pulled inside the wire."""
    from opendarts.pipeline import ScoreResult

    syn, calibrations = _three_ring_cameras()
    a18 = sector_center_angle_deg(18)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(98.5, a18),
            1: polar_to_xy_mm(99.0, a18),
            2: polar_to_xy_mm(86.5, a18),
        },
    )
    print(f"2of3 ring beds={beds}")
    assert all(b[0] == "18" for b in beds.values())
    rings = [b[1] for b in beds.values()]
    assert rings.count("treble") == 2
    assert rings.count("single_inner") == 1
    scored = ScoreResult(
        ok=True,
        sector="18",
        ring="single_inner",
        board_xy_mm=polar_to_xy_mm(97.31, a18),
        triangulation=None,
        n_cameras_used=3,
    )
    # Ring lock must not steal this: result is *inward* of the majority.
    d_r = float(
        np.hypot(*scored.board_xy_mm)
        - np.hypot(*_mean_xy([rec[0], rec[1]]))
    )
    print(f"2of3 inward d_r={d_r:.3f}")
    assert d_r < 0.0
    leftover_dr = abs(float(np.hypot(*rec[2]) - np.hypot(*_mean_xy([rec[0], rec[1]]))))
    print(f"2of3 leftover Δr={leftover_dr:.3f} gate={INSECTOR_RING_MAJORITY_MIN_LEFTOVER_DR_MM}")
    assert leftover_dr > INSECTOR_RING_MAJORITY_MIN_LEFTOVER_DR_MM
    assert _lock_centerline_ring(cl_pixels, calibrations, scored) is None
    result = _lock_insector_ring_majority_2of3(cl_pixels, calibrations, scored)
    assert result is not None
    assert result.ring == "treble"
    assert "2-of-3 in-sector ring majority" in result.reason


def test_lock_insector_ring_majority_2of3_skips_wire_adjacent_straddle():
    """013-S1: 2 treble + 1 inner, Δr=5.5mm -- triangulation already right."""
    syn, calibrations = _three_ring_cameras()
    a1 = sector_center_angle_deg(1)
    cl_pixels, rec, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(95.1, a1),
            1: polar_to_xy_mm(98.8, a1),
            2: polar_to_xy_mm(102.5, a1),
        },
    )
    print(f"2of3 wire-straddle beds={beds}")
    leftover_dr = abs(float(np.hypot(*rec[0]) - np.hypot(*_mean_xy([rec[1], rec[2]]))))
    print(f"2of3 wire-straddle leftover Δr={leftover_dr:.3f}")
    assert leftover_dr < INSECTOR_RING_MAJORITY_MIN_LEFTOVER_DR_MM
    scored = ScoreResult(
        ok=True,
        sector="1",
        ring="single_inner",
        board_xy_mm=polar_to_xy_mm(95.6, a1),
        triangulation=None,
        n_cameras_used=3,
    )
    assert _lock_insector_ring_majority_2of3(cl_pixels, calibrations, scored) is None


def test_lock_insector_ring_majority_2of3_skips_when_result_already_maj():
    syn, calibrations = _three_ring_cameras()
    a18 = sector_center_angle_deg(18)
    cl_pixels, _, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(113.5, a18),
            1: polar_to_xy_mm(98.3, a18),
            2: polar_to_xy_mm(108.6, a18),
        },
    )
    print(f"2of3 already-maj beds={beds}")
    scored = ScoreResult(
        ok=True,
        sector="18",
        ring="single_outer",
        board_xy_mm=polar_to_xy_mm(110.0, a18),
        triangulation=None,
        n_cameras_used=3,
    )
    assert _lock_insector_ring_majority_2of3(cl_pixels, calibrations, scored) is None


# Measured on this file's noiseless E scene:
#   recovery_error_vs_recovered_xy_mm = 0.000000000000e+00
#   recovery_error_vs_true_xy_mm      = 1.421085471520e-14
# 1e-11 mm is the same bound as _MAX_RECOVERY_ERROR_MM.
_MAX_FAR_DROP_RECOVERY_ERROR_MM = 1e-11


def test_drop_far_cap_rays_keeps_one_onboard_and_one_cam_z0_recovers_it():
    """E: two cap rays past FAR_CAP_RADIUS_MM are dropped; the remaining
    on-board camera's ray∩Z=0 is the score."""
    syn, calibrations = _three_ring_cameras()
    angle_9 = sector_center_angle_deg(9)
    p_on = polar_to_xy_mm(140.0, angle_9)
    p_far_a = polar_to_xy_mm(400.0, angle_9)
    p_far_b = polar_to_xy_mm(550.0, angle_9)
    print(
        f"E FAR_CAP_RADIUS_MM={FAR_CAP_RADIUS_MM} "
        f"on={sector_ring_for_point(*p_on)} r={np.hypot(*p_on)} "
        f"far_a_r={np.hypot(*p_far_a)} far_b_r={np.hypot(*p_far_b)}"
    )
    assert FAR_CAP_RADIUS_MM == 250.0
    assert np.hypot(*p_on) < FAR_CAP_RADIUS_MM
    assert np.hypot(*p_far_a) > FAR_CAP_RADIUS_MM
    assert np.hypot(*p_far_b) > FAR_CAP_RADIUS_MM
    assert sector_ring_for_point(*p_on) == ("9", "single_outer")

    pixels, recovered, beds = _project_cl(
        syn, calibrations, {0: p_on, 1: p_far_a, 2: p_far_b}
    )
    assert beds[0] == ("9", "single_outer")
    assert beds[1] == (None, "outside")
    assert beds[2] == (None, "outside")

    kept, far = drop_far_cap_rays(pixels, calibrations)
    print(f"E kept={list(kept)} far={far}")
    assert set(kept) == {0}
    assert set(far) == {1, 2}
    assert far[1] > FAR_CAP_RADIUS_MM
    assert far[2] > FAR_CAP_RADIUS_MM

    scored = one_cam_z0(0, kept[0], calibrations)
    assert scored is not None
    vs_rec = float(
        np.hypot(
            scored.board_xy_mm[0] - recovered[0][0],
            scored.board_xy_mm[1] - recovered[0][1],
        )
    )
    vs_true = float(
        np.hypot(
            scored.board_xy_mm[0] - p_on[0],
            scored.board_xy_mm[1] - p_on[1],
        )
    )
    print(
        f"E one_cam_z0 xy={scored.board_xy_mm} "
        f"bed={(scored.sector, scored.ring)} "
        f"recovery_error_vs_recovered_xy_mm={vs_rec:.12e} "
        f"recovery_error_vs_true_xy_mm={vs_true:.12e} "
        f"n_cameras_used={scored.n_cameras_used} "
        f"cameras_used={scored.cameras_used}"
    )
    assert vs_rec < _MAX_FAR_DROP_RECOVERY_ERROR_MM, (
        f"E one_cam_z0 vs recovered XY {vs_rec:.12e} mm exceeds "
        f"{_MAX_FAR_DROP_RECOVERY_ERROR_MM} mm (measured noiseless 0.0 mm)"
    )
    assert vs_true < _MAX_FAR_DROP_RECOVERY_ERROR_MM, (
        f"E one_cam_z0 vs true XY {vs_true:.12e} mm exceeds "
        f"{_MAX_FAR_DROP_RECOVERY_ERROR_MM} mm "
        f"(measured noiseless 1.421085471520e-14 mm)"
    )
    assert scored.ok is True
    assert (scored.sector, scored.ring) == ("9", "single_outer")
    assert scored.n_cameras_used == 1
    assert scored.cameras_used == (0,)
    assert "1-cam Z=0" in scored.reason


def test_drop_far_cap_rays_keeps_all_three_when_onboard():
    """E must not drop cameras whose cap ray∩Z=0 is on the board."""
    syn, calibrations = _three_ring_cameras()
    angle_9 = sector_center_angle_deg(9)
    pixels, recovered, beds = _project_cl(
        syn,
        calibrations,
        {
            0: polar_to_xy_mm(50.0, angle_9),
            1: polar_to_xy_mm(80.0, angle_9),
            2: polar_to_xy_mm(140.0, angle_9),
        },
    )
    for cam, rec in recovered.items():
        r = float(np.hypot(rec[0], rec[1]))
        print(f"E all-onboard cam{cam} r={r:.12e} bed={beds[cam]}")
        assert r < FAR_CAP_RADIUS_MM
        assert beds[cam][1] != "outside"
    kept, far = drop_far_cap_rays(pixels, calibrations)
    print(f"E all-onboard kept={list(kept)} far={far}")
    assert set(kept) == {0, 1, 2}
    assert far == {}


def test_drop_far_cap_rays_drops_291_junk_keeps_220_near_outside():
    """Measured gap: 024-S4 junk is r≈291 (drop); real near-outsides
    live in [170, 220) (keep). 250 sits in that gap; 340 kept the 291
    ray and missed 024-S4. 300 also misses it.

    Measured on this noiseless scene:
      constructed/recovered keep_r = 220.0, drop_r = 291.0, on_r = 140.0
      far[1] = 291.00000000000006
    """
    syn, calibrations = _three_ring_cameras()
    angle_4 = sector_center_angle_deg(4)
    p_keep = polar_to_xy_mm(220.0, angle_4)
    p_drop = polar_to_xy_mm(291.0, angle_4)
    p_on = polar_to_xy_mm(140.0, angle_4)
    r_keep = float(np.hypot(*p_keep))
    r_drop = float(np.hypot(*p_drop))
    r_on = float(np.hypot(*p_on))
    print(
        f"gap FAR_CAP_RADIUS_MM={FAR_CAP_RADIUS_MM} "
        f"keep_r={r_keep:.12e} drop_r={r_drop:.12e} on_r={r_on:.12e} "
        f"keep_bed={sector_ring_for_point(*p_keep)} "
        f"drop_bed={sector_ring_for_point(*p_drop)} "
        f"on_bed={sector_ring_for_point(*p_on)}"
    )
    assert FAR_CAP_RADIUS_MM == 250.0
    assert r_keep == 220.0
    assert r_drop == 291.0
    assert r_on == 140.0
    assert r_keep < FAR_CAP_RADIUS_MM < r_drop
    assert sector_ring_for_point(*p_keep) == (None, "outside")
    assert sector_ring_for_point(*p_drop) == (None, "outside")
    assert sector_ring_for_point(*p_on) == ("4", "single_outer")

    pixels, recovered, beds = _project_cl(
        syn, calibrations, {0: p_keep, 1: p_drop, 2: p_on}
    )
    rec_keep = float(np.hypot(*recovered[0]))
    rec_drop = float(np.hypot(*recovered[1]))
    rec_on = float(np.hypot(*recovered[2]))
    print(
        f"gap recovered keep_r={rec_keep:.12e} drop_r={rec_drop:.12e} "
        f"on_r={rec_on:.12e} beds={beds}"
    )
    assert abs(rec_keep - 220.0) < _MAX_FAR_DROP_RECOVERY_ERROR_MM
    assert abs(rec_drop - 291.0) < _MAX_FAR_DROP_RECOVERY_ERROR_MM
    assert abs(rec_on - 140.0) < _MAX_FAR_DROP_RECOVERY_ERROR_MM
    assert beds[0] == (None, "outside")
    assert beds[1] == (None, "outside")
    assert beds[2] == ("4", "single_outer")

    kept, far = drop_far_cap_rays(pixels, calibrations)
    print(f"gap kept={list(kept)} far={far}")
    assert set(kept) == {0, 2}
    assert set(far) == {1}
    assert far[1] > FAR_CAP_RADIUS_MM
    assert abs(far[1] - 291.0) < _MAX_FAR_DROP_RECOVERY_ERROR_MM


# Measured on this file's noiseless slide scene:
#   true axis (40, 25, 0) -> (20, 10, 80), tilt 17.35deg
#   point at z=+5: (38.75, 24.0625, 5)
#   analytic hit at z=-1.5: (40.375, 25.28125, -1.5)
#   slide_xy_error_mm = 0.000000000000e+00
# 1e-11 mm is the same bound as _MAX_RECOVERY_ERROR_MM.
_MAX_SLIDE_ERROR_MM = 1e-11


def test_slide_point_along_dart_axis_hits_sisal_plane_on_known_tilt():
    """A 3D point at z=+5 on a known-tilt dart axis slides to z=-1.5
    along that same reconstructed axis, not a per-camera ray."""
    assert BOARD_HIT_Z_MM == -1.5
    true_hit = np.array([40.0, 25.0, 0.0], dtype=np.float64)
    p_shaft = np.array([20.0, 10.0, 80.0], dtype=np.float64)
    axis = p_shaft - true_hit
    tilt_deg = float(np.degrees(np.arccos(axis[2] / np.linalg.norm(axis))))
    print(f"slide true axis={axis} tilt_deg={tilt_deg}")

    t_z5 = 5.0 / axis[2]
    point_z5 = true_hit + t_z5 * axis
    print(f"slide point_z5={point_z5}")
    assert abs(float(point_z5[2]) - 5.0) < _MAX_SLIDE_ERROR_MM

    t_sisal = BOARD_HIT_Z_MM / axis[2]
    analytic = true_hit + t_sisal * axis
    print(f"slide analytic_hit={analytic}")
    assert abs(float(analytic[2]) - BOARD_HIT_Z_MM) < _MAX_SLIDE_ERROR_MM

    K = make_camera_matrix(fov_deg=90.0)
    syn_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=K) for i in range(3)]
    planes = []
    for i, cam in enumerate(syn_cams):
        p1, p2 = _project_line_px(cam, true_hit, p_shaft)
        plane = plane_from_image_line(p1, p2, _as_calibration(cam))
        assert plane is not None
        planes.append(plane)

    slid = slide_point_along_dart_axis(point_z5, planes)
    assert slid is not None
    err = float(np.hypot(slid[0] - analytic[0], slid[1] - analytic[1]))
    print(
        f"slide slid_xy={slid} analytic_xy=({analytic[0]}, {analytic[1]}) "
        f"slide_xy_error_mm={err:.12e}"
    )
    assert err < _MAX_SLIDE_ERROR_MM, (
        f"axis-slide XY error {err:.12e} mm exceeds {_MAX_SLIDE_ERROR_MM} mm "
        f"(measured noiseless 0.0 mm)"
    )
    # Sliding to Z=0 instead of -1.5 would land on true_hit (40, 25);
    # the sisal hit is 0.375mm radially outward along this axis.
    dist_to_z0 = float(np.hypot(slid[0] - true_hit[0], slid[1] - true_hit[1]))
    print(f"slide dist_to_z0_hit_mm={dist_to_z0:.12e}")
    assert dist_to_z0 > 0.3


def test_snap_xy_to_plane_line_is_closest_point_on_ax_plus_by():
    """n=(1,0,0), c=5, z=0 is the line x=5. (8,3) snaps to (5,3)."""
    n = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    snapped = snap_xy_to_plane_line((8.0, 3.0), (n, 5.0), 0.0)
    assert snapped is not None
    err = float(np.hypot(snapped[0] - 5.0, snapped[1] - 3.0))
    print(f"snap_xy={snapped} err={err:.12e}")
    assert err < 1e-12
    on_line = snap_xy_to_plane_line((5.0, -2.0), (n, 5.0), 0.0)
    on_err = float(np.hypot(on_line[0] - 5.0, on_line[1] + 2.0))
    print(f"snap_on_line={on_line} err={on_err:.12e}")
    assert on_err < 1e-12
