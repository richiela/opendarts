"""Tests for opendarts.geometry.board_color -- the board-paint color sanity
check. Three layers, matching the task this shipped for:

1. The color-pattern model (expected_color()) -- pure lookup, validated
   against the FULL real alternation measured off the real corpus (see
   board_color.py's own module docstring for the raw measurement this
   is derived from).
2. The color sampler (classify_bgr(), sample_board_color(),
   sample_board_color_multi_camera()) -- validated against real,
   measured BGR values (not synthetic/guessed colors) plus synthetic
   edge cases (out-of-frame projections, tie-breaking).
3. A real, corpus-gated measurement test (skips cleanly with no real
   corpus present, same pattern as
   tests/test_board_geometry.py::test_ad_own_points_agree_with_our_geometry_on_the_real_corpus)
   that reproduces this task's own corpus-wide validation: does color
   disagreement predict a real miss? Documents the real measured numbers
   rather than hardcoding brittle exact counts against a corpus this
   project explicitly treats as living/evolving (docs/DESIGN.md).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from opendarts.geometry.board import SECTOR_NUMBERS_CLOCKWISE
from opendarts.geometry.board_color import (
    BRIGHTNESS_THRESHOLD_BLACK_CREAM,
    CHROMA_THRESHOLD,
    PATCH_RADIUS_PX,
    MultiCameraColorSample,
    classify_bgr,
    color_agrees,
    evaluate_color_sanity,
    expected_color,
    sample_board_color,
    sample_board_color_multi_camera,
    sample_patch_bgr,
    sector_accent_color,
    sector_single_color,
)

# ---------------------------------------------------------------------
# 1. Color-pattern model
# ---------------------------------------------------------------------


def test_expected_color_outside_is_none():
    assert expected_color("5", "outside") is None
    assert expected_color(None, "outside") is None


def test_expected_color_bull_and_outer_bull_are_fixed_regardless_of_sector():
    assert expected_color(None, "bull") == "red"
    assert expected_color(None, "outer_bull") == "green"


def test_expected_color_full_alternation_matches_real_measured_pattern():
    """Real measured pattern (board_color.py's own module docstring,
    558 real background-pixel samples across all 3 corpus sessions):
    sector 20 (idx 0, even) is BLACK single / RED treble+double; sector
    1 (idx 1, odd) is CREAM single / GREEN treble+double; strictly
    alternating from there with ZERO exceptions across all 20 sectors.
    This is the mirror of the generic description this task started
    from -- verified against real images, not assumed."""
    for idx, number in enumerate(SECTOR_NUMBERS_CLOCKWISE):
        expected_single = "black" if idx % 2 == 0 else "cream"
        expected_accent = "red" if idx % 2 == 0 else "green"
        assert sector_single_color(number) == expected_single, f"sector {number}"
        assert sector_accent_color(number) == expected_accent, f"sector {number}"
        assert expected_color(str(number), "single_inner") == expected_single
        assert expected_color(str(number), "single_outer") == expected_single
        assert expected_color(str(number), "treble") == expected_accent
        assert expected_color(str(number), "double") == expected_accent


def test_expected_color_sector_20_and_1_anchor_values():
    """The two sectors this task's own investigation centered on --
    pinned explicitly (not just covered by the sweep above) so a future
    change to the pattern's phase is caught immediately and obviously,
    not just as one entry failing in a 20-way parametrized loop."""
    assert sector_single_color(20) == "black"
    assert sector_accent_color(20) == "red"
    assert sector_single_color(1) == "cream"
    assert sector_accent_color(1) == "green"


def test_expected_color_unknown_sector_is_none_not_a_crash():
    assert expected_color("99", "single_inner") is None
    assert expected_color("not-a-number", "treble") is None
    assert expected_color(None, "single_inner") is None  # single_inner needs a sector


# ---------------------------------------------------------------------
# 2. Color sampler
# ---------------------------------------------------------------------

# Real BGR samples pulled directly from this task's own corpus
# measurement (cam0, throw_1786730490647) --
# not synthetic/guessed colors. See board_color.py's module docstring
# for the full 558-sample measurement these were drawn from.
_REAL_BLACK_BGR = (86.1, 72.2, 67.0)   # sector 20 single, cam0
_REAL_CREAM_BGR = (237.2, 249.1, 239.4)  # sector 1 single, cam0
_REAL_RED_BGR = (145.1, 137.8, 238.3)   # sector 20 treble, cam0
_REAL_GREEN_BGR = (123.7, 194.6, 115.2)  # sector 1 treble, cam0


@pytest.mark.parametrize(
    "bgr,expected",
    [
        (_REAL_BLACK_BGR, "black"),
        (_REAL_CREAM_BGR, "cream"),
        (_REAL_RED_BGR, "red"),
        (_REAL_GREEN_BGR, "green"),
    ],
)
def test_classify_bgr_real_measured_samples(bgr, expected):
    assert classify_bgr(*bgr) == expected


def test_classify_bgr_thresholds_are_the_measured_values():
    """Pins the actual constants to the real measured numbers documented
    in board_color.py's module docstring -- a silent edit to either
    threshold should fail a test, not just silently drift."""
    assert BRIGHTNESS_THRESHOLD_BLACK_CREAM == 129.0
    assert CHROMA_THRESHOLD == 35.0
    assert PATCH_RADIUS_PX == 3


def _solid_image(bgr: tuple[float, float, float], size: int = 40) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def test_sample_patch_bgr_returns_none_when_center_out_of_frame():
    img = _solid_image(_REAL_BLACK_BGR)
    assert sample_patch_bgr(img, -1, 5) is None
    assert sample_patch_bgr(img, 5, 999) is None


def test_sample_patch_bgr_clips_partial_edge_overlap_rather_than_refusing():
    img = _solid_image(_REAL_CREAM_BGR, size=10)
    # Center pixel (0, 0) is in-frame; a radius-3 patch around it is
    # mostly off-frame -- should still average what's actually there.
    result = sample_patch_bgr(img, 0, 0, patch_radius=3)
    assert result is not None
    for channel, expected in zip(result, _REAL_CREAM_BGR):
        assert channel == pytest.approx(expected, abs=1.0)


class _FakeCalib:
    """Minimal stand-in for opendarts.pipeline.CameraCalibration -- only
    needs to support project_board_point_px() via a monkeypatched
    cv2.projectPoints, which these tests avoid entirely by testing
    sample_board_color() at the sample_patch_bgr layer instead (see
    test_sample_board_color_* below, which patch project_board_point_px
    directly rather than fighting with real camera intrinsics)."""


def test_sample_board_color_single_camera(monkeypatch):
    img = _solid_image(_REAL_RED_BGR)
    import opendarts.geometry.board_color as board_color_mod

    monkeypatch.setattr(board_color_mod, "project_board_point_px", lambda xy, calib: (20.0, 20.0))
    result = sample_board_color((99.5, 0.0), _FakeCalib(), img)
    assert result == "red"


def test_sample_board_color_none_when_projection_off_frame(monkeypatch):
    img = _solid_image(_REAL_RED_BGR)
    import opendarts.geometry.board_color as board_color_mod

    monkeypatch.setattr(board_color_mod, "project_board_point_px", lambda xy, calib: (-500.0, -500.0))
    result = sample_board_color((99.5, 0.0), _FakeCalib(), img)
    assert result is None


def test_multi_camera_majority_vote_outvotes_a_single_bad_camera(monkeypatch):
    """The real motivating measurement (board_color.py's module
    docstring): single-camera self-consistency was 95.9%, majority-
    vote-of-3 was 100.0% on the same real points -- because the errors
    (one camera's shadowed/edge sample) are decorrelated across
    cameras. This test proves the voting mechanism itself does what
    that measurement depends on, with a controlled synthetic case."""
    import opendarts.geometry.board_color as board_color_mod

    imgs = {
        0: _solid_image(_REAL_RED_BGR),
        1: _solid_image(_REAL_RED_BGR),
        2: _solid_image(_REAL_BLACK_BGR),  # the "bad" camera -- shadowed/wrong
    }
    calibs = {0: _FakeCalib(), 1: _FakeCalib(), 2: _FakeCalib()}
    monkeypatch.setattr(board_color_mod, "project_board_point_px", lambda xy, calib: (20.0, 20.0))

    result = sample_board_color_multi_camera((99.5, 0.0), calibs, imgs)
    assert isinstance(result, MultiCameraColorSample)
    assert result.per_camera == {0: "red", 1: "red", 2: "black"}
    assert result.majority == "red"
    assert result.agreement == pytest.approx(2 / 3)


def test_multi_camera_sample_skips_cameras_missing_a_bg_image(monkeypatch):
    import opendarts.geometry.board_color as board_color_mod

    imgs = {0: _solid_image(_REAL_GREEN_BGR)}  # camera 1 has no bg image
    calibs = {0: _FakeCalib(), 1: _FakeCalib()}
    monkeypatch.setattr(board_color_mod, "project_board_point_px", lambda xy, calib: (20.0, 20.0))

    result = sample_board_color_multi_camera((0.0, 0.0), calibs, imgs)
    assert result.per_camera == {0: "green"}
    assert result.majority == "green"
    assert result.agreement == 1.0


def test_multi_camera_sample_no_votes_returns_none_majority(monkeypatch):
    import opendarts.geometry.board_color as board_color_mod

    imgs = {0: _solid_image(_REAL_RED_BGR)}
    calibs = {0: _FakeCalib()}
    # Every camera's projection lands off-frame.
    monkeypatch.setattr(board_color_mod, "project_board_point_px", lambda xy, calib: (-999.0, -999.0))

    result = sample_board_color_multi_camera((0.0, 0.0), calibs, imgs)
    assert result.per_camera == {0: None}
    assert result.majority is None
    assert result.agreement is None


def test_color_agrees_true_false_and_not_applicable():
    assert color_agrees("black", "black") is True
    assert color_agrees("black", "red") is False
    assert color_agrees(None, "red") is None
    assert color_agrees("black", None) is None


def test_evaluate_color_sanity_not_applicable_when_ring_missing_or_off_board(monkeypatch):
    result = evaluate_color_sanity(None, None, None, {}, {})
    assert result.agrees is None
    assert result.expected is None
    assert result.sampled is None

    result_outside = evaluate_color_sanity("5", "outside", (200.0, 0.0), {}, {})
    assert result_outside.expected is None
    assert result_outside.agrees is None


def test_evaluate_color_sanity_agrees_on_a_correct_call(monkeypatch):
    import opendarts.geometry.board_color as board_color_mod

    imgs = {0: _solid_image(_REAL_RED_BGR)}
    calibs = {0: _FakeCalib()}
    monkeypatch.setattr(board_color_mod, "project_board_point_px", lambda xy, calib: (20.0, 20.0))

    # Sector 20, treble -- real expected color is "red" (see the anchor
    # test above), and the sampled image is solid red.
    result = evaluate_color_sanity("20", "treble", (99.5, 0.0), calibs, imgs)
    assert result.expected == "red"
    assert result.sampled == "red"
    assert result.agrees is True


def test_evaluate_color_sanity_disagrees_on_a_wrong_call(monkeypatch):
    import opendarts.geometry.board_color as board_color_mod

    imgs = {0: _solid_image(_REAL_GREEN_BGR)}  # actual paint is green
    calibs = {0: _FakeCalib()}
    monkeypatch.setattr(board_color_mod, "project_board_point_px", lambda xy, calib: (20.0, 20.0))

    # Sector 20, treble -- real expected color is "red", but the board
    # itself (per the sampled image) shows green here.
    result = evaluate_color_sanity("20", "treble", (99.5, 0.0), calibs, imgs)
    assert result.expected == "red"
    assert result.sampled == "green"
    assert result.agrees is False


# ---------------------------------------------------------------------
# 3. Real corpus-gated validation -- does disagreement predict a miss?
# ---------------------------------------------------------------------


def _corpus_root() -> Path:
    return Path(
        os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
        or (Path(__file__).resolve().parent.parent / "data" / "archive" / "clean")
    )


def test_color_sanity_check_self_consistency_on_real_corpus_geometry():
    """Reproduces the pure-geometry half of this task's own corpus
    measurement without needing an engine replay: for every real
    package's own calibration + background images, sample the color at
    the midpoint of EVERY sector's single/treble/double band (the exact
    ground truth this module's pattern model asserts) and confirm the
    sampler + pattern model agree with each other at a high rate. This
    is the same self-consistency measurement board_color.py's own
    module docstring reports (95.9% single-camera / 100% majority-vote
    on 3 sampled packages) -- run here against WHATEVER real corpus is
    on disk, skipping cleanly if none is present, and asserting a real
    (not aspirational) bound.
    """
    import json

    import cv2

    from opendarts.capture.throw_package import calibration_from_dict
    from opendarts.geometry.board import (
        DOUBLE_INNER_RADIUS_MM,
        DOUBLE_OUTER_RADIUS_MM,
        OUTER_BULL_RADIUS_MM,
        TREBLE_INNER_RADIUS_MM,
        TREBLE_OUTER_RADIUS_MM,
        polar_to_xy_mm,
        sector_center_angle_deg,
    )

    root = _corpus_root()
    if not root.exists():
        pytest.skip(f"no real corpus at {root} -- this measurement needs real throws")

    # One package per session for lighting diversity, same as this
    # task's own tmp/ measurement script -- keeps this test fast (a
    # handful of packages x 20 sectors x 3 cameras, not the whole corpus).
    session_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not session_dirs:
        pytest.skip(f"corpus at {root} has no session directories")

    pkg_dirs = []
    for session_dir in session_dirs:
        throws = sorted(session_dir.glob("throw_*"))
        if throws:
            pkg_dirs.append(throws[len(throws) // 2])  # a representative mid-session throw
    if not pkg_dirs:
        pytest.skip(f"corpus at {root} has no throw packages")

    single_r = (OUTER_BULL_RADIUS_MM + TREBLE_INNER_RADIUS_MM) / 2.0
    treble_r = (TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2.0
    double_r = (DOUBLE_INNER_RADIUS_MM + DOUBLE_OUTER_RADIUS_MM) / 2.0

    n_checked = 0
    n_agree = 0
    for pkg_dir in pkg_dirs:
        calib_path = pkg_dir / "calibration.json"
        if not calib_path.exists():
            continue
        calib_raw = json.loads(calib_path.read_text())
        calibrations = {int(k): calibration_from_dict(v) for k, v in calib_raw.items()}
        bg_images = {}
        for cam in calibrations:
            bg_path = pkg_dir / f"cam{cam}_bg.png"
            if bg_path.exists():
                bg_images[cam] = cv2.imread(str(bg_path))
        if not bg_images:
            continue

        for number in SECTOR_NUMBERS_CLOCKWISE:
            angle = sector_center_angle_deg(number)
            for ring, radius in [("single_inner", single_r), ("treble", treble_r), ("double", double_r)]:
                xy_mm = polar_to_xy_mm(radius, angle)
                result = evaluate_color_sanity(str(number), ring, xy_mm, calibrations, bg_images)
                if result.agrees is None:
                    continue
                n_checked += 1
                if result.agrees:
                    n_agree += 1

    if n_checked == 0:
        pytest.skip(f"corpus at {root} produced no checkable color samples")

    agreement_rate = n_agree / n_checked
    # Real measured bound (module docstring: 100% majority-vote
    # self-consistency on the 3 packages/9 cam-samples this was
    # originally derived from). Kept slightly below 100% here (95%) so
    # this test is robust to a genuinely different lighting/session mix
    # on whatever corpus happens to be on disk, while still failing
    # loudly if the sampler or pattern model regress in a real way.
    assert agreement_rate >= 0.95, (
        f"color sampler only agreed with the pattern model on "
        f"{n_agree}/{n_checked} ({agreement_rate*100:.1f}%) of real sector "
        f"midpoints across {len(pkg_dirs)} packages -- expected >=95% "
        f"(measured baseline was 100% majority-vote on the original 3-package sample)"
    )


@pytest.mark.slow  # full-corpus replay, ~67s measured 2026-08-16 -- see tests/conftest.py
def test_color_disagreement_correlates_with_real_misses_on_full_corpus():
    """The actual validation question this task exists to answer: when
    an engine's own scored (sector, ring) disagrees with the board's
    real paint color at its own triangulated board_xy_mm, is the TRUTH
    (operator-confirmed if present, else AD -- opendarts.live.server's own
    `_operator_truth_for(ad_gt) or ad_gt` precedence, reused verbatim
    here, not re-derived) more often on the side of "the engine was
    actually wrong"?

    Real measured result (360-throw corpus, 2026-08-14, see this task's
    own final report for the full table): color disagreement DOES
    correlate with a real elevated miss rate for Apollo (14.8% vs
    1.7% baseline, ~8.7x) and Talos (9.7% vs 0.7%, ~14x) -- but with
    low PRECISION (only 15-20% of "disagree" throws are actual misses;
    the rest are throws the engine already scored correctly, near a
    wire/boundary color-bleed ambiguity). Athena showed no real
    correlation (5.0% vs 4.1%). This is why board_color.py ships as an
    available diagnostic (`evaluate_color_sanity()`), not an
    auto-correction wired into any live scoring path -- see that
    function's own docstring.

    This test reproduces the SAME measurement dynamically against
    whatever real corpus is on disk (not hardcoded counts -- this
    corpus is explicitly living/evolving per docs/DESIGN.md) and asserts the
    real, structural claim: for at least one of the three engines, a
    color disagreement's miss-rate is measurably higher than an
    agreement's miss-rate. Uses opendarts.capture.replay.
    replay_throw_with_engine, per docs/DESIGN.md's "Replay is the source of truth" -- never
    reads a package's stored result.json to produce an engine's answer.
    """
    from opendarts.capture.replay import replay_throw_with_engine
    from opendarts.capture.throw_package import load_throw_package
    from opendarts.live.server import _match_fields_for_section, _operator_truth_for

    root = _corpus_root()
    if not root.exists():
        pytest.skip(f"no real corpus at {root} -- this measurement needs real throws")

    pkg_dirs = sorted(root.glob("*/throw_*"))
    if not pkg_dirs:
        pytest.skip(f"corpus at {root} has no throw packages")

    engines = ["Apollo", "Talos", "Athena"]
    cross = {name: {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0} for name in engines}

    for pkg_dir in pkg_dirs:
        try:
            package = load_throw_package(pkg_dir)
        except Exception:
            continue
        truth = _operator_truth_for(package.ad_ground_truth) or package.ad_ground_truth

        for engine_name in engines:
            try:
                result = replay_throw_with_engine(package, engine_name)
            except Exception:
                continue
            if not result.ok or result.board_xy_mm is None:
                continue

            sanity = evaluate_color_sanity(
                result.sector, result.ring, result.board_xy_mm,
                package.calibrations, package.bg_frames,
            )
            if sanity.agrees is None:
                continue

            section = {
                "sector": result.sector,
                "ring": result.ring,
                "board_xy_mm": list(result.board_xy_mm),
            }
            truth_match, _ = _match_fields_for_section(section, truth)
            if truth_match is None:
                continue

            cross[engine_name][(sanity.agrees, truth_match)] += 1

    # At least one engine must have SOME real signal to validate against.
    total_checked = sum(sum(c.values()) for c in cross.values())
    if total_checked == 0:
        pytest.skip(f"corpus at {root} produced no (color-checked, truth-checked) throws")

    any_engine_shows_real_lift = False
    details = []
    for name in engines:
        c = cross[name]
        n_disagree = c[(False, True)] + c[(False, False)]
        n_agree = c[(True, True)] + c[(True, False)]
        p_miss_given_disagree = c[(False, False)] / n_disagree if n_disagree else None
        p_miss_given_agree = c[(True, False)] / n_agree if n_agree else None
        details.append((name, n_disagree, p_miss_given_disagree, n_agree, p_miss_given_agree))
        if p_miss_given_disagree is not None and p_miss_given_agree is not None:
            if p_miss_given_disagree > p_miss_given_agree:
                any_engine_shows_real_lift = True

    assert any_engine_shows_real_lift, (
        "expected at least one engine to show P(miss | color disagrees) > "
        f"P(miss | color agrees) on the real corpus; got: {details}"
    )
