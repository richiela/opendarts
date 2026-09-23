"""Tests for opendarts.geometry.board_color_calibration -- per-installation
derivation of board_color.py's raw pixel classification thresholds
(BRIGHTNESS_THRESHOLD_BLACK_CREAM, CHROMA_THRESHOLD) from real sampled
pixels, using the known-universal color phase as ground truth.

Layers, matching the module's own structure:
1. Reference geometry (reference_points()) -- pure lookup, no images.
2. Sample collection (collect_color_samples()) -- synthetic solid-color
   images + monkeypatched projection, same pattern
   tests/test_board_color.py already uses.
3. Threshold derivation (_derive_gap_threshold()/derive_thresholds()) --
   synthetic ColorSample pools covering clean separation, low-confidence
   overlap, and insufficient-data edge cases.
4. Session-level entry points (derive_session_board_color_calibration(),
   write/load sibling-file round trip) -- real-corpus-gated where a real
   session is needed, tmp_path-isolated where it isn't.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from opendarts.geometry.board import SECTOR_NUMBERS_CLOCKWISE
from opendarts.geometry.board_color import PATCH_RADIUS_PX, expected_color
from opendarts.geometry.board_color_calibration import (
    BOARD_COLOR_CALIBRATION_FILENAME,
    SCHEMA,
    BoardColorCalibrationResult,
    ColorSample,
    ThresholdDerivation,
    _derive_gap_threshold,
    collect_color_samples,
    derive_thresholds,
    load_session_board_color_calibration,
    reference_points,
    result_to_payload,
)

# ---------------------------------------------------------------------
# 1. Reference geometry
# ---------------------------------------------------------------------


def test_reference_points_count_and_shape():
    """20 sectors x 4 bands (single_inner/single_outer/treble/double)
    + bull + outer_bull = 82."""
    points = reference_points()
    assert len(points) == 20 * 4 + 2


def test_reference_points_expected_colors_match_board_color_module():
    """Every reference point's expected_color must agree with
    board_color.expected_color() itself -- this module never invents
    its own color-phase logic, it only reuses the public function."""
    points = reference_points()
    for pt in points:
        sector_str = str(pt.sector) if pt.sector is not None else None
        assert pt.expected_color == expected_color(sector_str, pt.ring)


def test_reference_points_group_assignment():
    points = reference_points()
    for pt in points:
        if pt.ring in ("single_inner", "single_outer"):
            assert pt.group == "achromatic"
        else:
            assert pt.group == "chromatic"


def test_reference_points_cover_every_sector():
    points = reference_points()
    sectors_seen = {pt.sector for pt in points if pt.sector is not None}
    assert sectors_seen == set(SECTOR_NUMBERS_CLOCKWISE)


# ---------------------------------------------------------------------
# 2. Sample collection
# ---------------------------------------------------------------------


def _solid_image(bgr: tuple[float, float, float], size: int = 40) -> np.ndarray:
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


class _FakeCalib:
    pass


def test_collect_color_samples_uses_projection_and_patch_sampler(monkeypatch):
    import opendarts.geometry.board_color_calibration as mod

    img = _solid_image((86.1, 72.2, 67.0))  # a real black-ish BGR sample
    monkeypatch.setattr(mod, "project_board_point_px", lambda xy, calib: (20.0, 20.0))

    points = reference_points()[:5]
    samples = collect_color_samples("pkg1", {0: _FakeCalib()}, {0: img}, points=points)
    assert len(samples) == 5
    for s in samples:
        assert s.package_id == "pkg1"
        assert s.camera == 0
        # uint8 image storage truncates the fractional BGR -- tolerance
        # reflects that, not floating point noise.
        assert s.b == pytest.approx(86.1, abs=1.0)
        assert s.g == pytest.approx(72.2, abs=1.0)
        assert s.r == pytest.approx(67.0, abs=1.0)


def test_collect_color_samples_skips_camera_missing_bg_image(monkeypatch):
    import opendarts.geometry.board_color_calibration as mod

    monkeypatch.setattr(mod, "project_board_point_px", lambda xy, calib: (20.0, 20.0))
    points = reference_points()[:3]
    samples = collect_color_samples("pkg1", {0: _FakeCalib(), 1: _FakeCalib()}, {0: _solid_image((1, 2, 3))}, points=points)
    assert all(s.camera == 0 for s in samples)


def test_collect_color_samples_skips_off_frame_projection(monkeypatch):
    import opendarts.geometry.board_color_calibration as mod

    monkeypatch.setattr(mod, "project_board_point_px", lambda xy, calib: (-999.0, -999.0))
    points = reference_points()[:4]
    samples = collect_color_samples("pkg1", {0: _FakeCalib()}, {0: _solid_image((1, 2, 3))}, points=points)
    assert samples == []


def test_color_sample_brightness_and_chroma_properties():
    s = ColorSample(package_id="p", camera=0, sector=20, ring="single_inner", expected_color="black", group="achromatic", b=10.0, g=20.0, r=30.0)
    assert s.brightness == pytest.approx(20.0)
    assert s.chroma == pytest.approx(20.0)  # max(30)-min(10)


# ---------------------------------------------------------------------
# 3. Threshold derivation
# ---------------------------------------------------------------------


def test_derive_gap_threshold_insufficient_data_when_a_group_is_empty():
    result = _derive_gap_threshold([], [10.0, 20.0])
    assert result.confidence == "insufficient_data"
    assert result.value is None


def test_derive_gap_threshold_clean_separation_is_high_confidence():
    low = [10.0, 12.0, 11.0, 13.0, 9.0]
    high = [200.0, 210.0, 195.0, 205.0, 198.0]
    result = _derive_gap_threshold(low, high)
    assert result.confidence == "high"
    assert result.gap > 0
    assert low[-1] <= result.value <= high[0] or result.value < min(high)  # midpoint sits between the groups
    assert result.low_group_n == 5
    assert result.high_group_n == 5


def test_derive_gap_threshold_outlier_robust_percentile_beats_naive_minmax():
    """The real reason this module uses a percentile-trimmed group edge
    instead of literal max()/min() (see module's own ROBUST_PERCENTILE
    docstring/comment): one bad outlier sample on each side should not
    single-handedly flip a genuinely clean separation into 'invalid'.
    Built from the real shape found in this task's own corpus
    validation (one low-group outlier reads high, one high-group
    outlier reads low), not a contrived edge case."""
    # 100 clean low samples clustered near 10, one outlier at 300.
    low = [10.0 + i * 0.01 for i in range(100)] + [300.0]
    # 100 clean high samples clustered near 250, one outlier at 5.
    high = [250.0 + i * 0.01 for i in range(100)] + [5.0]

    naive_low_max = max(low)
    naive_high_min = min(high)
    assert naive_low_max > naive_high_min  # literal max/min would "invert" -- confirms the setup

    result = _derive_gap_threshold(low, high, robust_percentile=5.0)
    assert result.confidence == "high"
    assert result.gap > 0


def test_derive_thresholds_from_synthetic_clean_samples():
    """A fully synthetic, hand-built sample pool -- no images, no real
    corpus -- exercising derive_thresholds() end to end: 2 packages x 2
    cameras x (10 black-single + 10 cream-single + 10 red-treble + 10
    green-double), with clearly separated BGR values."""
    samples: list[ColorSample] = []
    for pkg in ("pkg0", "pkg1"):
        for cam in (0, 1):
            for i in range(10):
                # black single (low brightness, achromatic)
                samples.append(ColorSample(pkg, cam, 20, "single_inner", "black", "achromatic", 60.0 + i, 60.0 + i, 60.0 + i))
                # cream single (high brightness, achromatic)
                samples.append(ColorSample(pkg, cam, 1, "single_inner", "cream", "achromatic", 240.0 + i * 0.1, 240.0 + i * 0.1, 240.0 + i * 0.1))
                # red treble (chromatic, r > g)
                samples.append(ColorSample(pkg, cam, 20, "treble", "red", "chromatic", 60.0, 60.0, 220.0 + i))
                # green double (chromatic, g > r)
                samples.append(ColorSample(pkg, cam, 1, "double", "green", "chromatic", 60.0, 220.0 + i, 60.0))

    result = derive_thresholds(samples)
    assert result.brightness_threshold.confidence == "high"
    assert result.chroma_threshold.confidence == "high"
    # Self-consistency accuracy should be perfect on this clean synthetic set.
    assert result.accuracy_single_camera == pytest.approx(1.0)
    assert result.accuracy_majority_vote == pytest.approx(1.0)
    assert result.n_samples == len(samples)
    assert result.n_packages == 2
    assert result.warnings == []


def test_derive_thresholds_empty_input_reports_insufficient_data_not_a_crash():
    result = derive_thresholds([])
    assert result.brightness_threshold.confidence == "insufficient_data"
    assert result.chroma_threshold.confidence == "insufficient_data"
    assert result.accuracy_single_camera is None
    assert result.accuracy_majority_vote is None
    assert result.n_samples == 0


def test_derive_thresholds_to_dict_is_json_serializable():
    samples = [
        ColorSample("p", 0, 20, "single_inner", "black", "achromatic", 60.0, 60.0, 60.0),
        ColorSample("p", 0, 1, "single_inner", "cream", "achromatic", 240.0, 240.0, 240.0),
        ColorSample("p", 0, 20, "treble", "red", "chromatic", 60.0, 60.0, 220.0),
        ColorSample("p", 0, 1, "double", "green", "chromatic", 60.0, 220.0, 60.0),
    ]
    result = derive_thresholds(samples)
    json.dumps(result.to_dict())  # must not raise


def test_load_session_board_color_calibration_returns_none_when_absent(tmp_path):
    session = tmp_path / "empty_session"
    session.mkdir()
    assert load_session_board_color_calibration(session) is None


def test_load_session_board_color_calibration_returns_none_on_corrupt_json(tmp_path):
    """Real gap (OpenDarts pre-hardware-audit finding, confirmed to
    apply here too, 2026-08-21) -- identical to `opendarts.calibration.
    ring_boundary_offset`'s own sibling fix: a truncated/corrupt file
    used to raise straight through load_throw_package(), before
    set_board_color_thresholds() ever ran, defeating the "never leak a
    prior session's thresholds into a later load" guarantee. Must
    degrade exactly like "no file at all", not crash."""
    session = tmp_path / "corrupt_session"
    session.mkdir()
    (session / BOARD_COLOR_CALIBRATION_FILENAME).write_text("{not valid json")
    assert load_session_board_color_calibration(session) is None


# ---------------------------------------------------------------------
# THE V2 PACKAGE SCHEMA (2026-08-27) -- result_to_payload() made
# public so bootstrap_calibrations() and the offline session writer
# (dev/calibration/board_color_session.py) build the SAME payload shape.
# ---------------------------------------------------------------------


def _empty_result() -> BoardColorCalibrationResult:
    empty_derivation = ThresholdDerivation(
        value=None, low_group_stat=None, high_group_stat=None,
        low_group_n=0, high_group_n=0, gap=None, confidence="insufficient_data",
    )
    return BoardColorCalibrationResult(
        brightness_threshold=empty_derivation, chroma_threshold=empty_derivation,
        patch_radius=PATCH_RADIUS_PX, n_samples=0, n_packages=0, n_packages_attempted=0,
        accuracy_single_camera=None, accuracy_majority_vote=None,
    )


def test_result_to_payload_matches_write_session_own_shape():
    """Both writers -- the live calibration event and the offline
    session writer -- go through this function, so its output IS the
    written shape: same schema, same solved_by default,
    `**result.to_dict()` merged in at the top level, not nested under a
    sub-key."""
    result = _empty_result()
    payload = result_to_payload(result)
    assert payload["schema"] == SCHEMA
    assert "offline" in payload["solved_by"]
    assert payload == {"schema": SCHEMA, "solved_by": payload["solved_by"], **result.to_dict()}


def test_result_to_payload_solved_by_override_is_used_verbatim():
    payload = result_to_payload(_empty_result(), solved_by="a live custom description")
    assert payload["solved_by"] == "a live custom description"
