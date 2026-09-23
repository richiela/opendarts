"""Tests for `opendarts.calibration.ring_correlation_orientation`.

Real-image tests (marked `@pytest.mark.slow` -- each
`solve_frame_orientation()` call is a real cv2 pipeline, not mocked) use
reference photographs kept outside this repo; set
`OPENDARTS_FIXTURES_ROOT` to enable them, otherwise they skip. The
pure-math tests (angle-convention derivation, aggregation/agreement
logic against hand-built inputs) always run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import os

import pytest

import opendarts.calibration.ring_correlation_orientation as rco

CALIB_TEST_ROOT = Path(
    os.environ.get("OPENDARTS_FIXTURES_ROOT", "/nonexistent/reference-images")
) / "calib_reference"
HAS_CALIB_TEST_IMAGES = CALIB_TEST_ROOT.is_dir() and any(
    CALIB_TEST_ROOT.glob("set_*/*.png")
)

# Recorded reference values (image, angle_degrees_clockwise_
# from_up, confidence) -- the exact numbers this port must reproduce, to
# within real floating-point/library-version noise, not byte-identically.
REFERENCE_VALUES = {
    "set_a_2026-08-27/cam0.png": (178.1, 0.117),
    "set_a_2026-08-27/cam1.png": (74.1, 0.139),
    "set_a_2026-08-27/cam2.png": (285.0, 0.056),
    "set_b_2026-08-28/cam0.png": (3.0, 0.111),
    "set_b_2026-08-28/cam1.png": (105.0, 0.161),
    "set_b_2026-08-28/cam2.png": (256.3, 0.075),
}


# ---------------------------------------------------------------------
# Pure-math tests -- no real images needed, always run.
# ---------------------------------------------------------------------


def test_imgang_matches_hint_deg_from_clockwise_from_up_convention():
    """Independent re-derivation check (this module's own docstring's
    "ANGLE CONVENTION" section) -- imgang() at the four cardinal
    directions must equal the "0=up, 90=right clockwise" convention this
    rig's orientation hints have always used."""
    assert rco.imgang(0.0, -1.0) == pytest.approx(0.0)      # up
    assert rco.imgang(1.0, 0.0) == pytest.approx(90.0)       # right
    assert rco.imgang(0.0, 1.0) == pytest.approx(180.0)      # down
    assert rco.imgang(-1.0, 0.0) == pytest.approx(270.0)     # left


def test_seq_matches_regulation_sequence_and_board_module():
    from opendarts.geometry.board import SECTOR_NUMBERS_CLOCKWISE
    assert rco.SEQ == SECTOR_NUMBERS_CLOCKWISE
    assert rco.SEQ[:5] == [20, 1, 18, 4, 13]


def test_digits_template_matches_one_and_two_digit_numbers():
    # index 0 is "20" (2 digits), index 1 is "1" (1 digit)
    assert rco.DIGITS[0] == 2.0
    assert rco.DIGITS[1] == 1.0


def _fake_result(angle_degrees: float, answered: bool = True, corr_best: float = 0.9) -> rco.RingCorrelationResult:
    return rco.RingCorrelationResult(
        target_sector=0, angle_degrees=angle_degrees, confidence=0.2,
        answered=answered, corr_best=corr_best, flags=[], hard_fails=[],
        bull_px=(0.0, 0.0), debug={},
    )


def test_ring_correlation_result_hint_deg_property_matches_module_function():
    r = _fake_result(90.0)
    assert r.hint_deg == pytest.approx(rco.hint_deg_from_clockwise_from_up(90.0))


def test_ring_correlation_orientation_for_camera_empty_frames():
    result = rco.ring_correlation_orientation_for_camera([])
    assert result.ok is False
    assert result.hint_deg is None
    assert result.n_frames == 0
    assert result.per_frame == []


def test_ring_correlation_orientation_for_camera_bad_frame_declines_gracefully():
    # A blank frame has no real board geometry -- solve_frame_orientation()
    # must raise RuntimeError, and the aggregation function must catch it
    # per-frame rather than propagating.
    blank = np.zeros((200, 300, 3), dtype=np.uint8)
    result = rco.ring_correlation_orientation_for_camera([blank])
    assert result.ok is False
    assert result.hint_deg is None
    assert result.n_frames == 1
    assert result.n_passed == 0
    assert len(result.per_frame) == 1
    assert result.per_frame[0].result is None
    assert result.per_frame[0].error is not None
    assert result.per_frame[0].hint_deg is None


def test_solve_frame_orientation_normalizes_any_internal_exception_to_runtimeerror():
    """Porting-adaptation contract (this module's own docstring): every
    internal failure mode, whatever its native exception type, must
    surface as RuntimeError so a burst-processing caller only ever needs
    to catch one exception type."""
    blank = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        rco.solve_frame_orientation(blank)


def test_ring_correlation_orientation_for_camera_agreement_required_not_just_pass_count(monkeypatch):
    """Same real safety property DOHS2's own aggregation test exercises:
    frames that each individually pass but DISAGREE with each other must
    NOT count toward the aggregate pass fraction -- only mutually
    agreeing frames do."""
    angles_clockwise_from_up = [
        100.0, 101.0, 99.0,   # cluster A: hint_deg ~= 10, 11, 9
        290.0, 291.0,         # cluster B: hint_deg ~= 200, 201
    ]
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in angles_clockwise_from_up]
    results_by_id = {id(f): _fake_result(a) for f, a in zip(frames, angles_clockwise_from_up)}

    def fake_solve(image_bgr):
        return results_by_id[id(image_bgr)]

    monkeypatch.setattr(rco, "solve_frame_orientation", fake_solve)

    result = rco.ring_correlation_orientation_for_camera(frames, min_pass_fraction=0.5)
    assert result.n_passed == 5
    assert result.n_agreeing == 3
    assert result.pass_fraction == pytest.approx(3 / 5)
    assert result.ok is True
    assert result.hint_deg == pytest.approx(10.0, abs=1.0)


def test_ring_correlation_orientation_for_camera_min_pass_fraction_gate(monkeypatch):
    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(5)]
    angles = [100.0, 101.0, 99.0, 290.0, 291.0]
    results_by_id = {id(f): _fake_result(a) for f, a in zip(frames, angles)}
    monkeypatch.setattr(rco, "solve_frame_orientation", lambda img: results_by_id[id(img)])

    result = rco.ring_correlation_orientation_for_camera(frames, min_pass_fraction=0.8)
    assert result.ok is False
    assert result.hint_deg is None
    assert result.pass_fraction == pytest.approx(0.6)
    assert result.majority_hint_deg == pytest.approx(10.0, abs=1.0)


def test_ring_correlation_orientation_for_camera_n1_requires_the_one_frame_to_pass(monkeypatch):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        rco, "solve_frame_orientation", lambda img: _fake_result(100.0, answered=False)
    )
    result = rco.ring_correlation_orientation_for_camera([frame])  # default min_pass_fraction=1.0
    assert result.ok is False
    assert result.n_passed == 0
    assert result.pass_fraction == 0.0


# ---------------------------------------------------------------------
# Real-image tests -- reference photographs, skipped when absent.
# ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_calib_test_results():
    if not HAS_CALIB_TEST_IMAGES:
        pytest.skip(f"real calib_test images not found under {CALIB_TEST_ROOT}")
    import cv2

    out = {}
    for rel_path in REFERENCE_VALUES:
        path = CALIB_TEST_ROOT / rel_path
        image = cv2.imread(str(path))
        assert image is not None, f"could not read {path}"
        out[rel_path] = rco.solve_frame_orientation(image)
    return out


@pytest.mark.slow
@pytest.mark.parametrize("rel_path", sorted(REFERENCE_VALUES))
def test_solve_frame_orientation_reproduces_reference_values(
    real_calib_test_results, rel_path,
):
    """Must reproduce the recorded reference angle/confidence for all 6
    real images -- a regression pin on the solver's real-image output."""
    expected_angle, expected_confidence = REFERENCE_VALUES[rel_path]
    result = real_calib_test_results[rel_path]
    assert result.answered is True
    assert result.angle_degrees == pytest.approx(expected_angle, abs=0.5)
    assert result.confidence == pytest.approx(expected_confidence, abs=0.02)




@pytest.mark.slow
def test_ring_correlation_orientation_for_camera_single_real_frame(real_calib_test_results):
    import cv2

    rel_path = "set_a_2026-08-27/cam0.png"
    image = cv2.imread(str(CALIB_TEST_ROOT / rel_path))
    result = rco.ring_correlation_orientation_for_camera([image])
    expected = real_calib_test_results[rel_path]
    assert result.ok is True
    assert result.n_frames == 1
    assert result.n_passed == 1
    assert result.n_agreeing == 1
    assert result.pass_fraction == pytest.approx(1.0)
    assert result.hint_deg == pytest.approx(expected.hint_deg, abs=0.5)


@pytest.mark.slow
def test_ring_correlation_orientation_for_camera_real_disagreeing_images(real_calib_test_results):
    """Two genuinely different real images (different physical setups,
    different days) fed as one camera's "burst" -- both
    individually pass, but disagree by tens of degrees. The aggregate
    must NOT report high confidence just because both passed."""
    import cv2

    rel_a = "set_a_2026-08-27/cam0.png"
    rel_b = "set_b_2026-08-28/cam1.png"
    img_a = cv2.imread(str(CALIB_TEST_ROOT / rel_a))
    img_b = cv2.imread(str(CALIB_TEST_ROOT / rel_b))
    hint_a = real_calib_test_results[rel_a].hint_deg
    hint_b = real_calib_test_results[rel_b].hint_deg
    real_diff = min(abs(hint_a - hint_b) % 360, 360 - abs(hint_a - hint_b) % 360)
    assert real_diff > rco.AGREEMENT_TOLERANCE_DEG

    result = rco.ring_correlation_orientation_for_camera([img_a, img_b], min_pass_fraction=1.0)
    assert result.n_passed == 2
    assert result.n_agreeing == 1
    assert result.pass_fraction == pytest.approx(0.5)
    assert result.ok is False
