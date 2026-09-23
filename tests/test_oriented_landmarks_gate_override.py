"""Regression tests for `oriented_landmarks`' phase-confidence gate
being OVERRIDABLE.

Why this file exists (a real bug, not a hypothetical). `MIN_PHASE_CONFIDENCE
= 2.1` was chosen by a real threshold sweep plus leave-one-session-out
cross-validation over 505 real archived calibration frames. Once that
value was baked into `find_oriented_landmarks()` with no way past it, the
very scripts that produced it stopped working: they only looked at frames
where `result.ok` was True, so every frame they could see had ALREADY
passed 2.1, and sweeping candidate thresholds against that pre-filtered
pool measured nothing -- every candidate at or below 2.1 gave an
identical answer, and the frames a looser gate would have admitted were
invisible. The tuning was circular, silently.

The fix is the `min_phase_confidence=` parameter. These tests pin down
the property that makes it real -- **the override actually changes which
frames are accepted** -- so nobody can reintroduce a hardcoded,
un-overridable gate without a red test.

All synthetic: a real `cv2.projectPoints` board render, progressively
blurred to produce a real, measured spread of phase confidences. No
archived data required.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from opendarts.calibration import oriented_landmarks
from opendarts.calibration.oriented_landmarks import (
    MIN_PHASE_CONFIDENCE,
    correspond_landmarks_oriented,
    find_oriented_landmarks,
)
from tests.test_oriented_landmarks import (
    render_synthetic_board,
    synthetic_board_homography,
)


def _board_image(cam_index: int = 0, blur: int = 0) -> np.ndarray:
    H_mm, _ = synthetic_board_homography(cam_index)
    img = render_synthetic_board(H_mm)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img


# Measured on this render (cam0), gate disabled, printed before this test
# was written per this project's "measure the real number first" rule:
#   blur  0 -> confidence 2.678, colour margin 1.000
#   blur  3 -> 2.605 / 1.000
#   blur  9 -> 2.360 / 1.000
#   blur 21 -> 1.920 / 0.174
#   blur 41 -> 1.669 / 0.259
# Every one of those clears the colour-margin gate (0.05), so phase
# confidence is the ONLY thing separating them -- which is exactly what a
# gate-override test needs.
_BLURS = (0, 3, 9, 21, 41)


def test_override_changes_which_frames_are_accepted():
    """THE regression assertion. A sweep of candidate thresholds over a
    fixed set of real frames must produce DIFFERENT accepted sets -- that
    is the whole thing an un-overridable gate destroys."""
    images = [_board_image(blur=b) for b in _BLURS]
    accepted_by_threshold = {}
    for threshold in (0.0, 1.8, 2.1, 2.5, 3.0):
        accepted_by_threshold[threshold] = [
            find_oriented_landmarks(img, min_phase_confidence=threshold).ok
            for img in images
        ]

    counts = {t: sum(v) for t, v in accepted_by_threshold.items()}
    # Monotone: a stricter gate can never accept more frames.
    thresholds = sorted(counts)
    for lo, hi in zip(thresholds, thresholds[1:]):
        assert counts[hi] <= counts[lo], counts
    # And genuinely varying -- the exact failure mode this file guards.
    assert counts[0.0] == len(images), counts
    assert counts[3.0] == 0, counts
    assert len(set(counts.values())) >= 3, counts


def test_disabled_gate_reports_the_real_confidence_of_a_frame_the_default_rejects():
    """With the gate off, a frame the SHIPPED gate rejects still comes
    back ok, carrying its real sub-threshold confidence -- the raw number
    a tuning script has to be able to see."""
    img = _board_image(blur=41)

    default = find_oriented_landmarks(img)
    assert not default.ok
    assert "phase lock untrustworthy" in default.reason
    assert default.phase_confidence < MIN_PHASE_CONFIDENCE

    raw = find_oriented_landmarks(img, min_phase_confidence=0.0)
    assert raw.ok, raw.reason
    assert raw.phase_confidence == pytest.approx(default.phase_confidence)
    assert raw.phase_confidence < MIN_PHASE_CONFIDENCE
    assert raw.quad_px is not None
    # The override touches the GATE only -- every stage above it must
    # produce byte-identical geometry.
    assert np.array_equal(raw.quad_px, default.quad_px)
    assert raw.phase_deg == pytest.approx(default.phase_deg)


def test_gate_is_read_from_the_module_constant_not_hardcoded(monkeypatch):
    """`None` must mean "whatever `MIN_PHASE_CONFIDENCE` currently says",
    resolved per call. If someone inlines the number again, this fails."""
    img = _board_image()
    baseline = find_oriented_landmarks(img, min_phase_confidence=0.0)
    assert baseline.ok

    monkeypatch.setattr(
        oriented_landmarks, "MIN_PHASE_CONFIDENCE", baseline.phase_confidence + 0.1
    )
    assert not find_oriented_landmarks(img).ok
    # ... and the override still wins over the (now stricter) constant.
    assert find_oriented_landmarks(img, min_phase_confidence=0.0).ok

    monkeypatch.setattr(
        oriented_landmarks, "MIN_PHASE_CONFIDENCE", baseline.phase_confidence - 0.1
    )
    assert find_oriented_landmarks(img).ok


def test_a_threshold_just_above_a_frames_own_confidence_rejects_it():
    """Both sides of the same frame's real confidence, so the test can
    only pass if the parameter is genuinely consulted."""
    img = _board_image()
    confidence = find_oriented_landmarks(img, min_phase_confidence=0.0).phase_confidence
    assert find_oriented_landmarks(img, min_phase_confidence=confidence - 0.01).ok
    rejected = find_oriented_landmarks(img, min_phase_confidence=confidence + 0.01)
    assert not rejected.ok
    assert f"{confidence:.2f}" in rejected.reason


def test_correspond_landmarks_oriented_forwards_the_override():
    """The production wrapper must expose the same escape hatch -- a
    tuning script that goes through the correspondence API instead of the
    finder API must not be silently gated either."""
    img = _board_image(blur=41)
    hints = {0: _hint_for(img)}
    assert correspond_landmarks_oriented(img, 0, orientation_hints_deg=hints) is None
    got = correspond_landmarks_oriented(
        img, 0, orientation_hints_deg=hints, min_phase_confidence=0.0
    )
    assert got is not None
    obj, px = got
    assert obj.shape == (4, 3)
    assert px.shape == (4, 2)


def _hint_for(image_bgr: np.ndarray) -> float:
    """The real orientation hint for a synthetic render, derived the same
    way the rig's own constants are (`derive_orientation_hint_deg`)."""
    H_mm, _ = synthetic_board_homography(0)
    return oriented_landmarks.derive_orientation_hint_deg(image_bgr, _as_3x3(H_mm))


def _as_3x3(H_mm) -> np.ndarray:
    return np.asarray(H_mm, dtype=np.float64).reshape(3, 3)
