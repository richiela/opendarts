"""TalosEngine.score() contract now that the engine is real (not a stub).

Empty / insufficient cameras: ok=False (needs >=2 shaft planes). Tiny
blank images fail blob/plane reconstruction the same way. No calibrate().
The 3D plane∩board recovery lives in tests/test_engine_talos_geometry.py.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from opendarts.engines.talos import TalosEngine
from opendarts.engines.talos.shaft_line import fit_shaft_line_px
from opendarts.pipeline import CameraCalibration


def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3, dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=np.zeros((3, 1), dtype=np.float64),
        tvec=np.array([[0.0], [0.0], [500.0]], dtype=np.float64),
    )


def test_talos_has_no_calibrate_method():
    # Uses the primary engine's already-solved pose; omitting calibrate()
    # is the contract (engine_has_custom_calibrate checks hasattr).
    engine = TalosEngine()
    assert not hasattr(engine, "calibrate")


def test_talos_empty_inputs_are_ok_false_and_do_not_hang():
    """Measured: score({}, {}, {}) returns in ~5e-6 s with ok=False,
    ring=None (not the old stub's ok=True / ring='outside'). A hang
    would be seconds; 1.0s is a hang detector, not a performance bound."""
    engine = TalosEngine()
    start = time.monotonic()
    result = engine.score({}, {}, {})
    elapsed = time.monotonic() - start
    print(f"Talos.score({{}}, {{}}, {{}}) elapsed={elapsed:.6f}s")
    assert elapsed < 1.0, f"Talos.score() hung: {elapsed:.3f}s"
    assert result.ok is False
    assert result.sector is None
    assert result.ring is None
    assert result.board_xy_mm is None
    assert ">=2 shaft planes" in result.reason
    assert result.diagnostics.get("engine") == "Talos"
    assert result.diagnostics.get("n_planes") == 0


def test_talos_blank_tiny_images_cannot_form_shaft_planes():
    """Same 6x6 all-black / all-white pair the offline-tooling fixture
    uses. Measured: blob/plane reconstruction yields 0 planes, ok=False,
    ring=None -- not a scored miss."""
    engine = TalosEngine()
    bg_images = {
        0: np.zeros((6, 6, 3), dtype=np.uint8),
        1: np.zeros((6, 6, 3), dtype=np.uint8),
    }
    frame_images = {
        0: np.full((6, 6, 3), 255, dtype=np.uint8),
        1: np.full((6, 6, 3), 255, dtype=np.uint8),
    }
    calibration = {0: _fake_calibration(), 1: _fake_calibration()}
    result = engine.score(bg_images, frame_images, calibration)
    assert result.ok is False
    assert result.sector is None
    assert result.ring is None
    assert result.board_xy_mm is None
    assert result.diagnostics.get("engine") == "Talos"
    assert result.diagnostics.get("n_planes") == 0
    assert ">=2 shaft planes" in result.reason


def test_talos_score_does_not_expose_opened_blob_pixels_in_diagnostics():
    """The detector's opened pixel array is an internal handoff only.

    This compact synthetic silhouette is large enough to produce a real
    opened blob and a usable shaft line, while the deliberately identical
    calibrations make the final geometry fail harmlessly.  Either success
    or failure must still leave no thousands-of-pixels array in the
    EngineResult diagnostics.
    """
    height = width = 128
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    frame = bg.copy()
    center_y = height // 2
    cv2.fillPoly(
        frame,
        [
            np.array(
                [
                    [30, center_y + 4],
                    [40, center_y - 4],
                    [70, center_y - 4],
                    [70, center_y + 4],
                ],
                dtype=np.int32,
            )
        ],
        (255, 255, 255),
    )

    fitted = fit_shaft_line_px(bg, frame)
    assert fitted is not None
    assert len(fitted[2]["opened_pts"]) > 0

    calibration = {0: _fake_calibration(), 1: _fake_calibration()}
    result = TalosEngine().score(
        {0: bg, 1: bg},
        {0: frame, 1: frame},
        calibration,
    )

    assert "opened_pts" not in result.diagnostics
    line_diagnostics = result.diagnostics.get("line_px", {})
    assert line_diagnostics
    assert all("opened_pts" not in diag for diag in line_diagnostics.values())
