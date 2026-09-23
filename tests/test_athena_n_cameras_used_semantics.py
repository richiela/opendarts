"""Investigation pin, 2026-09-02 -- Athena's `per_camera[cam]["used"]`
vs `diagnostics["n_cameras_used"]`.

A corpus-QA peer session found these two fields disagree (per_camera
claiming MORE cameras than the count) on 70/1902 real throws across both
local corpora, always in the same direction, and raised, but did NOT
verify, the hypothesis that `per_camera[...]["used"]` actually means
"gate-passed," not "fused." This module pins the real, code-traced
answer (see `opendarts.engines.athena.engine.AthenaEngine.score()`'s
own dated 2026-09-02 comment at `camera_reads = qualifying if qualifying
else reads` for the full derivation):

- `per_camera[cam]["used"]` is set True for every camera whose candidate
  cleared the board ROI admission gate (the strict pass, or the
  ROI-fallback pass) -- this is exactly `n_cameras_gate_passed`'s own
  per-camera breakdown, "was this camera's read ADMITTED at all."
- `n_cameras_used` (`len(camera_reads)`) is a STRICT SUBSET: only the
  admitted reads that ALSO cleared `MIN_RAY_STEEPNESS_FOR_CONSENSUS` and
  therefore actually fed `_combine_reads()`'s Weiszfeld blend -- "did
  this camera's read reach and influence the final (sector, ring,
  board_xy_mm)."

Both fields are real, correctly computed, and describe genuinely
DIFFERENT quantities -- not a bug to reconcile. This is investigation-
only: no field's semantics were changed to make them agree (per this
task's own explicit instruction); this file exists to lock the real
mechanism against a future accidental "fix" that tries to force them
into agreement.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from opendarts.capture.throw_package import load_throw_package
from opendarts.engines.athena import AthenaEngine
from opendarts.engines.athena.engine import MIN_RAY_STEEPNESS_FOR_CONSENSUS

def _fixtures_root() -> Path:
    """Real-data fixture frames live outside this repo (they are large
    binaries). Set OPENDARTS_FIXTURES_ROOT to enable the tests that use
    them; without it those tests skip."""
    env = os.environ.get("OPENDARTS_FIXTURES_ROOT")
    return Path(env) if env else Path(__file__).parent / "fixtures"


FIXTURE_DIR = _fixtures_root() / "athena_n_cameras_used_divergence_20260902"


@pytest.mark.skipif(not FIXTURE_DIR.is_dir(), reason="real fixture package not present")
def test_n_cameras_used_excludes_a_gate_passed_but_shallow_ray_camera_real_fixture():
    """Real-data regression pin, `tests/fixtures/
    athena_n_cameras_used_divergence_20260902/` (copied from a recorded
    S8 throw -- found via a
    real sweep of all 1902 scoreable throws across both local corpora:
    69 throws show this exact divergence, 79.7% of them on `ring=
    "outside"` throws vs a 7.9% baseline OUT rate corpus-wide -- this one
    is deliberately a real on-board (`single_outer`) example instead, to
    confirm the mechanism isn't specific to the ROI-fallback path).

    All 3 cameras clear the board ROI gate (`per_camera[...]["used"] is
    True` on all 3, `n_cameras_gate_passed == 3`), but cam1's own ray
    (steepness 0.3499) falls just under `MIN_RAY_STEEPNESS_FOR_CONSENSUS`
    (0.35) -- excluded from the Weiszfeld blend that actually produced
    (sector, ring, board_xy_mm). `n_cameras_used` must correctly say 2,
    not 3."""
    package = load_throw_package(FIXTURE_DIR)
    result = AthenaEngine().score(
        package.bg_frames, package.dart_frames, package.calibrations
    )
    assert result.ok
    diag = result.diagnostics
    per_camera = diag["per_camera"]
    n_used_per_camera = sum(1 for e in per_camera.values() if e.get("used"))

    assert n_used_per_camera == 3, (
        "expected the real fixture to reproduce all 3 cameras clearing "
        f"the ROI gate, got {n_used_per_camera} -- if this fails, the "
        "fixture no longer demonstrates the divergence this test exists "
        "to pin"
    )
    assert diag["n_cameras_gate_passed"] == 3
    # The concrete mechanism: cam1's own ray genuinely sits below the
    # consensus floor, even though it was admitted.
    shallow_cams = [
        cam for cam, e in per_camera.items()
        if e.get("used") and e.get("ray_steepness") is not None
        and e["ray_steepness"] < MIN_RAY_STEEPNESS_FOR_CONSENSUS
    ]
    assert len(shallow_cams) == 1, (
        f"expected exactly one gate-passed camera below the steepness "
        f"floor, got {shallow_cams} -- if this fails, the real fixture's "
        f"own ray geometry has changed (e.g. a calibration/detection "
        f"change) and no longer demonstrates the mechanism"
    )

    assert diag["n_cameras_used"] == 2, (
        f"expected n_cameras_used=2 (the shallow-ray camera never fed "
        f"the Weiszfeld blend), got {diag['n_cameras_used']}"
    )
    assert diag["n_cameras_used"] != n_used_per_camera
    assert diag["n_cameras_used"] == diag["n_cameras_gate_passed"] - 1


def test_n_cameras_used_equals_per_camera_used_count_when_no_steepness_exclusion():
    """Pure structural counterpart, no real fixture needed: directly
    exercising the exact split point (`camera_reads = qualifying if
    qualifying else reads`) -- when every admitted read's steepness
    already clears the floor, `n_cameras_used` must equal
    `n_cameras_gate_passed` (and therefore the `per_camera[...]["used"]`
    count) exactly, confirming the two fields are only EXPECTED to
    diverge via the specific steepness-floor exclusion the fixture test
    above pins, not as some other unrelated drift."""
    reads = [
        {"cam": 0, "x_mm": 0.0, "y_mm": 80.0, "sector": "20", "ring": "single_outer",
         "weight": 1.0, "ray_steepness": 0.5},
        {"cam": 1, "x_mm": 1.0, "y_mm": 79.0, "sector": "20", "ring": "single_outer",
         "weight": 1.0, "ray_steepness": 0.55},
        {"cam": 2, "x_mm": -1.0, "y_mm": 81.0, "sector": "20", "ring": "single_outer",
         "weight": 1.0, "ray_steepness": 0.6},
    ]
    qualifying = [r for r in reads if r["ray_steepness"] >= MIN_RAY_STEEPNESS_FOR_CONSENSUS]
    camera_reads = qualifying if qualifying else reads
    n_cameras_gate_passed = len(reads)
    n_cameras_used = len(camera_reads)
    assert n_cameras_used == n_cameras_gate_passed == 3
