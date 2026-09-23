"""Tests for the sub-component salvage fix (2026-09-02) --
opendarts/engines/athena/crossing_detection.py's
`_largest_elongated_subcomponent_points()` /
`MIN_SALVAGED_SUBCOMPONENT_AREA_PX`.

Independent Athena re-implementation of the identical fix shipped in
`opendarts.engines.apollo.tip_detection` -- see
`tests/test_engine_apollo_component_salvage.py`'s own module
docstring for the full real-incident write-up
(the recorded outside throw, "there is a bug in the apollo and athena
engines that made it not find the right tip"). Confirmed reproducible in
BOTH engines from the same real frame bytes despite genuinely
independent implementations (Athena's own `crossing_detection.py`
module docstring: "NOT a port of `opendarts.engines.apollo.
tip_detection`... a fresh implementation with its own constants") -- cam0
locked onto the same wrong region in both.

The area-floor tightening is likewise independently re-measured for this
module (not copied blind from Apollo's own number, though it landed on
the identical 3000px value): a first draft gated the sub-piece purely on
this module's own generic `MIN_COMPONENT_AREA_PX` (18px) floor and was
found, via this project's own local corpus regression check
(the session corpus, 951 throws), to let a real throw
(the recorded S12 throw) pick a spurious 76px noise fragment over
the pre-existing "whole blob" fallback, which had that throw right.

Synthetic geometry below was MEASURED first (an exploratory script,
not shipped, run against Apollo's own module -- Athena shares the identical frame-spanning-candidate filter
and morphological-bridging shape, confirmed directly against THIS module
too before shipping): a directly-TOUCHING dart+artifact pair fuses into
one solid blob with no disjoint sub-piece to salvage, and a too-small
canvas makes the merged blob's own bounding box exceed the "more than
60% of the frame -> lighting drift, not a dart" rejection filter both
this module and Apollo's own share -- both would have made an earlier
draft of this test pass vacuously, testing a scenario the real pipeline
never even considers.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import opendarts.engines.athena.crossing_detection as cd
from opendarts.engines.athena.crossing_detection import detect_crossing

def _fixtures_root() -> Path:
    """Real-data fixture frames live outside this repo (they are large
    binaries). Set OPENDARTS_FIXTURES_ROOT to enable the tests that use
    them; without it those tests skip."""
    env = os.environ.get("OPENDARTS_FIXTURES_ROOT")
    return Path(env) if env else Path(__file__).parent / "fixtures"


FIXTURE_DIR = _fixtures_root() / "apollo_athena_wrong_tip_20260902"


def _merged_blob_pair():
    """Same validated shape as Apollo's own
    test_engine_apollo_component_salvage.py::_merged_blob_pair -- see
    that function's own docstring for the full derivation. A real
    camera-frame-sized canvas (720x1280), a genuinely elongated dart
    separated from a much larger low-elongation artifact by a real 15px
    gap (bridged by this module's own `CLOSE_KERNEL_PX`, but disjoint at
    native resolution), plus a separate small unrelated elongated "wrong
    candidate" blob."""
    import cv2

    h, w = 720, 1280
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    frame = bg.copy()
    artifact_top, artifact_bottom = 30, 250
    gap = 15
    dart_top, dart_bottom = artifact_bottom + gap, artifact_bottom + gap + 150
    cv2.rectangle(frame, (250, artifact_top), (650, artifact_bottom), (200, 200, 200), -1)
    cv2.rectangle(frame, (440, dart_top), (460, dart_bottom), (200, 200, 200), -1)
    cv2.rectangle(frame, (800, 300), (820, 350), (200, 200, 200), -1)
    return bg, frame, (dart_top, dart_bottom)


def test_sub_component_salvage_recovers_the_true_dart_over_an_unrelated_elongated_artifact():
    """Same real mechanism as Apollo's own test -- see this module's
    docstring for the full write-up."""
    bg, frame, (dart_top, dart_bottom) = _merged_blob_pair()
    result = detect_crossing(bg, frame)
    assert result.ok, result.reason

    x, y = result.tip_px
    assert 400 <= x <= 500 and dart_top - 20 <= y <= dart_bottom + 20, (
        f"expected the detected tip pixel ({x:.1f}, {y:.1f}) to land on "
        "the true dart's own region, not the unrelated wrong candidate "
        "at x~800-820 -- if this fails, the fix regressed to picking the "
        "wrong object again"
    )
    assert (
        result.diagnostics.get("component_area_px", 0)
        >= cd.MIN_SALVAGED_SUBCOMPONENT_AREA_PX
    )

    # Prove the fix was actually load-bearing: with the salvage
    # disabled, detect_crossing() must fall through to the pre-existing
    # "take the largest candidate regardless of elongation" fallback,
    # landing on the merged blob's own extreme end -- NOT the true
    # dart's own tip (reproducing the pre-fix bug).
    orig = cd._largest_elongated_subcomponent_points
    cd._largest_elongated_subcomponent_points = lambda *a, **k: None
    try:
        result_without_fix = detect_crossing(bg, frame)
    finally:
        cd._largest_elongated_subcomponent_points = orig
    assert result_without_fix.ok
    wx, wy = result_without_fix.tip_px
    in_dart_region = 400 <= wx <= 500 and dart_top - 20 <= wy <= dart_bottom + 20
    assert not in_dart_region, (
        f"expected the pre-fix (salvage disabled) behavior to NOT land "
        f"on the true dart's own region, got ({wx:.1f}, {wy:.1f}) -- if "
        "this assertion fails, the synthetic scenario no longer "
        "demonstrates the bug this test exists to pin"
    )


def test_sub_component_salvage_ignores_a_small_noise_fragment_below_the_area_floor():
    """Regression pin for the area-floor tightening itself (see this
    module's own docstring for the real local-corpus regression a
    too-permissive floor caused, the recorded S12 throw). A real
    disjoint but TINY (well under the area floor) elongated fragment,
    separated by a real gap from a much larger low-elongation artifact,
    must NOT be salvaged."""
    import cv2

    h, w = 900, 900
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    frame = bg.copy()
    cv2.rectangle(frame, (250, 50), (650, 400), (200, 200, 200), -1)
    # A real gap, then a TINY elongated sliver (well under the area
    # floor, but easily clears MIN_ELONGATION_RATIO on its own).
    cv2.rectangle(frame, (446, 415), (454, 455), (200, 200, 200), -1)

    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cd._diff_mask(bg_gray, frame_gray)
    sub = cd._largest_elongated_subcomponent_points(mask.astype(bool))
    assert sub is None, (
        "expected no qualifying sub-component below the area floor -- "
        f"got one anyway: {len(sub) if sub is not None else None}px"
    )


@pytest.mark.skipif(not FIXTURE_DIR.is_dir(), reason="real fixture frames not present")
def test_real_20260902_g4_113_out_cam0_locates_the_true_dart():
    """Real-data regression pin, same fixture Apollo's own test uses
    (tests/fixtures/apollo_athena_wrong_tip_20260902/) -- Athena's own
    independent detector must also now find the real dart on this exact
    camera's real bytes, not just Apollo's."""
    import cv2

    bg = cv2.imread(str(FIXTURE_DIR / "cam0_bg.png"))
    frame = cv2.imread(str(FIXTURE_DIR / "cam0_frame.png"))
    assert bg is not None and frame is not None, "fixture frames failed to load"

    result = detect_crossing(bg, frame)
    assert result.ok, result.reason

    true_tip = (647.4, 476.7)
    x, y = result.tip_px
    dist = float(np.hypot(x - true_tip[0], y - true_tip[1]))
    assert dist < 60.0, (
        f"expected the detected tip ({x:.1f}, {y:.1f}) within 60px of "
        f"the real reference tip {true_tip}, got {dist:.1f}px"
    )
