"""Tests for the sub-component salvage fix (2026-09-02) --
opendarts/engines/apollo/tip_detection.py's
`_largest_elongated_subcomponent()` / `MIN_SALVAGED_SUBCOMPONENT_AREA_PX`.

Real incident this pins: the recorded outside throw cam0. Two darts on
the board with adjacent shafts; cam0's diff mask correctly captured the
true, newly-thrown dart's own elongated silhouette (a real 9684px
sub-region), but the 31px dilation `detect_tip()` applies before
connected-component labeling bridged it into the SAME label as ~77 tiny
(5-194px), spatially scattered diff specks along an unrelated board-text/
number-ring band, dragging the WHOLE merged blob's own PCA elongation
down to 1.37 (well under `MIN_ELONGATION_RATIO`'s 2.5). The pre-fix code
correctly rejected that whole merged blob, but had no way to recover the
genuinely elongated dart shaft still sitting inside it -- it fell
straight through to a smaller, entirely unrelated, coincidentally-
elongated component instead (a wrong-OBJECT lock, not a wrong-END-of-
the-right-object mislabeling -- see `opendarts/engines/apollo/
tip_detection.py`'s own module docstring for the full
write-up). Real per-camera evidence, all 3 cameras, confirmed the frames
and calibration were NOT at fault -- Ares/Talos (independent
detectors) found the correct tip on every camera from these exact
bytes.

The area-floor tightening (`MIN_SALVAGED_SUBCOMPONENT_AREA_PX = 3000`,
not the module's own much smaller generic `MIN_COMPONENT_PIXELS = 15`
floor) is itself a real, measured correction, not a guess -- a first
draft of this fix gated the sub-piece purely on `MIN_COMPONENT_PIXELS`
and was found, via this project's own local corpus regression check
(the session corpus, 951 throws), to let 3 separate real
throws pick a spurious 266-670px NOISE fragment (elongated by chance)
over the pre-existing, cruder "take the whole blob regardless of shape"
fallback, which had been getting those 3 throws right. See that
constant's own comment in `tip_detection.py` for the real numbers.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import opendarts.engines.apollo.tip_detection as td
from opendarts.engines.apollo.tip_detection import detect_tip
from opendarts.engines.apollo.board_roi import reject_outside_roi
from opendarts.capture.throw_package import calibration_from_dict

def _fixtures_root() -> Path:
    """Real-data fixture frames live outside this repo (they are large
    binaries). Set OPENDARTS_FIXTURES_ROOT to enable the tests that use
    them; without it those tests skip."""
    env = os.environ.get("OPENDARTS_FIXTURES_ROOT")
    return Path(env) if env else Path(__file__).parent / "fixtures"


FIXTURE_DIR = _fixtures_root() / "apollo_athena_wrong_tip_20260902"


def _merged_blob_pair():
    """A background/frame pair reproducing the real incident's own
    shape, structurally (not pixel-for-pixel): a genuinely elongated
    dart-shaped component (20 wide x 150 tall, area ~3500px after
    morphology) separated from a much larger, wide, low-elongation
    artifact (400 wide x 220 tall) by a real 15px gap -- small enough
    for `DILATE_KERNEL_PX` (31px) to bridge them into ONE connected diff
    component (matching the real incident: the true dart's own diff
    pixels were bridged into the SAME dilated label as unrelated,
    spatially separate artifact specks), but a real enough gap that the
    UN-DILATED footprint keeps them as two genuinely disjoint native
    sub-components -- confirmed directly,
    this task's own exploratory script, not shipped (a first draft used
    directly-TOUCHING rectangles, which fuse into one solid blob with no
    disjoint sub-piece to salvage at all -- caught only by rerunning this
    exact scenario through the real `detect_tip()`, not by inspection,
    per this project's own "measure, don't assume" discipline). Uses a
    REAL camera-frame-sized canvas (720x1280, matching every other real
    camera frame this module processes) -- a smaller canvas made the
    merged blob's own bounding box exceed detect_tip()'s own "more than
    60% of the frame -> treat as lighting drift, not a dart" rejection
    threshold, which silently made an earlier draft of this test
    meaningless (it was testing a scenario the real pipeline never even
    considered as a candidate).

    Plus a SEPARATE, smaller, genuinely elongated but entirely UNRELATED
    "wrong candidate" blob elsewhere in the frame (20 wide x 50 tall)
    that clears the elongation bar on its own and would win without this
    fix, exactly as `comp3` did on the real incident."""
    import cv2

    h, w = 720, 1280
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    frame = bg.copy()
    artifact_top, artifact_bottom = 30, 250
    gap = 15
    dart_top, dart_bottom = artifact_bottom + gap, artifact_bottom + gap + 150
    # A much wider, bulkier artifact...
    cv2.rectangle(frame, (250, artifact_top), (650, artifact_bottom), (200, 200, 200), -1)
    # ...a real gap below it, then the true dart: a narrow, tall
    # rectangle -- disjoint from the artifact at native resolution, but
    # bridged into the same dilated label.
    cv2.rectangle(frame, (440, dart_top), (460, dart_bottom), (200, 200, 200), -1)
    # A separate, smaller, genuinely elongated but UNRELATED component --
    # the real incident's own comp3 stand-in.
    cv2.rectangle(frame, (800, 300), (820, 350), (200, 200, 200), -1)
    return bg, frame, (dart_top, dart_bottom)


def test_sub_component_salvage_recovers_the_true_dart_over_an_unrelated_elongated_artifact():
    """The core mechanism this fix exists for: when the top-area
    candidate (the merged dart+artifact blob) fails the whole-blob
    elongation gate, the fix must recover the genuinely elongated dart
    sub-region living inside it, rather than falling through to a
    smaller, unrelated, coincidentally-elongated component."""
    bg, frame, (dart_top, dart_bottom) = _merged_blob_pair()
    result = detect_tip(bg, frame)
    assert result.ok, result.reason

    x, y = result.tip_px
    # The true dart spans x in [440, 460], y in [dart_top, dart_bottom];
    # the wrong candidate sits entirely at x in [800, 820] -- a tight
    # bound that actually EXCLUDES the wrong candidate's own location
    # (an earlier draft's bound was wide enough to accidentally include
    # both regions and passed even when the fix picked the wrong one).
    assert 400 <= x <= 500 and dart_top - 20 <= y <= dart_bottom + 20, (
        f"expected the detected tip pixel ({x:.1f}, {y:.1f}) to land on "
        "the true dart's own region, not the unrelated wrong candidate "
        "at x~800-820 -- if this fails, the fix regressed to picking the "
        "wrong object again"
    )
    assert result.diagnostics.get("component_area_px", 0) >= td.MIN_SALVAGED_SUBCOMPONENT_AREA_PX

    # Prove the fix was actually load-bearing for this exact shape: with
    # the salvage disabled, detect_tip() must fall through to the
    # pre-existing "take the largest candidate regardless of elongation"
    # fallback, landing on the merged blob's own extreme end -- NOT the
    # true dart's own tip (reproducing the pre-fix bug).
    orig = td._largest_elongated_subcomponent
    td._largest_elongated_subcomponent = lambda *a, **k: None
    try:
        result_without_fix = detect_tip(bg, frame)
    finally:
        td._largest_elongated_subcomponent = orig
    assert result_without_fix.ok
    wx, wy = result_without_fix.tip_px
    in_dart_region = 400 <= wx <= 500 and dart_top - 20 <= wy <= dart_bottom + 20
    assert not in_dart_region, (
        f"expected the pre-fix (salvage disabled) behavior to NOT land on "
        f"the true dart's own region, got ({wx:.1f}, {wy:.1f}) -- if this "
        "assertion fails, the synthetic scenario no longer demonstrates "
        "the bug this test exists to pin"
    )


def test_sub_component_salvage_ignores_a_small_noise_fragment_below_the_area_floor():
    """Regression pin for the area-floor tightening itself (see this
    module's own docstring for the real local-corpus regressions a
    too-permissive floor caused). A merged blob containing only a TINY
    (well under `MIN_SALVAGED_SUBCOMPONENT_AREA_PX`) elongated fragment
    -- real noise-fragment scale, not a dart -- must NOT be salvaged;
    `_largest_elongated_subcomponent` must return None so detect_tip()
    falls through to its existing, pre-fix "take the largest candidate
    regardless of elongation" fallback instead."""
    import cv2

    h, w = 900, 900
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    frame = bg.copy()
    # A big, low-elongation artifact...
    cv2.rectangle(frame, (250, 50), (650, 400), (200, 200, 200), -1)
    # ...touching a TINY elongated sliver (well under the area floor,
    # but easily clears MIN_ELONGATION_RATIO on its own: 8 wide x 40
    # tall, elongation ~5, area ~320px).
    cv2.rectangle(frame, (446, 400), (454, 440), (200, 200, 200), -1)

    mask = td._diff_mask(
        cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
    )
    opened = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (td.OPEN_KERNEL_PX, td.OPEN_KERNEL_PX)),
    )
    sub = td._largest_elongated_subcomponent(opened.astype(bool))
    assert sub is None, (
        "expected no qualifying sub-component below the area floor -- "
        f"got one anyway: {sub if sub is None else sub[3]}px"
    )


@pytest.mark.skipif(not FIXTURE_DIR.is_dir(), reason="real fixture frames not present")
def test_real_20260902_g4_113_out_cam0_locates_the_true_dart():
    """Real-data regression pin, per this project's own established
    "test against real captured data, not synthetic-only" convention for
    engine bugs -- uses the ACTUAL cam0 bg/frame pair from the real
    defect throw (the recorded outside throw, copied into
    tests/fixtures/ so this test is self-contained, matching the
    precedent set by tests/fixtures/missed_dart_top_crop_20260831/).

    Ground truth:
    AD truth tip_xy_mm=(14.35, -154.41); Ares/Talos (independent
    detectors, unaffected by this fix) both found cam0's own true tip
    pixel near (647.4, 476.7) from these exact bytes. Before this fix,
    Apollo's own detect_tip() locked onto a completely different
    object near (948.0, 213.1) -- ~380px away. After this fix, it must
    land close to the real true tip instead."""
    import cv2

    bg = cv2.imread(str(FIXTURE_DIR / "cam0_bg.png"))
    frame = cv2.imread(str(FIXTURE_DIR / "cam0_frame.png"))
    assert bg is not None and frame is not None, "fixture frames failed to load"

    result = detect_tip(bg, frame)
    assert result.ok, result.reason

    true_tip = (647.4, 476.7)
    x, y = result.tip_px
    dist = float(np.hypot(x - true_tip[0], y - true_tip[1]))
    # Measured directly against this exact fixture: the fix lands at
    # (655.5, 451.8), ~26px from the real reference -- generous margin
    # kept here since this is a real, independently-labeled reference
    # pixel, not a synthetic exact-truth shape.
    assert dist < 60.0, (
        f"expected the detected tip ({x:.1f}, {y:.1f}) within 60px of "
        f"the real reference tip {true_tip}, got {dist:.1f}px -- before "
        "this fix, detect_tip() locked onto an unrelated object ~380px "
        "away instead"
    )

    # The pre-fix behavior locked onto an unrelated component and, as a
    # direct DOWNSTREAM consequence, failed the board-ROI gate entirely
    # (the wrong object's pixel sits off the board's projected ROI) --
    # confirm the fixed detection now passes it, using the throw's own
    # real calibration (also copied into the fixture directory).
    import json

    calib_dict = json.loads((FIXTURE_DIR / "calibration.json").read_text())
    cam0_calib = calibration_from_dict(calib_dict["0"])
    gated = reject_outside_roi(result, cam0_calib)
    assert gated.ok, (
        "expected the fixed cam0 detection to pass the board-ROI gate -- "
        f"before this fix it was rejected (diagnostics: {gated.diagnostics})"
    )
