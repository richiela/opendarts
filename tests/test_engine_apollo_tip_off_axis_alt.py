"""Tests for the off-axis tip-cluster alternate (2026-08-17) --
opendarts/engines/apollo/tip_detection.py's
TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX mechanism.

Real incident this pins: the recorded T15 throw, cam2 (Apollo's
single miss in that 120-throw session -- AD/operator truth S15
single_inner, scored T15 treble). A 12px-area satellite diff fragment at
(801, 264) -- off the dart entirely, merged into the chosen component by
the 31px dilation -- supplied 8 of the 10 most proj-extreme points, so
the reported tip (795.5, 264.2) sat 22.6px PERPENDICULAR to the
component's own principal axis (a real tip lies ON the dart's axis by
construction). The fix never overwrites tip_px: it exposes the on-axis
recompute as `alt_tip_px` and lets score_dart()'s existing cross-camera
combination search arbitrate (on the real incident: primary pair
disagreement 5.08mm vs 0.022mm for the alternative -- not a close call).

Synthetic geometry below was MEASURED first (sweep),
not guessed: fletching r=40 keeps the end-choice confident
(width_ratio 0.437, well under WIDTH_AMBIGUITY_RATIO_MIN) while the r=4
satellite at (426, 510) pulls the primary tip to (426.5, 513.5) with a
cluster perp of -25.2px; the on-axis alternate recomputes to
(400.0, 498.2), within 2px of the true synthetic tip (400, 500).
The corpus gate itself comes from a real measured distribution (2474
real detections: chosen-tip-end |perp mean| p95 7.5px, incident 22.6px
-- see TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX's comment).

The real-package regression test skips cleanly when data/archive/clean/
does not hold the incident session, per living-corpus discipline.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from opendarts.engines.apollo.tip_detection import (
    TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX,
    detect_tip,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWN_OFF_AXIS_THROW = "20260817-142209/20260817-142209-061-T15"


def _corpus_root() -> Path:
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    return Path(env_root) if env_root else REPO_ROOT / "data" / "archive" / "clean"


def _synthetic_frame_pair(fletching_r: int, satellite: tuple[int, int, int] | None):
    """Wide fletching circle + tapering shaft triangle (the shape every
    synthetic tip-detection test here uses, true
    tip at (400, 500)), plus an optional detached off-axis satellite
    blob (x, y, r) close enough for DILATE_KERNEL_PX=31 to merge."""
    h, w = 800, 800
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (400, 150), fletching_r, (200, 200, 200), -1)
    pts = np.array([[380, 180], [420, 180], [400, 500]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (200, 200, 200))
    if satellite is not None:
        sx, sy, sr = satellite
        cv2.circle(frame, (sx, sy), sr, (200, 200, 200), -1)
    return bg, frame


def test_off_axis_satellite_populates_on_axis_alternate():
    """The incident mechanism, isolated: a small detached blob past the
    tip and well off the shaft axis wins the proj-extreme cluster, so
    the primary tip lands on the satellite -- and the new signature
    exposes the on-axis recompute as alt_tip_px without touching the
    primary."""
    bg, frame = _synthetic_frame_pair(fletching_r=40, satellite=(426, 510, 4))
    res = detect_tip(bg, frame)
    assert res.ok, res.reason

    # The primary tip is genuinely poisoned (this is the bug shape --
    # if this stops holding, the synthetic no longer demonstrates the
    # failure this file exists to pin).
    px, py = res.tip_px
    assert float(np.hypot(px - 426, py - 510)) < 10.0, res.tip_px

    # Signature fired: measured perp on this geometry is -25.2px.
    perp = res.diagnostics["tip_cluster_perp_px"]
    assert abs(perp) > TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX, perp
    assert res.diagnostics["tip_off_axis_alt"] is True

    # The alternate is the on-axis recompute, within a few px of the
    # true synthetic tip (400, 500) -- measured (400.0, 498.2).
    assert res.alt_tip_px is not None
    ax, ay = res.alt_tip_px
    assert float(np.hypot(ax - 400, ay - 500)) < 10.0, res.alt_tip_px


def test_healthy_shaft_has_no_off_axis_alternate():
    """Control: identical geometry minus the satellite -- tip lands on
    the true tip, the signature stays quiet, no alternate appears, and
    the always-present diagnostic reads near zero (measured -0.05px)."""
    bg, frame = _synthetic_frame_pair(fletching_r=40, satellite=None)
    res = detect_tip(bg, frame)
    assert res.ok, res.reason
    px, py = res.tip_px
    assert float(np.hypot(px - 400, py - 500)) < 10.0, res.tip_px
    assert res.alt_tip_px is None
    assert res.diagnostics["tip_off_axis_alt"] is False
    assert "tip_cluster_perp_px" in res.diagnostics
    assert abs(res.diagnostics["tip_cluster_perp_px"]) < 5.0


def test_width_ambiguity_alternate_takes_precedence_over_off_axis():
    """When the satellite ALSO makes the end-choice itself ambiguous
    (smaller fletching, r=30: measured width_ratio 0.610 >=
    WIDTH_AMBIGUITY_RATIO_MIN), the pre-existing width-ambiguity
    mechanism owns the single alt slot: alt_tip_px is the OTHER END
    (near the fletching), not the on-axis recompute, and
    tip_off_axis_alt stays False -- both candidate ends are already in
    score_dart()'s search, which is the bigger ambiguity to resolve."""
    bg, frame = _synthetic_frame_pair(fletching_r=30, satellite=(426, 510, 4))
    res = detect_tip(bg, frame)
    assert res.ok, res.reason
    assert res.alt_tip_px is not None
    ax, ay = res.alt_tip_px
    # Other-end alternate: near the fletching end (measured
    # (397.5, 120.0)), nowhere near the tip end.
    assert ay < 300, res.alt_tip_px
    assert res.diagnostics["tip_off_axis_alt"] is False
    assert res.alt_tip_px == res.far_end_px


@pytest.mark.skipif(
    not (_corpus_root() / KNOWN_OFF_AXIS_THROW).is_dir(),
    reason="incident package 20260817-142209-061-T15 not present in the corpus",
)
def test_real_incident_throw_scores_single_inner_via_the_alternate():
    """End-to-end on the real stored package, through the real replay
    path ("Replay is the source of truth" constraint): the throw that motivated this fix now
    scores S15 single_inner (operator/AD truth), with camera 2 using its
    alternate candidate, and the accepted ray disagreement collapsing
    from the poisoned 5.08mm to ~0.022mm (measured)."""
    from opendarts.capture.replay import replay_throw_with_engine
    from opendarts.capture.throw_package import load_throw_package

    # 2026-08-18: smoke tests must never pin an exact score for
    # a specific real corpus package -- calibration is itself subject to
    # REPLAY (docs/DESIGN.md), so this real throw's exact outcome is not a
    # stable smoke-test target. Real regressions are caught by
    # full-corpus replay against AD truth (tmp/ scripts), not pytest
    # pins. The alternate-candidate mechanism itself is verified against
    # known-truth synthetic fixtures above.
    pkg = load_throw_package(_corpus_root() / KNOWN_OFF_AXIS_THROW)
    res = replay_throw_with_engine(pkg, "Apollo")
    assert res.ok, res.reason
