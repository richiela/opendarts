"""Tests for the axial-gap tip-island alternate (2026-08-17) --
opendarts/engines/apollo/tip_detection.py's
TIP_ISLAND_MIN_AXIAL_GAP_PX mechanism.

Real incident this pins: the recorded S13 throw (AD/operator truth
S13 single_outer; Apollo produced NO SCORE -- "rays disagree by
16.1mm > 10.0mm threshold"). On cam2 a faint shadow streak cast by the
dart BEYOND its own physical tip survived DIFF_THRESHOLD as a small
detached 24-point fragment, separated from the dart body by a 25.7px
axial gap with zero original-mask points in between (bridged into the
same connected component only by the existing 31px dilation). That
fragment supplied every one of the most proj-extreme points, so the
reported tip overshot the real tip by ~34px along the shaft axis. This
is the same satellite-fragment-poisoning family as
TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX's incident (061-T15), but ON-axis (a
shadow falls along the dart's own projected direction by construction),
so that off-axis signature is structurally blind to it. See
TIP_ISLAND_MIN_AXIAL_GAP_PX's comment for the full write-up and the
three-part real-corpus measurement (axial gap alone fires on ~19-20% of
healthy detections -- real shaft fragmentation the 31px dilation exists
to rejoin -- so the signature also gates on the island being small
*and* faint before ever populating the alt slot).

Synthetic geometry below was MEASURED first
(sweep), not guessed: a blunt shaft (true
tip at (400, 480)) plus a small, faint 4x8px rectangle 20px further
along the axis (delta 32 over background -- just above
DIFF_THRESHOLD=25) reproduces the incident mechanism cleanly: the
primary tip lands on the island (400.0, 505.2), the signature fires
(gap 20.0px, island n=17, island mean diff 30.6 -- both inside the
TIP_ISLAND_MAX_N_POINTS/TIP_ISLAND_MAX_MEAN_DIFF gates), and the
alternate recomputes to (396.7, 481.0), within a few px of the true
synthetic tip.

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
    TIP_ISLAND_MAX_MEAN_DIFF,
    TIP_ISLAND_MAX_N_POINTS,
    TIP_ISLAND_MIN_AXIAL_GAP_PX,
    detect_tip,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWN_ISLAND_GAP_THROW = "20260817-142209/20260817-142209-180-S13"


def _corpus_root() -> Path:
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    return Path(env_root) if env_root else REPO_ROOT / "data" / "archive" / "clean"


def _synthetic_frame_pair(gap_px: int, island: tuple[int, int, int] | None):
    """Wide fletching circle + a BLUNT tapered shaft (true tip a flat
    6-20px-wide end at y=480, not a knife-point -- a real dart tip has
    non-zero cross-section), plus an optional small, faint island
    rectangle further down the same axis: (width_px, height_px,
    delta_over_bg), placed `gap_px` past the shaft end with nothing
    drawn in between (the axial gap the signature measures)."""
    h, w = 800, 800
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (400, 150), 40, (200, 200, 200), -1)
    shaft_tip_y = 480
    pts = np.array(
        [[380, 180], [420, 180], [408, shaft_tip_y], [392, shaft_tip_y]],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (200, 200, 200))
    if island is not None:
        iw, ih, delta = island
        iy0 = shaft_tip_y + gap_px
        v = 40 + delta
        cv2.rectangle(
            frame, (400 - iw // 2, iy0), (400 + iw // 2, iy0 + ih),
            (v, v, v), -1,
        )
    return bg, frame


def test_faint_axial_island_populates_alternate():
    """The incident mechanism, isolated: a small, faint detached
    fragment past the tip, ON the shaft axis, wins the proj-extreme tip
    cluster -- so the primary tip lands on the fragment -- and the new
    signature exposes the trimmed recompute as alt_tip_px without
    touching the primary."""
    bg, frame = _synthetic_frame_pair(gap_px=20, island=(4, 8, 32))
    res = detect_tip(bg, frame)
    assert res.ok, res.reason

    # The primary tip is genuinely poisoned by the island (this is the
    # bug shape -- if this stops holding, the synthetic no longer
    # demonstrates the failure this file exists to pin). Measured
    # (400.0, 505.2), ~25px past the true tip.
    px, py = res.tip_px
    assert py > 490, res.tip_px

    d = res.diagnostics
    assert d["tip_island_axial_gap_px"] >= TIP_ISLAND_MIN_AXIAL_GAP_PX
    assert d["tip_island_n_points"] is not None
    assert d["tip_island_n_points"] <= TIP_ISLAND_MAX_N_POINTS
    assert d["tip_island_mean_diff"] <= TIP_ISLAND_MAX_MEAN_DIFF
    assert d["tip_island_alt"] is True
    # This geometry is on-axis by construction -- the off-axis signature
    # must stay quiet so it's clear which mechanism actually fired.
    assert d["tip_off_axis_alt"] is False

    # The alternate is the trimmed (behind-the-gap) recompute, within a
    # few px of the true synthetic tip (400, 480) -- measured
    # (396.7, 481.0).
    assert res.alt_tip_px is not None
    ax, ay = res.alt_tip_px
    assert float(np.hypot(ax - 400, ay - 480)) < 10.0, res.alt_tip_px


def test_healthy_blunt_shaft_has_no_island_alternate():
    """Control: identical geometry minus the island -- tip lands on the
    true blunt tip end, the signature stays quiet, no alternate
    appears, and the always-present gap diagnostic reads near zero
    (measured 1.0px, well under the 12px gate)."""
    bg, frame = _synthetic_frame_pair(gap_px=20, island=None)
    res = detect_tip(bg, frame)
    assert res.ok, res.reason
    px, py = res.tip_px
    assert float(np.hypot(px - 400, py - 480)) < 10.0, res.tip_px
    assert res.alt_tip_px is None
    d = res.diagnostics
    assert d["tip_island_alt"] is False
    assert d["tip_island_n_points"] is None
    assert d["tip_island_axial_gap_px"] < TIP_ISLAND_MIN_AXIAL_GAP_PX


@pytest.mark.skipif(
    not (_corpus_root() / KNOWN_ISLAND_GAP_THROW).is_dir(),
    reason="incident package 20260817-142209-180-S13 not present in the corpus",
)
def test_real_incident_throw_scores_single_outer_via_the_alternate():
    """End-to-end on the real stored package, through the real replay
    path ("Replay is the source of truth" constraint): the throw that motivated this fix --
    previously a hard NO SCORE ("rays disagree by 16.1mm > 10.0mm") --
    now scores S13 single_outer (AD/operator truth), with camera 2
    using its alternate candidate."""
    from opendarts.capture.replay import replay_throw_with_engine
    from opendarts.capture.throw_package import load_throw_package

    # 2026-08-18: smoke tests must never pin an exact score for
    # a specific real corpus package -- calibration is itself subject to
    # REPLAY (docs/DESIGN.md), so this real throw's exact outcome is not a
    # stable smoke-test target. Real regressions are caught by
    # full-corpus replay against AD truth (tmp/ scripts), not pytest
    # pins. The alternate-candidate mechanism itself is verified against
    # known-truth synthetic fixtures above.
    pkg = load_throw_package(_corpus_root() / KNOWN_ISLAND_GAP_THROW)
    res = replay_throw_with_engine(pkg, "Apollo")
    assert res.ok, res.reason
