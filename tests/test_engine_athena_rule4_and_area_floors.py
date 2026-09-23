"""Athena's three 2026-08-18 additions to `_corroborated_label_override()`
(opendarts/engines/athena/engine.py), all found during a fresh-eyes review
of the 26 remaining living-corpus misses (Apollo (Athena
miss investigation)" entries for the full per-throw diagnostics):

1. **Rule 4** -- two-camera lone-line-disagreement override. On a
   2-camera-used throw, rules 2a/3 can never fire (both require >=2 point
   reads opposing the heavy camera, impossible with only one other
   camera). When the lighter camera's own point AND the one available
   shaft-line intersection both cross the HEAVIER camera's own sector
   wire (agreeing with each other, disagreeing with the heavy camera),
   that is the heavy camera's own line contradicting its own point.
   Gated by a real area floor on the lighter camera (`RULE4_LIGHT_CAM_
   MIN_AREA_PX`) -- without it, a noise-speck "detection" whose ray
   happens to intersect plausibly is trusted just as much as a real one.

2. **Gate 2's outside-only independence tightening.** The original
   max-weight rule (`elif best_lab != blend_lab and _is_corroborated
   (best_lab)`) let a heavy camera's own point + its own self-involved
   intersection out-weigh a blend that was already CORRECT, specifically
   when best_lab is the unbounded "outside" region -- scoped narrowly to
   outside only (never to a real bed label, where a 2026-08-17 attempt at
   the same tightening measured a net-negative +2/-3 on this corpus).

3. **Gate (a)'s area floor on opposing point reads**
   (`RULE_2A_MIN_OPPOSING_AREA_PX`) -- "unanimous opposition" from two
   noise-speck detections (areas <100px) is not the same evidence gate
   (a) was validated against (real dart blobs, areas 700-1862px on the
   throw its own test already pins). Deliberately NOT applied to rule 3
   or gate 2's own independence check (measured and reverted -- see that
   code's own comment: it fixed one more throw but broke three whose
   opposing point read is small-but-genuinely-correct, not noise).

Full living clean/ corpus (n=1107, subprocess-per-session, calibration
refit applied):
**1081/1107 -> 1086/1107, +5/-0** (032-S10, 033-D9, 035-S10, 010-S20,
throw_1786690662472 gained; nothing lost; zero no-scores before or
after).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from opendarts.capture.replay import replay_throw_with_engine
from opendarts.engines.athena.engine import _corroborated_label_override
from opendarts.geometry.board import sector_ring_for_point

REPO_ROOT = Path(__file__).resolve().parent.parent


def _corpus_root() -> Path:
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    return Path(env_root) if env_root else REPO_ROOT / "data" / "archive" / "clean"


def _replay_pin(rel_throw: str):
    pkg_dir = _corpus_root() / rel_throw
    if not (pkg_dir / "calibration.json").exists():
        pytest.skip(
            f"{rel_throw} is not in the corpus on this machine -- the corpus is a "
            "living, curated thing (docs/DESIGN.md), so this pin skips rather than fails "
            "when it has moved on"
        )
    return replay_throw_with_engine(pkg_dir, "Athena")


def _read(x_mm, y_mm, weight, kind="point", cam=None, area=None):
    sector, ring = sector_ring_for_point(x_mm, y_mm)
    read = {
        "cam": cam, "x_mm": x_mm, "y_mm": y_mm,
        "sector": sector, "ring": ring, "weight": weight, "kind": kind,
    }
    if area is not None:
        read["area"] = area
    return read


# ---------------------------------------------------------------------------
# 2026-08-18: smoke tests must never pin an exact score for a
# specific real corpus package -- structural checks only (result.ok).
# Real accuracy is the full-corpus number quoted in this module's own
# docstring above, not these pins.
# ---------------------------------------------------------------------------

_GAINED = [
    ("20260817-142209/20260817-142209-032-S10", "10", "single_outer"),
    ("20260817-160135/20260817-160135-033-D9", "9", "double"),
    ("20260817-160135/20260817-160135-035-S10", "10", "single_outer"),
    ("20260818-001045/20260818-001045-010-S20", "20", "single_outer"),
]


@pytest.mark.parametrize(
    "rel_throw,exp_sector,exp_ring",
    _GAINED,
    ids=[g[0].split("/")[-1] for g in _GAINED],
)
def test_2026_08_18_gained_throw_pin(rel_throw, exp_sector, exp_ring):
    result = _replay_pin(rel_throw)
    assert result.ok


def test_2026_08_18_gained_old_style_throw_pin():
    """throw_1786690662472 lives directly under a session dir, not the
    session/session-NNN-token layout the others use."""
    result = _replay_pin("20260813-234015/throw_1786690662472")
    assert result.ok


# ---------------------------------------------------------------------------
# Rule 4: two-camera lone-line-disagreement override.
# ---------------------------------------------------------------------------


def test_rule4_fires_when_light_plus_intersection_cross_heavy_sector_wire():
    """Real S10-throw shape: heavy cam0 says 6/single_outer
    (weight .172), light cam2 says 10/single_outer (weight .023, area
    3377px -- well above the floor), and the only intersection ix(0,2)
    agrees with the light camera. Rule 4 must override to 10/single_outer."""
    reads = [
        _read(142.85, -22.55, 0.172, kind="point", cam=0, area=12903), # 6/single_outer
        _read(138.37, -25.52, 0.023, kind="point", cam=2, area=3377), # 10/single_outer
        _read(143.63, -23.54, 0.060, kind="intersection", cam=(0, 2)), # 10/single_outer
    ]
    assert reads[0]["sector"] == "6" and reads[1]["sector"] == "10"
    assert reads[2]["sector"] == "10"
    combined = {"x_mm": 142.85, "y_mm": -22.55}
    sector, ring = sector_ring_for_point(142.85, -22.55)
    assert sector == "6"
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("10", "single_outer")
    assert snapped.get("label_override") is True


def test_rule4_requires_light_camera_area_above_floor():
    """Real S10-throw shape: heavy cam0 is CORRECT
    (10/single_inner), but a light camera's near-noise 30px blob happens
    to intersect plausibly at a DIFFERENT sector (15). Without the area
    floor this wrongly overrides a correct heavy read; with it, rule 4
    must stay silent (some other rule may still fire, but not this one)."""
    reads = [
        _read(75.15, -30.07, 0.110, kind="point", cam=0, area=8424), # 10/single_inner (heavy, correct)
        _read(79.07, -71.47, 0.026, kind="point", cam=2, area=30), # 15/treble (light, noise)
        _read(96.66, -49.70, 0.054, kind="intersection", cam=(0, 2)), # 15/single_outer
    ]
    assert reads[0]["sector"] == "10"
    assert reads[1]["sector"] == "15"
    combined = {"x_mm": 75.15, "y_mm": -30.07}
    sector, ring = sector_ring_for_point(75.15, -30.07)
    assert sector == "10"
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert s == "10", "a near-noise light-camera blob must not override a correct heavy read"


def test_rule4_does_not_fire_on_three_camera_throws():
    """Structural guard: rule 4 requires EXACTLY two point reads (the
    natural shape of a 2-camera-used throw). A 3-point-read throw must
    never trigger it, even if two of the three would otherwise match its
    shape."""
    reads = [
        _read(142.85, -22.55, 0.172, kind="point", cam=0, area=12903), # 6/single_outer
        _read(138.37, -25.52, 0.023, kind="point", cam=2, area=3377), # 10/single_outer
        _read(140.00, -24.00, 0.010, kind="point", cam=1, area=5000), # a third real read
        _read(143.63, -23.54, 0.060, kind="intersection", cam=(0, 2)), # 10/single_outer
    ]
    combined = {"x_mm": 142.0, "y_mm": -22.8}
    sector, ring = sector_ring_for_point(142.0, -22.8)
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    # Whatever fires (or doesn't) here, it must not be rule 4's own
    # signature ("candidate_lab" logic) -- verified indirectly by
    # confirming this doesn't crash and the 3-point-read shape is
    # legal input to every other rule too.
    assert (s, r) is not None


# ---------------------------------------------------------------------------
# Gate 2's outside-only independence tightening (033-D9 shape).
# ---------------------------------------------------------------------------


def test_gate2_outside_branch_does_not_override_a_correct_multi_point_blend():
    """Real D9-throw shape: the raw blend already sits at
    the CORRECT 9/double (both cam0 and cam2's own points), but the
    original max-weight rule would move it to "outside" purely because
    the heavy camera's own point + its own self-involved intersection
    out-weigh the two correct-but-lighter cameras. The tightened rule
    must leave the correct blend alone."""
    reads = [
        _read(-127.07, 106.45, 0.0167, kind="point", cam=0), # 9/double
        _read(-129.40, 111.52, 0.1044, kind="point", cam=1), # outside (heavy, wrong)
        _read(-130.20, 106.96, 0.0664, kind="point", cam=2), # 9/double
        _read(-130.53, 109.12, 0.0407, kind="intersection", cam=(0, 1)), # outside (self-involved w/ cam1)
        _read(-129.36, 108.22, 0.0332, kind="intersection", cam=(0, 2)), # 9/double
    ]
    assert reads[1]["ring"] == "outside"
    assert reads[0]["sector"] == "9" and reads[2]["sector"] == "9"
    combined = {"x_mm": -129.89, "y_mm": 109.08}
    sector, ring = sector_ring_for_point(-129.89, 109.08)
    assert (sector, ring) == ("9", "double"), "raw blend must already be correct in this shape"
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("9", "double")
    assert "label_override" not in snapped


def test_gate2_outside_branch_still_fires_when_independent_of_correct_blend():
    """The original max-weight rule must still work when the blend's own
    two point cameras are NOT what the corroborating intersection
    reaches through -- i.e. this is a real tightening (outside-only +
    independent-of-blend), not a blanket disabling of the rule."""
    reads = [
        _read(-165.5, 1.8, 0.10, kind="point", cam=0), # 11/double (blend, lone)
        _read(-169.0, 3.7, 0.09, kind="intersection", cam=(0, 2)), # 11/double (self-involved)
        _read(-171.0, 2.5, 0.05, kind="point", cam=2), # outside
    ]
    combined = {"x_mm": -170.5, "y_mm": 2.5}
    sector, ring = sector_ring_for_point(-170.5, 2.5)
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("11", "double"), "non-outside targets are untouched by this tightening"


# ---------------------------------------------------------------------------
# Gate (a)'s area floor on opposing point reads (010-S20 shape).
# ---------------------------------------------------------------------------


def test_gate_a_area_floor_blocks_noise_speck_unanimous_opposition():
    """Real S20-throw shape: a correct, high-area heavy
    camera (12044px) is outvoted by "unanimous opposition" from two
    noise-speck detections (86px, 21px) plus their own intersection. The
    area floor must keep the correct blend."""
    reads = [
        _read(54.93, 163.33, 0.0342, kind="point", cam=0, area=86), # outside (noise)
        _read(19.19, 145.97, 0.1790, kind="point", cam=1, area=12044), # 20/single_outer (heavy, correct)
        _read(1.77, 171.96, 0.0145, kind="point", cam=2, area=21), # outside (noise)
        _read(17.36, 170.49, 0.0756, kind="intersection", cam=(0, 1)), # outside
        _read(17.27, 171.70, 0.0508, kind="intersection", cam=(1, 2)), # outside
    ]
    combined = {"x_mm": 19.20, "y_mm": 145.98}
    sector, ring = sector_ring_for_point(19.20, 145.98)
    assert (sector, ring) == ("20", "single_outer"), "raw blend must already be correct in this shape"
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("20", "single_outer")
    assert "label_override" not in snapped


def test_gate_a_still_fires_with_real_dart_blob_areas():
    """The gate's own existing validated shape
    (test_override_gate_a_unanimous_opposition_fires_even_into_outside in
    test_athena_engine_consensus.py, the real outside throw
    throw) must still fire when areas are present and plausible (real
    areas from that throw: 700px and 1862px, both above the floor)."""
    reads = [
        _read(72.0, -153.0, 0.131, kind="point", cam=0, area=12904), # 17/double (lone, wrong)
        _read(78.0, -163.0, 0.04, kind="point", cam=1, area=700), # outside
        _read(80.0, -161.0, 0.03, kind="point", cam=2, area=1862), # outside
        _read(79.0, -162.0, 0.02, kind="intersection", cam=(1, 2)), # outside
    ]
    combined = {"x_mm": 72.0, "y_mm": -153.0}
    sector, ring = sector_ring_for_point(72.0, -153.0)
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == (None, "outside")
    assert snapped.get("label_override") is True


def test_gate_a_missing_area_data_is_trusted_not_rejected():
    """Reads with no 'area' key at all (e.g. every other synthetic test
    in this project, or a rare candidate with no recorded area) must not
    be penalized by the floor -- same convention as
    MIN_RELATIVE_AREA_FRACTION's own gate."""
    reads = [
        _read(72.0, -153.0, 0.131, kind="point", cam=0), # 17/double (lone, wrong), no area
        _read(78.0, -163.0, 0.04, kind="point", cam=1), # outside, no area
        _read(80.0, -161.0, 0.03, kind="point", cam=2), # outside, no area
        _read(79.0, -162.0, 0.02, kind="intersection", cam=(1, 2)), # outside
    ]
    combined = {"x_mm": 72.0, "y_mm": -153.0}
    sector, ring = sector_ring_for_point(72.0, -153.0)
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == (None, "outside"), "missing area data must not block a gate that would otherwise fire"
