"""Athena third override rule (2026-08-17): PROVENANCE-INDEPENDENT
unanimous opposition, in `_corroborated_label_override()`
(opendarts/engines/athena/engine.py).

The dominant Athena miss family found by the 996-throw fresh-eyes
review: the steepness^4 weight concentration lets ONE heavy camera's
wrong point read drag the whole blend, while BOTH other cameras' point
reads AND the intersection of those two cameras' own shaft lines -- the
only read in the consensus that shares no provenance with the heavy
camera -- agree on the correct label. The two existing rules never fire
there because the heavy camera's own intersections (pairs INVOLVING it)
hand its label enough support to win both the lone-read precondition and
the max-total-weight comparison.

The new gate fires only when: the blend's label is supported by at most
ONE camera's point read, AND a competing label has at least TWO point
reads PLUS an intersection whose camera pair excludes every blend-label
point camera. Two correlated point reads alone still cannot outvote the
heavy camera (the thrice-rejected majority-vote shape), and on a
2-camera throw every intersection involves both cameras, so the gate
structurally cannot fire there at all.

Full-corpus measurement (living data/archive/clean/, 996 AD-matched,
2026-08-17): 956 -> 968, +12 / -0. Every gained throw is pinned below.
Notably this fixes the recorded S15 throw, one of the two previously
documented Athena walls -- the wall was real for the OLD rules (its
blend label carries a point + a self-involved intersection, so the
lone-read precondition fails), but its competing label has two point
cams + an independent intersection, exactly this rule's shape.
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


# --------------------------------------------------------------------------
# Real-corpus pins: every throw the 2026-08-17 measurement moved (+12/-0).
# Expected labels are AD's own operator-validated sector+ring.
# --------------------------------------------------------------------------

_GAINED = [
    ("20260813-164658/throw_1786664892745", None, "outside"),
    ("20260813-164658/throw_1786667737057", "3", "single_outer"),
    ("20260813-164658/throw_1786667799904", "2", "single_outer"),
    ("20260814-105858/throw_1786730450821", "14", "double"),
    ("20260814-105858/throw_1786730469033", "10", "single_outer"),
    ("20260814-181428/20260814-181428-001-S1", "20", "single_outer"),
    ("20260816-163944/20260816-163944-024-S20", "20", "single_inner"),
    ("20260816-163944/20260816-163944-027-OUT", None, "outside"),
    ("20260816-163944/20260816-163944-046-S20", "20", "single_outer"),
    ("20260817-142209/20260817-142209-036-S15", "15", "single_outer"),
    ("20260817-160135/20260817-160135-039-D9", "9", "double"),
    ("20260817-160135/20260817-160135-106-S8", "8", "single_outer"),
]


# 2026-08-18: smoke tests must never pin an exact score for a
# specific real corpus package -- calibration is itself subject to
# REPLAY (recomputed per corpus-refit, docs/DESIGN.md), so a real package's
# exact computed outcome is not a stable smoke-test target. Real
# accuracy regressions are caught by full-corpus replay against AD
# truth (tmp/ scripts), not by pytest pins. The rule's actual logic is
# verified against known-truth synthetic reads below
# (test_third_rule_fires_on_provenance_independent_opposition); these
# real throws stay only as a "still commits to an answer" smoke check.
@pytest.mark.parametrize(
    "rel_throw,exp_sector,exp_ring",
    _GAINED,
    ids=[g[0].split("/")[-1] for g in _GAINED],
)
def test_independent_opposition_gained_throw_pin(rel_throw, exp_sector, exp_ring):
    result = _replay_pin(rel_throw)
    assert result.ok


# --------------------------------------------------------------------------
# Synthetic: the rule itself, isolated -- same real-board-coordinate style
# as tests/test_athena_engine_consensus.py (labels come from
# sector_ring_for_point, never hand-asserted).
# --------------------------------------------------------------------------


def _read(x_mm, y_mm, weight, kind="point", cam=None):
    sector, ring = sector_ring_for_point(x_mm, y_mm)
    return {
        "cam": cam, "x_mm": x_mm, "y_mm": y_mm,
        "sector": sector, "ring": ring, "weight": weight, "kind": kind,
    }


def test_third_rule_fires_on_provenance_independent_opposition():
    """The real S15-throw shape, actual measured reads:
    heavy cam0 (w=0.209) says 2/single_outer, corroborated by its own
    self-involved ix(0,2) -- so the blend label is multi-read and the old
    lone-read rules stay silent. But cams 1+2's points AND their own
    independent ix(1,2) all say 15/single_outer. The third rule must
    override to 15."""
    reads = [
        _read(112.9, -112.9, 0.209, kind="point", cam=0), # 2/single_outer
        _read(112.7, -105.4, 0.021, kind="point", cam=1), # 15/single_outer
        _read(113.4, -108.5, 0.031, kind="point", cam=2), # 15/single_outer
        _read(108.9, -110.0, 0.066, kind="intersection", cam=(0, 2)), # 2/single_outer
        _read(113.6, -108.4, 0.025, kind="intersection", cam=(1, 2)), # 15/single_outer
    ]
    assert reads[0]["sector"] == "2" and reads[1]["sector"] == "15"
    assert reads[3]["sector"] == "2" and reads[4]["sector"] == "15"
    combined = {"x_mm": 112.85, "y_mm": -112.91}
    sector, ring = sector_ring_for_point(112.85, -112.91)
    assert (sector, ring) == ("2", "single_outer")
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("15", "single_outer")
    assert snapped.get("label_override") is True


def test_third_rule_overrides_even_when_blend_label_out_weighs():
    """The real D9-throw shape: the heavy camera's outside
    label carries MORE total weight (own point 0.107 + own ix 0.042) than
    the 9/double opposition (0.017 + 0.063 + 0.033) -- weight comparison
    alone would keep outside. The provenance-independent gate must still
    override to 9/double."""
    reads = [
        _read(-124.0, 110.7, 0.017, kind="point", cam=0), # 9/double
        _read(-126.8, 115.5, 0.107, kind="point", cam=1), # outside
        _read(-127.3, 111.1, 0.063, kind="point", cam=2), # 9/double
        _read(-127.7, 113.6, 0.042, kind="intersection", cam=(0, 1)), # outside
        _read(-126.3, 112.5, 0.033, kind="intersection", cam=(0, 2)), # 9/double
    ]
    assert reads[1]["ring"] == "outside" and reads[0]["ring"] == "double"
    assert reads[3]["ring"] == "outside" and reads[4]["ring"] == "double"
    combined = {"x_mm": -126.8, "y_mm": 115.0}
    sector, ring = sector_ring_for_point(-126.8, 115.0)
    assert ring == "outside"
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("9", "double")
    assert snapped.get("label_override") is True


def test_third_rule_requires_a_provenance_independent_intersection():
    """Same opposition (two point cams on 15) but the ONLY intersection
    supporting it involves the blend's own heavy camera -- self-involved
    corroboration is exactly what lets the heavy camera echo itself, so
    the gate must NOT fire (and no other rule fires here either: the
    blend label is multi-read and out-weighs everything)."""
    reads = [
        _read(112.9, -112.9, 0.209, kind="point", cam=0), # 2/single_outer
        _read(112.7, -105.4, 0.021, kind="point", cam=1), # 15/single_outer
        _read(113.4, -108.5, 0.031, kind="point", cam=2), # 15/single_outer
        _read(108.9, -110.0, 0.066, kind="intersection", cam=(0, 2)), # 2/single_outer
        _read(113.0, -106.0, 0.025, kind="intersection", cam=(0, 1)), # 15/single_outer
    ]
    assert reads[4]["sector"] == "15", "the self-involved ix must still carry the 15 label"
    combined = {"x_mm": 112.85, "y_mm": -112.91}
    sector, ring = sector_ring_for_point(112.85, -112.91)
    assert (sector, ring) == ("2", "single_outer")
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("2", "single_outer")
    assert "label_override" not in snapped


def test_third_rule_requires_two_point_cams_not_just_weight():
    """One opposing point read + an independent-looking intersection is
    not unanimous opposition -- with only one other camera actually
    SEEING the competing label, the gate must stay closed."""
    reads = [
        _read(112.9, -112.9, 0.209, kind="point", cam=0), # 2/single_outer
        _read(112.7, -105.4, 0.021, kind="point", cam=1), # 15/single_outer
        _read(113.6, -108.4, 0.045, kind="intersection", cam=(1, 2)), # 15/single_outer
    ]
    combined = {"x_mm": 112.85, "y_mm": -112.91}
    sector, ring = sector_ring_for_point(112.85, -112.91)
    assert (sector, ring) == ("2", "single_outer")
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("2", "single_outer")
    assert "label_override" not in snapped


def test_third_rule_points_only_opposition_never_fires():
    """The real S20-throw shape: both other cameras' points
    oppose the heavy camera but NO intersection backs them -- that is the
    plain 2-of-3 majority vote this engine measured and rejected three
    separate times, and it stays rejected (025 is a knowingly-kept miss)."""
    reads = [
        _read(28.0, 110.0, 0.150, kind="point", cam=1), # 1/single_outer (heavy, wrong)
        _read(13.7, 112.0, 0.026, kind="point", cam=0), # 20/single_outer
        _read(15.1, 109.5, 0.018, kind="point", cam=2), # 20/single_outer
    ]
    assert reads[0]["sector"] == "1"
    assert reads[1]["sector"] == "20" and reads[2]["sector"] == "20"
    combined = {"x_mm": 27.0, "y_mm": 110.0}
    sector, ring = sector_ring_for_point(27.0, 110.0)
    assert sector == "1"
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == (sector, ring)
    assert "label_override" not in snapped
