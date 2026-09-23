"""Unit tests for the pure, deterministic pieces of
opendarts.engines.athena.engine's consensus logic (_camera_weight,
_combine_reads) -- isolated from image/camera pipeline entirely, so
these pin the actual MATH independent of any real or synthetic image
data (complementary to dev/tests/test_engine_athena_accuracy.py's
real-corpus, end-to-end coverage).
"""
from __future__ import annotations

import pytest

from opendarts.engines.athena.engine import _camera_weight, _combine_reads


def test_camera_weight_increases_monotonically_with_steepness():
    """A steeper (more perpendicular-to-board) ray must never be trusted
    LESS than a shallower one -- the entire point of this signal (see
    _camera_weight's own docstring)."""
    w_shallow = _camera_weight(0.1)
    w_mid = _camera_weight(0.5)
    w_steep = _camera_weight(0.9)
    assert w_shallow < w_mid < w_steep


def test_camera_weight_is_positive_even_at_zero_steepness():
    """The floor keeps a fully-grazing ray from getting literally zero
    weight (it may still be the ONLY camera available, in which case
    _combine_reads must still be able to use it)."""
    assert _camera_weight(0.0) > 0.0


def test_combine_reads_single_read_returns_it_unchanged():
    reads = [{"x_mm": 12.5, "y_mm": -7.0, "weight": 1.0}]
    combined = _combine_reads(reads)
    assert combined == {"x_mm": 12.5, "y_mm": -7.0}


def test_combine_reads_equal_weights_symmetric_points_average_to_center():
    """Three equally-weighted reads forming a symmetric triangle around
    the origin -- the weighted geometric median of a symmetric
    configuration is the centroid, a simple, exactly-checkable case."""
    reads = [
        {"x_mm": 10.0, "y_mm": 0.0, "weight": 1.0},
        {"x_mm": -5.0, "y_mm": 8.660254, "weight": 1.0},
        {"x_mm": -5.0, "y_mm": -8.660254, "weight": 1.0},
    ]
    combined = _combine_reads(reads)
    assert combined["x_mm"] == pytest.approx(0.0, abs=1e-3)
    assert combined["y_mm"] == pytest.approx(0.0, abs=1e-3)


def test_combine_reads_two_reads_actually_blends_rather_than_picking_the_heavier():
    """**The n=2 special case, and the exact regression it guards against**
    (2026-08-14, see _combine_reads()'s own docstring): the weighted
    GEOMETRIC MEDIAN is degenerate at exactly two points -- it minimises
    `w1*|p-p1| + w2*|p-p2|`, whose minimum over the segment is the heavier
    endpoint -- so running Weiszfeld here silently discarded the lighter
    camera's read entirely. Measured on the real corpus at the time: 107
    of 111 two-camera throws came out within 0.01mm of the heavier
    camera's own point, and the discarded camera was the MORE accurate one
    34.2% of the time.

    This pins the fix at the mechanism level rather than via a corpus
    number: with two reads the result must be a genuine weighted mean,
    strictly BETWEEN the two points and pulled proportionally toward the
    heavier one -- never equal to either endpoint.
    """
    reads = [
        {"x_mm": 0.0, "y_mm": 0.0, "weight": 3.0},
        {"x_mm": 100.0, "y_mm": 0.0, "weight": 1.0},
    ]
    combined = _combine_reads(reads)
    # Exact weighted mean: (3*0 + 1*100) / 4 = 25.0 -- computed, not a
    # round number picked to be comfortably true.
    assert combined["x_mm"] == pytest.approx(25.0, abs=1e-9)
    assert combined["y_mm"] == pytest.approx(0.0, abs=1e-9)
    # The property that actually matters: strictly interior, so neither
    # camera is thrown away.
    assert 0.0 < combined["x_mm"] < 100.0


def test_combine_reads_two_equal_weight_reads_land_exactly_midway():
    """Degenerate-tie version of the case above: two equally-trusted
    cameras must average, not resolve arbitrarily to one of them (which is
    what the geometric median's tie behaviour did)."""
    reads = [
        {"x_mm": -40.0, "y_mm": 10.0, "weight": 2.5},
        {"x_mm": 20.0, "y_mm": -30.0, "weight": 2.5},
    ]
    combined = _combine_reads(reads)
    assert combined["x_mm"] == pytest.approx(-10.0, abs=1e-9)
    assert combined["y_mm"] == pytest.approx(-10.0, abs=1e-9)


def test_combine_reads_two_reads_with_zero_weights_still_returns_their_midpoint():
    """Guard the divide-by-zero path: if both cameras somehow carry zero
    weight, fall back to the plain midpoint rather than raising."""
    reads = [
        {"x_mm": 4.0, "y_mm": 6.0, "weight": 0.0},
        {"x_mm": 8.0, "y_mm": -2.0, "weight": 0.0},
    ]
    combined = _combine_reads(reads)
    assert combined["x_mm"] == pytest.approx(6.0, abs=1e-9)
    assert combined["y_mm"] == pytest.approx(2.0, abs=1e-9)


def test_combine_reads_three_reads_still_use_the_robust_geometric_median():
    """The n=2 fix must NOT leak into n>=3, where the geometric median is
    a genuine robust blend and is what makes one gross read survivable --
    a plain weighted mean there measured materially worse (140/154 vs
    149/154 on the real corpus). Two tightly-agreeing reads plus one gross
    outlier of comparable weight: the geometric median stays near the
    agreeing pair, whereas a weighted mean would be dragged a long way
    toward the outlier."""
    reads = [
        {"x_mm": 0.0, "y_mm": 0.0, "weight": 1.0},
        {"x_mm": 2.0, "y_mm": 0.0, "weight": 1.0},
        {"x_mm": 300.0, "y_mm": 0.0, "weight": 1.0},
    ]
    combined = _combine_reads(reads)
    weighted_mean_x = (0.0 + 2.0 + 300.0) / 3.0
    assert combined["x_mm"] < 10.0, "geometric median should resist the outlier"
    assert combined["x_mm"] < weighted_mean_x / 5.0


def test_combine_reads_high_weight_camera_dominates_over_two_low_weight_agreeing_cameras():
    """The exact scenario this whole design exists to handle better than
    naive majority vote: two LOW-weight cameras agree with each other at
    one point, one HIGH-weight camera reports a different point --
    the combined result must land closer to the high-weight camera's
    answer than to the agreeing pair's, despite being outnumbered 2-to-1
    (the correlated-bias framing -- see _combine_reads()'s own
    docstring)."""
    reads = [
        {"x_mm": 0.0, "y_mm": 0.0, "weight": 50.0},  # the trustworthy outlier
        {"x_mm": 100.0, "y_mm": 0.0, "weight": 0.5},  # agreeing pair, low trust
        {"x_mm": 100.0, "y_mm": 5.0, "weight": 0.5},
    ]
    combined = _combine_reads(reads)
    dist_to_high_weight = ((combined["x_mm"] - 0.0) ** 2 + (combined["y_mm"] - 0.0) ** 2) ** 0.5
    dist_to_pair = ((combined["x_mm"] - 100.0) ** 2 + (combined["y_mm"] - 2.5) ** 2) ** 0.5
    assert dist_to_high_weight < dist_to_pair


# ---------------------------------------------------------------------------
# _corroborated_label_override -- the mixed-kind label override and its
# 2026-08-17 second firing condition (gates (a)/(b), see the function's
# own docstring for the full 13-case corpus probe these rules were
# separated from). All coordinates below are REAL board-plane positions
# whose sector/ring labels come from sector_ring_for_point itself, so
# these tests pin behavior against the real board geometry, not
# hand-asserted labels.
# ---------------------------------------------------------------------------

from opendarts.engines.athena.engine import _corroborated_label_override  # noqa: E402
from opendarts.geometry.board import sector_ring_for_point  # noqa: E402


def _read(x_mm, y_mm, weight, kind="point", cam=None):
    sector, ring = sector_ring_for_point(x_mm, y_mm)
    return {
        "cam": cam, "x_mm": x_mm, "y_mm": y_mm,
        "sector": sector, "ring": ring, "weight": weight, "kind": kind,
    }


def test_override_rule1_max_weight_corroborated_label_still_fires():
    """The original (pre-2026-08-17) rule is untouched: when the
    max-total-weight label differs from the blend's and carries mixed
    point+intersection support, it wins."""
    reads = [
        _read(-165.5, 1.8, 0.10, kind="point", cam=0),          # 11/double
        _read(-169.0, 3.7, 0.09, kind="intersection", cam=(0, 2)),  # 11/double
        _read(-171.0, 2.5, 0.05, kind="point", cam=2),          # outside
    ]
    assert reads[0]["ring"] == "double" and reads[2]["ring"] == "outside"
    combined = {"x_mm": -170.5, "y_mm": 2.5}
    sector, ring = sector_ring_for_point(-170.5, 2.5)
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("11", "double")
    assert snapped.get("label_override") is True


def test_override_gate_b_lone_outside_point_loses_to_corroborated_bed():
    """Gate (b), a real recorded throw's shape: a lone point read
    lands 0.15mm past the double wire (ring='outside') and out-weighs a
    point+intersection pair independently agreeing on 11/double. The
    corroborated bed must win despite losing the weight comparison."""
    reads = [
        _read(-170.1, 2.6, 0.164, kind="point", cam=2),           # outside (lone, heavy)
        _read(-165.5, 1.8, 0.018, kind="point", cam=0),           # 11/double
        _read(-169.0, 3.7, 0.053, kind="intersection", cam=(0, 2)),  # 11/double
    ]
    assert reads[0]["ring"] == "outside"
    assert reads[1]["sector"] == "11" and reads[2]["sector"] == "11"
    combined = {"x_mm": -170.1, "y_mm": 2.6}
    sector, ring = sector_ring_for_point(-170.1, 2.6)
    assert ring == "outside"
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("11", "double")
    assert snapped.get("label_override") is True
    snap_lab = sector_ring_for_point(snapped["x_mm"], snapped["y_mm"])
    assert snap_lab == ("11", "double"), "snapped point must carry the label it claims"


def test_override_gate_b_mirror_never_fires_into_outside():
    """The measured mirror of gate (b) -- a lone ON-BOARD point read vs a
    single-point-corroborated OUTSIDE label -- lost every observed case
    in the 885-throw corpus probe (4 real doubles read slightly long by
    one camera). It must NOT override."""
    reads = [
        _read(-163.0, -42.0, 0.20, kind="point", cam=2),            # 8/double (lone, correct)
        _read(-171.5, -40.0, 0.03, kind="point", cam=1),            # outside
        _read(-172.0, -41.0, 0.04, kind="intersection", cam=(1, 2)),  # outside
    ]
    assert reads[0]["sector"] == "8" and reads[0]["ring"] == "double"
    assert reads[1]["ring"] == "outside" and reads[2]["ring"] == "outside"
    combined = {"x_mm": -163.0, "y_mm": -42.0}
    sector, ring = sector_ring_for_point(-163.0, -42.0)
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("8", "double"), "single-point corroboration must not drag a bed to outside"
    assert "label_override" not in snapped


def test_override_gate_a_unanimous_opposition_fires_even_into_outside():
    """Gate (a): when BOTH other cameras' own point reads AND
    intersection evidence agree on the competing label, the lone blend
    read loses -- including in the into-outside direction gate (b) alone
    forbids (the real outside-throw shape)."""
    reads = [
        _read(72.0, -153.0, 0.131, kind="point", cam=0),             # 17/double (lone, wrong)
        _read(78.0, -163.0, 0.04, kind="point", cam=1),              # outside
        _read(80.0, -161.0, 0.03, kind="point", cam=2),              # outside
        _read(79.0, -162.0, 0.02, kind="intersection", cam=(1, 2)),  # outside
    ]
    assert reads[0]["ring"] == "double"
    assert all(r["ring"] == "outside" for r in reads[1:])
    combined = {"x_mm": 72.0, "y_mm": -153.0}
    sector, ring = sector_ring_for_point(72.0, -153.0)
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == (None, "outside")
    assert snapped.get("label_override") is True


def test_override_two_points_without_intersection_do_not_outvote():
    """Two point reads agreeing WITHOUT any intersection corroboration is
    exactly the plain majority label vote this engine measured and
    rejected three separate times -- the mixed-kind requirement must keep
    it out even under the second firing condition."""
    reads = [
        _read(-170.1, 2.6, 0.164, kind="point", cam=2),   # outside (lone)
        _read(-165.5, 1.8, 0.05, kind="point", cam=0),    # 11/double
        _read(-166.0, 2.0, 0.04, kind="point", cam=1),    # 11/double
    ]
    combined = {"x_mm": -170.1, "y_mm": 2.6}
    sector, ring = sector_ring_for_point(-170.1, 2.6)
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == (sector, ring), "no intersection evidence -> no override, ever"
    assert "label_override" not in snapped


def test_override_leaves_multi_read_blend_labels_alone():
    """The second firing condition requires the blend's own label to be a
    LONE single read -- a blend label with its own multi-read support
    (a simplified shape from a recorded throw, where the blend label was
    itself point+intersection corroborated) must never be second-guessed
    by a lighter corroborated competitor WITH ONLY ONE POINT CAM. (The
    REAL 036 throw is since fixed by the 2026-08-17 third rule -- its
    actual competing label has TWO point cams plus a provenance-
    independent intersection, which this synthetic shape deliberately
    does not: see tests/test_engine_athena_independent_opposition.py.)"""
    reads = [
        _read(112.9, -112.9, 0.209, kind="point", cam=0),             # 2/single_outer
        _read(108.9, -110.0, 0.066, kind="intersection", cam=(0, 2)),  # 2/single_outer
        _read(112.7, -105.4, 0.021, kind="point", cam=1),             # 15/single_outer
        _read(113.6, -108.4, 0.025, kind="intersection", cam=(1, 2)),  # 15/single_outer
    ]
    assert reads[0]["sector"] == "2" and reads[2]["sector"] == "15"
    combined = {"x_mm": 112.85, "y_mm": -112.91}
    sector, ring = sector_ring_for_point(112.85, -112.91)
    assert (sector, ring) == ("2", "single_outer")
    snapped, s, r = _corroborated_label_override(reads, combined, sector, ring)
    assert (s, r) == ("2", "single_outer")
    assert "label_override" not in snapped
