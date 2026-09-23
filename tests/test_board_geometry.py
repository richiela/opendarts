"""Tests for opendarts.geometry.board -- validated against known dartboard
facts (regulation dimensions, standard sector layout), not just internal
self-consistency."""
from __future__ import annotations

import math
import os

import pytest

from opendarts.geometry.board import (
    BULL_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_INNER_SCORING_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    INNER_RING_SCORING_OFFSET_MM,
    OUTER_BULL_RADIUS_MM,
    SECTOR_ANGLE_DEG,
    SECTOR_NUMBERS_CLOCKWISE,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_INNER_SCORING_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    polar_to_xy_mm,
    sector_center_angle_deg,
    sector_ring_for_point,
    sector_ring_to_token,
    wire_boundary_angle_deg,
    wire_intersection_landmarks,
)


def test_sector_count_and_uniqueness():
    assert len(SECTOR_NUMBERS_CLOCKWISE) == 20
    assert len(set(SECTOR_NUMBERS_CLOCKWISE)) == 20
    assert sum(SECTOR_NUMBERS_CLOCKWISE) == sum(range(1, 21))


def test_20_is_at_top_1_and_5_are_neighbors():
    # Standard dartboard: 20 at 12 o'clock, 1 clockwise-next, 5
    # counter-clockwise-next (5 and 20 are neighbors -- the S5/S20
    # near-wire example from docs/DESIGN.md).
    assert SECTOR_NUMBERS_CLOCKWISE[0] == 20
    assert SECTOR_NUMBERS_CLOCKWISE[1] == 1
    assert SECTOR_NUMBERS_CLOCKWISE[-1] == 5


@pytest.mark.parametrize("number", SECTOR_NUMBERS_CLOCKWISE)
def test_sector_center_angle_matches_polar_conversion(number):
    # 130mm is between treble_outer (107) and double_inner (162) --
    # the "single_outer" band, standard darts terminology (outer single
    # is the larger area between treble and double; inner single is the
    # smaller area between bull and treble). Parametrized over all 20
    # sectors (was 4) -- no bug found on manual check
    # of the other 16, but no coverage existed before.
    angle = sector_center_angle_deg(number)
    x, y = polar_to_xy_mm(130.0, angle)
    sector, ring = sector_ring_for_point(x, y)
    assert sector == str(number), f"number={number} angle={angle} got {sector}"
    assert ring == "single_outer"


def test_bull_and_outer_bull():
    assert sector_ring_for_point(0.0, 0.0) == (None, "bull")
    assert sector_ring_for_point(0.0, BULL_RADIUS_MM - 0.1) == (None, "bull")
    assert sector_ring_for_point(0.0, BULL_RADIUS_MM + 0.1) == (None, "outer_bull")
    assert sector_ring_for_point(0.0, OUTER_BULL_RADIUS_MM - 0.1)[1] == "outer_bull"


def test_treble_and_double_rings():
    # Sector 20 center line, at various radii.
    x, y = polar_to_xy_mm((TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "treble")
    # Between treble_outer and double_outer midpoint sits in "single_outer"
    # (the outer single band, between treble and double rings).
    x, y = polar_to_xy_mm((TREBLE_OUTER_RADIUS_MM + DOUBLE_OUTER_RADIUS_MM) / 2, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "single_outer")
    # Between outer_bull and treble_inner sits in "single_inner".
    x, y = polar_to_xy_mm((OUTER_BULL_RADIUS_MM + TREBLE_INNER_RADIUS_MM) / 2, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "single_inner")


def test_inner_scoring_radii_sit_the_measured_offset_inside_the_regulation_ones():
    """`INNER_RING_SCORING_OFFSET_MM` applies to the two INNER ring
    boundaries only -- see its own comment in opendarts/geometry/board.py.
    The regulation radii themselves must stay untouched: they are also the
    3D landmark model PnP calibrates against."""
    assert TREBLE_INNER_RADIUS_MM == 99.0
    assert DOUBLE_INNER_RADIUS_MM == 162.0
    assert TREBLE_INNER_SCORING_RADIUS_MM == TREBLE_INNER_RADIUS_MM - INNER_RING_SCORING_OFFSET_MM
    assert DOUBLE_INNER_SCORING_RADIUS_MM == DOUBLE_INNER_RADIUS_MM - INNER_RING_SCORING_OFFSET_MM
    # The default must stay inside the [1.25, 2.00]mm band where the
    # oracle-labelled corpus check below scores every throw correctly --
    # a shipped value outside it would be a real regression, not a taste
    # difference.
    assert 1.25 <= INNER_RING_SCORING_OFFSET_MM <= 2.00


def test_inner_ring_transitions_happen_at_the_scoring_radius_not_the_regulation_one():
    x, y = polar_to_xy_mm(TREBLE_INNER_SCORING_RADIUS_MM + 0.05, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "treble")
    x, y = polar_to_xy_mm(TREBLE_INNER_SCORING_RADIUS_MM - 0.05, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "single_inner")
    x, y = polar_to_xy_mm(DOUBLE_INNER_SCORING_RADIUS_MM + 0.05, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "double")
    x, y = polar_to_xy_mm(DOUBLE_INNER_SCORING_RADIUS_MM - 0.05, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "single_outer")


def test_outer_ring_and_bull_boundaries_are_unshifted():
    """The outer treble, outer double and bull boundaries are not
    shifted: the scoring offset applies to the two inner boundaries only,
    and these stay at their regulation values."""
    x, y = polar_to_xy_mm(TREBLE_OUTER_RADIUS_MM - 0.05, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "treble")
    x, y = polar_to_xy_mm(TREBLE_OUTER_RADIUS_MM + 0.05, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "single_outer")
    x, y = polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM - 0.05, 0.0)
    assert sector_ring_for_point(x, y) == ("20", "double")
    assert sector_ring_for_point(*polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM + 0.05, 0.0))[1] == "outside"
    assert sector_ring_for_point(0.0, BULL_RADIUS_MM - 0.05) == (None, "bull")
    assert sector_ring_for_point(0.0, OUTER_BULL_RADIUS_MM - 0.05)[1] == "outer_bull"


def test_ad_own_points_agree_with_our_geometry_on_the_real_corpus():
    """Oracle-accuracy check on the board geometry alone: for every real
    throw in data/archive/clean/ with matched oracle ground truth, the
    oracle's reported tip coordinate, run through our
    sector_ring_for_point(), must land in the (sector, ring) label the
    oracle reported for it. No opendarts detection, calibration or
    triangulation is involved, so a failure here is a pure board-geometry
    regression. Skips cleanly when the (gitignored) corpus isn't on this
    machine."""
    import json
    from pathlib import Path

    root = Path(os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT") or
                (Path(__file__).resolve().parent.parent / "data" / "archive" / "clean"))
    if not root.exists():
        pytest.skip(f"no real corpus at {root} -- this measurement needs real throws")

    disagreements = []
    n = 0
    for gt_path in sorted(root.rglob("ad_ground_truth.json")):
        gt = json.loads(gt_path.read_text())
        if not gt.get("matched") or gt.get("tip_xy_mm") is None:
            continue
        n += 1
        x_mm, y_mm = gt["tip_xy_mm"]
        got = sector_ring_for_point(float(x_mm), float(y_mm))
        if got != (gt.get("sector"), gt.get("ring")):
            disagreements.append(
                f"{gt_path.parent.name}: AD={(gt.get('sector'), gt.get('ring'))} "
                f"ours={got} r={math.hypot(x_mm, y_mm):.3f}mm"
            )
    if n == 0:
        pytest.skip(f"corpus at {root} has no AD-matched packages")
    assert not disagreements, (
        f"{len(disagreements)}/{n} real throws where AD's own tip coordinate lands in a "
        f"different segment than AD's own operator-confirmed label:\n "
        + "\n ".join(disagreements)
    )


def test_outside_beyond_double():
    sector, ring = sector_ring_for_point(0.0, DOUBLE_OUTER_RADIUS_MM + 1.0)
    assert ring == "outside"
    assert sector is None


def test_wire_intersection_landmarks_count_and_structure():
    points = wire_intersection_landmarks()
    # bull + 20 sectors x 4 rings
    assert len(points) == 1 + 20 * 4
    labels = {p.label for p in points}
    assert "bull" in labels
    assert "double_outer_20" in labels
    assert "treble_inner_5" in labels
    # All Z=0 (board face)
    assert all(p.xyz[2] == 0.0 for p in points)
    # double_outer points should all be at radius DOUBLE_OUTER_RADIUS_MM
    for p in points:
        if p.label.startswith("double_outer_"):
            r = math.hypot(p.x_mm, p.y_mm)
            assert r == pytest.approx(DOUBLE_OUTER_RADIUS_MM, abs=1e-6)


@pytest.mark.parametrize("number", SECTOR_NUMBERS_CLOCKWISE)
def test_wire_boundary_angle_is_a_real_boundary_not_a_sector_center(number):
    """Fixed 2026-08-12: `wire_intersection_
    landmarks()` used to place every point at `sector_center_angle_deg`
    -- a real naming/implementation mismatch, since a radial wire runs
    BETWEEN two adjacent sectors, not through
    one sector's own center. Confirm, for every sector, the wire angle is
    exactly half a sector width (9deg) away from that sector's own
    center -- i.e. genuinely a boundary, never coincident with a center."""
    center = sector_center_angle_deg(number)
    wire = wire_boundary_angle_deg(number)
    diff = min((wire - center) % 360.0, (center - wire) % 360.0)
    assert diff == pytest.approx(SECTOR_ANGLE_DEG / 2.0, abs=1e-9)


def test_wire_intersection_landmarks_match_real_ad_quad_angles():
    """Cross-check against the 4-point cardinal-wire quad's
    independently-derived positions -- both should agree it sits at
    board angles 9/99/189/279deg. Before the 2026-08-12 fix this cross-check
    would have failed for every sector (the old center-angle
    implementation never produced any of these 4 values) -- this is the
    real, numeric regression test for that fix, not just a structural
    shape check.

    The quad is built here rather than imported: the reference
    calibration file's own frame puts its four cardinal wire angles at
    279/9/99/189 deg, offset -90 deg from this project's board-angle
    convention, so adding 90 deg gives this project's own angles. Those
    four numbers are the whole of the independent derivation, and
    inlining them keeps this test of SHIPPED geometry independent of the
    reference-data reader (dev/calibration/real_correspondences.py),
    which is developer tooling and does not ship."""
    ad4pt_base_deg_original_frame = (279.0, 9.0, 99.0, 189.0)
    real_quad = [
        (*polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, (base_deg + 90.0) % 360.0), 0.0)
        for base_deg in ad4pt_base_deg_original_frame
    ]

    points = wire_intersection_landmarks()
    by_label = {p.label: p for p in points}
    for x_mm, y_mm, z_mm in real_quad:
        angle = math.degrees(math.atan2(x_mm, y_mm)) % 360.0
        # Find the wire_intersection_landmarks() double_outer point at
        # this same angle (checked by round-tripping through sector
        # number, not by re-deriving the angle formula a third time).
        match = None
        for number in SECTOR_NUMBERS_CLOCKWISE:
            if wire_boundary_angle_deg(number) == pytest.approx(angle, abs=1e-6):
                match = by_label[f"double_outer_{number}"]
                break
        assert match is not None, f"no wire_intersection_landmarks() point at angle {angle}"
        assert match.x_mm == pytest.approx(x_mm, abs=1e-6)
        assert match.y_mm == pytest.approx(y_mm, abs=1e-6)


# ---------------------------------------------------------------------------
# sector_ring_to_token() -- the inverse-ish companion added 2026-08-14 for
# opendarts.live.capture_daemon.handle_ready_to_capture()'s self-descriptive
# throw-package naming. Covers every ring value in the real
# vocabulary sector_ring_for_point() itself can return, plus the
# collapsing/edge-case behavior called out in the function's own
# docstring.
# ---------------------------------------------------------------------------


def test_sector_ring_to_token_bull():
    assert sector_ring_to_token(None, "bull") == "DB"


def test_sector_ring_to_token_outer_bull():
    assert sector_ring_to_token(None, "outer_bull") == "OB"


def test_sector_ring_to_token_outside():
    assert sector_ring_to_token(None, "outside") == "OUT"


@pytest.mark.parametrize("number", SECTOR_NUMBERS_CLOCKWISE)
def test_sector_ring_to_token_treble(number):
    assert sector_ring_to_token(str(number), "treble") == f"T{number}"


@pytest.mark.parametrize("number", SECTOR_NUMBERS_CLOCKWISE)
def test_sector_ring_to_token_double(number):
    assert sector_ring_to_token(str(number), "double") == f"D{number}"


@pytest.mark.parametrize("number", SECTOR_NUMBERS_CLOCKWISE)
def test_sector_ring_to_token_single_inner_and_outer_collapse_to_plain_s(number):
    # single_inner/single_outer score identically -- both must produce
    # the exact same plain "S{sector}" token, not two different labels
    # for the same scored value.
    assert sector_ring_to_token(str(number), "single_inner") == f"S{number}"
    assert sector_ring_to_token(str(number), "single_outer") == f"S{number}"


def test_sector_ring_to_token_matches_a_real_historical_example():
    # a real package directory name ending in "-T8" -- the "T8" token is
    # exactly what sector "8" + ring "treble" must produce.
    assert sector_ring_to_token("8", "treble") == "T8"


def test_sector_ring_to_token_rejects_an_unrecognized_ring():
    # A real, honest failure -- guards against silent vocabulary drift
    # between this function and sector_ring_for_point() (see this
    # function's own docstring) rather than guessing a fallback token.
    with pytest.raises(ValueError, match="unrecognized ring"):
        sector_ring_to_token("8", "not_a_real_ring")


def test_sector_ring_to_token_covers_every_real_ring_sector_ring_for_point_can_return():
    # Cross-check against the real inverse function's own output, at a
    # sweep of real board-plane points, rather than only the vocabulary
    # this test file already knows about -- proves sector_ring_to_token()
    # never raises on anything sector_ring_for_point() can actually
    # produce.
    for x_mm in range(-180, 181, 5):
        for y_mm in range(-180, 181, 5):
            sector, ring = sector_ring_for_point(float(x_mm), float(y_mm))
            token = sector_ring_to_token(sector, ring)
            assert isinstance(token, str) and token
