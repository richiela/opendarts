"""Apollo's per-camera vote override only overrides with cameras that agree
on where the dart is, and never reports a position that contradicts its call.

Each camera's own Z=0 point is stubbed per pixel, so these test the voting
rule itself rather than any camera geometry.
"""
from __future__ import annotations

import pytest

import opendarts.engines.apollo.engine as apollo
from opendarts.pipeline import ScoreResult


def _fused(ring, sector, xy):
    return ScoreResult(
        ok=True, sector=sector, ring=ring, board_xy_mm=xy, triangulation=None,
        n_cameras_used=3, cameras_used=(0, 1, 2), max_ray_disagreement_mm=3.0,
    )


def _override(monkeypatch, fused, points):
    """points: {cam: (x, y)} -- that camera's own single-ray board point."""
    pixels = {cam: (float(cam), 0.0) for cam in points}
    monkeypatch.setattr(apollo, "_single_ray_board_xy", lambda px, calib: points[int(px[0])])
    return apollo._per_camera_vote_override(fused, pixels, {}, {}, {cam: object() for cam in points})


def test_two_cameras_either_side_of_the_bull_do_not_turn_a_bull_into_25(monkeypatch):
    # The real 2026-09-26 throw: fused bull at 4.0mm; cam0 and cam2 each "outer
    # bull" but 13mm apart, on opposite sides -- their average is dead centre.
    fused = _fused("bull", None, (3.8, -1.28))
    out = _override(monkeypatch, fused, {0: (-6.8, -1.03), 1: (4.87, 0.18), 2: (6.28, -1.45)})
    assert (out.ring, out.board_xy_mm) == ("bull", (3.8, -1.28))


def test_two_cameras_that_agree_on_the_spot_still_override(monkeypatch):
    # Two cameras 4mm apart, both in single 5; the fused answer was single 20.
    # (6.1mm apart also overrode correctly on a real throw -- see the limit.)
    fused = _fused("single_inner", "20", (-6.0, 95.0))
    out = _override(monkeypatch, fused, {0: (-17.0, 94.0), 1: (-5.0, 95.0), 2: (-16.0, 90.2)})
    assert (out.sector, out.ring) == ("5", "single_inner")
    assert apollo.sector_ring_for_point(*out.board_xy_mm) == ("5", "single_inner")


def test_one_camera_alone_never_overrides(monkeypatch):
    # Nobody agrees with the fused bed individually, and one camera votes
    # elsewhere: one camera is not a majority of anything.
    fused = _fused("treble", "9", (0.0, 103.0))
    out = _override(monkeypatch, fused, {0: (0.0, 110.0)})
    assert (out.sector, out.ring) == ("9", "treble")


@pytest.mark.parametrize("gap_mm", [13.1, 24.9])
def test_cameras_far_apart_do_not_override(monkeypatch, gap_mm):
    fused = _fused("double", "19", (-50.0, -155.0))
    a = (-50.0, -172.0 - gap_mm / 2)
    b = (-50.0, -172.0 + gap_mm / 2)
    out = _override(monkeypatch, fused, {0: a, 1: (-50.0, -160.0), 2: b})
    assert (out.sector, out.ring) == ("19", "double")
