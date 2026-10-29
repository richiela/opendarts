"""opendarts.engines.ares: a miss that only one camera's shaft line agrees
with is still an OUT, not "no score".

Consensus can settle on an off-board hypothesis and keep just the one
camera that agrees with it. One line with no tip point cannot fix a
position, so the solve returns nothing; the hypothesis already says the
dart is off the board."""
from __future__ import annotations

import numpy as np

from opendarts.engines.ares.engine import _off_board_hypothesis
from opendarts.engines.ares.fusion import LineObservation, fuse
from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM, sector_ring_for_point


def _line(point, direction, cam=0):
    d = np.asarray(direction, dtype=np.float64)
    return LineObservation(point=np.asarray(point, dtype=np.float64),
                           direction=d / np.linalg.norm(d), weight=0.34, cam=cam)


def test_one_line_off_the_board_is_scored_out():
    hyp = np.array([-85.9, -298.7])                      # the real case: 311 mm out
    lines = [_line((-200.5, -62.3), (0.44, -0.9))]
    assert fuse(lines, [], init_xy=hyp) is None          # the solve alone gives up
    fused = _off_board_hypothesis(hyp, lines, [])
    assert fused is not None
    assert sector_ring_for_point(*fused.xy_mm)[1] == "outside"


def test_an_on_board_hypothesis_is_never_scored_this_way():
    hyp = np.array([0.0, DOUBLE_OUTER_RADIUS_MM - 5.0])
    assert _off_board_hypothesis(hyp, [_line((0.0, 0.0), (0.0, 1.0))], []) is None
    assert _off_board_hypothesis(None, [], []) is None
