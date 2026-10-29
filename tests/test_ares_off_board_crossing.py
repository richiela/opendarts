"""opendarts.engines.ares: a dart that lands off the board is an OUT, even
when something on the board (a flight blob, a speck) also changed.

Two cameras' shaft lines crossing off the board are the real dart. Ares
used to discard any crossing beyond its tip gate (1.2x the double ring),
so the only hypothesis left was the on-board junk, and the OUT was scored
as a hit."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from opendarts.engines.ares.engine import TIP_GATE_RADIUS_MM, _CamRead, _consensus_select


def _read(cam, tip_xy, line=None):
    r = _CamRead.__new__(_CamRead)
    r.cam, r.contaminated, r.flipped = cam, False, False
    r.cand = SimpleNamespace(degraded=False, shaft_len_px=120.0, fit_rms_px=0.8, mean_width_px=8.0)
    r.tip_xy = np.asarray(tip_xy, dtype=np.float64)
    r.base_xy = None
    if line is not None:
        p, d = (np.asarray(v, dtype=np.float64) for v in line)
        r.line = (p, d / np.linalg.norm(d))
    else:
        r.line = None
    return r


def test_two_cameras_crossing_off_the_board_beat_an_on_board_blob():
    out = np.array([0.0, 230.0])                       # off the board, past the tip gate
    assert np.hypot(*out) > TIP_GATE_RADIUS_MM
    reads = {
        0: [_read(0, out + [0.0, 30.0], line=(out, (1.0, 1.0)))],
        1: [_read(1, out + [0.0, 30.0], line=(out, (-1.0, 1.0)))],
        2: [_read(2, (50.0, 50.0))],                  # a flight blob on the board, no shaft line
    }
    _, hyp, _ = _consensus_select(reads)
    assert hyp is not None and np.hypot(*(hyp - out)) < 1.0


def test_an_on_board_dart_still_wins_over_an_off_board_lone_crossing():
    dart = np.array([40.0, -60.0])
    reads = {
        0: [_read(0, dart, line=(dart, (1.0, 0.2)))],
        1: [_read(1, dart, line=(dart, (-0.3, 1.0)))],
        2: [_read(2, dart, line=(dart, (1.0, -1.0)))],
    }
    _, hyp, _ = _consensus_select(reads)
    assert np.hypot(*(hyp - dart)) < 1.0
