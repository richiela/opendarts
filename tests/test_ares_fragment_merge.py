"""Ares collinear-fragment merge (2026-09-28): a dart broken into collinear
diff pieces is offered as ONE extra merged candidate; pieces that only
look nearby are not; an intact dart gets no merge; and the shared
precomputed front end gives the same merge as the engine's own (see
opendarts/engines/ares/detection.py's MERGE_* constants)."""
from __future__ import annotations

import math
import numpy as np

from opendarts import imageops
from opendarts.engines.ares.detection import PRECOMPUTE_REQUIREMENTS, detect_candidates


def _scene():
    bg = np.full((480, 640, 3), 60, dtype=np.uint8)
    return bg, bg.copy()


def _segment(frame, a, b, width=3):
    import cv2

    cv2.line(
        frame,
        tuple(int(round(v)) for v in a),
        tuple(int(round(v)) for v in b),
        (200, 200, 200),
        width,
    )


def _merged(cands):
    return [c for c in cands if c.merged_fragments]


def test_broken_up_dart_is_offered_as_one_merged_candidate():
    """Three 50px pieces of one shaft separated by 35px gaps (wider than
    the 15px close can bridge) come back as three short candidates PLUS
    one merged candidate spanning the whole dart, on the drawn axis."""
    bg, frame = _scene()
    tip = np.array([120.0, 380.0])
    d = np.array([math.cos(math.radians(-30.0)), math.sin(math.radians(-30.0))])
    for s0 in (0.0, 85.0, 170.0):
        _segment(frame, tip + s0 * d, tip + (s0 + 50.0) * d)

    cands = detect_candidates(bg, frame)
    merged = _merged(cands)
    assert len(merged) == 1
    m = merged[0]
    assert m.merged_fragments == 3
    assert not m.degraded
    assert m.shaft_len_px > 180.0  # the three pieces end to end, not one piece
    tilt = math.degrees(math.acos(min(1.0, abs(float(np.array(m.axis_unit) @ d)))))
    assert tilt < 1.0
    ends = [np.array(m.tip_px), np.array(m.base_px)]
    assert min(np.linalg.norm(e - tip) for e in ends) < 6.0
    # The originals are all still offered; consensus decides.
    assert len(cands) - len(merged) == 3


def test_side_by_side_pieces_are_not_merged():
    """Two parallel pieces 12px apart and overlapping along their length
    (two darts side by side, or a shaft and a shadow edge) share no single
    line: the common line runs between them, 6px from each."""
    bg, frame = _scene()
    _segment(frame, (100, 200), (250, 200))
    _segment(frame, (120, 212), (270, 212))
    assert _merged(detect_candidates(bg, frame)) == []


def test_crossing_piece_is_not_merged():
    """A long piece running ACROSS the line (a flight edge, another dart)
    is not chained on even when its centre sits on the line."""
    bg, frame = _scene()
    _segment(frame, (100, 240), (200, 240))
    _segment(frame, (260, 200), (300, 280))  # centre (280, 240), ~63 deg off
    assert _merged(detect_candidates(bg, frame)) == []


def test_intact_dart_gets_no_merged_candidate():
    bg, frame = _scene()
    _segment(frame, (120, 380), (330, 260))
    cands = detect_candidates(bg, frame)
    assert cands and _merged(cands) == []


def test_merge_is_identical_through_the_shared_precomputed_crop():
    """Zeus hands every engine one cropped diff front end; the merge must
    read the same pixels and values through it as without it."""
    bg, frame = _scene()
    tip = np.array([120.0, 380.0])
    d = np.array([math.cos(math.radians(-30.0)), math.sin(math.radians(-30.0))])
    for s0 in (0.0, 85.0, 170.0):
        _segment(frame, tip + s0 * d, tip + (s0 + 50.0) * d)
    pc = imageops.precompute_diff_crop(bg, frame, PRECOMPUTE_REQUIREMENTS)
    assert pc is not None and not pc.is_full_frame
    plain = _merged(detect_candidates(bg, frame))
    shared = _merged(detect_candidates(bg, frame, precomputed=pc))
    assert len(plain) == len(shared) == 1
    for a, b in zip(plain, shared):
        assert (a.tip_px, a.base_px, a.fit_rms_px, a.merged_fragments) == (
            b.tip_px, b.base_px, b.fit_rms_px, b.merged_fragments
        )
