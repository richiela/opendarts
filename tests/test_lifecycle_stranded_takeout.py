"""A takeout that could never clear (2026-09-28 on the Pi).

After a clean pull of three darts the change against the reference covered
only 0.65-0.69 of the darts' area -- readopts while the darts settled had
absorbed part of it -- so the removal was taken as partial, the now-empty
board was adopted as "the darts left in", and the visit sat in takeout for
good. Scenes come from test_lifecycle_state.
"""
from __future__ import annotations

from tests.test_lifecycle_state import CX, CY, Scene, new_lifecycle, run, warm

from opendarts.lifecycle.signals import to_small_gray
from opendarts.lifecycle.state import (
    CLEARED_NOT_PARTIAL,
    CLEARED_STRANDED,
    Action,
    LifecycleConfig,
    Phase,
)

DART_A = (CX - 60, CY - 60, 16, 100)
DART_B = (CX + 40, CY - 40, 16, 90)
DART_C = (CX - 10, CY + 20, 16, 80)
# a sideways settle of this many pixels after landing, absorbed by the quiet
# readopts, leaves a clean pull covering ~0.5 of the darts' area: the
# partial band (``takeout_partial_coverage`` .. ``takeout_coverage``)
SETTLE_PX = 8


def _shifted(dart, dx):
    x, y, w, h = dart
    return (x + dx, y, w, h)


def _visit(lc, s, darts):
    warm(lc, s)
    for i in range(len(darts)):
        s.darts = list(darts[: i + 1])
        run(lc, s, 15)
    assert lc.dart_count == len(darts)


def _strand(lc, s, frame):
    """What the partial adopt did live: ``frame`` becomes the reference and
    the darts stay counted."""
    lc._adopt({0: frame}, {0: to_small_gray(frame)})
    lc._partial_adopted = True
    lc.phase = Phase.IDLE
    lc._c.reset_idle()


def _cleared(ticks, reason=None):
    return [
        t for t in ticks
        if t.action is Action.CLEARED and (reason is None or t.reason.endswith(f"({reason})"))
    ]


def test_a_clean_pull_read_as_partial_clears_the_visit():
    lc, s = new_lifecycle(), Scene()
    _visit(lc, s, [DART_A, DART_B])
    s.darts = [_shifted(d, SETTLE_PX) for d in (DART_A, DART_B)]
    settle = run(lc, s, 15)
    assert not any(t.action in (Action.COMMIT, Action.CLEARED) for t in settle)
    s.darts = []
    ticks = run(lc, s, 80)
    assert len(_cleared(ticks, CLEARED_NOT_PARTIAL)) == 1
    assert not any(t.action in (Action.PARTIAL_ADOPT, Action.COMMIT) for t in ticks)
    assert lc.dart_count == 0


def test_a_real_partial_removal_still_waits_for_the_rest():
    lc, s = new_lifecycle(), Scene()
    _visit(lc, s, [DART_A, DART_B])
    s.darts = [_shifted(d, SETTLE_PX) for d in (DART_A, DART_B)]
    run(lc, s, 15)
    s.darts = [_shifted(DART_B, SETTLE_PX)]          # A pulled, B still in
    ticks = run(lc, s, 120)
    assert not _cleared(ticks)
    assert lc.dart_count == 2
    s.darts = []                                      # B pulled
    ticks = run(lc, s, 120)
    assert len(_cleared(ticks)) == 1 and lc.dart_count == 0
    assert not any(t.action is Action.COMMIT for t in ticks)


def test_a_stranded_visit_clears_once_the_board_matches_the_empty_board():
    lc, s = new_lifecycle(), Scene()
    _visit(lc, s, [DART_A, DART_B, DART_C])
    s.darts = []
    _strand(lc, s, s.render())
    ticks = run(lc, s, 60)
    cleared = _cleared(ticks, CLEARED_STRANDED)
    assert len(cleared) == 1 and lc.dart_count == 0
    # about a second of stable frames, not the first one
    assert ticks.index(cleared[0]) + 1 >= lc.cfg.stranded_clear_frames
    assert not any(t.action is Action.COMMIT for t in ticks)


def test_the_backstop_never_clears_with_a_dart_still_in():
    lc, s = new_lifecycle(), Scene()
    _visit(lc, s, [DART_A, DART_B, DART_C])
    s.darts = [DART_B]
    _strand(lc, s, s.render())
    ticks = run(lc, s, 200)
    assert not _cleared(ticks) and lc.dart_count == 3


def test_the_backstop_waits_while_a_hand_is_in_view():
    lc, s = new_lifecycle(), Scene()
    _visit(lc, s, [DART_A, DART_B, DART_C])
    s.darts = []
    _strand(lc, s, s.render())
    hand_ticks = []
    for i in range(60):                               # an arm moving in the corner
        s.blobs = [(10 + 4 * (i % 10), 10, 90, 90)]
        hand_ticks += run(lc, s, 1)
    assert not _cleared(hand_ticks) and lc.dart_count == 3
    s.blobs = []
    assert len(_cleared(run(lc, s, 60), CLEARED_STRANDED)) == 1


def test_the_backstop_needs_a_known_empty_board():
    lc, s = new_lifecycle(), Scene()
    _visit(lc, s, [DART_A, DART_B, DART_C])
    s.darts = []
    _strand(lc, s, s.render())
    lc._empty = {}
    ticks = run(lc, s, 200)
    assert not _cleared(ticks) and lc.dart_count == 3


def test_the_backstop_does_not_end_a_visit_before_its_takeout():
    # one dart counted, nothing on the board, no partial removal: a visit
    # in progress, not a takeout -- leave it to the player's next dart
    lc, s = new_lifecycle(), Scene()
    _visit(lc, s, [DART_A])
    s.darts = []
    frame = s.render()
    lc._adopt({0: frame}, {0: to_small_gray(frame)})
    ticks = run(lc, s, 200)
    assert not _cleared(ticks) and lc.dart_count == 1


def test_the_backstop_frame_count_is_configurable():
    lc, s = new_lifecycle(LifecycleConfig(warmup_stable_frames=3, stranded_clear_frames=10)), Scene()
    _visit(lc, s, [DART_A, DART_B, DART_C])
    s.darts = []
    _strand(lc, s, s.render())
    ticks = run(lc, s, 30)
    cleared = _cleared(ticks, CLEARED_STRANDED)
    assert len(cleared) == 1 and 10 <= ticks.index(cleared[0]) + 1 < 20
