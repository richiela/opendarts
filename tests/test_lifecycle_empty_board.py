"""The empty-board guards (2026-09-26 phantom S12 on the Pi).

A takeout was cleared while the hand was still pulling the last dart, so
the new reference held that dart and the hand; pulling it out then read as
a dart arriving where it had been. Scenes come from test_lifecycle_state.
"""
from __future__ import annotations

from tests.test_lifecycle_state import CX, CY, Scene, new_lifecycle, run, warm

from opendarts.lifecycle.signals import to_small_gray
from opendarts.lifecycle.state import Action, Phase

DART_A = (CX - 60, CY - 60, 16, 100)
DART_B = (CX + 40, CY - 40, 16, 90)
# a hand closed round dart B, on the board face: bright, still, and not
# moving outside the board -- the hand rule does not fire
HAND_ON_B = (CX + 30, CY - 50, 36, 110)


def _visit_of_two(lc, s):
    warm(lc, s)
    s.darts.append(DART_A)
    run(lc, s, 15)
    s.darts.append(DART_B)
    run(lc, s, 15)
    assert lc.dart_count == 2


def test_a_takeout_does_not_clear_while_the_last_dart_is_still_being_pulled():
    lc, s = new_lifecycle(), Scene()
    _visit_of_two(lc, s)
    s.darts = [DART_B]          # A pulled
    s.blobs = [HAND_ON_B]       # the hand is on B, still in the board
    ticks = run(lc, s, 30)
    assert not any(t.action is Action.CLEARED for t in ticks)
    assert any(t.reason == "waiting for an empty board" for t in ticks)
    s.darts, s.blobs = [], []   # B pulled, hand gone
    ticks = run(lc, s, 40)
    assert sum(t.action is Action.CLEARED for t in ticks) == 1
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert lc.dart_count == 0


def test_a_removal_is_not_scored_even_after_an_early_clear():
    # The reference already holds the hand and the dart (the early clear
    # happened); taking them away returns the board to empty. That is a
    # removal, not a throw.
    lc, s = new_lifecycle(), Scene()
    warm(lc, s)
    s.darts, s.blobs = [DART_B], [HAND_ON_B]
    frame = s.render()
    lc._adopt({0: frame}, {0: to_small_gray(frame)})
    s.darts, s.blobs = [], []
    ticks = run(lc, s, 30)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert any(t.reason == "a removal, not a dart" for t in ticks)
    assert lc.dart_count == 0 and lc.phase is Phase.IDLE


def test_a_real_dart_after_a_takeout_still_scores():
    lc, s = new_lifecycle(), Scene()
    _visit_of_two(lc, s)
    s.darts = []
    ticks = run(lc, s, 40)
    assert sum(t.action is Action.CLEARED for t in ticks) == 1
    s.darts = [DART_A]          # thrown again, into a spot a dart just left
    ticks = run(lc, s, 15)
    assert sum(t.action is Action.COMMIT for t in ticks) == 1
    assert lc.dart_count == 1


def test_a_takeout_that_never_looks_empty_clears_after_the_wait_and_its_removal_is_not_scored():
    # Something stays over where a dart was (a hand resting on it) long
    # past any pull: the board must not stay in the old visit for ever.
    # The clear it finally makes holds that hand and dart, so their later
    # removal is a removal, not a throw.
    lc, s = new_lifecycle(), Scene()
    _visit_of_two(lc, s)
    s.darts, s.blobs = [DART_B], [HAND_ON_B]
    ticks = run(lc, s, 150)
    assert sum(t.action is Action.CLEARED for t in ticks) == 1
    assert lc.dart_count == 0
    s.darts, s.blobs = [], []
    ticks = run(lc, s, 30)
    assert not any(t.action is Action.COMMIT for t in ticks)
    assert lc.dart_count == 0


DART_C = (CX - 10, CY + 30, 16, 90)


def _visit_of_three(lc, s):
    warm(lc, s)
    for d in (DART_A, DART_B, DART_C):
        s.darts.append(d)
        run(lc, s, 15)
    assert lc.dart_count == 3


def test_a_forgotten_dart_is_never_scored_when_it_is_pulled_later_and_play_carries_on():
    # Two darts pulled, one forgotten, player walks away; back ~5 s later
    # for the last one -- once with the hand seen, once with it missed.
    # Depending on what the cameras read, the turn is either held and then
    # cleared anyway, or kept open as a partial takeout; either way the
    # late pull is not a throw, and the next real dart scores.
    from tests.test_lifecycle_state import _arm_in, _arm_out
    for hand_seen in (True, False):
        lc, s = new_lifecycle(), Scene()
        _visit_of_three(lc, s)
        s.darts = [DART_C]                      # A and B pulled, C forgotten
        ticks = run(lc, s, 150)                 # ~5 s away from the board
        if hand_seen:
            ticks += _arm_in(lc, s)
            s.darts = []
            ticks += run(lc, s, 3) + _arm_out(lc, s)
        else:
            s.darts = []                        # the pull the hand rule missed
        ticks += run(lc, s, 60)
        assert not any(t.action is Action.COMMIT for t in ticks), hand_seen
        assert lc.dart_count == 0, hand_seen
        s.darts = [DART_B]                      # the next turn's first dart
        ticks = run(lc, s, 15)
        assert sum(t.action is Action.COMMIT for t in ticks) == 1, hand_seen
