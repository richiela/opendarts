"""Tests for opendarts/live/board_status.py -- the standardized 4-name
board-status vocabulary (stopped/ready/takeout/unknown) shared by
opendarts/live/ad_ws_listener.py and opendarts/live/server.py for the Scoring
tab's AD indicator light (the second external board's light was removed
once it became a real registry engine).

2026-08-14: the indicator light uses one naming convention that is
standard across all boards rather than any source's own -- see
board_status.py's own module docstring for the full rationale. The four constants below ARE the wire-format
contract opendarts/live/server.py's state_dict() (`ad_board_status`) and
the dashboard's own JS `renderBoardStatus()` depend on, so pinning their
literal string spelling (not merely "4 distinct values") is the actual
point of the first block of tests below -- a silent rename of any one of
them would be a real breaking change to that wire format, not just an
internal refactor.

Fixtures for classify_board_status() are the exact two real example
snapshots documented in board_status.py's own module docstring (a real
AD-shaped dict, a real OpenDarts-shaped dict, both confirmed live
2026-08-14 against the actual rig), plus honest edge cases (empty dict,
None, missing keys, an unrecognized shape) the docstring also calls out
by name.
"""
from __future__ import annotations

from opendarts.live.board_status import (
    BOARD_STATUS_COLOR,
    BOARD_STATUS_READY,
    BOARD_STATUS_STOPPED,
    BOARD_STATUS_TAKEOUT,
    BOARD_STATUS_UNKNOWN,
)


# ---------------------------------------------------------------------------
# The vocabulary itself.
# ---------------------------------------------------------------------------


def test_the_four_status_constants_are_exactly_this_standardized_spelling():
    """Not any source's own status spelling --
    these four lowercase names are this project's own internal
    vocabulary, by deliberate design decision. A future refactor renaming
    one of these (even to something that reads as "equivalent") would
    silently break the dashboard's JS and any other consumer of
    state_dict()'s ad_board_status/od_board_status keys."""
    assert BOARD_STATUS_STOPPED == "stopped"
    assert BOARD_STATUS_READY == "ready"
    assert BOARD_STATUS_TAKEOUT == "takeout"
    assert BOARD_STATUS_UNKNOWN == "unknown"


def test_the_four_constants_are_pairwise_distinct():
    assert len(
        {BOARD_STATUS_STOPPED, BOARD_STATUS_READY, BOARD_STATUS_TAKEOUT, BOARD_STATUS_UNKNOWN}
    ) == 4


def test_board_status_color_covers_exactly_the_four_names_with_no_extras():
    assert set(BOARD_STATUS_COLOR) == {
        BOARD_STATUS_STOPPED, BOARD_STATUS_READY, BOARD_STATUS_TAKEOUT, BOARD_STATUS_UNKNOWN,
    }


def test_board_status_colors_match_the_documented_traffic_light_convention():
    assert BOARD_STATUS_COLOR[BOARD_STATUS_STOPPED] == "red"
    assert BOARD_STATUS_COLOR[BOARD_STATUS_READY] == "green"
    assert BOARD_STATUS_COLOR[BOARD_STATUS_TAKEOUT] == "yellow"
    assert BOARD_STATUS_COLOR[BOARD_STATUS_UNKNOWN] == "grey"


# ---------------------------------------------------------------------------
# classify_board_status() -- real example snapshots from the module's own
# docstring, confirmed live 2026-08-14 against the actual rig.
# ---------------------------------------------------------------------------

_REAL_AD_STOPPED_SNAPSHOT = {
    "connected": True, "running": False, "status": "Stopped",
    "event": "Stopped", "numThrows": 0,
}
_REAL_OD_STOPPED_SNAPSHOT = {
    "status": "Stopped", "phase": "Idle", "substate": "Wait", "running": False,
}

