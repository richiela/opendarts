"""opendarts/live/board_status.py -- standardized board-status vocabulary
shared by opendarts/live/ad_ws_listener.py and opendarts/live/server.py (the
Scoring tab's Autodarts indicator light -- the second external board's
light was removed once it became a real registry engine). A separate module, not
defined in server.py, specifically so the WS listener module can import
it without a circular import (server.py already imports it too).

2026-08-14: the indicator light uses one naming standard across every
board rather than any one system's. Neither Autodarts' event spelling ("Stopped"/
"Throw detected"/"Takeout finished") nor OpenDarts' own ("Stopped"/
"Idle"/"Wait"/"Takeout", per its written API doc) is used as this
project's internal vocabulary -- both get normalized INTO these four
names, the same way this project already has its own standardized
ThrowState/PRIMARY_INFO vocabulary for its OWN board rather than
adopting either external system's spelling.
"""
from __future__ import annotations


BOARD_STATUS_STOPPED = "stopped"
BOARD_STATUS_READY = "ready"
BOARD_STATUS_TAKEOUT = "takeout"
BOARD_STATUS_UNKNOWN = "unknown"

# Colors are a separate, trivial derived step so "what state is it in"
# and "what color do we show" can never drift apart into two competing
# sources of truth.
BOARD_STATUS_COLOR = {
    BOARD_STATUS_STOPPED: "red",
    BOARD_STATUS_READY: "green",
    BOARD_STATUS_TAKEOUT: "yellow",
    BOARD_STATUS_UNKNOWN: "grey",
}

