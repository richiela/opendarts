"""opendarts/live/oracle_hold.py -- "don't score yet: the oracle isn't ready".

2026-09-28, asked for from the rig: with the oracle switched on, a dart
Autodarts can't read is a dart the corpus can't check, and the only
place anyone would notice Autodarts had stopped or was stuck in takeout
was Autodarts' own light -- invisible from a game on another screen. So
when the oracle is ON, OD itself is not ready while the oracle isn't:
it holds (reports takeout) and does not score. A dart thrown while held
is simply not scored; pull it, reset, throw again.

This module only answers the question. The lifecycle asks it through a
generic ``hold`` provider (opendarts/lifecycle/state.py, ``Lifecycle._commit``)
and knows nothing about Autodarts; the server reports the same answer on
``GET /api/state`` and folds it into the retail status. One function, so
the light and the scoring can never disagree about why.

With the oracle off (no listener in this process, or ``ad_enabled``
switched off) there is never a hold: nothing changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opendarts.live.board_status import (
    BOARD_STATUS_CAMERAS_DOWN,
    BOARD_STATUS_READY,
    BOARD_STATUS_STOPPED,
    BOARD_STATUS_TAKEOUT,
)

if TYPE_CHECKING:
    from opendarts.live.ad_ws_listener import AdWsListener

HOLD_TAKEOUT = "takeout"
HOLD_STOPPED = "stopped"
HOLD_OFFLINE = "offline"
#: connected and running by its own account, but its cameras deliver no
#: frames (BOARD_STATUS_CAMERAS_DOWN): it cannot see a dart. Never graced --
#: a blind board is not a reaction to the dart that just landed.
HOLD_CAMERAS = "cameras"
#: connected, but the board has not said what it is doing yet (a fresh
#: connection before its status is hydrated). Held too: "not known to be
#: ready" is not ready.
HOLD_UNKNOWN = "unknown"

#: A takeout younger than this at the moment a dart first shows up is that
#: dart's own doing, not a reason to hold it. Autodarts goes into takeout
#: on the third throw it detects, and it usually reports a throw BEFORE OD
#: sees it: across 1,933 matched corpus darts its event arrived a median
#: 72 ms before OD's capture, 5% of them 221 ms or more before, and at the
#: extreme 1.5 s before (OD's capture itself trails the dart's first frame
#: by up to 1.5 s more on a slow settle). A stuck takeout is minutes old;
#: a walk to the board and back is well over this.
TAKEOUT_GRACE_S = 3.0

_REASONS = {
    HOLD_TAKEOUT: "oracle in takeout",
    HOLD_STOPPED: "oracle stopped",
    HOLD_OFFLINE: "oracle offline",
    HOLD_CAMERAS: "oracle cameras down",
    HOLD_UNKNOWN: "oracle status unknown",
}


@dataclass(frozen=True)
class OracleHold:
    code: str
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "reason": self.reason}


def oracle_hold(
    listener: "AdWsListener | None", *, takeout_grace_s: float = 0.0
) -> OracleHold | None:
    """Why OD should not score right now because of the oracle, or None.
    Pure in-memory reads off the listener -- cheap enough for every tick.

    ``takeout_grace_s``: a takeout that began less than this long ago is
    not a hold (see TAKEOUT_GRACE_S). Reporting passes 0 -- the light shows
    every takeout; the scoring gate passes TAKEOUT_GRACE_S."""
    if listener is None or not listener.is_enabled():
        return None
    if not listener.is_connected():
        code = HOLD_OFFLINE
    else:
        status = listener.board_status()[0]
        if status == BOARD_STATUS_READY:
            return None
        if status == BOARD_STATUS_TAKEOUT:
            age = listener.board_status_age_sec()
            if takeout_grace_s > 0 and age is not None and age < takeout_grace_s:
                return None
            code = HOLD_TAKEOUT
        elif status == BOARD_STATUS_STOPPED:
            code = HOLD_STOPPED
        elif status == BOARD_STATUS_CAMERAS_DOWN:
            code = HOLD_CAMERAS
        else:
            code = HOLD_UNKNOWN
    return OracleHold(code, _REASONS[code])


def hold_reason_provider(listener: "AdWsListener | None"):
    """The lifecycle's ``hold`` provider for this listener: a reason string
    while held, None when free to score. Asked once, when a dart first
    shows up (see ``Lifecycle._step``), so it judges the oracle as it was
    when the dart landed."""
    def provider() -> "str | None":
        hold = oracle_hold(listener, takeout_grace_s=TAKEOUT_GRACE_S)
        return hold.reason if hold is not None else None
    return provider
