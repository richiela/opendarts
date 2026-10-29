"""opendarts/live/oracle_watch.py -- notice when the oracle and this rig
disagree about HOW MANY darts are in the board, so the frames get kept.

The oracle (Autodarts) counts throws per visit; so does OpenDarts. When the
oracle is at 3 and OpenDarts is still at 2 a moment later, OpenDarts missed
a dart -- a grouped third dart on 2026-10-04 that only the oracle scored was
found by hand, 35 s later, after the 20 s frame ring had moved past it.
When OpenDarts is ahead of the oracle, OpenDarts probably counted a dart
that is not there (a phantom). Either way the frames around it are exactly
what the detection work needs, and they exist only for the ring's length.

The oracle is never used to SCORE here. A mismatch only triggers a capture
and a notice; the visit keeps the darts OpenDarts committed.

Judged while OpenDarts is running and not in its own takeout, and the
oracle is connected and either ready or in the takeout it starts by itself
on its third throw -- the moment a missed THIRD dart shows (3 against 2);
standing down for that one hid exactly that miss (2026-10-05 15:21:07). A
takeout the oracle has finished (its count back to 0) or OpenDarts' own
takeout is never judged: the two reset at slightly different moments. A
disagreement must last unchanged -- MISSED_GRACE_S when the oracle is ahead,
GRACE_S when OpenDarts is -- before it fires, once per visit and count.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: A disagreement must hold this long: the two detect the same dart a few
#: hundred ms apart, and OpenDarts can hold a grouped dart for a few frames.
GRACE_S = 2.5
#: Shorter when the oracle is ahead: a missed third dart is usually pulled
#: within a few seconds, and OpenDarts' own takeout ends the judging.
MISSED_GRACE_S = 1.5
#: At most one capture this often: a dump is ~300 MB.
MIN_INTERVAL_S = 60.0

MISSED = "oracle_saw_dart_od_missed"
EXTRA = "od_counted_dart_oracle_did_not"


@dataclass(frozen=True)
class Mismatch:
    kind: str           # MISSED or EXTRA
    visit_id: str
    oracle_n: int
    od_n: int
    oracle_call: "str | None"   # the oracle's latest throw, e.g. "S20" (a note, never a score)

    def reason(self) -> str:
        if self.kind == MISSED:
            return (f"oracle saw dart {self.oracle_n} ({self.oracle_call or '?'}) "
                    f"that OpenDarts did not (OpenDarts at {self.od_n})")
        return (f"OpenDarts counted dart {self.od_n} that the oracle did not "
                f"(oracle at {self.oracle_n}) -- possible phantom")

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "visit_id": self.visit_id, "oracle_n": self.oracle_n,
                "od_n": self.od_n, "oracle_call": self.oracle_call, "reason": self.reason()}


class OracleCountWatch:
    """Feed it both counts every poll; it returns a Mismatch the one time a
    disagreement has held for GRACE_S."""

    def __init__(self, grace_s: float = GRACE_S, min_interval_s: float = MIN_INTERVAL_S,
                 missed_grace_s: float = MISSED_GRACE_S) -> None:
        self.grace_s = grace_s
        self.missed_grace_s = missed_grace_s
        self.min_interval_s = min_interval_s
        self._key: "tuple | None" = None
        self._since = 0.0
        self._fired: set[tuple] = set()
        self._visit: "str | None" = None
        self._last_fire = float("-inf")

    def observe(self, *, judging: bool, visit_id: str, oracle_n: int, od_n: int,
                oracle_call: "str | None", now: float) -> "Mismatch | None":
        if visit_id != self._visit:
            self._visit, self._fired = visit_id, set()
        if not judging or oracle_n == od_n:
            self._key = None
            return None
        kind = MISSED if oracle_n > od_n else EXTRA
        key = (kind, visit_id, oracle_n, od_n)
        if key != self._key:
            self._key, self._since = key, now
            return None
        grace = self.missed_grace_s if kind == MISSED else self.grace_s
        if key in self._fired or now - self._since < grace:
            return None
        self._fired.add(key)
        if now - self._last_fire < self.min_interval_s:
            return None
        self._last_fire = now
        return Mismatch(kind, visit_id, oracle_n, od_n, oracle_call)
