"""opendarts.live.oracle_watch: when the oracle and this rig disagree on how
many darts are in the board, fire once, after the disagreement has lasted."""
from __future__ import annotations

from opendarts.live.oracle_watch import EXTRA, MISSED, OracleCountWatch


def _run(watch, steps, judging=True, visit="v1", call="S20"):
    """steps: [(t, oracle_n, od_n)] -> the mismatches fired."""
    out = []
    for t, a, o in steps:
        m = watch.observe(judging=judging, visit_id=visit, oracle_n=a, od_n=o, oracle_call=call, now=t)
        if m is not None:
            out.append(m)
    return out


def test_a_dart_only_the_oracle_saw_fires_once_after_the_grace():
    # The 2026-10-04 22:14:46 visit: the oracle reached 3, OpenDarts stayed at 2.
    w = OracleCountWatch(grace_s=2.5)
    fired = _run(w, [(t / 2, 3, 2) for t in range(0, 20)])
    assert len(fired) == 1
    m = fired[0]
    assert m.kind == MISSED and (m.oracle_n, m.od_n, m.oracle_call) == (3, 2, "S20")
    assert "oracle saw dart 3 (S20)" in m.reason()


def test_the_two_detecting_a_moment_apart_is_not_a_mismatch():
    w = OracleCountWatch(grace_s=2.5)
    assert _run(w, [(0.0, 1, 0), (0.5, 1, 0), (1.0, 1, 1), (5.0, 1, 1)]) == []


def test_this_rig_ahead_of_the_oracle_is_a_possible_phantom():
    w = OracleCountWatch(grace_s=2.5)
    fired = _run(w, [(t / 2, 0, 1) for t in range(0, 10)])
    assert [m.kind for m in fired] == [EXTRA]


def test_takeouts_and_holds_are_never_judged():
    w = OracleCountWatch(grace_s=2.5)
    assert _run(w, [(t / 2, 0, 3) for t in range(0, 20)], judging=False) == []


def test_a_new_visit_can_fire_again_but_not_within_the_interval():
    w = OracleCountWatch(grace_s=1.0, min_interval_s=60.0)
    assert len(_run(w, [(0, 3, 2), (1.5, 3, 2)], visit="v1")) == 1
    assert _run(w, [(10, 3, 2), (11.5, 3, 2)], visit="v2") == []        # 10 s later: rate-limited
    assert len(_run(w, [(100, 3, 2), (101.5, 3, 2)], visit="v3")) == 1  # a minute on: fires


def test_a_missed_third_dart_fires_before_a_quick_pull():
    # 2026-10-05 15:21:07: the oracle reached 3 and went into its own
    # takeout; OpenDarts stayed at 2. The darts were pulled a few seconds
    # later, so the oracle-ahead case fires after 1.5 s, not 2.5.
    w = OracleCountWatch()
    fired = _run(w, [(0.0, 3, 2), (0.5, 3, 2), (1.0, 3, 2), (1.5, 3, 2)])
    assert [(m.kind, m.oracle_n, m.od_n) for m in fired] == [(MISSED, 3, 2)]
    w2 = OracleCountWatch()
    assert _run(w2, [(0.0, 0, 1), (0.5, 0, 1), (1.0, 0, 1), (1.5, 0, 1)]) == []   # the other way still waits 2.5 s
