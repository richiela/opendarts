"""Zeus -- registry wiring + the (now 4-engine, floor=3) vote logic
itself, exercised against SYNTHETIC sub-engine results (mocked
Apollo/Talos/Athena/Ares via monkeypatching
`opendarts.engines.registry.get_engine`, the exact function
`opendarts.engines.zeus.engine._score_sub_engine` calls, lazily, to avoid a
registry<->zeus import cycle -- see that module's own docstring). This
file pins the VOTE LOGIC deterministically and cheaply; the real-corpus
proof (does the 4-engine config still match the 3-engine baseline on the
current corpus) lives in dev/tests/test_engine_zeus_real_corpus.py
(run once by hand 2026-08-25 -- see opendarts/engines/zeus/engine.py's
own module docstring for the numbers). The old 3-engine 419/420 number (a
DIFFERENT, now-historical config, on the OLD data/archive/clean/ corpus
location) is preserved as historical record in that same module
docstring, not reproduced by anything in this file anymore.

**2026-08-25: Ares added as Zeus's 4th voter, quorum floor raised
2->3.** Every test below that names sub-engines by hand now includes
Ares; `_ok`/`_miss` helpers are unchanged. See
`opendarts/engines/zeus/engine.py`'s own module docstring for the full
reasoning -- this file only pins the resulting
behavior, it doesn't re-derive it."""
from __future__ import annotations

import time

import pytest

from opendarts.engines.base import EngineResult
from opendarts.engines.registry import ENGINES, engine_names, get_engine, is_registered
from opendarts.engines.zeus import ZEUS_SUB_ENGINE_NAMES, ZeusEngine
from opendarts.engines.zeus.engine import MIN_SUB_ENGINES_TO_VOTE


# --- registry wiring -------------------------------------------------

def test_zeus_is_registered():
    assert "Zeus" in engine_names()
    assert is_registered("Zeus")
    engine = get_engine("Zeus")
    assert isinstance(engine, ZeusEngine)
    assert ENGINES["Zeus"] is engine
    assert engine.name == "Zeus"


def test_zeus_sub_engine_names_are_the_four_real_registered_engines_in_priority_order():
    # Priority order matters -- it's what the tie-break below is defined
    # in terms of (first-inserted-into-the-vote-dict wins a tie).
    # Apollo MUST be first. The
    # remaining three (Talos, Athena, Ares) match
    # opendarts/engines/registry.py's own ENGINES registration order --
    # a deliberate, documented choice for the rarer "tie not involving
    # Apollo" case, not incidental.
    assert ZEUS_SUB_ENGINE_NAMES == ("Apollo", "Talos", "Athena", "Ares")
    assert ZEUS_SUB_ENGINE_NAMES[0] == "Apollo"
    for name in ZEUS_SUB_ENGINE_NAMES:
        assert is_registered(name)


def test_min_sub_engines_to_vote_is_three_of_four():
    # Raised from 2 (of the old 3) to 3 (of the new 4) 2026-08-25 when
    # Ares was added as a voter -- see opendarts/engines/zeus/engine.py's
    # own module docstring ("The quorum gate" section) for the full
    # reasoning: naively keeping the floor at 2 would let Zeus vote off
    # as few as half the registered engines, an unexamined weakening,
    # not a neutral carry-over.
    assert MIN_SUB_ENGINES_TO_VOTE == 3


# --- synthetic sub-engine result plumbing -----------------------------

class _FakeEngine:
    """Stands in for a registered sub-engine -- returns a pre-canned
    EngineResult regardless of the images/calibration passed in, so vote
    logic can be tested without any real detection/triangulation code
    running."""

    def __init__(self, result: EngineResult) -> None:
        self._result = result

    def score(self, bg_images, frame_images, calibration) -> EngineResult:
        return self._result


def _ok(sector, ring, xy=(1.0, 2.0)) -> EngineResult:
    return EngineResult(ok=True, sector=sector, ring=ring, board_xy_mm=xy)


def _miss(reason: str = "no dart found") -> EngineResult:
    return EngineResult(ok=False, sector=None, ring=None, board_xy_mm=None, reason=reason)


def _patch_sub_engines(monkeypatch, mapping: dict[str, EngineResult]) -> None:
    """Monkeypatches opendarts.engines.registry.get_engine -- the exact
    function opendarts.engines.zeus.engine._score_sub_engine imports lazily
    and calls -- so ZeusEngine.score() sees exactly the synthetic results
    in `mapping` for each of ZEUS_SUB_ENGINE_NAMES, real registry lookups
    for anything else untouched."""
    real_get_engine = get_engine

    def fake_get_engine(name):
        if name in mapping:
            return _FakeEngine(mapping[name])
        return real_get_engine(name)

    monkeypatch.setattr("opendarts.engines.registry.get_engine", fake_get_engine)


@pytest.fixture()
def zeus() -> ZeusEngine:
    return ZeusEngine()


def _call(zeus: ZeusEngine) -> EngineResult:
    return zeus.score({}, {}, {})


# --- vote logic: unanimous, 3-1 majority, ties, <3 usable --------------

def test_unanimous_4_of_4_agree(monkeypatch, zeus):
    _patch_sub_engines(monkeypatch, {
        "Apollo": _ok("20", "double"),
        "Talos": _ok("20", "double"),
        "Athena": _ok("20", "double"),
        "Ares": _ok("20", "double"),
    })
    result = _call(zeus)
    assert result.ok is True
    assert (result.sector, result.ring) == ("20", "double")
    assert result.diagnostics["agreement"] == "unanimous"
    assert result.diagnostics["tie_break_applied"] is False
    assert result.diagnostics["n_usable"] == 4
    assert {"sector": "20", "ring": "double", "count": 4} in result.diagnostics["vote_tally"]


def test_3_of_4_majority_wins_over_the_lone_dissenter(monkeypatch, zeus):
    _patch_sub_engines(monkeypatch, {
        "Apollo": _ok("7", "single_outer"),
        "Talos": _ok("7", "single_outer"),
        "Athena": _ok("7", "single_outer"),
        "Ares": _ok("19", "treble"),
    })
    result = _call(zeus)
    assert result.ok is True
    assert (result.sector, result.ring) == ("7", "single_outer")
    assert result.diagnostics["agreement"] == "majority"
    assert result.diagnostics["tie_break_applied"] is False
    assert result.diagnostics["winning_engine"] == "Apollo"


def test_3_of_4_majority_only_needs_the_agreeing_three_to_be_ok(monkeypatch, zeus):
    # Fourth engine returns ok=False entirely (not just a different vote)
    # -- still a clean 3/3 majority among the usable trio, and exactly
    # at the MIN_SUB_ENGINES_TO_VOTE=3 floor (a real edge-of-quorum case).
    _patch_sub_engines(monkeypatch, {
        "Apollo": _ok("11", "bull"),
        "Talos": _ok("11", "bull"),
        "Athena": _ok("11", "bull"),
        "Ares": _miss(),
    })
    result = _call(zeus)
    assert result.ok is True
    assert (result.sector, result.ring) == ("11", "bull")
    assert result.diagnostics["n_usable"] == 3
    assert result.diagnostics["agreement"] == "unanimous" # 3/3 of the usable ones


def test_2_2_tie_with_apollo_in_the_tied_group_apollo_wins(monkeypatch, zeus):
    """The real, new-with-a-4th-voter scenario the project's own instruction
    is about ("Apollo has the tie breaking vote"): Apollo and Talos
    agree on one answer, Athena and Ares agree on a different one --
    a genuine 2-2 split, no majority. Apollo's own answer must win
    because Apollo is listed first in ZEUS_SUB_ENGINE_NAMES, so its
    (sector, ring) key is inserted into the vote Counter first. This is
    not a hypothetical -- see opendarts/engines/zeus/engine.py's own module
    docstring: this exact 2-2 shape occurred twice on the real corpus
    (2026-08-25 measurement) and resolved correctly both times."""
    _patch_sub_engines(monkeypatch, {
        "Apollo": _ok("12", "single_inner"),
        "Talos": _ok("12", "single_inner"),
        "Athena": _ok("5", "single_inner"),
        "Ares": _ok("5", "single_inner"),
    })
    result = _call(zeus)
    assert result.ok is True
    assert (result.sector, result.ring) == ("12", "single_inner")
    assert result.diagnostics["agreement"] == "tie"
    assert result.diagnostics["tie_break_applied"] is True
    assert result.diagnostics["winning_engine"] == "Apollo"
    assert result.diagnostics["n_usable"] == 4


def test_4way_tie_breaks_toward_apollo_first_inserted(monkeypatch, zeus):
    """All 4 disagree, no majority -- extends the original 3-engine
    tie-break convention (preserved, not reinvented) to a 4th voter: the winner is whichever (sector, ring)
    key was inserted into the vote-counting dict FIRST, which -- because
    votes are collected by iterating ZEUS_SUB_ENGINE_NAMES in order --
    means Apollo's own answer wins when all four give four different
    answers."""
    _patch_sub_engines(monkeypatch, {
        "Apollo": _ok("1", "single_inner"),
        "Talos": _ok("2", "single_inner"),
        "Athena": _ok("3", "single_inner"),
        "Ares": _ok("4", "single_inner"),
    })
    result = _call(zeus)
    assert result.ok is True
    assert (result.sector, result.ring) == ("1", "single_inner")
    assert result.diagnostics["agreement"] == "tie"
    assert result.diagnostics["tie_break_applied"] is True
    assert result.diagnostics["winning_engine"] == "Apollo"
    assert "4-way tie" in result.reason


def test_tie_not_involving_apollo_resolves_to_talos_next_in_priority_order(monkeypatch, zeus):
    """The residual-ordering case (step 1 of the Ares task): Apollo
    itself returns ok=False, so it casts no vote at all and cannot be
    "in" any tie -- the other three (Talos, Athena, Ares) each give
    a different answer, a genuine 3-way tie with exactly 3 usable votes
    (at the MIN_SUB_ENGINES_TO_VOTE=3 floor). The winner must be
    Talos's own answer: Talos is listed immediately after Apollo in
    ZEUS_SUB_ENGINE_NAMES, so among the USABLE votes it is inserted into
    the Counter first. This pins the deliberate residual ordering
    (Talos, then Athena, then Ares -- matching
    opendarts/engines/registry.py's own registration order, see
    opendarts/engines/zeus/engine.py's module docstring) against a real
    scenario, not just Apollo's own tie-break."""
    _patch_sub_engines(monkeypatch, {
        "Apollo": _miss(),
        "Talos": _ok("5", "outside"),
        "Athena": _ok("6", "outside"),
        "Ares": _ok("7", "outside"),
    })
    result = _call(zeus)
    assert result.ok is True
    assert (result.sector, result.ring) == ("5", "outside")
    assert result.diagnostics["agreement"] == "tie"
    assert result.diagnostics["tie_break_applied"] is True
    assert result.diagnostics["winning_engine"] == "Talos"
    assert result.diagnostics["n_usable"] == 3


def test_only_2_usable_subresults_returns_honest_no_score(monkeypatch, zeus):
    # Below the MIN_SUB_ENGINES_TO_VOTE=3 floor (2 of 4) -- deliberately
    # a NO-SCORE now, unlike the old 2-of-3 config where 2 usable was
    # already enough to vote. See opendarts/engines/zeus/engine.py's own
    # module docstring, "The quorum gate" section, for why this floor
    # was raised.
    _patch_sub_engines(monkeypatch, {
        "Apollo": _ok("20", "double"),
        "Talos": _ok("20", "double"),
        "Athena": _miss(),
        "Ares": _miss("timed out"),
    })
    result = _call(zeus)
    assert result.ok is False
    assert result.sector is None
    assert result.ring is None
    assert result.board_xy_mm is None
    assert "only 2 of 4" in result.reason
    assert "need >=3" in result.reason
    assert result.diagnostics["n_usable"] == 2


def test_0_usable_subresults_returns_honest_no_score(monkeypatch, zeus):
    _patch_sub_engines(monkeypatch, {
        "Apollo": _miss(),
        "Talos": _miss(),
        "Athena": _miss(),
        "Ares": _miss(),
    })
    result = _call(zeus)
    assert result.ok is False
    assert "only 0 of 4" in result.reason
    assert result.diagnostics["n_usable"] == 0


def test_a_raising_sub_engine_is_treated_as_unusable_not_a_crash(monkeypatch, zeus):
    class _RaisingEngine:
        def score(self, bg_images, frame_images, calibration):
            raise RuntimeError("boom")

    real_get_engine = get_engine

    def fake_get_engine(name):
        if name == "Talos":
            return _RaisingEngine()
        if name == "Apollo":
            return _FakeEngine(_ok("9", "single_inner"))
        if name == "Athena":
            return _FakeEngine(_ok("9", "single_inner"))
        if name == "Ares":
            return _FakeEngine(_ok("9", "single_inner"))
        return real_get_engine(name)

    monkeypatch.setattr("opendarts.engines.registry.get_engine", fake_get_engine)
    result = _call(zeus)
    # Talos raising should not crash Zeus -- Apollo + Athena +
    # Ares still form a usable 3-of-4 majority (exactly the floor).
    assert result.ok is True
    assert (result.sector, result.ring) == ("9", "single_inner")
    assert result.diagnostics["n_usable"] == 3
    talos_sub = result.diagnostics["sub_results"]["Talos"]
    assert talos_sub["ok"] is False
    assert "raised" in talos_sub["reason"]


def test_unknown_sub_engine_name_is_treated_as_unusable_not_a_crash(monkeypatch, zeus):
    def fake_get_engine(name):
        raise KeyError(name)

    monkeypatch.setattr("opendarts.engines.registry.get_engine", fake_get_engine)
    result = _call(zeus)
    assert result.ok is False
    assert result.diagnostics["n_usable"] == 0
    for name in ZEUS_SUB_ENGINE_NAMES:
        assert "not registered" in result.diagnostics["sub_results"][name]["reason"]


def test_diagnostics_include_full_nested_sub_engine_results(monkeypatch, zeus):
    """Rich-diagnostics contract: a human must be able to drill from
    "Zeus said X" down to each sub-engine's own full EngineResult,
    including ITS OWN diagnostics dict, not just a flat (sector, ring)."""
    apollo_result = EngineResult(
        ok=True, sector="20", ring="double", board_xy_mm=(1.0, 2.0),
        reason="triangulated", diagnostics={"max_ray_disagreement_mm": 0.4},
    )
    _patch_sub_engines(monkeypatch, {
        "Apollo": apollo_result,
        "Talos": _ok("20", "double"),
        "Athena": _ok("20", "double"),
        "Ares": _miss(),
    })
    result = _call(zeus)
    sub = result.diagnostics["sub_results"]
    assert set(sub.keys()) == {"Apollo", "Talos", "Athena", "Ares"}
    assert sub["Apollo"]["diagnostics"]["max_ray_disagreement_mm"] == 0.4
    assert sub["Apollo"]["reason"] == "triangulated"
    assert sub["Ares"]["ok"] is False
    assert result.diagnostics["votes"] == {
        "Apollo": ["20", "double"], "Talos": ["20", "double"],
        "Athena": ["20", "double"],
    }


def test_winner_board_xy_mm_is_the_winning_sub_engines_own_value(monkeypatch, zeus):
    _patch_sub_engines(monkeypatch, {
        "Apollo": _ok("20", "double", xy=(11.0, 22.0)),
        "Talos": _ok("20", "double", xy=(11.5, 21.5)),
        "Athena": _ok("20", "double", xy=(11.2, 21.8)),
        "Ares": _miss(),
    })
    result = _call(zeus)
    # Winning engine is whichever of the agreeing trio was inserted
    # first in priority order -- Apollo here.
    assert result.diagnostics["winning_engine"] == "Apollo"
    assert result.board_xy_mm == (11.0, 22.0)


def test_score_with_no_cameras_returns_ok_false_not_a_crash(zeus):
    """No monkeypatching -- runs the REAL Apollo/Talos/Athena/
    Ares sub-engines with empty images/calibration, exactly like
    OpenDartsEngine's own equivalent test. All four real engines handle
    empty input gracefully (ok=False), so Zeus's own <3-usable gate
    fires honestly."""
    result = zeus.score({}, {}, {})
    assert result.ok is False
    assert result.sector is None
    assert result.ring is None
    assert "need >=3" in result.reason


# ---------------------------------------------------------------------------
# 2026-08-27 perf task -- sub-engines run in parallel, not sequentially.
# Real measured before/after numbers live in docs/DESIGN.md's dated entry and
# this module's own docstring; these tests pin the two properties that
# actually matter for correctness: (1) it's genuinely faster (not just
# "should be"), (2) parallelizing doesn't change the deterministic
# tie-break outcome, regardless of which sub-engine happens to finish
# first under real thread scheduling.
# ---------------------------------------------------------------------------


class _SleepingFakeEngine:
    """Like _FakeEngine, but sleeps a controllable amount before
    returning -- lets a test simulate real per-engine latency without
    depending on any real detection code's own timing."""

    def __init__(self, result: EngineResult, sleep_s: float = 0.0) -> None:
        self._result = result
        self._sleep_s = sleep_s

    def score(self, bg_images, frame_images, calibration) -> EngineResult:
        if self._sleep_s:
            time.sleep(self._sleep_s)
        return self._result


def _patch_sub_engines_with_sleep(
    monkeypatch, mapping: dict[str, tuple[EngineResult, float]]
) -> None:
    """Same idea as _patch_sub_engines(), but each entry is
    (result, sleep_s) so a test can control per-sub-engine timing."""
    real_get_engine = get_engine

    def fake_get_engine(name):
        if name in mapping:
            result, sleep_s = mapping[name]
            return _SleepingFakeEngine(result, sleep_s)
        return real_get_engine(name)

    monkeypatch.setattr("opendarts.engines.registry.get_engine", fake_get_engine)


def test_score_runs_sub_engines_concurrently_not_sequentially(monkeypatch, zeus):
    """THE real performance fix: before 2026-08-27, ZeusEngine.score()
    called its 4 sub-engines via a plain sequential dict comprehension,
    so total latency was the SUM of all 4 durations. Real measured live
    numbers (see docs/DESIGN.md's dated entry): sequential sums of
    0.44-0.56s vs a parallel max of 0.13-0.17s on real throws. This test
    reproduces the same shape synthetically (4 sub-engines that each
    take SLEEP_S) and asserts wall time is close to ONE sleep, not four
    -- a generous bound (2.5x a single sleep), not a tight timing
    assertion, since CI/sandbox scheduling jitter is real."""
    SLEEP_S = 0.15
    _patch_sub_engines_with_sleep(monkeypatch, {
        "Apollo": (_ok("20", "double"), SLEEP_S),
        "Talos": (_ok("20", "double"), SLEEP_S),
        "Athena": (_ok("20", "double"), SLEEP_S),
        "Ares": (_ok("20", "double"), SLEEP_S),
    })

    started = time.monotonic()
    result = zeus.score({}, {}, {})
    elapsed = time.monotonic() - started

    assert result.ok is True
    assert elapsed < SLEEP_S * 2.5, (
        f"elapsed {elapsed:.3f}s -- expected close to one sleep "
        f"({SLEEP_S:.3f}s, parallel), not ~4x that ({SLEEP_S * 4:.3f}s, "
        f"the old sequential sum) -- this is the exact regression this "
        f"test exists to catch"
    )


def test_tie_break_deterministic_regardless_of_sub_engine_completion_order(monkeypatch, zeus):
    """Parallelizing sub-engine calls must NOT change the deterministic
    tie-break outcome -- see _score_all_sub_engines()'s own docstring,
    "Determinism preserved" section: votes/winner depend on iterating
    the FIXED ZEUS_SUB_ENGINE_NAMES tuple, never on completion order.
    Forces Apollo (the tie-break winner) to be the LAST future to
    actually finish -- a real race a naive completion-order-dependent
    implementation could get wrong -- and confirms Apollo's tie-break
    still fires correctly."""
    _patch_sub_engines_with_sleep(monkeypatch, {
        # Apollo finishes LAST (slowest) -- if anything depended on
        # completion order rather than the fixed tuple order, this is
        # exactly the case that would expose it.
        "Apollo": (_ok("20", "single_inner"), 0.08),
        "Talos": (_ok("5", "single_inner"), 0.0),
        "Athena": (_ok("20", "single_inner"), 0.0),
        "Ares": (_ok("5", "single_inner"), 0.0),
    })
    result = _call(zeus)
    # A genuine 2-2 tie ((20, single_inner) vs (5, single_inner));
    # Apollo is in the (20, single_inner) group, so it must win
    # regardless of finishing last under real thread scheduling.
    assert result.ok is True
    assert (result.sector, result.ring) == ("20", "single_inner")
    assert result.diagnostics["agreement"] == "tie"
    assert result.diagnostics["winning_engine"] == "Apollo"


def test_sub_results_carry_real_honest_duration_s_and_timed_out(monkeypatch, zeus):
    """`EngineResult`'s own docstring: duration_s/timed_out are normally
    filled in by opendarts.engines.dispatch, never by an engine's own
    score(). Zeus calls its sub-engines directly (not through
    dispatch_engines()), so `_score_sub_engine()` now stamps these
    itself (2026-08-27 perf task) -- this is what lets
    opendarts.live.capture_daemon's reused-from-Zeus also-run path carry
    REAL timing data instead of a silently-wrong 0.0/False default. Real
    per-sub-engine sleep times, asserted against the actual
    diagnostics["sub_results"][name]["duration_s"] Zeus wrote."""
    SLEEP_S = {"Apollo": 0.05, "Talos": 0.0, "Athena": 0.0, "Ares": 0.0}
    _patch_sub_engines_with_sleep(monkeypatch, {
        n: (_ok("20", "double"), s) for n, s in SLEEP_S.items()
    })
    result = _call(zeus)
    sub_results = result.diagnostics["sub_results"]
    for name in ZEUS_SUB_ENGINE_NAMES:
        assert sub_results[name]["timed_out"] is False
        # Real timing data, not the dataclass default 0.0 -- the slept
        # engine's own recorded duration must be at least its real sleep.
        assert sub_results[name]["duration_s"] >= SLEEP_S[name]
    # Apollo (the one that actually slept) has a measurably larger
    # duration_s than one that didn't -- confirms this is real per-call
    # timing, not a shared/copied value.
    assert sub_results["Apollo"]["duration_s"] > sub_results["Talos"]["duration_s"]


# --- shared front end (2026-09-06): one DiffCrop per camera, handed only
# to sub-engines that declare the `precomputed` keyword ------------------


def test_shared_front_end_reaches_only_capable_sub_engines(monkeypatch, zeus):
    import cv2
    import numpy as np

    from opendarts.imageops import DiffCrop, PrecomputeRequirements

    seen: dict[str, object] = {}

    class _Capable:
        precompute_requirements = PrecomputeRequirements(ksize=5, threshold=25.0, pad_px=19)

        def __init__(self, name, result):
            self._name, self._result = name, result

        def score(self, bg_images, frame_images, calibration, *, precomputed=None):
            seen[self._name] = precomputed
            return self._result

    class _Legacy:
        def __init__(self, name, result):
            self._name, self._result = name, result

        def score(self, bg_images, frame_images, calibration):
            seen[self._name] = "not-passed"
            return self._result

    fakes = {
        "Apollo": _Capable("Apollo", _ok("20", "triple")),
        "Talos": _Capable("Talos", _ok("20", "triple")),
        "Athena": _Legacy("Athena", _ok("20", "triple")),
        "Ares": _Capable("Ares", _ok("20", "triple")),
    }
    real_get_engine = get_engine
    monkeypatch.setattr(
        "opendarts.engines.registry.get_engine",
        lambda name: fakes[name] if name in fakes else real_get_engine(name),
    )

    bg = np.full((120, 160, 3), 80, np.uint8)
    fr = bg.copy()
    cv2.line(fr, (40, 30), (110, 90), (220, 220, 220), 3)
    result = zeus.score({0: bg, 1: bg}, {0: fr, 1: bg.copy()}, {0: object(), 1: object()})
    assert result.ok and (result.sector, result.ring) == ("20", "triple")

    assert seen["Athena"] == "not-passed"
    for name in ("Apollo", "Talos", "Ares"):
        pc = seen[name]
        assert isinstance(pc, dict) and set(pc) == {0, 1}
        assert all(isinstance(v, DiffCrop) for v in pc.values())
    # Every capable sub-engine gets the SAME bundle objects (computed once).
    assert seen["Apollo"][0] is seen["Talos"][0] is seen["Ares"][0]
    # Camera 0 changed along a streak -> cropped; camera 1 did not -> whole frame.
    assert not seen["Apollo"][0].is_full_frame
    assert seen["Apollo"][1].is_full_frame
    assert seen["Apollo"][0].accepts(_Capable.precompute_requirements, bg.shape)


def test_shared_front_end_is_skipped_when_no_sub_engine_can_use_it(monkeypatch, zeus):
    """Legacy-signature sub-engines (every existing test double) must see
    exactly the call they always did -- no `precomputed` kwarg."""
    _patch_sub_engines(monkeypatch, {
        "Apollo": _ok("20", "triple"), "Talos": _ok("20", "triple"),
        "Athena": _ok("20", "triple"), "Ares": _ok("20", "triple"),
    })
    import numpy as np

    bg = np.zeros((32, 32, 3), np.uint8)
    result = zeus.score({0: bg}, {0: bg}, {0: object()})
    assert result.ok and result.sector == "20"


def test_shared_front_end_failure_is_not_fatal(monkeypatch, zeus):
    from opendarts.imageops import PrecomputeRequirements

    seen: dict[str, object] = {}

    class _Capable:
        precompute_requirements = PrecomputeRequirements(ksize=5, threshold=25.0, pad_px=19)

        def __init__(self, name):
            self._name = name

        def score(self, bg_images, frame_images, calibration, *, precomputed=None):
            seen[self._name] = precomputed
            return _ok("20", "triple")

    fakes = {n: _Capable(n) for n in ("Apollo", "Talos", "Athena", "Ares")}
    real_get_engine = get_engine
    monkeypatch.setattr(
        "opendarts.engines.registry.get_engine",
        lambda name: fakes[name] if name in fakes else real_get_engine(name),
    )
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("cv2 exploded")

    monkeypatch.setattr("opendarts.engines.zeus.engine.precompute_diff_crop", boom)
    result = zeus.score({0: "not-an-image"}, {0: "not-an-image"}, {0: object()})
    assert calls, "the precompute was attempted"
    assert result.ok and result.sector == "20"
    assert all(v is None for v in seen.values()) # sub-engines ran without a bundle
