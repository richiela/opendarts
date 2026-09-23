"""Zeus -- the registry entry named "Zeus" (see
opendarts/engines/registry.py). A COMBINER engine, not its own
detector/calibration consumer: it runs the real, already-registered
sub-engines named in `ZEUS_SUB_ENGINE_NAMES` against the SAME
`bg_images`/`frame_images`/`calibration` triple Zeus itself was called
with, and picks whichever `(sector, ring)` answer at least
`MIN_SUB_ENGINES_TO_VOTE` of them agree on.

**2026-08-25 -- Ares added as a 4th voter.** Majority still wins, and
Apollo holds the tie-breaking vote. This is a deliberate, narrow override
of the standing docs/DESIGN.md guardrail that previously said Zeus's
voters were Apollo/Talos/Athena only -- it adds Ares specifically (see
docs/DESIGN.md's own dated entry for the standing-guardrail update).
`ZEUS_SUB_ENGINE_NAMES` is now `("Apollo", "Talos", "Athena", "Ares")`
-- Apollo stays FIRST in the tuple on purpose (see the tie-break section
below: this is what keeps Apollo's tie-breaking vote true). Measured on
the full session corpus (614 throws, 8 sessions, each
package's own stored `calibration.json`, operator-marked-wrong-aware
ground truth): the 3-engine config scores 612/614 (99.7%); adding
Ares as a 4th voter is EXACTLY UNCHANGED, still 612/614 (99.7%) --
zero throws flipped in either direction on this corpus. Apollo's own
tie-break fired on 2 real throws in this corpus (both genuine 2-2 ties,
Apollo+Talos vs Athena+Athena... see below) and both resolved
correctly. Full numbers, the two genuine tie-break throws, and the two
residual (pre-existing, Ares-independent) misses are in this task's
own final report -- see `dev/tests/test_engine_zeus_real_corpus.py`'s
module docstring for the reproducible measurement.

**Why this exists, and the real (now-historical) number it reproduces.**
A scratch prototype (in a development checkout, not
part of this package) measured 419/420 (99.8%) on the
full `data/archive/clean/` corpus (the OLD, now-superseded corpus
location -- see docs/DESIGN.md's 2026-08-22 data-layout guardrail) with a
fresh per-session wire-junction-refined calibration -- the best of any
engine measured at the time. This module is the real, registered engine that
reproduced that result faithfully -- see
`dev/tests/test_engine_zeus_real_corpus.py` for the measured proof.
**This 419/420 number describes the OLD 3-engine
(Apollo/Talos/Athena) configuration only, not today's 4-engine
config** -- preserved here as real historical record (this project's own
"measured, dated, preserved, not deleted" documentation convention), not
because it still describes current behavior.

**Tie-break convention -- preserved EXACTLY as prototyped, extended (not
reinvented) for a 4th voter.** Build a `collections.Counter` over the
sub-engines' `(sector, ring)` answers, counting only sub-engines that
returned `ok=True`. Find the max vote count; if more than one
`(sector, ring)` key is tied for that max, the winner is whichever key
was inserted into the vote-counting dict FIRST. Because votes are
collected by iterating `ZEUS_SUB_ENGINE_NAMES` in a fixed order
(`Apollo`, `Talos`, `Athena`, `Ares`) and a `Counter` built from
a dict's `.values()` preserves that same insertion order, this means:
whenever Apollo is part of the tied group (any tie Apollo
participates in -- a 2-way, 3-way, or 4-way tie), `Apollo`'s own
answer wins. This is the literal mechanism that implements "Apollo has
the tie breaking vote" -- not because it is judged "best" in any
per-throw sense, purely because that is the existing, already-measured,
deterministic tie-break the original 3-engine prototype used, now
extended to a 4th voter without changing its shape.

**Residual ordering (Talos before Athena before Ares) -- a real,
separate decision, secondary to Apollo's tie-break.** Apollo being
first is what guarantees Apollo wins any tie it's part of; where the
OTHER three sit relative to each other only matters for the rarer case
of a tie that does NOT include Apollo at all (Apollo returned
`ok=False`, or -- vanishingly rare -- all 4 sub-engines gave 4 different
answers with Apollo's own answer among the tied group at count 1, in
which case Apollo still wins per the paragraph above; the genuinely
distinct case is Apollo simply not voting). Chosen order: `Talos`,
then `Athena`, then `Ares` -- this matches `opendarts/engines/
registry.py`'s own `ENGINES` dict registration order exactly (Apollo,
Talos, Athena, Ares, then Zeus, which doesn't vote).
Rationale: registration order is a real, already-existing, legible
ordering with no per-engine favoritism baked in beyond "the order they
were built and added to this project" -- using it here means the
residual tie-break isn't a fresh, unexplainable judgment call invented
just for this rare case. See
`tests/test_engine_zeus.py::test_tie_not_involving_apollo_resolves_to_talos_next_in_priority_order`
for a real test pinning this behavior.

**The quorum gate -- raised from 2-of-3 to `MIN_SUB_ENGINES_TO_VOTE = 3`
(of 4) for the 4-engine config, a deliberate decision.** The old 2-of-3
gate required a real supermajority of the registered voter set (>=66.7%
present, at most 1 missing) before Zeus would trust a vote at all. Naively
keeping the floor at 2 after adding a 4th voter would let Zeus vote off
as few as HALF the registered engines (2 of 4), with 2 silently not
responding at all -- a real, unexamined weakening of what "the vote" is
supposed to mean, not a neutral carry-over. Raised to 3 instead: still
requires a real majority of the full 4-engine set to be present (>=75%),
consistent in spirit with the old 2-of-3 floor, and -- checked, not
assumed -- this costs NOTHING on the real corpus: measured 2026-08-25 on
the full 614-throw session corpus, only 2 of 614
packages ever had fewer than all 4 sub-engines return `ok=True` (both
still had exactly 3 usable), so floor=2 and floor=3 produce byte-identical
outcomes (612/614, same throws) on every real package measured. Kept at
3 anyway because the reasoning holds independently of today's
measurement: a genuinely degraded capture where only 2 of 4 engines
produce anything is exactly the scenario a quorum gate exists to be
skeptical of, and there's no measured cost to being more conservative
here. If a future corpus measurement finds real throws where floor=3
newly returns `ok=False` and floor=2 would have scored them correctly,
that's the moment to revisit this, with real numbers, not before.

**Diagnostics** are deliberately rich, matching this project's
established "enough to investigate a specific throw" bar:
every sub-engine's own full `(sector, ring, ok)` plus its own complete
`EngineResult.to_dict()` (so a human can drill from "Zeus said X" all the
way down to "here is exactly why Apollo/Talos/Athena each said what
they said"), the vote tally, and which sub-engine's answer won the
tie-break when there was one.

**2026-08-24 fix -- prior-dart contamination guard was not reaching
Apollo when Zeus is primary.** Real incident:
the recorded S10 throw (dart 2 of a 3-dart visit, thrown right after a D15 that landed 60mm away on the
board but close enough in camera 0's own 2D projection to trip
Apollo's `prior_dart_line_px` contamination signal on that camera --
see `opendarts.engines.apollo.prior_dart_context`'s own module
docstring for the general mechanism). With Zeus configured as the live
primary engine (this session's real config) and Apollo/Talos/Athena
all also-run, `opendarts.live.capture_daemon.handle_ready_to_capture()`'s
prior-dart lookup (added 2026-08-16) was gated on
`primary_name == "Apollo"` literally -- true only when Apollo itself
is configured as the primary engine, never when Zeus wraps it (as a
sub-engine call, right here in this module) or when Apollo runs as an
also-run engine (`opendarts.engines.dispatch.dispatch_engines()`, which
only ever threaded `prior_board_xy_mm`, Talos's own unrelated
prior-dart-erase feature). Result: Apollo got zero contamination
protection on this real throw and returned an honest `ok=False` (rays
disagreeing 10.9mm, camera 0's contaminated tip candidate paired with
camera 2); Talos and Athena have no equivalent guard at all and both
silently trusted the same bad camera-0 candidate, so Zeus's 2-of-3 vote
landed on their shared wrong answer (`single_inner`, 75.4mm from
center) instead of the correct `treble` (oracle: 102.8mm). Confirmed by
replay: passing the SAME package's `prior_dart_line_px` (recomputed
fresh from the prior throw's own stored raw frames, per this project's
"Replay is the source of truth" constraint) into `ApolloEngine.score()` directly
reproduces the ORIGINAL stored `ok=False` result byte-for-byte when
omitted, and correctly resolves to `treble` (94.16, -37.75mm --
~3.7mm from the oracle's 94.10, -41.42mm) when supplied.

The fix threads `prior_dart_line_px` into every sub-engine uniformly,
regardless of which one is the "primary" product answer: Zeus's own
sub-engine dispatch (this function) forwards it to Apollo specifically,
since only Apollo's `score()` declares the parameter. `opendarts.engines.dispatch.dispatch_engines()` (the also-run
path) and `opendarts.live.capture_daemon.handle_ready_to_capture()`'s
primary-engine branch condition were fixed in the same pass -- see
their own module comments for the paired half of this fix; all three
call sites needed fixing together for Apollo to get consistent
contamination protection regardless of how it's invoked (primary
directly, wrapped by Zeus as primary, or as an also-run engine).
"""
from __future__ import annotations

import concurrent.futures
import time
from collections import Counter

import numpy as np

from opendarts.engines.base import EngineResult
from opendarts.engines.apollo.prior_dart_context import (
    PriorDartLinePx,
    engine_accepts_prior_dart_line_px,
)
from opendarts.imageops import (
    DiffCrop,
    engine_accepts_precomputed,
    merge_requirements,
    precompute_diff_crop,
)
from opendarts.pipeline import CameraCalibration

# Fixed priority order this engine's own tie-break (and the quorum gate
# below) is defined in terms of -- see this module's own docstring,
# "Tie-break convention" and "Residual ordering" sections, for the full
# reasoning. Apollo FIRST is load-bearing (Apollo's tie-break); the
# rest (Talos, Athena, Ares) matches opendarts/engines/registry.py's
# own ENGINES registration order, chosen 2026-08-25 when Ares was
# added as the 4th voter -- see this module's docstring for why.
ZEUS_SUB_ENGINE_NAMES: tuple[str, ...] = ("Apollo", "Talos", "Athena", "Ares")

# Raised from 2 (of 3) to 3 (of 4) 2026-08-25 when Ares was added as
# a 4th voter -- see this module's own docstring, "The quorum gate"
# section, for the full reasoning and the real corpus measurement
# (floor=2 and floor=3 are byte-identical on the full real corpus as
# measured, so this is a deliberate conservative choice, not one forced
# by any observed regression).
MIN_SUB_ENGINES_TO_VOTE = 3


def _score_sub_engine(
    name: str,
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    calibration: dict[int, CameraCalibration],
    *,
    prior_dart_line_px: "PriorDartLinePx | None" = None,
    precomputed: "dict[int, DiffCrop] | None" = None,
) -> EngineResult:
    """Runs one sub-engine, never raising -- a sub-engine that throws is
    treated the same as one that returns ok=False (unusable for voting),
    not a crash that takes Zeus down with it. Mirrors
    opendarts.engines.dispatch._run_one's own "never raises" contract at
    the top-level dispatch layer, applied here one level down since Zeus
    calls its sub-engines directly (in parallel, via
    `_score_all_sub_engines()` below -- see that function's own
    docstring), not through `opendarts.engines.dispatch.dispatch_engines()`.

    `prior_dart_line_px` (2026-08-24 fix -- see this module's own
    docstring "Prior-dart contamination guard was not reaching Apollo
    when Zeus is primary" section): forwarded ONLY to whichever
    sub-engine's own `score()` actually declares the parameter, via the
    same capability check (`engine_accepts_prior_dart_line_px()`)
    `opendarts.live.capture_daemon.handle_ready_to_capture()` already uses
    for the direct-Apollo-primary case -- never assumed based on name
    alone, so a stub/fake sub-engine registered under "Apollo" with an
    older 3-arg signature keeps working unmodified (same discipline as
    every other real caller of this capability check).

    `precomputed` (2026-09-06 perf pass): the per-camera shared front end
    (`opendarts.imageops.DiffCrop`, built once by `_score_all_sub_engines()`
    for all four sub-engines), forwarded -- by the same duck-typed
    capability discipline (`engine_accepts_precomputed()`) -- only to a
    sub-engine whose `score()` declares the keyword. Each sub-engine
    validates the bundle against its own requirements and falls back to
    its own full-frame front end if it does not fit, so this is a pure
    latency change: every sub-engine's result is bit-identical with or
    without it.

    **`duration_s`/`timed_out` (2026-08-27, perf task)**: `EngineResult`'s
    own docstring says these are "NOT set by the engine's own score()...
    filled in by opendarts.engines.dispatch, which is the only code that
    actually knows how long a call took." That's exactly true for a
    sub-engine's OWN `score()` (never touched here) but this function
    itself is the one place that stands in for `dispatch_engines()` for
    Zeus's own sub-engine calls, so it now mirrors `dispatch.py`'s own
    `_run_one()` and stamps both fields itself: `duration_s` is the real
    wall-clock time this ONE sub-engine call took (honest, not a stand-in
    -- real timing data, not left at the dataclass default 0.0), and
    `timed_out` is always `False` because Zeus has no per-sub-engine
    timeout mechanism of its own (unlike `dispatch_engines()`'s shared
    `timeout_s` deadline) -- a sub-engine call here either finishes or
    this whole `score()` call blocks on it, exactly like before this
    task's parallelization. This is what lets
    `opendarts.live.capture_daemon`'s reused-from-Zeus also-run path (see
    that module's own dated comment) carry a real, honest `duration_s`
    instead of a silently-wrong 0.0."""
    from opendarts.engines.registry import get_engine # lazy: avoids a
    # registry<->zeus import cycle (registry.py imports ZeusEngine from
    # this package at module load time; this function only needs
    # get_engine() once score() is actually called, well after both
    # modules have finished importing).

    start = time.monotonic()
    try:
        engine = get_engine(name)
    except KeyError:
        return EngineResult(
            ok=False, sector=None, ring=None, board_xy_mm=None,
            reason=f"Zeus sub-engine {name!r} is not registered",
            duration_s=time.monotonic() - start, timed_out=False,
        )
    kwargs = {}
    if prior_dart_line_px is not None and engine_accepts_prior_dart_line_px(engine):
        kwargs["prior_dart_line_px"] = prior_dart_line_px
    if precomputed and engine_accepts_precomputed(engine):
        kwargs["precomputed"] = precomputed
    try:
        result = engine.score(bg_images, frame_images, calibration, **kwargs)
    except Exception as exc: # noqa: BLE001 -- see docstring above
        return EngineResult(
            ok=False, sector=None, ring=None, board_xy_mm=None,
            reason=f"Zeus sub-engine {name!r} raised {type(exc).__name__}: {exc}",
            duration_s=time.monotonic() - start, timed_out=False,
        )
    result.duration_s = time.monotonic() - start
    result.timed_out = False
    return result


def _score_all_sub_engines(
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    calibration: dict[int, CameraCalibration],
    *,
    prior_dart_line_px: "PriorDartLinePx | None" = None,
) -> dict[str, EngineResult]:
    """Runs all `ZEUS_SUB_ENGINE_NAMES` CONCURRENTLY -- 2026-08-27
    performance task. **Before this**: `ZeusEngine.score()` called
    `_score_sub_engine()` for each of the 4 sub-engines SEQUENTIALLY (a
    plain dict comprehension), so Zeus's own total latency was the SUM of
    all 4 sub-engine durations, even though the 4 calls share no state and
    have no data dependency on each other (each is an independent,
    pure-per-the-Engine-contract call against the SAME `bg_images`/
    `frame_images`/`calibration` triple). Real measured per-throw
    durations from live throws BEFORE this fix (see this task's own
    docs/DESIGN.md entry for the full table): sequential sums of
    0.44-0.56s vs a parallel max of 0.13-0.17s -- Zeus (the live
    PRIMARY engine, on the throw-scoring CRITICAL PATH per
    `opendarts.live.capture_daemon.handle_ready_to_capture()`, which calls
    `primary_engine.score()` synchronously and blocks the throw
    from being recorded/visible until it returns) was paying the full
    sequential sum on every single live throw.

    **Why a plain `ThreadPoolExecutor` context manager here, not
    `opendarts.engines.dispatch.dispatch_engines()`'s own more careful
    `concurrent.futures.wait(..., timeout=...)` + `shutdown(wait=False,
    cancel_futures=True)` pattern** (that module's own docstring explains
    why IT needs that care: a single shared deadline across
    simultaneously-started futures, and never blocking the caller on a
    thread it's already given up on for timing out). Neither concern
    applies here: Zeus's own sub-engine calls have NO per-engine timeout
    of their own today (never did, before or after this change -- a
    sub-engine call either finishes or this whole `score()` call blocks
    on it, exactly the same blocking contract the old sequential
    dict-comprehension had). With no timeout to enforce, there is no
    "restart each future's own clock" hazard to guard against, and no
    reason to avoid `pool.shutdown(wait=True)` (the default the `with`
    block below uses) -- every submitted future WILL complete (nothing
    here ever gives up on one early), so waiting for that is exactly the
    same blocking behavior this function replaces, just parallelized.
    Reaching for `dispatch_engines()`'s own machinery here would import a
    timeout concept Zeus has never had and this task was never asked to
    add -- REUSING it wasn't an option either: `dispatch_engines()`
    dispatches by REGISTRY NAME through `opendarts.engines.registry.ENGINES`
    directly, with no hook for `_score_sub_engine()`'s own
    `prior_dart_line_px` capability-filtering wrapper, so calling it here
    would mean either bypassing that wrapper (losing the 2026-08-24
    contamination-guard fix) or forking a parallel code path inside
    `dispatch_engines()` just for Zeus -- a real, unnecessary complexity
    a plain `ThreadPoolExecutor` avoids entirely.

    **Determinism preserved, checked, not assumed**: the tie-break
    convention (see this module's own docstring, "Tie-break convention")
    depends on `votes` being built by iterating the FIXED
    `ZEUS_SUB_ENGINE_NAMES` tuple in order, doing `votes[n] = sub_results[n]`
    for each -- NOT on `sub_results`' own dict insertion order. Since this
    function's return value is only ever consumed via `sub_results[n]`
    lookups in that fixed order (both in `ZeusEngine.score()`'s `votes`
    loop and in `base_diagnostics["sub_results"]`'s own dict
    comprehension), the ORDER sub-engine calls actually complete in
    (nondeterministic under real parallel execution -- whichever finishes
    first) has zero effect on Zeus's own vote outcome or its diagnostics
    shape. Confirmed by a dedicated test
    (`tests/test_engine_zeus.py::test_tie_break_deterministic_regardless_
    of_sub_engine_completion_order`) that forces a specific out-of-order
    completion sequence and asserts the winner is unchanged."""
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(ZEUS_SUB_ENGINE_NAMES), thread_name_prefix="zeus-sub-engine"
    ) as pool:
        precomputed = _precompute_shared_front_end(bg_images, frame_images, pool)
        future_to_name = {
            pool.submit(
                _score_sub_engine, n, bg_images, frame_images, calibration,
                prior_dart_line_px=prior_dart_line_px, precomputed=precomputed,
            ): n
            for n in ZEUS_SUB_ENGINE_NAMES
        }
        sub_results: dict[str, EngineResult] = {}
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            # _score_sub_engine() never raises (see its own docstring),
            # so future.result() here can't either -- no try/except
            # needed, unlike dispatch.py's own _run_one() callers.
            sub_results[name] = future.result()
    return sub_results


def _precompute_shared_front_end(
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    pool: concurrent.futures.ThreadPoolExecutor,
) -> "dict[int, DiffCrop] | None":
    """The four sub-engines open with the same per-camera front end:
    grayscale both images, |bg - frame|, 5x5 Gaussian blur, threshold.
    Run in four threads under one GIL that is 4x the same work fighting
    for the interpreter (2026-09-06 perf pass; see `opendarts.imageops.
    DiffCrop` for the exactness argument). This computes it ONCE per
    camera here -- cameras in parallel on the same pool, before the
    sub-engines are submitted -- and crops the blurred diff to where the
    frame actually changed (a median ~12% of the frame), so each
    sub-engine's threshold/morphology then runs on the crop instead of
    1280x720.

    Requirements come from each registered sub-engine's own
    `precompute_requirements` (blur kernel, threshold, morphology pad)
    and are merged: same kernel (or no bundle at all -- a different blur
    is a different image), the LOWEST threshold (the crop must hold every
    pixel any consumer thresholds in), the LARGEST pad. Returns None
    when nothing can be shared; sub-engines then run exactly as before.
    Never raises -- a failure here must not take a throw down, it just
    costs the optimization."""
    from opendarts.engines.registry import get_engine # lazy, see _score_sub_engine

    try:
        reqs = []
        for n in ZEUS_SUB_ENGINE_NAMES:
            try:
                engine = get_engine(n)
            except KeyError:
                continue
            req = getattr(engine, "precompute_requirements", None)
            if req is not None and engine_accepts_precomputed(engine):
                reqs.append(req)
        merged = merge_requirements(reqs)
        if merged is None:
            return None
        cams = sorted(set(bg_images) & set(frame_images))
        crops = list(pool.map(
            lambda cam: precompute_diff_crop(bg_images[cam], frame_images[cam], merged),
            cams,
        ))
    except Exception: # noqa: BLE001 -- optimization only, never fatal
        return None
    out = {cam: crop for cam, crop in zip(cams, crops) if crop is not None}
    return out or None


class ZeusEngine:
    """The registry entry named "Zeus" -- see opendarts/engines/registry.py
    and this module's own top docstring for the full design and the
    tie-break convention."""

    name = "Zeus"

    def score(
        self,
        bg_images: dict[int, np.ndarray],
        frame_images: dict[int, np.ndarray],
        calibration: dict[int, CameraCalibration],
        *,
        prior_dart_line_px: "PriorDartLinePx | None" = None,
    ) -> EngineResult:
        # `prior_dart_line_px` (2026-08-24, see this module's own
        # docstring): declaring this parameter on Zeus's OWN score() is
        # what makes `engine_accepts_prior_dart_line_px(zeus_engine)`
        # return True for callers (capture_daemon.py, replay.py) that use
        # that capability check to decide whether to bother looking it
        # up at all -- forwarded to `_score_sub_engine()` for every
        # sub-engine, which itself only actually passes it on to
        # Apollo (the one sub-engine whose own score() declares it).
        # 2026-08-27 perf task: parallelized (was a sequential dict
        # comprehension, one sub-engine at a time) -- see
        # _score_all_sub_engines()'s own docstring for the real measured
        # before/after numbers and why this is a pure performance change
        # (tie-break determinism unaffected).
        sub_results: dict[str, EngineResult] = _score_all_sub_engines(
            bg_images, frame_images, calibration,
            prior_dart_line_px=prior_dart_line_px,
        )

        # Only ok=True sub-results with a real (sector, ring) are usable
        # votes -- insertion order here is ZEUS_SUB_ENGINE_NAMES order,
        # which is what makes the tie-break below deterministic and
        # reproducing the prototype's exact behavior.
        votes: dict[str, tuple[str | None, str]] = {}
        for n in ZEUS_SUB_ENGINE_NAMES:
            r = sub_results[n]
            if r.ok:
                # A sub-engine can legitimately return ok=True with
                # sector=None (bull/outer_bull/outside have no sector,
                # per opendarts.geometry.board.sector_ring_for_point's own
                # convention)
                # -- what makes a result unusable for voting is ok=False,
                # not a None sector on its own.
                votes[n] = (r.sector, r.ring)

        base_diagnostics = {
            "sub_engine_names": list(ZEUS_SUB_ENGINE_NAMES),
            "sub_results": {n: sub_results[n].to_dict() for n in ZEUS_SUB_ENGINE_NAMES},
            "votes": {n: list(votes[n]) for n in votes},
            "n_usable": len(votes),
        }

        if len(votes) < MIN_SUB_ENGINES_TO_VOTE:
            return EngineResult(
                ok=False,
                sector=None,
                ring=None,
                board_xy_mm=None,
                reason=(
                    f"only {len(votes)} of {len(ZEUS_SUB_ENGINE_NAMES)} sub-engines "
                    f"produced a result -- need >={MIN_SUB_ENGINES_TO_VOTE} to vote"
                ),
                diagnostics=base_diagnostics,
            )

        # Counter over an insertion-ordered dict's .values() preserves
        # that same order (CPython dict/Counter both iterate in
        # insertion order) -- winners[0] below is deterministic, not
        # incidental. See this module's own docstring for why this exact
        # tie-break (not a "smarter" one) is intentional.
        counts = Counter(votes.values())
        best_count = max(counts.values())
        winners = [key for key, c in counts.items() if c == best_count]
        winner_key = winners[0]
        tie = len(winners) > 1

        winning_engine = next(n for n in ZEUS_SUB_ENGINE_NAMES if votes.get(n) == winner_key)

        # board_xy_mm has no well-defined combined value across
        # disagreeing sub-engines (they may not even agree on which
        # sector/ring, let alone a shared xy) -- take the winning
        # sub-engine's own board_xy_mm honestly, rather than inventing an
        # average across engines that voted for something else entirely.
        winner_board_xy_mm = sub_results[winning_engine].board_xy_mm

        vote_tally = [
            {"sector": key[0], "ring": key[1], "count": c}
            for key, c in counts.items()
        ]

        if best_count == len(votes):
            agreement = "unanimous"
        elif tie:
            agreement = "tie"
        else:
            agreement = "majority"

        reason = (
            f"Zeus {agreement} vote: {best_count}/{len(votes)} usable sub-engines "
            f"agree on sector={winner_key[0]!r} ring={winner_key[1]!r}"
        )
        if tie:
            # len(winners) is dynamic now (2/3/4-way, not always 3) --
            # see this module's own docstring "Tie-break convention" /
            # "Residual ordering" sections for the full 4-engine reasoning.
            reason += (
                f" ({len(winners)}-way tie broken in favor of "
                f"{winning_engine!r}'s own answer)"
            )

        return EngineResult(
            ok=True,
            sector=winner_key[0],
            ring=winner_key[1],
            board_xy_mm=winner_board_xy_mm,
            reason=reason,
            diagnostics={
                **base_diagnostics,
                "vote_tally": vote_tally,
                "winner": list(winner_key),
                "winning_engine": winning_engine,
                "tie_break_applied": tie,
                "agreement": agreement,
            },
        )
