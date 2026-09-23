"""Replay — the actual point of opendarts/capture/. See docs/DESIGN.md's
"Replay is the source of truth". This module NEVER reads a package's stored result to
produce its answer -- it reruns tip detection + triangulation + scoring
from the stored RAW images + calibration through whatever pipeline code
is CURRENTLY installed, so it can differ from the original if the
pipeline has changed since capture.

Engine-aware (docs/ENGINES.md's "Offline tooling" section, added
2026-08-12): `replay_throw_with_engine()`/`replay_and_compare()` accept
any registered engine name via a REQUIRED `engine_name` argument -- this
is what makes testing a brand new engine against the full real corpus
(250+ real throws) possible before ever enabling it live, per that
section.

**2026-09-05 -- `engine_name` made a REQUIRED argument everywhere in
this module; `replay_throw()` (the old `Apollo`-only convenience
wrapper, and the source of this bug) DELETED entirely.** the project's own
framing: "We need to fix the replay_throw call so it doesn't default to
cv1 ... that's just a bug waiting to happen," "Or just get rid of it
since we have an explicit with_engine function." Real, confirmed
incident this closes: both `replay_throw()` and `replay_throw_with_
engine()`'s own `engine_name: str = DEFAULT_PRIMARY_ENGINE` default
(`"Apollo"`) silently diverged from what live production actually
scores with (`Zeus`, the 4-engine consensus) -- an offline replay
comparison run all day against the SILENT Apollo default, compared
against live 4-engine consensus output, produced a false "huge
STORE!=SCORE bug" report to that didn't actually exist. A silent-
wrong-engine default in a REPLAY tool is exactly the "confidently wrong,
not visibly wrong" failure class this project's own calibration work
has repeatedly had to guard against elsewhere (see docs/DESIGN.md's rig-
consensus-orientation entries) -- the fix here is the same shape:
refuse to guess, make the caller say what they mean.

`replay_throw()` itself was judged NOT worth keeping even as a thinner,
required-engine wrapper: it had exactly one internal caller
(`replay_and_compare()`, fixed in place below) and its own `ScoreResult`
return shape adds nothing a caller can't already get from
`replay_throw_with_engine(package, engine_name)` +
`opendarts.engines.apollo.engine_result_to_score_result()` (for
Apollo) or `opendarts.engines.base.engine_result_to_score_result()` (for
any other engine) -- exactly the two-line conversion `replay_and_
compare()` now does inline. Keeping a same-shaped wrapper around that
conversion would have reintroduced the exact same "which engine does
this actually mean" ambiguity one call deeper, for zero real
convenience gained.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opendarts.capture.throw_package import ThrowPackage, load_throw_package
from opendarts.engines.base import EngineResult
from opendarts.engines.base import engine_result_to_score_result as generic_engine_result_to_score_result
from opendarts.engines.apollo import engine_result_to_score_result as apollo_engine_result_to_score_result
from opendarts.engines.apollo.prior_dart_context import (
    engine_accepts_prior_dart_line_px,
    find_prior_dart_line_px,
)
from opendarts.engines.talos.prior_dart import (
    engine_accepts_prior_board_xy_mm,
    find_prior_board_xy_mm,
)
from opendarts.engines.registry import DEFAULT_PRIMARY_ENGINE, get_engine
from opendarts.pipeline import ScoreResult


@dataclass
class ReplayComparison:
    package_dir: Path
    fresh_result: ScoreResult
    original_result: dict | None
    sector_changed: bool | None # None if no original_result to compare against
    board_xy_changed_mm: float | None # distance between fresh and original board_xy, if both present


def replay_throw_with_engine(package: ThrowPackage | Path, engine_name: str) -> EngineResult:
    """Engine-aware replay -- runs ANY registered engine (docs/ENGINES.md's
    "Offline tooling" section) against a package's stored raw images +
    calibration, exactly the same images-in/calibration-in interface a
    live throw gives an engine. `engine_name` is REQUIRED, deliberately,
    since 2026-09-05 (see this module's own docstring) -- there is no
    "the" engine a replay should silently assume; say which one you mean
    every time (e.g. `"Zeus"` to match live production, `"Apollo"` to
    isolate one sub-engine's own behavior). Returns an `EngineResult`;
    convert to the legacy `ScoreResult` shape yourself if you need it

    **REAL COST TRAP, 2026-09-05 -- read before using this (or
    `replay_and_compare()` below) to BENCHMARK an engine, not just to
    replay one.** For any engine that accepts prior-dart-line context
    (e.g. Zeus/Apollo -- see `engine_accepts_prior_dart_line_px()`
    below), this function calls `find_prior_dart_line_px()` BEFORE
    `engine.score()` on every call -- a genuine, DISK-BASED prior-throw
    `detect_tip()` recompute this project's LIVE path never pays
    (`opendarts.live.capture_daemon.handle_ready_to_capture()` has had an
    in-memory fast path for this since the 2026-09-01 `cached_prior_
    frames`/`own_tip_line_px_out` work -- see that function's own
    docstring -- which this offline replay path has no access to and
    therefore always falls through past). Measured real cost (live
    corpus, n=150): ~0.005ms on a turn's first dart (nothing to look
    up), but a MEDIAN ~97ms on darts 2/3 of a turn. This already
    produced one real wrong conclusion during this same task's own
    verification work: a naive before/after wrapper around this
    function attributed ~95ms of phantom "Zeus overhead" to the engine
    change actually being measured, when it was entirely this lookup.
    **Any real latency/throughput measurement of an engine must call
    `engine.score()` (or the registry engine object) directly, with
    pre-loaded frames, never through this wrapper** -- see this
    project's own 2026-08-27 Zeus-latency measurement scripts
    (referenced from docs/DESIGN.md) for the
    established pattern.
    (`opendarts.engines.apollo.engine_result_to_score_result()` for
    Apollo's full-fidelity round trip, `opendarts.engines.base.
    engine_result_to_score_result()`'s honestly-lossy generic adapter for
    any other engine -- see `replay_and_compare()` below for exactly this
    pattern, or `opendarts.live.capture_daemon.handle_ready_to_capture()`'s
    own identical live-path branch).
    """
    if isinstance(package, Path) or isinstance(package, str):
        package = load_throw_package(package)
    engine = get_engine(engine_name)

    if engine_name == "Talos" and engine_accepts_prior_board_xy_mm(engine):
        # 2026-08-17 -- Talos prior-dart erase, replay side. Same
        # find_prior_board_xy_mm() lookup the live capture path uses, so
        # a package captured live and later replayed sees the same prior
        # board-XY (docs/DESIGN.md's "Replay is the source of truth"). Capability check
        # keeps a test-double registered as "Talos" working unmodified.
        prior_board_xy_mm = find_prior_board_xy_mm(
            package.package_dir.parent, package.visit_id, package.visit_index
        )
        return engine.score(
            package.bg_frames, package.dart_frames, package.calibrations,
            prior_board_xy_mm=prior_board_xy_mm or None,
        )

    if engine_accepts_prior_dart_line_px(engine):
        # 2026-08-16 -- prior-dart-in-visit contamination guard, replay
        # side. Per docs/DESIGN.md's "Replay is the source of truth": a live capture
        # and an offline replay of the SAME package must reach the SAME
        # decision, so this uses the exact same
        # find_prior_dart_line_px() lookup opendarts.live.capture_daemon.
        # handle_ready_to_capture() uses live -- see
        # opendarts.engines.apollo.prior_dart_context's module docstring
        # for why this is one shared function, not two independently-
        # maintained copies. `package.package_dir.parent` is this
        # package's own session directory (see ThrowPackage/
        # save_throw_package()'s own directory layout). The
        # `engine_accepts_prior_dart_line_px()` capability check (see
        # that function's own docstring) is what keeps a test double
        # registered under the "Apollo" name working unmodified.
        #
        # 2026-08-24 -- dropped the `engine_name == DEFAULT_PRIMARY_ENGINE`
        # name restriction (was: `engine_name == DEFAULT_PRIMARY_ENGINE and
        # engine_accepts_prior_dart_line_px(engine)`). Found investigating
        # the recorded S10 throw: Zeus's own score() now also
        # declares `prior_dart_line_px` (see opendarts.engines.zeus.engine's
        # module docstring for the full incident) and forwards it to its
        # Apollo sub-call, so replaying "Zeus" must reach this branch
        # too, or replay would silently under-report what the CURRENT
        # live pipeline actually does for Zeus-as-primary sessions --
        # exactly the "Replay is the source of truth" constraint this module exists to
        # uphold. The capability check alone is now the complete gate.
        prior_dart_line_px = find_prior_dart_line_px(
            package.package_dir.parent, package.visit_id, package.visit_index
        )
        return engine.score(
            package.bg_frames, package.dart_frames, package.calibrations,
            prior_dart_line_px=prior_dart_line_px,
        )

    return engine.score(package.bg_frames, package.dart_frames, package.calibrations)


def replay_and_compare(package_dir: Path, engine_name: str) -> ReplayComparison:
    """Replay a package and compare against its originally-stored result
    -- the actual drift-detection tool: run this across a whole batch of
    old packages after a pipeline change to see what moved.

    **Same real cost trap as `replay_throw_with_engine()` above, 2026-
    09-05 -- this function calls straight through to it, so it inherits
    the SAME disk-based prior-dart-line lookup on darts 2/3 of a turn
    (median ~97ms, live corpus n=150) that production's live path never
    pays. Never use this to measure engine latency/throughput** -- it
    is a correctness/drift tool, not a benchmarking one; see that
    function's own docstring for the full incident and the correct
    pattern (call `engine.score()` directly).

    `engine_name` is REQUIRED, same reasoning as `replay_throw_with_
    engine()` above (this module's own docstring) -- pass whichever
    engine actually produced (or should be compared against) the
    package's own stored result; there is no silent default any more.
    See `opendarts.capture.rescore_all` for the engine-aware batch
    entrypoint that drives this across a whole package root.

    Converts the engine's `EngineResult` back to the legacy `ScoreResult`
    shape this function's own callers expect using the SAME conversion
    `opendarts.live.capture_daemon.handle_ready_to_capture()` uses for its
    real live primary-engine write: Apollo's own full-fidelity
    round-trip converter when `engine_name == "Apollo"` (recovers every
    field `save_throw_package()`'s on-disk schema persists), the generic,
    honestly-lossy adapter otherwise (no other engine populates the
    Apollo-specific diagnostics shape the full-fidelity converter
    unpacks)."""
    import numpy as np

    package = load_throw_package(package_dir)
    engine_result = replay_throw_with_engine(package, engine_name)
    fresh = (
        apollo_engine_result_to_score_result(engine_result)
        if engine_name == DEFAULT_PRIMARY_ENGINE
        else generic_engine_result_to_score_result(engine_result)
    )

    sector_changed = None
    board_xy_changed_mm = None
    if package.original_result is not None:
        orig_sector = package.original_result.get("sector")
        sector_changed = fresh.sector != orig_sector
        orig_xy = package.original_result.get("board_xy_mm")
        if orig_xy is not None and fresh.board_xy_mm is not None:
            board_xy_changed_mm = float(
                np.linalg.norm(np.array(fresh.board_xy_mm) - np.array(orig_xy))
            )

    return ReplayComparison(
        package_dir=Path(package_dir),
        fresh_result=fresh,
        original_result=package.original_result,
        sector_changed=sector_changed,
        board_xy_changed_mm=board_xy_changed_mm,
    )
