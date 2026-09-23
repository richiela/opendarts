"""opendarts/capture/rescore_all.py -- batch re-scoring across EVERY saved
throw package on disk. See docs/DESIGN.md's "Replay is the source of truth":
`opendarts/capture/replay.py` already implements the actual replay/
comparison mechanism (`replay_throw()` / `replay_and_compare()`); this
module's only job is to run that at BATCH scale across a whole package
root and produce a genuinely readable summary of what changed. It does
not reimplement tip detection, triangulation, or scoring.

Why this exists: "be able to score that data based on the data
you got" -- decouple LIVE scoring correctness from capture reliability (a
separate, concurrent effort owns capture reliability itself). Whatever
got captured -- even a package whose LIVE score came back `ok=False`, or
was scored while the pipeline had a bug that's since been fixed -- must
be re-scorable offline, in batch, against the CURRENT pipeline code,
without re-throwing those darts.

Engine-aware (docs/ENGINES.md's "Offline tooling" section, added
2026-08-12): `rescore_all(engine=...)` / `--engine` default to `Apollo`
(today's only real live engine, and this module's own pre-existing
behavior, 100% unchanged when left at the default) but accept ANY
registered engine name -- this is what makes testing a brand new engine
against the full real corpus (250+ real throws) possible before ever
enabling it live.

Usage:
    .venv/bin/python3 -m opendarts.capture.rescore_all [--package-root PATH] [--out PATH] [--engine NAME]

Package discovery: reuses `opendarts.live.server.discover_packages()`
directly rather than re-walking `<package_root>/<session>/<throw_id>/`
by hand -- that function already implements the exact directory-walking
pattern (glob `*/*/meta.json`) and the "tolerant of a package a
concurrent live capture process is actively mid-write on" behavior this
module also needs (a corrupt/partial `meta.json` is silently skipped by
discover_packages() itself, same as the live dashboard would see it).
This module adds its own second layer of tolerance on top for failures
`discover_packages()` can't see (it only reads `meta.json`/`result.json`,
never the PNGs/`calibration.json`): a package whose PNGs or calibration
are missing/corrupt fails at `load_throw_package()` time instead, and is
caught and reported per-throw here, not allowed to abort the whole batch.

`replay_and_compare()` vs `replay_throw()`: this module always calls
`replay_and_compare()`, even for packages with no `original_result`.
`replay_and_compare()` already degrades cleanly in that case (built-in
`sector_changed=None`/`board_xy_changed_mm=None`, see replay.py) -- there
is no reason to hand-roll a separate `replay_throw()`-only code path for
"no original result" when the one function already does the right thing
for both cases. This module still reports every packages fresh result
regardless of whether a comparison was possible.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from opendarts.capture.replay import ReplayComparison, replay_and_compare, replay_throw_with_engine
from opendarts.capture.throw_package import load_throw_package
from opendarts.engines.base import EngineResult
from opendarts.engines.registry import DEFAULT_PRIMARY_ENGINE, engine_names, is_registered
from opendarts.live.capture_daemon import DEFAULT_PACKAGE_ROOT
from opendarts.live.server import discover_packages

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from opendarts.paths import DATA_DIR
# Same tree as DEFAULT_PACKAGE_ROOT (data/packages) and
# opendarts/live/logging_setup.py's DEFAULT_LOG_DIR (data/logs) -- already
# gitignored via the repo's existing /data/ entry.
DEFAULT_REPORT_DIR = DATA_DIR / "rescore_reports"


@dataclass
class RescoreOutcome:
    """One package's replay outcome. `status` is always one of "scored",
    "load_failed", "replay_failed" -- exactly one of these three, never
    left ambiguous, so a caller can filter/count reliably."""

    package_dir: Path
    session: str | None
    throw_id: str
    status: str # "scored" | "load_failed" | "replay_failed"
    error: str | None = None

    # Real capture timestamp (meta.json's captured_at_utc), not the
    # session/throw_id string -- 2026-08-12 real-corpus investigation
    # (the "got worse and worse" live-testing report): a chronological
    # trend across a whole day's packages can
    # only be trusted against the ACTUAL capture time, not directory
    # names or discover_packages()'s glob order (which is filesystem
    # order, not chronological -- confirmed not sorted before this).
    # None only for a package whose meta.json itself couldn't be read at
    # all (shouldn't happen -- discover_packages() already requires a
    # parseable meta.json to list a package in the first place).
    captured_at_utc: str | None = None

    had_original_result: bool = False
    original_ok: bool | None = None
    original_sector: str | None = None
    original_ring: str | None = None
    original_reason: str | None = None

    fresh_ok: bool | None = None
    fresh_sector: str | None = None
    fresh_ring: str | None = None
    fresh_reason: str | None = None
    # Added alongside the above chronological-sort fix: the task this
    # tool exists for ("be able to score that data based on the data you
    # got") needs these two numbers per throw to actually diagnose WHY a
    # throw failed/degraded, not just whether it did -- both were being
    # computed by score_dart() already (ScoreResult.n_cameras_used /
    # .max_ray_disagreement_mm) but silently dropped on the floor here.
    fresh_n_cameras_used: int | None = None
    fresh_max_ray_disagreement_mm: float | None = None

    # None whenever had_original_result is False -- "no comparison was
    # possible", not "nothing changed".
    ok_changed: bool | None = None
    sector_changed: bool | None = None
    ring_changed: bool | None = None
    board_xy_changed_mm: float | None = None

    @property
    def newly_ok(self) -> bool:
        """Originally ok=False, now ok=True -- exactly the case the project's
        2026-08-12 context called out: a package rejected live (e.g. for
        camera-ray disagreement, or a since-fixed bug) that the CURRENT
        pipeline can now score."""
        return self.had_original_result and self.original_ok is False and self.fresh_ok is True

    @property
    def regressed(self) -> bool:
        """Originally ok=True, now ok=False -- a real regression the
        operator needs to see, not just the "improved" direction."""
        return self.had_original_result and self.original_ok is True and self.fresh_ok is False

    @property
    def anything_changed(self) -> bool:
        return bool(self.had_original_result and (self.ok_changed or self.sector_changed or self.ring_changed))

    def to_json_dict(self) -> dict[str, Any]:
        d = {
            "package_dir": str(self.package_dir),
            "session": self.session,
            "throw_id": self.throw_id,
            "captured_at_utc": self.captured_at_utc,
            "status": self.status,
            "error": self.error,
            "had_original_result": self.had_original_result,
            "original_ok": self.original_ok,
            "original_sector": self.original_sector,
            "original_ring": self.original_ring,
            "original_reason": self.original_reason,
            "fresh_ok": self.fresh_ok,
            "fresh_sector": self.fresh_sector,
            "fresh_ring": self.fresh_ring,
            "fresh_reason": self.fresh_reason,
            "fresh_n_cameras_used": self.fresh_n_cameras_used,
            "fresh_max_ray_disagreement_mm": self.fresh_max_ray_disagreement_mm,
            "ok_changed": self.ok_changed,
            "sector_changed": self.sector_changed,
            "ring_changed": self.ring_changed,
            "board_xy_changed_mm": self.board_xy_changed_mm,
        }
        if self.status == "scored":
            d["newly_ok"] = self.newly_ok
            d["regressed"] = self.regressed
            d["anything_changed"] = self.anything_changed
        return d


@dataclass
class RescoreSummary:
    package_root: Path
    generated_at_utc: str
    outcomes: list[RescoreOutcome] = field(default_factory=list)

    # -- derived counts, computed on demand rather than tracked
    # incrementally, so they can never drift out of sync with
    # `outcomes` (the actual per-throw records) --------------------

    @property
    def total_found(self) -> int:
        return len(self.outcomes)

    def _scored(self) -> list[RescoreOutcome]:
        return [o for o in self.outcomes if o.status == "scored"]

    def _compared(self) -> list[RescoreOutcome]:
        return [o for o in self._scored() if o.had_original_result]

    @property
    def load_failed(self) -> list[RescoreOutcome]:
        return [o for o in self.outcomes if o.status == "load_failed"]

    @property
    def replay_failed(self) -> list[RescoreOutcome]:
        return [o for o in self.outcomes if o.status == "replay_failed"]

    @property
    def no_original_result(self) -> list[RescoreOutcome]:
        return [o for o in self._scored() if not o.had_original_result]

    @property
    def newly_ok(self) -> list[RescoreOutcome]:
        return [o for o in self._compared() if o.newly_ok]

    @property
    def regressed(self) -> list[RescoreOutcome]:
        return [o for o in self._compared() if o.regressed]

    @property
    def sector_changed(self) -> list[RescoreOutcome]:
        return [o for o in self._compared() if o.sector_changed]

    @property
    def ring_changed(self) -> list[RescoreOutcome]:
        return [o for o in self._compared() if o.ring_changed]

    @property
    def changed(self) -> list[RescoreOutcome]:
        return [o for o in self._compared() if o.anything_changed]

    @property
    def unchanged(self) -> list[RescoreOutcome]:
        return [o for o in self._compared() if not o.anything_changed]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "package_root": str(self.package_root),
            "generated_at_utc": self.generated_at_utc,
            "totals": {
                "total_found": self.total_found,
                "scored": len(self._scored()),
                "load_failed": len(self.load_failed),
                "replay_failed": len(self.replay_failed),
                "compared_against_original": len(self._compared()),
                "no_original_result": len(self.no_original_result),
                "changed": len(self.changed),
                "unchanged": len(self.unchanged),
                "newly_ok": len(self.newly_ok),
                "regressed": len(self.regressed),
                "sector_changed": len(self.sector_changed),
                "ring_changed": len(self.ring_changed),
            },
            "throws": [o.to_json_dict() for o in self.outcomes],
        }


def _outcome_from_comparison(
    pkg_dir: Path, session: str | None, throw_id: str, captured_at_utc: str | None, cmp: ReplayComparison
) -> RescoreOutcome:
    orig = cmp.original_result
    had_original = orig is not None

    original_ok = orig.get("ok") if orig is not None else None
    original_sector = orig.get("sector") if orig is not None else None
    original_ring = orig.get("ring") if orig is not None else None
    original_reason = orig.get("reason") if orig is not None else None

    fresh = cmp.fresh_result

    ok_changed = (fresh.ok != original_ok) if had_original else None
    ring_changed = (fresh.ring != original_ring) if had_original else None
    # cmp.sector_changed is already computed by replay_and_compare() --
    # reuse it rather than recomputing, so this module can never disagree
    # with the function that actually owns that comparison.
    sector_changed = cmp.sector_changed

    return RescoreOutcome(
        package_dir=pkg_dir,
        session=session,
        throw_id=throw_id,
        captured_at_utc=captured_at_utc,
        status="scored",
        had_original_result=had_original,
        original_ok=original_ok,
        original_sector=original_sector,
        original_ring=original_ring,
        original_reason=original_reason,
        fresh_ok=fresh.ok,
        fresh_sector=fresh.sector,
        fresh_ring=fresh.ring,
        fresh_reason=fresh.reason,
        fresh_n_cameras_used=fresh.n_cameras_used,
        fresh_max_ray_disagreement_mm=fresh.max_ray_disagreement_mm,
        ok_changed=ok_changed,
        sector_changed=sector_changed,
        ring_changed=ring_changed,
        board_xy_changed_mm=cmp.board_xy_changed_mm,
    )


def _outcome_from_engine_result(
    pkg_dir: Path,
    session: str | None,
    throw_id: str,
    captured_at_utc: str | None,
    original_result: dict | None,
    fresh: EngineResult,
) -> RescoreOutcome:
    """The engine-aware counterpart to `_outcome_from_comparison()` above
    -- same derived-fields shape, but reading a generic `EngineResult`
    (any registered engine, docs/ENGINES.md) instead of a
    Apollo-specific `ScoreResult`/`ReplayComparison`. `fresh_n_cameras_used`/
    `fresh_max_ray_disagreement_mm` are recovered from `diagnostics` only
    if the engine happens to have put one there under that exact key
    (Apollo always does; a different engine's diagnostics may simply
    not have an equivalent concept -- left honestly `None`, not invented).
    """
    import numpy as np

    had_original = original_result is not None
    original_ok = original_result.get("ok") if had_original else None
    original_sector = original_result.get("sector") if had_original else None
    original_ring = original_result.get("ring") if had_original else None
    original_reason = original_result.get("reason") if had_original else None

    ok_changed = (fresh.ok != original_ok) if had_original else None
    sector_changed = (fresh.sector != original_sector) if had_original else None
    ring_changed = (fresh.ring != original_ring) if had_original else None
    board_xy_changed_mm = None
    if had_original:
        orig_xy = original_result.get("board_xy_mm")
        if orig_xy is not None and fresh.board_xy_mm is not None:
            board_xy_changed_mm = float(
                np.linalg.norm(np.array(fresh.board_xy_mm) - np.array(orig_xy))
            )

    return RescoreOutcome(
        package_dir=pkg_dir,
        session=session,
        throw_id=throw_id,
        captured_at_utc=captured_at_utc,
        status="scored",
        had_original_result=had_original,
        original_ok=original_ok,
        original_sector=original_sector,
        original_ring=original_ring,
        original_reason=original_reason,
        fresh_ok=fresh.ok,
        fresh_sector=fresh.sector,
        fresh_ring=fresh.ring,
        fresh_reason=fresh.reason,
        fresh_n_cameras_used=fresh.diagnostics.get("n_cameras_used"),
        fresh_max_ray_disagreement_mm=fresh.diagnostics.get("max_ray_disagreement_mm"),
        ok_changed=ok_changed,
        sector_changed=sector_changed,
        ring_changed=ring_changed,
        board_xy_changed_mm=board_xy_changed_mm,
    )


def rescore_all(
    package_root: Path = DEFAULT_PACKAGE_ROOT, engine: str = DEFAULT_PRIMARY_ENGINE
) -> RescoreSummary:
    """Walk every throw package under `package_root` (via
    opendarts.live.server.discover_packages(), see module docstring) and
    replay+compare each one against the CURRENT pipeline code. Never
    raises on a single bad package -- a load failure or a replay-time
    exception is caught and recorded as that package's own outcome, and
    the batch keeps going.

    Outcomes are returned in TRUE chronological order (each package's
    real meta.json `captured_at_utc`, not discover_packages()'s glob
    order -- which is filesystem/directory order and NOT chronological,
    confirmed by direct inspection before this was fixed). Added
    2026-08-12 during the real 54-package "got worse and worse"
    live-testing investigation: a trend
    analysis across a whole day's real packages is only trustworthy
    against the ACTUAL capture time. A package with no parseable
    `captured_at_utc` (shouldn't happen -- discover_packages() already
    requires a parseable meta.json to list it at all) sorts last rather
    than crashing the sort or silently defaulting to "first".

    engine: which registered engine (opendarts.engines.registry) to replay
    every package through -- default `Apollo` reuses
    `opendarts.capture.replay.replay_and_compare()` exactly as before this
    parameter existed (100% unchanged behavior at the default). Any
    OTHER registered name goes through
    `opendarts.capture.replay.replay_throw_with_engine()` instead, via
    `_outcome_from_engine_result()` above -- see docs/ENGINES.md's
    "Offline tooling" section. Raises `ValueError` immediately (before
    touching any package) on an unregistered engine name -- a typo here
    should never silently walk the entire corpus before failing.
    """
    if not is_registered(engine):
        raise ValueError(f"unknown engine {engine!r} -- registered engines: {engine_names()}")

    package_root = Path(package_root)
    generated_at_utc = datetime.now(timezone.utc).isoformat()
    summary = RescoreSummary(package_root=package_root, generated_at_utc=generated_at_utc)

    entries = discover_packages(package_root)
    entries.sort(key=lambda e: e.get("captured_at_utc") or "")
    log.info(
        "rescore_all: found %d package(s) under %s (engine=%s)",
        len(entries), package_root, engine,
    )

    for entry in entries:
        pkg_dir = Path(entry["path"])
        session = entry.get("session")
        throw_id = entry.get("throw_id") or pkg_dir.name
        captured_at_utc = entry.get("captured_at_utc")

        try:
            if engine == DEFAULT_PRIMARY_ENGINE:
                # replay_and_compare()'s own `engine_name` is required as
                # of 2026-09-05 (opendarts/capture/replay.py's own docstring
                # -- no silent per-engine default any more); `engine` is
                # already resolved from this module's own explicit
                # `--engine`/argparse default, so this is a required
                # threading-through, not a new default.
                comparison = replay_and_compare(pkg_dir, engine)
            else:
                package = load_throw_package(pkg_dir)
                fresh = replay_throw_with_engine(package, engine)
                comparison = None
        except Exception as exc: # noqa: BLE001 -- deliberately broad, see module docstring
            # Distinguish "never even loaded" from "loaded fine but
            # replay itself blew up" for a clearer operator-facing
            # message. This second load attempt only runs on the
            # (expected-rare) failure path, so the extra I/O cost is
            # negligible against a whole-batch run.
            try:
                load_throw_package(pkg_dir)
                status = "replay_failed"
            except Exception: # noqa: BLE001
                status = "load_failed"
            log.warning("rescore_all: %s failed (%s): %s", pkg_dir, status, exc)
            summary.outcomes.append(
                RescoreOutcome(
                    package_dir=pkg_dir,
                    session=session,
                    throw_id=throw_id,
                    captured_at_utc=captured_at_utc,
                    status=status,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        if comparison is not None:
            summary.outcomes.append(
                _outcome_from_comparison(pkg_dir, session, throw_id, captured_at_utc, comparison)
            )
        else:
            summary.outcomes.append(
                _outcome_from_engine_result(
                    pkg_dir, session, throw_id, captured_at_utc,
                    package.original_result, fresh,
                )
            )

    return summary


# --------------------------------------------------------------------------
# Human-readable stdout report.
# --------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def _print_table(headers: list[str], rows: list[list[Any]]) -> None:
    if not rows:
        print(" (none)")
        return
    str_rows = [[_fmt(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: list[str]) -> str:
        return " " + " ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(_line(headers))
    print(_line(["-" * w for w in widths]))
    for row in str_rows:
        print(_line(row))


def print_summary(summary: RescoreSummary) -> None:
    t = summary.to_json_dict()["totals"]

    print("=" * 78)
    print(f"RESCORE ALL -- {summary.package_root}")
    print(f"generated {summary.generated_at_utc}")
    print("=" * 78)
    print(f"packages found: {t['total_found']}")
    print(f" scored (replayed OK): {t['scored']}")
    print(f" load failed: {t['load_failed']}")
    print(f" replay failed: {t['replay_failed']}")
    print(f" no original_result to diff: {t['no_original_result']}")
    print(f" compared against original: {t['compared_against_original']}")
    print(f" unchanged: {t['unchanged']}")
    print(f" changed (ok/sector/ring): {t['changed']}")
    print(f" newly ok=True: {t['newly_ok']} (rejected live, now scores)")
    print(f" regressed to ok=False: {t['regressed']} (scored live, now rejects)")
    print(f" sector changed: {t['sector_changed']}")
    print(f" ring changed: {t['ring_changed']}")

    compared = summary._compared() # noqa: SLF001 -- same module, intentional reuse
    xy_deltas = [o.board_xy_changed_mm for o in compared if o.board_xy_changed_mm is not None]
    if xy_deltas:
        print(
            f" board_xy drift across all compared throws: "
            f"mean {sum(xy_deltas) / len(xy_deltas):.2f}mm, max {max(xy_deltas):.2f}mm"
        )

    print()
    print(f"CHANGED -- before/after, per throw, chronological ({len(summary.changed)}):")
    _print_table(
        ["captured_at_utc", "session", "throw_id", "orig_ok", "fresh_ok", "orig_sector", "fresh_sector",
         "orig_ring", "fresh_ring", "moved_mm"],
        [
            [o.captured_at_utc, o.session, o.throw_id, o.original_ok, o.fresh_ok, o.original_sector,
             o.fresh_sector, o.original_ring, o.fresh_ring, o.board_xy_changed_mm]
            for o in summary.changed
        ],
    )

    print()
    print(f"ALL THROWS -- chronological, fresh result only ({len(summary._scored())}):") # noqa: SLF001
    _print_table(
        ["captured_at_utc", "session", "throw_id", "fresh_ok", "fresh_sector", "fresh_ring",
         "n_cams", "max_ray_disagree_mm"],
        [
            [o.captured_at_utc, o.session, o.throw_id, o.fresh_ok, o.fresh_sector, o.fresh_ring,
             o.fresh_n_cameras_used, o.fresh_max_ray_disagreement_mm]
            for o in summary._scored() # noqa: SLF001 -- same module, intentional reuse
        ],
    )

    if summary.no_original_result:
        print()
        print(f"NO ORIGINAL RESULT -- fresh result only, nothing to compare ({len(summary.no_original_result)}):")
        _print_table(
            ["session", "throw_id", "fresh_ok", "fresh_sector", "fresh_ring", "fresh_reason"],
            [
                [o.session, o.throw_id, o.fresh_ok, o.fresh_sector, o.fresh_ring, o.fresh_reason]
                for o in summary.no_original_result
            ],
        )

    failures = summary.load_failed + summary.replay_failed
    if failures:
        print()
        print(f"FAILURES -- kept going, did not abort the batch ({len(failures)}):")
        _print_table(
            ["session", "throw_id", "status", "error"],
            [[o.session, o.throw_id, o.status, o.error] for o in failures],
        )

    print()
    print(f"unchanged (compared, nothing different): {len(summary.unchanged)}")


def write_json_summary(summary: RescoreSummary, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary.to_json_dict(), indent=2))


def _default_out_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_REPORT_DIR / f"rescore_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-replay every saved throw package under a package root against "
            "the CURRENT pipeline code, and report what changed vs the originally "
            "-stored live result (sector/ring/ok flips, board_xy drift). See "
            "docs/DESIGN.md's 'Replay is the source of truth'."
        )
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=DEFAULT_PACKAGE_ROOT,
        help="Directory saved throw packages live under (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Where to write the machine-readable JSON summary "
            f"(default: {DEFAULT_REPORT_DIR}/rescore_<UTC timestamp>.json)"
        ),
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=DEFAULT_PRIMARY_ENGINE,
        help=(
            "Which registered engine (docs/ENGINES.md) to replay every package "
            f"through -- one of {engine_names()} (default: %(default)s, today's "
            "real live pipeline, unchanged behavior)"
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    summary = rescore_all(args.package_root, engine=args.engine)
    print_summary(summary)

    out_path = args.out if args.out is not None else _default_out_path()
    write_json_summary(summary, out_path)
    print(f"\nmachine-readable summary written to {out_path}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
