"""opendarts/capture/rescore_store.py -- keep every package's score from the
CURRENT code next to the live one, with a history by code version.

WHY. `result.json` is what OpenDarts scored live, and never changes. Asking
"which darts are still wrong?" of it answers for whatever code ran that
night: on 2026-10-07 a review of 27 "Apollo lone misses" found 15 already
fixed by later code. So each package also gets `rescore.json`:

    {"schema": "rescore-v1",
     "history": [ {code_version, scored_at_utc, truth, ad, zeus, engines,
                   correct, ad_correct}, ... ]}

one entry per code version it has been rescored with. Its last entry is
"today's" score; the first version where a call turns right is the one
that fixed it. Every run also appends one line per code version to
`<report dir>/rescore_history.jsonl`: accuracy per engine, the oracle's
accuracy against the operator's corrections, and which darts turned
right or wrong since the previous version.

TRUTH is the operator's correction when there is one, else the oracle's
call when it matched and was not marked wrong (the same rule every
accuracy figure here uses). Calls are compared as `sector_ring_to_token`
tokens, so an inner and an outer single are the same call. A package with
no truth is still rescored; its correctness is just left as None.

BACKFILL. `--code-root <a checkout of an older commit>` scores with THAT
commit's engines (this file is run as a script; it imports `opendarts`
from the given root) and records that commit's version. The engine API it
needs, `replay_throw_with_engine`, exists from 2026-09-05 on.

    .venv/bin/python3 -m opendarts.capture.rescore_store [--package-root P] [--workers N] [--force]
    .venv/bin/python3 opendarts/capture/rescore_store.py --code-root /tmp/od-at-<sha>
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

SCHEMA = "rescore-v1"
FILENAME = "rescore.json"
HISTORY_FILENAME = "rescore_history.jsonl"
ENGINES = ("Apollo", "Talos", "Athena", "Ares", "Argus")
#: diagnostics worth keeping per engine (the rest stay in a fresh replay)
_DIAG_KEYS = ("n_cameras_used", "cameras_used", "max_ray_disagreement_mm",
              "outlier_camera", "confidence")


def _token(sector: Any, ring: Any, ok: Any = True) -> "str | None":
    if not ok or not ring:
        return None
    from opendarts.geometry.board import sector_ring_to_token
    try:
        return sector_ring_to_token(None if sector is None else str(sector), ring)
    except Exception:  # noqa: BLE001 -- an odd stored ring is "no call", not a crash
        return None


def truth_of(gt: "dict[str, Any] | None") -> "tuple[str | None, str | None]":
    """(truth token, source) for an ad_ground_truth.json dict: the operator's
    correction, else the oracle's matched call; (None, None) without one."""
    if not gt:
        return None, None
    if gt.get("operator_confirmed_ring"):
        return _token(gt.get("operator_confirmed_sector"), gt["operator_confirmed_ring"]), "operator"
    if gt.get("matched") and not gt.get("operator_marked_wrong"):
        return _token(gt.get("sector"), gt.get("ring")), "oracle"
    return None, None


def _engine_entry(sector, ring, ok, xy, diagnostics) -> dict[str, Any]:
    d = diagnostics or {}
    return {"token": _token(sector, ring, ok), "ok": bool(ok),
            "board_xy_mm": [round(float(v), 2) for v in xy] if xy is not None else None,
            **{k: d[k] for k in _DIAG_KEYS if k in d}}


def rescore_package(pkg: Path, code_version: "str | None",
                    code_committed_at: "str | None" = None,
                    code_depth: "int | None" = None) -> dict[str, Any]:
    """Score one package with the importable engines; the history entry."""
    from opendarts.capture.replay import replay_throw_with_engine

    gt = None
    gt_path = pkg / "ad_ground_truth.json"
    if gt_path.exists():
        try:
            gt = json.loads(gt_path.read_text())
        except (OSError, ValueError):
            gt = None
    truth, source = truth_of(gt)
    entry: dict[str, Any] = {
        "code_version": code_version,
        "code_committed_at": code_committed_at,
        "code_depth": code_depth,
        "scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "truth": {"token": truth, "source": source},
        "ad": {"token": _token(gt.get("sector"), gt.get("ring"), gt.get("matched")) if gt else None},
    }
    try:
        r = replay_throw_with_engine(pkg, "Zeus")
    except Exception as exc:  # noqa: BLE001 -- one bad package is recorded, not fatal
        entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return entry
    diag = r.diagnostics or {}
    entry["zeus"] = _engine_entry(r.sector, r.ring, r.ok, getattr(r, "board_xy_mm", None), None)
    entry["zeus"].update({k: diag[k] for k in ("winning_engine", "tie_break_applied") if k in diag})
    engines = {}
    for name, sub in (diag.get("sub_results") or {}).items():
        sub = sub if isinstance(sub, dict) else getattr(sub, "__dict__", {})
        engines[name] = _engine_entry(sub.get("sector"), sub.get("ring"), sub.get("ok"),
                                      sub.get("board_xy_mm"), sub.get("diagnostics"))
    entry["engines"] = engines
    if truth is not None:
        entry["correct"] = {"Zeus": entry["zeus"]["token"] == truth,
                            **{n: e["token"] == truth for n, e in engines.items()}}
    if source == "operator":
        entry["ad_correct"] = entry["ad"]["token"] == truth
    return entry


def _order(h: dict[str, Any]) -> tuple:
    """History is in COMMIT order -- a backfill scores old commits after new
    ones, and "which version fixed it" must follow the code, not the run.
    By the commit's depth (its count of ancestors) where known: commit times
    tie to the second when commits land together; depth never ties along
    one line of history."""
    depth = h.get("code_depth")
    return (0 if depth is None else 1, depth if depth is not None else 0,
            h.get("code_committed_at") or h.get("scored_at_utc") or "")


def load_history(pkg: Path) -> list[dict[str, Any]]:
    try:
        doc = json.loads((pkg / FILENAME).read_text())
    except (OSError, ValueError):
        return []
    return doc.get("history", []) if doc.get("schema") == SCHEMA else []


def save_entry(pkg: Path, entry: dict[str, Any]) -> None:
    """Add (or replace, for the same code version) this entry; atomic write."""
    history = [h for h in load_history(pkg) if h.get("code_version") != entry.get("code_version")]
    history.append(entry)
    history.sort(key=_order)
    tmp = pkg / (FILENAME + ".tmp")
    tmp.write_text(json.dumps({"schema": SCHEMA, "history": history}, indent=1))
    os.replace(tmp, pkg / FILENAME)


def first_right_version(pkg: Path, engine: str = "Zeus") -> "str | None":
    """The first code version in this package's history whose `engine` call
    was right, after one that was wrong -- the version that fixed it. None
    when it was never wrong, or is still wrong."""
    was_wrong = False
    for h in load_history(pkg):
        ok = (h.get("correct") or {}).get(engine)
        if ok is False:
            was_wrong = True
        elif ok and was_wrong:
            return h.get("code_version")
    return None


def _code_version_of(root: Path) -> "tuple[str | None, str | None, int | None]":
    """(commit sha, commit time ISO, depth) of the checkout at `root`."""
    try:
        out = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%H %cI"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        depth = subprocess.run(["git", "-C", str(root), "rev-list", "--count", "HEAD"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        if len(out) != 2:
            return None, None, None
        return out[0], out[1], int(depth) if depth.isdigit() else None
    except Exception:  # noqa: BLE001
        return None, None, None


def _discover(root: Path) -> list[Path]:
    return sorted(p.parent for p in root.glob("*/*/meta.json"))


_CODE_VERSION: "tuple[str | None, str | None, int | None]" = (None, None, None)


def _work(pkg_str: str) -> "tuple[str, dict[str, Any]]":
    pkg = Path(pkg_str)
    entry = rescore_package(pkg, *_CODE_VERSION)
    save_entry(pkg, entry)
    return pkg_str, entry


def _init(code_version: "tuple[str | None, str | None, int | None]") -> None:
    global _CODE_VERSION
    _CODE_VERSION = code_version
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def summarize(pkgs: list[Path], code_version: "str | None") -> dict[str, Any]:
    """Accuracy per engine at `code_version`, the oracle against the
    operator, and which darts changed since each package's previous version."""
    n = {k: 0 for k in ("Zeus",) + ENGINES}
    right = dict(n)
    ad_n = ad_right = 0
    fixed: list[str] = []
    broke: list[str] = []
    for pkg in pkgs:
        hist = load_history(pkg)
        cur = next((h for h in hist if h.get("code_version") == code_version), None)
        if cur is None or "correct" not in cur:
            continue
        for k, ok in cur["correct"].items():
            if k in n:
                n[k] += 1
                right[k] += bool(ok)
        if "ad_correct" in cur:
            ad_n += 1
            ad_right += bool(cur["ad_correct"])
        earlier = [h for h in hist if h is not cur and "correct" in h and _order(h) < _order(cur)]
        if earlier:
            was, now = earlier[-1]["correct"].get("Zeus"), cur["correct"].get("Zeus")
            if was is False and now:
                fixed.append(pkg.name)
            elif was and now is False:
                broke.append(pkg.name)
    return {"code_version": code_version,
            "at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "darts_with_truth": n["Zeus"],
            "right": right, "accuracy": {k: round(right[k] / n[k], 5) if n[k] else None for k in n},
            "oracle_vs_operator": {"reviewed": ad_n, "oracle_right": ad_right},
            "zeus_fixed_since_previous": fixed, "zeus_broken_since_previous": broke}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package-root", type=Path, default=None)
    ap.add_argument("--report-dir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="rescore even packages already at this version")
    ap.add_argument("--code-root", type=Path, default=None,
                    help="score with the engines of this checkout (an older commit) instead")
    args = ap.parse_args(argv)

    here = Path(__file__).resolve().parents[2]
    code_root = (args.code_root or here).resolve()
    if args.code_root is not None:
        # An older checkout's engines; the data stays this checkout's. Older
        # commits may not have opendarts.paths, so it is not imported then.
        sys.path.insert(0, str(code_root))
        data_dir = here / "data"
    else:
        from opendarts.paths import DATA_DIR
        data_dir = DATA_DIR
    root = args.package_root or data_dir / "packages"
    report_dir = args.report_dir or data_dir / "rescore_reports"
    code_version, committed_at, depth = _code_version_of(code_root)
    pkgs = _discover(root)
    todo = pkgs if args.force else [p for p in pkgs
                                    if not any(h.get("code_version") == code_version
                                               for h in load_history(p))]
    print(f"code {code_version and code_version[:7]} from {code_root}: "
          f"{len(todo)} of {len(pkgs)} packages to rescore", flush=True)
    t0 = time.monotonic()
    with Pool(args.workers, initializer=_init, initargs=((code_version, committed_at, depth),)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_work, [str(p) for p in todo], chunksize=4), 1):
            if i % 500 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    s = summarize(pkgs, code_version)
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / HISTORY_FILENAME, "a") as fh:
        fh.write(json.dumps(s) + "\n")
    acc = "  ".join(f"{k} {v:.2%}" for k, v in s["accuracy"].items() if v is not None)
    print(f"done in {time.monotonic() - t0:.0f}s -- {s['darts_with_truth']} darts with truth: {acc}")
    o = s["oracle_vs_operator"]
    print(f"oracle right on {o['oracle_right']} of the {o['reviewed']} darts the operator corrected; "
          f"Zeus fixed {len(s['zeus_fixed_since_previous'])}, broken "
          f"{len(s['zeus_broken_since_previous'])} since the previous version")
    return 0


if __name__ == "__main__":
    sys.exit(main())
