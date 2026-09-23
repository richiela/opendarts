"""Shared pytest configuration -- the fast/slow split.

Why this exists (2026-08-16): the full pytest suite grew to 1300+ tests
/ ~10 minutes wall-clock, which made it too expensive to run after every
small change -- so it stopped being run after every small change, which
defeats the point of a smoke suite.  Profiling (`pytest --durations=0`)
showed the time is dominated by a small number of tests doing real
work: full data/archive/clean/ corpus replays (420 packages x 3 cameras
of real image I/O + detection per test), real subprocess lifecycles,
and multi-frame calibration solves.  The bulk of the test COUNT is fast
unit tests contributing very little wall-clock.

The fix is a tiering, not a deletion -- no coverage was removed:

* ``pytest tests/``          -> fast smoke tests only (the default;
                                seconds, run after every change).
* ``pytest tests/ --slow``   -> the full thing, slow tests included
                                (run before merging to main).
* ``pytest tests/ -m slow --slow`` -> only the slow tier, if needed.

Tests are opted INTO the slow tier explicitly via
``@pytest.mark.slow`` (registered in pytest.ini).  When ``--slow`` is
absent, slow-marked tests are reported as skipped (visibly, with a
reason) rather than silently deselected, so a default run always shows
that a slow tier exists.

The second thing this file does is keep a run OUT OF THE CHECKOUT --
every default config/log/package/capture/scratch path `opendarts` would
otherwise write to is redirected into throwaway directories. See the
long comment above `_PATH_MODULES` for what was measured and why it
takes two layers.
"""

from __future__ import annotations

import importlib
import inspect
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# A test run writes NOTHING into the checkout (2026-09-17)
# ---------------------------------------------------------------------------
#
# Measured, not assumed: a full `pytest tests/` in a pristine `git archive`
# export used to leave the export dirty -- data/config.json (the capability
# probe rewriting its `capabilities` section), an empty data/packages/, a
# PNG under tmp/live_server_scratch/, and ~18 tmp/test_* scratch trees.
# On a RIG that is not cosmetic: data/config.json is the operator's real
# config, data/logs/ the real logs, and data/packages/ the replay corpus a
# rescore is graded against -- a stray test package there would corrupt a
# rescore result.
#
# Two things had to happen, and both live here rather than in each test:
#
# 1. THE SESSION SANDBOX (below). Every REPO_ROOT-derived default path in
#    `opendarts` is repointed into one throwaway directory, at conftest
#    IMPORT time -- which is the only moment early enough: pytest imports
#    this file before any test module, and the config.json leak above
#    happened at test-module import time (tests/test_calibration_package.py
#    resolves IN_GIT_CHECKOUT at module level, which probes capabilities,
#    which persists a record). A fixture can never be early enough for that.
#
# 2. THE PER-TEST REDIRECT (`_isolate_repo_paths` below) moves the same set
#    again, to that test's own tmp_path, so tests cannot see each other's
#    leftovers either.
#
# Both go through `_point_repo_paths_at()`, which handles the subtlety that
# makes a plain `monkeypatch.setattr(mod, "DEFAULT_X", ...)` insufficient
# here: several of these constants are also bound as DEFAULT ARGUMENT
# VALUES (`def create_app(package_root=DEFAULT_PACKAGE_ROOT,
# scratch_dir=DEFAULT_SCRATCH_DIR, ...)`), evaluated once at import, so the
# module attribute and the function default are two independent copies --
# tests/test_live_server.py's `no_config_writes` fixture documents having
# been bitten by exactly this. Both copies are moved, together, so they
# never disagree.
#
# Read paths are deliberately untouched: only defaults under <repo>/data
# or <repo>/tmp are moved, and each keeps its path SHAPE relative to the
# repo root (data/config.json stays `<root>/data/config.json`), because a
# few tests assert on that shape.

#: Modules that own a REPO_ROOT-derived default. Imported eagerly so the
#: scan below sees them; importing `server` and `run_product` pulls in the
#: rest of the live stack, which every real test imports anyway (~0.3s).
_PATH_MODULES = (
    "opendarts.live.logging_setup",
    "opendarts.live.config",
    "opendarts.live.update_policy",
    "opendarts.live.capabilities",
    "opendarts.capture.calibration_package",
    "opendarts.capture.throw_capture",
    "opendarts.capture.rescore_all",
    "opendarts.live.capture_daemon",
    "opendarts.live.server",
    "opendarts.live.run_product",
)

_attr_targets: list[tuple[object, str, str]] = []
_pos_default_targets: list[tuple[object, int, str]] = []
_kw_default_targets: list[tuple[object, str, str]] = []
#: The originals, keyed "<module>.<attr>" -- see original_path().
_original_paths: dict[str, Path] = {}


def _writable_roots() -> "list[tuple[Path, str]]":
    """The trees a run is allowed to create, and the name each keeps when
    it is moved into a sandbox.

    `<repo>/data` and `<repo>/tmp` are the usual pair. `OPENDARTS_DATA_DIR`
    moves the first one anywhere (opendarts/paths.py), and when it does,
    the defaults built from it no longer live under the repo at all -- so
    matching on the repo alone would silently collect nothing, leave every
    default pointing at the operator's real data directory, and let a test
    run write into it. Ask the product where its data actually is.
    """
    from opendarts.paths import DATA_DIR, SCRATCH_ROOT

    roots = [(DATA_DIR, "data"), (SCRATCH_ROOT, "tmp"),
             (REPO_ROOT / "data", "data"), (REPO_ROOT / "tmp", "tmp")]
    seen: list[tuple[Path, str]] = []
    for root, name in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if (resolved, name) not in seen:
            seen.append((resolved, name))
    return seen


def _relocatable(value: object) -> str | None:
    """`value`'s path under a sandbox when it is a writable default.

    A Path pointing at source (docs/, run.sh) is left alone. The shape is
    preserved: `<data>/config.json` becomes `<sandbox>/data/config.json`,
    wherever `<data>` really is.
    """
    if not isinstance(value, Path):
        return None
    try:
        resolved = value.resolve()
    except OSError:
        return None
    for root, name in _writable_roots():
        try:
            rel = resolved.relative_to(root)
        except ValueError:
            continue
        return str(Path(name) / rel) if rel.parts else name
    return None


def _module_functions(mod):
    """Functions defined by `mod`, including methods of its own classes."""
    for value in list(vars(mod).values()):
        if inspect.isfunction(value) and getattr(value, "__module__", None) == mod.__name__:
            yield value
        elif inspect.isclass(value) and getattr(value, "__module__", None) == mod.__name__:
            for member in list(vars(value).values()):
                if inspect.isfunction(member):
                    yield member


def _collect_repo_path_targets() -> None:
    for name in _PATH_MODULES:
        importlib.import_module(name)
    for name, mod in sorted(sys.modules.items()):
        if not name.startswith("opendarts.") or mod is None:
            continue
        for attr, value in sorted(vars(mod).items()):
            rel = _relocatable(value)
            if rel is not None:
                _attr_targets.append((mod, attr, rel))
                _original_paths[f"{name}.{attr}"] = value
        for fn in _module_functions(mod):
            for i, default in enumerate(fn.__defaults__ or ()):
                rel = _relocatable(default)
                if rel is not None:
                    _pos_default_targets.append((fn, i, rel))
            for key, default in (fn.__kwdefaults__ or {}).items():
                rel = _relocatable(default)
                if rel is not None:
                    _kw_default_targets.append((fn, key, rel))


def _point_repo_paths_at(root: Path) -> None:
    """Move every collected default under `root`, shape preserved."""
    for mod, attr, rel in _attr_targets:
        setattr(mod, attr, root / rel)
    for fn, i, rel in _pos_default_targets:
        defaults = list(fn.__defaults__ or ())
        defaults[i] = root / rel
        fn.__defaults__ = tuple(defaults)
    for fn, key, rel in _kw_default_targets:
        fn.__kwdefaults__[key] = root / rel


def original_path(dotted: str) -> Path:
    """The un-redirected value of one default, e.g.
    ``original_path("opendarts.live.capture_daemon.DEFAULT_PACKAGE_ROOT")``.

    For the handful of tests whose SUBJECT is the constant itself -- "the
    shipped default must live inside the repo, not under $HOME" -- and
    which therefore need the real thing, not the sandboxed stand-in.
    """
    return _original_paths[dotted]


# The sandbox itself. One per pytest invocation: xdist workers inherit the
# environment variable from the controller (same mechanism the log dir has
# always used) so every process in a run agrees on it, and only the process
# that created it removes it.
_TEST_SANDBOX = os.environ.get("OPENDARTS_TEST_SANDBOX")
_OWNS_SANDBOX = _TEST_SANDBOX is None
if _TEST_SANDBOX is None:
    _TEST_SANDBOX = tempfile.mkdtemp(prefix="opendarts-test-sandbox-")
    os.environ["OPENDARTS_TEST_SANDBOX"] = _TEST_SANDBOX
SANDBOX_ROOT = Path(_TEST_SANDBOX)

# Keep test runs out of the real data/logs. Set here, at conftest import,
# because opendarts.live.logging_setup reads it once when first imported,
# and before any test module imports it. Subprocesses inherit it -- which
# is why the log directory is an environment variable and not just another
# entry in the redirect table: a test that starts the real entrypoint in a
# SUBPROCESS gets no benefit from an in-process monkeypatch.
if not os.environ.get("OPENDARTS_LOG_DIR"):
    os.environ["OPENDARTS_LOG_DIR"] = str(SANDBOX_ROOT / "data" / "logs")

_collect_repo_path_targets()
_point_repo_paths_at(SANDBOX_ROOT)


def pytest_unconfigure(config: pytest.Config) -> None:
    if _OWNS_SANDBOX and not hasattr(config, "workerinput"):
        shutil.rmtree(SANDBOX_ROOT, ignore_errors=True)


@pytest.fixture(autouse=True)
def _isolate_repo_paths(tmp_path):
    """Second layer: each test gets its OWN copy of the default paths.

    The session sandbox above already keeps the checkout clean; this makes
    the isolation per-test as well, so one test's stray package/config/
    scratch file cannot be found by the next one (or race it under
    `pytest -n auto`). Restores the sandbox values afterwards -- never the
    repo ones, so anything that runs between tests still writes nowhere
    real.
    """
    _point_repo_paths_at(tmp_path / "repo")
    yield
    _point_repo_paths_at(SANDBOX_ROOT)

_SLOW_HELP = (
    "also run @pytest.mark.slow tests (real corpus replays, real "
    "subprocess lifecycles, heavy calibration solves). Default runs "
    "skip them; use this for the full pre-merge run."
)


def pytest_addoption(parser: pytest.Parser) -> None:
    try:
        parser.addoption("--slow", action="store_true", default=False, help=_SLOW_HELP)
    except ValueError:
        # dev/tests/conftest.py reuses this module's fast/slow split and
        # registers the same option, so a run covering BOTH directories
        # (`pytest tests dev/tests`) gets here twice, in whichever order
        # pytest happens to load the two conftests. Adding it once is
        # what matters; the second call is a no-op, not a failure.
        pass


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--slow"):
        return
    skip_slow = pytest.mark.skip(
        reason="slow test skipped by default; run `pytest tests/ --slow` for the full suite"
    )
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(autouse=True)
def _reset_live_derived_module_globals():
    """Real bug found during this integration task's own live-wiring
    pass (2026-08-21): `opendarts.geometry.board.set_ring_boundary_offsets()`
    and `opendarts.capture.board_disc.set_calibrated_board_disc_masks()`
    both mutate plain module-level globals in place -- the correct,
    established pattern for this project's "current rig" state
    (`MEASURED_FOCAL_LENGTH_PX`/`MEASURED_CAMERA_ORIENTATION_HINTS_DEG`
    already work the same way), and correct in production (a single
    process scores one physical rig/session at a time, and deliberately
    KEEPS the last successfully-derived value across a later failed
    bootstrap rather than silently reverting to the hardcoded default --
    same "keep the last good value" posture
    `dev.calibration.intrinsics_derivation.check_intrinsics_drift()`
    already uses for focal length).

    But that exact "keep the last value" behavior is a real hazard
    ACROSS TESTS in one pytest process: a test that exercises
    `bootstrap_calibrations()` (board-disc masks) or
    `load_throw_package()` (ring-boundary offset) with real-shaped
    calibration/session data can leave these globals mutated for every
    OTHER test that runs afterward in the same session -- caught for
    real here (2026-08-21): 19 tests of the since-deleted legacy trigger
    that passed in isolation started failing the moment they ran after
    tests/test_capture_daemon.py in the same pytest invocation, because
    an earlier test's own bootstrap_calibrations() call had derived and
    left in place non-default motion thresholds. Reset to the hardcoded
    defaults before AND after every single test, unconditionally, so no
    test's own global mutation can ever leak into another's.

    The legacy trigger's own derived globals (motion thresholds, per-
    camera top-crop fractions) are gone with it (2026-09); the only
    calibration-derived global the capture loop sets today is the
    board-disc mask registry in `opendarts.capture.board_disc`, which the
    lifecycle's `masks_provider` reads.

    `opendarts.live.diagnostics_gate` (2026-09-04, live-diagnostics runtime
    switch) is the SAME class of process-lifetime global one level
    further up the stack -- a `threading.Event`, not a plain module
    variable, but the exact same cross-test leakage risk applies: a test
    that calls `diagnostics_gate.set_enabled(True)` to exercise the
    gated-on behavior must not leave it on for every OTHER test that
    happens to run afterward in the same pytest process/worker. Reset
    to the default OFF state before AND after every single test, same
    posture as every other global here."""
    from opendarts.capture.board_disc import set_calibrated_board_disc_masks
    from opendarts.geometry.board import set_ring_boundary_offsets
    from opendarts.geometry.board_color import set_board_color_thresholds
    from opendarts.live import diagnostics_gate

    set_ring_boundary_offsets(None, None)
    set_board_color_thresholds(None, None)
    set_calibrated_board_disc_masks(None)
    diagnostics_gate.set_enabled(False)
    yield
    set_ring_boundary_offsets(None, None)
    set_board_color_thresholds(None, None)
    set_calibrated_board_disc_masks(None)
    diagnostics_gate.set_enabled(False)


@pytest.fixture(autouse=True)
def _isolate_capability_probe(tmp_path):
    """`opendarts.live.capabilities` is the same class of process-lifetime
    global as the fixture above: a memoized probe plus a PERSISTED record
    in config.json. Left alone, the first test to touch anything
    audio/calibration-shaped would probe this dev machine's real PATH and
    write a "capabilities" section into the REPO's own data/
    config.json -- a test mutating a real operator file. Point the
    persistence at a per-test tmp file and drop the memo before and
    after every test, so each test probes fresh and writes nowhere real.

    A PRIVATE MonkeyPatch, not the `monkeypatch` fixture, deliberately:
    requesting `monkeypatch` from an autouse fixture makes it set up
    before every test's OWN fixtures -- and therefore torn down after
    them -- which broke two real tests that patch shutil.rmtree and rely
    on their own teardown seeing the real one restored first."""
    from opendarts.live import capabilities

    mp = pytest.MonkeyPatch()
    mp.setattr(capabilities, "CONFIG_PATH",
               tmp_path / "capabilities_config.json")
    capabilities._probed = None
    capabilities._probed_at = None
    yield
    mp.undo()
    capabilities._probed = None
    capabilities._probed_at = None


def stub_confident_orientation(monkeypatch, capture_daemon, hint_deg: float = 100.0):
    """Stub the per-camera orientation solve with a confident, deterministic
    answer.

    For tests whose subject is NOT orientation (calibration packaging,
    replay, tripwires): they need a hint to exist so the bootstrap can
    proceed, but should not depend on real ring-correlation succeeding on
    synthetic frames.
    """
    from opendarts.calibration.ring_correlation_orientation import (
        RingCorrelationOrientationResult,
    )

    def _fake(frames, **_kwargs):
        n = len(list(frames))
        return RingCorrelationOrientationResult(
            ok=True, hint_deg=hint_deg, pass_fraction=1.0,
            n_frames=n, n_passed=n, n_agreeing=n,
            majority_hint_deg=hint_deg, per_frame=[], reason="ok",
        )

    monkeypatch.setattr(capture_daemon, "ring_correlation_orientation_for_camera", _fake)
