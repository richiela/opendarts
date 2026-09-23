"""Engine registry -- name -> engine instance. See docs/ENGINES.md's
"Registering an engine" section: an engine module lives at
opendarts/engines/<name>.py, implements the score() (and optionally
calibrate()) interface in opendarts.engines.base, and gets one line added
here. That's the entire integration surface for a new engine -- it never
touches capture, triggering, packages, or the dashboard directly.
"""
from __future__ import annotations

from opendarts.engines.athena import AthenaEngine
from opendarts.engines.apollo import ApolloEngine
from opendarts.engines.ares import AresEngine
from opendarts.engines.talos import TalosEngine
from opendarts.engines.zeus import ZeusEngine
from opendarts.engines.zeus.engine import ZEUS_SUB_ENGINE_NAMES

ENGINES: dict[str, object] = {
    "Apollo": ApolloEngine(),
    "Talos": TalosEngine(),
    "Athena": AthenaEngine(),
    # Greenfield 2D-per-camera engine built on board-plane shaft-line
    # concurrency (opendarts.engines.ares) -- registered as an also-run.
    # 2026-08-25: added to ZEUS_SUB_ENGINE_NAMES as Zeus's 4th voter,
    # the project's own direct instruction ("we're going to throw ares into
    # the Zeus engine") -- see opendarts/engines/zeus/engine.py's own
    # module docstring for the full quote, reasoning, and real corpus
    # measurement, and docs/DESIGN.md's matching dated guardrail update.
    "Ares": AresEngine(),
    # A majority-vote COMBINER over Apollo/Talos/Athena/Ares's own
    # real outputs (opendarts.engines.zeus) -- not its own detector or
    # calibration consumer. Read opendarts/engines/zeus/engine.py's own top
    # docstring for the tie-break convention (Apollo always wins
    # a tie it's part of), the quorum gate (MIN_SUB_ENGINES_TO_VOTE=3 of
    # 4, as of 2026-08-25), and the real corpus numbers -- the original
    # 3-engine 419/420 (data/archive/clean/, now-superseded corpus
    # location) is preserved there as historical record; the current
    # 4-engine config measures 612/614 on today's session corpus,
    # unchanged from the 3-engine baseline on the same corpus.
    "Zeus": ZeusEngine(),
}

# The engine used when nothing else has been configured. Zeus is the
# consensus combiner -- it is the engine that gives the answer, with the
# four detection engines feeding it.
DEFAULT_PRIMARY_ENGINE = "Zeus"

# Also-run engines for a fresh config: Zeus's own sub-engines, so each
# one's individual answer is recorded alongside the consensus and the
# dashboard's Engines tab has per-engine rows to show. Derived from
# `ZEUS_SUB_ENGINE_NAMES` rather than repeated, so the two cannot drift.
#
# Not extra work: when Zeus is primary AND its sub-engines are also-run,
# their results are REUSED from the consensus pass rather than
# re-dispatched (see `_reuse_zeus_sub_results_for_also_run()`).
DEFAULT_ALSO_RUN: tuple[str, ...] = ZEUS_SUB_ENGINE_NAMES


def engine_names() -> list[str]:
    """Every registered engine name, in registry-definition order --
    used by the dashboard's Config tab to render the primary radio /
    also-run checkboxes without hardcoding the list twice."""
    return list(ENGINES.keys())


def get_engine(name: str) -> object:
    """Look up a registered engine by name. Raises KeyError (not None)
    on an unknown name -- an unknown engine name is a real configuration
    bug (typo, stale name after a rename) that should surface loudly,
    not be silently skipped; callers accepting user/config input (e.g.
    opendarts.live.capture_daemon.EngineConfigStore.set()) validate against
    engine_names() BEFORE it ever reaches here, so a KeyError here should
    never actually happen in normal operation."""
    return ENGINES[name]


def is_registered(name: str) -> bool:
    return name in ENGINES
