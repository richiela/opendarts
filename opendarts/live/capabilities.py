"""opendarts/live/capabilities.py -- which external tools this rig
actually has, probed once per process and recorded in the shared config.

WHY THIS EXISTS (2026-09-12, the owner's own request): tool availability
used to be discovered by ATTEMPTING the work and handling the failure.
On the Windows PC, which has no ffmpeg, every calibration event walked
into the raw-video encode and reported "raw video encode failed" -- a
per-session error for a fact about the machine that never changes
mid-session. The request, verbatim: "check dependencies on first run and
store it in configs ... such as 'hasffmpeg': true and only try the call
encode if the tool exists". So call sites now ask `has()`/`tool_path()`
up front and skip work that cannot succeed, and the probed answer is
written to config.json's "capabilities" section where an operator
(or a remote `curl /api/health`) can read what this rig could do.

UPDATE (2026-09-20): ffmpeg -- the tool that motivated this whole module
-- is no longer used anywhere (calibration raw video now encodes via
cv2's bundled FFV1; macOS camera listing moved to system_profiler), so it
was dropped from the probe list. `git` (calibration provenance) is now
the only external tool probed. The pattern stays: it is the right shape
for the next optional tool, and the git probe still gates a doomed spawn.

WHAT THE CACHE ACTUALLY BUYS, honestly: not the probe cost. Every probe
here is a `shutil.which` (a handful of directory stats) or a stdlib
import -- microseconds. What the gate buys is never SPAWNING a doomed
subprocess (an encode, a render, an enumeration) and never logging its
failure as if something broke; what the config record buys is
visibility -- a durable per-rig statement of what was available, on disk
next to the operator's other settings.

STALENESS -- the design decision, and why. A persisted "false" that
outlived an ffmpeg install would keep raw-video encodes off FOREVER,
invisibly: the feature just never comes back, and nothing looks broken.
That failure mode is worse than anything the persistence could save,
and what it could save is microseconds. So the persisted record is a
RECORD, never an authority: this module re-probes fresh on first use in
every process, memoizes in memory for the process lifetime, and writes
the config section only to keep the on-disk record current (and only
when it changed, so a stable rig never churns the file). The staleness
window is therefore one process lifetime -- the same contract as
calibration_package._code_version()'s lru_cache ("a fresh process (a
real deploy) gets a fresh cache"), and every deploy already restarts
the process. `refresh()` exists for anything that wants a mid-process
re-probe. A corrupt or hand-mangled config section costs nothing:
reads here only decide whether to REWRITE the section, never what the
answer is, so garbage degrades to "write a good record over it" --
and write_config_section itself refuses to clobber an unparseable
file, preserving the operator's other settings.

NEVER RAISES, never spawns. This is consulted from the capture path
(calibration packaging, camera enumeration); scoring correctness always
wins, so every failure here degrades to "tool absent" plus a log line.
Nothing in this module runs a subprocess -- presence is established by
PATH lookup, and whether the tool actually WORKS is still the
call site's problem to handle at attempt time (a stale "true" after an
uninstall self-corrects there: the attempt fails and is handled, same
as before this module existed).
"""
from __future__ import annotations

import logging
import platform
import shutil
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from opendarts.live.config import DEFAULT_CONFIG_PATH, read_config_section, write_config_section

log = logging.getLogger("opendarts.live.capabilities")

CONFIG_SECTION = "capabilities"

#: Where the record persists. Module-level (not a per-call parameter) on
#: purpose: call sites ask `has("ffmpeg")` from deep inside capture code
#: and cannot thread a path through; tests point this at a tmp file.
CONFIG_PATH = DEFAULT_CONFIG_PATH

#: Every external executable any part of this product shells out to.
#: git -- calibration provenance (`_code_version()`). Probed on every
#: platform: `which` for a tool the platform never ships is a fast honest
#: "no", and one uniform record beats per-platform key sets.
#:
#: FFMPEG IS GONE (2026-09-20): calibration raw-video now encodes through
#: cv2.VideoWriter (the FFV1 codec bundled in the opencv wheel), and macOS
#: camera listing moved to `system_profiler` -- so nothing in the product
#: shells out to ffmpeg any more, and a probe with no call site is worse
#: than none (it publishes a /api/health fact that reads like a capability
#: the product has an opinion about). Same reasoning that retired the
#: audio probes (2026-09-15): `say`/`afplay`/`aplay`/`paplay` went when
#: TTS and spoken calls moved into the browser, and `winsound` took the
#: whole stdlib-import probe branch with it (its only ever entry).
WHICH_TOOLS = ("git",)

_lock = threading.Lock()
_probed: Optional[dict[str, Optional[str]]] = None
_probed_at: Optional[str] = None


def _probe_one(name: str) -> Optional[str]:
    """Resolved location of one tool, or None. Never raises, never spawns."""
    try:
        return shutil.which(name)
    except Exception:  # noqa: BLE001 -- a broken PATH env must read as "absent", not crash
        log.warning("probe for %r failed -- treating as absent", name, exc_info=True)
        return None


def _record(paths: dict[str, Optional[str]], probed_at: str) -> dict[str, Any]:
    """The persisted/reported shape. `present` is the flag call sites and
    operators care about; `path` says WHICH binary answered, so "it says
    yes but the encode fails" is debuggable from the record alone."""
    return {
        "platform": platform.system(),
        "probed_at_utc": probed_at,
        "tools": {
            name: {"present": path is not None, "path": path}
            for name, path in paths.items()
        },
    }


def _ensure_probed() -> dict[str, Optional[str]]:
    global _probed, _probed_at
    with _lock:
        if _probed is None:
            paths = {name: _probe_one(name) for name in WHICH_TOOLS}
            probed_at = datetime.now(timezone.utc).isoformat()
            _persist(paths, probed_at)
            # Assigned only after a fully-successful probe pass, so a
            # half-built dict can never be memoized.
            _probed, _probed_at = paths, probed_at
        return _probed


def _persist(paths: dict[str, Optional[str]], probed_at: str) -> None:
    """Bring the on-disk record up to date; write only on change.

    Change detection compares `present` flags plus platform, not paths or
    timestamps -- a Homebrew upgrade moving a binary is not worth a config
    write, and comparing the timestamp would mean writing every boot."""
    try:
        record = _record(paths, probed_at)
        stored = read_config_section(CONFIG_SECTION, path=CONFIG_PATH)
        stored_tools = stored.get("tools") if isinstance(stored, dict) else None
        stored_present: dict[str, Any] = {}
        if isinstance(stored_tools, dict):
            for name, entry in stored_tools.items():
                if isinstance(entry, dict):
                    stored_present[name] = entry.get("present")
        fresh_present = {n: p is not None for n, p in paths.items()}
        stored_platform = stored.get("platform") if isinstance(stored, dict) else None
        if stored_present == fresh_present and stored_platform == record["platform"]:
            return
        # A tool appearing or vanishing since the last recorded probe is
        # exactly the event an operator asks about ("I installed ffmpeg,
        # did the rig notice?") -- one info line answers it.
        for name, present in sorted(fresh_present.items()):
            was = stored_present.get(name)
            if was is not None and bool(was) != present:
                log.info("capability change: %s was %s, now %s",
                         name, "present" if was else "absent",
                         "present" if present else "absent")
        write_config_section(CONFIG_SECTION, record, path=CONFIG_PATH)
    except Exception:  # noqa: BLE001 -- the record is a convenience; the probe result stands
        log.warning("could not persist capability record", exc_info=True)


def tool_path(name: str) -> Optional[str]:
    """Resolved path for `name`, or None when the rig does not have it.

    Unknown names are probed live (never raise): a future call site that
    asks about a tool this registry has not caught up with should get a
    truthful answer, just not a persisted one."""
    probed = _ensure_probed()
    if name in probed:
        return probed[name]
    log.debug("capability %r is not in the registry -- probing unpersisted", name)
    return _probe_one(name)


def has(name: str) -> bool:
    """Whether this rig has `name`. The gate call sites use."""
    return tool_path(name) is not None


def snapshot() -> dict[str, Any]:
    """The full capability record (same shape as persisted), for
    diagnostics surfaces. Memoized-cheap after first use."""
    probed = _ensure_probed()
    return _record(probed, _probed_at or "")


def refresh() -> dict[str, Any]:
    """Drop the process memo, re-probe everything, persist, and return
    the fresh record -- the mid-process escape hatch for "I just
    installed the tool, look again"."""
    global _probed, _probed_at
    with _lock:
        _probed = None
        _probed_at = None
    return snapshot()
