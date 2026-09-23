"""opendarts/live/build_info.py -- what this process IS, for the operator
and for a bug report.

Answers one question: **what am I running?** Until this existed there was
no way to tell from the UI, and two sessions independently resorted to
`curl` and `ssh git log -1` to confirm a deploy had landed. A build
identifier is the single most valuable field in any bug report, because
without it every report starts with "which version are you on?".

WHAT IS DELIBERATELY NOT HERE. No branch, no working-tree-dirty flag, no
repo paths, no pid, no poll intervals, no module names -- those are
developer trivia, and a page full of them is what made the old Info tab
feel like a scratchpad rather than part of a product. Live camera state is
also absent on purpose: it has its own surfaces (`/api/cameras/status` and
the preview tiles) which UPDATE, and a static copy beside them would
eventually disagree with them.

Every value degrades to `None` rather than raising. This is read by a
status page; a machine that cannot report its own OpenCV build must still
serve darts.
"""
from __future__ import annotations

import os
import platform
import sys
import time
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

# Reuses the SAME function that stamps calibration packages, rather than
# re-shelling out to git here. One source of truth means the build shown
# on screen is provably the build written into a package's provenance --
# if they could drift, the screen would eventually lie about which code
# produced a saved throw. It is `lru_cache`d there, so this costs one
# subprocess for the life of the process and nothing after.
from opendarts.capture.calibration_package import _code_version

@lru_cache(maxsize=1)
def _app_version() -> str | None:
    """The released version, read from pyproject.toml -- the one place it is set.

    Distinct from `code_version`, and not a replacement for it. That is a git
    SHA, and it is `None` for everyone running a published copy: the artifact
    ships WITHOUT `.git` on purpose, so the dashboard has been telling every
    downloaded user "no git checkout -- running from a copy?" where a version
    belongs. A release number is the only version such a copy can report, and
    the only one a stranger can put in a bug report.

    Read from the file rather than `importlib.metadata` because OpenDarts runs
    from a checkout, not a `pip install` -- the metadata may not exist. Falls
    back to it anyway if someone does install this, then to `None`, because
    this module never raises at a status page.
    """
    try:
        with (Path(__file__).resolve().parents[2] / "pyproject.toml").open("rb") as fh:
            v = tomllib.load(fh)["project"]["version"]
            return str(v) if v else None
    except Exception:
        pass
    try:
        from importlib import metadata
        return metadata.version("opendarts")
    except Exception:
        return None


#: Process start, captured at import. Import happens once at startup, so
#: this is the process's own birth time to within the import itself.
_STARTED_AT = time.time()


def _uptime_s() -> float:
    return max(0.0, time.time() - _STARTED_AT)


def _opencv() -> "tuple[str | None, str | None, int | None]":
    """(version, parallel framework, configured thread count).

    The framework is reported because it decides whether the thread count
    beside it can be believed at all: on a macOS build ("GCD")
    `setNumThreads(1)` genuinely halves CPU per unit of work while
    `getNumThreads()` keeps reporting the full core count. The Windows rig
    builds against "Concurrency", where the number does round-trip.
    """
    try:
        import cv2
    except Exception: # noqa: BLE001
        return None, None, None
    version = getattr(cv2, "__version__", None)
    framework = None
    try:
        for line in cv2.getBuildInformation().splitlines():
            if "Parallel framework" in line:
                framework = line.split(":", 1)[1].strip()
                break
    except Exception: # noqa: BLE001
        pass
    try:
        threads = int(cv2.getNumThreads())
    except Exception: # noqa: BLE001
        threads = None
    return version, framework, threads


def build_info() -> dict[str, Any]:
    """Identity and runtime of this process.

    `code_version` is `None` whenever git is unavailable or this is not a
    checkout -- running from a tarball, which stops being an edge case the
    moment this repo is copied to a public git. Callers must render that
    as an honest "unknown" rather than a blank cell, which reads as a
    broken row.
    """
    cv_version, cv_framework, cv_threads = _opencv()
    try:
        numpy_version: str | None = __import__("numpy").__version__
    except Exception: # noqa: BLE001
        numpy_version = None
    try:
        host = platform.node() or None
    except Exception: # noqa: BLE001
        host = None

    return {
        "app_version": _app_version(),
        "code_version": _code_version(),
        "started_at_epoch": _STARTED_AT,
        "uptime_s": round(_uptime_s(), 1),
        "hostname": host,
        "platform": platform.system() or None,
        "platform_release": platform.release() or None,
        "python_version": platform.python_version(),
        "opencv_version": cv_version,
        "opencv_parallel_framework": cv_framework,
        "opencv_threads": cv_threads,
        "numpy_version": numpy_version,
        "cpu_count": os.cpu_count(),
        "executable": os.path.basename(sys.executable) or None,
    }
