"""Every live module must IMPORT on a machine without the UNIX-only stdlib.

WHY THIS FILE EXISTS. On 2026-09-15 the Windows rig was restarted, pulled
the Linux virtual-camera work for the first time, and refused to boot:

    ModuleNotFoundError: No module named 'fcntl'

`opendarts/live/v4l2_publish.py` imported `fcntl` at module level, and
`opendarts/live/vcam.py` imports BOTH backends before choosing one by
platform. So the Linux-only import ran on Windows during startup -- long
before anything asked whether virtual-camera publishing was available.

The bug is invisible to the rest of the suite because CI and both dev
machines are UNIX, where `import fcntl` simply works. Nothing here checks
behaviour; the only claim is that importing the module does not explode,
which is precisely the claim that was false.

Whole-package rather than a single regression test for fcntl: the next one
of these will be `termios` or `pwd` in some other backend, and naming one
module would not have caught this one either.
"""
from __future__ import annotations

import builtins
import importlib
import pkgutil
import sys

import pytest

#: Present on Linux and macOS, absent on Windows. `resource` belongs here
#: too -- it is only ever imported inside a function.
UNIX_ONLY = {"fcntl", "termios", "tty", "pwd", "grp", "resource", "syslog", "curses"}


@pytest.fixture
def no_unix_stdlib(monkeypatch):
    """Make the UNIX-only stdlib modules look absent, as they are on
    Windows. Patching __import__ rather than sys.modules because the
    failure being reproduced happens AT import, not at attribute access."""
    real = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in UNIX_ONLY:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    # Drop anything already imported so the guard is actually exercised
    # rather than served from cache.
    for mod in [m for m in sys.modules if m.startswith("opendarts")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    yield


def _live_module_names() -> list[str]:
    import opendarts.live as live
    return sorted("opendarts.live." + m.name for m in pkgutil.iter_modules(live.__path__))


def test_every_live_module_imports_without_the_unix_only_stdlib(no_unix_stdlib):
    broken: list[str] = []
    for name in _live_module_names():
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as exc:
            # ONLY a missing UNIX module is this test's business. A rig
            # without opencv or fastapi installed is a different problem,
            # and failing on it here would make this test unrunnable
            # rather than informative.
            if any(u in str(exc) for u in UNIX_ONLY):
                broken.append(f"{name}: {exc}")
        except Exception: # noqa: BLE001 -- import-time side effects are not this test's claim
            pass
    assert not broken, (
        "these modules cannot be imported on Windows:\n  " + "\n  ".join(broken)
    )


def test_the_virtual_camera_dispatcher_survives_it(no_unix_stdlib):
    """The actual startup path that broke: vcam imports both backends and
    only then picks one."""
    vcam = importlib.import_module("opendarts.live.vcam")
    assert vcam.publish.__name__.endswith("vcam_publish"), (
        "off Linux the dispatcher must land on the Windows backend")


def test_linux_publishing_reports_unavailable_rather_than_half_working(no_unix_stdlib):
    """Importing is not enough -- `available()` must say False, or a caller
    offers a control whose one ioctl would raise."""
    v4l2 = importlib.import_module("opendarts.live.v4l2_publish")
    assert v4l2.fcntl is None, "the guard did not take effect"
    assert v4l2.available() is False
