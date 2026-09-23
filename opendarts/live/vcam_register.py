"""Register and unregister the DirectShow virtual cameras with Windows.

The companion to `vcam_publish`, which fills the shared memory those
cameras read from. This module is about whether the cameras EXIST as far
as Windows is concerned -- publishing frames to devices nothing has
registered is writing into a buffer with no reader.

Both follow the oracle toggle for the same reason. The virtual
cameras exist so other software can watch the same board as this
product on Windows, where the two cannot share a physical camera:
DirectShow takes a device exclusively. With the toggle off there is
nobody to watch them, and three synthetic cameras left in every capture application's
device list on the machine is litter with no purpose.

Windows only, and a no-op everywhere else -- macOS shares cameras
natively through AVFoundation, and Linux reaches the same place through
v4l2loopback, which is a kernel module rather than a COM registration.

ALWAYS UNREGISTERS BEFORE REGISTERING. Registration is registry state
that outlives the process that wrote it, so the machine can be carrying
a previous registration made by a different copy of this DLL -- an older
build, a different checkout path, or a machine-wide one from before
per-user registration existed. Registering on top of that is what
produces the genuinely confusing failure: the enumerator lists devices
whose InprocServer32 points somewhere the DLL no longer is, so they
appear in every application's camera list and fail to open. The DLL's
own DllUnregisterServer sweeps BOTH registry hives, so a single
unregister clears every variant, and doing it unconditionally means
there is only ever one registration to reason about.

NEVER RAISES. This sits on the oracle toggle, which an operator hits
mid-session. A registry write that fails must leave the toggle working
and say so, not surface as an error on a control that has nothing to do
with the registry.
"""
from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("opendarts.live.vcam_register")

#: Shipped prebuilt, because the machines that need it are Windows boxes
#: with no C++ toolchain -- see tools/winvcam/README.md.
DLL_PATH = (Path(__file__).resolve().parent.parent.parent
            / "tools" / "winvcam" / "vcam_probe.dll")

#: regsvr32 is synchronous and touches only the registry, so it is quick.
#: The timeout exists for the pathological case (a hung COM server during
#: DllRegisterServer) rather than as a realistic expectation.
TIMEOUT_S = 30.0


def available() -> bool:
    """Whether registration is possible on this machine at all.

    Reported honestly so a caller can say "not applicable here" rather
    than offering a control that silently does nothing.
    """
    return platform.system() == "Windows" and DLL_PATH.is_file()


def _regsvr32(*args: str) -> "tuple[bool, str]":
    """One regsvr32 invocation. Returns (ok, detail)."""
    cmd = ["regsvr32", "/s", *args, str(DLL_PATH)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        # /s suppresses the message box, which also means the only thing
        # left to report is the exit code -- the DLL's own log at
        # %TEMP%\vcam_probe.log carries the actual per-step reason.
        detail = (proc.stderr or proc.stdout or "").strip()
        return False, (f"regsvr32 exit {proc.returncode}"
                       + (f": {detail}" if detail else
                          " -- see %TEMP%\\vcam_probe.log for the failing step"))
    return True, ""


def unregister() -> "dict[str, Any]":
    """Remove any existing registration, in both hives.

    Not an error when nothing was registered: the DLL deletes keys that
    may not exist and reports success, which is what makes this safe to
    call unconditionally before a register.
    """
    if not available():
        return _not_applicable()
    ok, detail = _regsvr32("/u")
    if ok:
        log.info("virtual cameras unregistered")
    else:
        log.warning("virtual camera unregister failed: %s", detail)
    return {"ok": ok, "applicable": True, "action": "unregister",
            "reason": detail or None}


def register() -> "dict[str, Any]":
    """Unregister, then register. See the module docstring for why the
    first half is unconditional."""
    if not available():
        return _not_applicable()
    # Result deliberately ignored: a failure here is usually "nothing was
    # registered", which is the normal first-run state and must not stop
    # the registration that follows.
    _regsvr32("/u")
    ok, detail = _regsvr32()
    if ok:
        # Worth saying every time. A consuming app may enumerate devices
        # only at startup, so a registration that happens while it is
        # already running can be invisible to it until it restarts -- which reads as
        # "the virtual cameras do not work" rather than "look again".
        log.info("virtual cameras registered -- restart the consuming app "
                 "for it to enumerate them")
    else:
        log.warning("virtual camera registration failed: %s", detail)
    return {"ok": ok, "applicable": True, "action": "register",
            "reason": detail or None}


def apply(enabled: bool) -> "dict[str, Any]":
    """Make registration match the oracle toggle.

    The single entry point callers should use, so "on means registered,
    off means not" lives in one place rather than at each call site.
    """
    return register() if enabled else unregister()


def _not_applicable() -> "dict[str, Any]":
    """The macOS/Linux answer, and the missing-DLL one.

    `ok` is TRUE here: nothing was attempted and nothing failed. It used
    to be False, which made the correct, healthy state on every
    non-Windows machine indistinguishable from a registry write that
    genuinely blew up -- so a caller checking `ok` would report a working
    Mac as broken. `applicable` carries the real distinction, and a
    caller that means "are the cameras actually registered" wants
    `ok and applicable`.
    """
    return {"ok": True, "applicable": False, "action": None,
            "reason": _unavailable_reason()}


def _unavailable_reason() -> str:
    if platform.system() != "Windows":
        return "virtual cameras are Windows-only"
    return f"the filter DLL is not present at {DLL_PATH}"
