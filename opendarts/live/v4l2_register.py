"""Check that the v4l2loopback virtual cameras exist on Linux.

The Linux counterpart of `opendarts.live.vcam_register`, and deliberately
the same shape: `available()`, `apply(enabled)`, and the same result dict,
so the call sites in `run_product` and `server` do not care which platform
they are on.

WHERE IT DIFFERS FROM WINDOWS, and why it has to. `regsvr32` writes COM
keys under HKCU, so the Windows build can create and destroy its own
devices as an ordinary user, and `apply()` really does register and
unregister. v4l2loopback is a KERNEL MODULE: creating the devices is
`modprobe`, which is root-only, and no amount of arranging makes that
something a capture process should do to the machine it runs on.

So `apply()` here reports rather than acts. It answers "are the loopback
devices present and usable", which is the question the caller actually has
-- the Windows call answers the same question by making it true, and this
one by checking. A rig without the module gets a clear reason naming the
command to run, instead of publishing into nothing.

WHY THE DEVICES PERSIST, where Windows unregisters. On Windows
`apply(False)` really does remove the virtual cameras, because leaving
them registered puts three synthetic devices in every capture
application's list on that machine, and `regsvr32` can do it unprivileged
under HKCU. v4l2loopback can do the same at runtime -- the
`/dev/v4l2loopback` control device takes ADD and REMOVE ioctls -- but that
node is root:root mode 600, so a capture process cannot use it without
either a udev rule granting the video group access or a privilege
escalation this has no business performing.

Chosen deliberately, 2026-09-15: the devices are created once by the
setup script and stay. The oracle toggle controls PUBLISHING only. The
cost is three idle entries in other applications' camera lists; the
alternative costs a privilege grant, and the ADD ioctl's config struct
has changed shape between v4l2loopback releases, so it is version-bound
in a way nothing else here is. Revisit if the idle entries become a real
nuisance.

Setting the devices up is a one-time root step, documented in
docs/LINUX.md:

    sudo ./scripts/setup_linux.sh

which reduces to, on a Secure Boot machine:

    sudo apt install linux-modules-v4l2loopback-generic
    sudo modprobe v4l2loopback devices=3 video_nr=10,11,12 \\
         card_label="OpenDarts Cam 0,OpenDarts Cam 1,OpenDarts Cam 2" \\
         exclusive_caps=0,0,0

exclusive_caps=0 is NOT the value the guides give. Measured on the rig:
with 1, a node returns EBUSY to any open while a producer is attached,
and advertises no capture capability while idle -- so a consumer that
enumerates by opening each node can never see it, either way.
"""
from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any

log = logging.getLogger("opendarts.live.v4l2_register")

#: Where the loopback devices are expected. Matches the `video_nr=`
#: argument in the documented modprobe line. Deliberately a fixed range
#: rather than "scan every /dev/video*": the real cameras are also
#: /dev/video* nodes, and publishing into one of those would feed a
#: camera its own picture.
DEFAULT_DEVICE_NUMBERS = (10, 11, 12)

#: v4l2loopback reports itself here once the module is loaded, whether or
#: not any device is currently open.
MODULE_MARKER = Path("/sys/module/v4l2loopback")


def module_loaded() -> bool:
    """Whether the v4l2loopback kernel module is loaded."""
    return MODULE_MARKER.is_dir()


def loopback_devices(numbers: "tuple[int, ...]" = DEFAULT_DEVICE_NUMBERS) -> "list[Path]":
    """The loopback device nodes that actually exist, in slot order.

    Existence only -- whether they are WRITABLE is a separate question,
    answered at publish time, because a device can exist and still be
    denied by group membership.
    """
    return [p for n in numbers if (p := Path(f"/dev/video{n}")).exists()]


def available() -> bool:
    """Whether publishing to loopback devices is possible at all here.

    Reported honestly so a caller can say "not applicable" rather than
    offering a control that silently does nothing -- the same contract
    `vcam_register.available()` has on Windows.
    """
    return platform.system() == "Linux" and module_loaded() and bool(loopback_devices())


def apply(enabled: bool) -> "dict[str, Any]":
    """Make the reported state match the oracle toggle.

    The single entry point callers use, mirroring `vcam_register.apply`.
    Nothing is created or destroyed here -- see the module docstring --
    so this reduces to reporting whether the devices the toggle needs are
    actually there.
    """
    if platform.system() != "Linux":
        return _not_applicable("v4l2loopback is Linux-only")
    if not enabled:
        # Nothing to tear down: the devices are the operator's, created
        # out-of-band, and a capture process has no business removing
        # them just because it stopped publishing.
        return {"ok": True, "applicable": True, "action": "idle",
                "reason": None}
    if not module_loaded():
        return _not_applicable(
            "the v4l2loopback kernel module is not loaded -- run "
            "`sudo ./scripts/setup_linux.sh` (see docs/LINUX.md)"
        )
    devices = loopback_devices()
    if not devices:
        return _not_applicable(
            f"v4l2loopback is loaded but none of "
            f"{', '.join('/dev/video%d' % n for n in DEFAULT_DEVICE_NUMBERS)} exist "
            "-- reload the module with the documented video_nr="
        )
    log.info("v4l2loopback devices present: %s -- restart the consuming app "
             "for it to enumerate them", ", ".join(str(d) for d in devices))
    return {"ok": True, "applicable": True, "action": "verify",
            "reason": None, "devices": [str(d) for d in devices]}


def _not_applicable(reason: str) -> "dict[str, Any]":
    """Nothing was attempted and nothing failed.

    `ok` is TRUE for the same reason it is in `vcam_register`: a machine
    where this does not apply is healthy, not broken, and a caller
    checking `ok` must not report it as a failure. `applicable` carries
    the real distinction.
    """
    return {"ok": True, "applicable": False, "action": None, "reason": reason}
