"""One free-space floor, consulted by everything that writes big files.

Two things on a rig grow without bound and nothing deletes either of
them: throw packages (~6.5 MB each, one per scored throw) and frame-ring
captures (one per missed-dart or misscore trigger, sized by the ring
window -- roughly 8 MB per second of ring per camera with the cameras'
own JPEG frames, and several times that for raw ones; a 22-second raw
dump on a real rig came to 4.7 GB in one file). A rig that fills its disk
does not merely stop recording: the log stops, the config file stops
being writable, and the next thing to fail is whatever was still working.

So both writers ask here first, and both get the SAME answer from the
SAME floor -- one number in one config key (`min_free_disk_gb`), not a
constant per writer that would drift.

WHAT EACH WRITER DOES WITH A "no":

  * throw packages    skip the package, KEEP SCORING. The throw still
                      scores, the live board and match history are
                      unaffected, and nothing downstream is handed a
                      package directory that does not exist -- exactly
                      the shape `store_packages: false` already has.
                      Logged loudly ONCE per session: a line per throw
                      is how an operator learns to scroll past it.
  * ring dumps        refuse, with the numbers, the way an aged-out
                      request is already refused (see
                      `opendarts.capture.throw_capture`). Never an empty
                      file, never a silent no-op -- docs/DESIGN.md's "a
                      refusal, cap or fallback must say so".

THE ESTIMATE MATTERS AS MUCH AS THE READING. "Is there space right now"
is the wrong question for a ring dump: the ring knows its own byte size,
and a 4.7 GB dump onto 5.1 GB of free space passes that question and
still ends the session with a full disk. So `needed_bytes` is part of the
check, and the refusal quotes both numbers.

FAILING OPEN IS DELIBERATE. If the free-space reading itself is
unavailable -- an unreadable mount, a platform that will not answer --
the check returns `ok=True` and says why in `reason`. A guard that
cannot read the disk must not be the reason a rig stops recording
evidence; it logs, and gets out of the way.

INJECTION, SO THIS IS PROVABLE WITHOUT FILLING A DISK. `free_bytes()` is
a module-level function and `check_free_space()` looks it up at CALL
time, so a test monkeypatching `opendarts.disk_space.free_bytes` changes
what every caller sees. A caller that would rather pass its own reading
(the throw-capture service does, so one service can be given one) uses
the `free_bytes_fn` parameter instead. Nothing calls `shutil.disk_usage`
at a write site.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("opendarts.disk_space")

#: Decimal GB, matching `opendarts.capture.frame_ring.format_bytes()` and
#: the sizing figures in docs/DEPLOYMENT.md. Binary units here would make
#: a 5 GB setting read as 4.66 and invite the wrong comparison.
BYTES_PER_GB = 1_000_000_000

#: What `min_free_disk_gb` resolves to when the key is absent, zero, or
#: unreadable. Five gigabytes is enough headroom for the logs, the config
#: file and one more capture to land safely, and small enough not to
#: refuse writes on a rig that is merely well-used.
DEFAULT_MIN_FREE_DISK_GB = 5.0


def format_gb(n: "float | None") -> str:
    """Bytes as decimal GB, or "unknown" for a reading that failed.

    "unknown" rather than 0 or a guess: a refusal that quotes 0.00 GB
    free when the truth is "the mount would not answer" sends an operator
    to delete files that were never the problem.
    """
    if n is None:
        return "unknown"
    return f"{n / BYTES_PER_GB:.2f} GB"


def resolve_floor_gb(configured: "float | None") -> float:
    """The floor this rig actually uses, in GB.

    `None` (key absent) and `0` both mean "use the default" -- zero is
    the value an operator types when they mean "I have not thought about
    this", and a literal zero floor would be a guard that never fires
    while still looking configured.

    A NEGATIVE value DISABLES the guard entirely. That is the deliberate
    opt-out, and it is spelled as a sign rather than as a second boolean
    key so there is exactly one thing to read to know what this rig does.
    It matches `cv2_num_threads`, where a negative value likewise means
    "stand aside".
    """
    if configured is None:
        return DEFAULT_MIN_FREE_DISK_GB
    try:
        value = float(configured)
    except (TypeError, ValueError):
        return DEFAULT_MIN_FREE_DISK_GB
    if value != value:  # NaN compares unequal to itself
        return DEFAULT_MIN_FREE_DISK_GB
    if value == 0:
        return DEFAULT_MIN_FREE_DISK_GB
    return value


def free_bytes(path: "Path | str") -> "int | None":
    """Free bytes on the filesystem that would hold `path`, or None.

    Walks UP to the first directory that exists, because every caller
    asks about somewhere it has not created yet -- a throw package's
    `<root>/<session>/<throw>` directory, a dump's timestamped directory.
    Asking about the parent is not an approximation: it is the same
    filesystem, which is the only thing this reading is about.

    None (never a fabricated number) when nothing on the way up answers.
    See the module docstring: the guard then stands aside.
    """
    probe = Path(path)
    seen = 0
    while seen < 64:
        seen += 1
        try:
            return int(shutil.disk_usage(probe).free)
        except FileNotFoundError:
            parent = probe.parent
            if parent == probe:
                break
            probe = parent
        except OSError as exc:
            log.warning(
                "free-space check: cannot read the disk holding %s (%s) -- "
                "the free-space guard stands aside for this write", path, exc,
            )
            return None
    log.warning(
        "free-space check: no existing directory above %s answered -- "
        "the free-space guard stands aside for this write", path,
    )
    return None


@dataclass(frozen=True)
class DiskSpaceCheck:
    """One answer, with every number that went into it.

    `reason` is None when `ok` is True AND the guard was actually able to
    check; it carries the explanation otherwise -- including the two
    "ok=True anyway" cases (guard disabled, reading unavailable), because
    a caller reporting "space is fine" should be able to say whether
    anyone actually looked.
    """

    ok: bool
    path: str
    #: False when `min_free_disk_gb` is negative -- the guard is off and
    #: `ok` is True without a reading having been taken.
    enabled: bool
    floor_gb: float
    floor_bytes: int
    #: None when the reading failed; see `free_bytes()`.
    free_bytes: "int | None"
    #: What the caller is about to write, 0 when it does not know.
    needed_bytes: int
    reason: "str | None" = None

    @property
    def free_after_bytes(self) -> "int | None":
        if self.free_bytes is None:
            return None
        return self.free_bytes - self.needed_bytes

    def as_dict(self) -> dict[str, Any]:
        """The shape a refusal payload carries, so a caller (and, later, a
        dashboard) reports the numbers rather than re-deriving them."""
        return {
            "ok": self.ok,
            "enabled": self.enabled,
            "path": self.path,
            "floor_gb": self.floor_gb,
            "floor_bytes": self.floor_bytes,
            "free_bytes": self.free_bytes,
            "free_gb": (
                None if self.free_bytes is None
                else round(self.free_bytes / BYTES_PER_GB, 2)
            ),
            "needed_bytes": self.needed_bytes,
            "free_after_bytes": self.free_after_bytes,
            "reason": self.reason,
        }


def check_free_space(
    path: "Path | str",
    *,
    floor_gb: "float | None" = None,
    needed_bytes: int = 0,
    free_bytes_fn: "Callable[[Path], int | None] | None" = None,
) -> DiskSpaceCheck:
    """Would writing `needed_bytes` at `path` leave the floor intact?

    `floor_gb` is the RAW configured value, not a resolved one --
    `resolve_floor_gb()` is applied here so every caller treats absent,
    zero and negative the same way.

    `free_bytes_fn` is the injection point for a caller that holds its
    own reading. Left None, the module-level `free_bytes` is looked up at
    call time, which is what makes monkeypatching it work.
    """
    floor = resolve_floor_gb(floor_gb)
    target = str(path)
    needed = max(0, int(needed_bytes))

    if floor < 0:
        return DiskSpaceCheck(
            ok=True, path=target, enabled=False, floor_gb=floor,
            floor_bytes=0, free_bytes=None, needed_bytes=needed,
            reason=(
                f"the free-space guard is disabled (min_free_disk_gb={floor:g}, "
                "a negative value means stand aside) -- nothing was checked"
            ),
        )

    floor_bytes = int(floor * BYTES_PER_GB)
    reader = free_bytes_fn if free_bytes_fn is not None else free_bytes
    free = reader(Path(path))

    if free is None:
        return DiskSpaceCheck(
            ok=True, path=target, enabled=True, floor_gb=floor,
            floor_bytes=floor_bytes, free_bytes=None, needed_bytes=needed,
            reason=(
                f"free space on {target} could not be read, so the "
                f"{format_gb(floor_bytes)} floor could not be enforced -- "
                "allowing the write rather than refusing on a number nobody has"
            ),
        )

    free = int(free)
    remaining = free - needed
    if remaining >= floor_bytes:
        return DiskSpaceCheck(
            ok=True, path=target, enabled=True, floor_gb=floor,
            floor_bytes=floor_bytes, free_bytes=free, needed_bytes=needed,
        )

    if needed > 0:
        reason = (
            f"{format_gb(free)} is free on {target} and this write is about "
            f"{format_gb(needed)}, which would leave {format_gb(remaining)} -- "
            f"below the {format_gb(floor_bytes)} floor (min_free_disk_gb="
            f"{floor:g})"
        )
    else:
        reason = (
            f"only {format_gb(free)} is free on {target} -- below the "
            f"{format_gb(floor_bytes)} floor (min_free_disk_gb={floor:g})"
        )
    return DiskSpaceCheck(
        ok=False, path=target, enabled=True, floor_gb=floor,
        floor_bytes=floor_bytes, free_bytes=free, needed_bytes=needed,
        reason=reason,
    )
