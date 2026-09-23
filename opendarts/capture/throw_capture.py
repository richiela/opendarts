"""opendarts/capture/throw_capture.py -- the two triggers that turn the
frame ring into evidence on disk, and the honesty rules they obey.

TWO FAILURES, TWO WINDOWS. They are not variations of one another; they
want different amounts of the ring and they treat the ring differently
while writing:

                 window                          ring during write
  missed dart    the whole buffer                stop, dump, restart
  misscore       ~1s around THE THROW'S TIME     keep running

A missed dart has no throw package and therefore no timestamp to anchor
on, so the only defensible window is everything held. At 22 seconds and
three 720p cameras that is ~6GB, which is why this one pauses the ring
for the duration: peak memory stays at one buffer instead of one buffer
plus whatever arrives during the write.

A misscore has a package, so it has an anchor, and ~1s around that anchor
is ~270MB -- small enough that the ring goes on filling throughout and
the rig never stops recording.

THE ANCHOR IS THE THROW'S RECORDED TIME, NOT "THE LAST N FRAMES". By the
time anyone presses a misscore button they have walked to a screen and
the board is empty; the last N frames are frames of an empty board. This
is the single most important thing about the misscore path and it is why
`capture_misscore()` takes a wall-clock instant rather than a count.

THE WINDOW IS ASYMMETRIC, MORE BEFORE THAN AFTER, AND DELIBERATELY
GENEROUS. Detection fires several frames AFTER the landing -- the
lifecycle waits out `dart_stable_frames` plus its cooldowns -- so the
recorded timestamp is already post-settle. There is a second, larger
offset on top of that, and it is documented in the capture daemon itself
(`_build_capture_diagnostics()`): `meta.captured_at_utc` is stamped at the
END of handle_ready_to_capture()'s synchronous path, i.e. AFTER the
engine has scored, so the frames were in hand `handle_total_s` earlier
than the timestamp says. `anchor_wall_s_for_package()` below subtracts
that back out when the package recorded it, and says which anchor it
used.

-0.5s/+0.2s is a starting point, not a tuned value, and it must not be
tightened on reasoning. Capture real misscores first, look at where the
landing actually falls relative to the timestamp, then trim with
evidence. The saving from a tighter window is ~190MB against a 436GB
disk; clipping the landing off a capture that cannot be retaken is the
expensive mistake, and it is expensive in the direction that cannot be
undone.

WHO PULLS THE TRIGGER:

  * misscore, oracle available     -> automatic, on disagreement
  * misscore, no oracle            -> a button on the engine row (the
                                      dashboard already has per-throw
                                      correction, so it already knows
                                      WHICH throw -- session + throw_id)
  * missed dart, no oracle         -> a button; nothing can detect a miss
                                      without an oracle, by definition
  * missed dart, oracle available  -> automatic: the oracle committed a
                                      throw and this rig did not. On a
                                      stream-fed consumer this is the most
                                      valuable signal available, because
                                      it is exactly the cross-platform
                                      difference the test rigs exist to
                                      find, and it needs nobody present.

THREE HONESTY REQUIREMENTS, the first two the same class of bug that cost a
full day on 2026-09-15, and all of them covered by docs/DESIGN.md's "a
refusal, cap or fallback must say so":

  * AGED OUT. If the requested throw is older than the oldest frame in the
    ring, say so WITH THE NUMBERS ("the oldest frame is 22s old"). Never
    write an empty file, and never no-op while looking successful. The
    refusal is `FrameRing.slice_around()`'s, the log line is here, and the
    caller gets `ok: False` with the reason in it.
  * DISK FLOOR. A dump is the largest single write this product makes,
    and the ring knows its own byte size BEFORE anything is written --
    so "is there room for THIS dump" is answerable in advance, and is
    asked in advance (`opendarts.disk_space`). Below the floor the dump
    is refused in the same shape as an aged-out request -- `ok: False`
    and a reason carrying the numbers -- rather than half-written or
    silently skipped. Asking only "is there free space right now" would
    let a 4.7 GB dump onto 5.1 GB of disk and end the session with none.
  * INTEGRITY. The package's dart frame came from the same array that is
    in the ring, so it must be BYTE-IDENTICAL to one frame in the slice.
    That is asserted after every misscore dump that names a package (see
    `frame_dump.verify_package_frames_in_dump()`), and the result is
    written back into the dump directory. If it ever differs, something
    re-encoded or mutated a frame between capture and save -- which would
    mean a misscore dump is a record of something other than what was
    scored, and that is worth knowing about immediately rather than
    discovering during an investigation that depends on it.

EXPLICITLY OUT OF SCOPE HERE: overlapping captures (one at a time, and a
second trigger while a write is in flight is refused with a reason --
enforced in `FrameDumpWriter.submit()`), and any retention policy for
accumulated dumps. The second is worth deciding before a session fills a
disk; `sweep_old_dumps()` does not exist on purpose, so that decision is
made deliberately rather than inherited from a default nobody chose.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from opendarts.capture.frame_dump import (
    KIND_MISSCORE,
    KIND_MISSED_DART,
    DumpProgress,
    FrameDumpWriter,
    verify_package_frames_in_dump,
)
from opendarts.capture import clip
from opendarts.capture.frame_ring import FrameRing
from opendarts.disk_space import check_free_space

log = logging.getLogger("opendarts.capture.throw_capture")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from opendarts.paths import DATA_DIR

#: Where dumps land. Beside `data/packages`, not inside it: a dump is
#: evidence ABOUT a throw, and `discover_packages()` walks
#: `<root>/<session>/<throw>/meta.json` -- a dump directory under that
#: tree would either be skipped silently or, worse, half-parsed.
DEFAULT_CAPTURE_ROOT = DATA_DIR / "captures"

#: The misscore window. See this module's docstring for why it is
#: asymmetric, why it is generous, and why it must not be tightened
#: without real captures to tighten it against.
MISSCORE_WINDOW_BEFORE_S = 0.5
MISSCORE_WINDOW_AFTER_S = 0.2

#: The per-throw CLIP SEARCH window, sliced from the ring around a throw's
#: recorded time. This is only the in-memory window that must CONTAIN the
#: commit frame plus enough on either side for clip.py to anchor on it --
#: it is NOT what ends up on disk. clip.finalize_throw_clip trims the
#: written clip to a per-throw frame COUNT around the commit (before =
#: settle -> frames + lead-in, capped; after = 1), so widening this window
#: costs nothing on disk; it only makes the before/after runs reliably
#: available. `before` must comfortably hold
#: clip.CLIP_MAX_FRAMES_BEFORE_COMMIT frames even at a slower capture rate
#: (0.6s ~= 15-18 frames), since a slow settle can reach that cap. Kept
#: before-heavy (like the misscore window) because the package timestamp
#: lands after the landing, so the interesting frames are on the `before`
#: side.
CLIP_WINDOW_BEFORE_S = 0.6
CLIP_WINDOW_AFTER_S = 0.2
#: How long the "all" trigger waits after a commit before slicing, so the
#: after-window has actually landed in the ring first.
_CLIP_ALL_DEFER_S = CLIP_WINDOW_AFTER_S + 0.3

INTEGRITY_FILENAME = "integrity.json"

#: How many captures the dashboard's list carries. The DUMPS are the
#: record; this list is only what a screen shows without walking the whole
#: root, so it is bounded on purpose -- a rig left running for a month
#: must not turn a status poll into a thousand-directory walk. The full
#: count and the full byte total come from `measure_captures()`, which is
#: called when someone is about to delete, not on a poll.
CAPTURE_LIST_LIMIT = 50


def _parse_capture_dir_name(name: str) -> "tuple[str | None, str | None]":
    """(kind, at_utc) read out of a dump directory's own name, or (None,
    None) for a directory this module did not write.

    `_dest_dir()` names every dump `<%Y%m%dT%H%M%S>-<micros>-<kind>`, and
    `kind` is the only one of the three that contains an underscore rather
    than a hyphen, so a three-way split is exact. Parsing the NAME rather
    than opening each `manifest.json` is what keeps listing a root cheap
    enough to do on a status poll.

    Tolerant by design: a directory that does not match is still a real
    directory holding real bytes, and reporting it as an unknown capture
    is more honest than pretending it is not there.
    """
    parts = name.split("-")
    if len(parts) != 3:
        return None, None
    stamp, micros, kind = parts
    if kind not in (KIND_MISSED_DART, KIND_MISSCORE):
        kind = None
    try:
        when = datetime.strptime(stamp, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return kind, None
    if len(micros) == 6 and micros.isdigit():
        when = when.replace(microsecond=int(micros))
    return kind, when.isoformat()


def _capture_dir_record(path: Path) -> dict[str, Any]:
    """One on-disk capture, sized by one pass over its own files.

    Sizes come from the `scandir` entries themselves, so the file list and
    the byte total cost one walk rather than two, and `integrity` is
    reported as a FILE PRESENT rather than parsed -- whether the check ran
    is the question a list answers; what it said is in the dump.
    """
    kind, at_utc = _parse_capture_dir_name(path.name)
    nbytes = 0
    n_files = 0
    has_integrity = False
    try:
        for entry in os.scandir(path):
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                nbytes += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            n_files += 1
            if entry.name == INTEGRITY_FILENAME:
                has_integrity = True
    except OSError:
        pass
    return {
        "name": path.name,
        "dest": str(path),
        "kind": kind,
        "at_utc": at_utc,
        "bytes": nbytes,
        "n_files": n_files,
        "on_disk": True,
        "integrity_checked": has_integrity,
        "integrity": None,
    }


def list_captures_on_disk(
    capture_root: "Path | str", *, limit: "int | None" = CAPTURE_LIST_LIMIT,
) -> list[dict[str, Any]]:
    """The captures that are REALLY there, oldest first.

    The dashboard used to list this process's own in-memory record of what
    it had written, which empties on restart while every file stays on
    disk -- a list that says "no captures" beside 4 GB of captures. Reading
    the root is the only listing that cannot drift from the thing it
    describes.

    `limit` keeps the newest N (see `CAPTURE_LIST_LIMIT`); None lists all.
    """
    root = Path(capture_root)
    try:
        entries = sorted(
            (e for e in os.scandir(root) if e.is_dir(follow_symlinks=False)),
            key=lambda e: e.name,
        )
    except OSError:
        return []
    if limit is not None and limit >= 0:
        entries = entries[-limit:]
    return [_capture_dir_record(Path(e.path)) for e in entries]


def measure_captures(capture_root: "Path | str") -> dict[str, Any]:
    """How many captures are on this rig and what they weigh, in full.

    Walks every file under the root, which is exactly why it is NOT on the
    status poll: it answers the one question a confirmation dialog has to
    ask before it deletes anything ("3 captures (420 MB)"), and it is
    called at that moment.

    `count` is directories, because a capture is a directory. Loose files
    directly under the root are nobody's capture but they are still bytes
    that a delete will take, so they are in `bytes` and not in `count`.
    """
    root = Path(capture_root)
    count = 0
    nbytes = 0
    try:
        entries = list(os.scandir(root))
    except OSError:
        return {"root": str(root), "count": 0, "bytes": 0}
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                count += 1
                nbytes += dir_size_bytes(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):
                nbytes += entry.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return {"root": str(root), "count": count, "bytes": nbytes}


def dir_size_bytes(path: "Path | str") -> int:
    """Total size of the regular files under `path`, skipping what will
    not answer.

    Skipping rather than raising: this number exists to be shown to a
    person before they delete something, and one unreadable file is not a
    reason to show them nothing. Symlinks are never followed -- a link out
    of the tree is not bytes this delete would free.
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, onerror=None, followlinks=False):
        for name in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                continue
    return total


def _parse_utc(value: Any) -> "float | None":
    """An ISO-8601 UTC string from `meta.json` as epoch seconds, or None.

    Returns None rather than raising or guessing: a package with an
    unparseable timestamp is a package this feature cannot anchor on, and
    the caller must hear that as a refusal rather than as a capture
    centred on 1970.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    # `datetime.fromisoformat` on this venv's Python does not accept a
    # trailing "Z"; every timestamp this project writes comes from
    # `datetime.now(timezone.utc).isoformat()` and carries "+00:00"
    # instead, but a hand-edited or externally-produced package is a real
    # possibility and this costs one line.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


@dataclass
class ThrowAnchor:
    """A wall-clock instant to centre a misscore window on, and an honest
    account of how it was arrived at."""

    wall_s: "float | None"
    #: "capture_instant" when handle_total_s was available and subtracted;
    #: "package_timestamp" when it was not (the anchor is then later than
    #: the real landing by however long scoring took, which the generous
    #: `before` window is sized to absorb); None when no anchor exists.
    basis: "str | None" = None
    detail: "str | None" = None


def anchor_wall_s_for_package(package_dir: Path) -> ThrowAnchor:
    """When the frames of this throw were actually in hand.

    `meta.captured_at_utc` is stamped at the END of
    `handle_ready_to_capture()`'s synchronous path, after the engine has
    scored -- the capture daemon documents this itself and points at the
    recovery: "captured_at_utc - handle_total_s is when the frames were
    actually in hand". `handle_total_s` lives in the package's own
    `capture_diagnostics.json` (schema_version 2 onward).

    So: subtract it when it is there, and SAY which basis was used when it
    is not. A package written before that field existed still anchors
    fine -- the window's generous `before` side is exactly what absorbs
    the unknown offset -- but a caller reading "package_timestamp" knows
    the centre of its window is late by however long scoring took, and can
    read a clipped capture as a known cause rather than a mystery.
    """
    package_dir = Path(package_dir)
    meta_path = package_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return ThrowAnchor(None, None, f"cannot read {meta_path}: {exc}")

    stamped = _parse_utc(meta.get("captured_at_utc"))
    if stamped is None:
        return ThrowAnchor(
            None, None,
            f"{meta_path} has no usable captured_at_utc "
            f"({meta.get('captured_at_utc')!r}) -- nothing to anchor a window on",
        )

    handle_total_s: "float | None" = None
    try:
        diagnostics = json.loads((package_dir / "capture_diagnostics.json").read_text())
        raw = (diagnostics.get("timings") or {}).get("handle_total_s")
        handle_total_s = float(raw) if isinstance(raw, (int, float)) else None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
        handle_total_s = None

    if handle_total_s is not None and handle_total_s >= 0:
        return ThrowAnchor(
            stamped - handle_total_s,
            "capture_instant",
            f"captured_at_utc minus handle_total_s ({handle_total_s:.3f}s), which is "
            "when the frames were really in hand -- the timestamp itself is stamped "
            "after the engine has scored",
        )
    return ThrowAnchor(
        stamped,
        "package_timestamp",
        "this package records no handle_total_s, so the anchor is the raw "
        "captured_at_utc -- which is LATER than the landing by however long "
        "scoring took. The window's -"
        f"{MISSCORE_WINDOW_BEFORE_S:.1f}s side is what absorbs that.",
    )


class ThrowCaptureService:
    """The ring, the writer, and the two triggers, in one object a route or
    the capture loop can hold.

    One per process, built beside the hub and given the hub's own ring --
    there is exactly one ring per hub, and a service holding a second one
    would be buffering nothing while reporting a size.
    """

    def __init__(
        self,
        ring: "FrameRing | None",
        *,
        capture_root: Path = DEFAULT_CAPTURE_ROOT,
        writer: "FrameDumpWriter | None" = None,
        min_free_disk_gb: "float | None" = None,
        free_bytes_fn: "Callable[[Path], int | None] | None" = None,
        record_mode: str = "mismatch",
    ) -> None:
        self.ring = ring
        self.capture_root = Path(capture_root)
        self.writer = writer or FrameDumpWriter()
        #: The per-throw video-record mode (opendarts.live.config
        #: video_record_mode): "never" | "mismatch" | "all". Decides which
        #: trigger records a per-throw clip INTO the package (record_throw_
        #: clip); the whole-ring missed-dart / misscore dumps are unchanged.
        self.record_mode = record_mode
        #: The RAW configured floor, not a resolved one -- None means
        #: "use the default", 0 means the same, and a negative value
        #: disables the guard. `opendarts.disk_space.resolve_floor_gb()`
        #: is the one place that is decided; see
        #: `opendarts.live.config.min_free_disk_gb()` for the config key.
        self.min_free_disk_gb = min_free_disk_gb
        #: How this service reads free space. Left None it uses
        #: `opendarts.disk_space.free_bytes`, looked up at call time; a
        #: test hands its own in so the floor is provable without
        #: filling a real disk.
        self.free_bytes_fn = free_bytes_fn
        self._lock = threading.Lock()
        self._captures: list[dict[str, Any]] = []

    # -- the two triggers -----------------------------------------------

    def capture_missed_dart(
        self,
        *,
        reason: str,
        source: str = "manual",
        extra: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
        """Write the WHOLE ring, pausing it for the duration.

        `source` records who pulled the trigger -- "manual" for the
        dashboard button, "oracle" when the oracle committed a throw this
        rig did not. On a stream-fed consumer the oracle case is the most
        valuable signal available, because it is exactly the
        cross-platform difference the test rigs exist to find and it needs
        nobody present; recording which one fired is what lets a session's
        dumps be read afterwards without guessing.
        """
        refusal = self._ring_refusal()
        if refusal is not None:
            return refusal

        ring = self.ring
        assert ring is not None  # _ring_refusal() has already proven this
        # PAUSE FIRST, SNAPSHOT SECOND. The other order leaves a window in
        # which arrivals join the snapshot and then also survive the
        # pause, so the buffer being written is not the buffer that was
        # frozen -- a small discrepancy that would be invisible and would
        # make the recorded span disagree with the manifest.
        ring.pause()
        ring_slice = ring.snapshot()
        if not ring_slice.sets:
            ring.resume()
            message = (
                "the frame ring holds nothing to write -- it is enabled but empty. "
                "Either the capture loop has not started, or the pump has produced "
                "no frames since it was cleared."
            )
            log.warning("missed-dart capture REFUSED: %s", message)
            return {"ok": False, "reason": message, "ring": ring.stats()}

        disk_refusal = self._disk_refusal(ring_slice.nbytes, kind=KIND_MISSED_DART)
        if disk_refusal is not None:
            # Resume first: this pause has no writer thread to un-pause
            # it, and a rig that stopped recording because it could not
            # write one dump would have turned a disk problem into a
            # capture problem.
            ring.resume()
            disk_refusal["ring"] = ring.stats()
            return disk_refusal

        dest = self._dest_dir(KIND_MISSED_DART)
        log.info(
            "missed-dart capture (%s): %d set(s), %.1fs, %.0f MB -> %s. Reason: %s",
            source, len(ring_slice.sets), ring_slice.span_s,
            ring_slice.nbytes / 1e6, dest, reason,
        )
        submitted = self.writer.submit(
            ring_slice, dest,
            kind=KIND_MISSED_DART,
            reason=ring_slice.reason,
            extra={"trigger": source, "operator_reason": reason, **(extra or {})},
            # The ring is resumed in the writer's own `finally`, so a
            # writer thread that dies cannot leave the tap off for the
            # rest of the session.
            pause_ring=ring,
        )
        if not submitted.get("ok"):
            # Refused by the one-at-a-time rule. Resume immediately: this
            # pause has no writer to un-pause it.
            ring.resume()
            return submitted
        self._record(submitted["job"], kind=KIND_MISSED_DART, source=source, dest=dest)
        return submitted

    def capture_misscore(
        self,
        anchor_wall_s: "float | None",
        *,
        reason: str,
        source: str = "manual",
        package_dir: "Path | None" = None,
        before_s: float = MISSCORE_WINDOW_BEFORE_S,
        after_s: float = MISSCORE_WINDOW_AFTER_S,
        extra: "dict[str, Any] | None" = None,
    ) -> dict[str, Any]:
        """Write ~1s around a recorded throw, leaving the ring running.

        `anchor_wall_s` is WALL clock, because the only timestamp a
        package carries is wall clock. Pass None and a `package_dir` and
        the anchor is resolved from the package -- see
        `anchor_wall_s_for_package()` for why that is not simply
        `captured_at_utc`.
        """
        refusal = self._ring_refusal()
        if refusal is not None:
            return refusal
        ring = self.ring
        assert ring is not None

        anchor_detail = None
        anchor_basis = None
        if anchor_wall_s is None and package_dir is not None:
            anchor = anchor_wall_s_for_package(package_dir)
            anchor_wall_s, anchor_basis, anchor_detail = (
                anchor.wall_s, anchor.basis, anchor.detail,
            )
        if anchor_wall_s is None:
            message = (
                "no anchor: a misscore capture is centred on the throw's own "
                "recorded time, and none could be determined"
                + (f" -- {anchor_detail}" if anchor_detail else "")
            )
            log.warning("misscore capture REFUSED: %s", message)
            return {"ok": False, "reason": message, "ring": ring.stats()}

        ring_slice = ring.slice_around(anchor_wall_s, before_s=before_s, after_s=after_s)
        if ring_slice.aged_out or not ring_slice.sets:
            # AGED OUT, WITH THE NUMBERS. Never an empty file, never a
            # silent success. The reason comes from the ring itself, which
            # is the only thing that knows how far back it really reaches.
            message = ring_slice.reason or (
                "the ring holds no frames from that moment and could not say why"
            )
            return {
                "ok": False,
                "reason": message,
                "aged_out": ring_slice.aged_out,
                "anchor_wall_s": anchor_wall_s,
                "anchor_basis": anchor_basis,
                "ring": ring.stats(),
            }

        disk_refusal = self._disk_refusal(ring_slice.nbytes, kind=KIND_MISSCORE)
        if disk_refusal is not None:
            disk_refusal["ring"] = ring.stats()
            disk_refusal["anchor_wall_s"] = anchor_wall_s
            disk_refusal["anchor_basis"] = anchor_basis
            return disk_refusal

        dest = self._dest_dir(KIND_MISSCORE)
        log.info(
            "misscore capture (%s): %d set(s) in [-%.2fs, +%.2fs] around %.3f, "
            "%.0f MB -> %s. Reason: %s",
            source, len(ring_slice.sets), before_s, after_s, anchor_wall_s,
            ring_slice.nbytes / 1e6, dest, reason,
        )
        submitted = self.writer.submit(
            ring_slice, dest,
            kind=KIND_MISSCORE,
            reason=ring_slice.reason,
            extra={
                "trigger": source,
                "operator_reason": reason,
                "anchor_basis": anchor_basis,
                "anchor_detail": anchor_detail,
                "package_dir": str(package_dir) if package_dir else None,
                **(extra or {}),
            },
            # NOT paused. ~270MB is a fraction of a second of writing and
            # the rig must keep recording -- a misscore is usually the
            # first of several, and stopping the ring to investigate one
            # would lose the next.
            pause_ring=None,
            on_done=(
                (lambda progress: self._verify(progress, Path(package_dir), dest))
                if package_dir is not None else None
            ),
        )
        if submitted.get("ok"):
            self._record(submitted["job"], kind=KIND_MISSCORE, source=source, dest=dest)
        return submitted

    # -- the per-throw clip recorder (video_record_mode) ----------------

    def record_throw_clip(
        self,
        package_dir: Path,
        anchor_wall_s: "float | None" = None,
        *,
        reason: str,
        source: str,
    ) -> "dict[str, Any]":
        """Slice the window around a throw and UPGRADE its package's clips
        to a recording (opendarts.capture.clip.finalize_throw_clip): the
        two-frame bg+commit clip save_throw_package() wrote is replaced by
        the bg..commit+1 run out of the ring. This is the record path for
        video_record_mode; the ring keeps running.

        Never raises, and a failure costs the package nothing: it is
        already complete on disk (its stills clip holds both scored
        frames), and the upgrade only swaps the clips in once every camera
        has a verified replacement -- through an atomic meta.json rewrite,
        so even a crash mid-upgrade leaves the stills clip in charge. A
        failure is logged and returned; it never propagates into the
        capture path."""
        refusal = self._ring_refusal()
        if refusal is not None:
            return refusal
        ring = self.ring
        assert ring is not None
        try:
            if anchor_wall_s is None:
                anchor_wall_s = anchor_wall_s_for_package(package_dir).wall_s
            if anchor_wall_s is None:
                return {"ok": False, "reason": "no anchor to centre the clip on"}
            ring_slice = ring.slice_around(
                anchor_wall_s, before_s=CLIP_WINDOW_BEFORE_S, after_s=CLIP_WINDOW_AFTER_S,
            )
            if ring_slice.aged_out or not ring_slice.sets:
                return {
                    "ok": False,
                    "reason": ring_slice.reason or "the ring held no frames there",
                    "aged_out": ring_slice.aged_out,
                    "ring": ring.stats(),
                }
            out = clip.finalize_throw_clip(package_dir, ring_slice.sets)
            if out.get("ok"):
                log.info(
                    "recorded throw clip (%s) for %s: %d set(s), cams %s. %s",
                    source, package_dir.name, len(ring_slice.sets),
                    out.get("cameras"), reason,
                )
            else:
                log.warning(
                    "throw clip NOT recorded for %s: %s",
                    package_dir.name, out.get("reason"),
                )
            return out
        except Exception as exc:  # noqa: BLE001 -- recording must never break capture
            log.exception("throw clip recording failed for %s", package_dir)
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    def schedule_throw_clip(
        self,
        package_dir: Path,
        anchor_wall_s: "float | None" = None,
        *,
        reason: str,
        source: str,
    ) -> None:
        """`record_throw_clip` on a daemon thread after a short defer, so
        the after-window has actually landed in the ring first. Used by the
        "all" trigger, which fires at commit -- before the post-commit
        frames exist. Fire-and-forget: throws are seconds apart, so at most
        a couple ever run at once."""
        def _run() -> None:
            time.sleep(_CLIP_ALL_DEFER_S)
            self.record_throw_clip(package_dir, anchor_wall_s, reason=reason, source=source)

        threading.Thread(target=_run, name="throw-clip", daemon=True).start()

    # -- internals ------------------------------------------------------

    def _ring_refusal(self) -> "dict[str, Any] | None":
        """The one place "there is no ring" is worded, so the two triggers
        cannot describe the same condition two different ways."""
        if self.ring is None:
            message = (
                "no frame ring is attached to this process -- nothing has been "
                "buffered, so there is nothing to capture. Set frame_ring_seconds "
                "in data/config.json and restart the capture loop."
            )
            log.warning("frame capture REFUSED: %s", message)
            return {"ok": False, "reason": message}
        if not self.ring.enabled:
            message = (
                f"the frame ring is disabled (frame_ring_seconds={self.ring.seconds:g}) "
                "-- no frames are being retained, so there is nothing to capture."
            )
            log.warning("frame capture REFUSED: %s", message)
            return {"ok": False, "reason": message, "ring": self.ring.stats()}
        return None

    def _disk_refusal(self, nbytes: int, *, kind: str) -> "dict[str, Any] | None":
        """Refuse a dump that would take the disk below the floor.

        ONE wording for both triggers, for the same reason `_ring_refusal()`
        above has one: two descriptions of one condition eventually
        disagree, and an operator reading a refusal should not have to
        work out whether the missed-dart path and the misscore path mean
        the same thing by it.

        `nbytes` is the slice's own size -- an ESTIMATE of the dump, and
        a close one (the writer stores the same frame bytes it was handed
        plus a small manifest), which is why it is checked rather than
        only asking whether there is space at this instant.

        Returns None when the dump may proceed, so the call site reads as
        a guard clause rather than a branch.
        """
        check = check_free_space(
            self.capture_root,
            floor_gb=self.min_free_disk_gb,
            needed_bytes=nbytes,
            free_bytes_fn=self.free_bytes_fn,
        )
        if check.ok:
            return None
        message = (
            f"not enough free disk to write this capture: {check.reason}. "
            "Nothing was written. Free space on this rig, copy the existing "
            "captures off it, or lower min_free_disk_gb in data/config.json."
        )
        log.warning("%s capture REFUSED: %s", kind, message)
        return {
            "ok": False,
            "reason": message,
            "disk": check.as_dict(),
            "estimated_bytes": int(nbytes),
        }

    def _dest_dir(self, kind: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        # Microseconds in the suffix, not just seconds: two automatic
        # triggers a few hundred ms apart is an ordinary thing for an
        # oracle to do, and two dumps sharing a directory would interleave
        # their frames.bin.
        micros = datetime.now(timezone.utc).strftime("%f")
        return self.capture_root / f"{stamp}-{micros}-{kind}"

    def _verify(self, progress: DumpProgress, package_dir: Path, dest: Path) -> None:
        """Run the integrity check once the dump is on disk, and WRITE THE
        ANSWER DOWN beside it.

        On the writer's own thread, via its `on_done` hook, so a check
        that decodes several PNGs never delays the trigger's return. A
        failed write is not checked -- there is nothing to check against,
        and reporting "integrity failed" for a dump that does not exist
        would point at the wrong problem.
        """
        if progress.state != "done":
            return
        result = verify_package_frames_in_dump(package_dir, dest)
        payload = {
            **result.as_dict(),
            "package_dir": str(package_dir),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "what_this_proves": (
                "The dart frame this package stored as lossless PNG is the same "
                "array the ring retained. If this is false, something re-encoded or "
                "mutated a frame between capture and save, and this dump is not a "
                "faithful record of what was scored."
            ),
        }
        try:
            (dest / INTEGRITY_FILENAME).write_text(json.dumps(payload, indent=2))
        except OSError:
            log.exception("could not write %s", dest / INTEGRITY_FILENAME)
        with self._lock:
            for record in self._captures:
                if record.get("dest") == str(dest):
                    record["integrity"] = result.as_dict()

    def _record(self, job: dict[str, Any], *, kind: str, source: str, dest: Path) -> None:
        with self._lock:
            self._captures.append(
                {
                    "job_id": job.get("job_id"),
                    "kind": kind,
                    "source": source,
                    "dest": str(dest),
                    "at_utc": datetime.now(timezone.utc).isoformat(),
                    "integrity": None,
                }
            )
            # Bounded, because this is a session-long list on a process
            # that runs for days. The dumps themselves are the record; this
            # is only what the dashboard shows without walking the disk.
            del self._captures[:-50]

    # -- what the dashboard reads ---------------------------------------

    def _captures_for_display(
        self, remembered: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """The captures ON DISK, wearing whatever this session happens to
        remember about them.

        THE DISK IS THE LIST. Until 2026-09-17 this was `self._captures`
        alone -- what THIS process had written -- so a restart emptied the
        list while every file stayed exactly where it was, and the panel
        said "no captures" beside gigabytes of them. A list of evidence
        that disagrees with the evidence is worse than no list.

        The in-memory record is still merged in, because it knows three
        things the directory name cannot: which trigger fired
        (`source`), the writer's job id, and what the integrity check
        actually said. A remembered capture whose directory is NOT on disk
        is kept and flagged `on_disk: False` rather than dropped -- it was
        written and then something removed it, and silently agreeing with
        the removal would hide exactly the case worth seeing.
        """
        on_disk = list_captures_on_disk(self.capture_root)
        by_dest = {record["dest"]: record for record in on_disk}
        missing: list[dict[str, Any]] = []
        for record in remembered:
            dest = str(record.get("dest") or "")
            match = by_dest.get(dest)
            if match is None:
                missing.append({**record, "on_disk": False, "bytes": None})
                continue
            match.update({k: v for k, v in record.items() if v is not None})
        return sorted(
            on_disk + missing,
            key=lambda r: (r.get("at_utc") or "", r.get("dest") or ""),
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            remembered = list(self._captures)
        captures = self._captures_for_display(remembered)
        return {
            "ring": self.ring.stats() if self.ring is not None else {"enabled": False},
            "writer": self.writer.status(),
            "capture_root": str(self.capture_root),
            "window": {
                "misscore_before_s": MISSCORE_WINDOW_BEFORE_S,
                "misscore_after_s": MISSCORE_WINDOW_AFTER_S,
            },
            "captures": captures,
        }
