"""opendarts/capture/frame_dump.py -- writing a slice of the frame ring to
disk, reading it back byte-identically, and turning it into something a
human can watch.

THE DUMP HOLDS EACH FRAME AS THE RING DOES -- NO ENCODE, EVER.
PNG-encoding a full buffer would be ~2000 frames x 49.5ms/frame ~ 100
seconds of CPU before the file existed, during which the rig is scoring
darts. So each frame is written in the form the ring already has it:

  jpeg  the camera's own JPEG, for a slot in passthrough (see
        local_capture's JPEG PASSTHROUGH section). ~36x smaller than
        pixels: a Windows rig's 5-second, three-camera ring was 34.6 MB as JPEG and
        1.24 GB as pixels (2026-09-17). Nothing is lost: the rig scored a
        decode of these very bytes, and `read_frame()` decodes them again.
  raw   the pixel buffer, for a slot with no camera JPEG (a Mac, or a
        Windows camera on the DirectShow fallback). A straight write of
        memory that already exists: measured on an NVMe rig at
        3.3 GB/s.

Either way it is a sequential write, whose speed depends on the disk -- a
Mac on SATA or a VM on shared storage is several times slower -- which
is why the writer reports bytes written against bytes total rather than
leaving a button that looks hung.

ON-DISK SHAPE, and why it is one big file rather than a file per frame:

    <dump_dir>/manifest.json     -- every set, both clocks, and the byte
                                    range of each frame inside frames.bin
    <dump_dir>/frames.bin        -- the pixels, concatenated, nothing else

A 22-second three-camera dump is ~2000 sets and ~6000 frames. Six
thousand small files is six thousand directory entries, six thousand
opens on read-back, and a filesystem doing metadata work at exactly the
moment the rig needs the disk for a throw package. One append-only
stream plus an index is the same bytes with none of that, and the index
is what makes a read-back a seek rather than a scan.

BYTE-IDENTICAL IS THE ENTIRE PREMISE. A dump exists so detection,
lifecycle and settling can be re-run on exactly the pixels that were
scored. `frames.bin` holds each frame's bytes verbatim -- the array's own
buffer, or the JPEG those pixels were decoded from -- and the manifest
records its encoding, shape and dtype, so `read_frame()` reconstructs the
identical array. There is no encode step anywhere on this path to be
lossy in. `tests/test_frame_ring_roundtrip.py` asserts that against real
frames rather than assuming it.

WHY A WRITER THREAD. Python releases the GIL for the duration of a file
write, so the capture loop goes on scoring throughout. The trigger itself
never writes anything: it takes a snapshot of REFERENCES (instant -- a
list copy, never a pixel) and hands it to this writer, so a button can
say "saving" immediately instead of hanging for the length of a 6GB
write.

ONE AT A TIME, AND IT SAYS SO. A second trigger while a write is in
flight is refused with a reason naming the job already running -- see
`FrameDumpWriter.submit()`. Overlapping captures are explicitly out of
scope: two 6GB writes at once on a rig with 14.4GB free is how this
feature would take the product down.

RESUMING THE RING IS IN A `finally`. A missed-dart capture pauses the
ring so peak memory stays flat across the write. A writer thread that
dies with the ring still paused would leave the tap off for the rest of
the session with every other surface reporting health -- the exact "no
errors on the rig" failure docs/DESIGN.md is written about. The resume is
in a `finally`, and `tests/test_frame_dump.py` proves it by making the
write raise.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from opendarts.capture.frame_ring import FrameRing, RingSlice, format_bytes
from opendarts.live import jpeg_info

log = logging.getLogger("opendarts.capture.frame_dump")

#: Bumped only if the on-disk shape changes incompatibly. Written on every
#: dump and CHECKED on every read: a reader that silently accepts a shape
#: it does not understand is how a replay ends up scoring garbage and
#: blaming the pipeline.
DUMP_SCHEMA = "frame-dump/v2"
#: v1 dumps (2026-09-16, pixels only) have the same layout without the
#: per-frame `encoding`, which reads as "raw".
_READABLE_SCHEMAS = ("frame-dump/v1", DUMP_SCHEMA)

MANIFEST_FILENAME = "manifest.json"
FRAMES_FILENAME = "frames.bin"

#: What a dump is FOR, recorded in the manifest. Two values, because there
#: are exactly two questions the ring exists to answer and they want
#: different windows -- see `opendarts/capture/throw_capture.py`.
KIND_MISSED_DART = "missed_dart"
KIND_MISSCORE = "misscore"

#: How often the writer refreshes its byte counter. Once per frame set
#: rather than once per frame: a set is ~8MB, so this is ~30 updates a
#: second of real progress, and a per-frame update would be three times
#: the lock traffic to tell the operator the same thing.
_PROGRESS_EVERY_SETS = 1


@dataclass
class DumpProgress:
    """What a dump is doing right now, cheap enough for the dashboard to
    poll while a 6GB write is in flight.

    `bytes_written`/`bytes_total` rather than a percentage because the
    total is the number that tells an operator whether a slow write is a
    slow disk or a huge capture, and a percentage throws that away."""

    job_id: str
    kind: str
    dest_dir: str
    state: str = "queued"            # queued | writing | done | failed
    bytes_total: int = 0
    bytes_written: int = 0
    sets_total: int = 0
    sets_written: int = 0
    started_wall_s: "float | None" = None
    finished_wall_s: "float | None" = None
    duration_s: "float | None" = None
    error: "str | None" = None
    #: Carried from the RingSlice: why the capture is partial, or why it
    #: was refused. Never dropped on the way to the UI -- a capture that
    #: silently lost its first half is the failure this field exists for.
    reason: "str | None" = None
    ring_paused_for_write: bool = False

    def mb_or_unknown(self) -> str:
        if self.bytes_total:
            return f"{self.bytes_written / 1e6:.0f} of {self.bytes_total / 1e6:.0f} MB"
        return "size not yet known"

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["mb_written"] = round(self.bytes_written / 1e6, 1)
        d["mb_total"] = round(self.bytes_total / 1e6, 1)
        # A real rate, from this write, on this disk -- not a reference rig's
        # 3.3 GB/s repeated as if it were a property of the software.
        if self.duration_s and self.duration_s > 0:
            d["bytes_per_s"] = round(self.bytes_written / self.duration_s)
        elif self.state == "writing" and self.started_wall_s:
            elapsed = time.time() - self.started_wall_s
            d["bytes_per_s"] = round(self.bytes_written / elapsed) if elapsed > 0 else None
        else:
            d["bytes_per_s"] = None
        return d


def _manifest_for(
    ring_slice: RingSlice,
    *,
    kind: str,
    reason: "str | None",
    extra: "dict[str, Any] | None",
) -> tuple[dict[str, Any], list[list[np.ndarray]]]:
    """Build the index and the write plan in one pass, so a frame's
    recorded byte range and the order it is actually written in cannot
    drift apart. Returns (manifest, [[array, ...] per SET]) -- grouped by
    set so the writer advances its progress counter on real set
    boundaries rather than re-deriving them.

    THE SAME ARRAY CAN APPEAR IN SEVERAL SETS, and the plan writes its
    bytes ONCE. When a slot's read fails, `_pump_once()` deliberately does
    not null the cache -- "a momentary hiccup does not null out an
    otherwise-good previous frame" -- so the next cycle publishes the
    SAME array object again. That repetition is real and the manifest
    keeps it (each set records what the pump actually served that cycle),
    but storing identical megabytes twice is not: the second entry simply
    points at the first one's offset. On a rig with one stalled camera
    this is the difference between a dump that fits and one that does not,
    and on a healthy rig it never fires at all.

    Identity, not content, is the key. Two genuinely different frames that
    happen to be pixel-identical are still two frames, and comparing 2.7MB
    of pixels per frame to find out would cost more than the bytes saved.
    Every array is alive for the whole call (the slice holds a reference),
    so `id()` is stable here -- which it would NOT be if the plan outlived
    the slice."""
    plan: list[list[np.ndarray]] = []
    sets_out: list[dict[str, Any]] = []
    n_frames = 0
    offset = 0
    slots: set[int] = set()
    offsets_by_id: dict[int, dict[str, Any]] = {}
    for fs in ring_slice.sets:
        frames_out: list[dict[str, Any]] = []
        set_plan: list[Any] = []
        for slot in fs.slots:
            # A JPEG slot (see FrameSet) is written AS ITS JPEG. Its shape
            # comes from the frame header -- what read_frame() decodes it
            # to, always 3-channel uint8. Identity is the bytes object's,
            # which a stalled camera re-serves unchanged, so repeats are
            # still recognised.
            arr: Any = fs.jpegs.get(slot)
            if arr is not None:
                dims = jpeg_info.dimensions(arr)
                if dims is None:
                    raise ValueError(f"slot {slot}: retained JPEG has no frame header")
                shape = [int(dims[0]), int(dims[1]), 3]
                encoding, nbytes, dtype = "jpeg", len(arr), "uint8"
            else:
                arr = fs.pixels[slot]
                shape = [int(x) for x in arr.shape]
                encoding, nbytes, dtype = "raw", int(arr.nbytes), str(arr.dtype)
            seen = offsets_by_id.get(id(arr))
            entry = {
                "slot": int(slot),
                "offset": seen["offset"] if seen is not None else offset,
                "nbytes": nbytes,
                "encoding": encoding,
                "shape": shape,
                "dtype": dtype,
                # True when this entry points at bytes an earlier set
                # already wrote -- so a reader (and a human reading the
                # manifest) can tell "the camera stalled and this cycle
                # re-served the previous frame" from "the camera produced
                # a new frame that happens to look the same".
                "repeat_of_earlier_set": seen is not None,
            }
            frames_out.append(entry)
            n_frames += 1
            if seen is None:
                offsets_by_id[id(arr)] = entry
                set_plan.append(arr)
                offset += nbytes
            slots.add(int(slot))
        plan.append(set_plan)
        sets_out.append(
            {
                "generation": fs.generation,
                # BOTH CLOCKS, ALWAYS. wall matches a throw package's
                # captured_at_utc; monotonic orders frames and measures
                # durations. Stored side by side and never subtracted
                # across each other -- see frame_ring.py's docstring for
                # the 2026-09-15 incident that rule comes from.
                "wall_s": fs.wall_s,
                "monotonic_s": fs.monotonic_s,
                "frames": frames_out,
            }
        )

    manifest = {
        "schema": DUMP_SCHEMA,
        "kind": kind,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        # Empty collections are [], never null or absent -- docs/DESIGN.md.
        "slots": sorted(slots),
        "sets": sets_out,
        "bytes_total": offset,
        "n_sets": len(sets_out),
        "n_frames": n_frames,
        "span_s": round(ring_slice.span_s, 3),
        "anchor_wall_s": ring_slice.anchor_wall_s,
        "window_before_s": ring_slice.window_before_s,
        "window_after_s": ring_slice.window_after_s,
        # The refusal/partial explanation travels WITH the data. A dump
        # whose first half was clipped must carry that fact on disk, not
        # only in a log line on a rig nobody is watching.
        "reason": reason,
        "extra": dict(extra or {}),
    }
    return manifest, plan


def write_dump(
    ring_slice: RingSlice,
    dest_dir: Path,
    *,
    kind: str,
    reason: "str | None" = None,
    extra: "dict[str, Any] | None" = None,
    progress: "DumpProgress | None" = None,
) -> Path:
    """Write one slice, synchronously. Raises rather than writing a
    partial dump that LOOKS complete -- the same posture
    `save_throw_package()` takes, and for the same reason: a truncated
    dump would silently corrupt a later replay.

    REFUSES AN EMPTY SLICE. Never writes an empty file and never no-ops
    while looking successful: a slice with no sets means the ring did not
    hold that moment, and the caller must hear that as a failure with a
    reason, not as a dump directory full of nothing.
    """
    if not ring_slice.sets:
        raise ValueError(
            "refusing to write an empty frame dump -- "
            + (reason or "the requested window matched no retained frame set")
        )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest, plan = _manifest_for(ring_slice, kind=kind, reason=reason, extra=extra)

    if progress is not None:
        progress.bytes_total = manifest["bytes_total"]
        progress.sets_total = manifest["n_sets"]
        progress.state = "writing"
        progress.started_wall_s = time.time()

    t0 = time.monotonic()
    frames_path = dest_dir / FRAMES_FILENAME
    # A request-unique temp name then os.replace(), the same atomic-write
    # discipline local_capture.fetch_snapshot() adopted after a real torn
    # -read incident: a reader that finds frames.bin must find a complete
    # one, never one a writer is still mid-way through.
    tmp_frames = dest_dir / f".tmp.{os.getpid()}.{time.monotonic_ns()}.{FRAMES_FILENAME}"
    written = 0
    sets_done = 0
    try:
        with open(tmp_frames, "wb", buffering=1024 * 1024) as fh:
            for set_plan in plan:
                for arr in set_plan:
                    if isinstance(arr, bytes):
                        fh.write(arr)
                        written += len(arr)
                        continue
                    # C-contiguous is what makes this a memcpy into the
                    # file buffer rather than a per-element gather. Every
                    # array the pump produces already is (cv2 hands back
                    # contiguous frames), so `ascontiguousarray` is a
                    # no-op in practice -- kept because a non-contiguous
                    # array here would otherwise write the WRONG BYTES
                    # silently, which is unrecoverable for a capture that
                    # cannot be retaken.
                    buf = np.ascontiguousarray(arr)
                    fh.write(memoryview(buf).cast("B"))
                    written += int(buf.nbytes)
                sets_done += 1
                if progress is not None and sets_done % _PROGRESS_EVERY_SETS == 0:
                    progress.bytes_written = written
                    progress.sets_written = sets_done
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_frames, frames_path)
    except BaseException:
        Path(tmp_frames).unlink(missing_ok=True)
        raise

    (dest_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2))
    duration = time.monotonic() - t0
    if progress is not None:
        progress.bytes_written = written
        progress.sets_written = manifest["n_sets"]
        progress.duration_s = duration
        progress.finished_wall_s = time.time()
        progress.state = "done"

    log.info(
        "%s dump written to %s -- %d set(s), %d frame(s), %s in %.2fs (%s/s)%s",
        kind, dest_dir, manifest["n_sets"], manifest["n_frames"],
        format_bytes(written), duration,
        format_bytes(written / duration) if duration > 0 else "n/a",
        f" -- NOTE: {reason}" if reason else "",
    )
    return dest_dir


class FrameDumpWriter:
    """One dump at a time, on a background thread, with the ring resumed in
    a `finally`.

    The single-job rule is enforced here rather than at each trigger site,
    so the automatic oracle triggers and the dashboard buttons cannot
    disagree about it -- and so the refusal is worded once.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: "DumpProgress | None" = None
        self._last: "DumpProgress | None" = None
        self._thread: "threading.Thread | None" = None
        self._refusals = 0

    def submit(
        self,
        ring_slice: RingSlice,
        dest_dir: Path,
        *,
        kind: str,
        reason: "str | None" = None,
        extra: "dict[str, Any] | None" = None,
        pause_ring: "FrameRing | None" = None,
        on_done: "Callable[[DumpProgress], None] | None" = None,
    ) -> dict[str, Any]:
        """Start a write. Returns immediately -- the snapshot it was given
        is references, so nothing has been copied yet and nothing will
        block the caller.

        A second submit while one is in flight is REFUSED, with the job
        already running named in the reason and a log line to match. It
        does not queue: a queued 6GB write is a 6GB snapshot held alive
        for the length of another 6GB write, which is how a rig with
        14.4GB free runs out.
        """
        job_id = f"{kind}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
        with self._lock:
            if self._current is not None and self._current.state in ("queued", "writing"):
                self._refusals += 1
                running = self._current
                message = (
                    f"a {running.kind} capture is already writing to "
                    f"{running.dest_dir} ({running.mb_or_unknown()}) -- "
                    "one capture at a time; try again when it finishes"
                )
                log.warning(
                    "frame dump REFUSED (%d so far this session): %s", self._refusals, message
                )
                return {"ok": False, "reason": message, "running": running.as_dict()}

            progress = DumpProgress(
                job_id=job_id, kind=kind, dest_dir=str(dest_dir), reason=reason
            )
            progress.sets_total = len(ring_slice.sets)
            progress.bytes_total = ring_slice.nbytes
            progress.ring_paused_for_write = pause_ring is not None
            self._current = progress
            thread = threading.Thread(
                target=self._run,
                args=(ring_slice, Path(dest_dir), kind, reason, extra, pause_ring,
                      progress, on_done),
                name=f"opendarts-frame-dump-{job_id}",
                daemon=True,
            )
            self._thread = thread
        thread.start()
        return {"ok": True, "job": progress.as_dict()}

    def _run(
        self,
        ring_slice: RingSlice,
        dest_dir: Path,
        kind: str,
        reason: "str | None",
        extra: "dict[str, Any] | None",
        pause_ring: "FrameRing | None",
        progress: DumpProgress,
        on_done: "Callable[[DumpProgress], None] | None",
    ) -> None:
        try:
            write_dump(
                ring_slice, dest_dir, kind=kind, reason=reason,
                extra=extra, progress=progress,
            )
        except BaseException as exc: # noqa: BLE001 -- recorded, never swallowed
            progress.state = "failed"
            progress.error = f"{type(exc).__name__}: {exc}"
            progress.finished_wall_s = time.time()
            log.exception("frame dump to %s FAILED -- the ring is being resumed", dest_dir)
        finally:
            # THE `finally` THAT MATTERS. If this thread dies for any
            # reason -- a full disk, a permissions error, an
            # interpreter-level failure -- the ring must start admitting
            # frames again. Leaving it paused would leave the rig unable
            # to capture anything for the rest of the session while every
            # status surface reported health.
            if pause_ring is not None:
                try:
                    pause_ring.resume()
                except Exception: # noqa: BLE001
                    log.exception("failed to resume the frame ring after a dump")
            with self._lock:
                self._last = progress
                self._current = None
            if on_done is not None:
                try:
                    on_done(progress)
                except Exception: # noqa: BLE001 -- a callback must never strand a job
                    log.exception("frame dump on_done callback raised")

    def status(self) -> dict[str, Any]:
        with self._lock:
            current = self._current
            last = self._last
            refusals = self._refusals
        return {
            "busy": current is not None,
            "current": current.as_dict() if current is not None else None,
            "last": last.as_dict() if last is not None else None,
            "refusals": refusals,
        }

    def join(self, timeout: float = 30.0) -> None:
        """For tests and shutdown. Not used on the live path -- nothing in
        the product ever waits for a dump."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)


# -- reading back -------------------------------------------------------


class FrameDumpReader:
    """Random access to a written dump, byte-for-byte.

    Backed by one open file rather than an mmap on purpose: a 6GB mmap on
    a 16GB rig that is also running other software invites the page cache to
    evict something the capture loop needs, and every consumer here reads
    sequentially anyway. A seek plus a read of a known length is the same
    two syscalls with none of that.

    Every returned array OWNS ITS BUFFER (`np.frombuffer` over bytes this
    reader just read, then reshaped) -- so a consumer may hold it, write
    into it, or hand it to the pump, and closing this reader cannot pull
    the memory out from under it.
    """

    def __init__(self, dump_dir: Path) -> None:
        self.dump_dir = Path(dump_dir)
        manifest_path = self.dump_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"no {MANIFEST_FILENAME} in {self.dump_dir}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text())
        schema = self.manifest.get("schema")
        if schema not in _READABLE_SCHEMAS:
            # Refuse rather than guess. A dump written by a future shape
            # read with today's offsets would produce frames that are
            # confidently wrong rather than visibly broken.
            raise ValueError(
                f"{self.dump_dir}: frame dump schema is {schema!r}, this build "
                f"reads {', '.join(_READABLE_SCHEMAS)} -- refusing to guess at the layout"
            )
        self.sets: list[dict[str, Any]] = list(self.manifest.get("sets") or [])
        self.slots: list[int] = list(self.manifest.get("slots") or [])
        self._frames_path = self.dump_dir / FRAMES_FILENAME
        self._fh = open(self._frames_path, "rb")
        self._read_lock = threading.Lock()

    # Context-manager support so a caller cannot leak the handle.
    def __enter__(self) -> "FrameDumpReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception: # noqa: BLE001
            pass

    def read_frame(self, entry: dict[str, Any]) -> np.ndarray:
        """One frame, reconstructed exactly as it was retained.

        The seek+read is serialised: a hub replaying three slots reads
        this file from three pump worker threads at once, and a shared
        file object's position is exactly the kind of state that produces
        frames from the wrong offsets under concurrency -- silently, and
        as plausible-looking images from the wrong moment."""
        with self._read_lock:
            self._fh.seek(int(entry["offset"]))
            raw = self._fh.read(int(entry["nbytes"]))
        if len(raw) != int(entry["nbytes"]):
            raise IOError(
                f"{self._frames_path}: short read at offset {entry['offset']} "
                f"-- wanted {entry['nbytes']} bytes, got {len(raw)}. The dump is "
                "truncated; do not trust anything replayed from it."
            )
        shape = tuple(entry["shape"])
        if entry.get("encoding", "raw") == "jpeg":
            import cv2

            arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if arr is None or arr.shape != shape:
                raise IOError(
                    f"{self._frames_path}: the JPEG at offset {entry['offset']} "
                    f"decoded to {None if arr is None else arr.shape}, not {shape}. "
                    "Do not trust anything replayed from this dump."
                )
            return arr
        arr = np.frombuffer(raw, dtype=np.dtype(entry["dtype"]))
        return arr.reshape(shape)

    def read_jpeg(self, entry: dict[str, Any]) -> "bytes | None":
        """The frame's JPEG exactly as the camera sent it, or None for a
        frame stored as pixels."""
        if entry.get("encoding", "raw") != "jpeg":
            return None
        with self._read_lock:
            self._fh.seek(int(entry["offset"]))
            return self._fh.read(int(entry["nbytes"]))

    def entries_for_slot(self, slot: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """[(set_record, frame_entry), ...] for one slot, in recorded
        order. Returned as pairs because a replay source needs the SET's
        timestamps to pace itself and the FRAME's byte range to read --
        handing back only the frame entries would force every caller to
        re-walk the manifest to find the timestamps again."""
        out: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for fs in self.sets:
            for entry in fs.get("frames", []):
                if int(entry["slot"]) == int(slot):
                    out.append((fs, entry))
        return out

    def iter_frames(self) -> "Iterable[tuple[dict[str, Any], dict[str, Any], np.ndarray]]":
        for fs in self.sets:
            for entry in fs.get("frames", []):
                yield fs, entry, self.read_frame(entry)


# -- the integrity check ------------------------------------------------


@dataclass
class IntegrityResult:
    """The answer to "is the frame in the package the same array that is in
    the dump", with enough detail to act on a NO."""

    ok: bool
    checked: int = 0
    matched: int = 0
    details: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "matched": self.matched,
            "details": list(self.details),
        }



def _package_dart_frames(package_dir: Path) -> "list[tuple[int, str, np.ndarray | None]]":
    """``(cam, source, array-or-None)`` for every scored dart frame a
    package holds, from wherever that package keeps it.

    Since 2026-09-22 every package keeps it inside its per-camera clip
    (opendarts.capture.clip) and writes no cam*_frame.png at all; the
    existing corpus still has the PNGs. Reading only the PNGs here would
    make every new package fail its own dump check with "nothing to check
    against" -- a false alarm on exactly the throws the dump exists for.
    None means the camera's frame was there but would not decode, which
    the caller reports as a failure rather than skipping."""
    import cv2

    from opendarts.capture import clip

    out: list[tuple[int, str, np.ndarray | None]] = []
    meta_path = package_dir / "meta.json"
    try:
        video = json.loads(meta_path.read_text()).get("video") if meta_path.is_file() else None
    except (OSError, ValueError):
        video = None
    for key in sorted((video or {}).get("cameras") or {}, key=int):
        cam = int(key)
        try:
            arr = clip.read_commit_frame(package_dir, video, cam)
        except Exception:  # noqa: BLE001 -- an unreadable clip is a failed check
            arr = None
        out.append((cam, video["cameras"][key].get("clip", "clip"), arr))
    if out:
        return out
    for png in sorted(package_dir.glob("cam*_frame.png")):
        try:
            cam = int(png.name[len("cam"):-len("_frame.png")])
        except ValueError:
            continue
        out.append((cam, png.name, cv2.imread(str(png), cv2.IMREAD_COLOR)))
    return out

def verify_package_frames_in_dump(package_dir: Path, dump_dir: Path) -> IntegrityResult:
    """Assert that every camera's DART frame in a throw package appears,
    byte-identical, somewhere in the dump.

    This must hold by construction. The package's dart frame is
    `trigger.last_frame[cam]`, which is the array `hub.grab_all()` handed
    the capture loop, which is the same object `_pump_once()` put in the
    cache and the ring. The package writes it as lossless PNG (see
    `save_throw_package()`: "PNG is lossless -- SCORE==STORE requires
    exact pixels, never JPEG"), so decoding it back must reproduce the
    array exactly.

    So this check is not defensive padding -- it is a tripwire for the one
    thing that would quietly invalidate the whole feature: something
    re-encoding or mutating a frame between capture and save. If it ever
    fails, a misscore dump is no longer a record of what was scored, and
    the difference is precisely the sort that produces a confident wrong
    answer rather than a visible error.

    Returns a result rather than raising, so a caller can record the
    finding on the dump and keep the capture (which is still evidence)
    instead of throwing away a throw that cannot be retaken.
    """
    import cv2

    package_dir = Path(package_dir)
    result = IntegrityResult(ok=True)
    try:
        reader = FrameDumpReader(dump_dir)
    except (OSError, ValueError) as exc:
        return IntegrityResult(ok=False, details=[f"cannot read the dump: {exc}"])

    try:
        # Group the dump's frames by slot once, so an N-camera package is
        # N scans of a slot rather than N scans of everything.
        by_slot: dict[int, list[dict[str, Any]]] = {}
        for fs in reader.sets:
            for entry in fs.get("frames", []):
                by_slot.setdefault(int(entry["slot"]), []).append(entry)

        for cam, source, stored in _package_dart_frames(package_dir):
            result.checked += 1
            if stored is None:
                result.ok = False
                result.details.append(f"cam{cam}: {source} could not be decoded")
                continue
            candidates = by_slot.get(cam, [])
            if not candidates:
                result.ok = False
                result.details.append(
                    f"cam{cam}: the dump holds no frame for this slot at all"
                )
                continue
            hit = False
            for entry in candidates:
                if tuple(entry["shape"]) != stored.shape:
                    continue
                if np.array_equal(reader.read_frame(entry), stored):
                    hit = True
                    break
            if hit:
                result.matched += 1
            else:
                result.ok = False
                result.details.append(
                    f"cam{cam}: the package's dart frame is NOT byte-identical to any "
                    f"of the {len(candidates)} frame(s) the dump holds for that slot. "
                    "Something re-encoded or mutated a frame between capture and save "
                    "-- this dump is not a faithful record of what was scored."
                )
    finally:
        reader.close()

    if result.checked == 0:
        result.ok = False
        result.details.append(
            f"{package_dir} holds no dart frame at all (no clip, no "
            "cam*_frame.png) -- nothing to check against"
        )
    if not result.ok:
        log.error("frame dump integrity check FAILED: %s", "; ".join(result.details))
    else:
        log.info(
            "frame dump integrity check ok -- %d/%d package dart frame(s) found "
            "byte-identical in %s", result.matched, result.checked, dump_dir,
        )
    return result


# -- the watchable version ----------------------------------------------

#: What the UI must say next to a "watch this" control. At 33fps a dart is
#: in flight for ~30ms, so a capture holds ONE blurred frame of flight,
#: maybe two. The value of a capture is the SETTLE SEQUENCE, not the
#: flight. Anyone expecting slow motion will be disappointed by physics,
#: not by this feature, and saying so up front is cheaper than explaining
#: it afterwards.
FLIGHT_EXPECTATION_NOTE = (
    "At ~33fps a dart is in flight for about 30ms, so a capture contains one "
    "blurred frame of flight, maybe two. What this shows you is the SETTLE "
    "SEQUENCE -- how the dart came to rest and when the board went still -- "
    "not slow-motion flight."
)


def encode_preview(
    dump_dir: Path,
    slot: int,
    dest_path: Path,
    *,
    fps: "float | None" = None,
    quality: int = 85,
) -> Path:
    """Turn one slot of a dump into a watchable MJPEG file, ON DEMAND.

    Raw stays the record; this is generated when a human asks. ~2000
    frames at the measured 1.3ms/frame for JPEG q85 is ~2.6 seconds of
    work, which is a fine price for something nobody needs until they ask
    for it, and an unacceptable one to pay on every capture.

    `fps` defaults to the dump's OWN measured rate, from its recorded
    monotonic stamps, so the playback runs at the speed the rig really
    captured at rather than a nominal 30 that would quietly stretch or
    compress the settle.
    """
    import cv2

    reader = FrameDumpReader(dump_dir)
    try:
        pairs = reader.entries_for_slot(slot)
        if not pairs:
            raise ValueError(
                f"{dump_dir}: no frames for slot {slot} -- the dump holds slots "
                f"{reader.slots}"
            )
        if fps is None:
            # Measured, not assumed. Monotonic-to-monotonic, never across
            # clocks. Falls back to the nominal rate only when a single
            # frame makes a rate undefinable, and says which it used.
            span = pairs[-1][0]["monotonic_s"] - pairs[0][0]["monotonic_s"]
            if span > 0 and len(pairs) > 1:
                fps = (len(pairs) - 1) / span
            else:
                fps = 30.0
                log.info(
                    "%s slot %d: too few frames to measure a rate -- encoding at a "
                    "nominal %.0ffps", dump_dir, slot, fps,
                )
        h, w = pairs[0][1]["shape"][0], pairs[0][1]["shape"][1]
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(dest_path), cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (int(w), int(h))
        )
        if not writer.isOpened():
            raise IOError(f"cv2.VideoWriter refused to open {dest_path}")
        try:
            for _fs, entry, frame in (
                (fs, entry, reader.read_frame(entry)) for fs, entry in pairs
            ):
                writer.write(frame)
        finally:
            writer.release()
        log.info(
            "encoded %d frame(s) of slot %d from %s to %s at %.1ffps -- %s",
            len(pairs), slot, dump_dir, dest_path, fps, FLIGHT_EXPECTATION_NOTE,
        )
        return dest_path
    finally:
        reader.close()
