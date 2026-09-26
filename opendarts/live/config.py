"""opendarts/live/config.py -- machine-local config file for the live
server/capture-loop process (opendarts.live.run_product, and
opendarts.live.server's own standalone CLI).

2026-08-17: a place for config data, starting with three real
operator-tunable values -- the port to run on, a per-camera reprojection
threshold, and the Autodarts URL -- that were previously
either a hardcoded module constant (the reprojection target -- see
opendarts.live.capture_daemon.CALIBRATION_TARGET_REPROJECTION_ERROR_PX,
applied UNIFORMLY to every camera despite real per-camera differences in
what's actually achievable) or CLI flags you had to remember to pass
every single launch (--port, --ad-base-url).

A plain JSON file under a gitignored `data/` directory, loaded once at
process start and hand-editable -- a familiar shape, not a
new convention invented for this project.

**`config.example.json` at the repo root is a checked-in TEMPLATE of this file,
not this file.** Nothing reads it; it exists so every key and its real
default can be seen without reading source. `data/config.json`
remains the only path any of this module's functions touch.

**Real JSON shape** (every key optional -- omit anything you want left
at its code default):
```json
{
  "port": 8420,
  "host": "0.0.0.0",
  "ad_base_url": "http://localhost:3180",
  "reprojection_targets_px": {"0": 1.0, "1": 3.1, "2": 1.0},
  "camera_resolutions": {"0": "auto", "1": "1920x1080"}
}
```
(`reprojection_targets_px` keys are camera indices -- JSON object keys
are always strings, so they're read back as strings and converted to
int here; a camera not listed keeps using
CALIBRATION_TARGET_REPROJECTION_ERROR_PX, the existing uniform default,
so this file only needs to name the cameras you actually want to
override.)

`camera_resolutions` -- added 2026-08-20, by design's own direction for
the "hardcoded 1280x720 with no runtime check" bug fix (see
opendarts.live.camera_resolution's module docstring for the full bug this
closes): same per-camera-index-keyed-dict shape as
`reprojection_targets_px` above. Each value is either the literal string
`"auto"` (probe this camera's genuinely-supported resolutions at open
time -- opendarts.live.camera_resolution.probe_resolutions() -- and use the
HIGHEST one found, by design's explicit "our default should be highest
available") or an explicit `"WIDTHxHEIGHT"` string (a fixed operator
override, no probing). **A camera missing from this dict is NOT the same
as `"auto"`** -- it means no override at all, i.e. this camera keeps
using `opendarts.live.local_capture.DEFAULT_WIDTH`/`DEFAULT_HEIGHT` (1280x720)
exactly as it does today with zero config file present. This is the
field's own load-bearing backward-compatibility guarantee: an absent
`data/config.json`, or one that simply doesn't mention
`camera_resolutions`, produces an empty dict here, which
`opendarts.live.local_capture.camera_configs_from_resolution_preferences()`
resolves to the exact same fixed-1280x720 `CameraConfig` list
`LocalCameraHub()`'s own bare default already builds -- so "config
absent" and "today's behavior" are provably identical, not just assumed
to be.

**Deliberately loaded ONLY at the CLI entrypoint (opendarts.live.run_product.
main() / opendarts.live.server.main()), never as a bare function-signature
default anywhere in the call chain below that** (AppState.__init__,
create_app(), bootstrap_calibrations(), run_capture_loop_body(), etc. all
keep taking plain values/None) -- same reasoning already established for
DEFAULT_AD_BASE (see run_product.py's own dated comment: "this project's
tests must never make a real network call... by surprise"). A stray
`data/config.json` left over on some machine must never silently
change what a unit test constructing these objects directly sees.
"""
from __future__ import annotations

import json
import threading
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("opendarts.live.config")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from opendarts.paths import DATA_DIR
DEFAULT_CONFIG_PATH = DATA_DIR / "config.json"


@dataclass
class LiveConfig:
    """None for any field means "no override -- caller keeps using its
    own existing code-level default", never a value this module invents
    itself. `reprojection_targets_px` is always a (possibly empty) dict,
    never None, so callers can do `cfg.reprojection_targets_px.get(cam,
    CALIBRATION_TARGET_REPROJECTION_ERROR_PX)` without a None-check."""

    port: int | None = None

    # The interface the dashboard/API binds to. None ("no override")
    # resolves to opendarts.live.server.DEFAULT_HOST, "0.0.0.0" -- every
    # interface, which is what a headless rig on a LAN actually needs.
    #
    # Added 2026-09-13 because `--host` was a CLI flag and nothing else:
    # /api/state reported the bound address but there was no way to SET
    # it short of editing the launcher script. Deliberately file-only,
    # with no dashboard control -- see the Config tab's own comment on
    # the read-only bind-address row for why a wrong value here is not
    # recoverable from the web UI on the Windows rig.
    host: str | None = None

    ad_base_url: str | None = None
    reprojection_targets_px: dict[int, float] = field(default_factory=dict)
    # camera_resolutions -- values are the PARSED form already
    # (tuple[int, int] | None, None meaning "auto"/probe-for-highest),
    # not the raw "auto"/"WxH" JSON strings -- load_live_config() does
    # that parsing once, at load time (via
    # opendarts.live.camera_resolution.parse_resolution_preference()), so
    # every other caller of this dataclass gets an already-normalized
    # value and never has to re-parse a resolution string itself. A
    # camera index absent from this dict means "no override" -- same
    # "None/absent means keep the caller's own existing code-level
    # default" contract as every other field on this dataclass, NOT the
    # same thing as an explicit "auto" entry (which IS present, valued
    # None, meaning "probe" rather than "no override at all") -- see
    # this field's own module-docstring section above for the full
    # distinction and why it matters for backward compatibility.
    camera_resolutions: dict[int, tuple[int, int] | None] = field(default_factory=dict)

    # Which HARDWARE device index feeds each camera slot, in slot order:
    # [1, 2, 3] means slot 0 reads device 1, slot 1 reads device 2, slot 2
    # reads device 3. Added 2026-09-10 for a real case this could not
    # express: a laptop with four cameras, where the built-in webcam takes
    # index 0 and the three board cameras land on 1/2/3. Until now the
    # device list was `DEFAULT_CAMERA_DEVICES` ([0, 1, 2]) baked into
    # `LocalCameraHub.__init__`'s own default, so the built-in was always
    # slot 0 and one board camera was simply unreachable -- with no config
    # key and no UI able to say otherwise.
    #
    # `None` (absent from the file) means "no override" -- the same
    # contract every other field here uses -- and resolves to
    # DEFAULT_CAMERA_DEVICES, so an existing rig behaves exactly as before.
    # Deliberately a LIST, not a slot->device dict: slot order is the
    # meaning, the length IS the camera count, and a list cannot express a
    # gap (slot 2 configured while slot 1 is not) that the rest of the
    # system has no representation for.
    camera_devices: list[int] | None = None

    # Per-slot stream URL, where a slot reads frames over the network
    # instead of off local hardware. Same slot order as camera_devices, and
    # a None entry means "this slot uses its device index as before".
    #
    # Added 2026-09-15 so a machine with no cameras -- a VM, a second rig,
    # a laptop -- can score from another machine's published streams. A URL
    # and a device index are mutually exclusive FOR ONE SLOT, so this is a
    # parallel list rather than a polymorphic camera_devices: keeping
    # "which hardware" and "which address" in separate fields means neither
    # has to be type-sniffed at the point of use.
    camera_urls: "list[str | None] | None" = None

    # Whether Autodarts is consulted as an oracle at all. None means "no
    # override" and resolves to DISABLED as of 2026-09-16. It used to
    # resolve to enabled, matching the behaviour from before this key
    # existed. Turning it on is not free: against an unreachable server
    # the WS listener retries forever in the log. A fresh clone with no Autodarts
    # would pay that to reach a service it has never heard of.
    ad_enabled: bool | None = None

    # How many worker threads OpenCV may use. None ("no override") resolves
    # to opendarts.live.cv2_threads.DEFAULT_CV2_NUM_THREADS, which is 1 --
    # unusually for this dataclass, the code-level default here deliberately
    # CHANGES OpenCV's own behaviour rather than leaving it alone, because
    # OpenCV's self-chosen pool is precisely what is being corrected.
    # Measured on the Windows rig: an unbounded pool cost 62.8% of total
    # process CPU at idle (238.9% of one core down to 89.1%) with no change
    # to any scored call, and calibration measurably unchanged. A
    # NEGATIVE value
    # means "keep OpenCV's default", so a rig with different hardware can
    # opt out from the config file with no code change. See
    # opendarts.live.cv2_threads for the full measurement.
    cv2_num_threads: int | None = None

    # Pixel format published to the Linux v4l2loopback virtual cameras.
    # None means "the publisher's own default" (MJPEG).
    #
    # "BGR24" hands the consuming app the SAME array OpenDarts scored, with no
    # second JPEG generation -- and costs 0.10ms per camera instead of
    # 3.33ms, which at three cameras and 30fps is ~10ms of every 33ms pump
    # cycle handed back to the thread that detects darts.
    #
    # The price is an ORDERING RULE, not a configuration problem: a
    # v4l2loopback node's format is fixed by whoever opens it first, and
    # VIDIOC_S_FMT returns success even when it cannot honour the request.
    # So start OpenDarts before the consuming app, or that app may keep
    # decoding MJPG and show black.
    v4l2_format: str | None = None

    # How many SECONDS of raw frames the throw-capture ring retains, so a
    # missed or misscored dart can be examined after the fact -- see
    # opendarts/capture/frame_ring.py for what it is and
    # opendarts/capture/throw_capture.py for what triggers a dump.
    #
    # In SECONDS because that is the question an operator actually has
    # ("how far back can I reach"), not megabytes. The cost is real and
    # large -- three 720p cameras produce ~270 MB/s on the measured rig,
    # so 15s is ~4GB, 22s is ~6GB, 30s is ~8GB -- which is why the
    # dashboard shows the megabytes beside the seconds as the value is
    # chosen, rather than leaving the arithmetic to the operator.
    #
    # `None` ("no override") resolves to DEFAULT_FRAME_RING_SECONDS, which
    # is deliberately CONSERVATIVE rather than generous: this is memory
    # spent on every rig whether or not anyone ever presses the button,
    # and a rig with 14.4GB free may be running other software too. `0` disables
    # it outright and is a real, supported answer -- a rig that has
    # decided not to spend the memory must be able to say so.
    frame_ring_seconds: float | None = None

    # An optional hard ceiling on the ring, in GB. None means the seconds
    # window is the only bound, which is the honest default: a ceiling
    # SHORTENS the window below what the seconds setting promises, and a
    # setting that silently means something else is the failure this
    # project keeps finding. When one is set and it fires, the ring logs
    # it and reports `capped` so the dashboard can say the window is
    # shorter than configured.
    frame_ring_max_gb: float | None = None

    # How much per-throw video the rig records (opendarts.capture.clip):
    #   "never"    -- no video at all; the frame ring is not created, so its
    #                 RAM is never spent, and throws store bg+dart PNGs as
    #                 before.
    #   "mismatch" -- record a clip only when the oracle (Autodarts)
    #                 disagreed with our score (needs AD connected + matching).
    #   "all"      -- record a clip for every scored dart. The default.
    # `None` resolves to DEFAULT_VIDEO_RECORD_MODE.
    video_record_mode: str | None = None


#: What `frame_ring_seconds=None` resolves to. Twenty seconds is cheap
#: where the ring holds the camera's JPEG (~200MB at three 720p cameras on
#: Linux/Windows passthrough) and heavier where it holds decoded pixels
#: (~5.4GB on macOS) -- the frame_ring_max_gb ceiling is the backstop that
#: keeps the expensive platform sane. Enough history to hold a whole settle
#: plus margin either side. `0` disables it; raising it is one config key.
DEFAULT_FRAME_RING_SECONDS = 20.0

#: The three per-throw video-record modes (LiveConfig.video_record_mode).
VIDEO_RECORD_MODES = ("never", "mismatch", "all")
#: What `video_record_mode=None` resolves to -- record every dart. "all"
#: is the default because the oracle-mismatch path only records when
#: Autodarts is connected AND matching, which a rig cannot count on; "all"
#: captures the evidence regardless. Set "mismatch" or "never" per rig to
#: spend less disk.
DEFAULT_VIDEO_RECORD_MODE = "all"


def normalise_video_record_mode(value: "Any", *, path: "Any" = None) -> "str | None":
    """Validate a `video_record_mode`, or None (resolve to the default) for
    an absent/invalid one. Whitelisted -- a typo must not silently disable
    or over-enable recording."""
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().lower() not in VIDEO_RECORD_MODES:
        log.warning(
            "live config %s: 'video_record_mode' must be one of %s (%r) -- ignoring, "
            "the default (%s) applies",
            path, "/".join(VIDEO_RECORD_MODES), value, DEFAULT_VIDEO_RECORD_MODE,
        )
        return None
    return value.strip().lower()


#: The only pixel formats the Linux virtual-camera publisher accepts.
VALID_V4L2_FORMATS = ("MJPEG", "BGR24")


def normalise_v4l2_format(value: "Any", *, path: "Any" = None) -> "str | None":
    """Validate a `v4l2_format` value, or None if it is unusable.

    ONE implementation, because two callers need it: the LiveConfig parse
    at startup and run_product's own deferred read. Whitelisted rather
    than passed through -- an unrecognised format reaches VIDIOC_S_FMT,
    which SUCCEEDS WITHOUT HONOURING IT, so a typo would present as a
    black camera in the consuming app with no error anywhere.
    """
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().upper() not in VALID_V4L2_FORMATS:
        log.warning(
            "live config %s: 'v4l2_format' must be one of %s (%r) -- ignoring, "
            "the publisher default applies",
            path, "/".join(VALID_V4L2_FORMATS), value,
        )
        return None
    return value.strip().upper()


def _positive_number(
    value: Any, key: str, path: Any, *, allow_zero: bool
) -> "float | None":
    """A non-negative number, or None with a warning naming the key.

    Shared by the two frame-ring keys because they have the same shape and
    the same failure mode, and two copies of this would eventually
    disagree about whether `0` means "off" or "invalid". `bool` is
    rejected explicitly -- it is an `int` subclass in Python, so `true`
    would otherwise read as one second.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        log.warning(
            "live config %s: %r is not a number (%r) -- ignoring", path, key, value
        )
        return None
    number = float(value)
    if number < 0 or (number == 0 and not allow_zero):
        log.warning(
            "live config %s: %r must be %s (%r) -- ignoring",
            path, key, "zero or positive" if allow_zero else "positive", value,
        )
        return None
    return number


_CONFIG_WRITE_LOCK = threading.Lock()


def read_config_section(section: str, path: Path = DEFAULT_CONFIG_PATH) -> Any:
    """Read ONE top-level key out of the shared config file, or None when
    the file/key is absent or unreadable.

    Live-mutable settings (idle timeout, engine config, lifecycle
    settings) all live in this one file rather than a directory each --
    one place an operator edits, one place to back up."""
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("live config %s: unreadable (%s) -- treating %r as absent", path, exc, section)
        return None
    if not isinstance(raw, dict):
        return None
    return raw.get(section)


def write_config_section(section: str, value: Any, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Set ONE top-level key in the shared config file, PRESERVING every
    other key.

    Read-modify-write under a process-wide lock (several stores write
    this same file from different threads), then an atomic replace so a
    crash mid-write can never leave a truncated config behind.

    **Refuses to write over a file it cannot parse.** A hand-edited
    config with a syntax error is a mistake to report, not to silently
    overwrite -- overwriting would destroy the operator's own port /
    camera settings to persist one toggle."""
    with _CONFIG_WRITE_LOCK:
        raw: dict = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning(
                    "live config %s: cannot parse (%s) -- REFUSING to overwrite it; "
                    "%r was not persisted. Fix the file by hand.", path, exc, section,
                )
                return
            if not isinstance(raw, dict):
                log.warning(
                    "live config %s: not a JSON object -- REFUSING to overwrite it; "
                    "%r was not persisted.", path, section,
                )
                return
        raw[section] = value
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(raw, indent=2) + "\n")
            os.replace(tmp, path)
        except Exception as exc:  # noqa: BLE001 -- best effort; never break the dashboard
            log.warning("live config %s: failed to persist %r: %s", path, section, exc)


def store_packages_enabled(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Whether completed throws are written to disk as replay packages.

    The ONE interpreter of this key, shared by the capture loop's own
    reader (opendarts.live.run_product._read_store_packages, which is
    what actually decides a session's behaviour) and by the dashboard
    endpoint that reports and sets it (GET/POST /api/store-packages).
    Deliberately one function rather than the same five lines in two
    places: the two must never disagree about what a missing or
    malformed value means, or the dashboard would show a state the
    capture loop is not in.

    Default TRUE, and a malformed value ALSO reads as TRUE -- a rig that
    silently discarded its own evidence because of a typo would be the
    worst possible failure direction for this particular key.
    """
    try:
        value = read_config_section("store_packages", path)
    except Exception:  # noqa: BLE001 -- a bad config must never stop a session starting
        return True
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    log.warning(
        "config.json store_packages=%r is not a boolean -- storing packages", value
    )
    return True


def detect_from_small_decode_enabled(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Whether detection decodes each camera JPEG straight to small grey
    (see opendarts/capture/lazy_frame.py), leaving the full decode to the
    frames that are scored.

    Default TRUE. False restores the pre-2026-09-26 path exactly: every
    frame fully decoded in the pump, detection shrinking it itself. A
    malformed value reads as the default, with a warning -- it is a
    performance switch, and neither direction loses evidence.
    """
    try:
        value = read_config_section("detect_from_small_decode", path)
    except Exception:  # noqa: BLE001 -- a bad config must never stop a session starting
        return True
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    log.warning(
        "config.json detect_from_small_decode=%r is not a boolean -- using true", value
    )
    return True


def min_free_disk_gb(path: Path = DEFAULT_CONFIG_PATH) -> float:
    """The free-space floor both on-disk writers stop at, in GB.

    The ONE interpreter of this key, the same way
    `store_packages_enabled()` above is the one interpreter of its own --
    the throw-package writer and the frame-ring dump must never disagree
    about where the floor is, or a rig would refuse one and allow the
    other on the same disk.

    Absent or `0` both mean the default (5 GB); a NEGATIVE value disables
    the guard entirely. That resolution lives in
    `opendarts.disk_space.resolve_floor_gb()`, not here, so the config
    file and a caller passing a value directly get the same answer.

    A malformed value reads as the default rather than as "off": the
    failure direction that matters here is a rig filling its disk, not a
    rig refusing to write one package.
    """
    from opendarts.disk_space import DEFAULT_MIN_FREE_DISK_GB, resolve_floor_gb

    try:
        value = read_config_section("min_free_disk_gb", path)
    except Exception:  # noqa: BLE001 -- a bad config must never stop a session starting
        return DEFAULT_MIN_FREE_DISK_GB
    if value is None:
        return DEFAULT_MIN_FREE_DISK_GB
    # bool is an int subclass in Python, so `true` would otherwise read as
    # a 1 GB floor -- a number nobody typed.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        log.warning(
            "config.json min_free_disk_gb=%r is not a number -- using the "
            "default floor of %.1f GB", value, DEFAULT_MIN_FREE_DISK_GB,
        )
        return DEFAULT_MIN_FREE_DISK_GB
    return resolve_floor_gb(float(value))


# -- when the launcher pulls -------------------------------------------
#
# TWO KEYS, AND THE PRODUCT NEVER ACTS ON EITHER. `run.sh`/`run.ps1` are
# the only readers: they sit between the old process exiting and the new
# one starting, which is the one moment in a rig's life when a
# `git pull` is safe. Everything here is therefore a plain config
# accessor -- no live effect, no restart-required notice, nothing to
# apply.
#
# WHY THEY EXIST AT ALL (2026-09-17). Both launchers used to pull on
# EVERY relaunch, unconditionally. A rig that crashed mid-match came back
# on whatever was on main -- possibly a commit pushed ten minutes earlier
# by someone who had no idea a match was running. Updating became a thing
# that happened TO an operator rather than something they asked for.
#
#   always_update           -- for the dev VMs that are supposed to
#                              follow main: pull every relaunch, exactly
#                              as both scripts did before this existed.
#   update_on_next_restart  -- a ONE-SHOT request. The launcher clears it
#                              as it reads it, so an update happens once
#                              and is never repeated by a later crash.
#
# BOTH DEFAULT FALSE, which is the whole point: with neither set, a rig
# that falls over comes back on the identical code it was running.

#: The one-shot key's name, in one place: it is written by
#: POST /api/restart, read and cleared by the launchers, and documented in
#: config.example.json. Three readers of one string is how a typo ships.
UPDATE_ON_NEXT_RESTART_KEY = "update_on_next_restart"

#: The standing "follow main" key's name, same reasoning.
ALWAYS_UPDATE_KEY = "always_update"


def _update_flag(key: str, path: Path) -> bool:
    """One of the two update flags, FALSE for anything that is not a real
    `true`.

    Shared by both readers below for the same reason
    `_positive_number()` is shared by the two frame-ring keys: two copies
    would eventually disagree about what a missing or malformed value
    means, and here that disagreement moves a rig onto code nobody asked
    for.

    The failure direction is deliberate and opposite to
    `store_packages_enabled()`'s. There, a typo must not cost the rig its
    evidence, so a malformed value reads as ON. Here, a malformed value
    reading as ON would mean an unreadable config file silently restored
    the exact behaviour these keys were added to stop -- so anything that
    is not a JSON `true` reads as OFF, and says so in the log.
    """
    try:
        value = read_config_section(key, path)
    except Exception:  # noqa: BLE001 -- a bad config must never stop a rig starting
        return False
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    log.warning(
        "config.json %s=%r is not a boolean -- treating it as false (not pulling)",
        key, value,
    )
    return False


def always_update(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Whether the launcher pulls on EVERY relaunch.

    True is the pre-2026-09-17 behaviour of both launcher scripts, kept
    as an explicit opt-in for the dev VMs that exist to run main.
    """
    return _update_flag(ALWAYS_UPDATE_KEY, path)


def update_on_next_restart(path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Whether ONE pull has been requested for the next relaunch.

    Set by POST /api/restart {"update": true}; cleared by the launcher as
    it acts on it (see opendarts.live.update_policy), so it can never
    turn a crash loop into a rig that keeps pulling.
    """
    return _update_flag(UPDATE_ON_NEXT_RESTART_KEY, path)


def set_update_on_next_restart(value: bool, path: Path = DEFAULT_CONFIG_PATH) -> bool:
    """Persist the one-shot flag, and say whether it really landed.

    READ BACK rather than trusting the write, for the same reason
    POST /api/port does: `write_config_section()` DECLINES (and only
    logs) when the existing file cannot be parsed, so a bare try/except
    would report success for a write that never happened -- and here that
    would mean telling an operator their rig is about to update when it
    is not.
    """
    try:
        write_config_section(UPDATE_ON_NEXT_RESTART_KEY, bool(value), path)
    except Exception as exc:  # noqa: BLE001 -- never break a restart over a config write
        log.warning("live config %s: %s not persisted: %s", path, UPDATE_ON_NEXT_RESTART_KEY, exc)
        return False
    return update_on_next_restart(path) == bool(value)


def load_live_config(path: Path = DEFAULT_CONFIG_PATH) -> LiveConfig:
    """Real, tolerant load -- a missing file is the expected common case
    (nobody's overridden anything yet), returned as an all-None/empty
    LiveConfig, not an error. A malformed file (bad JSON, wrong value
    types) is logged as a warning and ALSO degrades to an all-None/empty
    LiveConfig -- same "a config problem must never crash the process,
    just fall back to the existing hardcoded defaults" posture this
    project already applies to calibration/AD-reachability failures
    (see opendarts.live.server.AppState._refresh_calibration_blocking's own
    docstring for the established precedent)."""
    if not path.exists():
        return LiveConfig()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("failed to read live config at %s (using code defaults): %s", path, exc)
        return LiveConfig()
    if not isinstance(raw, dict):
        log.warning("live config at %s is not a JSON object (using code defaults)", path)
        return LiveConfig()

    port = raw.get("port")
    if port is not None:
        try:
            port = int(port)
        except (TypeError, ValueError):
            log.warning("live config %s: 'port' is not a real int (%r) -- ignoring", path, port)
            port = None

    # An empty string is rejected as well as a non-string: argparse would
    # happily hand "" straight to uvicorn, which binds it as "every
    # interface" on some stacks and refuses it on others. "I left the key
    # blank" is a mistake, never an intent, so it reads as absent.
    host = raw.get("host")
    if host is not None and (not isinstance(host, str) or not host.strip()):
        log.warning(
            "live config %s: 'host' is not a non-empty string (%r) -- ignoring", path, host
        )
        host = None
    elif isinstance(host, str):
        host = host.strip()

    ad_base_url = raw.get("ad_base_url")
    if ad_base_url is not None and not isinstance(ad_base_url, str):
        log.warning(
            "live config %s: 'ad_base_url' is not a string (%r) -- ignoring", path, ad_base_url
        )
        ad_base_url = None

    reprojection_targets_px: dict[int, float] = {}
    raw_targets = raw.get("reprojection_targets_px")
    if isinstance(raw_targets, dict):
        for cam_key, target_val in raw_targets.items():
            try:
                cam = int(cam_key)
                target = float(target_val)
            except (TypeError, ValueError):
                log.warning(
                    "live config %s: ignoring reprojection_targets_px entry %r=%r "
                    "(camera index and target must both be real numbers)",
                    path, cam_key, target_val,
                )
                continue
            reprojection_targets_px[cam] = target
    elif raw_targets is not None:
        log.warning(
            "live config %s: 'reprojection_targets_px' is not a JSON object (%r) -- ignoring",
            path, raw_targets,
        )

    camera_resolutions: dict[int, tuple[int, int] | None] = {}
    raw_resolutions = raw.get("camera_resolutions")
    if isinstance(raw_resolutions, dict):
        # Local import -- keeps this module's own import list free of a
        # cv2-adjacent dependency for callers that only want port/
        # ad_base_url/reprojection_targets_px and never touch resolution
        # config at all (this mirrors camera_resolution.py's own choice
        # to import cv2 lazily, same reasoning: don't force a heavier
        # dependency on a caller that doesn't need it).
        from opendarts.live.camera_resolution import parse_resolution_preference

        for cam_key, pref_val in raw_resolutions.items():
            try:
                cam = int(cam_key)
            except (TypeError, ValueError):
                log.warning(
                    "live config %s: ignoring camera_resolutions entry %r=%r "
                    "(camera index must be a real int)",
                    path, cam_key, pref_val,
                )
                continue
            if not isinstance(pref_val, str):
                log.warning(
                    "live config %s: ignoring camera_resolutions entry for cam%d "
                    "(%r) -- expected a string ('auto' or 'WIDTHxHEIGHT')",
                    path, cam, pref_val,
                )
                continue
            try:
                camera_resolutions[cam] = parse_resolution_preference(pref_val)
            except ValueError as exc:
                log.warning(
                    "live config %s: ignoring camera_resolutions entry for cam%d -- %s",
                    path, cam, exc,
                )
    elif raw_resolutions is not None:
        log.warning(
            "live config %s: 'camera_resolutions' is not a JSON object (%r) -- ignoring",
            path, raw_resolutions,
        )

    camera_devices: list[int] | None = None
    raw_devices = raw.get("camera_devices")
    if isinstance(raw_devices, list):
        parsed_devices: list[int] = []
        bad = False
        for entry in raw_devices:
            # bool is an int subclass; `True` as a device index is a
            # mistake, never an intent, so reject it explicitly.
            if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
                log.warning(
                    "live config %s: ignoring 'camera_devices' -- entry %r is not a "
                    "non-negative int device index",
                    path, entry,
                )
                bad = True
                break
            parsed_devices.append(entry)
        if not bad:
            if len(set(parsed_devices)) != len(parsed_devices):
                # Two slots on one device is not a working configuration --
                # the same hardware cannot be two views of the board -- and
                # silently accepting it would present as "cam1 and cam2 see
                # the same thing", which reads as a mounting problem.
                log.warning(
                    "live config %s: ignoring 'camera_devices' %r -- the same device "
                    "index is assigned to more than one slot",
                    path, raw_devices,
                )
            elif not parsed_devices:
                log.warning(
                    "live config %s: ignoring empty 'camera_devices' -- omit the key "
                    "entirely to use the default device list",
                    path,
                )
            else:
                camera_devices = parsed_devices
    elif raw_devices is not None:
        log.warning(
            "live config %s: 'camera_devices' is not a JSON array (%r) -- ignoring",
            path, raw_devices,
        )

    # Entries are URLs or null; anything else is ignored with a warning
    # rather than silently dropping the whole key, so one bad entry cannot
    # quietly send every slot back to local hardware.
    camera_urls: "list[str | None] | None" = None
    raw_urls = raw.get("camera_urls")
    if isinstance(raw_urls, list):
        parsed_urls: list[str | None] = []
        for entry in raw_urls:
            if entry is None or (isinstance(entry, str) and not entry.strip()):
                parsed_urls.append(None)
            elif isinstance(entry, str) and entry.strip().startswith(("http://", "https://")):
                parsed_urls.append(entry.strip())
            else:
                log.warning(
                    "live config %s: camera_urls entry %r is not an http(s) URL -- "
                    "that slot falls back to its device index", path, entry,
                )
                parsed_urls.append(None)
        camera_urls = parsed_urls if any(u is not None for u in parsed_urls) else None
    elif raw_urls is not None:
        log.warning("live config %s: 'camera_urls' is not a list (%r) -- ignoring", path, raw_urls)

    ad_enabled = raw.get("ad_enabled")
    if ad_enabled is not None and not isinstance(ad_enabled, bool):
        log.warning(
            "live config %s: 'ad_enabled' is not a bool (%r) -- ignoring", path, ad_enabled
        )
        ad_enabled = None

    # `bool` is a subclass of `int` in Python, so `True` would otherwise be
    # accepted here as "one thread" -- rejected explicitly, same as every
    # other typed key above.
    cv2_num_threads = raw.get("cv2_num_threads")
    if cv2_num_threads is not None and (
        isinstance(cv2_num_threads, bool) or not isinstance(cv2_num_threads, int)
    ):
        log.warning(
            "live config %s: 'cv2_num_threads' is not an int (%r) -- ignoring",
            path, cv2_num_threads,
        )
        cv2_num_threads = None

    v4l2_format = normalise_v4l2_format(raw.get("v4l2_format"), path=path)

    frame_ring_seconds = _positive_number(
        raw.get("frame_ring_seconds"), "frame_ring_seconds", path, allow_zero=True
    )
    frame_ring_max_gb = _positive_number(
        raw.get("frame_ring_max_gb"), "frame_ring_max_gb", path, allow_zero=False
    )
    video_record_mode = normalise_video_record_mode(
        raw.get("video_record_mode"), path=path
    )

    return LiveConfig(
        port=port,
        host=host,
        ad_base_url=ad_base_url,
        reprojection_targets_px=reprojection_targets_px,
        camera_resolutions=camera_resolutions,
        camera_devices=camera_devices,
        camera_urls=camera_urls,
        ad_enabled=ad_enabled,
        cv2_num_threads=cv2_num_threads,
        v4l2_format=v4l2_format,
        frame_ring_seconds=frame_ring_seconds,
        frame_ring_max_gb=frame_ring_max_gb,
        video_record_mode=video_record_mode,
    )

