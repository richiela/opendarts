"""opendarts/live/config_document.py -- the rig's settings as ONE document.

Until 2026-09-17 every setting had a route pair of its own:
`/api/ad-config`, `/api/camera-devices`, `/api/store-packages`,
`/api/frame-ring`, `/api/port`, `/api/diagnostics`,
`/api/detection-time`, `/api/idle-timeout`. Eight little contracts, one
of them (idle-timeout) missing its GET entirely, several of them polled
by the dashboard on the same tick, and each one carrying its own private
idea of what a valid value is. A ninth setting -- `engine_config`,
`cv2_num_threads`, `v4l2_format`, `min_free_disk_gb`,
`camera_resolutions`, `reprojection_targets_px` -- had no route at all
and could only be hand-edited.

The replacement is the shape Autodarts already uses for the same
problem: `GET /api/config` hands back the WHOLE effective document,
`PATCH /api/config` takes a partial one and merges it. This module is
the part of that with no FastAPI in it: what the keys are, what each one
means, what a valid value is, and what the value in force is when the
file says nothing.

THE REGISTRY IS THE CONTRACT. `CONFIG_KEYS` below is the single list of
every key the product reads. `effective_document()` renders it; PATCH
validates against it; `config.example.json` is checked against it by a
test. A key that is not in this list is not a config key -- which is
what makes "unknown key" a refusal rather than a silent write of a
typo into the operator's file.

THREE FLAGS PER KEY, and they are not the same question:

* `persist` -- does the value go into `data/config.json`? True for
  everything except `diagnostics`, which is deliberately a
  within-one-process A/B switch (see `opendarts.live.diagnostics_gate`'s
  own docstring: persisting it would add a "did I leave this on"
  footgun to exactly the measurement it exists for).
* `restart` -- is the value read once at startup, so a change cannot
  reach the running process? True for the addresses (`port`, `host`) and
  for everything consumed while the process is coming up.
* `writable` -- may a PATCH touch it at all? False only for
  `capabilities`, which is written by the startup probe and would be
  overwritten by the next launch anyway.

VALIDATION MIRRORS THE LOADER, DELIBERATELY. The rule this project
already applied to `POST /api/camera-devices` ("a value this endpoint
accepts must be a value the loader will accept on the way back up, or
Save would appear to work and then be silently discarded at the next
start") is now the rule for every key, because there is now one place to
state it. The same validator is used in both directions: reading the
effective value runs the file's value through the validator and falls
back to the code default when it does not pass -- which is precisely
what `load_live_config()` does, so the document can never report a value
the next launch will not use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from opendarts.live.config import (
    VALID_V4L2_FORMATS,
    DEFAULT_FRAME_RING_SECONDS,
    DEFAULT_VIDEO_RECORD_MODE,
    VIDEO_RECORD_MODES,
    read_config_section,
    write_config_section,
)

log = logging.getLogger("opendarts.live.config_document")


class ConfigValueError(ValueError):
    """A value a key cannot hold. The message is shown to the operator
    beside the field that produced it, so it names the key's rule, never
    a Python type."""


#: The hard ceiling on the frame ring, in seconds. Sixty seconds of three
#: 720p cameras is ~16GB, the whole of the rig this runs on. It lives
#: here rather than in `server.py` because the validator that enforces it
#: lives here and a second copy would eventually disagree.
MAX_FRAME_RING_SECONDS = 60.0


# ---------------------------------------------------------------------------
# Small shared validators
# ---------------------------------------------------------------------------

def _bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigValueError(f"{key} must be true or false, got {value!r}")
    return value


def _number(value: Any, key: str) -> float:
    # bool is an int subclass in Python, so `true` would otherwise read as
    # 1 -- a number nobody typed. Rejected explicitly for every numeric
    # key, exactly as load_live_config()'s own `_positive_number` does.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigValueError(f"{key} must be a number, got {value!r}")
    return float(value)


def _whole_number(value: Any, key: str) -> int:
    """An int, a float with no fraction, or a numeric string.

    All three are real shapes: a JSON number arrives as int or float, and
    a form/text field arrives as a string. Everything else is a typo --
    `int("8420abc")` raises and `int(8420.7)` silently truncates, and
    both would write a value the operator did not choose.
    """
    if isinstance(value, bool):
        raise ConfigValueError(f"{key} must be a whole number, not a boolean")
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            raise ConfigValueError(
                f"{key} must be a whole number, got {value!r}"
            ) from None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ConfigValueError(f"{key} must be a whole number, got {value!r}")


def _http_url(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigValueError(f"{key} must be a non-empty string")
    candidate = value.strip().rstrip("/")
    # Checked before anything is applied or written: a URL with no scheme
    # silently becomes an unreachable host later, which presents as "it
    # stopped working" a long way from the edit that caused it.
    if not candidate.startswith(("http://", "https://")):
        raise ConfigValueError(
            f"{key} must start with http:// or https:// (got {candidate!r})"
        )
    return candidate


def _camera_index_map(value: Any, key: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigValueError(f"{key} must be an object keyed by camera index")
    out: dict[str, Any] = {}
    for cam_key, entry in value.items():
        try:
            cam = int(cam_key)
        except (TypeError, ValueError):
            raise ConfigValueError(
                f"{key}: {cam_key!r} is not a camera index"
            ) from None
        if cam < 0:
            raise ConfigValueError(f"{key}: camera index {cam} is negative")
        out[str(cam)] = entry
    return out


# ---------------------------------------------------------------------------
# Per-key validators
# ---------------------------------------------------------------------------

def _v_port(value: Any, ctx: dict) -> int:
    port = _whole_number(value, "port")
    if not (1 <= port <= 65535):
        raise ConfigValueError(f"port must be between 1 and 65535, got {port}")
    return port


def _v_host(value: Any, ctx: dict) -> str:
    # An empty string is rejected as well as a non-string: uvicorn binds
    # "" as "every interface" on some stacks and refuses it on others, so
    # a blank value is a mistake, never an intent.
    if not isinstance(value, str) or not value.strip():
        raise ConfigValueError("host must be a non-empty address string")
    return value.strip()


def _v_ad_base_url(value: Any, ctx: dict) -> str:
    return _http_url(value, "ad_base_url")


def _v_camera_devices(value: Any, ctx: dict) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ConfigValueError("camera_devices must be a non-empty array of device indices")
    cleaned: list[int] = []
    for entry in value:
        if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
            raise ConfigValueError(f"device index {entry!r} is not a non-negative int")
        cleaned.append(entry)
    if len(set(cleaned)) != len(cleaned):
        # Two slots on one device is not a working configuration -- the
        # same hardware cannot be two views of the board -- and accepting
        # it silently presents as "cam1 and cam2 see the same thing",
        # which reads as a mounting problem.
        raise ConfigValueError("the same device is assigned to more than one slot")
    return cleaned


def _v_camera_urls(value: Any, ctx: dict) -> "list[str | None] | None":
    """Per-slot stream URLs, or null for "every slot reads local hardware".

    A blank entry is null, not an error: that is how the dashboard says
    "this slot went back to its device". A non-blank entry must be a real
    http(s) URL, and ONE bad entry rejects the whole key rather than
    half-applying an assignment.
    """
    if value is None:
        return None
    if not isinstance(value, list):
        raise ConfigValueError("camera_urls must be an array (null for a local slot) or null")
    cleaned: list[str | None] = []
    for entry in value:
        if entry is None or (isinstance(entry, str) and not entry.strip()):
            cleaned.append(None)
            continue
        if not isinstance(entry, str):
            raise ConfigValueError(f"url entry {entry!r} is not a string")
        candidate = entry.strip()
        if not candidate.startswith(("http://", "https://")):
            raise ConfigValueError(
                f"url must start with http:// or https:// -- got {candidate!r}"
            )
        cleaned.append(candidate)
    return cleaned


def _v_camera_resolutions(value: Any, ctx: dict) -> dict:
    raw = _camera_index_map(value, "camera_resolutions")
    # Local import: keeps this module's import list free of a cv2-adjacent
    # dependency for the callers that never touch resolution config, the
    # same way load_live_config() defers it.
    from opendarts.live.camera_resolution import parse_resolution_preference

    for cam, pref in raw.items():
        if not isinstance(pref, str):
            raise ConfigValueError(
                f"camera_resolutions[{cam}] must be 'auto' or 'WIDTHxHEIGHT', got {pref!r}"
            )
        try:
            parse_resolution_preference(pref)
        except ValueError as exc:
            raise ConfigValueError(f"camera_resolutions[{cam}]: {exc}") from None
    return raw


def _v_reprojection_targets(value: Any, ctx: dict) -> dict:
    raw = _camera_index_map(value, "reprojection_targets_px")
    out: dict[str, float] = {}
    for cam, target in raw.items():
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise ConfigValueError(
                f"reprojection_targets_px[{cam}] must be a number of pixels, got {target!r}"
            )
        if target <= 0:
            raise ConfigValueError(
                f"reprojection_targets_px[{cam}] must be positive, got {target!r}"
            )
        out[cam] = float(target)
    return out


def _v_store_packages(value: Any, ctx: dict) -> bool:
    return _bool(value, "store_packages")


def _v_video_record_mode(value: Any, ctx: dict) -> str:
    if not isinstance(value, str) or value.strip().lower() not in VIDEO_RECORD_MODES:
        raise ConfigValueError(
            "video_record_mode must be one of " + "/".join(VIDEO_RECORD_MODES)
        )
    return value.strip().lower()


def _v_min_free_disk_gb(value: Any, ctx: dict) -> float:
    # Negative is a real, supported answer here: it disables the guard.
    # See opendarts.disk_space.resolve_floor_gb().
    return _number(value, "min_free_disk_gb")


def _v_frame_ring_seconds(value: Any, ctx: dict) -> float:
    seconds = _number(value, "frame_ring_seconds")
    if seconds < 0:
        raise ConfigValueError("frame_ring_seconds cannot be negative")
    if seconds > MAX_FRAME_RING_SECONDS:
        # REFUSED WITH THE ARITHMETIC. The caller supplies this rig's own
        # bytes/second when it has one, so the refusal quotes the memory
        # the operator was about to spend rather than an abstract cap.
        per_second = ctx.get("frame_ring_bytes_per_s")
        n_slots = ctx.get("n_slots")
        if per_second and n_slots:
            from opendarts.capture import frame_ring

            estimate = frame_ring.format_bytes(float(per_second) * seconds)
            raise ConfigValueError(
                f"{seconds:g}s across {n_slots} slot(s) is about {estimate} of memory, "
                f"and this build caps the frame ring at {MAX_FRAME_RING_SECONDS:g}s. "
                "Raise the cap in source if a machine really has the RAM for it."
            )
        raise ConfigValueError(
            f"this build caps the frame ring at {MAX_FRAME_RING_SECONDS:g}s, "
            f"got {seconds:g}s"
        )
    return seconds


def _v_frame_ring_max_gb(value: Any, ctx: dict) -> "float | None":
    if value is None:
        return None
    gb = _number(value, "frame_ring_max_gb")
    if gb <= 0:
        raise ConfigValueError("frame_ring_max_gb must be positive (omit it for no ceiling)")
    return gb


def _v_publish_virtual_cameras(value: Any, ctx: dict) -> "bool | None":
    # null is meaningful and is the shipped state: it means "follow
    # ad_enabled". An explicit true/false wins in either direction.
    if value is None:
        return None
    return _bool(value, "publish_virtual_cameras")


def _v_cv2_num_threads(value: Any, ctx: dict) -> int:
    # A NEGATIVE value means "keep OpenCV's own pool" -- a rig with
    # different hardware opts out from the config file, no code change.
    return _whole_number(value, "cv2_num_threads")


def _v_v4l2_format(value: Any, ctx: dict) -> "str | None":
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().upper() not in VALID_V4L2_FORMATS:
        # Whitelisted rather than passed through: an unrecognised format
        # reaches VIDIOC_S_FMT, which SUCCEEDS WITHOUT HONOURING IT, so a
        # typo would present as a black camera in the consuming app with
        # no error anywhere.
        raise ConfigValueError(
            f"v4l2_format must be one of {'/'.join(VALID_V4L2_FORMATS)}, got {value!r}"
        )
    return value.strip().upper()


def _v_idle_timeout_sec(value: Any, ctx: dict) -> int:
    # CLAMPED, not rejected: 0 or below disables the auto-stop entirely
    # and a negative number is simply that request spelled awkwardly.
    # This is CaptureLoopController.set_idle_timeout_sec()'s own rule.
    return max(0, _whole_number(value, "idle_timeout_sec"))


def _v_diagnostics(value: Any, ctx: dict) -> dict:
    if not isinstance(value, dict):
        raise ConfigValueError("diagnostics must be an object with an 'enabled' flag")
    unknown = sorted(set(value) - {"enabled"})
    if unknown:
        raise ConfigValueError(f"diagnostics has no key {unknown[0]!r}")
    if "enabled" not in value:
        raise ConfigValueError("diagnostics must include 'enabled'")
    return {"enabled": _bool(value["enabled"], "diagnostics.enabled")}


def _v_lifecycle_settings(value: Any, ctx: dict) -> dict:
    from opendarts.lifecycle.settings import (
        DART_STABLE_FRAMES_MAX,
        DART_STABLE_FRAMES_MIN,
    )

    if not isinstance(value, dict):
        raise ConfigValueError("lifecycle_settings must be an object")
    unknown = sorted(set(value) - {"dart_stable_frames"})
    if unknown:
        raise ConfigValueError(f"lifecycle_settings has no key {unknown[0]!r}")
    n = value.get("dart_stable_frames")
    if not isinstance(n, int) or isinstance(n, bool) or not (
        DART_STABLE_FRAMES_MIN <= n <= DART_STABLE_FRAMES_MAX
    ):
        raise ConfigValueError(
            f"dart_stable_frames must be an int in "
            f"[{DART_STABLE_FRAMES_MIN}, {DART_STABLE_FRAMES_MAX}], got {n!r}"
        )
    return {"dart_stable_frames": n}


def _v_engine_config(value: Any, ctx: dict) -> dict:
    from opendarts.engines.registry import engine_names, is_registered

    if not isinstance(value, dict):
        raise ConfigValueError("engine_config must be an object")
    unknown = sorted(set(value) - {"primary", "also_run", "timeout_s"})
    if unknown:
        raise ConfigValueError(f"engine_config has no key {unknown[0]!r}")
    primary = value.get("primary")
    if not isinstance(primary, str) or not is_registered(primary):
        raise ConfigValueError(
            f"engine_config.primary must be one of {', '.join(engine_names())}, "
            f"got {primary!r}"
        )
    also_run = value.get("also_run", [])
    if not isinstance(also_run, (list, tuple)):
        raise ConfigValueError("engine_config.also_run must be an array of engine names")
    for name in also_run:
        if not isinstance(name, str) or not is_registered(name):
            raise ConfigValueError(
                f"engine_config.also_run names an unregistered engine {name!r}; "
                f"available: {', '.join(engine_names())}"
            )
    timeout_s = value.get("timeout_s")
    timeout_s = _number(timeout_s, "engine_config.timeout_s")
    if timeout_s <= 0:
        raise ConfigValueError("engine_config.timeout_s must be positive")
    return {
        "primary": primary,
        "also_run": [str(n) for n in also_run],
        "timeout_s": timeout_s,
    }


def _v_always_update(value: Any, ctx: dict) -> bool:
    return _bool(value, "always_update")


def _v_update_on_next_restart(value: Any, ctx: dict) -> bool:
    return _bool(value, "update_on_next_restart")


# ---------------------------------------------------------------------------
# Code defaults -- what the value is when the file says nothing
# ---------------------------------------------------------------------------

def _default_publish_virtual_cameras(_: dict) -> None:
    return None


def _default_engine_config(_: dict) -> dict:
    from opendarts.engines.registry import DEFAULT_ALSO_RUN, DEFAULT_PRIMARY_ENGINE
    from opendarts.engines.dispatch import DEFAULT_ENGINE_TIMEOUT_S

    return {
        "primary": DEFAULT_PRIMARY_ENGINE,
        "also_run": list(DEFAULT_ALSO_RUN),
        "timeout_s": float(DEFAULT_ENGINE_TIMEOUT_S),
    }


def _default_lifecycle_settings(_: dict) -> dict:
    from opendarts.lifecycle.state import DEFAULT_CONFIG

    return {"dart_stable_frames": DEFAULT_CONFIG.dart_stable_frames}


def _default_port(_: dict) -> int:
    from opendarts.live.server import DEFAULT_PORT

    return DEFAULT_PORT


def _default_host(_: dict) -> str:
    from opendarts.live.server import DEFAULT_HOST

    return DEFAULT_HOST


def _default_ad_base_url(_: dict) -> str:
    from opendarts.live.ad_ground_truth import DEFAULT_AD_BASE

    return DEFAULT_AD_BASE


def _default_camera_devices(_: dict) -> list:
    from opendarts.live.local_capture import DEFAULT_CAMERA_DEVICES

    return list(DEFAULT_CAMERA_DEVICES)


def _default_min_free_disk_gb(_: dict) -> float:
    from opendarts.disk_space import DEFAULT_MIN_FREE_DISK_GB

    return float(DEFAULT_MIN_FREE_DISK_GB)


def _default_cv2_num_threads(_: dict) -> int:
    from opendarts.live.cv2_threads import DEFAULT_CV2_NUM_THREADS

    return int(DEFAULT_CV2_NUM_THREADS)


def _default_idle_timeout_sec(_: dict) -> int:
    from opendarts.live.capture_daemon import IDLE_TIMEOUT_SEC_DEFAULT

    return int(IDLE_TIMEOUT_SEC_DEFAULT)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigKey:
    """One top-level key of `data/config.json`.

    `default` is a callable rather than a value because almost every code
    default lives in the module that consumes it (`DEFAULT_PORT` in the
    server, `IDLE_TIMEOUT_SEC_DEFAULT` in the capture daemon), and this
    module must import none of them at module scope -- several would be
    import cycles, and all of them would make reading one config key pull
    in the whole live stack.
    """

    name: str
    default: Callable[[dict], Any]
    validate: "Callable[[Any, dict], Any] | None" = None
    #: False for `capabilities` -- written by the startup probe, and
    #: anything typed into it is overwritten at the next launch.
    writable: bool = True
    #: True when the value is read once while the process starts, so a
    #: change cannot reach the running one.
    restart: bool = True
    #: False only for `diagnostics`: a within-one-process A/B switch that
    #: is deliberately not remembered across a restart.
    persist: bool = True
    #: One line, shown in the API docs and the template. Present tense,
    #: says what the key DOES, never its type.
    doc: str = ""


CONFIG_KEYS: tuple[ConfigKey, ...] = (
    ConfigKey(
        "host", _default_host, _v_host, restart=True,
        doc="the interface the dashboard and API bind to",
    ),
    ConfigKey(
        "port", _default_port, _v_port, restart=True,
        doc="the TCP port the dashboard and API are served on",
    ),
    ConfigKey(
        "ad_base_url", _default_ad_base_url, _v_ad_base_url, restart=False,
        doc="where the Autodarts instance being compared against is",
    ),
    ConfigKey(
        "ad_enabled", lambda ctx: False,
        lambda v, ctx: _bool(v, "ad_enabled"), restart=False,
        doc="whether Autodarts is consulted as a second opinion at all",
    ),
    ConfigKey(
        "camera_devices", _default_camera_devices, _v_camera_devices, restart=False,
        doc="which hardware device index feeds each camera slot, in slot order",
    ),
    ConfigKey(
        "camera_urls", lambda ctx: None, _v_camera_urls, restart=False,
        doc="per-slot stream URL, where a slot reads frames over the network",
    ),
    ConfigKey(
        "camera_resolutions", lambda ctx: {}, _v_camera_resolutions, restart=True,
        doc="per-camera capture resolution: 'auto' or 'WIDTHxHEIGHT'",
    ),
    ConfigKey(
        "reprojection_targets_px", lambda ctx: {}, _v_reprojection_targets, restart=True,
        doc="per-camera calibration reprojection error, in pixels, to aim for",
    ),
    ConfigKey(
        "store_packages", lambda ctx: True, _v_store_packages, restart=False,
        doc="whether completed throws are written to disk as replay packages",
    ),
    ConfigKey(
        "video_record_mode", lambda ctx: DEFAULT_VIDEO_RECORD_MODE,
        _v_video_record_mode, restart=True,
        doc="per-throw video: 'never' (no frame ring), 'mismatch' (only when Autodarts disagrees), or 'all'",
    ),
    ConfigKey(
        "min_free_disk_gb", _default_min_free_disk_gb, _v_min_free_disk_gb, restart=False,
        doc="the free-space floor both on-disk writers stop at",
    ),
    ConfigKey(
        "frame_ring_seconds", lambda ctx: float(DEFAULT_FRAME_RING_SECONDS),
        _v_frame_ring_seconds, restart=False,
        doc="how many seconds of raw camera frames are kept in memory",
    ),
    ConfigKey(
        "frame_ring_max_gb", lambda ctx: None, _v_frame_ring_max_gb, restart=False,
        doc="an optional hard ceiling on the frame ring, in GB",
    ),
    ConfigKey(
        "publish_virtual_cameras", _default_publish_virtual_cameras,
        _v_publish_virtual_cameras, restart=True,
        doc="whether the virtual cameras other software reads are fed (null follows ad_enabled)",
    ),
    ConfigKey(
        "cv2_num_threads", _default_cv2_num_threads, _v_cv2_num_threads, restart=True,
        doc="how many worker threads OpenCV may use (negative keeps OpenCV's own pool)",
    ),
    ConfigKey(
        "v4l2_format", lambda ctx: None, _v_v4l2_format, restart=True,
        doc="pixel format published to the Linux v4l2loopback cameras (null is the publisher default)",
    ),
    ConfigKey(
        "idle_timeout_sec", _default_idle_timeout_sec, _v_idle_timeout_sec, restart=False,
        doc="seconds of no darts before the capture loop auto-stops; 0 disables it",
    ),
    ConfigKey(
        "always_update", lambda ctx: False, _v_always_update, restart=False,
        doc="whether every restart pulls the latest code before relaunching",
    ),
    ConfigKey(
        "update_on_next_restart", lambda ctx: False, _v_update_on_next_restart,
        restart=False,
        doc="pull the latest code at the NEXT restart only; the launcher clears it",
    ),
    ConfigKey(
        "lifecycle_settings", _default_lifecycle_settings, _v_lifecycle_settings,
        restart=False,
        doc="how many consecutive still frames a landed dart must hold before it is scored",
    ),
    ConfigKey(
        "engine_config", _default_engine_config, _v_engine_config, restart=True,
        doc="which engine is primary, which also run, and the per-engine timeout",
    ),
    ConfigKey(
        "diagnostics", lambda ctx: {"enabled": False}, _v_diagnostics,
        restart=False, persist=False,
        doc="the live per-dart diagnostics switch; deliberately not remembered across a restart",
    ),
    ConfigKey(
        "capabilities", lambda ctx: None, None, writable=False, restart=True,
        doc="what external tools this machine has -- written by the startup probe, read-only here",
    ),
)

KEYS_BY_NAME: dict[str, ConfigKey] = {k.name: k for k in CONFIG_KEYS}

#: The keys a PATCH may touch, in document order. `capabilities` is not
#: one of them.
WRITABLE_KEYS: tuple[str, ...] = tuple(k.name for k in CONFIG_KEYS if k.writable)


# ---------------------------------------------------------------------------
# Reading the effective document
# ---------------------------------------------------------------------------

def stored_value(key: str) -> Any:
    """The RAW value in `data/config.json`, or None when the key (or the
    file) is absent. Never validated -- callers that want the value in
    force want `effective_value()`."""
    try:
        return read_config_section(key)
    except Exception:  # noqa: BLE001 -- a bad config must never break the dashboard
        log.warning("could not read %r from the config file", key, exc_info=True)
        return None


def effective_value(key: str, ctx: "dict | None" = None) -> Any:
    """The value actually in force for one key: the file's, when the file
    has one that the loader would accept, and the code default otherwise.

    A file value that does not pass validation reads as ABSENT, which is
    exactly what `load_live_config()` does with it -- so this can never
    report a value the next launch will not use.
    """
    spec = KEYS_BY_NAME[key]
    ctx = ctx or {}
    raw = stored_value(key)
    if raw is None:
        return spec.default(ctx)
    if spec.validate is None:
        return raw
    try:
        return spec.validate(raw, ctx)
    except ConfigValueError as exc:
        log.warning("config.json %s is not usable (%s) -- the default applies", key, exc)
        return spec.default(ctx)


def effective_document(ctx: "dict | None" = None,
                       overrides: "dict[str, Any] | None" = None) -> dict[str, Any]:
    """The whole document, in registry order.

    `overrides` is how the LIVE value wins over the file's for the
    handful of keys a running process holds in memory (the idle timeout
    on the capture controller, the lifecycle settings store, the
    diagnostics gate). Those three can legitimately differ from the file
    for the life of one process, and the document is supposed to say
    what is IN FORCE.
    """
    ctx = ctx or {}
    overrides = overrides or {}
    out: dict[str, Any] = {}
    for spec in CONFIG_KEYS:
        if spec.name in overrides:
            out[spec.name] = overrides[spec.name]
        else:
            out[spec.name] = effective_value(spec.name, ctx)
    return out


# ---------------------------------------------------------------------------
# Validating a partial document
# ---------------------------------------------------------------------------

#: Keys whose value is an object that a PATCH merges INTO rather than
#: replaces, so `{"lifecycle_settings": {...}}` cannot silently drop a
#: sibling sub-key. The camera-index maps are NOT here: those are whole
#: values (a slot absent from `camera_resolutions` means something, and a
#: merge could never remove one).
MERGED_GROUPS: tuple[str, ...] = ("lifecycle_settings", "engine_config", "diagnostics")


def validate_patch(
    body: Any, ctx: "dict | None" = None, current: "dict[str, Any] | None" = None,
) -> "tuple[dict[str, Any], dict[str, str]]":
    """`(cleaned, errors)` for a partial document.

    ALL-OR-NOTHING: the caller applies `cleaned` only when `errors` is
    empty, so a request naming five keys and fumbling one changes none of
    them. That is the whole reason validation is separated from
    application -- a per-key loop that wrote as it went would leave the
    rig in a state no operator asked for and no response could describe.

    `errors` is keyed BY KEY, never a single string: a form with three
    fields needs to know which one it got wrong.
    """
    ctx = ctx or {}
    errors: dict[str, str] = {}
    cleaned: dict[str, Any] = {}
    if not isinstance(body, dict):
        return {}, {"_body": "the request body must be a JSON object of config keys"}
    if not body:
        return {}, {"_body": "the request body must name at least one config key"}
    for key, value in body.items():
        spec = KEYS_BY_NAME.get(key)
        if spec is None:
            errors[key] = (
                f"unknown config key {key!r} -- this rig has no such setting"
            )
            continue
        if not spec.writable or spec.validate is None:
            errors[key] = (
                f"{key} is read-only: it is written by the startup probe, "
                "not by hand"
            )
            continue
        merged = value
        if key in MERGED_GROUPS and isinstance(value, dict):
            # Merge onto what is in force so a partial group is really
            # partial. Validated as a WHOLE afterwards, because a group's
            # rules can span its sub-keys (engine_config.primary is
            # required whenever the section exists at all).
            base = (current or {}).get(key)
            if base is None:
                base = effective_value(key, ctx)
            if isinstance(base, dict):
                merged = {**base, **value}
        try:
            cleaned[key] = spec.validate(merged, ctx)
        except ConfigValueError as exc:
            errors[key] = str(exc)
        except Exception as exc:  # noqa: BLE001 -- a validator bug must not 500 the dashboard
            log.warning("validating %r raised", key, exc_info=True)
            errors[key] = f"{key} could not be validated: {exc}"
    return cleaned, errors


def persist(key: str, value: Any) -> bool:
    """Write one key and PROVE it landed.

    `write_config_section()` deliberately REFUSES (and only logs) when the
    existing file cannot be parsed -- a hand-edited config with a syntax
    error is a mistake to report, not to silently overwrite. So trusting
    the write would report success for a no-op, which is why every one of
    the retired routes read its own key back and why this does too.
    """
    try:
        write_config_section(key, value)
    except Exception as exc:  # noqa: BLE001 -- never break the dashboard over a config write
        log.warning("%s not persisted: %s", key, exc)
        return False
    return stored_value(key) == value
