"""Displays: screens that only show, set up from any other screen.

A display is a browser opened at ``/?display`` -- typically a TV
beside the board, in a kiosk with no keyboard. It never shows config,
takes no input, and cannot be set up from itself. Every setting it has
(what it shows, how big, which paper, whether it speaks) lives HERE, on
the rig, keyed by the display's id; any controller (the ordinary dashboard
page on a phone or laptop) edits it, and the change is pushed to the
display over the events socket the moment it is saved.

So the settings belong to the display, not to whoever changed them last:
two displays can look different, any controller can change either, and
a display that is wiped, rebooted or replaced under the same id comes
back looking the same.

What is persisted: each display's name and settings, as the
``displays`` section of data/config.json (``snapshot_path``). What is not:
when it was last seen and what it reported about itself -- that is
liveness, and it is only true while the display keeps reporting.

``DisplayStore(snapshot_path=None)`` stays in memory, as tests and any
embedding that never passed a path expect.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# What a display can show. "split" is the TV default: the scoreboard on
# one side and every engine's call on the other.
LAYOUTS = ("split", "scoring", "engines")
TEXT_SIZES = ("normal", "large", "huge")
THEMES = ("dark", "light")          # the lab half: blueprint or paper
BOARD_VIEWS = ("photo", "diagram")

# A TV is dark: the scoreboard half always is, so the engines half is a
# blueprint beside it rather than a sheet of white paper.
DEFAULT_SETTINGS: dict[str, Any] = {
    "layout": "split",
    "text_size": "normal",
    "theme": "dark",
    "board_view": "photo",
    "sound": False,
    "voice": "",
    "volume": 0.8,
    "locked": False,
}

# A display reports in every ~20 s; three missed reports is "not showing".
ONLINE_S = 70.0
MAX_DISPLAYS = 24
NAME_MAX = 40
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class DisplayError(ValueError):
    """A request the store refuses; ``status`` is the HTTP answer."""

    def __init__(self, reason: str, status: int = 400) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status = status


def valid_id(display_id: Any) -> bool:
    return isinstance(display_id, str) and bool(_ID_RE.match(display_id))


def _clean_settings(raw: Any, base: dict[str, Any]) -> dict[str, Any]:
    """``base`` with every valid key of ``raw`` applied. An unknown key or a
    bad value is an error, not a silent drop: a controller that sends
    something this store cannot keep must be told, or it will show a
    setting the TV never received."""
    if not isinstance(raw, dict):
        raise DisplayError("settings must be an object")
    out = dict(base)
    for key, value in raw.items():
        if key == "layout":
            ok = value in LAYOUTS
        elif key == "text_size":
            ok = value in TEXT_SIZES
        elif key == "theme":
            ok = value in THEMES
        elif key == "board_view":
            ok = value in BOARD_VIEWS
        elif key in ("sound", "locked"):
            ok = isinstance(value, bool)
        elif key == "voice":
            ok = isinstance(value, str) and len(value) <= 64
        elif key == "volume":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1
            if ok:
                value = round(float(value), 3)
        else:
            raise DisplayError(f"unknown setting {key!r}")
        if not ok:
            raise DisplayError(f"invalid value for {key!r}: {value!r}")
        out[key] = value
    return out


def _clean_name(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise DisplayError("name must be a non-empty string")
    return " ".join(name.split())[:NAME_MAX]


class DisplayStore:
    """Thread-safe registry of displays: persisted settings, live presence."""

    def __init__(self, *, snapshot_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._path = Path(snapshot_path) if snapshot_path is not None else None
        # id -> {"name", "settings", "created_at_utc"}; insertion order is
        # the order displays were first seen, which is the order listed.
        self._displays: dict[str, dict[str, Any]] = {}
        # id -> {"seen_monotonic", "seen_at_utc", "info"}
        self._presence: dict[str, dict[str, Any]] = {}
        if self._path is not None:
            self._load()

    # ---- persistence ----
    def _load(self) -> None:
        from opendarts.live.config import read_config_section

        raw = read_config_section("displays", self._path)
        if not isinstance(raw, dict):
            return
        for display_id, rec in raw.items():
            if not valid_id(display_id) or not isinstance(rec, dict):
                continue
            try:
                name = _clean_name(rec.get("name") or display_id)
                settings = _clean_settings(
                    {k: v for k, v in (rec.get("settings") or {}).items() if k in DEFAULT_SETTINGS},
                    DEFAULT_SETTINGS)
            except DisplayError as exc:
                log.warning("displays: ignoring saved display %r (%s)", display_id, exc.reason)
                continue
            self._displays[display_id] = {
                "name": name, "settings": settings,
                "created_at_utc": rec.get("created_at_utc"),
            }

    def _save(self) -> None:
        """Best-effort: a disk hiccup must not fail the change the TV has
        already been told about. Caller holds the lock."""
        if self._path is None:
            return
        from opendarts.live.config import write_config_section

        try:
            write_config_section("displays", {
                did: {"name": rec["name"], "settings": rec["settings"],
                      "created_at_utc": rec.get("created_at_utc")}
                for did, rec in self._displays.items()
            }, self._path)
        except Exception:  # noqa: BLE001 -- persistence is best-effort
            log.exception("displays: could not persist")

    # ---- views ----
    def _view(self, display_id: str, now: float) -> dict[str, Any]:
        rec = self._displays[display_id]
        seen = self._presence.get(display_id)
        age = None if seen is None else round(now - seen["seen_monotonic"], 1)
        return {
            "id": display_id,
            "name": rec["name"],
            "settings": dict(rec["settings"]),
            "online": age is not None and age <= ONLINE_S,
            "age_s": age,
            "seen_at_utc": None if seen is None else seen["seen_at_utc"],
            "info": {} if seen is None else dict(seen["info"]),
        }

    def get(self, display_id: str) -> dict[str, Any] | None:
        with self._lock:
            if display_id not in self._displays:
                return None
            return self._view(display_id, time.monotonic())

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()
            return [self._view(did, now) for did in self._displays]

    # ---- changes ----
    def hello(self, display_id: str, info: Any = None, name: Any = None) -> dict[str, Any]:
        """A display reporting in. The first report registers it (named
        ``name`` if the kiosk URL gave one, else "Display N"); every report
        refreshes its presence. Returns what it should look like."""
        if not valid_id(display_id):
            raise DisplayError("display_id must be 1-64 letters, digits, - or _")
        info = info if isinstance(info, dict) else {}
        with self._lock:
            if display_id not in self._displays:
                if len(self._displays) >= MAX_DISPLAYS:
                    # Forget the longest-unseen one rather than refuse a TV.
                    oldest = min(self._displays, key=lambda d: (self._presence.get(d) or {}).get("seen_monotonic", 0.0))
                    del self._displays[oldest]
                    self._presence.pop(oldest, None)
                try:
                    label = _clean_name(name)
                except DisplayError:
                    label = self._next_name()
                self._displays[display_id] = {
                    "name": label, "settings": dict(DEFAULT_SETTINGS),
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                self._save()
            # Only small, flat facts about itself: this is shown, not trusted.
            kept = {k: v for k, v in info.items()
                    if isinstance(k, str) and len(k) <= 32
                    and (v is None or isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= 120))}
            self._presence[display_id] = {
                "seen_monotonic": time.monotonic(),
                "seen_at_utc": datetime.now(timezone.utc).isoformat(),
                "info": dict(list(kept.items())[:16]),
            }
            return self._view(display_id, time.monotonic())

    def _next_name(self) -> str:
        taken = {rec["name"] for rec in self._displays.values()}
        n = 1
        while f"Display {n}" in taken:
            n += 1
        return f"Display {n}"

    def update(self, display_id: str, *, name: Any = None, settings: Any = None) -> dict[str, Any]:
        """Rename and/or change settings. A locked display takes no change
        except the one that unlocks it -- a guard against the wrong row
        being tapped, not a security boundary (see Rig › Displays)."""
        with self._lock:
            rec = self._displays.get(display_id)
            if rec is None:
                raise DisplayError("no such display", 404)
            new_settings = rec["settings"]
            if settings is not None:
                new_settings = _clean_settings(settings, rec["settings"])
            unlocking = rec["settings"].get("locked") and new_settings.get("locked") is False
            if rec["settings"].get("locked") and not unlocking:
                raise DisplayError("this display is locked -- unlock it first", 409)
            if unlocking and (name is not None or set(settings or {}) - {"locked"}):
                raise DisplayError("unlock first, then change it", 409)
            new_name = rec["name"] if name is None else _clean_name(name)
            rec["name"] = new_name
            rec["settings"] = new_settings
            self._save()
            return self._view(display_id, time.monotonic())

    def forget(self, display_id: str) -> bool:
        with self._lock:
            if display_id not in self._displays:
                return False
            del self._displays[display_id]
            self._presence.pop(display_id, None)
            self._save()
            return True
