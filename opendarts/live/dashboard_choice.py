"""Which dashboard ``/`` serves: the new one or the classic one.

One rig-wide switch, flipped from either dashboard, with no restart: the
next load of ``/`` gets the other page, and screens already open are told
to reload (``DASHBOARD_SWITCHED``). Kept as the ``dashboard_ui`` section of
data/config.json when ``snapshot_path`` is given (run_product), in memory
otherwise (tests).

Only which page -- nothing else about either dashboard depends on it. A
display (``/?display``) is always the new page's display view: the classic
dashboard has no such role.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

log = logging.getLogger(__name__)

CHOICES = ("new", "classic")
DEFAULT = "new"


class DashboardChoice:
    def __init__(self, *, snapshot_path: Path | None = None, default: str = DEFAULT) -> None:
        self._lock = threading.Lock()
        self._path = Path(snapshot_path) if snapshot_path is not None else None
        self._ui = default if default in CHOICES else DEFAULT
        if self._path is not None:
            from opendarts.live.config import read_config_section

            saved = read_config_section("dashboard_ui", self._path)
            if saved in CHOICES:
                self._ui = saved
            elif saved is not None:
                log.warning("dashboard_ui %r in %s is not one of %s -- serving %r", saved, self._path, CHOICES, self._ui)

    def get(self) -> str:
        with self._lock:
            return self._ui

    def set(self, ui: str) -> str:
        if ui not in CHOICES:
            raise ValueError(f"dashboard must be one of {CHOICES}, not {ui!r}")
        with self._lock:
            self._ui = ui
            if self._path is not None:
                from opendarts.live.config import write_config_section

                try:
                    write_config_section("dashboard_ui", ui, self._path)
                except Exception:  # noqa: BLE001 -- the switch still holds until a restart
                    log.exception("dashboard_ui: could not persist")
            return self._ui
