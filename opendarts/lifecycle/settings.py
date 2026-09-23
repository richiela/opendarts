"""Operator-tunable lifecycle settings, persisted across process restarts.

``LifecycleSettingsStore`` is the same thread-safe, optional-snapshot
holder as ``EngineConfigStore`` in ``opendarts.live.capture_daemon``: a
lock, ``get()`` handing back the same frozen ``LifecycleConfig`` until
``set()`` replaces it (identity is the driver's change detection),
partial ``set()``, and ``meta()`` for the dashboard. The snapshot JSON
only stores the keys an operator can actually change; everything else
stays on the constructor's ``LifecycleConfig``.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path

from opendarts.lifecycle.state import DEFAULT_CONFIG, LifecycleConfig

log = logging.getLogger(__name__)

DART_STABLE_FRAMES_MIN = 1
# The dashboard's "Detection time" control: 1 (fastest commit) .. 5
# (most evidence before a commit, fewest ghosts).
DART_STABLE_FRAMES_MAX = 5


class LifecycleSettingsStore:
    """Thread-safe holder for the live ``LifecycleConfig``.

    ONE instance per ``opendarts.live.run_product`` process, built before
    the capture loop starts and shared with ``AppState``. Pass
    ``snapshot_path`` to load ``latest.json`` at construction and rewrite
    it on every successful ``set()`` (best-effort -- a disk hiccup must
    not break the dashboard POST). ``None`` (the default) stays
    in-memory, matching tests and any caller that doesn't pass a path.
    """

    def __init__(
        self,
        config: LifecycleConfig = DEFAULT_CONFIG,
        *,
        snapshot_path: Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._snapshot_path = Path(snapshot_path) if snapshot_path is not None else None
        self._config = config
        if self._snapshot_path is not None:
            loaded = self._load_snapshot(config)
            if loaded is not None:
                self._config = loaded

    def _load_snapshot(self, base: LifecycleConfig) -> LifecycleConfig | None:
        """Best-effort read. Missing/corrupt/out-of-range falls back to
        ``base`` rather than crashing process startup."""
        assert self._snapshot_path is not None
        from opendarts.live.config import read_config_section

        try:
            raw = read_config_section("lifecycle_settings", self._snapshot_path)
            if raw is None:
                return None
            n = raw["dart_stable_frames"]
            if not isinstance(n, int) or isinstance(n, bool) or not (
                DART_STABLE_FRAMES_MIN <= n <= DART_STABLE_FRAMES_MAX
            ):
                log.warning(
                    "lifecycle settings snapshot %s has out-of-range "
                    "dart_stable_frames=%r, ignoring",
                    self._snapshot_path,
                    n,
                )
                return None
            return dataclasses.replace(base, dart_stable_frames=n)
        except Exception as exc:  # noqa: BLE001 -- a bad snapshot must not break startup
            log.warning(
                "failed to load lifecycle settings from %s: %s",
                self._snapshot_path,
                exc,
            )
            return None

    def _save_snapshot(self, cfg: LifecycleConfig) -> None:
        if self._snapshot_path is None:
            return
        from opendarts.live.config import write_config_section

        try:
            write_config_section(
                "lifecycle_settings",
                {"dart_stable_frames": cfg.dart_stable_frames},
                self._snapshot_path,
            )
        except Exception as exc:  # noqa: BLE001 -- best-effort, same as EngineConfigStore
            log.warning(
                "failed to persist lifecycle settings to %s: %s",
                self._snapshot_path,
                exc,
            )

    def get(self) -> LifecycleConfig:
        with self._lock:
            return self._config

    def set(self, *, dart_stable_frames: int | None = None) -> LifecycleConfig:
        """Partial update. ``None``-only is a no-op that returns the
        current instance. Raises ``ValueError`` if ``dart_stable_frames``
        is not an int in ``[DART_STABLE_FRAMES_MIN, DART_STABLE_FRAMES_MAX]``."""
        with self._lock:
            if dart_stable_frames is None:
                return self._config
            if not isinstance(dart_stable_frames, int) or isinstance(dart_stable_frames, bool) or not (
                DART_STABLE_FRAMES_MIN <= dart_stable_frames <= DART_STABLE_FRAMES_MAX
            ):
                raise ValueError(
                    f"dart_stable_frames must be an int in "
                    f"[{DART_STABLE_FRAMES_MIN}, {DART_STABLE_FRAMES_MAX}], "
                    f"got {dart_stable_frames!r}"
                )
            self._config = dataclasses.replace(
                self._config, dart_stable_frames=dart_stable_frames
            )
            self._save_snapshot(self._config)
            return self._config

    def meta(self) -> dict:
        with self._lock:
            return {
                "dart_stable_frames": self._config.dart_stable_frames,
                "min": DART_STABLE_FRAMES_MIN,
                "max": DART_STABLE_FRAMES_MAX,
            }
