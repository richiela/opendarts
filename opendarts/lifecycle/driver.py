"""Runs a :class:`Lifecycle` inside the live capture loop and records what
it decides.

The driver's ticks are translated by :mod:`opendarts.lifecycle.adapter` into
the trigger state the loop consumes, so commits drive scoring and clears
drive the visit. It never raises into the loop; after ``MAX_ERRORS``
consecutive failures it disables itself (the loop then logs an error and
detects nothing -- there is no fallback trigger).

``config_provider`` (optional) is polled at the start of every
``observe()``; a new frozen ``LifecycleConfig`` identity hot-swaps
``self.cfg`` (and the live ``Lifecycle.cfg``) without rebuilding state,
so the dashboard's Detection-time slider takes effect on the next frame.

Log: one JSON line per interesting tick in ``<log_dir>/lifecycle-<run>.jsonl``
(every tick outside IDLE, every action, and one IDLE heartbeat per
``idle_heartbeat_ticks``). With ``save_commit_frames=True`` (off in the
daemon -- the throw package has the frames) each commit's (bg, frame)
pair is also written to ``<log_dir>/lifecycle-<run>/commit-NNN/``.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from opendarts.lifecycle.state import (
    DEFAULT_CONFIG,
    Action,
    Lifecycle,
    LifecycleConfig,
    Phase,
    Tick,
)

log = logging.getLogger(__name__)

#: always worth a line in the daemon log
_INFO_ACTIONS = frozenset({Action.COMMIT, Action.CLEARED, Action.SCENE_CHANGE, Action.FORCED_ADOPT, Action.READY})
#: everything else (hand enter/exit, partial adopts, READOPT housekeeping)
#: is DEBUG only -- the loop logs hand toggles as trigger-state transitions
#: already, and the JSONL has it all


#: Lifecycle logs kept per rig: the newest files, up to both limits. One is
#: written per capture session with no cap until 2026-09-17 -- one rig
#: made 45 (35 MB) in its first day.
MAX_LOG_FILES = 100
MAX_LOG_BYTES = 1_000_000_000


def _prune_old_logs(log_dir: Path, keep_path: "Path | None" = None) -> None:
    """Delete the oldest lifecycle-*.jsonl files (and their evidence
    folders) beyond MAX_LOG_FILES / MAX_LOG_BYTES. Never raises."""
    import shutil

    try:
        files = sorted(log_dir.glob("lifecycle-*.jsonl"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        kept_bytes = 0
        for n, path in enumerate(files):
            size = path.stat().st_size
            if path == keep_path or (n < MAX_LOG_FILES and kept_bytes + size <= MAX_LOG_BYTES):
                kept_bytes += size
                continue
            path.unlink(missing_ok=True)
            evidence = path.with_suffix("")
            if evidence.is_dir():
                shutil.rmtree(evidence, ignore_errors=True)
    except OSError as exc:
        log.debug("lifecycle: log pruning skipped: %s", exc)


@dataclass
class DriverStats:
    ticks: int = 0
    commits: int = 0
    clears: int = 0
    hand_enters: int = 0
    errors: int = 0
    last_phase: str = ""
    observe_ms_max: float = 0.0
    observe_ms_sum: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticks": self.ticks,
            "commits": self.commits,
            "clears": self.clears,
            "hand_enters": self.hand_enters,
            "errors": self.errors,
            "last_phase": self.last_phase,
            "observe_ms_max": round(self.observe_ms_max, 2),
            "observe_ms_mean": round(self.observe_ms_sum / self.ticks, 3) if self.ticks else 0.0,
        }


class LifecycleDriver:
    """Owns a Lifecycle for the current calibration and writes the log.

    ``masks_provider`` is called every tick and must return the per-camera
    full-resolution board-disc masks (``{cam: bool ndarray}``); when the
    returned dict changes identity/shape (recalibration) the Lifecycle is
    rebuilt. Until it returns a non-empty dict nothing is observed.
    ``config_provider`` is an optional ``() -> LifecycleConfig``; a new
    object identity is swapped onto the live Lifecycle between ticks
    (no rebuild, no reset).
    """

    MAX_ERRORS = 20
    MAX_COMMIT_DIRS = 500

    def __init__(
        self,
        *,
        log_dir: Path,
        masks_provider: Callable[[], dict[int, np.ndarray]],
        config: LifecycleConfig = DEFAULT_CONFIG,
        run_id: str | None = None,
        on_tick: Callable[[Tick], None] | None = None,
        save_commit_frames: bool = False,
        idle_heartbeat_ticks: int = 30,
        summary_interval_s: float = 300.0,
        config_provider: Callable[[], LifecycleConfig] | None = None,
    ) -> None:
        self.cfg = config
        self._config_provider = config_provider
        self._masks_provider = masks_provider
        self._on_tick = on_tick
        self._save_commit_frames = save_commit_frames
        self._heartbeat = idle_heartbeat_ticks
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_path = self.log_dir / f"lifecycle-{self.run_id}.jsonl"
        self.evidence_dir = self.log_dir / f"lifecycle-{self.run_id}"
        self._fh = None
        self._lc: Lifecycle | None = None
        self._masks_sig: tuple | None = None
        self._masks: dict[int, np.ndarray] | None = None
        self._disabled = False
        self.stats = DriverStats()
        self._last_logged_phase: Phase | None = None
        self._idle_since_log = 0
        # periodic "is the loop keeping up" summary
        self._summary_interval = summary_interval_s
        self._window_started: float | None = None
        self._window_ms: list[float] = []
        self._window_counts = (0, 0, 0)  # commits, clears, hand_enters at window start
        self._dropped_total: int | None = None
        self._dropped_at_window_start: int | None = None

    # ------------------------------------------------------------------
    @property
    def lifecycle(self) -> Lifecycle | None:
        return self._lc

    @property
    def disabled(self) -> bool:
        """True once the driver gave up after repeated observe() errors."""
        return self._disabled

    def _open_log(self) -> None:
        if self._fh is None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            _prune_old_logs(self.log_dir, keep_path=self.log_path)
            self._fh = open(self.log_path, "a", buffering=1)
            log.info("lifecycle: logging to %s", self.log_path)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _masks_signature(self, masks: dict[int, np.ndarray]) -> tuple:
        return tuple(sorted((cam, id(m), m.shape) for cam, m in masks.items()))

    def _masks_changed(self, masks: dict[int, np.ndarray]) -> bool:
        """Cheap identity check first; only when array objects differ,
        compare content -- fresh-but-identical arrays are not a change."""
        sig = self._masks_signature(masks)
        if sig == self._masks_sig:
            return False
        if self._masks is not None and masks.keys() == self._masks.keys() and all(
            m.shape == self._masks[c].shape and np.array_equal(m, self._masks[c]) for c, m in masks.items()
        ):
            self._masks_sig = sig
            return False
        return True

    def _ensure_lifecycle(self) -> Lifecycle | None:
        masks = self._masks_provider() or {}
        if not masks:
            return None
        if self._lc is None or self._masks_changed(masks):
            self._lc = Lifecycle(masks, self.cfg)
            self._masks_sig = self._masks_signature(masks)
            self._masks = dict(masks)
            log.info("lifecycle: (re)built for cameras %s", sorted(masks))
            self._write({"event": "lifecycle_built", "cameras": sorted(masks),
                         "config": _config_dict(self.cfg)})
        return self._lc

    def reset(self) -> None:
        if self._lc is not None:
            self._lc.reset()
            self._write({"event": "reset"})

    # ------------------------------------------------------------------
    def observe(
        self,
        frames: dict[int, np.ndarray],
        *,
        extra: dict[str, Any] | None = None,
        dropped_frames_total: int | None = None,
    ) -> Tick | None:
        """Feed one tick. Never raises into the caller: after MAX_ERRORS
        consecutive failures the driver disables itself.

        ``dropped_frames_total`` is the loop's running count of camera pump
        cycles it skipped (it fell behind); it is only reported in the
        periodic summary."""
        if self._disabled:
            return None
        try:
            if self._config_provider is not None:
                new = self._config_provider()
                if new is not self.cfg:
                    self.cfg = new
                    if self._lc is not None:
                        self._lc.cfg = new
                    log.info(
                        "lifecycle: config changed dart_stable_frames=%d",
                        new.dart_stable_frames,
                    )
                    self._write({"event": "config_changed", "config": _config_dict(new)})
            lc = self._ensure_lifecycle()
            if lc is None or not frames:
                return None
            t0 = time.perf_counter()
            tick = lc.observe(frames)
            ms = (time.perf_counter() - t0) * 1000.0
            self._record(tick, ms, extra)
            if self._on_tick is not None:
                self._on_tick(tick)
            self.stats.errors = 0
            if dropped_frames_total is not None:
                self._dropped_total = dropped_frames_total
            self._maybe_summary(ms)
            return tick
        except Exception:
            self.stats.errors += 1
            log.exception("lifecycle: observe failed (%d)", self.stats.errors)
            if self.stats.errors >= self.MAX_ERRORS:
                self._disabled = True
                log.error("lifecycle: disabled after repeated errors")
            return None

    # ------------------------------------------------------------------
    def _maybe_summary(self, observe_ms: float) -> None:
        """One INFO line per ``summary_interval_s``: tick rate, observe()
        cost percentiles, actions, and pump cycles the loop dropped. This
        is the standing answer to "is the loop keeping up with the
        cameras" -- the per-tick numbers live only in the JSONL."""
        now = time.monotonic()
        st = self.stats
        if self._window_started is None:
            self._window_started = now
            self._window_counts = (st.commits, st.clears, st.hand_enters)
            self._dropped_at_window_start = self._dropped_total
        self._window_ms.append(observe_ms)
        elapsed = now - self._window_started
        if elapsed < self._summary_interval:
            return
        ms = np.asarray(self._window_ms, dtype=np.float64)
        c0, k0, h0 = self._window_counts
        dropped = ""
        if self._dropped_total is not None:
            d = self._dropped_total - (self._dropped_at_window_start or 0)
            dropped = f" dropped_pump_cycles={d}"
        log.info(
            "lifecycle %dm summary: ticks=%d (%.1f/s) observe p50/p95/max=%.1f/%.1f/%.1fms "
            "commits=%d clears=%d hands=%d darts_in=%d phase=%s%s",
            round(elapsed / 60.0),
            len(ms),
            len(ms) / elapsed if elapsed > 0 else 0.0,
            float(np.percentile(ms, 50)),
            float(np.percentile(ms, 95)),
            float(ms.max()),
            st.commits - c0,
            st.clears - k0,
            st.hand_enters - h0,
            self._lc.dart_count if self._lc is not None else 0,
            st.last_phase,
            dropped,
        )
        self._write(
            {
                "event": "summary",
                "window_s": round(elapsed, 1),
                "ticks": int(len(ms)),
                "observe_ms": {
                    "p50": round(float(np.percentile(ms, 50)), 2),
                    "p95": round(float(np.percentile(ms, 95)), 2),
                    "max": round(float(ms.max()), 2),
                },
                "commits": st.commits - c0,
                "clears": st.clears - k0,
                "hand_enters": st.hand_enters - h0,
                "dropped_pump_cycles": (
                    self._dropped_total - (self._dropped_at_window_start or 0)
                    if self._dropped_total is not None
                    else None
                ),
            }
        )
        self._window_started = now
        self._window_ms = []
        self._window_counts = (st.commits, st.clears, st.hand_enters)
        self._dropped_at_window_start = self._dropped_total

    # ------------------------------------------------------------------
    def _record(
        self,
        tick: Tick,
        observe_ms: float,
        extra: dict[str, Any] | None,
    ) -> None:
        st = self.stats
        st.ticks += 1
        st.last_phase = tick.phase.value
        st.observe_ms_max = max(st.observe_ms_max, observe_ms)
        st.observe_ms_sum += observe_ms
        if tick.action is Action.COMMIT:
            st.commits += 1
        elif tick.action is Action.CLEARED:
            st.clears += 1
        elif tick.action is Action.HAND_ENTER:
            st.hand_enters += 1

        interesting = (
            tick.action is not Action.NONE
            or tick.phase is not Phase.IDLE
            or self._last_logged_phase is not tick.phase
        )
        if tick.phase is Phase.IDLE and not interesting:
            self._idle_since_log += 1
            if self._idle_since_log < self._heartbeat:
                return
        self._idle_since_log = 0
        self._last_logged_phase = tick.phase

        rec = tick.as_dict()
        rec["t"] = time.time()
        rec["ms"] = round(observe_ms, 2)
        if extra:
            rec["extra"] = extra
        if tick.action is Action.COMMIT and tick.commit is not None:
            rec["commit"] = {
                "dart_index": tick.commit.dart_index,
                "dart_cams": tick.commit.dart_cams,
                "forced": tick.commit.forced,
            }
            if self._save_commit_frames:
                rec["commit"]["evidence"] = self._save_commit(tick)
        self._write(rec)

        if tick.action is Action.NONE:
            return
        # hand enter/exit and partial adopts stay at DEBUG: the loop already
        # logs them as trigger-state transitions
        level = logging.INFO if tick.action in _INFO_ACTIONS else logging.DEBUG
        if log.isEnabledFor(level):
            log.log(
                level,
                "lifecycle: %s -> %s%s (darts=%d) cams=%s",
                tick.action.value,
                tick.phase.value,
                f" [{tick.reason}]" if tick.reason else "",
                tick.dart_count,
                # (board_px | board_new_px | in_union_px | outside_px):
                # the dart gate reads board_px and compares board_new_px
                # against in_union_px -- log what the gate reads. This
                # line used to show only the union-subtracted pair, which
                # sent an investigation chasing numbers the gate does not
                # see.
                {
                    c: (s.board_px, s.board_new_px, s.in_union_px, s.outside_px)
                    for c, s in tick.signals.items()
                },
            )

    def _save_commit(self, tick: Tick) -> str | None:
        if self.stats.commits > self.MAX_COMMIT_DIRS or tick.commit is None:
            return None
        d = self.evidence_dir / f"commit-{self.stats.commits:03d}"
        try:
            d.mkdir(parents=True, exist_ok=True)
            for cam, img in tick.commit.bg.items():
                cv2.imwrite(str(d / f"cam{cam}_bg.png"), img)
            for cam, img in tick.commit.frames.items():
                cv2.imwrite(str(d / f"cam{cam}_frame.png"), img)
            (d / "meta.json").write_text(
                json.dumps(
                    {
                        "tick": tick.n,
                        "dart_index": tick.commit.dart_index,
                        "dart_count_after": tick.dart_count,
                        "dart_cams": tick.commit.dart_cams,
                        "forced": tick.commit.forced,
                        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                        "signals": {str(c): s.as_dict() for c, s in tick.signals.items()},
                    },
                    indent=2,
                )
            )
            return str(d)
        except Exception:
            log.exception("lifecycle: failed to save commit evidence")
            return None

    def _write(self, rec: dict[str, Any]) -> None:
        try:
            self._open_log()
            rec.setdefault("t", time.time())
            self._fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except Exception:
            log.exception("lifecycle: log write failed")


def _config_dict(cfg: LifecycleConfig) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in vars(cfg).items():
        if k == "signals":
            out[k] = dict(vars(v))
        else:
            out[k] = v
    return out
