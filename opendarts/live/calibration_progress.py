"""How far along a calibration is, for the people waiting on it.

Calibration is the one thing an operator does that takes many seconds with
nothing on screen changing. The work already happens in named stages
(capture, orientation, per-camera solve rounds, best-of-N refinement,
finish -- ``capture_daemon._bootstrap_calibrations_unlocked``); this
records which stage it is in, from the calibration's own thread, so the
server can push it to every screen (``CALIBRATION_PROGRESS``) and a page
can show a real progress bar instead of a spinner.

Purely a report: nothing reads it back to make a decision, and a failure
to record can never touch the calibration itself (every method swallows).
One process runs one calibration at a time (``bootstrap_calibrations``
holds a lock), so one module-level instance is the whole design.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

# The stages, in order, with the share of the bar each one fills. Weights
# are rough wall-clock shares from rig logs: solving dominates, capture
# and orientation are short, refinement is a few fresh captures.
STAGES: tuple[tuple[str, str, float], ...] = (
    ("capture", "Grabbing frames from every camera", 0.10),
    ("orientation", "Finding the board's orientation", 0.15),
    ("solve", "Finding the board in each camera", 0.45),
    ("refine", "Refining each camera", 0.22),
    ("finish", "Saving the calibration", 0.08),
)
_STAGE_INDEX = {key: i for i, (key, _, _) in enumerate(STAGES)}


class CalibrationProgress:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._state: dict[str, Any] = {"active": False}
        self._t0 = 0.0
        self._t_end: float | None = None

    def _bump(self, **changes: Any) -> None:
        self._state.update(changes)
        self._seq += 1

    def start(self, cameras: "list[int] | None" = None) -> None:
        try:
            with self._lock:
                self._t0 = time.monotonic()
                self._t_end = None
                self._state = {
                    "active": True, "stage": "capture", "detail": None, "stage_fraction": 0.0,
                    "cameras": {str(c): "waiting" for c in (cameras or [])},
                    "ok": None, "error": None,
                }
                self._seq += 1
        except Exception:  # noqa: BLE001 -- a report must never break the work
            log.debug("calibration progress: start failed", exc_info=True)

    def stage(self, key: str, detail: "str | None" = None, fraction: float = 0.0) -> None:
        try:
            with self._lock:
                if not self._state.get("active") or key not in _STAGE_INDEX:
                    return
                self._bump(stage=key, detail=detail, stage_fraction=max(0.0, min(1.0, fraction)))
        except Exception:  # noqa: BLE001
            log.debug("calibration progress: stage failed", exc_info=True)

    def camera(self, cam: int, state: str) -> None:
        """``state``: "working", "done" or "failed". During the solve stage
        the stage's own fill is the share of cameras finished."""
        try:
            with self._lock:
                if not self._state.get("active"):
                    return
                cams = dict(self._state.get("cameras") or {})
                cams[str(cam)] = state
                finished = sum(1 for s in cams.values() if s in ("done", "failed"))
                changes: dict[str, Any] = {"cameras": cams}
                if self._state.get("stage") == "solve" and cams:
                    changes["stage_fraction"] = finished / len(cams)
                self._bump(**changes)
        except Exception:  # noqa: BLE001
            log.debug("calibration progress: camera failed", exc_info=True)

    def finish(self, ok: bool, error: "str | None" = None) -> None:
        try:
            with self._lock:
                if not self._state.get("active"):
                    return
                self._t_end = time.monotonic()
                self._bump(active=False, ok=bool(ok), error=error, stage="finish", stage_fraction=1.0)
        except Exception:  # noqa: BLE001
            log.debug("calibration progress: finish failed", exc_info=True)

    @property
    def seq(self) -> int:
        return self._seq

    def snapshot(self) -> dict[str, Any]:
        """The state, plus what a page needs to draw it: the overall
        fraction, each stage's label and whether it is done, and how long
        it has been running."""
        with self._lock:
            st = dict(self._state)
            seq = self._seq
            t0 = self._t0
            t_end = self._t_end
        out: dict[str, Any] = {"seq": seq, "active": bool(st.get("active")), "succeeded": st.get("ok"),
                               "error": st.get("error"), "detail": st.get("detail"),
                               "cameras": st.get("cameras") or {}}
        key = st.get("stage")
        if key is None:
            out.update(stage=None, fraction=0.0, stages=[], elapsed_s=None)
            return out
        idx = _STAGE_INDEX.get(key, 0)
        frac = sum(w for _, _, w in STAGES[:idx]) + STAGES[idx][2] * float(st.get("stage_fraction") or 0.0)
        if st.get("ok"):
            frac = 1.0
        out.update(
            stage=key,
            label=STAGES[idx][1],
            fraction=round(min(1.0, frac), 3),
            elapsed_s=round((t_end or time.monotonic()) - t0, 1) if t0 else None,
            stages=[{"key": k, "label": label,
                     "state": "done" if (i < idx or st.get("ok")) else "now" if i == idx else "todo"}
                    for i, (k, label, _) in enumerate(STAGES)],
        )
        return out


PROGRESS = CalibrationProgress()
