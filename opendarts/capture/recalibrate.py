"""Read a session's offline calibration refit, when one exists.

`load_throw_package()` calls `load_session_refit()` so that replaying a
package prefers a session-level `calibration_refit.json` over the
`calibration.json` frozen inside the package at capture time. That is
the whole of what the product does with refits: it READS them.

WHY A REFIT CAN EXIST AT ALL. `calibration.json` is stored frozen at
capture-era quality and replay consumes it as-is, so every improvement
to the calibration stack is structurally exempt from the REPLAY first
principle (replaying old packages through NEW code must produce updated
results). The raw input needed to re-derive calibration -- the bg frames
-- is already in every package, so a session's cameras CAN be re-solved
from stored images through today's code. Measured on a real session
whose calibration predated the 2026-08-15 wire-junction fix, re-solving
flipped all four of that session's scoring misses to AD's exact
sector+ring and regressed none.

WHO WRITES ONE. `dev/calibration/recalibrate.py`, a developer tool, not
the rig: see its docstring for the measurement, the sampling-bug
correction, and why the capability is not wired into the product today.
It is non-destructive -- the refit is a sibling file, and no package's
own `calibration.json` is ever touched -- so a session either has one or
does not, and this reader answers None when it does not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from opendarts.pipeline import CameraCalibration

REFIT_FILENAME = "calibration_refit.json"


def load_session_refit(session_dir: Path) -> dict[int, CameraCalibration] | None:
    """Load a previously written session refit as engine-ready
    `CameraCalibration`s, or None if the session has no refit file."""
    path = Path(session_dir) / REFIT_FILENAME
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return {
        int(cam): CameraCalibration(
            camera_matrix=np.array(c["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(c["dist_coeffs"], dtype=np.float64),
            rvec=np.array(c["rvec"], dtype=np.float64),
            tvec=np.array(c["tvec"], dtype=np.float64),
            landmark_spread_ok=bool(c.get("landmark_spread_ok", True)),
        )
        for cam, c in payload["cameras"].items()
    }
