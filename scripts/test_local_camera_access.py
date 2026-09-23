#!/usr/bin/env python3
"""scripts/test_local_camera_access.py -- standalone verification for
opendarts/live/local_capture.py's direct cv2.VideoCapture frame path.

*** RUN THIS DIRECTLY ON THE RIG, AT THE PHYSICAL MACHINE. ***
*** DO NOT RUN THIS OVER SSH OR REMOTELY. ***

This needs real, local access to the rig's 3 physical USB cameras,
which no remote dev-machine session has or can reach over the network. That's
the whole point of opendarts/live/local_capture.py existing -- it only
works standing at the rig itself.

What this does: opens all 3 cameras via
opendarts.live.local_capture.LocalCameraHub (real device indices 0/1/2 --
see that module's docstring for where those were confirmed), grabs one
frame from each, prints a rich status line per camera (backend actually
used, negotiated vs. requested resolution/fps, open/first-frame
latency, frame stats), runs a basic sanity check that each frame isn't
all-black/all-one-value (a common real symptom of a camera that
"opened" but isn't actually delivering a live image), and saves each
frame as a PNG under <repo>/tmp/local_camera_test/ so you can eyeball
the result directly. Deliberately simple and fast -- one frame per
camera, no continuous loop.

Usage (on the rig):
    <repo>/.venv/bin/python3 scripts/test_local_camera_access.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from opendarts.live.local_capture import LocalCameraHub  # noqa: E402

OUT_DIR = REPO_ROOT / "tmp" / "local_camera_test"


def _sanity_check(frame: np.ndarray) -> tuple[bool, str]:
    """Fast, basic real-image sanity check -- NOT proof of correctness
    (it can't tell you the dartboard is actually in frame, only that the
    frame isn't a dead/black/stuck buffer), but a real, common failure
    mode worth catching automatically rather than only by eyeballing."""
    std = float(frame.std())
    channel0 = frame[:, :, 0] if frame.ndim == 3 else frame
    n_unique = int(len(np.unique(channel0)))
    if std < 1.0:
        return False, f"looks flat: std={std:.3f}, {n_unique} unique values in channel 0 (likely black/stuck)"
    return True, f"std={std:.2f}, {n_unique} unique values in channel 0 -- looks like a real image"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("opendarts local camera access test -- direct cv2.VideoCapture, run ON THE RIG ONLY")
    print("=" * 72)

    hub = LocalCameraHub()
    t0 = time.monotonic()
    ok_flags = hub.open_all()
    open_total = time.monotonic() - t0

    print()
    print(f"open_all() finished in {open_total:.3f}s -- per-camera results:")
    for i, ok in enumerate(ok_flags):
        tag = "OPENED" if ok else "FAILED"
        print(f"  cam{i}: {tag} -- {hub.status[i].summary()}")

    print()
    print("grabbing one frame per camera...")
    any_problem = False

    for i in range(len(hub.configs)):
        status = hub.status[i]
        if not status.opened:
            print(f"cam{i}: SKIPPED (never opened) -- {status.last_error}")
            any_problem = True
            continue

        frame = hub.grab(i)
        if frame is None:
            print(f"cam{i}: grab() FAILED -- {status.last_error}")
            any_problem = True
            continue

        h, w = frame.shape[:2]
        ok_sanity, sanity_msg = _sanity_check(frame)
        dest = OUT_DIR / f"cam{i}.png"
        cv2.imwrite(str(dest), frame)

        first_frame_latency = (
            f"{status.first_frame_latency_s:.3f}s" if status.first_frame_latency_s is not None else "n/a"
        )
        print(
            f"cam{i}: {w}x{h}  backend={status.backend_used}  "
            f"fps(actual)={status.actual_fps:.1f}  "
            f"open_latency={status.open_latency_s:.3f}s  "
            f"first_frame_latency={first_frame_latency}"
        )
        print(f"      sanity: {'OK' if ok_sanity else 'SUSPECT'} -- {sanity_msg}")
        print(f"      saved: {dest}")
        if not ok_sanity:
            any_problem = True

    hub.close_all()

    print()
    print("=" * 72)
    if any_problem:
        print("RESULT: at least one camera failed to open, failed to grab a frame,")
        print("or produced a suspect (flat/stuck-looking) image -- see lines above.")
        print(f"Check the saved images directly in {OUT_DIR}.")
    else:
        print(f"RESULT: all {len(hub.configs)} cameras opened, grabbed a real-looking")
        print(f"frame, and were saved to {OUT_DIR}.")
        print("Still eyeball the PNGs yourself -- this script's sanity check only")
        print("catches a dead/black/stuck buffer, not whether the dartboard is")
        print("actually in frame, focused, or correctly oriented.")
    print("=" * 72)

    return 1 if any_problem else 0


if __name__ == "__main__":
    sys.exit(main())
