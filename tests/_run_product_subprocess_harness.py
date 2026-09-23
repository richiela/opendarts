"""Subprocess harness for tests/test_run_product.py's real-shutdown
regression test -- NOT a test file itself (no test_ functions, pytest
won't collect it), just the script the test launches via
`subprocess.Popen([sys.executable, THIS_FILE, ...])` so the shutdown
hang this session found (and fixed) can be exercised against a REAL OS
process + REAL delivered SIGINT + a REAL WebSocket connection, not an
in-process monkeypatch that can't see process-level hangs.

Runs the real `opendarts.live.run_product.main()` CLI entrypoint with
exactly two things swapped for fakes, both via this project's own
existing, already-tested seams:
  1. `cv2.VideoCapture` -> an in-memory fake camera (same shape as
     tests/test_local_capture.py's own FakeVideoCapture), so
     LocalCameraHub.open_all() succeeds with no real hardware.
  2. `opendarts.live.run_product.run_capture_loop_body` -> a trivial loop
     that just waits on stop_event (same shape as
     tests/test_run_product.py's own fake_loop_body), so the capture
     thread doesn't immediately crash on "no camera calibrated" -- the
     fake frames have no real board/landmarks to calibrate against,
     which is irrelevant to what this harness actually exercises:
     uvicorn's real graceful-shutdown behavior with a real WebSocket
     connection open, real signal handling, real process exit.

Everything else is the real thing: real uvicorn.Server, real FastAPI
app via create_app(), real /api/events websocket route, real
SIGINT/SIGTERM handling via run_product's own _install_signal_handlers +
shutdown() + shutdown watchdog.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np


class _FakeVideoCapture:
    def __init__(self, device, backend):
        self.device = device
        self.backend = backend
        self._opened = True

    def isOpened(self):  # noqa: N802 -- matches cv2's own method name
        return self._opened

    def release(self):
        self._opened = False

    def set(self, prop, value):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 1280.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 720.0
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def read(self):
        frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        return True, frame


cv2.VideoCapture = _FakeVideoCapture

from opendarts.live import run_product  # noqa: E402 -- must follow the cv2.VideoCapture patch above


def _fake_loop_body(
    *, hub, package_root, poll_interval_s, stop_event, on_event=None
):
    # Mirrors tests/test_run_product.py's own fake_loop_body exactly --
    # real capture/calibration/scoring machinery is out of scope here
    # (already covered by tests/test_capture_daemon.py); this harness
    # exists to exercise uvicorn/signal shutdown behavior only.
    while not stop_event.is_set():
        stop_event.wait(0.05)


run_product.run_capture_loop_body = _fake_loop_body

if __name__ == "__main__":
    sys.exit(run_product.main(sys.argv[1:]))
