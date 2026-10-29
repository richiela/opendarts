"""opendarts.live.native_stderr: libjpeg's padding warning is dropped, other
stderr output is not. Run in a child process -- it rewires fd 2."""
from __future__ import annotations

import os
import subprocess
import sys

CHILD = r"""
import os, time
import cv2, numpy as np
from opendarts.live import native_stderr
native_stderr.install()
# A noisy frame with junk before the end-of-image marker, as the cameras
# send: libjpeg decodes it and prints "Corrupt JPEG data: ... extraneous bytes".
ok, jpg = cv2.imencode(".jpg", np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
data = jpg.tobytes()
padded = data[:-2] + bytes.fromhex("50f404ae50f4") + data[-2:]
for _ in range(3):
    img = cv2.imdecode(np.frombuffer(padded, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape == (64, 64, 3)
os.write(2, b"a real error line\n")
time.sleep(0.3)
print("dropped", native_stderr.dropped_count(), flush=True)
"""


def _run(**env):
    full = {k: v for k, v in os.environ.items() if k != "OPENDARTS_RAW_STDERR"}
    full.update(env)
    return subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60, env=full)


def test_padding_warning_is_dropped_and_other_stderr_passes():
    r = _run()
    assert r.returncode == 0, r.stderr
    assert "Corrupt JPEG data" not in r.stderr
    assert "a real error line" in r.stderr
    assert "dropped 3" in r.stdout


def test_opt_out_keeps_raw_stderr():
    r = _run(OPENDARTS_RAW_STDERR="1")
    assert r.returncode == 0, r.stderr
    assert "Corrupt JPEG data" in r.stderr
    assert "a real error line" in r.stderr
