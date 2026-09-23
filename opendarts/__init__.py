"""opendarts.

Deliberately almost empty. The one thing here is an environment variable
that MUST be set before OpenCV is imported anywhere in the process, and
this module is the only place guaranteed to run first: Python executes a
package's __init__ before any of its submodules, so importing ANY
`opendarts.*` module reaches this code first.

WHY IT CANNOT LIVE NEXT TO THE CAMERA CODE (2026-09-10, a real bug):
`opendarts.live.local_capture` set this immediately above its own
`import cv2` and it did nothing at all. Windows camera opens stayed at
~45s. `opendarts.live.run_product` imports
`opendarts.lifecycle.settings` one line BEFORE it imports
`local_capture`, and that pulls cv2 in transitively -- so by the time
local_capture ran, cv2 had already initialised and read the old value.
A setting that is correct but late is indistinguishable from no setting.
"""
from __future__ import annotations

import os
import platform

# MSMF's hardware-transform pipeline is why opening the cameras took ~45s
# on a Windows laptop against ~2.2-2.3s per camera on the Mac rig -- same
# three cameras, same code, and the opens already run concurrently. With
# it enabled (the default) every VideoCapture open and every cap.set()
# can renegotiate the stream through a hardware topology, seconds each,
# and the capture code issues three set() calls per camera right after
# opening. Disabling it takes the software path, which is what this
# project wants anyway: frames come back as numpy arrays and nothing
# downstream uses a hardware surface.
#
# setdefault, not assignment: an operator who has already chosen a value
# keeps it. Windows only -- the variable is inert elsewhere, and setting
# it there would imply it does something.
if platform.system() == "Windows":
    os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
