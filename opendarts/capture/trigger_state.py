"""The trigger-state contract between the lifecycle and the capture loop.

``opendarts.lifecycle`` decides *when* a dart landed or the board was
cleared; the capture loop (``opendarts.live.capture_daemon``), the live
server and the package writer consume that decision through one small
object, :class:`ThrowTriggerState`, produced every tick by
``opendarts.lifecycle.adapter.LifecycleTriggerAdapter``.

Only what those consumers read lives here:

* ``state`` / ``dart_count`` -- UI events, the READY_TO_CAPTURE branch,
  the visit model.
* ``last_frame`` -- the commit frames the engines score.
* ``true_baseline_frames`` -- the lifecycle's current per-camera
  reference (the board right before this dart on a commit tick).
* ``last_frame_jpegs`` / ``bg_jpegs`` -- the camera's own JPEG bytes
  for exactly those arrays, when the capture loop could prove the pairing
  (see ``opendarts.live.capture_daemon._FrameJpegIndex``). Storage only:
  nothing scores from them.
* ``settle_started_monotonic`` / ``camera_settled_at_monotonic`` /
  ``settle_duration_s`` -- package diagnostics.

The legacy ``opendarts.capture.throw_trigger`` state machine that used to
own these types is gone; there is one lifecycle.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class ThrowState(enum.Enum):
    IDLE = "idle"  # empty or dart-laden board at rest -- waiting for a throw
    MOTION_DETECTED = "motion_detected"  # hand in frame or scene change
    SETTLING = "settling"  # a board change is being judged
    READY_TO_CAPTURE = "ready_to_capture"  # a dart committed this tick -- score it
    TAKEOUT_WAITING = "takeout_waiting"  # visit is full or darts are being pulled


#: A visit is at most this many darts; the fourth "dart" is a takeout.
MAX_DARTS_PER_TURN = 3


@dataclass
class ThrowTriggerState:
    state: ThrowState = ThrowState.IDLE
    dart_count: int = 0
    #: frames the engines score on a READY_TO_CAPTURE tick
    last_frame: dict[int, np.ndarray] | None = None
    #: the lifecycle's per-camera reference at this tick
    true_baseline_frames: dict[int, np.ndarray] | None = None
    #: The camera's own JPEG bytes for the arrays in ``last_frame``, per
    #: camera, set only on a READY_TO_CAPTURE tick and only for a camera
    #: whose bytes were read at the SAME pump cycle as its array (paired
    #: by array identity, never by "the latest bytes"). A camera that is
    #: absent here is normal -- a slot with no JPEG, a reopened one, a frame whose
    #: bytes had already been overwritten -- and only means its package
    #: clip is FFV1 instead of a stream copy.
    last_frame_jpegs: dict[int, bytes] | None = None
    #: The same, for the bg reference the commit is scored against (the
    #: ``bg_frames`` handed to handle_ready_to_capture alongside this
    #: trigger -- NOT ``true_baseline_frames``, which on a commit tick may
    #: already be the post-dart reference).
    bg_jpegs: dict[int, bytes] | None = None
    settle_started_monotonic: float | None = None
    camera_settled_at_monotonic: dict[int, float] = field(default_factory=dict)
    settle_duration_s: float | None = None
