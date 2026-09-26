"""Per-camera reference frames and committed-dart masks.

The *reference* is the residual background every signal is measured
against: the scene as it looked the last time the state machine
decided nothing was happening (startup, after a commit's cooldown,
after a takeout, after a long quiet stretch). It is kept in two forms:

* ``small`` -- downscaled gray, what the change detector compares to;
* ``full``  -- the original BGR frame, what the scoring engines receive
               as ``bg`` when a dart is committed (the board exactly as it
               looked before this dart). With ``detect_from_small_decode``
               this is a LazyFrame (its JPEG), decoded only when a dart
               actually needs it -- adoption happens on nearly every quiet
               frame, and decoding each one would cost the whole saving.

Both are always adopted together from the same frame, so the engine's
``bg`` is by construction the same scene the lifecycle judged against.

The *dart union* is the OR of the change masks recorded at each commit
this visit (small scale, board region only). Takeout is judged by how
much of that union currently differs from the reference -- the darts we
put there are gone -- rather than by resemblance to a startup frame.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from opendarts.capture.lazy_frame import as_frames

_DILATE_KERNEL = np.ones((3, 3), dtype=np.uint8)


@dataclass
class CameraReference:
    small: np.ndarray
    full: Any  # np.ndarray, or a LazyFrame (opendarts.capture.lazy_frame)
    board_mask: np.ndarray
    prev_small: np.ndarray | None = None
    dart_masks: list[np.ndarray] = field(default_factory=list)
    #: OR of the committed darts' change masks -- takeout coverage is
    #: measured against this (the darts' own pixels).
    dart_union: np.ndarray | None = None
    #: the same, dilated by one pixel (4 full-res px): pixels that change
    #: inside it are the darts settling/shadowing, not something new. Used
    #: only to exclude spill from ``board_new``.
    dart_union_grown: np.ndarray | None = None

    def adopt(self, full: np.ndarray, small: np.ndarray) -> None:
        self.full = full
        self.small = small

    def add_dart_mask(self, mask: np.ndarray) -> None:
        board_only = mask & self.board_mask
        grown = cv2.dilate(board_only.astype(np.uint8), _DILATE_KERNEL).astype(bool) & self.board_mask
        self.dart_masks.append(board_only)
        if self.dart_union is None:
            self.dart_union = board_only.copy()
            self.dart_union_grown = grown
        else:
            self.dart_union = self.dart_union | board_only
            self.dart_union_grown = self.dart_union_grown | grown

    def remove_from_union(self, removed: np.ndarray) -> None:
        """Drop pixels that have returned to background from the union
        (partial takeout that persisted long enough to be accepted)."""
        if self.dart_union is None:
            return
        self.dart_union &= ~removed
        if not self.dart_union.any():
            self.dart_union = None
            self.dart_union_grown = None
        else:
            self.dart_union_grown = (
                cv2.dilate(self.dart_union.astype(np.uint8), _DILATE_KERNEL).astype(bool) & self.board_mask
            )

    def clear_darts(self) -> None:
        self.dart_masks.clear()
        self.dart_union = None
        self.dart_union_grown = None


class ReferenceSet:
    """All cameras' references, keyed by camera index."""

    def __init__(self) -> None:
        self.cams: dict[int, CameraReference] = {}

    def __contains__(self, cam: int) -> bool:
        return cam in self.cams

    def __getitem__(self, cam: int) -> CameraReference:
        return self.cams[cam]

    def cameras(self) -> list[int]:
        return sorted(self.cams)

    def seed(self, cam: int, full: np.ndarray, small: np.ndarray, board_mask: np.ndarray) -> None:
        self.cams[cam] = CameraReference(small=small, full=full, board_mask=board_mask)

    def adopt_all(self, fulls: dict[int, np.ndarray], smalls: dict[int, np.ndarray]) -> None:
        for cam, ref in self.cams.items():
            if cam in fulls and cam in smalls:
                ref.adopt(fulls[cam], smalls[cam])

    def adopt_outside(self, smalls: dict[int, np.ndarray]) -> None:
        """Refresh only the outside-board pixels of the small (detection)
        reference. The full-res engine reference is left alone."""
        for cam, ref in self.cams.items():
            small = smalls.get(cam)
            if small is None:
                continue
            patched = ref.small.copy()
            outside = ~ref.board_mask
            patched[outside] = small[outside]
            ref.small = patched

    def full_handles(self) -> dict[int, Any]:
        """Each camera's reference frame as held: an array, or a LazyFrame
        that has not necessarily been decoded. Reading this decodes nothing."""
        return {cam: ref.full for cam, ref in self.cams.items()}

    def bg_full(self) -> "Mapping[int, np.ndarray]":
        """The references as full pixels -- a plain dict when every one is
        an array, else a LazyFrames that decodes on access."""
        return as_frames(self.full_handles())

    def clear_darts(self) -> None:
        for ref in self.cams.values():
            ref.clear_darts()

    def has_darts(self) -> bool:
        return any(ref.dart_union is not None for ref in self.cams.values())
