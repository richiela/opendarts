"""Per-camera, per-frame lifecycle signals.

Everything the state machine decides on is computed here, from three
downscaled grayscale images per camera:

    cur   -- this frame
    ref   -- the live reference (residual background, see reference.py)
    prev  -- the previous frame (for temporal stability)

and one boolean board-region mask (from calibration, via
``opendarts.capture.board_disc.board_disc_mask``).

The change detector is deliberately simple: absolute difference, fixed
threshold, one morphological open, drop tiny components. It runs at
1/``scale`` resolution (default 1/4 -> 320x180 for a 720p camera), so a
full three-camera tick costs well under a millisecond. No blur, no crop,
no shape features -- those belong to the engines, downstream.

Units: every count in :class:`CamSignals` is in *small-scale* pixels.
Thresholds in ``state.py`` are expressed in the same units, and as
fractions of the region they apply to where that matters for
portability across resolutions.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SignalConfig:
    """Change-detector parameters. All in small-scale pixel units."""

    scale: int = 4
    #: gray levels; a pixel counts as changed when |cur - ref| >= this.
    diff_threshold: int = 16
    #: size of the MORPH_OPEN structuring element; 0 disables the open.
    open_kernel: int = 3
    #: connected components smaller than this are dropped as noise.
    min_blob_px: int = 4
    #: the frame-to-frame (stability) mask is NOT opened -- a settling
    #: dart's wobble is a 1-2 px strip at this scale and the open would
    #: erase it; only the tiny-component filter is applied.
    delta_min_blob_px: int = 2

    def for_delta(self) -> "SignalConfig":
        return SignalConfig(
            scale=self.scale,
            diff_threshold=self.diff_threshold,
            open_kernel=0,
            min_blob_px=self.delta_min_blob_px,
            delta_min_blob_px=self.delta_min_blob_px,
        )


DEFAULT_SIGNAL_CONFIG = SignalConfig()


def to_small_gray(frame_bgr: np.ndarray, scale: int = 4) -> np.ndarray:
    """BGR (or already-gray) full-res frame -> uint8 grayscale at 1/scale.

    INTER_AREA averages the source pixels, which both suppresses sensor
    noise and keeps a thin dart shaft from aliasing away, unlike a
    nearest/linear pick.
    """
    if frame_bgr.ndim == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr
    if scale == 1:
        return np.ascontiguousarray(gray)
    h, w = gray.shape[:2]
    return cv2.resize(gray, (w // scale, h // scale), interpolation=cv2.INTER_AREA)


def to_small_mask(mask_full: np.ndarray, small_shape: tuple[int, int]) -> np.ndarray:
    """Boolean full-res mask -> boolean mask at the small shape (h, w)."""
    h, w = small_shape
    small = cv2.resize(
        mask_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    return small.astype(bool)


#: The 8 neighbours of a pixel, not the pixel itself (see change_mask).
_NEIGHBOURS_ONLY = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def change_mask(
    a: np.ndarray, b: np.ndarray, cfg: SignalConfig = DEFAULT_SIGNAL_CONFIG
) -> np.ndarray:
    """Boolean mask of pixels that differ between two small gray images.

    absdiff -> threshold -> MORPH_OPEN -> drop components < min_blob_px.
    """
    diff = cv2.absdiff(a, b)
    _, binary = cv2.threshold(diff, cfg.diff_threshold - 1, 255, cv2.THRESH_BINARY)
    if cfg.open_kernel > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.open_kernel, cfg.open_kernel)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Two exact shortcuts for the component filter, which was over half of
    # a tick's cost on the Pi 5 (a 320x180 connectedComponentsWithStats
    # is ~0.25 ms, and the keep[labels] lookup another ~0.15 ms, twice
    # per camera per tick). Both return exactly the mask the filter
    # below would, for every input:
    #
    #   * After a 3x3 OPEN, every set pixel lies in a set 3x3 square
    #     (the opening is the union of the squares centred on the eroded
    #     pixels), clipped to 2x2 at worst in a corner -- erosion treats
    #     outside the image as set, dilation as unset. So in an image at
    #     least 2 px each way every component already has >= 4 pixels and
    #     a min_blob_px <= 4 drops nothing.
    #   * With min_blob_px == 2 the filter only drops components of ONE
    #     pixel, i.e. set pixels with no set 8-neighbour -- one dilation
    #     with the centre left out of the kernel finds those.
    if (cfg.min_blob_px > 1 and cfg.open_kernel == 3 and cfg.min_blob_px <= 4
            and min(binary.shape[:2]) >= 2):
        return binary.astype(bool)
    if cfg.min_blob_px == 2:
        neighbours = cv2.dilate(
            binary, _NEIGHBOURS_ONLY, borderType=cv2.BORDER_CONSTANT, borderValue=0
        )
        return (binary != 0) & (neighbours != 0)
    if cfg.min_blob_px > 1 and np.any(binary):
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if n > 1:
            keep = stats[:, cv2.CC_STAT_AREA] >= cfg.min_blob_px
            keep[0] = False
            return keep[labels]
    return binary.astype(bool)


def largest_blob_px(mask: np.ndarray) -> int:
    """Area of the largest 8-connected component of a boolean mask."""
    if not mask.any():
        return 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


@dataclass(frozen=True)
class CamSignals:
    """One camera's numbers for one tick. Small-scale pixel counts."""

    #: changed pixels (vs reference) inside the board region
    board_px: int
    #: changed pixels (vs reference) outside the board region
    outside_px: int
    #: changed pixels between this frame and the previous frame, inside the board region
    board_delta_px: int
    #: changed pixels between this frame and the previous frame, outside the board region
    outside_delta_px: int
    #: changed pixels (vs reference) inside the union of committed-dart masks
    in_union_px: int
    #: changed pixels (vs reference) inside the board region but outside that union
    board_new_px: int
    #: the largest connected component of those new board pixels. A dart
    #: is one blob; exposure drift is confetti -- same total, tiny blobs.
    board_new_blob_px: int
    #: size of the committed-dart union (0 when no darts are committed)
    union_px: int
    #: region sizes, for fraction reporting
    board_area_px: int
    outside_area_px: int

    @property
    def delta_px(self) -> int:
        return self.board_delta_px + self.outside_delta_px

    @property
    def board_frac(self) -> float:
        return self.board_px / self.board_area_px if self.board_area_px else 0.0

    @property
    def union_coverage(self) -> float:
        """Fraction of the committed-dart union that currently differs
        from the reference -- ~1.0 once those darts have been removed."""
        return self.in_union_px / self.union_px if self.union_px else 0.0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "board_px": self.board_px,
            "outside_px": self.outside_px,
            "board_delta_px": self.board_delta_px,
            "outside_delta_px": self.outside_delta_px,
            "in_union_px": self.in_union_px,
            "board_new_px": self.board_new_px,
            "board_new_blob_px": self.board_new_blob_px,
            "union_px": self.union_px,
            "union_coverage": round(self.union_coverage, 3),
        }


def compute_signals(
    cur: np.ndarray,
    ref: np.ndarray,
    prev: np.ndarray | None,
    board_mask: np.ndarray,
    dart_union: np.ndarray | None = None,
    dart_union_grown: np.ndarray | None = None,
    cfg: SignalConfig = DEFAULT_SIGNAL_CONFIG,
) -> tuple[CamSignals, np.ndarray]:
    """Compute one camera's :class:`CamSignals`.

    All images are small-scale gray (``to_small_gray``); the masks are
    boolean at the same shape. ``dart_union`` is the committed darts'
    own pixels (coverage is measured against it); ``dart_union_grown``
    is its dilation (changes inside it are excluded from ``board_new``).
    Returns the signals and the cur-vs-ref change mask (the caller uses
    it to build the committed-dart mask at commit time).
    """
    changed = change_mask(cur, ref, cfg)
    board_changed = changed & board_mask
    board_px = int(np.count_nonzero(board_changed))
    outside_px = int(np.count_nonzero(changed)) - board_px
    if prev is not None:
        delta = change_mask(cur, prev, cfg.for_delta())
        board_delta_px = int(np.count_nonzero(delta & board_mask))
        outside_delta_px = int(np.count_nonzero(delta)) - board_delta_px
    else:
        board_delta_px = 0
        outside_delta_px = 0
    if dart_union is not None and dart_union.any():
        union_px = int(np.count_nonzero(dart_union))
        in_union_px = int(np.count_nonzero(board_changed & dart_union))
        exclude = dart_union_grown if dart_union_grown is not None else dart_union
        board_new = board_changed & ~exclude
    else:
        union_px = 0
        in_union_px = 0
        board_new = board_changed
    board_new_px = int(np.count_nonzero(board_new))
    board_new_blob_px = largest_blob_px(board_new)
    board_area = int(np.count_nonzero(board_mask))
    total_area = int(board_mask.size)
    signals = CamSignals(
        board_px=board_px,
        outside_px=outside_px,
        board_delta_px=board_delta_px,
        outside_delta_px=outside_delta_px,
        in_union_px=in_union_px,
        board_new_px=board_new_px,
        board_new_blob_px=board_new_blob_px,
        union_px=union_px,
        board_area_px=board_area,
        outside_area_px=total_area - board_area,
    )
    return signals, changed
