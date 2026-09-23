"""Shared, stateless image-op helpers for the engines' frame-differencing
front-ends (2026-09-05, engine perf pass).

Every engine in `opendarts/engines/` opens with the same shape of pipeline:
gray -> absdiff -> GaussianBlur -> threshold -> ellipse morphology ->
connectedComponentsWithStats -> per-component pixel extraction. Profiling
all four against the real corpus showed 60-77% of each engine's time in
two of those steps run on the FULL 1280x720 frame -- the ellipse
morphology and the per-component `np.nonzero` -- even though the binary
mask they operate on is non-zero inside a bounding box that is a median
3-6% of the frame. The helpers here do the same operations restricted to
that bounding box, with the argument for why each is bit-identical to
the full-frame version spelled out on the function itself.

Design rules for this module:

- Pure functions only. No module state beyond the read-only kernel cache.
  Engines are module-level singletons called concurrently from
  `opendarts.engines.zeus.engine`'s thread pool, so anything here must be
  safe to call from several threads at once on different frames.
- No engine imports, no calibration, no board geometry. This is
  pixel-space arithmetic that every engine may share without any engine
  depending on another's DECISIONS (thresholds, kernel sizes, candidate
  ranking, tip picking all stay in the engine that owns them). Sharing
  the mechanics of "skip computing zeros" is no more a coupling between
  voters than sharing `cv2` itself is.
- Bit-identity to the full-frame equivalent is the contract. Anything
  that would change a result by even one ulp does not belong here.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import cv2
import numpy as np


@functools.lru_cache(maxsize=None)
def ellipse_kernel(size: int) -> np.ndarray:
    """`cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))`, built
    once per size and marked read-only. cv2 never writes to a kernel
    argument, so one shared array is safe across threads."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    kernel.setflags(write=False)
    return kernel


def abs_diff_f32(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """`cv2.absdiff(a.astype(float32), b.astype(float32))`, computed with
    one float32 conversion instead of two when both inputs are uint8.

    Identity: for uint8 inputs |a - b| is in 0..255, so the uint8
    `absdiff` never saturates and every result is exactly representable
    in float32 -- converting the uint8 difference gives the same float32
    array as differencing two float32 conversions. Non-uint8 inputs take
    the original two-conversion path unchanged."""
    if a.dtype == np.uint8 and b.dtype == np.uint8:
        return cv2.absdiff(a, b).astype(np.float32)
    return cv2.absdiff(a.astype(np.float32), b.astype(np.float32))


def blurred_abs_diff(a: np.ndarray, b: np.ndarray, ksize: int) -> np.ndarray:
    """Gaussian-blurred float32 absolute difference -- the pre-threshold
    magnitude image every engine's diff mask is cut from. The blur itself
    is untouched (same float32 input, same `cv2.GaussianBlur(..., 0)`)."""
    return cv2.GaussianBlur(abs_diff_f32(a, b), (ksize, ksize), 0)


def threshold_mask(diff: np.ndarray, threshold: float) -> np.ndarray:
    """`(diff > threshold).astype(np.uint8) * 255` as one GIL-releasing
    `cv2.compare` pass (0/255 uint8) instead of three numpy passes
    (bool compare, bool->uint8 cast, multiply). Same support, same dtype,
    same values. The engines' thresholds are small integers, exactly
    representable in float32, so `>` agrees in both formulations."""
    return cv2.compare(diff, float(threshold), cv2.CMP_GT)


def morph_on_bbox(mask: np.ndarray, op: int, kernel: np.ndarray) -> np.ndarray:
    """`cv2.morphologyEx(mask, op, kernel)` evaluated only inside the
    bounding box of `mask`'s non-zero pixels (padded by the kernel size),
    pasted back into a zeroed full-size buffer. Returns a full-size array
    of `mask`'s shape/dtype -- callers see exactly what the full-frame
    call would have returned. An all-zero `mask` returns all zeros, which
    is also what the full-frame call returns.

    Why the output is bit-identical (`op` in DILATE / ERODE / OPEN /
    CLOSE, any ellipse/rect kernel with the default centred anchor):

    - Let r be the kernel's reach from its anchor (<= kernel size) and
      let B be the bounding box of the non-zero input pixels. Dilation
      can only set pixels within r of a non-zero input pixel, i.e. inside
      B expanded by r; everything outside is zero in both versions.
    - For a two-pass op (close = dilate then erode; open = erode then
      dilate) the second pass reads pixels within r of its own centre,
      so any output pixel within r of B reads only pixels within 2r of B.
      The crop is padded by the full kernel size (>= 2r + 1), so all of
      those reads land on real data inside the crop, never on the
      border. Output pixels farther than r from B are zero after the
      first pass in both versions (their own centre pixel is zero for
      erode; nothing reaches them for dilate), so the second pass gives
      zero there too, regardless of how the crop border is treated.
    - Where the padded crop is clipped by the image edge, the crop's
      border coincides with the image's border, so both versions apply
      the identical default border handling there.
    """
    x, y, w, h = cv2.boundingRect(mask)
    out = np.zeros_like(mask)
    if w == 0 or h == 0:
        return out
    pad = int(max(kernel.shape[0], kernel.shape[1]))
    img_h, img_w = mask.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    out[y0:y1, x0:x1] = cv2.morphologyEx(mask[y0:y1, x0:x1], op, kernel)
    return out


def component_pixels(
    labels: np.ndarray,
    label_id: int,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray | None = None,
    mask_origin: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray]:
    """`ys, xs = np.nonzero((labels == label_id) & (mask != 0))` restricted
    to the label's own bounding box `bbox = (x, y, w, h)` (as returned in
    `cv2.connectedComponentsWithStats`' `stats[label_id, :4]`), with the
    offset added back so the returned `(xs, ys)` are full-frame pixel
    coordinates. `mask=None` means `labels == label_id` alone.

    `mask_origin=(ox, oy)`: `mask` may be a crop of the full frame whose
    top-left pixel sits at full-frame `(ox, oy)` (see `DiffCrop`); the
    label's bbox is then read from `mask` at `(x - ox, y - oy)`. The
    caller guarantees the label lies inside the crop (true whenever the
    labels came from a mask that is zero outside it).

    Identity: every pixel of a label lies inside that label's bounding
    box by definition, so no pixel is lost. `np.nonzero` enumerates a
    C-contiguous array in row-major order, and row-major order restricted
    to a sub-rectangle is the same relative order, so the returned arrays
    hold the same values in the same order as the full-frame call. That
    ordering matters downstream (PCA input order, `argsort` tie-breaking,
    floating-point summation order), which is why this returns the raw
    `(xs, ys)` pair rather than anything derived. Integer offsets on
    integer index arrays are exact."""
    x, y, w, h = (int(v) for v in bbox)
    sub = labels[y:y + h, x:x + w] == label_id
    if mask is not None:
        mx, my = x - int(mask_origin[0]), y - int(mask_origin[1])
        sub &= mask[my:my + h, mx:mx + w] != 0
    ys, xs = np.nonzero(sub)
    if x:
        xs += x
    if y:
        ys += y
    return xs, ys


@dataclass(frozen=True)
class PrecomputeRequirements:
    """What an engine's own front end (gray -> |diff| -> Gaussian blur ->
    threshold) needs from a shared `DiffCrop` for the crop to be a
    drop-in replacement. Engines expose one of these as a class
    attribute; `merge_requirements()` combines them for a caller (Zeus)
    that computes the crop once for several engines."""

    ksize: int            # Gaussian kernel size the engine blurs with
    threshold: float      # engine's own mask threshold on the blurred diff
    pad_px: int           # margin the engine's morphology needs around the mask


def merge_requirements(reqs: "list[PrecomputeRequirements]") -> PrecomputeRequirements | None:
    """One requirement satisfying every entry, or None if they cannot
    share a crop (different blur kernels produce different diff images).
    The threshold is the minimum (the crop's box must contain every
    pixel any consumer will threshold in) and the pad the maximum."""
    if not reqs:
        return None
    ksizes = {int(r.ksize) for r in reqs}
    if len(ksizes) != 1:
        return None
    return PrecomputeRequirements(
        ksize=ksizes.pop(),
        threshold=min(float(r.threshold) for r in reqs),
        pad_px=max(int(r.pad_px) for r in reqs),
    )


@dataclass(frozen=True)
class DiffCrop:
    """The shared front end of the four diff-based engines, computed once
    per camera and cropped to where the frame actually changed.

    `bg_gray` / `frame_gray` are the FULL-frame grayscale images (engines
    read background texture patches anywhere). `diff_blur` is the crop
    `[y0:y0+h, x0:x0+w]` of the full-frame `GaussianBlur(|bg - frame|,
    ksize)` (float32); it is exactly what the engine would have computed
    itself, restricted to a window that contains every pixel above
    `threshold` plus `pad_px` of margin on each side (clipped to the
    frame). All arrays are read-only: one bundle is shared across threads.

    Why cropping the later stages is exact (see `DiffCrop.accepts()` for
    the conditions an engine checks):
    * threshold: pointwise, and every pixel above any threshold >= the
      bundle's lies inside the crop, so the full-frame mask is zero
      outside it;
    * opening / closing / dilation of that mask: each output pixel
      depends on a window of radius r around it; a pixel whose window
      reaches outside the crop is at distance > pad_px - r from the
      mask, so its true output is 0 (erode: it is 0 itself; dilate:
      nothing within r) -- the same as cv2 computes on the crop with its
      default border (which never adds foreground). Hence
      `pad_px >= kernel_size` of the engine's largest morphology step;
    * the crop never touches connected-component labeling: engines paste
      the cropped mask back into a zero full-frame buffer before
      labeling, so label numbering is untouched.
    """

    x0: int
    y0: int
    img_h: int
    img_w: int
    ksize: int
    threshold: float
    pad_px: int
    bg_gray: np.ndarray
    frame_gray: np.ndarray
    diff_blur: np.ndarray

    @property
    def origin(self) -> tuple[int, int]:
        return (self.x0, self.y0)

    @property
    def is_full_frame(self) -> bool:
        return self.diff_blur.shape[:2] == (self.img_h, self.img_w)

    def accepts(self, req: PrecomputeRequirements, frame_shape: tuple[int, ...]) -> bool:
        """Whether an engine with `req` can use this bundle in place of
        its own full-frame front end on a frame of `frame_shape`."""
        return (
            tuple(frame_shape[:2]) == (self.img_h, self.img_w)
            and int(req.ksize) == self.ksize
            and float(req.threshold) >= self.threshold
            and int(req.pad_px) <= self.pad_px
        )

    def paste_full(self, crop_mask: np.ndarray) -> np.ndarray:
        """A full-frame uint8 mask equal to `crop_mask` inside the crop
        and zero elsewhere -- the input connected-component labeling
        needs. Returns `crop_mask` itself when the crop is the whole
        frame."""
        if self.is_full_frame:
            return crop_mask
        h, w = crop_mask.shape[:2]
        full = np.zeros((self.img_h, self.img_w), dtype=crop_mask.dtype)
        full[self.y0:self.y0 + h, self.x0:self.x0 + w] = crop_mask
        return full


def precompute_diff_crop(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    req: PrecomputeRequirements,
) -> DiffCrop | None:
    """Build a `DiffCrop` for one camera, or None when the two images
    are not same-shaped 3-channel BGR (the engine's own mismatch path
    then reports it exactly as before). If nothing in the frame exceeds
    `req.threshold` the crop is the whole frame, so consumers behave
    precisely as they would without a bundle."""
    if bg_bgr is None or frame_bgr is None or bg_bgr.shape != frame_bgr.shape or bg_bgr.ndim != 3:
        return None
    img_h, img_w = bg_bgr.shape[:2]
    bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    diff_blur = blurred_abs_diff(bg_gray, frame_gray, int(req.ksize))
    x, y, w, h = cv2.boundingRect(threshold_mask(diff_blur, float(req.threshold)))
    pad = int(req.pad_px)
    if w == 0 or h == 0:
        x0, y0, x1, y1 = 0, 0, img_w, img_h
    else:
        # Even-aligned origin: keeps the crop on the same 2x2 pixel grid
        # as the full frame (harmless now; a prerequisite for ever
        # labeling on the crop directly).
        x0, y0 = max(0, x - pad) & ~1, max(0, y - pad) & ~1
        x1, y1 = min(img_w, x + w + pad), min(img_h, y + h + pad)
    crop = diff_blur[y0:y1, x0:x1]
    for arr in (bg_gray, frame_gray, diff_blur, crop):
        arr.setflags(write=False)
    return DiffCrop(
        x0=x0, y0=y0, img_h=img_h, img_w=img_w,
        ksize=int(req.ksize), threshold=float(req.threshold), pad_px=pad,
        bg_gray=bg_gray, frame_gray=frame_gray, diff_blur=crop,
    )


def engine_accepts_precomputed(engine: object) -> bool:
    """Duck-typed capability check: does `engine.score()` declare a
    `precomputed` keyword? Mirrors `engine_accepts_prior_dart_line_px()`
    -- test doubles registered under a real engine's name keep the plain
    3-argument signature and must not be handed a kwarg they lack."""
    import inspect

    score = getattr(engine, "score", None)
    if not callable(score):
        return False
    try:
        return "precomputed" in inspect.signature(score).parameters
    except (TypeError, ValueError):
        return False
