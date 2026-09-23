"""opendarts.imageops -- every helper must be BIT-IDENTICAL to the full-frame
operation it replaces. These tests assert exactly that, on synthetic
masks built to hit the edge cases the identity arguments rely on (empty
mask, blobs touching every image border, blobs closer together than the
kernel, single pixels, even-sized kernels) and, when the local corpus is
present, on real diff masks from real throw packages."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from opendarts import imageops

H, W = 720, 1280
OPS = (cv2.MORPH_DILATE, cv2.MORPH_ERODE, cv2.MORPH_OPEN, cv2.MORPH_CLOSE)
KERNEL_SIZES = (3, 15, 19, 31, 4, 10)  # includes even sizes (asymmetric anchor)


def _sessions_root() -> Path | None:
    """Optional on-disk session corpus. Absent on a fresh clone.

    Set OPENDARTS_SESSIONS to enable the real-mask checks; they skip
    otherwise. Do not hardcode a maintainer home path here.
    """
    raw = os.environ.get("OPENDARTS_SESSIONS")
    if not raw:
        return None
    root = Path(raw).expanduser()
    return root if root.is_dir() else None


def _rng() -> np.random.Generator:
    return np.random.default_rng(20260905)


def _synthetic_masks() -> list[np.ndarray]:
    rng = _rng()
    masks: list[np.ndarray] = []

    empty = np.zeros((H, W), np.uint8)
    masks.append(empty)

    single = np.zeros((H, W), np.uint8)
    single[400, 600] = 255
    masks.append(single)

    corners = np.zeros((H, W), np.uint8)
    corners[0, 0] = corners[0, W - 1] = corners[H - 1, 0] = corners[H - 1, W - 1] = 255
    masks.append(corners)

    for (y, x) in ((0, 640), (H - 1, 640), (360, 0), (360, W - 1)):
        m = np.zeros((H, W), np.uint8)
        cv2.circle(m, (x, y), 12, 255, -1)
        masks.append(m)

    streak = np.zeros((H, W), np.uint8)
    cv2.line(streak, (300, 500), (700, 200), 255, 3)
    masks.append(streak)

    two_close = np.zeros((H, W), np.uint8)
    cv2.circle(two_close, (500, 300), 10, 255, -1)
    cv2.circle(two_close, (540, 300), 10, 255, -1)  # gap 20px < 31px kernel
    masks.append(two_close)

    far_apart = np.zeros((H, W), np.uint8)
    cv2.circle(far_apart, (100, 100), 8, 255, -1)
    cv2.circle(far_apart, (1150, 650), 8, 255, -1)
    masks.append(far_apart)

    noise = (rng.random((H, W)) < 0.002).astype(np.uint8) * 255
    masks.append(noise)

    full = np.full((H, W), 255, np.uint8)
    masks.append(full)

    frame_streak = np.zeros((H, W), np.uint8)
    cv2.line(frame_streak, (0, 0), (W - 1, H - 1), 255, 5)
    masks.append(frame_streak)

    return masks


def _real_masks(limit: int = 12) -> list[np.ndarray]:
    root = _sessions_root()
    if root is None:
        return []
    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    if not sessions:
        return []
    masks: list[np.ndarray] = []
    for throw_dir in sorted(sessions[-1].iterdir()):
        for cam in range(3):
            bg = throw_dir / f"cam{cam}_bg.png"
            fr = throw_dir / f"cam{cam}_frame.png"
            if not (bg.is_file() and fr.is_file()):
                continue
            bg_gray = cv2.cvtColor(cv2.imread(str(bg)), cv2.COLOR_BGR2GRAY)
            fr_gray = cv2.cvtColor(cv2.imread(str(fr)), cv2.COLOR_BGR2GRAY)
            diff = cv2.GaussianBlur(
                cv2.absdiff(bg_gray.astype(np.float32), fr_gray.astype(np.float32)), (5, 5), 0
            )
            mask = (diff > 30.0).astype(np.uint8) * 255
            opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, imageops.ellipse_kernel(3))
            masks.append(opened)
            if len(masks) >= limit:
                return masks
    return masks


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("ksize", KERNEL_SIZES)
def test_morph_on_bbox_matches_full_frame_synthetic(op, ksize):
    kernel = imageops.ellipse_kernel(ksize)
    for mask in _synthetic_masks():
        expected = cv2.morphologyEx(mask, op, kernel)
        got = imageops.morph_on_bbox(mask, op, kernel)
        assert got.dtype == expected.dtype and got.shape == expected.shape
        assert np.array_equal(got, expected)


def test_morph_on_bbox_matches_cv2_dilate_and_returns_new_array():
    kernel = imageops.ellipse_kernel(31)
    mask = _synthetic_masks()[8]  # streak
    expected = cv2.dilate(mask, kernel)
    got = imageops.morph_on_bbox(mask, cv2.MORPH_DILATE, kernel)
    assert np.array_equal(got, expected)
    assert got is not mask and not np.shares_memory(got, mask)


@pytest.mark.parametrize("op", (cv2.MORPH_DILATE, cv2.MORPH_CLOSE))
@pytest.mark.parametrize("ksize", (15, 19, 31))
def test_morph_on_bbox_matches_full_frame_real_masks(op, ksize):
    masks = _real_masks()
    if not masks:
        pytest.skip("local session corpus not present (set OPENDARTS_SESSIONS to enable)")
    kernel = imageops.ellipse_kernel(ksize)
    for mask in masks:
        assert np.array_equal(
            imageops.morph_on_bbox(mask, op, kernel), cv2.morphologyEx(mask, op, kernel)
        )


def test_ellipse_kernel_is_cached_readonly_and_equal_to_cv2():
    k1 = imageops.ellipse_kernel(31)
    k2 = imageops.ellipse_kernel(31)
    assert k1 is k2
    assert not k1.flags.writeable
    assert np.array_equal(k1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    # cv2 must accept the read-only kernel.
    cv2.dilate(np.zeros((8, 8), np.uint8), k1)


def test_abs_diff_f32_matches_two_conversion_path():
    rng = _rng()
    a = rng.integers(0, 256, (H, W), dtype=np.uint8)
    b = rng.integers(0, 256, (H, W), dtype=np.uint8)
    # Force the extremes too.
    a[0, :] = 0
    b[0, :] = 255
    a[1, :] = 255
    b[1, :] = 0
    expected = cv2.absdiff(a.astype(np.float32), b.astype(np.float32))
    got = imageops.abs_diff_f32(a, b)
    assert got.dtype == np.float32
    assert np.array_equal(got, expected)
    # Non-uint8 inputs take the original path unchanged.
    af, bf = a.astype(np.float32) * 1.5, b.astype(np.float32) * 0.25
    assert np.array_equal(imageops.abs_diff_f32(af, bf), cv2.absdiff(af, bf))


def test_blurred_abs_diff_matches_original_chain():
    rng = _rng()
    a = rng.integers(0, 256, (H, W), dtype=np.uint8)
    b = rng.integers(0, 256, (H, W), dtype=np.uint8)
    expected = cv2.GaussianBlur(
        cv2.absdiff(a.astype(np.float32), b.astype(np.float32)), (5, 5), 0
    )
    assert np.array_equal(imageops.blurred_abs_diff(a, b, 5), expected)


@pytest.mark.parametrize("threshold", (30.0, 31.0, 24, 0.0))
def test_threshold_mask_matches_numpy_compare(threshold):
    rng = _rng()
    diff = (rng.random((H, W)) * 60.0).astype(np.float32)
    # Plant exact-threshold values: `>` must exclude them in both versions.
    diff[5, :100] = np.float32(threshold)
    expected = (diff > threshold).astype(np.uint8) * 255
    got = imageops.threshold_mask(diff, threshold)
    assert got.dtype == np.uint8 and got.shape == expected.shape
    assert np.array_equal(got, expected)


def _labels_for(mask: np.ndarray):
    dilated = cv2.dilate(mask, imageops.ellipse_kernel(31))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    return n, labels, stats


@pytest.mark.parametrize("with_mask", (True, False))
def test_component_pixels_matches_full_frame_nonzero(with_mask):
    masks = _synthetic_masks()[1:] + _real_masks(limit=6)
    checked = 0
    for mask in masks:
        n, labels, stats = _labels_for(mask)
        for i in range(1, n):
            if with_mask:
                expected = np.nonzero(mask.astype(bool) & (labels == i))
            else:
                expected = np.nonzero(labels == i)
            exp_ys, exp_xs = expected
            xs, ys = imageops.component_pixels(
                labels, i, tuple(stats[i, :4]), mask if with_mask else None
            )
            assert xs.dtype == exp_xs.dtype and ys.dtype == exp_ys.dtype
            assert np.array_equal(xs, exp_xs)
            assert np.array_equal(ys, exp_ys)
            checked += 1
    assert checked > 20


def test_component_pixels_accepts_bool_mask_and_numpy_int_bbox():
    mask = _synthetic_masks()[9]  # two_close
    n, labels, stats = _labels_for(mask)
    assert n >= 2
    i = 1
    exp_ys, exp_xs = np.nonzero(mask.astype(bool) & (labels == i))
    xs, ys = imageops.component_pixels(labels, i, stats[i, :4], mask.astype(bool))
    assert np.array_equal(xs, exp_xs) and np.array_equal(ys, exp_ys)


# --- DiffCrop: the shared, cropped front end (2026-09-06) ------------------
# The identity being tested: for an engine with (ksize, threshold, pad),
# threshold -> morphology on the crop, pasted back into a zero full frame,
# equals the same chain on the full frame. Morphology chains below are
# the four engines' own (open3+dilate31, close19, close15).

CHAINS = (
    ("open3_dilate31", 3 + 31 // 2, lambda m: cv2.dilate(
        cv2.morphologyEx(m, cv2.MORPH_OPEN, imageops.ellipse_kernel(3)),
        imageops.ellipse_kernel(31))),
    ("close19", 19, lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, imageops.ellipse_kernel(19))),
    ("close15", 15, lambda m: cv2.morphologyEx(m, cv2.MORPH_CLOSE, imageops.ellipse_kernel(15))),
)


def _synthetic_frames() -> list[tuple[np.ndarray, np.ndarray]]:
    """(bg, frame) BGR pairs: a dart-like streak, a streak in a corner, a
    blob touching each border, noise only, identical frames, and a real
    package pair when the corpus is present."""
    rng = _rng()
    pairs = []
    bg = rng.integers(60, 90, (H, W, 3), dtype=np.uint8)
    for (p1, p2) in (((300, 500), (700, 200)), ((0, 0), (60, 40)), ((W - 1, H - 1), (W - 80, H - 30))):
        fr = bg.copy()
        cv2.line(fr, p1, p2, (200, 200, 200), 4)
        fr = np.clip(fr.astype(np.int16) + rng.integers(-40, 41, fr.shape), 0, 255).astype(np.uint8)
        pairs.append((bg, fr))
    for (x, y) in ((640, 0), (640, H - 1), (0, 360), (W - 1, 360)):
        fr = bg.copy()
        cv2.circle(fr, (x, y), 14, (230, 230, 230), -1)
        pairs.append((bg, fr))
    noise_only = np.clip(bg.astype(np.int16) + rng.integers(-45, 46, bg.shape), 0, 255).astype(np.uint8)
    pairs.append((bg, noise_only))
    pairs.append((bg, bg.copy()))
    root = _sessions_root()
    if root is not None:
        sessions = sorted(p for p in root.iterdir() if p.is_dir())
        for throw_dir in sorted(sessions[-1].iterdir())[:2]:
            for cam in range(3):
                b, f = throw_dir / f"cam{cam}_bg.png", throw_dir / f"cam{cam}_frame.png"
                if b.is_file() and f.is_file():
                    pairs.append((cv2.imread(str(b)), cv2.imread(str(f))))
    return pairs


def _reference_chain(bg, fr, ksize, threshold, morph):
    g0 = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    diff = cv2.GaussianBlur(cv2.absdiff(g0.astype(np.float32), g1.astype(np.float32)), (ksize, ksize), 0)
    mask = (diff > threshold).astype(np.uint8) * 255
    return g0, g1, diff, mask, morph(mask)


@pytest.mark.parametrize("name,pad,morph", CHAINS)
@pytest.mark.parametrize("threshold,floor", ((25.0, 25.0), (31.0, 25.0), (30.0, 25.0)))
def test_diff_crop_chain_matches_full_frame(name, pad, morph, threshold, floor):
    req = imageops.PrecomputeRequirements(ksize=5, threshold=floor, pad_px=max(pad, 19))
    engine_req = imageops.PrecomputeRequirements(ksize=5, threshold=threshold, pad_px=pad)
    for bg, fr in _synthetic_frames():
        pc = imageops.precompute_diff_crop(bg, fr, req)
        assert pc is not None and pc.accepts(engine_req, bg.shape)
        g0, g1, diff, mask, morphed = _reference_chain(bg, fr, 5, threshold, morph)
        assert np.array_equal(pc.bg_gray, g0) and np.array_equal(pc.frame_gray, g1)
        x0, y0 = pc.origin
        h, w = pc.diff_blur.shape
        assert np.array_equal(pc.diff_blur, diff[y0:y0 + h, x0:x0 + w])
        # The full-frame mask must be zero outside the crop.
        outside = mask.copy()
        outside[y0:y0 + h, x0:x0 + w] = 0
        assert not outside.any()
        crop_mask = imageops.threshold_mask(pc.diff_blur, threshold)
        assert np.array_equal(crop_mask, mask[y0:y0 + h, x0:x0 + w])
        pasted = pc.paste_full(morph(crop_mask))
        assert pasted.shape == morphed.shape and pasted.dtype == morphed.dtype
        assert np.array_equal(pasted, morphed)
        # Labeling the pasted mask is labeling the full-frame mask.
        n_a, l_a, s_a, _ = cv2.connectedComponentsWithStats(pasted, connectivity=8)
        n_b, l_b, s_b, _ = cv2.connectedComponentsWithStats(morphed, connectivity=8)
        assert n_a == n_b and np.array_equal(l_a, l_b) and np.array_equal(s_a, s_b)
        # component_pixels with a cropped mask and origin == full-frame call.
        for i in range(1, n_a):
            exp_ys, exp_xs = np.nonzero(mask.astype(bool) & (l_a == i))
            xs, ys = imageops.component_pixels(l_a, i, tuple(s_a[i, :4]), crop_mask, pc.origin)
            assert np.array_equal(xs, exp_xs) and np.array_equal(ys, exp_ys)


def test_diff_crop_arrays_are_read_only_and_origin_even():
    bg, fr = _synthetic_frames()[0]
    pc = imageops.precompute_diff_crop(bg, fr, imageops.PrecomputeRequirements(5, 25.0, 19))
    for arr in (pc.bg_gray, pc.frame_gray, pc.diff_blur):
        assert not arr.flags.writeable
    assert pc.x0 % 2 == 0 and pc.y0 % 2 == 0
    assert pc.diff_blur.dtype == np.float32
    assert not pc.is_full_frame  # a single streak crops well below the frame


def test_diff_crop_is_whole_frame_when_nothing_changed_and_none_on_mismatch():
    bg = _synthetic_frames()[0][0]
    pc = imageops.precompute_diff_crop(bg, bg.copy(), imageops.PrecomputeRequirements(5, 25.0, 19))
    assert pc.is_full_frame and pc.origin == (0, 0)
    m = np.zeros((H, W), np.uint8)
    assert pc.paste_full(m) is m
    assert imageops.precompute_diff_crop(bg, bg[:-1], imageops.PrecomputeRequirements(5, 25.0, 19)) is None
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    assert imageops.precompute_diff_crop(gray, gray, imageops.PrecomputeRequirements(5, 25.0, 19)) is None


def test_diff_crop_accepts_rejects_mismatched_requirements():
    bg, fr = _synthetic_frames()[0]
    pc = imageops.precompute_diff_crop(bg, fr, imageops.PrecomputeRequirements(5, 25.0, 19))
    ok = imageops.PrecomputeRequirements(5, 30.0, 15)
    assert pc.accepts(ok, bg.shape)
    assert not pc.accepts(imageops.PrecomputeRequirements(7, 30.0, 15), bg.shape)   # other blur
    assert not pc.accepts(imageops.PrecomputeRequirements(5, 20.0, 15), bg.shape)   # lower threshold
    assert not pc.accepts(imageops.PrecomputeRequirements(5, 30.0, 20), bg.shape)   # needs more pad
    assert not pc.accepts(ok, (H - 1, W, 3))                                        # other frame size


def test_merge_requirements():
    r = imageops.merge_requirements([
        imageops.PrecomputeRequirements(5, 25.0, 18),
        imageops.PrecomputeRequirements(5, 31.0, 19),
        imageops.PrecomputeRequirements(5, 30.0, 15),
    ])
    assert r == imageops.PrecomputeRequirements(5, 25.0, 19)
    assert imageops.merge_requirements([]) is None
    assert imageops.merge_requirements([
        imageops.PrecomputeRequirements(5, 25.0, 18), imageops.PrecomputeRequirements(7, 25.0, 18),
    ]) is None


def test_engine_accepts_precomputed_capability_check():
    class WithKw:
        def score(self, bg, fr, cal, *, precomputed=None): ...

    class Without:
        def score(self, bg, fr, cal): ...

    assert imageops.engine_accepts_precomputed(WithKw())
    assert not imageops.engine_accepts_precomputed(Without())
    assert not imageops.engine_accepts_precomputed(object())
