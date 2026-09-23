"""Real landmark detection: finding the board's double-ring outer quad
(the "realistic ~4 independent points" per camera that are the actual
real-world calibration input) in an ACTUAL real camera image -- not
synthetic data.

This was long the single hardest unsolved piece. The candidate technique
is Hough-line
intersection + ellipse fitting on the board's double ring, a standard
technique for this kind of problem (see docs/DESIGN.md).

What this module actually does (the "ellipse fitting" half of that
technique; NOT full Hough-line radial-wire detection -- see "Known
limitations" below):

1. Color-segment the board's red+green double/treble rings via HSV
   thresholds (tuned against real `bg_cam*.png` images).
2. Separate the OUTER ring (double) from the INNER ring (treble) via
   connected-component size (the double ring's bounding box is always
   larger in every real image checked).
3. Trace the double ring's OUTER boundary specifically (not its inner
   edge, not its centerline) by taking, in each of 720 angular bins
   around a rough center estimate, only the mask pixel at MAXIMUM
   radius -- this directly targets the true outer edge rather than
   letting inner-edge points bias a convex-hull fit inward.
4. Fit an ellipse to those boundary points (`cv2.fitEllipse`), then
   robustify with iterative algebraic-residual outlier rejection (a
   cheap stand-in for full RANSAC, adequate at this data quality).
5. Sample 4 representative points off the CLEAN FITTED ellipse (not raw
   pixels) at evenly-spaced parametric angles -- using the fitted model
   rather than raw pixels is what keeps the output usable even in
   angular regions where raw segmentation is locally corrupted (see
   "Known limitations").

Known limitations, measured not guessed against a full real-image
evaluation:

- **No sector correspondence.** This module answers "where is the
  double ring's outer boundary in this image", not "which board sector
  does each detected point belong to". A real PnP solve needs matched
  2D<->3D correspondences (see opendarts/geometry/board.py
  wire_intersection_landmarks()) -- establishing that correspondence
  (e.g. via radial-wire Hough-line detection intersected with this
  ellipse, or a one-time per-camera orientation constant for this
  specific fixed-pose rig) is NOT solved here and is real, separate
  future work. `opendarts/live/run_live_calibration_test.py`'s
  `detect_landmarks()` stub cannot be safely filled in with this
  module's output alone for that reason -- see its docstring.
- **Printed board branding used to locally corrupt segmentation under
  the fixed-HSV path -- FIXED 2026-08-21 for images where the adaptive
  method is confident.** Real boards print manufacturer text/logos
  directly over parts of the double ring (confirmed by directly viewing
  a real image crop, see docs/DESIGN.md). Under the OLD fixed-HSV-only
  segmentation, the true colored ring pixels there were physically
  occluded by white text and the fitted ellipse was measurably less
  accurate in that one angular region even after outlier rejection
  (measured across the FULL real dataset available, 360 real images,
  1440 reference-point comparisons: overall median error 3.0px, p90
  12.8px, but cam0's `src[2]` -- where the "BLADES" logo overlaps the
  ring -- alone had median 15.7px / max 83.7px). `detect_double_ring_quad()`
  now tries `opendarts.calibration.adaptive_ring_color.adaptive_ring_color_mask()`
  first (see `_mask_for_detection()`'s own docstring) -- re-measured on
  the same `src[2]` reference point across a 5-image sample:
  0.17-0.61px, adaptive segmentation
  confidently accepted on every one. This is a genuine fix, not a
  reformulated caveat -- but ONLY for images where the adaptive method's
  own confidence gate accepts (measured 114/117 = 97.4% on the full
  corpus, see that module's own docstring); an image where it refuses
  still falls back to this exact fixed-HSV path and can still show this
  original failure mode.
- **Lens distortion is not modeled.** The real rig shows visible
  barrel/fisheye-style distortion; a
  distorted circle is not exactly an ellipse, so `cv2.fitEllipse` is
  itself only an approximation of the true boundary shape, on top of
  (and not fully separable from) the branding-occlusion issue above.
- **Tuned on one physical rig / one session's lighting.** HSV thresholds
  below were tuned against the real bg_cam*.png images available at
  prototyping time (one session, 3 cameras, ~120 frames each -- see
  docs/DESIGN.md). Different lighting or a different physical board could
  need different thresholds; nothing here is normalized against
  exposure/white-balance.
- **"Largest color component is the double ring" has no independent
  sanity check.** `_outer_ring_component_mask` just picks the largest
  connected color blob after closing -- true for every one of the 360
  real images tested, but nothing checks
  that the chosen blob is actually ring-shaped or board-centered. A
  future image with a large stray red/green object in frame (clothing,
  another light) could silently pick the wrong blob with no error
  raised. Not hit in practice yet, but a real gap, not a proven-safe
  case.
- **No sub-image validated against a background image not already in
  this dataset.** Every number above comes from the same single real
  session (docs/DESIGN.md: "only one session available in goldens" at
  prototyping time). Generalization to a different physical setup/day
  is untested.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Tuned constants -------------------------------------------------
#
# HSV thresholds for the board's red/green ring segments. Tuned by
# direct inspection of real bg_cam0/1/2.png images -- not guessed
# defaults. OpenCV hue range is 0-179; red
# wraps around 0/179 so two ranges are ORed together.
GREEN_HSV_LOW = (35, 60, 40)
GREEN_HSV_HIGH = (85, 255, 255)
RED_HSV_LOW_1 = (0, 60, 40)
RED_HSV_HIGH_1 = (10, 255, 255)
RED_HSV_LOW_2 = (170, 60, 40)
RED_HSV_HIGH_2 = (179, 255, 255)

# Morphological close kernel used only to merge each ring's ~20 sector
# wedges into one connected component for outer/inner-ring separation
# (see _outer_ring_component) -- NOT used for the boundary trace itself,
# which reads the original (unclosed) mask.
CLOSE_KERNEL_PX = 9

ANGULAR_BINS = 720
MIN_MASK_PIXELS = 50
MIN_BOUNDARY_POINTS_FOR_ELLIPSE = 5

ROBUST_FIT_ITERS = 3
ROBUST_FIT_RESIDUAL_THRESH = 1.15
ROBUST_FIT_MIN_POINTS = 20


@dataclass
class Ellipse:
    """A fitted ellipse in image pixel coordinates -- thin wrapper around
    cv2.fitEllipse's ((cx, cy), (major, minor), angle_deg) tuple so
    callers outside this module don't need to remember that shape."""

    cx: float
    cy: float
    major_axis_px: float
    minor_axis_px: float
    angle_deg: float

    @classmethod
    def from_cv2(cls, ellipse_tuple) -> "Ellipse":
        (cx, cy), (major, minor), angle = ellipse_tuple
        return cls(cx, cy, major, minor, angle)

    def to_cv2(self):
        return ((self.cx, self.cy), (self.major_axis_px, self.minor_axis_px), self.angle_deg)

    def boundary_samples(self, n: int = 2000) -> np.ndarray:
        """n points evenly spaced by parametric angle along this ellipse's
        boundary, shape (n, 2)."""
        a, b = self.major_axis_px / 2.0, self.minor_axis_px / 2.0
        t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        ex = a * np.cos(t)
        ey = b * np.sin(t)
        theta = np.deg2rad(self.angle_deg)
        ct, st = np.cos(theta), np.sin(theta)
        rot = np.array([[ct, -st], [st, ct]])
        # np.errstate: this 2x2 @ 2xN matmul has been observed to raise
        # spurious "divide by zero"/"invalid value" FP warnings on some
        # numpy+Accelerate (macOS ARM BLAS) builds even though no actual
        # NaN/Inf appears in the (verified-finite) output -- confirmed by
        # direct inspection, not silencing a
        # real problem.
        with np.errstate(all="ignore"):
            pts = (rot @ np.vstack([ex, ey])).T
        pts[:, 0] += self.cx
        pts[:, 1] += self.cy
        return pts

    def sample_quad(self, start_angle_deg: float = 0.0) -> np.ndarray:
        """4 points on this ellipse's boundary at parametric angles
        start, start+90, start+180, start+270 -- a "representative quad"
        spread maximally around the ring, matching the spirit of the
        well-spread-quad requirement that matters for PnP conditioning. These are NOT
        tied to any particular board sector -- see module docstring
        "No sector correspondence"."""
        a, b = self.major_axis_px / 2.0, self.minor_axis_px / 2.0
        theta = np.deg2rad(self.angle_deg)
        ct, st = np.cos(theta), np.sin(theta)
        rot = np.array([[ct, -st], [st, ct]])
        out = []
        for k in range(4):
            t = np.deg2rad(start_angle_deg + 90.0 * k)
            ex, ey = a * np.cos(t), b * np.sin(t)
            p = rot @ np.array([ex, ey])
            out.append((p[0] + self.cx, p[1] + self.cy))
        return np.array(out, dtype=np.float64)


@dataclass
class LandmarkDetectionResult:
    ok: bool
    ellipse: Ellipse | None
    quad_points_px: np.ndarray | None  # (4, 2), see Ellipse.sample_quad
    n_raw_boundary_points: int
    n_inlier_boundary_points: int
    reason: str = ""
    # Additive, backward-compatible: the ACTUAL robustified (post-outlier-
    # rejection) traced boundary pixels the ellipse was fit from, (N, 2)
    # float32, or None if detection failed. Exposed so a caller can match
    # a real, per-image detected boundary pixel near an expected direction
    # instead of only ever reading an interpolated point off the smoothed
    # ellipse MODEL -- see opendarts/calibration/active_landmarks.py, which is
    # the reason this field was added (2026-08-13, landmark-detection task).
    # `detect_double_ring_quad()` already computes these internally
    # (`kept_pts`); this just stops throwing them away.
    boundary_points_px: np.ndarray | None = None


def _color_mask(image_bgr: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
    red1 = cv2.inRange(hsv, RED_HSV_LOW_1, RED_HSV_HIGH_1)
    red2 = cv2.inRange(hsv, RED_HSV_LOW_2, RED_HSV_HIGH_2)
    return green | red1 | red2


def _outer_ring_component_mask(mask: np.ndarray) -> np.ndarray | None:
    """Of the two ring-shaped color blobs (double outer ring + treble
    inner ring) present in `mask`, return a boolean mask of just the
    OUTER (double) ring's pixels -- identified as the largest connected
    component after a morphological close merges each ring's ~20 sector
    wedges into one blob. Returns None if no plausible ring component is
    found (e.g. mask is empty/near-empty)."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return None
    # label 0 is background; pick the largest non-background component by
    # area, which is the double ring in every real image checked, since
    # it has a strictly larger bounding
    # box/area than the treble ring or any background noise blob.
    areas = stats[1:, cv2.CC_STAT_AREA]
    outer_label = 1 + int(np.argmax(areas))
    return labels == outer_label


def _trace_outer_boundary(mask: np.ndarray, component_mask: np.ndarray, nbins: int = ANGULAR_BINS) -> np.ndarray | None:
    """Within `component_mask`, take the true (unclosed) mask pixels and,
    per angular bin around their centroid, keep only the max-radius
    pixel -- directly traces the ring's true OUTER edge (see module
    docstring step 3)."""
    region = mask.astype(bool) & component_mask
    ys, xs = np.nonzero(region)
    if len(xs) < MIN_MASK_PIXELS:
        return None
    cx, cy = xs.mean(), ys.mean()
    ang = np.arctan2(ys - cy, xs - cx)
    r = np.hypot(xs - cx, ys - cy)
    bin_idx = ((ang + np.pi) / (2.0 * np.pi) * nbins).astype(int) % nbins
    best_r = np.full(nbins, -1.0)
    best_pt = np.zeros((nbins, 2))
    # Max per bin, genuinely vectorized (2026-09-11 calibration-speed
    # pass -- the old code here SAID "vectorized" but looped in Python
    # over every mask pixel, ~33k iterations per frame on this rig's
    # real ring masks). Equivalence with the old loop, argued exactly:
    # iterating `order` (radius ascending), every element of a bin
    # satisfied `r[i] >= best_r[b]` (each is >= all its bin's earlier,
    # smaller-or-equal radii), so EVERY element overwrote and the final
    # winner was simply the bin's LAST element in `order` -- including
    # tie behaviour, which was already defined by argsort's own
    # permutation, reused unchanged here. NumPy fancy assignment with
    # duplicate indices stores values in index-array order with the last
    # write winning (documented indexing behaviour), which reproduces
    # that exactly.
    order = np.argsort(r)
    sorted_bins = bin_idx[order]
    best_r[sorted_bins] = r[order]
    best_pt[sorted_bins, 0] = xs[order]
    best_pt[sorted_bins, 1] = ys[order]
    valid = best_r > 0
    if valid.sum() < MIN_BOUNDARY_POINTS_FOR_ELLIPSE:
        return None
    return best_pt[valid].astype(np.float32)


def _algebraic_residual(ellipse_tuple, pts: np.ndarray) -> np.ndarray:
    (cx, cy), (major, minor), angle = ellipse_tuple
    a, b = major / 2.0, minor / 2.0
    theta = np.deg2rad(angle)
    ct, st = np.cos(theta), np.sin(theta)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    xr = ct * dx + st * dy
    yr = -st * dx + ct * dy
    return (xr / a) ** 2 + (yr / b) ** 2


def _robust_fit_ellipse(
    pts: np.ndarray,
    iters: int = ROBUST_FIT_ITERS,
    thresh: float = ROBUST_FIT_RESIDUAL_THRESH,
    min_points: int = ROBUST_FIT_MIN_POINTS,
):
    """cv2.fitEllipse with iterative algebraic-residual outlier
    rejection: points whose normalized squared distance from the current
    fit falls outside [1/thresh, thresh] are dropped and the ellipse is
    refit, up to `iters` times. A cheap stand-in for full RANSAC --
    adequate here because the outliers this needs to reject (branding
    text notches, stray background color pixels) are a small minority of
    a large, otherwise-clean point set. Returns (ellipse_tuple, kept_pts)
    or (None, pts) if fewer than 5 points are available to fit at all."""
    import cv2

    cur = pts.copy()
    if len(cur) < 5:
        return None, cur
    ellipse = cv2.fitEllipse(cur)
    for _ in range(iters):
        (_, _), (major, minor), _ = ellipse
        if not (np.isfinite(major) and np.isfinite(minor)) or major <= 1e-6 or minor <= 1e-6:
            break
        val = _algebraic_residual(ellipse, cur)
        keep = (val > 1.0 / thresh) & (val < thresh)
        if keep.sum() < min_points:
            break
        cur = cur[keep]
        if len(cur) < 5:
            break
        refit = cv2.fitEllipse(cur)
        (_, _), (refit_major, refit_minor), _ = refit
        if (
            not (np.isfinite(refit_major) and np.isfinite(refit_minor))
            or refit_major <= 1e-6
            or refit_minor <= 1e-6
        ):
            # A degenerate refit (near-collinear surviving points) is
            # worse than the ellipse we already had -- keep the last
            # good one instead of propagating NaN/inf downstream (see
            # Ellipse.boundary_samples, which divides by the axis
            # lengths).
            break
        ellipse = refit
    return ellipse, cur


# ROI-restricted second pass (2026-09-03, cam0 seed-ellipse fragmentation
# fix -- see `_mask_for_detection()`'s own updated docstring below for
# the full mechanism/root-cause). `_board_roi_from_mask()` union-boxes
# the top few closed components of pass 1's WHOLE-FRAME mask -- real
# ring components (double+treble combined, after the SAME morphological
# close `_outer_ring_component_mask()` already uses to merge each ring's
# ~20 sector wedges into one blob) measure 6000-15000px on this rig's
# real images -- nowhere near this floor, which exists only to reject
# tiny noise specks from ever contributing to the ROI box.
BOARD_ROI_MIN_COMPONENT_AREA_PX = 300
BOARD_ROI_MAX_COMPONENTS = 4
# Measured INSENSITIVE, not a tuned threshold: identical real-frame
# outcomes at 0.15/0.30/0.50 padding fraction on this rig's own real
# corpus (see the cam0 ROI-fix investigation this constant implements) --
# 0.30 is a reasonable middle-of-the-insensitive-range pick, not a value
# that needed fitting.
BOARD_ROI_PAD_FRACTION = 0.30


def _board_roi_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Union bounding box of `mask`'s largest closed connected components
    (see the constants above), padded by `BOARD_ROI_PAD_FRACTION` and
    clamped to the image bounds -- the "where on this frame is the board,
    roughly" estimate `_mask_for_detection()`'s second, ROI-restricted
    adaptive-colour pass needs. Returns None if `mask` has no qualifying
    component at all (caller degrades to the fixed-HSV fallback, same as
    any other pass-1 refusal)."""
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_PX, CLOSE_KERNEL_PX))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1][:BOARD_ROI_MAX_COMPONENTS]
    boxes = []
    for idx in order:
        area = int(areas[idx])
        if area < BOARD_ROI_MIN_COMPONENT_AREA_PX:
            continue
        label = 1 + int(idx)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        boxes.append((x, y, x + w, y + h))
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pad_w = int(round((x1 - x0) * BOARD_ROI_PAD_FRACTION))
    pad_h = int(round((y1 - y0) * BOARD_ROI_PAD_FRACTION))
    img_h, img_w = mask.shape[:2]
    x0 = max(0, x0 - pad_w)
    y0 = max(0, y0 - pad_h)
    x1 = min(img_w, x1 + pad_w)
    y1 = min(img_h, y1 + pad_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _mask_for_detection(image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """LIVE-DERIVED RING-COLOR SEGMENTATION, wired 2026-08-21, ROI-
    restricted two-pass fix added 2026-09-03 -- this is the FIRST step
    of the entire calibration pipeline, before any pose exists, per this
    module's own docstring, so every change here is treated with real
    caution.

    Tries `opendarts.calibration.adaptive_ring_color.adaptive_ring_color_mask()`
    (per-image adaptive hue clustering -- see that
    module's own docstring for the full derivation and its real
    114/117-image, zero-false-accept validation) -- UNCHANGED, called
    exactly as before, TWICE now instead of once.

    ROI-RESTRICTED SECOND PASS, 2026-09-03 -- real root cause, confirmed
    on real failing frames (see `MERGE_20260903.md`'s own follow-up
    investigation): pass 1's Otsu
    saturation threshold is computed over the WHOLE FRAME, which is
    dominated by the dark surround outside the board -- this drags the
    threshold up (measured 125-136 on real failing cam0 frames) and
    starves the ring's own dim, oblique-angle paint just enough to break
    8-connectivity at one flank, which is WHY `_outer_ring_component_mask()`
    then selects a fragment (a bottom arc, not the whole ring) rather
    than a genuine double+treble MERGE (that earlier hypothesis was
    investigated and REFUTED -- treble is a separate connected component
    in 600/600 real frames tested; there is no merge to correct for).
    Restricting the SAME clustering to a rough board ROI (a first-pass
    estimate of "roughly where the board's colour blobs are," via
    `_board_roi_from_mask()`) removes the dark surround from the
    threshold computation entirely -- measured on the same real failing
    frames, the ROI-restricted threshold drops to 75-84, the right-flank
    connected-pixel band goes from 342-814px (broken) to 2243-2286px
    (stays connected, whole ring selected), and this measurably fixes
    end-to-end seed->bull->reseat detection (`tests/test_landmark_
    detection_board_roi_20260903.py`'s own real-corpus regression tests
    carry the full before/after numbers, not repeated here). Real,
    broader finding, not chased further here: this whole-frame-vs-ROI gap
    affects EVERY camera to some degree (ROI thresholds ~75-96 sit well
    below even "clean" full-frame values ~121-142 on this rig) -- the
    previously-known-failing packages just happened to cross the real
    starvation edge; this fix's real benefit is broader than the 2 known
    incidents that motivated it.

    `adaptive_ring_color_mask()` ITSELF is entirely untouched by this fix
    -- called twice, the second time over a cropped sub-image, nothing
    about its own clustering/gating logic changes.

    Falls back to the fixed fallback thresholds (`_color_mask()` below,
    hardcoded HSV cutoffs) at any refusal
    point -- pass 1 refuses, pass 1 produces no usable ROI, or pass 2
    refuses -- exactly the same "can only ever be as bad as today's
    prior behavior, never worse" safety posture this module's own
    history already established, now with one more fallback trigger, not
    a changed contract.

    Returns (mask, source) where source is "adaptive_roi" (both passes
    succeeded), "fixed_hsv_fallback" (any refusal), or (unreachable in
    practice today, kept only as a defensive label in case pass 1 ever
    legitimately can't produce a usable ROI while still being confident
    enough itself to be worth reporting separately) "adaptive" -- not
    currently surfaced on `LandmarkDetectionResult` (a real, separate
    schema change on a result dataclass consumed by callers this task
    did not audit for that change) but logged, so a live operator/log-
    reader can see which path a given calibration actually used."""
    import logging

    log = logging.getLogger(__name__)
    try:
        from opendarts.calibration.adaptive_ring_color import adaptive_ring_color_mask

        pass1 = adaptive_ring_color_mask(image_bgr)
        if not (pass1.ok and pass1.mask is not None):
            log.info(
                "adaptive ring-color segmentation (pass 1, whole-frame) "
                "refused (%s) -- falling back to fixed HSV thresholds",
                pass1.quality.reason if pass1.quality is not None else "no reason given",
            )
            return _color_mask(image_bgr), "fixed_hsv_fallback"

        roi = _board_roi_from_mask(pass1.mask)
        if roi is None:
            log.info(
                "adaptive ring-color segmentation (pass 1) produced no "
                "usable board ROI -- falling back to fixed HSV thresholds"
            )
            return _color_mask(image_bgr), "fixed_hsv_fallback"

        x0, y0, x1, y1 = roi
        pass2 = adaptive_ring_color_mask(image_bgr[y0:y1, x0:x1])
        if not (pass2.ok and pass2.mask is not None):
            log.info(
                "adaptive ring-color segmentation (pass 2, ROI-restricted) "
                "refused (%s) -- falling back to fixed HSV thresholds",
                pass2.quality.reason if pass2.quality is not None else "no reason given",
            )
            return _color_mask(image_bgr), "fixed_hsv_fallback"

        full_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = pass2.mask
        return full_mask, "adaptive_roi"
    except Exception:  # noqa: BLE001 -- never break calibration's first step over this
        log.exception(
            "adaptive ring-color segmentation raised -- falling back to "
            "fixed HSV thresholds"
        )
    return _color_mask(image_bgr), "fixed_hsv_fallback"


def detect_double_ring_quad(image_bgr: np.ndarray) -> LandmarkDetectionResult:
    """Detect the double ring's outer-boundary ellipse in a real board
    image and return a 4-point representative quad sampled off it.

    `image_bgr`: a real captured frame (e.g. loaded via cv2.imread from a
    bg_cam*.png, or a live snapshot), BGR channel order, uint8.
    """
    mask, _mask_source = _mask_for_detection(image_bgr)
    component_mask = _outer_ring_component_mask(mask)
    if component_mask is None:
        return LandmarkDetectionResult(
            ok=False, ellipse=None, quad_points_px=None,
            n_raw_boundary_points=0, n_inlier_boundary_points=0,
            reason="no connected color component found (empty/near-empty color mask)",
        )

    boundary_pts = _trace_outer_boundary(mask, component_mask)
    if boundary_pts is None:
        return LandmarkDetectionResult(
            ok=False, ellipse=None, quad_points_px=None,
            n_raw_boundary_points=0, n_inlier_boundary_points=0,
            reason="too few outer-ring boundary points traced",
        )

    ellipse_tuple, kept_pts = _robust_fit_ellipse(boundary_pts)
    if ellipse_tuple is None:
        return LandmarkDetectionResult(
            ok=False, ellipse=None, quad_points_px=None,
            n_raw_boundary_points=len(boundary_pts), n_inlier_boundary_points=0,
            reason="ellipse fit failed (fewer than 5 boundary points)",
        )

    ellipse = Ellipse.from_cv2(ellipse_tuple)
    quad = ellipse.sample_quad()
    return LandmarkDetectionResult(
        ok=True,
        ellipse=ellipse,
        quad_points_px=quad,
        n_raw_boundary_points=len(boundary_pts),
        n_inlier_boundary_points=len(kept_pts),
        boundary_points_px=kept_pts,
    )


def nearest_point_on_ellipse(ellipse: Ellipse, point_px, n: int = 4000) -> tuple[float, np.ndarray]:
    """(distance_px, closest_point_xy) from `point_px` to the closest
    sampled point on `ellipse`'s boundary. Brute-force over `n` samples
    -- simple and accurate enough for accuracy reporting (sub-0.1px
    sampling error at n=4000 for boards of this pixel size), not meant
    for a hot path.

    This is the accuracy metric used to compare detected ellipses
    against a known real reference point (e.g. calibration.json's `src`
    quad for the same image)."""
    samples = ellipse.boundary_samples(n)
    point_px = np.asarray(point_px, dtype=np.float64)
    d = np.hypot(samples[:, 0] - point_px[0], samples[:, 1] - point_px[1])
    i = int(np.argmin(d))
    return float(d[i]), samples[i]
