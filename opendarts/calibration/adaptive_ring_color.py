"""Adaptive (per-image) red/green ring color segmentation -- a from-
scratch alternative to `landmark_detection.py`'s fixed HSV constants
(`GREEN_HSV_LOW/HIGH`, `RED_HSV_LOW_1/HIGH_1`, `RED_HSV_LOW_2/HIGH_2`),
which were "[t]uned by direct inspection of real bg_cam0/1/2 images" --
i.e. hardcoded to this one rig's lighting/camera color response.

**NOT wired into `detect_double_ring_quad()`'s live call path.** This is
purely additive, validation-stage work -- see module docstring of
`landmark_detection.py` for why that function is treated with extra
caution ("literally the FIRST step of the entire calibration pipeline").
See `dev/calibration/validate_adaptive_ring_color.py` for the
full-corpus comparison this module was built to support.

## Why fixed hue cutoffs don't bootstrap safely

Real WDF/PDC dartboards universally use red and green for the double/
treble rings (the red/green PAIR is a universal convention; which exact
segment is red vs green is the only thing that occasionally varies on
rare televised special-event boards -- confirmed via web search). Any
real board image therefore contains two dominant, strongly-saturated,
well-separated-in-hue RING color clusters. Rather than assume *where*
on the hue wheel those two clusters sit (what a fixed threshold does),
this module discovers them directly from the image: filter down to
saturated/bright ("colorful") pixels, cluster their hue values, then
use color-NAME domain knowledge (what counts as "a red" / "a green" on
the real color wheel -- universal, not rig-specific) ONLY to decide
which clusters are the red and green ring paint -- never to constrain
where the clustering itself puts the centers.

## The frame is not the board: escalating k

The BOARD has exactly two ring paint colors, but the FRAME may contain
any number of other saturated surfaces (measured real case on this
corpus: one camera's field of view includes a large wall/backdrop
region above the board that renders as saturated cyan-blue under that
camera's white balance -- 4-6x MORE pixels than the real green ring,
every session). A fixed k=2 clustering of the whole frame therefore
FAILS structurally on such a camera: the biggest colorful population
takes over one cluster and the real ring color is absorbed/dragged.
Fixed by escalating k: try k=2, then 3, then 4; at each k, clusters are
LABELED by color band (red / green / neither="junk"), same-band
clusters are merged, junk clusters are discarded, and the merged
red+green pair must pass the separation and compactness gates below.
The first k that yields a passing red/green pair wins. On the full
corpus this accepts 71 images at k=2 (nothing else saturated in frame),
and correctly isolates wall/backdrop populations into a discarded junk
cluster at k=3/4 on the cameras that see them.

## Handling hue's circular nature

OpenCV hue is 0-179, representing the real 0-360-degree color wheel at
half scale (so 1 unit of OpenCV hue = 2 real degrees) and wrapping
there -- the existing fixed thresholds already have to OR together two
ranges (`RED_HSV_LOW_1/HIGH_1` and `RED_HSV_LOW_2/HIGH_2`) because red
straddles that 0/179 seam. Naive 1-D k-means on raw hue values fails
near that seam (it can't tell 178 and 2 are close). Fixed here by
embedding each hue value on the unit circle *at its real angle*:

    theta = hue * 2 * (pi / 180)      # OpenCV hue -> real radians
    x, y  = cos(theta), sin(theta)

and running ordinary Euclidean k-means (`cv2.kmeans`) on the (x, y)
points. Euclidean (chord) distance on a unit circle is a strictly
increasing function of true angular distance for any two angles in
[0, 2*pi), so k-means in this embedding is exactly k-means on circular
distance -- no seam, no special-casing.

## The output mask is built FROM the clustering

A returned pixel is foreground only if it (a) was assigned to a red or
green ring cluster (junk-cluster members are excluded outright) AND
(b) sits within `MAX_PIXEL_DIST_TO_CENTER_DEG` real degrees of its
merged cluster's center. (An earlier version of this module returned
every "colorful" pixel regardless of cluster membership -- the
clustering had zero effect on the mask, so a localized patch of
unrelated saturated color could silently remain baked into an ok=True
mask. A verifier pass caught that; this per-pixel membership+distance
gate is the fix, with the gate value MEASURED, not guessed -- see
`MAX_PIXEL_DIST_TO_CENTER_DEG`'s own docstring.)

## Confidence signal

Never silently trust a degenerate result (same discipline as every
other derived-calibration piece built for this project). See
`ClusterQuality` -- callers should check `.ok` before trusting the
returned mask/cluster centers, and can inspect `.reason` when it's
False.

Measured full-corpus results, 2026-08-20 (13 sessions x 3 cams x
first/middle/last throw = 117 images, see
`dev/calibration/validate_adaptive_ring_color.py`): accepted 114/117
(97.4%) -- per camera 39/39, 37/39, 38/39, where the original
k=2-only design accepted 71/117 and refused 37/39 on the camera whose
wall/backdrop region dominates its colorful pixels (cam1 -- verified
at pixel level: the wall population is 4-6x LARGER than the real green
ring's, so k=2 was structurally unable to isolate it). Downstream
ellipse fit (same boundary-trace/fit code as the fixed path) agrees
with the fixed-HSV method to median 3.3px / p90 18.4px center delta on
the 114 comparable pairs; the max delta (141px, one cam1 image) was
overlaid and inspected -- the FIXED method's fit is the wrong one there
(minor axis 1126px on a 720px-tall image, sweeping through the wall),
the adaptive fit traces the real ring. The one systematic per-camera
offset (cam2, ~16px median center delta) was also overlaid and
inspected: the fixed method's cam2 ellipse consistently bulges past
the ring's true outer edge through the board's red branding
text/stickers ("BLADE6"/"GEN 6"), while the adaptive fit hugs the
edge -- the delta is dominated by the fixed method's own bias, not an
adaptive error. The 3 refusals (2 cam1, 1 cam2, each the session's
FIRST throw in 2 cases) are genuinely marginal frames -- warm
orange-toned stragglers inside the red label band keep merged-red
compactness at 14.5-16.7 deg across every k -- and refusing them is
the intended honest outcome, not a gap to tune away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# --- Color-band LABELING constants ----------------------------------
#
# Used ONLY to label which already-found clusters are "red" / "green"
# ring paint (and which are neither) -- NEVER to constrain where
# clustering puts the centers. These encode universal color-NAME
# knowledge (where on the real color wheel a human calls a hue "red"
# or "green"), not anything measured from this rig:
#
#   red   = within 40 real degrees of 0 (the red point of the wheel)
#   green = real degrees 80..180 (yellow-green through green to the
#           green/cyan divide at 180)
#
# The gap regions (40-80: orange/yellow; 180-320: cyan/blue/magenta)
# are "junk" -- no real board paints its rings those colors, and the
# real non-ring populations measured on this corpus land squarely in
# them (wall/backdrop cyan-blue at ~196-245 real degrees, surround
# wood-tone orange at ~41-53 real degrees).
RED_LABEL_MAX_REAL_DEG = 40.0
GREEN_LABEL_LOW_REAL_DEG = 80.0
GREEN_LABEL_HIGH_REAL_DEG = 180.0

# Minimum number of "colorful" (post S/V filter) pixels required to
# attempt clustering at all -- below this, k-means on a handful of
# points is noise, not signal.
MIN_COLORFUL_PIXELS = 200

# A degenerate/low-confidence result: the two (merged) ring cluster
# centers should be far apart on the real color wheel (true red and
# true green are roughly opposite-ish, well over 90 degrees apart on
# every real board). If they collapse to two nearby centers, something
# is wrong (e.g. only one ring color is actually visible in this
# image/crop).
MIN_CLUSTER_SEPARATION_DEG = 60.0

# A degenerate/low-confidence result, take 2: either merged ring
# cluster's OWN members should sit tightly around its center (true ring
# paint is a narrow real hue band). A cluster that has ABSORBED a
# strongly-saturated non-ring object elsewhere in frame shows up as
# elevated MEAN angular distance of members to their own center, even
# though the two centers themselves stayed far apart. Measured on the
# full `data/archive/clean/` corpus (see git history of this file for
# the original k=2 measurement): every genuinely good fit had
# max(red, green) compactness <= 13.6 deg, every wall-absorption case
# had >= 15.4 -- a clean, real gap, not a guessed round number. 14.0
# sits in that gap. Under escalating k this gate is what drives the
# escalation: a k whose merged red/green pair fails it is rejected and
# the next k is tried.
MAX_CLUSTER_COMPACTNESS_DEG = 14.0

# Per-PIXEL foreground gate (real degrees): a ring-cluster member pixel
# only enters the output mask if its hue sits within this angular
# distance of its merged cluster's center. MEASURED on the full corpus
# (13 sessions x 3 cams x 3 throws = 117 images, escalating-k
# clustering, tmp measurement script 2026-08-20, reproduced in
# dev/calibration/validate_adaptive_ring_color.py's sweep): pixels
# belonging to discarded junk clusters were NEVER within 20 real
# degrees of a ring center (closest single junk pixel across every
# image: 20.5 deg; per-image median of the closest: 24.9), while the
# genuine ring-paint core sits well inside (per-cluster mean member
# distance 5-10 deg on clean images). The 8.4%-of-members tail beyond
# 20 deg is absorbed non-ring pixels on k=2-accepted images -- exactly
# what this gate exists to exclude. 20.0 is the measured edge of the
# junk gap; the downstream-ellipse sweep (reproducible:
# validate_adaptive_ring_color.py --sweep-pixel-gate; measured
# 2026-08-20: center-delta median 3.25/3.32/3.35/3.35 px at gate
# 15/20/25/30, acceptance 114/117 at every value) confirmed the fit is
# insensitive across 15-30 (the trace takes the max-radius pixel per
# angular bin from the largest component, so a few hue-marginal
# ring-edge pixels in or out don't move the fitted boundary), so the
# gate sits at the measured junk edge, not tuned to the fit.
MAX_PIXEL_DIST_TO_CENTER_DEG = 20.0

# k values tried, in order, by the escalating-k design (see module
# docstring). 4 is enough for every real failure mode measured on this
# corpus (2 ring colors + up to 2 distinct non-ring saturated
# populations, e.g. cyan-blue wall + orange wood tones); a frame with
# MORE distinct saturated non-ring populations than that fails the
# compactness gate and is refused rather than silently mis-clustered.
KMEANS_K_VALUES = (2, 3, 4)

KMEANS_ATTEMPTS = 5
KMEANS_MAX_ITER = 50
KMEANS_EPS = 1e-4


@dataclass
class ClusterQuality:
    """Confidence/quality signal for one image's adaptive color
    clustering -- see module docstring. Always returned, even on
    failure (`ok=False`), so a caller can report *why* rather than just
    silently getting nothing."""

    ok: bool
    reason: str

    n_total_px: int = 0
    n_colorful_px: int = 0
    colorful_fraction: float = 0.0

    sat_threshold: int = 0
    val_threshold: int = 0

    # Which k the escalating-k search settled on (0 = none accepted).
    k_used: int = 0

    # Merged ring-cluster centers, in OpenCV hue units (0-179), after
    # band labeling. NaN when no accepted clustering exists.
    red_hue_cv2: float = float("nan")
    green_hue_cv2: float = float("nan")

    # Pixels in the RETURNED MASK per ring color (i.e. cluster members
    # that also passed the per-pixel distance gate). 0 on failure.
    n_red_px: int = 0
    n_green_px: int = 0

    # Colorful pixels assigned to discarded junk clusters (0 at k=2),
    # plus ring-cluster members excluded by the per-pixel distance
    # gate -- the two populations the clustering keeps OUT of the mask.
    n_junk_px: int = 0
    n_gated_out_px: int = 0

    # Centers of any discarded junk clusters (OpenCV hue units) -- for
    # honest reporting of what was in frame, not used for anything.
    junk_center_hues_cv2: list[float] = field(default_factory=list)

    # Real-world degrees between the two merged ring-cluster centers
    # (0-180, the shorter arc) -- the main separation/confidence signal.
    cluster_separation_deg: float = 0.0

    # Mean real-world angular distance (degrees) of each merged ring
    # cluster's own member pixels to the merged center -- lower is a
    # tighter, more confident cluster. NaN if never computed.
    red_compactness_deg: float = float("nan")
    green_compactness_deg: float = float("nan")


@dataclass
class AdaptiveColorResult:
    ok: bool
    mask: np.ndarray | None  # uint8 0/255, same shape as input, or None
    quality: ClusterQuality


def _hue_to_unit_circle(hue_cv2: np.ndarray) -> np.ndarray:
    """OpenCV hue (0-179) -> (N, 2) array of (cos, sin) unit-circle
    points at the hue's REAL angle (2x OpenCV hue, since OpenCV hue is
    half the real 0-360 wheel)."""
    theta = hue_cv2.astype(np.float64) * 2.0 * (np.pi / 180.0)
    return np.stack([np.cos(theta), np.sin(theta)], axis=1)


def _unit_circle_to_hue(xy: np.ndarray) -> float:
    """Inverse of _hue_to_unit_circle for a single (x, y) point ->
    OpenCV hue units (0-179, wrapped)."""
    theta = np.arctan2(xy[1], xy[0])  # real radians, (-pi, pi]
    real_deg = np.degrees(theta) % 360.0
    return (real_deg / 2.0) % 180.0


def _circular_deg_distance(hue_a_cv2: float, hue_b_cv2: float) -> float:
    """Shortest-arc real-world degrees between two OpenCV-hue values."""
    real_a = (hue_a_cv2 * 2.0) % 360.0
    real_b = (hue_b_cv2 * 2.0) % 360.0
    d = abs(real_a - real_b) % 360.0
    return min(d, 360.0 - d)


def _circular_deg_distance_arr(hues_cv2: np.ndarray, center_hue_cv2: float) -> np.ndarray:
    """Vectorized `_circular_deg_distance`: shortest-arc real-world
    degrees between each OpenCV-hue value in `hues_cv2` and a single
    center hue."""
    d = np.abs(hues_cv2 * 2.0 - center_hue_cv2 * 2.0) % 360.0
    return np.minimum(d, 360.0 - d)


def _circular_mean_hue(hues_cv2: np.ndarray) -> float | None:
    """Circular mean of OpenCV-hue values (computed at their REAL
    angles, so the 0/179 seam is handled), returned in OpenCV hue
    units. None if the input is empty or the mean resultant vector is
    degenerate (members spread uniformly -- no meaningful center).

    The explicit empty-input check matters: `np.cos([]).mean()` is NaN,
    and NaN comparisons are all False, so without it an empty cluster
    would fall through BOTH the `< 1e-9` guard here and the `is None`
    checks in `_try_k`, then silently pass the separation and
    compactness gates too (`nan < thresh` and `nan > thresh` are both
    False) -- an ok=True result with a NaN center. Not observed with
    this OpenCV build (its kmeans++ reassigns empty clusters
    internally), but a real defense-in-depth gap a verifier pass
    caught."""
    if hues_cv2.size == 0:
        return None
    theta = hues_cv2.astype(np.float64) * 2.0 * (np.pi / 180.0)
    x, y = np.cos(theta).mean(), np.sin(theta).mean()
    if math.hypot(x, y) < 1e-9:
        return None
    real_deg = math.degrees(math.atan2(y, x)) % 360.0
    return (real_deg / 2.0) % 180.0


def _label_band(center_hue_cv2: float) -> str:
    """'red' / 'green' / 'junk' color-NAME label for a cluster center --
    see the RED/GREEN label constants' comment block. The red and green
    bands are disjoint (red ends 40 real degrees from 0, green starts
    at 80), so a center has exactly one label."""
    real = (center_hue_cv2 * 2.0) % 360.0
    if min(real, 360.0 - real) <= RED_LABEL_MAX_REAL_DEG:
        return "red"
    if GREEN_LABEL_LOW_REAL_DEG <= real <= GREEN_LABEL_HIGH_REAL_DEG:
        return "green"
    return "junk"


@dataclass
class _ClusterAttempt:
    """One k's clustering outcome -- internal to the escalating-k loop."""

    k: int
    ok: bool
    fail_reason: str = ""
    red_center_hue: float = float("nan")
    green_center_hue: float = float("nan")
    separation_deg: float = 0.0
    red_compactness: float = float("nan")
    green_compactness: float = float("nan")
    # boolean masks over the colorful-pixel array (not the image)
    red_members: np.ndarray | None = None
    green_members: np.ndarray | None = None
    n_junk_px: int = 0
    junk_center_hues: list[float] = field(default_factory=list)


def _try_k(points_f32: np.ndarray, hue_vals: np.ndarray, k: int) -> _ClusterAttempt:
    """Cluster the unit-circle embedding into k groups, band-label and
    merge, and evaluate the separation/compactness gates. Returns the
    attempt either way; `.ok` says whether this k is acceptable."""
    import cv2

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, KMEANS_MAX_ITER, KMEANS_EPS)
    _compactness, labels, centers = cv2.kmeans(
        points_f32, k, None, criteria, KMEANS_ATTEMPTS, cv2.KMEANS_PP_CENTERS
    )
    labels = labels.flatten()

    center_hues = [_unit_circle_to_hue(centers[i]) for i in range(k)]
    bands = [_label_band(h) for h in center_hues]
    red_ks = [i for i in range(k) if bands[i] == "red"]
    green_ks = [i for i in range(k) if bands[i] == "green"]
    junk_ks = [i for i in range(k) if bands[i] == "junk"]

    att = _ClusterAttempt(k=k, ok=False)
    att.junk_center_hues = [float(center_hues[i]) for i in junk_ks]
    att.n_junk_px = int(np.isin(labels, junk_ks).sum()) if junk_ks else 0

    if not red_ks or not green_ks:
        missing = []
        if not red_ks:
            missing.append("red")
        if not green_ks:
            missing.append("green")
        att.fail_reason = (
            f"k={k}: no cluster center in the {'/'.join(missing)} label band "
            f"(centers at cv2-hue {', '.join(f'{h:.1f}' for h in center_hues)})"
        )
        return att

    red_members = np.isin(labels, red_ks)
    green_members = np.isin(labels, green_ks)

    # Merged center = circular mean of the merged members' own hues (not
    # a mean of the k-means centers -- member-derived so a merge of a
    # big and a small fragment weights correctly).
    red_center = _circular_mean_hue(hue_vals[red_members])
    green_center = _circular_mean_hue(hue_vals[green_members])
    if (
        red_center is None
        or green_center is None
        or math.isnan(red_center)
        or math.isnan(green_center)
    ):
        # The isnan checks are belt-and-suspenders on top of
        # _circular_mean_hue's own empty/degenerate -> None contract:
        # a NaN center would otherwise silently PASS both gates below
        # (NaN comparisons are all False) -- see that function's
        # docstring.
        att.fail_reason = f"k={k}: degenerate circular mean for a merged ring cluster"
        return att

    att.red_center_hue = red_center
    att.green_center_hue = green_center
    att.red_members = red_members
    att.green_members = green_members
    att.separation_deg = _circular_deg_distance(red_center, green_center)
    att.red_compactness = float(_circular_deg_distance_arr(hue_vals[red_members], red_center).mean())
    att.green_compactness = float(_circular_deg_distance_arr(hue_vals[green_members], green_center).mean())

    if att.separation_deg < MIN_CLUSTER_SEPARATION_DEG:
        att.fail_reason = (
            f"k={k}: merged red/green separation {att.separation_deg:.1f} deg < "
            f"{MIN_CLUSTER_SEPARATION_DEG:.0f} deg"
        )
        return att
    worst = max(att.red_compactness, att.green_compactness)
    if worst > MAX_CLUSTER_COMPACTNESS_DEG:
        att.fail_reason = (
            f"k={k}: merged cluster compactness {worst:.1f} deg > "
            f"{MAX_CLUSTER_COMPACTNESS_DEG:.0f} deg (a ring cluster likely absorbed a "
            f"non-ring saturated population this k could not isolate)"
        )
        return att

    att.ok = True
    return att


def _otsu_threshold(channel_u8: np.ndarray) -> int:
    import cv2

    t, _ = cv2.threshold(channel_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def adaptive_ring_color_mask(image_bgr: np.ndarray) -> AdaptiveColorResult:
    """Adaptively segment `image_bgr`'s red+green ring pixels with no
    fixed hue thresholds, per module docstring. Returns a uint8 0/255
    mask shaped like the fixed-threshold `_color_mask()` in
    `landmark_detection.py` (drop-in comparable, NOT wired to replace
    it), plus a `ClusterQuality` confidence signal.

    `image_bgr`: real captured frame, BGR channel order, uint8.
    """
    import cv2

    h, w = image_bgr.shape[:2]
    n_total = h * w
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Step 1: adaptively (Otsu, not a fixed constant) split "colorful"
    # (ring paint) pixels from the board's low-saturation surfaces
    # (white/black/cream segments, printed numbers, wire, background).
    # Otsu on the whole-image saturation histogram finds the real
    # bimodal split for THIS image/lighting rather than assuming a fixed
    # cutoff generalizes across rigs.
    sat_thresh = _otsu_threshold(sat)
    colorful = sat > sat_thresh

    # Step 2: within the already-colorful pixels, drop any that are
    # still too dark (shadowed edges, motion blur on the ring boundary)
    # to trust their hue -- again Otsu, computed only over the
    # colorful subset so a generally-dark image doesn't bias the cutoff.
    if colorful.sum() >= MIN_COLORFUL_PIXELS:
        val_thresh = _otsu_threshold(val[colorful])
    else:
        val_thresh = 0
    colorful &= val > val_thresh

    n_colorful = int(colorful.sum())
    colorful_fraction = n_colorful / n_total if n_total else 0.0

    quality = ClusterQuality(
        ok=False,
        reason="",
        n_total_px=n_total,
        n_colorful_px=n_colorful,
        colorful_fraction=colorful_fraction,
        sat_threshold=sat_thresh,
        val_threshold=val_thresh,
    )

    if n_colorful < MIN_COLORFUL_PIXELS:
        quality.reason = (
            f"too few colorful pixels after adaptive S/V filter "
            f"({n_colorful} < {MIN_COLORFUL_PIXELS}; sat_thresh={sat_thresh}, val_thresh={val_thresh})"
        )
        return AdaptiveColorResult(ok=False, mask=None, quality=quality)

    hue_vals = hue[colorful].astype(np.float64)
    points = _hue_to_unit_circle(hue_vals).astype(np.float32)

    # Step 3: escalating-k circular clustering (see module docstring
    # "The frame is not the board"). First k whose band-labeled, merged
    # red/green pair passes both gates wins.
    attempts: list[_ClusterAttempt] = []
    chosen: _ClusterAttempt | None = None
    for k in KMEANS_K_VALUES:
        att = _try_k(points, hue_vals, k)
        attempts.append(att)
        if att.ok:
            chosen = att
            break

    if chosen is None:
        # Report the most informative failed attempt: prefer one that at
        # least found both ring bands (its gate numbers mean something),
        # picking the lowest worst-compactness among those; fall back to
        # the last attempt otherwise.
        with_bands = [a for a in attempts if a.red_members is not None]
        best = (
            min(with_bands, key=lambda a: max(a.red_compactness, a.green_compactness))
            if with_bands
            else attempts[-1]
        )
        quality.k_used = 0
        quality.red_hue_cv2 = best.red_center_hue
        quality.green_hue_cv2 = best.green_center_hue
        quality.cluster_separation_deg = float(best.separation_deg)
        quality.red_compactness_deg = best.red_compactness
        quality.green_compactness_deg = best.green_compactness
        quality.n_junk_px = best.n_junk_px
        quality.junk_center_hues_cv2 = best.junk_center_hues
        quality.reason = "no k accepted: " + "; ".join(a.fail_reason for a in attempts)
        return AdaptiveColorResult(ok=False, mask=None, quality=quality)

    quality.k_used = chosen.k
    quality.red_hue_cv2 = float(chosen.red_center_hue)
    quality.green_hue_cv2 = float(chosen.green_center_hue)
    quality.cluster_separation_deg = float(chosen.separation_deg)
    quality.red_compactness_deg = chosen.red_compactness
    quality.green_compactness_deg = chosen.green_compactness
    quality.n_junk_px = chosen.n_junk_px
    quality.junk_center_hues_cv2 = chosen.junk_center_hues

    # Step 4: build the output mask FROM the clustering -- a pixel is
    # foreground only if it belongs to a merged ring cluster AND sits
    # within the measured per-pixel distance gate of that cluster's
    # center (see MAX_PIXEL_DIST_TO_CENTER_DEG). Junk-cluster members
    # never enter the mask.
    red_close = (
        _circular_deg_distance_arr(hue_vals, chosen.red_center_hue)
        <= MAX_PIXEL_DIST_TO_CENTER_DEG
    )
    green_close = (
        _circular_deg_distance_arr(hue_vals, chosen.green_center_hue)
        <= MAX_PIXEL_DIST_TO_CENTER_DEG
    )
    red_keep = chosen.red_members & red_close
    green_keep = chosen.green_members & green_close
    keep = red_keep | green_keep

    quality.n_red_px = int(red_keep.sum())
    quality.n_green_px = int(green_keep.sum())
    quality.n_gated_out_px = int(
        (chosen.red_members & ~red_close).sum() + (chosen.green_members & ~green_close).sum()
    )

    quality.ok = True
    quality.reason = "ok" if chosen.k == KMEANS_K_VALUES[0] else (
        f"ok at k={chosen.k} (junk clusters isolated: "
        f"{', '.join(f'{h:.1f}' for h in chosen.junk_center_hues) or 'none'} cv2-hue; "
        f"earlier attempts: {'; '.join(a.fail_reason for a in attempts[:-1])})"
    )

    mask = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.nonzero(colorful)
    mask[ys[keep], xs[keep]] = 255

    return AdaptiveColorResult(ok=True, mask=mask, quality=quality)
