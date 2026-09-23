"""Talos's own 2D observation: a shaft line through the detected tip.

Blob *selection* reuses Apollo's shared tip-detection machinery
verbatim (same diff mask, same opened/dilated connected
components, same elongation ranking, same companion-blob tip-end
localization) -- Talos does not have its own blob detector, and never
duplicates that shared code here, only calls into it. What IS Talos's
own opinion, and therefore lives in this module: once a blob and its tip
pixel are found, Talos forces a 2D line through that tip pixel whose
*direction* comes from re-PCA'ing only the tip-ward
`_SHAFT_TIPWARD_FRACTION` of the blob's pixels (not the whole-blob PCA
axis, which is rotated off the barrel by the flight's wider far end).
That line is `plane_from_image_line()`'s (`opendarts.engines.talos.
plane_geometry`) input for the 3D reconstruction, and its endpoints/
opened pixels also feed `opendarts.engines.talos.consensus`'s outward-edge
pick.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from opendarts.imageops import DiffCrop

# Usable shaft span (px) below which a 2D line does not define a
# direction. Re-measured on data/archive/clean, 507 camera-lines, after
# the 12px gate (/ span histogram):
#   dropped already: 1 at 3px
#   kept below 30px: 15.1, 22.0, 24.0  (three singletons)
#   then a real cluster: 10 lines in [30, 40), p10 of kept = 53
# The 15.1px and 24.0px lines, used as equal votes in a 3-plane SVD,
# produced a reconstructed axis whose tilt was either 81° (physically
# a dart lying in the board) or a 10mm adjacent-sector miss whose
# other-two-camera pair was 13° / 0.8mm. A 15-24px shaft-ward span is
# not an observable direction -- angular uncertainty ~1/span. 25.0 sits
# in the gap before the 30px cluster. Not swept against AD%.
MIN_SHAFT_SPAN_PX = 25.0

# Fraction of the blob's axis-span, starting at the tip, treated as
# shaft pixels. The flight is the wide far end (~30% of dart length);
# whole-blob PCA of that silhouette is rotated off the barrel. 0.60 is
# the geometric cut (tip-ward 60% of length), not a value tuned against oracle labels.
_SHAFT_TIPWARD_FRACTION = 0.60


def fit_shaft_line_px(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    *,
    precomputed: "DiffCrop | None" = None,
) -> tuple[tuple[float, float], tuple[float, float], dict] | None:
    """2D shaft line through the detected tip, from shaft pixels not flights.

    Same blob selection as detect_tip() (diff mask, elongation ranking,
    companion-blob tip-end localization). The line is forced through the
    tip pixel; direction is PCA of the tip-ward _SHAFT_TIPWARD_FRACTION
    of that blob. Returns None if no blob; returns dropped=True in the
    diag (and dummy endpoints) if the usable shaft span is below
    MIN_SHAFT_SPAN_PX so the caller can omit that camera.

    `precomputed`: optional `opendarts.imageops.DiffCrop` (2026-09-06 perf
    pass) -- the gray/|diff|/blur front end already computed by the
    caller (Zeus, once per camera for all its sub-engines) and cropped to
    where the frame changed. Used only if it satisfies
    `blob_detection.PRECOMPUTE_REQUIREMENTS` for these images, otherwise
    ignored; bit-identical either way (the cropped dilated mask is pasted
    into a zero full frame before labeling). Must be None for a frame the
    prior-dart erase modified (the caller handles that).
    """
    import cv2

    # Frozen local copy, not opendarts.engines.apollo.tip_detection --
    # see opendarts/engines/talos/blob_detection.py's own module note for
    # why this is Talos's own private dependency, not a live import.
    from opendarts.engines.talos.blob_detection import (
        DIFF_THRESHOLD,
        DILATE_KERNEL_PX,
        MIN_ELONGATION_RATIO,
        OPEN_KERNEL_PX,
        PRECOMPUTE_REQUIREMENTS,
        TOP_K_AREA_CANDIDATES,
        _component_stats_in_bbox,
        _diff_mask,
        _find_companion_end,
        _locate_tip_in_component,
    )
    from opendarts.imageops import ellipse_kernel, morph_on_bbox, threshold_mask

    if bg_bgr.shape != frame_bgr.shape:
        return None

    pc = precomputed
    if pc is not None and not pc.accepts(PRECOMPUTE_REQUIREMENTS, bg_bgr.shape):
        pc = None
    if pc is None:
        img_h, img_w = bg_bgr.shape[:2]
        bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = _diff_mask(bg_gray, frame_gray)
        origin = (0, 0)
    else:
        img_h, img_w = pc.img_h, pc.img_w
        bg_gray = pc.bg_gray
        mask = threshold_mask(pc.diff_blur, DIFF_THRESHOLD)  # crop-sized
        origin = pc.origin
    # 2026-09-05 perf pass: this function is ~94% of Talos's per-throw
    # time, and the full-frame 31px ellipse dilate below was ~46% of it
    # on its own (OpenCV does not parallelise ellipse kernels). It now
    # runs on the opened mask's padded non-zero bounding box (median ~4%
    # of the frame), bit-identical -- see opendarts.imageops.morph_on_bbox.
    # The 3x3 opening runs on the whole (crop) array: before it, above-
    # threshold sensor noise covers most of the frame, so there is
    # nothing to crop to yet.
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(OPEN_KERNEL_PX))
    dilated = morph_on_bbox(opened, cv2.MORPH_DILATE, ellipse_kernel(DILATE_KERNEL_PX))
    if pc is not None:
        dilated = pc.paste_full(dilated)
    n, comp_labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n <= 1:
        return None

    candidates = []
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        bboxes[i] = (x, y, w, h)
        if w > 0.6 * img_w or h > 0.6 * img_h:
            continue
        candidates.append((area, i))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    top = candidates[:TOP_K_AREA_CANDIDATES]

    # Per-label pixels from each label's own bbox (same pixels, same
    # order) rather than a full-frame ==/&/nonzero per candidate -- the
    # other ~26% of this engine's time, and the whole of its worst-case
    # tail via _find_companion_end()'s loop over every other component.
    chosen = None
    elongation = None
    for area, i in top:
        result = _component_stats_in_bbox(comp_labels, i, bboxes[i], opened, origin)
        if result is None:
            continue
        pts, centered, principal, elong = result
        if elong >= MIN_ELONGATION_RATIO:
            chosen = (pts, centered, principal, area, i)
            elongation = elong
            break
    if chosen is None:
        result = _component_stats_in_bbox(
            comp_labels, top[0][1], bboxes[top[0][1]], opened, origin
        )
        if result is None:
            return None
        pts, centered, principal, elongation = result
        chosen = (pts, centered, principal, top[0][0], top[0][1])

    pts, centered, principal, area, comp_id = chosen
    with np.errstate(all="ignore"):
        chosen_proj = centered @ principal
    chosen_centroid = pts.mean(axis=0)
    other_candidates = [(a, i) for a, i in candidates if i != comp_id]
    companion_end = _find_companion_end(
        comp_labels, opened, other_candidates,
        chosen_centroid, principal,
        float(chosen_proj.min()), float(chosen_proj.max()),
        bboxes, origin,
    )
    tip_px, tip_diag = _locate_tip_in_component(
        pts, centered, principal, forced_non_tip_end=companion_end, bg_gray=bg_gray,
    )
    if not (np.isfinite(tip_px[0]) and np.isfinite(tip_px[1])):
        return None

    line = _shaft_line_through_tip(pts, principal, tip_px)
    if line is None:
        return None
    p1, p2, shaft_span, n_shaft = line

    diag = {
        "p1_px": p1,
        "p2_px": p2,
        "span_px": float(shaft_span),
        "component_area_px": int(area),
        "elongation_ratio": float(elongation),
        "tip_px": (float(tip_px[0]), float(tip_px[1])),
        "n_shaft_pixels": int(n_shaft),
        "shaft_tipward_fraction": _SHAFT_TIPWARD_FRACTION,
        "dropped": False,
        "tip_end_width_px": tip_diag.get("tip_end_width_px"),
        "opened_pts": pts,
    }
    if shaft_span < MIN_SHAFT_SPAN_PX:
        diag["dropped"] = True
        diag["drop_reason"] = (
            f"shaft span {shaft_span:.2f}px < MIN_SHAFT_SPAN_PX="
            f"{MIN_SHAFT_SPAN_PX} (15-24px lines on clean/ are not an "
            f"observable shaft direction; gap before the 30px cluster)"
        )
        return p1, p2, diag
    return p1, p2, diag


def _shaft_line_through_tip(
    pts: np.ndarray,
    principal: np.ndarray,
    tip_px: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], float, int] | None:
    """Force a 2D line through `tip_px`; direction from tip-ward pixels.

    `principal` is the whole-blob PCA axis, used only to *order* pixels
    from tip to flight. Direction is then re-estimated from the tip-ward
    subset (PCA of those pixels, falling back to tip → subset centroid).
    """
    tip = np.asarray(tip_px, dtype=np.float64)
    axis = np.asarray(principal, dtype=np.float64)
    with np.errstate(all="ignore"):
        proj = (pts - tip) @ axis
    if not np.all(np.isfinite(proj)):
        return None
    # Positive = into the blob (away from the tip).
    if float(np.median(proj)) < 0:
        axis = -axis
        proj = -proj
    pmin = float(proj.min())
    pmax = float(proj.max())
    if pmax - pmin < 1e-6:
        return None
    cutoff = pmin + _SHAFT_TIPWARD_FRACTION * (pmax - pmin)
    mask = proj <= cutoff
    n_shaft = int(mask.sum())
    if n_shaft < 8:
        return None
    shaft = pts[mask]
    direction = _direction_from_shaft_pixels(shaft, tip)
    if direction is None:
        return None
    with np.errstate(all="ignore"):
        sproj = (shaft - tip) @ direction
    if not np.all(np.isfinite(sproj)):
        return None
    span = float(sproj.max() - sproj.min())
    extent = float(max(abs(sproj.max()), abs(sproj.min()), 1.0))
    p1 = (float(tip[0]), float(tip[1]))
    p2 = (float(tip[0] + direction[0] * extent), float(tip[1] + direction[1] * extent))
    return p1, p2, span, n_shaft


def _direction_from_shaft_pixels(shaft: np.ndarray, tip: np.ndarray) -> np.ndarray | None:
    """Unit direction from the tip along the shaft.

    Re-PCA the shaft pixels; orient it from the tip into the subset.
    If PCA is degenerate, fall back to tip → centroid.
    """
    centroid = shaft.mean(axis=0)
    to_cen = centroid - tip
    fallback_norm = float(np.linalg.norm(to_cen))
    centered = shaft - centroid
    if len(shaft) >= 8:
        cov = np.cov(centered.T)
        if np.all(np.isfinite(cov)):
            evals, evecs = np.linalg.eigh(cov)
            direction = evecs[:, int(np.argmax(evals))]
            if float(np.dot(direction, to_cen)) < 0:
                direction = -direction
            nrm = float(np.linalg.norm(direction))
            if nrm >= 1e-8 and np.all(np.isfinite(direction)):
                return direction / nrm
    if fallback_norm < 1e-6:
        return None
    return to_cen / fallback_norm
