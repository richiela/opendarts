"""Calibration confirmation overlay -- a visual (not just numeric) way
for an operator to SEE whether calibration got the board orientation
right, drawn directly on a camera's live image.

Projects through THIS project's real
`opendarts.pipeline.CameraCalibration` (rvec/tvec/camera_matrix/dist_coeffs)
via `cv2.projectPoints`, reusing the exact projection primitive already
established by `opendarts.geometry.board_color.project_board_point_px()` --
not a second, parallel projection helper.

Three layers, drawn in this order:
1. A semi-transparent blue "wash" over the whole double-ring disk.
2. A magenta/red filled highlight on the wedge the calibration's own
   sector geometry identifies as sector 20 (or whichever
   `highlight_number` is asked for) -- projected from the SAME board-mm
   geometry `opendarts.geometry.board` already uses for scoring
   (`sector_center_angle_deg`), so if calibration/orientation is wrong,
   the highlighted wedge visibly lands on the wrong physical sector.
3. A thin cyan "spider": ring outlines (double/treble both radii +
   outer/inner bull) plus 20 radial wire lines from the outer-bull
   boundary out to the double-outer boundary, at each of the 20 real
   wire angles (`opendarts.geometry.board.wire_boundary_angle_deg`).

Ring radii used here are `opendarts.geometry.board`'s own REGULATION
radii (the same physical model `wire_intersection_landmarks()` places
3D calibration landmarks against) -- not the measured
`*_SCORING_RADIUS_MM` values (`INNER_RING_SCORING_OFFSET_MM`). This
overlay's whole point is showing where the calibration's OWN board-plane
geometry model lands in pixel space, not the fitted scoring-boundary
correction layered on top of it for the separate purpose of matching a
different rig's board.

No caching, deliberately -- see `opendarts.live.server`'s
`GET /api/cameras/{cam_id}/overlay.png` route, which renders this fresh
against whatever calibration is CURRENTLY live on every single request,
the same way `snapshot.png` already serves a fresh frame per request.
That is what actually satisfies "clear it when re-calibrating": there is
no stale image to clear because nothing is ever kept around past one
request.
"""
from __future__ import annotations

import cv2
import numpy as np

from opendarts.geometry.board import (
    BULL_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    SECTOR_ANGLE_DEG,
    SECTOR_NUMBERS_CLOCKWISE,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    polar_to_xy_mm,
    sector_center_angle_deg,
    wire_boundary_angle_deg,
)
from opendarts.geometry.board_color import project_board_point_px

# Regulation ring radii (mm) this overlay draws -- see module docstring
# for why these (not the measured scoring-offset radii) are the right
# choice for a calibration-CONFIRMATION overlay.
RING_RADII_MM: dict[str, float] = {
    "double_outer": DOUBLE_OUTER_RADIUS_MM,
    "double_inner": DOUBLE_INNER_RADIUS_MM,
    "treble_outer": TREBLE_OUTER_RADIUS_MM,
    "treble_inner": TREBLE_INNER_RADIUS_MM,
    "outer_bull": OUTER_BULL_RADIUS_MM,
    "bull": BULL_RADIUS_MM,
}
# Rings drawn with a heavier line (double_outer/treble_outer/outer_bull
# thicker, everything else thin).
_THICK_RINGS = {"double_outer", "treble_outer", "outer_bull"}

# BGR colors -- a cool blue wash over the double-ring disk, a
# magenta-red highlight on the sector-20 wedge, and cyan wires: high
# contrast against a real board's red/green/black/cream paint. A color
# choice, not board geometry.
WIRE_COLOR_BGR = (240, 210, 110)
WASH_COLOR_BGR = (160, 70, 10)
HIGHLIGHT_COLOR_BGR = (45, 35, 190)
WASH_ALPHA = 0.55
HIGHLIGHT_ALPHA = 0.55

DEFAULT_HIGHLIGHT_NUMBER = 20
_RING_POLY_STEPS = 96
_WEDGE_ARC_STEPS = 8


def _ring_polygon_px(calib, radius_mm: float, n: int = _RING_POLY_STEPS) -> np.ndarray:
    """Cam-space polygon approximating the circle of radius `radius_mm`
    around the bull, via `n` points forward-projected through `calib`."""
    pts = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        angle_deg = 360.0 * i / n
        xy_mm = polar_to_xy_mm(radius_mm, angle_deg)
        pts[i] = project_board_point_px(xy_mm, calib)
    return pts


def _wedge_polygon_px(
    calib,
    number: int,
    *,
    r_outer_mm: float = DOUBLE_OUTER_RADIUS_MM,
    steps: int = _WEDGE_ARC_STEPS,
) -> np.ndarray:
    """Cam-space fan polygon for sector `number`'s wedge: bull center out
    to an arc at `r_outer_mm` spanning the wedge's own angular width
    (`sector_center_angle_deg(number)` +/- half a sector) -- the real
    board-plane wedge boundary `opendarts.geometry.board` defines for
    scoring, not a separate angle convention."""
    center = sector_center_angle_deg(number)
    half = SECTOR_ANGLE_DEG / 2.0
    pts = [project_board_point_px((0.0, 0.0), calib)]
    for i in range(steps + 1):
        angle_deg = center - half + (2.0 * half) * i / steps
        xy_mm = polar_to_xy_mm(r_outer_mm, angle_deg)
        pts.append(project_board_point_px(xy_mm, calib))
    return np.asarray(pts, dtype=np.float32)


def draw_calibration_overlay(
    frame: np.ndarray,
    calib,
    *,
    highlight_number: int | None = DEFAULT_HIGHLIGHT_NUMBER,
) -> np.ndarray:
    """Draw the 3-layer calibration confirmation overlay on a COPY of
    `frame` (never mutates the input) using `calib`
    (`opendarts.pipeline.CameraCalibration`) to project board-plane
    geometry into this camera's pixel space. `highlight_number=None`
    skips layer 2 (the wedge highlight) -- everything else still draws.
    """
    out = frame.copy()

    # Layer 1: soft blue wash over the whole double-ring disk.
    disk = _ring_polygon_px(calib, DOUBLE_OUTER_RADIUS_MM)
    wash_layer = out.copy()
    cv2.fillPoly(wash_layer, [disk.astype(np.int32)], WASH_COLOR_BGR)
    cv2.addWeighted(wash_layer, WASH_ALPHA, out, 1.0 - WASH_ALPHA, 0, out)

    # Layer 2: filled highlight on the identified sector's wedge.
    if highlight_number is not None:
        wedge = _wedge_polygon_px(calib, highlight_number)
        red_layer = out.copy()
        cv2.fillPoly(red_layer, [wedge.astype(np.int32)], HIGHLIGHT_COLOR_BGR)
        cv2.addWeighted(red_layer, HIGHLIGHT_ALPHA, out, 1.0 - HIGHLIGHT_ALPHA, 0, out)

    # Layer 3a: ring outlines (bull/treble/double).
    for name, radius_mm in RING_RADII_MM.items():
        poly = _ring_polygon_px(calib, radius_mm)
        thickness = 2 if name in _THICK_RINGS else 1
        cv2.polylines(out, [poly.astype(np.int32)], True, WIRE_COLOR_BGR, thickness, cv2.LINE_AA)

    # Layer 3b: 20 radial wire lines, outer-bull boundary -> double-outer
    # boundary, at each real wire angle.
    for number in SECTOR_NUMBERS_CLOCKWISE:
        angle_deg = wire_boundary_angle_deg(number)
        p_inner = project_board_point_px(polar_to_xy_mm(OUTER_BULL_RADIUS_MM, angle_deg), calib)
        p_outer = project_board_point_px(polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, angle_deg), calib)
        cv2.line(
            out,
            (int(round(p_inner[0])), int(round(p_inner[1]))),
            (int(round(p_outer[0])), int(round(p_outer[1]))),
            WIRE_COLOR_BGR,
            1,
            cv2.LINE_AA,
        )

    return out


def draw_calibration_overlay_rgba(
    size_wh: "tuple[int, int]",
    calib,
    *,
    highlight_number: int | None = DEFAULT_HIGHLIGHT_NUMBER,
) -> np.ndarray:
    """The SAME overlay as draw_calibration_overlay(), rendered onto a
    transparent BGRA canvas instead of onto a camera frame.

    Returns HxWx4 (BGRA, straight/un-premultiplied alpha) for
    `size_wh = (width, height)`. Needs NO frame -- only the calibration
    and the pixel dimensions to project into -- which is the whole point:

    **`size_wh` MUST be the frame size `calib` was solved at.** This
    projects through `calib.camera_matrix` and does not rescale, so the
    geometry lands wherever that matrix puts it. Passing a smaller canvas
    does not shrink the board to fit -- it renders full-resolution
    coordinates on a small canvas, i.e. a board too large and pushed
    toward the bottom-right, cropped at the edges. That shipped once, as
    a 960-wide preview-sized overlay over a 1280-wide calibration. To
    show this at another size, render at the native size and scale the
    IMAGE (which is what the dashboard does -- the browser stretches the
    layer and the video stream to the same box).

    `draw_calibration_overlay()` returns an opaque picture, a copy of the
    frame with the overlay painted into it. That makes it a REPLACEMENT
    for the live view, not something that can sit on top of one, and it
    is why the dashboard used to drop a camera tile off its MJPEG stream
    onto a 3-second still the moment that camera calibrated -- a working
    rig got a slideshow and a broken one got video. Layering needs an
    image that is transparent everywhere the overlay does not draw, and
    that is this function.

    Composited over the frame by the browser, the result is intended to
    be pixel-equivalent to the baked version -- tests/test_board_overlay_
    rgba.py asserts exactly that against draw_calibration_overlay()
    rather than trusting the two to stay in step by inspection. Both
    functions read the same module-level colours, alphas, radii and
    thicknesses and call the same projection helpers, so the geometry
    cannot drift; only the compositing differs.

    Cheap by construction: no frame grab, no contention with the MJPEG
    stream for frames, and one render per recalibration rather than one
    per camera every few seconds.
    """
    width, height = int(size_wh[0]), int(size_wh[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"size_wh must be positive, got {size_wh!r}")

    # Accumulated in PREMULTIPLIED float so layers can be composited with
    # ordinary source-over and the result stays correct where they
    # overlap (the wedge highlight sits on top of the wash on every real
    # board). Un-premultiplied once, at the end.
    rgb_premul = np.zeros((height, width, 3), dtype=np.float32)
    alpha = np.zeros((height, width), dtype=np.float32)

    def _composite(mask: np.ndarray, color_bgr: "tuple[int, int, int]", layer_alpha: float) -> None:
        """Source-over one layer. `mask` is 0..255 -- antialiased edges
        arrive as intermediate values and become intermediate alpha,
        which is what keeps the lines as smooth here as they are when
        cv2 draws them straight onto a frame."""
        src_a = (mask.astype(np.float32) / 255.0) * layer_alpha
        if not src_a.any():
            return
        src_a3 = src_a[:, :, None]
        color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
        np.multiply(rgb_premul, (1.0 - src_a3), out=rgb_premul)
        rgb_premul[...] += color * src_a3
        np.multiply(alpha, (1.0 - src_a), out=alpha)
        alpha[...] += src_a

    def _blank_mask() -> np.ndarray:
        return np.zeros((height, width), dtype=np.uint8)

    # Layer 1: soft blue wash over the whole double-ring disk.
    wash_mask = _blank_mask()
    disk = _ring_polygon_px(calib, DOUBLE_OUTER_RADIUS_MM)
    cv2.fillPoly(wash_mask, [disk.astype(np.int32)], 255)
    _composite(wash_mask, WASH_COLOR_BGR, WASH_ALPHA)

    # Layer 2: filled highlight on the identified sector's wedge.
    if highlight_number is not None:
        wedge_mask = _blank_mask()
        wedge = _wedge_polygon_px(calib, highlight_number)
        cv2.fillPoly(wedge_mask, [wedge.astype(np.int32)], 255)
        _composite(wedge_mask, HIGHLIGHT_COLOR_BGR, HIGHLIGHT_ALPHA)

    # Layer 3a/3b: ring outlines and radial wires. Drawn into ONE mask,
    # then composited once -- drawing them separately would let each
    # antialiased line composite over the previous one and darken every
    # crossing, which the baked version does not do.
    wire_mask = _blank_mask()
    for name, radius_mm in RING_RADII_MM.items():
        poly = _ring_polygon_px(calib, radius_mm)
        thickness = 2 if name in _THICK_RINGS else 1
        cv2.polylines(wire_mask, [poly.astype(np.int32)], True, 255, thickness, cv2.LINE_AA)
    for number in SECTOR_NUMBERS_CLOCKWISE:
        angle_deg = wire_boundary_angle_deg(number)
        p_inner = project_board_point_px(polar_to_xy_mm(OUTER_BULL_RADIUS_MM, angle_deg), calib)
        p_outer = project_board_point_px(polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, angle_deg), calib)
        cv2.line(
            wire_mask,
            (int(round(p_inner[0])), int(round(p_inner[1]))),
            (int(round(p_outer[0])), int(round(p_outer[1]))),
            255,
            1,
            cv2.LINE_AA,
        )
    _composite(wire_mask, WIRE_COLOR_BGR, 1.0)

    out = np.zeros((height, width, 4), dtype=np.uint8)
    opaque_enough = alpha > 0.0
    straight = np.zeros_like(rgb_premul)
    np.divide(rgb_premul, alpha[:, :, None], out=straight, where=opaque_enough[:, :, None])
    out[:, :, :3] = np.clip(np.rint(straight), 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return out
