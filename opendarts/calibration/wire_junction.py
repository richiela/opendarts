"""Local per-landmark wire-junction refinement -- the "read the point off
the image, not off the ellipse" stage.

WHY THIS EXISTS (measured, 2026-08-15 wire-crossing root-cause task; see
docs/DESIGN.md): `oriented_landmarks.find_oriented_landmarks()`
places every one of its 20 ring landmarks with `ellipse_ray_intersection`
-- the point is *constrained to lie on the globally fitted reseat
ellipse*, and the per-wire refine step is ANGULAR only. So any local
radial error of that shared ellipse passes straight into the landmark.
That error is real, systematic BIAS, not noise: mask spill onto fixed
scene features survives `_robust_fit_ellipse`'s algebraic band
(~±25px-wide at this rig's scale) in stable angular arcs and drags the
fit locally outward. Measured against the oracle's calibration quad (which
zoomed crops confirm sits on the true physical wire junctions): 40-frame
production-averaged deltas up to 7.4px / 4.0mm (cam1 board-angle-99),
with per-frame scatter of only ~0.2-2px -- N-frame averaging structurally
cannot remove it. A global conic simply cannot represent an error field
that is local to one part of the ring.

THE FIX: treat the ellipse landmark as a SEED only, then locate the true
junction from local image evidence. Each of the 20 landmarks is a
physical crossing of two thin bright metal wires:

  * the SECTOR ("spoke") wire, running board-radially -- its image is a
    straight line through the bull's image, exactly (homographies map
    lines through the board centre to lines through the bull);
  * the DOUBLE-OUTER RING wire, locally an arc parallel to the fitted
    ellipse.

Both are thin bright ridges (a white top-hat isolates them cleanly on
every camera -- wide bright regions like beds/branding are suppressed,
thin bright wires survive). The refinement:

  1. **Bounded matched-junction search** over a (tangential x radial)
     grid of candidate junction positions around the seed. Each
     candidate is scored by mean ridge energy along its two predicted
     wire arms (ring arc both directions + spoke line inward), sampled
     a few px away from the junction itself. The RING arm follows the
     radially-displaced ellipse arc, so ring curvature is modelled
     exactly rather than approximated by a tangent line. All window
     sizes are fractions of the locally-predicted double-bed width in
     pixels (from the frame's own locked homography), so nothing here
     hardcodes this rig's pixel scale. The radial window is deliberately
     capped below the double-bed width so the search can never reach the
     double-INNER wire's junction (the nearest lookalike feature).
  2. **Sub-pixel arm fitting**: from the best grid cell, measure the
     perpendicular ridge-peak offset (quadratic sub-pixel interpolation)
     at several stations along each arm, robust-fit a line per arm (one
     trim round), and intersect the two fitted lines. The spoke arm also
     uses the short OUTWARD wire overshoot beyond the ring (present on
     this board's spider, observed on real crops) when it carries real
     contrast -- extra leverage right at the junction.
  3. **Per-point quality gates + graceful fallback**: a refinement with
     too few valid arm stations, a bad line fit, a weak matched-filter
     contrast, or a result that ran to the window edge is REJECTED and
     that single landmark falls back to its ellipse-based seed. One
     occluded or washed-out junction never degrades the other 19.

Every gate threshold below was set from measured distributions on real
archived corpus frames (4 sessions x 3 cameras),
not guessed -- see each constant's own comment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from opendarts.calibration.landmark_detection import Ellipse

# --- Geometry of the search, all relative to local detected scale -------
#
# The one physical scale that matters here is the local pixel width of
# the double bed (the 162mm->170mm band) along this landmark's own ray,
# predicted from the frame's own locked homography. The nearest
# structure that could be mistaken for the target junction is the
# double-INNER wire's crossing, exactly one bed width inward -- so every
# window is sized as a fraction of that width, with the radial window
# capped strictly below it.
RADIAL_WINDOW_INWARD_FRACTION = 0.80    # of local bed width, toward the bull
RADIAL_WINDOW_OUTWARD_FRACTION = 0.55   # away from the board
# The tangential window may be wider than the radial one: radially the
# nearest lookalike (the double-INNER wire) is exactly one bed width
# away, but tangentially the nearest parallel structure is the
# neighbouring sector's spoke, a full 18 board-degrees (~10 bed widths)
# away. Too tight a window was measured to starve the spoke profile of
# background and kill its contrast z exactly where the tangential seed
# error was largest (cam2 board-angle-9: true offset -4px inside a
# +-6px window scored 1.6-3.4 sigma; widening restores the contrast).
TANGENTIAL_WINDOW_FRACTION = 1.20
RADIAL_WINDOW_MIN_PX = 5.0              # floors/caps: keep the window sane on
RADIAL_WINDOW_MAX_PX = 14.0             # very small/large boards in frame
TANGENTIAL_WINDOW_MIN_PX = 7.0
TANGENTIAL_WINDOW_MAX_PX = 16.0

# Arm sampling stations, px along the wire from the candidate junction.
# Start past the junction's own bright blob; end before the arm can run
# into unrelated structure (number ring digits sit ~a bed-width outside).
# Kept SHORT deliberately: the ring arm follows the radially-displaced
# global ellipse, and where the global fit is locally dragged (the exact
# failure this module exists to fix) the true wire diverges from that
# displaced arc as the arm gets longer -- a 16px arm measurably starved
# the true junction of stations at the worst-bias point (cam1
# board-angle-99) while a 12px arm keeps every station on the wire.
ARM_NEAR_PX = 4.0
ARM_FAR_PX = 12.0
ARM_STEP_PX = 1.5
# Spoke overshoot beyond the ring wire (stage 2 only): short, real,
# observed on this board's spider on every camera.
OVERSHOOT_NEAR_PX = 4.0
OVERSHOOT_FAR_PX = 9.0

# Perpendicular scan half-width for stage-2 ridge centring. Must be
# comfortably under half the bed width (so a scan from the true wire
# cannot centre on the neighbouring wire) and over the worst residual
# error of the stage-1 grid (1px grid -> <=0.7px diagonal residual).
PERP_SCAN_HALF_PX = 3.5
PERP_SCAN_STEP_PX = 0.5

# Lateral-inhibition offset for stage-1 station scoring, px. A station's
# evidence is ridge(p) - mean(ridge(p +- LATERAL_PX * perpendicular)):
# a thin wire centred on the arm scores its full height, while broad
# bright texture (number-ring digits, branding, washed-out bed paint)
# subtracts itself out. Sized to clear the wire's own blurred footprint
# (~2-4px on every camera measured) without reaching the neighbouring
# bed wire.
LATERAL_PX = 3.0

# Top-hat structuring element, relative to local bed width: the wire is
# ~1.5mm against an 8mm bed, i.e. ~0.2 bed widths. A kernel ~0.55 bed
# widths passes the wire (and even a slightly blurred wire) while
# suppressing anything bed-scale or wider.
TOPHAT_KERNEL_BED_FRACTION = 0.55
TOPHAT_KERNEL_MIN_PX = 5
TOPHAT_KERNEL_MAX_PX = 13

# --- Quality gates (values measured before being set) ----------------
#
# Minimum stations per fitted arm. The ring arm has up to 12 stations
# (6 each side), the spoke up to 6 inward (+4 optional overshoot);
# demanding half still tolerates one fully-occluded half-arm (a dart
# shaft lying along the ring, say).
MIN_RING_STATIONS = 6
MIN_SPOKE_STATIONS = 4
# A station only counts when its perpendicular ridge peak carries real
# contrast over the arm's own background. Measured on real frames the
# on-wire peak response is 5-30x the window's 25th-percentile top-hat
# level; 1.6x is far below every real wire station observed and above
# flat noise.
STATION_MIN_PEAK_RATIO = 1.6
# Line-fit RMS gates, px, per arm, measured on 620 real refinement
# attempts (12 bg frames of one recorded
# session + the 40 live frames of the prior investigation, all
# stage-2 gates disabled, every candidate's delta vs the oracle's calibration
# recorded, then every gate combination simulated offline). The RING
# gate is the load-bearing one:
# harmful refinements' ring-arm RMS median was 1.35-1.84 across sweeps
# vs 0.18-0.23 for improving ones, and every systematic failure cluster
# (the branding-occluded cam0 board-angle-189 region, the foreshortened
# 5px-bed cam0 board-angle-279 region) carries ring RMS above 1. The
# SPOKE gate is deliberately loose: with the ring gate passed, sloppy
# spoke fits (RMS 1.3-1.9) still produced sub-1.5px full-mode junctions
# at cam2 board-angle-9 (measured on 51/51 frames), and tightening it
# only converted those wins into radial-only non-fixes.
MAX_RING_FIT_RMS_PX = 1.0
MAX_SPOKE_FIT_RMS_PX = 2.0
# Stage-1 per-axis profile contrast: robust z of the profile's peak
# against the profile's own distribution ((best - median) /
# (1.4826 * MAD)). A WEAK floor, not the primary gate: measured z on
# genuinely good refinements ranges 1.9-45 (narrow beds score low
# because half the window sits on real structure, inflating the MAD),
# so anything above ~4 starves real fixes. 2.0 rejects flat/washed
# windows while letting stage 2's much sharper station/RMS gates do the
# real discrimination -- the full grid simulation put this combination
# at 15 mildly-harmful accepts (13px summed) against a 2.6px -> 1.35px
# overall median improvement on 620 attempts.
MIN_RING_Z = 2.0
MIN_SPOKE_Z = 2.0
# The stage-2 intersection must stay within the stage-1 grid cell's
# neighbourhood -- an intersection that ran away means at least one arm
# locked onto something else.
MAX_STAGE2_SHIFT_PX = 3.0


@dataclass
class JunctionRefinement:
    """One landmark's refinement outcome. `xy` is ALWAYS usable: the
    refined junction when `ok`, the untouched seed when not.

    `mode` records HOW the point was refined:
      * "full" -- both wires' evidence passed; `xy` is the fitted
        ring-line x spoke-line intersection (both axes corrected).
      * "radial_only" -- the ring wire's evidence passed but the spoke's
        did not; `xy` is the seed slid radially onto the fitted ring
        wire (radial axis corrected, tangential left as seeded). This is
        a real, measured mode, not a consolation: at some landmarks the
        spoke wire is genuinely dim/washed while the ring wire and the
        radial bias are both clear.
      * "rejected" -- no gate-passing evidence; `xy` is the seed.
    """
    ok: bool
    xy: tuple[float, float]
    seed_xy: tuple[float, float]
    mode: str = "rejected"
    shift_px: float = 0.0
    ring_z: float = 0.0
    spoke_z: float = 0.0
    n_ring_stations: int = 0
    n_spoke_stations: int = 0
    ring_fit_rms_px: float = 0.0
    spoke_fit_rms_px: float = 0.0
    # Both candidate outputs, when computable, regardless of gating --
    # diagnostics for offline threshold measurement, never consumed by
    # production code (production reads `xy`).
    xy_full: tuple[float, float] | None = None
    xy_radial: tuple[float, float] | None = None
    reason: str = ""


def ridge_map(image_bgr: np.ndarray, kernel_px: int) -> np.ndarray:
    """White top-hat of the grayscale image: keeps thin bright structure
    (wires) and suppresses anything wider than `kernel_px` (beds, bare
    sisal, branding blocks). float32, same shape as the input."""
    import cv2

    k = max(3, int(kernel_px) | 1)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)


def _sample(ridge: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear samples of `ridge` at float coordinates; 0 outside."""
    import cv2

    shape = xs.shape
    out = cv2.remap(
        ridge,
        xs.astype(np.float32).reshape(1, -1),
        ys.astype(np.float32).reshape(1, -1),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    return out.reshape(shape)


def _local_bed_width_px(H: np.ndarray, bull_px, angle_deg: float,
                        r_inner_frac: float,
                        Hinv: np.ndarray | None = None) -> float:
    """Pixel width of the double bed along the ray at `angle_deg`,
    predicted exactly from the locked homography `H` (unit-disk board
    frame -> pixels): distance between the images of the two bed wires
    on this landmark's own board ray.

    `Hinv`, if given, must be `np.linalg.inv(H)` -- a pure caching hook
    (2026-09-11 calibration-speed pass: `refine_ring_junctions()` calls
    this 20 times per frame with the SAME `H`, and np.linalg.inv is
    deterministic, so hoisting the inverse changes nothing)."""
    from opendarts.calibration.oriented_landmarks import apply_homography

    bx, by = float(bull_px[0]), float(bull_px[1])
    if Hinv is None:
        Hinv = np.linalg.inv(H)
    # Unit-disk direction of this ray: preimage of a point 1px out along it.
    probe = apply_homography(Hinv, [(bx + math.cos(math.radians(angle_deg)),
                                     by + math.sin(math.radians(angle_deg)))])[0]
    n = float(math.hypot(probe[0], probe[1]))
    if not np.isfinite(n) or n < 1e-9:
        return 0.0
    d = probe / n
    outer = apply_homography(H, [d])[0]
    inner = apply_homography(H, [d * r_inner_frac])[0]
    if not (np.isfinite(outer).all() and np.isfinite(inner).all()):
        return 0.0
    return float(math.hypot(outer[0] - inner[0], outer[1] - inner[1]))


def suggested_tophat_kernel_px(bed_width_px: float) -> int:
    k = int(round(TOPHAT_KERNEL_BED_FRACTION * bed_width_px))
    k = max(TOPHAT_KERNEL_MIN_PX, min(TOPHAT_KERNEL_MAX_PX, k))
    return k | 1


def _quadratic_peak(vals: np.ndarray, idx: int) -> tuple[float, float]:
    """Sub-sample peak position (in index units) and height around a
    discrete argmax via 3-point parabola. Returns (di, height)."""
    if idx <= 0 or idx >= len(vals) - 1:
        return 0.0, float(vals[idx])
    ym, y0, yp = float(vals[idx - 1]), float(vals[idx]), float(vals[idx + 1])
    denom = ym - 2.0 * y0 + yp
    if abs(denom) < 1e-9 or y0 < ym or y0 < yp:
        return 0.0, y0
    di = max(-0.5, min(0.5, 0.5 * (ym - yp) / denom))
    return di, y0 - 0.25 * (ym - yp) * di


def _fit_arm_line(stations: np.ndarray, offsets: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares offset(s) = c0 + c1*s with one residual-trim round.
    Returns (c0, c1, rms) or None if degenerate.

    The trim threshold is MAD-based, not RMS-based: a single station
    centred on the wrong structure inflates the RMS enough that an
    RMS-multiple threshold fails to flag the very outlier that inflated
    it (measured: a 5px-off station among six clean ones left the fit
    0.56px off with a 2.5*RMS trim, and near-exact with this one). Only
    attempted with >= 5 stations -- a MAD over fewer points is noise.
    """
    s, u = np.asarray(stations, float), np.asarray(offsets, float)
    for _ in range(2):
        if len(s) < 3 or float(s.max() - s.min()) < 1e-6:
            return None
        A = np.stack([np.ones_like(s), s], axis=1)
        coef, *_ = np.linalg.lstsq(A, u, rcond=None)
        resid = u - A @ coef
        rms = float(np.sqrt(np.mean(resid ** 2)))
        if len(s) < 5:
            return float(coef[0]), float(coef[1]), rms
        med = float(np.median(resid))
        sigma = 1.4826 * float(np.median(np.abs(resid - med)))
        bad = np.abs(resid - med) > max(1.0, 3.0 * sigma)
        if not bad.any() or bad.all():
            return float(coef[0]), float(coef[1]), rms
        s, u = s[~bad], u[~bad]
    return float(coef[0]), float(coef[1]), rms


def refine_wire_junction(
    ridge: np.ndarray,
    ellipse: Ellipse,
    bull_px,
    angle_deg: float,
    seed_xy,
    *,
    bed_width_px: float,
) -> JunctionRefinement:
    """Refine ONE ring landmark from local ridge evidence.

    `ridge` is `ridge_map()`'s output for the whole frame; `angle_deg`
    is this wire's image angle about the bull (the same angle
    `find_oriented_landmarks` used to place the seed); `seed_xy` is the
    ellipse-based landmark; `bed_width_px` is `_local_bed_width_px()`'s
    prediction for this ray (pass 0 to let the refinement reject
    itself).
    """
    from opendarts.calibration.oriented_landmarks import ellipse_ray_intersections

    seed = (float(seed_xy[0]), float(seed_xy[1]))
    if not (np.isfinite(bed_width_px) and bed_width_px > 3.0):
        return JunctionRefinement(False, seed, seed,
                                  reason="degenerate local bed width")

    bx, by = float(bull_px[0]), float(bull_px[1])
    r_seed = math.hypot(seed[0] - bx, seed[1] - by)
    if r_seed < 10.0:
        return JunctionRefinement(False, seed, seed, reason="seed too close to bull")

    r_in = min(max(RADIAL_WINDOW_INWARD_FRACTION * bed_width_px,
                   RADIAL_WINDOW_MIN_PX), RADIAL_WINDOW_MAX_PX)
    r_out = min(max(RADIAL_WINDOW_OUTWARD_FRACTION * bed_width_px,
                    RADIAL_WINDOW_MIN_PX), RADIAL_WINDOW_MAX_PX)
    t_win = min(max(TANGENTIAL_WINDOW_FRACTION * bed_width_px,
                    TANGENTIAL_WINDOW_MIN_PX), TANGENTIAL_WINDOW_MAX_PX)

    # ------------------------------------------------------------------
    # Stage 1: bounded, AXIS-DECOMPOSED matched search.
    #
    # The two wires localise the two axes INDEPENDENTLY, by construction:
    # sliding a candidate along the ring arc leaves the ring-arm score
    # unchanged (the arc slides along itself), and sliding it along the
    # spoke line leaves the spoke-arm score unchanged. A joint 2-D argmax
    # over the sum therefore rides whichever arm is brighter and lets
    # noise place the other axis -- measured directly: over half of all
    # attempts pinned the argmax to a window edge along the
    # weakly-constrained axis. So instead:
    #
    #   1. the RING arm's evidence picks the radial offset `dr`
    #      (profile over dr, stations along the displaced arc);
    #   2. the SPOKE arm's evidence, evaluated AT that dr, picks the
    #      tangential offset `dt` (profile over dt).
    #
    # Each station is scored with LATERAL INHIBITION -- centre response
    # minus the mean response LATERAL_PX to either side, perpendicular to
    # the arm -- so only a thin ridge CENTRED on the arm counts; broad
    # bright texture (number-ring digits, branding) subtracts itself out.
    # Stations aggregate by a trimmed mean (middle ~2/3): a digit stroke
    # crossing the arm lights up 1-2 stations (trimmed away); the true
    # wire lights up nearly all of them.
    # ------------------------------------------------------------------
    dts = np.arange(-round(t_win), round(t_win) + 1, 1.0)
    drs = np.arange(-round(r_in), round(r_out) + 1, 1.0)
    dangs = np.degrees(dts / max(r_seed, 1e-6))                 # tangential px -> deg

    arm_s = np.arange(ARM_NEAR_PX, ARM_FAR_PX + 1e-9, ARM_STEP_PX)
    ring_s = np.concatenate([-arm_s[::-1], arm_s])              # both directions

    def _inhibited(base_x, base_y, perp_x, perp_y):
        c = _sample(ridge, base_x, base_y)
        p = _sample(ridge, base_x + LATERAL_PX * perp_x, base_y + LATERAL_PX * perp_y)
        m = _sample(ridge, base_x - LATERAL_PX * perp_x, base_y - LATERAL_PX * perp_y)
        return c - 0.5 * (p + m)

    def _trimmed(a: np.ndarray) -> np.ndarray:
        srt = np.sort(a, axis=-1)
        k = max(1, a.shape[-1] // 6)
        return srt[..., k:a.shape[-1] - k].mean(axis=-1)

    def _profile_peak(profile: np.ndarray) -> tuple[int, float]:
        med = float(np.median(profile))
        mad = float(np.median(np.abs(profile - med)))
        i = int(np.argmax(profile))
        z = (float(profile[i]) - med) / max(1.4826 * mad, 1e-6)
        return i, z

    # Ring-arm base geometry per station s (at the seed's own angle):
    # ellipse hit + its own radial unit, then displaced by each dr.
    ring_ang0 = angle_deg + np.degrees(ring_s / max(r_seed, 1e-6))
    hits = ellipse_ray_intersections(ellipse, (bx, by), ring_ang0)
    if not np.isfinite(hits).all():
        return JunctionRefinement(False, seed, seed,
                                  reason="ring arc left the ellipse")
    hr = np.hypot(hits[:, 0] - bx, hits[:, 1] - by)
    rux = (hits[:, 0] - bx) / hr
    ruy = (hits[:, 1] - by) / hr
    rx = hits[None, :, 0] + drs[:, None] * rux[None, :]          # (n_dr, n_s)
    ry = hits[None, :, 1] + drs[:, None] * ruy[None, :]
    ring_g = _inhibited(rx, ry, np.broadcast_to(rux[None, :], rx.shape),
                        np.broadcast_to(ruy[None, :], rx.shape))
    ring_vals = _sample(ridge, rx, ry)                           # raw, for stage-2 floor
    ring_profile = _trimmed(ring_g)                              # (n_dr,)
    bj, ring_z = _profile_peak(ring_profile)
    # The RING wire is mandatory: without it neither axis is trustworthy
    # (the radial axis IS the bias this module exists to fix).
    if ring_z < MIN_RING_Z:
        return JunctionRefinement(False, seed, seed, ring_z=ring_z,
                                  reason=f"weak ring-wire contrast ({ring_z:.2f} sigma)")
    if bj in (0, len(drs) - 1):
        return JunctionRefinement(False, seed, seed, ring_z=ring_z,
                                  reason="ring search hit the radial window edge")
    dr1 = float(drs[bj])

    # Spoke-arm samples per (dt, t): straight line inward from the
    # candidate junction at radial offset dr1. Direction: the candidate's
    # own ray unit; perpendicular for inhibition is its normal. The spoke
    # is OPTIONAL: a landmark whose spoke evidence fails its gates
    # degrades to a radial-only refinement instead of rejecting outright.
    spoke_ok = True
    spoke_note = ""
    hit0 = ellipse_ray_intersections(ellipse, (bx, by), angle_deg + dangs)
    if not np.isfinite(hit0).all():
        return JunctionRefinement(False, seed, seed, ring_z=ring_z,
                                  reason="spoke rays left the ellipse")
    sux = np.cos(np.radians(angle_deg + dangs))
    suy = np.sin(np.radians(angle_deg + dangs))
    c0x = hit0[:, 0] + dr1 * sux                                 # (n_dt,)
    c0y = hit0[:, 1] + dr1 * suy
    sx = c0x[:, None] - arm_s[None, :] * sux[:, None]            # (n_dt, n_t)
    sy = c0y[:, None] - arm_s[None, :] * suy[:, None]
    spoke_g = _inhibited(sx, sy,
                         np.broadcast_to(-suy[:, None], sx.shape),
                         np.broadcast_to(sux[:, None], sx.shape))
    spoke_vals = _sample(ridge, sx, sy)                          # raw, for stage-2 floor
    spoke_profile = _trimmed(spoke_g)                            # (n_dt,)
    bi, spoke_z = _profile_peak(spoke_profile)
    if spoke_z < MIN_SPOKE_Z:
        spoke_ok, spoke_note = False, f"weak spoke contrast ({spoke_z:.2f} sigma)"
        bi = int(np.argmin(np.abs(dts)))                         # dt = 0
    elif bi in (0, len(dts) - 1):
        spoke_ok, spoke_note = False, "spoke search hit the tangential window edge"
        bi = int(np.argmin(np.abs(dts)))

    p1 = np.array([c0x[bi], c0y[bi]])
    ang1 = angle_deg + float(dangs[bi])
    u1 = np.array([math.cos(math.radians(ang1)), math.sin(math.radians(ang1))])
    n1 = np.array([-u1[1], u1[0]])

    # ------------------------------------------------------------------
    # Stage 2: per-arm perpendicular ridge centring + line fits.
    # ------------------------------------------------------------------
    perp = np.arange(-PERP_SCAN_HALF_PX, PERP_SCAN_HALF_PX + 1e-9, PERP_SCAN_STEP_PX)
    # Background level of the local window: a LOW percentile of the raw
    # samples (most of the window is off-wire background). The median was
    # measured too high next to a bright specular spoke -- it culled every
    # ring station of a genuinely present, but dimmer, ring wire.
    floor = max(1.0, float(np.percentile(np.concatenate(
        [ring_vals.ravel(), spoke_vals.ravel()]), 25.0)))

    def _centre_stations(base_x, base_y, perp_x, perp_y):
        """base/perp arrays (n_st,), (n_st,): scan perpendicular at each
        station, return (kept_station_idx, offsets, peak_heights)."""
        xs = base_x[:, None] + perp[None, :] * perp_x[:, None]
        ys = base_y[:, None] + perp[None, :] * perp_y[:, None]
        vals = _sample(ridge, xs, ys)                            # (n_st, n_perp)
        idx = np.argmax(vals, axis=1)
        keep, offs, peaks = [], [], []
        for k in range(len(base_x)):
            di, h = _quadratic_peak(vals[k], int(idx[k]))
            if h < STATION_MIN_PEAK_RATIO * floor:
                continue
            off = (perp[int(idx[k])] + di * PERP_SCAN_STEP_PX)
            keep.append(k); offs.append(off); peaks.append(h)
        return np.asarray(keep, int), np.asarray(offs, float), np.asarray(peaks, float)

    # Ring arm: stations along the dr1-displaced arc through p1;
    # perpendicular = the local radial direction. The ring arm is
    # mandatory -- every failure here is a full rejection.
    ring_ang = ang1 + np.degrees(ring_s / max(r_seed, 1e-6))
    rh = ellipse_ray_intersections(ellipse, (bx, by), ring_ang)
    rr = np.hypot(rh[:, 0] - bx, rh[:, 1] - by)
    with np.errstate(all="ignore"):
        rpx = (rh[:, 0] - bx) / rr
        rpy = (rh[:, 1] - by) / rr
    rbx = rh[:, 0] + dr1 * rpx
    rby = rh[:, 1] + dr1 * rpy
    rk, roffs, _ = _centre_stations(rbx, rby, rpx, rpy)
    if len(rk) < MIN_RING_STATIONS:
        return JunctionRefinement(False, seed, seed, ring_z=ring_z, spoke_z=spoke_z,
                                  n_ring_stations=len(rk),
                                  reason=f"only {len(rk)} ring stations")
    ring_fit = _fit_arm_line(ring_s[rk], roffs)
    if ring_fit is None:
        return JunctionRefinement(False, seed, seed, ring_z=ring_z, spoke_z=spoke_z,
                                  reason="degenerate ring arm fit")
    r_c0, r_c1, r_rms = ring_fit
    if r_rms > MAX_RING_FIT_RMS_PX:
        return JunctionRefinement(False, seed, seed, ring_z=ring_z, spoke_z=spoke_z,
                                  ring_fit_rms_px=r_rms,
                                  reason=f"ring arm fit rms {r_rms:.2f}px")

    def ring_locus(s_val: float) -> np.ndarray:
        """Fitted ring-wire position at arc station s (px along the arc
        from p1): arc(s) + (dr1 + r_c0 + r_c1*s) * radial(s)."""
        ang = ang1 + math.degrees(s_val / max(r_seed, 1e-6))
        h = ellipse_ray_intersections(ellipse, (bx, by), [ang])[0]
        rv = np.array([h[0] - bx, h[1] - by])
        rv /= max(float(np.hypot(rv[0], rv[1])), 1e-9)
        return np.array([h[0], h[1]]) + (dr1 + r_c0 + r_c1 * s_val) * rv

    # The radial-only candidate: the SEED slid along its own ray onto the
    # fitted ring wire (tangential position untouched).
    s_seed = math.radians(((angle_deg - ang1 + 180.0) % 360.0) - 180.0) * r_seed
    xy_radial = ring_locus(s_seed)

    # Spoke arm: stations inward along the wire line, plus the short
    # outward overshoot; perpendicular = n1. Optional -- failures degrade
    # to radial-only.
    xy_full = None
    s_rms = 0.0
    n_spoke = 0
    if spoke_ok:
        spoke_t = np.concatenate([
            -np.arange(OVERSHOOT_NEAR_PX, OVERSHOOT_FAR_PX + 1e-9, ARM_STEP_PX),
            np.arange(ARM_NEAR_PX, ARM_FAR_PX + 1e-9, ARM_STEP_PX),
        ])
        sbx = p1[0] - spoke_t * u1[0]
        sby = p1[1] - spoke_t * u1[1]
        sk, soffs, _ = _centre_stations(sbx, sby,
                                        np.full_like(spoke_t, n1[0]),
                                        np.full_like(spoke_t, n1[1]))
        n_spoke = int(len(sk))
        # overshoot stations are optional extras; the INWARD count is
        # what the gate demands (the overshoot may legitimately not
        # exist on another board's spider).
        n_inward = int((spoke_t[sk] > 0).sum()) if len(sk) else 0
        spoke_fit = _fit_arm_line(spoke_t[sk], soffs) if n_inward >= MIN_SPOKE_STATIONS else None
        if spoke_fit is None:
            spoke_ok = False
            spoke_note = (f"only {n_inward} spoke stations"
                          if n_inward < MIN_SPOKE_STATIONS else "degenerate spoke arm fit")
        else:
            s_c0, s_c1, s_rms = spoke_fit
            if s_rms > MAX_SPOKE_FIT_RMS_PX:
                spoke_ok = False
                spoke_note = f"spoke arm fit rms {s_rms:.2f}px"

    if spoke_ok:
        # Intersect the two fitted arm lines. Ring side: a short chord
        # through s = +-ARM_NEAR_PX (arc curvature over that span is
        # <0.05px). Spoke side: p1 - t*u1 + (s_c0 + s_c1*t) * n1.
        a0, a1 = ring_locus(-ARM_NEAR_PX), ring_locus(ARM_NEAR_PX)
        b0 = p1 + OVERSHOOT_FAR_PX * u1 + (s_c0 - s_c1 * OVERSHOOT_FAR_PX) * n1
        b1 = p1 - ARM_FAR_PX * u1 + (s_c0 + s_c1 * ARM_FAR_PX) * n1
        da_v, db_v = a1 - a0, b1 - b0
        denom = da_v[0] * db_v[1] - da_v[1] * db_v[0]
        cross = abs(denom) / max(float(np.linalg.norm(da_v) * np.linalg.norm(db_v)), 1e-12)
        if cross < 0.17:  # arms within ~10 deg of parallel: no stable intersection
            spoke_ok, spoke_note = False, "arms nearly parallel"
        else:
            t_par = ((b0[0] - a0[0]) * db_v[1] - (b0[1] - a0[1]) * db_v[0]) / denom
            cand = a0 + t_par * da_v
            if float(np.hypot(cand[0] - p1[0], cand[1] - p1[1])) > MAX_STAGE2_SHIFT_PX:
                spoke_ok, spoke_note = False, "stage-2 intersection ran from matched cell"
            else:
                xy_full = cand

    refined = xy_full if (spoke_ok and xy_full is not None) else xy_radial
    mode = "full" if (spoke_ok and xy_full is not None) else "radial_only"
    shift = float(np.hypot(refined[0] - seed[0], refined[1] - seed[1]))
    return JunctionRefinement(
        ok=True,
        xy=(float(refined[0]), float(refined[1])),
        seed_xy=seed,
        mode=mode,
        shift_px=shift,
        ring_z=ring_z,
        spoke_z=spoke_z,
        n_ring_stations=len(rk),
        n_spoke_stations=n_spoke,
        ring_fit_rms_px=r_rms,
        spoke_fit_rms_px=s_rms,
        xy_full=(float(xy_full[0]), float(xy_full[1])) if xy_full is not None else None,
        xy_radial=(float(xy_radial[0]), float(xy_radial[1])),
        reason="ok" if mode == "full" else f"radial-only ({spoke_note})",
    )


def refine_ring_junctions(
    image_bgr: np.ndarray,
    ellipse: Ellipse,
    bull_px,
    wire_angles_deg,
    ring_px: np.ndarray,
    H: np.ndarray,
    r_inner_frac: float,
) -> list[JunctionRefinement]:
    """Refine all 20 ring landmarks of one frame. Returns one
    `JunctionRefinement` per landmark, index-aligned with `ring_px`;
    entries that failed a gate carry the seed in `.xy` (always usable).

    `H` is the frame's locked board->image homography (unit-disk frame,
    `board_to_image_homography` output) -- used only to predict the local
    double-bed pixel width per ray; `r_inner_frac` is the double-inner
    radius as a fraction of the double-outer (162/170 for a regulation
    board, from `opendarts.geometry.board` at the call site, not hardcoded
    here).
    """
    Hinv = np.linalg.inv(H)
    widths = [
        _local_bed_width_px(H, bull_px, float(a), r_inner_frac, Hinv=Hinv)
        for a in wire_angles_deg
    ]
    finite = [w for w in widths if np.isfinite(w) and w > 0]
    k = suggested_tophat_kernel_px(float(np.median(finite)) if finite else 0.0)
    ridge = ridge_map(image_bgr, k)
    return [
        refine_wire_junction(
            ridge, ellipse, bull_px, float(a), tuple(p), bed_width_px=w
        )
        for a, p, w in zip(wire_angles_deg, np.asarray(ring_px, float), widths)
    ]
