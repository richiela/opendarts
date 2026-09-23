"""Ares's fusion: ONE unified robust weighted least-squares solve over
per-camera shaft LINES (strong evidence) and tip POINTS (weak evidence),
seeded from a consensus hypothesis chosen by the engine.

Why lines, not points (measured, 2026-08-24, full 374-throw opendarts corpus,
944 per-camera reads vs AD's own tip_xy_mm, plane z=0, no corrections):

    along-shaft error:  median |e| 3.60mm, median e -2.77mm (biased)
    perpendicular:      median |e| 0.74mm, median e +0.28mm (unbiased)

Every camera's board-plane shaft line passes through the dart's true
entry point (the plane through the camera centre and the dart's 3D axis
meets the board plane in that line -- a projective fact, no
reconstruction involved), so line evidence carries the 0.74mm-class
perpendicular signal. Tip points carry the full ~3.6mm-class error but
constrain BOTH directions -- exactly what lines cannot do alone when
they are few or near-parallel.

The unified solve makes the old special cases (single point, point
median, near-parallel "fill the weak eigendirection") fall out of one
estimator: minimize

    sum_lines  (w_i / sigma_line^2)  * huber(perp_i(x);  delta_line)
  + sum_points (v_j / sigma_point^2) * huber(|x - t_j|; delta_point)

via IRLS. sigma_line/sigma_point are the measured error scales above, so
a full-quality line is worth (sigma_p/sigma_l)^2 = 16x a full-quality
point per unit weight -- lines dominate wherever they can see, points
take over exactly where lines are blind (along-line directions,
single-camera reads). Any point evidence at all makes the normal matrix
nonsingular, so there is no degenerate-geometry special case left.

- At two good lines + tips, the solution is the pairwise intersection
  (explicitly allowed 2D evidence) nudged by the tips -- redundancy the
  exact-intersection version measurably lacked (three of the 357-run's
  ring misses were 2-line exact intersections a few mm radially off
  while the tips sat on the right ring).
- Huber IRLS from the consensus seed keeps one bad line's pull bounded;
  the engine's consensus scoring already excluded evidence inconsistent
  with the chosen hypothesis, so IRLS only has to polish, not rescue.
- No 2-of-3 label vote anywhere -- fusion is continuous in board mm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Measured error scales (see module docstring). These set the RELATIVE
# strength of line vs point evidence, not any absolute gate.
SOLVE_SIGMA_LINE_MM = 1.0
SOLVE_SIGMA_POINT_MM = 4.0
# Huber knees, in residual mm: past these an observation's pull grows
# linearly instead of quadratically. Line knee ~= p90 of good-line perp
# error; point knee ~= p75-p80 of tip-point error (both measured).
HUBER_DELTA_LINE_MM = 2.5
HUBER_DELTA_POINT_MM = 6.0
IRLS_ITERATIONS = 20
CONVERGENCE_MM = 1e-4


@dataclass
class LineObservation:
    point: np.ndarray  # (2,) a point on the line, board mm
    direction: np.ndarray  # (2,) unit direction
    weight: float
    cam: int


@dataclass
class PointObservation:
    xy: np.ndarray  # (2,) board mm
    weight: float
    cam: int


@dataclass
class FusionResult:
    xy_mm: tuple[float, float]
    method: str
    n_lines_used: int
    n_points_used: int
    per_line_residual_mm: dict[int, float]
    condition_ratio: float | None
    spread_mm: float | None


def line_intersection(
    p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray
) -> np.ndarray | None:
    """Intersection of two board-plane lines, or None when near-parallel."""
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cross
    return p1 + t * d1


def fuse(
    lines: list[LineObservation],
    points: list[PointObservation],
    init_xy: np.ndarray | None = None,
) -> FusionResult | None:
    """The unified IRLS solve described in the module docstring. Returns
    None only when the evidence cannot determine a point at all (e.g.
    lines-only near-parallel geometry with no points -- callers always
    pass tip points alongside lines, so in practice: no evidence)."""
    if not lines and not points:
        return None

    if init_xy is not None:
        x = np.asarray(init_xy, dtype=np.float64).copy()
    elif points:
        ws = np.array([max(1e-9, p.weight) for p in points])
        x = np.average(np.array([p.xy for p in points]), axis=0, weights=ws)
    else:
        x = np.mean(np.array([ln.point for ln in lines]), axis=0)

    w_l = np.array([max(0.0, ln.weight) for ln in lines]) / SOLVE_SIGMA_LINE_MM**2
    w_p = np.array([max(0.0, p.weight) for p in points]) / SOLVE_SIGMA_POINT_MM**2

    evals = None
    for _ in range(IRLS_ITERATIONS):
        A = np.zeros((2, 2))
        b = np.zeros(2)
        for ln, w in zip(lines, w_l):
            nvec = np.array([-ln.direction[1], ln.direction[0]])
            r = abs(float(nvec @ (x - ln.point)))
            hub = 1.0 if r <= HUBER_DELTA_LINE_MM else HUBER_DELTA_LINE_MM / r
            ww = w * hub
            A += ww * np.outer(nvec, nvec)
            b += ww * nvec * float(nvec @ ln.point)
        for pt, v in zip(points, w_p):
            r = float(np.linalg.norm(x - pt.xy))
            hub = 1.0 if r <= HUBER_DELTA_POINT_MM else HUBER_DELTA_POINT_MM / r
            vv = v * hub
            A += vv * np.eye(2)
            b += vv * pt.xy
        evals, _ = np.linalg.eigh(A)
        if evals[0] <= 1e-12:
            # Only possible with zero points and degenerate lines.
            return None
        x_new = np.linalg.solve(A, b)
        if float(np.linalg.norm(x_new - x)) < CONVERGENCE_MM:
            x = x_new
            break
        x = x_new

    residuals: dict[int, float] = {}
    for ln in lines:
        nvec = np.array([-ln.direction[1], ln.direction[0]])
        residuals[ln.cam] = abs(float(nvec @ (x - ln.point)))

    return FusionResult(
        xy_mm=(float(x[0]), float(x[1])),
        method="unified",
        n_lines_used=len(lines),
        n_points_used=len(points),
        per_line_residual_mm=residuals,
        condition_ratio=(
            float(evals[0] / evals[1]) if evals is not None and evals[1] > 0 else None
        ),
        spread_mm=float(np.median(list(residuals.values()))) if residuals else None,
    )

