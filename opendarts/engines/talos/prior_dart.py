"""Prior-dart-in-visit context for Talos -- erase a ghost tip that is
actually last throw's dart, nudged into this throw's motion-diff.

Real incident: the recorded T15 throw. Visit dart 0 was D10; dart 1
is S15. The new dart bumped the old one. Cam2's diff merged both into
one blob and Talos picked the old dart's end. Cam 0 already had the new
hole (inner). Split-mean of the good cam0 centerline and the ghost cam2
centerline crossed the treble-inner wire. Athena kept the new dart on
cam2 at (765, 259).

**Not Apollo's `prior_dart_line_px` path.** That guard is a
perpendicular-distance-to-the-old-shaft check inside Apollo's own tip
detector. Talos's miss is a 2D blob-end pick, so the measured fix is
different: copy the background into a disk around each known prior
board-XY *on the poisoned camera only*, then re-detect. On T15 that
moves cam2's tip to (765, 259) and the 0+2 pair scores inner.

Gates, measured 2026-08-16 on living `data/archive/clean/` (705
AD-matched throws):

- Ungated erase of every prior disk: T15 wins, **+4/-19** (grouped
  new darts sit near the old hole too).
- Erase a camera whose tip is within `PRIOR_ERASE_RADIUS_PX` of a
  projected prior AND whose chosen-end MAE ratio is below
  `PRIOR_GHOST_MAE_RATIO` (the end looks like a wiggle, not a new
  hole): T15 wins, **+1/-2**.
- Same, plus require **two other cameras** whose tips are *outside*
  that radius (one poisoned view, two clean): **+1/-0**, T15 only.

Without `prior_board_xy_mm`, `score()` is byte-identical to before
this module existed. Live/replay must pass the list or the gate never
fires -- see `find_prior_board_xy_mm()`, called from both
`opendarts.live.capture_daemon` and `opendarts.capture.replay` so a package
captured live and later replayed sees the same prior XY.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from opendarts.engines.talos.blob_detection import N_TIP_POINTS_AVERAGED
from opendarts.geometry.board_color import project_board_point_px
from opendarts.pipeline import CameraCalibration

log = logging.getLogger(__name__)

# Image-space disk around a projected prior board-XY. T15 cam2's ghost
# tip sits 48.1px from D10; erase radius 48 re-picks Athena's pixel
# (765, 259) and the 0+2 pair scores inner. 50+ walks the remaining
# blob onto a leftover that scores 10. Keep 48.
PRIOR_ERASE_RADIUS_PX = 48

# A camera is a *candidate ghost* when its current tip is this close to
# a projected prior. T15 cam2 is 48.1px, so this must sit above 48.1.
# Separate from the erase disk on purpose: the detection gate is "is
# this tip sitting on the old dart", the disk is "how much of it to
# paint out so the new dart remains".
PRIOR_TIP_NEAR_PX = 52

# MAE(frame, bg) at the chosen tip end / MAE at the other blob end.
# T15 cam2 = 0.324 (nudge: low motion at the old dart, high motion at
# the new flights). Grouped new darts have a high ratio at the true
# tip (the hole just appeared). 0.35 sits above T15 and below the
# +1/-2 losses that a looser cut admitted.
PRIOR_GHOST_MAE_RATIO = 0.35

# Do not erase unless at least this many cameras have their tip
# OUTSIDE PRIOR_ERASE_RADIUS_PX. One poisoned view + two clean is the
# T15 pattern; grouping typically lights up every camera.
PRIOR_MIN_CLEAN_CAMERAS = 2

_MAE_PATCH_PX = 8


def engine_accepts_prior_board_xy_mm(engine: object) -> bool:
    """True only if `engine.score()` declares `prior_board_xy_mm`.

    Duck-typed capability check, same reason as Apollo's
    `engine_accepts_prior_dart_line_px`: tests substitute stub engines
    under a real registry name, and those stubs keep the 3-arg
    `score()` signature on purpose.
    """
    import inspect

    score = getattr(engine, "score", None)
    if score is None or not callable(score):
        return False
    try:
        params = inspect.signature(score).parameters
    except (TypeError, ValueError):
        return False
    return "prior_board_xy_mm" in params


def find_prior_board_xy_mm(
    session_dir: Path,
    visit_id: str | None,
    visit_index: int | None,
) -> tuple[tuple[float, float], ...]:
    """Board-plane XY of earlier darts in this visit, from sibling
    packages already on disk.

    Prefers each prior package's stored Talos also-run `board_xy_mm`
    (self-consistent with this engine; T15's disk is measured against
    that XY). Falls back to the package primary `result.json` board_xy
    (what live has if Talos was not also-run on the prior throw). Never
    reads the oracle's answer: ground truth grades a throw, it never
    feeds the scoring of the next one. Never raises. Empty tuple when
    there is nothing usable (no visit, first dart, missing session dir).
    """
    if visit_id is None or visit_index is None or visit_index <= 0:
        return ()
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        return ()
    found: list[tuple[int, tuple[float, float]]] = []
    try:
        throw_dirs = [p for p in session_dir.iterdir() if p.is_dir()]
    except OSError:
        return ()
    for throw_dir in throw_dirs:
        meta_path = throw_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("visit_id") != visit_id:
            continue
        idx = meta.get("visit_index")
        if idx is None or idx >= visit_index:
            continue
        xy = _board_xy_from_package_dir(throw_dir)
        if xy is not None:
            found.append((int(idx), xy))
    found.sort(key=lambda row: row[0])
    return tuple(xy for _, xy in found)


def find_prior_board_xy_mm_for_package(package_dir: Path) -> tuple[tuple[float, float], ...]:
    """Convenience: read this package's visit from its own meta.json."""
    package_dir = Path(package_dir)
    meta_path = package_dir / "meta.json"
    if not meta_path.exists():
        return ()
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return ()
    return find_prior_board_xy_mm(
        package_dir.parent, meta.get("visit_id"), meta.get("visit_index"),
    )


def _board_xy_from_package_dir(throw_dir: Path) -> tuple[float, float] | None:
    result_path = throw_dir / "result.json"
    data = None
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text())
        except (OSError, ValueError):
            data = None
    if isinstance(data, dict):
        talos = (data.get("other_engines") or {}).get("Talos") or {}
        if isinstance(talos, dict):
            xy = _xy_pair(talos.get("board_xy_mm"))
            if xy is not None:
                return xy
    if isinstance(data, dict):
        xy = _xy_pair(data.get("board_xy_mm"))
        if xy is not None:
            return xy
    return None


def _xy_pair(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None


ShaftFit = tuple[tuple[float, float], tuple[float, float], dict]


def apply_prior_erase(
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    calibration: dict[int, CameraCalibration],
    prior_board_xy_mm: tuple[tuple[float, float], ...] | list[tuple[float, float]],
    precomputed: dict | None = None,
) -> tuple[dict[int, np.ndarray], list[int], dict[int, ShaftFit | None]]:
    """Erase prior darts on ghost cameras.

    `precomputed`: optional per-camera `opendarts.imageops.DiffCrop` for the
    ORIGINAL frames, forwarded to the ghost gate's `fit_shaft_line_px()`
    calls (2026-09-06 perf pass; results are bit-identical with or
    without it). Never applies to an erased frame -- those are re-fit by
    the caller without it.

    Returns (frames, erased cams, shaft_fits). Frames dict is the input
    object when nothing erases (no copy).

    `shaft_fits` (2026-09-01 de-dup): the per-camera
    `fit_shaft_line_px()` results the ghost gate already computed on the
    ORIGINAL frames -- ~19ms/camera, by far the most expensive step in
    `score()`, and previously recomputed from scratch on the identical
    inputs by `score()`'s own main loop for every non-erased camera
    (i.e. all of them, on the overwhelmingly common no-ghost path).
    Keyed by camera for every common camera the gate examined; a value
    of None means the fit itself legitimately returned None (cache that
    too -- rerunning would return None again). A camera that actually
    got erased is REMOVED from the dict: its frame changed, so its
    cached fit is stale and `score()` must re-fit it.
    """
    priors = tuple(prior_board_xy_mm) if prior_board_xy_mm else ()
    if not priors:
        return frame_images, [], {}
    ghosts, fits = ghost_cameras(
        bg_images, frame_images, calibration, priors, precomputed=precomputed,
    )
    if not ghosts:
        return frame_images, [], fits
    out = dict(frame_images)
    for cam in ghosts:
        bg = bg_images.get(cam)
        frame = frame_images.get(cam)
        calib = calibration.get(cam)
        if bg is None or frame is None or calib is None:
            continue
        out[cam] = erase_priors_in_frame(bg, frame, calib, priors)
        fits.pop(cam, None)
    return out, ghosts, fits


def ghost_cameras(
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    calibration: dict[int, CameraCalibration],
    prior_board_xy_mm: tuple[tuple[float, float], ...],
    precomputed: dict | None = None,
) -> tuple[list[int], dict[int, ShaftFit | None]]:
    """Cameras whose current tip is a prior-dart wiggle, with two clean views.

    Also returns every `fit_shaft_line_px()` result computed while
    gating (keyed by camera, None cached as None), so the caller can
    reuse them instead of re-running the detection pass -- see
    `apply_prior_erase()`'s docstring.
    """
    from opendarts.engines.talos.shaft_line import fit_shaft_line_px

    dmin: dict[int, float] = {}
    ratio: dict[int, float | None] = {}
    fits: dict[int, ShaftFit | None] = {}
    common = sorted(set(bg_images) & set(frame_images) & set(calibration))
    for cam in common:
        pc_kwargs = {"precomputed": precomputed[cam]} if precomputed and cam in precomputed else {}
        fitted = fit_shaft_line_px(bg_images[cam], frame_images[cam], **pc_kwargs)
        fits[cam] = fitted
        if fitted is None:
            continue
        _p1, _p2, diag = fitted
        tip = diag.get("tip_px")
        pts = diag.get("opened_pts")
        if not tip:
            continue
        ds = []
        for xy in prior_board_xy_mm:
            px, py = project_board_point_px(xy, calibration[cam])
            ds.append(math.hypot(float(tip[0]) - px, float(tip[1]) - py))
        if not ds:
            continue
        dmin[cam] = min(ds)
        ratio[cam] = _mae_ratio(
            bg_images[cam], frame_images[cam], tip, pts,
        )
    n_clean = sum(1 for d in dmin.values() if d >= PRIOR_TIP_NEAR_PX)
    if n_clean < PRIOR_MIN_CLEAN_CAMERAS:
        return [], fits
    ghosts = [
        cam for cam, d in dmin.items()
        if d < PRIOR_TIP_NEAR_PX
        and ratio.get(cam) is not None
        and ratio[cam] < PRIOR_GHOST_MAE_RATIO
    ]
    return ghosts, fits


def erase_priors_in_frame(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    calib: CameraCalibration,
    prior_board_xy_mm: tuple[tuple[float, float], ...],
    radius_px: int = PRIOR_ERASE_RADIUS_PX,
) -> np.ndarray:
    """Copy bg into a disk around each projected prior so the bump
    disappears from the motion-diff."""
    import cv2

    mask = np.zeros(bg_bgr.shape[:2], dtype=np.uint8)
    for xy in prior_board_xy_mm:
        px, py = project_board_point_px(xy, calib)
        cv2.circle(mask, (int(round(px)), int(round(py))), int(radius_px), 255, -1)
    if not np.any(mask):
        return frame_bgr
    out = frame_bgr.copy()
    out[mask > 0] = bg_bgr[mask > 0]
    return out


def _mae_ratio(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    tip_px: tuple[float, float],
    opened_pts: np.ndarray | None,
) -> float | None:
    """MAE at the chosen blob end / MAE at the other end.

    Uses the opened-mask pixels of the same component `fit_shaft_line_px`
    already chose, so the ratio is the one measured on T15 / the 705.
    """
    import cv2

    if opened_pts is None or len(opened_pts) < 4:
        return None
    pts = np.asarray(opened_pts, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    if centered.shape[0] < 4:
        return None
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    principal = evecs[:, int(np.argmax(evals))]
    with np.errstate(all="ignore"):
        proj = centered @ principal
    if not np.all(np.isfinite(proj)):
        return None
    k = min(N_TIP_POINTS_AVERAGED, len(pts))
    pt_a = pts[np.argsort(proj)[:k]].mean(axis=0)
    pt_b = pts[np.argsort(-proj)[:k]].mean(axis=0)
    da = math.hypot(float(tip_px[0]) - float(pt_a[0]), float(tip_px[1]) - float(pt_a[1]))
    db = math.hypot(float(tip_px[0]) - float(pt_b[0]), float(tip_px[1]) - float(pt_b[1]))
    chosen, other = (pt_a, pt_b) if da <= db else (pt_b, pt_a)
    bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mae_c = _patch_mae(frame_gray, bg_gray, chosen)
    mae_o = _patch_mae(frame_gray, bg_gray, other)
    if mae_c is None or mae_o is None or mae_o < 1e-6:
        return None
    return mae_c / mae_o


def _patch_mae(frame_gray: np.ndarray, bg_gray: np.ndarray, xy, r: int = _MAE_PATCH_PX) -> float | None:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    h, w = frame_gray.shape
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return float(np.mean(np.abs(
        frame_gray[y0:y1, x0:x1].astype(np.float32)
        - bg_gray[y0:y1, x0:x1].astype(np.float32)
    )))
