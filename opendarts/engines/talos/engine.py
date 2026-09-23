"""TalosEngine -- the registry entry (`name = "Talos"`), and the thin
orchestration that ties this package's three focused pieces together per
thrown dart:

1. `opendarts.engines.talos.shaft_line.fit_shaft_line_px()` -- per camera,
   find the dart's blob and a 2D shaft line through its tip.
2. `opendarts.engines.talos.consensus` -- the 2D observation is the
   shaft-line centerline tip (`tip_px`), frozen `score_dart()`, then
   `_pair_if_sector_compromise()`. After triangulation, the 3D point is
   slid along the shaft-plane dart axis to BOARD_HIT_Z_MM (-1.5mm, the
   sisal behind the wire face) before pair/C/D/overrides.
   **2026-08-24**: the outward-cap centroid (`_outward_edge_pixel`) is
   no longer the live observation. On the 374-throw opendarts corpus it was
   the source of Talos's one-sided +2.3mm radial bias vs AD (cap-median
   Δr +5.2mm, CL-median Δr +0.11mm, shaft-axis Δr +0.13mm). Skipping it
   is +7/-3 on that corpus (363→367/374); the function remains as a
   measured primitive, not a scoring input.
3. `opendarts.engines.talos.plane_geometry.intersect_planes_with_board()`
   -- the line∩plane fallback used only when ray triangulation itself
   cannot score (fewer than 2 usable ray pixels). Centerline overrides
   apply here too.

**What is different from Apollo / today's score_dart()**
Apollo finds one tip pixel per camera (narrow-end mean of the opened
blob), back-projects those pixels to 3D rays, takes the closest point
to the rays, and scores (X, Y) of that point (drops Z).

Talos uses the same blob as detect_tip(); the 2D observation is that
blob's centerline tip pixel. An earlier outward-cap centroid (pixels
within OUTWARD_CAP_MM of max board-radius in the tip window) was built
to avoid picking a silhouette *corner*, but on the 2026-08-24 living
corpus it is a systematic outward bias, not a corner-correction -- see
the dated note in this docstring's item 2. Line∩plane is kept as a
fallback when ray triangulation cannot score.

Blob *selection* is deliberately the same as detect_tip() (same diff /
elongation ranking). The 2D line (PCA of the tip-ward portion) is
still computed for the fallback path and diagnostics.

A camera whose usable shaft span is too short to define a direction is
dropped (geometric observability, not a score gate). Three planes that
are not concurrent (3-plane residual in the explosion tail) fall back
to the 2-plane pair whose reconstructed line is least incompatible with
the unused plane -- analog of score_dart()'s 2-of-3 ray fallback, with
a residual gate sitting in a measured gap between concurrent and
explosion modes. See `shaft_line.MIN_SHAFT_SPAN_PX` /
`plane_geometry.CONCURRENCE_GATE_MM`.

No calibrate() -- uses the primary engine's already-solved pose. Live
capture only runs this when it is listed in also-run; it never writes
the primary result.json fields. Offline:
`replay_throw_with_engine(..., "Talos")` / `rescore_all(..., engine="Talos")`.

**2026-08-13 -- restructured from a single 807-line `talos.py` module
into this subpackage**, matching the shape `opendarts.engines.apollo`/
`opendarts.engines.athena` already use (thin `engine.py` adapter + focused
internal modules) -- a pure move, every real tuned constant/threshold
preserved exactly (proven at the time by a real corpus-wide before/after:
150/169 = 88.8% on data/archive/clean/, identical either side). See `opendarts/engines/talos/__init__.py` for
the full module-by-module mapping.
"""
from __future__ import annotations

import math

import numpy as np

from opendarts.engines.base import EngineResult
from opendarts.engines.talos.confidence import calibrated_confidence, throw_confidence
from opendarts.engines.talos.consensus import (
    FAR_CAP_RADIUS_MM,
    _pair_if_sector_compromise,
    apply_centerline_overrides,
    drop_far_cap_rays,
    one_cam_cl_pullback,
    one_cam_z0,
)
from opendarts.engines.talos.plane_geometry import (
    BOARD_HIT_Z_MM,
    _pixel_board_xy,
    dart_axis_xy_at_z,
    intersect_planes_with_board,
    plane_from_image_line,
    slide_point_along_dart_axis,
)
from opendarts.engines.talos.prior_dart import apply_prior_erase
from opendarts.engines.talos.blob_detection import PRECOMPUTE_REQUIREMENTS
from opendarts.engines.talos.shaft_line import fit_shaft_line_px
from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM, sector_ring_for_point
from opendarts.pipeline import CameraCalibration, ScoreResult

# score_dart() is Talos's own frozen copy of the ray-triangulation-with-
# fallback strategy it was actually built and measured against -- see
# ray_fallback.py's own module docstring for why this is a local, frozen
# dependency and not a live import from opendarts.pipeline (where this
# historically lived) or opendarts.engines.apollo.scoring (where it lives
# today, but has kept evolving independently since Talos's own lineage
# diverged from it).
from opendarts.engines.talos.ray_fallback import score_dart


def _override_observation(reason: str) -> str:
    if "centerline ring lock" in reason:
        return "centerline_unanimous_ring"
    if "lonely centerline" in reason:
        return "axis_ring_lonely_cl"
    if "double-to-outer" in reason:
        return "centerline_double_to_outer"
    if "split centerline leftover" in reason:
        return "centerline_split_leftover_axis"
    if "split centerline" in reason:
        return "centerline_split_mean"
    if "2-of-3 in-sector ring majority" in reason:
        return "centerline_insector_ring_majority"
    if "pair unanimous centerline" in reason:
        return "centerline_pair_unanimous_axis_lock"
    if "unanimous centerline" in reason:
        return "centerline_unanimous_sector_lock"
    if "cap walked" in reason:
        return "centerline_cap_walked_radial"
    if "shaft snap" in reason:
        return "shaft_snap_consensus"
    if "one-CL outside rescue" in reason:
        return "centerline_lonely_cl_axis_outside_rescue"
    if "outside rescue" in reason:
        return "centerline_outside_rescue"
    if "single-cam centerline pullback" in reason:
        return "single_cam_cl_pullback"
    return "centerline_override"


def _engine_result_from_score_dart(scored, extra: dict) -> EngineResult:
    return EngineResult(
        ok=scored.ok,
        sector=scored.sector,
        ring=scored.ring,
        board_xy_mm=scored.board_xy_mm,
        reason=scored.reason,
        diagnostics={
            "engine": "Talos",
            "observation": "centerline_ray",
            "n_cameras_used": scored.n_cameras_used,
            **extra,
        },
    )


def _stamp_confidence(result, cl_pixels, calibration, axis_xy) -> EngineResult:
    # 2026-08-14: result.confidence/diagnostics["confidence"] are now the
    # CALIBRATED value (opendarts.engines.talos.confidence.
    # calibrated_confidence(), LOSO-validated, directly comparable to
    # Apollo's/Athena's own calibrated confidence) -- the raw
    # rule-based mapping-agreement score this file's own docstring
    # describes is kept under diagnostics["raw_confidence"] rather than
    # discarded, since it's still a real, useful diagnostic signal, just
    # not the "official" cross-engine-comparable value anymore.
    obs = (result.diagnostics or {}).get("observation")
    raw = throw_confidence(
        cl_pixels, calibration, result, axis_xy=axis_xy, observation=obs,
    )
    c = calibrated_confidence(raw)
    result.confidence = c
    result.diagnostics["confidence"] = c
    result.diagnostics["raw_confidence"] = raw
    return result


def _plane_miss(reason: str, planes, common, line_diags, dropped) -> EngineResult:
    return EngineResult(
        ok=False,
        sector=None,
        ring=None,
        board_xy_mm=None,
        reason=reason,
        diagnostics={
            "engine": "Talos",
            "n_planes": len(planes),
            "line_px": line_diags,
            "dropped_cameras": dropped,
            "n_common": len(common),
        },
    )


def score(
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    calibration: dict[int, CameraCalibration],
    prior_board_xy_mm: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None = None,
    *,
    precomputed: dict | None = None,
) -> EngineResult:
    """`precomputed`: optional per-camera `opendarts.imageops.DiffCrop`
    (2026-09-06 perf pass) -- the shared gray/|diff|/blur front end a
    caller (Zeus) computed once for all its sub-engines, on the frames
    as passed in. Forwarded to `fit_shaft_line_px()`, which validates
    and may ignore it; a camera whose frame the prior-dart erase
    modified is re-fit without it (the bundle describes the original
    frame). Results are bit-identical with or without it."""
    prior_extra: dict = {}
    # 2026-09-01: fits the prior-erase ghost gate already computed on
    # the ORIGINAL frames -- reused below for every camera whose frame
    # the gate did not modify, instead of re-running the whole
    # fit_shaft_line_px() detection pass on identical inputs (this
    # duplicate pass was ~57ms/throw, the single reason Talos was the
    # slowest engine on every dart 2/3 of a visit). A camera whose
    # frame WAS erased is absent from this dict and gets a fresh fit.
    prior_fits: dict = {}
    if prior_board_xy_mm:
        frame_images, erased, prior_fits = apply_prior_erase(
            bg_images, frame_images, calibration, prior_board_xy_mm,
            precomputed=precomputed,
        )
        if erased:
            if precomputed:
                precomputed = {c: v for c, v in precomputed.items() if c not in erased}
            prior_extra = {
                "prior_dart_erased_cameras": [int(c) for c in erased],
                "prior_n_priors": len(tuple(prior_board_xy_mm)),
            }
    common = sorted(set(bg_images) & set(frame_images) & set(calibration))
    planes: list[tuple[int, np.ndarray, float]] = []
    line_diags: dict[int, dict] = {}
    dropped: dict[int, dict] = {}
    ray_pixels: dict[int, tuple[float, float]] = {}
    cl_pixels: dict[int, tuple[float, float]] = {}
    for cam in common:
        if cam in prior_fits:
            fitted = prior_fits[cam]
        else:
            # `precomputed` is only passed when there is a bundle for this
            # camera -- test doubles keep the plain two-argument signature.
            pc_kwargs = {"precomputed": precomputed[cam]} if precomputed and cam in precomputed else {}
            fitted = fit_shaft_line_px(bg_images[cam], frame_images[cam], **pc_kwargs)
        if fitted is None:
            continue
        p1, p2, line_diag = fitted
        line_diag.pop("opened_pts", None)
        if line_diag.get("dropped"):
            dropped[cam] = line_diag
            line_diags[cam] = line_diag
        else:
            line_diags[cam] = line_diag
            plane = plane_from_image_line(p1, p2, calibration[cam])
            if plane is not None:
                n, c = plane
                planes.append((cam, n, c))
        tip = line_diag.get("tip_px")
        if not tip:
            continue
        pixel = (float(tip[0]), float(tip[1]))
        cl_pixels[cam] = pixel
        ray_pixels[cam] = pixel

    extra = {
        "line_px": line_diags,
        "dropped_cameras": dropped,
        "n_planes": len(planes),
        **prior_extra,
    }
    kept, far = drop_far_cap_rays(ray_pixels, calibration)
    if far:
        extra = {
            **extra,
            "far_cap_radius_mm": FAR_CAP_RADIUS_MM,
            "far_dropped_cap_r_mm": {
                str(cam): round(r, 2) for cam, r in sorted(far.items())
            },
        }

    use = kept
    plane_list = [(n, c) for _, n, c in planes]
    axis_xy = dart_axis_xy_at_z(plane_list) if len(plane_list) >= 2 else None

    def _finish(result: EngineResult) -> EngineResult:
        return _stamp_confidence(result, cl_pixels, calibration, axis_xy)

    if len(use) == 1:
        cam, px = next(iter(use.items()))
        scored = one_cam_z0(cam, px, calibration)
        if scored is not None and scored.ok:
            extra = {**extra, "observation": "single_cam_z0_after_far_drop"}
            pulled = one_cam_cl_pullback(
                cam, cl_pixels, calibration, scored, axis_xy=axis_xy,
            )
            if pulled is not None:
                extra = {
                    **extra,
                    "observation": _override_observation(pulled.reason),
                    "pre_centerline_override_bed": [scored.sector, scored.ring],
                }
                scored = pulled
            override = apply_centerline_overrides(
                cl_pixels, calibration, scored,
                cap_pixels=ray_pixels, axis_xy=axis_xy, planes=planes,
                far_cams=sorted(far),
            )
            if override is not None:
                extra = {
                    **extra,
                    "observation": _override_observation(override.reason),
                    "pre_centerline_override_bed": [scored.sector, scored.ring],
                }
                scored = override
            return _finish(_engine_result_from_score_dart(scored, extra))
        use = ray_pixels
    elif len(use) < 2:
        use = ray_pixels

    if len(use) >= 2:
        ray_scored = score_dart(use, calibration)
        if ray_scored.ok:
            tri = ray_scored.triangulation
            xyz = None if tri is None else getattr(tri, "point_xyz", None)
            slid_xy = slide_point_along_dart_axis(xyz, plane_list)
            if slid_xy is not None:
                sector, ring = sector_ring_for_point(slid_xy[0], slid_xy[1])
                extra = {
                    **extra,
                    "board_hit_z_mm": BOARD_HIT_Z_MM,
                    "pre_axis_slide_bed": [ray_scored.sector, ray_scored.ring],
                }
                ray_scored = ScoreResult(
                    ok=True,
                    sector=sector,
                    ring=ring,
                    board_xy_mm=slid_xy,
                    triangulation=tri,
                    n_cameras_used=ray_scored.n_cameras_used,
                    cameras_used=ray_scored.cameras_used,
                    reason=ray_scored.reason,
                    max_ray_disagreement_mm=ray_scored.max_ray_disagreement_mm,
                    outlier_camera=ray_scored.outlier_camera,
                    alt_candidates_used=ray_scored.alt_candidates_used,
                )
            pair = _pair_if_sector_compromise(use, calibration, ray_scored)
            scored = pair if pair is not None else ray_scored
            if pair is not None:
                extra = {
                    **extra,
                    "observation": "outward_edge_ray_pair_sector_compromise",
                    "compromise_full_bed": [ray_scored.sector, ray_scored.ring],
                    "compromise_pair_cameras": list(pair.cameras_used or []),
                }
            override = apply_centerline_overrides(
                cl_pixels, calibration, scored,
                cap_pixels=ray_pixels, axis_xy=axis_xy, planes=planes,
                far_cams=sorted(far),
            )
            if override is not None:
                extra = {
                    **extra,
                    "observation": _override_observation(override.reason),
                    "pre_centerline_override_bed": [scored.sector, scored.ring],
                }
                scored = override
            return _finish(_engine_result_from_score_dart(scored, extra))

    # 2026-08-18 -- line∩plane fallback: leftover-dart cameras whose
    # centerline already projected off the board still contribute a
    # shaft plane, and that plane pulls a single real onboard shaft
    # across a ring wire (005-S2: cam0 inner, cam1/cam2 CL r=224-284).
    # If exactly one onboard-CL plane remains, that camera's own Z=0
    # is the observation -- same as the far-cap one-cam path, scoped
    # to this fallback so ray triangulation is unchanged.
    def _cl_onboard(cam: int) -> bool:
        px = cl_pixels.get(cam)
        if px is None or cam not in calibration:
            return False
        xy = _pixel_board_xy(px, calibration[cam])
        return (
            xy is not None
            and math.hypot(xy[0], xy[1]) <= DOUBLE_OUTER_RADIUS_MM
        )

    onboard_planes = [(cam, n, c) for cam, n, c in planes if _cl_onboard(cam)]
    if len(onboard_planes) == 1 and len(planes) >= 2:
        cam = onboard_planes[0][0]
        px = ray_pixels.get(cam) or cl_pixels.get(cam)
        if px is not None:
            scored = one_cam_z0(cam, px, calibration)
            if scored is not None and scored.ok:
                extra = {
                    **extra,
                    "observation": "single_cam_z0_onboard_plane",
                    "fallback_dropped_offboard_planes": [
                        int(c) for c, _, _ in planes if c != cam
                    ],
                }
                # Axis from the full plane set includes the off-board
                # leftover cameras we just dropped. Lonely-CL / pullback
                # would trust that poisoned axis over this camera's own
                # inner cap (005-S2). No independent 2nd onboard shaft
                # exists, so there is no axis to corroborate.
                axis_one = None
                pulled = one_cam_cl_pullback(
                    cam, cl_pixels, calibration, scored, axis_xy=axis_one,
                )
                if pulled is not None:
                    extra = {
                        **extra,
                        "observation": _override_observation(pulled.reason),
                        "pre_centerline_override_bed": [
                            scored.sector, scored.ring,
                        ],
                    }
                    scored = pulled
                override = apply_centerline_overrides(
                    cl_pixels, calibration, scored,
                    cap_pixels=ray_pixels, axis_xy=axis_one, planes=onboard_planes,
                    far_cams=sorted(far),
                )
                if override is not None:
                    extra = {
                        **extra,
                        "observation": _override_observation(override.reason),
                        "pre_centerline_override_bed": [
                            scored.sector, scored.ring,
                        ],
                    }
                    scored = override
                return _finish(_engine_result_from_score_dart(scored, extra))

    if len(planes) < 2:
        return _finish(_plane_miss(
            (
                f"need >=2 shaft planes to reconstruct a 3D dart axis, "
                f"got {len(planes)} (of {len(common)} cameras)"
            ),
            planes, common, line_diags, dropped,
        ))

    hit = intersect_planes_with_board([(n, c) for _, n, c in planes])
    if hit is None:
        return _finish(_plane_miss(
            "dart axis parallel to the board (or degenerate planes)",
            planes, common, line_diags, dropped,
        ))

    xy, geom = hit
    used_idx = geom.get("planes_used_indices") or list(range(len(planes)))
    cameras_used = [planes[i][0] for i in used_idx]
    sector, ring = sector_ring_for_point(xy[0], xy[1])
    fallback = EngineResult(
        ok=True,
        sector=sector,
        ring=ring,
        board_xy_mm=(float(xy[0]), float(xy[1])),
        reason="",
        diagnostics={
            "engine": "Talos",
            "observation": "line_plane_fallback",
            "n_planes": len(planes),
            "cameras_used": cameras_used,
            "line_px": line_diags,
            "dropped_cameras": dropped,
            **geom,
        },
    )
    override = apply_centerline_overrides(
        cl_pixels, calibration, fallback,
        cap_pixels=ray_pixels, axis_xy=axis_xy, planes=planes,
        far_cams=sorted(far),
    )
    if override is not None:
        extra = {
            **extra,
            "observation": _override_observation(override.reason),
            "pre_centerline_override_bed": [fallback.sector, fallback.ring],
            "fallback_was_line_plane": True,
        }
        return _finish(_engine_result_from_score_dart(override, extra))
    return _finish(fallback)


class TalosEngine:
    """The registry entry named "Talos" -- see opendarts/engines/registry.py."""

    name = "Talos"
    # What a caller-shared `opendarts.imageops.DiffCrop` must satisfy for
    # this engine to use it -- see `fit_shaft_line_px()`'s `precomputed`.
    precompute_requirements = PRECOMPUTE_REQUIREMENTS

    def score(
        self,
        bg_images: dict[int, np.ndarray],
        frame_images: dict[int, np.ndarray],
        calibration: dict[int, CameraCalibration],
        prior_board_xy_mm: tuple[tuple[float, float], ...] | list[tuple[float, float]] | None = None,
        *,
        precomputed: dict | None = None,
    ) -> EngineResult:
        return score(
            bg_images, frame_images, calibration,
            prior_board_xy_mm=prior_board_xy_mm,
            precomputed=precomputed,
        )
