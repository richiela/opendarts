"""Talos's own frozen copy of the ray-triangulation-with-fallback
`score_dart()` strategy it depends on as a real, load-bearing fallback
(not just for comparison -- see `engine.py`'s own call site).

**Why a frozen local copy instead of importing the live one.** This
function used to live in `opendarts.pipeline` (where Talos's own code was
originally written and measured against, on the branch it was rescued
from -- see docs/DESIGN.md's real dated entries and
`docs/DESIGN.md`'s "Talos restructured into a subpackage" entry for the
full 150/169=88.8% / 161/169=95.3% proof this exact behavior was
verified against). Since then, on `main`, this function moved into
`opendarts.engines.apollo.scoring` and kept evolving there as part of
Apollo's own real, measured tuning (this project's own history records
real changes here after Talos's lineage diverged -- e.g. the 2-of-3
RANSAC-skip-at-4-points work). Importing today's live version at merge
time would have silently changed Talos's actual scored output with no
new measurement behind it -- caught for real: doing exactly that during
this merge broke the exact-match assertions against the pre-refactor
baseline.

This module is Talos's own frozen dependency instead: the exact
`score_dart()` implementation (verbatim, from `opendarts.pipeline` as of
the commit Talos's real implementation was rescued from) that Talos
was actually built and measured against, kept local so it can never
silently drift out from under Talos again. It is NOT the same object as
`opendarts.engines.apollo.scoring.score_dart` -- the two have likely
diverged since, and that's fine; each engine owns its own opinion about
this fallback now. Reuses only genuinely-stable shared primitives
(`back_project_ray`/`triangulate` from `opendarts.triangulation.rays`,
`sector_ring_for_point` from `opendarts.geometry.board`,
`ScoreResult` from `opendarts.pipeline`, unchanged since this was written).

A real architectural question worth revisiting later, not solved here:
this function's ray-triangulation-with-fallback strategy is apparently
useful to more than one engine, which suggests it may actually be
universal/shared machinery rather than any one engine's private opinion
-- but reconciling Apollo's evolved version with this frozen one into
a single shared implementation is a real, separate task (needs its own
measurement to confirm it doesn't change either engine's proven
numbers), not something to do silently as a side effect of this merge.
"""
from __future__ import annotations

import itertools


from opendarts.calibration.pnp import MIN_LANDMARK_HULL_AREA_FRACTION
from opendarts.geometry.board import sector_ring_for_point
from opendarts.pipeline import CameraCalibration, ScoreResult
from opendarts.triangulation.rays import Ray, TriangulationResult, back_project_ray, triangulate

MAX_RAY_DISAGREEMENT_MM = 10.0
MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR = 5.0


def score_dart(
    tip_pixels: dict[int, tuple[float, float]],
    calibrations: dict[int, CameraCalibration],
    alt_tip_pixels: dict[int, tuple[float, float]] | None = None,
) -> ScoreResult:
    """The end-to-end scoring call: given per-camera tip pixel detections
    and per-camera calibration, produce a sector/ring score via 3D
    triangulation of the camera rays.

    tip_pixels: {cam_index: (x, y)} -- only cameras that saw the tip
    (ok=True in the source detection) should be included; this function
    does not itself distinguish "camera didn't see it" from "camera
    wasn't asked" -- that's the caller's job (the same convention the
    reference dataset's own per-camera `ok` flag used).

    alt_tip_pixels: added 2026-08-12, real incident (throw
    throw_1786580447119).
    Optional {cam_index: (x, y)} -- a SECOND tip-pixel candidate for
    cameras whose TipDetectionResult.alt_tip_px was non-None, i.e. that
    camera's tip-vs-non-tip end call was genuinely ambiguous on that one
    image alone (no companion blob found, width comparison inconclusive).
    A camera with no entry here (the common case -- most detections are
    NOT ambiguous) stays fixed at its single tip_pixels[cam] value,
    exactly as before this parameter existed. When one or more cameras
    DO have an alt candidate, this function searches every combination of
    {primary, alt} across those cameras (every other camera fixed) --
    at most 2**3=8 for a 3-camera rig, brute-force, see ScoreResult.
    alt_candidates_used for which combination actually won. Only cameras
    already present in `tip_pixels` may have an entry here (an alt
    candidate with no primary makes no sense and is ignored -- see
    `common_cams` below).

    The <=1-camera case is an explicit,
    known degradation to old-architecture behavior -- NOT implemented
    here (this function only does real triangulation). A caller needing
    single-ray fallback must implement that separately and explicitly,
    never silently inside this "the new, better path" function.
    """
    common_cams = sorted(set(tip_pixels) & set(calibrations))
    if len(common_cams) < 2:
        return ScoreResult(
            ok=False,
            sector=None,
            ring=None,
            board_xy_mm=None,
            triangulation=None,
            n_cameras_used=len(common_cams),
            reason=(
                f"only {len(common_cams)} camera(s) with both a tip pixel "
                "and a calibration -- need >=2 for real triangulation; "
                "the <=1-camera fallback is "
                "the CALLER's explicit responsibility, not done here"
            ),
        )

    # Calibration-side check FIRST, independent of triangulation entirely
    # -- see the comment block above CameraCalibration.landmark_spread_ok
    # above. Deliberately checked before
    # triangulating: a badly-conditioned camera calibration isn't made
    # trustworthy by what the other cameras' rays happen to agree on.
    poorly_spread = [
        cam_idx for cam_idx in common_cams
        if calibrations[cam_idx].landmark_spread_ok is False
    ]
    if poorly_spread:
        return ScoreResult(
            ok=False,
            sector=None,
            ring=None,
            board_xy_mm=None,
            triangulation=None,
            n_cameras_used=len(common_cams),
            reason=(
                f"camera(s) {poorly_spread} had a poorly-spread landmark "
                "quad at calibration time (landmark_hull_area_fraction < "
                f"{MIN_LANDMARK_HULL_AREA_FRACTION}) -- that camera's pose "
                "can't be trusted regardless of ray agreement (this "
                "is exactly the correlated-bias case ray agreement "
                "alone cannot catch)"
            ),
        )

    full_cams = list(common_cams)

    def _rays_for(pixels: dict[int, tuple[float, float]]) -> dict[int, Ray]:
        rays: dict[int, Ray] = {}
        for cam_idx in full_cams:
            calib = calibrations[cam_idx]
            pixel = pixels[cam_idx]
            rays[cam_idx] = back_project_ray(
                pixel, calib.camera_matrix, calib.dist_coeffs, calib.rvec, calib.tvec, cam=cam_idx
            )
        return rays

    def _triangulate_subset(
        rays_by_cam: dict[int, Ray], cams: list[int]
    ) -> tuple[TriangulationResult, float | None]:
        result = triangulate([rays_by_cam[c] for c in cams])
        disagreement = max(result.per_ray_distance_mm) if result.ok else None
        return result, disagreement

    def _attempt(pixels: dict[int, tuple[float, float]]) -> dict:
        """Run the existing full-set-then-2-of-3-RANSAC-fallback
        procedure for ONE specific per-camera pixel assignment
        (unchanged logic, just parameterized -- see the comment block
        above MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR for why the fallback
        pair uses a stricter threshold than the full set)."""
        rays_by_cam = _rays_for(pixels)
        full_tri, full_disagreement = _triangulate_subset(rays_by_cam, full_cams)
        full_accepted = (
            full_tri.ok
            and full_disagreement is not None
            and full_disagreement <= MAX_RAY_DISAGREEMENT_MM
        )
        tri, max_disagreement, cams_used, outlier_camera = (
            full_tri, full_disagreement, full_cams, None
        )
        if not full_accepted and len(full_cams) >= 3:
            best: tuple[float, TriangulationResult, tuple[int, int]] | None = None
            for pair in itertools.combinations(full_cams, 2):
                pair_tri, pair_disagreement = _triangulate_subset(rays_by_cam, list(pair))
                if not pair_tri.ok or pair_disagreement is None:
                    continue
                if pair_disagreement > MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR:
                    continue
                if best is None or pair_disagreement < best[0]:
                    best = (pair_disagreement, pair_tri, pair)
            if best is not None:
                pair_disagreement, pair_tri, pair = best
                tri, max_disagreement = pair_tri, pair_disagreement
                cams_used = list(pair)
                outlier_camera = next(c for c in full_cams if c not in pair)
        accepted = (
            tri.ok and max_disagreement is not None and max_disagreement <= MAX_RAY_DISAGREEMENT_MM
        )
        return {
            "tri": tri,
            "max_disagreement": max_disagreement,
            "cams_used": cams_used,
            "outlier_camera": outlier_camera,
            "full_disagreement": full_disagreement,
            "accepted": accepted,
        }

    # Real incident, 2026-08-12 (throw
    # throw_1786580447119). Cameras present in `alt_tip_pixels` (intersected
    # with common_cams -- an alt for a camera with no primary tip_px is
    # meaningless) had a genuinely ambiguous monocular tip-end call;
    # search every {primary, alt} combination across just those cameras
    # (every other camera fixed at its single tip_pixels value), brute
    # force, at most 2**3=8 for a 3-camera rig. The very FIRST combo
    # generated below is always the all-primary one (product() preserves
    # each per-camera choice-list's order, and "primary" is listed
    # first for every camera) -- this is relied on below so a total
    # rejection degrades to EXACTLY today's primary-only behavior, never
    # a different (even if superficially similar) rejection shaped by an
    # alt attempt that also failed.
    ambiguous_cams = sorted(
        c for c in full_cams if alt_tip_pixels and c in alt_tip_pixels
    )
    choice_lists = [
        (
            [("primary", tip_pixels[c]), ("alt", alt_tip_pixels[c])]
            if c in ambiguous_cams
            else [("primary", tip_pixels[c])]
        )
        for c in full_cams
    ]
    combos: list[tuple[dict[int, str], dict[int, tuple[float, float]]]] = []
    for combo in itertools.product(*choice_lists):
        labels = {c: label for c, (label, _px) in zip(full_cams, combo)}
        pixels = {c: px for c, (_label, px) in zip(full_cams, combo)}
        combos.append((labels, pixels))

    attempts = [(labels, _attempt(pixels)) for labels, pixels in combos]

    # Preference order, matching the single-combo code's own existing
    # preference (full ray set over a 2-of-3 fallback pair -- more rays
    # is more information) extended across combos: any combo whose FULL
    # ray set was accepted, lowest disagreement among those wins; else
    # any combo whose fallback PAIR was accepted, lowest disagreement
    # among those wins; else fall through to the all-primary combo's own
    # (failed) attempt, unchanged from today.
    full_accepted_attempts = [
        (labels, a) for labels, a in attempts
        if a["outlier_camera"] is None and a["accepted"]
    ]
    pair_accepted_attempts = [
        (labels, a) for labels, a in attempts
        if a["outlier_camera"] is not None and a["accepted"]
    ]
    if full_accepted_attempts:
        winner_labels, winner = min(
            full_accepted_attempts, key=lambda la: la[1]["max_disagreement"]
        )
    elif pair_accepted_attempts:
        winner_labels, winner = min(
            pair_accepted_attempts, key=lambda la: la[1]["max_disagreement"]
        )
    else:
        # No combination cleared any threshold -- degrade EXACTLY as
        # today: attempts[0] is always the all-primary combo (see
        # comment above), so this is bit-for-bit the same rejection
        # score_dart() would have produced with no alt_tip_pixels at all.
        winner_labels, winner = attempts[0]

    tri = winner["tri"]
    max_disagreement = winner["max_disagreement"]
    cams_used = winner["cams_used"]
    outlier_camera = winner["outlier_camera"]
    full_disagreement = winner["full_disagreement"]
    alt_cams_in_winner = tuple(
        c for c in full_cams if winner_labels.get(c) == "alt"
    )
    alt_candidates_used = alt_cams_in_winner if alt_cams_in_winner else None

    if not tri.ok:
        return ScoreResult(
            ok=False,
            sector=None,
            ring=None,
            board_xy_mm=None,
            triangulation=tri,
            n_cameras_used=len(cams_used),
            cameras_used=tuple(cams_used),
            reason=f"triangulation failed/untrustworthy: {tri.reason}",
            max_ray_disagreement_mm=max_disagreement,
            alt_candidates_used=alt_candidates_used,
        )

    if max_disagreement is None or max_disagreement > MAX_RAY_DISAGREEMENT_MM:
        # The critical finding, see MAX_RAY_DISAGREEMENT_MM
        # docstring above: rays.triangulate()'s own ok=True is not
        # sufficient here -- it doesn't check whether the rays actually
        # agree, only whether the geometry was well-conditioned and
        # forward-facing. This is the actual fix, not just a diagnostic.
        # Reaching this point means the RANSAC fallback above (if it even
        # ran) also failed to find any 2-camera pair that agrees --
        # reported against the FULL set for diagnostics, since that's
        # the most informative "why" for a caller/operator to see.
        return ScoreResult(
            ok=False,
            sector=None,
            ring=None,
            board_xy_mm=tri.board_plane_xy,
            triangulation=tri,
            n_cameras_used=len(cams_used),
            cameras_used=tuple(cams_used),
            reason=(
                f"rays disagree by {max_disagreement:.1f}mm (> "
                f"{MAX_RAY_DISAGREEMENT_MM}mm threshold) -- likely a bad "
                "per-camera calibration or tip detection (e.g. clustered/"
                "occluded landmark quad), not trustworthy even though "
                "the triangulation math "
                "itself succeeded; no 2-of-3 camera pair agreed well "
                "enough either"
                if max_disagreement is not None
                else "triangulation reported ok but no disagreement figure was available"
            ),
            max_ray_disagreement_mm=max_disagreement,
            alt_candidates_used=alt_candidates_used,
        )

    x_mm, y_mm = tri.board_plane_xy
    sector, ring = sector_ring_for_point(x_mm, y_mm)

    reason = ""
    if outlier_camera is not None:
        reason = (
            f"accepted via 2-of-3 RANSAC fallback: cameras {cams_used} "
            f"agreed to {max_disagreement:.1f}mm; camera {outlier_camera} "
            "excluded as the apparent outlier (full 3-ray set disagreed "
            f"by {full_disagreement:.1f}mm)"
            if full_disagreement is not None
            else f"accepted via 2-of-3 RANSAC fallback: cameras {cams_used} "
            f"agreed to {max_disagreement:.1f}mm; camera {outlier_camera} "
            "excluded as the apparent outlier"
        )
    if alt_cams_in_winner:
        # Real incident, 2026-08-12 (throw
        # throw_1786580447119) -- report which camera(s)' alternate,
        # previously-ambiguous tip candidate the winning combination
        # actually used, so a since-rescued throw is debuggable later
        # rather than looking like an ordinary accept.
        alt_note = (
            f"camera(s) {list(alt_cams_in_winner)} used their alternate "
            "ambiguous tip candidate (alt_tip_px, searched via "
            "score_dart()'s primary/alt combination search) instead of "
            "the primary width/texture-"
            "tiebreak guess"
        )
        reason = f"{reason}; {alt_note}" if reason else alt_note

    return ScoreResult(
        ok=True,
        sector=sector,
        ring=ring,
        board_xy_mm=(x_mm, y_mm),
        triangulation=tri,
        n_cameras_used=len(cams_used),
        cameras_used=tuple(cams_used),
        reason=reason,
        max_ray_disagreement_mm=max_disagreement,
        outlier_camera=outlier_camera,
        alt_candidates_used=alt_candidates_used,
    )


