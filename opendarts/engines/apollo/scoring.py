"""Apollo's scoring strategy -- turning per-camera tip pixels + solved
calibrations into an actual sector/ring score. **Moved here 2026-08-12**
from `opendarts/pipeline.py` (`score_dart()` and everything specific to how
Apollo turns triangulated rays into an accepted-or-rejected score) --
see `opendarts/engines/apollo/__init__.py`'s module docstring for the
full move rationale and what stayed behind in `opendarts/pipeline.py`
(calibration-SOLVING and the genuinely-shared `CameraCalibration`/
`ScoreResult` dataclasses).

This is a MOVE, not a reimplementation -- every real number/threshold
below (`MAX_RAY_DISAGREEMENT_MM`, `MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR`)
and the full 2-of-3 RANSAC fallback / alt-tip-candidate combination-search
logic is preserved exactly as it was measured and tuned in
`opendarts/pipeline.py`; only the module changed.

**2026-08-12 -- alt-tip-candidate combination search in `score_dart()`
(real incident, throw `throw_1786580447119`).** See opendarts/engines/apollo/tip_detection.py's
module docstring for the full real-incident write-up: two darts landing
close together merged into one diff blob for cam1 and cam2, each camera's
monocular tip-vs-non-tip disambiguation picked the wrong end
(102.8px/158.3px off AD's real ground-truth tip respectively), and the
throw was rejected outright (`max_ray_disagreement_mm=32.4mm` on the
real package). `TipDetectionResult` now optionally carries `alt_tip_px`
for exactly the cases where a camera's own end-choice was genuinely
unresolved on that single image alone. `score_dart()` accepts a matching
optional `alt_tip_pixels` dict and, when given, brute-force searches
every {primary, alt} combination across just the cameras that have one
(at most 2**3=8 for a 3-camera rig), reusing the EXACT SAME
`back_project_ray`/`triangulate`/2-of-3-RANSAC-fallback machinery per
combination, unchanged -- picking whichever combination the existing
`per_ray_distance_mm` machinery agrees on best. Re-running the fixed
pipeline against the real `throw_1786580447119` package: cameras 1 and 2
both used their alt candidate, `max_ray_disagreement_mm` dropped
32.4mm->1.79mm, and the recovered `board_xy_mm` landed 2.48mm from AD's
real ground-truth tip (same sector AND ring: `2`/`single_inner`, exactly
matching AD) -- a genuinely recovered, trustworthy score, not a forced
pass.
Corpus regression check (129 real throw packages across every archived
rig session from 2026-08-12, re-scored with
this change on vs. off on the exact same `detect_tip()` outputs): 3
previously-rejected throws now score (including this exact incident),
**zero** throws regressed from ok=True to ok=False, and **zero**
already-correctly-scored throws changed sector or ring. Honest caveat:
~20 already-`ok=True` throws shifted their board_xy by a
small amount (mean 3.1mm, max 9.0mm, same sector/ring throughout) when
an alt candidate produced a lower-disagreement combination than the
primary-only one -- an expected consequence of always picking the
lowest-disagreement combination, not a new risk class.
"""
from __future__ import annotations

import itertools

from opendarts.calibration.pnp import MIN_LANDMARK_HULL_AREA_FRACTION
from opendarts.geometry.board import sector_ring_for_point
from opendarts.pipeline import CameraCalibration, ScoreResult
from opendarts.triangulation.rays import Ray, TriangulationResult, back_project_ray, triangulate

# Verifier pass 4 (2026-08-12) -- the critical finding of this pass:
# above this max per-ray perpendicular distance, treat a triangulation as
# untrustworthy even though rays.triangulate() itself reports ok=True
# (well-conditioned geometry + all rays in front of their cameras, but
# the rays may still simply DISAGREE on where the point is -- which
# happens when calibration for one camera is badly wrong in a way that
# doesn't show up as poor reprojection error or poor ray-set
# conditioning, e.g. coplanar 4-point PnP pose ambiguity from a
# clustered/occluded landmark quad). Demonstrated concretely: composing
# a clustered-quad-calibrated camera (a known risk) into score_dart()
# produced ok=True with a WRONG sector in 27% of
# 300 adversarial-but-plausible trials, with per_ray_distance_mm showing
# 34-47mm disagreement while condition_number stayed near the healthy
# baseline the whole time -- condition_number and reprojection error
# measure different things than ray agreement and neither one catches
# this. Measured healthy-case ceiling (well-spread quad, 0.5px noise, 60
# seeds): max 1.44mm. 10mm leaves >20x margin above healthy, while
# sitting far below the demonstrated bad-case range (34-47mm) --
# comfortable separation, not a knife-edge number.
MAX_RAY_DISAGREEMENT_MM = 10.0

# RANSAC-style 2-of-3 fallback (2026-08-12, real-throw investigation):
# the ORIGINAL policy above rejected the whole throw whenever the FULL
# ray set disagreed by more than MAX_RAY_DISAGREEMENT_MM, even when two
# of the three rays actually agreed closely and only one camera's
# detection was the problem -- demonstrated concretely on that real
# throw: cam0's ray disagreed with the other two by 27-31mm (a real
# dart-tip-detection bug since fixed, see opendarts/engines/apollo/
# tip_detection.py's companion-blob comment), but cam1+cam2 alone agreed
# to 2.5mm, comfortably inside the threshold. Discarding two good rays
# because a third was bad is not required by anything about the
# triangulation math -- this is the standard multi-view "try every
# minimal subset, keep whichever one the data actually supports" idea
# (RANSAC), applied here at the smallest useful scale (3 cameras -> at
# most 3 candidate pairs). score_dart() now tries the full ray set
# first (unchanged behavior, and still preferred when it agrees -- more
# rays is more information); only when the full set disagrees does it
# fall back to the single best-agreeing 2-camera pair, and only when
# THAT still fails does it reject outright.
#
# **The fallback pair is held to a STRICTER threshold than the full set
# (MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR, not MAX_RAY_DISAGREEMENT_MM)
# -- this is not cosmetic, it was required by a real measured failure.**
# A first version of this fallback reused MAX_RAY_DISAGREEMENT_MM
# (10.0mm) for pairs too, and it broke an existing, already-passing
# test: tests/test_pipeline_end_to_end.py's shared-systematic-calibration
# -bias probe went from "0 accepted-and-
# wrong across the sweep" to "60/60 accepted-and-wrong at 12px bias" --
# because when EVERY camera shares a correlated bias (not one genuinely
# bad ray), the best 2-of-3 PAIR can still agree with itself reasonably
# well purely by geometric chance, even though all three are jointly
# wrong. Measured directly to find a real,
# usable separation: in a genuine single-bad-ray scenario (clean
# calibration, one camera's detection corrupted -- the actual real-world
# case this fallback exists for) the GOOD pair's disagreement measured
# 0.15-0.70mm (synthetic, 60 seeds) and 2.5mm on the one real throw this
# was built from; in the correlated-shared-bias scenario, the BEST
# possible pair's disagreement measured 5.98-7.69mm even at the smallest
# bias (12px) that the full-set threshold already rejects -- a real,
# comfortable gap, not a knife-edge number, hence 5.0mm (roughly
# midway, well above the single-outlier ceiling, well below the
# correlated-bias floor). This does NOT prove every possible correlated-
# bias mechanism is caught by this specific gap (the correlated-bias
# honesty caveat still applies) -- it proves
# THIS specific, previously-passing test's scenario is still caught
# after adding the fallback, on the same measured numbers that scenario
# was built from.
MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR = 5.0

# 2026-08-12, batch8 mismatch investigation -- comment-only addition,
# no behavior change. An
# attempt to lower the trigger for preferring a 2-of-3 pair over the full
# set (from "full set disagrees >10mm" down to ">3mm") measured a real
# corpus-wide accuracy gain but regressed
# test_landmark_spread_gate_does_not_claim_to_catch_every_bias_mechanism
# (tests/test_pipeline_end_to_end.py) -- root-caused to the exact
# mechanism this constant's own docstring above already names: per-ray
# disagreement MAGNITUDE cannot separate "one genuinely bad ray" from
# "correlated bias across all three, coincidentally producing one
# tighter-looking pair." A real field instance of this same ambiguity
# was found in THIS already-shipped fallback (not the attempted change):
# real throw throw_1786593923839 (an archived 2026-08-12 session) has this exact fallback pick a pair that
# agrees to 0.33mm and lands on the wrong sector, while the pair it
# rejected would have landed on the right one. Not fixed here -- flagged
# so the next person tuning either threshold in this file knows the
# ambiguity is real and measured on live data, not just a synthetic
# what-if.


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
    throw_1786580447119 -- see opendarts/engines/
    apollo/tip_detection.py's module docstring for the full write-up).
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
    # (opendarts/pipeline.py). Deliberately
    # checked before triangulating: a badly-conditioned camera calibration
    # isn't made trustworthy by what the other cameras' rays happen to
    # agree on.
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
    # throw_1786580447119 -- see opendarts/engines/apollo/tip_detection.py's
    # module docstring). Cameras present in `alt_tip_pixels` (intersected
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
            "ambiguous tip candidate (opendarts.engines.apollo.tip_detection's "
            "alt_tip_px, searched via score_dart()'s primary/alt "
            "combination search) instead of the primary width/texture-"
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
