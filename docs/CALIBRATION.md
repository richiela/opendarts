# Calibration methodology

What a calibration event does, phase by phase.

**When it runs.** The first Start calibrates automatically. After that the
saved calibration is reused across restarts, and is recalculated only when
you press **Calibrate** in the dashboard sidebar
(`POST /api/calibration/refresh`) — do that whenever a camera or the board
has moved. Entry point:
`opendarts.live.capture_daemon.bootstrap_calibrations()`.

**Output**, per camera: a `CameraCalibration` — `camera_matrix`,
`dist_coeffs`, `rvec`, `tvec`, plus the `pnp_result` carrying the
reprojection error the accept gate judged. That is the object every scoring
engine projects board geometry through.

**Standing constraint**: calibration
uses no external scorer's data, no constants copied from one,
and no template cropped from one physical board. Several of the design
choices below only make sense in that light — each is derived from scratch
rather than taken from a fixed reference.

---

## Phase 1 — Capture

`CALIBRATION_N_FRAMES = 50` independent frames per camera, from the shared
hub — not one frame, because per-frame landmark error has a component a
single frame cannot average away.

Of those, `CALIBRATION_N_FRAMES_DETECT = 5` are what the main round loop
actually runs detection and the solve on. Orientation resolution (phase 3)
runs earlier, in its own pass, on its own fixed 3 frames
(`RING_CORRELATION_ORIENTATION_N_FRAMES`).

## Phase 2 — Find the board in each frame

`oriented_landmarks.find_oriented_landmarks()`. Nine stages, each with a
real reject, so a bad step fails loudly instead of cascading:

1. **Illuminant normalisation** (`normalise_illuminant`) — white balance.
2. **Seed ellipse** — `detect_double_ring_quad()` fits the double ring.
3. **Bull** — `detect_bull()`, by a board-relative colour/shape signature.
4. **Bull-anchored re-seat** — `reseat_ellipse()` re-fits the ellipse using
   the bull as an anchor, then re-locks the bull to the re-seated ellipse.
5. **Angular edge profile** — `angular_edge_profile()` over edge magnitude.
6. **Projective phase lock** — `lock_phase()`. This is the heart of it, and
   the reason this module exists. The 20 wires are 18° apart *on the board*,
   but perspective does not preserve angular spacing, so a fixed
   reference-angle lookup is mis-registered by an amount that grows with
   camera obliquity. This solves for the board's real rotational phase in
   the image instead.
7. **Orientation lock** — `lock_orientation()`. Phase 6 locates the 20 wire
   points but only up to a rotation; this picks which rotation is real.
   Two signals in order: double-ring red/green alternation (period 2, so it
   narrows 20 candidates to 10 and provably cannot do better), then a
   per-camera orientation hint to choose among the 10, which are ~36° apart.
8. **Bounded per-wire refine** — `refine_wire_junction()` per junction.
9. **Quality gates** — phase confidence, colour margin.

## Phase 3 — Rig-consensus orientation

`rig_ring_geometry.resolve_rig_consensus_orientation()`. The hint phase 7
needs is the one thing not derivable from a single camera's view of the
board, so it is derived from the rig as a whole:

1. Cameras clearing `min_confidence` are candidate anchors. **Fewer than two
   → refuse.** At least two anchors must agree before either vouches for a
   third.
2. **Every** camera is cross-checked against what the others plus the ring
   geometry predict for it — including confident ones. A camera that merely
   clears the floor is not evidence of being right on this rig. Disagreement
   by more than half a sector rejects it regardless of confidence.

There is **no hardcoded fallback constant**: rig consensus or refuse.
`rig_move_reseed.py` detects when the rig has physically moved and the
stored absolute hint should no longer be trusted.

**When the gaps between cameras change**, by more than
`RING_GEOMETRY_DRIFT_THRESHOLD_DEG`, a camera has been moved or remounted.
The calibration relearns the rig's layout from its own event and carries
on, rather than refusing: it only gets here when every camera derived its
orientation live, so none of them leaned on the stored layout. The move is
reported instead — in the log, on the action log and on the calibration
panel (`ring_geometry_relearned`), with how far the gaps shifted. A camera
that cannot establish its orientation at all still refuses; relearning
would hide a real camera problem.

## Phase 4 — Correspondences

`correspond_landmarks_oriented()` returns **4 points**, not 20:
`AD_QUAD_RING_INDICES = (0, 5, 10, 15)` — four double-outer wire points 90°
apart. The convention is regulation board geometry, cross-checked against
`opendarts.geometry.board`'s independent wire-boundary-angle formula.

`sector_correspondence.average_correspondences()` then combines the
per-frame `(object_points_mm, image_points_px)` sets for a camera into one,
trimming per-(frame, landmark-index) outliers first.

## Phase 5 — Intrinsics, derived fresh every event

Not a stored per-camera-index constant: a camera plugged into a different
USB port changes which physical camera sits at an index, and a stale focal
length there cost a measured ~5% (3.25px reprojection against 2.94px).
Deriving from this event's own images tracks whatever camera is actually
there.

Method is Zhang's single-plane-homography orthogonality constraint, tried as
three tiers of increasing richness, each only adopted if self-consistent:

1. `derive_focal_length_from_oriented_results()` — f alone.
2. `derive_focal_and_k1_from_oriented_results()` — joint (f, k1).
3. `derive_focal_k1_cx_from_oriented_results()` — joint (f, k1, cx).
   cx only; cy is deliberately **not** solved for, on conditioning grounds.

Failing all three, a persisted last-known-good table is used
(`load_focal_length_fallback()` / `write_focal_length_fallback_entry()`).

## Phase 6 — Pose

`pnp.solve_extrinsics()` — `cv2.solvePnPRansac` by default, for outlier
rejection against imperfect real detections. With exactly 4 coplanar points
(`COPLANAR_POINT_COUNT`) it takes a documented non-RANSAC path.

## Phase 7 — Accept, or capture more

`CALIBRATION_TARGET_REPROJECTION_ERROR_PX = 2.5` (overridable per camera via
`reprojection_targets_px` in `config.json`). A camera under target is
accepted. Over target, capture another `CALIBRATION_RETRY_BATCH_SIZE = 25`
frames and re-solve from the **accumulated** pool — never restarted — up to
`CALIBRATION_MAX_N_FRAMES = 200`.

The best (lowest-error) calibration seen across all rounds is what gets
adopted, not the last one. The **Calibrate** button additionally runs
best-of-5 attempts per round (`CALIBRATION_N_REPROJECTION_ATTEMPTS`; the
automatic calibration at Start runs one). Because it only ever adopts an
improvement, it can match or improve a camera's error but never worsen it.

---

## Honest limits

- **The orientation hint is the one remaining rig dependency** (phase 7 of
  the per-frame pipeline). Two attempts to remove it were made and both
  failed on real data: "boards hang with 20 up and cameras are upright" was
  refuted by measurement (board angle 0 projected to 343.7° / 88.2° / 195.3°
  on cams 0/1/2 — tracking each camera's mounting azimuth, not any board
  property), and a template-free number-ring lock rectified legibly but every
  ink measurement tried was dominated by the illumination gradient rather
  than by digit content. Phase 3 is the mitigation, not a removal.
- **Retry accumulation bias is known and unfixed.** The trim threshold is
  computed per round rather than from the full accumulated pool. It is
  exercised by any camera that is genuinely bad, not merely by one missing a
  tight target.
