"""Real dart-tip detection: finding a thrown dart's tip pixel location in
a raw, unlabeled camera image -- the FIRST time this has been built in
this project (see docs/DESIGN.md / docs/DESIGN.md). Every
prior use of a tip pixel in this repo either used a synthetic position or
a label already extracted from an earlier reference dataset's own
(different, 2D) detector -- this module is
the first independent attempt to actually FIND the tip from pixels.

Deliberately decoupled from calibration/triangulation: given ONE
camera's (background, dart-frame) image pair, output
a `(pixel_x, pixel_y)` tip estimate. Combining per-camera estimates into
a 3D point is opendarts.triangulation's job, not this module's -- nothing
here reads calibration.json or does any board-geometry reasoning.

Approach (classical CV, diff-based; the specific parameters come from a
real-image exploration):

1. Grayscale absolute difference between a background frame (no dart --
   in practice this project's real case data uses `bg_cam{i}.png`, which
   is the board state immediately BEFORE this exact dart, not necessarily
   an empty board -- see "bg_cam is not always empty" below) and a frame
   known to contain the new dart, Gaussian-blurred to suppress
   single-pixel sensor noise.
2. Threshold the blurred diff at a fixed absolute level (see
   `DIFF_THRESHOLD`, tuned against real data -- see below). A first
   attempt at data-adaptive thresholding (Otsu, then percentile-based)
   was tried and abandoned: both were measurably WORSE than a fixed
   threshold on this real dataset, because the diff image's histogram
   has no clean bimodal separation (background lighting drift + sub-pixel
   misregistration between the two captures produces a long noise tail
   that swamps percentile/Otsu-style adaptive cutoffs -- confirmed by
   direct measurement, not assumed).
3. Morphological open (remove salt-and-pepper noise) then a much larger
   dilation to bridge real gaps WITHIN the dart's own silhouette (the
   thin shaft frequently threshold-fragments into 2-3 disconnected
   pieces at this threshold level -- confirmed on real images).
4. Connected-component analysis on the dilated mask; rank the top
   candidates by pixel area, and pick the first one (in area order) whose
   PCA-based elongation ratio (major/minor eigenvalue, sqrt) clears
   `MIN_ELONGATION_RATIO` -- a real dart silhouette (shaft + fletching)
   is always strongly elongated, which is what lets this step reject
   compact, non-dart-shaped diff blobs (see "known false-positive source"
   below) that happen to have larger raw area than the real dart.
5. Within the chosen component's ORIGINAL (non-dilated) pixels, run PCA
   to get a principal axis, project all points onto it, and compare the
   pixel spread (perpendicular to that axis) at each of the two extreme
   ends. The end with SMALLER perpendicular spread is called the tip --
   this directly encodes "the tip is the thin, converging end; the
   fletching end is wide" (docs/DESIGN.md's framing of this problem). When
   that width comparison is itself highly ambiguous (near-tied widths --
   see TEXTURE_TIEBREAK_RATIO_MIN's comment) and no companion blob was
   found (step 5b below), a local-background-texture tiebreak decides
   instead. The returned pixel is the mean of the 3 most extreme points
   at the chosen end (a small amount of averaging for stability, not true
   sub-pixel refinement -- see "Known limitations").
5b. Companion-blob disambiguation and the local-background-texture
   tiebreak (both added 2026-08-12, see COMPANION_MAX_GAP_PX's and
   TEXTURE_TIEBREAK_RATIO_MIN's comments below for the full real-evidence
   write-up): when the width comparison in step 5 is ambiguous, first
   check for a separated companion blob (a real, specific piece of
   evidence -- typically the dart's own flight, split off by a
   diff-mask gap wider than dilation bridges); if none is found AND the
   ambiguity is severe, fall back to comparing local background texture
   at each end (board face vs. ceiling/wall backdrop) as a weaker,
   more generic signal.

Measured, not guessed (run against the full real dataset available at
prototyping time -- 120 real
dart-folders x 3 real cameras = 354 usable (bg, frame, real tip label)
pairs, one physical session; the real tip label being compared against is
`liveThrow.cameras[].tipImage` from the SAME real image's `case.json`,
i.e. that reference dataset's own different 2D detector's output on that exact
frame -- used here strictly as a validation reference, never as ground
truth. External score/sector outputs are not trusted as truth anywhere in
this repo --
a raw per-camera pixel detection is a much narrower, more defensible
thing to compare against than a final scored sector):

  n=354, 0 hard failures (always returns SOME candidate pixel -- "ok"
  reflects confidence, not "did the pipeline crash")
  mean 19.98px | median 2.10px | p90 76.79px | p99 170.25px | max 707.82px
  fraction within  5px: 68.1%
  fraction within 10px: 81.1%
  fraction within 20px: 84.7%
  fraction within 50px: 87.0%

  (Companion-blob fix + this session's local-background-texture tiebreak,
  combined, both added 2026-08-12 -- previous numbers, companion-blob fix
  only: mean 23.05px | median 2.31px | p90 91.60px | p99 176.26px |
  max 707.82px | frac<20px 81.6%.)

**2026-08-13 -- re-measured on this same 354-pair corpus after
`DIFF_THRESHOLD` was re-tuned 30.0 -> 25.0 against a DIFFERENT and newer
corpus (this rig's own 349 archived throws -- see that constant's own
comment for the full sweep, the per-session cross-check, and the
end-to-end numbers).** Reported here because this corpus is a genuine
held-out set for that tune: it is a different physical session, and the
tune never looked at it.

  n=353 usable comparisons (see the honest regression noted below)
  mean 19.19px | median 2.00px | p90 64.24px | p99 272.64px |
  max 402.80px | frac<5px 68.3% | frac<10px 80.7% | frac<20px 85.6%
  cam0 mean 14.79px / p90 58.62px / frac<20px 84.0%
  cam1 mean 31.40px / p90 110.63px / frac<20px 80.2%
  cam2 mean 11.61px / p90 11.45px / frac<20px 92.4%

Better on mean (19.98 -> 19.19), median (2.10 -> 2.00), p90 (76.79 ->
64.24), max (707.82 -> 402.80) and frac<20px (84.7% -> 85.6%) -- the
held-out set agrees with the tuning set that the fat tail shrinks. TWO
honest regressions on this corpus, not hidden: **p99 got WORSE (170.25
-> 272.64px)** -- the far tail redistributed rather than uniformly
shrinking, so a handful of cases that used to be ~170-200px wrong are now
~270-400px wrong -- and **one pair (the recorded D8 throw cam2) that
previously returned a tip now hard-fails** with "largest diff component
has too few original-resolution pixels", which is why n is 353 rather
than 354. That one is a SAFE failure (an explicit ok=False the ROI gate
and score_dart both already handle), not a new confidently-wrong answer,
but it is a real behaviour change and it is counted against the tune, not
around it.

**Honest headline: this is a bimodal result, not a uniformly-accurate
one.** In the common case (now closer to 5 out of 6 real images, up from
roughly 3 out of 4 before this session) this gets close to the reference
detector's answer (median error 2.1px is comparable to
opendarts/calibration/landmark_detection.py's own 3.0px median on a
DIFFERENT, easier, static-target sub-problem). But it still has a real
fat tail: p90 is 77px and the worst case is 708px -- most of the frame
height -- meaning roughly 1 in 6-7 real images still gets a confidently
wrong answer with no internal signal distinguishing it from a correct
one (see "Known limitations" below -- both the width-disambiguation
mechanism now measurably improved and the rarer, still-unfixed
wrong-blob-entirely mechanism contribute to what tail remains).
`TipDetectionResult.ok` is a
sanity/crash signal, not a correctness signal -- do not treat `ok=True`
as "trustworthy" without accounting for this tail.

Per-camera breakdown (same 354-pair run, `evaluate_tip_detection.py`):
cam0 mean 22.17px / median 2.69px / p90 81px / frac<20px 79.0%; cam1 mean
29.13px / median 2.52px / p90 102px / frac<20px 80.2% (also this
camera's single worst-case outlier, 708px -- see "known bright-region
false positive" below, a DIFFERENT and much rarer mechanism than the one
the texture tiebreak fixes); cam2 mean 8.89px / median 1.50px / p90 8px /
frac<20px 95.0%. cam2 remains meaningfully more reliable than cam0/cam1
on this rig/session even after this session's fix. (Previous per-camera
numbers, companion-blob fix only: cam0 mean 24.56px/frac<20px 77.3%; cam1
mean 33.97px/frac<20px 74.1%; cam2 mean 10.88px/frac<20px 93.3%.)

Known limitations, measured/observed not guessed:

- **[FIXED, 2026-08-12 -- was previously mis-diagnosed as the dominant
  mechanism] Width-based tip/fletching disambiguation picks the wrong
  end when the non-tip end's diff signal fades out gradually against a
  low-contrast background.** Direct real-image inspection of cam1's
  worst failures found that the ORIGINAL "bright
  reflection/light-strip hijacks the whole blob" theory (still real, see
  the next bullet, but rarer than assumed) was NOT what most of cam1's
  tail actually was: in the majority of inspected cases, the CORRECT
  dart-shaped component was found (elongation, area, position all
  checked out), but step 5's width comparison picked the wrong end,
  because on cam1 the non-tip end (typically near the flight, closer to
  the dark ceiling/wall backdrop above the board rim) frequently has a
  diff signal that thins out gradually from weak contrast rather than
  genuinely converging -- producing a near-tied, unreliable width
  comparison (measured: the width heuristic's own end-choice accuracy
  drops from ~94% when width_ratio<0.4 to ~57% -- barely better than
  chance -- when width_ratio>=0.8, real 354-pair corpus). Fixed with
  `TEXTURE_TIEBREAK_RATIO_MIN`: when width is this ambiguous and no
  companion blob was found, a local-background-texture check (std-dev of
  a small BACKGROUND-image patch at each end -- board face is texture-
  rich, ceiling/wall backdrop is comparatively uniform) breaks the tie
  instead (measured 70.2% correct in that same high-ambiguity regime,
  n=47, real corpus). This is a genuinely different signal from the
  brightness-based exclusion tried earlier (see next bullet) -- texture
  (local variance), not brightness (local mean), and applied to
  DISAMBIGUATING WHICH END of an already-correctly-selected blob, not to
  EXCLUDING a whole candidate blob. See `TEXTURE_TIEBREAK_RATIO_MIN`'s
  comment in the constants section above for the full sweep. Real,
  measured, whole-corpus improvement: mean 23.05->19.98px, p90
  91.60->76.79px, p99 176.26->170.25px, frac<20px 81.6%->84.7%; cam1
  specifically mean 33.97->29.13px, p90 113.01->102.17px, frac<20px
  74.1%->80.2%. Not a universal win at the individual-case level -- the
  texture signal is right ~66-70% of the time it fires, not 100%, so a
  minority of individual cases get WORSE (one measured example,
  one session's cam1: 65px->159px) in exchange for more cases
  getting fixed elsewhere; reported honestly, not hidden, same as this
  module's other tuned-parameter tradeoffs.
- **[REMAINING FAILURE MODE, real but rarer than previously assumed]
  Largest-elongated-diff-blob is not always the dart.** The elongation
  filter (step 4) helps a lot (raised frac<20px from ~64% to ~76% and
  cut p90 from ~180px to ~118px during tuning) but does not fully solve
  it: bright
  reflections/specular highlights near the top of frame (the overhead
  LED light strip, worst on cam1) occasionally form a diff blob that is
  ALSO elongated enough to pass the filter and larger in area than the
  real dart's diff blob, in which case this detector confidently returns
  a pixel near the light strip instead of near the dart. A brightness-
  based exclusion heuristic was tried (deprioritize components whose
  underlying background pixels are unusually bright) and MEASURED to
  have literally zero effect on this dataset (identical numbers across
  every threshold tried) because the false-positive regions are not
  reliably brighter than legitimate bright regions of the board itself
  (white segments, sector numbers) -- reported honestly as a tried-and-
  failed mitigation, not silently dropped. **Honest correction to this
  module's own earlier framing**: this was previously described as "the
  direct cause of the fat tail" / "the MAIN failure mode" -- real
  per-case inspection this session found that framing overstated how
  common it actually is. Of cam1's real corpus failures (error >20px),
  only ONE case in the full 354-pair corpus (the recorded M8 throw,
  707.8px, this module's single worst-case outlier -- see
  `tests/test_tip_detection.py`'s pinning test) clearly matches this
  "wrong blob entirely, near a real reflection artifact" mechanism on
  direct visual inspection; the rest of the tail was the width-
  disambiguation mechanism above (now measurably improved, not
  eliminated). This mechanism is real and still unfixed (the texture
  tiebreak above operates WITHIN an already-correctly-chosen component
  and cannot help when the WRONG component was chosen in the first
  place), just narrower in scope than this docstring previously implied.
  The real fix this module does NOT implement: a board-region-of-interest
  mask (deliberately out of scope -- this module is decoupled from
  calibration, and a calibrated board ROI is
  exactly the kind of cross-module coupling that decoupling is meant to
  avoid; a future caller that DOES have calibration available should
  crop/mask to the known board region before calling this module, which
  would likely close most of this specific gap).
- **`bg_cam` is not always an empty board.** Confirmed by direct
  pixel-diff inspection: for darts that are not the first of a visit
  (throwIndex 1 or 2, 80/120 real cases in this dataset), `bg_cam{i}.png`
  already contains the PRIOR dart(s) stuck in the board, not an empty
  board -- `clean_cam{i}.png` is the actual empty-board reference. This
  is exactly what this module wants, though: it isolates only the NEWEST
  dart by diffing against the immediately-prior board state, not the
  original empty board, so a diff against `clean_cam` would incorrectly
  include every earlier dart in this visit as part of the "new" blob.
  Confirmed no code change needed here, but any caller must pass the
  right background image (the state immediately before THIS dart, not
  necessarily an empty board) -- documented because it is easy to get
  backwards.
- **No sub-pixel refinement.** The returned tip pixel is the mean of the
  3 most extreme thresholded pixels at the chosen end -- accurate to
  roughly a pixel or two in the common case (median 2.5px total error,
  which already includes whatever error is in the reference label
  itself, not purely this module's own error), but nothing here does a
  true corner/edge sub-pixel fit (e.g. cv2.cornerSubPix-style refinement)
  the way a sub-pixel refinement stretch goal originally sketched.
- **No multi-dart / overlap handling.** Every real image evaluated
  contains exactly one NEW dart (verified via the manifest this was
  tested against, which is one-dart-per-case by construction). Two
  darts landing in the same
  vote window, or a new dart's diff blob touching/overlapping an
  already-embedded dart from earlier in the visphase (not the same as
  the "prior dart in bg_cam" case above, which this module DOES handle;
  this is specifically a new dart whose silhouette visually overlaps an
  old one in the SAME frame), is untested and not specifically handled --
  the largest-elongated-component logic would likely merge the two into
  one blob and misidentify the tip. Real, plausible, unproven risk, not
  covered by any test here.
- **Near-double-ring / near-edge tips untested for a distinct reason.**
  Nothing in this module treats board position specially (fully
  decoupled from board geometry), so there is no reason to
  expect systematically worse accuracy specifically near the double ring
  vs elsewhere -- but this was not separately measured (the manifest
  spans typical real throws; no attempt was made to bucket by
  proximity to the board edge). Flagged as untested, not claimed safe.
- **Bounce-outs are not represented in the validation data.** All 354
  labels come from `liveThrow.cameras[].tipImage` of DARTS THAT SCORED,
  i.e. stuck in the board -- the reference dataset's bounce-out cases
  was not part of this evaluation. A dart that struck and bounced off
  would show a very different, much shorter-lived diff signature (no
  final embedded silhouette to converge PCA on) that this module has
  never been run against.
- **Tuned on one physical rig / one session's lighting**, same caveat as
  opendarts/calibration/landmark_detection.py: `DIFF_THRESHOLD` and the
  other tuned constants below were picked by measuring against this one
  session's real images. A different rig/lighting setup could need
  different constants; nothing here normalizes for exposure/white
  balance.

**2026-08-12 -- `alt_tip_px`: two darts' shafts merging into one blob
(real incident, throw `throw_1786580447119`,
NOT part of the 354-pair evaluation corpus above -- found live, not
during prototyping).** A dart landed very close to a dart already stuck
in the board from the same visit. This module has NO multi-dart/overlap
handling (see "Known limitations" above) -- diffing against `bg_cam`
(the board state immediately before THIS dart) is supposed to isolate
only the newest dart, but when two darts land close enough together in
the SAME diff window, their shafts can merge into one connected
component after dilation. Forward-projecting AD's real ground-truth tip
(`board_xy (51.99, -67.54)mm`) through each camera's real calibration
and comparing to what `detect_tip()` actually returned on the real
frames: cam0 detected 13.3px off (normal noise, unaffected); cam1
detected **102.8px** off (`width_ratio=0.995`, no companion found,
texture tiebreak fired and picked wrong); cam2 detected **158.3px** off
(`width_ratio=0.857`, no companion found). The mechanism: a merged
two-dart blob has no genuine taper at either end (neither end is a real
converging single-dart tip), so the width-based "narrow end = tip"
assumption this module's whole disambiguation chain (step 5/5b) is built
on breaks by construction -- both ends can look similarly narrow, and
the texture tiebreak (a weak, ~70%-correct-at-best signal even in its
intended regime, see TEXTURE_TIEBREAK_RATIO_MIN's comment) cannot
reliably tell "this dart's tip" from "the OTHER dart's tip" when both
sit on the textured board face.

Deliberately NOT fixed with a smarter 2D heuristic here (that path --
companion-blob, then texture tiebreak -- was already tried twice and
each addition needed real measurement and still left a real gap, per
this module's own documented history above). Instead: when this
module's own end-choice is genuinely unresolved on a single image alone
(no companion blob found AND the width comparison never cleared
`WIDTH_AMBIGUITY_RATIO_MIN` -- the same condition that currently falls
through to the texture tiebreak or a bare, near-coin-flip width guess,
see `_locate_tip_in_component`'s `alt_unresolved`), `TipDetectionResult`
now ALSO returns `alt_tip_px`: the tip point from the end that LOST that
tiebreak, mirroring `tip_px`'s own construction. `tip_px` itself is
completely unchanged by this -- re-running the full 354-pair evaluation
corpus after this change reproduced IDENTICAL numbers (mean 19.98px,
median 2.10px, p90 76.79px, p99 170.25px, max 707.82px, frac<20px
84.7%, same per-camera breakdown) confirming zero effect on any existing
single-image detection. `opendarts/pipeline.py`'s `score_dart()` is the
consumer: with two independent monocular guesses instead of one forced
guess, cross-camera geometric ray agreement -- already-proven, tested
machinery -- gets to arbitrate instead of this module guessing alone.
See `score_dart()`'s own docstring and module-level comments for the
combination search and the real before/after numbers this produced,
including the real dart-17/throw_1786580447119 incident recovering a
score within 2.48mm of AD's own ground truth (previously rejected
outright, 32.4mm ray disagreement).

**2026-08-16 -- `prior_dart_line_px`: an OLD dart nudged by a NEW one
poisons the motion-diff, real incident, the recorded T15 throw**
(the prior dart, same visit, is `20260816-171712-064-D10`). A different failure mode from the two above -- not two
NEW darts merging, but a dart already stuck in the board from an
EARLIER throw of the SAME visit getting physically nudged by the impact
of THIS throw's dart. `bg_bgr` for this throw already contains that
prior dart (see "`bg_cam` is not always an empty board" above), so
normally its own pixels cancel out of the diff entirely -- but a nudge
means the prior dart's silhouette differs between `bg_bgr` (captured
before this impact) and `frame_bgr` (captured after), so the diff mask
picks up a real "motion ghost" running along the PRIOR dart's own shaft,
which can merge with the true NEW dart's own diff blob into one
connected component (measured on the real incident: cam2,
`n_area_candidates=1`, `width_ratio=0.48` -- a CONFIDENT, not ambiguous,
width call that still picked the wrong end).

Measured directly (not guessed) what actually distinguishes this case:
neither raw point-distance from the returned `tip_px` to the prior
dart's own last-known tip, nor "does the chosen component's pixel
footprint pass near the prior dart's position", separates real
contamination from ordinary close grouping -- both have heavy overlap
with plainly-correct throws (a new dart landing physically near an old
one is a NORMAL, common darts outcome, not itself suspicious; measured
on the real 797-camera-row corpus of every multi-dart visit in
`data/archive/clean/`: correct throws range from 3.4px to hundreds of
px on point-distance, with no clean separation from the real
contamination case). What DOES separate cleanly, measured on that same
corpus: the PERPENDICULAR distance from BOTH of the current throw's own
two candidate ends (`tip_px` AND `far_end_px`) to the INFINITE LINE
defined by the prior throw's own two ends (recomputed fresh via this
module's own `detect_tip()` on the prior throw's stored images -- see
`opendarts.engines.apollo.prior_dart_context.find_prior_dart_line_px()`,
the caller-side plumbing that locates and recomputes this). On the real
incident (throw 065, cam2): `tip_px` sits 8.2px, `far_end_px` 10.8px
perpendicular from the prior dart D10's own line -- both ends
essentially ON that line -- while the genuinely uncontaminated cameras
on the SAME throw (cam0, cam1) sit 112.9px/112.1px and 128.7px/100.5px
away respectively, comfortably separated. `PRIOR_DART_LINE_MAX_PERP_PX`
(see that constant's own comment) is the real, measured threshold this
produced. Deliberately requires BOTH ends close (not either alone) --
this is the literal signature of a component that is mostly/entirely a
reappearance of the prior dart's own shaft (both its near AND far
extent line up with where it already was), which a genuinely new,
different dart merely landing nearby would not by chance also do.

`prior_dart_line_px` is `None` by default -- every existing caller
(every test, every offline eval script, any caller with no visit
context) is completely unaffected, exactly today's behavior. Only a
caller that explicitly has and passes a same-camera prior-throw line
(currently: `opendarts.engines.apollo.engine.ApolloEngine.score()`,
threaded from `opendarts.live.capture_daemon.handle_ready_to_capture()`
live and `opendarts.capture.replay.replay_throw_with_engine()` offline, per
docs/DESIGN.md's "Replay is the source of truth") even computes this signal at all.

**This module does NOT reject the camera itself, even when the signal
fires** -- a deliberate, measured design change from an earlier version
of this guard. `tip_px`/`alt_tip_px`/`far_end_px`/`ok` are all completely
UNCHANGED by this parameter; the only effect is one new diagnostics key,
`prior_dart_contamination_suspected`. An earlier version of this guard
DID reject directly from here (returning `ok=False` whenever the signal
fired) and was measured, on the same real corpus, to regularly destroy
otherwise-correct multi-camera throws: a later dart legitimately thrown
close to an earlier one -- a normal, common darts outcome, not evidence
of contamination -- trips this same perpendicular-line-proximity signal
just as often as genuine contamination does (measured: 53/779, 6.8%, of
real CORRECT camera-detections on multi-dart visits also fall at or
under `PRIOR_DART_LINE_MAX_PERP_PX`, with no clean separation from the
real incident's own 8.2px/10.8px -- e.g. one real false-positive
camera in the recorded D2 throw measured 8.4px/6.4px,
TIGHTER than the true incident, yet was a perfectly good detection).
Point-in-time per-camera geometry alone cannot tell these apart. Hard-
rejecting from here regressed 4 previously-correct real throws in one
test corpus pull, including throws where the FULL 3-camera set already
agreed to within ~0.3-1.3mm.

The actual accept-or-reject DECISION is made one layer up, in
`opendarts.engines.apollo.engine.ApolloEngine.score()`: this flag
becomes one CANDIDATE combination (drop every flagged camera) for
`opendarts.engines.apollo.scoring.score_dart()`'s own already-proven,
already-tested ray-disagreement metric to arbitrate against the
unmodified full set -- exactly the same "offer a candidate, let real
measured triangulation agreement decide" pattern this codebase's own
`alt_tip_pixels` combination search above already uses, not a new
magnitude threshold gating what score_dart() itself does (the earlier,
already-rejected batch8 lever, see `scoring.py`'s own module docstring).
See that method's own dated comment for the full real-incident write-up,
including a SECOND real trap found during validation (throw
the recorded D2 throw: naively preferring whichever candidate has
LOWER ray disagreement chose a confidently-agreeing but WRONG 2-camera
pair over an already-correct 3-camera answer -- the same correlated-bias
correlated-bias mechanism this project already documents, reached via a
new path) and the resulting extra gate that fixes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from opendarts.imageops import (
    DiffCrop,
    PrecomputeRequirements,
    blurred_abs_diff,
    component_pixels,
    ellipse_kernel,
    morph_on_bbox,
    threshold_mask,
)

# --- Tuned constants ---------------------------------------------------
# All tuned against the real 354-pair dataset described in the module
# docstring above -- not guessed defaults, but the output of a parameter
# sweep.

# Fixed absolute grayscale-diff threshold (0-255 scale), applied AFTER a
# 5x5 Gaussian blur. A data-adaptive threshold (Otsu, then percentile-
# based) was tried first and measurably worse -- see module docstring
# step 2.
#
# **Re-tuned 30.0 -> 25.0, 2026-08-13**, against THIS rig's own 349 real
# archived throws (`data/archive/clean/`, 8 sessions) rather than the
# older 354-pair reference pixel-label corpus the 30.0 was picked from. The
# reference used for the re-tune is the oracle's ground-truth `tip_xy_mm`
# forward-projected through each camera's real calibration -- a
# per-camera pixel reference on the exact images the engine is judged on
# (1047 real camera-detections). What moved is the FAT TAIL, not the
# median:
#
#   pooled per-camera pixel error, 1047 real detections
#     30.0 (was): median 6.92px  p90 77.6px  within-20px 82.8%
#     25.0 (now): median 6.98px  p90 31.1px  within-20px 85.6%
#
# A lower threshold keeps more of the dart's own weakly-contrasted
# silhouette (especially the tip end against the board face), so the
# chosen component tapers where the real dart tapers instead of where the
# threshold happened to cut it -- which is exactly the input step 5's
# width comparison depends on. The median barely moves because the
# already-easy cases were never threshold-limited.
#
# Swept 20/22/24/25/26/28/30/35/45. 22-28 is a broad plateau on the
# pooled pixel metric (84.2-86.0% within-20px, all better than 30.0's
# 82.8%), so this is a plateau, not a spike -- 24.0 is the pixel-metric
# optimum by a hair (86.0%) but costs one throw on the historical
# corpus's best condition, and 25.0 is the middle of the region that both
# improves the pixel metric AND regresses no end-to-end condition. 35.0
# scores higher on ONE condition (166/180 vs 163/180) while being worse
# on every other metric including the tail (p90 104px) -- read as noise
# in a discrete 180-sample metric and deliberately not chosen.
# Per-session, not just pooled (all 8 real sessions scored independently,
# so a change carried by one big session would show): within-20px
# improves in 6 of 8 sessions and drops slightly in 2 (a 23-throw and a
# 10-throw session).
#
# End-to-end effect, all four corpus x calibration conditions:
#   fresh 180 throws, package (live) calibration:   163/180 -> 163/180
#   fresh 180, re-derived oriented_landmarks calib: 148/180 -> 150/180
#   historical 169, package calibration:            139/169 -> 143/169
#   historical 169, re-derived calibration:         163/169 -> 163/169
# i.e. two conditions improve, two are unchanged, none regresses.
# DILATE_KERNEL_PX was re-swept at the new threshold (21/25/27/31/35) and
# 31 is still the optimum, so only this one constant moved.
DIFF_THRESHOLD = 25.0

GAUSSIAN_BLUR_KSIZE = 5
OPEN_KERNEL_PX = 3
# Deliberately large: real dart silhouettes fragment into 2-3 disconnected
# pieces at DIFF_THRESHOLD (thin shaft loses contiguity); this bridges
# real gaps within one dart's silhouette. See module docstring step 3.
# Re-tuned 23->31 (2026-08-12, real throw investigation): a real capture
# showed the flight/shaft gap can be wider than 23px can bridge even
# after the companion-blob fix below; a fresh sweep of the real 354-pair
# corpus with that fix already active found DILATE=31 strictly better
# than 23 on every measured metric (mean 29.3->23.1px, p90 118->92px,
# p99 239->176px, frac<20px 0.76->0.82) with zero new hard failures --
# not a tradeoff, a clean win. DILATE=35+ starts introducing a hard
# failure and gives back some of the p99 gain, so 31 (not higher) is the
# measured optimum, not a round-number guess.
DILATE_KERNEL_PX = 31

# What a caller-supplied `opendarts.imageops.DiffCrop` must satisfy for
# `detect_tip()` to use it in place of its own gray/diff/blur front end
# (2026-09-06 perf pass). The pad covers the dilation's reach (radius 15)
# and the opening's 3px window, so every morphology output pixel inside
# the crop is computed from real data and the true output is zero outside
# it -- see `DiffCrop`'s docstring for the argument.
PRECOMPUTE_REQUIREMENTS = PrecomputeRequirements(
    ksize=GAUSSIAN_BLUR_KSIZE,
    threshold=DIFF_THRESHOLD,
    pad_px=DILATE_KERNEL_PX // 2 + OPEN_KERNEL_PX,
)

# --- Constants re-swept 2026-08-13 and left unchanged -------------------
# When DIFF_THRESHOLD was re-tuned (see its comment above), every other
# tuned constant in this module was re-swept against the same real
# 349-throw corpus at the NEW threshold, so none of them is left sitting
# at a value that was only optimal for the old one. All measurements are
# pooled per-camera pixel error over 1047 real detections plus end-to-end
# sector+ring on four corpus x calibration conditions; the shipped
# configuration scores within-20px 85.6% / 163/180 / 150/180 / 169's
# 143 and 163.
#
#   TOP_K_AREA_CANDIDATES  2, 3, 5, 8      -> byte-identical at every value
#   MIN_ELONGATION_RATIO   1.8-3.5         -> 2.5 at/near optimum (2.2 wins
#                                             one throw out of 698, inside
#                                             noise; 1.8 and 3.5 both worse
#                                             on the pixel metric)
#   GAUSSIAN_BLUR_KSIZE    3, 5, 7, 9      -> 5 best overall (9 buys 2
#                                             throws on one condition and
#                                             loses 4 on another)
#   OPEN_KERNEL_PX         3, 5, 7         -> 3 clearly best (5 drops
#                                             within-20px to 77.6%, 7 to
#                                             57.6%)
#   WIDTH_AMBIGUITY_RATIO_MIN 0.5-0.9      -> 0.6 at/near optimum (0.5 wins
#                                             one throw net, inside noise;
#                                             0.8+ clearly worse)
#   TEXTURE_TIEBREAK_RATIO_MIN 0.75-1.01   -> 0.86 fine; disabling the
#                                             tiebreak entirely (1.01)
#                                             costs 1.4pts of within-20px
#                                             and zero throws, so it still
#                                             earns its place on pixel
#                                             accuracy but is no longer
#                                             load-bearing end-to-end
#   TEXTURE_PATCH_RADIUS_PX 4-32           -> inert, see its own comment
#
# DILATE_KERNEL_PX's own re-sweep is recorded in its comment above.
MIN_COMPONENT_PIXELS = 15
# Rank-ordered by area; the first ranked candidate whose elongation ratio
# (sqrt(major_eigenvalue / minor_eigenvalue) of its PCA covariance) meets
# this clears the "is this dart-shaped" bar. See module docstring step 4.
TOP_K_AREA_CANDIDATES = 3
MIN_ELONGATION_RATIO = 2.5

# Fraction (of the component's along-axis span) used to define "near an
# end" when measuring perpendicular spread at each end -- see module
# docstring step 5.
#
# **Re-tuned 2026-08-14, Apollo miss-investigation against the real
# 360-throw `data/archive/clean/` corpus (3 sessions:
# three recorded sessions of 180, 120 and 60 throws).**
# Neither END_WINDOW_FRACTION nor N_TIP_POINTS_AVERAGED (below) was in
# the "constants re-swept 2026-08-13" list this module's own header
# comment enumerates -- both had been sitting at their original values
# since before that pass. Investigating the corpus's real remaining
# misses found several throws where one camera's `tip_px` sat 20-30px
# from where AD's own ground-truth tip forward-projects even though the
# CORRECT end/component was chosen (not an end-flip, not a wrong blob --
# see tip_detection.py's dated 2026-08-14 investigation note) -- plain
# localization noise in exactly the "mean of N most extreme points"
# construction these two constants control. Swept jointly (not just each
# alone, since they interact -- END_WINDOW_FRACTION sets how many pixels
# feed the width comparison AND, downstream, how large a component's
# "end" is; N_TIP_POINTS_AVERAGED sets how many of THOSE feed the
# reported point) against the full 360-throw corpus, replayed end-to-end
# through score_dart(), per-session broken out so no single session's
# quirk could carry the result:
#
#   N=3  (was) x W=0.15 (was): 345/360 = 95.8%  (169/180, 117/120, 59/60)
#   N=10       x W=0.20      : 348/360 = 96.7%  (171/180, 118/120, 59/60)
#
# A real, broad joint plateau, not a spike: N in {10,15} x W in
# {0.20,0.22,0.25} all land at 348/360 with the IDENTICAL per-session
# split (171/180, 118/120, 59/60) -- four independent (N,W) pairs
# agreeing exactly, not one lucky combination. N=3/W=0.15 is off the
# plateau on the low side; N=100 (swept in isolation) drops to 341/360,
# confirming the window can't just be maximized -- past some point
# averaging starts pulling the "tip" point down the shaft toward its
# true width, biasing away from the real apex, the same mechanism this
# constant has always controlled. `END_WINDOW_MIN_PX` was independently
# swept 3-12px at the new (N=10, W=0.20) operating point and is
# confirmed INERT here too (byte-identical 348/360 at every value,
# matching this module's own prior finding for `TEXTURE_PATCH_RADIUS_PX`
# -- left at its shipped default, nothing to tune). Net effect measured
# via `opendarts.capture.replay`-equivalent direct replay (detect_tip ->
# reject_outside_roi -> score_dart, current code, real stored
# calibration.json per throw -- not the stale result.json some throws in
# one recorded session were originally captured with, see docs/DESIGN.md for
# why those two numbers differ for that session specifically): the three
# newly-fixed throws' triangulated board_xy moved only ~0.2-1.3mm closer
# to AD's own ground truth (this is fundamentally a near-wire
# noise-reduction win, not a gross-error fix), and the one newly-broken
# throw (`throw_1786666300736`) is itself a coin-flip sector-wire case
# (Δ4.78mm -> Δ5.26mm, i.e. barely worse, not a new gross error) -- net
# +3/-1 on real, previously-mismeasured or previously-correct throws,
# not a threshold gaming a discrete count. See docs/DESIGN.md's dated
# 2026-08-14 Apollo entry for the full sweep table and per-throw
# before/after.
END_WINDOW_FRACTION = 0.20
END_WINDOW_MIN_PX = 6.0
N_TIP_POINTS_AVERAGED = 10

# Companion-blob tip disambiguation -- see module docstring "Known
# limitations" for the failure mode this addresses (found on a real
# throw, 2026-08-12, on cam0). When the dart's flight forms its OWN
# diff blob, separated from the thin shaft by a gap wider than
# DILATE_KERNEL_PX can bridge (measured on that real image: 38px), the
# flight blob is compact (elongation ~1.3, correctly rejected by
# MIN_ELONGATION_RATIO) so only the shaft survives as the chosen
# candidate -- but a shaft with the flight physically amputated no
# longer has the wide/narrow end asymmetry step 5 relies on (measured:
# 2.95px vs 3.27px there, a ~10% difference well within this
# measurement's own noise), so the width comparison picks the wrong end
# roughly at chance. These thresholds bound how far past a shaft's own
# end (COMPANION_MAX_GAP_PX, along its principal axis) and how far off
# that axis (COMPANION_MAX_PERP_PX) a rejected candidate's centroid may
# sit to be treated as "almost certainly the separated flight" rather
# than an unrelated blob elsewhere in frame -- not tuned precisely, just
# measured to be conservative: on the real 354-pair corpus this fires on
# only a handful of cases and never regresses frac<20px.
COMPANION_MAX_GAP_PX = 50.0
COMPANION_MAX_PERP_PX = 20.0
# Gate on the width-based decision's own confidence -- see the comment
# in _locate_tip_in_component where this is used. min(width_a,width_b)/
# max(width_a,width_b); 1.0 = identical widths (fully ambiguous), 0.0 =
# one end has zero measured spread. Value chosen by sweeping the real
# 354-pair corpus.
#
# **Considered and REJECTED, 2026-08-14, raising 0.6 -> 0.8** (same
# Apollo miss-investigation as END_WINDOW_FRACTION/
# N_TIP_POINTS_AVERAGED above). On the real 360-throw
# `data/archive/clean/` corpus alone this measured a clean +1/-0
# (`throw_1786730562953`'s cam2 has width_ratio=0.764 --
# a CORRECT width call that the old 0.6 gate wrongly flagged ambiguous,
# offering `score_dart()`'s combination search a worse alt candidate it
# then picked for having lower 3-ray disagreement, not for being right).
# **But this directly conflicts with an existing, already-pinned real
# recovery on the separate reference pixel-label corpus**
# (`tests/test_engine_apollo_board_roi.py::
# test_reject_outside_roi_recovers_a_known_bad_real_case_via_its_alt_candidate`,
# one of its cam1 cases, width_ratio=0.665): that case NEEDS to
# stay flagged ambiguous for its own primary-off-board/alt-on-board ROI
# rescue to fire at all (378.1px error unrescued vs 3.9px rescued). The
# two real cases need the gate on opposite sides of the same narrow
# window (0.665 must stay >= gate, 0.764 must become < gate) -- no
# single global threshold satisfies both, a genuine structural
# conflict, not a tuning miss. Kept at 0.6. Recorded here so a future pass doesn't re-discover the
# same one-corpus-only +1 without re-finding the conflict.
WIDTH_AMBIGUITY_RATIO_MIN = 0.6

# --- Local-background-texture tiebreak (2026-08-12) -------------------
# Found by directly inspecting real cam1 failures, not guessed:
# cam1's dominant real tail-error mechanism is NOT actually "a
# bright reflection blob gets selected as the dart" (the original
# hypothesis, which measurably still happens but turned out to
# be rare -- one case in the whole 354-pair corpus, an M8 throw,
# matches it exactly). The dominant mechanism is different: the
# CORRECT dart-shaped component is found (elongation, area, position all
# check out), but step 5's width comparison picks the WRONG end, because
# on cam1 the non-tip end's diff signal frequently fades out gradually
# against the dark ceiling/wall backdrop behind the board rim (weak,
# thinning contrast, not a real narrowing of the physical dart) while the
# true tip end sits against the bright/textured board face and keeps a
# crisp, non-tapering diff edge right up to the true tip -- producing a
# near-tied, unreliable width_ratio (measured: the width heuristic's own
# accuracy drops from ~94% when width_ratio<0.4 to ~57% -- barely better
# than chance -- when width_ratio>=0.8, real 354-pair corpus).
#
# A brightness-based exclusion was already tried (see module docstring)
# and measured to have zero effect -- this is a genuinely different
# signal: local background TEXTURE (std-dev of grayscale intensity in a
# small patch of the BACKGROUND image, not the diff/frame), which
# distinguishes "board face" (numbers, colored segments, wire boundaries
# -- high local variance) from "ceiling/wall backdrop" (fairly uniform
# gray, low local variance) independent of raw brightness. Measured
# directly against the real corpus (only on the SAME width_ratio>=0.8
# regime where width itself is unreliable): texture-based end choice is
# right 70.2% of the time there (n=47) vs width's 57.4% -- a real,
# specific improvement in exactly the regime where the existing signal
# is weakest, not a global replacement (texture alone is WORSE than
# width overall, 75.7% vs 85.9%, so it must stay gated to only the
# already-ambiguous regime, same "gate on the confidence of the primary
# signal" pattern as WIDTH_AMBIGUITY_RATIO_MIN below).
#
# Threshold swept 0.6-0.95 against the real 354-pair corpus with the
# tiebreak wired in end-to-end (measuring final pixel error, not just
# end-choice accuracy): a real, if modest, plateau from 0.8-0.9 (overall
# mean 20.0-20.6px vs 23.1px baseline, cam1 frac<20px 0.78-0.80 vs 0.741
# baseline); 0.86 is the measured optimum within that plateau (overall
# mean 19.98px, p90 76.79px, frac<20px 0.847; cam1 mean 33.97->29.13px,
# p90 113.01->102.17px, frac<20px 0.741->0.802). Deliberately set
# ABOVE WIDTH_AMBIGUITY_RATIO_MIN (0.6): the companion-blob override
# above already handles the more common ambiguous-width case where a
# real found companion object gives DIRECT evidence; texture only
# applies when no companion was found AND the width signal is even more
# degenerate than the companion gate's own threshold -- a weaker, more
# generic fallback signal that should only fire when nothing more
# specific is available.
TEXTURE_TIEBREAK_RATIO_MIN = 0.86
# Half-width (px) of the square background patch sampled at each
# candidate end point. Chosen to be large enough to capture real board
# texture (segment/number-width scale) but small enough to stay local to
# the actual end point, comparable to END_WINDOW_MIN_PX.
#
# **Previously flagged here as never independently swept** (a bug in the
# original sweep script silently reused one radius for every value
# tried). **Swept for real 2026-08-13** against this rig's 349 archived
# throws -- 4, 6, 8, 12, 16, 24 and 32 px, measuring both the pooled
# per-camera pixel error (1047 real detections) and end-to-end sector+ring
# accuracy on all four corpus x calibration conditions. The result is
# that this constant is **completely inert on real data**: every value
# tried produced BYTE-IDENTICAL numbers (pixel within-20px 85.6%, median
# 6.98px, p90 31.1px; 163/180, 150/180, 143/169, 163/169). Not "roughly
# similar" -- identical, meaning the texture tiebreak's end-choice never
# flipped for any real image at any patch size in an 8x range. The board
# face vs. backdrop texture contrast is evidently far larger than the
# measurement's sensitivity to window size. 12 is kept purely because it
# is what shipped; nothing here justifies preferring it to any other
# value in that range, and a future change to it should expect no effect
# rather than a tuning opportunity. Honest unknown closed with a real
# measurement rather than left standing as a caveat.
TEXTURE_PATCH_RADIUS_PX = 12

# --- Prior-dart-in-visit contamination guard (2026-08-16) --------------
# See module docstring's dated "prior_dart_line_px" section above for the
# full real-incident write-up and the measurement this threshold comes
# from. Max perpendicular pixel distance, from BOTH of the current
# throw's own candidate ends (tip_px and far_end_px) to the infinite
# line through the PRIOR throw's own two ends (same camera), for this
# camera to be flagged `prior_dart_contamination_suspected` -- a
# DIAGNOSTIC-ONLY flag, see the module docstring's dated
# "prior_dart_line_px" entry for why this module itself never rejects
# on this signal (a real, measured, corpus-wide false-positive rate too
# high to trust as a hard per-camera filter) -- the actual accept/reject
# decision, informed by this flag plus real triangulated ray agreement,
# is made in opendarts.engines.apollo.engine.ApolloEngine.score().
#
# Measured against the real 797-camera-row corpus of every multi-dart
# visit in data/archive/clean/ (max(perp_tip, perp_far), split by
# whether that throw's real, unmodified Apollo score matched AD ground
# truth): the real incident this guard targets (the recorded T15 throw,
# cam2) measures 10.8px; the two genuinely uncontaminated
# cameras on that SAME throw measure 112.9px and 128.7px -- a wide,
# comfortable gap at that one throw. Corpus-wide, "correct" throws have
# NO clean floor (min 1.5px -- two darts can legitimately land such that
# a later one's ends project near an earlier one's own line by
# coincidence of camera perspective, not contamination -- one measured
# false positive, the recorded D2 throw cam0, sits at 8.4px/6.4px,
# TIGHTER than the real incident itself), so this is not a knife-edge
# separation the way MAX_RAY_DISAGREEMENT_MM's healthy/bad gap is --
# 15.0px is a deliberately conservative choice inside the real incident's
# own margin (10.8px) while flagging only 53/779 (6.8%) of real correct
# camera-rows AS SUSPECTED (not rejected -- see above). See
# ApolloEngine.score()'s own dated comment for the real end-to-end
# corpus before/after this produces once combined with that method's own
# elevated-disagreement gate.
PRIOR_DART_LINE_MAX_PERP_PX = 15.0

# --- Off-axis tip-cluster alternate (2026-08-17) -----------------------
# Real incident: the recorded T15 throw, cam2 (Apollo's single
# miss in that 120-throw session -- AD/operator truth S15, scored T15).
# The diff mask's shaft column ended exactly at the true tip
# (~774, 261), but a 12px-area satellite diff fragment at (801, 264) --
# off the dart entirely, merged into the chosen component by the 31px
# dilation -- supplied 8 of the N_TIP_POINTS_AVERAGED most-extreme
# points along the principal axis. The reported tip (795.5, 264.2) was
# pulled ~21px along the board, moving the triangulated radius from
# 95.7mm (single_inner, 0.1mm from AD's own tip radius) to 103.4mm
# (treble). The measurable geometric contradiction: that cluster's mean
# PERPENDICULAR offset from the component's own principal axis was
# -22.6px -- a real tip lies ON the dart's axis by construction (the
# axis is fit to the whole component), so a tip cluster this far
# off-axis cannot be the dart's tip.
#
# Measured on the full real corpus (827 packages x 3 cameras, 2474
# accepted detections): the CHOSEN tip end's
# cluster |mean perp| is p50 1.1px / p90 4.9px / p95 7.5px / p99 21.1px
# / max 57.4px. 12.0 sits 1.6x above the healthy p95 and roughly half
# the real incident's own 22.6px, firing on 2.59% of real detections.
# This is deliberately NOT a knife-edge rejection threshold: when the
# signature fires, this module does NOT overwrite tip_px -- it exposes
# the on-axis recompute (the N most proj-extreme points among those
# within this same perp distance of the axis) as `alt_tip_px`, and
# score_dart()'s existing cross-camera combination search arbitrates,
# exactly the established alt-candidate pattern (2026-08-12). A false
# fire therefore costs nothing unless another camera's ray decisively
# agrees better with the alternative AND every existing acceptance
# threshold passes; on the real incident the primary combination
# disagreed 5.08mm while the on-axis alternative agreed with cam0 to
# 0.022mm -- not a close call. The incident's recomputed on-axis tip is
# insensitive to this constant across 8-15px (identical result).
TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX = 12.0

# --- Axial-gap tip-island alternate (2026-08-17) -----------------------
# Real incident: the recorded S13 throw (AD/operator truth S13
# single_outer; Apollo produced NO SCORE -- "rays disagree by 16.1mm").
# On cam2 the ORIGINAL (non-dilated) diff mask ended at the true tip
# (~678, 208 -- an independent detector found (678, 208) on the same frame),
# but a faint, thin streak of the dart's own shadow cast on the board
# BEYOND the physical tip survived DIFF_THRESHOLD as a small detached
# fragment: 24 points, perpendicular std ~1.4px, mean blurred diff 31.6
# (threshold 25.0), separated from the dart body by a 25.7px stretch of
# axis with ZERO original-mask points -- connected to the component only
# by the 31px dilation. Those 24 points supplied every one of the
# N_TIP_POINTS_AVERAGED most proj-extreme points, so the reported tip
# (672.9, 241.1) overshot the real tip by ~34px ALONG the axis --
# ~50mm on the board plane for this oblique camera -- and no 2-of-3
# camera pair could agree. This is the same satellite-fragment-poisoning
# mechanism as TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX's incident (061-T15),
# but ON-axis (cluster mean perp 5.7px, under that gate) -- a shadow
# falls along the dart's own projected direction by construction, so
# the off-axis signature is structurally blind to it.
#
# The signature is deliberately THREE-part -- an axial gap ALONE is not
# rare and must never trigger a trim. Measured on the full real corpus
# (885 packages x 3 cameras, 2654 accepted detections): the chosen tip
# end's largest
# within-window axial gap is p50 4.7px / p90 16.9px / p95 20.5px /
# p99 26.7px / max 30.7px (bounded by what DILATE_KERNEL_PX=31 can
# bridge) -- 20.2% of real, healthy detections have a gap > 12px,
# because the thin shaft/tip section genuinely does threshold-fragment
# off the body (the exact cases the 31px dilation exists to rejoin;
# see module docstring step 3). What separates a REAL detached tip
# section from a shadow fragment, measured on those same 536
# over-12px-gap detections, is the island itself: real detached tip
# sections are LARGE (island n_points p25 70 / p50 182 / p75 368) and
# STRONG in the pre-threshold blurred diff (island mean p25 47.1 /
# p50 63.6); the incident's shadow island is 24 points at mean diff
# 31.6 -- barely above DIFF_THRESHOLD=25. So the alternate fires only
# when ALL THREE hold: axial gap > 12px (between healthy CONNECTED-run
# gaps and the incident's 25.7px), island n_points <=
# TIP_ISLAND_MAX_N_POINTS = 30 (between the incident's 24 and the
# real-detached-tip p25 of 70), and island mean blurred diff <=
# TIP_ISLAND_MAX_MEAN_DIFF = 35.0 (DIFF_THRESHOLD + 10 -- between the
# incident's 31.6 and the real-detached-tip p25 of 47.1). Combined
# fire rate on the real corpus: 45/2654 detections (1.70%),
# comparable to the off-axis signature's own 2.59%.
#
# NOT a rejection and never an overwrite (the diff mask genuinely can
# fragment at the thin tip itself -- amputating on this signal alone
# would break exactly the cases DILATE_KERNEL_PX exists for): when the
# full signature fires and no other mechanism already claimed the alt
# slot, the trimmed recompute (the N most proj-extreme points BEHIND
# the first over-gate gap) is exposed as `alt_tip_px`, and
# score_dart()'s existing cross-camera combination search arbitrates --
# the exact contract of the 2026-08-12 width-ambiguity alt and the
# off-axis alt above. A false fire keeps its correct primary unless
# every other camera's ray decisively prefers the trimmed candidate
# AND all existing acceptance thresholds pass. On the real incident:
# primary full-set disagreement 16.1mm (rejected, NO SCORE); with
# cam2's trimmed alternate (679.3, 207.1) -- within 1.5px of the
# independent detection (678, 208) on the same frame -- the full
# 3-camera set agrees at 6.5mm -> S13 single_outer, matching
# AD/operator truth.
TIP_ISLAND_MIN_AXIAL_GAP_PX = 12.0
TIP_ISLAND_MAX_N_POINTS = 30
TIP_ISLAND_MAX_MEAN_DIFF = 35.0
# How far back from the tip-end extreme (along the principal axis, px)
# a gap is still considered "at the tip end". A shadow/glint fragment
# bridged by ONE dilation span sits within ~DILATE_KERNEL_PX of the
# body; 80px allows a chain of two bridges plus the fragment's own
# extent while staying far from the fletching end (real component spans
# measure 240-300px+). The incident's island (10px extent + 25.7px gap)
# sits entirely within 36px of the extreme.
TIP_ISLAND_SEARCH_WINDOW_PX = 80.0


def _perp_dist_to_line(
    point_px: tuple[float, float],
    line_a_px: tuple[float, float],
    line_b_px: tuple[float, float],
) -> float:
    """Perpendicular distance from `point_px` to the INFINITE line
    through `line_a_px`/`line_b_px` -- not distance to either endpoint,
    and not clamped to the segment between them (see
    PRIOR_DART_LINE_MAX_PERP_PX's comment for why the infinite line,
    not the segment, is the right test here: a merged contaminated
    component's own span is typically LARGER than the prior dart's own
    shaft, extending past both of its ends)."""
    a = np.array(line_a_px, dtype=np.float64)
    b = np.array(line_b_px, dtype=np.float64)
    p = np.array(point_px, dtype=np.float64)
    d = b - a
    norm = float(np.linalg.norm(d))
    if norm < 1e-6:
        # Degenerate (near-zero-length) prior line -- fall back to plain
        # point distance from the shared endpoint rather than dividing
        # by ~zero.
        return float(np.linalg.norm(p - a))
    d_unit = d / norm
    v = p - a
    proj_len = float(v @ d_unit)
    perp = v - proj_len * d_unit
    return float(np.linalg.norm(perp))


@dataclass
class TipDetectionResult:
    """Result of one detect_tip() call.

    `ok`: a sanity/crash signal (did the pipeline find ANY plausible
    dart-shaped candidate), NOT a correctness signal -- see the module
    docstring's "Honest headline" section. A confidently-wrong result on
    the fat-tail failure mode still reports `ok=True`.
    """

    ok: bool
    tip_px: tuple[float, float] | None
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)
    # Added 2026-08-12 -- real incident, throw
    # throw_1786580447119 (see module docstring's dated entry below for
    # the full write-up). `tip_px` keeps meaning exactly what it always
    # has: the single best-guess tip pixel, unchanged for every existing
    # caller. `alt_tip_px` is populated ONLY when the end-choice this
    # module just made was genuinely a coin flip on THIS image alone --
    # no companion blob found (the one case that DOES give direct
    # evidence, see _find_companion_end) AND the width comparison itself
    # never cleared WIDTH_AMBIGUITY_RATIO_MIN (whether or not the texture
    # tiebreak fired within that ambiguous regime) -- see
    # _locate_tip_in_component's `alt_unresolved` for the exact condition.
    # `None` in every other case: unambiguous width call, OR a companion
    # blob resolved it (a different, already-validated code path, left
    # untouched). Downstream (opendarts.engines.apollo.scoring.score_dart), a non-None
    # alt_tip_px is a second candidate for cross-camera geometric
    # agreement to arbitrate, instead of this module forcing one guess
    # monocularly in a situation it cannot actually resolve alone.
    #
    # 2026-08-17 -- a SECOND population case, same downstream contract:
    # when the chosen tip cluster's mean perpendicular offset from the
    # component's own principal axis exceeds
    # TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX (see that constant's comment --
    # real incident, the recorded T15 throw cam2, an off-axis satellite
    # diff fragment pulled the tip ~21px off the dart), the on-axis
    # recompute is exposed here (diagnostics `tip_off_axis_alt`=True).
    # The width-ambiguity case above takes precedence when both fire.
    alt_tip_px: tuple[float, float] | None = None
    # Added 2026-08-14. The OTHER end of the same chosen component,
    # ALWAYS populated (whenever a tip was found at all) -- deliberately
    # unlike `alt_tip_px`, which appears only when this module's own
    # width comparison judged the end-choice a coin flip. Constructed
    # identically to `tip_px` (mean of the N_TIP_POINTS_AVERAGED most
    # extreme points), just at the losing end.
    #
    # Why a separate field rather than widening `alt_tip_px`: they mean
    # different things and have different downstream contracts.
    # `alt_tip_px` is this module SAYING it could not decide -- an
    # invitation for opendarts.engines.apollo.scoring.score_dart() to
    # arbitrate between two equally-credible candidates by cross-camera
    # agreement. `far_end_px` carries no such claim: the end-choice may
    # have been perfectly confident. It exists so a caller holding
    # evidence this module structurally cannot have (a calibrated board
    # ROI -- see opendarts.engines.apollo.board_roi) can act on a
    # CONFIDENTLY WRONG end call, which is a real, measured failure mode
    # on this rig: a dart landing high on the board with its flight
    # angled up out of the board face gets a confident width verdict for
    # the flight end. Handing this to score_dart() as an ordinary
    # alternate instead was measured on the real 300-throw corpus and is
    # a clean +0/-0 no-op at every width-ratio gate swept (0.0-0.8) --
    # geometric ray agreement alone does not catch these, the board ROI
    # does. `None` only when no tip was found or the far end came out
    # non-finite.
    far_end_px: tuple[float, float] | None = None


def _blurred_diff(bg_gray: np.ndarray, frame_gray: np.ndarray) -> np.ndarray:
    """The Gaussian-blurred absolute grayscale diff `_diff_mask` has
    always thresholded (split out 2026-08-17 so `detect_tip` can keep
    the pre-threshold magnitudes for the tip-island faintness
    diagnostics -- see TIP_ISLAND_MIN_AXIAL_GAP_PX's comment -- without
    computing the diff twice or changing `_diff_mask`'s public
    behavior, which opendarts.engines.talos.shaft_line also imports).

    2026-09-05 perf pass: the arithmetic lives in `opendarts.imageops`
    (uint8 absdiff -> one float32 cast, bit-identical to the former two
    float32 casts -- see `imageops.abs_diff_f32`'s own docstring)."""
    return blurred_abs_diff(bg_gray, frame_gray, GAUSSIAN_BLUR_KSIZE)


def _diff_mask(bg_gray: np.ndarray, frame_gray: np.ndarray) -> np.ndarray:
    return threshold_mask(_blurred_diff(bg_gray, frame_gray), DIFF_THRESHOLD)


def _component_stats_in_bbox(
    comp_labels: np.ndarray,
    comp_id: int,
    bbox: tuple[int, int, int, int],
    opened: np.ndarray | None,
    mask_origin: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """PCA of one label's pixels (returns (points_xy, centered_points,
    elongation_ratio), or None if too few points), evaluated
    only inside the label's own bounding box (2026-09-05 perf pass). The
    full-frame `==`/`&`/`np.nonzero` triple this replaces was ~1.6ms per
    call on 1280x720 and ran once per candidate PLUS once per OTHER
    component in `_find_companion_end()` -- a third of this engine's
    whole per-throw cost. Same pixels in the same order, so the PCA
    input (and everything downstream) is bit-identical -- see
    `opendarts.imageops.component_pixels`. `mask_origin` is the full-frame
    position of `opened`'s top-left pixel when `opened` is a crop."""
    xs, ys = component_pixels(comp_labels, comp_id, bbox, opened, mask_origin)
    return _component_stats_from_pixels(xs, ys)


def _component_stats_from_pixels(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float] | None:
    if len(xs) < MIN_COMPONENT_PIXELS:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)
    minor_eval, major_eval = evals[order[0]], evals[order[1]]
    elongation = float(np.sqrt(major_eval / max(minor_eval, 1e-6)))
    principal = evecs[:, order[1]]
    return pts, centered, principal, elongation


# 2026-09-02 -- see _largest_elongated_subcomponent()'s own docstring for
# the full derivation. A first version of that function gated the
# sub-piece purely on MIN_COMPONENT_PIXELS (15px, the generic "enough to
# PCA at all" floor) -- measured on this project's own local corpus
# (the session corpus, 951 throws) to be a real mistake,
# not a safe default: it let 3 separate real throws pick a spurious
# 266-670px NOISE fragment (elongated by chance, but clearly not a dart)
# over the pre-existing, cruder "take the whole blob" fallback, which had
# been getting those 3 throws right. The genuine positive case this
# function exists for (the recorded outside throw cam0) has a real
# sub-piece of 9684px -- a clean, order-of-magnitude gap above every
# measured false positive. Set well inside that gap (roughly 4x the
# largest measured false positive, roughly 3x below the one measured true
# positive) -- not a tight fit to either number, since only a handful of
# real examples of each class have been measured so far; re-measure if a
# future corpus pull produces a real case landing between 670 and 9684.
MIN_SALVAGED_SUBCOMPONENT_AREA_PX = 3000


def _largest_elongated_subcomponent(
    mask_bool: np.ndarray,
    crop_origin_and_shape: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float] | None:
    """2026-09-02 -- real incident, the recorded outside throw cam0 (see
    module docstring's dated entry for the full write-up). The 31px
    dilation `detect_tip()` applies before connected-component labeling
    can bridge the TRUE dart's own diff pixels into the SAME label as an
    unrelated, spatially SEPARATE diff artifact -- confirmed on this
    incident's own real frame: a real, isolated 9684px elongated dart
    shaft sub-region sat inside a 46004px dilated label whose OVERALL PCA
    elongation (1.37, well under `MIN_ELONGATION_RATIO`'s 2.5) was
    dragged down by ~77 tiny (5-194px each), spatially scattered specks
    along an unrelated board-text/number-ring band, none individually
    close in size to the real shaft. The label-level elongation gate in
    `detect_tip()` correctly rejected the WHOLE merged blob, but the
    existing code then had no way to recover the genuinely elongated
    dart shaft still sitting inside it -- it fell straight through to a
    smaller, entirely unrelated, coincidentally-elongated component
    instead (this incident's own root cause: a wrong-object lock, not a
    wrong-END-of-the-right-object mislabeling).

    Re-segments `mask_bool` (the SAME un-dilated `opened_bool & region`
    array this already receives for a dilated label) via a
    plain connected-components pass at NATIVE (non-dilated) resolution,
    then returns the LARGEST-by-area sub-component that independently
    clears `MIN_ELONGATION_RATIO` (genuinely dart-shaped on its own, not
    just "the biggest leftover speck") AND
    `MIN_SALVAGED_SUBCOMPONENT_AREA_PX` -- `None` if nothing qualifies.
    Always tries LARGEST-first: on the real incident above, this reliably
    picks the 9684px real shaft over any of the ~77 tiny specks.

    **The area floor is real, measured, and load-bearing, not the generic
    `MIN_COMPONENT_PIXELS` floor `_component_stats_from_pixels` itself
    uses** -- see
    `MIN_SALVAGED_SUBCOMPONENT_AREA_PX`'s own comment for the real local
    corpus evidence this was tightened from an earlier, too-permissive
    draft (a small, elongated-by-chance noise fragment can and does clear
    `MIN_ELONGATION_RATIO` on real data -- confirmed as a real, measured
    regression on this project's own local corpus, not a theoretical
    risk).

    Called from `detect_tip()` as a fallback tier BETWEEN the main
    top-K/elongation loop and the final "accept the largest candidate
    regardless of elongation" fallback -- deliberately not folded into
    that final fallback's own unconditional accept, since a genuinely
    dart-shaped, dart-SIZED sub-region (this function's own return) is
    real, positive evidence a bare "biggest blob, elongated or not"
    accept is not. Purely additive: when this returns None (no
    candidate's footprint contains a qualifying elongated, sufficiently
    large sub-piece -- expected to be the common case, since most real
    detections either already clear the whole-blob elongation gate
    directly, or have no real dart-sized sub-structure to recover at
    all), `detect_tip()`'s existing fallback behavior is completely
    unchanged.

    `crop_origin_and_shape` (2026-09-05 perf pass): `detect_tip()` now
    hands in the footprint cropped to the label's bounding box, as
    `((x, y), (img_h, img_w))`. The full-frame mask is rebuilt here so
    the connected-components pass below sees EXACTLY the input it always
    did (label numbering, which the area-sorted tie order depends on,
    therefore unchanged). This path fires on ~6% of detections, so the
    rebuild is cheap in aggregate; `None` keeps the original full-frame
    calling convention."""
    import cv2

    if crop_origin_and_shape is not None:
        (ox, oy), full_shape = crop_origin_and_shape
        sub_mask = np.zeros(full_shape, dtype=np.uint8)
        sub_mask[oy:oy + mask_bool.shape[0], ox:ox + mask_bool.shape[1]] = (
            mask_bool.astype(np.uint8) * 255
        )
    else:
        sub_mask = mask_bool.astype(np.uint8) * 255
    n, sub_labels, sub_stats, _ = cv2.connectedComponentsWithStats(
        sub_mask, connectivity=8
    )
    if n <= 1:
        return None
    order = sorted(
        range(1, n), key=lambda i: int(sub_stats[i, cv2.CC_STAT_AREA]), reverse=True
    )
    for i in order:
        area = int(sub_stats[i, cv2.CC_STAT_AREA])
        if area < MIN_SALVAGED_SUBCOMPONENT_AREA_PX:
            # Areas are sorted descending -- every remaining candidate is
            # smaller still, so nothing further can qualify either.
            break
        result = _component_stats_in_bbox(sub_labels, i, sub_stats[i, :4], None)
        if result is None:
            continue
        pts, centered, principal, elongation = result
        if elongation >= MIN_ELONGATION_RATIO:
            return pts, centered, principal, area, elongation
    return None


def _local_bg_texture(bg_gray: np.ndarray, cx: float, cy: float,
                       r: float = TEXTURE_PATCH_RADIUS_PX) -> float:
    """Std-dev of grayscale intensity in a small square patch of the
    BACKGROUND image centered at (cx, cy) -- see TEXTURE_TIEBREAK_RATIO_MIN's
    comment above for why this (not brightness) discriminates "board face"
    from "ceiling/wall backdrop" on cam1. Returns 0.0 for a patch that
    falls entirely outside the image."""
    h, w = bg_gray.shape[:2]
    x0, x1 = max(0, int(cx - r)), min(w, int(cx + r))
    y0, y1 = max(0, int(cy - r)), min(h, int(cy + r))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(bg_gray[y0:y1, x0:x1].std())


def _find_companion_end(
    comp_labels: np.ndarray,
    opened: np.ndarray,
    other_candidates: list[tuple[int, int]],
    centroid: np.ndarray,
    principal: np.ndarray,
    pmin: float,
    pmax: float,
    bboxes: dict[int, tuple[int, int, int, int]],
    mask_origin: tuple[int, int] = (0, 0),
) -> str | None:
    """Look for a compact companion blob (e.g. the dart's flight,
    separated from the shaft by too wide a diff-mask gap -- see
    COMPANION_MAX_GAP_PX's comment) near an extension of the chosen
    component's own principal axis, just past one of its two ends.

    Returns 'a' or 'b' -- the end WITH a nearby companion, i.e. the
    end that is NOT the tip -- or None if no qualifying companion is
    found. `other_candidates` should be every OTHER diff component in
    frame (not just the top-K by area), since the flight may not be
    among the largest few.

    Uses each candidate's individual PIXELS, not its centroid: a real
    flight blob has real width (measured on the real case this was
    built from: perpendicular spread up to +-35px around its own
    centroid), so a centroid-based gap/offset overstates both the true
    edge-to-edge gap and understates how well-aligned the blob's nearest
    edge actually is with the shaft's axis. What matters is whether the
    companion candidate has ANY pixel close to, and roughly on-axis
    with, the shaft's own extent -- not where its centroid sits.

    `opened` is the uint8 opened mask and `bboxes` maps every candidate
    label to its `(x, y, w, h)` from connectedComponentsWithStats
    (2026-09-05 perf pass) -- each candidate's pixels are read from its
    own bounding box, not the full frame. With up to ~23 other
    candidates on a noisy frame this loop was the source of this
    engine's worst-case latency tail (73ms).
    """
    perp_axis = np.array([-principal[1], principal[0]])
    best_end = None
    best_gap = None
    for _, comp_id in other_candidates:
        result = _component_stats_in_bbox(
            comp_labels, comp_id, bboxes[comp_id], opened, mask_origin
        )
        if result is None:
            continue
        other_pts, _, _, _ = result
        vec = other_pts - centroid
        with np.errstate(all="ignore"):
            t = vec @ principal
            p = vec @ perp_axis
        # Consider only points that project PAST one of the shaft's own
        # ends; among those, the one closest to the shaft (smallest gap)
        # determines whether/where this candidate qualifies as a
        # companion.
        past_a = t < pmin
        past_b = t > pmax
        for mask_, end, gap_arr in (
            (past_a, "a", pmin - t),
            (past_b, "b", t - pmax),
        ):
            if not np.any(mask_):
                continue
            idx = np.argmin(gap_arr[mask_])
            gap = float(gap_arr[mask_][idx])
            perp = float(np.abs(p[mask_][idx]))
            if gap > COMPANION_MAX_GAP_PX or perp > COMPANION_MAX_PERP_PX:
                continue
            if best_gap is None or gap < best_gap:
                best_gap, best_end = gap, end
    return best_end


def _locate_tip_in_component(
    pts: np.ndarray,
    centered: np.ndarray,
    principal: np.ndarray,
    forced_non_tip_end: str | None = None,
    bg_gray: np.ndarray | None = None,
    diff_blur: np.ndarray | None = None,
    diff_origin: tuple[int, int] = (0, 0),
) -> tuple[tuple[float, float], dict]:
    """Given a dart-shaped component's points, find the tip end: the
    extreme (along the principal axis) with SMALLER perpendicular spread
    -- see module docstring step 5.

    `forced_non_tip_end`: 'a', 'b', or None. When a companion blob was
    found near one end (see _find_companion_end), that end is treated
    as the non-tip end regardless of the width comparison -- the width
    comparison is exactly the signal that's unreliable in this
    situation (see COMPANION_MAX_GAP_PX's comment above).

    `bg_gray`: optional background grayscale image, used ONLY when no
    companion was found and the width comparison is itself very
    ambiguous (see TEXTURE_TIEBREAK_RATIO_MIN's comment above) -- a
    weaker, more generic fallback disambiguator than the companion-blob
    check. `None` disables the tiebreak entirely (falls back to the
    plain width comparison), which is also what happens automatically
    when width_ratio doesn't clear the gate.

    `diff_blur`: optional pre-threshold blurred diff image (the exact
    array `_diff_mask` thresholds -- see `_blurred_diff`), used ONLY for
    the tip-island faintness DIAGNOSTICS (see
    TIP_ISLAND_MIN_AXIAL_GAP_PX's comment) -- never affects tip_px or
    the alt decision itself. `None` simply leaves those diagnostic
    fields None. `diff_origin` is the full-frame position of
    `diff_blur`'s top-left pixel when `diff_blur` is a crop (all `pts`
    lie inside it -- they are mask pixels)."""
    perp_axis = np.array([-principal[1], principal[0]])
    # np.errstate: this small matmul has been observed to raise spurious
    # "divide by zero"/"invalid value"/"overflow" FP warnings on some
    # numpy+Accelerate (macOS ARM BLAS) builds even though the actual
    # output is finite -- same environment quirk documented and verified
    # in opendarts/calibration/landmark_detection.py's Ellipse.boundary_samples
    # there, not silencing a real problem.
    with np.errstate(all="ignore"):
        proj = centered @ principal
        perp = centered @ perp_axis
    # 2026-09-05 perf pass: this function used to call np.argsort(proj) /
    # np.argsort(-proj) up to seven times on the same two arrays. argsort
    # is deterministic on identical input, so each is computed once. The
    # two are NOT interchangeable (introsort's tie order differs between
    # `proj` and `-proj`, and exact ties do occur for symmetric integer
    # pixel pairs), which is why both are kept rather than reversing one.
    proj_asc = np.argsort(proj)
    proj_desc = np.argsort(-proj)
    pmin, pmax = float(proj.min()), float(proj.max())
    span = pmax - pmin
    window = max(END_WINDOW_MIN_PX, END_WINDOW_FRACTION * span)

    end_a = proj <= pmin + window
    end_b = proj >= pmax - window
    width_a = float(perp[end_a].std()) if end_a.sum() >= 2 else 0.0
    width_b = float(perp[end_b].std()) if end_b.sum() >= 2 else 0.0

    # Only let a companion override a CONFIDENT width-based decision when
    # that decision is itself ambiguous (widths nearly equal) -- measured
    # on the real 354-pair corpus: when width
    # comparison already clearly favors one end (ratio below this gate),
    # the companion is more often a broken-off fragment of the TRUE tip
    # itself (real, distinct failure mode from the flight-separation one
    # this override targets) than the flight, and overriding a confident
    # width verdict regressed several real cases substantially (up to
    # +262px) for a similar total of real wins -- a wash, not a net gain,
    # until gated by this ambiguity check.
    width_ratio = (
        min(width_a, width_b) / max(width_a, width_b)
        if max(width_a, width_b) > 0
        else 1.0
    )
    ambiguous = width_ratio >= WIDTH_AMBIGUITY_RATIO_MIN
    texture_used = False
    tex_a = tex_b = None
    # Real incident, 2026-08-12 (throw
    # throw_1786580447119, see module docstring): set True exactly when
    # this call's end-choice is unresolved by any DIRECT evidence -- no
    # companion blob found (forced_non_tip_end is None) AND the width
    # comparison itself is ambiguous (>= WIDTH_AMBIGUITY_RATIO_MIN). This
    # covers BOTH sub-cases that currently fall into the `else` branch
    # below: the texture tiebreak firing (width_ratio also >=
    # TEXTURE_TIEBREAK_RATIO_MIN) and the plain width guess in the
    # 0.6-0.86 gap where neither a companion nor the texture tiebreak is
    # available -- texture is a weak, ~70%-correct signal at best (see
    # TEXTURE_TIEBREAK_RATIO_MIN's comment), and a bare width comparison
    # in this range is close to a coin flip by construction, so neither
    # sub-case's single verdict is trustworthy alone. Deliberately does
    # NOT include forced_non_tip_end-resolved cases (companion evidence
    # is a real, different, already-validated signal, untouched here) or
    # the unambiguous case (width_ratio < WIDTH_AMBIGUITY_RATIO_MIN,
    # already reliable ~94% per the module docstring's measured numbers).
    alt_unresolved = False
    if forced_non_tip_end == "a" and ambiguous:
        tip_is_a = False
    elif forced_non_tip_end == "b" and ambiguous:
        tip_is_a = True
    else:
        tip_is_a = width_a <= width_b
        if forced_non_tip_end is None and ambiguous:
            alt_unresolved = True
        # No companion evidence, and the width signal is even MORE
        # ambiguous than the companion gate's own threshold -- fall back
        # to local background texture at each end (see
        # TEXTURE_TIEBREAK_RATIO_MIN's comment above). Only applies when
        # a background image was actually supplied.
        if (
            forced_non_tip_end is None
            and bg_gray is not None
            and width_ratio >= TEXTURE_TIEBREAK_RATIO_MIN
        ):
            k0 = min(N_TIP_POINTS_AVERAGED, len(pts))
            pt_a = pts[proj_asc[:k0]].mean(axis=0)
            pt_b = pts[proj_desc[:k0]].mean(axis=0)
            tex_a = _local_bg_texture(bg_gray, pt_a[0], pt_a[1])
            tex_b = _local_bg_texture(bg_gray, pt_b[0], pt_b[1])
            # Higher local texture = more likely the board face (numbers,
            # colored segments) rather than the uniform ceiling/wall
            # backdrop -- the board face is where the true tip is.
            tip_is_a = tex_a >= tex_b
            texture_used = True
    k = min(N_TIP_POINTS_AVERAGED, len(pts))
    idxs = proj_asc[:k] if tip_is_a else proj_desc[:k]
    tip_pt = pts[idxs].mean(axis=0)

    # The losing end, always -- see TipDetectionResult.far_end_px for why
    # this is computed unconditionally while alt_tip_px below is not.
    far_idxs = proj_desc[:k] if tip_is_a else proj_asc[:k]
    far_pt = pts[far_idxs].mean(axis=0)
    far_end_px = (
        (float(far_pt[0]), float(far_pt[1]))
        if (np.isfinite(far_pt[0]) and np.isfinite(far_pt[1]))
        else None
    )

    alt_tip_px = None
    if alt_unresolved:
        # Mirrors tip_pt's own construction (mean of the N most extreme
        # points), just for the end that LOST the width/texture
        # tiebreak -- the other candidate a cross-camera geometric check
        # (opendarts.engines.apollo.scoring.score_dart) can arbitrate between, since this
        # single image cannot.
        alt_idxs = proj_desc[:k] if tip_is_a else proj_asc[:k]
        alt_pt = pts[alt_idxs].mean(axis=0)
        if np.isfinite(alt_pt[0]) and np.isfinite(alt_pt[1]):
            alt_tip_px = (float(alt_pt[0]), float(alt_pt[1]))

    # 2026-08-17 -- off-axis tip-cluster alternate, real incident
    # the recorded T15 throw, cam2 (see
    # TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX's comment for the full write-up
    # and corpus measurement). A real tip lies ON the component's own
    # principal axis; a tip cluster whose MEAN perpendicular offset
    # exceeds the gate is dominated by an off-axis satellite diff
    # fragment (shadow/glint merged in by the 31px dilation), not the
    # dart. Never overwrites tip_px -- exposes the on-axis recompute as
    # `alt_tip_px` for score_dart()'s existing cross-camera combination
    # search to arbitrate, the same contract as the width-ambiguity alt
    # above. When the width-ambiguity alt is ALREADY set (`alt_unresolved`
    # -- the end choice itself was a coin flip), that existing,
    # separately-validated mechanism keeps the single alt slot untouched:
    # both candidate ENDS are already in the search, which is the bigger
    # ambiguity to resolve.
    tip_cluster_perp_px = float(perp[idxs].mean()) if len(idxs) else 0.0
    tip_off_axis_alt = False
    if (
        alt_tip_px is None
        and abs(tip_cluster_perp_px) > TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX
    ):
        on_axis = np.abs(perp) <= TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX
        if int(on_axis.sum()) >= 2:
            pts_on = pts[on_axis]
            proj_on = proj[on_axis]
            k_on = min(k, len(pts_on))
            on_idxs = (
                np.argsort(proj_on)[:k_on] if tip_is_a else np.argsort(-proj_on)[:k_on]
            )
            cand = pts_on[on_idxs].mean(axis=0)
            if np.isfinite(cand[0]) and np.isfinite(cand[1]):
                alt_tip_px = (float(cand[0]), float(cand[1]))
                tip_off_axis_alt = True

    # 2026-08-17 -- axial-gap tip-island alternate, real incident
    # S13 throw cam2 (see TIP_ISLAND_MIN_AXIAL_GAP_PX's
    # comment for the full write-up and corpus measurement). A faint
    # shadow/glint fragment BEYOND the physical tip, detached in the
    # original mask but bridged into the component by the 31px dilation,
    # wins the proj-extreme tip cluster while sitting ON the principal
    # axis -- invisible to the off-axis signature above. The measurable
    # contradiction: an axial run with zero original-mask points between
    # the tip cluster and the component body. Same contract as both
    # alternates above: never overwrites tip_px, populates the single
    # alt slot only when no other mechanism already claimed it, and
    # score_dart()'s existing cross-camera combination search
    # arbitrates. Diagnostics are computed unconditionally so the
    # healthy population stays measurable from stored packages.
    t_axis = -proj if tip_is_a else proj  # chosen tip end at max t_axis
    # argsort(-t_axis) is argsort(-(-proj)) == argsort(proj) when tip_is_a
    # (IEEE-754 negation is exact, so the input array is bit-identical)
    # and argsort(-proj) otherwise -- both already computed above.
    order_desc = proj_asc if tip_is_a else proj_desc
    ts_sorted = t_axis[order_desc]
    tip_island_axial_gap_px = 0.0
    tip_island_n_points = None
    tip_island_mean_diff = None
    tip_island_body_mean_diff = None
    tip_island_alt = False
    if len(ts_sorted) >= 2:
        seq_gaps = ts_sorted[:-1] - ts_sorted[1:]
        # A gap counts as "at the tip end" when its far (body) side is
        # still within the search window of the extreme point.
        within = (ts_sorted[0] - ts_sorted[1:]) <= TIP_ISLAND_SEARCH_WINDOW_PX
        if within.any():
            gaps_in_window = seq_gaps[within]
            tip_island_axial_gap_px = float(gaps_in_window.max())
        over_gate = np.where(
            (seq_gaps > TIP_ISLAND_MIN_AXIAL_GAP_PX) & within
        )[0]
        if len(over_gate):
            gi = int(over_gate[0])  # FIRST over-gate gap from the tip end
            island_idx = order_desc[: gi + 1]
            body_idx = order_desc[gi + 1 :]
            tip_island_n_points = int(len(island_idx))
            if diff_blur is not None:
                dox, doy = diff_origin
                ipx = pts[island_idx].astype(int)
                tip_island_mean_diff = float(
                    diff_blur[ipx[:, 1] - doy, ipx[:, 0] - dox].mean()
                )
                body_near = body_idx[
                    ts_sorted[gi + 1] - t_axis[body_idx] <= 30.0
                ]
                if len(body_near):
                    bpx = pts[body_near].astype(int)
                    tip_island_body_mean_diff = float(
                        diff_blur[bpx[:, 1] - doy, bpx[:, 0] - dox].mean()
                    )
            # Full three-part signature required (see
            # TIP_ISLAND_MIN_AXIAL_GAP_PX's comment: a gap alone occurs
            # on 20.2% of healthy detections -- real detached tip
            # sections; the island size + faintness gates are what
            # separate a shadow fragment from a real tip fragment).
            island_is_shadow_like = (
                tip_island_n_points <= TIP_ISLAND_MAX_N_POINTS
                and tip_island_mean_diff is not None
                and tip_island_mean_diff <= TIP_ISLAND_MAX_MEAN_DIFF
            )
            if island_is_shadow_like and alt_tip_px is None and len(body_idx) >= 2:
                k_b = min(k, len(body_idx))
                cand = pts[body_idx[:k_b]].mean(axis=0)
                if np.isfinite(cand[0]) and np.isfinite(cand[1]):
                    alt_tip_px = (float(cand[0]), float(cand[1]))
                    tip_island_alt = True

    diag = {
        "alt_tip_px": alt_tip_px,
        "far_end_px": far_end_px,
        "span_px": span,
        "tip_end": "a" if tip_is_a else "b",
        "tip_end_width_px": width_a if tip_is_a else width_b,
        "other_end_width_px": width_b if tip_is_a else width_a,
        "n_points": int(len(pts)),
        # "companion_found": a qualifying companion blob was located near
        # this end (see _find_companion_end), regardless of whether it
        # was actually USED. "companion_override": non-None only when it
        # was both found AND applied (ambiguous width decision) -- these
        # differ exactly when a companion was found near an end but the
        # width comparison was already confident (see the ambiguity-gate
        # comment above), i.e. the companion was found but ignored.
        "companion_found": forced_non_tip_end,
        "companion_override": forced_non_tip_end if ambiguous else None,
        "width_ratio": width_ratio,
        # "texture_tiebreak_used": True only when no companion applied AND
        # width was ambiguous enough to fall back to local background
        # texture (see TEXTURE_TIEBREAK_RATIO_MIN's comment above).
        "texture_tiebreak_used": texture_used,
        "texture_a": tex_a,
        "texture_b": tex_b,
        # 2026-08-17 -- off-axis tip-cluster signature (see the dated
        # comment above the tip_cluster_perp_px computation).
        # `tip_cluster_perp_px` is always present so the healthy
        # population stays measurable from stored packages;
        # `tip_off_axis_alt` is True only when the alternate actually
        # came from the on-axis recompute (never when the width-
        # ambiguity mechanism populated it).
        "tip_cluster_perp_px": tip_cluster_perp_px,
        "tip_off_axis_alt": tip_off_axis_alt,
        # 2026-08-17 -- axial-gap tip-island signature (see the dated
        # comment above). `tip_island_axial_gap_px` is always present
        # (largest within-window gap, 0.0 when none) so the healthy
        # population stays measurable from stored packages; the island
        # stats are non-None only when an over-gate gap exists, and
        # `tip_island_alt` is True only when the alternate actually came
        # from the island trim (never when another mechanism populated
        # the slot).
        "tip_island_axial_gap_px": tip_island_axial_gap_px,
        "tip_island_n_points": tip_island_n_points,
        "tip_island_mean_diff": tip_island_mean_diff,
        "tip_island_body_mean_diff": tip_island_body_mean_diff,
        "tip_island_alt": tip_island_alt,
    }
    return (float(tip_pt[0]), float(tip_pt[1])), diag


def detect_tip(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    prior_dart_line_px: tuple[tuple[float, float], tuple[float, float]] | None = None,
    *,
    precomputed: DiffCrop | None = None,
) -> TipDetectionResult:
    """Detect a thrown dart's tip pixel in `frame_bgr`, given a
    `bg_bgr` reference frame of the SAME camera showing the board state
    immediately before this dart landed.

    Both images must be same-shape BGR uint8 arrays from the same
    camera (e.g. `cv2.imread(...)` output). No calibration, board
    geometry, or other-camera information is used or required -- see
    module docstring for why (decoupled from opendarts.triangulation).

    `prior_dart_line_px`: added 2026-08-16, see module docstring's dated
    "prior_dart_line_px" section for the full real-incident write-up.
    Optional `(tip_px, far_end_px)` -- both ends of the SAME camera's own
    prior-throw-in-visit detection (see `opendarts.engines.apollo.
    prior_dart_context.find_prior_dart_line_px()` for the real caller-
    side lookup this is meant to be filled from). `None` (the default --
    every existing caller, and any caller with no visit/prior-throw
    context) disables this check entirely, byte-identical to this
    function's behavior before this parameter existed. When given, this
    is still plain pixel-space geometry (two more points, like
    `bg_gray`'s local-texture patches above) -- calibration/board
    geometry stays entirely out of this module.

    `precomputed`: optional `opendarts.imageops.DiffCrop` (2026-09-06 perf
    pass) -- this function's own gray/|diff|/blur front end, already
    computed by the caller (Zeus computes it once per camera and shares
    it across its sub-engines) and cropped to where the frame changed.
    Used only if it satisfies `PRECOMPUTE_REQUIREMENTS` for these exact
    images; otherwise ignored. Bit-identical either way: the crop
    contains every pixel the threshold/opening/dilation can light up
    (see `DiffCrop`'s docstring), the cropped dilated mask is pasted back
    into a zero full frame before connected-component labeling, and
    every pixel coordinate downstream is full-frame as before.
    """
    import cv2

    if bg_bgr.shape != frame_bgr.shape:
        return TipDetectionResult(
            ok=False, tip_px=None,
            reason=f"shape mismatch: bg {bg_bgr.shape} vs frame {frame_bgr.shape}",
        )

    pc = precomputed
    if pc is not None and not pc.accepts(PRECOMPUTE_REQUIREMENTS, bg_bgr.shape):
        pc = None
    if pc is None:
        img_h, img_w = bg_bgr.shape[:2]
        bg_gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        diff_blur = _blurred_diff(bg_gray, frame_gray)
        origin = (0, 0)
    else:
        img_h, img_w = pc.img_h, pc.img_w
        bg_gray = pc.bg_gray
        diff_blur = pc.diff_blur  # crop of the identical full-frame array
        origin = pc.origin

    # From here to the connected-component labeling every array is
    # crop-sized (the whole frame when there is no `precomputed`).
    mask = threshold_mask(diff_blur, DIFF_THRESHOLD)
    # The 3x3 opening runs on the whole (crop) array: before it, sensor
    # noise above threshold is spread over ~80% of the frame, so there is
    # no tight region to crop to yet. The 31px dilation AFTER it is a
    # different story -- the opened mask's non-zero bounding box is a
    # median ~4% of the frame, and the full-frame ellipse dilate alone was
    # 43% of this engine's per-throw time (2026-09-05 perf pass;
    # bit-identical, see `opendarts.imageops.morph_on_bbox`).
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ellipse_kernel(OPEN_KERNEL_PX))
    dilated = morph_on_bbox(opened, cv2.MORPH_DILATE, ellipse_kernel(DILATE_KERNEL_PX))
    if pc is not None:
        dilated = pc.paste_full(dilated)

    n, comp_labels, stats, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
    if n <= 1:
        return TipDetectionResult(
            ok=False, tip_px=None,
            reason="no diff components found (empty/near-empty diff mask)",
        )

    candidates = []
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        bboxes[i] = (x, y, w, h)
        # A component spanning most of the frame is diffuse lighting
        # drift or a global exposure shift, not a dart -- reject outright
        # rather than let it dominate by raw area.
        if w > 0.6 * img_w or h > 0.6 * img_h:
            continue
        candidates.append((area, i))
    if not candidates:
        return TipDetectionResult(
            ok=False, tip_px=None,
            reason="only frame-spanning diff components found (likely global lighting drift)",
        )
    candidates.sort(reverse=True)
    top = candidates[:TOP_K_AREA_CANDIDATES]

    chosen = None
    chosen_elongation = None
    for area, i in top:
        result = _component_stats_in_bbox(comp_labels, i, bboxes[i], opened, origin)
        if result is not None:
            pts, centered, principal, elongation = result
            if elongation >= MIN_ELONGATION_RATIO:
                chosen = (pts, centered, principal, area, i)
                chosen_elongation = elongation
                break
        # 2026-09-02 -- sub-component salvage, see
        # _largest_elongated_subcomponent()'s own docstring for the full
        # real-incident write-up (the recorded outside throw cam0). THIS
        # candidate's own whole-blob elongation failed (or it had too few
        # points for PCA at all) -- before moving on to a SMALLER top-K
        # candidate that might pass the whole-blob check on its own
        # (real risk: a smaller, unrelated, coincidentally-elongated
        # artifact outranking the true dart purely because the true
        # dart's own blob got merged with something else, dragging its
        # OWN elongation down -- exactly this incident's real root
        # cause), check whether THIS candidate's undilated footprint
        # still contains a genuinely elongated sub-piece. A qualifying
        # sub-piece from a LARGER candidate is preferred over a smaller
        # candidate's own whole-blob pass, since candidates are visited
        # in area-descending order and this check runs before advancing
        # to the next one.
        bx, by, bw, bh = bboxes[i]
        ox, oy = origin
        footprint = (comp_labels[by:by + bh, bx:bx + bw] == i) & (
            opened[by - oy:by - oy + bh, bx - ox:bx - ox + bw] != 0
        )
        sub = _largest_elongated_subcomponent(footprint, ((bx, by), (img_h, img_w)))
        if sub is not None:
            pts, centered, principal, sub_area, sub_elongation = sub
            chosen = (pts, centered, principal, sub_area, i)
            chosen_elongation = sub_elongation
            break

    if chosen is None:
        # Fall back to the single largest candidate even if it didn't
        # clear the elongation bar -- better than refusing outright, but
        # this is exactly the branch most likely to be a false positive
        # (see module docstring "Known limitations").
        area, i = top[0]
        result = _component_stats_in_bbox(comp_labels, i, bboxes[i], opened, origin)
        if result is None:
            return TipDetectionResult(
                ok=False, tip_px=None,
                reason="largest diff component has too few original-resolution pixels",
            )
        pts, centered, principal, elongation = result
        chosen = (pts, centered, principal, area, i)
        chosen_elongation = elongation

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
        pts, centered, principal, forced_non_tip_end=companion_end,
        bg_gray=bg_gray, diff_blur=diff_blur, diff_origin=origin,
    )
    if not (np.isfinite(tip_px[0]) and np.isfinite(tip_px[1])):
        # Belt-and-suspenders guard against a degenerate (near-zero-
        # variance) component producing a NaN/Inf principal axis -- same
        # class of latent bug landmark_detection.py already
        # found and fixed (an np.isfinite check is required, not `<=`
        # comparisons, since `nan <= x` is silently False in Python).
        return TipDetectionResult(
            ok=False, tip_px=None,
            reason="degenerate component produced a non-finite tip pixel",
        )

    far_end_px = tip_diag.get("far_end_px")
    # 2026-08-16 -- prior-dart-in-visit contamination SUSPICION signal,
    # see module docstring's dated "prior_dart_line_px" section and
    # PRIOR_DART_LINE_MAX_PERP_PX's own comment for the full real-
    # incident write-up and measurement (including why this is
    # DIAGNOSTIC-ONLY here, never an outright rejection -- an earlier
    # version of this guard rejected the camera directly from THIS
    # function and was measured, on the real corpus, to regularly
    # destroy legitimate well-agreeing multi-camera throws whenever a
    # later dart was thrown close to an earlier one, a normal and common
    # real darts outcome, not evidence of contamination by itself; see
    # opendarts.engines.apollo.engine.ApolloEngine.score() for where
    # the actual exclude-or-keep DECISION is made instead, using this
    # flag only as one candidate combination for score_dart()'s existing
    # disagreement-based selection to weigh against the unmodified full
    # set -- never a hard pre-filter). `tip_px`/`alt_tip_px`/`ok` are
    # completely unaffected by this signal -- purely additive
    # diagnostics, zero behavior change to this module's own return
    # value on the tip-pixel fields themselves. Requires BOTH ends close
    # to the prior dart's own line (not either alone, see that
    # constant's comment for why); skipped entirely when `far_end_px`
    # came out None (rare) or no prior context was given, in which case
    # `prior_dart_contamination_suspected` is simply absent from
    # diagnostics -- matching this module's "absent, never a guess"
    # discipline.
    prior_dart_diag: dict = {}
    if prior_dart_line_px is not None and far_end_px is not None:
        prior_a, prior_b = prior_dart_line_px
        perp_tip = _perp_dist_to_line(tip_px, prior_a, prior_b)
        perp_far = _perp_dist_to_line(far_end_px, prior_a, prior_b)
        prior_dart_diag = {
            "prior_dart_contamination_suspected": (
                max(perp_tip, perp_far) <= PRIOR_DART_LINE_MAX_PERP_PX
            ),
            "prior_dart_contamination_perp_tip_px": perp_tip,
            "prior_dart_contamination_perp_far_px": perp_far,
        }

    diagnostics = {
        "component_area_px": area,
        "elongation_ratio": chosen_elongation,
        "n_area_candidates": len(candidates),
        **tip_diag,
        **prior_dart_diag,
    }
    return TipDetectionResult(
        ok=True, tip_px=tip_px, reason="ok", diagnostics=diagnostics,
        alt_tip_px=tip_diag.get("alt_tip_px"),
        far_end_px=far_end_px,
    )
