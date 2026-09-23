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

Measured, not guessed (`opendarts/detection/evaluate_tip_detection.py`, run
against the full real dataset available at prototyping time -- 120 real
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
The caller's own "did this find anything" flag is a
sanity/crash signal, not a correctness signal -- do not treat a
successful detection as "trustworthy" without accounting for this tail.

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
and comparing to what this machinery actually returned on the real
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
see `_locate_tip_in_component`'s `alt_unresolved`), the result
now ALSO carries `alt_tip_px`: the tip point from the end that LOST that
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
"""
# --- Frozen-copy note (talos subpackage, 2026-08-13) -----------------
# Talos's own frozen copy of Apollo's original blob-detection
# machinery (diff mask, opened/dilated connected components, elongation
# ranking, companion-blob tip-end disambiguation) -- same reasoning as
# ray_fallback.py in this same package (read that module's docstring
# first). Talos's shaft_line.py was built and measured reusing this
# exact module verbatim when it lived at opendarts.detection.tip_detection
# -- that module was later deleted entirely during the Apollo
# subpackage refactor (moved into opendarts.engines.apollo.tip_detection,
# kept evolving there since). Frozen here as Talos's own private
# dependency so it can't silently drift out from under Talos's proven
# 88.8%/95.3% numbers again. shaft_line.py imports the constants and
# component helpers it needs (DILATE_KERNEL_PX, MIN_ELONGATION_RATIO,
# OPEN_KERNEL_PX, TOP_K_AREA_CANDIDATES, _component_stats_in_bbox,
# _diff_mask, _find_companion_end, _locate_tip_in_component); the
# whole-image entry point the original module also carried was never
# reached from Talos and was trimmed 2026-09-17.
# -------------------------------------------------------------------
from __future__ import annotations

import numpy as np

from opendarts.imageops import (
    PrecomputeRequirements,
    blurred_abs_diff,
    component_pixels,
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
DIFF_THRESHOLD = 30.0

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
# `shaft_line.fit_shaft_line_px()` to use it in place of this module's
# own gray/diff/blur front end (2026-09-06 perf pass): the pad covers
# the dilation's reach (radius 15) plus the opening's 3px window -- see
# `DiffCrop`'s docstring for why that makes the cropped morphology exact.
PRECOMPUTE_REQUIREMENTS = PrecomputeRequirements(
    ksize=GAUSSIAN_BLUR_KSIZE,
    threshold=DIFF_THRESHOLD,
    pad_px=DILATE_KERNEL_PX // 2 + OPEN_KERNEL_PX,
)

MIN_COMPONENT_PIXELS = 15
# Rank-ordered by area; the first ranked candidate whose elongation ratio
# (sqrt(major_eigenvalue / minor_eigenvalue) of its PCA covariance) meets
# this clears the "is this dart-shaped" bar. See module docstring step 4.
TOP_K_AREA_CANDIDATES = 3
MIN_ELONGATION_RATIO = 2.5

# Fraction (of the component's along-axis span) used to define "near an
# end" when measuring perpendicular spread at each end -- see module
# docstring step 5.
END_WINDOW_FRACTION = 0.15
END_WINDOW_MIN_PX = 6.0
N_TIP_POINTS_AVERAGED = 3

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
# candidate end point. Not independently swept (a bug in the sweep
# script silently reused one radius for every value tried); chosen to
# be large
# enough to capture real board texture (segment/number-width scale) but
# small enough to stay local to the actual end point, comparable to
# END_WINDOW_MIN_PX. Flagged honestly as an unswept parameter, same
# caveat this project applies to any constant it hasn't exhaustively
# tuned (see COMPANION_MAX_GAP_PX's comment above for the same pattern).
TEXTURE_PATCH_RADIUS_PX = 12


def _diff_mask(bg_gray: np.ndarray, frame_gray: np.ndarray) -> np.ndarray:
    # 2026-09-05 perf pass: same arithmetic via opendarts.imageops (uint8
    # absdiff -> one float32 cast, cv2.compare threshold), bit-identical.
    return threshold_mask(
        blurred_abs_diff(bg_gray, frame_gray, GAUSSIAN_BLUR_KSIZE), DIFF_THRESHOLD
    )


def _component_stats_in_bbox(
    comp_labels: np.ndarray,
    comp_id: int,
    bbox: tuple[int, int, int, int],
    opened: np.ndarray | None,
    mask_origin: tuple[int, int] = (0, 0),
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """PCA of one label's pixels, evaluated
    only inside the label's own bounding box (2026-09-05 perf pass) --
    same pixels in the same order, so the PCA input is bit-identical; see
    `opendarts.imageops.component_pixels`. The full-frame triple this
    replaces was ~1.6ms per call and ran per candidate plus per OTHER
    component in `_find_companion_end()`. `mask_origin` is the full-frame
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

    `opened` is the uint8 opened mask and `bboxes` maps every label to
    its `(x, y, w, h)` from connectedComponentsWithStats (2026-09-05
    perf pass): each candidate's pixels are read from its own bounding
    box, not the full frame -- this loop over every OTHER component was
    the source of this engine's worst-case latency tail.
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
    when width_ratio doesn't clear the gate."""
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
            pt_a = pts[np.argsort(proj)[:k0]].mean(axis=0)
            pt_b = pts[np.argsort(-proj)[:k0]].mean(axis=0)
            tex_a = _local_bg_texture(bg_gray, pt_a[0], pt_a[1])
            tex_b = _local_bg_texture(bg_gray, pt_b[0], pt_b[1])
            # Higher local texture = more likely the board face (numbers,
            # colored segments) rather than the uniform ceiling/wall
            # backdrop -- the board face is where the true tip is.
            tip_is_a = tex_a >= tex_b
            texture_used = True
    k = min(N_TIP_POINTS_AVERAGED, len(pts))
    idxs = np.argsort(proj)[:k] if tip_is_a else np.argsort(-proj)[:k]
    tip_pt = pts[idxs].mean(axis=0)

    alt_tip_px = None
    if alt_unresolved:
        # Mirrors tip_pt's own construction (mean of the N most extreme
        # points), just for the end that LOST the width/texture
        # tiebreak -- the other candidate a cross-camera geometric check
        # (opendarts.pipeline.score_dart) can arbitrate between, since this
        # single image cannot.
        alt_idxs = np.argsort(-proj)[:k] if tip_is_a else np.argsort(proj)[:k]
        alt_pt = pts[alt_idxs].mean(axis=0)
        if np.isfinite(alt_pt[0]) and np.isfinite(alt_pt[1]):
            alt_tip_px = (float(alt_pt[0]), float(alt_pt[1]))

    diag = {
        "alt_tip_px": alt_tip_px,
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
    }
    return (float(tip_pt[0]), float(tip_pt[1])), diag
