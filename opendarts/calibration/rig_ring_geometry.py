"""RIG-CONSENSUS ORIENTATION -- learned ring geometry, from the rig's own
calibration history.

Built 2026-08-29 for two decisions:

    1. No hardcoded orientation fallback: a calibration that cannot
       determine orientation makes the user act or refuses to
       calibrate. This is what makes R1 (delete the hardcoded
       `MEASURED_CAMERA_ORIENTATION_HINTS_DEG` fallback, refuse loudly
       on failure) possible: this module is what a caller consults
       INSTEAD of that constant.
    2. Learned ring geometry is a primary method, not a fallback: this
       module is consulted on EVERY calibration that has ring geometry
       already learned (not just as a last-resort fallback branch), per
       spec R2.

THE PHYSICAL FACT THIS RESTS ON (spec section 2, measured on 21 real
all-live calibrations spanning a real board-rotation + camera-
re-enumeration event, from both rigs' stored calibration packages): the
board's rotation is one physical number, so the
*absolute* per-camera orientation hint drifts the instant the board (or
a camera) moves -- but the three cameras sit on a rigid mounting ring,
so the *gaps* between their hints (the arcs a rigid ring's fixed
azimuths carve out of the circle) are a property of the RING, not of
the board's current rotation or of which OS index a camera happens to
enumerate as. Verified independently, this module's own reasoning, on
the same real corpus: computing
these gaps as the CIRCULAR CONSECUTIVE DIFFERENCES of the three hints,
sorted ascending, reproduces the spec's own quoted table almost exactly
(103.3deg / 107.8deg / 148.9deg mean, sub-2deg spread) -- confirming
this module's `raw_ordered_gaps()` is computing the same real quantity
the spec's own measurement used, not a differently-defined one.

WHAT "IDENTIFY THE ARRANGEMENT" ACTUALLY REQUIRES -- a real subtlety
this module's own validation surfaced, not assumed correct on the first
attempt (see the same validation script's own commit history in `tmp/`
for the three iterations this took): given only TWO confidently-live
cameras and geometry with N=3 gaps, the pair's own mutual offset alone
is NOT always enough to place the missing third camera uniquely. When
the two known cameras are the ADJACENT pair (separated by exactly one
stored gap), the missing camera sits on the OTHER, two-gap arc -- and
matching by TOTAL MAGNITUDE alone is a real, provable tie between the
two possible internal splits of that two-gap arc (summing "all but the
skipped gap" gives an identical total regardless of which of the two
remaining gaps comes first). This is not a corner case this module
mostly avoids -- for N=3 with exactly one missing camera, EVERY
prediction has this exact shape (the missing camera is always on one of
exactly two arcs between the two known cameras). Confirmed empirically:
picking the wrong split of the tied pair produces errors of 40deg+,
not a few degrees of noise -- silently wrong in a way reprojection
error cannot catch, the same "confidently wrong" failure class this
whole spec exists to close.

**The tie-break: use the missing camera's own best (even sub-floor)
candidate hint whenever one exists** -- exactly spec R2.2 step 3's own
wording ("its own best candidate either matches the prediction or is
absent"). Validated on the real leave-one-out corpus with this
tie-break wired in: mean |error|
0.24deg / worst 0.91deg across all three cameras' own leave-one-out
predictions (63 predictions, 21 events); cam2 specifically (matching
the spec's own worked example) 0.19deg mean / 0.46deg worst -- both
comfortably inside an order of magnitude of the spec's own quoted
0.12deg / 0.37deg (the spec's own number is presumably a slightly
different sampling/averaging choice; both numbers land at the same
conclusion: a ~25-75x margin under the 9deg half-sector decision
boundary). D2 alias rejection (a synthetic +162deg alias injected into
one camera's own hint, predicted independently from the other two via
this exact machinery): 21/21 real events correctly reject the alias
(every tied candidate disagrees with the aliased value by *far* more
than the 9deg half-sector threshold -- no tie-break needed there, since
162deg dwarfs the ~1deg-scale ambiguity this module's tie-break exists
to resolve).

**POST-SHIP FIX, 2026-08-31 -- the mirror-ambiguity tie-break,
RESOLVED.** The paragraph above used to
describe this module's own real, documented residual risk: when a
camera has NO candidate at all (or, per a second real bug this same
finding surfaced, only a candidate the D2 cross-check has already
proven untrustworthy) AND `candidate_predictions()` returns a genuine
tie (the structural mirror ambiguity `RingGeometry`'s own docstring
above describes -- the gap sequence read forward vs backward both fit
equally well), this module used to silently pick whichever candidate
the canonical direction's own enumeration order produced first --
"confidently wrong, not visibly wrong," the EXACT failure class this
whole spec exists to eliminate, just relocated from a stale hardcoded
constant into the tie-break itself. Measured on real deployed geometry
by a QA peer session after shipping R1/R2/R3: a blended pre-/post-
board-rotation sample window (gaps `[107.77, 148.94, 103.29]`, a real,
reachable condition -- R3's own drift check correctly does NOT reject
this, since a board rotation shifts the gaps by only ~1.3deg between
cameras that never moved relative to each other) produced two tied
candidates (`err 0.76` each) 41.7deg (2.3 wedges) apart, with the
WRONG one silently chosen and reported `ok=True`, `"rig_consensus"`.

**Fixed per the spec's own "Recommended direction": when candidates are
tied within tolerance and there is no TRUSTWORTHY tie-break available --
the camera produced no candidate at all, OR its own candidate was
proven wrong by the D2 cross-check -- `predict_missing_hint()` now
returns `None` (refuse) instead of picking one by enumeration order.**
This is R1's own "refuse rather than silently guess" philosophy applied
one level deeper, exactly as the spec recommends, and it reuses this
module's EXISTING refusal contract rather than inventing a new one:
`predict_missing_hint() -> None` was already the established "cannot
resolve this camera" signal (the empty-candidates case), and
`resolve_rig_consensus_orientation()`'s step 3 fill-in loop already
turns a `None` prediction into a proper `RigConsensusResult(ok=False,
refusal_reason=...)` -- see that function's own docstring, unchanged by
this fix. A camera with zero candidate that is NOT ambiguous (only one
candidate survives `candidate_predictions()`'s own tolerance grouping)
is still filled in exactly as before -- refusing there would be overly
conservative and not what the spec asks for; only a GENUINE tie with no
trustworthy tiebreak now refuses.

**The second real bug this same finding named, also fixed here**: the
D2 cross-check loop inside `resolve_rig_consensus_orientation()` used to
only record a camera in `d2_rejected_own_value` (the set excluded from
being used as its own tiebreak) `if cam in anchors` -- i.e. only for a
camera that had already cleared the confidence floor. A camera that was
already sub-floor (never an anchor to begin with) but whose own
candidate ALSO failed the D2 cross-check kept its own just-proven-wrong
value eligible as a tiebreak anyway, defeating the entire point of
having rejected it (confirmed reproducible: a sub-floor aliased
candidate a D2 check correctly flagged as 120.89deg off consensus was
still used to pick between two tied fill-in candidates, landing 41.1deg
from truth). Fixed by tracking D2 rejection unconditionally, regardless
of whether the camera was ever an anchor -- "this camera's own value has
been proven untrustworthy by cross-check" is the same fact either way.

RING GEOMETRY MUST BE LEARNED, NEVER HARDCODED (spec R3) -- storage
mirrors this project's own already-established "small, self-correcting,
JSON-backed last-known-good value" convention exactly
(`focal_length_fallback.json`, `distortion_fallback.json`,
`principal_point_fallback.json` -- see `opendarts/calibration/
focal_length.py` for the precedent this module's own
load/save/schema/degrade-safely shape is deliberately copied from, not
reinvented). `ring_geometry_fallback.json`, same directory
(`calibration_package_root`), schema `ring-geometry-v1`. Provenance
carried on every write: `n_events` (lifetime count, monotonic, never
reset by the sample window below), `first_learned_utc`,
`last_updated_utc`, and `spread_deg` (population std per gap slot,
computed from the CURRENT sample window, not the lifetime average --
see `RING_GEOMETRY_MAX_SAMPLES` below).

**Update uses a capped recent-sample window (`RING_GEOMETRY_MAX_SAMPLES
= 50`), not a lifetime running mean.** A lifetime mean would let a slow,
genuinely-real physical drift (mount settling, a slight re-tightening)
asymptotically vanish into an ever-larger denominator, defeating the
whole point of a self-correcting value; a capped window keeps the
stored geometry representative of *recent* reality, the same
"staleness is bounded to since the last N updates" property
`focal_length_fallback.json`'s own docstring already establishes for a
different constant.

DRIFT DETECTION (spec R3's own "if observed gaps drift beyond a
threshold, refuse and tell the operator a camera has moved"):
`RING_GEOMETRY_DRIFT_THRESHOLD_DEG = 4.0`, matching the spec's own
suggested starting point -- confirmed against this module's own
leave-one-out measurement on the SAME real 21-event corpus
-- not accepted blindly: the real
observed leave-one-out max per-gap deviation (predicting the LEFT-OUT
event's own aligned gaps against geometry learned from the other 20)
is 1.248deg, mean 0.391deg -- across ALL 21 events, INCLUDING the three
genuinely cross-era (pre- vs post-board-rotation) ones once correctly
aligned. 4.0deg is ~3.2x this real worst-case same-rig figure (the
spec's own reasoning uses its own slightly different "worst spread
1.65deg" measurement for the same 2.4x-ish margin -- both land on the
same real conclusion) while staying well under the 9deg half-sector
identification boundary (so a genuine gap-identity swap could never be
confused with ordinary noise). **Recommendation, not a unilaterally
final decision** -- flagged explicitly as an open question, exactly as
every other measured
threshold in this project's history has been presented for review
before being treated as permanently settled.

INPUTS, verified for this module specifically (every calibration input
is derived from this rig's own frames): every number this module ever
touches originates from
`opendarts.calibration.ring_correlation_orientation`'s own
camera-image-only signature (via each event's already-computed
`orientation_hint_deg`) -- this module only ever combines those
per-camera outputs across cameras and across time; it reads nothing
outside this rig's own real calibration history.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

RING_GEOMETRY_FALLBACK_FILENAME = "ring_geometry_fallback.json"
SCHEMA = "ring-geometry-v1"

# See this module's own top docstring, "RING GEOMETRY MUST BE LEARNED"
# section, for why a capped window rather than a lifetime mean.
RING_GEOMETRY_MAX_SAMPLES = 50

# See this module's own top docstring, "DRIFT DETECTION" section, for
# the real leave-one-out measurement this is set from.
RING_GEOMETRY_DRIFT_THRESHOLD_DEG = 4.0

# lock_orientation()'s own candidates are one sector (18deg) apart.
# Half a sector is the natural "which side of the
# boundary" decision margin for the D2 alias cross-check (spec R2.2
# step 2's own "more than a half-sector (9deg)" wording).
HALF_SECTOR_DEG = 9.0

# How close a candidate prediction's own MATCH ERROR (how well the
# known-camera pair's observed offset fits a candidate span of stored
# gaps) has to be to the single best match before it's treated as a
# genuine tie requiring the sub-floor-candidate tie-break, rather than a
# clearly-worse alternative that can just be discarded. Real gaps are
# 4deg+ apart in magnitude (spec section 2) with sub-2deg spread, so a
# genuine tie (the "which side of the two-gap arc" ambiguity this
# module's own top docstring describes) always lands within a couple of
# degrees of the best match's own error; a spuriously-close WRONG match
# from an unrelated gap combination would need a coincidence far larger
# than the real spread ever shows. 3.0deg has real margin above the
# largest real per-event alignment error this module's own leave-one-out
# measurement found (1.248deg) without being anywhere near the 9deg
# half-sector danger zone.
_CANDIDATE_MATCH_TOLERANCE_DEG = 3.0
_TIE_GROUP_TOLERANCE_DEG = 0.5


def _circ_diff(a: float, b: float) -> float:
    """Smallest angular distance between two directions, degrees,
    always >= 0."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


@dataclass
class RingGeometry:
    """Learned ring geometry for THIS rig -- see this module's own top
    docstring. `gaps_deg` is the canonical ORDERED (not sorted-by-
    magnitude) list of N consecutive circular gaps between the N
    cameras' hints, in a FIXED cyclic direction established by whichever
    event first seeded this geometry and preserved (via alignment, see
    `_align_gaps()`) by every update since -- reflection is a real,
    legitimate alignment operation during LEARNING (a later event's own
    raw gap sequence may need to be read in reverse to match the
    existing canonical direction; see this module's own top docstring
    for the real cross-era case that made this necessary). `candidate_
    predictions()` ALSO tries both directions at prediction time -- see
    its own docstring for why an earlier draft that searched the
    canonical direction only was empirically wrong on real data.
    """
    gaps_deg: list[float]
    n_events: int
    first_learned_utc: str
    last_updated_utc: str
    spread_deg: list[float]
    # Capped recent-sample window (see RING_GEOMETRY_MAX_SAMPLES) --
    # each entry is one event's own ALIGNED raw gap list, oldest first.
    # Persisted (not just derived transiently) so a fresh process
    # reloading this file can keep updating the SAME window rather than
    # restarting it from a single fresh sample.
    recent_samples: list[list[float]] = field(default_factory=list)

    @property
    def n_cameras(self) -> int:
        return len(self.gaps_deg)


def raw_ordered_gaps(hints: dict[int, float]) -> list[float]:
    """The N consecutive circular gaps between `hints`' values, sorted
    ascending by HINT VALUE (not camera index) before differencing --
    see this module's own top docstring for why this is a real,
    physically meaningful, index-agnostic quantity (survives both board
    rotation and USB re-enumeration) and reproduces the spec's own
    quoted real numbers almost exactly."""
    n = len(hints)
    ordered = sorted(hints.values())
    return [(ordered[(i + 1) % n] - ordered[i]) % 360.0 for i in range(n)]


def _align_gaps(gaps: list[float], reference: list[float]) -> list[float]:
    """Find the rotation (and, if needed, reflection) of `gaps` that
    best matches `reference`, minimizing total absolute per-slot
    deviation -- both real, legitimate alignment operations at LEARNING
    time (see this module's own top docstring). Returns the aligned
    version of `gaps` (same values, reordered/reflected), NOT a merged
    or averaged result."""
    n = len(gaps)
    best_err: float | None = None
    best_cand: list[float] = gaps
    for reflect in (False, True):
        g = list(reversed(gaps)) if reflect else list(gaps)
        for rot in range(n):
            cand = g[rot:] + g[:rot]
            err = sum(abs(cand[i] - reference[i]) for i in range(n))
            if best_err is None or err < best_err:
                best_err = err
                best_cand = cand
    return best_cand


def max_gap_deviation_deg(hints: dict[int, float], geometry: RingGeometry) -> float | None:
    """How far `hints`' own raw ordered gaps (aligned to `geometry`'s
    canonical direction) deviate from `geometry.gaps_deg`, per-slot,
    worst-case. `None` if `hints` doesn't have exactly `geometry.
    n_cameras` entries (nothing to compare). This is what R3's drift
    refusal is gated on -- see RING_GEOMETRY_DRIFT_THRESHOLD_DEG's own
    comment for the real measurement behind the threshold."""
    if len(hints) != geometry.n_cameras:
        return None
    raw = raw_ordered_gaps(hints)
    aligned = _align_gaps(raw, geometry.gaps_deg)
    return max(abs(aligned[i] - geometry.gaps_deg[i]) for i in range(geometry.n_cameras))


def update_ring_geometry(
    existing: RingGeometry | None,
    hints: dict[int, float],
    now_utc: str,
    *,
    max_samples: int = RING_GEOMETRY_MAX_SAMPLES,
    drift_threshold_deg: float = RING_GEOMETRY_DRIFT_THRESHOLD_DEG,
) -> tuple[RingGeometry, float | None]:
    """Update (or seed) ring geometry from ONE all-cameras-live,
    mutually-agreeing calibration event's own final hints (spec R3:
    "updated whenever a calibration lands with all cameras live and
    mutually consistent"). Returns `(new_or_unchanged_geometry,
    drift_deg)`.

    `drift_deg` is `None` when there was nothing to compare against yet
    (this is the FIRST event -- geometry seeded fresh, exactly as R3's
    own "on a virgin rig with no history, the first calibration runs
    per-camera only" already describes) or the camera COUNT doesn't
    match (also treated as "nothing to learn from this event," not an
    error). Otherwise `drift_deg` is the real measured deviation (see
    `max_gap_deviation_deg()`) -- **the caller decides what to do with
    it**, this function does not itself refuse: if `drift_deg >
    drift_threshold_deg`, `existing` is returned UNCHANGED (never
    silently absorbed into the running average) and the caller is
    expected to treat this as R3's "a camera has moved" refusal
    condition, exactly like every other real "measure the number, let
    the caller decide the consequence" split this module's own siblings
    (`focal_length.py` et al) already use.
    """
    n = len(hints)
    raw = raw_ordered_gaps(hints)
    if existing is None:
        return (
            RingGeometry(
                gaps_deg=raw,
                n_events=1,
                first_learned_utc=now_utc,
                last_updated_utc=now_utc,
                spread_deg=[0.0] * n,
                recent_samples=[raw],
            ),
            None,
        )
    if n != existing.n_cameras:
        log.warning(
            "ring geometry update skipped: this event has %d camera(s), stored "
            "geometry was learned from %d -- nothing to compare, leaving stored "
            "geometry unchanged.",
            n, existing.n_cameras,
        )
        return existing, None

    aligned = _align_gaps(raw, existing.gaps_deg)
    drift = max(abs(aligned[i] - existing.gaps_deg[i]) for i in range(n))
    if drift > drift_threshold_deg:
        # The CALLER decides what to do about it -- capture_daemon's
        # ring_geometry_for_this_event() relearns the layout from this event
        # and says so in its own line right after this one.
        log.warning(
            "ring geometry drift %.3fdeg exceeds the %.1fdeg threshold -- this "
            "update leaves the stored geometry alone (a physical camera move is "
            "the likely explanation; this event's own gaps: %s vs stored %s).",
            drift, drift_threshold_deg,
            [round(x, 2) for x in aligned], [round(x, 2) for x in existing.gaps_deg],
        )
        return existing, drift

    samples = existing.recent_samples + [aligned]
    if len(samples) > max_samples:
        samples = samples[-max_samples:]
    new_gaps = [sum(s[i] for s in samples) / len(samples) for i in range(n)]
    if len(samples) > 1:
        spread = [
            (sum((s[i] - new_gaps[i]) ** 2 for s in samples) / len(samples)) ** 0.5
            for i in range(n)
        ]
    else:
        spread = [0.0] * n
    return (
        RingGeometry(
            gaps_deg=new_gaps,
            n_events=existing.n_events + 1,
            first_learned_utc=existing.first_learned_utc,
            last_updated_utc=now_utc,
            spread_deg=spread,
            recent_samples=samples,
        ),
        drift,
    )


def candidate_predictions(
    known_hints: dict[int, float],
    geometry: RingGeometry,
    missing_cam: int,
    *,
    match_tolerance_deg: float = _CANDIDATE_MATCH_TOLERANCE_DEG,
) -> list[tuple[float, float]]:
    """Every plausible `(match_error_deg, predicted_hint_deg)` pair for
    `missing_cam`, given `known_hints` (must be exactly
    `geometry.n_cameras - 1` entries, NOT including `missing_cam`) and
    the stored `geometry`. Sorted by match_error ascending. Empty if
    `known_hints` doesn't have exactly N-1 entries, or `missing_cam` is
    already in `known_hints`, or geometry doesn't have exactly N
    cameras' worth of gaps (N-camera support beyond N=3 is untested by
    this module's own real-corpus validation, but the algorithm itself
    is not hardcoded to N=3).

    **Searches BOTH the canonical direction of `geometry.gaps_deg` and
    its reflection** -- a real, measured necessity, not a theoretical
    hedge: this module's own real-corpus validation
    (this task) found that SOME real
    calibration events' own raw gap sequence only aligns to the stored
    canonical geometry in reflection (the same real cross-era case
    `RingGeometry`'s own docstring describes for LEARNING). An earlier
    draft of this function searched the canonical direction only,
    reasoning that camera-index re-enumeration (a pure relabeling of
    already-index-agnostic hint values) can never itself induce a
    reflection -- true, but empirically insufficient: leave-one-out
    testing against this exact module showed canonical-only search
    reproduces the SAME ~40-46deg-error failure this whole investigation
    exists to prevent, on real data, for real events. WHY some events
    need reflection and others don't is not fully explained by this task
    (a genuine physical ring reconfiguration is presumably
    indistinguishable from this at inference time using gap-magnitude
    matching alone) -- reported honestly as an open question, not
    papered over; searching both directions and relying on the
    tie-break below to pick correctly is the empirically-validated fix
    (mean |error| 0.24deg / worst 0.91deg across all three cameras'
    leave-one-out predictions on the real 21-event corpus, WITH the
    tie-break wired in -- WITHOUT it, canonical-direction-only search
    was silently wrong by 40deg+ on exactly the real events that needed
    reflection). Returns MULTIPLE candidates (not just the single best)
    specifically so a caller can apply spec R2.2 step 3's own tie-break
    (`predict_missing_hint()` below) rather than silently picking one --
    see this module's own top docstring for the real, provable
    ambiguity this exists to surface rather than hide."""
    n = geometry.n_cameras
    if n < 2 or len(known_hints) != n - 1 or missing_cam in known_hints:
        return []
    known_cams = sorted(known_hints)
    a, b = known_cams[0], known_cams[1]
    observed = (known_hints[b] - known_hints[a]) % 360.0
    out: list[tuple[float, float]] = []
    for g in (geometry.gaps_deg, list(reversed(geometry.gaps_deg))):
        for start in range(n):
            for length in range(1, n):
                span = sum(g[(start + j) % n] for j in range(length))
                err = min(abs(span - observed), 360.0 - abs(span - observed))
                if err > match_tolerance_deg:
                    continue
                cum = [0.0]
                acc = 0.0
                for j in range(n - 1):
                    acc += g[(start + j) % n]
                    cum.append(acc)
                used = {0, length % n}
                missing_slots = [s for s in range(n) if s not in used]
                if len(missing_slots) != 1:
                    continue
                pred = (known_hints[a] + cum[missing_slots[0]]) % 360.0
                out.append((err, pred))
    out.sort(key=lambda t: t[0])
    return out


def predict_missing_hint(
    known_hints: dict[int, float],
    geometry: RingGeometry,
    missing_cam: int,
    *,
    tiebreak_hint: float | None = None,
    tie_group_tolerance_deg: float = _TIE_GROUP_TOLERANCE_DEG,
) -> float | None:
    """`missing_cam`'s predicted hint from `known_hints` + `geometry`,
    resolving the real tie this module's own top docstring describes
    via `tiebreak_hint` (spec R2.2 step 3: the missing camera's own
    best, even sub-floor, candidate -- pass it whenever ANY candidate
    was produced for that camera, even a low-confidence one, AND it
    hasn't been proven untrustworthy by the D2 cross-check -- see
    `resolve_rig_consensus_orientation()`'s own `d2_rejected_own_value`
    handling for that exclusion). `None` if `candidate_predictions()`
    has nothing to offer at all.

    **`None` is ALSO returned when candidates are genuinely tied (within
    `tie_group_tolerance_deg`) and `tiebreak_hint` is `None`** -- this
    module's own top docstring's "POST-SHIP FIX" section: a real, provable
    mirror ambiguity (the gap sequence read forward vs backward both fit
    the known pair's own observed offset equally well) with no trustworthy
    signal to break it is a REFUSAL condition (spec R1's own philosophy,
    "refuse rather than silently guess," applied one level deeper), never
    a silent pick by whichever candidate the canonical direction's own
    enumeration order happens to produce first. A single (non-tied)
    candidate is still returned even with `tiebreak_hint=None` -- there is
    nothing ambiguous to refuse in that case."""
    cands = candidate_predictions(known_hints, geometry, missing_cam)
    if not cands:
        return None
    best_err = cands[0][0]
    tied = [pred for err, pred in cands if err <= best_err + tie_group_tolerance_deg]
    if len(tied) > 1:
        if tiebreak_hint is None:
            return None
        tied.sort(key=lambda p: _circ_diff(p, tiebreak_hint))
    return tied[0]


def cross_check_disagreement_deg(
    cam: int,
    observed_hint: float,
    other_hints: dict[int, float],
    geometry: RingGeometry,
) -> float | None:
    """D2 cross-check (spec R2.2 step 2): predict `cam`'s hint as if it
    were the missing camera, using `other_hints` (every OTHER
    confidently-established camera this event, live OR already
    consensus-filled) + `geometry`, and return the circular disagreement
    between that prediction and `observed_hint` (whatever `cam` itself
    reported, regardless of its own confidence -- **the caller must run
    this on EVERY camera, including ones that cleared the live
    confidence floor**, not just as a rescue for cameras that failed
    it -- see this module's own predict_missing_hint() docstring and
    docs/DESIGN.md's 2026-08-29 dated entry for why a marginal live pass is
    not, by itself, evidence of being right on this rig).

    `None` when a prediction isn't possible (not exactly N-1 OTHER
    cameras, geometry camera-count mismatch, etc) -- the caller should
    treat `None` as "cannot cross-check this camera this event," not as
    "passed."""
    pred = predict_missing_hint(other_hints, geometry, cam, tiebreak_hint=observed_hint)
    if pred is None:
        return None
    return _circ_diff(pred, observed_hint)


# ---------------------------------------------------------------------
# Persistence -- same "small, self-correcting, JSON-backed last-known-
# good value" convention as focal_length.py/distortion.py/etc (see this
# module's own top docstring). Deliberately NOT merged into any of those
# other fallback files -- ring geometry is a property of the RIG as a
# whole (not per-camera), a genuinely different shape, and this project's
# own established precedent is one file per independently-evolving
# concern (see distortion.py's principal_point_fallback.json being kept
# separate from distortion_fallback.json for the identical reason).
# ---------------------------------------------------------------------

def load_ring_geometry(calibration_package_root: Path | None) -> RingGeometry | None:
    """Load persisted ring geometry from
    `<calibration_package_root>/ring_geometry_fallback.json`, or `None`
    if the root is `None`, the file doesn't exist, doesn't parse, or
    doesn't match this module's current `SCHEMA` -- same
    absent/corrupt/stale-schema-all-degrade-safely posture every sibling
    fallback file in this project already uses. `None` is the "no
    geometry known yet" signal a caller uses to pick Mode B (spec
    R2.1) -- this function never fabricates a geometry."""
    if calibration_package_root is None:
        return None
    path = Path(calibration_package_root) / RING_GEOMETRY_FALLBACK_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        log.exception("%s: failed to read/parse -- treating as absent (no ring "
                      "geometry known yet).", path)
        return None
    if payload.get("schema") != SCHEMA:
        log.warning("%s: schema %r does not match current %r -- treating as absent",
                    path, payload.get("schema"), SCHEMA)
        return None
    try:
        gaps = [float(x) for x in payload["gaps_deg"]]
        n_events = int(payload["n_events"])
        first_learned_utc = str(payload["first_learned_utc"])
        last_updated_utc = str(payload["last_updated_utc"])
        spread_deg = [float(x) for x in payload["spread_deg"]]
        recent_samples = [[float(x) for x in s] for s in payload.get("recent_samples", [])]
    except (KeyError, TypeError, ValueError):
        log.exception("%s: malformed contents -- treating as absent.", path)
        return None
    if len(gaps) < 2 or len(spread_deg) != len(gaps):
        log.warning("%s: gaps_deg/spread_deg shape mismatch -- treating as absent.", path)
        return None
    return RingGeometry(
        gaps_deg=gaps, n_events=n_events, first_learned_utc=first_learned_utc,
        last_updated_utc=last_updated_utc, spread_deg=spread_deg,
        recent_samples=recent_samples,
    )


def save_ring_geometry(calibration_package_root: Path, geometry: RingGeometry) -> None:
    """Overwrite the persisted ring geometry file -- plain
    `path.write_text(json.dumps(...))`, matching every sibling fallback
    file's own convention (a torn write from a crash mid-write degrades
    to "absent/corrupt", which `load_ring_geometry()` already treats as
    a safe, loud, non-fatal fallback state)."""
    calibration_package_root = Path(calibration_package_root)
    calibration_package_root.mkdir(parents=True, exist_ok=True)
    path = calibration_package_root / RING_GEOMETRY_FALLBACK_FILENAME
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "gaps_deg": geometry.gaps_deg,
        "n_events": geometry.n_events,
        "first_learned_utc": geometry.first_learned_utc,
        "last_updated_utc": geometry.last_updated_utc,
        "spread_deg": geometry.spread_deg,
        "recent_samples": geometry.recent_samples,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


# ---------------------------------------------------------------------
# ORCHESTRATION -- spec R2.2's full "identify the arrangement, cross-
# check, fill in, refuse if it can't be formed" sequence, as ONE pure
# function operating on plain per-camera candidate data. Deliberately
# factored out of `opendarts.live.capture_daemon.bootstrap_calibrations()`
# (a large, delicate, already-heavily-verified function) so this real
# logic -- the actual content of R2.2 -- is unit-testable in complete
# isolation, without needing to drive that function's own capture/detect
# machinery to exercise it. capture_daemon.py's own integration is a
# thin caller: gather each camera's
# `ring_correlation_orientation_for_camera()` result into
# `CameraOrientationCandidate`s and apply whatever this
# function decides.
# ---------------------------------------------------------------------

@dataclass
class CameraOrientationCandidate:
    """One camera's own orientation-hint evidence for THIS bootstrap
    event, regardless of whether it cleared the live-confidence floor --
    `hint_deg=None` means this camera produced literally zero candidate
    (not even a low-confidence guess); `confidence` is meaningless in
    that case and should be `0.0`."""
    hint_deg: float | None
    confidence: float


@dataclass
class RigConsensusResult:
    """Result of `resolve_rig_consensus_orientation()`. `ok=False` means
    spec R1's refusal condition was hit -- `refusal_reason` is an
    operator-actionable message (per spec section 3's own required-
    behaviour-on-failure wording), and `resolved`/`rejected_cameras` may
    still carry partial diagnostic information but MUST NOT be treated
    as a usable orientation resolution. `resolved` maps camera index to
    `(hint_deg, source)` where `source` is `"live"` (this camera's own
    derivation was trusted, cross-check included) or `"rig_consensus"`
    (this camera's hint was PREDICTED from other cameras + geometry --
    spec R2.2 step 3's own explicit requirement that this is NEVER
    labeled `"live"`). `rejected_cameras` maps camera index to a
    human-readable reason it did NOT keep its own live-derived value
    (D2 cross-check failure, sub-floor confidence, or no candidate at
    all) -- purely diagnostic, for logging."""
    ok: bool
    resolved: dict[int, tuple[float, str]] = field(default_factory=dict)
    refusal_reason: str | None = None
    rejected_cameras: dict[int, str] = field(default_factory=dict)


def resolve_rig_consensus_orientation(
    candidates: dict[int, CameraOrientationCandidate],
    geometry: RingGeometry,
    *,
    min_confidence: float,
    half_sector_deg: float = HALF_SECTOR_DEG,
) -> RigConsensusResult:
    """Spec R2.2, the full sequence, as one pure function:

    1. **Identify the arrangement / anchor discipline (R2.2 step 1,
       R2.3)** -- cameras whose own `confidence >= min_confidence` are
       candidate anchors. Fewer than two -> refuse (R2.3: "at least two
       anchors must agree with each other before either vouches for a
       third").
    2. **Cross-check EVERY camera with a candidate (R2.2 step 2, the D2
       alias detector) -- including confident ones**, per docs/DESIGN.md's
       2026-08-29 dated entry / this task's own coordinator amendment: a
       camera that merely clears the floor is not, by itself, evidence
       of being right on this rig (cam2's own real ~1.8-1.9sigma passes
       sit barely above a measured corrupted-evidence ceiling of
       1.754sigma). Any camera whose own candidate disagrees with what
       the OTHER cameras + geometry predict for it by more than
       `half_sector_deg` is rejected, regardless of confidence. A camera
       cannot be cross-checked when fewer than `n_cameras - 1` OTHER
       cameras have any candidate at all (not enough evidence to predict
       its expected position) -- in that case it is trusted as-is if it
       was already an anchor (R2.3's own "two anchors mutually agree"
       via the geometry MATCH in step 1 already is the available check
       in that reduced-evidence case), or left for step 3 to fill in if
       it wasn't.
    3. **Fill in the rest (R2.2 step 3)** -- any camera without a
       trusted own-derived value (never had a confident candidate, OR
       was rejected by cross-check) is predicted from the currently-
       trusted anchor set via `predict_missing_hint()`, tie-broken by
       its own best candidate when one exists (even sub-floor) --
       exactly spec R2.2 step 3's own wording. Recorded as
       `orientation_hint_source: "rig_consensus"`, never `"live"`.
    4. **Refuse if consensus cannot be formed** -- fewer than two
       mutually-agreeing anchors after cross-check rejections, or a
       camera that needs filling in but the remaining trusted anchor set
       can't produce a prediction for it.

    Only exercised when the caller already has `geometry` (spec R2.1
    Mode A) -- Mode B (no stored geometry) does not call this at all;
    every camera must derive live in Mode B, or the caller refuses
    per R1/section 3 directly, unrelated to this function."""
    n = geometry.n_cameras
    if len(candidates) != n:
        return RigConsensusResult(
            ok=False,
            refusal_reason=(
                f"rig-consensus orientation resolution needs exactly {n} camera(s) "
                f"(matching the stored ring geometry's own camera count), got "
                f"{len(candidates)} -- refusing rather than guessing."
            ),
        )

    anchors: set[int] = {
        cam for cam, c in candidates.items()
        if c.hint_deg is not None and c.confidence >= min_confidence
    }
    if len(anchors) < 2:
        detail = ", ".join(
            f"cam{cam}: {'no candidate' if c.hint_deg is None else f'{c.confidence:.2f}sigma (best {c.hint_deg:.2f}deg)'}"
            for cam, c in sorted(candidates.items())
        )
        return RigConsensusResult(
            ok=False,
            refusal_reason=(
                f"rig-consensus orientation could not be established: fewer than "
                f"2 cameras cleared the {min_confidence:.2f}sigma confidence floor "
                f"this event ({detail}). At least two mutually-agreeing anchors "
                f"are required before any camera can be predicted from ring "
                f"geometry -- refusing rather than trusting a single unverified "
                f"camera."
            ),
        )

    def _hints(cams: set[int]) -> dict[int, float]:
        return {c: candidates[c].hint_deg for c in cams if candidates[c].hint_deg is not None}

    # Step 1: verify the anchor set is actually consistent with stored
    # geometry -- this is the real "mutually agree" check when there
    # are exactly 2 anchors and no 3rd candidate at all to cross-check
    # against (candidate_predictions() returning nothing means their own
    # pairwise offset doesn't match ANY stored gap combination).
    trusted: set[int] = set(anchors)
    rejected: dict[int, str] = {}
    has_full_evidence = len({c for c in candidates if candidates[c].hint_deg is not None}) == n

    if not has_full_evidence and len(anchors) == 2:
        a, b = sorted(anchors)
        pair_hints = {a: candidates[a].hint_deg, b: candidates[b].hint_deg}
        missing_cam = next(iter(set(candidates) - {a, b}))
        if not candidate_predictions(pair_hints, geometry, missing_cam):
            return RigConsensusResult(
                ok=False,
                refusal_reason=(
                    f"rig-consensus orientation could not be established: the two "
                    f"anchor cameras (cam{a} {candidates[a].hint_deg:.2f}deg, "
                    f"cam{b} {candidates[b].hint_deg:.2f}deg) do not agree with "
                    f"the stored ring geometry at all -- their own observed offset "
                    f"({abs(candidates[b].hint_deg - candidates[a].hint_deg) % 360:.2f}deg) "
                    f"does not match any known gap combination. Refusing rather than "
                    f"guessing; this may mean a camera has moved on the ring."
                ),
            )

    # Step 2: D2 cross-check EVERY camera that has enough OTHER evidence
    # to be checked at all -- including anchors. May reject an anchor,
    # which is why this runs before step 3's fill-in.
    #
    # `d2_rejected_own_value` tracks EVERY camera whose OWN candidate is
    # what got proven inconsistent by the D2 cross-check -- step 3 below
    # must NOT use that same untrustworthy value as its fill-in
    # tie-break (a real bug this module's own real-corpus D2-alias
    # validation caught: using the aliased camera's own +162deg-off
    # value to pick between two tied fill-in candidates defeats the
    # entire point of having rejected it). This is UNCONDITIONAL on D2
    # rejection, regardless of whether `cam` was ever an anchor --
    # POST-SHIP FIX, 2026-08-31 (see this module's own top docstring,
    # "POST-SHIP FIX" section, and spec section 9): an EARLIER version
    # of this code only tracked this `if cam in anchors`, so a camera
    # that was ALREADY sub-floor (never an anchor to begin with) but
    # whose own candidate ALSO failed the D2 cross-check kept that same
    # just-proven-wrong value eligible as its own tiebreak anyway --
    # confirmed reproducible on real data: a sub-floor aliased candidate
    # D2 correctly flagged as 120.89deg off consensus was still used to
    # pick between two tied fill-in candidates, landing 41.1deg from
    # truth. "This camera's own value has been proven untrustworthy by
    # cross-check" is the same fact whether or not it ever cleared the
    # confidence floor -- a camera that was merely sub-floor and PASSED
    # D2 (never reached this branch at all) is still NOT in this set --
    # its own weak-but-unrefuted candidate remains real, useful evidence
    # for the tie-break, per spec R2.2 step 3's own wording. Only a
    # candidate D2 has actively disproven is excluded.
    d2_rejected_own_value: set[int] = set()
    for cam, cand in sorted(candidates.items()):
        if cand.hint_deg is None:
            continue
        others = _hints(set(candidates) - {cam})
        if len(others) != n - 1:
            continue # not enough other evidence to predict this camera's slot
        disagreement = cross_check_disagreement_deg(cam, cand.hint_deg, others, geometry)
        if disagreement is not None and disagreement > half_sector_deg:
            trusted.discard(cam)
            d2_rejected_own_value.add(cam)
            rejected[cam] = (
                f"D2 cross-check: own candidate {cand.hint_deg:.2f}deg disagrees "
                f"with ring-consensus prediction by {disagreement:.2f}deg "
                f"(> {half_sector_deg:.1f}deg half-sector) -- rejected regardless "
                f"of its own {cand.confidence:.2f}sigma confidence."
            )

    trusted_anchors = trusted & anchors
    if len(trusted_anchors) < 2:
        return RigConsensusResult(
            ok=False,
            refusal_reason=(
                "rig-consensus orientation could not be established: fewer than "
                "2 anchors remain after the D2 cross-check rejected one or more "
                f"as inconsistent with ring geometry ({rejected}). Refusing "
                "rather than extending from a single unverified camera."
            ),
            rejected_cameras=rejected,
        )

    # Step 3: fill in everything not in `trusted` (rejected anchors,
    # never-confident cameras) from `trusted`'s own hints.
    resolved: dict[int, tuple[float, str]] = {
        cam: (candidates[cam].hint_deg, "live") for cam in trusted
    }
    trusted_hints = _hints(trusted)
    for cam in sorted(set(candidates) - trusted):
        # See d2_rejected_own_value's own comment above: a D2-rejected
        # anchor's own value is exactly what was proven untrustworthy --
        # never use it as the fill-in tie-break. A genuinely sub-floor
        # camera's own weak candidate is real evidence and IS used.
        tiebreak = None if cam in d2_rejected_own_value else candidates[cam].hint_deg
        pred = predict_missing_hint(trusted_hints, geometry, cam, tiebreak_hint=tiebreak)
        if pred is None:
            # Distinguish, for the operator, "no candidates fit the
            # trusted anchors at all" from "a genuine mirror-ambiguity
            # tie with no trustworthy signal to break it" -- see this
            # module's own top docstring, "POST-SHIP FIX" section, and
            # `predict_missing_hint()`'s own docstring for why this
            # second case now refuses instead of silently picking one.
            fit_cands = candidate_predictions(trusted_hints, geometry, cam)
            if fit_cands:
                own_candidate_state = (
                    "D2-rejected as inconsistent" if cam in d2_rejected_own_value
                    else "absent this event"
                )
                tie_detail = (
                    f"{len(fit_cands)} candidate(s) fit equally well within "
                    f"tolerance and no trustworthy tie-break was available for "
                    f"cam{cam} (its own candidate is {own_candidate_state}) -- "
                    f"this is a genuine mirror ambiguity (the ring's gap sequence "
                    f"read forward vs backward both fit), not a small noise "
                    f"difference."
                )
            else:
                tie_detail = "no candidate fits the trusted anchor set's own gaps at all."
            return RigConsensusResult(
                ok=False,
                refusal_reason=(
                    f"rig-consensus orientation could not fill in cam{cam}: the "
                    f"trusted anchor set could not produce a prediction for it "
                    f"(trusted anchors: {sorted(trusted)}) -- {tie_detail} "
                    f"Refusing rather than guessing."
                ),
                rejected_cameras=rejected,
            )
        resolved[cam] = (pred, "rig_consensus")
        if cam not in rejected:
            rejected[cam] = (
                "no candidate this event" if tiebreak is None
                else f"below the {min_confidence:.2f}sigma confidence floor "
                     f"(own candidate {tiebreak:.2f}deg)"
            )

    return RigConsensusResult(ok=True, resolved=resolved, rejected_cameras=rejected)
