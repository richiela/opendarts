"""RIG-CONSENSUS ORIENTATION -- absolute-hint reseed detector.

Built 2026-08-31, a real, confirmed gap in `opendarts/calibration/
rig_ring_geometry.py` (see that module's own top docstring for the full
rig-consensus design this sits alongside): `RingGeometry`'s own drift
check (`RING_GEOMETRY_DRIFT_THRESHOLD_DEG`, `update_ring_geometry()`)
only ever compares the *relative* gaps between the three cameras' own
orientation hints -- a real physical camera reposition can shift EVERY
camera's *absolute* hint by 1-3 degrees while barely moving the gaps
between them (a repositioning that doesn't much rotate the cameras
relative to each other), which never trips that check. Confirmed on the
live rig the same night this module was built, real numbers:

```
                  before move      after move       shift
cam0 hint_deg:    ~272.6-273.4deg  ~269.9-270.2deg  ~3deg
cam1 hint_deg:    ~14.8-15.0deg    ~15.84-15.96deg  ~1deg
cam2 hint_deg:    ~165.9-166.3deg  ~164.5-164.7deg  ~1.3-1.4deg

learned gaps (relative spacing) before: [151.09, 106.89, 102.02]
learned gaps (relative spacing) after:  [150.65, 106.91, 102.44]  (~0.44deg max)
```

Left uncaught, `resolve_rig_consensus_orientation()` keeps trusting
stale learned geometry indefinitely and can silently predict a camera's
orientation 3-4deg off its own correct live reading -- exactly the
"confidently wrong, not visibly wrong" failure class this whole line of
work (see docs/DESIGN.md's 2026-08-29 dated entries) exists to eliminate. The
existing recovery mechanism already works -- a human manually deleting
`ring_geometry_fallback.json` lets the rig relearn cleanly within about
10 events, proven live -- what's missing is detecting the need for that
deletion automatically instead of waiting for a human to notice.

THIS MODULE, NOT `rig_ring_geometry.py` ITSELF -- deliberately kept as a
separate, additive, PARALLEL signal, never touching
`RING_GEOMETRY_DRIFT_THRESHOLD_DEG` or `update_ring_geometry()` (per this
task's own explicit scope). Tracks each camera's own ABSOLUTE orientation
hint over time (not the gaps between cameras) and recognizes a real,
PERSISTENT, RIG-WIDE shift -- as opposed to ordinary per-event detection
noise, or one camera having a genuine but temporary bad stretch -- then
triggers an automatic reseed: clearing both this module's own learned
baseline AND `rig_ring_geometry.py`'s own persisted `RingGeometry`
(exactly what a human manually deleting `ring_geometry_fallback.json`
already does today, proven to recover correctly).

WHY THREE SEPARATE REQUIREMENTS MATTER, EACH WITH A REAL FAILURE MODE
IT PREVENTS:

1. **Noise floor.** This rig's own real per-camera absolute-hint noise,
   BEFORE tonight's move, was stable to within roughly 0.1-0.6deg across
   dozens of real events on a given night (see this module's own test
   suite for the synthetic reconstruction of this). A detector that
   fires on ordinary noise would be worse than the status quo --
   thrashing away a perfectly good learned geometry on every noisy
   reading.
2. **Persistence across multiple events, not a single reading.**
   Tonight's real move held STEADY across an entire 10-event batch, not
   a one-off blip -- `CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED` requires
   the SAME rig-wide signal to repeat across several consecutive
   calibration events before acting, so one noisy reading (even a
   genuinely large one) can never trigger a reseed on its own.
3. **Agreement across most/all cameras, not just one.** A single
   camera's own detection quality can genuinely have a real but
   temporary bad stretch (lighting, a stray occlusion, whatever) --
   this rig's own documented history shows exactly this for its
   historically weakest camera. That is a per-camera detection-quality
   problem, not evidence the RIG moved. `MIN_CAMERAS_DEVIATING` (derived
   per-event as `max(2, n_cameras - 1)`, i.e. "all but at most one")
   means a single camera drifting alone -- however persistently -- can
   NEVER by itself accumulate toward a reseed, because a reseed can only
   even start counting on an event where MOST cameras deviate together.

THE THRESHOLD, real numbers, not a round guess. Noise ceiling ~0.6deg;
smallest real per-camera shift observed tonight ~1.0deg (cam1).
`ABSOLUTE_HINT_DEVIATION_THRESHOLD_DEG = 0.8` sits at the midpoint of
that real 0.6-1.0deg gap -- ~33% margin above the noise ceiling, ~20%
margin below the smallest real shift. This margin is real but tighter
than most other measured thresholds in this project's history (which
typically enjoy an order-of-magnitude gap between "noise" and "signal"
-- see e.g. `rig_ring_geometry.RING_GEOMETRY_DRIFT_THRESHOLD_DEG`'s own
~3.2x margin) -- stated plainly, not hidden, because the real physical
quantities involved here (per-camera absolute noise vs. cam1's
comparatively small real shift) just don't leave more room than that.
`CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED` is exactly what makes this
tighter per-event margin still safe: 3 independent per-camera-pair
coincidences (>= 2-of-3 cameras crossing an already-tight-but-real
threshold, in the SAME direction of "this camera's own baseline",
repeated 3 times running) is a vanishingly unlikely false-positive
combination even with a tighter single-event margin than usual, while
still being fast enough (3 calibration events, not 10) to catch a real
move well before an operator would otherwise notice via degraded
scoring accuracy.

WHY THE BASELINE FREEZES WHILE A DEVIATION STREAK ACCUMULATES. A naive
design that kept updating each camera's own rolling-average baseline on
EVERY event (including ones that are themselves part of a developing
real move) would have the baseline "chase" the move -- by the time 3
events had passed, the baseline itself would have partly absorbed the
shift, weakening the very signal being measured. Instead,
`record_absolute_orientation_event()` freezes `baselines`/
`stable_samples` UNCHANGED for the whole duration a deviation streak is
accumulating (`consecutive_deviating_events` still below the
requirement) -- the comparison for event N+1 of a streak is still made
against the SAME pre-move baseline event 1 was compared against, not a
partially-shifted one. Only a genuinely NON-deviating event (fewer than
`min_cameras_deviating` cameras crossing the threshold that event) is
allowed to update the rolling baseline -- this is also exactly what
naturally absorbs a single-camera-only temporary struggle without ever
letting it corrupt the OTHER cameras' own baselines (each camera's
baseline only updates from its OWN sample stream; a struggling camera
skews only its own eventual baseline, not the other two's).

CIRCULAR-MEAN, NOT PLAIN ARITHMETIC MEAN, for the rolling baseline --
this project has already independently hit and fixed a genuine
circular-mean-across-0deg/360deg-wraparound bug once before, on
2026-08-20, in the session-level orientation aggregation. A camera
whose true hint sits near 0deg/360deg (not the case for any of THIS
rig's three cameras today, but not guaranteed to stay that way forever,
and this module is meant to generalize) would have a plain arithmetic
mean of its own sample window badly wrong across that wraparound.
`_circular_mean_deg()` here uses the same atan2-of-summed-unit-vectors
technique to avoid repeating that exact class of bug.

STORAGE -- same "small, self-correcting, JSON-backed last-known-good
value" convention this project's whole calibration-fallback family
already uses (`focal_length_fallback.json`, `distortion_fallback.json`,
`principal_point_fallback.json`, `ring_geometry_fallback.json` -- see
`opendarts/calibration/focal_length.py`'s own docstring for the precedent
this module's load/save/schema/degrade-safely shape is deliberately
copied from, not reinvented). `absolute_orientation_reseed_fallback.
json`, same `calibration_package_root` directory as its siblings, schema
`absolute-orientation-reseed-v1`. Kept in its OWN file, not merged into
`ring_geometry_fallback.json` -- a genuinely different concern (absolute
per-camera history vs. relative inter-camera gaps), matching this
project's own established "one file per independently-evolving concern"
precedent (`principal_point_fallback.json` kept separate from
`distortion_fallback.json` for the identical reason).

LOUD, NEVER SILENT, PER THIS TASK'S OWN REQUIREMENT 3. A confirmed
reseed logs a WARNING naming exactly what fired (the per-camera baseline
BEFORE vs. the triggering event's own observed hints AFTER, the real
deviation magnitudes, and which cameras contributed) -- see
`record_absolute_orientation_event()`'s own reseed branch. This is
NEVER a refusal of the calibration event that detects it -- the event
whose own hints triggered the reseed still completes normally with its
own already-derived values; the reseed only affects what the NEXT event
reads back from disk (a fresh, empty `RingGeometry` and a fresh,
single-sample-seeded `AbsoluteOrientationHistory`).

NOT WIRED INTO ANY LIVE CALL SITE. Per this task's own explicit scope,
this module is a built-and-validated CAPABILITY, not an active part of
the live pipeline -- `check_for_rig_move_and_reseed_if_confirmed()` is
the intended single entry point a future caller would use, but nothing
in `opendarts/live/capture_daemon.py` calls it. See this module's own
"RECOMMENDED WIRING (not applied)" section at the bottom of this
docstring for where and how the orchestrating session should consider
wiring it in.

RECOMMENDED WIRING (not applied) -- `opendarts.live.capture_daemon.
_bootstrap_calibrations_unlocked()` already computes a `hints_for_
geometry: dict[int, float]` (the same per-camera resolved-hint dict fed
to `rig_ring_geometry.update_ring_geometry()`, at all of its 3 real call
sites in that file) immediately before each `update_ring_geometry()`
call. The natural wiring point is calling `check_for_rig_move_and_
reseed_if_confirmed(calibration_package_root, hints_for_geometry,
now_utc)` at each of those same 3 sites, right alongside (not instead
of) the existing `update_ring_geometry()`/`save_ring_geometry()` calls
-- this module's own reseed action (clearing `ring_geometry_fallback.
json`) would then simply mean the VERY NEXT `load_ring_geometry()` call
in that same function (which already always runs first) sees `None` and
correctly falls back to Mode B for the following event, no other
plumbing required. This is a recommendation for the orchestrating
session to review, not something this task applies itself, per this
task's own explicit instructions.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opendarts.calibration.rig_ring_geometry import RING_GEOMETRY_FALLBACK_FILENAME

log = logging.getLogger(__name__)

RESEED_DETECTOR_FALLBACK_FILENAME = "absolute_orientation_reseed_fallback.json"
SCHEMA = "absolute-orientation-reseed-v1"

# Capped recent-sample window per camera for the rolling baseline mean --
# mirrors rig_ring_geometry.RING_GEOMETRY_MAX_SAMPLES's own "staleness
# bounded to the last N updates" property, for the identical reason (a
# slow genuine physical drift shouldn't vanish into an ever-larger
# lifetime denominator).
ABSOLUTE_HINT_HISTORY_MAX_SAMPLES = 50

# See this module's own top docstring, "THE THRESHOLD" section, for the
# real noise-ceiling (~0.6deg) vs. smallest-real-shift (~1.0deg) numbers
# this midpoint is set from.
ABSOLUTE_HINT_DEVIATION_THRESHOLD_DEG = 0.8

# See this module's own top docstring, "THE THRESHOLD" section, for why
# 3 (not 1, not 10) is the real, justified choice.
CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED = 3


def _circ_diff(a: float, b: float) -> float:
    """Smallest angular distance between two directions, degrees,
    always >= 0. Same definition as rig_ring_geometry._circ_diff --
    duplicated (not imported) since that helper is private to that
    module and this is a small, self-contained, one-line piece of
    math, not a real shared dependency worth coupling the two modules
    over."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _circular_mean_deg(values: list[float]) -> float:
    """Circular mean of `values` (degrees), robust to 0deg/360deg
    wraparound -- see this module's own top docstring, "CIRCULAR-MEAN"
    section, for why this project specifically does not use a plain
    arithmetic mean here."""
    sin_sum = sum(math.sin(math.radians(v)) for v in values)
    cos_sum = sum(math.cos(math.radians(v)) for v in values)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0


@dataclass
class AbsoluteOrientationHistory:
    """Learned per-camera absolute-orientation-hint history for THIS
    rig. `baselines[cam]` is the camera's own current "typical value"
    (a circular mean over `stable_samples[cam]`, its own capped rolling
    window of raw hints from events NOT currently part of an
    accumulating deviation streak -- see this module's own top
    docstring, "WHY THE BASELINE FREEZES" section). `consecutive_
    deviating_events` is the live streak counter toward a reseed."""
    baselines: dict[int, float]
    stable_samples: dict[int, list[float]]
    consecutive_deviating_events: int
    n_events: int
    first_learned_utc: str
    last_updated_utc: str


@dataclass
class ReseedDecision:
    """Result of `record_absolute_orientation_event()`. `reseed_
    triggered=True` means a persistent, rig-wide absolute-orientation
    shift was just confirmed and `new_history` has ALREADY been reset
    to a fresh, single-event-seeded state (this event's own observed
    hints become the new baseline) -- the caller's job is only to
    persist `new_history` and (if it wants the automatic-reseed
    behaviour, see `check_for_rig_move_and_reseed_if_confirmed()`) also
    clear the sibling `ring_geometry_fallback.json`. `deviations_deg`/
    `cameras_deviating` describe THIS event's own comparison against the
    baseline that was in effect going into it (useful for logging
    regardless of whether a reseed fired)."""
    new_history: AbsoluteOrientationHistory
    reseed_triggered: bool
    reseed_reason: str | None = None
    deviations_deg: dict[int, float] = field(default_factory=dict)
    cameras_deviating: list[int] = field(default_factory=list)


def record_absolute_orientation_event(
    existing: AbsoluteOrientationHistory | None,
    hints: dict[int, float],
    now_utc: str,
    *,
    max_samples: int = ABSOLUTE_HINT_HISTORY_MAX_SAMPLES,
    deviation_threshold_deg: float = ABSOLUTE_HINT_DEVIATION_THRESHOLD_DEG,
    consecutive_events_required: int = CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED,
    min_cameras_deviating: int | None = None,
) -> ReseedDecision:
    """Pure function: given the existing learned history (or `None` on a
    virgin rig) and one calibration event's own resolved per-camera
    absolute orientation hints, decide whether this event continues,
    breaks, or completes a persistent rig-wide deviation streak.

    `min_cameras_deviating` defaults to `max(2, n_cameras - 1)` (i.e.
    "all but at most one" camera) -- see this module's own top
    docstring, requirement 3, for why this is deliberately NOT "any one
    camera." A camera count below 2 can never have a meaningful
    "most cameras agree" signal at all (mirrors `rig_ring_geometry`'s
    own `len(frames_by_cam) >= 2` guard for the identical reason) --
    such an event is always treated as a normal/stable update, never
    contributing to a reseed streak.

    Every branch below is a real, distinct case, matching this task's
    own required test scenarios:

    - `existing is None`: virgin history -- seed baselines directly from
      this event's own hints, no comparison possible yet (nothing to
      have drifted FROM).
    - A camera in `hints` with no existing baseline (new to this rig's
      history, e.g. a camera that was down for prior events): simply
      excluded from this event's own deviation computation -- there is
      nothing to compare it against yet.
    - `len(cameras_deviating) >= min_cameras_deviating` this event
      (a real rig-wide-shift SIGNAL, not proof by itself): increments
      the streak. Reaching `consecutive_events_required` on this
      increment triggers a reseed. Below that, the streak grows but
      `baselines`/`stable_samples` stay FROZEN (see top docstring).
    - Otherwise (fewer than `min_cameras_deviating` cameras deviating --
      covers both true noise-floor stability AND a single camera's own
      isolated struggle): the streak resets to 0 and every present
      camera's own rolling baseline updates normally.
    """
    n = len(hints)
    if min_cameras_deviating is None:
        min_cameras_deviating = max(2, n - 1) if n >= 2 else n + 1  # unreachable floor for n<2

    if existing is None:
        stable = {cam: [v] for cam, v in hints.items()}
        baselines = dict(hints)
        return ReseedDecision(
            new_history=AbsoluteOrientationHistory(
                baselines=baselines,
                stable_samples=stable,
                consecutive_deviating_events=0,
                n_events=1,
                first_learned_utc=now_utc,
                last_updated_utc=now_utc,
            ),
            reseed_triggered=False,
        )

    deviations: dict[int, float] = {}
    for cam, v in hints.items():
        base = existing.baselines.get(cam)
        if base is None:
            continue
        deviations[cam] = _circ_diff(v, base)

    cameras_deviating = sorted(
        cam for cam, d in deviations.items() if d > deviation_threshold_deg
    )
    is_rig_wide_deviation_event = (
        n >= 2
        and len(deviations) >= min_cameras_deviating
        and len(cameras_deviating) >= min_cameras_deviating
    )

    if is_rig_wide_deviation_event:
        new_streak = existing.consecutive_deviating_events + 1
        if new_streak >= consecutive_events_required:
            detail = ", ".join(
                f"cam{cam}: baseline {existing.baselines[cam]:.2f}deg -> "
                f"observed {hints[cam]:.2f}deg (delta {deviations[cam]:.2f}deg)"
                for cam in sorted(hints) if cam in deviations
            )
            reason = (
                f"automatic rig-move reseed triggered: {len(cameras_deviating)}/{n} "
                f"camera(s) showed a >{deviation_threshold_deg:.2f}deg absolute-"
                f"orientation deviation from their own learned baseline for "
                f"{new_streak} consecutive calibration events running -- "
                f"{detail}. Treating this as a real physical rig change (not "
                f"per-event noise, not a single camera's own detection "
                f"struggle) and reseeding: this event's own observed hints "
                f"become the new baseline, and the sibling ring geometry "
                f"(rig_ring_geometry.RingGeometry) is being cleared too so "
                f"it relearns from scratch starting with the next event."
            )
            log.warning("%s", reason)
            stable = {cam: [v] for cam, v in hints.items()}
            baselines = dict(hints)
            return ReseedDecision(
                new_history=AbsoluteOrientationHistory(
                    baselines=baselines,
                    stable_samples=stable,
                    consecutive_deviating_events=0,
                    n_events=1,
                    first_learned_utc=now_utc,
                    last_updated_utc=now_utc,
                ),
                reseed_triggered=True,
                reseed_reason=reason,
                deviations_deg=deviations,
                cameras_deviating=cameras_deviating,
            )
        log.info(
            "possible rig move: %d/%d camera(s) deviated >%.2fdeg from their own "
            "baseline this event (streak %d/%d, not yet reseeding) -- %s",
            len(cameras_deviating), n, deviation_threshold_deg,
            new_streak, consecutive_events_required,
            {cam: round(d, 2) for cam, d in deviations.items()},
        )
        return ReseedDecision(
            new_history=AbsoluteOrientationHistory(
                baselines=existing.baselines,
                stable_samples=existing.stable_samples,
                consecutive_deviating_events=new_streak,
                n_events=existing.n_events + 1,
                first_learned_utc=existing.first_learned_utc,
                last_updated_utc=now_utc,
            ),
            reseed_triggered=False,
            deviations_deg=deviations,
            cameras_deviating=cameras_deviating,
        )

    # Normal/stable event -- including a single camera's own isolated
    # deviation, which never meets min_cameras_deviating alone. Streak
    # resets; every present camera's own rolling baseline updates.
    new_stable = dict(existing.stable_samples)
    new_baselines = dict(existing.baselines)
    for cam, v in hints.items():
        samples = list(existing.stable_samples.get(cam, []))
        samples.append(v)
        if len(samples) > max_samples:
            samples = samples[-max_samples:]
        new_stable[cam] = samples
        new_baselines[cam] = _circular_mean_deg(samples)

    return ReseedDecision(
        new_history=AbsoluteOrientationHistory(
            baselines=new_baselines,
            stable_samples=new_stable,
            consecutive_deviating_events=0,
            n_events=existing.n_events + 1,
            first_learned_utc=existing.first_learned_utc,
            last_updated_utc=now_utc,
        ),
        reseed_triggered=False,
        deviations_deg=deviations,
        cameras_deviating=cameras_deviating,
    )


# ---------------------------------------------------------------------
# Persistence -- same "small, self-correcting, JSON-backed last-known-
# good value" convention as rig_ring_geometry.py/focal_length.py/etc.
# (see this module's own top docstring). Deliberately its own file, not
# merged into ring_geometry_fallback.json -- see top docstring.
# ---------------------------------------------------------------------

def load_absolute_orientation_history(
    calibration_package_root: Path | None,
) -> AbsoluteOrientationHistory | None:
    """Load persisted history from `<calibration_package_root>/
    absolute_orientation_reseed_fallback.json`, or `None` if the root is
    `None`, the file doesn't exist, doesn't parse, or doesn't match this
    module's current `SCHEMA` -- same absent/corrupt/stale-schema-all-
    degrade-safely posture every sibling fallback file in this project
    already uses. `None` is the "no history known yet" signal -- the
    next `record_absolute_orientation_event()` call seeds fresh."""
    if calibration_package_root is None:
        return None
    path = Path(calibration_package_root) / RESEED_DETECTOR_FALLBACK_FILENAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        log.exception(
            "%s: failed to read/parse -- treating as absent (no absolute-"
            "orientation history known yet).", path,
        )
        return None
    if payload.get("schema") != SCHEMA:
        log.warning(
            "%s: schema %r does not match current %r -- treating as absent",
            path, payload.get("schema"), SCHEMA,
        )
        return None
    try:
        baselines = {int(k): float(v) for k, v in payload["baselines"].items()}
        stable_samples = {
            int(k): [float(x) for x in v]
            for k, v in payload["stable_samples"].items()
        }
        consecutive_deviating_events = int(payload["consecutive_deviating_events"])
        n_events = int(payload["n_events"])
        first_learned_utc = str(payload["first_learned_utc"])
        last_updated_utc = str(payload["last_updated_utc"])
    except (KeyError, TypeError, ValueError, AttributeError):
        log.exception("%s: malformed contents -- treating as absent.", path)
        return None
    if set(baselines) != set(stable_samples):
        log.warning(
            "%s: baselines/stable_samples camera-key mismatch -- treating as absent.",
            path,
        )
        return None
    return AbsoluteOrientationHistory(
        baselines=baselines,
        stable_samples=stable_samples,
        consecutive_deviating_events=consecutive_deviating_events,
        n_events=n_events,
        first_learned_utc=first_learned_utc,
        last_updated_utc=last_updated_utc,
    )


def save_absolute_orientation_history(
    calibration_package_root: Path, history: AbsoluteOrientationHistory,
) -> None:
    """Overwrite the persisted history file -- plain `path.write_text(
    json.dumps(...))`, matching every sibling fallback file's own
    convention (a torn write from a crash mid-write degrades to
    "absent/corrupt", which `load_absolute_orientation_history()`
    already treats as a safe, loud, non-fatal fallback state)."""
    calibration_package_root = Path(calibration_package_root)
    calibration_package_root.mkdir(parents=True, exist_ok=True)
    path = calibration_package_root / RESEED_DETECTOR_FALLBACK_FILENAME
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "baselines": {str(k): v for k, v in history.baselines.items()},
        "stable_samples": {str(k): v for k, v in history.stable_samples.items()},
        "consecutive_deviating_events": history.consecutive_deviating_events,
        "n_events": history.n_events,
        "first_learned_utc": history.first_learned_utc,
        "last_updated_utc": history.last_updated_utc,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def clear_ring_geometry_on_reseed(calibration_package_root: Path) -> bool:
    """Delete `rig_ring_geometry.py`'s own `ring_geometry_fallback.json`
    -- the second half of an automatic reseed (see this module's own top
    docstring, requirement 2): exactly reproduces the same manual
    recovery action a human operator already performs today (proven to
    recover cleanly within about 10 events on the real rig -- see
    docs/DESIGN.md's 2026-08-29 dated "MERGED to main" entry). Returns `True`
    if a file was actually present and removed, `False` if there was
    nothing to clear (idempotent -- safe to call even when no geometry
    was ever learned, e.g. a virgin rig).

    Deliberately does NOT import or call anything from `rig_ring_
    geometry.py` beyond its own `RING_GEOMETRY_FALLBACK_FILENAME`
    constant -- this function does not touch, and is not touched by,
    `update_ring_geometry()`'s own relative-gap drift logic, per this
    task's own explicit scope."""
    path = Path(calibration_package_root) / RING_GEOMETRY_FALLBACK_FILENAME
    if path.exists():
        path.unlink()
        log.warning(
            "%s: deleted as part of an automatic rig-move reseed -- ring "
            "geometry will relearn from scratch starting with the next "
            "calibration event.", path,
        )
        return True
    return False


def check_for_rig_move_and_reseed_if_confirmed(
    calibration_package_root: Path | None,
    hints: dict[int, float],
    now_utc: str,
    *,
    max_samples: int = ABSOLUTE_HINT_HISTORY_MAX_SAMPLES,
    deviation_threshold_deg: float = ABSOLUTE_HINT_DEVIATION_THRESHOLD_DEG,
    consecutive_events_required: int = CONSECUTIVE_EVENTS_REQUIRED_FOR_RESEED,
    min_cameras_deviating: int | None = None,
) -> ReseedDecision:
    """Convenience end-to-end orchestration for a future caller (see
    this module's own top docstring, "RECOMMENDED WIRING" section --
    NOT called from anywhere in the live pipeline by this task): loads
    persisted history, runs the pure decision function, persists the
    result, and -- ONLY when a reseed is actually confirmed -- also
    clears the sibling `ring_geometry_fallback.json` so BOTH pieces of
    learned state reset together in one action.

    `calibration_package_root=None` runs the decision on an ephemeral
    (never loaded, never persisted) state -- useful for a caller/test
    that wants to exercise the pure logic without a real filesystem
    root; there is nothing to reseed on disk in that case, and nothing
    persists for a future call to build on either."""
    if calibration_package_root is None:
        return record_absolute_orientation_event(
            None, hints, now_utc,
            max_samples=max_samples,
            deviation_threshold_deg=deviation_threshold_deg,
            consecutive_events_required=consecutive_events_required,
            min_cameras_deviating=min_cameras_deviating,
        )
    existing = load_absolute_orientation_history(calibration_package_root)
    decision = record_absolute_orientation_event(
        existing, hints, now_utc,
        max_samples=max_samples,
        deviation_threshold_deg=deviation_threshold_deg,
        consecutive_events_required=consecutive_events_required,
        min_cameras_deviating=min_cameras_deviating,
    )
    save_absolute_orientation_history(calibration_package_root, decision.new_history)
    if decision.reseed_triggered:
        clear_ring_geometry_on_reseed(calibration_package_root)
    return decision
