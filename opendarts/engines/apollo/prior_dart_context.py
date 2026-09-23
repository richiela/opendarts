"""Prior-dart-in-visit pixel context for Apollo's tip-detection
contamination guard -- see `tip_detection.py`'s module docstring,
"prior-dart line contamination" section, for the full real-incident
write-up (the recorded T15 throw, an old dart nudged by a new
one poisoning that camera's motion-diff blob).

**Why this is a separate module, not inlined in `capture_daemon.py` or
`replay.py`.** Per docs/DESIGN.md's "Replay is the source of truth", a live capture
and an offline replay of the SAME package must be able to reach the
EXACT SAME contamination decision -- so this lookup cannot live only in
`capture_daemon.py`'s live call path (which has no reason to ever be
invoked offline) or only in `replay.py` (which has no live trigger
context). Built once, imported by both:
`opendarts.live.capture_daemon.handle_ready_to_capture()` (live) and
`opendarts.capture.replay.replay_throw_with_engine()` (offline) call this
exact function, so a package captured live and later replayed sees
identical prior-dart context, not two independently-maintained copies
that could silently drift apart.

**What this deliberately does NOT do**: read any STORED per-camera tip
pixel from the prior throw's own `result.json` -- Apollo's own
`result.json` diagnostics never persisted per-camera tip pixels in the
first place (confirmed by reading `opendarts.engines.apollo.engine.
score_result_to_engine_result()`: only the triangulated 3D point and
per-ray distances are persisted, not per-camera pixel detections), and
even if it had, reading a STALE stored value would violate REPLAY's
"rerun the CURRENT pipeline" requirement -- a package captured before
today's `tip_detection.py` changed would report a tip pixel the CURRENT
code would no longer produce. Instead, this always recomputes the prior
throw's own `detect_tip()` fresh, from ITS stored raw images, through
whatever `tip_detection.py` is CURRENTLY installed -- exactly the same
"replay from raw inputs, not from a cached result" discipline every
other real number in this project follows.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from opendarts.engines.apollo.tip_detection import detect_tip

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

# {cam_index: (tip_px, far_end_px)} -- both ends of whatever the prior
# throw's own (freshly recomputed) detect_tip() call found for that
# camera. `tip_detection.detect_tip()`'s own `prior_dart_line_px`
# parameter (see that module) consumes exactly this shape.
PriorDartLinePx = dict[int, tuple[tuple[float, float], tuple[float, float]]]


@dataclass(frozen=True)
class CachedPriorThrowFrames:
    """A live capture loop's own in-memory record of the immediately-
    prior throw's own bg/dart frames -- added 2026-09-01, the "why call
    anything off disk at all when we have the prior dart in memory"
    finding (see docs/DESIGN.md's dated entry; measured 98.7ms disk vs
    32.1ms memory, byte-identical result).

    Passed to `find_prior_dart_line_px()` as `cached_frames` to skip its
    own disk round-trip (directory scan + `load_throw_package()`, real
    disk I/O + PNG decode) when the live capture loop already has the
    EXACT SAME raw pixels sitting in memory from processing that throw
    moments ago -- `run_capture_loop_body()` already builds `bg_images`/
    `current_frames` for every throw it captures (the same data
    `save_throw_package()` persists in the package's per-camera
    clip); this just carries that same, already-computed data forward
    one iteration instead of writing it to disk and immediately reading
    it back for the NEXT throw's own prior-dart lookup.

    Still runs `detect_tip()` FRESH on the cached pixels, exactly like
    the disk path -- REPLAY's "recompute via current code, never trust a
    stored VALUE" discipline is completely unaffected; only the SOURCE
    of the raw pixel bytes changed (memory instead of a disk round-trip
    to fetch bytes the process already had), never the computation. The
    offline replay caller (`opendarts.capture.replay.replay_throw_with_
    engine()`) never has a live loop's own in-memory state to draw on --
    it never passes this, so replay is structurally unaffected and
    always takes the exact same disk path it does today.

    `find_prior_dart_line_px()` itself validates `visit_id`/`visit_index`
    match what's actually being asked for BEFORE trusting this (see that
    function's own fast-path check) -- a caller passing a stale/
    mismatched cache (a turn boundary just rotated, or a fresh process
    with nothing cached yet) safely falls through to the existing
    disk-based lookup, never silently uses the wrong throw's frames.

    `precomputed_tip_line` (added same day, a THIRD, faster tier on top
    of the frame cache above, stacking directly on it): the prior throw's own ALREADY-COMPUTED (tip_px, far_end_px)
    per camera, straight from that throw's own real ApolloEngine.
    score() call (`EngineResult.diagnostics["own_tip_line_px"]`, see that
    engine's own `score()` docstring section for where this is built --
    the SAME raw, pre-ROI-gate detect_tip() call `find_prior_dart_line_
    px()`'s own disk/cache paths would otherwise redo). When present,
    `find_prior_dart_line_px()` returns it DIRECTLY -- skips `detect_
    tip()` entirely, not just the disk read. `None` (the default) falls
    through to the frame-cache tier above, exactly as before this field
    existed. Verified empirically, not assumed: a real throw's own
    `own_tip_line_px` is byte-identical to independently recomputing
    detect_tip() on that same throw's own saved bg/dart frames (see
    tests/test_engine_apollo_prior_dart_contamination.py) -- this
    field is a genuine cache of an identical computation, never a
    different, cheaper approximation of it."""

    visit_id: str
    visit_index: int
    bg_frames: "dict[int, np.ndarray]"
    dart_frames: "dict[int, np.ndarray]"
    precomputed_tip_line: PriorDartLinePx | None = None
    #: Where the prior throw's scoring WILL write its tip line, when that
    #: scoring runs on a background thread and has not necessarily finished
    #: when this record is built (the live default since 2026-09-06). Read at
    #: lookup time -- by then, a second or so later, it normally has. Holds
    #: the same value `precomputed_tip_line` would; an empty dict (scoring
    #: not finished, or no tip line) falls through to the frame cache.
    precomputed_tip_line_out: "dict[str, PriorDartLinePx] | None" = None

    def resolved_tip_line(self) -> "PriorDartLinePx | None":
        if self.precomputed_tip_line is not None:
            return self.precomputed_tip_line
        if self.precomputed_tip_line_out is not None:
            return self.precomputed_tip_line_out.get("own_tip_line_px")
        return None


def _tip_lines_from_frames(
    bg_frames: "dict[int, np.ndarray]", dart_frames: "dict[int, np.ndarray]"
) -> PriorDartLinePx | None:
    """Shared by both the cached (in-memory) and disk-loaded code paths
    in `find_prior_dart_line_px()` below -- the actual `detect_tip()`
    call and result-shape construction, factored out so the two paths
    can never independently drift on what "recompute the prior dart's
    line" actually means. Pure function, no I/O of its own."""
    result: PriorDartLinePx = {}
    for cam, bg in bg_frames.items():
        frame = dart_frames.get(cam)
        if frame is None:
            continue
        det = detect_tip(bg, frame)
        if det.ok and det.tip_px is not None and det.far_end_px is not None:
            result[cam] = (det.tip_px, det.far_end_px)
    return result or None


def engine_accepts_prior_dart_line_px(engine: object) -> bool:
    """True only if `engine.score()` actually declares a
    `prior_dart_line_px` parameter -- a duck-typed CAPABILITY check
    (mirrors `opendarts.engines.base.engine_has_custom_calibrate()`'s own
    `hasattr`-based pattern for the same reason: a registered engine
    NAME is not proof of which concrete class is actually behind it).
    `opendarts.live.capture_daemon.handle_ready_to_capture()` and
    `opendarts.capture.replay.replay_throw_with_engine()` both special-case
    on `primary_name == DEFAULT_PRIMARY_ENGINE` ("Apollo") to decide
    WHETHER to look up prior-dart context at all -- cheap, name-based,
    and fine for that -- but must NOT assume the object `get_engine()`
    actually returns for that name is the real `ApolloEngine` class
    that added this parameter: several real tests substitute a stub/fake
    engine registered under the same name (`monkeypatch.setattr(...,
    "get_engine", lambda name: fake_engine)`) whose own `score()` keeps
    the pre-2026-08-16 3-argument signature on purpose. Calling THIS
    check first, and only passing `prior_dart_line_px=...` when it's
    True, keeps every such test double working unmodified rather than
    requiring every test-double engine in this codebase to grow a
    parameter it has no use for.
    """
    import inspect

    score = getattr(engine, "score", None)
    if score is None or not callable(score):
        return False
    try:
        params = inspect.signature(score).parameters
    except (TypeError, ValueError):
        return False
    return "prior_dart_line_px" in params


def find_prior_dart_line_px(
    session_dir: Path,
    visit_id: str | None,
    visit_index: int | None,
    *,
    cached_frames: "CachedPriorThrowFrames | None" = None,
) -> PriorDartLinePx | None:
    """Locate the immediately-prior throw of the SAME visit
    (`visit_index - 1`) already saved under `session_dir`, and return
    that throw's own per-camera `(tip_px, far_end_px)` -- recomputed
    fresh via `detect_tip()` on ITS stored raw images (see module
    docstring for why this is never read from a stored result).

    Returns `None` whenever there is nothing usable to return -- no
    visit_id/visit_index at all (packages predating the visit model),
    `visit_index <= 0` (this IS the first dart of the visit, nothing
    prior to compare against), the prior throw's own directory can't be
    found, or its own detection fails -- matching this project's
    "absent, never malformed" discipline for optional per-throw context.
    Never raises.

    `cached_frames` (added 2026-09-01, see `CachedPriorThrowFrames`'s own
    docstring for the full "why call anything off disk at all" writeup):
    when given AND its `visit_id`/`visit_index` actually match what's
    being asked for here (`target_index`, computed below), skips the
    disk scan/load entirely and recomputes `detect_tip()` directly on
    the cached in-memory pixels instead -- same computation, faster
    source. A `None`/mismatched cache (the default, and every caller
    that doesn't have a live loop's own in-memory state -- offline
    replay in particular) falls through to the existing disk-based
    lookup, byte-for-byte unchanged from before this parameter existed.
    """
    if visit_id is None or visit_index is None or visit_index <= 0:
        return None

    target_index = visit_index - 1
    if (
        cached_frames is not None
        and cached_frames.visit_id == visit_id
        and cached_frames.visit_index == target_index
    ):
        precomputed = cached_frames.resolved_tip_line()
        if precomputed is not None:
            # Fastest tier: the prior throw's own real detect_tip() call
            # already produced this exact value moments ago -- return it
            # directly, skip recomputing anything at all.
            return precomputed or None
        return _tip_lines_from_frames(cached_frames.bg_frames, cached_frames.dart_frames)

    if not session_dir.exists():
        return None
    # Most-recent-first: visits are sequential and short (<=
    # MAX_DARTS_PER_TURN darts, see opendarts.capture.throw_trigger), so the
    # prior throw of THIS visit -- if it exists at all -- is always among
    # the most recently saved throw directories. Scanning newest-first
    # finds it in O(1) in the common case rather than walking the whole
    # session; throw directory names are zero-padded sequential throw
    # numbers (see handle_ready_to_capture()'s `throw_id` construction),
    # so lexicographic sort order is chronological order.
    try:
        throw_dirs = sorted(
            (p for p in session_dir.iterdir() if p.is_dir()), reverse=True
        )
    except OSError:
        return None

    for throw_dir in throw_dirs:
        meta_path = throw_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        if meta.get("visit_id") != visit_id or meta.get("visit_index") != target_index:
            continue
        # Local import: opendarts.capture.throw_package -> ... has no
        # import of opendarts.engines.apollo anywhere, so this is not a
        # real cycle -- deferred purely so importing this small module
        # doesn't always pull in the full capture/package machinery for
        # callers (e.g. a unit test constructing prior_dart_line_px by
        # hand) that never call this function.
        from opendarts.capture.throw_package import load_throw_package

        try:
            package = load_throw_package(throw_dir)
        except Exception:
            log.warning(
                "found prior throw dir %s for visit %s/%d but failed to load "
                "its package -- returning no prior-dart context rather than "
                "raising (this must never break the CURRENT throw's own "
                "scoring)",
                throw_dir, visit_id, target_index,
            )
            return None

        return _tip_lines_from_frames(package.bg_frames, package.dart_frames)

    return None
