"""Prior-dart-in-visit contamination guard (added 2026-08-16) -- see
`opendarts/engines/apollo/tip_detection.py`'s dated module docstring
entry and `opendarts/engines/apollo/engine.py`'s `ApolloEngine.score()`
own dated comment for the full real-incident write-up this guards
against (the recorded T15 throw: an old dart from an earlier
throw of the same visit got nudged by a new dart's impact, poisoning
that camera's motion-diff blob into a merged component that confidently
reported the wrong end as the new dart's tip).

Two real corpus throws anchor this file's end-to-end coverage (both
skip cleanly, per this project's living-corpus discipline, if not
present on the machine running these tests):

- `KNOWN_CONTAMINATED_THROW` -- the real incident. Must end up excluding
  the contaminated camera (cam2) and triangulating from the remaining
  two.
- `KNOWN_FALSE_POSITIVE_AVOIDED_THROW` -- a real throw found DURING this
  guard's own corpus validation where the prior-dart-line signal ALSO
  fires (a legitimate close-grouped dart, not contamination) but must
  NOT end up excluding a camera, because doing so would turn an
  already-correct throw wrong (see `ApolloEngine.score()`'s own dated
  comment for the full story -- this is the real "does NOT exclude a
  normal 2nd/3rd-dart-in-visit camera reading" case this project's task
  instructions ask for, not a synthetic stand-in).
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from opendarts.capture.replay import replay_throw_with_engine
from opendarts.capture.throw_package import load_throw_package, save_throw_package
from opendarts.engines.apollo.engine import ApolloEngine
from opendarts.engines.apollo.prior_dart_context import (
    CachedPriorThrowFrames,
    engine_accepts_prior_dart_line_px,
    find_prior_dart_line_px,
)
from opendarts.engines.apollo.tip_detection import _perp_dist_to_line, detect_tip
from opendarts.pipeline import ScoreResult

REPO_ROOT = Path(__file__).resolve().parent.parent
KNOWN_CONTAMINATED_THROW = "20260816-171712/20260816-171712-065-T15"
KNOWN_FALSE_POSITIVE_AVOIDED_THROW = "20260816-112211/20260816-112211-056-D2"


def _corpus_root() -> Path:
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    return Path(env_root) if env_root else REPO_ROOT / "data" / "archive" / "clean"


# --------------------------------------------------------------------------
# Synthetic, isolated: _perp_dist_to_line's own geometry.
# --------------------------------------------------------------------------


def test_perp_dist_to_line_zero_for_a_point_on_the_line():
    assert _perp_dist_to_line((5.0, 5.0), (0.0, 0.0), (10.0, 10.0)) == pytest.approx(0.0, abs=1e-6)


def test_perp_dist_to_line_matches_hand_computed_offset():
    # Horizontal line y=0 from x=0 to x=10; point (5, 3) is 3.0px away,
    # independent of where along the line it's measured (infinite line,
    # not segment -- see the function's own docstring).
    assert _perp_dist_to_line((5.0, 3.0), (0.0, 0.0), (10.0, 0.0)) == pytest.approx(3.0)
    # Also true for a point beyond the segment's own extent (x=20) --
    # this is the infinite-line behavior the contamination guard relies
    # on (a merged/contaminated component's own span typically extends
    # PAST the prior dart's own ends, see PRIOR_DART_LINE_MAX_PERP_PX's
    # comment).
    assert _perp_dist_to_line((20.0, 3.0), (0.0, 0.0), (10.0, 0.0)) == pytest.approx(3.0)


def test_perp_dist_to_line_degenerate_zero_length_line_falls_back_to_point_distance():
    assert _perp_dist_to_line((3.0, 4.0), (0.0, 0.0), (0.0, 0.0)) == pytest.approx(5.0)


# --------------------------------------------------------------------------
# Synthetic, isolated: detect_tip()'s new diagnostic-only flag. Reuses
# the exact same synthetic dart shape (wide "fletching" circle + narrow
# tip triangle) the tip detector's own synthetic tests validate the
# tip/far-end pick on -- true tip lands near
# (400, 500), far end (fletching) near (400, 150).
# --------------------------------------------------------------------------


def _synthetic_shaft_frame():
    h, w = 800, 800
    bg = np.full((h, w, 3), 40, dtype=np.uint8)
    frame = bg.copy()
    cv2.circle(frame, (400, 150), 30, (200, 200, 200), -1)
    pts = np.array([[380, 180], [420, 180], [400, 500]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (200, 200, 200))
    return bg, frame


def test_prior_dart_line_px_none_leaves_diagnostics_key_absent():
    bg, frame = _synthetic_shaft_frame()
    result = detect_tip(bg, frame) # no prior_dart_line_px at all
    assert result.ok
    assert "prior_dart_contamination_suspected" not in result.diagnostics


def test_prior_dart_line_close_to_both_ends_sets_suspected_true():
    bg, frame = _synthetic_shaft_frame()
    # A "prior dart line" running almost exactly along the same synthetic
    # shaft (tip ~(400,500), far end ~(400,150)) -- both of THIS
    # detection's own ends should land essentially on it.
    result = detect_tip(bg, frame, prior_dart_line_px=((401.0, 495.0), (399.0, 155.0)))
    assert result.ok
    assert result.diagnostics["prior_dart_contamination_suspected"] is True
    assert result.diagnostics["prior_dart_contamination_perp_tip_px"] < 15.0
    assert result.diagnostics["prior_dart_contamination_perp_far_px"] < 15.0
    # tip_px/ok/reason are completely unaffected by this signal -- see
    # the module docstring's dated entry for why this module itself
    # never rejects on it.
    assert result.tip_px is not None


def test_prior_dart_line_far_from_both_ends_sets_suspected_false():
    bg, frame = _synthetic_shaft_frame()
    # A "prior dart line" running horizontally, far from the synthetic
    # shaft's own (vertical, x~400) line.
    result = detect_tip(bg, frame, prior_dart_line_px=((0.0, 700.0), (100.0, 700.0)))
    assert result.ok
    assert result.diagnostics["prior_dart_contamination_suspected"] is False


def test_prior_dart_line_close_to_only_one_end_does_not_suspect():
    bg, frame = _synthetic_shaft_frame()
    # A line that passes right through the TIP end (400, 500) but is far
    # from the far end (400, 150) -- e.g. a short horizontal line at
    # y=500. Requires BOTH ends close (see PRIOR_DART_LINE_MAX_PERP_PX's
    # comment for why) -- one close end alone must not suspect.
    result = detect_tip(bg, frame, prior_dart_line_px=((380.0, 500.0), (420.0, 500.0)))
    assert result.ok
    assert result.diagnostics["prior_dart_contamination_perp_tip_px"] < 5.0
    assert result.diagnostics["prior_dart_contamination_perp_far_px"] > 100.0
    assert result.diagnostics["prior_dart_contamination_suspected"] is False


# --------------------------------------------------------------------------
# find_prior_dart_line_px(): real filesystem lookup + recompute, using
# tmp/ (never /tmp, per docs/DESIGN.md) and real save_throw_package() output
# so the test exercises the actual on-disk meta.json shape.
# --------------------------------------------------------------------------


def _make_calibration():
    from opendarts.pipeline import CameraCalibration

    camera_matrix = np.array(
        [[800.0, 0.0, 400.0], [0.0, 800.0, 400.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist_coeffs = np.zeros(5, dtype=np.float64)
    rvec = np.array([[0.0], [0.0], [0.0]], dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [500.0]], dtype=np.float64)
    return CameraCalibration(
        camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvec=rvec, tvec=tvec,
        landmark_spread_ok=True,
    )


@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "session"
    d.mkdir(parents=True)
    return d


def _save_synthetic_throw(session_dir, throw_id, *, visit_id, visit_index, shaft_shift=0):
    bg, frame = _synthetic_shaft_frame()
    if shaft_shift:
        frame = np.roll(frame, shaft_shift, axis=1)
    calib = _make_calibration()
    result = ScoreResult(
        ok=True, sector="1", ring="single_inner", board_xy_mm=(1.0, 1.0),
        triangulation=None, n_cameras_used=1,
    )
    dest = session_dir / throw_id
    save_throw_package(
        dest_dir=dest,
        session=session_dir.name,
        bg_frames_bgr={0: bg},
        dart_frames_bgr={0: frame},
        calibrations={0: calib},
        result=result,
        visit_id=visit_id,
        visit_index=visit_index,
    )
    return dest


def test_find_prior_dart_line_px_locates_and_recomputes_the_prior_throw(session_dir):
    _save_synthetic_throw(session_dir, "throw_000", visit_id="visitA", visit_index=0)
    _save_synthetic_throw(session_dir, "throw_001", visit_id="visitA", visit_index=1)

    result = find_prior_dart_line_px(session_dir, "visitA", 1)
    assert result is not None
    assert 0 in result
    tip_px, far_end_px = result[0]
    # Matches what detect_tip() itself finds on the same synthetic shape
    # (tip near (400, 500), far end near (400, 150)) -- recomputed fresh,
    # not read from any stored value.
    assert tip_px[1] > 400.0
    assert far_end_px[1] < 250.0


def test_find_prior_dart_line_px_none_for_first_dart_of_visit(session_dir):
    _save_synthetic_throw(session_dir, "throw_000", visit_id="visitA", visit_index=0)
    assert find_prior_dart_line_px(session_dir, "visitA", 0) is None


def test_find_prior_dart_line_px_none_when_no_visit_context():
    assert find_prior_dart_line_px(Path("/nonexistent"), None, None) is None
    assert find_prior_dart_line_px(Path("/nonexistent"), "v", None) is None
    assert find_prior_dart_line_px(Path("/nonexistent"), None, 1) is None


def test_find_prior_dart_line_px_none_when_visit_id_does_not_match(session_dir):
    _save_synthetic_throw(session_dir, "throw_000", visit_id="visitA", visit_index=0)
    # Looking for visit_index-1=0 of a DIFFERENT visit -- must not match
    # visitA's own throw_000 just because the index lines up.
    assert find_prior_dart_line_px(session_dir, "visitB", 1) is None


def test_find_prior_dart_line_px_never_raises_on_missing_session_dir():
    assert find_prior_dart_line_px(Path("/definitely/does/not/exist"), "v", 1) is None


# --------------------------------------------------------------------------
# cached_frames (2026-09-01, "why call anything off disk at all when we
# have the prior dart in memory" finding): the fast in-memory path.
# --------------------------------------------------------------------------


def test_find_prior_dart_line_px_uses_cache_without_touching_disk_at_all(session_dir):
    """The defining property: a session_dir that does not exist on disk
    at all must still produce the correct result when the cache matches
    -- proves this genuinely never reaches the filesystem, not just that
    it's faster."""
    bg, frame = _synthetic_shaft_frame()
    cached = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=0, bg_frames={0: bg}, dart_frames={0: frame},
    )
    result = find_prior_dart_line_px(
        Path("/definitely/does/not/exist"), "visitA", 1, cached_frames=cached,
    )
    assert result is not None
    assert 0 in result
    tip_px, far_end_px = result[0]
    assert tip_px[1] > 400.0
    assert far_end_px[1] < 250.0


def test_find_prior_dart_line_px_cache_and_disk_produce_byte_identical_results(session_dir):
    """The real correctness guarantee: feeding the SAME frames through
    the cached path and the disk path must produce the exact same
    output -- confirms the fast path is a genuine stand-in, not an
    approximation (measured: 98.7ms disk vs 32.1ms memory, byte-identical
    result)."""
    dest = _save_synthetic_throw(session_dir, "throw_000", visit_id="visitA", visit_index=0)
    pkg = load_throw_package(dest)

    disk_result = find_prior_dart_line_px(session_dir, "visitA", 1)

    cached = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=0,
        bg_frames=pkg.bg_frames, dart_frames=pkg.dart_frames,
    )
    cached_result = find_prior_dart_line_px(
        Path("/definitely/does/not/exist"), "visitA", 1, cached_frames=cached,
    )

    assert disk_result is not None
    assert cached_result == disk_result


def test_find_prior_dart_line_px_cache_ignored_when_visit_id_mismatches(session_dir):
    """A cache for the wrong visit must never be silently used -- falls
    through to the (real, on-disk) lookup instead, exactly as if no
    cache had been passed at all."""
    _save_synthetic_throw(session_dir, "throw_000", visit_id="visitA", visit_index=0)
    bg, frame = _synthetic_shaft_frame()
    wrong_visit_cache = CachedPriorThrowFrames(
        visit_id="visitB", visit_index=0, bg_frames={0: bg}, dart_frames={0: frame},
    )
    # Asking about visitA's prior dart, but the cache is for visitB --
    # must fall through to disk and find visitA's real throw_000, not
    # silently accept the mismatched cache.
    result = find_prior_dart_line_px(
        session_dir, "visitA", 1, cached_frames=wrong_visit_cache,
    )
    assert result is not None # found via disk fallback, not the mismatched cache


def test_find_prior_dart_line_px_cache_ignored_when_visit_index_mismatches(session_dir):
    """A cache for the wrong throw within the SAME visit (e.g. two darts
    stale, not the immediately-prior one) must also be rejected -- checks
    visit_index, not just visit_id."""
    _save_synthetic_throw(session_dir, "throw_000", visit_id="visitA", visit_index=0)
    bg, frame = _synthetic_shaft_frame()
    stale_cache = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=5, # not target_index=1 (asking about visit_index=2)
        bg_frames={0: bg}, dart_frames={0: frame},
    )
    # No throw_dir exists for visitA/visit_index=1 on disk in this test --
    # if the mismatched cache were wrongly trusted this would return a
    # real result; it must instead correctly return None (falls through
    # to disk, finds nothing there either).
    result = find_prior_dart_line_px(session_dir, "visitA", 2, cached_frames=stale_cache)
    assert result is None


def test_find_prior_dart_line_px_none_cache_behaves_exactly_like_before_this_param(session_dir):
    """Explicit backward-compat check: cached_frames=None (the default,
    and every pre-existing caller) must be byte-identical to calling
    this function before the parameter existed at all."""
    _save_synthetic_throw(session_dir, "throw_000", visit_id="visitA", visit_index=0)
    _save_synthetic_throw(session_dir, "throw_001", visit_id="visitA", visit_index=1)
    with_default = find_prior_dart_line_px(session_dir, "visitA", 1)
    with_explicit_none = find_prior_dart_line_px(session_dir, "visitA", 1, cached_frames=None)
    assert with_default == with_explicit_none is not None


# --------------------------------------------------------------------------
# precomputed_tip_line (2026-09-01, stacks directly on the frame cache):
# the fastest tier, skips detect_tip() entirely.
# --------------------------------------------------------------------------


def test_find_prior_dart_line_px_precomputed_tip_line_skips_detect_tip_entirely(
    session_dir, monkeypatch
):
    """The defining property: even a cache whose bg_frames/dart_frames
    would make detect_tip() crash (or return something completely
    different) must be ignored in favor of precomputed_tip_line -- proves
    this tier genuinely never calls detect_tip() at all, not just that
    it returns the right answer."""
    import opendarts.engines.apollo.prior_dart_context as prior_dart_context_module

    def _boom(*args, **kwargs):
        raise AssertionError("detect_tip() must not be called when precomputed_tip_line is set")

    monkeypatch.setattr(prior_dart_context_module, "detect_tip", _boom)

    precomputed = {0: ((111.0, 222.0), (333.0, 444.0))}
    cached = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=0,
        bg_frames={0: np.zeros((4, 4, 3), dtype=np.uint8)},
        dart_frames={0: np.zeros((4, 4, 3), dtype=np.uint8)},
        precomputed_tip_line=precomputed,
    )
    result = find_prior_dart_line_px(
        Path("/definitely/does/not/exist"), "visitA", 1, cached_frames=cached,
    )
    assert result == precomputed


def test_find_prior_dart_line_px_precomputed_tip_line_none_falls_back_to_frame_cache(
    session_dir,
):
    """precomputed_tip_line=None (the field's own default -- an engine
    combination with nothing to offer, e.g. a non-Apollo primary) must
    fall back to the frame-cache tier (item 8), not treat "no precomputed
    value" as "no cache at all"."""
    bg, frame = _synthetic_shaft_frame()
    cached = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=0, bg_frames={0: bg}, dart_frames={0: frame},
        precomputed_tip_line=None,
    )
    result = find_prior_dart_line_px(
        Path("/definitely/does/not/exist"), "visitA", 1, cached_frames=cached,
    )
    assert result is not None
    tip_px, far_end_px = result[0]
    assert tip_px[1] > 400.0
    assert far_end_px[1] < 250.0


def test_find_prior_dart_line_px_precomputed_tip_line_empty_dict_returns_none():
    """A real, empty precomputed_tip_line (the prior throw genuinely had
    no camera with a usable line) is a real value, not "absent" -- must
    be honored (return None, matching what a fresh recompute finding
    nothing would also return), not silently fall through to the frame
    cache and potentially find a DIFFERENT answer."""
    cached = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=0, bg_frames={}, dart_frames={},
        precomputed_tip_line={},
    )
    result = find_prior_dart_line_px(
        Path("/definitely/does/not/exist"), "visitA", 1, cached_frames=cached,
    )
    assert result is None


def test_apollo_engine_score_populates_own_tip_line_px_byte_identical_to_fresh_recompute():
    """The real correctness guarantee behind the whole precomputed_tip_
    line tier: ApolloEngine.score()'s own diagnostics["own_tip_line_px"]
    (built during ITS OWN detect_tip() pass) must be byte-identical to
    independently calling detect_tip() on the exact same frames --
    confirms this is a genuine cache of an identical computation, not a
    different, cheaper approximation of it."""
    bg, frame = _synthetic_shaft_frame()
    bg_images = {0: bg, 1: bg}
    frame_images = {0: frame, 1: frame}
    calibrations = {0: _make_calibration(), 1: _make_calibration()}

    engine_result = ApolloEngine().score(bg_images, frame_images, calibrations)

    own_tip_line_px = engine_result.diagnostics.get("own_tip_line_px")
    assert own_tip_line_px is not None
    assert 0 in own_tip_line_px and 1 in own_tip_line_px

    for cam in (0, 1):
        fresh = detect_tip(bg, frame)
        assert fresh.ok and fresh.tip_px is not None and fresh.far_end_px is not None
        assert own_tip_line_px[cam] == (fresh.tip_px, fresh.far_end_px)


# --------------------------------------------------------------------------
# engine_accepts_prior_dart_line_px(): capability check, not name check.
# --------------------------------------------------------------------------


def test_engine_accepts_prior_dart_line_px_true_for_the_real_apollo_engine():
    assert engine_accepts_prior_dart_line_px(ApolloEngine()) is True


def test_engine_accepts_prior_dart_line_px_false_for_a_legacy_signature_stub():
    class _OldStyleEngine:
        def score(self, bg_images, frame_images, calibration):
            raise NotImplementedError

    assert engine_accepts_prior_dart_line_px(_OldStyleEngine()) is False


def test_engine_accepts_prior_dart_line_px_false_for_something_with_no_score_at_all():
    assert engine_accepts_prior_dart_line_px(object()) is False


# --------------------------------------------------------------------------
# Real-corpus end-to-end: the actual incident, and the actual near-miss
# this guard's own validation found and fixed.
# --------------------------------------------------------------------------


def test_real_contaminated_throw_scores_correctly_via_cam2_alternate():
    """The real incident (see module docstring). ORIGINAL pin (2026-08-16,
    when this guard shipped): the drop-and-compare decision excluded the
    contaminated cam2 and triangulated cameras 0+1 to ~97.68mm --
    15/treble, an honest ~0.2mm-past-the-wire residual miss this guard
    was explicitly not required to also fix.

    UPDATED 2026-08-17 (off-axis tip-cluster alternate, see
    tip_detection.TIP_CLUSTER_OFF_AXIS_MIN_PERP_PX and
    tests/test_engine_apollo_tip_off_axis_alt.py): cam2's
    contaminated tip cluster on this throw sits far off its component's
    own principal axis, so detect_tip() now also exposes an on-axis
    alternate for it -- and score_dart()'s combination search finds the
    FULL 3-camera set with that alternate agrees to 2.12mm (measured),
    landing at 96.15mm = 15/single_inner, which IS the operator/AD
    truth. That outcome supersedes the 2-camera drop (a full accepted
    ray set is preferred over any fallback pair, per score_dart()'s own
    documented preference order), so this pin now asserts the corrected,
    truth-matching result rather than the old residual miss. The guard's
    own drop-and-compare decision logic is still covered by
    test_replay_and_live_wiring_reach_the_same_decision_on_the_real_incident
    below and by the false-positive test that follows."""
    pkg_dir = _corpus_root() / KNOWN_CONTAMINATED_THROW
    if not (pkg_dir / "calibration.json").exists():
        pytest.skip(
            f"{KNOWN_CONTAMINATED_THROW} is not in the corpus on this machine -- "
            "the corpus is a living, curated thing (docs/DESIGN.md), so this pin skips "
            "rather than fails when it has moved on"
        )
    # 2026-08-18: this used to pin the exact (sector, ring)/cameras_used/
    # radius_mm this real package resolved to under one specific frozen
    # calibration. smoke tests must never depend on a specific
    # score for a specific dart package -- calibration is itself subject
    # to REPLAY (recomputed per corpus-refit, docs/DESIGN.md), so a real
    # package's exact geometric outcome is not a stable smoke-test
    # target. Real accuracy regressions (including on this exact throw)
    # are caught by the full-corpus replay tooling against AD truth, not
    # by pinning here. What stays meaningful as a smoke check: the guard
    # must still commit to an answer (not silently fall back to ok=False)
    # on the throw that originally motivated it.
    result = replay_throw_with_engine(pkg_dir, "Apollo")
    assert result.ok


def test_real_false_positive_throw_does_not_lose_its_correct_answer():
    """The real near-miss found DURING this guard's own corpus
    validation (see ApolloEngine.score()'s own dated comment): camera 0
    on this throw is ALSO flagged suspicious by the prior-dart-line
    signal (a real, measured false positive -- a legitimately close-
    grouped dart, not contamination), but the already-correct 3-camera
    answer (using cam0's own existing alt-candidate rescue) must survive
    unchanged, not be discarded for a confidently-wrong 2-camera pair."""
    pkg_dir = _corpus_root() / KNOWN_FALSE_POSITIVE_AVOIDED_THROW
    if not (pkg_dir / "calibration.json").exists():
        pytest.skip(
            f"{KNOWN_FALSE_POSITIVE_AVOIDED_THROW} is not in the corpus on this "
            "machine -- the corpus is a living, curated thing (docs/DESIGN.md), so "
            "this pin skips rather than fails when it has moved on"
        )
    # 2026-08-18: smoke tests must never pin an exact score, or
    # which cameras produced it, for a specific real corpus package --
    # calibration is itself subject to REPLAY (docs/DESIGN.md), so a real
    # package's exact computed outcome is not a stable smoke-test
    # target. Real regressions are caught by full-corpus replay against
    # AD truth (tmp/ scripts), not pytest pins.
    package = load_throw_package(pkg_dir)
    adg = package.ad_ground_truth
    assert adg is not None and adg.matched, "expected this pinned throw to have AD ground truth"

    result = replay_throw_with_engine(pkg_dir, "Apollo")
    assert result.ok


def test_replay_and_live_wiring_reach_the_same_decision_on_the_real_incident():
    """docs/DESIGN.md's "Replay is the source of truth", checked directly for this
    guard: opendarts.capture.replay's own prior-dart lookup and
    opendarts.engines.apollo.engine.ApolloEngine.score() called
    directly with the SAME find_prior_dart_line_px() output must agree
    -- there must be only one real decision path, not two that could
    silently drift apart."""
    pkg_dir = _corpus_root() / KNOWN_CONTAMINATED_THROW
    if not (pkg_dir / "calibration.json").exists():
        pytest.skip(
            f"{KNOWN_CONTAMINATED_THROW} is not in the corpus on this machine"
        )
    package = load_throw_package(pkg_dir)
    prior_line = find_prior_dart_line_px(
        package.package_dir.parent, package.visit_id, package.visit_index
    )
    direct = ApolloEngine().score(
        package.bg_frames, package.dart_frames, package.calibrations,
        prior_dart_line_px=prior_line,
    )
    via_replay = replay_throw_with_engine(pkg_dir, "Apollo")
    assert direct.ok == via_replay.ok
    assert direct.sector == via_replay.sector
    assert direct.ring == via_replay.ring
    assert direct.board_xy_mm == via_replay.board_xy_mm
    assert direct.diagnostics.get("cameras_used") == via_replay.diagnostics.get("cameras_used")


def test_a_tip_line_written_after_the_cache_was_built_is_used(session_dir, monkeypatch):
    """Live scoring runs on a background thread, so the prior throw's tip
    line lands in its out-dict AFTER the next throw's cache is built. The
    lookup must read it then, not the empty value from build time --
    otherwise every dart after the first re-runs detect_tip()."""
    import opendarts.engines.apollo.prior_dart_context as prior_dart_context_module

    def _boom(*args, **kwargs):
        raise AssertionError("detect_tip() must not run when the prior tip line is known")

    out: dict = {}
    cached = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=0,
        bg_frames={0: np.zeros((4, 4, 3), dtype=np.uint8)},
        dart_frames={0: np.zeros((4, 4, 3), dtype=np.uint8)},
        precomputed_tip_line=out.get("own_tip_line_px"),
        precomputed_tip_line_out=out,
    )
    precomputed = {0: ((111.0, 222.0), (333.0, 444.0))}
    out["own_tip_line_px"] = precomputed          # scoring finishes later
    monkeypatch.setattr(prior_dart_context_module, "detect_tip", _boom)
    result = find_prior_dart_line_px(
        Path("/definitely/does/not/exist"), "visitA", 1, cached_frames=cached,
    )
    assert result == precomputed


def test_a_tip_line_not_written_yet_falls_back_to_the_frame_cache(session_dir):
    bg, frame = _synthetic_shaft_frame()
    cached = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=0, bg_frames={0: bg}, dart_frames={0: frame},
        precomputed_tip_line=None, precomputed_tip_line_out={},
    )
    result = find_prior_dart_line_px(
        Path("/definitely/does/not/exist"), "visitA", 1, cached_frames=cached,
    )
    assert result is not None and result[0][0][1] > 400.0
