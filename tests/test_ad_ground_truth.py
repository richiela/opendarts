"""Tests for opendarts/live/ad_ground_truth.py -- the AD ground-truth
RECORD, its schema and its parsers -- plus the
opendarts/capture/throw_package.py extension that lets a saved throw
package carry AD ground truth alongside it, and the operator's own
"AD was wrong" annotation on top of it.

The REST poller and its batch backfill moved to dev/ with their own
tests (dev/tests/test_ad_ground_truth_rest.py); nothing here makes or
mocks an HTTP call. Package trees
are built directly on disk under ``<repo>/tmp/`` (gitignored, per
docs/DESIGN.md's filesystem discipline -- never ``/tmp``), mirroring
``opendarts.capture.throw_package``'s real on-disk layout without needing
real camera frames/calibration (this file is about the ground-truth
fetch/match/storage plumbing, not detection accuracy -- same split of
concerns as ``tests/test_rescore_all.py``).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from opendarts.capture.throw_package import (
    load_ad_ground_truth,
    load_throw_package,
    mark_operator_ad_wrong,
    save_ad_ground_truth,
)
from opendarts.live.ad_ground_truth import (
    AdGroundTruth,
    segment_to_sector_ring,
)

@pytest.fixture()
def package_root(tmp_path):
    root = tmp_path / "packages"
    root.mkdir(parents=True, exist_ok=True)
    return root


# A real detection entry's shape, adapted from an actual captured
# ad_detections.json -- see ad_ground_truth.py's
# module docstring for the field-by-field provenance.
def _real_shaped_detection(*, name="S17", number=17, bed="SingleInner", multiplier=1, x=0.21, y=-0.48):
    return {
        "detections": [{"vote": {"coords": {"x": x, "y": y}}}] * 3,
        "intersections": [],
        "method": "UnanimousCam",
        "bouncer": False,
        "coords": {"x": x, "y": y},
        "segment": {"name": name, "number": number, "bed": bed, "multiplier": multiplier},
        "timings": {},
    }


# --------------------------------------------------------------------------
# 1. AD's normalized board units -> mm.
# --------------------------------------------------------------------------


def test_tip_mm_conversion_matches_170mm_double_outer_radius():
    from opendarts.live.ad_ground_truth import _tip_xy_mm

    det = _real_shaped_detection(x=0.5, y=-0.25)
    mm = _tip_xy_mm(det)
    assert mm == pytest.approx((85.0, -42.5))


@pytest.mark.parametrize(
    "segment,expected",
    [
        ({"name": "S17", "number": 17, "bed": "SingleInner", "multiplier": 1}, ("17", "single_inner")),
        ({"name": "S17", "number": 17, "bed": "SingleOuter", "multiplier": 1}, ("17", "single_outer")),
        ({"name": "T20", "number": 20, "bed": "Triple", "multiplier": 3}, ("20", "treble")),
        ({"name": "D5", "number": 5, "bed": "Double", "multiplier": 2}, ("5", "double")),
        ({"name": "Bull", "number": 25, "bed": "Double", "multiplier": 2}, (None, "bull")),
        ({"name": "25", "number": 25, "bed": "Single", "multiplier": 1}, (None, "outer_bull")),
        ({"name": "25", "number": 25, "bed": "OuterBull", "multiplier": 1}, (None, "outer_bull")),
        ({"name": "M2", "number": 0, "bed": "Outside", "multiplier": 0}, (None, "outside")),
        (None, (None, None)),
        ({}, (None, None)),
    ],
)
def test_segment_to_sector_ring_real_shapes(segment, expected):
    assert segment_to_sector_ring(segment) == expected


# --------------------------------------------------------------------------
# ThrowPackage / throw_package.py storage extension.
# --------------------------------------------------------------------------

def _write_bare_package(root: Path, session: str, throw_id: str, *, captured_at_utc: str) -> Path:
    pkg_dir = root / session / throw_id
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "meta.json").write_text(
        json.dumps({"session": session, "cameras": [0, 1, 2], "captured_at_utc": captured_at_utc})
    )
    return pkg_dir


def test_save_and_load_ad_ground_truth_round_trip(package_root):
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    gt = AdGroundTruth(
        matched=True,
        match_reason="ok",
        ad_base_url="http://fake:3180",
        fetched_at_utc=datetime.now(timezone.utc).isoformat(),
        opendarts_captured_at_utc=None,
        staleness_sec=1.2,
        window_sec=12.0,
        sector="20",
        ring="treble",
        tip_xy_mm=(12.5, -3.4),
        ad_method="UnanimousCam",
        ad_bouncer=False,
        ad_n_cam_detections=3,
        raw_segment={"name": "T20", "number": 20, "bed": "Triple", "multiplier": 3},
        source_index=0,
        n_detections_in_response=1,
    )

    path = save_ad_ground_truth(pkg_dir, gt)
    assert path.exists()

    loaded = load_ad_ground_truth(pkg_dir)
    assert loaded is not None
    assert loaded.sector == "20"
    assert loaded.ring == "treble"
    assert loaded.tip_xy_mm == pytest.approx((12.5, -3.4))
    assert loaded.matched is True
    assert loaded.raw_segment == gt.raw_segment


def test_load_ad_ground_truth_missing_file_returns_none(package_root):
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    assert load_ad_ground_truth(pkg_dir) is None


def test_save_ad_ground_truth_refuses_nonexistent_directory(package_root):
    with pytest.raises(FileNotFoundError):
        save_ad_ground_truth(package_root / "does_not_exist", AdGroundTruth(
            matched=False, match_reason="x", ad_base_url="http://fake",
            fetched_at_utc="x", opendarts_captured_at_utc=None, staleness_sec=None,
            window_sec=12.0,
        ))


def test_throw_package_without_ad_ground_truth_still_loads(package_root, monkeypatch):
    """Backward compatibility: a package saved before ad_ground_truth.json
    existed must still load fine, with ad_ground_truth=None -- and so
    must one carrying the retired ad_calibration.json, which is ignored."""
    import cv2

    pkg_dir = package_root / "sess1" / "throw1"
    pkg_dir.mkdir(parents=True)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    for cam in (0,):
        cv2.imwrite(str(pkg_dir / f"cam{cam}_bg.png"), frame)
        cv2.imwrite(str(pkg_dir / f"cam{cam}_frame.png"), frame)
    (pkg_dir / "calibration.json").write_text(json.dumps({
        "0": {
            "camera_matrix": np.eye(3).tolist(),
            "dist_coeffs": np.zeros(5).tolist(),
            "rvec": np.zeros(3).tolist(),
            "tvec": np.array([0, 0, 1000]).tolist(),
            "landmark_spread_ok": True,
        }
    }))
    (pkg_dir / "meta.json").write_text(json.dumps({
        "session": "sess1", "cameras": [0],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    }))

    (pkg_dir / "ad_calibration.json").write_text(json.dumps({"quads_px": []}))

    pkg = load_throw_package(pkg_dir)
    assert pkg.ad_ground_truth is None


# --------------------------------------------------------------------------
# 3b. The "AD was wrong" operator flag (opendarts/live/server.py's Scoring-tab
# mark/unmark toggle, POST /api/packages/{session}/{throw_id}/mark-ad-wrong)
# -- a HUMAN judgment call persisted onto ad_ground_truth.json itself, added
# 2026-08-12: live-testing, hit a throw opendarts scored T1 correctly
# while AD's own ground truth disagreed/showed a miss. Adapted from real
# prior art: an operator-triggered, undoable toggle.
# --------------------------------------------------------------------------

def test_ad_ground_truth_operator_fields_default_false_and_none():
    """A freshly-constructed AdGroundTruth (e.g. every existing call site
    in this module/backfill CLI, none of which know about the operator
    flag) never accidentally starts pre-flagged."""
    gt = AdGroundTruth(
        matched=True, match_reason="ok", ad_base_url="http://fake",
        fetched_at_utc="now", opendarts_captured_at_utc=None, staleness_sec=None,
        window_sec=12.0,
    )
    assert gt.operator_marked_wrong is False
    assert gt.operator_note is None


def test_ad_ground_truth_to_dict_from_dict_round_trip_includes_operator_fields():
    gt = AdGroundTruth(
        matched=True, match_reason="ok", ad_base_url="http://fake",
        fetched_at_utc="now", opendarts_captured_at_utc=None, staleness_sec=None,
        window_sec=12.0, sector="20", ring="treble",
        operator_marked_wrong=True, operator_note="dart bounced, AD registered a miss",
    )
    d = gt.to_dict()
    assert d["operator_marked_wrong"] is True
    assert d["operator_note"] == "dart bounced, AD registered a miss"

    back = AdGroundTruth.from_dict(d)
    assert back.operator_marked_wrong is True
    assert back.operator_note == "dart bounced, AD registered a miss"


def test_ad_ground_truth_from_dict_backward_compatible_when_operator_fields_missing():
    """A real ad_ground_truth.json written before this field existed has
    no "operator_marked_wrong"/"operator_note" keys at all -- from_dict()
    must degrade to the honest False/None default, not raise a KeyError."""
    old_style = {
        "schema": "ad-ground-truth-v1",
        "matched": True,
        "match_reason": "ok",
        "ad_base_url": "http://fake",
        "fetched_at_utc": "now",
        "opendarts_captured_at_utc": None,
        "staleness_sec": None,
        "window_sec": 12.0,
        "sector": "20",
        "ring": "treble",
    }
    assert "operator_marked_wrong" not in old_style # this test's own premise

    gt = AdGroundTruth.from_dict(old_style)
    assert gt.operator_marked_wrong is False
    assert gt.operator_note is None
    assert gt.sector == "20" # everything else still loads correctly


def test_mark_operator_ad_wrong_creates_placeholder_when_no_prior_ad_data(package_root):
    """The literal "AD MISSES" case: AD never
    registered/committed the throw at all (no ad_ground_truth.json on
    disk), yet a human watching live knows AD was wrong (missed it
    entirely). A minimal placeholder record is created purely so the
    operator flag has somewhere durable to live."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    assert load_ad_ground_truth(pkg_dir) is None

    gt = mark_operator_ad_wrong(pkg_dir, True, "AD never showed anything for this dart")
    assert gt.operator_marked_wrong is True
    assert gt.operator_note == "AD never showed anything for this dart"
    assert gt.matched is False
    # Deliberately distinguishable from a real fetch attempt's own
    # no-match reasons ("stale", "fetch_error", "no_detections", ...).
    assert gt.match_reason == "operator_marked_no_ad_data"

    reloaded = load_ad_ground_truth(pkg_dir)
    assert reloaded is not None
    assert reloaded.operator_marked_wrong is True
    assert reloaded.operator_note == "AD never showed anything for this dart"


def test_mark_operator_ad_wrong_preserves_existing_real_ad_data(package_root):
    """The case actually hit live: AD DID commit a result (a real
    ad_ground_truth.json already exists), it was just wrong. Marking it
    must annotate, not clobber, the real fetched sector/ring/tip data --
    that data stays useful for comparison regardless of the flag."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    real_gt = AdGroundTruth(
        matched=True, match_reason="ok", ad_base_url="http://localhost:3180",
        fetched_at_utc="2026-08-12T00:00:00+00:00", opendarts_captured_at_utc=None,
        staleness_sec=1.0, window_sec=12.0, sector=None, ring="outside",
        tip_xy_mm=None, ad_method="UnanimousCam",
    )
    save_ad_ground_truth(pkg_dir, real_gt)

    gt = mark_operator_ad_wrong(pkg_dir, True, "opendarts scored T1 correctly, AD showed MISS")
    assert gt.operator_marked_wrong is True
    assert gt.matched is True # untouched
    assert gt.ring == "outside" # untouched -- AD's own (wrong) real answer, preserved
    assert gt.ad_method == "UnanimousCam" # untouched

    reloaded = load_ad_ground_truth(pkg_dir)
    assert reloaded.ring == "outside"
    assert reloaded.operator_marked_wrong is True
    assert reloaded.operator_note == "opendarts scored T1 correctly, AD showed MISS"


def test_mark_operator_ad_wrong_marking_twice_is_idempotent(package_root):
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    mark_operator_ad_wrong(pkg_dir, True, "first note")
    gt = mark_operator_ad_wrong(pkg_dir, True, "second note")
    assert gt.operator_marked_wrong is True
    assert gt.operator_note == "second note" # re-marking updates the note, doesn't error


def test_mark_operator_ad_wrong_toggle_unmarks_and_clears_note(package_root):
    """direct: "make the action a toggle" -- undoing a mistaken
    flag must clear the flag AND its note (they're one annotation), while
    leaving the underlying real AD data (when present) completely alone."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    real_gt = AdGroundTruth(
        matched=True, match_reason="ok", ad_base_url="http://fake",
        fetched_at_utc="now", opendarts_captured_at_utc=None, staleness_sec=1.0,
        window_sec=12.0, sector="17", ring="single_inner",
    )
    save_ad_ground_truth(pkg_dir, real_gt)

    mark_operator_ad_wrong(pkg_dir, True, "accidental click")
    assert load_ad_ground_truth(pkg_dir).operator_marked_wrong is True

    gt = mark_operator_ad_wrong(pkg_dir, False)
    assert gt.operator_marked_wrong is False
    assert gt.operator_note is None
    assert gt.sector == "17" # real AD data untouched by the toggle
    assert gt.ring == "single_inner"

    reloaded = load_ad_ground_truth(pkg_dir)
    assert reloaded.operator_marked_wrong is False
    assert reloaded.operator_note is None


def test_mark_operator_ad_wrong_unmark_when_never_marked_is_a_safe_noop(package_root):
    """Idempotent-safe in the OTHER direction too: unmarking a package
    that was never flagged (e.g. a stale/duplicate client request) must
    not raise or produce a confusing state."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    gt = mark_operator_ad_wrong(pkg_dir, False)
    assert gt.operator_marked_wrong is False
    assert gt.operator_note is None


def test_mark_operator_ad_wrong_raises_on_nonexistent_package_dir(package_root):
    """Same discipline as save_ad_ground_truth(): this only annotates an
    already-saved throw package, it never creates one."""
    with pytest.raises(FileNotFoundError):
        mark_operator_ad_wrong(package_root / "does_not_exist", True)


# --------------------------------------------------------------------------
# 3c. "AD was wrong -- and THIS is what was actually right".
# The operator_confirmed_source/_sector/_ring fields on AdGroundTruth --
# same file, same annotation, same toggle semantics as the flag/note
# above, not a parallel record.
# --------------------------------------------------------------------------

def test_ad_ground_truth_confirmed_fields_default_none():
    """Every pre-existing construction site (this module's own fetch path,
    the backfill CLI, mark_operator_ad_wrong's placeholder) never
    accidentally starts with a phantom human confirmation on it."""
    gt = AdGroundTruth(
        matched=True, match_reason="ok", ad_base_url="http://fake",
        fetched_at_utc="now", opendarts_captured_at_utc=None, staleness_sec=None,
        window_sec=12.0,
    )
    assert gt.operator_confirmed_source is None
    assert gt.operator_confirmed_sector is None
    assert gt.operator_confirmed_ring is None


def test_ad_ground_truth_round_trip_includes_confirmed_fields():
    gt = AdGroundTruth(
        matched=True, match_reason="ok", ad_base_url="http://fake",
        fetched_at_utc="now", opendarts_captured_at_utc=None, staleness_sec=None,
        window_sec=12.0, sector="5", ring="single_outer",
        operator_marked_wrong=True,
        operator_confirmed_source="Talos",
        operator_confirmed_sector="20",
        operator_confirmed_ring="treble",
    )
    d = gt.to_dict()
    assert d["operator_confirmed_source"] == "Talos"
    assert d["operator_confirmed_sector"] == "20"
    assert d["operator_confirmed_ring"] == "treble"

    back = AdGroundTruth.from_dict(d)
    assert back.operator_confirmed_source == "Talos"
    assert back.operator_confirmed_sector == "20"
    assert back.operator_confirmed_ring == "treble"


def test_ad_ground_truth_from_dict_backward_compatible_when_confirmed_fields_missing():
    """THE backward-compat gate for this feature: a real
    ad_ground_truth.json written before the operator_confirmed_* fields
    existed -- including one that already carries the OLDER
    operator_marked_wrong/operator_note pair -- must still load, degrade
    the three new keys to None, and leave everything else untouched."""
    old_style = {
        "schema": "ad-ground-truth-v1",
        "matched": True,
        "match_reason": "ok",
        "ad_base_url": "http://fake",
        "fetched_at_utc": "now",
        "opendarts_captured_at_utc": None,
        "staleness_sec": None,
        "window_sec": 12.0,
        "sector": "20",
        "ring": "treble",
        "tip_xy_mm": [1.5, -2.5],
        "operator_marked_wrong": True,
        "operator_note": "AD showed a miss",
    }
    for key in ("operator_confirmed_source", "operator_confirmed_sector", "operator_confirmed_ring"):
        assert key not in old_style # this test's own premise

    gt = AdGroundTruth.from_dict(old_style)
    assert gt.operator_confirmed_source is None
    assert gt.operator_confirmed_sector is None
    assert gt.operator_confirmed_ring is None
    # Nothing else regressed -- the pre-existing fields still load exactly
    # as they did before the three new ones were added.
    assert gt.operator_marked_wrong is True
    assert gt.operator_note == "AD showed a miss"
    assert gt.sector == "20"
    assert gt.ring == "treble"
    assert gt.tip_xy_mm == (1.5, -2.5)


def test_mark_operator_ad_wrong_persists_confirmed_answer(package_root):
    """The real case: AD said S5, a human watching says Talos's T20 was
    right. Both AD's own (wrong) answer and the human's confirmation live
    on the SAME record -- AD's raw data is annotated, never overwritten,
    so replay/comparison keeps both sides."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    save_ad_ground_truth(pkg_dir, AdGroundTruth(
        matched=True, match_reason="ok", ad_base_url="http://fake",
        fetched_at_utc="now", opendarts_captured_at_utc=None, staleness_sec=1.0,
        window_sec=12.0, sector="5", ring="single_outer", tip_xy_mm=(40.0, -110.0),
    ))

    gt = mark_operator_ad_wrong(
        pkg_dir, True, "AD had S5, Talos had it right",
        confirmed_source="Talos", confirmed_sector="20", confirmed_ring="treble",
    )
    assert gt.operator_marked_wrong is True
    assert gt.operator_confirmed_source == "Talos"
    assert gt.operator_confirmed_sector == "20"
    assert gt.operator_confirmed_ring == "treble"
    # AD's own answer preserved untouched alongside it.
    assert gt.sector == "5"
    assert gt.ring == "single_outer"
    assert gt.tip_xy_mm == (40.0, -110.0)

    reloaded = load_ad_ground_truth(pkg_dir)
    assert reloaded.operator_confirmed_source == "Talos"
    assert reloaded.operator_confirmed_sector == "20"
    assert reloaded.operator_confirmed_ring == "treble"
    assert reloaded.sector == "5"


def test_mark_operator_ad_wrong_persists_manual_confirmation_with_no_sector(package_root):
    """A manually-entered bull/outer_bull/outside confirmation has a real
    ring and a legitimately-None sector (that IS
    opendarts.geometry.board.sector_ring_for_point's own return shape) -- the
    absent sector must not be mistaken for "no confirmation was made"."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    gt = mark_operator_ad_wrong(
        pkg_dir, True, None, confirmed_source="manual", confirmed_sector=None, confirmed_ring="bull"
    )
    assert gt.operator_confirmed_source == "manual"
    assert gt.operator_confirmed_sector is None
    assert gt.operator_confirmed_ring == "bull"
    assert load_ad_ground_truth(pkg_dir).operator_confirmed_ring == "bull"


def test_mark_operator_ad_wrong_unmark_clears_confirmed_answer(package_root):
    """Same "one annotation, cleared whole" rule operator_note already
    follows: undoing the flag undoes its entire explanation, confirmation
    included -- while leaving AD's own real data alone."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    save_ad_ground_truth(pkg_dir, AdGroundTruth(
        matched=True, match_reason="ok", ad_base_url="http://fake",
        fetched_at_utc="now", opendarts_captured_at_utc=None, staleness_sec=1.0,
        window_sec=12.0, sector="5", ring="single_outer",
    ))
    mark_operator_ad_wrong(
        pkg_dir, True, "note", confirmed_source="Talos",
        confirmed_sector="20", confirmed_ring="treble",
    )
    assert load_ad_ground_truth(pkg_dir).operator_confirmed_ring == "treble"

    gt = mark_operator_ad_wrong(pkg_dir, False)
    assert gt.operator_marked_wrong is False
    assert gt.operator_note is None
    assert gt.operator_confirmed_source is None
    assert gt.operator_confirmed_sector is None
    assert gt.operator_confirmed_ring is None
    assert gt.sector == "5" # AD's real data untouched by the toggle
    assert load_ad_ground_truth(pkg_dir).operator_confirmed_ring is None


def test_mark_operator_ad_wrong_rewrites_confirmation_whole_never_merges(package_root):
    """Re-marking writes the WHOLE annotation from this call's arguments,
    exactly as operator_note has always behaved -- it does not merge a new
    call's confirmation on top of a previous one's leftovers. Corrections
    from the dashboard's modal always resend the full answer, so a
    partial-merge would only ever be able to resurrect a stale field."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    mark_operator_ad_wrong(
        pkg_dir, True, "first", confirmed_source="Talos",
        confirmed_sector="20", confirmed_ring="treble",
    )
    gt = mark_operator_ad_wrong(
        pkg_dir, True, "corrected", confirmed_source="manual",
        confirmed_sector=None, confirmed_ring="bull",
    )
    assert gt.operator_confirmed_source == "manual"
    assert gt.operator_confirmed_sector is None # NOT still "20"
    assert gt.operator_confirmed_ring == "bull"
    assert gt.operator_note == "corrected"


def test_mark_operator_ad_wrong_without_confirmation_is_unchanged(package_root):
    """The pre-2026-08-13 call shape (positional wrong/note only, which is
    still exactly what the dashboard's Unmark button and every older
    caller send) keeps working and leaves all three new fields None."""
    pkg_dir = _write_bare_package(
        package_root, "sess1", "throw1", captured_at_utc=datetime.now(timezone.utc).isoformat()
    )
    gt = mark_operator_ad_wrong(pkg_dir, True, "AD was wrong, dunno who was right")
    assert gt.operator_marked_wrong is True
    assert gt.operator_note == "AD was wrong, dunno who was right"
    assert gt.operator_confirmed_source is None
    assert gt.operator_confirmed_sector is None
    assert gt.operator_confirmed_ring is None
