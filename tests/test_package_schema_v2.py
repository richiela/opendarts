"""Tests for the v2 package schema.

Covers:
  1. `meta.throw_number` / `meta.frame_cameras` -- new, additive fields
     on every FRESHLY saved throw package.
  2. `ad_ground_truth.json`'s `opendarts_captured_at_utc` -> `captured_at_utc`
     rename.
  3. The non-negotiables: schema-dispatch backward compat (existing
     v1-shaped packages/files keep loading, never rewritten), and a real
     "validate what save_throw_package() just wrote" CI check -- the
     validators themselves live in opendarts.capture.throw_package /
     opendarts.live.ad_ground_truth; this file is what actually calls them.

The `schema` string bump ("ad-ground-truth-v1" -> "ad-ground-truth-v2")
was deliberately HELD BACK through three rounds and landed 2026-08-27:
writing "v2" before the fields it promises were actually in place would
have made the version string assert a false guarantee and poison any
schema-dispatching reader. See
`opendarts.live.ad_ground_truth.AD_GROUND_TRUTH_SCHEMA_V2`'s own module
comment for the full story.

Deliberately does NOT re-test the base save/load round trip
(tests/test_capture_replay.py already owns that) -- only the new/changed
surface.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from opendarts.capture.throw_package import (
    load_throw_package,
    save_throw_package,
    validate_throw_package_meta_v2,
)
from opendarts.live.ad_ground_truth import (
    AD_GROUND_TRUTH_SCHEMA_V1_OpenDarts,
    AD_GROUND_TRUTH_SCHEMA_V2,
    AdGroundTruth,
    validate_ad_ground_truth_v2,
    _ad_ground_truth_schema_version,
    _resolve_ad_ground_truth_captured_at_utc,
)
from opendarts.pipeline import CameraCalibration, ScoreResult

@pytest.fixture()
def pkg_dir(tmp_path):
    return tmp_path / "pkg"


def _marker_image(cam_idx: int, seed: int, h: int = 12, w: int = 16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img[0, 0, 0] = cam_idx
    return img


def _calib(seed: int) -> CameraCalibration:
    rng = np.random.default_rng(seed)
    return CameraCalibration(
        camera_matrix=np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.zeros(5, dtype=np.float64),
        rvec=rng.uniform(-0.1, 0.1, size=3).astype(np.float64),
        tvec=np.array([0.0, 0.0, 400.0], dtype=np.float64),
        pnp_result=None,
        landmark_spread_ok=True,
    )


def _result() -> ScoreResult:
    return ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0), triangulation=None,
        n_cameras_used=3, max_ray_disagreement_mm=0.5,
    )


# ---------------------------------------------------------------------------
# 1. meta.throw_number / meta.frame_cameras -- new writes.
# ---------------------------------------------------------------------------

def test_save_throw_package_writes_throw_number_and_frame_cameras(pkg_dir):
    bg = {i: _marker_image(i, 10 + i) for i in range(3)}
    dart = {i: _marker_image(i, 20 + i) for i in range(3)}
    calibs = {i: _calib(30 + i) for i in range(3)}

    save_throw_package(pkg_dir, "sess-v2", bg, dart, calibs, _result(), throw_number=7)

    meta = json.loads((pkg_dir / "meta.json").read_text())
    assert meta["throw_number"] == 7
    assert meta["frame_cameras"] == [0, 1, 2]
    # The CI validator (the v2 package schema) must accept
    # exactly what save_throw_package() just wrote -- the real
    # "validate on write" check this task was asked to build.
    validate_throw_package_meta_v2(meta)

    pkg = load_throw_package(pkg_dir)
    assert pkg.throw_number == 7
    assert pkg.frame_cameras == [0, 1, 2]


def test_save_throw_package_throw_number_omitted_when_not_supplied(pkg_dir):
    """Same absent-not-fabricated convention as visit_id/calibration_
    package_id -- a caller that doesn't track throw_number (most offline
    tooling/tests) gets an honestly-missing key, not a guessed 0/None
    literal written into the file."""
    bg = {0: _marker_image(0, 1)}
    dart = {0: _marker_image(0, 2)}
    calibs = {0: _calib(3)}

    save_throw_package(pkg_dir, "sess-no-throw-number", bg, dart, calibs, _result())

    meta = json.loads((pkg_dir / "meta.json").read_text())
    assert "throw_number" not in meta
    # frame_cameras is NEVER optional -- always known, always written,
    # unlike throw_number.
    assert meta["frame_cameras"] == [0]
    validate_throw_package_meta_v2(meta)

    pkg = load_throw_package(pkg_dir)
    assert pkg.throw_number is None


def test_frame_cameras_genuinely_differs_from_cameras_when_calibration_missing(pkg_dir):
    """The real substance of the v2 package schema's frame_cameras field:
    a camera that produced a raw dart frame but has NO calibration this
    event is excluded from the persisted `cameras` (and its PNG is never
    written -- save_throw_package()'s image-write loop only iterates
    `cameras`), but frame_cameras must still honestly record that it
    DID produce a frame. This is exactly the "opendarts's integrity checks
    silently accept a degraded capture" gap the spec calls out -- before
    this field, that loss was invisible."""
    bg = {i: _marker_image(i, 40 + i) for i in range(3)}
    dart = {i: _marker_image(i, 50 + i) for i in range(3)}
    # Camera 2 produced a real dart frame but was never calibrated this
    # event (e.g. a mid-session calibration failure for just that cam).
    calibs = {0: _calib(60), 1: _calib(61)}

    save_throw_package(pkg_dir, "sess-degraded", bg, dart, calibs, _result())

    meta = json.loads((pkg_dir / "meta.json").read_text())
    assert meta["cameras"] == [0, 1]
    assert meta["frame_cameras"] == [0, 1, 2]
    validate_throw_package_meta_v2(meta)

    # cam2 genuinely never got a clip -- frame_cameras is not lying about
    # a file that doesn't exist, it's exposing a real, previously-silent
    # gap between "produced a frame" and "was persisted".
    assert set(meta["video"]["cameras"]) == {"0", "1"}
    assert not list(pkg_dir.glob("*cam2*"))
    for cam in ("0", "1"):
        assert (pkg_dir / meta["video"]["cameras"][cam]["clip"]).exists()

    pkg = load_throw_package(pkg_dir)
    assert pkg.cameras == [0, 1]
    assert pkg.frame_cameras == [0, 1, 2]


def test_frame_cameras_never_guessed_it_is_always_the_real_dart_frame_keys(pkg_dir):
    """The v2 package schema -- never coerce unknown into a real
    value. There is no "unknown" case for frame_cameras (dart_frames_bgr
    is always a required, already-populated dict), but this test pins
    down that the value is the REAL key set of that argument, not
    inferred from disk globbing or from `cameras`/bg_frames/calibrations
    -- swap which dict has the extra camera to prove it's specifically
    dart_frames_bgr's own keys being read, not some other input's."""
    bg = {i: _marker_image(i, 70 + i) for i in range(3)} # all 3 have bg
    dart = {0: _marker_image(0, 80), 1: _marker_image(1, 81)} # only 2 produced a dart frame
    calibs = {i: _calib(90 + i) for i in range(3)} # all 3 calibrated

    save_throw_package(pkg_dir, "sess-fewer-dart-frames", bg, dart, calibs, _result())

    meta = json.loads((pkg_dir / "meta.json").read_text())
    # cam2 has bg + calibration but never produced a dart frame this
    # throw -- frame_cameras must NOT include it, even though it's
    # "available" via the other two inputs.
    assert meta["frame_cameras"] == [0, 1]
    assert meta["cameras"] == [0, 1]


# ---------------------------------------------------------------------------
# 2. Real v1-shaped on-disk fixture (a package saved before this change)
# must still load correctly -- REPLAY principle.
# ---------------------------------------------------------------------------

def test_load_throw_package_v1_fixture_with_no_throw_number_or_frame_cameras(pkg_dir):
    """Reconstructs the REAL on-disk shape of every one of the 765
    packages under the session corpus as of this change
    (confirmed by directly reading a real package's meta.json before
    writing this test) -- no `throw_number`, no `frame_cameras` key at
    all. Must load fine, with both new ThrowPackage fields honestly None,
    never fabricated."""
    bg = {i: _marker_image(i, 100 + i) for i in range(3)}
    dart = {i: _marker_image(i, 110 + i) for i in range(3)}
    calibs = {i: _calib(120 + i) for i in range(3)}
    save_throw_package(pkg_dir, "sess-v1-fixture", bg, dart, calibs, _result())

    # Simulate a real pre-change package: strip the two new keys back out
    # of an already-saved meta.json, exactly like every package captured
    # before today.
    meta_path = pkg_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["frame_cameras"]
    assert "throw_number" not in meta # wasn't supplied above -- already absent
    meta_path.write_text(json.dumps(meta))

    pkg = load_throw_package(pkg_dir)
    assert pkg.cameras == [0, 1, 2]
    assert pkg.throw_number is None
    assert pkg.frame_cameras is None
    assert pkg.bg_frames is not None and len(pkg.bg_frames) == 3


# ---------------------------------------------------------------------------
# 3. AdGroundTruth schema/field rename (Part B decisions #4, #8).
# ---------------------------------------------------------------------------

def _v2_ad_ground_truth() -> AdGroundTruth:
    return AdGroundTruth(
        matched=True, match_reason="ok_ws", ad_base_url="http://localhost:3180",
        fetched_at_utc="2026-08-26T00:00:00+00:00",
        opendarts_captured_at_utc="2026-08-26T00:00:00.100000+00:00",
        staleness_sec=0.1, window_sec=12.0, sector="20", ring="treble",
    )


def test_ad_ground_truth_to_dict_writes_plain_captured_at_utc_and_bumped_schema():
    """2026-08-27 update: the schema bump (held back through three prior
    rounds specifically so it could land only once opendarts's own real gaps
    were fixed) lands in this same round as this file's null-vs-[]
    fixes -- see opendarts.live.ad_ground_truth.AD_GROUND_TRUTH_SCHEMA_V2's
    own module comment for the full story, including the real VALUE
    correction (an earlier ambiguous spec draft had this constant holding
    "dart-package/v2", which is actually meta.schema's value, not this
    file's)."""
    gt = _v2_ad_ground_truth()
    d = gt.to_dict()
    assert d["captured_at_utc"] == "2026-08-26T00:00:00.100000+00:00"
    assert "opendarts_captured_at_utc" not in d
    assert d["schema"] == AD_GROUND_TRUTH_SCHEMA_V2 == "ad-ground-truth-v2"
    # "ad-ground-truth-v1" is the pre-flip literal `to_dict()` used to
    # write (2026-08-27: AD_GROUND_TRUTH_SCHEMA_CURRENT, the named
    # constant this used to compare against, was removed as dead code --
    # see opendarts.live.ad_ground_truth's module comment at that spot).
    assert d["schema"] != "ad-ground-truth-v1"
    # The real CI validator this task was asked to build -- now checks
    # both the field rename AND the bumped schema string.
    validate_ad_ground_truth_v2(d)


def test_ad_ground_truth_round_trips_through_v2_dict():
    gt = _v2_ad_ground_truth()
    back = AdGroundTruth.from_dict(gt.to_dict())
    assert back.opendarts_captured_at_utc == gt.opendarts_captured_at_utc
    assert back.sector == "20"
    assert back.ring == "treble"


def test_ad_ground_truth_from_dict_reads_real_v1_opendarts_fixture():
    """A REAL v1-opendarts ad_ground_truth.json, byte-shape confirmed against
    a real package under the session corpus before writing
    this test: schema "ad-ground-truth-v1", key "opendarts_captured_at_utc"
    -- never rewritten on disk, must keep parsing correctly forever."""
    v1_fixture = {
        "schema": "ad-ground-truth-v1",
        "matched": True,
        "match_reason": "ok_ws",
        "ad_base_url": "http://localhost:3180",
        "fetched_at_utc": "2026-08-26T00:58:07.241999+00:00",
        "opendarts_captured_at_utc": "2026-08-26T00:58:07.241024+00:00",
        "staleness_sec": -0.720589,
        "window_sec": 12.0,
        "sector": "11",
        "ring": "single_outer",
        "tip_xy_mm": [-118.20187698152914, 16.350274087774824],
        "ad_method": None,
        "ad_bouncer": None,
        "ad_n_cam_detections": None,
        "raw_segment": {"name": "S11", "number": 11, "bed": "SingleOuter", "multiplier": 1},
        "source_index": 1,
        "n_detections_in_response": 1,
        "operator_marked_wrong": False,
        "operator_note": None,
        "operator_confirmed_source": None,
        "operator_confirmed_sector": None,
        "operator_confirmed_ring": None,
    }
    gt = AdGroundTruth.from_dict(v1_fixture)
    assert gt.opendarts_captured_at_utc == "2026-08-26T00:58:07.241024+00:00"
    assert gt.sector == "11"
    assert gt.ring == "single_outer"
    assert gt.matched is True


def test_ad_ground_truth_from_dict_missing_schema_key_treated_as_v1():
    """A package saved even earlier -- before ad_ground_truth.json had a
    "schema" key at all -- must also read the old key name correctly, not
    crash or silently read nothing. (Resolved by key PRESENCE right now,
    not by the schema string -- see _resolve_ad_ground_truth_captured_at_
    utc()'s own docstring for why, given the schema bump is held back.)"""
    no_schema_fixture = {
        "matched": True, "match_reason": "ok", "ad_base_url": "http://fake",
        "fetched_at_utc": "now", "opendarts_captured_at_utc": "then",
        "staleness_sec": None, "window_sec": 12.0, "sector": "5", "ring": "double",
    }
    assert _ad_ground_truth_schema_version(no_schema_fixture) == AD_GROUND_TRUTH_SCHEMA_V1_OpenDarts
    assert _resolve_ad_ground_truth_captured_at_utc(no_schema_fixture) == "then"
    gt = AdGroundTruth.from_dict(no_schema_fixture)
    assert gt.opendarts_captured_at_utc == "then"


def test_ad_ground_truth_schema_dispatch_now_reaches_v2_branch():
    """2026-08-27 update: `_ad_ground_truth_schema_version()` was real,
    tested, and ready for the bump since it was first added -- now that
    to_dict() actually emits AD_GROUND_TRUTH_SCHEMA_V2, a freshly-built
    payload reaches that branch, not the V1_OpenDarts retro-label."""
    d = _v2_ad_ground_truth().to_dict()
    assert _ad_ground_truth_schema_version(d) == AD_GROUND_TRUTH_SCHEMA_V2


def test_resolve_captured_at_utc_prefers_new_key_when_schema_string_is_still_old():
    """The exact interim shape this task now produces: schema is still
    the old literal, but the key is already renamed. Key-presence
    resolution must read the NEW value correctly despite the old schema
    string -- this is precisely the gap a naive schema-string dispatch
    would have silently gotten wrong (see this module's own comments)."""
    interim_payload = {
        "schema": "ad-ground-truth-v1",
        "captured_at_utc": "2026-08-26T01:00:00+00:00",
    }
    assert _resolve_ad_ground_truth_captured_at_utc(interim_payload) == "2026-08-26T01:00:00+00:00"


def test_validate_ad_ground_truth_v2_rejects_old_shaped_payload():
    old_style = {
        "schema": "ad-ground-truth-v1",
        "opendarts_captured_at_utc": "x",
    }
    with pytest.raises(AssertionError):
        validate_ad_ground_truth_v2(old_style)


def test_validate_ad_ground_truth_v2_requires_bumped_schema():
    """2026-08-27 update: the validator now DOES require the bumped
    schema string -- accepts a real payload this project's own to_dict()
    produces today (already bumped), and rejects one still holding the
    pre-flip literal (the exact shape every write made before this round
    produced)."""
    payload = _v2_ad_ground_truth().to_dict()
    assert payload["schema"] == "ad-ground-truth-v2"
    validate_ad_ground_truth_v2(payload) # must not raise

    pre_flip_payload = dict(payload, schema="ad-ground-truth-v1")
    with pytest.raises(AssertionError):
        validate_ad_ground_truth_v2(pre_flip_payload)


# ---------------------------------------------------------------------------
# Validator negative cases (a validator that can't fail is
# worse than no validator).
# ---------------------------------------------------------------------------

def test_validate_throw_package_meta_v2_rejects_missing_frame_cameras():
    meta = {"session": "s", "cameras": [0, 1], "captured_at_utc": "now"}
    with pytest.raises(AssertionError):
        validate_throw_package_meta_v2(meta)


def test_validate_throw_package_meta_v2_rejects_non_int_throw_number():
    meta = {
        "session": "s", "cameras": [0], "captured_at_utc": "now",
        "frame_cameras": [0], "throw_number": "7",
    }
    with pytest.raises(AssertionError):
        validate_throw_package_meta_v2(meta)


def test_validate_throw_package_meta_v2_accepts_real_v1_fixture_shape_is_false():
    """A real pre-change package (no frame_cameras key at all) correctly
    FAILS this v2-only validator -- it's not meant to certify old
    packages, only to gate new writes. Confirms the validator doesn't
    silently pass everything."""
    v1_meta = {
        "session": "20260825-174759",
        "cameras": [0, 1, 2],
        "captured_at_utc": "2026-08-26T00:58:07.241024+00:00",
        "visit_id": "visit_1787705882141",
        "visit_index": 0,
        "calibration_package_id": "calib_20260826-004759-7095ac17",
    }
    with pytest.raises(AssertionError):
        validate_throw_package_meta_v2(v1_meta)
