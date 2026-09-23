"""Engine interface + registry basics -- docs/ENGINES.md's "Engine
interface" / "Registering an engine" sections."""
from __future__ import annotations

import pytest

from opendarts.engines.base import (
    EngineResult,
    engine_has_custom_calibrate,
    engine_result_from_dict,
    engine_result_to_score_result,
)
from opendarts.engines.apollo import ApolloEngine
from opendarts.engines.talos import TalosEngine
from opendarts.engines.registry import (
    DEFAULT_PRIMARY_ENGINE,
    ENGINES,
    engine_names,
    get_engine,
    is_registered,
)




def test_default_config_is_the_production_shape():
    """Production runs all four detection engines and lets Zeus aggregate
    them -- that is the DEFAULT, not something an operator has to switch
    on. Zeus is a combiner with no detector of its own, so the four
    sub-engines are also-run alongside it, which is what puts each
    engine's individual answer in the package."""
    from opendarts.engines.registry import DEFAULT_ALSO_RUN
    from opendarts.engines.zeus.engine import ZEUS_SUB_ENGINE_NAMES

    assert DEFAULT_PRIMARY_ENGINE == "Zeus"
    assert is_registered(DEFAULT_PRIMARY_ENGINE)
    # derived from Zeus's own roster, so the two cannot drift apart
    assert DEFAULT_ALSO_RUN == ZEUS_SUB_ENGINE_NAMES
    for name in DEFAULT_ALSO_RUN:
        assert is_registered(name)


def test_is_registered_and_get_engine_agree():
    for name in engine_names():
        assert is_registered(name)
    assert not is_registered("NotARealEngine")
    with pytest.raises(KeyError):
        get_engine("NotARealEngine")


def test_registry_is_a_plain_dict_engine_names_preserves_order():
    # Registration order matters for the dashboard's radio/checkbox
    # rendering (docs/ENGINES.md doesn't require a specific order, but it
    # should be STABLE, not re-derived differently each call).
    assert engine_names() == list(ENGINES.keys())
    assert engine_names() == engine_names()


def test_engine_result_to_dict_round_trips_core_fields():
    er = EngineResult(
        ok=True, sector="20", ring="double", board_xy_mm=(1.5, -2.5),
        reason="ok", diagnostics={"foo": "bar"}, duration_s=0.01, timed_out=False,
    )
    d = er.to_dict()
    assert d == {
        "ok": True, "sector": "20", "ring": "double", "board_xy_mm": [1.5, -2.5],
        "reason": "ok", "diagnostics": {"foo": "bar"}, "duration_s": 0.01, "timed_out": False,
        "confidence": None,
    }


def test_engine_result_to_dict_handles_none_board_xy():
    er = EngineResult(ok=False, sector=None, ring=None, board_xy_mm=None)
    assert er.to_dict()["board_xy_mm"] is None


# 2026-08-27 perf task -- engine_result_from_dict() is the symmetric
# inverse of to_dict(), added so opendarts.live.capture_daemon's
# reused-from-Zeus also-run path can reconstruct real EngineResult
# objects from Zeus's own already-serialized diagnostics["sub_results"]
# (which MUST be plain dicts -- diagnostics has to stay JSON-serializable
# -- never raw EngineResult objects). See its own docstring for why this
# is a lossless round trip, not a re-derivation.
def test_engine_result_from_dict_round_trips_to_dict_output():
    original = EngineResult(
        ok=True, sector="20", ring="double", board_xy_mm=(1.5, -2.5),
        reason="triangulated", diagnostics={"foo": "bar"},
        duration_s=0.123, timed_out=False, confidence=0.9,
    )
    reconstructed = engine_result_from_dict(original.to_dict())
    assert reconstructed == original


def test_engine_result_from_dict_handles_none_board_xy():
    original = EngineResult(ok=False, sector=None, ring=None, board_xy_mm=None)
    reconstructed = engine_result_from_dict(original.to_dict())
    assert reconstructed.board_xy_mm is None
    assert reconstructed == original


def test_engine_result_from_dict_defaults_missing_optional_fields():
    # A minimal dict (e.g. hand-built in a test, or an older on-disk
    # shape missing a field to_dict() always writes today) must not
    # raise -- falls back to EngineResult's own dataclass defaults.
    reconstructed = engine_result_from_dict({"ok": True, "sector": "20", "ring": "single"})
    assert reconstructed.ok is True
    assert reconstructed.board_xy_mm is None
    assert reconstructed.reason == ""
    assert reconstructed.diagnostics == {}
    assert reconstructed.duration_s == 0.0
    assert reconstructed.timed_out is False
    assert reconstructed.confidence is None


def test_engine_result_from_dict_preserves_real_duration_s_and_timed_out():
    """The exact honesty property opendarts.live.capture_daemon's
    reused-from-Zeus path relies on: a real, non-default duration_s/
    timed_out survives the round trip unchanged."""
    original = EngineResult(
        ok=True, sector="5", ring="treble", board_xy_mm=(0.0, 0.0),
        duration_s=0.087, timed_out=False,
    )
    reconstructed = engine_result_from_dict(original.to_dict())
    assert reconstructed.duration_s == 0.087
    assert reconstructed.timed_out is False


# 2026-08-27, the v2 package schema null-vs-[] follow-up -- see
# docs/DESIGN.md's dated entry for the full context. QA's own real measurement
# found `other_engines.<Name>.diagnostics` (written via `EngineResult.
# to_dict()`, generic pass-through of `self.diagnostics`, for ANY engine
# dispatched as also-run) was never covered by the 2026-08-27 "final
# round"'s null-vs-[] fix -- that round only touched Apollo's own
# `score_result_to_engine_result()` and throw_package.py's top-level
# rollup, both PRIMARY-engine-only call sites. `to_dict()` is the one
# real generic choke point every engine's diagnostics passes through
# (both the primary write and the also-run write), so the fix lives
# here -- "can't be missed by a new engine, can't drift when an existing
# one is edited," per QA's own framing.
def test_engine_result_to_dict_normalizes_null_cameras_used_to_empty_list():
    er = EngineResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
        diagnostics={"cameras_used": None},
    )
    assert er.to_dict()["diagnostics"]["cameras_used"] == []


def test_engine_result_to_dict_normalizes_null_alt_candidates_used_to_empty_list():
    er = EngineResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
        diagnostics={"alt_candidates_used": None},
    )
    assert er.to_dict()["diagnostics"]["alt_candidates_used"] == []


def test_engine_result_to_dict_normalizes_null_nested_per_ray_distance_mm():
    er = EngineResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
        diagnostics={"triangulation": {"ok": False, "per_ray_distance_mm": None}},
    )
    tri = er.to_dict()["diagnostics"]["triangulation"]
    assert tri["per_ray_distance_mm"] == []
    assert tri["ok"] is False  # untouched sibling field


def test_engine_result_to_dict_leaves_real_collection_values_alone():
    er = EngineResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
        diagnostics={
            "cameras_used": [0, 1, 2],
            "alt_candidates_used": [1],
            "triangulation": {"per_ray_distance_mm": [0.5, 0.6]},
        },
    )
    d = er.to_dict()["diagnostics"]
    assert d["cameras_used"] == [0, 1, 2]
    assert d["alt_candidates_used"] == [1]
    assert d["triangulation"]["per_ray_distance_mm"] == [0.5, 0.6]


def test_engine_result_to_dict_never_fabricates_absent_keys():
    """The critical don't-overreach guard: an engine that never tracks
    `alt_candidates_used`/`cameras_used`/`triangulation` at all (e.g.
    Talos's real "centerline_ray" diagnostics shape, confirmed by direct
    trace of opendarts/engines/talos/engine.py -- see
    test_talos_centerline_ray_diagnostics_never_sets_cameras_used_or_alt_
    candidates_used_key below) must keep NOT having those keys. Having a
    key and it being null is a different, real bug; not having the key at
    all is legitimate and must stay that way."""
    er = EngineResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
        diagnostics={"engine": "Talos", "observation": "centerline_ray", "n_cameras_used": 3},
    )
    d = er.to_dict()["diagnostics"]
    assert "cameras_used" not in d
    assert "alt_candidates_used" not in d
    assert "triangulation" not in d
    assert d == {"engine": "Talos", "observation": "centerline_ray", "n_cameras_used": 3}


def test_engine_result_to_dict_leaves_genuine_nullable_scalars_untouched():
    """`outlier_camera` (and any similar genuine scalar, e.g. Talos's own
    ray_fallback.py disagreement field) is correctly nullable -- QA's own
    re-confirmed ruling: null is CORRECT there, per the empty-collection-
    vs-null-scalar distinction. Must never be touched by this
    normalization."""
    er = EngineResult(
        ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0),
        diagnostics={"outlier_camera": None, "cameras_used": None},
    )
    d = er.to_dict()["diagnostics"]
    assert d["outlier_camera"] is None
    assert d["cameras_used"] == []


def test_engine_result_to_dict_does_not_mutate_original_diagnostics():
    """to_dict() is called more than once for the same EngineResult in
    real code (e.g. once for the primary write, again if the same object
    is inspected/logged elsewhere) -- normalization must not leave a
    side effect on the object's own .diagnostics attribute."""
    original = {"cameras_used": None, "triangulation": {"per_ray_distance_mm": None}}
    er = EngineResult(ok=True, sector="20", ring="single", board_xy_mm=(1.0, 2.0), diagnostics=original)
    er.to_dict()
    assert original["cameras_used"] is None
    assert original["triangulation"]["per_ray_distance_mm"] is None


def test_engine_result_to_dict_handles_non_dict_triangulation():
    # triangulation can legitimately be None (no triangulation attempted
    # at all) -- must not crash trying to normalize a nested field of it.
    er = EngineResult(
        ok=False, sector=None, ring=None, board_xy_mm=None,
        diagnostics={"cameras_used": None, "triangulation": None},
    )
    d = er.to_dict()["diagnostics"]
    assert d["cameras_used"] == []
    assert d["triangulation"] is None


def test_engine_has_custom_calibrate_false_for_both_current_engines():
    # Neither Apollo nor Talos implements calibrate() -- both use the
    # default (reuse the primary engine's already-solved calibration) per
    # docs/ENGINES.md's "most won't implement this."
    assert engine_has_custom_calibrate(ApolloEngine()) is False
    assert engine_has_custom_calibrate(TalosEngine()) is False


def test_engine_has_custom_calibrate_true_when_present():
    class FakeEngineWithCalibrate:
        def score(self, bg_images, frame_images, calibration):
            raise NotImplementedError

        def calibrate(self, raw_frames):
            return {}

    assert engine_has_custom_calibrate(FakeEngineWithCalibrate()) is True


def test_engine_result_to_score_result_adapter_maps_core_fields():
    er = EngineResult(
        ok=True, sector="7", ring="treble", board_xy_mm=(3.0, 4.0),
        reason="stub reason", diagnostics={"max_ray_disagreement_mm": 2.5},
    )
    sr = engine_result_to_score_result(er)
    assert sr.ok is True
    assert sr.sector == "7"
    assert sr.ring == "treble"
    assert sr.board_xy_mm == (3.0, 4.0)
    assert sr.reason == "stub reason"
    assert sr.max_ray_disagreement_mm == 2.5
    # Honest "not applicable" defaults -- see the adapter's own docstring.
    assert sr.triangulation is None
    assert sr.n_cameras_used == 0
    assert sr.cameras_used is None
    assert sr.outlier_camera is None
    assert sr.alt_candidates_used is None


def test_engine_result_to_score_result_adapter_handles_missing_diagnostics_key():
    er = EngineResult(ok=False, sector=None, ring="outside", board_xy_mm=None, diagnostics={})
    sr = engine_result_to_score_result(er)
    assert sr.max_ray_disagreement_mm is None
