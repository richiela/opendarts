"""Argus -- registry wiring, the point-selection rule
(`choose_board_point`), the token->(sector, ring) map
(`token_to_sector_ring`), end-to-end behavior against a real package, and
never-crash guarantees. Style matches tests/test_engine_zeus.py: cheap,
synthetic/hand-built inputs for the pure logic, one real-corpus smoke
test for the end-to-end wiring.

2026-10-08: Argus is Zeus's 5th voter -- see
opendarts/engines/zeus/engine.py's own module docstring and
opendarts/engines/argus/engine.py's module docstring for the design.
This file also pins the two capability-forwarding seams Argus adds
(`visit_bg_images`, through Zeus and through dispatch_engines) and the
visit-context lookup (`opendarts.engines.argus.visit_context`) that
feeds it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from opendarts.capture.replay import replay_throw_with_engine
from opendarts.engines.apollo import ApolloEngine
from opendarts.engines.apollo.prior_dart_context import engine_accepts_prior_dart_line_px
from opendarts.engines.argus import ArgusEngine
from opendarts.engines.argus.engine import choose_board_point, token_to_sector_ring
from opendarts.engines.argus.visit_context import (
    CachedVisitStart,
    engine_accepts_visit_bg_images,
    find_visit_bg_images,
)
from opendarts.engines.base import EngineResult
from opendarts.engines.dispatch import dispatch_engines
from opendarts.engines.registry import (
    DEFAULT_ALSO_RUN,
    ENGINES,
    engine_names,
    get_engine,
    is_registered,
)
from opendarts.engines.zeus import ZEUS_SUB_ENGINE_NAMES, ZeusEngine
from opendarts.engines.zeus.engine import MIN_SUB_ENGINES_TO_VOTE
from opendarts.pipeline import CameraCalibration


def _ok(sector, ring, xy=(1.0, 2.0)) -> EngineResult:
    return EngineResult(ok=True, sector=sector, ring=ring, board_xy_mm=xy)


def _miss(reason: str = "no dart found") -> EngineResult:
    return EngineResult(ok=False, sector=None, ring=None, board_xy_mm=None, reason=reason)


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #

def test_argus_is_registered():
    assert "Argus" in engine_names()
    assert is_registered("Argus")
    engine = get_engine("Argus")
    assert isinstance(engine, ArgusEngine)
    assert ENGINES["Argus"] is engine
    assert engine.name == "Argus"


def test_zeus_sub_engine_names_has_five_with_argus_last():
    assert ZEUS_SUB_ENGINE_NAMES == ("Apollo", "Talos", "Athena", "Ares", "Argus")


def test_min_sub_engines_to_vote_is_still_three():
    assert MIN_SUB_ENGINES_TO_VOTE == 3


def test_default_also_run_includes_argus_and_matches_zeus_sub_engines():
    assert "Argus" in DEFAULT_ALSO_RUN
    assert DEFAULT_ALSO_RUN == ZEUS_SUB_ENGINE_NAMES


def test_argus_accepts_prior_dart_line_px():
    assert engine_accepts_prior_dart_line_px(ArgusEngine()) is True


def test_visit_bg_images_capability_check_argus_and_zeus_true_apollo_false():
    assert engine_accepts_visit_bg_images(ArgusEngine()) is True
    assert engine_accepts_visit_bg_images(ZeusEngine()) is True
    assert engine_accepts_visit_bg_images(ApolloEngine()) is False


# --------------------------------------------------------------------------- #
# choose_board_point -- the point-selection rule, hand-built decision/cams
# --------------------------------------------------------------------------- #

def test_choose_board_point_fused_point_whose_label_matches_the_call_wins():
    # label(0, 50, off=0) == "S20" (r=50mm, inside the treble ring, at the
    # sector-20 angle -- theta=atan2(0, 50)=0deg) -- matches the call, so
    # the fused point is used directly, no camera consulted at all.
    decision = {"fused_xy": (0.0, 50.0), "cams": {}}
    point, method, cam = choose_board_point("S20", decision, {}, 0.0)
    assert point == (0.0, 50.0)
    assert method == "fused"
    assert cam is None


def test_choose_board_point_fused_disagrees_uses_the_most_confident_agreeing_camera():
    # label(0, 103, off=0) == "T20" -- the fused point's own bed does NOT
    # match the call ("S5"), so Argus falls through to the cameras. Camera
    # "0" is the MOST confident overall but calls "T20" (disagrees);
    # camera "1" is less confident but is the one that actually agrees
    # with the call -- it must win, not the merely-most-confident one.
    decision = {
        "fused_xy": (0.0, 103.0),
        "cams": {
            "0": {"confidence": 0.9, "call_token": "T20"},
            "1": {"confidence": 0.5, "call_token": "S5"},
        },
    }
    cams = {
        "0": {"board_xy": (1.0, 2.0)},
        "1": {"board_xy": (3.0, 4.0)},
    }
    point, method, cam = choose_board_point("S5", decision, cams, 0.0)
    assert point == (3.0, 4.0)
    assert method == "camera_agrees"
    assert cam == "1"


def test_choose_board_point_no_fused_point_same_rule_as_disagreement():
    decision = {
        "fused_xy": None,
        "cams": {
            "0": {"confidence": 0.9, "call_token": "T20"},
            "1": {"confidence": 0.5, "call_token": "S5"},
        },
    }
    cams = {
        "0": {"board_xy": (1.0, 2.0)},
        "1": {"board_xy": (3.0, 4.0)},
    }
    point, method, cam = choose_board_point("S5", decision, cams, 0.0)
    assert point == (3.0, 4.0)
    assert method == "camera_agrees"
    assert cam == "1"


def test_choose_board_point_no_camera_agrees_falls_back_to_the_most_confident_camera():
    decision = {
        "fused_xy": None,
        "cams": {
            "0": {"confidence": 0.9, "call_token": "T20"},
            "1": {"confidence": 0.5, "call_token": "D5"},
        },
    }
    cams = {
        "0": {"board_xy": (1.0, 2.0)},
        "1": {"board_xy": (3.0, 4.0)},
    }
    # Neither camera's own call is "S9" -- fall back to the most
    # confident camera's own point regardless of what it called.
    point, method, cam = choose_board_point("S9", decision, cams, 0.0)
    assert point == (1.0, 2.0)
    assert method == "camera_top"
    assert cam == "0"


def test_choose_board_point_nothing_usable_returns_none():
    point, method, cam = choose_board_point("S9", {"fused_xy": None, "cams": {}}, {}, 0.0)
    assert point is None
    assert method == "none"
    assert cam is None


# --------------------------------------------------------------------------- #
# token_to_sector_ring -- Argus token -> OD's (sector, ring) vocabulary
# --------------------------------------------------------------------------- #

def test_token_to_sector_ring_bull_outer_bull_and_out():
    assert token_to_sector_ring("DB", None) == (None, "bull")
    assert token_to_sector_ring("OB", None) == (None, "outer_bull")
    assert token_to_sector_ring("OUT", None) == (None, "outside")


def test_token_to_sector_ring_treble_and_double():
    assert token_to_sector_ring("T20", (0.0, 103.0)) == ("20", "treble")
    assert token_to_sector_ring("D3", (0.0, -166.0)) == ("3", "double")


def test_token_to_sector_ring_single_splits_inner_vs_outer_by_radius():
    # _SINGLE_SPLIT_MM = (TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2
    # = (99 + 107) / 2 = 103mm.
    assert token_to_sector_ring("S20", (0.0, 50.0)) == ("20", "single_inner")
    assert token_to_sector_ring("S20", (0.0, 130.0)) == ("20", "single_outer")


# --------------------------------------------------------------------------- #
# End-to-end on one real package (skipped cleanly if the corpus isn't here)
# --------------------------------------------------------------------------- #

def _real_packages_root() -> "Path | None":
    """A directory of real throw packages -- same convention
    tests/test_package_clip_modes.py's own `_real_packages_root()` uses
    (`OPENDARTS_REAL_PACKAGES`, else this checkout's own `data/packages`)."""
    env = os.environ.get("OPENDARTS_REAL_PACKAGES")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).resolve().parent.parent / "data" / "packages")
    for root in candidates:
        if any(root.glob("*/*/meta.json")):
            return root
    return None


def _find_dart_2_package(root: "Path | None") -> "Path | None":
    """One real package that is the SECOND dart of its visit
    (`visit_index == 1`) -- exercises Argus's `visit_bg_images` path for
    real, not just the no-prior-context case."""
    if root is None:
        return None
    for session in sorted(p for p in root.iterdir() if p.is_dir()):
        for pkg in sorted(p for p in session.iterdir() if (p / "meta.json").is_file()):
            try:
                meta = json.loads((pkg / "meta.json").read_text())
            except (OSError, ValueError):
                continue
            if meta.get("visit_index") == 1 and meta.get("visit_id"):
                return pkg
    return None


_REAL_PACKAGES_ROOT = _real_packages_root()
_ARGUS_DART_2_PACKAGE = _find_dart_2_package(_REAL_PACKAGES_ROOT)


@pytest.mark.skipif(
    _ARGUS_DART_2_PACKAGE is None,
    reason=f"no real dart-2 package found (root={_REAL_PACKAGES_ROOT})",
)
def test_argus_end_to_end_on_a_real_dart_2_package():
    result = replay_throw_with_engine(_ARGUS_DART_2_PACKAGE, "Argus")

    assert result.ok is True
    assert result.ring in (
        "bull", "outer_bull", "outside",
        "single_inner", "single_outer", "treble", "double",
    )
    assert result.board_xy_mm is not None

    diagnostics = result.diagnostics
    assert "method" in diagnostics
    assert diagnostics["visit_bg_used"] is True

    per_camera = diagnostics["per_camera"]
    ok_cams = {c: info for c, info in per_camera.items() if info.get("ok")}
    assert ok_cams, "expected at least one camera's diagnostics to be ok"
    for info in ok_cams.values():
        assert info["board_xy_mm"] is not None
        assert "call" in info
        assert "confidence" in info


# --------------------------------------------------------------------------- #
# Never crash
# --------------------------------------------------------------------------- #

def test_never_raises_when_extract_features_raises(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("feature boom")

    monkeypatch.setattr("opendarts.engines.argus.engine.extract_features", boom)
    result = ArgusEngine().score(
        {0: np.zeros((4, 4, 3), np.uint8)}, {0: np.zeros((4, 4, 3), np.uint8)}, {0: object()},
    )
    assert result.ok is False
    assert "RuntimeError" in result.reason
    assert "feature boom" in result.reason


def test_never_raises_on_empty_dicts():
    result = ArgusEngine().score({}, {}, {})
    assert result.ok is False


def _dummy_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rvec=np.zeros(3),
        tvec=np.array([0.0, 0.0, 1000.0]),
    )


def test_never_raises_on_mismatched_bg_frame_shapes():
    bg = {0: np.zeros((10, 10, 3), np.uint8)}
    frame = {0: np.zeros((20, 20, 3), np.uint8)} # deliberately mismatched
    result = ArgusEngine().score(bg, frame, {0: _dummy_calibration()})
    assert isinstance(result.ok, bool)


def test_never_raises_on_garbage_arrays():
    rng = np.random.default_rng(0)
    bg = {0: rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)}
    frame = {0: rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)}
    result = ArgusEngine().score(bg, frame, {0: _dummy_calibration()})
    assert isinstance(result.ok, bool)


def test_never_raises_with_calibration_object_missing_attributes():
    class _IncompleteCalibration:
        """Has SOME of CameraCalibration's attributes, not all -- a
        realistic partial-object shape, not just a bare object()."""

        def __init__(self):
            self.camera_matrix = np.eye(3)
            # Deliberately no rvec/tvec/dist_coeffs.

    bg = {0: np.zeros((8, 8, 3), np.uint8)}
    frame = {0: np.zeros((8, 8, 3), np.uint8)}
    result = ArgusEngine().score(bg, frame, {0: _IncompleteCalibration()})
    assert isinstance(result.ok, bool)
    assert result.ok is False


# --------------------------------------------------------------------------- #
# visit_context -- find_visit_bg_images
# --------------------------------------------------------------------------- #

def test_find_visit_bg_images_none_for_first_dart_of_a_visit(tmp_path):
    assert find_visit_bg_images(tmp_path, "v1", 0) is None


def test_find_visit_bg_images_none_when_visit_index_is_none():
    assert find_visit_bg_images(Path("/nonexistent"), "v1", None) is None


def test_find_visit_bg_images_none_when_visit_id_is_none(tmp_path):
    assert find_visit_bg_images(tmp_path, None, 1) is None


def test_find_visit_bg_images_uses_the_cache_without_touching_disk():
    cached_frames = {0: np.full((2, 2, 3), 9, np.uint8)}
    cached = CachedVisitStart(visit_id="v1", bg_frames=cached_frames)
    # A session dir that does not exist -- if the cache check didn't
    # short-circuit before any disk access, this would have to resolve
    # to None (a missing dir), not the cached frames.
    missing_dir = Path("/definitely/does/not/exist/argus-visit-context-test")
    result = find_visit_bg_images(missing_dir, "v1", 1, cached=cached)
    assert set(result) == {0}
    assert np.array_equal(result[0], cached_frames[0])


def test_find_visit_bg_images_falls_back_to_disk_when_cache_is_another_visit(
    tmp_path, monkeypatch,
):
    session_dir = tmp_path / "sess"
    dart0_dir = session_dir / "sess-001-S20"
    dart0_dir.mkdir(parents=True)
    (dart0_dir / "meta.json").write_text(json.dumps({"visit_id": "v1", "visit_index": 0}))

    disk_frames = {0: np.full((3, 3, 3), 5, np.uint8)}

    class _FakePackage:
        bg_frames = disk_frames

    def fake_load_throw_package(path):
        assert path == dart0_dir
        return _FakePackage()

    monkeypatch.setattr(
        "opendarts.capture.throw_package.load_throw_package", fake_load_throw_package,
    )

    cached = CachedVisitStart(
        visit_id="some_other_visit", bg_frames={0: np.zeros((1, 1, 3), np.uint8)},
    )
    result = find_visit_bg_images(session_dir, "v1", 1, cached=cached)
    assert set(result) == {0}
    assert np.array_equal(result[0], disk_frames[0])


def test_find_visit_bg_images_none_when_no_dart_0_present(tmp_path):
    session_dir = tmp_path / "sess"
    dart1_dir = session_dir / "sess-002-S4"
    dart1_dir.mkdir(parents=True)
    (dart1_dir / "meta.json").write_text(json.dumps({"visit_id": "v1", "visit_index": 1}))
    assert find_visit_bg_images(session_dir, "v1", 1) is None


# --------------------------------------------------------------------------- #
# Zeus forwards visit_bg_images only to sub-engines that declare it
# --------------------------------------------------------------------------- #

class _PlainFakeEngine:
    """Legacy 3-argument score() -- must not be handed visit_bg_images,
    and must not break just because Zeus was called with it."""

    def __init__(self, result: EngineResult) -> None:
        self._result = result

    def score(self, bg_images, frame_images, calibration) -> EngineResult:
        return self._result


class _VisitBgSpyEngine:
    """Records whatever visit_bg_images it was actually called with."""

    def __init__(self, result: EngineResult) -> None:
        self._result = result
        self.seen: list = []

    def score(self, bg_images, frame_images, calibration, *, visit_bg_images=None) -> EngineResult:
        self.seen.append(visit_bg_images)
        return self._result


def test_zeus_forwards_visit_bg_images_only_to_the_sub_engine_declaring_it(monkeypatch):
    spy = _VisitBgSpyEngine(_ok("20", "double"))
    fakes = {
        "Apollo": _PlainFakeEngine(_ok("20", "double")),
        "Talos": _PlainFakeEngine(_ok("20", "double")),
        "Athena": _PlainFakeEngine(_ok("20", "double")),
        "Ares": _PlainFakeEngine(_ok("20", "double")),
        "Argus": spy,
    }
    monkeypatch.setattr("opendarts.engines.registry.get_engine", lambda name: fakes[name])

    visit_bg_images = {0: np.zeros((2, 2, 3), np.uint8)}
    result = ZeusEngine().score({}, {}, {}, visit_bg_images=visit_bg_images)

    assert result.ok is True
    assert result.diagnostics["agreement"] == "unanimous"
    # Only the declaring (Argus) fake actually received it.
    assert spy.seen == [visit_bg_images]


# --------------------------------------------------------------------------- #
# dispatch_engines forwards visit_bg_images to a declaring engine
# --------------------------------------------------------------------------- #

class _DispatchVisitBgSpy:
    def __init__(self) -> None:
        self.seen: list = []

    def score(self, bg_images, frame_images, calibration, *, visit_bg_images=None) -> EngineResult:
        self.seen.append(visit_bg_images)
        return EngineResult(ok=True, sector="1", ring="single_inner", board_xy_mm=(1.0, 1.0))


class _DispatchPlainEngine:
    def score(self, bg_images, frame_images, calibration) -> EngineResult:
        return EngineResult(ok=True, sector="2", ring="single_inner", board_xy_mm=(2.0, 2.0))


def test_dispatch_engines_forwards_visit_bg_images_to_a_declaring_engine():
    spy = _DispatchVisitBgSpy()
    plain = _DispatchPlainEngine()
    visit_bg_images = {0: np.zeros((2, 2, 3), np.uint8)}

    results = dispatch_engines(
        {}, {}, {}, ["Spy", "Plain"], timeout_s=1.0,
        registry={"Spy": spy, "Plain": plain},
        visit_bg_images=visit_bg_images,
    )

    assert results["Spy"].ok is True
    assert results["Plain"].ok is True
    assert spy.seen == [visit_bg_images]


# --------------------------------------------------------------------------- #
# Cross-view fallback: cameras that do not fuse are re-read from Apollo's
# cross-view blob pick (the live S16 third dart Argus called OUT)
# --------------------------------------------------------------------------- #

def _cam_feat(x, y, token):
    return {
        "ok": True, "tip_px": [100.0, 100.0], "board_xy": [x, y],
        "token": token, "paint_token": token, "margin_mm": 5.0, "mm_per_px": 0.5,
        "width_ratio": 0.3, "elongation_ratio": 4.0, "texture_tiebreak_used": False,
        "contam_suspected": False, "entry_occluded": False, "clarity_z": 5.0,
        "paint_margin_ratio": 0.5, "agree_mm": 1.0, "needle_w_mm": 2.0,
        "jac_u": [0.5, 0.0], "jac_v": [0.0, 0.5],
    }


# Three off-board points hundreds of mm apart (no consistent pair), as in
# the live miss; and the same dart read from the right blobs.
_SCATTERED = {"0": _cam_feat(-170.4, -94.3, "OUT"), "1": _cam_feat(84.9, 287.6, "OUT"),
              "2": _cam_feat(-197.2, 100.0, "OUT")}
_ON_DART = {"0": _cam_feat(-58.5, -32.0, "S16"), "1": _cam_feat(-57.1, -33.5, "S16"),
            "2": _cam_feat(-57.7, -32.7, "S16")}


def _fake_extract(calls):
    def extract(bg, fr, cal, *, prior_dart_line_px=None, visit_bg_images=None, tip_overrides=None):
        calls.append(tip_overrides)
        return {"cams": dict(_ON_DART if tip_overrides is not None else _SCATTERED),
                "parallax": {"method": "detect_tip"}}
    return extract


def test_cameras_that_do_not_fuse_are_reread_from_the_cross_view_pick(monkeypatch):
    import opendarts.engines.argus.engine as argus_engine
    calls = []
    monkeypatch.setattr(argus_engine, "extract_features", _fake_extract(calls))
    pick = {0: ((655.3, 211.0), (645.5, 137.3)), 1: ((500.5, 326.1), (555.7, 41.4)),
            2: ((752.3, 274.1), (657.0, 3.7))}
    monkeypatch.setattr(argus_engine, "_cross_view_overrides", lambda bg, fr, cal: pick)
    res = ArgusEngine().score({}, {}, {})
    assert res.ok
    assert (res.sector, res.ring) == ("16", "single_inner")
    assert res.diagnostics["cross_view"] == "used"
    assert calls == [None, pick]


def test_no_cross_view_pick_keeps_argus_own_reading(monkeypatch):
    # A real miss looks the same to Argus (nothing consistent on the
    # board) and the pick finds no dart either: the call stays OUT.
    import opendarts.engines.argus.engine as argus_engine
    calls = []
    monkeypatch.setattr(argus_engine, "extract_features", _fake_extract(calls))
    monkeypatch.setattr(argus_engine, "_cross_view_overrides", lambda bg, fr, cal: None)
    res = ArgusEngine().score({}, {}, {})
    assert res.ok and res.ring == "outside"
    assert res.diagnostics["cross_view"] == "none_found"
    assert calls == [None]


def test_cross_view_pick_not_consulted_when_cameras_fuse(monkeypatch):
    import opendarts.engines.argus.engine as argus_engine

    def extract(bg, fr, cal, **kw):
        return {"cams": dict(_ON_DART), "parallax": {"method": "parallax"}}

    def must_not_run(*a):
        raise AssertionError("cross-view pick consulted although the cameras fuse")

    monkeypatch.setattr(argus_engine, "extract_features", extract)
    monkeypatch.setattr(argus_engine, "_cross_view_overrides", must_not_run)
    res = ArgusEngine().score({}, {}, {})
    assert res.ok and res.diagnostics["cross_view"] == "not_needed"


def test_live_third_dart_called_out_now_scores_s16():
    """The real package (third dart of a visit; Argus called OUT live,
    every other engine but one said S16)."""
    root = _real_packages_root()
    pkg = None if root is None else root / "20261008-234739" / "20261008-234739-027-S16"
    if pkg is None or not (pkg / "meta.json").is_file():
        pytest.skip("package 20261008-234739-027-S16 not available")
    res = replay_throw_with_engine(pkg, "Argus")
    assert res.ok
    assert (res.sector, res.ring) == ("16", "single_inner")
    assert res.diagnostics["cross_view"] == "used"
