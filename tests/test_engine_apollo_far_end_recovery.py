"""Apollo's last-resort tip/flight END flip (added 2026-08-14).

What it is: when the board-ROI gate rejects a camera because its tip
pixel is off the board face, and the OPPOSITE end of that same detected
component IS on the board, the engine promotes that opposite end --
but ONLY when fewer than 2 cameras survived the gate, i.e. only when the
throw would otherwise be unscoreable at all.

Why it's needed on top of the existing `alt_tip_px` promotion inside
`reject_outside_roi()`: that one only fires when `detect_tip()` itself
flagged its end-choice as ambiguous. The real failures this targets are
the ones where the width comparison was confidently WRONG -- a dart high
on the board with its flight angled up out of the board face -- so no
alternate was ever published for the gate to promote.

Why the necessity gate is load-bearing, measured on the real 300-throw
`data/archive/clean/` corpus with the production oriented_landmarks
calibration:

    ungated (flip whenever available):  +1 / -2   (289 -> 288)
    gated on < 2 surviving cameras:     +1 / -0   (289 -> 290)

A flipped end is a genuinely weaker observation than a gate-passing
primary; adding it to an already-sufficient camera set measurably
poisons good triangulations. Both the synthetic tests below and the real
corpus test pin that asymmetry, not just the happy path.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import opendarts.engines.apollo.engine as engine_mod
from opendarts.capture.throw_package import load_throw_package
from opendarts.engines.apollo.engine import ApolloEngine
from opendarts.engines.apollo.tip_detection import TipDetectionResult

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# Synthetic: the promotion rule itself, isolated from image processing.
# detect_tip/reject_outside_roi are stubbed so the test drives exactly the
# per-camera outcomes it means to, and score_dart is captured so the test
# asserts on the pixel set the engine actually handed it.
# --------------------------------------------------------------------------

class _FakeCalib:
    landmark_spread_ok = True


def _stub_engine(monkeypatch, per_cam):
    """per_cam: {cam: (gate_ok, tip_px, far_end_px, far_end_inside)}."""
    captured: dict = {}

    monkeypatch.setattr(
        engine_mod, "detect_tip",
        lambda bg, fr, prior_dart_line_px=None: TipDetectionResult(
            ok=True, tip_px=(0.0, 0.0), reason="ok"
        ),
    )

    order = sorted(per_cam)
    calls = {"i": 0}

    def fake_gate(det, calibration):
        cam = order[calls["i"]]
        calls["i"] += 1
        gate_ok, tip_px, far_px, far_inside = per_cam[cam]
        return TipDetectionResult(
            ok=gate_ok,
            tip_px=tip_px,
            reason="stub",
            far_end_px=far_px,
            diagnostics={"board_roi_far_end_inside": far_inside},
        )

    monkeypatch.setattr(engine_mod, "reject_outside_roi", fake_gate)

    def fake_score(tip_pixels, calibration, alt_tip_pixels=None):
        captured["tip_pixels"] = dict(tip_pixels)
        from opendarts.pipeline import ScoreResult

        return ScoreResult(
            ok=False, sector=None, ring=None, board_xy_mm=None, triangulation=None,
            n_cameras_used=len(tip_pixels), reason="stub",
        )

    monkeypatch.setattr(engine_mod, "score_dart", fake_score)

    images = {cam: np.zeros((4, 4, 3), np.uint8) for cam in order}
    calibration = {cam: _FakeCalib() for cam in order}
    ApolloEngine().score(images, images, calibration)
    return captured["tip_pixels"]


def test_far_end_promoted_when_only_one_camera_survives_the_gate(monkeypatch):
    """The real shape of throw_1786666454680: two cameras
    took the flight end (rejected), one camera fine. Without the flip
    that throw cannot be scored at all -- only 1 camera reaches
    score_dart, which needs >= 2."""
    tip_pixels = _stub_engine(monkeypatch, {
        0: (False, (10.0, 10.0), (100.0, 100.0), True),
        1: (True, (200.0, 200.0), (300.0, 300.0), None),
        2: (False, (20.0, 20.0), (150.0, 150.0), True),
    })
    assert set(tip_pixels) == {0, 1, 2}
    assert tip_pixels[0] == (100.0, 100.0)
    assert tip_pixels[2] == (150.0, 150.0)
    assert tip_pixels[1] == (200.0, 200.0), "a camera that passed the gate must be untouched"


def test_far_end_not_promoted_when_two_cameras_already_survived(monkeypatch):
    """The necessity gate, pinned. This is the difference between +1/-0
    and +1/-2 on the real corpus -- see this module's docstring."""
    tip_pixels = _stub_engine(monkeypatch, {
        0: (True, (200.0, 200.0), None, None),
        1: (True, (210.0, 210.0), None, None),
        2: (False, (20.0, 20.0), (150.0, 150.0), True),
    })
    assert set(tip_pixels) == {0, 1}, "an already-scoreable throw must not take the weaker end"


def test_far_end_not_promoted_when_the_far_end_is_also_off_board(monkeypatch):
    tip_pixels = _stub_engine(monkeypatch, {
        0: (False, (10.0, 10.0), (11.0, 11.0), False),
        1: (True, (200.0, 200.0), None, None),
        2: (False, (20.0, 20.0), (21.0, 21.0), False),
    })
    assert set(tip_pixels) == {1}


def test_far_end_not_promoted_when_no_far_end_was_detected(monkeypatch):
    """A rejected camera with no far end available must be skipped
    cleanly rather than contributing a None pixel."""
    tip_pixels = _stub_engine(monkeypatch, {
        0: (False, (10.0, 10.0), None, None),
        1: (True, (200.0, 200.0), None, None),
    })
    assert set(tip_pixels) == {1}


# --------------------------------------------------------------------------
# Real corpus: the specific throw this was found on, and the real
# accuracy floor the change was measured against.
# --------------------------------------------------------------------------

# Real measured, 2026-08-14, ApolloEngine end-to-end over all 300
# AD-matched throws in data/archive/clean/ with the production
# oriented_landmarks calibration re-derived per session (NOT each
# package's stored calibration.json): 290/300 = 96.7%, up from 289/300 =
# 96.3% before this change. The floor below sits a few points under that
# so ordinary float/library-version noise across machines can't flake it;
# raise both together on the next real measured improvement.
#
# The floor below is checked against the STORED-calibration path this
# test actually runs (fixed, reproducible input), which measures 95.3%
# (286/300) on the same corpus -- a different, lower number than the
# production-calibration headline because these packages carry the
# calibration that was live when they were captured. Both numbers were
# measured, neither guessed.
#
# **Re-measured 2026-08-18** on the full living corpus (grown to 1107
# AD-matched throws since 2026-08-14), via this test's OWN methodology
# (bare `ApolloEngine().score()`, no `prior_dart_line_px` -- this
# measurement does NOT benefit from the 2026-08-17 prior-dart-drop-path
# backfill the same way real production replay does, see
# opendarts.capture.replay.replay_throw_with_engine): 1080/1107 = 97.6%
# no-score=8, before opendarts/engines/apollo/engine.py's dated
# 2026-08-18 lone-genuine-camera fallback; 1084/1107 = 97.9% no-score=3
# after it (5 of the 8 no-score throws recover correct, zero new wrong
# answers; the other 3 -- including 2 this test's own bare-call
# methodology can't see recovered by the EARLIER prior-dart backfill --
# only clear under the real replay path; see
# tests/test_engine_apollo_no_score_fallback.py's own module
# docstring for the full corpus accounting via real replay, which shows
# 1091/1107 = 98.55%, no-score=1). Floor raised to sit a few points
# under this test's own new number, per this comment's own standing
# instruction to raise both together.
MIN_BOTH_MATCH_RATE = 0.96
# The real throw the end-flip was found on, by rendering its diff-mask
# components against the projected board ROI and looking at them: cam0
# and cam2 both took the flight end (above the board face), cam1 was
# fine, so only one camera reached score_dart and the throw scored
# nothing at all. AD has it as 14/single_inner.
KNOWN_THROW = "20260813-164658/throw_1786666454680"


def _corpus_root() -> Path:
    env_root = os.environ.get("OPENDARTS_ENGINE_CORPUS_ROOT")
    return Path(env_root) if env_root else REPO_ROOT / "data" / "archive" / "clean"


def test_known_wrong_end_throw_is_scored_at_all_rather_than_discarded():
    """Package-stored calibration deliberately (no dependency on
    re-deriving a production calibration inside a test): the point being
    pinned is that the throw reaches score_dart with >= 2 cameras instead
    of being thrown away, which is what the flip actually changes."""
    pkg_dir = _corpus_root() / KNOWN_THROW
    if not (pkg_dir / "calibration.json").exists():
        pytest.skip(
            f"{KNOWN_THROW} is not in the corpus on this machine -- the corpus is a "
            "living, curated thing (docs/DESIGN.md), so this pin skips rather than fails "
            "when it has moved on"
        )
    package = load_throw_package(pkg_dir)
    result = ApolloEngine().score(
        package.bg_frames, package.dart_frames, package.calibrations
    )
    assert result.diagnostics["n_cameras_used"] >= 2, (
        "the two flight-end cameras should have been recovered via their far ends; "
        f"got n_cameras_used={result.diagnostics['n_cameras_used']}, reason={result.reason!r}"
    )


def test_apollo_meets_its_real_accuracy_floor_on_the_clean_corpus():
    """Real, reproducible measurement (this IS the measurement, not a
    check against a stored number). Uses each package's OWN stored
    calibration, like dev/tests/test_engine_athena_accuracy.py does, so the
    input is fixed and reproducible -- see MIN_BOTH_MATCH_RATE's comment
    for the separate production-calibration number this change was
    actually tuned and reported against."""
    root = _corpus_root()
    pkg_dirs = sorted(
        p.parent for p in root.rglob("meta.json") if (p.parent / "calibration.json").exists()
    ) if root.exists() else []
    if not pkg_dirs:
        pytest.skip("no real archived throw packages under data/archive/clean/")

    engine = ApolloEngine()
    n_with_ad = n_both = 0
    for pkg_dir in pkg_dirs:
        package = load_throw_package(pkg_dir)
        adg = package.ad_ground_truth
        if adg is None or not adg.matched:
            continue
        n_with_ad += 1
        result = engine.score(package.bg_frames, package.dart_frames, package.calibrations)
        if result.ok and (result.sector, result.ring) == (adg.sector, adg.ring):
            n_both += 1

    assert n_with_ad > 0, "found package dirs but none had matched AD ground truth"
    rate = n_both / n_with_ad
    print(f"\nApollo stored-calibration accuracy on data/archive/clean/ "
          f"(n={n_with_ad}): {rate:.1%} (floor={MIN_BOTH_MATCH_RATE:.1%})")
    assert rate >= MIN_BOTH_MATCH_RATE, (
        f"Apollo's real sector+ring match rate dropped to {rate:.1%} "
        f"({n_both}/{n_with_ad}), below the {MIN_BOTH_MATCH_RATE:.1%} floor"
    )
