"""Engine interface -- EXACTLY the shape agreed in docs/ENGINES.md's
"Engine interface" section. An engine author needs to know only this
file; everything else (registration, scheduling, writing results) is the
framework's job.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from opendarts.pipeline import CameraCalibration, ScoreResult


@dataclass
class EngineResult:
    """One engine's answer for one thrown dart. Core fields mirror
    opendarts.pipeline.ScoreResult's own required fields (`ok`, `sector`,
    `ring`, `board_xy_mm`, `reason`) -- deliberately the same shape, so
    `Apollo` can wrap `ScoreResult` almost field-for-field. `diagnostics`
    is an OPEN bucket for anything engine-specific (today's
    `max_ray_disagreement_mm`/`per_ray_distance_mm`/etc. live there for
    `Apollo`; a different engine might put something else there
    entirely -- the framework never inspects it, purely opaque
    pass-through, matching docs/ENGINES.md's "the framework doesn't
    care" wording). `confidence` is optional (0-1); Talos fills it
    from mapping agreement. Other engines may leave it None. It does
    not change the scored bed.

    `duration_s`/`timed_out` are NOT set by the engine's own `score()` --
    they're filled in by opendarts.engines.dispatch, which is the only code
    that actually knows how long a call took or whether it hit the
    per-engine timeout. An engine returning an EngineResult itself always
    leaves these at their defaults (0.0 / False).
    """

    ok: bool
    sector: str | None
    ring: str | None
    board_xy_mm: tuple[float, float] | None
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)
    duration_s: float = 0.0
    timed_out: bool = False
    confidence: float | None = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "sector": self.sector,
            "ring": self.ring,
            "board_xy_mm": list(self.board_xy_mm) if self.board_xy_mm is not None else None,
            "reason": self.reason,
            "diagnostics": _normalize_diagnostics_null_collections(self.diagnostics),
            "duration_s": self.duration_s,
            "timed_out": self.timed_out,
            "confidence": self.confidence,
        }


def engine_result_from_dict(d: dict) -> "EngineResult":
    """The symmetric inverse of `EngineResult.to_dict()` -- lossless
    round trip (every field `to_dict()` writes has a matching read here,
    `board_xy_mm`'s list<->tuple conversion undone). Added 2026-08-27
    (perf task, see `opendarts.engines.zeus.engine`'s own module docstring,
    "duplicate sub-engine compute" section): `ZeusEngine.score()` already
    serializes each sub-engine's own `EngineResult` via `.to_dict()` into
    `diagnostics["sub_results"]` (it has to -- `diagnostics` must stay
    JSON-serializable, so it can never hold a raw `EngineResult` object
    directly). `opendarts.live.capture_daemon`'s reused-from-Zeus also-run
    path needs those SAME sub-results back as real `EngineResult` objects
    (matching `opendarts.engines.dispatch.dispatch_engines()`'s own
    `dict[str, EngineResult]` return shape, so the reuse path is a
    drop-in substitute, not a special case downstream callers need to
    know about) -- this is that reconstruction, not a re-derivation:
    since `d` is exactly `.to_dict()`'s own output, this is a pure,
    deterministic, information-preserving inverse, never a guess."""
    board_xy_mm = d.get("board_xy_mm")
    return EngineResult(
        ok=d["ok"],
        sector=d.get("sector"),
        ring=d.get("ring"),
        board_xy_mm=tuple(board_xy_mm) if board_xy_mm is not None else None,
        reason=d.get("reason", ""),
        diagnostics=d.get("diagnostics") or {},
        duration_s=d.get("duration_s", 0.0),
        timed_out=d.get("timed_out", False),
        confidence=d.get("confidence"),
    )


# The v2 package schema null-vs-[] rule (2026-08-27, second real gap: the
# 2026-08-27 "final round" fix -- see docs/DESIGN.md's dated entry -- only
# applied this normalization at TWO specific call sites
# (`opendarts.engines.apollo.engine.score_result_to_engine_result()` and
# `opendarts.capture.throw_package._score_result_to_dict()`, the top-level
# `result.json` rollup for the PRIMARY engine only). It never touched
# `other_engines.<Name>.diagnostics` for any engine dispatched as an
# also-run -- QA's own real measurement found Talos's ("Talos") entry
# still capable of carrying `null` for these same fields there.
#
# This is the single generic choke point EVERY engine's diagnostics
# passes through on the way to disk (`EngineResult.to_dict()`, called
# from both `save_throw_package()`'s primary-engine write and
# `write_other_engines_result()`'s also-run write) -- fixing it here
# closes the gap for every engine, present and future, not just the one
# QA happened to measure. `diagnostics` is deliberately documented above
# as an OPEN/opaque per-engine bucket the framework never inspects --
# this function respects that: it does NOT validate or reshape anything
# beyond the 3 specific well-known collection-typed keys QA's own rule
# names (`cameras_used`, `alt_candidates_used`, and the nested
# `triangulation.per_ray_distance_mm`), and ONLY when the key is
# PRESENT and explicitly `None`. It never fabricates a key that isn't
# there at all -- an engine that doesn't track `alt_candidates_used`
# (e.g. Talos's dominant "centerline_ray" path, confirmed by direct
# trace: that path's diagnostics dict never sets `cameras_used`/
# `alt_candidates_used` as keys at all, so this function is a correct
# no-op for it, exactly as it should be) keeps NOT having that key --
# genuinely different from having it and it being null.
def _normalize_diagnostics_null_collections(diagnostics: dict) -> dict:
    if not isinstance(diagnostics, dict):
        return diagnostics
    needs_top_copy = (
        "cameras_used" in diagnostics and diagnostics["cameras_used"] is None
    ) or (
        "alt_candidates_used" in diagnostics and diagnostics["alt_candidates_used"] is None
    )
    out = dict(diagnostics) if needs_top_copy else diagnostics
    if needs_top_copy:
        if "cameras_used" in out and out["cameras_used"] is None:
            out["cameras_used"] = []
        if "alt_candidates_used" in out and out["alt_candidates_used"] is None:
            out["alt_candidates_used"] = []
    tri = out.get("triangulation")
    if (
        isinstance(tri, dict)
        and "per_ray_distance_mm" in tri
        and tri["per_ray_distance_mm"] is None
    ):
        if out is diagnostics:
            out = dict(diagnostics)
        out["triangulation"] = {**tri, "per_ray_distance_mm": []}
    return out


class Engine(Protocol):
    """The two responsibilities an engine may implement. See
    docs/ENGINES.md's "Engine interface" section for the full contract
    each one is held to (pure-in-spirit, no camera access, no shared
    mutable state, no writing files itself, must not hang forever -- the
    "must not hang forever" part is enforced BY THE CALLER, via
    opendarts.engines.dispatch's per-engine timeout, not something an engine
    implementation needs to defend itself).

    Documentation only -- NOT used for isinstance checks anywhere in this
    package (calibrate() is optional per docs/ENGINES.md, so a Protocol
    isinstance check that requires every declared method would wrongly
    reject an engine that correctly omits calibrate(); see
    engine_has_custom_calibrate() below for the actual duck-typed check
    used instead)."""

    def score(
        self,
        bg_images: dict[int, np.ndarray],
        frame_images: dict[int, np.ndarray],
        calibration: dict[int, CameraCalibration],
    ) -> EngineResult:
        """Required. Called once per thrown dart. See module docstring
        and docs/ENGINES.md."""
        ...

    def calibrate(
        self, raw_frames: dict[int, list[np.ndarray]]
    ) -> dict[int, CameraCalibration]:
        """Optional -- most engines don't implement this at all (a class
        simply omitting the method is the correct way to opt out; see
        opendarts.engines.dispatch.engine_has_custom_calibrate(), which
        checks for the method's presence via hasattr rather than
        requiring every engine to implement a no-op). When absent, the
        engine uses the primary engine's already-solved calibration --
        zero extra work for most engines, per docs/ENGINES.md's
        "Calibration solve (optional)" section."""
        ...


def engine_result_to_score_result(result: EngineResult) -> ScoreResult:
    """Lossy-but-honest adapter, used ONLY when a NON-Apollo engine is
    configured as the live PRIMARY (see opendarts.live.capture_daemon.
    handle_ready_to_capture's own docstring -- an edge case; the real,
    default, byte-identical-to-today path is Apollo-as-primary, which
    never goes through this function at all). Exists purely so
    opendarts.capture.throw_package.save_throw_package()'s existing
    ScoreResult-shaped signature/schema keeps working no matter which
    engine is primary, rather than giving save_throw_package() a second,
    parallel signature.

    `triangulation`/`cameras_used`/`outlier_camera`/`alt_candidates_used`
    are specific to Apollo's own ray-triangulation internals and have
    no generic equivalent for an arbitrary engine -- left at their honest
    "not applicable" defaults (None) here rather than invented.
    `n_cameras_used` similarly has no generic engine-level equivalent --
    left at 0. `max_ray_disagreement_mm` is recovered from `diagnostics`
    only if the engine happens to have put one there under that exact
    key (Apollo always does, via
    opendarts.engines.apollo.score_result_to_engine_result -- so even
    routing Apollo through this adapter, which normal live operation
    never does, would still carry that one field through).
    """
    return ScoreResult(
        ok=result.ok,
        sector=result.sector,
        ring=result.ring,
        board_xy_mm=result.board_xy_mm,
        triangulation=None,
        n_cameras_used=0,
        reason=result.reason,
        max_ray_disagreement_mm=result.diagnostics.get("max_ray_disagreement_mm"),
        cameras_used=None,
        outlier_camera=None,
        alt_candidates_used=None,
    )


def engine_has_custom_calibrate(engine: object) -> bool:
    """True only if `engine` defines its own `calibrate()` -- used by
    callers deciding whether to bother invoking it vs. just reusing the
    primary engine's calibration (the default path per docs/ENGINES.md,
    "most won't implement this"). Protocol membership alone (isinstance
    against `Engine`) doesn't distinguish "has calibrate" from "doesn't"
    since `calibrate` is an optional Protocol method -- check for the
    concrete attribute instead."""
    return hasattr(engine, "calibrate") and callable(getattr(engine, "calibrate"))
