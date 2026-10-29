"""Argus -- the visual hybrid scorer. The registry entry named "Argus",
Zeus's fifth voter.

Where the other engines turn each camera's view into a ray or a line
and intersect them, Argus asks each camera what it SEES at the needle
tip and lets the board geometry referee:

1. `features` -- per camera: Apollo's own `detect_tip` (with the prior
   dart's line), the needle ends chosen by board-plane parallax across
   cameras, then the bed under the tip by calibration map and by board
   paint colour, the needle's visibility, width and clarity, whether its
   entry sits on an earlier dart of this visit, and the local
   pixel-to-board Jacobian.
2. `decision` -- a confidence-weighted visual vote, refereed by the
   cameras' fused board position (see that module's docstring).

Inputs beyond the standard three, both optional and keyword-only, both
supplied the way every other engine's extra context is (live capture
and replay share one lookup each, so a replay reaches the live call):

- `prior_dart_line_px` -- the previous dart's line per camera, exactly
  what Apollo receives (`opendarts.engines.apollo.prior_dart_context`).
- `visit_bg_images` -- the visit's empty board per camera: the first
  dart's background (`opendarts.engines.argus.visit_context`).

Without them Argus still scores; it just cannot tell an entry hidden by
an earlier dart from a visible one.

`board_xy_mm` is always set when ok, so the dashboard can draw Argus's
dot beside its peers (most useful exactly when it disagrees):

- the fused point, when there is one and its bed is Argus's call;
- else the board point of the most confident camera whose own call is
  Argus's call;
- else the most confident camera's board point.

`diagnostics["method"]` says which; `diagnostics["per_camera"]` holds
every camera's board point, call and confidence.

Cross-view fallback (2026-10-09). When no two of Argus's own camera
points are consistent on the board, the cameras are re-read from Apollo's
cross-view blob pick (`opendarts.engines.apollo.cross_view`): the blob
and end in each camera that fits the other views in 3D. The live third
dart that prompted this (S16, darts already in the board) had one camera
on its flight end, one on an earlier dart's flight and one on a speck
below the board; three points hundreds of mm apart, all off the board,
and Argus called OUT. The pick found the dart in all three. Real misses
look the same to Argus (no consistent point) and the pick finds no dart
on the board for them, so they stay OUT. `diagnostics["cross_view"]`:
`not_needed`, `used`, or `none_found`.

Measured on the 4,087-dart truth corpus (replay, frozen parameters):
as Zeus's fifth voter, with Apollo keeping the tie-break, Zeus's misses
go 5 -> 1 (4 fixed, 0 broken); all four fixes were 2-2 ties among the
other four engines.
"""
from __future__ import annotations

import math

import numpy as np

from opendarts.engines.apollo.cross_view import select_blobs
from opendarts.engines.argus.decision import PARAMS, decide, label
from opendarts.engines.argus.features import extract_features
from opendarts.engines.base import EngineResult
from opendarts.geometry.board import TREBLE_INNER_RADIUS_MM, TREBLE_OUTER_RADIUS_MM
from opendarts.pipeline import CameraCalibration

# A single call ("S<n>") is split into OD's single_inner / single_outer by
# the radius of the board point Argus reports: inside the middle of the
# treble ring is the inner single. Any split inside the treble ring gives
# the same answer for a point whose own bed is a single; the middle is the
# fair choice for a fallback point that sits in the treble itself.
_SINGLE_SPLIT_MM = (TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2.0


def token_to_sector_ring(token: str, board_xy_mm: tuple[float, float] | None) -> tuple[str | None, str]:
    """Argus token -> OD's (sector, ring) vocabulary."""
    if token == "DB":
        return None, "bull"
    if token == "OB":
        return None, "outer_bull"
    if token == "OUT":
        return None, "outside"
    kind, number = token[0], token[1:]
    if kind == "T":
        return number, "treble"
    if kind == "D":
        return number, "double"
    if kind == "S":
        r = math.hypot(*board_xy_mm) if board_xy_mm is not None else 0.0
        return number, ("single_inner" if r < _SINGLE_SPLIT_MM else "single_outer")
    raise ValueError(f"unknown Argus token {token!r}")


def choose_board_point(call: str, decision: dict, cams: dict, off: float):
    """(board_xy_mm, method, camera) for Argus's call -- see the module
    docstring for the rule. `cams` are the per-camera features,
    `decision["cams"]` the per-camera confidence and call."""
    fused = decision.get("fused_xy")
    if fused is not None and label(fused[0], fused[1], off) == call:
        return (float(fused[0]), float(fused[1])), "fused", None
    ranked = sorted(
        ((c, info) for c, info in (decision.get("cams") or {}).items()
         if cams.get(c, {}).get("board_xy") is not None),
        key=lambda kv: -kv[1]["confidence"],
    )
    for c, info in ranked:
        if info["call_token"] == call:
            x, y = cams[c]["board_xy"]
            return (float(x), float(y)), "camera_agrees", c
    if ranked:
        c = ranked[0][0]
        x, y = cams[c]["board_xy"]
        return (float(x), float(y)), "camera_top", c
    return None, "none", None


def _r(v, nd=3):
    return None if v is None else round(float(v), nd)


def _per_camera_diag(cams: dict, decision: dict) -> dict:
    out = {}
    for c, feat in cams.items():
        if not feat.get("ok"):
            out[c] = {"ok": False, "error": feat.get("error")}
            continue
        info = (decision.get("cams") or {}).get(c, {})
        bx, by = feat["board_xy"]
        out[c] = {
            "ok": True,
            "board_xy_mm": [_r(bx, 2), _r(by, 2)],
            "tip_px": [_r(v, 1) for v in feat["tip_px"]],
            "call": info.get("call_token"),
            "confidence": _r(info.get("confidence")),
            "map_call": feat.get("token"),
            "paint_call": feat.get("paint_token"),
            "entry_visible": info.get("entry_visible"),
            "entry_occluded": feat.get("entry_occluded"),
            "prior_dart_suspected": feat.get("contam_suspected"),
            "needle_w_mm": _r(feat.get("needle_w_mm"), 2),
        }
    return out


def _cross_view_overrides(bg_images, frame_images, calibration):
    """{camera: (tip, far) or None} from Apollo's cross-view blob pick, or
    None when no combination of blobs is consistent in 3D."""
    cams = [c for c in sorted(set(bg_images) & set(frame_images) & set(calibration))
            if bg_images[c].shape == frame_images[c].shape]
    try:
        pick = select_blobs({c: bg_images[c] for c in cams}, {c: frame_images[c] for c in cams},
                            {c: calibration[c] for c in cams})
    except AttributeError:
        return None
    if pick is None:
        return None
    return {c: (None if o is None else (o.tip, o.far)) for c, o in pick.picks.items()}


class ArgusEngine:
    """The registry entry named "Argus" -- see module docstring."""

    name = "Argus"

    def score(
        self,
        bg_images: dict[int, np.ndarray],
        frame_images: dict[int, np.ndarray],
        calibration: dict[int, CameraCalibration],
        *,
        prior_dart_line_px: dict | None = None,
        visit_bg_images: dict[int, np.ndarray] | None = None,
    ) -> EngineResult:
        # Never raise: a scoring crash must cost this one voter, not the
        # throw. Zeus and dispatch already treat ok=False as "no vote".
        try:
            return self._score(bg_images, frame_images, calibration,
                               prior_dart_line_px, visit_bg_images)
        except Exception as exc:  # noqa: BLE001 -- see comment above
            return EngineResult(
                ok=False, sector=None, ring=None, board_xy_mm=None,
                reason=f"Argus raised {type(exc).__name__}: {exc}",
                diagnostics={"error": f"{type(exc).__name__}: {exc}"},
            )

    def _score(self, bg_images, frame_images, calibration, prior_dart_line_px, visit_bg_images):
        off = PARAMS["inner_off"]
        feats = extract_features(
            bg_images, frame_images, calibration,
            prior_dart_line_px=prior_dart_line_px,
            visit_bg_images=visit_bg_images,
        )
        cams = feats["cams"]
        decision = decide(cams, PARAMS)
        # No consistent pair of cameras: each camera's own detect_tip may
        # have taken the wrong blob or the wrong end (a crowded third dart,
        # an earlier dart's flight). Ask Apollo's cross-view pick which
        # blob in each camera fits the other views in 3D, and score those.
        # Only here: where Argus's cameras already agree, the pick is not
        # consulted (using it on every dart measured +5/-2 on the corpus).
        cross_view = "not_needed"
        if decision.get("fused_xy") is None:
            cross_view = "none_found"
            overrides = _cross_view_overrides(bg_images, frame_images, calibration)
            if overrides is not None:
                feats = extract_features(
                    bg_images, frame_images, calibration,
                    prior_dart_line_px=prior_dart_line_px,
                    visit_bg_images=visit_bg_images,
                    tip_overrides=overrides,
                )
                cams = feats["cams"]
                decision = decide(cams, PARAMS)
                cross_view = "used"
        call = decision["call"]

        diagnostics = {
            "call": call,
            "decision": decision["method"],
            "per_camera": _per_camera_diag(cams, decision),
            "fused_xy_mm": ([_r(v, 2) for v in decision["fused_xy"]]
                            if decision.get("fused_xy") is not None else None),
            "fused_cameras": decision.get("fused_cams") or [],
            "candidates": decision.get("candidates"),
            "parallax": (feats.get("parallax") or {}).get("method"),
            "visit_bg_used": bool(visit_bg_images),
            "cross_view": cross_view,
            "prior_line_cameras": sorted(str(c) for c in (prior_dart_line_px or {})),
        }

        if call is None:
            diagnostics["method"] = "none"
            return EngineResult(
                ok=False, sector=None, ring=None, board_xy_mm=None,
                reason="Argus: no camera found a needle tip",
                diagnostics=diagnostics,
            )

        board_xy, method, point_cam = choose_board_point(call, decision, cams, off)
        diagnostics["method"] = method
        diagnostics["point_camera"] = point_cam
        if board_xy is None:
            return EngineResult(
                ok=False, sector=None, ring=None, board_xy_mm=None,
                reason=f"Argus: call {call} but no camera has a board point",
                diagnostics=diagnostics,
            )

        sector, ring = token_to_sector_ring(call, board_xy)
        total = sum(info["confidence"] for info in decision["cams"].values())
        agree = sum(info["confidence"] for info in decision["cams"].values()
                    if info["call_token"] == call)
        confidence = round(agree / total, 4) if total > 0 else None

        return EngineResult(
            ok=True,
            sector=sector,
            ring=ring,
            board_xy_mm=board_xy,
            reason=f"Argus {decision['method']}: {call} ({len(decision['cams'])} camera(s), point {method})",
            diagnostics=diagnostics,
            confidence=confidence,
        )
