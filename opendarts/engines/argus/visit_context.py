"""The visit's empty board, per camera, for Argus's occlusion check.

Argus asks of every camera's needle entry: is it sitting on a dart thrown
EARLIER in this visit? That needs the board as it was before the visit's
first dart -- the first dart's own background frame, "S0" -- next to this
throw's background (which already shows the earlier darts). Where the two
differ is where earlier darts now sit.

One lookup, shared by live capture and replay (docs/DESIGN.md, "Replay
is the source of truth"), the same way `opendarts.engines.apollo.
prior_dart_context.find_prior_dart_line_px` serves the prior dart's line:

- live: the capture loop keeps the visit's first throw's background in
  memory (`CachedVisitStart`) and passes it here, so a dart never waits
  on the first dart's package being written (that save runs in the
  background, and may be switched off entirely);
- replay: no cache, so the visit's first package in the same session
  directory is loaded from disk.

Returns None for the first dart of a visit (nothing earlier to occlude),
for packages without visit metadata, and whenever the first dart cannot
be found. Never raises.
"""
from __future__ import annotations

import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: {cam_index: bg frame of the visit's first dart}
VisitBgImages = dict[int, np.ndarray]


@dataclass(frozen=True)
class CachedVisitStart:
    """The live capture loop's record of the current visit's first throw:
    its visit id and the background frames it was scored against."""

    visit_id: str
    bg_frames: VisitBgImages


def engine_accepts_visit_bg_images(engine: object) -> bool:
    """True only if `engine.score()` declares `visit_bg_images` -- the
    same duck-typed capability check `engine_accepts_prior_dart_line_px`
    uses, so callers skip the lookup for engines that never read it."""
    score = getattr(engine, "score", None)
    if score is None or not callable(score):
        return False
    try:
        params = inspect.signature(score).parameters
    except (TypeError, ValueError):
        return False
    return "visit_bg_images" in params


def find_visit_bg_images(
    session_dir: Path,
    visit_id: str | None,
    visit_index: int | None,
    *,
    cached: CachedVisitStart | None = None,
) -> VisitBgImages | None:
    """The visit's first throw's background frames, per camera, or None
    (see module docstring)."""
    if visit_id is None or visit_index is None or visit_index <= 0:
        return None
    if cached is not None and cached.visit_id == visit_id:
        return dict(cached.bg_frames) or None
    try:
        if not session_dir.exists():
            return None
        throw_dirs = sorted(p for p in session_dir.iterdir() if p.is_dir())
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
        if meta.get("visit_id") != visit_id or meta.get("visit_index") != 0:
            continue
        from opendarts.capture.throw_package import load_throw_package
        try:
            package = load_throw_package(throw_dir)
        except Exception:  # noqa: BLE001 -- never break the current throw
            log.warning("visit %s: first throw %s did not load -- no visit context",
                        visit_id, throw_dir)
            return None
        return dict(package.bg_frames) or None
    return None
