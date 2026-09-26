"""Replay package format — see docs/DESIGN.md's "Replay is the source of truth" before
touching this file. The whole point: a package stores the exact raw
inputs a throw was scored from, not just the result, so replaying it
through DIFFERENT (e.g. newer/fixed) pipeline code can produce a
DIFFERENT, updated answer. Store what was scored, exactly — exact
pixels, no lossy re-encode,
established here from day one instead of retrofitted later.

Package layout on disk (one directory per throw):
    <package_dir>/
        meta.json -- capture time, session id, camera list,
                                      and two ADDITIVE
                                      fields:
                                        throw_number: int | None -- the
                                          same sequential per-session throw
                                          number already embedded in this
                                          package's own directory name
                                          (opendarts.live.capture_daemon.
                                          handle_ready_to_capture()'s own
                                          `throw_number` local -- the real
                                          source of truth for the
                                          "{session}-{throw_number:03d}-
                                          {sector_token}" throw_id, computed
                                          well before this call and simply
                                          THREADED THROUGH here, never
                                          re-derived a second way). None
                                          for a package saved before this
                                          field existed, or by any caller
                                          that doesn't track it (e.g. most
                                          offline tooling/tests) -- same
                                          absent-not-fabricated convention
                                          as visit_id/calibration_package_id
                                          below.
                                        frame_cameras: list[int] -- which
                                          cameras actually produced a raw
                                          DART frame for this throw (i.e.
                                          the real key set of the
                                          `dart_frames_bgr` argument this
                                          function was called with), always
                                          computed and always written --
                                          unlike `cameras` (unchanged,
                                          still the bg-frame/dart-frame/
                                          calibration INTERSECTION that
                                          controls which cameras' frames
                                          actually get written below),
                                          `frame_cameras` is NOT
                                          intersected against calibration
                                          or background-frame availability.
                                          The two can genuinely differ: a
                                          camera that produced a real dart
                                          frame but had no calibration (or
                                          no bg frame) for this event is in
                                          `frame_cameras` but NOT in
                                          `cameras` -- and, because the
                                          per-camera frame-write below
                                          only iterates `cameras`, that
                                          camera's frame is silently never
                                          persisted to disk at all. Before
                                          this field existed, that gap was
                                          invisible -- opendarts's own integrity
                                          checks could only infer camera
                                          count from files actually on
                                          disk, with no way to tell "this
                                          was really only a 2-camera
                                          capture" from "a 3rd camera
                                          produced a frame that got
                                          silently dropped". Never guessed
                                          from file presence -- always the
                                          real `dart_frames_bgr.keys()` at
                                          the moment of the call, per
                                          the v2 package schema's
                                          "never coerce unknown into a real
                                          value" non-negotiable (there is
                                          no "unknown" case here: this
                                          argument is always a required,
                                          already-populated dict). Absent
                                          (None on `ThrowPackage`, not an
                                          empty list) only for a package
                                          saved before this field existed --
                                          see `load_throw_package()`'s own
                                          `meta.get("frame_cameras")`.
                                        generation: int | None -- (2026-08-27,
                                          the v2 package schema,
                                          the first real v2-session QA
                                          pass) the same counting
                                          GENERATION already embedded in
                                          this package's own directory
                                          name as the `-g{N}-` infix (see
                                          `opendarts.live.capture_daemon.
                                          handle_ready_to_capture()`'s own
                                          `generation`/`generation_infix`
                                          locals, and
                                          `_reset_session_throw_numbering()`'s
                                          docstring for what a "generation"
                                          is), now ALSO written as a real
                                          int field rather than living only
                                          inside a string a consumer has to
                                          re-parse -- QA's own finding:
                                          "opendarts encodes the generation only
                                          in the directory name... which is
                                          exactly what broke tooling when
                                          the g<N> infix first appeared."
                                          Threaded straight through, never
                                          re-derived a second way. None for
                                          a package saved before this field
                                          existed, or by any caller that
                                          doesn't track generations (most
                                          offline tooling/tests) -- same
                                          absent-not-fabricated convention
                                          as `throw_number` above (NOT the
                                          same as generation 0 -- a real
                                          generation-0 package written by a
                                          caller that DOES pass this field
                                          gets the real int `0`, not None).
                                        schema: str | None -- (2026-08-27,
                                          the v2 package schema,
                                          the coordinated flip) the
                                          package-level version marker,
                                          always `META_SCHEMA_V2`
                                          ("dart-package/v2") on every
                                          fresh write -- unlike every
                                          other field on this page, NOT
                                          gated behind a None default; it
                                          is a declaration of package
                                          shape, not optional caller
                                          context. None for a package
                                          saved before this field existed
                                          (every real package predating
                                          this change). See
                                          META_SCHEMA_V2's own
                                          module-level comment.
                                      and (2026-08-14) visit_id/visit_index:
                                      WHICH turn this dart belonged to and
                                      which dart of that turn it was (0/1/2).
                                      See opendarts/live/capture_daemon.py's
                                      `new_visit_id()` and the VISIT model
                                      section of docs/LIVE_API.md. Added as
                                      two fields on the EXISTING meta.json,
                                      not a parallel file -- the same
                                      in-place-extension precedent
                                      `AdGroundTruth`'s own operator_* fields
                                      set. Both absent for every package
                                      saved before this existed; loading one
                                      of those must still work unchanged
                                      (backward compatible -- both degrade to
                                      None). Also (2026-08-20)
                                      calibration_package_id: which
                                      opendarts.capture.calibration_package this
                                      throw's calibrations came from, if any
                                      -- same in-place-extension/absent-not-
                                      fabricated convention. See that
                                      module's own docstring.
        stills_cam{N}.mkv / clip_cam{N}.mkv -- this camera's frames,
                                      LOSSLESS, located by meta.json's
                                      `video` block (see
                                      opendarts.capture.clip; always
                                      follow its `clip`, never build the
                                      name). stills_cam{N}.mkv holds two
                                      frames -- the bg and the scored
                                      commit -- for an ordinary package;
                                      clip_cam{N}.mkv holds the whole
                                      bg..commit+1 run out of the frame
                                      ring for a RECORDED one. A package
                                      has one or the other, written once
                                      (2026-09-26). Since
                                      2026-09-22 every package has these
                                      and none has frame PNGs: the same
                                      frames of a real 3-camera package
                                      cost 6,317 KB as PNGs and 465 KB
                                      as MJPEG stills clips (4,691 KB as
                                      FFV1 on a macOS package), with
                                      byte-identical pixels either way.
        cam{N}_bg.png -- background frame, per camera, LOSSLESS.
                                      ONLY on a package from before that
                                      change (the existing corpus is ~578
                                      of them). Never written now, always
                                      still READ -- see
                                      load_throw_package(), which prefers
                                      the clip and falls back to these.
        cam{N}_frame.png -- frame-with-dart, per camera, LOSSLESS. Same
                                      read-only, corpus-compatibility
                                      story as cam{N}_bg.png above.
        calibration.json -- exact CameraCalibration used, per camera
        result.json -- the ORIGINAL live result (for audit/
                                        comparison only -- replay must NEVER
                                        read this to produce its own answer,
                                        only to compare against it afterward).
                                        (2026-08-27, v2 package-schema
                                        normalization, the "3 keys the
                                        spec missed" pass) three ADDITIVE
                                        top-level keys:
                                          label: str | None -- a
                                            human-readable "S4"-style call
                                            (or "NR"), derived from
                                            sector/ring via the SAME
                                            opendarts.geometry.board.
                                            sector_ring_to_token() this
                                            package's own directory name
                                            already uses. Present for every
                                            REAL engine result; honestly
                                            omitted (never raises, never
                                            blocks the save) for the rare
                                            synthetic/test ScoreResult
                                            whose ring value isn't in that
                                            function's strict vocabulary --
                                            see _safe_label()'s own
                                            docstring.
                                          camera_mode: str | None -- which
                                            frame-source path this throw's
                                            raw frames came through. Only
                                            the direct LocalCameraHub path
                                            ("real") still exists; older
                                            packages can carry another
                                            value. Absent for any caller
                                            that doesn't track a frame
                                            source.
                                          agreement: str | None -- an
                                            "X/Y" snapshot of how many
                                            sub-engines concurred, read
                                            straight off the PRIMARY
                                            engine's own vote-tally
                                            diagnostics when the primary
                                            is a vote-based consensus
                                            engine (Zeus/"Zeus"). Absent
                                            when the primary isn't
                                            vote-based -- opendarts's
                                            primary engine is
                                            operator-configurable, so this
                                            can genuinely not exist for a
                                            given deployment/config.
                                        See save_throw_package()'s own
                                        docstring for the full story on
                                        each.
                                        (2026-08-27, the first real
                                        v2-session QA pass) two further
                                        fixes, both to
                                        already-existing keys, not new
                                        ones:
                                          Top-level rollup (`cameras_used`/
                                          `triangulation`/
                                          `max_ray_disagreement_mm`/
                                          `n_cameras_used`) now carries
                                          REAL values (not null/0) whenever
                                          the PRIMARY engine is a vote-based
                                          consensus engine (Zeus/"Zeus")
                                          whose winning sub-engine's own
                                          diagnostics have them -- QA's own
                                          finding: these were null/0 on
                                          90/90 real opendarts packages, because
                                          the generic `ScoreResult` adapter
                                          used for a non-Apollo primary
                                          has no ray-triangulation data of
                                          its own to report, even though
                                          the WINNING sub-engine (whichever
                                          of Apollo/Talos/Athena/
                                          Ares actually produced the
                                          consensus answer) usually does.
                                          Sourced from the raw primary
                                          `EngineResult.diagnostics`
                                          (`winning_engine`/`sub_results`),
                                          captured in
                                          `handle_ready_to_capture()`
                                          BEFORE the ScoreResult conversion
                                          discards `.diagnostics` -- same
                                          "compute it before it's gone"
                                          pattern `agreement`/`camera_mode`
                                          above already established. Still
                                          null/0 when genuinely
                                          unavailable (primary IS
                                          Apollo with no winner to
                                          promote from -- already correct;
                                          or the winning sub-engine's own
                                          diagnostics don't carry a given
                                          field, e.g. Talos/Athena never
                                          populate `triangulation` at all)
                                          -- see
                                          `_rollup_fields_from_winning_sub_engine_diagnostics()`
                                          below.
        ad_ground_truth.json -- OPTIONAL. Autodarts' own committed
                                        throw for this same physical dart,
                                        via opendarts.live.ad_ground_truth --
                                        see that module for the fetch +
                                        time-window match strategy. Own
                                        per-file `schema` field bumped
                                        2026-08-27 (the v2 package schema,
                                        the coordinated flip):
                                        "ad-ground-truth-v1" ->
                                        "ad-ground-truth-v2" -- see
                                        opendarts.live.ad_ground_truth.
                                        AD_GROUND_TRUTH_SCHEMA_V2's own
                                        module comment for the full story.
                                        Written
                                        as a SEPARATE file (not a required
                                        save_throw_package() argument) since
                                        it is fetched from a live AD
                                        instance as a distinct step, often
                                        after the fact via the backfill CLI
                                        (dev/ad/backfill_ad_ground_truth.py)
                                        -- a package with no AD available at
                                        capture time must still save fine.
                                        Absent for every package saved
                                        before this field existed; loading
                                        one of those must still work
                                        unchanged (backward compatible).
                                        Also carries (2026-08-12)
                                        ``operator_marked_wrong``/
                                        ``operator_note`` -- a HUMAN
                                        judgment flag, set/unset via
                                        ``mark_operator_ad_wrong()`` below,
                                        NOT part of the automated fetch --
                                        and (2026-08-13)
                                        ``operator_confirmed_source``/
                                        ``_sector``/``_ring``: WHICH answer
                                        that same human says was actually
                                        right, which the dashboard then
                                        grades every engine against instead
                                        of AD's own. Same function sets all
                                        five; see its docstring.
        (Packages saved before 2026-09-17 may also hold an
        ad_calibration.json; nothing reads it, and loading ignores it.)

        capture_diagnostics.json -- OPTIONAL. Added 2026-08-16 after a
                                        real incident (throws 45 and 57 of
                                        one recorded session --
                                        both showed
                                        `ad_ground_truth.json`'s
                                        `match_reason:
                                        "ws_no_buffered_events"` and both
                                        had by far the largest inter-throw
                                        capture-timing gaps in that whole
                                        session, and diagnosing WHY took
                                        real after-the-fact archaeology
                                        because nothing durable had been
                                        recorded about the settle timeline
                                        or the AD WS buffer's state at the
                                        time). Self-contained diagnostic
                                        data for THIS throw only -- no
                                        external log needed to answer "why
                                        did capture take so long" or "why
                                        didn't this throw match AD":
                                          "settle": {
                                            "settle_duration_s": float|null,
                                            "straggler_camera": int|null,
                                            "per_camera_settle_offset_s":
                                              {"<cam>": float, ...},
                                            "ambiguous_settle_fired": bool,
                                            "ambiguous_settle_outcome":
                                              "at_baseline"|"grew"
                                              |"unchanged"|null,
                                          },
                                          "ad_ws_buffer_at_capture": {...}|null
                                            -- opendarts.live.ad_ws_listener.
                                            AdWsListener.
                                            diagnostics_snapshot()'s own
                                            shape (see that method's
                                            docstring), captured
                                            SYNCHRONOUSLY at throw-capture
                                            time by
                                            opendarts.live.capture_daemon.
                                            handle_ready_to_capture() --
                                            null when no AdWsListener was
                                            wired for this run (AD ground
                                            truth disabled).
                                        Written via
                                        save_capture_diagnostics() below,
                                        same optional/absent-not-
                                        malformed convention as
                                        ad_ground_truth.json -- a package
                                        saved before this field existed
                                        (or by a caller that doesn't build
                                        diagnostics) simply has no such
                                        file and must still load fine.

    NOTE on a removed file: this package format used to carry an
    OPTIONAL second-oracle ground-truth JSON -- another system's
    live-scored result for the same physical dart, recorded off its own
    event stream. Removed once that system became a real, replayable
    registry engine here: an external answer for any package now comes
    from actually running that engine against the package's own stored
    frames, not from a separately-captured live field.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from opendarts.geometry.board import sector_ring_to_token
from opendarts.pipeline import CameraCalibration, ScoreResult

log = logging.getLogger(__name__)


def _write_json(path: Path, data) -> None:
    """Write a package's JSON file ATOMICALLY: a temporary file in the same
    folder, then a rename over the real one.

    Several of these files are rewritten after the package exists --
    result.json by the background also-run engines, ad_ground_truth.json by
    an operator's mark -- while the dashboard and the next dart's lookups
    read them. A plain write_text() truncates first, so a reader could get
    an empty file: seen as a JSONDecodeError in the test suite when the
    background write landed mid-read (2026-09-17).
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

if TYPE_CHECKING:
    from opendarts.live.ad_ground_truth import AdGroundTruth

# The v2 package schema (2026-08-27, the coordinated flip):
# `meta.json`'s own package-level version marker. NEW -- no meta.json
# version field existed before this (unlike ad_ground_truth.json, which
# already had a per-FILE `schema` string; see
# opendarts.live.ad_ground_truth.AD_GROUND_TRUTH_SCHEMA_V2 for that one's own
# separate flip). Unconditionally written by save_throw_package() on
# every FRESH package from this change forward (same "always written, not
# gated behind a None-default" convention as `frame_cameras` -- this is a
# declaration of package shape, not optional caller context). A package
# saved before this field existed simply has no `schema` key at all --
# ThrowPackage.schema/load_throw_package() below degrade that to None,
# never fabricate a value for an old package.
META_SCHEMA_V2 = "dart-package/v2"


@dataclass
class ThrowPackage:
    package_dir: Path
    session: str
    cameras: list[int]
    # Populated by load_throw_package(); None until loaded.
    bg_frames: dict[int, np.ndarray] | None = None
    dart_frames: dict[int, np.ndarray] | None = None
    calibrations: dict[int, CameraCalibration] | None = None
    original_result: dict | None = None
    # None means "no ad_ground_truth.json in this package" -- either
    # never fetched, or a pre-existing package saved before this field
    # existed. Distinct from an AdGroundTruth with matched=False (a real
    # fetch attempt that concluded no confident match), which IS present
    # here once loaded.
    ad_ground_truth: "AdGroundTruth | None" = None
    # Which turn this dart belonged to, and which dart of that turn it
    # was (0-based, 0..MAX_DARTS_PER_TURN-1). Both None for a package
    # saved before the visit model existed, and for any caller that
    # doesn't track visits (e.g. an offline tool re-saving a package) --
    # honestly absent rather than fabricated as "visit 0, dart 0".
    visit_id: str | None = None
    visit_index: int | None = None
    # Real capture-time diagnostics (2026-08-16, "persist real
    # diagnostics" task -- see docs/DESIGN.md and capture_diagnostics.json's
    # own module-docstring section below): settle timeline, straggler
    # camera, ambiguous-settle classification, and the AD WS listener's
    # buffer state, all as they stood AT CAPTURE TIME for this specific
    # throw. None for a package saved before this field existed, or one
    # saved by a caller that doesn't build diagnostics (e.g. most offline
    # tooling/tests) -- same absent-not-malformed convention as
    # ad_ground_truth above.
    capture_diagnostics: dict | None = None
    # Which opendarts.capture.calibration_package this throw's `calibrations`
    # (the exact CameraCalibration values above, ALSO still duplicated
    # in full into this package's own calibration.json -- keeping both
    # is deliberate, see save_throw_package()'s own docstring) came from,
    # if any. None for a package saved before calibration packages
    # existed, or whose calibration came from a bootstrap that opted out
    # of package saving -- same absent-not-fabricated convention as
    # visit_id/visit_index above.
    calibration_package_id: str | None = None
    # the v2 package-schema additions (2026-08-26, see module docstring's
    # meta.json section for the full story). `throw_number`: the
    # session-sequential throw number already embedded in this package's
    # own directory name -- None for a package saved before this field
    # existed, or by a caller that doesn't track it. `frame_cameras`:
    # which cameras actually produced a raw dart frame for this throw --
    # NOT the same as `cameras` above (that stays the persisted-PNG
    # intersection); None (not an empty list) for a package saved before
    # this field existed -- an empty list would falsely claim "zero
    # cameras produced a frame", which is never true of a saved package.
    throw_number: int | None = None
    frame_cameras: list[int] | None = None
    # The v2 package schema (2026-08-27, first real v2-session
    # QA pass): the same counting GENERATION already embedded in this
    # package's own directory name as the `-g{N}-` infix (see
    # opendarts.live.capture_daemon.handle_ready_to_capture()'s own
    # `generation` local) -- now ALSO a real int field, not just a string
    # a consumer has to re-parse. None for a package saved before this
    # field existed, or by a caller that doesn't track generations -- same
    # absent-not-fabricated convention as `throw_number` above. NOT the
    # same as a real generation 0 (written as the int `0`, not None, by
    # any caller that does pass this field).
    generation: int | None = None
    # The v2 package schema (2026-08-27, the coordinated flip):
    # the package-level version marker save_throw_package() now always
    # writes -- see META_SCHEMA_V2's own comment above. None for a
    # package saved before this field existed (every real package
    # predating this change); no reader currently branches on this value
    # (every meta.json field this project adds is purely additive and
    # already read generically via `.get()`, so nothing structurally
    # NEEDS the schema string to parse correctly) -- it exists as a
    # declarative marker, not a dispatch key, unlike ad_ground_truth.json's
    # own `schema` (see opendarts.live.ad_ground_truth's module for why THAT
    # one's key rename needed real dispatch logic and this one doesn't).
    schema: str | None = None


def calibration_to_dict(calib: CameraCalibration) -> dict:
    """Public (not module-private) since 2026-08-12 -- the exact same
    on-disk shape a throw package's own calibration.json already uses,
    so there's one serialization format for "a real CameraCalibration
    on disk" in this project, not two. (Originally reused by
    opendarts.live.capture_daemon.save_calibration_snapshot(), removed
    2026-08-22 as genuinely dead code -- see that module's own note at
    the removed function's former location.)"""
    return {
        "camera_matrix": calib.camera_matrix.tolist(),
        "dist_coeffs": calib.dist_coeffs.tolist(),
        "rvec": calib.rvec.tolist(),
        "tvec": calib.tvec.tolist(),
        "landmark_spread_ok": calib.landmark_spread_ok,
    }


def calibration_from_dict(d: dict) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.array(d["camera_matrix"], dtype=np.float64),
        dist_coeffs=np.array(d["dist_coeffs"], dtype=np.float64),
        rvec=np.array(d["rvec"], dtype=np.float64),
        tvec=np.array(d["tvec"], dtype=np.float64),
        pnp_result=None, # not needed for replay, not round-tripped
        landmark_spread_ok=d.get("landmark_spread_ok"),
    )


def _safe_label(result: ScoreResult) -> str | None:
    """`sector_ring_to_token()`, but never allowed to block a package
    write -- see `_score_result_to_dict()`'s own call site for why. That
    function's own docstring deliberately "still raises on an unrecognized
    ring value... rather than silently guessing", which is the right
    contract for its REAL callers (capture_daemon.py's own `sector_token`
    local, matching this vocabulary against `sector_ring_for_point()`'s).
    But `save_throw_package()` is called far more broadly -- offline
    tools, and plenty of test fixtures across this project's own suite
    that construct a synthetic `ScoreResult` with a looser shorthand ring
    value (e.g. `"single"` instead of `"single_inner"`/`"single_outer"`)
    that a real engine never actually produces. Label is a display
    convenience, not load-bearing data (see module docstring: result.json
    is "for audit/comparison only"), so it degrades to None (omitted from
    the written dict, same absent-not-fabricated convention as
    `camera_mode`/`agreement`) rather than raising and refusing to save an
    otherwise-complete, real package over a cosmetic formatting gap.
    """
    if not result.ok:
        return "NR"
    try:
        return sector_ring_to_token(result.sector, result.ring)
    except ValueError:
        return None


def _score_result_to_dict(result: ScoreResult) -> dict:
    tri = result.triangulation
    label = _safe_label(result)
    return {
        "ok": result.ok,
        "sector": result.sector,
        "ring": result.ring,
        # The v2 package schema (2026-08-27, "3 keys the spec
        # missed" pass): a human-readable "S4"-style call, derived here
        # from the SAME (sector, ring) this dict already carries via the
        # SAME
        # canonical vocabulary `opendarts.live.capture_daemon.
        # handle_ready_to_capture()` already uses to build a package's own
        # `throw_id`/directory name (`sector_ring_to_token()` + its own
        # "NR" no-result convention) -- reused verbatim, not
        # reimplemented, so this can never drift from the directory name a
        # human is already looking at. Present for every REAL engine
        # result (production ring values always match
        # `sector_ring_for_point()`'s own vocabulary); see `_safe_label()`
        # for the one case it's honestly omitted instead.
        **({"label": label} if label is not None else {}),
        "board_xy_mm": result.board_xy_mm,
        "n_cameras_used": result.n_cameras_used,
        "reason": result.reason,
        "max_ray_disagreement_mm": result.max_ray_disagreement_mm,
        # Added 2026-08-12 alongside score_dart()'s 2-of-3 RANSAC
        # fallback -- optional/additive only: older result.json files on
        # disk simply won't have these keys, and nothing reads this dict
        # back into a ScoreResult (replay always re-scores from raw
        # pixels, per this module's own docstring), so there's no
        # backward-compat parsing concern.
        # The v2 package schema null-vs-[] pass (2026-08-27, QA's own rule:
        # "a collection with no members -> [] never null, never absent"):
        # `cameras_used` is list-typed on the wire regardless of WHY it's
        # empty (triangulation never attempted vs. attempted-but-rejected
        # vs. an abstained ok=False engine) -- `None` here used to mean
        # all three indistinguishably, forcing every reader to null-check
        # before iterating. `result.cameras_used is None` at the Python
        # level still legitimately distinguishes "not attempted" from an
        # (unreachable in practice, since triangulate() needs >=2 rays)
        # "attempted with zero" -- that distinction is NOT lost by this
        # fix, it just never needed the JSON array type to carry it; nothing
        # downstream reads result.cameras_used itself as None vs () (every
        # real call site checks falsiness, not identity -- see
        # opendarts.engines.apollo.engine.py's own `result.cameras_used or
        # ()` / `not result.cameras_used` guards).
        "cameras_used": list(result.cameras_used) if result.cameras_used is not None else [],
        "outlier_camera": result.outlier_camera,
        "triangulation": None
        if tri is None
        else {
            "ok": tri.ok,
            "point_xyz": None if tri.point_xyz is None else tri.point_xyz.tolist(),
            "board_plane_xy": tri.board_plane_xy,
            "plane_discrepancy_mm": tri.plane_discrepancy_mm,
            # Same null-vs-[] fix, same rule: `TriangulationResult` has a
            # real "_empty_result()" case (rays.py -- singular/degenerate
            # ray geometry) where `tri` itself is NOT None (so the outer
            # `triangulation` object above IS written) but every one of
            # its own fields, including this list, is `None` --
            # previously produced a null list nested INSIDE an already
            # non-null object, exactly the "schema parity, no data"
            # pattern QA's rule targets.
            "per_ray_distance_mm": (
                list(tri.per_ray_distance_mm) if tri.per_ray_distance_mm is not None else []
            ),
            "n_rays": tri.n_rays,
        },
    }


def save_throw_package(
    dest_dir: Path,
    session: str,
    bg_frames_bgr: dict[int, np.ndarray],
    dart_frames_bgr: dict[int, np.ndarray],
    calibrations: dict[int, CameraCalibration],
    result: ScoreResult,
    visit_id: str | None = None,
    visit_index: int | None = None,
    calibration_package_id: str | None = None,
    throw_number: int | None = None,
    camera_mode: str | None = None,
    agreement: str | None = None,
    generation: int | None = None,
    primary_engine_diagnostics: dict | None = None,
    captured_at_utc: str | None = None,
    video: dict | None = None,
    host: str | None = None,
    build: str | None = None,
    bg_jpegs: dict[int, bytes] | None = None,
    dart_jpegs: dict[int, bytes] | None = None,
    defer_clips: bool = False,
) -> Path:
    """Write a complete replay package. Fails loudly (raises) rather than
    writing a partial package -- an incomplete package that LOOKS
    complete is worse than no package, since it would silently corrupt a
    later replay-based drift check.

    ``visit_id``/``visit_index`` (2026-08-14): which turn this dart
    belonged to and which dart of that turn it was (0-based). Optional
    and defaulted to None so every existing caller/test is unaffected and
    a package written without them stays valid -- the keys are simply
    omitted from meta.json in that case, matching how a package with no
    AD ground truth just has no ad_ground_truth.json rather than a file
    full of nulls. The real producer is
    ``opendarts.live.capture_daemon.handle_ready_to_capture()``.

    ``calibration_package_id`` (2026-08-20): which
    ``opendarts.capture.calibration_package`` this throw's ``calibrations``
    came from, if any -- see that module's own docstring for the full
    "REPLAY applied to calibration bootstrap" rationale. Same
    optional/omitted-not-null convention as ``visit_id``/``visit_index``
    above. Deliberately ADDITIVE, not a replacement for this package's
    own ``calibration.json`` (which keeps storing the exact
    ``CameraCalibration`` values used, in full, as it always has) -- a deliberate design decision
    throw package fully self-contained even after a calibration package
    is later cleaned up/deleted (see that module's rig-side cleanup) has
    real value a bare reference alone would lose.

    ``throw_number`` (2026-08-26, the v2 package schema -- see
    module docstring's meta.json section): the same session-sequential
    throw number the real caller
    (``opendarts.live.capture_daemon.handle_ready_to_capture()``) already
    computes and embeds in this package's own directory name -- passed
    straight through, never re-derived here. Optional/omitted-not-null,
    same convention as ``visit_id``/``calibration_package_id`` above, for
    any caller (offline tooling, most tests) that doesn't track it.

    ``frame_cameras`` is NOT a parameter -- it needs no caller input at
    all, since it's always exactly ``sorted(dart_frames_bgr.keys())``,
    computed unconditionally below. See module docstring for why this is
    a real, distinct signal from ``cameras``, not a duplicate.

    ``camera_mode`` (2026-08-27, the v2 package schema, "3 keys
    the spec missed"): which frame-source code path this throw's raw
    frames actually came through -- ``"real"`` for
    ``opendarts.live.local_capture.LocalCameraHub``'s direct-camera
    path. These are genuine physical camera frames -- there is no
    simulated/fabricated-frame fallback (see
    ``opendarts.live.local_capture.LocalCameraHub``: a camera-open failure
    raises loud, it never substitutes a synthetic frame), so this field is
    not protecting against that exact incident here. It is still a real,
    always-computed-not-hardcoded distinction (never a literal ``"real"``
    with no code path that could say otherwise) recording which of the
    two genuinely different capture transports produced this package's
    frames. Optional/omitted-not-null, same convention as ``throw_number``
    above, for any caller that doesn't track a frame source (offline
    tooling, most tests, and any package pre-dating this field). The real
    producer is ``opendarts.live.capture_daemon.handle_ready_to_capture()``,
    given straight from ``run_capture_loop_body()``.

    ``agreement`` (2026-08-27, the same v2 package schema pass): a
    capture-time snapshot of how many sub-engines concurred, formatted
    ``"X/Y"`` -- e.g. ``"4/4"``. Recovered from the
    PRIMARY engine's own vote-tally diagnostics
    (``opendarts.engines.zeus.engine.ZeusEngine.score()``'s ``n_usable``/
    ``winner``/``vote_tally``, the same numbers its own ``reason`` string
    already narrates in prose) when the primary engine is a vote-based
    consensus engine (Zeus/"Zeus") -- see
    ``opendarts.live.capture_daemon._agreement_string_from_engine_result()``
    for the extraction, called on the primary engine's raw ``EngineResult``
    BEFORE it is narrowed to a ``ScoreResult`` (which drops
    ``diagnostics`` entirely). ``None`` (omitted, never fabricated) when
    the primary engine is not vote-based -- opendarts's primary engine is
    operator-configurable (``opendarts.engines.registry.
    DEFAULT_PRIMARY_ENGINE`` is ``"Apollo"``; live production has it set
    to Zeus, confirmed 2026-08-27 against a real on-disk package, but nothing
    guarantees that for every deployment), so this field is not always
    derivable. That is a real architectural consequence, not an
    oversight. Optional/omitted-not-null, same convention as
    ``camera_mode``/``throw_number`` above.

    ``generation`` (2026-08-27, the v2 package schema, first
    real v2-session QA pass): the same counting GENERATION already
    embedded in this package's own directory name as the ``-g{N}-``
    infix -- see module docstring's meta.json section, and
    ``opendarts.live.capture_daemon.handle_ready_to_capture()``'s own
    ``generation`` local (the real source of truth, threaded straight
    through here, never re-derived). Optional/omitted-not-null, same
    convention as ``throw_number`` above -- NOT written as a bare
    ``0`` default when absent; a caller that genuinely tracks generation
    0 must pass ``0`` explicitly to get it recorded.

    ``primary_engine_diagnostics`` (2026-08-27, the same v2 package
    schema pass): the PRIMARY engine's raw ``EngineResult.diagnostics``
    dict -- same "captured before the ScoreResult conversion discards
    ``.diagnostics``" pattern ``agreement``/``camera_mode`` above already
    use. Used ONLY to fill ``result.json``'s top-level rollup fields
    (``cameras_used``/``triangulation``/``max_ray_disagreement_mm``/
    ``n_cameras_used``) when ``result`` itself (the narrowed
    ``ScoreResult``) left them at their honest "not populated" defaults
    -- see ``_rollup_fields_from_winning_sub_engine_diagnostics()`` above
    for the extraction and exactly when it does/doesn't find something.
    Never overwrites a value ``result`` already carries (e.g. a real
    Apollo-as-primary throw, whose own adapter already populates these
    correctly) -- this parameter can only ever fill an existing gap, not
    change an already-real answer. ``None`` (the default -- every
    pre-existing caller/test) means "no extra rollup context available,"
    which degrades to exactly today's pre-fix behavior (null/0 for a
    non-Apollo primary with no winner to promote from).

    ``captured_at_utc`` (2026-09-01, "background the throw-package save"
    task): an explicit, caller-supplied timestamp to
    write into ``meta.json`` instead of this function calling
    ``datetime.now(timezone.utc)`` itself. ``None`` (the default -- every
    pre-existing caller/test) preserves today's exact behavior: the
    timestamp is generated right here, at write time. The real reason
    this exists: ``opendarts.live.capture_daemon.handle_ready_to_capture()``
    now runs the actual disk write on a background thread (see that
    function's own docstring section) to get the ``THROW_DETECTED`` event
    out the door before the write completes, not after -- which means
    the event needs a timestamp BEFORE the write has even started, and a
    single value computed once and threaded through both the event and
    this call is the only way for them to agree, now that reading
    ``meta.json`` back after the write (the old mechanism) is no longer
    available at the moment the event needs to fire.

    ``bg_jpegs``/``dart_jpegs`` (2026-09-22, the "no PNGs, ever" change):
    the camera's OWN JPEG bytes for the very frames in ``bg_frames_bgr``/
    ``dart_frames_bgr``, per camera. When both are present for a camera
    its clip is a stream copy of those exact bytes rather than an FFV1
    encode of the array -- measured on a real 3-camera package, 465 KB
    against 2,973 KB for the same six frames, with byte-identical pixels either
    way (the clip is read straight back and compared before it is
    accepted; see opendarts.capture.clip.write_still_clips).

    Optional per camera and optional entirely, because a Mac genuinely
    has no camera JPEG to offer (OpenCV's AVFoundation backend never
    surfaces it -- docs/STREAMING.md) and because a caller that cannot
    prove a JPEG belongs to the same tick as the array MUST pass nothing
    for that frame. A bigger file is a cost; a clip holding a frame
    nothing scored would break SCORE==STORE invisibly, which is not a
    cost but a corruption. The live producer
    (``opendarts.live.capture_daemon.run_capture_loop_body()``) pairs
    them by ARRAY IDENTITY against the frame it is about to score, so
    "the bytes and the array came from the same pump cycle" is checked
    rather than assumed.

    ``defer_clips`` (2026-09-26): write the data files only -- no clip, and
    no ``video`` block in meta.json. The live path uses it: a package's
    ONE clip is written just after its data, once the recording decision
    is known, and meta.json is then pointed at it
    (``opendarts.capture.throw_capture.ThrowCaptureService.
    write_package_clip`` / ``opendarts.capture.clip.point_meta_at_clip``).
    Until then the package has no readable frames.
    """
    from opendarts.capture import clip as clip_mod

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    cameras = sorted(set(bg_frames_bgr) & set(dart_frames_bgr) & set(calibrations))
    if not cameras:
        raise ValueError("no camera has all of {bg frame, dart frame, calibration} -- refusing to write a partial package")

    # EVERY PACKAGE IS JSON + ONE CLIP PER CAMERA (2026-09-22). No frame
    # PNG is written here any more, in any mode.
    #
    # The frames still have to be SCORE==STORE exact, and they are: the
    # two-frame stills clip is either a stream copy of the camera's own
    # JPEG bytes (so the container holds exactly what the camera sent,
    # and decoding it reproduces exactly what was scored) or a lossless
    # FFV1 encode of the scored array. opendarts.capture.clip.
    # write_still_clips reads both frames straight back and compares them
    # before returning, so an inexact clip cannot reach disk.
    #
    # Called without `defer_clips` (offline tooling, tests), the stills
    # clip is written here, from the two arrays already in hand, with no
    # ring and no timing involved. The live capture path defers it: the
    # package's ONE clip -- a recorded window or these same two frames --
    # is written right after this returns, once its recording decision is
    # known (see this function's docstring). A failure there still ends in
    # the stills clip, which needs nothing but the arrays; only a process
    # killed in between leaves a package with data and no frames.
    #
    # `video` already set means the caller wrote the clips itself and is
    # handing over the finished pointer block; nothing to do here.
    if video is None and not defer_clips:
        video = clip_mod.write_still_clips(
            dest_dir,
            {cam: bg_frames_bgr[cam] for cam in cameras},
            {cam: dart_frames_bgr[cam] for cam in cameras},
            # The camera's own JPEG bytes for these exact frames, when the
            # caller could pair them with the arrays at the same tick
            # (opendarts.live.capture_daemon). Absent per camera, or
            # entirely, is normal and only costs file size -- see
            # write_still_clips() for the pairing hazard and the
            # verification that closes it.
            bg_jpegs=(
                {cam: bg_jpegs[cam] for cam in cameras if cam in bg_jpegs}
                if bg_jpegs else None
            ),
            commit_jpegs=(
                {cam: dart_jpegs[cam] for cam in cameras if cam in dart_jpegs}
                if dart_jpegs else None
            ),
        )

    calib_out = {str(cam): calibration_to_dict(calibrations[cam]) for cam in cameras}
    _write_json(dest_dir / "calibration.json", calib_out)

    # The v2 package schema (2026-08-26): which cameras actually
    # produced a raw DART frame for this throw -- the real
    # `dart_frames_bgr` key set, BEFORE the bg/calibration intersection
    # above narrows it down to `cameras`. Always known (this argument is
    # always a required, already-populated dict -- never "unknown"), so
    # always written, unlike the optional/omitted-when-absent fields
    # below. See module docstring's meta.json section for the full
    # "why this can genuinely differ from `cameras`" story.
    frame_cameras = sorted(dart_frames_bgr.keys())

    meta = {
        "session": session,
        "cameras": cameras,
        "captured_at_utc": captured_at_utc or datetime.now(timezone.utc).isoformat(),
        "frame_cameras": frame_cameras,
        # The v2 package schema (2026-08-27, the coordinated
        # flip) -- always written, like frame_cameras, never gated behind
        # a caller-supplied None default. See META_SCHEMA_V2's own comment.
        "schema": META_SCHEMA_V2,
    }
    # Omitted entirely (not written as null) when the caller doesn't
    # track visits -- see this function's own docstring.
    if visit_id is not None:
        meta["visit_id"] = visit_id
    if visit_index is not None:
        meta["visit_index"] = visit_index
    if calibration_package_id is not None:
        meta["calibration_package_id"] = calibration_package_id
    if throw_number is not None:
        meta["throw_number"] = int(throw_number)
    if generation is not None:
        meta["generation"] = int(generation)
    # WHERE and WHAT PRODUCED THIS THROW (2026-09-22). A package is
    # otherwise silent about its own origin: two packages captured by two
    # different rigs are structurally identical, so the moment they are
    # pulled to one machine the only thing distinguishing them is which
    # directory someone happened to rsync them into. That is fine on the
    # rig, where there is one answer, and useless afterwards -- and the
    # corpus exists precisely to compare rigs scoring the SAME darts.
    #
    # `build` is the code version that scored it, which is a different
    # question from `schema` (the file's shape) and matters just as much:
    # this corpus was captured across several builds in one afternoon, the
    # AD matching logic changing mid-session, and nothing in a package
    # said which side of that change it came from.
    #
    # Passed IN rather than read here: opendarts.capture must not import
    # from opendarts.live (build_info lives there), and the caller already
    # knows both. Additive and optional, so every package written before
    # this loads unchanged -- the same backward-compatible convention the
    # visit/calibration fields above follow, which is why the schema stays
    # at v2.
    if host is not None:
        meta["host"] = str(host)
    if build is not None:
        meta["build"] = str(build)
    # The clip pointer block (opendarts.capture.clip): names each
    # camera's clip and the byte-identical bg/commit indices inside it.
    # Present on every finished package written now -- here when the clip
    # was written above, or added by point_meta_at_clip() just after a
    # deferred one -- and absent on a package from before 2026-09-22 that
    # never had a recording, which is how a reader still tells a clip
    # package from a bg+dart-PNG one. Whether the clip is a
    # RECORDING is a separate question with its own answer: the block's
    # `kind` (opendarts.capture.clip.is_recorded_clip).
    if video is not None:
        meta["video"] = video
    _write_json(dest_dir / "meta.json", meta)

    result_out = _score_result_to_dict(result)
    # The v2 package schema (2026-08-27) -- omitted entirely
    # (not written as null) when the caller doesn't have this context,
    # same convention as meta.json's own optional fields above. `label`
    # is NOT here -- it's always computed inside `_score_result_to_dict()`
    # itself, since it needs no caller context (see that function).
    if camera_mode is not None:
        result_out["camera_mode"] = camera_mode
    if agreement is not None:
        result_out["agreement"] = agreement
    # The v2 package schema (2026-08-27) -- fill the top-level
    # rollup ONLY where `result` itself left the honest "not populated"
    # default (never overwrite a real value the ScoreResult already
    # carries). See `_rollup_fields_from_winning_sub_engine_diagnostics()`'s
    # own docstring and this function's own docstring for
    # `primary_engine_diagnostics`.
    if primary_engine_diagnostics is not None:
        rollup = _rollup_fields_from_winning_sub_engine_diagnostics(primary_engine_diagnostics)
        if not result_out.get("cameras_used") and "cameras_used" in rollup:
            result_out["cameras_used"] = rollup["cameras_used"]
        if result_out.get("triangulation") is None and "triangulation" in rollup:
            result_out["triangulation"] = rollup["triangulation"]
        if result_out.get("max_ray_disagreement_mm") is None and "max_ray_disagreement_mm" in rollup:
            result_out["max_ray_disagreement_mm"] = rollup["max_ray_disagreement_mm"]
        if not result_out.get("n_cameras_used") and "n_cameras_used" in rollup:
            result_out["n_cameras_used"] = rollup["n_cameras_used"]
    _write_json(dest_dir / "result.json", result_out)

    return dest_dir


def validate_throw_package_meta_v2(meta: dict) -> None:
    """Schema-validate an already-parsed ``meta.json`` dict against the
    shape ``save_throw_package()`` now always produces --
    the v2 package schema's "validate on write, in CI" -- meant
    to be called from the fast pytest suite right after a real
    ``save_throw_package()`` call, on the JSON it just wrote, NOT applied
    to any package already on disk (those stay whatever shape they were
    saved in forever -- see this module's docstring's "never rewritten"
    discipline and ``load_throw_package()``'s own backward-compat
    reading). Raises ``AssertionError`` with a message naming the exact
    field and the full dict on any violation -- never returns a bool, so
    a caller can't accidentally ignore a failure.

    Only checks fields this schema-normalization task actually touches
    (``cameras``/``captured_at_utc``/``session`` were already correct and
    unchanged; ``frame_cameras`` is new and always written;
    ``throw_number``/``generation`` (2026-08-27) are new and
    OPTIONAL -- see ``save_throw_package()``'s own docstring for why they
    stay omittable rather than required, same as
    ``visit_id``/``calibration_package_id``; ``schema`` (2026-08-27, the
    coordinated flip) is new and ALWAYS written, same as
    ``frame_cameras``). Does not attempt to be a general-purpose meta.json
    schema validator beyond that scope.
    """
    for key in ("session", "captured_at_utc"):
        value = meta.get(key)
        assert isinstance(value, str) and value, (
            f"meta.json missing/invalid required string field {key!r}: {meta!r}"
        )
    cameras = meta.get("cameras")
    assert isinstance(cameras, list) and cameras and all(isinstance(c, int) for c in cameras), (
        f"meta.json missing/invalid 'cameras' (must be a non-empty list[int]): {meta!r}"
    )
    frame_cameras = meta.get("frame_cameras")
    assert isinstance(frame_cameras, list) and all(isinstance(c, int) for c in frame_cameras), (
        "meta.json missing/invalid 'frame_cameras' -- the v2 package schema "
        f"requires save_throw_package() to always write a list[int] here: {meta!r}"
    )
    assert meta.get("schema") == META_SCHEMA_V2, (
        "meta.json missing/invalid 'schema' -- the v2 package schema "
        f"requires save_throw_package() to always write {META_SCHEMA_V2!r} here: {meta!r}"
    )
    if "throw_number" in meta:
        assert isinstance(meta["throw_number"], int) and not isinstance(meta["throw_number"], bool), (
            f"meta.json 'throw_number' must be an int when present: {meta!r}"
        )
    if "generation" in meta:
        assert isinstance(meta["generation"], int) and not isinstance(meta["generation"], bool), (
            f"meta.json 'generation' must be an int when present: {meta!r}"
        )


def validate_throw_package_result_v2(result: dict) -> None:
    """Schema-validate an already-parsed ``result.json`` dict against the
    shape ``save_throw_package()`` now produces -- the v2 package
    schema's "3 keys the spec missed" (``camera_mode``/``label``/
    ``agreement``, the three top-level keys that were found missing).
    Same discipline as ``validate_throw_package_meta_v2()``
    immediately above: meant to run in the fast pytest suite right after a
    real ``save_throw_package()`` call, NEVER against an already-on-disk
    package (pre-existing packages keep whichever shape they were saved
    in). Raises ``AssertionError`` naming the exact field, never returns a
    bool.

    All three (``label``/``camera_mode``/``agreement``) are OPTIONAL, same
    omitted-not-null convention as ``throw_number`` in the meta.json
    validator -- see ``save_throw_package()``'s/``_safe_label()``'s own
    docstrings for why each can legitimately be absent (``label``: a
    synthetic/test ``ScoreResult`` whose ring value isn't in
    ``sector_ring_to_token()``'s strict vocabulary -- never true for a
    real engine result; ``camera_mode``: no tracked frame source;
    ``agreement``: a non-vote-based primary engine with no also-run
    vote-based engine either).
    """
    if "label" in result:
        assert isinstance(result["label"], str) and result["label"], (
            f"result.json 'label' must be a non-empty str when present: {result!r}"
        )
    if "camera_mode" in result:
        assert isinstance(result["camera_mode"], str) and result["camera_mode"], (
            f"result.json 'camera_mode' must be a non-empty str when present: {result!r}"
        )
    if "agreement" in result:
        assert isinstance(result["agreement"], str) and result["agreement"], (
            f"result.json 'agreement' must be a non-empty str when present: {result!r}"
        )


# Package-only display-name mapping: a codename
# swap applied ONLY at the point engine identities get written into a
# throw package's result.json ("primary_engine" value + "other_engines"
# keys) -- for forward-compat testing, not a rename of anything else.
def _rollup_fields_from_winning_sub_engine_diagnostics(diagnostics: dict | None) -> dict:
    """The v2 package schema (2026-08-27, first real v2-session
    QA pass): recover `result.json`'s top-level rollup fields
    (`cameras_used`/`triangulation`/`max_ray_disagreement_mm`/
    `n_cameras_used`) from a vote-based primary engine's (Zeus/"Zeus")
    OWN diagnostics -- specifically the WINNING sub-engine's own
    diagnostics, which a real triangulating sub-engine (Apollo always;
    Talos/Athena partially -- see below) already populates, even
    though Zeus's own `EngineResult` (a consensus VOTE, not a
    triangulation) has none of its own to report. QA's own finding:
    the keys were written and left empty instead of being sourced from
    the winning sub-engine's own diagnostics -- "the
    `primary_engine: null` mistake one level down."

    Deliberately keyed on DIAGNOSTICS SHAPE (`winning_engine` +
    `sub_results` both present -- the exact two keys
    `opendarts.engines.zeus.engine.ZeusEngine.score()` always sets on a
    successful vote), not a hardcoded `"Zeus"` name check -- same
    discipline `agreement_string_from_diagnostics()` above already
    established: any current or future vote-based primary with this same
    diagnostics shape is picked up for free, and a non-vote-based primary
    (whose own diagnostics never has this shape) correctly gets `{}` back
    -- caller leaves its own already-correct/already-null fields exactly
    as they were.

    Returns only the keys the winning sub-engine's own diagnostics
    ACTUALLY has, never a fabricated 0/None for one it doesn't --
    `Apollo` (via `opendarts.engines.apollo.engine.
    score_result_to_engine_result()`) always carries all four; `Talos`/
    `Athena` carry `cameras_used`/`n_cameras_used` but never
    `triangulation`/`max_ray_disagreement_mm` (they are not ray-
    triangulation engines); `Ares` carries none of the four under
    these exact key names as of this writing. A caller (`save_
    throw_package()`) only fills a field from this dict's return value
    when its own already-computed value is still the honest "not
    populated" default (`None`/falsy) -- this function itself never
    overwrites anything, it only ever supplies what a caller is missing.
    """
    if not diagnostics:
        return {}
    winning_engine = diagnostics.get("winning_engine")
    sub_results = diagnostics.get("sub_results")
    if winning_engine is None or not isinstance(sub_results, dict):
        return {}
    winner = sub_results.get(winning_engine)
    if not isinstance(winner, dict):
        return {}
    winner_diagnostics = winner.get("diagnostics")
    if not isinstance(winner_diagnostics, dict):
        return {}
    out: dict = {}
    cameras_used = winner_diagnostics.get("cameras_used")
    if cameras_used is not None:
        out["cameras_used"] = list(cameras_used)
    triangulation = winner_diagnostics.get("triangulation")
    if triangulation is not None:
        out["triangulation"] = triangulation
    max_ray_disagreement_mm = winner_diagnostics.get("max_ray_disagreement_mm")
    if max_ray_disagreement_mm is not None:
        out["max_ray_disagreement_mm"] = max_ray_disagreement_mm
    n_cameras_used = winner_diagnostics.get("n_cameras_used")
    if n_cameras_used:
        out["n_cameras_used"] = n_cameras_used
    return out


def agreement_string_from_diagnostics(diagnostics: dict | None) -> str | None:
    """The v2 package schema (2026-08-27, "3 keys the spec
    missed"): recover an "X/Y usable sub-engines agree" string from a
    vote-based consensus
    engine's OWN diagnostics dict, never re-derive/re-count the vote here.
    `opendarts.engines.zeus.engine.ZeusEngine.score()` already computes
    `n_usable`/`winner`/`vote_tally` (the same numbers its own `reason`
    string already narrates in prose, e.g. "Zeus unanimous vote: 4/4
    usable sub-engines agree..."); this just reads them back out of the
    already-built `diagnostics` dict -- works identically whether that
    dict came from a live `EngineResult.diagnostics` or one already
    round-tripped through `.to_dict()`/JSON (the plain-dict shape is
    unchanged either way).

    Deliberately keyed on the DIAGNOSTICS SHAPE (`sub_engine_names`
    present), not on a hardcoded engine-name check -- opendarts's primary
    engine is operator-configurable, so any current or future
    vote-based engine with this same diagnostics shape is picked up for
    free, and any non-vote-based engine (Apollo/Talos/Athena/
    Ares alone, or a future engine with no vote
    diagnostics at all) correctly returns None -- never a fabricated
    "1/1" for an engine that never voted on anything.

    Returns None (never "0/0" or similar) when `diagnostics` is falsy or
    has no `sub_engine_names` key at all -- the "not a vote-based engine"
    case, not a data-quality failure. Returns f"0/{N}" (matching
    `_real_agreement_from_zeus()`'s own "0/3" convention) when voting WAS
    attempted but never reached quorum (`ZeusEngine.MIN_SUB_ENGINES_TO_
    VOTE`) -- `winner`/`vote_tally` are absent from `diagnostics` in that
    case, but `sub_engine_names`/`n_usable` are always present regardless
    of whether the vote succeeded (see `ZeusEngine.score()`'s own
    `base_diagnostics`).

    Two real call sites use this, both the v2 package schema's own
    "wherever the number already exists" instruction, never a third
    independent computation: `opendarts.live.capture_daemon.
    handle_ready_to_capture()` (the PRIMARY engine's raw `EngineResult`,
    when the primary itself is vote-based) and `write_other_engines_result()`
    immediately below (an also-run "Zeus" entry, ONLY when the primary
    path above didn't already produce one -- i.e. the primary engine
    wasn't vote-based but Zeus ran anyway as an also-run engine).
    """
    if not diagnostics:
        return None
    sub_engine_names = diagnostics.get("sub_engine_names")
    if sub_engine_names is None:
        return None
    n_usable = diagnostics.get("n_usable") or 0
    winner = diagnostics.get("winner")
    vote_tally = diagnostics.get("vote_tally") or []
    if not n_usable or winner is None:
        return f"0/{len(sub_engine_names)}"
    for entry in vote_tally:
        if [entry.get("sector"), entry.get("ring")] == list(winner):
            return f"{entry.get('count', 0)}/{n_usable}"
    return f"0/{n_usable}"


def write_other_engines_result(
    dest_dir: Path,
    primary_engine_name: str,
    other_engine_results: dict,
) -> None:
    """Adds `primary_engine`/`other_engines` to an ALREADY-SAVED package's
    `result.json` -- the multi-engine framework's "also-run" write, see
    docs/ENGINES.md's "Execution model" section. Called from
    `opendarts.live.capture_daemon`'s background also-run-engine dispatch,
    which mirrors `save_ad_ground_truth()`'s own established "attach more
    info after the fact, in a background thread, without touching the
    primary/top-level fields" pattern (see that function's docstring) --
    same discipline applied to `other_engines` instead of
    `ad_ground_truth.json`.

    Called EXACTLY ONCE per throw that has any also-run engine configured
    -- not incremental per-engine (docs/ENGINES.md: "one file, written
    once" for the combined also-run set) -- `opendarts.engines.dispatch.
    dispatch_engines()` already waits for every also-run engine to finish
    or hit its own timeout before this is ever called, so there is only
    ONE additional writer of `result.json` beyond the original
    `save_throw_package()` call, never several trickling in independently
    -- no incremental per-engine write races.

    Deliberately does NOT touch any of the OTHER top-level (primary)
    fields -- reads the existing dict, adds `primary_engine`/
    `other_engines`, and (2026-08-27, the v2 package schema)
    backfills `agreement` ONLY when it isn't already there. `other_engine_
    results` values may be `opendarts.engines.base.EngineResult` (the normal
    case -- `.to_dict()` used) or already a plain dict (e.g. a test
    fixture, or a value freshly loaded back off disk) -- accepted
    duck-typed the same way `save_ad_ground_truth()` accepts either an
    `AdGroundTruth` or a plain dict.

    `agreement` backfill (2026-08-27): `save_throw_package()` already set
    `agreement` when the PRIMARY engine itself was vote-based (see
    `agreement_string_from_diagnostics()`'s own docstring for the two real
    call sites). When it didn't -- the primary wasn't vote-based, but
    Zeus ("Zeus") is configured as an also-run engine anyway -- this is
    the ONLY other place that information ever becomes available, so it's
    filled in here from Zeus's own now-just-dispatched diagnostics,
    exactly once, never overwriting a value the primary path already set.
    Read straight off `data["other_engines"]["Zeus"]` (the SAME dict just
    built two lines above), not re-dispatched or re-computed a second way.

    `primary_engine`/`other_engines` KEYS are the real engine names, the
    same identities dispatch, the registry and Zeus's own vote use.
    """
    dest_dir = Path(dest_dir)
    result_path = dest_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"{result_path} does not exist -- write_other_engines_result() only "
            "attaches to an already-saved package's result.json, it never creates one"
        )
    data = json.loads(result_path.read_text())
    data["primary_engine"] = primary_engine_name
    data["other_engines"] = {
        name: (result.to_dict() if hasattr(result, "to_dict") else dict(result))
        for name, result in other_engine_results.items()
    }
    # The v2 package schema (2026-08-27) -- see this function's
    # own docstring's "agreement backfill" section. Only fires when the
    # primary engine wasn't vote-based (agreement absent so far) AND Zeus
    # actually ran as an also-run engine this throw.
    if "agreement" not in data:
        zeus_result = data["other_engines"].get("Zeus")
        if zeus_result is not None:
            backfilled = agreement_string_from_diagnostics(zeus_result.get("diagnostics"))
            if backfilled is not None:
                data["agreement"] = backfilled
    _write_json(result_path, data)


def _package_ring_boundary_offsets(
    pkg_derived: dict | None,
) -> tuple[float | None, float | None]:
    """THE V2 PACKAGE SCHEMA (2026-08-27) -- (treble_inner_offset_mm,
    double_inner_offset_mm) from a calibration package's
    `derived_calibration.json`, reading whichever real on-disk shape this
    specific package actually has. A package saved BEFORE the 2026-08-27
    reshape (`opendarts.capture.calibration_package.save_calibration_
    package()`) has the flat `treble_inner_offset_mm`/
    `double_inner_offset_mm` keys and nothing else; one saved AFTER it has
    ONLY the nested `ring_boundary_offset.boundaries.{treble_inner,
    double_inner}.offset_mm` path. Dispatches on KEY PRESENCE (`"treble_
    inner_offset_mm" in pkg_derived`) rather than a schema string --
    `derived_calibration.json`'s own `schema` field was NOT bumped by
    this reshape (this is a shape change, not a new schema version; see
    that module's own dated comment), so a schema-string check cannot
    distinguish the two real on-disk forms here the way it could for a
    coordinated, versioned flip. Returns `(None, None)` when `pkg_derived`
    is `None` or neither shape yields both values -- the caller's
    existing behavior (fall through to the session-level lookup) is
    unchanged either way."""
    if pkg_derived is None:
        return None, None
    if "treble_inner_offset_mm" in pkg_derived or "double_inner_offset_mm" in pkg_derived:
        return pkg_derived.get("treble_inner_offset_mm"), pkg_derived.get("double_inner_offset_mm")
    nested = pkg_derived.get("ring_boundary_offset") or {}
    boundaries = nested.get("boundaries") or {}
    treble = (boundaries.get("treble_inner") or {}).get("offset_mm")
    double = (boundaries.get("double_inner") or {}).get("offset_mm")
    return treble, double


def _package_board_color_threshold_value(
    pkg_derived: dict | None, flat_key: str, nested_key: str,
) -> float | None:
    """THE V2 PACKAGE SCHEMA (2026-08-27) -- one board-color
    threshold's value from a calibration package's `derived_calibration.
    json`, same key-presence dispatch as `_package_ring_boundary_
    offsets()` immediately above (see that function's own docstring for
    the full reasoning). `flat_key` is the OLD flat scalar name
    (`"brightness_threshold"`/`"chroma_threshold"` -- already the
    resolved, possibly-`None`-for-rejected value in the old shape, no
    further confidence check needed); `nested_key` is the matching key
    inside the NEW `board_color_calibration` block
    (`"brightness_threshold_black_cream"`/`"chroma_threshold"`), where a
    value is only adopted at `confidence == "high"`, mirroring exactly
    the gate the old flat writer already applied before ever writing a
    non-None value."""
    if pkg_derived is None:
        return None
    if flat_key in pkg_derived:
        return pkg_derived[flat_key]
    nested = pkg_derived.get("board_color_calibration") or {}
    entry = nested.get(nested_key) or {}
    return entry.get("value") if entry.get("confidence") == "high" else None


def load_throw_package(package_dir: Path) -> ThrowPackage:
    """Load a package's raw inputs back into memory. Loads bg/dart frames
    and calibrations (what replay actually needs) plus the original
    result (for comparison only -- see module docstring: replay must
    never use this to produce its own answer).

    Package `meta.json` files come in two shapes: some name the camera
    list `cameras`, others `frame_cameras`. Both are accepted, preferring
    `cameras` when both are present. Neither present is a real
    corruption/incompatibility case and raises loud (KeyError) rather
    than loading a silently empty package.

    The two keys can also disagree on purpose: `cameras` may list every
    camera on the rig while `frame_cameras` lists only those whose PNGs
    were actually written, so a package captured with a camera down has a
    `cameras` entry with no file behind it. When a `cameras` frame is
    missing and `frame_cameras` is a different list, retry with
    `frame_cameras`; still raise if those files are missing too.
    """
    import cv2

    package_dir = Path(package_dir)
    meta = json.loads((package_dir / "meta.json").read_text())
    if "cameras" in meta:
        cameras = meta["cameras"]
    elif "frame_cameras" in meta:
        cameras = meta["frame_cameras"]
    else:
        raise KeyError(
            f"{package_dir / 'meta.json'} has neither 'cameras' nor "
            "'frame_cameras' -- not a recognized throw-package schema"
        )

    # Two package shapes, both permanent. Since 2026-09-22 EVERY package
    # has a `video` block and no frame PNGs: the dart ("after") frame
    # lives in the per-camera clip -- a two-frame stills clip, or a
    # recorded window -- byte-identical to what was scored (the writer
    # read it back and compared before keeping it). Before that, a
    # non-recorded package had no clip at all and kept both frames as
    # PNGs; that is most of the existing corpus.
    # The bg ("before") frame comes from the clip too when the
    # pointer carries a `bg_index` (throw-clip/v2 onward), and from
    # cam{N}_bg.png otherwise -- a v1 package, or one whose bg could not be
    # de-duplicated into its clip, still has the PNG. Both paths yield the
    # byte-identical reference: this frame is not only what detection
    # diffed against, it is the input dev/calibration/recalibrate.py
    # re-solves a session's calibration FROM, so "close enough" is not a
    # thing here.
    video_meta = meta.get("video")

    def _read_frames(cams: list[int]):
        bg_frames = {}
        dart_frames = {}
        missing = []
        for cam in cams:
            bg = frame = None
            if video_meta and str(cam) in (video_meta.get("cameras") or {}):
                from opendarts.capture import clip
                # ONE decode for both frames: every package written since
                # 2026-09-22 keeps both inside its clip, and load is the
                # hot path of replay/rescore over the whole corpus.
                try:
                    bg, frame = clip.read_bg_and_commit_frames(
                        package_dir, video_meta, cam)
                except Exception:  # noqa: BLE001 -- a broken clip is a missing frame
                    bg = frame = None
            else:
                # A package from before clips existed at all: both frames
                # are PNGs. The existing corpus is hundreds of these, so
                # this branch is permanent, not transitional.
                frame = cv2.imread(str(package_dir / f"cam{cam}_frame.png"))
            if bg is None:
                # No bg pointer in the clip (throw-clip/v1, or a window
                # clip that could not de-duplicate it) -- the PNG is the
                # copy. A clip-less package reads it here too.
                bg = cv2.imread(str(package_dir / f"cam{cam}_bg.png"))
            if bg is None or frame is None:
                missing.append(int(cam))
                continue
            bg_frames[cam] = bg
            dart_frames[cam] = frame
        return bg_frames, dart_frames, missing

    bg_frames, dart_frames, missing = _read_frames(cameras)
    # Some packages list the rig's cameras=[0,1,2] even when a camera
    # dropped this throw; the actual PNGs are `frame_cameras`. Prefer
    # that list rather than raising on a documented 2-cam package
    # (measured: 1 of 374 throws, an S1 throw).
    if missing and "frame_cameras" in meta:
        alt = list(meta["frame_cameras"])
        if alt != list(cameras):
            bg_frames, dart_frames, missing = _read_frames(alt)
            cameras = alt
    if missing:
        raise IOError(
            f"missing/corrupt frames (clip or PNG) for camera {missing[0]} in {package_dir}"
        )

    calib_raw = json.loads((package_dir / "calibration.json").read_text())
    calibrations = {int(k): calibration_from_dict(v) for k, v in calib_raw.items()}
    # 2026-08-18 -- prefer a session-level calibration_refit.json when
    # present. Local import: opendarts.capture.
    # recalibrate itself imports load_throw_package from this module, so
    # a top-level import here would be circular. A missing/absent refit
    # (the common case -- most sessions/all synthetic test fixtures)
    # leaves `calibrations` exactly as loaded above, unchanged.
    from opendarts.capture.recalibrate import load_session_refit
    refit = load_session_refit(package_dir.parent)
    if refit is not None:
        calibrations = refit

    # LIVE-DERIVED RING-BOUNDARY OFFSET -- wired 2026-08-21, STOPPED BEING
    # APPLIED 2026-09-02 (see docs/DESIGN.md's dated entry for the full
    # decisive-test data). The
    # per-board measured inner-wire offset this section used to read back
    # and apply turned out to be measuring the wrong quantity entirely:
    # tip-measurement bias, not board geometry. A hypothetically PERFECT
    # wire-position detector scores exactly as badly (on the 241-throw
    # decisive test) as this rig's own worst live drift --
    # because wire position was never the scoring boundary in the first
    # place. `INNER_RING_SCORING_OFFSET_MM` (opendarts/geometry/board.py --
    # the historical, already-validated 1.5mm default) now stands for
    # BOTH inner wires, unconditionally, on every live AND replay path,
    # never gated behind confidence/acceptance/anything else (the
    # brief's own data: confidence does not predict correctness here).
    # `set_ring_boundary_offsets(None, None)` below is therefore
    # UNCONDITIONAL -- no longer read from a package/session file the
    # way it used to be.
    #
    # The calibration-package-level lookup immediately below
    # (`calibration_package_id`/`pkg_derived`) is deliberately KEPT, not
    # deleted -- `pkg_derived` also feeds the LIVE-DERIVED BOARD-COLOR
    # THRESHOLDS lookup a few lines down, an entirely separate,
    # unaffected derivation this task does not touch. The ring-boundary-
    # specific extraction (`_package_ring_boundary_offsets()`) is kept
    # only to LOG what this throw's calibration event actually measured,
    # for visibility -- never to apply it. The session-level `ring_
    # boundary_offset.json` lookup (`load_session_ring_boundary_offset()`)
    # is no longer consulted here at all -- it has nothing left to
    # contribute to this function now that neither source is ever
    # applied; that file is still written by the offline recalibrate/
    # full-corpus-measure tools (real diagnostic value, per the brief's
    # own "keep measuring and recording" instruction) and still read back
    # by other tooling that wants the raw measurement, just not by this
    # apply step.
    from opendarts.geometry.board import set_ring_boundary_offsets

    # CALIBRATION-PACKAGE-LEVEL LOOKUP, added 2026-08-22 -- still used
    # below for the board-color-threshold lookup; `meta.get(
    # "calibration_package_id")` is None for every package saved before
    # 2026-08-20 (confirmed: 125/185 real packages across this project's
    # full archive have no such field) and for any package saved with
    # calibration packages disabled -- both simply mean `pkg_derived`
    # stays `None`, same as "package not found"/"field not present",
    # never an error.
    calibration_package_id = meta.get("calibration_package_id")
    pkg_derived: dict | None = None
    if calibration_package_id is not None:
        from opendarts.capture.calibration_package import load_calibration_package_derived_values
        pkg_derived = load_calibration_package_derived_values(calibration_package_id)

    # THE V2 PACKAGE SCHEMA (2026-08-27) -- `_package_ring_
    # boundary_offsets()` dispatches on KEY PRESENCE (old flat keys vs
    # the new nested `ring_boundary_offset.boundaries.*` shape), same
    # discipline this project already established for `ad_ground_truth.
    # captured_at_utc`'s own interim-format window -- unchanged by this
    # task, still correct regardless of which format wrote this specific
    # package. Its result is now diagnostic-only (see the STOP APPLYING
    # comment above): logged when present, never passed to `set_ring_
    # boundary_offsets()`.
    treble_from_pkg, double_from_pkg = _package_ring_boundary_offsets(pkg_derived)
    if treble_from_pkg is not None or double_from_pkg is not None:
        log.debug(
            "%s: calibration package recorded a live-measured ring-boundary "
            "offset (treble_inner=%s double_inner=%s) -- NOT applied to "
            "scoring, per the 2026-09-02 stop-deriving change; this throw "
            "scores against the hardcoded INNER_RING_SCORING_OFFSET_MM "
            "default for both inner wires",
            package_dir, treble_from_pkg, double_from_pkg,
        )
    set_ring_boundary_offsets(None, None)

    # LIVE-DERIVED BOARD-COLOR THRESHOLDS, wired 2026-08-21 (item 6,
    # lowest risk -- a downstream sanity check, not primary scoring).
    # Same sibling-file / reset-on-absence pattern as the ring-boundary
    # offset just above. Per-threshold: only a `confidence == "high"`
    # derived value is adopted, matching this module's own
    # `ThresholdDerivation.confidence` field -- a lower-confidence
    # threshold falls back to the hardcoded default for THAT threshold
    # specifically rather than either blocking the other or being
    # trusted anyway.
    from opendarts.geometry.board_color import set_board_color_thresholds
    from opendarts.geometry.board_color_calibration import load_session_board_color_calibration

    # CALIBRATION-PACKAGE-LEVEL LOOKUP, added 2026-08-22 -- same REPLAY-
    # persistence reasoning as the ring-boundary-offset block above,
    # reusing the SAME `pkg_derived` this function already fetched.
    # Independent per-threshold (unlike ring-boundary-offset's joint
    # gate) -- matches the existing per-threshold confidence discipline
    # this function already applies to the session-level fallback below,
    # so a package with only ONE threshold accepted live still lets the
    # OTHER threshold fall through to the session-level file (if any)
    # rather than losing it just because its sibling was rejected.
    # THE V2 PACKAGE SCHEMA (2026-08-27) -- same reasoning as
    # `_package_ring_boundary_offsets()` immediately above (flat keys
    # gone from new writes, nested `board_color_calibration.*` is the
    # sole new-format source, key-presence dispatch, no interim window).
    brightness_value = _package_board_color_threshold_value(
        pkg_derived, "brightness_threshold", "brightness_threshold_black_cream",
    )
    chroma_value = _package_board_color_threshold_value(
        pkg_derived, "chroma_threshold", "chroma_threshold",
    )
    if brightness_value is None or chroma_value is None:
        color_payload = load_session_board_color_calibration(package_dir.parent)
        if color_payload is not None:
            if brightness_value is None:
                brightness = color_payload["brightness_threshold_black_cream"]
                brightness_value = brightness["value"] if brightness["confidence"] == "high" else None
            if chroma_value is None:
                chroma = color_payload["chroma_threshold"]
                chroma_value = chroma["value"] if chroma["confidence"] == "high" else None
    set_board_color_thresholds(
        brightness_threshold=brightness_value,
        chroma_threshold=chroma_value,
    )

    original_result = None
    result_path = package_dir / "result.json"
    if result_path.exists():
        original_result = json.loads(result_path.read_text())

    return ThrowPackage(
        package_dir=package_dir,
        session=meta["session"],
        cameras=cameras,
        bg_frames=bg_frames,
        dart_frames=dart_frames,
        calibrations=calibrations,
        original_result=original_result,
        ad_ground_truth=load_ad_ground_truth(package_dir),
        # .get(), not [] -- a package saved before the visit model
        # existed simply has no such key (see save_throw_package).
        visit_id=meta.get("visit_id"),
        visit_index=meta.get("visit_index"),
        capture_diagnostics=load_capture_diagnostics(package_dir),
        calibration_package_id=meta.get("calibration_package_id"),
        # the v2 package-schema additions (2026-08-26) -- .get(), not a
        # fabricated default: a package saved before these fields existed
        # simply has neither key, and must degrade to None (throw_number)
        # / None (frame_cameras, deliberately NOT []) rather than a
        # guessed value. Read straight off `meta` (the raw parsed dict,
        # not the possibly-reassigned `cameras` local above) -- this is
        # the real persisted `frame_cameras` value from THIS package's own
        # write, unrelated to the `cameras`/`frame_cameras`-key fallback
        # this function already does above to decide which PNG files to
        # actually load for OpenDarts-shaped packages.
        throw_number=meta.get("throw_number"),
        frame_cameras=meta.get("frame_cameras"),
        # The v2 package schema (2026-08-27) -- same .get(),
        # not-a-fabricated-default convention as throw_number/
        # frame_cameras immediately above: a package saved before this
        # field existed simply has no such key and degrades to None.
        generation=meta.get("generation"),
        # The v2 package schema (2026-08-27, the coordinated
        # flip) -- same .get() convention; a package saved before this
        # field existed -- every real package predating this change --
        # simply has no `schema` key and degrades to None. Not used
        # for any structural dispatch here (see META_SCHEMA_V2's own
        # comment) -- purely informational.
        schema=meta.get("schema"),
    )


AD_GROUND_TRUTH_FILENAME = "ad_ground_truth.json"


def save_ad_ground_truth(dest_dir: Path, ad_ground_truth) -> Path:
    """Write ``ad_ground_truth.json`` into an EXISTING package directory.

    Separate from ``save_throw_package()`` on purpose (see module
    docstring): AD ground truth is fetched as its own step, often after
    the throw package itself was already saved (typically via
    ``dev.ad.backfill_ad_ground_truth``). ``ad_ground_truth`` may be
    either an ``AdGroundTruth`` (has ``.to_dict()``) or a plain dict
    already in that shape -- accepted duck-typed so this module never
    needs a hard import of ``opendarts.live.ad_ground_truth`` at call time
    either.
    """
    dest_dir = Path(dest_dir)
    if not dest_dir.is_dir():
        raise FileNotFoundError(
            f"{dest_dir} is not an existing package directory -- "
            "save_ad_ground_truth() only attaches to an already-saved "
            "throw package, it does not create one"
        )
    payload = ad_ground_truth.to_dict() if hasattr(ad_ground_truth, "to_dict") else dict(ad_ground_truth)
    path = dest_dir / AD_GROUND_TRUTH_FILENAME
    _write_json(path, payload)
    return path


def load_ad_ground_truth(package_dir: Path) -> "AdGroundTruth | None":
    """Read ``ad_ground_truth.json`` if present, else None (a package with
    no AD ground truth -- either never fetched, or saved before this
    field existed -- must load fine; this is the backward-compat path).
    """
    from opendarts.live.ad_ground_truth import AdGroundTruth

    package_dir = Path(package_dir)
    path = package_dir / AD_GROUND_TRUTH_FILENAME
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return AdGroundTruth.from_dict(raw)


CAPTURE_DIAGNOSTICS_FILENAME = "capture_diagnostics.json"


def save_capture_diagnostics(dest_dir: Path, diagnostics: dict) -> Path:
    """Write ``capture_diagnostics.json`` into an EXISTING package
    directory -- 2026-08-16, "persist real diagnostics" task (see
    docs/DESIGN.md and this module's own docstring section for the real
    incident this responds to). Structural sibling of
    ``save_ad_ground_truth()`` above: a
    separate file, attached right after (per
    ``opendarts.live.capture_daemon.handle_ready_to_capture()``'s real
    wiring) the package's own ``save_throw_package()`` call, rather than
    a required argument to it -- this diagnostics dict is assembled from
    live, in-process state (the trigger's own settle timeline, the AD WS
    listener's buffer) that ``save_throw_package()`` itself has no
    knowledge of and shouldn't need to.

    ``diagnostics`` is written verbatim -- this function has no opinion
    on its shape (the real shape is documented in this module's own
    docstring and built by
    ``opendarts.live.capture_daemon._build_capture_diagnostics()``); a
    plain dict is all this needs, same duck-typed-at-the-boundary
    posture as ``save_ad_ground_truth()``
    (which additionally accepts a dataclass with ``.to_dict()`` -- not
    needed here since the caller already builds a plain dict).
    """
    dest_dir = Path(dest_dir)
    if not dest_dir.is_dir():
        raise FileNotFoundError(
            f"{dest_dir} is not an existing package directory -- "
            "save_capture_diagnostics() only attaches to an already-saved "
            "throw package, it does not create one"
        )
    path = dest_dir / CAPTURE_DIAGNOSTICS_FILENAME
    _write_json(path, diagnostics)
    return path


def load_capture_diagnostics(package_dir: Path) -> dict | None:
    """Read ``capture_diagnostics.json`` if present, else None -- the
    backward-compat path: every package saved before this field existed
    (or by a caller that doesn't build diagnostics, e.g. most offline
    tooling/tests) has no such file and must still load fine."""
    package_dir = Path(package_dir)
    path = package_dir / CAPTURE_DIAGNOSTICS_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def mark_operator_ad_wrong(
    dest_dir: Path,
    wrong: bool = True,
    note: str | None = None,
    confirmed_source: str | None = None,
    confirmed_sector: str | None = None,
    confirmed_ring: str | None = None,
) -> "AdGroundTruth":
    """Toggle a HUMAN judgment call onto a throw package: "the operator
    watched this throw happen and knows AD's own ground truth (or the
    total absence of it) was wrong here" -- something opendarts cannot
    determine algorithmically. Added 2026-08-12 in response to
    live-testing and hitting exactly this case: opendarts scored a throw T1,
    correctly, while AD's own ground truth disagreed/showed a miss; he
    asked for a real, persistent way to flag it, "not just a passing
    observation."

    An operator-triggered toggle persisted
    to a durable log. Reshaped for this project's own persistence model:
    opendarts has no append-only event log, so the durable record here
    IS the throw package's
    own ``ad_ground_truth.json`` (extended with two new fields --
    ``operator_marked_wrong``/``operator_note``, see ``AdGroundTruth``),
    not a second parallel file, per this task's own instruction.

    **A toggle, not a one-way flag**: ``wrong=True`` marks/re-marks (idempotent -- calling it
    twice in a row leaves the same end state, no error); ``wrong=False``
    un-marks a mistaken click and clears any note (the flag and its note
    are one annotation -- undoing the flag undoes the explanation too),
    also idempotent when the package was never marked in the first place.

    **"...then WHICH one was right?"**.
    ``confirmed_source``/``confirmed_sector``/``confirmed_ring`` carry
    that answer: the engine NAME the human picked (or the literal
    ``"manual"`` when they typed a segment in by hand), plus the segment
    itself in ``opendarts.geometry.board.sector_ring_for_point``'s own
    vocabulary. Persisted onto the SAME ``AdGroundTruth`` record (see its
    own ``operator_confirmed_*`` docstring) -- still one file per throw,
    not a parallel one. ``opendarts/live/server.py``'s ``discover_packages()``
    then prefers this over AD's own raw answer when scoring every engine
    row, which is the whole point of collecting it.

    These follow ``operator_note``'s EXACT existing convention, not a new
    one, because they're part of the same single annotation:
    ``wrong=True`` writes exactly what this call passed (so re-marking
    with no confirmation clears a previous one -- the write is idempotent
    on the whole annotation, never a partial merge), and ``wrong=False``
    clears all four operator fields together (undoing the flag undoes its
    entire explanation, confirmation included).

    Handles BOTH real shapes this needs to cover:

    1. **A package that already has ``ad_ground_truth.json``**. The existing record's real
       fetched ``sector``/``ring``/``tip_xy_mm``/etc. are preserved
       untouched -- only the two operator fields change.
    2. **A package with NO ``ad_ground_truth.json`` at all** (AD never
       registered the throw -- arguably the more literal reading of
       "AD MISSES", and a real possibility given
       ``opendarts/live/ad_ws_listener.py``'s own documented takeout-clear
       race). A minimal placeholder ``AdGroundTruth`` is created
       (``matched=False``, ``match_reason="operator_marked_no_ad_data"``,
       deliberately distinguishable from a real fetch attempt's own
       no-match reasons like ``"stale"``/``"fetch_error"``) purely so the
       operator flag has somewhere durable to live.

    **Documented trade-off for case 2**: this means
    ``discover_packages()``'s ``ad_matched`` field flips from ``None``
    ("never tried") to ``False`` ("tried, human says AD was wrong here")
    for a package flagged this way with no prior AD data at all. This is
    deliberate, not an oversight -- the file existing at all now means "a
    human has real, asserted information about this throw's AD ground
    truth", which is true and more useful than staying silently ``None``.

    Raises ``FileNotFoundError`` if ``dest_dir`` isn't an existing
    package directory -- same discipline as ``save_ad_ground_truth()``:
    this only annotates an already-saved throw package, it never creates
    one.
    """
    return _write_operator_annotation(
        dest_dir,
        caller="mark_operator_ad_wrong",
        marked_wrong=bool(wrong),
        # One annotation, written whole or cleared whole -- see this
        # function's own docstring. The `(x or None) if wrong else None`
        # shape is applied HERE (the policy), not in the shared writer
        # below (the mechanism), because it is specific to this public
        # function's toggle contract: record_throw_correction() below
        # deliberately does NOT clear a confirmation when it decides AD
        # itself was not wrong.
        note=(note or None) if wrong else None,
        confirmed_source=(confirmed_source or None) if wrong else None,
        confirmed_sector=(confirmed_sector or None) if wrong else None,
        confirmed_ring=(confirmed_ring or None) if wrong else None,
    )


def _write_operator_annotation(
    dest_dir: Path,
    *,
    caller: str,
    marked_wrong: bool,
    note: str | None,
    confirmed_source: str | None,
    confirmed_sector: str | None,
    confirmed_ring: str | None,
) -> "AdGroundTruth":
    """The ONE place a human annotation is actually written onto a throw
    package (2026-08-14). Extracted verbatim out of
    ``mark_operator_ad_wrong()`` when ``record_throw_correction()`` below
    was added, specifically so there is exactly one "patch a package's
    recorded truth" code path in this project rather than two that can
    drift -- the two public entry points differ only in the POLICY they
    apply before calling this (which fields to write vs. clear), never in
    the mechanism.

    Takes the five operator fields already resolved to their final
    values; writes them verbatim. Never touches ANY other field on an
    existing record (AD's own fetched sector/ring/tip_xy_mm/etc. survive
    untouched -- per docs/DESIGN.md's "Replay is the source of truth": a correction adds a
    human's answer alongside the machine's, it never overwrites it), and
    never touches ``result.json`` at all, so the original engine call
    stays exactly as scored live.

    ``caller`` only shapes the FileNotFoundError message, so a failure
    still names the public function the caller actually called.
    """
    from opendarts.live.ad_ground_truth import AdGroundTruth

    dest_dir = Path(dest_dir)
    if not dest_dir.is_dir():
        raise FileNotFoundError(
            f"{dest_dir} is not an existing package directory -- "
            f"{caller}() only annotates an already-saved "
            "throw package, it does not create one"
        )

    existing = load_ad_ground_truth(dest_dir)
    if existing is None:
        existing = AdGroundTruth(
            matched=False,
            match_reason="operator_marked_no_ad_data",
            ad_base_url="",
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            opendarts_captured_at_utc=None,
            staleness_sec=None,
            window_sec=0.0,
        )

    existing.operator_marked_wrong = marked_wrong
    existing.operator_note = note
    existing.operator_confirmed_source = confirmed_source
    existing.operator_confirmed_sector = confirmed_sector
    existing.operator_confirmed_ring = confirmed_ring

    save_ad_ground_truth(dest_dir, existing)
    return existing


def record_throw_correction(
    dest_dir: Path,
    sector: str | None,
    ring: str,
    source: str = "manual",
    note: str | None = None,
) -> "AdGroundTruth":
    """A GAME-DRIVER correction: "the dart that was scored here actually
    landed in <sector, ring>" (2026-08-14, added alongside the visit
    model -- see ``POST /api/visits/{visit_id}/throws/{index}/correct`` in
    ``opendarts/live/server.py`` and docs/LIVE_API.md), shaped for this
    project's own persistence model.

    **Deliberately NOT a second patching mechanism.** It writes through
    the exact same ``_write_operator_annotation()`` /
    ``ad_ground_truth.json`` path ``mark_operator_ad_wrong()`` uses, and
    lands in the same ``operator_confirmed_source/_sector/_ring`` fields
    ``opendarts/live/server.py``'s ``_operator_truth_for()`` already grades
    every engine row against. The difference is only WHO is asserting and
    WHEN (a live game driver correcting the current visit vs. an operator
    reviewing history in the Scoring tab), not what gets written.

    **The original engine call is never touched.** ``result.json`` --
    including the live ``sector``/``ring``/``board_xy_mm`` the primary
    engine produced, and every also-run engine's section -- is not read
    or written here at all. That is the whole point per docs/DESIGN.md's
    REPLAY principle: replaying this package through newer code must
    still reproduce/compare against what the engine actually said, with
    the human's answer sitting alongside it as truth, not in place of it.

    ``operator_marked_wrong`` is DERIVED, not asserted by the caller: it
    means "AD's own answer for this throw disagrees with what the human
    just said was right," which is exactly computable from the record
    already on disk (True as well when there is no matched AD answer at
    all -- there is then nothing that agrees with the correction). This
    is why this function does not simply call ``mark_operator_ad_wrong``:
    that one's contract ties the confirmation to the flag (``wrong=False``
    clears the confirmation), which would silently discard a correction
    on any throw AD happened to get right -- precisely the throws where
    OUR engine was the wrong one and the correction matters most.

    Idempotent: correcting the same throw twice leaves the same end
    state, last write wins -- a correction endpoint must be safe to call
    twice.
    """
    existing = load_ad_ground_truth(dest_dir)
    ad_disagrees = not (
        existing is not None
        and existing.matched
        and existing.sector == sector
        and existing.ring == ring
    )
    return _write_operator_annotation(
        dest_dir,
        caller="record_throw_correction",
        marked_wrong=ad_disagrees,
        note=note or None,
        confirmed_source=source or "manual",
        confirmed_sector=sector or None,
        confirmed_ring=ring,
    )
