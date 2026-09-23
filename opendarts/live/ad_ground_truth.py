"""Autodarts' (AD) own committed-throw ground truth: the RECORD, its
schema, and the parsers that turn AD's payload into this project's
vocabulary.

Why this exists: AD is this project's ground truth for day 1 ("on day
1, AD is your truth").
A opendarts-captured throw package carries a trusted reference
sector/tip position alongside its own triangulated result, in
`ad_ground_truth.json`, and this module defines what that file is.

WHAT LIVES HERE AND WHAT DOES NOT. This module is the record and the
parsing, with no I/O of its own: `AdGroundTruth` (+ its schema and
validation), `segment_to_sector_ring()` (AD's `{name, number, bed,
multiplier}` -> this project's `(sector, ring)`), `_tip_xy_mm()` (AD's
normalized board units -> mm) and the shared defaults. Deliberately
dependency-light -- no numpy, no cv2, and no HTTP client.

WHERE IT COMES FROM AT RUN TIME. The live path is
`opendarts.live.ad_ws_listener`: a WebSocket listener told when AD
committed a throw, which stamps its own receive time and matches on it.
That is the only fetcher the product ships.

There was an earlier REST poller that asked
`/api/state/detections` for AD's most recent throw and decided from
timing alone whether it was the same dart. It could never separate "AD
committed late" from "AD committed a DIFFERENT dart moments later", so
the listener replaced it. It still exists for backfilling a session
captured with the oracle off, as `dev/ad/ad_ground_truth_rest.py`, and
imports its record type and parsers from here -- one definition of the
on-disk file, whichever path wrote it.

## API shape

Both paths read the same fields out of AD's throw payload (others are
ignored):

    {
        "method": "...", # stored as an opaque string
        "bouncer": false,
        "coords": {"x": <float>, "y": <float>}, # normalized board units
        "segment": {"name": "S17", "number": 17, "bed": "SingleInner",
                     "multiplier": 1},
    }

`coords.x`/`coords.y` are normalized board units where 1.0 == the
double-outer ring radius; convert to mm via `* 170.0` (this is not an
AD-specific magic number -- it is the board's own
`DOUBLE_OUTER_RADIUS_MM` from `opendarts.geometry.board`, confirmed
equal to 170.0, see `AD_BOARD_UNIT_TO_MM` below).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

DEFAULT_AD_BASE = "http://localhost:3180"

# How long after opendarts's own capture timestamp we still trust AD's most
# recent /api/state/detections entry as "probably the same throw".
#
# UNMEASURED against the real rig (same "don't pretend a guess is a
# tuned number" discipline as opendarts/live/server.py's own
# DEFAULT_PACKAGE_POLL_INTERVAL_SECONDS) -- chosen by reasoning, not by
# fitting to real timing data, because it was built and tested offline.
# Picked
# as a deliberate middle ground between two failure modes:
# - too narrow: opendarts's own settle-then-capture latency
# (throw_trigger.py's SETTLING window) plus this module's own fetch
# latency (network round-trip to the rig) can easily add up to a
# few real seconds between "dart physically landed" and "we asked
# AD about it" -- a window that's too tight would produce spurious
# "stale, no match" results even when AD really did commit the same
# throw we just captured.
# - too wide: a human's next dart in the same visit typically lands
# within a few seconds of the previous one settling; too wide a
# window risks confidently attaching the WRONG (later) AD throw to
# an EARLIER opendarts package.
# 12 seconds is generous relative to expected settle+network latency
# (low single-digit seconds) while still being tight relative to
# realistic inter-dart timing in a visit (rarely under ~2-3s for a human
# throwing three darts). MUST be re-validated against real recorded
# inter-dart timing on a rig with Autodarts running --
# noted as an open question, not silently assumed correct.
DEFAULT_MATCH_WINDOW_SEC = 12.0

DEFAULT_TIMEOUT_SEC = 5.0

# Board geometry constant AD's normalized coords are scaled by to reach
# mm -- equal to opendarts.geometry.board.DOUBLE_OUTER_RADIUS_MM. Not
# imported from there to keep this module dependency-light (no numpy/cv2
# pulled in just to fetch an HTTP response); the two are asserted equal
# in tests/test_ad_ground_truth.py instead, so they can't silently drift
# apart.
AD_BOARD_UNIT_TO_MM = 170.0

# The v2 package schema.
#
# **The schema BUMP LANDS NOW, 2026-08-27** --
# held back through three prior rounds specifically so it could land only
# once opendarts's own real gaps (throw_number/frame_cameras, camera_mode/
# label/agreement, the rollup/generation/reason fixes, and this same
# round's null-vs-[] fixes above) were actually in place -- see this
# module's own git history / docs/DESIGN.md for that sequencing. Landing it
# early would have made the version string assert a guarantee that wasn't
# true yet, poisoning the one thing a version string is FOR (drift
# detection). `to_dict()` below now writes this literal.
#
# **Real correction to the literal VALUE, made in this same round**: an
# earlier draft of the v2 package schema said "schema -> dart-package/v2,
# both sides" without saying which FILE -- ambiguous, since two files
# (`meta.json`, `ad_ground_truth.json`) both needed a version marker and
# had never had one. It was later resolved into
# TWO distinct fields with TWO distinct values: `meta.schema =
# "dart-package/v2"` (a brand-new field, see
# `opendarts.capture.throw_package.META_SCHEMA_V2`) and
# `ad_ground_truth.schema = "ad-ground-truth-v2"` (a bump of the existing
# per-file `schema` key, unifying two differently-named predecessors --
# `"ad-ground-truth-v1"` and `"autodarts-ground-truth-v1"` -- under one
# shared name). This constant was originally defined (2026-08-26) holding the
# WRONG value for this file (`"dart-package/v2"`, the ambiguous early
# draft's literal) -- corrected here to the real, current spec's value.
AD_GROUND_TRUTH_SCHEMA_V2 = "ad-ground-truth-v2"
# 2026-08-27 (this round's null-vs-[] follow-up): the `AD_GROUND_TRUTH_
# SCHEMA_CURRENT = "ad-ground-truth-v1"` constant that used to live here
# (the value `to_dict()` wrote before the flip above) was confirmed
# genuinely dead in production code -- nothing reads it at runtime any
# more (`_resolve_ad_ground_truth_captured_at_utc()` below dispatches on
# KEY PRESENCE, not the schema string). It was still referenced by 4
# test fixtures as a literal "the old schema value" -- those now use the
# bare literal `"ad-ground-truth-v1"` directly (QA's own suggested
# tripwire cleanup: a dead constant only a test still imports is worth
# removing so nothing accidentally treats it as a live dispatch target
# again). See tests/test_package_schema_v2.py / test_package_schema_v2f.py.
# Retro-label for a genuinely pre-existing package -- anything that
# isn't literally `AD_GROUND_TRUTH_SCHEMA_V2`
# (missing key, the old `"ad-ground-truth-v1"` literal, or anything
# unrecognized) retro-labels as this.
AD_GROUND_TRUTH_SCHEMA_V1_OpenDarts = "v1-opendarts"


def _ad_ground_truth_schema_version(d: dict[str, Any]) -> str:
    """Real dispatch target, now reachable: `to_dict()` writes
    `AD_GROUND_TRUTH_SCHEMA_V2` as of this round's flip, so a payload built
    from a live `AdGroundTruth` now actually returns
    `AD_GROUND_TRUTH_SCHEMA_V2` here, not just `AD_GROUND_TRUTH_SCHEMA_
    V1_OpenDarts` for everything (the case before the flip landed). Still NOT
    used by `from_dict()`/`_resolve_ad_ground_truth_captured_at_utc()`
    below to resolve which key holds the capture timestamp -- see that
    function's own comment for why key presence, not this schema string,
    remains the correct signal even after the flip (real packages exist
    on disk in an interim state -- old schema string, new key name --
    that a schema-string dispatch would misread)."""
    schema = d.get("schema")
    if schema == AD_GROUND_TRUTH_SCHEMA_V2:
        return schema
    return AD_GROUND_TRUTH_SCHEMA_V1_OpenDarts


def _resolve_ad_ground_truth_captured_at_utc(d: dict[str, Any]) -> str | None:
    """Resolve the v2 package schema's renamed field
    (`opendarts_captured_at_utc` -> `captured_at_utc`) by KEY PRESENCE, not
    by dispatching on `schema` -- deliberately, and this remains true even
    after this round's schema-string flip lands (checked, not assumed,
    before simplifying this away -- see below). Three real states exist
    across packages this module must keep reading forever (REPLAY, per
    docs/DESIGN.md's "Replay is the source of truth"): (1) a genuinely old package, schema
    `"ad-ground-truth-v1"`, key `opendarts_captured_at_utc` only; (2) an
    INTERIM package written between the key-rename round landing
    (2026-08-26) and this round's schema bump -- schema STILL
    `"ad-ground-truth-v1"`, but key ALREADY `captured_at_utc` (real
    production packages exist in exactly this state, captured over that
    window, and REPLAY still applies to whichever of them remain or get
    pulled later per this task's own instructions); (3) a fresh package
    from this round onward -- schema `"ad-ground-truth-v2"`, key
    `captured_at_utc`. A schema-string dispatch cannot distinguish (1)
    from (2) -- both claim the OLD schema literal, but only (1) actually
    has the old key -- so it would silently read `None` off every real
    state-(2) package. Key presence has no such gap in any of the three
    states: prefers the new key when present (states 2 and 3), falls back
    to the old key otherwise (state 1) -- correct regardless of what the
    schema string claims. NOT safe to simplify to schema-string dispatch
    while state-(2) packages can still exist on disk/get pulled later."""
    if "captured_at_utc" in d:
        return d.get("captured_at_utc")
    return d.get("opendarts_captured_at_utc")


def validate_ad_ground_truth_v2(payload: dict[str, Any]) -> None:
    """Schema-validate an already-serialized `AdGroundTruth.to_dict()`
    payload against the shape this project's v2 package-schema work adds
    -- the v2 package schema's "validate on write, in CI", the
    ad_ground_truth.json half (see
    `opendarts.capture.throw_package.validate_throw_package_meta_v2()` for
    the meta.json half). Raises `AssertionError` naming the exact
    violation.

    Now asserts `payload["schema"] == AD_GROUND_TRUTH_SCHEMA_V2` too
    (2026-08-27) -- the bump held back through three prior rounds
    specifically so this assertion could eventually be added without
    failing every real write; it is added in the SAME commit as the flip
    itself, never before it (see AD_GROUND_TRUTH_SCHEMA_V2's own comment).
    """
    assert "captured_at_utc" in payload, (
        f"ad_ground_truth.json missing 'captured_at_utc' (the v2 package schema "
        f"-- the frame-capture-time field, plain name): {payload!r}"
    )
    assert "opendarts_captured_at_utc" not in payload, (
        "ad_ground_truth.json must not write the old 'opendarts_captured_at_utc' "
        f"alias any more, per the v2 package schema: {payload!r}"
    )
    assert payload.get("schema") == AD_GROUND_TRUTH_SCHEMA_V2, (
        "ad_ground_truth.json 'schema' must be the bumped "
        f"{AD_GROUND_TRUTH_SCHEMA_V2!r} literal, per the v2 package "
        f"schema: {payload!r}"
    )


@dataclass
class AdGroundTruth:
    """One AD committed throw's ground truth, normalized into opendarts's
    own (sector, ring) vocabulary (opendarts.geometry.board.sector_ring_for_point's
    return shape) so it's directly comparable to a opendarts ScoreResult --
    not the sN/SN/TN/DN spelling AD itself uses.
    """

    matched: bool
    match_reason: str

    ad_base_url: str
    fetched_at_utc: str
    opendarts_captured_at_utc: str | None
    staleness_sec: float | None
    window_sec: float

    sector: str | None = None
    ring: str | None = None
    tip_xy_mm: tuple[float, float] | None = None

    ad_method: str | None = None
    ad_bouncer: bool | None = None
    ad_n_cam_detections: int | None = None
    raw_segment: dict[str, Any] | None = None
    source_index: int | None = None
    n_detections_in_response: int | None = None

    # Human-asserted "AD was wrong on this throw" flag -- NOT something
    # opendarts determines algorithmically.
    # See opendarts.capture.throw_package.mark_operator_ad_wrong -- the only
    # code that should ever set these -- for the full mechanism.
    # Both default False/None so every package's
    # ad_ground_truth.json written before this field existed loads
    # unchanged (from_dict() below degrades missing keys to these same
    # defaults) -- backward compatible by construction.
    operator_marked_wrong: bool = False
    operator_note: str | None = None

    # WHICH answer the human says was actually right on this throw
    #. Extends THIS
    # dataclass rather than introducing a parallel file, for exactly the
    # same reason operator_marked_wrong/operator_note above do -- the
    # throw package's own ad_ground_truth.json is this project's one
    # durable per-throw truth record, and splitting a flag from its own
    # explanation across two files is how they drift.
    #
    # operator_confirmed_source: the engine NAME whose answer the human
    # picked (e.g. "Talos"), or the literal "manual" when the human
    # entered a sector/ring by hand because no engine had it right.
    # operator_confirmed_sector/_ring: the confirmed answer itself, in
    # opendarts.geometry.board.sector_ring_for_point's own vocabulary
    # (sector is the wedge number as a string, or None for
    # bull/outer_bull/outside; ring is one of bull/outer_bull/treble/
    # double/single_inner/single_outer/outside).
    #
    # There is deliberately NO confirmed tip_xy_mm: a human confirming
    # "Talos had the right segment" is asserting a SEGMENT, not a
    # millimetre coordinate -- see opendarts/live/server.py's
    # _operator_truth_for() (which builds a truth object with
    # tip_xy_mm=None from these fields) and docs/DESIGN.md's 2026-08-12
    # behavioral-correction entry making exactly that distinction about
    # AD-as-truth ("x/y millimetre deltas... are still not independently
    # verified; crossing a sector/ring boundary vs AD is").
    #
    # All three default None so every ad_ground_truth.json written before
    # they existed loads unchanged -- same backward-compat-by-
    # construction convention as the two fields above (from_dict() below
    # degrades a missing key to None).
    operator_confirmed_source: str | None = None
    operator_confirmed_sector: str | None = None
    operator_confirmed_ring: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            # The v2 package schema (2026-08-26) -- the on-disk
            # JSON key renames to the plain `captured_at_utc`. The Python
            # attribute name stays `opendarts_captured_at_utc` (many call
            # sites across opendarts.live.ad_ws_listener/this module use it
            # as a function parameter name too -- renaming those is out
            # of scope; only the PERSISTED key name changes).
            # `schema` bumped 2026-08-27 (the coordinated flip) -- see AD_GROUND_TRUTH_SCHEMA_V2's own
            # module-level comment for the full held-back-then-landed
            # story and the real value correction made in this same round.
            "schema": AD_GROUND_TRUTH_SCHEMA_V2,
            "matched": self.matched,
            "match_reason": self.match_reason,
            "ad_base_url": self.ad_base_url,
            "fetched_at_utc": self.fetched_at_utc,
            "captured_at_utc": self.opendarts_captured_at_utc,
            "staleness_sec": self.staleness_sec,
            "window_sec": self.window_sec,
            "sector": self.sector,
            "ring": self.ring,
            "tip_xy_mm": None if self.tip_xy_mm is None else list(self.tip_xy_mm),
            "ad_method": self.ad_method,
            "ad_bouncer": self.ad_bouncer,
            "ad_n_cam_detections": self.ad_n_cam_detections,
            "raw_segment": self.raw_segment,
            "source_index": self.source_index,
            "n_detections_in_response": self.n_detections_in_response,
            "operator_marked_wrong": self.operator_marked_wrong,
            "operator_note": self.operator_note,
            "operator_confirmed_source": self.operator_confirmed_source,
            "operator_confirmed_sector": self.operator_confirmed_sector,
            "operator_confirmed_ring": self.operator_confirmed_ring,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AdGroundTruth":
        tip = d.get("tip_xy_mm")
        # The v2 package schema (2026-08-26): resolved by KEY
        # PRESENCE, not by dispatching on `schema` -- see
        # `_resolve_ad_ground_truth_captured_at_utc()`'s own docstring for
        # why (the schema bump is deliberately held back while the key
        # rename already lands, so the schema string alone can't
        # currently distinguish old-key from new-key packages).
        captured_at_utc = _resolve_ad_ground_truth_captured_at_utc(d)
        return AdGroundTruth(
            matched=bool(d.get("matched")),
            match_reason=str(d.get("match_reason") or ""),
            ad_base_url=str(d.get("ad_base_url") or ""),
            fetched_at_utc=str(d.get("fetched_at_utc") or ""),
            opendarts_captured_at_utc=captured_at_utc,
            staleness_sec=d.get("staleness_sec"),
            window_sec=float(d.get("window_sec") or DEFAULT_MATCH_WINDOW_SEC),
            sector=d.get("sector"),
            ring=d.get("ring"),
            tip_xy_mm=None if tip is None else (float(tip[0]), float(tip[1])),
            ad_method=d.get("ad_method"),
            ad_bouncer=d.get("ad_bouncer"),
            ad_n_cam_detections=d.get("ad_n_cam_detections"),
            raw_segment=d.get("raw_segment"),
            source_index=d.get("source_index"),
            n_detections_in_response=d.get("n_detections_in_response"),
            # bool(None or False) == False -- a package saved before this
            # field existed has no "operator_marked_wrong" key at all, so
            # this degrades to the honest default rather than raising.
            operator_marked_wrong=bool(d.get("operator_marked_wrong") or False),
            operator_note=d.get("operator_note"),
            # Same backward-compat path as the two fields above: a package
            # saved before these keys existed simply has no such key, and
            # .get() degrades it to the honest None default.
            operator_confirmed_source=d.get("operator_confirmed_source"),
            operator_confirmed_sector=d.get("operator_confirmed_sector"),
            operator_confirmed_ring=d.get("operator_confirmed_ring"),
        )


def _tip_xy_mm(detection: dict[str, Any]) -> tuple[float, float] | None:
    coords = detection.get("coords") or {}
    if "x" not in coords or "y" not in coords:
        return None
    try:
        return (
            float(coords["x"]) * AD_BOARD_UNIT_TO_MM,
            float(coords["y"]) * AD_BOARD_UNIT_TO_MM,
        )
    except (TypeError, ValueError):
        return None


def segment_to_sector_ring(segment: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Map AD's raw ``segment`` dict ({name, number, bed, multiplier}) into
    opendarts's own (sector, ring) vocabulary --
    opendarts.geometry.board.sector_ring_for_point()'s return shape: ring in
    {"bull", "outer_bull", "treble", "double", "single_inner",
    "single_outer", "outside"}, sector is the wedge number as a string or
    None when no wedge applies (bull/outer_bull/outside).

    Expressed in this project's ring vocabulary rather than the
    external sN/SN/TN/DN/D25/25 spelling -- see this module's own
    docstring for why: this project's ScoreResult uses the ring-vocab
    convention, so THIS is the comparable shape.

    Returns (None, None) -- not a raise -- for a segment with no
    recognizable bed/name (defensive: an AD field-name change should
    degrade to "no ground truth extracted", never crash the fetch path).
    """
    if not segment:
        return None, None

    bed = str(segment.get("bed") or "")
    name = str(segment.get("name") or "")
    number = segment.get("number")
    try:
        number_i = int(number) if number is not None else None
    except (TypeError, ValueError):
        number_i = None
    if number_i is None and len(name) >= 2 and name[1:].isdigit():
        number_i = int(name[1:])

    if bed == "SingleInner" and number_i is not None:
        return str(number_i), "single_inner"
    if bed == "SingleOuter" and number_i is not None:
        return str(number_i), "single_outer"
    if bed == "Triple" and number_i is not None:
        return str(number_i), "treble"
    if bed == "Double" and number_i == 25:
        return None, "bull"
    if bed == "Double" and number_i is not None:
        return str(number_i), "double"

    try:
        mult_i = int(segment.get("multiplier")) if segment.get("multiplier") is not None else None
    except (TypeError, ValueError):
        mult_i = None

    if (
        bed in ("InnerBull", "BullsEye", "Bullseye")
        or name in ("D25", "DB")
        or (name in ("Bull", "BULL") and (mult_i == 2 or bed == "Double"))
    ):
        return None, "bull"
    if (
        bed in ("OuterBull", "SB")
        or name in ("25", "SB", "B")
        or (name in ("Bull", "BULL") and mult_i == 1)
    ):
        return None, "outer_bull"

    # Miss / outside (AD names these "M1"/"M2"/"M3"/"M4" for the 4
    # missed-board quadrants) -- no sector/ring in opendarts's vocabulary,
    # but not an error either.
    return None, "outside"


def _parse_iso(ts: str) -> datetime | None:
    try:
        s = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None
