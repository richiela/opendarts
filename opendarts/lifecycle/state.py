"""The lifecycle state machine.

One :class:`Lifecycle` object per camera session. Call
:meth:`Lifecycle.observe` once per pump tick with the full-resolution
frames of every camera; it returns a :class:`Tick` describing the phase,
any action to take (commit a dart, clear the visit, hand entered/left),
and the per-camera signals that produced the decision.

Decision model (all thresholds in :class:`LifecycleConfig`, small-scale
pixel units unless stated otherwise):

    stable       every camera's frame-to-frame change is small
    hand         some camera sees a LARGE and MOVING change outside the
                 board for a few consecutive frames -- a person, not a
                 dart's static flight over the rim
    scene        some camera sees a change so wide it can't be a dart or
                 a hand (lighting step) -> re-adopt, never commit
    dart cams    cameras whose changed board pixels (vs the reference,
                 committed-dart union NOT subtracted) reach
                 ``dart_min_px``, with the change not dominated by the
                 union itself -- something new is on the board, in that
                 camera
    removal      most of the committed-dart union differs from the
                 reference and nothing new is on the board -> the darts
                 we committed are gone

Priority per tick: cooldown/warmup > scene > hand > dart > removal > quiet.

A dart is committed once a dart camera has been seen on
``dart_stable_frames`` frames (moving or not -- a settling dart is
evidence) and the board is currently stable, unless a camera shows
something sitting on top of the committed darts (hand pulling them) --
then we hold, until
that overlap itself stays stable and hand-free for
``over_darts_hold_frames`` (a grouped dart, not a takeout). A board
change that is still present when a hand leaves was made BY the hand
(pull, nudge, placement): it is never committed; with darts on the
board it clears the visit, otherwise it is absorbed. After a
commit or a clear the machine cools down for a few frames, re-adopting
the reference every tick so settling wobble and the arm's retreat are
absorbed instead of re-triggering. While the board is quiet the
reference is re-adopted on every stable frame, so exposure drift is
absorbed the moment it appears and can never accumulate into a
phantom.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from opendarts.capture.lazy_frame import as_frames, handles_of, pixels_of
from opendarts.lifecycle import same_dart
from opendarts.lifecycle.reference import ReferenceSet
from opendarts.lifecycle.signals import (
    DEFAULT_SIGNAL_CONFIG,
    CamSignals,
    SignalConfig,
    change_mask,
    compute_signals,
    to_small_gray,
    to_small_mask,
)


#: ``_clear`` reasons for a takeout cleared because the board matches the
#: empty board again (the driver logs each at INFO)
CLEARED_NOT_PARTIAL = "board matches the empty board, not a partial removal"
CLEARED_STRANDED = "board matches the empty board, was stuck waiting for a takeout"
#: a clear made because the board changed under a hand -- nothing verified empty
CLEARED_UNDER_HAND = "board changed under hand"


class Phase(enum.Enum):
    WARMUP = "warmup"
    IDLE = "idle"
    PENDING_DART = "pending_dart"
    HAND = "hand"
    SCENE_CHANGE = "scene_change"
    TAKEOUT_PENDING = "takeout_pending"
    COOLDOWN = "cooldown"


class Action(enum.Enum):
    NONE = "none"
    COMMIT = "commit"
    CLEARED = "cleared"
    HAND_ENTER = "hand_enter"
    HAND_EXIT = "hand_exit"
    SCENE_CHANGE = "scene_change"
    READOPT = "readopt"
    PARTIAL_ADOPT = "partial_adopt"
    FORCED_ADOPT = "forced_adopt"
    READY = "ready"


@dataclass(frozen=True)
class LifecycleConfig:
    signals: SignalConfig = DEFAULT_SIGNAL_CONFIG

    # --- dart -----------------------------------------------------------
    #: changed board pixels vs the reference (committed-dart union NOT
    #: subtracted) in ONE camera that count as "something new on the
    #: board". Corpus, best camera per throw, this pipeline's own change
    #: mask with the board mask applied (3,418 historical packages plus
    #: 141 current-rig packages = 3,559 real darts): min 92, p1 ~170,
    #: p5 ~245, p50 ~465. The one captured ghost (a room light switched
    #: off; single camera, others at literal zero) read 67; the one
    #: missed adjacent dart read 132 on its best camera. 80 sits mid-gap:
    #: every recorded real dart passes, the ghost fails. Honest limits:
    #:   * 92 is the weakest DETECTED dart, almost certainly a third dart
    #:     (signal declines through the visit: current-rig minima 298 /
    #:     227 / 127 for darts 1/2/3). Darts too weak to detect were
    #:     never written to any corpus, so the true low tail extends
    #:     below 92 by an unknown amount -- hence 80, not 90: headroom
    #:     below the observed minimum, not a measured boundary.
    #:   * the separation window (67, 92] is only 25 px wide, and this
    #:     floor does not defend alone: sub-threshold artifacts are also
    #:     absorbed by the every-stable-frame reference refresh (quiet
    #:     branch of ``_step``). A brighter single-camera artifact
    #:     (~100 px) WOULD commit. Accepted trade, not a solved problem.
    #: There is deliberately NO blob-size floor and NO union subtraction
    #: here: both destroyed the evidence for a dart landing against an
    #: already-committed neighbour (the union swallowed most of its
    #: pixels and fragmented the remainder into sub-floor blobs, so it
    #: was never scored at all).
    #: 2026-10-05, lowered 80 -> 60. On a rig with low cameras a third dart
    #: grouped behind the first two read 78/61/26 and 9/63/72 px and was
    #: never scored (the oracle scored both, S20/T20). Replayed over every
    #: logged session on both rigs: at 60 both are caught, and every other
    #: change the lower floor admits is seen by one camera only (the second
    #: at <= 6 px), which the second-camera rule below already refuses -- no
    #: new commits. The floor is a trigger; the one-camera rule is what
    #: keeps ghosts out (see docs/LIFECYCLE.md).
    dart_min_px: int = 60
    #: frames of dart evidence -- a camera passing the dart test, stable
    #: or NOT -- required before a commit; the commit itself additionally
    #: waits for a currently-stable frame. This is deliberately not
    #: "consecutive stable frames": a thrown dart is visible while still
    #: settling, and resetting the counter on every unstable frame threw
    #: the entire arrival away. The counter resets only when the dart
    #: test fails, on commit, and on clear. Operator-tunable live as the
    #: dashboard's "Detection time" (:mod:`opendarts.lifecycle.settings`,
    #: range 1..5). Default 2, carried over from the previous
    #: consecutive-stable semantics (84 live darts replayed: 2 frames
    #: committed one tick -- ~33 ms median -- sooner than 3, zero
    #: ghosts); under evidence semantics the typical commit lands one
    #: further tick earlier, because the arrival frame now counts.
    dart_stable_frames: int = 2
    #: give up waiting for stability and commit anyway after this many
    #: frames of a persistent board change (flapping flight, vibration).
    dart_force_frames: int = 45
    #: hold the commit while some camera sees this fraction of the
    #: committed-dart union changed (something on top of the darts).
    over_darts_coverage: float = 0.5
    #: a change over the committed darts that stays stable this many
    #: consecutive frames with no hand signal anywhere is a grouped
    #: dart, not a hand pulling darts; commit it.
    over_darts_hold_frames: int = 6

    # --- stability ------------------------------------------------------
    stable_max_board_delta_px: int = 25
    stable_max_outside_delta_px: int = 80

    # --- hand -----------------------------------------------------------
    #: A camera sees a hand when the outside-board change is either
    #: (a) very large -- an arm across the frame, moving or not, or
    #: (b) large AND moving -- a hovering hand. A dart's static flight
    #:     over the rim is neither.
    hand_static_outside_px: int = 2500
    hand_min_outside_px: int = 400
    hand_min_outside_delta_px: int = 120
    #: consecutive hand frames before entering HAND.
    hand_enter_frames: int = 2
    #: consecutive non-hand frames before the hand is considered gone.
    hand_exit_frames: int = 4
    #: a board change first seen while/just after a hand was present
    #: must stay stable this long before it may commit.
    post_hand_stable_frames: int = 6
    #: a HAND that is completely still this long is not a hand (a
    #: permanent large outside change) -- leave HAND so a pending dart
    #: can still be judged.
    hand_static_max_frames: int = 45
    #: A board change that is still there when the hand leaves was CAUSED
    #: by the hand (a pull, a nudge, a placement) unless it had already
    #: been pending this many frames when the hand arrived. Nobody
    #: reaches the board within a second of their own throw; an arm
    #: reaching in shows up as a pending change for at most ~4 frames
    #: before the outside-board hand rule fires (shadow session max = 4).
    dart_before_hand_frames: int = 6

    # --- scene change (lighting) ---------------------------------------
    #: EVERY camera's board region changed by at least this fraction: a
    #: lighting step, not a dart (<= ~10%) and not a hand (one view).
    scene_board_frac: float = 0.35
    scene_stable_frames: int = 10

    # --- takeout --------------------------------------------------------
    #: union coverage at which a camera says "the darts are gone".
    takeout_coverage: float = 0.7
    #: consecutive stable frames of removal before clearing the visit.
    takeout_stable_frames: int = 5
    #: a majority of union cameras must read removal -- except within
    #: this many frames after a hand left, when ONE is enough (a hand on
    #: a populated board is a takeout; oblique cameras often read only
    #: 0.5-0.7 coverage for a single pulled dart).
    hand_takeout_frames: int = 30
    #: below this coverage the union is considered untouched.
    takeout_partial_coverage: float = 0.25
    #: a partial removal that stays this long is accepted into the reference.
    partial_adopt_frames: int = 45

    # --- the empty board ------------------------------------------------
    # 2026-09-26, a phantom S12 on the Pi: the takeout was cleared while
    # the hand was still pulling the last dart, so the new reference held
    # the hand and that dart; pulling it out then read as a dart arriving
    # where it had been, and was scored. Both guards below judge against
    # the EMPTY board -- the board as it was before the visit's first dart
    # -- which the dart-by-dart reference cannot remember.
    #: board pixels differing from the empty board (in one camera) below
    #: which that camera "looks empty" -- the dart floor, so anything that
    #: could be a dart counts as "not empty".
    empty_match_max_px: int = 80
    #: a would-be dart whose changed pixels mostly match the empty board
    #: again is something LEAVING the board, not arriving: never scored.
    #: The fraction of its changed pixels still differing from the empty
    #: board, below which a camera reads it as a removal.
    removal_match_frac: float = 0.3
    #: share of the committed darts' area (per camera, with at least
    #: ``empty_match_max_px`` pixels) that may still differ from the empty
    #: board when a takeout clears. Measured: the phantom's hand-and-dart
    #: frame left 21/25/50% of it off-empty; the takeouts of 1,360 corpus
    #: darts left at most 9%, all thin edge lines from the board or a
    #: camera shifting a pixel as the darts came out.
    empty_union_frac: float = 0.15
    #: a takeout clears only once, where the committed darts were, every
    #: camera matches the empty board again. None (the default since
    #: 2026-10-01): it waits for that however long it takes -- a dart left
    #: in keeps OD in takeout, not scoring, until it is pulled or Reset. A
    #: number clears anyway after that many stable frames of removal; the
    #: old 90 (~3 s) cleared relaxed pulls with a dart still in, and the
    #: board photo taken then showed it (rla: rather not score than that).
    clear_force_frames: int | None = None
    #: 2026-09-28, a stuck takeout on the Pi: after a clean pull the change
    #: covered only 0.65-0.69 of the darts' area (readopts during the visit
    #: had absorbed part of it), so the removal was taken as partial, the
    #: now-empty board was adopted as "the darts left in", and the visit
    #: could never clear -- there was nothing left to remove. While a visit
    #: waits for its takeout (full, or after a partial removal), a board
    #: that matches the empty board on every camera for this many stable
    #: frames (~1 s) clears it.
    stranded_clear_frames: int = 30

    # --- one camera alone is not a dart -----------------------------------
    # 2026-09-28, three OUTs on the Pi from a hand behind the board: cam1's
    # board area reaches the top of its frame, past the rim, so a hand up
    # there read as a dart in cam1 alone (82-163 px) with the other two
    # cameras at exactly 0. A dart is on the board, which every camera
    # watches: of 1,321 corpus darts, the second-strongest camera never read
    # below 20 px (p1 49, median 277). A would-be dart needs a second camera
    # with at least this many changed board pixels (when the rig has two).
    corroborate_min_px: int = 8
    #: a one-camera change that stays this many stable frames is absorbed
    #: into the reference -- long enough that a real dart's other views,
    #: a frame or two behind while it settles, are never cut short.
    single_cam_absorb_frames: int = 6
    # ... unless another camera saw THE SAME dart off the board or as a
    # 1 px shaft: at least same_dart_min_px of its changed pixels, and
    # same_dart_share of them, on the band the seeing camera's pixels map
    # to (opendarts/lifecycle/same_dart.py; needs the calibration). Real
    # darts: 1,241 of 1,242 one-camera trials at 0.5 or more (lowest
    # 0.46); ghosts never over 0.43. The pixel floor: across 1,783
    # one-camera trials of real darts the confirming camera never had
    # fewer than 144 px on the band (the missed dart this was built for:
    # 215); the strip switching off at 2026-10-04 10:13:06 confirmed
    # itself with 11 of 11 px at the old floor of 8.
    same_dart_min_px: int = 50
    same_dart_share: float = 0.5
    same_dart_len_mm: float = 150.0
    same_dart_lean_deg: float = 35.0
    same_dart_max_rays: int = 400
    same_dart_band_px: int = 2
    # --- two cameras near the floor, one dart ------------------------------
    # 2026-10-09, a third dart grouped on two S1 darts was never scored
    # (the oracle called S1). Its black flight landed on the earlier black
    # flights, so most of its change sat on them or off the board: replayed
    # from the frame dump, cameras 0/2 read 57/41, 75/60 and 57/55 board px
    # on three frames running. The one frame at 60 restarted the evidence
    # count, the next fell back under it, and the quiet branch adopted the
    # dart as background. A dart under the one-camera floor still counts
    # when the two cameras that see it most agree it is ONE dart:
    #   * each has at least pair_min_px board pixels;
    #   * each one's change lies on the other's dart band, both directions
    #     (same_dart.py, at same_dart_min_px / same_dart_share);
    #   * each shows a dart shape at full resolution (Apollo's detect_tip
    #     finds a component elongated past its MIN_ELONGATION_RATIO).
    # Neither test alone keeps the recorded phantoms out: the band passes a
    # light glinting on the cabinet bracket above the board (10-05 18:28,
    # a real object in the dart's space, seen by two cameras), the shape
    # gate passes two others (thin elongated edges in two cameras); both
    # together refuse all four. pair_min_px is a trigger, like dart_min_px:
    # in 2,721 quiet frames of five frame dumps no camera ever read a
    # single changed board pixel outside a dart, so the floor is not what
    # keeps noise out -- it is set low enough that this dart is caught with
    # margin (at 50 it commits on its last frame, by 5 px).
    pair_min_px: int = 40
    #: the pair must also look like a dart at full resolution in both
    #: cameras (Apollo's detect_tip finds an elongated component)
    pair_shape_check: bool = True

    # --- cooldowns / adoption ------------------------------------------
    commit_cooldown_frames: int = 6
    clear_cooldown_frames: int = 10
    #: FALLBACK refresh cadence. The reference normally refreshes on
    #: every quiet stable frame (quiet branch of ``_step``); this counter
    #: only covers frames where the BOARD is delta-quiet but something
    #: outside keeps ``stable`` false (a screen, spectators), so the
    #: reference cannot go stale indefinitely while board drift stacks
    #: underneath the churn.
    idle_readopt_frames: int = 20
    #: a stable board change that can't be resolved (e.g. 4th dart) is
    #: absorbed into the reference after this many frames.
    stuck_adopt_frames: int = 90
    warmup_stable_frames: int = 15

    max_darts: int = 3


DEFAULT_CONFIG = LifecycleConfig()


@dataclass
class Commit:
    """Frames handed to the scoring path on a dart commit."""

    dart_index: int  # 0-based within the visit
    bg: dict[int, np.ndarray]
    frames: dict[int, np.ndarray]
    dart_cams: list[int]
    forced: bool = False


@dataclass
class Tick:
    n: int
    phase: Phase
    action: Action
    dart_count: int
    signals: dict[int, CamSignals]
    stable: bool
    hand: bool
    dart_cams: list[int]
    reason: str = ""
    commit: Commit | None = None
    #: darts removed by a CLEARED action
    cleared: int = 0
    #: a would-be dart refused as a removal: the frames it would have been
    #: scored from (bg = the reference it was judged against), for evidence
    removed: Commit | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "n": self.n,
            "phase": self.phase.value,
            "action": self.action.value,
            "dart_count": self.dart_count,
            "stable": self.stable,
            "hand": self.hand,
            "dart_cams": self.dart_cams,
            "cams": {str(c): s.as_dict() for c, s in self.signals.items()},
        }
        if self.reason:
            d["reason"] = self.reason
        return d


@dataclass
class _Counters:
    warm: int = 0
    hand: int = 0
    hand_quiet: int = 0
    hand_static: int = 0
    #: frames since the last HAND exit (large when no hand has been seen)
    since_hand: int = 10_000
    #: the pending board change pre-dates the current hand (a landed dart)
    dart_before_hand: bool = False
    #: the current pending board change appeared under a hand
    hand_change: bool = False
    dart: int = 0
    pending: int = 0
    takeout: int = 0
    partial: int = 0
    quiet: int = 0
    scene: int = 0
    stuck: int = 0
    cooldown: int = 0
    over_darts: int = 0
    #: stable removal frames spent waiting for the board to look empty
    clear_wait: int = 0
    #: stable frames a would-be dart has been seen by one camera only
    single: int = 0
    #: stable quiet frames the board has matched the empty board while
    #: darts are still counted (``stranded_clear_frames``)
    stranded: int = 0
    #: why the pending change may not be scored, asked of the ``hold``
    #: provider on its first frame (None: free to score)
    held: str | None = None

    def reset_idle(self) -> None:
        self.dart = self.pending = self.takeout = self.partial = 0
        self.quiet = self.scene = self.stuck = self.over_darts = 0
        self.clear_wait = 0
        self.single = 0
        self.stranded = 0
        self.hand_change = False
        self.held = None


class Lifecycle:
    """Per-session throw lifecycle. Not thread-safe; drive it from the
    capture loop thread."""

    def __init__(
        self,
        board_masks: dict[int, np.ndarray],
        config: LifecycleConfig = DEFAULT_CONFIG,
        hold: Callable[[], "str | None"] | None = None,
        calibrations: dict[int, Any] | None = None,
    ) -> None:
        self.cfg = config
        #: per-camera calibrations the board masks were built from; lets a
        #: one-camera dart be confirmed by another camera seeing the same
        #: dart (same_dart.py). None/empty: the one-camera rule as before.
        self._calibrations = dict(calibrations) if calibrations else {}
        #: ``() -> reason | None``: a reason not to score right now, from
        #: outside the board (the oracle is on and not ready --
        #: opendarts/live/oracle_hold.py). Asked when a dart first shows up;
        #: a dart that arrived while held is never scored (``_commit``).
        #: None: every dart may score, exactly as before.
        self._hold = hold
        self._board_masks_full = dict(board_masks)
        self.refs = ReferenceSet()
        self.phase = Phase.WARMUP
        self.dart_count = 0
        self.tick_n = 0
        self._c = _Counters()
        self._last_change: dict[int, np.ndarray] = {}
        #: per camera, the small grey board as it was with no darts on it
        #: (see LifecycleConfig's "empty board" section). None until warm.
        self._empty: dict[int, np.ndarray] = {}
        #: a partial removal has been accepted this visit: darts are still
        #: counted, and the rest of the takeout is awaited
        self._partial_adopted = False
        #: set by a verified clear: retake the empty board when its cooldown ends
        self._retake_empty = False

    def reset(self) -> None:
        """Operator reset: forget darts and re-warm on the current scene."""
        self.refs.clear_darts()
        self.dart_count = 0
        self._partial_adopted = False
        self.phase = Phase.WARMUP
        self._c = _Counters()

    # ------------------------------------------------------------------
    def observe(self, frames: dict[int, np.ndarray]) -> Tick:
        cfg = self.cfg
        self.tick_n += 1
        # `frames` may be a LazyFrames (opendarts.capture.lazy_frame): each
        # camera's small grey picture then comes straight from its JPEG
        # (the hub's reduced decode) and the full frame is NOT decoded here
        # -- the reference and the commit hold the lazy handle, and only a
        # commit decodes it. A plain dict of arrays takes exactly the old
        # path.
        fulls = handles_of(frames)
        small_of = getattr(frames, "small_gray", None)
        smalls = {}
        for cam, f in fulls.items():
            small = small_of(cam, cfg.signals.scale) if small_of is not None else None
            smalls[cam] = small if small is not None else to_small_gray(
                pixels_of(f), cfg.signals.scale)

        for cam, small in smalls.items():
            if cam not in self.refs:
                mask_full = self._board_masks_full.get(cam)
                if mask_full is None:
                    continue
                self.refs.seed(cam, fulls[cam], small, to_small_mask(mask_full, small.shape))

        signals: dict[int, CamSignals] = {}
        for cam in self.refs.cameras():
            if cam not in smalls:
                continue
            ref = self.refs[cam]
            sig, changed = compute_signals(
                smalls[cam],
                ref.small,
                ref.prev_small,
                ref.board_mask,
                ref.dart_union,
                ref.dart_union_grown,
                cfg.signals,
            )
            signals[cam] = sig
            self._last_change[cam] = changed
            ref.prev_small = smalls[cam]

        if not signals:
            return Tick(self.tick_n, self.phase, Action.NONE, self.dart_count, {}, False, False, [], "no cameras")

        agg = self._aggregate(signals)
        if not agg.dart_cams:
            agg.dart_cams = self._pair_dart_cams(signals, agg, fulls, smalls)
        tick = self._step(agg, signals, fulls, smalls)
        return tick

    # ------------------------------------------------------------------
    @dataclass
    class _Agg:
        stable: bool
        hand_now: bool
        scene: bool
        dart_cams: list[int]
        union_cams: list[int]
        removal_cams: list[int]
        partial_cams: list[int]
        over_darts: bool

    def _aggregate(self, signals: dict[int, CamSignals]) -> "Lifecycle._Agg":
        cfg = self.cfg
        stable = all(
            s.board_delta_px <= cfg.stable_max_board_delta_px
            and s.outside_delta_px <= cfg.stable_max_outside_delta_px
            for s in signals.values()
        )
        hand_now = any(
            s.outside_px >= cfg.hand_static_outside_px
            or (
                s.outside_px >= cfg.hand_min_outside_px
                and s.outside_delta_px >= cfg.hand_min_outside_delta_px
            )
            for s in signals.values()
        )
        scene = all(s.board_frac >= cfg.scene_board_frac for s in signals.values())
        union_cams = [c for c, s in signals.items() if s.union_px > 0]
        # A camera reads as "removal" when most of the committed union has
        # changed AND the change is dominated by those union pixels (the
        # small spill outside the union is settle-wobble and shadow, not a
        # new object). Such a camera is never a dart camera: a new dart on
        # a populated board covers <= ~45% of the union (corpus p95 = 11%).
        removal_cams = [
            c
            for c in union_cams
            if signals[c].union_coverage >= cfg.takeout_coverage
            and signals[c].board_new_px < signals[c].in_union_px
        ]
        # A dart camera is judged on the FULL board change vs the
        # reference -- the committed-dart union is NOT subtracted. A dart
        # landing against an already-committed neighbour puts most of its
        # pixels inside the neighbour's grown union; subtracting the
        # union deleted that evidence (and fragmented what was left below
        # any blob floor), which is how a real third dart went unscored.
        # The guard against the opposite failure -- a departing dart's
        # own pixels reading as "something new" mid-removal -- is the
        # same union-dominance test removal_cams uses, at ANY coverage:
        # when the change is mostly the committed darts' own pixels
        # (board_new_px < in_union_px) it is the darts leaving, not a new
        # object. A partial removal sits at 0.25-0.7 coverage, below the
        # removal_cams floor, and must not commit a phantom; this is why
        # ``board_new_px`` / ``in_union_px`` stay computed unchanged.
        # ... but only once darts could really be leaving: at least
        # takeout_partial_coverage of the committed darts changed. A new
        # dart landing IN FRONT of a group puts most of its change on the
        # darts already there while changing little of them (0.15 and 0.01
        # for the two real darts this guard dropped on the low-camera rig, 10-04
        # 22:14:46 and 10-02 10:32:21); a pull changes far more. Below the
        # max dart count only -- at the max nothing can commit anyway.
        # See docs/LIFECYCLE.md, "The union-dominance guard".
        dart_cams = [
            c
            for c, s in signals.items()
            if c not in removal_cams
            and s.board_px >= cfg.dart_min_px
            and (s.board_new_px >= s.in_union_px
                 or (s.union_coverage < cfg.takeout_partial_coverage
                     and self.dart_count < cfg.max_darts))
        ]
        partial_cams = [c for c in union_cams if signals[c].union_coverage >= cfg.takeout_partial_coverage]
        over_darts = any(signals[c].union_coverage >= cfg.over_darts_coverage for c in union_cams)
        return Lifecycle._Agg(
            stable, hand_now, scene, dart_cams, union_cams, removal_cams, partial_cams, over_darts
        )

    # ------------------------------------------------------------------
    def _adopt(self, frames: dict[int, np.ndarray], smalls: dict[int, np.ndarray]) -> None:
        self.refs.adopt_all(frames, smalls)
        # whatever was pending is background (or committed) now
        self._c.dart_before_hand = False

    # ---- the empty board -------------------------------------------------
    def _set_empty(self, smalls: dict[int, np.ndarray]) -> None:
        self._empty = {cam: smalls[cam] for cam in self.refs.cameras() if cam in smalls}

    def _off_empty(self, cam: int, small: np.ndarray) -> np.ndarray | None:
        """Board pixels of ``small`` that differ from the empty board."""
        empty = self._empty.get(cam)
        if empty is None:
            return None
        return change_mask(small, empty, self.cfg.signals) & self.refs[cam].board_mask

    def _looks_empty(self, smalls: dict[int, np.ndarray]) -> bool:
        """Every camera's board matches the empty board (True while none is
        known yet, so a cold start behaves exactly as before)."""
        for cam, small in smalls.items():
            off = self._off_empty(cam, small) if cam in self.refs else None
            if off is not None and int(off.sum()) >= self.cfg.empty_match_max_px:
                return False
        return True

    def _darts_gone(self, smalls: dict[int, np.ndarray]) -> bool:
        """Where the committed darts were, every camera matches the empty
        board again. Judged only there, so a dart nudged elsewhere, a mark
        on the board or slow lighting drift cannot hold a takeout open."""
        for cam, small in smalls.items():
            if cam not in self.refs:
                continue
            where = self.refs[cam].dart_union_grown
            off = self._off_empty(cam, small)
            if where is None or off is None:
                continue
            n = int((off & where).sum())
            if n >= self.cfg.empty_match_max_px and n >= self.cfg.empty_union_frac * int(where.sum()):
                return False
        return True

    def _refresh_empty(self, smalls: dict[int, np.ndarray]) -> None:
        """Follow slow drift on an empty board -- but only from a frame that
        still matches it, so a hand or a dart is never taken as empty."""
        if self.dart_count == 0 and self._empty and self._looks_empty(smalls):
            self._set_empty(smalls)

    def _corroborated(self, signals, smalls: dict[int, np.ndarray]) -> str | None:
        """Why a second camera counts as seeing the change -- or None.
        The rig may have only one camera; otherwise a second camera sees it
        on the board, or sees the same dart (see same_dart.py)."""
        cfg = self.cfg
        need = min(2, len(signals))
        seen = [cam for cam, s in signals.items() if s.board_px >= cfg.corroborate_min_px]
        if len(seen) >= need:
            return "on the board"
        if len(seen) != 1 or not self._calibrations:
            return None
        cam = seen[0]
        others = {c: (smalls[c], self.refs[c].small) for c in signals if c != cam and c in smalls}
        found = same_dart.support(
            self._calibrations, cam, self._last_change[cam], others,
            cfg.signals, dart_len_mm=cfg.same_dart_len_mm, lean_deg=cfg.same_dart_lean_deg,
            max_rays=cfg.same_dart_max_rays, band_px=cfg.same_dart_band_px,
        )
        for other, (on, total) in sorted(found.items()):
            if on >= cfg.same_dart_min_px and on >= cfg.same_dart_share * total:
                return f"same dart seen by camera {other} ({on}/{total} px)"
        return None

    def _pair_dart_cams(self, signals, agg, frames, smalls: dict[int, np.ndarray]) -> list[int]:
        """The two strongest cameras, when each sees ``pair_min_px`` board
        pixels, each one's change lies on the other's dart band (both
        directions of same_dart.py), and each shows a dart shape at full
        resolution: one dart, seen from two sides, under the one-camera
        floor in both. Else [] (see ``pair_min_px``)."""
        cfg = self.cfg
        if self.dart_count >= cfg.max_darts or not self._calibrations:
            return []
        # the same union-dominance guard as ``dart_cams`` (see _aggregate)
        near = sorted(
            (c for c, s in signals.items()
             if c not in agg.removal_cams and c in smalls and c in frames
             and c in self._calibrations and s.board_px >= cfg.pair_min_px
             and (s.board_new_px >= s.in_union_px
                  or s.union_coverage < cfg.takeout_partial_coverage)),
            key=lambda c: (-signals[c].board_px, c),
        )
        if len(near) < 2:
            return []
        a, b = near[0], near[1]
        for seen, other in ((a, b), (b, a)):
            found = same_dart.support(
                self._calibrations, seen, self._last_change[seen],
                {other: (smalls[other], self.refs[other].small)},
                cfg.signals, dart_len_mm=cfg.same_dart_len_mm, lean_deg=cfg.same_dart_lean_deg,
                max_rays=cfg.same_dart_max_rays, band_px=cfg.same_dart_band_px,
            )
            on, total = found.get(other, (0, 0))
            if on < cfg.same_dart_min_px or on < cfg.same_dart_share * total:
                return []
        if cfg.pair_shape_check:
            # Only here, a few frames per visit at most, are the full frames
            # decoded: Apollo's own dart-shape gate on each camera's change.
            from opendarts.engines.apollo.tip_detection import MIN_ELONGATION_RATIO, detect_tip

            for cam in (a, b):
                found = detect_tip(pixels_of(self.refs[cam].full), pixels_of(frames[cam]))
                elongation = (found.diagnostics or {}).get("elongation_ratio") or 0.0
                if not found.ok or elongation < MIN_ELONGATION_RATIO:
                    return []
        return [a, b]

    def _refuse(self, agg, signals, frames, smalls, reason: str) -> Tick:
        """Absorb a would-be dart into the reference without scoring it,
        keeping what it would have been scored from, so a refusal can be
        checked."""
        bg = {cam: ref.full for cam, ref in self.refs.cams.items() if cam in frames}
        refused = Commit(
            dart_index=self.dart_count,
            bg=as_frames(bg),
            frames=as_frames({cam: frames[cam] for cam in bg}),
            dart_cams=list(agg.dart_cams),
        )
        self._adopt(frames, smalls)
        self.phase = Phase.IDLE
        self._c.reset_idle()
        tick = self._tick(agg, signals, Action.READOPT, reason)
        tick.removed = refused
        return tick

    def _is_removal(self, dart_cams: list[int], smalls: dict[int, np.ndarray]) -> bool:
        """The change a would-be dart made has mostly returned the board to
        empty, in most of its cameras: something left, nothing arrived."""
        votes = 0
        judged = 0
        for cam in dart_cams:
            changed = self._last_change.get(cam)
            off = self._off_empty(cam, smalls[cam]) if cam in smalls else None
            if changed is None or off is None:
                continue
            changed = changed & self.refs[cam].board_mask
            n = int(changed.sum())
            if n == 0:
                continue
            judged += 1
            if int((changed & off).sum()) < self.cfg.removal_match_frac * n:
                votes += 1
        return judged > 0 and votes * 2 > judged

    def _tick(self, agg: "Lifecycle._Agg", signals, action: Action, reason: str = "", commit: Commit | None = None) -> Tick:
        return Tick(
            n=self.tick_n,
            phase=self.phase,
            action=action,
            dart_count=self.dart_count,
            signals=signals,
            stable=agg.stable,
            hand=self.phase is Phase.HAND,
            dart_cams=agg.dart_cams,
            reason=reason,
            commit=commit,
        )

    def _step(self, agg, signals, frames, smalls) -> Tick:
        cfg, c = self.cfg, self._c

        # ---- warmup: adopt every tick until the scene has been still a while
        if self.phase is Phase.WARMUP:
            self._adopt(frames, smalls)
            c.warm = c.warm + 1 if agg.stable else 0
            if c.warm >= cfg.warmup_stable_frames:
                self.phase = Phase.IDLE
                c.reset_idle()
                if self.dart_count == 0:
                    self._set_empty(smalls)
                return self._tick(agg, signals, Action.READY, "warmup complete")
            return self._tick(agg, signals, Action.NONE)

        # ---- cooldown: absorb wobble / arm retreat into the reference
        if self.phase is Phase.COOLDOWN:
            self._adopt(frames, smalls)
            c.cooldown -= 1
            if c.cooldown <= 0:
                self.phase = Phase.IDLE
                c.reset_idle()
                # The empty board is "the board before the visit's first
                # dart" (see the config notes above): taken at warmup, and
                # again at the end of every verified clear -- which is
                # exactly that moment. Taking it only once let a picture
                # made with darts in the board (a restart or Reset mid-visit)
                # stand for good, and a later dart landing where one of
                # those had been was refused as "a removal" (2026-10-05).
                if self.dart_count == 0 and (not self._empty or self._retake_empty):
                    self._set_empty(smalls)
                self._retake_empty = False
            return self._tick(agg, signals, Action.NONE)

        # ---- scene change (lighting step): every camera's board changed.
        # Evaluated before the hand rules because a whole-frame step also
        # satisfies the static-arm rule; never commit, re-adopt when still.
        if self.phase is Phase.SCENE_CHANGE:
            # "Passed" only when the board is back to the reference. Lights
            # that fade in, then auto-exposure pulling most of the board
            # back, shrink the change below scene_board_frac while it is
            # still moving -- and what stays different must be adopted
            # below once still, not judged as a dart against the old
            # reference (a phantom at every Start that lit the cabinet).
            if not agg.scene and not agg.dart_cams:
                self.phase = Phase.IDLE
                c.reset_idle()
                return self._tick(agg, signals, Action.NONE, "scene change passed")
            c.scene = c.scene + 1 if agg.stable else 0
            if c.scene >= cfg.scene_stable_frames:
                self._adopt(frames, smalls)
                self.phase = Phase.IDLE
                c.reset_idle()
                # with darts on the board the new scene is not an empty
                # one: forget it, and take the next empty board after a clear
                if self.dart_count == 0:
                    self._set_empty(smalls)
                else:
                    self._empty = {}
                return self._tick(agg, signals, Action.READOPT, "scene change adopted")
            return self._tick(agg, signals, Action.NONE)

        if agg.scene and self.phase is not Phase.HAND:
            self.phase = Phase.SCENE_CHANGE
            c.reset_idle()
            return self._tick(agg, signals, Action.SCENE_CHANGE)

        # ---- hand tracking (priority over everything below)
        if agg.hand_now:
            c.hand += 1
            c.hand_quiet = 0
        else:
            c.hand = 0
            c.hand_quiet += 1

        if self.phase is Phase.HAND:
            c.hand_static = c.hand_static + 1 if agg.stable else 0
            if c.hand_quiet >= cfg.hand_exit_frames:
                self.phase = Phase.IDLE
                c.reset_idle()
                c.since_hand = 0
                # Whatever differs on the board as the hand leaves was
                # done BY the hand -- unless it was a landed dart the hand
                # merely came near (see dart_before_hand_frames). That
                # flag outlives a pause-and-resume of the same hand and is
                # dropped when the change resolves or is adopted.
                c.hand_change = bool(agg.dart_cams) and not c.dart_before_hand
                return self._tick(agg, signals, Action.HAND_EXIT)
            if c.hand_static >= cfg.hand_static_max_frames:
                # Not a hand: something outside the board changed for
                # good. Absorb it into the OUTSIDE part of the detection
                # reference only, so the board is still judged against
                # the pre-change reference (and the engines' bg is
                # untouched), then let the board be evaluated.
                self.refs.adopt_outside(smalls)
                self.phase = Phase.IDLE
                c.reset_idle()
                c.since_hand = 0
                c.hand = 0
                c.dart_before_hand = False
                return self._tick(agg, signals, Action.HAND_EXIT, "static outside change absorbed")
            return self._tick(agg, signals, Action.NONE)

        if c.hand >= cfg.hand_enter_frames:
            self.phase = Phase.HAND
            # a change that has been pending well before the hand showed
            # up, and did not itself start under the previous hand, is a
            # landed dart: it may still commit once the hand is gone.
            c.dart_before_hand = c.dart_before_hand or (
                c.pending >= cfg.dart_before_hand_frames and c.since_hand > c.pending
            )
            c.reset_idle()
            c.hand_static = 0
            return self._tick(agg, signals, Action.HAND_ENTER)

        c.since_hand += 1

        # ---- something new on the board
        if agg.dart_cams:
            c.takeout = c.partial = c.quiet = c.stranded = 0
            c.pending += 1
            if c.pending == 1 and self._hold is not None:
                # judged as the dart lands, not when it commits: the oracle
                # can change its mind in between -- often because of this
                # very dart (a third throw puts Autodarts into takeout)
                c.held = self._hold()
            if c.hand_change:
                # The hand did this. With darts committed it is a takeout
                # we could not read from the union (pulled + nudged, or a
                # dart that was in the reference all along); with none it
                # is background now. Never a throw.
                self.phase = Phase.PENDING_DART
                # evidence accumulates through instability, same as the
                # main dart branch below; acting on it still waits for a
                # currently-stable frame
                c.dart += 1
                if (
                    c.dart >= cfg.post_hand_stable_frames and agg.stable
                ) or c.pending >= cfg.dart_force_frames:
                    if self.dart_count > 0:
                        return self._clear(agg, signals, frames, smalls, CLEARED_UNDER_HAND)
                    self._adopt(frames, smalls)
                    self.phase = Phase.IDLE
                    c.reset_idle()
                    return self._tick(agg, signals, Action.FORCED_ADOPT, "board changed under hand absorbed")
                return self._tick(agg, signals, Action.NONE, "hand-caused change")
            if self.dart_count >= cfg.max_darts:
                self.phase = Phase.PENDING_DART
                c.stuck = c.stuck + 1 if agg.stable else 0
                if c.stuck >= cfg.stuck_adopt_frames:
                    self._adopt(frames, smalls)
                    c.reset_idle()
                    self.phase = Phase.IDLE
                    return self._tick(agg, signals, Action.FORCED_ADOPT, "board change with max darts absorbed")
                return self._tick(agg, signals, Action.NONE, "max darts")
            # Dart evidence accumulates on EVERY frame the dart test
            # passes, stable or not -- a thrown dart is visible while it
            # is still settling, and resetting on each unstable frame
            # discarded the entire arrival. The commit itself still waits
            # for a currently-stable frame below.
            c.dart += 1
            self.phase = Phase.PENDING_DART
            forced = c.pending >= cfg.dart_force_frames
            # a change that began while a hand was leaving needs longer
            need = cfg.dart_stable_frames
            if c.since_hand - c.pending < cfg.post_hand_stable_frames:
                need = max(need, cfg.post_hand_stable_frames)
            if (c.dart >= need and agg.stable) or forced:
                if self._is_removal(agg.dart_cams, smalls):
                    # Back to the empty board where it changed: something
                    # was taken away (a dart pulled after an early clear),
                    # not thrown.
                    return self._refuse(agg, signals, frames, smalls, "a removal, not a dart")
                why = self._corroborated(signals, smalls)
                if why is None:
                    # One camera alone -- a hand at the edge of one view,
                    # not a dart on the board every camera watches. Hold a
                    # few stable frames (a real dart's other views can lag
                    # while it settles), then absorb it.
                    c.single = c.single + 1 if agg.stable else c.single
                    if c.single >= cfg.single_cam_absorb_frames or forced:
                        return self._refuse(agg, signals, frames, smalls, "seen by one camera only")
                    return self._tick(agg, signals, Action.NONE, "held: seen by one camera only")
                if agg.over_darts and not forced:
                    if agg.stable and not agg.hand_now:
                        c.over_darts += 1
                    else:
                        c.over_darts = 0
                    if c.over_darts >= cfg.over_darts_hold_frames:
                        return self._commit(
                            agg, signals, frames, smalls, False, "over darts, stable"
                        )
                    return self._tick(agg, signals, Action.NONE, "held: change over committed darts")
                return self._commit(agg, signals, frames, smalls, forced,
                                    why if why.startswith("same dart") else "")
            c.over_darts = 0
            return self._tick(agg, signals, Action.NONE)

        c.dart = c.pending = c.stuck = c.over_darts = 0
        c.hand_change = c.dart_before_hand = False
        c.held = None

        # ---- committed darts being removed
        # A takeout is entered on a STILL frame, never a moving one. A dart in
        # flight passing in front of a committed one reads, for that single
        # blurred frame, exactly like that dart being pulled (low-camera
        # rig 10-05 14:55:07.958: 29% of dart 1 "changed", the next frame was
        # dart 2), and entering TAKEOUT_PENDING announces a takeout to every
        # listener -- the cabinet strip flashed yellow after dart 2. Every
        # count below already ignores moving frames, so this only delays the
        # announcement (replayed: median 2 frames, 28 such flashes removed
        # across both rigs' logs); when a visit clears is unchanged.
        if (agg.union_cams and (agg.removal_cams or agg.partial_cams)
                and not agg.stable and self.phase is not Phase.TAKEOUT_PENDING):
            c.takeout = c.partial = c.quiet = c.stranded = 0
            return self._tick(agg, signals, Action.NONE)

        if agg.union_cams and agg.removal_cams:
            self.phase = Phase.TAKEOUT_PENDING
            c.partial = c.quiet = c.stranded = 0
            majority = (
                len(agg.removal_cams) * 2 >= len(agg.union_cams)
                or c.since_hand <= cfg.hand_takeout_frames
            )
            # A camera hovering at the coverage threshold must not keep
            # restarting the count (shadow session: 0.697/0.712 flicker
            # turned a 0.2 s clear into 4 s). Non-majority frames pause.
            if not agg.stable:
                c.takeout = 0
            elif majority:
                c.takeout += 1
            if c.takeout >= cfg.takeout_stable_frames:
                if not self._darts_gone(smalls) and (
                    cfg.clear_force_frames is None or c.clear_wait < cfg.clear_force_frames
                ):
                    # The union reads "removed", but where the darts were the
                    # board is not empty yet -- usually the last dart, with
                    # the hand pulling it. Clearing now would make that the
                    # new reference, and its removal would read as a throw.
                    c.clear_wait += 1
                    return self._tick(agg, signals, Action.NONE, "waiting for an empty board")
                return self._clear(agg, signals, frames, smalls)
            return self._tick(agg, signals, Action.NONE)

        c.takeout = 0

        if agg.union_cams and agg.partial_cams:
            self.phase = Phase.TAKEOUT_PENDING
            c.quiet = c.stranded = 0
            c.partial = c.partial + 1 if agg.stable else 0
            if c.partial >= cfg.partial_adopt_frames:
                if self._empty and self._looks_empty(smalls):
                    # Only part of the darts' area reads "removed" against
                    # the reference, but every camera matches the empty
                    # board: the darts are all out. Adopting this as "the
                    # darts left in" would strand the visit. The whole-board
                    # test, not ``_darts_gone``: one dart left in can be
                    # under ``empty_union_frac`` of a grown multi-dart union,
                    # but never under ``empty_match_max_px`` (the dart floor)
                    # in every camera.
                    return self._clear(agg, signals, frames, smalls, CLEARED_NOT_PARTIAL)
                for cam in agg.partial_cams:
                    changed = self._last_change.get(cam)
                    if changed is not None:
                        self.refs[cam].remove_from_union(changed)
                self._adopt(frames, smalls)
                self._partial_adopted = True
                self.phase = Phase.IDLE
                c.reset_idle()
                return self._tick(agg, signals, Action.PARTIAL_ADOPT, "partial removal accepted")
            return self._tick(agg, signals, Action.NONE)

        c.partial = 0

        # ---- quiet: keep the reference fresh. Nothing new on the board,
        # nothing being removed: whatever differs from the reference now
        # (drift, a sub-threshold artifact, an object placed in view) is
        # background, absorbed on EVERY stable frame rather than on a
        # 20-frame cadence. This continuous refresh replaces the removed
        # blob-size floor as the drift defence: drift slower than
        # ``diff_threshold`` gray levels per frame can never accumulate
        # into a phantom, because the baseline is at most one frame old
        # (an N-frame cadence let any ramp >= diff_threshold/N gray per
        # frame cross the dart floor between readopts -- including
        # contiguous single-camera fades the old blob floor never caught).
        # Costs, stated honestly:
        #   * a real dart flickering around ``dart_min_px`` is adopted as
        #     background on the first stable frame it dips below the
        #     floor; there is no longer a ~20-frame grace window in which
        #     it could re-cross. Such a dart sits below every corpus
        #     recording, so its population size is unknown (survivorship
        #     -- see ``dart_min_px``).
        #   * frame-to-frame drift was never measured (packages hold no
        #     consecutive frames); this branch rests on the pipeline's
        #     own arithmetic, not on corpus data.
        # READOPT is reported only when something nonzero was absorbed,
        # so the idle tick log does not flood.
        self.phase = Phase.IDLE
        if (
            agg.stable
            and not agg.hand_now
            and self.dart_count > 0
            and (self.dart_count >= cfg.max_darts or self._partial_adopted)
            and self._empty
            and self._looks_empty(smalls)
        ):
            # Darts are counted and the takeout is awaited, but every camera
            # sees the empty board: nothing is left to remove, so no removal
            # can ever be read (see ``stranded_clear_frames``). A dart still
            # in the board keeps its camera well past ``empty_match_max_px``.
            c.stranded += 1
            if c.stranded >= cfg.stranded_clear_frames:
                return self._clear(agg, signals, frames, smalls, CLEARED_STRANDED)
        else:
            c.stranded = 0
        if agg.stable:
            absorbed = any(s.board_px or s.outside_px for s in signals.values())
            self._adopt(frames, smalls)
            self._refresh_empty(smalls)
            c.quiet = 0
            if absorbed:
                return self._tick(agg, signals, Action.READOPT)
            return self._tick(agg, signals, Action.NONE)
        # Fallback: the board itself is delta-quiet but something outside
        # keeps ``stable`` false. Refresh on the old cadence so the
        # reference cannot go stale indefinitely under outside churn.
        if all(
            s.board_delta_px <= cfg.stable_max_board_delta_px
            for s in signals.values()
        ):
            c.quiet += 1
            if c.quiet >= cfg.idle_readopt_frames:
                self._adopt(frames, smalls)
                c.quiet = 0
                return self._tick(
                    agg, signals, Action.READOPT, "board quiet under outside churn"
                )
        else:
            c.quiet = 0
        return self._tick(agg, signals, Action.NONE)

    # ------------------------------------------------------------------
    def _commit(self, agg, signals, frames, smalls, forced: bool, reason: str = "") -> Tick:
        cfg, c = self.cfg, self._c
        if c.held is not None:
            # A dart that landed while held is not scored. Absorbed like any
            # refusal: the reference takes it, the visit's count and the
            # empty board do not, so pulling it later reads as a removal
            # (``_is_removal``) and scores nothing either.
            return self._refuse(agg, signals, frames, smalls, f"not scored: {c.held}")
        bg = {cam: ref.full for cam, ref in self.refs.cams.items() if cam in frames}
        # Handles, not pixels: lazy frames are decoded by whoever reads
        # them (the adapter decodes both sets in parallel). Plain arrays
        # pass through as the same dicts as ever.
        commit = Commit(
            dart_index=self.dart_count,
            bg=as_frames(bg),
            frames=as_frames({cam: frames[cam] for cam in bg}),
            dart_cams=list(agg.dart_cams),
            forced=forced,
        )
        for cam in self.refs.cameras():
            changed = self._last_change.get(cam)
            if changed is not None:
                self.refs[cam].add_dart_mask(changed)
        self.dart_count += 1
        self._adopt(frames, smalls)
        self.phase = Phase.COOLDOWN
        c.reset_idle()
        c.cooldown = cfg.commit_cooldown_frames
        if not reason:
            reason = "forced" if forced else ""
        return self._tick(agg, signals, Action.COMMIT, reason, commit)

    def _clear(self, agg, signals, frames, smalls, reason: str = "") -> Tick:
        cfg, c = self.cfg, self._c
        cleared = self.dart_count
        # Retake the empty board after the cooldown only when the board
        # vouched for it: it matched the empty board (stranded / not
        # partial), or the darts were gone where they had been. Not after a
        # clear forced by clear_force_frames (something may still be over a
        # dart) nor one made because the board changed under a hand.
        self._retake_empty = (reason in (CLEARED_NOT_PARTIAL, CLEARED_STRANDED)
                              or (not reason and self._darts_gone(smalls)))
        self.refs.clear_darts()
        self.dart_count = 0
        self._partial_adopted = False
        self._adopt(frames, smalls)
        self.phase = Phase.COOLDOWN
        c.reset_idle()
        c.cooldown = cfg.clear_cooldown_frames
        detail = f" ({reason})" if reason else ""
        tick = self._tick(agg, signals, Action.CLEARED, f"{cleared} darts removed{detail}")
        tick.cleared = cleared
        return tick
