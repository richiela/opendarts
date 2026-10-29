# Lifecycle

Dart throws and takeouts are decided by `opendarts.lifecycle`, not by the
capture loop. `opendarts/live/capture_daemon.py`'s `_lifecycle_step()` is the
loop's single decision seam — everything about *when* a throw has happened
lives behind it.

| Module | Role |
|---|---|
| `signals.py` | per-camera, per-frame motion/settle signals |
| `state.py` | the state machine: commit a dart, clear the visit, hand entered/left |
| `driver.py` | runs the state machine inside the capture loop and records its decisions |
| `adapter.py` | translates those decisions into the trigger state the capture loop consumes |
| `settings.py` | operator-tunable thresholds, persisted across restarts |
| `reference.py` | per-camera reference (empty-board) frames and committed-dart masks |

Keeping the decision in one place means throw detection can be tested without
cameras, on recorded frames. The frames are already kept: a recorded throw
package holds the run from the empty board to the scored frame for every dart
that was detected, and a whole-ring dump (the missed-dart button) holds the
frames around one that was not — see `docs/PACKAGES.md`. The shipped replay
tool re-scores packages; it does not re-run detection.

## How detection decides, and the rules it has grown

Detection is deliberately cheap: every frame, each camera's picture is
shrunk to a quarter and compared with the last empty-board reference, and
the state machine reasons only about **how many pixels changed, in how many
cameras, and whether they are still**. It never looks at shape — that
belongs to the engines, which run only after a dart is committed and must
always answer (one exception: "Two cameras, one dart" below borrows
Apollo's shape gate, only for a dart under the floor). So every rule below is a way of telling two situations apart
that look alike in pixel counts. Each one was added because a real throw
went wrong.

That is the risk this section exists to manage: each fix splits one rule
into two, and the rules interact. Before adding or changing one, read this
list, and add an entry for anything new.

| Rule | Tells apart | Threshold | Evidence | Known limits |
|---|---|---|---|---|
| Dart size floor (`dart_min_px`) | something that may be a dart from drift; a **trigger**, not the ghost guard | 60 px in one camera (was 80) | at 80, two grouped third darts on a low-camera rig were never scored (78/61/26 and 9/63/72 px; the oracle scored both). Replayed over every logged session on both rigs: at 60 both are caught, and every other change it admits is one-camera, refused by the next rule — no new commits | below ~60 it mostly adds one-camera noise holds |
| Second camera (`corroborate_min_px`) | a dart, which every camera watches, from a hand or light at the edge of one view | 8 board px in a second camera | three hand-behind-the-board ghosts, others at 0 | a dart can reach the others only off the board — see same-dart |
| Same dart (`same_dart.py`, `same_dart_min_px`, `same_dart_share`) | another camera seeing *this* dart off the board or as a 1 px shaft, from a change elsewhere | ≥ 50 px and ≥ half of that camera's change on the dart's band | 1,783 one-camera trials of real darts: never under 144 px; 16 saved ghosts refused; the strip switching off scored 11 px | needs the calibration; only consulted when one camera alone sees a dart |
| Two cameras, one dart (`pair_min_px`, `pair_shape_check`) | a dart grouped on earlier darts, under the one-camera floor in every camera, from a change two cameras merely share | the two strongest cameras each ≥ 40 board px, each one's change on the other's dart band (both directions, same-dart thresholds), and each a dart shape at full resolution (Apollo's elongation gate) | 2026-10-09: a third dart on two S1 darts read 57/41, 75/60, 57/55 px on cameras 0/2 and was adopted as background; caught at set 546 of its frame dump, nothing else in five dumps changes. Recorded phantoms: the band alone passes 1 of 4 (a light glinting on the cabinet bracket), the shape alone 2 of 4, both together 0 | phantom risk measured only on 100 s of continuous video and 4 recorded phantoms; the rig logs were not replayed. Decodes full frames, only when it runs |
| Lighting step (scene change) | the whole board brightening or darkening from a dart | ≥ 35% of every camera's board | — | a fade's first frames can be patchy enough to pass as a two-camera dart before this fires (parked, see below) |
| Takeout entry | a dart in flight passing in front of a committed one (one blurred frame) from darts being pulled | enters takeout only on a still frame | replayed: 28 post-dart takeout announcements (cabinet strip flashes) removed; real takeouts announced a median 2 frames later; when a visit clears unchanged | — |
| Empty board picture | "the board before the visit's first dart", which the removal and takeout checks judge against | taken at warmup and again after every clear the board vouched for (it matched the empty board, or the darts were gone where they had been) | a restart mid-visit baked darts into it; a later dart on that spot was refused as a removal | not retaken after a forced clear (`clear_force_frames`) or one made under a hand |
| Union-dominance guard | darts **leaving** (most of the change sits on the committed darts) from a dart **landing in front of** them | applies only once ≥ 25% (`takeout_partial_coverage`) of the committed darts has changed, and below the dart limit | replayed over 68 logged sessions on two rigs: catches 2 real darts the old guard dropped (they changed 15% and 1% of the group), 0 new phantoms | thresholds are shared with partial takeout |

### Changes, with the throw that caused each

- **Two cameras, one dart** — a third dart grouped on two S1 darts showed
  under 60 px in every camera on its three settled frames, so its evidence
  count restarted and it was adopted as background (the oracle scored S1).
  Two cameras near the floor now count together when the change is one dart
  in 3D and looks like a dart in both. The band check alone would also pass
  a glint on the cabinet hardware, which is why the shape check is there.

- **Lighting step shrinks before it settles** — lights fading in, then auto
  exposure pulling the board back, made the scene rule exit without
  re-adopting, and the leftover was committed: a phantom OUT at every Start
  that lit the cabinet. It now leaves the scene state only when the board
  matches the reference again.
- **Same dart** — a dart grouped with two others on a low-camera rig was
  seen whole by one camera and only as a flight and a 1 px shaft by the
  others, and was refused as one-camera. Added the band check; its pixel
  floor went from 8 to 50 after the strip switching off confirmed itself
  with 11 px.
- **Union-dominance guard narrowed** — a third dart landing in front of two
  S20s was read as the darts leaving and never scored (Autodarts scored it
  S20). The guard now waits for a pull-sized share of the group to change.

- **Takeout entered on a still frame** — dart 2 in flight passed in front of
  dart 1 for one frame, OD announced a takeout, and the cabinet strip
  flashed yellow until dart 2 scored. Not a new rule: every takeout count
  already ignored moving frames; now the announcement does too.
- **Empty board retaken after each clean takeout** — not a new rule either:
  the existing definition (the board before the visit's first dart) applied
  at every visit instead of only at warmup.

- **Dart size floor 80 → 60** — measured, not guessed: the floor had been
  doing two jobs, and the one-camera rule already does the ghost-guarding.
  The floor is now a trigger set where the evidence says, the same split
  between an easy trigger and a strict later check that the replay plan
  below aims for everywhere.

### Open, deliberately parked

- **Phantom OUTs.** A commit always reaches the engines, which must answer;
  with no dart in the frames they put a "tip" off the board and every
  engine says outside. Candidate gates measured so far (blob shape, cameras
  used, engine tip agreement, same-object share on every commit) each fail
  on real OUTs near the edge or on one of the collected phantoms. Phantoms
  are being collected; the honest answer is a dart recogniser, not another
  threshold.
  With the oracle on, both directions now collect themselves: a lasting
  disagreement in dart count keeps the frame ring (`docs/LIVE_API.md`,
  "Oracle count mismatch") -- the raw material for the replay set below.
- **Patchy light fades** committing before the lighting rule fires.

### The way out of rule-stacking

1. **An end-to-end replay set.** Packages hold two frames and re-score; a
   whole-ring dump holds 20 s of every camera. Keep dumps of real sequences
   — grouped darts, pulls, hands, lights, phantoms — and replay every
   detection change through the whole lifecycle against all of them, so
   "did this break anything" is measured, not argued.
2. **Then consolidate rather than add.** The idea that keeps winning is
   *a dart is one consistent object across cameras* (the same-dart band).
   With the replay set in place, test whether it can replace several pixel
   rules instead of sitting on top of them.
