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
