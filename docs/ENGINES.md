# Engines

A scoring engine turns background frames + dart frames + calibration into one
`(sector, ring)` answer. Engines are interchangeable and comparable: the same
throw package can be replayed through any of them.

## Registered engines

| Name | Shape |
|---|---|
| `Apollo` | per-camera tip detection, board-ROI gate, ray triangulation |
| `Talos` | blob + shaft-line detection with a consensus/override layer |
| `Athena` | per-camera board-crossing, ray/plane intersection, no triangulation |
| `Ares` | board-plane shaft-line concurrency, 2D per camera |
| `Argus` | visual hybrid: what each camera sees at the tip (paint, map, needle visibility), refereed by the cameras' fused board position |
| `Zeus` | majority-vote combiner over the five above; not its own detector |

`Zeus` is the default primary engine (`DEFAULT_PRIMARY_ENGINE`). It runs the
five detection engines in `ZEUS_SUB_ENGINE_NAMES` (`Apollo`, `Talos`,
`Athena`, `Ares`, `Argus`) and takes the `(sector, ring)` most of them
returned. At least `MIN_SUB_ENGINES_TO_VOTE` (3 of the 5) must produce an
answer or Zeus declines to score. A tie goes to the engine listed first in
that order, so `Apollo` wins any tie it is part of and `Argus` is last.

### Argus

Argus scores each camera on its own, from what it sees at the needle tip:
the bed under the tip by board paint colour and by calibration map, how
visible the needle's entry is, how clear the tip is, how far it is from a
wire, and whether the entry sits on a dart thrown earlier in the visit or
on the previous dart's line. Those give each camera a confidence and a
call, and a confidence-weighted vote picks one. The cameras' tip positions
are then fused on the board (each camera is precise across its needle and
uncertain along it); the vote stands if its bed lies inside the fused
point's uncertainty, otherwise the best-supported bed that does wins.

It needs two pieces of visit context, both looked up the same way live and
in replay: the previous dart's line (`prior_dart_line_px`, as Apollo) and
the visit's empty board (`visit_bg_images`, the first dart's background).
`board_xy_mm` is always set when it scores: the fused point if its bed is
Argus's call, else the board point of the most confident camera that made
that call, else the most confident camera's point
(`diagnostics.method`: `fused`, `camera_agrees`, `camera_top`).
`diagnostics.per_camera` holds each camera's board point, call and
confidence.

## Interface

Implement `score()` (and optionally `calibrate()`) from
`opendarts.engines.base`, then add one line to `opendarts/engines/registry.py`.
That is the whole integration surface — an engine never touches capture,
triggering, packages, or the dashboard directly.

Optional capabilities are detected by signature, not by name, so an engine
opts in simply by declaring the parameter:

- `prior_dart_line_px` — previous darts in the visit, to avoid re-detecting them
- `prior_board_xy_mm` — previous darts' board positions
- `visit_bg_images` — the visit's empty board per camera (the first dart's
  background), so an engine can tell what earlier darts now cover

Name-based special-casing is deliberately avoided; a capability check keeps
combiners and stubs working without the dispatcher knowing who it is talking to.

## Configuration

One engine is *primary* (its answer is the throw's result). Any others can be
listed as *also-run*: they score the same throw in the background for
comparison and their results are stored alongside, never substituted. Both are
set by `engine_config` in `data/config.json`:

```json
"engine_config": {"primary": "Zeus", "also_run": ["Apollo", "Talos", "Athena", "Ares", "Argus"]}
```

That is the default. Zeus always votes with all five engines; a rig whose
`data/config.json` lists `also_run` explicitly only stores (and shows) the
engines named there, so add `"Argus"` to such a list to see its column.
