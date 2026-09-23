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
| `Zeus` | majority-vote combiner over the four above; not its own detector |

`Zeus` is the default primary engine (`DEFAULT_PRIMARY_ENGINE`). It runs the
four detection engines in `ZEUS_SUB_ENGINE_NAMES` (`Apollo`, `Talos`,
`Athena`, `Ares`) and takes the `(sector, ring)` most of them returned. At
least `MIN_SUB_ENGINES_TO_VOTE` (3 of the 4) must produce an answer or Zeus
declines to score. A tie goes to the engine listed first in that order, so
`Apollo` wins any tie it is part of.

## Interface

Implement `score()` (and optionally `calibrate()`) from
`opendarts.engines.base`, then add one line to `opendarts/engines/registry.py`.
That is the whole integration surface — an engine never touches capture,
triggering, packages, or the dashboard directly.

Optional capabilities are detected by signature, not by name, so an engine
opts in simply by declaring the parameter:

- `prior_dart_line_px` — previous darts in the visit, to avoid re-detecting them
- `prior_board_xy_mm` — previous darts' board positions

Name-based special-casing is deliberately avoided; a capability check keeps
combiners and stubs working without the dispatcher knowing who it is talking to.

## Configuration

One engine is *primary* (its answer is the throw's result). Any others can be
listed as *also-run*: they score the same throw in the background for
comparison and their results are stored alongside, never substituted. Both are
set by `engine_config` in `data/config.json`:

```json
"engine_config": {"primary": "Zeus", "also_run": ["Apollo", "Talos", "Athena", "Ares"]}
```

That is the default.
