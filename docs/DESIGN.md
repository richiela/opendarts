# Design

## What the system does

Three cameras watch a dartboard. The lifecycle layer decides when a dart has
landed and when it has been taken out. On a landing, the capture layer freezes
a background/dart frame pair per camera and hands them, with calibration, to a
scoring engine. The engine returns one `(sector, ring)` answer.

## Layers

**Capture** (`opendarts/capture/`) — a *throw package* is the unit of record:
the raw frames, the calibration in force, and the result. Packages are
self-contained and replayable, which is what makes engine comparison and
regression testing possible without the rig.

**Lifecycle** (`opendarts/lifecycle/`) — throw/takeout decisions live behind
one seam so they can be tested and replayed deterministically. See
`docs/LIFECYCLE.md`.

**Calibration** (`opendarts/calibration/`) — camera intrinsics and the board
solve. Calibration is data, not code: an engine consumes it and must never
depend on how it was produced.

**Engines** (`opendarts/engines/`) — interchangeable scoring implementations
behind one interface. See `docs/ENGINES.md`.

**Live** (`opendarts/live/`) — the capture loop, dashboard, and API. See
`docs/LIVE_API.md`.

## Standing constraints

**Replay is the source of truth.** Any accuracy claim is measured by replaying
saved packages, never by reasoning about the code. A change that cannot be
measured against the corpus is not known to be an improvement.

**Engines are compared, not tuned against one corpus.** An engine that has
been fitted to a specific set of throws is not evidence of anything. Prefer
mechanisms that are explainable from geometry.

**Capability checks, not name checks.** The dispatcher asks whether an engine
declares a parameter, never whether it is called something in particular.
Name-based special-casing has caused real misses.

**Production code stays dependency-light.** Live paths do not import from
`tools/`. Where a small formula must exist in both, it is duplicated and a
test asserts the two agree, so they cannot silently drift.

**A refusal, cap, or fallback must say so in the log.** Returning the reason
to the caller is not enough: a rig turning every request away looks exactly
like a healthy one from the outside, and "no errors on the rig" becomes true
and useless. Silent failures of exactly this kind — a platform-specific import
that stopped another platform booting, a connection cap that refused streams
only in a response body — were found by a person noticing behaviour, not by a
log. So if code declines to do something, degrades to a fallback, or hits a
ceiling, it emits a line saying which and why — throttled if it can repeat,
never suppressed. The cost is a log line; the alternative is an afternoon.

**A health flag must be able to move both ways.** A dashboard audio check
once read:

```js
if (state === 'running') audioBlocked = false;
```

which can only ever clear the warning. So a device whose audio the OS took
away went silent *and went on reporting itself healthy*. A flag that can only
move toward "healthy" guarantees a false negative, and is worse than no flag,
because someone reads it and is reassured.

Assign such a flag from the condition (`blocked = state !== 'running'`)
rather than clearing it on the good branch and forgetting the bad one.
The same applies to any "ok", "connected", "ready" or "healthy" bit: if
you cannot point at the line that sets it false, it does not have one.

**Empty collections are `[]`, never `null` or absent.** On-disk schema treats
"no members" uniformly regardless of why.

**Machine-local values are not tracked.** Ports, camera resolutions, and
integration URLs differ per rig and live in `data/config.json`.

**Captured data is not tracked.** Sessions, calibration packages,
and real-frame test fixtures are large binaries: they live in the untracked
`data/` directory or outside the checkout. Tests that need them skip unless `OPENDARTS_FIXTURES_ROOT`
or `OPENDARTS_ENGINE_CORPUS_ROOT` points at them.
