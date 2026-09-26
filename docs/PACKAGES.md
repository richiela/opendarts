# Throw packages

Every scored throw can be saved as a **replay package**: a self-contained
directory holding the exact inputs the throw was scored from, not just the
answer it produced.

That distinction is the point. Because the raw frames and the calibration in
force are both stored, the same throw can be re-scored later through newer
code and produce a *different, better* answer. It is what makes engine
comparison and accuracy regression testing possible without the rig — and
what makes an accuracy claim checkable rather than asserted.

## What's in one

One directory per throw: a few JSON files, plus **one MKV per camera** that
holds the frames. There are no image files.

| File | Size | What it holds |
|---|---|---|
| `stills_cam{0,1,2}.mkv` | ~150 KB each | *unrecorded package*: two frames per camera — the empty board immediately before the dart landed, then the scored frame with the dart in it |
| `clip_cam{0,1,2}.mkv` | ~0.5 MB each | *recorded package*, instead of the stills: the whole run from that empty board to one frame past the scored one — see [Video clips](#video-clips) |
| `result.json` | ~19 KB | what every engine decided — see below |
| `calibration.json` | ~2 KB | the calibration **in force at capture**: `camera_matrix`, `dist_coeffs`, `rvec`, `tvec` per camera |
| `capture_diagnostics.json` | ~2 KB | timings, the settle decision, board-disc state |
| `meta.json` | ~1 KB | session, capture time, visit and throw numbers, calibration package id, and the `video` block that says which file holds each camera's frames and where the two scored frames sit inside it |
| `ad_ground_truth.json` | ~0.8 KB | what Autodarts called the same throw, if the oracle was on |

Clip sizes are from a Linux rig; they scale with the JPEG size of each
frame, so they vary with the camera and resolution.

The frames are stored as **the exact JPEG bytes that were decoded and scored**,
never re-encoded — a replay has to see the exact pixels the live scorer saw,
or it is not a replay. On Linux and Windows those are the camera's own JPEG
bytes. On macOS the system decodes the camera's MJPEG before any application
can see it, so OpenDarts encodes each frame to JPEG itself (quality 50) and
scores the decode of *those* bytes; storage and scoring still see the same
pixels. Either way every clip is checked after it is written: its two scored
frames are read back out of the file as raw JPEG bytes and must equal the
bytes the scored pixels were decoded from (a frame held only as pixels, with
no JPEG, is decoded and compared with the scored pixels instead). A clip that
fails is thrown away and the next fallback written.

A package gets **one** clip per camera, written once. The JSON files come
first, the moment the throw is scored; the clip follows as soon as it is
known which kind it will be (normally a frame period or less later — up to a
few seconds in `mismatch` mode, which waits for Autodarts), and only then
does `meta.json` gain its `video` block. Until that moment the package has
data but no readable frames.

## What `result.json` holds

`result.json` carries the whole picture rather than a single call: the primary
answer (`sector`, `ring`, `board_xy_mm`, `n_cameras_used`,
`max_ray_disagreement_mm`, which camera was an outlier), plus an entry for
**each engine** with its own answer, confidence, diagnostics and how long it
took.

That is what the dashboard's Engines tab reads, and it is why engine
disagreement can be studied after the fact instead of only in the moment. A
throw where three engines agreed and one did not is a recorded fact, months
later, rather than something you had to notice while it happened.

## Video clips

By default every saved throw also records a short **video clip** — a few
frames of the board around the moment the dart landed, one clip per camera in
an MKV container. `video_record_mode` (the Config tab, or the key in
`config.json`) chooses when:

- **`all`** (default) — record a clip for every throw
- **`mismatch`** — record only when Autodarts was on and disagreed with us
- **`never`** — no recordings (every package gets its two-frame stills instead), and the frame ring is not even created

The clip starts at the empty-board frame the throw was scored against and
ends one frame after the scored frame. The capture loop records which frame
of its in-memory frame ring each of those two is, so the clip is taken out of
the ring by those numbers — nothing is searched for — and both ends are
**byte-identical** to what the live scorer used, verified at write time, so a
recorded package re-scores exactly like a still-only one. That is usually 5–8
frames, capped so a board that never settled cannot write a giant clip. If a
recording cannot be had (the frames have left the ring, the ring is off, or
the check fails on any camera), the package gets its two-frame stills
instead and loses nothing.

The clip holds the same JPEG bytes the stills would. A frame with no JPEG
bytes (rare) falls back to a lossless FFV1 encode. Nothing shells out to a
system ffmpeg — video is written through the PyAV wheel, which bundles the
libraries.

Every package is viewable from the dashboard's Engines tab: the **View**
button opens a standalone page, `/packages/{session}/{throw_id}/viewer`. Its
header shows the score, the capture time, the package size, and whether it
holds a recorded clip or stills only; beside the score are the visit's three
darts, and clicking one switches to it. Below are the before/after stills
(click any image for the full-resolution frame) and, for a recorded throw, a
frame-by-frame scrubber (play at 0.25×/0.5×/1×, arrow keys to step).

## Replaying one

Batch-replay every package under a root against the current pipeline code, and
report what changed against the originally-stored live result — sector/ring
flips, `ok` flips, `board_xy` drift:

```sh
./.venv/bin/python3 -m opendarts.capture.rescore_all \
    --package-root data/packages \
    --engine Apollo
```

| Flag | What it does |
| --- | --- |
| `--package-root` | where saved packages live |
| `--engine` | which registered engine to replay through — see [ENGINES.md](ENGINES.md) |
| `--out` | where to write the machine-readable JSON summary |

A summary prints, and the JSON lands under `data/rescore_reports/` by default.

Programmatically, `opendarts.capture.replay` exposes
`replay_throw_with_engine(package, engine_name)` for one throw and
`replay_and_compare(package_dir, engine_name)` to diff a replay against the
package's own stored result.

**A replay reads the package's calibration, never the rig's current one.** The
engine is handed `package.bg_frames`, `package.dart_frames` and
`package.calibrations`, so re-scoring a throw captured under an old solve
reproduces that solve. Otherwise a recalibration would silently rewrite
history and every old package would score against geometry it never saw.

**Do not use replay to measure engine speed.** It is a correctness and drift
tool. The replay path does a disk-based prior-dart-line lookup on darts 2 and
3 of a turn that the live path never pays — median ~97 ms — so timings taken
through it describe the harness, not the engine. To measure an engine, call
`engine.score()` directly.

## Whole-ring frame dumps

Separately from packages: when Autodarts is on and its call disagrees with
ours, a whole-ring dump is written into `data/captures/` — about 0.5 s before
the throw and 0.2 s after, as `manifest.json` + `frames.bin`, each frame
stored as the ring holds it (the JPEG bytes that were scored), with no
re-encode.

A missed dart — nothing detected at all — can dump the whole ring the same
way, from the dashboard or automatically if the oracle scored and we did not.
That is the case a package cannot capture by definition: there is no throw to
write a package for.

## Turning it off

**Config → Capture → Save throw packages.**

Off keeps **scoring, the live board and match history exactly as they are** —
none of those read packages. What stops is the replay corpus, per-throw
diagnostics, Autodarts ground truth on disk, and the Engines tab, which has
nothing to read without them.

The setting is read once per session, so a change applies at the next
**Start**, not to a session already running.

Packages are the reason the switch exists: at around 1.5 MB per throw a
corpus adds up to gigabytes per thousand darts. Turn it off for a rig that only needs
to score. Leave it on if you care about accuracy — a corpus of real throws is
the only way to tell whether a change made scoring better or worse. If you
want packages but not the clip storage, set
`video_record_mode` to `mismatch` or `never` rather than turning packages off
entirely.
