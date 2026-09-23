# Deployment

The rig runs one process: `opendarts.live.run_product` — capture loop,
dashboard, and event push sharing a single camera hub. Do not run
`opendarts.live.capture_daemon` standalone alongside it; both would open the
cameras.

## On the rig

`run.sh` is the supervisor loop: it creates `.venv` if missing, **pulls the
current branch only if asked to** (see "Deploying a new version" below),
installs `requirements.txt`, sources `data/run.env` if present,
starts the product, and restarts it after an exit. `requirements.txt` is
what the product needs to run; it does not install the test dependencies
(`requirements-dev.txt`), which a rig has no use for.

## Deploying a new version

**Push, then ask the rig for an update.** A relaunch runs the same commit
the rig was already running unless something asked for an update. That
keeps "deploy" and "crash" separate events: a rig that falls over
mid-match comes back on the code it was running, not on whatever was
pushed a minute ago.

Three ways to ask:

1. **The dashboard** — Config tab, bottom, *Maintenance* → **Update and
   restart**. Confirms, restarts, waits for the rig to answer, reloads.
   This is the normal path.
2. **The API** — `POST /api/restart {"update": true}`:

   ```bash
   curl -sX POST http://<rig>:8420/api/restart \
        -H 'Content-Type: application/json' -d '{"update": true}'
   ```

   `POST /api/restart` **with no body restarts without pulling**.
3. **`always_update`** — set it to `true` in `data/config.json` on a
   machine that is supposed to follow its branch (a test machine, say).
   That rig pulls on every relaunch. The dashboard hides "Update and
   restart" there and says so on the plain Restart button instead.

The pull happens in the launcher, between the old process exiting and the
new one starting — never inside the running product. **A failed pull is
loud and does not retry**: the one-shot flag is cleared as it is read, the
launcher logs the failure, and the product starts on the code already on
the machine. Ask again once the branch can fast-forward. The launcher
pulls whichever branch is checked out, so a rig can follow a feature
branch by checking it out once.

Capture is **not** started automatically after a restart. Press Start.

## As a service

`run.sh` (macOS/Linux) and `run.ps1` (Windows) already restart the product
whenever it exits, which is the whole supervision requirement. To have it
start at boot without an interactive terminal session, point whichever
service manager your OS uses (launchd, systemd, Task Scheduler) at that
script, with the checkout as its working directory.

## Machine-local configuration

`data/config.json` holds per-machine values — serve port, camera
resolutions, optional Autodarts URL, and whether the launcher pulls. It
is deliberately not tracked: the same code runs on rigs with different
hardware.

Two of those keys are read by `run.sh`/`run.ps1` and by nothing else:

| key | default | meaning |
| --- | --- | --- |
| `always_update` | `false` | Pull on **every** relaunch. For a machine that is supposed to follow its branch. |
| `update_on_next_restart` | `false` | Pull **once**, on the next relaunch. Set by `POST /api/restart {"update": true}`; cleared by the launcher as it reads it. |

`config.example.json` at the repo root is a **reference, read by
nothing** — editing it changes nothing. It lists every key with its real
default, and its `_readme` array carries the per-key notes. Every key is
optional and an absent key means "use the code default", so copy only the
lines you want to change. It omits `capabilities`, which the startup
probe writes and overwrites at every launch.

**The dashboard writes this same file.** `GET /api/config` reports every
key with the value actually in force, and `PATCH /api/config` merges a
partial change into `data/config.json` — the same keys you would
hand-edit. A PATCH is all-or-nothing (an unknown key or a bad value is
refused by name and nothing is written), and its reply names the keys
that only take effect at the next restart. See `docs/LIVE_API.md` for
the key table.

`"host"` and `"port"` set what the dashboard binds to. Precedence is
**CLI flag → config file → code default** (`--host`/`--port` on
`opendarts.live.run_product`, then these keys, then `0.0.0.0` and `8420`).
The Config tab's **Network** section edits the port; it is read once at
process start, so a change applies at the next restart rather than to the
running server, and `PATCH /api/config` says so by returning `port`
under `restart_required`. The **bind address is read-only in the
dashboard on purpose**: a wrong value takes the dashboard away with it at
the next restart, and on a headless rig the fix would be a trip to the
machine. Edit `host` in the file, or PATCH it from a shell on the
machine itself.

`"store_packages": false` turns off writing a replay package per throw
(see Disk use below) for a rig that only needs to score. Scoring, the
live board and match history are unaffected — none of them read packages —
but replay, per-throw diagnostics, the saved Autodarts comparison,
corrections and the Engines tab all go with it. The Config tab's
**Capture** section has a **Save throw packages** control for this key;
it is read once per session, so a change applies at the next Start.

`"cv2_num_threads"` bounds OpenCV's internal worker pool, defaulting to
**1**. Left to itself OpenCV sizes a pool from the core count and wakes
every worker for each of hundreds of tiny calls a second. Measured on an
8-core rig, idle and armed, pinning the pool to 1 cut total process CPU by
**62.8%** (29.9% → 11.1% of the machine). Replaying 130 real throws
across all five engines produced **zero** differing calls, scoring cost
**+2–4 ms per dart** (~3%), and calibration time showed no measurable
change.

Set a **negative** value to keep OpenCV's own default on a rig where that
trade differs. See `opendarts/live/cv2_threads.py` for the full
measurement.

## Where the data lives

Everything this rig writes — `config.json`, logs, throw packages,
calibration packages, frame-ring captures — sits under `data/` in the
checkout, which git ignores. One directory to back up, point elsewhere, or
wipe.

`OPENDARTS_DATA_DIR` moves all of it, which is how you keep the data on a
bigger or faster disk than the one holding the code:

```sh
OPENDARTS_DATA_DIR=/mnt/darts ./run.sh
```

It is read once at startup. `OPENDARTS_LOG_DIR` moves the logs alone
(default `data/logs`).

## Disk use

Two things accumulate on a rig, and nothing deletes either of them for you.

**Throw packages**, one per scored throw: a small JSON file plus one MKV
clip per camera. Every package holds at least a two-frame clip per camera
(baseline and scored frame) — well under 1 MB for the throw. With
`video_record_mode` at its default, `all`, each throw instead keeps a
recorded clip of up to 14 frames per camera, a few MB per throw; `mismatch`
records only the throws Autodarts disagreed with, and `never` none.
Packages are what makes replay possible, so a rig you develop against
should keep them; a rig that only needs to score can turn them off with
`"store_packages": false`.

**Frame-ring captures**, written when someone presses **Save a missed
dart** or captures a misscore. A capture holds every camera frame in the
ring window, stored as the JPEGs the rig held, so its size follows
`frame_ring_seconds`: roughly 2–5 MB per second of ring per camera.

Pull both off the rig if you want to keep them. The Engines tab's
**Delete recorded data** button removes both on that rig.

### The free-space floor

Neither writer will fill the disk. Both of them check the free space
first, against one floor, and stop there:

| `min_free_disk_gb` | what it means |
| --- | --- |
| absent, or `0` | the default floor, **5 GB** |
| a positive number | that many GB |
| a negative number | **the guard is off** — both writers proceed regardless |

Below the floor:

* **Throw packages are skipped.** The throw still scores. The live board,
  the match history and the retail channel are unaffected — none of them
  reads a package — so a rig that runs out of disk keeps playing darts and
  simply stops recording evidence, exactly as `"store_packages": false`
  behaves. The log says so once per session (not once per throw), at
  error level, with the free space, the floor and the throw it skipped.
* **A frame-ring capture is refused**, and the refusal says why, with the
  numbers, where the dashboard button shows it. Nothing half-written, and
  no empty directory left behind.

A ring dump is checked against its own size as well as against the free
space right now — the ring knows how many bytes it is holding before
anything is written. A 4.7 GB dump onto 5.1 GB of free disk is refused,
because allowing it would leave the rig with nothing.

If the free space cannot be read at all, the guard stands aside and logs
that it did, rather than stop the rig recording.

## Diagnostics switch

The rig can compute per-dart and per-tick detail while it scores. It is
**off by default**, and off is the honest setting for measuring how fast
scoring really is, because the work is skipped entirely rather than
computed and thrown away.

Turn it on when you are investigating something:

```sh
curl -X PATCH http://<rig>:8420/api/config -H 'content-type: application/json' \
  -d '{"diagnostics": {"enabled": true}}'
curl -s http://<rig>:8420/api/config            # the whole config, this key included
```

It is the one config key that is **not written to `data/config.json`**:
it applies to the running process only and is back to off at the next
start, so a switch left on cannot quietly distort the next latency
measurement.

With it on, the capture loop records timing and frame-age detail into
the product's log, `data/logs/run_product.log`, and the `websockets` and
`uvicorn` loggers go from warnings-only to full detail. Turn it off again
when you are done.

Calibration diagnostics are not affected by this switch. Calibration runs
once and takes tens of seconds, so its detail is always recorded.

## Security

OpenDarts has **no authentication**, and binds to every network interface
by default. Anyone who can reach the rig on your network can open the
dashboard, watch the camera streams, read the logs and press every button,
including the destructive ones.

That is deliberate for a board on a home or club network, where other
machines and phones need to reach it. It also means:

- **Do not expose a rig to the internet**, directly or through a port
  forward.
- Camera frames are served to anyone on the network, live and at full
  resolution. That is how a second machine consumes this rig's cameras.
- A web page someone on your network opens can, in principle, trigger the
  actions that need no request body, including restart and delete.

If you need a rig somewhere less trusted, put it on its own network
segment rather than relying on the product to keep anyone out.
