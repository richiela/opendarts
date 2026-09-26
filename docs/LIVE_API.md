# Live API

Served by `opendarts.live.server` (FastAPI), started via
`opendarts.live.run_product`. Default port `8420`, set in
`data/config.json` or with `--port`.

`GET /` serves the dashboard: the new one or the classic one, whichever the
rig's switch says (`/api/dashboard` below; the same API underneath both).
`/?ui=new` or `/?ui=classic` shows the other on one screen without
flipping it; `/?display` is the new one in its display role (Displays,
below). This lists the whole HTTP and WebSocket surface. There
is no authentication (see `docs/DEPLOYMENT.md`, Security). For a scoreboard
or game client, `docs/RETAIL_API.md` is the smaller, stable surface to build
on.

## State and control

| Route | Purpose |
|---|---|
| `GET /api/health` | liveness, plus `capabilities` (which external tools exist on this machine), `build` (the running version), `streams` (open and maximum MJPEG connections) and this process's `pid` — comparing the pid is how a caller tells "the rig is back" from "the process it asked to restart has not died yet" |
| `GET /api/state` | current rig state. Its `capture_loop` section carries `status_epoch` / `status_seq` (see [Ordering stamps](#ordering-stamps)) |
| `GET /api/cameras/status` | per-camera backend, resolution, frame counts and errors, plus where the slot's JPEG comes from: `jpeg_passthrough` (the camera's own) or `jpeg_synthetic` / `jpeg_synthetic_quality` (encoded by the rig — see `docs/STREAMING.md`) |
| `GET /api/frame-health` | whether frames are keeping up on **both** sides: `capture` (is the pump reading every frame the camera produces — a 30fps camera delivering 21 is degrading silently) and `publish` (is the virtual-camera consumer collecting every frame we hand it) |
| `POST /api/start` / `POST /api/stop` | start or stop the capture loop. Replies carry `status_epoch` / `status_seq` |
| `POST /api/reset` | reset session state |
| `POST /api/restart` | exit cleanly so the launcher (`run.sh` / `run.ps1`) starts the process again; replies with `restarting` and the old `pid`. Without a launcher loop it only stops the process. Optional body `{"update": true}` asks the launcher to `git pull` first: it records `update_on_next_restart` in `data/config.json`, and refuses to restart if it cannot. **No body restarts without pulling** |
| `GET /api/restart` | reports, without restarting anything, the two launcher update flags (`always_update`, `update_on_next_restart`) — what a restart *would* do. Both are also keys of the config document |
| `GET /api/logs/{name}` | tail a log file as plain text; `?n=` lines, default 200. `name` is `run_product` (the product's log), or `server` / `capture_daemon` when those modules are run standalone |
| `GET /api/live/recent` | completed visits, for a retail client reconnecting mid-match |

## Configuration

**Every setting is one document.** `GET /api/config` returns the whole
effective config; `PATCH /api/config` merges a partial one into
`data/config.json`. There is no per-setting route.

| Route | Purpose |
|---|---|
| `GET /api/config` | the whole **effective** config: every key the product reads, valued at what is actually in force — the file's value where it has a usable one, the code default where it does not, and the LIVE value for the keys a running process holds in memory. Plus `restart_required` (keys whose saved value is not the one this process is running) and `runtime` (the facts beside the settings: the port really bound, whether Autodarts is connected (`ad`), what hardware sits at each device index, what the ring holds now, whether a session is running). `?refresh_camera_names=true` re-asks the platform for camera names, which is otherwise cached |
| `PATCH /api/config` | merge a partial document. **All-or-nothing**: every key named is validated before any key is written, so a request that fumbles one field changes nothing. An unknown key, a bad value, or `capabilities` (written by the startup probe) is a `400` with one reason *per key*. On success: `200` with `changed`, `persisted`, `applied_live`, `restart_required`, `notes` (why a key was saved but not applied), the new `config` and `runtime` |

### The document

Keys are exactly `data/config.json`'s, nested the way that file nests
them — this is the same file an operator hand-edits, not a parallel
store. See `config.example.json` for every default.

| Key | What it is | When a change takes effect |
|---|---|---|
| `host` | the interface the dashboard and API bind to | next restart. Writable here, but the dashboard shows it read-only on purpose: a wrong bind address takes the dashboard away with it |
| `port` | the TCP port the dashboard and API are served on | next restart — uvicorn's socket is bound before the app object exists |
| `ad_base_url`, `ad_enabled` | where Autodarts is, and whether it is consulted | **live** — the listener reconnects or disconnects on the spot. `ad_enabled` also drives virtual-camera publishing and device registration |
| `camera_devices`, `camera_urls` | which device index or stream URL feeds each slot, in slot order | **live** — the hub is reconfigured in place, cycling the capture loop if cameras are open. Refused while darts are on the board; then it is saved and reported restart-required |
| `camera_resolutions`, `reprojection_targets_px` | per-camera capture resolution and calibration target | next restart |
| `store_packages` | whether throws are saved to disk at all | next **Start** — read once per session, so a session cannot change its storage behaviour halfway through its own package set |
| `video_record_mode` | per-throw video: `never` (no frame ring at all), `mismatch` (record a clip only when Autodarts disagreed), or `all` (every dart) | next restart — decides whether the frame ring is even created |
| `min_free_disk_gb` | the free-space floor both on-disk writers stop at | **live** — both writers re-read it per write |
| `frame_ring_seconds` | seconds of raw frames kept in memory | **live**, deliberately: this setting is holding gigabytes right now. Capped at 60s, and the refusal quotes the memory the window would cost on this rig |
| `frame_ring_max_gb` | optional hard ceiling on the ring, in GB | next restart (read when the ring is created) |
| `publish_virtual_cameras`, `v4l2_format` | whether the virtual cameras are fed (`null` follows `ad_enabled`), and the Linux loopback pixel format | next restart |
| `cv2_num_threads` | how many worker threads OpenCV may use | next restart |
| `detect_from_small_decode` | detection reads each frame's JPEG decoded straight to 1/4 grey, and only scored frames are decoded in full (`true`, the default); `false` decodes every frame in full, as before | next restart |
| `idle_timeout_sec` | seconds of no darts before the capture loop auto-stops; `0` disables it | **live**. Clamped to `max(0, n)` |
| `lifecycle_settings.dart_stable_frames` | still frames a landed dart must hold before it is scored (1–5; fewer is faster but more error-prone) | **live**, on the next frame |
| `engine_config` | which engine is primary, which also run, the per-engine timeout | next restart |
| `always_update`, `update_on_next_restart` | whether a restart pulls the latest code first | read by the launcher at the next restart |
| `diagnostics.enabled` | the per-dart **diagnostics switch**, default off. See the Diagnostics switch section of `docs/DEPLOYMENT.md` | **live**, on the next capture-loop iteration. The one key that is **not persisted**: it returns to off at the next start |
| `capabilities` | what external tools this machine has | **read-only** — written by the startup probe, and a PATCH naming it is refused |

A key whose live half is not wired in this process (no capture loop, no
Autodarts listener, no camera hub) is still persisted — it is the rig's
own config file either way — and the reply says so in `notes` and lists
it under `restart_required` rather than implying it took effect.

## Cameras

Which device or stream feeds each slot is a **setting** — see
`camera_devices` / `camera_urls` in the config document above. These are
the picture routes.

| Route | Purpose |
|---|---|
| `GET /api/cameras/{cam_id}/stream.mjpg` | live MJPEG preview (`multipart/x-mixed-replace`, 960px wide, q75, ≤12 fps), renders in a plain `<img>`. `503` when the capture loop is not running or the connection budget is full; `404` for an unknown camera. `?full=1` is the full-resolution feed another rig scores from: it forwards the JPEG the slot holds, unre-encoded — the camera's own on Linux and Windows, a synthetic q50 JPEG on macOS. See `docs/STREAMING.md` |
| `GET /api/cameras/{cam_id}/overlay-rgba.png` | the calibration overlay as a **transparent layer**, sized to match the MJPEG preview, for compositing over the live stream. What the dashboard uses |
| `GET /api/cameras/{cam_id}/overlay.png` | the same geometry **baked into a still photograph**, for a single saveable composited image |
| `GET /api/cameras/{cam_id}/snapshot.png` | full-resolution lossless still, fetched fresh per request. Not the preview path — for anything wanting an inspectable frame |

## Throws and packages

A saved throw package is `meta.json` plus **one MKV clip per camera** —
no loose image files. Each package gets ONE clip per camera, written
just after its JSON: a *recorded* clip of the whole throw when
`video_record_mode` asks for one, otherwise (or when a recording cannot be
had) a two-frame *stills* clip -- baseline and scored frame. `meta.json`
names the clip once it is written. See `docs/PACKAGES.md`.

| Route | Purpose |
|---|---|
| `GET /api/packages` | list saved throw packages. `has_video` on a record means it has a **recorded** clip, not the two-frame stills clip an unrecorded package has |
| `GET /packages/{session}/{throw_id}/viewer` | an HTML **page** (not JSON) for one throw: a 2×3 stills grid (3 cameras × before/after), plus a clip scrubber when the throw has a recorded clip. It fetches its data from the routes below |
| `GET /api/packages/{session}/{throw_id}/frame/{cam}/{kind}.png` | one still of a throw — `kind` is `bg` (baseline) or `after` (the scored frame), read from the package. Default `fmt=png` is full-resolution lossless; `?fmt=jpeg` is a smaller display-quality encode |
| `GET /api/packages/{session}/{throw_id}/clip/{cam}/{index}.png` | one clip frame at full resolution, lossless |
| `GET /api/packages/{session}/{throw_id}/clip.json` | every frame of every camera's recorded clip as base64 JPEG data URLs, plus `fps` and each camera's `commit_index`, in one payload. Display quality only. `404` when the throw has no recorded clip |
| `GET /api/packages/{session}/{throw_id}/clip/{cam}.mjpg` | a camera's recorded clip as a looping MJPEG stream an `<img>` can play (browsers do not decode MKV). Display quality only; `404` without a recorded clip |
| `GET /api/recorded-data` | how much recorded data is on this rig, per kind — throw packages and frame-ring captures, each with a count and a byte total, plus whether a dump is mid-write. Walks the disk, so call it before a delete, not on a poll |
| `POST /api/packages/delete-all` | delete ALL of this rig's recorded data — every throw package **and** every frame-ring capture. Reports what went per kind (counts and bytes) and broadcasts `PACKAGES_UPDATED` (with `count`) so open tabs refresh. Refused, with a reason and nothing deleted, while a capture is being written |
| `GET /api/board/photo` | the Scoring tab's board photo: a straight-on JPEG of the empty board, built from every camera just after Start and again after each takeout or Reset, once the board has settled (about 0.3 s after `VISIT_CLEARED`) — never while a dart is being scored. A takeout's photo is skipped when the empty board has not visibly changed since the last one (same calibration, mean grey change ≤ 2 levels and ≤ 0.2% of the board changed by 25+ levels, in every camera), so the version stays put. It spans ±235 mm around the bull, +y up. `404` until the first photo, a moment after Start. Its version is `visit.board_photo_version` in `/api/state` and the `BOARD_PHOTO` event; fetch `?v=<version>` and it can be cached for good |
| `POST /api/packages/{session}/{throw_id}/mark-ad-wrong` | operator marks "Autodarts was wrong on this throw" |
| `POST /api/visits/{visit_id}/throws/{index}/correct` | record a ground-truth correction — body and errors in `docs/RETAIL_API.md` |
| `POST /api/packages/{session}/{throw_id}/capture-misscore` | write the raw frames from around THIS throw out of the frame ring — ~1s, anchored on the throw's own recorded capture instant, not on "the last N frames". Refuses, **with the numbers**, when the throw is older than the ring still reaches |
| `GET /api/calibration` | the full adopted calibration: camera_matrix / dist_coeffs / rvec / tvec per camera, plus the derived camera position, azimuth, elevation and distance in board coordinates |
| `POST /api/calibration/refresh` | re-solve calibration |
| `GET /api/calibration/progress` | how far along the current (or last) calibration is: the same body as the `CALIBRATION_PROGRESS` event, for a page that opens part-way through |
| `POST /api/calibration/relearn-ring-geometry` | delete this rig's learned ring geometry so it relearns from scratch. Rarely needed: a calibration that finds the cameras have moved relearns it by itself and says so (`ring_geometry_relearned` on `/api/state`'s calibration section, the refresh response and `CALIBRATION_STATUS`). Idempotent: `cleared: false` when nothing was stored |

## Throw capture ring

An in-memory ring of the last N seconds of every camera's frames, tapped
off the camera hub, so a dart that was MISSED (no package exists) or
MISSCORED (a package exists but the call is wrong) can still be examined
afterwards. Package replay tests scoring; frame replay tests detection,
lifecycle and settling — and a missed dart has no package at all, so this
is the only way one can be debugged.

A slot that holds a JPEG — the camera's own or a synthetic one, which on
local cameras is every slot — is kept as that JPEG, roughly 80–160 KB a
720p frame: a 5-second, three-camera buffer is a few tens of MB. A slot
with no JPEG is kept as raw pixels at ~2.7 MB a frame, where three
cameras cost ~270 MB/s. Reading a capture back decodes it to the same
pixels the rig scored. The Config tab prices the setting from the rate
this rig's ring has actually measured (`estimate_source: "measured"`), and
from uncompressed pixels until it has one (`"pixels"`). See
`opendarts/capture/frame_ring.py` and `opendarts/capture/throw_capture.py`.

The window is `frame_ring_seconds` in the config document, and the ring's
own state — seconds held, MB, whether it is capped or paused, writer
progress, what the current setting costs on this rig's camera count —
comes back under `runtime.frame_ring` on `GET`/`PATCH /api/config`.

| Route | Purpose |
|---|---|
| `POST /api/frame-ring/capture-missed` | write the WHOLE buffer — the missed-dart trigger. Pauses the ring during the write so peak memory stays flat, then resumes it |

## Audio

Spoken dart calls, pre-rendered to clips and **played in the browser**.
The rig plays nothing; it serves the clips and says what to say.

On/off, volume and voice are per-device values in each browser's
localStorage, so there is no route for them: the TV above the board and
the tablet in your hand are different listeners, and a single rig-wide
switch could not express that.

| Route | Purpose |
|---|---|
| `GET /api/audio/voices` | everything a browser needs to arm itself, in one request: the installed voice sets each with its **phrase-set coverage**, the default, and the whole phrase → filename map |
| `GET /api/audio/clips/{voice}/{name}` | the bytes of one clip, `audio/mpeg`, cacheable for a day. `voice` must be an installed set and `name` one of the generated filenames from the map |
| `GET` `POST /api/audio/clients` | each screen self-reporting whether it can actually be heard: `client_id`, `label`, `enabled`, `blocked`, `voice`, `volume`, `ready`, `context_state`. `GET` returns the fresh ones with an `age_s`, dropping anything quiet for 45s. **A report, never a control** — nothing here decides what gets broadcast |

`/api/audio/clients` exists because of browser autoplay policy: a TV
gets its permitting tap at setup and silently loses it on any reload.
The TV knows; nobody at the oche does — so every dashboard shows every
other dashboard's state.

## Dashboard

Which page `/` serves — `new` (the default) or `classic` — is one rig-wide
switch, kept as the `dashboard_ui` section of `data/config.json` and flipped
from either dashboard, with no restart.

| Route | Purpose |
|---|---|
| `GET /api/dashboard` | `ui` (`new` or `classic`) and the `choices` |
| `PUT /api/dashboard` | `{"ui": "new" \| "classic"}` — flip it; broadcasts `DASHBOARD_SWITCHED`. Anything else is a 400 |

## Displays

A display is a screen that only shows — a TV beside the board, in a kiosk
with no keyboard. `GET /?display` is the new dashboard in that role: no
bar, no pointer, nothing to press. Its settings live **on the rig**, keyed
by the display's id, and any controller (the ordinary dashboard) changes
them; each change is pushed to the display at once as `DISPLAY_UPDATED`.
So two displays can look different, any controller can change either, and
a display that reboots comes back looking the same. Settings persist as
the `displays` section of `data/config.json`; presence does not.

`?display=tv` (or `?id=tv`) on the display's URL pins its identity (otherwise it makes one up
and keeps it in the browser); `?name=Lounge%20TV` names it on first sight.

| Route | Purpose |
|---|---|
| `GET /?display` | the page, in its display role (always the new dashboard, whatever the switch says) |
| `GET /api/displays` | every display: `id`, `name`, `settings`, `online` (reported within 70 s), `age_s`, and `info` (what it last said about itself: size, sound state) |
| `POST /api/displays/hello` | a display reporting in, every ~20 s: `display_id`, optional `name` (used only when it is first seen), `info`. Registers it the first time; answers with its `display` record |
| `PATCH /api/displays/{display_id}` | change a display: `name` and/or a partial `settings` — `layout` (`split` = scoring beside every engine's call, `scoring`, `engines`), `text_size` (`normal` `large` `huge`), `theme` (`dark` blueprint, `light` paper), `board_view` (`photo` `diagram`), `sound`, `voice` (`""` = the rig's default), `volume` (0–1), `locked`. An unknown key or bad value is a 400; a locked display takes only `{"settings": {"locked": false}}` and answers anything else with 409. Broadcasts `DISPLAY_UPDATED` |
| `POST /api/displays/{display_id}/identify` | put the display's name across its screen for a few seconds (`DISPLAY_IDENTIFY`), to tell which screen a row is |
| `POST /api/displays/{display_id}/reload` | reload the page on a display nobody can reach with a keyboard (`DISPLAY_RELOAD`) |
| `DELETE /api/displays/{display_id}` | forget it (`DISPLAY_FORGOTTEN`). A display still open registers again, fresh, on its next report |

The lock is a guard against the wrong row being tapped, not a security
boundary: like everything here, anyone who can reach the rig can unlock it.

## Events

`WebSocket /api/events` is the dashboard's push stream. The server
ignores anything a client sends. Treat it as at-least-once and reconcile
against `GET /api/state` on reconnect.

Every message is a JSON object with a `type` and, except `IDLE_TIMEOUT`,
an ISO `ts`:

| `type` | when | notable fields |
|---|---|---|
| `HELLO` | first message on connect | `page_version`, `state` (same as `GET /api/state`), `count` (total packages), `packages` (newest 20) |
| `TRIGGER_STATE` | the board's lifecycle state changes | `state`, `session`, `dart_count`, `visit_id`, `emitted_at_utc`, ordering stamp |
| `CAPTURE_LOOP_STATUS` | Start / Stop progress and results | `ok`, `starting`, `running`, `reason`, ordering stamp |
| `DART_CALL` | a dart was scored — sent just **before** its `THROW_DETECTED` | `phrase` (e.g. `"treble 20"`, already what a caller would say), `visit_id` |
| `THROW_DETECTED` | a dart was scored | `visit_id`, `visit_index`, `sector`, `ring`, `ok`, `board_xy_mm`, `captured_at_utc`, `emitted_at_utc`, `session`, `throw_id`; `dart_axis` (unit shaft direction, board frame, +z toward the thrower) and `flight_color` (`#rrggbb`) when they could be worked out |
| `BOARD_PHOTO` | a new board photo is ready: just after Start, and after a takeout or Reset when the board looks different | `version`, `visit_id` (`null` for the Start photo, else the visit that begins on this board) — fetch `GET /api/board/photo?v=<version>` |
| `CALIBRATION_PROGRESS` | a calibration moved on — pushed only while one runs, and once when it ends | `active`, `stage` (`capture` `orientation` `solve` `refine` `finish`), `label`, `detail` (e.g. `"round 2"`, `"attempt 1 of 2"`), `fraction` (0–1, the whole calibration), `stages` (each `key`, `label`, `state`: `done` `now` `todo`), `cameras` (`"0"`: `waiting` `working` `done` `failed`), `elapsed_s`, `succeeded` (once it ends), `error`, `seq` |
| `DASHBOARD_SWITCHED` | the rig's dashboard switch flipped — open dashboards reload onto the other page (a display, or a screen opened with `?ui=`, stays) | `ui` (`new` or `classic`) |
| `DISPLAY_UPDATED` | a display's name or settings changed | `display` (the record `GET /api/displays` lists) |
| `DISPLAY_IDENTIFY` | a controller asked a display to show its name | `display_id`, `name` |
| `DISPLAY_RELOAD` | a controller asked a display to reload | `display_id` |
| `DISPLAY_FORGOTTEN` | a display was forgotten | `display_id` |
| `THROW_CORRECTED` | a correction was recorded | `visit_id`, `visit_index`, `live_sector` / `live_ring` (what was scored), `corrected_sector` / `corrected_ring`, `source`, `note` |
| `VISIT_CLEARED` | a turn ended (board cleared, or Reset) | `previous_visit_id`, `visit_id` (the new one), `n_darts`, `reason` |
| `PACKAGES_UPDATED` | packages were added, changed or deleted | `count` (true total), `new_count`, `packages` (the changed records) |
| `CALIBRATION_STATUS` | calibration ran (at Start or on request) | `cameras` (per-camera status), `source`, `ring_geometry_relearned` |
| `AD_BOARD_STATUS` | Autodarts' own board status changed | — |
| `AD_CONNECTION` | the Autodarts listener connected or disconnected | same shape as `runtime.ad` in the config document |
| `IDLE_TIMEOUT` | the capture loop auto-stopped after `idle_timeout_sec` without darts | `idle_timeout_sec` |

`DART_CALL` is its own message so it can leave first (sound is the
slowest thing a human notices) and so a speaker-only client can handle
one type and ignore the scoring feed. It is sent unconditionally; whether
a noise is made is each device's own decision.

**`page_version`** fingerprints the dashboard's page assets. A dashboard
that reconnects and sees a different value reloads itself, so open
screens pick up a new version after an update.

`WebSocket /api/live` is the **retail** channel — see `docs/RETAIL_API.md`.

### Ordering stamps

An HTTP reply and a WebSocket message can overtake each other, so a
snapshot can arrive after a newer one. Two facts carry a stamp so a
client can drop the stale one:

- **Capture status** — `/api/state`'s `capture_loop` section, the
  `POST /api/start` / `/api/stop` replies, and every
  `CAPTURE_LOOP_STATUS` and `TRIGGER_STATE` carry `status_epoch`
  (per process) and `status_seq` (rises with every snapshot).
- **Autodarts connection** — `runtime.ad` and `AD_CONNECTION` carry
  `connection_epoch` and `connection_seq`.

Keep the newest `seq` you have applied for the current epoch and ignore
anything lower. A different epoch means the server restarted: accept it
and start counting again.

## Optional external integration

The rig can optionally fetch committed-throw data from an Autodarts instance
for side-by-side comparison. It is a second opinion recorded alongside a
throw, never a substitute for this project's own result, and the rig runs
fully without it.

Where it is and whether it is consulted are `ad_base_url` and
`ad_enabled` in the config document; both apply live. `runtime.ad` on
that same answer is `{available, connected, connection_epoch,
connection_seq}`: `available` says whether this process has a listener
at all, and `connected` whether its socket is up **right now** —
"enabled" and "reachable" are different facts. `AD_CONNECTION` pushes the
same object whenever the connection comes up or drops.

**Turning it off is not cosmetic.** While it is on, the rig keeps a
WebSocket connection open (retrying with backoff if Autodarts is not
running) and matches each throw against Autodarts' own call. Disabling
stops Autodarts being contacted at all.

On Windows and Linux the same toggle also decides, unless
`publish_virtual_cameras` is set explicitly, whether the virtual cameras
that other software reads are registered and fed (see
`docs/STREAMING.md`).
