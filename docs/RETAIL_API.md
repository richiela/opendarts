# Retail API

Everything a scoreboard, overlay or game client needs. One WebSocket and
one HTTP endpoint.

This is the *retail* surface — deliberately smaller than the operational
one. It answers "what is the game state, and what just happened". It
carries no engine internals, no camera health and no diagnostics.

```
ws://<rig>:8420/api/live          live state
GET http://<rig>:8420/api/live/recent    completed visits, for reconnects
```

No authentication. Anything that can reach the rig can subscribe.

---

## The socket

Connect and you immediately receive a `hello` — a complete snapshot of
now, before any live traffic:

```json
{
  "type": "state",
  "event": "hello",
  "running": false,
  "status": "no_live_capture",
  "visit": null,
  "n_darts": 0,
  "darts": [],
  "at": "2026-09-09T22:24:07.658489+00:00"
}
```

After that you get two message types, told apart by `type`: `state` and
`THROW_DETECTED`.

### `state` — a complete snapshot, never a delta

Sent whenever the game state genuinely changes. Every field is always
present; an empty turn is `n_darts: 0, darts: []`, never an omitted key.
You never need to merge — replace your local state with each message.

| field | meaning |
|---|---|
| `event` | why this was sent (see below) |
| `running` | is the capture loop running |
| `status` | board state — `throw`, `takeout`, `stopped`, `connecting`, `camera_error`, `no_live_capture` |
| `visit` | id of the turn in progress, or `null` |
| `n_darts` | darts thrown this turn |
| `darts` | the darts, in throw order |
| `at` | when this message was sent |

**You will not get a message on every tick.** A `state` is published only
when the snapshot actually differs from the last one sent, or when a
named event says something happened that a client should know about even
if the snapshot looks the same. This keeps the channel quiet on an idle
board.

### `THROW_DETECTED` — the raw scoring event

Fired the instant a dart is scored, just before the `state` that includes
it. It is the capture loop's own event, rebroadcast verbatim (abridged
here; it carries a few more fields, such as `throw_id` and `session`,
that a scoreboard can ignore):

```json
{
  "type": "THROW_DETECTED",
  "visit_id": "visit_1788937678491",
  "visit_index": 0,
  "sector": 20,
  "ring": "treble",
  "ok": true,
  "captured_at_utc": "2026-09-09T22:00:00+00:00",
  "emitted_at_utc": "2026-09-09T22:00:00.1+00:00",
  "ts": "2026-09-09T22:24:18.776177+00:00"
}
```

It uses the *internal* vocabulary (`sector: 20`, `ring: "treble"`), not
the wire vocabulary below. **If you only want scores, ignore it and read
`darts` on the `state` that follows.** It is here for a client that wants
the earliest possible signal, or the `emitted_at_utc` source timestamp
for latency work.

---

## Events

| `event` | what happened |
|---|---|
| `hello` | your connection's opening snapshot |
| `throw_detected` | a dart was scored |
| `throw_corrected` | a dart's score was changed after the fact |
| `visit_complete` | the turn ended — board cleared |
| `manual_reset` | the turn was abandoned by an operator Reset |
| `takeout_started` | darts are being removed (the board clearing is then `visit_complete`) |
| `started` / `stopped` | capture loop started or stopped |
| `connecting` | cameras opening |
| `camera_error` / `camera_error_cleared` | a camera failed or recovered |
| `no_live_capture` | this process has no capture loop at all |

Treat any other `event` value as a plain snapshot: the message is still
complete, and `status` says where the board is.

`visit_complete` and `manual_reset` are deliberately distinct: a
completed turn and an abandoned one are different things to a game
engine.

---

## Darts

Each entry in `darts`:

```json
{ "label": "T20", "sector": 20, "ring": "treble", "value": 60,
  "captured_at_utc": "2026-09-09T22:00:00+00:00" }
```

`value` is the dart's score — already multiplied. Sum them for the turn
total; do not compute from `sector` and `ring` yourself.

| what landed | `label` | `sector` | `ring` | `value` |
|---|---|---|---|---|
| treble 20 | `T20` | 20 | `treble` | 60 |
| double 16 | `D16` | 16 | `double` | 32 |
| single 5 | `S5` | 5 | `single_inner` or `single_outer` | 5 |
| inner bull | `BULL` | 25 | `bull` | 50 |
| outer bull | `25` | 25 | `outer_bull` | 25 |
| off the board | `MISS` | 0 | `miss` | 0 |
| landed, unreadable | `failed to score` | 0 | `""` | 0 |

`single_inner` and `single_outer` are both worth the sector value — the
distinction is where on the bed it landed, and both label as `S<n>`.

**`failed to score` is a real outcome, not an error.** The dart is
physically in the board but no score could be read. It occupies its slot
and counts toward `n_darts`, so your dart count and the board always
agree.

### Corrections

A dart already reported can be corrected by an operator. When that
happens you receive a `state` with `event: "throw_corrected"` and the
affected dart carries `"corrected": true`, with `label` / `value`
showing the **corrected** score:

```json
{ "label": "T20", "sector": 20, "ring": "treble", "value": 60,
  "corrected": true, "captured_at_utc": "..." }
```

Replace your stored value. The original is not resent.

### Sending a correction

Corrections come from the dashboard's Scoring tab or from a client of
yours. To send one:

```
POST /api/visits/{visit_id}/throws/{index}/correct
```

`visit_id` is the `visit` id you were given on the live channel, and
`index` is the dart's position in that visit, `0`, `1` or `2`. The body:

```json
{ "ring": "treble", "sector": "20", "source": "manual", "note": "operator call" }
```

| field | required | value |
| --- | --- | --- |
| `ring` | yes | one of `bull`, `outer_bull`, `single_inner`, `treble`, `single_outer`, `double`, `outside` |
| `sector` | only when the ring has one | the wedge number as a string, `"1"`–`"20"`. Omit it for `bull`, `outer_bull` and `outside` |
| `source` | no | who says so; defaults to `manual` |
| `note` | no | free text kept with the correction |

Use these words exactly. They are the vocabulary the scorer itself
produces, so a correction written any other way could never be compared
against what an engine answered.

A correction that does not name a saved throw answers `404`. The
correction is written into that throw's package, so a rig with
`store_packages` off (or below its free-disk floor) has nothing to
correct and answers `404` too. One that
names an impossible combination, a treble with no sector, or a bull in
sector 20, answers `400` and changes nothing.

On success the rig records the correction against the saved throw and
sends every listener a `state` with `event: "throw_corrected"`, exactly
as described above, so your own client sees its correction come back
through the same channel as everyone else's.

---

## Reconnecting mid-match

The socket carries no history — `hello` tells you about *now*, not what
came before. For the turns already played:

```
GET /api/live/recent
GET /api/live/recent?limit=5
```

```json
{
  "visits": [
    { "visit": "visit_1788937663938",
      "n_darts": 3,
      "darts": [ ... ],
      "total": 100,
      "completed_at_utc": "2026-09-09T07:07:58.191000+00:00",
      "reason": "takeout" }
  ],
  "count": 1,
  "limit": 12,
  "buffer_max": 12
}
```

Oldest first, so you can replay in order. Darts use the same shape and
the same correction handling as the socket.

Three things to know:

- **The turn in progress is never included.** The socket owns that one.
  Take history here, current state from `hello`, and you will not double
  count.
- **It is in memory, not on disk.** Bounded to the most recent 12 visits,
  and it does not survive a rig restart. It is deliberately independent
  of whether throws are being saved as files, so clearing that storage
  never erases your match history.
- **Empty turns are skipped** — a Reset with nothing thrown is not a
  visit you need to replay.

---

## Suggested client shape

1. `GET /api/live/recent` for the turns already played.
2. Connect the socket; `hello` gives the turn in progress.
3. Replace local state on every `state` message.
4. On `visit_complete` or `manual_reset`, bank the turn and start a new one.
5. On `throw_corrected`, update the dart the message identifies.
6. Reconnect on drop and repeat from step 1.

Game rules — 501, cricket, players, legs — are entirely yours. This
surface reports what landed in the board and nothing about what it means.

## What is not here

Engine internals, per-camera detail, calibration, latency measurements,
camera images and diagnostics all live on the operational surface and are
out of scope for a retail client.
