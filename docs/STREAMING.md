# Camera streaming

OpenDarts opens each physical camera itself. Two other things often want those
same frames at the same time: **another machine** on the network (a second rig,
or a viewer), and **another program on the same machine** (Autodarts, say,
scoring the same board). A physical camera cannot simply be shared by two
programs, so OpenDarts re-publishes what it captures.

There are two independent mechanisms. The **network stream** (HTTP) feeds other
machines. The **local virtual camera** feeds other programs on the same box.

---

## 1. The network stream (HTTP MJPEG)

    GET /api/cameras/{cam}/stream.mjpg          # preview
    GET /api/cameras/{cam}/stream.mjpg?full=1   # transport

A `multipart/x-mixed-replace` MJPEG stream, one per camera slot. It comes in two
grades because it serves two opposite purposes:

| | preview (default) | transport (`?full=1`) |
|---|---|---|
| purpose | a picture in a dashboard | another machine's **camera feed**, scored from |
| resolution | downscaled to 960px wide | full, never downscaled |
| quality | q75 encode | the JPEG the slot already holds, forwarded as-is (see §2) |
| rate | capped at 12 fps | every new frame the camera produces |
| if it stalls | closes after 5 s of no new frames, so a frozen picture is never served as if live | same |

Both grades return `503` with a JSON reason when the capture loop is not
running, and `404` for a camera id this rig does not have.

The two grades have **separate connection budgets** — each allows 4 viewers per
camera plus one spare set, so `5 × cameras` connections — so the cosmetic
preview can never starve a load-bearing transport consumer. A refused stream
returns `503` with a reason, and `GET /api/health` reports the open and maximum
counts under `streams`.

Full-resolution single frames are also available as `snapshot.png` /
`overlay.png` (lossless PNG) for anyone inspecting detail rather than watching.

### Consuming it from another machine

Point a second OpenDarts process at the first with `--camera-url` (or the
`camera_urls` config key). It expands to the transport URL per slot:

    http://<rig>:8420/api/cameras/{0,1,2}/stream.mjpg?full=1

and reads it through `opendarts.live.remote_capture` exactly as if the cameras
were local. `?full=1` is what asks for the full-resolution frame; without it the
consumer would be scoring a downscaled q75 preview. The consumer keeps the JPEG
parts it receives, so if it restreams them they pass on unchanged.

---

## 2. What the transport stream sends

Every local camera slot holds a JPEG next to its decoded pixels, and the rig
scores the decode of exactly those bytes. `?full=1` forwards those bytes
verbatim — no encode — so the consumer scores the same picture this rig
scored. Where the JPEG comes from depends on the capture backend:

| OS | capture backend | JPEG the slot holds | `/api/cameras/status` |
|---|---|---|---|
| **Linux** | V4L2 with `CAP_PROP_CONVERT_RGB` off — hands back the camera's JPEG as a 1×N buffer | the **camera's own** | `jpeg_passthrough: true` |
| **Windows** | a Media Foundation Source Reader asked for the MJPG native type with converters disabled (`win_mf_capture`) | the **camera's own** | `jpeg_passthrough: true` |
| **macOS** | AVFoundation, which hands OpenCV decoded pixels only | a **synthetic** q50 JPEG the rig encodes itself | `jpeg_synthetic: true`, `jpeg_synthetic_quality: 50` |

A **synthetic JPEG** is used for any local camera that yields no JPEG of its
own — every macOS camera, and a Linux or Windows camera that fell back to a
pixels-only backend. The rig encodes the frame at q50, decodes it straight back,
and scores *that* decode, so the stored and streamed bytes still match what was
scored exactly. It is always on and has no switch.

On macOS the cameras do send MJPEG over USB (see `docs/CAMERAS.md`), but the
operating system decodes it before any application can see the original bytes.
So a Mac streams a second-generation q50 JPEG where Linux and Windows forward
the camera's first-generation one. q50 was validated against the corpus and
live play with no change in any score; it is below the camera's own quality
but loses nothing the scorer uses.

A q85 encode (`MJPEG_TRANSPORT_QUALITY`) is only the fallback for a slot that
holds no JPEG at all, which in practice means the synthetic encode failed.

Passthrough frames are repaired (`jpeg_info.repaired`) and decoded before use,
so a camera frame that will not decode is dropped, as a failed read would be.
Camera-JPEG passthrough is on by default; `OPENDARTS_RAW_JPEG=0` turns it off,
and those slots then fall back to the synthetic q50 JPEG like a Mac.

---

## 3. The local virtual camera (same-machine sharing)

When another program on the **same** machine needs the board — most often
Autodarts running its own detection for comparison — it cannot open the physical
camera OpenDarts already holds. The fix is to give that program a **different**
device: a virtual camera fed from OpenDarts' own capture.

| OS | mechanism | why |
|---|---|---|
| **Windows** | a DirectShow virtual-camera filter (`tools/winvcam`), fed over shared memory by `vcam_publish` | DirectShow takes a device **exclusively**; the second opener is locked out |
| **Linux** | `v4l2loopback` virtual devices, fed by `v4l2_publish` | V4L2 allows multiple *opens* but only one *streamer*; the second gets `EBUSY` |
| **macOS** | nothing — AVFoundation shares the physical camera natively | multiple programs can open the same AVFoundation device at once |

The virtual cameras carry the same JPEG the slot holds (the camera's own, with
passthrough on), so the consuming program sees the picture the camera sent, not
a re-encode. Publishing is downstream of the capture loop and never raises: a
failure to publish degrades to "not published" (logged once) rather than
touching scoring.

Whether they are fed is `publish_virtual_cameras` in the config document; left
unset, it follows `ad_enabled`. See `docs/WINDOWS.md` and `docs/LINUX.md` for
setup.

---

## Summary

- **Other machine:** HTTP MJPEG stream, `?full=1` for the scoreable full-res
  feed.
- **Same machine:** a virtual camera (DirectShow on Windows, v4l2loopback on
  Linux; macOS shares natively and needs none).
- **What `?full=1` sends:** the slot's JPEG, never re-encoded — the camera's own
  on Linux and Windows, a synthetic q50 JPEG on macOS. Either way it is the
  exact bytes this rig scored.
