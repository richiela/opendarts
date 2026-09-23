# OpenDarts Changelog

Releases are listed oldest first. **Unreleased**, at the bottom, lists changes
made since the latest release.

---

## v0.81.1 — OpenDarts.v0.81.1 (Sep 22, 2026)

- First public release. Camera-based automatic scoring for steel-tip darts: three commodity USB cameras watch the board, and when a dart lands the rig freezes a frame set and resolves a single `(sector, ring)` call. Four independent engines score every throw from different evidence — Apollo (3D from the dart tip), Talos (3D from the shaft, corrected to the sisal rather than the wire), Athena (2D, three separate per-camera calls combined by confidence), and Ares (2D, where the shaft lines cross) — with Zeus combining them by majority and Apollo holding the tie-break.
- **Replay packages.** Every scored throw can be saved with the exact inputs it was scored from — the empty-board frame, the scored frame and the calibration in force at capture — so the same throw re-scores through newer code and can produce a different, better answer. That is what makes an accuracy claim checkable rather than asserted.
- **One package shape.** A package is a few JSON files plus one MKV per camera holding the JPEG frames that were actually scored, with no re-encode. A throw not recorded as video keeps a two-frame clip — empty board, then scored frame — written the moment it is saved. With video on, a clip runs from the empty-board frame to one frame past the scored frame, both ends byte-identical to what the live scorer used and verified at write time; it replaces the two-frame clip only once every camera's clip has verified. `video_record_mode` chooses `all`, `mismatch` or `never`.
- **JPEG on every platform.** On Linux and Windows the cameras' own MJPEG bytes are stored as sent. macOS decodes MJPEG before any application can reach the bytes, so there OpenDarts encodes each frame to JPEG itself and scores the decode of those exact bytes — what is scored and what is stored match everywhere. Across 437 recorded darts from two camera models, a JPEG round trip anywhere from q100 down to q25 changed no score. `docs/CAMERAS.md` records what each camera actually sends.
- **No system ffmpeg.** Video is written through the PyAV wheel, which bundles the libraries; nothing shells out to an ffmpeg binary. A startup probe reports what external tools the rig actually has.
- **Autodarts as an optional oracle.** Each call can be compared against Autodarts to measure accuracy, with a whole-ring frame dump written when the two disagree. The Config tab shows the connection state live, in every open dashboard.
- **Throw viewer.** Any saved throw opens with when it was thrown, the package size and whether it holds a recorded clip, the visit's three darts beside the score — click one to step through the turn — plus the stills and a frame-by-frame clip scrubber.
- **Built for a screen nobody touches.** `?sound=on` in the dashboard URL turns spoken calls on for that screen. Open dashboards reload themselves when an update changes the page, and changes such as deleting recorded data reach every open screen.
- **Camera republishing.** OpenDarts hands its cameras on to other software on the same machine, so another scorer such as Autodarts can run beside it on the same board: Windows gets DirectShow virtual cameras and Linux v4l2loopback devices, and macOS needs neither because it shares cameras natively. `docs/STREAMING.md` has the details.
- **Network camera stream.** Every platform serves an HTTP MJPEG stream per camera, so other machines can watch the same board.
- Runs on Linux, macOS and Windows from a checkout, not a `pip install`. Dashboard and HTTP/WebSocket API on port 8420.
- AGPL-3.0-or-later. Issues welcome; pull requests cannot be merged yet — see CONTRIBUTING.md for why.

## Unreleased — since OpenDarts.v0.81.1
