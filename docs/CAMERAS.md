# What the cameras actually send

Measured at the USB layer: everything here came from the device's own
descriptors and a real captured frame, not from what a capture library chose
to hand over.

## The short version

**Every camera tested sends MJPEG on the wire, macOS included.** The JPEG
quality is a property of the *camera*, not of the platform, and it varies
between models.

| rig | camera | wire format | JPEG quality | frame @1280x720 |
|---|---|---|---|---|
| macOS rig | Sonix `0c45:6340` ×3 | MJPEG (also offers YUY2) | **q85** luma / q82 chroma | 158 KB |
| Linux rig | Realtek `0bda:5844` ×3 | MJPEG (also offers YUY2) | **~q46** | 79 KB |

The Realtek figure was confirmed twice over: 40 real camera frames from the
corpus (q46 on every one) and a separate physical unit probed at the USB layer.
Its fit error is high (~1372), so it uses a **custom** quantisation table and
"q46" means "equivalent to about q46", not a literal encoder setting. The Sonix
fits at 396 and is a cleaner q85.

**vid:pid is a CHIPSET id, not a unit id.** `0bda:5844` is a generic Realtek
controller reused by many OEM modules with different sensors and firmware, and
the quantisation tables live in firmware. Two cameras with the same vid:pid can
encode differently -- compare `bcdDevice` (the firmware revision in each unit's
USB descriptor) and, when it matters, capture a frame. The two Realtek units
tested happened to agree; that was luck, not a rule.

Both models advertise 18 frame sizes in *both* MJPEG and uncompressed YUY2, up
to 1280x720. The 720p mode on the Sonix runs at **33.000033 fps** — not 30.

For scale, that same 720p frame stored losslessly as PNG is **1,166 KB**: the
camera's own JPEG is **7.4× smaller**, and it is what gets scored.

## What macOS actually does

It is easy to conclude that macOS "has no JPEG" and that its cameras give raw
pixels. That is wrong, and the distinction matters:

- **True:** OpenCV's AVFoundation backend is hard-wired to BGRA and never
  surfaces the compressed bytes.
- **False:** that the camera sent pixels. It sent MJPEG; AVFoundation decoded
  it and threw the bytes away.

So a macOS rig never sees pristine sensor data: it gets JPEG-decoded pixels,
the same *kind* of data a Linux rig scores. "The library cannot surface it" is
not "the hardware does not produce it" — see `docs/DESIGN.md` on capability
checks. Windows had the same problem, which is why OpenDarts reads cameras
there with its own Media Foundation reader.

**What OpenDarts does about it.** On macOS every frame is encoded to JPEG by
OpenDarts itself (quality 50), decoded straight back, and *that* decode is what
gets scored and stored. From there on a macOS rig behaves exactly like a
Linux or Windows one: same frame ring, same MJPEG clips, same package shape,
at roughly an eighth of the disk that lossless PNG frames would take. It is
always on for local cameras and has no setting. Re-encoding at q50 moved no
scores in testing (437 recorded darts re-scored at q100 down to q25, then live
darts against Autodarts at q50).

## Why macOS can't pass the camera's JPEG through

The bytes exist on the wire but **cannot be reached from userland without
root**, and root is not an option for the product. Two independent blocks, both
measured directly on these cameras:

1. **You cannot get the USB interface.** Apple's `UVCAssistant` CMIO system
   extension permanently holds *both* UVC interfaces of *every* published
   camera, including idle ones. `IOUSBHostInterface` open fails;
   `.deviceSeize` is a polite *request* the owner ignores; `.deviceCapture`
   needs root or the `com.apple.vm.device-access` entitlement. This is the
   "Access denied" from libusb (`darwin_usb.c:3467`), and TCC has nothing to
   do with it -- the gate is exclusive ownership.

2. **Even with the interface, the bytes are already gone.** `UVCAssistant`
   decodes MJPEG *inside itself* before publishing to CoreMediaIO. The stream
   advertises only `420v`/`yuvs` -- **no `dmb1`** -- and every `420v` format is
   tagged `decompressed_from_format_type = 'dmb1'`, which is the decode saying
   so in its own metadata.

Consequences worth knowing:

- `videoSettings = @{}` (ffmpeg's `capture_raw_data`) returns decoded `420v`
  pixel buffers here. ffmpeg's option exists for FireWire DV "muxed" devices;
  it never claims MJPEG.
- Chromium's code *does* pass JPEG through -- but only `when the device
  publishes dmb1`, which this macOS + UVC-extension combination does not.
  OBS never captures compressed on macOS at all.
- The only sanctioned route is replacing Apple's UVC stack with a DriverKit
  dext plus your own CMIO Camera Extension: a multi-week driver project needing
  Apple-approved entitlements tied to the camera's vendor ID, after which every
  other app (Autodarts included) sees the cameras only through that driver.

### The trap that looks like success

`AVCaptureVideoDataOutput` with `AVVideoCodecKey: .jpeg` **does** return
`ffd8...ffd9` block buffers, around 80 KB. It is **not** the camera's JPEG --
it is a fresh VideoToolbox re-encode of the already-decoded frame, at the
standard ITU T.81 Annex K.1 tables (q50), 4:2:0, with `SpatialQuality=512` in
the format description, and it costs real CPU. The camera's own frame is 158 KB
at q85. Anything checking only "did I get JPEG bytes" ships this believing it
has passthrough, when it is really a re-encode of a decode.

### Root capture is hazardous, not just privileged

`sudo` works because libusb's device-capture path forcibly terminates every
existing client and driver on the device. That is a seize, not a graceful
detach: it leaves the camera **driverless and unusable by the product and by
Autodarts** until it is `IOUSBHostDevice.reset()` or replugged. Measure this
way sparingly, and never on a rig mid-session.

## How it was measured

Formats came from each camera's USB configuration descriptor, which lists
every format and frame size the camera offers and can be read without opening
the device. Quality came from a real captured frame: the JPEG's DQT marker
holds its quantisation tables, and back-solving the standard IJG quality
factor from the luma table gives the number above. The *fit error* of that
back-solve matters: small means a standard libjpeg-scaled table and a
trustworthy number; large means a custom table and an approximate one. The
Sonix fits at 396 (luma) -- close enough to call it q85, not exact.

## Measurement traps

1. **Do not estimate quality from decoded pixels.** Reconstructing the
   quantisation lattice from a decoded frame gave **q94** for a camera that
   really sends **q85**. Measure the bytes, not a decode of them.

2. **macOS camera access needs a GUI session (TCC).** Over ssh, both OpenCV
   and PyAV fail with a bare `Errno 5 Input/output error` — which reads like a
   bad framerate or a busy device and is neither. libuvc and libusb are not
   affected: descriptor reads work fine over ssh.

3. **`uvc_open` returns "Access denied" without root.** macOS's `AppleUSBVideo`
   driver claims the interface and libusb cannot detach it. `sudo` gets past
   it. This is not a TCC problem and no amount of GUI session fixes it.

4. **The framerate must match a supported mode exactly.** These cameras run
   720p at 33.000033 fps; asking for a round 30 fails as `Invalid mode` from
   libuvc and as `Errno 5` from AVFoundation — both of which read like "this
   camera has no 720p MJPEG", which is false. Relatedly, `pixel_format` in
   ffmpeg's avfoundation input takes **ffmpeg pix_fmt names, never FourCCs**:
   passing `dmb1` is `EINVAL` and means nothing about MJPEG support.

## Why this matters

Storing JPEG bytes is **not** a lossiness decision — the stored bytes are
exactly what was decoded and scored. Re-encoding them as PNG would not recover
detail the camera never sent; it would just cost 7.4× the disk. The only reason
a macOS rig encodes its own JPEG instead of keeping the camera's is that the
originals cannot be reached, which is a capture-path limitation and not a
property of the hardware.
