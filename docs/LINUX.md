# Running on Linux

The product needs nothing unusual on Linux, but three things bite on a
first install that bite on neither macOS nor Windows, and each reports
itself as something else entirely.

**You may not need most of this page.** `./run.sh` checks on startup for
the two that need root (sections 1 and 4). If either is missing it says
so, and offers to run the one script that fixes both:

```
[run.sh] This Linux rig is missing things only root can set up:
  * you are not in the 'video' group -- every camera open will fail
  * the v4l2loopback module is not loaded -- other software cannot share the cameras
[run.sh] Run 'sudo ./scripts/setup_linux.sh' now? [Y/n]
```

Say yes and the rig works on its first run, group change included (see
[picking up a group without logging out](#picking-up-a-group-without-logging-out)).
With no terminal to ask at — a `systemd` unit, an unattended restart
loop — it prints the instructions and starts anyway. Answering `n` is
remembered in `data/.linux-setup-declined` (delete it to be asked again);
`OPENDARTS_SKIP_LINUX_SETUP=1` turns the check off.

The rest of this page is what that script does and why, for when it
refuses (an unrecognised distribution) or you would rather do it by hand.
To check without changing anything:

```sh
./scripts/check_linux_cameras.sh
```

It is read-only: it never installs anything and never calls `sudo`.

A Linux rig needs no sound hardware or `audio` group: spoken calls play in
the browser you watch the dashboard from.

---

## 1. The `video` group

`/dev/video*` is owned `root:video` with mode `0660`, so a user outside
the `video` group cannot open a camera at all. OpenCV reports that as:

```
can't open camera by index
backend is generally available but can't be used to capture by index
```

which points nowhere near permissions. Add yourself:

```sh
sudo usermod -aG video "$USER"
```

Supplementary groups are fixed when a process starts and never refresh.
A shell opened before the change keeps failing, and so does anything
launched from it — including a supervisor that restarts the product, so
restarting only the product changes nothing.

### Picking up a group without logging out

`sg` checks the group *database* rather than the calling process, so once
`/etc/group` lists you it starts a command with the group added, with no
password and no logout:

```sh
sg video -c 'id -nG'      # video appears immediately
```

`run.sh` does this to itself: if the group has been granted but the
running process predates it, it re-executes through `sg` and carries on.
That is what lets the run that installs the prerequisites also open the
cameras.

It does not help a process that is already running: anything else that
opens the cameras needs restarting (not a reboot) after a group change.

Check with `id`: `video` must appear. If it appears in `getent group video`
but not in `id`, your shell is stale. A fresh login shell fixes it:

```sh
exec su - "$USER"
```

---

## 2. Which `/dev/video*` are actually cameras

A UVC camera registers **two** device nodes: one for video and one for
metadata. Three cameras give you `/dev/video0` through `/dev/video5`,
where only `0`, `2` and `4` deliver frames. The intuitive
`camera_devices = [0, 1, 2]` is therefore two cameras and a metadata node,
and the metadata node fails to open with the same misleading message as a
permissions problem. (macOS and Windows do not list metadata nodes.)

```sh
./scripts/check_linux_cameras.sh     # prints the right numbers
v4l2-ctl --list-devices              # if v4l-utils is installed
```

Pick them with the device selector on each camera's preview card in the
dashboard's **Config** tab, or set them in `data/config.json`:

```json
{ "camera_devices": [0, 2, 4] }
```

Device numbers follow enumeration order and are **not guaranteed stable
across reboots**. If the cameras come up wrong after a cold boot, re-check
before assuming the config broke. `/dev/v4l/by-path/` holds stable names.

---

## 3. Pixel format

Nothing to configure: the product requests MJPG automatically. It is
documented because a V4L2 driver's default is uncompressed YUYV, and
1280x720 YUYV (about 27 MB/s) is more than USB 2.0 sustains above 10fps.
Measured on one camera, seconds apart:

| requested | negotiated |
|---|---|
| nothing (driver default) | 1280x720 **@10fps** YUYV |
| MJPG | 1280x720 **@30fps** MJPG |

So a rig stuck at 10fps is a format problem, not slow cameras.

Each slot keeps the camera's own JPEG bytes (passthrough —
`jpeg_passthrough` on `/api/cameras/status` says which slots do). The
rig scores the decode of those bytes and saves the bytes themselves in
each throw's clips, with no re-encode. `OPENDARTS_RAW_JPEG=0` in the
environment turns passthrough off.

Some cameras (the Scolia, Sonix 0c45:6340) often leave off the JPEG end
marker. Every frame is repaired — cut at its last end marker, or given
one — and decoded before it is used, so any frame that is kept or
forwarded is one the rig could score.

---

## 4. Sharing the cameras with other software

**Skip this section unless another program (Autodarts, for example) needs
the same cameras while OpenDarts runs.**

V4L2 lets many processes *open* a camera but only one *stream* from it.
So OpenDarts owns the real cameras and republishes the frames into
**virtual** ones that the other program reads instead: on Linux that is
`v4l2loopback`, a kernel module (on Windows, the filter in
`tools/winvcam/`). It has to be a kernel module for any program that opens
`/dev/videoN` directly; userland alternatives such as PipeWire only work
for clients that opt into them.

### Setup

```sh
sudo ./scripts/setup_linux.sh
```

This is the script `run.sh` offers to run. It installs the module,
creates three devices at `/dev/video10,11,12`, makes them load at boot
and adds you to the `video` group. It refuses rather than guesses on
distributions where it cannot be certain of the package name.

By hand:

```sh
# Debian/Ubuntu, Secure Boot OFF
sudo apt install v4l2loopback-dkms
# Debian/Ubuntu, Secure Boot ON  -- see below
sudo apt install linux-modules-v4l2loopback-generic

sudo modprobe v4l2loopback devices=3 video_nr=10,11,12 \
     card_label="OpenDarts Cam 0,OpenDarts Cam 1,OpenDarts Cam 2" \
     exclusive_caps=0,0,0
```

**Secure Boot rejects DKMS builds.** The module compiles cleanly, then
fails to load with `Key was rejected by service`, which does not read as a
signing problem. With Secure Boot on (`mokutil --sb-state`) you need the
distribution's signed prebuilt — on Ubuntu
`linux-modules-v4l2loopback-<flavour>`, the flavour being the tail of
`uname -r`.

**`exclusive_caps=0`, not the `1` most guides recommend.** With `1`, the
devices are unusable here in both directions: with a producer attached a
new open gets `EBUSY`, and with none attached they advertise no capture
capability, so a consumer can neither find nor stream from them. (`1`
exists for browsers that reject a device advertising both capture and
output — not a concern for this use.)

**`card_label` takes one comma-separated list**, not a quoted string per
label. Quoting each one leaks the quote marks into the names, which then
show up in camera pickers as `OpenDarts Cam 0"`.

### Pointing the other program at them

The devices appear in camera lists as `OpenDarts Cam 0/1/2`; select them
in the other program like any camera. Leave OpenDarts on the *real*
cameras (`0, 2, 4`). It publishes to the loopbacks automatically whenever
the Autodarts comparison is enabled, or when `publish_virtual_cameras` is
`true` in `data/config.json`.

The devices exist whether or not publishing is on; the setting controls
publishing, not the devices.

### Start OpenDarts first

**The pixel format is fixed when a device is first opened, by either
side**, and v4l2loopback does not report a mismatch: if anything already
has the device open, a request for a different format keeps the existing
one and still returns success. A consumer that opened in one format and is
then fed another typically shows black while frames keep arriving.

So **start OpenDarts publishing first, then (re)start the other program.**
Restarting OpenDarts alone does not help; the other program is the one
that has to re-open. OpenDarts refuses to publish into a format it did not
get, so on its side the failure is loud.

### Formats

MJPEG by default; set `"v4l2_format": "BGR24"` in `data/config.json` for
raw frames. In MJPEG, a passthrough slot publishes the camera's own bytes
unchanged; a slot without passthrough is encoded at quality 85.
