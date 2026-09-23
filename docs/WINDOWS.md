# Running on Windows

Copy-paste recipes for a Windows box. Everything here is PowerShell.

---

## 1. Install prerequisites

```powershell
winget install --id Git.Git -e --source winget
winget install --id Python.Python.3.12 -e --source winget
```

On **ARM64 Windows** (for example a UTM/Parallels VM on an Apple Silicon
Mac), install the x64 Python instead — see [ARM64 Windows](#arm64-windows).
You can also skip Python here: if `run.ps1` finds no Python 3.12+, it
offers to install the right build with winget.

**Close and reopen PowerShell.** PATH changes do not apply to the session
that ran the installer — this is the usual "git is not recognized" moment.

```powershell
git --version
py -3.12 --version
```

---

## 2. Clone and run

```powershell
git clone https://github.com/richiela/opendarts.git opendarts
cd opendarts
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

`run.ps1` creates the venv, installs `requirements.txt`, starts the
server, and restarts it whenever it exits — the Windows counterpart of
`run.sh`. **Ctrl-C stops it.**

Dashboard: <http://localhost:8420>

**No scoring without calibration.** `data/` is gitignored and calibration
is specific to camera placement, so a fresh clone gives you a running
server and live previews, not scores. Assign the cameras (section 4),
then press Start: the first Start calibrates, and the sidebar's
**Calibrate** button redoes it after a camera moves.

`-ExecutionPolicy Bypass` as shown is per-invocation: it needs no admin,
persists nothing, and works where the default policy would refuse the
script. To set it for the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### When the script still will not run

Check which policy is actually in force, and in which scope:

```powershell
Get-ExecutionPolicy -List
```

The strongest non-`Undefined` scope wins.

| What you see | What to do |
|---|---|
| Everything `Undefined`, or `Restricted` at `LocalMachine` | `-ExecutionPolicy Bypass` per invocation, as above |
| You run scripts here often | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` — persists, needs no admin |
| `MachinePolicy` or `UserPolicy` is set | Group Policy. Neither command above can override it; that needs a domain admin or a local GPO edit |
| Policy looks fine and one specific file is still refused | Not the policy — see below |

Prefer `RemoteSigned` over `Unrestricted` or a machine-wide `Bypass`:
local scripts run, downloaded ones do not.

**Mark-of-the-web.** A file that came from a zip or a browser download
carries a "downloaded from the internet" marker, and `RemoteSigned`
refuses it whatever the policy says. The error still mentions the
execution policy, so `Set-ExecutionPolicy` looks like the fix and is not.
Clear the marker instead:

```powershell
Unblock-File .\run.ps1
```

A `git clone` never has this problem; a zip downloaded from GitHub does.

### ARM64 Windows

Works, but **install the x64 Python, not the ARM64 one**:

```powershell
winget install -e --id Python.Python.3.12 --architecture x64 --source winget
```

`opencv-python-headless` publishes no `win_arm64` wheel, so a native
ARM64 Python tries to compile OpenCV from source and fails on any machine
without CMake and Visual Studio. pip reports only:

```
note: This error originates from a subprocess, and is likely not a
problem with pip.
```

which names neither OpenCV nor the architecture. ARM64 Windows runs x64
binaries under emulation, so an x64 Python gets every wheel. `run.ps1`
picks x64 automatically when it installs Python itself, but uses
whatever Python it finds first — so if an ARM64 Python is already
installed, remove it or install the x64 one alongside it.

Don't switch to an x86 Windows VM to avoid this: on Apple Silicon that
emulates the whole OS rather than one process, and is far slower.

---

## 3. Update to the latest code

A plain restart runs the code already on disk; `run.ps1` pulls only when
asked to.

- **Dashboard:** Config tab → **Maintenance** → **Update and restart**.
  It confirms, restarts, waits for the rig to answer and reloads the page.
- **PowerShell:**

  ```powershell
  Invoke-RestMethod -Method Post http://localhost:8420/api/restart `
    -ContentType 'application/json' -Body '{"update": true}'
  ```

  Without the body it restarts **without** pulling:

  ```powershell
  Invoke-RestMethod -Method Post http://localhost:8420/api/restart
  ```

A machine that should always follow the branch it has checked out (a
dev VM rather than a rig) can tick **Always update on restart** in the
same Maintenance section, or set `"always_update": true` in
`data\config.json`. It defaults to `false`.

If the pull cannot fast-forward, `run.ps1` says so loudly and starts the
existing code; it does not retry by itself. Capture is never started for
you after a restart — press Start.

On Windows the restart is a hard kill (see
[Known differences](#7-known-differences-from-macoslinux)). Before
pulling, `run.ps1` also moves a locked `vcam_probe.dll` aside, so a
camera app holding it open cannot leave a half-finished pull behind.

---

## 4. Camera assignment

Config tab: each camera's preview card has a device selector. Changing
it saves and applies immediately — the preview switches, and
that is the confirmation.

From the command line:

```powershell
# every setting, including what each slot is using now (under runtime.cameras)
Invoke-RestMethod http://localhost:8420/api/config

# assign slot 0 -> device 1, slot 1 -> device 2, slot 2 -> device 3
$body = @{ camera_devices = @(1,2,3) } | ConvertTo-Json
Invoke-RestMethod -Method Patch -ContentType 'application/json' `
  -Body $body http://localhost:8420/api/config
```

Or edit `data\config.json` directly:

```json
{"camera_devices": [1, 2, 3]}
```

---

## 5. Camera backends

Cameras are opened in this order:

| | how | gives | shares the camera |
|---|---|---|---|
| 1 | our own Media Foundation reader (`MF_JPEG`) | the camera's own JPEG | yes, via Camera Frame Server |
| 2 | OpenCV DirectShow (`CAP_DSHOW`) | pixels only | **no** — exclusive |

OpenCV's own MSMF backend is not used (the `opencv-python-headless`
Windows wheel has none). A slot's device number counts in Media
Foundation's order, which is also what the device selector shows.

### 1. The camera's own JPEG: `MF_JPEG`

`opendarts/live/win_mf_capture.py` asks the camera for its native MJPG
stream with converters disabled, so the slot keeps the exact JPEG bytes
the camera sent (`backend_used: "MF_JPEG"`, `jpeg_passthrough: true` in
`/api/cameras/status`). Those bytes feed the virtual cameras, the
`?full=1` streams, the frame ring and the saved per-throw clips.

A camera with no MJPG type at the configured size, or whose frames won't
decode, falls back to DirectShow with a log line saying why. A slot set
to `auto` resolution skips this reader. `OPENDARTS_RAW_JPEG=0` in the
environment turns it off.

If another app takes the camera, Media Foundation hands it over and our
reads fail with `PREEMPTED`; the slot retries every 2 seconds.

### 2. The fallback: DirectShow, matched by device path

DirectShow numbers devices differently from Media Foundation: it also
lists virtual cameras, ours included. So the fallback opens the
DirectShow device whose **device path** matches the camera, never the
same number — opening by number could read our own virtual camera back
into scoring. No match, no open: the slot fails and the log says why.
DirectShow still asks the camera for MJPG, but hands us decoded pixels.

DirectShow takes the camera exclusively. That only matters if another
program needs the physical camera; other software should read the
virtual cameras instead.

### The virtual cameras

Other software on the same machine should open the virtual cameras
(`OpenDarts Probe Cam 0/1/2`, see `tools/winvcam/README.md`), not the
physical ones. The filter offers MJPG only while OpenDarts is publishing
the camera's JPEG:

| OpenDarts publishes | client asks for | connection | client gets |
|---|---|---|---|
| JPEG | MJPG | MJPG | the camera's bytes; Windows' MJPEG decoder does the rest, as on real hardware |
| JPEG | nothing | RGB24 | a decode by the filter |
| pixels | anything | RGB24 | the pixels, untouched |

`%TEMP%\vcam_probe.log` records what each connection agreed
(`slot 0 connected as MJPG`, or `RGB24 (writer is publishing pixels)`).
The type is fixed when the client connects; if the slot changes path
afterwards, frames are converted until the client reconnects.

**Updating the filter DLL:** Windows locks it while any client has it
loaded. `run.ps1` renames a locked copy aside before pulling; the client
keeps using the old one until it restarts.

### Which backend served us, and how slow

From the repo directory:

```powershell
Select-String -Path data\logs\run_product.log -Pattern 'opened via' | Select-Object -Last 5
```

From anywhere, via the server:

```powershell
(Invoke-WebRequest http://localhost:8420/api/logs/run_product?n=200).Content -split "`n" |
  Select-String 'opened via' | Select-Object -Last 5
```

Each line reads `cam0 (device=1): opened via MF_JPEG in 0.047s` (or
`CAP_DSHOW` for the fallback).

---

## 6. Autodarts on another machine

The Autodarts URL defaults to `http://localhost:3180`. Point elsewhere in
`data\config.json`:

```json
{"ad_base_url": "http://<rig-address>:3180"}
```

The WebSocket URL is derived from it, so this moves both the REST reads
and the event subscription.

---

## 7. Known differences from macOS/Linux

- **`POST /api/restart` is a hard kill.** It sends `SIGTERM`, which
  Windows Python maps to `TerminateProcess`, so the clean-shutdown path
  never runs and the process can die mid-write. `run.ps1` still
  relaunches correctly. Prefer restarting between visits.
- **If Start is slow**, read the `opened via` log lines (above) first —
  they name the backend and how long each open took.
