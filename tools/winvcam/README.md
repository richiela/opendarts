# Virtual cameras (Windows)

`vcam_probe.dll` is a DirectShow filter that registers three video-input
devices, `OpenDarts Probe Cam 0/1/2`, and streams the rig's camera frames
into them from the capture hub through shared memory. Other software on the
same machine opens these instead of the physical cameras. With no capture
hub running, each device shows a moving test pattern.

The product registers and unregisters the filter itself
(`opendarts/live/vcam_register.py`); the steps below are for doing it by
hand.

## Install

From this folder (or wherever you copied the DLL):

```powershell
regsvr32 vcam_probe.dll
```

**Administrator is not required**, and neither is signing — this is
user-mode COM, not a kernel driver. Unelevated, `regsvr32` registers under
`HKCU\Software\Classes` (current user only); elevated, under
`HKLM\Software\Classes` (every account on the machine). DirectShow finds
the filter either way. The DLL's log names the hive it chose:

```powershell
Select-String 'registry hive' $env:TEMP\vcam_probe.log
```

## Check

```powershell
Get-Content $env:TEMP\vcam_probe.log -Tail 20
```

Then look for `OpenDarts Probe Cam 0/1/2` in the consuming app's camera
list. An app that was already running may only list new devices after it
restarts.

## Remove

```powershell
regsvr32 /u vcam_probe.dll
```

That unregisters all three and deletes the CLSID keys from **both**
hives, whichever one the install used — so a registration made elevated
is still cleaned up by an unelevated removal, and the reverse. Nothing
else is left on the machine.

## Build

Cross-compiled from macOS/Linux; no Windows toolchain required.

```bash
brew install mingw-w64      # or: apt install mingw-w64
./build.sh
```

### The committed `vcam_probe.dll`

The built DLL is committed on purpose: the machine that needs it is a
Windows rig with no compiler, and `opendarts/live/vcam_register.py` only
enables the virtual cameras when this exact file exists. A checkout
without it silently has no virtual cameras.

**Verifying or replacing it.** It is built from `vcam_probe.cpp`,
`vcam_probe.def` and `shared_frame.h`, all in this directory. A rebuild is
not byte-identical (mingw embeds timestamps and paths), so compare
behaviour, not hashes: register it and check that `%TEMP%\vcam_probe.log`
names the hive it chose, as described under Install. If you would rather
not run a binary you did not build, build it and overwrite the committed
copy; nothing checks a hash of it.
