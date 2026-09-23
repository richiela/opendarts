#!/bin/bash
# Diagnose Linux camera access for this rig, and say what to do about it.
#
# READ-ONLY. It never installs, never loads a module, never calls sudo. It
# measures this machine and prints the command YOU should run. That
# division is deliberate: everything it asserts is checked here, and the
# only thing that varies between distributions is a string it hands you.
# A script that guessed package names across five package managers -- and
# on Fedora silently enabled a third-party repo -- would be wrong
# eventually, and wrong with root.
#
# It exists because two Linux-only failures are both invisible at the
# point they bite:
#
#   1. Not being in the `video` group. /dev/video* is root:video 0660, so
#      an open() fails, and OpenCV reports it as "can't open camera by
#      index" -- which points nowhere near group membership. Worse, adding
#      yourself to the group does not affect processes that are ALREADY
#      running: supplementary groups are fixed at process creation, so a
#      shell started before the change keeps failing forever.
#
#   2. UVC cameras expose TWO device nodes each -- a capture node and a
#      metadata node, interleaved. Three cameras give /dev/video0..5 where
#      only 0, 2 and 4 can deliver frames. The obvious guess (0, 1, 2) is
#      two cameras and a metadata node.
#
# Usage:  ./scripts/check_linux_cameras.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOOPBACK_DEVICES="10 11 12"          # matches docs/LINUX.md's modprobe line
ok=0; warn=0; fail=0

say()  { printf '%s\n' "$*"; }
good() { printf '  \033[32mOK\033[0m    %s\n' "$*"; ok=$((ok+1)); }
note() { printf '  \033[33mNOTE\033[0m  %s\n' "$*"; warn=$((warn+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
fixit(){ printf '        \033[1m%s\033[0m\n' "$*"; }

if [ "$(uname -s)" != "Linux" ]; then
  say "This script is for Linux. On $(uname -s) the cameras need none of it."
  exit 0
fi

say ""
say "opendarts -- Linux camera check"
say "$(uname -srm)  |  $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || echo "unknown distro")"
say ""

# ---------------------------------------------------------------- group --
say "1. Camera permissions"
if getent group video >/dev/null 2>&1; then
  if id -nG | tr ' ' '\n' | grep -qx video; then
    good "this shell is in the 'video' group"
  elif getent group video | grep -q "\b$(id -un)\b"; then
    # The distinction that cost two hours: on disk but not in this process.
    bad "'$(id -un)' IS in the 'video' group on disk, but THIS SHELL is not"
    say  "        Supplementary groups are fixed when a process starts, so a"
    say  "        shell (and anything it launched, including run.sh) opened"
    say  "        before the change never picks it up. Log out and back in,"
    say  "        or start a fresh login shell:"
    fixit "exec su - $(id -un)"
  else
    bad "'$(id -un)' is not in the 'video' group -- /dev/video* is root:video 0660"
    fixit "sudo usermod -aG video $(id -un)   # then log out and back in"
  fi
else
  note "no 'video' group on this system -- check what owns /dev/video*"
fi

# --------------------------------------------------------------- cameras --
say ""
say "2. Cameras"
shopt -s nullglob
nodes=(/dev/video*)
if [ ${#nodes[@]} -eq 0 ]; then
  bad "no /dev/video* nodes at all -- nothing is plugged in, or no UVC driver"
else
  # QUERYCAP via python3 rather than the sysfs 'index' file: index is a
  # convention that happens to hold for UVC, device_caps is the driver's
  # own answer. v4l2-ctl would also do it but is not installed by default.
  if command -v python3 >/dev/null 2>&1; then
    python3 - "${nodes[@]}" <<'PY'
import ctypes, fcntl, os, sys
VIDIOC_QUERYCAP = 0x80685600
CAPTURE, META = 0x00000001, 0x00800000
class Cap(ctypes.Structure):
    _fields_ = [("driver", ctypes.c_char*16), ("card", ctypes.c_char*32),
                ("bus_info", ctypes.c_char*32), ("version", ctypes.c_uint32),
                ("capabilities", ctypes.c_uint32), ("device_caps", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32*3)]
capture = []
for path in sys.argv[1:]:
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        print("  \033[31mFAIL\033[0m  %s cannot be opened: %s" % (path, exc.strerror))
        continue
    try:
        c = Cap(); fcntl.ioctl(fd, VIDIOC_QUERYCAP, c)
        if c.device_caps & CAPTURE:
            capture.append(path)
            print("  \033[32mOK\033[0m    %-14s capture   (%s)" % (path, c.card.decode().strip()))
        elif c.device_caps & META:
            print("        %-14s metadata  -- cannot deliver frames" % path)
        else:
            print("        %-14s neither capture nor metadata" % path)
    except OSError as exc:
        print("        %-14s QUERYCAP failed: %s" % (path, exc.strerror))
    finally:
        os.close(fd)
if capture:
    idx = [p.replace("/dev/video", "") for p in capture]
    print("")
    print("  Use these device numbers in the dashboard's Config tab:")
    print("        camera_devices = [%s]" % ", ".join(idx))
# Exit status carries the capture-node count back to the shell so the
# summary at the bottom counts these checks too -- they are printed from
# in here and would otherwise never reach the counter.
sys.exit(min(len(capture), 250))
PY
    found=$?
    if [ "$found" -gt 0 ]; then ok=$((ok+found)); else bad "no capture-capable camera nodes"; fi
  else
    note "python3 not found -- falling back to the sysfs index convention"
    for d in /sys/class/video4linux/video*; do
      [ -e "$d/index" ] || continue
      [ "$(cat "$d/index")" = "0" ] && good "/dev/$(basename "$d") looks like a capture node"
    done
  fi
fi

# ----------------------------------------------------------------- audio --
say ""
say "3. Spoken dart calls"
# THIS SECTION GOT MUCH SHORTER on 2026-09-15. It used to check three
# things this rig no longer needs: membership of the `audio` group
# (/dev/snd/* is root:audio 0660), the presence of aplay/paplay/ffplay,
# and whether /proc/asound/cards saw a sound card at all. Spoken calls
# now play in the BROWSER -- see opendarts/live/audio.py -- so a Linux
# rig needs no sound hardware, no sound group and no player. It needs the
# clips, and that is all this can usefully check from here.
#
# What it CANNOT check is the half that now matters: whether the screen
# someone is actually looking at has been allowed to make a noise. That
# lives in a browser on another device entirely, which is exactly why the
# dashboard reports it per screen (Config -> Audio -> "Screens watching
# this rig") instead of a shell script guessing at it.
if [ -d "$REPO_ROOT/assets/voices" ]; then
  sets=$(find "$REPO_ROOT/assets/voices" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
  [ "$sets" -gt 0 ] && good "$sets voice set(s) present in assets/voices/" \
                    || note "no voice sets in assets/voices/ -- the dashboard will have nothing to play"
else
  note "assets/voices/ is missing -- run from a full checkout"
fi
say  "        Sound comes out of the BROWSER now, not this machine."
say  "        Turn it on per screen: Config -> Audio."

# -------------------------------------------------------------- loopback --
say ""
say "4. Virtual cameras (only needed to share the cameras with other software)"
if [ -d /sys/module/v4l2loopback ]; then
  good "the v4l2loopback module is loaded"
  missing=""
  for n in $LOOPBACK_DEVICES; do
    [ -e "/dev/video$n" ] || missing="$missing $n"
  done
  if [ -z "$missing" ]; then
    good "loopback devices present: $(for n in $LOOPBACK_DEVICES; do printf '/dev/video%s ' "$n"; done)"
  else
    note "module loaded but /dev/video$(echo "$missing" | tr -s ' ' ',') missing -- reload with the documented video_nr="
  fi
else
  note "v4l2loopback is NOT loaded -- opendarts and other software cannot share the cameras"
  say  "        V4L2 allows many opens but only one streamer, so whichever"
  say  "        starts second gets nothing. Skip this section if you only"
  say  "        ever run one of them at a time."
  say  ""

  # Secure Boot decides DKMS vs a signed prebuilt, and getting it wrong
  # produces a module that compiles cleanly and then refuses to load with
  # "Key was rejected by service" -- which does not look like a signing
  # problem to anyone who has not seen it before.
  sb="unknown"
  if command -v mokutil >/dev/null 2>&1; then
    mokutil --sb-state 2>/dev/null | grep -qi enabled && sb="enabled" || sb="disabled"
  else
    f=$(ls /sys/firmware/efi/efivars/SecureBoot-* 2>/dev/null | head -1)
    if [ -n "$f" ]; then
      [ "$(od -An -t u1 "$f" 2>/dev/null | awk '{print $5}')" = "1" ] && sb="enabled" || sb="disabled"
    else
      sb="disabled"   # no EFI vars at all means a BIOS boot
    fi
  fi
  say "        Secure Boot: $sb"

  distro=$(. /etc/os-release 2>/dev/null && echo "${ID:-unknown} ${ID_LIKE:-}")
  case "$distro" in
    *ubuntu*|*debian*)
      if [ "$sb" = "enabled" ]; then
        flavour=$(uname -r | sed 's/^[0-9.]*-[0-9]*-//')   # e.g. generic
        say "        Secure Boot will REJECT an unsigned DKMS module, so use"
        say "        the distro's signed prebuilt for your kernel flavour:"
        fixit "sudo apt install linux-modules-v4l2loopback-${flavour:-generic}"
      else
        fixit "sudo apt install v4l2loopback-dkms"
      fi
      ;;
    *fedora*|*rhel*)
      say "        v4l2loopback lives in RPM Fusion, which is a third-party"
      say "        repository this script will not enable for you:"
      fixit "sudo dnf install v4l2loopback   # needs RPM Fusion free enabled"
      ;;
    *arch*)
      fixit "sudo pacman -S v4l2loopback-dkms   # plus headers for your kernel"
      ;;
    *suse*)
      fixit "sudo zypper install v4l2loopback-kmp-default"
      ;;
    *)
      note "unrecognised distribution ($distro) -- no package name to offer"
      say  "        Install v4l2loopback however your distribution ships it,"
      say  "        then see docs/LINUX.md. Do not use a DKMS build if"
      say  "        Secure Boot is $sb."
      ;;
  esac
  say ""
  say "        Then create the devices (same on every distribution):"
  fixit "sudo modprobe v4l2loopback devices=3 video_nr=10,11,12 \\"
  fixit "     card_label=\"OpenDarts Cam 0,OpenDarts Cam 1,OpenDarts Cam 2\" \\"
  fixit "     exclusive_caps=0,0,0"
  say "        exclusive_caps=0, NOT the 1 the guides give. Measured: with 1"
  say "        a node returns EBUSY to any open while something is publishing,"
  say "        and offers no capture capability while idle -- so another"
  say "        program can neither open nor find it either way."
  say "        Easier: sudo ./scripts/setup_linux.sh"
fi

say ""
say "----"
printf '%d ok, %d to look at, %d blocking\n' "$ok" "$warn" "$fail"
say "Full reasoning: docs/LINUX.md"
[ "$fail" -gt 0 ] && exit 1
exit 0
