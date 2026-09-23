#!/bin/bash
# Everything a Linux dart rig needs that only root can arrange. Run with
# sudo -- or let ./run.sh call it for you on a fresh clone, which is how
# most people will meet it.
#
# Two things, neither of which the product can do for itself:
#   1. membership of the `video` group -- /dev/video* is root:video 0660,
#      so without it every camera open fails
#   2. the v4l2loopback kernel module, so other software can share the
#      cameras with opendarts (see below)
#
# There was a third for a few hours on 2026-09-15: a WAV player
# (alsa-utils) plus membership of the `audio` group, so spoken dart calls
# could be heard. Both went when spoken calls moved into the BROWSER --
# see opendarts/live/audio.py. Installing an audio stack on a headless
# server so it could talk to an empty room was always the weakest thing
# this script did, and nothing on a rig opens /dev/snd any more.
#
# Named setup_linux_loopback.sh until 2026-09-15, when run.sh started
# invoking it and the loopback module stopped being the only thing in it.
#
# WHY THIS EXISTS AT ALL. V4L2 lets many processes open a camera but only
# one stream from it, so whichever of the two starts second gets nothing.
# The fix is the same one this project already ships on Windows: opendarts
# owns the real cameras and republishes the frames into virtual ones that
# the other software reads instead. On Windows that virtual camera is a DirectShow
# filter we wrote (tools/winvcam/); on Linux it is v4l2loopback, a kernel
# module someone else wrote and the distribution signs.
#
# WHAT IT REFUSES TO DO. It installs a package only on Debian-family
# systems, where the package name can be derived with certainty. It does
# not enable third-party repositories -- Fedora's v4l2loopback lives in
# RPM Fusion, and silently adding a repo as root is not a thing a setup
# script should do behind your back. On anything it does not recognise it
# prints what is needed and stops, rather than guessing a package name and
# running apt/dnf/pacman against it with root.
#
# exclusive_caps=0, NOT 1. Every guide says 1, and on this rig 1 was
# actively harmful, measured: with a producer attached the nodes returned
# EBUSY to any new open, and with none attached they advertised no capture
# capability at all, so a consumer could neither open nor find them either
# way. With 0 they advertise capture unconditionally and stay probeable.
#
# The usual argument for 1 is that some consumers (Chrome notably) reject
# a device advertising both capture and output. That is a real concern for
# a webcam you also use in a browser, and not one here.
#
# SECURE BOOT decides which package. A DKMS build compiles cleanly and
# then refuses to load with "Key was rejected by service", which does not
# look like a signing problem to anyone who has not hit it before. With
# Secure Boot on, only a distro-signed prebuilt will load.
#
# Safe to re-run: every step checks its own result first.
#
# Usage:
#   sudo ./scripts/setup_linux.sh              # 3 cameras, video10-12
#   sudo ./scripts/setup_linux.sh --devices 2 --start-nr 20
set -uo pipefail

DEVICES=3
START_NR=10
MODPROBE_CONF=/etc/modprobe.d/v4l2loopback.conf
LOAD_CONF=/etc/modules-load.d/v4l2loopback.conf

while [ $# -gt 0 ]; do
  case "$1" in
    --devices)  DEVICES="${2:?--devices needs a number}"; shift 2 ;;
    --start-nr) START_NR="${2:?--start-nr needs a number}"; shift 2 ;;
    -h|--help)  sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mok\033[0m    %s\n' "$*"; }
info() { printf '    ....  %s\n' "$*"; }
die()  { printf '    \033[31mstop\033[0m  %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || die "this is for Linux; $(uname -s) needs none of it"
[ "$(id -u)" = "0" ] || die "run with sudo -- installing a kernel module needs root"

# The user who invoked sudo, not root: group membership below is about them.
REAL_USER="${SUDO_USER:-$(id -un)}"

# sed strips a trailing comma: BSD seq emits one after the last
# value, GNU seq does not, and modprobe rejects "10,11,12,".
VIDEO_NRS=$(seq -s, "$START_NR" $((START_NR + DEVICES - 1)) | sed 's/,$//')
LABELS=$(for i in $(seq 0 $((DEVICES - 1))); do printf 'OpenDarts Cam %d,' "$i"; done | sed 's/,$//')
EXCLUSIVE=$(for _ in $(seq 1 "$DEVICES"); do printf '0,'; done | sed 's/,$//')
# An ARRAY, not a string. card_label values contain spaces ("OpenDarts
# Cam 0"), so an unquoted string expansion splits them into separate
# modprobe arguments and the labels come out mangled -- which is how the
# first version produced devices called: OpenDarts Cam 0"
MODPROBE_ARGS=(
  "devices=$DEVICES"
  "video_nr=$VIDEO_NRS"
  "card_label=$LABELS"
  "exclusive_caps=$EXCLUSIVE"
)
# The modprobe.d form needs the spaces quoted for the same reason.
OPTIONS="devices=$DEVICES video_nr=$VIDEO_NRS card_label=\"$LABELS\" exclusive_caps=$EXCLUSIVE"

printf '\nopendarts -- v4l2loopback setup\n'
printf '%s\n' "$(uname -srm)"
. /etc/os-release 2>/dev/null && printf '%s\n' "${PRETTY_NAME:-unknown distro}"
printf 'creating %d device(s) at /dev/video%s\n' "$DEVICES" "$(echo "$VIDEO_NRS" | tr ',' ' /dev/video')"

# ------------------------------------------------------------ secure boot --
step "Checking Secure Boot"
SECUREBOOT=disabled
if command -v mokutil >/dev/null 2>&1; then
  mokutil --sb-state 2>/dev/null | grep -qi enabled && SECUREBOOT=enabled
else
  f=$(ls /sys/firmware/efi/efivars/SecureBoot-* 2>/dev/null | head -1)
  # Byte 5 of that variable is the flag; no EFI vars at all means BIOS boot.
  [ -n "$f" ] && [ "$(od -An -t u1 "$f" 2>/dev/null | awk '{print $5}')" = "1" ] && SECUREBOOT=enabled
fi
ok "Secure Boot is $SECUREBOOT"

# ---------------------------------------------------------------- install --
step "Installing the module"
if [ -d /sys/module/v4l2loopback ] || modinfo v4l2loopback >/dev/null 2>&1; then
  ok "v4l2loopback is already available on this system"
else
  . /etc/os-release 2>/dev/null
  case "${ID:-} ${ID_LIKE:-}" in
    *ubuntu*|*debian*)
      if [ "$SECUREBOOT" = "enabled" ]; then
        # e.g. 7.0.0-31-generic -> generic. The signed package is per
        # flavour, and installing the wrong one silently installs a module
        # for a kernel that is not running.
        FLAVOUR=$(uname -r | sed 's/^[0-9.]*-[0-9]*-//')
        PKG="linux-modules-v4l2loopback-${FLAVOUR}"
        info "Secure Boot is on, so a DKMS build would not load -- using the signed prebuilt"
      else
        PKG="v4l2loopback-dkms"
      fi
      info "installing $PKG"
      apt-get update -qq || info "apt-get update failed -- trying the install anyway"
      apt-get install -y "$PKG" || die "could not install $PKG -- see the output above"
      ok "installed $PKG"
      ;;
    *fedora*|*rhel*|*centos*)
      die "on Fedora/RHEL v4l2loopback comes from RPM Fusion, a third-party
          repository this script will not enable for you. Enable it, then:
              sudo dnf install v4l2loopback
          and re-run this script."
      ;;
    *arch*)
      die "on Arch install it yourself, then re-run this script:
              sudo pacman -S v4l2loopback-dkms
          (plus the headers matching your kernel)"
      ;;
    *suse*)
      die "on openSUSE install it yourself, then re-run this script:
              sudo zypper install v4l2loopback-kmp-default"
      ;;
    *)
      die "unrecognised distribution (${ID:-unknown}). Install v4l2loopback
          however this system ships it, then re-run. Do NOT use a DKMS build:
          Secure Boot is $SECUREBOOT, and an unsigned module will not load."
      ;;
  esac
fi

# --------------------------------------------------------------- persist ---
step "Making it survive a reboot"
if [ -f "$MODPROBE_CONF" ] && ! grep -qF "$OPTIONS" "$MODPROBE_CONF" 2>/dev/null; then
  cp "$MODPROBE_CONF" "$MODPROBE_CONF.bak"
  info "existing config differed -- kept a copy at $MODPROBE_CONF.bak"
fi
printf 'options v4l2loopback %s\n' "$OPTIONS" > "$MODPROBE_CONF"
ok "wrote $MODPROBE_CONF"
printf 'v4l2loopback\n' > "$LOAD_CONF"
ok "wrote $LOAD_CONF"

# ------------------------------------------------------------------ load ---
step "Loading the module"
if [ -d /sys/module/v4l2loopback ]; then
  # Reloading picks up changed options, but only if nothing is streaming.
  # A busy module is not an error worth stopping for -- the config above is
  # already written, so the next boot gets it regardless.
  if modprobe -r v4l2loopback 2>/dev/null; then
    info "unloaded the previous instance to apply these options"
  else
    info "module is in use (something is streaming from it) -- keeping it"
    info "the new options apply after a reboot, or stop both and re-run"
  fi
fi
if [ ! -d /sys/module/v4l2loopback ]; then
  modprobe v4l2loopback "${MODPROBE_ARGS[@]}" || die "modprobe failed -- see the output above.
          If this says 'Key was rejected by service', the module is unsigned
          and Secure Boot refused it: install the distro's signed build
          instead of a DKMS one."
fi
ok "module loaded"

# ---------------------------------------------------------------- verify ---
step "Verifying"
missing=""
for n in $(echo "$VIDEO_NRS" | tr ',' ' '); do
  if [ -e "/dev/video$n" ]; then ok "/dev/video$n exists"; else missing="$missing $n"; fi
done
[ -n "$missing" ] && die "these did not appear:$missing -- check 'dmesg | tail' for the module's own complaint"

# The group is a separate failure that looks identical from the outside:
# /dev/video* is root:video 0660, so without it every open fails and
# OpenCV reports the unhelpful "can't open camera by index".
step "Checking camera permissions for $REAL_USER"
if id -nG "$REAL_USER" 2>/dev/null | tr ' ' '\n' | grep -qx video; then
  ok "$REAL_USER is in the 'video' group"
else
  usermod -aG video "$REAL_USER" && ok "added $REAL_USER to the 'video' group"
  printf '    \033[33mNOTE\033[0m  log out and back in before starting opendarts.\n'
  printf '          Supplementary groups are fixed when a process starts, so a\n'
  printf '          shell opened before now will keep failing. New login:\n'
  printf '              exec su - %s\n' "$REAL_USER"
fi

cat <<EOF

Done.

Next:
  1. In the app that should share the cameras, select
     $(for n in $(echo "$VIDEO_NRS" | tr ',' ' '); do printf '/dev/video%s ' "$n"; done)
     (they appear as "OpenDarts Cam 0/1/2"). If it was already running,
     restart it before looking.
  2. Leave opendarts pointed at the REAL cameras. Run
     ./scripts/check_linux_cameras.sh to see which numbers those are --
     UVC cameras expose a metadata node beside each capture node, so they
     are usually 0, 2, 4 rather than 0, 1, 2.
  3. The loopback devices carry frames only while opendarts is capturing.
     The consuming app will see a camera that opens and delivers nothing
     if opendarts is stopped.

Full reasoning: docs/LINUX.md
EOF
