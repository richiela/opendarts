#!/bin/bash
# opendarts run_product launcher.
#
# Meant to run under its own restart loop (a login item, a screen/tmux
# session, or just `nohup ./run.sh &`) -- every time the process exits
# for any reason, this script starts it again.
# This replaces a manual `while true; do
# .venv/bin/python3 -m opendarts.live.run_product; done` terminal loop with
# one committed, self-contained script.
#
# IT DOES NOT PULL UNLESS ASKED (changed 2026-09-17 -- it used to pull on
# every relaunch, including after a crash). Deploying is now "push, then
# ask for an update": the dashboard's Config tab -> Maintenance -> Update
# and restart, or POST /api/restart {"update": true}, or the standing
# `always_update` key for a machine that is supposed to follow main. See
# pull_if_requested() at the bottom of this file, and docs/DEPLOYMENT.md.
#
# docs/DESIGN.md's own standing restart procedure (kill the PID, this loop
# relaunches -- never `nohup`/launch a fresh process over SSH yourself) is
# unchanged by this script; it's what makes that procedure possible in the
# first place. What changed is only WHICH CODE comes back: the same
# commit, unless an update was asked for.
set -uo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------- linux prerequisites --
# TWO THINGS ON LINUX BELONG TO ROOT, and a fresh clone has neither:
# membership of the `video` group (/dev/video* is root:video 0660) and
# the v4l2loopback kernel module.
#
# There were three until later the same day (2026-09-15): a WAV player
# for spoken calls was the third. It went when spoken calls moved into
# the browser -- the rig plays nothing at all now, so there is nothing
# for a player to be missing FOR. That item was always the odd one out
# here: the other two stop cameras working, while a silent server in an
# empty room was a problem nobody could hear.
#
# Until 2026-09-15 the answer was a line in docs/LINUX.md telling you to
# go and run a second script with sudo first. That is a bad first run: it
# only helps someone who read the docs BEFORE the failure, and the
# failure itself -- OpenCV's "can't open camera by index" -- names none of
# the causes. Measured on the rig, all of them bit us in one morning.
#
# So this asks, here, once. It probes first and says nothing at all when
# there is nothing to do, prints exactly what it wants to change, and
# takes a yes/no before running ONE command with sudo. It never runs
# without a terminal to ask at -- a systemd unit or the restart loop
# below gets the instructions printed instead of hanging on an
# unanswerable password prompt -- and answering `n` is remembered, so a
# rig that does not want any of this is not re-asked on every boot.
#
#   OPENDARTS_SKIP_LINUX_SETUP=1 ./run.sh   skips the whole section
#   rm data/.linux-setup-declined           asks again after a `n`
SELF="$PWD/$(basename "$0")"
DECLINED_MARKER="data/.linux-setup-declined"

# STDIN ONLY, deliberately. `read -p` writes its prompt to stderr and
# reads the answer from stdin, and sudo asks for a password on /dev/tty,
# so stdin is the only stream that decides whether anyone can answer.
# Testing stdout as well looks more careful and is wrong: it turns the
# very common `./run.sh 2>&1 | tee run.log` into a silent skip.
have_tty() { [ -t 0 ]; }

# Two DIFFERENT questions, and the gap between them is the whole reason
# `sg` below exists. With a user operand `id` reads the group database;
# without one it reports the calling process's own credentials. A process
# started before `usermod -aG` ran sees the old set forever -- supplementary
# groups are fixed at process creation and there is no API to add one.
group_granted() { id -nG "$(id -un)" 2>/dev/null | tr ' ' '\n' | grep -qx "$1"; }
group_active()  { id -nG            2>/dev/null | tr ' ' '\n' | grep -qx "$1"; }

# The other prerequisite, as a named probe for the same reason: one
# definition, and something the suite can stub. v4l2loopback reports
# itself under /sys/module whether or not a device is currently open --
# see opendarts/live/v4l2_register.py, which checks the same marker.
loopback_loaded()    { [ -d /sys/module/v4l2loopback ]; }

linux_first_run() {
  [ "$(uname -s)" = "Linux" ] || return 0
  [ -n "${OPENDARTS_SKIP_LINUX_SETUP:-}" ] && return 0
  [ -f "$DECLINED_MARKER" ] && return 0
  [ -x scripts/setup_linux.sh ] || return 0

  local missing=()
  group_granted video    || missing+=("  * you are not in the 'video' group -- every camera open will fail")
  loopback_loaded        || missing+=("  * the v4l2loopback module is not loaded -- other software cannot share the cameras")
  [ ${#missing[@]} -eq 0 ] && return 0

  echo
  echo "[run.sh] This Linux rig is missing things only root can set up:"
  printf '%s\n' "${missing[@]}"
  echo "[run.sh] scripts/setup_linux.sh fixes these. It adds you to the"
  echo "[run.sh] 'video' group, installs the v4l2loopback kernel module,"
  echo "[run.sh] and makes the module load at boot."

  if ! have_tty || ! command -v sudo >/dev/null 2>&1; then
    # No one to ask. Say what to run and carry on -- a rig with no camera
    # access still serves its dashboard, which is where the operator will
    # read this same complaint.
    echo "[run.sh] No terminal to ask at, so nothing was changed. Run this yourself:"
    echo "[run.sh]     sudo ./scripts/setup_linux.sh"
    echo
    return 0
  fi

  local reply=""
  read -r -p "[run.sh] Run 'sudo ./scripts/setup_linux.sh' now? [Y/n] " reply
  case "$reply" in
    [Nn]*)
      mkdir -p "$(dirname "$DECLINED_MARKER")" 2>/dev/null
      printf 'Declined on %s. Delete this file to be asked again.\n' "$(date)" \
        > "$DECLINED_MARKER" 2>/dev/null
      echo "[run.sh] Skipped, and remembered. Run it yourself any time:"
      echo "[run.sh]     sudo ./scripts/setup_linux.sh"
      echo "[run.sh] (or: rm $DECLINED_MARKER, to be asked again)"
      ;;
    *)
      # Not fatal on failure: the script refuses to guess a package name
      # on distributions it does not recognise, and that is a reason to
      # start anyway with a clear message, not to refuse to run.
      sudo ./scripts/setup_linux.sh || \
        echo "[run.sh] setup did not finish -- starting anyway, see the output above" >&2
      ;;
  esac
  echo
}

# WHY NOT "LOG OUT AND BACK IN". `usermod -aG video` above changes the
# group DATABASE, and this already-running script keeps the groups it was
# created with, so the very run that installs everything would still fail
# to open a camera -- the single most confusing outcome available here.
#
# `sg` is the way out: it is setgid-root and checks /etc/group, so for a
# user the database already lists it starts a shell with that group added
# and asks for no password. Re-exec through it and the loop below has the
# groups it needs, on the same run, with no logout.
#
# One group per re-exec. There is only `video` to pick up now -- `audio`
# was in this list until spoken calls moved into the browser and /dev/snd
# stopped being something this product ever opens -- but the loop stays a
# loop rather than being unrolled to one name: the next device class this
# rig needs (a serial board, a second capture device) arrives as another
# group, not as a rewrite. The depth guard is a backstop against a system
# where sg silently fails to add the group -- without it, that is an exec
# loop.
regroup_if_needed() {
  [ "$(uname -s)" = "Linux" ] || return 0
  if ! command -v sg >/dev/null 2>&1; then
    # Without sg there is no way to pick the group up in this process, so
    # the old advice is the only advice -- but it is still better than
    # starting silently and failing at the first camera open.
    if group_granted video && ! group_active video; then
      echo "[run.sh] WARNING: you are in the 'video' group but this session is not --" >&2
      echo "[run.sh] cameras will fail until you start a new login:  exec su - $(id -un)" >&2
    fi
    return 0
  fi
  local depth="${OPENDARTS_REGROUP_DEPTH:-0}"
  [ "$depth" -ge 4 ] && return 0
  local want
  for want in video; do
    if group_granted "$want" && ! group_active "$want"; then
      echo "[run.sh] '$want' was granted but this process predates it -- re-entering via sg (no logout needed)"
      export OPENDARTS_REGROUP_DEPTH=$((depth + 1))
      exec sg "$want" -c "$(printf '%q ' "$SELF" "$@")"
    fi
  done
}

linux_first_run
regroup_if_needed "$@"

# WHICH PYTHON. A bare `python3` is whatever the login PATH happens to
# resolve, and on macOS that is /usr/bin/python3 -- the system 3.9. A macOS
# rig silently ran on 3.9.6 for months this way while a Homebrew 3.12 sat
# installed but off the non-interactive shell's PATH, which is exactly
# the sort of thing nobody notices until a dependency stops supporting
# the old one. Prefer a modern interpreter explicitly, newest first, and
# fall back to `python3` so a machine with none of them still starts.
# Both Homebrew prefixes are listed by absolute path: /opt/homebrew on
# Apple Silicon, /usr/local on Intel. Neither is on a non-login shell's
# PATH by default, so `command -v python3.12` alone misses a Homebrew
# install that is sitting right there.
#
# 3.12 IS THE FLOOR. pyproject.toml declares requires-python >= 3.12,
# because 3.12 is what the suite actually runs on. This function used to
# return the first interpreter that EXISTED, so a Mac with only Apple's
# 3.9.6 got a 3.9 venv -- which then could not install the pinned-latest
# numpy and died on `import numpy`. Measured on a fresh macOS VM,
# 2026-09-16. Now a candidate has to meet the floor to be chosen, and the
# old fallback -- "use python3 whatever it is" -- is gone: prints nothing
# and fails, so the caller can say what is missing instead of building a
# venv that cannot work.
PYTHON_MIN="3.12"
py_ok() {
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1
}
pick_python() {
  for candidate in \
      python3.14 python3.13 python3.12 \
      /opt/homebrew/opt/python@3.14/bin/python3.14 \
      /opt/homebrew/opt/python@3.13/bin/python3.13 \
      /opt/homebrew/opt/python@3.12/bin/python3.12 \
      /opt/homebrew/bin/python3 \
      /usr/local/opt/python@3.14/bin/python3.14 \
      /usr/local/opt/python@3.13/bin/python3.13 \
      /usr/local/opt/python@3.12/bin/python3.12 \
      python3; do
    if command -v "$candidate" >/dev/null 2>&1 && py_ok "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# Tested for the interpreter, NOT for the directory. `python3 -m venv` on
# Debian/Ubuntu without the python3-venv package CREATES .venv and then
# fails at ensurepip, leaving a directory with no bin/pip in it. Guarding
# on `[ -d .venv ]` treats that wreckage as a finished venv, so it is
# never repaired: the loop below then runs the system python3, which has
# no numpy, and reports `ModuleNotFoundError: numpy` every 3 seconds
# forever -- three layers away from "a package is missing on this box".
if [ ! -x .venv/bin/python3 ] || ! py_ok .venv/bin/python3; then
  if [ -x .venv/bin/python3 ]; then
    # A venv is pinned to the interpreter that built it, so one built on an
    # old Python stays old forever unless something rebuilds it. Every rig
    # in service was on 3.12+ when this landed, so this only ever touches a
    # venv that could not have worked.
    echo "[run.sh] .venv was built on $(.venv/bin/python3 --version 2>&1); this project needs $PYTHON_MIN+ -- rebuilding it"
    rm -rf .venv
  elif [ -d .venv ]; then
    echo "[run.sh] .venv exists but has no interpreter -- removing the partial one"
    rm -rf .venv
  fi
  PY_BIN="$(pick_python)" || PY_BIN=""
  # NOTHING >= 3.12 FOUND: offer to install one where that can be done
  # safely, and otherwise stop with instructions. Never fall back to an
  # older interpreter -- that is how a Mac got a 3.9 venv that could not
  # import numpy (fresh macOS VM, 2026-09-16).
  #
  # macOS: Homebrew, found by its install paths rather than PATH, because
  # a non-login shell's PATH omits /opt/homebrew/bin -- which is exactly
  # why that Mac saw only Apple's /usr/bin/python3 (3.9.6) while Homebrew
  # sat installed. pick_python checks the keg paths, so a re-pick finds the
  # new interpreter with no PATH changes. Asked, never assumed, and only
  # with a terminal -- same posture as the Linux and Windows prerequisites.
  if [ -z "$PY_BIN" ] && [ "$(uname -s)" = "Darwin" ]; then
    BREW=""
    for candidate in brew /opt/homebrew/bin/brew /usr/local/bin/brew; do
      if command -v "$candidate" >/dev/null 2>&1; then BREW="$(command -v "$candidate")"; break; fi
    done
    echo "[run.sh] No Python $PYTHON_MIN+ found (macOS's built-in python3 is $(python3 --version 2>&1 | cut -d' ' -f2))."
    if [ -n "$BREW" ] && have_tty; then
      reply=""
      read -r -p "[run.sh] Install Python 3.12 with Homebrew now? [Y/n] " reply
      case "$reply" in
        [Nn]*) ;;
        *)
          if "$BREW" install python@3.12; then
            PY_BIN="$(pick_python)" || PY_BIN=""
          else
            # Checked: a failed install must not be narrated as a success.
            echo "[run.sh] brew install failed -- see its output above." >&2
          fi
          ;;
      esac
    fi
  fi

  if [ -z "$PY_BIN" ]; then
    echo "[run.sh] ERROR: this project needs Python $PYTHON_MIN or newer, and none was found." >&2
    case "$(uname -s)" in
      Darwin)
        echo "[run.sh] Install it with Homebrew (https://brew.sh):  brew install python@3.12" >&2
        echo "[run.sh] or from https://www.python.org/downloads/macos/" >&2 ;;
      Linux)
        echo "[run.sh] Install python3.12 (or newer) with your distribution's package manager," >&2
        echo "[run.sh] or use https://github.com/astral-sh/uv (uv python install 3.12)." >&2 ;;
    esac
    echo "[run.sh] Then run ./run.sh again." >&2
    exit 1
  fi
  echo "[run.sh] no .venv found, creating one with $("$PY_BIN" --version 2>&1) ($PY_BIN)..."
  # Checked, and fatal. This used to run unchecked, which is what let a
  # failure here surface as a missing third-party module much later.
  if ! "$PY_BIN" -m venv .venv || [ ! -x .venv/bin/python3 ]; then
    rm -rf .venv
    # THE ACTUAL FIRST FAILURE of a fresh clone on Ubuntu Server, measured
    # 2026-09-15: Debian splits `venv` out of the python3 package, so the
    # stdlib module a script is entitled to assume is simply absent. Same
    # reasoning as the section at the top -- print the command and you
    # have helped only the person who reads stderr; offer to run it and
    # the clone works on the first try.
    if [ "$(uname -s)" = "Linux" ] && have_tty && command -v sudo >/dev/null 2>&1 \
       && command -v apt-get >/dev/null 2>&1; then
      echo "[run.sh] Creating a virtualenv failed. On Debian/Ubuntu that is"
      echo "[run.sh] almost always the python3-venv package being missing."
      reply=""
      read -r -p "[run.sh] Run 'sudo apt-get install -y python3-venv' now? [Y/n] " reply
      case "$reply" in
        [Nn]*) ;;
        *)
          sudo apt-get install -y python3-venv || true
          # Re-picked: installing python3-venv can also be what finally
          # makes a newer interpreter usable, so do not reuse the old choice.
          # Keep the current choice unless the re-pick finds something:
          # `X="$(f)" || true` would still assign empty on failure.
          NEW_PY="$(pick_python)" && PY_BIN="$NEW_PY"
          "$PY_BIN" -m venv .venv || rm -rf .venv
          ;;
      esac
    fi
  fi
  if [ ! -x .venv/bin/python3 ]; then
    rm -rf .venv
    echo "[run.sh] ERROR: could not create a virtualenv with $PY_BIN." >&2
    echo "[run.sh] On Debian/Ubuntu the venv module ships separately:" >&2
    echo "[run.sh]     sudo apt install python3-venv" >&2
    echo "[run.sh] Then run ./run.sh again." >&2
    exit 1
  fi
fi

# Same reasoning one level down: a venv can exist with a working
# interpreter and no pip (python3 -m venv --without-pip, or an ensurepip
# that failed after the interpreter was linked).
#
# Probed with `python3 -m pip`, NOT by testing for .venv/bin/pip. Measured
# 2026-09-15: `python3 -m ensurepip` installs pip perfectly well and
# creates `pip3` and `pip3.9` WITHOUT a bare `pip` script. Testing for the
# script therefore condemns a working venv, and the install line below
# used to invoke that same missing name.
if ! ./.venv/bin/python3 -m pip --version >/dev/null 2>&1; then
  echo "[run.sh] .venv has no pip -- bootstrapping it with ensurepip..."
  ./.venv/bin/python3 -m ensurepip --upgrade >/dev/null 2>&1 || true
  if ! ./.venv/bin/python3 -m pip --version >/dev/null 2>&1; then
    echo "[run.sh] ERROR: .venv has no pip and ensurepip could not supply it." >&2
    echo "[run.sh] On Debian/Ubuntu: sudo apt install python3-venv" >&2
    echo "[run.sh] Then: rm -rf .venv && ./run.sh" >&2
    exit 1
  fi
fi

# ------------------------------------------------------- update policy --
# THIS LOOP USED TO PULL ON EVERY SINGLE RELAUNCH, and that is the bug
# this function exists to fix (2026-09-17). The relaunch is triggered by
# the process EXITING -- which includes crashing -- so a rig that fell
# over mid-match came back on whatever happened to be on main at that
# second, possibly a commit pushed minutes earlier by someone who did not
# know a match was running. Updating was a thing that happened TO the
# operator.
#
# Now it pulls only when the config file says to, via two keys in
# data/config.json (both default false, both documented in
# config.example.json):
#
#   always_update           -- this machine follows main; pull every time.
#                              The old behaviour, as an opt-in, for the
#                              dev VMs that want it.
#   update_on_next_restart  -- a ONE-SHOT request, set by the dashboard's
#                              "Update and restart" button (POST
#                              /api/restart {"update": true}).
#
# THE FLAGS ARE READ BY THE VENV PYTHON, NOT BY THIS SCRIPT. They live in
# a JSON file, and grep/sed against hand-editable JSON is a second parser
# that will eventually disagree with opendarts.live.config about the same
# file -- on the one file that decides which code this rig runs. See
# opendarts/live/update_policy.py, which also CLEARS the one-shot flag as
# it reads it, so a failed pull (or a crash during one) can never leave a
# rig pulling on every restart again.
pull_if_requested() {
  local branch="$1" decision=""
  decision="$(./.venv/bin/python3 -m opendarts.live.update_policy 2>/dev/null)"
  if [ -z "$decision" ]; then
    # Told apart from "skip" on purpose: a rig behaving as configured and
    # a rig whose configuration could not be read are different problems,
    # and only one of them is worth shouting about.
    echo "[run.sh] WARNING: could not read the update flags out of data/config.json." >&2
    echo "[run.sh] NOT pulling -- starting the code that is already here." >&2
    return 0
  fi
  case "$decision" in
    pull*) ;;
    *)
      echo "[run.sh] not pulling: always_update and update_on_next_restart are both off in data/config.json."
      echo "[run.sh] To update: the dashboard's Config tab -> Maintenance -> Update and restart."
      return 0
      ;;
  esac

  echo "[run.sh] $(date): pulling latest ($branch) -- $decision"
  if ! git pull --ff-only origin "$branch"; then
    # LOUD, AND IT DOES NOT RETRY. The one-shot flag was already cleared
    # above, so the next relaunch will not try again -- which is
    # deliberate: a rig whose pull cannot fast-forward (local changes, a
    # rewritten branch) would otherwise spend every restart failing at
    # the same thing, and the product would still be starting on the old
    # code each time anyway. Same "run the existing code as-is" outcome
    # as before this function existed; only the volume changed.
    echo "[run.sh] ERROR: git pull failed or was not a fast-forward." >&2
    echo "[run.sh] STARTING THE EXISTING CODE AS-IS. The update request has already" >&2
    echo "[run.sh] been cleared, so this will NOT retry by itself -- ask for it again" >&2
    echo "[run.sh] once the repository can fast-forward." >&2
  fi
}

while true; do
  # Follows whatever branch is checked out (normally main), so a rig can
  # run a feature branch by checking it out once.
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo main)"
  pull_if_requested "$branch"

  # Warned, not fatal: a transient network failure should not stop a rig
  # that already has its dependencies from coming back up. Silence here
  # was the problem -- an unreported failure presents later as a missing
  # module, which reads as a code bug rather than a failed install.
  if ! ./.venv/bin/python3 -m pip install -q -r requirements.txt; then
    echo "[run.sh] WARNING: pip install failed -- starting with whatever is already installed" >&2
  fi

  # Optional per-rig environment (untracked): KEY=VALUE lines. (The old
  # OPENDARTS_LIFECYCLE_MODE switch is gone -- the lifecycle is the only
  # trigger; the variable is ignored if still present.)
  if [ -f data/run.env ]; then
    echo "[run.sh] loading data/run.env"
    set -a; . ./data/run.env; set +a
  fi

  echo "[run.sh] $(date): starting opendarts run_product (port from data/config.json, default 8420)..."
  ./.venv/bin/python3 -m opendarts.live.run_product "$@"

  echo "[run.sh] $(date): process exited, restarting in 3s..."
  sleep 3
done
