"""The shell half of the product, pinned.

run.sh and scripts/*.sh are not covered by anything else in this suite,
and they are the first code a fresh clone runs -- a syntax error or a
regressed branch in them is a rig that will not start, discovered by a
person standing at a dartboard rather than by CI.

WHAT IS WORTH PINNING HERE is the behaviour that is hard to try by hand,
because trying it means breaking a working machine: what run.sh does on a
Linux rig that is missing its root-owned prerequisites. Specifically that
it never blocks a start -- a rig with no camera access still serves its
dashboard, which is where the operator reads the same complaint -- and
that it never runs sudo without a terminal to ask at, which would hang a
systemd unit on an unanswerable password prompt.

The functions are sourced out of run.sh rather than duplicated, so this
tests the shipped code. `uname` is stubbed to Linux so the Linux branches
run on a developer's Mac.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SH = REPO_ROOT / "run.sh"

#: Everything above this line in run.sh is definitions; the line itself is
#: the first thing executed. Extracting to here gives a source-able file
#: with no side effects -- and pins that ordering, which matters: the
#: group re-exec has to happen before any camera is opened.
FIRST_STATEMENT = "linux_first_run"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="needs bash"
)


def shell_scripts() -> "list[Path]":
    return sorted([RUN_SH, *(REPO_ROOT / "scripts").glob("*.sh")])


@pytest.mark.parametrize("script", shell_scripts(), ids=lambda p: p.name)
def test_parses(script: Path) -> None:
    """Every shipped script is syntactically valid bash.

    Cheap, and it catches the one failure mode that is guaranteed fatal
    and easy to introduce -- an unbalanced quote in a heredoc or a `case`
    arm reads fine and does not run.
    """
    done = subprocess.run(["bash", "-n", str(script)],
                          capture_output=True, text=True)
    assert done.returncode == 0, f"{script.name}: {done.stderr}"


def functions_from_run_sh(*names: str) -> str:
    """Just the named function definitions from run.sh, nothing executed.

    `prelude()` stops before the first top-level call, which is fine for
    the Linux section but the Python helpers live further down, past
    top-level code that must not run in a test.
    """
    text = RUN_SH.read_text()
    out = []
    for name in names:
        start = text.index(f"\n{name}() {{") + 1
        end = text.index("\n}\n", start) + 3
        out.append(text[start:end])
    return "\n".join(out)


def prelude() -> str:
    """run.sh's function definitions, with nothing executed."""
    lines = RUN_SH.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == FIRST_STATEMENT:
            return "\n".join(lines[:i])
    raise AssertionError(
        f"run.sh no longer calls {FIRST_STATEMENT!r} at the start of a line -- "
        "if that section moved, move this test with it"
    )


def run_snippet(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    """Source run.sh's functions in a sandbox and run `body` against them.

    The sandbox gets a fake setup script and a `sudo` that records rather
    than escalates, so a test that wrongly takes the sudo branch fails
    visibly instead of prompting.
    """
    (tmp_path / "scripts").mkdir()
    setup = tmp_path / "scripts" / "setup_linux.sh"
    setup.write_text("#!/bin/bash\nexit 0\n")
    setup.chmod(0o755)
    (tmp_path / "data").mkdir()

    # A REAL FILE, not a shell function: regroup_if_needed uses `exec sg`,
    # and exec resolves an external program and never a function. A stub
    # defined in the harness would silently never be reached -- which is
    # how the first version of this test passed for the wrong reason.
    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    fake_sg = bin_dir / "sg"
    fake_sg.write_text(
        '#!/bin/bash\n'
        'echo "SG-GROUP=$1"\n'
        'exec bash -c "$3"\n'
    )
    fake_sg.chmod(0o755)

    script = tmp_path / "harness.sh"
    script.write_text(
        "cd \"$(dirname \"$0\")\"\n"
        # Stubbed BEFORE sourcing: the definitions capture nothing, but the
        # functions call `uname` at call time, so this reaches them.
        "uname() { [ \"${1:-}\" = -s ] && echo Linux || echo Linux; }\n"
        "sudo() { echo \"SUDO-CALLED: $*\"; return 0; }\n"
        + prelude() + "\n" + body + "\n"
    )
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ.get('PATH', '')}")
    env.pop("OPENDARTS_REGROUP_DEPTH", None)
    env.pop("OPENDARTS_SKIP_LINUX_SETUP", None)
    return subprocess.run(["bash", str(script)], capture_output=True,
                          text=True, cwd=tmp_path, env=env)


def test_silent_when_nothing_is_missing(tmp_path: Path) -> None:
    """A healthy rig sees no output at all.

    The section has to be invisible in the normal case or it becomes noise
    every operator learns to scroll past -- including on the run where it
    finally has something to say.
    """
    done = run_snippet(tmp_path, """
        group_granted() { return 0; }
        loopback_loaded() { return 0; }
        linux_first_run
        echo "RC=$?"
    """)
    assert "RC=0" in done.stdout
    assert "missing things only root" not in done.stdout


def test_the_wav_player_is_no_longer_a_prerequisite(tmp_path: Path) -> None:
    """There were THREE root-owned prerequisites until 2026-09-15 and
    there are two. Spoken calls moved into the browser, so a rig plays
    nothing and has no reason to want alsa-utils, an `audio` group or a
    sound card at all.

    Pinned two ways, because each catches a different half of a partial
    revert: the probe function must be gone (a stub for it in the test
    above would silently do nothing, which is how a dead prerequisite
    check survives), and a rig missing only its cameras must not be told
    about sound."""
    assert "wav_player_present" not in RUN_SH.read_text()
    done = run_snippet(tmp_path, """
        group_granted() { return 1; }
        loopback_loaded() { return 0; }
        linux_first_run
    """)
    for gone in ("WAV", "aplay", "paplay", "'audio' group"):
        assert gone not in done.stdout, f"run.sh still offers to fix {gone}"


def test_missing_prereqs_without_a_tty_warns_but_starts(tmp_path: Path) -> None:
    """No terminal: report and continue, never prompt.

    This is the systemd/restart-loop path. Returning non-zero here, or
    reaching sudo, would take a rig that merely cannot see its cameras and
    stop it from serving its dashboard too.
    """
    done = run_snippet(tmp_path, """
        group_granted() { return 1; }
        linux_first_run
        echo "RC=$?"
    """)
    assert "RC=0" in done.stdout
    assert "not in the 'video' group" in done.stdout
    assert "sudo ./scripts/setup_linux.sh" in done.stdout
    assert "SUDO-CALLED" not in done.stdout, "prompted for root with no tty"


def test_declining_is_not_written_by_the_no_tty_path(tmp_path: Path) -> None:
    """Only a person saying "no" suppresses the offer.

    A headless first boot must not silently opt the rig out of a setup its
    owner never saw offered.
    """
    run_snippet(tmp_path, "group_granted() { return 1; }\nlinux_first_run")
    assert not (tmp_path / "data" / ".linux-setup-declined").exists()


def test_declined_marker_and_skip_env_suppress_the_offer(tmp_path: Path) -> None:
    """Both opt-outs work, so a rig can be told once and believed."""
    done = run_snippet(tmp_path, """
        group_granted() { return 1; }
        echo x > data/.linux-setup-declined
        echo "MARKER:[$(linux_first_run 2>&1)]"
        rm -f data/.linux-setup-declined
        echo "ENV:[$(OPENDARTS_SKIP_LINUX_SETUP=1 linux_first_run 2>&1)]"
    """)
    assert "MARKER:[]" in done.stdout
    assert "ENV:[]" in done.stdout


def test_regroup_re_execs_through_sg_preserving_arguments(tmp_path: Path) -> None:
    """The group is picked up in THIS run, with the command line intact.

    Re-exec is the whole point -- `usermod -aG` cannot affect a running
    process, so without this the run that installs the prerequisites is
    still the run that cannot open a camera. Arguments go through a shell
    inside `sg -c`, so quoting is the thing most likely to break.
    """
    done = run_snippet(tmp_path, """
        group_granted() { return 0; }
        group_active()  { [ "$1" = video ] && return 1; return 0; }
        SELF="$PWD/echo-args.sh"
        cat > echo-args.sh <<'EOS'
#!/bin/bash
for a in "$@"; do echo "ARG=[$a]"; done
EOS
        chmod +x echo-args.sh
        regroup_if_needed "a b" "it's" "--flag=x y"
    """)
    assert "SG-GROUP=video" in done.stdout
    assert "ARG=[a b]" in done.stdout
    assert "ARG=[it's]" in done.stdout
    assert "ARG=[--flag=x y]" in done.stdout


def test_regroup_depth_guard_stops_an_exec_loop(tmp_path: Path) -> None:
    """A system where sg does not actually add the group must not spin.

    Without the guard that case is an infinite exec loop, which on a rig
    under a restart loop is indistinguishable from a hang.
    """
    done = run_snippet(tmp_path, """
        group_granted() { return 0; }
        group_active()  { return 1; }
        SELF="$PWD/echo-args.sh"
        printf '#!/bin/bash\necho SG-REACHED\n' > echo-args.sh
        chmod +x echo-args.sh
        OPENDARTS_REGROUP_DEPTH=4 regroup_if_needed "x"
        echo "RETURNED"
    """)
    assert "RETURNED" in done.stdout
    assert "SG-REACHED" not in done.stdout


def test_regroup_without_sg_still_explains_the_stale_group(tmp_path: Path) -> None:
    """No sg is no excuse for a silent, confusing camera failure."""
    done = run_snippet(tmp_path, """
        PATH="/usr/bin:/bin"   # drops the harness's fake sg
        group_granted() { return 0; }
        group_active()  { return 1; }
        regroup_if_needed
        echo "RC=$?"
    """)
    assert "RC=0" in done.stdout
    assert "new login" in done.stderr


# ---------------------------------------------------------------------------
# run.ps1
#
# NOT EXECUTED, and it cannot be: there is no PowerShell on the CI hosts or
# on either developer machine, both of which are UNIX. That is the same
# blind spot that let a Linux-only `import fcntl` stop the Windows rig
# booting on 2026-09-15 -- the suite could not fail, so it passed.
#
# Text-level invariants are a poor substitute for running the thing, and
# they are what is available. Each one below is a bug that actually
# happened or was caught in review, not a guess at what might.

PS1 = REPO_ROOT / "run.ps1"


def test_run_ps1_exists() -> None:
    assert PS1.is_file()


def test_the_store_stub_is_rejected_structurally() -> None:
    """Windows' fake python.exe must never be used.

    %LOCALAPPDATA%\\Microsoft\\WindowsApps\\python.exe is a stub that prints
    "Python was not found; run without arguments to install from the
    Microsoft Store" and exits. It is on PATH by default and it satisfies
    Get-Command, so a naive check believes Python is installed. Worse, it
    SHADOWS a real installation further down PATH, so installing Python
    does not necessarily fix it -- which makes the obvious error message,
    "is Python installed and on PATH?", actively misleading.

    Measured on a brand-new Windows VM, 2026-09-15: a fresh clone failed
    here with exactly that wrong question.
    """
    text = PS1.read_text(encoding="utf-8")
    assert r"WindowsApps" in text, (
        "run.ps1 no longer rejects the Microsoft Store python stub. A clean "
        "Windows install will run the stub, get a Store advert, and report "
        "that Python is missing when it may well be installed."
    )


def test_splatting_uses_a_variable_not_an_array_subexpression() -> None:
    """`@($x)` is not splatting, and the difference is silent.

    Only `@name` against a VARIABLE expands to separate arguments. The
    array subexpression operator `@(...)` passes the whole array as one
    argument, which for `py -3 -m venv` means handing the interpreter a
    single argument that happens to look like a list. Caught in review
    rather than by running it, because nothing here can run it.
    """
    # CODE ONLY. The first version of this test matched the comment that
    # explains why the wrong form is wrong, and failed on a correct file --
    # a test that cannot distinguish an explanation from the mistake it
    # describes will eventually be silenced rather than fixed.
    code = "\n".join(
        line for line in PS1.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "@pyArgs" in code, "the splat variable is gone"
    assert "@($py.Args)" not in code, (
        "run.ps1 is back to `@($py.Args)`, which is the array subexpression "
        "operator and passes the array as a single argument -- not splatting"
    )


def test_path_is_refreshed_in_process_after_installing_python() -> None:
    """A first run must not end in "now open a new terminal and do it again".

    PATH is captured when a process starts, so a freshly installed Python
    is invisible to the shell that installed it. The installer writes the
    real value to the registry, so re-reading Machine+User PATH from there
    makes it visible without a restart.
    """
    text = PS1.read_text(encoding="utf-8")
    assert "GetEnvironmentVariable('Path', 'Machine')" in text, (
        "run.ps1 no longer re-reads PATH from the registry after installing "
        "Python, so a successful install still ends with the user being told "
        "to open a new window"
    )


def test_python_install_is_offered_never_silent() -> None:
    """Installing software is asked for, and only where someone can answer.

    Same posture as the Linux prerequisites: a restart loop or a service
    with no terminal must not have Python installed underneath it without
    anyone's say-so.
    """
    text = PS1.read_text(encoding="utf-8")
    assert "Read-Host" in text, "the install is no longer offered, just done"
    assert "UserInteractive" in text, (
        "run.ps1 no longer checks for an interactive session before "
        "prompting -- a headless restart loop would hang on the prompt"
    )


def test_run_ps1_has_no_shell_style_quote_escaping() -> None:
    r"""PowerShell escapes a quote by doubling it, not the POSIX way.

    Written while editing this file from a UNIX shell, `'...winget'"'"'s...'`
    is the bash idiom for an apostrophe inside a single-quoted string. In
    PowerShell it is a syntax error that takes the entire script with it --
    so a rig would not start at all, and the failure would be at parse
    time with no useful line of its own.

    Caught by reading the generated file rather than by running it,
    because nothing here can run it. The PowerShell form is `''`.
    """
    text = PS1.read_text(encoding="utf-8")
    assert "'\"'\"'" not in text, (
        "run.ps1 contains POSIX-style quote escaping ('\"'\"'), which is a "
        "PowerShell syntax error. Double the quote instead: ''"
    )


def test_run_ps1_single_quotes_balance_on_every_line() -> None:
    """An odd number of quotes on a line is almost always a broken literal.

    Crude on purpose -- a real parser is what this deserves and there is
    no PowerShell here to provide one. It is still enough to catch the
    unterminated-string class, which is fatal at parse time and therefore
    stops a rig booting rather than misbehaving in some visible way.

    Comments are skipped: prose about `don't` is not a code defect, and a
    check that cannot tell the difference gets silenced rather than fixed.
    """
    offenders = []
    for n, line in enumerate(PS1.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if line.count("'") % 2:
            offenders.append(f"{n}: {stripped[:70]}")
    assert not offenders, (
        "these lines have an odd number of single quotes, which usually "
        "means an unterminated string literal:\n  " + "\n  ".join(offenders)
    )


def test_a_failed_pip_install_is_not_ignored() -> None:
    """Starting after a failed install produces the wrong error message.

    Measured on a fresh Windows VM, 2026-09-15: pip failed, run.ps1 said
    nothing, launched the product, and it died with

        ModuleNotFoundError: No module named 'numpy'

    three frames into an import chain with the real cause scrolled off
    above. Anyone reading that goes looking at imports, packaging, or the
    venv -- everywhere except the install that had already reported
    failing.
    """
    text = PS1.read_text(encoding="utf-8")
    pip_at = text.index("pip install")
    after = text[pip_at:pip_at + 900]
    assert "$LASTEXITCODE" in after, (
        "run.ps1 runs pip install and never checks the result, so a failed "
        "install is followed by a start that fails with a misleading error"
    )


def test_pip_failure_is_fatal_on_a_fresh_venv_only() -> None:
    """Both halves matter, and they pull in opposite directions.

    A venv created seconds ago with nothing installed cannot run anything,
    so continuing guarantees a confusing failure. An established rig that
    hits a transient network failure already has its dependencies, and
    refusing to start there would turn a flaky mirror into an outage --
    which on a dartboard means no scoring all evening.
    """
    text = PS1.read_text(encoding="utf-8")
    assert "$freshVenv" in text, (
        "run.ps1 no longer distinguishes a fresh venv from an established "
        "one when pip fails -- one case must be fatal and the other must not"
    )
    pip_at = text.index("pip install")
    after = text[pip_at:pip_at + 1600]
    assert "exit 1" in after, "a fresh venv with no dependencies still starts"
    assert "WARNING" in after, (
        "an established rig no longer gets the warning-and-continue path"
    )


def test_run_ps1_uses_powershell_comments_only() -> None:
    """`//` is a comment in C and JavaScript and a syntax error here.

    Easy to type when the same person has been editing JS and bash all
    day, fatal at parse time, and invisible to every other check in this
    file. Written after doing exactly that on 2026-09-15.
    """
    offenders = [
        f"{n}: {line.strip()[:70]}"
        for n, line in enumerate(PS1.read_text(encoding="utf-8").splitlines(), 1)
        if line.lstrip().startswith("//")
    ]
    assert not offenders, (
        "C-style comments in a PowerShell script -- a parse error, so the "
        "script will not run at all:\n  " + "\n  ".join(offenders)
    )


def test_a_broken_venv_is_rebuilt_not_reused() -> None:
    """A Windows venv can exist and be unusable, and Test-Path cannot tell.

    Its python.exe is a real file that re-execs the interpreter recorded
    by absolute path in pyvenv.cfg. Move or remove that base Python and
    the venv keeps a python.exe that only ever prints

        No Python at 'C:\\...\\Python312-arm64\\python.exe'

    and exits 103. Measured on an ARM64 Windows VM, 2026-09-15, where
    run.ps1 reused the wreckage forever: pip failed, the product failed,
    and the restart loop span every three seconds without ever repairing
    it.

    Linux mostly escapes this because the venv interpreter there is a
    symlink that dangles, which is why run.sh's existence check looks
    sufficient and is not on Windows.
    """
    text = PS1.read_text(encoding="utf-8")
    assert "Test-VenvUsable" in text, (
        "run.ps1 is back to testing only that the venv interpreter EXISTS. "
        "On Windows a venv whose base Python has gone still has a "
        "python.exe, so it will be reused forever and never repaired."
    )
    assert "Remove-Item" in text, (
        "a venv that fails to run is detected but never removed, so the "
        "rebuild cannot happen"
    )


def test_macos_offers_python_312_instead_of_building_on_system_39() -> None:
    """A fresh Mac must not quietly get a venv on Apple's frozen 3.9.

    Measured on a fresh macOS VM, 2026-09-16: Homebrew was installed at
    /opt/homebrew/bin, which a non-login shell's PATH does not include,
    so the only interpreter visible was /usr/bin/python3 (3.9.6) and the
    venv was built on it without a word. A venv is pinned to the
    interpreter that created it, so that choice sticks until someone
    knows to delete .venv.
    """
    text = RUN_SH.read_text()
    assert "brew\" install python@3.12" in text or 'install python@3.12' in text, (
        "run.sh no longer offers to install Python 3.12 on macOS"
    )
    assert "/opt/homebrew/bin/brew" in text, (
        "Homebrew is looked up via PATH only -- on a non-login shell that "
        "misses /opt/homebrew/bin/brew entirely"
    )
    assert "/usr/local/opt/python@3.12/bin/python3.12" in text, (
        "pick_python misses Intel Homebrew's keg path, so an Intel Mac "
        "installs 3.12 and then cannot find it"
    )


def test_run_sh_enforces_the_declared_python_floor(tmp_path: Path) -> None:
    """Only interpreters that pass the 3.12 probe may be chosen.

    pick_python used to return the first python that EXISTED, so a Mac
    with only Apple's 3.9.6 got a 3.9 venv that could not import numpy.

    py_ok is exercised for real against two fake interpreters, one that
    passes the probe and one that fails it. pick_python itself is checked
    structurally: it searches absolute Homebrew paths, so on a developer
    Mac with Homebrew installed no PATH trick can make it find nothing,
    and a test that depended on that would pass or fail by machine.
    """
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.write_text("#!/bin/sh\nexit 0\n")
    bad.write_text("#!/bin/sh\nexit 1\n")
    good.chmod(0o755)
    bad.chmod(0o755)
    script = tmp_path / "probe.sh"
    script.write_text(
        functions_from_run_sh("py_ok")
        + f"\npy_ok {good} && echo good-ok || echo good-rejected"
        + f"\npy_ok {bad} && echo bad-ok || echo bad-rejected\n"
    )
    out = subprocess.run(["/bin/bash", str(script)], capture_output=True, text=True).stdout
    assert "good-ok" in out and "bad-rejected" in out, out

    body = functions_from_run_sh("pick_python")
    assert 'py_ok "$candidate"' in body, (
        "pick_python no longer checks the version floor before choosing"
    )
    assert "echo python3" not in body, (
        "pick_python is back to falling back to whatever `python3` is, "
        "which is how a 3.9 venv got built"
    )
    assert "return 1" in body, "pick_python must fail when nothing qualifies"


def test_both_launchers_state_the_same_floor_as_pyproject() -> None:
    """One number, three places; drift here is a silent install failure."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.12"' in pyproject
    assert "sys.version_info >= (3, 12)" in RUN_SH.read_text()
    assert "sys.version_info >= (3, 12)" in PS1.read_text(encoding="utf-8")


def test_run_ps1_moves_a_locked_filter_dll_aside_before_pulling() -> None:
    """A pull that cannot replace a loaded DLL stops halfway and leaves the
    files it already wrote as local changes, blocking every later pull."""
    text = (REPO_ROOT / "run.ps1").read_text()
    move = text.index("Move-Item -LiteralPath $dll")
    pull = text.index("& git pull --ff-only")
    assert move < pull
    assert "& git checkout -- tools/winvcam/vcam_probe.dll" in text
    assert "vcam_probe.dll.old-*" in (REPO_ROOT / ".gitignore").read_text()


# ---------------------------------------------------------------------------
# Pull only when asked (2026-09-17)
#
# BOTH LAUNCHERS USED TO PULL ON EVERY RELAUNCH. A relaunch is triggered by
# the process EXITING, which includes crashing, so a rig that fell over
# mid-match came back on whatever was on main at that second -- possibly a
# commit pushed minutes earlier by someone who had no idea a match was
# running. Two keys in data/config.json now decide (always_update,
# update_on_next_restart), read by the venv python rather than by a JSON
# parser written twice, once per shell.
#
# This is exactly the kind of thing that cannot be tried by hand: trying it
# means letting a real rig crash and watching which commit comes back.

#: The module both launchers invoke to read and clear the flags.
UPDATE_POLICY_MODULE = "opendarts.live.update_policy"


def run_pull_snippet(
    tmp_path: Path, decision_script: str, body: str, *, git_rc: int = 0
) -> subprocess.CompletedProcess:
    """Run run.sh's own `pull_if_requested` against fake tools.

    `decision_script` becomes `.venv/bin/python3` -- whatever it prints on
    stdout is what the launcher reads as the decision, which is how a
    "skip", a "pull ..." and a broken interpreter are all reachable from
    one harness. `git` is a recorder, never the real thing: a test that
    wrongly decides to pull would otherwise fetch over the network and
    move the checkout it is running from.
    """
    venv_bin = tmp_path / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake_python = venv_bin / "python3"
    fake_python.write_text(decision_script)
    fake_python.chmod(0o755)

    bin_dir = tmp_path / "fakebin"
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text(f'#!/bin/bash\necho "GIT: $*"\nexit {git_rc}\n')
    fake_git.chmod(0o755)

    script = tmp_path / "harness.sh"
    script.write_text(
        'cd "$(dirname "$0")"\n'
        + functions_from_run_sh("pull_if_requested")
        + "\n" + body + "\n"
    )
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ.get('PATH', '')}")
    return subprocess.run(["bash", str(script)], capture_output=True,
                          text=True, cwd=tmp_path, env=env)


def _decision(line: str, rc: int = 0) -> str:
    return f'#!/bin/bash\necho "{line}"\nexit {rc}\n'


def test_run_sh_does_not_pull_when_neither_flag_is_set(tmp_path: Path) -> None:
    """The whole point. A relaunch with nothing asked for runs the commit
    the machine already has."""
    done = run_pull_snippet(tmp_path, _decision("skip"), 'pull_if_requested main\necho "RC=$?"')
    assert "GIT:" not in done.stdout, "run.sh pulled without being asked"
    assert "not pulling" in done.stdout
    assert "always_update and update_on_next_restart are both off" in done.stdout
    assert "RC=0" in done.stdout, "a skipped pull must not stop the product starting"


def test_run_sh_pulls_the_checked_out_branch_when_asked(tmp_path: Path) -> None:
    for decision in ("pull update_on_next_restart", "pull always_update"):
        sandbox = tmp_path / decision.split()[-1]
        sandbox.mkdir()
        done = run_pull_snippet(sandbox, _decision(decision), "pull_if_requested some-branch")
        assert "GIT: pull --ff-only origin some-branch" in done.stdout, decision
        # The reason rides in the log line, so "why did this rig just move
        # onto new code" is answerable from the startup banner alone.
        assert decision in done.stdout


def test_run_sh_does_not_pull_when_the_flags_cannot_be_read(tmp_path: Path) -> None:
    """A broken/half-installed checkout is NOT a request to update, and it
    is not the same thing as a clean skip either -- it is loud."""
    done = run_pull_snippet(tmp_path, _decision("", rc=1), 'pull_if_requested main\necho "RC=$?"')
    assert "GIT:" not in done.stdout
    assert "could not read the update flags" in done.stderr
    assert "RC=0" in done.stdout, "an unreadable config must not stop the product starting"


def test_run_sh_survives_a_failed_pull_loudly_and_does_not_retry(tmp_path: Path) -> None:
    """Today's behaviour on a failed pull -- start the existing code --
    kept, with the volume turned up. It must NOT loop retrying: the
    one-shot flag was already cleared when it was read, and a rig whose
    branch cannot fast-forward would otherwise fail identically on every
    single restart."""
    done = run_pull_snippet(
        tmp_path, _decision("pull update_on_next_restart"),
        'pull_if_requested main\necho "RC=$?"', git_rc=1,
    )
    assert "RC=0" in done.stdout, "a failed pull must still start the product"
    assert "git pull failed or was not a fast-forward" in done.stderr
    assert "STARTING THE EXISTING CODE AS-IS" in done.stderr
    assert "will NOT retry" in done.stderr


def test_run_sh_reads_and_clears_the_real_flags_through_the_real_module(tmp_path: Path) -> None:
    """END TO END, with the real reader and a real config file.

    Every other test here fakes the decision, which pins the shell and
    nothing else. This one puts the actual `.venv/bin/python3 -m
    opendarts.live.update_policy` call the launcher makes against a real
    data/config.json in a scratch directory -- so a rename, a changed
    output word or a flag that is read but never cleared fails here rather
    than on a rig.
    """
    import json
    import sys

    data = tmp_path / "data"
    data.mkdir()
    cfg = data / "config.json"
    cfg.write_text(json.dumps({"port": 8420, "update_on_next_restart": True}))

    shim = (
        "#!/bin/bash\n"
        f'export PYTHONPATH="{REPO_ROOT}"\n'
        f'export OPENDARTS_DATA_DIR="{data}"\n'
        "export PYTHONDONTWRITEBYTECODE=1\n"
        f'exec "{sys.executable}" "$@"\n'
    )
    done = run_pull_snippet(tmp_path, shim, "pull_if_requested main\npull_if_requested main")

    assert done.stdout.count("GIT: pull --ff-only origin main") == 1, (
        "one request must produce exactly one pull -- a flag that is read "
        "but not cleared turns a crash loop back into pull-every-restart"
    )
    assert "pull update_on_next_restart" in done.stdout
    assert "not pulling" in done.stdout, "the second pass should have skipped"
    raw = json.loads(cfg.read_text())
    assert raw["update_on_next_restart"] is False
    assert raw["port"] == 8420, "clearing the flag rewrote the operator's config"


def test_run_sh_reads_the_flags_with_python_not_with_grep() -> None:
    """The flags live in hand-editable JSON, and a grep/sed parser here
    would be a second implementation that eventually disagrees with
    opendarts.live.config about the one file that decides which code a rig
    runs."""
    body = functions_from_run_sh("pull_if_requested")
    assert f"-m {UPDATE_POLICY_MODULE}" in body
    for hand_rolled in ("grep", "sed", "awk", "cut", "tr "):
        assert hand_rolled not in body, f"run.sh parses config.json with {hand_rolled}"


def test_run_ps1_asks_the_same_module_before_pulling() -> None:
    """The Windows half, pinned at text level (there is no PowerShell on
    any machine this suite runs on -- see the section header above).

    ORDER IS THE ASSERTION: the decision has to be read before the pull,
    and the pull has to be inside the branch the decision guards. A
    `git pull` that merely follows the read in the file would still run
    unconditionally.
    """
    text = PS1.read_text(encoding="utf-8")
    ask = text.index(f"-m {UPDATE_POLICY_MODULE}")
    guard = text.index("$decision.StartsWith('pull')")
    pull = text.index("& git pull --ff-only")
    assert ask < guard < pull
    # The locked-DLL dance is only worth doing when something is about to
    # be written into the checkout, so it moved inside the same branch.
    assert guard < text.index("Move-Item -LiteralPath $dll") < pull


def test_run_ps1_survives_a_failed_pull_loudly_and_does_not_retry() -> None:
    text = PS1.read_text(encoding="utf-8")
    assert "git pull failed or was not a fast-forward" in text
    assert "STARTING THE EXISTING CODE AS-IS" in text
    assert "will NOT retry by itself" in text


def test_neither_launcher_pulls_unconditionally_any_more() -> None:
    """The regression that matters: someone restoring the old one-liner.

    Both scripts' `git pull` must be inside something that read the flags
    first, which at text level means the module name appears before it in
    both files -- and the old unguarded messages must be gone.
    """
    for text, name in ((RUN_SH.read_text(), "run.sh"),
                       (PS1.read_text(encoding="utf-8"), "run.ps1")):
        assert text.index(UPDATE_POLICY_MODULE) < text.index("git pull --ff-only"), name
        assert "running existing code as-is" not in text, (
            f"{name} is back to the unconditional pull's own message"
        )


def test_both_launchers_name_the_same_two_config_keys() -> None:
    """One pair of keys, two scripts, one config file. Drift here is a rig
    that silently ignores the flag an operator just set."""
    for text, name in ((RUN_SH.read_text(), "run.sh"),
                       (PS1.read_text(encoding="utf-8"), "run.ps1")):
        assert "always_update" in text, name
        assert "update_on_next_restart" in text, name
    example = (REPO_ROOT / "config.example.json").read_text()
    assert '"always_update": false' in example
    assert '"update_on_next_restart": false' in example
