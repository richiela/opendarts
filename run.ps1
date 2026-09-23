param(
    # Everything after the script name is handed straight to run_product,
    # so `run.ps1 --port 8421` works the same way `run.sh --port 8421` does.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PassThruArgs
)

# opendarts run_product launcher -- the Windows counterpart of run.sh.
#
#   powershell -ExecutionPolicy Bypass -File .\run.ps1
#
# Same contract as run.sh: every time the process exits for any reason, start
# it again. Nothing here relaunches anything itself.
#
# AND, LIKE run.sh SINCE 2026-09-17, IT ONLY PULLS WHEN ASKED. It used to
# pull on every relaunch, so a rig that crashed mid-match came back on
# whatever was on main at that moment. Deploying is now "push, then ask for
# an update": the dashboard's Config tab -> Maintenance -> Update and
# restart, or POST /api/restart {"update": true}, or the standing
# `always_update` key in data\config.json for a machine that is supposed to
# follow main. Both keys default to false. See docs\DEPLOYMENT.md and
# docs\WINDOWS.md.
#
# WHY POWERSHELL AND NOT A .bat, which is what this replaced: cmd.exe asks
# "Terminate batch job (Y/N)?" on every Ctrl-C inside a batch file. That
# prompt is cmd's own and cannot be suppressed from inside the script, and
# answering N resumes the batch at the NEXT line -- which in a goto restart
# loop means the server you just interrupted comes straight back up.
# PowerShell has no equivalent prompt: Ctrl-C ends the script.
#
# The -ExecutionPolicy flag above is not optional advice: a bare .\run.ps1 can
# be refused on a machine with a restrictive policy, while the per-invocation
# flag needs no admin and persists nothing.
#
# Deliberately NOT strict-mode: this is a supervisor that must keep running
# through a failed git pull or a missing data\run.env, not abort on the first
# surprise. Failures are reported and stepped over.

$ErrorActionPreference = 'Continue'

Set-Location -LiteralPath $PSScriptRoot

function Now { Get-Date -Format 'yyyy-MM-dd HH:mm:ss' }

# Ctrl-C protection, and its LIMIT -- read before trusting this.
#
# What actually stops this loop on Ctrl-C is PowerShell itself: the host
# unwinds the running script. The exit-code check below is a second line of
# defence for the case where the child is killed outright and the interrupt
# never reaches the host -- a process terminated by the console's default
# handler exits with STATUS_CONTROL_C_EXIT (0xC000013A, -1073741510 signed).
#
# It does NOT cover the ordinary case, and this was verified rather than
# assumed: run_product installs its own SIGINT handler
# (_install_signal_handlers in opendarts/live/run_product.py), so Ctrl-C
# produces a CLEAN shutdown and main() returns 0 -- byte-identical to any
# other normal exit. No launcher can tell those apart from the exit code
# alone. If this loop is ever seen resurrecting a server after Ctrl-C, the
# fix is a distinct exit code from run_product (130 on SIGINT, the POSIX
# convention, leaving SIGTERM at 0 so POST /api/restart still redeploys),
# not more logic here.
$StatusControlCExit = -1073741510

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

# -- finding a real Python ----------------------------------------------------
#
# THE STORE ALIAS IS THE WHOLE PROBLEM HERE. Windows ships a stub
# python.exe in %LOCALAPPDATA%\Microsoft\WindowsApps that is not Python at
# all -- it prints "Python was not found; run without arguments to install
# from the Microsoft Store" and exits. It is on PATH by default, it
# SHADOWS a real installation that is further down PATH, and it satisfies
# Get-Command, so "is Python installed and on PATH?" is both the obvious
# question and the wrong one. Someone can install Python, watch this fail
# identically, and have no idea why.
#
# So resolve it structurally rather than by trying and hoping: anything
# under \WindowsApps\ is the stub and is never usable, whatever it claims.
# 3.12 IS THE FLOOR -- pyproject.toml declares requires-python >= 3.12,
# because that is what the suite runs on. This used to accept any Python
# that answered --version, so an old install would have produced a venv
# that cannot import the product. run.sh had exactly that bug on macOS
# (Apple's 3.9.6, 2026-09-16); this is the same guard on this side.
$PythonMinCheck = 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'

function Test-PythonOk {
    param([string]$Exe, [string[]]$PrefixArgs = @())
    & $Exe @PrefixArgs -c $PythonMinCheck *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-RealPython {
    # The py launcher first. It ships with python.org installs, is not
    # shadowed by the alias, and picks the newest interpreter itself.
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher -and (Test-PythonOk 'py' @('-3'))) {
        return @{ Exe = 'py'; Args = @('-3') }
    }
    foreach ($name in @('python3', 'python')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        if ($cmd.Source -like '*\WindowsApps\*') { continue }  # the stub
        if (Test-PythonOk $cmd.Source) { return @{ Exe = $cmd.Source; Args = @() } }
    }
    return $null
}

function Show-PythonHelp {
    Write-Host ''
    Write-Host '[run.ps1] No Python 3.12 or newer was found on this machine.'
    $stub = Get-Command python -ErrorAction SilentlyContinue
    if ($stub -and $stub.Source -like '*\WindowsApps\*') {
        Write-Host '[run.ps1]'
        Write-Host '[run.ps1] NOTE: the `python` on your PATH is the Microsoft Store'
        Write-Host '[run.ps1] placeholder, not Python. It shadows a real install, so'
        Write-Host '[run.ps1] installing Python may not be enough on its own -- turn it'
        Write-Host '[run.ps1] off in Settings > Apps > Advanced app settings >'
        Write-Host '[run.ps1] App execution aliases (both python.exe and python3.exe).'
    }
    Write-Host '[run.ps1]'
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host '[run.ps1] Install Python with either:'
        Write-Host '[run.ps1]     winget install -e --id Python.Python.3.12 --source winget'
        Write-Host '[run.ps1]     https://www.python.org/downloads/windows/'
    } else {
        # winget ships in App Installer, which is present on Windows 11 and
        # on Windows 10 1809+ THAT HAVE BEEN UPDATED -- a stock image or a
        # fresh VM often has neither. Worth naming, because "winget is not
        # recognized" reads as a typo rather than as a missing component.
        Write-Host '[run.ps1] winget is not available on this machine, so it cannot'
        Write-Host '[run.ps1] install Python for you. winget ships in "App Installer":'
        Write-Host '[run.ps1]     https://apps.microsoft.com/detail/9nblggh4nns1'
        Write-Host '[run.ps1] Or install Python directly and skip winget entirely:'
        Write-Host '[run.ps1]     https://www.python.org/downloads/windows/'
    }
    Write-Host '[run.ps1] and tick "Add python.exe to PATH" if you use the installer.'
    Write-Host '[run.ps1] Then open a NEW PowerShell window and run .\run.ps1 again.'
    Write-Host ''
}

# -- venv -------------------------------------------------------------------
# RUN IT, DO NOT STAT IT. A Windows venv's python.exe is a real file that
# re-execs the interpreter it was created from, recorded by absolute path
# in pyvenv.cfg. Remove, move or replace that base Python and the venv
# keeps its python.exe -- the file exists, Test-Path is happy, and every
# invocation prints
#
#     No Python at 'C:\...\Python312-arm64\python.exe'
#
# and exits 103. run.ps1 then reused that wreckage forever: pip failed,
# the product failed, the loop restarted every 3 seconds, and nothing ever
# repaired it. Measured on an ARM64 Windows VM, 2026-09-15, where a
# Python312-arm64 install had been replaced.
#
# Linux is mostly spared because there the venv interpreter is a SYMLINK,
# so it dangles and `-x` catches it -- which is why run.sh's equivalent
# guard looks like it is enough and is not, here.
function Test-VenvUsable {
    param([string]$Exe)
    if (-not (Test-Path -LiteralPath $Exe)) { return $false }
    # Runs AND meets the floor. A venv built on an old Python stays old
    # forever -- it is pinned to the interpreter that made it -- so one
    # below 3.12 is rebuilt just like one whose interpreter has vanished.
    return (Test-PythonOk $Exe)
}

$freshVenv = $false
if (-not (Test-VenvUsable $python)) {
    $venvDir = Join-Path $PSScriptRoot '.venv'
    if (Test-Path -LiteralPath $venvDir) {
        Write-Host '[run.ps1] .venv is unusable -- its interpreter either does not run'
        Write-Host '[run.ps1] (the Python it was built from has moved or been removed)'
        Write-Host '[run.ps1] or is older than the 3.12 this project requires.'
        Write-Host '[run.ps1] Removing the broken venv and rebuilding it.'
        Remove-Item -LiteralPath $venvDir -Recurse -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $venvDir) {
            Write-Host '[run.ps1] ERROR: could not remove .venv -- is something using it?'
            Write-Host '[run.ps1] Close any running process from it and try again.'
            exit 1
        }
    }
    $freshVenv = $true
    Write-Host '[run.ps1] no usable .venv -- creating one...'
    $py = Get-RealPython

    # OFFERED, NOT ASSUMED. Same posture as scripts/setup_linux.sh: this
    # will happily install Python for someone standing at the machine, and
    # will never do it behind the back of a service or a restart loop with
    # no one to ask. winget is present on Windows 10 1809+ and Windows 11.
    if (-not $py -and [Environment]::UserInteractive `
            -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host '[run.ps1] Python is required and was not found.'
        $answer = Read-Host '[run.ps1] Install it now with winget? [Y/n]'
        if ($answer -notmatch '^[Nn]') {
            # --source winget, EXPLICITLY. Without it winget searches every
            # configured source, and a broken one takes the whole command
            # down with it: measured on a fresh Windows VM, 2026-09-15, the
            # msstore source failed certificate validation
            # (0x8a15005e) and winget then refused to install the package
            # it had already found in the `winget` source, asking for
            # --source to disambiguate. Naming the source skips the store
            # entirely -- we want the python.org build anyway, because it
            # ships the `py` launcher and is not subject to the Store
            # alias nonsense above.
            # X64 EVEN ON ARM64, deliberately, and this is measured rather
            # than cautious. As of 2026-09-15 PyPI publishes
            # opencv-python-headless wheels for win32 and win_amd64 only --
            # there is no win_arm64 build. numpy has one; OpenCV does not.
            # So a native ARM64 Python finds no wheel, tries to compile
            # OpenCV from source, and fails on any machine without CMake
            # and the full Visual Studio toolchain. That is exactly what a
            # fresh ARM64 Windows VM did today, reporting only pip's
            # "this error originates from a subprocess".
            #
            # ARM64 Windows runs x64 binaries under emulation, so an x64
            # Python gets every wheel and works. It costs some speed in
            # OpenCV, which is the expensive part of this product -- so
            # revisit if opencv-python ever ships win_arm64, at which point
            # deleting this argument is the whole change.
            $arch = @()
            if ($env:PROCESSOR_ARCHITECTURE -eq 'ARM64') {
                Write-Host '[run.ps1] ARM64 Windows detected -- installing the x64 Python build,'
                Write-Host '[run.ps1] because OpenCV publishes no ARM64 Windows wheel and a native'
                Write-Host '[run.ps1] build would try to compile it from source.'
                $arch = @('--architecture', 'x64')
            }
            winget install -e --id Python.Python.3.12 --source winget @arch `
                --accept-package-agreements --accept-source-agreements
            $wingetExit = $LASTEXITCODE
            # CHECKED, because the first version did not. When winget
            # failed, this code went straight on to re-probe, found no
            # Python, and reported "Python was installed, but this window
            # still cannot see it" -- a confident, specific and completely
            # false statement about what had just happened, sending the
            # reader to look at PATH when the install had never run. An
            # exit code was available the whole time and nothing read it.
            if ($wingetExit -ne 0) {
                Write-Host ''
                Write-Host "[run.ps1] winget failed (exit $wingetExit) -- Python was NOT installed."
                Write-Host '[run.ps1] The message above came from winget, not from us.'
                Show-PythonHelp
                exit 1
            }

            # PATH IS READ ONCE AT PROCESS START, so this window cannot
            # see what winget just installed -- the classic "install it,
            # then open a new terminal" step. That step is avoidable and
            # worth avoiding: a first run that ends in "now do it again
            # somewhere else" is a first run that failed.
            #
            # The installer writes the real PATH to the registry, so
            # re-read it from there (Machine then User, the order Windows
            # itself composes them in) and rebuild this process's copy.
            # Then re-probe.
            $env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') `
                + ';' + [System.Environment]::GetEnvironmentVariable('Path', 'User')
            $py = Get-RealPython
            if (-not $py) {
                Write-Host ''
                Write-Host '[run.ps1] winget reported success, but no usable Python can be'
                Write-Host '[run.ps1] found even after re-reading PATH from the registry.'
                Write-Host '[run.ps1] Open a NEW PowerShell window and run .\run.ps1 again;'
                Write-Host '[run.ps1] if it still fails there, Python did not actually install.'
                Write-Host ''
                exit 1
            }
            Write-Host '[run.ps1] Python installed and found without reopening the shell.'
        }
    }

    if (-not $py) {
        Show-PythonHelp
        exit 1
    }

    # SPLATTED VIA A VARIABLE, which is the only form PowerShell accepts.
    # `@($py.Args)` looks equivalent and is not: that is the array
    # subexpression operator, and it passes the whole array as ONE
    # argument. Only `@name` against a variable expands to separate
    # arguments, and an empty array then correctly expands to nothing.
    $pyArgs = $py.Args
    & $py.Exe @pyArgs -m venv .venv
    if (-not (Test-Path -LiteralPath $python)) {
        Write-Host "[run.ps1] FAILED to create .venv using $($py.Exe). See the output above."
        exit 1
    }
}

while ($true) {
    # Follows whatever branch is checked out, so a rig can run a feature
    # branch by checking it out once. Defaults to main if git cannot answer
    # (detached HEAD, no repo) rather than guessing wrong.
    $branch = 'main'
    if (Get-Command git -ErrorAction SilentlyContinue) {
        $head = & git rev-parse --abbrev-ref HEAD 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($head)) {
            $branch = $head.Trim()
        }
    }

    # SHOULD THIS RELAUNCH PULL? Read out of data\config.json by the venv
    # python -- never by PowerShell. The same two keys run.sh reads
    # (always_update, update_on_next_restart, both default false), and the
    # same one reader: opendarts/live/update_policy.py, which also CLEARS
    # the one-shot flag as it reads it, so a failed pull cannot leave a rig
    # pulling on every restart. A second JSON parser written here would
    # eventually disagree with the Python one about the same file, on the
    # one file that decides which code this rig runs.
    #
    # Captured as an array so $LASTEXITCODE survives: a non-zero exit means
    # "could not answer at all" (no venv, a half-installed checkout), which
    # is a different thing from a clean "skip" and is reported differently.
    $decisionLines = @(& $python -m opendarts.live.update_policy 2>$null)
    $decision = ''
    if ($LASTEXITCODE -eq 0 -and $decisionLines.Count -gt 0) {
        $decision = "$($decisionLines[0])".Trim()
    }

    if ([string]::IsNullOrWhiteSpace($decision)) {
        Write-Host '[run.ps1] WARNING: could not read the update flags out of data\config.json.'
        Write-Host '[run.ps1] NOT pulling -- starting the code that is already here.'
    } elseif (-not $decision.StartsWith('pull')) {
        Write-Host '[run.ps1] not pulling: always_update and update_on_next_restart are both off in data\config.json.'
        Write-Host '[run.ps1] To update: the dashboard Config tab -> Maintenance -> Update and restart.'
    } else {

        # A LOADED DLL CANNOT BE REPLACED, BUT IT CAN BE RENAMED. Any program
        # showing the virtual cameras (a camera app, for example) keeps
        # vcam_probe.dll loaded. A pull that updates it then writes the other
        # files, fails on the DLL, and stops -- leaving those files behind as
        # "local changes" that block every later pull (seen on a Windows rig, 2026-09-17). So a
        # locked DLL is renamed aside and a fresh, unlocked copy checked out for
        # the pull to replace. The program holding the old one keeps using it
        # until it restarts; the renamed copy is deleted on a later start.
        $dll = Join-Path $PSScriptRoot 'tools\winvcam\vcam_probe.dll'
        Get-ChildItem -LiteralPath (Split-Path $dll) -Filter 'vcam_probe.dll.old-*' -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue
        if (Test-Path -LiteralPath $dll) {
            $locked = $false
            try { [IO.File]::Open($dll, 'Open', 'ReadWrite', 'None').Close() } catch { $locked = $true }
            if ($locked) {
                $aside = "$dll.old-$(Get-Date -Format yyyyMMddHHmmss)"
                try {
                    Move-Item -LiteralPath $dll -Destination $aside -ErrorAction Stop
                    & git checkout -- tools/winvcam/vcam_probe.dll
                    Write-Host '[run.ps1] vcam_probe.dll is in use -- moved it aside so the pull can update it'
                } catch {
                    Write-Host "[run.ps1] vcam_probe.dll is in use and could not be moved aside: $_"
                }
            }
        }

        Write-Host "[run.ps1] $(Now): pulling latest ($branch) -- $decision"
        & git pull --ff-only origin $branch
        if ($LASTEXITCODE -ne 0) {
            # LOUD, AND IT DOES NOT RETRY -- the one-shot flag was already
            # cleared when it was read, deliberately. A rig whose pull cannot
            # fast-forward would otherwise fail at the same thing on every
            # restart, while starting the old code each time regardless.
            Write-Host '[run.ps1] ERROR: git pull failed or was not a fast-forward.'
            Write-Host '[run.ps1] STARTING THE EXISTING CODE AS-IS. The update request has already'
            Write-Host '[run.ps1] been cleared, so this will NOT retry by itself -- ask for it again'
            Write-Host '[run.ps1] once the repository can fast-forward.'
        }
    }

    # CHECKED, and the check is why this exists. It used to run unguarded,
    # so a failed install was followed immediately by a normal-looking
    # start and then:
    #
    #     ModuleNotFoundError: No module named 'numpy'
    #
    # three frames into an import chain, with the real error scrolled off
    # above. Measured on a fresh Windows VM, 2026-09-15.
    #
    # Fatal on a FRESH venv, a warning otherwise. The distinction is the
    # whole point: a brand-new venv that could not install anything has
    # nothing to run, and starting is guaranteed to fail in a way that
    # blames the wrong thing. An established rig that hits a transient
    # network failure already has its dependencies and should come back
    # up -- refusing to start there would turn a flaky mirror into an
    # outage. Same reasoning as run.sh's equivalent.
    # FULL -> HEADLESS OPENCV, ONCE. Windows ran the full opencv-python from
    # 2026-09-16 to 2026-09-17 for OpenCV's Media Foundation backend; cameras
    # are now read by our own Media Foundation code, so requirements.txt is
    # back to headless. The two packages install into the SAME cv2/
    # directory, so a leftover full build is removed along with any headless
    # one, and the install below puts headless back cleanly.
    & $python -m pip show opencv-python *> $null
    if ($LASTEXITCODE -eq 0) {
        Write-Host '[run.ps1] replacing opencv-python with opencv-python-headless'
        & $python -m pip uninstall -y opencv-python opencv-python-headless
    }
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        if ($freshVenv) {
            Write-Host ''
            Write-Host "[run.ps1] ERROR: installing dependencies failed (pip exit $LASTEXITCODE)."
            Write-Host '[run.ps1] This venv was created moments ago, so nothing is installed'
            Write-Host '[run.ps1] and starting now would only fail with a confusing'
            Write-Host '[run.ps1] ModuleNotFoundError. The real error is in the pip output above.'
            Write-Host '[run.ps1]'
            Write-Host '[run.ps1] A package that has to be COMPILED is the usual cause on a new'
            Write-Host '[run.ps1] Windows box -- pip says "this error originates from a'
            Write-Host '[run.ps1] subprocess" when a wheel was unavailable and the build failed.'
            Write-Host '[run.ps1] Either install the Visual Studio Build Tools, or use a Python'
            Write-Host '[run.ps1] version that has prebuilt wheels (3.12 is a safe choice).'
            Write-Host "[run.ps1] This venv is running $(& $python --version 2>&1)."
            Write-Host ''
            exit 1
        }
        Write-Host ''
        Write-Host "[run.ps1] WARNING: pip install failed (exit $LASTEXITCODE) -- starting with"
        Write-Host '[run.ps1] whatever is already installed. If the process below dies on a'
        Write-Host '[run.ps1] missing module, THIS is why.'
        Write-Host ''
    }

    # Optional per-rig environment (untracked): KEY=VALUE lines. Comment and
    # blank lines are skipped, and only the FIRST "=" splits, so values
    # containing "=" survive.
    $envFile = Join-Path $PSScriptRoot 'data\run.env'
    if (Test-Path -LiteralPath $envFile) {
        Write-Host '[run.ps1] loading data\run.env'
        foreach ($line in Get-Content -LiteralPath $envFile) {
            $trimmed = $line.Trim()
            if ($trimmed -eq '' -or $trimmed.StartsWith('#')) { continue }
            $split = $trimmed.IndexOf('=')
            if ($split -lt 1) { continue }
            $key = $trimmed.Substring(0, $split).Trim()
            $value = $trimmed.Substring($split + 1)
            # One layer of surrounding quotes is stripped, so FOO="bar"
            # and FOO=bar set the same value -- matching how run.sh's
            # shell sourcing treats them.
            if ($value.Length -ge 2 -and
                (($value.StartsWith('"') -and $value.EndsWith('"')) -or
                 ($value.StartsWith("'") -and $value.EndsWith("'")))) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }

    Write-Host "[run.ps1] $(Now): starting opendarts run_product (port from data\config.json, default 8420)..."
    & $python -m opendarts.live.run_product @PassThruArgs
    $code = $LASTEXITCODE

    if ($code -eq $StatusControlCExit) {
        Write-Host '[run.ps1] interrupted by Ctrl-C -- not restarting.'
        break
    }

    Write-Host "[run.ps1] $(Now): process exited ($code), restarting in 3s..."
    Start-Sleep -Seconds 3
}
