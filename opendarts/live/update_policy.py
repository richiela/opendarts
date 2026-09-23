"""Does the launcher pull this time? -- the one answer, given once.

`run.sh` and `run.ps1` call this module as a CLI, between the old
process exiting and the new one starting:

    .venv/bin/python3 -m opendarts.live.update_policy

It prints exactly ONE line -- `pull always_update`, `pull
update_on_next_restart`, or `skip` -- and, as a side effect, clears
`update_on_next_restart` in `data/config.json` if it was set.

WHY A PYTHON MODULE AND NOT FIVE LINES OF SHELL. The flags live in
`data/config.json`, and both launchers already hold a venv interpreter at
the point they need the answer. Parsing JSON in bash means `grep`/`sed`
against a hand-editable file, and in PowerShell it means a second,
separate implementation of the same rules -- two parsers that will
eventually disagree with `opendarts.live.config`, on the one file that
decides which code a rig runs. This module imports nothing but the
standard library and `opendarts.live.config`, so it also works on a
fresh clone whose venv has not installed requirements.txt yet.

WHY IT CLEARS THE ONE-SHOT FLAG *BEFORE* THE PULL RATHER THAN AFTER.
The flag is a REQUEST, and the request has been received by the time
this prints. Clearing afterwards would leave a window -- a pull that
hangs and is killed, a machine that loses power mid-fetch -- where the
flag survives and the next relaunch pulls again, and the one after that,
which is precisely the every-restart-pulls behaviour these keys exist to
end. A cleared-then-failed pull is also the behaviour the brief asks
for: the product starts on the old code, loudly, and does not retry by
itself. Read-modify-write is a single call into
`config.set_update_on_next_restart()`, so the file is never rewritten
from a stale copy.

NOTHING HERE PULLS. This module reads and clears; the `git pull` itself
stays in the launcher, where the branch, the network and the failure
message already live.
"""
from __future__ import annotations

import sys
from pathlib import Path

from opendarts.live.config import (
    DEFAULT_CONFIG_PATH,
    always_update,
    set_update_on_next_restart,
    update_on_next_restart,
)

#: The three lines this module can print. The launchers match on the
#: FIRST WORD only (`pull` vs anything else), so the reason after it is
#: free to change without touching either script -- it exists to put "why
#: did this rig just move onto new code" in the launcher log, where an
#: operator reading a startup banner will actually see it.
PULL_ALWAYS = "pull always_update"
PULL_ONCE = "pull update_on_next_restart"
SKIP = "skip"


def decide(path: Path = DEFAULT_CONFIG_PATH) -> str:
    """Read both flags, clear the one-shot, and return the decision.

    `always_update` wins the naming race when both are set -- the rig
    would pull either way, and "this machine always pulls" is the more
    useful thing to read in a log than "someone pressed the button". The
    one-shot is still cleared in that case: a flag left set on an
    always-update rig would come back to life the day someone turns
    `always_update` off.
    """
    once = update_on_next_restart(path)
    if once:
        # Cleared even when the pull below is about to fail, and cleared
        # even on an always_update rig -- see the module docstring.
        set_update_on_next_restart(False, path)
    if always_update(path):
        return PULL_ALWAYS
    if once:
        return PULL_ONCE
    return SKIP


def main(argv: "list[str] | None" = None) -> int:
    """Print the decision. Exit code is 0 for all three answers.

    A non-zero exit is reserved for "this module could not answer at
    all", which is how the launchers tell a real failure (no venv, a
    half-installed checkout) from a plain `skip` -- they treat the two
    differently, because one is a rig behaving as configured and the
    other is a rig whose configuration could not be read.
    """
    del argv  # no flags: the config file is the whole interface
    print(decide())
    return 0


if __name__ == "__main__":  # pragma: no cover -- exercised as a subprocess
    sys.exit(main(sys.argv[1:]))
