"""Where this rig keeps its own data.

Everything the product writes -- config, logs, throw packages, calibration
packages, frame-ring captures -- lives under ONE directory, `data/` beside
the checkout. OpenDarts runs from a checkout (see the README), so that is
the same folder the launcher pulls updates into, and one directory is one
thing to back up, point at another disk, or wipe.

`OPENDARTS_DATA_DIR` moves all of it somewhere else, which is how you put
the data on a bigger or faster disk than the one holding the code:

    OPENDARTS_DATA_DIR=/mnt/darts ./run.sh

Read once, at import: a path that changed under a running process would
leave half its files in the old place. The test suite redirects these
paths directly rather than through this variable (tests/conftest.py).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The rig's own data directory. `OPENDARTS_DATA_DIR` overrides it.
DATA_DIR = (Path(os.environ["OPENDARTS_DATA_DIR"]).expanduser().resolve()
            if os.environ.get("OPENDARTS_DATA_DIR")
            else REPO_ROOT / "data")

#: Scratch space for frames a request is mid-way through writing. Not
#: under DATA_DIR on purpose: it is throwaway, and a rig pointing DATA_DIR
#: at a slow network disk should not pay for it twice.
SCRATCH_ROOT = REPO_ROOT / "tmp"
