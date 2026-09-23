"""Retired diagnostic routes must not come back.

Each of these was removed deliberately: the probe routes once served
per-thread CPU and memory detail, and the two camera probes answered a
capture-format question that is settled. The rule they were removed
under is that a shipped rig should not carry diagnostic routes at all,
so this test is the alarm if one reappears -- adding a route back is a
decision, not an accident.

The measurement code itself lives in dev/instrumentation/, outside the
shipped tree, along with the tests that exercise it directly.
"""
from __future__ import annotations

from pathlib import Path

SERVER_PY = Path(__file__).resolve().parent.parent / "opendarts" / "live" / "server.py"

RETIRED_ROUTES = (
    "/api/threads",
    "/api/threads/cv2-threads",
    "/api/memory",
    "/api/client-debug-snapshot",
    "/api/diagnostics/raw-capture-probe",
    "/api/diagnostics/mf-mjpeg-probe",
    "/api/captures",
    "/api/events/recent",
    # The eight per-setting route pairs the config document replaced on
    # 2026-09-17. The owner's rule for a pre-ship API was explicit -- get
    # the contracts right, no backwards-compatibility aliases -- so an
    # alias reappearing here is exactly the decision that has to be made
    # deliberately rather than by a well-meaning patch.
    "/api/ad-config",
    "/api/camera-devices",
    "/api/store-packages",
    "/api/frame-ring",
    "/api/port",
    "/api/diagnostics",
    "/api/detection-time",
    "/api/idle-timeout",
)


def test_retired_diagnostic_routes_are_not_registered():
    src = SERVER_PY.read_text()
    for path in RETIRED_ROUTES:
        # Exact-path match: `"/api/frame-ring"` must not fire on the
        # capture-missed route that still lives one segment deeper.
        assert f'"{path}")' not in src, (
            f"{path} is reachable again. It was removed on purpose; if it is "
            f"genuinely needed, decide that deliberately and update this test."
        )


def test_server_has_not_grown_its_own_probe_plumbing():
    """The Windows memory probe lived in the measurement module, never in
    the server. A copy here would be the route coming back by another name."""
    assert "GetProcessMemoryInfo" not in SERVER_PY.read_text()
