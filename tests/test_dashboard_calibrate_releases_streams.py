"""Calibrate must not starve the page it runs from.

MEASURED ON AN IPAD, 2026-09-15. Everything worked until Calibrate was
pressed; from that moment every request in the page failed at once with
Safari's generic `TypeError: Load failed` — camera status, the audio
heartbeat, and the calibration re-read in the error handler too. It read
exactly like the rig had fallen over.

The rig was fine. Measured on it during a live calibration, with a stream
open: the POST returned in 10.02s and concurrent status polls were served
in 0.6ms. Nothing server-side was slow or refusing.

It was the browser's connection budget. The Config tab holds one
never-ending MJPEG connection per camera — three — plus the events
WebSocket, and WebKit allows about six per host. The calibrate POST is
the fifth, and it lasts as long as the calibration does. That leaves one
socket for the status poll, three overlay images and the heartbeat, so
they fail together for precisely the duration of the request.

So the handler closes the streams for the duration and restores them
after. That also stops three MJPEG encoders competing with the most
CPU-hungry operation the rig performs, which is worth having on its own.

These are string-level checks on rendered JS. Weak, and what is available
without a browser: the bug is invisible in Python, invisible in the JS as
written, and only appears when a real engine runs out of sockets.
"""
from __future__ import annotations

import re

import pytest

from opendarts.live.server import _render_dashboard_html


@pytest.fixture(scope="module")
def dashboard_js() -> str:
    """The whole rendered dashboard script."""
    html = _render_dashboard_html(3)
    return "\n".join(re.findall(r"<script>(.*?)</script>", html, re.S))


@pytest.fixture(scope="module")
def calibrate_handler(dashboard_js: str) -> str:
    """Just the Calibrate button's click handler."""
    start = dashboard_js.index("getElementById('btn-refresh-calib').onclick")
    end = dashboard_js.index("getElementById('btn-start').onclick")
    assert start < end
    return dashboard_js[start:end]


def test_streams_are_closed_on_page_unload(dashboard_js: str) -> None:
    """A reload/close/navigate must release the MJPEG stream sockets.

    A multipart/x-mixed-replace connection does not close cleanly on its
    own; left to the browser it lingers as a half-open socket against the
    ~6-per-origin pool. Across the many reloads and OD restarts a rig sees,
    enough strand that every later fetch to the hostname origin sits
    'pending' forever (the IP origin, a separate pool, looks fine). So the
    page must tear the streams down on pagehide.
    """
    assert "addEventListener('pagehide'" in dashboard_js, (
        "no pagehide handler -- a reload/close leaves the three preview "
        "MJPEG sockets to linger against the per-origin connection pool"
    )
    tail = dashboard_js[dashboard_js.index("addEventListener('pagehide'"):]
    handler = tail[:tail.index("});") + 3]
    assert "stopAllCamStreams()" in handler, (
        "pagehide handler does not close the camera streams"
    )


def test_streams_are_released_when_the_events_socket_drops(dashboard_js: str) -> None:
    """OD restarting is the recurring trigger for the stranded-socket wedge.

    A multipart MJPEG preview that ends server-side does not fire
    img.onerror -- the client socket lingers. The events WebSocket closing
    is the reliable 'OD went away' signal, so ws.onclose must release the
    previews so their sockets are freed rather than accumulated across every
    restart/deploy.
    """
    close = dashboard_js[dashboard_js.index("ws.onclose"):]
    close = close[:close.index("};") + 2]
    assert "stopAllCamStreams()" in close, (
        "ws.onclose does not release the camera streams -- each OD restart "
        "then strands three sockets against the connection pool"
    )


def test_streams_are_gated_on_window_focus_not_just_visibility(
    dashboard_js: str,
) -> None:
    """Two visible Config WINDOWS must not both stream.

    document.hidden is false for every visible window, so two side-by-side
    Config views each opened three never-ending MJPEG connections -- six
    against a browser budget of ~6 sockets PER ORIGIN, shared across every
    window and tab of the profile. That consumed the whole budget and every
    later fetch queued forever with no error, reading as a dead rig.
    Reproduced 2026-09-21 on both the hostname and the IP (the budget is
    keyed on scheme+host+port, so those are merely two separate pools --
    which is why 'it works on the IP' looked like DNS for hours).

    Gating on focus keeps the cost at one window's worth however many are
    open.
    """
    start = dashboard_js.index("function updateCameraFeeds()")
    body = dashboard_js[start:dashboard_js.index("\n}", start)]
    assert "document.hasFocus()" in body, (
        "camera previews are no longer gated on window focus -- two visible "
        "Config windows will both stream, exhausting the ~6-per-origin "
        "socket budget and wedging every fetch on the page"
    )
    for ev in ("'focus'", "'blur'"):
        assert "window.addEventListener(" + ev in dashboard_js, (
            f"no window {ev} listener -- focus gating cannot re-evaluate, so "
            "streams would not stop when the window loses focus (or restart "
            "when it regains it)"
        )


def test_unfocused_window_says_why_it_is_paused(dashboard_js: str) -> None:
    """A dark tile on a healthy rig has to explain itself.

    The tile must never show a frozen last frame (the stale-feed lie this
    file guards against elsewhere), but it also must not go dark silently --
    that reads as a broken camera.
    """
    start = dashboard_js.index("function updateCameraFeeds()")
    body = dashboard_js[start:dashboard_js.index("\n}", start)]
    assert "onConfig && !focused" in body and "paused" in body, (
        "an unfocused Config window drops its streams but no longer labels "
        "the tiles 'paused' -- an operator sees dark tiles on a working rig"
    )


def test_stop_all_cam_streams_closes_every_camera(dashboard_js: str) -> None:
    """The helper must actually abort each stream (clear src via
    stopCamStream), not merely hide the tiles."""
    start = dashboard_js.index("function stopAllCamStreams()")
    body = dashboard_js[start:dashboard_js.index("\n}", start)]
    assert "for (const c of CAM_IDS)" in body and "stopCamStream(c)" in body, (
        "stopAllCamStreams must call stopCamStream for every camera -- that "
        "is the call that clears img.src and aborts the MJPEG connection"
    )


def test_streams_are_closed_when_calibration_starts(calibrate_handler: str) -> None:
    """Three sockets have to be given back before the POST goes out.

    Without this the page keeps three never-ending MJPEG connections open
    across a request that can run for tens of seconds, and everything else
    it needs fails for the duration.
    """
    assert "stopCamStream" in calibrate_handler, (
        "the Calibrate handler no longer releases the camera streams. The "
        "Config tab holds one MJPEG connection per camera; with the "
        "WebSocket and this POST that exhausts WebKit's ~6-per-host budget "
        "and every other request in the page fails until it finishes."
    )
    before_fetch = calibrate_handler[:calibrate_handler.index("fetch('/api/calibration/refresh'")]
    assert "stopCamStream" in before_fetch, (
        "streams are released, but only after the POST is issued -- the "
        "socket pressure is during the request, so it must happen first"
    )


def test_streams_are_restored_in_a_finally(calibrate_handler: str) -> None:
    """A failed calibration must not leave every tile dark.

    They are closed unconditionally, so they have to be restored
    unconditionally -- including when the request throws, which is exactly
    the case this whole bug produced.
    """
    tail = calibrate_handler[calibrate_handler.index("finally"):]
    assert "updateCameraFeeds()" in tail, (
        "streams are closed for calibration but not restored in the "
        "`finally` -- an aborted or failed calibration would leave the "
        "previews dark until the next tab switch"
    )


def test_restore_reads_live_state_rather_than_a_remembered_list(
    calibrate_handler: str,
) -> None:
    """Restoring must honour a tab switch made during calibration.

    updateCameraFeeds() decides from current tab/visibility/capture state.
    Reinstating a list captured at click time would reopen streams on a tab
    the operator has since left -- reintroducing the same socket pressure
    somewhere it is not even visible.
    """
    tail = calibrate_handler[calibrate_handler.index("finally"):]
    assert "img.src" not in tail and "stream.mjpg" not in tail, (
        "the restore path looks like it reopens streams directly instead of "
        "going through updateCameraFeeds(), which is what consults the "
        "current tab and visibility state"
    )
