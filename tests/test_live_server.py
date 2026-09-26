"""Tests for opendarts/live/server.py -- the FastAPI dashboard/API server.

Per docs/DESIGN.md's filesystem discipline, uses <repo>/tmp/ (not pytest's
built-in tmp_path, which lives under the system temp dir) for scratch
package roots, same pattern as tests/test_capture_replay.py.

enable_background_poll=False everywhere here -- the real background poll
(package-root polling; calibration has no poll loop at all anymore, see
AppState.refresh_calibration's own docstring -- it only updates on an
explicit call now) is exercised by running the server for real, not by
this fast/deterministic test suite; leaving it on here would make tests
depend on real camera hardware, which this suite must not require to
pass.

FRAME SOURCE, since this session's rewire: create_app()'s default is now
LOCAL direct-camera access.
None of these tests use TestClient as a context manager (`with
TestClient(app) as client:`), so create_app()'s lifespan -- and
therefore its automatic "open a real LocalCameraHub at startup" step --
never actually runs for any test here (confirmed: this whole suite still
completes in well under a second with zero camera hardware present).
Tests that need a working local-mode snapshot/calibration path inject a
`local_hub=` built against a monkeypatched cv2.VideoCapture (same
FakeVideoCapture pattern tests/test_local_capture.py already
established), exactly the dependency-injection seam create_app()'s
`local_hub` parameter exists for.
"""
from __future__ import annotations

import collections
import json
import os
import re
import signal
import shutil
import subprocess
import threading
import time
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

# fastapi/uvicorn are an optional dependency of this repo's .venv as of
# this writing (added to requirements.txt for opendarts/live/server.py, but
# not yet `pip install`-ed here -- see that module's own docstring and
# docs/DEPLOYMENT.md's "Remote retrieval" section for why this wasn't
# done silently). A hard ImportError at module scope would make pytest
# abort collecting the ENTIRE suite (not just this file) -- importorskip
# instead cleanly SKIPS just this module's tests when the dependency is
# missing, so the rest of the suite still runs. Once fastapi/uvicorn are
# installed, these tests run for real, not skip.
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient # noqa: E402

from tests.support.synthetic import make_camera_matrix, make_ring_camera
from opendarts.capture import frame_ring
from opendarts.capture.throw_package import save_ad_ground_truth, save_throw_package
from opendarts.live import local_capture
from opendarts.live import server as server_module
from opendarts.live.ad_ground_truth import AdGroundTruth
from opendarts.live.board_status import (
    BOARD_STATUS_READY,
    BOARD_STATUS_TAKEOUT,
    BOARD_STATUS_UNKNOWN,
)
from opendarts.live.server import create_app, discover_packages
from opendarts.pipeline import CameraCalibration, ScoreResult

@pytest.fixture()
def package_root(tmp_path):
    """<tmp_path>/run/packages -- note the extra level. Each test needs its
    OWN parent folder, because session_throw_counters lives at
    package_root.parent, not under package_root (see
    handle_ready_to_capture()'s 2026-08-17 comment for why). A shared
    parent leaked counters between tests (2026-08-22), and under `pytest -n
    auto` -- which spreads this file's tests across workers -- two tests
    raced on the same counter files and failed intermittently
    (2026-09-17)."""
    return tmp_path / "run" / "packages"


def _synthetic_calibration(seed: int = 0) -> CameraCalibration:
    camera_matrix = make_camera_matrix()
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)
    return CameraCalibration(
        camera_matrix=cam.camera_matrix,
        dist_coeffs=cam.dist_coeffs,
        rvec=cam.rvec,
        tvec=cam.tvec,
        pnp_result=None,
        landmark_spread_ok=True,
    )


def _write_sample_package(
    dest_dir: Path,
    session: str = "session-test",
    *,
    sector: str = "S20",
    ring: str = "single",
    board_xy_mm: tuple[float, float] = (12.3, -4.5),
) -> Path:
    """Writes one real, complete throw package via the same
    save_throw_package() function the live daemon uses -- not a hand-
    rolled fake meta.json/result.json, so this test proves the server
    reads the ACTUAL on-disk format, not a format this test invented."""
    calibrations = {0: _synthetic_calibration()}
    bg_frames = {0: np.zeros((24, 32, 3), dtype=np.uint8)}
    dart_frames = {0: np.full((24, 32, 3), 255, dtype=np.uint8)}
    result = ScoreResult(
        ok=True,
        sector=sector,
        ring=ring,
        board_xy_mm=board_xy_mm,
        triangulation=None,
        n_cameras_used=1,
        reason="",
        max_ray_disagreement_mm=None,
    )
    return save_throw_package(
        dest_dir=dest_dir,
        session=session,
        bg_frames_bgr=bg_frames,
        dart_frames_bgr=dart_frames,
        calibrations=calibrations,
        result=result,
    )


def _attach_ad_gt(
    pkg_dir: Path,
    *,
    matched: bool = True,
    sector: str | None = "S20",
    ring: str | None = "single",
    tip_xy_mm: tuple[float, float] | None = (12.3, -4.5),
    match_reason: str = "ok",
) -> AdGroundTruth:
    """Writes ad_ground_truth.json straight onto disk via the real
    save_ad_ground_truth() (the same function every ground-truth writer
    calls) -- not a hand-rolled JSON file -- so these tests exercise
    the real on-disk shape opendarts.capture.throw_package.load_ad_ground_truth()
    reads back."""
    gt = AdGroundTruth(
        matched=matched,
        match_reason=match_reason,
        ad_base_url="http://localhost:3180",
        fetched_at_utc="2026-08-12T00:00:00+00:00",
        opendarts_captured_at_utc="2026-08-12T00:00:00+00:00",
        staleness_sec=1.0,
        window_sec=12.0,
        sector=sector,
        ring=ring,
        tip_xy_mm=tip_xy_mm,
        ad_method="UnanimousCam",
    )
    save_ad_ground_truth(pkg_dir, gt)
    return gt


# --------------------------------------------------------------------------
# / -- dashboard HTML
# --------------------------------------------------------------------------

def test_root_returns_200_html(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    resp = client.get("/?ui=classic")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "opendarts live dashboard" in resp.text


def test_root_html_has_four_real_tabs(package_root):
    """The tab set is Scoring / Engines / Config / Info.

    History, because the name of this test kept outliving the tabs it
    checks: Cameras+Calibration merged into one tab, that tab was renamed
    Config, the old Config was renamed Info once its only mutable control
    moved to the sidebar, Info was deleted on 2026-09-13 -- and restored
    the same day with entirely different content.

    The deletion and the restoration were both right. The OLD Info tab
    printed read-only copies of config values, so one subject lived in
    two places and its note claimed all eight "are set in
    data/config.json" when host, package root and frame source never
    were. What is there now is build, machine, versions and what this
    process is wired to -- facts about the rig that duplicate no setting.
    Config is what you edit; Info is what this rig IS.

    Checks the tab buttons AND their panels, not just the word "tab"
    somewhere, and that every tab this dashboard has genuinely dropped is
    gone rather than unlinked."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    html = client.get("/?ui=classic").text

    for tab in ("scoring", "engines", "config", "info"):
        assert f'data-tab="{tab}"' in html
        assert f'id="tab-{tab}"' in html
    # Exactly four -- a fifth reappearing is the regression here, and
    # "the four we want are present" would not catch it.
    assert html.count('class="tab active" data-tab=') + html.count('class="tab" data-tab=') == 4

    # Every retired tab must be gone entirely, not merely hidden -- no
    # stray nav button, no stray panel.
    for gone in ("calibration", "cameras"):
        assert f'data-tab="{gone}"' not in html
        assert f'id="tab-{gone}"' not in html

    # The camera cards survived the merge into Config: each camera's
    # snapshot tile, fetch/status detail, AND its calibration badge/
    # metric/note all live in ONE card -- checking they're all present
    # confirms the merge actually happened, not just a renamed empty tab.
    assert 'id="cam-img-0"' in html
    assert 'id="cam-detail-0"' in html
    assert 'id="calib-badge-0"' in html

    # And the rig facts live in Info, NOT in Config -- the whole point of
    # restoring the tab. A settings page and a facts page are different
    # subjects and this is what keeps them apart.
    info = html[html.index('id="tab-info"'):html.index('id="tab-engines"')]
    config = html[html.index('id="tab-config"'):html.index('id="tab-info"')]
    assert 'id="diagnostics-tbody"' in info and 'About this rig' in info
    assert 'id="diagnostics-tbody"' not in config
    assert 'id="calib-err-0"' in html
    assert 'id="calib-note-0"' in html
    assert 'id="btn-refresh-calib"' in html
    assert "/api/calibration/refresh" in html
    # "cam0" must not be printed twice per card -- exactly one cam-label per camera.
    assert html.count('cam0</span>') == 1
    # Scoring tab: recent-throws table including the board_xy column.
    assert 'id="packages-tbody"' in html
    assert "board xy (mm)" in html
    # What the Info tab's read-only rows became: a Diagnostics table at
    # the bottom of Config, still labelled read-only.
    assert 'id="diagnostics-tbody"' in html
    assert "read-only" in html.lower()
    # The old id must be gone, not left behind rendering into nothing.
    assert 'id="config-tbody"' not in html


# --------------------------------------------------------------------------
# Broken-image fallback on a stopped/unreachable camera. Real incident
# (in the snapshot-polling era, refreshSnapshots(); the handlers now live
# in updateCameraFeeds(), the 2026-09-11 MJPEG-streaming successor):
# img.onerror used to only update the small cam-fetch-status-* text label
# ("unreachable") -- it never touched img.src or hid the <img> element
# itself, so a refused camera fetch (cameras not
# started/stopped, a normal expected state now that cameras don't
# auto-open at process startup) left the browser's own native broken-image
# icon sitting in the tile. JS-source-assertion tests (no real browser in
# this suite, same style as the status-pill/button-feedback tests above):
# prove the shipped error handler swaps in a REAL placeholder element (not
# just a status-text update), and that a subsequent successful load
# restores the real image -- both directions.
# --------------------------------------------------------------------------


def test_root_html_has_a_real_placeholder_element_per_camera(package_root):
    """The placeholder must be a real, distinct DOM element per camera --
    not just a CSS class name mentioned somewhere -- present for every
    configured camera, and VISIBLE by default.

    REVERSED 2026-09-13, and the original reasoning is worth recording
    because it was carefully argued and wrong. This test used to require
    `hidden` in the initial markup, to avoid "a default-visible flash" of
    the placeholder before the first updateCameraFeeds() decided.

    That traded the flash for something worse. An `<img>` with no `src` is
    not blank: the browser paints its alt text next to a broken-image
    icon. So hiding the placeholder did not show nothing, it showed
    "camera 0 preview" and a broken icon -- the rig looking broken rather
    than not-yet-started. And on any path where updateCameraFeeds() never
    sets a src, that state was permanent, not a flash. Confirmed live: a
    backgrounded tab takes exactly that path, because the feed loop
    deliberately refuses to open streams while `document.hidden`.

    The honest empty state belongs in the markup, not in whatever
    JavaScript manages to run.
    """
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert 'id="cam-placeholder-0"' in html
    assert 'class="cam-placeholder"' in html
    idx = html.index('id="cam-placeholder-0"')
    tag_end = html.index(">", idx)
    assert "hidden" not in html[idx:tag_end], (
        "the placeholder must start visible -- hiding it exposes the <img>'s "
        "alt text and broken-image icon underneath"
    )


def test_camera_feed_error_handler_swaps_in_the_placeholder_not_just_status_text(
    package_root,
):
    """THE real fix, proven at the JS-source level: img.onerror must
    actually hide the <img> and un-hide the placeholder, not just update
    cam-fetch-status-*'s text (the old, incomplete behavior that left the
    browser's native broken-image icon showing underneath the text
    label). Originally written against the snapshot-polling
    refreshSnapshots(); the requirement survived the 2026-09-11 move to
    MJPEG streaming unchanged -- a refused stream (capture not running
    -> 503) still fires onerror on the <img>, and the placeholder swap
    is still what stands between the operator and a broken-image
    icon."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "function updateCameraFeeds()" in html
    start_idx = html.index("function updateCameraFeeds()")
    end_idx = html.index("\n}", start_idx)
    body = html[start_idx:end_idx]

    assert "img.onerror" in body
    onerror_idx = body.index("img.onerror")
    onerror_block_end = body.index("};", onerror_idx)
    onerror_block = body[onerror_idx:onerror_block_end]
    # The status-text update must still be there (not removed, just no
    # longer the ONLY thing that happens)...
    assert "status.textContent" in onerror_block
    # ...AND the <img> itself must be hidden, AND the placeholder shown.
    assert "img.style.display = 'none'" in onerror_block
    assert "placeholder.hidden = false" in onerror_block


def test_camera_feed_load_handler_restores_the_real_image(package_root):
    """The OTHER direction, just as real a requirement: a camera that
    starts producing frames again after an earlier failed attempt must
    switch BACK to showing the real <img>, not stay stuck on the
    placeholder forever. updateCameraFeeds() retries stream connections
    on CAMERA_FEED_TICK_MS and binds onload/onerror fresh per attempt --
    this proves the onload side actually reverses what onerror did."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    start_idx = html.index("function updateCameraFeeds()")
    end_idx = html.index("\n}", start_idx)
    body = html[start_idx:end_idx]

    assert "img.onload" in body
    onload_idx = body.index("img.onload")
    onload_block_end = body.index("};", onload_idx)
    onload_block = body[onload_idx:onload_block_end]
    assert "status.textContent" in onload_block
    # img.style.display reset back to visible ('' clears the inline
    # 'none' onerror set), and the placeholder re-hidden.
    assert "img.style.display = ''" in onload_block
    assert "placeholder.hidden = true" in onload_block

    # Both handlers must be bound fresh on every connection attempt, not
    # just once at page load -- otherwise a camera that fails then
    # recovers would still be governed by a stale, one-time-bound
    # onerror/onload pair from its very first attempt.
    assert body.count("img.onload") == 1
    assert body.count("img.onerror") == 1
    # The raw preview is the MJPEG stream, not a re-fetched snapshot.
    assert "img.src = '/api/cameras/' + c + '/stream.mjpg?t=' + Date.now()" in body


# --------------------------------------------------------------------------
# Split-panel layout restructure: a split panel (left control bar,
# static header bar, tabbed right window). Ported
# the dashboard's own structure:
# css/admin.css. These tests prove the STRUCTURE actually
# exists (sticky header, CSS-grid layout, sidebar with real-only buttons,
# stage holding the tabs) -- test_root_html_has_three_real_tabs above
# already proves the tab/panel content itself is intact post-relocation.
# --------------------------------------------------------------------------


def test_root_html_has_a_sticky_header(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert '<header class="top">' in html
    # The sticky/flex-0 CSS rule for header.top -- proves this isn't just
    # a plain <header>, it's the real "stays visible while the stage
    # scrolls" element header.top is.
    assert "header.top {" in html
    assert "position: sticky; top: 0; z-index: 10;" in html


def test_root_html_has_od_style_grid_layout(package_root):
    """main.layout is the real CSS-grid two-column body (sidebar +
    stage) -- the `display: grid; grid-template-columns: var(--side-w)
    1fr;` pattern, not a flat single-column page."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert '<main class="layout">' in html
    assert "main.layout {" in html
    assert "grid-template-columns: var(--side-w) 1fr;" in html
    assert "--side-w:" in html


def test_root_html_has_sidebar_with_controls_heading(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert '<aside class="sidebar">' in html
    assert '<section class="side-block">' in html
    assert "<h2>Controls</h2>" in html
    assert '<div class="btn-grid">' in html


def test_sidebar_start_stop_reset_calibrate_are_all_real_and_enabled(package_root):
    """CHANGED 2026-08-12: Start/Stop/Reset/Calibrate (the button
    SET in the Controls block) are now ALL real, wired
    actions -- none of the four should be `disabled` anymore, and each
    must still show its correct label (this project's "never fake
    capability that doesn't exist" rule, applied in the OTHER direction
    now: a real capability must not be left looking disabled either)."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    for btn_id, label in (
        ("btn-start", "Start"),
        ("btn-stop", "Stop"),
        ("btn-reset", "Reset"),
        ("btn-refresh-calib", "Calibrate"),
    ):
        marker = f'id="{btn_id}"'
        assert marker in html
        start = html.index(marker)
        tag_start = html.rindex("<button", 0, start)
        tag_end = html.index(">", start)
        tag = html[tag_start:tag_end]
        assert "disabled" not in tag, f"{btn_id} must be a real, enabled button now"
        assert 'title="' in tag, f"{btn_id} should still explain what it does via a title tooltip"
        after_tag = html[tag_end + 1 :]
        rendered_label = after_tag.split("<", 1)[0].strip()
        assert rendered_label == label


def test_sidebar_calibrate_button_is_wired_to_the_refresh_endpoint(package_root):
    """Calibrate (one of now FOUR real sidebar actions, see
    test_sidebar_start_stop_reset_calibrate_are_all_real_and_enabled
    above) must be enabled (no `disabled` attribute) and wired to the
    real POST /api/calibration/refresh endpoint, moved here (not
    duplicated) from its old home inside the Cameras tab."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    marker = 'id="btn-refresh-calib"'
    assert marker in html
    start = html.index(marker)
    tag_start = html.rindex("<button", 0, start)
    tag_end = html.index(">", start)
    tag = html[tag_start:tag_end]
    assert "disabled" not in tag
    assert "/api/calibration/refresh" in html
    # Exactly one Calibrate button in the whole page -- no leftover
    # duplicate still sitting in the Cameras tab after the move.
    assert html.count('id="btn-refresh-calib"') == 1
    # The old Cameras-tab wording pointing to itself must be gone; the
    # tab's own explanatory text now points at the sidebar instead.
    assert "click Refresh below" not in html
    assert "Calibrate in the sidebar" in html


def test_import_ad_omitted_not_faked(package_root):
    """There is no calibration-import button: opendarts's calibration is
    a from-scratch multi-camera PnP pipeline, so there is nothing to
    import. It must be genuinely absent, not present-but-disabled.

    Checked against the html with HTML COMMENTS STRIPPED: what must be
    absent is rendered UI, not words in an explanatory comment."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    rendered = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    assert "Import" not in rendered
    # The id is checked against the FULL html: no comment mentions it, and
    # a commented-out button element is still a button someone can
    # uncomment, unlike a sentence of prose.
    assert "btn-import-ad" not in html


def test_root_html_has_stage_holding_the_tabs(package_root):
    """The tab bar + tab panels must live INSIDE <section class="stage">
    (the right column), not floating at the top level the way the old
    flat <main> used to hold them."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    stage_start = html.index('<section class="stage">')
    stage_end = html.index("</section>", stage_start)
    stage_html = html[stage_start:stage_end]

    assert '<nav class="tabs">' in stage_html
    # The live tab set, 2026-09-13 (Info folded into Config -- see
    # test_root_html_has_three_real_tabs for that history).
    for tab in ("scoring", "engines", "config"):
        assert f'data-tab="{tab}"' in stage_html
        assert f'id="tab-{tab}"' in stage_html
    # The sidebar's own controls must NOT be inside the stage section --
    # proves this is a real two-column split, not the sidebar nested
    # inside the stage or vice versa.
    assert '<h2>Controls</h2>' not in stage_html
    assert 'id="btn-refresh-calib"' not in stage_html # moved to the sidebar, not duplicated here


def test_all_preserved_tab_functionality_survives_the_relocation(package_root):
    """One consolidated check that every real feature built across prior
    sessions -- status pill, WebSocket-driven cameras/scoring/config
    tabs, AD comparison, the view filter, row numbers, mark-AD-wrong --
    is still present verbatim after the structural move, not silently
    dropped or rewritten. Narrower single-feature tests already exist
    elsewhere in this file (status pill: test_root_html_has_status_pill_
    wired_to_all_real_throw_states; view filter:
    test_scoring_tab_html_has_start_new_session_view_controls; row
    numbers: test_scoring_table_has_row_number_column; mark-AD-wrong:
    test_scoring_tab_html_has_ad_wrong_controls_and_summary) -- this is a
    single broad smoke test tying them together post-relocation, not a
    replacement for those."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    # Status pill, still in the header, still wired.
    assert 'class="status-pill" id="status-pill"' in html
    assert "TRIGGER_STATE" in html

    # Cameras tab: snapshot + calibration merge intact.
    assert 'id="cam-img-0"' in html
    assert 'id="calib-badge-0"' in html

    # Scoring tab: AD comparison (now a per-throw AD row + one row per
    # engine, 2026-08-13 restructure -- see test_scoring_tab_html_has_
    # ad_comparison_columns for the detailed check), view filter, row
    # numbers, mark-AD-wrong.
    assert 'class="ad-row"' in html
    # Opening-quote-only, deliberately: the primary engine appends a
    # second class ('engine-row primary'), so pinning the closing quote
    # asserted that no engine row may ever carry another class -- which
    # was never the point. What matters is that the row is still
    # rendered with its class at all.
    assert 'class="engine-row' in html
    assert 'id="btn-new-session-view"' in html
    assert 'id="engine-tally-bar"' in html
    assert "ad-wrong-btn" in html
    assert "#</th>" in html or "<th>#</th>" in html

    # The read-only rows that used to be the Info tab: now the
    # Diagnostics table at the bottom of Config (2026-09-13), still
    # labelled read-only, still fed by the same /api/state config block.
    assert 'id="diagnostics-tbody"' in html
    assert "read-only" in html.lower()

    # WebSocket wiring untouched by the HTML move.
    with TestClient(app).websocket_connect("/api/events") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "HELLO"


# --------------------------------------------------------------------------
# /api/state -- valid JSON, honest about what isn't wired
# --------------------------------------------------------------------------

def test_api_state_returns_valid_json(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    resp = client.get("/api/state")
    assert resp.status_code == 200
    body = resp.json()

    assert body["package_root"] == str(package_root)
    # No live daemon process/IPC exists -- /api/state must say so plainly
    # rather than inventing a trigger state (see AppState.state_dict()).
    assert body["trigger"]["available"] is False
    assert "calibration" in body and "cameras" in body["calibration"]
    # enable_background_poll=False -> never refreshed -> honestly "null".
    assert body["calibration"]["checked_at_utc"] is None
    # Default frame source, since this session's rewire, is local direct
    # camera access -- not the HTTP path.
    assert body["frame_source"] == "local"




def test_api_state_config_section_reflects_real_settings(package_root):
    """The Config tab reads this section verbatim -- must contain the
    actual settings this server was built with (package root, base
    URL, poll interval, host/port when given), not fabricated ones.
    calibration_poll_interval_s is GONE (2026-08-12: the automatic
    calibration poll it configured was removed entirely, not just
    hidden -- see AppState.refresh_calibration's own docstring)."""
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        package_poll_interval_s=7.5,
        host="0.0.0.0",
        port=18799,
    )
    client = TestClient(app)
    cfg = client.get("/api/state").json()["config"]

    assert cfg["package_root"] == str(package_root)
    assert cfg["package_poll_interval_s"] == 7.5
    assert "calibration_poll_interval_s" not in cfg
    assert cfg["host"] == "0.0.0.0"
    assert cfg["port"] == 18799
    assert cfg["frame_source"] == "local"
    # Real constant from opendarts.live.capture_daemon, not a made-up number
    # -- see server.py's own import of POLL_INTERVAL_SECONDS.
    from opendarts.live.capture_daemon import POLL_INTERVAL_SECONDS

    assert cfg["capture_loop_poll_interval_s"] == POLL_INTERVAL_SECONDS
    assert cfg["live_events_enabled"] is False
    # No calibration_store was passed to create_app() here -- honestly
    # False, a manual recalibrate in this standalone configuration can
    # only ever update this dashboard's own display.
    assert cfg["calibration_store_wired"] is False


def test_api_state_config_host_port_default_to_none_when_not_given(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    cfg = TestClient(app).get("/api/state").json()["config"]
    assert cfg["host"] is None
    assert cfg["port"] is None


# --------------------------------------------------------------------------
# AppState._refresh_calibration_blocking -- local mode has NO dependency
# guessed True/False), and degrades gracefully (no raise) when the hub
# isn't initialized yet.
# --------------------------------------------------------------------------


def test_refresh_calibration_local_mode_degrades_gracefully_without_a_hub(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    assert state.hub is None # lifespan never ran (see module docstring)

    data = state._refresh_calibration_blocking() # noqa: SLF001 -- test needs the blocking call directly

    assert data["calibrations"] == {}
    assert "calibration_error" in data


def test_refresh_calibration_failure_trims_the_heap_after_dropping_the_exception(
    package_root, monkeypatch
):
    """A failed calibration's frames stay pinned by its traceback until the
    exception is gone, so bootstrap_calibrations()'s own trim cannot free
    them -- the refresh path must trim again once it has swallowed it. See
    opendarts.live.heap_trim."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    state.hub = object() # non-None, and no .configs, so the readiness poll is skipped

    def failing_bootstrap(*a, **k):
        raise RuntimeError("simulated refusal")

    trims: list[str] = []
    monkeypatch.setattr(server_module, "bootstrap_calibrations", failing_bootstrap)
    monkeypatch.setattr(server_module, "release_freed_heap", trims.append)

    data = state._refresh_calibration_blocking() # noqa: SLF001 -- test needs the blocking call directly

    assert data["calibration_error"] == "simulated refusal"
    assert data["raw_calibrations"] == {}
    assert trims == ["failed calibration"]


# --------------------------------------------------------------------------
# /api/packages -- reflects what's actually on disk, via the real
# save_throw_package() writer, not a hand-rolled fixture.
# --------------------------------------------------------------------------

def test_api_packages_exposes_ad_latency_ms(package_root):
    """`ad_latency_ms` is AD's answer arrival minus ours, in signed ms,
    read straight off the package's own ad_ground_truth.json. Negative
    means AD's answer arrived first. Throw-level, never per-engine: the engines
    run concurrently against one capture and share one answer instant."""
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    gt = throw_dir / "ad_ground_truth.json"
    gt.write_text(json.dumps({
        "schema": "ad-ground-truth-v2", "matched": True,
        "staleness_sec": -0.018219, "sector": "20", "ring": "single",
    }))

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    pkg = client.get("/api/packages").json()[0]
    assert pkg["ad_latency_ms"] == -18.2


def test_api_packages_ad_latency_ms_is_none_without_ad_truth(package_root):
    """No AD ground truth on disk -> the key is present and honestly
    None. It must never degrade to 0.0, which would read as "AD and we
    answered simultaneously" -- a real measurement, not a missing one."""
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    pkg = client.get("/api/packages").json()[0]
    assert "ad_latency_ms" in pkg
    assert pkg["ad_latency_ms"] is None


def test_api_packages_reflects_saved_package(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    resp = client.get("/api/packages")
    assert resp.status_code == 200
    packages = resp.json()

    assert len(packages) == 1
    pkg = packages[0]
    assert pkg["session"] == "session-test"
    assert pkg["throw_id"] == "throw_1"
    assert pkg["path"] == str(throw_dir)
    assert pkg["ok"] is True
    assert pkg["sector"] == "S20"
    assert pkg["ring"] == "single"
    # board_xy_mm is what the Scoring tab's "board xy (mm)" column reads --
    # matches the (12.3, -4.5) ScoreResult _write_sample_package() built.
    assert pkg["board_xy_mm"] == [12.3, -4.5]
    assert pkg["captured_at_utc"] # non-empty ISO timestamp


def test_api_packages_empty_when_no_package_root(package_root):
    # package_root fixture creates the *name* but not the directory --
    # discover_packages() (and therefore /api/packages) must degrade to
    # an empty list rather than raising when nothing has been saved yet.
    assert not package_root.exists()
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    resp = client.get("/api/packages")
    assert resp.status_code == 200
    assert resp.json() == []


def test_discover_packages_sorts_newest_first(package_root):
    _write_sample_package(package_root / "s1" / "throw_a")
    import time

    time.sleep(0.01)
    _write_sample_package(package_root / "s1" / "throw_b")

    packages = discover_packages(package_root)
    assert len(packages) == 2
    assert packages[0]["captured_at_utc"] >= packages[1]["captured_at_utc"]


# --------------------------------------------------------------------------
# /api/cameras/{cam_id}/snapshot.png
#
# A hub is injected via create_app(local_hub=...) using the same
# monkeypatch-cv2.VideoCapture pattern tests/test_local_capture.py
# already established (FakeVideoCapture below is a smaller, local copy
# of that file's own fixture -- kept self-contained here rather than
# importing test internals across files).
# --------------------------------------------------------------------------


class FakeVideoCapture:
    """Minimal stand-in for cv2.VideoCapture -- always opens, always
    returns a fixed-size solid-color frame. See
    tests/test_local_capture.py's own FakeVideoCapture for the fuller
    version this is trimmed from (backend-fallback/negotiated-size
    behavior isn't this file's concern; local_capture.py's own test
    suite already covers that)."""

    def __init__(self, device, backend) -> None:
        self.device = device
        self.backend = backend

    def isOpened(self) -> bool: # noqa: N802 -- matches cv2's own method name
        return True

    def release(self) -> None:
        pass

    def set(self, prop: int, value: float) -> bool:
        return True

    def get(self, prop: int) -> float:
        return 0.0

    def read(self):
        return True, np.full((48, 64, 3), 128, dtype=np.uint8)


@pytest.fixture(autouse=True)
def _stop_pump_threads_after_every_test():
    """Same cleanup fixture as tests/test_local_capture.py's own (see
    that file's copy for the fuller explanation) -- added alongside the
    2026-08-12 pump-thread architecture change
    (opendarts/live/local_capture.py's LocalCameraHub now starts a real
    background thread + ThreadPoolExecutor in open_all()).
    _open_fake_local_hub() below builds a real LocalCameraHub directly
    (not via create_app()'s own lifespan, which never runs in this file
    -- see module docstring), so nothing else in this suite would ever
    call close_all() on it; left alone this leaks one busy-spinning
    daemon pump thread per test using it for the rest of the pytest
    process (observed for real: noisy 'cannot schedule new futures after
    interpreter shutdown' tracebacks from orphaned pump threads hitting
    an already-shut-down ThreadPoolExecutor at process exit, harmless to
    test results but real, avoidable noise/waste)."""
    created: list[local_capture.LocalCameraHub] = []
    original_init = local_capture.LocalCameraHub.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    local_capture.LocalCameraHub.__init__ = _tracking_init # type: ignore[method-assign]
    try:
        yield
    finally:
        local_capture.LocalCameraHub.__init__ = original_init # type: ignore[method-assign]
        for hub in created:
            hub.close_all()


class WideFakeVideoCapture(FakeVideoCapture):
    """FakeVideoCapture at a REALISTIC 1280x720 instead of 64x48.

    Needed by the overlay-layer tests specifically. The 64x48 fake is
    narrower than MJPEG_MAX_WIDTH, so the preview downscale is a no-op on
    it -- which is exactly why the alignment bug (an overlay rendered at
    the 960-wide preview size while the calibration projects into the
    camera's native 1280-wide space) sailed through a green suite and was
    caught by eye on the rig. Any test asserting where the overlay lands
    has to run at a width the downscale actually changes.
    """

    def read(self):
        return True, np.full((720, 1280, 3), 128, dtype=np.uint8)


def _open_wide_fake_local_hub(monkeypatch) -> local_capture.LocalCameraHub:
    monkeypatch.setattr(cv2, "VideoCapture", WideFakeVideoCapture)
    hub = local_capture.LocalCameraHub(configs=[local_capture.CameraConfig(device=0)])
    hub.open_all()
    return hub


def _open_fake_local_hub(monkeypatch) -> local_capture.LocalCameraHub:
    monkeypatch.setattr(cv2, "VideoCapture", FakeVideoCapture)
    hub = local_capture.LocalCameraHub(configs=[local_capture.CameraConfig(device=0)])
    hub.open_all()
    return hub


def test_camera_snapshot_local_mode_serves_real_png_from_injected_hub(package_root, monkeypatch):
    hub = _open_fake_local_hub(monkeypatch)
    app = create_app(package_root=package_root, enable_background_poll=False, local_hub=hub)
    client = TestClient(app)

    resp = client.get("/api/cameras/0/snapshot.png")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n" # real PNG signature


def test_camera_snapshot_local_mode_returns_502_when_hub_not_initialized(package_root):
    """No local_hub injected and enable_background_poll=False (which also
    means lifespan never runs for these non-context-manager TestClient
    calls -- see this file's module docstring) -- state.hub stays None,
    and the endpoint must degrade gracefully, not crash."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    resp = client.get("/api/cameras/0/snapshot.png")
    assert resp.status_code == 502
    body = resp.json()
    assert body["ok"] is False
    assert body["cam"] == 0
    assert "not initialized" in body["reason"]




# --------------------------------------------------------------------------
# /api/cameras/{cam_id}/overlay.png -- the calibration confirmation
# overlay (opendarts.geometry.board_overlay.draw_calibration_overlay()),
# the board-overlay drawing contract:
# draw_board_overlay() + pipeline/detector.py::get_overlay_jpeg(). The
# real geometry correctness proof lives in tests/test_board_overlay.py
# (known calibration, known board point, exact pixel-location checks) --
# these tests are specifically the SERVER-side wiring: does the route
# reach a real frame + the real live calibration, degrade honestly when
# either is missing, and -- the explicit requirement this task was given
# ("make sure to clear it when re-calibrating") -- does swapping the
# CalibrationStore's held calibration between two requests actually
# change what the very next request renders, with no caching in between.
# --------------------------------------------------------------------------


def _ring_calibration(index: int, n_cameras: int = 3):
    """A real ring-camera CameraCalibration with a KNOWN pose (same
    tests.support.synthetic fixture already imported above) --
    varying `index` gives a genuinely different camera pose, so two
    calls with different indices render visibly different overlays.
    camera_matrix is sized for FakeVideoCapture's own 64x48 fake frame
    (`_open_fake_local_hub` above) -- a camera_matrix built for the
    default 1280x720 would center its principal point far outside a
    64x48 canvas and project every board point off-frame, so the
    overlay would draw real geometry that simply never lands inside the
    visible pixels these tests inspect."""
    camera_matrix = make_camera_matrix(image_width=64, image_height=48)
    cam = make_ring_camera(index, n_cameras=n_cameras, camera_matrix=camera_matrix)
    return CameraCalibration(
        camera_matrix=cam.camera_matrix,
        dist_coeffs=cam.dist_coeffs,
        rvec=cam.rvec,
        tvec=cam.tvec,
        pnp_result=None,
        landmark_spread_ok=True,
    )


def test_camera_overlay_local_mode_returns_502_when_hub_not_initialized(package_root):
    """Same honest-502 degrade as snapshot.png -- no frame, nothing to
    draw an overlay on top of."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    resp = client.get("/api/cameras/0/overlay.png")
    assert resp.status_code == 502
    body = resp.json()
    assert body["ok"] is False
    assert body["cam"] == 0


def test_camera_overlay_without_a_calibration_falls_back_to_the_plain_snapshot(
    package_root, monkeypatch
):
    """No calibration_store wired at all -- there is nothing honest to
    draw (see draw_calibration_overlay()'s own precondition), so the
    route must still succeed and serve the real, unmodified snapshot
    frame rather than erroring or fabricating an overlay."""
    hub = _open_fake_local_hub(monkeypatch)
    app = create_app(package_root=package_root, enable_background_poll=False, local_hub=hub)
    client = TestClient(app)

    snap_resp = client.get("/api/cameras/0/snapshot.png")
    overlay_resp = client.get("/api/cameras/0/overlay.png")

    assert overlay_resp.status_code == 200
    assert overlay_resp.headers["content-type"] == "image/png"
    assert overlay_resp.content[:8] == b"\x89PNG\r\n\x1a\n"
    # Same underlying frame, no calibration to overlay against -> pixel-
    # identical to the plain snapshot once both are decoded (re-encoding
    # can legitimately change the exact PNG BYTES even for identical
    # pixels, so compare decoded arrays, not raw content).
    snap_arr = cv2.imdecode(np.frombuffer(snap_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    overlay_arr = cv2.imdecode(
        np.frombuffer(overlay_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert np.array_equal(snap_arr, overlay_arr)


def test_camera_overlay_with_a_live_calibration_draws_real_geometry_on_top_of_the_frame(
    package_root, monkeypatch
):
    from opendarts.live.capture_daemon import CalibrationStore

    hub = _open_fake_local_hub(monkeypatch)
    calibration_store = CalibrationStore({0: _ring_calibration(0)}, source="startup")
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        calibration_store=calibration_store,
    )
    client = TestClient(app)

    snap_resp = client.get("/api/cameras/0/snapshot.png")
    overlay_resp = client.get("/api/cameras/0/overlay.png")

    assert overlay_resp.status_code == 200
    assert overlay_resp.headers["content-type"] == "image/png"
    snap_arr = cv2.imdecode(np.frombuffer(snap_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    overlay_arr = cv2.imdecode(
        np.frombuffer(overlay_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert overlay_arr.shape == snap_arr.shape
    assert not np.array_equal(overlay_arr, snap_arr), (
        "a real, live calibration was wired for cam0 -- the overlay route must have "
        "actually drawn calibration geometry on top of the raw frame, not just echoed it"
    )


def test_camera_overlay_reflects_the_current_live_calibration_with_no_stale_caching(
    package_root, monkeypatch
):
    """THE real requirement this task was given, verbatim: "make sure to
    clear it when re-calibrating" -- swap the SAME CalibrationStore
    object's held calibration (calling .set() the exact way
    /api/calibration/refresh's real wiring, and the auto-calibrate-on-
    Start path, already do -- see AppState.refresh_calibration()) between
    two overlay requests and assert the second request's rendered pixels
    actually reflect the NEW calibration, not a cached render of the old
    one. draw_calibration_overlay() takes zero milliseconds of caching
    anywhere in this route (see its own docstring) -- this is the
    end-to-end proof that design choice actually delivers the required
    behavior, not just a claim about it."""
    from opendarts.live.capture_daemon import CalibrationStore

    hub = _open_fake_local_hub(monkeypatch)
    calibration_store = CalibrationStore({0: _ring_calibration(0)}, source="startup")
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        calibration_store=calibration_store,
    )
    client = TestClient(app)

    before_resp = client.get("/api/cameras/0/overlay.png")
    assert before_resp.status_code == 200
    before_arr = cv2.imdecode(
        np.frombuffer(before_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR
    )

    # A real recalibration landing mid-session -- same .set() call
    # AppState.refresh_calibration() makes on a manual
    # POST /api/calibration/refresh (and on the auto-calibrate-on-Start
    # path), with a genuinely different camera pose (index=1 instead of
    # 0 on the same synthetic ring).
    calibration_store.set(
        {0: _ring_calibration(1)}, source="manual", checked_at_utc="2026-08-15T00:00:00Z"
    )

    after_resp = client.get("/api/cameras/0/overlay.png")
    assert after_resp.status_code == 200
    after_arr = cv2.imdecode(np.frombuffer(after_resp.content, dtype=np.uint8), cv2.IMREAD_COLOR)

    assert not np.array_equal(before_arr, after_arr), (
        "overlay.png served pixel-identical geometry before and after a real "
        "recalibration swapped the live CalibrationStore's held calibration -- "
        "this is exactly the stale-overlay-after-recalibrate bug this endpoint must not have"
    )


# --------------------------------------------------------------------------
# /api/cameras/status -- real per-camera CameraStatus from LocalCameraHub
# (backend used, negotiated resolution/fps, latency, frame count, last
# error). This data has always existed inside LocalCameraHub.status --
# these tests are about the new endpoint that actually surfaces it.
# --------------------------------------------------------------------------


def test_api_cameras_status_local_mode_with_hub_reports_real_fields(package_root, monkeypatch):
    hub = _open_fake_local_hub(monkeypatch)
    app = create_app(package_root=package_root, enable_background_poll=False, local_hub=hub)
    client = TestClient(app)

    resp = client.get("/api/cameras/status")
    assert resp.status_code == 200
    body = resp.json()

    assert body["frame_source"] == "local"
    assert body["available"] is True
    cam0 = body["cameras"]["0"]
    # These are the exact fields opendarts.live.local_capture.CameraStatus
    # tracks -- the whole point of this endpoint is surfacing them, not a
    # trimmed-down subset.
    assert cam0["opened"] is True
    assert cam0["backend_used"] # FakeVideoCapture always opens
    assert cam0["actual_width"] == 64
    assert cam0["actual_height"] == 48
    assert cam0["frame_count"] >= 1
    assert cam0["last_read_ok"] is True


def test_api_cameras_status_reports_closed_hub_honestly_not_stale_opened_true(
    package_root, monkeypatch
):
    """Real end-to-end regression test for the 2026-08-12 status-honesty
    incident, through the ACTUAL HTTP endpoint (not just LocalCameraHub's
    own unit tests) -- hit a real live `/api/stop` where the
    idle-timeout had already closed the hub ~88 minutes earlier, but `GET
    /api/cameras/status` kept reporting `opened: true, last_read_ok: true`
    for all 3 cameras the entire time while a real live snapshot fetch
    during that window failed outright. Proves the endpoint reflects a
    real close_all() honestly, not a frozen last-known-open snapshot."""
    hub = _open_fake_local_hub(monkeypatch)
    app = create_app(package_root=package_root, enable_background_poll=False, local_hub=hub)
    client = TestClient(app)

    # Before close: opened and reading fine, same as the incident's
    # window right up until the moment the pump actually died.
    before = client.get("/api/cameras/status").json()["cameras"]["0"]
    assert before["opened"] is True
    assert before["last_read_ok"] is True
    assert before["closed_at"] is None

    hub.close_all()

    after = client.get("/api/cameras/status").json()["cameras"]["0"]
    assert after["opened"] is False, (
        "the endpoint must report a closed hub as closed -- not the stale "
        "'opened: true' the real incident exposed"
    )
    assert after["last_read_ok"] is False
    assert after["closed_at"] is not None
    # A snapshot fetch through the same closed hub must fail honestly too
    # (mirrors the real incident's "failed to grab a frame -- None", 502),
    # not silently return a stale cached frame alongside opened: false.
    snap_resp = client.get("/api/cameras/0/snapshot.png")
    assert snap_resp.status_code == 502


def test_api_cameras_status_local_mode_without_hub_reports_unavailable(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    body = client.get("/api/cameras/status").json()

    assert body["frame_source"] == "local"
    assert body["available"] is False
    assert body["cameras"] == {}
    assert "not initialized" in body["reason"]




# --------------------------------------------------------------------------
# /api/calibration/refresh -- POST is the ONLY way calibration ever
# updates now. When a real opendarts.live.capture_daemon.
# CalibrationStore is shared (opendarts/live/run_product.py's real usage),
# a manual refresh must actually replace what the capture loop scores
# against, not just refresh this dashboard's own display -- and persist a
# durable record to disk ("store that calib data").
# --------------------------------------------------------------------------


def test_api_calibration_refresh_updates_and_returns_status(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    assert state.calibration_checked_at_utc is None # never refreshed yet

    client = TestClient(app)
    resp = client.post("/api/calibration/refresh")
    assert resp.status_code == 200
    body = resp.json()

    # No hub in this test (lifespan never ran) -- degrades gracefully,
    # same honest behavior as _refresh_calibration_blocking's own test,
    # but critically it DID run and DID update state, proving this is a
    # real on-demand action, not a no-op button.
    assert body["checked_at_utc"] is not None
    assert state.calibration_checked_at_utc == body["checked_at_utc"]
    assert body["cameras"] == {}
    # No calibration_store was wired -- honestly None, not a fabricated
    # "it worked" claim.
    assert body["live_source"] is None


def test_app_state_no_longer_has_a_calibration_poll_loop_at_all():
    """Not just "unused" -- genuinely REMOVED. A future
    regression re-adding this method (even if nothing calls it) would be
    exactly the kind of half-removed cleanup this test exists to catch."""
    assert not hasattr(server_module.AppState, "_calibration_poll_loop")


def test_start_background_tasks_never_schedules_a_calibration_poll(package_root):
    """start_background_tasks() calls asyncio.create_task(), which
    requires a running loop -- run it (and the cleanup) inside a real
    asyncio.run() rather than pytest-asyncio, since this project's test
    suite doesn't otherwise depend on that plugin."""
    import asyncio

    async def _run() -> set[str]:
        app = create_app(package_root=package_root, enable_background_poll=False)
        state = app.state.opendarts_state
        state.start_background_tasks()
        try:
            return {t.get_name() for t in state._tasks} # noqa: SLF001 -- test needs the real task list
        finally:
            await state.stop_background_tasks()

    task_names = asyncio.run(_run())
    assert "opendarts-calibration-poll" not in task_names
    assert "opendarts-package-poll" in task_names # the one real poll that remains


def test_api_calibration_refresh_without_a_calibration_store_never_crashes_and_says_so(
    package_root, monkeypatch
):
    """A real, successful calibration compute (bootstrap mocked to
    succeed) but with NO calibration_store wired (this module's own
    standalone CLI) -- must still update the display cleanly and report
    live_source honestly as None, never silently pretend it affected live
    scoring."""
    fake_calib = {
        0: CameraCalibration(
            camera_matrix=np.eye(3), dist_coeffs=np.zeros(5),
            rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 555.0]),
            pnp_result=None, landmark_spread_ok=True,
        )
    }
    monkeypatch.setattr(server_module, "bootstrap_calibrations", lambda *a, **k: fake_calib)

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    state.hub = object() # non-None: takes the "hub is open" branch (mocked bootstrap doesn't care)
    assert state.calibration_store is None

    client = TestClient(app)
    body = client.post("/api/calibration/refresh").json()

    assert body["cameras"]["0"]["ok"] is True
    assert body["live_source"] is None


def test_api_calibration_refresh_fails_fast_when_local_hub_cameras_not_ready(
    package_root, monkeypatch
):
    """2026-08-16: refresh returns an error when the cameras are off.
    A real LocalCameraHub-shaped
    object (has .configs/.grab_all(), unlike the bare `object()` sentinel
    other tests here use) whose grab_all() never returns a frame for
    every camera -- proves the readiness pre-check actually fires and
    bootstrap_calibrations() is never even called, not just that the
    response happens to look right."""
    bootstrap_called = False

    def _bootstrap_should_not_be_called(*a, **k):
        nonlocal bootstrap_called
        bootstrap_called = True
        raise AssertionError("bootstrap_calibrations() must not run when cameras aren't ready")

    monkeypatch.setattr(server_module, "bootstrap_calibrations", _bootstrap_should_not_be_called)

    class _NeverReadyHub:
        configs = [object(), object(), object()] # 3 configured cameras

        def grab_all(self):
            return {} # no camera has ever produced a frame

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    state.hub = _NeverReadyHub()

    client = TestClient(app)
    t0 = time.monotonic()
    body = client.post("/api/calibration/refresh").json()
    elapsed = time.monotonic() - t0

    assert bootstrap_called is False
    assert "cameras not ready" in body["calibration_error"]
    assert elapsed < 5.0, f"should fail fast (~2s bound), took {elapsed:.1f}s"


def test_api_calibration_refresh_pushes_into_a_shared_calibration_store(
    package_root, monkeypatch, tmp_path
):
    """THE real correctness requirement: a manual
    recalibrate must replace what the capture loop actually scores
    against (opendarts.live.capture_daemon.CalibrationStore), not just this
    dashboard's own display. CalibrationStore itself has its own dedicated
    thread-safety/correctness coverage in tests/test_capture_daemon.py;
    this test is specifically the server-side wiring: does
    AppState.refresh_calibration() actually call .set() on the SAME
    object opendarts/live/run_product.py would hand to the capture loop.

    (2026-08-22: this test used to also assert a durable disk record via
    save_calibration_snapshot() -- that mechanism was confirmed dead code
    [zero readers anywhere in the codebase; the real, replay-capable
    calibration record is opendarts/capture/calibration_package.py's package
    root] and removed, along with calibration_snapshot_dir. The
    CalibrationStore wiring below is real, unrelated functionality and
    stays.)"""
    from opendarts.live.capture_daemon import CalibrationStore

    fake_calib = {
        0: CameraCalibration(
            camera_matrix=np.eye(3), dist_coeffs=np.zeros(5),
            rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 777.0]),
            pnp_result=None, landmark_spread_ok=True,
        )
    }
    monkeypatch.setattr(server_module, "bootstrap_calibrations", lambda *a, **k: fake_calib)

    calibration_store = CalibrationStore()
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        calibration_store=calibration_store,
    )
    state = app.state.opendarts_state
    state.hub = object()
    assert calibration_store.get() == {} # nothing set yet -- proves the write below is real

    client = TestClient(app)
    body = client.post("/api/calibration/refresh").json()

    assert body["live_source"]["source"] == "manual"
    assert body["live_source"]["n_cameras"] == 1

    # The store the capture loop reads from is now real -- the EXACT
    # object bootstrap_calibrations() produced, not a copy with different
    # identity (proves refresh_calibration() didn't reconstruct/re-derive
    # it, just plumbed the real result through).
    stored = calibration_store.get()
    assert stored[0] is fake_calib[0]


# --------------------------------------------------------------------------
# /api/reset -- added 2026-08-12, the project's "wire the reset button"
# request. Mirrors the calibration-refresh tests' own shape: the endpoint
# itself is a thin wrapper around opendarts.live.capture_daemon.ResetRequest
# (which has its own dedicated unit/thread-safety coverage in
# tests/test_capture_daemon.py) -- these tests are specifically the
# server-side WIRING: does the route actually call .request() on the
# SAME object opendarts/live/run_product.py would hand to the capture loop.
# --------------------------------------------------------------------------


def test_api_reset_without_a_reset_request_reports_honestly(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    assert app.state.opendarts_state.reset_request is None

    body = TestClient(app).post("/api/reset").json()
    assert body == {"requested": False, "loop_listening": False}


def test_api_reset_signals_the_shared_reset_request(package_root):
    from opendarts.live.capture_daemon import ResetRequest

    reset_request = ResetRequest()
    app = create_app(
        package_root=package_root, enable_background_poll=False, reset_request=reset_request
    )
    assert reset_request.check_and_clear() is False # nothing pending yet

    body = TestClient(app).post("/api/reset").json()
    assert body["requested"] is True
    assert body["loop_listening"] is True
    assert body["last_requested_at_utc"] is not None

    # The REAL object the capture loop would poll -- proves this endpoint
    # didn't reconstruct/re-derive its own separate ResetRequest.
    assert reset_request.check_and_clear() is True


def test_api_reset_touches_the_controller_as_real_activity(package_root):
    """A manual Reset click counts as idle-timeout activity too (see
    CaptureLoopController's own docstring) -- proven here directly
    against the shared controller object."""
    from opendarts.live.capture_daemon import CaptureLoopController, ResetRequest

    controller = CaptureLoopController(idle_timeout_sec=1)
    reset_request = ResetRequest()
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        reset_request=reset_request,
        controller=controller,
    )
    with controller._lock: # noqa: SLF001 -- test-only direct backdate, mirrors test_capture_daemon.py's own pattern
        controller._idle_timeout_sec = 0.05
        controller._running = True
    import time as _time

    _time.sleep(0.15)
    assert controller.idle_timeout_due() is True

    TestClient(app).post("/api/reset")
    assert controller.idle_timeout_due() is False


# --------------------------------------------------------------------------
# /api/start, /api/stop, the idle timeout -- added 2026-08-12 to wire
# the Start and Stop buttons. See
# opendarts.live.capture_daemon.CaptureLoopController's own docstring for the
# real design (an explicit start/stop pair plus a touch/idle-loop timeout,
# adapted for opendarts's thread-based architecture).
# --------------------------------------------------------------------------


class _FakeHubForStartStop:
    """Minimal local_hub stand-in -- records open/close calls, never
    touches cv2, mirrors tests/test_run_product.py's own FakeHub."""

    def __init__(self, open_ok: bool = True) -> None:
        self.open_calls = 0
        self.close_calls = 0
        self._open_ok = open_ok

    def open_all(self):
        self.open_calls += 1
        return [self._open_ok, self._open_ok]

    def close_all(self):
        self.close_calls += 1

    def status_report(self) -> str:
        return "fake hub status"


def test_api_start_without_a_controller_reports_honestly(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    assert app.state.opendarts_state.controller is None

    body = TestClient(app).post("/api/start").json()
    assert body["ok"] is False
    assert "standalone" in body["reason"]


def test_api_start_opens_the_hub_and_requests_a_session(package_root):
    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop(open_ok=True)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        controller=controller,
    )
    assert controller.is_running() is False

    body = TestClient(app).post("/api/start").json()
    assert body["ok"] is True
    assert hub.open_calls == 1
    assert controller.is_running() is True
    assert controller.start_requested.is_set() is True


def test_api_start_is_idempotent_when_already_running(package_root):
    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop(open_ok=True)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        controller=controller,
    )
    client = TestClient(app)
    client.post("/api/start")
    assert hub.open_calls == 1

    body = client.post("/api/start").json()
    assert body["ok"] is True
    assert body.get("already_running") is True
    assert hub.open_calls == 1 # NOT opened a second time


def test_api_start_is_idempotent_while_still_starting_not_yet_running(package_root):
    """Real incident, 2026-08-17: a second POST /api/start arriving while
    a FIRST one is still inside the ~7s hub.open_all() sailed straight
    past the old guard (`if self.controller.is_running():` only) --
    is_running() doesn't flip True until AFTER open_all() finishes, so
    the window between "capture_starting set True" and "is_running()
    becomes True" had no idempotency protection. Two concurrent
    open_all() calls on the same hub is a real, reproducible trigger for
    a native camera-pipeline crash (see this fix's own commit message
    for the full incident). Simulates the race's STATE directly
    (capture_starting=True, is_running() still False) rather than real
    thread timing -- same pattern already used elsewhere in this suite
    for state-based race reproduction."""
    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop(open_ok=True)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        controller=controller,
    )
    state = app.state.opendarts_state
    state.capture_starting = True
    assert controller.is_running() is False # the exact window this fix closes

    body = TestClient(app).post("/api/start").json()

    assert body["ok"] is True
    assert body.get("already_running") is True
    assert hub.open_calls == 0 # NEVER opened -- the real bug would make this 1


def test_api_start_reports_ok_false_when_zero_cameras_open(package_root):
    """The real, honest failure mode moved here from
    opendarts/live/run_product.py's own _build_components() (which used to
    raise RuntimeError at process-startup time, before this feature --
    see tests/test_run_product.py's own
    test_build_components_never_probes_cameras_even_with_an_always_failing_hub)."""
    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop(open_ok=False)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        controller=controller,
    )
    body = TestClient(app).post("/api/start").json()
    assert body["ok"] is False
    assert body["reason"] == "no cameras opened"
    assert controller.is_running() is False


def test_api_restart_schedules_a_real_self_sigterm_via_the_signal_handler_path(package_root, monkeypatch):
    """The actual point of /api/restart: it must call the real
    `os.kill(this_pid, SIGTERM)` -- the same call a manual `kill <PID>`
    makes -- not some bespoke shutdown path. `os.kill` is monkeypatched
    to a recording stub rather than actually invoked: this test runs
    IN the pytest process itself, so a real SIGTERM here would kill the
    test run, not some fake subprocess -- same "never let a live
    process-killing signal actually fire in a shared test process"
    discipline as _restored_signal_handlers() elsewhere in this file,
    just applied one level earlier (patching the call site instead of
    the handler)."""
    import signal as _signal

    calls = []
    monkeypatch.setattr(server_module.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    # Fire the scheduled timer immediately rather than waiting out the
    # real RESTART_SIGTERM_DELAY_S -- same intent, deterministic test.
    monkeypatch.setattr(
        server_module.threading, "Timer",
        lambda _delay, fn, args: type("ImmediateTimer", (), {"start": lambda self: fn(*args)})(),
    )

    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).post("/api/restart")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["pid"] == os.getpid()
    assert "SIGTERM" in body["message"]
    assert calls == [(os.getpid(), _signal.SIGTERM)]


def test_api_restart_works_with_no_capture_loop_wired(package_root):
    """Unlike /api/start-/api/stop-/api/reset (which act ON the capture
    loop and honestly no-op without one), /api/restart acts on the WHOLE
    process -- must stay available even for this module's own standalone
    CLI with no controller/hub wired at all."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("opendarts.live.server.os.kill", lambda pid, sig: None)
        mp.setattr(
            "opendarts.live.server.threading.Timer",
            lambda _delay, fn, args: type("ImmediateTimer", (), {"start": lambda self: fn(*args)})(),
        )
        resp = TestClient(app).post("/api/restart")
    assert resp.json()["ok"] is True


# ---------------------------------------------------------------------------
# POST /api/restart {"update": true} -- asking the LAUNCHER to pull
#
# The flag this writes is read by run.sh/run.ps1 between the old process
# exiting and the new one starting; nothing in this process ever acts on
# it. See opendarts/live/update_policy.py and tests/test_shell_scripts.py
# for the other half.


def _immediate_restart(monkeypatch):
    """Record the SIGTERM instead of sending it, and fire the timer now.

    A real SIGTERM here would kill the pytest process itself, not a
    subprocess -- same discipline as
    test_api_restart_schedules_a_real_self_sigterm_... above, extracted
    because every test below needs it.
    """
    kills: "list[tuple[int, int]]" = []
    monkeypatch.setattr(server_module.os, "kill", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(
        server_module.threading, "Timer",
        lambda _delay, fn, args: type("ImmediateTimer", (), {"start": lambda self: fn(*args)})(),
    )
    return kills


def _restart_client(package_root, monkeypatch, tmp_path, initial=None):
    """A client whose /api/restart reads and writes a THROWAWAY config.

    The route's config helpers bind their path default at import, so the
    real functions are rebound here against `tmp_path` rather than
    stubbed -- these tests then assert on the actual file the launcher
    would read, not on a recording of a call that was made.
    """
    import json as _json
    from functools import partial

    from opendarts.live import config as config_module

    cfg = tmp_path / "config.json"
    cfg.write_text(_json.dumps(initial if initial is not None else {"port": 8420}))
    for name in ("always_update", "update_on_next_restart", "set_update_on_next_restart"):
        monkeypatch.setattr(
            server_module, name, partial(getattr(config_module, name), path=cfg)
        )
    app = create_app(package_root=package_root, enable_background_poll=False)
    return TestClient(app), cfg


def test_api_restart_with_no_body_restarts_and_leaves_the_config_alone(
    package_root, monkeypatch, tmp_path
):
    """FlightDeck calls this route with no body and always has. The
    optional body must not have made the old request a second-class one:
    same restart, and config.json is not written -- not even read."""
    import json as _json

    kills = _immediate_restart(monkeypatch)
    client, cfg = _restart_client(package_root, monkeypatch, tmp_path)
    before = cfg.read_text()

    body = client.post("/api/restart").json()

    assert body["ok"] is True
    assert body["update"] is False
    assert body["pid"] == os.getpid()
    assert kills == [(os.getpid(), signal.SIGTERM)]
    assert cfg.read_text() == before
    assert "update_on_next_restart" not in _json.loads(cfg.read_text())


def test_api_restart_update_true_sets_the_flag_the_launcher_reads(
    package_root, monkeypatch, tmp_path
):
    """The flag has to be in the FILE, spelled the way the launcher reads
    it -- an in-memory record would restart a rig onto the same code."""
    import json as _json

    from opendarts.live.update_policy import PULL_ONCE, decide

    kills = _immediate_restart(monkeypatch)
    client, cfg = _restart_client(package_root, monkeypatch, tmp_path)

    body = client.post("/api/restart", json={"update": True}).json()

    assert body["ok"] is True
    assert body["update"] is True
    assert kills == [(os.getpid(), signal.SIGTERM)]
    raw = _json.loads(cfg.read_text())
    assert raw["update_on_next_restart"] is True
    assert raw["port"] == 8420, "the operator's other settings were rewritten"
    # And the launcher's own reader agrees, rather than this test agreeing
    # with itself about a key name.
    assert decide(cfg) == PULL_ONCE


def test_api_restart_update_false_is_an_explicit_no_pull(package_root, monkeypatch, tmp_path):
    """`{"update": false}` is a real answer, not a malformed one -- and it
    must CLEAR a queued request rather than silently leaving it set."""
    import json as _json

    kills = _immediate_restart(monkeypatch)
    client, cfg = _restart_client(
        package_root, monkeypatch, tmp_path, initial={"update_on_next_restart": False}
    )

    body = client.post("/api/restart", json={"update": False}).json()

    assert body["ok"] is True
    assert body["update"] is False
    assert kills == [(os.getpid(), signal.SIGTERM)]
    assert _json.loads(cfg.read_text())["update_on_next_restart"] is False


def test_api_restart_rejects_a_non_boolean_update_without_restarting(
    package_root, monkeypatch, tmp_path
):
    """`"true"` and `1` are the realistic typos, and `bool(raw)` would
    accept both -- along with the string `"false"`. This route moves a rig
    onto different code; it does not guess."""
    import json as _json

    kills = _immediate_restart(monkeypatch)
    client, cfg = _restart_client(package_root, monkeypatch, tmp_path)

    for bad in ("true", "false", 1, 0, None, [], {"a": 1}):
        body = client.post("/api/restart", json={"update": bad}).json()
        assert body["ok"] is False, bad
        assert body["restarting"] is False, bad
        assert "must be true or false" in body["reason"], bad
    assert kills == [], "a rejected request restarted the rig anyway"
    assert "update_on_next_restart" not in _json.loads(cfg.read_text())


def test_api_restart_rejects_unknown_fields_without_restarting(
    package_root, monkeypatch, tmp_path
):
    """A caller who typed `{"pull": true}` meant something, and it was not
    "restart onto the same code" -- so say which field is wrong instead of
    quietly restarting without updating."""
    kills = _immediate_restart(monkeypatch)
    client, _cfg = _restart_client(package_root, monkeypatch, tmp_path)

    body = client.post("/api/restart", json={"pull": True, "update": True}).json()

    assert body["ok"] is False
    assert body["restarting"] is False
    assert "pull" in body["reason"]
    assert kills == []


def test_api_restart_rejects_a_malformed_body_without_restarting(
    package_root, monkeypatch, tmp_path
):
    """Not JSON at all, and a JSON value that is not an object. Both are
    refused by the request layer before the handler runs -- pinned because
    "the body was unreadable so we restarted without updating" is the one
    failure this feature must not have."""
    kills = _immediate_restart(monkeypatch)
    client, _cfg = _restart_client(package_root, monkeypatch, tmp_path)

    for bad in ('{"update": tru', "not json at all", "[1, 2, 3]", '"update"'):
        resp = client.post(
            "/api/restart", content=bad, headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 422, (bad, resp.status_code)
    assert kills == [], "a malformed body restarted the rig"


def test_api_restart_refuses_to_restart_when_the_request_cannot_be_recorded(
    package_root, monkeypatch, tmp_path
):
    """A HAND-EDITED CONFIG WITH A SYNTAX ERROR, which is the real case:
    write_config_section() declines to overwrite one and only logs.

    Restarting anyway would be the worst available outcome -- the rig
    comes back looking exactly as it should, running the old code, and the
    next thing anyone does is wonder why their fix is not live.
    """
    kills = _immediate_restart(monkeypatch)
    client, cfg = _restart_client(package_root, monkeypatch, tmp_path)
    cfg.write_text("{ this was hand-edited badly")

    body = client.post("/api/restart", json={"update": True}).json()

    assert body["ok"] is False
    assert body["restarting"] is False
    assert "config.json" in body["reason"]
    assert kills == [], "restarted despite not recording the update request"
    assert cfg.read_text() == "{ this was hand-edited badly"


def test_get_api_restart_reports_the_flags_and_restarts_nothing(
    package_root, monkeypatch, tmp_path
):
    """What the dashboard reads to decide whether "Update and restart" is
    a meaningful button on this rig."""
    kills = _immediate_restart(monkeypatch)
    client, _cfg = _restart_client(
        package_root, monkeypatch, tmp_path,
        initial={"always_update": True, "update_on_next_restart": False},
    )

    body = client.get("/api/restart").json()

    assert body == {"ok": True, "always_update": True, "update_on_next_restart": False}
    assert kills == [], "a GET killed the process"


def test_get_api_restart_defaults_to_false_on_a_rig_that_never_set_them(
    package_root, monkeypatch, tmp_path
):
    client, _cfg = _restart_client(package_root, monkeypatch, tmp_path, initial={})
    body = client.get("/api/restart").json()
    assert body["always_update"] is False
    assert body["update_on_next_restart"] is False


def test_health_reports_this_process_pid(package_root):
    """The restart poll needs to tell "the rig is back" from "the process
    it asked to restart has not died yet" -- /api/restart schedules its
    SIGTERM half a second out, so the doomed process answers perfectly in
    the meantime and the two are otherwise identical over HTTP."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    assert TestClient(app).get("/api/health").json()["pid"] == os.getpid()


def test_api_stop_without_a_controller_reports_honestly(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).post("/api/stop").json()
    assert body["ok"] is False


def test_api_stop_is_a_real_noop_when_nothing_is_running(package_root):
    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop()
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    body = TestClient(app).post("/api/stop").json()
    assert body["ok"] is True
    assert body.get("already_stopped") is True
    assert hub.close_calls == 0


def test_api_stop_closes_the_hub_after_the_capture_thread_acknowledges(package_root):
    """Real end-to-end proof of the bounded-wait-then-close mechanism
    -- a background thread stands in for the real capture thread,
    acknowledging (`stopped_ack.set()`) only once `session_stop_event`
    is observed, exactly like opendarts/live/run_product.py's own
    `_capture_thread_target` does via run_capture_loop_body's
    `also_stop=`."""
    import threading as _threading

    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop()
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    client = TestClient(app)
    client.post("/api/start")
    assert controller.is_running() is True

    def fake_session() -> None:
        controller.session_stop_event.wait()
        controller.mark_session_ended()

    t = _threading.Thread(target=fake_session, daemon=True)
    t.start()

    body = client.post("/api/stop").json()
    assert body["ok"] is True
    assert body["acknowledged"] is True
    assert hub.close_calls == 1
    assert controller.is_running() is False
    t.join(timeout=2.0)


def test_api_stop_resets_stale_trigger_state_to_honest_null(package_root):
    """Real regression test for the 2026-08-12 status-honesty audit (see
    opendarts/live/server.py's own module docstring dated entry): before this
    fix, `AppState.trigger_state`/`trigger_dart_count` were ONLY ever
    written by a real TRIGGER_STATE event pushed from the capture thread
    -- when a session ended (manual Stop or idle-timeout),
    `CaptureLoopController.mark_session_ended()` flipped `running` False
    but pushed no event of its own, so `trigger_state` sat frozen at
    whatever throw-in-progress state it last saw (e.g. TAKEOUT_WAITING)
    FOREVER after the loop had actually stopped -- `/api/state` would
    keep claiming a stopped capture loop was still mid-takeout. Proves the
    state actually flips to honest-null on a real `/api/stop`, not just
    that the dashboard's own JS pill happens to mask it client-side."""
    import threading as _threading

    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop()
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    client = TestClient(app)
    client.post("/api/start")
    assert controller.is_running() is True

    # Simulate a real live TAKEOUT_WAITING event having landed mid-session
    # -- exactly the state a stale trigger_state would be frozen at.
    state.trigger_state = "TAKEOUT_WAITING"
    state.trigger_dart_count = 2

    def fake_session() -> None:
        controller.session_stop_event.wait()
        controller.mark_session_ended()

    t = _threading.Thread(target=fake_session, daemon=True)
    t.start()

    body = client.post("/api/stop").json()
    t.join(timeout=2.0)

    assert body["ok"] is True
    # The stale value must actually FLIP, not just stop being updated.
    assert state.trigger_state is None, (
        "trigger_state must reset to honest-null when the session actually stops, "
        "not stay frozen at the last real throw-in-progress state forever"
    )
    assert state.trigger_dart_count is None
    assert state.trigger_last_event_utc is not None

    # /api/state (any caller, not just the dashboard's own JS pill) must
    # reflect the same honest reset.
    api_state = client.get("/api/state").json()
    assert api_state["trigger"]["state"] is None
    assert api_state["trigger"]["dart_count"] is None

    # And every connected WebSocket client gets a real TRIGGER_STATE reset
    # broadcast, not just a silent internal flip only the next poll sees --
    # sent AFTER CAPTURE_LOOP_STATUS so `cl.running: false` is already
    # known client-side before the (now-null) trigger state arrives.
    sent_types = [json.loads(m)["type"] for m in fake_ws.sent]
    assert "TRIGGER_STATE" in sent_types
    assert "CAPTURE_LOOP_STATUS" in sent_types
    assert sent_types.index("CAPTURE_LOOP_STATUS") < sent_types.index("TRIGGER_STATE")
    reset_msg = json.loads(fake_ws.sent[sent_types.index("TRIGGER_STATE")])
    assert reset_msg["state"] is None
    assert reset_msg["dart_count"] is None


def test_idle_timeout_without_a_controller_is_saved_but_not_applied(package_root):
    """The route this replaced (`POST /api/idle-timeout`, retired
    2026-09-17) refused outright here and wrote nothing, which meant a
    process with no capture loop could not configure its own config file.
    The document persists it and says plainly that nothing in THIS
    process read it."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).patch("/api/config", json={"idle_timeout_sec": 60}).json()
    assert body["ok"] is True
    assert body["persisted"] == ["idle_timeout_sec"]
    assert body["applied_live"] == []
    assert "no capture loop in this process" in body["notes"]["idle_timeout_sec"]


def test_idle_timeout_sets_and_clamps_negative_to_zero(package_root):
    from opendarts.live.capture_daemon import CaptureLoopController

    controller = CaptureLoopController()
    app = create_app(package_root=package_root, enable_background_poll=False, controller=controller)
    client = TestClient(app)

    body = client.patch("/api/config", json={"idle_timeout_sec": 42}).json()
    assert body["ok"] is True
    assert body["config"]["idle_timeout_sec"] == 42
    assert body["applied_live"] == ["idle_timeout_sec"]
    assert controller.get_idle_timeout_sec() == 42

    # CLAMPED, not refused: 0 or below disables the auto-stop entirely,
    # and a negative number is that request spelled awkwardly.
    body2 = client.patch("/api/config", json={"idle_timeout_sec": -5}).json()
    assert body2["config"]["idle_timeout_sec"] == 0
    assert controller.get_idle_timeout_sec() == 0


def test_a_non_numeric_idle_timeout_is_refused_with_a_reason(package_root):
    """The retired route ran `int(body[...])` bare, so a string took the
    process to a 500. The document validates it like every other key."""
    from opendarts.live.capture_daemon import CaptureLoopController

    controller = CaptureLoopController(idle_timeout_sec=900)
    app = create_app(package_root=package_root, enable_background_poll=False, controller=controller)
    resp = TestClient(app).patch("/api/config", json={"idle_timeout_sec": "soon"})
    assert resp.status_code == 400
    assert resp.json()["errors"]["idle_timeout_sec"] == (
        "idle_timeout_sec must be a whole number, got 'soon'"
    )
    assert controller.get_idle_timeout_sec() == 900


def test_state_dict_capture_loop_field_is_none_without_a_controller(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = TestClient(app).get("/api/state").json()
    assert state["capture_loop"] is None


# ---------------------------------------------------------------------------
# `diagnostics` on the config document -- opendarts.live.diagnostics_gate,
# 2026-09-04. The switch is a process-wide module-level singleton (not
# per-AppState), so unlike `idle_timeout_sec` it needs no controller/store
# wired to work: it applies regardless of whether a capture loop shares
# this process.
#
# THE ONE KEY THAT IS NOT PERSISTED, and deliberately (see the
# diagnostics_gate module docstring): the whole point is a fast live A/B
# toggle within one running process, always back to OFF on a fresh start.
# Remembering it across a restart would add a "did I leave this on"
# footgun to the true-latency measurement it exists for. It had a route
# pair of its own until 2026-09-17.
# ---------------------------------------------------------------------------


def test_diagnostics_reports_off_by_default(package_root):
    from opendarts.live import diagnostics_gate

    assert diagnostics_gate.enabled() is False
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).get("/api/config").json()
    assert body["config"]["diagnostics"] == {"enabled": False}


def test_diagnostics_toggles_on_and_off(package_root):
    from opendarts.live import diagnostics_gate

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    on = client.patch("/api/config", json={"diagnostics": {"enabled": True}}).json()
    assert on["config"]["diagnostics"] == {"enabled": True}
    assert on["applied_live"] == ["diagnostics"]
    assert diagnostics_gate.enabled() is True

    off = client.patch("/api/config", json={"diagnostics": {"enabled": False}}).json()
    assert off["config"]["diagnostics"] == {"enabled": False}
    assert diagnostics_gate.enabled() is False


def test_diagnostics_is_applied_but_never_written_to_the_config_file(package_root):
    """Live, and gone at the next start. A `diagnostics` key appearing in
    data/config.json would be this decision quietly reversed."""
    import json as _json

    from opendarts.live.config import DEFAULT_CONFIG_PATH
    from opendarts.live.config_document import KEYS_BY_NAME

    assert KEYS_BY_NAME["diagnostics"].persist is False
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).patch(
        "/api/config", json={"diagnostics": {"enabled": True}}
    ).json()
    assert body["ok"] is True
    assert body["applied_live"] == ["diagnostics"]
    assert body["persisted"] == []
    saved = _json.loads(DEFAULT_CONFIG_PATH.read_text()) if DEFAULT_CONFIG_PATH.exists() else {}
    assert "diagnostics" not in saved


def test_a_diagnostics_value_that_is_not_a_boolean_is_refused(package_root):
    """The retired route ran `bool(body["on"])`, so the string "off"
    switched diagnostics ON. `{}` is NOT an error, though: a group is
    MERGED onto what is in force, so naming no sub-key is a no-op."""
    from opendarts.live import diagnostics_gate

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    resp = client.patch("/api/config", json={"diagnostics": {"enabled": "off"}})
    assert resp.status_code == 400
    assert resp.json()["errors"]["diagnostics"] == (
        "diagnostics.enabled must be true or false, got 'off'"
    )
    resp = client.patch("/api/config", json={"diagnostics": {"on": True}})
    assert resp.status_code == 400
    assert resp.json()["errors"]["diagnostics"] == "diagnostics has no key 'on'"
    assert diagnostics_gate.enabled() is False


def test_diagnostics_toggle_takes_effect_immediately_no_restart(package_root):
    """The literal requirement: the toggle affects the very next
    read, no process restart / no new AppState needed -- proven by
    reading `opendarts.live.diagnostics_gate.enabled()` directly (what the
    real capture loop reads every iteration) right after the PATCH, in
    the SAME process, not just re-reading the HTTP response."""
    from opendarts.live import diagnostics_gate

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    assert diagnostics_gate.enabled() is False
    client.patch("/api/config", json={"diagnostics": {"enabled": True}})
    assert diagnostics_gate.enabled() is True, (
        "the module-level switch a live capture loop reads must reflect the "
        "toggle immediately, in-process, without any restart"
    )


def test_diagnostics_toggle_flips_third_party_logger_level(package_root):
    """The other real requirement: the toggle must ACTUALLY call
    logging.getLogger(...).setLevel(...) on websockets/uvicorn.error, not
    just flip an internal flag some other code reads later."""
    import logging

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    # Default OFF state: quiet.
    assert logging.getLogger("websockets").level == logging.WARNING
    assert logging.getLogger("uvicorn.error").level == logging.WARNING

    client.patch("/api/config", json={"diagnostics": {"enabled": True}})
    assert logging.getLogger("websockets").level == logging.NOTSET
    assert logging.getLogger("uvicorn.error").level == logging.NOTSET

    client.patch("/api/config", json={"diagnostics": {"enabled": False}})
    assert logging.getLogger("websockets").level == logging.WARNING
    assert logging.getLogger("uvicorn.error").level == logging.WARNING


def test_diagnostics_reflected_in_state_dict(package_root):
    from opendarts.live import diagnostics_gate

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    assert client.get("/api/state").json()["diagnostics"] == {"enabled": False}
    diagnostics_gate.set_enabled(True)
    assert client.get("/api/state").json()["diagnostics"] == {"enabled": True}


def test_state_dict_capture_loop_field_reflects_real_controller_status(package_root):
    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop()
    controller = CaptureLoopController(idle_timeout_sec=123)
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    client = TestClient(app)
    state_before = client.get("/api/state").json()
    assert state_before["capture_loop"]["running"] is False
    assert state_before["capture_loop"]["idle_timeout_sec"] == 123

    client.post("/api/start")
    state_after = client.get("/api/state").json()
    assert state_after["capture_loop"]["running"] is True


def test_idle_timeout_loop_auto_stops_a_running_session(package_root, monkeypatch):
    """Real, end-to-end (within this AppState's own asyncio background
    task) proof that idle-timeout auto-stop actually fires -- a fast
    configured timeout + a fast poll cadence (monkeypatched
    IDLE_CHECK_INTERVAL_SECONDS, same pattern this project already uses
    for _HEARTBEAT_EVERY_S/other timing constants), not a real 900s/5s
    wait."""
    import asyncio
    import threading as _threading

    from opendarts.live.capture_daemon import CaptureLoopController

    monkeypatch.setattr(server_module, "IDLE_CHECK_INTERVAL_SECONDS", 0.05)

    hub = _FakeHubForStartStop()
    controller = CaptureLoopController(idle_timeout_sec=1)
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state

    async def _run() -> None:
        with TestClient(app) as client:
            client.post("/api/start")
        assert controller.is_running() is True
        with controller._lock: # noqa: SLF001 -- test-only direct backdate
            controller._idle_timeout_sec = 0.05

        def fake_session() -> None:
            controller.session_stop_event.wait()
            controller.mark_session_ended()

        t = _threading.Thread(target=fake_session, daemon=True)
        t.start()

        task = asyncio.create_task(state._idle_timeout_loop())
        try:
            deadline = asyncio.get_event_loop().time() + 2.0
            while controller.is_running() and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.02)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        t.join(timeout=2.0)

    asyncio.run(_run())
    assert controller.is_running() is False
    assert hub.close_calls == 1


def test_the_idle_timeout_frees_the_frame_ring(package_root, monkeypatch):
    """A stopped session's frames sat in memory for as long as the rig was
    idle; after the idle timeout none of them is worth keeping."""
    import asyncio
    import threading as _threading
    import types

    import numpy as np

    from opendarts.capture.frame_ring import FrameRing
    from opendarts.live.capture_daemon import CaptureLoopController

    monkeypatch.setattr(server_module, "IDLE_CHECK_INTERVAL_SECONDS", 0.05)
    ring = FrameRing(60.0)
    ring.append({0: np.zeros((4, 4, 3), np.uint8)}, wall_s=1.0, monotonic_s=1.0, generation=0)
    hub = _FakeHubForStartStop()
    controller = CaptureLoopController(idle_timeout_sec=1)
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state
    state.throw_capture = types.SimpleNamespace(ring=ring)

    async def _run() -> None:
        with TestClient(app) as client:
            client.post("/api/start")
        with controller._lock:  # noqa: SLF001
            controller._idle_timeout_sec = 0.05

        def fake_session() -> None:
            controller.session_stop_event.wait()
            controller.mark_session_ended()

        _threading.Thread(target=fake_session, daemon=True).start()
        task = asyncio.create_task(state._idle_timeout_loop())
        try:
            deadline = asyncio.get_event_loop().time() + 2.0
            while ring.stats()["sets"] and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.02)
        finally:
            task.cancel()

    asyncio.run(_run())
    assert ring.stats()["sets"] == 0


# --------------------------------------------------------------------------
# /api/events -- WebSocket sends a HELLO with current state + packages.
# --------------------------------------------------------------------------

def test_websocket_events_sends_hello(package_root):
    _write_sample_package(package_root / "s1" / "throw_a")
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    with client.websocket_connect("/api/events") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "HELLO"
        assert msg["state"]["package_root"] == str(package_root)
        assert len(msg["packages"]) == 1
        assert msg["packages"][0]["sector"] == "S20"
        assert msg["count"] == 1


def test_websocket_events_hello_count_reflects_the_true_total_past_the_20_cap(
    package_root,
):
    """REAL BUG, 2026-08-24 (flagged by the pull/QA process, caught
    because the pulled packages disagreed with the dashboard): one
    recorded session captured 120 real throws, the dashboard showed
    102 and computed every Scoring-tab percentage over that
    undercounted subset. Root cause: HELLO's own `packages` list is
    capped to the newest 20 (harmless at normal pace, sorted newest-
    first) -- but until this fix, HELLO carried no `count` field at
    all, so a client reconnecting after a long-enough WebSocket outage
    (more than 20 throws' worth) had no signal it was missing anything
    older than the newest 20. `count` must be the TRUE total, not
    len(packages) (which is always <=20) -- this proves the two
    genuinely diverge once there are more than 20 real packages."""
    for i in range(25):
        _write_sample_package(package_root / "s1" / f"throw_{i:03d}")
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    with client.websocket_connect("/api/events") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "HELLO"
        assert len(msg["packages"]) == 20 # the cap, unchanged
        assert msg["count"] == 25 # the TRUE total -- this is the fix


# --------------------------------------------------------------------------
# Header status pill. Three things
# proven here: (1) the pill's HTML/CSS/JS structure actually exists and is
# wired to every real opendarts.capture.throw_trigger.ThrowState, not just a
# generic-looking div; (2) /api/state's "trigger" section -- what the
# dashboard's own loadInitial()/renderState() calls on page load, BEFORE
# any WebSocket message ever arrives -- carries dart_count and reflects
# real AppState fields, not a blank/default flash; (3) a real
# AppState._handle_live_event(...) call (the exact coroutine
# opendarts/live/run_product.py's live_event_queue plumbing invokes for every
# real TRIGGER_STATE the capture loop emits) updates AppState AND
# broadcasts the same shape to connected WebSocket clients, including the
# dart_count field added alongside this pill.
# --------------------------------------------------------------------------


def test_root_html_has_status_pill_wired_to_all_real_throw_states(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert 'class="status-pill" id="status-pill"' in html
    assert 'id="status-dot"' in html
    assert 'id="status-text"' in html
    assert 'id="status-detail"' in html
    # The old full-width #trigger-banner row is genuinely gone (replaced
    # by the pill living IN the header), not just unlinked/hidden.
    assert 'id="trigger-banner"' not in html

    # Every real ThrowState name (opendarts/capture/throw_trigger.py) must be
    # mapped to a pill color/label/detail -- a future new state silently
    # falling through to the "unrecognized state" branch would be a real
    # regression this test would catch.
    for state_name in (
        "IDLE",
        "MOTION_DETECTED",
        "SETTLING",
        "READY_TO_CAPTURE",
        "TAKEOUT_WAITING",
    ):
        assert f"{state_name}:" in html

    # Honest "no live capture loop" case (this module's own standalone
    # CLI, no opendarts.live.run_product in-process) is a real, distinct,
    # explicitly-labeled code path -- not a silently-wrong default state.
    assert "No live capture" in html
    # WebSocket TRIGGER_STATE handling + dart_count-aware secondary text
    # are both actually wired into the page's own JS, not just present in
    # the backend.
    assert "TRIGGER_STATE" in html
    assert "dart_count" in html


def test_status_pill_manual_calibrate_flag_has_a_correctly_ordered_lifecycle(package_root):
    """Narrower companion to
    test_status_pill_waiting_state_covers_both_starting_and_manual_calibrate
    (which already guards the label/text substrings) -- this one guards
    the actual ORDER of `manualCalibrating`'s three mutation sites, which
    that test does not check. If a future edit ever moved the `= false`
    reset out of the click handler's `finally` (or dropped it), a failed
    /api/calibration/refresh request would leave the pill permanently
    stuck on 'Calibrating' for that tab -- exactly the class of bug this
    whole feature exists to prevent, just relocated one level in. Also
    locks D4 (reuse Starting's blue, don't introduce a new color): only
    one WAITING entry can exist in PRIMARY_INFO, so the manual-calibrate
    branch applying PRIMARY_INFO.WAITING structurally cannot diverge from
    Starting's own color.
    """
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    # Declared false (renderPill()'s own default, before any click).
    assert "let manualCalibrating = false;" in html
    # Set true immediately before the fetch, reset false in `finally` --
    # both edges must exist, and the reset must come after the set, so
    # a thrown fetch still clears it.
    after_set = html.split("manualCalibrating = true;", 1)[1]
    assert "manualCalibrating = false;" in after_set

    # The manual-calibrate branch in renderPill() applies PRIMARY_INFO.WAITING
    # -- the SAME primary state Starting uses, not an independent one.
    # Checked against the served page rather than the source file, which is
    # opendarts/live/dashboard/app.js.
    assert "if (manualCalibrating) {" in html
    manual_branch = html.split("if (manualCalibrating)", 1)[1].split("}", 1)[0]
    assert "PRIMARY_INFO.WAITING" in manual_branch

    # D4: only one WAITING entry exists in PRIMARY_INFO. Sliced to the
    # PRIMARY_INFO object itself (not the whole page) so PHASE_DETAIL's
    # unrelated TAKEOUT_WAITING key -- a substring match for "WAITING:"
    # too -- can't produce a false pass here.
    primary_info_block = html.split("const PRIMARY_INFO = {", 1)[1].split("};", 1)[0]
    assert primary_info_block.count("WAITING:") == 1


def test_api_state_trigger_section_includes_dart_count_honestly_null_by_default(package_root):
    """No live_event_queue at all (this module's own standalone CLI) --
    dart_count must stay honestly None, exactly like trigger.state
    already does, never a fabricated 0."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).get("/api/state").json()
    assert "dart_count" in body["trigger"]
    assert body["trigger"]["dart_count"] is None
    assert body["trigger"]["available"] is False


def test_api_state_trigger_available_but_state_and_dart_count_null_before_first_event(
    package_root,
):
    """A live_event_queue IS wired (opendarts/live/run_product.py's shape)
    but no event has arrived in THIS process yet -- available flips True
    immediately (the queue exists), while state/dart_count stay honestly
    None until a real TRIGGER_STATE lands. This is exactly the initial
    /api/state page-load case the dashboard's loadInitial() hits right
    after a fresh opendarts.live.run_product process starts, before its
    capture thread has pushed its first event -- the pill must render
    "Connecting..." here, not a blank flash or a fabricated IDLE."""
    import queue

    q: "queue.SimpleQueue[dict]" = queue.SimpleQueue()
    app = create_app(package_root=package_root, enable_background_poll=False, live_event_queue=q)
    body = TestClient(app).get("/api/state").json()
    assert body["trigger"]["available"] is True
    assert body["trigger"]["state"] is None
    assert body["trigger"]["dart_count"] is None


def test_handle_live_event_trigger_state_updates_appstate_and_broadcasts_dart_count(
    package_root,
):
    """Simulates the REAL production event shape -- capture_daemon.py's
    on_event callback emits exactly `{"type": "TRIGGER_STATE", "state":
    ..., "session": ..., "dart_count": ...}` (see
    run_capture_loop_body()'s three _emit(...) call sites) -- through the
    real AppState._handle_live_event() coroutine opendarts/live/run_product.py's
    live_event_queue wiring ultimately invokes for every one, via a real
    asyncio.run() (same pattern as
    test_start_background_tasks_never_schedules_a_calibration_poll above,
    since this project's suite doesn't depend on pytest-asyncio). Proves
    both halves: AppState's own trigger_state/trigger_dart_count/
    trigger_last_event_utc update (what the NEXT page load's /api/state
    would report), and the SAME event is broadcast verbatim to every
    connected WebSocket client (what an ALREADY-open dashboard tab's
    ws.onmessage TRIGGER_STATE handler receives) -- a fake WebSocket
    object standing in for a real browser connection, following this
    module's own AppState._broadcast() dead-socket tolerance (a plain
    object with an async send_text is all that's required)."""
    import asyncio

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)
    assert state.trigger_state is None
    assert state.trigger_dart_count is None

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001 -- same module, intentional, matches this file's existing pattern of exercising AppState internals directly
            {
                "type": "TRIGGER_STATE",
                "state": "SETTLING",
                "session": "session-test",
                "dart_count": 1,
            }
        )

    asyncio.run(_run())

    # AppState itself updated -- this is what the NEXT /api/state call (or
    # a freshly-connecting dashboard tab's HELLO message) would report.
    assert state.trigger_state == "SETTLING"
    assert state.trigger_dart_count == 1
    assert state.trigger_last_event_utc is not None

    # And the SAME transition was broadcast to the already-connected
    # client, with dart_count carried through -- not dropped on the way
    # from the capture loop's event to the WebSocket wire.
    assert len(fake_ws.sent) == 1
    msg = json.loads(fake_ws.sent[0])
    assert msg["type"] == "TRIGGER_STATE"
    assert msg["state"] == "SETTLING"
    assert msg["dart_count"] == 1
    assert msg["session"] == "session-test"


def test_handle_live_event_trigger_state_without_dart_count_degrades_to_none(package_root):
    """Defensive: an older/malformed event dict missing "dart_count"
    entirely must not raise -- degrades to None, same honest-missing-data
    convention as everywhere else, not a KeyError."""
    import asyncio

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001
            {"type": "TRIGGER_STATE", "state": "IDLE", "session": "session-test"}
        )

    asyncio.run(_run())
    assert state.trigger_state == "IDLE"
    assert state.trigger_dart_count is None


def test_handle_live_event_trigger_state_passes_through_emitted_at_utc(package_root):
    """2026-09-01 latency-instrumentation task, purely additive: a real
    `emitted_at_utc` (source-stamped by run_capture_loop_body(), see that
    function's own on_event docstring) on the incoming event must be
    broadcast through VERBATIM alongside `ts` (the dequeue-time stamp
    this method computes itself) -- both present, neither replacing the
    other, so a consumer can compute the real dispatch-latency gap
    (`ts - emitted_at_utc`) instead of only ever seeing dequeue time."""
    import asyncio

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    source_stamp = "2026-09-01T15:19:26.123456+00:00"

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001
            {
                "type": "TRIGGER_STATE",
                "state": "SETTLING",
                "session": "session-test",
                "dart_count": 1,
                "emitted_at_utc": source_stamp,
            }
        )

    asyncio.run(_run())

    assert len(fake_ws.sent) == 1
    msg = json.loads(fake_ws.sent[0])
    assert msg["emitted_at_utc"] == source_stamp
    # `ts` (dequeue-time) is still present and, correctly, NOT the same
    # value as the source stamp -- they measure two different moments.
    assert msg["ts"] is not None
    assert msg["ts"] != source_stamp


def test_handle_live_event_trigger_state_passes_through_settle_duration_when_present(
    package_root,
):
    """2026-09-01, same latency-instrumentation task: settle_duration_s/
    straggler_camera pass through verbatim when the source event carries
    them (a real READY_TO_CAPTURE transition)."""
    import asyncio

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001
            {
                "type": "TRIGGER_STATE",
                "state": "READY_TO_CAPTURE",
                "session": "session-test",
                "dart_count": 1,
                "settle_duration_s": 0.15,
                "straggler_camera": 1,
            }
        )

    asyncio.run(_run())

    assert len(fake_ws.sent) == 1
    msg = json.loads(fake_ws.sent[0])
    assert msg["settle_duration_s"] == 0.15
    assert msg["straggler_camera"] == 1


def test_handle_live_event_trigger_state_without_settle_duration_stays_absent(package_root):
    """The common case (every non-READY_TO_CAPTURE transition, or an
    older emitter that predates this field): settle_duration_s/
    straggler_camera must be genuinely ABSENT from the broadcast payload,
    never a fabricated None -- a consumer checking `"settle_duration_s"
    in msg` needs that to mean something real."""
    import asyncio

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001
            {"type": "TRIGGER_STATE", "state": "SETTLING", "session": "session-test"}
        )

    asyncio.run(_run())

    assert len(fake_ws.sent) == 1
    msg = json.loads(fake_ws.sent[0])
    assert "settle_duration_s" not in msg
    assert "straggler_camera" not in msg


def test_handle_live_event_trigger_state_without_emitted_at_utc_degrades_to_none(package_root):
    """An event predating this field (or a test double that doesn't set
    it) must not raise, and must not fabricate a value equal to `ts` --
    absent stays absent, same honest-missing-data convention as the
    dart_count test above, so a consumer can genuinely tell "no source
    stamp available" apart from "zero measured dispatch latency"."""
    import asyncio

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001
            {"type": "TRIGGER_STATE", "state": "IDLE", "session": "session-test"}
        )

    asyncio.run(_run())
    # No exception raised is the primary assertion here (a KeyError would
    # have failed _run() above); nothing else to check against AppState
    # itself since emitted_at_utc is broadcast-only, never stored.


# --------------------------------------------------------------------------
# Expanded status-pill vocabulary + real Start/Stop/Calibrate feedback
#. Verified
# against the two-axis status shape
# before implementing (see PRIMARY_INFO/PHASE_DETAIL in the rendered
# page's own JS).
# --------------------------------------------------------------------------


def test_status_pill_primary_axis_covers_every_real_capture_loop_state(package_root):
    """The PRIMARY status axis (No live capture / Connecting / Stopped /
    Waiting / Throw / Takeout) must actually be shipped in the page's own
    JS with the real labels the pill renders -- not just implied by the
    old single-axis TRIGGER_STATE_INFO this replaced."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "PRIMARY_INFO" in html
    for key, label in (
        ("NONE", "No live capture"),
        ("CONNECTING", "Connecting"),
        ("STOPPED", "Stopped"),
        ("WAITING", "Waiting"),
        ("THROW", "Throw"),
        ("TAKEOUT", "Takeout"),
    ):
        assert f"{key}:" in html
        assert label in html


def test_status_pill_waiting_state_covers_both_starting_and_manual_calibrate(package_root):
    """2026-08-14: WAITING is ONE main state with two real sub-state
    reasons -- opening cameras + auto-calibrating on Start, or a manual
    mid-session recalibrate. Before this date the latter had NO
    primary-axis representation at all (the real "I press Calibrate and
    it still says Throw" bug) -- this guards both reasons stay wired to
    the same WAITING label, not just the Start-time one."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "manualCalibrating" in html
    assert "PRIMARY_INFO.WAITING" in html
    assert "Calibrating" in html
    # 2026-08-15, the substate is just 'starting' then 'calibrating',
    # with no long verbose substates -- the detail
    # text this test originally checked ("Calibrating -- recalibrating
    # now") was shortened to plain "Calibrating" (full detail still
    # reaches the action log via logAction(), just not the compact
    # fixed-width pill). No longer asserting the removed verbose phrase.


def test_status_pill_phase_axis_still_covers_every_real_throw_state(package_root):
    """The SECONDARY phase axis (opendarts/capture/throw_trigger.py's real
    ThrowState names) must still all be mapped -- same regression guard
    the old TRIGGER_STATE_INFO-based test had, now against PHASE_DETAIL."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "PHASE_DETAIL" in html
    for state_name in (
        "IDLE",
        "MOTION_DETECTED",
        "SETTLING",
        "READY_TO_CAPTURE",
        "TAKEOUT_WAITING",
    ):
        assert f"{state_name}:" in html


def test_status_pill_renders_capture_loop_lifecycle_ahead_of_stale_trigger_state(package_root):
    """The pill's own renderPill() JS must check capture_loop.running/
    starting BEFORE falling back to the trigger's last-known state -- a
    Stopped or Starting session must never keep showing a stale Throw/
    Takeout left over from a previous session. Structural proof (no real
    browser in this suite): the shipped renderPill() checks `cl.running`
    and `cl.starting` before ever consulting PHASE_DETAIL."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert "function renderPill()" in html
    idx_running_check = html.index("if (cl && !cl.running)")
    idx_starting_check = html.index("if (cl && cl.starting)")
    idx_phase_lookup = html.index("const detailFn = PHASE_DETAIL[trig.state]")
    assert idx_running_check < idx_phase_lookup
    assert idx_starting_check < idx_phase_lookup


def test_state_dict_capture_loop_includes_starting_and_last_start_error(package_root):
    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop(open_ok=True)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    client = TestClient(app)
    before = client.get("/api/state").json()["capture_loop"]
    assert before["starting"] is False
    assert before["last_start_error"] is None

    client.post("/api/start")
    during = client.get("/api/state").json()["capture_loop"]
    assert during["running"] is True
    assert during["starting"] is True, "still starting -- no TRIGGER_STATE has arrived yet this session"
    assert during["last_start_error"] is None


def test_api_start_records_honest_last_start_error_when_zero_cameras_open(package_root):
    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop(open_ok=False)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state
    client = TestClient(app)

    body = client.post("/api/start").json()
    assert body["ok"] is False
    assert state.capture_last_start_error == "no cameras opened"
    assert client.get("/api/state").json()["capture_loop"]["last_start_error"] == "no cameras opened"


def test_capture_starting_clears_on_the_first_real_trigger_state_event(package_root):
    """The real "no longer starting" signal is the capture loop's own
    first TRIGGER_STATE event this session -- not a timer, not the /api/
    start response itself (that response returns before the capture
    thread's own bootstrap even begins, see CaptureLoopController's
    ARCHITECTURE NOTE 1)."""
    import asyncio

    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop(open_ok=True)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state
    TestClient(app).post("/api/start")
    assert state.capture_starting is True

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001
            {"type": "TRIGGER_STATE", "state": "IDLE", "session": "s1", "dart_count": 0}
        )

    asyncio.run(_run())
    assert state.capture_starting is False


def test_stop_capture_clears_starting_even_mid_bootstrap(package_root):
    """A session ending (manual Stop or idle-timeout) while still
    "Starting" (e.g. a slow calibration racing an idle-timeout) must not
    leave the pill stuck showing Starting forever."""
    import asyncio

    from opendarts.live.capture_daemon import CaptureLoopController

    hub = _FakeHubForStartStop(open_ok=True)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state
    TestClient(app).post("/api/start")
    assert state.capture_starting is True

    async def _run() -> None:
        await state.stop_capture(reason="manual")

    asyncio.run(_run())
    assert state.capture_starting is False


def test_handle_live_event_calibration_status_updates_appstate_and_broadcasts_source(
    package_root,
):
    """Real, visible confirmation of Start's auto-calibrate step (bug #3)
    -- simulates the real production event shape
    opendarts.live.capture_daemon.run_capture_loop_body's "Calibration
    bootstrap" section actually emits (source="startup"), through the
    real AppState._handle_live_event() coroutine. Proves both halves:
    AppState's own calibration_status/calibration_checked_at_utc update
    (what the next /api/state or the Cameras tab reflects), and the SAME
    `source` is broadcast verbatim so an already-open dashboard tab can
    tell this apart from a manual Calibrate click's own broadcast (which
    never sets `source` at all -- see the "no double-log" test below)."""
    import asyncio


    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)
    assert state.calibration_checked_at_utc is None

    calib = _synthetic_calibration()

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001
            {"type": "CALIBRATION_STATUS", "source": "startup", "calibrations": {0: calib}}
        )

    asyncio.run(_run())

    assert state.calibration_checked_at_utc is not None
    assert state.calibration_status[0]["ok"] is True

    assert len(fake_ws.sent) == 1
    msg = json.loads(fake_ws.sent[0])
    assert msg["type"] == "CALIBRATION_STATUS"
    assert msg["source"] == "startup"
    assert msg["cameras"]["0"]["ok"] is True


def test_handle_live_event_calibration_status_reused_source_is_distinct(package_root):
    import asyncio

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    calib = _synthetic_calibration()

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001
            {"type": "CALIBRATION_STATUS", "source": "startup_reused", "calibrations": {0: calib}}
        )

    asyncio.run(_run())
    assert state.calibration_status[0]["ok"] is True


def test_manual_calibration_refresh_broadcast_never_sets_source(package_root):
    """A manual "Calibrate" click's own broadcast (POST
    /api/calibration/refresh -> _broadcast_calibration_status) must NEVER
    carry `source` -- that field only ever comes from the async, Start-
    triggered auto-calibrate bootstrap. The dashboard's JS relies on this
    distinction to avoid double-logging a manual click's own already-
    visible result (see the WS CALIBRATION_STATUS handler in the rendered
    page)."""
    import asyncio

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    async def _run() -> None:
        await state._broadcast_calibration_status() # noqa: SLF001

    asyncio.run(_run())
    assert len(fake_ws.sent) == 1
    msg = json.loads(fake_ws.sent[0])
    assert "source" not in msg


def test_capture_loop_status_broadcast_on_real_start_and_stop(package_root):
    """Every connected dashboard tab (not just the one that clicked) must
    see Start/Stop reflected live -- direct fix for bug #2 ("start/stop
    buttons do something in the background but the pill doesn't reflect
    it").

    2026-08-12 EARLY-BROADCAST FIX: a successful start_capture() now
    broadcasts CAPTURE_LOOP_STATUS TWICE, not once -- an early "starting"
    broadcast before hub.open_all() is awaited, then a final "opened ok"
    broadcast after (see AppState.start_capture()'s own docstring for the
    full incident this fixes: the pill used to only learn "Starting" once
    camera-opening had ALREADY finished). Plus stop_capture()'s own
    broadcast, that's 3 total for one full start-then-stop cycle, not 2."""
    import asyncio

    from opendarts.live.capture_daemon import CaptureLoopController

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    hub = _FakeHubForStartStop(open_ok=True)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    async def _run() -> None:
        await state.start_capture()
        await state.stop_capture(reason="manual")

    asyncio.run(_run())

    types_seen = [json.loads(m)["type"] for m in fake_ws.sent]
    assert types_seen.count("CAPTURE_LOOP_STATUS") == 3
    early_msg = json.loads(fake_ws.sent[types_seen.index("CAPTURE_LOOP_STATUS")])
    assert early_msg["starting"] is True
    # The early broadcast fires BEFORE hub.open_all() -- controller.meta()
    # at that instant still honestly reports "running": False (request_start()
    # hasn't been called yet), same "no cameras" shape open_all() hasn't
    # populated yet either (no "cameras" key at all on this message).
    assert early_msg["running"] is False
    assert "cameras" not in early_msg
    final_msg = json.loads(fake_ws.sent[types_seen.index("CAPTURE_LOOP_STATUS", 1)])
    assert final_msg["starting"] is True
    assert final_msg["running"] is True
    assert final_msg["cameras"] == [True, True]
    stop_msg = json.loads(fake_ws.sent[len(types_seen) - 1 - types_seen[::-1].index("CAPTURE_LOOP_STATUS")])
    assert stop_msg["starting"] is False
    assert stop_msg["running"] is False


def test_capture_loop_status_early_broadcast_fires_before_open_all_resolves(package_root):
    """THE real ordering proof for bug #2 item 1 -- not just "two messages
    eventually arrive" but that the FIRST one is sent and observable
    BEFORE the slow hub.open_all() call resolves, exactly the property
    the project's complaint depends on ("Start button takes a long time to
    visibly reach 'Starting'"). Mocks open_all() with an artificial delay
    (a real asyncio.Event a background task waits on, not a fragile
    real-wall-clock sleep) and asserts the early broadcast has already
    happened while that delay is still unresolved."""
    import asyncio

    from opendarts.live.capture_daemon import CaptureLoopController

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    class _SlowHub(_FakeHubForStartStop):
        """open_all() blocks on a real threading.Event until the test
        explicitly releases it -- runs on a real worker thread (this
        object's open_all() is invoked via asyncio.to_thread in
        start_capture()), so blocking here does not stall the event loop
        the assertions below run on."""

        def __init__(self) -> None:
            super().__init__(open_ok=True)
            self.release_event = threading.Event()
            self.open_call_started = threading.Event()

        def open_all(self):
            self.open_call_started.set()
            self.release_event.wait(timeout=5.0)
            return super().open_all()

    hub = _SlowHub()
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    async def _run() -> None:
        start_task = asyncio.create_task(state.start_capture())
        # Wait for open_all() to have actually been entered (i.e. the
        # early guards passed and start_capture() reached the slow call)
        # without relying on a fixed sleep duration.
        await asyncio.to_thread(hub.open_call_started.wait, 5.0)
        # THE assertion that matters: the early broadcast + capture_starting
        # flip must have already happened while open_all() is STILL
        # blocked (release_event not yet set) -- proving ordering, not
        # just eventual state.
        assert hub.release_event.is_set() is False
        assert state.capture_starting is True
        assert len(fake_ws.sent) == 1
        early_msg = json.loads(fake_ws.sent[0])
        assert early_msg["type"] == "CAPTURE_LOOP_STATUS"
        assert early_msg["starting"] is True

        hub.release_event.set()
        await start_task

    asyncio.run(_run())
    # After open_all() resolves, the final broadcast has also gone out.
    types_seen = [json.loads(m)["type"] for m in fake_ws.sent]
    assert types_seen.count("CAPTURE_LOOP_STATUS") == 2


def test_capture_loop_status_broadcast_on_failed_start_carries_the_reason(package_root):
    """2026-08-12 EARLY-BROADCAST FIX: a FAILED start_capture() also now
    broadcasts twice -- the early "starting" broadcast (fired before
    open_all() even runs, so it has no way yet to know the attempt will
    fail) and the final "no cameras opened" failure broadcast, not just
    one message as before."""
    import asyncio

    from opendarts.live.capture_daemon import CaptureLoopController

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    hub = _FakeHubForStartStop(open_ok=False)
    controller = CaptureLoopController()
    app = create_app(
        package_root=package_root, enable_background_poll=False, local_hub=hub, controller=controller
    )
    state = app.state.opendarts_state
    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    async def _run() -> None:
        await state.start_capture()

    asyncio.run(_run())
    assert len(fake_ws.sent) == 2
    early_msg = json.loads(fake_ws.sent[0])
    assert early_msg["type"] == "CAPTURE_LOOP_STATUS"
    assert early_msg["ok"] is True
    assert early_msg["starting"] is True

    final_msg = json.loads(fake_ws.sent[1])
    assert final_msg["type"] == "CAPTURE_LOOP_STATUS"
    assert final_msg["ok"] is False
    assert final_msg["reason"] == "no cameras opened"
    # capture_starting must not stay stuck True forever on a failed start
    # -- see self.capture_starting's own __init__ docstring for the third
    # clearing transition this fix required.
    assert final_msg["starting"] is False
    assert state.capture_starting is False


def test_action_log_and_shared_busy_state_are_wired_into_every_control_button(package_root):
    """Real, persistent, timestamped feedback for every Start/Stop/Reset/
    Calibrate click (bug #2's direct fix) -- not just a transient button-
    label change. Checks the actual shipped JS wires logAction()/
    updateActionLine()/setControlsBusy() into all four button handlers,
    following this file's own established "assert the literal shipped
    structure" pattern for JS-behavior tests that can't run a real
    browser."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert 'id="action-log"' in html
    assert "function logAction(" in html
    assert "function updateActionLine(" in html
    assert "function setControlsBusy(" in html
    assert "CONTROL_BUTTON_IDS" in html
    assert "'btn-start'" in html and "'btn-stop'" in html
    assert "'btn-reset'" in html and "'btn-refresh-calib'" in html

    for handler_id in ("btn-start", "btn-stop", "btn-reset", "btn-refresh-calib"):
        start_idx = html.index(f"document.getElementById('{handler_id}').onclick")
        # The real end-of-handler marker is a "};" that sits ALONE at the
        # start of a line (the outer arrow function's own close, right
        # after the finally block's closing "}" on the line above) -- a
        # bare "};" is NOT unique to the handler's end: btn-start's own
        # optimistic `pillCaptureLoop = { running: false, starting: true
        # };` assignment also ends in "};", but on the SAME line as other
        # content, which used to truncate this search window before
        # reaching the real logAction()/updateActionLine() calls below.
        end_idx = html.index("\n};", start_idx)
        block = html[start_idx:end_idx]
        assert "setControlsBusy(true)" in block
        assert "logAction(" in block
        assert "updateActionLine(" in block


def test_auto_calibrate_confirmation_wired_into_websocket_handler(package_root):
    """Bug #3's direct fix: the WS CALIBRATION_STATUS handler must react
    to `source` (only ever set by the async Start-triggered bootstrap, see
    test_manual_calibration_refresh_broadcast_never_sets_source above) by
    updating the pill's Starting detail AND logging a real action-log
    line -- not silently ignoring the extra field."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert "msg.source === 'startup'" in html
    assert "startingCalibDetail" in html
    assert "'Calibrate (auto, on Start)'" in html


# --------------------------------------------------------------------------
# AD ground truth surfaced in discover_packages()/GET /api/packages -- see
# docs/DESIGN.md's "Dashboard wiring spec for opendarts/live/server.py" section.
# A package with no ad_ground_truth.json on disk must cleanly omit (None)
# every AD field, exactly like the pre-existing sector/ring/etc. fields
# already degrade to None when there's no result.json.
# --------------------------------------------------------------------------


def test_api_packages_ad_fields_absent_when_no_ad_ground_truth_json(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    assert not (throw_dir / "ad_ground_truth.json").exists()

    app = create_app(package_root=package_root, enable_background_poll=False)
    pkg = TestClient(app).get("/api/packages").json()[0]

    for key in (
        "ad_matched",
        "ad_match_reason",
        "ad_sector",
        "ad_ring",
        "ad_tip_xy_mm",
        "ad_method",
        "sector_match",
        "tip_distance_mm",
    ):
        assert pkg[key] is None, f"{key} should be None with no ad_ground_truth.json, got {pkg[key]!r}"


def test_api_packages_surfaces_ad_match_when_sector_and_ring_agree(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir, sector="S20", ring="single", board_xy_mm=(12.3, -4.5))
    _attach_ad_gt(throw_dir, matched=True, sector="S20", ring="single", tip_xy_mm=(15.3, -0.5))

    app = create_app(package_root=package_root, enable_background_poll=False)
    pkg = TestClient(app).get("/api/packages").json()[0]

    assert pkg["ad_matched"] is True
    assert pkg["ad_match_reason"] == "ok"
    assert pkg["ad_sector"] == "S20"
    assert pkg["ad_ring"] == "single"
    assert pkg["ad_tip_xy_mm"] == [15.3, -0.5]
    assert pkg["ad_method"] == "UnanimousCam"
    assert pkg["sector_match"] is True
    # opendarts board_xy_mm=(12.3,-4.5), AD tip_xy_mm=(15.3,-0.5) --
    # hypot(15.3-12.3, -0.5-(-4.5)) = hypot(3.0, 4.0) = 5.0mm, a real
    # measured value (3-4-5 triangle), not a guessed tolerance.
    assert pkg["tip_distance_mm"] == pytest.approx(5.0)


def test_api_packages_sector_match_false_when_sectors_disagree(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir, sector="S20", ring="single")
    _attach_ad_gt(throw_dir, matched=True, sector="S5", ring="single")

    app = create_app(package_root=package_root, enable_background_poll=False)
    pkg = TestClient(app).get("/api/packages").json()[0]

    assert pkg["ad_matched"] is True
    assert pkg["ad_sector"] == "S5"
    assert pkg["sector_match"] is False


def test_api_packages_derived_fields_none_when_ad_ground_truth_did_not_match(package_root):
    """A real ad_ground_truth.json can exist with matched=False (e.g. AD's
    fetch was too stale, or unreachable) -- see
    opendarts/live/ad_ground_truth.py's _no_match(). sector_match/
    tip_distance_mm must stay None (never compared against an untrusted
    non-match), and ad_sector/ad_ring reflect the real shape of a
    _no_match() result: also None."""
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir, sector="S20", ring="single")
    _attach_ad_gt(
        throw_dir,
        matched=False,
        sector=None,
        ring=None,
        tip_xy_mm=None,
        match_reason="stale: fetched 20.0s after opendarts capture (window=12.0s)",
    )

    app = create_app(package_root=package_root, enable_background_poll=False)
    pkg = TestClient(app).get("/api/packages").json()[0]

    assert pkg["ad_matched"] is False
    assert "stale" in pkg["ad_match_reason"]
    assert pkg["ad_sector"] is None
    assert pkg["ad_ring"] is None
    assert pkg["sector_match"] is None
    assert pkg["tip_distance_mm"] is None


def test_scoring_tab_html_has_ad_comparison_columns(package_root):
    # 2026-08-13: restructured from one row-per-throw with side-
    # by-side "sector"/"AD sector" columns into a row GROUP per throw --
    # a single AD reference row (its own "sector" cell IS the AD sector,
    # no separate column needed anymore) followed by one row per engine
    # that scored it, each showing PASS/FAIL against the AD row above.
    # "AD sector" as a literal column header is gone by design -- replaced
    # by engineSectionsFor()'s row-per-engine structure and the AD row's
    # own "AD" engine-column label.
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "AD sector" not in html
    assert "AD ring" not in html
    assert "<th>session</th>" not in html
    assert "<th>throw</th>" not in html
    assert "<th>path</th>" not in html
    assert "<th>engine</th>" in html
    assert "PASS" in html and "FAIL" in html
    assert "engineSectionsFor" in html
    assert "fmtDeltaXy" in html
    assert "tip" in html and "mm" in html
    assert "fmtSectorRing" in html


def test_scoring_tab_captured_time_renders_in_browser_local_time(package_root):
    """2026-08-12: times display in local time -- fmtCaptured() now
    converts the stored UTC timestamp to the browser's local time zone via
    `new Date(iso)` + local getters, instead of just trimming the raw UTC
    ISO string. Checking the shipped JS source for the real mechanism
    (not a UTC-string-slice) rather than trying to fake a browser time
    zone in this backend-only test suite."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "function fmtCaptured" in html
    assert "new Date(iso)" in html
    assert "d.getHours()" in html
    # The old UTC-string-slicing approach must be gone, not just
    # supplemented -- a leftover regex trim would silently keep showing
    # UTC even with the new function present.
    assert "d.getFullYear()" in html


def test_scoring_tab_ad_wrong_button_shown_only_when_theres_a_miss(package_root):
    """The AD-wrong control shows only for a REAL miss -- an engine
    disagreeing with an AD answer that actually arrived (sector_match ===
    false) -- or a throw a human already marked. When AD is offline / never
    matched, sector_match is null (not false), so the control stays hidden
    rather than making every throw look like a miss (2026-09-21: this was
    the row-clutter fix; previously the gate was `sector_match !== true`,
    which treated a missing AD answer as a miss). An already-marked throw
    still always shows (badge + Unmark). White-box source check, same style
    as the rest of this file's shipped-JS-behavior tests."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "function fmtAdWrongCell(p, sections)" in html
    assert "sector_match === false" in html  # a real miss needs AD to have answered
    assert "sector_match !== true" not in html  # the old, AD-offline-blind gate is gone
    assert "ad-wrong-btn" in html
    assert "AD Wrong" in html
    assert "Unmark" in html
    assert "ad_operator_marked_wrong" in html


def test_scoring_tab_manual_ad_refresh_button_removed(package_root):
    """The manual "Fetch AD ground truth now" button was removed on
    2026-08-12 once opendarts/live/ad_ws_listener.py began attaching
    ground truth automatically, in real time. The backing endpoint was
    kept for a year as a scriptable escape hatch and then removed too
    (2026-09-09) -- an API audit found nothing had ever called it: no UI,
    no harness, no script, only these tests. Ground truth is attached at
    package-save time by capture_daemon._attach_ad_ground_truth_from_ws();
    batch repair lives in dev/ad/backfill_ad_ground_truth.py."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert 'id="btn-refresh-ad"' not in html
    assert 'id="ad-refresh-status"' not in html
    assert "Fetch AD ground truth now" not in html
    # And the endpoint is gone -- not merely unreferenced.
    assert not any(r.path == "/api/ad-ground-truth/refresh" for r in app.routes)












# --------------------------------------------------------------------------
# "Start new session view" -- a PURELY VISUAL, client-side filter on the
# Scoring table. Explicitly, deliberately NOT a
# data-deletion/archival feature -- see docs/DESIGN.md's "Replay is the
# source of truth". No real
# browser JS engine exists in this suite (same standing gap as the status
# pill's own tests above) -- what's proven here: (1) the real UI elements
# exist; (2) the client-side filter predicate's semantics are correct,
# cross-checked against the LITERAL expression actually shipped in the
# page so this test can't silently drift from the real code; (3) the two
# new buttons never make a network call, structurally proving "purely
# client-side, nothing sent to the server" rather than just asserting it
# in a comment; (4) every real packet-arrival path (initial load, HELLO,
# PACKAGES_UPDATED) routes through the same ingestPackages() function, so
# a live-pushed new throw cannot bypass the filter; (5) the server itself
# never filters/suppresses PACKAGE_SAVED broadcasts for any reason related
# to this feature -- proving a live throw really would still reach the
# client during an active filter; (6) no delete/file-removal code path
# exists anywhere in server.py, and using the feature leaves the real
# on-disk package count/files completely unchanged.
# --------------------------------------------------------------------------


def test_scoring_tab_html_has_start_new_session_view_controls(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert 'id="btn-new-session-view"' in html
    # 2026-08-13, the verbiage was replaced by a small Clear
    # button -- the two long explanatory paragraphs above the
    # table are gone; the button itself is now labeled "Clear" (the real
    # non-destructive explanation moved into its title= tooltip instead
    # of taking up visible page space).
    assert "Clear" in html
    assert 'id="btn-show-all"' in html
    # "Show all" starts hidden -- only appears once a filter is active
    # (see renderFilterIndicator()'s own JS). No margin-left needed on
    # its own since 2026-08-14 -- the buttons live in a flex row with
    # `gap` now (see .scoring-header-buttons), not manually spaced.
    assert 'id="btn-show-all" class="action" type="button" style="display:none;">' in html
    assert 'id="view-filter-indicator"' in html

    # Non-destructive-sounding VISIBLE labels only (the task's own explicit
    # instruction) -- checked against the actual button/indicator text the
    # user reads, not the whole page (which legitimately discusses "not
    # deleted"/"no delete...code path" in JS comments and this file's own
    # doc-comments explaining what the feature deliberately is NOT).
    new_view_label = html[html.index('id="btn-new-session-view"') :].split(">", 1)[1].split("<")[0]
    show_all_label = html[html.index('id="btn-show-all"') :].split(">", 1)[1].split("<")[0]
    assert new_view_label.strip() == "Clear"
    assert show_all_label.strip() == "Show all throws"
    for banned_word in ("clear all", "delete", "archive", "remove data", "purge"):
        assert banned_word not in new_view_label.lower()
        assert banned_word not in show_all_label.lower()


def test_new_session_view_buttons_are_purely_client_side_no_network_call(package_root):
    """Structural proof (not just a code comment's say-so) that clicking
    either button never talks to the server at all -- extracts each
    button's actual onclick handler body out of the rendered page and
    asserts neither references any /api endpoint."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    def _handler_body(marker: str) -> str:
        start = html.index(marker)
        end = html.index("};", start)
        return html[start:end]

    new_view_body = _handler_body("btn-new-session-view').onclick")
    show_all_body = _handler_body("btn-show-all').onclick")

    assert "/api" not in new_view_body
    assert "/api" not in show_all_body
    # The only real work these handlers do: flip the marker, re-render.
    assert "viewFilterSinceMs = Date.now()" in new_view_body
    assert "renderScoringTable()" in new_view_body
    assert "viewFilterSinceMs = null" in show_all_body
    assert "renderScoringTable()" in show_all_body


def test_client_side_filter_predicate_matches_shipped_js(package_root):
    """The view filter's actual decision logic lives in client-side JS
    (server.py's renderScoringTable()) -- there is no real browser JS
    engine in this suite (same standing gap as this file's other
    HTML/JS-structure tests). This test proves two things together: (1)
    the EXACT comparison expression mirrored below is what's actually
    shipped, by asserting the literal JS substring is present in the
    rendered page (so this test cannot silently drift out of sync with
    the real code); (2) that expression's semantics -- comparing
    Date.parse(p.captured_at_utc) against a numeric epoch-millisecond
    marker, not a raw string compare -- are correctly reproduced here in
    Python via datetime.fromisoformat(), against a REAL captured_at_utc
    value written by the real save_throw_package(), not a synthetic
    string this test invented. (A raw string compare was deliberately
    rejected: Python's isoformat() and JS's Date.now()/toISOString()
    produce differently-shaped ISO-8601 strings -- 6-digit microseconds +
    "+00:00" vs 3-digit milliseconds + "Z" -- so lexicographic comparison
    between the two forms is not reliably correct in every case.)"""
    from datetime import datetime, timezone

    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    shipped_predicate = "Date.parse(p.captured_at_utc) >= viewFilterSinceMs"
    assert shipped_predicate in html, (
        "the shipped JS filter predicate changed -- update this test's "
        "Python mirror below to match the new expression"
    )

    def _client_filter_predicate(captured_at_utc, since_ms: float) -> bool:
        """Python mirror of the exact JS expression asserted above."""
        if not captured_at_utc:
            return False
        return datetime.fromisoformat(captured_at_utc).timestamp() * 1000 >= since_ms

    # A real package, real save_throw_package() timestamp.
    before_dir = package_root / "session-test" / "throw_before"
    _write_sample_package(before_dir, sector="S1")
    before_ts = json.loads((before_dir / "meta.json").read_text())["captured_at_utc"]
    before_epoch_ms = datetime.fromisoformat(before_ts).timestamp() * 1000

    marker_ms = before_epoch_ms + 5000 # a marker set 5s after that throw
    after_ts = datetime.fromtimestamp(marker_ms / 1000 + 5, tz=timezone.utc).isoformat()
    exactly_at_marker_ts = datetime.fromtimestamp(marker_ms / 1000, tz=timezone.utc).isoformat()

    assert _client_filter_predicate(before_ts, marker_ms) is False # captured before -- hidden
    assert _client_filter_predicate(after_ts, marker_ms) is True # captured after -- shown
    assert _client_filter_predicate(exactly_at_marker_ts, marker_ms) is True # inclusive (">=")
    assert _client_filter_predicate(None, marker_ms) is False # unknown timestamp -- hidden, not crashed


def test_ingest_packages_is_the_single_entry_point_for_every_live_data_path(package_root):
    """Every real place packages reach the page -- initial /api/packages
    load, the WebSocket HELLO message, and the WebSocket PACKAGES_UPDATED
    push -- must call the SAME ingestPackages() function (not the raw
    renderPackages() renderer directly), which is what makes item 3 (a
    live-pushed new throw still appearing while a filter is active) true:
    there is no separate "live" code path that could bypass the filter.
    A regression here (e.g. a future change calling renderPackages()
    directly again for one of these paths) would silently break "new
    throws still show up live while filtered" without any other test
    catching it."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "ingestPackages(await pkgResp.json())" in html
    assert html.count("ingestPackages(msg.packages)") == 2 # HELLO + PACKAGES_UPDATED
    # The raw renderer is never called directly from any of those three
    # ingestion call sites anymore -- only from inside renderScoringTable()
    # itself.
    assert "renderPackages(await pkgResp.json())" not in html
    assert "renderPackages(msg.packages)" not in html


def test_check_package_count_and_resync_wired_to_both_hello_and_packages_updated(
    package_root,
):
    """2026-08-24 fix (see events_ws()'s own dated docstring for the
    real incident this closes: a WebSocket outage silently dropped
    throws older than the newest-20 cap on reconnect). The self-heal
    (checkPackageCountAndResync()) must run after EVERY ingestPackages()
    call, not just one of the two live paths -- a mismatch can occur on
    a fresh HELLO (a new tab landing between loadInitial() and the
    first HELLO) just as much as on a PACKAGES_UPDATED push."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert html.count("checkPackageCountAndResync(msg.count)") == 2 # HELLO + PACKAGES_UPDATED

    start = html.index("async function checkPackageCountAndResync")
    end = html.index("\n}", start)
    body = html[start:end]
    # The actual self-heal: on a real mismatch, refetch the UNCAPPED
    # /api/packages (not the 20-capped WebSocket payload) and re-ingest
    # through the same single entry point every other live path uses
    # (see test_ingest_packages_is_the_single_entry_point_for_every_live_data_path
    # above) -- never a second, parallel rendering path.
    assert "fetch('/api/packages')" in body
    assert "ingestPackages(fresh)" in body
    # The full list REPLACES what the tab has: merging never dropped a
    # package that had left the disk, so the counts never matched again.
    assert "allKnownPackages.clear();" in body
    # Guarded on a real mismatch, not an unconditional refetch on every
    # single message -- that would defeat the whole point of capping
    # the wire payload in the first place.
    assert "count === allKnownPackages.size" in body


def test_package_saved_live_event_broadcasts_full_data_so_client_filter_still_works(package_root):
    """The view filter is 100% client-side (see server.py's
    ingestPackages()/renderScoringTable() design-decision comment) -- the
    SERVER never knows about any marker and must never suppress data
    because of one. This proves the real live-push path
    (AppState._handle_live_event's PACKAGE_SAVED branch -- the exact
    coroutine opendarts/live/run_product.py's capture-loop thread drives via
    live_event_queue for every real saved throw) broadcasts BOTH an
    already-old package and a brand-new one, unfiltered, to every
    connected client -- exactly the full payload ingestPackages() needs on
    the client side to correctly keep the new one and hide the old one
    (proven algorithmically in
    test_client_side_filter_predicate_matches_shipped_js above) even while
    "Start new session view" is active."""
    import asyncio

    class _FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, data: str) -> None:
            self.sent.append(data)

    old_dir = package_root / "session-test" / "throw_old"
    _write_sample_package(old_dir, sector="S1")

    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    assert len(state.list_packages()) == 1

    fake_ws = _FakeWebSocket()
    state.clients.add(fake_ws)

    new_dir = package_root / "session-test" / "throw_new"
    _write_sample_package(new_dir, sector="S20")

    async def _run() -> None:
        await state._handle_live_event( # noqa: SLF001 -- same module, matches this file's existing pattern
            {"type": "PACKAGE_SAVED", "path": str(new_dir), "session": "session-test"}
        )

    asyncio.run(_run())

    assert len(fake_ws.sent) == 1
    msg = json.loads(fake_ws.sent[0])
    assert msg["type"] == "PACKAGES_UPDATED"
    sectors = {p["sector"] for p in msg["packages"]}
    # BOTH the pre-existing and brand-new package are present -- the
    # server does not filter anything out for any "session view" concept;
    # that decision is made entirely client-side.
    assert sectors == {"S1", "S20"}


def test_no_delete_or_file_removal_code_path_for_view_filter_feature(package_root):
    """docs/DESIGN.md's "Replay is the source of truth": every throw package must
    stay on disk, forever, replayable, UNLESS a human explicitly asks
    for real deletion through its own dedicated, confirmed control. This
    feature (the Clear/"Show all throws" VIEW FILTER, purely client-side)
    is explicitly NOT that operation -- proven two ways: (1) a static sweep of
    ONLY the view-filter buttons' own onclick handler bodies (same
    extraction technique test_new_session_view_buttons_are_purely_client_
    side_no_network_call above already uses, so this can't drift from
    what "the feature" actually means) for any delete/removal-shaped
    call turns up nothing; (2) using the feature end-to-end (loading the
    dashboard, fetching packages -- the real HTTP traffic "Clear" and
    "Show all throws" would ever generate is exactly zero, per that same
    neighboring test, so this is already the full extent of what "using
    the feature" can mean at the server level) leaves the raw on-disk
    package count and discover_packages()'s own return value completely
    unchanged."""
    import re

    _write_sample_package(package_root / "session-test" / "throw_1")
    _write_sample_package(package_root / "session-test" / "throw_2")
    before = discover_packages(package_root)
    assert len(before) == 2

    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    html = client.get("/?ui=classic").text

    def _handler_body(marker: str) -> str:
        start = html.index(marker)
        end = html.index("};", start)
        return html[start:end]

    view_filter_source = (
        _handler_body("btn-new-session-view').onclick") + _handler_body("btn-show-all').onclick")
    )
    dangerous_patterns = [
        r"\.unlink\(",
        r"\brmtree\(",
        r"\bos\.remove\(",
        r"\bos\.rmdir\(",
        r"shutil\.rmtree",
        r"\.rmdir\(",
        r"/api/packages/delete",
    ]
    for pattern in dangerous_patterns:
        assert not re.search(pattern, view_filter_source), (
            f"found dangerous pattern {pattern!r} in the view-filter buttons' own handlers"
        )

    client.get("/api/packages")

    after = discover_packages(package_root)
    assert len(after) == 2
    assert {p["throw_id"] for p in after} == {p["throw_id"] for p in before}
    for pkg in before:
        assert Path(pkg["path"]).exists()
        assert (Path(pkg["path"]) / "meta.json").exists()


# --------------------------------------------------------------------------
# GET /api/logs/{name} -- read-only log tail, no SSH/elevated access
# needed.
# --------------------------------------------------------------------------


def test_api_logs_unknown_name_returns_404(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).get("/api/logs/not-a-real-log")
    assert resp.status_code == 404
    assert resp.json()["ok"] is False


def test_api_logs_missing_file_returns_200_with_explanation(package_root, monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "DEFAULT_LOG_DIR", tmp_path / "nonexistent_logs")
    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).get("/api/logs/run_product")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    assert "no log file yet" in resp.text


def test_api_logs_returns_tail_of_real_file(package_root, monkeypatch, tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    lines = [f"line {i}" for i in range(10)]
    (log_dir / "server.log").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(server_module, "DEFAULT_LOG_DIR", log_dir)

    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).get("/api/logs/server?n=3")

    assert resp.status_code == 200
    body_lines = resp.text.strip("\n").split("\n")
    assert body_lines == ["line 7", "line 8", "line 9"]


# --------------------------------------------------------------------------
# Row numbers on the Scoring table ("dart N")
# (2026-08-12) Rows are numbered so a specific throw, e.g. an AD miss,
# can be referred to as "dart 5". Numbering is
# CHRONOLOGICAL (oldest = 1) within whatever's currently shown (full
# history, or the "since session start" filtered set from the earlier
# clear-view feature) -- see renderPackages()'s own comment in server.py
# for why oldest=1 was chosen over newest=1: a new throw landing must
# never renumber an EARLIER row that's already been talked about as
# "dart 5" mid-conversation. No real browser JS engine exists in this
# suite (same standing gap as this file's other HTML/JS-structure tests)
# -- the literal shipped expression is asserted so this test can't
# silently drift out of sync with the real code, and its semantics are
# proven with a direct Python mirror.
# --------------------------------------------------------------------------


def test_scoring_table_has_row_number_column(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert "<th>#</th>" in html
    # colspan history, because this number has moved three times and the
    # reason matters more than the value: 8 -> 7 on 2026-08-13, when the
    # row-per-engine restructure removed the separate "AD sector" column
    # (engines became ROWS, not paired columns -- see
    # engineSectionsFor()); then 7 -> 8 on 2026-09-08, when the
    # "delta lat ms" column landed between `match` and `delta tip mm`;
    # then 8 -> 9 on 2026-09-21, when View moved out of the `match` cell
    # into a column of its own (a corrected throw put four badges in that
    # cell and clipped one of them).
    # The assertion pair is the point: the OLD value must be fully gone,
    # never left sitting alongside the new one in some untouched
    # template branch.
    assert 'colspan="9"' in html
    assert 'colspan="8"' not in html


def test_scoring_table_has_one_col_width_per_header_column(package_root):
    """The table is `table-layout: fixed`, which hands out the <colgroup>
    widths by POSITION. Adding View to the header on 2026-09-21 without a
    matching <col> shifted every later width one column left and left
    `board xy` with none -- it rendered off the edge of the table, empty.
    Counting only colspan (the test above) could not catch that; this pins
    <col> count == <th> count == colspan, and that the widths still fill
    the table."""
    import re
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    table = html[html.index('id="packages-table"'):]
    table = table[:table.index("</table>")]
    cols = re.findall(r"<col class=\"(col-[\w-]+)\">", table)
    header = table[table.index("<thead>"):table.index("</thead>")]
    ths = re.findall(r"<th[ >]", header)
    assert len(cols) == len(ths) == 9, (cols, len(ths))
    assert cols[1] == "col-view", "View's width must sit at View's position"
    assert '<th>#</th><th class="col-view"></th>' in header, "View sits beside #"

    css = (Path(__file__).resolve().parents[1]
           / "opendarts/live/dashboard/app.css").read_text()
    widths = {name: int(w) for name, w in re.findall(
        r"#packages-table col\.(col-[\w-]+) \{ width: (\d+)%", css)}
    assert set(cols) <= set(widths), f"no CSS width for {set(cols) - set(widths)}"
    assert sum(widths[c] for c in cols) == 100


def test_row_number_formula_is_chronological_oldest_is_1(package_root):
    """Proves the literal shipped formula, then mirrors its semantics in
    Python against a fake newest-first list -- exactly the shape
    renderScoringTable() passes to renderPackages()."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert "const n = packages.length;" in html
    assert "(n - i)" in html, (
        "the shipped row-number formula changed -- update this test's "
        "Python mirror to match"
    )

    def _row_numbers(packages_newest_first: list) -> list[int]:
        n = len(packages_newest_first)
        return [n - i for i in range(n)]

    fake_newest_first = ["newest", "b", "c", "d", "oldest"]
    numbers = _row_numbers(fake_newest_first)
    assert numbers == [5, 4, 3, 2, 1]
    by_name = dict(zip(fake_newest_first, numbers))
    assert by_name["oldest"] == 1
    assert by_name["newest"] == 5


def test_row_numbers_are_scoped_to_the_currently_visible_set_not_full_history(package_root):
    """renderPackages() derives `n` from its OWN `packages` argument, and
    renderScoringTable() always calls it with `visible` (the post-filter
    list), never `all` -- so when "Start new session view" is active,
    numbering restarts within the filtered set, since that is what
    'dart 5' means to someone watching a live session. Proven structurally (the real call site),
    not just asserted in a comment."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert "renderPackages(visible, emptyMessage)" in html
    assert "renderPackages(all," not in html


def test_row_numbers_stay_stable_as_new_throws_arrive(package_root):
    """The actual reason oldest=1 beats newest=1: an EARLIER throw's
    number must never change when a LATER one is captured, so "dart 5"
    keeps meaning the same physical throw for the rest of a live
    conversation. Same Python mirror as the formula test above, this time
    simulating a new throw arriving (prepended, since the real list is
    newest-first)."""

    def _row_numbers(packages_newest_first: list) -> dict:
        n = len(packages_newest_first)
        return {p: n - i for i, p in enumerate(packages_newest_first)}

    before = ["d4", "d3", "d2", "d1"]
    numbers_before = _row_numbers(before)
    assert numbers_before["d1"] == 1
    assert numbers_before["d4"] == 4

    after = ["d5"] + before # a new throw lands, prepended (newest-first)
    numbers_after = _row_numbers(after)
    assert numbers_after["d1"] == 1 # unchanged
    assert numbers_after["d4"] == 4 # unchanged
    assert numbers_after["d5"] == 5 # only the new throw gets a new number


# --------------------------------------------------------------------------
# POST /api/packages/{session}/{throw_id}/mark-ad-wrong -- the operator's
# "AD was wrong on this throw" toggle: opendarts scored a throw T1, correctly, while AD's own ground
# truth disagreed/showed a miss -- a HUMAN judgment call opendarts cannot
# make algorithmically. Adapted from real prior art: an operator-driven
# mark_ad_wrong(). Durable persistence is opendarts.capture.throw_package.
# mark_operator_ad_wrong() -- see tests/test_ad_ground_truth.py for that
# function's own focused unit tests (placeholder creation, preserving
# real AD data, idempotency, toggle/unmark, backward compat); this
# section proves the SERVER wiring on top of it: the real HTTP route,
# path-traversal guard, WebSocket broadcast, discover_packages()
# surfacing, and the Scoring tab's HTML controls.
# --------------------------------------------------------------------------


def test_mark_ad_wrong_persists_and_returns_updated_package(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir, sector="T1", ring="treble")
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    resp = client.post(
        "/api/packages/session-test/throw_1/mark-ad-wrong",
        json={"wrong": True, "note": "AD showed a miss, opendarts correctly scored T1"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["operator_marked_wrong"] is True
    assert body["operator_note"] == "AD showed a miss, opendarts correctly scored T1"
    assert body["package"]["ad_operator_marked_wrong"] is True

    # Really landed on disk (not just an in-memory response) -- the real
    # opendarts.capture.throw_package.mark_operator_ad_wrong() ran.
    from opendarts.capture.throw_package import load_ad_ground_truth as _load_gt

    gt = _load_gt(throw_dir)
    assert gt.operator_marked_wrong is True
    assert gt.operator_note == "AD showed a miss, opendarts correctly scored T1"

    # /api/packages reflects it immediately, no server restart needed.
    pkg = client.get("/api/packages").json()[0]
    assert pkg["ad_operator_marked_wrong"] is True
    assert pkg["ad_operator_note"] == "AD showed a miss, opendarts correctly scored T1"


def test_mark_ad_wrong_toggle_unmarks(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    client.post(
        "/api/packages/session-test/throw_1/mark-ad-wrong", json={"wrong": True, "note": "x"}
    )
    resp = client.post("/api/packages/session-test/throw_1/mark-ad-wrong", json={"wrong": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["operator_marked_wrong"] is False
    assert body["operator_note"] is None
    assert body["package"]["ad_operator_marked_wrong"] is False


def test_mark_ad_wrong_is_idempotent_when_called_repeatedly(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    for _ in range(3):
        resp = client.post(
            "/api/packages/session-test/throw_1/mark-ad-wrong",
            json={"wrong": True, "note": "same"},
        )
        assert resp.status_code == 200
        assert resp.json()["operator_marked_wrong"] is True

    for _ in range(2):
        resp = client.post("/api/packages/session-test/throw_1/mark-ad-wrong", json={"wrong": False})
        assert resp.status_code == 200
        assert resp.json()["operator_marked_wrong"] is False


def test_mark_ad_wrong_defaults_wrong_to_true_when_omitted(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).post("/api/packages/session-test/throw_1/mark-ad-wrong", json={})
    assert resp.status_code == 200
    assert resp.json()["operator_marked_wrong"] is True


def test_mark_ad_wrong_returns_404_for_unknown_package(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).post(
        "/api/packages/session-test/throw_does_not_exist/mark-ad-wrong", json={"wrong": True}
    )
    assert resp.status_code == 404
    assert resp.json()["ok"] is False


def test_mark_ad_wrong_rejects_path_traversal_reaching_the_handler(package_root):
    """session/throw_id build a real filesystem path under package_root --
    same defensive discipline as GET /api/logs/{name}'s VALID_LOG_NAMES
    allowlist. A ".." with no "/" reaches this route's own handler (proven
    directly, not just relying on Starlette's own routing normalization
    for slash-containing traversal attempts, which never dispatches to
    this app at all -- confirmed separately, both are safe, but this test
    exercises THIS endpoint's own explicit validation code path)."""
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    resp = client.post("/api/packages/session-test/..secret/mark-ad-wrong", json={"wrong": True})
    assert resp.status_code == 400
    assert resp.json()["ok"] is False

    resp2 = client.post("/api/packages/..secret/throw_1/mark-ad-wrong", json={"wrong": True})
    assert resp2.status_code == 400
    assert resp2.json()["ok"] is False


def test_mark_ad_wrong_creates_placeholder_for_throw_with_no_prior_ad_data(package_root):
    """The literal "AD MISSES" case -- AD never registered the throw at
    all (no ad_ground_truth.json on disk), yet the operator still needs
    to flag it. One layer above tests/test_ad_ground_truth.py's own unit
    test for mark_operator_ad_wrong() -- this exercises it through the
    real HTTP route."""
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    assert not (throw_dir / "ad_ground_truth.json").exists()

    app = create_app(package_root=package_root, enable_background_poll=False)
    resp = TestClient(app).post(
        "/api/packages/session-test/throw_1/mark-ad-wrong",
        json={"wrong": True, "note": "AD never showed anything"},
    )
    assert resp.status_code == 200
    assert (throw_dir / "ad_ground_truth.json").exists()
    assert resp.json()["package"]["ad_matched"] is False
    assert resp.json()["package"]["ad_operator_marked_wrong"] is True


def test_mark_ad_wrong_broadcasts_over_websocket(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    with client.websocket_connect("/api/events") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "HELLO"

        resp = client.post(
            "/api/packages/session-test/throw_1/mark-ad-wrong",
            json={"wrong": True, "note": "flagged live"},
        )
        assert resp.status_code == 200

        msg = ws.receive_json()
        assert msg["type"] == "PACKAGES_UPDATED"
        pkgs = {p["throw_id"]: p for p in msg["packages"]}
        assert pkgs["throw_1"]["ad_operator_marked_wrong"] is True
        assert pkgs["throw_1"]["ad_operator_note"] == "flagged live"


def test_mark_ad_wrong_unmark_also_broadcasts(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    client.post("/api/packages/session-test/throw_1/mark-ad-wrong", json={"wrong": True})

    with client.websocket_connect("/api/events") as ws:
        ws.receive_json() # HELLO
        resp = client.post("/api/packages/session-test/throw_1/mark-ad-wrong", json={"wrong": False})
        assert resp.status_code == 200
        msg = ws.receive_json()
        assert msg["type"] == "PACKAGES_UPDATED"
        pkgs = {p["throw_id"]: p for p in msg["packages"]}
        assert pkgs["throw_1"]["ad_operator_marked_wrong"] is False


def test_api_packages_surfaces_operator_fields_default_false_none_when_never_marked(package_root):
    throw_dir = package_root / "session-test" / "throw_1"
    _write_sample_package(throw_dir)
    app = create_app(package_root=package_root, enable_background_poll=False)
    pkg = TestClient(app).get("/api/packages").json()[0]
    assert pkg["ad_operator_marked_wrong"] is False
    assert pkg["ad_operator_note"] is None


def test_scoring_tab_html_has_ad_wrong_controls_and_tally_bar(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert 'id="engine-tally-bar"' in html
    assert "ad-wrong-btn" in html
    assert "AD Wrong" in html
    assert "AD wrong (human)" in html # the human-asserted indicator badge
    assert "/mark-ad-wrong" in html
    assert "renderEngineTally" in html
    assert "computeEngineTally" in html


def test_engine_tally_scoped_to_visible_packages_python_mirror(package_root):
    """2026-08-13: the old AD-only "N of M flagged wrong" summary was
    replaced by a running per-engine (AD included) correct/total tally
    wrong as well." Still scoped to `visible` -- the same post-filter set
    row numbers use -- not allKnownPackages as a whole. Proven the same
    way as this file's other HTML/JS-structure tests: assert the literal
    shipped scoping, then mirror the count logic in Python."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert "renderEngineTally(visible)" in html

    def _compute_tally(visible: list) -> dict:
        # Mirrors computeEngineTally()'s real JS logic: AD counted only
        # for throws with real ground truth (ad_matched is not False),
        # "correct" = not operator-flagged wrong; each engine section
        # counted for every throw it ran on, "correct" = sector_match.
        counts: dict[str, list[int]] = {}

        def bump(name: str, is_correct: bool) -> None:
            c = counts.setdefault(name, [0, 0])
            c[1] += 1
            if is_correct:
                c[0] += 1

        for p in visible:
            if p.get("ad_matched") is not False:
                bump("AD", not p.get("ad_operator_marked_wrong"))
            for name, section in (p.get("engines") or {}).items():
                bump(name, section.get("sector_match") is True)
            if p.get("primary_engine"):
                bump(p["primary_engine"], p.get("sector_match") is True)
        return counts

    fake_visible = [
        {"ad_matched": True, "ad_operator_marked_wrong": True, "primary_engine": "Apollo",
         "sector_match": False, "engines": {"Talos": {"sector_match": True}}},
        {"ad_matched": True, "ad_operator_marked_wrong": False, "primary_engine": "Apollo",
         "sector_match": True, "engines": {"Talos": {"sector_match": True}}},
        {"ad_matched": False, "primary_engine": "Apollo", "sector_match": True, "engines": {}},
    ]
    tally = _compute_tally(fake_visible)
    assert tally["AD"] == [1, 2] # 1 correct (not flagged) of 2 throws with real AD data
    assert tally["Apollo"] == [2, 3]
    assert tally["Talos"] == [2, 2]
    assert _compute_tally([]) == {}


# --------------------------------------------------------------------------
# "AD Wrong" -> "...so WHICH one was actually right?".
#
# Persistence is tests/test_ad_ground_truth.py's job (the
# operator_confirmed_* fields on AdGroundTruth + mark_operator_ad_wrong's
# confirm/clear paths). THIS section proves the server-side consequence:
# discover_packages() grades every engine row against the human's
# confirmed segment instead of AD's own once one exists -- and, just as
# importantly, that the un-confirmed path is completely unchanged.
# --------------------------------------------------------------------------


def _write_two_engine_package(
    package_root: Path,
    *,
    primary_sector: str = "5",
    primary_ring: str = "single_outer",
    talos_sector: str = "20",
    talos_ring: str = "treble",
) -> Path:
    """One real throw package scored by two engines (primary Apollo +
    also-run Talos), written via the same save_throw_package()/
    write_other_engines_result() the live daemon uses."""
    from opendarts.capture.throw_package import write_other_engines_result
    from opendarts.engines.base import EngineResult

    dest_dir = package_root / "session-test" / "throw_1"
    save_throw_package(
        dest_dir=dest_dir,
        session="session-test",
        bg_frames_bgr={0: np.zeros((24, 32, 3), dtype=np.uint8)},
        dart_frames_bgr={0: np.full((24, 32, 3), 255, dtype=np.uint8)},
        calibrations={0: _synthetic_calibration()},
        result=ScoreResult(
            ok=True, sector=primary_sector, ring=primary_ring, board_xy_mm=(40.0, -110.0),
            triangulation=None, n_cameras_used=1,
        ),
    )
    write_other_engines_result(
        dest_dir,
        "Apollo",
        {"Talos": EngineResult(
            ok=True, sector=talos_sector, ring=talos_ring, board_xy_mm=(30.0, 95.0)
        )},
    )
    return dest_dir


def test_discover_packages_is_byte_identical_when_no_operator_confirmation(package_root):
    """THE regression gate for this feature. Marking AD wrong WITHOUT a
    confirmation (the pre-2026-08-13 behavior, and still exactly what the
    Unmark button and any older client send) must leave every single
    derived match/tip field untouched -- the operator-truth path is only
    ever taken when a confirmation actually exists.

    Compares the WHOLE package dict before and after, minus the operator
    fields that are legitimately supposed to change, rather than spot-
    checking a couple of keys."""
    throw_dir = _write_two_engine_package(package_root)
    _attach_ad_gt(throw_dir, sector="5", ring="single_outer", tip_xy_mm=(41.0, -111.0))

    before = discover_packages(package_root)[0]

    from opendarts.capture.throw_package import mark_operator_ad_wrong

    mark_operator_ad_wrong(throw_dir, True, "AD was wrong, no idea who was right")
    after = discover_packages(package_root)[0]

    changed = {k for k in before if before[k] != after[k]}
    assert changed == {"ad_operator_marked_wrong", "ad_operator_note"}
    # Named explicitly too, so a future refactor that quietly drops a key
    # from the dict can't make the set-comparison above pass vacuously.
    assert after["sector_match"] == before["sector_match"]
    assert after["tip_distance_mm"] == before["tip_distance_mm"]
    # Talos -> Talos
    assert after["engines"]["Talos"] == before["engines"]["Talos"]
    assert after["ad_operator_confirmed_source"] is None
    assert after["ad_operator_confirmed_ring"] is None


def test_operator_confirmation_regrades_every_engine_row_against_the_human_answer(package_root):
    """The real "before confirmation vs after confirmation" proof.

    Setup mirrors the project's own live case: AD says S5, the primary engine
    agrees with AD (so it looks right), Talos says T20 (so it looks
    wrong). A human who watched the throw confirms Talos had it. After
    that, Talos must show a match and the primary must not -- the whole
    grading flips, from the same on-disk engine results, purely because
    the truth object changed."""
    throw_dir = _write_two_engine_package(
        package_root,
        primary_sector="5", primary_ring="single_outer",
        talos_sector="20", talos_ring="treble",
    )
    _attach_ad_gt(throw_dir, sector="5", ring="single_outer", tip_xy_mm=(41.0, -111.0))

    before = discover_packages(package_root)[0]
    assert before["sector_match"] is True # primary agreed with AD
    # Talos -> Talos
    assert before["engines"]["Talos"]["sector_match"] is False # Talos (Talos) "wrong" vs AD
    assert before["tip_distance_mm"] is not None # real AD tip -> real delta

    from opendarts.capture.throw_package import mark_operator_ad_wrong

    mark_operator_ad_wrong(
        throw_dir, True, "watched it land in T20",
        confirmed_source="Talos", confirmed_sector="20", confirmed_ring="treble",
    )
    after = discover_packages(package_root)[0]

    assert after["engines"]["Talos"]["sector_match"] is True # vindicated
    assert after["sector_match"] is False # primary was the wrong one
    # AD's own raw answer is preserved untouched for display/comparison --
    # the confirmation annotates, it never rewrites what AD said.
    assert after["ad_sector"] == "5"
    assert after["ad_ring"] == "single_outer"
    assert after["ad_operator_confirmed_source"] == "Talos"
    assert after["ad_operator_confirmed_sector"] == "20"
    assert after["ad_operator_confirmed_ring"] == "treble"


def test_operator_confirmation_reports_no_tip_distance_rather_than_a_fabricated_one(package_root):
    """A human confirming "Talos had the right segment" asserts a
    SEGMENT, not a millimetre coordinate (docs/DESIGN.md, 2026-08-12: "x/y
    millimetre deltas vs AD are still not independently verified;
    crossing a sector/ring boundary vs AD is"). So tip_distance_mm must
    go honestly None once operator truth is in play -- NOT keep measuring
    against AD's tip, which the same human just rejected."""
    throw_dir = _write_two_engine_package(package_root)
    _attach_ad_gt(throw_dir, sector="5", ring="single_outer", tip_xy_mm=(41.0, -111.0))
    assert discover_packages(package_root)[0]["tip_distance_mm"] is not None

    from opendarts.capture.throw_package import mark_operator_ad_wrong

    mark_operator_ad_wrong(
        throw_dir, True, None,
        confirmed_source="Talos", confirmed_sector="20", confirmed_ring="treble",
    )
    pkg = discover_packages(package_root)[0]
    assert pkg["tip_distance_mm"] is None
    assert pkg["engines"]["Talos"]["tip_distance_mm"] is None # Talos -> Talos
    # AD's own tip is still surfaced for display -- only the DERIVED
    # comparison went None.
    assert pkg["ad_tip_xy_mm"] == [41.0, -111.0]


def test_operator_confirmation_grades_engines_on_a_throw_ad_never_saw(package_root):
    """The literal "AD MISSES" case: no AD ground truth at all, so every
    engine's sector_match was honestly None (nothing to compare against).
    A human confirmation gives those rows a real answer to be graded
    against for the first time."""
    throw_dir = _write_two_engine_package(package_root)
    assert not (throw_dir / "ad_ground_truth.json").exists()

    before = discover_packages(package_root)[0]
    assert before["sector_match"] is None
    assert before["engines"]["Talos"]["sector_match"] is None # Talos -> Talos

    from opendarts.capture.throw_package import mark_operator_ad_wrong

    mark_operator_ad_wrong(
        throw_dir, True, "AD never registered it; it was T20",
        confirmed_source="Talos", confirmed_sector="20", confirmed_ring="treble",
    )
    after = discover_packages(package_root)[0]
    assert after["engines"]["Talos"]["sector_match"] is True
    assert after["sector_match"] is False


def test_manual_confirmation_with_no_sector_still_grades(package_root):
    """A manually-entered bull/outer_bull/outside answer has a real ring
    and a legitimately-None sector. `_operator_truth_for()` keys off the
    RING for exactly this reason -- keying off sector would silently
    ignore every bull/miss confirmation."""
    throw_dir = _write_two_engine_package(
        package_root, primary_sector=None, primary_ring="bull", talos_sector="20", talos_ring="treble"
    )
    from opendarts.capture.throw_package import mark_operator_ad_wrong

    mark_operator_ad_wrong(
        throw_dir, True, "it was in the bull",
        confirmed_source="manual", confirmed_sector=None, confirmed_ring="bull",
    )
    pkg = discover_packages(package_root)[0]
    assert pkg["ad_operator_confirmed_source"] == "manual"
    assert pkg["ad_operator_confirmed_sector"] is None
    assert pkg["ad_operator_confirmed_ring"] == "bull"
    assert pkg["sector_match"] is True # primary said bull
    assert pkg["engines"]["Talos"]["sector_match"] is False # Talos (Talos) said T20


def test_unmark_clears_the_confirmation_and_restores_ad_grading(package_root):
    """Undoing the flag undoes its whole explanation -- and grading
    therefore falls all the way back to AD's own answer, identical to a
    package that was never flagged at all."""
    throw_dir = _write_two_engine_package(package_root)
    _attach_ad_gt(throw_dir, sector="5", ring="single_outer", tip_xy_mm=(41.0, -111.0))
    pristine = discover_packages(package_root)[0]

    from opendarts.capture.throw_package import mark_operator_ad_wrong

    mark_operator_ad_wrong(
        throw_dir, True, "note", confirmed_source="Talos",
        confirmed_sector="20", confirmed_ring="treble",
    )
    assert discover_packages(package_root)[0]["engines"]["Talos"]["sector_match"] is True

    mark_operator_ad_wrong(throw_dir, False)
    restored = discover_packages(package_root)[0]
    assert restored["ad_operator_confirmed_source"] is None
    assert restored["ad_operator_confirmed_sector"] is None
    assert restored["ad_operator_confirmed_ring"] is None
    assert restored["sector_match"] == pristine["sector_match"]
    assert restored["tip_distance_mm"] == pristine["tip_distance_mm"]
    assert restored["engines"] == pristine["engines"]


def test_mark_ad_wrong_route_accepts_and_persists_the_confirmation(package_root):
    """End-to-end through the real HTTP route the modal actually posts
    to, not just the persistence function underneath it."""
    throw_dir = _write_two_engine_package(package_root)
    _attach_ad_gt(throw_dir, sector="5", ring="single_outer", tip_xy_mm=(41.0, -111.0))
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    resp = client.post(
        "/api/packages/session-test/throw_1/mark-ad-wrong",
        json={
            "wrong": True,
            "note": "watched it land in T20",
            "confirmed_source": "Talos",
            "confirmed_sector": "20",
            "confirmed_ring": "treble",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["operator_confirmed_source"] == "Talos"
    assert body["operator_confirmed_sector"] == "20"
    assert body["operator_confirmed_ring"] == "treble"
    # The response's own package dict is already regraded, so the calling
    # tab's ingestPackages() re-render shows the flip immediately.
    # Talos -> Talos
    assert body["package"]["engines"]["Talos"]["sector_match"] is True
    assert body["package"]["sector_match"] is False

    # Really on disk, in the same single ad_ground_truth.json.
    from opendarts.capture.throw_package import load_ad_ground_truth as _load_gt

    gt = _load_gt(throw_dir)
    assert gt.operator_confirmed_source == "Talos"
    assert gt.operator_confirmed_ring == "treble"
    assert gt.sector == "5" # AD's own answer still there

    # And /api/packages agrees, no restart needed.
    pkg = client.get("/api/packages").json()[0]
    assert pkg["ad_operator_confirmed_source"] == "Talos"
    assert pkg["engines"]["Talos"]["sector_match"] is True


def test_mark_ad_wrong_route_without_confirmation_fields_still_works(package_root):
    """The old request shape ({"wrong": true} / {"wrong": false}) -- which
    the Unmark button still sends verbatim -- must remain valid with the
    three new optional fields absent."""
    throw_dir = _write_sample_package(package_root / "session-test" / "throw_1")
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)

    resp = client.post("/api/packages/session-test/throw_1/mark-ad-wrong", json={"wrong": True})
    assert resp.status_code == 200
    assert resp.json()["operator_confirmed_source"] is None
    assert resp.json()["package"]["ad_operator_confirmed_ring"] is None


def test_unmark_through_the_route_clears_the_confirmation(package_root):
    throw_dir = _write_two_engine_package(package_root)
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    client.post(
        "/api/packages/session-test/throw_1/mark-ad-wrong",
        json={"wrong": True, "confirmed_source": "Talos",
              "confirmed_sector": "20", "confirmed_ring": "treble"},
    )
    resp = client.post("/api/packages/session-test/throw_1/mark-ad-wrong", json={"wrong": False})

    assert resp.status_code == 200
    body = resp.json()
    assert body["operator_confirmed_source"] is None
    assert body["operator_confirmed_ring"] is None
    assert body["package"]["ad_operator_confirmed_ring"] is None
    assert body["package"]["engines"]["Talos"]["sector_match"] is None # back to no truth at all (Talos -> Talos)


def test_api_packages_confirmed_fields_default_none_when_never_confirmed(package_root):
    _write_sample_package(package_root / "session-test" / "throw_1")
    app = create_app(package_root=package_root, enable_background_poll=False)
    pkg = TestClient(app).get("/api/packages").json()[0]
    assert pkg["ad_operator_confirmed_source"] is None
    assert pkg["ad_operator_confirmed_sector"] is None
    assert pkg["ad_operator_confirmed_ring"] is None


def test_board_ring_names_matches_the_real_scorer_vocabulary():
    """The modal's manual ring picker must offer exactly the strings
    opendarts.geometry.board.sector_ring_for_point actually produces -- a
    confirmed segment spelled any other way could never compare equal to
    an engine's answer. board_ring_names() derives them by probing that
    function rather than hardcoding a list; this pins the real, measured
    result (inner-to-outer) so a silent change to either side fails."""
    from opendarts.geometry.board import sector_ring_for_point
    from opendarts.live.server import board_ring_names

    names = board_ring_names()
    assert names == [
        "bull", "outer_bull", "single_inner", "treble", "single_outer", "double", "outside",
    ]
    # Every name is genuinely reachable from the real scorer (not just a
    # list this test and the function happen to agree on).
    reachable = {sector_ring_for_point(0.0, r)[1] for r in (0.0, 10.0, 50.0, 103.0, 130.0, 166.0, 200.0)}
    assert reachable == set(names)


def test_dashboard_ships_the_which_was_right_modal(package_root):
    """White-box source check -- TestClient never executes JS, so this
    asserts the real shipped markup/handlers, the same convention the
    rest of this file's frontend tests use."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    for needle in [
        'id="ad-wrong-modal"',
        "Which was actually right?",
        'id="ad-wrong-options"',
        'id="ad-wrong-manual"',
        'id="ad-wrong-manual-sector"',
        'id="ad-wrong-manual-ring"',
        'id="ad-wrong-cancel"',
        'id="ad-wrong-submit"',
        "function openAdWrongModal",
        "function closeAdWrongModal",
        "None of these",
    ]:
        assert needle in html, needle

    # The modal ships CLOSED (a `hidden` attribute in the initial HTML,
    # not a JS-applied class), so it can never flash on page load.
    assert '<div id="ad-wrong-modal" class="modal-backdrop" hidden>' in html

    # Options are built from the throw's own engine sections -- the same
    # function the table rows use -- not a hardcoded engine list.
    assert "const sections = engineSectionsFor(p);" in html

    # The manual picker's vocabulary is injected from the real board
    # module, not typed into the template.
    assert "const BOARD_SECTORS = " in html
    assert "const BOARD_RINGS = " in html
    assert '"single_inner"' in html and '"outer_bull"' in html
    assert '"20", "1", "18"' in html or '"20","1","18"' in html


def test_dashboard_js_opens_the_modal_on_mark_and_posts_only_on_confirm(package_root):
    """The behavioral contract, checked against the real shipped source:
    clicking "AD Wrong" opens the modal (no request); Unmark posts
    straight away (nothing to ask); the modal's own Confirm is what sends
    the confirmed_* fields."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    # The row button no longer fetches directly -- it opens the modal.
    assert "openAdWrongModal(session, throwId);" in html
    # ...except on the unmark path, which posts immediately.
    assert "postMarkAdWrong(session, throwId, {wrong: false})" in html
    # One shared POST helper, so both paths go through the same
    # ingestPackages() re-render.
    assert "async function postMarkAdWrong(" in html
    assert "ingestPackages([body.package]);" in html
    # Confirm sends the real field names the route's MarkAdWrongRequest
    # declares.
    for field in ("confirmed_source:", "confirmed_sector:", "confirmed_ring:"):
        assert field in html, field
    # Cancel closes without any request of its own.
    assert "document.getElementById('ad-wrong-cancel').onclick = closeAdWrongModal;" in html


def test_ad_row_shows_the_confirmed_truth_and_fails_ad_when_it_differs(package_root):
    """DESIGN CALL, asserted rather than left implicit (see
    fmtAdWrongCell's own comment): the AD row normally carries no
    PASS/FAIL badge (it's the reference, not a graded row), but once a
    human confirms a DIFFERENT segment it gets an explicit FAIL plus a
    badge naming the confirmed answer and its source."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "if (p.ad_operator_confirmed_ring) {" in html
    assert "const adDiffers = (p.ad_operator_confirmed_sector !== p.ad_sector)" in html
    assert "if (adDiffers) html += fmtMatch(false) + ' ';" in html
    assert ">truth: ' +" in html


def test_engine_tally_prefers_operator_truth_python_mirror(package_root):
    """The tally's engine percentages need no separate preference logic --
    they read `sector_match`, which discover_packages() already computed
    against operator truth (asserted directly above in this file). The one
    thing that DID change in the JS is AD's own gate: a throw AD never saw
    stops being excluded as "no data" once a human confirms the real
    answer, because AD demonstrably didn't have it.

    Same proof convention as
    test_engine_tally_scoped_to_visible_packages_python_mirror above:
    assert the literal shipped condition, then mirror it in Python."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert "const adWasAsked = (p.ad_matched === true || p.ad_matched === false);" in html
    assert ("if (adWasAsked && (p.ad_matched !== false || p.ad_operator_confirmed_ring)) {"
            in html)
    assert "bump(s.name, s.sector_match === true);" in html

    def _ad_counted(p: dict) -> bool:
        # null/absent means AD was never ASKED (switched off for this
        # throw, so no ad_ground_truth.json exists at all) -- distinct from
        # False, which means asked and found nothing. Counting a
        # never-asked throw would credit AD with darts it never saw.
        was_asked = p.get("ad_matched") in (True, False)
        return was_asked and (
            p.get("ad_matched") is not False or bool(p.get("ad_operator_confirmed_ring"))
        )

    # AD switched off for this throw -> neither right nor wrong, and
    # crucially not silently counted as right.
    assert _ad_counted({}) is False
    assert _ad_counted({"ad_matched": None}) is False
    # Asked, nothing found, nobody confirmed -> neither right nor wrong.
    assert _ad_counted({"ad_matched": False}) is False
    # No AD data, but a human confirmed the real answer -> AD really did
    # miss this throw, so it counts (and against it, via the existing
    # `!ad_operator_marked_wrong` correctness term).
    assert _ad_counted({"ad_matched": False, "ad_operator_confirmed_ring": "treble"}) is True
    # Real AD data -> counted exactly as before.
    assert _ad_counted({"ad_matched": True}) is True


# --------------------------------------------------------------------------
# POST /api/packages/delete-all -- 2026-08-14, widened to both kinds
# 2026-09-17. The button DELETES this rig's recorded data: every throw
# package AND every frame-ring capture. The one deliberately real,
# permanent deletion this dashboard exposes -- gated entirely on the
# frontend's own explicit confirm(), which is itself tested here against
# the page the server really serves (see
# test_the_confirmation_names_the_real_counts_and_sizes).
# --------------------------------------------------------------------------


def _delete_handler_source(html: str) -> str:
    """The Delete button's onclick handler, as the browser receives it."""
    start = html.index("document.getElementById('btn-delete-recorded').onclick")
    return html[start:html.index("\n};", start)]


def _balanced_call_argument(js: str, call: str) -> str:
    """The literal argument expression of `call` in `js`.

    A paren counter that knows about single-quoted strings and their
    escapes -- enough for this dashboard's own style, and it means the
    string a human reads is taken from the page rather than retyped into
    the test.
    """
    i = js.index(call) + len(call)
    start, depth = i, 1
    while i < len(js):
        c = js[i]
        if c == "'":
            i += 1
            while i < len(js) and js[i] != "'":
                i += 2 if js[i] == "\\" else 1
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return js[start:i]
        i += 1
    raise AssertionError(f"unbalanced {call} in the served page")


def _write_sample_capture(
    capture_root: Path,
    name: str = "20260917T120000-000001-missed_dart",
    *,
    n_cameras: int = 3,
    bytes_per_frame: int = 4096,
    n_sets: int = 2,
) -> Path:
    """One capture directory, in the shape FrameDumpWriter really writes.

    Real directory, real files, real bytes -- `frames.bin` is sized per
    camera per set the way a dump from a three-camera rig is, so the byte
    totals these tests assert on are totals of something.
    """
    dest = capture_root / name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "frames.bin").write_bytes(b"\x01" * (bytes_per_frame * n_cameras * n_sets))
    (dest / "manifest.json").write_text(json.dumps({
        "schema": "frame-dump-v1",
        "kind": name.rsplit("-", 1)[-1],
        "slots": list(range(n_cameras)),
        "n_sets": n_sets,
        "n_frames": n_cameras * n_sets,
        "bytes_total": bytes_per_frame * n_cameras * n_sets,
    }))
    return dest


def _bytes_under(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


class TestDeleteRecordedData:
    """The Engines tab's "Delete recorded data" button, at the API level.

    WIDENED 2026-09-17 (this class was TestDeletePackages). The route
    deletes BOTH of the things a rig writes without bound -- throw
    packages and frame-ring captures -- because it used to delete only the
    first while the second, the larger by an order of magnitude, had
    nothing in the product that removed it at all. Every property the
    package half already had is still asserted here; the capture half is
    held to the same ones.
    """

    def _client(self, package_root, capture_root, **kwargs):
        app = create_app(package_root=package_root, capture_root=capture_root,
                         enable_background_poll=False, **kwargs)
        return TestClient(app)

    # -- packages, exactly as before ------------------------------------

    def test_deletes_every_real_package_from_disk(self, package_root, tmp_path):
        _write_sample_package(package_root / "session-a" / "throw_1")
        _write_sample_package(package_root / "session-a" / "throw_2")
        _write_sample_package(package_root / "session-b" / "throw_1")
        assert len(discover_packages(package_root)) == 3

        client = self._client(package_root, tmp_path / "captures")
        body = client.post("/api/packages/delete-all").json()
        assert body["ok"] is True
        assert body["deleted"] == 3
        assert body["packages"]["deleted"] == 3
        assert discover_packages(package_root) == []

    def test_package_root_itself_survives_the_delete(self, package_root, tmp_path):
        """Real deletion of package CONTENTS, not the directory a live
        capture_daemon.py process still expects to exist and write into
        on the very next throw."""
        _write_sample_package(package_root / "session-a" / "throw_1")
        self._client(package_root, tmp_path / "captures").post("/api/packages/delete-all")
        assert package_root.exists()
        assert package_root.is_dir()

    def test_missing_package_root_is_a_clean_no_op(self, package_root, tmp_path):
        """package_root doesn't even exist yet (fresh install, nothing
        ever captured) -- not an error."""
        assert not package_root.exists()
        client = self._client(package_root, tmp_path / "captures")
        body = client.post("/api/packages/delete-all").json()
        assert body["ok"] is True
        assert body["deleted"] == 0

    def test_broadcasts_packages_updated_to_connected_tabs(self, package_root, tmp_path):
        _write_sample_package(package_root / "session-a" / "throw_1")
        client = self._client(package_root, tmp_path / "captures")
        with client.websocket_connect("/api/events") as ws:
            ws.receive_json() # HELLO
            client.post("/api/packages/delete-all")
            msg = ws.receive_json()
            assert msg["type"] == "PACKAGES_UPDATED"
            assert msg["packages"] == []
            # The count is what makes every OTHER screen let go of what was
            # deleted: clients merge `packages`, so an empty list alone
            # removes nothing, and only a count mismatch makes them replace
            # their list. Without it a kiosk kept showing the deleted darts.
            assert msg["count"] == 0

    def test_deletes_the_persisted_throw_number_counter_for_each_session_removed(
        self, package_root, tmp_path
    ):
        """2026-08-22 fix: after a delete, dart numbering did not
        restart. Root cause: the NEXT throw's number comes from a counter
        persisted OUTSIDE package_root (opendarts.live.capture_daemon.
        handle_ready_to_capture()'s data/session_throw_counters/
        <session_id>.count, deliberately kept out of package_root so the
        pull/quarantine SOP can't sweep it -- see that function's own
        2026-08-17 dated comment), which delete-all used to leave
        completely untouched -- so a session's next throw after "delete
        everything" silently continued from wherever the old count left
        off instead of restarting at 001. This proves the counter file for
        a deleted session is gone too, so the next
        handle_ready_to_capture() call for that same still-live
        session_id hits the bootstrap path and restarts at
        throw_number=1."""
        counters_dir = package_root.parent / "session_throw_counters"
        counters_dir.mkdir(parents=True, exist_ok=True)
        (counters_dir / "session-a.count").write_text("61")
        (counters_dir / "session-b.count").write_text("9")
        _write_sample_package(package_root / "session-a" / "throw_1")
        _write_sample_package(package_root / "session-b" / "throw_1")

        client = self._client(package_root, tmp_path / "captures")
        assert client.post("/api/packages/delete-all").json()["ok"] is True

        assert not (counters_dir / "session-a.count").exists()
        assert not (counters_dir / "session-b.count").exists()

    def test_bumps_the_generation_for_each_session_removed(self, package_root, tmp_path):
        """2026-08-22, second half of the same fix: resetting the counter
        to 0 IN PLACE (the first version of this fix) still risks a real
        collision if some of a session's throws were already pulled off
        the rig before the rest got deleted here -- this process can't know
        what's already archived elsewhere. The delete now calls the
        shared _reset_session_throw_numbering() helper, which bumps a
        counting GENERATION instead -- proving that here directly, since
        it's the actual collision-avoidance mechanism, not just "the
        counter file is gone" (already covered by the test above)."""
        counters_dir = package_root.parent / "session_throw_counters"
        counters_dir.mkdir(parents=True, exist_ok=True)
        (counters_dir / "session-a.count").write_text("61")
        _write_sample_package(package_root / "session-a" / "throw_1")

        client = self._client(package_root, tmp_path / "captures")
        assert client.post("/api/packages/delete-all").json()["ok"] is True

        generation_file = counters_dir / "session-a.generation"
        assert generation_file.exists()
        assert generation_file.read_text() == "1"

    def test_a_session_that_fails_to_delete_keeps_its_counter_and_generation_too(
        self, package_root, tmp_path, monkeypatch
    ):
        """The counter/generation reset is scoped to sessions ACTUALLY
        removed -- mirrors the existing `deleted`/`errors` partial-failure
        handling. A session whose rmtree failed still has real throw
        packages sitting on disk, so its numbering must survive too (never
        renumber a still-real, on-disk throw's session from 0, and never
        bump its generation either -- nothing about it actually
        changed)."""
        counters_dir = package_root.parent / "session_throw_counters"
        counters_dir.mkdir(parents=True, exist_ok=True)
        (counters_dir / "session-a.count").write_text("61")
        _write_sample_package(package_root / "session-a" / "throw_1")

        def failing_rmtree(path, *a, **k):
            raise OSError("simulated rmtree failure")

        monkeypatch.setattr(server_module.shutil, "rmtree", failing_rmtree)
        client = self._client(package_root, tmp_path / "captures")
        body = client.post("/api/packages/delete-all").json()
        assert body["ok"] is False
        assert "session-a" in body["reason"]
        assert (counters_dir / "session-a.count").exists()
        assert (counters_dir / "session-a.count").read_text() == "61"
        assert not (counters_dir / "session-a.generation").exists()

    # -- captures, held to the same properties --------------------------

    def test_deletes_both_kinds_and_reports_the_counts_and_the_bytes(
        self, package_root, tmp_path
    ):
        """The whole point of the widening: one press clears the rig, and
        the answer says what it took, per kind, in real bytes."""
        capture_root = tmp_path / "captures"
        _write_sample_package(package_root / "session-a" / "throw_1")
        _write_sample_package(package_root / "session-a" / "throw_2")
        _write_sample_package(package_root / "session-b" / "throw_1")
        _write_sample_capture(capture_root, "20260917T120000-000001-missed_dart")
        _write_sample_capture(capture_root, "20260917T120500-000002-misscore")
        package_bytes = _bytes_under(package_root)
        capture_bytes = _bytes_under(capture_root)
        assert package_bytes > 0 and capture_bytes > 0

        client = self._client(package_root, capture_root)
        body = client.post("/api/packages/delete-all").json()

        assert body["ok"] is True
        assert body["packages"] == {
            "deleted": 3, "bytes": package_bytes,
            "label": frame_ring.format_bytes(package_bytes),
            "root": str(package_root),
        }
        assert body["captures"] == {
            "deleted": 2, "bytes": capture_bytes,
            "label": frame_ring.format_bytes(capture_bytes),
            "root": str(capture_root),
        }
        assert body["bytes"] == package_bytes + capture_bytes
        assert body["deleted"] == 3, "the top-level count is still throw packages"
        assert list(capture_root.iterdir()) == []
        assert discover_packages(package_root) == []

    def test_capture_root_itself_survives_the_delete(self, package_root, tmp_path):
        """Same reason package_root does: the writer's next dump has to
        have somewhere to land without anyone recreating it."""
        capture_root = tmp_path / "captures"
        _write_sample_capture(capture_root)
        self._client(package_root, capture_root).post("/api/packages/delete-all")
        assert capture_root.is_dir()

    def test_a_capture_that_fails_to_delete_is_reported_not_swallowed(
        self, package_root, tmp_path, monkeypatch
    ):
        capture_root = tmp_path / "captures"
        _write_sample_capture(capture_root, "20260917T120000-000001-misscore")

        def failing_rmtree(path, *a, **k):
            raise OSError("simulated rmtree failure")

        monkeypatch.setattr(server_module.shutil, "rmtree", failing_rmtree)
        body = self._client(package_root, capture_root).post(
            "/api/packages/delete-all").json()
        assert body["ok"] is False
        assert "20260917T120000-000001-misscore" in body["reason"]
        assert body["captures"]["deleted"] == 0
        assert body["captures"]["bytes"] == 0, "nothing was freed, so nothing is claimed"

    def test_it_deletes_what_the_capture_SERVICE_actually_wrote(
        self, package_root, tmp_path
    ):
        """No injected root at all: the route finds the capture root on
        the running service, which is the directory the writer really
        fills. A delete pointed at a different one would look like it
        worked and leave the files."""
        from opendarts.capture.frame_ring import FrameRing
        from opendarts.capture.throw_capture import ThrowCaptureService

        ring = FrameRing(22.0)
        for i in range(30):
            ring.append(
                {slot: np.full((8, 12, 3), (slot + i) % 255, dtype=np.uint8)
                 for slot in (0, 1, 2)},
                wall_s=1_757_000_000.0 + i / 30.0,
                monotonic_s=4321.0 + i / 30.0,
                generation=i,
            )
        service = ThrowCaptureService(ring, capture_root=tmp_path / "service-captures")
        service.capture_missed_dart(reason="a dart landed and nothing scored")
        service.writer.join()
        assert len(list((tmp_path / "service-captures").iterdir())) == 1

        app = create_app(package_root=package_root, enable_background_poll=False,
                         throw_capture=service)
        body = TestClient(app).post("/api/packages/delete-all").json()
        assert body["ok"] is True
        assert body["captures"]["deleted"] == 1
        assert body["captures"]["root"] == str(tmp_path / "service-captures")
        assert list((tmp_path / "service-captures").iterdir()) == []

    def test_a_capture_being_written_refuses_the_WHOLE_delete(
        self, package_root, tmp_path
    ):
        """A dump is mid-write for seconds at a time, into the directory
        this route is about to remove. Refusing both halves rather than
        doing the packages anyway keeps the button's meaning intact: it
        clears the rig, or it says why it did not."""
        capture_root = tmp_path / "captures"
        _write_sample_package(package_root / "session-a" / "throw_1")
        _write_sample_capture(capture_root)

        class _BusyWriter:
            def status(self):
                return {"busy": True, "current": {
                    "dest_dir": str(capture_root / "in-flight"),
                    "kind": "missed_dart"}}

        class _BusyService:
            ring = None
            writer = _BusyWriter()

        service = _BusyService()
        service.capture_root = capture_root
        client = self._client(package_root, capture_root, throw_capture=service)
        body = client.post("/api/packages/delete-all").json()

        assert body["ok"] is False
        assert "being written right now" in body["reason"]
        assert body["deleted"] == 0
        assert len(discover_packages(package_root)) == 1, "the packages went anyway"
        assert (capture_root / "20260917T120000-000001-missed_dart").exists()

    # -- nothing to delete ----------------------------------------------

    def test_empty_roots_are_a_clean_no_op_that_says_so(self, package_root, tmp_path):
        capture_root = tmp_path / "captures"
        package_root.mkdir(parents=True)
        capture_root.mkdir(parents=True)
        client = self._client(package_root, capture_root)

        counts = client.get("/api/recorded-data").json()
        assert counts["total"] == {"count": 0, "bytes": 0, "label": "0 B"}

        body = client.post("/api/packages/delete-all").json()
        assert body["ok"] is True
        assert body["deleted"] == 0
        assert body["packages"]["deleted"] == 0
        assert body["captures"]["deleted"] == 0
        assert body["bytes"] == 0

    def test_the_page_says_nothing_to_delete_rather_than_asking(
        self, package_root, tmp_path
    ):
        """The empty case never reaches a confirm() -- an "are you sure"
        over nothing is a dialog that teaches people to click through
        dialogs."""
        html = self._client(package_root, tmp_path / "captures").get("/?ui=classic").text
        handler = _delete_handler_source(html)
        assert "nothing to delete" in handler
        assert "no throw packages and no captures" in handler
        assert handler.index("nothing to delete") < handler.index("window.confirm(")

    # -- the counts the question is asked with ---------------------------

    def test_recorded_data_counts_and_weighs_both_kinds(self, package_root, tmp_path):
        capture_root = tmp_path / "captures"
        _write_sample_package(package_root / "session-a" / "throw_1")
        _write_sample_package(package_root / "session-a" / "throw_2")
        _write_sample_capture(capture_root, "20260917T120000-000001-missed_dart")
        client = self._client(package_root, capture_root)

        body = client.get("/api/recorded-data").json()
        assert body["ok"] is True
        assert body["busy"] is False
        assert body["packages"]["count"] == 2
        assert body["packages"]["bytes"] == _bytes_under(package_root)
        assert body["packages"]["root"] == str(package_root)
        assert body["captures"]["count"] == 1
        assert body["captures"]["bytes"] == _bytes_under(capture_root)
        assert body["captures"]["root"] == str(capture_root)
        assert body["total"]["count"] == 3
        assert body["total"]["bytes"] == (
            body["packages"]["bytes"] + body["captures"]["bytes"])

    def test_counting_deletes_nothing_and_creates_nothing(self, package_root, tmp_path):
        capture_root = tmp_path / "captures"
        _write_sample_package(package_root / "session-a" / "throw_1")
        _write_sample_capture(capture_root)
        listing = lambda: sorted(  # noqa: E731
            str(p) for p in list(package_root.rglob("*")) + list(capture_root.rglob("*")))
        before = listing()

        self._client(package_root, capture_root).get("/api/recorded-data")

        assert listing() == before

    def test_the_confirmation_names_the_real_counts_and_sizes(
        self, package_root, tmp_path
    ):
        """THE STRING A HUMAN READS, built by the page's own code from the
        route's own numbers -- not a mock of either.

        The served page's `countLabel`/`recordedDataPhrase` and the
        literal `window.confirm(...)` expression are lifted out of the
        HTML this server just returned and evaluated in node against the
        real payload of a real GET against a real fixture. If the wording
        or the arithmetic drifts, this fails with the sentence that would
        have been shown instead.
        """
        node = shutil.which("node")
        if not node:
            pytest.skip(
                "node not installed -- the confirmation text was NOT evaluated. "
                "A structural check cannot prove the sentence a human reads."
            )
        capture_root = tmp_path / "captures"
        for i in range(3):
            _write_sample_package(package_root / "session-a" / f"throw_{i}")
        _write_sample_capture(capture_root, "20260917T120000-000001-missed_dart")
        _write_sample_capture(capture_root, "20260917T120500-000002-misscore")

        client = self._client(package_root, capture_root)
        snapshot = client.get("/api/recorded-data").json()
        html = client.get("/?ui=classic").text

        helpers = html[html.index("function countLabel("):
                       html.index("document.getElementById('btn-delete-recorded')")]
        expr = _balanced_call_argument(_delete_handler_source(html), "window.confirm(")
        script = tmp_path / "confirm.js"
        script.write_text(helpers + "\nconst snap = " + json.dumps(snapshot) + ";\n"
                          + "process.stdout.write(" + expr + ");\n")
        proc = subprocess.run([node, str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        message = proc.stdout

        assert message.startswith(
            "Delete 3 throw packages (" + snapshot["packages"]["label"]
            + ") and 2 captures (" + snapshot["captures"]["label"]
            + ")? This cannot be undone."
        ), message
        assert str(package_root) in message
        assert str(capture_root) in message

    def test_the_confirmation_counts_one_of_each_in_the_singular(
        self, package_root, tmp_path
    ):
        node = shutil.which("node")
        if not node:
            pytest.skip("node not installed -- the confirmation text was NOT evaluated")
        capture_root = tmp_path / "captures"
        _write_sample_package(package_root / "session-a" / "throw_1")
        _write_sample_capture(capture_root)

        client = self._client(package_root, capture_root)
        snapshot = client.get("/api/recorded-data").json()
        html = client.get("/?ui=classic").text
        helpers = html[html.index("function countLabel("):
                       html.index("document.getElementById('btn-delete-recorded')")]
        script = tmp_path / "phrase.js"
        script.write_text(helpers + "\nprocess.stdout.write(recordedDataPhrase("
                          + json.dumps(snapshot) + "));\n")
        proc = subprocess.run([node, str(script)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.startswith("1 throw package (")
        assert " and 1 capture (" in proc.stdout

    # -- the gate in front of it ----------------------------------------

    def test_dashboard_delete_button_asks_for_confirmation_before_calling_the_api(
        self, package_root, tmp_path
    ):
        """Structural proof the button can't fire the real delete without
        a human confirming first -- the onclick handler's own body must
        reach window.confirm() before the destructive POST. The COUNTING
        fetch is deliberately allowed to come first: it reads and deletes
        nothing, and it is where the numbers in the question come from."""
        html = self._client(package_root, tmp_path / "captures").get("/?ui=classic").text
        handler = _delete_handler_source(html)
        assert "window.confirm(" in handler
        assert handler.index("window.confirm(") < handler.index("/api/packages/delete-all")

    def test_the_button_and_its_tooltip_say_recorded_data_not_packages(
        self, package_root, tmp_path
    ):
        """The label is the only description most operators will ever read
        of what this button takes, and it took captures silently for as
        long as it said "packages"."""
        html = self._client(package_root, tmp_path / "captures").get("/?ui=classic").text
        start = html.index('id="btn-delete-recorded"')
        button = html[start:html.index("</button>", start)]
        assert ">Delete recorded data" in button
        assert "throw packages" in button and "captures" in button
        assert "Delete packages<" not in html


# --------------------------------------------------------------------------
# AD's board-status indicator light (Scoring tab header, 2026-08-14).
# state_dict()'s "ad_board_status" key is read LIVE off the listener's own
# board_status() every single call -- no cached copy on AppState, so there
# is exactly one place this can ever be wrong (see that section of
# state_dict()'s own comment). _handle_live_event's AD_BOARD_STATUS branch
# is a PURE forwarded broadcast with no AppState field of its own to
# update -- unlike TRIGGER_STATE/VISIT_CLEARED, which do mutate AppState.
# The second external board's light/toggle was removed once it became a
# real registry engine -- only AD has status now.
# --------------------------------------------------------------------------


class _FakeBoardStatusListener:
    """Minimal duck-typed stand-in for AdWsListener -- only the methods
    AppState ever actually calls on it (start()/stop()/board_status()/
    board_status_age_sec() -- confirmed by grepping server.py's own
    self.ad_ws_listener. call sites, not assumed)."""

    def __init__(self, status: str, raw=None, age_sec=None):
        self._status = status
        self._raw = raw
        self._age_sec = age_sec
        self.start_calls = 0
        self.stop_calls = 0

    def board_status_age_sec(self):
        """Seconds in the current status -- what tells a normal takeout
        apart from a board wedged in one."""
        return self._age_sec

    def start(self):
        self.start_calls += 1

    def stop(self, timeout: float = 3.0):
        self.stop_calls += 1

    def board_status(self):
        return self._status, self._raw


class _FakeBroadcastWebSocket:
    """Same minimal stand-in test_handle_live_event_trigger_state_updates_
    appstate_and_broadcasts_dart_count already uses -- a plain object with
    an async send_text is all AppState._broadcast() actually requires."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


class TestBoardStatusIndicatorLights:
    def test_state_dict_reports_unknown_when_no_listener_is_wired(self, package_root):
        """This module's own standalone CLI (no listener passed) -- honest
        "unknown", never a fabricated status."""
        app = create_app(package_root=package_root, enable_background_poll=False)
        body = app.state.opendarts_state.state_dict()
        assert body["ad_board_status"] == BOARD_STATUS_UNKNOWN

    def test_state_dict_reflects_the_ad_listeners_real_board_status_live(self, package_root):
        listener = _FakeBoardStatusListener(BOARD_STATUS_TAKEOUT, {"event": "Throw detected"})
        app = create_app(package_root=package_root, enable_background_poll=False, ad_ws_listener=listener)
        body = app.state.opendarts_state.state_dict()
        assert body["ad_board_status"] == BOARD_STATUS_TAKEOUT

    def test_state_dict_reflects_a_change_in_the_listeners_own_board_status_between_calls(self, package_root):
        """No cached copy on AppState -- proven by mutating the fake
        listener's own returned status BETWEEN two state_dict() calls on
        the SAME AppState instance and confirming both reads are live, not
        a one-time snapshot taken at AppState construction time."""
        listener = _FakeBoardStatusListener(BOARD_STATUS_READY)
        app = create_app(package_root=package_root, enable_background_poll=False, ad_ws_listener=listener)
        state = app.state.opendarts_state
        assert state.state_dict()["ad_board_status"] == BOARD_STATUS_READY

        listener._status = BOARD_STATUS_TAKEOUT
        assert state.state_dict()["ad_board_status"] == BOARD_STATUS_TAKEOUT

    def test_handle_live_event_broadcasts_ad_board_status_verbatim_to_connected_clients(self, package_root):
        import asyncio

        app = create_app(package_root=package_root, enable_background_poll=False)
        state = app.state.opendarts_state
        fake_ws = _FakeBroadcastWebSocket()
        state.clients.add(fake_ws)

        async def _run() -> None:
            await state._handle_live_event( # noqa: SLF001 -- same module, intentional, matches this file's existing pattern
                {"type": "AD_BOARD_STATUS", "status": "ready"}
            )

        asyncio.run(_run())

        assert len(fake_ws.sent) == 1
        msg = json.loads(fake_ws.sent[0])
        assert msg["type"] == "AD_BOARD_STATUS"
        assert msg["status"] == "ready"
        # Every _broadcast() message here gets a server-side ts merged in
        # (`{**event, "ts": ts}`), same convention as TRIGGER_STATE/
        # VISIT_CLEARED -- confirms this branch follows that same shape.
        assert "ts" in msg

    def test_handle_live_event_never_mutates_appstate_for_a_board_status_event(self, package_root):
        """Per _handle_live_event's own comment: "No AppState field to
        update: state_dict() reads board_status() live off the listener
        itself" -- this branch is a pure forwarded broadcast, unlike
        TRIGGER_STATE/VISIT_CLEARED which DO write into AppState fields.
        Proven here by confirming state_dict()'s own ad_board_status stays
        exactly "unknown" (no listener wired in this app at all) both
        before and after an AD_BOARD_STATUS event is handled -- if this
        branch ever started writing a cached value onto AppState instead
        of relying purely on the live listener read, this would catch it
        going stale/wrong the moment a listener wasn't wired."""
        import asyncio

        app = create_app(package_root=package_root, enable_background_poll=False)
        state = app.state.opendarts_state
        before = state.state_dict()["ad_board_status"]

        async def _run() -> None:
            await state._handle_live_event({"type": "AD_BOARD_STATUS", "status": "ready"}) # noqa: SLF001

        asyncio.run(_run())

        after = state.state_dict()["ad_board_status"]
        assert before == after == BOARD_STATUS_UNKNOWN


# --------------------------------------------------------------------------
# Detection speed: values 1..5 consecutive still frames, fastest to
# safest. The guard is that the five values stay in the right order with
# the right direction of travel, so an operator cannot pick the opposite
# of what they meant, and that each option carries only its own label.
# --------------------------------------------------------------------------

def test_detection_speed_options_are_ordered_fast_to_safe_without_ad_labels(package_root):
    """Values 1..5 in order, 1 the fastest and 5 the safest, and no
    extra captions anywhere in the control."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    full_html = TestClient(app).get("/?ui=classic").text

    # Scope to THIS select first. The camera-timeout select above it also
    # has an <option value="5"> ("5 min"), so an unscoped search finds
    # the wrong element and the assertion reports a confusing mismatch
    # rather than a real inversion.
    assert "Detection speed" in full_html
    sel_start = full_html.index('<select id="detection-time-select">')
    html = full_html[sel_start:full_html.index("</select>", sel_start)]

    # Every value still offered, still in ascending order -- a reordering
    # would put "fastest" on the wrong end of the list.
    positions = []
    for frames in (1, 2, 3, 4, 5):
        marker = f'<option value="{frames}">'
        assert marker in html, f"{frames} frames is no longer offered"
        start = html.index(marker)
        positions.append(start)
        option = html[start:html.index("</option>", start)]
        # The number and the word "frame(s)" are what the operator is
        # actually choosing.
        assert f">{frames} frame" in option, option
        assert "(ad " not in option.lower(), (
            f"an extra caption is back on {frames} frame(s): {option!r}"
        )
    assert positions == sorted(positions)

    # The direction of travel, on the two extremes only -- these are the
    # ones an inversion would swap.
    assert "1 frame &mdash; fastest" in html
    assert "5 frames &mdash; safest" in html

    # And the note says what the trade actually costs, and nothing more.
    field = full_html[full_html.rindex("<label", 0, sel_start):
                      full_html.index("</label>", sel_start)]
    assert "fewer frames = faster but more errors" in field.lower()
    assert "what AD calls" not in field


def test_calibrate_click_puts_camera_badges_into_a_calibrating_state(package_root):
    """Pressing Calibrate must not leave a green "calibrated" badge
    standing. During a manual recalibration the badge is the one thing
    guaranteed not to describe current state -- the overlay was already
    pulled for exactly this reason (2026-08-16); the badge was missed.

    Asserts the wiring in the shipped client: the helper exists, the
    click handler calls it alongside the overlay reset, and the failure
    path re-reads real state instead of stranding the badge."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert "function setCalibratingBadges()" in html
    assert 'class="badge warn">calibrating' in html

    start = html.index("document.getElementById('btn-refresh-calib').onclick")
    # Slice to the handler's real end, not a fixed character count. This
    # was `start + 3000`, which silently truncated mid-handler the moment
    # anything was added above the catch block -- the assertions below
    # then failed for a reason that had nothing to do with what they
    # test. A window that has to be re-tuned whenever the code grows is
    # not a window.
    handler = html[start:html.index("\n};", start)]
    assert "setCalibratingBadges();" in handler, (
        "Calibrate's click handler must set the calibrating state, not only "
        "clear the overlay"
    )
    # It has to happen on CLICK, not after the POST resolves -- the whole
    # point is covering the in-flight window.
    assert handler.index("setCalibratingBadges();") < handler.index(
        "/api/calibration/refresh"
    )
    # And a failed request must not strand the badge mid-state.
    assert "calibration re-read after failure also failed" in handler


def test_ad_latency_column_is_coloured_by_who_answered_first(package_root):
    """Coloured by sign: a positive delta (our answer arrived first) is
    green, a negative one red. Exactly zero stays neutral: rounding can
    arrive there from either direction."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    start = html.index("function fmtLatency(")
    fn = html[start:html.index("\n}", start)]
    assert "lat-ahead" in fn and "lat-behind" in fn
    # Sign, not magnitude, decides the colour -- and zero picks neither.
    assert "v > 0 ? 'lat-ahead' : (v < 0 ? 'lat-behind' : '')" in fn
    assert ".lat-ahead {" in html and ".lat-behind {" in html


# --------------------------------------------------------------------------
# The camera assignment -- slot -> hardware device, 2026-09-10. It had a
# route pair of its own (`GET`/`POST /api/camera-devices`) until
# 2026-09-17; `camera_devices`/`camera_urls` are now keys of the config
# document, and what the hub is ACTUALLY reading rides in
# `runtime.cameras` beside them.
# --------------------------------------------------------------------------


class _StubHub:
    """Minimal stand-in: the endpoint reads `.configs[*].device` and calls
    `reconfigure()`. `refuse` makes it behave like an OPEN hub."""

    def __init__(self, devices, refuse=False):
        from opendarts.live.local_capture import CameraConfig

        self.configs = [CameraConfig(device=d) for d in devices]
        self.refuse = refuse

    def reconfigure(self, configs):
        if self.refuse:
            raise RuntimeError("cannot reconfigure while cameras are open (slots [0])")
        self.configs = list(configs)


def _client_with_hub(package_root, devices=(0, 1, 2), refuse=False):
    hub = _StubHub(devices, refuse=refuse)
    app = create_app(package_root=package_root, enable_background_poll=False,
                     local_hub=hub)
    return TestClient(app), hub


def test_camera_devices_get_reports_what_the_process_is_actually_using(package_root):
    """Read off the LIVE hub, not re-read from the file -- the two differ
    exactly when an assignment is saved but not restarted into, which is
    the state an operator most needs to see."""
    client, _hub = _client_with_hub(package_root, devices=(3, 1, 2))
    body = client.get("/api/config").json()
    assert body["runtime"]["cameras"]["devices"] == [3, 1, 2]
    # And the document itself reports the live assignment, not the file's,
    # for the same reason.
    assert body["config"]["camera_devices"] == [3, 1, 2]


def test_camera_devices_post_rejects_duplicate_slots(package_root, no_config_writes):
    """One camera cannot be two views of the board."""
    client, _hub = _client_with_hub(package_root)
    resp = client.patch("/api/config", json={"camera_devices": [1, 1, 2]})
    assert resp.status_code == 400
    assert "more than one slot" in resp.json()["errors"]["camera_devices"]


def test_camera_devices_post_rejects_bools_and_negatives(package_root, no_config_writes):
    client, _hub = _client_with_hub(package_root)
    for bad in ([True, 2, 3], [-1, 2, 3], ["0", 1, 2]):
        resp = client.patch("/api/config", json={"camera_devices": bad})
        assert resp.status_code == 400, bad
        assert "non-negative int" in resp.json()["errors"]["camera_devices"], bad


def test_camera_devices_post_rejects_empty(package_root, no_config_writes):
    client, _hub = _client_with_hub(package_root)
    resp = client.patch("/api/config", json={"camera_devices": []})
    assert resp.status_code == 400
    assert "non-empty" in resp.json()["errors"]["camera_devices"]


def test_camera_devices_post_validation_matches_the_config_loader(package_root, no_config_writes, tmp_path):
    """A value the document ACCEPTS must be one load_live_config() accepts
    on the way back up. If they drift, Save appears to work and is then
    silently discarded at the next start -- the worst failure this feature
    can have. It is now the rule for every key, not just this one: see
    opendarts/live/config_document.py's module docstring."""
    import json as _json

    from opendarts.live.config import load_live_config

    client, _hub = _client_with_hub(package_root)
    for candidate in ([1, 1, 2], [True, 2], [-1], []):
        accepted = client.patch(
            "/api/config", json={"camera_devices": candidate}
        ).json()["ok"]
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(_json.dumps({"camera_devices": candidate}))
        loader_kept = load_live_config(cfg_path).camera_devices is not None
        assert accepted == loader_kept, candidate


def test_ad_status_note_lives_inside_the_oracle_field(package_root):
    """The live Autodarts status shipped in a leftover div that wore the
    deleted camera-devices section's class -- a name with no CSS rule
    anywhere, so the line floated under the two-card grid, aligned with
    nothing and detached from the toggle it reports on, while repeating
    what the static note inside that toggle's card already said.

    Pinned because nothing else here can see it: the span exists, the id
    is right, renderAdConfig() fills it, and it is in the wrong place.
    """
    client, _hub = _client_with_hub(package_root)
    html = client.get("/?ui=classic").text

    # Inside the Autodarts card, after its select -- not in any container
    # that follows the closing </div> of the config-grid.
    field = html.split('<select id="ad-enabled-toggle">', 1)[1].split("</label>", 1)[0]
    assert 'id="ad-config-note"' in field, "the status note left the Autodarts field"
    # The orphan container, and the class it carried, are gone for good.
    assert "camera-devices-actions" not in html
    # One note in that card, not two: the static duplicate is gone and its
    # one unique fact moved into the off-state string in renderAdConfig().
    assert field.count("config-field-note") == 1
    # The static note's one unique fact: turning this off does something
    # beyond not-asking-Autodarts. It now covers BOTH halves of that --
    # publishing stops and the devices are unregistered (see
    # opendarts/live/vcam_register.py) -- so the sentence has to keep
    # describing what the switch really does, not just the frames.
    assert "the virtual cameras are removed" in html


# --------------------------------------------------------------------------
# Config tab: nothing on this page may move, and nothing on it may look
# like a different KIND of control than its neighbour. Both were live
# defects (2026-09-12) that no functional test could see, because the
# page kept working perfectly while it jumped around.
# --------------------------------------------------------------------------

def test_config_field_notes_reserve_space_so_a_changing_note_cannot_move_the_page(
        package_root):
    """#ad-config-note and #store-packages-note are live STATE, rewritten
    at every poll with strings of very different lengths. Without
    reserved height the field re-flowed on each change, and because
    .config-grid is a flex row (items stretched to the tallest) the field
    beside it moved too -- the whole tab shifted under the operator while
    they were reaching for a control."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    rule_start = html.index(".config-field-note {")
    rule = html[rule_start:html.index("}", rule_start)]
    assert "min-height" in rule, (
        "a note whose text changes needs reserved height, or the field "
        f"re-flows: {rule!r}"
    )
    # Enough for more than one line: a single-line reservation is the
    # same as none for the two-line strings renderAdConfig() produces.
    height = float(rule.split("min-height:")[1].split("em")[0].strip())
    assert height >= 2.8, f"min-height {height}em is under two lines"


def test_config_text_input_is_styled_like_the_selects_beside_it(package_root):
    """The Autodarts URL box used the UA's own styling -- a pale native
    input sitting in the same grid row as the dark selects, reading as a
    different kind of control rather than the same control holding
    text."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    start = html.index(".config-field select")
    selector = html[start:html.index("{", start)]
    assert ".config-field input" in selector, (
        "the text input must share the selects' rule, not fall back to "
        f"the browser default: {selector!r}"
    )
    rule = html[html.index("{", start):html.index("}", start)]
    # The four properties that made them look like different controls.
    for prop in ("background", "border", "padding", "font-family"):
        assert prop in rule, f"{prop} must be pinned for both: {rule!r}"


def test_camera_cards_do_not_stretch_to_each_other(package_root):
    """Opening ONE camera's Diagnostics disclosure grew all three cards.

    The <details> elements were never linked -- .cams is a flex row, and
    a flex row stretches every item to the height of the tallest, so the
    two untouched cards gained the same empty height and it read as
    "clicking one opened all of them"."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    start = html.index(".cams {")
    rule = html[start:html.index("}", start)]
    assert "align-items: flex-start" in rule, (
        f"camera cards must size to their own content: {rule!r}"
    )


def test_the_test_button_is_gated_on_this_browser_having_the_clips(package_root):
    """The gate keeps moving because what can fail keeps moving. It was
    two gates (Render clips on can_render, Test on available), then one
    (`a.available` off the server's coverage reply), and is now a fact
    about THIS BROWSER: playback happens here, so the only thing that can
    stop Test working is this device not having decoded the clips yet.

    The ASSIGNMENT, not a mention -- a test matching the identifier
    anywhere would pass on a comment about it."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    import re as _re

    fn_start = html.index("function renderAudioPanel()")
    fn = html[fn_start:html.index("\n}", fn_start)]

    assert _re.search(r"test\.disabled\s*=\s*!\(audioBuffers", fn), (
        "Test must be disabled until this browser holds the decoded clips"
    )
    assert "a.available" not in fn, "there is no server-side playback flag any more"
    assert "btn-audio-prerender" not in html, "the Render clips button was removed"

def test_config_tab_does_not_claim_cameras_refresh_every_three_seconds(package_root):
    """Stale since 7525bb4 made the previews MJPEG: there is no 3-second
    snapshot poll to describe any more, and how the picture arrives was
    never something the operator could act on."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    # Comments stripped first: the engineering note that records WHY the
    # line went away has to be free to quote it.
    import re as _re
    visible = _re.sub(r"<!--.*?-->", "", html, flags=_re.S)
    assert "refresh every 3" not in visible.lower()
    assert "Cameras refresh" not in visible


def test_camera_device_options_carry_no_uncertainty_marker(package_root):
    """"We did the best we can, no point calling it out" -- a
    non-authoritative name used to render as "dev 2 — USB Camera ?",
    asking the operator to resolve a doubt they cannot resolve from this
    page. The device number beside it was exact the whole time."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    fn_start = html.index("function cameraDeviceOptionsHtml(")
    fn = html[fn_start:html.index("\n}", fn_start)]
    assert "' ?'" not in fn and '" ?"' not in fn, (
        f"the uncertainty marker is back in the picker: {fn!r}"
    )
    # The flag itself is unchanged and still served -- this is about the
    # label, not about pretending the enumeration is provable.
    assert '"names_authoritative"' in html or "names_authoritative" in html


def test_camera_assignment_selector_lives_on_the_preview_card(package_root):
    """The selector sits on each camera card, not in its own section:
    choosing a device and seeing what it shows were two places on one
    page, and the only question it answers -- 'is this slot pointed at the
    right camera' -- is answerable only while looking at the picture."""
    client, _hub = _client_with_hub(package_root)
    html = client.get("/?ui=classic").text
    assert 'id="cam-device-0"' in html
    assert "cam-device-select" in html
    # Standalone section and Save button are gone: the preview IS the
    # confirmation, so a separate save step only adds doubt.
    assert 'id="camera-devices-grid"' not in html
    assert 'id="camera-devices-save"' not in html


def test_camera_selector_is_styled_and_sits_below_the_preview(package_root):
    """It shipped with NO rule of its own -- a native widget wearing the
    OS's colours and typeface inside a dark panel, and one that grew with
    its longest option, which turned the header into a bold label and a
    badge squeezed apart by a control between them the moment real device
    names replaced 'dev 3'.

    Pinned because 'no CSS at all' is invisible to every other test here:
    the element is present, the ids are right, the handlers fire, and it
    still looks broken.
    """
    client, _hub = _client_with_hub(package_root)
    html = client.get("/?ui=classic").text

    assert ".cam-device-select {" in html, "the selector has no style rule"
    # appearance:none is load-bearing -- the OS arrow is the one part of a
    # native select that cannot be themed, so dropping it would take the
    # custom chevron's place and leave two arrows.
    assert "appearance: none" in html
    assert ".cam-device::after" in html, "no replacement chevron"
    # Options are a separate rendering surface from the closed control:
    # without their own rule the open list falls back to OS colours.
    assert ".cam-device-select option {" in html

    # Below the picture, not wedged into the header beside the badge.
    for cam in (0, 1, 2):
        img = html.index(f'id="cam-img-{cam}"')
        sel = html.index(f'id="cam-device-{cam}"')
        badge = html.index(f'id="calib-badge-{cam}"')
        assert badge < img < sel, (
            f"cam{cam}: selector must follow the preview, not sit in the header")


def test_camera_devices_post_applies_live_when_cameras_are_closed(package_root, no_config_writes):
    """The whole point of reconfigure(): a reassignment takes effect on
    the running process, no restart. Save that only persisted read as
    "the button does nothing"."""
    client, hub = _client_with_hub(package_root, devices=(0, 1, 2))
    body = client.patch("/api/config", json={"camera_devices": [3, 1, 2]}).json()
    assert body["ok"] is True
    assert body["applied_live"] == ["camera_devices"]
    assert body["restart_required"] == []
    assert [c.device for c in hub.configs] == [3, 1, 2], "the LIVE hub must have changed"
    assert client.get("/api/config").json()["runtime"]["cameras"]["devices"] == [3, 1, 2]


def test_camera_devices_post_reports_honestly_when_it_cannot_apply(package_root, no_config_writes):
    """An open hub refuses. Say so and flag the restart rather than
    claiming a change that did not happen -- still persisted, so the
    value is not lost."""
    client, hub = _client_with_hub(package_root, devices=(0, 1, 2), refuse=True)
    body = client.patch("/api/config", json={"camera_devices": [3, 1, 2]}).json()
    assert body["persisted"] == ["camera_devices"]
    assert body["applied_live"] == []
    assert body["restart_required"] == ["camera_devices"]
    assert "cameras are open" in body["notes"]["camera_devices"]
    assert [c.device for c in hub.configs] == [0, 1, 2], "must not partially apply"


# --------------------------------------------------------------------------
# Swapping a slot between a local camera and a stream, LIVE. The user's
# requirement was explicit: "i might want to use local cameras then swap to
# a feed" -- with no restart. These drive the real hub through the real
# endpoint, because the interesting failures are in the wiring between
# them, not in either piece alone.
# --------------------------------------------------------------------------


def _client_with_switchable(package_root, devices=(0, 1, 2)):
    """The REAL hub behind the real endpoint.

    A fake used to stand in here, because the hub was a wrapper and the
    fake was its local child. There is one hub now, and these tests never
    call open_all(), so the real one opens no device and touches no
    network -- which makes using a fake pure downside: the wiring between
    the endpoint and the hub is exactly what these tests are for.
    """
    from opendarts.live import remote_capture

    hub = remote_capture.build_hub(list(devices), None)
    app = create_app(package_root=package_root, enable_background_poll=False,
                     local_hub=hub)
    return TestClient(app), hub


def test_one_paste_can_fill_every_camera_slot(package_root):
    """Typing the same host into all three boxes is the papercut this
    feature exists to avoid -- the CLI has expanded a bare `--camera-url`
    into every slot since it shipped, and the dashboard was the half that
    still made you do it by hand. One machine publishes all three cameras
    in every realistic setup, so the box is checked by default.
    """
    client, _hub = _client_with_hub(package_root)
    html = client.get("/?ui=classic").text

    assert 'id="cam-url-all"' in html, "no all-cameras control in the modal"
    assert "checked" in html[html.index('id="cam-url-all"'):html.index('id="cam-url-all"') + 120]
    # Meaningless once the address names ONE stream, so it hides rather
    # than sitting there lying about what Save will do.
    assert "function isBareServerAddress" in html
    assert 'id="cam-url-all-row"' in html

    # ONE apply for all three slots. Three separate posts would each stop
    # and restart capture, and the first two would run a half-applied mix.
    save = html[html.index("document.getElementById('cam-url-save').onclick"):]
    save = save[:save.index("// The checkbox is meaningless")]
    # Comments stripped first -- the explanatory comment above the branch
    # names the call, and counting it makes this pass for the wrong reason.
    code = "\n".join(l for l in save.splitlines() if not l.strip().startswith("//"))
    assert code.count("applyCameraDevices()") == 1, "the swap must apply once, not per slot"
    # Slot count off the DOM, not a constant that can drift from the cards.
    assert "querySelectorAll('.cam-device-select')" in save


def test_a_bare_server_address_expands_to_one_stream_per_slot():
    """The server-side half of the same expansion, which the CLI and the
    dashboard both rely on: slot N must get camera N, not all three
    pointed at camera 0."""
    from opendarts.live.remote_capture import urls_for

    urls = list(urls_for("http://192.0.2.10:8420", 3))
    assert len(urls) == 3
    assert len(set(urls)) == 3, "every slot got the same camera"
    for i, u in enumerate(urls):
        assert f"/api/cameras/{i}/" in u
        assert "full=1" in u, "the preview stream is rate-limited -- transport mode is the point"


def test_pointing_a_slot_at_a_stream_applies_live(package_root, no_config_writes):
    client, hub = _client_with_switchable(package_root)
    before = id(hub)
    url = "http://rig:8420/api/cameras/0/stream.mjpg?full=1"

    body = client.patch("/api/config",
                        json={"camera_devices": [0, 1, 2],
                              "camera_urls": [url, None, None]}).json()
    assert body["ok"] is True
    assert "camera_urls" in body["applied_live"], (
        f"must not need a restart: {body.get('notes')}")

    assert id(hub) == before, "the hub object must survive the swap"
    assert hub.slot_urls == [url, None, None]
    runtime = client.get("/api/config").json()["runtime"]["cameras"]
    assert runtime["urls"] == [url, None, None]


def test_swapping_back_to_local_cameras_applies_live_too(package_root, no_config_writes):
    """The return trip is the half that is easy to leave broken."""
    client, hub = _client_with_switchable(package_root)
    url = "http://rig:8420/api/cameras/0/stream.mjpg?full=1"
    client.patch("/api/config", json={"camera_devices": [0, 1, 2],
                                      "camera_urls": [url, None, None]})

    body = client.patch("/api/config",
                        json={"camera_devices": [0, 1, 2],
                              "camera_urls": [None, None, None]}).json()
    assert "camera_urls" in body["applied_live"]
    assert hub.slot_urls == [None, None, None]


def test_changing_a_device_does_not_wipe_a_stream_on_another_slot(package_root, no_config_writes):
    """The dashboard sends only `devices` when a device dropdown moves.
    Treating a missing `urls` as "make everything local" would silently
    drop a working feed because an unrelated slot changed."""
    client, hub = _client_with_switchable(package_root)
    url = "http://rig:8420/api/cameras/1/stream.mjpg?full=1"
    client.patch("/api/config", json={"camera_devices": [0, 1, 2],
                                      "camera_urls": [None, url, None]})
    assert hub.slot_urls[1] == url

    client.patch("/api/config", json={"camera_devices": [4, 1, 2]})  # devices only
    assert hub.slot_urls[1] == url, "an unrelated device change dropped the stream"


def test_a_hub_that_cannot_read_streams_says_so_rather_than_ignoring_the_url(
    package_root, no_config_writes
):
    """_StubHub predates stream support. Quietly dropping the URL would
    leave the dashboard showing a feed the process is not reading."""
    client, hub = _client_with_hub(package_root, devices=(0, 1, 2))
    body = client.patch("/api/config",
                        json={"camera_devices": [0, 1, 2],
                              "camera_urls": ["http://rig:8420/x", None, None]}).json()
    assert body["applied_live"] == []
    assert "cannot read camera streams" in body["notes"]["camera_urls"]


# --------------------------------------------------------------------------
# The Autodarts comparison -- URL + on/off, 2026-09-10. `GET`/`POST
# /api/ad-config` was retired on 2026-09-17; `ad_base_url` and
# `ad_enabled` are keys of the config document, and both still apply LIVE
# (the listener reconnects or disconnects on the spot).
# --------------------------------------------------------------------------


class _StubListener:
    """Minimal AdWsListener stand-in for the endpoint's surface."""

    def __init__(self, base_url="http://localhost:3180", enabled=True):
        self.base_url = base_url
        self._enabled = enabled
        self.connected = False

    def is_enabled(self):
        return self._enabled

    def is_connected(self):
        return self.connected

    def oracle_base_url(self):
        return self.base_url if self._enabled else None

    def set_enabled(self, v):
        self._enabled = bool(v)

    def set_base_url(self, url):
        if not url.strip():
            raise ValueError("AD base URL cannot be empty")
        self.base_url = url.strip().rstrip("/")


@pytest.fixture
def no_config_writes(monkeypatch):
    """Capture what the config document would persist, and write nothing.

    tests/conftest.py already redirects every default config path into
    this test's tmp directory, so this is about SEEING the writes rather
    than about protection. The config document persists through
    opendarts.live.config_document.persist(), which proves each write by
    reading it back -- so the fake reader has to answer from the same
    dict, or every write would report itself as declined.
    """
    written = {}

    def _fake(section, value, path=None):
        written[section] = value

    monkeypatch.setattr("opendarts.live.config_document.write_config_section", _fake)
    monkeypatch.setattr("opendarts.live.config_document.stored_value", written.get)
    return written


def _ad_client(package_root, listener=None, monkeypatch=None):
    listener = listener or _StubListener()
    app = create_app(package_root=package_root, enable_background_poll=False,
                     ad_ws_listener=listener)
    return TestClient(app), listener


def test_ad_config_get_reports_the_live_listener(package_root):
    client, _ = _ad_client(package_root, _StubListener("http://box:3180", enabled=False))
    body = client.get("/api/config").json()
    assert body["ok"] is True
    assert body["config"]["ad_enabled"] is False
    assert body["config"]["ad_base_url"] == "http://box:3180"
    # "enabled" and "reachable" are different facts, and an operator
    # staring at a blank comparison needs to tell them apart.
    ad = body["runtime"]["ad"]
    assert (ad["available"], ad["connected"]) == (True, False)
    # Ordering stamp -- see tests/test_ad_connection_push.py.
    assert isinstance(ad["connection_seq"], int) and ad["connection_epoch"]


def test_ad_config_can_be_switched_off(package_root, no_config_writes):
    client, listener = _ad_client(package_root)
    body = client.patch("/api/config", json={"ad_enabled": False}).json()
    assert body["ok"] is True
    assert body["config"]["ad_enabled"] is False
    assert body["applied_live"] == ["ad_enabled"]
    assert listener.oracle_base_url() is None, (
        "the throw path reads oracle_base_url(); if it still returns a URL the "
        "toggle has not actually stopped AD being contacted"
    )


def test_ad_toggle_drives_virtual_camera_registration(package_root,
                                                      no_config_writes,
                                                      monkeypatch):
    """The devices follow the toggle, not just the frames into them.

    Publishing to devices Windows has not registered writes to a buffer
    with no reader; leaving them registered puts three synthetic cameras
    in every capture application's list on the machine with nothing
    behind them. The wiring is what rots silently -- vcam_register has
    its own tests, but nothing else proves the endpoint calls it.
    """
    from opendarts.live import vcam_register

    applied: list[bool] = []
    monkeypatch.setattr(vcam_register, "apply",
                        lambda enabled: applied.append(enabled) or
                        {"ok": True, "applicable": True,
                         "action": "register" if enabled else "unregister"})

    client, _listener = _ad_client(package_root)

    off = client.patch("/api/config", json={"ad_enabled": False}).json()
    assert applied == [False]
    assert off["virtual_cameras"]["action"] == "unregister"

    on = client.patch("/api/config", json={"ad_enabled": True}).json()
    assert applied == [False, True]
    assert on["virtual_cameras"]["action"] == "register"


def test_ad_url_change_alone_does_not_touch_registration(package_root,
                                                         no_config_writes,
                                                         monkeypatch):
    """Registration is a registry write and a subprocess. Doing it on
    every URL edit would make retyping a host re-register the devices,
    which could silently invalidate devices a consumer is already
    using."""
    from opendarts.live import vcam_register

    def fail(enabled):
        raise AssertionError("registration must not run for a URL change")

    monkeypatch.setattr(vcam_register, "apply", fail)

    client, _listener = _ad_client(package_root)
    body = client.patch("/api/config",
                        json={"ad_base_url": "http://box:3180"}).json()
    assert body["ok"] is True
    assert "virtual_cameras" not in body, (
        "a URL change must not even report a registration it never ran"
    )


def test_ad_config_rejects_a_url_with_no_scheme(package_root, no_config_writes):
    """A schemeless URL silently becomes an unreachable host later, which
    presents as 'AD stopped working' long after the edit that caused it."""
    client, listener = _ad_client(package_root)
    resp = client.patch("/api/config", json={"ad_base_url": "192.0.2.26:3180"})
    assert resp.status_code == 400
    assert "http://" in resp.json()["errors"]["ad_base_url"]
    assert listener.base_url == "http://localhost:3180", "must not partially apply"


def test_ad_config_rejects_empty_url(package_root, no_config_writes):
    client, _ = _ad_client(package_root)
    resp = client.patch("/api/config", json={"ad_base_url": "   "})
    assert resp.status_code == 400
    assert resp.json()["errors"]["ad_base_url"] == "ad_base_url must be a non-empty string"


def test_ad_config_applies_url_live(package_root, no_config_writes):
    client, listener = _ad_client(package_root)
    body = client.patch("/api/config",
                        json={"ad_base_url": "http://192.0.2.26:3180/"}).json()
    assert body["ok"] is True
    assert body["applied_live"] == ["ad_base_url"]
    # The trailing slash is normalised away before anything is applied or
    # written, so the live listener and the file agree.
    assert listener.base_url == "http://192.0.2.26:3180"
    assert no_config_writes["ad_base_url"] == "http://192.0.2.26:3180"


def test_ad_config_honest_when_no_listener_is_wired(package_root, no_config_writes):
    """No listener: the key is still the rig's own config, so it persists
    -- but nothing in this process applied it, and the reply says which."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    client = TestClient(app)
    runtime = client.get("/api/config").json()["runtime"]["ad"]
    assert runtime["available"] is False
    assert "not wired in this process" in runtime["reason"]

    body = client.patch("/api/config", json={"ad_enabled": False}).json()
    assert body["ok"] is True
    assert body["applied_live"] == []
    assert "not wired in this process" in body["notes"]["ad_enabled"]
    assert no_config_writes["ad_enabled"] is False


def test_root_html_exposes_the_ad_oracle_controls(package_root):
    client, _ = _ad_client(package_root)
    html = client.get("/?ui=classic").text
    assert 'id="ad-enabled-toggle"' in html
    assert 'id="ad-base-url"' in html


def test_packages_table_omits_the_ad_row_when_ad_was_never_asked(package_root):
    """An AD row only when AD actually ANSWERED.

    With AD switched off no ad_ground_truth.json is written at all, so
    ad_matched is null. A blank AD row reads as "AD failed on this throw";
    the honest picture is that AD was not part of it.

    Tightened 2026-09-21 from "true or false" to "true": ad_matched=false
    was also rendering a row, as "Autodarts / no data yet" with a dash for
    a sector -- every cell saying nothing. That is not just noise, because
    marking a throw wrong when AD had no answer WRITES an
    ad_ground_truth.json (match_reason operator_marked_no_ad_data) to
    record the human judgement: correcting a throw conjured an empty AD
    row beneath it. Correcting a throw must not invent a reference row for
    a reference that never spoke.
    """
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/?ui=classic").text
    assert "const adAsked = (p.ad_matched === true);" in html
    assert "let html = !adAsked ? '' : (" in html
    # the old "no data yet" BADGE is gone from the markup, not merely
    # hidden (the phrase still appears in the comment explaining why)
    assert '<span class="badge unknown">no data yet</span>' not in html


def test_packages_table_promotes_the_primary_row_when_there_is_no_ad_row(package_root):
    """The AD row owns the throw number and capture time for the group --
    engine rows deliberately leave both blank. With no AD row the first
    engine row (the primary, Zeus) has to carry them, or the whole group
    renders numberless and undated and reads as a continuation of the
    throw above it."""
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/?ui=classic").text
    assert "sections.map((s, si)" in html
    assert "(!adAsked && si === 0 ? (n - i) : '')" in html
    assert "(!adAsked && si === 0 ? fmtCaptured(p.captured_at_utc) : '')" in html


# --------------------------------------------------------------------------
# /api/frame-health -- is the pipeline keeping up, at both ends?
# --------------------------------------------------------------------------


def test_frame_health_reports_effective_fps_per_camera(package_root):
    """A camera configured for 30fps that delivers 21 degrades silently:
    the driver drops those frames before this process sees them, so the
    rate deficit is the only evidence there is."""
    from opendarts.live.local_capture import CameraConfig

    class _Hub:
        def __init__(self):
            from opendarts.live.local_capture import CameraStatus
            self.configs = [CameraConfig(device=0)]
            st = CameraStatus(device=0)
            st.opened = True
            st.requested_fps = 30
            st.effective_fps = 21.0
            st.frame_count = 500
            self.status = {0: st}
            self.frame_sink_errors = 0

    app = create_app(package_root=package_root, enable_background_poll=False,
                     local_hub=_Hub())
    body = TestClient(app).get("/api/frame-health").json()
    cam = body["capture"][0]
    assert cam["requested_fps"] == 30
    assert cam["effective_fps"] == 21.0
    assert cam["keeping_up"] is False, "21 of 30 is a real shortfall"


def test_frame_health_does_not_judge_before_a_rate_is_measured(package_root):
    """keeping_up must be None, not False, until a window has completed --
    otherwise every start reports the cameras as failing."""
    from opendarts.live.local_capture import CameraConfig, CameraStatus

    class _Hub:
        def __init__(self):
            self.configs = [CameraConfig(device=0)]
            st = CameraStatus(device=0)
            st.opened = True
            st.requested_fps = 30
            st.effective_fps = None          # no window has completed yet
            self.status = {0: st}
            self.frame_sink_errors = 0

    app = create_app(package_root=package_root, enable_background_poll=False,
                     local_hub=_Hub())
    assert TestClient(app).get("/api/frame-health").json()["capture"][0]["keeping_up"] is None


def test_frame_health_reports_publishing_as_off_when_not_enabled(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).get("/api/frame-health").json()
    assert body["ok"] is True
    assert body["publish"]["enabled"] is False


def test_frame_health_surfaces_the_consumer_drop_counters(package_root):
    """Counting what we published proves only that we published. The
    consumer's own missed count is the direct evidence."""
    class _Set:
        def stats(self):
            return [{"slot": 0, "published": 900, "read": 880,
                     "missed": 20, "torn": 1, "consumer_attached": True,
                     "drop_rate": 20 / 900}]

    app = create_app(package_root=package_root, enable_background_poll=False,
                     vcam_set=_Set())
    body = TestClient(app).get("/api/frame-health").json()
    assert body["publish"]["enabled"] is True
    slot = body["publish"]["slots"][0]
    assert slot["missed"] == 20
    assert slot["consumer_attached"] is True


def test_disabling_ad_also_stops_virtual_camera_publishing(package_root, no_config_writes):
    """Publishing follows the Autodarts toggle. Left running with the
    comparison off it copies megabytes per frame to a consumer nobody
    asked for -- and a toggle that did not stop
    it would not really be off."""
    class _Hub:
        def __init__(self):
            self.configs = []
            self.status = {}
            self.sink = "something"
            self.frame_sink_errors = 0

        def set_frame_sink(self, fn):
            self.sink = fn

    class _Set:
        def publish_all(self, frames):
            return 0

        def stats(self):
            return []

    hub, vset, listener = _Hub(), _Set(), _StubListener()
    app = create_app(package_root=package_root, enable_background_poll=False,
                     local_hub=hub, vcam_set=vset, ad_ws_listener=listener)
    client = TestClient(app)

    client.patch("/api/config", json={"ad_enabled": False})
    assert hub.sink is None, "publishing must stop with the toggle"

    client.patch("/api/config", json={"ad_enabled": True})
    assert hub.sink == vset.publish_all, "and resume with it"


def test_ad_toggle_is_harmless_where_publishing_was_never_set_up(package_root, no_config_writes):
    """macOS and Linux have no virtual cameras, so there is nothing to
    attach -- the toggle must still work rather than erroring."""
    client, _listener = _ad_client(package_root)
    assert client.patch("/api/config", json={"ad_enabled": False}).json()["ok"] is True


def test_camera_reassignment_applies_while_capture_is_running_and_board_is_clear(
    package_root, no_config_writes
):
    """A running loop with an empty board must not block a reassignment."""
    from opendarts.live.capture_daemon import CaptureLoopController

    class _Ctl(CaptureLoopController):
        def is_running(self):
            return True

    hub = _StubHub((0, 1, 2))
    app = create_app(package_root=package_root, enable_background_poll=False,
                     local_hub=hub, controller=_Ctl())
    body = TestClient(app).patch("/api/config", json={"camera_devices": [3, 1, 2]}).json()
    assert body["ok"] is True
    assert body["applied_live"] == ["camera_devices"], body.get("notes")
    assert [c.device for c in hub.configs] == [3, 1, 2]


def test_camera_reassignment_guard_reads_the_lifecycle_count_not_the_retail_visit():
    """visit_throws is the RETAIL visit list and survives a visit that
    never got a clean takeout, so it reports darts on an empty board and
    refuses a reassignment for a game that finished long ago. The guard
    has to read the lifecycle's own count.

    Asserted against the source because the AppState the applier closes
    over is not reachable from a constructed app -- weaker than driving
    it, but it pins the signal, which is the thing that was wrong.
    """
    import inspect

    import opendarts.live.server as server_mod

    src = inspect.getsource(server_mod)
    body = src[src.index("async def _apply_camera_assignment"):]
    guard = body[:body.index("async def _apply_config_changes")]
    assert "state.trigger_dart_count" in guard
    # The attribute ACCESS, not the word: the comment explaining why the
    # retail list is the wrong signal mentions it by name.
    assert "state.visit_throws" not in guard, (
        "the guard is back on the retail visit list, which reports darts "
        "on a board that is already clear"
    )


def test_engine_tally_skips_throws_with_no_reference_to_grade_against(package_root):
    """A null sector_match means no comparison was possible -- no matched
    AD ground truth and no operator-confirmed answer. Those throws must be
    skipped, not counted as misses.

    Reported live 2026-09-12: with the comparison switched off, every throw is
    ungradeable, and counting each one as a miss dragged the primary
    engine to 115/129 against a reference that was never consulted.

    Same convention as the other tally mirrors here: assert the literal
    shipped condition, then mirror it in Python."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert ("if (s.sector_match === null || s.sector_match === undefined) continue;"
            in html)

    def _counted(section: dict) -> bool:
        return section.get("sector_match") is not None

    # No reference at all -> neither right nor wrong.
    assert _counted({"sector_match": None}) is False
    assert _counted({}) is False
    # Compared and wrong -> a real miss, still counted.
    assert _counted({"sector_match": False}) is True
    # Compared and right.
    assert _counted({"sector_match": True}) is True


def test_engine_tally_still_counts_an_engine_that_ran_and_failed(package_root):
    """The skip must not excuse a failing engine. One that ran, produced
    nothing, and was compared against a real reference comes back false --
    it had its shot and missed, so it stays in the denominator."""
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/?ui=classic").text
    # The guard keys on null/undefined only, never on ok/timed_out.
    assert "s.sector_match === null || s.sector_match === undefined" in html
    assert "bump(s.name, s.sector_match === true);" in html


# -- the calibration overlay as a LAYER over the live stream --------------
#
# The overlay used to be a baked-in still that REPLACED a tile's MJPEG
# stream the moment that camera's calibration came back ok -- so the one
# state where you most want to see live video (a working, calibrated rig)
# was the one state that showed a 3-second slideshow. These pin the
# replacement: a transparent PNG layered over a stream that keeps running,
# fetched when the calibration changes rather than on a timer.


def test_overlay_rgba_is_transparent_and_sized_to_the_preview(package_root, monkeypatch):
    """It has to be an actual layer -- opaque corners would hide the very
    video it sits on -- and it has to match the stream's own dimensions,
    or its rings sit off the board and read as bad calibration."""
    import numpy as _np
    from opendarts.live.capture_daemon import CalibrationStore

    hub = _open_wide_fake_local_hub(monkeypatch)
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        calibration_store=CalibrationStore({0: _ring_calibration(0)}, source="startup"),
    )
    resp = TestClient(app).get("/api/cameras/0/overlay-rgba.png")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"

    png = cv2.imdecode(_np.frombuffer(resp.content, dtype=_np.uint8), cv2.IMREAD_UNCHANGED)
    assert png is not None and png.shape[2] == 4, "an overlay with no alpha is not a layer"
    alpha = png[:, :, 3]
    assert alpha[0, 0] == 0 and alpha[-1, -1] == 0, "corners must be see-through"
    assert (alpha > 0).any(), "nothing was drawn at all"

    status = hub.status[0]
    assert (png.shape[1], png.shape[0]) == (status.actual_width, status.actual_height), (
        "the overlay must be rendered in the calibration's own pixel space"
    )


def test_overlay_rgba_grabs_no_frame(package_root, monkeypatch):
    """The point of the endpoint: it needs the calibration and a size, so
    it must never touch the hub's pump and compete with the stream."""
    from opendarts.live import local_capture as _lc
    from opendarts.live.capture_daemon import CalibrationStore

    hub = _open_fake_local_hub(monkeypatch)
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        calibration_store=CalibrationStore({0: _ring_calibration(0)}, source="startup"),
    )

    def _explode(*a, **k):
        raise AssertionError("overlay-rgba.png must not fetch a frame")

    monkeypatch.setattr(_lc, "fetch_snapshot", _explode)
    assert TestClient(app).get("/api/cameras/0/overlay-rgba.png").status_code == 200


def test_overlay_rgba_404s_without_a_calibration(package_root, monkeypatch):
    """Nothing honest to draw -- and the dashboard shows the stream alone
    rather than an empty layer."""
    hub = _open_fake_local_hub(monkeypatch)
    app = create_app(package_root=package_root, enable_background_poll=False, local_hub=hub)
    resp = TestClient(app).get("/api/cameras/0/overlay-rgba.png")
    assert resp.status_code == 404
    assert "calibration" in resp.json()["reason"]


def test_a_calibrated_camera_is_not_taken_off_its_stream(package_root):
    """The regression this whole change exists for. updateCameraFeeds()
    used to skip the stream entirely for any camera whose overlay was on,
    handing the tile to a 3-second still."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    fn_start = html.index("function updateCameraFeeds()")
    fn = html[fn_start:html.index("\n}", fn_start)]
    assert "camOverlayOn" not in fn, (
        "updateCameraFeeds() must not branch on the overlay -- that is what "
        "dropped a calibrated camera off its live stream"
    )
    assert "stream.mjpg" in fn


def test_the_overlay_is_layered_over_the_stream_not_swapped_into_it(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text
    assert 'id="cam-overlay-0"' in html and 'class="cam-overlay"' in html
    assert 'id="cam-img-0"' in html
    # Positioned over the picture, and never intercepting a click.
    css = html[html.index(".cam-overlay {"):html.index("}", html.index(".cam-overlay {"))]
    assert "position: absolute" in css and "pointer-events: none" in css
    # The layer must point at the TRANSPARENT endpoint, never the baked one.
    js = html.split("<script>")[-1].split("</script>")[0]
    fn_start = js.index("function refreshCalibrationOverlays()")
    fn = js[fn_start:js.index("\n}", fn_start)]
    assert "overlay-rgba.png" in fn and "overlay.png?t=" not in fn


def test_the_overlay_is_not_on_a_timer(package_root):
    """It changes only when calibration is re-derived. A 3-second poll for
    a picture of a stationary board was work with nothing to find."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    js = TestClient(app).get("/?ui=classic").text.split("<script>")[-1].split("</script>")[0]

    import re as _re

    for m in _re.finditer(r"setInterval\((.*?),\s*[A-Z_]+\)", js):
        assert "refreshCalibrationOverlays" not in m.group(1), (
            f"the overlay is back on a timer: {m.group(0)}"
        )
    # ...and the token it keys off is the calibration package id, which is
    # what makes "when the calibration changes" different from "every poll".
    assert "calibration_package_id" in js


def test_the_overlay_hides_when_there_is_no_live_picture_under_it(package_root):
    """A layer floating over a black "No signal" placeholder claims a
    calibration is being confirmed against a camera that is not running."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    js = TestClient(app).get("/?ui=classic").text.split("<script>")[-1].split("</script>")[0]
    fn_start = js.index("function refreshCalibrationOverlays()")
    fn = js[fn_start:js.index("\n}", fn_start)]
    assert "camStreamState[c] !== 'live'" in fn
    assert "hideCalibrationOverlay" in fn


def test_overlay_rgba_is_rendered_in_the_calibrations_own_pixel_space(
    package_root, monkeypatch
):
    """The alignment regression, end to end.

    A first cut served this layer at the STREAM's preview size (960 wide)
    while the calibration's camera_matrix projects into the camera's
    native 1280-wide space. Nothing rescales in between, so the board came
    out 1.33x too large and shoved toward the bottom-right -- shipped to
    both rigs and caught by eye, because the unit equivalence test passes
    the same size to both renderers and so has a scale factor of 1 always.

    This composites the SERVED layer over the SERVED snapshot and compares
    it against the SERVED baked overlay, which has always been correct.
    Any disagreement about pixel space shows up here as a gross mismatch.
    """
    import numpy as _np
    from opendarts.live.capture_daemon import CalibrationStore

    hub = _open_wide_fake_local_hub(monkeypatch)
    app = create_app(
        package_root=package_root,
        enable_background_poll=False,
        local_hub=hub,
        calibration_store=CalibrationStore({0: _ring_calibration(0)}, source="startup"),
    )
    client = TestClient(app)

    def _decode(resp, flags=cv2.IMREAD_COLOR):
        assert resp.status_code == 200, resp.status_code
        return cv2.imdecode(_np.frombuffer(resp.content, dtype=_np.uint8), flags)

    baked = _decode(client.get("/api/cameras/0/overlay.png"))
    frame = _decode(client.get("/api/cameras/0/snapshot.png"))
    layer = _decode(client.get("/api/cameras/0/overlay-rgba.png"), cv2.IMREAD_UNCHANGED)

    # Same pixel space as the picture it sits on -- the fake hub serves a
    # constant frame, so the two are directly comparable.
    assert layer.shape[:2] == frame.shape[:2] == baked.shape[:2]

    a = (layer[:, :, 3].astype(_np.float32) / 255.0)[:, :, None]
    composited = _np.clip(
        layer[:, :, :3].astype(_np.float32) * a + frame.astype(_np.float32) * (1.0 - a),
        0, 255,
    ).astype(_np.uint8)

    diff = _np.abs(composited.astype(_np.int16) - baked.astype(_np.int16))
    assert diff.max() <= 6, (
        f"layer does not composite to the baked overlay (max {diff.max()}) -- "
        "the two disagree about what pixel space they are drawing in"
    )


# -- the full adopted calibration, over HTTP ----------------------------
#
# The solved pose has always existed in CalibrationStore and in every
# calibration package, and had no route. /api/state's calibration.cameras
# is a display shape (ok / reprojection_error_px / landmark_spread_ok)
# built for the dashboard's badges, and calibration packages have no
# route either -- so on a rig with no shell the numbers were unreachable,
# and "how far apart are the cameras" had to be answered by re-solving
# poses from preview snapshots. That is a reconstruction of this data,
# not this data.


def _calibration_client(package_root, monkeypatch):
    from opendarts.live.capture_daemon import CalibrationStore

    hub = _open_fake_local_hub(monkeypatch)
    store = CalibrationStore(
        {0: _ring_calibration(0), 1: _ring_calibration(1), 2: _ring_calibration(2)},
        source="startup",
    )
    app = create_app(
        package_root=package_root, enable_background_poll=False,
        local_hub=hub, calibration_store=store,
    )
    return TestClient(app)


def test_calibration_endpoint_serves_the_solved_pose(package_root, monkeypatch):
    body = _calibration_client(package_root, monkeypatch).get("/api/calibration").json()

    assert body["ok"] is True
    assert set(body["cameras"]) == {"0", "1", "2"}
    for cam in ("0", "1", "2"):
        c = body["cameras"][cam]
        # The pose itself -- the whole reason this route exists.
        assert len(c["rvec"]) == 3 and len(c["tvec"]) == 3
        assert len(c["camera_matrix"]) == 3 and len(c["camera_matrix"][0]) == 3
        assert c["focal_length_px"] > 0
        assert len(c["principal_point_px"]) == 2
        assert isinstance(c["dist_coeffs"], list)
    # Provenance travels with it, so a reader can tell a fresh solve from
    # one restored off disk.
    assert body["source"] == "startup"


def test_calibration_endpoint_derives_the_camera_position(package_root, monkeypatch):
    """position_mm is -R^T t in board coordinates. Derived server-side
    because every caller wanting "where is this camera" would otherwise
    repeat it, and getting the transpose wrong is silent."""
    import numpy as _np

    body = _calibration_client(package_root, monkeypatch).get("/api/calibration").json()

    for cam in ("0", "1", "2"):
        c = body["cameras"][cam]
        rvec = _np.array(c["rvec"], dtype=float).reshape(3, 1)
        tvec = _np.array(c["tvec"], dtype=float).reshape(3, 1)
        rot, _ = cv2.Rodrigues(rvec)
        expected = (-rot.T @ tvec).ravel()
        assert _np.allclose(c["position_mm"], expected, atol=1e-6), (
            "position_mm must be the camera centre in board coordinates"
        )
        # ...and the spherical form must agree with the cartesian one.
        assert c["distance_mm"] == pytest.approx(float(_np.linalg.norm(expected)))
        assert 0.0 <= c["azimuth_deg"] < 360.0


def test_calibration_endpoint_azimuths_of_a_ring_rig_are_120_apart(package_root, monkeypatch):
    """The question this route was added to answer. The synthetic fixture
    is a real 3-camera ring, so the derived azimuths must come out 120
    degrees apart -- proving the derivation is oriented correctly and not
    merely self-consistent."""
    body = _calibration_client(package_root, monkeypatch).get("/api/calibration").json()

    az = sorted(body["cameras"][c]["azimuth_deg"] for c in ("0", "1", "2"))
    gaps = [(az[(i + 1) % 3] - az[i]) % 360.0 for i in range(3)]
    for gap in gaps:
        assert gap == pytest.approx(120.0, abs=1.0), f"gaps {gaps} are not a 120-degree ring"


def test_calibration_endpoint_is_honest_with_no_capture_loop(package_root):
    """Standalone dashboard -- no store to read. Must say so rather than
    return an empty success that reads as "no cameras calibrated"."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    body = TestClient(app).get("/api/calibration").json()
    assert body["ok"] is False
    assert body["cameras"] == {}
    assert "no capture loop" in body["reason"]


# ---------------------------------------------------------------------------
# THREAD/CPU DIAGNOSTICS, 2026-09-13 (CPU task). /api/threads is the other
# half of a join: an external sampler reads per-OS-thread CPU on the
# Windows rig (which knows nothing of Python thread names), and native_id is
# the key that attaches a role to each figure.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# OPENCV THREAD BOUND, 2026-09-13 (CPU task). Measured on the Windows rig:
# an unbounded pool cost 62.8% of this process's total CPU at idle. See
# opendarts/live/cv2_threads.py for the full before/after.
# ---------------------------------------------------------------------------


def test_cv2_thread_default_is_one():
    """The measured default, chosen 2026-09-13 once the numbers were in:
    no gains were found in more threads, on any path."""
    from opendarts.live import cv2_threads

    assert cv2_threads.DEFAULT_CV2_NUM_THREADS == 1


def test_apply_cv2_thread_limit_uses_the_default_when_unconfigured(monkeypatch):
    """None means "no override configured", and unusually for this project
    that still APPLIES a value -- OpenCV's self-chosen pool is the thing
    being corrected, so leaving it alone would defeat the point."""
    import cv2
    from opendarts.live import cv2_threads

    seen = []
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: seen.append(n))
    cv2_threads.apply_cv2_thread_limit(None)
    assert seen == [cv2_threads.DEFAULT_CV2_NUM_THREADS] == [1]

    seen.clear()
    cv2_threads.apply_cv2_thread_limit(4)
    assert seen == [4]


def test_apply_cv2_thread_limit_negative_keeps_opencv_default(monkeypatch):
    """A negative value is the documented opt-out for a rig that wants
    OpenCV's own pool back -- it must NOT call setNumThreads at all."""
    import cv2
    from opendarts.live import cv2_threads

    seen = []
    monkeypatch.setattr(cv2, "setNumThreads", lambda n: seen.append(n))
    cv2_threads.apply_cv2_thread_limit(-1)
    assert seen == [], "a negative count must leave OpenCV's default untouched"


def test_apply_cv2_thread_limit_never_raises(monkeypatch):
    """Runs at process start: a rig that cannot tune its pool must still
    boot and score darts."""
    import cv2
    from opendarts.live import cv2_threads

    def boom(_n):
        raise RuntimeError("no threading for you")

    monkeypatch.setattr(cv2, "setNumThreads", boom)
    assert cv2_threads.apply_cv2_thread_limit(1) is None


def test_live_config_reads_and_validates_cv2_num_threads(tmp_path):
    """`True` is an int in Python and must not be accepted as a thread
    count, same as every other typed key in this file."""
    import json as _json
    from opendarts.live.config import load_live_config

    def cfg(value):
        p = tmp_path / f"cfg_{value!r}.json"
        p.write_text(_json.dumps({"cv2_num_threads": value}))
        return load_live_config(p)

    assert cfg(1).cv2_num_threads == 1
    assert cfg(-1).cv2_num_threads == -1
    for bad in (True, "2", 1.5, None):
        assert cfg(bad).cv2_num_threads is None, f"{bad!r} should not be accepted"

    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    assert load_live_config(empty).cv2_num_threads is None


def test_calibrate_action_line_reports_how_long_it_took(package_root):
    """A calibration is a ~25s operation whose cost the operator has no
    other way to see -- the phase breakdown goes to the log and the
    package, neither of which is in front of someone who just clicked
    Calibrate. Both outcomes carry the time: a fast refusal ("cameras not
    ready", ~2s) and a slow failure are different problems, and the
    duration is what distinguishes them."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    js = TestClient(app).get("/?ui=classic").text.split("<script>")[-1].split("</script>")[0]

    start = js.index("btn-refresh-calib').onclick")
    handler = js[start:js.index("\n};", start)]

    # Elapsed measured with performance.now(), not Date.now(): a system
    # clock adjustment landing mid-calibration must not corrupt it.
    assert "performance.now()" in handler
    assert "Date.now()" not in handler.split("calibStartedMs")[0][-400:], (
        "elapsed time must not be measured with the wall clock"
    )
    assert "camera(s) ok in ' + calibSecs + 's'" in handler
    assert "' (after ' + calibSecs + 's)'" in handler, (
        "the failure path needs the duration too"
    )
    # Started before the request, so it covers what the operator waited.
    assert handler.index("calibStartedMs = performance.now()") < handler.index(
        "fetch('/api/calibration/refresh'"
    )


def test_every_route_is_documented_in_live_api():
    """`docs/LIVE_API.md` is the published surface of a shipping product.
    An undocumented live route is the worse loose end: nobody knows it
    exists to decide about it, and it cannot be deprecated because it was
    never announced.

    This caught 16 undocumented routes and two wrong entries -- delete-all
    documented as DELETE when it is POST (a client following the doc gets
    405), and /api/diagnostics described as throw data when it is the
    on/off gate.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    src = (root / "opendarts" / "live" / "server.py").read_text()
    doc = (root / "docs" / "LIVE_API.md").read_text()

    routes = re.findall(r'@app\.(?:get|post|patch|put|delete|websocket)\("([^"]+)"', src)
    undocumented = sorted({r for r in routes if r != "/" and r not in doc})
    assert not undocumented, (
        "these live routes are missing from docs/LIVE_API.md: "
        + ", ".join(undocumented)
    )


def test_live_api_documents_the_right_http_methods():
    """A documented method that does not exist is worse than no docs --
    it sends a client to a 405. `delete-all` was documented as DELETE
    while implemented as POST, which is exactly that.

    Checked per DOC LINE rather than per backtick, because a table row
    legitimately reads ``\u0060GET\u0060 \u0060POST /api/audio\u0060`` -- two spans, one
    route, both verbs real.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    src = (root / "opendarts" / "live" / "server.py").read_text()
    doc_lines = (root / "docs" / "LIVE_API.md").read_text().splitlines()

    implemented: dict[str, set[str]] = {}
    for method, route in re.findall(
        r'@app\.(get|post|patch|put|delete|websocket)\("([^"]+)"', src
    ):
        implemented.setdefault(route, set()).add(method.upper())

    for route, methods in implemented.items():
        if route == "/":
            continue
        lines = [ln for ln in doc_lines if route in ln]
        if not lines:
            continue  # completeness is the other test's job
        verbs = set()
        for ln in lines:
            verbs |= {v.upper() for v in re.findall(r"GET|POST|PATCH|PUT|DELETE|WebSocket", ln)}
        missing = methods - verbs
        assert not missing, (
            f"{route} is implemented as {sorted(missing)} but the doc line only "
            f"mentions {sorted(verbs)}"
        )


# ---------------------------------------------------------------------------
# ABOUT THIS RIG, 2026-09-13. The Config tab's read-only table plus its
# "Copy diagnostics" button. See opendarts/live/build_info.py for what is
# deliberately absent (branch, dirty flag, repo paths, live camera state).
# ---------------------------------------------------------------------------


def test_health_carries_build_identity():
    """`build` is the first field any bug report needs. It rides on
    /api/health for the same reason `capabilities` does."""
    app = create_app(package_root=Path(tempfile.mkdtemp()), enable_background_poll=False)
    build = TestClient(app).get("/api/health").json()["build"]

    for key in ("code_version", "started_at_epoch", "uptime_s", "hostname",
                "platform", "python_version", "opencv_version",
                "opencv_parallel_framework", "opencv_threads", "cpu_count"):
        assert key in build, f"/api/health build is missing {key}"
    assert build["python_version"]
    assert build["started_at_epoch"] > 0


def test_build_info_never_raises_and_degrades_to_none(monkeypatch):
    """Read by a status page. A machine that cannot report its own OpenCV
    build must still serve darts."""
    from opendarts.live import build_info as bi

    monkeypatch.setattr(bi, "_code_version", lambda: None)
    info = bi.build_info()
    assert info["code_version"] is None  # rendered as "unknown", never blank
    assert info["python_version"]


def test_dashboard_diagnostics_uses_only_helpers_that_exist():
    """The JS syntax check cannot see undefined REFERENCES -- a call to a
    helper that does not exist parses fine and then blanks the table at
    runtime. This caught exactly that: an invented `escapeHtml` where the
    file's real helper is `escapeAttr`."""
    src = (Path(__file__).resolve().parent.parent
           / "opendarts" / "live" / "dashboard" / "app.js").read_text()

    for helper in ("escapeAttr", "fmtBool", "fmtUptime", "diagVal",
                   "diagReachAt", "diagAdSummary", "diagAudioSummary",
                   "diagnosticsText", "copyDiagnostics", "refreshBuildInfo",
                   "renderDiagnostics"):
        assert f"function {helper}(" in src, f"{helper}() is called but never defined"

    # The specific invented name, pinned so it cannot come back.
    assert "escapeHtml" not in src, "escapeHtml does not exist in this file; use escapeAttr"


def test_diagnostics_table_and_copy_button_are_rendered():
    app = create_app(package_root=Path(tempfile.mkdtemp()), enable_background_poll=False)
    html = TestClient(app).get("/?ui=classic").text

    assert 'id="diagnostics-tbody"' in html
    assert 'id="btn-copy-diagnostics"' in html
    assert "About this rig" in html
    # The button must be wired, not decorative.
    assert "document.getElementById('btn-copy-diagnostics').onclick" in html


# ---------------------------------------------------------------------------
# CAMERA TILE EMPTY STATE + SUBROW LABELLING, 2026-09-13.
# ---------------------------------------------------------------------------


def test_camera_tile_paints_the_placeholder_before_js_runs(package_root):
    """An <img> with no src is NOT blank -- the browser paints its alt text
    beside a broken-image icon. So the honest "not started" state rendered
    as "this is broken" until updateCameraFeeds() arrived, and indefinitely
    on any path where it never sets a src (a hidden tab, for one).

    The empty state must therefore be the DEFAULT in the markup, not
    something JavaScript installs.
    """
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/?ui=classic").text

    for cam in range(3):
        img_start = html.index(f'id="cam-img-{cam}"')
        img_tag = html[img_start:html.index(">", img_start)]
        assert "display:none" in img_tag.replace(" ", ""), (
            f"cam{cam}'s <img> starts visible with no src, so it paints alt text"
        )
        ph_start = html.index(f'id="cam-placeholder-{cam}"')
        ph_tag = html[ph_start:html.index(">", ph_start)]
        assert "hidden" not in ph_tag, (
            f"cam{cam}'s placeholder starts hidden, so nothing covers the broken image"
        )


def test_camera_subrow_labels_the_preview_so_it_cannot_read_as_calibration(package_root):
    """Two unrelated facts share that row: the live preview stream on the
    left, the calibration solve's quality on the right. Unlabelled, a green
    "calibrated" badge sitting above "not yet fetched" reads as the badge
    lying -- when a camera is routinely calibrated while its preview is not
    running."""
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/?ui=classic").text

    assert '<span class="cam-subrow-label">preview</span>' in html
    for cam in range(3):
        assert f'id="calib-metric-{cam}"' in html, "the metric needs an id to explain itself"
    # The metric must say what it is; "px reproj." alone is not self-evident.
    assert "Reprojection error of this camera's calibration solve" in html


def test_absent_reprojection_says_why_when_the_calibration_is_good(package_root):
    """`ok: true` with a null reprojection error is a real, common state: a
    calibration loaded from disk is valid, but its error was measured in
    the session that solved it. A bare em-dash next to a green "calibrated"
    badge reads as missing data; "not measured" is the honest answer.

    Reverting to the unconditional em-dash fails this.
    """
    html = TestClient(
        create_app(package_root=package_root, enable_background_poll=False)
    ).get("/?ui=classic").text

    assert "'not measured'" in html, "an absent-but-valid reprojection must explain itself"
    assert "calib-metric-absent" in html, "and be styled as absent rather than as a failure"
    # Still a plain dash when the calibration genuinely is not ok.
    assert "row.ok === true ? 'not measured'" in html


# --------------------------------------------------------------------------
# Ctrl-C with streams open. A multipart/x-mixed-replace response never ends
# on its own, so uvicorn's graceful drain waits the full timeout on every
# open stream and then force-cancels the task -- "Cancel 6 running task(s),
# timeout graceful shutdown exceeded" plus an ASGI traceback, on every
# Ctrl-C. Reliable once a second machine consumes the transport streams:
# unlike a browser tab, those consumers never disconnect on their own.
# --------------------------------------------------------------------------


def test_a_stream_is_not_shutting_down_when_nothing_supplied_a_check(package_root):
    """A TestClient app, or any embedding that never owned a uvicorn
    Server, must behave exactly as before -- the check is opt-in."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    assert state.should_exit_check is None
    assert state.is_shutting_down() is False


def test_the_shutdown_check_is_consulted(package_root):
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    exiting = {"v": False}
    state.should_exit_check = lambda: exiting["v"]
    assert state.is_shutting_down() is False
    exiting["v"] = True
    assert state.is_shutting_down() is True


def test_a_broken_shutdown_check_never_breaks_a_live_stream(package_root):
    """This is consulted once per frame on every open stream. A raising
    check must degrade to 'not shutting down' rather than killing the
    streams it exists to end politely."""
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state

    def boom():
        raise RuntimeError("server went away")

    state.should_exit_check = boom
    assert state.is_shutting_down() is False


def test_the_stream_loop_checks_shutdown_before_the_capture_state():
    """ORDER MATTERS. During shutdown the capture controller can still
    report running, so a shutdown check placed after that condition would
    never be reached and the stream would keep going until it was
    cancelled."""
    from pathlib import Path

    from opendarts.live import server as server_mod

    src = Path(server_mod.__file__).read_text()
    body = src[src.index("async def frame_parts():"):]
    body = body[:body.index("return StreamingResponse")]
    i_shutdown = body.index("state.is_shutting_down()")
    i_running = body.index("state.controller.is_running()")
    assert i_shutdown < i_running, (
        "the shutdown check must come first, or capture still reporting "
        "running keeps the stream alive through the drain")


def test_run_product_wires_the_shutdown_check_to_uvicorn(monkeypatch, tmp_path):
    """The hook is useless unwired, and nothing else would notice: streams
    would simply go back to being force-cancelled on every Ctrl-C."""
    from pathlib import Path

    from opendarts.live import run_product

    src = Path(run_product.__file__).read_text()
    built = src[src.index("server = uvicorn.Server(uvicorn_config)"):]
    built = built[:built.index("def _on_capture_event")]
    assert "should_exit_check" in built, "run_product never wires the check"
    # Comments stripped: the explanation beside the wiring names the very
    # thing the last assertion forbids, and counting prose would make this
    # fail for the wrong reason.
    code = "\n".join(
        l for l in built.splitlines() if not l.strip().startswith("#"))
    # server.should_exit, NOT the stop event: uvicorn installs its own
    # signal handlers for the duration of server.run(), so the handler
    # that sets the stop event does not run until the drain is already over.
    assert "server.should_exit" in code
    assert "stop_event" not in code


# --------------------------------------------------------------------------
# Stream admission. A PREVIEW is a picture in a dashboard; a TRANSPORT
# stream (?full=1) is another machine's camera feed, which that machine
# SCORES from. A shared ceiling let the cosmetic one starve the load-bearing
# one -- and because the ceiling counts CONNECTIONS while every viewer costs
# one per camera, the old cap of 8 admitted two viewers of a 3-camera rig
# and refused the third. Observed on the real rig: blank tiles everywhere,
# every new stream 503ing, and a completely clean log.
# --------------------------------------------------------------------------


def test_the_cap_is_expressed_in_viewers_not_connections():
    """The bug was a units one: 8 'clients' sounds like plenty and is two
    and a bit viewers once every viewer opens one stream per camera."""
    from opendarts.live.server import _mjpeg_cap

    # 3 cameras, 4 viewers, plus one camera-set of reconnect slack.
    assert _mjpeg_cap(3, 4) == 15
    # A rig with a different camera count scales rather than silently
    # allowing fewer viewers.
    assert _mjpeg_cap(2, 4) == 10
    assert _mjpeg_cap(1, 4) == 5


def test_a_transport_consumer_is_not_refused_by_open_dashboards():
    """THE POINT OF SPLITTING THE BUDGETS. Browser tabs must never be able
    to stop another machine reading this rig's cameras."""
    from opendarts.live import server as server_mod

    app = create_app(package_root=Path(tempfile.mkdtemp()),
                     enable_background_poll=False)
    state = app.state.opendarts_state
    # Every preview slot taken.
    state.mjpeg_client_count = server_mod._mjpeg_cap(
        3, server_mod.MJPEG_MAX_PREVIEW_VIEWERS)
    assert state.mjpeg_transport_count == 0, (
        "a full preview budget must leave the transport budget untouched")


def test_the_two_budgets_are_counted_separately():
    app = create_app(package_root=Path(tempfile.mkdtemp()),
                     enable_background_poll=False)
    state = app.state.opendarts_state
    assert hasattr(state, "mjpeg_client_count")
    assert hasattr(state, "mjpeg_transport_count")
    assert state.mjpeg_client_count == 0 and state.mjpeg_transport_count == 0


def test_health_reports_the_stream_ceilings(package_root):
    """A rig at its ceiling looks identical to a broken one from outside:
    blank tiles, nothing in the log. Before this the only way to tell them
    apart was to open a stream and read the status code."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    streams = client.get("/api/health").json()["streams"]
    for key in ("preview_open", "preview_max", "transport_open",
                "transport_max", "cameras"):
        assert key in streams, f"/api/health streams is missing {key}"
    assert streams["preview_max"] > streams["preview_open"]


def test_a_refusal_is_logged_not_only_returned():
    """The operator's evidence was 'no errors on the rig' while every new
    stream was being turned away. Silence made a solved problem look like
    a mystery."""
    from pathlib import Path as _P

    from opendarts.live import server as server_mod

    src = _P(server_mod.__file__).read_text()
    block = src[src.index("if in_use >= cap:"):]
    block = block[:block.index("status_code=503")]
    assert "log.warning" in block, "a refused stream must say so in the log"


# --------------------------------------------------------------------------
# One encode per frame, however many consumers want it. Every MJPEG
# connection used to run its own encode, so the same frame was compressed
# once per open stream: 3 cameras x 3 consuming machines x 30fps is 270
# encodes/sec of which 180 are duplicates -- ~37% of a core to produce ~12%
# of a core's worth of distinct bytes (measured at 1.4ms per 1280x720 q85).
# --------------------------------------------------------------------------


def _fake_frame():
    import numpy as np
    return np.zeros((8, 8, 3), dtype=np.uint8)


def test_many_consumers_of_one_frame_encode_it_once(monkeypatch):
    """THE RACE THE LOCK EXISTS FOR. All consumers wake on the same new
    frame, so without it they all miss the cache together and start N
    encodes -- the exact duplication this is meant to remove."""
    import asyncio as _a

    from opendarts.live import server as server_mod

    calls = {"n": 0}

    def slow_encode(frame, max_w, quality):
        calls["n"] += 1
        return b"JPEGBYTES"

    monkeypatch.setattr(server_mod, "_encode_preview_jpeg", slow_encode)
    cache = server_mod._SharedJpegCache()

    async def drive():
        # Ten consumers asking for the SAME frame at the same moment.
        return await _a.gather(*[
            cache.encoded(0, True, 42, _fake_frame()) for _ in range(10)
        ])

    out = _a.run(drive())
    assert all(o == b"JPEGBYTES" for o in out), "every consumer must get bytes"
    assert calls["n"] == 1, f"encoded {calls['n']} times for one frame, expected 1"
    assert cache.encodes == 1 and cache.hits == 9


def test_a_new_frame_is_not_served_from_the_old_one(monkeypatch):
    """Reusing bytes across frames would freeze every consumer's picture --
    a far worse bug than the duplication being fixed."""
    import asyncio as _a

    from opendarts.live import server as server_mod

    seq = {"n": 0}

    def encode(frame, max_w, quality):
        seq["n"] += 1
        return f"frame{seq['n']}".encode()

    monkeypatch.setattr(server_mod, "_encode_preview_jpeg", encode)
    cache = server_mod._SharedJpegCache()

    async def drive():
        a = await cache.encoded(0, True, 1, _fake_frame())
        b = await cache.encoded(0, True, 1, _fake_frame())   # same frame
        c = await cache.encoded(0, True, 2, _fake_frame())   # NEW frame
        return a, b, c

    a, b, c = _a.run(drive())
    assert a == b, "the same frame must be reused"
    assert c != a, "a new frame must be re-encoded"
    assert cache.encodes == 2


def test_preview_and_transport_are_cached_separately(monkeypatch):
    """They are genuinely different images -- different resolution and
    quality -- so one must never be served in place of the other."""
    import asyncio as _a

    from opendarts.live import server as server_mod

    def encode(frame, max_w, quality):
        return f"w={max_w},q={quality}".encode()

    monkeypatch.setattr(server_mod, "_encode_preview_jpeg", encode)
    cache = server_mod._SharedJpegCache()

    async def drive():
        return (await cache.encoded(0, True, 7, _fake_frame()),
                await cache.encoded(0, False, 7, _fake_frame()))

    full, preview = _a.run(drive())
    assert full != preview, "transport and preview bytes must not be shared"
    assert cache.encodes == 2


def test_the_cache_cannot_grow_without_bound(monkeypatch):
    """One entry per camera per mode. Frames arrive forever, so anything
    that accumulated per FRAME would be a leak."""
    import asyncio as _a

    from opendarts.live import server as server_mod

    monkeypatch.setattr(server_mod, "_encode_preview_jpeg",
                        lambda f, w, q: b"x")
    cache = server_mod._SharedJpegCache()

    async def drive():
        for n in range(200):
            for cam in (0, 1, 2):
                await cache.encoded(cam, True, n, _fake_frame())

    _a.run(drive())
    assert len(cache._entries) == 3, (
        f"600 frames left {len(cache._entries)} cached entries")


# --------------------------------------------------------------------------
# The Publishing section on the Info tab.
#
# The virtual-camera state existed only in the log and in /api/frame-health.
# On 2026-09-15 that meant the one fact needed to diagnose a consuming app
# rendering black -- which pixel format is actually in force -- was invisible
# from the dashboard, and the failure it causes looks like a healthy rig.
# --------------------------------------------------------------------------


def test_the_info_tab_has_a_publishing_section(package_root):
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    assert '<h2 class="config-section-title">Publishing</h2>' in html
    assert 'id="publish-tbody"' in html
    # Between the rig's own facts and the Copy-diagnostics button, so a
    # bug report reads top to bottom: what this rig is, then what it is
    # publishing, then the button that copies both.
    assert (html.index("About this rig</h2>")
            < html.index("Publishing</h2>")
            < html.index("btn-copy-diagnostics"))


def test_publishing_is_refreshed_not_loaded_once(package_root):
    """It changes with Start/Stop and with the Autodarts toggle, so a
    load-once value would show a stale format after every change -- which
    is exactly the value someone checks before starting a consuming
    app."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    block = html[html.index("renderDiagnostics(state.config);"):]
    block = block[:block.index("}") + 1]
    assert "refreshPublishing()" in block


def test_a_format_mismatch_is_called_out_rather_than_left_to_the_reader(package_root):
    """'We asked for BGR24' and 'the kernel gave us BGR24' are different
    facts, and they disagree exactly when something is broken --
    v4l2loopback keeps its old format under a live consumer and returns
    success anyway. Printing both side by side and hoping someone compares
    them is how this went unnoticed for an afternoon."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    fn = html[html.index("function renderPublishing"):]
    fn = fn[:fn.index("function fourccMatches")]
    assert "fourccMatches" in fn, "the requested/actual formats must be compared"
    assert "requested" in fn and "in force" in fn, "a mismatch must be spelled out"


def test_the_fourcc_comparison_knows_the_two_spellings(package_root):
    """BGR24/BGR3 and MJPEG/MJPG are the same thing named two ways -- the
    format we ask for and the fourcc the kernel answers with. A literal
    comparison would flag every healthy rig as mismatched."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    fn = html[html.index("function fourccMatches"):]
    fn = fn[:fn.index("async function refreshPublishing")]
    for pair in ("'BGR24'", "'BGR3'", "'MJPEG'", "'MJPG'"):
        assert pair in fn, f"{pair} missing from the equivalence check"


def test_not_publishing_says_what_it_means_for_autodarts(package_root):
    """'not publishing' on its own reads as a missing value. It has a real
    consequence and the row should state it."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    fn = html[html.index("function renderPublishing"):]
    fn = fn[:fn.index("function fourccMatches")]
    assert "not publishing" in fn
    assert "no virtual cameras are being fed" in fn


def test_publishing_renders_the_windows_backend_too(package_root):
    """BOTH backends land in this table, and they can honestly report
    different things. Linux names a /dev node and the fourcc the kernel
    agreed to; Windows writes into shared memory and has neither -- but
    its reader writes counters BACK, so it alone knows whether the
    consumer is actually collecting frames. A renderer that assumed the
    Linux shape would show a Windows rig a row of 'unknown'."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    fn = html[html.index("function renderPublishing"):]
    fn = fn[:fn.index("function fourccMatches")]

    # Linux-only fields must be conditional, not assumed.
    assert "if (s.device)" in fn, "device must be optional -- Windows has none"
    # Windows-only consumer feedback must be surfaced; it is the one thing
    # v4l2loopback cannot report at all.
    for field in ("consumer_attached", "s.missed", "s.torn", "s.read"):
        assert field in fn, f"{field} missing -- Windows reports it and it matters"
    # "Nothing ever read this" and "the reader is behind" are different
    # failures and must not collapse into one badge.
    assert "no consumer" in fn


def test_the_windows_publisher_reports_its_geometry_and_format():
    """It knew both and reported neither, so a Windows rig's row read as
    'unknown' where the value was simply fixed."""
    import inspect

    from opendarts.live import vcam_publish

    src = inspect.getsource(vcam_publish.VirtualCameraPublisher.reader_stats)
    for key in ('"format"', '"width"', '"height"'):
        assert key in src, f"{key} missing from the Windows reader_stats"
    assert "_FORMAT_NAMES[self._format]" in src, (
        "the Windows backend must say which format it last published")


def test_frame_health_reports_whether_a_sink_is_attached(package_root):
    """frame_sink_errors cannot answer this -- 0 means both "never called"
    and "called and fine", so a rig publishing nothing looks identical to
    one publishing perfectly. That ambiguity cost a debugging session on a
    fresh Windows rig: capture running, a publisher set built, and nothing
    connecting the two."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    body = client.get("/api/frame-health").json()
    assert "frame_sink_attached" in body


def test_an_unknown_hub_shape_reports_none_not_false():
    """A wrong False would send someone hunting a wiring bug that does not
    exist. 'Cannot tell' and 'nothing attached' are different answers."""
    from opendarts.live.server import _frame_sink_attached

    class _Opaque:
        pass

    assert _frame_sink_attached(_Opaque()) is None
    assert _frame_sink_attached(None) is None


def test_a_sink_is_detected_on_the_real_hub_whatever_its_slots_read():
    """The all-stream case is the one that was broken, so it is the one
    asserted here.

    The hub used to be a wrapper that forwarded set_frame_sink to a LOCAL
    child only. On an all-stream rig that child never pumped, so nothing
    was ever published while this function -- reading the child -- still
    answered True. Drive the real hub, both ways round, because a fake
    would just agree with whatever this function does."""
    from opendarts.live.remote_capture import build_hub
    from opendarts.live.server import _frame_sink_attached

    for urls in (None, ["http://a/0", "http://a/1", "http://a/2"]):
        hub = build_hub([0, 1, 2], urls)
        assert _frame_sink_attached(hub) is False, f"urls={urls}"
        hub.set_frame_sink(lambda frames: None)
        assert _frame_sink_attached(hub) is True, f"urls={urls}"
        # And back: a flag that can only move toward "attached" is not a
        # diagnostic (docs/DESIGN.md). The AD toggle detaches the sink.
        hub.set_frame_sink(None)
        assert _frame_sink_attached(hub) is False, f"urls={urls}"


# --------------------------------------------------------------------------
# Disk space on the Info tab.
#
# Throw packages are the one thing on a rig that grows without bound, and a
# full disk does not fail where the space went -- it fails at the next write
# on the SCORING path. That is the worst place to learn about a capacity
# problem, so the number belongs somewhere a human sees it first.
# --------------------------------------------------------------------------


def test_disk_usage_is_reported_for_the_package_volume(tmp_path):
    from opendarts.live.server import disk_usage_for

    u = disk_usage_for(tmp_path)
    assert u is not None
    for key in ("total_bytes", "free_bytes", "used_bytes", "free_pct"):
        assert key in u, f"disk usage is missing {key}"
    assert u["total_bytes"] > 0
    assert 0 <= u["free_pct"] <= 100


def test_an_unreadable_volume_reports_none_not_zeros(tmp_path, monkeypatch):
    """A real 0 bytes free is an emergency. It must not be
    indistinguishable from 'could not read', which is a non-event.

    The failure is injected rather than faked with a nonexistent path:
    a path that does not exist YET is now measured via its nearest
    existing parent (see the next test), so it no longer stands in for
    "unreadable".
    """
    import shutil

    from opendarts.live.server import disk_usage_for

    def boom(_):
        raise OSError("volume unreadable")

    monkeypatch.setattr(shutil, "disk_usage", boom)
    assert disk_usage_for(tmp_path) is None


def test_a_packages_dir_that_does_not_exist_yet_still_reports_its_volume(tmp_path):
    """A fresh rig has no data/packages until the first throw is saved.

    shutil.disk_usage raises on a missing path, so every newly set-up
    machine showed "unknown" for disk -- three of the fleet's machines on
    2026-09-16, none of them with a real disk problem. The volume the
    packages WILL land on is the answer, so measure the nearest existing
    ancestor.
    """
    from opendarts.live.server import disk_usage_for

    missing = tmp_path / "data" / "packages"
    assert not missing.exists()
    u = disk_usage_for(missing)
    assert u is not None, "a not-yet-created packages dir reported unknown"
    assert u["total_bytes"] > 0
    assert not missing.exists(), "measuring must not create the directory"


def test_the_state_payload_carries_disk(package_root):
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    cfg = client.get("/api/state").json().get("config") or {}
    assert "disk" in cfg
    assert cfg["disk"] is None or "free_bytes" in cfg["disk"]

# --------------------------------------------------------------------------
# THE FREE-SPACE GUARD, ON THE SCREEN (2026-09-17, decision item 32b/32c)
#
# `opendarts.disk_space` already stops both big writers below a floor
# (`min_free_disk_gb`, 5 GB by default): the capture daemon skips the throw
# package and a ring dump is refused outright. Until now those numbers
# existed only inside the refusal whichever caller happened to trigger one
# got back -- so a rig could stop recording entirely while every screen
# reported a healthy number of gigabytes free.
#
# These pin the two things that fixes: the guard's OWN verdict rides on
# /api/state's existing disk row (not a second comparison written in the
# dashboard), and the page says what has stopped rather than only how full
# the disk is.
# --------------------------------------------------------------------------

_FAKE_TOTAL_BYTES = 500 * 1_000_000_000


def _stub_free_space(monkeypatch, free_bytes: int) -> None:
    """Make every free-space reading in this process return `free_bytes`.

    Injected rather than filling a disk, and injected at `shutil.disk_usage`
    specifically -- the one call `disk_usage_for()` makes -- so the guard's
    answer is derived from the same reading the row reports rather than
    from a second stub that could be set to disagree.
    """
    Usage = collections.namedtuple("Usage", "total used free")

    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda _p: Usage(_FAKE_TOTAL_BYTES, _FAKE_TOTAL_BYTES - free_bytes, free_bytes),
    )


def _disk_row(package_root, monkeypatch, free_bytes: int) -> dict:
    _stub_free_space(monkeypatch, free_bytes)
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    return client.get("/api/state").json()["config"]["disk"]


def test_the_disk_row_carries_the_guards_floor_and_verdict(package_root, monkeypatch):
    disk = _disk_row(package_root, monkeypatch, free_bytes=100 * 1_000_000_000)
    guard = disk["guard"]
    assert guard["enabled"] is True
    assert guard["floor_gb"] == 5.0, "the shipped default floor"
    assert guard["floor_label"] == "5.00 GB"
    assert guard["below_floor"] is False
    assert guard["free_bytes"] == disk["free_bytes"], "two readings, not one"


def test_a_rig_below_the_floor_says_so_with_the_numbers(package_root, monkeypatch):
    guard = _disk_row(package_root, monkeypatch, free_bytes=2 * 1_000_000_000)["guard"]
    assert guard["below_floor"] is True
    assert guard["ok"] is False
    assert guard["free_label"] == "2.00 GB"
    assert guard["floor_label"] == "5.00 GB"
    # The wording comes from the guard itself, so the sentence the screen
    # shows and the sentence a refused capture carries are the same one.
    assert "below the 5.00 GB floor" in guard["reason"]
    assert "min_free_disk_gb=5" in guard["reason"]


def test_the_row_uses_this_rigs_configured_floor_not_the_default(
    package_root, monkeypatch
):
    """20 GB free is fine under the default floor and is NOT fine on a rig
    that asked for 50. The screen has to agree with the writers, which read
    the same key."""
    from opendarts.live.config import write_config_section

    write_config_section("min_free_disk_gb", 50.0)
    guard = _disk_row(package_root, monkeypatch, free_bytes=20 * 1_000_000_000)["guard"]
    assert guard["floor_gb"] == 50.0
    assert guard["floor_label"] == "50.00 GB"
    assert guard["below_floor"] is True


def test_a_disabled_guard_is_reported_as_disabled_not_as_healthy(
    package_root, monkeypatch
):
    """A negative floor is the deliberate opt-out. "Nothing is checking"
    is a different fact from "there is room", and a row that showed the
    second would be the reason nobody noticed the disk filling."""
    from opendarts.live.config import write_config_section

    write_config_section("min_free_disk_gb", -1)
    guard = _disk_row(package_root, monkeypatch, free_bytes=1_000_000)["guard"]
    assert guard["enabled"] is False
    assert guard["below_floor"] is False
    assert "disabled" in guard["reason"]


def _run_note_renderer(tmp_path, html: str, cfg: dict) -> dict:
    """Run the page's own renderRecordedDataNote() against `cfg` in node.

    The element it writes into is stubbed; everything else is the real
    function, lifted out of the HTML the server just served. What comes
    back is what the operator would be looking at.
    """
    start = html.index("function renderRecordedDataNote(")
    source = html[start:html.index("\n}", start) + 2]
    script = tmp_path / "note.js"
    script.write_text(
        "const el = {hidden: true, textContent: '', className: ''};\n"
        "const document = {getElementById: () => el};\n"
        + source
        + "\nrenderRecordedDataNote(" + json.dumps(cfg) + ");\n"
        "process.stdout.write(JSON.stringify(el));\n"
    )
    proc = subprocess.run([shutil.which("node"), str(script)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_the_engines_tab_note_is_silent_while_there_is_room(
    package_root, monkeypatch, tmp_path
):
    if not shutil.which("node"):
        pytest.skip("node not installed -- the rendered note was NOT evaluated")
    _stub_free_space(monkeypatch, 100 * 1_000_000_000)
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    cfg = client.get("/api/state").json()["config"]
    el = _run_note_renderer(tmp_path, client.get("/?ui=classic").text, cfg)
    assert el["hidden"] is True
    assert el["textContent"] == ""


def test_the_engines_tab_note_says_what_has_stopped_below_the_floor(
    package_root, monkeypatch, tmp_path
):
    """The operator is standing at the board, not on the Info tab. Below
    the floor the line has to name both consequences -- packages are not
    being written, and a capture would be refused -- because "low disk" on
    its own does not tell anyone the rig has stopped recording."""
    if not shutil.which("node"):
        pytest.skip("node not installed -- the rendered note was NOT evaluated")
    _stub_free_space(monkeypatch, 2 * 1_000_000_000)
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    cfg = client.get("/api/state").json()["config"]
    el = _run_note_renderer(tmp_path, client.get("/?ui=classic").text, cfg)
    assert el["hidden"] is False
    assert el["textContent"] == (
        "Only 2.00 GB free — below the 5.00 GB floor. Throw packages are not "
        "being written, and a capture would be refused. Copy what you need off "
        "this rig, then delete the recorded data here."
    )
    assert el["className"].endswith("disk-floor-bad")



def test_the_info_tab_renders_disk_in_gb_with_a_threshold(package_root):
    """Nobody reads 468331323392, and '436 GB free' means nothing without
    knowing whether that is most of the disk or the last of it."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    assert "function fmtDisk" in html
    fn = html[html.index("function fmtDisk"):]
    fn = fn[:fn.index("function diagVal")]
    assert "GB free of" in fn, "must report free against total, not free alone"
    # Judged for the reader rather than left as a bare number -- the point
    # of surfacing it is that someone notices before the scoring path does.
    assert "< 5" in fn and "badge bad" in fn
    assert "< 10" in fn and "badge warn" in fn


def test_the_disk_row_is_not_double_escaped(package_root):
    """fmtDisk returns markup when space is low. Passing it through diagVal
    would print the tags as text -- a badge that reads '<span class=...>'
    is worse than no badge."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    block = html[html.index("const preRendered"):]
    block = block[:block.index("}") + 1] if "}" in block[:400] else block[:400]
    assert "'disk'" in block
    assert "preRendered.has(k)" in html


# --------------------------------------------------------------------------
# The top-bar sound indicator.
#
# THREE states, not two. "Sound on" is the user's choice; whether anything
# will actually play is a separate fact the browser owns, and they disagree
# exactly when it matters -- an iPad whose AudioContext moved to
# 'interrupted' reported itself enabled and healthy while silent for a whole
# session. A green/red pair would have said "on" throughout.
# --------------------------------------------------------------------------


def test_the_top_bar_has_a_sound_indicator(package_root):
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    assert 'id="audio-indicator"' in html
    # In the header, beside the connection badge -- not buried in Config,
    # which is the tab you go to when you already suspect something.
    head = html[html.index('<header class="top">'):html.index("</header>")]
    assert 'id="audio-indicator"' in head


def test_the_indicator_distinguishes_blocked_from_off(package_root):
    """Blocked is NOT off and must not be shown as off: off is what the
    user chose, blocked is the user asking for sound and not getting it."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    fn = html[html.index("function renderAudioIndicator"):]
    fn = fn[:fn.index("function renderAudioPanel")]
    assert "sound off" in fn
    assert "sound blocked" in fn
    assert "sound on" in fn
    # blocked must be the alarming colour, not the neutral one
    blocked = fn[fn.index("audioBlocked"):fn.index("sound blocked") + 40]
    assert "badge bad" in blocked, "blocked must not be shown as merely unknown"


def test_the_indicator_reads_the_same_state_as_the_panel(package_root):
    """A second source of truth for 'is sound on' is one that can disagree
    with the first, and the disagreement shows as a green badge over a
    silent room."""
    client = TestClient(create_app(package_root=package_root,
                                   enable_background_poll=False))
    html = client.get("/?ui=classic").text
    fn = html[html.index("function renderAudioIndicator"):]
    fn = fn[:fn.index("function renderAudioPanel")]
    assert "audioSettings.enabled" in fn
    assert "audioBlocked" in fn
    # and it must be driven by the panel's own render, so no path can
    # update one without the other
    panel = html[html.index("function renderAudioPanel"):]
    panel = panel[:panel.index("const enabledSel")]
    assert "renderAudioIndicator()" in panel


# -- per-package cache updates (2026-09-17) ------------------------------


class _SentSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)


def test_a_saved_package_is_read_on_its_own_not_by_rescanning(package_root, monkeypatch):
    """Every saved throw used to re-read the whole archive, twice."""
    import asyncio

    import opendarts.live.server as server_module

    _write_sample_package(package_root / "session-test" / "throw_old", sector="S1")
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state

    def _no_rescan(*a, **k):
        raise AssertionError("a saved package must not trigger a full rescan")

    monkeypatch.setattr(server_module, "discover_packages", _no_rescan)
    new_dir = package_root / "session-test" / "throw_new"
    _write_sample_package(new_dir, sector="S20")
    asyncio.run(state._handle_live_event(  # noqa: SLF001
        {"type": "PACKAGE_SAVED", "path": str(new_dir), "session": "session-test"}))
    assert {p["sector"] for p in state.list_packages()} == {"S1", "S20"}
    # The follow-up save for the same throw replaces, never duplicates.
    asyncio.run(state._handle_live_event(  # noqa: SLF001
        {"type": "PACKAGE_SAVED", "path": str(new_dir), "session": "session-test"}))
    assert len(state.list_packages()) == 2


def test_the_poll_reads_only_new_packages_and_drops_removed_ones(package_root, monkeypatch):
    import asyncio
    import shutil

    import opendarts.live.server as server_module

    keep = _write_sample_package(package_root / "s" / "keep", sector="S1")
    gone = _write_sample_package(package_root / "s" / "gone", sector="S2")
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    state.package_poll_interval_s = 0
    fresh = _write_sample_package(package_root / "s" / "fresh", sector="S3")
    shutil.rmtree(gone)

    reads: list = []
    real = server_module._package_record

    def _counting(meta_path):
        reads.append(meta_path.parent.name)
        return real(meta_path)

    monkeypatch.setattr(server_module, "_package_record", _counting)
    sock = _SentSocket()
    state.clients.add(sock)

    async def _one_poll() -> None:
        task = asyncio.create_task(state._package_poll_loop())  # noqa: SLF001
        while not sock.sent:
            await asyncio.sleep(0.01)
        task.cancel()

    asyncio.run(_one_poll())
    assert reads == ["fresh"]
    assert {p["sector"] for p in state.list_packages()} == {"S1", "S3"}
    msg = json.loads(sock.sent[0])
    assert msg["count"] == 2 and msg["new_count"] == 1
    assert msg["packages"][0]["path"] == str(fresh)
    assert keep  # still listed


def test_an_edit_to_an_older_throw_reaches_open_tabs(package_root):
    """The broadcast carried only the newest 20, so marking an older
    throw changed nothing on screen until a reload."""
    import asyncio

    for n in range(22):
        _write_sample_package(package_root / "session-test" / f"t{n:02d}", sector="S1")
    app = create_app(package_root=package_root, enable_background_poll=False)
    state = app.state.opendarts_state
    oldest = state.list_packages()[-1]
    sock = _SentSocket()
    state.clients.add(sock)
    asyncio.run(state.mark_ad_wrong(oldest["session"], oldest["throw_id"], True, None))
    updates = [json.loads(m) for m in sock.sent if '"PACKAGES_UPDATED"' in m]
    assert updates and updates[-1]["packages"][0]["path"] == oldest["path"]
    assert updates[-1]["count"] == 22


def test_the_engines_table_is_only_rebuilt_while_it_can_be_seen(package_root):
    """One row per engine per throw, rebuilt whole twice per throw -- on a TV
    showing the Scoring tab nobody sees it."""
    html = TestClient(create_app(package_root=package_root,
                                 enable_background_poll=False)).get("/?ui=classic").text
    body = html[html.index("function renderScoringTable()"):]
    body = body[:body.index("\nfunction computeEngineTally")]
    assert "if (packagesTableShowing()) {" in body
    assert "packagesTableStale = true;" in body
    assert "renderPackagesTableIfStale();" in html          # on tab switch
    assert "addEventListener('visibilitychange', renderPackagesTableIfStale)" in html


def test_config_only_polls_run_only_while_their_tab_is_showing(package_root):
    html = TestClient(create_app(package_root=package_root,
                                 enable_background_poll=False)).get("/?ui=classic").text
    assert "setInterval(refreshCameraStatus," not in html
    assert "setInterval(refreshAudioClients," not in html
    assert "if (tabShowing('config') || tabShowing('info')) refreshCameraStatus();" in html
    assert "if (tabShowing('config')) refreshAudioClients();" in html


def test_the_dashboard_page_is_gzipped_when_the_browser_accepts_it(package_root):
    client = TestClient(create_app(package_root=package_root, enable_background_poll=False))
    zipped = client.get("/", headers={"Accept-Encoding": "gzip"})
    assert zipped.headers["content-encoding"] == "gzip"
    assert "<title>" in zipped.text                     # decoded transparently
    plain = client.get("/", headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    assert plain.text == zipped.text
