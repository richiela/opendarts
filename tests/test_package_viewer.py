"""The per-throw viewer routes: the standalone HTML window, the bg/after
still PNGs (after byte-identical to the scored commit, from the clip), and
the looping-MJPEG clip stream.
"""
from __future__ import annotations

import re
import types

import cv2
import numpy as np
from fastapi.testclient import TestClient

from opendarts.capture import clip
from opendarts.capture.throw_package import save_throw_package
from opendarts.live.server import create_app
from opendarts.pipeline import CameraCalibration, ScoreResult


def _frames(n, w=64, h=48, seed=0):
    rng = np.random.default_rng(seed)
    base = np.repeat((np.add.outer(np.arange(h), np.arange(w)) % 256).astype(np.uint8)[:, :, None], 3, 2)
    return [np.clip(base + rng.integers(0, 30, (h, w, 3)), 0, 255).astype(np.uint8) for _ in range(n)]


def _calib(seed):
    rng = np.random.default_rng(seed)
    return CameraCalibration(
        camera_matrix=np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]], np.float64),
        dist_coeffs=np.zeros(5, np.float64), rvec=rng.uniform(-0.1, 0.1, 3).astype(np.float64),
        tvec=np.array([0.0, 0.0, 400.0], np.float64), pnp_result=None, landmark_spread_ok=True)


def _make_recorded_package(root, session="sess", throw="t1"):
    cams = range(3)
    fr = _frames(9, seed=1)
    sets = [types.SimpleNamespace(generation=i, wall_s=float(i), pixels={c: fr[i] for c in cams},
                                  jpegs={}) for i in range(len(fr))]
    commit = {c: fr[4] for c in cams}
    # The bg is a frame the ring actually held, named by its generation
    # like the commit -- the package's one clip runs from it.
    bg = {c: fr[1] for c in cams}
    pkg = root / session / throw
    save_throw_package(pkg, session, bg, commit, {c: _calib(c) for c in cams},
                       ScoreResult(ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
                                   triangulation=None, n_cameras_used=3, max_ray_disagreement_mm=0.5),
                       defer_clips=True)
    scored = clip.ScoredFrames(bg=bg, commit=commit,
                               bg_generations={c: 1 for c in cams},
                               commit_generations={c: 4 for c in cams})
    clip.point_meta_at_clip(pkg, clip.write_window_clips(pkg, sets, scored, cams))
    return commit, bg


def _client(root):
    return TestClient(create_app(package_root=root, enable_background_poll=False))


def test_viewer_html_lists_cameras_and_clips(tmp_path):
    _make_recorded_package(tmp_path)
    r = _client(tmp_path).get("/packages/sess/t1/viewer")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    for c in range(3):
        assert f"/frame/{c}/bg.png" in r.text and f"/frame/{c}/after.png" in r.text
    # a recorded throw shows the scrubber, which pulls frames from clip.json
    assert "/clip.json" in r.text
    assert 'id="slider"' in r.text
    assert 'id="btnPlay"' in r.text
    assert 'data-speed="0.25"' in r.text and 'data-speed="0.5"' in r.text


def test_viewer_playback_does_not_loop_and_binds_arrow_keys(tmp_path):
    """Customer-facing scrubber: plays once (no loop) and steps on the
    arrow keys."""
    _make_recorded_package(tmp_path)
    html = _client(tmp_path).get("/packages/sess/t1/viewer").text
    assert "play once, no loop" in html
    assert "cur >= nFrames - 1" in html    # play stops at the last frame
    assert "ArrowLeft" in html and "ArrowRight" in html


def test_viewer_images_open_full_res_lightbox(tmp_path):
    """Stills and clip frames are click-to-zoom into a full-resolution
    lightbox modal."""
    _make_recorded_package(tmp_path)
    html = _client(tmp_path).get("/packages/sess/t1/viewer").text
    assert 'id="lightbox"' in html
    assert 'class="zoomable"' in html            # stills are zoomable
    assert "img.zoomable" in html                # the click handler targets them
    assert "img.className = 'zoomable'" in html  # scrubber frames too


def test_viewer_grid_uses_compressed_and_lightbox_uses_full_res(tmp_path):
    """The grid loads the compressed (jpeg) still; the lightbox opens the
    full-resolution version via data-full. Clip frames get a lossless
    per-index png as their data-full."""
    _make_recorded_package(tmp_path)
    html = _client(tmp_path).get("/packages/sess/t1/viewer").text
    # stills: grid src is jpeg, data-full is the plain png
    assert "/frame/0/bg.png?fmt=jpeg" in html
    assert 'data-full="/api/packages/sess/t1/frame/0/bg.png"' in html
    # the lightbox prefers data-full
    assert "img.dataset.full" in html
    # scrubber frames point data-full at the per-index lossless png route
    assert "'/clip/' + c.id + '/' + j + '.png'" in html


def test_still_fmt_jpeg_is_smaller_than_png_same_pixels(tmp_path):
    """?fmt=jpeg returns a smaller image/jpeg; default png is lossless and
    the two decode to the same size image."""
    _make_recorded_package(tmp_path)
    client = _client(tmp_path)
    png = client.get("/api/packages/sess/t1/frame/0/bg.png")
    jpg = client.get("/api/packages/sess/t1/frame/0/bg.png?fmt=jpeg")
    assert png.headers["content-type"] == "image/png"
    assert jpg.headers["content-type"] == "image/jpeg"
    assert len(jpg.content) < len(png.content)          # compressed
    a = cv2.imdecode(np.frombuffer(png.content, np.uint8), cv2.IMREAD_COLOR)
    b = cv2.imdecode(np.frombuffer(jpg.content, np.uint8), cv2.IMREAD_COLOR)
    assert a.shape == b.shape                            # same dimensions
    assert client.get("/api/packages/sess/t1/frame/0/bg.png?fmt=bogus").status_code == 400


def test_clip_frame_png_is_lossless_and_matches_clip(tmp_path):
    """The per-index clip-frame png is byte-identical to the frame in the
    clip (lossless), and 404s out of range / for a cam with no clip."""
    _make_recorded_package(tmp_path)
    client = _client(tmp_path)
    frames = clip.read_clip_frames(
        tmp_path / "sess" / "t1" / "clip_cam0.mkv")
    r = client.get("/api/packages/sess/t1/clip/0/0.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    got = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(got, frames[0])
    assert client.get("/api/packages/sess/t1/clip/0/999.png").status_code == 404
    assert client.get("/api/packages/sess/t1/clip/9/0.png").status_code == 404


def test_clip_json_serves_data_url_frames_per_camera(tmp_path):
    commit, _bg = _make_recorded_package(tmp_path)
    r = _client(tmp_path).get("/api/packages/sess/t1/clip.json")
    assert r.status_code == 200
    data = r.json()
    assert data["fps"] > 0
    assert set(data["cameras"]) == {"0", "1", "2"}
    for cam_key, entry in data["cameras"].items():
        assert entry["frames"], f"cam{cam_key} has no frames"
        assert all(f.startswith("data:image/jpeg;base64,") for f in entry["frames"])
        assert 0 <= entry["commit_index"] < len(entry["frames"])


def test_clip_json_404_when_no_video(tmp_path):
    cams = range(2)
    bg = {c: _frames(1, seed=c)[0] for c in cams}
    dart = {c: _frames(1, seed=40 + c)[0] for c in cams}
    save_throw_package(tmp_path / "s" / "plain", "s", bg, dart,
                       {c: _calib(c) for c in cams},
                       ScoreResult(ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
                                   triangulation=None, n_cameras_used=2, max_ray_disagreement_mm=0.5))
    assert _client(tmp_path).get("/api/packages/s/plain/clip.json").status_code == 404


def test_after_frame_is_byte_identical_to_scored_commit(tmp_path):
    commit, bg = _make_recorded_package(tmp_path)
    client = _client(tmp_path)
    for c in range(3):
        r = client.get(f"/api/packages/sess/t1/frame/{c}/after.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        arr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(arr, commit[c]), f"cam{c} after frame not byte-identical"
        rbg = client.get(f"/api/packages/sess/t1/frame/{c}/bg.png")
        assert np.array_equal(cv2.imdecode(np.frombuffer(rbg.content, np.uint8), cv2.IMREAD_COLOR), bg[c])


def test_clip_stream_route_404_for_camera_without_a_clip(tmp_path):
    """A camera index with no clip returns 404 immediately (no stream).

    The streaming SUCCESS path is deliberately not exercised here: it is an
    intentionally endless multipart response that only stops on client
    disconnect, and the sync TestClient never signals disconnect, so
    consuming it would hang the suite. The frame content is covered by the
    clip module tests and the after.png route above; the live 200/multipart
    response is checked with curl against a real server (P6)."""
    _make_recorded_package(tmp_path)
    client = _client(tmp_path)
    assert client.get("/api/packages/sess/t1/clip/9.mjpg").status_code == 404


def test_bad_package_and_bad_kind(tmp_path):
    client = _client(tmp_path)
    assert client.get("/packages/sess/nope/viewer").status_code == 404
    _make_recorded_package(tmp_path)
    assert client.get("/api/packages/sess/t1/frame/0/bogus.png").status_code == 400
    assert client.get("/api/packages/sess/../etc/frame/0/bg.png").status_code in (400, 404)


def test_api_packages_reports_has_video(tmp_path):
    """A recorded package reports has_video=True (dashboard hides "Save
    frames" and shows the clip); an unrecorded one reports False."""
    _make_recorded_package(tmp_path, session="s", throw="rec")
    # an unrecorded (bg+dart PNG) throw
    cams = range(2)
    bg = {c: _frames(1, seed=c)[0] for c in cams}
    dart = {c: _frames(1, seed=40 + c)[0] for c in cams}
    save_throw_package(tmp_path / "s" / "plain", "s", bg, dart,
                       {c: _calib(c) for c in cams},
                       ScoreResult(ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
                                   triangulation=None, n_cameras_used=2, max_ray_disagreement_mm=0.5))
    pkgs = {p["throw_id"]: p for p in _client(tmp_path).get("/api/packages").json()}
    assert pkgs["rec"]["has_video"] is True
    assert pkgs["plain"]["has_video"] is False


def test_dashboard_hides_save_frames_when_recorded_and_always_shows_view(tmp_path):
    """Source check: Save frames is gated on the record MODE (and only then
    on has_video); View is not gated at all.

    Gating on has_video alone was not enough: the clip is finalised a moment
    AFTER the package is saved, so a freshly-saved throw reads
    has_video=false and the useless button flashed up on every dart even
    with video_record_mode="all".
    """
    from opendarts.live.server import create_app
    from fastapi.testclient import TestClient
    html = TestClient(create_app(package_root=tmp_path, enable_background_poll=False)).get("/?ui=classic").text
    assert "VIDEO_RECORD_MODE !== 'all' && !p.has_video" in html
    assert "OD_BOOTSTRAP.video_record_mode" in html  # the mode reaches the page
    assert "view-throw-btn" in html             # View always rendered
    assert "sector_match === false" in html     # AD controls gated on a real miss


def test_view_button_reachable_when_ad_never_answered(tmp_path):
    """A throw AD never answered must still offer View.

    The per-throw controls lived only in the AD row, and that row is not
    rendered at all when AD was not asked (ad_matched null). So those
    throws had no View button -- the frames were unreachable from the
    dashboard on exactly the throws you most want to inspect. With no AD
    row the FIRST engine row has to carry them, the same way it already
    carries the throw number and capture time.
    """
    from opendarts.live.server import create_app
    from fastapi.testclient import TestClient
    html = TestClient(create_app(package_root=tmp_path, enable_background_poll=False)).get("/?ui=classic").text
    # the engine row's match cell falls back to the throw-level controls
    assert "!adAsked && si === 0" in html
    assert "? fmtAdWrongCell(p, sections)" in html


def test_view_button_has_its_own_column_not_the_match_cell(tmp_path):
    """View must not share the cell that overflows.

    On a corrected throw the match cell carries four things already --
    "AD wrong (human)", AD's own FAIL, the "truth: <segment> (<engine>)"
    badge and Unmark. A fifth control pushed the row wide enough to clip
    the truth badge, so the one button you always want was the one
    squeezed out. Its own column keeps it in the same place on every
    throw whatever the match cell happens to hold.
    """
    from opendarts.live.server import create_app
    from fastapi.testclient import TestClient
    html = TestClient(create_app(package_root=tmp_path, enable_background_poll=False)).get("/?ui=classic").text
    assert 'class="col-view"' in html            # the column exists
    assert "function fmtViewCell(p)" in html     # built separately from the badges
    # and it is NOT emitted from inside the badge cell any more
    m = re.search(r"function fmtAdWrongCell\(p, sections\) \{(.*?)\n\}", html, re.S)
    assert m, "fmtAdWrongCell not found"
    assert "view-throw-btn" not in m.group(1), (
        "View is being rendered inside fmtAdWrongCell again -- that is the "
        "crowded cell it was moved out of"
    )


def test_dashboard_warns_when_ad_is_stuck_in_takeout(tmp_path):
    """Takeout is normal for seconds; stuck is a silent loss of the oracle.

    A wedged AD reports no further throws while its buffer keeps serving
    the finished visit. The matcher now refuses those stale answers, so the
    data is honest -- but the operator still needs telling, during the
    session, that AD has stopped being a reference at all.
    """
    from opendarts.live.server import create_app
    from fastapi.testclient import TestClient
    html = TestClient(create_app(package_root=tmp_path, enable_background_poll=False)).get("/?ui=classic").text
    assert 'id="ad-takeout-warning"' in html          # the banner exists
    assert "AD_TAKEOUT_STUCK_SEC" in html             # and a threshold to fire on
    assert "state.ad_board_status_age_sec" in html    # fed by how long it has sat there
    assert "stuck in takeout" in html.lower()


def test_dashboard_bootstrap_carries_the_record_mode(tmp_path):
    """The record mode must actually be serialised into the page bootstrap,
    or the gate above silently reads undefined and shows the button."""
    import json as _json
    import re as _re
    from opendarts.live.server import create_app
    from fastapi.testclient import TestClient
    html = TestClient(create_app(package_root=tmp_path, enable_background_poll=False)).get("/?ui=classic").text
    m = _re.search(
        r'<script id="bootstrap" type="application/json">(.*?)</script>', html, _re.S)
    assert m, "bootstrap JSON block not found in the page"
    boot = _json.loads(m.group(1))
    assert boot["video_record_mode"] in ("never", "mismatch", "all")


# -- the header: when, how big, the visit, and the throws either side --------

def _save_plain(root, session, throw, *, captured, visit_id=None, visit_index=None):
    cams = range(3)
    fr = _frames(2, seed=len(throw))
    save_throw_package(root / session / throw, session, {c: fr[0] for c in cams},
                       {c: fr[1] for c in cams}, {c: _calib(c) for c in cams},
                       ScoreResult(ok=True, sector="20", ring="single_inner", board_xy_mm=(1.0, 2.0),
                                   triangulation=None, n_cameras_used=3, max_ray_disagreement_mm=0.5),
                       visit_id=visit_id, visit_index=visit_index, captured_at_utc=captured)


def _header(html: str) -> str:
    return html[html.index('<header class="top">'):html.index("</header>")]


def test_header_shows_capture_time_size_and_what_was_recorded(tmp_path):
    _save_plain(tmp_path, "s1", "s1-001-S20", captured="2026-09-22T23:50:36.000000+00:00")
    head = _header(_client(tmp_path).get("/packages/s1/s1-001-S20/viewer").text)
    # The ISO time rides in `datetime` for the script to localise; the raw
    # value is the visible fallback if it never runs.
    assert '<time id="capturedAt" datetime="2026-09-22T23:50:36.000000+00:00">' in head
    assert re.search(r"\d+ KB|\d+\.\d MB", head), "package size missing"
    assert "3 cameras" in head and "stills only" in head
    # The package name no longer headlines the page -- the score does.
    assert "<h1>S20</h1>" in head


def test_visit_strip_lists_the_whole_visit_in_dart_order(tmp_path):
    for i, (tid, t) in enumerate([("s1-001-S20", "00"), ("s1-002-T19", "05"), ("s1-003-S1", "10")]):
        _save_plain(tmp_path, "s1", tid, captured=f"2026-09-22T23:50:{t}+00:00",
                    visit_id="v1", visit_index=i)
    head = _header(_client(tmp_path).get("/packages/s1/s1-002-T19/viewer").text)
    darts = re.findall(r'class="dart[^"]*"[^>]*>([^<]+)<', head)
    assert darts == ["S20", "T19", "S1"]
    assert '<span class="dart dart-now" aria-current="true">T19</span>' in head
    assert 'href="/packages/s1/s1-001-S20/viewer"' in head
    assert 'href="/packages/s1/s1-003-S1/viewer"' in head


def test_an_unfinished_visit_shows_its_empty_slots(tmp_path):
    _save_plain(tmp_path, "s1", "s1-001-S20", captured="2026-09-22T23:50:00+00:00",
                visit_id="v1", visit_index=0)
    head = _header(_client(tmp_path).get("/packages/s1/s1-001-S20/viewer").text)
    assert head.count("dart-empty") == 2


def test_there_is_no_older_newer_paging_the_visit_is_the_navigation(tmp_path):
    """People step through a turn's three darts, not from turn to turn."""
    for i, (tid, t) in enumerate([("s1-001-S20", "00"), ("s1-002-T19", "05"), ("s1-003-S1", "10"),
                                  ("s1-004-D5", "40")]):
        _save_plain(tmp_path, "s1", tid, captured=f"2026-09-22T23:50:{t}+00:00",
                    visit_id="v1" if i < 3 else "v2", visit_index=i % 3)
    head = _header(_client(tmp_path).get("/packages/s1/s1-002-T19/viewer").text)
    assert "Older" not in head and "Newer" not in head
    # The other visit's dart is never offered.
    assert "s1-004-D5" not in head
    # The strip sits in the title row, beside the score.
    title = head[head.index('<div class="title-row">'):head.index('<div class="details">')]
    assert '<nav class="visit"' in title
