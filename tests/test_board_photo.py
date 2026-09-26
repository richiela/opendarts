"""The Scoring tab's board photo and the per-dart facts it is drawn with.

Everything here runs against synthetic cameras placed the way a real rig's
are -- three of them around the board, ~240 mm out, looking in steeply --
so the geometry is exercised for real without any rig data:

  * the photo reproduces a known board texture from three camera views;
  * the shaft direction comes back from per-camera shaft lines;
  * the flight colour comes back from a painted flight;
  * the server stores a BOARD_PHOTO, serves it, and tells screens.
"""
from __future__ import annotations

import asyncio
import threading

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendarts.live import board_photo as bp
from opendarts.live.server import create_app
from opendarts.pipeline import CameraCalibration

W, H = 1280, 720
K = np.array([[740.0, 0, W / 2], [0, 740.0, H / 2], [0, 0, 1]])
# Where a real rig's cameras sit (board mm): upper left, right, lower left.
CAMERA_CENTRES = {0: (-160.0, 308.0, 250.0), 1: (345.0, -20.0, 235.0), 2: (-190.0, -292.0, 240.0)}


def _looking_at_bull(centre) -> CameraCalibration:
    C = np.array(centre, dtype=float)
    fwd = -C / np.linalg.norm(C)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.array([right, down, fwd])
    rvec, _ = cv2.Rodrigues(R)
    return CameraCalibration(
        camera_matrix=K.copy(), dist_coeffs=np.zeros((5, 1)),
        rvec=rvec, tvec=(-R @ C).reshape(3, 1),
        pnp_result=None, landmark_spread_ok=True,
    )


@pytest.fixture(scope="module")
def cals():
    return {cam: _looking_at_bull(c) for cam, c in CAMERA_CENTRES.items()}


# A smooth colour texture over the board plane, 2 px per mm, +/-240 mm.
TEX_PX_PER_MM, TEX_HALF = 2.0, 240.0
TEX_N = int(2 * TEX_HALF * TEX_PX_PER_MM)


def _texture() -> np.ndarray:
    v, u = np.mgrid[0:TEX_N, 0:TEX_N].astype(np.float32)
    x = u / TEX_PX_PER_MM - TEX_HALF
    y = TEX_HALF - v / TEX_PX_PER_MM
    return np.stack([
        128 + 100 * np.sin(x / 23.0), 128 + 100 * np.cos(y / 31.0), 128 + 90 * np.sin((x + y) / 41.0),
    ], axis=2).clip(0, 255).astype(np.uint8)


def _camera_view(tex: np.ndarray, cal: CameraCalibration) -> np.ndarray:
    """What `cal` sees of a flat board painted with `tex`."""
    R, _ = cv2.Rodrigues(cal.rvec)
    board_to_px = cal.camera_matrix @ np.column_stack([R[:, 0], R[:, 1], cal.tvec.ravel()])
    tex_to_board = np.array([[1 / TEX_PX_PER_MM, 0, -TEX_HALF], [0, -1 / TEX_PX_PER_MM, TEX_HALF], [0, 0, 1]])
    return cv2.warpPerspective(tex, board_to_px @ tex_to_board, (W, H))


def test_the_photo_reproduces_the_board_straight_on(cals):
    tex = _texture()
    frames = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    maps = bp.BoardPhotoMaps(cals, {cam: (W, H) for cam in cals})
    photo = maps.render(frames).astype(np.float32)

    ys, xs = np.mgrid[0:bp.SIZE_PX, 0:bp.SIZE_PX].astype(np.float32)
    X = (xs - bp.SIZE_PX / 2) / bp.PX_PER_MM
    Y = (bp.SIZE_PX / 2 - ys) / bp.PX_PER_MM
    expected = cv2.remap(tex, (X + TEX_HALF) * TEX_PX_PER_MM, (TEX_HALF - Y) * TEX_PX_PER_MM,
                         cv2.INTER_LINEAR).astype(np.float32)
    board = np.hypot(X, Y) < 200
    err = np.abs(photo - expected).mean(axis=2)[board]
    assert np.median(err) < 3.0, "a flat board must come back where it is"
    assert np.percentile(err, 95) < 12.0


def test_the_number_ring_takes_each_number_from_one_camera(cals):
    """A soft blend of two cameras doubles raised digits; in the number
    ring every spot must come (almost) entirely from one camera."""
    maps = bp.BoardPhotoMaps(cals, {cam: (W, H) for cam in cals})
    ys, xs = np.mgrid[0:bp.SIZE_PX, 0:bp.SIZE_PX]
    X = (xs - bp.SIZE_PX / 2) / bp.PX_PER_MM
    Y = (bp.SIZE_PX / 2 - ys) / bp.PX_PER_MM
    r = np.hypot(X, Y)
    theta = np.degrees(np.arctan2(Y, X)) % 18.0
    # a number's own middle: well inside its 18 degree slice, on the digits
    digits = (r > 190) & (r < 215) & (np.abs(theta - 9.0) > 4.0)
    top = np.max([m.weight for m in maps.cameras.values()], axis=0)
    covered = np.sum([m.weight for m in maps.cameras.values()], axis=0) > 0.5
    assert np.all(top[digits & covered] > 0.97)


def _checker_texture(square_mm: float = 6.0) -> np.ndarray:
    """Hard edges everywhere, like wires and digits: what join alignment
    measures against."""
    v, u = np.mgrid[0:TEX_N, 0:TEX_N].astype(np.float32)
    x = u / TEX_PX_PER_MM - TEX_HALF
    y = TEX_HALF - v / TEX_PX_PER_MM
    cell = (np.floor(x / square_mm) + np.floor(y / square_mm)) % 2
    g = (40 + 180 * cell).astype(np.uint8)
    return np.stack([g, g, g], axis=2)


def _nudged(cal: CameraCalibration, dx_mm: float, dy_mm: float) -> CameraCalibration:
    """The same camera with a slightly wrong solve -- what a real rig's
    calibration is, a little, at the rim."""
    return CameraCalibration(camera_matrix=cal.camera_matrix, dist_coeffs=cal.dist_coeffs, rvec=cal.rvec,
                             tvec=cal.tvec + np.array([[dx_mm], [dy_mm], [0.0]]), pnp_result=None,
                             landmark_spread_ok=True)


def _float_map(maps, cam):
    return cv2.convertMaps(maps.cameras[cam].map1, maps.cameras[cam].map2, cv2.CV_32FC1)


def test_joins_between_cameras_are_measured_and_lined_up(cals):
    """Frames from the true cameras, maps from a slightly wrong solve of one
    of them: the joins it takes part in are found out of line, and moving
    each side half-way brings them together."""
    tex = _checker_texture()
    frames = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    solved = dict(cals)
    solved[1] = _nudged(cals[1], 3.0, -2.0)
    maps = bp.BoardPhotoMaps(solved, {cam: (W, H) for cam in cals})
    joins = maps.align_joins(frames)
    touching = [j for j in joins if 1 in j["cameras"]]
    assert touching, "camera 1 hands over to another somewhere in the ring"
    assert all(j["applied"] and j["shift_mm"] > 0.5 for j in touching), touching
    assert all(j["edge_diff"][1] < 0.85 * j["edge_diff"][0] for j in touching), touching


def test_aligning_never_moves_the_scoring_area(cals):
    tex = _checker_texture()
    frames = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    solved = dict(cals)
    solved[1] = _nudged(cals[1], 3.0, -2.0)
    maps = bp.BoardPhotoMaps(solved, {cam: (W, H) for cam in cals})
    before = {cam: _float_map(maps, cam) for cam in cals}
    maps.align_joins(frames)
    ys, xs = np.mgrid[0:bp.SIZE_PX, 0:bp.SIZE_PX]
    r = np.hypot((xs - bp.SIZE_PX / 2) / bp.PX_PER_MM, (bp.SIZE_PX / 2 - ys) / bp.PX_PER_MM)
    scoring = r < 176.0     # the double wire, and the spider's tips past it
    for cam in cals:
        after = _float_map(maps, cam)
        assert np.allclose(after[0][scoring], before[cam][0][scoring], atol=0.05)
        assert np.allclose(after[1][scoring], before[cam][1][scoring], atol=0.05)


def test_joins_that_already_agree_are_left_alone(cals):
    tex = _checker_texture()
    frames = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    maps = bp.BoardPhotoMaps(cals, {cam: (W, H) for cam in cals})
    joins = maps.align_joins(frames)
    assert joins and not any(j["applied"] for j in joins), joins


def _shaft_lines(cals, tip, axis, *, flip=False, drop=()):
    lines = {}
    for cam, cal in cals.items():
        pts = np.array([tip, np.array(tip) + np.array(axis) * 60.0])
        uv, _ = cv2.projectPoints(pts, cal.rvec, cal.tvec, cal.camera_matrix, cal.dist_coeffs)
        p1, p2 = uv.reshape(2, 2).tolist()
        if flip:
            p1, p2 = p2, p1
        lines[str(cam)] = {"p1_px": p1, "p2_px": p2, "dropped": cam in drop}
    return lines


def _angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def test_dart_axis_comes_back_from_the_shaft_lines(cals):
    true_axis = np.array([0.10, 0.20, 0.97])
    tip = [40.0, 60.0, 0.0]
    zeus = {"sub_results": {"Talos": {"diagnostics": {"line_px": _shaft_lines(cals, tip, true_axis)}}}}
    axis = bp.dart_axis(zeus, cals)
    assert axis is not None and _angle_deg(axis, true_axis) < 0.3
    # Talos as the primary engine carries the lines directly; the line
    # direction does not matter, the answer always points out of the board.
    talos = {"line_px": _shaft_lines(cals, tip, true_axis, flip=True)}
    axis = bp.dart_axis(talos, cals)
    assert axis is not None and axis[2] > 0 and _angle_deg(axis, true_axis) < 0.3


def test_no_axis_without_two_usable_lines(cals):
    tip, axis = [0.0, 100.0, 0.0], [0.0, 0.1, 0.99]
    one = {"line_px": _shaft_lines(cals, tip, axis, drop=(1, 2))}
    assert bp.dart_axis(one, cals) is None
    assert bp.dart_axis(None, cals) is None
    assert bp.dart_axis({"sub_results": {}}, cals) is None


def test_flight_color_is_read_from_the_flights(cals):
    tip, axis = (30.0, -50.0), bp.dart_axis(
        {"line_px": _shaft_lines(cals, [30.0, -50.0, 0.0], [0.05, 0.15, 0.99])}, cals)
    bg = {cam: np.full((H, W, 3), 110, np.uint8) for cam in cals}
    frames = {}
    a = np.array(axis)
    e1 = np.cross(a, [0, 1, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    flight = [np.array([*tip, 0.0]) + a * s + e * w
              for s in np.linspace(90, 125, 15) for e in (e1, -e1, e2, -e2) for w in np.linspace(0, 25, 12)]
    for cam, cal in cals.items():
        img = bg[cam].copy()
        uv, _ = cv2.projectPoints(np.array(flight), cal.rvec, cal.tvec, cal.camera_matrix, cal.dist_coeffs)
        for u, v in uv.reshape(-1, 2):
            cv2.circle(img, (int(u), int(v)), 4, (30, 40, 220), -1)   # BGR: red
        frames[cam] = img
    colour = bp.flight_color(frames, bg, cals, tip, axis)
    assert colour is not None
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    assert r > 180 and g < 80 and b < 80, colour
    # nothing painted: nothing to read, and no guess
    assert bp.flight_color(bg, bg, cals, tip, axis) is None


def test_the_renderer_hands_back_a_jpeg_and_swallows_failures(cals):
    tex = _texture()
    frames = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    got, done = [], threading.Event()

    def on_done(jpeg):
        got.append(jpeg)
        done.set()

    r = bp.BoardPhotoRenderer()
    r.submit({}, cals, on_done)           # nothing to render: logged, not raised
    r.submit(frames, cals, on_done)
    assert done.wait(20)
    img = cv2.imdecode(np.frombuffer(got[-1], np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (bp.SIZE_PX, bp.SIZE_PX, 3)


def _noisy(frames, seed, sigma=1.5, shift=0.0):
    """The same board a while later: sensor noise, optionally an exposure
    shift of `shift` grey levels."""
    rng = np.random.default_rng(seed)
    return {cam: np.clip(f.astype(np.float32) + shift + rng.normal(0, sigma, f.shape), 0, 255).astype(np.uint8)
            for cam, f in frames.items()}


def test_an_unchanged_board_is_not_re_rendered(cals):
    tex = _texture()
    board = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    r = bp.BoardPhotoRenderer()
    assert r.render_jpeg(board, cals, skip_if_unchanged=True) is not None, "no previous photo: always render"
    assert r.render_jpeg(_noisy(board, 1), cals, skip_if_unchanged=True) is None
    assert r.render_jpeg(_noisy(board, 2), cals, skip_if_unchanged=True) is None
    # the Start photo never skips
    assert r.render_jpeg(_noisy(board, 3), cals) is not None


def test_a_changed_board_is_re_rendered(cals):
    tex = _texture()
    board = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    r = bp.BoardPhotoRenderer()
    r.render_jpeg(board, cals)
    # the lights changed: a uniform step well past the mean limit
    assert r.render_jpeg(_noisy(board, 1, shift=6.0), cals, skip_if_unchanged=True) is not None
    # compared with THAT photo now; a dart left in the board is a small
    # patch -- the mean hardly moves, but the patch is a real change
    brighter = _noisy(board, 1, shift=6.0)
    dart = {cam: f.copy() for cam, f in brighter.items()}
    for cam, cal in cals.items():
        uv, _ = cv2.projectPoints(np.array([[30.0, 80.0, 0.0]]), cal.rvec, cal.tvec,
                                  cal.camera_matrix, cal.dist_coeffs)
        u, v = uv.reshape(2).astype(int)
        cv2.rectangle(dart[cam], (u - 6, v - 40), (u + 6, v), (10, 10, 10), -1)
    before = r._last[1]  # noqa: SLF001
    after = r._maps.summary(dart)  # noqa: SLF001
    assert max(float(np.abs(before[c] - after[c]).mean()) for c in cals) <= bp.UNCHANGED_MAX_MEAN_DIFF
    assert r.render_jpeg(dart, cals, skip_if_unchanged=True) is not None


def test_a_new_calibration_always_re_renders(cals):
    tex = _texture()
    board = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    r = bp.BoardPhotoRenderer()
    r.render_jpeg(board, cals)
    moved = dict(cals)
    c0 = cals[0]
    moved[0] = CameraCalibration(
        camera_matrix=c0.camera_matrix, dist_coeffs=c0.dist_coeffs,
        rvec=np.asarray(c0.rvec) + 1e-4, tvec=c0.tvec, pnp_result=None, landmark_spread_ok=True)
    assert r.render_jpeg(board, moved, skip_if_unchanged=True) is not None
    # and a camera coming or going is a different board too
    assert r.render_jpeg({c: board[c] for c in (0, 1)}, moved, skip_if_unchanged=True) is not None


def test_a_skipped_render_does_not_call_back(cals):
    tex = _texture()
    board = {cam: _camera_view(tex, cal) for cam, cal in cals.items()}
    r = bp.BoardPhotoRenderer()
    got, done = [], threading.Event()
    r.submit(board, cals, lambda jpeg: (got.append(jpeg), done.set()))
    assert done.wait(20)
    done.clear()
    r.submit(_noisy(board, 5), cals, lambda jpeg: (got.append(jpeg), done.set()), skip_if_unchanged=True)
    # the skip is decided on the render thread: wait for it to go idle
    for _ in range(200):
        with r._lock:  # noqa: SLF001
            if not r._busy:  # noqa: SLF001
                break
        threading.Event().wait(0.05)
    assert len(got) == 1 and not done.is_set()


def test_the_server_stores_serves_and_announces_a_board_photo(tmp_path):
    app = create_app(package_root=tmp_path, enable_background_poll=False)
    state = app.state.opendarts_state
    sent = []

    async def _broadcast(msg):
        sent.append(msg)

    state._broadcast = _broadcast  # noqa: SLF001
    client = TestClient(app)
    assert client.get("/api/board/photo").status_code == 404
    assert client.get("/api/state").json()["visit"]["board_photo_version"] is None

    ok, buf = cv2.imencode(".jpg", np.full((20, 20, 3), 90, np.uint8))
    asyncio.run(state._handle_live_event(  # noqa: SLF001
        {"type": "BOARD_PHOTO", "jpeg": buf.tobytes(), "visit_id": "visit_1"}))
    assert [m["type"] for m in sent] == ["BOARD_PHOTO"]
    assert "jpeg" not in sent[0], "the image is fetched, never pushed down the socket"
    version = sent[0]["version"]
    assert client.get("/api/state").json()["visit"]["board_photo_version"] == version
    resp = client.get(f"/api/board/photo?v={version}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == buf.tobytes()

    # garbage is ignored, not stored
    asyncio.run(state._handle_live_event({"type": "BOARD_PHOTO", "jpeg": None}))  # noqa: SLF001
    assert len(sent) == 1


class _Engine:
    def __init__(self, result):
        self._result = result

    def score(self, bg_images, frame_images, calibrations, **_):
        return self._result


def _score_one(tmp_path, monkeypatch, cals, *, visit_index, diagnostics):
    from opendarts.engines.base import EngineResult
    from opendarts.live import capture_daemon

    result = EngineResult(ok=True, sector=20, ring="single_inner", board_xy_mm=(30.0, 80.0),
                          reason="", diagnostics=diagnostics)
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: _Engine(result))
    submitted = []
    monkeypatch.setattr(bp.RENDERER, "submit", lambda frames, c, on_done: submitted.append(sorted(frames)))
    frame = {cam: np.full((H, W, 3), 100, np.uint8) for cam in cals}
    trigger = capture_daemon.ThrowTriggerState(
        state=capture_daemon.ThrowState.READY_TO_CAPTURE, dart_count=visit_index + 1, last_frame=frame)
    events = []
    capture_daemon.handle_ready_to_capture(
        trigger, {cam: f.copy() for cam, f in frame.items()}, cals, tmp_path / "packages", "s1",
        on_event=events.append, visit_id="visit_1", visit_index=visit_index,
        engine_config_store=capture_daemon.EngineConfigStore(primary="Zeus", also_run=()),
        background_save=False,
    )
    thrown = next(e for e in events if e["type"] == "THROW_DETECTED")
    return thrown, submitted


def test_a_scored_dart_carries_its_lean_and_no_dart_renders_a_photo(tmp_path, monkeypatch, cals):
    """The photo is taken after the takeout now, never inside a dart's
    burst -- not even the first dart's, which is where it used to be."""
    true_axis = [0.10, 0.20, 0.97]
    lines = _shaft_lines(cals, [30.0, 80.0, 0.0], true_axis)
    zeus = {"sub_results": {"Talos": {"diagnostics": {"line_px": lines}}}}

    first, asked = _score_one(tmp_path, monkeypatch, cals, visit_index=0, diagnostics=zeus)
    assert _angle_deg(first["dart_axis"], true_axis) < 0.3
    assert "flight_color" not in first, "an unchanged frame has no flight to read"
    assert asked == [], "dart 1 must not render the photo"

    second, asked = _score_one(tmp_path, monkeypatch, cals, visit_index=1, diagnostics=zeus)
    assert "dart_axis" in second and asked == []

    bare, _ = _score_one(tmp_path, monkeypatch, cals, visit_index=2, diagnostics={})
    assert "dart_axis" not in bare, "no lines, no lean -- omitted, never guessed"


def test_start_takes_a_photo_before_the_first_dart():
    """Without this the Photo view shows nothing from boot until someone
    throws. The render must follow the first frames of the session and use
    the calibration already in force."""
    import inspect

    from opendarts.live import capture_daemon

    body = inspect.getsource(capture_daemon.run_capture_loop_body)
    first = body.index("_wait_for_first_frames(")
    photo = body.index("board_photo.RENDERER.submit(", first)
    assert body.index("bg_frames, calibrations", photo) - photo < 200
    assert photo < body.index("while not stop_event.is_set() and not (also_stop", first)
