"""The transparent overlay must composite to the SAME picture as the
baked one.

`draw_calibration_overlay()` paints onto a copy of the camera frame and
returns an opaque image -- a replacement for the live view, not a layer
over it. That is why the dashboard dropped a camera tile off its MJPEG
stream onto a 3-second still the moment that camera calibrated: a
calibrated rig got a slideshow, an uncalibrated one got video.

`draw_calibration_overlay_rgba()` renders the same three layers onto a
transparent canvas so the browser can composite it over the live stream.
The claim that makes it a safe substitute is equivalence, so it is
asserted here against the original rather than left to inspection --
otherwise the two could drift and the only symptom would be a preview
that looks subtly wrong to nobody in particular.
"""

import numpy as np
import pytest

from opendarts.geometry.board_overlay import (
    draw_calibration_overlay,
    draw_calibration_overlay_rgba,
)

from .test_board_overlay import _synthetic_calib


def _composite_over(rgba: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
    """Source-over, exactly what a browser does painting an RGBA <img>
    on top of the video beneath it."""
    a = (rgba[:, :, 3].astype(np.float32) / 255.0)[:, :, None]
    over = rgba[:, :, :3].astype(np.float32)
    under = frame_bgr.astype(np.float32)
    return np.clip(np.rint(over * a + under * (1.0 - a)), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("highlight", [20, None])
@pytest.mark.parametrize(
    "frame_fill",
    [0, 255, 128],
    ids=["over-black", "over-white", "over-mid-grey"],
)
def test_compositing_matches_the_baked_overlay(highlight, frame_fill):
    """Three backgrounds, not one: a wrong alpha can still look right
    over black (where premultiplied and straight colour coincide) and be
    obviously wrong over white."""
    calib = _synthetic_calib()
    frame = np.full((720, 1280, 3), frame_fill, dtype=np.uint8)

    baked = draw_calibration_overlay(frame, calib, highlight_number=highlight)
    layered = _composite_over(
        draw_calibration_overlay_rgba((1280, 720), calib, highlight_number=highlight),
        frame,
    )

    diff = np.abs(baked.astype(np.int16) - layered.astype(np.int16))
    over_one = float((diff > 1).sum()) / diff.size

    # Measured, not guessed (all three backgrounds, both highlight
    # modes): max 5, mean <= 0.024, and 0.017% of subpixels off by more
    # than 1. The residue is antialiasing order at line CROSSINGS -- the
    # baked path draws each wire over the previous result, this one
    # merges the wires into a single mask -- so it is bounded to a few
    # units on a few hundred subpixels out of 2.7M and cannot grow.
    #
    # The bound that actually matters is the one below it: a genuine
    # alpha bug (premultiplied colour served as straight, or the wash
    # alpha dropped) barely shows over black, where the two coincide,
    # and is worth 60-140 units over white. Holding the SAME tight bound
    # across all three backgrounds is what makes this a real equivalence
    # check rather than a smoke test.
    assert diff.max() <= 6, (
        f"layered overlay differs from baked by up to {diff.max()} "
        f"on a {frame_fill} background -- an alpha error, not AA rounding"
    )
    assert diff.mean() <= 0.05, f"mean error {diff.mean():.4f} is too high to be AA rounding"
    assert over_one <= 0.0005, (
        f"{over_one * 100:.4f}% of subpixels differ by more than 1 -- AA "
        "residue is ~0.017%, so this is something structural"
    )


def test_it_is_transparent_where_the_overlay_does_not_draw():
    """The whole point. An overlay opaque in its corners would hide the
    video it is supposed to sit on -- and would look completely normal
    in a screenshot of a static board."""
    calib = _synthetic_calib()
    rgba = draw_calibration_overlay_rgba((1280, 720), calib)
    alpha = rgba[:, :, 3]

    assert alpha[0, 0] == 0 and alpha[0, -1] == 0
    assert alpha[-1, 0] == 0 and alpha[-1, -1] == 0
    assert (alpha == 0).any(), "nothing is transparent -- this is not a layer"
    assert (alpha > 0).any(), "nothing is drawn at all"


def test_it_needs_no_frame_only_a_size():
    """It must not require a camera frame -- that is what lets the
    endpoint serve it without grabbing from the hub and contending with
    the MJPEG stream."""
    import inspect

    params = inspect.signature(draw_calibration_overlay_rgba).parameters
    assert "frame" not in params
    assert list(params)[0] == "size_wh"


def test_the_returned_size_follows_the_request():
    calib = _synthetic_calib()
    rgba = draw_calibration_overlay_rgba((640, 360), calib)
    assert rgba.shape == (360, 640, 4)


@pytest.mark.parametrize("bad", [(0, 720), (1280, 0), (-1, 10)])
def test_a_nonsense_size_is_refused(bad):
    calib = _synthetic_calib()
    with pytest.raises(ValueError):
        draw_calibration_overlay_rgba(bad, calib)


def test_the_canvas_size_is_the_calibrations_pixel_space_not_a_scale():
    """The bug this exists for. draw_calibration_overlay_rgba() projects
    through calib.camera_matrix and does NOT rescale, so a canvas smaller
    than the frame the calibration was solved at does not shrink the
    board -- it draws full-resolution coordinates on a small canvas, and
    the board comes out oversized and shoved toward the bottom-right.

    A 960-wide overlay over a 1280-wide calibration shipped to both rigs
    and was spotted by eye, not by a test: the equivalence test above
    passes the SAME size to both renderers, so its scale factor is always
    1 and it can never see this. This one compares two sizes.
    """
    calib = _synthetic_calib()

    def _drawn_bbox(rgba):
        ys, xs = np.nonzero(rgba[:, :, 3])
        return xs.min(), ys.min(), xs.max(), ys.max()

    native = draw_calibration_overlay_rgba((1280, 720), calib)
    smaller = draw_calibration_overlay_rgba((960, 540), calib)

    nx0, ny0, nx1, ny1 = _drawn_bbox(native)
    sx0, sy0, sx1, sy1 = _drawn_bbox(smaller)

    # Had the geometry scaled with the canvas, the smaller render's bbox
    # would be 0.75x the native one. It is not -- it is the SAME
    # coordinates, merely clipped by a smaller canvas. Pinning that keeps
    # the contract explicit: callers must render at the native size.
    assert (sx0, sy0, sx1, sy1) == (nx0, ny0, nx1, ny1), (
        "the geometry moved with the canvas -- if this now rescales, the "
        "endpoint must stop passing native dimensions (camera_overlay_rgba)"
    )

    # Spell out what "not scaled" costs, so the contract is unmistakable:
    # a 0.75x canvas would have needed a 0.75x bbox, and the difference is
    # entire board-radii of error, not rounding.
    scale = 960 / 1280
    assert abs(sx1 - nx1 * scale) > 20, (
        "bbox is suspiciously close to a scaled one -- this test would no "
        "longer distinguish scaled from unscaled"
    )
