"""opendarts/live/camera_resolution.py -- real resolution probing for an
already-open cv2.VideoCapture, plus the pure "resolve a LiveConfig
resolution preference into a real CameraConfig list" logic.

WHY THIS EXISTS (the real bug, not a hypothetical): opendarts/live/
capture_daemon.py hardcodes `IMAGE_WIDTH = 1280`, `IMAGE_HEIGHT = 720`
and uses them to build every camera's intrinsics matrix principal point
(`_camera_matrix_for()`) and to call `opendarts.pipeline.calibrate_camera()`
-- with ZERO runtime check that the camera actually opened at that exact
resolution. `opendarts/live/local_capture.py`'s `DEFAULT_WIDTH`/
`DEFAULT_HEIGHT` (also 1280/720) are only the REQUESTED mode passed to
`cap.set(cv2.CAP_PROP_FRAME_WIDTH, ...)` -- a real, common webcam-driver
behavior is to silently negotiate something else instead of honoring the
request outright (a request is not a promise). On different hardware
that negotiates a different resolution, nothing here errors -- it
silently produces a wrong principal point and a wrongly-scaled focal
length (`MEASURED_FOCAL_LENGTH_PX` in capture_daemon.py is itself
resolution-dependent, see that constant's own dated comment), with no
warning anywhere. the project's explicit design direction: "find all
acceptable 'common' resolutions and let the user choose as part of
config... our default should be highest available."

WHAT THIS MODULE DOES NOT DO, on purpose: it does not decide when/if a
real `LocalCameraHub` should probe during `open_all()` -- that decision
(and the actual runtime-safety fix, reading back whatever resolution a
camera genuinely negotiated instead of trusting a hardcoded constant for
intrinsics math) lives in `local_capture.py` (opt-in per-camera, see
`CameraConfig.width`/`height` = None meaning "auto probe") and
`capture_daemon.py` (`_negotiated_resolution_for()`, always-on, reads
`LocalCameraHub.status` instead of the hardcoded module constants). This
module is pure, camera-object-agnostic logic: given something that looks
like a `cv2.VideoCapture` (real or a test double), find out what it
genuinely supports.

Cannot be tested against real camera hardware from this dev environment
(no physical USB cameras reachable, and per docs/DESIGN.md no agent may
launch/restart a real camera-using process remotely) -- see
tests/test_camera_resolution.py for the mocked-cv2.VideoCapture coverage
this module's logic gets instead, and this module's own functions'
docstrings for what remains genuinely unverified until run on the rig.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger("opendarts.live.camera_resolution")

# The real, standard-practice list of "common" USB/UVC webcam resolutions
# to probe -- OpenCV's own cv2.VideoCapture has no reliable cross-backend
# way to enumerate every mode a device supports (this varies by backend:
# AVFoundation/V4L2/DirectShow each expose this differently, and OpenCV's
# Python bindings do not wrap any of those enumeration APIs), so the
# practical, standard approach (used by e.g. `v4l2-ctl --list-formats-ext`
# on Linux, or any consumer webcam-utility app) is exactly what
# asked for: try a small known list of modes that real USB webcams
# commonly support, and read back what actually got negotiated for each.
# Ordered ascending by pixel count so "pick the last one that's genuinely
# supported" already means "pick the highest" without a separate sort at
# every call site (callers that DO want highest-of-supported should still
# use highest_supported_resolution() below rather than relying on this
# ordering directly, since a real driver can support a "smaller-looking"
# 4:3 mode like 1280x960 -- MORE pixels than 1280x720 -- ahead of it in
# this list; sorted-by-request-order and sorted-by-pixel-count are not
# the same thing here, which is exactly why highest_supported_resolution()
# re-sorts by actual pixel count rather than trusting list position).
#
# Sourced from the modes that show up across common UVC/USB webcam
# datasheets and driver mode tables (VGA-era 4:3 legacy modes through
# 1080p 16:9 HD modes) -- the same span of modes
# `v4l2-ctl --list-formats-ext` typically reports for a garden-variety
# UVC webcam:
# 640x480 (VGA) -- near-universal legacy minimum
# 800x600 (SVGA)
# 1024x768 (XGA)
# 1280x720 (HD/720p) -- this rig's current hardcoded default
# 1280x960 (SXGA-, 4:3 analog of 720p pixel count)
# 1600x1200 (UXGA)
# 1920x1080 (Full HD/1080p)
COMMON_RESOLUTIONS: list[tuple[int, int]] = [
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1600, 1200),
    (1920, 1080),
]


class SupportsSetGet(Protocol):
    """Structural type for "anything that looks enough like a
    cv2.VideoCapture to probe" -- lets tests pass a lightweight fake
    instead of a real cv2.VideoCapture (which cannot be constructed
    against real hardware in this sandbox) without needing to import cv2
    at all for the pure-logic tests. A real cv2.VideoCapture instance
    satisfies this structurally, no adapter needed."""

    def set(self, prop_id: int, value: float) -> bool: ... # noqa: D102

    def get(self, prop_id: int) -> float: ... # noqa: D102


@dataclass(frozen=True)
class ProbeResult:
    """One candidate resolution's real probe outcome: what was
    requested, what the driver actually reports negotiating afterward,
    and whether those match. `honored=False` is a normal, expected,
    common real-webcam-driver outcome (a request is not a promise) --
    NOT an error."""

    requested: tuple[int, int]
    negotiated: tuple[int, int]
    honored: bool


def probe_resolutions(
    cap: SupportsSetGet,
    candidates: list[tuple[int, int]] | None = None,
    *,
    width_prop: int | None = None,
    height_prop: int | None = None,
) -> list[ProbeResult]:
    """Try to SET each candidate (width, height) on the already-open
    `cap`, then READ BACK what was actually negotiated -- the real,
    standard technique this module exists to implement (OpenCV has no
    reliable cross-backend "list supported modes" call). Returns one
    `ProbeResult` per candidate, in the same order as `candidates`
    (default `COMMON_RESOLUTIONS`) -- callers that only want the
    genuinely-supported subset should use
    `genuinely_supported_resolutions()` below, not filter this list
    themselves, to keep the "honored" definition in exactly one place.

    Deliberately does NOT deduplicate or reorder -- every candidate is
    tried, even if an earlier one already negotiated the same result,
    because a real camera's response to one `set()` call is not assumed
    to predict its response to a different one (an honest reflection of
    "we don't actually know how the driver decides this without asking
    it directly," not an oversight).

    `width_prop`/`height_prop` default to `cv2.CAP_PROP_FRAME_WIDTH`/
    `cv2.CAP_PROP_FRAME_HEIGHT` -- resolved lazily (only imports `cv2`
    when neither is explicitly passed) so this function's pure logic is
    testable with a fake `cap` and no real `cv2` dependency getting in
    the way of what's actually being tested here (the set/get-and-compare
    logic, not cv2 itself).
    """
    if width_prop is None or height_prop is None:
        import cv2 # local import -- see docstring above for why

        if width_prop is None:
            width_prop = cv2.CAP_PROP_FRAME_WIDTH
        if height_prop is None:
            height_prop = cv2.CAP_PROP_FRAME_HEIGHT

    candidates = candidates if candidates is not None else COMMON_RESOLUTIONS
    results: list[ProbeResult] = []
    for width, height in candidates:
        cap.set(width_prop, width)
        cap.set(height_prop, height)
        actual_width = int(cap.get(width_prop) or 0)
        actual_height = int(cap.get(height_prop) or 0)
        negotiated = (actual_width, actual_height)
        honored = negotiated == (width, height)
        results.append(
            ProbeResult(requested=(width, height), negotiated=negotiated, honored=honored)
        )
    return results


def genuinely_supported_resolutions(
    cap: SupportsSetGet,
    candidates: list[tuple[int, int]] | None = None,
    **kwargs,
) -> list[tuple[int, int]]:
    """The real, de-duplicated list of resolutions `cap` actually honors
    when asked -- "genuinely supported" is defined here, precisely, as
    "we asked for exactly this and the driver reported negotiating
    exactly this back" (`ProbeResult.honored`), never a near-match or a
    driver-substituted alternative (that's real, useful information --
    see `probe_resolutions()`'s own full results -- but it is not the
    same claim as "this exact mode is supported", and conflating the two
    would silently trust a resolution nobody actually confirmed).
    De-duplicated, order preserved (first occurrence wins) -- a driver
    that "honors" two different candidates by negotiating the same
    result for both (unusual, but not impossible) should not appear
    twice.
    """
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for result in probe_resolutions(cap, candidates, **kwargs):
        if result.honored and result.negotiated not in seen:
            seen.add(result.negotiated)
            out.append(result.negotiated)
    return out


def highest_supported_resolution(
    cap: SupportsSetGet,
    candidates: list[tuple[int, int]] | None = None,
    **kwargs,
) -> tuple[int, int] | None:
    """The single highest-pixel-count genuinely-supported resolution --
    "highest available", by design's own explicit direction for this
    fix's default. Returns None if NOTHING in `candidates` was honored
    (a real, if unlikely, possible outcome for an unusual camera) --
    callers must handle that case explicitly (fall back to a known-safe
    fixed default) rather than assuming a value always comes back."""
    supported = genuinely_supported_resolutions(cap, candidates, **kwargs)
    if not supported:
        return None
    return max(supported, key=lambda wh: wh[0] * wh[1])


# --- LiveConfig <-> real resolution-preference parsing -------------------
#
# Kept here (not in opendarts/live/config.py) so config.py's own parsing
# stays free of any cv2-adjacent concept -- config.py just needs to know
# "this string is either 'auto' or 'WxH'", not what a probe even is.
# Mirrors LiveConfig's OWN established parsing conventions (see that
# module's `reprojection_targets_px` handling): tolerant, logs a warning
# and treats a malformed entry as absent (never raises), so a hand-edited
# typo in a JSON config file degrades to "no override for this camera"
# rather than crashing the whole process.

AUTO_RESOLUTION = "auto"


def parse_resolution_preference(raw: str) -> tuple[int, int] | None:
    """Parse one `camera_resolutions` JSON value -- either the literal
    string `"auto"` (returns None, meaning "probe for the highest
    genuinely-supported resolution at open time", the field's own
    documented default meaning) or an explicit `"WxH"` string (e.g.
    `"1920x1080"`, returns `(1920, 1080)`, a fixed operator override).
    Malformed input (wrong shape, non-integer parts, non-positive
    dimensions) returns... it does not return -- it RAISES ValueError
    with a real, human-readable reason, so the one caller
    (`opendarts.live.config.load_live_config`) can log a real warning
    naming the actual bad value and degrade to "ignore this entry"
    itself, matching every other field's own tolerant-load posture --
    this function's job is only to parse correctly, not to decide what a
    caller does with a parse failure.
    """
    text = raw.strip().lower()
    if text == AUTO_RESOLUTION:
        return None
    if "x" not in text:
        raise ValueError(f"expected 'auto' or 'WIDTHxHEIGHT', got {raw!r}")
    width_str, _, height_str = text.partition("x")
    try:
        width = int(width_str)
        height = int(height_str)
    except ValueError as exc:
        raise ValueError(f"expected 'auto' or 'WIDTHxHEIGHT', got {raw!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {raw!r}")
    return (width, height)
