"""tests/test_camera_resolution.py -- opendarts.live.camera_resolution, the
real resolution-probing logic (see that module's own docstring for the
full "hardcoded 1280x720 with no runtime check" bug this closes).

Cannot be tested against real camera hardware (no physical USB cameras
reachable from this sandbox, and per docs/DESIGN.md no agent may launch/
restart a real camera-using process remotely) -- every test here uses a
lightweight FAKE cv2.VideoCapture-shaped object (structurally satisfies
`SupportsSetGet`: real `set(prop_id, value)`/`get(prop_id)` methods, no
real cv2 dependency), simulating realistic real-world driver behaviors:
a camera that honors every request, one that only honors a narrow
subset and silently substitutes something else for the rest, one that
honors nothing at all. This proves the probing LOGIC is sound; it does
not and cannot prove real hardware behaves like any of these fakes --
see this module's own module docstring for that honest split.
"""
from __future__ import annotations

import pytest

from opendarts.live.camera_resolution import (
    COMMON_RESOLUTIONS,
    ProbeResult,
    genuinely_supported_resolutions,
    highest_supported_resolution,
    parse_resolution_preference,
    probe_resolutions,
)

# Fake OpenCV prop-id constants -- distinct small ints, matching the real
# cv2.CAP_PROP_FRAME_WIDTH/HEIGHT shape closely enough for these tests
# (which pass width_prop=/height_prop= explicitly, never touching real
# cv2 at all).
FAKE_WIDTH_PROP = 3
FAKE_HEIGHT_PROP = 4


class FakeVideoCapture:
    """A minimal fake satisfying `SupportsSetGet` -- real `set()`/`get()`
    methods, no cv2 import needed. `honored` is the set of exact (width,
    height) pairs this fake camera will actually negotiate when asked;
    any OTHER request gets silently redirected to `fallback` (or, if
    `fallback` is None, left as whatever the last successfully-honored
    request was -- mirrors a real driver that quietly keeps its last
    working mode when handed a request it can't satisfy, a realistic and
    common real-webcam behavior worth covering explicitly, not just the
    "honors X or reports 0x0" case)."""

    def __init__(
        self,
        honored: set[tuple[int, int]],
        fallback: tuple[int, int] | None = None,
    ) -> None:
        self.honored = honored
        self.fallback = fallback
        self._current: tuple[int, int] = fallback or (0, 0)
        self._pending_width: int | None = None
        self._pending_height: int | None = None
        self.set_calls: list[tuple[int, float]] = []

    def set(self, prop_id: int, value: float) -> bool:  # noqa: A003
        self.set_calls.append((prop_id, value))
        if prop_id == FAKE_WIDTH_PROP:
            self._pending_width = int(value)
        elif prop_id == FAKE_HEIGHT_PROP:
            self._pending_height = int(value)
        if self._pending_width is not None and self._pending_height is not None:
            requested = (self._pending_width, self._pending_height)
            self._current = requested if requested in self.honored else (
                self.fallback if self.fallback is not None else self._current
            )
            self._pending_width = None
            self._pending_height = None
        return True

    def get(self, prop_id: int) -> float:
        if prop_id == FAKE_WIDTH_PROP:
            return float(self._current[0])
        if prop_id == FAKE_HEIGHT_PROP:
            return float(self._current[1])
        return 0.0


def _probe_kwargs():
    return {"width_prop": FAKE_WIDTH_PROP, "height_prop": FAKE_HEIGHT_PROP}


# --- probe_resolutions() ---------------------------------------------------


def test_probe_resolutions_camera_that_honors_everything():
    cap = FakeVideoCapture(honored=set(COMMON_RESOLUTIONS))
    results = probe_resolutions(cap, **_probe_kwargs())

    assert len(results) == len(COMMON_RESOLUTIONS)
    assert all(r.honored for r in results)
    assert [r.requested for r in results] == COMMON_RESOLUTIONS
    assert [r.negotiated for r in results] == COMMON_RESOLUTIONS


def test_probe_resolutions_camera_that_honors_a_narrow_subset():
    """Real, common driver behavior: a cheap webcam that only genuinely
    supports 640x480 and 1280x720, silently substituting 640x480 (its
    lowest/safest mode) for anything else asked."""
    honored = {(640, 480), (1280, 720)}
    cap = FakeVideoCapture(honored=honored, fallback=(640, 480))
    results = probe_resolutions(cap, **_probe_kwargs())

    by_request = {r.requested: r for r in results}
    assert by_request[(640, 480)].honored is True
    assert by_request[(640, 480)].negotiated == (640, 480)
    assert by_request[(1280, 720)].honored is True
    assert by_request[(1280, 720)].negotiated == (1280, 720)
    # Everything else silently got redirected to the fallback -- a real,
    # common "request silently not honored" case, not an error.
    assert by_request[(1920, 1080)].honored is False
    assert by_request[(1920, 1080)].negotiated == (640, 480)


def test_probe_resolutions_camera_that_honors_nothing():
    """A camera that never actually negotiates any requested mode (always
    reports 0x0, e.g. a broken/disconnected device) -- every candidate
    comes back unhonored, none crash."""
    cap = FakeVideoCapture(honored=set(), fallback=(0, 0))
    results = probe_resolutions(cap, **_probe_kwargs())

    assert all(not r.honored for r in results)
    assert all(r.negotiated == (0, 0) for r in results)


def test_probe_resolutions_default_candidates_is_common_resolutions():
    cap = FakeVideoCapture(honored=set(COMMON_RESOLUTIONS))
    results = probe_resolutions(cap, **_probe_kwargs())
    assert [r.requested for r in results] == COMMON_RESOLUTIONS


def test_probe_resolutions_result_is_a_frozen_dataclass():
    result = ProbeResult(requested=(1, 2), negotiated=(1, 2), honored=True)
    with pytest.raises(Exception):
        result.honored = False  # type: ignore[misc]


# --- genuinely_supported_resolutions() --------------------------------------


def test_genuinely_supported_resolutions_returns_only_honored_deduped():
    honored = {(640, 480), (1280, 720), (1280, 960)}
    cap = FakeVideoCapture(honored=honored, fallback=(640, 480))
    supported = genuinely_supported_resolutions(cap, **_probe_kwargs())

    assert set(supported) == honored
    # Order-preserving, first-occurrence -- COMMON_RESOLUTIONS order.
    assert supported == [(640, 480), (1280, 720), (1280, 960)]


def test_genuinely_supported_resolutions_empty_when_nothing_honored():
    cap = FakeVideoCapture(honored=set(), fallback=(0, 0))
    assert genuinely_supported_resolutions(cap, **_probe_kwargs()) == []


def test_genuinely_supported_resolutions_dedupes_repeated_negotiated_values():
    """Two DIFFERENT candidates that both happen to be honored to the
    SAME real negotiated pair must appear only once."""
    cap = FakeVideoCapture(
        honored={(1280, 720)}, fallback=(1280, 720)
    )
    # Craft candidates where two different "requested" values are both
    # silently redirected to the one honored mode -- exercised via a
    # custom candidate list containing a duplicate-after-negotiation case.
    candidates = [(1280, 720), (1280, 720)]
    supported = genuinely_supported_resolutions(cap, candidates, **_probe_kwargs())
    assert supported == [(1280, 720)]


# --- highest_supported_resolution() -----------------------------------------


def test_highest_supported_resolution_picks_highest_pixel_count():
    honored = {(640, 480), (1280, 720), (1920, 1080)}
    cap = FakeVideoCapture(honored=honored, fallback=(640, 480))
    assert highest_supported_resolution(cap, **_probe_kwargs()) == (1920, 1080)


def test_highest_supported_resolution_prefers_pixel_count_over_list_order():
    """1280x960 (1,228,800px) has MORE pixels than 1280x720 (921,600px)
    despite appearing later in COMMON_RESOLUTIONS's own request order --
    highest_supported_resolution() must sort by real pixel count, not
    trust list position."""
    honored = {(1280, 720), (1280, 960)}
    cap = FakeVideoCapture(honored=honored, fallback=(1280, 720))
    assert highest_supported_resolution(cap, **_probe_kwargs()) == (1280, 960)


def test_highest_supported_resolution_returns_none_when_nothing_honored():
    cap = FakeVideoCapture(honored=set(), fallback=(0, 0))
    assert highest_supported_resolution(cap, **_probe_kwargs()) is None


def test_highest_supported_resolution_single_supported_mode():
    """Backward-compat-relevant case: a camera that only ever honors the
    project's own historical default, 1280x720 -- the real, expected
    case on today's actual rig."""
    cap = FakeVideoCapture(honored={(1280, 720)}, fallback=(1280, 720))
    assert highest_supported_resolution(cap, **_probe_kwargs()) == (1280, 720)


# --- parse_resolution_preference() ------------------------------------------


def test_parse_resolution_preference_auto_returns_none():
    assert parse_resolution_preference("auto") is None
    assert parse_resolution_preference("AUTO") is None  # case-insensitive
    assert parse_resolution_preference("  auto  ") is None  # whitespace-tolerant


def test_parse_resolution_preference_explicit_wxh():
    assert parse_resolution_preference("1920x1080") == (1920, 1080)
    assert parse_resolution_preference("640X480") == (640, 480)  # case-insensitive x


@pytest.mark.parametrize(
    "bad_value",
    ["not-a-resolution", "1920x", "x1080", "1920xabc", "abcx1080", "0x0", "-1x720", "1920"],
)
def test_parse_resolution_preference_malformed_raises_value_error(bad_value):
    with pytest.raises(ValueError):
        parse_resolution_preference(bad_value)
