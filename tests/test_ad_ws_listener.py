"""Tests for opendarts/live/ad_ws_listener.py -- the persistent AD
``/api/events`` WebSocket listener that replaced the original (never-
shipped) REST-poll-inline design (see that module's own docstring for
the full "why" -- the REST detection list is not durable, so the event
stream is used).

**Mocking strategy, and why**: an earlier version of this file used a
REAL local ``websockets.sync.server`` bound to 127.0.0.1 -- discovered
live, in THIS sandbox, that binding any TCP server socket (even
127.0.0.1, even an OS-assigned ephemeral port) raises
``PermissionError: [Errno 1] Operation not permitted`` -- the exact same
sandbox restriction ``tests/test_run_product.py``'s own
``TestRealProcessShutdownWithOpenWebsocket`` already documents and
accepts as a known, pre-existing gap (real subprocess + real bound
uvicorn port). Rather than add MORE tests to that same "known sandbox-
limited" bucket, this file instead follows this project's other stated
pattern (``tests/test_ad_ground_truth.py``, ``tests/test_live_server.py``):
mock at the network boundary. Here that boundary is
``websockets.sync.client.connect`` (monkeypatched to return a fake
in-memory connection object implementing exactly the two methods
``AdWsListener._run()`` actually calls -- context-manager protocol and
``recv(timeout=...)``) -- no real socket, in this sandbox or any other,
ever gets bound. This still exercises ALL of this module's own real
logic (message parsing, buffer/dedup/clear semantics, reconnect/backoff,
stop()) -- only the literal TCP handshake is faked, exactly as
``urlopen`` is faked for the REST module's own tests.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

import websockets.exceptions

from opendarts.live.ad_ws_listener import AdWsListener, AdWsThrow, _http_to_ws
from opendarts.live.board_status import (
    BOARD_STATUS_READY,
    BOARD_STATUS_STOPPED,
    BOARD_STATUS_TAKEOUT,
    BOARD_STATUS_UNKNOWN,
)


# ---------------------------------------------------------------------------
# _http_to_ws() -- pure mapping, no connection needed.
# ---------------------------------------------------------------------------

def test_http_to_ws_maps_http_to_ws():
    assert _http_to_ws("http://localhost:3180") == "ws://localhost:3180"


def test_http_to_ws_maps_https_to_wss():
    assert _http_to_ws("https://example.com:443") == "wss://example.com:443"


def test_http_to_ws_passes_through_existing_ws_scheme():
    assert _http_to_ws("ws://already-ws:1234") == "ws://already-ws:1234"


def test_http_to_ws_strips_trailing_slash():
    assert _http_to_ws("http://host:80/") == "ws://host:80"


# ---------------------------------------------------------------------------
# Fake connection / fake `websockets.sync.client.connect` -- see module
# docstring above for why this replaces a real bound socket.
# ---------------------------------------------------------------------------

def _real_shaped_throw(
    *, name="S17", number=17, bed="SingleInner", multiplier=1, x=0.21, y=-0.48,
    method="UnanimousCam", bouncer=False,
):
    """Adapted from real prior art -- see
    opendarts/live/ad_ws_listener.py's own module docstring for exactly
    which functions
    confirm the WS throw object carries coords/segment directly."""
    return {
        "coords": {"x": x, "y": y},
        "segment": {"name": name, "number": number, "bed": bed, "multiplier": multiplier},
        "method": method,
        "bouncer": bouncer,
    }


def _state_message(*, event, num_throws, throws):
    return {"type": "state", "data": {"event": event, "numThrows": num_throws, "throws": throws}}


class _FakeConnection:
    """Stands in for websockets.sync.client.ClientConnection -- implements
    only what AdWsListener._run() actually uses: the context-manager
    protocol and recv(timeout=...). Each connection is handed a scripted
    list of messages (dicts, JSON-encoded here exactly like a real text
    frame would be); once exhausted, recv() either blocks-forever-until-
    stopped (raises TimeoutError repeatedly, simulating AD's real idle
    persistent connection) or raises ConnectionClosedOK (simulating the
    server dropping the connection), per `close_after`.
    """

    def __init__(self, messages, *, close_after: bool, on_recv_exhausted=None, message_delay_s: float = 0.0):
        self._messages = list(messages)
        self._close_after = close_after
        self._on_recv_exhausted = on_recv_exhausted
        self._exhausted_signaled = False
        self._message_delay_s = message_delay_s
        self._served_first = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def recv(self, timeout=None):
        if self._messages:
            # A small delay before every message AFTER the first gives a
            # polling test a real chance to observe the intermediate
            # buffer state between two scripted messages (e.g. "1 throw
            # buffered" between a Throw-detected and a later Takeout-
            # finished) -- without it, both can be processed faster than
            # any poll interval could ever observe, since this fake
            # connection has no real network latency of its own.
            if self._served_first and self._message_delay_s:
                time.sleep(self._message_delay_s)
            self._served_first = True
            return json.dumps(self._messages.pop(0))
        if not self._exhausted_signaled:
            self._exhausted_signaled = True
            if self._on_recv_exhausted is not None:
                self._on_recv_exhausted()
        if self._close_after:
            raise websockets.exceptions.ConnectionClosedOK(None, None)
        # A real idle recv(timeout=...) BLOCKS for the timeout before raising.
        # Raising instantly turned the listener's "nothing yet, loop again"
        # into a busy spin that held the GIL; any thread stuck there slowed
        # every later test in the process. Capped so stop() stays prompt.
        time.sleep(min(timeout if timeout is not None else 0.05, 0.05))
        raise TimeoutError()


class _FakeAdServer:
    """Tracks connect attempts and hands out a new _FakeConnection per
    connect() call, script driven by a caller-supplied factory -- the
    fake-network equivalent of the earlier real-server helper, with the
    exact same test-facing shape (``connection_count``, scripted
    messages) but with zero real socket I/O.
    """

    def __init__(self, connection_factory):
        """connection_factory(attempt_number: int) -> _FakeConnection | Exception instance to raise"""
        self._factory = connection_factory
        self.connection_count = 0
        self._lock = threading.Lock()

    def connect(self, uri, *, open_timeout=None, **kwargs):
        with self._lock:
            self.connection_count += 1
            attempt = self.connection_count
        result = self._factory(attempt)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture(autouse=True)
def no_listener_thread_outlives_its_test():
    """A leaked listener thread does not fail the test that leaked it -- it
    quietly slows everything after it, which is far harder to trace (it cost
    a forty-five-minute suite on 2026-09-22). So each test here must end
    with no listener thread it started still running."""
    before = {t.ident for t in threading.enumerate()}
    yield
    deadline = time.monotonic() + 3.0
    while True:
        leaked = [t.name for t in threading.enumerate()
                  if t.ident not in before and t.name.startswith("opendarts-ad-ws")
                  and t.is_alive()]
        if not leaked or time.monotonic() > deadline:
            break
        time.sleep(0.02)
    assert not leaked, f"listener thread(s) left running: {leaked} -- stop() what you start()"


@pytest.fixture()
def fake_server(monkeypatch):
    """Patches websockets.sync.client.connect (the exact call
    AdWsListener._run() makes, via its own local `import websockets.sync.client
    as ws_client; ws_client.connect(...)`) to a _FakeAdServer's connect().
    Returns a helper to install the server; the listener fixture below
    always starts against whatever this installs."""
    installed: list[_FakeAdServer] = []

    def _install(connection_factory) -> _FakeAdServer:
        server = _FakeAdServer(connection_factory)
        import websockets.sync.client as ws_client

        monkeypatch.setattr(ws_client, "connect", server.connect)
        installed.append(server)
        return server

    yield _install


@pytest.fixture()
def listener():
    listeners: list[AdWsListener] = []

    def _make(base_url="http://fake-ad:3180", **kwargs) -> AdWsListener:
        lst = AdWsListener(base_url, **kwargs)
        listeners.append(lst)
        return lst

    yield _make
    for lst in listeners:
        lst.stop(timeout=3.0)


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def _single_connection_factory(messages, *, close_after: bool = False):
    def factory(attempt):
        return _FakeConnection(messages, close_after=close_after)
    return factory


# ---------------------------------------------------------------------------
# Connect + buffer real-shaped "Throw detected" events.
# ---------------------------------------------------------------------------

def test_listener_connects_and_buffers_a_throw_detected_event(fake_server, listener):
    throw = _real_shaped_throw(name="T20", number=20, bed="Triple", x=0.5, y=0.1)
    fake_server(_single_connection_factory(
        [_state_message(event="Throw detected", num_throws=1, throws=[throw])]
    ))
    lst = listener()
    lst.start()

    assert _wait_until(lambda: len(lst.latest_events()) == 1), "throw never appeared in the buffer"
    assert _wait_until(lst.is_connected)

    events = lst.latest_events()
    assert events[0].n == 1
    assert events[0].throw["segment"]["name"] == "T20"
    # received_at_utc is real wall-clock, close to "now" -- proves this
    # process's own receive timestamp is being stamped, not something
    # from the message payload (which has none, same honest limitation
    # as the REST module).
    assert abs((datetime.now(timezone.utc) - events[0].received_at_utc).total_seconds()) < 5.0


def test_listener_ignores_non_state_message_types(fake_server, listener):
    """Non-"state" frame types -- this listener's _handle_message() filters on
    obj.get("type") != "state" instead (functionally equivalent, simpler
    -- one filter point, not two)."""
    fake_server(_single_connection_factory([
        {"type": "other", "data": {"value": 30}},
        {"type": "another", "data": {"flag": True}},
        _state_message(event="Throw detected", num_throws=1, throws=[_real_shaped_throw()]),
    ]))
    lst = listener()
    lst.start()
    assert _wait_until(lambda: len(lst.latest_events()) == 1)
    # Only the real throw made it into the buffer -- the two non-"state"
    # frames were silently ignored, not mis-parsed into junk entries.
    assert len(lst.latest_events()) == 1


def test_listener_does_not_duplicate_a_throw_with_a_non_increasing_numthrows(fake_server, listener):
    """A throw is only "new" if its
    numThrows is strictly greater than the last one recorded -- a
    duplicate/replayed "Throw detected" with the same n must not double-
    buffer it.

    Synchronization note (fixed after a real flakiness report -- see
    module docstring's own note on this file's mocking strategy): rather
    than a fixed `time.sleep()` to "give the duplicate a chance to land"
    (a race under load -- no sleep duration is ever provably long enough,
    only empirically usually-long-enough), a THIRD, distinguishable
    sentinel message (n=2) is appended after the duplicate. Because a
    single recv loop processes messages strictly in arrival order,
    waiting for the sentinel to appear is deterministic proof the
    duplicate attempt was already processed (successfully or not) by the
    time we assert on the final buffer contents -- no timing assumption
    at all.
    """
    throw = _real_shaped_throw()
    sentinel = _real_shaped_throw(number=2, name="S2")
    fake_server(_single_connection_factory([
        _state_message(event="Throw detected", num_throws=1, throws=[throw]),
        _state_message(event="Throw detected", num_throws=1, throws=[throw]), # duplicate, n not increasing
        _state_message(event="Throw detected", num_throws=2, throws=[sentinel]), # sentinel
    ]))
    lst = listener()
    lst.start()
    assert _wait_until(lambda: any(e.n == 2 for e in lst.latest_events())), "sentinel (n=2) never arrived"
    events = lst.latest_events()
    assert [e.n for e in events] == [1, 2], "the duplicate (repeated n=1) must not have been double-buffered"


def test_listener_clears_buffer_on_takeout_finished(fake_server, listener):
    def factory(attempt):
        return _FakeConnection(
            [
                _state_message(event="Throw detected", num_throws=1, throws=[_real_shaped_throw()]),
                _state_message(event="Takeout finished", num_throws=0, throws=[]),
            ],
            close_after=False,
            message_delay_s=0.3,
        )

    fake_server(factory)
    lst = listener()
    lst.start()
    assert _wait_until(lambda: len(lst.latest_events()) == 1)
    assert _wait_until(lambda: len(lst.latest_events()) == 0), (
        "buffer was not cleared after Takeout finished -- this is the exact real bug this "
        "whole module exists to avoid reintroducing (a stale prior-visit throw silently "
        "surviving past its own takeout)"
    )


def test_listener_buffer_respects_its_own_size_cap(fake_server, listener):
    """Synchronization note (fixed after a real flakiness report -- see
    module docstring's own note on this file's mocking strategy): the
    original version of this test waited for `len(buffer) == 3` and then
    a fixed `time.sleep(0.2)` before asserting the final contents -- a
    REAL latent race, not a false alarm: nothing stops `len() == 3` from
    being true TRANSIENTLY after only messages n=1,2,3 have landed (before
    n=4,5 arrive and trim the buffer down to n=3,4,5), and a fixed sleep
    is never provably long enough under load, only empirically usually-
    enough. Fixed by waiting for a deterministic completion signal
    instead: the LAST scripted message (n=5) becoming the newest buffer
    entry can only be true once every message has been processed, in
    order (a single recv loop), so there is no timing assumption left at
    all -- this either becomes true once processing genuinely finishes,
    or the test times out for a real reason.
    """
    msgs = [
        _state_message(event="Throw detected", num_throws=n, throws=[_real_shaped_throw(number=n)])
        for n in range(1, 6)
    ]
    fake_server(_single_connection_factory(msgs))
    lst = listener(buffer_size=3)
    lst.start()
    assert _wait_until(
        lambda: bool(lst.latest_events()) and lst.latest_events()[-1].n == 5
    ), "final scripted message (n=5) never became the newest buffer entry"
    events = lst.latest_events()
    assert len(events) == 3
    # Oldest-first, most recent 3 kept (n=3,4,5) -- not an arbitrary subset.
    assert [e.n for e in events] == [3, 4, 5]


# ---------------------------------------------------------------------------
# Reconnect / never-crash-on-disconnect.
# ---------------------------------------------------------------------------

def test_listener_reconnects_after_the_server_drops_the_connection(fake_server, listener):
    """First connection gets one throw then closes (ConnectionClosedOK);
    listener must notice, reconnect (its own backoff, kept tiny here so
    the test stays fast), and keep working -- proving this is a durable
    background service, not a one-shot connection that silently dies the
    first time AD's own process restarts or the network blips."""
    def factory(attempt):
        if attempt == 1:
            return _FakeConnection(
                [_state_message(event="Throw detected", num_throws=1, throws=[_real_shaped_throw()])],
                close_after=True,
            )
        return _FakeConnection([], close_after=False) # idle, stays "connected"

    server = fake_server(factory)
    lst = listener(max_backoff_s=0.02)
    lst.start()
    assert _wait_until(lambda: server.connection_count >= 2, timeout_s=5.0), (
        "listener never reconnected after the first connection closed"
    )
    # Still functioning after the reconnect -- is_connected() reflects
    # the (new) live connection state, not a stale "True" from before the
    # drop, and not stuck "disconnected" either.
    assert _wait_until(lst.is_connected, timeout_s=5.0)


def test_listener_start_is_idempotent_and_stop_is_clean(fake_server, listener):
    fake_server(_single_connection_factory(
        [_state_message(event="Throw detected", num_throws=1, throws=[_real_shaped_throw()])]
    ))
    lst = listener()
    lst.start()
    lst.start() # second call must be a harmless no-op, not a second thread
    assert _wait_until(lambda: len(lst.latest_events()) == 1)
    lst.stop(timeout=3.0)
    assert lst.is_connected() is False
    # Calling stop() again (already stopped) must not raise.
    lst.stop(timeout=1.0)


def test_listener_never_raises_out_of_its_background_thread_when_connect_fails(fake_server, listener):
    """connect() itself failing (server unreachable, DNS failure, etc.)
    must be swallowed by the reconnect loop, never surface as an
    unhandled exception that silently kills the background thread with
    no visible error -- the whole point of a persistent listener is that
    it keeps trying."""
    def factory(attempt):
        return ConnectionRefusedError("nobody is listening")

    server = fake_server(factory)
    lst = listener(max_backoff_s=0.02)
    lst.start()
    assert _wait_until(lambda: server.connection_count >= 2, timeout_s=3.0), (
        "listener stopped retrying after a connect failure instead of backing off and trying again"
    )
    assert lst.is_connected() is False
    assert lst.status()["last_error"] is not None


# ---------------------------------------------------------------------------
# match() -- exercises the buffer-scan logic directly, no connection needed.
# ---------------------------------------------------------------------------

def _listener_with_buffer(events) -> AdWsListener:
    """Builds an AdWsListener with a pre-seeded buffer, bypassing the real
    connect/receive machinery entirely -- for testing match()'s own
    selection logic in isolation, same "test the pure logic directly"
    style dev/tests/test_ad_ground_truth_rest.py already uses for
    match_ad_ground_truth()."""
    lst = AdWsListener("http://unused:0")
    lst._buffer = list(events) # noqa: SLF001 -- deliberate direct seed for this test module only
    lst._connected = True
    return lst


def test_match_picks_the_buffered_event_closest_in_time_within_window():
    now = datetime.now(timezone.utc)
    far = AdWsThrow(n=1, throw=_real_shaped_throw(number=5, name="S5"), received_at_utc=now - timedelta(seconds=9))
    near = AdWsThrow(n=2, throw=_real_shaped_throw(number=20, name="T20", bed="Triple"), received_at_utc=now - timedelta(seconds=1))
    lst = _listener_with_buffer([far, near])

    gt = lst.match(now.isoformat(), window_sec=12.0)
    assert gt.matched is True
    assert gt.match_reason == "ok_ws"
    assert gt.sector == "20"
    assert gt.ring == "treble"
    assert gt.staleness_sec == pytest.approx(-1.0, abs=0.5)


def test_expect_ordinal_never_attaches_a_different_darts_answer():
    """THE DART-72 BUG, 2026-09-21.

    We commit a throw ~100-270ms before AD reports the same dart, so when we
    look at AD's buffer the LAST dart of a visit is routinely still missing.
    Nearest-in-time then picked the PREVIOUS dart's answer and recorded it
    as this dart's ground truth -- a disagreement that never happened.

    Real numbers from the package: we captured dart 3 at 18:16:42.185 and
    read the buffer at .344; AD's own answer for dart 3 (S12 -- identical to
    ours) arrived at .403. We recorded AD's dart-2 answer (S11) instead.

    With the ordinal known, "AD has not said yet" must win over "AD said
    something about a different dart".
    """
    now = datetime.now(timezone.utc)
    d1 = AdWsThrow(n=1, throw=_real_shaped_throw(number=14, name="S14"),
                   received_at_utc=now - timedelta(seconds=3.70))
    d2 = AdWsThrow(n=2, throw=_real_shaped_throw(number=11, name="S11"),
                   received_at_utc=now - timedelta(seconds=1.85))
    lst = _listener_with_buffer([d1, d2])

    # Nearest-in-time (no ordinal) reproduces the bug: dart 3 gets S11.
    legacy = lst.match(now.isoformat(), window_sec=12.0)
    assert legacy.matched is True and legacy.sector == "11"

    # With the ordinal, dart 3 is honestly unmatched rather than mislabelled.
    gt = lst.match(now.isoformat(), window_sec=12.0, expect_ordinal=3)
    assert gt.matched is False
    assert gt.match_reason.startswith("ws_awaiting_ordinal")
    assert gt.sector is None, "must not borrow another dart's answer"


def test_expect_ordinal_matches_that_dart_once_ad_reports_it():
    """And once AD's own event for this dart lands, it is used -- which in
    the real case agreed with us exactly (both S12 single_inner)."""
    now = datetime.now(timezone.utc)
    d1 = AdWsThrow(n=1, throw=_real_shaped_throw(number=14, name="S14"),
                   received_at_utc=now - timedelta(seconds=3.70))
    d2 = AdWsThrow(n=2, throw=_real_shaped_throw(number=11, name="S11"),
                   received_at_utc=now - timedelta(seconds=1.85))
    d3 = AdWsThrow(n=3, throw=_real_shaped_throw(number=12, name="S12"),
                   received_at_utc=now + timedelta(seconds=0.218))
    lst = _listener_with_buffer([d1, d2, d3])

    gt = lst.match(now.isoformat(), window_sec=12.0, expect_ordinal=3)
    assert gt.matched is True
    assert gt.match_reason == "ok_ws"
    assert gt.sector == "12"
    assert gt.source_index == 3, "must report the ordinal it actually used"


def test_time_fallback_recovers_a_dart_after_ad_missed_an_earlier_one():
    """AD's numbering drifts the moment it misses one of OUR darts.

    Measured 2026-09-21 on a real rig: AD saw dart 1, never saw dart 2 (it
    went off the board), then reported dart 3 as its n=2. Demanding n=3 threw
    away a perfectly good answer -- and would have thrown away every dart
    after it too, for the rest of the visit.

    Tier 1 (exact n) cannot match here. Tier 2 matches on arrival time,
    which does not depend on AD's bookkeeping, and recovers it.
    """
    now = datetime.now(timezone.utc)                      # our dart 3's capture
    d1 = AdWsThrow(n=1, throw=_real_shaped_throw(number=19, name="S19"),
                   received_at_utc=now - timedelta(seconds=4.19))   # AD's dart 1
    d2 = AdWsThrow(n=2, throw=_real_shaped_throw(number=8, name="S8"),
                   received_at_utc=now + timedelta(seconds=0.337))  # AD's read of OUR dart 3
    lst = _listener_with_buffer([d1, d2])

    # Without the fallback: honest refusal, but the answer is lost.
    strict = lst.match(now.isoformat(), window_sec=12.0, expect_ordinal=3)
    assert strict.matched is False
    assert strict.match_reason.startswith("ws_awaiting_ordinal")

    # With it: the right answer, and the drift is recorded in the reason.
    gt = lst.match(now.isoformat(), window_sec=12.0, expect_ordinal=3,
                   allow_time_fallback=True)
    assert gt.matched is True
    assert gt.sector == "8"
    assert gt.source_index == 2, "matched AD's n=2, which really is our dart 3"
    assert gt.match_reason.startswith("ok_ws_by_time")
    assert "drifted" in gt.match_reason


def test_matching_ordinal_is_refused_when_ads_buffer_is_frozen():
    """A matching dart NUMBER is not evidence of the same dart.

    Measured 2026-09-21: AD wedged in takeout (it enters takeout on a
    visit's last throw and leaves only when it sees the darts removed), so
    it reported nothing further and its buffer kept serving that visit's
    events -- whose n values are 1,2,3, exactly what the next visit asks
    for. Six darts running took answers 20-55s old, the same three repeated
    twice, every one recorded as a clean match.

    Identity must be paired with proximity in time.
    """
    now = datetime.now(timezone.utc)
    frozen = AdWsThrow(n=1, throw=_real_shaped_throw(number=16, name="S16"),
                       received_at_utc=now - timedelta(seconds=25.09))
    lst = _listener_with_buffer([frozen])

    gt = lst.match(now.isoformat(), window_sec=12.0, expect_ordinal=1,
                   allow_time_fallback=True)
    assert gt.matched is False, "a 25s-old answer is not this dart's"
    assert gt.match_reason.startswith("ws_stale_buffer")
    assert "board status" in gt.match_reason
    assert gt.sector is None
    assert gt.staleness_sec == pytest.approx(-25.09, abs=0.5)


def test_fresh_ordinal_inside_the_skew_limit_still_matches():
    """The staleness guard must not reject real matches -- AD lands within
    a few hundred ms of our capture, on either side."""
    now = datetime.now(timezone.utc)
    for delta in (0.34, -0.04, 1.2):
        ev = AdWsThrow(n=2, throw=_real_shaped_throw(number=12, name="S12"),
                       received_at_utc=now + timedelta(seconds=delta))
        gt = _listener_with_buffer([ev]).match(
            now.isoformat(), window_sec=12.0, expect_ordinal=2)
        assert gt.matched is True, f"delta {delta}s should still match"
        assert gt.match_reason == "ok_ws"


def test_time_fallback_still_refuses_a_previous_darts_answer():
    """The fallback must not reopen the original bug.

    Dart 72: AD's dart-2 answer sat in the buffer 1.85s before our capture
    and nearest-in-time attached it to dart 3. The arrival floor excludes
    anything that stale, so even the fallback leaves it unmatched rather
    than borrowing a neighbour's answer.
    """
    now = datetime.now(timezone.utc)
    stale = AdWsThrow(n=2, throw=_real_shaped_throw(number=11, name="S11"),
                      received_at_utc=now - timedelta(seconds=1.85))
    lst = _listener_with_buffer([stale])

    gt = lst.match(now.isoformat(), window_sec=12.0, expect_ordinal=3,
                   allow_time_fallback=True)
    assert gt.matched is False
    assert gt.match_reason.startswith("ws_no_event_near_capture")
    assert gt.sector is None


def test_time_fallback_accepts_ad_arriving_slightly_before_our_capture():
    """AD is not always behind us -- measured 40ms AHEAD of a capture -- so
    the floor is a window on both sides, not "must be later than us"."""
    now = datetime.now(timezone.utc)
    early = AdWsThrow(n=2, throw=_real_shaped_throw(number=19, name="S19"),
                      received_at_utc=now - timedelta(seconds=0.04))
    lst = _listener_with_buffer([early])

    gt = lst.match(now.isoformat(), window_sec=12.0, expect_ordinal=3,
                   allow_time_fallback=True)
    assert gt.matched is True and gt.sector == "19"


def test_exact_ordinal_still_wins_over_the_time_fallback():
    """When AD has seen every dart, tier 1 matches and tier 2 never runs --
    the exact match keeps the plain ok_ws reason."""
    now = datetime.now(timezone.utc)
    d2 = AdWsThrow(n=2, throw=_real_shaped_throw(number=8, name="S8"),
                   received_at_utc=now - timedelta(seconds=0.1))
    d3 = AdWsThrow(n=3, throw=_real_shaped_throw(number=12, name="S12"),
                   received_at_utc=now + timedelta(seconds=0.2))
    lst = _listener_with_buffer([d2, d3])

    gt = lst.match(now.isoformat(), window_sec=12.0, expect_ordinal=3,
                   allow_time_fallback=True)
    assert gt.matched is True
    assert gt.match_reason == "ok_ws"
    assert gt.source_index == 3 and gt.sector == "12"


def test_match_returns_no_match_when_every_buffered_event_is_outside_the_window():
    now = datetime.now(timezone.utc)
    stale = AdWsThrow(n=1, throw=_real_shaped_throw(), received_at_utc=now - timedelta(seconds=30))
    lst = _listener_with_buffer([stale])

    gt = lst.match(now.isoformat(), window_sec=12.0)
    assert gt.matched is False
    assert gt.match_reason.startswith("stale_ws")
    assert gt.staleness_sec is not None


def test_match_with_empty_buffer_reports_not_connected_vs_no_events_honestly():
    lst = AdWsListener("http://unused:0")
    gt_never_connected = lst.match(datetime.now(timezone.utc).isoformat())
    assert gt_never_connected.matched is False
    assert gt_never_connected.match_reason == "ws_not_connected"

    lst._connected = True # noqa: SLF001
    gt_connected_but_empty = lst.match(datetime.now(timezone.utc).isoformat())
    assert gt_connected_but_empty.matched is False
    assert gt_connected_but_empty.match_reason == "ws_no_buffered_events"


def test_match_with_no_capture_time_returns_latest_but_flags_unvalidated():
    now = datetime.now(timezone.utc)
    older = AdWsThrow(n=1, throw=_real_shaped_throw(number=1, name="S1"), received_at_utc=now - timedelta(seconds=5))
    newest = AdWsThrow(n=2, throw=_real_shaped_throw(number=19, name="S19"), received_at_utc=now)
    lst = _listener_with_buffer([older, newest])

    gt = lst.match(None)
    assert gt.matched is True
    assert gt.match_reason == "ok_ws_no_capture_time_to_check"
    assert gt.sector == "19"


def test_match_output_schema_matches_the_rest_module_ad_ground_truth_shape():
    """Never a drift risk between the two paths: this asserts the SAME
    AdGroundTruth type/fields the REST module
    (dev.ad.ad_ground_truth_rest.match_ad_ground_truth) already produces,
    so opendarts.capture.throw_package.save_ad_ground_truth() and everything
    downstream that reads ad_ground_truth.json needs zero changes
    regardless of which path populated it."""
    from opendarts.live.ad_ground_truth import AdGroundTruth as RestAdGroundTruth

    now = datetime.now(timezone.utc)
    ev = AdWsThrow(n=1, throw=_real_shaped_throw(), received_at_utc=now)
    lst = _listener_with_buffer([ev])
    gt = lst.match(now.isoformat())

    assert type(gt) is RestAdGroundTruth
    payload = gt.to_dict()
    # The v2 package schema -- "opendarts_captured_at_utc" -> "captured_at_utc"
    # (2026-08-26) and the "schema" bump ("ad-ground-truth-v1" ->
    # "ad-ground-truth-v2", held back through three rounds and landed
    # 2026-08-27) both apply identically
    # regardless of which path (this WS listener's match(), or the REST
    # module's match_ad_ground_truth()) populated the AdGroundTruth -- see
    # opendarts.live.ad_ground_truth's own AD_GROUND_TRUTH_SCHEMA_V2 module
    # comment.
    from opendarts.live.ad_ground_truth import AD_GROUND_TRUTH_SCHEMA_V2
    assert payload["schema"] == AD_GROUND_TRUTH_SCHEMA_V2 == "ad-ground-truth-v2"
    assert set(payload) == {
        "schema", "matched", "match_reason", "ad_base_url", "fetched_at_utc",
        "captured_at_utc", "staleness_sec", "window_sec", "sector", "ring",
        "tip_xy_mm", "ad_method", "ad_bouncer", "ad_n_cam_detections", "raw_segment",
        "source_index", "n_detections_in_response",
        # operator_marked_wrong/operator_note added 2026-08-12 (the
        # Scoring tab's "mark AD wrong" human-judgment toggle, see
        # opendarts.capture.throw_package.mark_operator_ad_wrong) -- both
        # paths share the same AdGroundTruth dataclass, so this schema
        # check must include them too, not just the fields that existed
        # when this test was first written.
        "operator_marked_wrong", "operator_note",
        # operator_confirmed_* added 2026-08-13. Same reasoning
        # as the two above: one shared dataclass across both the REST and
        # WebSocket paths, so the schema check covers them too.
        "operator_confirmed_source", "operator_confirmed_sector", "operator_confirmed_ring",
    }


# ---------------------------------------------------------------------------
# Board-status indicator light -- 2026-08-14 (Scoring tab AD dot). See
# opendarts/live/board_status.py for the standardized STOPPED/READY/TAKEOUT/
# UNKNOWN vocabulary and _update_board_status()'s own docstring for this
# listener's honest, stated-plainly limitation: only READY and TAKEOUT are
# EVER set here, from the two confirmed real event names ("Takeout
# finished"/"Throw detected") this listener's own throw-buffer logic
# already keys on -- STOPPED is never detected by this listener at all
# (no confirmed live evidence of what, if anything, AD's WS sends when the
# board itself is stopped).
# ---------------------------------------------------------------------------


def test_board_status_defaults_to_unknown_with_no_raw_state_before_any_message():
    lst = AdWsListener("http://unused:0")
    status, raw = lst.board_status()
    assert status == BOARD_STATUS_UNKNOWN
    assert raw is None


def test_update_board_status_logs_every_real_transition(caplog):
    """AD writes no log file of its own, so this listener's INFO line is
    the ONLY durable record that AD changed state. Added 2026-09-10
    after "when did AD go to takeout?" turned out to be unanswerable:
    the transition was computed and pushed to the dashboard light, then
    dropped."""
    import logging

    lst = AdWsListener("http://unused:0")
    caplog.set_level(logging.INFO, logger="opendarts.live.ad_ws_listener")
    lst._update_board_status(
        "Takeout started", 3, {"event": "Takeout started", "status": "Takeout in progress"}
    )

    msgs = [r.getMessage() for r in caplog.records]
    assert any("AD board status" in m for m in msgs), msgs
    line = next(m for m in msgs if "AD board status" in m)
    # Names BOTH sides of the transition and the raw event that caused it
    # -- a line saying only the new value cannot be read as a history.
    assert BOARD_STATUS_UNKNOWN in line and BOARD_STATUS_TAKEOUT in line, line
    assert "Takeout started" in line, line


def test_update_board_status_does_not_log_an_unchanged_status(caplog):
    """Only real transitions. AD re-asserting the same state must stay
    silent or an idle board would fill the log with identical lines."""
    import logging

    lst = AdWsListener("http://unused:0")
    lst._update_board_status("Takeout finished", 0, {"event": "Takeout finished"})
    caplog.clear()
    caplog.set_level(logging.INFO, logger="opendarts.live.ad_ws_listener")
    lst._update_board_status("Takeout finished", 0, {"event": "Takeout finished"})

    assert not [r for r in caplog.records if "AD board status" in r.getMessage()]


def test_update_board_status_takeout_finished_sets_ready():
    """Per _update_board_status()'s own docstring: "Takeout finished" ->
    READY (the board is clear again, ready for the next throw) -- NOT a
    dedicated "just finished a takeout" status of its own. This listener
    never claims to detect the takeout PROCESS or a STOPPED board, only
    these two confirmed transitions."""
    lst = AdWsListener("http://unused:0")
    lst._update_board_status("Takeout finished", 0, {"event": "Takeout finished"})
    assert lst.board_status()[0] == BOARD_STATUS_READY


def test_update_board_status_throw_detected_below_max_darts_is_ready():
    lst = AdWsListener("http://unused:0")
    lst._update_board_status("Throw detected", 2, {"event": "Throw detected", "numThrows": 2})
    assert lst.board_status()[0] == BOARD_STATUS_READY


def test_update_board_status_throw_detected_at_max_darts_is_takeout():
    """n>=3 (this project's own MAX_DARTS_PER_TURN, see
    opendarts/capture/throw_trigger.py) -- the 3rd dart of a turn means a
    takeout is expected next, even though no explicit "takeout started"
    WS event is confirmed to exist on this stream at all."""
    lst = AdWsListener("http://unused:0")
    lst._update_board_status("Throw detected", 3, {"event": "Throw detected", "numThrows": 3})
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT


def test_update_board_status_throw_detected_above_max_darts_is_still_takeout():
    lst = AdWsListener("http://unused:0")
    lst._update_board_status("Throw detected", 5, {"event": "Throw detected", "numThrows": 5})
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT


def test_update_board_status_unrecognized_event_leaves_prior_status_standing():
    """The honest limitation, stated in the module's own docstring: an
    unrecognized/absent event must NOT reset status back to UNKNOWN -- it
    leaves whatever was last classified standing, exactly like
    non-"state" frames this listener's own throw-buffer logic already
    ignores."""
    lst = AdWsListener("http://unused:0")
    lst._update_board_status("Throw detected", 3, {"event": "Throw detected", "numThrows": 3})
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT

    lst._update_board_status("Unknown event", 3, {"event": "Unknown event"})
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT, "an unrecognized event must not reset status"

    lst._update_board_status(None, 0, {})
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT, "event=None must not reset status either"


def test_update_board_status_with_no_prior_message_and_unrecognized_event_stays_unknown():
    lst = AdWsListener("http://unused:0")
    lst._update_board_status("Unknown event", 0, {"event": "Unknown event"})
    assert lst.board_status()[0] == BOARD_STATUS_UNKNOWN


def test_board_status_raw_data_reflects_the_last_recognized_message():
    lst = AdWsListener("http://unused:0")
    data = {"event": "Throw detected", "numThrows": 1, "throws": []}
    lst._update_board_status("Throw detected", 1, data)
    assert lst.board_status() == (BOARD_STATUS_READY, data)


def test_on_status_change_fires_exactly_once_per_real_transition_not_per_message():
    calls: list[str] = []
    lst = AdWsListener("http://unused:0", on_status_change=calls.append)

    lst._update_board_status("Throw detected", 1, {"event": "Throw detected", "numThrows": 1}) # -> READY
    assert calls == [BOARD_STATUS_READY]

    lst._update_board_status("Throw detected", 2, {"event": "Throw detected", "numThrows": 2}) # still READY
    assert calls == [BOARD_STATUS_READY], "must not fire again for a classified value that did not change"

    lst._update_board_status("Throw detected", 3, {"event": "Throw detected", "numThrows": 3}) # -> TAKEOUT
    assert calls == [BOARD_STATUS_READY, BOARD_STATUS_TAKEOUT]

    lst._update_board_status("Takeout finished", 0, {"event": "Takeout finished"}) # -> READY
    assert calls == [BOARD_STATUS_READY, BOARD_STATUS_TAKEOUT, BOARD_STATUS_READY]


def test_on_status_change_none_is_a_safe_no_op():
    """The default (every existing caller before this feature) -- must
    not raise just because no callback was supplied."""
    lst = AdWsListener("http://unused:0")
    lst._update_board_status("Throw detected", 3, {"event": "Throw detected", "numThrows": 3})
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT


def test_handle_message_does_not_deadlock_calling_update_board_status_before_the_lock():
    """Real reentrancy-bug regression guard. `threading.Lock` is not
    reentrant, and `_update_board_status` acquires `self._lock` itself --
    `_handle_message` now calls it BEFORE its own `with self._lock:`
    block specifically to avoid a double-acquire deadlock (see
    `_handle_message`'s own docstring comment on this exact point). Run on
    a background thread with a bounded join so a real regression shows up
    as a clean test FAILURE (thread still alive after the timeout) instead
    of hanging this entire test run forever."""
    lst = AdWsListener("http://unused:0")
    obj = {
        "type": "state",
        "data": {"event": "Throw detected", "numThrows": 1, "throws": [_real_shaped_throw()]},
    }
    thread = threading.Thread(target=lambda: lst._handle_message(obj), daemon=True)
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "_handle_message deadlocked -- the reentrancy fix regressed"
    assert lst.board_status()[0] == BOARD_STATUS_READY
    assert len(lst.latest_events()) == 1


def test_board_status_updates_end_to_end_through_the_real_receive_loop(fake_server, listener):
    """Not just _update_board_status() exercised directly -- this proves
    the real background thread's own _handle_message() path (the exact
    one carrying the reentrancy fix above) actually reaches both
    board_status() and the on_status_change callback for a real scripted
    WS message stream, via a real (fake-network) AdWsListener connect/
    receive loop, not a hand-constructed listener with a pre-seeded
    buffer."""
    calls: list[str] = []
    fake_server(_single_connection_factory([
        _state_message(event="Throw detected", num_throws=1, throws=[_real_shaped_throw()]),
        _state_message(
            event="Throw detected", num_throws=3,
            throws=[_real_shaped_throw(number=3, name="S3")],
        ),
    ]))
    lst = listener(on_status_change=calls.append)
    lst.start()

    assert _wait_until(lambda: lst.board_status()[0] == BOARD_STATUS_TAKEOUT)
    assert calls == [BOARD_STATUS_READY, BOARD_STATUS_TAKEOUT]


# ---------------------------------------------------------------------------
# Durable logging + diagnostics_snapshot() (2026-08-16, "persist real
# diagnostics" task -- see docs/DESIGN.md and this module's own docstring
# addition on _handle_message() for the real incident: two throws in
# one recorded session, where "did AD's event arrive and get
# wiped, or never arrive at all" was unresolvable after the fact).
# ---------------------------------------------------------------------------

def test_handle_message_logs_throw_detected_received_with_segment_and_coords(caplog):
    lst = AdWsListener("http://unused:0")
    throw = _real_shaped_throw(name="T20", number=20, bed="Triple", x=0.5, y=0.1)
    with caplog.at_level("INFO", logger="opendarts.live.ad_ws_listener"):
        lst._handle_message( # noqa: SLF001 -- direct unit test of message handling
            {"type": "state", "data": {"event": "Throw detected", "numThrows": 1, "throws": [throw]}}
        )
    messages = [r.message for r in caplog.records]
    assert any("Throw detected" in m and "received" in m for m in messages)
    assert any("T20" in m for m in messages), "the real segment must appear in the log line"
    assert any("0.5" in m and "0.1" in m for m in messages), "the real coords must appear in the log line"


def test_handle_message_logs_throw_detected_received_even_when_it_is_a_duplicate(caplog):
    """A duplicate (non-increasing numThrows) is still RECEIVED -- the
    real event arriving must be logged regardless of whether it ends up
    re-buffered, since "did the raw event ever arrive" is exactly the
    fact this feature needs to answer independently of buffering state."""
    lst = AdWsListener("http://unused:0")
    throw = _real_shaped_throw()
    lst._handle_message( # noqa: SLF001
        {"type": "state", "data": {"event": "Throw detected", "numThrows": 1, "throws": [throw]}}
    )
    with caplog.at_level("INFO", logger="opendarts.live.ad_ws_listener"):
        caplog.clear()
        lst._handle_message( # noqa: SLF001 -- duplicate, same numThrows=1
            {"type": "state", "data": {"event": "Throw detected", "numThrows": 1, "throws": [throw]}}
        )
    assert any("Throw detected" in r.message and "received" in r.message for r in caplog.records)
    assert len(lst.latest_events()) == 1, "the duplicate itself must still not be re-buffered"


def test_handle_message_logs_takeout_finished_received(caplog):
    lst = AdWsListener("http://unused:0")
    with caplog.at_level("INFO", logger="opendarts.live.ad_ws_listener"):
        lst._handle_message( # noqa: SLF001
            {"type": "state", "data": {"event": "Takeout finished", "numThrows": 0, "throws": []}}
        )
    messages = [r.message for r in caplog.records]
    assert any("Takeout finished" in m and "received" in m for m in messages)


def test_handle_message_logs_buffer_clear_reason_takeout_with_prior_contents(caplog):
    lst = AdWsListener("http://unused:0")
    throw = _real_shaped_throw(name="S17", number=17)
    lst._handle_message( # noqa: SLF001
        {"type": "state", "data": {"event": "Throw detected", "numThrows": 1, "throws": [throw]}}
    )
    with caplog.at_level("INFO", logger="opendarts.live.ad_ws_listener"):
        caplog.clear()
        lst._handle_message( # noqa: SLF001
            {"type": "state", "data": {"event": "Takeout finished", "numThrows": 0, "throws": []}}
        )
    clear_lines = [r.message for r in caplog.records if "buffer cleared" in r.message]
    assert len(clear_lines) == 1
    assert "reason=takeout_finished" in clear_lines[0]
    assert "n_events_cleared=1" in clear_lines[0]
    assert "S17" in clear_lines[0], "what was in the buffer before the clear must be visible in the log"
    assert len(lst.latest_events()) == 0


def test_handle_message_logs_buffer_clear_reason_numthrows_reset_to_zero(caplog):
    """The defensive branch this module already had (numThrows dropping
    to 0 without an explicit 'Takeout finished' event) must be
    distinguishable in the log from a real takeout event -- this is
    exactly the ambiguity a future incident like throws 45/57 needs
    resolved."""
    lst = AdWsListener("http://unused:0")
    lst._handle_message( # noqa: SLF001
        {"type": "state", "data": {"event": "Throw detected", "numThrows": 1, "throws": [_real_shaped_throw()]}}
    )
    with caplog.at_level("INFO", logger="opendarts.live.ad_ws_listener"):
        caplog.clear()
        lst._handle_message( # noqa: SLF001 -- numThrows back to 0, no explicit takeout event
            {"type": "state", "data": {"event": "Manual reset", "numThrows": 0, "throws": []}}
        )
    clear_lines = [r.message for r in caplog.records if "buffer cleared" in r.message]
    assert len(clear_lines) == 1
    assert "reason=numthrows_reset_to_zero" in clear_lines[0]


def test_handle_message_logs_size_eviction_distinct_from_a_full_clear(caplog):
    lst = AdWsListener("http://unused:0", buffer_size=2)
    for n in range(1, 4): # 3 throws into a buffer_size=2 listener
        with caplog.at_level("INFO", logger="opendarts.live.ad_ws_listener"):
            caplog.clear()
            lst._handle_message( # noqa: SLF001
                {
                    "type": "state",
                    "data": {
                        "event": "Throw detected", "numThrows": n,
                        "throws": [_real_shaped_throw(number=n, name=f"S{n}")],
                    },
                }
            )
        if n == 3:
            evict_lines = [r.message for r in caplog.records if "size-evicted" in r.message]
            assert len(evict_lines) == 1
            assert "buffer_size=2" in evict_lines[0]
            assert "S1" in evict_lines[0], "the evicted (oldest) entry must be named, not just counted"
            clear_lines = [r.message for r in caplog.records if "buffer cleared" in r.message]
            assert not clear_lines, "a size eviction must never be logged as a full 'buffer cleared'"
    assert [ev.n for ev in lst.latest_events()] == [2, 3]


# ---------------------------------------------------------------------------
# diagnostics_snapshot()
# ---------------------------------------------------------------------------

def test_diagnostics_snapshot_reflects_never_cleared_empty_buffer():
    lst = AdWsListener("http://unused:0")
    snap = lst.diagnostics_snapshot()
    assert snap["connected"] is False
    assert snap["buffered_events"] == []
    assert snap["last_clear_reason"] is None
    assert snap["last_clear_at_utc"] is None
    # 2026-08-27, the v2 package schema null-vs-[] pass: [] not None, even
    # before any clear has happened -- last_clear_reason (still None,
    # asserted above) is what actually carries the "never cleared"
    # signal; see diagnostics_snapshot()'s own docstring.
    assert snap["last_clear_buffer_summary"] == []
    assert "snapshot_at_utc" in snap


def test_diagnostics_snapshot_reports_real_buffered_events():
    lst = AdWsListener("http://unused:0")
    lst._handle_message( # noqa: SLF001
        {
            "type": "state",
            "data": {
                "event": "Throw detected", "numThrows": 1,
                "throws": [_real_shaped_throw(name="T20", number=20, bed="Triple")],
            },
        }
    )
    snap = lst.diagnostics_snapshot()
    assert len(snap["buffered_events"]) == 1
    assert snap["buffered_events"][0]["n"] == 1
    assert snap["buffered_events"][0]["segment"]["name"] == "T20"
    assert "received_at_utc" in snap["buffered_events"][0]


def test_diagnostics_snapshot_after_a_clear_shows_what_was_wiped_and_why():
    """The exact real question that was unresolvable for throws 45/57:
    is the buffer empty because it was never populated, or because it
    genuinely arrived and was later wiped (and by what)? This must be
    answerable from diagnostics_snapshot() alone, even after the wipe."""
    lst = AdWsListener("http://unused:0")
    lst._handle_message( # noqa: SLF001
        {
            "type": "state",
            "data": {
                "event": "Throw detected", "numThrows": 1,
                "throws": [_real_shaped_throw(name="S17", number=17)],
            },
        }
    )
    lst._handle_message( # noqa: SLF001
        {"type": "state", "data": {"event": "Takeout finished", "numThrows": 0, "throws": []}}
    )
    snap = lst.diagnostics_snapshot()
    assert snap["buffered_events"] == [], "buffer really is empty now"
    assert snap["last_clear_reason"] == "takeout_finished"
    assert snap["last_clear_at_utc"] is not None
    assert len(snap["last_clear_buffer_summary"]) == 1
    assert snap["last_clear_buffer_summary"][0]["segment"]["name"] == "S17", (
        "the throw that arrived-then-was-wiped must still be visible in the snapshot"
    )


def test_diagnostics_snapshot_is_json_serializable():
    """The whole point is embedding this into capture_diagnostics.json --
    must round-trip through json.dumps with no manual conversion."""
    import json

    lst = AdWsListener("http://unused:0")
    lst._handle_message( # noqa: SLF001
        {"type": "state", "data": {"event": "Throw detected", "numThrows": 1, "throws": [_real_shaped_throw()]}}
    )
    lst._handle_message( # noqa: SLF001
        {"type": "state", "data": {"event": "Takeout finished", "numThrows": 0, "throws": []}}
    )
    json.dumps(lst.diagnostics_snapshot()) # must not raise


# ---------------------------------------------------------------------------
# Takeout detected from AD's own `status` field, not just from `event`.
#
# Reported from the live rig 2026-09-08: AD showed yellow and mid-takeout
# while this dashboard's AD light stayed green. Captured from the board in
# exactly that state:
#
#   {"connected":true,"running":true,"status":"Takeout in progress",
#    "event":"Takeout started","numThrows":3, ...}
#
# `status` was never read by this classifier at all -- only `event` and
# `running` -- so that payload fell through to the `running is True` arm
# and was reported READY. The payloads below are the real shapes observed
# from a running board ("Takeout", "Takeout in progress"), not invented
# ones.
# ---------------------------------------------------------------------------


def test_update_board_status_takeout_started_event_sets_takeout():
    lst = AdWsListener("http://unused:0")
    lst._update_board_status(
        "Takeout started", 3,
        {"running": True, "status": "Takeout in progress",
         "event": "Takeout started", "numThrows": 3},
    )
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT


def test_update_board_status_reads_takeout_from_status_without_the_event():
    """Connecting midway through a takeout means the transition event was
    already missed -- the status text is then the only evidence there is."""
    lst = AdWsListener("http://unused:0")
    lst._update_board_status(None, 3, {"running": True, "status": "Takeout"})
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT


def test_running_true_no_longer_masks_a_takeout_status():
    """The exact regression: running=True used to win and paint it READY."""
    lst = AdWsListener("http://unused:0")
    lst._update_board_status(
        "Takeout started", 3,
        {"running": True, "status": "Takeout in progress", "numThrows": 3},
    )
    assert lst.board_status()[0] != BOARD_STATUS_READY


def test_takeout_finished_beats_a_stale_takeout_status_text():
    """At the moment takeout completes, `event` flips to "Takeout
    finished" while `status` can still read "Takeout in progress". The
    event is the newer fact and must win, or the board would stay yellow
    after it is already clear."""
    lst = AdWsListener("http://unused:0")
    lst._update_board_status(
        "Takeout finished", 0,
        {"running": True, "status": "Takeout in progress",
         "event": "Takeout finished", "numThrows": 0},
    )
    assert lst.board_status()[0] == BOARD_STATUS_READY


def test_normal_throw_status_is_still_ready():
    """The live non-takeout shape must be unaffected."""
    lst = AdWsListener("http://unused:0")
    lst._update_board_status(
        "Started", 0,
        {"connected": True, "running": True, "status": "Throw",
         "event": "Started", "numThrows": 0},
    )
    assert lst.board_status()[0] == BOARD_STATUS_READY


# ---------------------------------------------------------------------------
# Board-status hydration on connect. AD's WS emits `state` only on a real
# TRANSITION -- never periodically, never as a snapshot on connect -- so a
# freshly started listener knows nothing until the board next changes.
# Observed 2026-09-08: seconds after a restart AD read status "Throw"
# while this listener still reported unknown and the AD light sat grey.
# ---------------------------------------------------------------------------


def test_hydrate_seeds_board_status_from_a_rest_snapshot(monkeypatch):
    import opendarts.live.ad_ws_listener as mod

    class _Resp:
        def __init__(self, payload): self._payload = payload
        def read(self): return json.dumps(self._payload).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    captured = {}

    def fake_urlopen(url, timeout=None):
        captured["url"] = url
        return _Resp({"connected": True, "running": True, "status": "Throw",
                      "event": "Takeout finished", "numThrows": 0})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    lst = mod.AdWsListener("http://ad.invalid:3180")
    assert lst.board_status()[0] == BOARD_STATUS_UNKNOWN
    lst._hydrate_board_status()
    assert lst.board_status()[0] == BOARD_STATUS_READY
    # It must read AD's state endpoint over http, not the ws events URL.
    assert captured["url"].startswith("http://")
    assert captured["url"].endswith("/api/state")


def test_hydrate_failure_leaves_status_untouched(monkeypatch):
    """Best-effort by design: hydration must never break the listener or
    invent a status. A failure leaves exactly the honest prior state."""
    import opendarts.live.ad_ws_listener as mod

    def boom(url, timeout=None):
        raise OSError("AD unreachable")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    lst = mod.AdWsListener("http://ad.invalid:3180")
    lst._hydrate_board_status()
    assert lst.board_status()[0] == BOARD_STATUS_UNKNOWN


def test_hydrate_picks_up_a_takeout_already_in_progress(monkeypatch):
    """The case that matters most: connecting midway through a takeout.
    The transition event is long gone, so the status text is the only
    evidence -- and without hydration the light would be grey, then wrong."""
    import opendarts.live.ad_ws_listener as mod

    class _Resp:
        def read(self): return json.dumps(
            {"running": True, "status": "Takeout in progress",
             "event": "Takeout started", "numThrows": 3}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr("urllib.request.urlopen", lambda url, timeout=None: _Resp())
    lst = mod.AdWsListener("http://ad.invalid:3180")
    lst._hydrate_board_status()
    assert lst.board_status()[0] == BOARD_STATUS_TAKEOUT


# ---------------------------------------------------------------------------
# Runtime enable/disable and repointing, 2026-09-10.
# Before this the only switch was --no-ad-ground-truth, which decides
# whether the listener is CONSTRUCTED -- so turning AD off meant restarting
# the process. That matters beyond convenience: the throw path fetches AD's
# calibration per throw, and against an unreachable AD that blocks for
# seconds on the scoring path.
# ---------------------------------------------------------------------------


def test_oracle_base_url_is_none_when_disabled():
    """The capture path treats None as "do not contact AD at all", so
    routing the switch through the URL keeps one decision in one place
    instead of a second flag every caller must remember."""
    lst = AdWsListener("http://unused:0")
    assert lst.oracle_base_url() == "http://unused:0"
    lst.set_enabled(False)
    assert lst.oracle_base_url() is None
    assert lst.base_url == "http://unused:0", "the URL itself must survive being disabled"


def test_set_enabled_round_trips_and_is_idempotent():
    lst = AdWsListener("http://unused:0")
    try:
        lst.set_enabled(False)
        lst.set_enabled(False)
        assert lst.is_enabled() is False
        lst.set_enabled(True)
        assert lst.is_enabled() is True
        assert lst.oracle_base_url() == "http://unused:0"
    finally:
        # set_enabled(True) STARTS the listener thread. Left running, it
        # outlived this test, reconnected through whatever fake a later test
        # had installed, and spun on that fake's idle connection for the rest
        # of the run -- turning a one-minute suite into forty-five.
        lst.stop(timeout=3.0)


def test_disabling_reports_the_board_status_as_stopped():
    """The dashboard indicator must not keep showing whatever AD last said
    before it was switched off -- a frozen 'ready' light is worse than no
    light."""
    seen = []
    lst = AdWsListener("http://unused:0", on_status_change=seen.append)
    lst._update_board_status("Throw detected", 1, {"event": "Throw detected"})
    assert lst.board_status()[0] == BOARD_STATUS_READY
    lst.set_enabled(False)
    assert lst.board_status()[0] == BOARD_STATUS_STOPPED
    assert seen[-1] == BOARD_STATUS_STOPPED


def test_set_base_url_updates_the_websocket_url_too():
    """The WS URL is derived, not configured separately -- repointing has
    to move the event subscription as well as the REST reads, or the two
    would talk to different boards."""
    lst = AdWsListener("http://localhost:3180")
    lst.set_enabled(False)
    lst.set_base_url("http://192.0.2.26:3180/")
    assert lst.base_url == "http://192.0.2.26:3180", "trailing slash should be normalised"
    assert lst.ws_url == "ws://192.0.2.26:3180/api/events"


def test_set_base_url_clears_buffered_throws():
    """Buffered throws belong to the PREVIOUS instance. Matching against
    them after repointing would attribute another board's darts to this
    one."""
    lst = AdWsListener("http://localhost:3180")
    lst.set_enabled(False)
    lst._handle_message({"type": "state", "data": {
        "event": "Throw detected", "numThrows": 1,
        "throws": [{"segment": {"name": "T20"}, "coords": {"x": 0.0, "y": 0.0}}],
    }})
    assert lst.latest_events(), "precondition: something is buffered"
    lst.set_base_url("http://192.0.2.26:3180")
    assert lst.latest_events() == []


def test_set_base_url_rejects_empty():
    lst = AdWsListener("http://localhost:3180")
    lst.set_enabled(False)
    with pytest.raises(ValueError):
        lst.set_base_url("   ")


def test_disconnect_during_shutdown_is_not_logged_as_a_fault(caplog):
    """Closing the socket is HOW stop() unblocks a thread parked in
    recv(), so a deliberate shutdown always raises in the receive loop.
    Logging that at warning reported an operator switching Autodarts off
    as a fault -- and promised a retry that could never happen, since the
    loop exits on the next line.

    Tested against the helper directly: the suite's fake connection does
    not die the way a real socket does when closed under a blocked
    recv(), so driving it through the listener cannot reproduce the case
    (an earlier attempt passed with the fix reverted, which is worse than
    no test at all)."""
    import logging
    import threading

    from opendarts.live.ad_ws_listener import _log_disconnect

    stopping = threading.Event()
    stopping.set()
    caplog.set_level(logging.DEBUG, logger="opendarts.live.ad_ws_listener")
    _log_disconnect(stopping, "disconnected", OSError("socket closed"), 1.5)

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "a deliberate stop was reported as a fault"
    )
    assert any("during shutdown" in r.getMessage() for r in caplog.records), (
        "quietened, but it must still be visible at debug"
    )


def test_an_unexpected_disconnect_is_still_a_warning(caplog):
    """The quietening must not swallow a real drop. AD going away while
    we still want it is exactly what the warning exists for."""
    import logging
    import threading

    from opendarts.live.ad_ws_listener import _log_disconnect

    caplog.set_level(logging.DEBUG, logger="opendarts.live.ad_ws_listener")
    _log_disconnect(threading.Event(), "disconnected", OSError("boom"), 2.0)

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    assert "retry in 2.0s" in warnings[0]


# ---------------------------------------------------------------------------
# on_connection_change (2026-09-22). The Config tab's "On but NOT
# connected" note stayed red after enabling AD until a reload, because no
# connect or disconnect was ever announced -- only board-status changes
# were, and those are not connection changes (a disconnect leaves the
# board status wherever AD last put it). See tests/test_ad_connection_push.py
# for the server and dashboard halves.
# ---------------------------------------------------------------------------

def test_on_connection_change_fires_on_connect_drop_reconnect_and_stop(fake_server, listener):
    def factory(attempt):
        if attempt == 1:
            return _FakeConnection([], close_after=True)   # up, then dropped
        return _FakeConnection([], close_after=False)      # up, stays up

    calls: list[bool] = []
    server = fake_server(factory)
    lst = listener(max_backoff_s=0.02, on_connection_change=calls.append)
    lst.start()
    assert _wait_until(lambda: server.connection_count >= 2 and lst.is_connected())
    assert calls == [True, False, True]
    lst.stop()
    assert calls == [True, False, True, False]
    assert lst.is_connected() is False


def test_on_connection_change_is_silent_for_repeated_failed_attempts(fake_server, listener):
    """Each failed reconnect re-reports "not connected"; only a real
    transition is news, or the dashboard would be sent a stream of
    identical pushes while AD is down."""
    calls: list[bool] = []
    server = fake_server(lambda attempt: OSError("refused"))
    lst = listener(max_backoff_s=0.02, on_connection_change=calls.append)
    lst.start()
    assert _wait_until(lambda: server.connection_count >= 2)
    lst.stop()
    assert calls == []


def test_on_connection_change_none_is_a_safe_no_op():
    lst = AdWsListener("http://unused:0")
    lst._set_connected(True)   # noqa: SLF001
    lst._set_connected(False)  # noqa: SLF001
    assert lst.is_connected() is False
