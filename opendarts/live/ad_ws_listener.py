"""opendarts/live/ad_ws_listener.py -- a persistent AD ``/api/events``
WebSocket listener, replacing the original (never-shipped) design of
REST-polling ``/api/state/detections`` right after each opendarts capture.

## Why this exists, and why the REST-poll-inline design was wrong

The REST detection list (``GET {ad_base}/api/state/detections``) is not
durable: it only covers the current visit, so throws opendarts had
captured minutes earlier can already be gone from it. A REST poll made
after a opendarts capture is racing the end of the visit; on a slow
poll, a fast operator, or just bad luck, the poll can find nothing, even
though AD genuinely scored the throw. This
is the same problem the original `opendarts/live/ad_ground_truth.py` module
already documented in its own module docstring (see
"The timestamp problem, and this module's matching strategy") but the
REST design could only make the fetch window narrower -- it could never
close the race entirely, because it never listens for anything, only
asks and hopes the throw is still listed.

**Real fix, on a PROVEN pattern**: open
one persistent WebSocket connection to AD's own live event stream and
keep a small rolling buffer of recent committed throws, each stamped
with THIS process's own receive timestamp the instant the message
arrives. There is no race anymore -- the throw is captured into the
buffer in near-real-time, well before any takeout could plausibly
happen, and opendarts's own capture code matches against that buffer
in-memory (no network call at match time at all).

## API shape

``ws://{ad_host}/api/events`` (``_http_to_ws()`` below: http->ws,
https->wss, host/port/path unchanged). Messages are JSON
text frames shaped:

    {"type": "state",
     "data": {"event": "Throw detected" | "Takeout finished" | ...,
               "numThrows": <int>,
               "throws": [<throw>, ...]}}

Confirmed via ``dual_listen_live.py``'s own ``_pump_ad()``/``_on_ad()``:
only ``type == "state"`` matters (other frame types are ignored); ``event == "Throw detected"``
with ``numThrows`` having increased means ``throws[-1]`` is the newest
throw; ``event == "Takeout finished"`` (or ``numThrows == 0`` after a
nonzero count) means the visit was cleared.

**The raw ``throw`` object itself carries ``coords``/``segment``
directly** -- confirmed against the real payloads consumed by
``ad_sector()``/``ad_mm()``, which operate on this exact object with NO
follow-up REST call, for exactly the two fields
(``opendarts.live.ad_ground_truth.segment_to_sector_ring()`` /
``_tip_xy_mm()``) this module needs. A follow-up REST call to
``/api/state/detections`` would add only per-camera detail this
project has no use for (it already has its OWN per-camera
frames/tip-detection) -- this module deliberately does NOT make that
follow-up call, keeping the "no REST call at throw time" property intact.

**Honest limitation, stated plainly**: this "throw object already carries
coords/segment" claim is confirmed by observation (two live
functions using it that way), NOT by capturing an actual WS frame off a
live Autodarts server -- it was built and tested offline. Tests below exercise a REAL
WebSocket client against a REAL local (127.0.0.1, synthetic) WebSocket
server built for the test, proving the wire protocol / reconnect / stop
logic genuinely works end to end -- but the PAYLOAD SHAPE fed to that
test server is still a construction from reading the protocol, not a
captured real message. Worth a real live smoke-test on a rig with
Autodarts running -- flagged in docs/DESIGN.md, not silently assumed
correct.

## Threading model

Uses ``websockets.sync.client`` (a project dependency in its own right
-- requirements.txt names it explicitly, as the WebSocket protocol
implementation ``/api/live`` is served over), NOT asyncio -- this
project's capture loop (``opendarts/live/capture_daemon.py``,
``opendarts/live/run_product.py``) is entirely thread-based outside of
``run_product``'s own uvicorn/FastAPI half, so a plain blocking
``threading.Thread`` running a synchronous recv loop (with a short
``recv(timeout=...)`` so it can notice a stop request promptly, instead
of blocking forever on a single ``recv()`` call) fits that model directly
-- no asyncio-in-a-thread bridging, no ``call_soon_threadsafe``.

Lifecycle mirrors ``opendarts.live.local_capture.LocalCameraHub``'s own
pattern (the closest existing analog: a long-lived external connection
object): a caller constructs an ``AdWsListener``, calls ``start()`` once
at startup, passes the SAME already-started instance into
``run_capture_loop_body()`` (never opened/closed by that function
itself, exactly like ``hub``), and calls ``stop()`` once at shutdown.

## Matching against the buffer -- a real improvement over the REST design

The REST poller (``dev.ad.ad_ground_truth_rest.match_ad_ground_truth()``)
could only
ever compare "AD's single most recent entry" against a staleness window,
because a REST GET has no way to see AD's throw HISTORY with real
per-throw timestamps -- only "whatever is most recent right now". This
module's buffer holds several recent throws, each with THIS process's own
real receive timestamp, so ``match()`` below picks the buffered event
CLOSEST in time to ``opendarts_captured_at_utc`` (within ``window_sec``),
not just blindly "the latest" -- a genuine, structural improvement in
match quality, not just a latency one, made possible by having real
history to search instead of a single point-in-time snapshot.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from opendarts.live.ad_ground_truth import (
    AdGroundTruth,
    DEFAULT_AD_BASE,
    DEFAULT_MATCH_WINDOW_SEC,
    _parse_iso,
    _tip_xy_mm,
    segment_to_sector_ring,
)
from opendarts.live.board_status import (
    BOARD_STATUS_READY,
    BOARD_STATUS_STOPPED,
    BOARD_STATUS_TAKEOUT,
    BOARD_STATUS_UNKNOWN,
)

log = logging.getLogger("opendarts.live.ad_ws_listener")

# How long to wait for the initial WS handshake before treating it as a
# failed connection attempt (same order of magnitude as
# ad_ground_truth.DEFAULT_TIMEOUT_SEC's REST timeout).
DEFAULT_AD_WS_OPEN_TIMEOUT_SEC = 5.0

# How often the recv loop wakes up on its own (via recv()'s own timeout,
# not a real message) to check whether stop() has been called. Small
# enough that stop() returns promptly; not so small it busy-spins --
# matches this project's existing "stop_event.wait()-not-a-busy-loop"
# discipline (see capture_daemon.py's own POLL_INTERVAL_SECONDS comment).
DEFAULT_AD_WS_RECV_TIMEOUT_SEC = 1.0

# Reconnect backoff -- starts at 1s, x1.5 each failure, capped. AD being briefly unreachable (rig
# restart, network blip) must not turn into a hot retry loop.
DEFAULT_AD_WS_MAX_BACKOFF_SEC = 10.0

# How many recent AD throws to keep in memory. A real turn is at most
# MAX_DARTS_PER_TURN (3, see opendarts/capture/throw_trigger.py) throws
# before a takeout clears the buffer anyway -- 8 is a generous cushion
# above that (covers a couple of turns' worth even if a clear event is
# somehow missed) without holding unbounded history.
DEFAULT_AD_WS_BUFFER_SIZE = 8

#: How far BEFORE our own capture an AD event may have arrived and still be
#: this dart's answer, used by match()'s time fallback (tier 2).
#:
#: AD is not reliably behind us: measured 2026-09-21 it landed 218ms after
#: one capture and 40ms BEFORE another. So this is a small window on both
#: sides of our capture, not a "must be later than us" rule. What it has to
#: exclude is the PREVIOUS dart's answer, which arrives seconds earlier --
#: that is the whole original bug, where an answer 1.85s stale was attached
#: to the next dart. Well under the physical gap between darts (~2s), so it
#: cannot reach back into the previous throw.
AD_LEAD_EPSILON_SEC = 0.5

#: How far an exact-ordinal (tier 1) match may sit from our capture before
#: it is treated as a frozen buffer rather than this dart's answer.
#:
#: A matching dart NUMBER is not on its own evidence of the same dart: if AD
#: stops reporting -- most often wedged in takeout, which it enters on a
#: visit's last throw and leaves only when it sees the darts removed -- its
#: buffer keeps serving that visit's events, and their n values are 1,2,3,
#: precisely what the next visit asks for. Measured 2026-09-21: six darts in
#: a row took answers 20-55s old, the same three repeated twice, each
#: recorded as a clean match.
#:
#: Real matches land within a few hundred ms of our capture on either side
#: (+0.34s and -0.04s observed), so 2s is ~6x the worst seen and still an
#: order of magnitude tighter than the legacy 12s nearest-in-time window
#: that let this through.
AD_ORDINAL_MAX_SKEW_SEC = 2.0


def _http_to_ws(url: str) -> str:
    """``http(s)://`` -> ``ws(s)://`` -- scheme swap only; host, port and
    path are left untouched."""
    u = url.rstrip("/")
    if u.startswith("https://"):
        return "wss://" + u[len("https://"):]
    if u.startswith("http://"):
        return "ws://" + u[len("http://"):]
    if u.startswith("ws://") or u.startswith("wss://"):
        return u
    return "ws://" + u


def _summarize_event(ev: "AdWsThrow") -> dict[str, Any]:
    """Small, JSON-serializable summary of one buffered throw -- shared by
    diagnostics_snapshot() and the buffer-clear/size-eviction log lines
    below, so the log and the persisted diagnostics can never describe
    the same event two different ways."""
    return {
        "n": ev.n,
        "segment": ev.throw.get("segment"),
        "received_at_utc": ev.received_at_utc.isoformat(),
    }


@dataclass
class AdWsThrow:
    """One AD throw as observed over the WS event stream -- the raw
    ``throw`` dict (has ``coords``/``segment``/``method``/``bouncer``,
    same shape ``opendarts.live.ad_ground_truth.segment_to_sector_ring()``/
    ``_tip_xy_mm()`` already parse) plus THIS process's own receive
    timestamp, which stands in for "when AD detected this throw" far more
    precisely than the REST module's ``fetched_at`` ever could (that was
    "whenever we happened to poll next"; this is "the instant the push
    arrived")."""

    n: int
    throw: dict[str, Any]
    received_at_utc: datetime


def _log_disconnect(stop_event: threading.Event, what: str,
                    exc: BaseException, backoff: float) -> None:
    """Report a lost connection at the right volume.

    Closing the socket is HOW stop() unblocks a thread parked in recv(),
    so a deliberate shutdown always raises in the receive loop. Logging
    that at warning reported an operator switching Autodarts off as a
    fault -- and promised a retry that could never happen, because the
    loop exits on the very next line.

    Module-level rather than a closure so it can be tested directly: the
    fake connection the suite uses does not die the way a real socket
    does when it is closed under a blocked recv(), so the interesting
    case cannot be reproduced through the listener itself.
    """
    if stop_event.is_set():
        log.debug("AD WS listener %s during shutdown: %s", what, exc)
        return
    log.warning("AD WS listener %s: %s -- retry in %.1fs", what, exc, backoff)


class AdWsListener:
    """Owns one persistent WS connection to AD's ``/api/events`` and a
    small in-memory buffer of recent committed throws. See module
    docstring for the full design/threading rationale.

    Never touches disk or any other opendarts module -- ``match()`` returns
    a plain ``AdGroundTruth`` (the exact same type/schema the REST module
    produces), so callers persist it via
    ``opendarts.capture.throw_package.save_ad_ground_truth()`` exactly as
    they already would for a REST-obtained result. This keeps the on-disk
    ``ad_ground_truth.json`` schema, and everything downstream that reads
    it (opendarts/live/server.py's dashboard, replay tooling), completely
    unchanged by this module's existence.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_AD_BASE,
        *,
        buffer_size: int = DEFAULT_AD_WS_BUFFER_SIZE,
        open_timeout_s: float = DEFAULT_AD_WS_OPEN_TIMEOUT_SEC,
        recv_timeout_s: float = DEFAULT_AD_WS_RECV_TIMEOUT_SEC,
        max_backoff_s: float = DEFAULT_AD_WS_MAX_BACKOFF_SEC,
        on_status_change: "Callable[[str], None] | None" = None,
        on_connection_change: "Callable[[bool], None] | None" = None,
    ) -> None:
        self.base_url = base_url
        self.ws_url = _http_to_ws(base_url) + "/api/events"
        # Runtime on/off. Until 2026-09-10 the only way to disable AD was
        # `--no-ad-ground-truth`, which decides whether this object gets
        # CONSTRUCTED -- so turning AD off meant restarting the process.
        self._enabled = True
        self._buffer_size = buffer_size
        self._open_timeout_s = open_timeout_s
        self._recv_timeout_s = recv_timeout_s
        self._max_backoff_s = max_backoff_s
        # Called from THIS listener's background thread, never the
        # asyncio loop -- pushes board-status transitions to whatever
        # sink the caller wired (see opendarts/live/run_product.py's
        # _make_board_status_pusher()).
        self._on_status_change = on_status_change
        # Called on every real CONNECTED <-> NOT-CONNECTED transition of
        # the WebSocket (added 2026-09-22). Deliberately a second callback
        # rather than a value squeezed into on_status_change: board status
        # (ready/takeout/stopped...) and connection are different facts.
        # A board status transition does not happen on every connect (the
        # hydrated status can equal the last one) and never happens on a
        # disconnect at all -- the status is left at whatever AD last
        # said -- so inferring "connected" from a status push would be
        # wrong in both directions. Before this existed nothing told the
        # dashboard the socket had come up, and the Config tab's "On but
        # NOT connected" note sat red until a page reload. Same threading
        # contract as on_status_change: may be called from this
        # listener's thread OR from whichever thread called stop(), so a
        # sink must be thread-safe (run_product's is a queue.put).
        self._on_connection_change = on_connection_change

        self._lock = threading.Lock()
        self._buffer: list[AdWsThrow] = []
        self._ad_num = 0
        self._connected = False
        self._last_error: str | None = None
        self._connect_count = 0
        self._board_status: str = BOARD_STATUS_UNKNOWN
        #: When the board entered its current status (None until the first
        #: real transition). Drives the stuck-in-takeout warning.
        self._board_status_since: "datetime | None" = None
        self._board_state_raw: dict[str, Any] | None = None
        # Diagnostics-only bookkeeping (2026-08-16, "persist real
        # diagnostics" task -- see docs/DESIGN.md): survives a buffer clear so
        # a caller reading diagnostics_snapshot() AFTER a clear can still
        # answer "was the buffer empty because it was cleared (and by
        # what), or because it was never populated" -- exactly the
        # question two throws in one recorded session left
        # unresolvable. None until the first clear this process lifetime.
        self._last_clear_reason: str | None = None
        self._last_clear_at_utc: datetime | None = None
        self._last_clear_buffer_summary: list[dict[str, Any]] | None = None

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        """Starts the background connect/receive thread. Idempotent --
        calling twice without an intervening stop() is a no-op (mirrors
        LocalCameraHub.open_all()'s own "safe to call defensively"
        posture)."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="opendarts-ad-ws-listener", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Signals the background thread to stop and joins it (bounded --
        the thread is daemon=True so it can never block process exit even
        if this bound is hit, same posture as run_product.py's own
        capture-thread join). Safe to call even if start() was never
        called or stop() already ran."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning(
                    "AD WS listener thread did not stop within %.1fs -- "
                    "proceeding anyway (daemon thread, cannot block process exit)",
                    timeout,
                )
            self._thread = None
        self._set_connected(False)

    # -- runtime configuration -------------------------------------------

    def oracle_base_url(self) -> "str | None":
        """The URL callers should use to talk to AD for THIS throw, or None
        when AD is switched off.

        Deliberately not just `self.base_url`: the capture path treats
        None as "do not contact AD at all", and routing the on/off switch
        through the URL keeps one decision in one place rather than adding
        a second flag every caller would have to remember to check.
        """
        with self._lock:
            return self.base_url if self._enabled else None

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """Turn AD on or off live, starting or stopping the background
        connection to match. Idempotent."""
        enabled = bool(enabled)
        with self._lock:
            if enabled == self._enabled:
                return
            self._enabled = enabled
        if enabled:
            log.info("AD enabled -- connecting to %s", self.ws_url)
            self.start()
        else:
            # Stop the socket too, not just the flag. A listener left
            # connected while disabled keeps reconnect-looping against a
            # host the operator has said not to use, and keeps pushing
            # board-status changes to a dashboard indicator that should
            # read as "off".
            log.info("AD disabled -- disconnecting")
            self.stop()
            self._update_board_status_off()

    def set_base_url(self, base_url: str) -> None:
        """Point at a different AD instance live. Reconnects if enabled."""
        base_url = base_url.strip().rstrip("/")
        if not base_url:
            raise ValueError("AD base URL cannot be empty")
        with self._lock:
            if base_url == self.base_url:
                return
            self.base_url = base_url
            self.ws_url = _http_to_ws(base_url) + "/api/events"
            enabled = self._enabled
            # Buffered throws belong to the PREVIOUS instance -- matching a
            # throw against them after repointing would attribute another
            # board's darts to this one. Cleared under the same lock that
            # swapped the URL, so no window exists where the new URL is
            # live but the old board's throws are still matchable.
            self._buffer.clear()
        log.info("AD base URL changed to %s", self.base_url)
        if enabled:
            self.stop()
            self.start()

    def _update_board_status_off(self) -> None:
        """Report the indicator as stopped when AD is switched off, rather
        than leaving whatever it last saw frozen on screen."""
        callback = None
        with self._lock:
            changed = self._board_status != BOARD_STATUS_STOPPED
            previous = self._board_status
            self._board_status = BOARD_STATUS_STOPPED
            if changed and self._on_status_change is not None:
                callback = self._on_status_change
        if changed:
            log.info("AD board status: %s -> %s (disabled by operator)",
                     previous, BOARD_STATUS_STOPPED)
        if callback is not None:
            callback(BOARD_STATUS_STOPPED)

    # -- status ----------------------------------------------------------

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def status(self) -> dict[str, Any]:
        """Cheap snapshot for logging/diagnostics -- not currently wired
        into any HTTP endpoint (server.py is out of scope for this
        change), but kept small and side-effect-free enough that a future
        caller can surface it without needing to touch this module."""
        with self._lock:
            return {
                "ws_url": self.ws_url,
                "connected": self._connected,
                "last_error": self._last_error,
                "buffered_events": len(self._buffer),
                "connect_count": self._connect_count,
            }

    def latest_events(self) -> list[AdWsThrow]:
        """A snapshot copy of the current buffer, oldest-first -- never
        the live list itself, so a caller iterating it can never race a
        concurrent append from the receive thread."""
        with self._lock:
            return list(self._buffer)

    def diagnostics_snapshot(self) -> dict[str, Any]:
        """Real, serializable diagnostic snapshot of this listener's
        current state -- built for `opendarts.live.capture_daemon.
        handle_ready_to_capture()` to embed into a throw package's
        `capture_diagnostics.json` (2026-08-16, "persist real
        diagnostics" task, see docs/DESIGN.md) so a FUTURE anomalous throw
        (like two throws in one recorded session) can be
        diagnosed from the saved package alone, without needing a live
        process log or code archaeology.

        Captured SYNCHRONOUSLY at whatever moment the caller calls this
        (`handle_ready_to_capture()` calls it right at throw-capture
        time, before the background AD-match attempt even starts) --
        this is a deliberate, honestly-stated limitation: it answers
        "what did this listener's buffer look like the instant the dart
        was captured," which is the earliest and most decision-relevant
        moment available without adding real synchronization between the
        capture path and the background match thread. `match()` itself
        performs NO buffer mutation (it only reads under the lock -- see
        that method's own docstring), so this snapshot is never
        invalidated by a subsequent match attempt; only a NEW WS event
        (a real "Throw detected"/"Takeout finished" arriving after this
        was taken) can make it stale, which is itself real, useful
        information (the package's own `captured_at_utc` vs this
        snapshot's `snapshot_at_utc` bounds exactly how stale it could be).

        Returns, all JSON-serializable:
          "connected": bool -- is the WS connection currently up.
          "snapshot_at_utc": ISO string, when THIS call was made.
          "buffered_events": [{"n", "segment", "received_at_utc"}, ...]
            oldest-first, exactly what's in the buffer right now --
            answers "did AD's event genuinely never arrive" (empty AND
            last_clear_reason is None) vs "it arrived and was later
            wiped" (empty AND last_clear_reason is set) vs "it's sitting
            right here" (non-empty).
          "last_clear_reason": "takeout_finished" | "numthrows_reset_to_zero"
            | None -- WHY the buffer was last fully cleared (None if
            never cleared this process lifetime). This is the single
            fact that was UNRESOLVABLE during the real 2026-08-16
            investigation this feature responds to.
          "last_clear_at_utc": ISO string | None -- when that clear
            happened.
          "last_clear_buffer_summary": same shape as "buffered_events" --
            what was IN the buffer immediately before that last clear (so
            a genuinely-arrived-then-wiped throw is still visible in this
            snapshot even after the wipe). ``[]``, never ``None``, when no
            clear has happened yet this process lifetime (2026-08-27,
            the v2 package schema null-vs-[] pass) -- this loses no
            information because "never cleared" is already, separately,
            exactly what ``last_clear_reason: None`` means; a real clear
            of an empty buffer ALSO produces ``[]`` here (see the
            ``_run()`` clear branch below: ``cleared_summary`` is always a
            real list, built from ``self._buffer`` at clear time, however
            many items -- 0 included), so ``[]`` cannot be confused for
            "never cleared" on its own -- callers needing that must (and
            already do) read ``last_clear_reason`` instead.
        """
        with self._lock:
            return {
                "connected": self._connected,
                "snapshot_at_utc": datetime.now(timezone.utc).isoformat(),
                "buffered_events": [_summarize_event(ev) for ev in self._buffer],
                "last_clear_reason": self._last_clear_reason,
                "last_clear_at_utc": (
                    self._last_clear_at_utc.isoformat() if self._last_clear_at_utc else None
                ),
                "last_clear_buffer_summary": (
                    self._last_clear_buffer_summary
                    if self._last_clear_buffer_summary is not None
                    else []
                ),
            }

    def _set_connected(self, value: bool, error: str | None = None) -> None:
        with self._lock:
            changed = value != self._connected
            self._connected = value
            if error is not None:
                self._last_error = error
            callback = self._on_connection_change if changed else None
        # Outside the lock, like _update_board_status(): the sink may block
        # and nothing reading is_connected() should wait on it. Only on a
        # real transition -- the reconnect loop re-reports "not connected"
        # on every failed attempt, and each of those is not news.
        if callback is not None:
            callback(value)

    def board_status(self) -> tuple[str, "dict[str, Any] | None"]:
        """(classified BOARD_STATUS_* name, raw last-seen `data` dict or
        None) -- the Scoring tab's AD indicator light, 2026-08-14. Pure
        in-memory read, no network call."""
        with self._lock:
            return self._board_status, self._board_state_raw

    def board_status_age_sec(self) -> "float | None":
        """Seconds since the board last CHANGED status, or None if it has
        not been seen to change yet.

        Exists so "takeout" can be told apart from "stuck in takeout".
        AD enters takeout on a visit's last throw and leaves it when it
        sees the darts removed; if it never sees that, it stays there,
        stops reporting throws, and its buffer keeps serving the finished
        visit -- which silently attached 20-55s-old answers to six live
        darts on 2026-09-21 before the staleness guard existed. A few
        seconds here is the normal path; a minute is a wedged board."""
        with self._lock:
            since = self._board_status_since
        if since is None:
            return None
        return (datetime.now(timezone.utc) - since).total_seconds()

    def _hydrate_board_status(self) -> None:
        """One REST read of ``/api/state`` immediately after connecting.

        AD's WS emits a ``state`` frame only on a real TRANSITION, never
        periodically and never as a snapshot on connect. So a listener
        that has just started knows nothing until the board next changes
        -- which, on an idle board between turns, can be a very long
        time. Observed directly 2026-09-08: seconds after a restart AD
        read ``status: "Throw"`` while this listener still reported
        ``unknown``, and the dashboard's AD light sat grey.

        Deliberately best-effort and never fatal: a failure here leaves
        the status exactly as it was (``unknown`` on a fresh listener),
        which is the same honest state as before this call existed. The
        WS remains the real source; this only removes the dead window in
        front of the first transition.
        """
        try:
            from urllib.request import urlopen

            state_url = self.ws_url.replace("ws://", "http://", 1).replace(
                "wss://", "https://", 1
            ).replace("/api/events", "/api/state", 1)
            with urlopen(state_url, timeout=3) as resp:
                data = json.loads(resp.read())
        except Exception as exc: # noqa: BLE001 -- hydration must never break the listener
            log.debug("AD board-status hydration skipped: %s", exc)
            return
        if not isinstance(data, dict):
            return
        try:
            n = int(data.get("numThrows") or 0)
        except (TypeError, ValueError):
            n = 0
        self._update_board_status(data.get("event"), n, data)
        log.info("AD board status hydrated on connect: %s", self.board_status()[0])

    def _update_board_status(self, event: "str | None", n: int, data: dict[str, Any]) -> None:
        """**Live-confirmed fix, 2026-08-14** (superseding the original
        "only READY and TAKEOUT are ever set" honest-limitation design):
        live-tested against the real rig and caught the gap: the board
        was running while this listener stayed stuck on
        BOARD_STATUS_UNKNOWN the entire time, because the board's real event stream includes event
        values this classifier never recognized ("Manual reset",
        confirmed via a live `GET {ad_base}/api/state` snapshot taken
        DURING that exact stuck-grey session:
        `{"connected":true,"running":true,"status":"Throw",
        "event":"Manual reset","numThrows":0}`) -- and a live WS capture
        (55s+ of real idle-rig `/api/events` traffic) showed "state"
        frames only fire on a real transition (not periodically), so a
        never-before-seen event string could leave a fresh connection
        stuck on UNKNOWN indefinitely.

        Fix: `data.get("running")` (present on that live-captured
        snapshot, and -- by direct analogy, since `event`/`numThrows`
        are already confirmed to appear on both the REST and WS
        surfaces from the exact same underlying state object -- inferred
        to appear on WS "state" data too, not a blind guess) is now
        checked FIRST for the unambiguous STOPPED/no-specific-event-yet
        cases, keeping the original event-name-based TAKEOUT/READY
        refinement (still the more precise signal DURING an active
        visit) ahead of it. "Manual reset" is recognized directly
        (real, observed event -- always READY, board cleared and
        waiting) as belt-and-suspenders in case a future payload lacks
        `running` for some reason. Only when NEITHER `running` nor a
        recognized event says anything does this leave the PRIOR status
        standing (or BOARD_STATUS_UNKNOWN if none has ever arrived) --
        now a genuinely rare case instead of the common one.
        TAKEOUT is still inferred at n>=3 (this project's own
        MAX_DARTS_PER_TURN) rather than waiting for an explicit "takeout
        started" WS event, since no such event is confirmed to exist on
        this stream at all.
        """
        running = data.get("running")
        status_text = str(data.get("status") or "").lower()
        if running is False:
            new_status = BOARD_STATUS_STOPPED
        elif event == "Takeout finished":
            # Checked BEFORE the takeout matches below: at the moment
            # takeout completes, `event` has already flipped to
            # "Takeout finished" while `status` may still read
            # "Takeout in progress". The event is the newer fact.
            new_status = BOARD_STATUS_READY
        elif event == "Takeout started" or "takeout" in status_text:
            # 2026-09-08, reported from the live rig: the board was
            # mid-takeout while this dashboard's Autodarts light stayed
            # green. Confirmed against the board in that exact state --
            # `{"running":true,"status":"Takeout in progress",
            #   "event":"Takeout started","numThrows":3}` -- which fell
            # all the way through to the `running is True` arm and was
            # reported as READY.
            #
            # Root cause: `status` was never read at all. Only `event`
            # and `running` were, so every status AD expresses ONLY
            # through that field was invisible here. Both are matched
            # now: the event for the transition, and the status text for
            # the state it leaves behind -- the board reports "Takeout"
            # and "Takeout in progress" on its own /api/state, so a takeout is
            # caught whether this listener sees the transition or
            # connects midway through one.
            new_status = BOARD_STATUS_TAKEOUT
        elif event == "Throw detected":
            new_status = BOARD_STATUS_TAKEOUT if n >= 3 else BOARD_STATUS_READY
        elif event == "Manual reset":
            new_status = BOARD_STATUS_READY
        elif running is True:
            new_status = BOARD_STATUS_READY
        else:
            return
        callback = None
        with self._lock:
            changed = new_status != self._board_status
            previous = self._board_status
            self._board_status = new_status
            # WHEN it entered this status, so "how long has it been here"
            # is answerable. Takeout is normal for a few seconds and a
            # problem after a minute; without a start time the indicator
            # cannot tell those apart. Only moved on a real transition --
            # a repeated status must not keep resetting the clock, or a
            # stuck board looks freshly arrived forever.
            if changed:
                self._board_status_since = datetime.now(timezone.utc)
            self._board_state_raw = data
            if changed and self._on_status_change is not None:
                callback = self._on_status_change
        # Logged, not just pushed. Until 2026-09-10 `changed` existed
        # solely to repaint the dashboard's AD indicator, so AD's state
        # was visible live and left no trace: asking "when did AD go to
        # takeout?" after the fact was unanswerable from this side. One
        # INFO line per real transition is
        # cheap -- a few per visit -- and makes AD's history readable in
        # run_product.log alongside our own throws. Outside the lock: a
        # log handler can block on I/O and no other thread should wait
        # on that to read a status.
        if changed:
            log.info(
                "AD board status: %s -> %s (event=%r status=%r running=%r n=%s)",
                previous, new_status, event, data.get("status"), running, n,
            )
        if callback is not None:
            callback(new_status)

    # -- connect / receive loop ------------------------------------------

    def _run(self) -> None:
        # Imported here, not at module scope: mirrors this project's
        # existing discipline of keeping optional/heavier third-party
        # imports local to where they're actually used (see
        # capture_daemon.bootstrap_calibrations' own local `import cv2`) --
        # a module that only ever constructs an AdWsListener but never
        # starts it should not need `websockets` importable at all.
        import websockets.exceptions
        import websockets.sync.client as ws_client

        backoff = 1.0
        log.info("AD WS listener connecting to %s", self.ws_url)
        while not self._stop_event.is_set():
            try:
                with ws_client.connect(
                    self.ws_url, open_timeout=self._open_timeout_s
                ) as ws:
                    with self._lock:
                        self._last_error = None
                        self._connect_count += 1
                    # Through _set_connected() so the transition is
                    # announced (on_connection_change) -- the old direct
                    # assignment here was the half of the bug that meant
                    # nobody ever heard the socket came up. Announced
                    # BEFORE hydration below, so a dashboard learns
                    # "connected" before any board status that follows.
                    self._set_connected(True)
                    log.info("AD WS listener connected (%s)", self.ws_url)
                    backoff = 1.0
                    self._hydrate_board_status()
                    while not self._stop_event.is_set():
                        try:
                            raw = ws.recv(timeout=self._recv_timeout_s)
                        except TimeoutError:
                            # Normal: just means no message arrived in the
                            # last recv_timeout_s window -- loop back and
                            # check _stop_event again, not a real failure.
                            continue
                        if isinstance(raw, bytes):
                            continue
                        try:
                            obj = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        self._handle_message(obj)
            except websockets.exceptions.ConnectionClosed as exc:
                self._set_connected(False, error=f"connection closed: {exc}")
                _log_disconnect(self._stop_event, "connection closed", exc, backoff)
            except Exception as exc: # noqa: BLE001 -- reconnect loop must survive ANY failure
                self._set_connected(False, error=f"{type(exc).__name__}: {exc}")
                _log_disconnect(self._stop_event, "disconnected", exc, backoff)
            if self._stop_event.is_set():
                break
            self._stop_event.wait(backoff)
            backoff = min(self._max_backoff_s, backoff * 1.5)
        self._set_connected(False)
        log.info("AD WS listener stopped")

    def _handle_message(self, obj: dict[str, Any]) -> None:
        """Only ``type == "state"`` matters; a throw is new only
        if its ``numThrows`` is strictly greater than the last one we
        recorded (guards against a duplicate/replayed message re-adding
        the same throw); a takeout (or ``numThrows`` dropping back to 0)
        clears the buffer entirely -- once a visit is over, this
        listener's buffered copy of it must not outlive that,
        or a later opendarts capture could wrongly match against a stale
        prior-visit throw.

        **Durable logging, added 2026-08-16** (real incident: two
        throws in one recorded session -- see docs/DESIGN.md's "persist real
        diagnostics" task and this module's own diagnostics_snapshot()
        docstring for the full story). Every real event this method acts
        on is now logged, INFO level, via the same root logger
        `opendarts.live.logging_setup.configure_console_and_file_logging()`
        persists to disk by default -- specifically the two events that
        made tonight's investigation impossible to settle definitively:
        a raw "Throw detected" arriving at all (with ITS OWN receive
        timestamp and segment/coords, independent of whether it ends up
        actually buffered -- e.g. a duplicate n is still logged as
        RECEIVED, just not re-buffered), and a raw "Takeout finished"
        arriving (the event that WIPES the buffer). The buffer-clear
        itself is logged separately from the raw event, stating WHY
        (`takeout_finished` vs `numthrows_reset_to_zero` -- AD dropping
        back to 0 without an explicit takeout event, the defensive
        branch this module already had) and exactly what was in the
        buffer immediately before the wipe -- so a future occurrence of
        "AD's event was gone by the time opendarts checked" can be answered
        definitively (arrived-then-wiped, with the wipe reason and
        timing, vs never-arrived-at-all) straight from the log file,
        without needing to reconstruct it from capture-timing math the
        way tonight did. A size-driven eviction (the buffer's own
        DEFAULT_AD_WS_BUFFER_SIZE bound aging out the oldest entries,
        NOT a takeout) is logged as its own distinct event for the same
        reason -- it is a different way real buffered evidence can stop
        being available later, and conflating it with a takeout-driven
        clear in the log would make the exact ambiguity this task exists
        to resolve harder to read back out.
        """
        if obj.get("type") != "state":
            return
        data = obj.get("data") or {}
        event = data.get("event")
        throws = list(data.get("throws") or [])
        try:
            n = int(data.get("numThrows") or 0)
        except (TypeError, ValueError):
            n = 0
        received_at = datetime.now(timezone.utc)

        if event == "Throw detected":
            latest_throw = throws[-1] if throws else None
            log.info(
                "AD WS: 'Throw detected' received (numThrows=%d, received_at_utc=%s) "
                "segment=%s coords=%s",
                n, received_at.isoformat(),
                latest_throw.get("segment") if latest_throw else None,
                latest_throw.get("coords") if latest_throw else None,
            )
        elif event == "Takeout finished":
            log.info(
                "AD WS: 'Takeout finished' received (numThrows=%d, received_at_utc=%s) "
                "-- this WIPES the in-memory throw buffer",
                n, received_at.isoformat(),
            )
        elif event == "Calibration finished":
            log.info(
                "AD WS: 'Calibration finished' received (received_at_utc=%s)",
                received_at.isoformat(),
            )

        # Called BEFORE the lock below, never nested inside it --
        # _update_board_status acquires self._lock itself, and this
        # project's plain threading.Lock is not reentrant.
        self._update_board_status(event, n, data)
        with self._lock:
            if event == "Takeout finished" or (n == 0 and self._ad_num > 0):
                reason = (
                    "takeout_finished" if event == "Takeout finished"
                    else "numthrows_reset_to_zero"
                )
                cleared_summary = [_summarize_event(ev) for ev in self._buffer]
                n_cleared = len(self._buffer)
                self._buffer.clear()
                self._ad_num = 0
                self._last_clear_reason = reason
                self._last_clear_at_utc = received_at
                self._last_clear_buffer_summary = cleared_summary
                log.info(
                    "AD WS buffer cleared: reason=%s n_events_cleared=%d "
                    "contents_before_clear=%s",
                    reason, n_cleared, cleared_summary,
                )
                return
            if event == "Throw detected" and n > self._ad_num and throws:
                self._ad_num = n
                self._buffer.append(
                    AdWsThrow(n=n, throw=throws[-1], received_at_utc=received_at)
                )
                if len(self._buffer) > self._buffer_size:
                    n_over = len(self._buffer) - self._buffer_size
                    evicted = self._buffer[:n_over]
                    self._buffer = self._buffer[-self._buffer_size :]
                    log.info(
                        "AD WS buffer size-evicted %d oldest event(s) (buffer_size=%d): %s",
                        len(evicted), self._buffer_size,
                        [_summarize_event(ev) for ev in evicted],
                    )

    # -- matching ----------------------------------------------------------

    def match(
        self,
        opendarts_captured_at_utc: str | datetime | None,
        *,
        window_sec: float = DEFAULT_MATCH_WINDOW_SEC,
        expect_ordinal: int | None = None,
        allow_time_fallback: bool = False,
    ) -> AdGroundTruth:
        """Match ``opendarts_captured_at_utc`` against this listener's own
        buffer -- NO network call, just a lock-protected list scan, so
        this is safe to call directly from a latency-sensitive caller (see
        module docstring's "Threading model" -- this is exactly why a
        WS-buffer design doesn't need the same inline-vs-background
        tradeoff the original REST design would have needed).

        Returns the SAME ``AdGroundTruth`` shape
        ``dev.ad.ad_ground_truth_rest.match_ad_ground_truth()`` returns,
        so ``ad_ground_truth.json`` and everything that reads it are
        unaffected by which path produced it -- ``match_reason`` values
        are prefixed ``ws_``/``ok_ws`` (vs. the REST path's bare
        ``ok``/``stale``/``fetch_error``/...) purely so a human reading a
        package's ``ad_ground_truth.json`` later can tell which path
        produced it, without that being a schema difference.

        ``expect_ordinal`` is AD's ``n`` for THIS dart (our
        ``visit_index + 1``; both sides reset on takeout). Pass it whenever
        it is known: it makes the match identity-based instead of
        nearest-in-time, which is the only way to avoid attaching the
        previous dart's answer to this one -- see the long note at the
        selection below. Omitted, the legacy nearest-in-time behaviour is
        used unchanged.

        Still NO blocking and no network here either way: when AD has not
        reported the expected dart yet this returns an unmatched result
        saying so, and it is the (background-thread) caller's business to
        wait and ask again.
        """
        matched_at = datetime.now(timezone.utc)
        matched_at_iso = matched_at.isoformat()

        if isinstance(opendarts_captured_at_utc, datetime):
            captured_dt = opendarts_captured_at_utc
            captured_iso = captured_dt.isoformat()
        elif isinstance(opendarts_captured_at_utc, str) and opendarts_captured_at_utc:
            captured_iso = opendarts_captured_at_utc
            captured_dt = _parse_iso(opendarts_captured_at_utc)
        else:
            captured_iso = None
            captured_dt = None

        def _no_match(reason: str) -> AdGroundTruth:
            return AdGroundTruth(
                matched=False,
                match_reason=reason,
                ad_base_url=self.base_url,
                fetched_at_utc=matched_at_iso,
                opendarts_captured_at_utc=captured_iso,
                staleness_sec=None,
                window_sec=window_sec,
            )

        events = self.latest_events()
        if not events:
            reason = "ws_no_buffered_events" if self.is_connected() else "ws_not_connected"
            return _no_match(reason)

        if captured_dt is not None and captured_dt.tzinfo is None:
            captured_dt = captured_dt.replace(tzinfo=timezone.utc)

        if captured_dt is None:
            # No capture time to check against -- same honest posture as
            # the REST module's own equivalent branch: return the latest
            # buffered throw, but say plainly this was never validated
            # against a real capture time.
            best = events[-1]
            return self._to_ground_truth(
                best, matched_at_iso, captured_iso, window_sec, len(events),
                staleness_sec=None, reason="ok_ws_no_capture_time_to_check",
            )

        # MATCH BY ORDINAL WHEN THE CALLER KNOWS WHICH DART THIS IS.
        #
        # Nearest-in-time is not safe on its own. We commit a throw roughly
        # 100-270ms BEFORE AD reports the same dart, so at the moment we
        # look, AD's buffer routinely holds only the PREVIOUS darts -- and
        # the nearest-in-time event is then the previous dart's answer,
        # silently attached to this one. Measured 2026-09-21: we captured
        # dart 3 at 18:16:42.185 and read the buffer at .344; AD's own
        # answer for dart 3 (S12, identical to ours) arrived at .403 -- 59ms
        # later. We recorded AD's dart-2 answer (S11) instead and
        # manufactured a disagreement that never existed. That hits the LAST
        # dart of nearly every visit.
        #
        # AD's `n` (numThrows) and our visit_index both reset on takeout, so
        # `n == visit_index + 1` identifies the same dart on both sides.
        # When the caller supplies it, that is the only acceptable match: if
        # AD has not reported this dart yet we say so and let the caller
        # wait, rather than falling back to an earlier dart's answer. A
        # missing label is recoverable; a confidently wrong one poisons the
        # accuracy corpus.
        if expect_ordinal is not None:
            # TIER 1 -- IDENTITY. AD's n and our visit_index both reset on
            # takeout, so when AD has seen every dart we have, n ==
            # visit_index + 1 names the same dart on both sides. Exact, and
            # preferred whenever it holds.
            # The ordinal names a dart only while AD's buffer is LIVE. If AD
            # stops reporting (it stops detecting, or sits in takeout) its
            # buffer FREEZES holding the last visit's events -- and their
            # n values are 1,2,3, which is exactly what the next visit's
            # darts ask for. Measured 2026-09-21: AD went quiet, and six
            # consecutive darts were handed answers 20-55 SECONDS old, the
            # same three repeated twice, every one recorded as a clean
            # match. So identity is necessary but not sufficient: the event
            # must also be near this dart in time. AD lands within a few
            # hundred ms of our capture either side, so anything beyond
            # AD_ORDINAL_MAX_SKEW_SEC is a stale buffer, not this throw.
            stale_hit = None
            for ev in events:
                if ev.n != expect_ordinal:
                    continue
                delta = (ev.received_at_utc - captured_dt).total_seconds()
                if abs(delta) <= AD_ORDINAL_MAX_SKEW_SEC:
                    return self._to_ground_truth(
                        ev, matched_at_iso, captured_iso, window_sec, len(events),
                        staleness_sec=delta, reason="ok_ws",
                    )
                stale_hit = delta
                break
            if stale_hit is not None:
                # Say it plainly: the number lined up, the clock did not.
                # Never fall through to the time fallback here -- if the
                # ordinal we want is this old, every event beside it is from
                # the same frozen buffer.
                # Name the board state in the refusal. The usual cause is AD
                # wedged in takeout: it flips to takeout on the visit's last
                # throw and, if it never sees the darts come out, never
                # returns to ready -- so it reports nothing further and the
                # buffer keeps serving that visit. Saying "board status:
                # takeout" turns an operator's "why is AD blank" into an
                # answer instead of an investigation.
                status, _raw = self.board_status()
                gt = _no_match(
                    f"ws_stale_buffer: AD's dart {expect_ordinal} is "
                    f"{stale_hit:+.1f}s from this capture (limit "
                    f"{AD_ORDINAL_MAX_SKEW_SEC:.1f}s) -- AD has stopped "
                    f"reporting and its buffer is frozen (board status: "
                    f"{status})"
                )
                gt.staleness_sec = stale_hit
                return gt
            have = ", ".join(str(ev.n) for ev in events) or "none"

            # TIER 2 -- TIME, once the caller has waited and tier 1 never
            # arrived. AD's counter counts AD's OWN darts: the moment it
            # misses one of ours, every later n in that visit is off by one
            # and tier 1 can never match again. Measured 2026-09-21: one
            # rig's AD reported 2 events for 3 darts (it never saw an
            # off-board dart), so our dart 3 WAS AD's n=2 -- and demanding
            # n=3 discarded a perfectly good answer, and would have
            # discarded every dart after it too.
            #
            # So fall back to arrival time, which does not depend on AD's
            # bookkeeping. The floor is what makes this safe: an event must
            # have arrived no earlier than AD_LEAD_EPSILON_SEC before our
            # capture, which excludes the PREVIOUS dart's answer (that one
            # lands seconds earlier -- the original nearest-in-time bug,
            # where it was 1.85s early and got attached anyway). AD is not
            # reliably behind us either -- it has landed 40ms AHEAD of a
            # capture -- so the floor is a small window on both sides
            # rather than "must be later than us".
            if allow_time_fallback:
                floor = captured_dt - timedelta(seconds=AD_LEAD_EPSILON_SEC)
                fresh = [ev for ev in events if ev.received_at_utc >= floor]
                if fresh:
                    best = min(
                        fresh,
                        key=lambda ev: abs((ev.received_at_utc - captured_dt).total_seconds()),
                    )
                    return self._to_ground_truth(
                        best, matched_at_iso, captured_iso, window_sec, len(events),
                        staleness_sec=(best.received_at_utc - captured_dt).total_seconds(),
                        # A DISTINCT reason: this throw's AD label came from
                        # timing because AD's numbering had drifted. Visible
                        # in the corpus rather than silently blended in with
                        # the exact matches.
                        reason=f"ok_ws_by_time: AD n={best.n} (expected {expect_ordinal}; "
                               f"AD numbering drifted, matched on arrival time)",
                    )
                return _no_match(
                    f"ws_no_event_near_capture: AD reported nothing within "
                    f"{AD_LEAD_EPSILON_SEC:.1f}s before this dart or since "
                    f"(expected {expect_ordinal}, buffered: {have})"
                )

            return _no_match(
                f"ws_awaiting_ordinal: AD has not reported dart {expect_ordinal} "
                f"yet (buffered: {have})"
            )

        best = None
        best_delta = None
        for ev in events:
            delta = (ev.received_at_utc - captured_dt).total_seconds()
            if abs(delta) <= window_sec and (best is None or abs(delta) < abs(best_delta)):
                best = ev
                best_delta = delta

        if best is None:
            nearest = min(events, key=lambda ev: abs((ev.received_at_utc - captured_dt).total_seconds()))
            nearest_delta = (nearest.received_at_utc - captured_dt).total_seconds()
            gt = _no_match(
                f"stale_ws: nearest buffered AD event {nearest_delta:+.1f}s from opendarts "
                f"capture (window={window_sec:.1f}s)"
            )
            gt.staleness_sec = nearest_delta
            return gt

        return self._to_ground_truth(
            best, matched_at_iso, captured_iso, window_sec, len(events),
            staleness_sec=best_delta, reason="ok_ws",
        )

    def _to_ground_truth(
        self,
        ev: AdWsThrow,
        matched_at_iso: str,
        captured_iso: str | None,
        window_sec: float,
        n_events: int,
        *,
        staleness_sec: float | None,
        reason: str,
    ) -> AdGroundTruth:
        sector, ring = segment_to_sector_ring(ev.throw.get("segment"))
        inner = ev.throw.get("detections")
        return AdGroundTruth(
            matched=True,
            match_reason=reason,
            ad_base_url=self.base_url,
            fetched_at_utc=matched_at_iso,
            opendarts_captured_at_utc=captured_iso,
            staleness_sec=staleness_sec,
            window_sec=window_sec,
            sector=sector,
            ring=ring,
            tip_xy_mm=_tip_xy_mm(ev.throw),
            ad_method=ev.throw.get("method"),
            ad_bouncer=ev.throw.get("bouncer"),
            ad_n_cam_detections=len(inner) if isinstance(inner, list) else None,
            raw_segment=ev.throw.get("segment"),
            source_index=ev.n,
            n_detections_in_response=n_events,
        )
