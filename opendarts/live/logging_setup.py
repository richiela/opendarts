"""opendarts/live/logging_setup.py -- shared console+file logging setup for
this project's live CLI entrypoints (run_product.py, capture_daemon.py,
server.py).

2026-08-12, after a live process needed manual `| tee` piping to
get logs onto disk for remote (SSH) inspection: "why dont you just write
it into the code to log to stdout and file... Doing file redirection is
stupid and lazy." Fixed properly: every CLI here always logs to BOTH
stdout (for whoever's watching the terminal) AND a real file on disk, so
it can be read after the fact -- e.g. over SSH -- without the operator
ever having to remember to redirect anything.

2026-08-13, console color: two round trips on this one, worth
recording so a future pass doesn't undo it by accident.
1. First pass: `opendarts/live/{run_product,server}.py` passed no
   `log_config` to uvicorn, so uvicorn's own default LOGGING_CONFIG
   attached its own colorized, stderr-only handler directly to the
   "uvicorn"/"uvicorn.error" loggers -- completely bypassing this
   module's root-logger setup. Result: inconsistent console (some lines
   colored via uvicorn's own handler, some plain via this module's), and
   the colored lines (including uvicorn's WebSocket accept/connect
   chatter) never reached the log file at all, since that handler was
   stderr-only and never registered here. Fixed by passing
   `log_config=None` to both uvicorn entrypoints -- uvicorn then installs
   no handlers of its own, so "uvicorn"/"uvicorn.error" fall back to
   normal propagate=True and flow into THIS module's root handlers.
2. Second pass, immediately after: wanted color kept, not
   dropped -- "can we go the other way instead? id like color coded and
   properly spaced and highlighted output". So the unification from step
   1 stays (one pipeline, not two), but the console formatter itself
   switched from plain `logging.Formatter` to `uvicorn.logging.
   DefaultFormatter` (see `_console_formatter()` below) -- a public
   formatter class from this project's own `uvicorn` dependency
   (requirements.txt). Net
   effect of steps 1+2 together: every line on the console -- this
   project's own `opendarts.*` loggers AND uvicorn's own -- now shares one
   colorized, level-tagged, aligned style, and still all reaches the log
   file (in plain form there -- see file_formatter below for why).
3. Third pass, same session: noticed long lines (this project's
   own per-camera "opened ok -- device=... backend=... requested=...
   actual=... open_latency=..." messages are frequent offenders) wrap at
   the terminal's own left edge, landing the continuation below the
   date/level/name prefix instead of under the message column -- "is
   there a way to make it wrap but show up where the tab is?" Added
   `_HangingIndentFormatter` below: measures the prefix's real on-screen
   width (via a parallel *uncolored* render of the same record -- ANSI
   codes are zero-width on screen, so the plain render's length IS the
   colored line's true visible prefix width) and re-wraps the message to
   the real terminal width, continuation lines padded to land under the
   message column. Console-only, same reasoning as color: a log FILE is
   read with `grep`/`less`/`cat`, and there's no fixed "terminal width"
   to wrap against for a file that might be read in a wide terminal or a
   narrow one -- wrapping there would just break line-based tools for no
   benefit. Skips wrapping entirely for anything with a traceback
   (already multi-line, and re-wrapping would silently drop the
   traceback text -- see the class docstring) or when stdout isn't a
   real terminal (piped/redirected -- there's no meaningful "width" to
   wrap against there either).

4. Fourth pass, 2026-08-16, after a real incident (two
   throws in one recorded session -- see the "persist real diagnostics"
   task in docs/DESIGN.md's spirit) where diagnosing an anomalous throw after
   the fact needed a durable process log that simply did not exist on
   the rig -- confirmed live: no log file anywhere, because this
   function's own FILE handler was a plain, unbounded `logging.
   FileHandler` and nothing had ever actually been run long enough
   against it to notice it just grows forever. Switched to
   `RotatingFileHandler` (size-based, not time-based -- this project's
   log volume tracks how much is THROWN, not how much wall-clock time
   passes, so a size cap is the more meaningful bound): `maxBytes=20MB`,
   `backupCount=10` -- up to 11 files (`<name>.log` + up to 10
   `<name>.log.N` backups), ~220MB worst case per log name
   (`run_product.log` / `capture_daemon.log`, the only two callers today
   -- see the bottom of each of those files' own `main()`). 20MB was
   picked, not measured against a real full-session log (none survived
   long enough to measure, which is the whole problem this fixes) --
   reasoned instead from this module's own DEBUG-level file handler
   already carrying full per-message chatter from `websockets`/uvicorn
   (see `console_level` docstring above), at a rough per-line estimate
   this comfortably covers a multi-hour live session before rotating,
   while 10 backups keeps roughly a day-plus of rotated history around
   without unbounded growth. Revisit with a real measured number once a
   live session actually rotates a file in practice. Still `mode="a"`
   (append) on the active file -- rotation only ever creates NEW backup
   files when the active one crosses `maxBytes`, it never truncates on
   process start, so the existing "diagnose a process that already
   exited" property from pass 1 above is unchanged.
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import textwrap
from logging.handlers import RotatingFileHandler
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

from opendarts.paths import DATA_DIR

# <repo>/data/logs/ -- same tree as DEFAULT_PACKAGE_ROOT
# (opendarts/live/capture_daemon.py's <repo>/data/packages/), already
# gitignored via the repo's existing /data/ entry.
#
# OPENDARTS_LOG_DIR overrides it. The test suite sets it (tests/conftest.py):
# tests that start the real entrypoints were appending to the rig's own
# run_product.log -- tens of thousands of lines of test chatter, rotating
# real history away (2026-09-17).
DEFAULT_LOG_DIR = (Path(os.environ["OPENDARTS_LOG_DIR"])
                   if os.environ.get("OPENDARTS_LOG_DIR")
                   else DATA_DIR / "logs")

_LOG_FORMAT = "%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# MILLISECOND TIMESTAMPS, 2026-09-04 -- real, live cost, not
# theoretical: a cross-repo latency-parity investigation the same night
# needed to measure a ~184ms gap between two log lines and could not,
# because every line here previously read at second granularity
# (`2026-09-04 11:55:35`) -- `logging.Formatter`'s own default `asctime`
# behavior DOES include milliseconds (`,%03d` appended automatically),
# but passing an explicit `datefmt=` (as this module always has, both
# handlers, since day one) SILENTLY SUPPRESSES that default -- `asctime`
# then renders using ONLY the strptime-style `datefmt` string, which has
# no millisecond directive (`%f` is a datetime.strftime thing, not a
# `time.strftime` one, and `asctime` is built via the latter -- see
# `logging.Formatter.formatTime()`'s own source). The investigation had
# to fall back to differencing internal monotonic iteration counters
# instead of just reading two log timestamps -- a real, avoidable gap.
# Fixed by keeping `datefmt=_DATE_FORMAT` (still the human-readable
# `YYYY-MM-DD HH:MM:SS` this project's log files/console have always
# used -- no reason to change that part) and appending `.%(msecs)03d`
# EXPLICITLY to the format string itself, right after `%(asctime)s` --
# `record.msecs` is a plain `LogRecord` attribute (set at record
# creation time, before any formatter ever touches it), so this works
# identically regardless of which `Formatter` subclass renders the line
# -- confirmed real for both `logging.Formatter` (the file handler,
# below) AND `uvicorn.logging.DefaultFormatter` (the console handler,
# `_console_formatter()` below) -- neither needed to change HOW it
# renders `asctime`, only what surrounds it.
#
# Applied to BOTH handlers, not just the file handler: the file handler
# is unquestionably the one this fix is FOR (a durable log a human/tool
# reads after the fact, exactly what the originating investigation was
# doing) -- but the console handler shares the exact same `_DATE_FORMAT`
# constant and renders through the same `%(asctime)s`-anchored format
# string shape, so leaving it at second-granularity would mean a human
# watching the live terminal during exactly this kind of latency
# investigation sees coarser timestamps than the file being tail -f'd
# alongside it, for no real benefit (a few extra characters on an
# already colorized, already-wrapped line is a trivial cost, see
# `_HangingIndentFormatter`'s own real-terminal-width accounting below,
# which correctly measures the new, slightly-longer prefix the same way
# it measures every other formatter change here).

# See module docstring's 4th pass (2026-08-16) for the reasoning behind
# these two numbers -- reasoned, not yet measured against a real full
# live session (none has ever survived long enough to measure, which was
# exactly the problem this fixes).
DEFAULT_LOG_MAX_BYTES = 20 * 1024 * 1024 # 20MB per file before rotating
DEFAULT_LOG_BACKUP_COUNT = 10 # + the active file = ~220MB worst case per log name

# THIRD-PARTY LOGGER VOLUME, 2026-09-04 -- part of the live-diagnostics
# gate (`opendarts.live.diagnostics_gate`). Real, measured finding from that
# night's instrumentation-cost audit: `websockets` (this project's own AD
# event-feed client + the dashboard's push websocket -- both real,
# constantly-open connections during live play) and `uvicorn.error`
# (which, confirmed by reading `uvicorn/protocols/websockets/
# websockets_impl.py` directly, is where uvicorn's own per-connection
# "connection open"/"connection closed" lines log from, at INFO level,
# not DEBUG) together accounted for >80% of this project's own file log
# volume during live play. Neither logger has its OWN level set anywhere
# else in this codebase (both default NOTSET, i.e. "inherit the
# effective level of the nearest ancestor with one set") -- and this
# module's own root logger is deliberately kept at DEBUG unconditionally
# (see `configure_console_and_file_logging()`'s own `level` default),
# because that's what lets this project's OWN `opendarts.*` DEBUG lines
# reach the file handler. That means, without this, `websockets`/
# `uvicorn.error` inherit DEBUG too -- confirmed live: `opendarts.capture.
# throw_trigger.advance()`'s own `log.isEnabledFor(logging.DEBUG)` guard
# on its per-camera settle-status line looked level-gated but was a
# no-op in production for the identical reason (root effective level is
# always DEBUG here).
#
# `websockets.{client,server,protocol}` (its own per-frame chatter, see
# that library's own `protocol.py`) all log at DEBUG -- WARNING already
# suppresses every one of them. `uvicorn.error`'s connect/close lines log
# at INFO -- WARNING is the lowest level that suppresses those too while
# still letting a genuine uvicorn ERROR/WARNING (a real problem) through
# unchanged. Setting the level on the PARENT logger name ("websockets",
# not each of "websockets.client"/"websockets.server"/"websockets.
# protocol" individually) is sufficient: none of those child loggers set
# their own explicit level anywhere in the installed library (grepped,
# zero `.setLevel(` calls) -- Python logging's own effective-level
# resolution walks up to the nearest ancestor WITH a level set, which
# this becomes.
_THIRD_PARTY_DIAGNOSTIC_LOGGER_NAMES: tuple[str, ...] = ("websockets", "uvicorn.error")
_THIRD_PARTY_QUIET_LEVEL = logging.WARNING


def set_third_party_diagnostics_enabled(enabled: bool) -> None:
    """Flip `websockets`/`uvicorn.error`'s own logger level at runtime --
    `logging.NOTSET` (inherit the root logger's own DEBUG, full detail
    restored) when `enabled`, `_THIRD_PARTY_QUIET_LEVEL` (WARNING,
    real-problems-only) when not. A genuine, real-time
    `logging.getLogger(name).setLevel(...)` call, not a flag read
    elsewhere -- a logger's own level IS the thing changing here, so
    nothing else needs to consult this function's own return value (it
    has none). Called by `opendarts.live.diagnostics_gate.set_enabled()`
    (the live A/B toggle this whole mechanism exists for) and once, at
    import time, by that same module to establish the quiet default
    before any toggle has ever been requested -- see that module's own
    docstring for the full design. Safe to call directly too (e.g. a
    test wanting to assert the real logger level changed, not just that
    some other flag flipped)."""
    level = logging.NOTSET if enabled else _THIRD_PARTY_QUIET_LEVEL
    for name in _THIRD_PARTY_DIAGNOSTIC_LOGGER_NAMES:
        logging.getLogger(name).setLevel(level)


class _HangingIndentFormatter(logging.Formatter):
    """Wraps a formatted console line so continuation lines land under the
    message column (a "hanging indent") instead of back at the
    terminal's left edge.

    Delegates the actual prefix rendering (timestamp, colored level tag,
    logger name) to `colored`; `plain` must render the SAME fields with
    `use_colors=False` -- used only to measure the prefix's true on-screen
    width, since ANSI escape codes add bytes but zero visible columns.
    That measured width can't be recovered from the colored string alone
    (color codes are embedded inside it), which is why this needs two
    parallel renders rather than one.

    Only re-wraps a plain single-line message with no exception/stack
    info attached -- for anything else (a traceback, an already
    multi-line message) `logging.Formatter.format()`'s own appended
    `exc_text`/`stack_info` lives outside what `record.getMessage()`
    returns, so blindly re-wrapping just `getMessage()` and reassembling
    would silently drop that text. Bailing out there is correct, not a
    missed case -- tracebacks are conventionally left-aligned at column 0
    anyway, not hanging-indented.
    """

    def __init__(self, colored: logging.Formatter, plain: logging.Formatter) -> None:
        super().__init__()
        self._colored = colored
        self._plain = plain

    def format(self, record: logging.LogRecord) -> str:
        rendered = self._colored.format(record)

        if record.exc_info or record.stack_info or record.exc_text:
            return rendered

        width = shutil.get_terminal_size(fallback=(0, 0)).columns
        if not width:
            return rendered # stdout isn't a real terminal -- nothing to wrap against

        message = record.getMessage()
        if not message or not rendered.endswith(message):
            # The colored render's tail isn't a plain, unmodified copy of
            # the message (e.g. some future formatter starts coloring the
            # message body too) -- can't safely locate/replace it, so
            # leave the line as the terminal would have wrapped it rather
            # than risk corrupting the output.
            return rendered

        plain_prefix_len = len(self._plain.format(record)) - len(message)
        if plain_prefix_len <= 0 or width <= plain_prefix_len + 20:
            return rendered # not enough room to make wrapping worthwhile

        avail = width - plain_prefix_len
        wrapped_lines = textwrap.wrap(message, width=avail, break_long_words=False, break_on_hyphens=False)
        if len(wrapped_lines) <= 1:
            return rendered # fits on one line already

        colored_prefix = rendered[: len(rendered) - len(message)]
        return colored_prefix + ("\n" + " " * plain_prefix_len).join(wrapped_lines)


def _console_formatter() -> logging.Formatter:
    """Colorized, aligned, hanging-indented formatter for the CONSOLE
    handler only.

    Reuses `uvicorn.logging.DefaultFormatter` -- it already does exactly
    what was asked (color-coded level tag: green INFO, yellow WARNING,
    red ERROR, bright-red CRITICAL; consistently padded so the message
    column lines up regardless of level-name length) rather than
    hand-rolling ANSI codes this project would then have to maintain.
    `use_colors=None` auto-detects via `sys.stdout.isatty()` at
    formatter-construction time -- colorized on a real terminal, plain
    when piped/redirected/non-tty (e.g. a CI log capture), matching how
    uvicorn's own default logging already behaved before this project
    briefly disabled it (see module docstring above). Wrapped in
    `_HangingIndentFormatter` so long messages wrap under the message
    column instead of back at column 0 -- see that class's docstring.

    Falls back to the plain formatter (still hanging-indented) if
    `uvicorn` isn't importable -- `opendarts.capture_daemon` imports this
    module too, and (unlike `run_product.py`/`server.py`) treats uvicorn
    as a genuinely optional dependency (no fastapi/uvicorn needed for the
    bare capture loop -- see `run_product.py`'s own local `import
    uvicorn` comments). Console color is a nice-to-have; it must never be
    why the capture loop's own logging setup fails to import.
    """
    fmt = "%(asctime)s.%(msecs)03d %(levelprefix)s %(name)s: %(message)s"
    try:
        from uvicorn.logging import DefaultFormatter
    except ImportError:
        plain = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        return _HangingIndentFormatter(colored=plain, plain=plain)

    colored = DefaultFormatter(fmt=fmt, datefmt=_DATE_FORMAT, use_colors=None)
    plain = DefaultFormatter(fmt=fmt, datefmt=_DATE_FORMAT, use_colors=False)
    return _HangingIndentFormatter(colored=colored, plain=plain)


def configure_console_and_file_logging(
    log_name: str,
    level: int = logging.DEBUG,
    log_dir: Path = DEFAULT_LOG_DIR,
    console_level: int | None = None,
    max_bytes: int = DEFAULT_LOG_MAX_BYTES,
    backup_count: int = DEFAULT_LOG_BACKUP_COUNT,
) -> Path:
    """Configure the root logger with two handlers: stdout (for a human
    watching the terminal live) and a real, SIZE-ROTATED file on disk at
    `<log_dir>/<log_name>.log` (append mode -- doesn't truncate a
    previous run's history, which matters for after-the-fact diagnosis
    of a process that's already exited or been killed; rotation itself
    only ever creates NEW `<log_name>.log.1`, `.2`, ... backups once the
    active file crosses `max_bytes` -- see module docstring's 4th pass,
    2026-08-16, for why this exists: a real incident where no persistent
    log existed at all to diagnose an anomalous throw after the fact).

    `max_bytes`/`backup_count`: default to `DEFAULT_LOG_MAX_BYTES`
    (20MB)/`DEFAULT_LOG_BACKUP_COUNT` (10) -- see those constants' own
    comment for the reasoning. Overridable per call (e.g. a test wanting
    a tiny cap to actually exercise rotation without writing 20MB of
    fixture data).

    `level`: the FILE handler's level -- defaults to DEBUG, full detail
    always preserved on disk regardless of what the console shows.
    `console_level`: the STDOUT handler's level -- defaults to INFO
    (2026-08-12, from live testing: debug output on stdout was too noisy,
    mostly websocket chatter -- with the root logger previously left at DEBUG
    with no console/file split, third-party libraries this process pulls
    in (`websockets`, uvicorn) also log at DEBUG, including their own
    per-frame/ping-pong chatter on both the AD event-feed connection and
    the dashboard's own push to an open browser tab -- none of that is
    useful on a live terminal, but it's still worth keeping in the file
    for after-the-fact debugging, hence the split rather than just
    lowering `level` outright). Pass `console_level=logging.DEBUG`
    explicitly to restore the old fully-verbose console for deep
    debugging.

    Returns the log file path so the caller can print it -- an operator
    (or whoever's reading over SSH) should always be told exactly where
    to look, not have to guess or grep for it.

    Safe to call more than once (e.g. from tests): only removes/replaces
    handlers this function itself previously added (tagged via a private
    marker attribute), never touches handlers configured by something
    else, so it never silently disables logging some other part of the
    process set up independently.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"

    resolved_console_level = logging.INFO if console_level is None else console_level

    root = logging.getLogger()
    # The root logger itself must be at least as permissive as the more
    # verbose of the two handlers, or records get filtered before either
    # handler ever sees them -- each handler's own level then does the
    # real per-destination filtering below.
    root.setLevel(min(level, resolved_console_level))

    for handler in list(root.handlers):
        if getattr(handler, "_opendarts_managed", False):
            root.removeHandler(handler)

    # Console gets the colorized formatter; the file handler stays plain
    # -- see _console_formatter()'s docstring for why the two must differ.
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_console_formatter())
    stream_handler.setLevel(resolved_console_level)
    stream_handler._opendarts_managed = True # type: ignore[attr-defined]
    root.addHandler(stream_handler)

    file_handler = RotatingFileHandler(
        log_path, mode="a", maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    file_handler.setLevel(level)
    file_handler._opendarts_managed = True # type: ignore[attr-defined]
    root.addHandler(file_handler)

    return log_path
