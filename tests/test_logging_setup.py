"""Tests for opendarts/live/logging_setup.py -- the dual console+file
logging, written into the code (2026-08-12) rather than relying on a
manual `| tee` shell redirection.

2026-08-16 additions: real size-based rotation (RotatingFileHandler),
added after a real incident (one recorded session) where no
persistent process log existed on the rig at all to diagnose an anomalous
throw -- see this module's own docstring's 4th pass and docs/DESIGN.md's
"persist real diagnostics" task.

2026-09-04 additions: millisecond timestamps on BOTH handlers (see
logging_setup.py's own module-level `_LOG_FORMAT`/`_console_formatter()`
comments for the full incident this closes -- a real cross-repo latency
investigation the same night couldn't measure a ~184ms gap because every
line here read at second granularity)."""
from __future__ import annotations

import io
import logging
import re
import sys
from logging.handlers import RotatingFileHandler

from opendarts.live.logging_setup import (
    configure_console_and_file_logging,
    set_third_party_diagnostics_enabled,
)


def test_configure_creates_log_file_and_writes_to_it(tmp_path):
    log_dir = tmp_path / "logs"
    log_path = configure_console_and_file_logging("test_run", log_dir=log_dir)

    assert log_path == log_dir / "test_run.log"
    assert log_path.parent.is_dir()

    logger = logging.getLogger("opendarts.test_logging_setup")
    logger.info("hello from the test")

    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_path.read_text()
    assert "hello from the test" in content


def test_configure_appends_across_calls_not_truncates(tmp_path):
    log_dir = tmp_path / "logs"
    log_path = configure_console_and_file_logging("test_run", log_dir=log_dir)
    logging.getLogger("opendarts.test_logging_setup").info("first run")
    for handler in logging.getLogger().handlers:
        handler.flush()

    # Simulate a second process start against the same log file (this is
    # the whole point -- diagnosing a process that already exited/was
    # killed shouldn't lose the previous run's history).
    configure_console_and_file_logging("test_run", log_dir=log_dir)
    logging.getLogger("opendarts.test_logging_setup").info("second run")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_path.read_text()
    assert "first run" in content
    assert "second run" in content


def test_configure_is_idempotent_no_duplicate_handlers(tmp_path):
    log_dir = tmp_path / "logs"
    configure_console_and_file_logging("test_run", log_dir=log_dir)
    configure_console_and_file_logging("test_run", log_dir=log_dir)
    configure_console_and_file_logging("test_run", log_dir=log_dir)

    managed = [h for h in logging.getLogger().handlers if getattr(h, "_opendarts_managed", False)]
    # Exactly one stream handler + one file handler, not accumulating
    # duplicates across repeated calls (e.g. from repeated test runs in
    # the same process, or a real process re-configuring for any reason).
    assert len(managed) == 2


def test_configure_does_not_remove_unrelated_handlers(tmp_path):
    root = logging.getLogger()
    sentinel = logging.NullHandler()
    root.addHandler(sentinel)
    try:
        configure_console_and_file_logging("test_run", log_dir=tmp_path / "logs")
        assert sentinel in root.handlers
    finally:
        root.removeHandler(sentinel)


# ---------------------------------------------------------------------------
# Real size-based rotation (2026-08-16) -- see module docstring's 4th pass.
# ---------------------------------------------------------------------------

def _file_handler() -> RotatingFileHandler:
    for handler in logging.getLogger().handlers:
        if getattr(handler, "_opendarts_managed", False) and isinstance(handler, RotatingFileHandler):
            return handler
    raise AssertionError("no managed RotatingFileHandler found on the root logger")


def test_configure_installs_a_rotating_file_handler_not_a_plain_one(tmp_path):
    configure_console_and_file_logging("test_run", log_dir=tmp_path / "logs")
    handler = _file_handler()
    assert isinstance(handler, RotatingFileHandler)


def test_configure_rotation_uses_the_requested_max_bytes_and_backup_count(tmp_path):
    configure_console_and_file_logging(
        "test_run", log_dir=tmp_path / "logs", max_bytes=12345, backup_count=3,
    )
    handler = _file_handler()
    assert handler.maxBytes == 12345
    assert handler.backupCount == 3


def test_configure_defaults_to_the_module_constants_when_not_overridden(tmp_path):
    from opendarts.live.logging_setup import DEFAULT_LOG_BACKUP_COUNT, DEFAULT_LOG_MAX_BYTES

    configure_console_and_file_logging("test_run", log_dir=tmp_path / "logs")
    handler = _file_handler()
    assert handler.maxBytes == DEFAULT_LOG_MAX_BYTES
    assert handler.backupCount == DEFAULT_LOG_BACKUP_COUNT


def test_configure_actually_rotates_once_max_bytes_is_exceeded(tmp_path):
    """Real behavior, not just constructor-argument plumbing: write enough
    real log lines to cross a tiny max_bytes cap and confirm a real
    `<name>.log.1` backup file appears, and the active file keeps
    growing from a small base afterward (the real point: an operator
    reading the log later gets bounded per-file size, not unbounded
    growth -- the exact gap that left NO log at all survivable on the rig
    before this existed)."""
    log_dir = tmp_path / "logs"
    log_path = configure_console_and_file_logging(
        "test_run", log_dir=log_dir, max_bytes=2000, backup_count=2,
    )
    logger = logging.getLogger("opendarts.test_logging_setup.rotation")
    for i in range(400):
        logger.info("filler log line number %04d to exceed the tiny rotation cap", i)
    for handler in logging.getLogger().handlers:
        handler.flush()

    backup_path = log_dir / "test_run.log.1"
    assert backup_path.exists(), (
        "expected at least one rotated backup file once max_bytes was exceeded"
    )
    # The active file itself must never exceed max_bytes by more than
    # roughly one record's worth (RotatingFileHandler rotates BEFORE the
    # write that would cross the cap, so a single record can slightly
    # overshoot, but nothing close to the full unrotated 400-line total).
    assert log_path.stat().st_size < 2000 + 500


def test_configure_rotation_survives_across_reconfigure_calls(tmp_path):
    """Calling configure_console_and_file_logging() again (simulating a
    second process start against the same log directory, same as
    test_configure_appends_across_calls_not_truncates above) must still
    produce a RotatingFileHandler, not silently regress to a plain
    unbounded one."""
    log_dir = tmp_path / "logs"
    configure_console_and_file_logging("test_run", log_dir=log_dir, max_bytes=5000, backup_count=1)
    configure_console_and_file_logging("test_run", log_dir=log_dir, max_bytes=5000, backup_count=1)
    handler = _file_handler()
    assert isinstance(handler, RotatingFileHandler)
    assert handler.maxBytes == 5000


# ---------------------------------------------------------------------------
# Millisecond timestamps (2026-09-04) -- real, against a REAL EMITTED LINE
# on both handlers, per this task's own explicit warning that the uvicorn
# formatter's interaction here "looks right in source and isn't" without
# actually checking.
# ---------------------------------------------------------------------------

# Matches "2026-09-04 13:19:11.696" -- a real YYYY-MM-DD HH:MM:SS.mmm
# timestamp, the exact shape both handlers must now produce.
_MS_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}")


def test_file_handler_emits_a_real_millisecond_timestamp(tmp_path):
    log_dir = tmp_path / "logs"
    log_path = configure_console_and_file_logging("ms_test_file", log_dir=log_dir)

    logger = logging.getLogger("opendarts.ms_test_file")
    logger.info("real emitted line for ms verification")
    for handler in logging.getLogger().handlers:
        handler.flush()

    content = log_path.read_text()
    assert _MS_TIMESTAMP_RE.search(content), (
        f"file handler line has no millisecond-precision timestamp: {content!r}"
    )


def test_console_handler_emits_a_real_millisecond_timestamp(tmp_path, monkeypatch):
    """Real stdout capture, not just reading the format string -- proves
    uvicorn.logging.DefaultFormatter (the console handler's real
    formatter, see _console_formatter()) actually renders %(msecs)03d
    the same as a plain logging.Formatter does; this is exactly the kind
    of interaction this task's own instructions warned could look right
    in source and not be."""
    log_dir = tmp_path / "logs"
    configure_console_and_file_logging("ms_test_console", log_dir=log_dir, console_level=logging.INFO)

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    # The console StreamHandler was constructed against the REAL sys.stdout
    # at configure-time, above -- point it at the capture buffer directly
    # (matching how every other real handler-introspection test in this
    # file reaches into the actual configured handler, not a fresh one).
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, RotatingFileHandler
        ):
            handler.stream = captured

    logger = logging.getLogger("opendarts.ms_test_console")
    logger.info("real emitted line for ms verification")
    for handler in logging.getLogger().handlers:
        handler.flush()

    output = captured.getvalue()
    assert _MS_TIMESTAMP_RE.search(output), (
        f"console handler line has no millisecond-precision timestamp: {output!r}"
    )


def test_ms_timestamp_keeps_the_same_human_readable_date_format(tmp_path):
    """Real regression guard: the fix must ADD milliseconds, not silently
    change the pre-existing YYYY-MM-DD HH:MM:SS shape (_DATE_FORMAT) the
    rest of this project's own tooling/humans already read log files
    against."""
    log_dir = tmp_path / "logs"
    log_path = configure_console_and_file_logging("ms_test_format", log_dir=log_dir)

    logger = logging.getLogger("opendarts.ms_test_format")
    logger.info("real emitted line for ms verification")
    for handler in logging.getLogger().handlers:
        handler.flush()

    line = next(
        line for line in log_path.read_text().splitlines() if "ms verification" in line
    )
    # e.g. "2026-09-04 13:19:11.696 INFO opendarts.ms_test_format: ..."
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} INFO ", line), (
        f"unexpected line shape: {line!r}"
    )


# ---------------------------------------------------------------------------
# set_third_party_diagnostics_enabled() -- opendarts.live.diagnostics_gate's
# own real, live-toggled `logging.getLogger(name).setLevel(...)` call for
# `websockets`/`uvicorn.error`, 2026-09-04. Confirmed live: these two
# loggers accounted for >80% of this project's own file log volume during
# live play.
# ---------------------------------------------------------------------------


def _reset_third_party_logger_levels():
    logging.getLogger("websockets").setLevel(logging.NOTSET)
    logging.getLogger("uvicorn.error").setLevel(logging.NOTSET)


def test_set_third_party_diagnostics_disabled_sets_warning_level():
    _reset_third_party_logger_levels()
    try:
        set_third_party_diagnostics_enabled(False)
        assert logging.getLogger("websockets").level == logging.WARNING
        assert logging.getLogger("uvicorn.error").level == logging.WARNING
    finally:
        _reset_third_party_logger_levels()


def test_set_third_party_diagnostics_enabled_restores_notset():
    _reset_third_party_logger_levels()
    try:
        set_third_party_diagnostics_enabled(False)
        set_third_party_diagnostics_enabled(True)
        assert logging.getLogger("websockets").level == logging.NOTSET
        assert logging.getLogger("uvicorn.error").level == logging.NOTSET
    finally:
        _reset_third_party_logger_levels()


def test_set_third_party_diagnostics_quiet_level_actually_suppresses_debug_and_info():
    """Real, direct proof this is genuinely effective, not just a level
    number changing -- a DEBUG record from `websockets` and an INFO
    record from `uvicorn.error` (the real level uvicorn's own websocket
    connect/close chatter logs at, confirmed by reading uvicorn's own
    protocols/websockets/websockets_impl.py) must both be filtered when
    disabled, and both pass through when enabled."""
    _reset_third_party_logger_levels()
    # The CHILD's own level, too: anything that built an app earlier in this
    # process has already quieted it (create_app calls
    # set_third_party_diagnostics_enabled), and an explicit level on the
    # child outlives a reset of the parents. Same reset the sibling test
    # below already does. Without it this test fails only when another
    # file's tests ran first -- which under `pytest -n auto` depends on the
    # worker split.
    logging.getLogger("websockets.client").setLevel(logging.NOTSET)
    # And the ROOT level, which is what a NOTSET child ultimately inherits:
    # pytest and anything that configured logging earlier in this process
    # leave it wherever they like, and at WARNING the "enabled" half of this
    # test can never pass. Restored in the finally.
    root = logging.getLogger()
    root_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        ws_logger = logging.getLogger("websockets.client") # a real child logger
        uv_logger = logging.getLogger("uvicorn.error")

        set_third_party_diagnostics_enabled(False)
        assert ws_logger.isEnabledFor(logging.DEBUG) is False
        assert uv_logger.isEnabledFor(logging.INFO) is False
        # A real problem (WARNING/ERROR) must still get through even
        # while quiet -- this is a volume filter, not a full silence.
        assert uv_logger.isEnabledFor(logging.WARNING) is True

        set_third_party_diagnostics_enabled(True)
        assert ws_logger.isEnabledFor(logging.DEBUG) is True
        assert uv_logger.isEnabledFor(logging.INFO) is True
    finally:
        root.setLevel(root_level)
        _reset_third_party_logger_levels()


def test_set_third_party_diagnostics_child_loggers_inherit_the_parent_level():
    """websockets.{client,server,protocol} never set their own explicit
    level anywhere in the installed library (confirmed by grep) -- only
    setting the PARENT ("websockets") logger's level must still control
    them, via Python logging's own effective-level inheritance."""
    _reset_third_party_logger_levels()
    logging.getLogger("websockets.server").setLevel(logging.NOTSET)
    logging.getLogger("websockets.protocol").setLevel(logging.NOTSET)
    try:
        set_third_party_diagnostics_enabled(False)
        for name in ("websockets.client", "websockets.server", "websockets.protocol"):
            assert logging.getLogger(name).getEffectiveLevel() == logging.WARNING, name
    finally:
        _reset_third_party_logger_levels()
