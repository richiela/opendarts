"""opendarts.live.heap_trim -- trims on glibc, a no-op everywhere else, and
never raises (it runs right after a calibration that already succeeded)."""
from __future__ import annotations

import sys

import pytest

from opendarts.live import heap_trim


@pytest.fixture(autouse=True)
def _fresh_resolution(monkeypatch):
    monkeypatch.setattr(heap_trim, "_RESOLVED", False)
    monkeypatch.setattr(heap_trim, "_MALLOC_TRIM", None)


def test_release_freed_heap_calls_malloc_trim_zero_when_available(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(heap_trim, "_malloc_trim", lambda: calls.append)
    assert heap_trim.release_freed_heap("test") is True
    assert calls == [0]


def test_release_freed_heap_is_a_noop_without_malloc_trim(monkeypatch):
    monkeypatch.setattr(heap_trim, "_malloc_trim", lambda: None)
    assert heap_trim.release_freed_heap("test") is False


def test_release_freed_heap_never_raises(monkeypatch):
    def boom(_pad):
        raise OSError("simulated")

    monkeypatch.setattr(heap_trim, "_malloc_trim", lambda: boom)
    assert heap_trim.release_freed_heap("test") is False


def test_malloc_trim_resolves_only_on_linux():
    fn = heap_trim._malloc_trim() # noqa: SLF001
    if sys.platform.startswith("linux"):
        assert fn is not None
        assert heap_trim.release_freed_heap("test") is True
    else:
        assert fn is None
