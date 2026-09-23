"""The Info tab's RAM row: memory_info() reads real physical memory, and
the /api/state config carries it beside disk.

memory_info reads each platform's own source (no psutil), so the value it
returns depends on the platform the tests run on -- these assert the SHAPE
and the invariants that must hold everywhere, not a specific number, plus
that a failure is reported as None rather than a fabricated zero.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from opendarts.live.server import _memory_dict, cpu_load, create_app, memory_info


def test_memory_info_reads_real_memory_on_this_platform():
    m = memory_info()
    # Every platform the fleet runs on (Linux/macOS/Windows) has a source;
    # if this returns None here, the probe is broken, not the machine.
    assert m is not None, "memory_info returned None on the test platform"
    assert m["total_bytes"] > 0
    assert 0 <= m["available_bytes"] <= m["total_bytes"]
    assert m["used_bytes"] == m["total_bytes"] - m["available_bytes"]
    assert 0 <= m["used_pct"] <= 100
    # The probe names how it measured, so a wrong number is traceable to a
    # source rather than guessed at.
    assert m["source"]
    assert isinstance(m["available_is_estimate"], bool)


def test_memory_dict_clamps_a_bad_available_reading():
    # vm_stat pages can momentarily sum past total between two reads; the
    # row must never show negative "used" or over-100% from that.
    over = _memory_dict(total=8, available=99, source="test", estimate=True)
    assert over["available_bytes"] == 8
    assert over["used_bytes"] == 0
    assert over["used_pct"] == 0.0

    under = _memory_dict(total=8, available=0, source="test", estimate=False)
    assert under["used_bytes"] == 8
    assert under["used_pct"] == 100.0


def test_cpu_load_reads_and_diffs_between_calls():
    # _cpu_prev is module state that any earlier test hitting /api/state will
    # have populated, so clear it to make "the first call has no baseline"
    # deterministic rather than test-order-dependent.
    from opendarts.live import server as _srv
    _srv._cpu_prev.clear()
    # First call has no baseline -> busy_pct None, but the shape and cores
    # must still be honest (never a fabricated percentage).
    first = cpu_load()
    assert first is not None, "cpu_load returned None on the test platform"
    assert first["busy_pct"] is None
    assert first["cores"] and first["cores"] > 0
    assert first["source"]
    # A second call after some work has a baseline; if the window is non-zero
    # it reports a real percentage in range, else honestly stays None.
    x = 0
    for i in range(2_000_000):
        x += i
    second = cpu_load()
    assert second is not None
    if second["busy_pct"] is not None:
        assert 0 <= second["busy_pct"] <= 100


def test_state_config_carries_the_load_trio():
    client = TestClient(
        create_app(package_root=Path(tempfile.mkdtemp()),
                   enable_background_poll=False))
    cfg = client.get("/api/state").json()["config"]
    # The Load block's three sources, together in one payload.
    for key in ("cpu", "memory", "disk"):
        assert key in cfg, f"{key} missing from /api/state config"
    assert cfg["memory"] is not None and cfg["memory"]["total_bytes"] > 0
    assert cfg["cpu"] is not None and cfg["cpu"]["cores"] > 0
