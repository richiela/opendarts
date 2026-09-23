"""Registering the DirectShow virtual cameras alongside the oracle toggle.

The behaviour under test is mostly about ORDER and about not throwing:
the module shells out to regsvr32, which cannot run here, so every test
fakes the subprocess and asserts on what would have been run.
"""
from __future__ import annotations

import subprocess

import pytest

from opendarts.live import vcam_register


@pytest.fixture
def runs(monkeypatch, tmp_path):
    """Capture regsvr32 invocations instead of running them.

    Also forces `available()` true: these assertions are about the
    Windows behaviour, and the tests would otherwise all pass vacuously
    by taking the not-applicable branch on the dev machine.
    """
    dll = tmp_path / "vcam_probe.dll"
    dll.write_bytes(b"MZ")
    monkeypatch.setattr(vcam_register, "DLL_PATH", dll)
    monkeypatch.setattr(vcam_register.platform, "system", lambda: "Windows")

    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(vcam_register.subprocess, "run", fake_run)
    return calls


def test_register_always_unregisters_first(runs):
    """The explicit requirement, and not merely tidiness.

    Registration is registry state that outlives the process, so the
    machine can be carrying one made by an older build, a different
    checkout path, or a machine-wide registration from before the
    per-user hive was used. Registering on top of that leaves devices
    whose InprocServer32 points where the DLL no longer is: they appear
    in every capture application's list and fail to open.
    """
    result = vcam_register.register()

    assert result["ok"] is True
    assert len(runs) == 2, "expected exactly an unregister then a register"
    assert "/u" in runs[0], "the FIRST call must be the unregister"
    assert "/u" not in runs[1], "the SECOND call must be the register"


def test_a_failed_unregister_does_not_stop_the_register(runs, monkeypatch):
    """Nothing registered yet is the normal first-run state, and reports
    as a failure. Treating it as fatal would mean the very first
    enable never registered anything."""
    codes = iter([1, 0])          # unregister fails, register succeeds

    def fake_run(cmd, **kwargs):
        runs.append(list(cmd))
        return subprocess.CompletedProcess(cmd, next(codes), stdout="", stderr="")

    monkeypatch.setattr(vcam_register.subprocess, "run", fake_run)

    assert vcam_register.register()["ok"] is True
    assert len(runs) == 2


def test_apply_follows_the_toggle_in_both_directions(runs):
    vcam_register.apply(True)
    assert len(runs) == 2 and "/u" in runs[0] and "/u" not in runs[1]

    runs.clear()
    vcam_register.apply(False)
    assert len(runs) == 1 and "/u" in runs[0], "off must only unregister"


def test_a_failing_regsvr32_reports_rather_than_raises(runs, monkeypatch):
    """This sits on a control an operator hits mid-session. A registry
    write that fails must leave the toggle working and say so."""
    def fake_run(cmd, **kwargs):
        runs.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 5, stdout="", stderr="")

    monkeypatch.setattr(vcam_register.subprocess, "run", fake_run)

    result = vcam_register.register()
    assert result["ok"] is False
    assert result["applicable"] is True
    assert "5" in result["reason"]


def test_a_real_failure_is_distinguishable_from_not_applicable(runs, monkeypatch):
    """The pair that has to stay apart: `ok` alone cannot mean both
    "nothing to do here" and "the registry write failed", because the
    first is every Mac and the second is a Windows rig with broken
    virtual cameras. `applicable` is what separates them."""
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 5, stdout="", stderr="")

    monkeypatch.setattr(vcam_register.subprocess, "run", fake_run)
    failed = vcam_register.register()

    monkeypatch.setattr(vcam_register.platform, "system", lambda: "Darwin")
    inert = vcam_register.register()

    assert (failed["ok"], failed["applicable"]) == (False, True)
    assert (inert["ok"], inert["applicable"]) == (True, False)


def test_a_missing_regsvr32_reports_rather_than_raises(runs, monkeypatch):
    def boom(cmd, **kwargs):
        raise OSError("regsvr32 not found")

    monkeypatch.setattr(vcam_register.subprocess, "run", boom)

    result = vcam_register.register()
    assert result["ok"] is False
    assert "OSError" in result["reason"]


def test_a_hung_regsvr32_is_bounded_rather_than_blocking_forever(runs, monkeypatch):
    """A COM server wedged inside DllRegisterServer must not park the
    caller indefinitely -- this runs off the event loop, but a thread
    held forever is still a thread never returned."""
    def hang(cmd, **kwargs):
        assert kwargs.get("timeout") == vcam_register.TIMEOUT_S, \
            "regsvr32 must be invoked with a timeout"
        raise subprocess.TimeoutExpired(cmd, vcam_register.TIMEOUT_S)

    monkeypatch.setattr(vcam_register.subprocess, "run", hang)

    result = vcam_register.register()
    assert result["ok"] is False
    assert "TimeoutExpired" in result["reason"]


def test_off_windows_is_a_no_op_that_says_so(monkeypatch, tmp_path):
    dll = tmp_path / "vcam_probe.dll"
    dll.write_bytes(b"MZ")
    monkeypatch.setattr(vcam_register, "DLL_PATH", dll)
    monkeypatch.setattr(vcam_register.platform, "system", lambda: "Darwin")

    def fail(*a, **k):
        raise AssertionError("nothing may be run off Windows")

    monkeypatch.setattr(vcam_register.subprocess, "run", fail)

    for result in (vcam_register.register(), vcam_register.unregister()):
        assert result["applicable"] is False
        assert "Windows" in result["reason"]
        # ok is TRUE: nothing was attempted, so nothing failed. Reporting
        # False here made the healthy state of every Mac and Linux box
        # indistinguishable from a registry write that genuinely blew up.
        assert result["ok"] is True, "not-applicable is not a failure"


def test_a_missing_dll_is_reported_by_path(monkeypatch, tmp_path):
    """Naming the path it looked for is the difference between 'the
    feature is broken' and 'this checkout has no DLL in it'."""
    monkeypatch.setattr(vcam_register, "DLL_PATH", tmp_path / "absent.dll")
    monkeypatch.setattr(vcam_register.platform, "system", lambda: "Windows")

    result = vcam_register.register()
    assert result["applicable"] is False
    assert result["ok"] is True, "not-applicable is not a failure"
    assert "absent.dll" in result["reason"]


def test_the_shipped_dll_is_where_this_module_looks_for_it():
    """Pins the module against the committed DLL. A move of either one
    without the other is silent: registration simply reports
    not-applicable forever, on the machines where it is the whole point.
    """
    assert vcam_register.DLL_PATH.is_file(), \
        f"no DLL at {vcam_register.DLL_PATH}"
    assert vcam_register.DLL_PATH.read_bytes()[:2] == b"MZ", "not a PE binary"
