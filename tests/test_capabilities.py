"""Tests for opendarts/live/capabilities.py -- the per-rig external-tool
probe -- and for the call sites it gates (calibration raw-video encode,
macOS camera enumeration, git provenance).

AUDIO IS NO LONGER ONE OF THEM, as of 2026-09-15. The probe carried
`say`, `afplay`, `aplay`, `paplay` and a stdlib-import check for
`winsound` purely to gate playback and rendering on a rig. Spoken calls
now play in the browser, so nothing in this product synthesises or plays
a sound and there is nothing left for those probes to gate -- they were
removed rather than left publishing facts in /api/health that the
product has no opinion about.

The two properties that matter most, per the module's own docstring:

  1. A persisted "absent" can NEVER permanently disable a feature: every
     process re-probes fresh on first use, so installing a tool comes
     back after (at worst) one restart. The persisted record is a
     record, not an authority.
  2. Gated call sites SPAWN NOTHING when the tool is absent -- proven
     here by replacing subprocess entry points with raising stubs, not
     by inspecting log output.

Note the autouse `_isolate_capability_probe` fixture in conftest.py:
every test here starts with an empty memo and a tmp-file config path.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from opendarts.capture import calibration_package as calib_pkg
from opendarts.live import camera_names, capabilities
from opendarts.live.server import create_app
from opendarts.pipeline import CameraCalibration, PnpResult


def _which_stub(monkeypatch, present: dict[str, str]):
    """shutil.which replacement: `present` maps tool -> fake path; every
    other name is absent. Returns the call recorder."""
    calls: list[str] = []

    def fake_which(name, *a, **k):
        calls.append(name)
        return present.get(name)

    monkeypatch.setattr(capabilities.shutil, "which", fake_which)
    return calls


def _no_spawn(monkeypatch, module):
    """Replace `module.subprocess.run`/`.Popen` with raising stubs and
    return the (shared) attempt recorder. Any spawn is a test failure at
    the exact line that attempted it."""
    attempts: list[list[str]] = []

    def boom(cmd, *a, **k):
        attempts.append(list(cmd))
        raise AssertionError(f"subprocess spawned for {cmd!r}")

    monkeypatch.setattr(module.subprocess, "run", boom)
    monkeypatch.setattr(module.subprocess, "Popen", boom)
    return attempts


# ---------------------------------------------------------------------------
# The probe module itself


def test_first_use_probes_and_persists_a_record(monkeypatch):
    _which_stub(monkeypatch, {"git": "/fake/bin/git"})
    assert capabilities.has("git") is True
    assert capabilities.tool_path("git") == "/fake/bin/git"

    raw = json.loads(capabilities.CONFIG_PATH.read_text())
    record = raw[capabilities.CONFIG_SECTION]
    assert record["tools"]["git"] == {"present": True, "path": "/fake/bin/git"}
    assert record["platform"] == capabilities.platform.system()
    assert record["probed_at_utc"]
    # Every registered executable appears -- the record is the whole
    # answer, not just whichever tool happened to be asked about first.
    for name in capabilities.WHICH_TOOLS:
        assert name in record["tools"]


def test_probe_runs_once_per_process(monkeypatch):
    calls = _which_stub(monkeypatch, {})
    for _ in range(3):
        capabilities.has("git")
        capabilities.snapshot()
    # One which() per registered executable, total -- not per query.
    assert sorted(calls) == sorted(capabilities.WHICH_TOOLS)


def test_cached_absent_does_not_outlive_an_install(monkeypatch):
    """THE staleness test. A rig probed without git carries
    {"present": false} in its config; git is then installed. The next
    process start must answer True and correct the record -- a persisted
    false that wins over reality is the one unacceptable design."""
    _which_stub(monkeypatch, {})
    assert capabilities.has("git") is False
    stored = json.loads(capabilities.CONFIG_PATH.read_text())
    assert stored[capabilities.CONFIG_SECTION]["tools"]["git"]["present"] is False

    # "git gets installed, the rig restarts": new process memo, same
    # config file on disk still saying absent.
    capabilities._probed = None
    _which_stub(monkeypatch, {"git": "/fake/bin/git"})
    assert capabilities.has("git") is True
    stored = json.loads(capabilities.CONFIG_PATH.read_text())
    assert stored[capabilities.CONFIG_SECTION]["tools"]["git"]["present"] is True


def test_unchanged_record_is_not_rewritten(monkeypatch):
    _which_stub(monkeypatch, {"ffmpeg": "/fake/bin/ffmpeg"})
    capabilities.snapshot()
    before = capabilities.CONFIG_PATH.read_text()

    writes: list[str] = []
    real_write = capabilities.write_config_section

    def counting_write(section, value, path):
        writes.append(section)
        return real_write(section, value, path=path)

    monkeypatch.setattr(capabilities, "write_config_section", counting_write)
    capabilities._probed = None  # "next process", identical machine
    capabilities.snapshot()
    assert writes == []
    assert capabilities.CONFIG_PATH.read_text() == before


def test_corrupt_config_section_degrades_to_probe(monkeypatch):
    _which_stub(monkeypatch, {"ffmpeg": "/fake/bin/ffmpeg"})
    capabilities.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    capabilities.CONFIG_PATH.write_text(
        json.dumps({"capabilities": "an operator typed this", "port": 8420})
    )
    assert capabilities.has("ffmpeg") is True
    raw = json.loads(capabilities.CONFIG_PATH.read_text())
    assert isinstance(raw["capabilities"], dict)
    # write_config_section's contract: every other key survives.
    assert raw["port"] == 8420


def test_unparseable_config_file_never_crashes_or_clobbers(monkeypatch):
    _which_stub(monkeypatch, {"ffmpeg": "/fake/bin/ffmpeg"})
    capabilities.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    capabilities.CONFIG_PATH.write_text("{this is not json")
    # The probe answer stands; the broken file is REPORTED, not replaced
    # (write_config_section's own refusal contract).
    assert capabilities.has("ffmpeg") is True
    assert capabilities.CONFIG_PATH.read_text() == "{this is not json"


def test_refresh_is_the_mid_process_escape_hatch(monkeypatch):
    _which_stub(monkeypatch, {})
    assert capabilities.has("git") is False
    _which_stub(monkeypatch, {"git": "/fake/bin/git"})
    # Without refresh the memo answers -- that is the documented window.
    assert capabilities.has("git") is False
    record = capabilities.refresh()
    assert capabilities.has("git") is True
    assert record["tools"]["git"]["present"] is True


def test_probe_module_never_spawns_anything(monkeypatch):
    """The probe is which() only -- proven by construction (the module
    never imports subprocess) and by a global raising stub. It used to
    also do a stdlib import, for `winsound`; that branch went with the
    audio probes on 2026-09-15."""
    assert not hasattr(capabilities, "subprocess")

    def boom(*a, **k):
        raise AssertionError("capabilities spawned a subprocess")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    capabilities.refresh()


def test_the_audio_probes_are_gone(monkeypatch):
    """Nothing in this product plays or renders a sound any more, so a
    probe for a player would be publishing a capability the product has
    no opinion about -- and /api/health would invite an operator to
    install alsa-utils on a rig that will never use it.

    `winsound` took the entire stdlib-import branch with it: it was the
    only entry IMPORT_TOOLS ever had."""
    _which_stub(monkeypatch, {})
    for gone in ("say", "afplay", "aplay", "paplay", "winsound"):
        assert gone not in capabilities.WHICH_TOOLS
        assert gone not in capabilities.snapshot()["tools"]
    assert not hasattr(capabilities, "IMPORT_TOOLS")


def test_unknown_tool_is_probed_live_and_not_persisted(monkeypatch):
    _which_stub(monkeypatch, {"definitely-not-registered": "/fake/tool"})
    assert capabilities.has("definitely-not-registered") is True
    raw = json.loads(capabilities.CONFIG_PATH.read_text())
    assert "definitely-not-registered" not in raw[capabilities.CONFIG_SECTION]["tools"]


def test_which_raising_reads_as_absent_not_a_crash(monkeypatch):
    def broken_which(name, *a, **k):
        raise OSError("PATH env is garbage")

    monkeypatch.setattr(capabilities.shutil, "which", broken_which)
    assert capabilities.has("ffmpeg") is False


# ---------------------------------------------------------------------------
# Gated call sites: calibration package


def _fake_calibration() -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rvec=np.zeros(3),
        tvec=np.array([0.0, 0.0, 1000.0]),
        pnp_result=PnpResult(
            ok=True, rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 1000.0]),
            reprojection_error_px=1.23,
        ),
        landmark_spread_ok=True,
    )


def test_calibration_raw_video_encode_spawns_no_subprocess(monkeypatch, tmp_path):
    """The owner's old complaint: the Windows PC reported 'raw video
    encode failed cause there's no FFMPEG' on every calibration. That
    failure mode is gone -- FFV1 now goes through cv2.VideoWriter (codec
    bundled in the opencv wheel), which is in-process, so a calibration
    save spawns NO external process for raw video AND the .mkv files are
    actually written on every platform."""
    attempts = _no_spawn(monkeypatch, calib_pkg)

    frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 2
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_gate", {0: frames, 1: frames}, {0: _fake_calibration(), 1: _fake_calibration()}
    )
    assert (package_dir / "derived_calibration.json").exists()
    assert (package_dir / "cam0_raw.mkv").exists()
    assert (package_dir / "cam1_raw.mkv").exists()
    meta = json.loads((package_dir / "meta.json").read_text())
    for cam in ("0", "1"):
        assert meta["cameras"][cam]["raw_video"] == f"cam{cam}_raw.mkv"
        assert meta["cameras"][cam]["error"] is None
    # No external process spawned for encoding (git provenance, gated
    # separately, is the only thing that may spawn).
    assert [a for a in attempts if a and a[0] != "git"] == []


def test_code_version_spawns_nothing_without_git(monkeypatch):
    calib_pkg._code_version.cache_clear()
    monkeypatch.setattr(capabilities, "has", lambda name: False)
    attempts = _no_spawn(monkeypatch, calib_pkg)
    try:
        assert calib_pkg._code_version() is None
        assert attempts == []
    finally:
        calib_pkg._code_version.cache_clear()


# ---------------------------------------------------------------------------
# macOS camera enumeration: system_profiler only, never ffmpeg


def test_macos_enumeration_never_spawns_ffmpeg(monkeypatch):
    """The whole point of the 2026-09-20 change: camera-name enumeration
    consults `system_profiler` and nothing else -- ffmpeg is never spawned
    (nor even probed for) on any code path."""
    spawned: list[list[str]] = []

    def fake_run(cmd, **k):
        spawned.append(list(cmd))
        raise OSError("real spawn blocked in test")

    monkeypatch.setattr(camera_names.subprocess, "run", fake_run)
    enum = camera_names._enumerate_macos()
    assert enum.source == "none"
    assert [c[0] for c in spawned] == ["system_profiler"]
    assert all(c[0] != "ffmpeg" for c in spawned)


# ---------------------------------------------------------------------------
# Diagnostics surface


def test_api_health_reports_capabilities(monkeypatch, tmp_path):
    _which_stub(monkeypatch, {"git": "/fake/bin/git"})
    pkg_root = tmp_path / "packages"
    pkg_root.mkdir()
    app = create_app(package_root=pkg_root, enable_background_poll=False)
    body = TestClient(app).get("/api/health").json()
    assert body["status"] == "ok"
    caps = body["capabilities"]
    assert caps["tools"]["git"]["present"] is True
    # ffmpeg is no longer a probed tool -- nothing shells out to it.
    assert "ffmpeg" not in caps["tools"]
    assert caps["platform"]
