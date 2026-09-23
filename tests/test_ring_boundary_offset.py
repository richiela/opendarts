"""Tests for opendarts.calibration.ring_boundary_offset -- the RECORD
half of the per-board ring-boundary offset work: the payload shape a
calibration event writes, and the reader that loads a session file back.

The MEASUREMENT half, and the tests that exercise it against a
synthetic board, moved to dev/ with it (2026-09-17):
`dev/tests/test_ring_boundary_measure.py`.
"""
from __future__ import annotations

from opendarts.calibration.ring_boundary_offset import (
    OFFSET_FILENAME,
    SCHEMA,
    RingBoundaryOffsetResult,
    load_session_ring_boundary_offset,
    result_to_payload,
)


def test_loader_returns_none_when_absent(tmp_path):
    assert load_session_ring_boundary_offset(tmp_path) is None


def test_loader_returns_none_on_corrupt_json_instead_of_raising(tmp_path):
    """Real gap (found during OpenDarts's pre-hardware-audit,
    confirmed to apply here too, 2026-08-21): a truncated/corrupt
    file used to raise json.JSONDecodeError straight out of this
    function -- through opendarts.capture.throw_package.
    load_throw_package(), BEFORE set_ring_boundary_offsets() ever
    ran for that call, defeating the "never leak a prior session's
    offset into a later load" guarantee that caller's own docstring
    promises. Must degrade exactly like "no file at all", not
    crash."""
    (tmp_path / OFFSET_FILENAME).write_text("{not valid json")
    assert load_session_ring_boundary_offset(tmp_path) is None


# ---------------------------------------------------------------------
# THE V2 PACKAGE SCHEMA (2026-08-27) -- result_to_payload() made
# public so
# `opendarts.live.capture_daemon.bootstrap_calibrations()` can build the
# SAME payload shape live, attributed with its own `solved_by`.
# ---------------------------------------------------------------------


def test_result_to_payload_default_solved_by_is_the_offline_description():
    result = RingBoundaryOffsetResult(boundaries={}, calibration_source="x", source_images=["a"])
    payload = result_to_payload(result)
    assert payload["schema"] == SCHEMA
    assert "offline" in payload["solved_by"]
    assert payload["calibration_source"] == "x"
    assert payload["source_images"] == ["a"]
    # Every one of the 11 real solver parameters is present -- the field
    # The v2 package schema's own spec calls "the single most important
    # addition."
    assert set(payload["parameters"].keys()) == {
        "radial_step_mm", "profile_smooth_samples", "min_contrast",
        "min_radial_scale_px_per_mm", "in_sector_angle_offsets_deg",
        "transition_lo", "transition_hi", "median_crossing_sustain_samples",
        "peak_significance_fraction", "peak_contiguity_stop_fraction",
        "min_profiles_per_camera",
    }


def test_result_to_payload_solved_by_override_is_used_verbatim():
    result = RingBoundaryOffsetResult(boundaries={})
    payload = result_to_payload(result, solved_by="a live custom description")
    assert payload["solved_by"] == "a live custom description"
