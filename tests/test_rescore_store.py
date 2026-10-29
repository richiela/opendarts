"""opendarts.capture.rescore_store: every package keeps its score from each
code version next to the live result, in commit order, so "is it still
wrong?" and "which version fixed it?" are answered from the package."""
from __future__ import annotations

import json

from opendarts.capture import rescore_store as rs


def _entry(version, committed, zeus_ok, apollo_ok=True):
    return {"code_version": version, "code_committed_at": committed,
            "scored_at_utc": "2026-10-08T00:00:00+00:00",
            "truth": {"token": "S20", "source": "oracle"},
            "correct": {"Zeus": zeus_ok, "Apollo": apollo_ok}}


def test_truth_is_the_operators_correction_then_the_oracles_matched_call():
    assert rs.truth_of({"matched": True, "sector": "20", "ring": "treble"}) == ("T20", "oracle")
    assert rs.truth_of({"matched": True, "sector": "20", "ring": "treble",
                        "operator_confirmed_sector": "5", "operator_confirmed_ring": "single_outer"}) == ("S5", "operator")
    assert rs.truth_of({"matched": True, "sector": "20", "ring": "treble",
                        "operator_marked_wrong": True}) == (None, None)
    assert rs.truth_of({"matched": False}) == (None, None)
    assert rs.truth_of(None) == (None, None)
    assert rs.truth_of({"matched": True, "sector": None, "ring": "outside"}) == ("OUT", "oracle")


def test_inner_and_outer_singles_are_the_same_call():
    assert rs._token("20", "single_inner") == rs._token("20", "single_outer") == "S20"
    assert rs._token("20", "treble", ok=False) is None


def test_history_is_kept_in_commit_order_and_a_version_is_replaced_not_duplicated(tmp_path):
    rs.save_entry(tmp_path, _entry("new", "2026-10-07T20:00:00-07:00", True))
    rs.save_entry(tmp_path, _entry("old", "2026-09-28T10:00:00-07:00", False))   # backfilled later
    rs.save_entry(tmp_path, _entry("new", "2026-10-07T20:00:00-07:00", True))    # same version again
    hist = rs.load_history(tmp_path)
    assert [h["code_version"] for h in hist] == ["old", "new"]
    assert json.loads((tmp_path / "rescore.json").read_text())["schema"] == rs.SCHEMA


def test_the_version_that_fixed_a_dart(tmp_path):
    d = tmp_path
    for v, t, ok in (("c", "2026-10-01T00:00:00Z", True), ("a", "2026-09-20T00:00:00Z", False),
                     ("b", "2026-09-28T00:00:00Z", False), ("d", "2026-10-07T00:00:00Z", True)):
        rs.save_entry(d, _entry(v, t, ok))
    assert rs.first_right_version(d) == "c"
    assert rs.first_right_version(d, "Apollo") is None          # never wrong


def test_summary_counts_accuracy_and_what_changed_since_the_previous_version(tmp_path):
    a, b = tmp_path / "s" / "a", tmp_path / "s" / "b"
    for p in (a, b):
        p.mkdir(parents=True)
    rs.save_entry(a, _entry("v1", "2026-10-01T00:00:00Z", False))
    rs.save_entry(a, _entry("v2", "2026-10-02T00:00:00Z", True))
    rs.save_entry(b, _entry("v1", "2026-10-01T00:00:00Z", True))
    rs.save_entry(b, _entry("v2", "2026-10-02T00:00:00Z", False, apollo_ok=False))
    s = rs.summarize([a, b], "v2")
    assert s["right"]["Zeus"] == 1 and s["accuracy"]["Apollo"] == 0.5
    assert s["zeus_fixed_since_previous"] == ["a"] and s["zeus_broken_since_previous"] == ["b"]


def test_commits_that_share_a_commit_time_are_ordered_by_depth(tmp_path):
    t = "2026-09-28T17:19:23-07:00"
    for v, depth, ok in (("c", 312, True), ("a", 310, False), ("b", 311, False)):
        e = _entry(v, t, ok)
        e["code_depth"] = depth
        rs.save_entry(tmp_path, e)
    assert [h["code_version"] for h in rs.load_history(tmp_path)] == ["a", "b", "c"]
    assert rs.first_right_version(tmp_path) == "c"
