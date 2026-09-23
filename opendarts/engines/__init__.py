"""opendarts/engines/ -- pluggable multi-engine scoring framework.

See docs/ENGINES.md for the full spec this package implements. Short
version: an "engine" is anything implementing the `Engine` protocol in
`opendarts.engines.base` (a `score()` method, optionally `calibrate()`).
Today there are three: `Apollo` (opendarts.engines.apollo -- wraps the
existing detect_tip()+score_dart() pipeline UNCHANGED), `Talos`
(opendarts.engines.talos -- 3D shaft as plane-intersection, line ∩ Z=0),
and `Athena` (opendarts.engines.athena -- see its own module docstring
for what it does differently). See `opendarts.engines.registry.ENGINES`
for the authoritative, currently-registered list.

Nothing in this package touches cameras, the trigger/settle state
machine, or package persistence directly -- see opendarts.engines.dispatch
for the concurrent-dispatch-with-timeout mechanism that calls engines
from opendarts.live.capture_daemon, and opendarts.engines.registry for the
name -> engine lookup used by both the live capture path and the offline
tools (opendarts/capture/replay.py, opendarts/capture/rescore_all.py).
"""
from __future__ import annotations
