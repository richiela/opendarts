"""opendarts/live/cv2_threads.py -- bound OpenCV's worker pool at process
start.

WHY THIS EXISTS, measured on the real Windows rig 2026-09-13. Nothing in
this project ever called ``cv2.setNumThreads()``, so OpenCV sized its own
pool from the core count (8 on that box) and fanned every internal
``parallel_for_`` across it. At this rig's real rates that is roughly 450
dispatches a second -- about five OpenCV calls per camera per tick, three
cameras, a ~30Hz frame-driven loop -- each waking eight workers for a task
measured in tens of microseconds. The pool spent far more time coordinating
than computing.

The measurement, taken live with per-thread CPU attribution
(an external per-thread CPU sampler joined to ``GET /api/threads`` on
``native_id``), same process, no restart between the two samples:

===========================  ==============  ==============
idle, armed, no darts        pool = 8        pool = 1
===========================  ==============  ==============
process CPU per 15s window   35.88 s         13.36 s
% of one core                238.9 %         89.1 %
% of all 8 cores             29.9 %          11.1 %
unnamed pool threads         9, 20.02 s      0, 0 s
our own named threads        11.95 s         10.78 s
===========================  ==============  ==============

The nine-thread band did not shrink, it vanished -- and our own threads got
slightly CHEAPER, so that work never relocated into them. Those twenty
seconds were pure coordination overhead: a 62.8% reduction in total CPU,
150% of one core given back.

WHAT IT COSTS, measured rather than assumed, because this sits on the
scoring path:

* **Scoring correctness: unchanged.** 130 real throw packages replayed
  from the corpus across all five engines (Zeus/Apollo/Talos/Athena/Ares)
  produced ZERO differing ``(sector, ring, ok)`` calls. Expected --
  parallelism splits work, it does not change arithmetic -- but verified
  on real throws rather than argued.
* **Scoring latency: +2 to +4 ms per dart (~3%).** Small but consistent
  across every engine, so it is a real cost and not noise. Set against a
  ~130 ms scoring pass, a dart lands ~3 ms later in exchange for 150% of a
  core.
* **Calibration: no measurable change.** This was the one path expected
  to want the extra threads -- heavy full-frame landmark and ellipse work
  rather than 320x180 scraps -- so it was measured on the real rig with
  the runtime endpoint, 6 pairs alternating stock/pinned within a single
  process and camera session: total 23.12 s stock vs 23.09 s pinned
  (difference +0.02 s, 95% CI +/-0.78 s), and ``best_of_n`` -- the largest
  phase at ~10 s of ~23 s, and the most parallel_for_-heavy -- flat to
  +/-0.27 s, if anything marginally SLOWER pinned, which is the direction
  losing parallelism predicts. So no carve-out is needed around
  ``bootstrap_calibrations()``, but on the grounds that pinning costs
  calibration nothing, NOT that it helps.

  A FIRST ATTEMPT HERE CLAIMED A 10.3% SPEEDUP AND WAS WRONG. It ran one
  calibration per arm, ordered stock/pinned/stock (23.01 / 20.28 /
  22.23 s), and treated the two stock runs bracketing the pinned one as
  evidence the effect was real. Within-arm spread across six runs is
  21.2-26.0 s, so a 2.7 s difference sits inside the noise that design
  cannot resolve; worse, calibration shows a strong warm-up drift
  (~26 s early, ~22 s late, regardless of setting), which alternating
  cancels and a stock/pinned/stock ordering partly aliases into the
  result. Recorded because "the arms bracket each other" reads as
  rigorous and is not, at n=1 per arm against a drifting metric.

WHY ONE AND NOT TWO, decided 2026-09-13 once the numbers were in: there
are no gains in more threads. The images on the detection path are small
enough that a second worker cannot pay back its own synchronisation, and
calibration -- the only genuinely heavy path -- measured faster pinned as
well.

A CONFIG KEY, NOT A CONSTANT, so a rig with different hardware can be
retuned without a code change, and so the experiment stays repeatable.
``cv2_num_threads`` absent from ``data/config.json`` means "no
override", which resolves to :data:`DEFAULT_CV2_NUM_THREADS` -- the same
"None means keep the code-level default" contract every other key in
:mod:`opendarts.live.config` already uses.

NOTE ON CONFIRMING A CHANGE TOOK EFFECT: ``getNumThreads()`` does not
round-trip ``setNumThreads()`` on every backend. The rig builds against
the ``Concurrency`` framework, where it does. A macOS build uses ``GCD``,
where ``setNumThreads(1)`` genuinely halved CPU per unit of work while
``getNumThreads()`` went on reporting the full core count and only ``n=0``
ever moved it. So this module logs the framework alongside the numbers,
and the honest confirmation is measured CPU, never the reported count.
``POST /api/threads/cv2-threads`` changes it at runtime for exactly that
kind of A/B.
"""
from __future__ import annotations

import logging

log = logging.getLogger("opendarts.live.cv2_threads")

#: Threads OpenCV may use when the config file says nothing. 1 by measured
#: result -- see this module's docstring for the full before/after.
DEFAULT_CV2_NUM_THREADS = 1


def parallel_framework() -> str | None:
    """OpenCV's build-time parallel backend ("Concurrency", "GCD", "TBB",
    ...), or None if it cannot be determined. Reported rather than assumed
    because it decides whether ``getNumThreads()`` can be trusted at all."""
    try:
        import cv2
        for line in cv2.getBuildInformation().splitlines():
            if "Parallel framework" in line:
                return line.split(":", 1)[1].strip()
    except Exception: # noqa: BLE001 -- a diagnostic must never break startup
        pass
    return None


def apply_cv2_thread_limit(n: int | None = None) -> int | None:
    """Bound OpenCV's worker pool. ``None`` means "no override configured",
    which applies :data:`DEFAULT_CV2_NUM_THREADS` rather than leaving
    OpenCV's own choice in place -- the whole point is that OpenCV's
    default is the thing being corrected.

    Pass a negative value to deliberately keep OpenCV's default (the same
    convention ``cv2.setNumThreads`` itself uses for "restore default"), so
    a rig can opt out via config without a code change.

    Returns the count reported afterwards, or None if OpenCV is
    unavailable. That return is for logging only -- see the module
    docstring on why it is not proof.

    Never raises: this runs at process start, and a rig that cannot tune
    its thread pool must still boot and score darts.
    """
    if n is None:
        n = DEFAULT_CV2_NUM_THREADS
    try:
        import cv2
    except Exception as exc: # noqa: BLE001
        log.warning("cv2 unavailable, leaving thread pool untouched (%s)", exc)
        return None

    framework = parallel_framework()
    try:
        before = int(cv2.getNumThreads())
        if n < 0:
            log.info(
                "cv2 thread pool left at OpenCV's default (%d, framework=%s) by explicit config",
                before, framework,
            )
            return before
        cv2.setNumThreads(n)
        after = int(cv2.getNumThreads())
        log.info(
            "cv2.setNumThreads(%d): getNumThreads %d -> %d (framework=%s). "
            "Measured on the rig: this is worth ~150%% of one core at idle; "
            "the reported count is not proof on every backend.",
            n, before, after, framework,
        )
        return after
    except Exception: # noqa: BLE001 -- must never prevent startup
        log.exception("failed to set cv2 thread count to %r -- continuing with OpenCV's default", n)
        return None
