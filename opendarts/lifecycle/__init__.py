"""Throw lifecycle -- the layer between the camera pump and the scoring
engines.

Decides, every frame, whether a dart has landed (and should be handed to
the engines), whether a hand is in the scene, and whether the visit's
darts have been taken out. This is the only trigger: the episode-based
``opendarts.capture.throw_trigger`` state machine it replaced has been
deleted. Per-tick, count-based design:

* ``signals``   -- per camera, per frame: changed pixels inside/outside
                   the board region vs a live reference, and vs the
                   previous frame (stability). Pure functions.
* ``reference`` -- the per-camera residual background policy and the
                   committed-dart mask bookkeeping used for takeout.
* ``state``     -- the state machine. A pure function of the current
                   signals plus small frame counters. Hand has priority
                   over everything; commits are debounced; actions are
                   followed by a cooldown during which the reference is
                   re-adopted every tick.
* ``driver``    -- the only integration point with the rest of opendarts
                   (hands (bg, frame) pairs to the existing scoring
                   path, emits events, writes the lifecycle log).

Design principles (see docs/LIFECYCLE.md):
  1. decide every frame from current counts, never from an episode;
  2. counts, not shapes -- shape judgement belongs to the engines;
  3. hand is a suppressor with priority, not a gate passed once;
  4. one reference per camera, kept live;
  5. takeout is about the darts we committed, not a startup frame;
  6. debounce with frame counts; cool down after every action.
"""
