"""Zeus -- a majority-vote COMBINER over four opendarts engines (Apollo,
Talos, Athena, Ares; quorum `MIN_SUB_ENGINES_TO_VOTE = 3`), not its own
detector/calibration consumer. See `engine.py`'s own module docstring for the full design and
the exact tie-break convention (deliberately preserved from the original
scratch prototype, not reinvented).
"""
from __future__ import annotations

from opendarts.engines.zeus.engine import ZEUS_SUB_ENGINE_NAMES, ZeusEngine

__all__ = ["ZeusEngine", "ZEUS_SUB_ENGINE_NAMES"]
