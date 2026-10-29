"""Argus -- the visual hybrid scorer: each camera reports what it sees at
the needle tip (board paint, map, needle visibility), and the cameras'
fused board position referees the vote. Zeus's fifth voter. See
`opendarts.engines.argus.engine`'s module docstring for the design."""
from opendarts.engines.argus.engine import ArgusEngine

__all__ = ["ArgusEngine"]
