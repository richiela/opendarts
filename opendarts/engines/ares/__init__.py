"""Ares -- a greenfield 2D-per-camera scoring engine built around
board-plane shaft-LINE concurrency (see `opendarts.engines.ares.engine`'s
module docstring for the full design, measurements, and rejected
alternatives). Not a fork of Athena: different detector
(thin-run centerline fit that excludes the flight's occlusion-diff mass),
different fusion (IRLS least-squares concurrency of per-camera shaft
lines at z=0, no millimetre plane/radial fudges), no label-override
ladder."""
from opendarts.engines.ares.engine import AresEngine

__all__ = ["AresEngine"]
