"""Optional LatentSight plugin modules."""

from .three_stage_distill import ThreeStageImaginationDistiller, build_imagination_distiller

__all__ = ["ThreeStageImaginationDistiller", "build_imagination_distiller"]
