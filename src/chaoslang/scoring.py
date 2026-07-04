"""Compatibility module; package import resolves to chaoslang.scoring package."""
from chaoslang.scoring.mdl import CodeLengthConfig, SimpleMDLScorer, TwoPartMDLScorer

__all__ = ["CodeLengthConfig", "SimpleMDLScorer", "TwoPartMDLScorer"]
