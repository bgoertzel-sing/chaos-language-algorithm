"""Scoring interfaces for chaoslang."""

from .mdl import CodeLengthConfig, SimpleMDLScorer, TwoPartMDLScorer

CompressionScorer = SimpleMDLScorer

__all__ = ["CodeLengthConfig", "SimpleMDLScorer", "TwoPartMDLScorer", "CompressionScorer"]
