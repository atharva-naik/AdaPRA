"""OpenHands-based harness utilities for DSGym trajectory collection."""

from __future__ import annotations

# Imported lazily so lightweight consumers (e.g. the in-container PRA) do not pull
# in dependencies they do not need.
__all__ = ["PRMScorer"]


def __getattr__(name: str):
    if name == "PRMScorer":
        from .prm_feedback import PRMScorer
        return PRMScorer
    raise AttributeError(name)
