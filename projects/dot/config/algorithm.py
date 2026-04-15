"""
Project-local AlgoConfig extension for DOT.

This keeps core AlgoConfig untouched while allowing custom fields used
by DOT (dynamic sampling, truncation, etc.).
"""

from dataclasses import dataclass, field
from typing import Any

from verl.trainer.config import AlgoConfig

__all__ = ["DotAlgoConfig"]


@dataclass
class DotAlgoConfig(AlgoConfig):
    """Algorithm config that supports DOT-only knobs.

    We keep the base AlgoConfig fields and add a handful of dict slots that
    the DOT trainer reads. Keeping them as dicts avoids tight coupling
    to core config schema.
    """

    _target_: str = "projects.dot.config.algorithm.DotAlgoConfig"

    filter_groups: dict[str, Any] = field(default_factory=dict)

    length_truncation: dict[str, Any] = field(default_factory=dict)
