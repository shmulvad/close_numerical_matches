"""Finds close numerical matches across two arrays, fast."""

from __future__ import annotations

from close_numerical_matches.find_matches import METRICS
from close_numerical_matches.find_matches import Distances
from close_numerical_matches.find_matches import Matches
from close_numerical_matches.find_matches import MatchIndex
from close_numerical_matches.find_matches import Metric
from close_numerical_matches.find_matches import find_matches
from close_numerical_matches.find_matches import naive_find_matches
from close_numerical_matches.version import __version__

__all__ = (
    'METRICS',
    'Distances',
    'MatchIndex',
    'Matches',
    'Metric',
    '__version__',
    'find_matches',
    'naive_find_matches',
)
