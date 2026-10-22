"""Stubs for the compiled Rust extension. Use the wrappers in the package instead."""

from __future__ import annotations

from typing import Any
from typing import final

import numpy as np
from numpy.typing import NDArray

__version__: str
METRIC_NAMES: list[str]

def find_matches(
    arr0: NDArray[np.floating[Any]],
    arr1: NDArray[np.floating[Any]],
    dist: str,
    tol: float,
    p: float | None,
    n_threads: int | None,
    return_distances: bool,
    naive: bool,
) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.floating[Any]]]: ...

@final
class MatchIndex:
    def __init__(
        self,
        arr: NDArray[np.floating[Any]],
        dist: str,
        p: float | None,
        n_threads: int | None,
    ) -> None: ...
    def query(
        self,
        other: NDArray[np.floating[Any]],
        tol: float,
        n_threads: int | None,
        return_distances: bool,
    ) -> NDArray[np.int64] | tuple[NDArray[np.int64], NDArray[np.floating[Any]]]: ...
    @property
    def n_points(self) -> int: ...
    @property
    def n_dims(self) -> int: ...
    @property
    def dist(self) -> str: ...
    @property
    def p(self) -> float | None: ...
    @property
    def dtype(self) -> str: ...
