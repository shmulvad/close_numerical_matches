"""Typed Python front end of the Rust core in `close_numerical_matches._core`."""

from __future__ import annotations

from typing import Any
from typing import Literal
from typing import TypeAlias
from typing import cast
from typing import final
from typing import overload

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray

from close_numerical_matches import _core

Metric: TypeAlias = Literal[
    'euclidean',
    'sqeuclidean',
    'manhattan',
    'chebyshev',
    'minkowski',
    'cosine',
    'norm',
    'l2',
    'cityblock',
    'l1',
    'max',
    'linf',
    'cos',
]
"""
The supported distance metrics. Aliases: `norm`/`l2` (euclidean), `cityblock`/`l1`
(manhattan), `max`/`linf` (chebyshev) and `cos` (cosine).
"""

Matches: TypeAlias = NDArray[np.int64]
"""A `(k, 2)` array of matching row indices `[i, j]`, sorted lexicographically."""

Distances: TypeAlias = NDArray[np.floating[Any]]
"""A `(k,)` array with the distance of each match."""

FloatDType: TypeAlias = type[np.float32] | type[np.float64]

METRICS: tuple[str, ...] = tuple(_core.METRIC_NAMES)


def _check_dist(dist: object) -> str:
    if isinstance(dist, str):
        return dist
    if callable(dist):
        raise TypeError(
            'Custom distance functions are not supported since version 0.3.0. '
            f'Use one of the built-in metrics: {", ".join(METRICS)}',
        )
    raise TypeError(f'dist must be a string, got {type(dist).__name__}')


def _as_2d(arr: ArrayLike, name: str) -> NDArray[Any]:
    """`arr` as a 2-D array of real numbers; a 1-D array becomes a single column."""
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    elif a.ndim != 2:
        raise ValueError(f'{name} must be a 1-D or 2-D array, got {a.ndim} dimensions')
    if a.dtype.kind not in 'biuf':
        raise TypeError(f'{name} must contain real numbers, got dtype {a.dtype}')
    return a


def _float_dtype(*arrays: NDArray[Any]) -> FloatDType:
    """float32 if every array is float32 (or float16), otherwise float64."""
    if all(a.dtype.kind == 'f' and a.dtype.itemsize <= 4 for a in arrays):
        return np.float32
    return np.float64


def _find(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric,
    tol: float,
    p: float | None,
    n_threads: int | None,
    return_distances: bool,
    naive: bool,
) -> Matches | tuple[Matches, Distances]:
    dist = cast('Metric', _check_dist(dist))
    a0, a1 = _as_2d(arr0, 'arr0'), _as_2d(arr1, 'arr1')
    dtype = _float_dtype(a0, a1)
    return _core.find_matches(
        np.ascontiguousarray(a0, dtype=dtype),
        np.ascontiguousarray(a1, dtype=dtype),
        dist,
        tol,
        p,
        n_threads,
        return_distances,
        naive,
    )


@overload
def find_matches(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric = 'euclidean',
    tol: float = 0.1,
    *,
    p: float | None = None,
    n_threads: int | None = 1,
    return_distances: Literal[False] = False,
) -> Matches: ...
@overload
def find_matches(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric = 'euclidean',
    tol: float = 0.1,
    *,
    p: float | None = None,
    n_threads: int | None = 1,
    return_distances: Literal[True],
) -> tuple[Matches, Distances]: ...
@overload
def find_matches(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric = 'euclidean',
    tol: float = 0.1,
    *,
    p: float | None = None,
    n_threads: int | None = 1,
    return_distances: bool,
) -> Matches | tuple[Matches, Distances]: ...
def find_matches(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric = 'euclidean',
    tol: float = 0.1,
    *,
    p: float | None = None,
    n_threads: int | None = 1,
    return_distances: bool = False,
) -> Matches | tuple[Matches, Distances]:
    """
    Finds all pairs of rows `(i, j)` where `arr0[i]` and `arr1[j]` are within `tol`
    of each other. Builds a KD-tree over the smaller array and searches it with the
    rows of the larger one.

    Parameters
    ----------
    arr0 : array_like
        First array, of shape (n, d). A 1-D array is treated as (n, 1).

    arr1 : array_like
        Second array, of shape (m, d). A 1-D array is treated as (m, 1).

    dist : str, default='euclidean'
        The distance metric: 'euclidean', 'sqeuclidean', 'manhattan', 'chebyshev',
        'minkowski' or 'cosine' (aliases: 'norm', 'l2', 'cityblock', 'l1', 'max',
        'linf', 'cos').

    tol : float, default=0.1
        Pairs with a distance of at most `tol` match. Must be >= 0.

    p : float, optional
        The order of the Minkowski distance (>= 1, default 2, `inf` allowed). Only
        allowed with `dist='minkowski'`.

    n_threads : int or None, default=1
        The number of threads to search with. `None` or 0 uses one per CPU core.
        Small inputs use fewer threads. The GIL is released either way.

    return_distances : bool, default=False
        Also return the distance of each match.

    Returns
    -------
    matches : np.ndarray
        A (k, 2) int64 array of the matching indices `[i, j]`, sorted
        lexicographically.

    distances : np.ndarray
        Only with `return_distances=True`: a (k,) array with the distance of each
        match, float32 if both inputs are float32 and float64 otherwise.

    Notes
    -----
    Rows that contain NaN or infinity never match, and nor do zero vectors with the
    cosine distance. Both inputs are computed in float32 if both are float32 (or
    float16), and in float64 otherwise.

    Examples
    --------
    >>> arr0 = np.array([[5.5, 4.5], [2.5, 1.5], [20.1, 20.1]])
    >>> arr1 = np.array([[2.4, 1.6], [5.5, 4.4], [5.5, 4.6], [4.0, 4.0]])
    >>> find_matches(arr0, arr1, tol=0.2)
    array([[0, 1], [0, 2], [1, 0]])
    >>> find_matches(arr0, arr1, tol=0.11)
    array([[0, 1], [0, 2]])
    >>> find_matches(arr0, arr1, dist='max', tol=0.11)
    array([[0, 1], [0, 2], [1, 0]])
    >>> matches, distances = find_matches(arr0, arr1, tol=0.2, return_distances=True)
    >>> distances.round(4)
    array([0.1   , 0.1   , 0.1414])
    """
    return _find(arr0, arr1, dist, tol, p, n_threads, return_distances, naive=False)


@overload
def naive_find_matches(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric = 'euclidean',
    tol: float = 0.1,
    *,
    p: float | None = None,
    n_threads: int | None = 1,
    return_distances: Literal[False] = False,
) -> Matches: ...
@overload
def naive_find_matches(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric = 'euclidean',
    tol: float = 0.1,
    *,
    p: float | None = None,
    n_threads: int | None = 1,
    return_distances: Literal[True],
) -> tuple[Matches, Distances]: ...
@overload
def naive_find_matches(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric = 'euclidean',
    tol: float = 0.1,
    *,
    p: float | None = None,
    n_threads: int | None = 1,
    return_distances: bool,
) -> Matches | tuple[Matches, Distances]: ...
def naive_find_matches(
    arr0: ArrayLike,
    arr1: ArrayLike,
    dist: Metric = 'euclidean',
    tol: float = 0.1,
    *,
    p: float | None = None,
    n_threads: int | None = 1,
    return_distances: bool = False,
) -> Matches | tuple[Matches, Distances]:
    """
    Like `find_matches`, but compares every row of `arr0` with every row of `arr1`,
    in O(n * m * d) time. Useful as a reference, and for tiny or very
    high-dimensional inputs where a tree cannot prune anything.

    >>> naive_find_matches([[0.0], [1.0]], [[0.05], [3.0]], tol=0.1)
    array([[0, 0]])
    """
    return _find(arr0, arr1, dist, tol, p, n_threads, return_distances, naive=True)


@final
class MatchIndex:
    """
    A KD-tree over the rows of `arr`, built once and searched many times with
    `query`. `MatchIndex(arr1).query(arr0, tol)` returns the same as
    `find_matches(arr0, arr1, tol=tol)` when both arrays have the same dtype. The
    index always computes in its own dtype: a float32 index converts float64 queries
    to float32, where `find_matches` would compute in float64. One-dimensional data is
    searched with the tree too, where `find_matches` sorts and sweeps instead.

    The index is immutable, so it can be shared by several threads.

    Parameters
    ----------
    arr : array_like
        The points to index, of shape (m, d). A 1-D array is treated as (m, 1). The
        index computes in float32 if `arr` is float32 (or float16), otherwise in
        float64. The data is copied, so later changes to `arr` do not affect it.

    dist : str, default='euclidean'
        The distance metric, as for `find_matches`.

    p : float, optional
        The order of the Minkowski distance, as for `find_matches`.

    n_threads : int or None, default=1
        The number of threads to build the tree with. `None` or 0 uses one per CPU
        core.

    Examples
    --------
    >>> index = MatchIndex([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    >>> index.query([[0.9, 1.0], [5.0, 5.05]], tol=0.15)
    array([[0, 1], [1, 2]])
    >>> index.query([[0.9, 1.0], [5.0, 5.05]], tol=0.01)
    array([], shape=(0, 2), dtype=int64)
    """

    __slots__ = ('_index',)

    def __init__(
        self,
        arr: ArrayLike,
        dist: Metric = 'euclidean',
        *,
        p: float | None = None,
        n_threads: int | None = 1,
    ) -> None:
        name = _check_dist(dist)
        a = _as_2d(arr, 'arr')
        values = np.ascontiguousarray(a, dtype=_float_dtype(a))
        self._index = _core.MatchIndex(values, name, p, n_threads)

    @overload
    def query(
        self,
        other: ArrayLike,
        tol: float = 0.1,
        *,
        n_threads: int | None = 1,
        return_distances: Literal[False] = False,
    ) -> Matches: ...
    @overload
    def query(
        self,
        other: ArrayLike,
        tol: float = 0.1,
        *,
        n_threads: int | None = 1,
        return_distances: Literal[True],
    ) -> tuple[Matches, Distances]: ...
    @overload
    def query(
        self,
        other: ArrayLike,
        tol: float = 0.1,
        *,
        n_threads: int | None = 1,
        return_distances: bool,
    ) -> Matches | tuple[Matches, Distances]: ...
    def query(
        self,
        other: ArrayLike,
        tol: float = 0.1,
        *,
        n_threads: int | None = 1,
        return_distances: bool = False,
    ) -> Matches | tuple[Matches, Distances]:
        """
        Finds all pairs `(i, j)` where `other[i]` and the indexed row `j` are within
        `tol` of each other, sorted lexicographically. `other` is converted to the
        dtype of the index (see `dtype`), and the distances have that dtype too. The
        other arguments and the return value are as for `find_matches`.
        """
        values = np.ascontiguousarray(_as_2d(other, 'other'), dtype=self.dtype)
        return self._index.query(values, tol, n_threads, return_distances)

    @property
    def n_points(self) -> int:
        """The number of indexed rows."""
        return self._index.n_points

    @property
    def n_dims(self) -> int:
        """The number of columns of the indexed rows."""
        return self._index.n_dims

    @property
    def dist(self) -> str:
        """The canonical name of the distance metric, e.g. 'euclidean' for 'norm'."""
        return self._index.dist

    @property
    def p(self) -> float | None:
        """The order of the Minkowski distance, or None for the other metrics."""
        return self._index.p

    @property
    def dtype(self) -> np.dtype[np.floating[Any]]:
        """The dtype the index computes in (float32 or float64)."""
        return np.dtype(self._index.dtype)

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        p = '' if self.p is None else f', p={self.p}'
        return (
            f'MatchIndex(n_points={self.n_points}, n_dims={self.n_dims}, '
            f'dist={self.dist!r}{p}, dtype={self.dtype})'
        )
