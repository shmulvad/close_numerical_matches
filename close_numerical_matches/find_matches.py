from __future__ import annotations

import itertools
from typing import Callable
from typing import Generator
from typing import Union

import numpy as np

DistFunc = Callable[[np.ndarray, np.ndarray], np.ndarray]
DistFuncArg = Union[str, DistFunc]
NORM, MAX, COS = 'norm', 'max', 'cos'

OVERHEAD_MULT = 5
DELTAS_1D = [-1, 0, 1]


def _normalize_vectors(arr: np.ndarray) -> np.ndarray:
    """
    Takes an array and returns a copy of it where each row has been
    normalized to unit norm
    """
    return arr / np.linalg.norm(arr, axis=1).reshape(-1, 1)


def _norm(arr: np.ndarray, row: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr - row, axis=1)


def _max_abs(arr: np.ndarray, row: np.ndarray) -> np.ndarray:
    return np.max(np.abs(arr - row), axis=1)


def _cosine_dist(arr: np.ndarray, row: np.ndarray) -> np.ndarray:
    num = arr @ row
    denom = np.linalg.norm(arr, axis=1) * np.linalg.norm(row)
    cosine_sim = num / denom
    # Value is now similarity in [-1, 1] so we want to
    # convert to distance in [0, 2]
    return 1.0 - cosine_sim


DIST_FUNCS: dict[str, DistFunc] = {
    NORM: _norm,
    MAX: _max_abs,
    COS: _cosine_dist,
}


def _get_dist_func(dist_metric: DistFuncArg) -> DistFunc:
    """Returns the appropriate distance function for a given dist_metric"""
    if isinstance(dist_metric, str):
        if dist_metric in DIST_FUNCS:
            return DIST_FUNCS[dist_metric]

        raise ValueError(
            'If dist_metric is a string, it has to be one of '
            + f'{list(DIST_FUNCS.keys())} but got "{dist_metric}"',
        )

    if callable(dist_metric):
        test_res = dist_metric(np.array([[1]]), np.array([1]))
        is_valid = (
            isinstance(test_res, np.ndarray)
            and len(test_res.shape) == 1
            and len(test_res) == 1
        )
        if is_valid:
            return dist_metric

        raise ValueError(
            "Supplied distance function doesn't return array of expected shape",
        )

    raise ValueError('Got distance metric of unexpected type')


def _naive_find_matches(
    arr0: np.ndarray,
    arr0_indices: Union[list[int], np.ndarray],  # noqa: UP007
    arr1: np.ndarray,
    arr1_indices: Union[list[int], np.ndarray],  # noqa: UP007
    dist_func: DistFunc,
    tol: float,
) -> list[list[int]]:
    """
    Finds matching indices within tolerance using nested for-loop approach.
    Only considers the indices given by the `arr0_indices` and `arr1_indices`
    parameters. Runtime is O(nmk) for two arrays of shape (n, k) and (m, k)

    Returns
    -------
    `matches` : `List[List[int]]`
        List of 2-length lists of integers corresponding to matching indices

    Examples
    --------
    >>> arr0 = np.array([[5.5, 4.5], [2.5, 1.5], [20.1, 20.1]])
    >>> arr1 = np.array([[2.4, 1.6], [5.5, 4.4], [5.5, 4.6], [4.0, 4.0]])
    >>> _naive_find_matches(arr0, [0, 1, 2], arr1, [0, 1, 2, 3], tol=0.2)
    [[0, 1], [0, 2], [1, 0]]
    """
    reverse = False
    if len(arr0_indices) > len(arr1_indices):
        reverse = True
        arr0, arr0_indices, arr1, arr1_indices \
            = arr1, arr1_indices, arr0, arr0_indices

    matches: list[list[int]] = []
    arr1_filtered = arr1[arr1_indices]
    for arr0_idx in arr0_indices:
        arr0_idx = int(arr0_idx)
        row = arr0[arr0_idx]
        inner_diff = dist_func(arr1_filtered, row)
        inner_matches_idx = np.where(inner_diff <= tol)[0]
        new_matches: list[list[int]] = [[arr0_idx, int(arr1_indices[i])] for i in inner_matches_idx]
        matches.extend(new_matches)

    if matches and reverse:
        matches = list(np.array(matches)[::, ::-1])

    return matches


def _assign_to_buckets(
    arr0: np.ndarray,
    arr1: np.ndarray,
    bucket_tol: float,
) -> dict[tuple[int, ...], tuple[list[int], list[int]]]:
    """
    Takes two 2D arrays of numbers and a tolerance `bucket_tol` and assigns
    their indices to buckets based on these values
    """
    buckets: dict[tuple[int, ...], tuple[list[int], list[int]]] = {}

    def assign_arr_to_buckets(arr: np.ndarray, arr_num: int) -> None:
        for i, row in enumerate(arr):
            key = tuple(int(elm // bucket_tol) for elm in row)
            if key not in buckets:
                buckets[key] = ([], [])

            buckets[key][arr_num].append(i)

    assign_arr_to_buckets(arr0, 0)
    assign_arr_to_buckets(arr1, 1)
    return buckets


def _make_deltas_iter(dim: int) -> Generator[tuple[int, ...], None, None]:
    """
    Constructs the delta values to use together with the base element for a
    given dimension to visit all neighbors and the base element itself

    Examples
    --------
    >>> list(_make_deltas_iter(1))
    [(-1,), (0,), (1,)]
    >>> list(_make_deltas_iter(2))
    [(-1, -1), (-1, 0), (-1, 1),
     (0, -1), (0, 0), (0, 1),
     (1, -1), (1, 0), (1, 1)]
    """
    yield from itertools.product(*tuple([DELTAS_1D] * dim))


def _bucket_keys_to_check(
    key: tuple[int, ...],
    deltas_iter: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """
    Returns a list of the keys to iterate for a given key and the deltas

    Examples
    --------
    >>> _bucket_keys_to_check((5,), _make_deltas_iter(1))
    [(4,), (5,), (6,)]
    >>> _bucket_keys_to_check((5, 7), _make_deltas_iter(2))
    [(4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 6), (6, 7), (6, 8)]
    """
    return [
        tuple(key_elm + delta for key_elm, delta in zip(key, deltas))
        for deltas in deltas_iter
    ]


def naive_find_matches(
    arr0: Union[np.ndarray, list[list[Union[int, float]]]],  # noqa: UP007
    arr1: Union[np.ndarray, list[list[Union[int, float]]]],  # noqa: UP007
    dist: DistFuncArg = NORM,
    tol: float = 0.1,
) -> np.ndarray:
    """
    Finds matching indices within tolerance using nested for-loop approach.
    parameters. Runtime is O(nmd) for two arrays of shape (n, d) and (m, d)

    Parameters
    ----------
    arr0 : np.ndarray
        First array to find matches against. Should be of size (n, d).

    arr1 : np.ndarray
        Second array to find matches against. Should be of size (m, d).

    dist : {'norm', 'max', 'cos'} or Callable[[np.ndarray, np.ndarray], np.ndarray]
        Distance metric to calculate distance. `'norm'`, `'max'` and `'cos'` are
        currently supported. If you want some other distance function, you can
        supply your own function. It should take an (n, d) array and (d,) array
        as argument and return an (n,) array.

    tol : float, default=0.1
        The tolerance where values are considered the similar enough to count
        as a match. Should be > 0.

    Returns
    -------
    matches : np.ndarray
        Array of corresponding indices of the numerical matches in the arrays.

    Examples
    --------
    >>> arr0 = np.array([[5.5, 4.5], [2.5, 1.5], [20.1, 20.1]])
    >>> arr1 = np.array([[2.4, 1.6], [5.5, 4.4], [5.5, 4.6], [4.0, 4.0]])
    >>> naive_find_matches(arr0, arr1, tol=0.2)
    array([[0, 1], [0, 2], [1, 0]])
    >>> naive_find_matches(arr0, arr1, tol=0.11)
    array([[0, 1], [0, 2]])
    >>> naive_find_matches(arr0, arr1, dist='max', tol=0.11)
    array([[0, 1], [0, 2], [1, 0]])
    """
    # Construct ndarrays if pure Python lists are passed in
    arr0 = np.array(arr0, copy=False)
    arr1 = np.array(arr1, copy=False)

    assert len(arr0.shape) == len(arr1.shape) == 2, \
        f'Arrays should be 2D but got {len(arr0.shape)} and {len(arr1.shape)}'

    assert arr0.shape[1] == arr1.shape[1], \
        ('Arrays should be of equivalent size in the second axis, but got'
         + f' {arr0.shape[1]} and {arr1.shape[1]}')

    assert tol > 0, f'Tolerance has to be strictly positive but got {tol}'

    if dist == COS:
        arr0, arr1 = _normalize_vectors(arr0), _normalize_vectors(arr1)

    matches = _naive_find_matches(
        arr0=arr0,
        arr0_indices=np.arange(len(arr0)),
        arr1=arr1,
        arr1_indices=np.arange(len(arr1)),
        dist_func=_get_dist_func(dist),
        tol=tol,
    )
    return np.array(matches, dtype=np.int64)


def find_matches(
    arr0: np.ndarray,
    arr1: np.ndarray,
    dist: DistFuncArg = NORM,
    tol: float = 0.1,
    bucket_tol_mult: float = 2.0,
) -> np.ndarray:
    """
    Finds all numerical matches in two 2D ndarrays of shape (n, d) and (m, d)
    that are within tolerance level and returns the indices. Works best for
    small d and large n and m where there are relatively few matches.

    Parameters
    ----------
    arr0 : np.ndarray
        First array to find matches against. Should be of size (n, d).

    arr1 : np.ndarray
        Second array to find matches against. Should be of size (m, d).

    dist : {'norm', 'max', 'cos'} or Callable[[np.ndarray, np.ndarray], np.ndarray]
        Distance metric to calculate distance. `'norm'`, `'max'` and `'cos'` are
        currently supported. If you want some other distance function, you can
        supply your own function. It should take an (n, d) array and (d,) array
        as argument and return an (n,) array.

    tol : float, default=0.1
        The tolerance where values are considered the similar enough to count
        as a match. Should be > 0.

    bucket_tol_mult : float, default=2.0
        The tolerance multiplier to use for assigning buckets. Can in some
        instances make algorithm faster to tweak this. For cosine distance,
        you likely want to set this much higher. Should never be less
        than 1.

    Returns
    -------
    matches : np.ndarray
        Array of corresponding indices of the numerical matches in the arrays.

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
    """
    # Construct ndarrays if pure Python lists are passed in
    arr0 = np.array(arr0, copy=False)
    arr1 = np.array(arr1, copy=False)

    assert len(arr0.shape) == len(arr1.shape) == 2, \
        f'Arrays should be 2D but got {len(arr0.shape)} and {len(arr1.shape)}'

    assert arr0.shape[1] == arr1.shape[1], \
        ('Arrays should be of equivalent size in the second axis, but got'
         + f' {arr0.shape[1]} and {arr1.shape[1]}')

    assert tol > 0, f'Tolerance has to be strictly positive but got {tol}'
    assert bucket_tol_mult >= 1.0, \
        f'bucket_tol_mult should be >= 1 but got {bucket_tol_mult}'

    if dist == COS:
        arr0, arr1 = _normalize_vectors(arr0), _normalize_vectors(arr1)

    buckets = _assign_to_buckets(arr0, arr1, bucket_tol_mult * tol)

    # Check if O(nmd) < O(bd^3) * const in which case it makes better sense
    # to run naive algorithm
    n, m, b, d = len(arr0), len(arr1), len(buckets), arr0.shape[1]
    if n * m * d < b * d**3 * OVERHEAD_MULT:
        return naive_find_matches(arr0, arr1, dist, tol)

    dist_func = _get_dist_func(dist)
    deltas_iter = list(_make_deltas_iter(d))
    matches = []
    for key, bucket in buckets.items():
        arr0_indices = bucket[0]
        if not arr0_indices:
            continue

        # Collect all indices from arr1 from this + neighboring buckets
        arr1_indices = []
        for bucket_key in _bucket_keys_to_check(key, deltas_iter):
            if bucket_key in buckets:
                arr1_indices.extend(buckets[bucket_key][1])

        if not arr1_indices:
            continue

        matches_inner = _naive_find_matches(
            arr0=arr0,
            arr0_indices=arr0_indices,
            arr1=arr1,
            arr1_indices=arr1_indices,
            dist_func=dist_func,
            tol=tol,
        )
        matches.extend(matches_inner)

    return np.array(matches, dtype=np.int64)
