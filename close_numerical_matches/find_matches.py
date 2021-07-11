from typing import Tuple, Dict, List, Callable, Union
import itertools

import numpy as np

Key = Tuple[int, ...]
Bucket = Tuple[List[int], List[int]]
DistFunc = Callable[[np.ndarray], np.ndarray]
DistFuncArg = Union[str, DistFunc]

OVERHEAD_MULT = 5
NORM, MAX = 'norm', 'max'


def _get_dist_func(dist_metric: DistFuncArg) -> DistFunc:
    """Returns the appropriate distance function for a given dist_metric"""
    if isinstance(dist_metric, str):
        if dist_metric == NORM:
            def norm(ndarray):
                return np.linalg.norm(ndarray, axis=1)
            return norm

        if dist_metric == MAX:
            def max_abs(ndarray):
                return np.max(np.abs(ndarray), axis=1)
            return max_abs

        raise ValueError((f'If dist_metric is a string, it has to be {NORM}'
                          + f' or {MAX} but got {dist_metric}'))

    if callable(dist_metric):
        test_res = dist_metric(np.array([[1, 1]]))
        if (isinstance(test_res, np.ndarray)
                and len(test_res.shape) == 1 and len(test_res) == 1):
            return dist_metric

        raise ValueError("Supplied distance function doesn't return array of "
                         + ' expected shape')

    raise ValueError('Got distance metric of unexpected type')


def _naive_find_matches(arr0: np.ndarray, arr0_indices: np.ndarray,
                        arr1: np.ndarray, arr1_indices: np.ndarray,
                        dist_func: DistFunc, tol: float) -> List[List[int]]:
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

    matches = []
    arr1_filtered = arr1[arr1_indices]
    for arr0_idx in arr0_indices:
        row = arr0[arr0_idx]
        inner_diff = dist_func(arr1_filtered - row)
        inner_matches_idx = np.where(inner_diff <= tol)[0]
        matches.extend([arr0_idx, arr1_indices[i]] for i in inner_matches_idx)

    if matches and reverse:
        matches = list(np.array(matches)[::, ::-1])

    return matches


def _assign_to_buckets(arr0: np.ndarray, arr1: np.ndarray, bucket_tol: float)\
                       -> Dict[Key, Bucket]:
    """
    Takes two 2D arrays of numbers and a tolerance `bucket_tol` and assigns
    their indices to buckets based on these values
    """
    buckets = {}

    def assign_arr_to_buckets(arr: np.ndarray, arr_num: int):
        for i, row in enumerate(arr):
            key = tuple(int(elm // bucket_tol) for elm in row)
            if key not in buckets:
                buckets[key] = ([], [])

            buckets[key][arr_num].append(i)

    assign_arr_to_buckets(arr0, 0)
    assign_arr_to_buckets(arr1, 1)
    return buckets


def _make_deltas_iter(dim: int) -> List[Key]:
    """
    Constructs the delta values to use together with the base element for a
    given dimension to visit all neighbours and the base element itself

    Examples
    --------
    >>> _make_deltas_iter(1)
    [(-1,), (0,), (1,)]
    >>> _make_deltas_iter(2)
    [(-1, -1), (-1, 0), (-1, 1),
     (0, -1), (0, 0), (0, 1),
     (1, -1), (1, 0), (1, 1)]
    """
    DELTAS_1D = [-1, 0, 1]
    return list(itertools.product(*tuple([DELTAS_1D] * dim)))


def _bucket_keys_to_check(key: Key, deltas_iter: List[Key]) -> List[Key]:
    """
    Returns a list of the keys to iterate for a given key and the deltas

    Examples
    --------
    >>> _bucket_keys_to_check((5,), _make_deltas_iter(1))
    [(4,), (5,), (6,)]
    >>> _bucket_keys_to_check((5, 7), _make_deltas_iter(2))
    [(4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (5, 8), (6, 6), (6, 7), (6, 8)]
    """
    return [tuple(key_elm + delta for key_elm, delta in zip(key, deltas))
            for deltas in deltas_iter]


def naive_find_matches(arr0: np.ndarray, arr1: np.ndarray,
                       dist: DistFuncArg = NORM, tol: float = 0.1) \
                       -> np.ndarray:
    """
    Finds matching indices within tolerance using nested for-loop approach.
    parameters. Runtime is O(nmd) for two arrays of shape (n, d) and (m, d)

    Parameters
    ----------
    arr0 : np.ndarray
        First array to find matches against. Should be of size (n, d).

    arr1 : np.ndarray
        Second array to find matches against. Should be of size (m, d).

    dist : {'norm', 'max'} or Callable[[np.ndarray], np.ndarray]
        Distance metric to calculate distance. `'norm'` and `'max'` are
        currently supported. If you want some other distance function, you can
        supply your own function. It should take an (n, d) array as argument
        and return an (n,) array.

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
    arr0, arr1 = np.array(arr0, copy=False), np.array(arr1, copy=False)

    assert len(arr0.shape) == len(arr1.shape) == 2, \
        f'Arrays should be 2D but got {len(arr0.shape)} and {len(arr1.shape)}'

    assert arr0.shape[1] == arr1.shape[1], \
        ('Arrays should be of equivalent size in the second axis, but got'
         + f' {arr0.shape[1]} and {arr1.shape[1]}')

    assert tol > 0, f'Tolerance has to be strictly positive but got {tol}'

    dist_func = _get_dist_func(dist)
    arr0_indices, arr1_indices = np.arange(len(arr0)), np.arange(len(arr1))
    matches = _naive_find_matches(arr0, arr0_indices, arr1, arr1_indices,
                                  dist_func, tol)
    return np.array(matches, dtype=np.int64)


def find_matches(arr0: np.ndarray, arr1: np.ndarray, dist: DistFuncArg = NORM,
                 tol: float = 0.1, bucket_tol_mult: int = 2) -> np.ndarray:
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

    dist : {'norm', 'max'} or Callable[[np.ndarray], np.ndarray]
        Distance metric to calculate distance. `'norm'` and `'max'` are
        currently supported. If you want some other distance function, you can
        supply your own function. It should take an (n, d) array as argument
        and return an (n,) array.

    tol : float, default=0.1
        The tolerance where values are considered the similar enough to count
        as a match. Should be > 0.

    bucket_tol_mult : int, default=2
        The tolerance multiplier to use for assigning buckets. Can in some
        instances make algorithm faster to tweak this. Should never be less
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
    arr0, arr1 = np.array(arr0, copy=False), np.array(arr1, copy=False)

    assert len(arr0.shape) == len(arr1.shape) == 2, \
        f'Arrays should be 2D but got {len(arr0.shape)} and {len(arr1.shape)}'

    assert arr0.shape[1] == arr1.shape[1], \
        ('Arrays should be of equivalent size in the second axis, but got'
         + f' {arr0.shape[1]} and {arr1.shape[1]}')

    assert tol > 0, f'Tolerance has to be strictly positive but got {tol}'
    assert bucket_tol_mult >= 1.0, \
        f'bucket_tol_mult should be >= 1 but got {bucket_tol_mult}'

    buckets = _assign_to_buckets(arr0, arr1, bucket_tol_mult * tol)

    # Check if O(nmd) < O(bd^3) * const in which case it makes better sense
    # to run naive algorithm
    n, m, b, d = len(arr0), len(arr1), len(buckets), arr0.shape[1]
    if n * m * d < b * d**3 * OVERHEAD_MULT:
        return naive_find_matches(arr0, arr1, dist, tol)

    dist_func = _get_dist_func(dist)
    deltas_iter = _make_deltas_iter(d)
    matches = []
    for key, bucket in buckets.items():
        arr0_indices = bucket[0]
        if not arr0_indices:
            continue

        # Collect all indices from arr1 from this + neighbouring buckets
        arr1_indices = []
        for bucket_key in _bucket_keys_to_check(key, deltas_iter):
            if bucket_key in buckets:
                arr1_indices.extend(buckets[bucket_key][1])

        if not arr1_indices:
            continue

        matches_inner = _naive_find_matches(arr0, arr0_indices,
                                            arr1, arr1_indices,
                                            dist_func, tol)
        matches.extend(matches_inner)

    return np.array(matches, dtype=np.int64)
