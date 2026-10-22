from __future__ import annotations

import numpy as np
import pytest

from close_numerical_matches import MatchIndex
from close_numerical_matches import find_matches


def test_query_equals_find_matches() -> None:
    rng = np.random.default_rng(0)
    arr0, arr1 = rng.random((3000, 3)), rng.random((2000, 3))
    index = MatchIndex(arr1)
    for tol in (0.0, 0.01, 0.05, 0.2):
        expected = find_matches(arr0, arr1, tol=tol)
        np.testing.assert_array_equal(index.query(arr0, tol), expected)
        np.testing.assert_array_equal(index.query(arr0, tol, n_threads=None), expected)


@pytest.mark.parametrize(
    ('dist', 'p', 'canonical'),
    [
        ('norm', None, 'euclidean'),
        ('l1', None, 'manhattan'),
        ('max', None, 'chebyshev'),
        ('cos', None, 'cosine'),
        ('minkowski', 3.0, 'minkowski'),
        ('minkowski', None, 'minkowski'),
    ],
)
def test_properties(dist: str, p: float | None, canonical: str) -> None:
    index = MatchIndex(np.ones((7, 3)), dist, p=p)  # type: ignore[arg-type]
    assert index.n_points == len(index) == 7
    assert index.n_dims == 3
    assert index.dist == canonical
    assert index.p == (None if canonical != 'minkowski' else (p or 2.0))
    assert index.dtype == np.float64
    assert repr(index).startswith('MatchIndex(n_points=7, n_dims=3')


def test_dtype_follows_input() -> None:
    arr = np.random.default_rng(1).random((100, 2))
    assert MatchIndex(arr.astype(np.float32)).dtype == np.float32
    assert MatchIndex(arr.astype(np.float16)).dtype == np.float32
    assert MatchIndex(arr).dtype == np.float64
    assert MatchIndex((arr * 10).astype(int)).dtype == np.float64
    index = MatchIndex(arr.astype(np.float32))
    matches, dists = index.query(arr, 0.05, return_distances=True)
    assert dists.dtype == np.float32
    assert len(matches) >= 100  # every point matches itself


def test_index_copies_its_input() -> None:
    arr = np.array([[0.0, 0.0], [1.0, 1.0]])
    index = MatchIndex(arr)
    arr[:] = 100.0
    assert index.query([[0.0, 0.0]], 0.1).tolist() == [[0, 0]]


def test_one_dimensional_input() -> None:
    index = MatchIndex([1.0, 2.0, 3.0])
    assert index.n_dims == 1
    assert index.query([2.05, 9.0], 0.1).tolist() == [[0, 1]]


def test_query_dimension_mismatch() -> None:
    index = MatchIndex(np.ones((4, 3)))
    with pytest.raises(ValueError, match='index has 3'):
        index.query(np.ones((2, 2)))


def test_query_errors() -> None:
    index = MatchIndex(np.ones((4, 3)))
    with pytest.raises(ValueError, match='tol must be'):
        index.query(np.ones((2, 3)), -1.0)
    with pytest.raises(ValueError, match='n_threads'):
        index.query(np.ones((2, 3)), n_threads=-2)
    with pytest.raises(TypeError, match='Custom distance'):
        MatchIndex(np.ones((4, 3)), lambda a, b: a)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='unknown distance metric'):
        MatchIndex(np.ones((4, 3)), 'hamming')  # type: ignore[arg-type]


def test_empty_index() -> None:
    index = MatchIndex(np.empty((0, 2)))
    assert index.n_points == 0
    assert index.query(np.ones((3, 2)), 10.0).shape == (0, 2)
