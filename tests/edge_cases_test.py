from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from close_numerical_matches import METRICS
from close_numerical_matches import find_matches
from close_numerical_matches import naive_find_matches

nan, inf = float('nan'), float('inf')


@pytest.mark.parametrize('func', [find_matches, naive_find_matches])
def test_empty_inputs(func: Callable[..., Any]) -> None:
    assert func(np.empty((0, 3)), np.ones((5, 3))).shape == (0, 2)
    assert func(np.ones((5, 3)), np.empty((0, 3))).shape == (0, 2)
    matches, dists = func(np.empty((0, 3)), np.empty((0, 3)), return_distances=True)
    assert matches.shape == (0, 2)
    assert dists.shape == (0,)


def test_zero_dimensions_match_everything() -> None:
    matches = find_matches(np.empty((2, 0)), np.empty((3, 0)), tol=0)
    assert matches.tolist() == [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]


def test_one_dimensional_arrays_are_columns() -> None:
    matches = find_matches([1.0, 5.0, 9.0], [5.05, 1.0, 20.0], tol=0.1)
    assert matches.tolist() == [[0, 1], [1, 0]]


@pytest.mark.parametrize('dist', METRICS)
def test_non_finite_rows_never_match(dist: str) -> None:
    arr0 = [[nan, 0.0], [1.0, 1.0], [inf, 1.0], [1.0, -inf]]
    arr1 = [[1.0, 1.0], [nan, nan], [inf, 1.0]]
    for tol in (0.0, 1.0, inf):
        assert find_matches(arr0, arr1, dist, tol).tolist() == [[1, 0]]  # type: ignore[arg-type]


def test_zero_vectors_never_match_by_cosine() -> None:
    arr0 = [[0.0, 0.0], [1.0, 0.0]]
    arr1 = [[0.0, 0.0], [2.0, 0.0]]
    assert find_matches(arr0, arr1, 'cosine', 2.0).tolist() == [[1, 1]]


def test_cosine_handles_extreme_magnitudes() -> None:
    arr0 = [[1e-300, 1e-300], [1e300, 1e300]]
    arr1 = [[3.0, 3.0]]
    matches, dists = find_matches(arr0, arr1, 'cos', 1e-9, return_distances=True)
    assert matches.tolist() == [[0, 0], [1, 0]]
    assert (dists < 1e-12).all()


def test_zero_tolerance_finds_exact_duplicates() -> None:
    arr0 = [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0 + 1e-15]]
    arr1 = [[3.0, 4.0], [1.0, 2.0]]
    assert find_matches(arr0, arr1, tol=0).tolist() == [[0, 1], [1, 0]]


def test_tolerance_is_inclusive() -> None:
    assert find_matches([[0, 0]], [[3, 4]], tol=5).tolist() == [[0, 0]]
    assert find_matches([[0, 0]], [[3, 4]], 'manhattan', 7).tolist() == [[0, 0]]
    assert find_matches([[0, 0]], [[3, 4]], 'chebyshev', 4).tolist() == [[0, 0]]
    assert find_matches([[0, 0]], [[3, 4]], 'sqeuclidean', 25).tolist() == [[0, 0]]
    assert find_matches([[0, 0]], [[3, 4]], 'chebyshev', 3.999).shape == (0, 2)


def test_infinite_tolerance_matches_all_finite_pairs() -> None:
    rng = np.random.default_rng(0)
    arr0, arr1 = rng.random((60, 2)) * 1e6, rng.random((40, 2))
    assert len(find_matches(arr0, arr1, tol=inf)) == 60 * 40


@pytest.mark.parametrize(
    ('dist', 'p', 'expected'),
    [
        ('euclidean', None, 5.0),
        ('sqeuclidean', None, 25.0),
        ('manhattan', None, 7.0),
        ('chebyshev', None, 4.0),
        ('minkowski', 1.0, 7.0),
        ('minkowski', 2.0, 5.0),
        ('minkowski', 3.0, 91 ** (1 / 3)),
        ('minkowski', inf, 4.0),
    ],
)
def test_distances(dist: str, p: float | None, expected: float) -> None:
    matches, dists = find_matches(
        [[1.0, 1.0]],
        [[4.0, 5.0]],
        dist,  # type: ignore[arg-type]
        100.0,
        p=p,
        return_distances=True,
    )
    assert matches.tolist() == [[0, 0]]
    assert dists.dtype == np.float64
    assert dists[0] == pytest.approx(expected, rel=1e-12)


def test_cosine_distance_values() -> None:
    _, dists = find_matches(
        [[1.0, 0.0]],
        [[2.0, 0.0], [0.0, 3.0], [-1.0, 0.0]],
        'cos',
        2.0,
        return_distances=True,
    )
    np.testing.assert_allclose(dists, [0.0, 1.0, 2.0], atol=1e-15)


def test_mixed_dtypes_use_float64() -> None:
    a32 = np.array([[0.0, 0.0]], dtype=np.float32)
    _, d = find_matches(a32, np.array([[0.1, 0.0]]), tol=1, return_distances=True)
    assert d.dtype == np.float64
    _, d = find_matches(a32, a32, tol=1, return_distances=True)
    assert d.dtype == np.float32
    ints = np.array([[1, 2]], dtype=np.int8)
    assert find_matches(ints, ints.astype(bool), tol=1.5).tolist() == [[0, 0]]


def test_non_contiguous_and_big_endian_input() -> None:
    rng = np.random.default_rng(2)
    base = rng.random((400, 6))
    arr0 = base[::2, ::3]  # strided view
    arr1 = np.asfortranarray(base[1::2, ::3])
    expected = find_matches(
        np.ascontiguousarray(arr0), np.ascontiguousarray(arr1), tol=0.05
    )
    assert len(expected) > 0
    np.testing.assert_array_equal(find_matches(arr0, arr1, tol=0.05), expected)
    np.testing.assert_array_equal(
        find_matches(arr0.astype('>f8'), arr1, tol=0.05), expected
    )


def test_input_is_not_modified() -> None:
    arr0 = np.array([[3.0, 4.0], [0.0, 1.0]])
    arr1 = np.array([[6.0, 8.0]])
    before0, before1 = arr0.copy(), arr1.copy()
    find_matches(arr0, arr1, 'cosine', 0.1)
    np.testing.assert_array_equal(arr0, before0)
    np.testing.assert_array_equal(arr1, before1)


def test_all_threads() -> None:
    rng = np.random.default_rng(3)
    arr0, arr1 = rng.random((20_000, 2)), rng.random((10_000, 2))
    expected = find_matches(arr0, arr1, tol=0.005)
    for n_threads in (0, None, 2, 64):
        np.testing.assert_array_equal(
            find_matches(arr0, arr1, tol=0.005, n_threads=n_threads), expected
        )


@pytest.mark.parametrize('func', [find_matches, naive_find_matches])
def test_float32_overflow_of_squared_tolerance(func: Callable[..., Any]) -> None:
    # tol**2 and the squared distances overflow float32.
    a0 = np.array([[0.0]], dtype=np.float32)
    far = np.array([[3e19]], dtype=np.float32)
    near = np.array([[1.5e19]], dtype=np.float32)
    assert func(a0, far, tol=2e19).shape == (0, 2)
    matches, dists = func(a0, near, tol=2e19, return_distances=True)
    assert matches.tolist() == [[0, 0]]
    assert dists[0] == pytest.approx(1.5e19, rel=1e-6)


@pytest.mark.parametrize('func', [find_matches, naive_find_matches])
@pytest.mark.parametrize(
    ('dtype', 'p', 'tol', 'outside', 'inside'),
    [
        (np.float64, 2000.0, 0.5, 0.6, 0.4),  # |d|**p underflows
        (np.float32, 10.0, 1e-5, 1.2e-5, 0.8e-5),
        (np.float64, 2.0, 1e-165, 2e-165, 5e-166),  # tol**2 underflows
        (np.float64, 3.0, 1e120, 2e120, 5e119),  # tol**p overflows
    ],
)
def test_minkowski_underflow_and_overflow(
    func: Callable[..., Any],
    dtype: type[np.floating[Any]],
    p: float,
    tol: float,
    outside: float,
    inside: float,
) -> None:
    a0 = np.array([[0.0], [0.0]], dtype=dtype)
    a1 = np.array([[outside], [inside]], dtype=dtype)
    matches, dists = func(a0[:1], a1, 'minkowski', tol, p=p, return_distances=True)
    assert matches.tolist() == [[0, 1]]
    assert dists[0] == pytest.approx(inside, rel=1e-5)


@pytest.mark.parametrize('func', [find_matches, naive_find_matches])
def test_euclidean_with_extreme_magnitudes(func: Callable[..., Any]) -> None:
    tiny = np.array([[0.0, 0.0], [2e-165, 0.0], [3e-166, 4e-166]])
    matches, dists = func(tiny[:1], tiny[1:], tol=1e-165, return_distances=True)
    assert matches.tolist() == [[0, 1]]
    assert dists[0] == pytest.approx(5e-166, rel=1e-12)
    huge = np.array([[0.0, 0.0], [3e200, 4e200], [3e200, 5e200]])
    matches, dists = func(huge[:1], huge[1:], tol=5e200, return_distances=True)
    assert matches.tolist() == [[0, 0]]
    assert dists[0] == pytest.approx(5e200, rel=1e-12)


def test_tree_and_naive_agree_on_extreme_values() -> None:
    rng = np.random.default_rng(4)
    for scale in (1e-170, 1e-30, 1.0, 1e30, 1e150):
        data = rng.random((3000, 2)) * scale
        a0, a1 = data[:1500], data[1500:]
        for dist, p in [('euclidean', None), ('minkowski', 3.0), ('minkowski', 500.0)]:
            tol = 0.01 * scale
            fast = find_matches(a0, a1, dist, tol, p=p, return_distances=True)
            slow = naive_find_matches(a0, a1, dist, tol, p=p, return_distances=True)
            assert len(slow[0]) > 0, (scale, dist)
            np.testing.assert_array_equal(fast[0], slow[0])
            np.testing.assert_array_equal(fast[1], slow[1])
