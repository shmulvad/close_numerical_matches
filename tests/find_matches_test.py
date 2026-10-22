from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from close_numerical_matches import find_matches
from close_numerical_matches import naive_find_matches


def as_set(matches: Any) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in np.asarray(matches).reshape(-1, 2)}


def is_same(matches0: Any, matches1: Any) -> bool:
    """Whether two collections of matches are the same, ignoring order."""
    matches0 = np.asarray(matches0).reshape(-1, 2)
    matches1 = np.asarray(matches1).reshape(-1, 2)
    set0, set1 = as_set(matches0), as_set(matches1)
    return set0 == set1 and len(set0) == len(matches0) == len(matches1)


class TestFindMatches:
    def test_small(self) -> None:
        arr0 = np.array(
            [
                [5.50000001, 4.50000001],
                [5.73212317, 8.73212317],
                [8.23410901, 9.26367166],
            ]
        )
        arr1 = np.array(
            [
                [5.50000002, 4.50000002],  # norm is too big for arr0[0]
                [5.73212317, 8.73212317],  # norm is within for arr[1]
                [5.500000015, 4.500000005],  # norm is within arr[0]
                [5.73212317, 8.732123182],  # norm is too big for arr[1]
            ]
        )
        matches = find_matches(arr0, arr1, tol=0.00000001)
        assert is_same(matches, [[0, 2], [1, 1]])

    def test_big(self) -> None:
        arr0 = np.array(
            [
                [550000001, 450000001],
                [573212317, 873212317],
                [823410901, 926367166],
            ]
        )
        arr1 = np.array(
            [
                [550000002, 450000002],  # norm is too big for arr0[0]
                [573212317, 873212317],  # norm is within for arr[1]
                [550000001, 450000000],  # norm is within arr[0]
                [573212317, 873212318],  # norm is too big for arr[1]
            ]
        )
        matches = find_matches(arr0, arr1, tol=1)
        assert is_same(matches, [[0, 2], [1, 1], [1, 3]])

    def test_all_matches(self) -> None:
        arr0 = np.array([[5, 5], [6, 6]])
        arr1 = np.array([[7, 7], [8, 8], [9, 9]])
        matches = find_matches(arr0, arr1, tol=999)
        expected = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]
        assert matches.tolist() == expected

    def test_max(self) -> None:
        arr0 = np.array([[5, 5]])
        arr1 = np.array([[4, 4], [6, 6], [5, 6.1]])
        matches = find_matches(arr0, arr1, tol=1.000001, dist='max')
        assert is_same(matches, [[0, 0], [0, 1]])

    def test_cosine(self) -> None:
        arr0 = np.array([[0, 1]])
        arr1 = np.array([[0, 5], [5, 0], [0, -5]])
        matches = find_matches(arr0, arr1, tol=0.1, dist='cos')
        assert is_same(matches, [[0, 0]])
        matches = find_matches(arr0, arr1, tol=1.01, dist='cos')
        assert is_same(matches, [[0, 0], [0, 1]])

    def test_no_matches(self) -> None:
        matches = find_matches(np.zeros((9, 3)), np.ones((5, 3)), tol=0.9)
        assert matches.shape == (0, 2)
        assert matches.dtype == np.int64

    def test_manhattan(self) -> None:
        arr0 = np.array([[3, 3, 3]])
        arr1 = np.array([[3, 4, 4], [4, 4, 4]])
        matches = find_matches(arr0, arr1, dist='manhattan', tol=2.001)
        assert is_same(matches, [[0, 0]])

    def test_result_is_sorted(self) -> None:
        rng = np.random.default_rng(0)
        arr0 = rng.random((2000, 2))
        arr1 = rng.random((300, 2))
        matches = find_matches(arr0, arr1, tol=0.05)
        assert len(matches) > 100
        order = np.lexsort((matches[:, 1], matches[:, 0]))
        assert (order == np.arange(len(matches))).all()

    def test_lists_as_input(self) -> None:
        matches = find_matches([[0, 0], [10, 10]], [[0.05, 0], [10, 10.01]])
        assert matches.tolist() == [[0, 0], [1, 1]]

    def test_naive_gives_same_result(self) -> None:
        rng = np.random.default_rng(1)
        arr0, arr1 = rng.random((500, 3)), rng.random((400, 3))
        expected = naive_find_matches(arr0, arr1, tol=0.08)
        assert len(expected) > 0
        assert (find_matches(arr0, arr1, tol=0.08) == expected).all()


class TestErrors:
    def test_custom_dist_func_is_rejected(self) -> None:
        def manhattan(arr: np.ndarray, row: np.ndarray) -> np.ndarray:
            return np.sum(np.abs(arr - row), axis=1)

        with pytest.raises(TypeError, match='Custom distance functions'):
            find_matches(np.ones((5, 4)), np.ones((5, 4)), dist=manhattan)  # type: ignore[arg-type]

    def test_non_str_dist(self) -> None:
        with pytest.raises(TypeError, match='dist must be a string'):
            find_matches(np.ones((5, 4)), np.ones((5, 4)), dist=1)  # type: ignore[arg-type]

    def test_unknown_dist(self) -> None:
        with pytest.raises(ValueError, match='unknown distance metric'):
            find_matches(np.ones((5, 4)), np.ones((5, 4)), dist='non-existing')  # type: ignore[arg-type]

    @pytest.mark.parametrize('tol', [-1.0, -1e-300, float('nan')])
    def test_invalid_tol(self, tol: float) -> None:
        with pytest.raises(ValueError, match='tol must be'):
            find_matches(np.ones((5, 4)), np.ones((5, 4)), tol=tol)

    def test_non_equivalent_shapes(self) -> None:
        with pytest.raises(ValueError, match='same number of columns'):
            find_matches(np.ones((5, 3)), np.ones((5, 4)))

    def test_incorrect_shape(self) -> None:
        with pytest.raises(ValueError, match='1-D or 2-D'):
            find_matches(np.ones((4, 5, 2)), np.ones((4, 5, 2)))
        with pytest.raises(ValueError, match='1-D or 2-D'):
            find_matches(np.float64(1.0), np.ones((4, 1)))

    def test_non_numeric_input(self) -> None:
        with pytest.raises(TypeError, match='real numbers'):
            find_matches(np.ones((2, 2), dtype=complex), np.ones((2, 2)))
        with pytest.raises(TypeError, match='real numbers'):
            find_matches([['a', 'b']], np.ones((2, 2)))

    def test_p_only_for_minkowski(self) -> None:
        with pytest.raises(ValueError, match='only used by the minkowski'):
            find_matches(np.ones((2, 2)), np.ones((2, 2)), p=3)
        with pytest.raises(ValueError, match='p must be >= 1'):
            find_matches(np.ones((2, 2)), np.ones((2, 2)), dist='minkowski', p=0.5)

    def test_negative_threads(self) -> None:
        with pytest.raises(ValueError, match='n_threads'):
            find_matches(np.ones((2, 2)), np.ones((2, 2)), n_threads=-1)

    def test_removed_bucket_tol_mult(self) -> None:
        with pytest.raises(TypeError):
            find_matches(np.ones((2, 2)), np.ones((2, 2)), bucket_tol_mult=2.0)  # type: ignore[call-overload]
