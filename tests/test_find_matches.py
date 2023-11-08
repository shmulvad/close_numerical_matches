from __future__ import annotations

import numpy as np
import pytest

from close_numerical_matches import find_matches


def is_same(matches0, matches1) -> bool:
    """
    Checks if two arrays of matches are the same, ignoring order
    """
    matches0_set = set(map(tuple, matches0))
    matches1_set = set(map(tuple, matches1))
    return matches0_set == matches1_set and len(matches0_set) == len(matches0) == len(matches1)


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
        expected = [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
        ]
        assert is_same(matches, expected)

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
        assert matches.shape == (0,)

    def test_custom_dist(self) -> None:
        def manhatten_dist(arr: np.ndarray, row: np.ndarray) -> np.ndarray:
            return np.sum(np.abs(arr - row), axis=1)

        arr0 = np.array([[3, 3, 3]])
        arr1 = np.array([[3, 4, 4], [4, 4, 4]])
        matches = find_matches(arr0, arr1, dist=manhatten_dist, tol=2.001)
        assert is_same(matches, [[0, 0]])

    def test_value_error_on_incorrect_dist_func(self) -> None:
        def my_incorrect_dist_func(*_) -> int:
            return 1

        arr0 = np.ones((5, 4))
        arr1 = np.ones((5, 4))
        with pytest.raises(ValueError):
            find_matches(arr0, arr1, dist=my_incorrect_dist_func)  # type: ignore

    def test_value_error_on_incorrect_dist_str(self) -> None:
        arr0 = np.ones((5, 4))
        arr1 = np.ones((5, 4))
        with pytest.raises(ValueError):
            find_matches(arr0, arr1, dist='non-existing func')

    def test_assert_error_on_zero_tol(self) -> None:
        arr0 = np.ones((5, 4))
        arr1 = np.ones((5, 4))
        with pytest.raises(AssertionError):
            find_matches(arr0, arr1, tol=0)

    def test_assert_error_on_too_small_bucket_tol_mult(self) -> None:
        arr0 = np.ones((5, 4))
        arr1 = np.ones((5, 4))
        with pytest.raises(AssertionError):
            find_matches(arr0, arr1, bucket_tol_mult=0)

    def test_assert_error_on_non_equiv_shape(self) -> None:
        arr0 = np.ones((5, 3))
        arr1 = np.ones((5, 4))
        with pytest.raises(AssertionError):
            find_matches(arr0, arr1)

    def test_assert_error_on_incorrect_shape(self) -> None:
        arr0 = np.ones((4, 5, 2))
        arr1 = np.ones((4, 5, 2))
        with pytest.raises(AssertionError):
            find_matches(arr0, arr1)
