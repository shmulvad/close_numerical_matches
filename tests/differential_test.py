"""
Compares the Rust engine against a numpy brute-force oracle and the naive Rust
search.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.typing import NDArray

from close_numerical_matches import MatchIndex
from close_numerical_matches import find_matches
from close_numerical_matches import naive_find_matches

ALL_METRICS: list[tuple[str, float | None]] = [
    ('euclidean', None),
    ('sqeuclidean', None),
    ('manhattan', None),
    ('chebyshev', None),
    ('cosine', None),
    ('minkowski', 1.5),
    ('minkowski', 3.0),
]


def brute_force(
    arr0: NDArray[Any],
    arr1: NDArray[Any],
    dist: str,
    p: float | None = None,
) -> NDArray[np.float64]:
    """All pairwise distances in float64; NaN where a pair can never match."""
    a0 = np.asarray(arr0, dtype=np.float64)
    a1 = np.asarray(arr1, dtype=np.float64)
    with np.errstate(all='ignore'):
        diff = np.abs(a0[:, None, :] - a1[None, :, :])
        if dist == 'euclidean':
            dists = np.sqrt((diff**2).sum(-1))
        elif dist == 'sqeuclidean':
            dists = (diff**2).sum(-1)
        elif dist == 'manhattan':
            dists = diff.sum(-1)
        elif dist == 'chebyshev':
            dists = diff.max(-1, initial=0.0)
        elif dist == 'minkowski':
            assert p is not None
            dists = (diff**p).sum(-1) ** (1 / p)
        elif dist == 'cosine':
            n0 = np.linalg.norm(a0, axis=1, keepdims=True)
            n1 = np.linalg.norm(a1, axis=1, keepdims=True)
            dists = np.clip(1.0 - (a0 / n0) @ (a1 / n1).T, 0.0, 2.0)
            dists[(n0 == 0).ravel(), :] = np.nan
            dists[:, (n1 == 0).ravel()] = np.nan
        else:
            raise ValueError(dist)
    dists[~np.isfinite(a0).all(1), :] = np.nan
    dists[:, ~np.isfinite(a1).all(1)] = np.nan
    return dists


def assert_matches_oracle(
    matches: NDArray[np.int64],
    distances: NDArray[Any] | None,
    oracle: NDArray[np.float64],
    tol: float,
    float32: bool = False,
) -> None:
    """
    `matches` must hold every pair clearly within `tol` and no pair clearly outside,
    in sorted order. Pairs within rounding error of `tol` may go either way.
    """
    eps = (1e-4 if float32 else 1e-9) * max(1.0, tol)
    assert matches.dtype == np.int64
    assert matches.ndim == 2
    assert matches.shape[1] == 2
    got = {(int(i), int(j)) for i, j in matches}
    assert len(got) == len(matches), 'duplicate matches'
    order = np.lexsort((matches[:, 1], matches[:, 0]))
    assert (order == np.arange(len(matches))).all(), 'matches are not sorted'
    with np.errstate(invalid='ignore'):
        sure = {(int(i), int(j)) for i, j in np.argwhere(oracle < tol - eps)}
        maybe = {(int(i), int(j)) for i, j in np.argwhere(oracle <= tol + eps)}
    assert sure <= got, f'missing matches: {sorted(sure - got)[:10]}'
    assert got <= maybe, f'extra matches: {sorted(got - maybe)[:10]}'
    if distances is not None:
        assert distances.shape == (len(matches),)
        expected = oracle[matches[:, 0], matches[:, 1]]
        np.testing.assert_allclose(
            distances, expected, rtol=1e-4 if float32 else 1e-9, atol=eps
        )


def make_data(kind: str, n: int, d: int, rng: np.random.Generator) -> NDArray[Any]:
    if kind == 'uniform':
        return rng.random((n, d))
    if kind == 'clustered':
        centers = rng.random((5, d)) * 10
        return centers[rng.integers(0, 5, n)] + rng.normal(0, 0.05, (n, d))
    if kind == 'integers':
        return rng.integers(-3, 4, (n, d))
    if kind == 'duplicates':
        base = rng.random((max(n // 20, 1), d))
        return base[rng.integers(0, len(base), n)]
    if kind == 'wide-range':
        return rng.random((n, d)) * 10.0 ** rng.integers(-3, 4, (1, d))
    raise ValueError(kind)


KINDS = ['uniform', 'clustered', 'integers', 'duplicates', 'wide-range']


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('d', [1, 2, 3, 5, 10])
@pytest.mark.parametrize(('dist', 'p'), ALL_METRICS)
def test_against_brute_force(kind: str, d: int, dist: str, p: float | None) -> None:
    rng = np.random.default_rng([d, len(kind), len(dist)])
    data = make_data(kind, 500, d, rng)
    arr0, arr1 = data[:300], data[300:]
    oracle = brute_force(arr0, arr1, dist, p)
    finite = oracle[np.isfinite(oracle)]
    # Tolerances that give few, some and many matches.
    tols = sorted({0.0, *np.quantile(finite, [0.001, 0.02, 0.3]).tolist()})
    for tol in tols:
        for n_threads in (1, 4):
            matches, distances = find_matches(
                arr0,
                arr1,
                dist,  # type: ignore[arg-type]
                tol,
                p=p,
                n_threads=n_threads,
                return_distances=True,
            )
            assert_matches_oracle(matches, distances, oracle, tol)
            plain = find_matches(arr0, arr1, dist, tol, p=p, n_threads=n_threads)  # type: ignore[arg-type]
            assert (plain == matches).all()


@pytest.mark.parametrize(('dist', 'p'), ALL_METRICS)
def test_float32(dist: str, p: float | None) -> None:
    rng = np.random.default_rng(3)
    arr0 = rng.random((400, 3), dtype=np.float32)
    arr1 = rng.random((300, 3), dtype=np.float32)
    oracle = brute_force(arr0, arr1, dist, p)
    tol = float(np.quantile(oracle, 0.01))
    matches, distances = find_matches(
        arr0,
        arr1,
        dist,  # type: ignore[arg-type]
        tol,
        p=p,
        return_distances=True,
    )
    assert distances.dtype == np.float32
    assert_matches_oracle(matches, distances, oracle, tol, float32=True)


@pytest.mark.parametrize(('dist', 'p'), ALL_METRICS)
def test_tree_naive_and_index_agree_exactly(dist: str, p: float | None) -> None:
    rng = np.random.default_rng(11)
    data = make_data('clustered', 5500, 2, rng)
    arr0, arr1 = data[:3000], data[3000:]
    tol = 0.02
    kwargs: dict[str, Any] = {'p': p, 'return_distances': True}
    expected, expected_dists = naive_find_matches(arr0, arr1, dist, tol, **kwargs)  # type: ignore[arg-type]
    assert len(expected) > 0
    results = [
        find_matches(arr0, arr1, dist, tol, n_threads=n, **kwargs)  # type: ignore[arg-type]
        for n in (1, 3, None)
    ]
    results.append(MatchIndex(arr1, dist, p=p).query(arr0, tol, return_distances=True))  # type: ignore[arg-type]
    for matches, dists in results:
        np.testing.assert_array_equal(matches, expected)
        np.testing.assert_array_equal(dists, expected_dists)


def points(n_max: int, d: int) -> st.SearchStrategy[NDArray[np.float64]]:
    elements = st.one_of(
        st.integers(-4, 4).map(float),
        st.floats(-5, 5, allow_nan=False, width=32),
        st.sampled_from([np.nan, np.inf, -np.inf, 0.0, 1e-300, 1e300]),
    )
    shape = st.tuples(st.integers(0, n_max), st.just(d))
    return hnp.arrays(np.float64, shape, elements=elements)


@settings(max_examples=150, deadline=None)
@given(
    data=st.data(),
    d=st.integers(0, 4),
    metric=st.sampled_from(ALL_METRICS),
    tol=st.one_of(st.sampled_from([0.0, 0.5, 1.0, 2.0]), st.floats(0, 10)),
)
def test_property_tree_equals_naive_and_oracle(
    data: st.DataObject,
    d: int,
    metric: tuple[str, float | None],
    tol: float,
) -> None:
    dist, p = metric
    arr0 = data.draw(points(40, d))
    arr1 = data.draw(points(40, d))
    fast, fast_dists = find_matches(arr0, arr1, dist, tol, p=p, return_distances=True)  # type: ignore[arg-type]
    naive, naive_dists = naive_find_matches(
        arr0,
        arr1,
        dist,  # type: ignore[arg-type]
        tol,
        p=p,
        return_distances=True,
    )
    np.testing.assert_array_equal(fast, naive)
    np.testing.assert_array_equal(fast_dists, naive_dists)
    # The oracle overflows or underflows for extreme values, where the engine
    # (which rescales before normalising) does not; only compare ordinary values.
    values = np.abs(np.concatenate([arr0.ravel(), arr1.ravel()]))
    values = values[np.isfinite(values) & (values > 0)]
    if ((values < 1e100) & (values > 1e-100)).all():
        oracle = brute_force(arr0, arr1, dist, p)
        assert_matches_oracle(fast, fast_dists, oracle, tol)
