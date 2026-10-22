"""
Benchmark find_matches on one thread and on all cores, with SciPy's KDTree as a
reference point.

    uv run --group bench python benchmarks/bench.py [--quick]

Every scenario checks that SciPy returns the same matches.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from close_numerical_matches import Metric
from close_numerical_matches import find_matches

try:
    from scipy.spatial import KDTree
except ImportError:  # pragma: no cover
    KDTree = None


@dataclass
class Scenario:
    name: str
    arr0: NDArray[Any]
    arr1: NDArray[Any]
    dist: Metric
    tol: float


def clustered(
    rng: np.random.Generator, n: int, d: int, n_clusters: int
) -> NDArray[Any]:
    centers = rng.random((n_clusters, d))
    return centers[rng.integers(0, n_clusters, n)] + rng.normal(0, 0.01, (n, d))


def scenarios(quick: bool) -> list[Scenario]:
    rng = np.random.default_rng(0)
    s = 10 if quick else 1
    cos_data = clustered(rng, 40_000 // s, 32, 200) - 0.5
    clus_data = clustered(rng, 100_000 // s, 8, 1000)
    return [
        Scenario(
            '2-D: 320k x 44k',
            rng.random((320_000 // s, 2)),
            rng.random((44_000 // s, 2)),
            'euclidean',
            0.001,
        ),
        Scenario(
            '1-D: 1M x 1M',
            rng.random((1_000_000 // s, 1)),
            rng.random((1_000_000 // s, 1)),
            'euclidean',
            1e-7,
        ),
        Scenario(
            '3-D: 200k x 200k',
            rng.random((200_000 // s, 3)),
            rng.random((200_000 // s, 3)),
            'euclidean',
            0.003,
        ),
        Scenario(
            '3-D chebyshev: 200k x 200k',
            rng.random((200_000 // s, 3)),
            rng.random((200_000 // s, 3)),
            'chebyshev',
            0.002,
        ),
        Scenario(
            '8-D clustered: 50k x 50k',
            clus_data[: len(clus_data) // 2],
            clus_data[len(clus_data) // 2 :],
            'euclidean',
            0.02,
        ),
        Scenario(
            '32-D cosine, clustered: 20k x 20k',
            cos_data[: len(cos_data) // 2],
            cos_data[len(cos_data) // 2 :],
            'cosine',
            0.001,
        ),
        Scenario(
            'Many matches: 3k x 3k, 2-D',
            rng.random((3_000, 2)),
            rng.random((3_000, 2)),
            'euclidean',
            0.3,
        ),
        Scenario(
            'Tiny: 100 x 100, 2-D',
            rng.random((100, 2)),
            rng.random((100, 2)),
            'euclidean',
            0.05,
        ),
    ]


def best_time(
    fn: Callable[[], Any], min_time: float = 1.0, max_reps: int = 1000
) -> tuple[float, Any]:
    result = fn()
    times: list[float] = []
    total = 0.0
    while total < min_time and len(times) < max_reps:
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        total += elapsed
    return min(times), result


def as_set(matches: Any) -> set[tuple[int, int]]:
    return {(int(i), int(j)) for i, j in np.asarray(matches).reshape(-1, 2)}


def scipy_matches(sc: Scenario) -> NDArray[np.int64]:
    assert KDTree is not None
    a0, a1, tol = sc.arr0, sc.arr1, sc.tol
    p = {'euclidean': 2.0, 'chebyshev': np.inf}.get(sc.dist)
    if sc.dist == 'cosine':
        a0 = a0 / np.linalg.norm(a0, axis=1, keepdims=True)
        a1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
        tol, p = float(np.sqrt(2 * tol)), 2.0
    assert p is not None
    pairs = KDTree(a0).sparse_distance_matrix(
        KDTree(a1), tol, p=p, output_type='ndarray'
    )
    return np.column_stack([pairs['i'], pairs['j']]).astype(np.int64)


def fmt(seconds: float | None) -> str:
    if seconds is None:
        return '-'
    if seconds < 1e-3:
        return f'{seconds * 1e6:.0f} µs'
    if seconds < 1:
        return f'{seconds * 1e3:.1f} ms'
    return f'{seconds:.2f} s'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='10x smaller inputs')
    args = parser.parse_args()

    n_cpu = os.cpu_count()
    print(f'Python {sys.version.split()[0]}, numpy {np.__version__}, {n_cpu} CPUs')
    print('| Scenario | Matches | 1 thread | All cores | SciPy KDTree |')
    print('|---|---:|---:|---:|---:|')
    for sc in scenarios(args.quick):
        t1, res = best_time(
            lambda sc=sc: find_matches(sc.arr0, sc.arr1, sc.dist, sc.tol)
        )
        tn, res_n = best_time(
            lambda sc=sc: find_matches(
                sc.arr0, sc.arr1, sc.dist, sc.tol, n_threads=None
            )
        )
        assert (res == res_n).all()
        expected = as_set(res)
        assert len(expected) == len(res)

        t_scipy = None
        if KDTree is not None:
            t_scipy, sp = best_time(lambda sc=sc: scipy_matches(sc), min_time=0.5)
            got = as_set(sp)
            if sc.dist == 'cosine':
                # SciPy has no cosine metric; via unit vectors it can differ by
                # rounding right at the tolerance.
                assert len(got ^ expected) <= len(expected) * 1e-4, sc.name
            else:
                assert got == expected, f'{sc.name}: SciPy disagrees'

        print(
            f'| {sc.name} | {len(res):,} | {fmt(t1)} | {fmt(tn)} | {fmt(t_scipy)} |',
            flush=True,
        )


if __name__ == '__main__':
    main()
