"""
Thread-safety tests. They are most meaningful on free-threaded Python (3.14t/3.15t),
where the threads really run in parallel.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from close_numerical_matches import MatchIndex
from close_numerical_matches import find_matches

N_THREADS = 8


def run_threads(target: Callable[[int], Any], n: int = N_THREADS) -> list[Any]:
    """Run `target(i)` in `n` threads started together; re-raise any error."""
    barrier = threading.Barrier(n)
    results: list[Any] = [None] * n
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        barrier.wait()
        try:
            results[i] = target(i)
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


@pytest.mark.skipif(
    not sysconfig.get_config_var('Py_GIL_DISABLED'),
    reason='only relevant for free-threaded Python',
)
def test_import_keeps_the_gil_disabled() -> None:
    code = (
        'import sys\n'
        'import close_numerical_matches\n'
        'assert not sys._is_gil_enabled(), "importing re-enabled the GIL"\n'
    )
    subprocess.run([sys.executable, '-c', code], check=True, timeout=60)


def test_shared_index() -> None:
    rng = np.random.default_rng(0)
    index = MatchIndex(rng.random((5000, 3)))
    queries = [rng.random((2000 + 100 * i, 3)) for i in range(N_THREADS)]
    expected = [index.query(q, 0.03) for q in queries]

    def work(i: int) -> None:
        for _ in range(5):
            np.testing.assert_array_equal(index.query(queries[i], 0.03), expected[i])
            index.query(queries[-1 - i], 0.01, n_threads=2)

    run_threads(work)


def test_concurrent_find_matches() -> None:
    rng = np.random.default_rng(1)
    shared = rng.random((4000, 2))
    inputs = [(rng.random((3000, 2)), dist) for dist in ('norm', 'max', 'cos', 'l1')]
    expected = [find_matches(a, shared, d, 0.01) for a, d in inputs]  # type: ignore[arg-type]

    def work(i: int) -> None:
        a, dist = inputs[i % len(inputs)]
        for n_threads in (1, 3, 1, None):
            got = find_matches(a, shared, dist, 0.01, n_threads=n_threads)  # type: ignore[arg-type]
            np.testing.assert_array_equal(got, expected[i % len(inputs)])

    run_threads(work)


def test_building_indexes_in_parallel() -> None:
    rng = np.random.default_rng(2)
    data = rng.random((20_000, 2))
    queries = rng.random((500, 2))
    expected = MatchIndex(data).query(queries, 0.01)

    def work(_: int) -> Any:
        return MatchIndex(data).query(queries, 0.01)

    for got in run_threads(work):
        np.testing.assert_array_equal(got, expected)
