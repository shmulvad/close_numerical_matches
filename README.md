# Close Numerical Matches

[![PyPI - Downloads](https://img.shields.io/pypi/dm/close_numerical_matches)][pypi]
[![PyPI - Version](https://img.shields.io/pypi/v/close_numerical_matches)][pypi]
![GitHub Workflow Status (branch)](https://github.com/shmulvad/close_numerical_matches/workflows/CI/badge.svg)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/shmulvad/close_numerical_matches/main)][codefactor]
[![GitHub issues](https://img.shields.io/github/issues/shmulvad/close_numerical_matches)][issues]
[![GitHub license](https://img.shields.io/github/license/shmulvad/close_numerical_matches)][license]
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/close_numerical_matches)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)][makeAPullRequest]

This package finds close numerical matches *fast* across two 2D arrays of shape (n, d) and (m, d): every pair of rows that lie within a given tolerance of each other under a chosen distance metric. It returns the indices of the matches.

The engine is written in Rust and uses a KD-tree, so it stays fast for large arrays, clustered data and higher dimensions. It releases the GIL, can use several threads, and supports free-threaded Python.

## Installation

You can install `close-numerical-matches` from [PyPI][pypi]:

```bash
$ pip install close-numerical-matches
```

The package supports CPython 3.11 to 3.15, including the free-threaded builds (3.14t, 3.15t), and requires numpy. Pre-built wheels are published for Linux (x86_64, aarch64), macOS (x86_64, arm64) and Windows (x86_64). Installing from the source distribution requires a [Rust toolchain](https://rustup.rs/).

## How to use

Import `find_matches` from `close_numerical_matches` and supply two arrays of shape (n, d) and (m, d) and a tolerance. The result is a (k, 2) array of index pairs `[i, j]` where `arr0[i]` and `arr1[j]` match, sorted lexicographically:

```python
>>> import numpy as np
>>> from close_numerical_matches import find_matches
>>> arr0 = np.array([[25, 24], [50, 50], [25, 26]])
>>> arr1 = np.array([[25, 23], [25, 25], [50.6, 50.6], [60, 60]])
>>> find_matches(arr0, arr1, tol=1.0001)
array([[0, 0], [0, 1], [1, 2], [2, 1]])
>>> find_matches(arr0, arr1, tol=0.9999)
array([[1, 2]])
>>> find_matches(arr0, arr1, tol=0.60001)
array([], shape=(0, 2), dtype=int64)
>>> find_matches(arr0, arr1, tol=0.60001, dist='max')
array([[1, 2]])
>>> find_matches([[0, 0.05]], [[0, 5], [0, -0.01]], tol=0.1, dist='cos')
array([[0, 0]])
>>> matches = find_matches(arr0, arr1, tol=1.0001, dist='manhattan')
>>> matches
array([[0, 0], [0, 1], [2, 1]])
>>> indices0, indices1 = matches.T
>>> arr0[indices0]
array([[25, 24], [25, 24], [25, 26]])
```

The arguments in more detail:

* `arr0`, `arr1` : array-like of shape (n, d) and (m, d). A 1-D array is treated as a single column.
* `dist` : `str`, default `'euclidean'`. The distance metric, see the table below.
* `tol` : `float`, default `0.1`. Pairs with a distance of at most `tol` match. Must be >= 0; `tol=0` finds exact duplicates.
* `p` : `float`, optional. The order of the Minkowski distance (>= 1, default 2, `inf` allowed). Only allowed with `dist='minkowski'`.
* `n_threads` : `int | None`, default `1`. The number of threads to use; `None` or `0` uses one per CPU core. Small inputs use fewer threads.
* `return_distances` : `bool`, default `False`. Also return a (k,) array with the distance of each match.

Rows containing NaN or infinity never match, and nor do zero vectors with the cosine distance.

### Distance metrics

| `dist` | Aliases | Distance |
|---|---|---|
| `'euclidean'` | `'norm'`, `'l2'` | √Σ(xᵢ − yᵢ)² |
| `'sqeuclidean'` | | Σ(xᵢ − yᵢ)² |
| `'manhattan'` | `'cityblock'`, `'l1'` | Σ\|xᵢ − yᵢ\| |
| `'chebyshev'` | `'max'`, `'linf'` | maxᵢ \|xᵢ − yᵢ\| |
| `'minkowski'` | | (Σ\|xᵢ − yᵢ\|ᵖ)^(1/p), with `p=...` |
| `'cosine'` | `'cos'` | 1 − x·y / (‖x‖ ‖y‖), in [0, 2] |

All metrics are exact: every pair within the tolerance is found.

### Distances

```python
>>> matches, distances = find_matches(arr0, arr1, tol=1.0001, return_distances=True)
>>> distances.round(3)
array([1.   , 1.   , 0.849, 1.   ])
```

### Searching the same array many times

If you search one array again and again, build a `MatchIndex` over it once. `MatchIndex(arr1).query(arr0, tol)` returns the same as `find_matches(arr0, arr1, tol=tol)` when both arrays have the same dtype (the index computes in its own dtype, so a float32 index converts float64 queries to float32). The index is immutable, so threads can share it.

```python
>>> from close_numerical_matches import MatchIndex
>>> index = MatchIndex(arr1)
>>> index
MatchIndex(n_points=4, n_dims=2, dist='euclidean', dtype=float64)
>>> index.query(arr0, tol=1.0001)
array([[0, 0], [0, 1], [1, 2], [2, 1]])
>>> index.query([[60, 60.5]], tol=1)
array([[0, 3]])
```

### float32

If both arrays are float32, the search runs in float32 and returns float32 distances, which halves the memory for large arrays. Everything else is computed in float64.

```python
>>> _, distances = find_matches(
...     arr0.astype(np.float32), arr1.astype(np.float32), tol=1, return_distances=True
... )
>>> distances.dtype
dtype('float32')
```

### Threads and free-threaded Python

The search does not hold the GIL, so other Python threads keep running, and with `n_threads` it runs on several cores itself. The inputs are copied before the GIL is released, so other threads may change them during a search. On free-threaded Python (3.14t, 3.15t) the module does not re-enable the GIL, and `find_matches` and `MatchIndex.query` can be called from many threads at once. As with any numpy function, do not write to an input array from another thread while a call is reading it.

## How fast is it?

Benchmarked on an Apple M5 Pro (18 cores) with Python 3.15, with SciPy's `KDTree.sparse_distance_matrix` as a reference point (SciPy returns the same matches). Run `uv run --group bench python benchmarks/bench.py` to reproduce.

| Scenario | Matches | 1 thread | All cores | SciPy KDTree |
|---|---:|---:|---:|---:|
| 2-D: 320k x 44k | 44,453 | 35.5 ms | 6.1 ms | 66.9 ms |
| 1-D: 1M x 1M | 199,474 | 188.1 ms | 37.0 ms | 261.4 ms |
| 3-D: 200k x 200k | 4,416 | 46.3 ms | 9.2 ms | 109.5 ms |
| 3-D chebyshev: 200k x 200k | 2,580 | 42.7 ms | 9.0 ms | 111.8 ms |
| 8-D clustered: 50k x 50k | 47,588 | 17.1 ms | 4.4 ms | 148.9 ms |
| 32-D cosine, clustered: 20k x 20k | 666,344 | 45.3 ms | 10.2 ms | 1.46 s |
| Many matches: 3k x 3k, 2-D | 1,928,484 | 10.5 ms | 5.6 ms | 9.4 ms |
| Tiny: 100 x 100, 2-D | 59 | 8 µs | 8 µs | 21 µs |

## How it works

A KD-tree is built over the smaller of the two arrays: the points are split recursively at the median of their widest dimension, and every node stores the bounding box of its points. Each row of the other array then walks the tree and skips every node whose box is further away than `tol`. If a whole box lies within `tol`, all its points match without computing distances. Otherwise only the points in the leaves that are left are compared.

Some details that make it fast:

* Every metric is compiled into its own specialised search loop. Euclidean-like metrics compare squared distances while pruning, and cosine searches on unit vectors (where the cosine distance is half the squared Euclidean distance), so it is exact too.
* The queries are processed in tree order, so consecutive queries visit the same nodes while those are in the CPU cache.
* With `n_threads`, the tree is built and the queries are searched in parallel, and the result is still the same, sorted array.

For low dimensions the runtime is close to O((n + m) log(min(n, m)) + k) for k matches. In high dimensions or with very many matches no search structure can avoid comparing most pairs, but the tree never costs much more than comparing all pairs, which `naive_find_matches` does.

## Migrating from 0.2.x

* Custom distance functions are no longer supported (`dist` must be one of the built-in metrics). The old example of a custom Manhattan distance is now `dist='manhattan'`.
* `bucket_tol_mult` is gone: the search is exact for every metric, so there is nothing to tune.
* An empty result has shape `(0, 2)` instead of `(0,)`, and the matches are always sorted.
* Invalid arguments raise `ValueError` or `TypeError` instead of `AssertionError`.
* Python 3.11 or newer is required.

## When NOT to use this library

* If you expect a large fraction of all pairs to match, the output itself is huge, so this library cannot be much faster than comparing every pair.
* For very high-dimensional data (hundreds of dimensions), trees cannot prune much; look into approximate nearest-neighbour libraries.

## Development

You need [uv](https://docs.astral.sh/uv/) and a [Rust toolchain](https://rustup.rs/). All Python tools (ruff, pyright, pytest, prek) are pinned in `uv.lock`, and CI uses the same versions.

```bash
$ uv sync                     # create .venv, install the tools and build the extension
$ uv run pytest               # Python tests, incl. differential tests against numpy
$ uv run pyright              # type checking
$ cargo test                  # Rust unit tests (set PYO3_PYTHON to a Python interpreter)
$ uv run prek install         # run ruff, pyright, cargo fmt and more on every commit
```

After changing Rust code, `uv sync` rebuilds the extension (or run `uv run maturin develop --release`).

[pypi]: https://pypi.org/project/close-numerical-matches/
[license]: https://github.com/shmulvad/close_numerical_matches/blob/master/LICENSE
[issues]: https://github.com/shmulvad/close_numerical_matches/issues
[codefactor]: https://www.codefactor.io/repository/github/shmulvad/close_numerical_matches
[makeAPullRequest]: https://makeapullrequest.com
