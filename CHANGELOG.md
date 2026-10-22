# Changelog

## 0.3.0

The matching engine is rewritten in Rust (with [PyO3](https://pyo3.rs)) and uses a KD-tree instead of grid buckets.

### Added

- Much faster: 20x to 3,000x in the benchmarks in the README on a single thread, and up to 11,000x with all cores.
- New distance metrics: `'manhattan'` (`'cityblock'`, `'l1'`), `'sqeuclidean'` and `'minkowski'` (with `p=...`), next to `'euclidean'` (`'norm'`, `'l2'`), `'chebyshev'` (`'max'`, `'linf'`) and `'cosine'` (`'cos'`).
- `MatchIndex`, to build the tree over one array once and search it many times.
- `return_distances=True` returns the distance of every match.
- `n_threads` to search on several threads. The GIL is released during the search.
- Native float32 support: float32 inputs are searched in float32.
- Support for free-threaded Python (3.14t, 3.15t) and Python 3.13 to 3.15.
- `tol=0` finds exact duplicates; 1-D arrays are treated as a single column, and one-dimensional inputs are matched by sorting and sweeping, which is faster still.

### Changed

- The search is exact for every metric. Before, cosine matches could be missed unless `bucket_tol_mult` was large.
- The matches are always sorted lexicographically, and an empty result has shape `(0, 2)` instead of `(0,)`.
- Invalid arguments raise `ValueError`/`TypeError` instead of `AssertionError`.
- Rows with NaN or infinite values never match.

### Removed

- Custom distance functions (callables as `dist`).
- The `bucket_tol_mult` argument.
- Support for Python 3.8 to 3.10.

## 0.2.3

- Support Python 3.12.
