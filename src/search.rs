//! Finding all close pairs: picks the searcher, runs the queries (in parallel if
//! asked) and assembles the sorted output.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::kdtree::{Brute, KdTree, Searcher};
use crate::metric::{Distance, Metric};
use crate::points::Points;
use crate::scalar::Scalar;
use crate::with_kernel;

/// Below this many queries per thread, starting threads costs more than it saves.
const MIN_QUERIES_PER_THREAD: usize = 1024;
/// Work is handed out in chunks; this many per thread balances uneven chunks.
const CHUNKS_PER_THREAD: usize = 8;

/// Matching pairs as a flat row-major `(k, 2)` array of input row indices, sorted
/// lexicographically, plus the distances (empty unless requested).
#[derive(Clone, Debug, PartialEq)]
pub struct Matches<T> {
    pub pairs: Vec<i64>,
    pub dists: Vec<T>,
}

impl<T> Default for Matches<T> {
    fn default() -> Self {
        Self {
            pairs: Vec::new(),
            dists: Vec::new(),
        }
    }
}

impl<T> Matches<T> {
    #[inline(always)]
    fn push(&mut self, i: i64, j: usize, dist: T, want_dist: bool) {
        self.pairs.extend_from_slice(&[i, j as i64]);
        if want_dist {
            self.dists.push(dist);
        }
    }

    pub fn len(&self) -> usize {
        self.pairs.len() / 2
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// The number of threads to use for `requested` (0 means one per CPU core).
pub fn resolve_threads(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        requested
    }
}

/// Runs `f(chunk)` for every chunk in `0..n_chunks` on up to `n_threads` threads and
/// returns the results in chunk order.
fn par_map<R: Send>(n_chunks: usize, n_threads: usize, f: impl Fn(usize) -> R + Sync) -> Vec<R> {
    if n_threads <= 1 || n_chunks <= 1 {
        return (0..n_chunks).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut slots: Vec<Option<R>> = (0..n_chunks).map(|_| None).collect();
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..n_threads.min(n_chunks))
            .map(|_| {
                scope.spawn(|| {
                    let mut done = Vec::new();
                    loop {
                        let chunk = next.fetch_add(1, Ordering::Relaxed);
                        if chunk >= n_chunks {
                            return done;
                        }
                        done.push((chunk, f(chunk)));
                    }
                })
            })
            .collect();
        for worker in workers {
            let done = worker.join().expect("search thread panicked");
            for (chunk, result) in done {
                slots[chunk] = Some(result);
            }
        }
    });
    slots
        .into_iter()
        .map(|r| r.expect("every chunk is processed"))
        .collect()
}

/// Splits `n` queries into chunks for `n_threads` threads.
fn chunking(n: usize, n_threads: usize) -> (usize, usize) {
    let n_threads = n_threads.min(n / MIN_QUERIES_PER_THREAD).max(1);
    if n_threads == 1 {
        return (1, n_threads);
    }
    let chunk = n.div_ceil(n_threads * CHUNKS_PER_THREAD).max(256);
    (n.div_ceil(chunk), n_threads)
}

/// The query range of chunk `c` out of `n_chunks` for `n` queries.
fn chunk_range(c: usize, n_chunks: usize, n: usize) -> std::ops::Range<usize> {
    let size = n.div_ceil(n_chunks.max(1));
    (c * size).min(n)..((c + 1) * size).min(n)
}

/// Reordering the queries only pays off when there are enough of them.
const REORDER_MIN_QUERIES: usize = 4096;

/// An order of the queries in which consecutive queries are close to each other, so
/// they visit the same tree nodes while those are still in the CPU cache. `None`
/// keeps the input order.
fn query_order<T: Scalar, S: Searcher<T>>(
    searcher: &S,
    queries: &Points<T>,
    n_threads: usize,
) -> Option<Vec<usize>> {
    let n = queries.len();
    let n_keys = searcher.n_locality_keys();
    if n < REORDER_MIN_QUERIES || n_keys <= 1 {
        return None;
    }
    let (n_chunks, n_threads) = chunking(n, n_threads);
    let keys: Vec<Vec<usize>> = par_map(n_chunks, n_threads, |c| {
        chunk_range(c, n_chunks, n)
            .map(|qi| searcher.locality_key(queries.row(qi)))
            .collect()
    });
    let mut starts = vec![0usize; n_keys + 1];
    for &key in keys.iter().flatten() {
        starts[key + 1] += 1;
    }
    for i in 1..starts.len() {
        starts[i] += starts[i - 1];
    }
    let mut order = vec![0usize; n];
    for (qi, &key) in keys.iter().flatten().enumerate() {
        order[starts[key]] = qi;
        starts[key] += 1;
    }
    Some(order)
}

/// Searches every query (in `order`, if given) and returns the pairs
/// `(query row, stored row)` grouped by query, in parts (one per chunk of work).
/// Within a query, the stored rows are sorted if `sort_each` is set.
#[allow(clippy::too_many_arguments)]
fn run_queries<T: Scalar, S: Searcher<T>, M: Distance<T>>(
    searcher: &S,
    m: &M,
    queries: &Points<T>,
    order: Option<&[usize]>,
    tol: T,
    n_threads: usize,
    want_dist: bool,
    sort_each: bool,
) -> Vec<Matches<T>> {
    let th = m.thresholds(tol, queries.dims());
    let n = queries.len();
    let (n_chunks, n_threads) = chunking(n, n_threads);
    par_map(n_chunks, n_threads, |c| {
        let mut out = Matches::default();
        let mut stack = Vec::new();
        let mut found: Vec<(usize, T)> = Vec::new();
        for pos in chunk_range(c, n_chunks, n) {
            let qi = order.map_or(pos, |o| o[pos]);
            let q = queries.row(qi);
            let q_orig = queries.orig(qi) as i64;
            if sort_each && !S::SORTED {
                // Collect this query's matches first, to sort them.
                found.clear();
                searcher.search(m, q, &th, want_dist, &mut stack, &mut |j, dist| {
                    found.push((j, dist));
                });
                found.sort_unstable_by_key(|&(j, _)| j);
                for &(j, dist) in &found {
                    out.push(q_orig, j, dist, want_dist);
                }
            } else {
                searcher.search(m, q, &th, want_dist, &mut stack, &mut |j, dist| {
                    out.push(q_orig, j, dist, want_dist);
                });
            }
        }
        out
    })
}

/// Concatenates the parts.
fn concat<T: Scalar>(mut parts: Vec<Matches<T>>) -> Matches<T> {
    if parts.len() == 1 {
        return parts.pop().unwrap_or_default();
    }
    let total: usize = parts.iter().map(Matches::len).sum();
    let want_dist = parts.iter().any(|p| !p.dists.is_empty());
    let mut out = Matches {
        pairs: Vec::with_capacity(2 * total),
        dists: Vec::with_capacity(if want_dist { total } else { 0 }),
    };
    for part in parts {
        out.pairs.extend_from_slice(&part.pairs);
        out.dists.extend_from_slice(&part.dists);
    }
    out
}

/// Stable counting sort of the pairs of all parts by column `col` (values in
/// `0..n_keys`). With `swap`, the two columns of every pair are swapped in the output.
fn counting_sort<T: Scalar>(
    parts: &[Matches<T>],
    col: usize,
    n_keys: usize,
    swap: bool,
) -> Matches<T> {
    let mut starts = vec![0usize; n_keys + 1];
    for part in parts {
        for pair in part.pairs.chunks_exact(2) {
            starts[pair[col] as usize + 1] += 1;
        }
    }
    for i in 1..starts.len() {
        starts[i] += starts[i - 1];
    }
    let total = starts[n_keys];
    let want_dist = parts.iter().any(|p| !p.dists.is_empty());
    let mut out = Matches {
        pairs: vec![0; 2 * total],
        dists: if want_dist {
            vec![T::ZERO; total]
        } else {
            Vec::new()
        },
    };
    let (a, b) = if swap { (1, 0) } else { (0, 1) };
    for part in parts {
        for (k, pair) in part.pairs.chunks_exact(2).enumerate() {
            let slot = &mut starts[pair[col] as usize];
            let pos = *slot;
            *slot += 1;
            out.pairs[2 * pos] = pair[a];
            out.pairs[2 * pos + 1] = pair[b];
            if want_dist {
                out.dists[pos] = part.dists[k];
            }
        }
    }
    out
}

/// Searches every query and returns pairs `(query row, stored row)`, sorted.
fn query_rows<T: Scalar, S: Searcher<T>, M: Distance<T>>(
    searcher: &S,
    m: &M,
    queries: &Points<T>,
    tol: T,
    n_threads: usize,
    want_dist: bool,
) -> Matches<T> {
    let order = query_order(searcher, queries, n_threads);
    let parts = run_queries(
        searcher,
        m,
        queries,
        order.as_deref(),
        tol,
        n_threads,
        want_dist,
        true,
    );
    match order {
        // Each query's matches are sorted, so a stable sort by query completes it.
        Some(_) => counting_sort(&parts, 0, queries.n_input(), false),
        None => concat(parts),
    }
}

/// Searches every query and returns pairs `(stored row, query row)`, sorted. Used when
/// the tree is built on the first array of `find_matches`.
fn query_rows_transposed<T: Scalar, S: Searcher<T>, M: Distance<T>>(
    searcher: &S,
    m: &M,
    queries: &Points<T>,
    n_stored_input: usize,
    tol: T,
    n_threads: usize,
    want_dist: bool,
) -> Matches<T> {
    let order = query_order(searcher, queries, n_threads);
    let mut parts = run_queries(
        searcher,
        m,
        queries,
        order.as_deref(),
        tol,
        n_threads,
        want_dist,
        false,
    );
    if order.is_some() {
        parts = vec![counting_sort(&parts, 0, queries.n_input(), false)];
    }
    // The pairs are now in increasing query row order, so a stable sort by stored row
    // sorts them completely (a two-pass radix sort).
    counting_sort(&parts, 1, n_stored_input, true)
}

/// A KD-tree over one set of points, built once and queried many times.
#[derive(Clone, Debug)]
pub struct Index<T> {
    metric: Metric,
    tree: KdTree<T>,
    n_input: usize,
    d: usize,
}

impl<T: Scalar> Index<T> {
    /// Builds an index over `points` on up to `n_threads` threads. The points must
    /// have been normalised if the metric needs it ([`Metric::normalizes`]).
    pub fn new(points: Points<T>, metric: Metric, n_threads: usize) -> Self {
        Self {
            metric,
            n_input: points.n_input(),
            d: points.dims(),
            tree: KdTree::build(points, n_threads),
        }
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The number of rows the index was built from (including rows that can never
    /// match).
    pub fn n_input(&self) -> usize {
        self.n_input
    }

    pub fn dims(&self) -> usize {
        self.d
    }

    /// All pairs `(query row, index row)` within `tol`. The queries must have the
    /// same number of dimensions and be normalised like the index.
    pub fn query(
        &self,
        queries: &Points<T>,
        tol: T,
        n_threads: usize,
        want_dist: bool,
    ) -> Matches<T> {
        assert_eq!(
            queries.dims(),
            self.d,
            "the queries have the wrong dimensions"
        );
        with_kernel!(self.metric.kernel(), tol, m => {
            query_rows(&self.tree, &m, queries, tol, n_threads, want_dist)
        })
    }
}

/// All pairs `(i, j)` where row `i` of `p0` and row `j` of `p1` are within `tol` of
/// each other, sorted lexicographically. The points must have the same number of
/// dimensions and be normalised if the metric needs it ([`Metric::normalizes`]).
///
/// The tree is built over the smaller set and searched with the larger one. With
/// `naive`, every pair is compared instead (for testing and reference).
pub fn find_matches<T: Scalar>(
    p0: Points<T>,
    p1: Points<T>,
    metric: Metric,
    tol: T,
    n_threads: usize,
    want_dist: bool,
    naive: bool,
) -> Matches<T> {
    assert_eq!(p0.dims(), p1.dims(), "the points have different dimensions");
    if p0.is_empty() || p1.is_empty() {
        return Matches::default();
    }
    let n0 = p0.n_input();
    with_kernel!(metric.kernel(), tol, m => {
        if naive {
            query_rows(&Brute { points: &p1 }, &m, &p0, tol, n_threads, want_dist)
        } else if p0.len() <= p1.len() {
            let tree = KdTree::build(p0, n_threads);
            query_rows_transposed(&tree, &m, &p1, n0, tol, n_threads, want_dist)
        } else {
            let tree = KdTree::build(p1, n_threads);
            query_rows(&tree, &m, &p0, tol, n_threads, want_dist)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::METRIC_NAMES;

    /// A small deterministic generator (xorshift), so the tests need no dependencies.
    struct Rng(u64);

    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn values(&mut self, n: usize, d: usize, scale: f64, round: bool) -> Vec<f64> {
            (0..n * d)
                .map(|_| {
                    let v = self.next_f64() * scale;
                    if round { v.round() } else { v }
                })
                .collect()
        }
    }

    /// `find_matches` on row-major slices.
    #[allow(clippy::too_many_arguments)]
    fn run<T: Scalar>(
        a0: &[T],
        n0: usize,
        a1: &[T],
        n1: usize,
        d: usize,
        metric: Metric,
        tol: T,
        n_threads: usize,
        want_dist: bool,
        naive: bool,
    ) -> Matches<T> {
        let p0 = Points::new(a0, n0, d, metric.normalizes());
        let p1 = Points::new(a1, n1, d, metric.normalizes());
        find_matches(p0, p1, metric, tol, n_threads, want_dist, naive)
    }

    fn pairs(m: &Matches<impl Sized>) -> Vec<(i64, i64)> {
        m.pairs.chunks_exact(2).map(|p| (p[0], p[1])).collect()
    }

    fn check_against_naive<T: Scalar>(a0: &[T], n0: usize, a1: &[T], n1: usize, d: usize) {
        for name in METRIC_NAMES {
            let metric = Metric::parse(name, None).unwrap();
            for tol in [0.0, 0.05, 0.3, 1.0, 2.5] {
                let tol = T::from_f64(tol);
                let naive = run(a0, n0, a1, n1, d, metric, tol, 1, true, true);
                for threads in [1, 3] {
                    let fast = run(a0, n0, a1, n1, d, metric, tol, threads, true, false);
                    assert_eq!(pairs(&fast), pairs(&naive), "{name} tol={tol:?} d={d}");
                    assert_eq!(fast.dists, naive.dists, "{name} tol={tol:?} d={d}");
                    let mut sorted = pairs(&fast);
                    sorted.sort_unstable();
                    assert_eq!(sorted, pairs(&fast), "output must be sorted");
                }
                let index = Index::new(Points::new(a1, n1, d, metric.normalizes()), metric, 2);
                let queries = Points::new(a0, n0, d, metric.normalizes());
                let via_index = index.query(&queries, tol, 2, true);
                assert_eq!(pairs(&via_index), pairs(&naive), "index {name} d={d}");
            }
        }
    }

    #[test]
    fn tree_matches_brute_force() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for d in [1, 2, 3, 5, 8] {
            for (n0, n1) in [(300, 200), (150, 400), (1, 50), (64, 1)] {
                let a0 = rng.values(n0, d, 3.0, false);
                let a1 = rng.values(n1, d, 3.0, false);
                check_against_naive(&a0, n0, &a1, n1, d);
                let a0: Vec<f32> = a0.iter().map(|&v| v as f32).collect();
                let a1: Vec<f32> = a1.iter().map(|&v| v as f32).collect();
                check_against_naive(&a0, n0, &a1, n1, d);
            }
        }
    }

    #[test]
    fn duplicates_and_integers() {
        let mut rng = Rng(42);
        let a0 = rng.values(400, 2, 4.0, true);
        let a1 = rng.values(300, 2, 4.0, true);
        check_against_naive(&a0, 400, &a1, 300, 2);
    }

    #[test]
    fn many_queries_in_parallel() {
        let mut rng = Rng(7);
        let (n0, n1, d) = (5000, 3000, 2);
        let a0 = rng.values(n0, d, 1.0, false);
        let a1 = rng.values(n1, d, 1.0, false);
        let metric = Metric::Euclidean;
        let one = run(&a0, n0, &a1, n1, d, metric, 0.01, 1, true, false);
        let many = run(&a0, n0, &a1, n1, d, metric, 0.01, 8, true, false);
        let naive = run(&a0, n0, &a1, n1, d, metric, 0.01, 4, true, true);
        assert!(!one.is_empty());
        assert_eq!(one, many);
        assert_eq!(one, naive);
    }

    #[test]
    fn empty_and_zero_dimensional_inputs() {
        let none = run::<f64>(
            &[],
            0,
            &[1.0, 2.0],
            1,
            2,
            Metric::Euclidean,
            1.0,
            1,
            false,
            false,
        );
        assert!(none.is_empty());
        let all = run::<f64>(&[], 2, &[], 3, 0, Metric::Euclidean, 0.0, 1, false, false);
        assert_eq!(
            pairs(&all),
            vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        );
        let cos = run::<f64>(&[], 2, &[], 3, 0, Metric::Cosine, 1.0, 1, false, false);
        assert!(cos.is_empty());
    }

    #[test]
    fn infinite_tolerance_matches_every_finite_pair() {
        let a0 = [0.0, 1e300, f64::NAN, 5.0];
        let a1 = [-1e300, 3.0, f64::INFINITY];
        for name in METRIC_NAMES {
            let metric = Metric::parse(name, None).unwrap();
            let m = run(&a0, 4, &a1, 3, 1, metric, f64::INFINITY, 1, false, false);
            let expected: Vec<(i64, i64)> = if metric == Metric::Cosine {
                // 0.0 is a zero vector, so it never matches by cosine.
                vec![(1, 0), (1, 1), (3, 0), (3, 1)]
            } else {
                vec![(0, 0), (0, 1), (1, 0), (1, 1), (3, 0), (3, 1)]
            };
            assert_eq!(pairs(&m), expected, "{name}");
        }
    }

    #[test]
    fn squared_tolerance_overflow_does_not_accept_far_pairs() {
        // 2e19^2 overflows f32, which used to make every box "fully within tol".
        let m = run::<f32>(
            &[0.0],
            1,
            &[3e19, 1.5e19],
            2,
            1,
            Metric::Euclidean,
            2e19,
            1,
            true,
            false,
        );
        assert_eq!(pairs(&m), vec![(0, 1)]);
        assert!((m.dists[0] - 1.5e19).abs() < 1e13);
    }

    #[test]
    fn powers_that_underflow_are_rescaled() {
        // 0.6^2000 and 0.5^2000 both underflow to 0 in f64.
        let metric = Metric::Minkowski(2000.0);
        for naive in [false, true] {
            let m = run::<f64>(&[0.0], 1, &[0.6, 0.4], 2, 1, metric, 0.5, 1, true, naive);
            assert_eq!(pairs(&m), vec![(0, 1)], "naive={naive}");
            assert!((m.dists[0] - 0.4).abs() < 1e-12);
        }
        let tiny = run::<f64>(
            &[0.0],
            1,
            &[2e-165, 5e-166],
            2,
            1,
            Metric::Euclidean,
            1e-165,
            1,
            true,
            false,
        );
        assert_eq!(pairs(&tiny), vec![(0, 1)]);
        assert!((tiny.dists[0] - 5e-166).abs() < 1e-178);
    }

    #[test]
    fn boundary_is_inclusive() {
        let a0 = [0.0, 0.0];
        let a1 = [0.0, 1.0, 1.0, 1.0];
        for (name, tol, expected) in [
            ("euclidean", 1.0, vec![(0, 0)]),
            ("chebyshev", 1.0, vec![(0, 0), (0, 1)]),
            ("manhattan", 2.0, vec![(0, 0), (0, 1)]),
            ("sqeuclidean", 2.0, vec![(0, 0), (0, 1)]),
        ] {
            let metric = Metric::parse(name, None).unwrap();
            let m = run(&a0, 1, &a1, 2, 2, metric, tol, 1, false, false);
            assert_eq!(pairs(&m), expected, "{name}");
        }
    }
}
