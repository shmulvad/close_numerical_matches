//! All close pairs in one dimension: sort both sets and sweep them together.
//!
//! In one dimension every metric is a monotone function of `|x - y|`, so the points
//! within the tolerance of a query form a contiguous window of the sorted other set,
//! and the window's start never moves backwards as the queries increase. Every
//! candidate is still checked with the kernel's own reduced distance and final test,
//! so the matches are exactly those of the tree and of the brute-force search.

use crate::metric::Distance;
use crate::points::Points;
use crate::scalar::Scalar;
use crate::search::{Matches, chunk_range, chunking, counting_sort, par_map};

/// The values of `points` with their input rows, sorted by value.
fn sorted<T: Scalar>(points: &Points<T>) -> Vec<(T, usize)> {
    let mut v: Vec<(T, usize)> = (0..points.len())
        .map(|i| (points.row(i)[0], points.orig(i)))
        .collect();
    v.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    v
}

/// All pairs `(i, j)` where row `i` of `p0` and row `j` of `p1` are within `tol` of
/// each other, sorted lexicographically. Both must be one-dimensional.
pub fn find_matches_1d<T: Scalar, M: Distance<T>>(
    p0: &Points<T>,
    p1: &Points<T>,
    m: &M,
    tol: T,
    n_threads: usize,
    want_dist: bool,
) -> Matches<T> {
    assert!(
        p0.dims() == 1 && p1.dims() == 1,
        "the points must be one-dimensional"
    );
    if p0.is_empty() || p1.is_empty() {
        return Matches::default();
    }
    let (q, x) = if n_threads > 1 {
        rayon::join(|| sorted(p0), || sorted(p1))
    } else {
        (sorted(p0), sorted(p1))
    };
    let th = m.thresholds(tol, 1);
    let (n_chunks, n_threads) = chunking(q.len(), n_threads);
    let parts = par_map(n_chunks, n_threads, |c| {
        let range = chunk_range(c, n_chunks, q.len());
        let mut out = Matches::default();
        if range.is_empty() {
            return out;
        }
        // The window of the first query: start below it and widen downwards.
        let first = q[range.start].0;
        let mut lo = x.partition_point(|&(v, _)| v < first);
        while lo > 0 && m.reduced(&[first], &[x[lo - 1].0]) <= th.prune {
            lo -= 1;
        }
        for &(qv, i) in &q[range] {
            // Points below the query that are too far away stay too far away.
            while lo < x.len() && x[lo].0 < qv && m.reduced(&[qv], &[x[lo].0]) > th.prune {
                lo += 1;
            }
            for &(xv, j) in &x[lo..] {
                let r = m.reduced(&[qv], &[xv]);
                if r > th.prune {
                    break;
                }
                let dist = m.finish(&[qv], &[xv], r);
                if dist <= th.tol {
                    out.push(i as i64, j, dist, want_dist);
                }
            }
        }
        out
    });
    // The pairs are grouped by query in value order; two stable passes sort them.
    let by_j = counting_sort(&parts, 1, p1.n_input(), false);
    counting_sort(&[by_j], 0, p0.n_input(), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kdtree::Brute;
    use crate::metric::{METRIC_NAMES, Metric};
    use crate::search::query_rows;
    use crate::with_kernel;

    fn brute<T: Scalar>(p0: &Points<T>, p1: &Points<T>, metric: Metric, tol: T) -> Matches<T> {
        with_kernel!(metric.kernel(), tol, m => {
            query_rows(&Brute { points: p1 }, &m, p0, tol, 1, true)
        })
    }

    fn sweep<T: Scalar>(
        p0: &Points<T>,
        p1: &Points<T>,
        metric: Metric,
        tol: T,
        n_threads: usize,
    ) -> Matches<T> {
        with_kernel!(metric.kernel(), tol, m => {
            find_matches_1d(p0, p1, &m, tol, n_threads, true)
        })
    }

    fn check<T: Scalar>(a0: &[T], a1: &[T]) {
        for name in METRIC_NAMES {
            let metric = Metric::parse(name, None).unwrap();
            let p0 = Points::new(a0, a0.len(), 1, metric.normalizes());
            let p1 = Points::new(a1, a1.len(), 1, metric.normalizes());
            for tol in [0.0, 0.05, 0.3, 1.0, 2.5, f64::INFINITY] {
                let tol = T::from_f64(tol);
                let expected = brute(&p0, &p1, metric, tol);
                for threads in [1, 3] {
                    let got = sweep(&p0, &p1, metric, tol, threads);
                    assert_eq!(got, expected, "{name} tol={tol:?} threads={threads}");
                }
            }
        }
    }

    #[test]
    fn sweep_matches_brute_force() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = |scale: f64, round: bool| -> f64 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let v = (seed >> 11) as f64 / (1u64 << 53) as f64 * scale - scale / 2.0;
            if round { v.round() } else { v }
        };
        for (n0, n1, scale, round) in [
            (300, 200, 3.0, false),
            (150, 400, 3.0, true),
            (1, 50, 1.0, false),
            (64, 1, 1.0, false),
            (2000, 1500, 20.0, false),
        ] {
            let a0: Vec<f64> = (0..n0).map(|_| rnd(scale, round)).collect();
            let a1: Vec<f64> = (0..n1).map(|_| rnd(scale, round)).collect();
            check(&a0, &a1);
            let a0: Vec<f32> = a0.iter().map(|&v| v as f32).collect();
            let a1: Vec<f32> = a1.iter().map(|&v| v as f32).collect();
            check(&a0, &a1);
        }
    }

    #[test]
    fn skipped_rows_keep_their_input_indices() {
        let a0 = [1.0, f64::NAN, 2.0, 1.25];
        let a1 = [f64::INFINITY, 2.0, 1.0];
        check(&a0, &a1);
        let p0 = Points::new(&a0, 4, 1, false);
        let p1 = Points::new(&a1, 3, 1, false);
        let m = sweep(&p0, &p1, Metric::Euclidean, 0.3, 1);
        assert_eq!(m.pairs, vec![0, 2, 2, 1, 3, 2]);
        assert_eq!(m.dists, vec![0.0, 0.0, 0.25]);
    }

    #[test]
    fn empty_inputs() {
        let p0 = Points::<f64>::new(&[], 0, 1, false);
        let p1 = Points::new(&[1.0], 1, 1, false);
        assert!(sweep(&p0, &p1, Metric::Euclidean, 1.0, 1).is_empty());
        assert!(sweep(&p1, &p0, Metric::Euclidean, 1.0, 1).is_empty());
    }
}
