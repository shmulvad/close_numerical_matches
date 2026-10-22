//! Distance metrics.
//!
//! Every metric works with a *reduced* distance that is cheap to compute and grows
//! with the true distance (for example the squared Euclidean distance). The search
//! prunes with lower and upper bounds of the reduced distance to a bounding box and
//! only computes the true distance for the candidates that survive.
//!
//! The bounds use the same floating point operations, in the same order, as the
//! reduced distance itself. Rounding is monotonic, so a bound computed this way can
//! never cross the reduced distance of a point inside the box. A small slack on the
//! pruning threshold covers the cases where the true distance is not exactly a
//! monotonic function of the reduced one in floating point (`sqrt`, `powf`).
//!
//! Sums of squares (or of `p`-th powers) can underflow to 0 or overflow to infinity
//! although the distance itself is representable. The thresholds account for that,
//! and a reduced distance outside the safe range is never trusted: the distance is
//! then recomputed with rescaling.

use crate::scalar::Scalar;

/// A distance metric, as chosen by the user.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    Euclidean,
    SqEuclidean,
    Manhattan,
    Chebyshev,
    Minkowski(f64),
    Cosine,
}

/// The names accepted by [`Metric::parse`], canonical names first.
pub const METRIC_NAMES: &[&str] = &[
    "euclidean",
    "sqeuclidean",
    "manhattan",
    "chebyshev",
    "minkowski",
    "cosine",
    "norm",
    "l2",
    "cityblock",
    "l1",
    "max",
    "linf",
    "cos",
];

impl Metric {
    /// Parses a metric name. `p` is the Minkowski order, only allowed for `minkowski`
    /// (default 2).
    pub fn parse(name: &str, p: Option<f64>) -> Result<Self, String> {
        let metric = match name {
            "euclidean" | "norm" | "l2" => Self::Euclidean,
            "sqeuclidean" => Self::SqEuclidean,
            "manhattan" | "cityblock" | "l1" => Self::Manhattan,
            "chebyshev" | "max" | "linf" => Self::Chebyshev,
            "cosine" | "cos" => Self::Cosine,
            "minkowski" => {
                let p = p.unwrap_or(2.0);
                if p.is_nan() || p < 1.0 {
                    return Err(format!(
                        "p must be >= 1 for the minkowski distance, got {p}"
                    ));
                }
                return Ok(Self::Minkowski(p));
            }
            _ => {
                return Err(format!(
                    "unknown distance metric {name:?}, expected one of: {}",
                    METRIC_NAMES.join(", ")
                ));
            }
        };
        if let Some(p) = p {
            return Err(format!(
                "p is only used by the minkowski distance, but got p={p} with {name:?}"
            ));
        }
        Ok(metric)
    }

    /// The canonical name of the metric.
    pub fn name(self) -> &'static str {
        match self {
            Self::Euclidean => "euclidean",
            Self::SqEuclidean => "sqeuclidean",
            Self::Manhattan => "manhattan",
            Self::Chebyshev => "chebyshev",
            Self::Minkowski(_) => "minkowski",
            Self::Cosine => "cosine",
        }
    }

    /// The Minkowski order, or `None` for the other metrics.
    pub fn p(self) -> Option<f64> {
        match self {
            Self::Minkowski(p) => Some(p),
            _ => None,
        }
    }

    /// Whether points are normalised to unit length before searching.
    pub fn normalizes(self) -> bool {
        matches!(self, Self::Cosine)
    }

    /// The kernel that computes this metric. Minkowski distances with p = 1, 2 or
    /// infinity use the faster dedicated kernels.
    pub fn kernel(self) -> Kernel {
        match self {
            Self::Euclidean | Self::Minkowski(2.0) => Kernel::Euclidean,
            Self::SqEuclidean => Kernel::SqEuclidean,
            Self::Manhattan | Self::Minkowski(1.0) => Kernel::Manhattan,
            Self::Chebyshev | Self::Minkowski(f64::INFINITY) => Kernel::Chebyshev,
            Self::Minkowski(p) => Kernel::Minkowski(p),
            Self::Cosine => Kernel::Cosine,
        }
    }
}

/// The distance kernel to use. See [`Metric::kernel`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kernel {
    Euclidean,
    SqEuclidean,
    Manhattan,
    Chebyshev,
    Minkowski(f64),
    Cosine,
}

/// Runs `$body` with `$m` bound to the distance kernel for `$kernel` and the
/// tolerance `$tol`, so that the body is compiled once per kernel and its inner loops
/// are fully specialised.
#[macro_export]
macro_rules! with_kernel {
    ($kernel:expr, $tol:expr, $m:ident => $body:expr) => {
        match $kernel {
            $crate::metric::Kernel::Euclidean => {
                let $m = $crate::metric::Euclidean;
                $body
            }
            $crate::metric::Kernel::SqEuclidean => {
                let $m = $crate::metric::SqEuclidean;
                $body
            }
            $crate::metric::Kernel::Manhattan => {
                let $m = $crate::metric::Manhattan;
                $body
            }
            $crate::metric::Kernel::Chebyshev => {
                let $m = $crate::metric::Chebyshev;
                $body
            }
            $crate::metric::Kernel::Minkowski(p) => {
                let $m = $crate::metric::Minkowski::new(p, $tol);
                $body
            }
            $crate::metric::Kernel::Cosine => {
                let $m = $crate::metric::Cosine;
                $body
            }
        }
    };
}

/// Thresholds derived from the tolerance, all in the space of the reduced distance
/// except `tol`.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds<T> {
    /// Skip a box (or a point) if its reduced distance is above this.
    pub prune: T,
    /// Accept a whole box without checking its points if its reduced distance upper
    /// bound is at most this. `-inf` disables the shortcut.
    pub accept: T,
    /// The tolerance for the true distance.
    pub tol: T,
}

/// A distance function in the form the search needs. `q` is always the query point.
pub trait Distance<T: Scalar>: Copy + Send + Sync {
    /// The reduced distance between the query `q` and the point `x`.
    fn reduced(&self, q: &[T], x: &[T]) -> T;
    /// A lower bound of `reduced(q, x)` for any `x` in the box `[lo, hi]`.
    fn box_min(&self, q: &[T], lo: &[T], hi: &[T]) -> T;
    /// An upper bound of `reduced(q, x)` for any `x` in the box `[lo, hi]`.
    fn box_max(&self, q: &[T], lo: &[T], hi: &[T]) -> T;
    /// The true distance, given the reduced distance `r = reduced(q, x)`.
    fn finish(&self, q: &[T], x: &[T], r: T) -> T;
    /// The thresholds for the tolerance `tol` in `d` dimensions.
    fn thresholds(&self, tol: T, d: usize) -> Thresholds<T>;
}

/// The amount by which `q` lies outside `[lo, hi]`, computed as the difference
/// `q - x` would be for the nearest `x`, so it never exceeds `|q - x|`.
#[inline(always)]
fn gap<T: Scalar>(q: T, lo: T, hi: T) -> T {
    if q < lo {
        (q - lo).abs()
    } else if q > hi {
        q - hi
    } else {
        T::ZERO
    }
}

/// The largest `|q - x|` for any `x` in `[lo, hi]`.
#[inline(always)]
fn far<T: Scalar>(q: T, lo: T, hi: T) -> T {
    (q - lo).abs().max((q - hi).abs())
}

#[inline(always)]
fn sum_by<T: Scalar>(n: usize, mut f: impl FnMut(usize) -> T) -> T {
    let mut acc = T::ZERO;
    for k in 0..n {
        acc += f(k);
    }
    acc
}

#[inline(always)]
fn max_by<T: Scalar>(n: usize, mut f: impl FnMut(usize) -> T) -> T {
    let mut acc = T::ZERO;
    for k in 0..n {
        acc = acc.max(f(k));
    }
    acc
}

/// The pruning threshold for the reduced tolerance `v`, which was computed with
/// relative error up to `rel`, for a sum of `d` terms that may each have underflowed.
/// Infinite if `v` is so large that the reduced distances of matching pairs could
/// overflow.
#[inline]
fn prune_limit<T: Scalar>(v: T, rel: T, d: usize) -> T {
    if !(v * T::from_f64(16.0)).is_finite() {
        return T::INFINITY;
    }
    v * (T::ONE + rel) + T::DENORM_MIN * T::from_f64(2.0 * d as f64 + 2.0)
}

/// The threshold for accepting a whole box: `v` shrunk by `rel`, or `-inf` (never
/// accept) if `v` overflowed or is so small that underflow could hide a distance
/// above the tolerance.
#[inline]
fn accept_limit<T: Scalar>(v: T, rel: T) -> T {
    let accept = v * (T::ONE - rel);
    if accept >= T::SAFE_MIN && accept.is_finite() {
        accept
    } else {
        -T::INFINITY
    }
}

/// Whether the reduced distance `r` may have lost its precision to underflow or
/// overflow, so the distance must be recomputed with rescaling.
#[inline(always)]
fn needs_rescale<T: Scalar>(r: T) -> bool {
    !(r >= T::SAFE_MIN && r.is_finite())
}

/// The largest `|q - x|` over all dimensions.
#[inline]
fn max_abs_diff<T: Scalar>(q: &[T], x: &[T]) -> T {
    let x = &x[..q.len()];
    max_by(q.len(), |k| (q[k] - x[k]).abs())
}

/// The Euclidean distance, scaled by the largest difference so that the sum of
/// squares can neither overflow nor underflow.
#[inline(never)]
fn rescaled_euclidean<T: Scalar>(q: &[T], x: &[T]) -> T {
    let m = max_abs_diff(q, x);
    if m == T::ZERO || !m.is_finite() {
        return m;
    }
    let x = &x[..q.len()];
    m * sum_by(q.len(), |k| {
        let s = (q[k] - x[k]) / m;
        s * s
    })
    .sqrt()
}

/// The Minkowski distance of order `p`, scaled like [`rescaled_euclidean`].
#[inline(never)]
fn rescaled_minkowski<T: Scalar>(q: &[T], x: &[T], p: T) -> T {
    let m = max_abs_diff(q, x);
    if m == T::ZERO || !m.is_finite() {
        return m;
    }
    let x = &x[..q.len()];
    m * sum_by(q.len(), |k| ((q[k] - x[k]).abs() / m).powf(p)).powf(T::ONE / p)
}

#[inline(always)]
fn sq_dist<T: Scalar>(q: &[T], x: &[T]) -> T {
    let x = &x[..q.len()];
    sum_by(q.len(), |k| {
        let diff = q[k] - x[k];
        diff * diff
    })
}

#[inline(always)]
fn sq_box_min<T: Scalar>(q: &[T], lo: &[T], hi: &[T]) -> T {
    let (lo, hi) = (&lo[..q.len()], &hi[..q.len()]);
    sum_by(q.len(), |k| {
        let g = gap(q[k], lo[k], hi[k]);
        g * g
    })
}

#[inline(always)]
fn sq_box_max<T: Scalar>(q: &[T], lo: &[T], hi: &[T]) -> T {
    let (lo, hi) = (&lo[..q.len()], &hi[..q.len()]);
    sum_by(q.len(), |k| {
        let f = far(q[k], lo[k], hi[k]);
        f * f
    })
}

/// The Euclidean (L2) distance.
#[derive(Clone, Copy, Debug)]
pub struct Euclidean;

impl<T: Scalar> Distance<T> for Euclidean {
    #[inline(always)]
    fn reduced(&self, q: &[T], x: &[T]) -> T {
        sq_dist(q, x)
    }
    #[inline(always)]
    fn box_min(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        sq_box_min(q, lo, hi)
    }
    #[inline(always)]
    fn box_max(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        sq_box_max(q, lo, hi)
    }
    #[inline(always)]
    fn finish(&self, q: &[T], x: &[T], r: T) -> T {
        if needs_rescale(r) {
            rescaled_euclidean(q, x)
        } else {
            r.sqrt()
        }
    }
    fn thresholds(&self, tol: T, d: usize) -> Thresholds<T> {
        let tol_sq = tol * tol;
        Thresholds {
            prune: prune_limit(tol_sq, T::PRUNE_SLACK, d),
            accept: accept_limit(tol_sq, T::PRUNE_SLACK),
            tol,
        }
    }
}

/// The squared Euclidean distance.
#[derive(Clone, Copy, Debug)]
pub struct SqEuclidean;

impl<T: Scalar> Distance<T> for SqEuclidean {
    #[inline(always)]
    fn reduced(&self, q: &[T], x: &[T]) -> T {
        sq_dist(q, x)
    }
    #[inline(always)]
    fn box_min(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        sq_box_min(q, lo, hi)
    }
    #[inline(always)]
    fn box_max(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        sq_box_max(q, lo, hi)
    }
    #[inline(always)]
    fn finish(&self, _q: &[T], _x: &[T], r: T) -> T {
        r
    }
    fn thresholds(&self, tol: T, _d: usize) -> Thresholds<T> {
        // The reduced distance is the distance, and the bounds are exact.
        Thresholds {
            prune: tol,
            accept: tol,
            tol,
        }
    }
}

/// The Manhattan (L1, city block) distance.
#[derive(Clone, Copy, Debug)]
pub struct Manhattan;

impl<T: Scalar> Distance<T> for Manhattan {
    #[inline(always)]
    fn reduced(&self, q: &[T], x: &[T]) -> T {
        let x = &x[..q.len()];
        sum_by(q.len(), |k| (q[k] - x[k]).abs())
    }
    #[inline(always)]
    fn box_min(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        let (lo, hi) = (&lo[..q.len()], &hi[..q.len()]);
        sum_by(q.len(), |k| gap(q[k], lo[k], hi[k]))
    }
    #[inline(always)]
    fn box_max(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        let (lo, hi) = (&lo[..q.len()], &hi[..q.len()]);
        sum_by(q.len(), |k| far(q[k], lo[k], hi[k]))
    }
    #[inline(always)]
    fn finish(&self, _q: &[T], _x: &[T], r: T) -> T {
        r
    }
    fn thresholds(&self, tol: T, _d: usize) -> Thresholds<T> {
        Thresholds {
            prune: tol,
            accept: tol,
            tol,
        }
    }
}

/// The Chebyshev (L-infinity, maximum) distance.
#[derive(Clone, Copy, Debug)]
pub struct Chebyshev;

impl<T: Scalar> Distance<T> for Chebyshev {
    #[inline(always)]
    fn reduced(&self, q: &[T], x: &[T]) -> T {
        max_abs_diff(q, x)
    }
    #[inline(always)]
    fn box_min(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        let (lo, hi) = (&lo[..q.len()], &hi[..q.len()]);
        max_by(q.len(), |k| gap(q[k], lo[k], hi[k]))
    }
    #[inline(always)]
    fn box_max(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        let (lo, hi) = (&lo[..q.len()], &hi[..q.len()]);
        max_by(q.len(), |k| far(q[k], lo[k], hi[k]))
    }
    #[inline(always)]
    fn finish(&self, _q: &[T], _x: &[T], r: T) -> T {
        r
    }
    fn thresholds(&self, tol: T, _d: usize) -> Thresholds<T> {
        Thresholds {
            prune: tol,
            accept: tol,
            tol,
        }
    }
}

/// The Minkowski distance of order `p` (`p >= 1`).
///
/// The differences are divided by the tolerance before taking the `p`-th power, so
/// the reduced distances of pairs near the tolerance are near 1 and cannot underflow
/// or overflow, even for large `p`.
#[derive(Clone, Copy, Debug)]
pub struct Minkowski<T> {
    p: T,
    scale: T,
}

impl<T: Scalar> Minkowski<T> {
    pub fn new(p: f64, tol: T) -> Self {
        let inv = T::ONE / tol;
        let scale = if tol > T::ZERO && inv.is_finite() && tol.is_finite() {
            inv
        } else {
            T::ONE
        };
        Self {
            p: T::from_f64(p),
            scale,
        }
    }

    #[inline(always)]
    fn term(&self, diff: T) -> T {
        (diff * self.scale).powf(self.p)
    }
}

impl<T: Scalar> Distance<T> for Minkowski<T> {
    #[inline(always)]
    fn reduced(&self, q: &[T], x: &[T]) -> T {
        let x = &x[..q.len()];
        sum_by(q.len(), |k| self.term((q[k] - x[k]).abs()))
    }
    #[inline(always)]
    fn box_min(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        let (lo, hi) = (&lo[..q.len()], &hi[..q.len()]);
        sum_by(q.len(), |k| self.term(gap(q[k], lo[k], hi[k])))
    }
    #[inline(always)]
    fn box_max(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        let (lo, hi) = (&lo[..q.len()], &hi[..q.len()]);
        sum_by(q.len(), |k| self.term(far(q[k], lo[k], hi[k])))
    }
    #[inline(always)]
    fn finish(&self, q: &[T], x: &[T], r: T) -> T {
        if needs_rescale(r) {
            rescaled_minkowski(q, x, self.p)
        } else {
            r.powf(T::ONE / self.p) / self.scale
        }
    }
    fn thresholds(&self, tol: T, d: usize) -> Thresholds<T> {
        let tol_p = self.term(tol);
        // `powf` is not exactly monotonic and the scaled tolerance is only 1 up to
        // rounding (which the power amplifies), so use a larger slack than for L2.
        let rel = T::PRUNE_SLACK * (T::from_f64(64.0) + T::from_f64(4.0) * self.p);
        Thresholds {
            prune: prune_limit(tol_p, rel, d),
            accept: accept_limit(tol_p, rel),
            tol,
        }
    }
}

/// The cosine distance, `1 - cos(angle)`, in `[0, 2]`.
///
/// Points are normalised to unit length up front. For unit vectors `u` and `v`,
/// `1 - u.v = |u - v|^2 / 2`, so the search runs on the squared Euclidean distance.
/// The right-hand side is also the more accurate formula: it has no cancellation for
/// small angles and is exactly 0 for vectors with the same direction.
#[derive(Clone, Copy, Debug)]
pub struct Cosine;

impl<T: Scalar> Distance<T> for Cosine {
    #[inline(always)]
    fn reduced(&self, q: &[T], x: &[T]) -> T {
        sq_dist(q, x)
    }
    #[inline(always)]
    fn box_min(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        sq_box_min(q, lo, hi)
    }
    #[inline(always)]
    fn box_max(&self, q: &[T], lo: &[T], hi: &[T]) -> T {
        sq_box_max(q, lo, hi)
    }
    #[inline(always)]
    fn finish(&self, q: &[T], x: &[T], r: T) -> T {
        let two = T::ONE + T::ONE;
        let r = if needs_rescale(r) {
            let l = rescaled_euclidean(q, x);
            l * l
        } else {
            r
        };
        (r / two).min(two)
    }
    fn thresholds(&self, tol: T, d: usize) -> Thresholds<T> {
        let two_tol = (T::ONE + T::ONE) * tol;
        Thresholds {
            // Opposite unit vectors can be slightly more than 2 apart after rounding,
            // and are clamped to a distance of 2, so prune a little later.
            prune: prune_limit(two_tol, T::PRUNE_SLACK, d),
            // Halving is exact, so `r <= 2 * tol` means `r / 2 <= tol`.
            accept: accept_limit(two_tol, T::ZERO),
            tol,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_names_and_aliases() {
        assert_eq!(Metric::parse("norm", None), Ok(Metric::Euclidean));
        assert_eq!(Metric::parse("l2", None), Ok(Metric::Euclidean));
        assert_eq!(Metric::parse("max", None), Ok(Metric::Chebyshev));
        assert_eq!(Metric::parse("cos", None), Ok(Metric::Cosine));
        assert_eq!(Metric::parse("l1", None), Ok(Metric::Manhattan));
        assert_eq!(Metric::parse("minkowski", None), Ok(Metric::Minkowski(2.0)));
        assert_eq!(
            Metric::parse("minkowski", Some(3.0)),
            Ok(Metric::Minkowski(3.0))
        );
        for name in METRIC_NAMES {
            assert!(Metric::parse(name, None).is_ok(), "{name}");
        }
    }

    #[test]
    fn parse_errors() {
        assert!(Metric::parse("nope", None).is_err());
        assert!(Metric::parse("minkowski", Some(0.5)).is_err());
        assert!(Metric::parse("minkowski", Some(f64::NAN)).is_err());
        assert!(Metric::parse("euclidean", Some(2.0)).is_err());
    }

    #[test]
    fn minkowski_special_cases_use_fast_kernels() {
        assert_eq!(Metric::Minkowski(1.0).kernel(), Kernel::Manhattan);
        assert_eq!(Metric::Minkowski(2.0).kernel(), Kernel::Euclidean);
        assert_eq!(Metric::Minkowski(f64::INFINITY).kernel(), Kernel::Chebyshev);
        assert_eq!(Metric::Minkowski(3.0).kernel(), Kernel::Minkowski(3.0));
    }

    #[test]
    fn distances() {
        let a = [0.0_f64, 0.0];
        let b = [3.0_f64, -4.0];
        let fin = |m: &dyn Fn(&[f64], &[f64]) -> f64| m(&a, &b);
        assert_eq!(
            fin(&|q, x| Euclidean.finish(q, x, Euclidean.reduced(q, x))),
            5.0
        );
        assert_eq!(fin(&|q, x| SqEuclidean.reduced(q, x)), 25.0);
        assert_eq!(fin(&|q, x| Manhattan.reduced(q, x)), 7.0);
        assert_eq!(fin(&|q, x| Chebyshev.reduced(q, x)), 4.0);
        let m3 = Minkowski::new(3.0, 1.0_f64);
        let d3 = m3.finish(&a, &b, m3.reduced(&a, &b));
        assert!((d3 - 91.0_f64.powf(1.0 / 3.0)).abs() < 1e-12);
        let u = [1.0_f64, 0.0];
        let v = [0.0_f64, 1.0];
        assert_eq!(Cosine.finish(&u, &v, Cosine.reduced(&u, &v)), 1.0);
        assert_eq!(Cosine.finish(&u, &u, Cosine.reduced(&u, &u)), 0.0);
    }

    #[test]
    fn box_bounds_bracket_the_distance() {
        let lo = [1.0_f64, -2.0, 0.5];
        let hi = [2.0_f64, 3.0, 0.5];
        let q = [-1.0_f64, 4.0, 0.25];
        let inside = [[1.0, -2.0, 0.5], [2.0, 3.0, 0.5], [1.5, 0.0, 0.5]];
        fn check<M: Distance<f64>>(m: M, q: &[f64], lo: &[f64], hi: &[f64], xs: &[[f64; 3]]) {
            let (bmin, bmax) = (m.box_min(q, lo, hi), m.box_max(q, lo, hi));
            for x in xs {
                let r = m.reduced(q, x);
                assert!(bmin <= r && r <= bmax, "{bmin} <= {r} <= {bmax}");
            }
        }
        check(Euclidean, &q, &lo, &hi, &inside);
        check(Manhattan, &q, &lo, &hi, &inside);
        check(Chebyshev, &q, &lo, &hi, &inside);
        check(Minkowski::new(1.5, 0.7_f64), &q, &lo, &hi, &inside);
    }
}
