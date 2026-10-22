//! The floating point types the engine works with (`f32` and `f64`).

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

/// A floating point number. Implemented for `f32` and `f64`, so every algorithm is
/// compiled once per type and the input never has to be converted.
pub trait Scalar:
    Copy
    + Debug
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
{
    const ZERO: Self;
    const ONE: Self;
    const INFINITY: Self;
    /// The smallest positive (subnormal) value: the largest absolute rounding error
    /// of one operation whose result is subnormal.
    const DENORM_MIN: Self;
    /// About the square root of the smallest normal value. A sum of squares (or of
    /// `p`-th powers) below this may have lost most of its precision to underflow, so
    /// the distance is recomputed with rescaling.
    const SAFE_MIN: Self;
    /// Relative slack added to the tolerance when pruning, so that rounding can never
    /// make the search skip a pair that the exact distance check would accept.
    const PRUNE_SLACK: Self;
    /// The numpy name of the type.
    const NAME: &'static str;

    fn from_f64(v: f64) -> Self;
    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn powf(self, p: Self) -> Self;
    fn is_finite(self) -> bool;

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if other > self { other } else { self }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        if other < self { other } else { self }
    }
}

macro_rules! impl_scalar {
    ($t:ty, $name:literal, $safe_min:expr, $prune:expr) => {
        impl Scalar for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const INFINITY: Self = <$t>::INFINITY;
            const DENORM_MIN: Self = <$t>::from_bits(1);
            const SAFE_MIN: Self = $safe_min;
            const PRUNE_SLACK: Self = $prune;
            const NAME: &'static str = $name;

            #[inline(always)]
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            #[inline(always)]
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            #[inline(always)]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline(always)]
            fn powf(self, p: Self) -> Self {
                <$t>::powf(self, p)
            }
            #[inline(always)]
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
        }
    };
}

impl_scalar!(f32, "float32", 1e-19, 1e-5);
impl_scalar!(f64, "float64", 1e-154, 1e-12);
