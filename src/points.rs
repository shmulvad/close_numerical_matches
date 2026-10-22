//! A set of points copied out of the caller's array.

use crate::scalar::Scalar;

/// Points in `d` dimensions, stored row-major and contiguous. Rows that can never
/// match (any NaN or infinite coordinate, or a zero vector when normalising) are
/// left out; `orig` maps each stored row back to its row in the input (`None` when
/// no row was left out).
#[derive(Clone, Debug)]
pub struct Points<T> {
    data: Vec<T>,
    orig: Option<Vec<usize>>,
    len: usize,
    d: usize,
    n_input: usize,
}

impl<T: Scalar> Points<T> {
    /// Copies the `n x d` row-major array `values` (`values.len() == n * d`).
    /// With `normalize`, every row is scaled to unit Euclidean length.
    pub fn new(values: &[T], n: usize, d: usize, normalize: bool) -> Self {
        assert_eq!(values.len(), n * d, "values must have n * d elements");
        if !normalize && values.iter().all(|v| v.is_finite()) {
            // The common case: nothing to skip, so copy in bulk.
            return Self {
                data: values.to_vec(),
                orig: None,
                len: n,
                d,
                n_input: n,
            };
        }
        let mut data = Vec::with_capacity(values.len());
        // Only allocated once a row is left out.
        let mut orig: Option<Vec<usize>> = None;
        let mut len = 0;
        for i in 0..n {
            let row = &values[i * d..(i + 1) * d];
            let skip = |orig: &mut Option<Vec<usize>>| {
                orig.get_or_insert_with(|| (0..len).collect());
            };
            if !row.iter().all(|v| v.is_finite()) {
                skip(&mut orig);
                continue;
            }
            if normalize {
                // Scale by the largest coordinate first so the norm cannot overflow.
                let scale = row.iter().fold(T::ZERO, |m, v| m.max(v.abs()));
                if scale == T::ZERO {
                    skip(&mut orig);
                    continue;
                }
                let norm = row
                    .iter()
                    .fold(T::ZERO, |acc, &v| {
                        let s = v / scale;
                        acc + s * s
                    })
                    .sqrt();
                data.extend(row.iter().map(|&v| (v / scale) / norm));
            } else {
                data.extend_from_slice(row);
            }
            if let Some(orig) = &mut orig {
                orig.push(i);
            }
            len += 1;
        }
        Self {
            data,
            orig,
            len,
            d,
            n_input: n,
        }
    }

    /// The coordinates (row-major) and the input row of every stored point.
    pub fn into_parts(self) -> (Vec<T>, Vec<usize>) {
        let orig = self.orig.unwrap_or_else(|| (0..self.len).collect());
        (self.data, orig)
    }

    /// The number of stored (matchable) points.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of rows in the input, including the ones that were left out.
    #[inline]
    pub fn n_input(&self) -> usize {
        self.n_input
    }

    #[inline]
    pub fn dims(&self) -> usize {
        self.d
    }

    /// The coordinates of stored point `i`.
    #[inline(always)]
    pub fn row(&self, i: usize) -> &[T] {
        &self.data[i * self.d..(i + 1) * self.d]
    }

    /// The input row of stored point `i`.
    #[inline(always)]
    pub fn orig(&self, i: usize) -> usize {
        match &self.orig {
            Some(orig) => orig[i],
            None => i,
        }
    }

    /// The input rows of all stored points, in increasing order.
    #[cfg(test)]
    fn orig_indices(&self) -> Vec<usize> {
        self.clone().into_parts().1
    }

    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_non_finite_rows() {
        let values = [1.0, 2.0, f64::NAN, 0.0, 3.0, f64::INFINITY, 5.0, 6.0];
        let p = Points::new(&values, 4, 2, false);
        assert_eq!(p.len(), 2);
        assert_eq!(p.n_input(), 4);
        assert_eq!(p.orig_indices(), [0, 3]);
        assert_eq!(p.row(1), &[5.0, 6.0]);
    }

    #[test]
    fn normalizes_and_drops_zero_vectors() {
        let values = [3.0_f64, 4.0, 0.0, 0.0, 1e300, 1e300];
        let p = Points::new(&values, 3, 2, true);
        assert_eq!(p.orig_indices(), [0, 2]);
        assert!((p.row(0)[0] - 0.6).abs() < 1e-15);
        assert!((p.row(0)[1] - 0.8).abs() < 1e-15);
        let h = std::f64::consts::FRAC_1_SQRT_2;
        assert!((p.row(1)[0] - h).abs() < 1e-15);
    }

    #[test]
    fn zero_dimensions() {
        let p = Points::<f64>::new(&[], 3, 0, false);
        assert_eq!(p.len(), 3);
        assert_eq!(p.row(2), &[] as &[f64]);
        // A zero-dimensional vector has no direction, so never matches by cosine.
        assert!(Points::<f64>::new(&[], 3, 0, true).is_empty());
    }
}
