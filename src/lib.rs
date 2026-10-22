//! Rust core of the `close_numerical_matches` Python package.
//!
//! Finds all pairs of rows from two arrays that lie within a tolerance of each
//! other under a chosen distance metric, using a KD-tree over the smaller array.

pub mod kdtree;
pub mod metric;
pub mod points;
pub mod scalar;
pub mod search;

mod py;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py::find_matches, m)?)?;
    m.add_class::<py::PyMatchIndex>()?;
    m.add("METRIC_NAMES", metric::METRIC_NAMES.to_vec())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
