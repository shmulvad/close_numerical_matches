//! The Python bindings. The Python wrapper in `close_numerical_matches/` turns the
//! user's input into C-contiguous 2-D float32/float64 arrays before calling these.

use numpy::ndarray::Array2;
use numpy::{Element, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::metric::Metric;
use crate::points::Points;
use crate::scalar::Scalar;
use crate::search::{self, Index, Matches};

fn parse_metric(dist: &str, p: Option<f64>) -> PyResult<Metric> {
    Metric::parse(dist, p).map_err(PyValueError::new_err)
}

fn check_tol(tol: f64) -> PyResult<()> {
    if tol.is_nan() || tol < 0.0 {
        return Err(PyValueError::new_err(format!(
            "tol must be a non-negative number, got {tol}"
        )));
    }
    Ok(())
}

fn threads(n_threads: Option<i64>) -> PyResult<usize> {
    match n_threads {
        None => Ok(search::resolve_threads(0)),
        Some(n) if n < 0 => Err(PyValueError::new_err(format!(
            "n_threads must be a positive integer, 0 or None, got {n}"
        ))),
        Some(n) => Ok(search::resolve_threads(n as usize)),
    }
}

/// The values of a 2-D array in row-major order, without copying when possible.
fn row_major<'a, T: Element + Copy>(arr: &'a PyReadonlyArray2<'_, T>) -> std::borrow::Cow<'a, [T]> {
    match arr.as_slice() {
        Ok(slice) => std::borrow::Cow::Borrowed(slice),
        Err(_) => std::borrow::Cow::Owned(arr.as_array().iter().copied().collect()),
    }
}

/// Converts matches to a `(k, 2)` int64 array, plus a `(k,)` distance array.
fn to_python<'py, T: Scalar + Element>(
    py: Python<'py>,
    matches: Matches<T>,
    want_dist: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let k = matches.len();
    let pairs = Array2::from_shape_vec((k, 2), matches.pairs)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let pairs = PyArray2::from_owned_array(py, pairs).into_any();
    if want_dist {
        let dists = PyArray1::from_vec(py, matches.dists).into_any();
        Ok(PyTuple::new(py, [pairs, dists])?.into_any())
    } else {
        Ok(pairs)
    }
}

enum Pair<'py> {
    F32(PyReadonlyArray2<'py, f32>, PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>, PyReadonlyArray2<'py, f64>),
}

fn extract_pair<'py>(arr0: &Bound<'py, PyAny>, arr1: &Bound<'py, PyAny>) -> PyResult<Pair<'py>> {
    if let (Ok(a), Ok(b)) = (arr0.extract(), arr1.extract()) {
        return Ok(Pair::F64(a, b));
    }
    if let (Ok(a), Ok(b)) = (arr0.extract(), arr1.extract()) {
        return Ok(Pair::F32(a, b));
    }
    Err(PyTypeError::new_err(
        "arr0 and arr1 must both be 2-D numpy arrays of float32 or both of float64",
    ))
}

/// Copies a 2-D array into [`Points`]. This must happen while attached to Python:
/// after detaching, other Python threads may write to the array.
fn to_points<T: Scalar + Element>(arr: &PyReadonlyArray2<'_, T>, metric: Metric) -> Points<T> {
    let [n, d] = arr.shape2();
    let values = row_major(arr);
    Points::new(&values, n, d, metric.normalizes())
}

#[allow(clippy::too_many_arguments)]
fn run_find<'py, T: Scalar + Element>(
    py: Python<'py>,
    a0: &PyReadonlyArray2<'py, T>,
    a1: &PyReadonlyArray2<'py, T>,
    metric: Metric,
    tol: f64,
    n_threads: usize,
    want_dist: bool,
    naive: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let ([_, d0], [_, d1]) = (a0.shape2(), a1.shape2());
    if d0 != d1 {
        return Err(PyValueError::new_err(format!(
            "arrays must have the same number of columns, got {d0} and {d1}"
        )));
    }
    let (p0, p1) = (to_points(a0, metric), to_points(a1, metric));
    let tol = T::from_f64(tol);
    let matches =
        py.detach(move || search::find_matches(p0, p1, metric, tol, n_threads, want_dist, naive));
    to_python(py, matches, want_dist)
}

trait Shape2 {
    fn shape2(&self) -> [usize; 2];
}

impl<T: Element> Shape2 for PyReadonlyArray2<'_, T> {
    fn shape2(&self) -> [usize; 2] {
        let s = self.shape();
        [s[0], s[1]]
    }
}

/// Low-level `find_matches`; see the Python wrapper for the documentation.
#[pyfunction]
#[pyo3(signature = (arr0, arr1, dist, tol, p, n_threads, return_distances, naive))]
#[allow(clippy::too_many_arguments)]
pub fn find_matches<'py>(
    py: Python<'py>,
    arr0: &Bound<'py, PyAny>,
    arr1: &Bound<'py, PyAny>,
    dist: &str,
    tol: f64,
    p: Option<f64>,
    n_threads: Option<i64>,
    return_distances: bool,
    naive: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let metric = parse_metric(dist, p)?;
    check_tol(tol)?;
    let n_threads = threads(n_threads)?;
    match extract_pair(arr0, arr1)? {
        Pair::F64(a, b) => run_find(py, &a, &b, metric, tol, n_threads, return_distances, naive),
        Pair::F32(a, b) => run_find(py, &a, &b, metric, tol, n_threads, return_distances, naive),
    }
}

enum AnyIndex {
    F32(Index<f32>),
    F64(Index<f64>),
}

/// Low-level `MatchIndex`; see the Python wrapper for the documentation.
#[pyclass(frozen, module = "close_numerical_matches._core", name = "MatchIndex")]
pub struct PyMatchIndex {
    index: AnyIndex,
}

fn build_index<T: Scalar + Element>(
    py: Python<'_>,
    arr: &PyReadonlyArray2<'_, T>,
    metric: Metric,
    n_threads: usize,
) -> Index<T> {
    let points = to_points(arr, metric);
    py.detach(move || Index::new(points, metric, n_threads))
}

fn query_index<'py, T: Scalar + Element>(
    py: Python<'py>,
    index: &Index<T>,
    other: &Bound<'py, PyAny>,
    tol: f64,
    n_threads: usize,
    want_dist: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let other: PyReadonlyArray2<'py, T> = other.extract().map_err(|_| {
        PyTypeError::new_err(format!(
            "the query must be a 2-D numpy array of {}",
            T::NAME
        ))
    })?;
    let [_, d] = other.shape2();
    if d != index.dims() {
        return Err(PyValueError::new_err(format!(
            "the query has {d} columns, but the index has {}",
            index.dims()
        )));
    }
    let queries = to_points(&other, index.metric());
    let tol = T::from_f64(tol);
    let matches = py.detach(|| index.query(&queries, tol, n_threads, want_dist));
    to_python(py, matches, want_dist)
}

#[pymethods]
impl PyMatchIndex {
    #[new]
    #[pyo3(signature = (arr, dist, p, n_threads))]
    fn new(
        py: Python<'_>,
        arr: &Bound<'_, PyAny>,
        dist: &str,
        p: Option<f64>,
        n_threads: Option<i64>,
    ) -> PyResult<Self> {
        let metric = parse_metric(dist, p)?;
        let n_threads = threads(n_threads)?;
        let index = if let Ok(a) = arr.extract::<PyReadonlyArray2<'_, f64>>() {
            AnyIndex::F64(build_index(py, &a, metric, n_threads))
        } else if let Ok(a) = arr.extract::<PyReadonlyArray2<'_, f32>>() {
            AnyIndex::F32(build_index(py, &a, metric, n_threads))
        } else {
            return Err(PyTypeError::new_err(
                "arr must be a 2-D numpy array of float32 or float64",
            ));
        };
        Ok(Self { index })
    }

    #[pyo3(signature = (other, tol, n_threads, return_distances))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        other: &Bound<'py, PyAny>,
        tol: f64,
        n_threads: Option<i64>,
        return_distances: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        check_tol(tol)?;
        let n_threads = threads(n_threads)?;
        match &self.index {
            AnyIndex::F64(i) => query_index(py, i, other, tol, n_threads, return_distances),
            AnyIndex::F32(i) => query_index(py, i, other, tol, n_threads, return_distances),
        }
    }

    #[getter]
    fn n_points(&self) -> usize {
        match &self.index {
            AnyIndex::F64(i) => i.n_input(),
            AnyIndex::F32(i) => i.n_input(),
        }
    }

    #[getter]
    fn n_dims(&self) -> usize {
        match &self.index {
            AnyIndex::F64(i) => i.dims(),
            AnyIndex::F32(i) => i.dims(),
        }
    }

    #[getter]
    fn dist(&self) -> &'static str {
        self.metric().name()
    }

    #[getter]
    fn p(&self) -> Option<f64> {
        self.metric().p()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        match &self.index {
            AnyIndex::F64(_) => f64::NAME,
            AnyIndex::F32(_) => f32::NAME,
        }
    }
}

impl PyMatchIndex {
    fn metric(&self) -> Metric {
        match &self.index {
            AnyIndex::F64(i) => i.metric(),
            AnyIndex::F32(i) => i.metric(),
        }
    }
}
