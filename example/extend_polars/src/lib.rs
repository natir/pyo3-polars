mod parallel_jaccard_mod;

use polars::prelude::*;
use polars_lazy::frame::IntoLazy;
use polars_lazy::prelude::LazyFrame;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyLazyFrame};

#[pyfunction]
fn parallel_jaccard(pydf: PyDataFrame, col_a: &str, col_b: &str) -> PyResult<PyDataFrame> {
    let df: DataFrame = pydf.into();
    let df = parallel_jaccard_mod::parallel_jaccard(df, col_a, col_b).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(df))
}

#[pyfunction]
fn lazy_parallel_jaccard(pydf: PyLazyFrame, col_a: &str, col_b: &str) -> PyResult<PyLazyFrame> {
    let df: LazyFrame = pydf.into();
    let df = parallel_jaccard_mod::parallel_jaccard(df.collect().unwrap(), col_a, col_b)
        .map_err(PyPolarsErr::from)?;
    Ok(PyLazyFrame(df.lazy()))
}

#[pyfunction]
fn lazy_sum(pydf: PyLazyFrame) -> PyResult<PyLazyFrame> {
    let mut lf: LazyFrame = pydf.into();

    lf = parallel_jaccard_mod::polars_sum(lf).map_err(PyPolarsErr::from)?;

    Ok(PyLazyFrame(lf))
}

/// A Python module implemented in Rust.
#[pymodule]
fn extend_polars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(lazy_parallel_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(lazy_sum, m)?)?;
    Ok(())
}
