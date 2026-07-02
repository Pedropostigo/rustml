use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use rayon::prelude::*;

// Rough, unbenchmarked cutoffs below which thread-dispatch overhead outweighs
// the benefit of spreading the work across rayon's thread pool. Tune once
// there's real workload data to profile against.
const PARALLEL_TRANSPOSE_ELEMENTS: usize = 100_000;
const PARALLEL_MATMUL_FLOPS: usize = 100_000;

#[pyclass]
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn index_of(&self, row: usize, col: usize) -> PyResult<usize> {
        if row >= self.rows || col >= self.cols {
            return Err(PyIndexError::new_err(format!(
                "index ({row}, {col}) out of bounds for a {}x{} matrix",
                self.rows, self.cols
            )));
        }
        Ok(row * self.cols + col)
    }
}

/// Accumulates row `i` of `a @ b` into `out_row`, where `a` is `_ x inner`
/// and `b` is `inner x cols`. Shared by the sequential and parallel matmul
/// paths so they can't drift apart.
fn matmul_row(i: usize, out_row: &mut [f64], a: &[f64], b: &[f64], inner: usize, cols: usize) {
    for k in 0..inner {
        let a_ik = a[i * inner + k];
        if a_ik == 0.0 {
            continue;
        }
        for (j, out) in out_row.iter_mut().enumerate() {
            *out += a_ik * b[k * cols + j];
        }
    }
}

#[pymethods]
impl Matrix {
    #[new]
    fn new(rows: usize, cols: usize, data: Vec<f64>) -> PyResult<Self> {
        if data.len() != rows * cols {
            return Err(PyValueError::new_err(format!(
                "data has {} elements, expected {rows} * {cols} = {}",
                data.len(),
                rows * cols
            )));
        }
        Ok(Matrix { rows, cols, data })
    }

    #[staticmethod]
    fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn __getitem__(&self, idx: (usize, usize)) -> PyResult<f64> {
        let i = self.index_of(idx.0, idx.1)?;
        Ok(self.data[i])
    }

    fn __setitem__(&mut self, idx: (usize, usize), value: f64) -> PyResult<()> {
        let i = self.index_of(idx.0, idx.1)?;
        self.data[i] = value;
        Ok(())
    }

    fn transpose(&self, py: Python<'_>) -> Matrix {
        let mut data = vec![0.0; self.data.len()];
        if self.data.len() < PARALLEL_TRANSPOSE_ELEMENTS {
            for r in 0..self.rows {
                for c in 0..self.cols {
                    data[c * self.rows + r] = self.data[r * self.cols + c];
                }
            }
        } else {
            // Chunk by output row (= input column) so each chunk is a
            // contiguous, disjoint slice rayon can hand to a worker thread
            // without unsafe code.
            py.allow_threads(|| {
                data.par_chunks_mut(self.rows)
                    .enumerate()
                    .for_each(|(c, out_row)| {
                        for (r, out) in out_row.iter_mut().enumerate() {
                            *out = self.data[r * self.cols + c];
                        }
                    });
            });
        }
        Matrix {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    fn __add__(&self, other: &Matrix) -> PyResult<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyValueError::new_err(format!(
                "cannot add a {}x{} matrix to a {}x{} matrix",
                other.rows, other.cols, self.rows, self.cols
            )));
        }
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Ok(Matrix {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    fn __mul__(&self, scalar: f64) -> Matrix {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|a| a * scalar).collect(),
        }
    }

    fn __matmul__(&self, py: Python<'_>, other: &Matrix) -> PyResult<Matrix> {
        if self.cols != other.rows {
            return Err(PyValueError::new_err(format!(
                "cannot multiply a {}x{} matrix by a {}x{} matrix",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let (rows, inner, cols) = (self.rows, self.cols, other.cols);
        let mut data = vec![0.0; rows * cols];
        if rows * inner * cols < PARALLEL_MATMUL_FLOPS {
            for (i, row) in data.chunks_mut(cols).enumerate() {
                matmul_row(i, row, &self.data, &other.data, inner, cols);
            }
        } else {
            // Release the GIL: this is pure Rust computation with no Python
            // API calls, and rayon spins up its own thread pool to do the work.
            py.allow_threads(|| {
                data.par_chunks_mut(cols)
                    .enumerate()
                    .for_each(|(i, row)| matmul_row(i, row, &self.data, &other.data, inner, cols));
            });
        }
        Ok(Matrix { rows, cols, data })
    }

    fn __repr__(&self) -> String {
        format!("Matrix(rows={}, cols={}, data={:?})", self.rows, self.cols, self.data)
    }

    fn __eq__(&self, other: &Matrix) -> bool {
        self == other
    }
}
