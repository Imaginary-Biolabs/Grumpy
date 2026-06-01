use crate::dtype::{DType, PyDType};
use crate::py_api::types::PyGrumpyArray;

use crate::layout::{build_array, GrumpyArray, Layout};
use crate::py_api::convert::{max_list_depth, normalize_dim, parse_dims, sizes_to_list_any, sizes_to_vec_usize_fast, flatten_collect, unflatten_rec};
use crate::py_api::indexing::find_leaf_fast;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pymethods]
impl PyGrumpyArray {
    #[pyo3(signature = (dim=None, but=None))]
    fn flatten(
        &self,
        py: Python<'_>,
        dim: Option<Bound<'_, PyAny>>,
        but: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        // Fast path: pure list-chain + default flatten (all axes) => just return the leaf view.
        // This matches the Awkward-style "avoid materialization": we reuse the contiguous leaf buffers
        // (Arc + copy-on-write) and only change metadata.
        if dim.is_none() && but.is_none() && self.inner.is_pure_list_chain() {
            if let Ok(leaf) = find_leaf_fast(&self.inner.layout) {
                return Ok(Self {
                    inner: GrumpyArray {
                        dtype: self.inner.dtype,
                        layout: Layout::Leaf(leaf.clone()),
                    },
                });
            }
        }

        let data = self.inner.to_py_list(py)?;
        let data_b = data.bind(py);
        let max_depth = max_list_depth(py, &data_b)?;

        let dim_is_none = dim.is_none();
        let mut dims = if let Some(ref d) = dim {
            parse_dims(py, d, max_depth)?
        } else {
            Vec::new()
        };

        if let Some(b) = but {
            let excluded = parse_dims(py, &b, max_depth)?;
            let mut set = std::collections::BTreeSet::new();
            for x in 0..max_depth {
                set.insert(x as isize);
            }
            for e in excluded {
                set.remove(&e);
            }
            dims = set.into_iter().collect();
        } else if dim_is_none {
            // default: flatten all axes
            dims = (0..max_depth).map(|x| x as isize).collect();
        }

        let mut axes = std::collections::BTreeSet::<usize>::new();
        for d in dims {
            axes.insert(d as usize);
        }
        let flat_elems = flatten_collect(py, &data_b, 0, &axes)?;
        let cur = if axes.contains(&0) {
            let out_list = pyo3::types::PyList::empty_bound(py);
            for e in flat_elems {
                out_list.append(e)?;
            }
            out_list.into_py(py)
        } else {
            if flat_elems.len() != 1 {
                return Err(PyValueError::new_err(
                    "Internal error: flatten root expected a single list result.",
                ));
            }
            flat_elems[0].clone_ref(py)
        };

        let out = build_array(py, &cur.bind(py), self.inner.dtype)?;
        Ok(Self { inner: out })
    }

    #[pyo3(signature = (sizes, dim=0))]
    fn unflatten(&self, py: Python<'_>, sizes: Bound<'_, PyAny>, dim: isize) -> PyResult<Self> {
        // Fast path: unflatten a 1D leaf along dim=0 using a 1D sizes vector (Python list or GrumpyArray[int64]).
        // Produces a ListOffset view sharing the leaf buffer (Arc + copy-on-write).
        if self.inner.is_pure_list_chain() && dim == 0 {
            if let Layout::Leaf(leaf) = &self.inner.layout {
                if let Some(szs) = sizes_to_vec_usize_fast(py, &sizes)? {
                    let mut offsets: Vec<i64> = Vec::with_capacity(szs.len() + 1);
                    offsets.push(0);
                    let mut acc: i64 = 0;
                    for s in szs {
                        acc += s as i64;
                        offsets.push(acc);
                    }
                    if acc as usize != leaf.len {
                        return Err(PyValueError::new_err(
                            "unflatten sizes must sum to the number of elements.",
                        ));
                    }
                    return Ok(Self {
                        inner: GrumpyArray {
                            dtype: self.inner.dtype,
                            layout: Layout::ListOffset(crate::layout::ListOffset {
                                offsets: std::sync::Arc::new(offsets),
                                content: Box::new(Layout::Leaf(leaf.clone())),
                            }),
                        },
                    });
                }
            }
        }

        let data = self.inner.to_py_list(py)?;
        let data_b = data.bind(py);
        let max_depth = max_list_depth(py, &data_b)?;
        let dim_u = normalize_dim(dim, max_depth as isize)?;

        let sizes_obj = sizes_to_list_any(py, &sizes)?;
        let out_py = unflatten_rec(py, &data_b, &sizes_obj.bind(py), dim_u as usize)?;
        let out = build_array(py, &out_py.bind(py), self.inner.dtype)?;
        Ok(Self { inner: out })
    }
}
