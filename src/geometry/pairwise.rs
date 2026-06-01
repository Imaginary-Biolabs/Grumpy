//! All-pairs Euclidean distances within point clouds.

use super::points::{dist2, load_point_cloud, point_start, PointCloudBatch};
use crate::dtype::DType;
use crate::error::{arg_invalid, internal};
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

pub fn pairwise_distances(x: &GrumpyArray, dim: isize) -> PyResult<GrumpyArray> {
    if x.dtype != DType::Float64 {
        return Err(arg_invalid(
            "dtype",
            "pairwise_distances requires float64 coordinates",
            "pass gr.array(..., dtype=gr.float64).",
        ));
    }
    let batch = load_point_cloud(x, dim)?;
    if batch.coord_dim == 0 {
        return build_pairwise_grouped(&[], &[], batch.n_groups);
    }
    cpu_pairwise(&batch)
}

fn cpu_pairwise(batch: &PointCloudBatch) -> PyResult<GrumpyArray> {
    let mut group_counts = Vec::with_capacity(batch.n_groups);
    let mut flat = Vec::new();
    for g in 0..batch.n_groups {
        let qps = batch.group_offsets[batch.group_base + g] as usize;
        let qpe = batch.group_offsets[batch.group_base + g + 1] as usize;
        let n = qpe - qps;
        group_counts.push(n);
        let rows: Vec<Vec<f64>> = (qps..qpe)
            .into_par_iter()
            .map(|qi| {
                let qstart = point_start(&batch.point_offsets, qi);
                let q = &batch.coords[qstart..qstart + batch.coord_dim];
                let mut row = Vec::with_capacity(n);
                for j in qps..qpe {
                    let jstart = point_start(&batch.point_offsets, j);
                    let dist = dist2(q, &batch.coords[jstart..jstart + batch.coord_dim]).sqrt();
                    row.push(dist);
                }
                row
            })
            .collect();
        for row in rows {
            flat.extend(row);
        }
    }
    build_pairwise_grouped(&group_counts, &flat, batch.n_groups)
}

fn build_pairwise_grouped(
    group_counts: &[usize],
    flat: &[f64],
    n_groups: usize,
) -> PyResult<GrumpyArray> {
    let mut out_group_offsets: Vec<i64> = Vec::with_capacity(n_groups + 1);
    out_group_offsets.push(0);
    let mut out_point_offsets: Vec<i64> = Vec::new();
    out_point_offsets.push(0);
    let mut pos = 0usize;
    for &n in group_counts {
        for _ in 0..n {
            let end = pos + n;
            if end > flat.len() {
                return Err(internal(
                    "pairwise_distances",
                    "distance buffer shorter than expected",
                ));
            }
            pos = end;
            out_point_offsets.push(out_point_offsets.last().copied().unwrap() + n as i64);
        }
        out_group_offsets.push(out_group_offsets.last().copied().unwrap() + n as i64);
    }
    if pos != flat.len() && !flat.is_empty() {
        return Err(internal(
            "pairwise_distances",
            "distance buffer longer than expected",
        ));
    }
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = flat.len();
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; leaf.len.max(1)]);
    leaf.buffer = LeafBuffer::F64(Arc::new(flat.to_vec()));
    let layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(out_group_offsets),
        content: Box::new(Layout::ListOffset(ListOffset {
            offsets: Arc::new(out_point_offsets),
            content: Box::new(Layout::Leaf(leaf)),
        })),
    });
    Ok(GrumpyArray {
        dtype: DType::Float64,
        layout,
    })
}
