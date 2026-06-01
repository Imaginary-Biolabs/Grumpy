//! Grid pooling (voxel occupancy counting) for point clouds.

use super::points::{load_point_cloud, point_start, PointCloudBatch};
use crate::dtype::DType;
use crate::error::{arg_invalid, internal};
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub struct GridSpec {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub origin: [f64; 3],
    pub voxel_size: [f64; 3],
}

impl GridSpec {
    pub fn n_cells(&self) -> usize {
        self.nx.saturating_mul(self.ny).saturating_mul(self.nz)
    }

    pub fn voxel_index(&self, x: f64, y: f64, z: f64) -> Option<usize> {
        let ix = ((x - self.origin[0]) / self.voxel_size[0]).floor() as i64;
        let iy = ((y - self.origin[1]) / self.voxel_size[1]).floor() as i64;
        let iz = ((z - self.origin[2]) / self.voxel_size[2]).floor() as i64;
        if ix < 0
            || iy < 0
            || iz < 0
            || ix >= self.nx as i64
            || iy >= self.ny as i64
            || iz >= self.nz as i64
        {
            return None;
        }
        Some(ix as usize + self.nx * (iy as usize + self.ny * iz as usize))
    }
}

pub fn grid_pool(x: &GrumpyArray, spec: GridSpec, dim: isize) -> PyResult<GrumpyArray> {
    if x.dtype != DType::Float64 {
        return Err(arg_invalid(
            "dtype",
            "grid_pool requires float64 coordinates",
            "pass gr.array(..., dtype=gr.float64).",
        ));
    }
    if spec.nx == 0 || spec.ny == 0 || spec.nz == 0 {
        return Err(arg_invalid(
            "grid_size",
            "each grid dimension must be >= 1",
            "pass grid_size=(nx, ny, nz) with positive sizes.",
        ));
    }
    let batch = load_point_cloud(x, dim)?;
    if batch.coord_dim != 3 {
        return Err(arg_invalid(
            "coords",
            format!("grid_pool requires 3D coordinates, got dim={}", batch.coord_dim),
            "use xyz coordinates with 3 components per point.",
        ));
    }
    cpu_grid_pool(&batch, &spec)
}

fn cpu_grid_pool(batch: &PointCloudBatch, spec: &GridSpec) -> PyResult<GrumpyArray> {
    let n_cells = spec.n_cells();
    let mut flat = vec![0.0f64; batch.n_groups.saturating_mul(n_cells)];
    for g in 0..batch.n_groups {
        let qps = batch.group_offsets[batch.group_base + g] as usize;
        let qpe = batch.group_offsets[batch.group_base + g + 1] as usize;
        let base = g * n_cells;
        let grid = &mut flat[base..base + n_cells];
        for qi in qps..qpe {
            let start = point_start(&batch.point_offsets, qi);
            let x = batch.coords[start];
            let y = batch.coords[start + 1];
            let z = batch.coords[start + 2];
            if let Some(vi) = spec.voxel_index(x, y, z) {
                grid[vi] += 1.0;
            }
        }
    }
    build_grid_grouped(batch, &flat)
}

fn build_grid_grouped(batch: &PointCloudBatch, flat: &[f64]) -> PyResult<GrumpyArray> {
    let n_cells = flat.len() / batch.n_groups.max(1);
    let mut out_group_offsets: Vec<i64> = Vec::with_capacity(batch.n_groups + 1);
    out_group_offsets.push(0);
    for g in 0..batch.n_groups {
        out_group_offsets.push((g as i64 + 1) * n_cells as i64);
    }
    let expected = batch.n_groups * n_cells;
    if flat.len() != expected {
        return Err(internal(
            "grid_pool",
            format!("expected {expected} voxel cells, got {}", flat.len()),
        ));
    }
    let mut leaf = Leaf::new(DType::Float64);
    leaf.len = flat.len();
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; leaf.len.max(1)]);
    leaf.buffer = LeafBuffer::F64(Arc::new(flat.to_vec()));
    let layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(out_group_offsets),
        content: Box::new(Layout::Leaf(leaf)),
    });
    Ok(GrumpyArray {
        dtype: DType::Float64,
        layout,
    })
}
