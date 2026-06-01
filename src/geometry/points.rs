//! Point cloud layout extraction (groups -> points -> xyz leaf).

use crate::dtype::DType;
use crate::error::{dtype_unsupported, layout_unsupported, shape_mismatch};
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use pyo3::prelude::*;
use std::sync::Arc;

pub struct PointCloudBatch {
    pub coords: Vec<f64>,
    pub group_offsets: Arc<Vec<i64>>,
    pub group_base: usize,
    pub point_offsets: Arc<Vec<i64>>,
    pub n_groups: usize,
    pub coord_dim: usize,
}

pub fn load_point_cloud(x: &GrumpyArray, dim: isize) -> PyResult<PointCloudBatch> {
    match dim {
        0 | -2 => load_dim0(x),
        1 | -3 => load_dim1(x),
        _ => Err(crate::error::arg_invalid(
            "dim",
            format!("unsupported dim={dim}"),
            "use dim=0 (single cloud) or dim=1 (grouped clouds).",
        )),
    }
}

fn load_dim0(x: &GrumpyArray) -> PyResult<PointCloudBatch> {
    if matches!(x.layout, Layout::OffsetView(_)) {
        if let Layout::OffsetView(v) = &x.layout {
            let lo = crate::layout::offsetview_to_listoffset(v)?;
            let temp = GrumpyArray {
                dtype: x.dtype,
                layout: Layout::ListOffset(lo),
            };
            return load_dim0(&temp);
        }
    }
    let (off, base, leaf, n, d) = point_list_leaf(&x.layout)?;
    let coords = coords_as_f64(x.dtype, leaf)?;
    let group_offsets = vec![0i64, n as i64];
    Ok(PointCloudBatch {
        coords,
        group_offsets: Arc::new(group_offsets),
        group_base: 0,
        point_offsets: Arc::new(off[base..=base + n].to_vec()),
        n_groups: 1,
        coord_dim: d,
    })
}

fn load_dim1(x: &GrumpyArray) -> PyResult<PointCloudBatch> {
    let (g_off, g_base, p, leaf, ng, d) = grouped_points(&x.layout)?;
    if leaf.has_nulls {
        return Err(crate::error::unsupported(
            "geometry",
            "null coordinates are not supported",
            "filter nulls before calling geometry ops.",
        ));
    }
    let coords = coords_as_f64(x.dtype, leaf)?;
    Ok(PointCloudBatch {
        coords,
        group_offsets: Arc::clone(g_off),
        group_base: g_base,
        point_offsets: Arc::clone(&p.offsets),
        n_groups: ng,
        coord_dim: d,
    })
}

fn point_list_leaf(layout: &Layout) -> PyResult<(Arc<Vec<i64>>, usize, &Leaf, usize, usize)> {
    match layout {
        Layout::ListOffset(lo) => {
            let leaf = match lo.content.as_ref() {
                Layout::Leaf(l) => l,
                _ => {
                    return Err(layout_unsupported(
                        "geometry",
                        "expected point cloud ListOffset -> Leaf",
                    ))
                }
            };
            let n = lo.len();
            if n == 0 {
                return Ok((Arc::clone(&lo.offsets), 0, leaf, 0, 0));
            }
            let d = (lo.offsets[1] - lo.offsets[0]) as usize;
            for i in 0..n {
                if (lo.offsets[i + 1] - lo.offsets[i]) as usize != d {
                    return Err(shape_mismatch(
                        "geometry",
                        "expected fixed coordinate dimension per point",
                        "ensure every point has the same number of coordinates.",
                    ));
                }
            }
            Ok((Arc::clone(&lo.offsets), 0, leaf, n, d))
        }
        Layout::Indexed(ix) => point_list_leaf(ix.content.as_ref()),
        _ => Err(layout_unsupported(
            "geometry",
            "expected point cloud ListOffset -> Leaf",
        )),
    }
}

fn grouped_points<'a>(
    layout: &'a Layout,
) -> PyResult<(&'a Arc<Vec<i64>>, usize, &'a ListOffset, &'a Leaf, usize, usize)> {
    let (g_off, g_base, g_content, ng) = match layout {
        Layout::ListOffset(g) => (&g.offsets, 0usize, g.content.as_ref(), g.len()),
        Layout::OffsetView(v) => (&v.offsets, v.start, v.content.as_ref(), v.len()),
        Layout::Indexed(ix) => return grouped_points(ix.content.as_ref()),
        _ => {
            return Err(layout_unsupported(
                "geometry",
                "expected grouped point cloud: groups->points->coords",
            ))
        }
    };
    let p = match g_content {
        Layout::ListOffset(p) => p,
        _ => {
            return Err(layout_unsupported(
                "geometry",
                "expected grouped point cloud: groups->points->coords",
            ))
        }
    };
    let leaf = match p.content.as_ref() {
        Layout::Leaf(l) => l,
        _ => {
            return Err(layout_unsupported(
                "geometry",
                "expected grouped point cloud: points->leaf coords",
            ))
        }
    };
    if p.len() == 0 {
        return Ok((g_off, g_base, p, leaf, ng, 0));
    }
    let d = (p.offsets[1] - p.offsets[0]) as usize;
    for i in 0..p.len() {
        let len = (p.offsets[i + 1] - p.offsets[i]) as usize;
        if len != d {
            return Err(shape_mismatch(
                "geometry",
                "expected fixed coordinate dimension for each point",
                "ensure every point has the same number of coordinates.",
            ));
        }
    }
    Ok((g_off, g_base, p, leaf, ng, d))
}

pub fn coords_as_f64(dtype: DType, leaf: &Leaf) -> PyResult<Vec<f64>> {
    Ok(match (&leaf.buffer, dtype) {
        (LeafBuffer::F64(v), _) => v.iter().cloned().collect(),
        (LeafBuffer::F32(v), _) => v.iter().map(|&x| x as f64).collect(),
        _ => return Err(dtype_unsupported("geometry coords", dtype)),
    })
}

#[inline]
pub fn dist2(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

pub fn point_start(point_offsets: &[i64], point_index: usize) -> usize {
    point_offsets[point_index] as usize
}

pub fn is_fixed_stride(point_offsets: &[i64], n_points: usize, d: usize) -> bool {
    if n_points == 0 {
        return true;
    }
    let d_i64 = d as i64;
    for i in 0..n_points {
        if point_offsets[i + 1] - point_offsets[i] != d_i64 {
            return false;
        }
    }
    true
}
