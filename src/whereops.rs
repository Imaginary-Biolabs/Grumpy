use crate::dtype::DType;
use crate::error::{dtype_mismatch, dtype_unsupported, layout_unsupported, shape_mismatch};
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::prelude::*;
use std::sync::Arc;

pub fn where_indices(_py: Python<'_>, cond: &GrumpyArray) -> PyResult<GrumpyArray> {
    let leaf = leaf_1d(&cond.layout)?;
    if cond.dtype != DType::Bool {
        return Err(dtype_mismatch(DType::Bool, cond.dtype, "in where(cond)"));
    }
    let v = match &leaf.buffer {
        LeafBuffer::Bool(v) => v.as_slice(),
        _ => unreachable!(),
    };
    let mut idx: Vec<i64> = Vec::new();
    for i in 0..leaf.len {
        if !leaf.validity[i] {
            continue;
        }
        if v[i] != 0 {
            idx.push(i as i64);
        }
    }
    Ok(GrumpyArray { dtype: DType::Int64, layout: Layout::Leaf(new_leaf_i64_from(idx)) })
}

pub fn argwhere(_py: Python<'_>, cond: &GrumpyArray) -> PyResult<GrumpyArray> {
    // 1D only: return list of singleton lists: [[i],[j],...]
    let idx = where_indices(_py, cond)?;
    let leaf = match &idx.layout {
        Layout::Leaf(l) => l,
        _ => unreachable!(),
    };
    let n = leaf.len;
    let offsets: Vec<i64> = (0..=n as i64).collect();
    let out_layout = Layout::ListOffset(ListOffset {
        offsets: Arc::new(offsets),
        content: Box::new(idx.layout.clone()),
    });
    Ok(GrumpyArray { dtype: DType::Int64, layout: out_layout })
}

pub fn where_select(_py: Python<'_>, cond: &GrumpyArray, x: &GrumpyArray, y: &GrumpyArray) -> PyResult<GrumpyArray> {
    let cl = leaf_1d(&cond.layout)?;
    let xl = leaf_1d(&x.layout)?;
    let yl = leaf_1d(&y.layout)?;
    if cond.dtype != DType::Bool {
        return Err(dtype_mismatch(DType::Bool, cond.dtype, "in where(cond,x,y)"));
    }
    if x.dtype != y.dtype {
        return Err(dtype_mismatch(x.dtype, y.dtype, "in where(cond,x,y)"));
    }
    if cl.len != xl.len || cl.len != yl.len {
        return Err(shape_mismatch(
            "where(cond,x,y)",
            "cond, x, and y must have the same length",
            "ensure all three arrays have equal length.",
        ));
    }
    let n = cl.len;
    let mut out = Leaf::new(x.dtype);
    out.len = n;
    out.has_nulls = cl.has_nulls || xl.has_nulls || yl.has_nulls;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    out.buffer = match x.dtype {
        DType::Int32 => LeafBuffer::I32(Arc::new(vec![0i32; n])),
        DType::Int64 => LeafBuffer::I64(Arc::new(vec![0i64; n])),
        DType::UInt32 => LeafBuffer::U32(Arc::new(vec![0u32; n])),
        DType::UInt64 => LeafBuffer::U64(Arc::new(vec![0u64; n])),
        DType::Float32 => LeafBuffer::F32(Arc::new(vec![0f32; n])),
        DType::Float64 => LeafBuffer::F64(Arc::new(vec![0f64; n])),
        DType::Bool => LeafBuffer::Bool(Arc::new(vec![0u8; n])),
        DType::Char => LeafBuffer::Char(Arc::new(vec![0u32; n])),
        _ => return Err(dtype_unsupported("where(cond,x,y)", x.dtype)),
    };
    let out_valid = Arc::make_mut(&mut out.validity);

    let cv = match &cl.buffer { LeafBuffer::Bool(v) => v.as_slice(), _ => unreachable!() };

    macro_rules! where_copy {
        ($variant:ident, $ty:ty) => {{
            let xv = match &xl.buffer { LeafBuffer::$variant(v) => v.as_slice(), _ => unreachable!() };
            let yv = match &yl.buffer { LeafBuffer::$variant(v) => v.as_slice(), _ => unreachable!() };
            let ov = match &mut out.buffer { LeafBuffer::$variant(v) => Arc::make_mut(v), _ => unreachable!() };
            for i in 0..n {
                if !cl.validity[i] {
                    out_valid.set(i, false);
                    continue;
                }
                let take_x = cv[i] != 0;
                if take_x {
                    if !xl.validity[i] {
                        out_valid.set(i, false);
                    } else {
                        ov[i] = xv[i] as $ty;
                    }
                } else {
                    if !yl.validity[i] {
                        out_valid.set(i, false);
                    } else {
                        ov[i] = yv[i] as $ty;
                    }
                }
            }
        }};
    }

    match x.dtype {
        DType::Int32 => where_copy!(I32, i32),
        DType::Int64 => where_copy!(I64, i64),
        DType::UInt32 => where_copy!(U32, u32),
        DType::UInt64 => where_copy!(U64, u64),
        DType::Float32 => where_copy!(F32, f32),
        DType::Float64 => where_copy!(F64, f64),
        DType::Bool => where_copy!(Bool, u8),
        DType::Char => where_copy!(Char, u32),
        _ => unreachable!(),
    }

    Ok(GrumpyArray { dtype: x.dtype, layout: Layout::Leaf(out) })
}

fn leaf_1d<'a>(layout: &'a Layout) -> PyResult<&'a Leaf> {
    match layout {
        Layout::Leaf(l) => Ok(l),
        Layout::OffsetView(v) => leaf_1d(v.content.as_ref()),
        Layout::Indexed(ix) => leaf_1d(ix.content.as_ref()),
        Layout::ListOffset(_) => Err(layout_unsupported("where", "expected a 1D leaf array")),
        Layout::UnionScalarList(_) => Err(layout_unsupported("where", "union arrays are not supported here")),
    }
}

fn new_leaf_i64_from(v: Vec<i64>) -> Leaf {
    let n = v.len();
    let mut leaf = Leaf::new(DType::Int64);
    leaf.len = n;
    leaf.has_nulls = false;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; n]);
    leaf.buffer = LeafBuffer::I64(Arc::new(v));
    leaf
}


