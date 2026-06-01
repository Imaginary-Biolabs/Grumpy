//! Layout-aware random sampling for Grumpy arrays.

use crate::dtype::DType;
use crate::layout::{
    drop_axis0_select_element, list_chain_depth, offsetview_to_listoffset, GrumpyArray, Layout,
    Leaf, LeafBuffer, ListOffset,
};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use half::f16;
use crate::error::{arg_invalid, shape_mismatch, dim_out_of_range, index_out_of_bounds, index_out_of_bounds_simple, internal, layout_unsupported, unsupported};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand_distr::{Distribution, StandardNormal, Uniform};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::sync::Arc;

/// Parsed ``size`` argument for :func:`choice`.
#[derive(Clone, Debug)]
pub enum ChoiceSize {
    Uniform(ChoiceCount),
    PerSlice(Vec<usize>),
}

#[derive(Clone, Copy, Debug)]
pub enum ChoiceCount {
    Count(usize),
    Fraction(f64),
}

pub struct GrumpyRng {
    inner: Pcg64,
}

impl GrumpyRng {
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Pcg64::seed_from_u64(seed),
        }
    }

    fn rng(&mut self) -> &mut Pcg64 {
        &mut self.inner
    }

    /// Fisher-Yates shuffle for a slice of indices (used by stream batch planning).
    pub fn shuffle_usizes(&mut self, items: &mut [usize]) {
        use rand::Rng;
        for i in (1..items.len()).rev() {
            let j = self.rng().gen_range(0..=i);
            items.swap(i, j);
        }
    }
}

fn is_integer_dtype(dt: DType) -> bool {
    matches!(
        dt,
        DType::Int8
            | DType::Int16
            | DType::Int32
            | DType::Int64
            | DType::UInt8
            | DType::UInt16
            | DType::UInt32
            | DType::UInt64
            | DType::Bool
    )
}

fn normalize_axis(dim: isize, depth: usize) -> PyResult<usize> {
    let ndims = depth as isize + 1;
    let mut d = dim;
    if d < 0 {
        d += ndims;
    }
    if d < 0 || d >= ndims {
        return Err(dim_out_of_range(dim, depth + 1));
    }
    Ok(d as usize)
}

fn resolve_count(spec: ChoiceCount, population: usize) -> PyResult<usize> {
    match spec {
        ChoiceCount::Count(k) => Ok(k),
        ChoiceCount::Fraction(f) => {
            if !(0.0..=1.0).contains(&f) {
                return Err(arg_invalid("size", "fraction must be in [0.0, 1.0]", "pass a fraction between 0 and 1 inclusive."));
            }
            Ok((f * population as f64).floor() as usize)
        }
    }
}

fn sample_indices(
    rng: &mut Pcg64,
    population: usize,
    k: usize,
    replace: bool,
) -> PyResult<Vec<usize>> {
    if population == 0 && k > 0 {
        return Err(arg_invalid("population", "cannot sample from an empty population", "ensure the axis has at least one element."));
    }
    if !replace && k > population {
        return Err(arg_invalid("size", format!("sample size {k} exceeds population {population} with replace=False"), "reduce k, enable replace=True, or enlarge the population."));
    }
    if k == 0 {
        return Ok(Vec::new());
    }
    if replace {
        Ok((0..k).map(|_| rng.gen_range(0..population)).collect())
    } else {
        let mut idx: Vec<usize> = (0..population).collect();
        for i in 0..k {
            let j = rng.gen_range(i..population);
            idx.swap(i, j);
        }
        idx.truncate(k);
        Ok(idx)
    }
}

fn ensure_pure_list_chain(layout: &Layout) -> PyResult<()> {
    if layout.has_union() {
        return Err(layout_unsupported("random ops", "require a pure list-chain array (no UnionScalarList)"));
    }
    Ok(())
}

fn canonical_listoffset(layout: &Layout) -> PyResult<ListOffset> {
    match layout {
        Layout::ListOffset(lo) => Ok(lo.clone()),
        Layout::OffsetView(v) => offsetview_to_listoffset(v),
        _ => Err(internal("canonical_listoffset", "expected ListOffset or OffsetView")),
    }
}

fn float_out_dtype(in_dt: DType) -> DType {
    match in_dt {
        DType::Float16 => DType::Float16,
        DType::Float32 => DType::Float32,
        _ => DType::Float64,
    }
}

fn reject_string_char(dt: DType, op: &str) -> PyResult<()> {
    match dt {
        DType::Char | DType::String => Err(unsupported(op, "char/string dtypes are not supported", "cast to a numeric dtype first.")),
        _ => Ok(()),
    }
}

#[derive(Clone, Copy)]
enum RandomFill {
    Uniform { low: f64, high: f64 },
    Normal { loc: f64, scale: f64 },
    Integers { low: i64, high: i64 },
}

pub fn uniform_like(
    rng: &mut GrumpyRng,
    x: &GrumpyArray,
    low: f64,
    high: f64,
) -> PyResult<GrumpyArray> {
    if low >= high {
        return Err(arg_invalid("low/high", "uniform_like requires low < high", "pass low strictly less than high."));
    }
    ensure_pure_list_chain(&x.layout)?;
    reject_string_char(x.dtype, "uniform_like")?;
    let out_dt = float_out_dtype(x.dtype);
    let layout = fill_random_layout(
        rng,
        &x.layout,
        out_dt,
        RandomFill::Uniform { low, high },
    )?;
    Ok(GrumpyArray {
        dtype: out_dt,
        layout,
    })
}

pub fn random_like(rng: &mut GrumpyRng, x: &GrumpyArray) -> PyResult<GrumpyArray> {
    uniform_like(rng, x, 0.0, 1.0)
}

pub fn normal_like(
    rng: &mut GrumpyRng,
    x: &GrumpyArray,
    loc: f64,
    scale: f64,
) -> PyResult<GrumpyArray> {
    if scale < 0.0 {
        return Err(arg_invalid("scale", "normal_like requires scale >= 0", "pass a non-negative scale."));
    }
    ensure_pure_list_chain(&x.layout)?;
    reject_string_char(x.dtype, "normal_like")?;
    let out_dt = float_out_dtype(x.dtype);
    let layout = fill_random_layout(
        rng,
        &x.layout,
        out_dt,
        RandomFill::Normal { loc, scale },
    )?;
    Ok(GrumpyArray {
        dtype: out_dt,
        layout,
    })
}

pub fn integers_like(
    rng: &mut GrumpyRng,
    x: &GrumpyArray,
    low: i64,
    high: i64,
    out_dt: DType,
) -> PyResult<GrumpyArray> {
    if low >= high {
        return Err(arg_invalid("low/high", "integers_like requires low < high", "pass low strictly less than high."));
    }
    ensure_pure_list_chain(&x.layout)?;
    reject_string_char(x.dtype, "integers_like")?;
    if !is_integer_dtype(out_dt) {
        return Err(arg_invalid("dtype", "integers_like dtype must be an integer dtype", "pass an integer dtype for integers_like."));
    }
    let layout = fill_random_layout(
        rng,
        &x.layout,
        out_dt,
        RandomFill::Integers { low, high },
    )?;
    Ok(GrumpyArray {
        dtype: out_dt,
        layout,
    })
}

pub fn integers(
    rng: &mut GrumpyRng,
    low: i64,
    high: i64,
    size: usize,
    out_dt: DType,
) -> PyResult<GrumpyArray> {
    if low >= high {
        return Err(arg_invalid("low/high", "integers requires low < high", "pass low strictly less than high."));
    }
    if !is_integer_dtype(out_dt) {
        return Err(arg_invalid("dtype", "integers dtype must be an integer dtype", "pass an integer dtype for integers."));
    }
    let range = (high - low) as u64;
    let mut leaf = Leaf::new(out_dt);
    leaf.len = size;
    leaf.validity = Arc::new(bitvec![u8, Lsb0; 1; size]);
    leaf.has_nulls = false;
    fill_leaf_int(
        &mut leaf,
        out_dt,
        |r| low + r.gen_range(0..range) as i64,
        rng.rng(),
    )?;
    Ok(GrumpyArray {
        dtype: out_dt,
        layout: Layout::Leaf(leaf),
    })
}

pub fn choice(
    rng: &mut GrumpyRng,
    x: &GrumpyArray,
    dim: isize,
    size: ChoiceSize,
    replace: bool,
) -> PyResult<GrumpyArray> {
    ensure_pure_list_chain(&x.layout)?;
    let depth = list_chain_depth(&x.layout)
        .ok_or_else(|| layout_unsupported("choice", "requires a pure list-chain array"))?;
    let axis = normalize_axis(dim, depth)?;
    let layout = choice_on_layout(rng, &x.layout, x.dtype, depth, axis, &size, replace)?;
    Ok(GrumpyArray {
        dtype: x.dtype,
        layout,
    })
}

pub fn permutation(rng: &mut GrumpyRng, x: &GrumpyArray, dim: isize) -> PyResult<GrumpyArray> {
    ensure_pure_list_chain(&x.layout)?;
    let depth = list_chain_depth(&x.layout)
        .ok_or_else(|| layout_unsupported("permutation", "requires a pure list-chain array"))?;
    let axis = normalize_axis(dim, depth)?;
    let layout = permute_axis_layout(rng, &x.layout, x.dtype, depth, axis)?;
    Ok(GrumpyArray {
        dtype: x.dtype,
        layout,
    })
}

pub fn shuffle(rng: &mut GrumpyRng, x: &mut GrumpyArray, dim: isize) -> PyResult<()> {
    x.uniquify_buffers();
    if let Layout::UnionScalarList(u) = &mut x.layout {
        let axis = normalize_axis(dim, 0)?;
        if axis != 0 {
            return Err(unsupported("shuffle", "union arrays support dim=0 only for now", "shuffle along the outer union axis only."));
        }
        let n = u.len();
        let idx = sample_indices(rng.rng(), n, n, false)?;
        let old_tags = std::mem::replace(&mut u.tags, Vec::with_capacity(n));
        let old_index = std::mem::replace(&mut u.index, Vec::with_capacity(n));
        u.tags = idx.iter().map(|&i| old_tags[i]).collect();
        u.index = idx.iter().map(|&i| old_index[i]).collect();
        return Ok(());
    }
    let out = permutation(rng, x, dim)?;
    x.layout = out.layout;
    Ok(())
}

fn choice_on_layout(
    rng: &mut GrumpyRng,
    layout: &Layout,
    dtype: DType,
    depth: usize,
    axis: usize,
    size: &ChoiceSize,
    replace: bool,
) -> PyResult<Layout> {
    if depth == 0 {
        if axis != 0 {
            return Err(dim_out_of_range(axis as isize, depth + 1));
        }
        let leaf = match layout {
            Layout::Leaf(l) => l,
            _ => unreachable!(),
        };
        let k = match size {
            ChoiceSize::Uniform(c) => resolve_count(*c, leaf.len)?,
            ChoiceSize::PerSlice(v) => {
                if v.len() != 1 {
                    return Err(shape_mismatch("choice", "size list length must match orthogonal slice count", "pass one size per orthogonal slice."));
                }
                v[0]
            }
        };
        let idx = sample_indices(rng.rng(), leaf.len, k, replace)?;
        return gather_leaf_segment(leaf, &idx);
    }

    if axis == 0 {
        let lo = canonical_listoffset(layout)?;
        let n = lo.len();
        let k = match size {
            ChoiceSize::Uniform(c) => resolve_count(*c, n)?,
            ChoiceSize::PerSlice(v) => {
                if v.len() != 1 {
                    return Err(arg_invalid("size", "size list is not supported with dim=0", "use an int, float fraction, or single-element list."));
                }
                v[0]
            }
        };
        let idx = sample_indices(rng.rng(), n, k, replace)?;
        let mut elems = Vec::with_capacity(k);
        for &i in &idx {
            elems.push(drop_axis0_select_element(
                &Layout::ListOffset(lo.clone()),
                i,
            )?);
        }
        if elems.len() == 1 {
            return Ok(elems.into_iter().next().unwrap());
        }
        return stack_axis0(&elems);
    }

    let lo = canonical_listoffset(layout)?;
    let n = lo.len();
    if let ChoiceSize::PerSlice(v) = size {
        if v.len() != n {
            return Err(shape_mismatch(
                "choice",
                format!(
                    "size list length ({}) must match outer length ({}) for dim={}",
                    v.len(),
                    n,
                    axis
                ),
                "pass one size per outer element.",
            ));
        }
    }
    let mut out_elems: Vec<Layout> = Vec::with_capacity(n);
    for i in 0..n {
        let child = drop_axis0_select_element(&Layout::ListOffset(lo.clone()), i)?;
        let child_size = match size {
            ChoiceSize::Uniform(c) => ChoiceSize::Uniform(*c),
            ChoiceSize::PerSlice(v) => ChoiceSize::Uniform(ChoiceCount::Count(v[i])),
        };
        out_elems.push(choice_on_layout(
            rng,
            &child,
            dtype,
            depth - 1,
            axis - 1,
            &child_size,
            replace,
        )?);
    }
    stack_axis0(&out_elems)
}

fn permute_axis_layout(
    rng: &mut GrumpyRng,
    layout: &Layout,
    dtype: DType,
    depth: usize,
    axis: usize,
) -> PyResult<Layout> {
    let _ = dtype;
    if depth == 0 {
        if axis != 0 {
            return Err(dim_out_of_range(axis as isize, depth + 1));
        }
        let leaf = match layout {
            Layout::Leaf(l) => l,
            _ => unreachable!(),
        };
        let n = leaf.len;
        let idx = sample_indices(rng.rng(), n, n, false)?;
        return gather_leaf_segment(leaf, &idx);
    }

    if axis == 0 {
        let lo = canonical_listoffset(layout)?;
        let n = lo.len();
        let idx = sample_indices(rng.rng(), n, n, false)?;
        let mut elems = Vec::with_capacity(n);
        for &i in &idx {
            elems.push(drop_axis0_select_element(
                &Layout::ListOffset(lo.clone()),
                i,
            )?);
        }
        return stack_axis0(&elems);
    }

    let lo = canonical_listoffset(layout)?;
    let n = lo.len();
    let mut out_elems: Vec<Layout> = Vec::with_capacity(n);
    for i in 0..n {
        let child = drop_axis0_select_element(&Layout::ListOffset(lo.clone()), i)?;
        let inner_n = child.len();
        let idx = sample_indices(rng.rng(), inner_n, inner_n, false)?;
        out_elems.push(permute_layout_inner(
            &child,
            depth - 1,
            axis - 1,
            &idx,
        )?);
    }
    stack_axis0(&out_elems)
}

fn permute_layout_inner(
    layout: &Layout,
    depth: usize,
    axis: usize,
    indices: &[usize],
) -> PyResult<Layout> {
    if depth == 0 {
        if axis != 0 {
            return Err(dim_out_of_range(axis as isize, depth + 1));
        }
        let leaf = match layout {
            Layout::Leaf(l) => l,
            _ => unreachable!(),
        };
        return gather_leaf_segment(leaf, indices);
    }
    if axis == 0 {
        let lo = canonical_listoffset(layout)?;
        let mut elems = Vec::with_capacity(indices.len());
        for &i in indices {
            elems.push(drop_axis0_select_element(
                &Layout::ListOffset(lo.clone()),
                i,
            )?);
        }
        return stack_axis0(&elems);
    }
    let lo = canonical_listoffset(layout)?;
    let n = lo.len();
    let mut out_elems: Vec<Layout> = Vec::with_capacity(n);
    for i in 0..n {
        let child = drop_axis0_select_element(&Layout::ListOffset(lo.clone()), i)?;
        out_elems.push(permute_layout_inner(
            &child,
            depth - 1,
            axis - 1,
            indices,
        )?);
    }
    stack_axis0(&out_elems)
}

fn gather_leaf_segment(leaf: &Leaf, indices: &[usize]) -> PyResult<Layout> {
    let mut out = Leaf::new(leaf.dtype);
    out.len = indices.len();
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; indices.len()]);
    out.has_nulls = false;
    out.buffer = empty_buffer(&leaf.buffer);
    let out_valid = Arc::make_mut(&mut out.validity);
    for (dst, &src) in indices.iter().enumerate() {
        if src >= leaf.len {
            return Err(index_out_of_bounds_simple("during random sampling"));
        }
        if !leaf.validity[src] {
            out_valid.set(dst, false);
            out.has_nulls = true;
        }
        out.buffer.push_from_index(&leaf.buffer, src)?;
    }
    Ok(Layout::Leaf(out))
}

fn empty_buffer(src: &LeafBuffer) -> LeafBuffer {
    match src {
        LeafBuffer::I8(_) => LeafBuffer::I8(Arc::new(Vec::new())),
        LeafBuffer::I16(_) => LeafBuffer::I16(Arc::new(Vec::new())),
        LeafBuffer::I32(_) => LeafBuffer::I32(Arc::new(Vec::new())),
        LeafBuffer::I64(_) => LeafBuffer::I64(Arc::new(Vec::new())),
        LeafBuffer::U8(_) => LeafBuffer::U8(Arc::new(Vec::new())),
        LeafBuffer::U16(_) => LeafBuffer::U16(Arc::new(Vec::new())),
        LeafBuffer::U32(_) => LeafBuffer::U32(Arc::new(Vec::new())),
        LeafBuffer::U64(_) => LeafBuffer::U64(Arc::new(Vec::new())),
        LeafBuffer::F16(_) => LeafBuffer::F16(Arc::new(Vec::new())),
        LeafBuffer::F32(_) => LeafBuffer::F32(Arc::new(Vec::new())),
        LeafBuffer::F64(_) => LeafBuffer::F64(Arc::new(Vec::new())),
        LeafBuffer::Bool(_) => LeafBuffer::Bool(Arc::new(Vec::new())),
        LeafBuffer::Char(_) => LeafBuffer::Char(Arc::new(Vec::new())),
        LeafBuffer::String(_) => LeafBuffer::String(Arc::new(Vec::new())),
    }
}

fn stack_axis0(elems: &[Layout]) -> PyResult<Layout> {
    if elems.is_empty() {
        return Ok(Layout::Leaf(Leaf::new(DType::Int64)));
    }
    if matches!(elems[0], Layout::Leaf(_)) {
        let all_scalar = elems
            .iter()
            .all(|e| matches!(e, Layout::Leaf(l) if l.len == 1));
        if all_scalar {
            return concat_scalar_leaves(elems);
        }
        let mut offsets: Vec<i64> = vec![0];
        let mut acc: i64 = 0;
        for e in elems {
            acc += e.len() as i64;
            offsets.push(acc);
        }
        let content = concat_leaves(elems)?;
        return Ok(Layout::ListOffset(ListOffset {
            offsets: Arc::new(offsets),
            content: Box::new(content),
        }));
    }
    let mut offsets: Vec<i64> = vec![0];
    let mut acc: i64 = 0;
    for e in elems {
        acc += e.len() as i64;
        offsets.push(acc);
    }
    let content = concat_axis0_layouts(elems)?;
    Ok(Layout::ListOffset(ListOffset {
        offsets: Arc::new(offsets),
        content: Box::new(content),
    }))
}

fn concat_scalar_leaves(elems: &[Layout]) -> PyResult<Layout> {
    let dt = match &elems[0] {
        Layout::Leaf(l) => l.dtype,
        _ => unreachable!(),
    };
    let mut out = Leaf::new(dt);
    out.len = elems.len();
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; elems.len()]);
    out.has_nulls = false;
    out.buffer = empty_buffer(match &elems[0] {
        Layout::Leaf(l) => &l.buffer,
        _ => unreachable!(),
    });
    let out_valid = Arc::make_mut(&mut out.validity);
    for (i, e) in elems.iter().enumerate() {
        let l = match e {
            Layout::Leaf(l) => l,
            _ => unreachable!(),
        };
        if !l.validity[0] {
            out_valid.set(i, false);
            out.has_nulls = true;
        }
        out.buffer.push_from_index(&l.buffer, 0)?;
    }
    Ok(Layout::Leaf(out))
}

fn concat_leaves(elems: &[Layout]) -> PyResult<Layout> {
    let dt = match &elems[0] {
        Layout::Leaf(l) => l.dtype,
        _ => return Err(internal("concat_len1_leaves", "expected leaf layouts")),
    };
    let total: usize = elems.iter().map(|e| e.len()).sum();
    let mut out = Leaf::new(dt);
    out.len = total;
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; total]);
    out.has_nulls = false;
    out.buffer = empty_buffer(match &elems[0] {
        Layout::Leaf(l) => &l.buffer,
        _ => unreachable!(),
    });
    let out_valid = Arc::make_mut(&mut out.validity);
    let mut dst = 0usize;
    for e in elems {
        let l = match e {
            Layout::Leaf(l) => l,
            _ => unreachable!(),
        };
        for i in 0..l.len {
            if !l.validity[i] {
                out_valid.set(dst, false);
                out.has_nulls = true;
            }
            out.buffer.push_from_index(&l.buffer, i)?;
            dst += 1;
        }
    }
    Ok(Layout::Leaf(out))
}

fn concat_axis0_layouts(layouts: &[Layout]) -> PyResult<Layout> {
    if layouts.is_empty() {
        return Err(internal("concat_layouts_axis0", "cannot concat empty layouts"));
    }
    match &layouts[0] {
        Layout::Leaf(_) => concat_leaves(layouts),
        _ => {
            let mut flat: Vec<Layout> = Vec::new();
            for l in layouts {
                let lo = canonical_listoffset(l)?;
                for i in 0..lo.len() {
                    flat.push(drop_axis0_select_element(
                        &Layout::ListOffset(lo.clone()),
                        i,
                    )?);
                }
            }
            concat_axis0_layouts(&flat)
        }
    }
}

fn fill_random_layout(
    rng: &mut GrumpyRng,
    layout: &Layout,
    out_dt: DType,
    mode: RandomFill,
) -> PyResult<Layout> {
    match layout {
        Layout::Leaf(l) => {
            let mut out = Leaf::new(out_dt);
            out.len = l.len;
            out.validity = Arc::new(bitvec![u8, Lsb0; 1; l.len]);
            out.has_nulls = false;
            match mode {
                RandomFill::Uniform { low, high } => {
                    let dist = Uniform::new(low, high);
                    fill_leaf_float(&mut out, out_dt, |r| dist.sample(r), rng.rng())?;
                }
                RandomFill::Normal { loc, scale } => {
                    let dist = StandardNormal;
                    fill_leaf_float(&mut out, out_dt, |r| {
                        let z: f64 = dist.sample(r);
                        loc + scale * z
                    }, rng.rng())?;
                }
                RandomFill::Integers { low, high } => {
                    let range = (high - low) as u64;
                    fill_leaf_int(
                        &mut out,
                        out_dt,
                        |r| low + r.gen_range(0..range) as i64,
                        rng.rng(),
                    )?;
                }
            }
            Ok(Layout::Leaf(out))
        }
        Layout::ListOffset(lo) => {
            let content = fill_random_layout(rng, lo.content.as_ref(), out_dt, mode)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: lo.offsets.clone(),
                content: Box::new(content),
            }))
        }
        Layout::OffsetView(v) => {
            let lo = offsetview_to_listoffset(v)?;
            fill_random_layout(rng, &Layout::ListOffset(lo), out_dt, mode)
        }
        Layout::Indexed(ix) => {
            let content = fill_random_layout(rng, ix.content.as_ref(), out_dt, mode)?;
            Ok(crate::layout::Layout::Indexed(crate::layout::Indexed {
                index: ix.index.clone(),
                content: Box::new(content),
            }))
        }
        Layout::UnionScalarList(_) => Err(layout_unsupported("shuffle", 
            "random ops require a pure list-chain array (no UnionScalarList).",
        )),
    }
}

fn fill_leaf_float<F>(leaf: &mut Leaf, dt: DType, mut sample: F, rng: &mut Pcg64) -> PyResult<()>
where
    F: FnMut(&mut Pcg64) -> f64,
{
    leaf.buffer = match dt {
        DType::Float16 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(f16::from_f64(sample(rng)).to_bits());
            }
            LeafBuffer::F16(Arc::new(v))
        }
        DType::Float32 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng) as f32);
            }
            LeafBuffer::F32(Arc::new(v))
        }
        _ => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng));
            }
            LeafBuffer::F64(Arc::new(v))
        }
    };
    Ok(())
}

fn fill_leaf_int<F>(leaf: &mut Leaf, dt: DType, mut sample: F, rng: &mut Pcg64) -> PyResult<()>
where
    F: FnMut(&mut Pcg64) -> i64,
{
    leaf.buffer = match dt {
        DType::Int8 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng) as i8);
            }
            LeafBuffer::I8(Arc::new(v))
        }
        DType::Int16 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng) as i16);
            }
            LeafBuffer::I16(Arc::new(v))
        }
        DType::Int32 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng) as i32);
            }
            LeafBuffer::I32(Arc::new(v))
        }
        DType::Int64 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng));
            }
            LeafBuffer::I64(Arc::new(v))
        }
        DType::UInt8 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng) as u8);
            }
            LeafBuffer::U8(Arc::new(v))
        }
        DType::UInt16 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng) as u16);
            }
            LeafBuffer::U16(Arc::new(v))
        }
        DType::UInt32 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng) as u32);
            }
            LeafBuffer::U32(Arc::new(v))
        }
        DType::UInt64 => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(sample(rng) as u64);
            }
            LeafBuffer::U64(Arc::new(v))
        }
        DType::Bool => {
            let mut v = Vec::with_capacity(leaf.len);
            for _ in 0..leaf.len {
                v.push(if sample(rng) & 1 == 0 { 0 } else { 1 });
            }
            LeafBuffer::Bool(Arc::new(v))
        }
        _ => {
            return Err(arg_invalid("dtype", "integers_like dtype must be an integer or bool dtype", "pass an integer or bool dtype."))
        }
    };
    Ok(())
}

/// Parse a Python ``size`` value for ``choice``.
pub fn parse_choice_size(py: Python<'_>, size: &Bound<'_, PyAny>) -> PyResult<ChoiceSize> {
    if let Ok(v) = size.extract::<usize>() {
        return Ok(ChoiceSize::Uniform(ChoiceCount::Count(v)));
    }
    if let Ok(v) = size.extract::<isize>() {
        if v < 0 {
            return Err(arg_invalid("size", "choice size must be non-negative", "pass size >= 0."));
        }
        return Ok(ChoiceSize::Uniform(ChoiceCount::Count(v as usize)));
    }
    if let Ok(v) = size.extract::<f64>() {
        return Ok(ChoiceSize::Uniform(ChoiceCount::Fraction(v)));
    }
    if let Ok(seq) = size.downcast::<pyo3::types::PySequence>() {
        let n = seq.len()?;
        let mut out = Vec::with_capacity(n as usize);
        for i in 0..n {
            let item = seq.get_item(i)?;
            let v: isize = item.extract()?;
            if v < 0 {
                return Err(arg_invalid("size", "choice size list entries must be non-negative", "pass non-negative counts or fractions."));
            }
            out.push(v as usize);
        }
        return Ok(ChoiceSize::PerSlice(out));
    }
    let _ = py;
    Err(arg_invalid("size", "must be an int, float fraction, or list of ints", "pass size as count, fraction in [0,1], or per-slice list."))
}

