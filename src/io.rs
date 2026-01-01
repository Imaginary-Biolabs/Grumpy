use crate::dataframe::{GrumpyDataFrame, Schema};
use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, OffsetView, UnionScalarList};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use zarrs::array::{Array, ArrayBuilder, DataType, ElementOwned, FillValue};
use zarrs::array::chunk_grid::ChunkGrid;
use zarrs::storage::ReadableWritableListableStorage;

const META_FILE: &str = "grumpy.json";
const FORMAT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum RootMeta {
    #[serde(rename = "array")]
    Array { dtype: DTypeSer, layout: LayoutMeta },
    #[serde(rename = "dataframe")]
    DataFrame { schema: Option<Vec<Vec<String>>>, columns: Vec<ColumnMeta> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileMeta {
    pub version: u32,
    pub root: RootMeta,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ColumnMeta {
    pub name: String,
    pub dtype: DTypeSer,
    pub layout: LayoutMeta,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DTypeSer {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    Float32,
    Float64,
    Bool,
    Char,
    String,
}

impl From<DType> for DTypeSer {
    fn from(v: DType) -> Self {
        match v {
            DType::Int8 => Self::Int8,
            DType::Int16 => Self::Int16,
            DType::Int32 => Self::Int32,
            DType::Int64 => Self::Int64,
            DType::UInt8 => Self::UInt8,
            DType::UInt16 => Self::UInt16,
            DType::UInt32 => Self::UInt32,
            DType::UInt64 => Self::UInt64,
            DType::Float16 => Self::Float16,
            DType::Float32 => Self::Float32,
            DType::Float64 => Self::Float64,
            DType::Bool => Self::Bool,
            DType::Char => Self::Char,
            DType::String => Self::String,
        }
    }
}

impl From<DTypeSer> for DType {
    fn from(v: DTypeSer) -> Self {
        match v {
            DTypeSer::Int8 => DType::Int8,
            DTypeSer::Int16 => DType::Int16,
            DTypeSer::Int32 => DType::Int32,
            DTypeSer::Int64 => DType::Int64,
            DTypeSer::UInt8 => DType::UInt8,
            DTypeSer::UInt16 => DType::UInt16,
            DTypeSer::UInt32 => DType::UInt32,
            DTypeSer::UInt64 => DType::UInt64,
            DTypeSer::Float16 => DType::Float16,
            DTypeSer::Float32 => DType::Float32,
            DTypeSer::Float64 => DType::Float64,
            DTypeSer::Bool => DType::Bool,
            DTypeSer::Char => DType::Char,
            DTypeSer::String => DType::String,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayoutMeta {
    Leaf { len: usize, values: String, validity: String },
    ListOffset { offsets: String, content: Box<LayoutMeta> },
    OffsetView { offsets: String, start: usize, stop: usize, content: Box<LayoutMeta> },
    Indexed { index: String, content: Box<LayoutMeta> },
    UnionScalarList { tags: String, index: String, scalars: Box<LayoutMeta>, lists: Box<LayoutMeta> },
}

struct SaveCtx {
    next: usize,
    chunk_size: usize,
}

pub fn save_array(py: Python<'_>, arr: &GrumpyArray, path: &str, chunk_size: usize) -> PyResult<()> {
    let _ = py;
    ensure_dir(path)?;
    let store = store_fs(path)?;
    init_root_group(&store)?;
    init_group(&store, "/buffers")?;
    let mut ctx = SaveCtx { next: 0, chunk_size: chunk_size.max(1) };
    let layout = save_layout(&store, &mut ctx, arr.dtype, &arr.layout)?;
    let meta = FileMeta { version: FORMAT_VERSION, root: RootMeta::Array { dtype: arr.dtype.into(), layout } };
    write_meta(path, &meta)?;
    Ok(())
}

pub fn load_array(py: Python<'_>, path: &str) -> PyResult<GrumpyArray> {
    let _ = py;
    let meta = read_meta(path)?;
    match meta.root {
        RootMeta::Array { dtype, layout } => {
            let store = store_fs(path)?;
            let dt: DType = dtype.into();
            let layout = load_layout(&store, dt, &layout)?;
            Ok(GrumpyArray { dtype: dt, layout })
        }
        _ => Err(PyValueError::new_err("Path does not contain a saved GrumpyArray.")),
    }
}

pub fn save_dataframe(py: Python<'_>, df: &GrumpyDataFrame, path: &str, chunk_size: usize) -> PyResult<()> {
    let _ = py;
    ensure_dir(path)?;
    let store = store_fs(path)?;
    init_root_group(&store)?;
    init_group(&store, "/buffers")?;
    let mut ctx = SaveCtx { next: 0, chunk_size: chunk_size.max(1) };
    let mut columns: Vec<ColumnMeta> = Vec::new();
    for (name, col) in df.names.iter().zip(df.cols.iter()) {
        let layout = save_layout(&store, &mut ctx, col.dtype, &col.layout)?;
        columns.push(ColumnMeta { name: name.clone(), dtype: col.dtype.into(), layout });
    }
    let schema_levels = df.schema.as_ref().map(|s| s.levels.clone());
    let meta = FileMeta { version: FORMAT_VERSION, root: RootMeta::DataFrame { schema: schema_levels, columns } };
    write_meta(path, &meta)?;
    Ok(())
}

pub fn load_dataframe(py: Python<'_>, path: &str) -> PyResult<GrumpyDataFrame> {
    let _ = py;
    let meta = read_meta(path)?;
    match meta.root {
        RootMeta::DataFrame { schema, columns } => {
            let store = store_fs(path)?;
            let mut names: Vec<String> = Vec::new();
            let mut cols: Vec<GrumpyArray> = Vec::new();
            for c in columns {
                let dt: DType = c.dtype.into();
                let layout = load_layout(&store, dt, &c.layout)?;
                names.push(c.name);
                cols.push(GrumpyArray { dtype: dt, layout });
            }
            let schema = match schema {
                None => None,
                Some(levels) => {
                    // Build Schema directly.
                    let mut name_to_level = std::collections::HashMap::new();
                    for (lvl, names) in levels.iter().enumerate() {
                        for n in names {
                            name_to_level.insert(n.clone(), lvl);
                        }
                    }
                    Some(Schema { levels, name_to_level })
                }
            };
            GrumpyDataFrame::new(names, cols, schema)
        }
        _ => Err(PyValueError::new_err("Path does not contain a saved GrumpyDataFrame.")),
    }
}

fn save_layout(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, dt: DType, layout: &Layout) -> PyResult<LayoutMeta> {
    match layout {
        Layout::Leaf(leaf) => save_leaf(store, ctx, dt, leaf),
        Layout::ListOffset(lo) => {
            let offsets = write_vec_i64(store, ctx, "offsets", lo.offsets.as_slice())?;
            let content = save_layout(store, ctx, dt, lo.content.as_ref())?;
            Ok(LayoutMeta::ListOffset { offsets, content: Box::new(content) })
        }
        Layout::OffsetView(v) => {
            let offsets = write_vec_i64(store, ctx, "offsets", v.offsets.as_slice())?;
            let content = save_layout(store, ctx, dt, v.content.as_ref())?;
            Ok(LayoutMeta::OffsetView { offsets, start: v.start, stop: v.stop, content: Box::new(content) })
        }
        Layout::Indexed(ix) => {
            let index = write_vec_i64(store, ctx, "index", ix.index.as_slice())?;
            let content = save_layout(store, ctx, dt, ix.content.as_ref())?;
            Ok(LayoutMeta::Indexed { index, content: Box::new(content) })
        }
        Layout::UnionScalarList(u) => {
            let tags = write_vec_u8(store, ctx, "tags", &u.tags)?;
            let index = write_vec_i64(store, ctx, "index", &u.index)?;
            let scalars = save_leaf(store, ctx, dt, &u.scalars)?;
            let lists = save_layout(store, ctx, dt, &Layout::ListOffset(u.lists.clone()))?;
            Ok(LayoutMeta::UnionScalarList { tags, index, scalars: Box::new(scalars), lists: Box::new(lists) })
        }
    }
}

fn save_leaf(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, dt: DType, leaf: &Leaf) -> PyResult<LayoutMeta> {
    let len = leaf.len;
    let validity_vec: Vec<bool> = leaf.validity.iter().by_vals().collect();
    let validity = write_vec_bool(store, ctx, "validity", validity_vec.as_slice())?;
    let values = match (&leaf.buffer, dt) {
        (LeafBuffer::I8(v), DType::Int8) => write_vec_i8(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::I16(v), DType::Int16) => write_vec_i16(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::I32(v), DType::Int32) => write_vec_i32(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::I64(v), DType::Int64) => write_vec_i64(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::U8(v), DType::UInt8) => write_vec_u8(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::U16(v), DType::UInt16) => write_vec_u16(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::U32(v), DType::UInt32) => write_vec_u32(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::U64(v), DType::UInt64) => write_vec_u64(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::F16(v), DType::Float16) => write_vec_u16(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::F32(v), DType::Float32) => write_vec_f32(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::F64(v), DType::Float64) => write_vec_f64(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::Bool(v), DType::Bool) => {
            let b: Vec<bool> = v.as_slice().iter().map(|&x| x != 0).collect();
            write_vec_bool(store, ctx, "values", &b)?
        }
        (LeafBuffer::Char(v), DType::Char) => write_vec_u32(store, ctx, "values", v.as_slice())?,
        (LeafBuffer::String(v), DType::String) => write_vec_string(store, ctx, "values", v.as_slice())?,
        _ => return Err(PyValueError::new_err("Internal error: dtype mismatch in save_leaf.")),
    };
    Ok(LayoutMeta::Leaf { len, values, validity })
}

fn load_layout(store: &ReadableWritableListableStorage, dt: DType, meta: &LayoutMeta) -> PyResult<Layout> {
    match meta {
        LayoutMeta::Leaf { len, values, validity } => {
            let valid = read_vec_bool(store, validity)?;
            if valid.len() != *len {
                return Err(PyValueError::new_err("Invalid validity length in file."));
            }
            let mut leaf = Leaf::new(dt);
            leaf.len = *len;
            leaf.has_nulls = valid.iter().any(|b| !*b);
            leaf.validity = Arc::new(bitvec::vec::BitVec::<u8, bitvec::order::Lsb0>::from_iter(valid.iter().copied()));
            leaf.buffer = match dt {
                DType::Int8 => LeafBuffer::I8(Arc::new(read_vec::<i8>(store, values)?)),
                DType::Int16 => LeafBuffer::I16(Arc::new(read_vec::<i16>(store, values)?)),
                DType::Int32 => LeafBuffer::I32(Arc::new(read_vec::<i32>(store, values)?)),
                DType::Int64 => LeafBuffer::I64(Arc::new(read_vec::<i64>(store, values)?)),
                DType::UInt8 => LeafBuffer::U8(Arc::new(read_vec::<u8>(store, values)?)),
                DType::UInt16 => LeafBuffer::U16(Arc::new(read_vec::<u16>(store, values)?)),
                DType::UInt32 => LeafBuffer::U32(Arc::new(read_vec::<u32>(store, values)?)),
                DType::UInt64 => LeafBuffer::U64(Arc::new(read_vec::<u64>(store, values)?)),
                DType::Float16 => LeafBuffer::F16(Arc::new(read_vec::<u16>(store, values)?)),
                DType::Float32 => LeafBuffer::F32(Arc::new(read_vec::<f32>(store, values)?)),
                DType::Float64 => LeafBuffer::F64(Arc::new(read_vec::<f64>(store, values)?)),
                DType::Bool => {
                    let b = read_vec::<bool>(store, values)?;
                    LeafBuffer::Bool(Arc::new(b.into_iter().map(|x| if x { 1 } else { 0 }).collect()))
                }
                DType::Char => LeafBuffer::Char(Arc::new(read_vec::<u32>(store, values)?)),
                DType::String => LeafBuffer::String(Arc::new(read_vec::<String>(store, values)?)),
            };
            Ok(Layout::Leaf(leaf))
        }
        LayoutMeta::ListOffset { offsets, content } => {
            let offs = read_vec::<i64>(store, offsets)?;
            let content = load_layout(store, dt, content)?;
            Ok(Layout::ListOffset(ListOffset { offsets: Arc::new(offs), content: Box::new(content) }))
        }
        LayoutMeta::OffsetView { offsets, start, stop, content } => {
            let offs = read_vec::<i64>(store, offsets)?;
            let content = load_layout(store, dt, content)?;
            Ok(Layout::OffsetView(OffsetView { offsets: Arc::new(offs), start: *start, stop: *stop, content: Box::new(content) }))
        }
        LayoutMeta::Indexed { index, content } => {
            let idx = read_vec::<i64>(store, index)?;
            let content = load_layout(store, dt, content)?;
            Ok(Layout::Indexed(crate::layout::Indexed { index: Arc::new(idx), content: Box::new(content) }))
        }
        LayoutMeta::UnionScalarList { tags, index, scalars, lists } => {
            let tags = read_vec::<u8>(store, tags)?;
            let index = read_vec::<i64>(store, index)?;
            let scal = match load_layout(store, dt, scalars)? {
                Layout::Leaf(l) => l,
                _ => return Err(PyValueError::new_err("Invalid union scalars layout in file.")),
            };
            let lists_layout = load_layout(store, dt, lists)?;
            let lists = match lists_layout {
                Layout::ListOffset(lo) => lo,
                _ => return Err(PyValueError::new_err("Invalid union lists layout in file.")),
            };
            Ok(Layout::UnionScalarList(UnionScalarList { tags, index, scalars: scal, lists }))
        }
    }
}

fn ensure_dir(path: &str) -> PyResult<()> {
    fs::create_dir_all(path).map_err(|e| PyValueError::new_err(format!("Failed to create directory: {e}")))?;
    Ok(())
}

fn store_fs(path: &str) -> PyResult<ReadableWritableListableStorage> {
    let store: ReadableWritableListableStorage = Arc::new(zarrs::filesystem::FilesystemStore::new(path).map_err(|e| PyValueError::new_err(format!("{e}")))?);
    Ok(store)
}

fn init_root_group(store: &ReadableWritableListableStorage) -> PyResult<()> {
    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|e| PyValueError::new_err(format!("{e}")))?
        .store_metadata()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(())
}

fn init_group(store: &ReadableWritableListableStorage, path: &str) -> PyResult<()> {
    zarrs::group::GroupBuilder::new()
        .build(store.clone(), path)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?
        .store_metadata()
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(())
}

fn write_meta(path: &str, meta: &FileMeta) -> PyResult<()> {
    let p = Path::new(path).join(META_FILE);
    let s = serde_json::to_string_pretty(meta).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    fs::write(p, s).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(())
}

fn read_meta(path: &str) -> PyResult<FileMeta> {
    let p = Path::new(path).join(META_FILE);
    let s = fs::read_to_string(p).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let meta: FileMeta = serde_json::from_str(&s).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    if meta.version != FORMAT_VERSION {
        return Err(PyValueError::new_err("Unsupported file version."));
    }
    Ok(meta)
}

fn next_path(ctx: &mut SaveCtx, prefix: &str) -> String {
    let id = ctx.next;
    ctx.next += 1;
    format!("/buffers/{prefix}_{id}")
}

fn write_1d<T: ElementOwned>(
    store: &ReadableWritableListableStorage,
    ctx: &mut SaveCtx,
    prefix: &str,
    dt: DataType,
    fill: FillValue,
    data: &[T],
) -> PyResult<String>
where
    T: Clone,
{
    let path = next_path(ctx, prefix);
    let n = data.len();
    let chunk = std::cmp::min(ctx.chunk_size, std::cmp::max(1, n));
    let nz = std::num::NonZeroU64::new(chunk as u64).unwrap();
    let chunk_grid = ChunkGrid::from(vec![nz]);
    let arr = ArrayBuilder::new(vec![n as u64], dt, chunk_grid, fill)
        .build(store.clone(), &path)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    arr.store_metadata().map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let subset = arr.subset_all();
    arr.store_array_subset_elements::<T>(&subset, &data.to_vec())
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(path)
}

fn read_vec<T: ElementOwned>(store: &ReadableWritableListableStorage, path: &str) -> PyResult<Vec<T>> {
    let arr = Array::open(store.clone(), path).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let subset = arr.subset_all();
    arr.retrieve_array_subset_elements::<T>(&subset)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

fn write_vec_i64(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[i64]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Int64, FillValue::from(0i64), data)
}
fn write_vec_i32(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[i32]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Int32, FillValue::from(0i32), data)
}
fn write_vec_i16(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[i16]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Int16, FillValue::from(0i16), data)
}
fn write_vec_i8(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[i8]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Int8, FillValue::from(0i8), data)
}
fn write_vec_u64(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[u64]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::UInt64, FillValue::from(0u64), data)
}
fn write_vec_u32(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[u32]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::UInt32, FillValue::from(0u32), data)
}
fn write_vec_u16(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[u16]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::UInt16, FillValue::from(0u16), data)
}
fn write_vec_u8(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[u8]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::UInt8, FillValue::from(0u8), data)
}
fn write_vec_f64(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[f64]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Float64, FillValue::from(0.0f64), data)
}
fn write_vec_f32(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[f32]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Float32, FillValue::from(0.0f32), data)
}
fn write_vec_bool(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[bool]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Bool, FillValue::from(false), data)
}
fn write_vec_string(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[String]) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::String, FillValue::from(""), data)
}

fn read_vec_bool(store: &ReadableWritableListableStorage, path: &str) -> PyResult<Vec<bool>> {
    read_vec::<bool>(store, path)
}


