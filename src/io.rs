use crate::dataframe::{GrumpyDataFrame, Schema};
use crate::dtype::DType;
use crate::layout::{GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, OffsetView, UnionScalarList};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use zarrs::array_subset::ArraySubset;
use zarrs::array::{Array, ArrayBuilder, DataType, ElementOwned, FillValue};
use zarrs::array::chunk_grid::ChunkGrid;
use zarrs::storage::ReadableWritableListableStorage;

static IO_BYTES_READ: AtomicUsize = AtomicUsize::new(0);

/// Bytes read from Zarr via partial I/O helpers (for tests).
pub fn io_bytes_read() -> usize {
    IO_BYTES_READ.load(Ordering::Relaxed)
}

/// Reset the partial I/O byte counter (for tests).
pub fn reset_io_bytes_read() {
    IO_BYTES_READ.store(0, Ordering::Relaxed);
}

fn record_io_bytes(n: usize) {
    IO_BYTES_READ.fetch_add(n, Ordering::Relaxed);
}

const META_FILE: &str = "grumpy.json";
const FORMAT_VERSION: u32 = 1;

#[derive(Clone)]
pub struct DatasetHandle {
    pub path: String,
    pub store: ReadableWritableListableStorage,
    pub meta: FileMeta,
}

impl DatasetHandle {
    pub fn open(path: &str) -> PyResult<Self> {
        let meta = read_meta(path)?;
        let store = store_fs(path)?;
        Ok(Self {
            path: path.to_string(),
            store,
            meta,
        })
    }

    pub fn axis0_len(&self) -> PyResult<usize> {
        match &self.meta.root {
            RootMeta::Array { layout, .. } => axis0_len_from_layout_meta(&self.store, layout),
            RootMeta::DataFrame { columns, .. } => {
                if columns.is_empty() {
                    return Ok(0);
                }
                let mut n = 0usize;
                for c in columns {
                    n = n.max(axis0_len_from_layout_meta(&self.store, &c.layout)?);
                }
                Ok(n)
            }
        }
    }

    pub fn primary_layout_meta(&self) -> PyResult<&LayoutMeta> {
        match &self.meta.root {
            RootMeta::Array { layout, .. } => Ok(layout),
            RootMeta::DataFrame { columns, .. } => columns
                .first()
                .map(|c| &c.layout)
                .ok_or_else(|| PyValueError::new_err("Empty dataframe has no layout.")),
        }
    }

    pub fn schema(&self) -> Option<SchemaRef> {
        match &self.meta.root {
            RootMeta::DataFrame { schema, .. } => schema.as_ref().map(|levels| SchemaRef {
                levels: levels.clone(),
            }),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SchemaRef {
    pub levels: Vec<Vec<String>>,
}

impl SchemaRef {
    pub fn level_index(&self, name: &str) -> PyResult<usize> {
        for (lvl, names) in self.levels.iter().enumerate() {
            if names.iter().any(|n| n == name) {
                return Ok(lvl);
            }
        }
        Err(PyValueError::new_err(format!(
            "Unknown schema level '{name}'."
        )))
    }
}

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
    /// When set, only leaf/offset buffers at this list depth use ``chunk_size``; others use one chunk.
    chunk_dim_depth: Option<usize>,
}

impl SaveCtx {
    fn effective_chunk_size(&self, depth: usize, n: usize) -> usize {
        let n = n.max(1);
        match self.chunk_dim_depth {
            Some(target) if target == depth => self.chunk_size.max(1).min(n),
            Some(_) => n,
            None => self.chunk_size.max(1).min(n),
        }
    }
}

pub fn save_array(
    py: Python<'_>,
    arr: &GrumpyArray,
    path: &str,
    chunk_size: usize,
    chunk_dim: Option<usize>,
) -> PyResult<()> {
    let _ = py;
    ensure_dir(path)?;
    let store = store_fs(path)?;
    init_root_group(&store)?;
    init_group(&store, "/buffers")?;
    let mut ctx = SaveCtx {
        next: 0,
        chunk_size: chunk_size.max(1),
        chunk_dim_depth: chunk_dim,
    };
    let layout = save_layout(&store, &mut ctx, arr.dtype, &arr.layout, 0)?;
    let meta = FileMeta {
        version: FORMAT_VERSION,
        root: RootMeta::Array {
            dtype: arr.dtype.into(),
            layout,
        },
    };
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

pub fn save_dataframe(
    py: Python<'_>,
    df: &GrumpyDataFrame,
    path: &str,
    chunk_size: usize,
    chunk_dim: Option<usize>,
) -> PyResult<()> {
    let _ = py;
    ensure_dir(path)?;
    let store = store_fs(path)?;
    init_root_group(&store)?;
    init_group(&store, "/buffers")?;
    let mut ctx = SaveCtx {
        next: 0,
        chunk_size: chunk_size.max(1),
        chunk_dim_depth: chunk_dim,
    };
    let mut columns: Vec<ColumnMeta> = Vec::new();
    for (name, col) in df.names.iter().zip(df.cols.iter()) {
        let layout = save_layout(&store, &mut ctx, col.dtype, &col.layout, 0)?;
        columns.push(ColumnMeta {
            name: name.clone(),
            dtype: col.dtype.into(),
            layout,
        });
    }
    let schema_levels = df.schema.as_ref().map(|s| s.levels.clone());
    let meta = FileMeta {
        version: FORMAT_VERSION,
        root: RootMeta::DataFrame {
            schema: schema_levels,
            columns,
        },
    };
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

/// Return axis-0 length from on-disk metadata and offset buffers without loading leaf data.
pub fn stored_axis0_len(path: &str) -> PyResult<usize> {
    let meta = read_meta(path)?;
    let store = store_fs(path)?;
    match meta.root {
        RootMeta::Array { layout, .. } => axis0_len_from_layout_meta(&store, &layout),
        RootMeta::DataFrame { columns, .. } => {
            if columns.is_empty() {
                return Ok(0);
            }
            let mut n = 0usize;
            for c in &columns {
                n = n.max(axis0_len_from_layout_meta(&store, &c.layout)?);
            }
            Ok(n)
        }
    }
}

fn axis0_len_from_layout_meta(
    store: &ReadableWritableListableStorage,
    meta: &LayoutMeta,
) -> PyResult<usize> {
    match meta {
        LayoutMeta::ListOffset { offsets, .. } => {
            let offs = read_vec::<i64>(store, offsets)?;
            Ok(offs.len().saturating_sub(1))
        }
        LayoutMeta::OffsetView { start, stop, .. } => Ok(stop.saturating_sub(*start)),
        LayoutMeta::Indexed { index, .. } => {
            let idx = read_vec::<i64>(store, index)?;
            Ok(idx.len())
        }
        LayoutMeta::Leaf { len, .. } => Ok(*len),
        LayoutMeta::UnionScalarList { tags, .. } => {
            let tags = read_vec::<u8>(store, tags)?;
            Ok(tags.len())
        }
    }
}

fn save_layout(
    store: &ReadableWritableListableStorage,
    ctx: &mut SaveCtx,
    dt: DType,
    layout: &Layout,
    depth: usize,
) -> PyResult<LayoutMeta> {
    match layout {
        Layout::Leaf(leaf) => save_leaf(store, ctx, dt, leaf, depth),
        Layout::ListOffset(lo) => {
            let offsets = write_vec_i64(store, ctx, "offsets", lo.offsets.as_slice(), depth)?;
            let content = save_layout(store, ctx, dt, lo.content.as_ref(), depth + 1)?;
            Ok(LayoutMeta::ListOffset {
                offsets,
                content: Box::new(content),
            })
        }
        Layout::OffsetView(v) => {
            let offsets = write_vec_i64(store, ctx, "offsets", v.offsets.as_slice(), depth)?;
            let content = save_layout(store, ctx, dt, v.content.as_ref(), depth)?;
            Ok(LayoutMeta::OffsetView {
                offsets,
                start: v.start,
                stop: v.stop,
                content: Box::new(content),
            })
        }
        Layout::Indexed(ix) => {
            let index = write_vec_i64(store, ctx, "index", ix.index.as_slice(), depth)?;
            let content = save_layout(store, ctx, dt, ix.content.as_ref(), depth)?;
            Ok(LayoutMeta::Indexed {
                index,
                content: Box::new(content),
            })
        }
        Layout::UnionScalarList(u) => {
            let tags = write_vec_u8(store, ctx, "tags", &u.tags, depth)?;
            let index = write_vec_i64(store, ctx, "index", &u.index, depth)?;
            let scalars = save_leaf(store, ctx, dt, &u.scalars, depth)?;
            let lists = save_layout(
                store,
                ctx,
                dt,
                &Layout::ListOffset(u.lists.clone()),
                depth,
            )?;
            Ok(LayoutMeta::UnionScalarList {
                tags,
                index,
                scalars: Box::new(scalars),
                lists: Box::new(lists),
            })
        }
    }
}

fn save_leaf(
    store: &ReadableWritableListableStorage,
    ctx: &mut SaveCtx,
    dt: DType,
    leaf: &Leaf,
    depth: usize,
) -> PyResult<LayoutMeta> {
    let len = leaf.len;
    let validity_vec: Vec<bool> = leaf.validity.iter().by_vals().collect();
    let validity = write_vec_bool(store, ctx, "validity", validity_vec.as_slice(), depth)?;
    let values = match (&leaf.buffer, dt) {
        (LeafBuffer::I8(v), DType::Int8) => write_vec_i8(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::I16(v), DType::Int16) => write_vec_i16(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::I32(v), DType::Int32) => write_vec_i32(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::I64(v), DType::Int64) => write_vec_i64(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::U8(v), DType::UInt8) => write_vec_u8(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::U16(v), DType::UInt16) => write_vec_u16(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::U32(v), DType::UInt32) => write_vec_u32(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::U64(v), DType::UInt64) => write_vec_u64(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::F16(v), DType::Float16) => write_vec_u16(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::F32(v), DType::Float32) => write_vec_f32(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::F64(v), DType::Float64) => write_vec_f64(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::Bool(v), DType::Bool) => {
            let b: Vec<bool> = v.as_slice().iter().map(|&x| x != 0).collect();
            write_vec_bool(store, ctx, "values", &b, depth)?
        }
        (LeafBuffer::Char(v), DType::Char) => write_vec_u32(store, ctx, "values", v.as_slice(), depth)?,
        (LeafBuffer::String(v), DType::String) => write_vec_string(store, ctx, "values", v.as_slice(), depth)?,
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
    depth: usize,
) -> PyResult<String>
where
    T: Clone,
{
    let path = next_path(ctx, prefix);
    let n = data.len();
    let chunk = ctx.effective_chunk_size(depth, n);
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

fn write_vec_i64(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[i64], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Int64, FillValue::from(0i64), data, depth)
}
fn write_vec_i32(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[i32], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Int32, FillValue::from(0i32), data, depth)
}
fn write_vec_i16(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[i16], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Int16, FillValue::from(0i16), data, depth)
}
fn write_vec_i8(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[i8], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Int8, FillValue::from(0i8), data, depth)
}
fn write_vec_u64(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[u64], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::UInt64, FillValue::from(0u64), data, depth)
}
fn write_vec_u32(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[u32], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::UInt32, FillValue::from(0u32), data, depth)
}
fn write_vec_u16(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[u16], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::UInt16, FillValue::from(0u16), data, depth)
}
fn write_vec_u8(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[u8], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::UInt8, FillValue::from(0u8), data, depth)
}
fn write_vec_f64(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[f64], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Float64, FillValue::from(0.0f64), data, depth)
}
fn write_vec_f32(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[f32], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Float32, FillValue::from(0.0f32), data, depth)
}
fn write_vec_bool(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[bool], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::Bool, FillValue::from(false), data, depth)
}
fn write_vec_string(store: &ReadableWritableListableStorage, ctx: &mut SaveCtx, prefix: &str, data: &[String], depth: usize) -> PyResult<String> {
    write_1d(store, ctx, prefix, DataType::String, FillValue::from(""), data, depth)
}

fn read_vec<T: ElementOwned>(store: &ReadableWritableListableStorage, path: &str) -> PyResult<Vec<T>> {
    let arr = Array::open(store.clone(), path).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let subset = arr.subset_all();
    arr.retrieve_array_subset_elements::<T>(&subset)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

fn read_vec_bool(store: &ReadableWritableListableStorage, path: &str) -> PyResult<Vec<bool>> {
    read_vec::<bool>(store, path)
}

fn read_vec_range<T: ElementOwned>(
    store: &ReadableWritableListableStorage,
    path: &str,
    start: usize,
    stop: usize,
) -> PyResult<Vec<T>> {
    if start >= stop {
        return Ok(Vec::new());
    }
    let arr = Array::open(store.clone(), path).map_err(|e| PyValueError::new_err(format!("{e}")))?;
    let subset = ArraySubset::new_with_ranges(&[start as u64..stop as u64]);
    record_io_bytes((stop - start) * std::mem::size_of::<T>());
    arr.retrieve_array_subset_elements::<T>(&subset)
        .map_err(|e| PyValueError::new_err(format!("{e}")))
}

/// Load an array batch covering axis-0 ``[start, stop)`` without reading unrelated leaf data.
pub fn load_array_axis0_slice(
    handle: &DatasetHandle,
    start: usize,
    stop: usize,
) -> PyResult<GrumpyArray> {
    match &handle.meta.root {
        RootMeta::Array { dtype, layout } => {
            let dt: DType = dtype.clone().into();
            let layout = load_layout_axis0_slice(&handle.store, dt, layout, start, stop)?;
            Ok(GrumpyArray { dtype: dt, layout })
        }
        _ => Err(PyValueError::new_err("Path does not contain a saved GrumpyArray.")),
    }
}

/// Load a dataframe batch covering axis-0 ``[start, stop)`` without reading unrelated leaf data.
pub fn load_dataframe_axis0_slice(
    handle: &DatasetHandle,
    start: usize,
    stop: usize,
) -> PyResult<GrumpyDataFrame> {
    match &handle.meta.root.clone() {
        RootMeta::DataFrame { schema, columns } => {
            let mut names: Vec<String> = Vec::new();
            let mut cols: Vec<GrumpyArray> = Vec::new();
            for c in columns {
                let dt: DType = c.dtype.clone().into();
                let layout = load_layout_axis0_slice(&handle.store, dt, &c.layout, start, stop)?;
                names.push(c.name.clone());
                cols.push(GrumpyArray { dtype: dt, layout });
            }
            let schema = match schema {
                None => None,
                Some(levels) => {
                    let mut name_to_level = std::collections::HashMap::new();
                    for (lvl, names) in levels.iter().enumerate() {
                        for n in names {
                            name_to_level.insert(n.clone(), lvl);
                        }
                    }
                    Some(Schema { levels: levels.clone(), name_to_level })
                }
            };
            GrumpyDataFrame::new(names, cols, schema)
        }
        _ => Err(PyValueError::new_err("Path does not contain a saved GrumpyDataFrame.")),
    }
}

fn load_layout_axis0_slice(
    store: &ReadableWritableListableStorage,
    dt: DType,
    meta: &LayoutMeta,
    start: usize,
    stop: usize,
) -> PyResult<Layout> {
    load_layout_take_range(store, dt, meta, start, stop)
}

/// Disk-backed analogue of in-memory ``take_range`` (partial leaf reads only).
fn load_layout_take_range(
    store: &ReadableWritableListableStorage,
    dt: DType,
    meta: &LayoutMeta,
    start: usize,
    end: usize,
) -> PyResult<Layout> {
    if start > end {
        return Err(PyValueError::new_err("Invalid range."));
    }
    match meta {
        LayoutMeta::Leaf { len, values, validity } => {
            if end > *len {
                return Err(PyValueError::new_err("Leaf slice out of bounds."));
            }
            let valid = read_vec_range::<bool>(store, validity, start, end)?;
            let new_len = end - start;
            if valid.len() != new_len {
                return Err(PyValueError::new_err("Invalid validity length in file."));
            }
            let mut leaf = Leaf::new(dt);
            leaf.len = new_len;
            leaf.has_nulls = valid.iter().any(|b| !*b);
            leaf.validity = Arc::new(bitvec::vec::BitVec::<u8, bitvec::order::Lsb0>::from_iter(
                valid.iter().copied(),
            ));
            leaf.buffer = match dt {
                DType::Int8 => LeafBuffer::I8(Arc::new(read_vec_range::<i8>(store, values, start, end)?)),
                DType::Int16 => LeafBuffer::I16(Arc::new(read_vec_range::<i16>(store, values, start, end)?)),
                DType::Int32 => LeafBuffer::I32(Arc::new(read_vec_range::<i32>(store, values, start, end)?)),
                DType::Int64 => LeafBuffer::I64(Arc::new(read_vec_range::<i64>(store, values, start, end)?)),
                DType::UInt8 => LeafBuffer::U8(Arc::new(read_vec_range::<u8>(store, values, start, end)?)),
                DType::UInt16 => LeafBuffer::U16(Arc::new(read_vec_range::<u16>(store, values, start, end)?)),
                DType::UInt32 => LeafBuffer::U32(Arc::new(read_vec_range::<u32>(store, values, start, end)?)),
                DType::UInt64 => LeafBuffer::U64(Arc::new(read_vec_range::<u64>(store, values, start, end)?)),
                DType::Float16 => LeafBuffer::F16(Arc::new(read_vec_range::<u16>(store, values, start, end)?)),
                DType::Float32 => LeafBuffer::F32(Arc::new(read_vec_range::<f32>(store, values, start, end)?)),
                DType::Float64 => LeafBuffer::F64(Arc::new(read_vec_range::<f64>(store, values, start, end)?)),
                DType::Bool => {
                    let b = read_vec_range::<bool>(store, values, start, end)?;
                    LeafBuffer::Bool(Arc::new(b.into_iter().map(|x| if x { 1 } else { 0 }).collect()))
                }
                DType::Char => LeafBuffer::Char(Arc::new(read_vec_range::<u32>(store, values, start, end)?)),
                DType::String => LeafBuffer::String(Arc::new(read_vec_range::<String>(store, values, start, end)?)),
            };
            Ok(Layout::Leaf(leaf))
        }
        LayoutMeta::ListOffset { offsets, content } => {
            let offs = read_vec::<i64>(store, offsets)?;
            if end > offs.len().saturating_sub(1) {
                return Err(PyValueError::new_err("Slice out of bounds."));
            }
            let child_start = offs[start] as usize;
            let child_end = offs[end] as usize;
            let mut new_offs: Vec<i64> = Vec::with_capacity(end - start + 1);
            new_offs.push(0i64);
            let mut acc = 0i64;
            for i in start..end {
                acc += offs[i + 1] - offs[i];
                new_offs.push(acc);
            }
            let inner = load_layout_take_range(store, dt, content, child_start, child_end)?;
            Ok(Layout::ListOffset(ListOffset {
                offsets: Arc::new(new_offs),
                content: Box::new(inner),
            }))
        }
        LayoutMeta::OffsetView {
            offsets,
            start: base,
            stop: base_stop,
            content,
        } => {
            let abs_start = base + start;
            let abs_end = base + end;
            if abs_end > *base_stop {
                return Err(PyValueError::new_err("Slice out of bounds."));
            }
            let offs = read_vec::<i64>(store, offsets)?;
            let child_start = offs[abs_start] as usize;
            let child_end = offs[abs_end] as usize;
            let inner = load_layout_take_range(store, dt, content, child_start, child_end)?;
            Ok(Layout::OffsetView(OffsetView {
                offsets: Arc::new(offs),
                start: abs_start,
                stop: abs_end,
                content: Box::new(inner),
            }))
        }
        LayoutMeta::Indexed { .. } => Err(PyValueError::new_err(
            "Indexed layout streaming slice is not supported.",
        )),
        LayoutMeta::UnionScalarList { .. } => Err(PyValueError::new_err(
            "UnionScalarList streaming slice is not supported.",
        )),
    }
}

/// Count entities at ``target_depth`` within each axis-0 row (reads offset buffers only).
pub fn row_entity_counts_at_depth(
    store: &ReadableWritableListableStorage,
    meta: &LayoutMeta,
    target_depth: usize,
) -> PyResult<Vec<usize>> {
    let n = axis0_len_from_layout_meta(store, meta)?;
    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        out.push(count_entities_in_axis0_row(store, meta, row, target_depth, 0)?);
    }
    Ok(out)
}

fn count_entities_in_axis0_row(
    store: &ReadableWritableListableStorage,
    meta: &LayoutMeta,
    row: usize,
    target_depth: usize,
    current_depth: usize,
) -> PyResult<usize> {
    match meta {
        LayoutMeta::ListOffset { offsets, content } => {
            let offs = read_vec::<i64>(store, offsets)?;
            if row + 1 >= offs.len() {
                return Err(PyValueError::new_err("Row index out of bounds."));
            }
            let leaf_lo = offs[row] as usize;
            let leaf_hi = offs[row + 1] as usize;
            entity_count_in_leaf_range(
                store,
                content,
                leaf_lo,
                leaf_hi,
                target_depth,
                current_depth + 1,
            )
        }
        LayoutMeta::OffsetView {
            offsets,
            start,
            stop: _,
            content,
        } => {
            let offs = read_vec::<i64>(store, offsets)?;
            let abs_row = start + row;
            if abs_row + 1 >= offs.len() {
                return Err(PyValueError::new_err("Row index out of bounds."));
            }
            let leaf_lo = offs[abs_row] as usize;
            let leaf_hi = offs[abs_row + 1] as usize;
            entity_count_in_leaf_range(
                store,
                content,
                leaf_lo,
                leaf_hi,
                target_depth,
                current_depth + 1,
            )
        }
        LayoutMeta::Leaf { .. } => Ok(if target_depth == current_depth {
            1
        } else {
            0
        }),
        LayoutMeta::Indexed { .. } | LayoutMeta::UnionScalarList { .. } => Err(PyValueError::new_err(
            "batch_on is not supported for Indexed/UnionScalarList layouts.",
        )),
    }
}

fn entity_count_in_leaf_range(
    store: &ReadableWritableListableStorage,
    meta: &LayoutMeta,
    leaf_lo: usize,
    leaf_hi: usize,
    target_depth: usize,
    current_depth: usize,
) -> PyResult<usize> {
    if current_depth == target_depth {
        return Ok(entity_count_at_depth(store, meta, leaf_lo, leaf_hi));
    }
    match meta {
        LayoutMeta::ListOffset { offsets, content } => {
            let offs = read_vec::<i64>(store, offsets)?;
            let n = offs.len().saturating_sub(1);
            let mut total = 0usize;
            for k in 0..n {
                if (offs[k + 1] as usize) <= leaf_lo {
                    continue;
                }
                if (offs[k] as usize) >= leaf_hi {
                    break;
                }
                let sub_lo = std::cmp::max(offs[k] as usize, leaf_lo);
                let sub_hi = std::cmp::min(offs[k + 1] as usize, leaf_hi);
                total += entity_count_in_leaf_range(
                    store,
                    content,
                    sub_lo,
                    sub_hi,
                    target_depth,
                    current_depth + 1,
                )?;
            }
            Ok(total)
        }
        LayoutMeta::Leaf { .. } => Ok(0),
        LayoutMeta::OffsetView { .. }
        | LayoutMeta::Indexed { .. }
        | LayoutMeta::UnionScalarList { .. } => Err(PyValueError::new_err(
            "batch_on depth counting unsupported for this layout node.",
        )),
    }
}

fn entity_count_at_depth(
    store: &ReadableWritableListableStorage,
    meta: &LayoutMeta,
    leaf_lo: usize,
    leaf_hi: usize,
) -> usize {
    match meta {
        LayoutMeta::ListOffset { offsets, .. } => {
            count_list_elements_in_leaf_range(store, meta, leaf_lo, leaf_hi)
        }
        LayoutMeta::Leaf { .. } => leaf_hi.saturating_sub(leaf_lo),
        _ => 0,
    }
}

fn count_entities_in_leaf_range(
    store: &ReadableWritableListableStorage,
    meta: &LayoutMeta,
    leaf_lo: usize,
    leaf_hi: usize,
    target_depth: usize,
    current_depth: usize,
) -> PyResult<usize> {
    entity_count_in_leaf_range(
        store,
        meta,
        leaf_lo,
        leaf_hi,
        target_depth,
        current_depth,
    )
}

fn count_list_elements_in_leaf_range(
    store: &ReadableWritableListableStorage,
    meta: &LayoutMeta,
    leaf_lo: usize,
    leaf_hi: usize,
) -> usize {
    match meta {
        LayoutMeta::ListOffset { offsets, .. } => {
            let offs = read_vec::<i64>(store, offsets).unwrap_or_default();
            let n = offs.len().saturating_sub(1);
            let mut count = 0usize;
            for k in 0..n {
                if (offs[k + 1] as usize) <= leaf_lo {
                    continue;
                }
                if (offs[k] as usize) >= leaf_hi {
                    break;
                }
                count += 1;
            }
            count
        }
        LayoutMeta::Leaf { .. } => 1,
        _ => 0,
    }
}

/// Append axis-0 rows to an existing saved array (load + concat + rewrite).
pub fn append_array_axis0(
    py: Python<'_>,
    path: &str,
    batch: &GrumpyArray,
    chunk_size: usize,
    chunk_dim: Option<usize>,
) -> PyResult<()> {
    let existing = load_array(py, path)?;
    let merged = crate::ops::concat_arrays_axis0(&existing, batch)?;
    save_array(py, &merged, path, chunk_size, chunk_dim)
}

/// Append axis-0 rows to an existing saved dataframe (load + concat + rewrite).
pub fn append_dataframe_axis0(
    py: Python<'_>,
    path: &str,
    batch: &GrumpyDataFrame,
    chunk_size: usize,
    chunk_dim: Option<usize>,
) -> PyResult<()> {
    let existing = load_dataframe(py, path)?;
    let merged = existing.concat_axis0(batch)?;
    save_dataframe(py, &merged, path, chunk_size, chunk_dim)
}

/// Resolve ``chunk_dim`` from an integer depth or schema level name.
pub fn resolve_chunk_dim_depth(df_schema: Option<&Schema>, chunk_dim: &str) -> PyResult<usize> {
    if let Ok(d) = chunk_dim.parse::<usize>() {
        return Ok(d);
    }
    if let Some(schema) = df_schema {
        for (lvl, names) in schema.levels.iter().enumerate() {
            if names.iter().any(|n| n == chunk_dim) {
                return Ok(lvl);
            }
        }
    }
    Err(PyValueError::new_err(format!(
        "Unknown chunk_dim '{chunk_dim}'."
    )))
}
