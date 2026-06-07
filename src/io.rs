use crate::dataframe::{CanonShape, GrumpyDataFrame, Schema};
use crate::dtype::DType;
use crate::error::{arg_invalid, index_out_of_bounds, internal, internal_dtype_buffer_mismatch, invalid_slice_range, io_failed, io_wrong_type, schema_violation, unsupported};
use crate::io_cache::{IoCache, IoReader};
use crate::layout::{concat_layout_segments, remap_union_pick, take_range, GrumpyArray, Layout, Leaf, LeafBuffer, ListOffset, OffsetView, UnionScalarList};
use bitvec::bitvec;
use bitvec::order::Lsb0;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};
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

pub(crate) fn record_io_bytes(n: usize) {
    IO_BYTES_READ.fetch_add(n, Ordering::Relaxed);
}

const META_FILE: &str = "grumpy.json";
const FORMAT_VERSION: u32 = 1;

static PATH_IO_CACHES: LazyLock<Mutex<HashMap<String, Arc<IoCache>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

static PATH_RESIDENT: LazyLock<Mutex<HashMap<String, Arc<ResidentDataset>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn path_io_cache(path: &str) -> Arc<IoCache> {
    let mut caches = PATH_IO_CACHES.lock().unwrap();
    if let Some(c) = caches.get(path) {
        return Arc::clone(c);
    }
    let c = Arc::new(IoCache::default());
    caches.insert(path.to_string(), Arc::clone(&c));
    c
}

fn path_resident(
    path: &str,
    store: &ReadableWritableListableStorage,
    cache: &IoCache,
    meta: &FileMeta,
) -> PyResult<Arc<ResidentDataset>> {
    let mut residents = PATH_RESIDENT.lock().unwrap();
    if let Some(r) = residents.get(path) {
        return Ok(Arc::clone(r));
    }
    let r = Arc::new(load_resident(store, cache, meta)?);
    residents.insert(path.to_string(), Arc::clone(&r));
    Ok(r)
}

/// Clear path-persistent I/O and resident caches (for tests).
pub fn clear_path_caches() {
    PATH_IO_CACHES.lock().unwrap().clear();
    PATH_RESIDENT.lock().unwrap().clear();
}

/// Drop path-scoped I/O and resident cache entries (called by :meth:`OpenDataFrame.close`).
pub fn release_path_resources(path: &str) {
    PATH_IO_CACHES.lock().unwrap().remove(path);
    PATH_RESIDENT.lock().unwrap().remove(path);
}

struct OpenSessionInner {
    handle: DatasetHandle,
    closed: AtomicBool,
}

/// Shared open-handle state for :class:`OpenDataFrame` and derived :class:`OpenColumn` proxies.
#[derive(Clone)]
pub struct OpenSession {
    inner: Arc<OpenSessionInner>,
}

impl OpenSession {
    pub fn new(handle: DatasetHandle) -> Self {
        Self {
            inner: Arc::new(OpenSessionInner {
                handle,
                closed: AtomicBool::new(false),
            }),
        }
    }

    pub fn path(&self) -> &str {
        &self.inner.handle.path
    }

    pub fn is_closed(&self) -> bool {
        self.inner.closed.load(Ordering::Acquire)
    }

    pub fn handle(&self) -> PyResult<&DatasetHandle> {
        if self.is_closed() {
            return Err(crate::error::io_closed(self.path()));
        }
        Ok(&self.inner.handle)
    }

    pub fn close(&self) {
        if self.inner.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        release_path_resources(self.path());
    }
}

#[derive(Clone)]
pub enum ResidentDataset {
    Array(GrumpyArray),
    DataFrame(GrumpyDataFrame),
}

#[derive(Clone)]
pub struct DatasetHandle {
    pub path: String,
    pub store: ReadableWritableListableStorage,
    pub meta: FileMeta,
    pub cache: Arc<IoCache>,
    pub resident: Option<Arc<ResidentDataset>>,
}

impl DatasetHandle {
    pub fn open(path: &str) -> PyResult<Self> {
        Self::open_with_mode(path, false)
    }

    pub fn open_with_mode(path: &str, in_memory: bool) -> PyResult<Self> {
        let meta = read_meta(path)?;
        let store = store_fs(path)?;
        let cache = path_io_cache(path);
        let resident = if in_memory {
            Some(path_resident(path, &store, cache.as_ref(), &meta)?)
        } else {
            None
        };
        Ok(Self {
            path: path.to_string(),
            store,
            meta,
            cache,
            resident,
        })
    }

    pub fn reader(&self) -> IoReader<'_> {
        IoReader {
            store: &self.store,
            cache: &self.cache,
        }
    }

    pub fn axis0_len(&self) -> PyResult<usize> {
        let io = self.reader();
        match &self.meta.root {
            RootMeta::Array { layout, .. } => axis0_len_from_layout_meta(&io, layout),
            RootMeta::DataFrame { columns, .. } => {
                if columns.is_empty() {
                    return Ok(0);
                }
                let mut n = 0usize;
                for c in columns {
                    n = n.max(axis0_len_from_layout_meta(&io, &c.layout)?);
                }
                Ok(n)
            }
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
        Err(schema_violation(
            format!("unknown schema level '{name}'"),
            "batch_on/chunk_dim must name a declared schema level.",
            "use a level from schema= or a numeric depth for arrays.",
        ))
    }
}

/// Canonical nested shape persisted in ``grumpy.json`` (no reference column required).
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct CanonMeta {
    pub nrows: Option<usize>,
    pub offsets: Vec<Option<Vec<i64>>>,
}

impl From<&CanonShape> for CanonMeta {
    fn from(c: &CanonShape) -> Self {
        Self {
            nrows: c.nrows,
            offsets: c.offsets.clone(),
        }
    }
}

impl From<CanonMeta> for CanonShape {
    fn from(m: CanonMeta) -> Self {
        Self {
            nrows: m.nrows,
            offsets: m.offsets,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum RootMeta {
    #[serde(rename = "array")]
    Array { dtype: DTypeSer, layout: LayoutMeta },
    #[serde(rename = "dataframe")]
    DataFrame {
        schema: Option<Vec<Vec<String>>>,
        columns: Vec<ColumnMeta>,
        #[serde(default)]
        canon: Option<CanonMeta>,
    },
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
    /// Dedup identical offset buffers (shared canonical offsets for dataframes).
    offset_dedup: std::collections::HashMap<Vec<i64>, String>,
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
        offset_dedup: std::collections::HashMap::new(),
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
            let cache = IoCache::default();
            let io = IoReader {
                store: &store,
                cache: &cache,
            };
            let dt: DType = dtype.into();
            let layout = load_layout(&io, dt, &layout)?;
            Ok(GrumpyArray { dtype: dt, layout })
        }
        _ => Err(io_wrong_type("array", path)),
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
        offset_dedup: std::collections::HashMap::new(),
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
            canon: Some(CanonMeta::from(&df.canon)),
        },
    };
    write_meta(path, &meta)?;
    Ok(())
}

pub fn load_dataframe(py: Python<'_>, path: &str) -> PyResult<GrumpyDataFrame> {
    let _ = py;
    let meta = read_meta(path)?;
    match meta.root {
        RootMeta::DataFrame {
            schema,
            columns,
            canon,
        } => {
            let store = store_fs(path)?;
            let cache = IoCache::default();
            let io = IoReader {
                store: &store,
                cache: &cache,
            };
            let mut names: Vec<String> = Vec::new();
            let mut cols: Vec<GrumpyArray> = Vec::new();
            for c in columns {
                let dt: DType = c.dtype.into();
                let layout = load_layout(&io, dt, &c.layout)?;
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
            let stored_canon = canon.map(CanonShape::from);
            GrumpyDataFrame::from_loaded(names, cols, schema, stored_canon)
        }
        _ => Err(io_wrong_type("dataframe", path)),
    }
}

fn load_resident(
    store: &ReadableWritableListableStorage,
    cache: &IoCache,
    meta: &FileMeta,
) -> PyResult<ResidentDataset> {
    let io = IoReader { store, cache };
    match &meta.root {
        RootMeta::Array { dtype, layout } => {
            let dt: DType = dtype.clone().into();
            Ok(ResidentDataset::Array(GrumpyArray {
                dtype: dt,
                layout: load_layout(&io, dt, layout)?,
            }))
        }
        RootMeta::DataFrame {
            schema,
            columns,
            canon,
        } => {
            let mut names: Vec<String> = Vec::new();
            let mut cols: Vec<GrumpyArray> = Vec::new();
            for c in columns {
                let dt: DType = c.dtype.clone().into();
                let layout = load_layout(&io, dt, &c.layout)?;
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
                    Some(Schema {
                        levels: levels.clone(),
                        name_to_level,
                    })
                }
            };
            let stored_canon = canon.clone().map(CanonShape::from);
            Ok(ResidentDataset::DataFrame(GrumpyDataFrame::from_loaded(
                names,
                cols,
                schema,
                stored_canon,
            )?))
        }
    }
}

/// Return axis-0 length from on-disk metadata and offset buffers without loading leaf data.
pub fn stored_axis0_len(path: &str) -> PyResult<usize> {
    let meta = read_meta(path)?;
    let store = store_fs(path)?;
    let cache = IoCache::default();
    let io = IoReader {
        store: &store,
        cache: &cache,
    };
    match meta.root {
        RootMeta::Array { layout, .. } => axis0_len_from_layout_meta(&io, &layout),
        RootMeta::DataFrame { columns, .. } => {
            if columns.is_empty() {
                return Ok(0);
            }
            let mut n = 0usize;
            for c in &columns {
                n = n.max(axis0_len_from_layout_meta(&io, &c.layout)?);
            }
            Ok(n)
        }
    }
}

fn axis0_len_from_layout_meta(io: &IoReader<'_>, meta: &LayoutMeta) -> PyResult<usize> {
    match meta {
        LayoutMeta::ListOffset { offsets, .. } => {
            let offs = io.cache.read_i64(io.store, offsets)?;
            Ok(offs.len().saturating_sub(1))
        }
        LayoutMeta::OffsetView { start, stop, .. } => Ok(stop.saturating_sub(*start)),
        LayoutMeta::Indexed { index, .. } => {
            let idx = io.cache.read_i64(io.store, index)?;
            Ok(idx.len())
        }
        LayoutMeta::Leaf { len, .. } => Ok(*len),
        LayoutMeta::UnionScalarList { tags, .. } => {
            let tags = io.cache.read_u8(io.store, tags)?;
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
            let offsets = write_offsets_dedup(store, ctx, lo.offsets.as_slice(), depth)?;
            let content = save_layout(store, ctx, dt, lo.content.as_ref(), depth + 1)?;
            Ok(LayoutMeta::ListOffset {
                offsets,
                content: Box::new(content),
            })
        }
        Layout::OffsetView(v) => {
            let offsets = write_offsets_dedup(store, ctx, v.offsets.as_slice(), depth)?;
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
        _ => return Err(internal_dtype_buffer_mismatch("save_leaf", dt)),
    };
    Ok(LayoutMeta::Leaf { len, values, validity })
}

fn write_offsets_dedup(
    store: &ReadableWritableListableStorage,
    ctx: &mut SaveCtx,
    data: &[i64],
    depth: usize,
) -> PyResult<String> {
    let key = data.to_vec();
    if let Some(path) = ctx.offset_dedup.get(&key) {
        return Ok(path.clone());
    }
    let path = write_vec_i64(store, ctx, "offsets", data, depth)?;
    ctx.offset_dedup.insert(key, path.clone());
    Ok(path)
}

fn load_layout(io: &IoReader<'_>, dt: DType, meta: &LayoutMeta) -> PyResult<Layout> {
    match meta {
        LayoutMeta::Leaf { len, values, validity } => {
            let valid = io.cache.read_bool(io.store, validity)?;
            if valid.len() != *len {
                return Err(internal("load_layout_take_range", "validity length mismatch in file"));
            }
            let mut leaf = Leaf::new(dt);
            leaf.len = *len;
            leaf.has_nulls = valid.iter().any(|b| !*b);
            leaf.validity = Arc::new(bitvec::vec::BitVec::<u8, bitvec::order::Lsb0>::from_iter(valid.iter().copied()));
            leaf.buffer = match dt {
                DType::Int8 => LeafBuffer::I8(io.cache.read_i8_arc(io.store, values)?),
                DType::Int16 => LeafBuffer::I16(io.cache.read_i16_arc(io.store, values)?),
                DType::Int32 => LeafBuffer::I32(io.cache.read_i32_arc(io.store, values)?),
                DType::Int64 => LeafBuffer::I64(io.cache.read_i64_arc(io.store, values)?),
                DType::UInt8 => LeafBuffer::U8(io.cache.read_u8_arc(io.store, values)?),
                DType::UInt16 => LeafBuffer::U16(io.cache.read_u16_arc(io.store, values)?),
                DType::UInt32 => LeafBuffer::U32(io.cache.read_u32_arc(io.store, values)?),
                DType::UInt64 => LeafBuffer::U64(io.cache.read_u64_arc(io.store, values)?),
                DType::Float16 => LeafBuffer::F16(io.cache.read_u16_arc(io.store, values)?),
                DType::Float32 => LeafBuffer::F32(io.cache.read_f32_arc(io.store, values)?),
                DType::Float64 => LeafBuffer::F64(io.cache.read_f64_arc(io.store, values)?),
                DType::Bool => {
                    let b = io.cache.read_bool(io.store, values)?;
                    LeafBuffer::Bool(Arc::new(b.into_iter().map(|x| if x { 1 } else { 0 }).collect()))
                }
                DType::Char => LeafBuffer::Char(io.cache.read_u32_arc(io.store, values)?),
                DType::String => LeafBuffer::String(io.cache.read_string_arc(io.store, values)?),
            };
            Ok(Layout::Leaf(leaf))
        }
        LayoutMeta::ListOffset { offsets, content } => {
            let offs = io.cache.read_i64_arc(io.store, offsets)?;
            let content = load_layout(io, dt, content)?;
            Ok(Layout::ListOffset(ListOffset { offsets: offs, content: Box::new(content) }))
        }
        LayoutMeta::OffsetView { offsets, start, stop, content } => {
            let offs = io.cache.read_i64_arc(io.store, offsets)?;
            let content = load_layout(io, dt, content)?;
            Ok(Layout::OffsetView(OffsetView { offsets: offs, start: *start, stop: *stop, content: Box::new(content) }))
        }
        LayoutMeta::Indexed { index, content } => {
            let idx = io.cache.read_i64(io.store, index)?;
            let content = load_layout(io, dt, content)?;
            Ok(Layout::Indexed(crate::layout::Indexed { index: Arc::new(idx), content: Box::new(content) }))
        }
        LayoutMeta::UnionScalarList { tags, index, scalars, lists } => {
            let tags = io.cache.read_u8(io.store, tags)?;
            let index = io.cache.read_i64(io.store, index)?;
            let scal = match load_layout(io, dt, scalars)? {
                Layout::Leaf(l) => l,
                _ => return Err(internal("load_layout", "invalid union scalars layout in file")),
            };
            let lists_layout = load_layout(io, dt, lists)?;
            let lists = match lists_layout {
                Layout::ListOffset(lo) => lo,
                _ => return Err(internal("load_layout", "invalid union lists layout in file")),
            };
            Ok(Layout::UnionScalarList(UnionScalarList { tags, index, scalars: scal, lists }))
        }
    }
}

fn ensure_dir(path: &str) -> PyResult<()> {
    fs::create_dir_all(path).map_err(|e| io_failed(format!("failed to create directory at {path}"), e.to_string(), "check path permissions and disk space."))?;
    Ok(())
}

fn store_fs(path: &str) -> PyResult<ReadableWritableListableStorage> {
    let store: ReadableWritableListableStorage = Arc::new(zarrs::filesystem::FilesystemStore::new(path).map_err(|e| io_failed(format!("failed to open store at {path}"), e.to_string(), "verify the path exists and is readable/writable."))?);
    Ok(store)
}

fn init_root_group(store: &ReadableWritableListableStorage) -> PyResult<()> {
    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?
        .store_metadata()
        .map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    Ok(())
}

fn init_group(store: &ReadableWritableListableStorage, path: &str) -> PyResult<()> {
    zarrs::group::GroupBuilder::new()
        .build(store.clone(), path)
        .map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?
        .store_metadata()
        .map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    Ok(())
}

fn write_meta(path: &str, meta: &FileMeta) -> PyResult<()> {
    let p = Path::new(path).join(META_FILE);
    let s = serde_json::to_string_pretty(meta).map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    fs::write(p, s).map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    Ok(())
}

fn read_meta(path: &str) -> PyResult<FileMeta> {
    let p = Path::new(path).join(META_FILE);
    let s = fs::read_to_string(p).map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    let meta: FileMeta = serde_json::from_str(&s).map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    if meta.version != FORMAT_VERSION {
        return Err(io_failed("unsupported file version", "this grumpy.json version is newer or incompatible", "upgrade grumpy or re-save the dataset with the current version."));
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
        .map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    arr.store_metadata().map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    let subset = arr.subset_all();
    arr.store_array_subset_elements::<T>(&subset, &data.to_vec())
        .map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
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

fn array_axis0_len(store: &ReadableWritableListableStorage, tags_path: &str) -> PyResult<usize> {
    let arr = Array::open(store.clone(), tags_path).map_err(|e| io_failed("I/O operation failed", format!("{e}"), "verify the saved dataset path and file permissions."))?;
    Ok(arr.shape()[0] as usize)
}

/// Load an array batch covering axis-0 ``[start, stop)`` without reading unrelated leaf data.
pub fn load_array_axis0_slice(
    handle: &DatasetHandle,
    start: usize,
    stop: usize,
) -> PyResult<GrumpyArray> {
    if let Some(ResidentDataset::Array(arr)) = handle.resident.as_deref() {
        let dt = arr.dtype;
        return Ok(GrumpyArray {
            dtype: dt,
            layout: take_range(&arr.layout, start, stop)?,
        });
    }
    match &handle.meta.root {
        RootMeta::Array { dtype, layout } => {
            let dt: DType = dtype.clone().into();
            let layout = load_layout_axis0_slice(handle, dt, layout, start, stop)?;
            Ok(GrumpyArray { dtype: dt, layout })
        }
        _ => Err(io_wrong_type("array", &handle.path)),
    }
}

/// Load a dataframe batch covering axis-0 ``[start, stop)`` without reading unrelated leaf data.
pub fn load_dataframe_axis0_slice(
    handle: &DatasetHandle,
    start: usize,
    stop: usize,
) -> PyResult<GrumpyDataFrame> {
    if let Some(ResidentDataset::DataFrame(df)) = handle.resident.as_deref() {
        let mut cols: Vec<GrumpyArray> = Vec::with_capacity(df.cols.len());
        for c in &df.cols {
            cols.push(GrumpyArray {
                dtype: c.dtype,
                layout: take_range(&c.layout, start, stop)?,
            });
        }
        return GrumpyDataFrame::new(df.names.clone(), cols, df.schema.clone());
    }
    match &handle.meta.root {
        RootMeta::DataFrame { schema, columns, .. } => {
            load_dataframe_columns_axis0_slice_inner(handle, None, columns, schema.as_ref(), start, stop)
        }
        _ => Err(io_wrong_type("dataframe", &handle.path)),
    }
}

/// Load selected columns (or all when ``colnames`` is ``None``) for axis-0 ``[start, stop)``.
pub fn load_dataframe_columns_axis0_slice(
    handle: &DatasetHandle,
    colnames: Option<&[String]>,
    start: usize,
    stop: usize,
) -> PyResult<GrumpyDataFrame> {
    if let Some(ResidentDataset::DataFrame(df)) = handle.resident.as_deref() {
        let filter: Option<std::collections::HashSet<&str>> =
            colnames.map(|ns| ns.iter().map(|s| s.as_str()).collect());
        let mut names: Vec<String> = Vec::new();
        let mut cols: Vec<GrumpyArray> = Vec::new();
        for (n, c) in df.names.iter().zip(df.cols.iter()) {
            if filter.as_ref().is_some_and(|f| !f.contains(n.as_str())) {
                continue;
            }
            names.push(n.clone());
            cols.push(GrumpyArray {
                dtype: c.dtype,
                layout: take_range(&c.layout, start, stop)?,
            });
        }
        return GrumpyDataFrame::new(names, cols, df.schema.clone());
    }
    match &handle.meta.root {
        RootMeta::DataFrame { schema, columns, .. } => {
            load_dataframe_columns_axis0_slice_inner(handle, colnames, columns, schema.as_ref(), start, stop)
        }
        _ => Err(io_wrong_type("dataframe", &handle.path)),
    }
}

fn load_dataframe_columns_axis0_slice_inner(
    handle: &DatasetHandle,
    colnames: Option<&[String]>,
    columns: &[ColumnMeta],
    schema_levels: Option<&Vec<Vec<String>>>,
    start: usize,
    stop: usize,
) -> PyResult<GrumpyDataFrame> {
    let filter: Option<std::collections::HashSet<&str>> =
        colnames.map(|ns| ns.iter().map(|s| s.as_str()).collect());
    let mut names: Vec<String> = Vec::new();
    let mut cols: Vec<GrumpyArray> = Vec::new();
    for c in columns {
        if filter.as_ref().is_some_and(|f| !f.contains(c.name.as_str())) {
            continue;
        }
        let dt: DType = c.dtype.clone().into();
        let layout = load_layout_axis0_slice(handle, dt, &c.layout, start, stop)?;
        names.push(c.name.clone());
        cols.push(GrumpyArray { dtype: dt, layout });
    }
    let schema = schema_from_levels(schema_levels);
    GrumpyDataFrame::new(names, cols, schema)
}

fn schema_from_levels(levels: Option<&Vec<Vec<String>>>) -> Option<Schema> {
    levels.map(|levels| {
        let mut name_to_level = std::collections::HashMap::new();
        for (lvl, names) in levels.iter().enumerate() {
            for n in names {
                name_to_level.insert(n.clone(), lvl);
            }
        }
        Schema {
            levels: levels.clone(),
            name_to_level,
        }
    })
}

/// Load one column for axis-0 ``[start, stop)`` without reading other columns.
pub fn load_column_axis0_slice(
    handle: &DatasetHandle,
    col_name: &str,
    start: usize,
    stop: usize,
) -> PyResult<GrumpyArray> {
    if let Some(ResidentDataset::DataFrame(df)) = handle.resident.as_deref() {
        for (n, c) in df.names.iter().zip(df.cols.iter()) {
            if n == col_name {
                return Ok(GrumpyArray {
                    dtype: c.dtype,
                    layout: take_range(&c.layout, start, stop)?,
                });
            }
        }
        return Err(crate::error::unknown_column(col_name));
    }
    match &handle.meta.root {
        RootMeta::DataFrame { columns, .. } => {
            for c in columns {
                if c.name == col_name {
                    let dt: DType = c.dtype.clone().into();
                    let layout = load_layout_axis0_slice(handle, dt, &c.layout, start, stop)?;
                    return Ok(GrumpyArray { dtype: dt, layout });
                }
            }
            Err(crate::error::unknown_column(col_name))
        }
        _ => Err(io_wrong_type("dataframe", &handle.path)),
    }
}

/// Canonical shape for an open handle (from persisted metadata or offset-buffer scan).
pub fn canon_from_handle(handle: &DatasetHandle) -> PyResult<CanonShape> {
    match &handle.meta.root {
        RootMeta::DataFrame {
            canon,
            columns,
            schema,
        } => {
            if let Some(c) = canon {
                return Ok(c.clone().into());
            }
            let nrows = handle.axis0_len()?;
            let nlev = schema.as_ref().map(|s| s.len()).unwrap_or(0);
            let mut offsets: Vec<Option<Vec<i64>>> = vec![None; nlev];
            if let Some(col) = columns.first() {
                collect_offsets_from_layout_meta(&handle.reader(), &col.layout, 0, &mut offsets)?;
            }
            Ok(CanonShape {
                nrows: Some(nrows),
                offsets,
            })
        }
        _ => Err(io_wrong_type("dataframe", &handle.path)),
    }
}

fn collect_offsets_from_layout_meta(
    io: &IoReader<'_>,
    meta: &LayoutMeta,
    list_depth: usize,
    out: &mut [Option<Vec<i64>>],
) -> PyResult<()> {
    match meta {
        LayoutMeta::ListOffset { offsets, content } => {
            let lev = list_depth + 1;
            if lev < out.len() && out[lev].is_none() {
                out[lev] = Some(io.cache.read_i64(io.store, offsets)?);
            }
            collect_offsets_from_layout_meta(io, content, list_depth + 1, out)?;
        }
        LayoutMeta::OffsetView { content, .. } | LayoutMeta::Indexed { content, .. } => {
            collect_offsets_from_layout_meta(io, content, list_depth, out)?;
        }
        LayoutMeta::UnionScalarList { lists, .. } => {
            collect_offsets_from_layout_meta(io, lists, list_depth, out)?;
        }
        LayoutMeta::Leaf { .. } => {}
    }
    Ok(())
}

pub fn schema_from_handle(handle: &DatasetHandle) -> PyResult<Option<Schema>> {
    match &handle.meta.root {
        RootMeta::DataFrame { schema, .. } => Ok(schema_from_levels(schema.as_ref())),
        _ => Err(io_wrong_type("dataframe", &handle.path)),
    }
}

pub fn column_names_from_handle(handle: &DatasetHandle) -> PyResult<Vec<String>> {
    match &handle.meta.root {
        RootMeta::DataFrame { columns, .. } => Ok(columns.iter().map(|c| c.name.clone()).collect()),
        _ => Err(io_wrong_type("dataframe", &handle.path)),
    }
}

fn load_layout_axis0_slice(
    handle: &DatasetHandle,
    dt: DType,
    meta: &LayoutMeta,
    start: usize,
    stop: usize,
) -> PyResult<Layout> {
    load_layout_take_range(&handle.reader(), dt, meta, start, stop)
}

/// Disk-backed analogue of in-memory ``take_range`` (partial leaf reads only).
fn load_layout_take_range(
    io: &IoReader<'_>,
    dt: DType,
    meta: &LayoutMeta,
    start: usize,
    end: usize,
) -> PyResult<Layout> {
    if start > end {
        return Err(invalid_slice_range(start, end, start.max(end)));
    }
    match meta {
        LayoutMeta::Leaf { len, values, validity } => {
            if end > *len {
                return Err(index_out_of_bounds(end, *len, "on leaf slice in file"));
            }
            let valid = io.cache.read_bool_range(io.store, validity, start, end)?;
            let new_len = end - start;
            if valid.len() != new_len {
                return Err(internal("load_layout_take_range", "validity length mismatch in file"));
            }
            let mut leaf = Leaf::new(dt);
            leaf.len = new_len;
            leaf.has_nulls = valid.iter().any(|b| !*b);
            leaf.validity = Arc::new(bitvec::vec::BitVec::<u8, bitvec::order::Lsb0>::from_iter(
                valid.iter().copied(),
            ));
            leaf.buffer = match dt {
                DType::Int8 => LeafBuffer::I8(Arc::new(io.cache.read_i8_range(io.store, values, start, end)?)),
                DType::Int16 => LeafBuffer::I16(Arc::new(io.cache.read_i16_range(io.store, values, start, end)?)),
                DType::Int32 => LeafBuffer::I32(Arc::new(io.cache.read_i32_range(io.store, values, start, end)?)),
                DType::Int64 => LeafBuffer::I64(Arc::new(io.cache.read_i64_range(io.store, values, start, end)?)),
                DType::UInt8 => LeafBuffer::U8(Arc::new(io.cache.read_u8_range(io.store, values, start, end)?)),
                DType::UInt16 => LeafBuffer::U16(Arc::new(io.cache.read_u16_range(io.store, values, start, end)?)),
                DType::UInt32 => LeafBuffer::U32(Arc::new(io.cache.read_u32_range(io.store, values, start, end)?)),
                DType::UInt64 => LeafBuffer::U64(Arc::new(io.cache.read_u64_range(io.store, values, start, end)?)),
                DType::Float16 => LeafBuffer::F16(Arc::new(io.cache.read_u16_range(io.store, values, start, end)?)),
                DType::Float32 => LeafBuffer::F32(Arc::new(io.cache.read_f32_range(io.store, values, start, end)?)),
                DType::Float64 => LeafBuffer::F64(Arc::new(io.cache.read_f64_range(io.store, values, start, end)?)),
                DType::Bool => {
                    let b = io.cache.read_bool_range(io.store, values, start, end)?;
                    LeafBuffer::Bool(Arc::new(b.into_iter().map(|x| if x { 1 } else { 0 }).collect()))
                }
                DType::Char => LeafBuffer::Char(Arc::new(io.cache.read_u32_range(io.store, values, start, end)?)),
                DType::String => LeafBuffer::String(Arc::new(io.cache.read_string_range(io.store, values, start, end)?)),
            };
            Ok(Layout::Leaf(leaf))
        }
        LayoutMeta::ListOffset { offsets, content } => {
            let offs = io.cache.read_i64_arc(io.store, offsets)?;
            if end > offs.len().saturating_sub(1) {
                return Err(index_out_of_bounds(end, offs.len().saturating_sub(1), "on list slice in file"));
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
            let inner = load_layout_take_range(io, dt, content, child_start, child_end)?;
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
                return Err(index_out_of_bounds(end, *base_stop - base, "on list slice in file"));
            }
            let offs = io.cache.read_i64_arc(io.store, offsets)?;
            let child_start = offs[abs_start] as usize;
            let child_end = offs[abs_end] as usize;
            let inner = load_layout_take_range(io, dt, content, child_start, child_end)?;
            Ok(Layout::OffsetView(OffsetView {
                offsets: offs,
                start: abs_start,
                stop: abs_end,
                content: Box::new(inner),
            }))
        }
        LayoutMeta::Indexed { .. } => Err(unsupported(
            "load_layout_take_range",
            "Indexed layout streaming slice is not supported",
            "materialize indexed views before saving.",
        )),
        LayoutMeta::UnionScalarList {
            tags,
            index,
            scalars,
            lists,
        } => load_union_scalar_list_take_range(
            io,
            dt,
            tags,
            index,
            scalars,
            lists,
            start,
            end,
        ),
    }
}

fn load_union_scalar_list_take_range(
    io: &IoReader<'_>,
    dt: DType,
    tags_path: &str,
    index_path: &str,
    scalars_meta: &LayoutMeta,
    lists_meta: &LayoutMeta,
    start: usize,
    end: usize,
) -> PyResult<Layout> {
    let n = array_axis0_len(io.store, tags_path)?;
    if end > n {
        return Err(index_out_of_bounds(end, n, "on union slice in file"));
    }
    let slice_tags = io.cache.read_u8_range(io.store, tags_path, start, end)?;
    let slice_index = io.cache.read_i64_range(io.store, index_path, start, end)?;
    let (new_index, scalar_src, list_src) = remap_union_pick(&slice_tags, &slice_index)?;

    let scalars = if scalar_src.is_empty() {
        Leaf::new(dt)
    } else {
        load_leaf_indices(io, dt, scalars_meta, &scalar_src)?
    };

    let lists = if list_src.is_empty() {
        ListOffset {
            offsets: Arc::new(vec![0i64]),
            content: Box::new(Layout::Leaf(Leaf::new(dt))),
        }
    } else {
        load_union_lists_pick(io, dt, lists_meta, &list_src)?
    };

    Ok(Layout::UnionScalarList(UnionScalarList {
        tags: slice_tags,
        index: new_index,
        scalars,
        lists,
    }))
}

fn coalesce_index_runs(sorted_unique: &[usize]) -> Vec<(usize, usize)> {
    if sorted_unique.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut run_start = sorted_unique[0];
    let mut run_end = run_start + 1;
    for &ix in &sorted_unique[1..] {
        if ix == run_end {
            run_end += 1;
        } else {
            runs.push((run_start, run_end));
            run_start = ix;
            run_end = ix + 1;
        }
    }
    runs.push((run_start, run_end));
    runs
}

fn load_leaf_indices(
    io: &IoReader<'_>,
    dt: DType,
    meta: &LayoutMeta,
    indices: &[usize],
) -> PyResult<Leaf> {
    if indices.is_empty() {
        return Ok(Leaf::new(dt));
    }
    let len = match meta {
        LayoutMeta::Leaf { len, .. } => *len,
        _ => {
            return Err(internal(
                "load_union_scalar_list_take_range",
                "expected leaf metadata for union scalars",
            ))
        }
    };
    if let Some(&max_ix) = indices.iter().max() {
        if max_ix >= len {
            return Err(index_out_of_bounds(max_ix, len, "on union scalar index in file"));
        }
    }

    let mut unique: Vec<usize> = indices.iter().copied().collect();
    unique.sort_unstable();
    unique.dedup();

    let mut gathered: Vec<(usize, Leaf)> = Vec::with_capacity(unique.len());
    for (run_start, run_end) in coalesce_index_runs(&unique) {
        let Layout::Leaf(chunk) = load_layout_take_range(io, dt, meta, run_start, run_end)? else {
            return Err(internal(
                "load_union_scalar_list_take_range",
                "partial leaf read did not return a leaf",
            ));
        };
        for local in 0..(run_end - run_start) {
            let mut one = Leaf::new(dt);
            one.len = 1;
            one.has_nulls = chunk.has_nulls && !chunk.validity[local];
            one.validity = Arc::new(bitvec![u8, Lsb0; chunk.validity[local] as u8; 1]);
            one.buffer = chunk.buffer.copy_range(local, local + 1);
            gathered.push((run_start + local, one));
        }
    }
    gathered.sort_unstable_by_key(|(src, _)| *src);

    let mut out = Leaf::new(dt);
    out.len = indices.len();
    out.validity = Arc::new(bitvec![u8, Lsb0; 1; indices.len()]);
    out.has_nulls = false;
    out.buffer = LeafBuffer::new(dt);
    let out_valid = Arc::make_mut(&mut out.validity);
    for (out_i, &src) in indices.iter().enumerate() {
        let pos = gathered
            .binary_search_by_key(&src, |(ix, _)| *ix)
            .map_err(|_| internal("load_union_scalar", "missing gathered scalar"))?;
        let elem = &gathered[pos].1;
        if !elem.validity[0] {
            out_valid.set(out_i, false);
            out.has_nulls = true;
        }
        out.buffer.push_from_index(&elem.buffer, 0)?;
    }
    Ok(out)
}

fn load_union_lists_pick(
    io: &IoReader<'_>,
    dt: DType,
    lists_meta: &LayoutMeta,
    list_src: &[usize],
) -> PyResult<ListOffset> {
    let (offsets_path, content_meta) = match lists_meta {
        LayoutMeta::ListOffset { offsets, content } => (offsets.as_str(), content.as_ref()),
        _ => {
            return Err(internal(
                "load_union_scalar_list_take_range",
                "invalid union lists metadata in file",
            ))
        }
    };
    let offs = io.cache.read_i64_arc(io.store, offsets_path)?;
    let mut new_offs = vec![0i64];
    let mut acc = 0i64;
    let mut content_segs: Vec<Layout> = Vec::with_capacity(list_src.len());
    for &li in list_src {
        if li + 1 >= offs.len() {
            return Err(index_out_of_bounds(li, offs.len().saturating_sub(1), "on union list index in file"));
        }
        let s = offs[li] as usize;
        let e = offs[li + 1] as usize;
        let seg = load_layout_take_range(io, dt, content_meta, s, e)?;
        acc += (e - s) as i64;
        new_offs.push(acc);
        content_segs.push(seg);
    }
    let list_content = if content_segs.is_empty() {
        Layout::Leaf(Leaf::new(dt))
    } else {
        concat_layout_segments(&content_segs)?
    };
    Ok(ListOffset {
        offsets: Arc::new(new_offs),
        content: Box::new(list_content),
    })
}

/// Count entities at ``target_depth`` within each axis-0 row (reads offset buffers only).
pub fn row_entity_counts_at_depth(
    io: &IoReader<'_>,
    meta: &LayoutMeta,
    target_depth: usize,
) -> PyResult<Vec<usize>> {
    let n = axis0_len_from_layout_meta(io, meta)?;
    let mut out = Vec::with_capacity(n);
    for row in 0..n {
        out.push(count_entities_in_axis0_row(io, meta, row, target_depth, 0)?);
    }
    Ok(out)
}

fn count_entities_in_axis0_row(
    io: &IoReader<'_>,
    meta: &LayoutMeta,
    row: usize,
    target_depth: usize,
    current_depth: usize,
) -> PyResult<usize> {
    match meta {
        LayoutMeta::ListOffset { offsets, content } => {
            let offs = io.cache.read_i64_arc(io.store, offsets)?;
            if row + 1 >= offs.len() {
                return Err(index_out_of_bounds(row, offs.len().saturating_sub(1), "on dataframe row in file"));
            }
            let leaf_lo = offs[row] as usize;
            let leaf_hi = offs[row + 1] as usize;
            entity_count_in_leaf_range(
                io,
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
            let offs = io.cache.read_i64_arc(io.store, offsets)?;
            let abs_row = start + row;
            if abs_row + 1 >= offs.len() {
                return Err(index_out_of_bounds(row, offs.len().saturating_sub(1), "on dataframe row in file"));
            }
            let leaf_lo = offs[abs_row] as usize;
            let leaf_hi = offs[abs_row + 1] as usize;
            entity_count_in_leaf_range(
                io,
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
        LayoutMeta::UnionScalarList {
            tags,
            index,
            scalars: _,
            lists,
        } => count_entities_in_union_axis0_row(
            io,
            tags,
            index,
            lists,
            row,
            target_depth,
            current_depth,
        ),
        LayoutMeta::Indexed { .. } => Err(unsupported(
            "row_entity_counts_at_depth",
            "batch_on is not supported for Indexed layouts",
            "materialize indexed views before saving.",
        )),
    }
}

fn count_entities_in_union_axis0_row(
    io: &IoReader<'_>,
    tags_path: &str,
    index_path: &str,
    lists: &LayoutMeta,
    row: usize,
    target_depth: usize,
    current_depth: usize,
) -> PyResult<usize> {
    if target_depth == current_depth {
        return Ok(1);
    }
    let all_tags = io.cache.read_u8_arc(io.store, tags_path)?;
    let all_index = io.cache.read_i64_arc(io.store, index_path)?;
    if row >= all_tags.len() {
        return Err(index_out_of_bounds(row, all_tags.len(), "on dataframe row in file"));
    }
    match all_tags[row] {
        0 => Ok(if target_depth == current_depth + 1 {
            1
        } else {
            0
        }),
        1 => {
            let list_row = all_index[row] as usize;
            match lists {
                LayoutMeta::ListOffset { offsets, content } => {
                    let offs = io.cache.read_i64_arc(io.store, offsets)?;
                    if list_row + 1 >= offs.len() {
                        return Ok(0);
                    }
                    let leaf_lo = offs[list_row] as usize;
                    let leaf_hi = offs[list_row + 1] as usize;
                    entity_count_in_leaf_range(
                        io,
                        content,
                        leaf_lo,
                        leaf_hi,
                        target_depth,
                        current_depth + 1,
                    )
                }
                _ => Err(internal(
                    "load_union_scalar_list_take_range",
                    "invalid union lists metadata in file",
                )),
            }
        }
        _ => Err(internal("load_union_slice", "invalid union tag in file")),
    }
}

fn entity_count_in_leaf_range(
    io: &IoReader<'_>,
    meta: &LayoutMeta,
    leaf_lo: usize,
    leaf_hi: usize,
    target_depth: usize,
    current_depth: usize,
) -> PyResult<usize> {
    if current_depth == target_depth {
        return Ok(entity_count_at_depth(io, meta, leaf_lo, leaf_hi));
    }
    match meta {
        LayoutMeta::ListOffset { offsets, content } => {
            let offs = io.cache.read_i64_arc(io.store, offsets)?;
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
                    io,
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
        LayoutMeta::OffsetView { .. } => Err(unsupported(
            "row_entity_counts_at_depth",
            "batch_on depth counting unsupported for OffsetView layouts",
            "materialize offset views before saving.",
        )),
        LayoutMeta::UnionScalarList {
            tags,
            index,
            scalars: _,
            lists,
        } => count_entities_in_union_leaf_range(
            io,
            tags,
            index,
            lists,
            leaf_lo,
            leaf_hi,
            target_depth,
            current_depth,
        ),
        LayoutMeta::Indexed { .. } => Err(unsupported(
            "row_entity_counts_at_depth",
            "batch_on depth counting unsupported for Indexed layouts",
            "materialize indexed views before saving.",
        )),
    }
}

fn count_entities_in_union_leaf_range(
    io: &IoReader<'_>,
    tags_path: &str,
    index_path: &str,
    lists: &LayoutMeta,
    leaf_lo: usize,
    leaf_hi: usize,
    target_depth: usize,
    current_depth: usize,
) -> PyResult<usize> {
    if target_depth == current_depth {
        return Ok(leaf_hi.saturating_sub(leaf_lo));
    }
    let all_tags = io.cache.read_u8_arc(io.store, tags_path)?;
    let mut total = 0usize;
    for row in leaf_lo..leaf_hi {
        if row >= all_tags.len() {
            break;
        }
        total += count_entities_in_union_axis0_row(
            io,
            tags_path,
            index_path,
            lists,
            row,
            target_depth,
            current_depth,
        )?;
    }
    Ok(total)
}

fn entity_count_at_depth(
    io: &IoReader<'_>,
    meta: &LayoutMeta,
    leaf_lo: usize,
    leaf_hi: usize,
) -> usize {
    match meta {
        LayoutMeta::ListOffset { .. } => {
            count_list_elements_in_leaf_range(io, meta, leaf_lo, leaf_hi)
        }
        LayoutMeta::Leaf { .. } => leaf_hi.saturating_sub(leaf_lo),
        _ => 0,
    }
}

fn count_list_elements_in_leaf_range(
    io: &IoReader<'_>,
    meta: &LayoutMeta,
    leaf_lo: usize,
    leaf_hi: usize,
) -> usize {
    match meta {
        LayoutMeta::ListOffset { offsets, .. } => {
            let offs = io.cache.read_i64_arc(io.store, offsets).unwrap_or_default();
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
    Err(arg_invalid("chunk_dim", format!("unknown chunk_dim '{chunk_dim}'"), "use a schema level name or numeric depth."))
}
