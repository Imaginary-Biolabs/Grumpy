//! Structured Grumpy error messages for Python.
//!
//! Every user-facing error should use this module so messages include:
//! - a stable `grumpy.<Code>` prefix
//! - a one-line summary of what failed
//! - `cause:` pointing at the root constraint that was violated
//! - optional `context:` key/value lines (layout, dtype, axis, …)
//! - `fix:` with a concrete remediation when possible
//!
//! See `docs/developer.md` and `CONTRIBUTING.md` for the contributor checklist.

use crate::dtype::DType;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::PyErr;

/// Stable error codes surfaced in messages and tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    ArgumentInvalid,
    BroadcastFailed,
    CastNotAllowed,
    ConcatIncompatible,
    DtypeMismatch,
    IndexOutOfBounds,
    Internal,
    IoFailed,
    LayoutUnsupported,
    ReduceDimInvalid,
    ReduceEmpty,
    SchemaViolation,
    ShapeMismatch,
    TypeMismatch,
    Unsupported,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::ArgumentInvalid => "ArgumentInvalid",
            ErrorCode::BroadcastFailed => "BroadcastFailed",
            ErrorCode::CastNotAllowed => "CastNotAllowed",
            ErrorCode::ConcatIncompatible => "ConcatIncompatible",
            ErrorCode::DtypeMismatch => "DtypeMismatch",
            ErrorCode::IndexOutOfBounds => "IndexOutOfBounds",
            ErrorCode::Internal => "InternalError",
            ErrorCode::IoFailed => "IoFailed",
            ErrorCode::LayoutUnsupported => "LayoutUnsupported",
            ErrorCode::ReduceDimInvalid => "ReduceDimInvalid",
            ErrorCode::ReduceEmpty => "ReduceEmpty",
            ErrorCode::SchemaViolation => "SchemaViolation",
            ErrorCode::ShapeMismatch => "ShapeMismatch",
            ErrorCode::TypeMismatch => "TypeMismatch",
            ErrorCode::Unsupported => "Unsupported",
        }
    }
}

/// Builder for formatted Grumpy errors.
pub struct ErrorBuilder {
    code: ErrorCode,
    summary: String,
    cause: Option<String>,
    fix: Option<String>,
    context: Vec<(String, String)>,
}

impl ErrorBuilder {
    pub fn new(code: ErrorCode, summary: impl Into<String>) -> Self {
        Self {
            code,
            summary: summary.into(),
            cause: None,
            fix: None,
            context: Vec::new(),
        }
    }

    pub fn cause(mut self, cause: impl Into<String>) -> Self {
        self.cause = Some(cause.into());
        self
    }

    pub fn fix(mut self, fix: impl Into<String>) -> Self {
        self.fix = Some(fix.into());
        self
    }

    pub fn ctx(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.push((key.into(), value.into()));
        self
    }

    pub fn format(self) -> String {
        let mut out = format!("grumpy.{}: {}", self.code.as_str(), self.summary);
        if let Some(cause) = self.cause {
            out.push_str("\n  cause: ");
            out.push_str(&cause);
        }
        for (k, v) in self.context {
            out.push_str("\n  ");
            out.push_str(&k);
            out.push(':');
            out.push(' ');
            out.push_str(&v);
        }
        if let Some(fix) = self.fix {
            out.push_str("\n  fix: ");
            out.push_str(&fix);
        }
        out
    }

    pub fn value_err(self) -> PyErr {
        PyValueError::new_err(self.format())
    }

    pub fn index_err(self) -> PyErr {
        PyIndexError::new_err(self.format())
    }

    pub fn key_err(self) -> PyErr {
        PyKeyError::new_err(self.format())
    }

    pub fn type_err(self) -> PyErr {
        PyTypeError::new_err(self.format())
    }
}

/// Start building an error: `err(ErrorCode::BroadcastFailed, "…")`.
pub fn err(code: ErrorCode, summary: impl Into<String>) -> ErrorBuilder {
    ErrorBuilder::new(code, summary)
}

// ---- Index / axis ----

pub fn index_out_of_bounds(index: usize, len: usize, at: &str) -> PyErr {
    err(
        ErrorCode::IndexOutOfBounds,
        format!("index {index} is out of range for length {len} {at}"),
    )
    .cause(format!(
        "valid indices for this axis are 0..{len} (or negative indices -{len}..-1 when supported)."
    ))
    .fix(format!(
        "use an index in [0, {len}) or check the array length with len(array) / array.shape(dim=…)."
    ))
    .index_err()
}

pub fn index_out_of_bounds_simple(at: &str) -> PyErr {
    err(
        ErrorCode::IndexOutOfBounds,
        format!("index out of bounds {at}"),
    )
    .cause("the requested index exceeds the length of the indexed axis.")
    .fix("verify index values against array length or slice bounds before indexing.")
    .index_err()
}

pub fn dim_out_of_range(dim: isize, ndim: usize) -> PyErr {
    err(
        ErrorCode::ReduceDimInvalid,
        format!("axis {dim} is out of range for ndim={ndim}"),
    )
    .cause(format!(
        "after normalizing negative axes, valid dim values are 0..{ndim} or -{ndim}..-1."
    ))
    .fix(format!(
        "pass dim in [0, {ndim}) or use dim=-1 for the innermost axis."
    ))
    .value_err()
}

pub fn invalid_slice_range(start: usize, stop: usize, len: usize) -> PyErr {
    err(
        ErrorCode::IndexOutOfBounds,
        format!("slice [{start}, {stop}) is invalid for length {len}"),
    )
    .cause("slice start must be <= stop and stop must be <= axis length.")
    .fix(format!("use 0 <= start <= stop <= {len}."))
    .value_err()
}

// ---- Dtype / cast ----

pub fn dtype_mismatch(expected: DType, got: DType, at: &str) -> PyErr {
    err(
        ErrorCode::DtypeMismatch,
        format!("expected dtype={} but got dtype={} {at}", expected.name(), got.name()),
    )
    .cause("Grumpy requires matching dtypes for this operation (or an explicit cast).")
    .fix(format!(
        "cast with array.astype({}) or construct inputs with gr.array(..., dtype=gr.{})",
        python_dtype_name(got),
        rust_dtype_alias(expected),
    ))
    .value_err()
}

pub fn cast_not_allowed(from: DType, to: DType, mode: &str) -> PyErr {
    err(
        ErrorCode::CastNotAllowed,
        format!(
            "cannot cast from {} to {} with casting='{}'",
            from.name(),
            to.name(),
            mode
        ),
    )
    .cause(format!(
        "casting='{}' rejects this conversion because values would be truncated, overflow, or lose precision.",
        mode
    ))
    .fix(format!(
        "use casting='unsafe' or 'same_kind' if intentional, or choose an intermediate dtype with gr.promote_types({}, {}).",
        python_dtype_name(from),
        python_dtype_name(to),
    ))
    .value_err()
}

pub fn dtype_unsupported(op: &str, dt: DType) -> PyErr {
    err(
        ErrorCode::Unsupported,
        format!("{op} is not implemented for dtype={}", dt.name()),
    )
    .cause("this kernel has no leaf buffer path for the requested dtype.")
    .fix("cast to a supported numeric dtype (e.g. float64) or open an issue with your use case.")
    .value_err()
}

// ---- Broadcast / shape ----

pub fn broadcast_failed(summary: impl Into<String>, cause: impl Into<String>, fix: impl Into<String>) -> PyErr {
    err(ErrorCode::BroadcastFailed, summary)
        .cause(cause)
        .fix(fix)
        .value_err()
}

pub fn broadcast_union_outer_mismatch(na: usize, nb: usize) -> PyErr {
    broadcast_failed(
        format!("incompatible union outer lengths {na} and {nb}"),
        "UnionScalarList broadcasting requires equal outer length, or one side with outer length 1.",
        "align outer lengths, insert a length-1 axis, or reshape so one array broadcasts.",
    )
}

pub fn broadcast_union_layout_kind() -> PyErr {
    broadcast_failed(
        "cannot broadcast union array with this layout kind",
        "the non-union operand is neither a scalar leaf (len==1) nor a compatible list/union layout.",
        "ensure both operands are union arrays, list-chains, or a broadcastable scalar.",
    )
}

pub fn shape_mismatch(op: &str, detail: impl Into<String>, fix: impl Into<String>) -> PyErr {
    err(ErrorCode::ShapeMismatch, format!("{op}: {}", detail.into()))
        .fix(fix)
        .value_err()
}

// ---- Layout / concat ----

pub fn layout_unsupported(op: &str, detail: impl Into<String>) -> PyErr {
    err(
        ErrorCode::LayoutUnsupported,
        format!("{op} does not support this layout"),
    )
    .cause(detail)
    .fix("use gr.array(...) to build a list-chain or UnionScalarList layout, or normalize views before calling this op.")
    .value_err()
}

pub fn concat_incompatible(detail: impl Into<String>, fix: impl Into<String>) -> PyErr {
    err(ErrorCode::ConcatIncompatible, "concatenation failed")
        .cause(detail)
        .fix(fix)
        .value_err()
}

pub fn union_op_dim_unsupported(op: &str, dim: isize, supported: &str) -> PyErr {
    err(
        ErrorCode::ReduceDimInvalid,
        format!("{op} on UnionScalarList does not support dim={dim}"),
    )
    .cause(format!("union {op} currently supports {supported} only."))
    .fix(format!("use {supported}, or convert to a pure list-chain if you need other axes."))
    .value_err()
}

// ---- Reduce ----

pub fn reduction_empty(op: &str) -> PyErr {
    err(
        ErrorCode::ReduceEmpty,
        format!("{op} on an empty array"),
    )
    .cause("all selected values were null, empty lists, or filtered out.")
    .fix("check for empty segments before reducing, or use nan-aware stats if applicable.")
    .value_err()
}

pub fn reduction_scalar_unsupported(op: &str) -> PyErr {
    err(
        ErrorCode::Unsupported,
        format!("compiled {op} cannot produce a Python scalar"),
    )
    .cause("Rust-scheduled reductions must return array layouts, not 0-d scalars.")
    .fix("reduce with an explicit dim, or run the op outside a compiled Rust pipeline.")
    .value_err()
}

// ---- Schema / dataframe ----

pub fn schema_violation(summary: impl Into<String>, cause: impl Into<String>, fix: impl Into<String>) -> PyErr {
    err(ErrorCode::SchemaViolation, summary)
        .cause(cause)
        .fix(fix)
        .value_err()
}

pub fn unknown_column(name: &str) -> PyErr {
    err(
        ErrorCode::SchemaViolation,
        format!("unknown column '{name}'"),
    )
    .cause("the dataframe has no column with this name.")
    .fix("use df.columns or dot-notation levels declared in schema= when constructing the dataframe.")
    .key_err()
}

// ---- I/O ----

pub fn io_failed(summary: impl Into<String>, cause: impl Into<String>, fix: impl Into<String>) -> PyErr {
    err(ErrorCode::IoFailed, summary)
        .cause(cause)
        .fix(fix)
        .value_err()
}

pub fn io_closed(path: &str) -> PyErr {
    io_failed(
        format!("OpenDataFrame('{path}') is closed"),
        "this lazy dataset handle was closed and can no longer be used for I/O.",
        "open the path again with gr.open(...) or use a with gr.open(...) as handle block.",
    )
}

pub fn io_wrong_type(expected: &str, path: &str) -> PyErr {
    io_failed(
        format!("{path} is not a saved Grumpy {expected}"),
        format!("grumpy.json at this path describes a different root type than {expected}."),
        format!("use gr.load(...) for arrays/dataframes, or gr.open(...) for lazy dataframe access."),
    )
}

// ---- Arguments ----

pub fn arg_invalid(name: &str, detail: impl Into<String>, fix: impl Into<String>) -> PyErr {
    err(
        ErrorCode::ArgumentInvalid,
        format!("invalid argument '{name}': {}", detail.into()),
    )
    .fix(fix)
    .value_err()
}

pub fn arg_must_be_positive(name: &str, value: impl std::fmt::Display) -> PyErr {
    arg_invalid(
        name,
        format!("got {value}, expected > 0"),
        format!("pass a positive integer for {name}."),
    )
}

// ---- Internal (still user-visible when surfaced) ----

pub fn internal(op: &str, detail: impl Into<String>) -> PyErr {
    err(
        ErrorCode::Internal,
        format!("unexpected state in {op}"),
    )
    .cause(detail)
    .fix("report a bug with a minimal reproducer (grumpy.__version__, input arrays, and the failing call).")
    .value_err()
}

pub fn internal_dtype_buffer_mismatch(op: &str, dt: DType) -> PyErr {
    internal(
        op,
        format!(
            "leaf declares dtype={} but the active buffer variant does not match",
            dt.name()
        ),
    )
}

pub fn unsupported(op: &str, detail: impl Into<String>, fix: impl Into<String>) -> PyErr {
    err(ErrorCode::Unsupported, format!("{op} is not supported"))
        .cause(detail)
        .fix(fix)
        .value_err()
}

fn python_dtype_name(dt: DType) -> &'static str {
    match dt {
        DType::Int8 => "gr.int8",
        DType::Int16 => "gr.int16",
        DType::Int32 => "gr.int32",
        DType::Int64 => "gr.int64",
        DType::UInt8 => "gr.uint8",
        DType::UInt16 => "gr.uint16",
        DType::UInt32 => "gr.uint32",
        DType::UInt64 => "gr.uint64",
        DType::Float16 => "gr.float16",
        DType::Float32 => "gr.float32",
        DType::Float64 => "gr.float64",
        DType::Bool => "gr.bool_",
        DType::Char => "gr.char",
        DType::String => "gr.string",
    }
}

fn rust_dtype_alias(dt: DType) -> &'static str {
    match dt {
        DType::Int32 => "int32",
        DType::Int64 => "int64",
        DType::Float32 => "float32",
        DType::Float64 => "float64",
        DType::Bool => "bool_",
        other => other.name(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_message_includes_sections() {
        let msg = err(ErrorCode::BroadcastFailed, "outer lengths differ")
            .cause("got 3 and 4")
            .ctx("axis", "0")
            .fix("align lengths or broadcast a length-1 axis")
            .format();
        assert!(msg.starts_with("grumpy.BroadcastFailed:"));
        assert!(msg.contains("cause: got 3 and 4"));
        assert!(msg.contains("axis: 0"));
        assert!(msg.contains("fix: align"));
    }
}
