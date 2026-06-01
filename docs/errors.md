# Error reporting

Grumpy errors are designed to be **actionable**: every user-facing failure should explain
what went wrong, why it happened, and how to fix it.

## Message format

Errors raised from Rust or Python use a shared shape:

```text
grumpy.<Code>: <one-line summary>
  cause: <root constraint that was violated>
  optional_context_key: value
  fix: <concrete remediation>
```

Example (broadcasting):

```text
grumpy.BroadcastFailed: incompatible union outer lengths 3 and 4
  cause: UnionScalarList broadcasting requires equal outer length, or one side with outer length 1.
  fix: align outer lengths, insert a length-1 axis, or reshape so one array broadcasts.
```

Example (streaming):

```text
grumpy.ArgumentInvalid: shuffle is set but seed is None
  cause: shuffled batch order requires a fixed seed for reproducible training.
  fix: pass seed=<int> when shuffle is True or a schema level name.
```

## Error codes

| Code | Typical use |
|------|-------------|
| `ArgumentInvalid` | Bad function arguments (`batch_size`, `compile`, …) |
| `BroadcastFailed` | Incompatible shapes for elementwise / broadcast |
| `CastNotAllowed` | `astype(..., casting='safe')` rejected a conversion |
| `ConcatIncompatible` | `gr.cat` cannot merge layouts or dtypes |
| `DtypeMismatch` | Operands require matching dtypes |
| `IndexOutOfBounds` | Indexing, slicing, or batch index out of range |
| `IoFailed` | Zarr / filesystem read or write problems |
| `LayoutUnsupported` | Op does not support this layout (union vs list-chain, views, …) |
| `ReduceDimInvalid` | Invalid `dim` for reduction on this array |
| `ReduceEmpty` | Reduction over empty or all-null data |
| `SchemaViolation` | DataFrame schema / column shape constraints |
| `ShapeMismatch` | Reshape, unflatten, or axis length mismatch |
| `Unsupported` | Valid call but not implemented for this dtype/layout |
| `InternalError` | Unexpected invariant violation (please report) |

Python exceptions remain subclasses of `ValueError` or `IndexError` so existing
`pytest.raises(ValueError)` tests keep working.

## Contributor checklist

When adding or changing an error path:

1. **Use helpers** — Rust: `crate::error::{err, index_out_of_bounds, …}`.
   Python: `grumpy.errors.raise_grumpy_error` / `arg_invalid`.
2. **Name the code** — pick the closest `ErrorCode`; add a new code only when needed.
3. **State the cause** — what invariant failed (lengths, dtypes, layout kind, file type).
4. **Suggest a fix** — cast, reshape, different `dim`, schema column, compile mode, etc.
5. **Add context** — axis index, dtype names, column name, path, `dim` value when helpful.
6. **Avoid vague text** — prefer `grumpy.IndexOutOfBounds: index 3 …` over `Index out of bounds.`
7. **Test the message** — assert `grumpy.<Code>` and `fix:` appear in new tests when behavior is user-facing.

See also [CONTRIBUTING.md](../CONTRIBUTING.md) and [AGENTS.md](../AGENTS.md).

## Implementation

- Rust: [`src/error.rs`](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/src/error.rs)
- Python: [`python/grumpy/errors.py`](https://github.com/Imaginary-Biolabs/Grumpy/blob/main/python/grumpy/errors.py)
