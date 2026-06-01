# Dtypes and casting

Grumpy arrays carry a single **homogeneous dtype** on all leaves. Construction infers dtype from Python values when omitted; explicit ``dtype=`` overrides inference.

## Supported dtypes

| Dtype | Role |
|-------|------|
| ``int8`` … ``int64`` | Signed integers |
| ``uint8`` … ``uint64`` | Unsigned integers |
| ``float16``, ``float32``, ``float64`` | Floating point |
| ``bool`` | Boolean (0/1 storage) |
| ``char`` | Single Unicode scalar |
| ``string`` | UTF-8 strings |

Inference defaults: Python ``int`` → ``int64``, ``float`` → ``float64``, ``bool`` → ``bool``.

## Casting modes

``GrumpyArray.astype(dtype, casting='safe')`` uses **layout-preserving** casts (list-chains and unions) without materializing Python lists.

| Mode | Behavior |
|------|----------|
| ``safe`` (default) | Widen integers/floats without loss; bool → numeric; ``char`` → ``string`` |
| ``same_kind`` | Safe casts plus float narrowing (``float64`` → ``float32`` → ``float16``) and integer narrowing with overflow errors |
| ``unsafe`` | All numeric casts; integer narrowing wraps; float→int truncates toward zero |

String/char never cast to/from numeric types.

## Promotion

Binary elementwise ops promote dtypes with NumPy ``promote_types`` rules:

```python
import grumpy as gr

gr.promote_types(gr.int32, gr.float32).name   # 'float64'
gr.promote_types(gr.uint32, gr.int32).name    # 'int64'

a = gr.array([[1, 2]], dtype=gr.int32)
b = gr.array([[1.0, 2.0]], dtype=gr.float64)
(a + b).dtype.name   # 'float64'
```

Use ``gr.can_cast(from_dtype, to_dtype, casting='safe')`` to check casts before calling ``astype``.

## Nulls

Null validity bitmaps are preserved across casts; null leaf slots are not converted.

```python
x = gr.array([1, None, 3], dtype=gr.int32)
x.astype(gr.float64).to_list()   # [1.0, None, 3.0]

u = gr.array([1, [None, 2], 3], dtype=gr.int32)
u.astype(gr.float64).to_list() # [1.0, [None, 2.0], 3.0]
```

## Reduction output dtypes

| Op | Integer input | Float input |
|----|---------------|-------------|
| ``sum`` | ``int64`` | ``float64`` |
| ``mean``, ``var``, ``std`` | ``float64`` | ``float64`` |
| ``min``, ``max``, ``ptp`` | same dtype | same dtype |

Division of integers produces ``float64``.
