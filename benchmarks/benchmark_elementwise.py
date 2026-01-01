from __future__ import annotations

import argparse
import platform
import sys
import time
from typing import Callable

import numpy as np

import grumpy as gr


def _timeit(fn: Callable[[], None], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark elementwise ops for Grumpy vs NumPy.")
    ap.add_argument("--nrows", type=int, default=4096)
    ap.add_argument("--ncols", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("## Grumpy vs NumPy elementwise benchmark")
    print()
    print(f"- python: {sys.version.split()[0]}")
    print(f"- numpy: {np.__version__}")
    print(f"- platform: {platform.platform()}")
    print(f"- nrows: {args.nrows}, ncols: {args.ncols}")
    print(f"- warmup: {args.warmup}, repeats: {args.repeats}")
    print()

    rng = np.random.default_rng(args.seed)

    # Rectangular case
    a_np = rng.integers(0, 1_000_000, size=(args.nrows, args.ncols), dtype=np.int32)
    b_np = rng.integers(1, 1_000_000, size=(args.nrows, args.ncols), dtype=np.int32)
    tmp = np.empty_like(a_np)
    a_gr = gr.array(a_np.tolist(), dtype=gr.int32)
    b_gr = gr.array(b_np.tolist(), dtype=gr.int32)

    # Slightly ragged case (+/- 1 element per row). NumPy baseline uses the flattened buffers.
    ragged_rows = []
    ragged_rows_b = []
    flat_a = []
    flat_b = []
    for i in range(args.nrows):
        d = -1 if (i % 2 == 0) else 1
        m = max(0, args.ncols + d)
        ra = rng.integers(0, 1_000_000, size=m, dtype=np.int32).tolist()
        rb = rng.integers(1, 1_000_000, size=m, dtype=np.int32).tolist()
        ragged_rows.append(ra)
        ragged_rows_b.append(rb)
        flat_a.extend(ra)
        flat_b.extend(rb)
    flat_a_np = np.asarray(flat_a, dtype=np.int32)
    flat_b_np = np.asarray(flat_b, dtype=np.int32)
    tmp_flat = np.empty_like(flat_a_np)
    a_gr_rag = gr.array(ragged_rows, dtype=gr.int32)
    b_gr_rag = gr.array(ragged_rows_b, dtype=gr.int32)

    def np_mul_rect() -> None:
        # Avoid allocation: reuse a temp buffer (still two passes: multiply then sum).
        np.multiply(a_np, b_np, out=tmp)
        s = int(tmp.sum())
        if s == -1:
            raise RuntimeError

    def gr_mul_rect_kernel() -> None:
        s = a_gr._mul2d_i32_sum_i64(b_gr)
        if int(s) == -1:
            raise RuntimeError

    def gr_mul_rect_via_op() -> None:
        s = a_gr._mul2d_i32_sum_via_op_i64(b_gr)
        if int(s) == -1:
            raise RuntimeError

    def np_add_rect() -> None:
        np.add(a_np, b_np, out=tmp)
        s = int(tmp.sum())
        if s == -1:
            raise RuntimeError

    def gr_add_rect_kernel() -> None:
        s = a_gr._add2d_i32_sum_i64(b_gr)
        if int(s) == -1:
            raise RuntimeError

    def gr_add_rect_via_op() -> None:
        s = a_gr._add2d_i32_sum_via_op_i64(b_gr)
        if int(s) == -1:
            raise RuntimeError

    def np_mul_ragged() -> None:
        np.multiply(flat_a_np, flat_b_np, out=tmp_flat)
        s = int(tmp_flat.sum())
        if s == -1:
            raise RuntimeError

    def gr_mul_ragged_kernel() -> None:
        s = a_gr_rag._mul2d_i32_sum_i64(b_gr_rag)
        if int(s) == -1:
            raise RuntimeError

    def gr_mul_ragged_via_op() -> None:
        s = a_gr_rag._mul2d_i32_sum_via_op_i64(b_gr_rag)
        if int(s) == -1:
            raise RuntimeError

    def np_add_ragged() -> None:
        np.add(flat_a_np, flat_b_np, out=tmp_flat)
        s = int(tmp_flat.sum())
        if s == -1:
            raise RuntimeError

    def gr_add_ragged_kernel() -> None:
        s = a_gr_rag._add2d_i32_sum_i64(b_gr_rag)
        if int(s) == -1:
            raise RuntimeError

    def gr_add_ragged_via_op() -> None:
        s = a_gr_rag._add2d_i32_sum_via_op_i64(b_gr_rag)
        if int(s) == -1:
            raise RuntimeError

    benches = [
        ("mul_rect_numpy", np_mul_rect, None),
        ("mul_rect_grumpy_kernel", None, gr_mul_rect_kernel),
        ("mul_rect_grumpy_via_op", None, gr_mul_rect_via_op),
        ("add_rect_numpy", np_add_rect, None),
        ("add_rect_grumpy_kernel", None, gr_add_rect_kernel),
        ("add_rect_grumpy_via_op", None, gr_add_rect_via_op),
        ("mul_ragged_numpy_flat", np_mul_ragged, None),
        ("mul_ragged_grumpy_kernel", None, gr_mul_ragged_kernel),
        ("mul_ragged_grumpy_via_op", None, gr_mul_ragged_via_op),
        ("add_ragged_numpy_flat", np_add_ragged, None),
        ("add_ragged_grumpy_kernel", None, gr_add_ragged_kernel),
        ("add_ragged_grumpy_via_op", None, gr_add_ragged_via_op),
    ]

    print("| op | time |")
    print("|---|---:|")
    results: dict[str, float] = {}
    for name, np_fn, gr_fn in benches:
        fn = np_fn if np_fn is not None else gr_fn
        assert fn is not None
        t = _timeit(fn, warmup=args.warmup, repeats=args.repeats)
        results[name] = t
        print(f"| {name} | {t*1e3:.3f} ms |")

    def _ratio(a: str, b: str) -> float:
        return results[a] / results[b] if results[b] > 0 else float("inf")

    print()
    print("### Ratios (lower is better for Grumpy)")
    print(f"- mul_rect_grumpy_kernel / mul_rect_numpy: {_ratio('mul_rect_grumpy_kernel','mul_rect_numpy'):.2f}×")
    print(f"- mul_rect_grumpy_via_op / mul_rect_numpy: {_ratio('mul_rect_grumpy_via_op','mul_rect_numpy'):.2f}×")
    print(f"- add_rect_grumpy_kernel / add_rect_numpy: {_ratio('add_rect_grumpy_kernel','add_rect_numpy'):.2f}×")
    print(f"- add_rect_grumpy_via_op / add_rect_numpy: {_ratio('add_rect_grumpy_via_op','add_rect_numpy'):.2f}×")
    print(f"- mul_ragged_grumpy_kernel / mul_ragged_numpy_flat: {_ratio('mul_ragged_grumpy_kernel','mul_ragged_numpy_flat'):.2f}×")
    print(f"- mul_ragged_grumpy_via_op / mul_ragged_numpy_flat: {_ratio('mul_ragged_grumpy_via_op','mul_ragged_numpy_flat'):.2f}×")
    print(f"- add_ragged_grumpy_kernel / add_ragged_numpy_flat: {_ratio('add_ragged_grumpy_kernel','add_ragged_numpy_flat'):.2f}×")
    print(f"- add_ragged_grumpy_via_op / add_ragged_numpy_flat: {_ratio('add_ragged_grumpy_via_op','add_ragged_numpy_flat'):.2f}×")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


