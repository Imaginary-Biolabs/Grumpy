from __future__ import annotations

import argparse
import platform
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

import grumpy as gr


@dataclass(frozen=True)
class Result:
    name: str
    seconds: float


def _timeit(fn: Callable[[], None], *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def _fmt_s(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.1f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _print_header(args: argparse.Namespace) -> None:
    print("## Grumpy vs NumPy indexing benchmark")
    print()
    print(f"- python: {sys.version.split()[0]}")
    print(f"- numpy: {np.__version__}")
    print(f"- platform: {platform.platform()}")
    print(f"- nrows: {args.nrows}, ncols: {args.ncols}, nfancy: {args.nfancy}")
    print(f"- warmup: {args.warmup}, repeats: {args.repeats}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark getitem/setitem for Grumpy vs NumPy.")
    ap.add_argument("--nrows", type=int, default=4096)
    ap.add_argument("--ncols", type=int, default=64)
    ap.add_argument("--nfancy", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _print_header(args)

    rng = np.random.default_rng(args.seed)

    # Rectangular reference data (NumPy typed, Grumpy built from nested Python lists).
    np_x0 = rng.integers(0, 1_000_000, size=(args.nrows, args.ncols), dtype=np.int32)
    py_x0 = np_x0.tolist()
    gr_x0 = gr.array(py_x0, dtype=gr.int32)

    # Indices for scalar access (random positions).
    scalar_rows = rng.integers(0, args.nrows, size=args.nfancy, dtype=np.int64)
    scalar_cols = rng.integers(0, args.ncols, size=args.nfancy, dtype=np.int64)

    # Indices for fancy access/assignment (vectorized in NumPy; fast-path in Grumpy for pure 2D list-chains).
    fancy_rows = scalar_rows.copy()
    fancy_cols = scalar_cols.copy()
    fancy_vals = rng.integers(0, 1_000_000, size=args.nfancy, dtype=np.int32)
    # Ensure stable dtypes without per-iteration conversion costs.
    fancy_rows_i64 = fancy_rows.astype(np.int64, copy=False)
    fancy_cols_i64 = fancy_cols.astype(np.int64, copy=False)
    fancy_vals_i32 = fancy_vals.astype(np.int32, copy=False)

    # ---- getitem scalar loop ----
    def np_get_scalar() -> None:
        s = 0
        for r, c in zip(scalar_rows, scalar_cols):
            s += int(np_x0[int(r), int(c)])
        if s == -1:
            raise RuntimeError

    def gr_get_scalar() -> None:
        s = 0
        for r, c in zip(scalar_rows, scalar_cols):
            s += int(gr_x0[int(r), int(c)])
        if s == -1:
            raise RuntimeError

    # ---- getitem fancy ----
    def np_get_fancy() -> None:
        out = np_x0[fancy_rows_i64, fancy_cols_i64]
        # force realization + a tiny reduction
        if int(out.sum()) == -1:
            raise RuntimeError

    def gr_get_fancy() -> None:
        # Avoid Python list conversion: pass NumPy arrays directly.
        # Kernel-only timing (no to_numpy/to_list): gather+reduce in Rust.
        s = gr_x0._gather2d_sum_i64(fancy_rows_i64, fancy_cols_i64)
        if int(s) == -1:
            raise RuntimeError

    # ---- setitem scalar loop ----
    def np_set_scalar() -> None:
        x = np_x0.copy()
        v = 7
        for r, c in zip(scalar_rows, scalar_cols):
            x[int(r), int(c)] = v
        if int(x[0, 0]) == -1:
            raise RuntimeError

    def gr_set_scalar() -> None:
        x = gr_x0.copy()
        v = 7
        for r, c in zip(scalar_rows, scalar_cols):
            x[int(r), int(c)] = v
        if int(x[0, 0]) == -1:
            raise RuntimeError

    # ---- setitem fancy ----
    def np_set_fancy() -> None:
        x = np_x0.copy()
        x[fancy_rows_i64, fancy_cols_i64] = fancy_vals_i32
        if int(x[0, 0]) == -1:
            raise RuntimeError

    def gr_set_fancy() -> None:
        x = gr_x0.copy()
        # Kernel-only timing: scatter in Rust (int32 only)
        x._scatter2d_i32(fancy_rows_i64, fancy_cols_i64, fancy_vals_i32)
        if int(x[0, 0]) == -1:
            raise RuntimeError

    benches = [
        ("getitem_scalar_loop", np_get_scalar, gr_get_scalar),
        ("getitem_fancy", np_get_fancy, gr_get_fancy),
        ("setitem_scalar_loop", np_set_scalar, gr_set_scalar),
        ("setitem_fancy", np_set_fancy, gr_set_fancy),
    ]

    results: list[tuple[str, Result, Result]] = []
    for name, np_fn, gr_fn in benches:
        np_t = _timeit(np_fn, warmup=args.warmup, repeats=args.repeats)
        gr_t = _timeit(gr_fn, warmup=args.warmup, repeats=args.repeats)
        results.append((name, Result("numpy", np_t), Result("grumpy", gr_t)))

    # Print table
    print("| op | numpy | grumpy | grumpy/numpy |")
    print("|---|---:|---:|---:|")
    for name, np_r, gr_r in results:
        ratio = gr_r.seconds / np_r.seconds if np_r.seconds > 0 else float("inf")
        print(f"| {name} | {_fmt_s(np_r.seconds)} | {_fmt_s(gr_r.seconds)} | {ratio:.2f}× |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


