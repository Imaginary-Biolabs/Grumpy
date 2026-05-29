"""Elementwise benchmarks: Grumpy vs NumPy vs Awkward (construction timed separately)."""

from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

import grumpy as gr

from _bench_common import print_header, timeit, try_import_awkward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=2048)
    ap.add_argument("--ncols", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ak = try_import_awkward()
    print_header(
        "Grumpy vs NumPy vs Awkward (elementwise mul + sum)",
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        awkward=ak.__version__ if ak else "(not installed)",
    )
    print(f"- nrows: {args.nrows}, ncols: {args.ncols}")
    print()

    rng = np.random.default_rng(args.seed)
    a_np = rng.integers(0, 1_000_000, size=(args.nrows, args.ncols), dtype=np.int32)
    b_np = rng.integers(1, 1_000_000, size=(args.nrows, args.ncols), dtype=np.int32)
    tmp = np.empty_like(a_np)

    # Construction (reported separately — not in kernel timing)
    t0 = time.perf_counter()
    a_gr = gr.array(a_np.tolist(), dtype=gr.int32)
    b_gr = gr.array(b_np.tolist(), dtype=gr.int32)
    t_gr_build_list = time.perf_counter() - t0

    t0 = time.perf_counter()
    a_gr_np = gr.array(a_np, dtype=gr.int32)
    b_gr_np = gr.array(b_np, dtype=gr.int32)
    t_gr_build_numpy = time.perf_counter() - t0

    t_ak_build = None
    a_ak = b_ak = None
    if ak is not None:
        t0 = time.perf_counter()
        a_ak = ak.from_numpy(a_np)
        b_ak = ak.from_numpy(b_np)
        t_ak_build = time.perf_counter() - t0

    def np_kernel() -> None:
        np.multiply(a_np, b_np, out=tmp)
        s = int(tmp.sum())
        if s == -1:
            raise RuntimeError

    def gr_kernel() -> None:
        s = a_gr._mul2d_i32_sum_i64(b_gr)
        if int(s) == -1:
            raise RuntimeError

    def ak_kernel() -> None:
        assert ak is not None and a_ak is not None and b_ak is not None
        out = a_ak * b_ak
        s = int(ak.sum(out))
        if s == -1:
            raise RuntimeError

    rows = [
        ("numpy_kernel", timeit(np_kernel, warmup=args.warmup, repeats=args.repeats)),
        ("grumpy_kernel", timeit(gr_kernel, warmup=args.warmup, repeats=args.repeats)),
    ]
    if ak is not None:
        rows.append(("awkward_kernel", timeit(ak_kernel, warmup=args.warmup, repeats=args.repeats)))

    print("| phase | time |")
    print("|---|---:|")
    print(f"| grumpy_construct (nested lists) | {t_gr_build_list * 1e3:.3f} ms |")
    print(f"| grumpy_construct (numpy array) | {t_gr_build_numpy * 1e3:.3f} ms |")
    if t_ak_build is not None:
        print(f"| awkward_construct | {t_ak_build * 1e3:.3f} ms |")
    print()
    print("| kernel | time |")
    print("|---|---:|")
    for name, t in rows:
        print(f"| {name} | {t * 1e3:.3f} ms |")
    if len(rows) >= 2:
        print()
        print(f"- grumpy_kernel / numpy_kernel: {rows[1][1] / rows[0][1]:.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
