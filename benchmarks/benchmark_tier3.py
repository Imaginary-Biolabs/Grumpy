#!/usr/bin/env python3
"""Benchmark Tier 3 features: casting, strings, unions, einsum."""

from __future__ import annotations

import argparse

import numpy as np

import grumpy as gr

from _bench_common import fmt_ms, timeit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()
    n = args.n

    rng = np.random.default_rng(0)
    int_a = gr.array(rng.integers(0, 1000, size=n, dtype=np.int32).tolist(), dtype=gr.int32)
    flt_b = gr.array(rng.random(n).tolist(), dtype=gr.float64)
    union_a = gr.array([1, [2, 3]] * max(1, n // 4), dtype=gr.int32)
    nstr = max(1, n // 8)
    str_a = gr.array([[f"s{i}", f"t{i}"] for i in range(nstr)], dtype=gr.string)
    str_b = gr.array([[f"_{i}", f"_{i}"] for i in range(nstr)], dtype=gr.string)
    mat_a = gr.array(rng.random((64, 64)).tolist(), dtype=gr.float64)
    mat_b = gr.array(rng.random((64, 64)).tolist(), dtype=gr.float64)

    np_int = rng.integers(0, 1000, size=n, dtype=np.int32)
    np_flt = rng.random(n)
    np_str_a = np.array([[f"s{i}", f"t{i}"] for i in range(nstr)], dtype=object)
    np_str_b = np.array([[f"_{i}", f"_{i}"] for i in range(nstr)], dtype=object)

    cases = [
        (
            "cast add int+float",
            lambda: (int_a + flt_b).to_list(),
            lambda: (np_int.astype(np.float64) + np_flt).tolist(),
        ),
        (
            "union add",
            lambda: (union_a + union_a).to_list(),
            None,
        ),
        (
            "string concat",
            lambda: (str_a + str_b).to_list(),
            lambda: (np_str_a + np_str_b).tolist(),
        ),
        (
            "string unique",
            lambda: gr.unique(str_a).to_list(),
            lambda: np.unique(np_str_a.ravel()).tolist(),
        ),
        (
            "einsum matmul 64x64",
            lambda: gr.einsum("ij,jk->ik", mat_a, mat_b).to_list(),
            lambda: np.einsum(
                "ij,jk->ik",
                rng.random((64, 64)),
                rng.random((64, 64)),
            ).tolist(),
        ),
    ]

    print(f"Tier 3 benchmarks (n={n}, warmup={args.warmup}, repeats={args.repeats})")
    print("| op | grumpy | numpy | gr/np |")
    print("|---|---:|---:|---:|")
    for name, gr_fn, np_fn in cases:
        t_gr = timeit(gr_fn, warmup=args.warmup, repeats=args.repeats)
        if np_fn is None:
            print(f"| {name} | {fmt_ms(t_gr)} | — | — |")
        else:
            t_np = timeit(np_fn, warmup=args.warmup, repeats=args.repeats)
            ratio = t_gr / t_np if t_np > 0 else float("inf")
            print(f"| {name} | {fmt_ms(t_gr)} | {fmt_ms(t_np)} | {ratio:.2f}× |")


if __name__ == "__main__":
    main()
