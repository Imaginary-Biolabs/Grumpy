#!/usr/bin/env python3
"""
Secondary ragged benchmark — **internal fused kernels** (engineers only).

Measures Rust micro-kernels (``_mul2d_i32_sum_i64``, etc.) that skip intermediate
array allocation. These are **not** the public Python API and are **not** suitable
for docs or user-facing performance claims.

For publishable numbers use ``benchmark_ragged_api.py``.
"""

from __future__ import annotations

import argparse
import platform
import sys

import numpy as np

import grumpy as gr

from _bench_common import print_header
from _ragged_bench import (
    add_ragged_args,
    build_dataset,
    checksum,
    header_extra,
    prepare_indexing_fixtures,
    prepare_setops_fixtures,
    print_cases_by_category,
    print_construction_table,
    require_awkward,
    run_timed_cases,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ragged benchmark — internal fused kernels (engineers)."
    )
    add_ragged_args(ap)
    args = ap.parse_args(argv)

    ak = require_awkward()
    rng = np.random.default_rng(args.seed)
    ds, build_times = build_dataset(rng, args.nrows, args.ncols, ak)
    rows_i64, cols_i64, test_vals = prepare_indexing_fixtures(
        rng, ds, args.ncols, args.nfancy
    )
    isin_test, gr_isin_test, _ak_isin_test = prepare_setops_fixtures(rng, ds, ak)

    np_tmp = np.empty_like(ds.np_rect_a)
    gr_flat_a = gr.array(ds.flat_a.tolist(), dtype=gr.int32)
    ak_flat_a = ak.flatten(ds.ak_a)

    print_header(
        "Grumpy vs NumPy vs Awkward — ragged fused kernels (internal)",
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        awkward=ak.__version__,
        extra=header_extra(args, ds.n_elements)
        + [
            "- **Not for docs:** private ``_mul2d_*`` helpers and preallocated NumPy buffers",
            "- Use ``benchmark_ragged_api.py`` for user-facing comparisons",
        ],
    )

    print_construction_table(build_times)

    def np_mul_sum() -> None:
        np.multiply(ds.np_rect_a, ds.np_rect_b, out=np_tmp)
        checksum(int(np_tmp.sum()))

    def gr_mul_sum() -> None:
        checksum(ds.gr_a._mul2d_i32_sum_i64(ds.gr_b))

    def ak_mul_sum() -> None:
        checksum(int(ak.sum(ds.ak_a * ds.ak_b)))

    def np_add_sum() -> None:
        np.add(ds.np_rect_a, ds.np_rect_b, out=np_tmp)
        checksum(int(np_tmp.sum()))

    def gr_add_sum() -> None:
        checksum(ds.gr_a._add2d_i32_sum_i64(ds.gr_b))

    def ak_add_sum() -> None:
        checksum(int(ak.sum(ds.ak_a + ds.ak_b)))

    def np_mul_scalar_sum() -> None:
        np.multiply(ds.np_rect_a, 2, out=np_tmp)
        checksum(int(np_tmp.sum()))

    def gr_mul_scalar_sum() -> None:
        checksum((ds.gr_a * 2)._sum2d_i32_i64())

    def ak_mul_scalar_sum() -> None:
        checksum(int(ak.sum(ds.ak_a * 2)))

    def np_mul_op_sum() -> None:
        checksum(int((ds.np_rect_a * ds.np_rect_b).sum(dtype=np.int64)))

    def gr_mul_op_sum() -> None:
        checksum(ds.gr_a._mul2d_i32_sum_via_op_i64(ds.gr_b))

    def ak_mul_op_sum() -> None:
        checksum(int(ak.sum(ds.ak_a * ds.ak_b)))

    def np_sum_all() -> None:
        checksum(int(ds.np_rect_a.sum(dtype=np.int64)))

    def gr_sum_all() -> None:
        checksum(ds.gr_a._sum2d_i32_i64())

    def ak_sum_all() -> None:
        checksum(int(ak.sum(ds.ak_a)))

    def np_sum_axis1() -> None:
        checksum(int(ds.np_rect_a.sum(axis=1, dtype=np.int64).sum()))

    def gr_sum_axis1() -> None:
        checksum(ds.gr_a._sum2d_dim1_i32_sum_i64())

    def ak_sum_axis1() -> None:
        checksum(int(ak.sum(ak.sum(ds.ak_a, axis=1))))

    def np_mean_axis1() -> None:
        checksum(float(ds.np_rect_a.mean(axis=1).sum()))

    def gr_mean_axis1() -> None:
        checksum(float(ds.gr_a._mean2d_dim1_i32_sum_f64()))

    def ak_mean_axis1() -> None:
        checksum(float(ak.sum(ak.mean(ds.ak_a, axis=1))))

    def np_isin() -> None:
        _ = np.isin(ds.flat_a, isin_test)
        checksum(1)

    def gr_isin() -> None:
        _ = gr.isin(gr_flat_a, gr_isin_test)
        checksum(1)

    def ak_isin() -> None:
        _ = np.isin(ak_flat_a, isin_test)
        checksum(1)

    def np_unique_len() -> None:
        checksum(len(np.unique(ds.flat_a)))

    def gr_unique_len() -> None:
        checksum(len(gr.unique(gr_flat_a).to_list()))

    def ak_unique_len() -> None:
        checksum(len(np.unique(ak_flat_a)))

    def np_get_scalar() -> None:
        s = sum(int(ds.np_rect_a[int(r), int(c)]) for r, c in zip(rows_i64, cols_i64))
        checksum(s)

    def gr_get_scalar() -> None:
        s = sum(int(ds.gr_a[int(r), int(c)]) for r, c in zip(rows_i64, cols_i64))
        checksum(s)

    def ak_get_scalar() -> None:
        s = sum(int(ds.ak_a[int(r), int(c)]) for r, c in zip(rows_i64, cols_i64))
        checksum(s)

    def np_get_fancy() -> None:
        checksum(int(ds.np_rect_a[rows_i64, cols_i64].sum()))

    def gr_get_fancy() -> None:
        checksum(ds.gr_a._gather2d_sum_i64(rows_i64, cols_i64))

    def ak_get_fancy() -> None:
        checksum(int(ak.sum(ds.ak_a[rows_i64, cols_i64])))

    def np_set_scalar() -> None:
        x = ds.np_rect_a.copy()
        for r, c in zip(rows_i64, cols_i64):
            x[int(r), int(c)] = 7
        checksum(int(x[0, 0]))

    def gr_set_scalar() -> None:
        x = ds.gr_a.copy()
        for r, c in zip(rows_i64, cols_i64):
            x[int(r), int(c)] = 7
        checksum(int(x[0, 0]))

    def ak_set_scalar() -> None:
        py = ak.to_list(ds.ak_a)
        for r, c in zip(rows_i64, cols_i64):
            py[int(r)][int(c)] = 7
        checksum(int(ak.Array(py)[0][0]))

    def np_set_fancy() -> None:
        x = ds.np_rect_a.copy()
        x[rows_i64, cols_i64] = test_vals
        checksum(int(x[0, 0]))

    def gr_set_fancy() -> None:
        x = ds.gr_a.copy()
        x._scatter2d_i32(rows_i64, cols_i64, test_vals)
        checksum(int(x[0, 0]))

    def ak_set_fancy() -> None:
        py = ak.to_list(ds.ak_a)
        for r, c, v in zip(rows_i64, cols_i64, test_vals):
            py[int(r)][int(c)] = int(v)
        checksum(int(ak.Array(py)[0][0]))

    cases = run_timed_cases(
        [
            (
                "Elementwise (fused / ufunc)",
                "mul + sum (fused gr)",
                "np.multiply(a, b, out=tmp); tmp.sum()",
                "a._mul2d_i32_sum_i64(b)",
                "ak.sum(a * b)",
                np_mul_sum,
                gr_mul_sum,
                ak_mul_sum,
            ),
            (
                "Elementwise (fused / ufunc)",
                "add + sum (fused gr)",
                "np.add(a, b, out=tmp); tmp.sum()",
                "a._add2d_i32_sum_i64(b)",
                "ak.sum(a + b)",
                np_add_sum,
                gr_add_sum,
                ak_add_sum,
            ),
            (
                "Elementwise (fused / ufunc)",
                "mul scalar + sum",
                "np.multiply(a, 2, out=tmp); tmp.sum()",
                "(a * 2)._sum2d_i32_i64()",
                "ak.sum(a * 2)",
                np_mul_scalar_sum,
                gr_mul_scalar_sum,
                ak_mul_scalar_sum,
            ),
            (
                "Elementwise (fused / ufunc)",
                "mul via elementwise + sum",
                "int((a * b).sum())",
                "a._mul2d_i32_sum_via_op_i64(b)",
                "ak.sum(a * b)",
                np_mul_op_sum,
                gr_mul_op_sum,
                ak_mul_op_sum,
            ),
            (
                "Reductions (fused gr)",
                "sum all",
                "int(a.sum())",
                "a._sum2d_i32_i64()",
                "ak.sum(a)",
                np_sum_all,
                gr_sum_all,
                ak_sum_all,
            ),
            (
                "Reductions (fused gr)",
                "sum axis=1",
                "a.sum(axis=1).sum()",
                "a._sum2d_dim1_i32_sum_i64()",
                "ak.sum(ak.sum(a, axis=1))",
                np_sum_axis1,
                gr_sum_axis1,
                ak_sum_axis1,
            ),
            (
                "Reductions (fused gr)",
                "mean axis=1",
                "a.mean(axis=1).sum()",
                "a._mean2d_dim1_i32_sum_f64()",
                "ak.sum(ak.mean(a, axis=1))",
                np_mean_axis1,
                gr_mean_axis1,
                ak_mean_axis1,
            ),
            (
                "Set routines",
                "isin (1D flat gr)",
                "np.isin(flat, test)",
                "gr.isin(flat, test)",
                "np.isin(ak.flatten(a), test)",
                np_isin,
                gr_isin,
                ak_isin,
            ),
            (
                "Set routines",
                "unique len (1D flat gr)",
                "len(np.unique(flat))",
                "len(gr.unique(flat).to_list())",
                "len(np.unique(ak.flatten(a)))",
                np_unique_len,
                gr_unique_len,
                ak_unique_len,
            ),
            (
                "Indexing",
                "scalar get loop",
                "sum(a[r, c] for ...)",
                "sum(a[r, c] for ...)",
                "sum(a[r, c] for ...)",
                np_get_scalar,
                gr_get_scalar,
                ak_get_scalar,
            ),
            (
                "Indexing",
                "fancy get (fused gr)",
                "a[rows, cols].sum()",
                "a._gather2d_sum_i64(rows, cols)",
                "ak.sum(a[rows, cols])",
                np_get_fancy,
                gr_get_fancy,
                ak_get_fancy,
            ),
            (
                "Indexing",
                "scalar set loop",
                "x = a.copy(); x[r, c] = 7",
                "x = a.copy(); x[r, c] = 7",
                "py = ak.to_list(a); ...",
                np_set_scalar,
                gr_set_scalar,
                ak_set_scalar,
            ),
            (
                "Indexing",
                "fancy set (fused gr)",
                "x[rows, cols] = vals",
                "x._scatter2d_i32(rows, cols, vals)",
                "py = ak.to_list(a); ...",
                np_set_fancy,
                gr_set_fancy,
                ak_set_fancy,
            ),
        ],
        warmup=args.warmup,
        repeats=args.repeats,
    )

    print_cases_by_category(cases)

    print("### Notes")
    print()
    print("- Grumpy ``_*`` methods are private; they exist to measure kernel throughput.")
    print("- NumPy ``out=`` buffers are preallocated outside the timed region.")
    print("- Awkward set paths still use ``to_list`` round-trips (immutable arrays).")
    print("- **Do not cite these numbers in user-facing docs.**")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
