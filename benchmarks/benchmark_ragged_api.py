#!/usr/bin/env python3
"""
Primary ragged benchmark — **public API** paths only.

Times the code a user actually writes: ``(a * b).sum(...)``, ``gr.array(...)``,
``gr.isin``, ``x[i, j]``, etc. Use this suite for docs, charts, and performance
claims. NumPy uses rectangular ``(nrows, ncols)``; Grumpy and Awkward use
slightly ragged rows (``ncols±1``) with the same total leaf count.

Engineers: see ``benchmark_ragged_kernels.py`` for fused micro-kernels.
"""

from __future__ import annotations

import argparse
import platform
import sys

import numpy as np

import grumpy as gr

from _bench_common import print_header
from _ragged_bench import (
    BenchReport,
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
    write_json_report,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Ragged benchmark — public API (Grumpy vs NumPy vs Awkward)."
    )
    add_ragged_args(ap)
    args = ap.parse_args(argv)

    ak = require_awkward()
    rng = np.random.default_rng(args.seed)
    ds, build_times = build_dataset(rng, args.nrows, args.ncols, ak)
    rows_i64, cols_i64, test_vals = prepare_indexing_fixtures(
        rng, ds, args.ncols, args.nfancy
    )
    isin_test, gr_isin_test, ak_isin_test = prepare_setops_fixtures(rng, ds, ak)
    ak_flat_a = ak.flatten(ds.ak_a)

    print_header(
        "Grumpy vs NumPy vs Awkward — ragged public API",
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        awkward=ak.__version__,
        extra=header_extra(args, ds.n_elements)
        + [
            "- **Timed region:** idiomatic library calls (includes temporaries)",
            "- Grumpy 2D reductions require explicit ``dim=`` (no bare ``.sum()``)",
        ],
    )

    print_construction_table(build_times)

    # --- elementwise ---
    def np_mul_sum() -> None:
        checksum(int((ds.np_rect_a * ds.np_rect_b).sum(dtype=np.int64)))

    def gr_mul_sum() -> None:
        checksum(int((ds.gr_a * ds.gr_b).sum(dim=1).sum(dim=0)))

    def ak_mul_sum() -> None:
        checksum(int(ak.sum(ds.ak_a * ds.ak_b)))

    def np_add_sum() -> None:
        checksum(int((ds.np_rect_a + ds.np_rect_b).sum(dtype=np.int64)))

    def gr_add_sum() -> None:
        checksum(int((ds.gr_a + ds.gr_b).sum(dim=1).sum(dim=0)))

    def ak_add_sum() -> None:
        checksum(int(ak.sum(ds.ak_a + ds.ak_b)))

    def np_mul_scalar_sum() -> None:
        checksum(int((ds.np_rect_a * 2).sum(dtype=np.int64)))

    def gr_mul_scalar_sum() -> None:
        checksum(int((ds.gr_a * 2).sum(dim=1).sum(dim=0)))

    def ak_mul_scalar_sum() -> None:
        checksum(int(ak.sum(ds.ak_a * 2)))

    # --- reductions ---
    def np_sum_all() -> None:
        checksum(int(ds.np_rect_a.sum(dtype=np.int64)))

    def gr_sum_all() -> None:
        checksum(int(ds.gr_a.sum(dim=1).sum(dim=0)))

    def ak_sum_all() -> None:
        checksum(int(ak.sum(ds.ak_a)))

    def np_sum_axis1() -> None:
        checksum(int(ds.np_rect_a.sum(axis=1, dtype=np.int64).sum()))

    def gr_sum_axis1() -> None:
        checksum(int(ds.gr_a.sum(dim=1).sum(dim=0)))

    def ak_sum_axis1() -> None:
        checksum(int(ak.sum(ak.sum(ds.ak_a, axis=1))))

    def np_mean_axis1() -> None:
        checksum(float(ds.np_rect_a.mean(axis=1).sum()))

    def gr_mean_axis1() -> None:
        checksum(float(ds.gr_a.mean(dim=1).sum(dim=0)))

    def ak_mean_axis1() -> None:
        checksum(float(ak.sum(ak.mean(ds.ak_a, axis=1))))

    # --- set routines ---
    def np_isin() -> None:
        _ = np.isin(ds.np_rect_a, isin_test)
        checksum(1)

    def gr_isin() -> None:
        _ = gr.isin(ds.gr_a, gr_isin_test)
        checksum(1)

    def ak_isin() -> None:
        _ = np.isin(ak_flat_a, isin_test)
        checksum(1)

    def np_unique_len() -> None:
        checksum(len(np.unique(ds.np_rect_a)))

    def gr_unique_len() -> None:
        checksum(len(gr.unique(ds.gr_a).to_list()))

    def ak_unique_len() -> None:
        checksum(len(np.unique(ak_flat_a)))

    # --- indexing ---
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
        checksum(int(ds.gr_a[rows_i64, cols_i64].sum(dim=0)))

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
        x[rows_i64, cols_i64] = test_vals
        checksum(int(x[0, 0]))

    def ak_set_fancy() -> None:
        py = ak.to_list(ds.ak_a)
        for r, c, v in zip(rows_i64, cols_i64, test_vals):
            py[int(r)][int(c)] = int(v)
        checksum(int(ak.Array(py)[0][0]))

    cases = run_timed_cases(
        [
            (
                "Elementwise",
                "(a * b).sum()",
                "int((a * b).sum())",
                "int((a * b).sum(dim=1).sum(dim=0))",
                "int(ak.sum(a * b))",
                np_mul_sum,
                gr_mul_sum,
                ak_mul_sum,
            ),
            (
                "Elementwise",
                "(a + b).sum()",
                "int((a + b).sum())",
                "int((a + b).sum(dim=1).sum(dim=0))",
                "int(ak.sum(a + b))",
                np_add_sum,
                gr_add_sum,
                ak_add_sum,
            ),
            (
                "Elementwise",
                "(a * 2).sum()",
                "int((a * 2).sum())",
                "int((a * 2).sum(dim=1).sum(dim=0))",
                "int(ak.sum(a * 2))",
                np_mul_scalar_sum,
                gr_mul_scalar_sum,
                ak_mul_scalar_sum,
            ),
            (
                "Reductions",
                "a.sum()",
                "int(a.sum())",
                "int(a.sum(dim=1).sum(dim=0))",
                "int(ak.sum(a))",
                np_sum_all,
                gr_sum_all,
                ak_sum_all,
            ),
            (
                "Reductions",
                "a.sum(axis=1).sum()",
                "int(a.sum(axis=1).sum())",
                "int(a.sum(dim=1).sum(dim=0))",
                "int(ak.sum(ak.sum(a, axis=1)))",
                np_sum_axis1,
                gr_sum_axis1,
                ak_sum_axis1,
            ),
            (
                "Reductions",
                "a.mean(axis=1).sum()",
                "float(a.mean(axis=1).sum())",
                "float(a.mean(dim=1).sum(dim=0))",
                "float(ak.sum(ak.mean(a, axis=1)))",
                np_mean_axis1,
                gr_mean_axis1,
                ak_mean_axis1,
            ),
            (
                "Set routines",
                "isin",
                "np.isin(a, test)",
                "gr.isin(a, test)",
                "np.isin(ak.flatten(a), test)",
                np_isin,
                gr_isin,
                ak_isin,
            ),
            (
                "Set routines",
                "len(unique(a))",
                "len(np.unique(a))",
                "len(gr.unique(a).to_list())",
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
                "fancy get + sum",
                "int(a[rows, cols].sum())",
                "int(a[rows, cols].sum(dim=0))",
                "int(ak.sum(a[rows, cols]))",
                np_get_fancy,
                gr_get_fancy,
                ak_get_fancy,
            ),
            (
                "Indexing",
                "scalar set loop",
                "x = a.copy(); x[r, c] = 7",
                "x = a.copy(); x[r, c] = 7",
                "py = ak.to_list(a); ...; ak.Array(py)",
                np_set_scalar,
                gr_set_scalar,
                ak_set_scalar,
            ),
            (
                "Indexing",
                "fancy set",
                "x = a.copy(); x[rows, cols] = vals",
                "x = a.copy(); x[rows, cols] = vals",
                "py = ak.to_list(a); ...; ak.Array(py)",
                np_set_fancy,
                gr_set_fancy,
                ak_set_fancy,
            ),
        ],
        warmup=args.warmup,
        repeats=args.repeats,
    )

    print_cases_by_category(cases)

    print("### Methodology")
    print()
    print("- Construction is timed **once** outside the tables above.")
    print("- Every timed call uses the **public API** shown in the expression tables.")
    print("- Intermediate arrays (e.g. ``a * b``) are **included** — this matches interactive use.")
    print("- NumPy baseline is rectangular with the same leaf count; Grumpy/Awkward are ragged ±1.")
    print("- Awkward **mutation** uses ``to_list`` → edit → ``ak.Array`` (arrays are immutable).")
    print("- For docs/charts: pass ``--json results.json`` for machine-readable output.")
    print()

    if args.json:
        report = BenchReport(
            suite="ragged_public_api",
            python=sys.version.split()[0],
            numpy=np.__version__,
            awkward=ak.__version__,
            platform=platform.platform(),
            nrows=args.nrows,
            ncols=args.ncols,
            n_elements=ds.n_elements,
            nfancy=args.nfancy,
            warmup=args.warmup,
            repeats=args.repeats,
            construction=build_times,
            cases=cases,
        )
        write_json_report(args.json, report)
        print(f"Wrote JSON report to `{args.json}`")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
