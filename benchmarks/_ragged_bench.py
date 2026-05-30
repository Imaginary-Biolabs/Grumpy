"""Shared dataset setup and helpers for ragged Grumpy vs NumPy vs Awkward benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import numpy as np

import grumpy as gr

from _bench_common import (
    RaggedDataset,
    fmt_ms,
    make_slightly_ragged_lists,
    make_valid_index_pairs,
    print_header,
    print_ratio_table,
    timeit,
    try_import_awkward,
)


def add_ragged_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--nrows", type=int, default=4096)
    ap.add_argument("--ncols", type=int, default=256)
    ap.add_argument("--nfancy", type=int, default=4096)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Write machine-readable results (API benchmark only).",
    )


def checksum(v: int | float) -> None:
    if v == -1:
        raise RuntimeError("checksum sentinel")


@dataclass
class BenchCase:
    name: str
    category: str
    numpy_ms: float
    grumpy_ms: float
    awkward_ms: float
    numpy_code: str = ""
    grumpy_code: str = ""
    awkward_code: str = ""


@dataclass
class BenchReport:
    suite: str
    python: str
    numpy: str
    awkward: str
    platform: str
    nrows: int
    ncols: int
    n_elements: int
    nfancy: int
    warmup: int
    repeats: int
    construction: dict[str, float] = field(default_factory=dict)
    cases: list[BenchCase] = field(default_factory=list)


def build_dataset(rng, nrows: int, ncols: int, ak) -> tuple[RaggedDataset, dict[str, float]]:
    t0 = time.perf_counter()
    lists = make_slightly_ragged_lists(rng, nrows, ncols)
    t_lists = time.perf_counter() - t0

    t0 = time.perf_counter()
    gr_a = gr.array(lists.ragged_a, dtype=gr.int32)
    gr_b = gr.array(lists.ragged_b, dtype=gr.int32)
    t_build_gr = time.perf_counter() - t0

    t0 = time.perf_counter()
    ak_a = ak.Array(lists.ragged_a)
    ak_b = ak.Array(lists.ragged_b)
    t_build_ak = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = lists.np_rect_a.copy()
    _ = lists.np_rect_b.copy()
    t_build_np = time.perf_counter() - t0

    ds = RaggedDataset(
        nrows=lists.nrows,
        ncols=lists.ncols,
        n_elements=lists.n_elements,
        ragged_a=lists.ragged_a,
        ragged_b=lists.ragged_b,
        flat_a=lists.flat_a,
        flat_b=lists.flat_b,
        np_rect_a=lists.np_rect_a,
        np_rect_b=lists.np_rect_b,
        gr_a=gr_a,
        gr_b=gr_b,
        ak_a=ak_a,
        ak_b=ak_b,
    )
    return ds, {
        "python_lists_ms": t_lists * 1e3,
        "grumpy_array_ms": t_build_gr * 1e3,
        "awkward_array_ms": t_build_ak * 1e3,
        "numpy_copy_ms": t_build_np * 1e3,
    }


def print_construction_table(build_times: dict[str, float]) -> None:
    print("### Construction (pre-built nested lists; two arrays each)")
    print()
    print("| phase | time |")
    print("|---|---:|")
    print(f"| python_lists (NumPy→`.tolist()` × 2) | {fmt_ms(build_times['python_lists_ms'] / 1e3)} |")
    print(f"| grumpy `gr.array` × 2 | {fmt_ms(build_times['grumpy_array_ms'] / 1e3)} |")
    print(f"| awkward `ak.Array` × 2 | {fmt_ms(build_times['awkward_array_ms'] / 1e3)} |")
    print(f"| numpy rect copy × 2 | {fmt_ms(build_times['numpy_copy_ms'] / 1e3)} |")
    print()


def run_timed_cases(
    cases: list[tuple[str, str, str, str, Callable[[], None], Callable[[], None], Callable[[], None]]],
    *,
    warmup: int,
    repeats: int,
) -> list[BenchCase]:
    out: list[BenchCase] = []
    for category, name, np_code, gr_code, ak_code, np_fn, gr_fn, ak_fn in cases:
        out.append(
            BenchCase(
                name=name,
                category=category,
                numpy_ms=timeit(np_fn, warmup=warmup, repeats=repeats) * 1e3,
                grumpy_ms=timeit(gr_fn, warmup=warmup, repeats=repeats) * 1e3,
                awkward_ms=timeit(ak_fn, warmup=warmup, repeats=repeats) * 1e3,
                numpy_code=np_code,
                grumpy_code=gr_code,
                awkward_code=ak_code,
            )
        )
    return out


def print_cases_by_category(cases: list[BenchCase]) -> None:
    categories: list[str] = []
    for c in cases:
        if c.category not in categories:
            categories.append(c.category)
    for category in categories:
        rows = [
            (c.name, c.numpy_ms / 1e3, c.grumpy_ms / 1e3, c.awkward_ms / 1e3)
            for c in cases
            if c.category == category
        ]
        print_ratio_table(category, rows)
        codes = [c for c in cases if c.category == category]
        if codes and codes[0].numpy_code:
            print(f"#### {category} — timed expressions")
            print()
            print("| op | numpy | grumpy | awkward |")
            print("|---|---|---|---|")
            for c in codes:
                print(f"| {c.name} | `{c.numpy_code}` | `{c.grumpy_code}` | `{c.awkward_code}` |")
            print()


def write_json_report(path: str, report: BenchReport) -> None:
    payload: dict[str, Any] = asdict(report)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def require_awkward():
    ak = try_import_awkward()
    if ak is None:
        print("awkward is required: pip install -e '.[benchmark]'", file=sys.stderr)
        raise SystemExit(1)
    return ak


def header_extra(args, n_elem: int) -> list[str]:
    return [
        f"- nrows: {args.nrows}, ncols (nominal): {args.ncols}, n_elements: {n_elem}",
        f"- nfancy: {args.nfancy}, warmup: {args.warmup}, repeats: {args.repeats}",
        "- NumPy: rectangular `(nrows, ncols)`; Grumpy/Awkward: alternating row length `ncols±1`",
        "- Same total leaf count (`nrows×ncols`) and flat value order across libraries",
    ]


def prepare_indexing_fixtures(rng, ds: RaggedDataset, ncols: int, nfancy: int):
    rows_i64, cols_i64 = make_valid_index_pairs(rng, ds.nrows, ncols, nfancy)
    test_vals = rng.integers(0, 1_000_000, size=nfancy, dtype=np.int32)
    return rows_i64, cols_i64, test_vals


def prepare_setops_fixtures(rng, ds: RaggedDataset, ak):
    n_elem = ds.n_elements
    isin_test = rng.integers(0, 1_000_000, size=max(1, n_elem // 10), dtype=np.int32)
    gr_isin_test = gr.array(isin_test.tolist(), dtype=gr.int32)
    ak_isin_test = ak.Array(isin_test.tolist())
    return isin_test, gr_isin_test, ak_isin_test
