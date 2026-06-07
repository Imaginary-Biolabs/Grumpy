#!/usr/bin/env python3
"""
Compare ``gr.load`` vs ``gr.open`` indexing for batched dataset access.

Measures full materialization, per-batch slice epochs (in-memory vs open vs
``load_slice``), and single-row materialization. Reports I/O bytes where useful.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

import grumpy as gr

from _bench_common import timeit
from _open_epoch import (
    consecutive_batch_windows,
    epoch_in_memory_batched,
    epoch_in_memory_windows,
    epoch_load_slice_batched,
    epoch_open_batched,
    epoch_open_load_only,
    epoch_open_windows,
    random_straddling_batch_windows,
)
from benchmark_compile_suite import _protein_dataframe

DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 3


@dataclass
class LoadOpenCase:
    name: str
    ms: float
    io_bytes: int | None = None
    notes: str = ""


@dataclass
class LoadOpenReport:
    suite: str
    python: str
    numpy: str
    platform: str
    n_molecules: int
    n_residues: int
    batch_size: int
    n_batches: int
    warmup: int
    repeats: int
    wall_time_s: float = 0.0
    cases: list[LoadOpenCase] = field(default_factory=list)


def _timed_ms(fn: Callable[[], None], *, warmup: int, repeats: int) -> float:
    return timeit(fn, warmup=warmup, repeats=repeats) * 1e3


def _io_reset() -> None:
    gr._core.reset_io_bytes_read()
    gr._core.clear_path_caches()


def _io_read() -> int:
    return gr._core.io_bytes_read()


def residue_center(batch):
    batch.residue.residue_center = batch.residue.atom_pos0.mean(dim=1)
    return batch


def build_cases(
    path: str,
    df_loaded: gr.GrumpyDataFrame,
    *,
    n_molecules: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    rng: np.random.Generator,
) -> list[LoadOpenCase]:
    cases: list[LoadOpenCase] = []

    def add(name: str, ms: float, *, io_bytes: int | None = None, notes: str = "") -> None:
        cases.append(LoadOpenCase(name=name, ms=ms, io_bytes=io_bytes, notes=notes))

    # --- one-shot full load ---
    _io_reset()
    load_ms = _timed_ms(lambda: gr.load(path), warmup=warmup, repeats=repeats)
    load_io = _io_read()
    add("gr.load (full dataset)", load_ms, io_bytes=load_io)

    _io_reset()
    open_load_ms = _timed_ms(
        lambda: gr.open(path).load(),
        warmup=warmup,
        repeats=repeats,
    )
    open_load_io = _io_read()
    add("gr.open().load()", open_load_ms, io_bytes=open_load_io, notes="open handle + full materialize")

    # --- batched epoch: load once, slice in RAM ---
    def load_once_slice_epoch() -> None:
        df = gr.load(path)
        epoch_in_memory_batched(df, residue_center, n_molecules=n_molecules, batch_size=batch_size)

    _io_reset()
    mem_epoch_ms = _timed_ms(load_once_slice_epoch, warmup=warmup, repeats=repeats)
    mem_epoch_io = _io_read()
    add(
        "load once + batched slice + transform",
        mem_epoch_ms,
        io_bytes=mem_epoch_io,
        notes="amortized: one gr.load per epoch, in-memory axis-0 slices",
    )

    # --- batched epoch: open + materialize each batch ---
    def open_epoch_transform() -> None:
        epoch_open_batched(
            path,
            residue_center,
            n_molecules=n_molecules,
            batch_size=batch_size,
        )

    _io_reset()
    open_epoch_ms = _timed_ms(open_epoch_transform, warmup=warmup, repeats=repeats)
    open_epoch_io = _io_read()
    add(
        "gr.open + batched index + transform",
        open_epoch_ms,
        io_bytes=open_epoch_io,
        notes="shared handle; each batch materializes subset from Zarr",
    )

    def open_epoch_load_only() -> None:
        epoch_open_load_only(path, n_molecules=n_molecules, batch_size=batch_size)

    _io_reset()
    open_load_only_ms = _timed_ms(open_epoch_load_only, warmup=warmup, repeats=repeats)
    open_load_only_io = _io_read()
    add(
        "gr.open + batched index (load only)",
        open_load_only_ms,
        io_bytes=open_load_only_io,
    )

    # --- load_slice per batch (new handle each call) ---
    def load_slice_epoch() -> None:
        epoch_load_slice_batched(
            path,
            residue_center,
            n_molecules=n_molecules,
            batch_size=batch_size,
        )

    _io_reset()
    slice_epoch_ms = _timed_ms(load_slice_epoch, warmup=warmup, repeats=repeats)
    slice_epoch_io = _io_read()
    add(
        "load_slice per batch + transform",
        slice_epoch_ms,
        io_bytes=slice_epoch_io,
        notes="gr._core.load_slice opens path each batch",
    )

    # --- single-row access (random access pattern) ---
    _io_reset()
    single_open_ms = _timed_ms(
        lambda: gr.open(path)[0],
        warmup=warmup,
        repeats=max(repeats, 5),
    )
    single_open_io = _io_read()
    add("gr.open[path][0] (single row)", single_open_ms, io_bytes=single_open_io)

    _io_reset()
    single_loaded_ms = _timed_ms(lambda: df_loaded[0], warmup=warmup, repeats=max(repeats, 5))
    add("loaded_df[0] (single row)", single_loaded_ms, notes="in-memory index")

    # --- indexing pattern comparison (consecutive vs random straddling) ---
    consecutive = consecutive_batch_windows(n_molecules, batch_size)
    random_straddle = random_straddling_batch_windows(n_molecules, batch_size, rng)
    n_consecutive = len(consecutive)
    n_random = len(random_straddle)

    def load_consecutive_epoch() -> None:
        df = gr.load(path)
        epoch_in_memory_windows(df, residue_center, consecutive)

    _io_reset()
    load_consecutive_ms = _timed_ms(load_consecutive_epoch, warmup=warmup, repeats=repeats)
    load_consecutive_io = _io_read()
    add(
        "gr.load + consecutive batches + transform",
        load_consecutive_ms,
        io_bytes=load_consecutive_io,
        notes=f"{n_consecutive} chunk-aligned windows in order",
    )

    def load_random_epoch() -> None:
        df = gr.load(path)
        epoch_in_memory_windows(df, residue_center, random_straddle)

    _io_reset()
    load_random_ms = _timed_ms(load_random_epoch, warmup=warmup, repeats=repeats)
    load_random_io = _io_read()
    add(
        "gr.load + random straddling batches + transform",
        load_random_ms,
        io_bytes=load_random_io,
        notes=f"{n_random} half-chunk-offset windows, shuffled order",
    )

    def open_consecutive_epoch() -> None:
        epoch_open_windows(path, residue_center, consecutive)

    _io_reset()
    open_consecutive_ms = _timed_ms(open_consecutive_epoch, warmup=warmup, repeats=repeats)
    open_consecutive_io = _io_read()
    add(
        "gr.open + consecutive batches + transform",
        open_consecutive_ms,
        io_bytes=open_consecutive_io,
        notes=f"{n_consecutive} chunk-aligned windows in order",
    )

    def open_consecutive_load_only() -> None:
        epoch_open_windows(path, None, consecutive)

    _io_reset()
    open_consecutive_load_ms = _timed_ms(open_consecutive_load_only, warmup=warmup, repeats=repeats)
    open_consecutive_load_io = _io_read()
    add(
        "gr.open + consecutive batches (load only)",
        open_consecutive_load_ms,
        io_bytes=open_consecutive_load_io,
    )

    def open_random_epoch() -> None:
        epoch_open_windows(path, residue_center, random_straddle)

    _io_reset()
    open_random_ms = _timed_ms(open_random_epoch, warmup=warmup, repeats=repeats)
    open_random_io = _io_read()
    add(
        "gr.open + random straddling batches + transform",
        open_random_ms,
        io_bytes=open_random_io,
        notes=f"{n_random} half-chunk-offset windows, shuffled order",
    )

    def open_random_load_only() -> None:
        epoch_open_windows(path, None, random_straddle)

    _io_reset()
    open_random_load_ms = _timed_ms(open_random_load_only, warmup=warmup, repeats=repeats)
    open_random_load_io = _io_read()
    add(
        "gr.open + random straddling batches (load only)",
        open_random_load_ms,
        io_bytes=open_random_load_io,
    )

    return cases


def print_report(report: LoadOpenReport) -> None:
    load_ms = next((c.ms for c in report.cases if c.name == "gr.load (full dataset)"), None)
    open_epoch_ms = next(
        (c.ms for c in report.cases if c.name == "gr.open + batched index + transform"),
        None,
    )
    mem_epoch_ms = next(
        (c.ms for c in report.cases if c.name == "load once + batched slice + transform"),
        None,
    )
    open_load_only_ms = next(
        (c.ms for c in report.cases if c.name == "gr.open + batched index (load only)"),
        None,
    )

    print("## gr.load vs gr.open indexing\n")
    print(f"- python: {report.python}")
    print(f"- numpy: {report.numpy}")
    print(f"- platform: {report.platform}")
    print(
        f"- n_molecules={report.n_molecules}, n_residues={report.n_residues}, "
        f"batch_size={report.batch_size}, n_batches={report.n_batches}, "
        f"warmup={report.warmup}, repeats={report.repeats}, wall={report.wall_time_s:.1f}s\n"
    )

    print("| case | ms | I/O bytes | notes |")
    print("|---|---:|---:|---|")
    for c in report.cases:
        io = "—" if c.io_bytes is None else f"{c.io_bytes:,}"
        notes = c.notes.replace("|", "\\|") if c.notes else ""
        print(f"| {c.name} | {c.ms:.1f} | {io} | {notes} |")

    print("\n### Summary\n")
    if load_ms and open_epoch_ms:
        print(f"- **open batched epoch / full load:** {open_epoch_ms / load_ms:.2f}× slower than one `gr.load`")
    if mem_epoch_ms and open_epoch_ms:
        print(f"- **open batched epoch / load-once slice epoch:** {open_epoch_ms / mem_epoch_ms:.2f}× slower")
    if open_load_only_ms and open_epoch_ms:
        transform_ms = max(0.0, open_epoch_ms - open_load_only_ms)
        pct = 100.0 * open_load_only_ms / open_epoch_ms if open_epoch_ms else 0.0
        print(f"- **open epoch I/O share:** {pct:.0f}% ({open_load_only_ms:.1f} ms load / {open_epoch_ms:.1f} ms total)")
        print(f"- **open epoch transform share:** {transform_ms:.1f} ms")
    if load_ms and open_load_only_ms:
        print(f"- **open load-only epoch / full load:** {open_load_only_ms / load_ms:.2f}×")

    indexing_cases = {
        "consecutive": (
            "gr.load + consecutive batches + transform",
            "gr.open + consecutive batches + transform",
            "gr.open + consecutive batches (load only)",
        ),
        "random": (
            "gr.load + random straddling batches + transform",
            "gr.open + random straddling batches + transform",
            "gr.open + random straddling batches (load only)",
        ),
    }
    by_name = {c.name: c for c in report.cases}
    if all(name in by_name for names in indexing_cases.values() for name in names):
        print("\n### Indexing patterns vs gr.load\n")
        print("| pattern | gr.load + batches | gr.open + batches | vs gr.load | open load-only | vs gr.load |")
        print("|---|---:|---:|---:|---:|---:|")
        for label, (load_name, open_name, open_load_name) in indexing_cases.items():
            load_case = by_name[load_name]
            open_case = by_name[open_name]
            open_load_case = by_name[open_load_name]
            load_base = load_ms or load_case.ms
            print(
                f"| {label} | {load_case.ms:.1f} | {open_case.ms:.1f} | "
                f"{open_case.ms / load_base:.2f}× | {open_load_case.ms:.1f} | "
                f"{open_load_case.ms / load_base:.2f}× |"
            )
        consec_open = by_name["gr.open + consecutive batches + transform"].ms
        random_open = by_name["gr.open + random straddling batches + transform"].ms
        print(f"\n- **random / consecutive (gr.open):** {random_open / consec_open:.2f}× slower")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gr.load vs gr.open indexing benchmark.")
    ap.add_argument("--n-molecules", type=int, default=256)
    ap.add_argument("--n-residues", type=int, default=128)
    ap.add_argument("--atoms-per-res", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", metavar="PATH", default=None)
    args = ap.parse_args(argv)

    wall_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    n_batches = (args.n_molecules + args.batch_size - 1) // args.batch_size

    with tempfile.TemporaryDirectory(prefix="grumpy_load_open_bench_") as tmp:
        path = str(Path(tmp) / "proteins.gr")
        df = _protein_dataframe(rng, args.n_molecules, args.n_residues, args.atoms_per_res)
        gr.save(df, path, chunk_size=args.batch_size)
        df_loaded = gr.load(path)

        cases = build_cases(
            path,
            df_loaded,
            n_molecules=args.n_molecules,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            rng=rng,
        )

    report = LoadOpenReport(
        suite="load_vs_open",
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        n_molecules=args.n_molecules,
        n_residues=args.n_residues,
        batch_size=args.batch_size,
        n_batches=n_batches,
        warmup=args.warmup,
        repeats=args.repeats,
        wall_time_s=time.perf_counter() - wall_start,
        cases=cases,
    )

    print_report(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
