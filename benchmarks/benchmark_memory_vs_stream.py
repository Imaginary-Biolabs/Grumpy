#!/usr/bin/env python3
"""
In-memory vs ``gr.open`` indexing benchmark.

Compares the same batch transforms over:
  - a dataset resident in RAM (manual axis-0 batch slices), and
  - ``gr.open`` batched indexing from a saved Zarr store.

Also times open **load-only** (no transform) to separate I/O from compute.
Defaults complete in **< 60 s** (256 structures × 96 residues, batch_size=32).
"""

from __future__ import annotations

import argparse
import json
import platform
import signal
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np

import grumpy as gr

from _bench_common import timeit
from _open_epoch import epoch_in_memory_batched, epoch_open_batched, epoch_open_load_only
from benchmark_compile_suite import _protein_coords, _protein_dataframe

DEFAULT_SUITE_BUDGET_S = 55.0
DEFAULT_CASE_TIMEOUT_S = 8.0

T = TypeVar("T")


class BenchTimeout(Exception):
    """Raised when a timed call exceeds its limit."""


@dataclass
class MemoryOpenCase:
    name: str
    in_memory_ms: float | None
    open_transform_ms: float | None
    open_load_only_ms: float | None
    grumpy_code: str = ""
    notes: str = ""


@dataclass
class MemoryOpenReport:
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
    suite_budget_s: float
    case_timeout_s: float
    wall_time_s: float = 0.0
    cases: list[MemoryOpenCase] = field(default_factory=list)


def _alarm_handler(_signum: int, _frame) -> None:
    raise BenchTimeout()


def _timed_call(fn: Callable[[], None], *, warmup: int, repeats: int, timeout_s: float) -> float:
    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return timeit(fn, warmup=warmup, repeats=repeats)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _bench_case(
    name: str,
    *,
    in_memory_obj: T,
    open_path: str,
    transform: Callable[[T], T],
    n_molecules: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    case_timeout_s: float,
    deadline: float,
    grumpy_code: str,
    notes: str,
) -> MemoryOpenCase | None:
    if time.perf_counter() >= deadline:
        print(f"  skip {name}: suite budget exhausted", file=sys.stderr, flush=True)
        return None

    print(f"  case: {name}", file=sys.stderr, flush=True)
    in_ms: float | None = None
    open_ms: float | None = None
    load_ms: float | None = None

    try:
        in_ms = (
            _timed_call(
                lambda: epoch_in_memory_batched(
                    in_memory_obj, transform, n_molecules=n_molecules, batch_size=batch_size
                ),
                warmup=warmup,
                repeats=repeats,
                timeout_s=case_timeout_s,
            )
            * 1e3
        )
    except BenchTimeout:
        print(f"  TIMEOUT {name}/in-memory > {case_timeout_s:.0f}s", file=sys.stderr, flush=True)
        return MemoryOpenCase(name, None, None, None, grumpy_code, notes + " (in-memory timed out)")

    if time.perf_counter() >= deadline:
        return MemoryOpenCase(name, in_ms, None, None, grumpy_code, notes)

    try:
        open_ms = (
            _timed_call(
                lambda: epoch_open_batched(
                    open_path,
                    transform,
                    n_molecules=n_molecules,
                    batch_size=batch_size,
                ),
                warmup=warmup,
                repeats=repeats,
                timeout_s=case_timeout_s,
            )
            * 1e3
        )
    except BenchTimeout:
        print(f"  TIMEOUT {name}/open-transform > {case_timeout_s:.0f}s", file=sys.stderr, flush=True)
        return MemoryOpenCase(name, in_ms, None, None, grumpy_code=grumpy_code, notes=notes)

    if time.perf_counter() >= deadline:
        return MemoryOpenCase(name, in_ms, open_ms, None, grumpy_code=grumpy_code, notes=notes)

    try:
        load_ms = (
            _timed_call(
                lambda: epoch_open_load_only(
                    open_path,
                    n_molecules=n_molecules,
                    batch_size=batch_size,
                ),
                warmup=warmup,
                repeats=repeats,
                timeout_s=case_timeout_s,
            )
            * 1e3
        )
    except BenchTimeout:
        print(f"  TIMEOUT {name}/open-load > {case_timeout_s:.0f}s", file=sys.stderr, flush=True)
        return MemoryOpenCase(name, in_ms, open_ms, None, grumpy_code=grumpy_code, notes=notes)

    return MemoryOpenCase(name, in_ms, open_ms, load_ms, grumpy_code=grumpy_code, notes=notes)


def build_cases(
    *,
    coord_arr: gr.GrumpyArray,
    coord_path: str,
    df_full: gr.GrumpyDataFrame,
    df_full_path: str,
    df_atoms: gr.GrumpyDataFrame,
    df_atoms_path: str,
    n_molecules: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    case_timeout_s: float,
    suite_budget_s: float,
) -> list[MemoryOpenCase]:
    cases: list[MemoryOpenCase] = []
    deadline = time.perf_counter() + suite_budget_s

    def scale(batch):
        batch = batch * 0.01
        return batch

    def center(batch):
        batch = batch + 1.0
        return batch

    def pool_residue(batch):
        batch = batch.mean(dim=1)
        return batch

    def featurize(batch):
        batch = scale(batch)
        batch = center(batch)
        return pool_residue(batch)

    def residue_center(batch):
        batch.residue.residue_center = batch.residue.atom_pos0.mean(dim=1)
        return batch

    specs = (
        (
            "coord featurize (array)",
            coord_arr,
            coord_path,
            featurize,
            "* 0.01; + 1.0; mean(dim=1) on CA coords",
            "single float column, residue×3 nesting",
        ),
        (
            "residue center (df, 3 cols)",
            df_full,
            df_full_path,
            residue_center,
            "batch.residue.residue_center = atom_pos0.mean(dim=1)",
            "loads molecule_id + residue_pos + atom_pos0 from Zarr",
        ),
        (
            "residue center (df, atom_pos0 only)",
            df_atoms,
            df_atoms_path,
            residue_center,
            "same transform, atom_pos0 column only on disk",
            "isolates unused-column I/O cost",
        ),
    )

    for name, mem_obj, path, transform, code, notes in specs:
        row = _bench_case(
            name,
            in_memory_obj=mem_obj,
            open_path=path,
            transform=transform,
            n_molecules=n_molecules,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
            case_timeout_s=case_timeout_s,
            deadline=deadline,
            grumpy_code=code,
            notes=notes,
        )
        if row is not None:
            cases.append(row)
    return cases


def print_report(report: MemoryOpenReport) -> None:
    print("## In-memory vs gr.open — batched epoch\n")
    print(f"- python: {report.python}")
    print(f"- numpy: {report.numpy}")
    print(f"- platform: {report.platform}")
    print(
        f"- n_molecules={report.n_molecules}, n_residues={report.n_residues}, "
        f"batch_size={report.batch_size} (chunk_size={report.batch_size}), n_batches={report.n_batches}, "
        f"warmup={report.warmup}, repeats={report.repeats}, "
        f"budget={report.suite_budget_s:.0f}s, wall={report.wall_time_s:.1f}s\n"
    )
    print("| case | in-memory | open (load+transform) | open load-only | transform overhead |")
    print("|---|---:|---:|---:|---:|")

    def cell(v: float | None) -> str:
        return "—" if v is None else f"{v:.1f}"

    for c in report.cases:
        overhead: float | None = None
        if c.open_transform_ms is not None and c.open_load_only_ms is not None:
            overhead = max(0.0, c.open_transform_ms - c.open_load_only_ms)
        print(
            f"| {c.name} | {cell(c.in_memory_ms)} | {cell(c.open_transform_ms)} | "
            f"{cell(c.open_load_only_ms)} | {cell(overhead)} |"
        )

    print()
    for c in report.cases:
        print(f"### {c.name}")
        if c.in_memory_ms and c.open_transform_ms:
            print(f"- open / in-memory: {c.open_transform_ms / c.in_memory_ms:.2f}×")
        if c.open_load_only_ms and c.open_transform_ms:
            overhead = max(0.0, c.open_transform_ms - c.open_load_only_ms)
            pct = 100.0 * c.open_load_only_ms / c.open_transform_ms if c.open_transform_ms else 0.0
            print(f"- transform overhead (open − load-only): {overhead:.1f} ms")
            print(f"- load share of open epoch: {pct:.0f}%")
        if c.notes:
            print(f"- {c.notes}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="In-memory vs gr.open benchmark.")
    ap.add_argument("--n-molecules", type=int, default=256)
    ap.add_argument("--n-residues", type=int, default=96)
    ap.add_argument("--atoms-per-res", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_SUITE_BUDGET_S)
    ap.add_argument("--case-timeout", type=float, default=DEFAULT_CASE_TIMEOUT_S)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", metavar="PATH", default=None)
    args = ap.parse_args(argv)

    wall_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    n_batches = (args.n_molecules + args.batch_size - 1) // args.batch_size

    with tempfile.TemporaryDirectory(prefix="grumpy_mem_open_bench_") as tmp:
        coord_path = str(Path(tmp) / "coords.gr")
        df_full_path = str(Path(tmp) / "proteins.gr")
        df_atoms_path = str(Path(tmp) / "proteins_atoms.gr")

        coords = _protein_coords(rng, args.n_molecules, args.n_residues, ragged=False)
        coord_arr = gr.array(coords, dtype=gr.float64)
        df_full = _protein_dataframe(rng, args.n_molecules, args.n_residues, args.atoms_per_res)

        t = np.arange(args.n_residues, dtype=np.float64)
        backbone = np.stack([0.38 * t, 2.0 * np.sin(t / 3.6), 2.0 * np.cos(t / 3.6)], axis=1)
        atom_idx = np.arange(args.atoms_per_res, dtype=np.float64)
        offsets = np.stack([0.03 * atom_idx, 0.01 * (atom_idx % 5), 0.02 * (atom_idx % 7)], axis=1)
        atom_template = (
            backbone[:, None, :] + offsets[None, :, :] + rng.normal(scale=0.02, size=(args.n_residues, args.atoms_per_res, 3))
        ).tolist()
        df_atoms = gr.dataframe(
            {"atom_pos0": [atom_template] * args.n_molecules},
            schema=["molecule", "residue", "atom"],
        )

        gr.save(coord_arr, coord_path, chunk_size=args.batch_size)
        gr.save(df_full, df_full_path, chunk_size=args.batch_size)
        gr.save(df_atoms, df_atoms_path, chunk_size=args.batch_size)

        cases = build_cases(
            coord_arr=coord_arr,
            coord_path=coord_path,
            df_full=df_full,
            df_full_path=df_full_path,
            df_atoms=df_atoms,
            df_atoms_path=df_atoms_path,
            n_molecules=args.n_molecules,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
            case_timeout_s=args.case_timeout,
            suite_budget_s=args.max_seconds,
        )

    wall_time = time.perf_counter() - wall_start
    report = MemoryOpenReport(
        suite="memory_vs_open",
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        n_molecules=args.n_molecules,
        n_residues=args.n_residues,
        batch_size=args.batch_size,
        n_batches=n_batches,
        warmup=args.warmup,
        repeats=args.repeats,
        suite_budget_s=args.max_seconds,
        case_timeout_s=args.case_timeout,
        wall_time_s=wall_time,
        cases=cases,
    )

    print_report(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
            f.write("\n")

    return 0 if wall_time <= args.max_seconds + 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
