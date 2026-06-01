#!/usr/bin/env python3
"""
Streaming compiler benchmark — where ``gr.compile`` pays off.

Simulates protein-structure training pipelines: Zarr-backed axis-0 streaming over
a saved dataset, comparing Python vs compiled transforms and Rust batch scheduling.

Defaults target **< 60 s** total wall time while keeping bio-realistic *per-structure*
shape (128-residue CA traces, batch_size=32, 4 heavy atoms/residue dataframe path).
Corpus size is a 256-structure mini-epoch (scale linearly to full training sets).
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
from typing import Callable

import numpy as np

import grumpy as gr
from grumpy.compiler import compile_pipeline_info

from _bench_common import row_length, timeit

# --- timing guards (seconds) ---
DEFAULT_SUITE_BUDGET_S = 55.0
DEFAULT_MODE_TIMEOUT_S = 8.0


class BenchTimeout(Exception):
    """Raised when a single timed epoch exceeds its limit."""


@dataclass
class CompileBenchCase:
    name: str
    complexity: str
    stream_py_cpu1_ms: float | None
    stream_compiled_cpu1_ms: float | None
    stream_compiled_cpu4_pysched_ms: float | None
    stream_compiled_cpu4_rust_ms: float | None
    grumpy_code: str = ""
    notes: str = ""


@dataclass
class CompileBenchReport:
    suite: str
    python: str
    numpy: str
    platform: str
    n_molecules: int
    n_residues: int
    n_coords: int
    batch_size: int
    cpu: int
    n_batches: int
    warmup: int
    repeats: int
    suite_budget_s: float
    mode_timeout_s: float
    wall_time_s: float = 0.0
    cases: list[CompileBenchCase] = field(default_factory=list)


def _alarm_handler(_signum: int, _frame) -> None:
    raise BenchTimeout()


def _time_epoch(
    path: str,
    fns: list[Callable],
    *,
    batch_size: int,
    cpu: int,
    compile: bool | str,
    scheduler: str,
    warmup: int,
    repeats: int,
    timeout_s: float,
) -> float:
    """Return best wall time (seconds) for one full pass over all stream batches."""

    def run() -> None:
        st = gr.stream(path, batch_size=batch_size, drop_last=False)
        for _ in st.apply(fns, cpu=cpu, compile=compile, scheduler=scheduler):
            pass

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return timeit(run, warmup=warmup, repeats=repeats)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _require_compiled(fns: list[Callable]) -> None:
    info = compile_pipeline_info(fns)
    if not info.fully_compiled:
        raise RuntimeError(f"pipeline did not fully compile: {fns!r}")


def _protein_coords(
    rng: np.random.Generator,
    n_molecules: int,
    n_residues: int,
    *,
    ragged: bool,
) -> list[list[list[float]]]:
    """CA trace: molecule > residue > (x, y, z). Optional ±1 residue raggedness."""
    template_t = np.arange(n_residues, dtype=np.float64)
    template = np.stack(
        [0.38 * template_t, 2.0 * np.sin(template_t / 3.6), 2.0 * np.cos(template_t / 3.6)],
        axis=1,
    )
    out: list[list[list[float]]] = []
    for i in range(n_molecules):
        n_res = row_length(n_residues, i) if ragged else n_residues
        if ragged and n_res != n_residues:
            t_mol = np.arange(n_res, dtype=np.float64)
            base = np.stack(
                [0.38 * t_mol, 2.0 * np.sin(t_mol / 3.6), 2.0 * np.cos(t_mol / 3.6)],
                axis=1,
            )
        else:
            base = template if n_res == n_residues else template[:n_res]
        out.append((base + rng.normal(scale=0.05, size=(n_res, 3))).tolist())
    return out


def _protein_dataframe(
    rng: np.random.Generator,
    n_molecules: int,
    n_residues: int,
    atoms_per_res: int,
) -> gr.GrumpyDataFrame:
    """Molecule > residue > atom dataframe (atom positions for center-of-mass featurization)."""
    schema = ["molecule", "residue", "atom"]
    molecule_id = [f"M{i}" for i in range(n_molecules)]
    t = np.arange(n_residues, dtype=np.float64)
    backbone = np.stack([0.38 * t, 2.0 * np.sin(t / 3.6), 2.0 * np.cos(t / 3.6)], axis=1)
    residue_pos = [backbone.tolist()] * n_molecules

    atom_idx = np.arange(atoms_per_res, dtype=np.float64)
    offsets = np.stack([0.03 * atom_idx, 0.01 * (atom_idx % 5), 0.02 * (atom_idx % 7)], axis=1)
    atom_template = (backbone[:, None, :] + offsets[None, :, :] + rng.normal(scale=0.02, size=(n_residues, atoms_per_res, 3))).tolist()
    atom_pos0 = [atom_template for _ in range(n_molecules)]

    return gr.dataframe(
        {"molecule_id": molecule_id, "residue_pos": residue_pos, "atom_pos0": atom_pos0},
        schema=schema,
    )


def _bench_stream_modes(
    path: str,
    fns: list[Callable],
    *,
    batch_size: int,
    cpu: int,
    warmup: int,
    repeats: int,
    mode_timeout_s: float,
    deadline: float,
    label: str,
) -> tuple[float | None, float | None, float | None, float | None]:
    _require_compiled(fns)
    modes = (
        ("stream_py_cpu1_ms", dict(cpu=1, compile=False, scheduler="auto")),
        ("stream_compiled_cpu1_ms", dict(cpu=1, compile=True, scheduler="auto")),
        ("stream_compiled_cpu4_pysched_ms", dict(cpu=cpu, compile=True, scheduler="python")),
        ("stream_compiled_cpu4_rust_ms", dict(cpu=cpu, compile=True, scheduler="auto")),
    )
    results: dict[str, float | None] = {k: None for k, _ in modes}

    for key, kw in modes:
        if time.perf_counter() >= deadline:
            print(f"  skip {label}/{key}: suite budget exhausted", file=sys.stderr, flush=True)
            break
        try:
            sec = _time_epoch(
                path,
                fns,
                batch_size=batch_size,
                warmup=warmup,
                repeats=repeats,
                timeout_s=mode_timeout_s,
                **kw,
            )
            results[key] = sec * 1e3
        except BenchTimeout:
            print(f"  TIMEOUT {label}/{key} > {mode_timeout_s:.0f}s — killed", file=sys.stderr, flush=True)
            break

    return (
        results["stream_py_cpu1_ms"],
        results["stream_compiled_cpu1_ms"],
        results["stream_compiled_cpu4_pysched_ms"],
        results["stream_compiled_cpu4_rust_ms"],
    )


def build_cases(
    coord_path: str,
    df_path: str,
    *,
    batch_size: int,
    cpu: int,
    n_residues: int,
    warmup: int,
    repeats: int,
    mode_timeout_s: float,
    suite_budget_s: float,
) -> list[CompileBenchCase]:
    cases: list[CompileBenchCase] = []
    suite_start = time.perf_counter()
    deadline = suite_start + suite_budget_s

    def remaining() -> float:
        return max(0.0, deadline - time.perf_counter())

    # --- shared transforms (module-level names help compile_pipeline_info) ---
    def scale(batch):
        batch = batch * 0.01
        return batch

    def center(batch):
        batch = batch + 1.0
        return batch

    def pool_residue(batch):
        batch = batch.mean(dim=1)
        return batch

    def knn_residues(batch):
        batch = gr.neighbors(batch, batch, k=8, dim=1, loop=False)
        return batch

    def residue_center(batch):
        batch.residue.residue_center = batch.residue.atom_pos0.mean(dim=1)
        return batch

    featurize = [scale, center, pool_residue]

    # 1. Coordinate normalize + pool
    print("  case: coord normalize + pool", file=sys.stderr, flush=True)
    py1, c1, c4_py, c4_rust = _bench_stream_modes(
        coord_path,
        featurize,
        batch_size=batch_size,
        cpu=cpu,
        warmup=warmup,
        repeats=repeats,
        mode_timeout_s=mode_timeout_s,
        deadline=deadline,
        label="coord pool",
    )
    cases.append(
        CompileBenchCase(
            name="coord normalize + pool",
            complexity="medium",
            stream_py_cpu1_ms=py1,
            stream_compiled_cpu1_ms=c1,
            stream_compiled_cpu4_pysched_ms=c4_py,
            stream_compiled_cpu4_rust_ms=c4_rust,
            grumpy_code="* 0.01; + 1.0; mean(dim=1)",
            notes=f"{n_residues}-residue CA traces, Zarr stream batch_size={batch_size}",
        )
    )

    if remaining() < 2.0:
        print("suite budget exhausted after case 1", file=sys.stderr, flush=True)
        return cases

    # 2. Dataframe residue center (compiled df_get / df_set path)
    print("  case: residue center (df)", file=sys.stderr, flush=True)
    py1, c1, c4_py, c4_rust = _bench_stream_modes(
        df_path,
        [residue_center],
        batch_size=batch_size,
        cpu=cpu,
        warmup=warmup,
        repeats=repeats,
        mode_timeout_s=mode_timeout_s,
        deadline=deadline,
        label="residue center",
    )
    cases.append(
        CompileBenchCase(
            name="residue center (df)",
            complexity="medium",
            stream_py_cpu1_ms=py1,
            stream_compiled_cpu1_ms=c1,
            stream_compiled_cpu4_pysched_ms=c4_py,
            stream_compiled_cpu4_rust_ms=c4_rust,
            grumpy_code="batch.residue.residue_center = atom_pos0.mean(dim=1)",
            notes="molecule>residue>atom schema, 4 atoms/residue",
        )
    )

    if remaining() < 2.0:
        return cases

    # 3. Fused normalize + kNN — compile fusion + Rust batch scheduler
    print("  case: normalize + kNN", file=sys.stderr, flush=True)
    pipeline = [scale, center, knn_residues]
    py1, c1, c4_py, c4_rust = _bench_stream_modes(
        coord_path,
        pipeline,
        batch_size=batch_size,
        cpu=cpu,
        warmup=warmup,
        repeats=repeats,
        mode_timeout_s=mode_timeout_s,
        deadline=deadline,
        label="normalize+kNN",
    )
    cases.append(
        CompileBenchCase(
            name="normalize + kNN",
            complexity="high",
            stream_py_cpu1_ms=py1,
            stream_compiled_cpu1_ms=c1,
            stream_compiled_cpu4_pysched_ms=c4_py,
            stream_compiled_cpu4_rust_ms=c4_rust,
            grumpy_code="* 0.01; + 1.0; neighbors(k=8, dim=1)",
            notes="fused CompiledPlan + Rayon scheduling (cpu=4)",
        )
    )

    return cases


def print_report(report: CompileBenchReport) -> None:
    print("## Streaming compile suite — full mini-epoch\n")
    print(f"- python: {report.python}")
    print(f"- numpy: {report.numpy}")
    print(f"- platform: {report.platform}")
    print(
        f"- n_molecules={report.n_molecules}, n_residues={report.n_residues}, "
        f"batch_size={report.batch_size}, cpu={report.cpu}, n_batches={report.n_batches}, "
        f"warmup={report.warmup}, repeats={report.repeats}, "
        f"budget={report.suite_budget_s:.0f}s, wall={report.wall_time_s:.1f}s\n"
    )
    print(
        "| pipeline | Python cpu=1 | Compiled cpu=1 | Compiled cpu=4 py | Compiled cpu=4 rust |"
    )
    print("|---|---:|---:|---:|---:|")
    for c in report.cases:
        def _cell(v: float | None) -> str:
            return "—" if v is None else f"{v:.1f}"

        print(
            f"| {c.name} | {_cell(c.stream_py_cpu1_ms)} | {_cell(c.stream_compiled_cpu1_ms)} | "
            f"{_cell(c.stream_compiled_cpu4_pysched_ms)} | {_cell(c.stream_compiled_cpu4_rust_ms)} |"
        )
    print()
    for c in report.cases:
        if c.stream_py_cpu1_ms and c.stream_compiled_cpu1_ms:
            print(f"### {c.name} — compile vs Python (cpu=1): {c.stream_py_cpu1_ms / c.stream_compiled_cpu1_ms:.2f}×")
        if c.stream_py_cpu1_ms and c.stream_compiled_cpu4_rust_ms:
            print(f"- Rust scheduler vs Python (cpu=1): {c.stream_py_cpu1_ms / c.stream_compiled_cpu4_rust_ms:.2f}×")
        if c.stream_compiled_cpu4_pysched_ms and c.stream_compiled_cpu4_rust_ms:
            print(f"- Rust vs ThreadPool (cpu=4): {c.stream_compiled_cpu4_pysched_ms / c.stream_compiled_cpu4_rust_ms:.2f}×")
        if c.notes:
            print(f"- {c.notes}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Streaming compiler benchmark with JSON export.")
    ap.add_argument("--n-molecules", type=int, default=256, help="Structures in Zarr store (mini-epoch).")
    ap.add_argument("--n-residues", type=int, default=96, help="Residues per structure (typical domain length).")
    ap.add_argument("--atoms-per-res", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_SUITE_BUDGET_S, help="Total suite wall budget.")
    ap.add_argument("--mode-timeout", type=float, default=DEFAULT_MODE_TIMEOUT_S, help="Per-mode epoch timeout.")
    ap.add_argument("--json", metavar="PATH", default=None)
    args = ap.parse_args(argv)

    wall_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    n_batches = (args.n_molecules + args.batch_size - 1) // args.batch_size

    with tempfile.TemporaryDirectory(prefix="grumpy_compile_bench_") as tmp:
        coord_path = str(Path(tmp) / "coords.gr")
        df_path = str(Path(tmp) / "proteins.gr")

        coords = _protein_coords(rng, args.n_molecules, args.n_residues, ragged=False)
        df = _protein_dataframe(rng, args.n_molecules, args.n_residues, args.atoms_per_res)

        gr.save(gr.array(coords, dtype=gr.float64), coord_path, chunk_size=args.batch_size)
        gr.save(df, df_path, chunk_size=args.batch_size)

        cases = build_cases(
            coord_path,
            df_path,
            batch_size=args.batch_size,
            cpu=args.cpu,
            n_residues=args.n_residues,
            warmup=args.warmup,
            repeats=args.repeats,
            mode_timeout_s=args.mode_timeout,
            suite_budget_s=args.max_seconds,
        )

    wall_time = time.perf_counter() - wall_start
    if wall_time > args.max_seconds:
        print(
            f"WARNING: suite exceeded budget ({wall_time:.1f}s > {args.max_seconds:.0f}s)",
            file=sys.stderr,
        )

    report = CompileBenchReport(
        suite="compile_streaming",
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        n_molecules=args.n_molecules,
        n_residues=args.n_residues,
        n_coords=3,
        batch_size=args.batch_size,
        cpu=args.cpu,
        n_batches=n_batches,
        warmup=args.warmup,
        repeats=args.repeats,
        suite_budget_s=args.max_seconds,
        mode_timeout_s=args.mode_timeout,
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
