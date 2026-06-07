#!/usr/bin/env python3
"""
Open-handle compiler benchmark — where ``gr.compile`` pays off.

Simulates protein-structure training pipelines: Zarr-backed batched indexing via
``gr.open``, comparing eager Python transforms vs ``compile_pipeline`` per batch.

Defaults target a **~3–5 min** full suite: 256-structure mini-epoch, 256-residue CA
traces, ``batch_size=32``. Pipelines are compute-heavy (fused elementwise chains and
kNN + pool); use ``--quick`` for a <60 s elementwise-only smoke run.
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
from grumpy.compiler import compile_pipeline, compile_pipeline_info

from _bench_common import row_length, timeit
from _open_epoch import epoch_open_batched

# --- timing guards (seconds) ---
DEFAULT_SUITE_BUDGET_S = 300.0
DEFAULT_MODE_TIMEOUT_S = 90.0


class BenchTimeout(Exception):
    """Raised when a single timed epoch exceeds its limit."""


@dataclass
class CompileBenchCase:
    name: str
    complexity: str
    open_py_ms: float | None
    open_compiled_ms: float | None
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


def _time_open_epoch(
    path: str,
    transform: Callable,
    *,
    n_molecules: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    timeout_s: float,
) -> float:
    """Return best wall time (seconds) for one full batched pass via ``gr.open``."""

    def run() -> None:
        epoch_open_batched(
            path,
            transform,
            n_molecules=n_molecules,
            batch_size=batch_size,
        )

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


def _py_pipeline(fns: list[Callable]) -> Callable:
    def run(batch):
        for fn in fns:
            batch = fn(batch)
        return batch

    return run


def _bench_open_modes(
    path: str,
    fns: list[Callable],
    *,
    n_molecules: int,
    batch_size: int,
    warmup: int,
    repeats: int,
    mode_timeout_s: float,
    deadline: float,
    label: str,
) -> tuple[float | None, float | None]:
    _require_compiled(fns)
    modes = (
        ("open_py_ms", _py_pipeline(fns)),
        ("open_compiled_ms", compile_pipeline(fns)),
    )
    results: dict[str, float | None] = {k: None for k, _ in modes}

    for key, transform in modes:
        if time.perf_counter() >= deadline:
            print(f"  skip {label}/{key}: suite budget exhausted", file=sys.stderr, flush=True)
            break
        try:
            sec = _time_open_epoch(
                path,
                transform,
                n_molecules=n_molecules,
                batch_size=batch_size,
                warmup=warmup,
                repeats=repeats,
                timeout_s=mode_timeout_s,
            )
            results[key] = sec * 1e3
        except BenchTimeout:
            print(f"  TIMEOUT {label}/{key} > {mode_timeout_s:.0f}s — skipped", file=sys.stderr, flush=True)
            continue

    return results["open_py_ms"], results["open_compiled_ms"]


def build_cases(
    coord_path: str,
    *,
    n_molecules: int,
    batch_size: int,
    cpu: int,
    n_residues: int,
    warmup: int,
    repeats: int,
    mode_timeout_s: float,
    suite_budget_s: float,
    quick: bool = False,
) -> list[CompileBenchCase]:
    cases: list[CompileBenchCase] = []
    suite_start = time.perf_counter()
    deadline = suite_start + suite_budget_s

    def remaining() -> float:
        return max(0.0, deadline - time.perf_counter())

    # Compute-heavy, fully compilable pipelines (defined here so inspect.getsource works).
    def heavy_featurize(batch):
        batch = batch * 0.01
        batch = batch + 1.0
        batch = batch * 2.0
        batch = batch - 0.5
        batch = batch / 1.1
        batch = batch.mean(dim=1)
        return batch

    def norm_knn_pool(batch):
        batch = batch * 0.01
        batch = batch + 1.0
        batch = gr.neighbors(batch, batch, k=16, dim=1, loop=False)
        batch = batch.mean(dim=1)
        return batch

    def knn_then_pool(batch):
        batch = gr.neighbors(batch, batch, k=16, dim=1, loop=False)
        batch = batch.mean(dim=1)
        return batch

    def stage_a(batch):
        batch = batch * 0.01
        batch = batch + 1.0
        return batch

    def stage_b(batch):
        batch = batch * 2.0
        batch = batch - 0.5
        return batch

    def stage_c(batch):
        batch = batch / 1.1
        batch = batch * 0.99
        return batch

    def stage_d(batch):
        batch = batch.mean(dim=1)
        return batch

    specs = (
        (
            "fused elementwise + pool",
            [heavy_featurize],
            "high",
            "* 0.01; + 1.0; * 2.0; - 0.5; / 1.1; mean(dim=1) in one CompiledPlan",
            "five fused scalar ops + reduce(dim=1)",
        ),
        (
            "staged elementwise (4 fns)",
            [stage_a, stage_b, stage_c, stage_d],
            "high",
            "four transform fns fused into one CompiledPlan",
            "same math as fused case; Python pays per-function dispatch",
        ),
        (
            "normalize + kNN + pool",
            [norm_knn_pool],
            "high",
            "* 0.01; + 1.0; neighbors(k=16); mean(dim=1)",
            "end-to-end fused normalize → kNN → pool",
        ),
        (
            "kNN (k=16) + pool",
            [knn_then_pool],
            "high",
            "neighbors(k=16, dim=1); mean(dim=1)",
            "kNN-dominated compute per batch",
        ),
    )

    for name, fns, complexity, code, notes in specs:
        if quick and "kNN" in name:
            continue
        if remaining() < 2.0:
            print(f"  skip {name}: suite budget exhausted", file=sys.stderr, flush=True)
            break
        print(f"  case: {name}", file=sys.stderr, flush=True)
        py_ms, compiled_ms = _bench_open_modes(
            coord_path,
            fns,
            n_molecules=n_molecules,
            batch_size=batch_size,
            warmup=warmup,
            repeats=repeats,
            mode_timeout_s=mode_timeout_s,
            deadline=deadline,
            label=name,
        )
        cases.append(
            CompileBenchCase(
                name=name,
                complexity=complexity,
                open_py_ms=py_ms,
                open_compiled_ms=compiled_ms,
                grumpy_code=code,
                notes=f"{notes}; {n_residues}-residue CA, batch_size={batch_size}",
            )
        )

    return cases


def print_report(report: CompileBenchReport) -> None:
    print("## Open-handle compile suite — full mini-epoch\n")
    print(f"- python: {report.python}")
    print(f"- numpy: {report.numpy}")
    print(f"- platform: {report.platform}")
    print(
        f"- n_molecules={report.n_molecules}, n_residues={report.n_residues}, "
        f"batch_size={report.batch_size}, cpu={report.cpu}, n_batches={report.n_batches}, "
        f"warmup={report.warmup}, repeats={report.repeats}, "
        f"budget={report.suite_budget_s:.0f}s, wall={report.wall_time_s:.1f}s\n"
    )
    print("| pipeline | Python (open) | Compiled (open) | speedup |")
    print("|---|---:|---:|---:|")
    for c in report.cases:
        def _cell(v: float | None) -> str:
            return "—" if v is None else f"{v:.1f}"

        speedup = "—"
        if c.open_py_ms and c.open_compiled_ms:
            speedup = f"{c.open_py_ms / c.open_compiled_ms:.2f}×"
        print(f"| {c.name} | {_cell(c.open_py_ms)} | {_cell(c.open_compiled_ms)} | {speedup} |")
    print()
    for c in report.cases:
        print(f"### {c.name}")
        if c.open_py_ms and c.open_compiled_ms:
            print(f"- compiled vs Python: {c.open_py_ms / c.open_compiled_ms:.2f}×")
        if c.notes:
            print(f"- {c.notes}")
        print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Open-handle compiler benchmark with JSON export.")
    ap.add_argument("--n-molecules", type=int, default=256, help="Structures in Zarr store (mini-epoch).")
    ap.add_argument("--n-residues", type=int, default=256, help="Residues per structure (typical domain length).")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Elementwise-only smoke run (<60 s): 96 residues, 55 s budget, 8 s mode timeout.",
    )
    ap.add_argument("--atoms-per-res", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_SUITE_BUDGET_S, help="Total suite wall budget.")
    ap.add_argument("--mode-timeout", type=float, default=DEFAULT_MODE_TIMEOUT_S, help="Per-mode epoch timeout.")
    ap.add_argument("--json", metavar="PATH", default=None)
    args = ap.parse_args(argv)

    if args.quick:
        args.n_residues = 96
        args.max_seconds = 55.0
        args.mode_timeout = 8.0

    wall_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    n_batches = (args.n_molecules + args.batch_size - 1) // args.batch_size

    with tempfile.TemporaryDirectory(prefix="grumpy_compile_bench_") as tmp:
        coord_path = str(Path(tmp) / "coords.gr")

        coords = _protein_coords(rng, args.n_molecules, args.n_residues, ragged=False)

        gr.save(gr.array(coords, dtype=gr.float64), coord_path, chunk_size=args.batch_size)

        cases = build_cases(
            coord_path,
            n_molecules=args.n_molecules,
            batch_size=args.batch_size,
            cpu=args.cpu,
            n_residues=args.n_residues,
            warmup=args.warmup,
            repeats=args.repeats,
            mode_timeout_s=args.mode_timeout,
            suite_budget_s=args.max_seconds,
            quick=args.quick,
        )

    wall_time = time.perf_counter() - wall_start
    if wall_time > args.max_seconds:
        print(
            f"WARNING: suite exceeded budget ({wall_time:.1f}s > {args.max_seconds:.0f}s)",
            file=sys.stderr,
        )

    report = CompileBenchReport(
        suite="compile_open",
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
