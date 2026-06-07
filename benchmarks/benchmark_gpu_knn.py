#!/usr/bin/env python3
"""Benchmark CPU vs GPU kNN (Metal/CUDA) on protein-like coordinate batches."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

import grumpy as gr

from _bench_common import timeit
from _open_epoch import epoch_open_batched


@dataclass
class GpuKnnReport:
    suite: str
    python: str
    numpy: str
    platform: str
    gpu_available: bool
    gpu_backend: str | None
    n_molecules: int
    n_residues: int
    k: int
    batch_size: int
    warmup: int
    repeats: int
    cpu_dim1_ms: float
    gpu_dim1_ms: float
    auto_dim1_ms: float | None = None
    cpu_open_ms: float | None = None
    gpu_open_ms: float | None = None
    auto_open_ms: float | None = None
    wall_time_s: float = 0.0


def _protein_coords(rng: np.random.Generator, n_molecules: int, n_residues: int) -> list:
    template_t = np.arange(n_residues, dtype=np.float64)
    template = np.stack(
        [0.38 * template_t, 2.0 * np.sin(template_t / 3.6), 2.0 * np.cos(template_t / 3.6)],
        axis=1,
    )
    out = []
    for _ in range(n_molecules):
        out.append((template + rng.normal(scale=0.05, size=(n_residues, 3))).tolist())
    return out


def _bench_neighbors(x: gr.GrumpyArray, *, gpu: bool | str, warmup: int, repeats: int) -> float:
    def run() -> None:
        out = gr.neighbors(x, x, k=args_k, dim=1, loop=False, gpu=gpu)
        _ = out.to_list()[0][0][0][0]

    return timeit(run, warmup=warmup, repeats=repeats) * 1e3


def _knn_pool(batch, *, gpu: bool | str):
    batch = gr.neighbors(batch, batch, k=args_k, dim=1, loop=False, gpu=gpu)
    return batch.mean(dim=1)


def _bench_open(path: str, *, gpu: bool | str, n_molecules: int, batch_size: int, warmup: int, repeats: int) -> float:
    def epoch() -> None:
        epoch_open_batched(
            path,
            lambda batch: _knn_pool(batch, gpu=gpu),
            n_molecules=n_molecules,
            batch_size=batch_size,
        )

    return timeit(epoch, warmup=warmup, repeats=repeats) * 1e3


args_k = 16


def main(argv: list[str] | None = None) -> int:
    global args_k
    ap = argparse.ArgumentParser(description="GPU kNN benchmark")
    ap.add_argument("--n-molecules", type=int, default=256)
    ap.add_argument("--n-residues", type=int, default=256)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    args_k = args.k

    if not gr.gpu_available():
        print("WARNING: no GPU backend available; GPU timings will fall back to CPU", file=sys.stderr)

    wall_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    coords = _protein_coords(rng, args.n_molecules, args.n_residues)
    x = gr.array(coords, dtype=gr.float64)

    cpu_ms = _bench_neighbors(x, gpu=False, warmup=args.warmup, repeats=args.repeats)
    gpu_ms = _bench_neighbors(x, gpu=True, warmup=args.warmup, repeats=args.repeats)
    auto_ms = _bench_neighbors(x, gpu="auto", warmup=args.warmup, repeats=args.repeats)

    open_cpu_ms = None
    open_gpu_ms = None
    open_auto_ms = None
    with tempfile.TemporaryDirectory(prefix="grumpy_gpu_bench_") as tmp:
        path = str(Path(tmp) / "coords.gr")
        gr.save(x, path, chunk_size=args.batch_size)
        open_cpu_ms = _bench_open(
            path,
            gpu=False,
            n_molecules=args.n_molecules,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        open_gpu_ms = _bench_open(
            path,
            gpu=True,
            n_molecules=args.n_molecules,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        open_auto_ms = _bench_open(
            path,
            gpu="auto",
            n_molecules=args.n_molecules,
            batch_size=args.batch_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    wall = time.perf_counter() - wall_start
    report = GpuKnnReport(
        suite="gpu_knn",
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        gpu_available=gr.gpu_available(),
        gpu_backend=gr.gpu_backend(),
        n_molecules=args.n_molecules,
        n_residues=args.n_residues,
        k=args.k,
        batch_size=args.batch_size,
        warmup=args.warmup,
        repeats=args.repeats,
        cpu_dim1_ms=cpu_ms,
        gpu_dim1_ms=gpu_ms,
        auto_dim1_ms=auto_ms,
        cpu_open_ms=open_cpu_ms,
        gpu_open_ms=open_gpu_ms,
        auto_open_ms=open_auto_ms,
        wall_time_s=wall,
    )

    print("## GPU kNN benchmark\n")
    print(f"- platform: {report.platform}")
    print(f"- gpu: {report.gpu_backend} (available={report.gpu_available})")
    print(f"- dataset: {report.n_molecules} molecules × {report.n_residues} residues, k={report.k}")
    print(f"- warmup={report.warmup}, repeats={report.repeats}, wall={report.wall_time_s:.1f}s\n")
    print("| mode | CPU ms | GPU force ms | GPU auto ms | speedup (auto) |")
    print("|---|---:|---:|---:|---:|")
    print(f"| neighbors dim=1 (in-memory) | {cpu_ms:.1f} | {gpu_ms:.1f} | {auto_ms:.1f} | {cpu_ms / auto_ms:.2f}× |")
    if open_cpu_ms is not None and open_gpu_ms is not None and open_auto_ms is not None:
        print(
            f"| open kNN+pool (compile) | {open_cpu_ms:.1f} | {open_gpu_ms:.1f} | {open_auto_ms:.1f} | "
            f"{open_cpu_ms / open_auto_ms:.2f}× |"
        )

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
