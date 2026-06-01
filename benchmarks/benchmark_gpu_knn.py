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

from _bench_common import row_length, timeit


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
    cpu_stream_ms: float | None = None
    gpu_stream_ms: float | None = None
    auto_stream_ms: float | None = None
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


def _bench_stream(path: str, *, gpu: bool | str, batch_size: int, warmup: int, repeats: int) -> float:
    def run() -> None:
        st = gr.stream(path, batch_size=batch_size, gpu=gpu)

        def knn_pool(batch):
            batch = gr.neighbors(batch, batch, k=args_k, dim=1, loop=False)
            return batch.mean(dim=1)

        for _ in st.apply(knn_pool, cpu=1, compile="auto"):
            pass

    return timeit(run, warmup=warmup, repeats=repeats) * 1e3


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
    gpu_ms = _bench_neighbors(x, gpu="force", warmup=args.warmup, repeats=args.repeats)
    auto_ms = _bench_neighbors(x, gpu="auto", warmup=args.warmup, repeats=args.repeats)

    stream_cpu_ms = None
    stream_gpu_ms = None
    stream_auto_ms = None
    with tempfile.TemporaryDirectory(prefix="grumpy_gpu_bench_") as tmp:
        path = str(Path(tmp) / "coords.gr")
        gr.save(x, path, chunk_size=args.batch_size)
        stream_cpu_ms = _bench_stream(path, gpu=False, batch_size=args.batch_size, warmup=args.warmup, repeats=args.repeats)
        stream_gpu_ms = _bench_stream(path, gpu="force", batch_size=args.batch_size, warmup=args.warmup, repeats=args.repeats)
        stream_auto_ms = _bench_stream(path, gpu="auto", batch_size=args.batch_size, warmup=args.warmup, repeats=args.repeats)

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
        cpu_stream_ms=stream_cpu_ms,
        gpu_stream_ms=stream_gpu_ms,
        auto_stream_ms=stream_auto_ms,
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
    if stream_cpu_ms is not None and stream_gpu_ms is not None and stream_auto_ms is not None:
        print(f"| stream kNN+pool (compile) | {stream_cpu_ms:.1f} | {stream_gpu_ms:.1f} | {stream_auto_ms:.1f} | {stream_cpu_ms / stream_auto_ms:.2f}× |")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
