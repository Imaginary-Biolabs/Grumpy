#!/usr/bin/env python3
"""Profile individual steps in the streaming pipeline (load, transform, I/O)."""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import grumpy as gr

from benchmark_compile_suite import _protein_dataframe
from benchmark_memory_vs_stream import _epoch_in_memory_batched


def ms(fn, *, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def profile_residue_center(batch):
    batch.residue.residue_center = batch.residue.atom_pos0.mean(dim=1)
    return batch


def time_stream_iter_batches(path: str, batch_size: int) -> list[float]:
    from grumpy._core import stream_batches

    it = stream_batches(path, batch_size, False, None, None, None, 0, 1, 0, None, False)
    times = []
    while True:
        t0 = time.perf_counter()
        try:
            batch = next(it)
        except StopIteration:
            break
        times.append((time.perf_counter() - t0) * 1e3)
        del batch
    return times


def main() -> None:
    rng = np.random.default_rng(42)
    n_molecules, n_residues, batch_size = 256, 96, 32
    n_batches = (n_molecules + batch_size - 1) // batch_size

    with tempfile.TemporaryDirectory(prefix="grumpy_profile_") as tmp:
        df_path = str(Path(tmp) / "proteins.gr")
        df = _protein_dataframe(rng, n_molecules, n_residues, 4)
        gr.save(df, df_path, chunk_size=batch_size)

        full = gr.load(df_path)
        t_full_load = ms(lambda: gr.load(df_path))

        t_manual_epoch = ms(
            lambda: _epoch_in_memory_batched(
                full, profile_residue_center, n_molecules=n_molecules, batch_size=batch_size
            )
        )
        t_manual_slice_only = ms(
            lambda: _epoch_in_memory_batched(
                full, lambda b: b, n_molecules=n_molecules, batch_size=batch_size
            )
        )
        t_manual_transform = ms(
            lambda: [
                profile_residue_center(full[i : min(i + batch_size, n_molecules)])
                for i in range(0, n_molecules, batch_size)
            ]
        )

        def epoch_load_slice_fresh():
            for start in range(0, n_molecules, batch_size):
                stop = min(start + batch_size, n_molecules)
                gr._core.load_slice(df_path, start, stop)

        t_load_slice_epoch = ms(epoch_load_slice_fresh)

        batch_times_fresh = []
        for start in range(0, n_molecules, batch_size):
            stop = min(start + batch_size, n_molecules)

            def one_batch(s=start, e=stop):
                gr._core.load_slice(df_path, s, e)

            batch_times_fresh.append(ms(one_batch, repeats=3, warmup=0))

        stream_iter_batch_times = time_stream_iter_batches(df_path, batch_size)

        def epoch_stream(workers: int = 0, in_memory: bool = False):
            st = gr.stream(df_path, batch_size=batch_size, workers=workers, in_memory=in_memory)
            for b in st:
                pass

        def epoch_stream_transform(workers: int = 0, in_memory: bool = False):
            st = gr.stream(df_path, batch_size=batch_size, workers=workers, in_memory=in_memory)
            for b in st:
                profile_residue_center(b)

        t_stream_load_w0 = ms(lambda: epoch_stream(0, False))
        t_stream_load_w4 = ms(lambda: epoch_stream(4, False))
        t_stream_load_inmem = ms(lambda: epoch_stream(0, True))
        t_stream_xform_w0 = ms(lambda: epoch_stream_transform(0, False))
        t_stream_xform_inmem = ms(lambda: epoch_stream_transform(0, True))
        t_stream_load_w0_e2 = ms(lambda: epoch_stream(0, False))
        t_stream_load_inmem_e2 = ms(lambda: epoch_stream(0, True))

        def stream_open_only(in_memory: bool):
            gr.stream(df_path, batch_size=batch_size, in_memory=in_memory)

        def stream_first_batch(in_memory: bool):
            st = gr.stream(df_path, batch_size=batch_size, in_memory=in_memory)
            next(iter(st))

        t_stream_open_disk = ms(lambda: stream_open_only(False))
        t_stream_open_inmem = ms(lambda: stream_open_only(True))
        t_open_disk_batch1 = ms(lambda: stream_first_batch(False))
        t_open_inmem_batch1 = ms(lambda: stream_first_batch(True))

        # atom_pos0-only column
        t_arr = np.arange(n_residues, dtype=np.float64)
        backbone = np.stack([0.38 * t_arr, 2.0 * np.sin(t_arr / 3.6), 2.0 * np.cos(t_arr / 3.6)], axis=1)
        atom_idx = np.arange(4, dtype=np.float64)
        offsets = np.stack([0.03 * atom_idx, 0.01 * (atom_idx % 5), 0.02 * (atom_idx % 7)], axis=1)
        atom_template = (
            backbone[:, None, :] + offsets[None, :, :] + rng.normal(scale=0.02, size=(n_residues, 4, 3))
        ).tolist()
        df_atoms_path = str(Path(tmp) / "atoms.gr")
        df_atoms = gr.dataframe({"atom_pos0": [atom_template] * n_molecules}, schema=["molecule", "residue", "atom"])
        gr.save(df_atoms, df_atoms_path, chunk_size=batch_size)

        t_atoms_epoch = ms(
            lambda: [
                gr._core.load_slice(df_atoms_path, i, min(i + batch_size, n_molecules))
                for i in range(0, n_molecules, batch_size)
            ]
        )

        gr._core.reset_io_bytes_read()
        epoch_load_slice_fresh()
        io_load_slice = gr._core.io_bytes_read()

        gr._core.reset_io_bytes_read()
        epoch_stream(0, False)
        io_stream = gr._core.io_bytes_read()

        gr._core.reset_io_bytes_read()
        gr.load(df_path)
        io_full = gr._core.io_bytes_read()

        t_transform_only = max(0.0, t_manual_transform - t_manual_slice_only)

        print("## Stream step profiler (256×96 df, batch_size=32, chunk_size=32)\n")
        print("| step | median ms | notes |")
        print("|---|---:|---|")
        rows = [
            ("gr.load (full dataset)", t_full_load, f"{io_full / 1e6:.2f} MB I/O counter"),
            ("manual epoch (slice + transform)", t_manual_epoch, "in-memory reference"),
            ("  slice only (manual)", t_manual_slice_only, ""),
            ("  transform only (manual)", t_transform_only, "mean(dim=1) + df_set"),
            ("load_slice full epoch", t_load_slice_epoch, f"{io_load_slice / 1e6:.2f} MB I/O; new handle each batch"),
            ("load_slice batch 1", batch_times_fresh[0], "cold handle"),
            ("load_slice batch 8", batch_times_fresh[-1], "cold handle"),
            ("stream iter batch 1 (shared handle)", stream_iter_batch_times[0], "includes plan open"),
            ("stream iter batch 8 (shared handle)", stream_iter_batch_times[-1], "leaf cache warm"),
            ("stream open (disk)", t_stream_open_disk, "plan + metadata only"),
            ("stream open (in_memory)", t_stream_open_inmem, "≈ full gr.load"),
            ("stream 1st batch (disk)", t_open_disk_batch1, "open + batch 1 load"),
            ("stream 1st batch (in_memory)", t_open_inmem_batch1, "open + batch 1 slice"),
            ("stream epoch load-only w=0", t_stream_load_w0, f"{io_stream / 1e6:.2f} MB I/O"),
            ("stream epoch load-only w=0 (repeat)", t_stream_load_w0_e2, "new handle; cache cold again"),
            ("stream epoch load-only w=4", t_stream_load_w4, "parallel prefetch overhead"),
            ("stream epoch load-only in_memory", t_stream_load_inmem, "pays full load every epoch"),
            ("stream epoch load+transform w=0", t_stream_xform_w0, ""),
            ("stream epoch load+transform in_memory", t_stream_xform_inmem, ""),
            ("atom_pos0-only load_slice epoch", t_atoms_epoch, "1 col vs 3 cols"),
        ]
        for name, val, note in rows:
            print(f"| {name} | {val:.1f} | {note} |")

        print(f"\n**load_slice per-batch (cold handle), ms:** {[round(x, 1) for x in batch_times_fresh]}")
        print(f"**stream iter per-batch (warm handle), ms:** {[round(x, 1) for x in stream_iter_batch_times]}")

        load_part = t_stream_load_w0
        xform_part = max(0, t_stream_xform_w0 - t_stream_load_w0)
        print("\n### Epoch breakdown (stream workers=0, df 3 cols)")
        print(f"- load: {load_part:.1f} ms ({100 * load_part / max(t_stream_xform_w0, 1):.0f}%)")
        print(f"- transform: {xform_part:.1f} ms ({100 * xform_part / max(t_stream_xform_w0, 1):.0f}%)")
        print(f"- vs manual: {t_stream_xform_w0 / max(t_manual_epoch, 1):.1f}× slower")

        if len(stream_iter_batch_times) >= 2:
            b1 = stream_iter_batch_times[0]
            b8 = stream_iter_batch_times[-1]
            rest = statistics.mean(stream_iter_batch_times[1:])
            print("\n### Within-epoch stream batches (shared IoCache)")
            print(f"- batch 1: {b1:.1f} ms (dominated by full-buffer warm + layout materialize)")
            print(f"- batches 2–8 avg: {rest:.1f} ms")
            print(f"- batch 8: {b8:.1f} ms")

        col_overhead = t_load_slice_epoch - t_atoms_epoch
        print("\n### Column / metadata overhead (load_slice epoch)")
        print(f"- atom_pos0 only: {t_atoms_epoch:.1f} ms")
        print(f"- 3 columns total: {t_load_slice_epoch:.1f} ms")
        print(f"- extra 2 cols (molecule_id, residue_pos): ~{col_overhead:.1f} ms")


if __name__ == "__main__":
    main()
