"""Shared batched-epoch helpers for ``gr.open`` and partial I/O benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import numpy as np

import grumpy as gr

T = TypeVar("T")

BatchWindow = tuple[int, int]


def saved_root_kind(path: str) -> str:
    """Return ``"dataframe"`` or ``"array"`` from ``grumpy.json``."""
    meta = json.loads(Path(path).joinpath("grumpy.json").read_text(encoding="utf-8"))
    return meta["root"]["kind"]


def consecutive_batch_windows(n_molecules: int, batch_size: int) -> list[BatchWindow]:
    """Chunk-aligned axis-0 windows in linear order (best case for Zarr)."""
    return [
        (start, min(start + batch_size, n_molecules))
        for start in range(0, n_molecules, batch_size)
    ]


def random_straddling_batch_windows(
    n_molecules: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[BatchWindow]:
    """Misaligned windows that cross chunk boundaries, visited in shuffled order."""
    n_batches = (n_molecules + batch_size - 1) // batch_size
    half = max(1, batch_size // 2)
    max_start = max(0, n_molecules - batch_size)
    windows: list[BatchWindow] = []
    for i in range(n_batches):
        # Offset by half a chunk so each batch spans two Zarr chunks.
        start = min((i * batch_size + half) % (max_start + 1), max_start)
        windows.append((start, start + batch_size))
    rng.shuffle(windows)
    return windows


def epoch_in_memory_windows(
    obj: T,
    transform: Callable[[T], T] | None,
    windows: Sequence[BatchWindow],
) -> None:
    for start, stop in windows:
        batch = obj[start:stop]
        if transform is not None:
            transform(batch)


def epoch_open_windows(
    path: str,
    transform: Callable[[T], T] | None,
    windows: Sequence[BatchWindow],
) -> None:
    with gr.open(path) as handle:
        for start, stop in windows:
            batch = handle[start:stop]
            if transform is not None:
                transform(batch)


def epoch_in_memory_batched(
    obj: T,
    transform: Callable[[T], T] | None,
    *,
    n_molecules: int,
    batch_size: int,
) -> None:
    epoch_in_memory_windows(
        obj,
        transform,
        consecutive_batch_windows(n_molecules, batch_size),
    )


def epoch_open_batched(
    path: str,
    transform: Callable[[T], T] | None,
    *,
    n_molecules: int,
    batch_size: int,
) -> None:
    """Batched axis-0 epoch via ``gr.open`` (dataframes) or ``load_slice`` (arrays)."""
    if saved_root_kind(path) == "dataframe":
        epoch_open_windows(
            path,
            transform,
            consecutive_batch_windows(n_molecules, batch_size),
        )
        return

    epoch_load_slice_batched(
        path,
        transform,
        n_molecules=n_molecules,
        batch_size=batch_size,
    )


def epoch_open_load_only(
    path: str,
    *,
    n_molecules: int,
    batch_size: int,
) -> None:
    epoch_open_batched(path, None, n_molecules=n_molecules, batch_size=batch_size)


def epoch_load_slice_batched(
    path: str,
    transform: Callable[[T], T] | None,
    *,
    n_molecules: int,
    batch_size: int,
) -> None:
    for start in range(0, n_molecules, batch_size):
        stop = min(start + batch_size, n_molecules)
        batch = gr._core.load_slice(path, start, stop)
        if transform is not None:
            transform(batch)
