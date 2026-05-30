"""Shared helpers for Grumpy vs NumPy vs Awkward benchmarks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

import grumpy as gr


def timeit(fn: Callable[[], None], *, warmup: int = 3, repeats: int = 7) -> float:
    """Return best wall-clock seconds over ``repeats`` timed calls."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def try_import_awkward():
    try:
        import awkward as ak  # noqa: F401

        return ak
    except ImportError:
        return None


def print_header(
    title: str,
    *,
    python: str,
    numpy: str,
    platform: str,
    awkward: Optional[str] = None,
    extra: Optional[list[str]] = None,
) -> None:
    print(f"## {title}")
    print()
    print(f"- python: {python}")
    print(f"- numpy: {numpy}")
    if awkward:
        print(f"- awkward: {awkward}")
    print(f"- platform: {platform}")
    if extra:
        for line in extra:
            print(line)
    print()


def fmt_ms(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    return f"{seconds * 1e3:.3f} ms"


def row_length(ncols: int, row_index: int) -> int:
    """Slightly ragged rows: even rows ncols-1, odd rows ncols+1."""
    delta = -1 if (row_index % 2 == 0) else 1
    return ncols + delta


@dataclass(frozen=True)
class RaggedLists:
    """Nested Python lists + NumPy views; arrays are built separately for fair timing."""

    nrows: int
    ncols: int
    n_elements: int
    ragged_a: list
    ragged_b: list
    flat_a: np.ndarray
    flat_b: np.ndarray
    np_rect_a: np.ndarray
    np_rect_b: np.ndarray


@dataclass(frozen=True)
class RaggedDataset(RaggedLists):
    gr_a: object
    gr_b: object
    ak_a: object
    ak_b: object


def make_slightly_ragged_lists(
    rng: np.random.Generator,
    nrows: int,
    ncols: int,
    *,
    low: int = 0,
    high: int = 1_000_000,
) -> RaggedLists:
    """
    Build paired slightly ragged nested lists (row length ``ncols±1``).

    Total leaf count is ``nrows * ncols``. NumPy rectangular views use
    ``flat.reshape(nrows, ncols)`` so kernel benchmarks can match element counts.
    """
    ragged_a: list = []
    ragged_b: list = []
    flat_a_list: list[int] = []
    flat_b_list: list[int] = []

    for i in range(nrows):
        m = row_length(ncols, i)
        ra = rng.integers(low, high, size=m, dtype=np.int32).tolist()
        rb = rng.integers(max(1, low), high, size=m, dtype=np.int32).tolist()
        ragged_a.append(ra)
        ragged_b.append(rb)
        flat_a_list.extend(ra)
        flat_b_list.extend(rb)

    flat_a = np.asarray(flat_a_list, dtype=np.int32)
    flat_b = np.asarray(flat_b_list, dtype=np.int32)
    n_elements = nrows * ncols
    if flat_a.size != n_elements:
        raise RuntimeError(f"expected {n_elements} elements, got {flat_a.size}")

    return RaggedLists(
        nrows=nrows,
        ncols=ncols,
        n_elements=n_elements,
        ragged_a=ragged_a,
        ragged_b=ragged_b,
        flat_a=flat_a,
        flat_b=flat_b,
        np_rect_a=flat_a.reshape(nrows, ncols),
        np_rect_b=flat_b.reshape(nrows, ncols),
    )


def materialize_ragged_dataset(lists: RaggedLists, *, ak) -> RaggedDataset:
    gr_a = gr.array(lists.ragged_a, dtype=gr.int32)
    gr_b = gr.array(lists.ragged_b, dtype=gr.int32)
    ak_a = ak.Array(lists.ragged_a)
    ak_b = ak.Array(lists.ragged_b)
    return RaggedDataset(
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


def make_slightly_ragged_int32(
    rng: np.random.Generator,
    nrows: int,
    ncols: int,
    *,
    ak,
    low: int = 0,
    high: int = 1_000_000,
) -> RaggedDataset:
    """Convenience wrapper: build lists and materialize Grumpy/Awkward arrays."""
    lists = make_slightly_ragged_lists(rng, nrows, ncols, low=low, high=high)
    return materialize_ragged_dataset(lists, ak=ak)


def make_valid_index_pairs(
    rng: np.random.Generator,
    nrows: int,
    ncols: int,
    nfancy: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Row/col pairs valid for every ragged row (col < row_length)."""
    rows = rng.integers(0, nrows, size=nfancy, dtype=np.int64)
    # Even rows have length ncols-1 → max col is ncols-2.
    cols = rng.integers(0, max(1, ncols - 1), size=nfancy, dtype=np.int64)
    return rows, cols


def print_ratio_table(title: str, rows: list[tuple[str, float, float, float]]) -> None:
    """Print numpy / grumpy / awkward times and ratios."""
    print(f"### {title}")
    print()
    print("| op | numpy | grumpy | awkward | gr/np | ak/np | gr/ak |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, np_t, gr_t, ak_t in rows:
        gr_np = gr_t / np_t if np_t > 0 else float("inf")
        ak_np = ak_t / np_t if np_t > 0 else float("inf")
        gr_ak = gr_t / ak_t if ak_t > 0 else float("inf")
        print(
            f"| {name} | {fmt_ms(np_t)} | {fmt_ms(gr_t)} | {fmt_ms(ak_t)} | "
            f"{gr_np:.2f}× | {ak_np:.2f}× | {gr_ak:.2f}× |"
        )
    print()
