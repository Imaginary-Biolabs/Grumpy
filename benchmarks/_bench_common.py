"""Shared helpers for Grumpy vs NumPy vs Awkward benchmarks."""

from __future__ import annotations

import time
from typing import Callable, Optional


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


def print_header(title: str, *, python: str, numpy: str, platform: str, awkward: Optional[str] = None) -> None:
    print(f"## {title}")
    print()
    print(f"- python: {python}")
    print(f"- numpy: {numpy}")
    if awkward:
        print(f"- awkward: {awkward}")
    print(f"- platform: {platform}")
    print()
