#!/usr/bin/env python3
"""Minimal performance smoke test for CI (generous thresholds, catches large regressions)."""

from __future__ import annotations

import sys

import numpy as np

import grumpy as gr

from _bench_common import make_slightly_ragged_lists, timeit


def _assert_under(name: str, seconds: float, limit_s: float) -> None:
    if seconds > limit_s:
        print(f"FAIL {name}: {seconds * 1e3:.1f} ms > limit {limit_s * 1e3:.1f} ms", file=sys.stderr)
        sys.exit(1)
    print(f"ok {name}: {seconds * 1e3:.1f} ms (limit {limit_s * 1e3:.1f} ms)")


def main() -> int:
    rng = np.random.default_rng(0)
    ds = make_slightly_ragged_lists(rng, nrows=512, ncols=64)
    a = gr.array(ds.ragged_a, dtype=gr.int32)
    b = gr.array(ds.ragged_b, dtype=gr.int32)

    _assert_under(
        "(a * b).sum()",
        timeit(lambda: (a * b).sum(), warmup=2, repeats=5),
        limit_s=2.0,
    )
    _assert_under(
        "gr.array construct",
        timeit(lambda: gr.array(ds.ragged_a, dtype=gr.int32), warmup=2, repeats=5),
        limit_s=1.5,
    )
    _assert_under(
        "axis-0 slice",
        timeit(lambda: a[0:256], warmup=2, repeats=5),
        limit_s=0.5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
