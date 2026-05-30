#!/usr/bin/env python3
"""
Run both ragged benchmark suites (public API + internal kernels).

Prefer invoking the individual scripts directly:

- ``benchmark_ragged_api.py`` — docs / charts / user-facing claims
- ``benchmark_ragged_kernels.py`` — engineer micro-kernels only
"""

from __future__ import annotations

import argparse
import sys

from _ragged_bench import add_ragged_args


def _args_to_argv(args: argparse.Namespace, *, include_json: bool) -> list[str]:
    argv = [
        "--nrows",
        str(args.nrows),
        "--ncols",
        str(args.ncols),
        "--nfancy",
        str(args.nfancy),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    if include_json and args.json:
        argv.extend(["--json", args.json])
    return argv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run ragged API + kernel benchmarks.")
    add_ragged_args(ap)
    args = ap.parse_args(argv)

    from benchmark_ragged_api import main as api_main
    from benchmark_ragged_kernels import main as kernel_main

    print("=" * 72)
    print("PRIMARY — public API benchmark (use for docs)")
    print("=" * 72)
    print()
    rc = api_main(_args_to_argv(args, include_json=True))
    if rc != 0:
        return rc

    print()
    print("=" * 72)
    print("SECONDARY — fused kernel benchmark (engineers only)")
    print("=" * 72)
    print()
    return kernel_main(_args_to_argv(args, include_json=False))


if __name__ == "__main__":
    raise SystemExit(main())
