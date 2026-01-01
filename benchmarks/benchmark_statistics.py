import argparse
import time

import numpy as np

import grumpy as gr


def bench(label, fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    best = 1e9
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return label, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=4096)
    ap.add_argument("--ncols", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    a = rng.normal(size=(args.nrows, args.ncols)).astype(np.float64)
    x = gr.array(a.tolist(), dtype=gr.float64)

    cases = []
    cases.append(bench("std_numpy_dim1", lambda: a.std(axis=1, ddof=0), args.warmup, args.repeats))
    cases.append(bench("std_grumpy_dim1", lambda: x.std(dim=1, ddof=0), args.warmup, args.repeats))
    cases.append(bench("var_numpy_dim1", lambda: a.var(axis=1, ddof=0), args.warmup, args.repeats))
    cases.append(bench("var_grumpy_dim1", lambda: x.var(dim=1, ddof=0), args.warmup, args.repeats))
    cases.append(bench("quantile_numpy_dim1", lambda: np.quantile(a, 0.5, axis=1, method="linear"), args.warmup, args.repeats))
    cases.append(bench("quantile_grumpy_dim1", lambda: x.quantile(0.5, dim=1), args.warmup, args.repeats))

    print("## Grumpy vs NumPy statistics benchmark\n")
    print(f"- nrows: {args.nrows}, ncols: {args.ncols}")
    print(f"- warmup: {args.warmup}, repeats: {args.repeats}\n")
    print("| op | time |")
    print("|---|---:|")
    for name, t in cases:
        print(f"| {name} | {t*1e3:.3f} ms |")
    d = dict(cases)
    print("\n### Ratios (Grumpy / NumPy)")
    print(f"- std_dim1: {d['std_grumpy_dim1']/d['std_numpy_dim1']:.2f}×")
    print(f"- var_dim1: {d['var_grumpy_dim1']/d['var_numpy_dim1']:.2f}×")
    print(f"- quantile_dim1: {d['quantile_grumpy_dim1']/d['quantile_numpy_dim1']:.2f}×")


if __name__ == '__main__':
    main()


