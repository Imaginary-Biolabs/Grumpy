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
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    a = rng.integers(-1_000_000, 1_000_000, size=(args.n,), dtype=np.int32)
    v = rng.integers(-1_000_000, 1_000_000, size=(args.n // 10,), dtype=np.int32)
    ga = gr.array(a.tolist(), dtype=gr.int32)
    gv = gr.array(v.tolist(), dtype=gr.int32)

    # Ensure search_sorted input is sorted
    a_sorted = np.sort(a)
    ga_sorted = gr.array(a_sorted.tolist(), dtype=gr.int32)

    cases = []
    cases.append(bench("sort_numpy", lambda: np.sort(a), args.warmup, args.repeats))
    cases.append(bench("sort_grumpy", lambda: ga.sort(), args.warmup, args.repeats))
    cases.append(bench("argsort_numpy", lambda: np.argsort(a), args.warmup, args.repeats))
    cases.append(bench("argsort_grumpy", lambda: ga.argsort(), args.warmup, args.repeats))
    cases.append(bench("searchsorted_numpy", lambda: np.searchsorted(a_sorted, v, side="left"), args.warmup, args.repeats))
    cases.append(bench("searchsorted_grumpy", lambda: gr.search_sorted(ga_sorted, gv, right=False), args.warmup, args.repeats))
    cases.append(bench("partition_numpy", lambda: np.partition(a, args.n // 2), args.warmup, args.repeats))
    cases.append(bench("partition_grumpy", lambda: ga.partition(args.n // 2), args.warmup, args.repeats))

    # 2D dim=1 (rectangular) quick check
    nrows = 512
    ncols = max(64, args.n // nrows)
    a2 = rng.integers(-1000, 1000, size=(nrows, ncols), dtype=np.int32)
    ga2 = gr.array(a2.tolist(), dtype=gr.int32)
    k2 = ncols // 2
    cases.append(bench("sort_dim1_numpy", lambda: np.sort(a2, axis=1), args.warmup, args.repeats))
    cases.append(bench("sort_dim1_grumpy", lambda: ga2.sort(dim=1), args.warmup, args.repeats))
    cases.append(bench("partition_dim1_numpy", lambda: np.partition(a2, k2, axis=1), args.warmup, args.repeats))
    cases.append(bench("partition_dim1_grumpy", lambda: ga2.partition(k2, dim=1), args.warmup, args.repeats))

    print("## Grumpy vs NumPy sort/search benchmark\n")
    print(f"- n: {args.n}")
    print(f"- warmup: {args.warmup}, repeats: {args.repeats}\n")
    print("| op | time |")
    print("|---|---:|")
    for name, t in cases:
        print(f"| {name} | {t*1e3:.3f} ms |")
    d = dict(cases)
    print("\n### Ratios (Grumpy / NumPy)")
    print(f"- sort: {d['sort_grumpy']/d['sort_numpy']:.2f}×")
    print(f"- argsort: {d['argsort_grumpy']/d['argsort_numpy']:.2f}×")
    print(f"- searchsorted: {d['searchsorted_grumpy']/d['searchsorted_numpy']:.2f}×")
    print(f"- partition: {d['partition_grumpy']/d['partition_numpy']:.2f}×")
    print(f"- sort(dim=1): {d['sort_dim1_grumpy']/d['sort_dim1_numpy']:.2f}×")
    print(f"- partition(dim=1): {d['partition_dim1_grumpy']/d['partition_dim1_numpy']:.2f}×")


if __name__ == '__main__':
    main()


