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
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    a = rng.integers(0, 1_000_000, size=(args.n,), dtype=np.int32)
    t = rng.integers(0, 1_000_000, size=(args.n // 10,), dtype=np.int32)
    ga = gr.array(a.tolist(), dtype=gr.int32)
    gt = gr.array(t.tolist(), dtype=gr.int32)

    cases = []
    cases.append(bench("unique_numpy", lambda: np.unique(a), args.warmup, args.repeats))
    cases.append(bench("unique_grumpy", lambda: gr.unique(ga), args.warmup, args.repeats))
    cases.append(bench("isin_numpy", lambda: np.isin(a, t), args.warmup, args.repeats))
    cases.append(bench("isin_grumpy", lambda: gr.isin(ga, gt), args.warmup, args.repeats))

    print("## Grumpy vs NumPy set routines benchmark\n")
    print(f"- n: {args.n}")
    print(f"- warmup: {args.warmup}, repeats: {args.repeats}\n")
    print("| op | time |")
    print("|---|---:|")
    for name, tsec in cases:
        print(f"| {name} | {tsec*1e3:.3f} ms |")
    d = dict(cases)
    print("\n### Ratios (Grumpy / NumPy)")
    print(f"- unique: {d['unique_grumpy']/d['unique_numpy']:.2f}×")
    print(f"- isin: {d['isin_grumpy']/d['isin_numpy']:.2f}×")


if __name__ == '__main__':
    main()


