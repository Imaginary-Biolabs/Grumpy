import argparse
import time

import numpy as np

import grumpy as gr


def bench(label, fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return label, min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrows", type=int, default=4096)
    ap.add_argument("--ncols", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    nrows, ncols = args.nrows, args.ncols
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1000, size=(nrows, ncols), dtype=np.int32)
    x = gr.array(a.tolist(), dtype=gr.int32)

    rag = [row.tolist()[: (ncols - 1 if (i & 1) == 0 else ncols + 1)] for i, row in enumerate(a)]
    xr = gr.array(rag, dtype=gr.int32)

    print("## Grumpy vs NumPy reductions benchmark\n")
    print(f"- nrows: {nrows}, ncols: {ncols}")
    print(f"- warmup: {args.warmup}, repeats: {args.repeats}\n")

    tmp = np.empty((nrows,), dtype=np.float64)

    cases = []
    cases.append(
        bench(
            "mean_dim1_numpy",
            lambda: a.mean(axis=1, dtype=np.float64, out=tmp),
            args.warmup,
            args.repeats,
        )
    )
    # kernel-only checksum (sum of row means) to avoid Grumpy output allocation
    cases.append(bench("mean_dim1_grumpy_kernel", lambda: x._mean2d_dim1_i32_sum_f64(), args.warmup, args.repeats))
    cases.append(bench("mean_dim1_grumpy", lambda: x.mean(dim=1), args.warmup, args.repeats))
    cases.append(bench("sum_dim1_numpy", lambda: a.sum(axis=1, dtype=np.int64), args.warmup, args.repeats))
    cases.append(bench("sum_dim1_grumpy_kernel", lambda: x._sum2d_dim1_i32_sum_i64(), args.warmup, args.repeats))
    cases.append(bench("sum_dim1_grumpy", lambda: x.sum(dim=1), args.warmup, args.repeats))

    # Ragged (dim=1)
    cases.append(bench("mean_dim1_ragged_grumpy", lambda: xr.mean(dim=1), args.warmup, args.repeats))
    cases.append(bench("sum_dim1_ragged_grumpy", lambda: xr.sum(dim=1), args.warmup, args.repeats))

    print("| op | time |")
    print("|---|---:|")
    for name, t in cases:
        print(f"| {name} | {t*1e3:.3f} ms |")

    # Ratios (Grumpy / NumPy) for rectangular dim=1
    d = {k: v for k, v in cases}
    print("\n### Ratios (lower is better for Grumpy)")
    print(f"- mean_dim1_grumpy_kernel / mean_dim1_numpy: {d['mean_dim1_grumpy_kernel']/d['mean_dim1_numpy']:.2f}×")
    print(f"- mean_dim1_grumpy / mean_dim1_numpy: {d['mean_dim1_grumpy']/d['mean_dim1_numpy']:.2f}×")
    print(f"- sum_dim1_grumpy_kernel / sum_dim1_numpy: {d['sum_dim1_grumpy_kernel']/d['sum_dim1_numpy']:.2f}×")
    print(f"- sum_dim1_grumpy / sum_dim1_numpy: {d['sum_dim1_grumpy']/d['sum_dim1_numpy']:.2f}×")


if __name__ == '__main__':
    main()


