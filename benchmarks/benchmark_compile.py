import argparse
import time

import numpy as np

import grumpy as gr
from grumpy.compiler import compile_pipeline


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
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=10)
    args = ap.parse_args()

    pts = np.stack([np.arange(args.n, dtype=np.float64) for _ in range(args.d)], axis=1)
    x = gr.array(pts.tolist(), dtype=gr.float64)

    def py_t1(batch):
        batch = batch * 1.001
        batch = batch + 2.0
        return batch

    def c_t1(batch):
        batch = batch * 1.001
        batch = batch + 2.0
        return batch
    c_t1c = compile_pipeline([c_t1])

    def py_knn(batch):
        _ = gr.neighbors(batch, batch, k=8, dim=0, loop=False)
        return batch

    def c_knn(batch):
        batch = gr.neighbors(batch, batch, k=8, dim=0, loop=False)
        return batch
    c_knnc = compile_pipeline([c_knn])

    def py_reduce(batch):
        return batch.mean(dim=1)

    def c_reduce(batch):
        batch = batch.mean(dim=1)
        return batch
    c_reducec = compile_pipeline([c_reduce])

    cases = []
    cases.append(bench("python_scalar_chain", lambda: py_t1(x), args.warmup, args.repeats))
    cases.append(bench("compiled_scalar_chain", lambda: c_t1c(x), args.warmup, args.repeats))
    cases.append(bench("python_neighbors", lambda: py_knn(x), args.warmup, args.repeats))
    cases.append(bench("compiled_neighbors", lambda: c_knnc(x), args.warmup, args.repeats))
    cases.append(bench("python_reduce_mean_dim1", lambda: py_reduce(x), args.warmup, args.repeats))
    cases.append(bench("compiled_reduce_mean_dim1", lambda: c_reducec(x), args.warmup, args.repeats))

    print("## Compile benchmark\n")
    print(f"- n={args.n}, d={args.d}")
    print(f"- warmup={args.warmup}, repeats={args.repeats}\n")
    print("| case | time |")
    print("|---|---:|")
    for name, t in cases:
        print(f"| {name} | {t*1e3:.3f} ms |")

    d = dict(cases)
    print("\n### Ratios (compiled / python)")
    print(f"- scalar_chain: {d['compiled_scalar_chain']/d['python_scalar_chain']:.2f}×")
    print(f"- neighbors: {d['compiled_neighbors']/d['python_neighbors']:.2f}×")
    print(f"- reduce_mean_dim1: {d['compiled_reduce_mean_dim1']/d['python_reduce_mean_dim1']:.2f}×")


if __name__ == "__main__":
    main()


