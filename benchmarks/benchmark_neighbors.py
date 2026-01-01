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
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    # Protein-like residues: 3D coordinates with mild structure (helix-ish), plus a tiny jitter.
    t = np.arange(args.n, dtype=np.float64)
    pts = np.stack(
        [
            0.38 * t,
            2.0 * np.sin(t / 3.6),
            2.0 * np.cos(t / 3.6),
        ],
        axis=1,
    )
    if args.d > 3:
        extra = rng.normal(scale=0.05, size=(args.n, args.d - 3)).astype(np.float64)
        pts = np.concatenate([pts, extra], axis=1)
    pts = pts + rng.normal(scale=0.01, size=pts.shape)
    pts = pts.astype(np.float64)

    gx = gr.array(pts.tolist(), dtype=gr.float64)

    sklearn = None
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        sklearn = NearestNeighbors(n_neighbors=args.k + 1, algorithm="brute", metric="euclidean")
        sklearn.fit(pts)
    except Exception as e:
        print("sklearn not available; will only benchmark grumpy.")
        print(f"  import error: {e!r}")

    cases = []

    def gr_knn():
        out = gr.neighbors(gx, gx, k=args.k, dim=0, loop=False)
        # Touch the result to avoid it being optimized away.
        _ = out.to_list()[0][0][0]

    cases.append(bench("grumpy_neighbors_knn", gr_knn, args.warmup, args.repeats))

    # Also benchmark a grouped (dim=1) residue kNN within proteins/chains (no sklearn comparator here).
    # Example: 512 proteins, 256 residues each -> groups=512, points=256, d=3.
    n_groups = 512
    n_res = 256
    pts_g = rng.normal(size=(n_groups, n_res, 3)).astype(np.float64)
    gxg = gr.array(pts_g.tolist(), dtype=gr.float64)

    def gr_knn_grouped():
        out = gr.neighbors(gxg, gxg, k=min(args.k, n_res - 1), dim=1, loop=False)
        _ = out.to_list()[0][0][0][0]

    cases.append(bench("grumpy_neighbors_knn_grouped_dim1", gr_knn_grouped, args.warmup, args.repeats))

    if sklearn is not None:

        def sk_knn():
            idx = sklearn.kneighbors(pts, return_distance=False)
            # drop self
            s = 0
            for i in range(idx.shape[0]):
                row = idx[i]
                # avoid allocating: count first k excluding self
                c = 0
                for j in row:
                    if j == i:
                        continue
                    s += int(j)
                    c += 1
                    if c == args.k:
                        break
            return s

        cases.append(bench("sklearn_neighbors_knn", sk_knn, args.warmup, args.repeats))

    print("## Grumpy vs sklearn KNN benchmark\n")
    print(f"- n: {args.n}, d: {args.d}, k: {args.k}")
    print(f"- warmup: {args.warmup}, repeats: {args.repeats}\n")
    print("| op | time |")
    print("|---|---:|")
    for name, t in cases:
        print(f"| {name} | {t*1e3:.3f} ms |")
    d = dict(cases)
    if "sklearn_neighbors_knn" in d:
        print("\n### Ratios (Grumpy / sklearn)")
        print(f"- knn: {d['grumpy_neighbors_knn']/d['sklearn_neighbors_knn']:.2f}×")


if __name__ == "__main__":
    main()


