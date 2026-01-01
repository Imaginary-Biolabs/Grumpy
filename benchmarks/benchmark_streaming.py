import time
from time import perf_counter

import grumpy as gr


def sleepy(batch):
    # releases GIL -> thread pool overlaps
    time.sleep(0.01)
    return batch


def knn_transform(batch):
    # Rust-heavy transform that should release the GIL (neighbors uses py.allow_threads).
    _ = gr.neighbors(batch, batch, k=8, dim=0, loop=False)
    return batch


def knn_compiled(batch):
    # Compilable: batch = gr.neighbors(batch,batch,...)
    batch = gr.neighbors(batch, batch, k=8, dim=0, loop=False)
    return batch


def many_ops(batch):
    batch = batch * 1.0001
    return batch


def many_ops_add(batch):
    batch = batch + 0.25
    return batch


def many_ops_reduce(batch):
    batch = batch.mean(dim=1)
    return batch


def main():
    # Stream a 2D point cloud so we can run kNN per batch.
    x = gr.array([[float(i), float(i % 17), float(i % 31), float(i % 7)] for i in range(4000)], dtype=gr.float64)
    path = ".bench_stream_tmp.gr"
    gr.save(x, path, chunk_size=1024)

    st = gr.stream(path, batch_size=32, drop_last=False)

    for name, fn in [("sleepy", sleepy), ("knn", knn_transform)]:
        t0 = perf_counter()
        for _ in st.apply(fn, cpu=1):
            pass
        t1 = perf_counter() - t0

        t0 = perf_counter()
        for _ in st.apply(fn, cpu=4):
            pass
        t4 = perf_counter() - t0

        print(f"stream.apply {name} cpu=1: {t1:.3f}s")
        print(f"stream.apply {name} cpu=4: {t4:.3f}s")
        print(f"speedup ({name}): {t1 / t4:.2f}x")

    # Compile fusion benchmark (cpu=1 to isolate overhead)
    transforms = [many_ops, many_ops_add] * 50 + [many_ops_reduce]
    t0 = perf_counter()
    for _ in st.apply(transforms, cpu=1, compile=False):
        pass
    plain = perf_counter() - t0

    t0 = perf_counter()
    for _ in st.apply(transforms, cpu=1, compile=True):
        pass
    comp = perf_counter() - t0

    print(f"stream.apply many_ops cpu=1 compile=False: {plain:.3f}s")
    print(f"stream.apply many_ops cpu=1 compile=True:  {comp:.3f}s")
    print(f"ratio (compiled/plain): {comp/plain:.2f}x")

    # Rust scheduling benchmark for a fully compiled pipeline (cpu=4):
    # This exercises: scalar chain (in-place COW) + neighbors + batch-parallel scheduling in Rust.
    transforms2 = [many_ops, many_ops_add] * 25 + [knn_compiled]

    t0 = perf_counter()
    for _ in st.apply(transforms2, cpu=4, compile=True, scheduler="python"):
        pass
    py_sched = perf_counter() - t0

    t0 = perf_counter()
    for _ in st.apply(transforms2, cpu=4, compile=True, scheduler="auto"):
        pass
    rust_sched = perf_counter() - t0

    print(f"stream.apply compiled+neighbors cpu=4 scheduler=python: {py_sched:.3f}s")
    print(f"stream.apply compiled+neighbors cpu=4 scheduler=auto:   {rust_sched:.3f}s")
    print(f"speedup (rust_sched/python_sched): {py_sched / rust_sched:.2f}x")


if __name__ == '__main__':
    main()


