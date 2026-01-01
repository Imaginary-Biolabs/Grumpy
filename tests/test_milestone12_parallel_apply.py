import time
from time import perf_counter

import grumpy as gr


def _sleepy_identity(batch):
    # sleep releases the GIL, so thread-parallel apply should overlap.
    time.sleep(0.05)
    return batch


def test_stream_yields_batches(tmp_path):
    x = gr.array([[1, 2, 3], [4, 5], [6]])
    p = tmp_path / "arr.gr"
    gr.save(x, str(p), chunk_size=8)

    st = gr.stream(str(p), batch_size=2)
    out = [b.to_list() for b in st]
    assert out == [[[1, 2, 3], [4, 5]], [[6]]]


def test_apply_parallel_overlaps_sleep(tmp_path):
    x = gr.array([[1], [2], [3], [4], [5], [6], [7], [8]])
    p = tmp_path / "arr.gr"
    gr.save(x, str(p), chunk_size=8)

    st = gr.stream(str(p), batch_size=1)

    t0 = perf_counter()
    out1 = [b.to_list() for b in st.apply(_sleepy_identity, cpu=1)]
    t1 = perf_counter() - t0

    t0 = perf_counter()
    out4 = [b.to_list() for b in st.apply(_sleepy_identity, cpu=4)]
    t4 = perf_counter() - t0

    assert out4 == out1
    # serial is ~0.4s; parallel with 4 workers should be materially faster even on loaded CI.
    assert t4 < t1 * 0.75


