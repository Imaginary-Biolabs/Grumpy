"""Cover Rust scheduling success path in stream.apply."""

import grumpy as gr


def test_stream_rust_scheduling_compiled_mul(tmp_path):
    x = gr.array([[1.0, 2.0], [3.0, 4.0]], dtype=gr.float64)
    p = tmp_path / "a.gr"
    gr.save(x, str(p), chunk_size=2)
    st = gr.stream(str(p), batch_size=1)

    def mul_only(batch):
        batch = batch * 2.0
        return batch

    out = list(st.apply(mul_only, cpu=2, compile=True, scheduler="rust", prefetch=0))
    assert [b.to_list() for b in out] == [[[2.0, 4.0]], [[6.0, 8.0]]]
