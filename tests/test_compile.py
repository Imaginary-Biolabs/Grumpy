import tempfile

import pytest

import grumpy as gr


def test_apply_compile_parity_array_pipeline():
    def t1(batch):
        batch = batch * 2
        return batch

    def t2(batch):
        batch = batch + 1
        return batch

    def t3(batch):
        batch = batch.sum(dim=1)
        return batch

    x = gr.array([[1, 2, 3], [4, 5], [6]], dtype=gr.int32)
    with tempfile.TemporaryDirectory() as td:
        path = td + "/x.gr"
        gr.save(x, path, chunk_size=2)
        st = gr.stream(path, batch_size=2, drop_last=False)

        out_plain = []
        for b in st.apply([t1, t2, t3], cpu=1, compile=False):
            out_plain.extend(b.to_list())

        out_comp = []
        for b in st.apply([t1, t2, t3], cpu=1, compile=True):
            out_comp.extend(b.to_list())

        assert out_comp == out_plain


def test_apply_compile_parity_dataframe_assignment():
    df = gr.dataframe(
        {
            "molecule_id": ["a", "b"],
            "residue_name": [["A", "B", "C"], ["D", "E"]],
            "atom_pos": [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [[3.0, 0.0], [4.0, 0.0]]],
        },
        schema=["molecule", "residue"],
    )

    def t(batch):
        batch.residue.residue_center = batch.residue.atom_pos.mean(dim=-1)
        return batch

    out_plain = t(df)

    # Run via stream.apply compile=True to exercise compiled path.
    with tempfile.TemporaryDirectory() as td:
        path = td + "/df.gr"
        gr.save(df, path, chunk_size=1)
        st = gr.stream(path, batch_size=1, drop_last=False)
        batches = list(st.apply([t], cpu=1, compile=True))
        assert len(batches) == 2
        # Merge the two one-row batches to compare by dict; simplest: just compare each row.
        assert batches[0]["residue_center"].to_dict()["residue_center"] == [out_plain["residue_center"].to_dict()["residue_center"][0]]
        assert batches[1]["residue_center"].to_dict()["residue_center"] == [out_plain["residue_center"].to_dict()["residue_center"][1]]


def test_apply_compile_warns_on_uncompilable_transform():
    def bad(batch):
        if True:  # control-flow unsupported by compiler
            return batch
        return batch

    x = gr.array([1, 2, 3], dtype=gr.int32)
    with tempfile.TemporaryDirectory() as td:
        path = td + "/x.gr"
        gr.save(x, path, chunk_size=2)
        st = gr.stream(path, batch_size=2, drop_last=False)
        with pytest.warns(UserWarning, match="could not be compiled"):
            _ = list(st.apply([bad], cpu=1, compile=True))


