import tempfile

import pytest

import grumpy as gr


def test_compile_parity_array_pipeline():
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
    plain = t3(t2(t1(x)))

    ct1 = gr.compile(t1)
    ct2 = gr.compile(t2)
    ct3 = gr.compile(t3)
    compiled = ct3(ct2(ct1(x)))
    assert compiled.to_list() == plain.to_list()


def test_compile_parity_dataframe_assignment():
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
    out_compiled = gr.compile(t)(df)
    assert out_compiled["residue_center"].to_dict() == out_plain["residue_center"].to_dict()


def test_compile_warns_on_uncompilable_transform():
    def bad(batch):
        if True:  # control-flow unsupported by compiler
            return batch
        return batch

    x = gr.array([1, 2, 3], dtype=gr.int32)
    with pytest.warns(UserWarning, match="could not be compiled"):
        _ = gr.compile(bad)(x)
