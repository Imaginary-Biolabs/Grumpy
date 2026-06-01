"""batch_on streaming for union-root layouts."""

from __future__ import annotations

import grumpy as gr


def _union_molecule_df():
    return gr.dataframe(
        {
            "molecule_id": [1, [2, 3], 4, [5]],
            "molecule_name": ["a", "b", "c", "d"],
        },
        schema=["molecule"],
    )


def _union_molecule_residue_df():
    return gr.dataframe(
        {
            "molecule_id": [1, [2, 3], 4],
            "residue_id": [10, [20, 30], 40],
            "residue_name": ["A", ["B", "C"], "D"],
        },
        schema=["molecule", "residue"],
    )


def test_batch_on_union_molecule(tmp_path):
    path = str(tmp_path / "union_mol.gr")
    gr.save(_union_molecule_df(), path, chunk_size=2)
    st = gr.stream(path, batch_size=2, batch_on="molecule")
    assert len(st) == 2
    batches = [len(b.to_dict()["molecule_id"]) for b in st]
    assert batches == [2, 2]


def test_batch_on_union_residue_counts(tmp_path):
    path = str(tmp_path / "union_res.gr")
    gr.save(_union_molecule_residue_df(), path)
    # residue counts per molecule row: [1, 2, 1] -> batch_size=2 packs rows 0+1, then row 2
    st = gr.stream(path, batch_size=2, batch_on="residue", drop_last=False)
    assert len(st) == 2
    assert [len(b.to_dict()["molecule_id"]) for b in st] == [2, 1]


def test_batch_on_union_parity_with_load_slice(tmp_path):
    path = str(tmp_path / "union_parity.gr")
    df = _union_molecule_residue_df()
    gr.save(df, path, chunk_size=2)
    full = gr.load(path)
    seen = 0
    for batch in gr.stream(path, batch_size=2, batch_on="molecule"):
        n = len(batch.to_dict()["molecule_id"])
        expected = gr._core.load_slice(path, seen, seen + n)
        assert batch.to_dict() == expected.to_dict()
        seen += n
    assert seen == len(full)


def test_batch_on_union_array_numeric_depth(tmp_path):
    x = gr.array([1, [2, 3], 4, [5, 6]], dtype=gr.int64)
    path = str(tmp_path / "union_arr.gr")
    gr.save(x, path, chunk_size=2)
    st = gr.stream(path, batch_size=2, batch_on="0")
    assert len(st) == 2
    assert [b.to_list() for b in st] == [[1, [2, 3]], [4, [5, 6]]]
