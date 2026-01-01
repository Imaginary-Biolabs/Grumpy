import grumpy as gr


def test_save_load_array_roundtrip(tmp_path):
    x = gr.array([[1, None, 3], [4, 5]], dtype=gr.int32)
    p = tmp_path / "arr.gr"
    gr.save(x, str(p), chunk_size=4)
    y = gr.load(str(p))
    assert y.to_list() == x.to_list()
    assert y.dtype.name == x.dtype.name


def test_save_load_array_strings_roundtrip(tmp_path):
    x = gr.array([["one", None], ["two", "three"]], dtype=gr.string)
    p = tmp_path / "s.gr"
    gr.save(x, str(p))
    y = gr.load(str(p))
    assert y.to_list() == x.to_list()
    assert y.dtype.name == "string"


def test_save_load_dataframe_roundtrip(tmp_path):
    df = gr.dataframe(
        {
            "molecule_id": ["one", "two"],
            "residue_name": [["A", "B", "C"], ["D", "E"]],
            "atom_number": [[[1, 2], [3, 4, 5], [6]], [[7, 8], [9]]],
        },
        schema=["molecule", ("residue", "group"), "atom"],
    )
    df.residue.residue_weight = [0.5, 0.7, 0.8, 0.9, 1.0]

    p = tmp_path / "df.gr"
    gr.save(df, str(p), chunk_size=8)
    df2 = gr.load(str(p))

    assert df2.to_dict() == df.to_dict()
    assert df2["residue_weight"].to_dict()["residue_weight"] == [[0.5, 0.7, 0.8], [0.9, 1.0]]


