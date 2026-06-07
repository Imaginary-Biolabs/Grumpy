"""Tests for partial I/O helpers (load_slice) retained after stream removal."""

from __future__ import annotations

import grumpy as gr


def _make_protein_like_df(n_scenes=4, mols_per_scene=3, chains_per_mol=1, residues_per_chain=6):
    schema = ["scene", "molecule", "chain", "residue", "atom", "frame"]
    scene_id = []
    molecule_id = []
    atom_number = []
    for s in range(n_scenes):
        scene_id.append(f"S{s}")
        mol_ids = []
        atom_scene = []
        for m in range(mols_per_scene):
            mol_ids.append(f"M{s}-{m}")
            atom_mol = []
            for c in range(chains_per_mol):
                atom_chain = []
                for r in range(residues_per_chain):
                    atom_chain.append([1 + r, 2 + r])
                atom_mol.append(atom_chain)
            atom_scene.append(atom_mol)
        molecule_id.append(mol_ids)
        atom_number.append(atom_scene)
    return gr.dataframe(
        {"scene_id": scene_id, "molecule_id": molecule_id, "atom_number": atom_number},
        schema=schema,
    )


def test_load_slice_parity_array(tmp_path):
    x = gr.array([[i, i + 1] for i in range(20)], dtype=gr.int64)
    path = str(tmp_path / "arr.gr")
    gr.save(x, path, chunk_size=4)
    full = gr.load(path)
    for start, stop in [(0, 4), (4, 11), (15, 20)]:
        partial = gr._core.load_slice(path, start, stop)
        expected = full[start:stop]
        assert partial.to_list() == expected.to_list()


def test_load_slice_parity_union(tmp_path):
    x = gr.array([1, [2, 3], 4, [5]], dtype=gr.int64)
    path = str(tmp_path / "union.gr")
    gr.save(x, path, chunk_size=2)
    full = gr.load(path)
    partial = gr._core.load_slice(path, 1, 3)
    assert partial.to_list() == full[1:3].to_list()


def test_load_slice_parity_dataframe(tmp_path):
    df = _make_protein_like_df(n_scenes=5, mols_per_scene=2)
    path = str(tmp_path / "df.gr")
    gr.save(df, path, chunk_size=8)
    full = gr.load(path)
    partial = gr._core.load_slice(path, 1, 4)
    assert partial.to_dict() == full[1:4].to_dict()


def test_partial_io_reads_leaf_data(tmp_path):
    x = gr.array(list(range(500)), dtype=gr.int64)
    path = str(tmp_path / "big.gr")
    gr.save(x, path, chunk_size=64)

    gr._core.reset_io_bytes_read()
    partial = gr._core.load_slice(path, 0, 32)
    partial_bytes = gr._core.io_bytes_read()
    assert len(partial) == 32
    assert partial_bytes > 0

    gr._core.reset_io_bytes_read()
    gr._core.clear_path_caches()
    gr._core.load_slice(path, 0, 500)
    full_slice_bytes = gr._core.io_bytes_read()
    assert partial_bytes < full_slice_bytes


def test_stored_len(tmp_path):
    x = gr.array(list(range(10)), dtype=gr.int64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path)
    assert gr._core.stored_len(path) == 10
