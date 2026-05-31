"""Tests for ML dataloader / streaming features (partial I/O, batch_on, shuffle, DDP)."""

from __future__ import annotations

import pytest

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


def _batch_ranges(st):
    """Collect axis-0 row counts per batch via scene_id column when present."""
    out = []
    for batch in st:
        if hasattr(batch, "to_dict"):
            d = batch.to_dict()
            if "scene_id" in d:
                out.append(len(d["scene_id"]))
            else:
                out.append(len(batch))
        else:
            out.append(len(batch))
    return out


def test_load_slice_parity_array(tmp_path):
    x = gr.array([[i, i + 1] for i in range(20)], dtype=gr.int64)
    path = str(tmp_path / "arr.gr")
    gr.save(x, path, chunk_size=4)
    full = gr.load(path)
    for start, stop in [(0, 4), (4, 11), (15, 20)]:
        partial = gr._core.load_slice(path, start, stop)
        expected = full[start:stop]
        assert partial.to_list() == expected.to_list()


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
    batches = list(gr.stream(path, batch_size=32))
    partial_bytes = gr._core.io_bytes_read()
    assert len(batches) == 16
    assert partial_bytes > 0

    gr._core.reset_io_bytes_read()
    gr._core.load_slice(path, 0, 500)
    full_slice_bytes = gr._core.io_bytes_read()
    assert partial_bytes <= full_slice_bytes * 2


def test_stream_len_axis0(tmp_path):
    x = gr.array(list(range(10)), dtype=gr.int64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path)
    assert len(gr.stream(path, batch_size=4)) == 3
    assert len(gr.stream(path, batch_size=4, drop_last=True)) == 2


def test_stream_batches_match_manual_slices(tmp_path):
    x = gr.array([[float(i)] for i in range(7)], dtype=gr.float64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path, chunk_size=2)
    st = gr.stream(path, batch_size=3, drop_last=False)
    got = [b.to_list() for b in st]
    assert got == [[[0.0], [1.0], [2.0]], [[3.0], [4.0], [5.0]], [[6.0]]]


def test_batch_on_molecule_packs_scenes(tmp_path):
    df = _make_protein_like_df(n_scenes=4, mols_per_scene=2)
    path = str(tmp_path / "prot.gr")
    gr.save(df, path, chunk_size=16)
    # 4 scenes * 2 molecules; batch_size=3 -> [2 scenes, 2 scenes]
    st = gr.stream(path, batch_size=3, batch_on="molecule")
    assert len(st) == 2
    counts = _batch_ranges(st)
    assert counts == [2, 2]


def test_batch_on_entity_counts(tmp_path):
    df = _make_protein_like_df(n_scenes=3, mols_per_scene=4)
    path = str(tmp_path / "prot.gr")
    gr.save(df, path)
    st = gr.stream(path, batch_size=5, batch_on="molecule", drop_last=False)
    # per scene: 4 molecules -> batches: scenes 0+1 (8 mols), scene 2 (4 mols)
    assert len(st) == 2
    assert _batch_ranges(st) == [2, 1]


def test_shuffle_batch_order_reproducible(tmp_path):
    x = gr.array(list(range(40)), dtype=gr.int64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path, chunk_size=8)
    st1 = gr.stream(path, batch_size=10, shuffle=True, seed=42)
    st2 = gr.stream(path, batch_size=10, shuffle=True, seed=42)
    a = [b.to_list() for b in st1]
    b = [b.to_list() for b in st2]
    assert a == b
    assert a != [list(range(i, i + 10)) for i in range(0, 40, 10)]


def test_shuffle_different_seeds_differ(tmp_path):
    x = gr.array(list(range(40)), dtype=gr.int64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path)
    a = [b.to_list() for b in gr.stream(path, batch_size=10, shuffle=True, seed=1)]
    b = [b.to_list() for b in gr.stream(path, batch_size=10, shuffle=True, seed=2)]
    assert a != b


def test_shuffle_requires_seed(tmp_path):
    x = gr.array(list(range(10)), dtype=gr.int64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path)
    with pytest.raises(ValueError, match="seed is required"):
        gr.stream(path, batch_size=4, shuffle=True)


def test_ddp_partition_covers_all_batches(tmp_path):
    x = gr.array(list(range(24)), dtype=gr.int64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path)
    world = 4
    all_batches = []
    for rank in range(world):
        st = gr.stream(path, batch_size=3, world_size=world, rank=rank)
        all_batches.extend(list(st))
    flat = sorted(sum((b.to_list() for b in all_batches), []))
    assert flat == list(range(24))


def test_ddp_len_per_rank(tmp_path):
    x = gr.array(list(range(20)), dtype=gr.int64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path)
    lens = [len(gr.stream(path, batch_size=4, world_size=3, rank=r)) for r in range(3)]
    assert sum(lens) == len(gr.stream(path, batch_size=4))


def test_workers_prefetch_yields_same_batches(tmp_path):
    df = _make_protein_like_df(n_scenes=3, mols_per_scene=2)
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    sync = [b.to_dict() for b in gr.stream(path, batch_size=1, workers=0)]
    prefetch = [b.to_dict() for b in gr.stream(path, batch_size=1, workers=2)]
    assert sync == prefetch


def test_stream_apply_still_works(tmp_path):
    x = gr.array([[1.0, 2.0], [3.0, 4.0]], dtype=gr.float64)
    path = str(tmp_path / "a.gr")
    gr.save(x, path, chunk_size=2)
    st = gr.stream(path, batch_size=1)
    out = list(st.apply(lambda b: b * 2.0, cpu=1))
    assert [b.to_list() for b in out] == [[[2.0, 4.0]], [[6.0, 8.0]]]


def test_batch_on_parity_with_full_load(tmp_path):
    df = _make_protein_like_df(n_scenes=5, mols_per_scene=2)
    path = str(tmp_path / "df.gr")
    gr.save(df, path, chunk_size=8)
    full = gr.load(path)
    st = gr.stream(path, batch_size=3, batch_on="molecule")
    seen = 0
    for batch in st:
        n = len(batch.to_dict()["scene_id"])
        expected = gr._core.load_slice(path, seen, seen + n)
        assert batch.to_dict() == expected.to_dict()
        seen += n
    assert seen == len(full)


def test_drop_last_batch_on(tmp_path):
    df = _make_protein_like_df(n_scenes=3, mols_per_scene=3)
    path = str(tmp_path / "df.gr")
    gr.save(df, path)
    # 9 molecules total, batch_size=5 -> [6,3]; drop_last drops final partial batch
    assert len(gr.stream(path, batch_size=5, batch_on="molecule", drop_last=True)) == 1
    assert len(gr.stream(path, batch_size=5, batch_on="molecule", drop_last=False)) == 2
