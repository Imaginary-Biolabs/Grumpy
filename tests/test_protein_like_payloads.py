import tempfile

import grumpy as gr


def _make_protein_like_df(
    n_scenes: int = 2,
    mols_per_scene: int = 2,
    chains_per_mol: int = 2,
    residues_per_chain: int = 12,
    frames: int = 8,
):
    """
    Build a small but structurally realistic protein dataset:
    scene > molecule > chain > residue > atom > frame.

    We keep sizes small so tests remain fast.
    """
    # Atom counts per residue: realistic-ish 1..30 but small here.
    def n_atoms_for_res(res_idx: int) -> int:
        return 1 + (res_idx * 7) % 12  # 1..12

    schema = ["scene", "molecule", "chain", "residue", "atom", "frame"]

    scene_id = []
    molecule_id = []
    chain_id = []
    residue_name = []
    atom_number = []
    # atom_pos0: (scene,mol,chain,residue,atom,3)  (single frame snapshot)
    atom_pos0 = []

    # residue_pos: (scene,mol,chain,residue,3)  (used for residue kNN)
    residue_pos = []

    for s in range(n_scenes):
        scene_id.append(f"S{s}")
        mol_id_scene = []
        chain_scene = []        # scene -> molecule -> chain_id(str)
        resname_scene = []      # scene -> molecule -> chain -> residue_name(str)
        atomnum_scene = []      # scene -> molecule -> chain -> residue -> atom_number(int)
        atompos_scene = []      # scene -> molecule -> chain -> residue -> atom -> coord(3)
        respos_scene = []       # scene -> molecule -> chain -> residue -> coord(3)

        for m in range(mols_per_scene):
            mol_id_scene.append(f"M{s}-{m}")
            chain_mol = []
            resname_mol = []
            atomnum_mol = []
            atompos_mol = []
            respos_mol = []

            for c in range(chains_per_mol):
                chain_mol.append(f"C{c}")
                resname_chain = []
                atomnum_chain = []
                atompos_chain = []
                respos_chain = []

                for r in range(residues_per_chain):
                    # Fake residue names A..Z
                    resname_chain.append(chr(ord("A") + (r % 26)))
                    na = n_atoms_for_res(r)
                    atomnum_res = list(range(1, na + 1))
                    atompos_res = []

                    # Deterministic "protein-like" geometry: a helix-ish backbone + atom offsets.
                    # Keep numbers small and stable (no RNG needed).
                    base_x = float(r) * 0.38
                    base_y = float((r % 7) - 3) * 0.15
                    base_z = float((r % 11) - 5) * 0.12
                    respos_chain.append([base_x, base_y, base_z])

                    for a in range(na):
                        dx = 0.03 * float(a)
                        dy = 0.01 * float((a * 3) % 5)
                        dz = 0.02 * float((a * 5) % 7)
                        atompos_res.append([base_x + dx, base_y + dy, base_z + dz])

                    atomnum_chain.append(atomnum_res)
                    atompos_chain.append(atompos_res)

                resname_mol.append(resname_chain)
                atomnum_mol.append(atomnum_chain)
                atompos_mol.append(atompos_chain)
                respos_mol.append(respos_chain)

            chain_scene.append(chain_mol)
            resname_scene.append(resname_mol)
            atomnum_scene.append(atomnum_mol)
            atompos_scene.append(atompos_mol)
            respos_scene.append(respos_mol)

        molecule_id.append(mol_id_scene)
        chain_id.append(chain_scene)
        residue_name.append(resname_scene)
        atom_number.append(atomnum_scene)
        atom_pos0.append(atompos_scene)
        residue_pos.append(respos_scene)

    # NOTE: We include a "frame" level in schema even though these columns don't reach it;
    # this mirrors real payloads where some columns do (e.g. atom_pos over time).
    return gr.dataframe(
        {
            "scene_id": scene_id,
            "molecule_id": molecule_id,
            "chain_id": chain_id,
            "residue_name": residue_name,
            "atom_number": atom_number,
            "atom_pos0": atom_pos0,
            "residue_pos": residue_pos,
        },
        schema=schema,
    )


def test_protein_like_schema_dot_access_and_knn():
    df = _make_protein_like_df()

    # Access residue positions grouped by chain (groups=chains, points=residues, d=3)
    x = df.chain.residue_pos
    out = gr.neighbors(x, x, k=3, dim=1, loop=False)

    # Basic structural checks (chain > residue > k > edgepair)
    out0 = out.to_list()[0][0]
    assert len(out0) == 3  # k
    assert len(out0[0]) == 2  # [src, dst]


def test_protein_like_residue_center_transform_and_assignment():
    df = _make_protein_like_df(residues_per_chain=10)

    # Compute residue centers (mean over atoms) for each chain separately.
    # df.chain.atom_pos0 has shape: chain > residue > atom > coord
    residue_center = df.chain.atom_pos0.mean(dim=2)  # reduce atom axis

    # Assign at chain-level is not re-nested today; assign at residue-level by flattening to residues first.
    # df.residue.atom_pos0 has shape: residue > atom > coord
    residue_center2 = df.residue.atom_pos0.mean(dim=1)  # residue > coord

    # Re-nesting for arbitrary schema levels should work now for residue-level columns.
    df.residue.residue_center = residue_center2
    got = df["residue_center"].to_dict()["residue_center"]
    assert isinstance(got, list)


def test_compile_protein_like_residue_center():
    df = _make_protein_like_df(n_scenes=1, mols_per_scene=2, chains_per_mol=1, residues_per_chain=8)

    def transform(batch):
        batch.residue.residue_center = batch.residue.atom_pos0.mean(dim=1)
        return batch

    with tempfile.TemporaryDirectory() as td:
        path = td + "/prot.gr"
        gr.save(df, path, chunk_size=64)
        out = gr.compile(transform)(gr.load(path))
        d0 = out.to_dict()
        assert "residue_center" in d0


