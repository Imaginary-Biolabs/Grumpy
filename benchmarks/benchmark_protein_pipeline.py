import time
from time import perf_counter

import numpy as np

import grumpy as gr


def _protein_like_df(n_proteins: int = 256, n_residues: int = 128, atoms_per_res: int = 8):
    """
    Build a reasonably realistic protein-like dataframe for benchmarking:
    molecule/protein > residue > atom.
    """
    schema = ["molecule", "residue", "atom"]
    molecule_id = [f"M{i}" for i in range(n_proteins)]

    # Residue positions (molecule > residue > coord)
    # Use a stable structured backbone.
    t = np.arange(n_residues, dtype=np.float64)
    backbone = np.stack([0.38 * t, 2.0 * np.sin(t / 3.6), 2.0 * np.cos(t / 3.6)], axis=1)

    residue_pos = []
    atom_pos0 = []
    for i in range(n_proteins):
        residue_pos.append(backbone.tolist())
        # atom_pos0: molecule > residue > atom > coord
        atoms_mol = []
        for r in range(n_residues):
            base = backbone[r]
            atoms_res = []
            for a in range(atoms_per_res):
                atoms_res.append((base + np.array([0.03 * a, 0.01 * (a % 5), 0.02 * (a % 7)])).tolist())
            atoms_mol.append(atoms_res)
        atom_pos0.append(atoms_mol)

    return gr.dataframe(
        {"molecule_id": molecule_id, "residue_pos": residue_pos, "atom_pos0": atom_pos0},
        schema=schema,
    )


def sleepy(batch):
    time.sleep(0.005)
    return batch


def residue_center_transform(batch):
    # Compilable: df_get(atom_pos0) + reduce(mean) + df_set(residue_center)
    batch.residue.residue_center = batch.residue.atom_pos0.mean(dim=1)
    return batch


def knn_residues(batch):
    # Compute per-protein residue kNN graph (discard output, just force execution).
    x = batch.molecule.residue_pos
    out = gr.neighbors(x, x, k=16, dim=1, loop=False)
    _ = out.to_list()[0][0][0][0]  # touch one scalar
    return batch


def main():
    df = _protein_like_df(n_proteins=256, n_residues=128, atoms_per_res=8)
    path = ".bench_protein_tmp.gr"
    gr.save(df, path, chunk_size=64)

    st = gr.stream(path, batch_size=32, drop_last=False)

    # Compare python vs rust scheduling for a fully compilable residue-center assignment.
    t0 = perf_counter()
    for _ in st.apply([residue_center_transform], cpu=4, compile=True, scheduler="python"):
        pass
    py_sched = perf_counter() - t0

    t0 = perf_counter()
    for _ in st.apply([residue_center_transform], cpu=4, compile=True, scheduler="auto"):
        pass
    rust_sched = perf_counter() - t0

    print("## Protein-like pipeline benchmark\n")
    print("- schema: molecule > residue > atom")
    print("- n_proteins=256, residues=128, atoms_per_res=8, batch_size=32, cpu=4\n")
    print(f"compiled residue_center scheduler=python: {py_sched:.3f}s")
    print(f"compiled residue_center scheduler=auto:   {rust_sched:.3f}s")
    print(f"speedup (auto/python): {py_sched / rust_sched:.2f}x\n")

    # Add a heavier op mix (not compilable end-to-end because knn is not expressed as batch rebinding here).
    t0 = perf_counter()
    for _ in st.apply([residue_center_transform, knn_residues], cpu=4, compile="auto", scheduler="auto"):
        pass
    mixed = perf_counter() - t0
    print(f"compiled residue_center + knn_residues (mixed): {mixed:.3f}s")


if __name__ == "__main__":
    main()


