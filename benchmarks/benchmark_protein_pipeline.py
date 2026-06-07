import time
from time import perf_counter

import numpy as np

import grumpy as gr
from grumpy.compiler import compile_pipeline

from _open_epoch import epoch_open_batched


def _protein_like_df(n_proteins: int = 256, n_residues: int = 128, atoms_per_res: int = 8):
    """
    Build a reasonably realistic protein-like dataframe for benchmarking:
    molecule/protein > residue > atom.
    """
    schema = ["molecule", "residue", "atom"]
    molecule_id = [f"M{i}" for i in range(n_proteins)]

    t = np.arange(n_residues, dtype=np.float64)
    backbone = np.stack([0.38 * t, 2.0 * np.sin(t / 3.6), 2.0 * np.cos(t / 3.6)], axis=1)

    residue_pos = []
    atom_pos0 = []
    for i in range(n_proteins):
        residue_pos.append(backbone.tolist())
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
    batch.residue.residue_center = batch.residue.atom_pos0.mean(dim=1)
    return batch


def knn_residues(batch):
    x = batch.molecule.residue_pos
    out = gr.neighbors(x, x, k=16, dim=1, loop=False)
    _ = out.to_list()[0][0][0][0]
    return batch


def _py_pipeline(fns):
    def run(batch):
        for fn in fns:
            batch = fn(batch)
        return batch

    return run


def main():
    n_proteins = 256
    batch_size = 32
    df = _protein_like_df(n_proteins=n_proteins, n_residues=128, atoms_per_res=8)
    path = ".bench_protein_tmp.gr"
    gr.save(df, path, chunk_size=64)

    py_run = _py_pipeline([residue_center_transform])
    compiled_run = compile_pipeline([residue_center_transform])

    t0 = perf_counter()
    epoch_open_batched(path, py_run, n_molecules=n_proteins, batch_size=batch_size)
    py_epoch = perf_counter() - t0

    t0 = perf_counter()
    epoch_open_batched(path, compiled_run, n_molecules=n_proteins, batch_size=batch_size)
    compiled_epoch = perf_counter() - t0

    print("## Protein-like pipeline benchmark\n")
    print("- schema: molecule > residue > atom")
    print("- n_proteins=256, residues=128, atoms_per_res=8, batch_size=32\n")
    print(f"open + Python residue_center: {py_epoch:.3f}s")
    print(f"open + compiled residue_center: {compiled_epoch:.3f}s")
    print(f"speedup (compiled/Python): {py_epoch / compiled_epoch:.2f}x\n")

    mixed_py = _py_pipeline([residue_center_transform, knn_residues])
    t0 = perf_counter()
    epoch_open_batched(path, mixed_py, n_molecules=n_proteins, batch_size=batch_size)
    mixed = perf_counter() - t0
    print(f"open + residue_center + knn_residues (Python): {mixed:.3f}s")


if __name__ == "__main__":
    main()
