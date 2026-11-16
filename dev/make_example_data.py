"""Write the synthetic example inputs under data/ from the bundled simulation.

The files are subsets of the simulated RNA/ATAC observations and candidate priors in
src/draft_paired/examples/simulation_9plus3 (seed 101 for lineage1, 102 for lineage2). RNA and
ATAC share cells, so each folder can be preprocessed with --mode paired or --mode unpaired.
Gene names, stage names, genomic coordinates and ligand-receptor pairs are placeholders.

usage: python dev/make_example_data.py [--output data] [--cells-per-stage 50]
"""

from __future__ import annotations

import argparse
import io
from importlib import resources
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def write_lineage(folder: Path, lineage: str, seed: int, cells_per_stage: int) -> None:
    content = (
        resources.files("draft_paired")
        .joinpath(f"examples/simulation_9plus3/inputs_{seed}.npz")
        .read_bytes()
    )
    with np.load(io.BytesIO(content), allow_pickle=False) as source:
        rna = np.concatenate([x[:, :cells_per_stage].T for x in source["rna_counts"]])
        atac = np.concatenate([x[:, :cells_per_stage].T for x in source["atac_counts"]])
        tf_re, re_tg, candidate = source["tf_re"], source["re_tg"], source["candidate"]
    n_tg, n_tf = candidate.shape
    tfs = [f"TF{i + 1}" for i in range(n_tf)]
    tgs = [f"TG{i + 1}" for i in range(n_tg)]
    n_stage = len(rna) // cells_per_stage
    stages = [f"stage{t + 1}" for t in range(n_stage)]
    cells = [f"{lineage}_{s}_cell{i + 1}" for s in stages for i in range(cells_per_stage)]
    labels = pd.DataFrame({"cell_id": cells, "cell_type": np.repeat(stages, cells_per_stage)})

    # Each region sits just downstream of the TSS of the gene it is linked to.
    target = np.argmax(re_tg, axis=0)
    offset = pd.Series(target).groupby(target).cumcount().to_numpy()
    regions = [
        f"chrExample:{g * 10000 + 1100 + k * 100}-{g * 10000 + 1150 + k * 100}"
        for g, k in zip(target, offset)
    ]
    assert len(set(regions)) == len(regions)

    folder.mkdir(parents=True, exist_ok=True)
    obs = pd.DataFrame(index=cells)
    rna_data = ad.AnnData(sparse.csr_matrix(rna.astype(np.float32)), obs=obs, var=pd.DataFrame(index=tfs + tgs))
    rna_data.layers["lognorm"] = sparse.csr_matrix(
        np.log1p(rna * 10000 / np.maximum(rna.sum(1, keepdims=True), 1)).astype(np.float32)
    )
    rna_data.write_h5ad(folder / "rna.h5ad", compression="gzip")
    atac_data = ad.AnnData(sparse.csr_matrix(atac.astype(np.float32)), obs=obs.copy(), var=pd.DataFrame(index=regions))
    atac_data.layers["counts"] = atac_data.X.copy()
    atac_data.write_h5ad(folder / "atac.h5ad", compression="gzip")

    labels.to_csv(folder / "rna_cells.tsv", sep="\t", index=False)
    labels.to_csv(folder / "atac_cells.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"node_id": stages, "parent_id": [""] + stages[:-1], "stage_id": range(n_stage)}
    ).to_csv(folder / "tree.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_id": tfs, "gene_name": tfs}).to_csv(folder / "TF.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "gene_id": tgs,
            "gene_name": tgs,
            "chr": "chrExample",
            "start": np.arange(n_tg) * 10000 + 1000,
            "end": np.arange(n_tg) * 10000 + 2000,
            "strand": "+",
        }
    ).to_csv(folder / "TG.tsv", sep="\t", index=False)
    re_idx, tf_idx = np.nonzero(tf_re)
    pd.DataFrame({"TF": np.asarray(tfs)[tf_idx], "RE": np.asarray(regions)[re_idx]}).to_csv(
        folder / "TF_RE.tsv", sep="\t", index=False
    )
    tg_idx, re_idx = np.nonzero(re_tg)
    pd.DataFrame({"RE": np.asarray(regions)[re_idx], "TG": np.asarray(tgs)[tg_idx]}).to_csv(
        folder / "RE_TG.tsv", sep="\t", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--cells-per-stage", type=int, default=50)
    args = parser.parse_args()
    write_lineage(args.output / "lineage1", "lineage1", 101, args.cells_per_stage)
    write_lineage(args.output / "lineage2", "lineage2", 102, args.cells_per_stage)
    pd.DataFrame({"ligand": ["TG1", "TG3", "TG5"], "receptor": ["TG2", "TG4", "TG6"]}).to_csv(
        args.output / "ligand_receptor.csv", index=False
    )
    print(f"Example inputs written to {args.output}/lineage1, {args.output}/lineage2 and ligand_receptor.csv")


if __name__ == "__main__":
    main()
