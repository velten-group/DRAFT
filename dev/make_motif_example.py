"""Write data/motif_example/: a small Drosophila input without regulatory priors.

It tests motif-based prior preparation (`--genome dm6`). Cells, genes and peaks are a subset
of the manuscript's fly testis multiome data (GEO GSE277132), as processed in the Zenodo
deposit of DRAFT-analysis (data/fly/raw):
- germline stages Germline_0 -> Germline_5 -> Germline_10, the first 40 cells of each
  (sorted by cell ID) measured in both assays;
- all TFs of the fly TF list; as TGs, the 500 non-TF genes with the highest mean squared
  log-normalized expression in these cells;
- ATAC peaks within 20 kb of those TGs.
Only expression/accessibility matrices, cell labels, the tree and gene tables are written.

usage: python dev/make_motif_example.py <fly_raw_dir> [--output data/motif_example]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

STAGES = ["Germline_0", "Germline_5", "Germline_10"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path, help="DRAFT-analysis data/fly/raw")
    parser.add_argument("--output", type=Path, default=Path("data/motif_example"))
    args = parser.parse_args()
    raw, out = args.raw, args.output
    out.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(raw / "cell_annotation.csv")
    ann.columns = ["cell", "time", "bin", "label"] + list(ann.columns[4:])
    ann = ann[ann.label.isin(STAGES)].copy()
    # Annotation IDs "BARCODE-1_sample3" are H5AD IDs "BARCODE-1#S3".
    ann["obs"] = ann.cell.str.replace(r"_sample(\d+)$", lambda m: f"#S{m.group(1)}", regex=True)

    rna = ad.read_h5ad(raw / "A_rna_focus.h5ad", backed="r")
    atac = ad.read_h5ad(raw / "B_atac_focus.h5ad", backed="r")
    ann = ann[ann.obs.isin(set(rna.obs_names) & set(atac.obs_names))]
    cells = pd.concat([ann[ann.label == s].sort_values("obs").head(40) for s in STAGES])

    tf = pd.read_csv(raw / "TF.tsv", sep="\t")
    tg = pd.read_csv(raw / "TG.tsv", sep="\t")
    named = tg.gene_name.notna() & (tg.gene_name.astype(str).str.strip() != "")
    tg = tg[named & ~tg.FBgn_ID.isin(tf.FBgn_ID)]
    tf = tf[tf.FBgn_ID.isin(rna.var_names)]
    tg = tg[tg.FBgn_ID.isin(rna.var_names)]

    rna_sub = rna[cells.obs.to_numpy()].to_memory()
    lognorm = rna_sub.layers["lognorm"]
    signal = pd.Series(np.asarray(lognorm.power(2).mean(0)).ravel(), index=rna_sub.var_names)
    tg = tg.assign(signal=tg.FBgn_ID.map(signal)).sort_values("signal", ascending=False).head(500)
    rna_sub = rna_sub[:, list(tf.FBgn_ID) + list(tg.FBgn_ID)]

    peaks = pd.DataFrame(
        [re.fullmatch(r"([^:]+):(\d+)-(\d+)", name).groups() for name in atac.var_names],
        columns=["chr", "start", "end"],
    )
    peak_start = peaks.start.astype(int).to_numpy()
    keep = np.zeros(len(peaks), dtype=bool)
    for chrom, part in tg.groupby("chr"):
        on_chrom = (peaks.chr == str(chrom)).to_numpy()
        for tss in part.start.to_numpy():
            keep |= on_chrom & (np.abs(peak_start - tss) < 20000)
    atac_sub = atac[cells.obs.to_numpy()].to_memory()[:, keep]

    obs = pd.DataFrame(index=cells.obs.to_numpy())
    rna_out = ad.AnnData(rna_sub.layers["lognorm"].copy(), obs=obs, var=pd.DataFrame(index=rna_sub.var_names))
    rna_out.layers["counts"] = rna_sub.layers["counts"].copy()
    rna_out.layers["lognorm"] = rna_sub.layers["lognorm"].copy()
    rna_out.write_h5ad(out / "rna.h5ad", compression="gzip")
    atac_out = ad.AnnData(atac_sub.layers["counts"].copy(), obs=obs.copy(), var=pd.DataFrame(index=atac_sub.var_names))
    atac_out.layers["counts"] = atac_sub.layers["counts"].copy()
    atac_out.write_h5ad(out / "atac.h5ad", compression="gzip")

    labels = pd.DataFrame({"cell_id": cells.obs.to_numpy(), "cell_type": cells.label.to_numpy()})
    labels.to_csv(out / "rna_cells.tsv", sep="\t", index=False)
    labels.to_csv(out / "atac_cells.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"node_id": STAGES, "parent_id": ["", STAGES[0], STAGES[1]], "stage_id": [0, 1, 2]}
    ).to_csv(out / "tree.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_id": tf.FBgn_ID, "gene_name": tf.gene_name}).to_csv(out / "TF.tsv", sep="\t", index=False)
    pd.DataFrame({"gene_id": tg.FBgn_ID, "gene_name": tg.gene_name}).to_csv(out / "TG.tsv", sep="\t", index=False)
    print(f"{out}: {len(cells)} cells, {len(tf)} TFs, {len(tg)} TGs, {int(keep.sum())} peaks")


if __name__ == "__main__":
    main()
