# Example inputs

Small placeholder inputs for following the README workflow. For a real analysis,
replace them with your own files in the same layout.

| Path | Contents |
|---|---|
| `lineage1/`, `lineage2/` | Two synthetic lineages (5 stages x 50 cells) with RNA, ATAC, cell labels, tree, TF/TG lists and TF-RE/RE-TG priors. RNA and ATAC share cells, so `--mode paired` and `--mode unpaired` both work. |
| `ligand_receptor.csv` | Ligand-receptor pairs among the example TGs, for the CCC step. |
| [`motif_example/`](motif_example/README.md) | A small fly input without priors, for building priors from motifs. |

Gene names, coordinates and ligand-receptor pairs in `lineage1/`, `lineage2/` and
`ligand_receptor.csv` are placeholders taken from the bundled simulated data
(`dev/make_example_data.py`).
