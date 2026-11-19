# DRAFT

DRAFT infers dynamic gene regulatory networks from single-cell RNA and ATAC
data along a lineage tree, and compares two lineages by developmental
alignment (DTW) and cell-cell communication (CCC).

> **Status:** DRAFT is being packaged for PyPI. Until the release, install it
> from this repository as shown below. Commands and options may still change.

![DRAFT model overview](docs/model.png)

DRAFT works with paired and unpaired data. The two modes differ only in the
preprocessing step:

| | Paired | Unpaired |
|---|---|---|
| Data | RNA and ATAC measured in the same cells | RNA and ATAC measured in different cells, each labelled with the same tree nodes |
| Preprocessing | `--mode paired` | `--mode unpaired` |
| Graph per tree node | TF, RE (ATAC peak) and TG nodes | TF and TG nodes; candidate TF-TG edges from ATAC and the priors |
| Input files, other commands | Same | Same |

## Installation

Python 3.10 or newer:

```bash
git clone https://github.com/velten-group/DRAFT.git
cd DRAFT
pip install -e ".[full]"
```

`[full]` installs everything the workflow below needs. Optional extras:
`[enrich]` adds GO enrichment of regulatory topics, `[tutorial]` runs the
[interactive example](docs/downstream.ipynb). A GPU speeds up training but is
not required for the examples. Run all commands from this `DRAFT/` directory.

## Quick start

```bash
draft simulation
```

This trains DRAFT on bundled simulated RNA/ATAC data with a known network and
writes `result/simulation/`: `runs.tsv` summarizes network recovery and
temporal smoothness, and each `seed_*/model_*/` folder holds the model,
embeddings, networks and figures. To simulate data from your own prior
network, see [simulation priors](docs/input_formats.md#simulation-priors).

## Tutorial

**Preprocess → train → networks and clustering → regulons → two-lineage
comparison**, run as written on the example inputs in [`data/`](data/README.md):
two small lineages (`data/lineage1/`, `data/lineage2/`) and ligand-receptor
pairs (`data/ligand_receptor.csv`). For your own analysis, replace them with
your files in the same layout ([file formats](docs/input_formats.md)) and use
your own TF and tree-node names.

### 1. Preprocess

One folder per lineage, in the same layout for both modes:

```text
data/lineage1/
  rna.h5ad, atac.h5ad
  rna_cells.tsv, atac_cells.tsv   # tree node of each cell; optional if the H5AD obs has cell_type
  tree.tsv, TF.tsv, TG.tsv
  TF_RE.tsv, RE_TG.tsv            # optional regulatory priors
```

```bash
draft preprocess --input-dir data/lineage1 --mode paired --output prepared/lineage1
draft preprocess --input-dir data/lineage2 --mode paired --output prepared/lineage2
```

- Use `--mode unpaired` for RNA and ATAC from different cells. The example's
  RNA and ATAC share cells, so both modes run on it.
- Without `TF_RE.tsv`/`RE_TG.tsv`, add `--genome hg38`, `mm10` or `dm6`, and
  DRAFT builds [motif-based priors](docs/input_formats.md#when-regulatory-priors-are-unavailable).
- Each output folder receives `graphs.pt` (the model input) and `train.yaml`,
  the config of every later step.

### 2. Train

```bash
draft train --config prepared/lineage1/train.yaml
draft train --config prepared/lineage2/train.yaml
```

Checkpoints go to `result/<lineage>/model/`.

### 3. Export networks and cluster edges

```bash
draft reproduce --config prepared/lineage1/train.yaml
draft network-export --config prepared/lineage1/train.yaml
draft edge-cluster --config prepared/lineage1/train.yaml
draft topic --config prepared/lineage1/train.yaml
```

Results go to `result/lineage1/` (`reproduce/`, `network/`, `edge_cluster/`,
`topic/`). On the small example, `network-export` warns that it kept each
stage's top 1% of TF-TG pairs ([why](docs/downstream.md#draft-network-export)).

### 4. Inspect TF regulons

```bash
draft single-tf --config prepared/lineage1/train.yaml --tf TF1 --stage stage1 --consensus-stages stage1 stage2
```

`TF1` is a TF and `stage1`, `stage2` are tree nodes of the example. Results go
to `result/lineage1/single_tf/`.

### 5. Compare two lineages (DTW + CCC)

```bash
draft ccc --lineage1 prepared/lineage1/train.yaml --lineage2 prepared/lineage2/train.yaml --lr data/ligand_receptor.csv --early1 1 --early2 1 --output result/ccc
```

`--early1`/`--early2` give the number of initial stages that form each
lineage's early window. Ligand and receptor genes must be in both lineages'
`TG.tsv`. The two models may be of either mode.

## Documentation

- [Input guide](docs/input_formats.md): file formats, preprocessing options,
  motif-based priors and the inputs of the other commands.
- [Downstream analyses](docs/downstream.md): outputs and options of each
  command after training.
- [Interactive example](docs/downstream.ipynb): export, clustering and CCC in a
  notebook (`pip install -e ".[full,tutorial]"`).
- `draft --help` and `draft <command> --help` list all options.

## Reproducing the manuscript

Retrains the mouse and Drosophila models; a GPU with at least **48 GB** is
recommended.

1. Install DRAFT as above.
2. Download `mouse_graphs.pt`, `dro_germ_graphs.pt` and `soma_graphs.pt` from
   [Zenodo](https://doi.org/10.5281/zenodo.23678662) into `DRAFT/graph_data/`.
3. Train:

   ```bash
   python scripts/train_reproduction.py --graph-dir graph_data
   ```

   Add `--dataset mouse`, `dro_germ` or `soma` for one model, or `--check-only`
   to validate the files first.

Results go to `result/reproduction/<dataset>/` (`model/`, `train/`,
`config.yaml`).

To reproduce the manuscript's downstream analyses and figures, use
[DRAFT-analysis](https://github.com/velten-group/DRAFT-analysis).

## Citation

If you use DRAFT, please cite:

Zhang Z, Santhi Sekhar P, van Nierop y Sanchez P, Lohmann I, Velten B. DRAFT: graph-recurrent learning of
dynamic gene regulatory networks along single-cell multi-omics trajectories. bioRxiv (2026).
[doi:10.64898/2026.11.17.767145](https://doi.org/10.64898/2026.11.17.767145)

## License and contact

MIT, see [LICENSE](LICENSE). Questions and bug reports:
[GitHub issues](https://github.com/velten-group/DRAFT/issues).
